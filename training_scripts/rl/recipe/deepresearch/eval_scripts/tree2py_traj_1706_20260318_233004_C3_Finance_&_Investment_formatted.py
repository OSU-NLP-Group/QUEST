import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "xrp_spot_etf_lowest_expense_ratio_2026"
TASK_DESCRIPTION = (
    "Identify the U.S.-listed spot XRP exchange-traded fund (ETF) that has the lowest expense ratio among all spot XRP "
    "ETFs currently trading as of March 2026. For the identified ETF, provide the following information: "
    "(1) Confirm it is a spot ETF that holds physical XRP (not futures or derivatives); "
    "(2) Name of the institutional custodian holding the ETF's XRP holdings; "
    "(3) The exact expense ratio (as a percentage); "
    "(4) The ticker symbol; "
    "(5) The fund inception or launch date; "
    "(6) The primary exchange where the ETF is listed. "
    "For each piece of information, provide a reference URL from the fund's official website or a reputable financial "
    "information source that confirms the stated information."
)

AS_OF_DATE_TEXT = "March 2026"
LAUNCH_RANGE_START = "2024-11-01"  # Inclusive lower bound
LAUNCH_RANGE_END = "2025-12-31"    # Inclusive upper bound
VALID_EXCHANGES = [
    "NYSE Arca", "NYSE Arca, Inc.",
    "Nasdaq", "Nasdaq Stock Market LLC", "The Nasdaq Stock Market LLC",
    "Cboe", "Cboe BZX", "Cboe BZX Exchange", "Cboe BZX Exchange, Inc."
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class URLSet(BaseModel):
    physical_holdings: List[str] = Field(default_factory=list)
    expense_ratio: List[str] = Field(default_factory=list)
    lowest_expense_ratio_comparison: List[str] = Field(default_factory=list)
    ticker: List[str] = Field(default_factory=list)
    custodian: List[str] = Field(default_factory=list)
    launch_date: List[str] = Field(default_factory=list)
    exchange: List[str] = Field(default_factory=list)
    benchmark: List[str] = Field(default_factory=list)
    general: List[str] = Field(default_factory=list)


class CompetitorETF(BaseModel):
    name: Optional[str] = None
    ticker: Optional[str] = None
    expense_ratio: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class XRPEtfExtraction(BaseModel):
    etf_name: Optional[str] = None
    ticker: Optional[str] = None
    expense_ratio: Optional[str] = None  # keep as string to allow formats like "0.19%"
    custodian: Optional[str] = None
    launch_date: Optional[str] = None    # keep string, e.g., "2025-01-15" or "Jan 15, 2025"
    exchange: Optional[str] = None
    benchmark_index: Optional[str] = None
    is_spot_physical: Optional[bool] = None

    urls: URLSet = Field(default_factory=URLSet)
    competitors: List[CompetitorETF] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_xrp_etf_info() -> str:
    return """
    Extract details for the single U.S.-listed spot XRP ETF that the answer claims has the lowest expense ratio as of March 2026.
    Return a JSON object matching the XRPEtfExtraction schema:

    Required top-level fields:
    - etf_name: Full ETF name (e.g., "<Issuer> <XRP> Trust/ETF")
    - ticker: ETF ticker symbol (string)
    - expense_ratio: The management fee/expense ratio as a percentage string (e.g., "0.19%")
    - custodian: Institutional custodian name that safekeeps the ETF's XRP (e.g., "Coinbase Custody Trust Company, LLC")
    - launch_date: The fund inception, first listing, or launch date (string; any reasonable date format accepted)
    - exchange: The primary listing exchange (e.g., "NYSE Arca", "Nasdaq", "Cboe BZX")
    - benchmark_index: The benchmark index or reference price the ETF tracks (e.g., "CF Benchmarks XRP-USD Settlement Price")
    - is_spot_physical: true if the fund holds XRP tokens directly (physically-backed/spot), false otherwise

    For each of the above items, also provide supporting URLs in the nested 'urls' object:
    - urls.physical_holdings: One or more URLs confirming it’s a spot/physically-backed XRP ETF
    - urls.expense_ratio: One or more URLs confirming the exact expense ratio
    - urls.lowest_expense_ratio_comparison: One or more URLs (if available) explicitly stating it has the lowest expense ratio among U.S.-listed spot XRP ETFs
    - urls.ticker: One or more URLs confirming the ticker
    - urls.custodian: One or more URLs confirming the custodian
    - urls.launch_date: One or more URLs confirming the launch/inception/listing date
    - urls.exchange: One or more URLs confirming the primary exchange
    - urls.benchmark: One or more URLs confirming the benchmark index
    - urls.general: One or more authoritative URLs related to the fund (e.g., issuer homepage, prospectus, fact sheet)

    Additionally, if the answer mentions competitor U.S.-listed spot XRP ETFs, include them under 'competitors' with:
    - name
    - ticker
    - expense_ratio
    - source_urls: URLs confirming each competitor's expense ratio

    Important extraction rules:
    - Only extract URLs that are explicitly present in the answer text (including markdown links).
    - Do not fabricate information. If an item is missing, set its value to null or an empty list as appropriate.
    - Keep strings exactly as presented (do not normalize percentages or dates).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_text(v: Optional[str]) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls or [])


def _merge_urls(*url_lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for ul in url_lists:
        if not ul:
            continue
        for u in ul:
            su = (u or "").strip()
            if su and su not in seen:
                merged.append(su)
                seen.add(su)
    return merged


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_primary_identification_subtree(evaluator: Evaluator, parent_node, data: XRPEtfExtraction) -> None:
    """
    Build and verify the 'ETF_Primary_Identification' subtree:
    - Verify spot/physical nature with evidence
    - Verify lowest expense ratio with evidence
    """
    etf_primary = evaluator.add_sequential(
        id="ETF_Primary_Identification",
        desc="Identify a specific spot XRP ETF and verify its core characteristics",
        parent=parent_node,
        critical=True,
    )

    # Minimal presence of the identified ETF fields (name + ticker)
    evaluator.add_custom_node(
        result=_has_text(data.etf_name) and _has_text(data.ticker),
        id="ETF_Identified_Presence",
        desc="ETF name and ticker are provided in the answer",
        parent=etf_primary,
        critical=True,
    )

    # 1) Spot / physically-backed verification
    spot_seq = evaluator.add_sequential(
        id="Spot_ETF_Type_Verification",
        desc="Verify the identified ETF is a spot ETF that holds physical XRP (not futures, derivatives, or leveraged products)",
        parent=etf_primary,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.is_spot_physical) and _has_urls(data.urls.physical_holdings),
        id="Spot_Physical_Presence",
        desc="Spot/physical status is provided with at least one supporting URL",
        parent=spot_seq,
        critical=True,
    )

    physical_confirm = evaluator.add_sequential(
        id="Physical_Holdings_Confirmation",
        desc="Confirm the ETF holds physical XRP tokens and provide evidence",
        parent=spot_seq,
        critical=True,
    )

    physical_leaf = evaluator.add_leaf(
        id="Physical_Holdings_Reference_URL",
        desc="Provide reference URL confirming physical XRP holdings",
        parent=physical_confirm,
        critical=True,
    )
    claim_phys = (
        f"The ETF '{data.etf_name}' with ticker '{data.ticker}' is a spot ETF that holds XRP tokens directly "
        f"(physically backed), not a futures-based or derivatives-based product."
    )
    await evaluator.verify(
        claim=claim_phys,
        node=physical_leaf,
        sources=_merge_urls(data.urls.physical_holdings, data.urls.general, data.urls.custodian),
        additional_instruction=(
            "Pass only if the page explicitly indicates that the fund is spot/physically-backed, holds XRP directly, "
            "or uses language such as 'physically held XRP', 'spot XRP ETF', 'in-kind creations/redemptions of XRP', "
            "or confirms a qualified crypto custodian safekeeps XRP tokens for the trust."
        ),
    )

    # 2) Lowest expense ratio verification
    lowest_seq = evaluator.add_sequential(
        id="Lowest_Expense_Ratio_Verification",
        desc="Verify the identified ETF has the lowest expense ratio among all U.S.-listed spot XRP ETFs",
        parent=etf_primary,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_text(data.expense_ratio) and _has_urls(_merge_urls(data.urls.expense_ratio, data.urls.lowest_expense_ratio_comparison)),
        id="Expense_Ratio_Presence",
        desc="Expense ratio is provided with at least one supporting URL",
        parent=lowest_seq,
        critical=True,
    )

    comp_seq = evaluator.add_sequential(
        id="Comparative_Expense_Ratio_Evidence",
        desc="Provide the ETF's expense ratio and evidence that it is the lowest among spot XRP ETFs",
        parent=lowest_seq,
        critical=True,
    )

    er_leaf = evaluator.add_leaf(
        id="Expense_Ratio_Reference_URL",
        desc="Provide reference URL confirming the expense ratio and/or comparison",
        parent=comp_seq,
        critical=True,
    )
    claim_lowest = (
        f"The expense ratio (management fee) of the ETF '{data.etf_name}' ({data.ticker}) is exactly '{data.expense_ratio}', "
        f"and as of {AS_OF_DATE_TEXT} it is the lowest among all U.S.-listed spot XRP ETFs currently trading."
    )
    await evaluator.verify(
        claim=claim_lowest,
        node=er_leaf,
        sources=_merge_urls(data.urls.lowest_expense_ratio_comparison, data.urls.expense_ratio, data.urls.general),
        additional_instruction=(
            "Prefer pages that explicitly state both the expense ratio and that it is the lowest among U.S.-listed spot XRP ETFs. "
            "Accept reputable sources such as the issuer's website/fact sheet, official prospectus/press release, major exchanges, "
            "or well-known financial data providers (Nasdaq, NYSE, Cboe, Morningstar, Bloomberg, Reuters, FactSet, etc.). "
            "If the page only confirms the expense ratio but does not substantiate 'lowest among peers', judge the 'lowest' part unsupported."
        ),
    )


async def build_required_details_subtree(evaluator: Evaluator, parent_node, data: XRPEtfExtraction) -> None:
    """
    Build and verify the 'ETF_Required_Details_Collection' subtree:
    - Ticker symbol
    - Custodian
    - Launch date (also check within [2024-11-01, 2025-12-31])
    - Exchange (must be NYSE Arca, Nasdaq, or Cboe)
    - Benchmark index
    """
    details_par = evaluator.add_parallel(
        id="ETF_Required_Details_Collection",
        desc="Collect all required identifying and operational details about the ETF",
        parent=parent_node,
        critical=True,
    )

    # Ticker
    evaluator.add_custom_node(
        result=_has_text(data.ticker) and _has_urls(_merge_urls(data.urls.ticker, data.urls.general, data.urls.exchange)),
        id="Ticker_Symbol_Presence",
        desc="Ticker symbol value and at least one URL are provided",
        parent=details_par,
        critical=True,
    )
    ticker_leaf = evaluator.add_leaf(
        id="Ticker_Symbol_Detail",
        desc="Provide the correct ticker symbol for the identified ETF",
        parent=details_par,
        critical=True,
    )
    claim_ticker = f"The ETF's ticker symbol is '{data.ticker}'."
    await evaluator.verify(
        claim=claim_ticker,
        node=ticker_leaf,
        sources=_merge_urls(data.urls.ticker, data.urls.general, data.urls.exchange),
        additional_instruction=(
            "Confirm the ticker symbol exactly as shown on the fund's official website, fact sheet, prospectus, "
            "or the primary exchange/professional data provider page."
        ),
    )

    # Custodian
    custodian_seq = evaluator.add_sequential(
        id="Custodian_Detail",
        desc="Provide the name of the institutional custodian holding the ETF's XRP and reference URL",
        parent=details_par,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(data.custodian) and _has_urls(_merge_urls(data.urls.custodian, data.urls.general)),
        id="Custodian_Presence",
        desc="Custodian value and at least one URL are provided",
        parent=custodian_seq,
        critical=True,
    )
    custodian_leaf = evaluator.add_leaf(
        id="Custodian_Reference_URL",
        desc="Provide reference URL confirming the custodian information",
        parent=custodian_seq,
        critical=True,
    )
    claim_custodian = f"The ETF's XRP holdings are custodied by '{data.custodian}'."
    await evaluator.verify(
        claim=claim_custodian,
        node=custodian_leaf,
        sources=_merge_urls(data.urls.custodian, data.urls.general),
        additional_instruction=(
            "Look for explicit custodian naming like 'Custodian', 'Trust Company', or 'Custodian Bank' on the issuer's materials "
            "or reputable sources. Examples: 'Coinbase Custody Trust Company, LLC'."
        ),
    )

    # Launch date
    launch_seq = evaluator.add_sequential(
        id="Launch_Date_Detail",
        desc="Provide the fund inception or launch date and verify it falls within November 2024 to December 2025",
        parent=details_par,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(data.launch_date) and _has_urls(_merge_urls(data.urls.launch_date, data.urls.general, data.urls.exchange)),
        id="Launch_Date_Presence",
        desc="Launch/Inception date value and at least one URL are provided",
        parent=launch_seq,
        critical=True,
    )
    launch_leaf = evaluator.add_leaf(
        id="Launch_Date_Reference_URL",
        desc="Provide reference URL confirming the launch date",
        parent=launch_seq,
        critical=True,
    )
    claim_launch = (
        f"The fund's inception/launch/listing date is '{data.launch_date}', and this date falls between "
        f"November 1, 2024 and December 31, 2025 inclusive."
    )
    await evaluator.verify(
        claim=claim_launch,
        node=launch_leaf,
        sources=_merge_urls(data.urls.launch_date, data.urls.general, data.urls.exchange),
        additional_instruction=(
            f"First, confirm the page states the fund's inception/launch/listing date. "
            f"Then verify that the date lies within {LAUNCH_RANGE_START} and {LAUNCH_RANGE_END} (inclusive). "
            f"Accept synonyms like 'Inception Date', 'First Trading Date', 'Listing Date'."
        ),
    )

    # Exchange
    exchange_seq = evaluator.add_sequential(
        id="Exchange_Listing_Detail",
        desc="Provide the primary exchange where the ETF is listed (must be NYSE Arca, Nasdaq, or CBOE)",
        parent=details_par,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(data.exchange) and _has_urls(_merge_urls(data.urls.exchange, data.urls.general)),
        id="Exchange_Presence",
        desc="Primary exchange value and at least one URL are provided",
        parent=exchange_seq,
        critical=True,
    )
    exchange_leaf = evaluator.add_leaf(
        id="Exchange_Reference_URL",
        desc="Provide reference URL confirming the exchange listing",
        parent=exchange_seq,
        critical=True,
    )
    claim_exchange = (
        f"The ETF is listed on the primary exchange '{data.exchange}', which must be NYSE Arca, Nasdaq, or Cboe."
    )
    await evaluator.verify(
        claim=claim_exchange,
        node=exchange_leaf,
        sources=_merge_urls(data.urls.exchange, data.urls.general),
        additional_instruction=(
            "Confirm the primary exchange name on the page. Consider synonymous legal names equivalent, e.g., "
            "'NYSE Arca, Inc.' == 'NYSE Arca'; 'The Nasdaq Stock Market LLC' == 'Nasdaq'; "
            "'Cboe BZX Exchange, Inc.' == 'Cboe'/'Cboe BZX'. If the exchange is not one of these families, judge incorrect."
        ),
    )

    # Benchmark index
    benchmark_seq = evaluator.add_sequential(
        id="Benchmark_Index_Detail",
        desc="Provide the industry-standard XRP price benchmark index that the ETF tracks",
        parent=details_par,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_text(data.benchmark_index) and _has_urls(_merge_urls(data.urls.benchmark, data.urls.general)),
        id="Benchmark_Presence",
        desc="Benchmark index value and at least one URL are provided",
        parent=benchmark_seq,
        critical=True,
    )
    benchmark_leaf = evaluator.add_leaf(
        id="Benchmark_Reference_URL",
        desc="Provide reference URL confirming the benchmark index",
        parent=benchmark_seq,
        critical=True,
    )
    claim_benchmark = f"The ETF tracks the benchmark index '{data.benchmark_index}'."
    await evaluator.verify(
        claim=claim_benchmark,
        node=benchmark_leaf,
        sources=_merge_urls(data.urls.benchmark, data.urls.general),
        additional_instruction=(
            "Accept standard XRP spot price indices such as those from CF Benchmarks, CoinDesk Indices, or similar. "
            "Synonymous index namings that clearly refer to the same benchmark are acceptable."
        ),
    )


async def build_verification_tree(evaluator: Evaluator, data: XRPEtfExtraction) -> None:
    """
    Build the full verification tree according to the rubric JSON.
    """
    # Root task node (critical, sequential)
    task_root = evaluator.add_sequential(
        id="XRP_ETF_Complete_Task",
        desc="Complete identification and verification of the U.S.-listed spot XRP ETF with the lowest expense ratio, with all required details and constraint verification",
        parent=evaluator.root,
        critical=True,
    )

    # Subtree: Primary identification (spot + lowest ER)
    await build_primary_identification_subtree(evaluator, task_root, data)

    # Subtree: Required details collection (parallel)
    await build_required_details_subtree(evaluator, task_root, data)


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the XRP spot ETF with the lowest expense ratio task.
    """
    # Initialize evaluator (root is non-critical; we add our own critical root task node)
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured ETF info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_xrp_etf_info(),
        template_class=XRPEtfExtraction,
        extraction_name="xrp_etf_extraction",
    )

    # Add handy ground-truth constraints for transparency (not used for verification)
    evaluator.add_ground_truth({
        "as_of": AS_OF_DATE_TEXT,
        "valid_exchanges": VALID_EXCHANGES,
        "launch_date_range_inclusive": [LAUNCH_RANGE_START, LAUNCH_RANGE_END],
        "require_official_or_reputable_sources": True
    })

    # Build verification tree per rubric and run checks
    await build_verification_tree(evaluator, extracted_info)

    # Return standardized summary
    return evaluator.get_summary()