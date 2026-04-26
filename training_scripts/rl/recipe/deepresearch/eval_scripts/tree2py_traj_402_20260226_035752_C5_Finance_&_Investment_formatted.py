import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "xrp_etf_lowest_fee_2025"
TASK_DESCRIPTION = """
Which U.S.-listed spot XRP exchange-traded fund (ETF) that launched between November 2025 and December 2025 (inclusive)
has the lowest stated expense ratio, excluding any temporary promotional fee waivers? Provide the ETF's ticker symbol,
issuer name, exact launch date, the stated expense ratio, and supporting reference URLs from official sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SelectedETF(BaseModel):
    """The ETF the answer identifies as having the lowest non‑waived expense ratio."""
    ticker: Optional[str] = None
    issuer: Optional[str] = None
    launch_date: Optional[str] = None  # Keep as string to be flexible (e.g., "Nov 12, 2025")
    expense_ratio: Optional[str] = None  # Keep as string (can be "0.25%" or "25 bps")
    exchange_name: Optional[str] = None  # If the answer mentions NYSE/Nasdaq/Cboe, etc.
    official_urls: List[str] = Field(default_factory=list)  # Fund page, SEC filings, exchange notices, etc.


class CompetitorETF(BaseModel):
    """Other ETFs in the stated window considered for comparison."""
    ticker: Optional[str] = None
    issuer: Optional[str] = None
    expense_ratio: Optional[str] = None  # Non‑waived stated expense ratio
    urls: List[str] = Field(default_factory=list)  # Official source URLs for this competitor


class XRPETFExtraction(BaseModel):
    """Aggregation of the selected ETF and competitors as presented in the answer."""
    selected: Optional[SelectedETF] = None
    competitors: List[CompetitorETF] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_xrp_etf_lowest_fee() -> str:
    return """
    From the answer, extract the single ETF the answer claims has the lowest stated (non‑waived) expense ratio among U.S.-listed
    spot XRP ETFs launched in Nov–Dec 2025, and also extract the competitors cited for comparison.

    1) selected:
       - ticker: the ETF ticker symbol
       - issuer: the issuer/sponsor name
       - launch_date: the exact launch date stated (keep as provided in the answer)
       - expense_ratio: the stated ongoing (non‑waived) expense ratio / management fee used for comparison
       - exchange_name: the U.S. exchange name if explicitly mentioned (e.g., NYSE, Nasdaq, Cboe)
       - official_urls: array of official source URLs referenced in the answer (issuer fund page, SEC filings/prospectus,
                        official exchange listing notices). Extract only URLs explicitly present in the answer.

    2) competitors: array of objects, each for another ETF considered in the comparison:
       - ticker: competitor ETF ticker
       - issuer: competitor issuer/sponsor
       - expense_ratio: competitor’s non‑waived stated expense ratio used for comparison
       - urls: array of official source URLs cited in the answer for this competitor (issuer/SEC/exchange) — only URLs explicitly present.

    Rules:
    - Do not invent information. If a field is missing in the answer, return null or an empty array as appropriate.
    - URLs must be actually present in the answer (including markdown links). Ignore non‑URL citations.
    - Prefer official sources (issuer/SEC/exchange) when the answer provides them; extract all official URLs mentioned for each item.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s) and bool(s.strip())


def _unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not _non_empty_str(u):
            continue
        key = u.strip()
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _collect_all_urls(selected: Optional[SelectedETF], competitors: List[CompetitorETF]) -> List[str]:
    urls: List[str] = []
    if selected and selected.official_urls:
        urls.extend(selected.official_urls)
    for comp in competitors:
        urls.extend(comp.urls)
    return _unique_urls(urls)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_answer_fields_provided(
    evaluator: Evaluator,
    parent_node,
    extraction: XRPETFExtraction
) -> Dict[str, Any]:
    """
    Build 'Answer_Fields_Provided' node and its critical leaf checks.
    Returns a dict with references to created nodes (useful as prerequisites).
    """
    selected = extraction.selected or SelectedETF()

    group = evaluator.add_parallel(
        id="Answer_Fields_Provided",
        desc="Answer provides all required output fields for the identified ETF.",
        parent=parent_node,
        critical=True,
    )

    n_ticker = evaluator.add_custom_node(
        result=_non_empty_str(selected.ticker),
        id="Ticker_Provided",
        desc="Provides the ETF ticker symbol.",
        parent=group,
        critical=True
    )

    n_issuer = evaluator.add_custom_node(
        result=_non_empty_str(selected.issuer),
        id="Issuer_Name_Provided",
        desc="Provides the issuer/sponsor name.",
        parent=group,
        critical=True
    )

    n_launch = evaluator.add_custom_node(
        result=_non_empty_str(selected.launch_date),
        id="Exact_Launch_Date_Provided",
        desc="Provides the ETF’s exact launch date.",
        parent=group,
        critical=True
    )

    n_expense = evaluator.add_custom_node(
        result=_non_empty_str(selected.expense_ratio),
        id="Stated_Expense_Ratio_Provided",
        desc="Provides the ETF’s stated ongoing expense ratio/management fee (not a temporary waived/promotional rate).",
        parent=group,
        critical=True
    )

    n_urls = evaluator.add_custom_node(
        result=bool(selected.official_urls) and len(selected.official_urls) > 0,
        id="Official_Source_URLs_Provided",
        desc="Provides supporting reference URL(s) from official sources (e.g., issuer fund page and/or SEC filing/prospectus and/or official exchange listing notice).",
        parent=group,
        critical=True
    )

    return {
        "group": group,
        "Ticker_Provided": n_ticker,
        "Issuer_Name_Provided": n_issuer,
        "Exact_Launch_Date_Provided": n_launch,
        "Stated_Expense_Ratio_Provided": n_expense,
        "Official_Source_URLs_Provided": n_urls,
    }


async def build_etf_qualification_constraints(
    evaluator: Evaluator,
    parent_node,
    extraction: XRPETFExtraction,
    prereq_nodes: Dict[str, Any]
) -> None:
    """
    Build 'ETF_Qualification_Constraints' and run verifications against official sources.
    """
    selected = extraction.selected or SelectedETF()
    sources = _unique_urls(selected.official_urls)

    group = evaluator.add_parallel(
        id="ETF_Qualification_Constraints",
        desc="The identified ETF satisfies all stated eligibility constraints.",
        parent=parent_node,
        critical=True
    )

    # US listed
    leaf_us_listed = evaluator.add_leaf(
        id="US_Listed",
        desc="ETF is U.S.-listed (supported by an official source citation).",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF {selected.ticker or ''} is listed in the United States.",
        node=leaf_us_listed,
        sources=sources,
        additional_instruction="Use official sources (issuer/SEC/exchange) to confirm U.S. listing.",
    )

    # Trades on major U.S. exchange
    leaf_major_exchange = evaluator.add_leaf(
        id="Trades_On_Major_US_Exchange",
        desc="ETF is trading on a major U.S. exchange and this is supported by an official exchange/issuer citation.",
        parent=group,
        critical=True
    )
    ex_name = selected.exchange_name or "a major U.S. exchange (NYSE, Nasdaq, or Cboe)"
    await evaluator.verify(
        claim=f"The ETF {selected.ticker or ''} trades on {ex_name}.",
        node=leaf_major_exchange,
        sources=sources,
        additional_instruction="Confirm via official exchange/issuer/SEC sources. Accept variations like 'listed on' or 'trading on'.",
    )

    # Spot XRP ETF
    leaf_spot_xrp = evaluator.add_leaf(
        id="Spot_XRP_ETF",
        desc="ETF is a spot XRP ETF (not futures-based or leveraged), supported by official documentation.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF {selected.ticker or ''} is a physically-backed spot XRP ETF (not futures-based or leveraged).",
        node=leaf_spot_xrp,
        sources=sources,
        additional_instruction="Check issuer/SEC/exchange docs for language indicating spot exposure, holding XRP, or physically-backed structure.",
    )

    # Launch window (Nov–Dec 2025 inclusive)
    leaf_launch_window = evaluator.add_leaf(
        id="Launch_Window_Nov_Dec_2025",
        desc="Official documentation supports that the launch date falls between Nov 2025 and Dec 2025 (inclusive).",
        parent=group,
        critical=True
    )
    ld_text = selected.launch_date or ""
    await evaluator.verify(
        claim=f"The official documentation shows a launch/trading/listing/commencement date of {ld_text} that falls between Nov 1, 2025 and Dec 31, 2025 inclusive.",
        node=leaf_launch_window,
        sources=sources,
        additional_instruction="Look for terms like 'launch', 'listing', 'trading commencement', or 'inception' date and ensure it lies within Nov–Dec 2025.",
    )

    # Launch date verifiable via SEC or exchange
    leaf_launch_verifiable = evaluator.add_leaf(
        id="Launch_Date_Verifiable_SEC_or_Exchange",
        desc="Launch date is verifiable through official SEC filings or an official exchange announcement/listing notice (via citation).",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The launch/trading date for {selected.ticker or ''} is verifiable from an SEC filing (e.g., prospectus/S-1/485) or an official exchange announcement/listing notice.",
        node=leaf_launch_verifiable,
        sources=sources,
        additional_instruction="Among the provided sources, at least one should be sec.gov or an official exchange domain (nasdaq.com, nyse.com, cboe.com) describing the listing/launch date.",
    )

    # Issuer filed for XRP ETF approval
    leaf_issuer_filed = evaluator.add_leaf(
        id="Issuer_Filed_For_XRP_ETF_Approval",
        desc="Issuer/sponsor has filed for XRP ETF approval (supported by regulatory/SEC filing evidence).",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The issuer {selected.issuer or ''} filed regulatory/SEC documents seeking approval for an XRP ETF.",
        node=leaf_issuer_filed,
        sources=sources,
        additional_instruction="Look for SEC filing references (e.g., 19b-4, S-1, N-1A, 485, or similar) tied to XRP exposure.",
    )

    # Issuer identified in official fund docs
    leaf_issuer_identified = evaluator.add_leaf(
        id="Issuer_Identified_In_Official_Fund_Documentation",
        desc="Issuer/sponsor is clearly identified in official fund documentation (e.g., prospectus/SEC filing/official fund page) via citation.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The issuer/sponsor name '{selected.issuer or ''}' is identified in official fund documentation for {selected.ticker or ''}.",
        node=leaf_issuer_identified,
        sources=sources,
        additional_instruction="Verify that issuer/sponsor entity appears explicitly in official docs or on the official fund page.",
    )

    # Expense ratio publicly disclosed
    leaf_expense_disclosed = evaluator.add_leaf(
        id="Expense_Ratio_Publicly_Disclosed",
        desc="Expense ratio/management fee is publicly disclosed in official fund documentation (e.g., prospectus/SEC filing/issuer page).",
        parent=group,
        critical=True
    )
    er_text = selected.expense_ratio or ""
    await evaluator.verify(
        claim=f"The ongoing (non‑waived) expense ratio/management fee for {selected.ticker or ''} is publicly disclosed as {er_text} in official documentation.",
        node=leaf_expense_disclosed,
        sources=sources,
        additional_instruction="Ensure the rate is the stated ongoing fee, not a temporary waived 'net' fee; use issuer/SEC documents to confirm.",
    )


async def build_lowest_expense_ratio_determination(
    evaluator: Evaluator,
    parent_node,
    extraction: XRPETFExtraction,
) -> None:
    """
    Build 'Lowest_Expense_Ratio_Determination' and run verifications.
    """
    selected = extraction.selected or SelectedETF()
    competitors = extraction.competitors or []
    all_sources = _collect_all_urls(selected, competitors)

    group = evaluator.add_parallel(
        id="Lowest_Expense_Ratio_Determination",
        desc="Correctly determines that the chosen ETF has the lowest stated expense ratio among all ETFs meeting the constraints, excluding temporary waivers.",
        parent=parent_node,
        critical=True
    )

    # Fee waiver exclusion
    leaf_fee_waiver_excluded = evaluator.add_leaf(
        id="Fee_Waiver_Excluded",
        desc="If any promotional/temporary fee waiver exists, the comparison uses the non-waived stated expense ratio rather than the waived/promotional rate.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The expense ratio used for {selected.ticker or ''} ({selected.expense_ratio or ''}) explicitly excludes temporary promotional fee waivers; it is the non‑waived ongoing rate.",
        node=leaf_fee_waiver_excluded,
        sources=_unique_urls(selected.official_urls),
        additional_instruction="Check official docs for waived/net vs. gross/ongoing fee. The chosen figure must be the non‑waived ongoing fee.",
    )

    # Lowest among qualifying ETFs
    leaf_lowest_among = evaluator.add_leaf(
        id="Lowest_Among_Qualifying_ETFs",
        desc="Provides sufficient evidence/citations to support that no other ETF meeting the same constraints in the specified window has a lower stated (non-waived) expense ratio.",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=f"No other U.S.-listed spot XRP ETF launched in Nov–Dec 2025 has a lower non‑waived stated expense ratio than {selected.expense_ratio or ''} for {selected.ticker or ''}.",
        node=leaf_lowest_among,
        sources=all_sources,
        additional_instruction=(
            "Use official sources for competitors (issuer/SEC/exchange) in the specified window (Nov–Dec 2025). "
            "Disregard temporary fee waivers. If a source states 'lowest' explicitly for the selected ETF, "
            "that can be considered sufficient if credible and official."
        ),
    )


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the XRP ETF lowest expense ratio task and return a structured result dictionary.
    """
    # Initialize evaluator with a CRITICAL root (all children under a critical parent must be critical)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
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

    # Make the root node critical according to rubric; children must also be critical
    root.critical = True

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_xrp_etf_lowest_fee(),
        template_class=XRPETFExtraction,
        extraction_name="xrp_etf_lowest_fee_extraction",
    )

    # Build 'Answer_Fields_Provided'
    prereq_nodes = await build_answer_fields_provided(evaluator, root, extraction)

    # Build 'ETF_Qualification_Constraints'
    await build_etf_qualification_constraints(evaluator, root, extraction, prereq_nodes)

    # Build 'Lowest_Expense_Ratio_Determination'
    await build_lowest_expense_ratio_determination(evaluator, root, extraction)

    # Add custom info about evaluation window for clarity
    evaluator.add_custom_info(
        info={"launch_window": {"start": "2025-11-01", "end": "2025-12-31", "inclusive": True}},
        info_type="window"
    )

    # Return structured summary
    return evaluator.get_summary()