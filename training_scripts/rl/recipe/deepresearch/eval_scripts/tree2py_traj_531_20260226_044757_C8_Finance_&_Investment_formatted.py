import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "xrp_spot_etf_lowest_fee"
TASK_DESCRIPTION = """
Among all SEC-approved spot XRP exchange-traded funds (ETFs) currently trading in the United States as of February 2026, identify the one with the lowest ongoing expense ratio (permanent management fee, not temporary promotional rates).

For the identified ETF, provide the following information:

1. ETF Details:
   - Full ETF product name
   - Issuing company/sponsor name
   - Official trading ticker symbol
   - The ongoing (permanent) expense ratio percentage

2. Regulatory and Trading Information:
   - Confirmation that it is a spot ETF (holds actual XRP, not futures or derivatives)
   - The U.S. stock exchange where it is listed
   - Current trading status

3. SEC Filing Information:
   - The exact date when the issuer filed Form S-1 with the SEC for this ETF
   - A direct URL link to the Form S-1 document on SEC EDGAR

4. Fee Context:
   - If the ETF has any temporary fee waiver program, clearly state:
     - The promotional fee (if different from ongoing fee)
     - The end date of the fee waiver
   - Distinguish between temporary promotional rates and the permanent ongoing expense ratio

5. Comparative Analysis:
   - List at least three other SEC-approved spot XRP ETFs
   - Provide their expense ratios to demonstrate why your identified ETF has the lowest ongoing fee

6. Verification:
   - Include a URL to the issuer's official product page
   - Include a URL to at least one third-party source (e.g., financial news site, ETF database) that confirms the fee structure

Important: The "ongoing expense ratio" refers to the permanent management fee that will apply after any promotional periods end, not temporary waived or discounted rates.
"""

ALLOWED_EXCHANGES = ["nyse arca", "cboe bzx", "nasdaq"]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SelectedETFInfo(BaseModel):
    # Core identifiers
    product_name: Optional[str] = None
    issuer_name: Optional[str] = None
    ticker: Optional[str] = None
    ongoing_expense_ratio: Optional[str] = None  # e.g., "0.19%" or "19 bps"

    # Product type and status
    spot_confirmed_text: Optional[str] = None  # text confirming spot holding
    sec_approved_text: Optional[str] = None    # text indicating SEC approval
    exchange_name: Optional[str] = None
    trading_status_asof_feb2026: Optional[str] = None

    # SEC filing info
    s1_filing_date: Optional[str] = None       # exact date string
    edgar_s1_url: Optional[str] = None         # direct EDGAR URL

    # Fee context
    has_fee_waiver: Optional[bool] = None
    promotional_fee: Optional[str] = None
    fee_waiver_end_date: Optional[str] = None
    fee_distinction_statement: Optional[str] = None  # explicit distinction text

    # Verification URLs
    issuer_product_page_url: Optional[str] = None
    third_party_fee_source_urls: List[str] = Field(default_factory=list)

    # AUM evidence
    aum_source_url: Optional[str] = None


class CompetitorETF(BaseModel):
    product_name: Optional[str] = None
    ticker: Optional[str] = None
    ongoing_expense_ratio: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class CompetitorExtraction(BaseModel):
    competitors: List[CompetitorETF] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_selected_etf() -> str:
    return """
    Extract details for the single ETF identified by the answer as having the lowest ongoing (permanent) expense ratio among SEC-approved spot XRP ETFs trading in the U.S. as of February 2026.

    Required fields:
    1) product_name: Full ETF product name.
    2) issuer_name: Full legal name of the issuing company/sponsor.
    3) ticker: Official trading ticker symbol.
    4) ongoing_expense_ratio: The permanent ongoing expense ratio AFTER promotions end (e.g., "0.19%" or "19 bps"). Do NOT extract temporary promotional rates here.

    Regulatory & trading:
    5) spot_confirmed_text: Text or phrase from the answer confirming it is a spot ETF that holds actual XRP (not futures/derivatives). If unclear or missing, set null.
    6) sec_approved_text: Text indicating the ETF is SEC-approved to list/trade in the U.S. If missing, set null.
    7) exchange_name: The named U.S. stock exchange where it is listed (e.g., NYSE Arca, Cboe BZX, Nasdaq). If missing, set null.
    8) trading_status_asof_feb2026: The stated trading status as of February 2026 (e.g., "actively trading"). If missing, set null.

    SEC filing information:
    9) s1_filing_date: The exact date the issuer filed Form S-1 for this ETF (as stated in the answer). If missing, set null.
    10) edgar_s1_url: Direct URL to the Form S-1 document on SEC EDGAR. If missing, set null.

    Fee context:
    11) has_fee_waiver: true/false if the answer states the selected ETF has a temporary fee waiver or promotional fee program. If not mentioned, set false.
    12) promotional_fee: The temporary promotional fee (if different from ongoing). If not applicable or missing, set null.
    13) fee_waiver_end_date: The end date for the temporary fee waiver (if any). If not applicable or missing, set null.
    14) fee_distinction_statement: Text explicitly distinguishing temporary promotional rates from the permanent ongoing expense ratio. If missing, set null.

    Verification URLs:
    15) issuer_product_page_url: URL to the issuer's official product page. If missing, set null.
    16) third_party_fee_source_urls: Array of at least one third-party URL (e.g., news site, ETF database) confirming the fee structure. If none provided, return empty array.

    AUM evidence:
    17) aum_source_url: A URL where the ETF publicly reports its AUM (if provided). If missing, set null.

    Return a single JSON object with these fields. Extract exactly what is present in the answer; do not invent information. If any required item is missing, set it to null unless specified otherwise.
    """


def prompt_extract_competitors(selected_name: Optional[str], selected_ticker: Optional[str]) -> str:
    base = """
    Extract at least three other SEC-approved spot XRP ETFs (competitors) mentioned in the answer for comparison. Do NOT include the selected ETF itself in this list.

    For each competitor, extract:
    1) product_name: Full product name.
    2) ticker: Official trading ticker (if provided).
    3) ongoing_expense_ratio: The permanent ongoing expense ratio (post-promotion). Do NOT extract temporary promotional rates here.
    4) source_urls: Array of URLs that support the expense ratio or product facts (preferably an issuer page and/or credible third-party).

    Return a JSON object with a 'competitors' array of objects with the above fields. If fewer than three competitors are present in the answer, include whatever is available.
    """
    excludes = []
    if selected_name:
        excludes.append(f'Exclude any ETF named "{selected_name}".')
    if selected_ticker:
        excludes.append(f'Exclude any ETF with ticker "{selected_ticker}".')
    exclude_text = ("\n" + "\n".join(excludes)) if excludes else ""
    return base + exclude_text


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_exchange_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.strip().lower()
    # Normalize common variants
    s = s.replace("exchange", "").replace("stock market", "").strip()
    s = s.replace("nasdaq", "nasdaq")
    s = s.replace("nysearca", "nyse arca").replace("nyse arca", "nyse arca")
    s = s.replace("cboe bzx", "cboe bzx").replace("cboe", "cboe").strip()
    return s


def is_allowed_exchange(name: Optional[str]) -> bool:
    n = normalize_exchange_name(name)
    if not n:
        return False
    # Allow substring match to handle minor variants
    return any(ex in n for ex in ALLOWED_EXCHANGES)


def parse_expense_ratio_to_percent(value: Optional[str]) -> Optional[float]:
    """
    Convert a string like "0.19%", "19 bps", "0.30 %" to a percent float (e.g., 0.19).
    If unable to parse, return None.
    """
    if not value:
        return None
    s = value.lower().strip()
    numbers = re.findall(r"[\d.]+", s)
    if not numbers:
        return None
    num = float(numbers[0])
    if "bp" in s or "bps" in s:
        # basis points to percent
        return num / 100.0
    if "%" in s:
        return num
    # No unit provided; heuristic: treat < 1 as percent (e.g., 0.19), >= 1 as percent already (e.g., 0.19 without %)
    return num


def first_n_competitors(comp: CompetitorExtraction, n: int = 3) -> List[CompetitorETF]:
    items = [c for c in comp.competitors if (c.product_name or c.ticker)]
    return items[:n]


def combine_sources(*args: List[Optional[str]]) -> List[str]:
    urls: List[str] = []
    for arg in args:
        if isinstance(arg, list):
            for u in arg:
                if isinstance(u, str) and u.strip():
                    urls.append(u.strip())
        elif isinstance(arg, str) and arg.strip():
            urls.append(arg.strip())
        # ignore None
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


# --------------------------------------------------------------------------- #
# Verification sub-tree construction                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    selected: SelectedETFInfo,
    competitors: CompetitorExtraction,
    parent_root
) -> None:
    # Top-level critical parallel node
    main_task = evaluator.add_parallel(
        id="Spot_XRP_ETF_Lowest_Fee_Task",
        desc="Evaluate identification and documentation of the SEC-approved, currently trading U.S. spot XRP ETF with the lowest ongoing (permanent) expense ratio, with required evidence/citations.",
        parent=parent_root,
        critical=True
    )

    # 1) ETF Basic Identification
    basic_node = evaluator.add_parallel(
        id="ETF_Basic_Identification",
        desc="Answer provides the core identifiers for the selected ETF.",
        parent=main_task,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(selected.product_name and selected.product_name.strip()),
        id="ETF_Full_Name_Provided",
        desc="States the full ETF product name.",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(selected.issuer_name and selected.issuer_name.strip()),
        id="Issuer_Legal_Name_Provided",
        desc="Provides the full legal name of the issuing company/sponsor.",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(selected.ticker and selected.ticker.strip()),
        id="Ticker_Provided",
        desc="Includes the official trading ticker symbol.",
        parent=basic_node,
        critical=True
    )

    # 2) Product Type & Status Constraints
    pts_node = evaluator.add_parallel(
        id="Product_Type_And_Status_Constraints",
        desc="Selected ETF satisfies product type, approval, listing, and trading-status constraints, and the answer states the requested listing/trading info.",
        parent=main_task,
        critical=True
    )

    spot_leaf = evaluator.add_leaf(
        id="Spot_XRP_Holding_Confirmed",
        desc="Confirms it is a spot XRP ETF that holds actual XRP (not futures/derivatives).",
        parent=pts_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF '{selected.product_name or 'UNKNOWN'}' is a spot XRP ETF that holds actual XRP (not futures or derivatives).",
        node=spot_leaf,
        sources=combine_sources(selected.issuer_product_page_url, selected.third_party_fee_source_urls),
        additional_instruction="Look for language such as 'spot ETF', 'holds XRP in trust', or similar. Confirm it is not a futures-based product."
    )

    sec_approved_leaf = evaluator.add_leaf(
        id="SEC_Approved_Confirmed",
        desc="Confirms the ETF is SEC-approved for U.S. listing/trading.",
        parent=pts_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF '{selected.product_name or 'UNKNOWN'}' is SEC-approved for U.S. listing/trading.",
        node=sec_approved_leaf,
        sources=combine_sources(selected.third_party_fee_source_urls, selected.issuer_product_page_url),
        additional_instruction="Confirm that the ETF has SEC approval (e.g., effectiveness of S-1 and/or approval of the rule change to list). News or issuer communications that explicitly state SEC approval are acceptable."
    )

    evaluator.add_custom_node(
        result=bool(selected.exchange_name and selected.exchange_name.strip()),
        id="US_Exchange_Named",
        desc="Names the U.S. exchange where it is listed.",
        parent=pts_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=is_allowed_exchange(selected.exchange_name),
        id="US_Exchange_Is_Recognized_Allowed_Exchange",
        desc="The named exchange is one of the allowed exchanges (NYSE Arca, Cboe BZX, or Nasdaq).",
        parent=pts_node,
        critical=True
    )

    trading_leaf = evaluator.add_leaf(
        id="Currently_Trading_Status_Stated_AsOf_Feb_2026",
        desc="States the ETF's trading status and indicates it is actively trading as of February 2026.",
        parent=pts_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ETF '{selected.product_name or 'UNKNOWN'}' is actively trading as of February 2026.",
        node=trading_leaf,
        sources=combine_sources(selected.issuer_product_page_url, selected.third_party_fee_source_urls),
        additional_instruction="Check issuer/exchange/product page or credible sources that indicate the ETF is live/actively trading in February 2026."
    )

    # 3) Expense Ratio & Lowest Fee Requirement
    fee_node = evaluator.add_parallel(
        id="Expense_Ratio_And_Lowest_Fee_Requirement",
        desc="Answer states the ongoing (permanent) expense ratio and correctly identifies the lowest such fee among SEC-approved spot XRP ETFs trading in the U.S. as of Feb 2026.",
        parent=main_task,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(selected.ongoing_expense_ratio and selected.ongoing_expense_ratio.strip()),
        id="Ongoing_Expense_Ratio_Percentage_Provided",
        desc="Provides the ongoing (permanent) expense ratio percentage for the selected ETF.",
        parent=fee_node,
        critical=True
    )

    # Prepare competitor info
    top_comp = first_n_competitors(competitors, 3)
    comp_ratios: List[Optional[float]] = [parse_expense_ratio_to_percent(c.ongoing_expense_ratio) for c in top_comp]
    selected_ratio = parse_expense_ratio_to_percent(selected.ongoing_expense_ratio)

    # Comparative nodes to use as prerequisites for "lowest fee" verification
    comp_node = evaluator.add_parallel(
        id="Comparative_Analysis_With_Other_ETFs",
        desc="Answer provides the required comparison set to justify the lowest-fee claim.",
        parent=main_task,
        critical=True
    )
    at_least_three_leaf = evaluator.add_custom_node(
        result=len(top_comp) >= 3,
        id="At_Least_Three_Other_Spot_XRP_ETFs_Listed",
        desc="Lists at least three other SEC-approved spot XRP ETFs.",
        parent=comp_node,
        critical=True
    )
    comp_ratios_provided_leaf = evaluator.add_custom_node(
        result=all(bool(c and c.strip()) for c in [tc.ongoing_expense_ratio or "" for tc in top_comp]),
        id="Competitor_Expense_Ratios_Provided",
        desc="Provides the expense ratios for the listed comparison ETFs to demonstrate why the selected ETF is the lowest ongoing fee.",
        parent=comp_node,
        critical=True
    )

    # Lowest fee selection check (logical comparison; depends on comparative leaves)
    lowest_leaf = evaluator.add_leaf(
        id="Lowest_Ongoing_Fee_Selection_Is_Correct",
        desc="The selected ETF is verifiably the lowest ongoing-expense-ratio option among SEC-approved spot XRP ETFs currently trading in the U.S. as of February 2026.",
        parent=fee_node,
        critical=True
    )
    # Build a clear numeric comparison claim
    comp_descs = []
    for i, c in enumerate(top_comp):
        c_pct = comp_ratios[i]
        if c_pct is not None:
            comp_descs.append(f"{c.product_name or c.ticker or f'Competitor #{i+1}'} ({c_pct}%)")
        else:
            comp_descs.append(f"{c.product_name or c.ticker or f'Competitor #{i+1}'} (unknown%)")
    claim_lowest = (
        f"The ETF '{selected.product_name or 'UNKNOWN'}' has an ongoing expense ratio of "
        f"{selected_ratio if selected_ratio is not None else selected.ongoing_expense_ratio or 'UNKNOWN'}% "
        f"which is lower than those of the other spot XRP ETFs: {', '.join(comp_descs)}."
    )
    await evaluator.verify(
        claim=claim_lowest,
        node=lowest_leaf,
        additional_instruction="Focus on the numeric comparison among the ongoing (permanent) expense ratios. Treat 'bps' as basis points (e.g., 19 bps = 0.19%). Allow minor rounding differences.",
        extra_prerequisites=[at_least_three_leaf, comp_ratios_provided_leaf]
    )

    # 4) Fee Context: Temporary vs Permanent
    fee_ctx_node = evaluator.add_parallel(
        id="Fee_Context_Temporary_Vs_Permanent",
        desc="Answer handles temporary promotional fees/waivers correctly (when present) and distinguishes them from the permanent ongoing expense ratio.",
        parent=main_task,
        critical=True
    )
    # Conditional leaf: pass if no waiver; else require both fields present
    has_waiver = bool(selected.has_fee_waiver) or bool(selected.promotional_fee) or bool(selected.fee_waiver_end_date)
    evaluator.add_custom_node(
        result=(not has_waiver) or (bool(selected.promotional_fee and selected.fee_waiver_end_date)),
        id="Temporary_Waiver_Details_If_Present",
        desc="If the ETF has a temporary fee waiver/promotional program, provides the promotional fee (if different) and the waiver end date.",
        parent=fee_ctx_node,
        critical=True
    )

    distinguish_leaf = evaluator.add_leaf(
        id="Temporary_Vs_Permanent_Fee_Distinguished",
        desc="Clearly distinguishes any temporary promotional/waived rate from the permanent ongoing expense ratio that applies after promotions end.",
        parent=fee_ctx_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer clearly distinguishes temporary promotional fee rates from the permanent ongoing expense ratio.",
        node=distinguish_leaf,
        additional_instruction="Examine the answer text to ensure it explicitly differentiates promotional/temporary fees from the ongoing (permanent) management fee."
    )

    # 5) SEC Form S-1 Filing Information
    s1_node = evaluator.add_parallel(
        id="SEC_Form_S1_Filing_Information",
        desc="Answer provides verifiable SEC Form S-1 filing details via EDGAR.",
        parent=main_task,
        critical=True
    )

    edgar_url_ok = bool(selected.edgar_s1_url and selected.edgar_s1_url.strip() and ("sec.gov" in selected.edgar_s1_url))
    evaluator.add_custom_node(
        result=edgar_url_ok,
        id="EDGAR_S1_URL_Provided",
        desc="Includes a direct URL link to the Form S-1 document on SEC EDGAR.",
        parent=s1_node,
        critical=True
    )

    s1_date_leaf = evaluator.add_leaf(
        id="S1_Filing_Date_Provided",
        desc="States the exact date the issuer filed Form S-1 for this ETF.",
        parent=s1_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Form S-1 filing date for '{selected.product_name or 'UNKNOWN'}' is {selected.s1_filing_date or 'UNKNOWN'}.",
        node=s1_date_leaf,
        sources=selected.edgar_s1_url if selected.edgar_s1_url else None,
        additional_instruction="On the EDGAR page, verify the stated date (e.g., 'Filing Date'). If multiple dates appear (amendments), use the filing date claimed in the answer."
    )

    # 6) Verification URLs Provided
    vurls_node = evaluator.add_parallel(
        id="Verification_URLs_Provided",
        desc="Answer includes the required verification sources.",
        parent=main_task,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(selected.issuer_product_page_url and selected.issuer_product_page_url.strip()),
        id="Issuer_Product_Page_URL",
        desc="Includes a URL to the issuer's official product page for the ETF.",
        parent=vurls_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(selected.third_party_fee_source_urls and len(selected.third_party_fee_source_urls) >= 1),
        id="Third_Party_Fee_Structure_Source_URL",
        desc="Includes at least one third-party URL that confirms the fee structure.",
        parent=vurls_node,
        critical=True
    )

    # 7) AUM Public Reporting Constraint
    aum_node = evaluator.add_parallel(
        id="AUM_Public_Reporting_Constraint_Satisfied",
        desc="Meets the constraint that the ETF publicly reports its Assets Under Management (AUM).",
        parent=main_task,
        critical=True
    )
    aum_leaf = evaluator.add_leaf(
        id="AUM_Publicly_Reported_Confirmed",
        desc="Provides evidence that the ETF publicly reports AUM (e.g., cites a source where AUM is shown).",
        parent=aum_node,
        critical=True
    )
    aum_sources = combine_sources(selected.aum_source_url, selected.issuer_product_page_url, selected.third_party_fee_source_urls)
    await evaluator.verify(
        claim=f"The ETF '{selected.product_name or 'UNKNOWN'}' publicly reports its AUM.",
        node=aum_leaf,
        sources=aum_sources if aum_sources else None,
        additional_instruction="Check the provided source(s) for an AUM figure or reporting section. Issuer pages or credible trackers are acceptable evidence."
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
    Evaluate an answer for the XRP spot ETF lowest ongoing fee task.
    """
    # Initialize evaluator
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

    # Extract selected ETF details first
    selected_info = await evaluator.extract(
        prompt=prompt_extract_selected_etf(),
        template_class=SelectedETFInfo,
        extraction_name="selected_etf_info"
    )

    # Extract competitors (use additional instruction to exclude selected)
    comp_info = await evaluator.extract(
        prompt=prompt_extract_competitors(selected_info.product_name, selected_info.ticker),
        template_class=CompetitorExtraction,
        extraction_name="competitors_info",
        additional_instruction="Ensure you exclude the selected ETF from the competitors list. Capture ongoing (permanent) expense ratios, not promotional rates."
    )

    # Add context information for transparency
    evaluator.add_ground_truth({
        "task_focus": "Identify SEC-approved U.S. spot XRP ETF with lowest ongoing (permanent) expense ratio as of Feb 2026",
        "allowed_exchanges": ALLOWED_EXCHANGES
    }, gt_type="task_context")

    # Build verification tree and run checks
    await build_verification_tree(evaluator, selected_info, comp_info, parent_root=root)

    # Return structured summary
    return evaluator.get_summary()