import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "xrp_spot_etf_nov2025_lowest_fee"
TASK_DESCRIPTION = """
Identify the U.S. spot XRP ETF with the lowest expense ratio among the first three such ETFs to launch in November 2025. Provide the ETF's ticker symbol, primary listing exchange, and exact launch date (inception date).
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ETFItem(BaseModel):
    etf_name: Optional[str] = None
    ticker: Optional[str] = None
    exchange: Optional[str] = None
    launch_date: Optional[str] = None  # keep as string for flexibility (e.g., "2025-11-12", "Nov 12, 2025")
    expense_ratio: Optional[str] = None  # keep as string; may be "0.20%" or "20 bps"
    sources: List[str] = Field(default_factory=list)


class ETFExtraction(BaseModel):
    selected_etf: Optional[ETFItem] = None
    first_three_etfs: List[ETFItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_selected_etf() -> str:
    return """
    From the provided answer, extract the ETF the answer identifies as the U.S. spot XRP ETF with the lowest expense ratio among the first three to launch in November 2025.
    Return a JSON object with a field 'selected_etf' containing:
      - etf_name: The ETF's official name as presented in the answer (if provided).
      - ticker: The ETF's ticker symbol (if provided).
      - exchange: The ETF's primary listing exchange (if provided), e.g., "NYSE Arca", "Nasdaq", "Cboe BZX".
      - launch_date: The ETF's exact inception/launch/first-trading date (if provided), in whatever format is used in the answer.
      - expense_ratio: The ETF's expense ratio/management fee as stated in the answer (if provided).
      - sources: An array of all URLs in the answer that support this ETF's details (prefer issuer pages, exchange listings, prospectus/SEC filings, press releases, or reputable news pages). Extract actual URLs only; if no URLs are present, return an empty array.
    If the answer does not clearly identify such an ETF, set all fields in 'selected_etf' to null and 'sources' to an empty list.
    """


def prompt_extract_first_three_etfs() -> str:
    return """
    From the provided answer, extract up to three U.S. spot XRP ETFs that the answer presents as among the first three to launch in November 2025.
    Return a JSON object with a field 'first_three_etfs' which is an array of up to three objects, each containing:
      - etf_name
      - ticker
      - exchange
      - launch_date  (as stated in the answer; any format acceptable)
      - expense_ratio (as stated in the answer)
      - sources: all URLs in the answer that support that ETF's details or the "first three" context.
    If fewer than three such ETFs are mentioned, include only those present. If none are mentioned, return an empty array.
    Note: Extract only from the answer text; do not invent or infer ETFs or URLs not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _etf_identifier(etf: Optional[ETFItem]) -> str:
    """
    Build a human-readable identifier for the ETF for use in verification claims.
    Prefer ticker when available; otherwise fallback to name or a generic placeholder.
    """
    if not etf:
        return "the identified ETF"
    if etf.ticker and etf.ticker.strip():
        return f"the ETF with ticker {etf.ticker.strip()}"
    if etf.etf_name and etf.etf_name.strip():
        return f"the ETF named '{etf.etf_name.strip()}'"
    return "the identified ETF"


def _collect_all_sources(selected: Optional[ETFItem], first_three: List[ETFItem]) -> List[str]:
    """
    Combine and de-duplicate all sources from the selected ETF and the first-three list.
    """
    srcs = []
    if selected and selected.sources:
        srcs.extend([u for u in selected.sources if isinstance(u, str) and u.strip()])
    for item in first_three:
        if item and item.sources:
            srcs.extend([u for u in item.sources if isinstance(u, str) and u.strip()])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in srcs:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root_node, extracted: ETFExtraction) -> None:
    """
    Build the verification tree based on the rubric and run leaf verifications.
    """

    # Top-level Task node (critical, sequential)
    task_node = evaluator.add_sequential(
        id="task",
        desc="Identify the U.S. spot XRP ETF with the lowest expense ratio among the first three to launch in November 2025, and provide its ticker symbol, listing exchange, and launch date",
        parent=root_node,
        critical=True
    )

    # ETF_Selection (critical, sequential)
    etf_selection_node = evaluator.add_sequential(
        id="etf_selection",
        desc="Correctly identify the target ETF by applying all selection criteria",
        parent=task_node,
        critical=True
    )

    selected = extracted.selected_etf
    first_three = extracted.first_three_etfs or []

    # Existence & minimal sources check (critical custom node)
    has_id = selected is not None and (
        (selected.ticker and selected.ticker.strip()) or
        (selected.etf_name and selected.etf_name.strip())
    )
    has_sources = selected is not None and bool(selected.sources)
    evaluator.add_custom_node(
        result=bool(has_id and has_sources),
        id="selected_etf_provided",
        desc="A specific ETF is identified and at least one supporting URL is provided",
        parent=etf_selection_node,
        critical=True
    )

    # Launch_Criteria (critical, sequential)
    launch_criteria_node = evaluator.add_sequential(
        id="launch_criteria",
        desc="The identified ETF is a U.S. spot XRP ETF that launched in November 2025 and is among the first three such ETFs to launch that month",
        parent=etf_selection_node,
        critical=True
    )

    etf_ref = _etf_identifier(selected)
    selected_sources = selected.sources if (selected and selected.sources) else []
    all_sources = _collect_all_sources(selected, first_three)

    # 1) U.S. Spot XRP ETF check (leaf, critical)
    us_spot_leaf = evaluator.add_leaf(
        id="is_us_spot_xrp",
        desc="The identified ETF is a U.S. spot XRP ETF",
        parent=launch_criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{etf_ref} is a U.S.-listed spot XRP ETF (i.e., holds XRP directly rather than via futures or synthetic exposure).",
        node=us_spot_leaf,
        sources=selected_sources,
        additional_instruction=(
            "Confirm that the ETF is (a) U.S.-listed (e.g., on NYSE Arca, Nasdaq, or Cboe BZX) and "
            "(b) a spot cryptocurrency ETF specifically for XRP, meaning it holds XRP (not futures). "
            "If the provided URLs do not state both elements, mark as not supported."
        )
    )

    # 2) Launch month/year check (leaf, critical)
    nov2025_leaf = evaluator.add_leaf(
        id="launched_in_nov_2025",
        desc="The identified ETF launched in November 2025",
        parent=launch_criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{etf_ref} launched (inception/first trading date) in November 2025.",
        node=nov2025_leaf,
        sources=selected_sources,
        additional_instruction=(
            "Verify that the ETF's inception/launch/first trading date falls in November 2025. "
            "Accept synonyms like 'inception date', 'launch date', 'first trading date', or 'listing date'."
        )
    )

    # 3) Among first three to launch that month (leaf, critical)
    first_three_leaf = evaluator.add_leaf(
        id="among_first_three",
        desc="The identified ETF is among the first three U.S. spot XRP ETFs to launch in November 2025",
        parent=launch_criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{etf_ref} was among the first three U.S. spot XRP ETFs to launch in November 2025.",
        node=first_three_leaf,
        sources=all_sources,
        additional_instruction=(
            "Look for pages that explicitly state the ETF was among the 'first three' U.S. spot XRP ETFs launched in November 2025, "
            "or list and rank the first three where this ETF is included. "
            "If the provided URLs do not substantiate this status, mark as not supported."
        )
    )

    # Expense_Ratio_Selection (critical, sequential)
    expense_ratio_node = evaluator.add_sequential(
        id="expense_ratio_selection",
        desc="Among the first three U.S. spot XRP ETFs to launch in November 2025, the identified ETF has the lowest expense ratio",
        parent=launch_criteria_node,
        critical=True
    )

    # Lowest expense ratio among first three (leaf, critical)
    lowest_fee_leaf = evaluator.add_leaf(
        id="lowest_expense_ratio",
        desc="Among the first three U.S. spot XRP ETFs to launch in November 2025, the identified ETF has the lowest expense ratio",
        parent=expense_ratio_node,
        critical=True
    )
    fee_desc = selected.expense_ratio.strip() if (selected and selected.expense_ratio) else "the lowest fee"
    await evaluator.verify(
        claim=f"Among the first three U.S. spot XRP ETFs launched in November 2025, {etf_ref} has the lowest expense ratio (tie for lowest is acceptable).",
        node=lowest_fee_leaf,
        sources=all_sources,
        additional_instruction=(
            "Use the provided URLs to compare expense ratios among the first three U.S. spot XRP ETFs launched in November 2025. "
            "Confirm that the identified ETF's fee is the lowest or tied for the lowest among those first three. "
            "If the URLs do not provide a clear comparison among the first three, mark as not supported."
        )
    )

    # Details_Verification (critical, parallel)
    details_node = evaluator.add_parallel(
        id="details_verification",
        desc="All required details about the identified ETF are correctly provided",
        parent=expense_ratio_node,
        critical=True
    )

    # Ticker_Symbol (leaf, critical)
    ticker_leaf = evaluator.add_leaf(
        id="ticker_symbol",
        desc="The ticker symbol provided matches the identified ETF's official ticker",
        parent=details_node,
        critical=True
    )
    provided_ticker = (selected.ticker or "").strip() if selected else ""
    await evaluator.verify(
        claim=f"The official ticker symbol for {etf_ref} is '{provided_ticker}'.",
        node=ticker_leaf,
        sources=selected_sources,
        additional_instruction=(
            "Verify the ETF's official ticker on issuer pages, exchange listings, or official filings. "
            "Allow minor casing differences. If the provided URLs do not clearly state the ticker, mark as not supported."
        )
    )

    # Exchange (leaf, critical)
    exchange_leaf = evaluator.add_leaf(
        id="listing_exchange",
        desc="The listing exchange provided matches the identified ETF's primary exchange",
        parent=details_node,
        critical=True
    )
    provided_exchange = (selected.exchange or "").strip() if selected else ""
    await evaluator.verify(
        claim=f"The primary listing exchange for {etf_ref} is '{provided_exchange}'.",
        node=exchange_leaf,
        sources=selected_sources,
        additional_instruction=(
            "Verify the ETF's primary listing exchange (e.g., NYSE Arca, Nasdaq, Cboe BZX) on issuer pages or exchange listings. "
            "Accept reasonable naming variants (e.g., 'Cboe BZX' vs 'Cboe BZX Exchange'). "
            "If the provided URLs do not clearly state the exchange, mark as not supported."
        )
    )

    # Launch_Date (leaf, critical)
    launch_date_leaf = evaluator.add_leaf(
        id="launch_date",
        desc="The launch date provided matches the identified ETF's official inception date",
        parent=details_node,
        critical=True
    )
    provided_date = (selected.launch_date or "").strip() if selected else ""
    await evaluator.verify(
        claim=f"The official inception/launch/first trading date for {etf_ref} is '{provided_date}'.",
        node=launch_date_leaf,
        sources=selected_sources,
        additional_instruction=(
            "Confirm the exact date from issuer pages, prospectus/SEC filings, exchange notices, or reputable press releases. "
            "Accept synonyms like 'inception date', 'launch date', 'first trading date', or 'listing date' as long as they refer to the first trading/inception. "
            "If the provided URLs do not clearly state the date, mark as not supported."
        )
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
    Evaluate an answer for the XRP spot ETF (Nov 2025, lowest expense) task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
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
        default_model=model
    )

    # Extract selected ETF and first-three list from the answer
    selected_info = await evaluator.extract(
        prompt=prompt_extract_selected_etf(),
        template_class=ETFExtraction,
        extraction_name="selected_etf_extraction"
    )

    first_three_info = await evaluator.extract(
        prompt=prompt_extract_first_three_etfs(),
        template_class=ETFExtraction,
        extraction_name="first_three_etfs_extraction"
    )

    # Merge results: prefer selected_etf from the first extraction; combine first_three lists
    merged = ETFExtraction()
    merged.selected_etf = selected_info.selected_etf
    # Merge and truncate to at most 3
    merged.first_three_etfs = []
    if first_three_info.first_three_etfs:
        merged.first_three_etfs.extend(first_three_info.first_three_etfs[:3])

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, merged)

    # Return structured summary
    return evaluator.get_summary()