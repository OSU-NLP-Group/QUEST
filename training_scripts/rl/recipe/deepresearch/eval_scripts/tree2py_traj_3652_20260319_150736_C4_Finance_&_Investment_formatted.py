import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "xrp_etf_lowest_expense_nov2025"
TASK_DESCRIPTION = (
    "Among XRP spot ETFs that are headquartered in the United States and launched in November 2025, "
    "identify the one with the lowest stated annual expense ratio. Provide the ETF's ticker symbol, "
    "expense ratio (as a percentage), headquarters city and state, launch date, and a reference URL supporting your answer."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class XRPETFSelection(BaseModel):
    """
    Extracted details for the ETF that the answer claims has the lowest expense ratio
    among U.S.-headquartered XRP spot ETFs launched in November 2025.
    """
    name: Optional[str] = None
    ticker: Optional[str] = None
    expense_ratio: Optional[str] = None  # Keep as a string like "0.25%" to maximize robustness
    headquarters_city: Optional[str] = None
    headquarters_state: Optional[str] = None
    headquarters_country: Optional[str] = None
    launch_date: Optional[str] = None  # Keep as-is (e.g., "2025-11-12" or "Nov 12, 2025")
    reference_urls: List[str] = Field(default_factory=list)   # URLs that support the specific ETF details
    comparison_urls: List[str] = Field(default_factory=list)  # URLs that support the "lowest expense ratio" claim


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_xrp_etf_selection() -> str:
    return """
    Extract the single ETF that the answer identifies as the U.S.-headquartered XRP spot ETF with the lowest expense ratio among those launched in November 2025.

    Return a JSON object with the following fields (use null when not provided):
    - name: The ETF or fund name as written in the answer.
    - ticker: The ETF ticker symbol.
    - expense_ratio: The stated annual expense ratio as written (e.g., "0.25%" or "0.25 %"). Do not convert to a number; keep the original formatting.
    - headquarters_city: The headquarters city provided in the answer for the ETF issuer/administrator/sponsor.
    - headquarters_state: The headquarters state provided in the answer for the ETF issuer/administrator/sponsor.
    - headquarters_country: The headquarters country provided (should typically be "United States" for U.S.-headquartered).
    - launch_date: The ETF launch (listing) date as written in the answer. Keep the original format (e.g., "2025-11-12" or "November 12, 2025" or "Nov 2025").
    - reference_urls: An array of all URLs explicitly cited in the answer that directly support the ETF’s own details (ticker, expense ratio, headquarters, launch date).
    - comparison_urls: An array of all URLs explicitly cited in the answer that support the claim that this ETF has the lowest expense ratio among U.S.-headquartered XRP spot ETFs launched in November 2025 (e.g., roundups, comparison tables, or credible news/press releases that explicitly make such a claim).

    Important rules:
    - Only extract URLs that are explicitly present in the answer (plain URLs or markdown links). Do not invent URLs.
    - If a URL is missing a protocol, prepend "http://".
    - If the answer provides multiple ETFs, select only the one that the answer claims is the lowest-expense-choice for this task. If the answer doesn’t clearly identify one ETF, extract the first ETF the answer ultimately recommends or names as lowest.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    cleaned: List[str] = []
    for u in urls:
        if not u or not isinstance(u, str):
            continue
        u2 = u.strip()
        if not u2:
            continue
        if u2 not in seen:
            seen.add(u2)
            cleaned.append(u2)
    return cleaned


def _coalesce(val: Optional[str], fallback: str = ""):  # small helper to avoid rendering "None"
    return val if isinstance(val, str) and val.strip() else fallback


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_selected_etf(evaluator: Evaluator, parent_node, etf: XRPETFSelection) -> None:
    """
    Build the verification leaves as specified by the rubric and run verifications.
    All leaves here are critical under a critical root, matching the rubric.
    """
    # Prepare sources
    ref_urls = _dedup_urls(etf.reference_urls or [])
    cmp_urls = _dedup_urls(etf.comparison_urls or [])
    all_urls = _dedup_urls(ref_urls + cmp_urls)

    # Convenience variables for claims
    name = _coalesce(etf.name)
    ticker = _coalesce(etf.ticker)
    expense = _coalesce(etf.expense_ratio)
    hq_city = _coalesce(etf.headquarters_city)
    hq_state = _coalesce(etf.headquarters_state)
    hq_country = _coalesce(etf.headquarters_country)
    launch_date = _coalesce(etf.launch_date)

    # Node: us_headquarters
    if all_urls:
        n_us = evaluator.add_leaf(
            id="us_headquarters",
            desc="The identified ETF is headquartered in the United States",
            parent=parent_node,
            critical=True,
        )
        claim_us = (
            f"The ETF's issuer/administrator/sponsor is headquartered in the United States."
            + (f" The answer lists headquarters as {hq_city}, {hq_state}, {hq_country}." if (hq_city or hq_state or hq_country) else "")
        )
        await evaluator.verify(
            claim=claim_us,
            node=n_us,
            sources=all_urls,
            additional_instruction=(
                "Verify from the provided pages whether the ETF's issuer/administrator/sponsor is U.S.-headquartered. "
                "Allow that some sources may mention only the country while others mention city/state."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="us_headquarters",
            desc="The identified ETF is headquartered in the United States (failed due to missing sources)",
            parent=parent_node,
            critical=True,
        )

    # Node: lowest_expense_ratio
    if all_urls:
        n_lowest = evaluator.add_leaf(
            id="lowest_expense_ratio",
            desc="The identified ETF has the lowest stated expense ratio among all U.S.-headquartered XRP spot ETFs that launched in November 2025",
            parent=parent_node,
            critical=True,
        )
        claim_lowest = (
            f"Among U.S.-headquartered XRP spot ETFs that launched in November 2025, "
            f"the ETF with ticker '{ticker}' has the lowest stated annual expense ratio of {expense}."
        )
        await evaluator.verify(
            claim=claim_lowest,
            node=n_lowest,
            sources=all_urls,
            additional_instruction=(
                "Use only the provided URLs. Accept if a page explicitly states this ETF has the lowest expense ratio "
                "in that cohort or presents a comparison (e.g., table or roundup) that clearly shows this ETF's expense ratio "
                "is the lowest. Consider only U.S.-headquartered XRP spot ETFs launched in November 2025."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="lowest_expense_ratio",
            desc="The identified ETF has the lowest stated expense ratio among the specified cohort (failed due to missing sources)",
            parent=parent_node,
            critical=True,
        )

    # Node: november_2025_launch
    if all_urls:
        n_nov = evaluator.add_leaf(
            id="november_2025_launch",
            desc="The identified ETF launched in November 2025",
            parent=parent_node,
            critical=True,
        )
        claim_nov = "The ETF was launched (listed) in November 2025."
        await evaluator.verify(
            claim=claim_nov,
            node=n_nov,
            sources=all_urls,
            additional_instruction=(
                "Confirm the month and year of the ETF's launch/listing from the provided pages. "
                "The month must be November and the year 2025."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="november_2025_launch",
            desc="The identified ETF launched in November 2025 (failed due to missing sources)",
            parent=parent_node,
            critical=True,
        )

    # Node: correct_ticker
    if all_urls:
        n_ticker = evaluator.add_leaf(
            id="correct_ticker",
            desc="The provided ticker symbol is correct for the identified ETF",
            parent=parent_node,
            critical=True,
        )
        claim_ticker = f"The ETF's ticker symbol is '{ticker}'."
        await evaluator.verify(
            claim=claim_ticker,
            node=n_ticker,
            sources=all_urls,
            additional_instruction=(
                "Verify that the referenced pages identify the ETF with this exact or equivalent ticker symbol."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="correct_ticker",
            desc="The provided ticker symbol is correct for the identified ETF (failed due to missing sources)",
            parent=parent_node,
            critical=True,
        )

    # Node: correct_expense_ratio
    if all_urls:
        n_exp = evaluator.add_leaf(
            id="correct_expense_ratio",
            desc="The provided expense ratio matches the ETF's stated annual expense ratio",
            parent=parent_node,
            critical=True,
        )
        claim_exp = f"The ETF's stated annual expense ratio is {expense}."
        await evaluator.verify(
            claim=claim_exp,
            node=n_exp,
            sources=all_urls,
            additional_instruction=(
                "Check the referenced pages for the ETF's stated expense ratio. "
                "Allow small formatting variations like spaces before % or wording such as 'expense ratio' vs 'management fee'."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="correct_expense_ratio",
            desc="The provided expense ratio matches the ETF's stated annual expense ratio (failed due to missing sources)",
            parent=parent_node,
            critical=True,
        )

    # Node: correct_headquarters
    if all_urls:
        n_hq = evaluator.add_leaf(
            id="correct_headquarters",
            desc="The provided headquarters city and state are correct for the identified ETF",
            parent=parent_node,
            critical=True,
        )
        claim_hq = (
            f"The ETF issuer/administrator/sponsor is headquartered in {hq_city}, {hq_state}, United States."
        )
        await evaluator.verify(
            claim=claim_hq,
            node=n_hq,
            sources=all_urls,
            additional_instruction=(
                "Verify that the provided city and state match what is shown in the sources. "
                "If multiple offices are listed, use the primary corporate headquarters."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="correct_headquarters",
            desc="The provided headquarters city and state are correct for the identified ETF (failed due to missing sources)",
            parent=parent_node,
            critical=True,
        )

    # Node: correct_launch_date
    if all_urls:
        n_ld = evaluator.add_leaf(
            id="correct_launch_date",
            desc="The provided launch date is correct for the identified ETF",
            parent=parent_node,
            critical=True,
        )
        claim_ld = f"The ETF was launched (listed) on {launch_date}."
        await evaluator.verify(
            claim=claim_ld,
            node=n_ld,
            sources=all_urls,
            additional_instruction=(
                "Confirm the ETF's launch/listing date from the provided pages. "
                "Allow reasonable date formatting variants (e.g., '2025-11-12', 'November 12, 2025', etc.)."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="correct_launch_date",
            desc="The provided launch date is correct for the identified ETF (failed due to missing sources)",
            parent=parent_node,
            critical=True,
        )

    # Node: valid_reference
    if all_urls:
        n_ref = evaluator.add_leaf(
            id="valid_reference",
            desc="A valid reference URL is provided that supports the identification and details of the ETF",
            parent=parent_node,
            critical=True,
        )
        claim_ref = (
            "At least one of the provided URLs is a valid, relevant source about the identified ETF and supports its key details "
            "(for example, the ticker symbol and/or the expense ratio and/or the launch date)."
        )
        await evaluator.verify(
            claim=claim_ref,
            node=n_ref,
            sources=all_urls,
            additional_instruction=(
                "Pass this verification if at least one URL is a credible page about the ETF that clearly mentions its identity "
                "and at least some key details (e.g., ticker, expense ratio, or launch date)."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="valid_reference",
            desc="A valid reference URL is provided that supports the identification and details of the ETF (failed due to missing sources)",
            parent=parent_node,
            critical=True,
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the XRP ETF lowest expense ratio (Nov 2025) task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates independent checks
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

    # Make the root critical as per rubric (all children must be critical too)
    root.critical = True

    # Extract the selected ETF information from the answer
    selection = await evaluator.extract(
        prompt=prompt_extract_xrp_etf_selection(),
        template_class=XRPETFSelection,
        extraction_name="selected_etf",
    )

    # Build leaves and run verifications as per rubric
    await verify_selected_etf(evaluator, root, selection or XRPETFSelection())

    # Return summary
    return evaluator.get_summary()