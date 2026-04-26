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
TASK_ID = "glacier_group_lottery_2025"
TASK_DESCRIPTION = (
    "A group of 6 hikers is planning a wilderness camping trip to Glacier National Park in summer 2025 and "
    "wants to apply through the standard group lottery system for advance reservations. "
    "What is the specific date of the lottery application window, what are the start and end dates of the "
    "booking period for lottery winners, and what is the lottery application fee?"
)

EXPECTED_VALUES = {
    "application_window_date": "March 15, 2025",
    "booking_start_date": "March 21, 2025",
    "booking_end_date": "April 30, 2025",
    "application_fee_amount": "$10",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class GlacierLotteryExtraction(BaseModel):
    """
    Structured extraction of the Glacier NP standard group wilderness permit lottery info from the answer.
    """
    application_window_date: Optional[str] = None
    application_sources: List[str] = Field(default_factory=list)

    booking_start_date: Optional[str] = None
    booking_end_date: Optional[str] = None
    booking_sources: List[str] = Field(default_factory=list)

    application_fee_amount: Optional[str] = None
    fee_sources: List[str] = Field(default_factory=list)

    # Fallback / general sources if the answer lists sources only once
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_glacier_lottery_info() -> str:
    return """
    Extract the specific information stated in the answer about Glacier National Park's STANDARD GROUP wilderness camping permit lottery for the 2025 season.

    Return a JSON object with the following fields:
    1. application_window_date: The single date when the standard group lottery application window opens (as stated in the answer). If a range is given, extract the open date. If missing, return null.
    2. application_sources: All URLs explicitly cited in the answer that support the application window date. If none, return an empty list.

    3. booking_start_date: The date when lottery winners can BEGIN making reservations (as stated). If missing, return null.
    4. booking_end_date: The deadline date by which lottery winners MUST complete their reservations (as stated). If missing, return null.
    5. booking_sources: All URLs explicitly cited that support the booking start/end dates. If none, return an empty list.

    6. application_fee_amount: The lottery APPLICATION fee amount (as stated). If the answer mentions permit fees or other fees, only extract the application fee amount. If missing, return null.
    7. fee_sources: All URLs explicitly cited that support the application fee amount. If none, return an empty list.

    8. general_sources: If the answer provides a single sources section or general references applicable to the entire lottery (dates and fee), list those URLs here.

    SPECIAL SOURCE RULES:
    - Extract only actual URLs mentioned in the answer (including markdown links). Do not invent URLs.
    - Include full URLs; prepend http:// if protocol is missing.
    - If the answer references a site (e.g., "NPS website") without a URL, do not add it.

    Be faithful to the wording in the answer; do not normalize or rewrite values beyond extracting them as strings.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _pick_sources(primary: List[str], fallback: List[str]) -> Optional[List[str]]:
    """
    Prefer primary sources if present; otherwise use fallback. If both empty, return None.
    """
    if primary and len(primary) > 0:
        return primary
    if fallback and len(fallback) > 0:
        return fallback
    return None


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root_node, ext: GlacierLotteryExtraction) -> None:
    """
    Build the verification tree and perform verifications according to the rubric.
    """

    # 1) Lottery application date (Critical leaf)
    app_sources = _pick_sources(ext.application_sources, ext.general_sources)
    app_leaf = evaluator.add_leaf(
        id="lottery_application_date",
        desc="States the correct date of the standard group lottery application window (March 15, 2025)",
        parent=root_node,
        critical=True,
    )
    app_claim = (
        f"In the answer, the stated date for Glacier National Park's STANDARD GROUP wilderness camping permit "
        f"lottery application window for 2025 is '{ext.application_window_date}'. "
        f"This should match the official date March 15, 2025."
    )
    await evaluator.verify(
        claim=app_claim,
        node=app_leaf,
        sources=app_sources,
        additional_instruction=(
            "Focus on the STANDARD GROUP lottery for Glacier National Park wilderness camping in 2025. "
            "Confirm the answer's stated date matches the official opening date March 15, 2025. "
            "Allow minor formatting differences (e.g., abbreviations, casing). "
            "If the answer omits the date or states a different date, mark as incorrect."
        ),
    )

    # 2) Winner booking period (Critical parallel node with two critical leaves)
    booking_node = evaluator.add_parallel(
        id="winner_booking_period",
        desc="States the correct booking period dates for lottery winners",
        parent=root_node,
        critical=True,
    )
    booking_sources = _pick_sources(ext.booking_sources, ext.general_sources)

    # 2a) Booking start date (Critical leaf)
    start_leaf = evaluator.add_leaf(
        id="booking_start_date",
        desc="States the correct start date when lottery winners can begin making reservations (March 21, 2025)",
        parent=booking_node,
        critical=True,
    )
    start_claim = (
        f"In the answer, the stated START date when lottery winners can begin making reservations is "
        f"'{ext.booking_start_date}', which should match the official start date March 21, 2025."
    )
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=booking_sources,
        additional_instruction=(
            "Verify the START date for reservations for winners of the STANDARD GROUP lottery in 2025 is March 21, 2025. "
            "If the answer lists a different date or omits it, it is incorrect."
        ),
    )

    # 2b) Booking end date (Critical leaf)
    end_leaf = evaluator.add_leaf(
        id="booking_end_date",
        desc="States the correct end date by which lottery winners must complete their reservations (April 30, 2025)",
        parent=booking_node,
        critical=True,
    )
    end_claim = (
        f"In the answer, the stated END deadline for lottery winners to complete reservations is "
        f"'{ext.booking_end_date}', which should match the official end date April 30, 2025."
    )
    await evaluator.verify(
        claim=end_claim,
        node=end_leaf,
        sources=booking_sources,
        additional_instruction=(
            "Verify the END deadline for completing reservations for winners of the STANDARD GROUP lottery in 2025 "
            "is April 30, 2025. If the answer lists a different date or omits it, it is incorrect."
        ),
    )

    # 3) Lottery application fee (Critical leaf)
    fee_sources = _pick_sources(ext.fee_sources, ext.general_sources)
    fee_leaf = evaluator.add_leaf(
        id="lottery_application_fee",
        desc="States the correct lottery application fee amount ($10)",
        parent=root_node,
        critical=True,
    )
    fee_claim = (
        f"In the answer, the stated lottery APPLICATION fee amount is '{ext.application_fee_amount}', "
        f"which should match the official application fee of $10."
    )
    await evaluator.verify(
        claim=fee_claim,
        node=fee_leaf,
        sources=fee_sources,
        additional_instruction=(
            "Confirm the fee is specifically the lottery APPLICATION fee for the STANDARD GROUP lottery and is $10. "
            "Do not confuse with permit issuance fees or other charges. Allow minor format variants like '10 USD' or 'USD $10'."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Glacier NP 2025 standard group wilderness camping permit lottery details.
    """
    # Initialize evaluator with root strategy matching the rubric (parallel)
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_glacier_lottery_info(),
        template_class=GlacierLotteryExtraction,
        extraction_name="glacier_lottery_info",
    )

    # Record ground truth for transparency
    evaluator.add_ground_truth(
        {
            "expected_application_window_date": EXPECTED_VALUES["application_window_date"],
            "expected_booking_start_date": EXPECTED_VALUES["booking_start_date"],
            "expected_booking_end_date": EXPECTED_VALUES["booking_end_date"],
            "expected_application_fee_amount": EXPECTED_VALUES["application_fee_amount"],
        },
        gt_type="expected_values",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extraction)

    # Return evaluation summary
    return evaluator.get_summary()