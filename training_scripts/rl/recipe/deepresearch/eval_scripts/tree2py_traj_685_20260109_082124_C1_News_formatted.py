import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "official_who_withdrawal_date"
TASK_DESCRIPTION = (
    "On January 20, 2025, President Trump signed an executive order announcing the United States' withdrawal from the World Health Organization. "
    "According to the United Nations confirmation, on what specific date will the United States officially leave the WHO?"
)
EXPECTED_OFFICIAL_DATE = "January 20, 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class WHOWithdrawalExtraction(BaseModel):
    """
    Structured extraction from the answer text.
    """
    official_withdrawal_date: Optional[str] = None
    mentions_un_confirmation: Optional[bool] = None
    un_confirmation_quote: Optional[str] = None
    cited_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_who_withdrawal_info() -> str:
    return """
    Extract from the answer:
    - official_withdrawal_date: The specific date stated in the answer for when the U.S. will officially leave the WHO. Return the date exactly as written in the answer (e.g., "January 20, 2026", "20 January 2026", "2026-01-20"). If not stated, return null.
    - mentions_un_confirmation: Return true if the answer explicitly attributes the official withdrawal date to confirmation by the United Nations (e.g., "according to the United Nations", "per UN confirmation", "the UN confirmed"). Accept synonyms like "UN", "U.N.", "United Nations Secretariat". Otherwise return false. If unclear, return false.
    - un_confirmation_quote: If mentions_un_confirmation is true, return the exact phrase or sentence from the answer that attributes the date to the UN; otherwise return null.
    - cited_urls: Extract all URLs explicitly mentioned in the answer text (if any). Only include valid URLs.

    Do not invent or infer information not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extraction: WHOWithdrawalExtraction
) -> None:
    """
    Build the verification tree according to the rubric and run the checks.
    """
    # Create the rubric root node (as a child of the evaluator's root)
    official_node = evaluator.add_parallel(
        id="Official_WHO_Withdrawal_Date",
        desc="Evaluates whether the answer identifies the official date the U.S. will leave the WHO, consistent with the stated one-year notice constraint and framed as the UN-confirmed date (per the question).",
        parent=evaluator.root,
        critical=True
    )

    # Leaf 1: Must state the official withdrawal date as January 20, 2026
    date_leaf = evaluator.add_leaf(
        id="Withdrawal_Date_Is_One_Year_After_Notice",
        desc="States the official withdrawal date as January 20, 2026 (one year after the January 20, 2025 notice/signing date, per the one-year notice constraint).",
        parent=official_node,
        critical=True
    )

    # Leaf 2: Must explicitly reference UN confirmation for that date
    un_leaf = evaluator.add_leaf(
        id="UN_Confirmation_Referenced",
        desc="Explicitly indicates that the stated official withdrawal date is according to (or confirmed by) the United Nations, matching the question prompt.",
        parent=official_node,
        critical=True
    )

    # Prepare claims and verify in parallel
    date_claim = (
        "Within the answer, the official withdrawal date the United States will leave the WHO is stated as January 20, 2026 "
        "(or an equivalent representation of that exact calendar date)."
    )
    date_additional_instruction = (
        "Focus only on what the answer states. The date must be explicitly given as January 20, 2026, allowing equivalent representations such as "
        "'Jan 20, 2026', 'January 20th, 2026', '20 January 2026', '2026-01-20', or '1/20/2026'. "
        "Do NOT accept vague phrasing like 'January 2026' or 'one year later' without the specific day. "
        "Ensure the date refers to the official exit date (one-year notice culmination), not just the signing/notice date."
    )

    un_claim = (
        "Within the answer, the stated official withdrawal date is explicitly attributed to confirmation by the United Nations "
        "(e.g., 'according to the United Nations', 'per UN confirmation', 'the UN confirmed')."
    )
    un_additional_instruction = (
        "Check that the attribution to the United Nations is explicit and connected to the official withdrawal date. "
        "Accept synonyms/variants such as 'United Nations', 'UN', 'U.N.', 'United Nations Secretariat'. "
        "Mere mention of the WHO or other organizations does not satisfy this. The answer need not include a URL."
    )

    await evaluator.batch_verify([
        (date_claim, None, date_leaf, date_additional_instruction),
        (un_claim, None, un_leaf, un_additional_instruction),
    ])


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
    Evaluate an answer for the official WHO withdrawal date (UN-confirmed) task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
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
        default_model=model
    )

    # Extract structured info from the answer (recorded for transparency; verification relies on answer text)
    extraction = await evaluator.extract(
        prompt=prompt_extract_who_withdrawal_info(),
        template_class=WHOWithdrawalExtraction,
        extraction_name="who_withdrawal_info"
    )

    # Optional: record ground truth for transparency
    evaluator.add_ground_truth({
        "expected_official_withdrawal_date": EXPECTED_OFFICIAL_DATE,
        "required_authority_reference": "United Nations (explicit confirmation mentioned in the answer)"
    }, gt_type="ground_truth")

    # Build tree and run verifications
    await build_and_verify_tree(evaluator, extraction)

    # Return structured result
    return evaluator.get_summary()