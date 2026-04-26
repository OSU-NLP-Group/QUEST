import asyncio
import logging
from typing import Any, Optional, Dict

from pydantic import BaseModel

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bruno_age_dwts_s33"
TASK_DESCRIPTION = """
How old was judge Bruno Tonioli on the day of the Dancing with the Stars Season 33 finale?
"""

GROUND_TRUTH = {
    "finale_date": "November 26, 2024",
    "birth_date": "November 25, 1955",
    "expected_age_on_finale": "69"
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BrunoAgeExtraction(BaseModel):
    """
    Information explicitly stated in the answer text regarding Bruno Tonioli's age,
    relevant dates, and any reasoning provided.
    """
    claimed_age_text: Optional[str] = None
    finale_date_mentioned: Optional[str] = None
    birth_date_mentioned: Optional[str] = None
    reasoning_excerpt: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_bruno_age_info() -> str:
    return """
    From the provided answer text, extract the following fields exactly as they appear (do not compute or infer):
    - claimed_age_text: The age explicitly stated for Bruno Tonioli on the Season 33 finale date (e.g., "69", "69 years old"). If not explicitly stated, return null.
    - finale_date_mentioned: The finale date mentioned in the answer (e.g., "November 26, 2024", "Nov 26, 2024", "11/26/2024"). If the answer mentions multiple dates, choose the one clearly tied to the DWTS Season 33 finale. If none, return null.
    - birth_date_mentioned: Bruno Tonioli's birth date if it is mentioned in the answer (e.g., "November 25, 1955", "25 November 1955"). If not mentioned, return null.
    - reasoning_excerpt: A short excerpt (one sentence is fine) of any reasoning that explains why he was 69 on the finale date (e.g., "He turned 69 on Nov 25, 2024, so he was 69 on Nov 26, 2024."). If no such reasoning is provided, return null.

    Important:
    - Extract only what the answer text explicitly states; do not add or infer any information.
    - Keep the extracted values as strings; do not convert formats.
    """


# --------------------------------------------------------------------------- #
# Verification building                                                       #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extraction: BrunoAgeExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and perform the checks.
    """

    # Create the main evaluation node under root (parallel aggregation, non-critical to allow partial credit)
    main_node = evaluator.add_parallel(
        id="Bruno_Tonioli_Age_On_DWTS_S33_Finale",
        desc="Evaluate whether the response correctly states Bruno Tonioli's age on the DWTS Season 33 finale date, consistent with the provided constraints.",
        parent=evaluator.root,
        critical=False  # Allow non-critical children as per rubric
    )

    # 1) Age on Finale Date: Critical
    age_leaf = evaluator.add_leaf(
        id="Age_On_Finale_Date",
        desc="State Bruno Tonioli's age on the Season 33 finale date (Nov 26, 2024) as 69 years old.",
        parent=main_node,
        critical=True
    )
    age_claim = "In the answer, Bruno Tonioli is said to be 69 years old on November 26, 2024 (the Dancing with the Stars Season 33 finale date)."
    await evaluator.verify(
        claim=age_claim,
        node=age_leaf,
        additional_instruction=(
            "Judge only based on the provided answer text. Accept clear equivalents like '69' or '69 years old' tied to the finale date. "
            "If the answer states a different age or does not state his age for the finale date, mark incorrect."
        ),
    )

    # 2) Finale Date Matches Constraint: Non-critical
    finale_leaf = evaluator.add_leaf(
        id="Finale_Date_Matches_Constraint",
        desc="Uses/mentions the Season 33 finale date as November 26, 2024.",
        parent=main_node,
        critical=False
    )
    finale_claim = (
        "The answer uses or mentions the Season 33 finale date as November 26, 2024."
    )
    await evaluator.verify(
        claim=finale_claim,
        node=finale_leaf,
        additional_instruction=(
            "Check if the answer explicitly references the finale date as November 26, 2024. "
            "Minor format variations like 'Nov 26, 2024', '11/26/2024', or 'November 26th, 2024' are acceptable. "
            "If it uses a different date or no date for the finale, mark incorrect."
        ),
    )

    # 3) Birthdate Matches Constraint: Non-critical
    birth_leaf = evaluator.add_leaf(
        id="Bruno_Birthdate_Matches_Constraint",
        desc="Uses/mentions Bruno Tonioli's birth date as November 25, 1955.",
        parent=main_node,
        critical=False
    )
    birth_claim = (
        "The answer mentions Bruno Tonioli's birth date as November 25, 1955."
    )
    await evaluator.verify(
        claim=birth_claim,
        node=birth_leaf,
        additional_instruction=(
            "Check whether the answer explicitly mentions his birth date as November 25, 1955. "
            "Accept equivalent formats like '25 November 1955'. If a different date is stated or it is not mentioned, mark incorrect."
        ),
    )

    # 4) Reasoning Explains Why 69: Non-critical and conditional
    reasoning_leaf = evaluator.add_leaf(
        id="Reasoning_Explains_Why_69",
        desc="If reasoning is provided, it correctly explains that he turned 69 on Nov 25, 2024 (the day before the finale), so he was 69 on Nov 26, 2024.",
        parent=main_node,
        critical=False
    )
    if extraction.reasoning_excerpt and extraction.reasoning_excerpt.strip():
        reasoning_claim = (
            "The answer's reasoning correctly explains that Bruno Tonioli turned 69 on November 25, 2024 (the day before the finale), "
            "so he was 69 on November 26, 2024."
        )
        await evaluator.verify(
            claim=reasoning_claim,
            node=reasoning_leaf,
            additional_instruction=(
                "Evaluate the correctness of the reasoning provided in the answer. "
                "It should connect turning 69 on Nov 25, 2024 to being 69 on Nov 26, 2024 (finale day). "
                "If no reasoning is present, this check should be considered skipped."
            ),
        )
    else:
        # No reasoning provided; mark this check as skipped
        reasoning_leaf.score = 0.0
        reasoning_leaf.status = "skipped"


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
    Evaluate an answer for Bruno Tonioli's age on the DWTS Season 33 finale date.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel aggregation
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
        prompt=prompt_extract_bruno_age_info(),
        template_class=BrunoAgeExtraction,
        extraction_name="bruno_age_extraction",
    )

    # Add ground truth constraints info
    evaluator.add_ground_truth({
        "expected_age_on_finale": GROUND_TRUTH["expected_age_on_finale"],
        "finale_date": GROUND_TRUTH["finale_date"],
        "birth_date": GROUND_TRUTH["birth_date"],
    }, gt_type="constraints")

    # Build verification tree and perform checks
    await build_verification_tree(evaluator, extraction)

    # Return the evaluation summary
    return evaluator.get_summary()