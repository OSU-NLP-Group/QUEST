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
TASK_ID = "boston_marathon_2026_men_18_34_qualifying_time"
TASK_DESCRIPTION = "What is the official qualifying time for men aged 18-34 to be eligible for the 2026 Boston Marathon?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class QualifyingExtraction(BaseModel):
    """
    Extracted information from the agent's answer:
    - men_18_34_time: The time string stated for men aged 18–34 (e.g., '2:55:00', '2h 55m').
    - age_determination_date: The specific date used for age determination, if stated (e.g., 'April 20, 2026').
    - chip_time_basis_statement: A phrase/sentence indicating qualifying times are based on official net/chip time.
    - gender_context_statement: A phrase/sentence acknowledging standards vary by gender and that this time is for men.
    - source_urls: All URLs cited in the answer.
    """
    men_18_34_time: Optional[str] = None
    age_determination_date: Optional[str] = None
    chip_time_basis_statement: Optional[str] = None
    gender_context_statement: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_qualifying_info() -> str:
    return (
        "From the answer, extract the following fields exactly as stated:\n"
        "1) men_18_34_time: The qualifying time stated for men aged 18–34 for the 2026 Boston Marathon. "
        "Return the exact time string (e.g., '2:55:00', '2:55', '2h 55m'). If multiple times are present, "
        "choose the one explicitly tied to men aged 18–34. If not stated, return null.\n"
        "2) age_determination_date: The specific date used to determine the runner’s age for qualifying standards "
        "(e.g., 'April 20, 2026'). If the answer only says 'age on race day' without giving the date, return null.\n"
        "3) chip_time_basis_statement: Extract the exact phrase or sentence indicating that qualifying times are based "
        "on official net time (chip time). If not mentioned, return null.\n"
        "4) gender_context_statement: Extract any phrase/sentence acknowledging that standards vary by gender and that "
        "the provided time corresponds to the men’s category. If not mentioned, return null.\n"
        "5) source_urls: Extract all URLs present in the answer text (including markdown links). Return a list of URLs. "
        "If none are present, return an empty list.\n"
        "Return a single JSON object containing these fields."
    )


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree_and_verify(evaluator: Evaluator, extraction: QualifyingExtraction) -> None:
    """
    Build the verification tree according to the rubric and perform verifications.
    The rubric (converted from JSON) contains one main parallel node and four leaf checks.
    """

    # Parent node from rubric (set non-critical to allow the non-critical child)
    parent_node = evaluator.add_parallel(
        id="Provide_Boston_Marathon_2026_Qualifying_Time",
        desc="Provide the official qualifying time for men aged 18-34 to be eligible for the 2026 Boston Marathon, consistent with the given constraints.",
        parent=evaluator.root,
        critical=False
    )

    # Leaf 1: Correct qualifying time stated as 2:55:00 (critical)
    leaf_time = evaluator.add_leaf(
        id="Correct_Qualifying_Time",
        desc="States the qualifying standard time for men aged 18-34 as 2:55:00.",
        parent=parent_node,
        critical=True
    )
    claim_time = (
        "The answer explicitly states that the qualifying standard time for men aged 18–34 "
        "for the 2026 Boston Marathon is 2:55:00."
    )
    await evaluator.verify(
        claim=claim_time,
        node=leaf_time,
        additional_instruction=(
            "Check the answer text to see if it clearly presents the men's 18–34 qualifying time as 2:55:00. "
            "Allow minor formatting variants like '2:55' or '2h 55m' if it unambiguously denotes 2 hours 55 minutes."
        )
    )

    # Leaf 2: Age determination date (critical)
    leaf_age_date = evaluator.add_leaf(
        id="Age_Determination_Date",
        desc="Indicates the qualifying standard is based on the runner's age on April 20, 2026 (race day).",
        parent=parent_node,
        critical=True
    )
    claim_age_date = (
        "The answer indicates that the qualifying standard is based on the runner's age on April 20, 2026 (race day)."
    )
    await evaluator.verify(
        claim=claim_age_date,
        node=leaf_age_date,
        additional_instruction=(
            "Verify the answer text mentions that age is determined on race day, and that it specifies the date "
            "as April 20, 2026. Accept reasonable phrasing variations as long as the meaning is clear."
        )
    )

    # Leaf 3: Net chip time basis (critical)
    leaf_chip_time = evaluator.add_leaf(
        id="Net_Chip_Time_Basis",
        desc="Indicates qualifying times are based on official net time (chip time).",
        parent=parent_node,
        critical=True
    )
    claim_chip_time = "The answer indicates qualifying times are based on official net time (chip time), not gun time."
    await evaluator.verify(
        claim=claim_chip_time,
        node=leaf_chip_time,
        additional_instruction=(
            "Look for language such as 'net time', 'chip time', or 'official net time'. The answer should clearly "
            "indicate that qualifying standards use net/chip time rather than gun time."
        )
    )

    # Leaf 4: Gender category context (non-critical)
    leaf_gender_ctx = evaluator.add_leaf(
        id="Gender_Category_Context",
        desc="Acknowledges that qualifying standards vary by gender category and that the provided time corresponds to the men's category.",
        parent=parent_node,
        critical=False
    )
    claim_gender_ctx = (
        "The answer acknowledges that qualifying standards vary by gender category and that the stated time "
        "corresponds to the men's category."
    )
    await evaluator.verify(
        claim=claim_gender_ctx,
        node=leaf_gender_ctx,
        additional_instruction=(
            "Confirm that the answer makes clear the time applies to men and notes that qualifying standards vary by gender."
        )
    )

    # Optional: add ground truth info to the summary for transparency (not used for scoring)
    evaluator.add_ground_truth({
        "expected_men_18_34_time": "2:55:00",
        "expected_age_determination_date": "April 20, 2026 (race day)",
        "expected_time_basis": "Official net time (chip time)",
        "note": "This ground truth is provided for context only; scoring is driven by verification of the answer content."
    })


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
    Evaluate an answer for the Boston Marathon 2026 qualifying time task.
    """
    # Initialize evaluator with a parallel root
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_qualifying_info(),
        template_class=QualifyingExtraction,
        extraction_name="extracted_qualifying_info"
    )

    # Build verification tree and perform checks
    await build_verification_tree_and_verify(evaluator, extraction)

    # Return the final summary
    return evaluator.get_summary()