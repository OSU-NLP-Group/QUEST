import asyncio
import logging
from typing import Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pmp_education_requirement"
TASK_DESCRIPTION = """
What is the minimum number of formal project management education hours required to be eligible for the PMP (Project Management Professional) certification?
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PMPEducationExtraction(BaseModel):
    """
    Extract key statements related to PMP formal project management education-hour eligibility
    from the answer text.
    """
    min_hours_value: Optional[str] = Field(
        default=None,
        description="The number of hours stated as the minimum formal project management education/training (e.g., '35')."
    )
    min_hours_phrase: Optional[str] = Field(
        default=None,
        description="The exact phrase from the answer that mentions the minimum education hours (e.g., '35 contact hours')."
    )
    states_applies_regardless_background: Optional[bool] = Field(
        default=None,
        description="True if the answer states the 35-hour requirement applies regardless of holding a four-year degree or a secondary diploma; False if it explicitly contradicts; null if not stated."
    )
    states_must_complete_before_exam: Optional[bool] = Field(
        default=None,
        description="True if the answer states the education hours must be completed before taking/scheduling the PMP exam; False if explicitly contradicts; null if not stated."
    )


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pmp_education() -> str:
    return """
    Extract the key elements related to PMP formal project management education-hour eligibility as stated in the answer.

    Required fields:
    1) min_hours_value: The number of hours mentioned as the minimum formal project management education/training. Return only the numeric part as a string if possible (e.g., "35"). If the answer does not mention any number, return null.
    2) min_hours_phrase: The exact phrase or short snippet from the answer that mentions the minimum hours (e.g., "35 contact hours", "at least 35 hours"). If not present, return null.
    3) states_applies_regardless_background: 
       - true if the answer explicitly states that this 35-hour requirement applies regardless of education background (e.g., applies both to those with a four-year degree and to those with a secondary diploma).
       - false if the answer explicitly says it differs or is not required for one of the backgrounds.
       - null if the answer does not address this point.
    4) states_must_complete_before_exam:
       - true if the answer explicitly states these education hours must be completed prior to taking/scheduling the PMP exam (or before submitting the exam application).
       - false if the answer explicitly says otherwise.
       - null if the answer does not address this point.

    Important notes:
    - Treat "contact hours" as equivalent to "formal project management education hours".
    - Do not infer or add information not stated in the answer.
    - If the answer is ambiguous or does not mention an item, return null for that item.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_pmp_requirements(
    evaluator: Evaluator,
    parent_node,
    extraction: PMPEducationExtraction
) -> None:
    """
    Build the rubric verification tree for PMP education requirements and run checks.
    All checks are critical under the PMP_Education_Requirement node as specified.
    """
    # Create the critical parent node for all PMP education-hour checks
    pmp_node = evaluator.add_parallel(
        id="PMP_Education_Requirement",
        desc="Evaluate whether the answer satisfies the stated PMP formal project management education-hour eligibility constraints.",
        parent=parent_node,
        critical=True
    )

    # Leaf 1: Minimum hours stated as 35 hours
    min_hours_leaf = evaluator.add_leaf(
        id="Minimum_Hours_Stated",
        desc="Answer states the minimum required formal project management education/training is 35 hours.",
        parent=pmp_node,
        critical=True
    )

    min_hours_claim = (
        "The answer explicitly states that the minimum required formal project management education or "
        "training for PMP eligibility is 35 hours (also commonly referred to as '35 contact hours')."
    )
    await evaluator.verify(
        claim=min_hours_claim,
        node=min_hours_leaf,
        additional_instruction=(
            "Judge only based on the provided answer text. Treat 'contact hours' as equivalent to formal project "
            "management education hours. Accept formulations like '35 hours', '35 contact hours', or 'at least 35 hours' "
            "as satisfying this requirement."
        )
    )

    # Leaf 2: Applies regardless of education background
    applies_leaf = evaluator.add_leaf(
        id="Applies_Regardless_of_Education_Background",
        desc="Answer states the 35-hour education requirement applies regardless of whether the applicant has a four-year degree or a secondary diploma.",
        parent=pmp_node,
        critical=True
    )

    applies_claim = (
        "The answer explicitly states that the 35-hour formal project management education requirement applies "
        "regardless of the applicant's education background — it applies to candidates with a four-year degree "
        "and to those with a secondary diploma alike."
    )
    await evaluator.verify(
        claim=applies_claim,
        node=applies_leaf,
        additional_instruction=(
            "Look for wording that clearly conveys 'applies to both', 'applies regardless of degree', "
            "'for all applicants', or equivalent. If the answer does not mention this point at all, the claim is incorrect."
        )
    )

    # Leaf 3: Must be completed before the exam
    before_exam_leaf = evaluator.add_leaf(
        id="Completed_Before_Exam",
        desc="Answer states the required education hours must be completed before taking the PMP certification exam.",
        parent=pmp_node,
        critical=True
    )

    before_exam_claim = (
        "The answer states that the required formal project management education (35 hours/contact hours) must be "
        "completed before taking or scheduling the PMP certification exam (or before submitting the exam application)."
    )
    await evaluator.verify(
        claim=before_exam_claim,
        node=before_exam_leaf,
        additional_instruction=(
            "Accept equivalent phrasing like 'before scheduling the exam' or 'prior to the exam application'. "
            "If the answer does not mention timing relative to the exam at all, this claim is incorrect."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the PMP education-hour requirement task.
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
        default_model=model
    )

    # Extract structured info from the answer (for recording and analysis)
    extraction = await evaluator.extract(
        prompt=prompt_extract_pmp_education(),
        template_class=PMPEducationExtraction,
        extraction_name="pmp_education_extraction"
    )

    # Add ground truth info for reference
    evaluator.add_ground_truth({
        "minimum_required_hours": "35",
        "notes": [
            "Treat '35 contact hours' as equivalent to '35 hours of formal project management education'.",
            "The 35-hour education requirement applies regardless of four-year degree or secondary diploma.",
            "The education hours must be completed before taking/scheduling the PMP exam."
        ]
    })

    # Build the verification tree and run checks
    await build_and_verify_pmp_requirements(evaluator, root, extraction)

    # Return summary with verification tree and recorded extraction
    return evaluator.get_summary()