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
TASK_ID = "rayjay_caltech_presidency_2026"
TASK_DESCRIPTION = """
Ray Jayawardhana, a renowned astrophysicist and academic leader, has been appointed as president of a major research university in the United States, with his term beginning on July 1, 2026. Identify the name of this university, specify which number president he will be in that institution's history, and provide the date when this appointment was officially announced.
"""

# Ground truth expectations
EXPECTED_UNIVERSITY = "California Institute of Technology (Caltech)"
EXPECTED_PRESIDENTIAL_NUMBER = "10th"
EXPECTED_ANNOUNCEMENT_DATE = "January 6, 2026"
EXPECTED_TERM_START = "July 1, 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AppointmentExtraction(BaseModel):
    """
    Structured extraction of the appointment details as stated in the agent's answer.
    """
    university: Optional[str] = None
    presidential_number: Optional[str] = None  # Accept formats like "10th", "tenth", "10"
    announcement_date: Optional[str] = None    # Accept formats like "January 6, 2026", "Jan 6, 2026", "2026-01-06"
    sources: List[str] = Field(default_factory=list)  # Any URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_appointment_info() -> str:
    return """
    Extract the following fields from the provided answer about Ray Jayawardhana’s presidential appointment:

    1) university: The name of the university where he has been appointed president.
       - Accept variants like "Caltech" or "California Institute of Technology".
       - If the university is not explicitly mentioned, set this to null.

    2) presidential_number: Which number president he will be in that institution's history.
       - Accept variants such as "10th", "tenth", or "10".
       - If not stated, set this to null.

    3) announcement_date: The date when the appointment was officially announced.
       - Accept common date formats like "January 6, 2026", "Jan 6, 2026", or ISO "2026-01-06".
       - If not provided, set this to null.

    4) sources: Extract all URLs explicitly cited in the answer that relate to this appointment (press releases, official news posts, etc.).
       - Include URLs presented directly, in markdown, or in a sources section.
       - Do not invent URLs; only extract those explicitly present.

    Return a single JSON object with keys: university, presidential_number, announcement_date, sources.
    If any field is missing, return null (for strings) or an empty list (for sources).
    """


# --------------------------------------------------------------------------- #
# Main evaluation logic                                                       #
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
    Evaluate an answer for Ray Jayawardhana's presidential appointment details.

    Verifies three critical requirements:
    - University identification (Caltech) with term beginning July 1, 2026.
    - Presidential number (10th).
    - Official announcement date (January 6, 2026).
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_appointment_info(),
        template_class=AppointmentExtraction,
        extraction_name="appointment_info"
    )

    # Add ground truth for transparency
    evaluator.add_ground_truth({
        "expected_university": EXPECTED_UNIVERSITY,
        "expected_presidential_number": EXPECTED_PRESIDENTIAL_NUMBER,
        "expected_announcement_date": EXPECTED_ANNOUNCEMENT_DATE,
        "expected_term_start": EXPECTED_TERM_START
    }, gt_type="ground_truth")

    # Parent node for this rubric
    parent_node = evaluator.add_parallel(
        id="Presidential_Appointment_Information",
        desc="Verify that the answer correctly identifies Ray Jayawardhana's upcoming presidential appointment details",
        parent=root,
        critical=False
    )

    # ------------------------- Leaf 1: University ------------------------- #
    uni_node = evaluator.add_leaf(
        id="University_Identification",
        desc="The answer must identify the California Institute of Technology (Caltech) as the institution where Ray Jayawardhana will serve as president starting July 1, 2026",
        parent=parent_node,
        critical=True
    )

    uni_claim = (
        "The answer identifies the California Institute of Technology (Caltech) as the institution where "
        f"Ray Jayawardhana will serve as president starting {EXPECTED_TERM_START}."
    )
    await evaluator.verify(
        claim=uni_claim,
        node=uni_node,
        # Prefer to ground with URLs if the answer provided any sources; otherwise simple verification
        sources=extracted.sources if extracted and extracted.sources else None,
        additional_instruction=(
            "Focus on whether the ANSWER explicitly states this institution and start date. "
            "Accept reasonable variants like 'Caltech' for 'California Institute of Technology'. "
            "Minor formatting differences are acceptable."
        )
    )

    # --------------------- Leaf 2: Presidential Number -------------------- #
    num_node = evaluator.add_leaf(
        id="Presidential_Number",
        desc="The answer must state that Ray Jayawardhana will be the 10th president in the institution's history",
        parent=parent_node,
        critical=True
    )

    num_claim = (
        "The answer states that Ray Jayawardhana will be the 10th president in the institution's history "
        "(e.g., '10th', 'tenth', or 'No. 10')."
    )
    await evaluator.verify(
        claim=num_claim,
        node=num_node,
        sources=extracted.sources if extracted and extracted.sources else None,
        additional_instruction=(
            "Check the ANSWER text for the ordinal or numeric representation indicating '10th'. "
            "Accept '10th', 'tenth', 'No. 10', or similar expressions."
        )
    )

    # --------------------- Leaf 3: Announcement Date ---------------------- #
    date_node = evaluator.add_leaf(
        id="Announcement_Date",
        desc="The answer must provide the announcement date of January 6, 2026",
        parent=parent_node,
        critical=True
    )

    date_claim = (
        "The answer provides the official announcement date as January 6, 2026."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=extracted.sources if extracted and extracted.sources else None,
        additional_instruction=(
            "Verify the ANSWER includes the official announcement date. "
            "Accept common variants like 'Jan 6, 2026', 'January 6, 2026', or '2026-01-06'."
        )
    )

    # Return summary
    return evaluator.get_summary()