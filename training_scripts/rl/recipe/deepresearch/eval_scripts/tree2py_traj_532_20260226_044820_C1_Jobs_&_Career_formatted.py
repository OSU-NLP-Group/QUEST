import asyncio
import logging
from typing import List, Optional, Dict

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "csu_uconn_coach_transition_2025"
TASK_DESCRIPTION = (
    "In November 2025, a college football head coach accepted a position at Colorado State University after departing "
    "from the University of Connecticut (UConn). Identify the coach's full name and state the month and year when they "
    "officially began their role at Colorado State."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class CoachTransitionExtraction(BaseModel):
    """Information extracted from the agent's answer regarding the coach transition."""
    coach_full_name: Optional[str] = None
    start_month: Optional[str] = None
    start_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coach_transition() -> str:
    return """
    Extract the specific information about the college football head coach who accepted the Colorado State University (CSU) head coaching job after departing from the University of Connecticut (UConn) in November 2025.

    Return the following fields:
    - coach_full_name: The full name of the coach identified in the answer as having accepted the CSU head coaching position after leaving UConn.
    - start_month: The month when the coach officially began their role at Colorado State (e.g., "December", "January").
    - start_year: The year when the coach officially began their role at Colorado State (e.g., "2025", "2026").
    - sources: A list of all URLs explicitly cited in the answer that support the identification of the coach and/or the official start date. Include any press releases, athletics announcements, or credible news articles. If the answer does not provide any URLs, return an empty list.

    Notes:
    - Extract only what is explicitly stated in the answer.
    - If a field is missing in the answer, set it to null. If no sources are cited, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Verification utility                                                        #
# --------------------------------------------------------------------------- #
def build_claims(extracted: CoachTransitionExtraction) -> Dict[str, str]:
    """
    Build verification claims for the two rubric checks from the extracted information.
    """
    full_name = extracted.coach_full_name or ""
    month = extracted.start_month or ""
    year = extracted.start_year or ""

    coach_claim = (
        f"The coach identified is {full_name}, who accepted the Colorado State University head football coaching "
        f"position after departing from the University of Connecticut (UConn) in November 2025."
    )

    start_date_claim = (
        f"The coach officially began their role at Colorado State in {month} {year}."
    )

    return {
        "coach_identification": coach_claim,
        "start_date": start_date_claim,
    }


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
    Evaluate the agent's answer for identifying the coach and official start date at Colorado State.
    """
    # Initialize evaluator (root is parallel; children will be critical to enforce all-or-nothing scoring)
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_coach_transition(),
        template_class=CoachTransitionExtraction,
        extraction_name="coach_transition_extraction",
    )

    # Build verification leaf nodes under the root (both critical)
    coach_node = evaluator.add_leaf(
        id="coach_identification",
        desc="The coach's full name is correctly identified as the person who accepted the Colorado State head coaching position after departing from UConn",
        parent=root,
        critical=True,
    )

    start_date_node = evaluator.add_leaf(
        id="start_date",
        desc="The month and year when the coach officially began their role at Colorado State are correctly stated",
        parent=root,
        critical=True,
    )

    # Construct claims and prepare sources
    claims = build_claims(extracted)
    sources = extracted.sources if extracted.sources else None

    # Verify both leaves (parallel)
    await evaluator.batch_verify(
        [
            (
                claims["coach_identification"],
                sources,
                coach_node,
                "Verify that the cited source(s) explicitly show the identified coach left UConn and accepted the CSU head coaching job in November 2025. Prefer official athletics announcements or reputable news outlets."
            ),
            (
                claims["start_date"],
                sources,
                start_date_node,
                "Verify that the cited source(s) explicitly indicate the month and year when the coach officially began their role at Colorado State (e.g., when the appointment became effective or when they officially started duties)."
            ),
        ]
    )

    # Return structured summary
    return evaluator.get_summary()