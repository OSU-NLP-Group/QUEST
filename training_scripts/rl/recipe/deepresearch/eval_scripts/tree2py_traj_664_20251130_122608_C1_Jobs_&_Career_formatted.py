import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "sonny_dykes_previous_role"
TASK_DESCRIPTION = """
Sonny Dykes is currently the head football coach at Texas Christian University (TCU). Before joining TCU in 2021, what university did he serve as head football coach? Please provide the name of the university, how many seasons he served in that position, and his overall win-loss record during that tenure.
"""


class PreviousHeadCoachExtraction(BaseModel):
    """Structured extraction of the immediately previous head-coaching role before TCU."""
    previous_university: Optional[str] = None  # e.g., "SMU" or "Southern Methodist University"
    role_title: Optional[str] = None           # e.g., "Head Coach"
    seasons_count: Optional[str] = None        # e.g., "four", "4", "2018–2021 (four seasons)"
    win_loss_record: Optional[str] = None      # e.g., "30-18" or "30–18"
    tenure_years: Optional[str] = None         # e.g., "2018-2021"
    sources: List[str] = Field(default_factory=list)  # URLs cited in the answer that support these facts


def prompt_extract_previous_head_coach() -> str:
    return """
    Extract Sonny Dykes' immediately previous head football coaching role before joining TCU in 2021 from the provided answer.

    Return the following fields:
    - previous_university: The name of the university where he served as head football coach immediately before TCU (prefer the full name like "Southern Methodist University" or the commonly used short form "SMU" exactly as written in the answer).
    - role_title: The role title as written in the answer for that previous position (e.g., "head coach", "Head Coach", "HC"). If not stated, return null.
    - seasons_count: The number of seasons he served in that role, exactly as stated in the answer (e.g., "four", "4", "four seasons", "2018–2021 (four seasons)"). If not mentioned, return null.
    - win_loss_record: The overall win-loss record at that university as stated in the answer (e.g., "30-18" or "30–18"). If not mentioned, return null.
    - tenure_years: The years span for that tenure, if present (e.g., "2018–2021"). If not mentioned, return null.
    - sources: All URLs the answer provides that support these facts. Only include actual URLs mentioned in the answer text; do not invent any. Include valid URLs even if they are in markdown link format.

    Notes:
    - Extract only facts explicitly stated in the answer; do not infer or invent any information.
    - If any field is not mentioned, set it to null. For sources, return an empty array if none are provided.
    - Keep text exactly as in the answer (including en dashes vs hyphens, capitalization, etc.).
    """


async def build_and_verify_previous_position(
    evaluator: Evaluator,
    parent_node,
    info: PreviousHeadCoachExtraction,
) -> None:
    """
    Build the verification subtree for the previous head-coaching position and run checks.
    All children are critical, aligned with the rubric. The parent aggregate node is critical.
    """
    prev_node = evaluator.add_parallel(
        id="Previous_Head_Coaching_Position",
        desc="Verify Sonny Dykes' head coaching position immediately before joining TCU in 2021, including university, seasons, and overall win-loss record, under the given constraints.",
        parent=parent_node,
        critical=True,
    )

    # 1) Role_Is_Head_Coaching_Position
    role_leaf = evaluator.add_leaf(
        id="Role_Is_Head_Coaching_Position",
        desc="Confirms the identified previous role is a head football coaching position (not an assistant/coordinator role).",
        parent=prev_node,
        critical=True,
    )
    role_claim = (
        "In the answer, Sonny Dykes's identified previous role before TCU is a head football coaching position "
        "(not an assistant or coordinator). Consider the textual title used (e.g., 'head coach', 'HC') to determine this."
    )
    await evaluator.verify(
        claim=role_claim,
        node=role_leaf,
        additional_instruction="Use the answer content. If the answer states 'head coach' (or equivalent), pass. If it suggests assistant/coordinator, fail.",
    )

    # 2) Immediately_Precedes_TCU_Job
    precedes_leaf = evaluator.add_leaf(
        id="Immediately_Precedes_TCU_Job",
        desc="Confirms the identified head coaching position is the one held immediately before being hired at TCU.",
        parent=prev_node,
        critical=True,
    )
    precedes_claim = (
        "In the answer, the identified head coaching position was the one immediately preceding Sonny Dykes's hire at TCU."
    )
    await evaluator.verify(
        claim=precedes_claim,
        node=precedes_leaf,
        additional_instruction="Ignore any non-head-coach roles. Immediate means his last head-coach job before TCU (SMU, 2018–2021).",
    )

    # 3) TCU_Hire_Date_Not_Contradicted
    tcu_date_leaf = evaluator.add_leaf(
        id="TCU_Hire_Date_Not_Contradicted",
        desc="Does not contradict the constraint that Sonny Dykes was hired at TCU on November 30, 2021.",
        parent=prev_node,
        critical=True,
    )
    tcu_date_claim = (
        "The answer's statements do not contradict that Sonny Dykes was hired as TCU's head coach on November 30, 2021."
    )
    await evaluator.verify(
        claim=tcu_date_claim,
        node=tcu_date_leaf,
        additional_instruction=(
            "If the answer does not mention a hire date, treat this as not contradicted (Correct). "
            "If it specifies a clearly different date, treat as contradicted (Incorrect). "
            "Minor phrasing like 'late November 2021' is not a contradiction."
        ),
    )

    # 4) University_Identification (must be SMU)
    uni_leaf = evaluator.add_leaf(
        id="University_Identification",
        desc="Provides the correct university name for the immediately-previous head coaching job (must be SMU / Southern Methodist University per constraints).",
        parent=prev_node,
        critical=True,
    )
    uni_claim = (
        "In the answer, the previous head coaching university is Southern Methodist University (SMU). "
        "Accept 'SMU' or 'Southern Methodist University' as equivalent."
    )
    await evaluator.verify(
        claim=uni_claim,
        node=uni_leaf,
        additional_instruction="Judge against the answer text. Accept common variations like 'SMU' or 'Southern Methodist'.",
    )

    # 5) Tenure_Duration (must be four seasons)
    tenure_leaf = evaluator.add_leaf(
        id="Tenure_Duration",
        desc="Provides the correct number of seasons served in that head coaching position (must be four seasons per constraints).",
        parent=prev_node,
        critical=True,
    )
    tenure_claim = (
        "In the answer, Sonny Dykes is stated to have served four seasons in that previous head coaching position."
    )
    await evaluator.verify(
        claim=tenure_claim,
        node=tenure_leaf,
        additional_instruction="Accept '4', 'four', or the years 2018–2021 indicating four seasons.",
    )

    # 6) Overall_Win_Loss_Record (must be 30-18)
    record_leaf = evaluator.add_leaf(
        id="Overall_Win_Loss_Record",
        desc="Provides the correct overall win-loss record during that tenure (must be 30-18 per constraints).",
        parent=prev_node,
        critical=True,
    )
    record_claim = (
        "In the answer, Sonny Dykes's overall win-loss record during that tenure is 30–18 (accept '30-18' or minor formatting variants)."
    )
    await evaluator.verify(
        claim=record_claim,
        node=record_leaf,
        additional_instruction="Accept minor formatting variants like hyphen or en dash. The numeric values must be 30 and 18.",
    )

    # 7) Publicly_Verifiable_Sourcing
    source_leaf = evaluator.add_leaf(
        id="Publicly_Verifiable_Sourcing",
        desc="Information is supported by publicly available/verifiable sources (e.g., provides at least one accessible reference URL supporting the claims).",
        parent=prev_node,
        critical=True,
    )
    # Choose a minimal core claim easier to verify from common sources:
    source_claim = (
        "Sonny Dykes was the head football coach at Southern Methodist University (SMU) immediately before joining TCU in 2021."
    )
    await evaluator.verify(
        claim=source_claim,
        node=source_leaf,
        sources=info.sources,
        additional_instruction=(
            "Pass if any single provided URL clearly states or implies that Dykes was SMU's head coach prior to TCU "
            "(e.g., Wikipedia, SMU/TCU official announcement, reputable news)."
        ),
    )


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
    Entry point to evaluate an agent's answer for Sonny Dykes' previous head-coaching position before TCU.
    """
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_previous_head_coach(),
        template_class=PreviousHeadCoachExtraction,
        extraction_name="previous_head_coach_extraction",
    )

    evaluator.add_ground_truth({
        "expected_previous_university": "Southern Methodist University (SMU)",
        "expected_seasons": "four",
        "expected_record": "30-18",
        "tcu_hire_date_constraint": "November 30, 2021",
    })

    await build_and_verify_previous_position(evaluator, root, extracted)

    return evaluator.get_summary()