import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "athletic_dept_positions_three_universities"
TASK_DESCRIPTION = """I am exploring career opportunities in collegiate athletic administration and am interested in positions at three specific universities. Please identify one current athletic department job opening at each of the following universities: Ferris State University, Miami University (Ohio), and Villanova University.

For each position, ensure that it meets ALL of the following criteria:
- The position must be at the assistant director level or higher in the athletic administration hierarchy (exclude entry-level, student positions, or graduate assistant positions)
- The position must require a minimum of 3 years of professional experience in athletics, coaching, or sports administration
- The position must require at least a bachelor's degree (positions that prefer a master's degree are acceptable)
- The position must be a full-time, regular position (not temporary, seasonal, part-time, or student employment)

For each of the three positions, provide:
1. The official job title
2. The minimum years of experience required as stated in the posting
3. The required degree level (bachelor's, master's, etc.)
4. A direct URL link to the official job posting
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PositionInfo(BaseModel):
    job_title: Optional[str] = None
    experience_years_required: Optional[str] = None
    degree_required: Optional[str] = None
    job_posting_url: Optional[str] = None


class PositionsExtraction(BaseModel):
    ferris_state: Optional[PositionInfo] = None
    miami_ohio: Optional[PositionInfo] = None
    villanova: Optional[PositionInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    Extract exactly one athletic department job position for each of the following universities as they appear in the answer text:
    - Ferris State University
    - Miami University (Ohio) (sometimes written as Miami University (OH) or Miami (OH))
    - Villanova University

    For each university, extract a single position and return the following fields:
    - job_title: the official job title as stated in the posting
    - experience_years_required: the minimum years of professional experience required (quote or paraphrase the exact minimum requirement, e.g., "Minimum 3 years" or "3+ years")
    - degree_required: the required minimum degree level (e.g., "Bachelor's required", "Master's required"; if master's preferred but bachelor's required, state "Bachelor's required; Master's preferred")
    - job_posting_url: a direct URL to the official job posting (HR system, university careers site, or official athletics site). Extract only URLs explicitly present in the answer.

    Important:
    - If multiple positions are mentioned for a university, choose the first one that matches the context of the answer.
    - If any field is not provided in the answer, set it to null.
    - Do not invent information or URLs. Only extract items explicitly mentioned.
    - Return a JSON object with top-level keys: ferris_state, miami_ohio, villanova, each being an object with the four fields above (or null if absent).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


async def verify_university_position(
    evaluator: Evaluator,
    parent_node,
    position: Optional[PositionInfo],
    group_id: str,
    group_desc: str,
    prefix: str
) -> None:
    """
    Build verification subtree for a single university position.
    """
    # Group node for this university (non-critical to allow partial credit per university)
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent_node,
        critical=False
    )

    # Information completeness (critical) - four required fields
    info_node = evaluator.add_parallel(
        id=f"{prefix}_Information_Completeness",
        desc=f"All required information is provided for the {group_desc.split(' at ')[-1]}",
        parent=group_node,
        critical=True
    )

    title_ok = _nonempty(position.job_title) if position else False
    exp_ok = _nonempty(position.experience_years_required) if position else False
    degree_ok = _nonempty(position.degree_required) if position else False
    url_ok = _nonempty(position.job_posting_url) if position else False

    evaluator.add_custom_node(
        result=title_ok,
        id=f"{prefix}_Job_Title",
        desc="The official job title is provided",
        parent=info_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=exp_ok,
        id=f"{prefix}_Experience_Years",
        desc="The minimum years of experience required is stated",
        parent=info_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=degree_ok,
        id=f"{prefix}_Degree_Level",
        desc="The required degree level is specified",
        parent=info_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=url_ok,
        id=f"{prefix}_Job_URL",
        desc="A direct URL to the job posting is provided",
        parent=info_node,
        critical=True
    )

    # Prepare source URL for verification leaves (may be None; if missing, these leaves will be auto-skipped due to critical sibling failure)
    src_url = position.job_posting_url if position and position.job_posting_url else None
    job_title_str = position.job_title if position and position.job_title else "the role"

    # Four critical verification leaves (policy checks), each grounded to the job posting URL
    # 1) Position Level
    level_node = evaluator.add_leaf(
        id=f"{prefix}_Position_Level",
        desc="The position is at assistant director level or higher in athletic administration hierarchy, excluding entry-level, student, or graduate assistant positions",
        parent=group_node,
        critical=True
    )
    level_claim = (
        f"According to this job posting, the role titled '{job_title_str}' is an athletics department administration role "
        f"at the assistant director level or higher (e.g., Assistant/Associate/Senior Associate/Deputy Athletic Director, "
        f"or Director-level). It is not an entry-level, student, intern, or graduate assistant position."
    )
    level_ai = (
        "Decide based on the job title and the posting's description. Accept titles like 'Assistant Athletic Director', "
        "'Associate Athletic Director', 'Senior Associate AD', 'Deputy AD', or any 'Director' level role within an athletics department "
        "(e.g., 'Director of Compliance' within Athletics). Reject roles that are Coordinators, Specialists, Assistants (non-director), "
        "Interns, Graduate Assistants, or student roles."
    )

    # 2) Experience Requirement (>= 3 years, required)
    exp_node = evaluator.add_leaf(
        id=f"{prefix}_Experience_Requirement",
        desc="The position requires a minimum of 3 years of professional experience in athletics, coaching, or sports administration",
        parent=group_node,
        critical=True
    )
    exp_claim = (
        "The posting explicitly requires a minimum of at least 3 years of relevant professional experience "
        "(e.g., athletics, coaching, college athletics administration, or closely related)."
    )
    exp_ai = (
        "Look at REQUIRED qualifications only. Phrases like 'minimum 3 years', '3+ years', or 'at least three years' qualify. "
        "If 3 years is only listed as preferred (not required), this does NOT satisfy the requirement."
    )

    # 3) Educational Requirement (>= Bachelor's required)
    edu_node = evaluator.add_leaf(
        id=f"{prefix}_Educational_Requirement",
        desc="The position requires at least a bachelor's degree (master's degree preferred is acceptable)",
        parent=group_node,
        critical=True
    )
    edu_claim = (
        "The posting requires at least a bachelor's degree (bachelor's or higher). "
        "If the posting only 'prefers' a degree and does not require it, this does not satisfy the requirement."
    )
    edu_ai = (
        "Accept 'Bachelor's degree required' (with or without 'Master's preferred'). "
        "Also accept if a Master's degree is required (that is higher than bachelor's). "
        "Do NOT accept if the degree is only 'preferred' and not required."
    )

    # 4) Employment Type (full-time, regular)
    emp_node = evaluator.add_leaf(
        id=f"{prefix}_Employment_Type",
        desc="The position is a full-time, regular position (not temporary, seasonal, part-time, or student position)",
        parent=group_node,
        critical=True
    )
    emp_claim = (
        "This position is a full-time, regular (benefits-eligible) employment role and is NOT temporary, seasonal, part-time, fixed-term, "
        "or student/graduate assistant employment."
    )
    emp_ai = (
        "Confirm that the posting states 'full-time' and indicates a regular/benefits-eligible role. "
        "If it is part-time, term-limited, seasonal, temporary, student, or graduate assistant, it does not qualify."
    )

    # Batch verify the four policy checks (they will be skipped automatically if critical info completeness failed)
    await evaluator.batch_verify(
        [
            (level_claim, src_url, level_node, level_ai),
            (exp_claim, src_url, exp_node, exp_ai),
            (edu_claim, src_url, edu_node, edu_ai),
            (emp_claim, src_url, emp_node, emp_ai),
        ]
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
    Evaluate an answer for the collegiate athletic administration positions task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Three universities are evaluated independently
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

    # Extraction
    extracted_positions = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction"
    )

    # Build verification subtrees per university
    await verify_university_position(
        evaluator=evaluator,
        parent_node=root,
        position=extracted_positions.ferris_state,
        group_id="Ferris_State_Position",
        group_desc="Identify a qualifying athletic department position at Ferris State University",
        prefix="FSU"
    )

    await verify_university_position(
        evaluator=evaluator,
        parent_node=root,
        position=extracted_positions.miami_ohio,
        group_id="Miami_University_Position",
        group_desc="Identify a qualifying athletic department position at Miami University (Ohio)",
        prefix="Miami"
    )

    await verify_university_position(
        evaluator=evaluator,
        parent_node=root,
        position=extracted_positions.villanova,
        group_id="Villanova_University_Position",
        group_desc="Identify a qualifying athletic department position at Villanova University",
        prefix="Villanova"
    )

    # Return summary
    return evaluator.get_summary()