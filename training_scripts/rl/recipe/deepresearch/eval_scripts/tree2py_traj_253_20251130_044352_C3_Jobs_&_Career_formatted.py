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
TASK_ID = "niaaa_cmaa_pathway"
TASK_DESCRIPTION = (
    "An individual wants to pursue certification as a Certified Master Athletic Administrator (CMAA) through the "
    "National Interscholastic Athletic Administrators Association (NIAAA) and needs to understand the complete "
    "certification pathway from the beginning. What are all the sequential requirements that must be met to earn the "
    "CMAA certification, including any prerequisite certifications, educational requirements, required Leadership "
    "Training Courses (LTC) by course number, experience requirements, work experience credits, examinations, and "
    "final project requirements?"
)

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class PathwayExtraction(BaseModel):
    """Structured extraction of whether the answer explicitly states each required item."""
    mentions_caa_prerequisite: Optional[bool] = None

    mentions_bachelor_degree: Optional[bool] = None
    mentions_experience_two_years: Optional[bool] = None
    mentions_employment_ad_caa: Optional[bool] = None

    mentions_basic_courses_501_502_503: Optional[bool] = None
    mentions_additional_courses_504_506: Optional[bool] = None

    mentions_work_experience_credits_65: Optional[bool] = None
    mentions_caa_exam: Optional[bool] = None

    mentions_employment_ad_cmaa: Optional[bool] = None
    mentions_required_courses_508_510: Optional[bool] = None
    mentions_one_600_level_course: Optional[bool] = None
    mentions_one_700_level_course: Optional[bool] = None
    mentions_three_electives_any_level: Optional[bool] = None
    mentions_ltc503_grandfathering_not_allowed: Optional[bool] = None

    mentions_cmaa_project_grad_level_written_or_oral: Optional[bool] = None

    # Optional lists of any LTC numbers the answer mentions, for reference
    ltc_numbers_mentioned: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pathway() -> str:
    return (
        "From the provided answer, determine whether each of the following statements is explicitly stated. "
        "Return booleans for each. Only mark true if the answer makes the requirement explicit (clear synonyms are okay). "
        "Also extract any LTC course numbers explicitly mentioned in the answer as a list.\n\n"
        "Fields to return:\n"
        "- mentions_caa_prerequisite: The answer states that Certified Athletic Administrator (CAA) certification must be obtained before applying for CMAA.\n"
        "- mentions_bachelor_degree: The answer states a bachelor's degree from an accredited institution is required (for CAA).\n"
        "- mentions_experience_two_years: The answer states at least 2 years of experience as an athletic administrator is required (for CAA).\n"
        "- mentions_employment_ad_caa: The answer states the applicant must be currently employed as an athletic director at the time of CAA application.\n"
        "- mentions_basic_courses_501_502_503: The answer identifies LTC 501, LTC 502, and LTC 503 as required basic courses for CAA.\n"
        "- mentions_additional_courses_504_506: The answer identifies LTC 504 and LTC 506 as required additional courses for CAA.\n"
        "- mentions_work_experience_credits_65: The answer states 65 credits through accumulated work experience are required for CAA.\n"
        "- mentions_caa_exam: The answer states the applicant must pass the CAA examination.\n"
        "- mentions_employment_ad_cmaa: The answer states the applicant must be employed as an athletic director at the time of CMAA application.\n"
        "- mentions_required_courses_508_510: The answer identifies LTC 508 and LTC 510 as required CMAA courses.\n"
        "- mentions_one_600_level_course: The answer states one LTC 600-level (Operations and Management) course is required for CMAA.\n"
        "- mentions_one_700_level_course: The answer states one LTC 700-level (Leadership) course is required for CMAA.\n"
        "- mentions_three_electives_any_level: The answer states three elective courses from any level (500, 600, 700, or 900) are required for CMAA.\n"
        "- mentions_ltc503_grandfathering_not_allowed: The answer explicitly states that LTC 503 must be completed to qualify for CMAA even if the candidate obtained CAA before LTC 503 became a CAA requirement (i.e., no grandfathering).\n"
        "- mentions_cmaa_project_grad_level_written_or_oral: The answer states a graduate-level written or oral project is required for CMAA.\n"
        "- ltc_numbers_mentioned: List any LTC course numbers (e.g., '501', '502', '503', '504', '506', '508', '510', '6xx', '7xx', '900') that are explicitly mentioned.\n\n"
        "If a statement is not explicitly present, return false."
    )


# --------------------------------------------------------------------------- #
# Ground truth reference (for summary only)                                   #
# --------------------------------------------------------------------------- #
GROUND_TRUTH = {
    "sequence": [
        "CAA prerequisite before CMAA",
        "CAA foundation (Bachelor's degree; 2 years experience; employed as AD at CAA application)",
        "CAA LTC required courses (501, 502, 503 basic; 504, 506 additional)",
        "CAA completion (65 work-experience credits; pass CAA exam)",
        "CMAA eligibility & coursework (employed as AD at CMAA application; LTC 508, LTC 510; one 600-level; one 700-level; three electives from any level incl. 900; LTC 503 must be completed even if CAA was earned before it became required)",
        "CMAA project (graduate-level written or oral)"
    ],
    "CAA_foundation": {
        "bachelor_degree_required": True,
        "two_years_experience_required": True,
        "employment_as_athletic_director_at_application": True
    },
    "CAA_courses": {
        "basic_required": ["LTC 501", "LTC 502", "LTC 503"],
        "additional_required": ["LTC 504", "LTC 506"]
    },
    "CAA_completion": {
        "work_experience_credits_required": 65,
        "must_pass_caa_exam": True
    },
    "CMAA_eligibility_courses": {
        "employment_as_athletic_director_at_application": True,
        "required_courses": ["LTC 508", "LTC 510"],
        "one_600_level_course": True,
        "one_700_level_course": True,
        "three_electives_any_level_incl_900": True,
        "ltc503_must_be_completed_no_grandfathering": True
    },
    "CMAA_project": {
        "graduate_level_written_or_oral_project_required": True
    }
}


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_pathway(
    evaluator: Evaluator,
    root_node,
) -> None:
    """
    Build the verification tree based on the rubric and perform LLM-based checks on the answer content.
    Root is a sequential node to enforce pathway order and skip subsequent checks if earlier essential steps fail.
    """

    # 1) CAA prerequisite (leaf, critical)
    leaf_caa_prereq = evaluator.add_leaf(
        id="CAA_Prerequisite",
        desc="Answer explicitly states that Certified Athletic Administrator (CAA) certification must be obtained before applying for CMAA",
        parent=root_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that CAA certification must be obtained before applying for CMAA.",
        node=leaf_caa_prereq,
        additional_instruction=(
            "Pass only if the answer clearly indicates CAA is a prerequisite for CMAA (e.g., 'must first obtain CAA', "
            "'CAA required prior to CMAA', 'CAA is a prerequisite for CMAA')."
        ),
    )

    # 2) CAA foundation (parallel, critical)
    node_caa_foundation = evaluator.add_parallel(
        id="CAA_Foundation",
        desc="Educational and employment/experience foundation requirements for CAA",
        parent=root_node,
        critical=True,
    )

    leaf_bachelor = evaluator.add_leaf(
        id="Bachelor_Degree",
        desc="Answer states a bachelor's degree from an accredited institution is required",
        parent=node_caa_foundation,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states a bachelor's degree from an accredited institution is required for CAA.",
        node=leaf_bachelor,
        additional_instruction="Accept clear synonyms or equivalent phrasing indicating a bachelor's degree from an accredited institution is required."
    )

    leaf_experience_2y = evaluator.add_leaf(
        id="Experience_Requirement",
        desc="Answer states at least 2 years of experience as an athletic administrator is required",
        parent=node_caa_foundation,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states at least two years of experience as an athletic administrator is required for CAA.",
        node=leaf_experience_2y,
        additional_instruction="Accept equivalent phrasing like 'minimum of 2 years' or 'at least two years'."
    )

    leaf_employed_caa = evaluator.add_leaf(
        id="Employment_as_AD_for_CAA",
        desc="Answer states the applicant must be currently employed as an athletic director at the time of CAA application",
        parent=node_caa_foundation,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states the applicant must be currently employed as an athletic director at the time of CAA application.",
        node=leaf_employed_caa,
        additional_instruction="The statement must apply to CAA application timing; accept close paraphrases indicating current AD employment is required."
    )

    # 3) CAA courses (parallel, critical)
    node_caa_courses = evaluator.add_parallel(
        id="CAA_Courses",
        desc="Leadership Training Course (LTC) requirements for CAA",
        parent=root_node,
        critical=True,
    )

    leaf_basic_501_502_503 = evaluator.add_leaf(
        id="Basic_Courses_501_502_503",
        desc="Answer identifies LTC 501, LTC 502, and LTC 503 as required basic courses for CAA",
        parent=node_caa_courses,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer identifies LTC 501, LTC 502, and LTC 503 as required basic courses for CAA.",
        node=leaf_basic_501_502_503,
        additional_instruction="The answer must list all three numbers 501, 502, and 503 as required for CAA."
    )

    leaf_additional_504_506 = evaluator.add_leaf(
        id="Additional_Courses_504_506",
        desc="Answer identifies LTC 504 and LTC 506 as required additional courses for CAA",
        parent=node_caa_courses,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer identifies LTC 504 and LTC 506 as required additional courses for CAA.",
        node=leaf_additional_504_506,
        additional_instruction="The answer must explicitly include both numbers 504 and 506 as required additional courses for CAA."
    )

    # 4) CAA completion (parallel, critical)
    node_caa_completion = evaluator.add_parallel(
        id="CAA_Completion",
        desc="Remaining requirements to complete CAA certification",
        parent=root_node,
        critical=True,
    )

    leaf_work_credits = evaluator.add_leaf(
        id="Work_Experience_Credits",
        desc="Answer states 65 credits through accumulated work experience are required for CAA",
        parent=node_caa_completion,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states 65 credits through accumulated work experience are required for CAA.",
        node=leaf_work_credits,
        additional_instruction="The answer must specify the number 65 and connect it to work-experience credits for CAA."
    )

    leaf_caa_exam = evaluator.add_leaf(
        id="CAA_Examination",
        desc="Answer states the applicant must pass the CAA examination",
        parent=node_caa_completion,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states the applicant must pass the CAA examination.",
        node=leaf_caa_exam,
        additional_instruction="Accept equivalent wording such as 'must pass the CAA exam' or 'passing the CAA exam is required'."
    )

    # 5) CMAA eligibility and courses (parallel, critical)
    node_cmaa_eligibility = evaluator.add_parallel(
        id="CMAA_Eligibility_and_Courses",
        desc="Employment and coursework requirements for CMAA beyond CAA",
        parent=root_node,
        critical=True,
    )

    leaf_employed_cmaa = evaluator.add_leaf(
        id="Employment_as_AD_for_CMAA",
        desc="Answer states the applicant must be employed as an athletic director at the time of CMAA application",
        parent=node_cmaa_eligibility,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states the applicant must be employed as an athletic director at the time of CMAA application.",
        node=leaf_employed_cmaa,
        additional_instruction="The statement must clearly apply to CMAA application timing; accept close paraphrases."
    )

    leaf_508_510 = evaluator.add_leaf(
        id="Required_Courses_508_510",
        desc="Answer identifies LTC 508 and LTC 510 as required CMAA courses",
        parent=node_cmaa_eligibility,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer identifies LTC 508 and LTC 510 as required CMAA courses.",
        node=leaf_508_510,
        additional_instruction="The answer must explicitly mention both LTC 508 and LTC 510 as required for CMAA."
    )

    leaf_600_level = evaluator.add_leaf(
        id="Course_600_Level",
        desc="Answer states one LTC 600-level (Operations and Management) course is required for CMAA",
        parent=node_cmaa_eligibility,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states one LTC 600-level (Operations and Management) course is required for CMAA.",
        node=leaf_600_level,
        additional_instruction="Accept wording that explicitly indicates 'one 600-level course' for CMAA; Operations & Management association can be noted."
    )

    leaf_700_level = evaluator.add_leaf(
        id="Course_700_Level",
        desc="Answer states one LTC 700-level (Leadership) course is required for CMAA",
        parent=node_cmaa_eligibility,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states one LTC 700-level (Leadership) course is required for CMAA.",
        node=leaf_700_level,
        additional_instruction="Accept wording that explicitly indicates 'one 700-level course' for CMAA; Leadership association can be noted."
    )

    leaf_electives = evaluator.add_leaf(
        id="Elective_Courses",
        desc="Answer states three elective courses from any level (500, 600, 700, or 900) are required for CMAA",
        parent=node_cmaa_eligibility,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states three elective courses from any level (500, 600, 700, or 900) are required for CMAA.",
        node=leaf_electives,
        additional_instruction="The answer should indicate 'three electives' and allow any of the levels including 900."
    )

    leaf_ltc503_ng = evaluator.add_leaf(
        id="LTC503_Grandfathering_Not_Allowed",
        desc="Answer explicitly states that LTC 503 must be completed to qualify for CMAA even if the candidate obtained CAA before LTC 503 became a CAA requirement",
        parent=node_cmaa_eligibility,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer explicitly states that LTC 503 must be completed to qualify for CMAA even if CAA was obtained "
            "before LTC 503 became a CAA requirement (i.e., no grandfathering)."
        ),
        node=leaf_ltc503_ng,
        additional_instruction="Pass only if the answer clearly states LTC 503 completion is required regardless of when CAA was earned."
    )

    # 6) CMAA project (leaf, critical)
    leaf_cmaa_project = evaluator.add_leaf(
        id="CMAA_Project",
        desc="Answer states a graduate-level written or oral project is required for CMAA",
        parent=root_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states a graduate-level written or oral project is required for CMAA.",
        node=leaf_cmaa_project,
        additional_instruction="Accept phrasing such as 'graduate-level project (written or oral) required for CMAA'."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the NIAAA CMAA certification pathway task.
    Builds a sequential verification tree and performs checks using the obj_task_eval framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # The pathway is sequential
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

    # Optional: extract a structured summary of what the answer claims
    extraction = await evaluator.extract(
        prompt=prompt_extract_pathway(),
        template_class=PathwayExtraction,
        extraction_name="cmaa_pathway_extraction",
    )

    # Add ground truth info into summary for transparency
    evaluator.add_ground_truth(GROUND_TRUTH, gt_type="expected_requirements")

    # Build and verify the tree according to the rubric
    await build_and_verify_pathway(evaluator, root)

    # Return the unified summary
    return evaluator.get_summary()