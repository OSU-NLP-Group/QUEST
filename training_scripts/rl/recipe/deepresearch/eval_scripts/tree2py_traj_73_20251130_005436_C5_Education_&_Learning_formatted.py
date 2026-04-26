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
TASK_ID = "academic_eligibility_requirements"
TASK_DESCRIPTION = (
    "A high school student-athlete in the United States is researching academic eligibility requirements to participate "
    "in interscholastic athletics. The student needs to understand requirements at the high school level in three "
    "different states and also needs to know the academic standards for NCAA Division I college athletics eligibility.\n\n"
    "For each of the following four athletic organizations, identify the specific academic eligibility requirements:\n\n"
    "1. OHSAA (Ohio High School Athletic Association): What is the minimum number of one-credit courses (or equivalent) "
    "that students in grades 9-12 must pass for athletic eligibility, and does the state association mandate a minimum GPA requirement?\n\n"
    "2. LHSAA (Louisiana High School Athletic Association): How many units must a student have earned from the previous "
    "school year to be eligible for the first semester, and what is the minimum GPA standard required?\n\n"
    "3. IHSA (Illinois High School Athletic Association): What is the minimum number of credit hours of high school work "
    "per week that a student must be passing for athletic eligibility?\n\n"
    "4. NCAA Division I: What is the minimum core-course GPA required for initial eligibility, and what is the total "
    "number of NCAA-approved core courses that must be completed?\n\n"
    "For each organization, provide: (a) the specific numeric requirements, (b) details about any GPA standards, and "
    "(c) an official reference URL from the organization's website or official documentation that supports your answer."
)

# Ground truth (for reference information in summary only; verification relies on cited sources)
GROUND_TRUTH = {
    "OHSAA": {
        "min_courses": "5 one-credit courses (or equivalent)",
        "gpa_policy": "No statewide minimum GPA mandated by OHSAA; local schools/districts may set GPA policies",
        "example_domains": ["ohsaa.org"]
    },
    "LHSAA": {
        "units_first_sem": "6 units earned from the previous school year",
        "gpa_requirement": "C average required as determined by the Local Education Authority (LEA)",
        "example_domains": ["lhsaa.org"]
    },
    "IHSA": {
        "weekly_credit_hours": "25 credit hours per week (generally 5 classes)",
        "example_domains": ["ihsa.org"]
    },
    "NCAA DI": {
        "min_core_gpa": "2.3 core-course GPA",
        "total_core_courses": "16 NCAA-approved core courses",
        "example_domains": ["ncaa.org", "eligibilitycenter.org"]
    }
}


# --------------------------------------------------------------------------- #
# Extraction data models                                                     #
# --------------------------------------------------------------------------- #
class OHSAAInfo(BaseModel):
    min_courses_pass_required: Optional[str] = None  # e.g., "5", "five", "5 one-credit courses"
    gpa_policy: Optional[str] = None  # e.g., "no minimum GPA at state level"
    source_urls: List[str] = Field(default_factory=list)  # official OHSAA URLs from the answer


class LHSAAInfo(BaseModel):
    units_required_prev_year_first_sem: Optional[str] = None  # e.g., "6"
    gpa_requirement: Optional[str] = None  # e.g., "C average determined by LEA"
    source_urls: List[str] = Field(default_factory=list)  # official LHSAA URLs from the answer


class IHSAInfo(BaseModel):
    weekly_credit_hours_required: Optional[str] = None  # e.g., "25"
    source_urls: List[str] = Field(default_factory=list)  # official IHSA URLs from the answer


class NCAADIInfo(BaseModel):
    min_core_course_gpa: Optional[str] = None  # e.g., "2.3"
    total_core_courses: Optional[str] = None  # e.g., "16"
    source_urls: List[str] = Field(default_factory=list)  # official NCAA URLs from the answer


class EligibilityExtraction(BaseModel):
    ohsaa: Optional[OHSAAInfo] = None
    lhsaa: Optional[LHSAAInfo] = None
    ihsa: Optional[IHSAAInfo] = None
    ncaadi: Optional[NCAADIInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_eligibility() -> str:
    return (
        "Extract the academic eligibility details for each organization as stated in the provided answer. "
        "Return the values exactly as they appear in the answer (prefer strings). Only extract URLs explicitly "
        "included in the answer.\n\n"
        "For each organization, capture the following fields:\n"
        "- OHSAA (Ohio High School Athletic Association):\n"
        "  • min_courses_pass_required: the stated minimum number of one-credit courses (or equivalent) students in grades 9–12 must pass\n"
        "  • gpa_policy: the stated policy regarding whether OHSAA mandates a statewide minimum GPA\n"
        "  • source_urls: list of official URLs from OHSAA cited in the answer (only URLs explicitly present)\n\n"
        "- LHSAA (Louisiana High School Athletic Association):\n"
        "  • units_required_prev_year_first_sem: the stated number of units required from the previous school year for first semester eligibility\n"
        "  • gpa_requirement: the stated minimum GPA standard (e.g., 'C average' determined by LEA)\n"
        "  • source_urls: list of official URLs from LHSAA cited in the answer\n\n"
        "- IHSA (Illinois High School Athletic Association):\n"
        "  • weekly_credit_hours_required: the stated minimum number of high school credit hours per week a student must be passing\n"
        "  • source_urls: list of official URLs from IHSA cited in the answer\n\n"
        "- NCAA Division I:\n"
        "  • min_core_course_gpa: the stated minimum core-course GPA for initial eligibility\n"
        "  • total_core_courses: the stated total number of NCAA-approved core courses required\n"
        "  • source_urls: list of official URLs from NCAA/Eligibility Center cited in the answer\n\n"
        "If any field is missing in the answer, set it to null. If no URLs are present for an organization, return an empty list for source_urls."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_ohsaa(evaluator: Evaluator, parent_node, info: OHSAAInfo) -> None:
    org_node = evaluator.add_parallel(
        id="OHSAA_Requirements",
        desc="Ohio High School Athletic Association academic eligibility requirements",
        parent=parent_node,
        critical=False
    )

    # Source group (Critical)
    source_group = evaluator.add_parallel(
        id="OHSAA_Source",
        desc="Provide official OHSAA source URL documenting eligibility requirements",
        parent=org_node,
        critical=True
    )
    src_provided = evaluator.add_custom_node(
        result=bool(info and info.source_urls and len(info.source_urls) > 0),
        id="OHSAA_Source_Provided",
        desc="Official OHSAA eligibility source URL is provided in the answer",
        parent=source_group,
        critical=True
    )
    src_verified = evaluator.add_leaf(
        id="OHSAA_Source_Verified",
        desc="OHSAA source page documents academic eligibility requirements",
        parent=source_group,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is an official OHSAA page that documents academic eligibility requirements for grades 9–12 athletics.",
        node=src_verified,
        sources=info.source_urls,
        additional_instruction="Confirm the page is on ohsaa.org and includes academic eligibility details (e.g., minimum course load)."
    )

    # Standards (Critical)
    standards_node = evaluator.add_parallel(
        id="OHSAA_Standards",
        desc="OHSAA academic standards",
        parent=org_node,
        critical=True
    )

    # Passing requirement (Critical sequential sub-flow)
    pass_seq = evaluator.add_sequential(
        id="OHSAA_Passing_Requirement",
        desc="Identify minimum number of one-credit courses required (5 courses)",
        parent=standards_node,
        critical=True
    )
    pass_provided = evaluator.add_custom_node(
        result=bool(info and info.min_courses_pass_required and str(info.min_courses_pass_required).strip()),
        id="OHSAA_Passing_Requirement_Provided",
        desc="Answer provides a numeric minimum for one-credit courses",
        parent=pass_seq,
        critical=True
    )
    pass_supported = evaluator.add_leaf(
        id="OHSAA_Passing_Requirement_Supported",
        desc="Minimum one-credit course requirement is supported by official OHSAA source",
        parent=pass_seq,
        critical=True
    )
    min_courses_text = (info.min_courses_pass_required or "").strip()
    await evaluator.verify(
        claim=f"OHSAA requires students in grades 9–12 to pass at least {min_courses_text} one-credit courses (or equivalent) each grading period for athletic eligibility.",
        node=pass_supported,
        sources=info.source_urls,
        additional_instruction="Verify the exact minimum course count (should be five). Accept '5' or 'five' as equivalent.",
        extra_prerequisites=[src_provided]
    )

    # GPA policy (Critical sequential sub-flow)
    gpa_seq = evaluator.add_sequential(
        id="OHSAA_GPA_Policy",
        desc="Correctly state that OHSAA has no minimum GPA requirement at state level",
        parent=standards_node,
        critical=True
    )
    gpa_provided = evaluator.add_custom_node(
        result=bool(info and info.gpa_policy and str(info.gpa_policy).strip()),
        id="OHSAA_GPA_Policy_Provided",
        desc="Answer provides a statement about OHSAA GPA policy",
        parent=gpa_seq,
        critical=True
    )
    gpa_supported = evaluator.add_leaf(
        id="OHSAA_GPA_Policy_Supported",
        desc="GPA policy statement is supported by official OHSAA source",
        parent=gpa_seq,
        critical=True
    )
    await evaluator.verify(
        claim="OHSAA does not mandate a statewide minimum GPA for athletic eligibility; GPA standards, if any, are set by local schools or districts.",
        node=gpa_supported,
        sources=info.source_urls,
        additional_instruction="Confirm the OHSAA documentation indicates eligibility is based on course load and does not set a statewide GPA minimum.",
        extra_prerequisites=[src_provided]
    )


async def verify_lhsaa(evaluator: Evaluator, parent_node, info: LHSAAInfo) -> None:
    org_node = evaluator.add_parallel(
        id="LHSAA_Requirements",
        desc="Louisiana High School Athletic Association academic eligibility requirements",
        parent=parent_node,
        critical=False
    )

    # Source group (Critical)
    source_group = evaluator.add_parallel(
        id="LHSAA_Source",
        desc="Provide official LHSAA source URL documenting eligibility requirements",
        parent=org_node,
        critical=True
    )
    src_provided = evaluator.add_custom_node(
        result=bool(info and info.source_urls and len(info.source_urls) > 0),
        id="LHSAA_Source_Provided",
        desc="Official LHSAA eligibility source URL is provided in the answer",
        parent=source_group,
        critical=True
    )
    src_verified = evaluator.add_leaf(
        id="LHSAA_Source_Verified",
        desc="LHSAA source page documents academic eligibility requirements",
        parent=source_group,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is an official LHSAA page that documents academic eligibility requirements.",
        node=src_verified,
        sources=info.source_urls,
        additional_instruction="Confirm the page is on lhsaa.org and includes academic eligibility details.",
    )

    # Standards (Critical)
    standards_node = evaluator.add_parallel(
        id="LHSAA_Standards",
        desc="LHSAA academic standards",
        parent=org_node,
        critical=True
    )

    # Unit requirement (Critical sequential sub-flow)
    unit_seq = evaluator.add_sequential(
        id="LHSAA_Unit_Requirement",
        desc="Identify that 6 units from previous school year are required for first semester eligibility",
        parent=standards_node,
        critical=True
    )
    unit_provided = evaluator.add_custom_node(
        result=bool(info and info.units_required_prev_year_first_sem and str(info.units_required_prev_year_first_sem).strip()),
        id="LHSAA_Unit_Requirement_Provided",
        desc="Answer provides the number of units required for first semester eligibility",
        parent=unit_seq,
        critical=True
    )
    unit_supported = evaluator.add_leaf(
        id="LHSAA_Unit_Requirement_Supported",
        desc="Unit requirement for first semester eligibility is supported by official LHSAA source",
        parent=unit_seq,
        critical=True
    )
    units_text = (info.units_required_prev_year_first_sem or "").strip()
    await evaluator.verify(
        claim=f"For first semester eligibility, LHSAA requires students to have earned {units_text} units from the previous school year.",
        node=unit_supported,
        sources=info.source_urls,
        additional_instruction="Verify that the policy specifies '6 units' for the first semester; allow minor wording differences like 'Carnegie units'.",
        extra_prerequisites=[src_provided]
    )

    # GPA requirement (Critical sequential sub-flow)
    gpa_seq = evaluator.add_sequential(
        id="LHSAA_GPA_Requirement",
        desc="Identify that a 'C' average is required as determined by Local Education Authority",
        parent=standards_node,
        critical=True
    )
    gpa_provided = evaluator.add_custom_node(
        result=bool(info and info.gpa_requirement and str(info.gpa_requirement).strip()),
        id="LHSAA_GPA_Requirement_Provided",
        desc="Answer provides the GPA standard (e.g., 'C' average by LEA)",
        parent=gpa_seq,
        critical=True
    )
    gpa_supported = evaluator.add_leaf(
        id="LHSAA_GPA_Requirement_Supported",
        desc="GPA standard ('C' average by LEA) is supported by official LHSAA source",
        parent=gpa_seq,
        critical=True
    )
    await evaluator.verify(
        claim="LHSAA requires a 'C' average for eligibility as determined by the Local Education Authority (LEA).",
        node=gpa_supported,
        sources=info.source_urls,
        additional_instruction="Confirm the page states that eligibility requires a 'C' average and that the determination is by the LEA.",
        extra_prerequisites=[src_provided]
    )


async def verify_ihsa(evaluator: Evaluator, parent_node, info: IHSAInfo) -> None:
    org_node = evaluator.add_parallel(
        id="IHSA_Requirements",
        desc="Illinois High School Athletic Association academic eligibility requirements",
        parent=parent_node,
        critical=False
    )

    # Source group (Critical)
    source_group = evaluator.add_parallel(
        id="IHSA_Source",
        desc="Provide official IHSA source URL documenting eligibility requirements",
        parent=org_node,
        critical=True
    )
    src_provided = evaluator.add_custom_node(
        result=bool(info and info.source_urls and len(info.source_urls) > 0),
        id="IHSA_Source_Provided",
        desc="Official IHSA eligibility source URL is provided in the answer",
        parent=source_group,
        critical=True
    )
    src_verified = evaluator.add_leaf(
        id="IHSA_Source_Verified",
        desc="IHSA source page documents academic eligibility requirements",
        parent=source_group,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is an official IHSA page that documents academic eligibility requirements.",
        node=src_verified,
        sources=info.source_urls,
        additional_instruction="Confirm the page is on ihsa.org and includes academic eligibility details.",
    )

    # Standards (Critical)
    standards_node = evaluator.add_parallel(
        id="IHSA_Standards",
        desc="IHSA academic standards",
        parent=org_node,
        critical=True
    )

    # Credit hours per week (Critical sequential sub-flow)
    credit_seq = evaluator.add_sequential(
        id="IHSA_Credit_Hours",
        desc="Identify that 25 credit hours per week are required (generally 5 classes)",
        parent=standards_node,
        critical=True
    )
    credit_provided = evaluator.add_custom_node(
        result=bool(info and info.weekly_credit_hours_required and str(info.weekly_credit_hours_required).strip()),
        id="IHSA_Credit_Hours_Provided",
        desc="Answer provides the weekly credit hours requirement",
        parent=credit_seq,
        critical=True
    )
    credit_supported = evaluator.add_leaf(
        id="IHSA_Credit_Hours_Supported",
        desc="Weekly credit hours requirement is supported by official IHSA source",
        parent=credit_seq,
        critical=True
    )
    credit_text = (info.weekly_credit_hours_required or "").strip()
    await evaluator.verify(
        claim=f"IHSA requires students to be passing at least {credit_text} credit hours of high school work per week (generally five classes).",
        node=credit_supported,
        sources=info.source_urls,
        additional_instruction="Verify that IHSA specifies '25 credit hours per week' (allow phrasing such as 'must be passing 25 hours').",
        extra_prerequisites=[src_provided]
    )


async def verify_ncaadi(evaluator: Evaluator, parent_node, info: NCAADIInfo) -> None:
    org_node = evaluator.add_parallel(
        id="NCAA_DI_Requirements",
        desc="NCAA Division I academic eligibility requirements",
        parent=parent_node,
        critical=False
    )

    # Source group (Critical)
    source_group = evaluator.add_parallel(
        id="NCAA_Source",
        desc="Provide official NCAA source URL documenting Division I eligibility requirements",
        parent=org_node,
        critical=True
    )
    src_provided = evaluator.add_custom_node(
        result=bool(info and info.source_urls and len(info.source_urls) > 0),
        id="NCAA_Source_Provided",
        desc="Official NCAA Division I eligibility source URL is provided in the answer",
        parent=source_group,
        critical=True
    )
    src_verified = evaluator.add_leaf(
        id="NCAA_Source_Verified",
        desc="NCAA Division I source page documents academic eligibility requirements",
        parent=source_group,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is an official NCAA page (or Eligibility Center page) that documents Division I initial eligibility academic requirements.",
        node=src_verified,
        sources=info.source_urls,
        additional_instruction="Confirm the page is on ncaa.org or eligibilitycenter.org and includes DI academic eligibility details.",
    )

    # Standards (Critical)
    standards_node = evaluator.add_parallel(
        id="NCAA_Standards",
        desc="NCAA Division I academic standards",
        parent=org_node,
        critical=True
    )

    # Core-course GPA (Critical sequential sub-flow)
    gpa_seq = evaluator.add_sequential(
        id="NCAA_Core_GPA",
        desc="Identify minimum core-course GPA of 2.3",
        parent=standards_node,
        critical=True
    )
    gpa_provided = evaluator.add_custom_node(
        result=bool(info and info.min_core_course_gpa and str(info.min_core_course_gpa).strip()),
        id="NCAA_Core_GPA_Provided",
        desc="Answer provides the minimum core-course GPA",
        parent=gpa_seq,
        critical=True
    )
    gpa_supported = evaluator.add_leaf(
        id="NCAA_Core_GPA_Supported",
        desc="Minimum core-course GPA is supported by official NCAA source",
        parent=gpa_seq,
        critical=True
    )
    min_gpa_text = (info.min_core_course_gpa or "").strip()
    await evaluator.verify(
        claim=f"For initial eligibility in NCAA Division I, the minimum core-course GPA is {min_gpa_text}.",
        node=gpa_supported,
        sources=info.source_urls,
        additional_instruction="Verify that the page states the DI minimum core-course GPA is 2.3.",
        extra_prerequisites=[src_provided]
    )

    # Total core courses (Critical sequential sub-flow)
    core_seq = evaluator.add_sequential(
        id="NCAA_Core_Courses",
        desc="Identify requirement of 16 core courses total",
        parent=standards_node,
        critical=True
    )
    core_provided = evaluator.add_custom_node(
        result=bool(info and info.total_core_courses and str(info.total_core_courses).strip()),
        id="NCAA_Core_Courses_Provided",
        desc="Answer provides the total number of NCAA-approved core courses required",
        parent=core_seq,
        critical=True
    )
    core_supported = evaluator.add_leaf(
        id="NCAA_Core_Courses_Supported",
        desc="Total number of core courses is supported by official NCAA source",
        parent=core_seq,
        critical=True
    )
    total_core_text = (info.total_core_courses or "").strip()
    await evaluator.verify(
        claim=f"For initial eligibility in NCAA Division I, students must complete {total_core_text} NCAA-approved core courses.",
        node=core_supported,
        sources=info.source_urls,
        additional_instruction="Verify that the page states 16 total core courses are required for DI initial eligibility.",
        extra_prerequisites=[src_provided]
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
) -> Dict[str, Any]:
    """
    Evaluate the academic eligibility requirements answer for OHSAA, LHSAA, IHSA, and NCAA Division I.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Organizations are independent checks
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

    # Root aggregator node (non-critical, parallel)
    main_node = evaluator.add_parallel(
        id="Academic_Eligibility_Requirements",
        desc="Complete academic eligibility requirements for four athletic organizations",
        parent=root,
        critical=False
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_eligibility(),
        template_class=EligibilityExtraction,
        extraction_name="eligibility_extraction"
    )

    # Ground truth info (for context in summary)
    evaluator.add_ground_truth(GROUND_TRUTH, gt_type="expected_values")

    # Build verification subtrees
    ohsaa_info = extracted.ohsaa or OHSAAInfo()
    lhsaa_info = extracted.lhsaa or LHSAAInfo()
    ihsa_info = extracted.ihsa or IHSAInfo()
    ncaadi_info = extracted.ncaadi or NCAADIInfo()

    await verify_ohsaa(evaluator, main_node, ohsaa_info)
    await verify_lhsaa(evaluator, main_node, lhsaa_info)
    await verify_ihsa(evaluator, main_node, ihsa_info)
    await verify_ncaadi(evaluator, main_node, ncaadi_info)

    # Return summary
    return evaluator.get_summary()