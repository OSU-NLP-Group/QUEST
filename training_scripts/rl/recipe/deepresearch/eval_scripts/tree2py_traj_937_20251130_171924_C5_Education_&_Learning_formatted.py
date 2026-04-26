import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "college_football_eligibility_pathway_il_to_ncaa_di"
TASK_DESCRIPTION = (
    "A junior football player at a high school in Illinois is planning to pursue NCAA Division I college football. "
    "Provide comprehensive information about: (1) the academic eligibility requirements they must currently maintain "
    "under IHSA rules to continue playing high school football, including both the weekly academic standard and the "
    "semester progression requirement; (2) the NCAA Division I initial eligibility requirements they must meet to be "
    "eligible to compete as college freshmen, including the number of NCAA-approved core courses required, the timeline "
    "for completing these courses, and the minimum core-course GPA; and (3) information about athletic scholarship "
    "availability across different types of Division I football programs, specifically addressing the scholarship "
    "policies for FBS programs, FCS programs, and Ivy League schools."
)

GROUND_TRUTH = {
    "ihsa_weekly": "Pass at least 5 half-credit classes or 25 credit hours each week to remain eligible.",
    "ihsa_semester": "Must pass required courses each semester (e.g., 25 hours) to remain eligible for the next semester.",
    "ncaa_core_courses_count": "16 NCAA-approved core courses are required.",
    "ncaa_core_timeline": "Core courses must be completed within 8 academic semesters (4 consecutive years) from the start of 9th grade.",
    "ncaa_min_core_gpa": "Minimum core-course GPA of 2.3 on a 4.0 scale.",
    "fbs_current": "FBS programs can offer up to 85 football scholarships.",
    "fbs_future": "FBS scholarship cap expanding to 105 in 2025-26.",
    "fcs": "FCS programs can offer up to 63 total football scholarships.",
    "ivy": "Ivy League schools do not offer athletic scholarships; only need-based financial aid is available."
}


class EligibilityPathwayExtraction(BaseModel):
    # IHSA weekly and semester requirements
    ihsa_weekly_requirement: Optional[str] = None
    ihsa_weekly_value: Optional[str] = None
    ihsa_weekly_sources: List[str] = Field(default_factory=list)

    ihsa_semester_requirement: Optional[str] = None
    ihsa_semester_sources: List[str] = Field(default_factory=list)

    # NCAA Division I initial eligibility: core courses, timeline, GPA
    ncaa_core_courses_count: Optional[str] = None
    ncaa_core_courses_count_sources: List[str] = Field(default_factory=list)

    ncaa_core_courses_timeline: Optional[str] = None
    ncaa_core_courses_timeline_sources: List[str] = Field(default_factory=list)

    ncaa_min_core_gpa: Optional[str] = None
    ncaa_min_core_gpa_sources: List[str] = Field(default_factory=list)

    # Division I scholarship availability
    fbs_scholarship_policy: Optional[str] = None
    fbs_scholarship_count: Optional[str] = None  # current count stated
    fbs_future_scholarship_count: Optional[str] = None  # future/expansion count stated
    fbs_sources: List[str] = Field(default_factory=list)

    fcs_scholarship_policy: Optional[str] = None
    fcs_scholarship_count: Optional[str] = None
    fcs_sources: List[str] = Field(default_factory=list)

    ivy_league_policy: Optional[str] = None
    ivy_sources: List[str] = Field(default_factory=list)


def prompt_extract_pathway() -> str:
    return """
    Extract the specific information provided in the answer related to Illinois high school football eligibility (IHSA), NCAA Division I initial eligibility, and Division I scholarship policies. Capture the statements exactly as written and extract any explicit URLs cited as sources for each item.

    Return a single JSON object with the following fields:

    IHSA weekly eligibility:
    - ihsa_weekly_requirement: The exact weekly academic requirement phrasing the answer provides (e.g., “students must pass at least 25 credit hours” or “pass at least five classes each week”).
    - ihsa_weekly_value: The numeric or textual value the answer uses to quantify the weekly requirement (e.g., “25 credit hours”, “5 classes”, “five half-credit classes”), if present.
    - ihsa_weekly_sources: An array of all URLs explicitly cited for the weekly IHSA requirement.

    IHSA semester progression:
    - ihsa_semester_requirement: The exact semester-to-semester progression requirement phrasing the answer provides (e.g., “must pass courses at the end of each semester to remain eligible for the next”).
    - ihsa_semester_sources: An array of all URLs explicitly cited for the IHSA semester progression requirement.

    NCAA Division I initial eligibility:
    - ncaa_core_courses_count: The answer’s statement of the required number of NCAA-approved core courses (e.g., “16 core courses”).
    - ncaa_core_courses_count_sources: URLs cited specifically for the core course count.
    - ncaa_core_courses_timeline: The timeline statement for completing the core courses (e.g., “within 8 academic semesters or 4 consecutive years from the start of 9th grade”).
    - ncaa_core_courses_timeline_sources: URLs cited specifically for the timeline requirement.
    - ncaa_min_core_gpa: The minimum core-course GPA statement (e.g., “2.3 core-course GPA on a 4.0 scale”).
    - ncaa_min_core_gpa_sources: URLs cited specifically for the minimum GPA requirement.

    Division I scholarship availability:
    - fbs_scholarship_policy: The answer’s text describing FBS scholarship policy.
    - fbs_scholarship_count: The current FBS scholarship number stated in the answer (e.g., “85” or “85 scholarships”), if present.
    - fbs_future_scholarship_count: Any future/expansion number stated (e.g., “105 in 2025-26”), if present.
    - fbs_sources: URLs cited for the FBS scholarship policy/numbers.
    - fcs_scholarship_policy: The answer’s text describing FCS scholarship policy.
    - fcs_scholarship_count: The FCS scholarship number stated (e.g., “63” or “63 scholarships”), if present.
    - fcs_sources: URLs cited for the FCS scholarship policy/numbers.
    - ivy_league_policy: The answer’s text for Ivy League scholarship policy.
    - ivy_sources: URLs cited for Ivy League scholarship policy.

    Rules:
    - Extract only information explicitly present in the answer. Do not infer or invent.
    - For URLs, include only actual URLs present; valid formats include plain URLs or markdown links. If no URL is given for an item, return an empty array for that item.
    - If a field isn’t mentioned in the answer, set it to null (or empty array for URLs).
    """


async def verify_ihsa_requirements(
    evaluator: Evaluator,
    parent_node,
    data: EligibilityPathwayExtraction
) -> None:
    ihsa_node = evaluator.add_parallel(
        id="IHSA_High_School_Requirements",
        desc="Current eligibility requirements under Illinois High School Association (IHSA) rules for high school football participation",
        parent=parent_node,
        critical=True
    )

    weekly_node = evaluator.add_sequential(
        id="Weekly_Academic_Standard",
        desc="Weekly academic credit requirement: students must pass at least 5 half-credit classes (or 25 credit hours equivalent) per week",
        parent=ihsa_node,
        critical=True
    )
    weekly_exists = evaluator.add_custom_node(
        result=(bool(data.ihsa_weekly_requirement) and bool(data.ihsa_weekly_sources)),
        id="ihsa_weekly_exists",
        desc="IHSA weekly requirement is stated and includes cited sources",
        parent=weekly_node,
        critical=True
    )
    weekly_value_node = evaluator.add_leaf(
        id="ihsa_weekly_value_correct",
        desc="The answer correctly states the IHSA weekly standard (≥ five half-credit classes or 25 credit hours per week)",
        parent=weekly_node,
        critical=True
    )
    weekly_value_claim = (
        f"The answer’s IHSA weekly eligibility requirement ('{data.ihsa_weekly_requirement or ''}' "
        f"{'with value ' + data.ihsa_weekly_value if data.ihsa_weekly_value else ''}) is equivalent to: "
        "students must pass at least five half-credit classes or 25 credit hours per week."
    )
    await evaluator.verify(
        claim=weekly_value_claim,
        node=weekly_value_node,
        additional_instruction="Determine whether the answer’s phrasing and numeric value match the canonical IHSA weekly standard (≥5 classes or 25 credit hours)."
    )
    weekly_source_node = evaluator.add_leaf(
        id="ihsa_weekly_sources_support",
        desc="IHSA weekly requirement is supported by the cited sources",
        parent=weekly_node,
        critical=True
    )
    await evaluator.verify(
        claim="IHSA weekly eligibility requires passing at least 25 credit hours (equivalent to five half-credit classes) each week to remain eligible.",
        node=weekly_source_node,
        sources=data.ihsa_weekly_sources,
        additional_instruction="Confirm that at least one cited IHSA or authoritative school policy page explicitly supports the weekly 25-credit-hours (or five classes) requirement."
    )

    semester_node = evaluator.add_sequential(
        id="Semester_Progression_Standard",
        desc="Semester-to-semester requirement: students must pass courses at the end of each semester to remain eligible for the next semester",
        parent=ihsa_node,
        critical=True
    )
    semester_exists = evaluator.add_custom_node(
        result=(bool(data.ihsa_semester_requirement) and bool(data.ihsa_semester_sources)),
        id="ihsa_semester_exists",
        desc="IHSA semester progression requirement is stated and includes cited sources",
        parent=semester_node,
        critical=True
    )
    semester_value_node = evaluator.add_leaf(
        id="ihsa_semester_value_correct",
        desc="The answer correctly states the IHSA semester-to-semester progression requirement",
        parent=semester_node,
        critical=True
    )
    semester_value_claim = (
        f"The answer’s IHSA semester progression requirement ('{data.ihsa_semester_requirement or ''}') "
        "is equivalent to: students must pass required courses each semester to be eligible for the next semester."
    )
    await evaluator.verify(
        claim=semester_value_claim,
        node=semester_value_node,
        additional_instruction="Judge equivalence with IHSA’s semester-based eligibility progression (passing sufficient courses each semester to remain eligible next semester)."
    )
    semester_source_node = evaluator.add_leaf(
        id="ihsa_semester_sources_support",
        desc="IHSA semester progression requirement is supported by the cited sources",
        parent=semester_node,
        critical=True
    )
    await evaluator.verify(
        claim="IHSA requires students to have passed required coursework at the end of a semester to be eligible in the subsequent semester.",
        node=semester_source_node,
        sources=data.ihsa_semester_sources,
        additional_instruction="Confirm that a cited IHSA by-law or school policy page explicitly ties semester completion/passing to next-semester eligibility."
    )


async def verify_ncaa_initial_eligibility(
    evaluator: Evaluator,
    parent_node,
    data: EligibilityPathwayExtraction
) -> None:
    ncaa_node = evaluator.add_parallel(
        id="NCAA_Division_I_Initial_Eligibility",
        desc="NCAA Division I initial eligibility standards that high school students must meet to be eligible to compete as freshmen",
        parent=parent_node,
        critical=True
    )

    count_node = evaluator.add_sequential(
        id="Core_Course_Count_Requirement",
        desc="Core course requirement: must complete 16 NCAA-approved core courses",
        parent=ncaa_node,
        critical=True
    )
    count_exists = evaluator.add_custom_node(
        result=(bool(data.ncaa_core_courses_count) and bool(data.ncaa_core_courses_count_sources)),
        id="ncaa_count_exists",
        desc="The answer states the NCAA core course count and provides sources",
        parent=count_node,
        critical=True
    )
    count_value_node = evaluator.add_leaf(
        id="ncaa_count_value_correct",
        desc="The answer correctly states the count: 16 NCAA-approved core courses",
        parent=count_node,
        critical=True
    )
    count_value_claim = (
        f"The answer’s NCAA core course count ('{data.ncaa_core_courses_count or ''}') "
        "is correct: 16 core courses are required for Division I initial eligibility."
    )
    await evaluator.verify(
        claim=count_value_claim,
        node=count_value_node,
        additional_instruction="Compare the count stated in the answer to the canonical requirement of 16 core courses."
    )
    count_source_node = evaluator.add_leaf(
        id="ncaa_count_sources_support",
        desc="The 16 core courses requirement is supported by cited sources",
        parent=count_node,
        critical=True
    )
    await evaluator.verify(
        claim="NCAA Division I initial eligibility requires completion of 16 NCAA-approved core courses.",
        node=count_source_node,
        sources=data.ncaa_core_courses_count_sources,
        additional_instruction="Verify that the NCAA Eligibility Center or authoritative NCAA documentation confirms the 16 core course requirement."
    )

    timeline_node = evaluator.add_sequential(
        id="Core_Course_Timeline",
        desc="Timeline requirement: the 16 core courses must be completed within 8 academic semesters or 4 consecutive academic years from the start of 9th grade",
        parent=ncaa_node,
        critical=True
    )
    timeline_exists = evaluator.add_custom_node(
        result=(bool(data.ncaa_core_courses_timeline) and bool(data.ncaa_core_courses_timeline_sources)),
        id="ncaa_timeline_exists",
        desc="The answer states the NCAA core course timeline and provides sources",
        parent=timeline_node,
        critical=True
    )
    timeline_value_node = evaluator.add_leaf(
        id="ncaa_timeline_value_correct",
        desc="The answer correctly states the core course timeline (within 8 semesters/4 consecutive years from starting 9th grade)",
        parent=timeline_node,
        critical=True
    )
    timeline_value_claim = (
        f"The answer’s NCAA core course timeline ('{data.ncaa_core_courses_timeline or ''}') "
        "is equivalent to: completion within 8 academic semesters (4 consecutive years) from the start of 9th grade."
    )
    await evaluator.verify(
        claim=timeline_value_claim,
        node=timeline_value_node,
        additional_instruction="Judge equivalence with the canonical NCAA timeline framing (8 semesters / 4 consecutive years from grade 9 start)."
    )
    timeline_source_node = evaluator.add_leaf(
        id="ncaa_timeline_sources_support",
        desc="The core course timeline requirement is supported by cited sources",
        parent=timeline_node,
        critical=True
    )
    await evaluator.verify(
        claim="NCAA requires core courses to be completed within eight academic semesters or four consecutive years from the start of ninth grade.",
        node=timeline_source_node,
        sources=data.ncaa_core_courses_timeline_sources,
        additional_instruction="Confirm the timeline statement is explicitly supported by an NCAA Eligibility Center or equivalent authoritative page."
    )

    gpa_node = evaluator.add_sequential(
        id="Minimum_Core_GPA",
        desc="GPA requirement: must achieve a minimum core-course GPA of 2.3 (calculated on a 4.000 scale) in the 16 core courses",
        parent=ncaa_node,
        critical=True
    )
    gpa_exists = evaluator.add_custom_node(
        result=(bool(data.ncaa_min_core_gpa) and bool(data.ncaa_min_core_gpa_sources)),
        id="ncaa_gpa_exists",
        desc="The answer states the minimum core-course GPA and provides sources",
        parent=gpa_node,
        critical=True
    )
    gpa_value_node = evaluator.add_leaf(
        id="ncaa_gpa_value_correct",
        desc="The answer correctly states the minimum core-course GPA: 2.3 (on a 4.0 scale)",
        parent=gpa_node,
        critical=True
    )
    gpa_value_claim = (
        f"The answer’s minimum core-course GPA ('{data.ncaa_min_core_gpa or ''}') "
        "is correct: at least 2.3 on a 4.0 scale is required for NCAA Division I initial eligibility."
    )
    await evaluator.verify(
        claim=gpa_value_claim,
        node=gpa_value_node,
        additional_instruction="Compare the answer’s GPA statement with the canonical requirement of minimum 2.3 core-course GPA (4.0 scale)."
    )
    gpa_source_node = evaluator.add_leaf(
        id="ncaa_gpa_sources_support",
        desc="The minimum core-course GPA requirement is supported by cited sources",
        parent=gpa_node,
        critical=True
    )
    await evaluator.verify(
        claim="NCAA Division I initial eligibility requires a minimum core-course GPA of 2.3 on a 4.0 scale.",
        node=gpa_source_node,
        sources=data.ncaa_min_core_gpa_sources,
        additional_instruction="Confirm the 2.3 core-course GPA threshold is explicitly stated by NCAA or the Eligibility Center."
    )


async def verify_division_i_scholarships(
    evaluator: Evaluator,
    parent_node,
    data: EligibilityPathwayExtraction
) -> None:
    scholarships_node = evaluator.add_parallel(
        id="Division_I_Scholarship_Availability",
        desc="Information about which types of NCAA Division I football programs can offer athletic scholarships and their scholarship limits",
        parent=parent_node,
        critical=True
    )

    fbs_node = evaluator.add_sequential(
        id="FBS_Scholarship_Policy",
        desc="Division I FBS scholarship policy: FBS programs can offer up to 85 football scholarships (expanding to 105 in 2025-26)",
        parent=scholarships_node,
        critical=True
    )
    fbs_exists = evaluator.add_custom_node(
        result=(bool(data.fbs_scholarship_policy) and bool(data.fbs_sources)),
        id="fbs_policy_exists",
        desc="The answer states FBS scholarship policy and provides sources",
        parent=fbs_node,
        critical=True
    )
    fbs_current_value_node = evaluator.add_leaf(
        id="fbs_current_value_correct",
        desc="The answer correctly states current FBS scholarship limit: 85",
        parent=fbs_node,
        critical=True
    )
    fbs_current_value_claim = (
        f"The answer’s FBS scholarship statement ('{data.fbs_scholarship_policy or ''}' "
        f"{'with number ' + data.fbs_scholarship_count if data.fbs_scholarship_count else ''}) "
        "is correct about current limit: FBS programs can offer up to 85 football scholarships."
    )
    await evaluator.verify(
        claim=fbs_current_value_claim,
        node=fbs_current_value_node,
        additional_instruction="Judge whether the answer affirms the 85-scholarship cap for FBS programs."
    )
    fbs_current_source_node = evaluator.add_leaf(
        id="fbs_current_sources_support",
        desc="The 85-scholarship FBS limit is supported by cited sources",
        parent=fbs_node,
        critical=True
    )
    await evaluator.verify(
        claim="FBS football programs can offer up to 85 athletic scholarships.",
        node=fbs_current_source_node,
        sources=data.fbs_sources,
        additional_instruction="Confirm that a cited NCAA or conference/official source explicitly supports the 85-scholarship cap."
    )
    fbs_future_exists = evaluator.add_custom_node(
        result=bool(data.fbs_future_scholarship_count),
        id="fbs_future_exists",
        desc="The answer addresses the FBS expansion to 105 in 2025-26",
        parent=fbs_node,
        critical=True
    )
    fbs_future_value_node = evaluator.add_leaf(
        id="fbs_future_value_correct",
        desc="The answer correctly states an expansion to 105 scholarships in 2025-26",
        parent=fbs_node,
        critical=True
    )
    fbs_future_value_claim = (
        f"The answer’s FBS future scholarship statement includes '{data.fbs_future_scholarship_count or ''}', "
        "which is correct about an expansion to 105 scholarships in 2025-26."
    )
    await evaluator.verify(
        claim=fbs_future_value_claim,
        node=fbs_future_value_node,
        additional_instruction="Judge whether the answer explicitly notes an expansion to 105 scholarships in the 2025-26 period."
    )
    fbs_future_source_node = evaluator.add_leaf(
        id="fbs_future_sources_support",
        desc="The FBS expansion to 105 is supported by cited sources",
        parent=fbs_node,
        critical=True
    )
    await evaluator.verify(
        claim="FBS football scholarship limit will expand to 105 in 2025-26.",
        node=fbs_future_source_node,
        sources=data.fbs_sources,
        additional_instruction="Confirm that at least one cited authoritative source explicitly discusses the expansion to 105 in 2025-26."
    )

    fcs_node = evaluator.add_sequential(
        id="FCS_Scholarship_Policy",
        desc="Division I FCS scholarship policy: FCS programs can offer up to 63 total football scholarships",
        parent=scholarships_node,
        critical=True
    )
    fcs_exists = evaluator.add_custom_node(
        result=(bool(data.fcs_scholarship_policy) and bool(data.fcs_sources)),
        id="fcs_policy_exists",
        desc="The answer states FCS scholarship policy and provides sources",
        parent=fcs_node,
        critical=True
    )
    fcs_value_node = evaluator.add_leaf(
        id="fcs_value_correct",
        desc="The answer correctly states the FCS scholarship limit: 63 total scholarships",
        parent=fcs_node,
        critical=True
    )
    fcs_value_claim = (
        f"The answer’s FCS scholarship statement ('{data.fcs_scholarship_policy or ''}' "
        f"{'with number ' + data.fcs_scholarship_count if data.fcs_scholarship_count else ''}) "
        "is correct: FCS programs can offer up to 63 total football scholarships."
    )
    await evaluator.verify(
        claim=fcs_value_claim,
        node=fcs_value_node,
        additional_instruction="Judge whether the answer affirms the 63-scholarship cap for FCS programs."
    )
    fcs_source_node = evaluator.add_leaf(
        id="fcs_sources_support",
        desc="The FCS 63-scholarship limit is supported by cited sources",
        parent=fcs_node,
        critical=True
    )
    await evaluator.verify(
        claim="FCS football programs can offer up to 63 athletic scholarships.",
        node=fcs_source_node,
        sources=data.fcs_sources,
        additional_instruction="Confirm that a cited NCAA or conference/official source explicitly supports the 63-scholarship limit."
    )

    ivy_node = evaluator.add_sequential(
        id="Ivy_League_Athletic_Scholarship_Policy",
        desc="Ivy League policy: Ivy League schools are Division I but do NOT offer athletic scholarships; they offer only need-based financial aid",
        parent=scholarships_node,
        critical=True
    )
    ivy_exists = evaluator.add_custom_node(
        result=(bool(data.ivy_league_policy) and bool(data.ivy_sources)),
        id="ivy_policy_exists",
        desc="The answer states Ivy League scholarship policy and provides sources",
        parent=ivy_node,
        critical=True
    )
    ivy_value_node = evaluator.add_leaf(
        id="ivy_value_correct",
        desc="The answer correctly states that Ivy League schools do not offer athletic scholarships (need-based aid only)",
        parent=ivy_node,
        critical=True
    )
    ivy_value_claim = (
        f"The answer’s Ivy League statement ('{data.ivy_league_policy or ''}') "
        "is correct: Ivy League schools do not offer athletic scholarships and provide only need-based aid."
    )
    await evaluator.verify(
        claim=ivy_value_claim,
        node=ivy_value_node,
        additional_instruction="Judge equivalence with the canonical Ivy League policy of no athletic scholarships and need-based financial aid only."
    )
    ivy_source_node = evaluator.add_leaf(
        id="ivy_sources_support",
        desc="Ivy League scholarship policy (no athletic scholarships) is supported by cited sources",
        parent=ivy_node,
        critical=True
    )
    await evaluator.verify(
        claim="Ivy League institutions do not offer athletic scholarships; only need-based financial aid is available.",
        node=ivy_source_node,
        sources=data.ivy_sources,
        additional_instruction="Confirm that an official Ivy League or institutional financial aid policy page explicitly states no athletic scholarships."
    )


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

    extraction = await evaluator.extract(
        prompt=prompt_extract_pathway(),
        template_class=EligibilityPathwayExtraction,
        extraction_name="eligibility_pathway_extraction"
    )

    evaluator.add_ground_truth(
        {
            "ihsa": {
                "weekly": GROUND_TRUTH["ihsa_weekly"],
                "semester": GROUND_TRUTH["ihsa_semester"]
            },
            "ncaa": {
                "core_courses": GROUND_TRUTH["ncaa_core_courses_count"],
                "timeline": GROUND_TRUTH["ncaa_core_timeline"],
                "min_gpa": GROUND_TRUTH["ncaa_min_core_gpa"]
            },
            "scholarships": {
                "fbs_current": GROUND_TRUTH["fbs_current"],
                "fbs_future": GROUND_TRUTH["fbs_future"],
                "fcs": GROUND_TRUTH["fcs"],
                "ivy": GROUND_TRUTH["ivy"]
            }
        },
        gt_type="ground_truth"
    )

    college_node = evaluator.add_parallel(
        id="College_Football_Eligibility_Pathway",
        desc="Complete information about the pathway from Illinois high school football to NCAA Division I college football, including current high school requirements, NCAA initial eligibility standards, and Division I scholarship availability",
        parent=root,
        critical=True
    )

    await verify_ihsa_requirements(evaluator, college_node, extraction)
    await verify_ncaa_initial_eligibility(evaluator, college_node, extraction)
    await verify_division_i_scholarships(evaluator, college_node, extraction)

    return evaluator.get_summary()