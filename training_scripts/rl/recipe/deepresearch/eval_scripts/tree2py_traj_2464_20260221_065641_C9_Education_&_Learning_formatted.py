import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "fcs_to_fbs_transition_2023_2026"
TASK_DESCRIPTION = (
    "Identify three universities that began their transition from the NCAA Football Championship Subdivision (FCS) "
    "to the Football Bowl Subdivision (FBS) between 2023 and 2026. For each university, provide comprehensive "
    "information including: University Name, Transition Timeline (start year), Target Conference, Conference Join Date, "
    "NCAA Application Fee, Conference Entry Fee, Transition Period length, Postseason Eligibility, Scholarship "
    "Requirements (min annual count, min annual value, min number of varsity sports, distribution requirements), "
    "Conference Membership Details (membership type, minimum FBS members required, membership term), and valid "
    "Reference URLs supporting the information."
)

# Optional baseline rules to record as ground truth context (not used for verification directly)
NCAA_RULE_BASELINE = {
    "transition_period_years": "2 years (mandatory ineligibility for postseason during the transition)",
    "application_fee_post_2023": "$5 million (NCAA FCS-to-FBS transition application fee for transitions starting in 2023 or later)",
    "scholarships_min_count": "210 scholarships per year",
    "scholarships_min_value": "$6 million in total annual scholarship value",
    "sports_minimum": "16 varsity sports required",
    "scholarship_distribution": "At least 90% of allowable scholarships across 16 sports over a two-year rolling period",
    "conference_min_members": "8 FBS members minimum for conference requirements related to top-tier participation"
}


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class ScholarshipRequirements(BaseModel):
    annual_scholarship_count: Optional[str] = None
    annual_scholarship_value_min: Optional[str] = None
    sports_sponsorship_min: Optional[str] = None
    scholarship_distribution: Optional[str] = None


class ConferenceMembershipDetails(BaseModel):
    membership_type: Optional[str] = None
    conference_minimum_fbs_members: Optional[str] = None
    membership_term: Optional[str] = None


class ProgramInfo(BaseModel):
    university_name: Optional[str] = None
    transition_start_year: Optional[str] = None
    target_conference: Optional[str] = None
    conference_join_date: Optional[str] = None
    ncaa_application_fee: Optional[str] = None
    conference_entry_fee: Optional[str] = None
    transition_period_length: Optional[str] = None
    postseason_eligibility: Optional[str] = None
    scholarship_requirements: ScholarshipRequirements = Field(default_factory=ScholarshipRequirements)
    conference_membership_details: ConferenceMembershipDetails = Field(default_factory=ConferenceMembershipDetails)
    reference_urls: List[str] = Field(default_factory=list)


class TransitionProgramsExtraction(BaseModel):
    programs: List[ProgramInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return (
        "From the provided answer, extract up to three distinct universities that began an FCS-to-FBS transition "
        "between 2023 and 2026 (inclusive). For each university, return the following fields under a 'programs' list:\n"
        "- university_name: The institution's full official name.\n"
        "- transition_start_year: The year the FBS transition officially began.\n"
        "- target_conference: The name of the FBS conference the university is joining.\n"
        "- conference_join_date: The official date when conference membership begins.\n"
        "- ncaa_application_fee: The NCAA application fee amount for the FCS-to-FBS transition (for transitions starting 2023 or later this is commonly $5 million).\n"
        "- conference_entry_fee: The entry fee or payment made to join the conference (if applicable and publicly available).\n"
        "- transition_period_length: The mandatory transition period length (commonly two years).\n"
        "- postseason_eligibility: A clear explanation of the team's postseason eligibility during and after the transition.\n"
        "- scholarship_requirements: An object with the following fields:\n"
        "   * annual_scholarship_count: Minimum annual scholarship count for FBS programs.\n"
        "   * annual_scholarship_value_min: Minimum total annual scholarship value.\n"
        "   * sports_sponsorship_min: Minimum number of varsity sports required.\n"
        "   * scholarship_distribution: Distribution requirements across sports (e.g., 90% across 16 sports over two years).\n"
        "- conference_membership_details: An object with:\n"
        "   * membership_type: Full conference member or football-only affiliate.\n"
        "   * conference_minimum_fbs_members: Minimum number of FBS members required for the conference.\n"
        "   * membership_term: Length or terms of membership agreement (if applicable).\n"
        "- reference_urls: An array of valid URLs cited in the answer that support the above information for the university.\n\n"
        "Rules:\n"
        "1) Extract only what is explicitly present in the answer; do not invent data.\n"
        "2) If a field is missing, set it to null (or empty array for reference_urls).\n"
        "3) If more than three qualifying universities are present, extract the first three.\n"
        "4) Reference URLs must be actual URLs present in the answer; prefer official school, conference, or NCAA sources and credible news.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def program_label(idx: int) -> str:
    if idx == 0:
        return "Program_1"
    if idx == 1:
        return "Program_2"
    return "Program_3"


def program_desc(idx: int) -> str:
    if idx == 0:
        return "First qualifying FCS-to-FBS transition program"
    if idx == 1:
        return "Second qualifying FCS-to-FBS transition program"
    return "Third qualifying FCS-to-FBS transition program"


# --------------------------------------------------------------------------- #
# Verification for one program                                                #
# --------------------------------------------------------------------------- #
async def verify_one_program(
    evaluator: Evaluator,
    parent_node,
    prog: ProgramInfo,
    idx: int,
) -> None:
    """
    Build the verification subtree for one transition program and run checks.
    """
    pfx = f"P{idx + 1}"
    label = program_label(idx)

    # Create the program-level node (parallel aggregation, allow partial credit per program)
    program_node = evaluator.add_parallel(
        id=label,
        desc=program_desc(idx),
        parent=parent_node,
        critical=False
    )

    # Basic existence precheck: require university name and at least one reference URL for meaningful verification
    has_basic = bool(prog.university_name and prog.university_name.strip()) and bool(prog.reference_urls)
    precheck_node = evaluator.add_custom_node(
        result=has_basic,
        id=f"{pfx}_Basic_Info_Provided",
        desc=f"{program_desc(idx)} has university name and at least one reference URL",
        parent=program_node,
        critical=True
    )

    # University Name (Critical)
    name_node = evaluator.add_leaf(
        id=f"{pfx}_University_Name",
        desc=f"Correct name of the {'first' if idx == 0 else ('second' if idx == 1 else 'third')} university",
        parent=program_node,
        critical=True
    )
    name_claim = (
        f"The provided sources indicate that the university involved in this FCS-to-FBS transition program is "
        f"'{prog.university_name}'."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=prog.reference_urls,
        additional_instruction=(
            "Confirm the university name appears on the cited page(s) and is referenced in the context of an FCS-to-FBS transition. "
            "Minor naming variations (e.g., abbreviations) are acceptable."
        ),
    )

    # Transition Timeline (Critical)
    timeline_node = evaluator.add_leaf(
        id=f"{pfx}_Transition_Timeline",
        desc="Transition start year between 2023-2026",
        parent=program_node,
        critical=True
    )
    timeline_claim = (
        f"The FBS transition for {prog.university_name} officially began in {prog.transition_start_year}, "
        f"and that year lies between 2023 and 2026 (inclusive)."
    )
    await evaluator.verify(
        claim=timeline_claim,
        node=timeline_node,
        sources=prog.reference_urls,
        additional_instruction=(
            "Check the announcement or official documentation for the stated start year. "
            "If the year is outside 2023–2026, mark as not supported."
        ),
    )

    # Target Conference (Critical)
    conference_node = evaluator.add_leaf(
        id=f"{pfx}_Target_Conference",
        desc="Name of the FBS conference the university is joining",
        parent=program_node,
        critical=True
    )
    conf_claim = (
        f"{prog.university_name} is joining the FBS conference '{prog.target_conference}'."
    )
    await evaluator.verify(
        claim=conf_claim,
        node=conference_node,
        sources=prog.reference_urls,
        additional_instruction=(
            "Verify that the sources explicitly state the FBS conference to be joined by the university."
        ),
    )

    # Conference Join Date (Non-Critical)
    join_date_node = evaluator.add_leaf(
        id=f"{pfx}_Conference_Join_Date",
        desc="Official date when conference membership begins",
        parent=program_node,
        critical=False
    )
    join_claim = (
        f"The official conference membership for {prog.university_name} begins on {prog.conference_join_date}."
    )
    await evaluator.verify(
        claim=join_claim,
        node=join_date_node,
        sources=prog.reference_urls,
        additional_instruction=(
            "Accept formats like 'effective July 1, 2025' or similar. "
            "If the date is not explicitly given in the sources, mark as not supported."
        ),
    )

    # NCAA Application Fee (Critical)
    ncaa_fee_node = evaluator.add_leaf(
        id=f"{pfx}_NCAA_Application_Fee",
        desc="NCAA fee amount for FCS-to-FBS transition ($5 million for transitions starting 2023 or later)",
        parent=program_node,
        critical=True
    )
    ncaa_fee_claim = (
        "The NCAA application fee for FCS-to-FBS transitions that start in 2023 or later is $5 million."
    )
    await evaluator.verify(
        claim=ncaa_fee_claim,
        node=ncaa_fee_node,
        sources=prog.reference_urls,
        additional_instruction=(
            "Look for NCAA documentation or credible coverage stating the $5 million fee for transitions beginning in 2023 or later."
        ),
    )

    # Conference Entry Fee (Non-Critical)
    conf_fee_node = evaluator.add_leaf(
        id=f"{pfx}_Conference_Entry_Fee",
        desc="Entry fee paid to the conference (amount and currency)",
        parent=program_node,
        critical=False
    )
    conf_fee_claim = (
        f"The conference entry fee associated with {prog.university_name}'s move is '{prog.conference_entry_fee}'."
    )
    await evaluator.verify(
        claim=conf_fee_claim,
        node=conf_fee_node,
        sources=prog.reference_urls,
        additional_instruction=(
            "Verify any stated conference entry fee or payment. If no amount appears on sources, mark as not supported."
        ),
    )

    # Transition Period (Critical)
    transition_period_node = evaluator.add_leaf(
        id=f"{pfx}_Transition_Period",
        desc="Length of mandatory transition period (two years per NCAA rules)",
        parent=program_node,
        critical=True
    )
    transition_period_claim = (
        "The mandatory NCAA transition period for an FCS-to-FBS move is two years."
    )
    await evaluator.verify(
        claim=transition_period_claim,
        node=transition_period_node,
        sources=prog.reference_urls,
        additional_instruction=(
            "Use NCAA documentation or credible sources. The claim should reflect the current rule that the mandatory transition period is two years."
        ),
    )

    # Postseason Eligibility (Critical)
    postseason_node = evaluator.add_leaf(
        id=f"{pfx}_Postseason_Eligibility",
        desc="Postseason eligibility status during and after transition",
        parent=program_node,
        critical=True
    )
    postseason_claim = (
        f"Postseason eligibility for {prog.university_name} during the transition is: {prog.postseason_eligibility}."
    )
    await evaluator.verify(
        claim=postseason_claim,
        node=postseason_node,
        sources=prog.reference_urls,
        additional_instruction=(
            "Confirm whether the team is ineligible for bowl games or other postseason during the transition, and when it becomes eligible after the period."
        ),
    )

    # Scholarship Requirements (Non-Critical, Parallel)
    scholarships_node = evaluator.add_parallel(
        id=f"{pfx}_Scholarship_Requirements",
        desc="Minimum scholarship requirements for FBS programs",
        parent=program_node,
        critical=False
    )

    # Annual Scholarship Count
    sch_count_node = evaluator.add_leaf(
        id=f"{pfx}_Annual_Scholarship_Count",
        desc="Minimum of 210 scholarships per year",
        parent=scholarships_node,
        critical=False
    )
    sch_count_claim = (
        f"The NCAA FBS minimum annual scholarship count is '{prog.scholarship_requirements.annual_scholarship_count}'."
    )
    await evaluator.verify(
        claim=sch_count_claim,
        node=sch_count_node,
        sources=prog.reference_urls,
        additional_instruction=(
            "Look for NCAA rule references or credible sources stating the minimum annual scholarship count (commonly 210)."
        ),
    )

    # Annual Scholarship Value
    sch_value_node = evaluator.add_leaf(
        id=f"{pfx}_Annual_Scholarship_Value",
        desc="Minimum total value of at least $6 million annually",
        parent=scholarships_node,
        critical=False
    )
    sch_value_claim = (
        f"The NCAA FBS minimum total annual scholarship value is '{prog.scholarship_requirements.annual_scholarship_value_min}'."
    )
    await evaluator.verify(
        claim=sch_value_claim,
        node=sch_value_node,
        sources=prog.reference_urls,
        additional_instruction=(
            "Verify that sources state a minimum annual scholarship value around $6 million for FBS programs."
        ),
    )

    # Sports Sponsorship
    sports_min_node = evaluator.add_leaf(
        id=f"{pfx}_Sports_Sponsorship",
        desc="Minimum of 16 varsity sports required",
        parent=scholarships_node,
        critical=False
    )
    sports_min_claim = (
        f"The NCAA FBS minimum number of varsity sports required is '{prog.scholarship_requirements.sports_sponsorship_min}'."
    )
    await evaluator.verify(
        claim=sports_min_claim,
        node=sports_min_node,
        sources=prog.reference_urls,
        additional_instruction=(
            "Confirm that sources state the minimum number of varsity sports required (commonly 16)."
        ),
    )

    # Scholarship Distribution
    sch_dist_node = evaluator.add_leaf(
        id=f"{pfx}_Scholarship_Distribution",
        desc="90% of allowable scholarships across 16 sports over two-year rolling period",
        parent=scholarships_node,
        critical=False
    )
    sch_dist_claim = (
        f"The scholarship distribution requirement is '{prog.scholarship_requirements.scholarship_distribution}'."
    )
    await evaluator.verify(
        claim=sch_dist_claim,
        node=sch_dist_node,
        sources=prog.reference_urls,
        additional_instruction=(
            "Look for language such as 'at least 90% of allowable scholarships across 16 sports over a two-year rolling period'."
        ),
    )

    # Conference Requirements (Non-Critical, Parallel)
    conf_req_node = evaluator.add_parallel(
        id=f"{pfx}_Conference_Requirements",
        desc="Conference membership requirements and structure",
        parent=program_node,
        critical=False
    )

    # Membership Type
    memb_type_node = evaluator.add_leaf(
        id=f"{pfx}_Membership_Type",
        desc="Type of conference membership (full member or football-only)",
        parent=conf_req_node,
        critical=False
    )
    memb_type_claim = (
        f"For this program, the conference membership type is '{prog.conference_membership_details.membership_type}'."
    )
    await evaluator.verify(
        claim=memb_type_claim,
        node=memb_type_node,
        sources=prog.reference_urls,
        additional_instruction=(
            "Verify whether the institution is a full conference member or a football-only affiliate."
        ),
    )

    # Conference Minimum Members
    conf_min_node = evaluator.add_leaf(
        id=f"{pfx}_Conference_Minimum_Members",
        desc="Minimum number of FBS members required for conference (8 for automatic CFP eligibility)",
        parent=conf_req_node,
        critical=False
    )
    conf_min_claim = (
        f"The minimum number of FBS members required for the conference is '{prog.conference_membership_details.conference_minimum_fbs_members}'."
    )
    await evaluator.verify(
        claim=conf_min_claim,
        node=conf_min_node,
        sources=prog.reference_urls,
        additional_instruction=(
            "Verify NCAA or conference rule references stating the minimum number of FBS members (commonly 8)."
        ),
    )

    # Membership Term
    memb_term_node = evaluator.add_leaf(
        id=f"{pfx}_Membership_Term",
        desc="Length or terms of conference membership agreement",
        parent=conf_req_node,
        critical=False
    )
    memb_term_claim = (
        f"The length/terms of the conference membership agreement are: '{prog.conference_membership_details.membership_term}'."
    )
    await evaluator.verify(
        claim=memb_term_claim,
        node=memb_term_node,
        sources=prog.reference_urls,
        additional_instruction=(
            "Check for any stated membership agreement length or term structure. If not present, mark as not supported."
        ),
    )

    # Reference URLs validity/support (Critical)
    refs_node = evaluator.add_leaf(
        id=f"{pfx}_Reference_URLs",
        desc=f"Valid reference URLs supporting all claims about {program_desc(idx)}",
        parent=program_node,
        critical=True
    )
    refs_claim = (
        f"At least one of the provided URLs directly supports that {prog.university_name} is transitioning from FCS to FBS and/or the associated details."
    )
    await evaluator.verify(
        claim=refs_claim,
        node=refs_node,
        sources=prog.reference_urls,
        additional_instruction=(
            "Confirm relevance: official school announcements, NCAA pages, conference releases, or credible news that state the transition/program details."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an agent's answer for the FCS-to-FBS transition research task (2023–2026).
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
        default_model=model,
    )

    # Create top-level research node (non-critical to allow partial credit across programs)
    research_node = evaluator.add_parallel(
        id="FBS_Transition_Research",
        desc="Comprehensive research on universities transitioning from FCS to FBS between 2023-2026, requiring identification of exactly three qualifying programs",
        parent=root,
        critical=False
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=TransitionProgramsExtraction,
        extraction_name="fbs_transitions_extraction",
    )

    # Record baseline rules as ground truth info (for transparency; verification relies on provided URLs)
    evaluator.add_ground_truth({
        "ncaa_rule_baseline": NCAA_RULE_BASELINE,
        "task_window_years": "2023–2026 inclusive"
    })

    # Normalize to exactly 3 programs (pad with empty placeholders if fewer)
    programs = list(extracted.programs)[:3]
    while len(programs) < 3:
        programs.append(ProgramInfo())

    # Build verification subtrees for each program
    for i, prog in enumerate(programs):
        await verify_one_program(evaluator, research_node, prog, i)

    # Return evaluation summary
    return evaluator.get_summary()