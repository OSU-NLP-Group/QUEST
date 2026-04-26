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
TASK_ID = "graduate_education_planning_2026"
TASK_DESCRIPTION = """
You are planning to pursue a Master's degree in Education and want to identify graduate programs that fit your schedule and budget, along with professional development opportunities for 2026.

Part 1: Graduate Programs
Identify two universities that offer Master's degree programs in Education (or directly related education fields) that meet ALL of the following requirements:

1. The program has a minimum GPA requirement of 3.0 or lower for admission
2. The program accepts applications for Fall 2026 enrollment
3. The application deadline for Fall 2026 admission is on or after April 1, 2026
4. The program requires 2-3 letters of recommendation as part of the application
5. The university or program offers graduate assistantships to master's students
6. Graduate assistantships provide tuition waiver, tuition remission, or significant tuition reduction

For each university, provide:
- The official name of the university
- The specific degree program name (e.g., M.Ed. in Curriculum and Instruction)
- The minimum GPA requirement for admission
- The Fall 2026 application deadline
- The number of letters of recommendation required
- Description of assistantship tuition benefits
- A direct link to the program's official webpage
- A direct link to the assistantship or funding information page

Part 2: Professional Development Conferences
Identify two major national or regional education conferences occurring between June and December 2026 that meet ALL of the following requirements:

1. The conference offers early bird registration with a clearly stated deadline
2. The conference provides both member and non-member registration pricing
3. The conference location and dates are publicly available

For each conference, provide:
- The official conference name
- The conference dates
- The conference location (city and state)
- The early bird registration deadline
- The member early bird registration fee
- A direct link to the conference registration or pricing page
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ProgramItem(BaseModel):
    university_name: Optional[str] = None
    program_name: Optional[str] = None
    min_gpa: Optional[str] = None
    accepts_fall_2026: Optional[str] = None
    fall_2026_deadline: Optional[str] = None
    letters_required: Optional[str] = None
    tuition_benefits_desc: Optional[str] = None
    program_url: Optional[str] = None
    assistantship_url: Optional[str] = None
    assistantship_financial_details: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class ProgramsExtraction(BaseModel):
    programs: List[ProgramItem] = Field(default_factory=list)


class ConferenceItem(BaseModel):
    name: Optional[str] = None
    dates_text: Optional[str] = None
    location: Optional[str] = None  # Expect "City, State" or similar
    early_bird_deadline: Optional[str] = None
    member_early_bird_fee: Optional[str] = None
    registration_url: Optional[str] = None
    non_member_pricing_text: Optional[str] = None
    extra_urls: List[str] = Field(default_factory=list)


class ConferencesExtraction(BaseModel):
    conferences: List[ConferenceItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
    Extract up to TWO Master's programs in Education (or directly related education fields) described in the answer.
    For each program, extract these fields exactly as stated in the answer:
    - university_name: official university name
    - program_name: the specific degree/program name (e.g., "M.Ed. in Curriculum and Instruction")
    - min_gpa: the minimum required GPA for admission (string as presented, e.g., "3.0", "2.75", "minimum 3.0")
    - accepts_fall_2026: whether the program accepts applications for Fall 2026 (copy a short phrase from the answer)
    - fall_2026_deadline: the application deadline for Fall 2026 (string date as presented)
    - letters_required: number of letters required (string such as "2", "3", "2-3", "two", etc.)
    - tuition_benefits_desc: description text about assistantship tuition benefits (tuition waiver/remission/reduction)
    - program_url: direct URL to the official program page
    - assistantship_url: direct URL to the assistantship/funding page for master's students
    - assistantship_financial_details: stipend/salary/hourly rate text if present
    - additional_urls: list any other relevant URLs cited for admissions requirements/deadlines/letters
    
    Rules:
    - Only extract URLs that are explicitly present in the answer. Keep them as full URLs if possible.
    - If any field is missing, set it to null (for strings) or an empty list (for additional_urls).
    - Return a JSON object with a 'programs' array of up to 2 items.
    """


def prompt_extract_conferences() -> str:
    return """
    Extract up to TWO education conferences described in the answer (occurring June–December 2026).
    For each conference, extract:
    - name: official conference name
    - dates_text: the stated conference dates as text (e.g., "Nov 12–15, 2026")
    - location: city and state as presented (e.g., "Denver, CO")
    - early_bird_deadline: the early bird registration deadline date as text
    - member_early_bird_fee: the member early-bird registration fee amount as text (e.g., "$299")
    - registration_url: direct link to the conference registration or pricing page
    - non_member_pricing_text: text indicating non-member prices are provided (optionally include an amount if present)
    - extra_urls: any additional URLs for the conference (e.g., event home page)
    
    Rules:
    - Only extract URLs that are explicitly present in the answer. Keep them as full URLs if possible.
    - If any field is missing, set it to null (for strings) or an empty list (for extra_urls).
    - Return a JSON object with a 'conferences' array of up to 2 items.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _sources_list(*args: Optional[str | List[str]]) -> List[str]:
    """Merge urls/urls-list, drop Nones/empties/dups, preserve order."""
    seen = set()
    out: List[str] = []
    for arg in args:
        if arg is None:
            continue
        if isinstance(arg, list):
            for u in arg:
                if _non_empty(u) and u not in seen:
                    out.append(u)
                    seen.add(u)
        else:
            if _non_empty(arg) and arg not in seen:
                out.append(arg)
                seen.add(arg)
    return out


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_program(evaluator: Evaluator, parent_node, program: ProgramItem, index: int) -> None:
    pi = index + 1
    prog_id_prefix = f"Program{pi}_"

    # Top-level node for this program (parallel, non-critical overall)
    prog_node = evaluator.add_parallel(
        id=f"Graduate_Program_{pi}",
        desc=f"{'First' if pi == 1 else 'Second'} identified graduate education program meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # 1) Basic Information (critical group)
    basic_node = evaluator.add_parallel(
        id=f"{prog_id_prefix}Basic_Information",
        desc="Fundamental program details and institution identification",
        parent=prog_node,
        critical=True
    )

    # University Name (presence)
    evaluator.add_custom_node(
        result=_non_empty(program.university_name),
        id=f"{prog_id_prefix}University_Name",
        desc="The official name of the university offering the program is provided",
        parent=basic_node,
        critical=True
    )

    # Degree Type (verify masters-level edu field) - use program_url as source
    degree_leaf = evaluator.add_leaf(
        id=f"{prog_id_prefix}Degree_Type",
        desc="The specific degree program name is provided and is a Master's degree in Education or directly related field (M.Ed., M.A. in Education, or equivalent)",
        parent=basic_node,
        critical=True
    )
    degree_claim = (
        f"The program '{program.program_name}' at '{program.university_name}' is a master's-level program in "
        f"Education or a directly related education field (e.g., M.Ed., M.A.Ed., Ed.M., M.S.Ed., MAT with an "
        f"education specialization)."
    )
    await evaluator.verify(
        claim=degree_claim,
        node=degree_leaf,
        sources=program.program_url,
        additional_instruction=(
            "Confirm the program is a master's degree within Education or a directly related subfield "
            "(e.g., Curriculum & Instruction, Educational Leadership, Special Education, Higher Education, TESOL, "
            "Instructional Design). Accept common degree abbreviations like MEd, M.A. in Education, Ed.M., M.S.Ed., "
            "or MAT when clearly education-focused."
        )
    )

    # Program official page URL (presence)
    evaluator.add_custom_node(
        result=_non_empty(program.program_url),
        id=f"{prog_id_prefix}Reference_URL",
        desc="Direct link to the program's official webpage showing program details is provided",
        parent=basic_node,
        critical=True
    )

    # 2) Admission Requirements (critical group)
    adm_node = evaluator.add_parallel(
        id=f"{prog_id_prefix}Admission_Requirements",
        desc="Admission criteria and application timeline requirements",
        parent=prog_node,
        critical=True
    )

    # Optional existence gates (to avoid unsupported verification without URLs/values)
    evaluator.add_custom_node(
        result=_non_empty(program.min_gpa),
        id=f"{prog_id_prefix}GPA_Requirement_Provided",
        desc="Minimum GPA value provided in the answer",
        parent=adm_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(program.accepts_fall_2026),
        id=f"{prog_id_prefix}Fall_2026_Admission_Provided",
        desc="Statement about Fall 2026 admissions provided in the answer",
        parent=adm_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(program.fall_2026_deadline),
        id=f"{prog_id_prefix}Application_Deadline_Provided",
        desc="A Fall 2026 deadline date is provided in the answer",
        parent=adm_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(program.letters_required),
        id=f"{prog_id_prefix}Letters_of_Recommendation_Provided",
        desc="The number of letters of recommendation is provided in the answer",
        parent=adm_node,
        critical=True
    )

    # Composite sources for admissions checks
    adm_sources = _sources_list(program.program_url, program.additional_urls)

    # GPA Requirement <= 3.0 (verify)
    gpa_leaf = evaluator.add_leaf(
        id=f"{prog_id_prefix}GPA_Requirement",
        desc="The program's stated minimum GPA requirement is provided and is 3.0 or lower",
        parent=adm_node,
        critical=True
    )
    gpa_claim = (
        f"According to the official page(s) for '{program.program_name}' at '{program.university_name}', "
        f"the minimum GPA required for admission is '{program.min_gpa}', and this threshold is 3.0 or lower."
    )
    await evaluator.verify(
        claim=gpa_claim,
        node=gpa_leaf,
        sources=adm_sources,
        additional_instruction=(
            "Check the minimum GPA for graduate admission to this program (or the institutional graduate minimum that "
            "applies to this program). Pass if the minimum is 3.0, 2.75, 2.5, etc. If only a higher minimum like 3.2 or "
            "3.5 is stated, fail. If the page states a minimum 3.0 with possible conditional admits below, still pass."
        )
    )

    # Fall 2026 admission accepted (verify)
    fall_leaf = evaluator.add_leaf(
        id=f"{prog_id_prefix}Fall_2026_Admission",
        desc="The program accepts applications for Fall 2026 enrollment",
        parent=adm_node,
        critical=True
    )
    fall_claim = (
        f"The program '{program.program_name}' at '{program.university_name}' is accepting applications for Fall 2026."
    )
    await evaluator.verify(
        claim=fall_claim,
        node=fall_leaf,
        sources=adm_sources,
        additional_instruction=(
            "Look for explicit mention of 'Fall 2026' admissions or a 2026 admissions cycle with Fall term. "
            "General perennial admissions pages that show term-specific 2026 dates for Fall count as acceptance."
        )
    )

    # Fall 2026 deadline on or after 2026-04-01 (verify)
    deadline_leaf = evaluator.add_leaf(
        id=f"{prog_id_prefix}Application_Deadline",
        desc="The Fall 2026 application deadline is provided and is on or after April 1, 2026",
        parent=adm_node,
        critical=True
    )
    deadline_claim = (
        f"For Fall 2026 admission to '{program.program_name}' at '{program.university_name}', "
        f"the application deadline is '{program.fall_2026_deadline}' and is on or after April 1, 2026."
    )
    await evaluator.verify(
        claim=deadline_claim,
        node=deadline_leaf,
        sources=adm_sources,
        additional_instruction=(
            "Confirm a specific deadline date for Fall 2026. Pass only if the deadline is 2026-04-01 or later. "
            "If multiple rounds exist, any listed Round for Fall 2026 with a date >= 2026-04-01 qualifies."
        )
    )

    # Letters of recommendation 2-3 (verify)
    lors_leaf = evaluator.add_leaf(
        id=f"{prog_id_prefix}Letters_of_Recommendation",
        desc="The number of required letters of recommendation is provided and is 2-3 letters",
        parent=adm_node,
        critical=True
    )
    lors_claim = (
        f"The application to '{program.program_name}' at '{program.university_name}' requires 2 or 3 letters of "
        f"recommendation (the answer states: '{program.letters_required}')."
    )
    await evaluator.verify(
        claim=lors_claim,
        node=lors_leaf,
        sources=adm_sources,
        additional_instruction=(
            "Pass if the requirement is 2 or 3 letters (including ranges like '2–3'). "
            "If the page clearly states 2 letters or clearly states 3 letters, pass. Otherwise fail."
        )
    )

    # 3) Assistantship Information (set to non-critical to allow a non-critical child)
    asst_node = evaluator.add_parallel(
        id=f"{prog_id_prefix}Assistantship_Information",
        desc="Graduate assistantship availability and benefits",
        parent=prog_node,
        critical=False
    )

    # Assistantship Availability (verify)
    asst_sources = _sources_list(program.assistantship_url, program.additional_urls, program.program_url)
    asst_avail_leaf = evaluator.add_leaf(
        id=f"{prog_id_prefix}Assistantship_Availability",
        desc="The university or program offers graduate assistantships to master's students",
        parent=asst_node,
        critical=True
    )
    asst_avail_claim = (
        f"Graduate assistantships (GA/TA/RA) are offered to master's students at '{program.university_name}' "
        f"for the program '{program.program_name}'."
    )
    await evaluator.verify(
        claim=asst_avail_claim,
        node=asst_avail_leaf,
        sources=asst_sources,
        additional_instruction=(
            "Pass if the funding page or program/graduate school page indicates graduate assistantships are available "
            "to graduate/master's students (GA/TA/RA). If only PhD is mentioned and master's are excluded, fail."
        )
    )

    # Tuition Coverage (verify)
    tuition_leaf = evaluator.add_leaf(
        id=f"{prog_id_prefix}Tuition_Coverage",
        desc="A description of assistantship tuition benefits is provided showing that assistantships provide tuition waiver, tuition remission, or significant tuition reduction",
        parent=asst_node,
        critical=True
    )
    tuition_claim = (
        "Graduate assistantships for master's students include tuition waiver, tuition remission, or a significant "
        "tuition reduction."
    )
    await evaluator.verify(
        claim=tuition_claim,
        node=tuition_leaf,
        sources=asst_sources,
        additional_instruction=(
            "Look for explicit benefits such as 'tuition waiver', 'tuition remission', 'in-state tuition provided', "
            "'significant tuition reduction', or similar. If only stipend with no tuition benefit is stated, fail."
        )
    )

    # Assistantship page URL presence
    evaluator.add_custom_node(
        result=_non_empty(program.assistantship_url),
        id=f"{prog_id_prefix}Assistantship_Reference_URL",
        desc="Direct link to the assistantship or funding information page is provided",
        parent=asst_node,
        critical=True
    )

    # Financial details (non-critical verification)
    fin_leaf = evaluator.add_leaf(
        id=f"{prog_id_prefix}Financial_Details",
        desc="The university provides specific information about assistantship stipend, salary, or hourly rate",
        parent=asst_node,
        critical=False
    )
    fin_claim = (
        "The assistantship information page provides a specific stipend amount, salary range, or hourly pay."
    )
    await evaluator.verify(
        claim=fin_claim,
        node=fin_leaf,
        sources=asst_sources,
        additional_instruction=(
            "Pass if the page shows a specific numeric stipend (e.g., $X per semester/year) or an hourly wage/range. "
            "If only generic 'stipend provided' with no numeric detail is given, fail."
        )
    )


async def verify_conference(evaluator: Evaluator, parent_node, conf: ConferenceItem, index: int) -> None:
    ci = index + 1
    conf_id_prefix = f"Conference{ci}_"

    conf_node = evaluator.add_parallel(
        id=f"Conference_{ci}",
        desc=f"{'First' if ci == 1 else 'Second'} identified national/major education conference with complete registration information",
        parent=parent_node,
        critical=False
    )

    # 1) Basic Details (critical)
    basic_node = evaluator.add_parallel(
        id=f"{conf_id_prefix}Basic_Details",
        desc="Conference identification and timing information",
        parent=conf_node,
        critical=True
    )

    # Conference Name presence
    evaluator.add_custom_node(
        result=_non_empty(conf.name),
        id=f"{conf_id_prefix}Name",
        desc="The official name of the education conference is provided",
        parent=basic_node,
        critical=True
    )

    # Dates between June 1 and Dec 31, 2026 (verify)
    dates_leaf = evaluator.add_leaf(
        id=f"{conf_id_prefix}Dates",
        desc="The conference dates are provided and fall between June 1 and December 31, 2026",
        parent=basic_node,
        critical=True
    )
    dates_claim = (
        f"The conference '{conf.name}' occurs between June 1, 2026 and December 31, 2026 inclusive. "
        f"The provided dates are: '{conf.dates_text}'."
    )
    await evaluator.verify(
        claim=dates_claim,
        node=dates_leaf,
        sources=_sources_list(conf.registration_url, conf.extra_urls),
        additional_instruction=(
            "Confirm the event dates fall within 2026-06-01 to 2026-12-31 inclusive. "
            "If the page shows dates entirely outside this range, fail."
        )
    )

    # Location verify
    loc_leaf = evaluator.add_leaf(
        id=f"{conf_id_prefix}Location",
        desc="The city and state where the conference will be held are provided",
        parent=basic_node,
        critical=True
    )
    loc_claim = (
        f"The conference '{conf.name}' will be held in '{conf.location}'."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=_sources_list(conf.registration_url, conf.extra_urls),
        additional_instruction="Verify the posted host city and state."
    )

    # 2) Registration Information (critical)
    reg_node = evaluator.add_parallel(
        id=f"{conf_id_prefix}Registration_Information",
        desc="Conference registration fees and deadlines",
        parent=conf_node,
        critical=True
    )

    # Early bird offered (verify)
    eb_off_leaf = evaluator.add_leaf(
        id=f"{conf_id_prefix}Early_Bird_Offered",
        desc="The conference offers an early bird registration discount",
        parent=reg_node,
        critical=True
    )
    eb_off_claim = "The registration page offers an early bird registration option or discount."
    await evaluator.verify(
        claim=eb_off_claim,
        node=eb_off_leaf,
        sources=conf.registration_url,
        additional_instruction="Look for 'Early Bird' pricing or equivalent discounted early registration tier."
    )

    # Early bird deadline (verify specific date)
    eb_deadline_leaf = evaluator.add_leaf(
        id=f"{conf_id_prefix}Early_Bird_Deadline",
        desc="The early bird registration deadline is provided with a specific date",
        parent=reg_node,
        critical=True
    )
    eb_deadline_claim = f"The early bird registration deadline is '{conf.early_bird_deadline}', a specific calendar date."
    await evaluator.verify(
        claim=eb_deadline_claim,
        node=eb_deadline_leaf,
        sources=conf.registration_url,
        additional_instruction=(
            "Pass if a specific early bird deadline date is shown (e.g., 'September 15, 2026' or '9/15/2026'). "
            "Generic phrases like 'early bird available' without a date should fail."
        )
    )

    # Member early bird fee provided (verify)
    member_fee_leaf = evaluator.add_leaf(
        id=f"{conf_id_prefix}Member_Fee_Provided",
        desc="The member early bird registration fee amount is provided",
        parent=reg_node,
        critical=True
    )
    member_fee_claim = (
        f"The registration page lists a member early bird registration fee amount: '{conf.member_early_bird_fee}'."
    )
    await evaluator.verify(
        claim=member_fee_claim,
        node=member_fee_leaf,
        sources=conf.registration_url,
        additional_instruction="Look for a dollar amount or numeric price specifically for 'member' under the early bird tier."
    )

    # Non-member pricing available (verify presence of any non-member price)
    nonmember_leaf = evaluator.add_leaf(
        id=f"{conf_id_prefix}Nonmember_Pricing_Available",
        desc="The conference provides non-member registration pricing information",
        parent=reg_node,
        critical=True
    )
    nonmember_claim = "The registration/pricing page displays non-member registration pricing."
    await evaluator.verify(
        claim=nonmember_claim,
        node=nonmember_leaf,
        sources=conf.registration_url,
        additional_instruction="Pass if a non-member price is present for any tier (early bird or regular)."
    )

    # Registration URL presence
    evaluator.add_custom_node(
        result=_non_empty(conf.registration_url),
        id=f"{conf_id_prefix}Reference_URL",
        desc="Direct link to the conference's official registration or pricing page is provided",
        parent=reg_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate the answer for the Graduate Education Planning Task (programs + conferences).
    """
    # Initialize evaluator (root is a neutral aggregator)
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

    # Add a task-level node to reflect the rubric root (set to non-critical to allow mixed children)
    task_node = evaluator.add_parallel(
        id="Graduate_Education_Planning_Task",
        desc="Complete evaluation of graduate education programs and professional development opportunities",
        parent=root,
        critical=False
    )

    # Extract structured info
    programs_extraction = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction"
    )

    conferences_extraction = await evaluator.extract(
        prompt=prompt_extract_conferences(),
        template_class=ConferencesExtraction,
        extraction_name="conferences_extraction"
    )

    # Prepare exactly two programs (pad with empty if needed)
    programs: List[ProgramItem] = list(programs_extraction.programs[:2])
    while len(programs) < 2:
        programs.append(ProgramItem())

    # Prepare exactly two conferences (pad with empty if needed)
    conferences: List[ConferenceItem] = list(conferences_extraction.conferences[:2])
    while len(conferences) < 2:
        conferences.append(ConferenceItem())

    # Program 1
    await verify_program(
        evaluator=evaluator,
        parent_node=task_node.add_node if False else evaluator.add_parallel(  # no-op trick to keep type hints happy
            id="noop", desc="noop", parent=task_node, critical=False
        ) if False else task_node,  # just use task_node
        program=programs[0],
        index=0
    )

    # Program 2
    await verify_program(
        evaluator=evaluator,
        parent_node=task_node,
        program=programs[1],
        index=1
    )

    # Conferences group (set non-critical to avoid child critical consistency issues)
    confs_parent = evaluator.add_parallel(
        id="Professional_Development_Conferences",
        desc="Two major education conferences occurring in 2026 with registration details",
        parent=task_node,
        critical=False
    )

    # Conference 1
    await verify_conference(
        evaluator=evaluator,
        parent_node=confs_parent,
        conf=conferences[0],
        index=0
    )

    # Conference 2
    await verify_conference(
        evaluator=evaluator,
        parent_node=confs_parent,
        conf=conferences[1],
        index=1
    )

    # Return evaluation summary
    return evaluator.get_summary()