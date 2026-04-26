import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "wake_co_pathway_to_engineering_with_coop"
TASK_DESCRIPTION = (
    "A high school sophomore in Wake County, North Carolina, is planning their comprehensive educational pathway to prepare for a career in engineering. "
    "They want to maximize their college credit accumulation while in high school and eventually attend a university with a cooperative education (co-op) program.\n\n"
    "Student's Profile and Goals:\n"
    "- Currently enrolled in Wake County Public Schools with a 3.7 GPA\n"
    "- Has a younger sibling who will attend the same school district\n"
    "- Wants to apply to specialized high school programs for junior and senior years\n"
    "- Plans to earn college credits before graduating high school\n"
    "- Aims to attend a regionally accredited university with mandatory co-op program\n"
    "- Wants to complete bachelor's degree in engineering (minimum 120 credits)\n\n"
    "Required Information:\n"
    "Identify and provide complete details for an educational pathway that includes:\n"
    "1. A specific Wake County magnet school or specialized program that offers opportunities to earn college credits\n"
    "2. The program's application process, including timeline and selection criteria\n"
    "3. All core requirements for program completion\n"
    "4. The maximum college credits or credentials achievable\n"
    "5. Requirements for participating in a university co-op program\n"
    "6. Credit transfer policies from high school to university\n"
    "7. Final degree completion requirements at the university level\n\n"
    "For each component, provide:\n"
    "- Specific program/school names\n"
    "- Exact dates and numerical requirements\n"
    "- Complete requirement lists\n"
    "- Supporting reference URLs from official sources"
)


# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class HSProgramInfo(BaseModel):
    program_name: Optional[str] = None
    program_type: Optional[str] = None  # e.g., "magnet", "CTE", "early college", "dual enrollment"
    offers_college_credit_text: Optional[str] = None
    official_urls: List[str] = Field(default_factory=list)


class ApplicationProcessInfo(BaseModel):
    application_window_text: Optional[str] = None
    application_window_urls: List[str] = Field(default_factory=list)
    results_timing_text: Optional[str] = None
    results_timing_urls: List[str] = Field(default_factory=list)
    selection_method_text: Optional[str] = None
    selection_method_urls: List[str] = Field(default_factory=list)
    selection_criteria_text: Optional[str] = None
    selection_criteria_urls: List[str] = Field(default_factory=list)


class HSCompletionRequirementsInfo(BaseModel):
    hs_grad_requirements_text: Optional[str] = None
    hs_grad_requirements_urls: List[str] = Field(default_factory=list)
    program_structure_text: Optional[str] = None
    program_structure_urls: List[str] = Field(default_factory=list)
    college_coursework_requirements_text: Optional[str] = None
    college_coursework_requirements_urls: List[str] = Field(default_factory=list)


class MaxCreditsInfo(BaseModel):
    max_credits_text: Optional[str] = None
    max_credits_urls: List[str] = Field(default_factory=list)


class UniversityCoopInfo(BaseModel):
    university_name: Optional[str] = None
    accreditation_text: Optional[str] = None
    accreditation_urls: List[str] = Field(default_factory=list)
    coop_available_text: Optional[str] = None
    coop_available_urls: List[str] = Field(default_factory=list)
    coop_mandatory_text: Optional[str] = None
    coop_mandatory_urls: List[str] = Field(default_factory=list)
    coop_prereq_three_semesters_text: Optional[str] = None
    coop_three_semesters_urls: List[str] = Field(default_factory=list)
    coop_prereq_min_gpa_text: Optional[str] = None
    coop_min_gpa_urls: List[str] = Field(default_factory=list)
    coop_prereq_full_time_text: Optional[str] = None
    coop_full_time_urls: List[str] = Field(default_factory=list)
    coop_duration_text: Optional[str] = None
    coop_duration_urls: List[str] = Field(default_factory=list)
    coop_hours_text: Optional[str] = None
    coop_hours_urls: List[str] = Field(default_factory=list)


class TransferPoliciesInfo(BaseModel):
    transfer_principle_text: Optional[str] = None
    transfer_principle_urls: List[str] = Field(default_factory=list)
    dual_enrollment_counts_text: Optional[str] = None
    dual_enrollment_urls: List[str] = Field(default_factory=list)
    university_specific_transfer_policy_text: Optional[str] = None
    university_specific_transfer_urls: List[str] = Field(default_factory=list)


class DegreeCompletionInfo(BaseModel):
    min_total_credits_text: Optional[str] = None
    upper_division_credits_text: Optional[str] = None
    graduation_min_gpa_text: Optional[str] = None
    degree_requirements_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hs_program() -> str:
    return """
    Extract the identified Wake County high school program from the answer.
    Fields:
    - program_name: The specific WCPSS magnet/specialized/early-college/CTE/dual enrollment program named.
    - program_type: Brief type label (e.g., "magnet", "early college", "CTE", "dual enrollment").
    - offers_college_credit_text: The exact statement from the answer about earning college credit/credentials in high school.
    - official_urls: All official URLs cited for this program (WCPSS, school/program page, partner college page, or NC DPI). Return an array of URLs.
    If a field is missing, return null or an empty array for URLs.
    """


def prompt_extract_application_process() -> str:
    return """
    Extract the application process information for the chosen WCPSS program.
    Fields:
    - application_window_text: The exact dates/window as written (e.g., "October 15–January 22").
    - application_window_urls: All official URLs supporting the window.
    - results_timing_text: The exact statement on when results are released (e.g., "starting February 19–20").
    - results_timing_urls: All official URLs supporting results timing.
    - selection_method_text: The method (e.g., "weighted lottery; not first-come-first-served").
    - selection_method_urls: All official URLs supporting selection method.
    - selection_criteria_text: The district/program-stated selection criteria/priority factors OR an explicit statement that none beyond lottery are specified.
    - selection_criteria_urls: All official URLs supporting the criteria (or the explicit absence of additional criteria).
    Return null for missing fields and empty arrays when no URLs are provided.
    """


def prompt_extract_hs_completion_requirements() -> str:
    return """
    Extract the core completion requirements for the chosen high school program.
    Fields:
    - hs_grad_requirements_text: Statement that WCPSS/NC high school graduation requirements must be completed (in addition to any college coursework), if stated.
    - hs_grad_requirements_urls: All official URLs supporting graduation requirements for the program or district.
    - program_structure_text: Statement of the program duration/structure. If the program is an Early College, capture that it is a four-year program beginning in 9th grade.
    - program_structure_urls: Official URLs supporting that structure/duration.
    - college_coursework_requirements_text: The program’s college-coursework participation/academic requirements for completion (if specified).
    - college_coursework_requirements_urls: Official URLs supporting these coursework requirements.
    """


def prompt_extract_max_credits() -> str:
    return """
    Extract the maximum college credits or credential achievable through the high school program.
    Fields:
    - max_credits_text: Exact statement on maximum credits/credential (e.g., "up to 60 credits" or "associate degree").
    - max_credits_urls: Official URLs supporting this maximum credit/credential statement.
    """


def prompt_extract_university_coop() -> str:
    return """
    Extract the selected university pathway and co-op details.
    Fields:
    - university_name: The specific university named for the engineering bachelor's degree.
    - accreditation_text: The statement about regional accreditation.
    - accreditation_urls: Official URLs supporting regional accreditation (institutional accreditation page or accreditor).
    - coop_available_text: Statement confirming a co-op program is available.
    - coop_available_urls: Official URLs supporting co-op availability.
    - coop_mandatory_text: Statement confirming co-op is mandatory for the specified engineering pathway (if claimed).
    - coop_mandatory_urls: Official URLs supporting co-op mandatory status (if claimed).
    - coop_prereq_three_semesters_text: Statement that co-op requires completion of three full-time fall/spring semesters as a matriculated student.
    - coop_three_semesters_urls: Official URLs supporting the three-semester prerequisite.
    - coop_prereq_min_gpa_text: Statement that co-op requires a minimum 2.0 GPA.
    - coop_min_gpa_urls: Official URLs supporting the GPA prerequisite.
    - coop_prereq_full_time_text: Statement that co-op requires full-time student status.
    - coop_full_time_urls: Official URLs supporting the full-time status prerequisite.
    - coop_duration_text: Statement that co-op duration is typically ~6 months / one semester.
    - coop_duration_urls: Official URLs supporting typical co-op duration.
    - coop_hours_text: Statement that full-time co-op is typically 32–40 hours/week.
    - coop_hours_urls: Official URLs supporting typical co-op weekly hours.
    """


def prompt_extract_transfer_policies() -> str:
    return """
    Extract transfer-credit policies relevant to high-school-earned credits.
    Fields:
    - transfer_principle_text: General principle that credits from regionally accredited institutions are more readily accepted for transfer.
    - transfer_principle_urls: Official URLs supporting that general principle.
    - dual_enrollment_counts_text: Statement that dual enrollment credits earned in high school can count toward college degree requirements (as applicable).
    - dual_enrollment_urls: Official URLs supporting dual enrollment credit applicability.
    - university_specific_transfer_policy_text: The selected university’s transfer-credit policy for high-school-earned credits (dual enrollment/AP/IB).
    - university_specific_transfer_urls: Official URLs supporting the university’s policy.
    """


def prompt_extract_degree_requirements() -> str:
    return """
    Extract the university-level degree completion requirements for the bachelor’s in engineering (general university minimums or engineering college minimums).
    Fields:
    - min_total_credits_text: Minimum total credits required (e.g., "minimum 120 credits").
    - upper_division_credits_text: Upper-division (300+ level) credit requirement (state the number if provided).
    - graduation_min_gpa_text: Minimum cumulative GPA required to graduate (e.g., "2.0 minimum").
    - degree_requirements_urls: Official URLs supporting these degree requirements.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_text(text: Optional[str]) -> bool:
    return bool(text and str(text).strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0)


def _first_or_placeholder(text: Optional[str], placeholder: str) -> str:
    return text.strip() if _non_empty_text(text) else placeholder


def _combine_urls(*url_lists: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    for ul in url_lists:
        if ul:
            for u in ul:
                if isinstance(u, str) and u.strip():
                    out.append(u.strip())
    # Remove duplicates while preserving order
    seen = set()
    deduped = []
    for u in out:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_hs_program_identification(evaluator: Evaluator, parent, hs: HSProgramInfo) -> None:
    node = evaluator.add_parallel(
        id="High_School_Program_Identification",
        desc="Identify a specific Wake County magnet/specialized high school program that offers college credit opportunities.",
        parent=parent,
        critical=True,
    )

    # Program_Name (existence)
    evaluator.add_custom_node(
        result=_non_empty_text(hs.program_name),
        id="Program_Name",
        desc="Provides a specific Wake County program/school name.",
        parent=node,
        critical=True,
    )

    # Program_Is_WCPSS_And_Specialized (verify with URLs)
    n_is_wcpss = evaluator.add_leaf(
        id="Program_Is_WCPSS_And_Specialized",
        desc="Confirms the program is within Wake County Public Schools and is a magnet or specialized program.",
        parent=node,
        critical=True,
    )
    prog_name = _first_or_placeholder(hs.program_name, "the identified program")
    claim_is_wcpss = f"The program named '{prog_name}' is a Wake County Public School System (WCPSS) magnet, specialized, CTE, or early college program."
    await evaluator.verify(
        claim=claim_is_wcpss,
        node=n_is_wcpss,
        sources=hs.official_urls,
        additional_instruction="Look for WCPSS branding or official program classification on the provided page(s). Treat magnet, early college, CTE pathway, or similar specialized designations as 'specialized'.",
    )

    # Program_Offers_College_Credit (verify with URLs)
    n_offers_credit = evaluator.add_leaf(
        id="Program_Offers_College_Credit",
        desc="Confirms the program offers opportunities to earn college credit/credentials while in high school.",
        parent=node,
        critical=True,
    )
    claim_credit = f"The program '{prog_name}' offers opportunities for students to earn college credits or credentials while in high school (e.g., dual enrollment, CTE articulated credit, CCP, or early college)."
    await evaluator.verify(
        claim=claim_credit,
        node=n_offers_credit,
        sources=hs.official_urls,
        additional_instruction="Verify explicit statements about earning college credits or credentials during high school.",
    )

    # High_School_Program_Official_URL (existence)
    evaluator.add_custom_node(
        result=_has_urls(hs.official_urls),
        id="High_School_Program_Official_URL",
        desc="Provides at least one official-source URL supporting the program identification (e.g., WCPSS/program page).",
        parent=node,
        critical=True,
    )


async def build_application_process(evaluator: Evaluator, parent, ap: ApplicationProcessInfo) -> None:
    node = evaluator.add_parallel(
        id="Application_Process",
        desc="Provide the program application process, including timeline and selection criteria.",
        parent=parent,
        critical=True,
    )

    # Application window (verify exact range Oct 15 – Jan 22)
    n_window = evaluator.add_leaf(
        id="Application_Window",
        desc="States the application window dates and matches the constraint (October 15 – January 22).",
        parent=node,
        critical=True,
    )
    claim_window = "The official application window runs from October 15 to January 22."
    await evaluator.verify(
        claim=claim_window,
        node=n_window,
        sources=ap.application_window_urls,
        additional_instruction="Confirm that the district/program application window described on the cited page spans October 15 through January 22 (inclusive). Ignore specific years; focus on the day/month range.",
    )
    evaluator.add_custom_node(
        result=_has_urls(ap.application_window_urls),
        id="Application_Window_Official_URL",
        desc="Provides at least one official-source URL supporting the application window dates.",
        parent=node,
        critical=True,
    )

    # Results timing (verify Feb 19–20 start)
    n_results = evaluator.add_leaf(
        id="Application_Results_Timing",
        desc="States when results are released and matches the constraint (starting February 19–20).",
        parent=node,
        critical=True,
    )
    claim_results = "Application results are released starting February 19 or February 20 (depending on program/grade)."
    await evaluator.verify(
        claim=claim_results,
        node=n_results,
        sources=ap.results_timing_urls,
        additional_instruction="Accept statements that results begin releasing on February 19 or February 20; minor wording variations are acceptable.",
    )
    evaluator.add_custom_node(
        result=_has_urls(ap.results_timing_urls),
        id="Application_Results_Official_URL",
        desc="Provides at least one official-source URL supporting the results release timing.",
        parent=node,
        critical=True,
    )

    # Selection method (weighted lottery, not FCFS)
    n_sel_method = evaluator.add_leaf(
        id="Selection_Method",
        desc="States selection is via weighted lottery (not first-come-first-served), per constraint.",
        parent=node,
        critical=True,
    )
    claim_sel = "Student selection is conducted via a weighted lottery and is not first-come, first-served."
    await evaluator.verify(
        claim=claim_sel,
        node=n_sel_method,
        sources=ap.selection_method_urls,
        additional_instruction="Look for explicit mention of 'lottery', 'weighted lottery', or similar language indicating NOT first-come-first-served.",
    )
    evaluator.add_custom_node(
        result=_has_urls(ap.selection_method_urls),
        id="Selection_Method_Official_URL",
        desc="Provides at least one official-source URL supporting the weighted-lottery (not first-come-first-served) selection method.",
        parent=node,
        critical=True,
    )

    # Selection criteria (either list criteria or explicitly none beyond lottery)
    n_sel_criteria = evaluator.add_leaf(
        id="Selection_Criteria",
        desc="Provides the program/district stated selection criteria or priority factors, OR explicitly states that official sources do not specify additional criteria beyond the lottery process.",
        parent=node,
        critical=True,
    )
    criteria_text = _first_or_placeholder(ap.selection_criteria_text, "no additional criteria beyond weighted lottery are specified")
    claim_criteria = (
        f"Official sources specify selection criteria/priority factors as follows: {criteria_text}. "
        "If not specified, the sources explicitly state that no additional criteria beyond the weighted lottery are used."
    )
    await evaluator.verify(
        claim=claim_criteria,
        node=n_sel_criteria,
        sources=ap.selection_criteria_urls,
        additional_instruction="Verify that the provided sources either list priority factors (e.g., siblings, continuity, base school) or explicitly indicate there are no additional criteria beyond the weighted lottery.",
    )
    evaluator.add_custom_node(
        result=_has_urls(ap.selection_criteria_urls),
        id="Selection_Criteria_Official_URL",
        desc="Provides at least one official-source URL supporting the stated selection criteria/priority factors (or the absence of additional criteria beyond the lottery).",
        parent=node,
        critical=True,
    )


async def build_hs_completion_requirements(evaluator: Evaluator, parent, comp: HSCompletionRequirementsInfo, hs: HSProgramInfo) -> None:
    node = evaluator.add_parallel(
        id="High_School_Program_Completion_Requirements",
        desc="Provide all core requirements for completing the chosen high school program.",
        parent=parent,
        critical=True,
    )

    # HS graduation requirements included
    n_grad = evaluator.add_leaf(
        id="HS_Graduation_Requirements_Included",
        desc="States that high school graduation requirements must be completed (in addition to any college coursework), per constraint.",
        parent=node,
        critical=True,
    )
    claim_grad = "Students must complete all high school graduation requirements in addition to any college coursework associated with the program."
    await evaluator.verify(
        claim=claim_grad,
        node=n_grad,
        sources=comp.hs_grad_requirements_urls,
        additional_instruction="Look for explicit language that district/NC graduation requirements must be met alongside any college coursework.",
    )
    evaluator.add_custom_node(
        result=_has_urls(comp.hs_grad_requirements_urls),
        id="HS_Graduation_Requirements_Official_URL",
        desc="Provides at least one official-source URL supporting the statement about completing high school graduation requirements in addition to college coursework (as applicable).",
        parent=node,
        critical=True,
    )

    # Program structure and duration (with Early College condition)
    n_structure = evaluator.add_leaf(
        id="Program_Structure_And_Duration",
        desc="States the program duration/structure; if Early College is chosen, it must reflect the constraint that it is a four-year program beginning freshman year.",
        parent=node,
        critical=True,
    )
    structure_text = _first_or_placeholder(comp.program_structure_text, "the program structure and duration specified by the program")
    ptype = (hs.program_type or "").lower()
    add_ins_struct = (
        "Verify the program's official description of structure/duration. If it is an Early College program, confirm it is a four-year program beginning in 9th grade (freshman year)."
    )
    claim_structure = f"The program structure/duration is as stated: {structure_text}. If the program is an Early College, it is a four-year program beginning in 9th grade."
    await evaluator.verify(
        claim=claim_structure,
        node=n_structure,
        sources=comp.program_structure_urls,
        additional_instruction=add_ins_struct,
    )
    evaluator.add_custom_node(
        result=_has_urls(comp.program_structure_urls),
        id="Program_Structure_Official_URL",
        desc="Provides at least one official-source URL supporting the program structure/duration statement.",
        parent=node,
        critical=True,
    )

    # College coursework requirements
    n_cc_reqs = evaluator.add_leaf(
        id="College_Coursework_Requirements",
        desc="Lists the program’s college-coursework participation/academic requirements for completion as specified by official sources.",
        parent=node,
        critical=True,
    )
    cc_text = _first_or_placeholder(comp.college_coursework_requirements_text, "the program specifies college-coursework participation requirements")
    claim_cc = f"The program specifies the following college-coursework participation/academic requirements for completion: {cc_text}."
    await evaluator.verify(
        claim=claim_cc,
        node=n_cc_reqs,
        sources=comp.college_coursework_requirements_urls,
        additional_instruction="Confirm the page lists academic/college-coursework participation or performance requirements tied to program completion.",
    )
    evaluator.add_custom_node(
        result=_has_urls(comp.college_coursework_requirements_urls),
        id="College_Coursework_Requirements_Official_URL",
        desc="Provides at least one official-source URL supporting the program’s college-coursework participation/academic requirements.",
        parent=node,
        critical=True,
    )


async def build_max_credits(evaluator: Evaluator, parent, mc: MaxCreditsInfo, hs: HSProgramInfo) -> None:
    node = evaluator.add_parallel(
        id="Maximum_Credits_Or_Credentials",
        desc="State the maximum college credits/credential achievable through the high school program.",
        parent=parent,
        critical=True,
    )

    n_max = evaluator.add_leaf(
        id="Maximum_Credit_Cap_Stated",
        desc="States the maximum achievable credits/credential; for Early College this must match the constraint (up to 60 credit hours or an associate degree).",
        parent=node,
        critical=True,
    )
    max_text = _first_or_placeholder(mc.max_credits_text, "a defined maximum number of college credits or an associate degree")
    claim_max = (
        f"The program's official materials state the maximum achievable credits/credential as: {max_text}. "
        "If the program is an Early College, it allows up to 60 credit hours or an associate degree."
    )
    await evaluator.verify(
        claim=claim_max,
        node=n_max,
        sources=mc.max_credits_urls,
        additional_instruction="If Early College, confirm 'up to 60 credits' and/or 'associate degree'; otherwise verify the maximum credits/credential as stated.",
    )

    evaluator.add_custom_node(
        result=_has_urls(mc.max_credits_urls),
        id="Max_Credits_Official_URL",
        desc="Provides at least one official-source URL supporting the maximum credits/credential claim.",
        parent=node,
        critical=True,
    )


async def build_university_and_coop(evaluator: Evaluator, parent, uni: UniversityCoopInfo) -> None:
    node = evaluator.add_parallel(
        id="University_Program_And_Coop",
        desc="Identify a university pathway meeting accreditation and co-op requirements, and provide co-op participation requirements.",
        parent=parent,
        critical=True,
    )

    # University_Name (existence)
    evaluator.add_custom_node(
        result=_non_empty_text(uni.university_name),
        id="University_Name",
        desc="Provides a specific university name for the engineering bachelor’s pathway.",
        parent=node,
        critical=True,
    )
    uname = _first_or_placeholder(uni.university_name, "the selected university")

    # Regional accreditation
    n_acc = evaluator.add_leaf(
        id="University_Regional_Accreditation",
        desc="Confirms the university is regionally accredited, per constraint.",
        parent=node,
        critical=True,
    )
    claim_acc = f"{uname} is regionally accredited by a recognized U.S. regional accreditor (e.g., SACSCOC, HLC, MSCHE, NECHE, NWCCU, WSCUC)."
    await evaluator.verify(
        claim=claim_acc,
        node=n_acc,
        sources=uni.accreditation_urls,
        additional_instruction="Verify institutional (regional) accreditation from official university or accreditor sources.",
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.accreditation_urls),
        id="University_Regional_Accreditation_Official_URL",
        desc="Provides at least one official-source URL supporting the university’s regional accreditation status.",
        parent=node,
        critical=True,
    )

    # Co-op available
    n_coop_avail = evaluator.add_leaf(
        id="University_Coop_Available",
        desc="Confirms the university has a cooperative education program available to students, per constraint.",
        parent=node,
        critical=True,
    )
    claim_coop_avail = f"{uname} offers a cooperative education (co-op) program for students (including engineering students)."
    await evaluator.verify(
        claim=claim_coop_avail,
        node=n_coop_avail,
        sources=uni.coop_available_urls,
        additional_instruction="Look for official pages describing co-op as an established program or option.",
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.coop_available_urls),
        id="University_Coop_Available_Official_URL",
        desc="Provides at least one official-source URL supporting that a co-op program is available.",
        parent=node,
        critical=True,
    )

    # Co-op mandatory for pathway
    n_coop_mand = evaluator.add_leaf(
        id="University_Coop_Mandatory_For_Pathway",
        desc="Confirms the engineering pathway includes a mandatory co-op program, as required by the proposed question.",
        parent=node,
        critical=True,
    )
    claim_coop_mand = f"For the specified bachelor’s engineering pathway at {uname}, participation in co-op is mandatory (a degree requirement)."
    await evaluator.verify(
        claim=claim_coop_mand,
        node=n_coop_mand,
        sources=uni.coop_mandatory_urls,
        additional_instruction="Confirm the page states co-op is a required/mandatory component to graduate for the given engineering program.",
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.coop_mandatory_urls),
        id="University_Coop_Mandatory_Official_URL",
        desc="Provides at least one official-source URL supporting that co-op is mandatory for the specified engineering pathway (if claimed).",
        parent=node,
        critical=True,
    )

    # Co-op prerequisites
    n_three_sem = evaluator.add_leaf(
        id="Coop_Prereq_Three_Semesters",
        desc="States co-op requires completion of three (3) full-time fall/spring semesters as a matriculated student, per constraint.",
        parent=node,
        critical=True,
    )
    claim_three_sem = "Eligibility for co-op requires completion of three full-time fall/spring semesters as a matriculated student at the university."
    await evaluator.verify(
        claim=claim_three_sem,
        node=n_three_sem,
        sources=uni.coop_three_semesters_urls,
        additional_instruction="Look for explicit prerequisites specifying at least three full-time fall/spring semesters completed before starting co-op.",
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.coop_three_semesters_urls),
        id="Coop_Prereq_Three_Semesters_Official_URL",
        desc="Provides at least one official-source URL supporting the three-semester matriculation prerequisite.",
        parent=node,
        critical=True,
    )

    n_min_gpa = evaluator.add_leaf(
        id="Coop_Prereq_Min_GPA",
        desc="States co-op requires minimum 2.0 GPA, per constraint.",
        parent=node,
        critical=True,
    )
    claim_min_gpa = "Eligibility for co-op requires a minimum cumulative GPA of at least 2.0 on a 4.0 scale."
    await evaluator.verify(
        claim=claim_min_gpa,
        node=n_min_gpa,
        sources=uni.coop_min_gpa_urls,
        additional_instruction="Verify that the eligibility section specifies a minimum GPA of 2.0 (or higher) for co-op.",
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.coop_min_gpa_urls),
        id="Coop_Prereq_Min_GPA_Official_URL",
        desc="Provides at least one official-source URL supporting the minimum GPA prerequisite.",
        parent=node,
        critical=True,
    )

    n_full_time = evaluator.add_leaf(
        id="Coop_Prereq_Full_Time_Status",
        desc="States co-op requires full-time student status, per constraint.",
        parent=node,
        critical=True,
    )
    claim_full_time = "Eligibility for co-op requires students to be enrolled full-time."
    await evaluator.verify(
        claim=claim_full_time,
        node=n_full_time,
        sources=uni.coop_full_time_urls,
        additional_instruction="Confirm that the policy requires full-time student status for co-op participation.",
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.coop_full_time_urls),
        id="Coop_Prereq_Full_Time_Status_Official_URL",
        desc="Provides at least one official-source URL supporting the full-time status prerequisite.",
        parent=node,
        critical=True,
    )

    # Co-op duration and hours
    n_duration = evaluator.add_leaf(
        id="Coop_Duration",
        desc="States the co-op duration is typically ~6 months / one semester, per constraint.",
        parent=node,
        critical=True,
    )
    claim_duration = "The typical duration of a co-op assignment is about one academic semester (approximately six months)."
    await evaluator.verify(
        claim=claim_duration,
        node=n_duration,
        sources=uni.coop_duration_urls,
        additional_instruction="Accept phrasing like 'one term', 'one semester', 'about six months', or equivalent durations.",
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.coop_duration_urls),
        id="Coop_Duration_Official_URL",
        desc="Provides at least one official-source URL supporting the typical co-op duration statement.",
        parent=node,
        critical=True,
    )

    n_hours = evaluator.add_leaf(
        id="Coop_Hours_Per_Week",
        desc="States co-op workload is typically 32–40 hours/week for full-time co-op, per constraint.",
        parent=node,
        critical=True,
    )
    claim_hours = "Full-time co-op positions typically require approximately 32 to 40 hours of work per week."
    await evaluator.verify(
        claim=claim_hours,
        node=n_hours,
        sources=uni.coop_hours_urls,
        additional_instruction="Confirm typical weekly hours for full-time co-op fall in the 32–40 hours/week range.",
    )
    evaluator.add_custom_node(
        result=_has_urls(uni.coop_hours_urls),
        id="Coop_Hours_Per_Week_Official_URL",
        desc="Provides at least one official-source URL supporting the typical co-op hours/week statement.",
        parent=node,
        critical=True,
    )


async def build_transfer_policies(evaluator: Evaluator, parent, tp: TransferPoliciesInfo) -> None:
    node = evaluator.add_parallel(
        id="Credit_Transfer_Policies",
        desc="Provide credit transfer policies from high school to university (dual enrollment/AP/IB as applicable).",
        parent=parent,
        critical=True,
    )

    n_reg_transfer = evaluator.add_leaf(
        id="Regional_Accreditation_Transfer_General_Principle",
        desc="States that credits from regionally accredited institutions are more readily accepted for transfer, per constraint (as applicable).",
        parent=node,
        critical=True,
    )
    claim_reg_transfer = "Universities are more likely to accept transfer credits from regionally accredited institutions."
    await evaluator.verify(
        claim=claim_reg_transfer,
        node=n_reg_transfer,
        sources=tp.transfer_principle_urls,
        additional_instruction="Look for official policy language indicating preference or requirement for credits to originate from regionally accredited institutions.",
    )
    evaluator.add_custom_node(
        result=_has_urls(tp.transfer_principle_urls),
        id="Regional_Accreditation_Transfer_Official_URL",
        desc="Provides at least one official-source URL supporting the general principle about regional accreditation and transfer acceptance (as stated).",
        parent=node,
        critical=True,
    )

    n_dual = evaluator.add_leaf(
        id="Dual_Enrollment_Can_Count_Toward_Degree",
        desc="States that dual enrollment credits earned in high school can count toward college degree requirements, per constraint (as applicable).",
        parent=node,
        critical=True,
    )
    claim_dual = "Dual enrollment credits earned in high school can count toward college degree requirements (subject to university policies)."
    await evaluator.verify(
        claim=claim_dual,
        node=n_dual,
        sources=tp.dual_enrollment_urls,
        additional_instruction="Verify that the policy allows dual enrollment credits to apply toward degree or general education/major requirements, not merely elective transfer.",
    )
    evaluator.add_custom_node(
        result=_has_urls(tp.dual_enrollment_urls),
        id="Dual_Enrollment_Counts_Official_URL",
        desc="Provides at least one official-source URL supporting that dual enrollment credits can count toward degree requirements (as stated).",
        parent=node,
        critical=True,
    )

    n_uni_policy = evaluator.add_leaf(
        id="University_Specific_Transfer_Policy",
        desc="Provides the selected university’s stated transfer-credit policy relevant to high-school-earned credits (e.g., dual enrollment/AP/IB), based on official sources.",
        parent=node,
        critical=True,
    )
    policy_text = _first_or_placeholder(tp.university_specific_transfer_policy_text, "the university describes how dual enrollment/AP/IB credit is awarded and applied")
    claim_uni_policy = f"The selected university's official policy describes how high-school-earned credits (dual enrollment/AP/IB) transfer and apply: {policy_text}."
    await evaluator.verify(
        claim=claim_uni_policy,
        node=n_uni_policy,
        sources=tp.university_specific_transfer_urls,
        additional_instruction="Confirm details like minimum grades, course equivalencies, limits, and applicability to degree requirements.",
    )
    evaluator.add_custom_node(
        result=_has_urls(tp.university_specific_transfer_urls),
        id="University_Specific_Transfer_Policy_Official_URL",
        desc="Provides at least one official-source URL supporting the selected university’s transfer-credit policy.",
        parent=node,
        critical=True,
    )


async def build_degree_requirements(evaluator: Evaluator, parent, deg: DegreeCompletionInfo, uni: UniversityCoopInfo) -> None:
    node = evaluator.add_parallel(
        id="University_Degree_Completion_Requirements",
        desc="Provide final bachelor’s degree completion requirements at the university level for engineering.",
        parent=parent,
        critical=True,
    )
    uname = _first_or_placeholder(uni.university_name, "the selected university")

    # Minimum total credits >= 120
    n_min_credits = evaluator.add_leaf(
        id="Minimum_Total_Credits",
        desc="States the bachelor’s degree minimum total credits (≥120), per constraint.",
        parent=node,
        critical=True,
    )
    claim_min_cred = f"The minimum total credits required to earn a bachelor's degree at {uname} is at least 120 credits."
    await evaluator.verify(
        claim=claim_min_cred,
        node=n_min_credits,
        sources=deg.degree_requirements_urls,
        additional_instruction="Verify that the minimum total credits to graduate is no less than 120 credits.",
    )

    # Upper-division credits between 36–48
    n_ud = evaluator.add_leaf(
        id="Upper_Division_Credits",
        desc="States the upper-division (300+ level) credit requirement (36–48), per constraint.",
        parent=node,
        critical=True,
    )
    claim_ud = f"{uname} requires an upper-division (300+ level) credit requirement that falls between 36 and 48 credits (inclusive)."
    await evaluator.verify(
        claim=claim_ud,
        node=n_ud,
        sources=deg.degree_requirements_urls,
        additional_instruction="Look for 'upper-division' or '300/400-level' minimum credit requirements. Accept a value within the 36–48 range.",
    )

    # Graduation minimum GPA >= 2.0
    n_grad_gpa = evaluator.add_leaf(
        id="Graduation_Minimum_GPA",
        desc="States the minimum cumulative GPA for graduation (≥2.0), per constraint.",
        parent=node,
        critical=True,
    )
    claim_gpa = "The minimum cumulative GPA required to graduate is at least 2.0."
    await evaluator.verify(
        claim=claim_gpa,
        node=n_grad_gpa,
        sources=deg.degree_requirements_urls,
        additional_instruction="Confirm a published minimum cumulative GPA threshold for graduation (2.0 or higher).",
    )

    evaluator.add_custom_node(
        result=_has_urls(deg.degree_requirements_urls),
        id="Degree_Requirements_Official_URL",
        desc="Provides at least one official-source URL supporting the university degree completion requirements stated (credits, upper-division credits, and graduation GPA).",
        parent=node,
        critical=True,
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
    model: str = "o4-mini",
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
        default_model=model,
    )

    # Create the critical top-level pathway node under the non-critical framework root
    pathway_root = evaluator.add_parallel(
        id="Educational_Pathway_Validation",
        desc="Validate the complete educational pathway from Wake County high school program through university engineering graduation, including official-source citations.",
        parent=root,
        critical=True,
    )

    # Run extractions (in parallel where possible)
    (
        hs_info,
        app_info,
        comp_info,
        max_cred_info,
        uni_info,
        transfer_info,
        degree_info,
    ) = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_hs_program(),
            template_class=HSProgramInfo,
            extraction_name="hs_program_info",
        ),
        evaluator.extract(
            prompt=prompt_extract_application_process(),
            template_class=ApplicationProcessInfo,
            extraction_name="application_process_info",
        ),
        evaluator.extract(
            prompt=prompt_extract_hs_completion_requirements(),
            template_class=HSCompletionRequirementsInfo,
            extraction_name="hs_completion_requirements_info",
        ),
        evaluator.extract(
            prompt=prompt_extract_max_credits(),
            template_class=MaxCreditsInfo,
            extraction_name="max_credits_info",
        ),
        evaluator.extract(
            prompt=prompt_extract_university_coop(),
            template_class=UniversityCoopInfo,
            extraction_name="university_coop_info",
        ),
        evaluator.extract(
            prompt=prompt_extract_transfer_policies(),
            template_class=TransferPoliciesInfo,
            extraction_name="transfer_policies_info",
        ),
        evaluator.extract(
            prompt=prompt_extract_degree_requirements(),
            template_class=DegreeCompletionInfo,
            extraction_name="degree_completion_info",
        ),
    )

    # Optional: record a brief summary of chosen entities
    evaluator.add_custom_info(
        info={
            "chosen_hs_program": hs_info.program_name,
            "hs_program_type": hs_info.program_type,
            "selected_university": uni_info.university_name,
        },
        info_type="selection_summary",
        info_name="selection_summary",
    )

    # Build verification subtrees
    await build_hs_program_identification(evaluator, pathway_root, hs_info)
    await build_application_process(evaluator, pathway_root, app_info)
    await build_hs_completion_requirements(evaluator, pathway_root, comp_info, hs_info)
    await build_max_credits(evaluator, pathway_root, max_cred_info, hs_info)
    await build_university_and_coop(evaluator, pathway_root, uni_info)
    await build_transfer_policies(evaluator, pathway_root, transfer_info)
    await build_degree_requirements(evaluator, pathway_root, degree_info, uni_info)

    # Return evaluation summary
    return evaluator.get_summary()