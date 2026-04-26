import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "creative_writing_ce_certificate"
TASK_DESCRIPTION = (
    "Identify a continuing education certificate program in Creative Writing offered by a major research university in the United States that meets ALL of the following requirements:\n\n"
    "1. The program must require completion of at least five specific named courses as core requirements that form a structured multi-year sequence\n"
    "2. The program must include at least one elective course requirement that must be chosen from a defined category or list of courses offered by the same institution\n"
    "3. The elective course must have a specified minimum letter grade requirement (such as C-, C, B-, B, etc.) that students must achieve for the course to count toward the certificate\n"
    "4. The program must specify a maximum time limit for completing all requirements, and this timeframe must be no more than 5 years\n"
    "5. The program must cap individual class section sizes at a maximum of 20 students or fewer\n"
    "6. The program must have a stated policy that allows consideration of prior coursework taken at the same institution under specific conditions (such as time limits since completion, program director approval, or maximum percentage of requirements that can be satisfied)\n"
    "7. The program must explicitly state that transfer credits or courses taken at other institutions (including other universities) are not accepted for fulfilling the core course requirements\n"
    "8. The program must require an admission, application, or formal candidacy process (not open enrollment for the certificate itself, though individual courses may be open enrollment)\n\n"
    "Provide the following information:\n"
    "- Name of the university\n"
    "- Name of the certificate program\n"
    "- List of the five or more required core courses with their course numbers\n"
    "- Description of the elective requirement and the minimum grade requirement\n"
    "- The maximum completion timeframe\n"
    "- The maximum class section size\n"
    "- Summary of the prior coursework policy with its specific conditions\n"
    "- Statement of the transfer credit policy regarding external institutions\n"
    "- Description of the admission/application process\n"
    "- Reference URLs supporting each piece of information"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CourseItem(BaseModel):
    course_name: Optional[str] = None
    course_number: Optional[str] = None
    course_urls: List[str] = Field(default_factory=list)


class ProgramExtraction(BaseModel):
    # University identity and evidence
    university_name: Optional[str] = None
    university_location: Optional[str] = None  # e.g., "United States", "USA", "New York, NY, USA"
    university_evidence_urls: List[str] = Field(default_factory=list)

    # Program identity
    program_name: Optional[str] = None
    program_type: Optional[str] = None  # e.g., "continuing education certificate"
    program_subject: Optional[str] = None  # e.g., "Creative Writing"
    program_urls: List[str] = Field(default_factory=list)

    # Core courses and structured sequence
    core_courses: List[CourseItem] = Field(default_factory=list)
    core_sequence_description: Optional[str] = None  # text indicating multi-year structured sequence
    core_courses_urls: List[str] = Field(default_factory=list)

    # Elective requirement and grade threshold
    elective_requirement_desc: Optional[str] = None
    elective_from_same_institution: Optional[bool] = None
    elective_min_grade_letter: Optional[str] = None  # e.g., "C", "B-", "C-"
    elective_urls: List[str] = Field(default_factory=list)

    # Maximum completion timeframe
    max_completion_timeframe: Optional[str] = None  # e.g., "up to 5 years"
    max_completion_urls: List[str] = Field(default_factory=list)

    # Class section size cap
    max_class_section_size: Optional[str] = None  # e.g., "Maximum 20 students per class"
    class_size_urls: List[str] = Field(default_factory=list)

    # Prior coursework policy from same institution
    prior_coursework_allowed: Optional[bool] = None
    prior_coursework_conditions: Optional[str] = None  # e.g., "director approval required; only courses within last 3 years; max 50%"
    prior_coursework_summary: Optional[str] = None
    prior_coursework_urls: List[str] = Field(default_factory=list)

    # Transfer credit policy
    transfer_core_policy: Optional[str] = None  # e.g., "No transfer credits for core courses"
    transfer_policy_urls: List[str] = Field(default_factory=list)

    # Admission/application process
    application_required: Optional[bool] = None
    admission_process_desc: Optional[str] = None
    admission_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_program() -> str:
    return (
        "Extract structured information about the single Creative Writing continuing education certificate program described in the answer.\n"
        "Return the following fields:\n"
        "1) university_name: The name of the university offering the program.\n"
        "2) university_location: The location string (should indicate USA/United States if present).\n"
        "3) university_evidence_urls: All URLs cited that support the university identity, location, and/or research status (e.g., official university site, Carnegie classification, AAU membership page, etc.).\n"
        "4) program_name: The name of the certificate program.\n"
        "5) program_type: The type of the program (e.g., 'continuing education certificate').\n"
        "6) program_subject: The subject area (should be 'Creative Writing').\n"
        "7) program_urls: URLs cited that describe the program identity/type/subject.\n"
        "8) core_courses: An array. For each core course, include:\n"
        "   - course_name\n"
        "   - course_number (course code)\n"
        "   - course_urls (URLs cited for that course, if any)\n"
        "9) core_sequence_description: Text describing that the core courses form a structured multi-term/multi-year sequence.\n"
        "10) core_courses_urls: URLs cited that support the core course list/count and the structured sequence claim.\n"
        "11) elective_requirement_desc: A description of the elective requirement, including how electives must be chosen (e.g., from a defined category/list).\n"
        "12) elective_from_same_institution: true/false if stated that electives must be from the same institution.\n"
        "13) elective_min_grade_letter: The minimum letter grade required (e.g., 'C', 'B-', 'C-') for the elective to count.\n"
        "14) elective_urls: URLs cited that support the elective requirement and grade policy.\n"
        "15) max_completion_timeframe: The stated maximum time to complete all requirements (e.g., '5 years').\n"
        "16) max_completion_urls: URLs cited that support the maximum completion timeframe.\n"
        "17) max_class_section_size: The stated maximum class section size (e.g., '20 students').\n"
        "18) class_size_urls: URLs cited that support the class size cap.\n"
        "19) prior_coursework_allowed: true/false indicating whether prior coursework at the same institution can be applied under conditions.\n"
        "20) prior_coursework_conditions: The text of specific conditions (e.g., time limit, director approval, max percentage) if provided.\n"
        "21) prior_coursework_summary: A concise summary of the prior coursework policy conditions.\n"
        "22) prior_coursework_urls: URLs cited that support the prior coursework policy.\n"
        "23) transfer_core_policy: The statement of transfer credit policy regarding external institutions for core requirements.\n"
        "24) transfer_policy_urls: URLs cited that support the external transfer policy.\n"
        "25) application_required: true/false indicating whether admission/application/formal candidacy is required for the certificate program.\n"
        "26) admission_process_desc: A description of the admission/application process.\n"
        "27) admission_urls: URLs cited that support the admission/application requirement and process.\n\n"
        "Rules:\n"
        "- Extract exactly what is present in the answer; do not invent details.\n"
        "- If a field is missing, use null or an empty list accordingly.\n"
        "- Extract all URLs in any reasonable format; return full URLs."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
_SPELLED_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12
}


def _first_integer_in_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # Look for digits first
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    # Try spelled-out numbers up to twelve
    lower = text.lower()
    for word, val in _SPELLED_NUMBERS.items():
        if re.search(rf"\b{word}\b", lower):
            return val
    return None


def _is_us_location(location_text: Optional[str]) -> bool:
    if not location_text:
        return False
    t = location_text.lower()
    return any(k in t for k in ["united states", "usa", "u.s.", "u.s.a", "us"])


def _nonempty_urls(*url_lists: List[str]) -> List[str]:
    for ul in url_lists:
        if ul and len(ul) > 0:
            # Filter obvious empties/malformed minimal
            return [u for u in ul if isinstance(u, str) and len(u.strip()) > 0]
    return []


def _has_at_least_one_url(*url_lists: List[str]) -> bool:
    return len(_nonempty_urls(*url_lists)) > 0


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_program_identification_and_eligibility(evaluator: Evaluator, parent_node, ex: ProgramExtraction) -> None:
    node = evaluator.add_parallel(
        id="program_identification_and_eligibility",
        desc="Program identification + baseline eligibility (university, program type, subject) with citations",
        parent=parent_node,
        critical=True
    )

    # University name provided
    evaluator.add_custom_node(
        result=bool(ex.university_name and ex.university_name.strip()),
        id="university_name_provided",
        desc="Provides the name of the university offering the program",
        parent=node,
        critical=True
    )

    # University in US
    uni_sources = _nonempty_urls(ex.university_evidence_urls, ex.program_urls)
    uni_in_us_leaf = evaluator.add_leaf(
        id="university_in_us",
        desc="Confirms the university is located in the United States",
        parent=node,
        critical=True
    )
    claim_us = f"The university '{ex.university_name or ''}' is located in the United States."
    await evaluator.verify(
        claim=claim_us,
        node=uni_in_us_leaf,
        sources=uni_sources,
        additional_instruction="Confirm that the university is based in the USA. Accept evidence such as official location pages or widely recognized references indicating U.S. location."
    )

    # University is major research university
    uni_major_leaf = evaluator.add_leaf(
        id="university_is_major_research",
        desc="Confirms the university qualifies as a major research university",
        parent=node,
        critical=True
    )
    claim_major = (
        f"The university '{ex.university_name or ''}' qualifies as a major research university "
        f"(e.g., Carnegie R1 classification, AAU membership, or other authoritative designation)."
    )
    await evaluator.verify(
        claim=claim_major,
        node=uni_major_leaf,
        sources=uni_sources,
        additional_instruction="Look for evidence such as Carnegie R1 classification, AAU membership, or equivalent authoritative indicators that the university is a major research institution."
    )

    # Program name provided
    evaluator.add_custom_node(
        result=bool(ex.program_name and ex.program_name.strip()),
        id="program_name_provided",
        desc="Provides the name of the continuing education certificate program",
        parent=node,
        critical=True
    )

    # Program is continuing ed certificate (not degree)
    prog_sources = _nonempty_urls(ex.program_urls)
    prog_type_leaf = evaluator.add_leaf(
        id="program_is_continuing_ed_certificate_not_degree",
        desc="Confirms the program is a continuing education certificate program (not a degree program)",
        parent=node,
        critical=True
    )
    claim_prog_type = (
        f"The program '{ex.program_name or ''}' at '{ex.university_name or ''}' is a continuing education certificate program, not a degree program."
    )
    await evaluator.verify(
        claim=claim_prog_type,
        node=prog_type_leaf,
        sources=prog_sources,
        additional_instruction="Confirm that the program is categorized as a continuing education certificate (professional or extension), rather than an academic degree."
    )

    # Program subject is Creative Writing
    subject_leaf = evaluator.add_leaf(
        id="program_subject_creative_writing",
        desc="Confirms the program focuses on Creative Writing",
        parent=node,
        critical=True
    )
    claim_subject = "The program focuses on Creative Writing."
    await evaluator.verify(
        claim=claim_subject,
        node=subject_leaf,
        sources=prog_sources,
        additional_instruction="Confirm that the program subject area is Creative Writing."
    )

    # Eligibility citations (must have university and program URLs)
    evaluator.add_custom_node(
        result=_has_at_least_one_url(ex.university_evidence_urls) and _has_at_least_one_url(ex.program_urls),
        id="eligibility_citations",
        desc="Provides reference URL(s) supporting the university identity/location/research status and the program identity/type/subject",
        parent=node,
        critical=True
    )


async def verify_core_course_requirements(evaluator: Evaluator, parent_node, ex: ProgramExtraction) -> None:
    node = evaluator.add_parallel(
        id="core_course_requirements",
        desc="Core-course requirements meet minimum count and are listed with course numbers, and form a structured multi-year sequence",
        parent=parent_node,
        critical=True
    )

    # Lists at least five required core courses, each with course number/code
    has_5 = len(ex.core_courses) >= 5
    all_have_numbers = all(
        (c.course_name and c.course_name.strip()) and (c.course_number and c.course_number.strip())
        for c in ex.core_courses
    ) if ex.core_courses else False
    evaluator.add_custom_node(
        result=has_5 and all_have_numbers,
        id="core_courses_count_and_list",
        desc="Lists at least five required core courses, each with its course number/code",
        parent=node,
        critical=True
    )

    # Structured multi-year sequence
    seq_leaf = evaluator.add_leaf(
        id="core_courses_structured_sequence_multi_year",
        desc="Confirms the core courses form a structured multi-year sequence/progression (not merely an unordered set)",
        parent=node,
        critical=True
    )
    core_sources = _nonempty_urls(ex.core_courses_urls, ex.program_urls)
    claim_seq = (
        "The program's required core courses form a structured multi-year sequence or progression, rather than a loose unordered set."
    )
    await evaluator.verify(
        claim=claim_seq,
        node=seq_leaf,
        sources=core_sources,
        additional_instruction="Look for explicit sequencing across terms/quarters/years, prerequisites, or stated multi-year structure."
    )

    # Core courses citations
    evaluator.add_custom_node(
        result=_has_at_least_one_url(ex.core_courses_urls) or _has_at_least_one_url(ex.program_urls),
        id="core_courses_citations",
        desc="Provides reference URL(s) supporting the core course list/count and the structured sequence claim",
        parent=node,
        critical=True
    )


async def verify_elective_and_grade_requirement(evaluator: Evaluator, parent_node, ex: ProgramExtraction) -> None:
    node = evaluator.add_parallel(
        id="elective_and_grade_requirement",
        desc="Elective requirement is defined, is from same institution, and includes a minimum letter-grade threshold",
        parent=parent_node,
        critical=True
    )

    elect_sources = _nonempty_urls(ex.elective_urls, ex.program_urls)

    # Elective requirement defined from list or category
    elective_defined_leaf = evaluator.add_leaf(
        id="elective_requirement_defined_from_list_or_category",
        desc="Confirms at least one elective is required and must be chosen from a defined category/list of courses",
        parent=node,
        critical=True
    )
    claim_elect_defined = (
        "At least one elective is required and must be selected from a defined category or list of courses."
    )
    await evaluator.verify(
        claim=claim_elect_defined,
        node=elective_defined_leaf,
        sources=elect_sources,
        additional_instruction="Confirm that the elective is not free-form; it must be chosen from a defined institutional category/list."
    )

    # Elective from same institution
    elective_same_inst_leaf = evaluator.add_leaf(
        id="elective_from_same_institution",
        desc="Confirms the elective must be taken from courses offered by the same institution",
        parent=node,
        critical=True
    )
    claim_same_inst = "The elective must be taken from courses offered by the same institution."
    await evaluator.verify(
        claim=claim_same_inst,
        node=elective_same_inst_leaf,
        sources=elect_sources,
        additional_instruction="Confirm that elective selection is constrained to courses offered by the same university/institution."
    )

    # Elective minimum letter grade specified
    grade_leaf = evaluator.add_leaf(
        id="elective_minimum_letter_grade_specified",
        desc="Confirms a minimum letter grade is specified for the elective to count (and provides the stated threshold)",
        parent=node,
        critical=True
    )
    grade_text = ex.elective_min_grade_letter or ""
    claim_grade = (
        f"The elective must be completed with a minimum letter grade of {grade_text} (or higher, as applicable) to count toward the certificate."
    )
    await evaluator.verify(
        claim=claim_grade,
        node=grade_leaf,
        sources=elect_sources,
        additional_instruction="Confirm explicit minimum letter grade requirement (e.g., C-, C, B-, B)."
    )

    # Elective citations
    evaluator.add_custom_node(
        result=_has_at_least_one_url(ex.elective_urls) or _has_at_least_one_url(ex.program_urls),
        id="elective_citations",
        desc="Provides reference URL(s) supporting the elective requirement and minimum grade policy",
        parent=node,
        critical=True
    )


async def verify_completion_time_limit(evaluator: Evaluator, parent_node, ex: ProgramExtraction) -> None:
    node = evaluator.add_parallel(
        id="completion_time_limit",
        desc="Maximum completion timeframe is stated and is no more than 5 years",
        parent=parent_node,
        critical=True
    )

    time_sources = _nonempty_urls(ex.max_completion_urls, ex.program_urls)
    # Time limit stated
    time_stated_leaf = evaluator.add_leaf(
        id="time_limit_stated",
        desc="Program states a maximum time limit for completing all requirements",
        parent=node,
        critical=True
    )
    tl_text = ex.max_completion_timeframe or ""
    claim_time = f"The program sets a maximum completion timeframe of {tl_text} for completing all requirements."
    await evaluator.verify(
        claim=claim_time,
        node=time_stated_leaf,
        sources=time_sources,
        additional_instruction="Confirm that a maximum completion time is stated (e.g., up to 5 years)."
    )

    # Time limit no more than 5 years (numeric check)
    years_val = _first_integer_in_text(ex.max_completion_timeframe)
    evaluator.add_custom_node(
        result=bool(years_val is not None and years_val <= 5),
        id="time_limit_no_more_than_5_years",
        desc="The stated maximum completion timeframe is 5 years or less",
        parent=node,
        critical=True
    )

    # Time limit citations present
    evaluator.add_custom_node(
        result=_has_at_least_one_url(ex.max_completion_urls) or _has_at_least_one_url(ex.program_urls),
        id="time_limit_citations",
        desc="Provides reference URL(s) supporting the stated maximum completion timeframe",
        parent=node,
        critical=True
    )


async def verify_class_section_size_cap(evaluator: Evaluator, parent_node, ex: ProgramExtraction) -> None:
    node = evaluator.add_parallel(
        id="class_section_size_cap",
        desc="Individual class sections are capped at 20 students or fewer",
        parent=parent_node,
        critical=True
    )

    class_sources = _nonempty_urls(ex.class_size_urls, ex.program_urls)

    # Cap stated
    cap_stated_leaf = evaluator.add_leaf(
        id="cap_stated",
        desc="Program states an individual class section size cap",
        parent=node,
        critical=True
    )
    cap_text = ex.max_class_section_size or ""
    claim_cap = f"Individual class sections are capped at {cap_text}."
    await evaluator.verify(
        claim=claim_cap,
        node=cap_stated_leaf,
        sources=class_sources,
        additional_instruction="Confirm the explicit maximum class section size cap stated by the program."
    )

    # Cap 20 or fewer
    cap_val = _first_integer_in_text(ex.max_class_section_size)
    evaluator.add_custom_node(
        result=bool(cap_val is not None and cap_val <= 20),
        id="cap_20_or_fewer",
        desc="The stated class section cap is 20 students or fewer",
        parent=node,
        critical=True
    )

    # Class size citations present
    evaluator.add_custom_node(
        result=_has_at_least_one_url(ex.class_size_urls) or _has_at_least_one_url(ex.program_urls),
        id="class_size_citations",
        desc="Provides reference URL(s) supporting the class size cap",
        parent=node,
        critical=True
    )


async def verify_prior_coursework_policy(evaluator: Evaluator, parent_node, ex: ProgramExtraction) -> None:
    node = evaluator.add_parallel(
        id="prior_coursework_same_institution_policy",
        desc="Policy allows consideration of prior coursework from the same institution under specific conditions",
        parent=parent_node,
        critical=True
    )

    prior_sources = _nonempty_urls(ex.prior_coursework_urls, ex.program_urls)

    # Prior coursework allowed from same institution
    prior_allowed_leaf = evaluator.add_leaf(
        id="prior_coursework_allowed_same_institution",
        desc="Program states prior coursework taken at the same institution may be considered/applied under some conditions",
        parent=node,
        critical=True
    )
    claim_prior_allowed = (
        "The program allows consideration/applicability of prior coursework taken at the same institution, subject to stated conditions."
    )
    await evaluator.verify(
        claim=claim_prior_allowed,
        node=prior_allowed_leaf,
        sources=prior_sources,
        additional_instruction="Confirm that prior coursework from the same institution can be considered/applied under specific conditions."
    )

    # Prior coursework conditions specified (verify by text)
    conditions_text = ex.prior_coursework_conditions or ex.prior_coursework_summary or ""
    conditions_leaf = evaluator.add_leaf(
        id="prior_coursework_conditions_specified",
        desc="Policy states specific limiting conditions (e.g., time-since-completion limit, director approval, and/or maximum portion allowed)",
        parent=node,
        critical=True
    )
    claim_conditions = (
        f"The policy specifies limiting conditions for applying prior coursework (e.g., time limits since completion, program director approval, maximum percentage). "
        f"Example conditions stated: {conditions_text}"
    )
    await evaluator.verify(
        claim=claim_conditions,
        node=conditions_leaf,
        sources=prior_sources,
        additional_instruction="Verify that specific conditions are indeed stated (such as time-since-completion limits, director approval, or maximum portion allowed)."
    )

    # Prior coursework conditions summarized (existence check)
    evaluator.add_custom_node(
        result=bool(ex.prior_coursework_summary and ex.prior_coursework_summary.strip()),
        id="prior_coursework_conditions_summarized",
        desc="Provides a summary of the specific conditions as written in the policy",
        parent=node,
        critical=True
    )

    # Prior coursework citations present
    evaluator.add_custom_node(
        result=_has_at_least_one_url(ex.prior_coursework_urls) or _has_at_least_one_url(ex.program_urls),
        id="prior_coursework_citations",
        desc="Provides reference URL(s) supporting the prior coursework policy and conditions",
        parent=node,
        critical=True
    )


async def verify_transfer_policy(evaluator: Evaluator, parent_node, ex: ProgramExtraction) -> None:
    node = evaluator.add_parallel(
        id="transfer_credit_policy_external_institutions",
        desc="Transfer credits/courses from other institutions are not accepted for fulfilling core course requirements",
        parent=parent_node,
        critical=True
    )

    transfer_sources = _nonempty_urls(ex.transfer_policy_urls, ex.program_urls)

    # External transfer not accepted for core
    transfer_leaf = evaluator.add_leaf(
        id="external_transfer_not_accepted_for_core",
        desc="Program explicitly states transfer credits/courses from external institutions are not accepted for core requirements",
        parent=node,
        critical=True
    )
    claim_transfer = (
        "Transfer credits or courses taken at other institutions are not accepted for fulfilling the program's core course requirements."
    )
    await evaluator.verify(
        claim=claim_transfer,
        node=transfer_leaf,
        sources=transfer_sources,
        additional_instruction="Confirm explicit prohibition of external transfer credits for core requirements."
    )

    # Transfer policy citations present
    evaluator.add_custom_node(
        result=_has_at_least_one_url(ex.transfer_policy_urls) or _has_at_least_one_url(ex.program_urls),
        id="transfer_policy_citations",
        desc="Provides reference URL(s) supporting the external transfer-credit prohibition for core requirements",
        parent=node,
        critical=True
    )


async def verify_admission_process(evaluator: Evaluator, parent_node, ex: ProgramExtraction) -> None:
    node = evaluator.add_parallel(
        id="admission_application_process",
        desc="Certificate requires an admission/application/formal candidacy process (not open enrollment for the certificate itself)",
        parent=parent_node,
        critical=True
    )

    adm_sources = _nonempty_urls(ex.admission_urls, ex.program_urls)

    # Application or candidacy required
    app_req_leaf = evaluator.add_leaf(
        id="application_or_candidacy_required",
        desc="Confirms admission/application/formal candidacy is required for the certificate program",
        parent=node,
        critical=True
    )
    claim_app_req = (
        "Admission, application, or formal candidacy is required for the certificate program; the certificate itself is not open enrollment."
    )
    await evaluator.verify(
        claim=claim_app_req,
        node=app_req_leaf,
        sources=adm_sources,
        additional_instruction="Confirm that the certificate program requires an application/admission or formal candidacy process (individual course enrollment may be open, but the certificate is not open enrollment)."
    )

    # Process described
    process_leaf = evaluator.add_leaf(
        id="process_described",
        desc="Provides a description of the admission/application process",
        parent=node,
        critical=True
    )
    process_text = ex.admission_process_desc or ""
    claim_process = f"The admission/application process is described as: {process_text}"
    await evaluator.verify(
        claim=claim_process,
        node=process_leaf,
        sources=adm_sources,
        additional_instruction="Verify that the admission/application process is explicitly described on the cited pages."
    )

    # Admission citations present
    evaluator.add_custom_node(
        result=_has_at_least_one_url(ex.admission_urls) or _has_at_least_one_url(ex.program_urls),
        id="admission_citations",
        desc="Provides reference URL(s) supporting the admission/application requirement and process description",
        parent=node,
        critical=True
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
    Evaluate an answer for the Creative Writing continuing education certificate program task.
    """
    # Initialize evaluator (root is non-critical by design; critical gating is handled by child nodes)
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

    # Extract structured program info from the answer
    ex = await evaluator.extract(
        prompt=prompt_extract_program(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction"
    )

    # Build verification tree according to rubric
    await verify_program_identification_and_eligibility(evaluator, root, ex)
    await verify_core_course_requirements(evaluator, root, ex)
    await verify_elective_and_grade_requirement(evaluator, root, ex)
    await verify_completion_time_limit(evaluator, root, ex)
    await verify_class_section_size_cap(evaluator, root, ex)
    await verify_prior_coursework_policy(evaluator, root, ex)
    await verify_transfer_policy(evaluator, root, ex)
    await verify_admission_process(evaluator, root, ex)

    # Return structured summary
    return evaluator.get_summary()