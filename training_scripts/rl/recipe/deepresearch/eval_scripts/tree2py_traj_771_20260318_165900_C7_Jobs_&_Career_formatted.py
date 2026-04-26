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
TASK_ID = "pa_elem_teaching_job"
TASK_DESCRIPTION = """Find one full-time certified teaching position for elementary grades (Kindergarten through 5th grade) in a Pennsylvania public school district that is currently accepting applications. The position must be posted on an official school district employment website or online application system.

For the position you identify, provide the following information:

1. Geographic Location: Confirm the position is in Pennsylvania and provide the specific school district name
2. Grade Level: Specify the exact grade level(s) for which the position is advertised
3. Position Type: Confirm the position is full-time and requires Pennsylvania state teacher certification/licensure
4. Position Title: Provide the complete position title and subject area (if specified)
5. School Identification: Identify the specific school or indicate if it's a district-wide posting
6. Application Deadline: Provide the specific deadline date for applications
7. Position Start Date: Indicate when the position is expected to begin (e.g., August 2026, 2026-2027 school year)
8. Degree Requirement: State the minimum degree level required (e.g., bachelor's degree, master's degree)
9. Certification Requirement: Specify the type of Pennsylvania teaching certification or license required
10. Experience Requirement: State whether prior teaching experience is required and if so, how many years
11. Salary Information: Provide either the salary range/amount or a reference to where the salary schedule can be found
12. Required Application Materials: List the documents and materials required to complete the application
13. Contact Information: Provide contact information for the school district's human resources department or hiring authority
14. Job Posting URL: Provide the direct URL link to the official job posting
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TeachingPositionExtraction(BaseModel):
    # Core identification
    district_name: Optional[str] = None
    state: Optional[str] = None  # e.g., "PA" or "Pennsylvania"
    position_title: Optional[str] = None
    grade_levels: Optional[str] = None  # e.g., "1st Grade", "Elementary K-5", "PK-4"
    position_type: Optional[str] = None  # e.g., "Full-Time", "FT", etc.
    school_name_or_scope: Optional[str] = None  # e.g., "Lincoln Elementary" or "District-wide"

    # Timeline
    application_deadline: Optional[str] = None  # specific date or "Open until filled"
    start_date: Optional[str] = None  # e.g., "August 2026", "2026-2027 school year"

    # Requirements
    degree_requirement: Optional[str] = None  # e.g., "Bachelor's degree required"
    certification_requirement: Optional[str] = None  # e.g., "PA Elementary K-6", "PA PK-4"
    experience_requirement: Optional[str] = None  # e.g., "2 years preferred", "No experience required"

    # Compensation
    salary_information: Optional[str] = None  # e.g., "$52,000-$61,000" or "per CBA/salary schedule"
    salary_reference_url: Optional[str] = None  # optional additional URL to salary schedule if cited

    # Application artifacts
    required_application_materials: Optional[str] = None  # free text list as provided by answer

    # Contact
    contact_information: Optional[str] = None  # e.g., HR email/phone/address
    contact_reference_url: Optional[str] = None  # optional additional contact/HR page URL if cited

    # URL to verify everything
    job_posting_url: Optional[str] = None  # direct link to official district job posting


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_teaching_position() -> str:
    return """
    Extract details for exactly one teaching position described in the answer. Return each field verbatim from the answer text without inventing content.

    Required fields to extract (use null when missing):
    - district_name: The public school district name.
    - state: The state mentioned (e.g., "PA" or "Pennsylvania").
    - position_title: The exact job title (include subject if present).
    - grade_levels: The exact grade level(s) or scope (e.g., "1st Grade", "Elementary K-5", "PK-4", "K-6").
    - position_type: The time basis (e.g., "Full-Time", "FT", "FTE 1.0").
    - school_name_or_scope: The specific school name or "district-wide" if stated.
    - application_deadline: The application deadline date or closing condition (e.g., "Open until filled").
    - start_date: Expected start date or school year (e.g., "August 2026", "2026-2027 school year").
    - degree_requirement: The minimum degree level required (e.g., "Bachelor's required").
    - certification_requirement: The Pennsylvania teaching certification required (e.g., "PA PK-4", "PA Elementary K-6", "Instructional I/II").
    - experience_requirement: The experience requirement as stated (e.g., "2 years required", "no experience required", "experience preferred").
    - salary_information: Salary range/amount OR reference like "per the salary schedule/CBA".
    - salary_reference_url: A URL to the salary schedule/CBA if explicitly provided in the answer; otherwise null.
    - required_application_materials: The list (as text) of documents/materials required.
    - contact_information: HR or hiring contact details (email/phone/address/name) as provided.
    - contact_reference_url: A URL to the HR/contact page if explicitly provided; otherwise null.
    - job_posting_url: The direct link to the official job posting on a school district employment website or district-run application system (e.g., Frontline/AppliTrack, TalentEd/PowerSchool, Workday, NeoGov). Do not return links to third-party aggregators like Indeed, ZipRecruiter, or LinkedIn.

    Rules:
    - Return exactly what the answer states. Do not infer or browse beyond the provided answer text.
    - If a URL field is missing protocol, prepend http://.
    - If multiple items are mentioned, choose the first complete one and ignore the rest.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str], fallback: str = "") -> str:
    return s if (s is not None and str(s).strip() != "") else fallback


def _combine_urls(*urls: Optional[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if u and isinstance(u, str) and u.strip():
            if u not in seen:
                out.append(u)
                seen.add(u)
    return out


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_teaching_position(evaluator: Evaluator, parent_node, data: TeachingPositionExtraction) -> None:
    """
    Build and execute verification checks according to the rubric tree.
    """

    # Create the main node as parallel aggregation (non-critical at parent level)
    info_node = evaluator.add_parallel(
        id="teaching_position_information",
        desc="All required information about the teaching position is provided",
        parent=parent_node,
        critical=False
    )

    job_url = data.job_posting_url

    # 14. Job Posting URL (CRITICAL)
    job_url_node = evaluator.add_leaf(
        id="job_posting_url",
        desc="A direct URL link to the official job posting on the school district's employment portal is provided",
        parent=info_node,
        critical=True,
    )
    claim_job_url = (
        "This URL is a direct link to an official job posting on a public school district's employment website "
        "or its official applicant system (e.g., Frontline/AppliTrack, TalentEd/PowerSchool, Workday, NeoGov, Munis). "
        "It is not a third-party aggregator (e.g., Indeed, ZipRecruiter, LinkedIn) and not a general home page."
    )
    await evaluator.verify(
        claim=claim_job_url,
        node=job_url_node,
        sources=job_url,
        additional_instruction="Confirm that the page is a specific job posting page from a U.S. public school district or its ATS vendor. "
                               "Indicators include district name/logo, job ID, 'Apply' button, and domain consistent with official district or ATS. "
                               "If the URL is missing or points to an aggregator, mark as not supported."
    )

    # 1. Geographic Location (CRITICAL)
    geo_node = evaluator.add_leaf(
        id="geographic_location",
        desc="The position is located in Pennsylvania and the specific school district name is provided",
        parent=info_node,
        critical=True
    )
    district_name = _safe(data.district_name, "UNKNOWN")
    claim_geo = (
        f"The job posting indicates the position is in Pennsylvania (PA) and the school district name appears on the page. "
        f"The answer states the district as '{district_name}'."
    )
    await evaluator.verify(
        claim=claim_geo,
        node=geo_node,
        sources=job_url,
        additional_instruction="Pass if the page makes it clear the district is in Pennsylvania (e.g., 'PA', 'Pennsylvania', address/city+PA) "
                               "and the district name is shown on the page. Also check that the district name provided in the answer "
                               "matches or reasonably corresponds to the district name on the page."
    )

    # 2. Grade Level Match (CRITICAL)
    grade_node = evaluator.add_leaf(
        id="grade_level_match",
        desc="The position is for elementary grades (K-5) or includes elementary grades within its scope (e.g., PK-12 positions that include elementary)",
        parent=info_node,
        critical=True
    )
    claim_grade = (
        "This job posting is for an elementary grade assignment (Kindergarten through Grade 5) "
        "or is a broader certification/assignment (e.g., PK-4, K-6, 4-8, K-12) that includes elementary grades."
    )
    await evaluator.verify(
        claim=claim_grade,
        node=grade_node,
        sources=job_url,
        additional_instruction="Consider text like 'Elementary Teacher', 'Primary', 'K-5', 'PK-4', 'K-6', or 'Grades 4-8'. "
                               "If the scope explicitly covers any of K through 5, pass."
    )

    # 3. Position Type (CRITICAL)
    pos_type_node = evaluator.add_leaf(
        id="position_type",
        desc="The position is identified as full-time and requires Pennsylvania state certification/licensure",
        parent=info_node,
        critical=True
    )
    claim_pos_type = (
        "The job posting indicates the position is full-time (or equivalent, e.g., 1.0 FTE) "
        "and that Pennsylvania state teacher certification/licensure is required."
    )
    await evaluator.verify(
        claim=claim_pos_type,
        node=pos_type_node,
        sources=job_url,
        additional_instruction="Look for 'Full-Time', 'FT', '1.0 FTE', or similar. For certification, look for PA-specific terms "
                               "like 'valid Pennsylvania teaching certificate', 'PA certification', 'Instructional I/II', "
                               "'PA PK-4', 'PA K-6', etc."
    )

    # 4. Position Title (NON-CRITICAL)
    title_node = evaluator.add_leaf(
        id="position_title",
        desc="The specific position title and subject area are provided",
        parent=info_node,
        critical=False
    )
    title_text = _safe(data.position_title, "UNKNOWN")
    claim_title = f"The job posting shows the position title as '{title_text}' or an equivalent title."
    await evaluator.verify(
        claim=claim_title,
        node=title_node,
        sources=job_url,
        additional_instruction="Verify that the page's displayed job title matches or is reasonably equivalent to the answer's title, "
                               "including any subject area if specified."
    )

    # 5. School/District Identification (NON-CRITICAL)
    school_node = evaluator.add_leaf(
        id="school_district_identification",
        desc="The school district name and specific school (if applicable) are identified",
        parent=info_node,
        critical=False
    )
    school_text = _safe(data.school_name_or_scope, "not specified")
    claim_school = (
        f"The job posting identifies the school district and indicates either a specific school assignment "
        f"(e.g., '{school_text}') or that the posting is district-wide."
    )
    await evaluator.verify(
        claim=claim_school,
        node=school_node,
        sources=job_url,
        additional_instruction="Pass if the district name appears and either a specific school is listed (e.g., elementary school name) "
                               "or the posting clearly states district-wide or assignment TBD."
    )

    # 6. Application Deadline (CRITICAL)
    deadline_node = evaluator.add_leaf(
        id="application_deadline",
        desc="A specific application deadline date or closing condition is provided",
        parent=info_node,
        critical=True
    )
    deadline_text = _safe(data.application_deadline, "not specified")
    claim_deadline = (
        f"The job posting provides an application deadline or closing condition (e.g., a date or 'Open until filled'): '{deadline_text}'."
    )
    await evaluator.verify(
        claim=claim_deadline,
        node=deadline_node,
        sources=job_url,
        additional_instruction="Accept explicit dates (any common format) or phrases like 'Open until filled', 'Until filled', or similar. "
                               "Reject if no closing/accepting information is present."
    )

    # 7. Position Start Date (NON-CRITICAL)
    start_node = evaluator.add_leaf(
        id="position_start_date",
        desc="The anticipated start date or school year for the position is indicated",
        parent=info_node,
        critical=False
    )
    start_text = _safe(data.start_date, "not specified")
    claim_start = f"The job posting indicates the expected start date or school year: '{start_text}'."
    await evaluator.verify(
        claim=claim_start,
        node=start_node,
        sources=job_url,
        additional_instruction="Accept general timing phrases like 'August 2026', '2026-2027 school year', or 'ASAP/Immediately'."
    )

    # 8. Degree Requirement (NON-CRITICAL)
    degree_node = evaluator.add_leaf(
        id="degree_requirement",
        desc="The required degree level (e.g., bachelor's, master's) is specified or can be inferred from standard teaching position requirements",
        parent=info_node,
        critical=False
    )
    degree_text = _safe(data.degree_requirement, "not specified")
    claim_degree = f"The job posting specifies the required degree level or equivalent expectation: '{degree_text}'."
    await evaluator.verify(
        claim=claim_degree,
        node=degree_node,
        sources=job_url,
        additional_instruction="Commonly 'Bachelor's degree required'; accept clear equivalents. If the page implies a standard teacher degree "
                               "requirement without explicit wording, that's acceptable."
    )

    # 9. Certification Requirement (NON-CRITICAL)
    cert_node = evaluator.add_leaf(
        id="certification_requirement",
        desc="The required Pennsylvania teaching certification or license type is specified",
        parent=info_node,
        critical=False
    )
    cert_text = _safe(data.certification_requirement, "not specified")
    claim_cert = (
        f"The job posting specifies the required Pennsylvania certification/license type, such as '{cert_text}' "
        f"(e.g., 'PA PK-4', 'PA K-6', 'Grades 4-8', 'Instructional I/II', 'Special Education PK-8')."
    )
    await evaluator.verify(
        claim=claim_cert,
        node=cert_node,
        sources=job_url,
        additional_instruction="Look for explicit PA certification names/codes. Minor naming variations are acceptable."
    )

    # 10. Experience Requirement (NON-CRITICAL)
    exp_node = evaluator.add_leaf(
        id="experience_requirement",
        desc="The experience requirements are stated or it is clear whether experience is required",
        parent=info_node,
        critical=False
    )
    exp_text = _safe(data.experience_requirement, "not specified")
    claim_exp = (
        f"The job posting states the experience requirement (including 'none' or 'preferred'): '{exp_text}'."
    )
    await evaluator.verify(
        claim=claim_exp,
        node=exp_node,
        sources=job_url,
        additional_instruction="Accept explicit statements of required/preferred years, student teaching, or 'no experience required'."
    )

    # 11. Salary Information (NON-CRITICAL)
    salary_node = evaluator.add_leaf(
        id="salary_information",
        desc="Salary information is provided, either as a specific amount/range or as a reference to a salary schedule or collective bargaining agreement",
        parent=info_node,
        critical=False
    )
    salary_text = _safe(data.salary_information, "not specified")
    salary_sources = _combine_urls(job_url, data.salary_reference_url)
    claim_salary = (
        f"The job posting (or directly linked district source) provides salary information, either a range/amount or a reference "
        f"to a salary schedule/CBA: '{salary_text}'."
    )
    await evaluator.verify(
        claim=claim_salary,
        node=salary_node,
        sources=salary_sources,
        additional_instruction="Pass if the posting shows pay, step/scale info, or references an accessible salary schedule/CBA. "
                               "If only vague statements like 'competitive pay' without schedule/reference, do not pass."
    )

    # 12. Required Application Materials (NON-CRITICAL)
    materials_node = evaluator.add_leaf(
        id="required_application_materials",
        desc="The required application materials or process for submitting a complete application is specified",
        parent=info_node,
        critical=False
    )
    materials_text = _safe(data.required_application_materials, "not specified")
    claim_materials = (
        f"The job posting lists the required application materials (e.g., cover letter, resume, transcripts, certifications, references): "
        f"'{materials_text}'."
    )
    await evaluator.verify(
        claim=claim_materials,
        node=materials_node,
        sources=job_url,
        additional_instruction="Accept when the page lists specific attachments/materials or clearly describes the required application packet."
    )

    # 13. Contact Information (NON-CRITICAL)
    contact_node = evaluator.add_leaf(
        id="contact_information",
        desc="Contact information for the school district's human resources department or hiring authority is provided",
        parent=info_node,
        critical=False
    )
    contact_text = _safe(data.contact_information, "not specified")
    contact_sources = _combine_urls(job_url, data.contact_reference_url)
    claim_contact = (
        f"The district's HR/hiring contact information is provided (email/phone/address/name): '{contact_text}'."
    )
    await evaluator.verify(
        claim=claim_contact,
        node=contact_node,
        sources=contact_sources,
        additional_instruction="Pass if contact is visible on the job posting or a directly linked official district HR/contact page."
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
    """
    Evaluate an answer for the Pennsylvania elementary teaching position task.
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_teaching_position(),
        template_class=TeachingPositionExtraction,
        extraction_name="teaching_position_extraction",
    )

    # Add helpful context info (not graded)
    evaluator.add_custom_info(
        info={
            "note": "This evaluation verifies that the identified job is an official PA public school district elementary posting, "
                    "and that the provided details are supported by the official job page."
        },
        info_type="context",
        info_name="evaluation_notes"
    )

    # Build verification tree and run checks
    await verify_teaching_position(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()