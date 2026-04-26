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
TASK_ID = "urban_district_admin_role"
TASK_DESCRIPTION = (
    "Identify one large urban school district in the United States that is currently hiring for a central office "
    "administrative position at the Assistant Superintendent level or Director level. For this position, provide the "
    "following information: (1) the official name of the school district, (2) the U.S. state where the district is "
    "located, (3) the exact job title of the position, (4) the minimum educational degree required, (5) the required "
    "or preferred field of study for the degree, (6) the minimum years of teaching experience required if specified, "
    "(7) the minimum years of administrative or leadership experience required if specified, (8) whether state "
    "administrative certification or licensure is required, (9) whether the position is full-time, part-time, or "
    "contract-based, (10) the specific department or division this position oversees, (11) the application deadline "
    "or closing date, and (12) a valid URL reference to the official job posting. All information must be verifiable "
    "from current job postings or official district career pages."
)

CURRENT_DATE_ISO = "2026-03-22"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PostingExtraction(BaseModel):
    # District basics
    district_name: Optional[str] = None
    district_state: Optional[str] = None

    # Sources for district identity/qualification
    district_official_url: Optional[str] = None  # e.g., district homepage/about/contact or careers root
    large_urban_evidence_urls: List[str] = Field(default_factory=list)  # official URLs explicitly saying "urban" and "large/major"
    extra_official_urls: List[str] = Field(default_factory=list)  # any additional official district/careers URLs cited

    # Position basics
    job_title: Optional[str] = None
    job_posting_url: Optional[str] = None  # required main evidence URL

    # Required Posting Details
    minimum_degree_required: Optional[str] = None  # e.g., "Master's", "Doctorate", etc.
    field_of_study: Optional[str] = None  # "not specified" if not present; else explicit wording
    min_teaching_experience_years: Optional[str] = None  # "not specified" if not present; else something like "3 years"
    min_admin_leadership_experience_years: Optional[str] = None  # "not specified" if not present
    certification_or_licensure_status: Optional[str] = None  # one of: "required", "preferred", "not mentioned"
    employment_type: Optional[str] = None  # one of: "full-time", "part-time", "contract-based", "not specified"
    department_or_division_overseen: Optional[str] = None  # "not specified" if not present
    application_deadline: Optional[str] = None  # explicit date or "Open until filled" or "not specified"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_posting_info() -> str:
    return """
    Extract exactly one qualifying district and one qualifying position from the answer text.

    IMPORTANT: Only extract information that is explicitly present in the answer text. Do not invent anything.
    For URL fields, only return valid URLs that are explicitly present in the answer (plain URL or markdown link).

    Return a JSON object with these fields:
    - district_name: string | null
    - district_state: string | null
    - district_official_url: string | null  (official district page such as homepage, about, HR/careers)
    - large_urban_evidence_urls: string[]   (official pages that explicitly describe the district as "urban" and "large/major"/"one of the largest")
    - extra_official_urls: string[]         (any additional official district or careers URLs cited in the answer)
    - job_title: string | null              (the exact job title as written in the answer)
    - job_posting_url: string | null        (a direct URL to the official job posting or an official careers page that lists the position)
    - minimum_degree_required: string | null   (the minimum degree explicitly required in the posting)
    - field_of_study: string | null            (if specified in the posting, return the required/preferred field wording; otherwise return "not specified")
    - min_teaching_experience_years: string | null  (if the posting specifies a minimum years requirement for teaching, return it; otherwise return "not specified")
    - min_admin_leadership_experience_years: string | null  (if specified, return it; otherwise "not specified")
    - certification_or_licensure_status: string | null  (one of: "required", "preferred", "not mentioned")
    - employment_type: string | null         (one of: "full-time", "part-time", "contract-based", or "not specified")
    - department_or_division_overseen: string | null  (explicit department/division/area; if not clear, return "not specified")
    - application_deadline: string | null    (explicit closing date or wording like "Open until filled"; if absent, return "not specified")

    URL extraction rules:
    - Extract only explicit URLs from the answer text; do not infer or construct.
    - Include the full URL with protocol.
    - If a URL appears in a markdown link, extract the URL target.

    If any requested field is not present in the answer, set it to null. For fields that ask to return "not specified"
    when absent from the posting (e.g., field_of_study), return exactly "not specified" if the posting does not mention it.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_nonempty(seq: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        if not x:
            continue
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _sources_for_district(data: PostingExtraction) -> List[str]:
    # Prefer explicit large/urban evidence URLs; otherwise fall back to district official page and job posting
    urls = list(data.large_urban_evidence_urls or [])
    if data.district_official_url:
        urls.append(data.district_official_url)
    if data.job_posting_url:
        urls.append(data.job_posting_url)
    urls.extend(data.extra_official_urls or [])
    return _unique_nonempty(urls)


async def _verify_with_sources_or_fail(
    evaluator: Evaluator,
    *,
    node_id: str,
    desc: str,
    parent,
    claim: str,
    sources: Optional[List[str]],
    additional_instruction: str,
    critical: bool = True,
) -> None:
    """
    Helper: If sources are missing/empty, mark the node as failed (critical). Otherwise, run a URL-grounded verification.
    """
    sources_list = _unique_nonempty(sources or [])
    if not sources_list:
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=f"{desc} (failed due to missing official source URL in the answer)",
            parent=parent,
            critical=critical,
        )
        return

    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources_list,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_district_info_checks(evaluator: Evaluator, parent, data: PostingExtraction) -> None:
    """
    Build 'District_Info_and_Qualification' subtree (all critical).
    """
    district_node = evaluator.add_parallel(
        id="District_Info_and_Qualification",
        desc="The selected district is a large urban school district in the United States, and the answer provides its basic identification details with official support.",
        parent=parent,
        critical=True,
    )

    # 1) District Name (must be supported by official page/posting)
    await _verify_with_sources_or_fail(
        evaluator,
        node_id="District_Name",
        desc="Provide the official name of the school district, supported by an official district page/posting citation.",
        parent=district_node,
        claim=f"The official name of the school district is '{(data.district_name or '').strip()}'.",
        sources=_unique_nonempty(
            [
                data.job_posting_url,
                data.district_official_url,
                * (data.extra_official_urls or []),
            ]
        ),
        additional_instruction=(
            "Confirm the exact district name appears clearly on the cited official page(s) (posting or district-owned page). "
            "Minor variations like 'Public Schools' vs 'School District' can be accepted only if the page clearly uses that official variant. "
            "Reject pages that are clearly school-specific rather than district-wide."
        ),
        critical=True,
    )

    # 2) District State (must be supported by official page/posting)
    await _verify_with_sources_or_fail(
        evaluator,
        node_id="District_State",
        desc="Provide the U.S. state where the district is located, supported by an official district page/posting citation.",
        parent=district_node,
        claim=f"The district is located in the U.S. state of '{(data.district_state or '').strip()}'.",
        sources=_unique_nonempty(
            [
                data.job_posting_url,
                data.district_official_url,
                * (data.extra_official_urls or []),
            ]
        ),
        additional_instruction=(
            "Verify that the cited page shows the state in the address, header/footer, 'About' content, or page metadata. "
            "If the state is not explicitly findable on the page, mark incorrect."
        ),
        critical=True,
    )

    # 3) Large Urban Qualification Evidence (must be explicitly stated on official sources)
    await _verify_with_sources_or_fail(
        evaluator,
        node_id="Large_Urban_Qualification_Evidence",
        desc='Provide an official citation (district or posting page) that explicitly describes the district as "urban" and "large/major/one of the largest".',
        parent=district_node,
        claim=(
            "An official district source explicitly describes the district as an urban and large/major district "
            "(e.g., the page includes the words 'urban' and either 'large', 'largest', 'major', or similar)."
        ),
        sources=_sources_for_district(data),
        additional_instruction=(
            "Only accept if the official district-owned page (or the official posting/careers page) explicitly uses wording like "
            "'urban' and 'large', 'largest', 'major', 'one of the largest', or similar in reference to the district. "
            "Do not accept third-party news or ranking sites. The evidence must be visible on the provided official page(s)."
        ),
        critical=True,
    )


async def build_position_qualification_checks(evaluator: Evaluator, parent, data: PostingExtraction) -> None:
    """
    Build 'Position_Qualification' subtree (all critical).
    """
    pos_node = evaluator.add_parallel(
        id="Position_Qualification",
        desc="The selected role is an open central-office administrative position at the Assistant Superintendent or Director level, supported by the official posting/careers page.",
        parent=parent,
        critical=True,
    )

    # 1) Official Posting URL (valid, accessible, and official)
    await _verify_with_sources_or_fail(
        evaluator,
        node_id="Official_Posting_URL",
        desc="Provide a valid, accessible URL to the official job posting or official district careers page where the position is listed.",
        parent=pos_node,
        claim=(
            "This webpage is an official district job posting or an official district careers page that lists the position, "
            f"and it pertains to the district '{(data.district_name or '').strip()}'."
        ),
        sources=_unique_nonempty([data.job_posting_url] if data.job_posting_url else []),
        additional_instruction=(
            "Accept if the page is on the district's own domain or on a recognized official recruiting platform used directly by the district "
            "(e.g., Workday, AppliTrack, GovernmentJobs/NEOGOV, Oracle/Taleo) and clearly identifies the district/employer and the listing. "
            "If the page is inaccessible, not a posting/careers page, or does not clearly list the position for the district, mark incorrect."
        ),
        critical=True,
    )

    # 2) Exact Job Title and Level (Assistant Superintendent or Director level)
    await _verify_with_sources_or_fail(
        evaluator,
        node_id="Exact_Job_Title_and_Level",
        desc="Provide the exact job title from the posting, which must be Assistant Superintendent level or Director level.",
        parent=pos_node,
        claim=(
            f"The exact job title shown on this page is '{(data.job_title or '').strip()}', and it is an Assistant "
            "Superintendent-level OR Director-level role."
        ),
        sources=_unique_nonempty([data.job_posting_url] if data.job_posting_url else []),
        additional_instruction=(
            "Match the exact title text on the page. For level, accept 'Assistant Superintendent' (any focus area) or Director-level titles such as "
            "'Director', 'Senior Director', 'Executive Director', or 'Area Director'. "
            "Do NOT accept 'Manager', 'Coordinator', 'Supervisor', or 'Associate Superintendent' as qualifying. "
            "Titles like 'Deputy Superintendent' are not 'Assistant Superintendent' and should not count unless the posting explicitly states it's Assistant Superintendent."
        ),
        critical=True,
    )

    # 3) Central Office Status (district-level, not school-based)
    await _verify_with_sources_or_fail(
        evaluator,
        node_id="Central_Office_Status",
        desc="Provide a citation showing it is a district-level/central-office role (not a school-based role).",
        parent=pos_node,
        claim=(
            "According to the posting, this role is a district-level/central-office position (e.g., located at district headquarters "
            "or within a district department/division), not a school-based campus role."
        ),
        sources=_unique_nonempty([data.job_posting_url] if data.job_posting_url else []),
        additional_instruction=(
            "Look for indicators such as department/division names, central office location, district HQ address, "
            "or explicit statements that the position is within a district department or central services. "
            "If the role is clearly tied to a single school campus (e.g., a principal or school-based director), mark incorrect."
        ),
        critical=True,
    )

    # 4) Currently Open Status
    await _verify_with_sources_or_fail(
        evaluator,
        node_id="Currently_Open_Status",
        desc="Provide verifiable support that the posting is currently open/active.",
        parent=pos_node,
        claim=(
            f"As of {CURRENT_DATE_ISO}, the posting is currently open/active or accepting applications "
            "(e.g., shows 'Open', 'Active', an enabled 'Apply' button, 'Open until filled', or a deadline not yet passed)."
        ),
        sources=_unique_nonempty([data.job_posting_url] if data.job_posting_url else []),
        additional_instruction=(
            f"Use the page status, apply button, or dates to judge current activity as of {CURRENT_DATE_ISO}. "
            "If the page shows a closing date that has already passed or the listing is archived/closed, mark incorrect. "
            "Accept 'Open Until Filled' or similar language as open."
        ),
        critical=True,
    )


async def build_required_details_checks(evaluator: Evaluator, parent, data: PostingExtraction) -> None:
    """
    Build 'Required_Posting_Details' subtree (all critical leaves).
    """
    details = evaluator.add_parallel(
        id="Required_Posting_Details",
        desc="Provide the required attributes about the position exactly as stated in the official posting/careers page (no fabrication).",
        parent=parent,
        critical=True,
    )

    posting_sources = _unique_nonempty([data.job_posting_url] if data.job_posting_url else [])

    # 1) Minimum Degree Required
    await _verify_with_sources_or_fail(
        evaluator,
        node_id="Minimum_Degree_Required",
        desc="State the minimum educational degree required as explicitly specified in the posting.",
        parent=details,
        claim=f"The minimum educational degree required is '{(data.minimum_degree_required or '').strip()}'.",
        sources=posting_sources,
        additional_instruction=(
            "Locate the minimum degree requirement in the posting (e.g., Bachelor's, Master's, Ed.S., Doctorate). "
            "If the page does not clearly specify a minimum degree, mark incorrect."
        ),
        critical=True,
    )

    # 2) Field of Study Required or Preferred
    field = (data.field_of_study or "").strip().lower() if data.field_of_study else ""
    if field == "not specified" or field == "":
        field_claim = "The posting does not specify any required or preferred field of study for the degree."
        field_addins = (
            "Scan the entire posting for any explicit field/discipline requirements or preferences (e.g., Education, Educational Leadership/"
            "Administration, Curriculum & Instruction). If any such field is mentioned, the claim is incorrect."
        )
    else:
        field_claim = (
            f"The posting specifies the required or preferred field of study as '{(data.field_of_study or '').strip()}', "
            "and this field is explicitly education-related."
        )
        field_addins = (
            "Confirm the field wording appears on the page and is education-related (e.g., contains 'Education', "
            "'Educational Administration/Leadership', 'Curriculum & Instruction'). If the wording is not present or "
            "is not clearly education-related, mark incorrect."
        )

    await _verify_with_sources_or_fail(
        evaluator,
        node_id="Field_of_Study_Required_or_Preferred",
        desc="State the required or preferred field of study exactly as the posting says; or explicitly 'not specified' if the posting does not state it.",
        parent=details,
        claim=field_claim,
        sources=posting_sources,
        additional_instruction=field_addins,
        critical=True,
    )

    # 3) Teaching Experience Years (if specified)
    teach = (data.min_teaching_experience_years or "").strip().lower() if data.min_teaching_experience_years else ""
    if teach == "not specified" or teach == "":
        teach_claim = "The posting does not specify any minimum years of teaching experience."
        teach_addins = (
            "Scan the posting for any explicit minimum years of teaching experience. If any minimum years are mentioned, "
            "this claim is incorrect."
        )
    else:
        teach_claim = f"The posting specifies that the minimum years of teaching experience required is '{(data.min_teaching_experience_years or '').strip()}'."
        teach_addins = (
            "Verify that the page includes the stated minimum teaching years (allow minor wording variations). "
            "If a different number is shown or it is absent, mark incorrect."
        )

    await _verify_with_sources_or_fail(
        evaluator,
        node_id="Teaching_Experience_Years_if_Specified",
        desc="State the minimum years of teaching experience if the posting specifies one; otherwise state that it is not specified.",
        parent=details,
        claim=teach_claim,
        sources=posting_sources,
        additional_instruction=teach_addins,
        critical=True,
    )

    # 4) Administrative/Leadership Experience Years (if specified)
    admin_yrs = (data.min_admin_leadership_experience_years or "").strip().lower() if data.min_admin_leadership_experience_years else ""
    if admin_yrs == "not specified" or admin_yrs == "":
        admin_claim = "The posting does not specify any minimum years of administrative or leadership experience."
        admin_addins = (
            "Scan the posting for any explicit minimum years of administrative or leadership experience. "
            "If any minimum years are mentioned, this claim is incorrect."
        )
    else:
        admin_claim = f"The posting specifies that the minimum years of administrative or leadership experience required is '{(data.min_admin_leadership_experience_years or '').strip()}'."
        admin_addins = (
            "Verify that the page includes the stated minimum administrative/leadership years (allow minor wording variations). "
            "If a different number is shown or it is absent, mark incorrect."
        )

    await _verify_with_sources_or_fail(
        evaluator,
        node_id="Admin_Leadership_Experience_Years_if_Specified",
        desc="State the minimum years of administrative/leadership experience if the posting specifies one; otherwise state that it is not specified.",
        parent=details,
        claim=admin_claim,
        sources=posting_sources,
        additional_instruction=admin_addins,
        critical=True,
    )

    # 5) Certification or Licensure Status
    cert_status = (data.certification_or_licensure_status or "").strip().lower()
    if cert_status == "required":
        cert_claim = "The posting requires state administrative certification or licensure (e.g., superintendent, principal, or administrator license)."
    elif cert_status == "preferred":
        cert_claim = "The posting prefers or accepts state administrative certification or licensure but does not require it."
    else:
        cert_claim = "The posting does not mention any state administrative certification or licensure requirement or preference."

    await _verify_with_sources_or_fail(
        evaluator,
        node_id="Certification_or_Licensure_Status",
        desc="State whether state administrative certification/licensure is required, preferred, or not mentioned, as supported by the posting.",
        parent=details,
        claim=cert_claim,
        sources=posting_sources,
        additional_instruction=(
            "Look for explicit certification/licensure language (e.g., superintendent/principal/administrator license). "
            "Determine if it is required, preferred, or unmentioned. If the page contradicts the claim, mark incorrect."
        ),
        critical=True,
    )

    # 6) Employment Type
    emp_type = (data.employment_type or "").strip().lower()
    if emp_type in {"full-time", "full time"}:
        emp_claim = "The position is full-time."
    elif emp_type in {"part-time", "part time"}:
        emp_claim = "The position is part-time."
    elif emp_type in {"contract-based", "contract", "contracted"}:
        emp_claim = "The position is contract-based."
    else:
        emp_claim = "The posting does not specify whether the position is full-time, part-time, or contract-based."

    await _verify_with_sources_or_fail(
        evaluator,
        node_id="Employment_Type",
        desc="State whether the position is full-time, part-time, or contract-based as explicitly specified in the posting.",
        parent=details,
        claim=emp_claim,
        sources=posting_sources,
        additional_instruction=(
            "Verify the employment type on the page. If the page clearly indicates full-time, part-time, or contract status, "
            "it must match the claim. If no such status is visible, the 'not specified' claim is acceptable."
        ),
        critical=True,
    )

    # 7) Department or Division Overseen
    dept = (data.department_or_division_overseen or "").strip()
    if dept == "" or dept.lower() == "not specified":
        dept_claim = "The posting does not clearly state a specific department, division, or area that this role oversees."
        dept_addins = (
            "Scan the posting for explicit oversight of a department/division/office/area. If clearly indicated, "
            "the claim is incorrect."
        )
    else:
        dept_claim = f"The position oversees the following department/division/area: '{dept}'."
        dept_addins = (
            "Confirm the page explicitly indicates oversight of the stated department/division/area (allow reasonable synonyms)."
        )

    await _verify_with_sources_or_fail(
        evaluator,
        node_id="Department_or_Division_Overseen",
        desc="State the specific department/division/area overseen as specified in the posting.",
        parent=details,
        claim=dept_claim,
        sources=posting_sources,
        additional_instruction=dept_addins,
        critical=True,
    )

    # 8) Application Deadline or Closing Date
    await _verify_with_sources_or_fail(
        evaluator,
        node_id="Application_Deadline_or_Closing_Date",
        desc="State the application deadline/closing date as explicitly specified in the posting.",
        parent=details,
        claim=f"The application deadline or closing date is '{(data.application_deadline or '').strip()}'.",
        sources=posting_sources,
        additional_instruction=(
            "Locate the explicit application deadline/closing date or equivalent wording like 'Open Until Filled'. "
            "If the page does not state any deadline or such wording, mark incorrect."
        ),
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
    """
    Evaluate an answer for the large urban district central-office role task.
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

    # Record current date context
    evaluator.add_custom_info({"current_date": CURRENT_DATE_ISO}, info_type="context", info_name="evaluation_context")

    # Extraction
    extracted: PostingExtraction = await evaluator.extract(
        prompt=prompt_extract_posting_info(),
        template_class=PostingExtraction,
        extraction_name="position_info",
    )

    # Create a critical "task root" under the non-critical framework root to enforce all-or-nothing scoring
    task_root = evaluator.add_parallel(
        id="Task_Critical_Root",
        desc="Identify one qualifying large urban U.S. school district and one currently open central-office Assistant Superintendent- or Director-level role, and provide all required details with verifiable support from an official district job posting/careers page.",
        parent=root,
        critical=True,
    )

    # Build subtrees
    await build_district_info_checks(evaluator, task_root, extracted)
    await build_position_qualification_checks(evaluator, task_root, extracted)
    await build_required_details_checks(evaluator, task_root, extracted)

    # Return structured summary
    return evaluator.get_summary()