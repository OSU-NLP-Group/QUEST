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
TASK_ID = "oh_hs_ad_2026_plus"
TASK_DESCRIPTION = (
    "Identify a specific Ohio high school athletic director position that was posted in January 2026 or later and "
    "meets the following requirements: (1) The position must explicitly require an Ohio Administrative License (such as "
    "a Principal's License or equivalent administrative certification), (2) The position must explicitly require a "
    "current/valid Ohio Pupil Activity Permit, (3) The job posting must specify minimum experience requirements. For the "
    "identified position, provide the following information: school name and district name, exact position title as "
    "stated in the posting, posting date or application deadline, required educational degree (Bachelor's or Master's), "
    "minimum years of experience required and type of experience, all required training certifications mentioned in the "
    "posting or required for the Pupil Activity Permit (such as background checks, coaching certifications, CPR, First "
    "Aid, etc.), application deadline and contact information, and reference URL(s) to verify all requirements. Your "
    "answer must be fully verifiable through the provided reference URLs and must include all credential requirements "
    "explicitly stated in the job posting."
)
MIN_POST_DATE_ISO = "2026-01-01"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PositionPosting(BaseModel):
    # Posting identifiers
    school_name: Optional[str] = None
    district_name: Optional[str] = None
    position_title: Optional[str] = None

    # Dates
    posting_date: Optional[str] = None  # Keep as free-text; answer may provide "Jan 12, 2026", etc.
    application_deadline: Optional[str] = None  # If not provided in posting, keep null or a phrase like "not provided"

    # Education / Experience
    degree_requirement: Optional[str] = None  # e.g., "Bachelor's", "Master's", "Bachelor's or Master's"
    experience_min_years: Optional[str] = None  # free text like "3 years", "two (2) years", or null
    experience_type: Optional[str] = None  # e.g., "athletic administration", "coaching", etc.

    # Contact info
    contact_information: Optional[str] = None  # free text (email/phone/contact person or portal instructions)

    # URLs: posting and external credential references
    posting_urls: List[str] = Field(default_factory=list)

    # Admin license external requirement sources
    admin_license_masters_urls: List[str] = Field(default_factory=list)
    admin_license_oae015_urls: List[str] = Field(default_factory=list)

    # Pupil Activity Permit related requirement sources (either posting or official PAP sources)
    req_fundamentals_of_coaching_urls: List[str] = Field(default_factory=list)
    req_cpr_urls: List[str] = Field(default_factory=list)
    req_first_aid_for_coaches_urls: List[str] = Field(default_factory=list)
    req_concussion_training_urls: List[str] = Field(default_factory=list)
    req_sudden_cardiac_arrest_urls: List[str] = Field(default_factory=list)
    req_mental_health_training_urls: List[str] = Field(default_factory=list)
    req_background_checks_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_position() -> str:
    return """
    Extract exactly what the answer reports for ONE Ohio high school Athletic Director job posting and the supporting URLs.
    STRICT RULES:
    - Extract ONLY what the answer explicitly states. Do not infer or add information not present in the answer text.
    - For all URL fields, extract only valid URLs explicitly present in the answer (plain links or markdown links). If none, return an empty array.
    - For date fields, keep the exact text format as reported (e.g., "January 12, 2026"; "1/12/2026"; or "not provided").
    - If any requested field is not clearly provided in the answer, set it to null (for strings) or [] (for lists).

    Required JSON fields to extract from the answer:
    1) Posting identifiers (from the job posting, as reported in the answer):
       - school_name: string or null
       - district_name: string or null
       - position_title: string or null

    2) Dates (as reported in the answer from the posting):
       - posting_date: string or null (e.g., "January 12, 2026")
       - application_deadline: string or null (if the answer says it's not provided in the posting, set to a phrase like "not provided" or set null)

    3) Education / Experience (as reported from the posting in the answer):
       - degree_requirement: string or null (e.g., "Bachelor's", "Master's", "Bachelor's or Master's")
       - experience_min_years: string or null (e.g., "3 years", "two years", "5+ years")
       - experience_type: string or null (e.g., "athletic administration", "coaching", "school administration")

    4) Contact information (from the posting, as reported):
       - contact_information: string or null (e.g., email, phone, contact person, portal instructions)

    5) Posting URLs (must be direct job posting reference URLs cited by the answer):
       - posting_urls: array of URLs (can be one or more). If the answer does not provide a direct posting URL, return an empty array.

    6) External URLs for credential requirements that are NOT necessarily in the posting, but are referenced by the answer:
       - admin_license_masters_urls: array of URLs that the answer cites to support that an Ohio Administrative (e.g., Principal) License requires a master's degree.
       - admin_license_oae015_urls: array of URLs that the answer cites to support that an Ohio Administrative (e.g., Principal) License requires passing the OAE 015 Educational Leadership exam.

       Pupil Activity Permit (PAP) related requirement URLs (these can be posting URLs or official ODE/OHSAA/ORC URLs the answer cites):
       - req_fundamentals_of_coaching_urls: array of URLs
       - req_cpr_urls: array of URLs
       - req_first_aid_for_coaches_urls: array of URLs
       - req_concussion_training_urls: array of URLs
       - req_sudden_cardiac_arrest_urls: array of URLs
       - req_mental_health_training_urls: array of URLs
       - req_background_checks_urls: array of URLs
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _uniq_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                result.append(u)
    return result


def _text_or_placeholder(val: Optional[str], placeholder: str = "") -> str:
    return val if (val is not None and str(val).strip() != "") else placeholder


def _claimed_deadline_is_missing(deadline_text: Optional[str]) -> bool:
    if deadline_text is None:
        return True
    low = deadline_text.strip().lower()
    return low in {"not provided", "n/a", "none", "unknown", "no deadline", "not listed", "not stated"}


# --------------------------------------------------------------------------- #
# Verification blocks                                                         #
# --------------------------------------------------------------------------- #
async def build_and_verify_step1(evaluator: Evaluator, root, data: PositionPosting):
    """
    Step 1: Identify a qualifying posting and verify all eligibility constraints.
    Parallel critical node (all children critical).
    """
    step1 = evaluator.add_parallel(
        id="Step_1_Identify_Qualifying_Posting",
        desc="Select a single job posting that satisfies all stated eligibility constraints, and provide the posting URL used for verification.",
        parent=root,
        critical=True
    )

    posting_urls = data.posting_urls

    # 1) Ohio HS Athletic Director position
    n1 = evaluator.add_leaf(
        id="Ohio_HS_Athletic_Director_Position",
        desc="The position is for an Ohio high school athletic director.",
        parent=step1,
        critical=True
    )
    claim1 = (
        "This job posting is for an Athletic Director (or Director of Athletics) position at a high school in the state of Ohio."
    )
    add1 = (
        "Confirm BOTH: (a) the role is clearly 'Athletic Director' (allow minor title variants like 'Director of Athletics'), "
        "and (b) it is at a high school in Ohio (e.g., location in Ohio, OH address, .oh.us domains, or district explicitly in Ohio). "
        "Do not accept middle school-only or strictly district-office-only roles unless the posting clearly specifies high school."
    )

    # 2) Posting date Jan 2026 or later
    n2 = evaluator.add_leaf(
        id="Posted_Jan_2026_Or_Later",
        desc="The posting provides evidence of a posting date (or equivalent timestamp) in January 2026 or later.",
        parent=step1,
        critical=True
    )
    claim2 = (
        f"The job posting shows an explicit posting date or equivalent visible timestamp on or after January 1, 2026. "
        f"For example, the reported date may be '{_text_or_placeholder(data.posting_date, 'unknown')}'."
    )
    add2 = (
        f"Accept 'posted', 'published', or 'updated' dates, portal timestamps, or equivalent indicators that can reliably date the posting. "
        f"The date must be >= {MIN_POST_DATE_ISO}. If no date or the date is earlier than Jan 2026, mark as not supported."
    )

    # 3) Admin License explicitly required
    n3 = evaluator.add_leaf(
        id="Admin_License_Explicitly_Required",
        desc="The job posting explicitly requires an Ohio Administrative License (e.g., Principal's License or equivalent administrative certification).",
        parent=step1,
        critical=True
    )
    claim3 = (
        "The job posting explicitly states that an Ohio Administrative License is required (e.g., a Principal's License or equivalent Ohio administrator certification). "
        "It must be a requirement, not merely preferred."
    )
    add3 = (
        "Look for explicit phrases like 'Ohio Administrative License required', 'Principal's License required', "
        "'appropriate Ohio administrative certification required'. Do not accept 'preferred' or 'ability to obtain' as equivalent to 'required'."
    )

    # 4) Pupil Activity Permit explicitly current/valid
    n4 = evaluator.add_leaf(
        id="PAP_Explicitly_Required_And_Current",
        desc="The job posting explicitly requires a current/valid Ohio Pupil Activity Permit.",
        parent=step1,
        critical=True
    )
    claim4 = (
        "The job posting explicitly requires a current or valid Ohio Pupil Activity Permit (PAP). "
        "It must clearly say required or must hold; not merely 'ability to obtain'."
    )
    add4 = (
        "Look for phrases like 'current/valid Ohio Pupil Activity Permit required', 'must hold a current Ohio PAP', "
        "or equivalent unambiguous requirement language."
    )

    # 5) Minimum experience requirements are present
    n5 = evaluator.add_leaf(
        id="Minimum_Experience_Requirements_Present",
        desc="The job posting specifies minimum experience requirements.",
        parent=step1,
        critical=True
    )
    claim5 = (
        "The job posting explicitly specifies minimum experience requirements (for example, a minimum number of years and/or a defined type of experience)."
    )
    add5 = (
        "Accept language like 'minimum X years', 'at least X years', or 'demonstrated experience in athletic administration/coaching' "
        "if it clearly sets a minimum or a defined requirement."
    )

    # 6) Posting URL provided (existence check)
    evaluator.add_custom_node(
        result=bool(posting_urls and len(posting_urls) > 0),
        id="Job_Posting_URL_Provided",
        desc="Provide a reference URL (or URLs) that directly links to the job posting used for verification.",
        parent=step1,
        critical=True
    )

    # Batch verify the 5 URL-grounded checks
    await evaluator.batch_verify(
        [
            (claim1, posting_urls, n1, add1),
            (claim2, posting_urls, n2, add2),
            (claim3, posting_urls, n3, add3),
            (claim4, posting_urls, n4, add4),
            (claim5, posting_urls, n5, add5),
        ]
    )

    return step1


async def build_and_verify_step2(evaluator: Evaluator, root, data: PositionPosting):
    """
    Step 2: Report all required fields with citations and verify each with appropriate source URLs.
    Entire step is critical, and all sub-nodes are critical.
    """
    step2 = evaluator.add_parallel(
        id="Step_2_Report_Required_Information_With_Citations",
        desc="Report the requested fields for the identified position and include citations sufficient to verify each field (posting URL for posting-derived claims; external URLs for credential requirements specified in constraints).",
        parent=root,
        critical=True
    )

    posting_urls = data.posting_urls

    # ------------------- Position Identifiers ------------------- #
    pos_id = evaluator.add_parallel(
        id="Position_Identifiers",
        desc="Report identifying details of the position as stated in the posting, with citations.",
        parent=step2,
        critical=True
    )

    n_school = evaluator.add_leaf(
        id="School_Name",
        desc="Provide the school name, with citation.",
        parent=pos_id,
        critical=True
    )
    claim_school = f"The job posting shows the school name as '{_text_or_placeholder(data.school_name, 'unknown')}'."
    add_school = "Allow minor formatting variations (e.g., 'HS' vs 'High School'). Verify against the posting page."

    n_district = evaluator.add_leaf(
        id="District_Name",
        desc="Provide the district name, with citation.",
        parent=pos_id,
        critical=True
    )
    claim_district = f"The job posting shows the district name as '{_text_or_placeholder(data.district_name, 'unknown')}'."
    add_district = "Verify the district or employer entity name as displayed on the posting page."

    n_title = evaluator.add_leaf(
        id="Exact_Position_Title",
        desc="Provide the exact position title as stated in the posting, with citation.",
        parent=pos_id,
        critical=True
    )
    claim_title = f"The exact position title on the posting is '{_text_or_placeholder(data.position_title, 'unknown')}'."
    add_title = "Allow minor capitalization or punctuation variations. The meaning must match (e.g., 'Athletic Director' vs 'Director of Athletics')."

    await evaluator.batch_verify(
        [
            (claim_school, posting_urls, n_school, add_school),
            (claim_district, posting_urls, n_district, add_district),
            (claim_title, posting_urls, n_title, add_title),
        ]
    )

    # ------------------- Dates ------------------- #
    dates = evaluator.add_parallel(
        id="Dates",
        desc="Report the posting date and application deadline fields requested, with citations.",
        parent=step2,
        critical=True
    )

    n_posting_date = evaluator.add_leaf(
        id="Posting_Date_Reported",
        desc="Report the posting date (or equivalent timestamp) shown in the posting, with citation.",
        parent=dates,
        critical=True
    )
    claim_posting_date = (
        f"The posting shows a posting (or equivalent) date as '{_text_or_placeholder(data.posting_date, 'unknown')}'."
    )
    add_posting_date = (
        "Match the reported date text against the posting page (accept posted/published/updated). "
        "Minor formatting differences are acceptable as long as it is the same date."
    )

    n_deadline = evaluator.add_leaf(
        id="Application_Deadline_Reported_If_Stated",
        desc="If the posting states an application deadline, report it with citation; if none is stated, indicate that it is not provided in the posting.",
        parent=dates,
        critical=True
    )
    if _claimed_deadline_is_missing(data.application_deadline):
        claim_deadline = "The job posting does not state an application deadline."
        add_deadline = "Confirm the page does not present any explicit 'deadline' or 'closing date' for applications."
    else:
        claim_deadline = f"The posting states an application deadline of '{_text_or_placeholder(data.application_deadline)}'."
        add_deadline = "Verify that this exact or equivalent deadline text appears on the posting page."

    await evaluator.batch_verify(
        [
            (claim_posting_date, posting_urls, n_posting_date, add_posting_date),
            (claim_deadline, posting_urls, n_deadline, add_deadline),
        ]
    )

    # ------------------- Education & Admin License Details ------------------- #
    edu_admin = evaluator.add_parallel(
        id="Education_And_Admin_License_Details",
        desc="Report degree requirement from the posting and include required external admin-license references from constraints.",
        parent=step2,
        critical=True
    )

    n_degree = evaluator.add_leaf(
        id="Degree_Requirement_From_Posting",
        desc="State the educational degree requirement indicated by the posting (Bachelor's or Master's), with citation to the posting.",
        parent=edu_admin,
        critical=True
    )
    claim_degree = (
        f"The posting states that the minimum educational degree requirement is '{_text_or_placeholder(data.degree_requirement, 'unknown')}'."
    )
    add_degree = "Verify the stated minimum degree requirement text from the posting (e.g., Bachelor's required, Master's required)."

    n_master_req = evaluator.add_leaf(
        id="Admin_License_Masters_Requirement",
        desc="Provide a verifiable reference that an Ohio Administrative License requires a master's degree from an accredited university (per constraints).",
        parent=edu_admin,
        critical=True
    )
    claim_master_req = (
        "For an Ohio Administrative (e.g., Principal) License, a master's degree from an accredited university is required."
    )
    add_master_req = (
        "Confirm via official Ohio Department of Education/ODE or equivalent authoritative sources that earning an Ohio principal/administrator license requires a master's degree."
    )

    n_oae015 = evaluator.add_leaf(
        id="Admin_License_OAE_015_Requirement",
        desc="Provide a verifiable reference that an Ohio Administrative License requires passing the OAE 015 Educational Leadership exam (per constraints).",
        parent=edu_admin,
        critical=True
    )
    claim_oae015 = (
        "For an Ohio Principal/Administrator license, passing the OAE 015 Educational Leadership exam is required."
    )
    add_oae015 = (
        "Confirm using authoritative sources (e.g., Ohio Assessments for Educators or ODE) that OAE 015 Educational Leadership is a required test for Ohio principal/administrator licensure."
    )

    await evaluator.batch_verify(
        [
            (claim_degree, posting_urls, n_degree, add_degree),
            (claim_master_req, data.admin_license_masters_urls, n_master_req, add_master_req),
            (claim_oae015, data.admin_license_oae015_urls, n_oae015, add_oae015),
        ]
    )

    # ------------------- Experience Details ------------------- #
    exp = evaluator.add_parallel(
        id="Experience_Details",
        desc="Report the minimum experience requirements from the posting, with citations.",
        parent=step2,
        critical=True
    )

    n_years = evaluator.add_leaf(
        id="Minimum_Years_Of_Experience",
        desc="If the posting specifies a minimum number of years, report it with citation; otherwise indicate that a numeric minimum is not specified.",
        parent=exp,
        critical=True
    )
    if _text_or_placeholder(data.experience_min_years).strip():
        claim_years = (
            f"The posting specifies a minimum experience requirement of '{_text_or_placeholder(data.experience_min_years)}'."
        )
        add_years = (
            "Verify that a numeric minimum or unambiguous minimum threshold (e.g., 'at least X years') appears on the posting."
        )
    else:
        claim_years = (
            "The job posting does not specify a numeric minimum number of years of experience."
        )
        add_years = "Confirm that no explicit numeric minimum years is stated on the posting."

    n_type = evaluator.add_leaf(
        id="Type_Of_Experience",
        desc="Report the type of experience required (e.g., coaching, administration, athletic administration), with citation.",
        parent=exp,
        critical=True
    )
    claim_type = (
        f"The posting states the required type of experience as '{_text_or_placeholder(data.experience_type, 'unknown')}'."
    )
    add_type = (
        "Verify the specific experience domain(s) listed in the posting (e.g., athletic administration, coaching, school administration)."
    )

    await evaluator.batch_verify(
        [
            (claim_years, posting_urls, n_years, add_years),
            (claim_type, posting_urls, n_type, add_type),
        ]
    )

    # ------------------- PAP Training & Certifications ------------------- #
    pap = evaluator.add_parallel(
        id="PAP_Training_And_Certifications",
        desc="List required training/certifications mentioned in the posting or required for the Ohio Pupil Activity Permit (per constraints), with verification URLs.",
        parent=step2,
        critical=True
    )

    def make_pap_leaf(leaf_id: str, desc: str, requirement_phrase: str, specific_urls: List[str]):
        node = evaluator.add_leaf(
            id=leaf_id,
            desc=desc,
            parent=pap,
            critical=True
        )
        claim = (
            f"Either the job posting or the official Ohio Pupil Activity Permit requirements indicate that '{requirement_phrase}' is required."
        )
        add = (
            "Accept verification from the job posting itself OR authoritative Ohio sources (ODE, OHSAA, ORC) describing PAP requirements. "
            "Allow common terminology variants (e.g., CPR/AED training for CPR; First Aid for Coaches or equivalent first aid training; "
            "recognized concussion training; sudden cardiac arrest training; mental health training; FBI/BCI background checks)."
        )
        urls = _uniq_urls(specific_urls, posting_urls)
        return (claim, urls, node, add)

    claims_sources_nodes = [
        make_pap_leaf(
            "Fundamentals_Of_Coaching",
            "Include Fundamentals of Coaching requirement (per constraints) with verification URL.",
            "Fundamentals of Coaching",
            data.req_fundamentals_of_coaching_urls
        ),
        make_pap_leaf(
            "CPR_Training",
            "Include CPR training requirement (per constraints) with verification URL.",
            "CPR training",
            data.req_cpr_urls
        ),
        make_pap_leaf(
            "First_Aid_For_Coaches",
            "Include First Aid for Coaches requirement (per constraints) with verification URL.",
            "First Aid for Coaches (or equivalent first aid training for PAP)",
            data.req_first_aid_for_coaches_urls
        ),
        make_pap_leaf(
            "Concussion_Training",
            "Include Concussion Training requirement (per constraints) with verification URL.",
            "Concussion training",
            data.req_concussion_training_urls
        ),
        make_pap_leaf(
            "Sudden_Cardiac_Arrest_Training",
            "Include Sudden Cardiac Arrest Training requirement (per constraints) with verification URL.",
            "Sudden Cardiac Arrest training",
            data.req_sudden_cardiac_arrest_urls
        ),
        make_pap_leaf(
            "Mental_Health_Training",
            "Include Mental Health Training requirement (per constraints) with verification URL.",
            "Student/coach mental health training (as required in Ohio for PAP/athletics staff)",
            data.req_mental_health_training_urls
        ),
        make_pap_leaf(
            "FBI_BCI_Background_Checks",
            "Include FBI/BCI background check requirement (per constraints) with verification URL.",
            "FBI/BCI background checks",
            data.req_background_checks_urls
        ),
    ]

    await evaluator.batch_verify(claims_sources_nodes)

    # ------------------- Contact Information ------------------- #
    n_contact = evaluator.add_leaf(
        id="Contact_Information",
        desc="Provide contact information for applications/inquiries as stated in the posting, with citation.",
        parent=step2,
        critical=True
    )
    claim_contact = (
        f"The posting includes contact or application information such as email, phone, contact person, or application portal instructions: "
        f"'{_text_or_placeholder(data.contact_information, 'unknown')}'."
    )
    add_contact = "Verify presence of contact details or application submission directions on the posting page."
    await evaluator.verify(
        claim=claim_contact,
        node=n_contact,
        sources=posting_urls,
        additional_instruction=add_contact
    )

    # ------------------- Overall Verifiability (custom gate) ------------------- #
    def _overall_verifiability_ok(d: PositionPosting) -> bool:
        # Ensure at least one posting URL for posting-derived fields
        if not d.posting_urls:
            return False

        # Admin license external references must be provided (at least one URL each)
        if not d.admin_license_masters_urls:
            return False
        if not d.admin_license_oae015_urls:
            return False

        # For each PAP sub-requirement, there must be at least one URL (either posting or requirement-specific)
        pap_url_groups = [
            _uniq_urls(d.req_fundamentals_of_coaching_urls, d.posting_urls),
            _uniq_urls(d.req_cpr_urls, d.posting_urls),
            _uniq_urls(d.req_first_aid_for_coaches_urls, d.posting_urls),
            _uniq_urls(d.req_concussion_training_urls, d.posting_urls),
            _uniq_urls(d.req_sudden_cardiac_arrest_urls, d.posting_urls),
            _uniq_urls(d.req_mental_health_training_urls, d.posting_urls),
            _uniq_urls(d.req_background_checks_urls, d.posting_urls),
        ]
        if any(len(group) == 0 for group in pap_url_groups):
            return False

        # If the answer reports any of these key values, ensure at least one posting URL exists (already ensured above).
        # This check focuses on citation presence rather than content correctness (which is verified by other leaves).
        return True

    evaluator.add_custom_node(
        result=_overall_verifiability_ok(data),
        id="Overall_Verifiability",
        desc="Every reported field/requirement has at least one provided reference URL that supports it (posting URL for posting-derived claims; external URLs for credential requirements specified in constraints).",
        parent=step2,
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
    Evaluate an answer for the Ohio High School Athletic Director posting task.
    """
    # Initialize evaluator with a sequential root: Step 1 must succeed before Step 2 is meaningful
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Record reference baseline info
    evaluator.add_custom_info(
        {"min_post_date_iso": MIN_POST_DATE_ISO},
        info_type="policy",
        info_name="posting_date_policy"
    )

    # Extract structured data from the answer
    extracted: PositionPosting = await evaluator.extract(
        prompt=prompt_extract_position(),
        template_class=PositionPosting,
        extraction_name="position_posting_extraction"
    )

    # Build and verify Step 1 (critical)
    await build_and_verify_step1(evaluator, root, extracted)

    # Build and verify Step 2 (critical)
    await build_and_verify_step2(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()