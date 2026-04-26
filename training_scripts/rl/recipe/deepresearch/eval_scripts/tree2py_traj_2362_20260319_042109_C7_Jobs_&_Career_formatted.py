import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "superintendent_opening_2025_2026"
TASK_DESCRIPTION = (
    "Identify a K-12 school district in the United States that currently has an open superintendent position that was "
    "posted or advertised between January 2025 and March 2026. For the position you identify, provide comprehensive "
    "information including: (1) The official name of the school district, (2) The state where the district is located, "
    "(3) The official title of the position, (4) Confirmation that it was posted between January 2025 and March 2026, "
    "(5) The application deadline or current status (open/closed), (6) The minimum educational degree requirement, "
    "(7) The administrative or leadership experience requirement, (8) Available salary information or compensation details, "
    "(9) The official website or job posting URL, (10) Information about where to find the detailed position requirements, "
    "(11) Details about the selection or interview process, (12) Confirmation that this is a full-time, permanent position "
    "(not interim or temporary), (13) How candidates can apply for this position, (14) Background information about the "
    "school district, and (15) Contact information for the district or hiring authority. Ensure all information is "
    "verifiable through official sources and provide reference URLs for each piece of information."
)

DATE_RANGE_START = datetime(2025, 1, 1)
DATE_RANGE_END = datetime(2026, 3, 31)
TODAY_STR = datetime.utcnow().strftime("%Y-%m-%d")

OFFICIAL_SOURCE_GUIDE = (
    "Use only official or authoritative sources for verification, such as: the district's own website (e.g., *.k12.* or the "
    "recognized official district domain), official board documents, or the official recruiting/job application portals used "
    "by districts (e.g., Frontline/AppliTrack, SchoolSpring, GovernmentJobs, Nimble, TalentEd, PeopleAdmin). Third-party "
    "blogs, news articles, or unverified aggregator pages should not be treated as authoritative unless they directly host "
    "the official posting on behalf of the district."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class FieldWithSources(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SuperintendentOpeningExtraction(BaseModel):
    # Core district identification
    district_name: Optional[FieldWithSources] = None
    district_state: Optional[FieldWithSources] = None
    entity_is_k12_district: Optional[FieldWithSources] = None

    # Position details
    position_title: Optional[FieldWithSources] = None
    posting_date: Optional[FieldWithSources] = None
    open_status_or_deadline: Optional[FieldWithSources] = None
    min_degree_requirement: Optional[FieldWithSources] = None
    leadership_experience_requirement: Optional[FieldWithSources] = None
    salary_compensation: Optional[FieldWithSources] = None
    full_time_permanent_confirmation: Optional[FieldWithSources] = None

    # Where/how and process
    posting_or_district_url: Optional[FieldWithSources] = None
    where_to_find_requirements: Optional[FieldWithSources] = None
    selection_process_details: Optional[FieldWithSources] = None
    application_instructions: Optional[FieldWithSources] = None

    # Background and contact
    district_background: Optional[FieldWithSources] = None
    contact_information: Optional[FieldWithSources] = None

    # Helpful global URLs explicitly mentioned in the answer
    district_website_url: Optional[str] = None
    job_posting_url: Optional[str] = None
    all_urls_mentioned: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_superintendent_opening() -> str:
    return f"""
Extract exactly ONE U.S. K-12 school district superintendent (or superintendent-level) opening from the answer, and structure it as JSON.

CRITICAL: Only extract information explicitly present in the answer. For every field below, also extract explicit source URL(s) that are present in the answer text. Accept URLs in plain form or markdown link form; output the actual URL string(s). Do NOT invent URLs.

For each field below, produce an object with:
- value: the textual content extracted from the answer for that field (string; return null if missing).
- sources: a list of URL strings that directly support that field as cited in the answer (empty list if none).

Required fields to extract:
1) district_name
2) district_state
3) entity_is_k12_district  (e.g., "Yes — public K-12 district", or a brief confirmation phrase)
4) position_title  (must show it's superintendent or superintendent-level)
5) posting_date  (the posted/advertised date as stated in the answer)
6) open_status_or_deadline  (e.g., "Open until filled" or a specific deadline/status as stated)
7) min_degree_requirement  (e.g., "Master’s degree required"; ensure degree level is explicit)
8) leadership_experience_requirement  (e.g., "3+ years administrative experience required")
9) salary_compensation  (e.g., a range/amount, or "competitive" if that's what the answer states)
10) posting_or_district_url  (a URL value pointing to the official posting or the district website home page)
11) where_to_find_requirements  (e.g., "See 'Qualifications' section" or "attached PDF link")
12) selection_process_details  (e.g., timeline, committee/interview details if provided)
13) full_time_permanent_confirmation  (e.g., "Full-time, permanent (not interim)")
14) application_instructions  (how to apply: portal/email/materials)
15) district_background  (short background description string)
16) contact_information  (email/phone/address/name as available)

Also extract these helpful global URL fields (strings):
- district_website_url: the district homepage URL explicitly mentioned in the answer, if present; else null.
- job_posting_url: the direct posting URL explicitly mentioned in the answer, if present; else null.

Finally:
- all_urls_mentioned: list all distinct URLs explicitly present anywhere in the answer.

Return a JSON object matching the SuperintendentOpeningExtraction schema precisely.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_url(s: Optional[str]) -> bool:
    if not s:
        return False
    s = s.strip()
    return s.startswith("http://") or s.startswith("https://")


def _filter_valid_urls(urls: List[str]) -> List[str]:
    uniq = []
    seen = set()
    for u in urls or []:
        if not isinstance(u, str):
            continue
        us = u.strip()
        if not _is_url(us):
            continue
        if us not in seen:
            seen.add(us)
            uniq.append(us)
    return uniq


def _gather_sources(primary: Optional[FieldWithSources], extracted: SuperintendentOpeningExtraction) -> List[str]:
    urls: List[str] = []
    if primary and primary.sources:
        urls.extend(primary.sources)
    # Try also to use the field value if it's itself a URL (e.g., posting_or_district_url.value)
    if primary and primary.value and _is_url(primary.value):
        urls.append(primary.value)
    # Fallback: job posting and district website URLs explicitly extracted
    if extracted.job_posting_url:
        urls.append(extracted.job_posting_url)
    if extracted.district_website_url:
        urls.append(extracted.district_website_url)
    # As last resort, include all URLs the answer mentioned
    if extracted.all_urls_mentioned:
        urls.extend(extracted.all_urls_mentioned)
    return _filter_valid_urls(urls)


def _fail_leaf_due_to_missing_sources(node) -> None:
    node.score = 0.0
    node.status = "failed"


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_superintendent_position(
    evaluator: Evaluator,
    parent_node,
    extracted: SuperintendentOpeningExtraction,
) -> None:
    """
    Build the verification tree for the superintendent opening and perform verifications.
    All children here are critical under a critical parent, matching the rubric.
    """
    # Create the main critical parallel node as described by the rubric
    main = evaluator.add_parallel(
        id="Superintendent_Position_Identification",
        desc="Identify ONE U.S. K-12 school district with a superintendent (or superintendent-level) opening posted/advertised between Jan 2025 and Mar 2026, and provide all requested details with official-source URLs for verification.",
        parent=parent_node,
        critical=True,
    )

    # Prepare leaf nodes and claims
    claims_and_sources = []  # For batch verification

    # 1) District official name with source
    node_1 = evaluator.add_leaf(
        id="District_Official_Name_With_Source",
        desc="Provide the official name of the school district AND at least one supporting official-source URL.",
        parent=main,
        critical=True,
    )
    district_name = extracted.district_name.value if extracted.district_name else None
    src_1 = _gather_sources(extracted.district_name, extracted)
    claim_1 = f"The official name of the school district is '{district_name}'." if district_name else "The official district name is provided and verifiable."
    if not src_1:
        _fail_leaf_due_to_missing_sources(node_1)
    else:
        claims_and_sources.append((
            claim_1,
            src_1,
            node_1,
            f"{OFFICIAL_SOURCE_GUIDE} Allow reasonable variants like 'Public Schools' vs 'School District' if obviously equivalent."
        ))

    # 2) District location (U.S. state) with source
    node_2 = evaluator.add_leaf(
        id="District_Location_US_State_With_Source",
        desc="Provide the U.S. state where the district is located AND a supporting official-source URL.",
        parent=main,
        critical=True,
    )
    state_val = extracted.district_state.value if extracted.district_state else None
    src_2 = _gather_sources(extracted.district_state, extracted)
    claim_2 = f"The school district is located in the U.S. state of {state_val}." if state_val else "The district's U.S. state location is provided and verifiable."
    if not src_2:
        _fail_leaf_due_to_missing_sources(node_2)
    else:
        claims_and_sources.append((
            claim_2,
            src_2,
            node_2,
            f"{OFFICIAL_SOURCE_GUIDE} State may appear as full name or postal abbreviation; either is acceptable."
        ))

    # 3) Entity is K-12 school district with source
    node_3 = evaluator.add_leaf(
        id="Entity_Is_K12_School_District_With_Source",
        desc="Confirm the entity is a U.S. K-12 school district AND provide a supporting official-source URL.",
        parent=main,
        critical=True,
    )
    src_3 = _gather_sources(extracted.entity_is_k12_district, extracted)
    claim_3 = (
        "The entity is a U.S. public K-12 school district (i.e., a public school district serving kindergarten through high school)."
    )
    if not src_3:
        _fail_leaf_due_to_missing_sources(node_3)
    else:
        claims_and_sources.append((
            claim_3,
            src_3,
            node_3,
            f"{OFFICIAL_SOURCE_GUIDE} Reject private schools, charter management organizations, or higher-education institutions as 'districts'."
        ))

    # 4) Position title superintendent-level with source
    node_4 = evaluator.add_leaf(
        id="Position_Title_Superintendent_Level_With_Source",
        desc="Provide the official title of the position, confirm it is superintendent or superintendent-level, AND provide a supporting official-source URL.",
        parent=main,
        critical=True,
    )
    title_val = extracted.position_title.value if extracted.position_title else None
    src_4 = _gather_sources(extracted.position_title, extracted)
    claim_4 = (
        f"The position is superintendent or superintendent-level with the official title '{title_val}'. "
        "Superintendent-level includes titles such as 'Superintendent', 'District Administrator' (e.g., in WI), "
        "'Chief Executive of the district', or equivalent top district executive. Assistant or deputy roles DO NOT qualify."
    )
    if not src_4:
        _fail_leaf_due_to_missing_sources(node_4)
    else:
        claims_and_sources.append((
            claim_4,
            src_4,
            node_4,
            f"{OFFICIAL_SOURCE_GUIDE} Confirm that this is the top district executive role (not assistant/deputy/interim unless explicitly permanent superintendent)."
        ))

    # 5) Posting date in range with source
    node_5 = evaluator.add_leaf(
        id="Posting_Date_In_Range_With_Source",
        desc="Confirm the job posting/advertisement date falls between January 2025 and March 2026 AND provide a supporting official-source URL.",
        parent=main,
        critical=True,
    )
    posting_date_val = extracted.posting_date.value if extracted.posting_date else None
    src_5 = _gather_sources(extracted.posting_date, extracted)
    claim_5 = (
        f"The job posting/advertisement date is {posting_date_val}, which falls between January 1, 2025 and March 31, 2026 (inclusive)."
        if posting_date_val else
        "The job posting/advertisement date falls between January 1, 2025 and March 31, 2026 (inclusive)."
    )
    if not src_5:
        _fail_leaf_due_to_missing_sources(node_5)
    else:
        claims_and_sources.append((
            claim_5,
            src_5,
            node_5,
            f"{OFFICIAL_SOURCE_GUIDE} If multiple dates appear (posted/updated), prefer the earliest clearly labeled 'posted' or 'advertised' date. "
            f"Accept any date in the inclusive range 2025-01-01 through 2026-03-31."
        ))

    # 6) Open status and deadline/status with source
    node_6 = evaluator.add_leaf(
        id="Open_Status_And_DeadlineOrStatus_With_Source",
        desc="Confirm (with an official-source URL) that the position is currently open, and provide either (a) the application deadline/closing date OR (b) an explicit current status indicator (e.g., 'open') as stated in official materials.",
        parent=main,
        critical=True,
    )
    open_deadline_val = extracted.open_status_or_deadline.value if extracted.open_status_or_deadline else None
    src_6 = _gather_sources(extracted.open_status_or_deadline, extracted)
    claim_6 = (
        f"The official materials indicate the position is open for applications and provide either a deadline or an explicit open status. Extracted status/deadline: '{open_deadline_val}'."
    )
    if not src_6:
        _fail_leaf_due_to_missing_sources(node_6)
    else:
        claims_and_sources.append((
            claim_6,
            src_6,
            node_6,
            f"{OFFICIAL_SOURCE_GUIDE} Look for phrases like 'Open until filled', 'Apply by', an application deadline, or a visible 'Open' status. Today is {TODAY_STR}."
        ))

    # 7) Minimum degree requirement (Master’s+) with source
    node_7 = evaluator.add_leaf(
        id="Minimum_Degree_Requirement_Masters_Plus_With_Source",
        desc="State the minimum educational degree requirement and confirm it is at least a master’s degree AND provide a supporting official-source URL.",
        parent=main,
        critical=True,
    )
    degree_val = extracted.min_degree_requirement.value if extracted.min_degree_requirement else None
    src_7 = _gather_sources(extracted.min_degree_requirement, extracted)
    claim_7 = (
        f"The minimum educational degree requirement for the position is at least a master's degree (e.g., Master's, Ed.S., Ed.D., or Ph.D.). Extracted: '{degree_val}'."
    )
    if not src_7:
        _fail_leaf_due_to_missing_sources(node_7)
    else:
        claims_and_sources.append((
            claim_7,
            src_7,
            node_7,
            f"{OFFICIAL_SOURCE_GUIDE} Confirm the minimum degree requirement; at least Master's degree is required for a PASS."
        ))

    # 8) Leadership/administrative experience requirement with source
    node_8 = evaluator.add_leaf(
        id="Leadership_Experience_Requirement_With_Source",
        desc="State the required administrative/leadership experience requirement in education AND provide a supporting official-source URL.",
        parent=main,
        critical=True,
    )
    leadership_val = extracted.leadership_experience_requirement.value if extracted.leadership_experience_requirement else None
    src_8 = _gather_sources(extracted.leadership_experience_requirement, extracted)
    claim_8 = f"The position requires administrative or leadership experience in education. Extracted requirement: '{leadership_val}'."
    if not src_8:
        _fail_leaf_due_to_missing_sources(node_8)
    else:
        claims_and_sources.append((
            claim_8,
            src_8,
            node_8,
            f"{OFFICIAL_SOURCE_GUIDE} Verify years or nature of leadership/administrative experience as stated."
        ))

    # 9) Salary/compensation info with source
    node_9 = evaluator.add_leaf(
        id="Salary_Or_Compensation_Info_With_Source",
        desc="Provide salary/compensation information (range/amount) OR an explicit compensation description (e.g., 'competitive') as stated in the posting/materials AND provide a supporting official-source URL.",
        parent=main,
        critical=True,
    )
    salary_val = extracted.salary_compensation.value if extracted.salary_compensation else None
    src_9 = _gather_sources(extracted.salary_compensation, extracted)
    claim_9 = f"The official materials provide salary or compensation information: '{salary_val}'. A range, amount, or explicit descriptor like 'competitive' counts."
    if not src_9:
        _fail_leaf_due_to_missing_sources(node_9)
    else:
        claims_and_sources.append((
            claim_9,
            src_9,
            node_9,
            f"{OFFICIAL_SOURCE_GUIDE} Accept explicit compensation text even if a numeric range is not listed, if that's what the official posting states."
        ))

    # 10) Official posting or district website URL (direct link)
    node_10 = evaluator.add_leaf(
        id="Official_Posting_Or_District_Website_URL",
        desc="Provide the official district website and/or official job posting URL for the position (a direct link).",
        parent=main,
        critical=True,
    )
    # For this, we want to verify the URL(s) themselves are official posting or district website
    # Use the 'posting_or_district_url' value if URL, else fall back to the extracted job/district URLs
    url_candidates: List[str] = []
    if extracted.posting_or_district_url and extracted.posting_or_district_url.value and _is_url(extracted.posting_or_district_url.value):
        url_candidates.append(extracted.posting_or_district_url.value)
    url_candidates.extend(_gather_sources(extracted.posting_or_district_url, extracted))
    url_candidates = _filter_valid_urls(url_candidates)
    claim_10 = "This webpage is the official district website or the official job posting for the superintendent opening."
    if not url_candidates:
        _fail_leaf_due_to_missing_sources(node_10)
    else:
        claims_and_sources.append((
            claim_10,
            url_candidates,
            node_10,
            f"{OFFICIAL_SOURCE_GUIDE} The page should clearly be the district homepage or the direct job posting maintained by/for the district."
        ))

    # 11) Where to find detailed requirements with source
    node_11 = evaluator.add_leaf(
        id="Where_To_Find_Detailed_Requirements_With_Source",
        desc="State where the detailed position requirements/qualifications can be found (e.g., section of posting, attached PDF, linked page) AND provide the supporting URL(s).",
        parent=main,
        critical=True,
    )
    req_loc_val = extracted.where_to_find_requirements.value if extracted.where_to_find_requirements else None
    src_11 = _gather_sources(extracted.where_to_find_requirements, extracted)
    claim_11 = f"The detailed position requirements/qualifications can be found at: {req_loc_val}."
    if not src_11:
        _fail_leaf_due_to_missing_sources(node_11)
    else:
        claims_and_sources.append((
            claim_11,
            src_11,
            node_11,
            f"{OFFICIAL_SOURCE_GUIDE} Look for 'Qualifications', 'Minimum Requirements', or a linked/attached PDF or page."
        ))

    # 12) Selection/interview process details with source
    node_12 = evaluator.add_leaf(
        id="Selection_Interview_Process_Details_With_Source",
        desc="Provide details about the selection/interview/hiring process (as available in official materials) AND provide a supporting official-source URL.",
        parent=main,
        critical=True,
    )
    sel_proc_val = extracted.selection_process_details.value if extracted.selection_process_details else None
    src_12 = _gather_sources(extracted.selection_process_details, extracted)
    claim_12 = f"The official materials include details about the selection/interview/hiring process: '{sel_proc_val}'."
    if not src_12:
        _fail_leaf_due_to_missing_sources(node_12)
    else:
        claims_and_sources.append((
            claim_12,
            src_12,
            node_12,
            f"{OFFICIAL_SOURCE_GUIDE} Accept any explicit process detail: timelines, rounds, committee/board interviews, etc., if stated."
        ))

    # 13) Full-time permanent (not interim) with source
    node_13 = evaluator.add_leaf(
        id="Full_Time_Permanent_Not_Interim_With_Source",
        desc="Confirm the role is full-time and permanent and NOT interim/temporary AND provide a supporting official-source URL.",
        parent=main,
        critical=True,
    )
    ftp_val = extracted.full_time_permanent_confirmation.value if extracted.full_time_permanent_confirmation else None
    src_13 = _gather_sources(extracted.full_time_permanent_confirmation, extracted)
    claim_13 = f"The role is full-time and permanent (not interim or temporary). Extracted confirmation: '{ftp_val}'."
    if not src_13:
        _fail_leaf_due_to_missing_sources(node_13)
    else:
        claims_and_sources.append((
            claim_13,
            src_13,
            node_13,
            f"{OFFICIAL_SOURCE_GUIDE} Reject postings that are explicitly 'Interim' or 'Temporary'. Accept if the materials clearly say 'full-time', 'regular', or similar."
        ))

    # 14) How to apply with source
    node_14 = evaluator.add_leaf(
        id="How_To_Apply_With_Source",
        desc="Explain how candidates can apply (method/portal/email/mail and any required materials if stated) AND provide a supporting official-source URL.",
        parent=main,
        critical=True,
    )
    apply_val = extracted.application_instructions.value if extracted.application_instructions else None
    src_14 = _gather_sources(extracted.application_instructions, extracted)
    claim_14 = f"Candidates can apply for this position as follows: '{apply_val}'."
    if not src_14:
        _fail_leaf_due_to_missing_sources(node_14)
    else:
        claims_and_sources.append((
            claim_14,
            src_14,
            node_14,
            f"{OFFICIAL_SOURCE_GUIDE} Verify application portal link, email address, required materials, or submission instructions."
        ))

    # 15) District background with source
    node_15 = evaluator.add_leaf(
        id="District_Background_With_Source",
        desc="Provide background information about the district AND provide a supporting official-source URL.",
        parent=main,
        critical=True,
    )
    bg_val = extracted.district_background.value if extracted.district_background else None
    src_15 = _gather_sources(extracted.district_background, extracted)
    claim_15 = f"Background information about the district: '{bg_val}'."
    if not src_15:
        _fail_leaf_due_to_missing_sources(node_15)
    else:
        claims_and_sources.append((
            claim_15,
            src_15,
            node_15,
            f"{OFFICIAL_SOURCE_GUIDE} Background can include size, number of schools, student population, geographic context, as stated on official sources."
        ))

    # 16) Contact information with source
    node_16 = evaluator.add_leaf(
        id="Contact_Information_With_Source",
        desc="Provide contact information for the district or hiring authority (email/phone/address/name as available) AND provide a supporting official-source URL.",
        parent=main,
        critical=True,
    )
    contact_val = extracted.contact_information.value if extracted.contact_information else None
    src_16 = _gather_sources(extracted.contact_information, extracted)
    claim_16 = f"Contact information for the district or hiring authority: '{contact_val}'."
    if not src_16:
        _fail_leaf_due_to_missing_sources(node_16)
    else:
        claims_and_sources.append((
            claim_16,
            src_16,
            node_16,
            f"{OFFICIAL_SOURCE_GUIDE} Verify that the contact (email/phone/address/name) is presented on an official page related to the district or posting."
        ))

    # Run batch verification for those leaves that have sources
    if claims_and_sources:
        await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the superintendent opening task and return a structured result.
    """
    # Initialize evaluator (root is non-critical; we will add a critical child node as per rubric)
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

    # Extract structured information from the answer
    extracted: SuperintendentOpeningExtraction = await evaluator.extract(
        prompt=prompt_extract_superintendent_opening(),
        template_class=SuperintendentOpeningExtraction,
        extraction_name="superintendent_opening_extraction",
    )

    # Add a concise summary to the report
    evaluator.add_custom_info(
        info={
            "district_name": (extracted.district_name.value if extracted.district_name else None),
            "state": (extracted.district_state.value if extracted.district_state else None),
            "position_title": (extracted.position_title.value if extracted.position_title else None),
            "posting_date": (extracted.posting_date.value if extracted.posting_date else None),
            "job_posting_url": extracted.job_posting_url,
            "district_website_url": extracted.district_website_url,
            "note": f"Target posting window: {DATE_RANGE_START.date()} to {DATE_RANGE_END.date()} (inclusive). Today: {TODAY_STR}."
        },
        info_type="summary",
        info_name="extracted_overview"
    )

    # Build tree and run verifications
    await build_and_verify_superintendent_position(evaluator, root, extracted)

    # Return final structured summary
    return evaluator.get_summary()