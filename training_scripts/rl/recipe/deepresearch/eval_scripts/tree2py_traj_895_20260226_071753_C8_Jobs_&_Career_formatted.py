import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "superintendent_positions_2026"
TASK_DESCRIPTION = """Identify four education leadership positions (Superintendent or Assistant Superintendent) in U.S. public K-12 school districts with active application periods. The positions must meet the following criteria:

1. Geographic Diversity: The four positions must be located in at least three different U.S. states.
2. Application Timeline: Each position must have an application deadline that is on or after March 1, 2026, and the deadline must be a specific date (not "open until filled" or "rolling basis").
3. District Size Diversity: At least two of the four positions must be from districts of different enrollment size categories:
   - Small district: fewer than 3,000 students
   - Medium district: 3,000 to 15,000 students
   - Large district: more than 15,000 students

For each of the four positions, provide:
- Position title (Superintendent or Assistant Superintendent)
- School district name
- State location
- Official job posting URL (from the district's website, a state education association website, or an authorized education executive search firm)
- Specific application deadline date
- Application method (online portal, email submission, or mailing address)
- Contact information for inquiries (if available)
- Salary range or minimum salary as disclosed in the job posting
- District student enrollment number
- Minimum education level required (e.g., Master's degree, Doctorate)
- Minimum years of professional experience required
- State certification or licensure requirement status (required, preferred, or not mentioned)

All information must be verifiable through the official job posting or the district's publicly accessible website."""

CURRENT_DATE = datetime(2026, 2, 26)
DEADLINE_CUTOFF = datetime(2026, 3, 1)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PositionRecord(BaseModel):
    # Basic Information
    position_title: Optional[str] = None
    district_name: Optional[str] = None
    state: Optional[str] = None
    posting_url: Optional[str] = None
    district_website_url: Optional[str] = None  # Optional district site for enrollment verification
    # Application Details
    deadline_date: Optional[str] = None  # As written in the posting
    apply_method: Optional[str] = None   # e.g., "online portal", "email", "mailing address"
    contact_info: Optional[str] = None   # Any contact info string
    # Compensation
    salary: Optional[str] = None         # salary range or minimum salary text
    # District Characteristics
    enrollment: Optional[str] = None     # student count text
    # Required Qualifications
    edu_required: Optional[str] = None
    exp_years_required: Optional[str] = None
    cert_requirement_status: Optional[str] = None  # required/preferred/not mentioned
    # Additional sources (optional)
    extra_source_urls: List[str] = Field(default_factory=list)


class PositionsExtraction(BaseModel):
    positions: List[PositionRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
Extract up to four Superintendent or Assistant Superintendent positions from the answer. Each position must be a role in a U.S. public K-12 school district. For each position, extract the following fields exactly as presented in the answer text:

Basic Information:
- position_title: The job title text (must be Superintendent or Assistant Superintendent; include modifiers like "Assistant Superintendent of Curriculum" if present).
- district_name: The name of the school district.
- state: The U.S. state of the district (full name or postal abbreviation).
- posting_url: The official job posting URL. Must be from the district website, a state education association website, or an authorized education executive search firm.
- district_website_url: If the answer includes a district homepage or district info page URL, extract it; else return null.

Application Details:
- deadline_date: The specific application deadline date (string as shown in posting; if the answer uses "open until filled" or similar, still extract that phrase).
- apply_method: The described method to apply (e.g., "online portal", "email submission", "mailing address"; include key link or address if present).
- contact_info: Any contact info provided (e.g., contact person/email/phone). If none, return null.

Compensation:
- salary: The salary range or minimum salary text from the posting.

District Characteristics:
- enrollment: The district’s student enrollment number (text as given; include commas or ranges if present).

Required Qualifications:
- edu_required: Minimum education level required (e.g., "Master's", "Doctorate").
- exp_years_required: Minimum years of professional experience required (text as provided).
- cert_requirement_status: The status of state certification/licensure requirement ("required", "preferred", or "not mentioned").

Additional sources:
- extra_source_urls: Any other URLs in the answer that relate to this position (e.g., district info pages, search firm description pages). Exclude duplicates.

Rules:
- Extract only what is explicitly in the answer. Return null for any unspecified field.
- If the answer lists more than four positions, extract only the first four.
- Use strings for all values (do not convert to numbers or dates).
"""


# --------------------------------------------------------------------------- #
# Helper functions for verification                                           #
# --------------------------------------------------------------------------- #
def _is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return bool(re.match(r"^https?://", url.strip()))


def _build_sources_list(position: PositionRecord) -> List[str]:
    urls = []
    if _is_valid_url(position.posting_url):
        urls.append(position.posting_url.strip())
    if _is_valid_url(position.district_website_url):
        urls.append(position.district_website_url.strip())
    for u in position.extra_source_urls:
        if _is_valid_url(u):
            urls.append(u.strip())
    # deduplicate while preserving order
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def _parse_enrollment_int(enroll_text: Optional[str]) -> Optional[int]:
    if not enroll_text:
        return None
    nums = re.findall(r"(\d[\d,]*)", enroll_text)
    if not nums:
        return None
    # Choose the largest numeric token to avoid issues with ranges or other numbers present
    try:
        parsed = [int(n.replace(",", "")) for n in nums]
        return max(parsed) if parsed else None
    except Exception:
        return None


def _categorize_enrollment(n: Optional[int]) -> Optional[str]:
    if n is None:
        return None
    if n < 3000:
        return "small"
    if 3000 <= n <= 15000:
        return "medium"
    return "large"


def _normalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    return re.sub(r"\s+", " ", state).strip().upper()


# --------------------------------------------------------------------------- #
# Verification for a single position                                          #
# --------------------------------------------------------------------------- #
async def verify_position(
    evaluator: Evaluator,
    parent_node,
    position: PositionRecord,
    idx: int,
) -> None:
    pos_num = idx + 1
    pos_node = evaluator.add_parallel(
        id=f"position_{pos_num}",
        desc=f"Position {pos_num} (one of four required positions)",
        parent=parent_node,
        critical=False
    )

    sources_all = _build_sources_list(position)
    posting_only = position.posting_url if _is_valid_url(position.posting_url) else None

    # 1) Role and District eligibility (critical)
    role_node = evaluator.add_parallel(
        id=f"role_and_district_{pos_num}",
        desc=f"Role and district eligibility for position {pos_num}",
        parent=pos_node,
        critical=True
    )

    # 1.a Title is Superintendent or Assistant Superintendent
    title_leaf = evaluator.add_leaf(
        id=f"title_{pos_num}",
        desc=f"Position title is Superintendent or Assistant Superintendent",
        parent=role_node,
        critical=True
    )
    title_claim = "This posting is for a Superintendent or Assistant Superintendent position."
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=posting_only,
        additional_instruction="Confirm the role title includes 'Superintendent' or 'Assistant Superintendent' (allow reasonable title variants like 'Asst Superintendent')."
    )

    # 1.b School district name is provided
    district_leaf = evaluator.add_leaf(
        id=f"district_name_{pos_num}",
        desc=f"School district name is provided",
        parent=role_node,
        critical=True
    )
    if position.district_name:
        district_claim = f"The job posting clearly identifies the employing school district as '{position.district_name}'."
    else:
        district_claim = "The job posting clearly identifies the employing school district by name."
    await evaluator.verify(
        claim=district_claim,
        node=district_leaf,
        sources=posting_only or sources_all,
        additional_instruction="Verify the page explicitly names the school district."
    )

    # 1.c U.S. state location is provided
    state_leaf = evaluator.add_leaf(
        id=f"state_{pos_num}",
        desc=f"U.S. state location is provided",
        parent=role_node,
        critical=True
    )
    if position.state:
        state_claim = f"The posting indicates the position is located in the U.S. state '{position.state}'."
    else:
        state_claim = "The posting indicates the U.S. state location for the position."
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=posting_only or sources_all,
        additional_instruction="Ensure the state location appears on the posting (or district site) in a reasonable place (header, footer, job details)."
    )

    # 1.d District is a U.S. public K-12 school district
    public_k12_leaf = evaluator.add_leaf(
        id=f"public_k12_{pos_num}",
        desc=f"District is identified as a U.S. public K-12 school district (verifiable via posting or district website)",
        parent=role_node,
        critical=True
    )
    public_k12_claim = "The employer is a U.S. public K-12 school district."
    await evaluator.verify(
        claim=public_k12_claim,
        node=public_k12_leaf,
        sources=sources_all or posting_only,
        additional_instruction="Use the page content and domain cues to determine public K-12 district status (e.g., .k12.xx.us domains, district mission, board of education mentions)."
    )

    # 2) Posting & Application details (critical)
    pa_node = evaluator.add_parallel(
        id=f"posting_and_application_{pos_num}",
        desc=f"Posting/source and application details for position {pos_num}",
        parent=pos_node,
        critical=True
    )

    # 2.a Official job posting URL provided from approved source
    official_url_leaf = evaluator.add_leaf(
        id=f"official_url_{pos_num}",
        desc=f"Official job posting URL is provided from the district website, a state education association website, or an authorized education executive search firm",
        parent=pa_node,
        critical=True,
        score=0.0 if not posting_only else 0.0,
        status="initialized"
    )
    if not posting_only:
        # Fail immediately if no posting URL present
        official_url_leaf.score = 0.0
        official_url_leaf.status = "failed"
    else:
        official_url_claim = "This URL is an official job posting page from either a school district website, a state education association website, or an authorized education executive search firm."
        await evaluator.verify(
            claim=official_url_claim,
            node=official_url_leaf,
            sources=posting_only,
            additional_instruction="Judge both domain origin and page content. Examples of authorized search firms include HYA, McPherson & Jacobson, Ray & Associates, etc."
        )

    # 2.b Deadline is a specific date and on/after March 1, 2026
    deadline_leaf = evaluator.add_leaf(
        id=f"deadline_{pos_num}",
        desc=f"Application deadline is a specific calendar date (not rolling/open-until-filled) and is on or after March 1, 2026",
        parent=pa_node,
        critical=True
    )
    deadline_claim = "The posting specifies a concrete calendar application deadline, not 'open until filled' or 'rolling', and that date is on or after March 1, 2026."
    await evaluator.verify(
        claim=deadline_claim,
        node=deadline_leaf,
        sources=posting_only,
        additional_instruction=f"Assume today's date is {CURRENT_DATE.strftime('%B %d, %Y')}. Treat 'first review date' as not a deadline. The date must be >= March 1, 2026."
    )

    # 2.c Application period is active/open
    active_leaf = evaluator.add_leaf(
        id=f"active_application_{pos_num}",
        desc=f"Job posting indicates the application period is active/open (i.e., accepting applications; not marked closed/filled/expired)",
        parent=pa_node,
        critical=True
    )
    active_claim = f"As of {CURRENT_DATE.strftime('%B %d, %Y')}, the posting indicates the application period is active/open and accepting applications (not closed/filled/expired)."
    await evaluator.verify(
        claim=active_claim,
        node=active_leaf,
        sources=posting_only,
        additional_instruction="Consider the presence of an apply link, instructions, and a future deadline. If the page explicitly says closed/filled/expired, fail."
    )

    # 2.d Application method is provided
    method_leaf = evaluator.add_leaf(
        id=f"apply_method_{pos_num}",
        desc=f"Application method is provided (online portal, email submission, or mailing address)",
        parent=pa_node,
        critical=True
    )
    if position.apply_method:
        method_claim = f"The posting provides an application method: {position.apply_method}."
    else:
        method_claim = "The posting provides an application method (online portal, email submission, or mailing address)."
    await evaluator.verify(
        claim=method_claim,
        node=method_leaf,
        sources=posting_only,
        additional_instruction="Look for 'apply' buttons/links, email addresses, or mailing instructions."
    )

    # 2.e Contact information for inquiries is provided
    contact_leaf = evaluator.add_leaf(
        id=f"contact_{pos_num}",
        desc=f"Contact information for inquiries is provided in the job posting or the district's publicly accessible website",
        parent=pa_node,
        critical=True
    )
    if position.contact_info:
        contact_claim = f"The posting or district site provides contact information for inquiries, such as '{position.contact_info}'."
    else:
        contact_claim = "The posting or district site provides contact information for inquiries (e.g., email or phone)."
    await evaluator.verify(
        claim=contact_claim,
        node=contact_leaf,
        sources=sources_all,
        additional_instruction="Contact information may be in the posting footer, HR section, or district HR page."
    )

    # 3) Compensation (critical)
    comp_node = evaluator.add_parallel(
        id=f"compensation_{pos_num}",
        desc=f"Compensation for position {pos_num}",
        parent=pos_node,
        critical=True
    )

    salary_leaf = evaluator.add_leaf(
        id=f"salary_{pos_num}",
        desc=f"Salary range or minimum salary is disclosed in the job posting",
        parent=comp_node,
        critical=True
    )
    if position.salary:
        salary_claim = f"The job posting discloses compensation information, specifically: {position.salary}."
    else:
        salary_claim = "The job posting discloses compensation information, including a salary range or minimum salary."
    await evaluator.verify(
        claim=salary_claim,
        node=salary_leaf,
        sources=posting_only,
        additional_instruction="Look for salary range, minimum salary, or compensation section within the posting."
    )

    # 4) District characteristics (critical)
    district_char_node = evaluator.add_parallel(
        id=f"district_characteristics_{pos_num}",
        desc=f"District characteristics for position {pos_num}",
        parent=pos_node,
        critical=True
    )

    enrollment_leaf = evaluator.add_leaf(
        id=f"enrollment_{pos_num}",
        desc=f"District student enrollment number is provided with a verifiable source (posting or district website)",
        parent=district_char_node,
        critical=True
    )
    if position.enrollment:
        enrollment_claim = f"The district’s student enrollment is stated (e.g., '{position.enrollment}')."
    else:
        enrollment_claim = "The district’s student enrollment number is stated on the job posting or district website."
    await evaluator.verify(
        claim=enrollment_claim,
        node=enrollment_leaf,
        sources=sources_all or posting_only,
        additional_instruction="Accept phrasing like 'Serving approximately 10,500 students'."
    )

    # 5) Qualifications (critical)
    qual_node = evaluator.add_parallel(
        id=f"qualifications_{pos_num}",
        desc=f"Required qualifications for position {pos_num}",
        parent=pos_node,
        critical=True
    )

    edu_leaf = evaluator.add_leaf(
        id=f"education_req_{pos_num}",
        desc=f"Minimum education level required is stated",
        parent=qual_node,
        critical=True
    )
    edu_claim = "The posting states the minimum education level required (e.g., Master's degree or Doctorate)."
    await evaluator.verify(
        claim=edu_claim,
        node=edu_leaf,
        sources=posting_only,
        additional_instruction="Search qualifications section for explicit education requirements."
    )

    exp_leaf = evaluator.add_leaf(
        id=f"experience_req_{pos_num}",
        desc=f"Minimum years of professional experience required is stated",
        parent=qual_node,
        critical=True
    )
    exp_claim = "The posting states the minimum years of professional experience required."
    await evaluator.verify(
        claim=exp_claim,
        node=exp_leaf,
        sources=posting_only,
        additional_instruction="Look for phrases like 'minimum X years' or 'at least X years of experience'."
    )

    cert_leaf = evaluator.add_leaf(
        id=f"cert_req_{pos_num}",
        desc=f"State certification/licensure requirement status is indicated (required/preferred/not mentioned)",
        parent=qual_node,
        critical=True
    )
    if position.cert_requirement_status:
        cert_claim = f"The posting indicates certification/licensure requirement status: {position.cert_requirement_status}."
    else:
        cert_claim = "The posting indicates whether state certification/licensure is required or preferred."
    await evaluator.verify(
        claim=cert_claim,
        node=cert_leaf,
        sources=posting_only,
        additional_instruction="Check for superintendent endorsement or administrative certification requirements."
    )

    # 6) Verifiability (critical)
    verif_leaf = evaluator.add_leaf(
        id=f"verifiability_{pos_num}",
        desc=f"All provided fields for position {pos_num} are verifiable via the cited official job posting and/or the district’s publicly accessible website",
        parent=pos_node,
        critical=True
    )
    # Summarize provided fields for instruction
    provided_fields = {
        "position_title": bool(position.position_title),
        "district_name": bool(position.district_name),
        "state": bool(position.state),
        "posting_url": bool(position.posting_url),
        "deadline_date": bool(position.deadline_date),
        "apply_method": bool(position.apply_method),
        "contact_info": bool(position.contact_info),
        "salary": bool(position.salary),
        "enrollment": bool(position.enrollment),
        "edu_required": bool(position.edu_required),
        "exp_years_required": bool(position.exp_years_required),
        "cert_requirement_status": bool(position.cert_requirement_status),
    }
    verif_claim = "All provided fields in this position record are explicitly stated on the job posting or the district website."
    await evaluator.verify(
        claim=verif_claim,
        node=verif_leaf,
        sources=sources_all,
        additional_instruction=f"Only judge fields that are provided in the answer. Provided flags: {provided_fields}. Each provided field should be findable on the posting or district site."
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
) -> Dict[str, Any]:
    # Initialize evaluator with PARALLEL root (critical root not allowed in framework; use critical children instead)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify four Superintendent/Assistant Superintendent positions in U.S. public K-12 school districts meeting deadline, disclosure, and diversity constraints, and provide all required verifiable fields",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract positions
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction"
    )

    # Keep exactly four positions; pad if fewer
    positions = list(extracted.positions[:4])
    while len(positions) < 4:
        positions.append(PositionRecord())

    # Build position verifications
    for idx, pos in enumerate(positions):
        await verify_position(evaluator, root, pos, idx)

    # Global constraints: Geographic diversity
    states_norm = [_normalize_state(p.state) for p in positions if _normalize_state(p.state)]
    unique_states = sorted(set(states_norm))
    geo_ok = len(unique_states) >= 3

    evaluator.add_custom_node(
        result=geo_ok,
        id="geographic_diversity",
        desc="Across the four positions, there are positions located in at least three different U.S. states",
        parent=root,
        critical=True
    )

    # Global constraints: District size diversity
    enroll_parsed = [_parse_enrollment_int(p.enrollment) for p in positions]
    categories = [_categorize_enrollment(n) for n in enroll_parsed if n is not None]
    unique_categories = sorted(set([c for c in categories if c is not None]))
    size_ok = len(unique_categories) >= 2

    evaluator.add_custom_node(
        result=size_ok,
        id="district_size_diversity",
        desc="Across the four positions, at least two are in different enrollment size categories (small <3,000; medium 3,000–15,000; large >15,000)",
        parent=root,
        critical=True
    )

    # Record custom info for transparency
    evaluator.add_custom_info(
        info={
            "unique_states": unique_states,
            "enrollment_parsed": enroll_parsed,
            "enrollment_categories": [ _categorize_enrollment(n) for n in enroll_parsed ],
            "geographic_diversity_pass": geo_ok,
            "district_size_diversity_pass": size_ok,
            "cutoff_date": DEADLINE_CUTOFF.strftime("%Y-%m-%d"),
            "current_date_assumed": CURRENT_DATE.strftime("%Y-%m-%d"),
        },
        info_type="computed_constraints",
        info_name="global_constraints_check"
    )

    # Return structured summary
    return evaluator.get_summary()