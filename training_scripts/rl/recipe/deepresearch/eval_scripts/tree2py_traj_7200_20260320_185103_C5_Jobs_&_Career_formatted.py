import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ncaa_career_services_diversity_3"
TASK_DESCRIPTION = (
    "Identify three NCAA Division I universities, each from a different U.S. state and a different athletic conference. "
    "For each university, provide the following information about their career services office or center: "
    "(1) The official name of the career services office/center; "
    "(2) Complete physical address, including building name or number, street address, city, state, and ZIP code; "
    "(3) Phone number and official institutional email address; "
    "(4) Standard operating hours (days of the week and specific times); "
    "(5) Typical qualification requirements for career counselor or career services coordinator positions at the university, including: "
    "the minimum degree level required (bachelor's or master's), at least two relevant academic fields or majors typically required or preferred, "
    "whether professional certification (such as NCDA CCC or GCDF) is required or preferred, and typical years of relevant experience required for entry-level positions. "
    "For each piece of information provided, include the reference URL from the university's official website where this information was found."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Address(BaseModel):
    building: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Contact(BaseModel):
    phone: Optional[str] = None
    phone_urls: List[str] = Field(default_factory=list)
    email: Optional[str] = None
    email_urls: List[str] = Field(default_factory=list)


class OperatingHours(BaseModel):
    text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CareerCenter(BaseModel):
    name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    address: Address = Field(default_factory=Address)
    contact: Contact = Field(default_factory=Contact)
    hours: OperatingHours = Field(default_factory=OperatingHours)


class Requirements(BaseModel):
    min_degree: Optional[str] = None  # e.g., "Bachelor's", "Master's"
    fields: List[str] = Field(default_factory=list)  # at least 2
    certification: Optional[str] = None  # e.g., "required", "preferred", "not required", with details
    years_experience: Optional[str] = None  # e.g., "1 year", "0-2 years"
    urls: List[str] = Field(default_factory=list)


class UniversityInfo(BaseModel):
    university: Optional[str] = None
    state: Optional[str] = None
    conference: Optional[str] = None
    selection_urls: List[str] = Field(default_factory=list)  # optional supporting links for state/conference if provided
    career_center: CareerCenter = Field(default_factory=CareerCenter)
    requirements: Requirements = Field(default_factory=Requirements)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to three universities and their required career services details from the provided answer text. 
    The universities must be NCAA Division I, and each should be from a different U.S. state and a different athletic conference.

    For each identified university, return an object with the following structure:

    {
      "university": string | null,
      "state": string | null,       // full state name or 2-letter postal abbreviation
      "conference": string | null,  // NCAA Division I conference name (e.g., "SEC", "Big Ten", "ACC")
      "selection_urls": string[]    // URLs (if any) from the answer that help justify state/conference membership
      "career_center": {
        "name": string | null,
        "urls": string[],           // URLs cited for the official career center/office name
        "address": {
          "building": string | null,
          "street": string | null,
          "city": string | null,
          "state": string | null,
          "zip": string | null,
          "urls": string[]          // URLs cited for the physical address
        },
        "contact": {
          "phone": string | null,
          "phone_urls": string[],   // URLs cited for the phone number
          "email": string | null,
          "email_urls": string[]    // URLs cited for the institutional email address
        },
        "hours": {
          "text": string | null,    // e.g., "Mon–Fri 8:00 AM–5:00 PM"
          "urls": string[]          // URLs cited for the standard operating hours
        }
      },
      "requirements": {
        "min_degree": string | null,        // e.g., "Bachelor's" or "Master's"
        "fields": string[],                 // at least two relevant academic fields typically required or preferred (e.g., "Counseling", "Higher Education", "Human Resources", etc.)
        "certification": string | null,     // whether professional certification is required or preferred; include any names if stated (e.g., "NCDA CCC preferred", "GCDF required", or "not required")
        "years_experience": string | null,  // typical years of relevant experience for entry-level roles (e.g., "1 year", "0–2 years")
        "urls": string[]                    // URLs cited for the qualification requirements (HR pages, job classification pages, or recent official job postings)
      }
    }

    Additional rules:
    1) Extract only what is explicitly present in the answer text; do not invent.
    2) For every data point (name, address, phone, email, hours, requirements), collect the URL(s) the answer cited from the university's official website. 
       If a field is present but no URL is cited in the answer for that specific field, leave the corresponding URL array empty.
    3) Prefer official university pages (.edu or official subdomains), HR/job classification pages hosted by the university, or official university-hosted applicant portals (e.g., *.myworkdayjobs.com pages branded for the university) if those were cited in the answer.
    4) Preserve the original formatting of names, phone numbers, emails, and time ranges as shown in the answer.
    5) If fewer than three universities are present, return as many as available.

    Return a JSON object with a single key:
    {
      "universities": [ ... up to three University objects as defined above ... ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _format_address(addr: Address) -> str:
    parts: List[str] = []
    if addr.building and addr.building.strip():
        parts.append(addr.building.strip())
    if addr.street and addr.street.strip():
        parts.append(addr.street.strip())
    city_state = ", ".join([p for p in [addr.city, addr.state] if p and p.strip()])
    if addr.zip and addr.zip.strip():
        if city_state:
            parts.append(f"{city_state} {addr.zip.strip()}")
        else:
            parts.append(addr.zip.strip())
    else:
        if city_state:
            parts.append(city_state)
    return ", ".join(parts)


def _non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _all_fields_have_urls(u: UniversityInfo) -> bool:
    """
    Check that every required piece of information has at least one URL provided.
    According to the task, every piece of info should include a reference URL from the university's official website.
    """
    cc = u.career_center
    req = u.requirements
    required_url_lists = [
        cc.urls,                      # career center name
        cc.address.urls,              # address
        cc.contact.phone_urls,        # phone
        cc.contact.email_urls,        # email
        cc.hours.urls,                # hours
        req.urls,                     # requirements
    ]
    # All non-empty lists:
    return all(isinstance(lst, list) and len(lst) > 0 for lst in required_url_lists)


def _get_first_two_fields(fields: List[str]) -> List[str]:
    return [f for f in fields if _non_empty(f)][:2]


# --------------------------------------------------------------------------- #
# Verification subroutine per university                                      #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityInfo,
    index: int,
) -> None:
    """
    Build verification nodes for a single university.
    All leaves under the university node are critical as per rubric.
    """
    uni_idx = index + 1
    uni_name = uni.university or f"University #{uni_idx}"

    u_node = evaluator.add_parallel(
        id=f"university_{uni_idx}",
        desc=f"Evaluate the completeness and accuracy of career services information for university #{uni_idx}",
        parent=parent_node,
        critical=False,  # University block is non-critical at Task_Completion level (partial credit allowed)
    )

    # 1) Career Center Name
    name_leaf = evaluator.add_leaf(
        id=f"u{uni_idx}_career_center_name",
        desc="Verify that the official name of the career services office/center is provided and supported by sources",
        parent=u_node,
        critical=True,
    )
    name_claim = (
        f"The official career services office/center name at {uni_name} is '{uni.career_center.name or ''}'. "
        "This exact or equivalent official name appears on the cited page(s)."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=uni.career_center.urls,
        additional_instruction="Confirm the page clearly states the career services office/center name. Allow minor punctuation/case/style variations."
    )

    # 2) Physical Address
    addr_text = _format_address(uni.career_center.address)
    address_leaf = evaluator.add_leaf(
        id=f"u{uni_idx}_physical_address",
        desc="Verify that complete physical address is provided including building, street, city, state, and ZIP code",
        parent=u_node,
        critical=True,
    )
    address_claim = (
        f"The physical address of the career services office/center at {uni_name} is '{addr_text}'. "
        "The cited page shows an address that includes building (or suite), street, city, state, and ZIP."
    )
    await evaluator.verify(
        claim=address_claim,
        node=address_leaf,
        sources=uni.career_center.address.urls,
        additional_instruction="Check that the page contains the building (or comparable location info), street, city, state, and ZIP code for the career center."
    )

    # 3) Phone Number
    phone_leaf = evaluator.add_leaf(
        id=f"u{uni_idx}_phone_number",
        desc="Verify that official phone number in valid U.S. format is provided and supported by sources",
        parent=u_node,
        critical=True,
    )
    phone_claim = (
        f"The official career services phone number for {uni_name} is '{uni.career_center.contact.phone or ''}'. "
        "It appears on the cited page(s) and is a valid U.S. phone number format."
    )
    await evaluator.verify(
        claim=phone_claim,
        node=phone_leaf,
        sources=uni.career_center.contact.phone_urls,
        additional_instruction="Validate that the number is present on the page and formatted as a standard U.S. phone number (allow punctuation/spacing variants)."
    )

    # 4) Email Address
    email_leaf = evaluator.add_leaf(
        id=f"u{uni_idx}_email_address",
        desc="Verify that official institutional email address is provided and supported by sources",
        parent=u_node,
        critical=True,
    )
    email_claim = (
        f"The official career services email address for {uni_name} is '{uni.career_center.contact.email or ''}', "
        "and it appears on the cited page(s)."
    )
    await evaluator.verify(
        claim=email_claim,
        node=email_leaf,
        sources=uni.career_center.contact.email_urls,
        additional_instruction="Confirm the email address is present on the page, and appears to be an official institutional address (often ending with .edu or an official subdomain)."
    )

    # 5) Operating Hours
    hours_leaf = evaluator.add_leaf(
        id=f"u{uni_idx}_operating_hours",
        desc="Verify that standard operating hours including days of week and specific times are provided and supported by sources",
        parent=u_node,
        critical=True,
    )
    hours_claim = (
        f"The standard operating hours for the career services office/center at {uni_name} are '{uni.career_center.hours.text or ''}'. "
        "The cited page shows typical open days of the week and specific time ranges."
    )
    await evaluator.verify(
        claim=hours_claim,
        node=hours_leaf,
        sources=uni.career_center.hours.urls,
        additional_instruction="Confirm the page lists regular business hours (e.g., Mon–Fri with specific times). Allow minor formatting or day-range shorthand."
    )

    # 6) Degree Level Requirement
    degree_leaf = evaluator.add_leaf(
        id=f"u{uni_idx}_degree_level",
        desc="Verify that the minimum degree level required (bachelor's or master's) for counselor/coordinator positions is identified and supported by sources",
        parent=u_node,
        critical=True,
    )
    degree_claim = (
        f"For entry-level career counselor or career services coordinator roles at {uni_name}, "
        f"the minimum degree level required is '{uni.requirements.min_degree or ''}'."
    )
    await evaluator.verify(
        claim=degree_claim,
        node=degree_leaf,
        sources=uni.requirements.urls,
        additional_instruction="Focus on minimum degree level stated in HR classification pages or recent official university job postings for these roles."
    )

    # 7) Relevant Academic Fields (at least two)
    fields_two = _get_first_two_fields(uni.requirements.fields)
    fields_leaf = evaluator.add_leaf(
        id=f"u{uni_idx}_relevant_fields",
        desc="Verify that at least two relevant academic fields or majors typically required or preferred are identified and supported by sources",
        parent=u_node,
        critical=True,
    )
    fields_claim = (
        f"Typical required or preferred academic fields for these roles at {uni_name} include: {fields_two}. "
        "At least two fields are identified on the cited page(s)."
    )
    await evaluator.verify(
        claim=fields_claim,
        node=fields_leaf,
        sources=uni.requirements.urls,
        additional_instruction="Confirm the page lists relevant fields/majors (e.g., Counseling, Higher Education, Human Resources, Psychology, etc.). Allow close synonyms."
    )

    # 8) Certification Requirements
    cert_leaf = evaluator.add_leaf(
        id=f"u{uni_idx}_certification_requirements",
        desc="Verify whether professional certification (such as NCDA CCC or GCDF) is required or preferred, supported by sources",
        parent=u_node,
        critical=True,
    )
    cert_claim = (
        f"For these roles at {uni_name}, professional certification status is described as: '{uni.requirements.certification or ''}'. "
        "This is supported by the cited page(s)."
    )
    await evaluator.verify(
        claim=cert_claim,
        node=cert_leaf,
        sources=uni.requirements.urls,
        additional_instruction="Look for mentions like NCDA CCC, GCDF, CCSP, or generic 'career counseling certification' as required or preferred. If explicitly not required, that should also be stated."
    )

    # 9) Years of Experience (entry-level)
    exp_leaf = evaluator.add_leaf(
        id=f"u{uni_idx}_experience_requirements",
        desc="Verify that typical years of relevant experience required for entry-level positions are identified and supported by sources",
        parent=u_node,
        critical=True,
    )
    exp_claim = (
        f"The typical years of relevant experience required for entry-level counselor/coordinator roles at {uni_name} is '{uni.requirements.years_experience or ''}'."
    )
    await evaluator.verify(
        claim=exp_claim,
        node=exp_leaf,
        sources=uni.requirements.urls,
        additional_instruction="Confirm the page lists a years-of-experience requirement (e.g., 0–2 years, 1 year). Allow reasonable numeric/wording variations."
    )

    # 10) Reference URLs provided for each piece (custom check)
    refs_ok = _all_fields_have_urls(uni)
    refs_leaf = evaluator.add_custom_node(
        result=refs_ok,
        id=f"u{uni_idx}_reference_urls",
        desc="Verify that reference URLs from the university's official website are provided for each required piece of information",
        parent=u_node,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Selection criteria verification                                             #
# --------------------------------------------------------------------------- #
def _distinct_nonempty_lower(values: List[Optional[str]]) -> int:
    return len({(v or "").strip().lower() for v in values if _non_empty(v)})


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
    Evaluate an answer for the NCAA Division I career services task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel as rubric's Task_Completion
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

    # Add an explicit Task_Completion node to mirror rubric hierarchy
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Evaluate overall task completion for three universities with diversity and complete info",
        parent=root,
        critical=False,
    )

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Ensure exactly 3 university slots (pad with empty if needed)
    universities: List[UniversityInfo] = list(extracted.universities[:3])
    while len(universities) < 3:
        universities.append(UniversityInfo())

    # Record some custom info/statistics
    evaluator.add_custom_info(
        info={
            "extracted_university_count": len(extracted.universities),
            "used_university_count": 3,
        },
        info_type="extraction_stats",
    )

    # Selection Criteria (critical)
    sel_node = evaluator.add_parallel(
        id="Selection_Criteria",
        desc="Verify that the three selected universities meet the geographic and conference diversity requirements",
        parent=task_node,
        critical=True,
    )

    # Three_Different_States
    states = [u.state for u in universities]
    three_states_distinct = (_distinct_nonempty_lower(states) == 3)
    evaluator.add_custom_node(
        result=three_states_distinct,
        id="Three_Different_States",
        desc="Confirm that each of the three universities is located in a different U.S. state",
        parent=sel_node,
        critical=True,
    )

    # Three_Different_Conferences
    conferences = [u.conference for u in universities]
    three_confs_distinct = (_distinct_nonempty_lower(conferences) == 3)
    evaluator.add_custom_node(
        result=three_confs_distinct,
        id="Three_Different_Conferences",
        desc="Confirm that each of the three universities belongs to a different NCAA Division I athletic conference",
        parent=sel_node,
        critical=True,
    )

    # Per-university verification blocks (non-critical under Task_Completion)
    for idx in range(3):
        await verify_university(evaluator, task_node, universities[idx], idx)

    # Return standardized summary
    return evaluator.get_summary()