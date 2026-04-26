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
TASK_ID = "greater_boston_career_centers"
TASK_DESCRIPTION = """
I am researching career center services for prospective students in the Greater Boston area. Please find four universities located in the Greater Boston area (Massachusetts) and provide the following information about each university's career center:

1. Contact Information:
   - Phone number
   - Email address
   - Physical address (including building name and street address)

2. Service Availability:
   - Whether drop-in advising hours are available (Yes/No)
   - Whether the Handshake platform is used for appointment scheduling (Yes/No)
   - Whether virtual or remote appointment options are available (Yes/No)

3. Office Hours:
   - Regular office hours (specify days and times)

4. Reference:
   - Provide the official university career center website URL

For each university, please organize the information clearly and ensure all details come from official university career center sources.
"""

# Broader but practical list to help the judge recognize Greater Boston localities
GREATER_BOSTON_LOCALITIES = [
    "Boston", "Cambridge", "Somerville", "Medford", "Brookline", "Allston", "Brighton",
    "Chestnut Hill", "Newton", "Waltham", "Watertown", "Belmont", "Arlington", "Chelsea",
    "Everett", "Revere", "Malden", "Quincy", "Milton", "Needham", "Dedham", "Wellesley",
    "Lexington", "Winchester"
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    university: Optional[str] = None
    career_center_name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address_building: Optional[str] = None
    address_street: Optional[str] = None
    address_city_state_zip: Optional[str] = None
    drop_in_available: Optional[str] = None  # Expect "Yes" / "No"
    handshake_used: Optional[str] = None     # Expect "Yes" / "No"
    virtual_appointments: Optional[str] = None  # Expect "Yes" / "No"
    office_hours: Optional[str] = None
    reference_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to four universities and their career center details exactly as presented in the answer.

    For each university provide the following fields:
    - university: The university's name.
    - career_center_name: The official name of the career center (if stated).
    - phone: The career center phone number exactly as written.
    - email: The career center email address exactly as written.
    - address_building: The building name for the career center, if available (e.g., "Smith Hall"). If not present, return null.
    - address_street: The street address line(s), including number, street, and suite/room if given (e.g., "77 Massachusetts Ave, Room E17-294"). If not present, return null.
    - address_city_state_zip: City, state, and ZIP/postal code if given (e.g., "Cambridge, MA 02139"). If not present, return null.
    - drop_in_available: Return "Yes" or "No" depending on whether drop-in or walk-in advising hours are available (if the answer explicitly states this). If not stated, return null.
    - handshake_used: Return "Yes" or "No" depending on whether the Handshake platform is used for appointment scheduling (if the answer explicitly states this). If not stated, return null.
    - virtual_appointments: Return "Yes" or "No" depending on whether virtual or remote appointment options are available (if the answer explicitly states this). If not stated, return null.
    - office_hours: Regular office hours text (days and times) exactly as written in the answer, if provided; otherwise null.
    - reference_url: The official university career center website URL as given in the answer. Prefer .edu career/career-services pages. If missing, return null.
    - additional_urls: Any additional URLs cited in the answer specifically for this university (e.g., a departmental career page or a Handshake help page). Return an array, possibly empty.

    Rules:
    - Do not invent information; only extract what is explicitly present in the answer.
    - If the answer includes more than four universities, only include the first four.
    - Always include full URLs (with http/https).
    - Preserve text formatting for times and addresses (do not normalize).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def first_n_or_pad(items: List[UniversityItem], n: int) -> List[UniversityItem]:
    result = list(items[:n])
    while len(result) < n:
        result.append(UniversityItem())
    return result


def collect_sources(u: UniversityItem) -> List[str]:
    urls: List[str] = []
    if u.reference_url and isinstance(u.reference_url, str) and u.reference_url.strip():
        urls.append(u.reference_url.strip())
    # Add additional URLs if any
    for s in (u.additional_urls or []):
        if isinstance(s, str) and s.strip():
            urls.append(s.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for x in urls:
        if x not in seen:
            deduped.append(x)
            seen.add(x)
    return deduped


def yes_no_from_str(val: Optional[str]) -> Optional[bool]:
    if not val:
        return None
    v = val.strip().lower()
    if v in ("yes", "y", "true", "t"):
        return True
    if v in ("no", "n", "false", "f"):
        return False
    return None


def format_address(u: UniversityItem) -> Optional[str]:
    parts = []
    if u.address_building and u.address_building.strip():
        parts.append(u.address_building.strip())
    if u.address_street and u.address_street.strip():
        parts.append(u.address_street.strip())
    if u.address_city_state_zip and u.address_city_state_zip.strip():
        parts.append(u.address_city_state_zip.strip())
    if not parts:
        return None
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Verification for a single university                                        #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    u: UniversityItem,
    idx: int
) -> None:
    uni_idx = idx + 1
    uni_node = evaluator.add_parallel(
        id=f"University_{uni_idx}",
        desc=(
            "First university's career center information is complete and accurate" if uni_idx == 1 else
            "Second university's career center information is complete and accurate" if uni_idx == 2 else
            "Third university's career center information is complete and accurate" if uni_idx == 3 else
            "Fourth university's career center information is complete and accurate"
        ),
        parent=parent_node,
        critical=False
    )

    sources = collect_sources(u)

    # 1) Reference URL (critical)
    if not u.reference_url or not u.reference_url.strip():
        evaluator.add_custom_node(
            result=False,
            id=f"University_{uni_idx}_Reference_URL",
            desc="Reference URL from official university career center website (.edu domain) is provided",
            parent=uni_node,
            critical=True
        )
    else:
        ref_node = evaluator.add_leaf(
            id=f"University_{uni_idx}_Reference_URL",
            desc="Reference URL from official university career center website (.edu domain) is provided",
            parent=uni_node,
            critical=True
        )
        ref_claim = (
            f"This URL is an official university career center webpage on a .edu domain"
            f"{f' for {u.university}.' if u.university else '.'}"
        )
        await evaluator.verify(
            claim=ref_claim,
            node=ref_node,
            sources=u.reference_url,
            additional_instruction=(
                "Confirm the URL is on a .edu domain and the page is clearly a university career center page "
                "(e.g., mentions 'Career Center', 'Career Services', 'Career Development'). "
                "If the URL is not .edu or not an official university site, mark as not supported."
            )
        )

    # 2) Location verification (critical)
    loc_node = evaluator.add_leaf(
        id=f"University_{uni_idx}_Location_Verification",
        desc="University is located in Greater Boston area (Massachusetts)",
        parent=uni_node,
        critical=True
    )
    loc_claim = (
        "Based on the provided official page(s), the university/career center is located in Massachusetts "
        "and is within the Greater Boston area (e.g., one of: "
        + ", ".join(GREATER_BOSTON_LOCALITIES) + ")."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=sources if sources else None,
        additional_instruction=(
            "Check the address or location shown on the webpage. If the address is in a city commonly "
            "considered part of Greater Boston (such as the examples listed), then this claim is supported. "
            "If no location info is available on the page(s), mark as not supported."
        )
    )

    # 3) Phone (critical)
    if not (u.phone and u.phone.strip()):
        evaluator.add_custom_node(
            result=False,
            id=f"University_{uni_idx}_Contact_Phone",
            desc="Career center phone number is provided",
            parent=uni_node,
            critical=True
        )
    else:
        phone_node = evaluator.add_leaf(
            id=f"University_{uni_idx}_Contact_Phone",
            desc="Career center phone number is provided",
            parent=uni_node,
            critical=True
        )
        phone_claim = f"The career center phone number listed on the official page is '{u.phone.strip()}'."
        await evaluator.verify(
            claim=phone_claim,
            node=phone_node,
            sources=sources if sources else None,
            additional_instruction=(
                "Confirm the exact phone number appears on the provided official career center page(s). "
                "Allow minor formatting differences (e.g., dots vs dashes, parentheses)."
            )
        )

    # 4) Email (critical)
    if not (u.email and u.email.strip()):
        evaluator.add_custom_node(
            result=False,
            id=f"University_{uni_idx}_Contact_Email",
            desc="Career center email address is provided",
            parent=uni_node,
            critical=True
        )
    else:
        email_node = evaluator.add_leaf(
            id=f"University_{uni_idx}_Contact_Email",
            desc="Career center email address is provided",
            parent=uni_node,
            critical=True
        )
        email_claim = f"The career center email address on the official page is '{u.email.strip()}'."
        await evaluator.verify(
            claim=email_claim,
            node=email_node,
            sources=sources if sources else None,
            additional_instruction=(
                "Confirm the exact email address appears on the official career center page(s). "
                "Allow case-insensitive match."
            )
        )

    # 5) Physical Address (critical) – must include building and street
    address_ok = bool(u.address_building and u.address_building.strip()) and bool(u.address_street and u.address_street.strip())
    if not address_ok:
        evaluator.add_custom_node(
            result=False,
            id=f"University_{uni_idx}_Physical_Address",
            desc="Career center physical address including building name and street address is provided",
            parent=uni_node,
            critical=True
        )
    else:
        addr_node = evaluator.add_leaf(
            id=f"University_{uni_idx}_Physical_Address",
            desc="Career center physical address including building name and street address is provided",
            parent=uni_node,
            critical=True
        )
        addr_str = format_address(u) or f"{u.address_building}, {u.address_street}"
        addr_claim = (
            f"The official page lists the career center physical address including building and street as: '{addr_str}'."
        )
        await evaluator.verify(
            claim=addr_claim,
            node=addr_node,
            sources=sources if sources else None,
            additional_instruction=(
                "Verify that BOTH the building name and street address are present on the page and match the text. "
                "Allow minor punctuation or whitespace differences."
            )
        )

    # 6) Drop-in advising (critical)
    drop_in_bool = yes_no_from_str(u.drop_in_available)
    if drop_in_bool is None:
        evaluator.add_custom_node(
            result=False,
            id=f"University_{uni_idx}_Drop_In_Hours",
            desc="Confirmation of whether drop-in advising hours are available is provided",
            parent=uni_node,
            critical=True
        )
    else:
        dropin_node = evaluator.add_leaf(
            id=f"University_{uni_idx}_Drop_In_Hours",
            desc="Confirmation of whether drop-in advising hours are available is provided",
            parent=uni_node,
            critical=True
        )
        dropin_claim = (
            "The official page indicates that drop-in (walk-in/no appointment) advising hours are available."
            if drop_in_bool else
            "The official page indicates that drop-in (walk-in) advising hours are NOT available."
        )
        await evaluator.verify(
            claim=dropin_claim,
            node=dropin_node,
            sources=sources if sources else None,
            additional_instruction=(
                "Look for 'drop-in', 'walk-in', 'no appointment needed', 'express advising' or similar phrases. "
                "If explicitly stated as not available, or only by appointment, treat as NOT available."
            )
        )

    # 7) Handshake platform use (critical)
    handshake_bool = yes_no_from_str(u.handshake_used)
    if handshake_bool is None:
        evaluator.add_custom_node(
            result=False,
            id=f"University_{uni_idx}_Handshake_Platform",
            desc="Confirmation of whether Handshake platform is used for appointments is provided",
            parent=uni_node,
            critical=True
        )
    else:
        handshake_node = evaluator.add_leaf(
            id=f"University_{uni_idx}_Handshake_Platform",
            desc="Confirmation of whether Handshake platform is used for appointments is provided",
            parent=uni_node,
            critical=True
        )
        handshake_claim = (
            "The official page indicates that Handshake is used for appointment scheduling (or related bookings)."
            if handshake_bool else
            "The official page indicates that Handshake is NOT used for appointment scheduling."
        )
        await evaluator.verify(
            claim=handshake_claim,
            node=handshake_node,
            sources=sources if sources else None,
            additional_instruction=(
                "Check for explicit mentions of 'Handshake' in the context of booking/scheduling appointments. "
                "If the page only mentions Handshake as a job board without using it for appointments, do NOT count "
                "as used for scheduling."
            )
        )

    # 8) Virtual/remote appointments (critical)
    virtual_bool = yes_no_from_str(u.virtual_appointments)
    if virtual_bool is None:
        evaluator.add_custom_node(
            result=False,
            id=f"University_{uni_idx}_Virtual_Appointments",
            desc="Confirmation of whether virtual/remote appointment options are available is provided",
            parent=uni_node,
            critical=True
        )
    else:
        virt_node = evaluator.add_leaf(
            id=f"University_{uni_idx}_Virtual_Appointments",
            desc="Confirmation of whether virtual/remote appointment options are available is provided",
            parent=uni_node,
            critical=True
        )
        virt_claim = (
            "The official page indicates that virtual/remote appointments (e.g., via Zoom/online) are available."
            if virtual_bool else
            "The official page indicates that virtual/remote appointments are NOT available."
        )
        await evaluator.verify(
            claim=virt_claim,
            node=virt_node,
            sources=sources if sources else None,
            additional_instruction=(
                "Look for words like 'virtual', 'remote', 'Zoom', 'online appointments'. "
                "If only in-person appointments are offered, treat as NOT available."
            )
        )

    # 9) Office hours (critical)
    if not (u.office_hours and u.office_hours.strip()):
        evaluator.add_custom_node(
            result=False,
            id=f"University_{uni_idx}_Office_Hours",
            desc="Regular office hours (days and times) are provided",
            parent=uni_node,
            critical=True
        )
    else:
        hours_node = evaluator.add_leaf(
            id=f"University_{uni_idx}_Office_Hours",
            desc="Regular office hours (days and times) are provided",
            parent=uni_node,
            critical=True
        )
        hours_claim = (
            f"The official page provides regular office hours (days/times) as: '{u.office_hours.strip()}'."
        )
        await evaluator.verify(
            claim=hours_claim,
            node=hours_node,
            sources=sources if sources else None,
            additional_instruction=(
                "Confirm that the stated days and times (or equivalent schedule) are present on the official page. "
                "Allow minor formatting differences and reasonable paraphrasing that preserves the same hours."
            )
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
    Evaluate an answer for the Greater Boston university career center services task.
    """
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Normalize to exactly 4 universities
    universities = first_n_or_pad(extracted.universities, 4)

    # Add the Task_Compliance aggregate node to mirror rubric structure
    task_node = evaluator.add_parallel(
        id="Task_Compliance",
        desc="Complete information gathering for 4 Greater Boston area universities' career centers",
        parent=root,
        critical=False
    )

    # Verify each university block
    for idx in range(4):
        await verify_university(evaluator, task_node, universities[idx], idx)

    # Return structured summary
    return evaluator.get_summary()