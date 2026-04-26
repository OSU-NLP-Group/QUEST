import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "se_career_centers"
TASK_DESCRIPTION = """Identify four universities located in the southeastern United States (Alabama, Arkansas, Florida, Georgia, Kentucky, Louisiana, Mississippi, North Carolina, South Carolina, Tennessee, Virginia, or West Virginia) whose career centers meet all of the following requirements:

1. The career center must explicitly offer scheduled career counseling appointments (bookable through an online system or by phone/email).
2. The career center must offer drop-in advising sessions, with specific times and locations clearly stated on their website.
3. The career center must explicitly state that alumni (in addition to current students) are eligible to use their services.
4. The career center must use Handshake or a university-branded online platform for posting job and internship opportunities.
5. The career center must have a physical location with a complete street address publicly listed.
6. The career center must operate Monday through Friday during standard business hours (at least 8:00 AM to 4:00 PM or equivalent).
7. The career center must provide both a phone number and an email address for contact.
8. The career center must offer career-related workshops, seminars, or professional development events (beyond individual appointments).

For each of the four universities, provide:
- The university name
- The career center's official name
- The complete physical street address
- Phone number and email address
- Operating hours (Monday-Friday)
- A brief description of the drop-in session availability (days, times, and location)
- The name of the online job/internship platform used (e.g., Handshake, or the institution's branded system)
- A reference URL to the career center's official website where this information can be verified
"""

SOUTHEAST_STATES = {
    "Alabama", "Arkansas", "Florida", "Georgia", "Kentucky", "Louisiana",
    "Mississippi", "North Carolina", "South Carolina", "Tennessee",
    "Virginia", "West Virginia"
}
STATE_ABBR = {
    "AL": "Alabama", "AR": "Arkansas", "FL": "Florida", "GA": "Georgia", "KY": "Kentucky",
    "LA": "Louisiana", "MS": "Mississippi", "NC": "North Carolina", "SC": "South Carolina",
    "TN": "Tennessee", "VA": "Virginia", "WV": "West Virginia"
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CareerCenterItem(BaseModel):
    university_name: Optional[str] = None
    career_center_name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Full name or abbreviation
    zip_code: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    operating_hours: Optional[str] = None  # e.g., "Mon-Fri 8:00 AM–5:00 PM"
    drop_in_description: Optional[str] = None  # e.g., "Mon/Wed 1–3 PM at Room 101"
    platform_name: Optional[str] = None  # e.g., "Handshake" or branded system name
    reference_url: Optional[str] = None  # official career center page


class CareerCentersExtraction(BaseModel):
    centers: List[CareerCenterItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_career_centers() -> str:
    return """
    From the provided answer, extract up to four university career centers that the answer claims meet ALL of the specified requirements.
    For each, return an object with these fields exactly:
      - university_name: The university's name.
      - career_center_name: The official name of the career center.
      - street_address: The complete street address of the career center (street and building info).
      - city: City name, if available.
      - state: The U.S. state (full name or two-letter abbreviation).
      - zip_code: ZIP/postal code, if available.
      - phone: A phone number for the career center (string as presented).
      - email: An email address for the career center (string as presented).
      - operating_hours: The stated operating hours (should include Monday–Friday, with times).
      - drop_in_description: A brief description of drop-in advising times and location.
      - platform_name: The job/internship platform used (e.g., Handshake or a branded platform name).
      - reference_url: A single official URL to the career center’s website where information can be verified.
    Notes:
      - Extract only what is explicitly present in the answer. Do not invent or infer missing data.
      - If the answer lists more than four, include the first four only.
      - If fewer than four are provided, include those present.
      - If a field is missing for an entry, set it to null.
      - The 'reference_url' should be a URL pointing to the official career center site where verification is possible.
    Return a JSON object with a single 'centers' array of up to four objects.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _source_list(item: CareerCenterItem) -> List[str]:
    """Build sources list for verification from the item."""
    urls: List[str] = []
    if item.reference_url:
        urls.append(item.reference_url)
    return urls


async def verify_career_center(
    evaluator: Evaluator,
    parent_node,
    item: CareerCenterItem,
    idx: int
) -> None:
    """
    Build verification sub-tree and run checks for one career center.
    """

    # Create career center node (parallel, non-critical to allow partial credit across centers)
    center_node = evaluator.add_parallel(
        id=f"career_center_{idx+1}",
        desc=(
            "First university career center meeting all requirements" if idx == 0 else
            "Second university career center meeting all requirements" if idx == 1 else
            "Third university career center meeting all requirements" if idx == 2 else
            "Fourth university career center meeting all requirements"
        ),
        parent=parent_node,
        critical=False
    )

    # Critical existence of the reference URL to gate other verifications
    ref_exists = bool(item.reference_url and item.reference_url.strip())
    evaluator.add_custom_node(
        result=ref_exists,
        id=f"career_center_{idx+1}_reference_url",
        desc="A reference URL to the career center's official website is provided",
        parent=center_node,
        critical=True
    )

    sources = _source_list(item)

    # Geographic location check (critical)
    geo_node = evaluator.add_leaf(
        id=f"career_center_{idx+1}_geographic_location",
        desc=("The university is located in a southeastern U.S. state (Alabama, Arkansas, Florida, Georgia, "
              "Kentucky, Louisiana, Mississippi, North Carolina, South Carolina, Tennessee, Virginia, or West Virginia)"),
        parent=center_node,
        critical=True
    )
    geo_claim = (
        "The career center webpage indicates the university is located in one of these southeastern U.S. states: "
        "Alabama, Arkansas, Florida, Georgia, Kentucky, Louisiana, Mississippi, North Carolina, South Carolina, "
        "Tennessee, Virginia, or West Virginia."
    )
    await evaluator.verify(
        claim=geo_claim,
        node=geo_node,
        sources=sources,
        additional_instruction=(
            "Use the street address or other location cues on the page to identify the state. "
            "Accept either full state names or two-letter abbreviations (e.g., FL=Florida). "
            "If the page indicates a state outside the provided list, mark as not supported."
        )
    )

    # Scheduled appointments (critical)
    sched_node = evaluator.add_leaf(
        id=f"career_center_{idx+1}_scheduled_appointments",
        desc="The career center explicitly offers scheduled career counseling appointments (not just drop-in)",
        parent=center_node,
        critical=True
    )
    sched_claim = (
        "The career center explicitly offers scheduled career counseling appointments that can be booked "
        "via an online system or by phone/email."
    )
    await evaluator.verify(
        claim=sched_claim,
        node=sched_node,
        sources=sources,
        additional_instruction=(
            "Look for phrasing such as 'schedule an appointment', 'book an appointment', 'one-on-one appointments', "
            "or booking portals. Generic 'contact us' without explicit appointment scheduling does not count. "
            "Drop-in only does not count."
        )
    )

    # Drop-in sessions (critical)
    dropin_node = evaluator.add_leaf(
        id=f"career_center_{idx+1}_drop_in_sessions",
        desc="The career center explicitly offers drop-in advising sessions with specific times/locations listed",
        parent=center_node,
        critical=True
    )
    dropin_claim = (
        "The career center webpage explicitly lists drop-in advising sessions, including specific times and a location."
    )
    await evaluator.verify(
        claim=dropin_claim,
        node=dropin_node,
        sources=sources,
        additional_instruction=(
            "Verify that 'drop-in', 'walk-in', or similar is explicitly mentioned and includes concrete times and a physical location. "
            "If times or location are missing or vague, mark as not supported."
        )
    )

    # Alumni eligibility (critical)
    alumni_node = evaluator.add_leaf(
        id=f"career_center_{idx+1}_alumni_eligibility",
        desc="The career center explicitly states that alumni are eligible to use their services",
        parent=center_node,
        critical=True
    )
    alumni_claim = (
        "The career center explicitly states that alumni (not just current students) are eligible to use career center services."
    )
    await evaluator.verify(
        claim=alumni_claim,
        node=alumni_node,
        sources=sources,
        additional_instruction=(
            "Look for explicit mentions of 'alumni' eligibility. If alumni are excluded or not mentioned, mark as not supported."
        )
    )

    # Online job platform (critical)
    platform_node = evaluator.add_leaf(
        id=f"career_center_{idx+1}_online_job_platform",
        desc="The career center uses Handshake or a university-branded online platform for job/internship postings",
        parent=center_node,
        critical=True
    )
    platform_claim = (
        f"The career center uses an online platform to post job and internship opportunities, specifically Handshake "
        f"or a university-branded system{(' (e.g., ' + item.platform_name + ')' if item.platform_name else '')}."
    )
    await evaluator.verify(
        claim=platform_claim,
        node=platform_node,
        sources=sources,
        additional_instruction=(
            "Confirm the page references 'Handshake' or a clearly institution-branded job/internship portal managed by the university. "
            "General social networks or external generic sites (e.g., LinkedIn alone) do not satisfy this requirement."
        )
    )

    # Physical location address (critical)
    address_node = evaluator.add_leaf(
        id=f"career_center_{idx+1}_physical_location",
        desc="A complete physical street address for the career center is provided",
        parent=center_node,
        critical=True
    )
    addr_claim = (
        "The career center webpage provides a complete physical street address for the career center, including street and city, "
        "and state/ZIP if available."
    )
    await evaluator.verify(
        claim=addr_claim,
        node=address_node,
        sources=sources,
        additional_instruction=(
            "Check for a full street address (street number/name and city). If only generic location info or a building name without a street address is shown, mark as not supported."
        )
    )

    # Operating hours (critical)
    hours_node = evaluator.add_leaf(
        id=f"career_center_{idx+1}_operating_hours",
        desc=("Operating hours showing Monday-Friday availability during standard business hours "
              "(at least 8:00 AM to 4:00 PM or similar) are provided"),
        parent=center_node,
        critical=True
    )
    hours_claim = (
        "The career center operates Monday through Friday during standard business hours, at minimum 8:00 AM to 4:00 PM (or equivalent)."
    )
    await evaluator.verify(
        claim=hours_claim,
        node=hours_node,
        sources=sources,
        additional_instruction=(
            "Verify that the page lists hours for Monday–Friday and that each day meets at least an 8-hour window from 8:00 AM to 4:00 PM or later (e.g., 8–5, 9–5 acceptable). "
            "If hours are missing, 'by appointment only', or weekends-only, mark as not supported."
        )
    )

    # Contact information group (critical parent with two critical leaves)
    contact_parent = evaluator.add_parallel(
        id=f"career_center_{idx+1}_contact_information",
        desc="Both a phone number and an email address are provided",
        parent=center_node,
        critical=True
    )

    phone_node = evaluator.add_leaf(
        id=f"career_center_{idx+1}_phone_number",
        desc="A phone number for the career center is provided",
        parent=contact_parent,
        critical=True
    )
    phone_claim = "The career center webpage provides a contact phone number."
    await evaluator.verify(
        claim=phone_claim,
        node=phone_node,
        sources=sources,
        additional_instruction=(
            "Look for a telephone number on the page. Format variants are acceptable. If no phone number is present, mark as not supported."
        )
    )

    email_node = evaluator.add_leaf(
        id=f"career_center_{idx+1}_email_address",
        desc="An email address for the career center is provided",
        parent=contact_parent,
        critical=True
    )
    email_claim = "The career center webpage provides a contact email address."
    await evaluator.verify(
        claim=email_claim,
        node=email_node,
        sources=sources,
        additional_instruction=(
            "Look for a specific email address (e.g., name@university.edu). A generic contact form alone does not count unless an email is explicitly shown."
        )
    )

    # Workshops / events (critical)
    events_node = evaluator.add_leaf(
        id=f"career_center_{idx+1}_workshops_events",
        desc="The career center offers workshops, seminars, or professional development events",
        parent=center_node,
        critical=True
    )
    events_claim = (
        "The career center offers career-related workshops, seminars, or professional development events beyond individual appointments."
    )
    await evaluator.verify(
        claim=events_claim,
        node=events_node,
        sources=sources,
        additional_instruction=(
            "Confirm the page references events such as 'workshops', 'seminars', 'professional development', 'career fairs', or an events calendar. "
            "If only appointments are mentioned without group events, mark as not supported."
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
    Evaluate an answer for the Southeastern US career centers task.
    """

    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates four centers in parallel
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

    # Record helpful info
    evaluator.add_custom_info(
        info={"states_full": sorted(list(SOUTHEAST_STATES)), "abbr_map": STATE_ABBR},
        info_type="southeast_states",
        info_name="southeast_states"
    )

    # Extract centers from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_career_centers(),
        template_class=CareerCentersExtraction,
        extraction_name="career_centers_extraction"
    )

    # Normalize to exactly 4 items (pad with empty if fewer)
    centers = extraction.centers[:4]
    while len(centers) < 4:
        centers.append(CareerCenterItem())

    # Build verification for each center
    for idx, item in enumerate(centers):
        await verify_career_center(evaluator, root, item, idx)

    return evaluator.get_summary()