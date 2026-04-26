import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "spring_2026_career_fairs_oh_pa_nj_dc"
TASK_DESCRIPTION = (
    "Identify four universities located in Ohio, Pennsylvania, New Jersey, or Washington D.C. that have dedicated "
    "career centers and are hosting career fairs between February 1 and April 1, 2026. For each university, provide: "
    "1) Career fair date, start time, end time, venue, registration method/platform, and the event page URL; "
    "2) Career center physical location (building, room/floor, street address); "
    "3) Career center contact info (phone, email, office hours); "
    "4) Career services offered (resume review, career counseling, and appointment scheduling method); "
    "5) URL to the university's official career center webpage. All information must be verifiable via official sources."
)

ALLOWED_STATES_HINT = "Ohio (OH), Pennsylvania (PA), New Jersey (NJ), or Washington, D.C. (DC)"
DATE_RANGE_START = "2026-02-01"
DATE_RANGE_END = "2026-04-01"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityFair(BaseModel):
    fair_date: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    venue: Optional[str] = None
    registration_method: Optional[str] = None
    registration_url: Optional[str] = None


class UniversityCareerCenterLocation(BaseModel):
    building_name: Optional[str] = None
    room_number: Optional[str] = None  # room number or floor designation
    street_address: Optional[str] = None


class UniversityCareerCenterContact(BaseModel):
    phone_number: Optional[str] = None
    email_address: Optional[str] = None
    office_hours: Optional[str] = None


class UniversityCareerServices(BaseModel):
    resume_review_service: Optional[str] = None  # e.g., "Yes", "Available via drop-ins", "No", etc.
    career_counseling_service: Optional[str] = None
    appointment_scheduling_method: Optional[str] = None  # e.g., "Handshake", "Navigate", "Phone", "Drop-in hours"


class UniversityItem(BaseModel):
    university_name: Optional[str] = None
    career_center_url: Optional[str] = None
    fair: Optional[UniversityFair] = None
    location: Optional[UniversityCareerCenterLocation] = None
    contact: Optional[UniversityCareerCenterContact] = None
    services: Optional[UniversityCareerServices] = None
    # Additional official URLs that may contain location/contact/services details (e.g., Contact or About page)
    extra_sources: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return f"""
Extract structured information for up to four universities mentioned in the answer that the answer claims meet the following criteria:
- The university is located in Ohio, Pennsylvania, New Jersey, or Washington, D.C.
- The university has a dedicated career center
- The university is hosting a career fair between February 1 and April 1, 2026 (inclusive)

For each university, extract the following fields exactly as stated in the answer (do not infer or invent):
universities: [
  {{
    university_name: string | null,
    career_center_url: string | null,                 // Official career center page URL as given in the answer
    fair: {{
      fair_date: string | null,                       // e.g., "February 15, 2026" or "02/15/2026"
      start_time: string | null,                      // e.g., "10:00 AM"
      end_time: string | null,                        // e.g., "2:00 PM"
      venue: string | null,                           // building/facility name
      registration_method: string | null,             // e.g., "Handshake", "Walk-in", "RSVP required"
      registration_url: string | null                 // URL to the event page or registration portal
    }} | null,
    location: {{
      building_name: string | null,
      room_number: string | null,                     // room number or floor designation
      street_address: string | null                   // full street address including city and state
    }} | null,
    contact: {{
      phone_number: string | null,
      email_address: string | null,
      office_hours: string | null
    }} | null,
    services: {{
      resume_review_service: string | null,           // a short confirmation text if present in the answer; otherwise null
      career_counseling_service: string | null,       // a short confirmation text if present in the answer; otherwise null
      appointment_scheduling_method: string | null    // e.g., "Handshake", "Navigate", "Drop-in", "Phone"
    }} | null,
    extra_sources: string[]                           // additional official URLs from the answer relevant to the career center info
  }}
]

Additional instructions:
- Only include URLs that explicitly appear in the answer text. If a URL is missing a protocol, prepend http://.
- Preserve text exactly as presented in the answer for dates, times, venue, and contact details (do not reformat).
- If any field is missing in the answer, return null for that field (or empty list for extra_sources).
- If more than four universities are mentioned, extract only the first four as they appear in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities for verification                                           #
# --------------------------------------------------------------------------- #
def _is_negative_statement(text: Optional[str]) -> bool:
    if not text:
        return False
    s = text.strip().lower()
    # Simple heuristics for negation
    return any(
        kw in s for kw in [
            "no", "not", "doesn't", "does not", "do not", "isn't", "is not",
            "unavailable", "no resume", "not offered", "none"
        ]
    )


def _sources_list(*lists: Optional[List[str]]) -> List[str]:
    urls: List[str] = []
    for lst in lists:
        if lst:
            for u in lst:
                if u and isinstance(u, str):
                    urls.append(u)
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def _mark_failed_leaf(node) -> None:
    node.score = 0.0
    node.status = "failed"


# --------------------------------------------------------------------------- #
# Verification for a single university item                                   #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    uni_index: int,
) -> None:
    """
    Build the verification subtree for a single university and run verifications.
    uni_index is 1-based (1..4) for ID readability.
    """
    uni_name = uni.university_name or f"University #{uni_index}"

    # Top-level node for this university (non-critical to allow partial credit across universities)
    uni_node = evaluator.add_parallel(
        id=f"university_{uni_index}",
        desc=f"{['First','Second','Third','Fourth'][uni_index-1]} university meeting all specified criteria (location, career fair timing, career center existence)",
        parent=parent_node,
        critical=False
    )

    # 1) Career center reference URL (critical)
    cc_url_leaf = evaluator.add_leaf(
        id=f"university_{uni_index}_career_center_reference_url",
        desc="URL to the university's official career center webpage",
        parent=uni_node,
        critical=True
    )
    if uni.career_center_url:
        claim = f"This URL is an official career center (career services/career development) webpage for {uni_name}."
        await evaluator.verify(
            claim=claim,
            node=cc_url_leaf,
            sources=uni.career_center_url,
            additional_instruction=(
                "Verify that the page represents the university's career center (or career services/development). "
                "Prefer official university domains (e.g., *.edu) or university subdomains. "
                "The page should clearly indicate it's a career center/services site."
            )
        )
    else:
        _mark_failed_leaf(cc_url_leaf)

    # 2) Career fair details (critical)
    fair_node = evaluator.add_parallel(
        id=f"university_{uni_index}_career_fair_details",
        desc="Complete information about the university's Spring 2026 career fair",
        parent=uni_node,
        critical=True
    )

    # Fair temporal information (critical)
    temporal_node = evaluator.add_parallel(
        id=f"university_{uni_index}_fair_temporal_information",
        desc="Date and time details for the career fair",
        parent=fair_node,
        critical=True
    )

    # Prepare event sources (registration_url serves as the event page URL)
    event_url = uni.fair.registration_url if (uni and uni.fair) else None

    # Fair date
    fair_date_leaf = evaluator.add_leaf(
        id=f"university_{uni_index}_fair_date",
        desc="The specific date of the career fair (must be between February 1 and April 1, 2026)",
        parent=temporal_node,
        critical=True
    )
    if event_url and uni.fair and uni.fair.fair_date:
        claim = (
            f"The career fair date shown on this event page is {uni.fair.fair_date}, and it falls between "
            f"February 1 and April 1, 2026 (inclusive)."
        )
        await evaluator.verify(
            claim=claim,
            node=fair_date_leaf,
            sources=event_url,
            additional_instruction=(
                f"Confirm the event date matches '{uni.fair.fair_date}' "
                f"AND that the date is within the inclusive range {DATE_RANGE_START} to {DATE_RANGE_END}. "
                "Be tolerant of formatting variants (e.g., 'Feb 15, 2026' vs 'February 15, 2026')."
            )
        )
    else:
        _mark_failed_leaf(fair_date_leaf)

    # Fair start time
    start_leaf = evaluator.add_leaf(
        id=f"university_{uni_index}_fair_start_time",
        desc="The start time of the career fair",
        parent=temporal_node,
        critical=True
    )
    if event_url and uni.fair and uni.fair.start_time:
        claim = f"The career fair start time is {uni.fair.start_time}."
        await evaluator.verify(
            claim=claim,
            node=start_leaf,
            sources=event_url,
            additional_instruction="Allow minor formatting variants (e.g., '10:00AM' vs '10:00 AM')."
        )
    else:
        _mark_failed_leaf(start_leaf)

    # Fair end time
    end_leaf = evaluator.add_leaf(
        id=f"university_{uni_index}_fair_end_time",
        desc="The end time of the career fair",
        parent=temporal_node,
        critical=True
    )
    if event_url and uni.fair and uni.fair.end_time:
        claim = f"The career fair end time is {uni.fair.end_time}."
        await evaluator.verify(
            claim=claim,
            node=end_leaf,
            sources=event_url,
            additional_instruction="Allow minor formatting variants (e.g., '2 PM' vs '2:00 PM')."
        )
    else:
        _mark_failed_leaf(end_leaf)

    # Fair venue (critical)
    venue_leaf = evaluator.add_leaf(
        id=f"university_{uni_index}_fair_venue",
        desc="The name of the building, facility, or venue where the career fair will be held",
        parent=fair_node,
        critical=True
    )
    if event_url and uni.fair and uni.fair.venue:
        claim = f"The career fair venue is '{uni.fair.venue}'."
        await evaluator.verify(
            claim=claim,
            node=venue_leaf,
            sources=event_url,
            additional_instruction="Match venue name approximately; allow building/facility naming variations."
        )
    else:
        _mark_failed_leaf(venue_leaf)

    # Registration information (critical)
    reg_node = evaluator.add_parallel(
        id=f"university_{uni_index}_registration_information",
        desc="How students can register for or access the career fair",
        parent=fair_node,
        critical=True
    )

    # Registration method
    reg_method_leaf = evaluator.add_leaf(
        id=f"university_{uni_index}_registration_method",
        desc="The method or platform for registration (e.g., online platform name, walk-in, etc.)",
        parent=reg_node,
        critical=True
    )
    if event_url and uni.fair and uni.fair.registration_method:
        claim = f"Students register for this career fair via '{uni.fair.registration_method}'."
        await evaluator.verify(
            claim=claim,
            node=reg_method_leaf,
            sources=event_url,
            additional_instruction=(
                "Verify the page indicates the stated method/platform (e.g., Handshake, Navigate, RSVP form, Walk-in). "
                "Allow synonymous phrasing (e.g., 'RSVP on Handshake' vs 'Handshake')."
            )
        )
    else:
        _mark_failed_leaf(reg_method_leaf)

    # Registration URL (event page)
    reg_url_leaf = evaluator.add_leaf(
        id=f"university_{uni_index}_registration_url",
        desc="URL to the career fair event page or registration portal",
        parent=reg_node,
        critical=True
    )
    if event_url:
        claim = (
            f"This URL is the official event page or registration portal for a {uni_name} career fair "
            f"scheduled between February 1 and April 1, 2026."
        )
        await evaluator.verify(
            claim=claim,
            node=reg_url_leaf,
            sources=event_url,
            additional_instruction=(
                "Confirm the page is an event or registration portal for the university's career fair within the given date range. "
                "Accept university-affiliated platforms (e.g., Handshake) if the event clearly pertains to the university."
            )
        )
    else:
        _mark_failed_leaf(reg_url_leaf)

    # 3) Career center information (critical)
    cci_node = evaluator.add_parallel(
        id=f"university_{uni_index}_career_center_information",
        desc="Physical location and contact details for the career center",
        parent=uni_node,
        critical=True
    )

    # Physical location (critical)
    loc_node = evaluator.add_parallel(
        id=f"university_{uni_index}_physical_location",
        desc="The physical office location of the career center on campus",
        parent=cci_node,
        critical=True
    )

    cc_sources = _sources_list([uni.career_center_url] if uni.career_center_url else None, uni.extra_sources)

    # Building name
    building_leaf = evaluator.add_leaf(
        id=f"university_{uni_index}_building_name",
        desc="The name of the building where the career center is located",
        parent=loc_node,
        critical=True
    )
    if cc_sources and uni.location and uni.location.building_name:
        claim = f"The career center is located in the building '{uni.location.building_name}'."
        await evaluator.verify(
            claim=claim,
            node=building_leaf,
            sources=cc_sources,
            additional_instruction="Match building name approximately; allow minor naming variants (e.g., 'Hall' omitted)."
        )
    else:
        _mark_failed_leaf(building_leaf)

    # Room number or floor
    room_leaf = evaluator.add_leaf(
        id=f"university_{uni_index}_room_number",
        desc="The room number or floor designation of the career center office",
        parent=loc_node,
        critical=True
    )
    if cc_sources and uni.location and uni.location.room_number:
        claim = f"The career center office is in '{uni.location.room_number}' (room number or floor designation)."
        await evaluator.verify(
            claim=claim,
            node=room_leaf,
            sources=cc_sources,
            additional_instruction="Allow formats like 'Suite 200', 'Room 210A', '2nd Floor', 'Floor 2'."
        )
    else:
        _mark_failed_leaf(room_leaf)

    # Street address
    address_leaf = evaluator.add_leaf(
        id=f"university_{uni_index}_street_address",
        desc="The complete street address including street name, city, and state",
        parent=loc_node,
        critical=True
    )
    if cc_sources and uni.location and uni.location.street_address:
        claim = f"The career center street address is '{uni.location.street_address}'."
        await evaluator.verify(
            claim=claim,
            node=address_leaf,
            sources=cc_sources,
            additional_instruction=(
                f"Confirm the full address string appears or is clearly supported. "
                f"If the page reveals the state, ensure it is in {ALLOWED_STATES_HINT}. "
                "Allow minor punctuation/formatting variants."
            )
        )
    else:
        _mark_failed_leaf(address_leaf)

    # Contact information (critical)
    contact_node = evaluator.add_parallel(
        id=f"university_{uni_index}_contact_information",
        desc="Contact methods for reaching the career center",
        parent=cci_node,
        critical=True
    )

    # Phone
    phone_leaf = evaluator.add_leaf(
        id=f"university_{uni_index}_phone_number",
        desc="The publicly listed phone number for the career center",
        parent=contact_node,
        critical=True
    )
    if cc_sources and uni.contact and uni.contact.phone_number:
        claim = f"The career center phone number is '{uni.contact.phone_number}'."
        await evaluator.verify(
            claim=claim,
            node=phone_leaf,
            sources=cc_sources,
            additional_instruction="Allow formatting differences (e.g., '(614) 555-1234' vs '614-555-1234')."
        )
    else:
        _mark_failed_leaf(phone_leaf)

    # Email
    email_leaf = evaluator.add_leaf(
        id=f"university_{uni_index}_email_address",
        desc="The publicly listed email address for the career center",
        parent=contact_node,
        critical=True
    )
    if cc_sources and uni.contact and uni.contact.email_address:
        claim = f"The career center email address is '{uni.contact.email_address}'."
        await evaluator.verify(
            claim=claim,
            node=email_leaf,
            sources=cc_sources,
            additional_instruction="Match the email address exactly; allow case-insensitive match."
        )
    else:
        _mark_failed_leaf(email_leaf)

    # Office hours
    hours_leaf = evaluator.add_leaf(
        id=f"university_{uni_index}_office_hours",
        desc="The operating hours of the career center including at least weekday coverage",
        parent=contact_node,
        critical=True
    )
    if cc_sources and uni.contact and uni.contact.office_hours:
        claim = f"The career center office hours include: {uni.contact.office_hours}"
        await evaluator.verify(
            claim=claim,
            node=hours_leaf,
            sources=cc_sources,
            additional_instruction=(
                "Verify that the page provides office/operating hours information. "
                "Allow flexible phrasing or tabular presentation."
            )
        )
    else:
        _mark_failed_leaf(hours_leaf)

    # 4) Career services offered (critical)
    services_node = evaluator.add_parallel(
        id=f"university_{uni_index}_career_services_offered",
        desc="Key career services provided by the career center",
        parent=uni_node,
        critical=True
    )

    # Resume review service
    resume_leaf = evaluator.add_leaf(
        id=f"university_{uni_index}_resume_review_service",
        desc="The career center offers resume review or resume critique services",
        parent=services_node,
        critical=True
    )
    if cc_sources and uni.services and uni.services.resume_review_service:
        neg = _is_negative_statement(uni.services.resume_review_service)
        if neg:
            claim = "The career center does NOT offer resume review (resume critique) services."
        else:
            claim = "The career center offers resume review (resume critique) services."
        await evaluator.verify(
            claim=claim,
            node=resume_leaf,
            sources=cc_sources,
            additional_instruction="Look for mentions of resume review/critiques, feedback, or similar services."
        )
    else:
        _mark_failed_leaf(resume_leaf)

    # Career counseling service
    counseling_leaf = evaluator.add_leaf(
        id=f"university_{uni_index}_career_counseling_service",
        desc="The career center offers individual career counseling or career advising appointments",
        parent=services_node,
        critical=True
    )
    if cc_sources and uni.services and uni.services.career_counseling_service:
        neg = _is_negative_statement(uni.services.career_counseling_service)
        if neg:
            claim = "The career center does NOT offer individual career counseling or advising appointments."
        else:
            claim = "The career center offers individual career counseling or advising appointments."
        await evaluator.verify(
            claim=claim,
            node=counseling_leaf,
            sources=cc_sources,
            additional_instruction="Look for mentions of career counseling/advising appointments (1:1 meetings)."
        )
    else:
        _mark_failed_leaf(counseling_leaf)

    # Appointment scheduling method
    appt_leaf = evaluator.add_leaf(
        id=f"university_{uni_index}_appointment_scheduling_method",
        desc="The method by which students can schedule appointments (online system name, drop-in hours, or phone scheduling)",
        parent=services_node,
        critical=True
    )
    if cc_sources and uni.services and uni.services.appointment_scheduling_method:
        claim = f"Students schedule appointments via '{uni.services.appointment_scheduling_method}'."
        await evaluator.verify(
            claim=claim,
            node=appt_leaf,
            sources=cc_sources,
            additional_instruction=(
                "Verify the page names the stated scheduling method/platform (e.g., Handshake, Navigate, Starfish), "
                "or mentions drop-in hours/phone scheduling as described."
            )
        )
    else:
        _mark_failed_leaf(appt_leaf)


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Entry point to evaluate an agent answer for Spring 2026 career fairs in OH/PA/NJ/DC.
    """
    # Initialize evaluator with a parallel root (non-critical to allow partial credit)
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

    # Extract up to four universities and their details from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="extracted_university_data"
    )

    # Keep at most 4 universities; pad with empty entries if fewer than 4
    universities: List[UniversityItem] = (extracted.universities or [])[:4]
    while len(universities) < 4:
        universities.append(UniversityItem())

    # Add ground truth/context info (constraints summary)
    evaluator.add_ground_truth({
        "constraints": {
            "allowed_regions": ALLOWED_STATES_HINT,
            "date_range_inclusive": [DATE_RANGE_START, DATE_RANGE_END],
            "required_fields_per_university": [
                "career center URL, fair date/start/end time/venue/registration method & URL,"
                " career center location (building/room/address), contact (phone/email/hours), services (resume review, "
                "career counseling, appointment method)"
            ]
        }
    }, gt_type="task_constraints")

    # Build subtrees for four universities (parallel)
    for i in range(4):
        try:
            await verify_university(
                evaluator=evaluator,
                parent_node=root,
                uni=universities[i],
                uni_index=i + 1
            )
        except Exception as e:
            # If an unexpected error happens, record a failed custom node to avoid breaking the tree.
            evaluator.add_custom_node(
                result=False,
                id=f"university_{i+1}_unexpected_error",
                desc=f"Unexpected error while verifying university #{i+1}: {str(e)}",
                parent=root,
                critical=False
            )

    return evaluator.get_summary()