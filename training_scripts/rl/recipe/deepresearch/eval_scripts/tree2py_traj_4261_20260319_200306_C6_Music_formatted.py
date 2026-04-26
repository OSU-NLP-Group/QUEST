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
TASK_ID = "newport_jazz_2025_db_entry"
TASK_DESCRIPTION = (
    "A music industry database administrator needs to compile verified information about Newport Jazz Festival 2025 "
    "for archival records. The database entry must include: "
    "(1) Venue Details - Identify the specific venue name and complete location (city and state) in Rhode Island where the festival was held, "
    "along with its daily attendance capacity; "
    "(2) Festival Schedule - Document the exact dates (including day of week, month, and year) when the three-day festival occurred in August 2025; "
    "(3) Headlining Performers - Identify the primary headlining artist for each of the three festival days (Friday, Saturday, and Sunday); "
    "(4) Booking Standards - Provide standard industry information about typical advance booking timelines for major music festivals and "
    "standard payment structures used in festival artist contracts; "
    "(5) Technical Requirements - Document standard technical production specifications including typical stage dimensions and load-in times "
    "for major festival performances. All information must be verifiable and include source references (URLs) for each major category of information."
)

# Ground-truth constraints expected by the rubric
EXPECTED = {
    "venue_name": "Fort Adams State Park",
    "venue_city": "Newport",
    "venue_state": "Rhode Island",
    "capacity_approx": "approximately 10,000",
    "dates": {
        "friday": "Friday, August 1, 2025",
        "saturday": "Saturday, August 2, 2025",
        "sunday": "Sunday, August 3, 2025",
    },
    "headliners": {
        "friday": "The Roots",
        "saturday": "Janelle Monáe",
        "sunday": "Jacob Collier",
    },
    "booking": {
        "timeline": "12–18 months in advance",
        "deposit": "50% deposit upon signing",
        "payment_models": "guarantee, percentage, or guarantee-plus-percentage (hybrid)",
        "final_payment": "final payment within 24 hours after performance",
    },
    "technical": {
        "stage_width": "40–60 feet",
        "stage_depth": "24–32 feet",
        "loadin_time": "3–4 hours",
        "tech_rider": "artists must provide technical riders",
        "crew_size": "5–40 core production crew members",
    },
}


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    daily_capacity: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ScheduleInfo(BaseModel):
    friday: Optional[str] = None
    saturday: Optional[str] = None
    sunday: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class DayHeadliner(BaseModel):
    name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HeadlinersInfo(BaseModel):
    friday: Optional[DayHeadliner] = None
    saturday: Optional[DayHeadliner] = None
    sunday: Optional[DayHeadliner] = None


class BookingStandardsInfo(BaseModel):
    timeline_claim: Optional[str] = None
    timeline_urls: List[str] = Field(default_factory=list)
    deposit_claim: Optional[str] = None
    deposit_urls: List[str] = Field(default_factory=list)
    payment_models_claim: Optional[str] = None
    payment_models_urls: List[str] = Field(default_factory=list)
    final_payment_claim: Optional[str] = None
    final_payment_urls: List[str] = Field(default_factory=list)


class TechnicalRequirementsInfo(BaseModel):
    stage_width_claim: Optional[str] = None
    stage_width_urls: List[str] = Field(default_factory=list)
    stage_depth_claim: Optional[str] = None
    stage_depth_urls: List[str] = Field(default_factory=list)
    loadin_time_claim: Optional[str] = None
    loadin_time_urls: List[str] = Field(default_factory=list)
    technical_rider_claim: Optional[str] = None
    technical_rider_urls: List[str] = Field(default_factory=list)
    core_crew_size_claim: Optional[str] = None
    core_crew_size_urls: List[str] = Field(default_factory=list)


class DatabaseEntryExtraction(BaseModel):
    venue: Optional[VenueInfo] = None
    schedule: Optional[ScheduleInfo] = None
    headliners: Optional[HeadlinersInfo] = None
    booking: Optional[BookingStandardsInfo] = None
    technical: Optional[TechnicalRequirementsInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_db_entry() -> str:
    return """
    Extract structured information about the Newport Jazz Festival 2025 from the answer. Only extract what is explicitly present in the answer text.

    Return a JSON object with this exact schema:

    {
      "venue": {
        "name": string or null,
        "city": string or null,
        "state": string or null,
        "daily_capacity": string or null,   // keep wording as stated, preserve approximations like "about 10,000"
        "urls": [array of URLs explicitly cited in the answer for venue info]
      },
      "schedule": {
        "friday": string or null,   // e.g., "Friday, August 1, 2025"
        "saturday": string or null, // e.g., "Saturday, August 2, 2025"
        "sunday": string or null,   // e.g., "Sunday, August 3, 2025"
        "urls": [array of URLs explicitly cited for the schedule/dates]
      },
      "headliners": {
        "friday": { "name": string or null, "urls": [array of URLs for Friday headliner] } or null,
        "saturday": { "name": string or null, "urls": [array of URLs for Saturday headliner] } or null,
        "sunday": { "name": string or null, "urls": [array of URLs for Sunday headliner] } or null
      },
      "booking": {
        "timeline_claim": string or null,            // e.g., "festivals book 12–18 months in advance"
        "timeline_urls": [array of URLs],
        "deposit_claim": string or null,             // e.g., "50% deposit upon signing"
        "deposit_urls": [array of URLs],
        "payment_models_claim": string or null,      // e.g., "guarantee, percentage, or hybrid"
        "payment_models_urls": [array of URLs],
        "final_payment_claim": string or null,       // e.g., "final payment due within 24 hours after performance"
        "final_payment_urls": [array of URLs]
      },
      "technical": {
        "stage_width_claim": string or null,         // e.g., "typical stage width 40–60 feet"
        "stage_width_urls": [array of URLs],
        "stage_depth_claim": string or null,         // e.g., "typical depth 24–32 feet"
        "stage_depth_urls": [array of URLs],
        "loadin_time_claim": string or null,         // e.g., "load-in 3–4 hours"
        "loadin_time_urls": [array of URLs],
        "technical_rider_claim": string or null,     // e.g., "artists must provide a technical rider"
        "technical_rider_urls": [array of URLs],
        "core_crew_size_claim": string or null,      // e.g., "touring acts travel with 5–40 crew"
        "core_crew_size_urls": [array of URLs]
      }
    }

    URL extraction rules:
    - Only include URLs explicitly present in the answer (plain or markdown). Do not invent URLs.
    - Ignore malformed URLs. If a URL misses http/https, prepend http://.

    If any field is not present in the answer, return null for that field (or empty list for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s or ""


def _combine_urls(*url_lists: Optional[List[str]]) -> List[str]:
    urls: List[str] = []
    for ul in url_lists:
        if not ul:
            continue
        for u in ul:
            if isinstance(u, str) and u.strip():
                urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_venue_nodes(evaluator: Evaluator, parent_node, ex: DatabaseEntryExtraction) -> None:
    node = evaluator.add_parallel(
        id="Venue_Details",
        desc="Verify venue name, location, and daily capacity match constraints and are supported by URLs.",
        parent=parent_node,
        critical=True,
    )

    venue = ex.venue or VenueInfo()

    # Source URLs presence (critical, gate others)
    venue_urls_exist = evaluator.add_custom_node(
        result=bool(venue.urls),
        id="Venue_Source_URLs",
        desc="Provides verifiable URL source reference(s) supporting venue name, location, and capacity.",
        parent=node,
        critical=True,
    )

    # Venue name check (answer states Fort Adams State Park)
    leaf_name = evaluator.add_leaf(
        id="Venue_Name_Is_Fort_Adams_State_Park",
        desc="States the venue name as Fort Adams State Park.",
        parent=node,
        critical=True,
    )
    claim_name = (
        f"The venue name stated in the answer ('{_safe(venue.name)}') refers to or matches 'Fort Adams State Park'."
    )
    await evaluator.verify(
        claim=claim_name,
        node=leaf_name,
        additional_instruction="Judge equivalence leniently: ignore case, allow minor punctuation or suffixes like city/state following the venue name.",
    )

    # Venue location check (city/state)
    leaf_loc = evaluator.add_leaf(
        id="Venue_Location_Is_Newport_Rhode_Island",
        desc="States the venue location as Newport, Rhode Island (city = Newport; state = Rhode Island).",
        parent=node,
        critical=True,
    )
    claim_loc = (
        f"The location stated in the answer indicates the city is 'Newport' and the state is 'Rhode Island' (allow 'RI'). "
        f"Extracted city='{_safe(venue.city)}'; extracted state='{_safe(venue.state)}'."
    )
    await evaluator.verify(
        claim=claim_loc,
        node=leaf_loc,
        additional_instruction="Mark Correct only if city matches 'Newport' and state matches 'Rhode Island' (or 'RI'), ignoring case and minor punctuation.",
    )

    # Capacity check (~10,000 approx)
    leaf_cap = evaluator.add_leaf(
        id="Venue_Daily_Capacity_Approx_10000",
        desc="States the venue daily attendance capacity as approximately 10,000 people (clearly indicated as an approximation).",
        parent=node,
        critical=True,
    )
    claim_cap = (
        f"The answer's capacity text ('{_safe(venue.daily_capacity)}') clearly communicates an approximate daily capacity "
        f"of about 10,000 people (accept variants like '~10,000', 'about 10k', 'approximately 10,000')."
    )
    await evaluator.verify(
        claim=claim_cap,
        node=leaf_cap,
        additional_instruction="Only mark Correct if the text conveys approximation and a value around ten thousand attendees per day.",
    )


async def build_schedule_nodes(evaluator: Evaluator, parent_node, ex: DatabaseEntryExtraction) -> None:
    node = evaluator.add_parallel(
        id="Festival_Schedule",
        desc="Verify the festival dates (including weekday) match constraints and are supported by URLs.",
        parent=parent_node,
        critical=True,
    )

    schedule = ex.schedule or ScheduleInfo()

    # Source URLs presence (critical, gate others)
    schedule_urls_exist = evaluator.add_custom_node(
        result=bool(schedule.urls),
        id="Schedule_Source_URLs",
        desc="Provides verifiable URL source reference(s) supporting the festival dates/schedule.",
        parent=node,
        critical=True,
    )

    # Dates check (exact three dates with weekdays)
    leaf_dates = evaluator.add_leaf(
        id="Festival_Dates_Are_Aug_1_2_3_2025_With_Weekdays",
        desc="States the three festival dates with weekday exactly as: Friday, August 1, 2025; Saturday, August 2, 2025; Sunday, August 3, 2025 (three consecutive days in August 2025).",
        parent=node,
        critical=True,
    )
    claim_dates = (
        "The festival dates stated in the answer exactly match the required set: "
        "Friday, August 1, 2025; Saturday, August 2, 2025; Sunday, August 3, 2025. "
        f"Extracted Friday='{_safe(schedule.friday)}'; Saturday='{_safe(schedule.saturday)}'; Sunday='{_safe(schedule.sunday)}'."
    )
    await evaluator.verify(
        claim=claim_dates,
        node=leaf_dates,
        additional_instruction="Mark Correct only if each extracted day matches the required date (allow minor punctuation/casing, but the weekday/month/day/year must match).",
    )


async def build_headliners_nodes(evaluator: Evaluator, parent_node, ex: DatabaseEntryExtraction) -> None:
    node = evaluator.add_parallel(
        id="Headlining_Performers",
        desc="Verify the primary headliner for each day matches constraints and is supported by URLs.",
        parent=parent_node,
        critical=True,
    )

    head = ex.headliners or HeadlinersInfo()
    fri = head.friday or DayHeadliner()
    sat = head.saturday or DayHeadliner()
    sun = head.sunday or DayHeadliner()

    # Source URLs presence (critical, gate others) — accept union across days
    head_urls_exist = evaluator.add_custom_node(
        result=bool(_combine_urls(fri.urls, sat.urls, sun.urls)),
        id="Headliners_Source_URLs",
        desc="Provides verifiable URL source reference(s) supporting the headliner identifications.",
        parent=node,
        critical=True,
    )

    # Friday headliner
    leaf_fri = evaluator.add_leaf(
        id="Friday_Headliner_Is_The_Roots",
        desc="Identifies The Roots as the Friday (Aug 1, 2025) primary headliner.",
        parent=node,
        critical=True,
    )
    claim_fri = (
        f"The Friday (Aug 1, 2025) primary headliner named in the answer ('{_safe(fri.name)}') refers to or matches 'The Roots'."
    )
    await evaluator.verify(
        claim=claim_fri,
        node=leaf_fri,
        additional_instruction="Allow minor naming variations and case differences.",
    )

    # Saturday headliner
    leaf_sat = evaluator.add_leaf(
        id="Saturday_Headliner_Is_Janelle_Monae",
        desc="Identifies Janelle Monáe as the Saturday (Aug 2, 2025) primary headliner.",
        parent=node,
        critical=True,
    )
    claim_sat = (
        f"The Saturday (Aug 2, 2025) primary headliner named in the answer ('{_safe(sat.name)}') refers to or matches 'Janelle Monáe'."
    )
    await evaluator.verify(
        claim=claim_sat,
        node=leaf_sat,
        additional_instruction="Treat 'Janelle Monae' (without accent) as equivalent; ignore case and minor punctuation/suffixes.",
    )

    # Sunday headliner
    leaf_sun = evaluator.add_leaf(
        id="Sunday_Headliner_Is_Jacob_Collier",
        desc="Identifies Jacob Collier as the Sunday (Aug 3, 2025) primary headliner.",
        parent=node,
        critical=True,
    )
    claim_sun = (
        f"The Sunday (Aug 3, 2025) primary headliner named in the answer ('{_safe(sun.name)}') refers to or matches 'Jacob Collier'."
    )
    await evaluator.verify(
        claim=claim_sun,
        node=leaf_sun,
        additional_instruction="Allow minor naming variations and case differences.",
    )


async def build_booking_nodes(evaluator: Evaluator, parent_node, ex: DatabaseEntryExtraction) -> None:
    node = evaluator.add_parallel(
        id="Booking_Standards",
        desc="Verify standard booking timeline and contract payment structure constraints are stated and supported by URLs.",
        parent=parent_node,
        critical=True,
    )

    booking = ex.booking or BookingStandardsInfo()

    # Source URLs presence (critical, gate others) — union across booking sub-items
    booking_urls = _combine_urls(
        booking.timeline_urls,
        booking.deposit_urls,
        booking.payment_models_urls,
        booking.final_payment_urls,
    )
    booking_urls_exist = evaluator.add_custom_node(
        result=bool(booking_urls),
        id="Booking_Standards_Source_URLs",
        desc="Provides verifiable URL source reference(s) supporting the booking timeline and payment-structure standards.",
        parent=node,
        critical=True,
    )

    # Timeline 12–18 months
    leaf_timeline = evaluator.add_leaf(
        id="Advance_Booking_Timeline_12_18_Months",
        desc="States that major festivals typically book headliners 12–18 months in advance.",
        parent=node,
        critical=True,
    )
    claim_timeline = (
        f"The answer states that major festivals typically book headliners 12–18 months in advance. "
        f"Extracted text: '{_safe(booking.timeline_claim)}'."
    )
    await evaluator.verify(
        claim=claim_timeline,
        node=leaf_timeline,
        additional_instruction="Accept phrasing equivalents such as 'a year to a year and a half' or '12 to 18 months'.",
    )

    # Deposit 50% upon signing
    leaf_deposit = evaluator.add_leaf(
        id="Contract_Deposit_50_Percent",
        desc="States that standard festival contracts require a 50% deposit upon signing.",
        parent=node,
        critical=True,
    )
    claim_deposit = (
        f"The answer states that standard festival contracts require a 50% deposit upon signing. "
        f"Extracted text: '{_safe(booking.deposit_claim)}'."
    )
    await evaluator.verify(
        claim=claim_deposit,
        node=leaf_deposit,
        additional_instruction="Accept synonyms for 'deposit' such as 'advance' or 'down payment' as long as 50% upon signing is clear.",
    )

    # Payment models listed
    leaf_models = evaluator.add_leaf(
        id="Payment_Models_Listed",
        desc="States that payment structures typically include guarantee, percentage, or guarantee-plus-percentage (hybrid) models.",
        parent=node,
        critical=True,
    )
    claim_models = (
        "The answer states that payment structures typically include guarantee, percentage, or guarantee-plus-percentage (hybrid) models. "
        f"Extracted text: '{_safe(booking.payment_models_claim)}'."
    )
    await evaluator.verify(
        claim=claim_models,
        node=leaf_models,
        additional_instruction="Mark Correct only if all three are present: guarantee, percentage (e.g., back-end/gross/box-office percentage), and a hybrid model.",
    )

    # Final payment within 24 hours
    leaf_final = evaluator.add_leaf(
        id="Final_Payment_Within_24_Hours",
        desc="States that final payment is typically due within 24 hours after performance.",
        parent=node,
        critical=True,
    )
    claim_final = (
        "The answer states that final payment is typically due within 24 hours after performance. "
        f"Extracted text: '{_safe(booking.final_payment_claim)}'."
    )
    await evaluator.verify(
        claim=claim_final,
        node=leaf_final,
        additional_instruction="Allow equivalent phrasing such as 'by the next day' or 'within one day' after the performance.",
    )


async def build_technical_nodes(evaluator: Evaluator, parent_node, ex: DatabaseEntryExtraction) -> None:
    node = evaluator.add_parallel(
        id="Technical_Requirements",
        desc="Verify standard technical production specification constraints are stated and supported by URLs.",
        parent=parent_node,
        critical=True,
    )

    tech = ex.technical or TechnicalRequirementsInfo()

    # Source URLs presence (critical, gate others) — union across tech sub-items
    tech_urls = _combine_urls(
        tech.stage_width_urls,
        tech.stage_depth_urls,
        tech.loadin_time_urls,
        tech.technical_rider_urls,
        tech.core_crew_size_urls,
    )
    tech_urls_exist = evaluator.add_custom_node(
        result=bool(tech_urls),
        id="Technical_Requirements_Source_URLs",
        desc="Provides verifiable URL source reference(s) supporting the technical production specifications.",
        parent=node,
        critical=True,
    )

    # Stage width 40–60 ft
    leaf_width = evaluator.add_leaf(
        id="Stage_Width_40_60_Feet",
        desc="States typical major-festival stage width range as 40–60 feet.",
        parent=node,
        critical=True,
    )
    claim_width = (
        "The answer states that typical major-festival stage width is in the range 40–60 feet. "
        f"Extracted text: '{_safe(tech.stage_width_claim)}'."
    )
    await evaluator.verify(
        claim=claim_width,
        node=leaf_width,
        additional_instruction="Allow 'ft' abbreviation and hyphen/dash variations (e.g., 40-60, 40–60).",
    )

    # Stage depth 24–32 ft
    leaf_depth = evaluator.add_leaf(
        id="Stage_Depth_24_32_Feet",
        desc="States typical major-festival stage depth range as 24–32 feet.",
        parent=node,
        critical=True,
    )
    claim_depth = (
        "The answer states that typical major-festival stage depth is in the range 24–32 feet. "
        f"Extracted text: '{_safe(tech.stage_depth_claim)}'."
    )
    await evaluator.verify(
        claim=claim_depth,
        node=leaf_depth,
        additional_instruction="Allow 'ft' abbreviation and hyphen/dash variations (e.g., 24-32, 24–32).",
    )

    # Load-in time 3–4 hours
    leaf_loadin = evaluator.add_leaf(
        id="LoadIn_Time_3_4_Hours",
        desc="States typical load-in time as 3–4 hours for major productions.",
        parent=node,
        critical=True,
    )
    claim_loadin = (
        "The answer states that typical load-in time for major productions is 3–4 hours. "
        f"Extracted text: '{_safe(tech.loadin_time_claim)}'."
    )
    await evaluator.verify(
        claim=claim_loadin,
        node=leaf_loadin,
        additional_instruction="Accept equivalents like 'three to four hours' or 'approx. 3-4 hrs'.",
    )

    # Technical rider requirement
    leaf_rider = evaluator.add_leaf(
        id="Technical_Rider_Requirement",
        desc="States that artists must provide technical riders specifying equipment requirements.",
        parent=node,
        critical=True,
    )
    claim_rider = (
        "The answer states that artists must provide technical riders specifying their technical/equipment requirements. "
        f"Extracted text: '{_safe(tech.technical_rider_claim)}'."
    )
    await evaluator.verify(
        claim=claim_rider,
        node=leaf_rider,
        additional_instruction="Accept equivalent phrasing like 'production rider' or 'stage plot + input list' as part of a technical rider.",
    )

    # Core crew size 5–40
    leaf_crew = evaluator.add_leaf(
        id="Core_Crew_Size_5_40",
        desc="States that major touring acts typically travel with 5–40 core production crew members.",
        parent=node,
        critical=True,
    )
    claim_crew = (
        "The answer states that major touring acts typically travel with about 5–40 core production crew members. "
        f"Extracted text: '{_safe(tech.core_crew_size_claim)}'."
    )
    await evaluator.verify(
        claim=claim_crew,
        node=leaf_crew,
        additional_instruction="Accept reasonable wording variants like 'five to forty' and the term 'crew' or 'production crew'.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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

    # Extract structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_db_entry(),
        template_class=DatabaseEntryExtraction,
        extraction_name="db_entry_extraction",
    )

    # Add expected constraints as "ground truth" info
    evaluator.add_ground_truth(
        {
            "expected": EXPECTED,
            "notes": "All category nodes are critical; failure in any critical leaf should fail its parent category.",
        },
        gt_type="expected_constraints",
    )

    # Create a critical child to mirror rubric root criticality
    entry_node = evaluator.add_parallel(
        id="Newport_Jazz_Festival_2025_Database_Entry",
        desc="Verify the database entry satisfies all specified constraints for Newport Jazz Festival 2025 and includes verifiable URL sources for each major category.",
        parent=root,
        critical=True,
    )

    # Build subtrees per rubric categories
    await build_venue_nodes(evaluator, entry_node, extraction)
    await build_schedule_nodes(evaluator, entry_node, extraction)
    await build_headliners_nodes(evaluator, entry_node, extraction)
    await build_booking_nodes(evaluator, entry_node, extraction)
    await build_technical_nodes(evaluator, entry_node, extraction)

    # Return evaluation summary
    return evaluator.get_summary()