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
TASK_ID = "la_cultural_venue"
TASK_DESCRIPTION = """
Identify a major cultural institution in Los Angeles, California that meets all of the following requirements: offers free general admission to the public, requires advance timed-entry reservations for visitors, charges a fee for parking, features art, architecture, and/or cultural exhibitions, provides an online digital reservation system for booking visits, maintains regular public operating hours, includes permanent collection access as part of free admission, and is part of an institution with multiple locations. Provide the complete official name of this venue, the specific parking fee amount, information about how far in advance reservations can be made, and details about public accessibility.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueURLs(BaseModel):
    official_homepage: Optional[str] = None
    reservations_url: Optional[str] = None
    hours_url: Optional[str] = None
    admissions_url: Optional[str] = None
    parking_url: Optional[str] = None
    collections_url: Optional[str] = None
    locations_url: Optional[str] = None
    accessibility_url: Optional[str] = None
    other_urls: List[str] = Field(default_factory=list)


class VenueExtraction(BaseModel):
    venue_name: Optional[str] = None

    location_city: Optional[str] = None
    location_state: Optional[str] = None
    location_full: Optional[str] = None

    free_admission_statement: Optional[str] = None
    timed_entry_required_statement: Optional[str] = None
    online_reservation_system_statement: Optional[str] = None
    operating_hours_statement: Optional[str] = None
    parking_fee_statement: Optional[str] = None
    multiple_locations_statement: Optional[str] = None
    permanent_collection_free_statement: Optional[str] = None

    parking_cost: Optional[str] = None
    advance_booking_period: Optional[str] = None
    accessibility_info: Optional[str] = None
    features_statement: Optional[str] = None

    urls: VenueURLs = Field(default_factory=VenueURLs)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return """
    From the answer, extract the single cultural venue (institution/site) the answer proposes. Return a JSON object with the following fields. Only extract information explicitly present in the answer. If a field is not present, set it to null (or empty array where appropriate).

    Top-level fields:
    - venue_name: The complete official name of the venue.
    - location_city: The city of the venue (e.g., "Los Angeles").
    - location_state: The state (e.g., "California" or "CA").
    - location_full: The full location string if provided (address or "Los Angeles, California", etc.).
    - free_admission_statement: Any statement that general admission is free.
    - timed_entry_required_statement: Statement that advance timed-entry reservations are required.
    - online_reservation_system_statement: Statement that the venue has an online digital reservation system (e.g., "Reserve Tickets" portal).
    - operating_hours_statement: Statement showing they maintain/announce regular public operating hours.
    - parking_fee_statement: Statement that parking is charged (not free).
    - multiple_locations_statement: Statement indicating the institution has multiple locations/campuses.
    - permanent_collection_free_statement: Statement that permanent collection access is included in free general admission.
    - parking_cost: The specific dollar amount or text describing the parking fee (e.g., "$20 per car").
    - advance_booking_period: How far in advance reservations can be made (e.g., "up to 2 weeks in advance").
    - accessibility_info: Any details about public accessibility (e.g., "open to the public with timed tickets", "public visitors welcome", etc.).
    - features_statement: Statement that it is a cultural institution featuring art, architecture, and/or cultural exhibitions.

    URLs object (extract only URLs explicitly present in the answer; if a URL is missing a protocol, prepend http://):
    - urls.official_homepage: Official site home page of the institution or venue.
    - urls.reservations_url: The online digital reservation/booking page.
    - urls.hours_url: A page that lists hours/open days.
    - urls.admissions_url: A page that lists admissions and whether general admission is free.
    - urls.parking_url: A parking page that lists parking policies and fees.
    - urls.collections_url: Page about the permanent collection/exhibitions.
    - urls.locations_url: A page that indicates multiple locations/campuses exist.
    - urls.accessibility_url: A page with public accessibility details (general visitor access info).
    - urls.other_urls: Any other URLs explicitly included in the answer that are relevant to this venue.

    Important rules:
    - Extract only one venue (the primary one the answer is recommending).
    - Do not invent any URLs. Only return URLs explicitly present in the answer text.
    - If any field is missing in the answer, set it to null (or empty array for other_urls).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _unique(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if not _nonempty(x):
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def collect_sources(v: VenueExtraction, keys: List[str]) -> List[str]:
    """
    Build a combined, de-duplicated list of URLs from the extracted 'urls' object
    according to the provided key names.
    """
    mapping = {
        "homepage": v.urls.official_homepage,
        "reservations": v.urls.reservations_url,
        "hours": v.urls.hours_url,
        "admissions": v.urls.admissions_url,
        "parking": v.urls.parking_url,
        "collections": v.urls.collections_url,
        "locations": v.urls.locations_url,
        "accessibility": v.urls.accessibility_url,
    }
    urls: List[str] = []
    for k in keys:
        if k in mapping and _nonempty(mapping[k]):
            urls.append(mapping[k])  # type: ignore
    urls.extend(v.urls.other_urls or [])
    return _unique(urls)


MISSING_VALUE_RULE = (
    "If the claim contains a blank or missing value (e.g., empty quotes or 'None'), "
    "or if the provided webpages do not explicitly support the statement, judge as not supported."
)


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_venue(evaluator: Evaluator, parent_node, extracted: VenueExtraction) -> None:
    """
    Build the rubric tree under the root based on the given JSON specification and
    run all leaf verifications using sources extracted from the answer.
    """
    # Parent node (critical, parallel)
    venue_node = evaluator.add_parallel(
        id="venue_identification",
        desc="Identify a cultural venue in Los Angeles, California that meets all stated access/reservation/parking/collection/multi-location requirements and provide the requested output fields.",
        parent=parent_node,
        critical=True
    )

    # Prepare all leaves according to the rubric
    # 1) Los Angeles location
    node_loc = evaluator.add_leaf(
        id="los_angeles_location",
        desc="The venue is located in Los Angeles, California.",
        parent=venue_node,
        critical=True
    )
    claim_loc = "The venue is located in the city of Los Angeles, California (not just Los Angeles County)."
    sources_loc = collect_sources(extracted, ["homepage", "hours", "admissions", "reservations", "accessibility"])
    add_ins_loc = (
        "Confirm the city is 'Los Angeles' in the state 'California' on the official site (address/visit/contact pages). "
        "Do not accept only 'Los Angeles County' if the city is clearly different. " + MISSING_VALUE_RULE
    )

    # 2) Cultural institution with art/architecture/cultural exhibitions
    node_cult = evaluator.add_leaf(
        id="cultural_institution",
        desc="The venue is a cultural institution featuring art, architecture, and/or cultural exhibitions.",
        parent=venue_node,
        critical=True
    )
    claim_cult = "This venue is a cultural institution that features art, architecture, and/or cultural exhibitions."
    sources_cult = collect_sources(extracted, ["homepage", "collections"])
    add_ins_cult = (
        "Look for terms like 'museum', 'art', 'architecture', 'exhibition(s)', 'collection(s)'. " + MISSING_VALUE_RULE
    )

    # 3) Free general admission
    node_free = evaluator.add_leaf(
        id="free_admission",
        desc="The venue offers free general admission to the public.",
        parent=venue_node,
        critical=True
    )
    claim_free = "General admission to the venue is free to the public."
    sources_free = collect_sources(extracted, ["admissions", "homepage"])
    add_ins_free = (
        "The page should explicitly indicate 'free' general admission or 'no charge' for general entry. "
        "Do not count special exhibits with separate fees. " + MISSING_VALUE_RULE
    )

    # 4) Timed-entry required
    node_timed = evaluator.add_leaf(
        id="timed_entry_required",
        desc="The venue requires advance timed-entry reservations for visitors.",
        parent=venue_node,
        critical=True
    )
    claim_timed = "Advance timed-entry reservations are required for visitors to enter the venue."
    sources_timed = collect_sources(extracted, ["reservations", "admissions", "homepage"])
    add_ins_timed = (
        "Confirm language like 'timed-entry', 'advance reservations required', or 'book a time slot'. " + MISSING_VALUE_RULE
    )

    # 5) Online reservation system
    node_online = evaluator.add_leaf(
        id="online_reservation_system",
        desc="The venue provides an online digital reservation system for booking visits.",
        parent=venue_node,
        critical=True
    )
    claim_online = "The venue provides an online digital reservation system (e.g., a booking portal/button) to book visits."
    sources_online = collect_sources(extracted, ["reservations", "admissions", "homepage"])
    add_ins_online = (
        "The page should show an online booking/checkout or a 'Reserve/Book Tickets' button linking to a digital system. "
        + MISSING_VALUE_RULE
    )

    # 6) Operating schedule
    node_hours = evaluator.add_leaf(
        id="operating_schedule",
        desc="The venue maintains regular public operating hours.",
        parent=venue_node,
        critical=True
    )
    claim_hours = "The venue maintains regular public operating hours with posted open days/times."
    sources_hours = collect_sources(extracted, ["hours", "homepage"])
    add_ins_hours = (
        "Look for a dedicated 'Hours' or 'Visit' page showing open days/times. " + MISSING_VALUE_RULE
    )

    # 7) Parking fee
    node_parking_fee = evaluator.add_leaf(
        id="parking_fee",
        desc="The venue charges a fee for parking.",
        parent=venue_node,
        critical=True
    )
    claim_parking_fee = "Parking for the venue is not free; a fee is charged."
    sources_parking_fee = collect_sources(extracted, ["parking", "visit", "homepage"])
    add_ins_parking_fee = (
        "Verify that parking requires payment (e.g., a specific dollar amount or 'paid parking'). "
        "If parking is explicitly free, the claim is not supported. " + MISSING_VALUE_RULE
    )

    # 8) Multiple locations
    node_multi = evaluator.add_leaf(
        id="multiple_locations",
        desc="The venue is part of an institution with multiple locations.",
        parent=venue_node,
        critical=True
    )
    claim_multi = "The institution operates multiple locations/campuses (e.g., more than one venue/site)."
    sources_multi = collect_sources(extracted, ["locations", "homepage"])
    add_ins_multi = (
        "Accept evidence such as 'two campuses/locations' listed on an official page. " + MISSING_VALUE_RULE
    )

    # 9) Permanent collection included in free admission
    node_perm = evaluator.add_leaf(
        id="permanent_collection_free",
        desc="Access to the permanent collection is included as part of free admission.",
        parent=venue_node,
        critical=True
    )
    claim_perm = "Access to the permanent collection is included as part of free general admission."
    sources_perm = collect_sources(extracted, ["admissions", "collections"])
    add_ins_perm = (
        "Confirm that the permanent collection (as opposed to special exhibitions) is included with free general admission. "
        + MISSING_VALUE_RULE
    )

    # 10) Venue name correctness
    node_name = evaluator.add_leaf(
        id="venue_name",
        desc="Provides the complete official name of the venue.",
        parent=venue_node,
        critical=True
    )
    venue_name_text = extracted.venue_name or ""
    claim_name = f"The complete official name of the venue is '{venue_name_text}'."
    sources_name = collect_sources(extracted, ["homepage"])
    add_ins_name = (
        "Verify that this exact official name appears on the venue's official website (home or about page). "
        + MISSING_VALUE_RULE
    )

    # 11) Parking cost amount
    node_parking_cost = evaluator.add_leaf(
        id="parking_cost",
        desc="Specifies the parking fee amount.",
        parent=venue_node,
        critical=True
    )
    parking_cost_text = extracted.parking_cost or ""
    claim_parking_cost = f"The parking fee amount for the venue is '{parking_cost_text}'."
    sources_parking_cost = collect_sources(extracted, ["parking"])
    add_ins_parking_cost = (
        "Check the official parking page for a specific dollar amount or clearly equivalent phrasing (e.g., '$20 per car'). "
        + MISSING_VALUE_RULE
    )

    # 12) Advance booking period
    node_adv = evaluator.add_leaf(
        id="advance_booking_period",
        desc="Provides information about how far in advance reservations can be made.",
        parent=venue_node,
        critical=True
    )
    advance_text = extracted.advance_booking_period or ""
    claim_adv = f"Visitors can make reservations {advance_text} in advance."
    sources_adv = collect_sources(extracted, ["reservations", "admissions"])
    add_ins_adv = (
        "Confirm a clear statement about the advance booking window (e.g., 'up to 2 weeks in advance'). "
        + MISSING_VALUE_RULE
    )

    # 13) Accessibility information (public access details)
    node_access = evaluator.add_leaf(
        id="accessibility_info",
        desc="Provides details about public accessibility (i.e., how visitors can access the venue).",
        parent=venue_node,
        critical=True
    )
    access_text = extracted.accessibility_info or ""
    claim_access = (
        f"Public accessibility details are as stated: '{access_text}'. "
        "These details appear on the official site (visitor info/admissions/reservations/hours)."
    )
    sources_access = collect_sources(extracted, ["accessibility", "admissions", "reservations", "hours", "homepage"])
    add_ins_access = (
        "Verify that the stated public access details (e.g., 'open to the public with timed tickets') are present on the official pages. "
        + MISSING_VALUE_RULE
    )

    # Perform verifications (in parallel using batch_verify)
    claims_and_sources: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = [
        (claim_loc, sources_loc if sources_loc else None, node_loc, add_ins_loc),
        (claim_cult, sources_cult if sources_cult else None, node_cult, add_ins_cult),
        (claim_free, sources_free if sources_free else None, node_free, add_ins_free),
        (claim_timed, sources_timed if sources_timed else None, node_timed, add_ins_timed),
        (claim_online, sources_online if sources_online else None, node_online, add_ins_online),
        (claim_hours, sources_hours if sources_hours else None, node_hours, add_ins_hours),
        (claim_parking_fee, sources_parking_fee if sources_parking_fee else None, node_parking_fee, add_ins_parking_fee),
        (claim_multi, sources_multi if sources_multi else None, node_multi, add_ins_multi),
        (claim_perm, sources_perm if sources_perm else None, node_perm, add_ins_perm),
        (claim_name, sources_name if sources_name else None, node_name, add_ins_name),
        (claim_parking_cost, sources_parking_cost if sources_parking_cost else None, node_parking_cost, add_ins_parking_cost),
        (claim_adv, sources_adv if sources_adv else None, node_adv, add_ins_adv),
        (claim_access, sources_access if sources_access else None, node_access, add_ins_access),
    ]

    await evaluator.batch_verify(claims_and_sources)


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
) -> Dict:
    """
    Evaluate a single answer for the LA cultural venue task and return a structured result dictionary.
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

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Optionally record any custom info about sources collected
    sources_summary = {
        "official_homepage": extracted.urls.official_homepage,
        "reservations_url": extracted.urls.reservations_url,
        "hours_url": extracted.urls.hours_url,
        "admissions_url": extracted.urls.admissions_url,
        "parking_url": extracted.urls.parking_url,
        "collections_url": extracted.urls.collections_url,
        "locations_url": extracted.urls.locations_url,
        "accessibility_url": extracted.urls.accessibility_url,
        "other_urls_count": len(extracted.urls.other_urls or []),
    }
    evaluator.add_custom_info(sources_summary, info_type="extracted_urls_summary")

    # Build verification tree and run checks
    await build_and_verify_venue(evaluator, root, extracted)

    # Return structured result
    return evaluator.get_summary()