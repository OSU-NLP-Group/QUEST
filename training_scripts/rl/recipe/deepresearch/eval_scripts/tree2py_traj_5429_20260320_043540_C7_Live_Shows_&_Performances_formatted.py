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
TASK_ID = "nyc_concert_venue_2026"
TASK_DESCRIPTION = """A major rock band is planning their 2026 North American concert tour and needs to identify a suitable venue in New York City for their spring 2026 performance. The tour management has specified the following requirements:

Location & Capacity: Must be in New York City (any borough), with concert seating between 15,000-25,000, and must be an indoor arena.

Technical & Production: Stage must support minimum 40 feet width and 30 feet depth, with professional concert audio system, rigging for concert lighting, and loading dock for equipment trucks.

Operational: Must be available March-May 2026, meet ADA accessibility standards, have backstage facilities, and provide parking for 500+ vehicles.

Identify ONE suitable venue meeting all requirements. Provide: venue name/location, concert capacity, indoor status confirmation, stage specifications, technical capabilities verification, availability confirmation, loading/backstage details, ADA compliance, parking information, and official website/booking contact.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueDetails(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None  # Borough/address or clear NYC location text
    capacity: Optional[str] = None  # Keep as string to allow ranges, "approx", etc.
    indoor_status: Optional[str] = None  # e.g., "indoor arena", "indoor", "Yes"
    stage_width_ft: Optional[str] = None  # text mentioning width in feet/meters
    stage_depth_ft: Optional[str] = None  # text mentioning depth in feet/meters
    stage_notes: Optional[str] = None

    audio_system: Optional[str] = None  # text description/evidence
    lighting_rigging: Optional[str] = None  # text description/evidence
    availability_window: Optional[str] = None  # text asserting Mar–May 2026 availability or booking window
    loading_dock: Optional[str] = None  # text proof
    backstage_facilities: Optional[str] = None  # text proof
    ada_compliance: Optional[str] = None  # text proof (accessibility page)
    parking_capacity: Optional[str] = None  # e.g., "over 500", "1000+", "shared garages", etc.

    official_website_url: Optional[str] = None
    booking_contact_url: Optional[str] = None
    booking_contact_text: Optional[str] = None  # email/phone or "contact us" reference

    evidence_urls: List[str] = Field(default_factory=list)  # all other URLs cited for verification


class VenuesExtraction(BaseModel):
    venues: List[VenueDetails] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract all unique venues explicitly mentioned in the answer (even if the instruction asks for one). Return them in an array under the key 'venues'. For each venue, extract the following fields exactly as stated in the answer:

    - name: The venue name (e.g., "Madison Square Garden").
    - location: The NYC location information (borough and/or street address or clearly identifying location text).
    - capacity: The stated or implied concert seating capacity (keep as a string as written, e.g., "19,500", "approx 18k", "up to 20,000").
    - indoor_status: Whether the venue is an "indoor arena" (record the text from the answer, e.g., "indoor", "indoor arena").
    - stage_width_ft: The stated stage width (as free text; include units as provided, e.g., "60 ft", "18 m", "40-80ft").
    - stage_depth_ft: The stated stage depth (as free text; include units as provided).
    - stage_notes: Any extra details about stage/performance area relevant to dimensions or layout.
    - audio_system: Text describing in-house professional concert audio or support for touring audio systems.
    - lighting_rigging: Text describing rigging points/trusses/roof load suitable for concert lighting.
    - availability_window: Text stating availability or booking window for March–May 2026.
    - loading_dock: Text confirming loading dock access suitable for equipment trucks.
    - backstage_facilities: Text confirming presence of backstage areas (dressing rooms, green rooms, hospitality).
    - ada_compliance: Text confirming ADA/accessibility compliance or accessibility policy.
    - parking_capacity: Text describing on-site or nearby parking capacity (e.g., "500+ vehicles", "garage capacity 1200").
    - official_website_url: The official venue website URL.
    - booking_contact_url: A booking/rentals/contact page URL (if provided).
    - booking_contact_text: Any booking contact information that is NOT a URL (email, phone).
    - evidence_urls: A list of ALL other URLs cited in the answer for this venue (technical specs, production guide PDFs, accessibility page, parking information, calendars, maps, etc.).

    Rules:
    - Only extract URLs that are explicitly provided in the answer (including markdown links).
    - If a field is not present in the answer, set it to null (for a string field) or [] (for the evidence_urls list).
    - Do not invent or infer values not clearly present in the answer text.
    - If multiple venues are mentioned, include all of them in the 'venues' array (order preserved as in the answer).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if not s:
            continue
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def gather_sources(v: VenueDetails) -> List[str]:
    sources: List[str] = []
    if v.official_website_url:
        sources.append(v.official_website_url)
    if v.booking_contact_url:
        sources.append(v.booking_contact_url)
    if v.evidence_urls:
        sources.extend(v.evidence_urls)
    return _dedup_preserve(sources)


# --------------------------------------------------------------------------- #
# Verification sub-tree construction                                          #
# --------------------------------------------------------------------------- #
async def build_venue_requirements_checks(evaluator: Evaluator, parent_node, venue: VenueDetails) -> None:
    """
    Build the 'Venue_Requirements_And_Reporting_Check' parallel node and all its children leaves.
    All children are critical because the parent node is critical.
    """
    sources_all = gather_sources(venue)

    # Venue requirements parent (critical)
    reqs_node = evaluator.add_parallel(
        id="Venue_Requirements_And_Reporting_Check",
        desc="Check the identified venue against stated constraints and that required information is provided.",
        parent=parent_node,
        critical=True
    )

    # 1) Venue_Name_And_Location_Provided (existence check)
    evaluator.add_custom_node(
        result=bool(venue.name and venue.name.strip()) and bool(venue.location and venue.location.strip()),
        id="Venue_Name_And_Location_Provided",
        desc="Solution provides the venue name and its NYC location (e.g., borough/address or equivalent identifying location information).",
        parent=reqs_node,
        critical=True
    )

    # 2) Geographic_Location_NYC
    geo_node = evaluator.add_leaf(
        id="Geographic_Location_NYC",
        desc="Venue is located within New York City (any borough).",
        parent=reqs_node,
        critical=True
    )
    geo_claim = f"The venue '{venue.name or 'the venue'}' is located within New York City (NYC). The stated location is: {venue.location or 'N/A'}."
    await evaluator.verify(
        claim=geo_claim,
        node=geo_node,
        sources=sources_all,
        additional_instruction="Confirm the venue's official address is within the five NYC boroughs: Manhattan, Brooklyn, Queens, The Bronx, or Staten Island. If the official page shows a city outside NYC (e.g., Elmont, Uniondale, Newark), this claim is NOT supported."
    )

    # 3) Capacity_Within_15000_to_25000
    cap_node = evaluator.add_leaf(
        id="Capacity_Within_15000_to_25000",
        desc="Solution provides a concert seating capacity value and it is between 15,000 and 25,000 inclusive.",
        parent=reqs_node,
        critical=True
    )
    cap_specific = f" Specifically, the reported capacity is {venue.capacity}." if venue.capacity else ""
    cap_claim = f"The concert seating capacity for {venue.name or 'the venue'} is within 15,000–25,000 (inclusive).{cap_specific}"
    await evaluator.verify(
        claim=cap_claim,
        node=cap_node,
        sources=sources_all,
        additional_instruction="Use official specs or reputable venue sources. If only sport-mode capacities are given, they still must fall within 15,000–25,000 to pass."
    )

    # 4) Indoor_Arena_Requirement
    indoor_node = evaluator.add_leaf(
        id="Indoor_Arena_Requirement",
        desc="Solution confirms the venue is an indoor arena (not an outdoor amphitheater or stadium).",
        parent=reqs_node,
        critical=True
    )
    indoor_claim = f"{venue.name or 'The venue'} is an indoor arena (enclosed/roofed venue), not an outdoor stadium or amphitheater."
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_node,
        sources=sources_all,
        additional_instruction="Look for classification terms such as 'indoor', 'arena', 'enclosed', or building photos/specs that clearly indicate an indoor arena."
    )

    # 5) Stage_Dimensions_Min_40x30_ft (split into two critical leaves under a critical sub-node)
    stage_node = evaluator.add_parallel(
        id="Stage_Dimensions_Min_40x30_ft",
        desc="Stage/performance area supports at least 40 ft width AND 30 ft depth.",
        parent=reqs_node,
        critical=True
    )
    stage_w_leaf = evaluator.add_leaf(
        id="Stage_Width_Min_40ft",
        desc="Stage width is at least 40 ft.",
        parent=stage_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The stage/performance area at {venue.name or 'the venue'} supports at least 40 feet of width.",
        node=stage_w_leaf,
        sources=sources_all,
        additional_instruction="Check production guides, tech specs, PDFs, or official pages mentioning stage dimensions or maximum clear width. If only metric is given, convert: 12.2 m ≈ 40 ft."
    )
    stage_d_leaf = evaluator.add_leaf(
        id="Stage_Depth_Min_30ft",
        desc="Stage depth is at least 30 ft.",
        parent=stage_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The stage/performance area at {venue.name or 'the venue'} supports at least 30 feet of depth.",
        node=stage_d_leaf,
        sources=sources_all,
        additional_instruction="Check production guides, tech specs, PDFs, or official pages mentioning stage depth or upstage clearance. If only metric is given, convert: 9.14 m ≈ 30 ft."
    )

    # 6) Professional_Audio_System
    audio_node = evaluator.add_leaf(
        id="Professional_Audio_System",
        desc="Venue has or supports installation of a professional-grade concert audio system.",
        parent=reqs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{venue.name or 'The venue'} has an in-house professional concert audio system or provides infrastructure to support touring concert-level audio.",
        node=audio_node,
        sources=sources_all,
        additional_instruction="Look for mentions of 'house PA', 'line array', 'front-of-house (FOH)', 'audio tie-ins', or power distro for touring sound."
    )

    # 7) Concert_Lighting_Rigging
    rig_node = evaluator.add_leaf(
        id="Concert_Lighting_Rigging",
        desc="Venue has rigging infrastructure capable of supporting concert-level lighting systems.",
        parent=reqs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{venue.name or 'The venue'} provides rigging points/trusses/catwalks with sufficient load ratings for concert lighting systems.",
        node=rig_node,
        sources=sources_all,
        additional_instruction="Look for 'rigging points', 'roof load', 'trusses', 'catwalk', 'grid', and load ratings. Production guides commonly list this."
    )

    # 8) Spring_2026_Availability
    avail_node = evaluator.add_leaf(
        id="Spring_2026_Availability",
        desc="Solution confirms the venue is available for booking during March–May 2026.",
        parent=reqs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{venue.name or 'The venue'} is available for booking during March, April, and May of 2026.",
        node=avail_node,
        sources=sources_all,
        additional_instruction="Verify explicit availability or booking acceptance for Mar–May 2026 via calendars, booking pages, or official statements. If no explicit reference to 2026 spring availability is present, mark as NOT supported."
    )

    # 9) Loading_Dock_Access
    dock_node = evaluator.add_leaf(
        id="Loading_Dock_Access",
        desc="Venue has loading dock facilities adequate for concert tour equipment trucks.",
        parent=reqs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{venue.name or 'The venue'} has a loading dock suitable for equipment trucks used in concert tours.",
        node=dock_node,
        sources=sources_all,
        additional_instruction="Look for 'loading dock', 'truck bays', 'freight elevators', 'loading access', or 'dock height'."
    )

    # 10) Backstage_Facilities
    backstage_node = evaluator.add_leaf(
        id="Backstage_Facilities",
        desc="Venue provides backstage facilities (e.g., dressing rooms and hospitality facilities).",
        parent=reqs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{venue.name or 'The venue'} provides backstage facilities such as dressing rooms, green rooms, and hospitality spaces.",
        node=backstage_node,
        sources=sources_all,
        additional_instruction="Production/venue specs commonly list dressing rooms, green rooms, lounges, showers, catering, etc."
    )

    # 11) ADA_Accessibility_Compliance
    ada_node = evaluator.add_leaf(
        id="ADA_Accessibility_Compliance",
        desc="Venue meets ADA accessibility compliance standards.",
        parent=reqs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{venue.name or 'The venue'} meets ADA accessibility standards (has accessibility/ADA compliance in place).",
        node=ada_node,
        sources=sources_all,
        additional_instruction="Look for an official accessibility/ADA policy page or explicit statements about ADA compliance, accessible seating, elevators, etc."
    )

    # 12) Parking_Minimum_500_Vehicles
    parking_node = evaluator.add_leaf(
        id="Parking_Minimum_500_Vehicles",
        desc="Venue provides on-site or nearby parking accommodating at least 500 vehicles.",
        parent=reqs_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{venue.name or 'The venue'} provides on-site or nearby walkable parking for at least 500 vehicles.",
        node=parking_node,
        sources=sources_all,
        additional_instruction="Support must quantify capacity to reach ≥500 spaces (single large lot/garage or aggregated nearby walkable garages). If no clear evidence of ≥500 spaces is available, mark as NOT supported."
    )

    # 13) Official_Website_URL (existence check)
    evaluator.add_custom_node(
        result=bool(venue.official_website_url and venue.official_website_url.strip()),
        id="Official_Website_URL",
        desc="Solution includes the venue's official website URL.",
        parent=reqs_node,
        critical=True
    )

    # 14) Verifiable_Booking_Contact
    booking_node = evaluator.add_leaf(
        id="Verifiable_Booking_Contact",
        desc="Solution includes verifiable booking contact information for the venue.",
        parent=reqs_node,
        critical=True
    )
    # Prefer a dedicated booking/contact URL if present; otherwise fall back to the official website URL
    booking_sources = []
    if venue.booking_contact_url:
        booking_sources.append(venue.booking_contact_url)
    elif venue.official_website_url:
        booking_sources.append(venue.official_website_url)

    booking_sources = _dedup_preserve(booking_sources or sources_all)

    booking_claim = (
        f"The provided booking/contact information is valid for {venue.name or 'the venue'} and is an official page for booking, rentals, or contacting the venue."
    )
    await evaluator.verify(
        claim=booking_claim,
        node=booking_node,
        sources=booking_sources,
        additional_instruction="Look for explicit 'Contact', 'Booking', 'Rentals', 'Venue Hire', or sales contact details on the linked page. If only a generic homepage is given without any contact/booking info, mark as NOT supported."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the NYC concert venue 2026 task and return a structured result dictionary.
    """
    # Initialize evaluator with a sequential root: first ensure exactly one venue is identified, then check requirements
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
        default_model=model,
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Child 1: Exactly One Venue Identified (critical)
    exactly_one = evaluator.add_custom_node(
        result=len(extracted.venues) == 1,
        id="Exactly_One_Venue_Identified",
        desc="Solution identifies exactly one specific venue (not multiple candidates).",
        parent=root,
        critical=True,
    )

    # Child 2: Venue Requirements & Reporting Check (critical, parallel) - only meaningful if exactly_one passed
    # We still build it; downstream verify() will auto-skip if prerequisite failed
    venue_req_parent = evaluator.add_parallel(
        id="Suitable_Concert_Venue_Identification",
        desc="Evaluate whether exactly ONE suitable NYC venue is identified and whether it meets all mandatory tour requirements and provides the requested details.",
        parent=root,
        critical=True,
    )

    # Choose the single venue if present; otherwise use an empty placeholder to allow node creation
    venue: VenueDetails = extracted.venues[0] if extracted.venues else VenueDetails()

    # Build all requirement checks for the (single) venue
    await build_venue_requirements_checks(evaluator, venue_req_parent, venue)

    # Return final structured summary
    return evaluator.get_summary()