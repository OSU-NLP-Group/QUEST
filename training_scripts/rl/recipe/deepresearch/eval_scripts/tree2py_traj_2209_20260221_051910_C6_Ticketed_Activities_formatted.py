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
TASK_ID = "chicago_concert_venues"
TASK_DESCRIPTION = """
Identify 4 live music concert venues in Chicago, Illinois that meet ALL of the following requirements:

1. Capacity Requirements:
   - Must have a seating/standing capacity between 2,000 and 8,000 people
   - Official capacity information must be publicly documented

2. Operational Status:
   - Must be currently operational and hosting events during 2024-2026
   - Must have hosted at least 2 ticketed concert events in 2024 or 2025

3. Ticket Offerings:
   - Must offer multiple ticket tiers (at least General Admission and one premium tier such as VIP, reserved seating, or premium packages)
   - Tickets must be available through major ticketing platforms (Ticketmaster, AXS, Live Nation, or similar)

4. Venue Characteristics:
   - Must be primarily used for live music concerts (not exclusively sports arenas or theaters, though multi-purpose venues that regularly host concerts are acceptable)
   - Must be an indoor venue

For each of the 4 venues, provide:
- Venue name
- Official capacity
- At least 2 examples of concerts held in 2024 or 2025
- Evidence of multiple ticket tier offerings
- Primary ticketing platform(s) used
- URL references supporting all claims
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConcertEvent(BaseModel):
    name: Optional[str] = None
    date: Optional[str] = None  # Keep as free-form string; may contain "2024-05-01", "May 2025", etc.
    url: Optional[str] = None
    platform: Optional[str] = None
    ticket_tiers: List[str] = Field(default_factory=list)


class VenueItem(BaseModel):
    name: Optional[str] = None

    # Capacity evidence
    capacity_text: Optional[str] = None
    capacity_urls: List[str] = Field(default_factory=list)

    # Location evidence
    location_text: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)

    # General venue profile / authoritative pages (official site, Wikipedia, etc.)
    venue_profile_urls: List[str] = Field(default_factory=list)

    # Venue characteristics evidence
    live_music_evidence_urls: List[str] = Field(default_factory=list)
    indoor_evidence_urls: List[str] = Field(default_factory=list)

    # Operational & events evidence
    events: List[ConcertEvent] = Field(default_factory=list)

    # Ticket tiers & platform evidence
    ticket_tier_urls: List[str] = Field(default_factory=list)
    ticketing_platforms: List[str] = Field(default_factory=list)  # e.g., ["Ticketmaster", "AXS"]
    platform_urls: List[str] = Field(default_factory=list)  # URLs on those platforms


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract all venues mentioned in the answer that are proposed as meeting the Chicago concert venue requirements.

    For each venue found in the answer, return an object with the following fields:

    - name: The venue name as stated.
    - capacity_text: The capacity number or textual description as stated (e.g., "4,500", "approx. 3,000", "2,500 standing").
    - capacity_urls: A list of URLs that the answer cites to support the capacity (official/authoritative pages preferred).
    - location_text: The location text as stated (should indicate Chicago, Illinois).
    - location_urls: A list of URLs that the answer cites to support the venue location (e.g., official site, Wikipedia, ticketing pages).
    - venue_profile_urls: URLs for general authoritative info about the venue (official website, Wikipedia, etc.).
    - live_music_evidence_urls: URLs showing the venue is primarily/regularly used for live music (e.g., concert calendar, venue about page).
    - indoor_evidence_urls: URLs explicitly or implicitly indicating the venue is an indoor facility (e.g., seating chart, "theatre", "ballroom").
    - events: An array of events the answer provides for this venue, where each event has:
        * name: Event/artist name
        * date: Event date string as stated (keep free-form)
        * url: URL to the event page or credible listing
        * platform: Ticketing platform name if mentioned (e.g., "Ticketmaster", "AXS")
        * ticket_tiers: A list of tier names present on the event page if mentioned (e.g., ["General Admission", "VIP"])
    - ticket_tier_urls: URLs (often event pages) where multiple ticket tiers (GA + premium) are visible for this venue.
    - ticketing_platforms: List of platform names used by the venue for ticketing (e.g., ["Ticketmaster", "AXS", "Live Nation", "SeatGeek"]).
    - platform_urls: URLs to those platform pages hosting tickets for concerts at the venue.

    IMPORTANT:
    - Extract only URLs that are explicitly present in the answer (plain URLs or markdown links).
    - If any field is missing or not provided in the answer, set null or an empty list as appropriate.
    - Return all venues mentioned. Do not invent venues or URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    clean = []
    seen = set()
    for u in urls or []:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            clean.append(s)
    return clean


def _select_recent_events(events: List[ConcertEvent]) -> List[ConcertEvent]:
    """Pick the first two events with dates indicating 2024 or 2025."""
    recent = []
    for ev in events or []:
        d = (ev.date or "").lower()
        if "2024" in d or "2025" in d:
            recent.append(ev)
        if len(recent) >= 2:
            break
    return recent


def _safe_venue_name(v: VenueItem) -> str:
    return v.name or "the venue"


def _collect_sources_for_location(v: VenueItem) -> List[str]:
    return _dedup_urls((v.location_urls or []) + (v.venue_profile_urls or []) + (v.platform_urls or []))


def _collect_sources_for_capacity(v: VenueItem) -> List[str]:
    return _dedup_urls((v.capacity_urls or []) + (v.venue_profile_urls or []))


def _collect_sources_for_operational(v: VenueItem) -> List[str]:
    event_urls = [e.url for e in v.events if e.url]
    return _dedup_urls(event_urls + (v.venue_profile_urls or []) + (v.platform_urls or []))


def _collect_sources_for_ticket_tiers(v: VenueItem) -> List[str]:
    event_urls = [e.url for e in v.events if e.url]
    return _dedup_urls((v.ticket_tier_urls or []) + event_urls + (v.platform_urls or []))


def _collect_sources_for_platforms(v: VenueItem) -> List[str]:
    event_urls = [e.url for e in v.events if e.url]
    return _dedup_urls((v.platform_urls or []) + event_urls)


def _collect_sources_for_live_music(v: VenueItem) -> List[str]:
    event_urls = [e.url for e in v.events if e.url]
    return _dedup_urls((v.live_music_evidence_urls or []) + (v.venue_profile_urls or []) + event_urls)


def _collect_sources_for_indoor(v: VenueItem) -> List[str]:
    return _dedup_urls((v.indoor_evidence_urls or []) + (v.venue_profile_urls or []))


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    venue_index: int
) -> None:
    """
    Build the verification subtree for a single venue, following the rubric.
    """
    vname = _safe_venue_name(venue)

    # Venue root node (non-critical; allows partial credit per venue)
    venue_node = evaluator.add_parallel(
        id=f"venue_{venue_index+1}",
        desc=f"{['First','Second','Third','Fourth'][venue_index]} venue meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # Geographic Location (critical leaf)
    geo_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index+1}_geographic_location",
        desc="Verify venue is located in Chicago, Illinois, United States",
        parent=venue_node,
        critical=True,
    )
    geo_sources = _collect_sources_for_location(venue)
    geo_claim = f"The venue '{vname}' is located in Chicago, Illinois, United States."
    await evaluator.verify(
        claim=geo_claim,
        node=geo_leaf,
        sources=geo_sources,
        additional_instruction="Use the provided URL(s) to confirm the venue's address or city. If no valid URL is provided, consider this claim NOT SUPPORTED."
    )

    # Capacity Requirements (critical parallel)
    cap_node = evaluator.add_parallel(
        id=f"venue_{venue_index+1}_capacity_requirements",
        desc="Verify venue capacity is between 2,000 and 8,000 with official documentation",
        parent=venue_node,
        critical=True
    )

    cap_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index+1}_capacity_in_range",
        desc="Official seating/standing capacity falls between 2,000 and 8,000 people (inclusive) and is publicly documented with verifiable URL reference",
        parent=cap_node,
        critical=True
    )
    cap_sources = _collect_sources_for_capacity(venue)
    cap_claim = f"The official capacity of '{vname}' is between 2,000 and 8,000 people."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=cap_sources,
        additional_instruction="The sources should explicitly state capacity. Prefer official or authoritative pages. If no valid URL is provided, mark NOT SUPPORTED."
    )

    # Operational Status (critical parallel)
    op_node = evaluator.add_parallel(
        id=f"venue_{venue_index+1}_operational_status",
        desc="Verify venue is currently operational and hosting concerts",
        parent=venue_node,
        critical=True
    )

    # Currently operational leaf
    current_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index+1}_currently_operational",
        desc="Venue is operational and hosting events during 2024-2026 period with URL reference confirming status",
        parent=op_node,
        critical=True
    )
    op_sources = _collect_sources_for_operational(venue)
    current_claim = f"The venue '{vname}' is operational and hosting events during 2024-2026."
    await evaluator.verify(
        claim=current_claim,
        node=current_leaf,
        sources=op_sources,
        additional_instruction="An events calendar or event pages dated 2024-2026 confirm operational status. If no valid URL is provided, mark NOT SUPPORTED."
    )

    # Recent Concert Activity (critical parallel)
    recent_node = evaluator.add_parallel(
        id=f"venue_{venue_index+1}_recent_concert_activity",
        desc="Venue has hosted at least 2 ticketed concert events in 2024 or 2025",
        parent=op_node,
        critical=True
    )

    recent_events = _select_recent_events(venue.events)

    # First concert evidence
    first_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index+1}_first_concert",
        desc="Evidence of first qualifying concert event in 2024 or 2025 with URL reference",
        parent=recent_node,
        critical=True
    )
    if len(recent_events) >= 1:
        ev1 = recent_events[0]
        ev1_sources = _dedup_urls([ev1.url] if ev1.url else [])
        ev1_claim = (
            f"The event '{ev1.name or 'Unnamed Event'}' at '{vname}' occurred in 2024 or 2025 and was a ticketed concert."
        )
        await evaluator.verify(
            claim=ev1_claim,
            node=first_leaf,
            sources=ev1_sources,
            additional_instruction="Confirm the page shows a concert at this venue and tickets/price/tier info. If no valid URL is provided, mark NOT SUPPORTED."
        )
    else:
        # No event; still run verify to set failure explicitly
        await evaluator.verify(
            claim=f"No valid first concert evidence was provided for '{vname}'.",
            node=first_leaf,
            sources=[],
            additional_instruction="No URL evidence provided; mark NOT SUPPORTED."
        )

    # Second concert evidence
    second_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index+1}_second_concert",
        desc="Evidence of second qualifying concert event in 2024 or 2025 with URL reference",
        parent=recent_node,
        critical=True
    )
    if len(recent_events) >= 2:
        ev2 = recent_events[1]
        ev2_sources = _dedup_urls([ev2.url] if ev2.url else [])
        ev2_claim = (
            f"The event '{ev2.name or 'Unnamed Event'}' at '{vname}' occurred in 2024 or 2025 and was a ticketed concert."
        )
        await evaluator.verify(
            claim=ev2_claim,
            node=second_leaf,
            sources=ev2_sources,
            additional_instruction="Confirm the page shows a concert at this venue and tickets/price/tier info. If no valid URL is provided, mark NOT SUPPORTED."
        )
    else:
        await evaluator.verify(
            claim=f"No valid second concert evidence was provided for '{vname}'.",
            node=second_leaf,
            sources=[],
            additional_instruction="No URL evidence provided; mark NOT SUPPORTED."
        )

    # Ticketing Requirements (critical parallel)
    ticket_node = evaluator.add_parallel(
        id=f"venue_{venue_index+1}_ticketing_requirements",
        desc="Verify venue offers multiple ticket tiers and uses major platforms",
        parent=venue_node,
        critical=True
    )

    # Multiple ticket tiers leaf
    tiers_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index+1}_multiple_ticket_tiers",
        desc="Venue offers at least General Admission and one premium tier (VIP, reserved seating, premium packages, etc.) with URL reference showing offerings",
        parent=ticket_node,
        critical=True
    )
    tier_sources = _collect_sources_for_ticket_tiers(venue)
    tiers_claim = (
        f"For concerts at '{vname}', there are multiple ticket tiers including General Admission and at least one premium tier "
        f"(e.g., VIP, reserved seating, or premium packages)."
    )
    await evaluator.verify(
        claim=tiers_claim,
        node=tiers_leaf,
        sources=tier_sources,
        additional_instruction="Look for tier labels on the ticketing/event pages. If no valid URL is provided, mark NOT SUPPORTED."
    )

    # Major ticketing platform leaf
    platforms_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index+1}_major_ticketing_platform",
        desc="Tickets available through major platforms (Ticketmaster, AXS, Live Nation, or similar) with URL reference confirming availability",
        parent=ticket_node,
        critical=True
    )
    plat_sources = _collect_sources_for_platforms(venue)
    platforms_list = ", ".join(venue.ticketing_platforms or [])
    platforms_claim = (
        f"Tickets for concerts at '{vname}' are available through major platforms such as {platforms_list}."
        if platforms_list else f"Tickets for concerts at '{vname}' are available through major platforms (e.g., Ticketmaster or AXS)."
    )
    await evaluator.verify(
        claim=platforms_claim,
        node=platforms_leaf,
        sources=plat_sources,
        additional_instruction="Confirm via domain/branding on the ticket pages (ticketmaster.com, axs.com, livenation.com, seatgeek.com). If no valid URL is provided, mark NOT SUPPORTED."
    )

    # Venue Characteristics (critical parallel)
    char_node = evaluator.add_parallel(
        id=f"venue_{venue_index+1}_venue_characteristics",
        desc="Verify venue type and indoor status",
        parent=venue_node,
        critical=True
    )

    # Primary use live music leaf
    primary_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index+1}_primary_use_live_music",
        desc="Venue is primarily used for live music concerts (multi-purpose venues that regularly host concerts are acceptable) with URL reference confirming primary use",
        parent=char_node,
        critical=True
    )
    primary_sources = _collect_sources_for_live_music(venue)
    primary_claim = (
        f"'{vname}' is primarily (or regularly) used for live music concerts."
    )
    await evaluator.verify(
        claim=primary_claim,
        node=primary_leaf,
        sources=primary_sources,
        additional_instruction="Use concert calendars, official 'about' pages, or consistent listings of concerts to confirm. If no valid URL is provided, mark NOT SUPPORTED."
    )

    # Indoor venue leaf
    indoor_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index+1}_indoor_venue",
        desc="Venue is an indoor facility (not outdoor amphitheater or open-air venue) with URL reference confirming indoor status",
        parent=char_node,
        critical=True
    )
    indoor_sources = _collect_sources_for_indoor(venue)
    indoor_claim = f"'{vname}' is an indoor venue."
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_leaf,
        sources=indoor_sources,
        additional_instruction="Look for terms like 'theatre', 'ballroom', seating charts, interior photos, or explicit 'indoor'. If no valid URL is provided, mark NOT SUPPORTED."
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
    Evaluate an answer for the Chicago concert venues task using the Mind2Web2 framework.
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

    # Extract structured venues info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Ensure we evaluate exactly 4 venues (pad with empty if fewer; truncate if more)
    venues: List[VenueItem] = (extracted.venues or [])[:4]
    while len(venues) < 4:
        venues.append(VenueItem())

    # Build verification tree for the 4 venues
    task_node = evaluator.add_parallel(
        id="Chicago_Concert_Venues_Task",
        desc="Identify 4 live music concert venues in Chicago, Illinois that meet all specified capacity, operational, ticketing, and venue characteristic requirements",
        parent=root,
        critical=False
    )

    for idx in range(4):
        await verify_single_venue(evaluator, task_node, venues[idx], idx)

    # Optional: record a custom info summary about extraction counts
    evaluator.add_custom_info(
        info={
            "total_venues_extracted": len(extracted.venues or []),
            "venues_evaluated": 4
        },
        info_type="extraction_stats",
        info_name="extraction_summary"
    )

    return evaluator.get_summary()