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
TASK_ID = "spring_2026_concerts"
TASK_DESCRIPTION = """I'm planning to attend live concerts in spring 2026 and want to explore major tours happening across different U.S. cities. Find four ticketed concert events scheduled between April 1, 2026 and May 31, 2026, with each event meeting the following requirements:

1. Each event must be held at a venue with a seating capacity of at least 10,000 attendees
2. Each event must be in a different U.S. city
3. Each event must offer at least two different ticket types or tiers (such as General Admission, Reserved Seating, or VIP packages)
4. VIP or premium ticket packages must be available for each event, with at least two specific documented perks or benefits
5. Each event must have at least one officially announced special guest or supporting act

For each of the four events, provide:
- Event date
- Primary performing artist
- Venue name and complete street address
- U.S. city and state
- Documented venue capacity
- Description of available ticket types/tiers
- List of VIP package perks
- Official ticket purchasing platform (Ticketmaster, Live Nation, or official artist website)
- At least one supporting act or special guest
- Estimated travel time from Chicago, Illinois (starting location: 312 S Michigan Ave, Chicago, IL 60604) to the venue by car
- Reference URLs for: event details, ticket purchasing page, VIP package information, and supporting act confirmation
"""

START_LOCATION = "312 S Michigan Ave, Chicago, IL 60604"
DATE_RANGE_DESC = "between April 1, 2026 and May 31, 2026"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventURLs(BaseModel):
    event_details_url: Optional[str] = None
    ticket_platform_url: Optional[str] = None
    vip_info_url: Optional[str] = None
    supporting_act_reference_url: Optional[str] = None
    capacity_reference_url: Optional[str] = None
    other_reference_urls: List[str] = Field(default_factory=list)


class EventInfo(BaseModel):
    # Basic information
    date: Optional[str] = None
    artist: Optional[str] = None
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    venue_capacity: Optional[str] = None  # keep as string to be robust

    # Ticket information
    ticket_types: List[str] = Field(default_factory=list)
    ticket_platform: Optional[str] = None  # Ticketmaster, Live Nation, or official artist website
    vip_perks: List[str] = Field(default_factory=list)

    # Supporting acts
    supporting_acts: List[str] = Field(default_factory=list)

    # Travel information
    starting_location: Optional[str] = None
    travel_time_by_car: Optional[str] = None

    # URLs
    urls: EventURLs = Field(default_factory=EventURLs)


class EventsExtraction(BaseModel):
    events: List[EventInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract up to four concert events described in the answer, following this schema for each event:

    Fields to extract per event:
    - date: The event date string exactly as written (e.g., "May 12, 2026")
    - artist: Primary performing artist name
    - venue_name: Venue name
    - venue_address: Full street address including city and state if present (e.g., "123 Main St, City, ST 12345")
    - city: U.S. city name (just the city, no state)
    - state: Two-letter U.S. state abbreviation (e.g., "IL") or full state name if that's how it's written
    - venue_capacity: The documented capacity text (e.g., "20,000", "approx. 18,500 seats")
    - ticket_types: Array of ticket types/tiers mentioned (e.g., ["General Admission", "Reserved Seating", "VIP"])
    - ticket_platform: Ticket platform (one of: "Ticketmaster", "Live Nation", "Official Artist Website") — if another official platform is explicitly named, include it as written
    - vip_perks: Array of specific VIP perks or benefits listed (e.g., ["early entry", "exclusive merchandise", "premium seating"])
    - supporting_acts: Array of supporting acts / special guests (at least one if provided)
    - starting_location: The starting location used for the travel time estimate (should be "312 S Michigan Ave, Chicago, IL 60604")
    - travel_time_by_car: The driving travel time estimate string as written (e.g., "4 hr 35 min")

    Reference URLs per event (embed under 'urls'):
    - event_details_url: Official event details page (venue site, artist site, or ticketing platform event page)
    - ticket_platform_url: Direct URL to the ticket purchasing page
    - vip_info_url: URL specifically detailing VIP or premium package info (can be same as ticket page if VIP details are there)
    - supporting_act_reference_url: URL confirming supporting act announcement (can be event page, artist news page, or official social post URL if included)
    - capacity_reference_url: URL that documents the venue capacity (venue’s official page or a reliable source)
    - other_reference_urls: Any other relevant URLs cited for the event (array)

    Rules:
    - Extract exactly as stated in the answer; do not invent data.
    - If a field is not present, set it to null (for strings) or an empty array (for list fields).
    - For URLs, extract valid, complete URLs that are explicitly present. If in markdown, return the actual URL.
    - Return the first four events if more than four are provided. Preserve order of appearance.

    Return a JSON object:
    {
      "events": [EventInfo, EventInfo, EventInfo, EventInfo]
    }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _collect_sources(*urls: Optional[str], extra: Optional[List[str]] = None) -> Optional[List[str]]:
    """Collect non-empty URLs into a list suitable for verify(); return None if no sources."""
    collected: List[str] = []
    for u in urls:
        if u and isinstance(u, str) and u.strip():
            collected.append(u.strip())
    if extra:
        for u in extra:
            if u and isinstance(u, str) and u.strip():
                collected.append(u.strip())
    if not collected:
        return None
    return collected


def _ensure_length(events: List[EventInfo], k: int = 4) -> List[EventInfo]:
    """Pad or truncate the events list to exactly k items."""
    evs = events[:k]
    while len(evs) < k:
        evs.append(EventInfo())
    return evs


def _city_state_str(city: Optional[str], state: Optional[str]) -> str:
    city_s = city or ""
    state_s = state or ""
    if city_s and state_s:
        return f"{city_s}, {state_s}"
    return (city_s or "") + (state_s or "")


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_basic_information(evaluator: Evaluator, parent_node, ev: EventInfo, idx: int, prior_cities: List[str]) -> None:
    basic = evaluator.add_parallel(
        id=f"Event_{idx+1}_Basic_Information",
        desc="Basic event details including date, artist, venue, and location",
        parent=parent_node,
        critical=True
    )

    sources_general = _collect_sources(
        ev.urls.event_details_url,
        ev.urls.ticket_platform_url,
        extra=ev.urls.other_reference_urls
    )

    # Date within range
    n_date = evaluator.add_leaf(
        id=f"Event_{idx+1}_Date_Verification",
        desc=f"Event date falls {DATE_RANGE_DESC}",
        parent=basic,
        critical=True
    )
    date_text = ev.date or ""
    await evaluator.verify(
        claim=f"The event date is '{date_text}', and it falls {DATE_RANGE_DESC}.",
        node=n_date,
        sources=sources_general,
        additional_instruction=f"Use the event date displayed on the provided page(s). Confirm the date lies between 2026-04-01 and 2026-05-31 (inclusive). If multiple dates are listed, verify that the specific show at {ev.venue_name or 'the venue'} in {_city_state_str(ev.city, ev.state)} is within the range."
    )

    # Artist identified and correct
    n_artist = evaluator.add_leaf(
        id=f"Event_{idx+1}_Artist_Identification",
        desc="Primary performing artist is clearly identified",
        parent=basic,
        critical=True
    )
    artist_text = ev.artist or ""
    await evaluator.verify(
        claim=f"The primary performing artist for this event is '{artist_text}'.",
        node=n_artist,
        sources=sources_general,
        additional_instruction="Verify the page clearly lists the main headliner/primary artist for the event."
    )

    # Venue name
    n_venue_name = evaluator.add_leaf(
        id=f"Event_{idx+1}_Venue_Name",
        desc="Venue name is provided",
        parent=basic,
        critical=True
    )
    venue_name_text = ev.venue_name or ""
    await evaluator.verify(
        claim=f"The venue for this event is '{venue_name_text}'.",
        node=n_venue_name,
        sources=sources_general,
        additional_instruction="Confirm the venue name shown on the page matches the provided venue name."
    )

    # Venue capacity >= 10,000
    n_capacity = evaluator.add_leaf(
        id=f"Event_{idx+1}_Venue_Capacity",
        desc="Venue has documented capacity of at least 10,000 attendees",
        parent=basic,
        critical=True
    )
    capacity_sources = _collect_sources(
        ev.urls.capacity_reference_url,
        ev.urls.event_details_url,
        extra=ev.urls.other_reference_urls
    )
    await evaluator.verify(
        claim=f"The venue '{venue_name_text}' has a documented seating capacity of at least 10,000.",
        node=n_capacity,
        sources=capacity_sources,
        additional_instruction="Look for capacity numbers on the venue's official site or a reliable page. If an exact figure is shown, confirm it meets or exceeds 10,000."
    )

    # City and state specified
    n_city_state = evaluator.add_leaf(
        id=f"Event_{idx+1}_City_State",
        desc="U.S. city and state are specified",
        parent=basic,
        critical=True
    )
    city_state_text = _city_state_str(ev.city, ev.state)
    await evaluator.verify(
        claim=f"The event takes place in {city_state_text}.",
        node=n_city_state,
        sources=sources_general,
        additional_instruction="Verify the listed city and state for the event."
    )

    # Different city checks for Events 2–4
    if idx >= 1:
        prev_unique_cities_lower = [c.lower() for c in prior_cities if c]
        current_city_lower = (ev.city or "").lower()
        different_city_result = bool(current_city_lower) and (current_city_lower not in prev_unique_cities_lower)
        evaluator.add_custom_node(
            result=different_city_result,
            id=f"Event_{idx+1}_Different_City",
            desc=f"Event is in a different U.S. city than prior event(s)",
            parent=basic,
            critical=True
        )

    # Venue address
    n_venue_address = evaluator.add_leaf(
        id=f"Event_{idx+1}_Venue_Address",
        desc="Complete venue street address is provided",
        parent=basic,
        critical=True
    )
    address_text = ev.venue_address or ""
    await evaluator.verify(
        claim=f"The venue's street address is '{address_text}'.",
        node=n_venue_address,
        sources=sources_general,
        additional_instruction="Confirm that a complete street address for the venue is provided on the page."
    )

    # Reference URL (official event details)
    evaluator.add_custom_node(
        result=bool(ev.urls.event_details_url and ev.urls.event_details_url.strip()),
        id=f"Event_{idx+1}_Reference_URL",
        desc="Official reference URL from venue, artist website, or ticketing platform is provided",
        parent=basic,
        critical=True
    )


async def verify_ticket_information(evaluator: Evaluator, parent_node, ev: EventInfo, idx: int) -> None:
    ticket = evaluator.add_parallel(
        id=f"Event_{idx+1}_Ticket_Information",
        desc="Ticket types, tiers, and purchasing information",
        parent=parent_node,
        critical=True
    )

    tickets_sources = _collect_sources(
        ev.urls.ticket_platform_url,
        ev.urls.event_details_url,
        extra=ev.urls.other_reference_urls
    )

    # Multiple ticket types
    n_multi_tickets = evaluator.add_leaf(
        id=f"Event_{idx+1}_Multiple_Ticket_Types",
        desc="At least two different ticket types or tiers are available (e.g., GA, Reserved, VIP)",
        parent=ticket,
        critical=True
    )
    ticket_types_text = ", ".join(ev.ticket_types) if ev.ticket_types else ""
    await evaluator.verify(
        claim=f"This event offers at least two distinct ticket types or tiers, such as: {ticket_types_text}.",
        node=n_multi_tickets,
        sources=tickets_sources,
        additional_instruction="Confirm that the ticketing page lists two or more distinct options (e.g., GA, reserved seating levels, floor vs. bowl, VIP tiers)."
    )

    # VIP package availability
    n_vip_avail = evaluator.add_leaf(
        id=f"Event_{idx+1}_VIP_Package_Availability",
        desc="VIP or premium ticket packages are available",
        parent=ticket,
        critical=True
    )
    vip_sources = _collect_sources(
        ev.urls.vip_info_url,
        ev.urls.ticket_platform_url,
        extra=ev.urls.other_reference_urls
    )
    await evaluator.verify(
        claim="VIP or premium ticket packages are available for this event.",
        node=n_vip_avail,
        sources=vip_sources,
        additional_instruction="Confirm that the page explicitly offers VIP/premium packages (e.g., VIP tickets, premium seating packages)."
    )

    # VIP package details (perks + reference URL)
    vip_details = evaluator.add_parallel(
        id=f"Event_{idx+1}_VIP_Package_Details",
        desc="Specific perks or benefits of VIP/premium packages are documented",
        parent=ticket,
        critical=True
    )

    n_vip_perks = evaluator.add_leaf(
        id=f"Event_{idx+1}_VIP_Perks_Listed",
        desc="At least two specific VIP perks are identified (e.g., early entry, merchandise, meet and greet, premium seating)",
        parent=vip_details,
        critical=True
    )
    vip_perks_text = ", ".join(ev.vip_perks) if ev.vip_perks else ""
    await evaluator.verify(
        claim=f"The VIP/premium package includes at least two specific perks: {vip_perks_text}.",
        node=n_vip_perks,
        sources=vip_sources,
        additional_instruction="Verify that there are at least two distinct, explicit perks listed for VIP/premium packages."
    )

    evaluator.add_custom_node(
        result=bool(ev.urls.vip_info_url and ev.urls.vip_info_url.strip()),
        id=f"Event_{idx+1}_VIP_Reference_URL",
        desc="Reference URL for VIP package information is provided",
        parent=vip_details,
        critical=True
    )

    # Ticket purchase platform (type)
    n_platform = evaluator.add_leaf(
        id=f"Event_{idx+1}_Ticket_Purchase_Platform",
        desc="Official ticket purchasing platform is identified (Ticketmaster, Live Nation, or official artist website)",
        parent=ticket,
        critical=True
    )
    platform_text = ev.ticket_platform or ""
    await evaluator.verify(
        claim=f"The official ticket purchasing platform for this event is '{platform_text}'.",
        node=n_platform,
        sources=tickets_sources,
        additional_instruction="Confirm that the identified platform matches the page (Ticketmaster, Live Nation, or official artist site)."
    )

    # Ticket platform URL
    n_platform_url = evaluator.add_leaf(
        id=f"Event_{idx+1}_Ticket_Platform_URL",
        desc="Direct URL to ticket purchasing page is provided",
        parent=ticket,
        critical=True
    )
    ticket_url_text = ev.urls.ticket_platform_url or ""
    await evaluator.verify(
        claim=f"This URL is the official ticket purchasing page for the event: {ticket_url_text}.",
        node=n_platform_url,
        sources=_collect_sources(ev.urls.ticket_platform_url),
        additional_instruction="Confirm that the given URL leads to the event's ticket buying page (not just a generic site homepage)."
    )


async def verify_supporting_acts(evaluator: Evaluator, parent_node, ev: EventInfo, idx: int) -> None:
    supp = evaluator.add_parallel(
        id=f"Event_{idx+1}_Supporting_Acts",
        desc="Information about special guests or supporting performers",
        parent=parent_node,
        critical=True
    )

    supp_sources = _collect_sources(
        ev.urls.supporting_act_reference_url,
        ev.urls.event_details_url,
        extra=ev.urls.other_reference_urls
    )

    # Supporting act identified
    n_supp_identified = evaluator.add_leaf(
        id=f"Event_{idx+1}_Supporting_Act_Identified",
        desc="At least one officially announced special guest or supporting act is identified",
        parent=supp,
        critical=True
    )
    supp_text = ", ".join(ev.supporting_acts) if ev.supporting_acts else ""
    await evaluator.verify(
        claim=f"At least one supporting act is officially announced for this event: {supp_text}.",
        node=n_supp_identified,
        sources=supp_sources,
        additional_instruction="Confirm that the page explicitly lists at least one supporting act or special guest."
    )

    evaluator.add_custom_node(
        result=bool(ev.urls.supporting_act_reference_url and ev.urls.supporting_act_reference_url.strip()),
        id=f"Event_{idx+1}_Supporting_Act_Reference",
        desc="Reference URL confirming the supporting act announcement is provided",
        parent=supp,
        critical=True
    )


async def verify_travel_information(evaluator: Evaluator, parent_node, ev: EventInfo, idx: int) -> None:
    travel = evaluator.add_sequential(
        id=f"Event_{idx+1}_Travel_Information",
        desc="Travel logistics from specified starting location",
        parent=parent_node,
        critical=True
    )

    # Starting location specified correctly
    n_start_loc = evaluator.add_leaf(
        id=f"Event_{idx+1}_Starting_Location",
        desc=f"Starting location for travel calculation is specified as Chicago, Illinois ({START_LOCATION})",
        parent=travel,
        critical=True
    )
    start_loc_text = ev.starting_location or ""
    await evaluator.verify(
        claim=f"The travel time calculation uses the starting location '{START_LOCATION}'. Provided starting location: '{start_loc_text}'.",
        node=n_start_loc,
        sources=None,
        additional_instruction="Verify that the answer explicitly uses the specified Chicago address as the starting point."
    )

    # Travel time estimate provided
    n_travel_time = evaluator.add_leaf(
        id=f"Event_{idx+1}_Travel_Time_Estimate",
        desc="Estimated travel time from starting location to venue is provided",
        parent=travel,
        critical=True
    )
    travel_time_text = ev.travel_time_by_car or ""
    await evaluator.verify(
        claim=f"The estimated travel time from Chicago to the venue by car is provided as '{travel_time_text}'.",
        node=n_travel_time,
        sources=None,
        additional_instruction="Confirm that an explicit time estimate string is present (e.g., '4 hr 35 min')."
    )

    # Travel mode by car/driving
    n_travel_mode = evaluator.add_leaf(
        id=f"Event_{idx+1}_Travel_Mode",
        desc="Travel time is calculated by car/driving as specified in the question",
        parent=travel,
        critical=True
    )
    await evaluator.verify(
        claim="The travel time estimate was calculated for driving by car.",
        node=n_travel_mode,
        sources=None,
        additional_instruction="Confirm that the travel mode is car/driving."
    )


async def verify_event(evaluator: Evaluator, root_node, ev: EventInfo, idx: int, prior_cities: List[str]) -> None:
    event_node = evaluator.add_parallel(
        id=f"Event_{idx+1}",
        desc=f"{['First','Second','Third','Fourth'][idx]} concert event meeting all specified criteria",
        parent=root_node,
        critical=False
    )

    await verify_basic_information(evaluator, event_node, ev, idx, prior_cities)
    await verify_ticket_information(evaluator, event_node, ev, idx)
    await verify_supporting_acts(evaluator, event_node, ev, idx)
    await verify_travel_information(evaluator, event_node, ev, idx)


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
    Evaluate an answer for the Spring 2026 Concert Events task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates events in parallel
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Find four ticketed concert events in spring 2026 (April 1 - May 31) in different U.S. cities, each at venues with 10,000+ capacity, with complete event details, ticket information, supporting acts, and travel logistics",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured events info
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction"
    )

    # Keep exactly 4 events
    events = _ensure_length(extracted.events, 4)

    # Pre-compute prior cities for uniqueness checks
    prior_cities: List[str] = []
    for idx, ev in enumerate(events):
        await verify_event(evaluator, root, ev, idx, prior_cities)
        # Update prior cities list
        if ev.city:
            prior_cities.append(ev.city)

    # Add custom info about constraints
    evaluator.add_custom_info(
        info={
            "date_range": DATE_RANGE_DESC,
            "min_capacity": 10000,
            "starting_location": START_LOCATION,
            "required_events": 4
        },
        info_type="constraints",
        info_name="task_constraints"
    )

    # Return final summary
    return evaluator.get_summary()