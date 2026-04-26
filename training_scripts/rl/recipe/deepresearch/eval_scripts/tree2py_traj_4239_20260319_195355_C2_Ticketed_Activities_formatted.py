import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "comedy_manhattan_2026"
TASK_DESCRIPTION = (
    "Identify one ticketed live comedy performance in Manhattan, New York City, that is scheduled between "
    "February 20 and March 10, 2026. For the comedy show you identify, provide the following information:\n\n"
    "1. Event Details: The show name, performer name(s), specific performance date and time, and show duration.\n"
    "2. Venue Information: The venue name, complete address (street address, city, state, ZIP code), and seating capacity.\n"
    "3. Ticketing Information: A link to an official ticketing platform where tickets can be purchased, and the ticket price range (minimum and maximum prices).\n"
    "4. Travel Information from Philadelphia: \n"
    "   - The travel time from Philadelphia's 30th Street Station to New York Penn Station via Amtrak\n"
    "   - The public transportation route (specific subway or bus line) from Penn Station to the venue, including estimated travel time\n"
    "   - The total travel time from Philadelphia's 30th Street Station to the venue\n\n"
    "Ensure that all information is verifiable through official sources, the venue is accessible via NYC public transportation "
    "(within a 10-minute walk from the nearest subway station), and the total travel time from Philadelphia is under 2.5 hours."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventDetails(BaseModel):
    show_name: Optional[str] = None
    performer_names: List[str] = Field(default_factory=list)
    performance_date: Optional[str] = None
    performance_time: Optional[str] = None
    duration: Optional[str] = None
    venue_name: Optional[str] = None
    official_event_urls: List[str] = Field(default_factory=list)


class VenueInfo(BaseModel):
    venue_name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    seating_capacity: Optional[str] = None
    venue_urls: List[str] = Field(default_factory=list)


class TicketingInfo(BaseModel):
    ticket_url: Optional[str] = None
    price_min: Optional[str] = None
    price_max: Optional[str] = None
    platform_name: Optional[str] = None


class TravelInfo(BaseModel):
    amtrak_travel_time: Optional[str] = None
    penn_to_venue_route: Optional[str] = None
    penn_to_venue_time: Optional[str] = None
    nearest_subway_station: Optional[str] = None
    walking_time_from_station: Optional[str] = None
    total_travel_time: Optional[str] = None
    travel_urls: List[str] = Field(default_factory=list)


class ComedyShowExtraction(BaseModel):
    event: Optional[EventDetails] = None
    venue: Optional[VenueInfo] = None
    ticketing: Optional[TicketingInfo] = None
    travel: Optional[TravelInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_comedy_show() -> str:
    return """
Extract the structured information for exactly one ticketed live comedy performance in Manhattan, NYC that the answer presents.

Return a JSON with fields:
- event:
  - show_name: The show name exactly as written.
  - performer_names: Array of performer names (stand-up comedian(s), troupe, or headliner(s)).
  - performance_date: The specific performance date for the Manhattan show (keep as text, e.g., "March 3, 2026" or "2026-03-03").
  - performance_time: The specific start time (keep as text, e.g., "8:00 PM").
  - duration: The show duration (e.g., "90 minutes" or "1h 30m") if provided; else null.
  - venue_name: The venue name.
  - official_event_urls: Array of URLs directly confirming the event details (acceptable: official venue/show pages, official ticketing pages).
- venue:
  - venue_name: The venue name.
  - street_address: Street address line (e.g., "123 Example St").
  - city: City (should be "New York" for Manhattan).
  - state: Two-letter state (e.g., "NY").
  - zip_code: ZIP or ZIP+4 if present (e.g., "10001").
  - seating_capacity: Seating capacity number or textual number if provided in the answer.
  - venue_urls: Array of URLs that confirm venue info (official venue website, venue profile pages, etc.).
- ticketing:
  - ticket_url: URL to an official ticketing platform page for purchasing tickets (e.g., venue ticketing, Ticketmaster, AXS, Eventbrite, TodayTix).
  - price_min: The minimum listed ticket price (as text, include currency symbol if present).
  - price_max: The maximum listed ticket price (as text).
  - platform_name: Name of the ticketing platform, if stated.
- travel:
  - amtrak_travel_time: The Amtrak travel time from Philadelphia 30th Street Station to New York Penn Station (as text, e.g., "1h 20m").
  - penn_to_venue_route: Specific subway or bus line(s) from Penn Station to the venue (e.g., "Take the 1 subway north...").
  - penn_to_venue_time: Estimated travel time for that public-transit segment (as text).
  - nearest_subway_station: The nearest NYC subway station to the venue (name).
  - walking_time_from_station: Estimated walk time from that station to the venue (as text, e.g., "6 minutes").
  - total_travel_time: The total time from Philadelphia 30th Street Station to the venue (as text).
  - travel_urls: Array of URLs supporting the travel timings and route (e.g., Amtrak schedules, MTA/Google Maps directions, Citymapper).

Rules:
- Extract only what is explicitly present in the answer. Do not invent.
- For any missing field, return null (or empty list for arrays).
- Only include URLs actually present in the answer text (any reasonable format, including markdown links).
- Keep all values as strings as they appear (do not normalize to numbers).
"""


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _is_valid_url(u: Optional[str]) -> bool:
    if not u:
        return False
    u = u.strip()
    return u.startswith("http://") or u.startswith("https://")


def _combine_sources(*url_lists: List[str]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if _is_valid_url(u) and u not in seen:
                combined.append(u)
                seen.add(u)
    return combined


def _as_name_list(names: Optional[List[str]]) -> str:
    if not names:
        return ""
    return ", ".join([n for n in names if n])


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_event_identification(evaluator: Evaluator, root_node, data: ComedyShowExtraction) -> None:
    event_node = evaluator.add_parallel(
        id="Event_Identification",
        desc="Verify that a valid comedy event has been identified with all required performance details",
        parent=root_node,
        critical=True
    )

    event = data.event or EventDetails()
    venue = data.venue or VenueInfo()
    ticket = data.ticketing or TicketingInfo()

    # URL Reference (existence gate)
    urls_event = _combine_sources(event.official_event_urls, [ticket.ticket_url] if _is_valid_url(ticket.ticket_url) else [])
    evaluator.add_custom_node(
        result=len(urls_event) > 0,
        id="Event_URL_Reference",
        desc="A valid URL from an official source (venue website, ticketing platform, or official show page) that confirms the event details is provided",
        parent=event_node,
        critical=True
    )

    # Event Type: live ticketed comedy with professional comedian(s)
    node_event_type = evaluator.add_leaf(
        id="Event_Type",
        desc="The identified event is a ticketed live comedy performance featuring professional comedian(s)",
        parent=event_node,
        critical=True
    )
    claim_event_type = (
        f"The event '{event.show_name or ''}' featuring {_as_name_list(event.performer_names)} is a live, ticketed comedy performance "
        f"by professional comedian(s)."
    )
    await evaluator.verify(
        claim=claim_event_type,
        node=node_event_type,
        sources=urls_event,
        additional_instruction=(
            "Confirm from the provided official pages that this is a comedy show (e.g., stand-up, improv, sketch). "
            "It must be a live event with tickets for sale (not a free open-mic, lecture, or non-comedy show)."
        ),
    )

    # Location: Manhattan, NYC
    node_location = evaluator.add_leaf(
        id="Location",
        desc="The venue is located in Manhattan, New York City",
        parent=event_node,
        critical=True
    )
    urls_location = _combine_sources(venue.venue_urls, event.official_event_urls)
    claim_location = (
        f"The venue '{venue.venue_name or ''}' is located in Manhattan, New York City (New York County, NY)."
    )
    await evaluator.verify(
        claim=claim_location,
        node=node_location,
        sources=urls_location,
        additional_instruction=(
            "Accept if the address or description clearly indicates Manhattan (e.g., borough is Manhattan, or "
            "neighborhoods like Midtown, SoHo, Lower East Side, Upper West Side, etc.). 'New York, NY' alone is insufficient "
            "unless the borough or context indicates Manhattan."
        ),
    )

    # Date Timeframe: between Feb 20 and Mar 10, 2026 inclusive
    node_timeframe = evaluator.add_leaf(
        id="Date_Timeframe",
        desc="The performance is scheduled between February 20, 2026, and March 10, 2026",
        parent=event_node,
        critical=True
    )
    claim_timeframe = (
        f"The performance date is {event.performance_date or ''} at {event.performance_time or ''}, and it falls "
        f"between February 20, 2026 and March 10, 2026 inclusive."
    )
    await evaluator.verify(
        claim=claim_timeframe,
        node=node_timeframe,
        sources=urls_event,
        additional_instruction=(
            "First confirm the specific performance date/time on the provided page(s). Then judge whether the date is on or after "
            "Feb 20, 2026 and on or before Mar 10, 2026. If multiple dates are shown, accept if the cited date/time is within the window."
        ),
    )

    # Performance Details provided and supported
    node_perf_details = evaluator.add_leaf(
        id="Performance_Details",
        desc="Specific performance date, time, performer names, and show duration are provided",
        parent=event_node,
        critical=True
    )
    claim_perf = (
        f"The event page(s) explicitly list: performer(s) {_as_name_list(event.performer_names)}, the performance date "
        f"{event.performance_date or ''}, the start time {event.performance_time or ''}, and the show duration "
        f"{event.duration or ''}."
    )
    await evaluator.verify(
        claim=claim_perf,
        node=node_perf_details,
        sources=urls_event,
        additional_instruction=(
            "All four elements (performer names, specific date, start time, and duration) must be visible or clearly stated. "
            "If duration is missing on all provided pages, this check should fail."
        ),
    )


async def verify_venue_information(evaluator: Evaluator, root_node, data: ComedyShowExtraction) -> None:
    venue_node = evaluator.add_parallel(
        id="Venue_Information",
        desc="Complete venue details are provided and verified",
        parent=root_node,
        critical=True
    )

    venue = data.venue or VenueInfo()
    urls_venue = venue.venue_urls or []

    # URL Reference (existence gate)
    evaluator.add_custom_node(
        result=any(_is_valid_url(u) for u in urls_venue),
        id="Venue_URL_Reference",
        desc="A valid URL confirming venue information is provided",
        parent=venue_node,
        critical=True
    )

    # Venue Name and Address
    node_addr = evaluator.add_leaf(
        id="Venue_Name_and_Address",
        desc="The venue name and complete address (including street address, city, state, and ZIP code) are provided",
        parent=venue_node,
        critical=True
    )
    address_full = " ".join(
        [p for p in [venue.street_address, venue.city, venue.state, venue.zip_code] if p]
    ).strip()
    claim_addr = (
        f"The venue is '{venue.venue_name or ''}' and its complete address is '{venue.street_address or ''}, "
        f"{venue.city or ''}, {venue.state or ''} {venue.zip_code or ''}', as shown on the referenced page(s)."
    )
    await evaluator.verify(
        claim=claim_addr,
        node=node_addr,
        sources=urls_venue,
        additional_instruction=(
            "Confirm the page(s) show the venue name and a complete mailing address including street, city, state (NY), and ZIP. "
            "Minor formatting differences are acceptable as long as the components are present."
        ),
    )

    # Venue Capacity
    node_capacity = evaluator.add_leaf(
        id="Venue_Capacity",
        desc="The venue's seating capacity is provided and verifiable",
        parent=venue_node,
        critical=True
    )
    claim_capacity = (
        f"The seating capacity of '{venue.venue_name or ''}' is {venue.seating_capacity or ''}."
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=node_capacity,
        sources=urls_venue,
        additional_instruction=(
            "Look for an explicit capacity figure on the official venue page or authoritative source page(s). "
            "If capacity is not provided or cannot be confirmed, this should fail."
        ),
    )


async def verify_ticketing_information(evaluator: Evaluator, root_node, data: ComedyShowExtraction) -> None:
    ticket_node = evaluator.add_parallel(
        id="Ticketing_Information",
        desc="Ticket availability and pricing information are provided",
        parent=root_node,
        critical=True
    )

    event = data.event or EventDetails()
    ticket = data.ticketing or TicketingInfo()
    ticket_url_list = [ticket.ticket_url] if _is_valid_url(ticket.ticket_url) else []

    # Ticket Availability
    node_tix_avail = evaluator.add_leaf(
        id="Ticket_Availability",
        desc="Tickets are confirmed to be available for purchase through official platforms with a valid ticketing URL provided",
        parent=ticket_node,
        critical=True
    )
    claim_tix_avail = (
        f"Tickets for '{event.show_name or ''}' on {event.performance_date or ''} at {event.performance_time or ''} "
        f"at '{event.venue_name or ''}' are purchasable on the official ticketing page {ticket.ticket_url or ''}."
    )
    await evaluator.verify(
        claim=claim_tix_avail,
        node=node_tix_avail,
        sources=ticket_url_list,
        additional_instruction=(
            "This must be an official ticketing purchase page (e.g., venue ticketing, Ticketmaster, AXS, Eventbrite, TodayTix), "
            "not just a listing or review. The page should indicate purchasable tickets for the specified date/time; if sold out "
            "or no purchasing flow is available, this should fail."
        ),
    )

    # Ticket Price Range
    node_price_range = evaluator.add_leaf(
        id="Ticket_Price_Range",
        desc="A specific ticket price range is provided with minimum and maximum prices",
        parent=ticket_node,
        critical=True
    )
    claim_price = (
        f"The listed ticket price range on the ticketing page is from {ticket.price_min or ''} to {ticket.price_max or ''}."
    )
    await evaluator.verify(
        claim=claim_price,
        node=node_price_range,
        sources=ticket_url_list,
        additional_instruction=(
            "Confirm that both a minimum and a maximum ticket price are visible (before fees is acceptable). "
            "If only a single price or a vague statement is shown, this should fail."
        ),
    )


async def verify_travel_accessibility(evaluator: Evaluator, root_node, data: ComedyShowExtraction) -> None:
    travel_node = evaluator.add_parallel(
        id="Travel_Accessibility",
        desc="Complete travel information from Philadelphia to the venue is provided",
        parent=root_node,
        critical=True
    )

    travel = data.travel or TravelInfo()
    event = data.event or EventDetails()
    venue = data.venue or VenueInfo()
    travel_urls = travel.travel_urls or []

    # Philadelphia -> NY Penn (Amtrak)
    node_amtrak = evaluator.add_leaf(
        id="Philadelphia_to_Penn_Station",
        desc="Travel time from Philadelphia 30th Street Station to New York Penn Station via Amtrak is provided",
        parent=travel_node,
        critical=True
    )
    claim_amtrak = (
        f"The typical Amtrak travel time from Philadelphia 30th Street Station to New York Penn Station is "
        f"{travel.amtrak_travel_time or ''}."
    )
    await evaluator.verify(
        claim=claim_amtrak,
        node=node_amtrak,
        sources=travel_urls,
        additional_instruction=(
            "Prefer official Amtrak schedules or clearly reliable sources. Accept a typical or representative "
            "duration for Northeast Regional or Acela services. If no credible duration is found on the provided URLs, fail."
        ),
    )

    # Penn Station -> Venue via public transit, walking within 10 minutes
    node_penn_to_venue = evaluator.add_leaf(
        id="Penn_Station_to_Venue",
        desc="Public transportation route (subway/bus line and estimated travel time) from Penn Station to the venue is provided, and the venue is verified to be within a 10-minute walk from the nearest subway station",
        parent=travel_node,
        critical=True
    )
    claim_route = (
        f"From Penn Station to '{venue.venue_name or ''}', the public transit route is: {travel.penn_to_venue_route or ''}, "
        f"with an estimated travel time of {travel.penn_to_venue_time or ''}. The venue is within a 10-minute walk from the "
        f"nearest subway station ({travel.nearest_subway_station or ''}), with a walking time of {travel.walking_time_from_station or ''}."
    )
    await evaluator.verify(
        claim=claim_route,
        node=node_penn_to_venue,
        sources=travel_urls,
        additional_instruction=(
            "Verify the named NYC subway/bus route(s) and estimated time from Penn Station. Also confirm that the nearest subway "
            "station to the venue is within a 10-minute walk (<= 10 minutes). Use MTA/Google Maps/Citymapper links if provided."
        ),
    )

    # Total travel time under 2.5 hours
    node_total_time = evaluator.add_leaf(
        id="Total_Travel_Time",
        desc="The total travel time from Philadelphia 30th Street Station to the venue is calculated and is under 2.5 hours",
        parent=travel_node,
        critical=True
    )
    claim_total = (
        f"The total travel time from Philadelphia 30th Street Station to '{venue.venue_name or ''}' is "
        f"{travel.total_travel_time or ''}, which is under 2.5 hours."
    )
    await evaluator.verify(
        claim=claim_total,
        node=node_total_time,
        sources=travel_urls,
        additional_instruction=(
            "Cross-check the total time implied by the Amtrak segment plus the NYC public transit and walking. "
            "If the stated total is clearly >= 2.5 hours, or cannot be supported by the provided URLs, this should fail."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Manhattan comedy show task.
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
        default_model=model,
    )

    # Extraction
    extracted: ComedyShowExtraction = await evaluator.extract(
        prompt=prompt_extract_comedy_show(),
        template_class=ComedyShowExtraction,
        extraction_name="comedy_show_extraction",
    )

    # Build top-level rubric branches (all critical per rubric)
    # We add them as containers; each leaf under them will be critical.
    # Note: We directly verify leaves in the helper functions using these containers.
    # Order of verification within each section ensures URL reference gates are evaluated early where applicable.

    # Event Identification
    await verify_event_identification(evaluator, root, extracted)

    # Venue Information
    await verify_venue_information(evaluator, root, extracted)

    # Ticketing Information
    await verify_ticketing_information(evaluator, root, extracted)

    # Travel Accessibility
    await verify_travel_accessibility(evaluator, root, extracted)

    return evaluator.get_summary()