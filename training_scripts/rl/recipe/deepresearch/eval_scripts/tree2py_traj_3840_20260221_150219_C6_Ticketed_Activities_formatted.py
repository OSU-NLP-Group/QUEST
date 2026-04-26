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
TASK_ID = "spring2026_events"
TASK_DESCRIPTION = """
You are planning entertainment options for visitors to the United States during spring 2026 (March 1 - May 31, 2026). Identify three distinct ticketed live entertainment events that meet the following requirements:

Event 1: A concert in New York City
- Must occur within the spring 2026 timeframe (March 1 - May 31, 2026)
- Venue must have a capacity of at least 1,000 people
- Provide: artist/band name, tour name (if applicable), venue name, complete venue address, specific performance date, performance start time (if available), venue capacity with verification source, starting ticket price with official ticketing source, and confirmation that tickets are available for purchase

Event 2: A Broadway theatrical show in New York City
- Must have performances during the spring 2026 timeframe (March 1 - May 31, 2026)
- Theater must have a seating capacity of at least 1,000 people
- Theater must offer wheelchair-accessible seating
- Provide: show name, theater name, complete theater address, specific performance dates available in spring 2026, show times (if available), theater seating capacity with verification source, starting ticket price with official ticketing source, wheelchair accessibility details (location of accessible seats and how to purchase), and confirmation that tickets are available for purchase

Event 3: A music festival in California
- Must occur within the spring 2026 timeframe (March 1 - May 31, 2026)
- Must span at least two consecutive days
- Venue must have a capacity of at least 1,000 people
- Provide: festival name, venue/park name, complete location (city and venue address), specific festival dates, festival hours (if available), attendance capacity with verification source, ticket types available (single-day, multi-day, etc.), starting ticket price for at least one ticket type with official ticketing source, and confirmation that tickets are available for purchase

For all three events, provide reference URLs to official sources (venue websites, Broadway.com, Ticketmaster, official festival websites, or other official ticketing platforms) where all the information can be verified.
"""

SPRING_2026_START = "2026-03-01"
SPRING_2026_END = "2026-05-31"
MIN_CAPACITY = 1000

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Event1Concert(BaseModel):
    artist_name: Optional[str] = None
    tour_name: Optional[str] = None
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    city: Optional[str] = None
    performance_date: Optional[str] = None
    performance_time: Optional[str] = None
    capacity: Optional[str] = None
    capacity_source_urls: List[str] = Field(default_factory=list)
    ticket_start_price: Optional[str] = None
    ticket_price_source_urls: List[str] = Field(default_factory=list)
    tickets_available: Optional[str] = None
    official_event_urls: List[str] = Field(default_factory=list)


class Event2Broadway(BaseModel):
    show_name: Optional[str] = None
    theater_name: Optional[str] = None
    theater_address: Optional[str] = None
    city: Optional[str] = None
    performance_dates: List[str] = Field(default_factory=list)
    show_times: List[str] = Field(default_factory=list)
    capacity: Optional[str] = None
    capacity_source_urls: List[str] = Field(default_factory=list)
    wheelchair_seating_available: Optional[str] = None
    wheelchair_accessibility_details: Optional[str] = None
    wheelchair_accessibility_source_urls: List[str] = Field(default_factory=list)
    ticket_start_price: Optional[str] = None
    ticket_price_source_urls: List[str] = Field(default_factory=list)
    tickets_available: Optional[str] = None
    official_show_urls: List[str] = Field(default_factory=list)


class Event3Festival(BaseModel):
    festival_name: Optional[str] = None
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    city: Optional[str] = None
    specific_dates: List[str] = Field(default_factory=list)
    festival_hours: List[str] = Field(default_factory=list)
    capacity: Optional[str] = None
    capacity_source_urls: List[str] = Field(default_factory=list)
    ticket_types: List[str] = Field(default_factory=list)
    ticket_start_price: Optional[str] = None
    ticket_price_source_urls: List[str] = Field(default_factory=list)
    tickets_available: Optional[str] = None
    official_festival_urls: List[str] = Field(default_factory=list)


class AllEventsExtraction(BaseModel):
    event1: Optional[Event1Concert] = None
    event2: Optional[Event2Broadway] = None
    event3: Optional[Event3Festival] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract structured details for three events described in the answer.

    EVENT 1: NYC Concert
    - artist_name: name of the performing artist or band
    - tour_name: tour or concert name if applicable
    - venue_name: exact venue name
    - venue_address: complete venue address (street, city, state)
    - city: the city where the venue is located
    - performance_date: specific performance date (YYYY-MM-DD if possible)
    - performance_time: start time if available
    - capacity: venue capacity number or description (e.g., "2,100")
    - capacity_source_urls: URLs that verify venue capacity (official venue site or authoritative sources)
    - ticket_start_price: starting ticket price (e.g., "$49", "from $49")
    - ticket_price_source_urls: URLs to official ticketing sources that show price (e.g., Ticketmaster, venue site)
    - tickets_available: indicate that tickets are available (e.g., "available", "on sale")
    - official_event_urls: official event pages or venue/ticketing URLs for this concert

    EVENT 2: Broadway Show (NYC)
    - show_name: Broadway show name
    - theater_name: exact theater name
    - theater_address: complete theater address (street, city, state)
    - city: the city where the theater is located
    - performance_dates: list of specific performance dates in spring 2026
    - show_times: list of performance times if available
    - capacity: theater seating capacity number or description
    - capacity_source_urls: URLs that verify theater capacity (official or authoritative sources)
    - wheelchair_seating_available: indicate wheelchair seating availability (e.g., "yes")
    - wheelchair_accessibility_details: description (location of accessible seats, how to purchase, companion seating info)
    - wheelchair_accessibility_source_urls: URLs confirming accessibility details (official theater or show pages)
    - ticket_start_price: starting ticket price
    - ticket_price_source_urls: official ticketing URLs showing price (Broadway.com, theater box office, Ticketmaster)
    - tickets_available: indicate tickets are available
    - official_show_urls: official show or theater pages

    EVENT 3: California Music Festival
    - festival_name: festival name
    - venue_name: venue or park name
    - venue_address: complete venue address (street, city, state)
    - city: city location
    - specific_dates: list of festival dates in spring 2026
    - festival_hours: start times or hours if available
    - capacity: attendance capacity number or description
    - capacity_source_urls: URLs verifying capacity
    - ticket_types: list of available ticket types (e.g., single-day, two-day, weekend pass)
    - ticket_start_price: starting price for at least one ticket type
    - ticket_price_source_urls: official ticketing URLs showing price
    - tickets_available: indicate tickets are available
    - official_festival_urls: official festival website or official ticketing pages

    Return a JSON with keys event1, event2, event3 corresponding to the above structures. If any item is missing, set its field(s) to null or empty list as appropriate. Extract only URLs explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _union_urls(*url_lists: List[str]) -> List[str]:
    """Combine multiple URL lists into a unique, flattened list."""
    seen = set()
    combined: List[str] = []
    for urls in url_lists:
        for u in urls or []:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification functions: Event 1 (NYC Concert)                               #
# --------------------------------------------------------------------------- #
async def verify_event_1(
    evaluator: Evaluator,
    parent_node,
    info: Event1Concert,
) -> None:
    event_node = evaluator.add_parallel(
        id="event_1_nyc_concert",
        desc="First event: A ticketed concert in New York City during spring 2026",
        parent=parent_node,
        critical=False
    )

    # Identification and basic requirements (critical group)
    ident_node = evaluator.add_parallel(
        id="event_1_identification",
        desc="Event identification and basic requirements",
        parent=event_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(info.artist_name and info.artist_name.strip()),
        id="event_1_artist_name",
        desc="The name of the performing artist or band must be provided",
        parent=ident_node,
        critical=True
    )

    # Location must be NYC
    loc_leaf = evaluator.add_leaf(
        id="event_1_location_nyc",
        desc="Event must be located in New York City",
        parent=ident_node,
        critical=True
    )
    loc_sources = _union_urls(info.official_event_urls, info.ticket_price_source_urls, info.capacity_source_urls)
    await evaluator.verify(
        claim=f"The event takes place in New York City, NY (venue: {info.venue_name or ''}, address: {info.venue_address or ''}).",
        node=loc_leaf,
        sources=loc_sources,
        additional_instruction="Confirm the venue address indicates New York City (Manhattan, Brooklyn, Queens, Bronx, or Staten Island)."
    )

    # Timeframe must be within Spring 2026
    timeframe_leaf = evaluator.add_leaf(
        id="event_1_timeframe",
        desc="Event must occur between March 1 and May 31, 2026",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The performance date {info.performance_date or ''} is between {SPRING_2026_START} and {SPRING_2026_END}.",
        node=timeframe_leaf,
        sources=loc_sources,
        additional_instruction="Check the official event or ticketing page to confirm the date is within Spring 2026."
    )

    # Type must be a ticketed live concert performance
    type_leaf = evaluator.add_leaf(
        id="event_1_type",
        desc="Event must be a ticketed live concert performance",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim="This event is a ticketed live concert performance.",
        node=type_leaf,
        sources=loc_sources,
        additional_instruction="Look for indicators such as concert description, artist performance, and ticket purchase options."
    )

    # Tour name (non-critical, separate to satisfy critical node consistency)
    evaluator.add_custom_node(
        result=bool(info.tour_name and info.tour_name.strip()),
        id="event_1_tour_name",
        desc="The tour or concert name must be provided if applicable",
        parent=event_node,
        critical=False
    )

    # Venue details (critical group)
    venue_node = evaluator.add_parallel(
        id="event_1_venue_details",
        desc="Venue specifications and location information",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.venue_name and info.venue_name.strip()),
        id="event_1_venue_name",
        desc="Exact venue name must be provided",
        parent=venue_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.venue_address and info.venue_address.strip()),
        id="event_1_venue_address",
        desc="Complete venue address must be provided",
        parent=venue_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.performance_date and info.performance_date.strip()),
        id="event_1_specific_date",
        desc="Specific performance date within spring 2026 must be provided",
        parent=venue_node,
        critical=True
    )

    # Showtime (non-critical, separate)
    evaluator.add_custom_node(
        result=bool(info.performance_time and info.performance_time.strip()),
        id="event_1_showtime",
        desc="Performance start time should be provided if available",
        parent=event_node,
        critical=False
    )

    # Capacity verification (critical group)
    cap_node = evaluator.add_parallel(
        id="event_1_capacity_verification",
        desc="Venue capacity requirements and verification",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.capacity and info.capacity.strip()),
        id="event_1_capacity_number",
        desc="The specific capacity number for the venue must be provided",
        parent=cap_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.capacity_source_urls),
        id="event_1_capacity_source",
        desc="Reference URL verifying the venue capacity must be provided",
        parent=cap_node,
        critical=True
    )
    cap_req_leaf = evaluator.add_leaf(
        id="event_1_capacity_requirement",
        desc="Venue must have a seating/attendance capacity of at least 1,000 people",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue has a capacity of at least 1,000 people.",
        node=cap_req_leaf,
        sources=info.capacity_source_urls,
        additional_instruction="Use the capacity source to confirm the venue's capacity meets or exceeds 1,000."
    )

    # Ticketing (critical group)
    ticket_node = evaluator.add_parallel(
        id="event_1_ticketing",
        desc="Ticket pricing and availability information",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.ticket_start_price and info.ticket_start_price.strip()),
        id="event_1_starting_price",
        desc="The starting ticket price must be provided",
        parent=ticket_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.ticket_price_source_urls),
        id="event_1_price_source",
        desc="Reference URL to official ticketing source verifying the price must be provided",
        parent=ticket_node,
        critical=True
    )
    avail_leaf = evaluator.add_leaf(
        id="event_1_ticket_availability",
        desc="Tickets must be currently available for purchase or confirmed to be on sale",
        parent=ticket_node,
        critical=True
    )
    await evaluator.verify(
        claim="Tickets for this concert are available for purchase (on sale).",
        node=avail_leaf,
        sources=_union_urls(info.ticket_price_source_urls, info.official_event_urls),
        additional_instruction="Confirm presence of 'Buy Tickets', 'Find Tickets', price display, or similar indicators on official ticketing or event pages."
    )

    # Official verification (critical group)
    official_node = evaluator.add_parallel(
        id="event_1_official_verification",
        desc="All information must be traceable to official sources",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.official_event_urls),
        id="event_1_official_event_page",
        desc="Reference URL to official event page (venue website or official ticketing platform) must be provided",
        parent=official_node,
        critical=True
    )
    verify_leaf = evaluator.add_leaf(
        id="event_1_verifiable_details",
        desc="All provided details (dates, venue, pricing) must match information on official sources",
        parent=official_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event at {info.venue_name or ''} on {info.performance_date or ''} with starting ticket price {info.ticket_start_price or ''} matches the information on the official sources.",
        node=verify_leaf,
        sources=_union_urls(info.official_event_urls, info.ticket_price_source_urls, info.capacity_source_urls),
        additional_instruction="Allow minor formatting differences. Confirm venue name, date, and starting price are consistent across official pages."
    )


# --------------------------------------------------------------------------- #
# Verification functions: Event 2 (Broadway Show)                             #
# --------------------------------------------------------------------------- #
async def verify_event_2(
    evaluator: Evaluator,
    parent_node,
    info: Event2Broadway,
) -> None:
    event_node = evaluator.add_parallel(
        id="event_2_broadway_show",
        desc="Second event: A Broadway theatrical show in New York City during spring 2026",
        parent=parent_node,
        critical=False
    )

    # Identification (critical group)
    ident_node = evaluator.add_parallel(
        id="event_2_identification",
        desc="Show identification and basic requirements",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.show_name and info.show_name.strip()),
        id="event_2_show_name",
        desc="The name of the Broadway show must be provided",
        parent=ident_node,
        critical=True
    )

    loc_leaf = evaluator.add_leaf(
        id="event_2_location_broadway",
        desc="Event must be a Broadway show in New York City",
        parent=ident_node,
        critical=True
    )
    base_sources = _union_urls(info.official_show_urls, info.ticket_price_source_urls)
    await evaluator.verify(
        claim="This is a Broadway show in New York City.",
        node=loc_leaf,
        sources=base_sources,
        additional_instruction="Confirm it's a Broadway production at a Broadway theater in NYC."
    )

    timeframe_leaf = evaluator.add_leaf(
        id="event_2_timeframe",
        desc="Show must have performances between March 1 and May 31, 2026",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"There are performance dates between {SPRING_2026_START} and {SPRING_2026_END}. Provided dates: {', '.join(info.performance_dates) if info.performance_dates else ''}.",
        node=timeframe_leaf,
        sources=base_sources,
        additional_instruction="Check the official schedule/calendar for spring 2026 dates."
    )

    type_leaf = evaluator.add_leaf(
        id="event_2_type",
        desc="Event must be a ticketed Broadway theatrical performance",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim="This is a ticketed Broadway theatrical performance.",
        node=type_leaf,
        sources=base_sources,
        additional_instruction="Look for ticket purchase links and show description indicating Broadway."
    )

    run_leaf = evaluator.add_leaf(
        id="event_2_run_status",
        desc="Show must be confirmed to be running during the specified timeframe",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim="The show is confirmed to be running during spring 2026.",
        node=run_leaf,
        sources=base_sources,
        additional_instruction="Confirm active run dates include at least one date in spring 2026."
    )

    # Theater details (critical group)
    theater_node = evaluator.add_parallel(
        id="event_2_theater_details",
        desc="Theater specifications and location information",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.theater_name and info.theater_name.strip()),
        id="event_2_theater_name",
        desc="Exact Broadway theater name must be provided",
        parent=theater_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.theater_address and info.theater_address.strip()),
        id="event_2_theater_address",
        desc="Complete theater address must be provided",
        parent=theater_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.performance_dates),
        id="event_2_specific_dates",
        desc="Specific performance dates available within spring 2026 must be provided",
        parent=theater_node,
        critical=True
    )

    # Showtimes (non-critical, separate)
    evaluator.add_custom_node(
        result=bool(info.show_times),
        id="event_2_showtimes",
        desc="Performance times should be provided if available",
        parent=event_node,
        critical=False
    )

    # Capacity verification (critical group)
    cap_node = evaluator.add_parallel(
        id="event_2_capacity_verification",
        desc="Theater capacity requirements and verification",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.capacity and info.capacity.strip()),
        id="event_2_capacity_number",
        desc="The specific seating capacity of the theater must be provided",
        parent=cap_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.capacity_source_urls),
        id="event_2_capacity_source",
        desc="Reference URL verifying the theater capacity must be provided",
        parent=cap_node,
        critical=True
    )
    cap_req_leaf = evaluator.add_leaf(
        id="event_2_capacity_requirement",
        desc="Theater must have a seating capacity of at least 1,000 people",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim="The theater has a capacity of at least 1,000 seats.",
        node=cap_req_leaf,
        sources=info.capacity_source_urls,
        additional_instruction="Use the capacity source to confirm seats >= 1,000."
    )

    # Ticketing (critical group)
    ticket_node = evaluator.add_parallel(
        id="event_2_ticketing",
        desc="Ticket pricing and availability information",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.ticket_start_price and info.ticket_start_price.strip()),
        id="event_2_starting_price",
        desc="The starting ticket price must be provided",
        parent=ticket_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.ticket_price_source_urls),
        id="event_2_price_source",
        desc="Reference URL to official ticketing source verifying the price must be provided",
        parent=ticket_node,
        critical=True
    )
    avail_leaf = evaluator.add_leaf(
        id="event_2_ticket_availability",
        desc="Tickets must be currently available for purchase or confirmed to be on sale",
        parent=ticket_node,
        critical=True
    )
    await evaluator.verify(
        claim="Tickets for this Broadway show are available for purchase (on sale).",
        node=avail_leaf,
        sources=_union_urls(info.ticket_price_source_urls, info.official_show_urls),
        additional_instruction="Confirm presence of active ticket purchase options on official sources."
    )

    # Accessibility (critical group)
    access_node = evaluator.add_parallel(
        id="event_2_accessibility",
        desc="Wheelchair accessibility requirements for Broadway theater",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.wheelchair_accessibility_source_urls),
        id="event_2_accessibility_source",
        desc="Reference URL confirming wheelchair accessibility information must be provided",
        parent=access_node,
        critical=True
    )
    wc_leaf = evaluator.add_leaf(
        id="event_2_wheelchair_seating",
        desc="Theater must have wheelchair-accessible seating available",
        parent=access_node,
        critical=True
    )
    await evaluator.verify(
        claim="The theater offers wheelchair-accessible seating.",
        node=wc_leaf,
        sources=info.wheelchair_accessibility_source_urls,
        additional_instruction="Confirm explicit mention of wheelchair seating availability."
    )
    wc_detail_leaf = evaluator.add_leaf(
        id="event_2_accessibility_details",
        desc="Details about wheelchair accessibility (location of accessible seats, how to purchase, companion seating) must be provided",
        parent=access_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Accessibility details include where accessible seats are located and how to purchase accessible tickets. Details: {info.wheelchair_accessibility_details or ''}.",
        node=wc_detail_leaf,
        sources=info.wheelchair_accessibility_source_urls,
        additional_instruction="Look for specifics on accessible seating locations and purchasing instructions; companion seating info is a plus."
    )

    # Official verification (critical group)
    official_node = evaluator.add_parallel(
        id="event_2_official_verification",
        desc="All information must be traceable to official sources",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.official_show_urls),
        id="event_2_official_show_page",
        desc="Reference URL to official show page (Broadway.com, theater website, or official ticketing platform) must be provided",
        parent=official_node,
        critical=True
    )
    verify_leaf = evaluator.add_leaf(
        id="event_2_verifiable_details",
        desc="All provided details (show dates, theater, pricing, accessibility) must match information on official sources",
        parent=official_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The show at {info.theater_name or ''} with spring 2026 dates and starting ticket price {info.ticket_start_price or ''} and wheelchair accessibility details matches official sources.",
        node=verify_leaf,
        sources=_union_urls(info.official_show_urls, info.ticket_price_source_urls, info.capacity_source_urls, info.wheelchair_accessibility_source_urls),
        additional_instruction="Confirm consistency across official pages for theater name, dates, price, and accessibility."
    )


# --------------------------------------------------------------------------- #
# Verification functions: Event 3 (California Music Festival)                 #
# --------------------------------------------------------------------------- #
async def verify_event_3(
    evaluator: Evaluator,
    parent_node,
    info: Event3Festival,
) -> None:
    event_node = evaluator.add_parallel(
        id="event_3_california_festival",
        desc="Third event: A multi-day music festival in California during spring 2026",
        parent=parent_node,
        critical=False
    )

    # Identification (critical group)
    ident_node = evaluator.add_parallel(
        id="event_3_identification",
        desc="Festival identification and basic requirements",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.festival_name and info.festival_name.strip()),
        id="event_3_festival_name",
        desc="The name of the music festival must be provided",
        parent=ident_node,
        critical=True
    )
    loc_leaf = evaluator.add_leaf(
        id="event_3_location_california",
        desc="Festival must be located in California",
        parent=ident_node,
        critical=True
    )
    base_sources = _union_urls(info.official_festival_urls, info.ticket_price_source_urls)
    await evaluator.verify(
        claim=f"The festival is located in California (venue: {info.venue_name or ''}, address: {info.venue_address or ''}, city: {info.city or ''}).",
        node=loc_leaf,
        sources=base_sources,
        additional_instruction="Confirm the venue address and city indicate a location in California."
    )

    timeframe_leaf = evaluator.add_leaf(
        id="event_3_timeframe",
        desc="Festival must occur between March 1 and May 31, 2026",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The festival takes place during Spring 2026 (between {SPRING_2026_START} and {SPRING_2026_END}). Provided dates: {', '.join(info.specific_dates) if info.specific_dates else ''}.",
        node=timeframe_leaf,
        sources=base_sources,
        additional_instruction="Check the schedule/dates on official pages."
    )

    type_leaf = evaluator.add_leaf(
        id="event_3_type",
        desc="Event must be a ticketed music festival",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim="This event is a ticketed music festival.",
        node=type_leaf,
        sources=base_sources,
        additional_instruction="Look for festival description and ticket purchase options."
    )

    duration_leaf = evaluator.add_leaf(
        id="event_3_duration",
        desc="Festival must span at least two consecutive days",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim="The festival spans at least two consecutive days.",
        node=duration_leaf,
        sources=base_sources,
        additional_instruction="Confirm the listed dates include at least two consecutive days."
    )

    # Venue details (critical group)
    venue_node = evaluator.add_parallel(
        id="event_3_venue_details",
        desc="Venue specifications and location information",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.venue_name and info.venue_name.strip()),
        id="event_3_venue_name",
        desc="Exact venue or park name must be provided",
        parent=venue_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.city and info.city.strip() and info.venue_address and info.venue_address.strip()),
        id="event_3_city_location",
        desc="Complete location including city and venue address must be provided",
        parent=venue_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.specific_dates),
        id="event_3_specific_dates",
        desc="Specific festival dates within spring 2026 must be provided",
        parent=venue_node,
        critical=True
    )

    # Festival hours (non-critical, separate)
    evaluator.add_custom_node(
        result=bool(info.festival_hours),
        id="event_3_festival_hours",
        desc="Festival start times or hours should be provided if available",
        parent=event_node,
        critical=False
    )

    # Capacity verification (critical group)
    cap_node = evaluator.add_parallel(
        id="event_3_capacity_verification",
        desc="Festival capacity requirements and verification",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.capacity and info.capacity.strip()),
        id="event_3_capacity_number",
        desc="The specific attendance capacity for the festival must be provided",
        parent=cap_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.capacity_source_urls),
        id="event_3_capacity_source",
        desc="Reference URL verifying the festival capacity must be provided",
        parent=cap_node,
        critical=True
    )
    cap_req_leaf = evaluator.add_leaf(
        id="event_3_capacity_requirement",
        desc="Festival venue must have an attendance capacity of at least 1,000 people",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim="The festival venue has an attendance capacity of at least 1,000 people.",
        node=cap_req_leaf,
        sources=info.capacity_source_urls,
        additional_instruction="Confirm capacity meets or exceeds 1,000 using the referenced source."
    )

    # Ticketing (critical group)
    ticket_node = evaluator.add_parallel(
        id="event_3_ticketing",
        desc="Ticket pricing and availability information",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.ticket_start_price and info.ticket_start_price.strip()),
        id="event_3_starting_price",
        desc="The starting ticket price for at least one ticket type must be provided",
        parent=ticket_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.ticket_price_source_urls),
        id="event_3_price_source",
        desc="Reference URL to official ticketing source verifying the price must be provided",
        parent=ticket_node,
        critical=True
    )
    avail_leaf = evaluator.add_leaf(
        id="event_3_ticket_availability",
        desc="Tickets must be currently available for purchase or confirmed to be on sale",
        parent=ticket_node,
        critical=True
    )
    await evaluator.verify(
        claim="Tickets for this music festival are available for purchase (on sale).",
        node=avail_leaf,
        sources=_union_urls(info.ticket_price_source_urls, info.official_festival_urls),
        additional_instruction="Look for active ticket purchase options or 'on sale' labels on official pages."
    )

    # Ticket types (non-critical, separate)
    evaluator.add_custom_node(
        result=bool(info.ticket_types),
        id="event_3_ticket_types",
        desc="Available ticket types (single-day, two-day, weekend pass) must be specified",
        parent=event_node,
        critical=False
    )

    # Official verification (critical group)
    official_node = evaluator.add_parallel(
        id="event_3_official_verification",
        desc="All information must be traceable to official sources",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(info.official_festival_urls),
        id="event_3_official_festival_page",
        desc="Reference URL to official festival website or official ticketing platform must be provided",
        parent=official_node,
        critical=True
    )
    verify_leaf = evaluator.add_leaf(
        id="event_3_verifiable_details",
        desc="All provided details (dates, venue, pricing, duration) must match information on official sources",
        parent=official_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The festival at {info.venue_name or ''} on dates {', '.join(info.specific_dates) if info.specific_dates else ''} with starting ticket price {info.ticket_start_price or ''} matches official sources and spans multiple consecutive days.",
        node=verify_leaf,
        sources=_union_urls(info.official_festival_urls, info.ticket_price_source_urls, info.capacity_source_urls),
        additional_instruction="Confirm consistency across official pages for venue, dates, pricing, and multi-day duration."
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
    Evaluate the answer for the Spring 2026 events planning task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root should allow parallel aggregation across events
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

    # Extract events info
    events = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=AllEventsExtraction,
        extraction_name="events_extraction"
    )

    # Ground truth policy and constraints added for transparency
    evaluator.add_ground_truth({
        "timeframe": {"start": SPRING_2026_START, "end": SPRING_2026_END},
        "minimum_capacity": MIN_CAPACITY,
        "event_requirements": {
            "event_1": ["NYC concert", "capacity >= 1000", "tickets available", "official sources"],
            "event_2": ["Broadway show in NYC", "capacity >= 1000", "wheelchair accessible", "tickets available", "official sources"],
            "event_3": ["California music festival", "multi-day (>= 2 consecutive days)", "capacity >= 1000", "tickets available", "official sources"]
        }
    })

    # Create verification subtrees for each event
    event1 = events.event1 or Event1Concert()
    event2 = events.event2 or Event2Broadway()
    event3 = events.event3 or Event3Festival()

    await verify_event_1(evaluator, root, event1)
    await verify_event_2(evaluator, root, event2)
    await verify_event_3(evaluator, root, event3)

    # Return structured summary
    return evaluator.get_summary()