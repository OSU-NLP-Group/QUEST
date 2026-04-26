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
TASK_ID = "nyc_live_apr_2026"
TASK_DESCRIPTION = """
Find four different types of live performances in New York City during April 2026, meeting the following specific requirements:

Performance 1 - Large Concert Venue:
Find a concert or music performance at a venue with a capacity of at least 5,000 people. Provide:
- Venue name and address
- Venue capacity
- Date and time of performance
- Name of performing artist(s)
- Ticket availability and price range
- Reference URL from an official source

Performance 2 - Broadway Show:
Find a Broadway theatrical production. Provide:
- Show/musical name
- Theater name and address
- Confirmation that it's an official Broadway theater (minimum 500 seats, located in the Theater District between 41st-54th Streets and 6th-8th Avenues)
- Date(s) available during April 2026
- Ticket availability and price range
- Reference URL from an official source

Performance 3 - Stand-Up Comedy:
Find a stand-up comedy show or performance. Provide:
- Comedian name(s)
- Venue name and address
- Confirmation that the venue hosts comedy performances
- Date and time of performance
- Ticket availability and price range
- Reference URL from an official source

Performance 4 - Mid-Sized Music Venue:
Find a music performance at a venue with a capacity between 1,000 and 3,000 people. Provide:
- Artist/band name
- Venue name and address
- Venue capacity
- Date and time of performance
- Ticket availability and price range
- Reference URL from an official source

All performances must be scheduled between April 1-30, 2026, and must be verifiable through official venue websites, ticketing platforms, or artist tour announcements.
"""

APRIL_2026_WINDOW_TEXT = "between April 1 and April 30, 2026 (inclusive)"
NYC_BOROUGHS_HINT = "New York, NY or a NYC borough (Manhattan, Brooklyn, Queens, The Bronx, Staten Island)"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PerformanceLargeConcert(BaseModel):
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    venue_capacity: Optional[str] = None  # Keep as free text; do not coerce to number
    date_time: Optional[str] = None
    artist_names: List[str] = Field(default_factory=list)
    ticket_availability: Optional[str] = None
    ticket_price_range: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class PerformanceBroadway(BaseModel):
    show_name: Optional[str] = None
    theater_name: Optional[str] = None
    theater_address: Optional[str] = None
    dates: Optional[str] = None  # Free text summary or a specific date string
    ticket_availability: Optional[str] = None
    ticket_price_range: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class PerformanceComedy(BaseModel):
    comedian_names: List[str] = Field(default_factory=list)
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    date_time: Optional[str] = None
    ticket_availability: Optional[str] = None
    ticket_price_range: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class PerformanceMidSized(BaseModel):
    artist_names: List[str] = Field(default_factory=list)
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    venue_capacity: Optional[str] = None  # Keep as free text
    date_time: Optional[str] = None
    ticket_availability: Optional[str] = None
    ticket_price_range: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class LivePerformancesExtraction(BaseModel):
    performance_1_large_concert: Optional[PerformanceLargeConcert] = None
    performance_2_broadway_show: Optional[PerformanceBroadway] = None
    performance_3_comedy_show: Optional[PerformanceComedy] = None
    performance_4_mid_sized_venue: Optional[PerformanceMidSized] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_live_performances() -> str:
    return f"""
    Extract structured information for four performances mentioned in the answer. Return null for any missing field; do not fabricate.

    General rules:
    - Extract exactly what appears in the answer.
    - For all dates/times, keep as a single free-text string exactly as shown.
    - For capacities and prices, keep free-text (e.g., "~5,600", "from $49", "$50–$120").
    - For URLs, extract all explicitly listed links relevant to that performance (venue site, primary ticketing, official artist/show sites).

    Performance 1 - Large Concert Venue (capacity ≥ 5,000):
      - venue_name (string)
      - venue_address (string)
      - venue_capacity (string, free-text)
      - date_time (string)
      - artist_names (array of strings)
      - ticket_availability (string)
      - ticket_price_range (string)
      - reference_urls (array of URLs)

    Performance 2 - Broadway Show:
      - show_name (string)
      - theater_name (string)
      - theater_address (string)
      - dates (string; any representation of April 2026 availability)
      - ticket_availability (string)
      - ticket_price_range (string)
      - reference_urls (array of URLs)

    Performance 3 - Stand-Up Comedy:
      - comedian_names (array of strings)
      - venue_name (string)
      - venue_address (string)
      - date_time (string)
      - ticket_availability (string)
      - ticket_price_range (string)
      - reference_urls (array of URLs)

    Performance 4 - Mid-Sized Music Venue (capacity 1,000–3,000):
      - artist_names (array of strings)
      - venue_name (string)
      - venue_address (string)
      - venue_capacity (string, free-text)
      - date_time (string)
      - ticket_availability (string)
      - ticket_price_range (string)
      - reference_urls (array of URLs)

    Output JSON fields:
      - performance_1_large_concert: object or null
      - performance_2_broadway_show: object or null
      - performance_3_comedy_show: object or null
      - performance_4_mid_sized_venue: object or null
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _names_to_str(names: Optional[List[str]]) -> str:
    return ", ".join(n for n in (names or []) if n) if names else ""


def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and len(u.strip()) > 0]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_performance_1_large_concert(
    evaluator: Evaluator,
    parent_node,
    p1: Optional[PerformanceLargeConcert],
) -> None:
    node = evaluator.add_parallel(
        id="performance_1_large_concert",
        desc="Identify a concert or music performance at a large venue (capacity ≥5,000) in NYC during April 2026",
        parent=parent_node,
        critical=False,
    )

    venue_name = (p1.venue_name if p1 else "") or ""
    venue_address = (p1.venue_address if p1 else "") or ""
    venue_capacity = (p1.venue_capacity if p1 else "") or ""
    date_time = (p1.date_time if p1 else "") or ""
    artist_str = _names_to_str(p1.artist_names if p1 else [])
    ticket_availability = (p1.ticket_availability if p1 else "") or ""
    ticket_price_range = (p1.ticket_price_range if p1 else "") or ""
    sources = _safe_urls(p1.reference_urls if p1 else [])

    # p1_venue_identification
    leaf = evaluator.add_leaf(
        id="p1_venue_identification",
        desc="Provide the name and address of a specific venue in New York City",
        parent=node,
        critical=True,
    )
    claim = (
        f"The venue is '{venue_name}' with address '{venue_address}', and the address is in {NYC_BOROUGHS_HINT}."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm the venue name and full address, and that the address is in New York City (or a NYC borough). Use the event or venue page.",
    )

    # p1_venue_capacity
    leaf = evaluator.add_leaf(
        id="p1_venue_capacity",
        desc="Verify that the venue has a capacity of at least 5,000 people for concerts/music events",
        parent=node,
        critical=True,
    )
    claim = "This venue's concert/event capacity is at least 5,000 attendees."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Look for capacity figures on the venue's official site, Wikipedia infobox, or reliable sources linked in the provided URLs. Approximate values (e.g., ~5,600) are acceptable if ≥ 5,000.",
    )

    # p1_date_verification
    leaf = evaluator.add_leaf(
        id="p1_date_verification",
        desc=f"Verify that the performance is scheduled during April 2026 (April 1-30, 2026)",
        parent=node,
        critical=True,
    )
    claim = f"The performance date/time '{date_time}' falls {APRIL_2026_WINDOW_TEXT}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Verify the event listing shows at least one date in April 2026. Minor formatting differences are acceptable.",
    )

    # p1_artist_identification
    leaf = evaluator.add_leaf(
        id="p1_artist_identification",
        desc="Provide the name of the performing artist(s) or musical act",
        parent=node,
        critical=True,
    )
    claim = f"The performing artist(s) for this event are: {artist_str}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm the named artist(s) are listed as the performers for the cited event.",
    )

    # p1_ticket_information
    leaf = evaluator.add_leaf(
        id="p1_ticket_information",
        desc="Provide ticket availability status and price range information",
        parent=node,
        critical=True,
    )
    claim = (
        f"Tickets are available for this performance, and the page shows a ticket price or range consistent with '{ticket_price_range}'."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm the page indicates tickets are available (or on sale) and shows a price or price range roughly matching the provided text (allowing minor variations or fees).",
    )

    # p1_reference_url
    leaf = evaluator.add_leaf(
        id="p1_reference_url",
        desc="Provide a reference URL from an official source (venue website, ticketing platform, or artist tour page) confirming the performance details",
        parent=node,
        critical=True,
    )
    claim = (
        "This URL is an official or primary listing confirming the performance details (e.g., venue's own site, "
        "artist's official tour page, or a primary ticketing platform like Ticketmaster/AXS/Telecharge)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Pass if at least one provided URL is an official venue page, the artist's official site/tour page, or a primary ticketing platform listing for this exact event.",
    )


async def verify_performance_2_broadway(
    evaluator: Evaluator,
    parent_node,
    p2: Optional[PerformanceBroadway],
) -> None:
    node = evaluator.add_parallel(
        id="performance_2_broadway_show",
        desc="Identify a Broadway theatrical production in NYC during April 2026",
        parent=parent_node,
        critical=False,
    )

    show_name = (p2.show_name if p2 else "") or ""
    theater_name = (p2.theater_name if p2 else "") or ""
    theater_address = (p2.theater_address if p2 else "") or ""
    dates = (p2.dates if p2 else "") or ""
    ticket_availability = (p2.ticket_availability if p2 else "") or ""
    ticket_price_range = (p2.ticket_price_range if p2 else "") or ""
    sources = _safe_urls(p2.reference_urls if p2 else [])

    # p2_show_identification
    leaf = evaluator.add_leaf(
        id="p2_show_identification",
        desc="Provide the name of a Broadway show/musical",
        parent=node,
        critical=True,
    )
    claim = f"The show/musical is titled '{show_name}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm the page clearly identifies the production by this title.",
    )

    # p2_venue_identification
    leaf = evaluator.add_leaf(
        id="p2_venue_identification",
        desc="Provide the name and address of the Broadway theater where the show is performed",
        parent=node,
        critical=True,
    )
    claim = f"The show '{show_name}' is performed at '{theater_name}' with address '{theater_address}' in New York City."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=f"Confirm the theater name and address, ensuring it's in {NYC_BOROUGHS_HINT}.",
    )

    # p2_broadway_verification
    leaf = evaluator.add_leaf(
        id="p2_broadway_verification",
        desc="Verify that the theater is an official Broadway theater (minimum 500 seats, located in the Theater District between 41st-54th Streets and 6th-8th Avenues)",
        parent=node,
        critical=True,
    )
    claim = (
        "This theater is an official Broadway theater. Evidence may include explicit designation on an authoritative site "
        "(e.g., Broadway League, Playbill list of Broadway theatres, the theater's official page stating 'Broadway'), "
        "or details showing ≥500 seats and a location within the Theater District (41st–54th Streets, 6th–8th Avenues)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Accept explicit 'Broadway theatre' designation from trusted sources. If designation is missing, verify seat count ≥500 and location within the Theater District boundaries as sufficient.",
    )

    # p2_date_verification
    leaf = evaluator.add_leaf(
        id="p2_date_verification",
        desc="Verify that the show has performances scheduled during April 2026 (April 1-30, 2026)",
        parent=node,
        critical=True,
    )
    claim = f"There is at least one performance date for this show in April 2026 {APRIL_2026_WINDOW_TEXT}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Check the schedule/calendar or ticketing for April 2026 dates.",
    )

    # p2_ticket_information
    leaf = evaluator.add_leaf(
        id="p2_ticket_information",
        desc="Provide ticket availability status and price range information",
        parent=node,
        critical=True,
    )
    claim = (
        f"Tickets are available for this Broadway show, and the page shows a ticket price or range consistent with '{ticket_price_range}'."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm availability and some price/price-range for tickets on an official/primary page.",
    )

    # p2_reference_url
    leaf = evaluator.add_leaf(
        id="p2_reference_url",
        desc="Provide a reference URL from an official source (Broadway.com, theater website, or ticketing platform) confirming the show details",
        parent=node,
        critical=True,
    )
    claim = (
        "This URL is an official or primary source for the Broadway production (e.g., the theater's site, Broadway League/Broadway.com listing, "
        "or primary ticketing like Telecharge/Ticketmaster)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Pass if at least one provided URL is an official/primary page confirming the show and venue.",
    )


async def verify_performance_3_comedy(
    evaluator: Evaluator,
    parent_node,
    p3: Optional[PerformanceComedy],
) -> None:
    node = evaluator.add_parallel(
        id="performance_3_comedy_show",
        desc="Identify a stand-up comedy show or performance in NYC during April 2026",
        parent=parent_node,
        critical=False,
    )

    comedian_str = _names_to_str(p3.comedian_names if p3 else [])
    venue_name = (p3.venue_name if p3 else "") or ""
    venue_address = (p3.venue_address if p3 else "") or ""
    date_time = (p3.date_time if p3 else "") or ""
    ticket_availability = (p3.ticket_availability if p3 else "") or ""
    ticket_price_range = (p3.ticket_price_range if p3 else "") or ""
    sources = _safe_urls(p3.reference_urls if p3 else [])

    # p3_comedian_identification
    leaf = evaluator.add_leaf(
        id="p3_comedian_identification",
        desc="Provide the name of the comedian(s) performing",
        parent=node,
        critical=True,
    )
    claim = f"The comedian(s) performing are: {comedian_str}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm the named comedian(s) are listed as performers on the cited page.",
    )

    # p3_venue_identification
    leaf = evaluator.add_leaf(
        id="p3_venue_identification",
        desc="Provide the name and address of the venue in New York City",
        parent=node,
        critical=True,
    )
    claim = f"The show takes place at '{venue_name}' with address '{venue_address}' in {NYC_BOROUGHS_HINT}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm venue name and NYC address on the event/venue page.",
    )

    # p3_venue_type
    leaf = evaluator.add_leaf(
        id="p3_venue_type",
        desc="Verify that the venue is a comedy club, theater, or venue hosting stand-up comedy performances",
        parent=node,
        critical=True,
    )
    claim = "This venue is a comedy club or a theater/venue that regularly hosts stand-up comedy performances."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Look for explicit mention of comedy programming, stand-up shows, or that the venue is known as a comedy club.",
    )

    # p3_date_verification
    leaf = evaluator.add_leaf(
        id="p3_date_verification",
        desc="Verify that the performance is scheduled during April 2026 (April 1-30, 2026)",
        parent=node,
        critical=True,
    )
    claim = f"The performance date/time '{date_time}' is {APRIL_2026_WINDOW_TEXT}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm an April 2026 date is shown on the event/ticket listing.",
    )

    # p3_ticket_information
    leaf = evaluator.add_leaf(
        id="p3_ticket_information",
        desc="Provide ticket availability status and price range information",
        parent=node,
        critical=True,
    )
    claim = (
        f"Tickets are available for this comedy performance, and a ticket price or range consistent with '{ticket_price_range}' is shown."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm availability and some price/price-range for tickets; approximate matches are acceptable.",
    )

    # p3_reference_url
    leaf = evaluator.add_leaf(
        id="p3_reference_url",
        desc="Provide a reference URL from an official source (venue website, ticketing platform, or comedian's tour page) confirming the performance details",
        parent=node,
        critical=True,
    )
    claim = (
        "This URL is an official or primary listing (venue site, comedian's official tour page, or a primary ticketing platform) confirming the comedy show details."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Pass if at least one URL qualifies as official/primary for the specific show.",
    )


async def verify_performance_4_mid_sized(
    evaluator: Evaluator,
    parent_node,
    p4: Optional[PerformanceMidSized],
) -> None:
    node = evaluator.add_parallel(
        id="performance_4_mid_sized_venue",
        desc="Identify a music performance at a mid-sized venue (capacity 1,000-3,000) in NYC during April 2026",
        parent=parent_node,
        critical=False,
    )

    artist_str = _names_to_str(p4.artist_names if p4 else [])
    venue_name = (p4.venue_name if p4 else "") or ""
    venue_address = (p4.venue_address if p4 else "") or ""
    venue_capacity = (p4.venue_capacity if p4 else "") or ""
    date_time = (p4.date_time if p4 else "") or ""
    ticket_availability = (p4.ticket_availability if p4 else "") or ""
    ticket_price_range = (p4.ticket_price_range if p4 else "") or ""
    sources = _safe_urls(p4.reference_urls if p4 else [])

    # p4_artist_identification
    leaf = evaluator.add_leaf(
        id="p4_artist_identification",
        desc="Provide the name of the performing artist(s) or musical act",
        parent=node,
        critical=True,
    )
    claim = f"The performing artist(s) for this event are: {artist_str}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm the named artist(s) are listed as performers for the cited event.",
    )

    # p4_venue_identification
    leaf = evaluator.add_leaf(
        id="p4_venue_identification",
        desc="Provide the name and address of a specific venue in New York City",
        parent=node,
        critical=True,
    )
    claim = f"The venue is '{venue_name}' with address '{venue_address}', and it is in {NYC_BOROUGHS_HINT}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm venue name and NYC address on the event/venue page.",
    )

    # p4_venue_capacity
    leaf = evaluator.add_leaf(
        id="p4_venue_capacity",
        desc="Verify that the venue has a capacity between 1,000 and 3,000 people",
        parent=node,
        critical=True,
    )
    claim = "This venue's concert/event capacity is between 1,000 and 3,000 (inclusive)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Look for capacity figures on reliable sources. Accept approximate figures if clearly within 1,000–3,000.",
    )

    # p4_date_verification
    leaf = evaluator.add_leaf(
        id="p4_date_verification",
        desc="Verify that the performance is scheduled during April 2026 (April 1-30, 2026)",
        parent=node,
        critical=True,
    )
    claim = f"The performance date/time '{date_time}' is {APRIL_2026_WINDOW_TEXT}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Verify the event listing shows an April 2026 date.",
    )

    # p4_ticket_information
    leaf = evaluator.add_leaf(
        id="p4_ticket_information",
        desc="Provide ticket availability status and price range information",
        parent=node,
        critical=True,
    )
    claim = (
        f"Tickets are available for this performance, and the page shows a ticket price or range consistent with '{ticket_price_range}'."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm availability and a price/price-range; minor differences or fees are acceptable.",
    )

    # p4_reference_url
    leaf = evaluator.add_leaf(
        id="p4_reference_url",
        desc="Provide a reference URL from an official source (venue website, ticketing platform, or artist tour page) confirming the performance details",
        parent=node,
        critical=True,
    )
    claim = (
        "This URL is an official or primary listing confirming the performance details (venue site, artist's official tour page, or a primary ticketing platform)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Pass if at least one URL qualifies as official/primary for the specific event.",
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
    Evaluate an answer for four NYC live performances in April 2026.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent categories
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
    # IMPORTANT: Keep root non-critical to allow partial credit across categories
    root.critical = False

    # 1) Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_live_performances(),
        template_class=LivePerformancesExtraction,
        extraction_name="performances_extraction",
    )

    # 2) Build verification subtrees for each performance
    await verify_performance_1_large_concert(evaluator, root, extracted.performance_1_large_concert)
    await verify_performance_2_broadway(evaluator, root, extracted.performance_2_broadway_show)
    await verify_performance_3_comedy(evaluator, root, extracted.performance_3_comedy_show)
    await verify_performance_4_mid_sized(evaluator, root, extracted.performance_4_mid_sized_venue)

    # Optional: add custom info for transparency
    evaluator.add_custom_info(
        {
            "required_time_window": APRIL_2026_WINDOW_TEXT,
            "nyc_location_hint": NYC_BOROUGHS_HINT,
        },
        info_type="hints",
        info_name="evaluation_hints",
    )

    # 3) Return the standard evaluation summary
    return evaluator.get_summary()