import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "live_events_east_2026_spring"
TASK_DESCRIPTION = """
Identify four live comedy or music performances taking place in the Eastern United States (states east of the Mississippi River) between March 1 and May 31, 2026. For each event, provide the following information:

1. Performance Details: The name of the performing artist or comedian, the specific date of the performance, and the door opening time and/or performance start time.

2. Venue Information: The official name of the venue, the complete physical address (including street address, city, state, and ZIP code), and confirmation that the venue is located in a state east of the Mississippi River.

3. Accessibility Compliance: Information confirming that the venue complies with ADA (Americans with Disabilities Act) accessibility requirements, and details about the availability of wheelchair-accessible seating and companion seats.

4. Ticketing Information: The name of the official ticket vendor or purchase location (such as the artist's website, Ticketmaster, or the venue box office), a direct URL to the ticket purchase page, and the current ticket availability status (available, limited tickets remaining, or sold out).

For each piece of information, include a URL reference that can be used to verify the details you provide.
"""


# --------------------------------------------------------------------------- #
# Utility: Eastern US state validation                                        #
# --------------------------------------------------------------------------- #
EAST_STATE_ABBR = {
    "AL", "CT", "DE", "FL", "GA", "IL", "IN", "KY", "ME", "MD", "MA", "MI",
    "MS", "NH", "NJ", "NY", "NC", "OH", "PA", "RI", "SC", "TN", "VT", "VA",
    "WV", "WI", "DC"
}
EAST_STATE_NAMES = {
    "ALABAMA": "AL",
    "CONNECTICUT": "CT",
    "DELAWARE": "DE",
    "FLORIDA": "FL",
    "GEORGIA": "GA",
    "ILLINOIS": "IL",
    "INDIANA": "IN",
    "KENTUCKY": "KY",
    "MAINE": "ME",
    "MARYLAND": "MD",
    "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI",
    "MISSISSIPPI": "MS",
    "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ",
    "NEW YORK": "NY",
    "NORTH CAROLINA": "NC",
    "OHIO": "OH",
    "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC",
    "TENNESSEE": "TN",
    "VERMONT": "VT",
    "VIRGINIA": "VA",
    "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI",
    "DISTRICT OF COLUMBIA": "DC",
    "WASHINGTON, DC": "DC",
    "WASHINGTON DC": "DC",
    "DC": "DC",
    "D.C.": "DC",
}

def is_eastern_state(state_str: Optional[str]) -> bool:
    if not state_str:
        return False
    s = state_str.strip().upper().replace(".", "")
    if s in EAST_STATE_ABBR:
        return True
    return EAST_STATE_NAMES.get(s, None) in EAST_STATE_ABBR


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    # Performance details
    performer_name: Optional[str] = None
    event_date: Optional[str] = None  # Keep as string to allow various formats
    doors_time: Optional[str] = None
    show_time: Optional[str] = None
    performance_detail_urls: List[str] = Field(default_factory=list)

    # Venue information
    venue_name: Optional[str] = None
    venue_street: Optional[str] = None
    venue_city: Optional[str] = None
    venue_state: Optional[str] = None
    venue_zip: Optional[str] = None
    venue_info_urls: List[str] = Field(default_factory=list)

    # Accessibility
    ada_compliance_text: Optional[str] = None
    accessible_seating_text: Optional[str] = None
    accessibility_urls: List[str] = Field(default_factory=list)

    # Ticketing
    ticket_source_name: Optional[str] = None
    ticket_purchase_url: Optional[str] = None
    ticket_availability_status: Optional[str] = None  # available | limited | sold out | waitlist | unknown
    ticket_info_urls: List[str] = Field(default_factory=list)


class EventsExtraction(BaseModel):
    events: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract up to 6 (at least 4 if available) live events from the answer. Each event should be a comedy or music performance scheduled between March 1, 2026 and May 31, 2026 in a U.S. state east of the Mississippi River.
    
    For each event, extract these fields exactly as stated in the answer:
    - performer_name: The main performing artist or comedian.
    - event_date: The specific date of the performance (accept any natural format, e.g., "March 10, 2026", "2026-03-10", "Fri, May 15, 2026").
    - doors_time: Door opening time if provided (string, any format). Use null if not provided.
    - show_time: Performance start time if provided (string, any format). Use null if not provided.
    - performance_detail_urls: All URLs cited that directly support the performance details (event date/time/artist). Return as an array; empty array if none.

    - venue_name: Official venue name (string).
    - venue_street: Street number and street name (e.g., "123 Main St"). If not explicitly present, set null.
    - venue_city: City name (string).
    - venue_state: State abbreviation or full name (e.g., "NY" or "New York").
    - venue_zip: ZIP code (5-digit or ZIP+4 string). If not present, set null.
    - venue_info_urls: URLs that support the venue info/address. Return as an array; can reuse event page; empty array if none.

    - ada_compliance_text: The text snippet or summary indicating ADA compliance. If not present, set null.
    - accessible_seating_text: The text snippet or summary indicating wheelchair-accessible seating and companion seats availability. If not present, set null.
    - accessibility_urls: URLs cited that support ADA/accessibility info (venue policy page, event page, or ticket policy). Return as an array; empty if none.

    - ticket_source_name: The official ticket vendor or purchase location name (e.g., "Ticketmaster", "Venue Box Office", "AXS", "Etix", artist site).
    - ticket_purchase_url: Direct URL for buying tickets (not a generic home page). If not provided, set null.
    - ticket_availability_status: One of ["available", "limited", "sold out", "waitlist", "unknown"]. Normalize obvious synonyms (e.g., "on sale" -> "available", "few left" -> "limited"). If unclear, use "unknown".
    - ticket_info_urls: Additional URLs (if any) that support ticketing details. Return as an array; can be empty.

    Rules:
    - Only extract information present in the provided answer. Do not invent URLs or details.
    - Always include full URLs with protocol.
    - If multiple URLs are provided for the same field, include all of them; otherwise, return an empty array when missing.
    - If the answer includes more than 4 qualifying events, extract them all; evaluation will use the first 4.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def first_n_events(extracted: EventsExtraction, n: int = 4) -> List[EventItem]:
    items = list(extracted.events or [])
    if len(items) >= n:
        return items[:n]
    # Pad with empty items
    pad = [EventItem() for _ in range(n - len(items))]
    return items + pad


def _non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip() != ""]


def _ticket_sources(item: EventItem) -> List[str]:
    urls = []
    if item.ticket_purchase_url and item.ticket_purchase_url.strip():
        urls.append(item.ticket_purchase_url.strip())
    urls.extend(_non_empty_urls(item.ticket_info_urls))
    return urls


def _venue_sources(item: EventItem) -> List[str]:
    # Prefer specific venue URLs, fall back to performance details
    v = _non_empty_urls(item.venue_info_urls)
    if v:
        return v
    return _non_empty_urls(item.performance_detail_urls)


def _access_sources(item: EventItem) -> List[str]:
    a = _non_empty_urls(item.accessibility_urls)
    if a:
        return a
    # Fallbacks if accessibility URLs missing
    v = _venue_sources(item)
    if v:
        return v
    return _ticket_sources(item)


def _perf_sources(item: EventItem) -> List[str]:
    return _non_empty_urls(item.performance_detail_urls)


def _time_display(item: EventItem) -> str:
    if item.show_time and item.show_time.strip():
        return item.show_time.strip()
    if item.doors_time and item.doors_time.strip():
        return item.doors_time.strip()
    return ""


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_performance_details(evaluator: Evaluator, parent, item: EventItem, idx: int) -> None:
    node = evaluator.add_parallel(
        id=f"event_{idx+1}_performance_details",
        desc="Complete performance information provided including artist/performer name, specific date, and showtime",
        parent=parent,
        critical=True
    )

    # Reference URL existence (critical prerequisite)
    perf_urls = _perf_sources(item)
    evaluator.add_custom_node(
        result=len(perf_urls) > 0,
        id=f"event_{idx+1}_performance_details_reference",
        desc="URL reference provided to verify performance details",
        parent=node,
        critical=True
    )

    # Artist identified
    leaf_artist = evaluator.add_leaf(
        id=f"event_{idx+1}_artist_identified",
        desc="Performance artist or comedian name is clearly identified",
        parent=node,
        critical=True
    )
    artist_name = item.performer_name or ""
    await evaluator.verify(
        claim=f"The performing artist/comedian for this event is '{artist_name}'.",
        node=leaf_artist,
        sources=perf_urls,
        additional_instruction="Confirm on the referenced event page that the named performer is the billed act."
    )

    # Event date valid and supported
    leaf_date = evaluator.add_leaf(
        id=f"event_{idx+1}_event_date_valid",
        desc="Performance date falls within March 1 - May 31, 2026",
        parent=node,
        critical=True
    )
    date_text = item.event_date or ""
    await evaluator.verify(
        claim=f"The event date is '{date_text}', and it occurs between March 1, 2026 and May 31, 2026 (inclusive).",
        node=leaf_date,
        sources=perf_urls,
        additional_instruction="Verify the date shown on the page matches the stated date and falls within the specified 2026 window."
    )

    # Show time provided
    leaf_time = evaluator.add_leaf(
        id=f"event_{idx+1}_show_time_provided",
        desc="Door opening time and/or performance start time is specified",
        parent=node,
        critical=True
    )
    time_text = _time_display(item)
    await evaluator.verify(
        claim=f"The event page lists a door opening time or a performance start time: '{time_text}'.",
        node=leaf_time,
        sources=perf_urls,
        additional_instruction="The page must present a time. Either a door time or a show start time counts."
    )


async def verify_venue_information(evaluator: Evaluator, parent, item: EventItem, idx: int) -> None:
    node = evaluator.add_parallel(
        id=f"event_{idx+1}_venue_information",
        desc="Complete venue details including name, full address, and venue type confirmation",
        parent=parent,
        critical=True
    )

    venue_sources = _venue_sources(item)
    # Reference URL existence (critical prerequisite)
    evaluator.add_custom_node(
        result=len(venue_sources) > 0,
        id=f"event_{idx+1}_venue_information_reference",
        desc="URL reference provided to verify venue information",
        parent=node,
        critical=True
    )

    # Venue name
    leaf_vname = evaluator.add_leaf(
        id=f"event_{idx+1}_venue_name_provided",
        desc="Official venue name is clearly stated",
        parent=node,
        critical=True
    )
    venue_name = (item.venue_name or "").strip()
    await evaluator.verify(
        claim=f"The official venue name is '{venue_name}'.",
        node=leaf_vname,
        sources=venue_sources,
        additional_instruction="Confirm that the page explicitly names the venue with the provided name (allowing minor formatting differences)."
    )

    # Complete address (nested parallel node)
    addr_node = evaluator.add_parallel(
        id=f"event_{idx+1}_complete_physical_address",
        desc="Full venue address including street address, city, state, and ZIP code",
        parent=node,
        critical=True
    )

    # Street address
    leaf_street = evaluator.add_leaf(
        id=f"event_{idx+1}_street_address",
        desc="Street number and street name provided",
        parent=addr_node,
        critical=True
    )
    street_text = (item.venue_street or "").strip()
    await evaluator.verify(
        claim=f"The venue address includes the street address '{street_text}'.",
        node=leaf_street,
        sources=venue_sources,
        additional_instruction="Check the page shows the street number and street name; allow minor formatting differences (e.g., St vs Street)."
    )

    # City, state, zip
    leaf_csz = evaluator.add_leaf(
        id=f"event_{idx+1}_city_state_zip",
        desc="City, state, and ZIP code provided",
        parent=addr_node,
        critical=True
    )
    city = (item.venue_city or "").strip()
    state = (item.venue_state or "").strip()
    zipc = (item.venue_zip or "").strip()
    await evaluator.verify(
        claim=f"The venue address includes city '{city}', state '{state}', and ZIP code '{zipc}'.",
        node=leaf_csz,
        sources=venue_sources,
        additional_instruction="Confirm all three appear on the page; minor formatting differences are acceptable."
    )

    # Geographic location validity (east of Mississippi) - custom factual check
    evaluator.add_custom_node(
        result=is_eastern_state(item.venue_state),
        id=f"event_{idx+1}_geographic_location_valid",
        desc="Venue is located in a state east of the Mississippi River",
        parent=node,
        critical=True
    )


async def verify_accessibility(evaluator: Evaluator, parent, item: EventItem, idx: int) -> None:
    node = evaluator.add_parallel(
        id=f"event_{idx+1}_accessibility_compliance",
        desc="Venue accessibility information documented including ADA compliance confirmation and accessible seating availability",
        parent=parent,
        critical=True
    )

    access_sources = _access_sources(item)

    # Accessibility reference existence (critical prerequisite)
    # Prefer explicit accessibility_urls; if not, we accept venue/ticket sources as fallback references.
    evaluator.add_custom_node(
        result=len(access_sources) > 0,
        id=f"event_{idx+1}_accessibility_reference",
        desc="URL reference provided to verify accessibility information",
        parent=node,
        critical=True
    )

    # ADA compliance confirmation
    leaf_ada = evaluator.add_leaf(
        id=f"event_{idx+1}_ada_compliance_status",
        desc="Information provided confirming venue complies with ADA accessibility requirements",
        parent=node,
        critical=True
    )
    ada_text = (item.ada_compliance_text or "").strip()
    await evaluator.verify(
        claim=f"The venue states that it complies with ADA (Americans with Disabilities Act) accessibility requirements. Evidence: '{ada_text}'.",
        node=leaf_ada,
        sources=access_sources,
        additional_instruction="Look for accessibility/ADA policy language (e.g., ADA-compliant, accessible facilities)."
    )

    # Accessible seating and companion seats
    leaf_seats = evaluator.add_leaf(
        id=f"event_{idx+1}_accessible_seating_information",
        desc="Information provided about availability of wheelchair-accessible seating and companion seats",
        parent=node,
        critical=True
    )
    acc_text = (item.accessible_seating_text or "").strip()
    await evaluator.verify(
        claim=f"The venue provides wheelchair-accessible seating and companion seats for events. Evidence: '{acc_text}'.",
        node=leaf_seats,
        sources=access_sources,
        additional_instruction="Confirm that wheelchair spaces and at least one adjacent companion seat availability are mentioned."
    )


async def verify_ticketing(evaluator: Evaluator, parent, item: EventItem, idx: int) -> None:
    node = evaluator.add_parallel(
        id=f"event_{idx+1}_ticketing_information",
        desc="Official ticket purchasing source identified with availability confirmation",
        parent=parent,
        critical=True
    )

    t_sources = _ticket_sources(item)

    # Official Ticket Source (nested parallel)
    src_node = evaluator.add_parallel(
        id=f"event_{idx+1}_official_ticket_source",
        desc="Official ticket vendor or purchase location identified (e.g., artist website, Ticketmaster, venue box office)",
        parent=node,
        critical=True
    )

    # Ticket source name verified by purchase page
    leaf_src_name = evaluator.add_leaf(
        id=f"event_{idx+1}_ticket_source_name",
        desc="Name of official ticket vendor provided",
        parent=src_node,
        critical=True
    )
    src_name = (item.ticket_source_name or "").strip()
    await evaluator.verify(
        claim=f"The official ticket vendor/purchase location for this event is '{src_name}'.",
        node=leaf_src_name,
        sources=t_sources,
        additional_instruction="Confirm from the purchase page or vendor-branded page. Consider domain/branding (e.g., ticketmaster.com => 'Ticketmaster')."
    )

    # Ticket purchase URL provided (existence)
    evaluator.add_custom_node(
        result=bool(item.ticket_purchase_url and item.ticket_purchase_url.strip()),
        id=f"event_{idx+1}_ticket_purchase_url",
        desc="Direct URL to ticket purchase page provided",
        parent=src_node,
        critical=True
    )

    # Ticket availability status verified
    leaf_status = evaluator.add_leaf(
        id=f"event_{idx+1}_ticket_availability_status",
        desc="Current ticket availability status confirmed (available, limited, sold out)",
        parent=node,
        critical=True
    )
    status_text = (item.ticket_availability_status or "").strip()
    await evaluator.verify(
        claim=f"The ticket availability status on the purchase page is '{status_text}' (allowing reasonable synonyms).",
        node=leaf_status,
        sources=t_sources,
        additional_instruction=(
            "Interpret synonyms: 'on sale'/'buy tickets' => available; 'low tickets'/'few left' => limited; "
            "'sold out'/'no tickets available' => sold out; 'waitlist' counts as not available. "
            "Match the stated status if clearly indicated."
        )
    )


async def verify_event(evaluator: Evaluator, root, item: EventItem, idx: int) -> None:
    event_node = evaluator.add_parallel(
        id=f"event_{idx+1}",
        desc=f"Event #{idx+1} meets all specified requirements with complete and accurate information",
        parent=root,
        critical=False
    )
    await verify_performance_details(evaluator, event_node, item, idx)
    await verify_venue_information(evaluator, event_node, item, idx)
    await verify_accessibility(evaluator, event_node, item, idx)
    await verify_ticketing(evaluator, event_node, item, idx)


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
) -> Dict:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel across events
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
    # Make sure root is non-critical to allow partial credit aggregation across events
    root.critical = False

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction"
    )

    # Normalize: take first 4 events, padding if needed
    events = first_n_events(extracted, n=4)
    evaluator.add_custom_info(
        info={"total_events_extracted": len(extracted.events), "evaluated_events": 4},
        info_type="extraction_stats",
        info_name="extraction_stats"
    )

    # Build verification subtrees
    for i, item in enumerate(events):
        await verify_event(evaluator, root, item, i)

    return evaluator.get_summary()