import asyncio
import logging
from typing import Any, List, Optional, Dict, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sf_bay_area_events_2026_q2"
TASK_DESCRIPTION = """
Find 5 ticketed entertainment events in the San Francisco Bay Area (including San Francisco, Oakland, San Jose, and surrounding cities) scheduled between April 1, 2026 and June 30, 2026. The events must collectively represent at least 3 different categories from the following: concerts/music performances, professional sports games, theater/performing arts shows, comedy shows, opera performances, ballet performances, or symphony orchestra concerts.

For each event, provide:
- Event name and performer/team
- Venue name and complete physical address
- Event date and start time
- A link to purchase tickets from an official source (venue website, Ticketmaster, SeatGeek, Live Nation, or other legitimate ticket vendor)
- Venue seating capacity (for at least 4 of the 5 events)

Ensure that all 5 events are distinct (different dates, venues, or performers/teams).
"""

ALLOWED_CATEGORIES = [
    "concerts/music performances",
    "professional sports games",
    "theater/performing arts shows",
    "comedy shows",
    "opera performances",
    "ballet performances",
    "symphony orchestra concerts",
]

# Normalized category keys for internal checks
CATEGORY_KEYS = {
    "concert": "concerts/music performances",
    "music": "concerts/music performances",
    "professional sports": "professional sports games",
    "sports": "professional sports games",
    "nba": "professional sports games",
    "mlb": "professional sports games",
    "nhl": "professional sports games",
    "nfl": "professional sports games",
    "mls": "professional sports games",
    "theater": "theater/performing arts shows",
    "theatre": "theater/performing arts shows",
    "performing arts": "theater/performing arts shows",
    "musical": "theater/performing arts shows",
    "comedy": "comedy shows",
    "stand-up": "comedy shows",
    "stand up": "comedy shows",
    "opera": "opera performances",
    "ballet": "ballet performances",
    "symphony": "symphony orchestra concerts",
    "orchestra": "symphony orchestra concerts",
    "philharmonic": "symphony orchestra concerts",
}

DATE_WINDOW_TEXT = "between April 1, 2026 and June 30, 2026 (inclusive)"

OFFICIAL_VENDOR_INSTRUCTION = """
Treat a ticket purchase URL as official/legitimate if it is one of:
- An official venue/operator site (e.g., chasecenter.com, sapcenter.com, oaklandarena.com, broadwaysf.com, etc.)
- Major primary ticketing platforms: ticketmaster.com, livenation.com or tickets.livenation.com, axs.com, seatgeek.com, tickets.com, fevo.com, universe.com, eventbrite.com
- Official team/league sites that link to ticketing (e.g., mlb.com, nba.com, nhl.com, mls.com, and official team domains like sfgiants.com)
Use the page content to confirm that it is the purchase page for the exact event (date/time and performer/team) and offers "Buy Tickets" or equivalent.
Minor URL query parameters are irrelevant. Do NOT consider resale marketplaces that don't clearly represent official primary sales as official unless the page itself indicates it is the official seller for this event.
"""

BAY_AREA_INSTRUCTION = """
Confirm the event venue is in the San Francisco Bay Area. The Bay Area generally includes these counties and cities:
- Counties: San Francisco, San Mateo, Santa Clara, Alameda, Contra Costa, Marin, Sonoma, Napa, Solano.
- Cities (examples): San Francisco, Oakland, San Jose, Berkeley, Daly City, South San Francisco, San Mateo, Redwood City, Menlo Park, Palo Alto, Mountain View, Sunnyvale, Cupertino, Santa Clara, Milpitas, Fremont, Hayward, San Leandro, Richmond, Concord, Walnut Creek, San Rafael, Petaluma, Santa Rosa, Napa, Vallejo, Fairfield.
A venue/address located in any of the above counties/cities should be considered in the SF Bay Area.
Allow minor formatting differences in addresses.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    event_name: Optional[str] = None
    performer_or_team: Optional[str] = None
    category: Optional[str] = None  # Free text; will be normalized
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None  # Expect full street address incl. city/state
    city: Optional[str] = None
    event_date: Optional[str] = None      # Keep as free text, e.g., "Apr 15, 2026" or "2026-04-15"
    start_time: Optional[str] = None      # Free text, e.g., "7:30 PM"
    ticket_url: Optional[str] = None
    venue_capacity: Optional[str] = None  # String to allow ranges/approx (e.g., "18,064")
    capacity_source_url: Optional[str] = None


class EventsExtraction(BaseModel):
    events: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
Extract up to the first 5 distinct ticketed entertainment events that the answer proposes for the SF Bay Area between Apr 1, 2026 and Jun 30, 2026. For each event, return:

- event_name: The event title/name
- performer_or_team: Main performer(s) or sports team(s)
- category: Choose one that best fits from this list (use exact spelling from the list): 
  ["concerts/music performances","professional sports games","theater/performing arts shows","comedy shows","opera performances","ballet performances","symphony orchestra concerts"]
- venue_name: The venue name
- venue_address: The complete physical address (street, city, state; ZIP if provided)
- city: City name (e.g., "San Francisco", "San Jose", etc.)
- event_date: The event date as written in the answer
- start_time: The start time as written in the answer
- ticket_url: A single ticket purchase URL explicitly present in the answer text (must be copied exactly; do not invent)
- venue_capacity: The venue seating capacity (if given in the answer). Keep as-is, including commas or words like "approx."
- capacity_source_url: A URL explicitly given in the answer that supports the venue capacity. If none, set to null.

Rules:
1) Extract only items explicitly present in the answer text. Do not fabricate.
2) The ticket_url and capacity_source_url must be explicit URLs in the answer (in any reasonable format).
3) If a field is missing for an event, set it to null.
4) Return at most 5 events in 'events' array (the first 5 if more are present).
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_category(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().lower()
    # Exact match first
    for allowed in ALLOWED_CATEGORIES:
        if s == allowed.lower():
            return allowed
    # Fuzzy contains
    for key, mapped in CATEGORY_KEYS.items():
        if key in s:
            return mapped
    return None


def make_non_empty(*vals: Optional[str]) -> bool:
    return all(isinstance(v, str) and v.strip() != "" for v in vals)


def unique_event_keys(events: List[EventItem]) -> Tuple[int, List[str]]:
    """
    Compute uniqueness by tuple of (event_name or performer_or_team), event_date, venue_name.
    Returns: (unique_count, list_of_keys)
    """
    keys = []
    for e in events:
        base = (e.event_name or e.performer_or_team or "") .strip().lower()
        date = (e.event_date or "").strip().lower()
        venue = (e.venue_name or "") .strip().lower()
        keys.append(f"{base}|{date}|{venue}")
    unique_set = set(k for k in keys if any(part for part in k.split("|")))
    return len(unique_set), keys


# --------------------------------------------------------------------------- #
# Verification routines per event                                             #
# --------------------------------------------------------------------------- #
async def verify_event_identification(evaluator: Evaluator, parent, idx: int, ev: EventItem) -> None:
    """
    Event_i_Identification node: sequential checks
    - Existence of key fields
    - Date within window (simple verify)
    - Ticket page matches the event details (by URL)
    - Category supported by the ticket page (optional non-critical but helpful)
    """
    node = evaluator.add_sequential(
        id=f"event_{idx}_identification",
        desc=f"Event {idx + 1} is identified with a specific name, performer/team, and confirmed date/time within the Apr 1 - Jun 30, 2026 timeframe in the SF Bay Area.",
        parent=parent,
        critical=False
    )

    # Existence gate
    existence_ok = make_non_empty(ev.event_name, ev.performer_or_team, ev.event_date, ev.start_time)
    evaluator.add_custom_node(
        result=existence_ok,
        id=f"event_{idx}_ident_fields_provided",
        desc=f"Event {idx + 1}: name, performer/team, date, and start time are provided",
        parent=node,
        critical=True
    )

    # Date in range (simple verify without URL)
    date_range_leaf = evaluator.add_leaf(
        id=f"event_{idx}_date_in_range",
        desc=f"Event {idx + 1}: date is within {DATE_WINDOW_TEXT}",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event date '{ev.event_date or ''}' falls {DATE_WINDOW_TEXT}.",
        node=date_range_leaf,
        additional_instruction="Interpret common date formats (e.g., 'Apr 5, 2026', '2026-04-05'). Treat the boundary dates (Apr 1, 2026 and Jun 30, 2026) as included."
    )

    # Ticket page matches event details (use ticket_url)
    match_leaf = evaluator.add_leaf(
        id=f"event_{idx}_ticket_page_matches_event",
        desc=f"Event {idx + 1}: ticket page corresponds to the named performer/team, date, time, and venue",
        parent=node,
        critical=True
    )
    match_claim = (
        f"This page sells tickets for the event '{ev.event_name or ''}' featuring '{ev.performer_or_team or ''}' "
        f"on {ev.event_date or ''} at {ev.start_time or ''} at {ev.venue_name or ''}."
    )
    await evaluator.verify(
        claim=match_claim,
        node=match_leaf,
        sources=ev.ticket_url or None,
        additional_instruction="Minor formatting differences or abbreviations are acceptable (e.g., '&' vs 'and', case-insensitive, exact time zone not required). The page should clearly correspond to the same event instance in the Bay Area."
    )

    # Category supported by the ticket page (non-critical)
    normalized_cat = normalize_category(ev.category)
    cat_leaf = evaluator.add_leaf(
        id=f"event_{idx}_category_supported",
        desc=f"Event {idx + 1}: category is supported by the ticket page",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=f"This event is a '{normalized_cat or (ev.category or '')}' type consistent with the allowed categories: {ALLOWED_CATEGORIES}.",
        node=cat_leaf,
        sources=ev.ticket_url or None,
        additional_instruction="Judge the type from performer/team and context (music => concerts; NBA/MLB/etc. => professional sports; play/musical/dance => theater/performing arts; stand-up => comedy; opera => opera; ballet => ballet; 'symphony/orchestra/philharmonic' => symphony)."
    )


async def verify_event_venue(evaluator: Evaluator, parent, idx: int, ev: EventItem) -> None:
    """
    Event_i_Venue node: sequential checks
    - Venue provided (name + full address)
    - Ticket page shows the same venue
    - Address matches (or is reasonably equivalent)
    - Venue is in the SF Bay Area
    """
    node = evaluator.add_sequential(
        id=f"event_{idx}_venue",
        desc=f"Event {idx + 1} has a verified venue name and complete physical address in the San Francisco Bay Area.",
        parent=parent,
        critical=False
    )

    # Existence gate
    venue_ok = make_non_empty(ev.venue_name, ev.venue_address)
    evaluator.add_custom_node(
        result=venue_ok,
        id=f"event_{idx}_venue_provided",
        desc=f"Event {idx + 1}: venue name and full address are provided",
        parent=node,
        critical=True
    )

    # Venue shown on ticket page
    venue_match_leaf = evaluator.add_leaf(
        id=f"event_{idx}_venue_matches_page",
        desc=f"Event {idx + 1}: ticket page shows the stated venue",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ticket page indicates the venue is '{ev.venue_name or ''}'.",
        node=venue_match_leaf,
        sources=ev.ticket_url or None,
        additional_instruction="Allow small variations or brandings in venue naming (e.g., sponsor names)."
    )

    # Address matches (accept close match)
    addr_leaf = evaluator.add_leaf(
        id=f"event_{idx}_address_complete_supported",
        desc=f"Event {idx + 1}: the complete venue address is supported by sources",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The complete physical address for '{ev.venue_name or ''}' is '{ev.venue_address or ''}'.",
        node=addr_leaf,
        sources=ev.ticket_url or None,
        additional_instruction="Accept if the page shows the same address or an equivalent formatting (street number/name, city, state). Minor punctuation/abbreviation differences are fine."
    )

    # Bay Area check
    bay_leaf = evaluator.add_leaf(
        id=f"event_{idx}_venue_in_bay_area",
        desc=f"Event {idx + 1}: venue is located in the San Francisco Bay Area",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{ev.venue_name or ''}' at address '{ev.venue_address or ''}' is located in the San Francisco Bay Area.",
        node=bay_leaf,
        sources=ev.ticket_url or None,
        additional_instruction=BAY_AREA_INSTRUCTION
    )


async def verify_event_tickets(evaluator: Evaluator, parent, idx: int, ev: EventItem) -> None:
    """
    Event_i_Tickets node: sequential checks
    - Ticket URL provided
    - Ticket link is official/legitimate source
    - Ticket page is specific to the event instance (redundant confirmation)
    """
    node = evaluator.add_sequential(
        id=f"event_{idx}_tickets",
        desc=f"Event {idx + 1} has a valid official ticket purchase link.",
        parent=parent,
        critical=False
    )

    # Existence gate
    has_ticket = make_non_empty(ev.ticket_url)
    evaluator.add_custom_node(
        result=has_ticket,
        id=f"event_{idx}_ticket_url_provided",
        desc=f"Event {idx + 1}: ticket URL is provided",
        parent=node,
        critical=True
    )

    # Official vendor/source
    official_leaf = evaluator.add_leaf(
        id=f"event_{idx}_ticket_link_official",
        desc=f"Event {idx + 1}: ticket URL is an official/legitimate purchase source",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This URL is an official or legitimate primary ticket purchase page for the specified event.",
        node=official_leaf,
        sources=ev.ticket_url or None,
        additional_instruction=OFFICIAL_VENDOR_INSTRUCTION
    )

    # Page is specific to the event (confirm again here)
    specific_leaf = evaluator.add_leaf(
        id=f"event_{idx}_ticket_page_specific",
        desc=f"Event {idx + 1}: ticket page specifically corresponds to the event instance",
        parent=node,
        critical=True
    )
    specific_claim = (
        f"The page sells tickets for '{ev.event_name or ''}'/'{ev.performer_or_team or ''}' on {ev.event_date or ''} at {ev.start_time or ''} "
        f"at '{ev.venue_name or ''}' in the Bay Area."
    )
    await evaluator.verify(
        claim=specific_claim,
        node=specific_leaf,
        sources=ev.ticket_url or None,
        additional_instruction="Look for the exact date/time and the same venue. Minor differences in formatting or abbreviations are acceptable."
    )


async def add_capacity_verifications(
    evaluator: Evaluator,
    parent,
    idx: int,
    ev: EventItem
) -> None:
    """
    Under the Capacity_Verification parent (parallel), add per-event checks:
    - Capacity provided (non-critical, existence)
    - Capacity supported by a source URL (non-critical, verify by URL)
    """
    # Existence (non-critical)
    evaluator.add_custom_node(
        result=make_non_empty(ev.venue_capacity),
        id=f"event_{idx}_capacity_provided",
        desc=f"Event {idx + 1}: venue seating capacity is provided",
        parent=parent,
        critical=False
    )

    # Supported by source (non-critical)
    cap_leaf = evaluator.add_leaf(
        id=f"event_{idx}_capacity_supported",
        desc=f"Event {idx + 1}: venue capacity is supported by cited source",
        parent=parent,
        critical=False
    )
    cap_claim = f"The seating capacity of the venue '{ev.venue_name or ''}' is '{ev.venue_capacity or ''}'."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=(ev.capacity_source_url or None),
        additional_instruction="Accept small rounding or approximations (e.g., 18,064 vs 18k). The source should clearly indicate the general seating capacity of the venue."
    )


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
) -> Dict:
    """
    Evaluate an answer for the SF Bay Area entertainment events (Q2 2026) task.
    Returns a standardized summary dictionary.
    """
    # Initialize evaluator with a parallel root
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

    # 1) Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction"
    )

    # Normalize and limit to 5 events; pad with empties if fewer
    events: List[EventItem] = list(extracted.events[:5])
    while len(events) < 5:
        events.append(EventItem())

    # Record some helpful custom info
    evaluator.add_custom_info(
        {
            "allowed_categories": ALLOWED_CATEGORIES,
            "date_window": DATE_WINDOW_TEXT,
            "extracted_event_count": len(extracted.events),
        },
        info_type="task_parameters",
        info_name="task_parameters"
    )

    # 2) Build rubric tree nodes corresponding to the rubric JSON (and necessary leaves)
    # Event-specific nodes (Identification / Venue / Tickets)
    for i, ev in enumerate(events):
        await verify_event_identification(evaluator, root, i, ev)
        await verify_event_venue(evaluator, root, i, ev)
        await verify_event_tickets(evaluator, root, i, ev)

    # Capacity Verification parent (non-critical, parallel children)
    cap_parent = evaluator.add_parallel(
        id="capacity_verification",
        desc="At least 4 of the 5 event venues have verifiable seating capacity information.",
        parent=root,
        critical=False
    )
    for i, ev in enumerate(events):
        await add_capacity_verifications(evaluator, cap_parent, i, ev)

    # Category diversity (critical)
    normalized_categories = [normalize_category(e.category) for e in events if e.category]
    unique_cats = set([c for c in normalized_categories if c])
    evaluator.add_custom_node(
        result=(len(unique_cats) >= 3),
        id="category_diversity",
        desc="The 5 events collectively represent at least 3 different categories from: concerts/music, professional sports, theater/performing arts, comedy, opera, ballet, or symphony orchestra.",
        parent=root,
        critical=True
    )

    # Event distinctness (critical)
    unique_count, _ = unique_event_keys(events)
    evaluator.add_custom_node(
        result=(unique_count == 5),
        id="event_distinctness",
        desc="All 5 events are distinct (differing by date and/or venue and/or performers/teams).",
        parent=root,
        critical=True
    )

    # 3) Return evaluation summary
    return evaluator.get_summary()