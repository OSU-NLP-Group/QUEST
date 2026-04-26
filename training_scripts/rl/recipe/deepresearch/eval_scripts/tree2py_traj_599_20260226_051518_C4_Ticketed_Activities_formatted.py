import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_events_feb_to_jul_2026"
TASK_DESCRIPTION = (
    "Identify 4 different ticketed events occurring in the United States between February 2026 and July 2026. "
    "The 4 events must collectively include: (1) one theatrical production (play, musical, or stage performance), "
    "(2) one fan convention featuring celebrity guests from entertainment media, (3) one concert taking place at a venue "
    "with a capacity of at least 20,000 people, and (4) one multi-day music festival. Additionally, the 4 events must span "
    "at least 3 different U.S. states and must utilize at least 3 different types of venues (e.g., theater, convention center, "
    "amphitheater, outdoor festival grounds). For each event, provide: the event name, venue name, city, state, dates, relevant "
    "details (such as performers, guests, or production title), and a reference URL."
)

# Date range requirement (inclusive)
DATE_RANGE_START_TEXT = "February 1, 2026"
DATE_RANGE_END_TEXT = "July 31, 2026"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    """Single event item as extracted from the answer."""
    category: Optional[str] = None  # one of: theatrical_production | fan_convention | concert_20k | multi_day_festival
    event_name: Optional[str] = None
    venue_name: Optional[str] = None
    venue_type: Optional[str] = None  # e.g., theater, convention_center, arena, stadium, amphitheater, outdoor_festival_grounds, park, fairgrounds, other
    city: Optional[str] = None
    state: Optional[str] = None  # full state name or USPS abbreviation
    start_date: Optional[str] = None  # prefer ISO yyyy-mm-dd or any explicit string
    end_date: Optional[str] = None    # prefer ISO yyyy-mm-dd or any explicit string; if single-date event, can equal start_date
    dates_text: Optional[str] = None  # raw date range text as shown in the answer
    details: Optional[str] = None     # performers, production title, guest names, etc.
    performers_or_guests: List[str] = Field(default_factory=list)  # optional structured names
    venue_capacity_text: Optional[str] = None  # any capacity info mentioned
    reference_urls: List[str] = Field(default_factory=list)        # URLs that support the event info
    capacity_source_urls: List[str] = Field(default_factory=list)  # URLs that support the venue capacity (if provided separately)


class EventsExtraction(BaseModel):
    """Complete extraction of up to four categorized events."""
    events: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    You must extract exactly 4 different ticketed events described in the answer and categorize each into one of the 4 required categories:
    - theatrical_production: a play, musical, or stage performance
    - fan_convention: a fan convention featuring celebrity guests from entertainment media (film/TV/anime/games/etc.)
    - concert_20k: a concert held at a venue with capacity ≥ 20,000
    - multi_day_festival: a music festival spanning more than one day

    If the answer includes more than 4 events, pick the first valid item for each category (at most 1 per category). If the answer includes fewer than 4 events or a category is not present, include a placeholder item with null fields for that missing category.

    For each event, extract the following fields:
    - category: one of [theatrical_production, fan_convention, concert_20k, multi_day_festival]
    - event_name: the event's name
    - venue_name: the venue or location name
    - venue_type: map to one of [theater, convention_center, arena, stadium, amphitheater, outdoor_festival_grounds, park, fairgrounds, other]; pick the best fit based on the answer
    - city: city name
    - state: U.S. state (full name or USPS abbreviation); if not explicit, infer from the answer text if safe
    - start_date: the start date in "YYYY-MM-DD" if possible; otherwise any explicit date string
    - end_date: the end date in "YYYY-MM-DD" if possible; otherwise any explicit date string; if single-day event, set end_date = start_date
    - dates_text: the raw date string exactly as shown in the answer for this event (e.g., "June 12–14, 2026")
    - details: any relevant details (performers for concerts, production title for theater, celebrity guests for conventions, lineup for festivals)
    - performers_or_guests: list of performer or guest names if mentioned (otherwise empty list)
    - venue_capacity_text: any capacity information stated in the answer (e.g., "capacity 20,000+")
    - reference_urls: list of URLs explicitly provided in the answer that support this event (must be actual URLs; include ticketing pages, official sites, event pages, etc.)
    - capacity_source_urls: if the answer provides separate URLs for venue capacity, include them here; otherwise leave empty

    IMPORTANT RULES:
    - Only extract information explicitly present in the answer. Do not invent any details.
    - The URLs must be actual links mentioned in the answer (plain URLs or markdown links).
    - If a required field is missing, set it to null or empty as appropriate.
    - Ensure the 4 items collectively cover the 4 distinct categories. If the answer fails to cover a category, include a placeholder with that category and null fields.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def pick_event_by_category(events: List[EventItem], category: str) -> Optional[EventItem]:
    for idx, e in enumerate(events):
        if (e.category or "").strip().lower() == category:
            return events.pop(idx)
    return None


def safe_sources(event: EventItem) -> List[str]:
    """Return a list of sources for verification, always a list (may be empty)."""
    urls = []
    urls.extend(event.reference_urls or [])
    return urls


def capacity_sources(event: EventItem) -> List[str]:
    urls = []
    urls.extend(event.capacity_source_urls or [])
    # If no dedicated capacity sources, fall back to event references
    if not urls:
        urls.extend(event.reference_urls or [])
    return urls


# --------------------------------------------------------------------------- #
# Verification builders for each category                                     #
# --------------------------------------------------------------------------- #
async def verify_theatrical_event(evaluator: Evaluator, parent, event: EventItem) -> None:
    node = evaluator.add_sequential(
        id="theatrical_production_event",
        desc="Solution includes one theatrical production event (play, musical, or stage performance) with event details and supporting URL",
        parent=parent,
        critical=False
    )

    # Required info check (critical)
    has_required = bool(event.event_name and event.venue_name and event.city and event.state and (event.start_date or event.dates_text) and event.reference_urls)
    evaluator.add_custom_node(
        result=has_required,
        id="theatrical_required_info",
        desc="Theatrical event has required information and at least one reference URL",
        parent=node,
        critical=True
    )

    # Event supported by URLs (critical)
    support_leaf = evaluator.add_leaf(
        id="theatrical_event_supported",
        desc="The theatrical event details (name, venue, city/state, dates) are supported by the referenced URLs",
        parent=node,
        critical=True
    )
    claim = f"The webpage(s) describe the event '{event.event_name}' at '{event.venue_name}' in {event.city}, {event.state}, with dates '{event.dates_text or (event.start_date or '')}{'' if not event.end_date or event.end_date == event.start_date else ' to ' + event.end_date}'."
    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=safe_sources(event),
        additional_instruction="Allow reasonable variants of names and formatting. Confirm the event name, venue, city/state, and the scheduled dates on the page."
    )

    # Ticketed check (critical)
    ticket_leaf = evaluator.add_leaf(
        id="theatrical_ticketed",
        desc="The theatrical event is ticketed (requires paid tickets or shows 'buy tickets' information)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This event requires purchasing a ticket or shows ticketing/admission information on the referenced page(s).",
        node=ticket_leaf,
        sources=safe_sources(event),
        additional_instruction="Look for 'Buy Tickets', 'Tickets', 'Admission', pricing, or registration fee indicators. If free but requires a paid badge, consider ticketed."
    )

    # Category check: theatrical
    cat_leaf = evaluator.add_leaf(
        id="theatrical_is_stage",
        desc="This event is a theatrical production (play, musical, or stage performance)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The event is a theatrical production (a play, musical, or stage performance).",
        node=cat_leaf,
        sources=safe_sources(event),
        additional_instruction="Consider typical theater signals (cast list, production title, showtimes) and venue type 'theater'."
    )

    # Dates supported explicitly
    dates_leaf = evaluator.add_leaf(
        id="theatrical_dates_supported",
        desc="The theatrical event dates are explicitly supported by the referenced URLs",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event occurs on the dates stated: '{event.dates_text or (event.start_date or '')}{'' if not event.end_date or event.end_date == event.start_date else ' to ' + event.end_date}'.",
        node=dates_leaf,
        sources=safe_sources(event),
        additional_instruction="Match the listed dates or schedule window on the page. Accept minor formatting differences."
    )


async def verify_fan_convention_event(evaluator: Evaluator, parent, event: EventItem) -> None:
    node = evaluator.add_sequential(
        id="fan_convention_event",
        desc="Solution includes one fan convention event featuring celebrity guests with details and supporting URL",
        parent=parent,
        critical=False
    )

    # Required info (critical)
    has_required = bool(event.event_name and event.venue_name and event.city and event.state and (event.start_date or event.dates_text) and event.reference_urls)
    evaluator.add_custom_node(
        result=has_required,
        id="fan_con_required_info",
        desc="Fan convention event has required information and at least one reference URL",
        parent=node,
        critical=True
    )

    # Event supported (critical)
    support_leaf = evaluator.add_leaf(
        id="fan_con_event_supported",
        desc="The fan convention event details are supported by the referenced URLs",
        parent=node,
        critical=True
    )
    claim = f"The webpage(s) describe the convention '{event.event_name}' at '{event.venue_name}' in {event.city}, {event.state}, scheduled on '{event.dates_text or (event.start_date or '')}{'' if not event.end_date or event.end_date == event.start_date else ' to ' + event.end_date}'."
    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=safe_sources(event),
        additional_instruction="Verify event name, venue, city/state, and dates. Allow reasonable variations."
    )

    # Ticketed check (critical)
    ticket_leaf = evaluator.add_leaf(
        id="fan_con_ticketed",
        desc="The fan convention is ticketed (paid tickets/admission or badge purchase)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This convention requires purchasing a badge/ticket or shows pricing/admission information on the page(s).",
        node=ticket_leaf,
        sources=safe_sources(event),
        additional_instruction="Look for 'Buy Badge', 'Registration', 'Tickets', 'Pricing'."
    )

    # Celebrity guests in entertainment media (critical)
    celebrity_leaf = evaluator.add_leaf(
        id="fan_con_has_celeb_guests",
        desc="The convention features celebrity guests from entertainment media",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The referenced page(s) list celebrity guests from entertainment media (film/TV/anime/games/etc.).",
        node=celebrity_leaf,
        sources=safe_sources(event),
        additional_instruction="Look for 'Guests' sections listing known actors, voice actors, creators, or similar entertainment figures."
    )

    # Dates supported
    dates_leaf = evaluator.add_leaf(
        id="fan_con_dates_supported",
        desc="The fan convention dates are explicitly supported by the referenced URLs",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The convention occurs on the dates stated: '{event.dates_text or (event.start_date or '')}{'' if not event.end_date or event.end_date == event.start_date else ' to ' + event.end_date}'.",
        node=dates_leaf,
        sources=safe_sources(event),
        additional_instruction="Confirm the schedule window shown on the page."
    )


async def verify_concert_20k_event(evaluator: Evaluator, parent, event: EventItem) -> None:
    node = evaluator.add_sequential(
        id="large_capacity_concert_event",
        desc="Solution includes one concert (venue capacity ≥ 20,000) with details and supporting URL",
        parent=parent,
        critical=False
    )

    # Required info (critical)
    has_required = bool(event.event_name and event.venue_name and event.city and event.state and (event.start_date or event.dates_text) and event.reference_urls)
    evaluator.add_custom_node(
        result=has_required,
        id="concert_required_info",
        desc="Concert event has required information and at least one reference URL",
        parent=node,
        critical=True
    )

    # Event supported (critical)
    support_leaf = evaluator.add_leaf(
        id="concert_event_supported",
        desc="The concert event details are supported by the referenced URLs",
        parent=node,
        critical=True
    )
    claim = f"The webpage(s) describe the concert '{event.event_name}' at '{event.venue_name}' in {event.city}, {event.state}, scheduled on '{event.dates_text or (event.start_date or '')}{'' if not event.end_date or event.end_date == event.start_date else ' to ' + event.end_date}'."
    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=safe_sources(event),
        additional_instruction="Verify event name, venue, city/state, and date(s)."
    )

    # Ticketed check (critical)
    ticket_leaf = evaluator.add_leaf(
        id="concert_ticketed",
        desc="The concert is ticketed (ticket purchase/admission info shown)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This concert requires purchasing tickets or shows pricing/admission information on the page(s).",
        node=ticket_leaf,
        sources=safe_sources(event),
        additional_instruction="Look for 'Buy Tickets', 'Tickets', 'Pricing', 'Admission'."
    )

    # Concert classification (critical)
    cat_leaf = evaluator.add_leaf(
        id="concert_is_concert",
        desc="This event is a live music concert",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The event is a live music concert performance.",
        node=cat_leaf,
        sources=safe_sources(event),
        additional_instruction="Identify artist/band names, tour titles, or concert signals."
    )

    # Capacity ≥ 20,000 (critical)
    cap_leaf = evaluator.add_leaf(
        id="concert_capacity_20k",
        desc="The venue capacity is at least 20,000",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue capacity is at least 20,000 attendees.",
        node=cap_leaf,
        sources=capacity_sources(event),
        additional_instruction="Use venue official site, Wikipedia, or ticketing site capacity references. Accept approximate statements (e.g., '20,000+')."
    )

    # Dates supported
    dates_leaf = evaluator.add_leaf(
        id="concert_dates_supported",
        desc="The concert dates are explicitly supported by the referenced URLs",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The concert occurs on the dates stated: '{event.dates_text or (event.start_date or '')}{'' if not event.end_date or event.end_date == event.start_date else ' to ' + event.end_date}'.",
        node=dates_leaf,
        sources=safe_sources(event),
        additional_instruction="Confirm the concert date(s) on the page."
    )


async def verify_multi_day_festival_event(evaluator: Evaluator, parent, event: EventItem) -> None:
    node = evaluator.add_sequential(
        id="multi_day_music_festival_event",
        desc="Solution includes one multi-day music festival with details and supporting URL",
        parent=parent,
        critical=False
    )

    # Required info (critical)
    has_required = bool(event.event_name and event.venue_name and event.city and event.state and (event.start_date or event.dates_text) and event.reference_urls)
    evaluator.add_custom_node(
        result=has_required,
        id="festival_required_info",
        desc="Festival event has required information and at least one reference URL",
        parent=node,
        critical=True
    )

    # Event supported (critical)
    support_leaf = evaluator.add_leaf(
        id="festival_event_supported",
        desc="The festival event details are supported by the referenced URLs",
        parent=node,
        critical=True
    )
    claim = f"The webpage(s) describe the festival '{event.event_name}' at '{event.venue_name}' in {event.city}, {event.state}, scheduled on '{event.dates_text or (event.start_date or '')}{'' if not event.end_date or event.end_date == event.start_date else ' to ' + event.end_date}'."
    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=safe_sources(event),
        additional_instruction="Verify event name, venue/location, city/state, and dates."
    )

    # Ticketed check (critical)
    ticket_leaf = evaluator.add_leaf(
        id="festival_ticketed",
        desc="The festival is ticketed (ticketing/admission information shown)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This festival requires purchasing tickets/passes or shows pricing/admission information.",
        node=ticket_leaf,
        sources=safe_sources(event),
        additional_instruction="Look for 'Tickets', 'Passes', 'Pricing', 'Admission'."
    )

    # Music festival classification (critical)
    cat_leaf = evaluator.add_leaf(
        id="festival_is_music_festival",
        desc="This event is a music festival",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The event is a music festival.",
        node=cat_leaf,
        sources=safe_sources(event),
        additional_instruction="Look for multiple performers/lineup across days, festival branding, stages."
    )

    # Multi-day check (critical)
    multi_leaf = evaluator.add_leaf(
        id="festival_is_multiday",
        desc="The festival spans multiple days",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The festival spans multiple days (more than one day).",
        node=multi_leaf,
        sources=safe_sources(event),
        additional_instruction="Confirm at least two calendar dates in the schedule."
    )

    # Dates supported
    dates_leaf = evaluator.add_leaf(
        id="festival_dates_supported",
        desc="The festival dates are explicitly supported by the referenced URLs",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The festival occurs on the dates stated: '{event.dates_text or (event.start_date or '')}{'' if not event.end_date or event.end_date == event.start_date else ' to ' + event.end_date}'.",
        node=dates_leaf,
        sources=safe_sources(event),
        additional_instruction="Match the listed festival dates on the page."
    )


# --------------------------------------------------------------------------- #
# Global constraints verification                                             #
# --------------------------------------------------------------------------- #
async def add_global_constraints(
    evaluator: Evaluator,
    parent,
    theatrical: EventItem,
    fancon: EventItem,
    concert: EventItem,
    festival: EventItem,
) -> None:
    # Date range compliance: verify each event occurs between Feb 1, 2026 and Jul 31, 2026
    date_node = evaluator.add_parallel(
        id="date_range_compliance",
        desc="All 4 events occur between February 2026 and July 2026 (inclusive)",
        parent=parent,
        critical=True
    )

    for tag, ev in [
        ("theatrical", theatrical),
        ("fan_con", fancon),
        ("concert", concert),
        ("festival", festival),
    ]:
        leaf = evaluator.add_leaf(
            id=f"date_range_ok_{tag}",
            desc=f"Event '{tag}' dates fall within Feb–Jul 2026",
            parent=date_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The event occurs between {DATE_RANGE_START_TEXT} and {DATE_RANGE_END_TEXT} (inclusive).",
            node=leaf,
            sources=safe_sources(ev),
            additional_instruction="Confirm the scheduled dates fall within the specified window."
        )

    # Geographic diversity: at least 3 different states (critical)
    states = set([
        (theatrical.state or "").strip(),
        (fancon.state or "").strip(),
        (concert.state or "").strip(),
        (festival.state or "").strip(),
    ])
    states = {s for s in states if s}  # remove empty
    evaluator.add_custom_node(
        result=len(states) >= 3,
        id="geographic_diversity",
        desc="The 4 events span at least 3 different U.S. states",
        parent=parent,
        critical=True
    )

    # Venue type diversity: at least 3 different types (critical)
    venue_types = set([
        (theatrical.venue_type or "").strip().lower(),
        (fancon.venue_type or "").strip().lower(),
        (concert.venue_type or "").strip().lower(),
        (festival.venue_type or "").strip().lower(),
    ])
    venue_types = {vt for vt in venue_types if vt}
    evaluator.add_custom_node(
        result=len(venue_types) >= 3,
        id="venue_type_diversity",
        desc="The 4 events utilize at least 3 different types of venues",
        parent=parent,
        critical=True
    )

    # Reference URLs presence: each event has ≥ 1 URL (critical)
    ref_node = evaluator.add_parallel(
        id="reference_urls",
        desc="Each of the 4 events is supported by at least one valid reference URL from a credible source",
        parent=parent,
        critical=True
    )
    for tag, ev in [
        ("theatrical", theatrical),
        ("fan_con", fancon),
        ("concert", concert),
        ("festival", festival),
    ]:
        evaluator.add_custom_node(
            result=bool(ev.reference_urls),
            id=f"reference_urls_present_{tag}",
            desc=f"Event '{tag}' includes at least one reference URL",
            parent=ref_node,
            critical=True
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
    Evaluate an answer for the US events (Feb–Jul 2026) task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent verification tracks and global constraints
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

    # Extract events
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction"
    )

    # Ensure we have exactly 4 items across the required categories; create placeholders if needed
    events_pool = list(extracted.events or [])
    theatrical = pick_event_by_category(events_pool, "theatrical_production") or EventItem(category="theatrical_production")
    fancon = pick_event_by_category(events_pool, "fan_convention") or EventItem(category="fan_convention")
    concert = pick_event_by_category(events_pool, "concert_20k") or EventItem(category="concert_20k")
    festival = pick_event_by_category(events_pool, "multi_day_festival") or EventItem(category="multi_day_festival")

    # Add per-category verification subtrees
    await verify_theatrical_event(evaluator, root, theatrical)
    await verify_fan_convention_event(evaluator, root, fancon)
    await verify_concert_20k_event(evaluator, root, concert)
    await verify_multi_day_festival_event(evaluator, root, festival)

    # Add global constraints at root
    await add_global_constraints(evaluator, root, theatrical, fancon, concert, festival)

    # Add summary info for diversity calculation transparency
    evaluator.add_custom_info(
        info={
            "selected_states": [theatrical.state, fancon.state, concert.state, festival.state],
            "selected_venue_types": [theatrical.venue_type, fancon.venue_type, concert.venue_type, festival.venue_type],
            "date_window": {"start": DATE_RANGE_START_TEXT, "end": DATE_RANGE_END_TEXT}
        },
        info_type="diversity_summary"
    )

    # Return structured result
    return evaluator.get_summary()