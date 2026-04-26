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
TASK_ID = "ca_indoor_arena_events_mar_apr_2026"
TASK_DESCRIPTION = (
    "I am planning to attend major entertainment events in California during March and April 2026. "
    "Please identify 5 different events scheduled between March 1 and April 30, 2026, that take place at indoor arena venues with a seating capacity of at least 15,000. "
    "The events should represent different types of entertainment (such as concerts, sports games, comedy shows, or other live performances).\n\n"
    "For each event, provide:\n"
    "1. Event Name and Description\n"
    "2. Date and Time\n"
    "3. Venue Name\n"
    "4. Venue Address\n"
    "5. Ticket Purchase URL\n"
    "6. Ticket Availability (currently available, not sold out)"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    event_name: Optional[str] = None
    entertainment_type: Optional[str] = None  # e.g., concert, sports, comedy, family, other
    description: Optional[str] = None
    date: Optional[str] = None               # e.g., "2026-03-15" or "March 15, 2026"
    start_time: Optional[str] = None         # e.g., "7:30 PM"
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None      # complete street address line
    city: Optional[str] = None
    state: Optional[str] = None              # expect "CA" or "California"
    ticket_url: Optional[str] = None         # direct purchase link
    ticket_availability: Optional[str] = None  # e.g., "Available", "On Sale", "Sold Out"
    capacity_mentioned: Optional[str] = None   # any capacity figure mentioned in answer
    source_urls: List[str] = Field(default_factory=list)  # any additional URLs cited for this event/venue


class EventsExtraction(BaseModel):
    events: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract all events listed in the answer that are intended as candidates for California indoor-arena entertainment during March–April 2026.

    For each event mentioned in the answer, extract the following fields (return null if missing):
    - event_name: The event title/name
    - entertainment_type: One of [concert, sports, comedy, family, other] (as implied by the answer)
    - description: A brief description of what the event is (as stated in the answer)
    - date: The scheduled calendar date for the event (keep the exact string form present in the answer)
    - start_time: The start time for the event (keep the exact string present in the answer)
    - venue_name: The venue name
    - venue_address: The complete street address of the venue (single line as written in the answer)
    - city: City of the venue if stated
    - state: State abbreviation or name if stated (e.g., "CA" or "California")
    - ticket_url: A direct link (URL) to purchase tickets online for this event (as provided in the answer)
    - ticket_availability: A short status phrase mentioned in the answer regarding tickets (e.g., "Available", "On sale", "Sold out", "Waitlist")
    - capacity_mentioned: Any capacity figure or phrase mentioned for the venue (e.g., "capacity 18,000")
    - source_urls: An array of any other URLs cited in the answer that support the event details or venue facts (e.g., official venue page, Wikipedia, team site).
      Do not invent URLs. Include the ticket_url as well if it appears in a separate "sources" section in the answer.

    IMPORTANT:
    - Only extract information explicitly present in the answer.
    - For URLs, include only valid URLs that appear in the answer (plain or within markdown links). Do not infer or create new URLs.
    - Keep the exact strings as shown in the answer for date/time/address.

    Return a JSON object with a single field:
    { "events": [ ... ] }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _dedup_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not _non_empty(u):
            continue
        u_norm = u.strip()
        if u_norm not in seen:
            seen.add(u_norm)
            out.append(u_norm)
    return out


def get_sources_for_event(evt: EventItem) -> List[str]:
    base = []
    if _non_empty(evt.ticket_url):
        base.append(evt.ticket_url.strip())
    if evt.source_urls:
        base.extend([u.strip() for u in evt.source_urls if _non_empty(u)])
    return _dedup_urls(base)


def event_signature(evt: EventItem) -> str:
    name = (evt.event_name or "").strip().lower()
    date = (evt.date or "").strip().lower()
    venue = (evt.venue_name or "").strip().lower()
    return f"{name}|{date}|{venue}"


# --------------------------------------------------------------------------- #
# Verification of a single event                                              #
# --------------------------------------------------------------------------- #
async def verify_single_event(
    evaluator: Evaluator,
    parent_node,
    evt: EventItem,
    idx: int,
) -> None:
    """
    Build the verification subtree for one event (Event idx+1).
    """
    display_num = idx + 1
    event_node = evaluator.add_parallel(
        id=f"Event_{display_num}",
        desc=f"Validate Event {display_num} against per-event constraints and required fields.",
        parent=parent_node,
        critical=False
    )

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=_non_empty(evt.event_name) and _non_empty(evt.description),
        id=f"event_{idx}_name_desc_provided",
        desc=f"Event {display_num} includes an event name and a description of what the event is.",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(evt.date),
        id=f"event_{idx}_date_provided",
        desc=f"Event {display_num} provides a specific event date.",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(evt.start_time),
        id=f"event_{idx}_start_time_provided",
        desc=f"Event {display_num} provides a start time.",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(evt.venue_name),
        id=f"event_{idx}_venue_name_provided",
        desc=f"Event {display_num} includes the venue name.",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(evt.venue_address),
        id=f"event_{idx}_venue_address_provided",
        desc=f"Event {display_num} includes the complete street address of the venue.",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(evt.ticket_url),
        id=f"event_{idx}_ticket_url_provided",
        desc=f"Event {display_num} provides a direct URL to purchase tickets online for that event.",
        parent=event_node,
        critical=True
    )

    # Fact checks with evidence (critical)
    sources = get_sources_for_event(evt)

    # 1) California location
    ca_node = evaluator.add_leaf(
        id=f"event_{idx}_CA_location",
        desc=f"Event {display_num} takes place in California, USA.",
        parent=event_node,
        critical=True
    )
    ca_claim = (
        f"The event '{(evt.event_name or '').strip()}' takes place in the U.S. state of California. "
        f"The venue is '{(evt.venue_name or '').strip()}', which is located in California (CA)."
    )
    # Prefer to verify via sources (ticket page or venue page)
    await evaluator.verify(
        claim=ca_claim,
        node=ca_node,
        sources=sources,
        additional_instruction=(
            "Verify from the provided page(s) that the venue/city is in California, USA. "
            "Accept if the address or city/state clearly indicates 'CA' or 'California'. "
            "If the webpage is unrelated or shows another state, mark this as not supported."
        )
    )

    # 2) Event date is within range
    inrange_node = evaluator.add_leaf(
        id=f"event_{idx}_date_in_range",
        desc=f"Event {display_num}'s event date is between 2026-03-01 and 2026-04-30 (inclusive).",
        parent=event_node,
        critical=True
    )
    if _non_empty(evt.date):
        range_claim = (
            f"The event is scheduled on {(evt.date or '').strip()}, and that date falls between "
            "March 1, 2026 and April 30, 2026 (inclusive)."
        )
    else:
        # If date missing, still let verifier check page date is in range (existence node will fail separately)
        range_claim = (
            "The event date shown on the referenced page falls between March 1, 2026 and April 30, 2026 (inclusive)."
        )
    await evaluator.verify(
        claim=range_claim,
        node=inrange_node,
        sources=sources,
        additional_instruction=(
            "Check the event date on the page. If the date indicates March or April 2026 within the 1st–30th (inclusive), pass. "
            "If the page shows 'TBA', missing, or a date outside that window, fail."
        )
    )

    # 3) Indoor arena venue
    indoor_node = evaluator.add_leaf(
        id=f"event_{idx}_indoor_arena",
        desc=f"Event {display_num} venue is an indoor arena.",
        parent=event_node,
        critical=True
    )
    indoor_claim = (
        f"The venue '{(evt.venue_name or '').strip()}' is an indoor arena (an enclosed multipurpose arena building)."
    )
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_node,
        sources=sources,
        additional_instruction=(
            "Look for explicit indications that the venue is an indoor/multipurpose arena. "
            "Venue name containing 'Arena' is suggestive, but prefer explicit statements on venue or reference pages. "
            "Do not accept open-air stadiums or amphitheaters."
        )
    )

    # 4) Venue capacity >= 15,000
    capacity_node = evaluator.add_leaf(
        id=f"event_{idx}_venue_capacity",
        desc=f"Event {display_num} venue seating capacity is at least 15,000.",
        parent=event_node,
        critical=True
    )
    capacity_claim = (
        f"The seating capacity of '{(evt.venue_name or '').strip()}' is at least 15,000."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_node,
        sources=sources,
        additional_instruction=(
            "Verify capacity from the referenced page(s). Basketball/hockey or concert configurations ≥ 15,000 qualify. "
            "If multiple capacities are listed, accept if any standard configuration is ≥ 15,000."
        )
    )

    # 5) Tickets currently available (not sold out)
    tix_node = evaluator.add_leaf(
        id=f"event_{idx}_tickets_available",
        desc=f"Event {display_num} confirms tickets are currently available for purchase (not sold out).",
        parent=event_node,
        critical=True
    )
    availability_claim = (
        "Tickets are currently available for purchase (not sold out) on the referenced ticket purchase page."
    )
    await evaluator.verify(
        claim=availability_claim,
        node=tix_node,
        sources=(evt.ticket_url or None),
        additional_instruction=(
            "Check the ticket purchase page for clear affordances like 'Buy Tickets', 'Find Tickets', seat selection, or prices. "
            "If the page indicates 'Sold Out', 'No tickets available', 'Waitlist', or only secondary resale without availability, fail."
        )
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
    Evaluate an answer for California indoor arena events (March–April 2026).
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
        default_model=model
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction"
    )

    # Determine event lists
    all_named_events = [e for e in (extracted.events or []) if _non_empty(e.event_name)]
    listed_count = len(all_named_events)

    # Choose top 5 to evaluate, pad with empties if needed
    chosen_events = all_named_events[:5]
    while len(chosen_events) < 5:
        chosen_events.append(EventItem())

    # Global constraints (critical)
    global_node = evaluator.add_parallel(
        id="Global_requirements",
        desc="Global constraints that apply across the full set of events.",
        parent=root,
        critical=True
    )

    # 1) Exactly 5 events listed (based on count of named events extracted from the answer)
    evaluator.add_custom_node(
        result=(listed_count == 5),
        id="Exactly_5_events_listed",
        desc="Response lists exactly 5 events.",
        parent=global_node,
        critical=True
    )

    # 2) All events distinct (use signature of first five named; only assess if exactly 5 named exist)
    if listed_count >= 5:
        first_five_named = all_named_events[:5]
    else:
        first_five_named = all_named_events[:listed_count]

    unique_sigs = set(event_signature(e) for e in first_five_named if _non_empty(e.event_name))
    all_distinct_result = (len(unique_sigs) == len(first_five_named) == 5)
    evaluator.add_custom_node(
        result=all_distinct_result,
        id="All_events_distinct",
        desc="All listed events are distinct (no duplicate events).",
        parent=global_node,
        critical=True
    )

    # 3) Entertainment type diversity (at least 2 distinct types among the five evaluated)
    types = set((e.entertainment_type or "").strip().lower() for e in chosen_events if _non_empty(e.entertainment_type))
    evaluator.add_custom_node(
        result=(len(types) >= 2),
        id="Entertainment_type_diversity",
        desc="Across the 5 events, at least 2 distinct entertainment types are represented (not all events are the same type).",
        parent=global_node,
        critical=True
    )

    # Per-event verification (non-critical group, but each per-leaf is critical as specified)
    for idx in range(5):
        await verify_single_event(evaluator, root, chosen_events[idx], idx)

    # Optional: record some custom info for debugging
    evaluator.add_custom_info(
        info={
            "total_events_found_in_answer": listed_count,
            "distinct_types_among_chosen": sorted(list(types)),
        },
        info_type="stats",
        info_name="extraction_stats"
    )

    return evaluator.get_summary()