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
TASK_ID = "California_Major_Events_Feb_Apr_2026"
TASK_DESCRIPTION = (
    "I am planning to attend major entertainment and sporting events in California during the early part of 2026. "
    "Please identify four different major ticketed events that meet all of the following criteria:\n\n"
    "1. The event must take place in California, United States\n"
    "2. The event must occur between February 1, 2026 and April 30, 2026 (inclusive)\n"
    "3. The event must be either a major sporting event (such as professional sports championships, all-star games, "
    "or major tournaments) or a large-scale music festival\n"
    "4. The venue must have a minimum capacity of at least 15,000 attendees\n"
    "5. The event must be officially announced with confirmed dates (not tentative or rumored)\n"
    "6. Each event must be held at a single, specific venue\n\n"
    "For each of the four events, please provide:\n"
    "- The official event name\n"
    "- The exact date(s) when the event takes place\n"
    "- The name of the venue\n"
    "- The city in California where the venue is located\n"
    "- The venue's capacity\n"
    "- A reference URL to an official or authoritative source confirming the event details"
)

DATE_RANGE_START = "2026-02-01"
DATE_RANGE_END = "2026-04-30"
MIN_CAPACITY = 15000

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    event_name: Optional[str] = None
    date_text: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    venue_state: Optional[str] = None
    venue_capacity: Optional[str] = None
    event_type: Optional[str] = None  # e.g., "major sporting event", "large-scale music festival"
    source_urls: List[str] = Field(default_factory=list)  # official/authoritative pages confirming event details
    venue_source_urls: List[str] = Field(default_factory=list)  # pages confirming venue capacity/details


class EventsExtraction(BaseModel):
    events: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract up to the first four event entries (in the order they appear) from the answer text, preserving the details as stated.
    For each event, extract the following fields exactly as written in the answer:
    - event_name: official event name (string)
    - date_text: exact date(s) as written (string; may be a range like "April 12–14, 2026")
    - start_date: if the answer provides a specific start date (string), else null
    - end_date: if the answer provides a specific end date (string), else null
    - venue_name: the single, specific venue name (string)
    - venue_city: the California city for the venue (string)
    - venue_state: the state if mentioned (string; e.g., "California" or "CA"), else null
    - venue_capacity: the venue capacity as stated in the answer (string, do not convert to number); if not provided in the answer, set to null
    - event_type: if the answer states the type (e.g., "major sporting event" or "large-scale music festival"), keep it as a short string; else null
    - source_urls: a list of authoritative/official URLs the answer provides to confirm the event details (event website, league site, Ticketmaster, venue page, or reputable news). Only include URLs explicitly present in the answer.
    - venue_source_urls: a list of URLs explicitly in the answer that specifically support the venue details (especially capacity). Only include URLs explicitly present in the answer.

    Important:
    - Do not invent or infer any values. Only extract what is explicitly present in the answer.
    - For URLs, extract full URLs as they appear (including protocol).
    - If some field is not present in the answer for an event, set it to null (or [] for URL lists).
    - Return a JSON object with a top-level "events" array of up to 4 objects (one per event).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_event(e: EventItem) -> bool:
    return bool(
        (e.event_name and e.event_name.strip())
        or (e.date_text and e.date_text.strip())
        or (e.venue_name and e.venue_name.strip())
        or (e.source_urls)
        or (e.venue_source_urls)
    )


def _collect_all_sources(e: EventItem) -> List[str]:
    seen = set()
    combined: List[str] = []
    for url in (e.source_urls or []):
        if isinstance(url, str) and url.strip() and url not in seen:
            combined.append(url.strip())
            seen.add(url.strip())
    for url in (e.venue_source_urls or []):
        if isinstance(url, str) and url.strip() and url not in seen:
            combined.append(url.strip())
            seen.add(url.strip())
    return combined


def _event_summary_for_distinctness(e: EventItem, idx: int) -> str:
    parts = []
    if e.event_name:
        parts.append(e.event_name)
    if e.date_text:
        parts.append(f"dates: {e.date_text}")
    if e.venue_name:
        parts.append(f"venue: {e.venue_name}")
    if e.venue_city:
        parts.append(f"city: {e.venue_city}")
    return f"Event #{idx + 1}: " + " | ".join(parts) if parts else f"Event #{idx + 1}: [no details]"


# --------------------------------------------------------------------------- #
# Verification for a single event                                             #
# --------------------------------------------------------------------------- #
async def verify_single_event(evaluator: Evaluator, parent_node, e: EventItem, idx: int) -> None:
    event_num = idx + 1
    ev_node = evaluator.add_parallel(
        id=f"Event_{event_num}",
        desc=f"{['First','Second','Third','Fourth'][idx]} event meets all constraints and includes all required fields.",
        parent=parent_node,
        critical=False  # allow partial credit across events
    )

    # Existence/provided checks (custom boolean)
    evaluator.add_custom_node(
        result=bool(e.event_name and e.event_name.strip()),
        id=f"Event_{event_num}_Name_Provided",
        desc="Official event name is provided.",
        parent=ev_node,
        critical=True
    )
    # Exact dates provided (use simple_verify on the answer content)
    exact_dates_node = evaluator.add_leaf(
        id=f"Event_{event_num}_Exact_Dates_Provided",
        desc="Exact event date(s) are provided (not just a month/season).",
        parent=ev_node,
        critical=True
    )
    dates_text_display = e.date_text if e.date_text else ""
    await evaluator.verify(
        claim=f"The provided date(s) for Event #{event_num} are specific exact day(s), not vague month/season text. Dates text: '{dates_text_display}'.",
        node=exact_dates_node,
        additional_instruction="Judge only from the answer text. Exact dates should include specific day(s) (e.g., 'April 12–14, 2026'); vague ranges like 'April 2026' or 'Spring 2026' are not exact."
    )

    # Dates in range (verify by URLs)
    dates_in_range_node = evaluator.add_leaf(
        id=f"Event_{event_num}_Dates_In_Range",
        desc="Event date(s) fall within Feb 1, 2026 through Apr 30, 2026 (inclusive).",
        parent=ev_node,
        critical=True
    )
    await evaluator.verify(
        claim="The event occurs entirely between February 1, 2026 and April 30, 2026 (inclusive).",
        node=dates_in_range_node,
        sources=_collect_all_sources(e),
        additional_instruction=(
            "Use the provided authoritative URLs to confirm the official dates. "
            "All event dates must be within 2026-02-01 and 2026-04-30 inclusive. "
            "If any part of the event occurs outside this range, mark as not supported."
        )
    )

    # Location in California, US (verify by URLs)
    location_node = evaluator.add_leaf(
        id=f"Event_{event_num}_Location_California_US",
        desc="Event takes place in California, United States.",
        parent=ev_node,
        critical=True
    )
    city_str = e.venue_city or ""
    await evaluator.verify(
        claim=f"The event takes place in California, United States. Venue city: '{city_str}'.",
        node=location_node,
        sources=_collect_all_sources(e),
        additional_instruction="Confirm the venue is in the state of California (United States) per the provided URLs."
    )

    # Event type validity (major sporting event or large-scale music festival)
    type_valid_node = evaluator.add_leaf(
        id=f"Event_{event_num}_Event_Type_Valid",
        desc="Event is either a major sporting event or a large-scale music festival.",
        parent=ev_node,
        critical=True
    )
    await evaluator.verify(
        claim="This event is a major sporting event (e.g., professional sports championship, all-star game, or major tournament) or a large-scale music festival.",
        node=type_valid_node,
        sources=_collect_all_sources(e),
        additional_instruction=(
            "Use the URLs to determine the event's nature. Accept top-tier pro sports, major tournaments, "
            "or large multi-artist festivals drawing large crowds. If it is a minor/local event, mark as not supported."
        )
    )

    # Ticketed event (verify by URLs)
    ticketed_node = evaluator.add_leaf(
        id=f"Event_{event_num}_Ticketed_Event",
        desc="Event is a ticketed event (not a free public event).",
        parent=ev_node,
        critical=True
    )
    await evaluator.verify(
        claim="This event requires paid tickets (i.e., it is a ticketed event).",
        node=ticketed_node,
        sources=_collect_all_sources(e),
        additional_instruction="Look for official ticketing language (e.g., tickets on sale, passes, Ticketmaster links)."
    )

    # Single, specific venue (verify by URLs)
    single_venue_node = evaluator.add_leaf(
        id=f"Event_{event_num}_Single_Specific_Venue",
        desc="Event is held at a single, specific venue (not a multi-venue tour).",
        parent=ev_node,
        critical=True
    )
    vn = e.venue_name or ""
    await evaluator.verify(
        claim=f"The event is held at a single, specific venue named '{vn}'. Multiple days at the same venue is acceptable.",
        node=single_venue_node,
        sources=_collect_all_sources(e),
        additional_instruction="If the event spans multiple distinct venues or is a touring series, mark as not supported."
    )

    # Venue name provided (custom)
    evaluator.add_custom_node(
        result=bool(e.venue_name and e.venue_name.strip()),
        id=f"Event_{event_num}_Venue_Name_Provided",
        desc="Specific venue name is provided.",
        parent=ev_node,
        critical=True
    )

    # Venue city provided (custom)
    evaluator.add_custom_node(
        result=bool(e.venue_city and e.venue_city.strip()),
        id=f"Event_{event_num}_Venue_City_Provided",
        desc="Venue city in California is provided.",
        parent=ev_node,
        critical=True
    )

    # Venue capacity provided (custom)
    evaluator.add_custom_node(
        result=bool(e.venue_capacity and e.venue_capacity.strip()),
        id=f"Event_{event_num}_Venue_Capacity_Provided",
        desc="Venue capacity is provided.",
        parent=ev_node,
        critical=True
    )

    # Venue capacity >= 15,000 (verify by URLs)
    capacity_min_node = evaluator.add_leaf(
        id=f"Event_{event_num}_Venue_Capacity_Min_15000",
        desc="Venue capacity is at least 15,000 attendees.",
        parent=ev_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{vn}' has a capacity of at least {MIN_CAPACITY} attendees.",
        node=capacity_min_node,
        sources=_collect_all_sources(e),
        additional_instruction=(
            "Prefer official venue pages or reputable sources (e.g., venue site, Wikipedia, Ticketmaster) to confirm capacity. "
            "Approximate or max capacity meeting or exceeding threshold is acceptable."
        )
    )

    # Officially announced with confirmed dates (verify by URLs)
    announced_node = evaluator.add_leaf(
        id=f"Event_{event_num}_Officially_Announced_Confirmed",
        desc="Event is officially announced with confirmed dates (not tentative/rumored).",
        parent=ev_node,
        critical=True
    )
    await evaluator.verify(
        claim="The event is officially announced with confirmed dates (not tentative or rumored).",
        node=announced_node,
        sources=_collect_all_sources(e),
        additional_instruction="Use official/authoritative pages (event website, league site, official press releases) to confirm dates."
    )

    # Authoritative reference URL provided (custom existence)
    evaluator.add_custom_node(
        result=len(_collect_all_sources(e)) > 0,
        id=f"Event_{event_num}_Authoritative_Reference_URL",
        desc="Provides at least one official or authoritative reference URL supporting the event details.",
        parent=ev_node,
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=(
            "Identify four different major ticketed events in California between Feb 1, 2026 and Apr 30, 2026 inclusive, "
            "each meeting all constraints and providing all required fields with authoritative sourcing."
        ),
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured events
    extracted: EventsExtraction = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction",
    )

    # Record constraints as ground truth/context
    evaluator.add_ground_truth(
        {
            "date_range_start": DATE_RANGE_START,
            "date_range_end": DATE_RANGE_END,
            "min_capacity": MIN_CAPACITY,
            "required_events": 4,
            "location": "California, United States",
            "valid_types": ["major sporting event", "large-scale music festival"],
        },
        gt_type="constraints",
    )

    # Normalize to exactly 4 events (filter first 4, pad if needed)
    events = list(extracted.events[:4])
    while len(events) < 4:
        events.append(EventItem())

    # Top-level critical checks
    # 1) Four entries provided
    nonempty_count = sum(1 for e in extracted.events if _nonempty_event(e))
    evaluator.add_custom_node(
        result=(nonempty_count >= 4),
        id="Four_Event_Entries_Provided",
        desc="Response provides four separate event entries (Event 1–Event 4).",
        parent=root,
        critical=True,
    )

    # 2) Events are distinct (simple verify based on answer content)
    distinct_node = evaluator.add_leaf(
        id="Events_Are_Distinct",
        desc="The four events are different from each other (no duplicates/near-duplicates presented as separate events).",
        parent=root,
        critical=True,
    )
    summaries = [ _event_summary_for_distinctness(e, i) for i, e in enumerate(events) ]
    distinct_claim = (
        "The listed events are four distinct events (not the same event repeated, not multiple days/sessions of a single event). "
        "Here are the entries:\n" + "\n".join(summaries)
    )
    await evaluator.verify(
        claim=distinct_claim,
        node=distinct_node,
        additional_instruction=(
            "Judge based on the answer details. Distinct events should not be separate days of the same festival or "
            "home games of the same team. If two entries refer to the same underlying event, mark as incorrect."
        ),
    )

    # Per-event verification
    # Build four parallel event subtrees
    await asyncio.gather(
        *[verify_single_event(evaluator, root, events[i], i) for i in range(4)]
    )

    # Return evaluation summary
    return evaluator.get_summary()