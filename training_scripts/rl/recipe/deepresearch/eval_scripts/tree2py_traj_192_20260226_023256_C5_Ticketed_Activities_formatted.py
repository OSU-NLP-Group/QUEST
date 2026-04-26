import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "spring2026_indoor_arena_events"
TASK_DESCRIPTION = (
    "I am planning to attend several major entertainment events during spring 2026 (March 1 through May 31, 2026) and want to experience events at mid-sized indoor arenas across different states. "
    "Find 4 different ticketed entertainment events during this timeframe, where each event takes place at an indoor arena with a seating capacity between 14,000 and 24,000 for that event type. "
    "Each of the 4 events must be in a different U.S. state. At least one event must be a professional wrestling event, and at least one must be a concert or musical performance. "
    "For each event, provide: the official venue name, the event date, the city and state, the venue's seating capacity for that event type, and a reference URL from an official source confirming these details."
)

DATE_WINDOW_DESC = "between March 1, 2026 and May 31, 2026 (inclusive)"
CAPACITY_MIN = 14000
CAPACITY_MAX = 24000


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    venue_name: Optional[str] = None
    event_date: Optional[str] = None  # Keep as string to be flexible with formats ("May 12, 2026", "2026-05-12", etc.)
    city: Optional[str] = None
    state: Optional[str] = None
    capacity_for_event_type: Optional[str] = None  # Keep as string; may include qualifiers like "for concerts"
    event_type: Optional[str] = None  # e.g., "professional wrestling", "concert", "comedy", etc.
    reference_url: Optional[str] = None


class EventsExtraction(BaseModel):
    events: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
Extract from the answer up to 6 distinct ticketed entertainment events that the answer proposes for Spring 2026. For each event, extract the following fields exactly as presented:

- venue_name: The official name of the indoor arena venue for the event.
- event_date: The event date as written (e.g., "March 15, 2026" or "2026-03-15").
- city: The city where the venue is located.
- state: The U.S. state (or District of Columbia) where the venue is located. Prefer the standard two-letter abbreviation if available; otherwise keep the full state name exactly as written.
- capacity_for_event_type: The venue's seating capacity specifically for the event configuration or event type (e.g., "18,000 for concerts", "15,500 for basketball", "16,000 for wrestling"). If the answer provides only a single capacity value that clearly applies to the event, extract that value verbatim.
- event_type: The event category (e.g., "professional wrestling", "concert", "musical performance", "comedy", etc.) as stated.
- reference_url: A single official reference URL that the answer cites to confirm the event details. This should be a venue’s official website, the promoter/league’s official site, or an official ticketing platform (e.g., Ticketmaster, AXS) for this specific event. If multiple are provided, pick the most official one.

Return a JSON object:
{
  "events": [
    {
      "venue_name": ...,
      "event_date": ...,
      "city": ...,
      "state": ...,
      "capacity_for_event_type": ...,
      "event_type": ...,
      "reference_url": ...
    },
    ...
  ]
}

Rules:
1) Do not invent or infer missing fields; if a field is not present in the answer, set it to null.
2) Keep text exactly as it appears in the answer; do not normalize formats.
3) Only include events that the answer explicitly proposes for Spring 2026.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip().lower()
    # Normalize common variations minimally
    replacements = {
        "d.c.": "dc",
        "washington, dc": "dc",
        "washington dc": "dc",
        "district of columbia": "dc",
    }
    if s in replacements:
        s = replacements[s]
    # Remove punctuation and excessive spaces
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace()).strip()
    return s


def is_wrestling(event_type: Optional[str]) -> bool:
    if not event_type:
        return False
    s = event_type.strip().lower()
    keywords = [
        "wrestling",
        "wwe",
        "aew",
        "njpw",
        "impact wrestling",
        "smackdown",
        "raw",
        "collision",
        "dynamite",
        "nxt",
    ]
    return any(k in s for k in keywords)


def is_concert(event_type: Optional[str]) -> bool:
    if not event_type:
        return False
    s = event_type.strip().lower()
    keywords = [
        "concert",
        "tour",
        "musical performance",
        "music performance",
        "orchestra",
        "symphony",
        "band",
        "singer",
        "live music",
        "recital",
        "gig"
    ]
    return any(k in s for k in keywords)


def ensure_events_length(events: List[EventItem], target: int = 4) -> List[EventItem]:
    if len(events) >= target:
        return events[:target]
    padded = list(events)
    while len(padded) < target:
        padded.append(EventItem())
    return padded


# --------------------------------------------------------------------------- #
# Verification for a single event                                             #
# --------------------------------------------------------------------------- #
async def verify_event(
    evaluator: Evaluator,
    parent_node,
    event: EventItem,
    index: int,
) -> None:
    event_num = index + 1
    ev_node = evaluator.add_parallel(
        id=f"event_{event_num}",
        desc=f"{['First','Second','Third','Fourth'][index]} event meeting all criteria",
        parent=parent_node,
        critical=False
    )

    # Prepare leaf nodes
    # 1) Venue name (critical)
    venue_leaf = evaluator.add_leaf(
        id=f"event_{event_num}_venue_name",
        desc="Provide the official name of the indoor arena venue",
        parent=ev_node,
        critical=True
    )
    venue_claim = f"The official venue name for this event is '{event.venue_name or ''}'."
    await evaluator.verify(
        claim=venue_claim,
        node=venue_leaf,
        sources=event.reference_url,
        additional_instruction="Verify that the page confirms the venue name for this specific event. Allow minor branding variations (e.g., with/without 'The')."
    )

    # 2) Date in range (critical)
    date_leaf = evaluator.add_leaf(
        id=f"event_{event_num}_date",
        desc=f"Event date is {DATE_WINDOW_DESC}",
        parent=ev_node,
        critical=True
    )
    date_claim = (
        f"The event date is '{event.event_date or ''}', and it falls {DATE_WINDOW_DESC}."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=event.reference_url,
        additional_instruction="Confirm the event date shown on the page and judge whether it is between March 1, 2026 and May 31, 2026 (inclusive)."
    )

    # 3) Location (critical)
    location_leaf = evaluator.add_leaf(
        id=f"event_{event_num}_location",
        desc="Provide the city and state where the venue is located",
        parent=ev_node,
        critical=True
    )
    loc_city = event.city or ""
    loc_state = event.state or ""
    location_claim = f"The event takes place in {loc_city}, {loc_state}."
    await evaluator.verify(
        claim=location_claim,
        node=location_leaf,
        sources=event.reference_url,
        additional_instruction="Verify the event's city and state as listed on the page."
    )

    # 4) Capacity for event type within range (critical)
    capacity_leaf = evaluator.add_leaf(
        id=f"event_{event_num}_capacity",
        desc="The venue's seating capacity for this event type is between 14,000 and 24,000",
        parent=ev_node,
        critical=True
    )
    capacity_str = event.capacity_for_event_type or ""
    capacity_claim = (
        f"The venue's seating capacity for this event type is '{capacity_str}', "
        f"and it falls between {CAPACITY_MIN} and {CAPACITY_MAX}."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=event.reference_url,
        additional_instruction="Confirm the seating capacity applicable to this event configuration (e.g., concert/wrestling setup). "
                              "If the page shows a capacity or a range clearly within 14,000–24,000 for this event, consider it supported. "
                              "If capacity info is missing or clearly outside the range, mark as not supported."
    )

    # 5) Event type identification (non-critical)
    type_leaf = evaluator.add_leaf(
        id=f"event_{event_num}_type",
        desc="Identify the event type (wrestling, concert, or other ticketed entertainment)",
        parent=ev_node,
        critical=False
    )
    etype_str = event.event_type or ""
    type_claim = f"This event is a '{etype_str}' event."
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=event.reference_url,
        additional_instruction="Confirm from the page whether the event is indeed of this type (e.g., professional wrestling, concert). "
                              "Allow synonymous phrasing (e.g., 'live music performance' ≈ 'concert')."
    )

    # 6) Reference URL (critical)
    ref_leaf = evaluator.add_leaf(
        id=f"event_{event_num}_reference",
        desc="Provide a reference URL from an official source confirming the event details",
        parent=ev_node,
        critical=True
    )
    ref_claim = (
        "This URL is an official source (e.g., the venue’s site, the promoter/league’s site, or an official ticketing platform) "
        "that confirms the event’s date, venue name, and location. It should ideally also confirm event-specific capacity."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=event.reference_url,
        additional_instruction="Evaluate whether the page is an official/authoritative source and whether it clearly confirms the event’s date, venue, and location. "
                              "Official sources include venue websites, official ticketing (Ticketmaster, AXS), or league/promoter pages (e.g., WWE, AEW). "
                              "If the URL is missing, invalid, or obviously not official, mark as not supported."
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

    # Extract events from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction"
    )

    # Ensure we have exactly 4 items for evaluation (pad or truncate)
    events = ensure_events_length(extracted.events, 4)

    # Build event verifications
    # All event nodes are children of root (parallel aggregation as per rubric)
    verify_tasks = []
    for i in range(4):
        verify_tasks.append(verify_event(evaluator, root, events[i], i))
    await asyncio.gather(*verify_tasks)

    # Global constraints (critical) as custom nodes:
    # 1) Unique states across the four events
    state_values = [normalize_state(ev.state) for ev in events if ev is not None]
    unique_states = set([s for s in state_values if s])
    state_unique_ok = len(unique_states) == 4
    evaluator.add_custom_node(
        result=state_unique_ok,
        id="state_uniqueness",
        desc="All 4 events must be in different U.S. states or districts",
        parent=root,
        critical=True
    )

    # 2) At least one professional wrestling event
    wrestling_ok = any(is_wrestling(ev.event_type) for ev in events)
    evaluator.add_custom_node(
        result=wrestling_ok,
        id="wrestling_requirement",
        desc="At least one of the 4 events must be a professional wrestling event",
        parent=root,
        critical=True
    )

    # 3) At least one concert or musical performance event
    concert_ok = any(is_concert(ev.event_type) for ev in events)
    evaluator.add_custom_node(
        result=concert_ok,
        id="concert_requirement",
        desc="At least one of the 4 events must be a concert or musical performance",
        parent=root,
        critical=True
    )

    # Add a small custom info block with constants for transparency
    evaluator.add_custom_info(
        {
            "date_window": DATE_WINDOW_DESC,
            "capacity_range": [CAPACITY_MIN, CAPACITY_MAX],
            "required_events": 4
        },
        info_type="constraints",
        info_name="evaluation_constraints"
    )

    return evaluator.get_summary()