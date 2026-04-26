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
TASK_ID = "ca_events_q1_2025_capacity_15k_25k"
TASK_DESCRIPTION = """
Identify two major entertainment or sporting events that took place in California between January 1 and March 31, 2025, where each event was held in a venue with a seating capacity between 15,000 and 25,000. For each event, provide: (1) The official name of the event, (2) The exact date it was held, (3) The specific venue name and city in California, (4) The seating capacity of the venue, (5) Reference URLs supporting your answer.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    official_name: Optional[str] = None
    event_type: Optional[str] = None  # e.g., "award ceremony", "all-star game", etc.
    date: Optional[str] = None        # exact date string as provided in the answer
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None       # e.g., "California", "CA"
    venue_capacity: Optional[str] = None  # capacity figure as written in the answer
    # Reference URLs explicitly mentioned in the answer text
    event_urls: List[str] = Field(default_factory=list)     # general/event overview refs
    date_urls: List[str] = Field(default_factory=list)      # refs confirming the date
    location_urls: List[str] = Field(default_factory=list)  # refs confirming city/venue
    capacity_urls: List[str] = Field(default_factory=list)  # refs confirming capacity


class EventsExtraction(BaseModel):
    events: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract all candidate events described in the answer. For each event, extract exactly these fields:
    - official_name: The official name of the event.
    - event_type: The event classification as written (e.g., "award ceremony", "all-star game", "concert"). If not stated, set to null.
    - date: The exact calendar date of the event as written in the answer (e.g., "February 9, 2025" or "2025-02-09"). If multiple days are listed, use the primary date mentioned or the specific event day.
    - venue_name: The specific venue name.
    - city: The California city where it took place.
    - state: The state string as written (e.g., "California" or "CA") if mentioned.
    - venue_capacity: The seating capacity value provided in the answer, as a string (e.g., "18,064", "around 18k"). Do not convert to a number; keep the original text.
    - event_urls: All URLs in the answer that directly reference the event overview or official page.
    - date_urls: All URLs in the answer that confirm the date of the event.
    - location_urls: All URLs in the answer that confirm the venue and city location.
    - capacity_urls: All URLs in the answer that confirm the venue's seating capacity.
    
    Rules:
    1) Extract only what is explicitly present in the answer. Do not invent URLs or details.
    2) URLs must be real full URLs (plain links or markdown links); extract the actual URL strings.
    3) If a field is missing, set it to null (for single fields) or [] for URL lists.
    4) Return a JSON with a top-level 'events' array, each item containing all the fields above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_number(text: Optional[str]) -> bool:
    return bool(text) and any(ch.isdigit() for ch in text)


def dedupe_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def coalesce_urls(*lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        if lst:
            merged.extend(lst)
    return dedupe_preserve_order(merged)


# --------------------------------------------------------------------------- #
# Verification logic per event                                                #
# --------------------------------------------------------------------------- #
async def verify_one_event(
    evaluator: Evaluator,
    parent_node,
    event: EventItem,
    idx: int
) -> None:
    # Event container node (non-critical under root; each event graded independently)
    ev_node = evaluator.add_parallel(
        id=f"event_{idx}",
        desc=f"{'First' if idx == 0 else 'Second'} qualifying event meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # -------------------- 1) Event identification -------------------------
    ident_node = evaluator.add_parallel(
        id=f"event_{idx}_identification",
        desc="Correct identification and classification of the event",
        parent=ev_node,
        critical=True
    )

    # 1.a Event name provided (existence)
    evaluator.add_custom_node(
        result=bool(event.official_name and event.official_name.strip()),
        id=f"event_{idx}_name",
        desc="Provides the official name of the event",
        parent=ident_node,
        critical=True
    )

    # 1.b Event type correctness (award ceremony or all-star sporting event), grounded to event URLs
    type_node = evaluator.add_leaf(
        id=f"event_{idx}_type",
        desc="Correctly identifies the event as a nationally or internationally recognized award ceremony or all-star sporting event",
        parent=ident_node,
        critical=True
    )
    type_claim = (
        f"The event '{event.official_name or 'UNKNOWN'}' is a nationally or internationally recognized "
        f"award ceremony or an all-star sporting event."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_node,
        sources=event.event_urls if event.event_urls else None,
        additional_instruction="Judge based on the provided webpage(s). Accept well-known awards (e.g., Grammys, Oscars) or recognized all-star sporting events (NBA All-Star, NHL All-Star, etc.)."
    )

    # 1.c Event reference URL(s) provided (existence)
    evaluator.add_custom_node(
        result=bool(event.event_urls and len(event.event_urls) > 0),
        id=f"event_{idx}_event_reference",
        desc="Provides a valid reference URL for the event",
        parent=ident_node,
        critical=True
    )

    # -------------------- 2) Temporal criteria ----------------------------
    temporal_node = evaluator.add_parallel(
        id=f"event_{idx}_temporal",
        desc="Event date verification",
        parent=ev_node,
        critical=True
    )

    # 2.a Date falls within Jan 1 – Mar 31, 2025 (logical verification)
    date_range_node = evaluator.add_leaf(
        id=f"event_{idx}_date_in_range",
        desc="Event date falls within January 1 - March 31, 2025",
        parent=temporal_node,
        critical=True
    )
    date_text = event.date or "UNKNOWN"
    range_claim = (
        f"The provided event date '{date_text}' falls between January 1, 2025 and March 31, 2025 (inclusive)."
    )
    await evaluator.verify(
        claim=range_claim,
        node=date_range_node,
        additional_instruction="Judge purely by the semantics of the date string; do not rely on your own calendar memory. If multiple days are implied, the primary day must be within the range."
    )

    # 2.b Specific date provided (existence/logical check)
    evaluator.add_custom_node(
        result=bool(event.date and event.date.strip()),
        id=f"event_{idx}_specific_date_provided",
        desc="Provides the exact date of the event",
        parent=temporal_node,
        critical=True
    )

    # 2.c Date reference URL(s), grounded
    date_ref_node = evaluator.add_leaf(
        id=f"event_{idx}_date_reference",
        desc="Provides a reference URL confirming the date",
        parent=temporal_node,
        critical=True
    )
    date_ref_claim = (
        f"The event '{event.official_name or 'UNKNOWN'}' took place on {date_text}."
    )
    await evaluator.verify(
        claim=date_ref_claim,
        node=date_ref_node,
        sources=event.date_urls if event.date_urls else None,
        additional_instruction="From the provided page(s), confirm the specific event date (allow minor format differences, e.g., 'Feb 9, 2025' vs 'February 9, 2025')."
    )

    # -------------------- 3) Location criteria ----------------------------
    loc_node = evaluator.add_parallel(
        id=f"event_{idx}_location",
        desc="Event location verification",
        parent=ev_node,
        critical=True
    )

    # 3.a California state (ground to location or general event URLs)
    ca_state_node = evaluator.add_leaf(
        id=f"event_{idx}_california_state",
        desc="Event takes place in the state of California",
        parent=loc_node,
        critical=True
    )
    ca_sources = event.location_urls if event.location_urls else event.event_urls
    ca_claim = f"The event '{event.official_name or 'UNKNOWN'}' took place in California, United States."
    await evaluator.verify(
        claim=ca_claim,
        node=ca_state_node,
        sources=ca_sources if ca_sources else None,
        additional_instruction="Verify that the event location is in the state of California."
    )

    # 3.b City provided (existence)
    evaluator.add_custom_node(
        result=bool(event.city and event.city.strip()),
        id=f"event_{idx}_city_provided",
        desc="Correctly identifies the California city where the event is held",
        parent=loc_node,
        critical=True
    )

    # 3.c Venue provided (existence)
    evaluator.add_custom_node(
        result=bool(event.venue_name and event.venue_name.strip()),
        id=f"event_{idx}_venue_provided",
        desc="Provides the specific venue name",
        parent=loc_node,
        critical=True
    )

    # 3.d Location reference (grounded verification for venue+city in CA)
    loc_ref_node = evaluator.add_leaf(
        id=f"event_{idx}_location_reference",
        desc="Provides a reference URL confirming the location",
        parent=loc_node,
        critical=True
    )
    loc_sources = event.location_urls if event.location_urls else event.event_urls
    loc_claim = (
        f"The event '{event.official_name or 'UNKNOWN'}' took place at '{event.venue_name or 'UNKNOWN VENUE'}' "
        f"in {event.city or 'UNKNOWN CITY'}, California."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_ref_node,
        sources=loc_sources if loc_sources else None,
        additional_instruction="The page should explicitly indicate both the venue name and the California city for this event."
    )

    # -------------------- 4) Capacity criteria ----------------------------
    cap_node = evaluator.add_parallel(
        id=f"event_{idx}_capacity",
        desc="Venue seating capacity verification",
        parent=ev_node,
        critical=True
    )

    # 4.a Capacity in range [15,000, 25,000] (grounded, allow any credible capacity page)
    cap_in_range_node = evaluator.add_leaf(
        id=f"event_{idx}_capacity_in_range",
        desc="Venue capacity is between 15,000 and 25,000 seats (inclusive)",
        parent=cap_node,
        critical=True
    )
    cap_sources = coalesce_urls(event.capacity_urls, event.location_urls, event.event_urls)
    cap_range_claim = (
        f"The standard seating capacity of the venue '{event.venue_name or 'UNKNOWN VENUE'}' "
        f"is between 15,000 and 25,000 inclusive."
    )
    await evaluator.verify(
        claim=cap_range_claim,
        node=cap_in_range_node,
        sources=cap_sources if cap_sources else None,
        additional_instruction="Use the provided source(s) to confirm the venue's standard seating capacity lies within the stated range. Minor variations across sources are acceptable."
    )

    # 4.b Capacity figure provided in the answer (existence with digits)
    evaluator.add_custom_node(
        result=has_number(event.venue_capacity),
        id=f"event_{idx}_capacity_figure_provided",
        desc="Provides the specific seating capacity figure",
        parent=cap_node,
        critical=True
    )

    # 4.c Capacity reference grounded (confirm approximate figure)
    cap_ref_node = evaluator.add_leaf(
        id=f"event_{idx}_capacity_reference",
        desc="Provides a reference URL confirming the venue capacity",
        parent=cap_node,
        critical=True
    )
    capacity_text = event.venue_capacity or "UNKNOWN CAPACITY"
    cap_ref_claim = (
        f"The seating capacity of '{event.venue_name or 'UNKNOWN VENUE'}' is approximately {capacity_text}."
    )
    await evaluator.verify(
        claim=cap_ref_claim,
        node=cap_ref_node,
        sources=cap_sources if cap_sources else None,
        additional_instruction="Confirm that the page states the venue's seating capacity close to the figure provided in the answer (allowing rounding like '18k' vs '18,064')."
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
    # Initialize evaluator (root node non-critical to allow partial credit across events)
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

    # Add task constraints as custom info (for transparency)
    evaluator.add_custom_info(
        {
            "time_window_inclusive": ["2025-01-01", "2025-03-31"],
            "venue_capacity_range_inclusive": [15000, 25000],
            "expected_events": 2
        },
        info_type="constraints",
        info_name="task_constraints"
    )

    # Extract structured event info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction"
    )

    # Select first two events; pad if fewer
    events = list(extracted.events[:2])
    while len(events) < 2:
        events.append(EventItem())

    # Build verification subtrees for two events
    for i in range(2):
        await verify_one_event(evaluator, root, events[i], i)

    return evaluator.get_summary()