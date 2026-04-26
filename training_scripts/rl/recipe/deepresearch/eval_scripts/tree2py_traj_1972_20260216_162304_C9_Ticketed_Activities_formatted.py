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
TASK_ID = "three_major_arena_events_2025"
TASK_DESCRIPTION = (
    "Find three upcoming ticketed events (concerts or professional sports games) occurring between April 1, 2025, and "
    "August 31, 2025, in three different major U.S. cities (New York City, Chicago, or Boston). Each event must be held "
    "at a major indoor arena venue with a minimum seating capacity of 18,000 people. For each event, provide event name, "
    "type, date, time, venue name, city, venue capacity with capacity reference URL, ticketing info with ticket URL, "
    "accessibility (wheelchair seating) with reference URL, age policy, and parking (availability and approximate cost). "
    "Ensure the three events are in three different cities (no repeats among NYC, Chicago, Boston)."
)

ALLOWED_CITIES = {"New York City", "Chicago", "Boston"}
DATE_WINDOW_START = "2025-04-01"
DATE_WINDOW_END = "2025-08-31"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    event_name: Optional[str] = None
    event_type: Optional[str] = None  # "concert" or "professional sports game"
    event_date: Optional[str] = None  # Keep as string to tolerate various formats
    event_time: Optional[str] = None  # Keep as string
    venue_name: Optional[str] = None
    city: Optional[str] = None

    capacity_value: Optional[str] = None  # e.g., "19,000", "19812 for basketball"
    capacity_urls: List[str] = Field(default_factory=list)

    ticket_urls: List[str] = Field(default_factory=list)

    accessibility_info: Optional[str] = None
    accessibility_urls: List[str] = Field(default_factory=list)

    age_policy: Optional[str] = None
    age_policy_url: Optional[str] = None  # optional

    parking_availability: Optional[str] = None
    parking_cost: Optional[str] = None
    parking_url: Optional[str] = None  # optional


class EventsExtraction(BaseModel):
    events: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return (
        "Extract up to three events from the answer. For each event, extract the following fields exactly as stated in the answer; "
        "do not infer or invent any missing facts.\n\n"
        "For each event, provide:\n"
        "- event_name: The specific event name/title (e.g., 'New York Knicks vs. Boston Celtics' or 'Taylor Swift Concert').\n"
        "- event_type: Either 'concert' or 'professional sports game' (use 'professional sports game' for NBA/NHL/MLB/NFL/MLS, etc.).\n"
        "- event_date: The event date string as presented in the answer.\n"
        "- event_time: The event start time string as presented.\n"
        "- venue_name: The venue/arena name.\n"
        "- city: The city string as presented.\n"
        "- capacity_value: The stated seating capacity relevant to this event type (e.g., basketball/hockey/concert capacity).\n"
        "- capacity_urls: A list of URL(s) that the answer cites as evidence for the capacity.\n"
        "- ticket_urls: A list of URL(s) where tickets can be purchased or were available (as cited in the answer).\n"
        "- accessibility_info: The wheelchair-accessible seating info text (if provided).\n"
        "- accessibility_urls: A list of URL(s) supporting the accessibility information (as cited).\n"
        "- age_policy: The age requirement/recommendation text (if provided).\n"
        "- age_policy_url: A URL for age policy if cited (optional; null if not provided).\n"
        "- parking_availability: The parking availability information text (if provided).\n"
        "- parking_cost: The approximate parking cost text (if provided).\n"
        "- parking_url: A URL for parking info if cited (optional; null if not provided).\n\n"
        "Rules for URL extraction:\n"
        "1) Only extract URLs explicitly present in the answer text (including markdown links). Do not infer or invent URLs.\n"
        "2) Include full URLs with http:// or https://; if missing protocol, prepend http://.\n"
        "3) If a required URL list is not present in the answer, return an empty list for that field.\n\n"
        "Return a JSON object with a top-level 'events' array (length 1–3). If the answer lists more than three, keep only the first three. "
        "If fewer, return only those available."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s or ""

def _first3_events(extracted: EventsExtraction) -> List[EventItem]:
    events = extracted.events[:3]
    # Pad to exactly 3 to keep evaluation structure stable
    while len(events) < 3:
        events.append(EventItem())
    return events

def _urls_or_none(urls: List[str]) -> Optional[List[str]]:
    return urls if urls else None

# --------------------------------------------------------------------------- #
# Verification for a single event                                             #
# --------------------------------------------------------------------------- #
async def verify_one_event(evaluator: Evaluator, parent_node, ev: EventItem, idx: int) -> None:
    i = idx + 1
    ev_node = evaluator.add_parallel(
        id=f"Event_{i}",
        desc=f"{['First','Second','Third'][idx]} event meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Ticketing (critical) - do URL presence first, then availability
    ticketing_node = evaluator.add_sequential(
        id=f"Event_{i}_Ticketing",
        desc="Ticketing information and availability",
        parent=ev_node,
        critical=True
    )
    # Ticket URL presence
    ticket_url_present = evaluator.add_custom_node(
        result=bool(ev.ticket_urls),
        id=f"Event_{i}_Ticket_URL",
        desc="A reference URL to where tickets can be purchased or obtained",
        parent=ticketing_node,
        critical=True
    )
    # Ticket availability supported by ticket URL(s)
    ticket_avail_leaf = evaluator.add_leaf(
        id=f"Event_{i}_Ticket_Availability",
        desc="Evidence that tickets are available or were available for purchase",
        parent=ticketing_node,
        critical=True
    )
    await evaluator.verify(
        claim="Tickets for this event are or were available for purchase on the cited ticketing page(s). "
              "If currently sold out, it still counts as 'were available'.",
        node=ticket_avail_leaf,
        sources=ev.ticket_urls,
        additional_instruction="Check the ticket page(s) for purchase options, availability states (including 'sold out'), or past availability indicators."
    )

    # Capacity verification (critical) - URL presence first, then numeric threshold
    capacity_node = evaluator.add_sequential(
        id=f"Event_{i}_Capacity_Verification",
        desc="Verification that the venue meets the minimum capacity requirement",
        parent=ev_node,
        critical=True
    )
    cap_url_present = evaluator.add_custom_node(
        result=bool(ev.capacity_urls),
        id=f"Event_{i}_Capacity_Reference",
        desc="A reference URL supporting the capacity information is provided",
        parent=capacity_node,
        critical=True
    )
    cap_value_leaf = evaluator.add_leaf(
        id=f"Event_{i}_Capacity_Value",
        desc="The venue's seating capacity for this event type is stated and is at least 18,000",
        parent=capacity_node,
        critical=True
    )
    cap_claim = (
        f"The seating capacity for this event type at the venue '{_safe(ev.venue_name)}' is at least 18,000. "
        f"The stated capacity is '{_safe(ev.capacity_value)}' (if provided). "
        "If multiple capacities (e.g., basketball vs. hockey vs. concerts) are listed, consider the one relevant to this event."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_value_leaf,
        sources=ev.capacity_urls,
        additional_instruction="Confirm from the capacity reference page that the relevant configuration capacity is >= 18,000."
    )

    # Accessibility (non-critical) - URL presence first, then content
    access_node = evaluator.add_sequential(
        id=f"Event_{i}_Accessibility",
        desc="Wheelchair-accessible seating availability",
        parent=ev_node,
        critical=False
    )
    access_url_present = evaluator.add_custom_node(
        result=bool(ev.accessibility_urls),
        id=f"Event_{i}_Accessibility_URL",
        desc="A reference URL supporting the accessibility information",
        parent=access_node,
        critical=False
    )
    access_leaf = evaluator.add_leaf(
        id=f"Event_{i}_Wheelchair_Seating",
        desc="Information confirms that wheelchair-accessible seating is available at the venue",
        parent=access_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"Wheelchair-accessible seating is available at '{_safe(ev.venue_name)}' for this event, as per the cited accessibility page(s).",
        node=access_leaf,
        sources=ev.accessibility_urls,
        additional_instruction="Venue-wide accessibility pages that specify ADA/wheelchair seating availability are acceptable."
    )

    # Event Name (critical) - verify against ticket page(s)
    name_leaf = evaluator.add_leaf(
        id=f"Event_{i}_Name",
        desc="The specific name or description of the event (e.g., 'New York Knicks vs. Boston Celtics' or 'Taylor Swift Concert')",
        parent=ev_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event title shown on the ticket page corresponds to '{_safe(ev.event_name)}' or an equivalent naming.",
        node=name_leaf,
        sources=ev.ticket_urls,
        additional_instruction="Allow minor formatting differences, abbreviations, or sponsor prefixes/suffixes."
    )

    # Event Type (critical) - verify category on ticket page(s)
    type_leaf = evaluator.add_leaf(
        id=f"Event_{i}_Type",
        desc="The event is identified as either a concert or a professional sports game",
        parent=ev_node,
        critical=True
    )
    if (_safe(ev.event_type)).strip().lower() == "concert":
        type_claim = "This event is a concert."
    else:
        # default to professional sports game if not strictly 'concert'
        type_claim = "This event is a professional sports game (e.g., NBA/NHL/MLB/NFL/MLS)."
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=ev.ticket_urls,
        additional_instruction="Use the ticket page to infer whether it is a concert or a pro sports game. "
                               "If it's an NBA/NHL/etc. matchup, treat it as professional sports."
    )

    # Event Date (critical) - verify date and within window
    date_leaf = evaluator.add_leaf(
        id=f"Event_{i}_Date",
        desc="The specific date of the event is provided and falls between April 1, 2025, and August 31, 2025",
        parent=ev_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event date is '{_safe(ev.event_date)}', and it falls between {DATE_WINDOW_START} and {DATE_WINDOW_END} inclusive.",
        node=date_leaf,
        sources=ev.ticket_urls,
        additional_instruction="Check the ticket page for the exact event date, and confirm it lies within the given 2025 date range."
    )

    # Event Time (critical) - verify start time
    time_leaf = evaluator.add_leaf(
        id=f"Event_{i}_Time",
        desc="The scheduled start time of the event is provided",
        parent=ev_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event's scheduled start time is '{_safe(ev.event_time)}'.",
        node=time_leaf,
        sources=ev.ticket_urls,
        additional_instruction="Match the start time string on the ticket page; allow minor formatting differences (e.g., AM/PM vs 24-hour)."
    )

    # Venue Name (critical) - verify against ticket page(s)
    venue_leaf = evaluator.add_leaf(
        id=f"Event_{i}_Venue_Name",
        desc="The specific name of the venue where the event takes place is provided",
        parent=ev_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue for this event is '{_safe(ev.venue_name)}'.",
        node=venue_leaf,
        sources=ev.ticket_urls,
        additional_instruction="Check the ticket page for the venue/arena name; allow sponsor name variants."
    )

    # City (critical) - verify city and membership in allowed set
    city_leaf = evaluator.add_leaf(
        id=f"Event_{i}_City",
        desc="The city is identified as one of: New York City, Chicago, or Boston",
        parent=ev_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event takes place in '{_safe(ev.city)}', which is one of New York City, Chicago, or Boston.",
        node=city_leaf,
        sources=ev.ticket_urls,
        additional_instruction="Confirm city per the ticket page. Accept borough references for NYC that clearly indicate New York City."
    )

    # Age Policy (non-critical) - presence check (documented)
    age_leaf = evaluator.add_custom_node(
        result=bool((_safe(ev.age_policy)).strip()),
        id=f"Event_{i}_Age_Policy",
        desc="The age requirement or recommendation for attending the event is documented",
        parent=ev_node,
        critical=False
    )

    # Parking (non-critical, parallel): availability and cost presence
    parking_node = evaluator.add_parallel(
        id=f"Event_{i}_Parking",
        desc="Parking availability and cost information",
        parent=ev_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=bool((_safe(ev.parking_availability)).strip()),
        id=f"Event_{i}_Parking_Availability",
        desc="Information confirms that parking is available at or near the venue",
        parent=parking_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=bool((_safe(ev.parking_cost)).strip()),
        id=f"Event_{i}_Parking_Cost",
        desc="Approximate parking cost is provided if available",
        parent=parking_node,
        critical=False
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    # Initialize evaluator/root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # parallel at root as per rubric
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

    # Extract structured event info
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction"
    )

    # Keep first three events and pad if needed
    events = _first3_events(extracted)

    # Add ground-truth style info (constraints)
    evaluator.add_ground_truth({
        "allowed_cities": list(ALLOWED_CITIES),
        "date_window_start": DATE_WINDOW_START,
        "date_window_end": DATE_WINDOW_END,
        "min_capacity": 18000
    }, gt_type="constraints")

    # Build verification tree for three events
    for idx, ev in enumerate(events):
        await verify_one_event(evaluator, root, ev, idx)

    # City Diversity check (critical)
    # Compute uniqueness and membership
    cities = [(_safe(ev.city)).strip() for ev in events if (_safe(ev.city)).strip()]
    unique_cities = set([c.lower() for c in cities])  # case-insensitive uniqueness
    membership_ok = all(any(c.lower() == ac.lower() for ac in ALLOWED_CITIES) for c in cities)
    diversity_ok = (len(cities) == 3) and (len(unique_cities) == 3) and membership_ok

    evaluator.add_custom_node(
        result=diversity_ok,
        id="City_Diversity",
        desc="The three events are located in three different cities (no city appears more than once)",
        parent=root,
        critical=True
    )

    # Return evaluation summary
    return evaluator.get_summary()