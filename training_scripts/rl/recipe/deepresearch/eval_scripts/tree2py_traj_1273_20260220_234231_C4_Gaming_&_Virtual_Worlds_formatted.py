import asyncio
import logging
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gaming_events_spring_2026_trip"
TASK_DESCRIPTION = (
    "I am planning a gaming event tour across the United States during the spring season of 2026. "
    "Identify three gaming events that meet ALL of the following requirements:\n\n"
    "1. Each event must take place between March 1, 2026, and June 30, 2026\n"
    "2. The three events must be located in three different U.S. cities\n"
    "3. The events' dates must not overlap, allowing me to physically attend all three\n"
    "4. Each event must span at least two consecutive days\n"
    "5. Each event must be open to the general public (not restricted to industry professionals or invitation-only)\n"
    "6. Each event must be primarily focused on gaming (video games, esports, or game development)\n\n"
    "For each event, provide:\n"
    "- The official event name\n"
    "- The exact dates (start date and end date)\n"
    "- The venue name and complete physical address (including street address, city, state, and ZIP code)\n"
    "- A URL to the event's official website or official announcement confirming the 2026 dates and venue"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    venue_name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    official_url: Optional[str] = None


class EventsExtraction(BaseModel):
    events: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract up to three gaming events presented in the answer. If the answer lists more than three events, extract the first three only (in the same order). If fewer than three events are present, return what is available.

    For each event, extract the following fields exactly as stated in the answer:
    - name: the official event name
    - start_date: the event start date (as written, e.g., "March 12, 2026" or "2026-03-12")
    - end_date: the event end date (as written)
    - venue_name: the venue's official name
    - street_address: the venue's street address (e.g., "123 Main St")
    - city: the venue city
    - state: the venue state (use state abbreviation if provided, otherwise as written)
    - zip_code: the ZIP code (5-digit or ZIP+4)
    - official_url: a single URL that the answer claims is the event’s official website or official announcement confirming the 2026 dates and venue

    Return a JSON object with a single key "events" that is an array of up to 3 objects, each having exactly the fields above. 
    If any field is missing in the answer for a given event, set it to null.

    Special rules for URL fields:
    - Extract only URLs explicitly present in the answer (including markdown links).
    - If a URL is missing a protocol (http/https), prepend "http://".
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
SPRING_START = date(2026, 3, 1)
SPRING_END = date(2026, 6, 30)

_DATE_PATTERNS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%B %d, %Y",  # March 1, 2026
    "%b %d, %Y",  # Mar 1, 2026
    "%d %B %Y",   # 1 March 2026
    "%d %b %Y",   # 1 Mar 2026
]


def _parse_date_safe(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    txt = s.strip()
    for pattern in _DATE_PATTERNS:
        try:
            return datetime.strptime(txt, pattern).date()
        except Exception:
            continue
    # Try to handle cases like "March 12 2026" (no comma)
    try:
        return datetime.strptime(txt.replace(",", ""), "%B %d %Y").date()
    except Exception:
        pass
    try:
        return datetime.strptime(txt.replace(",", ""), "%b %d %Y").date()
    except Exception:
        pass
    return None


def _non_empty(s: Optional[str]) -> bool:
    return s is not None and str(s).strip() != ""


def _event_dates(event: EventItem) -> Tuple[Optional[date], Optional[date]]:
    return _parse_date_safe(event.start_date), _parse_date_safe(event.end_date)


def _in_spring_range(event: EventItem) -> bool:
    sd, ed = _event_dates(event)
    if sd is None or ed is None:
        return False
    if ed < sd:
        return False
    return (SPRING_START <= sd <= SPRING_END) and (SPRING_START <= ed <= SPRING_END)


def _is_multi_day(event: EventItem) -> bool:
    sd, ed = _event_dates(event)
    if sd is None or ed is None:
        return False
    return (ed - sd) >= timedelta(days=1)


def _non_overlapping(e1: EventItem, e2: EventItem) -> bool:
    s1, e1d = _event_dates(e1)
    s2, e2d = _event_dates(e2)
    if s1 is None or e1d is None or s2 is None or e2d is None:
        return False
    # Overlap if s1 <= e2 and s2 <= e1
    overlap = (s1 <= e2d) and (s2 <= e1d)
    return not overlap


def _city_state(event: EventItem) -> Optional[Tuple[str, str]]:
    if not _non_empty(event.city) or not _non_empty(event.state):
        return None
    return (event.city.strip().lower(), event.state.strip().lower())


def _different_city_state(e1: EventItem, e2: EventItem) -> bool:
    c1 = _city_state(e1)
    c2 = _city_state(e2)
    if c1 is None or c2 is None:
        return False
    return c1 != c2


def _full_address_string(event: EventItem) -> Optional[str]:
    if not all([
        _non_empty(event.venue_name),
        _non_empty(event.street_address),
        _non_empty(event.city),
        _non_empty(event.state),
        _non_empty(event.zip_code),
    ]):
        return None
    return f"{event.venue_name}, {event.street_address}, {event.city}, {event.state} {event.zip_code}"


# --------------------------------------------------------------------------- #
# Tree-building helpers                                                       #
# --------------------------------------------------------------------------- #
def _add_event_presence_nodes(evaluator: Evaluator, parent, event: EventItem, idx: int, label: str) -> None:
    """
    Add presence checks for a single event under a non-critical 'Complete' node.
    Each presence check is a critical leaf under this event completeness node.
    """
    # Presence checks: These are critical for the 'complete' node, but the whole 'complete' node is non-critical under root.
    evaluator.add_custom_node(
        result=_non_empty(event.name),
        id=f"event_{idx}_name_present",
        desc=f"{label}: event name is provided",
        parent=parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(event.start_date),
        id=f"event_{idx}_start_date_present",
        desc=f"{label}: start date is provided",
        parent=parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(event.end_date),
        id=f"event_{idx}_end_date_present",
        desc=f"{label}: end date is provided",
        parent=parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(event.venue_name),
        id=f"event_{idx}_venue_name_present",
        desc=f"{label}: venue name is provided",
        parent=parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(event.street_address),
        id=f"event_{idx}_street_present",
        desc=f"{label}: street address is provided",
        parent=parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(event.city),
        id=f"event_{idx}_city_present",
        desc=f"{label}: city is provided",
        parent=parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(event.state),
        id=f"event_{idx}_state_present",
        desc=f"{label}: state is provided",
        parent=parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(event.zip_code),
        id=f"event_{idx}_zip_present",
        desc=f"{label}: ZIP code is provided",
        parent=parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(event.official_url),
        id=f"event_{idx}_official_url_present",
        desc=f"{label}: official URL is provided",
        parent=parent,
        critical=True,
    )


async def _add_top_level_date_range_checks(evaluator: Evaluator, root, events: List[EventItem]) -> None:
    node = evaluator.add_parallel(
        id="Events_In_Date_Range",
        desc="All identified events must take place between March 1, 2026, and June 30, 2026",
        parent=root,
        critical=True,
    )
    for i, ev in enumerate(events):
        result = _in_spring_range(ev)
        evaluator.add_custom_node(
            result=result,
            id=f"event_{i}_in_date_range",
            desc=f"Event #{i + 1} falls entirely within 2026-03-01 to 2026-06-30",
            parent=node,
            critical=True,
        )


async def _add_top_level_non_overlapping_checks(evaluator: Evaluator, root, events: List[EventItem]) -> None:
    node = evaluator.add_parallel(
        id="Events_Non_Overlapping",
        desc="The dates of all identified events must not overlap with each other",
        parent=root,
        critical=True,
    )
    pairs = [(0, 1), (0, 2), (1, 2)]
    for a, b in pairs:
        result = _non_overlapping(events[a], events[b])
        evaluator.add_custom_node(
            result=result,
            id=f"events_{a}_{b}_non_overlapping",
            desc=f"Events #{a + 1} and #{b + 1} do not overlap",
            parent=node,
            critical=True,
        )


async def _add_top_level_different_cities_checks(evaluator: Evaluator, root, events: List[EventItem]) -> None:
    node = evaluator.add_parallel(
        id="Events_In_Different_Cities",
        desc="All identified events must be located in three different U.S. cities",
        parent=root,
        critical=True,
    )
    pairs = [(0, 1), (0, 2), (1, 2)]
    for a, b in pairs:
        result = _different_city_state(events[a], events[b])
        evaluator.add_custom_node(
            result=result,
            id=f"events_{a}_{b}_different_cities",
            desc=f"Events #{a + 1} and #{b + 1} are in different cities/states",
            parent=node,
            critical=True,
        )


async def _add_top_level_multi_day_checks(evaluator: Evaluator, root, events: List[EventItem]) -> None:
    node = evaluator.add_parallel(
        id="All_Events_Multi_Day",
        desc="Each identified event must span at least two consecutive days",
        parent=root,
        critical=True,
    )
    for i, ev in enumerate(events):
        result = _is_multi_day(ev)
        evaluator.add_custom_node(
            result=result,
            id=f"event_{i}_is_multi_day",
            desc=f"Event #{i + 1} spans at least two consecutive days",
            parent=node,
            critical=True,
        )


async def _add_top_level_address_verification(evaluator: Evaluator, root, events: List[EventItem]) -> None:
    node = evaluator.add_parallel(
        id="All_Events_Have_Venue_Addresses",
        desc="Each identified event must have a complete, verifiable physical venue address (street, city, state, ZIP)",
        parent=root,
        critical=True,
    )
    # For each event, verify via the official URL that the venue and address are present (allow minor formatting differences).
    for i, ev in enumerate(events):
        address_str = _full_address_string(ev)
        if _non_empty(ev.official_url) and address_str:
            leaf = evaluator.add_leaf(
                id=f"event_{i}_address_verified",
                desc=f"Event #{i + 1} venue and full address are confirmed on the official page",
                parent=node,
                critical=True,
            )
            claim = (
                f"The official event page lists the venue and full address as: {address_str}. "
                f"Allow minor formatting differences, abbreviations (St./Street, Ave./Avenue), or punctuation; "
                f"but it should clearly match the same venue and postal address."
            )
            await evaluator.verify(
                claim=claim,
                node=leaf,
                sources=ev.official_url,
                additional_instruction=(
                    "Verify that the venue name and the full postal address (street, city, state, ZIP) appear on the page. "
                    "Formatting differences are acceptable as long as the substantive address matches."
                ),
            )
        else:
            # Missing URL or incomplete address -> cannot verify; fail this critical leaf
            evaluator.add_custom_node(
                result=False,
                id=f"event_{i}_address_verified_missing",
                desc=f"Event #{i + 1} venue/address verification is possible (URL and full address provided)",
                parent=node,
                critical=True,
            )


async def _add_top_level_consumer_accessibility(evaluator: Evaluator, root, events: List[EventItem]) -> None:
    node = evaluator.add_parallel(
        id="All_Events_Consumer_Accessible",
        desc="All identified events must be open to the general public (not industry-only or invite-only)",
        parent=root,
        critical=True,
    )
    for i, ev in enumerate(events):
        if _non_empty(ev.official_url):
            leaf = evaluator.add_leaf(
                id=f"event_{i}_public_accessible",
                desc=f"Event #{i + 1} is open to the general public",
                parent=node,
                critical=True,
            )
            claim = (
                "This event is open to the general public (i.e., general attendees can purchase tickets or register; "
                "it is not restricted to industry-only or invitation-only)."
            )
            await evaluator.verify(
                claim=claim,
                node=leaf,
                sources=ev.official_url,
                additional_instruction=(
                    "Confirm that the page indicates public access (e.g., tickets on sale, public registration, "
                    "or 'open to all'). If the page states 'industry-only', 'invite-only', or otherwise restricted, it should FAIL."
                ),
            )
        else:
            evaluator.add_custom_node(
                result=False,
                id=f"event_{i}_public_accessible_missing_url",
                desc=f"Event #{i + 1} public access cannot be verified due to missing official URL",
                parent=node,
                critical=True,
            )


async def _add_top_level_gaming_focus(evaluator: Evaluator, root, events: List[EventItem]) -> None:
    node = evaluator.add_parallel(
        id="All_Events_Gaming_Focused",
        desc="Each identified event must be primarily focused on gaming (video games, esports, or game development)",
        parent=root,
        critical=True,
    )
    for i, ev in enumerate(events):
        if _non_empty(ev.official_url):
            leaf = evaluator.add_leaf(
                id=f"event_{i}_gaming_focused",
                desc=f"Event #{i + 1} is primarily focused on gaming",
                parent=node,
                critical=True,
            )
            claim = (
                "This event is primarily focused on gaming (video games, esports, or game development), "
                "rather than a general tech or entertainment expo."
            )
            await evaluator.verify(
                claim=claim,
                node=leaf,
                sources=ev.official_url,
                additional_instruction=(
                    "Look for clear evidence on the official page that the core theme is gaming (e.g., video games, "
                    "esports tournaments, game developer conference). If gaming is a minor subset of a general expo, it should FAIL."
                ),
            )
        else:
            evaluator.add_custom_node(
                result=False,
                id=f"event_{i}_gaming_focus_missing_url",
                desc=f"Event #{i + 1} gaming focus cannot be verified due to missing official URL",
                parent=node,
                critical=True,
            )


async def _add_top_level_officially_announced(evaluator: Evaluator, root, events: List[EventItem]) -> None:
    node = evaluator.add_parallel(
        id="All_Events_Officially_Announced",
        desc="Each identified event must have officially announced dates for 2026 (not tentative or TBA)",
        parent=root,
        critical=True,
    )
    for i, ev in enumerate(events):
        if _non_empty(ev.official_url):
            leaf = evaluator.add_leaf(
                id=f"event_{i}_officially_announced_2026",
                desc=f"Event #{i + 1} has officially announced 2026 dates (not TBA)",
                parent=node,
                critical=True,
            )
            claim = (
                "The official event page confirms the 2026 dates are announced (explicit dates are shown), "
                "and they are not tentative (not labeled 'TBA', 'to be announced', or 'coming soon')."
            )
            await evaluator.verify(
                claim=claim,
                node=leaf,
                sources=ev.official_url,
                additional_instruction=(
                    "Confirm that the page explicitly lists 2026 dates (e.g., 'March 12–14, 2026'). "
                    "If dates are missing, only say '2026' without specific dates, or marked TBA, it should FAIL."
                ),
            )
        else:
            evaluator.add_custom_node(
                result=False,
                id=f"event_{i}_officially_announced_missing_url",
                desc=f"Event #{i + 1} official 2026 dates cannot be verified due to missing official URL",
                parent=node,
                critical=True,
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
    Evaluate an answer for the Gaming Events Spring 2026 trip planning task.
    """
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
        default_model=model,
    )

    # Extract up to 3 events from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction",
    )

    events: List[EventItem] = list(extracted.events[:3])
    while len(events) < 3:
        events.append(EventItem())

    # Record a bit of custom info for transparency
    evaluator.add_custom_info(
        info={
            "requested_date_range": {"start": str(SPRING_START), "end": str(SPRING_END)},
            "num_events_parsed": len(events),
        },
        info_type="task_constraints",
    )

    # Build per-event completeness nodes (non-critical under root)
    labels = ["First Event Complete", "Second Event Complete", "Third Event Complete"]
    json_descs = [
        "The first gaming event has been correctly identified with all required information: official event name, exact start and end dates, complete venue address (street, city, state, ZIP), and URL to official website or announcement",
        "The second gaming event has been correctly identified with all required information: official event name, exact start and end dates, complete venue address (street, city, state, ZIP), and URL to official website or announcement",
        "The third gaming event has been correctly identified with all required information: official event name, exact start and end dates, complete venue address (street, city, state, ZIP), and URL to official website or announcement",
    ]

    for i, ev in enumerate(events):
        event_node = evaluator.add_parallel(
            id="First_Event_Complete" if i == 0 else ("Second_Event_Complete" if i == 1 else "Third_Event_Complete"),
            desc=json_descs[i],
            parent=root,
            critical=False,
        )
        _add_event_presence_nodes(evaluator, event_node, ev, i, labels[i])

    # Top-level critical constraints
    await _add_top_level_non_overlapping_checks(evaluator, root, events)
    await _add_top_level_different_cities_checks(evaluator, root, events)
    await _add_top_level_date_range_checks(evaluator, root, events)
    await _add_top_level_multi_day_checks(evaluator, root, events)
    await _add_top_level_address_verification(evaluator, root, events)
    await _add_top_level_consumer_accessibility(evaluator, root, events)
    await _add_top_level_gaming_focus(evaluator, root, events)
    await _add_top_level_officially_announced(evaluator, root, events)

    # Return final structured result
    return evaluator.get_summary()