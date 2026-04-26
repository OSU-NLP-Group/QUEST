import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nyc_weekend_events_april_2026"
TASK_DESCRIPTION = (
    "Identify two different types of ticketed entertainment events in New York City that are scheduled to occur on "
    "weekends (Friday, Saturday, or Sunday) in April 2026. The first event must be a Broadway show performed at a "
    "theater with a seating capacity of at least 1,000 seats. The second event must be either a concert or a major "
    "sporting event (not a Broadway show). For each event, provide the following information: event name and specific "
    "date, venue name and its seating capacity (for the Broadway show), a reference URL to the official event or venue "
    "website showing the event details, and a reference URL to where tickets can be purchased or where an official "
    "waitlist can be joined."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    """Structured representation of a single event extracted from the answer."""
    name: Optional[str] = None
    date: Optional[str] = None  # Keep as string to be robust to diverse formats; we'll parse programmatically
    venue_name: Optional[str] = None
    seating_capacity: Optional[str] = None  # Only required for Broadway show (Event 1)
    official_event_url: Optional[str] = None
    ticket_url: Optional[str] = None
    category: Optional[str] = None  # e.g., "Broadway", "concert", "sporting", etc.
    city: Optional[str] = None
    state: Optional[str] = None


class EventsExtraction(BaseModel):
    """Two-event bundle: Event 1 must be Broadway; Event 2 must be concert or sporting."""
    event1: Optional[EventItem] = None
    event2: Optional[EventItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return (
        "From the provided answer text, extract exactly two events that the answer proposes:\n"
        "Event 1 must be a Broadway show in New York City with a specific performance date in April 2026 on a weekend "
        "(Friday, Saturday, or Sunday). Event 1 must also include the theater's seating capacity number.\n"
        "Event 2 must be either a concert or a major sporting event (not a Broadway show) in New York City with a "
        "specific date in April 2026 on a weekend.\n\n"
        "For each event, extract the following fields as available explicitly in the answer:\n"
        "- name: The event name or show title.\n"
        "- date: The specific date as stated (keep the original string; do not reformat).\n"
        "- venue_name: The venue or theater name.\n"
        "- seating_capacity: The seating capacity number string (only for Event 1 if provided; otherwise null if not present).\n"
        "- official_event_url: A URL to the official event or venue page showing details about the event.\n"
        "- ticket_url: A URL where tickets can be purchased or an official waitlist can be joined.\n"
        "- category: The event category as stated (e.g., 'Broadway', 'concert', 'sporting', etc.).\n"
        "- city: City string if mentioned (e.g., 'New York', 'New York City', 'Brooklyn', etc.).\n"
        "- state: State string if mentioned (e.g., 'NY').\n\n"
        "Selection rules when the answer includes multiple events:\n"
        "1) Choose the first Broadway show that meets the weekend-in-April-2026 criteria for Event 1; if multiple dates "
        "are listed, choose one weekend date in April 2026 explicitly mentioned.\n"
        "2) Choose the first event that is clearly a concert or major sporting event (not Broadway) for Event 2 with a "
        "weekend date in April 2026.\n"
        "3) Only extract URLs that are explicitly present in the answer text (plain URLs or markdown links). If a URL is "
        "not explicitly given, set the field to null.\n"
        "4) If any field is missing in the answer for an event, set it to null.\n"
        "Return a JSON object containing two nested objects: 'event1' and 'event2'."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _collect_sources(event: EventItem) -> List[str]:
    """Collect available sources for verification (official + ticket)."""
    urls: List[str] = []
    if event.official_event_url and event.official_event_url.strip():
        urls.append(event.official_event_url.strip())
    if event.ticket_url and event.ticket_url.strip():
        urls.append(event.ticket_url.strip())
    return urls


def _parse_capacity_to_int(cap_str: Optional[str]) -> Optional[int]:
    """Try to parse a seating capacity integer from a free-form string."""
    if not cap_str:
        return None
    s = cap_str.strip().lower()

    # Handle '1.2k' or '1k' formats
    k_match = re.search(r'(\d+(?:\.\d+)?)\s*k\b', s)
    if k_match:
        try:
            val = float(k_match.group(1))
            return int(round(val * 1000))
        except Exception:
            pass

    # Remove commas and non-digit except whitespace
    # Find all numbers; choose the largest to be safe if multiple present
    nums = re.findall(r'\d{1,6}', re.sub(r'[^\d]', ' ', s))
    if not nums:
        return None
    try:
        # Choose the maximum number as capacity (defensive)
        return max(int(n) for n in nums)
    except Exception:
        return None


def _parse_first_april_2026_date(date_str: Optional[str]) -> Optional[datetime]:
    """Extract a concrete April 2026 date from a free-form string and return a datetime."""
    if not date_str:
        return None
    s = date_str.strip()

    # Common explicit formats
    fmts = [
        "%B %d, %Y",        # April 12, 2026
        "%b %d, %Y",        # Apr 12, 2026
        "%A, %B %d, %Y",    # Sunday, April 12, 2026
        "%a, %b %d, %Y",    # Sun, Apr 12, 2026
        "%A, %b %d, %Y",
        "%a, %B %d, %Y",
        "%Y-%m-%d",         # 2026-04-12
        "%m/%d/%Y",         # 04/12/2026
        "%m/%d/%y",         # 04/12/26
    ]

    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.year == 2026 and dt.month == 4:
                return dt
        except Exception:
            pass

    # Regex fallback: "April 12, 2026"
    m = re.search(r'\bApril\s+(\d{1,2}),\s*2026\b', s, flags=re.IGNORECASE)
    if m:
        day = int(m.group(1))
        try:
            return datetime(2026, 4, day)
        except Exception:
            return None

    # Regex fallback: "2026-04-12"
    m = re.search(r'\b2026-04-(\d{2})\b', s)
    if m:
        day = int(m.group(1))
        try:
            return datetime(2026, 4, day)
        except Exception:
            return None

    # Regex fallback: "04/12/2026"
    m = re.search(r'\b04/(\d{1,2})/2026\b', s)
    if m:
        day = int(m.group(1))
        try:
            return datetime(2026, 4, day)
        except Exception:
            return None

    return None


def _is_weekend(dt: Optional[datetime]) -> bool:
    """Friday (4), Saturday (5), Sunday (6)."""
    if dt is None:
        return False
    return dt.weekday() in (4, 5, 6)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_event_1_broadway(evaluator: Evaluator, parent_node, event: EventItem) -> None:
    """
    Build the verification subtree for Event 1 (Broadway show) according to the rubric.
    Order of operations ensures official URL checks occur before dependent content checks.
    """
    # Event_1 parallel node (non-critical container)
    event1_node = evaluator.add_parallel(
        id="Event_1",
        desc="First entertainment event (Broadway show) meeting all specified criteria",
        parent=parent_node,
        critical=False,
    )

    # 1) Event details existence (critical parallel)
    details_node = evaluator.add_parallel(
        id="Event_1_Event_Details",
        desc="Required event information must be provided",
        parent=event1_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(event.name and event.name.strip()),
        id="Event_1_Name",
        desc="Event name must be provided",
        parent=details_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(event.date and event.date.strip()),
        id="Event_1_Specific_Date",
        desc="Specific date of the event must be provided",
        parent=details_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(event.venue_name and event.venue_name.strip()),
        id="Event_1_Venue_Name",
        desc="Venue name must be provided",
        parent=details_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(event.seating_capacity and re.search(r'\d', event.seating_capacity or "")),
        id="Event_1_Seating_Capacity_Value",
        desc="The venue's seating capacity number must be provided",
        parent=details_node,
        critical=True,
    )

    # 2) Official Event URL (critical)
    if event.official_event_url and event.official_event_url.strip():
        ev_url_node = evaluator.add_leaf(
            id="Event_1_Official_Event_URL",
            desc="Provide a reference URL to the official event or venue website showing the event details",
            parent=event1_node,
            critical=True,
        )
        claim = (
            f"This page is an official event or venue website showing event details for '{event.name or ''}' "
            f"on '{event.date or ''}' at '{event.venue_name or ''}'."
        )
        await evaluator.verify(
            claim=claim,
            node=ev_url_node,
            sources=event.official_event_url,
            additional_instruction=(
                "Confirm that the page is an official site (production or venue) and clearly shows event information "
                "such as title, schedule/date, and venue details. Third-party ticket aggregators alone are not sufficient."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Event_1_Official_Event_URL",
            desc="Provide a reference URL to the official event or venue website showing the event details",
            parent=event1_node,
            critical=True,
        )

    # 3) Ticket purchase/waitlist URL (critical)
    if event.ticket_url and event.ticket_url.strip():
        tix_node = evaluator.add_leaf(
            id="Event_1_Ticket_Purchase_URL",
            desc="Provide a reference URL to where tickets can be purchased or where an official waitlist can be joined",
            parent=event1_node,
            critical=True,
        )
        claim = (
            f"This page allows purchasing tickets or joining an official waitlist for '{event.name or ''}' "
            f"on '{event.date or ''}' at '{event.venue_name or ''}'."
        )
        await evaluator.verify(
            claim=claim,
            node=tix_node,
            sources=event.ticket_url,
            additional_instruction=(
                "Confirm the page includes ticketing or official waitlist functionality or links such as "
                "'Buy Tickets', 'Tickets', or 'Join Waitlist'."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Event_1_Ticket_Purchase_URL",
            desc="Provide a reference URL to where tickets can be purchased or where an official waitlist can be joined",
            parent=event1_node,
            critical=True,
        )

    # 4) Basic requirements (critical parallel): Location, Date, Category
    basic_node = evaluator.add_parallel(
        id="Event_1_Basic_Requirements",
        desc="Basic location, timing, and category requirements for Event 1",
        parent=event1_node,
        critical=True,
    )

    # 4.1 Location in NYC (verify via sources)
    loc_node = evaluator.add_leaf(
        id="Event_1_Location",
        desc="Event must be located in New York City",
        parent=basic_node,
        critical=True,
    )
    loc_claim = (
        f"The event '{event.name or ''}' takes place in New York City (NYC), including any of the five boroughs."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=_collect_sources(event),
        additional_instruction=(
            "Verify from the page that the venue is within New York City. Accept boroughs (Manhattan, Brooklyn, Queens, "
            "Bronx, Staten Island) or 'New York, NY'. Do not accept outside NYC municipalities (e.g., Newark, NJ; "
            "Westbury, NY; Nassau County)."
        ),
    )

    # 4.2 Date on a weekend in April 2026 (custom check)
    dt = _parse_first_april_2026_date(event.date)
    evaluator.add_custom_node(
        result=(dt is not None and _is_weekend(dt)),
        id="Event_1_Date",
        desc="Event must occur on a Friday, Saturday, or Sunday in April 2026",
        parent=basic_node,
        critical=True,
    )

    # 4.3 Category is Broadway (verify via sources)
    cat_node = evaluator.add_leaf(
        id="Event_1_Category",
        desc="Event must be a Broadway show",
        parent=basic_node,
        critical=True,
    )
    cat_claim = (
        f"The event '{event.name or ''}' is a Broadway show performed at a Broadway theater (not Off-Broadway)."
    )
    await evaluator.verify(
        claim=cat_claim,
        node=cat_node,
        sources=_collect_sources(event),
        additional_instruction=(
            "Confirm that this is a Broadway production (recognized Broadway theater, not Off-Broadway) based on the "
            "official event or venue page."
        ),
    )

    # 5) Venue capacity constraint >= 1,000 seats (custom check)
    capacity_int = _parse_capacity_to_int(event.seating_capacity)
    evaluator.add_custom_node(
        result=(capacity_int is not None and capacity_int >= 1000),
        id="Event_1_Venue_Capacity",
        desc="The theater venue must have a seating capacity of at least 1,000 seats",
        parent=event1_node,
        critical=True,
    )


async def verify_event_2_non_broadway(evaluator: Evaluator, parent_node, event: EventItem) -> None:
    """
    Build the verification subtree for Event 2 (concert or major sporting event, not Broadway).
    """
    # Event_2 parallel node (non-critical container)
    event2_node = evaluator.add_parallel(
        id="Event_2",
        desc="Second entertainment event (concert or sporting event) meeting all specified criteria",
        parent=parent_node,
        critical=False,
    )

    # 1) Event details existence (critical parallel)
    details_node = evaluator.add_parallel(
        id="Event_2_Event_Details",
        desc="Required event information must be provided",
        parent=event2_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(event.name and event.name.strip()),
        id="Event_2_Name",
        desc="Event name must be provided",
        parent=details_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(event.date and event.date.strip()),
        id="Event_2_Specific_Date",
        desc="Specific date of the event must be provided",
        parent=details_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(event.venue_name and event.venue_name.strip()),
        id="Event_2_Venue_Name",
        desc="Venue name must be provided",
        parent=details_node,
        critical=True,
    )

    # 2) Official Event URL (critical)
    if event.official_event_url and event.official_event_url.strip():
        ev_url_node = evaluator.add_leaf(
            id="Event_2_Official_Event_URL",
            desc="Provide a reference URL to the official event or venue website showing the event details",
            parent=event2_node,
            critical=True,
        )
        claim = (
            f"This page is an official event or venue website showing event details for '{event.name or ''}' "
            f"on '{event.date or ''}' at '{event.venue_name or ''}'."
        )
        await evaluator.verify(
            claim=claim,
            node=ev_url_node,
            sources=event.official_event_url,
            additional_instruction=(
                "Confirm that the page is an official site (artist, team, league, or venue) and clearly shows event "
                "information such as title/opponent, schedule/date, and venue details. Third-party ticket aggregators "
                "alone are not sufficient."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Event_2_Official_Event_URL",
            desc="Provide a reference URL to the official event or venue website showing the event details",
            parent=event2_node,
            critical=True,
        )

    # 3) Ticket purchase/waitlist URL (critical)
    if event.ticket_url and event.ticket_url.strip():
        tix_node = evaluator.add_leaf(
            id="Event_2_Ticket_Purchase_URL",
            desc="Provide a reference URL to where tickets can be purchased or where an official waitlist can be joined",
            parent=event2_node,
            critical=True,
        )
        claim = (
            f"This page allows purchasing tickets or joining an official waitlist for '{event.name or ''}' "
            f"on '{event.date or ''}' at '{event.venue_name or ''}'."
        )
        await evaluator.verify(
            claim=claim,
            node=tix_node,
            sources=event.ticket_url,
            additional_instruction=(
                "Confirm the page includes ticketing or official waitlist functionality or links such as "
                "'Buy Tickets', 'Tickets', or 'Join Waitlist'."
            ),
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Event_2_Ticket_Purchase_URL",
            desc="Provide a reference URL to where tickets can be purchased or where an official waitlist can be joined",
            parent=event2_node,
            critical=True,
        )

    # 4) Basic requirements (critical parallel): Location, Date, Category
    basic_node = evaluator.add_parallel(
        id="Event_2_Basic_Requirements",
        desc="Basic location, timing, and category requirements for Event 2",
        parent=event2_node,
        critical=True,
    )

    # 4.1 Location in NYC (verify via sources)
    loc_node = evaluator.add_leaf(
        id="Event_2_Location",
        desc="Event must be located in New York City",
        parent=basic_node,
        critical=True,
    )
    loc_claim = f"The event '{event.name or ''}' takes place in New York City (NYC), including any of the five boroughs."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=_collect_sources(event),
        additional_instruction=(
            "Verify from the page that the venue is within New York City. Accept boroughs (Manhattan, Brooklyn, Queens, "
            "Bronx, Staten Island) or 'New York, NY'. Do not accept outside NYC municipalities (e.g., Newark, NJ; "
            "Westbury, NY; Nassau County)."
        ),
    )

    # 4.2 Date on a weekend in April 2026 (custom check)
    dt = _parse_first_april_2026_date(event.date)
    evaluator.add_custom_node(
        result=(dt is not None and _is_weekend(dt)),
        id="Event_2_Date",
        desc="Event must occur on a Friday, Saturday, or Sunday in April 2026",
        parent=basic_node,
        critical=True,
    )

    # 4.3 Category: concert or sporting, not Broadway (verify via sources)
    cat_node = evaluator.add_leaf(
        id="Event_2_Category",
        desc="Event must be either a concert or a major sporting event (not a Broadway show)",
        parent=basic_node,
        critical=True,
    )
    cat_claim = (
        f"The event '{event.name or ''}' is either a concert (live music performance) or a major sporting event, "
        f"and it is not a Broadway show."
    )
    await evaluator.verify(
        claim=cat_claim,
        node=cat_node,
        sources=_collect_sources(event),
        additional_instruction=(
            "Check the official event or venue page to confirm the event type is a concert (artist/band performance) "
            "or a major sporting event (e.g., NBA, NHL, MLB, MLS, NFL, top-tier leagues) and explicitly not a Broadway show."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Entry point for evaluating the agent's answer against the rubric using the Mind2Web2 framework.
    """
    # Initialize evaluator with a parallel root (two events evaluated independently)
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

    # Extract structured event info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction",
    )

    # Build verification subtrees
    await verify_event_1_broadway(
        evaluator=evaluator,
        parent_node=root,
        event=extracted.event1 or EventItem(),
    )

    await verify_event_2_non_broadway(
        evaluator=evaluator,
        parent_node=root,
        event=extracted.event2 or EventItem(),
    )

    # Return standardized summary
    return evaluator.get_summary()