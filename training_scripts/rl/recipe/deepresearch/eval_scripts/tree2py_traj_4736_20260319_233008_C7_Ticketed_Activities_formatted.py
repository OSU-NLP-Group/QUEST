import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nyc_events_march_2026"
TASK_DESCRIPTION = """
Find three distinct ticketed entertainment events in New York City scheduled during March 2026. The three events must be of different categories (such as concert, Broadway show, sporting event, comedy show, or theater performance). For each event, provide: (1) Venue Name: The specific name of the venue where the event takes place; (2) Venue Address: The complete physical address of the venue; (3) Date and Time: The exact date and time the event is scheduled; (4) Ticket Purchase Link: A direct URL to purchase tickets from an official ticketing platform (such as Ticketmaster, StubHub, SeatGeek, Broadway.com, or the venue's box office); (5) Public Transportation: Specific subway lines or bus routes that serve the venue. All three events must be scheduled between March 1 and March 31, 2026.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    """One event extracted from the answer."""
    title: Optional[str] = None
    category: Optional[str] = None  # e.g., concert, Broadway show, sporting event, comedy, theater
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    date_time: Optional[str] = None  # free form, e.g., "March 12, 2026, 7:30 PM"
    ticket_url: Optional[str] = None
    transportation: List[str] = Field(default_factory=list)  # list of subway lines or bus routes, e.g., ["A", "C", "E", "M7"]
    source_urls: List[str] = Field(default_factory=list)  # any extra URLs cited (venue page, MTA, etc.)


class EventsExtraction(BaseModel):
    """List of events extracted from the answer."""
    events: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract all ticketed entertainment events mentioned in the answer. For each event, return an object with:
    - title: The event/show/game name, exactly as stated.
    - category: The entertainment category (e.g., concert, Broadway show, sporting event, comedy show, theater performance).
    - venue_name: The specific venue name.
    - venue_address: The full street address (street number, street name, city/borough, state abbreviation, and ZIP code if available).
    - date_time: The exact scheduled date and time as presented (e.g., "March 12, 2026, 7:30 PM").
    - ticket_url: A direct URL to purchase tickets (Ticketmaster, StubHub, SeatGeek, Broadway.com, or the venue’s official box office).
    - transportation: A list of specific public transit options serving the venue (subway lines like "A/C/E", "1/2/3", or bus routes like "M7", "B41"). Split multiple lines or routes into separate list items when obvious.
    - source_urls: Any additional URLs cited that support the venue info, address, date/time, or transportation (e.g., venue “Getting Here” page, MTA route pages). Include as many as are explicitly provided.

    Rules:
    1) Extract strictly from the answer text; do not invent information.
    2) If a value is missing, set it to null (or empty array for lists).
    3) Normalize URLs to include protocol; if missing, prepend "http://".
    4) Do not deduplicate or filter events here; just extract faithfully as written.

    Return a JSON object with one field:
    {
      "events": [ ... up to all events mentioned ... ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _collect_sources(evt: EventItem) -> List[str]:
    uniq = []
    for u in [evt.ticket_url, *(evt.source_urls or [])]:
        if _non_empty(u) and u not in uniq:
            uniq.append(u)  # keep order
    return uniq


def _safe_join(items: List[str]) -> str:
    return ", ".join([x for x in items if _non_empty(x)])


# --------------------------------------------------------------------------- #
# Verification for a single event                                             #
# --------------------------------------------------------------------------- #
async def verify_event(evaluator: Evaluator, parent_node, evt: EventItem, idx_zero_based: int) -> None:
    idx = idx_zero_based + 1
    event_node = evaluator.add_parallel(
        id=f"Event_{idx}",
        desc=f"{['First','Second','Third'][idx_zero_based]} ticketed entertainment event in New York City during March 2026",
        parent=parent_node,
        critical=False
    )

    sources = _collect_sources(evt)
    title_snippet = evt.title or f"Event #{idx}"
    venue_snippet = evt.venue_name or "the specified venue"

    # ---------------- Venue Name ----------------
    venue_name_group = evaluator.add_parallel(
        id=f"Event_{idx}_Venue_Name",
        desc=f"The {['first','second','third'][idx_zero_based]} event must identify a specific venue name where the event takes place",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(evt.venue_name),
        id=f"Event_{idx}_Venue_Name_Provided",
        desc=f"Venue name is provided for event #{idx}",
        parent=venue_name_group,
        critical=True
    )
    vn_supported = evaluator.add_leaf(
        id=f"Event_{idx}_Venue_Name_Supported",
        desc=f"Venue name is supported by cited sources for event #{idx}",
        parent=venue_name_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ticket or official source page for '{title_snippet}' confirms the venue is '{evt.venue_name}'.",
        node=vn_supported,
        sources=sources,
        additional_instruction="Allow minor naming variants. Confirm that the page explicitly lists this venue for this event."
    )

    # ---------------- Venue Address (+ NYC check) ----------------
    venue_addr_group = evaluator.add_parallel(
        id=f"Event_{idx}_Venue_Address",
        desc=f"The {['first','second','third'][idx_zero_based]} event must provide the complete physical address of the venue",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(evt.venue_address),
        id=f"Event_{idx}_Venue_Address_Provided",
        desc=f"Venue address is provided for event #{idx}",
        parent=venue_addr_group,
        critical=True
    )
    va_supported = evaluator.add_leaf(
        id=f"Event_{idx}_Venue_Address_Supported",
        desc=f"Venue address is supported by cited sources for event #{idx}",
        parent=venue_addr_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official source(s) list the venue address as '{evt.venue_address}'.",
        node=va_supported,
        sources=sources,
        additional_instruction="Accept minor formatting differences (e.g., abbreviations). Address must clearly match the venue."
    )
    in_nyc = evaluator.add_leaf(
        id=f"Event_{idx}_In_NYC",
        desc=f"Event #{idx} venue is in New York City (one of the five boroughs)",
        parent=venue_addr_group,
        critical=True
    )
    await evaluator.verify(
        claim="This event's venue is in New York City (Manhattan, Brooklyn, Queens, The Bronx, or Staten Island).",
        node=in_nyc,
        sources=sources,
        additional_instruction="Confirm the address or location references a NYC borough or 'New York, NY'. Borough-level city names (e.g., Brooklyn, NY) count as NYC."
    )

    # ---------------- Date & Time (must be in March 2026) ----------------
    date_time_group = evaluator.add_parallel(
        id=f"Event_{idx}_Date_Time",
        desc=f"The {['first','second','third'][idx_zero_based]} event must specify the exact date/time and the date must fall within March 2026 (March 1–31, 2026)",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(evt.date_time),
        id=f"Event_{idx}_Date_Time_Provided",
        desc=f"Exact date/time is provided for event #{idx}",
        parent=date_time_group,
        critical=True
    )
    dt_supported = evaluator.add_leaf(
        id=f"Event_{idx}_Date_Time_Supported",
        desc=f"Date/time and March 2026 constraint are supported by cited sources for event #{idx}",
        parent=date_time_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event '{title_snippet}' is scheduled on '{evt.date_time}', and the date is between March 1 and March 31, 2026 (inclusive) in New York local time.",
        node=dt_supported,
        sources=sources,
        additional_instruction="Check the event page's listed date/time. Accept standard variants (e.g., 'Mar' vs 'March'). Ensure the date is within March 2026."
    )

    # ---------------- Ticket Link (official purchase page) ----------------
    ticket_group = evaluator.add_parallel(
        id=f"Event_{idx}_Ticket_Link",
        desc=f"The {['first','second','third'][idx_zero_based]} event must include a direct URL to purchase tickets from an official platform",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(evt.ticket_url),
        id=f"Event_{idx}_Ticket_Link_Provided",
        desc=f"Direct ticket purchase URL is provided for event #{idx}",
        parent=ticket_group,
        critical=True
    )
    tl_official = evaluator.add_leaf(
        id=f"Event_{idx}_Ticket_Link_Official_Purchase_Page",
        desc=f"Ticket URL is a direct purchase page on an official ticketing platform or the venue's box office for event #{idx}",
        parent=ticket_group,
        critical=True
    )
    await evaluator.verify(
        claim="This URL is a direct ticket purchase page for the event, hosted by Ticketmaster, StubHub, SeatGeek, Broadway.com, or the venue’s official site (box office).",
        node=tl_official,
        sources=evt.ticket_url,
        additional_instruction="Open the page and confirm it allows buying tickets for the specified event (buy/select seats/see tickets)."
    )

    # ---------------- Transportation (subway/bus) ----------------
    transportation_group = evaluator.add_parallel(
        id=f"Event_{idx}_Transportation",
        desc=f"The {['first','second','third'][idx_zero_based]} event must provide specific public transportation (subway lines or bus routes) that serve the venue",
        parent=event_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(evt.transportation and len(evt.transportation) > 0),
        id=f"Event_{idx}_Transportation_Provided",
        desc=f"Specific public transit options are provided for event #{idx}",
        parent=transportation_group,
        critical=True
    )
    tr_supported = evaluator.add_leaf(
        id=f"Event_{idx}_Transportation_Supported",
        desc=f"Provided public transit options are supported by cited sources for event #{idx}",
        parent=transportation_group,
        critical=True
    )
    transit_list = _safe_join(evt.transportation or [])
    await evaluator.verify(
        claim=f"At least one of the following lines/routes serves {venue_snippet}: {transit_list}.",
        node=tr_supported,
        sources=sources,
        additional_instruction=(
            "Use venue 'Getting Here' pages, official MTA pages, or authoritative sources. "
            "A correct subset is acceptable (not all lines must be listed), but at least one provided line/route must be accurate."
        )
    )


# --------------------------------------------------------------------------- #
# Category diversity verification                                             #
# --------------------------------------------------------------------------- #
async def verify_category_diversity(evaluator: Evaluator, parent_node, events: List[EventItem]) -> None:
    """
    Critical check: the three events must represent different entertainment categories.
    Use an LLM simple verification to consider semantic equivalence (e.g., 'Broadway show' vs 'theater').
    """
    categories = [e.category or "" for e in events[:3]]
    cat_text = "; ".join([f"Event {i+1}: {c}" for i, c in enumerate(categories)])
    node = evaluator.add_leaf(
        id="Category_Diversity",
        desc="The three events must represent different entertainment categories (such as concert, Broadway show, sporting event, comedy show, or theater performance)",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The three categories are: {cat_text}. "
            "Determine if all three are genuinely different categories. "
            "If two are essentially the same (e.g., 'Broadway show' and 'theater performance' both count as theater), mark as NOT different."
        ),
        node=node,
        additional_instruction=(
            "Judge by semantics, not surface strings. Consider the examples given. "
            "Return Correct only if all three are distinct categories."
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
    Evaluate an answer for the NYC March 2026 events task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Events evaluated independently; category diversity is a separate critical gate
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

    # Extract events mentioned in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction"
    )

    # Normalize to exactly three items (pad with empty if fewer; trim if more)
    events: List[EventItem] = list(extracted.events or [])
    if len(events) < 3:
        events = events + [EventItem() for _ in range(3 - len(events))]
    else:
        events = events[:3]

    # Add category diversity check (critical)
    await verify_category_diversity(evaluator, root, events)

    # Build verification subtrees for each event
    for i, evt in enumerate(events):
        await verify_event(evaluator, root, evt, i)

    # Return the full evaluator summary
    return evaluator.get_summary()