import asyncio
import logging
from typing import Any, List, Optional, Dict, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nyc_upcoming_events_mar18_may18_2026"
TASK_DESCRIPTION = (
    "Find four different upcoming live performance events in New York City scheduled within the next two months "
    "(between March 18, 2026 and May 18, 2026). Each event must be at a different venue and represent a different type "
    "of live performance (such as a concert, comedy show, Broadway show, or sporting event). For each event, provide: "
    "Event name; Venue name and complete address; Date and start time; A link to purchase tickets."
)

DATE_RANGE_START = "March 18, 2026"
DATE_RANGE_END = "May 18, 2026"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    event_name: Optional[str] = None
    event_type: Optional[str] = None  # e.g., concert, comedy, Broadway show, sporting event, etc.
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None  # Full address string as given in the answer
    date: Optional[str] = None          # Keep as free text as provided (e.g., "April 12, 2026")
    start_time: Optional[str] = None    # Keep as free text (e.g., "7:30 PM")
    ticket_url: Optional[str] = None    # Direct link to purchase tickets
    reference_urls: List[str] = Field(default_factory=list)  # URLs cited to support the event details


class EventsExtraction(BaseModel):
    events: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract up to the first four (4) upcoming live performance events described in the answer.
    For each event, extract the following fields exactly as they appear:
    - event_name: the event's title/name
    - event_type: the kind of live performance (e.g., concert, comedy show, Broadway show/musical/play, sporting event, dance, opera, classical, festival, talk/lecture). If unclear, extract the label given by the answer.
    - venue_name: name of the venue
    - venue_address: complete address string as stated in the answer (include city, state, and any ZIP if present)
    - date: the calendar date of the event (e.g., "April 12, 2026")
    - start_time: the start time (e.g., "7:30 PM")
    - ticket_url: a URL that the answer claims is a direct page to purchase tickets for this event (extract only if explicitly present)
    - reference_urls: an array of any URLs the answer cites that describe or list the event (e.g., venue page, event listing, official announcement). These should be distinct from the ticket_url when possible, and must be explicitly present in the answer text.

    Rules:
    - Only extract URLs that are explicitly present in the answer text (including within markdown links). Do not invent or infer any URLs.
    - If the answer provides more than four events, keep only the first four in order of appearance.
    - If any field is missing for an event, set it to null. For reference_urls, use an empty list if none are provided.
    - Do not alter or normalize values; extract them as-is from the answer.
    
    Return a JSON object with a single field:
    {
      "events": [ {event_1_fields}, {event_2_fields}, {event_3_fields}, {event_4_fields} ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper normalization utilities                                              #
# --------------------------------------------------------------------------- #
def _normalize_text(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return " ".join(s.strip().lower().split())


def _normalize_venue(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    txt = _normalize_text(s)
    if not txt:
        return None
    # Simple de-article for some venues (e.g., "the beacon theatre" -> "beacon theatre")
    if txt.startswith("the "):
        txt = txt[4:]
    return txt


def _normalize_event_type(s: Optional[str]) -> Optional[str]:
    """
    Normalize event types into broad categories to avoid trivial string differences
    from defeating the "different types" constraint.
    """
    if not s:
        return None
    t = _normalize_text(s)

    # Broad buckets
    if any(k in t for k in ["broadway", "musical", "play", "theatre", "theater"]):
        return "theater"
    if any(k in t for k in ["comedy", "stand-up", "standup"]):
        return "comedy"
    if any(k in t for k in ["concert", "live music", "band", "tour", "gig"]):
        return "concert"
    if any(k in t for k in [
        "sport", "game", "match", "vs", "vs.", "fc", "knicks", "nets", "yankees", "mets",
        "rangers", "islanders", "giants", "jets", "liberty", "red bulls", "nycfc", "mls", "nba", "nhl", "mlb", "nfl"
    ]):
        return "sporting"
    if any(k in t for k in ["dance", "ballet", "choreography"]):
        return "dance"
    if "opera" in t:
        return "opera"
    if any(k in t for k in ["symphony", "orchestra", "philharmonic", "classical"]):
        return "classical"
    if "festival" in t:
        return "festival"
    if any(k in t for k in ["talk", "lecture", "conversation"]):
        return "talk"

    # Fallback to the raw lowered type string if no bucket matched
    return t


# --------------------------------------------------------------------------- #
# Event verification logic                                                    #
# --------------------------------------------------------------------------- #
class EventVerifyResult(BaseModel):
    norm_type: Optional[str] = None
    norm_venue: Optional[str] = None
    date_range_node_id: Optional[str] = None
    in_nyc_node_id: Optional[str] = None


async def verify_single_event(
    evaluator: Evaluator,
    parent_node,
    event: EventItem,
    index_1_based: int,
) -> EventVerifyResult:
    """
    Build a sequential verification sub-tree for a single event.
    Returns normalized metadata and node IDs useful for global constraints.
    """
    # Create a sequential node for this event to respect logical gating
    event_node = evaluator.add_sequential(
        id=f"event_{index_1_based}",
        desc=f"Event #{index_1_based}: Provided with complete details and valid ticket purchasing link",
        parent=parent_node,
        critical=False  # Non-critical per-item; global constraints will be critical
    )

    # 1) Existence/Completeness check (critical within this event)
    has_required = (
        bool(event and event.event_name and event.venue_name and event.venue_address and
             event.date and event.start_time and event.ticket_url and event.ticket_url.strip())
        and (event.reference_urls is not None and len(event.reference_urls) > 0)
    )

    evaluator.add_custom_node(
        result=has_required,
        id=f"event_{index_1_based}_required_fields",
        desc=f"Event #{index_1_based}: All required fields present (name, type, venue, full address, date, start time, ticket link, and at least one reference URL)",
        parent=event_node,
        critical=True
    )

    # 2) Event info is supported by at least one reference URL (name + venue)
    info_node = evaluator.add_leaf(
        id=f"event_{index_1_based}_info_supported",
        desc=f"Event #{index_1_based}: Reference URL(s) show the event name and venue",
        parent=event_node,
        critical=True
    )
    info_claim = (
        f"The webpage clearly references an upcoming event named '{event.event_name}' at the venue '{event.venue_name}'."
    )
    await evaluator.verify(
        claim=info_claim,
        node=info_node,
        sources=event.reference_urls,
        additional_instruction="Allow minor variations in formatting/casing for names. If multiple URLs are provided, any one that clearly supports the statement suffices."
    )

    # 3) Date falls within the required range (use references)
    date_node = evaluator.add_leaf(
        id=f"event_{index_1_based}_date_in_range",
        desc=f"Event #{index_1_based}: Scheduled between {DATE_RANGE_START} and {DATE_RANGE_END} (inclusive)",
        parent=event_node,
        critical=True
    )
    date_claim = (
        f"The event is scheduled between {DATE_RANGE_START} and {DATE_RANGE_END} (inclusive)."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=event.reference_urls,
        additional_instruction=f"Use the event page details to determine if the event date is within the inclusive range {DATE_RANGE_START} to {DATE_RANGE_END}."
    )

    # 4) Event is in New York City (boroughs acceptable) (use references)
    nyc_node = evaluator.add_leaf(
        id=f"event_{index_1_based}_in_nyc",
        desc=f"Event #{index_1_based}: Located in New York City (any of the five boroughs)",
        parent=event_node,
        critical=True
    )
    nyc_claim = (
        "The venue for this event is located in New York City (i.e., in Manhattan, Brooklyn, Queens, the Bronx, or Staten Island)."
    )
    await evaluator.verify(
        claim=nyc_claim,
        node=nyc_node,
        sources=event.reference_urls,
        additional_instruction="Confirm from the venue or event page that the city is New York, NY, or that the venue is explicitly in one of NYC's five boroughs."
    )

    # 5) Ticket URL is a purchase page (primary/official source preferred)
    tix_page_node = evaluator.add_leaf(
        id=f"event_{index_1_based}_ticket_purchase_page",
        desc=f"Event #{index_1_based}: Ticket URL is an actual purchase page for this event",
        parent=event_node,
        critical=True
    )
    tix_purchase_claim = (
        f"This URL is a page where tickets can be purchased for the event '{event.event_name}' at '{event.venue_name}' on {event.date}."
    )
    await evaluator.verify(
        claim=tix_purchase_claim,
        node=tix_page_node,
        sources=event.ticket_url,
        additional_instruction=(
            "Look for clear purchase intent indicators like 'Buy Tickets', selectable seats, ticket selections, or checkout steps. "
            "The page should be for the specified event (allow minor name/time formatting variations)."
        )
    )

    # 6) Ticket URL is an official/authorized source (venue site or primary ticketing vendor)
    tix_official_node = evaluator.add_leaf(
        id=f"event_{index_1_based}_ticket_official",
        desc=f"Event #{index_1_based}: Ticket URL is an official or authorized primary ticketing source",
        parent=event_node,
        critical=True
    )
    tix_official_claim = (
        "This ticket page is an official/authorized primary ticketing source (e.g., the venue's own site or a primary vendor like Ticketmaster, AXS, Telecharge, SeatGeek, or TodayTix)."
    )
    await evaluator.verify(
        claim=tix_official_claim,
        node=tix_official_node,
        sources=event.ticket_url,
        additional_instruction=(
            "Prefer venue-operated domains or primary ticketing providers (Ticketmaster, AXS, Telecharge, SeatGeek, TodayTix). "
            "If the page is a clear reseller/secondary marketplace (e.g., StubHub, Vivid Seats), mark as not official."
        )
    )

    # 7) Event type classification is consistent with references
    type_node = evaluator.add_leaf(
        id=f"event_{index_1_based}_type_consistent",
        desc=f"Event #{index_1_based}: Labeled type matches the event shown on references",
        parent=event_node,
        critical=True
    )
    type_claim = f"This event is a '{event.event_type}' type of live performance."
    await evaluator.verify(
        claim=type_claim,
        node=type_node,
        sources=event.reference_urls,
        additional_instruction="Allow reasonable category synonyms (e.g., Broadway show ~ theater; stand-up ~ comedy; symphony ~ classical)."
    )

    return EventVerifyResult(
        norm_type=_normalize_event_type(event.event_type),
        norm_venue=_normalize_venue(event.venue_name),
        date_range_node_id=date_node.id,
        in_nyc_node_id=nyc_node.id
    )


# --------------------------------------------------------------------------- #
# Global constraints verification (aggregate custom nodes)                    #
# --------------------------------------------------------------------------- #
def add_global_constraint_nodes(
    evaluator: Evaluator,
    root,
    events: List[EventItem],
    per_event_meta: List[EventVerifyResult],
) -> None:
    # All Events Different Types
    norm_types = [m.norm_type for m in per_event_meta]
    all_types_valid = all(t is not None and len(t) > 0 for t in norm_types)
    types_unique = len(set(t for t in norm_types if t)) == 4 if all_types_valid else False
    evaluator.add_custom_node(
        result=all_types_valid and types_unique,
        id="All_Events_Different_Types",
        desc="All four events represent different types of live performances (e.g., concert, comedy show, Broadway/theater, sporting, etc.).",
        parent=root,
        critical=True
    )

    # All Events Different Venues
    norm_venues = [m.norm_venue for m in per_event_meta]
    all_venues_valid = all(v is not None and len(v) > 0 for v in norm_venues)
    venues_unique = len(set(v for v in norm_venues if v)) == 4 if all_venues_valid else False
    evaluator.add_custom_node(
        result=all_venues_valid and venues_unique,
        id="All_Events_Different_Venues",
        desc="All four events are scheduled at different venues (no two events at the same venue).",
        parent=root,
        critical=True
    )

    # All Events Within Date Range (aggregate based on per-event date check results)
    date_nodes_ok = True
    for m in per_event_meta:
        node = evaluator.find_node(m.date_range_node_id) if m.date_range_node_id else None
        if not node or node.status != "passed":
            date_nodes_ok = False
            break
    evaluator.add_custom_node(
        result=date_nodes_ok,
        id="All_Events_Within_Date_Range",
        desc=f"All four events are scheduled between {DATE_RANGE_START} and {DATE_RANGE_END} (inclusive).",
        parent=root,
        critical=True
    )

    # All Events In NYC (aggregate based on per-event NYC check results)
    nyc_nodes_ok = True
    for m in per_event_meta:
        node = evaluator.find_node(m.in_nyc_node_id) if m.in_nyc_node_id else None
        if not node or node.status != "passed":
            nyc_nodes_ok = False
            break
    evaluator.add_custom_node(
        result=nyc_nodes_ok,
        id="All_Events_In_NYC",
        desc="All four events are located in New York City.",
        parent=root,
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
    Evaluate an answer for the NYC upcoming live events task.
    """
    # Initialize evaluator (root non-critical to allow mixture of critical/non-critical children)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Find four different upcoming live performance events in NYC (Mar 18, 2026 - May 18, 2026), all with distinct types and venues, including full details and a valid ticket purchase link.",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract events list from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction"
    )

    # Keep only the first 4 events; pad with empty placeholders if fewer than 4
    events: List[EventItem] = list(extracted.events[:4])
    while len(events) < 4:
        events.append(EventItem())

    # Add GT/context info for clarity
    evaluator.add_ground_truth({
        "required_count": 4,
        "date_range_inclusive": [DATE_RANGE_START, DATE_RANGE_END],
        "location": "New York City (any of five boroughs)",
        "distinct_requirements": ["different event types", "different venues"]
    }, gt_type="task_requirements")

    # Build per-event verification subtrees
    per_event_meta: List[EventVerifyResult] = []
    for i in range(4):
        meta = await verify_single_event(
            evaluator=evaluator,
            parent_node=root,
            event=events[i],
            index_1_based=i + 1
        )
        per_event_meta.append(meta)

    # Add global constraints as critical custom nodes
    add_global_constraint_nodes(
        evaluator=evaluator,
        root=root,
        events=events,
        per_event_meta=per_event_meta
    )

    # Return evaluation summary
    return evaluator.get_summary()