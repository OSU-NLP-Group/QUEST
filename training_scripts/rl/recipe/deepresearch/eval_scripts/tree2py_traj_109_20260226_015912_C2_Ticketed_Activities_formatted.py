import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_events_mar_may_2026"
TASK_DESCRIPTION = (
    "Identify two upcoming ticketed events (concerts, sports events, or theater performances) in California "
    "that are scheduled between March 1 and May 31, 2026. For each event, provide the following information: "
    "Event name and type (concert, sports event, or theater performance), exact date of the event, venue name "
    "and complete street address, venue seating capacity (must be at least 2,000 people), and a direct link "
    "to purchase tickets from an official ticketing platform (such as Ticketmaster, venue website, or other "
    "authorized ticket seller). Both events must have tickets currently available for purchase."
)

DATE_RANGE_START = datetime(2026, 3, 1)
DATE_RANGE_END = datetime(2026, 5, 31)

ALLOWED_EVENT_TYPES = {
    "concert",
    "sports event",
    "sports",
    "theater performance",
    "theatre performance",
    "theater",
    "theatre",
}

# --------------------------------------------------------------------------- #
# Pydantic models                                                             #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    name: Optional[str] = None
    event_type: Optional[str] = None
    date: Optional[str] = None  # Keep as free-form string for extraction robustness
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None  # Expect full street address, including city and state
    venue_capacity: Optional[str] = None  # Keep as string to allow ranges/approximate text
    purchase_url: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)  # other event/venue/ticketing URLs mentioned
    capacity_source_urls: List[str] = Field(default_factory=list)  # URLs specifically supporting capacity


class EventsExtraction(BaseModel):
    events: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract all events mentioned in the answer that are relevant to upcoming ticketed events in California.
    For each event, extract the following fields exactly as they appear in the answer:
      - name: The event's name (e.g., artist/team/play title).
      - event_type: One of ["concert", "sports event", "theater performance"]. If the answer uses similar terms,
        normalize them to the closest option (e.g., "sports", "game" -> "sports event"; "theatre" -> "theater performance").
      - date: The exact date of the event as presented in the answer (e.g., "May 10, 2026" or "2026-05-10").
      - venue_name: Venue name.
      - venue_address: Complete street address including number, street, city, and state (CA). Include ZIP if provided.
      - venue_capacity: The specific seating capacity number as presented (e.g., "18,200" or "about 18,000"). If not present, set to null.
      - purchase_url: A direct link to purchase tickets from an official ticketing platform (Ticketmaster, AXS, venue website, or another authorized seller). If multiple are present, choose one that directly leads to purchase/selection.
      - source_urls: All other URLs cited in the answer that relate to this event or venue (exclude the purchase_url if duplicated).
      - capacity_source_urls: Any URLs that explicitly state the venue's seating capacity (if present).
    Return a JSON object with an "events" array containing these objects. If any field is missing for a given event, set it to null (or [] for lists).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    clean: List[str] = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            clean.append(u)
    return clean


def parse_capacity_to_int(capacity_str: Optional[str]) -> Optional[int]:
    """
    Try to parse a seat capacity integer from a variety of human formats:
    - "18,200", "18.2k", "18k", "Capacity: 2001", "about 20,000", etc.
    Returns None if cannot parse any reasonable integer.
    """
    if not capacity_str:
        return None
    text = capacity_str.strip()

    # 1) e.g., "18.2k", "18k"
    m_k = re.search(r"(\d+(?:\.\d+)?)\s*[kK]\b", text)
    if m_k:
        val = float(m_k.group(1)) * 1000.0
        return int(round(val))

    # 2) numbers with separators e.g., "18,200" or "18.200"
    m_sep = re.search(r"(\d{1,3}(?:[,\.\s]\d{3})+)", text)
    if m_sep:
        digits = re.sub(r"[^\d]", "", m_sep.group(1))
        if digits.isdigit():
            return int(digits)

    # 3) fallback: first integer in the string
    m_int = re.search(r"(\d{3,})", text)  # require >= 3 digits to avoid "18" from "18k" which is handled already
    if m_int:
        try:
            return int(m_int.group(1))
        except Exception:
            return None
    return None


def event_type_valid(event_type: Optional[str]) -> bool:
    if not event_type:
        return False
    t = event_type.strip().lower()
    # Normalize some common aliases
    if t in {"theatre", "theatre performance"}:
        t = "theater performance"
    if t in {"sports", "game", "match"}:
        t = "sports event"
    return t in ALLOWED_EVENT_TYPES


def address_mentions_california(address: Optional[str]) -> bool:
    if not address:
        return False
    a = address.lower()
    return (" california" in a) or (", ca" in a) or (a.strip().endswith(" ca"))


def within_required_date_range(date_str: Optional[str]) -> bool:
    """
    Soft local check to help existence gating. We still verify against sources in a leaf.
    Accepts formats like 'May 10, 2026', '2026-05-10', etc. Returns True if parsed and within range.
    """
    if not date_str:
        return False

    candidates = [date_str.strip()]
    # Try a few parsing strategies
    fmts = ["%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y"]
    for s in candidates:
        for fmt in fmts:
            try:
                dt = datetime.strptime(s, fmt)
                return DATE_RANGE_START <= dt <= DATE_RANGE_END
            except Exception:
                continue
    # If not parsable locally, leave to web verification; return False for gating
    return False


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def verify_single_event(
    evaluator: Evaluator,
    parent_node,
    event: EventItem,
    which: str  # "First" or "Second"
) -> None:
    """
    Build verification sub-tree for a single event under the given parent node.
    """
    # Precompute URL bundles
    all_event_urls = _dedup_urls([event.purchase_url] + (event.source_urls or []))
    venue_urls = _dedup_urls((event.source_urls or []))
    capacity_urls = _dedup_urls((event.capacity_source_urls or []) + venue_urls + ([event.purchase_url] if event.purchase_url else []))

    # 0) Purchase URL existence gate (critical, to enforce source-grounding for ticket checks)
    purchase_url_exists = evaluator.add_custom_node(
        result=bool(event.purchase_url and event.purchase_url.strip()),
        id=f"{which.lower()}_purchase_url_provided",
        desc=f"{which} event has a direct purchase URL provided",
        parent=parent_node,
        critical=True
    )

    # 1) Event details (critical parallel)
    details_node = evaluator.add_parallel(
        id=f"{which.lower()}_event_details",
        desc="Basic event information must be complete and meet all requirements",
        parent=parent_node,
        critical=True
    )

    # 1.a) Event type and name existence/validity (critical custom)
    type_and_name_ok = evaluator.add_custom_node(
        result=bool(event.name and event.name.strip()) and event_type_valid(event.event_type),
        id=f"{which.lower()}_event_type_and_name",
        desc="Event must be a ticketed activity (concert, sports event, or theater performance) with the event name clearly identified",
        parent=details_node,
        critical=True
    )

    # 1.b) Date requirement (critical leaf; verify against purchase/event URLs)
    date_node = evaluator.add_leaf(
        id=f"{which.lower()}_date_requirement",
        desc="Event must be scheduled between March 1 and May 31, 2026, with the exact date provided",
        parent=details_node,
        critical=True
    )
    date_claim = (
        f"The event '{event.name or 'UNKNOWN'}' is scheduled on '{event.date or 'UNKNOWN DATE'}', "
        f"and this date falls between March 1 and May 31, 2026 (inclusive)."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=all_event_urls,
        additional_instruction=(
            "Use the provided ticket purchase or event page URL(s) to confirm the event's exact date. "
            "Then check whether the date is within 2026-03-01 and 2026-05-31 inclusive. "
            "Accept reasonable date formatting variations (e.g., 'Sat, May 9, 2026')."
        ),
    )

    # 1.c) Location requirement (critical leaf; verify event is in California)
    location_node = evaluator.add_leaf(
        id=f"{which.lower()}_location_requirement",
        desc="Event must be located in California",
        parent=details_node,
        critical=True
    )
    location_claim = (
        f"The event '{event.name or 'UNKNOWN'}' will take place in California (CA). "
        f"The venue address given is '{event.venue_address or 'UNKNOWN ADDRESS'}', which is in CA."
    )
    await evaluator.verify(
        claim=location_claim,
        node=location_node,
        sources=all_event_urls,
        additional_instruction=(
            "Check the venue address or location on the source page and confirm that it is within the State of California. "
            "Allow 'CA' or 'California' as valid indicators."
        ),
    )

    # 2) Venue information (critical parallel)
    venue_node = evaluator.add_parallel(
        id=f"{which.lower()}_venue_information",
        desc="Venue information must be complete and meet capacity requirements",
        parent=parent_node,
        critical=True
    )

    # 2.a) Venue identification (critical leaf; name + full address supported by sources)
    venue_ident_node = evaluator.add_leaf(
        id=f"{which.lower()}_venue_identification",
        desc="Venue name and complete street address must be provided",
        parent=venue_node,
        critical=True
    )
    venue_ident_claim = (
        f"The venue for the event '{event.name or 'UNKNOWN'}' is '{event.venue_name or 'UNKNOWN VENUE'}' "
        f"located at '{event.venue_address or 'UNKNOWN ADDRESS'}' (a complete street address)."
    )
    await evaluator.verify(
        claim=venue_ident_claim,
        node=venue_ident_node,
        sources=_dedup_urls([event.purchase_url] + venue_urls),
        additional_instruction=(
            "Verify that the page explicitly shows the venue name and a complete street address "
            "(street number and name, city, state; ZIP code if available). Minor formatting differences are acceptable."
        ),
    )

    # 2.b) Capacity minimum custom check (critical custom)
    parsed_capacity = parse_capacity_to_int(event.venue_capacity)
    capacity_min_check = evaluator.add_custom_node(
        result=(parsed_capacity is not None and parsed_capacity >= 2000),
        id=f"{which.lower()}_capacity_min_check",
        desc="Venue seating capacity (extracted) is at least 2,000 people",
        parent=venue_node,
        critical=True
    )

    # 2.c) Capacity requirement verification (critical leaf; capacity figure supported by sources)
    capacity_node = evaluator.add_leaf(
        id=f"{which.lower()}_capacity_requirement",
        desc="Venue seating capacity must be at least 2,000 people, with the specific capacity number stated",
        parent=venue_node,
        critical=True
    )
    cap_val_str = event.venue_capacity or "UNKNOWN"
    capacity_claim = (
        f"The seating capacity of the venue '{event.venue_name or 'UNKNOWN VENUE'}' is '{cap_val_str}', "
        f"and this capacity is at least 2,000."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_node,
        sources=capacity_urls,
        additional_instruction=(
            "Verify that the cited source page explicitly states the venue's capacity number (or a very close, standard figure) "
            "and that it is at least 2,000. Prefer official venue pages or reputable sources (venue website, official specs, "
            "Ticketmaster/AXS venue info pages)."
        ),
    )

    # 3) Ticket availability (critical leaf; on official purchasing page)
    tix_node = evaluator.add_leaf(
        id=f"{which.lower()}_ticket_availability",
        desc="Tickets must be currently available for purchase with a direct link to an official ticketing platform (such as Ticketmaster, venue website, or other authorized ticket seller)",
        parent=parent_node,
        critical=True
    )
    tix_claim = (
        f"Tickets for the event '{event.name or 'UNKNOWN'}' are currently available for purchase on this page, "
        f"and the page is an official ticketing platform (e.g., Ticketmaster/AXS/venue site/authorized seller)."
    )
    await evaluator.verify(
        claim=tix_claim,
        node=tix_node,
        sources=event.purchase_url or None,
        additional_instruction=(
            "Confirm that the provided URL is a legitimate purchase page for the specified event "
            "(Ticketmaster, AXS, official venue site, or another authorized seller) and that tickets are currently available "
            "(e.g., presence of 'Buy Tickets', 'Find Tickets', seat selection, or other clear purchase options; not 'Sold Out')."
        ),
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
    """
    Evaluate an answer for the California events (March–May 2026) task.
    """
    # Initialize Evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel: two events evaluated independently
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
        extraction_name="events_extraction",
    )

    # Select first two events; pad with empty placeholders if needed
    events: List[EventItem] = list(extracted.events or [])
    if len(events) < 2:
        events = events + [EventItem()] * (2 - len(events))
    else:
        events = events[:2]

    # Create top-level nodes for each event (parallel, non-critical as a group; children inside are critical)
    first_event_node = evaluator.add_parallel(
        id="first_event",
        desc="First ticketed event meeting all requirements",
        parent=root,
        critical=False
    )
    second_event_node = evaluator.add_parallel(
        id="second_event",
        desc="Second ticketed event meeting all requirements",
        parent=root,
        critical=False
    )

    # Build verification sub-trees
    await verify_single_event(evaluator, first_event_node, events[0], "First")
    await verify_single_event(evaluator, second_event_node, events[1], "Second")

    # Return final summary
    return evaluator.get_summary()