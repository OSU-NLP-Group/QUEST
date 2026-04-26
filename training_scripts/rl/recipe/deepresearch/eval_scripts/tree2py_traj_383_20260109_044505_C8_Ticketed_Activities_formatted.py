import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "la_ticketed_activities_2026_q1"
TASK_DESCRIPTION = (
    "Identify 4 distinct ticketed activities in Los Angeles that are taking place during January through March 2026. "
    "The activities should represent diverse types of entertainment (such as theater shows, concerts, museum exhibitions, or other live events). "
    "For each activity, provide the following information: (1) Event name and type, (2) Venue name and complete street address, "
    "(3) Event dates or date range, (4) Ticket pricing information (starting price, price range, or admission fee structure), "
    "(5) A direct link to purchase tickets or reserve admission, and (6) Venue capacity information (if applicable and available). "
    "Ensure that the four activities collectively offer variety in activity types."
)

WINDOW_START = "2026-01-01"
WINDOW_END = "2026-03-31"


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class Address(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None


class EventItem(BaseModel):
    event_name: Optional[str] = None
    event_type: Optional[str] = None
    venue_name: Optional[str] = None
    address: Address = Field(default_factory=Address)
    date_text: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    pricing_info: Optional[str] = None
    ticket_url: Optional[str] = None
    additional_sources: List[str] = Field(default_factory=list)
    capacity_info: Optional[str] = None


class EventsExtraction(BaseModel):
    activities: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract up to six ticketed activities mentioned in the answer that are located in Los Angeles or the greater Los Angeles area and scheduled during January through March 2026.

    For each activity, extract the following fields as strings (do not invent or infer beyond the answer; if absent, return null or empty list where appropriate):
    - event_name: The event name.
    - event_type: A concise type label (e.g., theater show, concert, museum exhibition, festival, comedy show, sports game, etc.).
    - venue_name: The venue name.
    - address.street: Street line (include suite or building info if present).
    - address.city: City.
    - address.state: State abbreviation (e.g., CA).
    - address.zip: ZIP code.
    - date_text: Exact date(s) or date range as stated (free text, e.g., "Jan 22–25, 2026" or "March 3, 2026", etc.).
    - start_date: If a range or known single date is given, provide start date in ISO-like string format if present (e.g., "2026-01-22"); otherwise null.
    - end_date: If a range is given, provide end date in ISO-like string format if present; otherwise null.
    - pricing_info: Explicit ticket pricing information (starting price, price range, or admission fee), if mentioned; otherwise null.
    - ticket_url: A direct URL explicitly provided in the answer to purchase tickets or reserve admission. If multiple URLs are included, choose the most direct ticketing/booking page. If no direct URL is present, return null.
    - additional_sources: An array of any other URLs explicitly present in the answer that support the event details (venue page, event listing, etc.). Only include URLs explicitly present in the answer.
    - capacity_info: Venue capacity/size if stated; otherwise return a text note like "not available" or "N/A" if the answer clearly indicates capacity is not provided.

    SPECIAL RULES FOR URL FIELDS:
    - Extract only URLs explicitly present in the answer, including markdown links. Do not invent or infer URLs.
    - Include full URLs (with http/https). If a URL is missing protocol, prepend http://.
    - ticket_url must be the direct purchase/reservation link if available; otherwise null.

    Return a JSON object with:
    {
      "activities": [EventItem, ...]
    }
    containing each activity with the above fields. If the answer includes more than 4 activities, still extract them all; the evaluator will select the first 4. If fewer than 4 are present, extract what exists.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_key(s: Optional[str]) -> str:
    if not s:
        return ""
    return " ".join(s.lower().strip().split())


def build_full_address(addr: Address) -> str:
    parts = [addr.street or "", addr.city or "", addr.state or "", addr.zip or ""]
    joined = ", ".join([p for p in parts if p])
    return joined


def is_address_complete(addr: Address) -> bool:
    return bool((addr.street or "").strip()) and bool((addr.city or "").strip()) and bool((addr.state or "").strip()) and bool((addr.zip or "").strip())


def gather_sources(item: EventItem) -> List[str]:
    urls = []
    if item.ticket_url and item.ticket_url.strip():
        urls.append(item.ticket_url.strip())
    for u in item.additional_sources:
        if u and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def has_value_or_na(s: Optional[str]) -> bool:
    if not s:
        return False
    val = normalize_key(s)
    return len(val) > 0


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_activity(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    item: EventItem,
) -> None:
    """
    Build verification nodes for one activity and perform checks.
    """
    act_node = evaluator.add_parallel(
        id=f"activity_{idx + 1}",
        desc=f"Activity {idx + 1} meets all per-activity constraints and includes all required fields.",
        parent=parent_node,
        critical=False
    )

    # 1. Event name and type provided
    event_name_and_type_node = evaluator.add_custom_node(
        result=bool((item.event_name or "").strip()) and bool((item.event_type or "").strip()),
        id=f"activity_{idx + 1}_event_name_and_type",
        desc="Event name and event type are provided.",
        parent=act_node,
        critical=True
    )

    # 2. Venue name provided
    venue_name_node = evaluator.add_custom_node(
        result=bool((item.venue_name or "").strip()),
        id=f"activity_{idx + 1}_venue_name",
        desc="Venue name is provided.",
        parent=act_node,
        critical=True
    )

    # 3. Complete venue street address provided
    address_full_node = evaluator.add_custom_node(
        result=is_address_complete(item.address),
        id=f"activity_{idx + 1}_venue_full_street_address",
        desc="Complete venue street address is provided (street, city, state, ZIP).",
        parent=act_node,
        critical=True
    )

    # 4. Location in LA area (verify via sources)
    location_leaf = evaluator.add_leaf(
        id=f"activity_{idx + 1}_location_in_la_area",
        desc="Venue location is in Los Angeles, CA or the greater Los Angeles area.",
        parent=act_node,
        critical=True
    )
    address_text = build_full_address(item.address)
    location_claim = f"The venue located at '{address_text}' is in Los Angeles, CA or the greater Los Angeles area."
    await evaluator.verify(
        claim=location_claim,
        node=location_leaf,
        sources=gather_sources(item),
        additional_instruction=(
            "Confirm using the page's address that the venue is in Los Angeles city or commonly recognized parts of the LA area "
            "(e.g., Santa Monica, Hollywood, West Hollywood, Beverly Hills, Pasadena, Glendale, Burbank, Inglewood, Long Beach, etc.). "
            "If the page shows a city within Los Angeles County or broadly considered greater Los Angeles, count as in LA area."
        )
    )

    # 5. Dates specified
    dates_specified_node = evaluator.add_custom_node(
        result=bool((item.date_text or "").strip()) or bool((item.start_date or "").strip()),
        id=f"activity_{idx + 1}_dates_specified",
        desc="Exact event date(s) or a date range is specified.",
        parent=act_node,
        critical=True
    )

    # 6. Date range overlaps Jan 1–Mar 31, 2026 (verify via sources)
    date_window_leaf = evaluator.add_leaf(
        id=f"activity_{idx + 1}_date_range_overlaps_window",
        desc="Event occurs during or overlaps Jan 1, 2026 through Mar 31, 2026.",
        parent=act_node,
        critical=True
    )
    date_instruction = (
        f"The answer states the event dates as: '{item.date_text or ''}'. "
        f"Verify on the source page that at least one performance/date falls within the window {WINDOW_START} through {WINDOW_END}. "
        "If a multi-day range spans beyond this window, it still passes as long as it overlaps with the window."
    )
    date_claim = f"The event has scheduled date(s) that occur between {WINDOW_START} and {WINDOW_END}."
    await evaluator.verify(
        claim=date_claim,
        node=date_window_leaf,
        sources=gather_sources(item),
        additional_instruction=date_instruction
    )

    # 7. Event confirmed (not tentative)
    confirmed_leaf = evaluator.add_leaf(
        id=f"activity_{idx + 1}_event_confirmed",
        desc="Event is an actual scheduled event with confirmed dates (not tentative/unconfirmed).",
        parent=act_node,
        critical=True
    )
    confirmed_claim = (
        "The event listing indicates confirmed scheduled date(s) (e.g., specific dates/times visible), "
        "and is not tentative (not marked TBA/TBD/waitlist-only/cancelled)."
    )
    await evaluator.verify(
        claim=confirmed_claim,
        node=confirmed_leaf,
        sources=gather_sources(item),
        additional_instruction="Pass if the page clearly shows specific scheduled date(s) or performances and does not indicate TBA/TBD/cancelled."
    )

    # 8. Pricing provided
    pricing_node = evaluator.add_custom_node(
        result=bool((item.pricing_info or "").strip()),
        id=f"activity_{idx + 1}_pricing_provided",
        desc="Explicit ticket pricing information is stated (starting price, price range, or admission fee structure).",
        parent=act_node,
        critical=True
    )

    # 9. Ticket purchase link provided
    ticket_link_node = evaluator.add_custom_node(
        result=bool((item.ticket_url or "").strip()),
        id=f"activity_{idx + 1}_ticket_purchase_link",
        desc="A direct URL to purchase tickets or reserve admission is provided.",
        parent=act_node,
        critical=True
    )

    # 10. Tickets available now (verify via ticket link)
    available_leaf = evaluator.add_leaf(
        id=f"activity_{idx + 1}_tickets_available_now",
        desc="Tickets/reservations are currently available via the provided link (not sold out / not waitlist-only / not unavailable).",
        parent=act_node,
        critical=True
    )
    available_claim = (
        "Tickets or reservations are currently available through this page (e.g., 'Buy Tickets', 'Reserve', 'Add to Cart', with available inventory). "
        "It should not indicate 'Sold Out', 'Waitlist Only', 'Unavailable', or 'Coming Soon'."
    )
    await evaluator.verify(
        claim=available_claim,
        node=available_leaf,
        sources=item.ticket_url or None,
        additional_instruction="Judge availability based on the visible status on the page (including screenshot if provided)."
    )

    # 11. Verifiable sources (official or reputable)
    sources_leaf = evaluator.add_leaf(
        id=f"activity_{idx + 1}_verifiable_sources",
        desc="Provided information is verifiable via official venue websites, ticketing platforms, or reputable event listing sources.",
        parent=act_node,
        critical=True
    )
    sources_claim = (
        "This page is an official or reputable ticketing/venue/event listing source for the activity (e.g., venue website, Ticketmaster, AXS, Eventbrite, SeatGeek, "
        "Universe, official museum site, or a well-known local event guide)."
    )
    # Prefer ticket_url; if not present, use additional_sources list (may be empty -> simple verify)
    srcs_for_verification = item.ticket_url if item.ticket_url else (item.additional_sources if item.additional_sources else None)
    await evaluator.verify(
        claim=sources_claim,
        node=sources_leaf,
        sources=srcs_for_verification,
        additional_instruction="Use the domain and page content to judge whether the source is official/reputable."
    )

    # 12. Venue capacity if applicable (non-critical presence or NA note)
    capacity_node = evaluator.add_custom_node(
        result=has_value_or_na(item.capacity_info),
        id=f"activity_{idx + 1}_venue_capacity_if_applicable",
        desc="Venue capacity/size is provided when applicable and available; otherwise notes that it is not available/applicable.",
        parent=act_node,
        critical=False
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the LA ticketed activities task.
    """
    # Initialize evaluator (root set to non-critical to allow partial scoring on per-activity items)
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

    # Extract activities
    extraction = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction"
    )

    # Select up to 4 activities (filter rule: first 4)
    selected: List[EventItem] = list(extraction.activities[:4])

    # Add padding to ensure we always create 4 activity nodes (placeholders with empty fields)
    while len(selected) < 4:
        selected.append(EventItem())

    # Critical check: set count = 4 and distinctness
    present_names = [normalize_key(e.event_name) for e in selected if (e.event_name or "").strip()]
    distinct_names_ok = len(set(present_names)) == len(present_names) and len(present_names) == 4

    evaluator.add_custom_node(
        result=distinct_names_ok,
        id="set_count_and_distinctness",
        desc="Exactly 4 activities are provided and they are distinct (not the same event repeated).",
        parent=root,
        critical=True
    )

    # Build and verify each activity
    for idx, item in enumerate(selected):
        await verify_activity(evaluator, root, idx, item)

    # Critical check: diversity of types across 4 activities (require not all the same type)
    types = [normalize_key(e.event_type) for e in selected if (e.event_type or "").strip()]
    unique_types = set(types)
    diversity_ok = len(unique_types) >= 2  # Not all the same type

    evaluator.add_custom_node(
        result=diversity_ok,
        id="diversity_of_types",
        desc="Across the 4 activities, the activity types are diverse (not all the same type).",
        parent=root,
        critical=True
    )

    # Add custom info for transparency
    evaluator.add_custom_info(
        info={
            "window_start": WINDOW_START,
            "window_end": WINDOW_END,
            "selected_count": len(selected),
            "distinct_event_names_count": len(set(present_names)),
            "unique_types_count": len(unique_types),
            "unique_types": sorted(list(unique_types)),
        },
        info_type="task_context",
        info_name="evaluation_context"
    )

    return evaluator.get_summary()