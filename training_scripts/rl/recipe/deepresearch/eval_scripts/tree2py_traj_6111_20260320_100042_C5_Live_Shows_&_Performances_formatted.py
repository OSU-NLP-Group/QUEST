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
TASK_ID = "chi_apr_2026_live_events"
TASK_DESCRIPTION = (
    "I'm planning to visit Chicago in April 2026 and want to attend some live entertainment events during my stay. "
    "Find four different upcoming live performances or shows (such as concerts, comedy shows, theater productions, or other live entertainment events) "
    "scheduled in Chicago, Illinois during April 2026. For each performance, provide: (1) The name of the performer, show, or event, "
    "(2) The venue name and complete address in Chicago, (3) The specific date and start time of the performance, and "
    "(4) A direct link to an official ticketing website where I can view or purchase tickets for that specific performance."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    name: Optional[str] = None
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None  # Full street address if provided in the answer
    date: Optional[str] = None           # Keep as free text as answers may vary in format, e.g., "April 12, 2026"
    time: Optional[str] = None           # Free text, e.g., "7:30 PM"
    ticket_urls: List[str] = Field(default_factory=list)  # One or more URLs provided for tickets


class EventsExtraction(BaseModel):
    events: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract up to four live performances or shows mentioned in the answer. Each item must be scheduled in Chicago, Illinois during April 2026.
    For each extracted performance, provide the following fields exactly as stated in the answer:
    - name: The performer, show, or event name.
    - venue_name: The venue name.
    - venue_address: The complete venue address as written in the answer (include street, city, state, and postal code if present).
    - date: The specific performance date string (e.g., "April 12, 2026").
    - time: The start time string (e.g., "7:30 PM").
    - ticket_urls: An array of direct URL(s) to official ticketing or venue pages for that specific performance. Extract only URLs explicitly shown in the answer. If multiple are given, include all.
    
    Rules:
    - Extract only from the answer text; do not invent or infer any information.
    - If more than four performances are given, include only the first four in the original order.
    - If any field is missing for a performance, set it to null (or an empty array for ticket_urls).
    - Only include URLs that are explicitly visible in the answer text (plain URLs or URLs inside markdown links).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Verification logic per performance                                          #
# --------------------------------------------------------------------------- #
async def verify_performance(
    evaluator: Evaluator,
    parent_node,
    ev: EventItem,
    index: int,
) -> None:
    perf_id = f"Performance_{index + 1}"
    perf_node = evaluator.add_parallel(
        id=perf_id,
        desc=f"{['First','Second','Third','Fourth'][index]} live performance in Chicago during April 2026 with complete information",
        parent=parent_node,
        critical=False,
    )

    # Details group (critical): presence + validity + source-backed checks
    details_node = evaluator.add_parallel(
        id=f"{perf_id}_Details",
        desc="The performance includes: (1) the name of the performer, show, or event; (2) the venue name and complete address in Chicago; and (3) the specific date within April 2026 (April 1-30, 2026) and start time",
        parent=perf_node,
        critical=True,
    )

    # Ticket link group (critical): URL presence and official/specificity checks
    ticket_node = evaluator.add_parallel(
        id=f"{perf_id}_Ticket_Link",
        desc="A direct URL link to an official ticketing platform (such as Ticketmaster, Live Nation, venue website, or other authorized ticket seller) where tickets for this specific performance can be viewed or purchased",
        parent=perf_node,
        critical=True,
    )

    # ---------------- Presence checks (critical) ----------------
    name_present = evaluator.add_custom_node(
        result=bool(ev.name and ev.name.strip()),
        id=f"{perf_id}_name_present",
        desc=f"{perf_id}: Event name is provided",
        parent=details_node,
        critical=True,
    )
    venue_present = evaluator.add_custom_node(
        result=bool(ev.venue_name and ev.venue_name.strip()),
        id=f"{perf_id}_venue_present",
        desc=f"{perf_id}: Venue name is provided",
        parent=details_node,
        critical=True,
    )
    address_present = evaluator.add_custom_node(
        result=bool(ev.venue_address and ev.venue_address.strip()),
        id=f"{perf_id}_address_present",
        desc=f"{perf_id}: Complete venue address is provided",
        parent=details_node,
        critical=True,
    )
    date_present = evaluator.add_custom_node(
        result=bool(ev.date and ev.date.strip()),
        id=f"{perf_id}_date_present",
        desc=f"{perf_id}: Date is provided",
        parent=details_node,
        critical=True,
    )
    time_present = evaluator.add_custom_node(
        result=bool(ev.time and ev.time.strip()),
        id=f"{perf_id}_time_present",
        desc=f"{perf_id}: Start time is provided",
        parent=details_node,
        critical=True,
    )

    # Ensure we have at least one ticketing URL to ground detail verifications
    urls_available_for_details = evaluator.add_custom_node(
        result=bool(ev.ticket_urls),
        id=f"{perf_id}_urls_available_for_details",
        desc=f"{perf_id}: At least one ticketing URL is provided to verify details",
        parent=details_node,
        critical=True,
    )

    # ---------------- Simple logical constraint (date within April 2026) ----------------
    date_in_april_leaf = evaluator.add_leaf(
        id=f"{perf_id}_date_in_april_2026",
        desc=f"{perf_id}: The provided date string is a date in April 2026 (April 1–30, 2026)",
        parent=details_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided date string '{_safe(ev.date)}' refers to a calendar date between April 1 and April 30, 2026 (inclusive).",
        node=date_in_april_leaf,
        additional_instruction="Judge based on the date string itself; allow common date formats (e.g., 'Apr 5, 2026', 'April 05, 2026').",
    )

    # ---------------- Source-backed checks against the ticketing page(s) ----------------
    # Build claims for verification via the provided ticket URLs
    claims_and_nodes: List[tuple] = []

    # 1) Name aligns with ticket page
    name_supported = evaluator.add_leaf(
        id=f"{perf_id}_name_supported_by_url",
        desc=f"{perf_id}: The performer/show name matches what is shown on the ticketing page",
        parent=details_node,
        critical=True,
    )
    claims_and_nodes.append((
        f"The ticketing webpage clearly indicates the performer/show/event as '{_safe(ev.name)}' (allow minor naming or formatting variations).",
        ev.ticket_urls,
        name_supported,
        "Accept minor differences, such as casing, abbreviations, subtitle/taglines, or 'Tour' suffixes."
    ))

    # 2) Venue aligns with ticket page
    venue_supported = evaluator.add_leaf(
        id=f"{perf_id}_venue_supported_by_url",
        desc=f"{perf_id}: The venue name matches what is shown on the ticketing page",
        parent=details_node,
        critical=True,
    )
    claims_and_nodes.append((
        f"The ticketing page shows the venue as '{_safe(ev.venue_name)}' or an equivalent/abbreviated form.",
        ev.ticket_urls,
        venue_supported,
        "Focus on the venue field on the page; allow minor variations like 'The Chicago Theatre' vs 'Chicago Theatre'."
    ))

    # 3) Address/city aligns with ticket page (Chicago, IL)
    address_supported = evaluator.add_leaf(
        id=f"{perf_id}_address_supported_by_url",
        desc=f"{perf_id}: The venue address on the page is in Chicago, Illinois and is consistent with the provided address",
        parent=details_node,
        critical=True,
    )
    claims_and_nodes.append((
        f"The ticketing page indicates the venue is located in Chicago, Illinois and the address is consistent with '{_safe(ev.venue_address)}' (formatting variations acceptable).",
        ev.ticket_urls,
        address_supported,
        "Look for 'Chicago, IL' or 'Chicago, Illinois' and compare the street address as reasonably formatted."
    ))

    # 4) Date aligns with ticket page (and is in April 2026)
    date_supported = evaluator.add_leaf(
        id=f"{perf_id}_date_supported_by_url",
        desc=f"{perf_id}: The performance date shown on the page matches the provided date and is in April 2026",
        parent=details_node,
        critical=True,
    )
    claims_and_nodes.append((
        f"The ticketing page shows the performance date in April 2026 and it matches '{_safe(ev.date)}' (allow day-of-week or minor format differences).",
        ev.ticket_urls,
        date_supported,
        "Accept common date formatting differences; ensure the month is April and the year is 2026."
    ))

    # 5) Time aligns with ticket page
    time_supported = evaluator.add_leaf(
        id=f"{perf_id}_time_supported_by_url",
        desc=f"{perf_id}: The performance start time on the page matches the provided time",
        parent=details_node,
        critical=True,
    )
    claims_and_nodes.append((
        f"The ticketing page shows a start time that matches '{_safe(ev.time)}' (allow minor formatting like '7:30PM' vs '7:30 PM').",
        ev.ticket_urls,
        time_supported,
        "Accept minor spacing/casing differences and common time representations."
    ))

    # Ticket link checks
    ticket_url_provided = evaluator.add_custom_node(
        result=bool(ev.ticket_urls),
        id=f"{perf_id}_ticket_url_provided",
        desc=f"{perf_id}: At least one direct ticketing URL is provided",
        parent=ticket_node,
        critical=True,
    )

    ticket_official_leaf = evaluator.add_leaf(
        id=f"{perf_id}_ticket_url_official",
        desc=f"{perf_id}: The provided URL(s) point to an official ticketing or the venue's official event page (not an unauthorized reseller)",
        parent=ticket_node,
        critical=True,
    )
    ticket_specific_leaf = evaluator.add_leaf(
        id=f"{perf_id}_ticket_url_specific_performance",
        desc=f"{perf_id}: The provided URL(s) are specific to this exact performance (date/venue), not just a generic listing",
        parent=ticket_node,
        critical=True,
    )

    # Prepare all verifications for batch where applicable
    to_verify_batch: List[tuple] = []

    # Detail verifications rely on URLs; they will be automatically skipped if any critical presence check or
    # the 'urls_available_for_details' node failed because they are critical siblings under the same parent node.
    for claim, sources, node, add_ins in claims_and_nodes:
        to_verify_batch.append((claim, sources, node, add_ins))

    # Ticket URL officialness and specificity — will be skipped if ticket_url_provided failed (critical sibling)
    to_verify_batch.append((
        "This page is an official primary ticketing platform (e.g., Ticketmaster, Live Nation, AXS, Etix, Dice, Eventbrite when used by the organizer), "
        "or the venue's own official event page where tickets can be viewed or purchased. It is not an unauthorized reseller like StubHub or Vivid Seats.",
        ev.ticket_urls,
        ticket_official_leaf,
        "Prefer pages that show 'Buy Tickets', seat selection, or direct purchase. Venue-branded event pages count as official."
    ))
    to_verify_batch.append((
        f"The page corresponds to the specific performance for '{_safe(ev.name)}' at '{_safe(ev.venue_name)}' on '{_safe(ev.date)}' in Chicago, IL (i.e., not just a generic schedule index).",
        ev.ticket_urls,
        ticket_specific_leaf,
        "Look for the exact date/time or a selected occurrence for the Chicago, IL performance."
    ))

    # Execute batch verifications in parallel
    await evaluator.batch_verify(to_verify_batch)


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

    # Extract up to four performances from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="extracted_performances",
    )

    events = list(extracted.events or [])
    # Keep only the first 4 events; if fewer than 4, pad with empty placeholders
    events = events[:4]
    while len(events) < 4:
        events.append(EventItem())

    # Add constraint info as ground truth context (not strict expected values, just task constraints)
    evaluator.add_ground_truth({
        "location_required": "Chicago, Illinois",
        "required_month_year": "April 2026",
        "required_fields_per_event": ["name", "venue_name", "venue_address", "date", "time", "ticket_url"],
        "count_required": 4
    })

    # Build tree: top-level is already parallel. Create a single container node to mirror rubric root text.
    rubric_root = evaluator.add_parallel(
        id="Find_Four_Live_Performances",
        desc="Find four distinct upcoming live performances or shows in Chicago, Illinois, scheduled during April 2026. Each performance should provide complete information including performer/show name, venue details, schedule, and ticketing.",
        parent=root,
        critical=False,
    )

    # Verify each performance subtree
    for i in range(4):
        await verify_performance(evaluator, rubric_root, events[i], i)

    return evaluator.get_summary()