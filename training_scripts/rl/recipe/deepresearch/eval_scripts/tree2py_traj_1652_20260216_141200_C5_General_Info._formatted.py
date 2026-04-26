import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nyc_spring_music_2026"
TASK_DESCRIPTION = """I am planning to attend live performances in New York City this spring and would like to experience classical music or jazz concerts at established performing arts venues. Please identify three (3) classical music concerts or jazz performances taking place at major concert halls or performing arts centers in Manhattan between March 1, 2026, and April 30, 2026. Each of the three events must be at a different venue. For each event, provide the following information: (1) Event Details: The specific performance date, start time, and the name of the performing artist, ensemble, or orchestra; (2) Venue Information: The official venue name, complete street address, and a link to the venue's official website; (3) Ticket Link: A direct link to the official page where tickets for this specific event can be purchased. Please ensure that all three events are classical music concerts or jazz performances (not Broadway shows, comedy shows, pop concerts, or other types of entertainment)."""

DATE_RANGE_START = "2026-03-01"
DATE_RANGE_END = "2026-04-30"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    event_date: Optional[str] = None            # e.g., "March 12, 2026" or "2026-03-12"
    start_time: Optional[str] = None            # e.g., "7:30 PM"
    artist: Optional[str] = None                # artist, ensemble, or orchestra
    genre: Optional[str] = None                 # "classical" or "jazz" if specified
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    venue_website_url: Optional[str] = None     # official venue site
    ticket_url: Optional[str] = None            # direct ticket purchase page


class EventsExtraction(BaseModel):
    events: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
Extract up to the first three (3) classical music or jazz events described in the answer, providing structured fields for each.

For each event, extract the following fields exactly as mentioned in the answer:
- event_date: The performance date for the specific show (string as written, not parsed).
- start_time: The performance start time for that show (string as written).
- artist: The performing artist, ensemble, orchestra, or band name.
- genre: A short label for the performance type if provided (e.g., "classical", "jazz"); if not mentioned, return null.
- venue_name: The official venue name.
- venue_address: The complete street address for the venue as provided.
- venue_website_url: A link to the venue's official website.
- ticket_url: A direct link to the official page where tickets for this specific event can be purchased.

Rules:
- Only extract what is explicitly present in the answer. Do not infer or invent.
- If any field is missing for a given event, set it to null.
- Return a JSON object with a field "events" that is an array of up to 3 event objects.
- Ensure that all URLs are the actual URLs presented in the answer (accept plain or markdown-format links).
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _available_urls_for_event(ev: EventItem) -> List[str]:
    urls: List[str] = []
    if ev.ticket_url and ev.ticket_url.strip():
        urls.append(ev.ticket_url.strip())
    if ev.venue_website_url and ev.venue_website_url.strip():
        # avoid duplication if same as ticket_url
        if ev.venue_website_url.strip() not in urls:
            urls.append(ev.venue_website_url.strip())
    return urls


def _norm_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()


# --------------------------------------------------------------------------- #
# Verification: Per-Event                                                     #
# --------------------------------------------------------------------------- #
async def verify_single_event(
    evaluator: Evaluator,
    parent: VerificationNode,
    ev: EventItem,
    event_idx: int,
) -> Dict[str, VerificationNode]:
    """
    Build and verify a single event subtree.
    Returns references to key prerequisite nodes (e.g., details_provided, venue_provided, ticket_provided)
    for use in global constraint checks if needed.
    """
    # Create event container (non-critical; allows partial credit across events)
    event_node = evaluator.add_parallel(
        id=f"event_{event_idx+1}",
        desc=f"Documentation and details for event #{event_idx+1} are complete and accurate",
        parent=parent,
        critical=False
    )

    # 1) Event Details group (critical group, all children critical)
    details_group = evaluator.add_parallel(
        id=f"event_{event_idx+1}_details",
        desc="The specific performance date, start time, and performing artist/ensemble/orchestra are provided and accurate",
        parent=event_node,
        critical=True
    )

    # 1.1 Existence of details
    details_provided = evaluator.add_custom_node(
        result=bool(_norm_text(ev.event_date) and _norm_text(ev.start_time) and _norm_text(ev.artist)),
        id=f"event_{event_idx+1}_details_provided",
        desc=f"Event #{event_idx+1}: Date, start time, and artist/ensemble/orchestra are provided",
        parent=details_group,
        critical=True
    )

    # 1.2 Date matches official/ticket page
    date_match_leaf = evaluator.add_leaf(
        id=f"event_{event_idx+1}_date_match",
        desc=f"Event #{event_idx+1}: Performance date matches the cited source",
        parent=details_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The performance date for this event is '{ev.event_date}'.",
        node=date_match_leaf,
        sources=_available_urls_for_event(ev),
        additional_instruction="Verify the specific performance date on the ticket or official event page. Allow minor formatting differences (e.g., 'Mar' vs 'March').",
        extra_prerequisites=[details_provided]
    )

    # 1.3 Time matches official/ticket page
    time_match_leaf = evaluator.add_leaf(
        id=f"event_{event_idx+1}_time_match",
        desc=f"Event #{event_idx+1}: Start time matches the cited source",
        parent=details_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The performance start time for this event is '{ev.start_time}'.",
        node=time_match_leaf,
        sources=_available_urls_for_event(ev),
        additional_instruction="Verify the specific performance start time on the ticket or official event page. Allow minor formatting differences (e.g., '7:30 PM' vs '7:30 p.m.').",
        extra_prerequisites=[details_provided]
    )

    # 1.4 Artist matches official/ticket page
    artist_match_leaf = evaluator.add_leaf(
        id=f"event_{event_idx+1}_artist_match",
        desc=f"Event #{event_idx+1}: Performing artist/ensemble/orchestra matches the cited source",
        parent=details_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The performing artist/ensemble/orchestra for this event is '{ev.artist}'.",
        node=artist_match_leaf,
        sources=_available_urls_for_event(ev),
        additional_instruction="Verify the named performer on the official event or ticket page. Allow minor spelling/casing variations; ensure it refers to the same entity.",
        extra_prerequisites=[details_provided]
    )

    # 2) Venue group (critical group, all children critical)
    venue_group = evaluator.add_parallel(
        id=f"event_{event_idx+1}_venue",
        desc="The official venue name, complete street address, and a link to the venue's official website are provided and accurate",
        parent=event_node,
        critical=True
    )

    # 2.1 Existence of venue info
    venue_provided = evaluator.add_custom_node(
        result=bool(_norm_text(ev.venue_name) and _norm_text(ev.venue_address) and _norm_text(ev.venue_website_url)),
        id=f"event_{event_idx+1}_venue_provided",
        desc=f"Event #{event_idx+1}: Venue name, complete address, and the venue's official website URL are provided",
        parent=venue_group,
        critical=True
    )

    # 2.2 Venue name matches official site
    venue_name_match_leaf = evaluator.add_leaf(
        id=f"event_{event_idx+1}_venue_name_match",
        desc=f"Event #{event_idx+1}: Venue name matches the official venue website",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official venue name is '{ev.venue_name}'.",
        node=venue_name_match_leaf,
        sources=ev.venue_website_url or None,
        additional_instruction="Verify the venue's name on its official website.",
        extra_prerequisites=[venue_provided]
    )

    # 2.3 Venue address matches official site
    venue_address_match_leaf = evaluator.add_leaf(
        id=f"event_{event_idx+1}_venue_address_match",
        desc=f"Event #{event_idx+1}: Venue street address matches the official venue website",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue's street address is '{ev.venue_address}'.",
        node=venue_address_match_leaf,
        sources=ev.venue_website_url or None,
        additional_instruction="Verify the official street address on the venue website. Allow minor formatting differences (abbreviations like 'St.' vs 'Street').",
        extra_prerequisites=[venue_provided]
    )

    # 2.4 Venue site is official
    venue_site_official_leaf = evaluator.add_leaf(
        id=f"event_{event_idx+1}_venue_site_official",
        desc=f"Event #{event_idx+1}: Provided venue URL is the official website",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim="This URL is the venue's official website.",
        node=venue_site_official_leaf,
        sources=ev.venue_website_url or None,
        additional_instruction="Assess whether the site clearly represents the official venue (branding, about page, contact info, first-party signals).",
        extra_prerequisites=[venue_provided]
    )

    # 3) Ticket group (critical group, all children critical)
    ticket_group = evaluator.add_parallel(
        id=f"event_{event_idx+1}_ticket",
        desc="A valid link to the official ticket purchasing page for this specific event is provided and correct",
        parent=event_node,
        critical=True
    )

    # 3.1 Existence of ticket link
    ticket_provided = evaluator.add_custom_node(
        result=bool(_norm_text(ev.ticket_url)),
        id=f"event_{event_idx+1}_ticket_provided",
        desc=f"Event #{event_idx+1}: Ticket purchase page URL is provided",
        parent=ticket_group,
        critical=True
    )

    # 3.2 Ticket page validity and specificity
    ticket_valid_leaf = evaluator.add_leaf(
        id=f"event_{event_idx+1}_ticket_valid",
        desc=f"Event #{event_idx+1}: The ticket URL is an official purchase page for this specific performance",
        parent=ticket_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"This page allows purchasing tickets for the specific performance featuring '{ev.artist}' on '{ev.event_date}' at '{ev.start_time}' at '{ev.venue_name}'.",
        node=ticket_valid_leaf,
        sources=ev.ticket_url or None,
        additional_instruction="Verify that the page is an official ticketing/purchase page for the stated event (look for 'Buy Tickets', date/time specificity, seat selection, etc.). Aggregators or articles without purchase capability should not pass.",
        extra_prerequisites=[ticket_provided, details_provided, venue_provided]
    )

    return {
        "event_node": event_node,
        "details_provided": details_provided,
        "venue_provided": venue_provided,
        "ticket_provided": ticket_provided,
    }


# --------------------------------------------------------------------------- #
# Verification: Global Constraints                                            #
# --------------------------------------------------------------------------- #
async def verify_global_constraints(
    evaluator: Evaluator,
    root: VerificationNode,
    events: List[EventItem],
    prereq_nodes: List[Dict[str, VerificationNode]]
) -> None:
    """
    Build global constraints under the root as critical nodes with child leaves per event.
    """
    # Date range constraint (critical)
    date_constraint = evaluator.add_parallel(
        id="date_range_constraint",
        desc="All events are scheduled between March 1, 2026 and April 30, 2026 (inclusive)",
        parent=root,
        critical=True
    )
    for idx, ev in enumerate(events):
        leaf = evaluator.add_leaf(
            id=f"event_{idx+1}_in_date_range",
            desc=f"Event #{idx+1} date falls within the required range (2026-03-01 to 2026-04-30)",
            parent=date_constraint,
            critical=True
        )
        await evaluator.verify(
            claim=f"The specific performance date for this event ('{ev.event_date}') falls between March 1, 2026 and April 30, 2026 inclusive.",
            node=leaf,
            sources=_available_urls_for_event(ev),
            additional_instruction="Read the event/ticket page to determine the actual performance date and check if it lies within the stated range.",
            extra_prerequisites=[prereq_nodes[idx]["details_provided"]]
        )

    # Manhattan location and established venue constraint (critical)
    manhattan_constraint = evaluator.add_parallel(
        id="manhattan_location_constraint",
        desc="All events occur at established performing arts venues or concert halls located in Manhattan, NYC",
        parent=root,
        critical=True
    )
    for idx, ev in enumerate(events):
        # 1) Located in Manhattan
        loc_leaf = evaluator.add_leaf(
            id=f"event_{idx+1}_venue_in_manhattan",
            desc=f"Event #{idx+1}: Venue is in Manhattan (New York County), NYC",
            parent=manhattan_constraint,
            critical=True
        )
        await evaluator.verify(
            claim=f"The venue '{ev.venue_name}' is located in Manhattan, New York City.",
            node=loc_leaf,
            sources=_available_urls_for_event(ev),
            additional_instruction="Use the venue's official address (e.g., 'New York, NY' with Manhattan/NY County or ZIPs in the 100xx range). Confirm the borough is Manhattan.",
            extra_prerequisites=[prereq_nodes[idx]["venue_provided"]]
        )

        # 2) Established performing arts venue or concert hall
        est_leaf = evaluator.add_leaf(
            id=f"event_{idx+1}_venue_established",
            desc=f"Event #{idx+1}: Venue is an established performing arts venue or concert hall",
            parent=manhattan_constraint,
            critical=True
        )
        await evaluator.verify(
            claim=f"'{ev.venue_name}' is an established performing arts venue or concert hall (not a bar, restaurant, or nightclub).",
            node=est_leaf,
            sources=ev.venue_website_url or None,
            additional_instruction="Check the venue's official site/about page for clear indications that it is a concert hall or performing arts center (e.g., Carnegie Hall, Lincoln Center, The Town Hall).",
            extra_prerequisites=[prereq_nodes[idx]["venue_provided"]]
        )

    # Performance type constraint (critical)
    perf_type_constraint = evaluator.add_parallel(
        id="performance_type_constraint",
        desc="All events are classical music concerts or jazz performances (not Broadway/comedy/pop/etc.)",
        parent=root,
        critical=True
    )
    for idx, ev in enumerate(events):
        leaf = evaluator.add_leaf(
            id=f"event_{idx+1}_is_classical_or_jazz",
            desc=f"Event #{idx+1}: Performance is classical music or jazz",
            parent=perf_type_constraint,
            critical=True
        )
        # Use ev.genre if present to make a more precise claim; otherwise generic.
        genre_phrase = ev.genre if _norm_text(ev.genre) else "classical music or jazz"
        await evaluator.verify(
            claim=f"This performance is {genre_phrase} (and not Broadway/comedy/pop/other).",
            node=leaf,
            sources=_available_urls_for_event(ev),
            additional_instruction="Use the event/ticket page description and categorization. Accept 'orchestral, chamber, recital, opera (in concert form), choir' for classical; 'jazz' incl. combos, big band. Reject Broadway musicals, stand-up comedy, pop/rock shows.",
            extra_prerequisites=[prereq_nodes[idx]["details_provided"]]
        )

    # Venue uniqueness constraint (critical) - simple logical check
    norm_names = [ _norm_text(ev.venue_name) for ev in events ]
    unique_venues = len({n for n in norm_names if n})
    evaluator.add_custom_node(
        result=(unique_venues == 3),
        id="venue_uniqueness_constraint",
        desc="Each of the three events is at a different venue (no repeats)",
        parent=root,
        critical=True
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
    Evaluate a single answer for the NYC Spring 2026 classical/jazz events task.
    """
    # Initialize evaluator with a parallel root (independent checks)
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

    # Extract events from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction"
    )

    # Normalize to exactly 3 events (first 3 if more; pad with empty if fewer)
    events: List[EventItem] = list(extracted.events[:3])
    while len(events) < 3:
        events.append(EventItem())

    # Build per-event subtrees first (to enable using them as prerequisites in global constraints)
    per_event_prereqs: List[Dict[str, VerificationNode]] = []
    for i, ev in enumerate(events):
        prereqs = await verify_single_event(evaluator, root, ev, i)
        per_event_prereqs.append(prereqs)

    # Build and verify global constraints
    await verify_global_constraints(evaluator, root, events, per_event_prereqs)

    # Return evaluation summary
    return evaluator.get_summary()