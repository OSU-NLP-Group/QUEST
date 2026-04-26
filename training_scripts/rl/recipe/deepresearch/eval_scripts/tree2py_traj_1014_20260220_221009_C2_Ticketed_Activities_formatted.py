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
TASK_ID = "concert_feb2026_tx_arena"
TASK_DESCRIPTION = (
    "Identify a concert scheduled for February 2026 at an indoor arena venue in Fort Worth or Arlington, Texas, "
    "where the performing artist has at least two consecutive show dates at the same venue. For this concert, provide "
    "the following information: (1) The name of the performing artist, (2) The name of the venue, (3) The concert dates, "
    "(4) The venue's maximum seating capacity, (5) The official URL where tickets are currently on sale, and "
    "(6) The URL where VIP packages or premium experiences are offered (if available)."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventExtraction(BaseModel):
    """Structured extraction of the concert event details from the agent's answer."""
    artist_name: Optional[str] = None
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    venue_state: Optional[str] = None
    concert_dates: List[str] = Field(default_factory=list)  # Include all dates in the run; strings are fine
    venue_max_capacity: Optional[str] = None

    official_ticket_url: Optional[str] = None
    vip_url: Optional[str] = None

    # Helpful supporting sources, if the answer provided them
    venue_info_urls: List[str] = Field(default_factory=list)            # Venue homepage or Wikipedia etc.
    capacity_source_urls: List[str] = Field(default_factory=list)       # URLs that explicitly state capacity
    additional_source_urls: List[str] = Field(default_factory=list)     # Any other URLs mentioned

# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_event() -> str:
    return """
    Extract exactly one concert event described in the answer that matches the task. If multiple events are mentioned, pick the first qualifying one.

    Return a JSON object with the following fields:
    1) artist_name: The name of the performing artist.
    2) venue_name: The name of the venue.
    3) venue_city: The venue's city (e.g., "Fort Worth" or "Arlington").
    4) venue_state: The venue's state (e.g., "Texas" or "TX").
    5) concert_dates: A list of all show dates for the consecutive run at this venue. Preserve the date strings as presented (e.g., "Feb 20, 2026", "2026-02-21").
    6) venue_max_capacity: The venue's maximum seating capacity value as stated in the answer (return as a string; do not convert).
    7) official_ticket_url: The official page URL where tickets are on sale now (Ticketmaster, AXS, the venue's box office, Live Nation, or the artist's official ticketing link). Include the protocol.
    8) vip_url: The URL for VIP packages or premium experiences related to this concert (if provided). If not available, return null.
    9) venue_info_urls: An array of URLs that link to venue or venue info pages (official venue site, Wikipedia, etc.). If none provided, return an empty array.
    10) capacity_source_urls: An array of URLs explicitly used to support the capacity figure. If none provided, return an empty array.
    11) additional_source_urls: Any other URLs the answer cites that are relevant to this event (exclude duplicates of the above). If none provided, return an empty array.

    Rules:
    - Extract only what is explicitly in the answer. Do not invent any information.
    - For all URLs, include full URLs with protocol. If the answer shows a URL without protocol, prepend "http://".
    - If a field is missing in the answer, return null (for single values) or [] (for arrays).
    """

# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _unique_nonempty(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        u_str = u.strip()
        if not u_str:
            continue
        if u_str not in seen:
            seen.add(u_str)
            out.append(u_str)
    return out

def gather_all_sources(event: EventExtraction) -> List[str]:
    """
    Collect a deduplicated list of all potential evidence URLs.
    """
    return _unique_nonempty(
        ([event.official_ticket_url, event.vip_url] if event else []) +
        (event.venue_info_urls if event else []) +
        (event.capacity_source_urls if event else []) +
        (event.additional_source_urls if event else [])
    )

# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_event_qualification_checks(
    evaluator: Evaluator,
    parent_node,
    event: EventExtraction
) -> None:
    """
    Build the 'Event_Qualification' parallel node and add all critical verification leaves.
    """
    qual_node = evaluator.add_parallel(
        id="Event_Qualification",
        desc="The identified concert event satisfies all qualifying constraints.",
        parent=parent_node,
        critical=True
    )

    all_sources = gather_all_sources(event)
    primary_ticket_source = event.official_ticket_url or None

    # 1) Geographic Location
    geo_node = evaluator.add_leaf(
        id="Geographic_Location",
        desc="Venue is located in Fort Worth or Arlington, Texas.",
        parent=qual_node,
        critical=True
    )
    geo_claim = (
        f"The venue '{event.venue_name or ''}' is located in Fort Worth, TX or Arlington, TX."
    )
    await evaluator.verify(
        claim=geo_claim,
        node=geo_node,
        sources=all_sources if all_sources else primary_ticket_source,
        additional_instruction=(
            "Verify that the venue city and state on the provided page(s) indicate Fort Worth, TX or Arlington, TX."
        )
    )

    # 2) Event Type
    type_node = evaluator.add_leaf(
        id="Event_Type",
        desc="Event is a concert (live musical performance).",
        parent=qual_node,
        critical=True
    )
    type_claim = (
        f"This event is a concert (live musical performance) for '{event.artist_name or ''}' at '{event.venue_name or ''}'."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_node,
        sources=primary_ticket_source or all_sources,
        additional_instruction=(
            "Confirm that the event is a concert/show/live musical performance. "
            "Accept phrases like 'concert', 'show', 'live performance'."
        )
    )

    # 3) Scheduled In February 2026
    feb_node = evaluator.add_leaf(
        id="Scheduled_In_February_2026",
        desc="The concert is scheduled to take place in February 2026 (i.e., at least one of the concert dates is in February 2026).",
        parent=qual_node,
        critical=True
    )
    date_list_str = "; ".join(event.concert_dates) if event.concert_dates else ""
    feb_claim = (
        f"At least one of the concert dates ({date_list_str}) occurs in February 2026."
    )
    await evaluator.verify(
        claim=feb_claim,
        node=feb_node,
        sources=primary_ticket_source or all_sources,
        additional_instruction=(
            "Check the event date(s) displayed on the ticketing or official event page(s). "
            "Pass if any listed date is in February 2026."
        )
    )

    # 4) Indoor Arena Type
    indoor_node = evaluator.add_leaf(
        id="Indoor_Arena_Type",
        desc="Venue is an indoor arena (not an outdoor stadium).",
        parent=qual_node,
        critical=True
    )
    indoor_claim = (
        f"The venue '{event.venue_name or ''}' is an indoor arena (an enclosed, roofed facility), not an outdoor stadium."
    )
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_node,
        sources=all_sources if all_sources else primary_ticket_source,
        additional_instruction=(
            "Use venue info (official site or Wikipedia) if available. "
            "Look for terms indicating indoor arena (e.g., 'arena', 'indoor', 'enclosed'). "
            "If only the ticketing page is available, infer from venue name and common descriptors on the page."
        )
    )

    # 5) Consecutive Multi-Date Run
    consecutive_node = evaluator.add_leaf(
        id="Consecutive_Multi_Date_Run",
        desc="Performing artist has at least two consecutive show dates at the same venue.",
        parent=qual_node,
        critical=True
    )
    consec_claim = (
        f"There are at least two consecutive show dates (back-to-back days) at the same venue '{event.venue_name or ''}' for '{event.artist_name or ''}'. "
        f"Dates provided: {date_list_str}."
    )
    await evaluator.verify(
        claim=consec_claim,
        node=consecutive_node,
        sources=primary_ticket_source or all_sources,
        additional_instruction=(
            "Confirm the page(s) show two or more consecutive dates (e.g., Feb 20 and Feb 21) at the SAME venue for the SAME artist. "
            "Phrases like 'two nights', 'back-to-back', or listing adjacent dates should count."
        )
    )

    # 6) Capacity Publicly Stated
    capacity_node = evaluator.add_leaf(
        id="Capacity_Publicly_Stated",
        desc="Venue has a publicly stated maximum seating capacity.",
        parent=qual_node,
        critical=True
    )
    capacity_value = event.venue_max_capacity or ""
    capacity_claim = (
        f"The maximum seating capacity of '{event.venue_name or ''}' is publicly stated as '{capacity_value}'."
    )
    capacity_sources = (
        event.capacity_source_urls if event.capacity_source_urls else
        (event.venue_info_urls if event.venue_info_urls else all_sources)
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_node,
        sources=capacity_sources if capacity_sources else primary_ticket_source,
        additional_instruction=(
            "Verify that a capacity figure is explicitly stated on a credible page (official venue site or Wikipedia preferred). "
            "Minor variations due to configuration are acceptable as long as a max capacity is stated."
        )
    )

    # 7) Tickets Currently On Sale (Official)
    tickets_node = evaluator.add_leaf(
        id="Tickets_Currently_On_Sale_Official",
        desc="Tickets are currently on sale via an official ticketing platform.",
        parent=qual_node,
        critical=True
    )
    tickets_claim = (
        f"Tickets for the '{event.artist_name or ''}' concert at '{event.venue_name or ''}' are currently on sale on an official platform at {event.official_ticket_url or ''}."
    )
    await evaluator.verify(
        claim=tickets_claim,
        node=tickets_node,
        sources=event.official_ticket_url or all_sources,
        additional_instruction=(
            "Look for 'Buy Tickets', 'Get Tickets', or an active on-sale state on platforms like Ticketmaster, AXS, Live Nation, or the venue's official box office page."
        )
    )

    # 8) VIP Offered Beyond Standard
    vip_node = evaluator.add_leaf(
        id="VIP_Offered_Beyond_Standard",
        desc="The concert offers VIP packages or premium experiences beyond standard tickets.",
        parent=qual_node,
        critical=True
    )
    vip_claim = (
        f"VIP packages or premium experiences are offered for this event at {event.vip_url or ''}."
    )
    await evaluator.verify(
        claim=vip_claim,
        node=vip_node,
        sources=event.vip_url or all_sources,
        additional_instruction=(
            "Verify the presence of VIP/premium offerings (e.g., 'VIP package', 'premium experience', 'meet & greet', 'platinum tickets'). "
            "If a dedicated VIP page exists, use that. Otherwise, check ticketing page for premium options."
        )
    )

async def build_required_fields_checks(
    evaluator: Evaluator,
    parent_node,
    event: EventExtraction
) -> None:
    """
    Build the 'Required_Response_Fields' parallel node and add critical existence checks.
    """
    fields_node = evaluator.add_parallel(
        id="Required_Response_Fields",
        desc="The response provides all required fields requested by the question.",
        parent=parent_node,
        critical=True
    )

    # Artist name provided
    evaluator.add_custom_node(
        result=bool(event.artist_name and event.artist_name.strip()),
        id="Artist_Name_Provided",
        desc="Provides the name of the performing artist.",
        parent=fields_node,
        critical=True
    )

    # Venue name provided
    evaluator.add_custom_node(
        result=bool(event.venue_name and event.venue_name.strip()),
        id="Venue_Name_Provided",
        desc="Provides the name of the venue.",
        parent=fields_node,
        critical=True
    )

    # Concert dates provided (expect at least two dates for the run)
    evaluator.add_custom_node(
        result=(bool(event.concert_dates) and len(event.concert_dates) >= 2),
        id="Concert_Dates_Provided",
        desc="Provides the concert dates (the multiple dates that are part of the consecutive run).",
        parent=fields_node,
        critical=True
    )

    # Venue max capacity value provided
    evaluator.add_custom_node(
        result=bool(event.venue_max_capacity and event.venue_max_capacity.strip()),
        id="Venue_Max_Capacity_Value_Provided",
        desc="Provides the venue's maximum seating capacity value.",
        parent=fields_node,
        critical=True
    )

    # Official ticket on-sale URL provided
    evaluator.add_custom_node(
        result=bool(event.official_ticket_url and event.official_ticket_url.strip()),
        id="Official_Ticket_On_Sale_URL_Provided",
        desc="Provides the official URL where tickets are currently on sale.",
        parent=fields_node,
        critical=True
    )

    # VIP or premium URL provided (the question says 'if available', but rubric marks it required)
    evaluator.add_custom_node(
        result=bool(event.vip_url and event.vip_url.strip()),
        id="VIP_or_Premium_URL_Provided",
        desc="Provides the URL where VIP packages or premium experiences are offered.",
        parent=fields_node,
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
    Evaluate an answer for the concert event identification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Top-level: sequential flow
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

    # Extract event information from the answer
    event = await evaluator.extract(
        prompt=prompt_extract_event(),
        template_class=EventExtraction,
        extraction_name="event_extraction",
    )

    # Build top-level node representing the concert event identification (critical, sequential)
    concert_node = evaluator.add_sequential(
        id="Concert_Event_Identification",
        desc="Identify one qualifying concert and provide all required details and URLs per the question/constraints.",
        parent=root,
        critical=True
    )

    # Build qualification checks
    await build_event_qualification_checks(evaluator, concert_node, event)

    # Build required fields checks
    await build_required_fields_checks(evaluator, concert_node, event)

    # Return evaluation summary
    return evaluator.get_summary()