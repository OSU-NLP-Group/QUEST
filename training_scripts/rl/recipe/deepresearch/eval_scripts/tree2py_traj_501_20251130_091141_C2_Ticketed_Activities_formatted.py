import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "chicago_concerts_dec_2025"
TASK_DESCRIPTION = """
I am looking for live music concerts happening in Chicago, Illinois during December 2025. Identify two concerts that meet the following requirements:

1. The event must be a live music concert (not a comedy show, theater performance, or sporting event)
2. The venue must have a minimum capacity of 15,000 people
3. Tickets must be currently available for purchase (the event must not be sold out)

For each concert, provide:
- The specific date and start time of the event
- The complete venue name and full street address
- A direct link to purchase tickets from an official ticketing platform or the venue's official website
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConcertItem(BaseModel):
    """Structured representation of a single concert as provided by the answer."""
    event_name: Optional[str] = None
    artist_or_headliner: Optional[str] = None
    date: Optional[str] = None  # Keep as string to tolerate various formats (e.g., "Dec 12, 2025")
    start_time: Optional[str] = None  # e.g., "7:30 PM"
    venue_name: Optional[str] = None
    street_address: Optional[str] = None  # full address string (include number, street, city, state, ZIP if available)
    city: Optional[str] = None
    state: Optional[str] = None
    ticket_url: Optional[str] = None  # direct purchase link
    official_listing_url: Optional[str] = None  # venue or official ticketing listing; can be same as ticket_url
    source_urls: List[str] = Field(default_factory=list)  # any other URLs cited in the answer
    capacity_evidence_urls: List[str] = Field(default_factory=list)  # URLs that can prove venue capacity >= 15,000


class ConcertsExtraction(BaseModel):
    concerts: List[ConcertItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_concerts() -> str:
    return """
    From the answer, extract up to TWO live music concerts that the answer claims meet ALL of the following:
    – Location: Chicago, Illinois (the venue address should indicate Chicago, IL or Chicago, Illinois)
    – Date: During December 2025
    – Event Type: A live music concert (not comedy, theater, or sports)
    – Venue capacity: At least 15,000 people (provide a capacity source URL if present)
    – Tickets: Currently available (not sold out), with a direct purchase link from an official ticketing platform or the venue's official site

    For each concert, extract the following fields (use null if missing):
    - event_name: The event or tour name (e.g., "Trans-Siberian Orchestra: The Ghosts of Christmas Eve")
    - artist_or_headliner: The headlining artist/band if stated
    - date: The specific calendar date (e.g., "December 12, 2025" or "2025-12-12")
    - start_time: The local start time (e.g., "7:30 PM")
    - venue_name: Complete venue name
    - street_address: Full street address including city and state (preferably with ZIP if present)
    - city: City name (expected "Chicago")
    - state: State abbreviation or name (expected "IL" or "Illinois")
    - ticket_url: Direct purchase link for tickets
    - official_listing_url: Official event listing URL (venue site or official ticketing platform). If the ticket_url already serves as the official listing, you may repeat it here.
    - source_urls: Any other URLs cited that support event details (date/time/venue/city)
    - capacity_evidence_urls: URLs that can be used to verify the venue’s capacity (official venue page, Wikipedia, etc.)
    
    Return the result as a JSON object:
    {
      "concerts": [ { ...concert item 1... }, { ...concert item 2... }, ... ]
    }
    Extract items in the same order as presented in the answer. Do not invent URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*url_lists: Optional[List[str]]) -> List[str]:
    """Merge multiple URL lists, deduplicate, and drop falsy values."""
    merged: List[str] = []
    seen = set()
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if u and isinstance(u, str):
                if u not in seen:
                    seen.add(u)
                    merged.append(u)
    return merged


def _safe_first_url(*candidates: Optional[str]) -> Optional[str]:
    """Return the first non-empty URL candidate if any."""
    for c in candidates:
        if c and isinstance(c, str) and c.strip():
            return c
    return None


# --------------------------------------------------------------------------- #
# Verification for one concert                                                #
# --------------------------------------------------------------------------- #
async def verify_concert(evaluator: Evaluator, parent_node, item: ConcertItem, idx: int) -> None:
    """
    Build verification subtree and run checks for a single concert.
    Matches the rubric structure:
      Concert_{i} (sequential)
        - Concert_{i}_Event_Criteria (parallel, critical)
          - Concert_{i}_In_Chicago_IL
          - Concert_{i}_In_December_2025
          - Concert_{i}_Is_Live_Music_Concert
          - Concert_{i}_Venue_Capacity_GTE_15000
          - Concert_{i}_Tickets_Available
        - Concert_{i}_Provided_Information (parallel, critical)
          - Concert_{i}_Specific_Date_Provided
          - Concert_{i}_Start_Time_Provided
          - Concert_{i}_Venue_Name_Provided
          - Concert_{i}_Full_Street_Address_Provided
          - Concert_{i}_Direct_Ticket_Purchase_Link_Provided
          - Concert_{i}_Ticket_Link_Is_Official
          - Concert_{i}_Official_Listing_Present
    """
    n = idx + 1
    concert_node = evaluator.add_sequential(
        id=f"Concert_{n}",
        desc=f"{'First' if n == 1 else 'Second'} concert meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # -------------------- Event Criteria (critical, parallel) --------------------
    criteria_node = evaluator.add_parallel(
        id=f"Concert_{n}_Event_Criteria",
        desc=f"Verify the {'first' if n == 1 else 'second'} concert satisfies all event constraints",
        parent=concert_node,
        critical=True
    )

    # Useful sources for criteria verifications
    primary_purchase_url = _safe_first_url(item.ticket_url)
    listing_url = _safe_first_url(item.official_listing_url, item.ticket_url)
    all_event_sources = _merge_sources(
        [primary_purchase_url] if primary_purchase_url else None,
        [listing_url] if listing_url else None,
        item.source_urls
    )

    # 1) Location = Chicago, IL
    city_leaf = evaluator.add_leaf(
        id=f"Concert_{n}_In_Chicago_IL",
        desc=f"Event location is Chicago, Illinois",
        parent=criteria_node,
        critical=True
    )
    city_claim = "The event takes place in Chicago, Illinois (Chicago, IL)."
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=all_event_sources,
        additional_instruction="Confirm that the event page clearly indicates that the venue/city is Chicago, IL (or Chicago, Illinois). Minor formatting differences are okay."
    )

    # 2) Date in December 2025
    date_leaf = evaluator.add_leaf(
        id=f"Concert_{n}_In_December_2025",
        desc=f"Event occurs during December 2025",
        parent=criteria_node,
        critical=True
    )
    # We incorporate the provided date (if any) to assist the judge
    extracted_date = item.date or "an unspecified date"
    date_claim = f"The event date is {extracted_date}, and it falls in December 2025."
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=all_event_sources,
        additional_instruction="Verify that the event date displayed on the page is in December 2025. If multiple dates are shown, ensure the one described in the answer is in December 2025."
    )

    # 3) Event type is a live music concert
    type_leaf = evaluator.add_leaf(
        id=f"Concert_{n}_Is_Live_Music_Concert",
        desc=f"Event is a live music concert (not comedy, theater, or sports)",
        parent=criteria_node,
        critical=True
    )
    type_claim = "This event is a live music concert (featuring music performance), not a comedy show, theater performance, or sporting event."
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=all_event_sources,
        additional_instruction="Use the event description/title/category to confirm that it is a concert/music performance. If it is clearly comedy, theater, or sports, mark as not supported."
    )

    # 4) Venue capacity >= 15,000
    capacity_leaf = evaluator.add_leaf(
        id=f"Concert_{n}_Venue_Capacity_GTE_15000",
        desc=f"Venue capacity is at least 15,000 people",
        parent=criteria_node,
        critical=True
    )
    venue_name_str = item.venue_name or "the venue"
    capacity_claim = f"The venue {venue_name_str} has a capacity of at least 15,000 people."
    capacity_sources = _merge_sources(item.capacity_evidence_urls)
    # If no dedicated capacity URLs were provided, we also include event/venue sources in case capacity info appears there
    if not capacity_sources:
        capacity_sources = all_event_sources
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=capacity_sources,
        additional_instruction="Confirm the maximum or typical capacity of the stated venue is ≥ 15,000. Accept official venue pages or reliable sources (e.g., Wikipedia) if they clearly indicate capacity ≥ 15,000."
    )

    # 5) Tickets available (not sold out)
    tickets_leaf = evaluator.add_leaf(
        id=f"Concert_{n}_Tickets_Available",
        desc=f"Tickets are currently available for purchase (not sold out)",
        parent=criteria_node,
        critical=True
    )
    tickets_claim = "Tickets are currently available for purchase on this page (the event is not sold out)."
    await evaluator.verify(
        claim=tickets_claim,
        node=tickets_leaf,
        sources=primary_purchase_url if primary_purchase_url else all_event_sources,
        additional_instruction="Look for signals like 'Buy Tickets', seat map availability, or non-sold-out messages; if the page clearly indicates 'Sold Out' or only waitlist, then mark as not supported."
    )

    # -------------------- Provided Information (critical, parallel) -------------
    info_node = evaluator.add_parallel(
        id=f"Concert_{n}_Provided_Information",
        desc=f"Verify all required information is provided for the {'first' if n == 1 else 'second'} concert",
        parent=concert_node,
        critical=True
    )

    # Existence checks (critical, custom)
    evaluator.add_custom_node(
        result=bool(item.date and item.date.strip()),
        id=f"Concert_{n}_Specific_Date_Provided",
        desc="Specific event date is provided",
        parent=info_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(item.start_time and item.start_time.strip()),
        id=f"Concert_{n}_Start_Time_Provided",
        desc="Event start time is provided",
        parent=info_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(item.venue_name and item.venue_name.strip()),
        id=f"Concert_{n}_Venue_Name_Provided",
        desc="Complete venue name is provided",
        parent=info_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(item.street_address and item.street_address.strip()),
        id=f"Concert_{n}_Full_Street_Address_Provided",
        desc="Full street address of the venue is provided",
        parent=info_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(item.ticket_url and item.ticket_url.strip()),
        id=f"Concert_{n}_Direct_Ticket_Purchase_Link_Provided",
        desc="A direct link to purchase tickets is provided",
        parent=info_node,
        critical=True
    )

    # Ticket link is official (verify)
    official_ticket_leaf = evaluator.add_leaf(
        id=f"Concert_{n}_Ticket_Link_Is_Official",
        desc="The ticket link is from an official ticketing platform or the venue's official website",
        parent=info_node,
        critical=True
    )
    official_ticket_claim = "This ticket link is an official primary purchase page for the event (official ticketing platform or the venue's official website)."
    await evaluator.verify(
        claim=official_ticket_claim,
        node=official_ticket_leaf,
        sources=primary_purchase_url if primary_purchase_url else all_event_sources,
        additional_instruction=(
            "Treat domains like ticketmaster.com, livenation.com, axs.com, seatgeek.com (when used as a venue's primary), etix.com, universe.com, and the venue's own domain as official. "
            "Resale/marketplaces like stubhub.com, vividseats.com, ticketnetwork.com, etc., are NOT official primary ticketing. "
            "Also confirm the page clearly corresponds to this specific event."
        )
    )

    # Official listing present (verify)
    official_listing_leaf = evaluator.add_leaf(
        id=f"Concert_{n}_Official_Listing_Present",
        desc="Event is officially listed on either the venue's official website or an official ticketing platform",
        parent=info_node,
        critical=True
    )
    # Use official_listing_url if present; otherwise, fall back to the ticket_url
    official_listing_source = _safe_first_url(item.official_listing_url, item.ticket_url)
    listing_event_name = item.event_name or item.artist_or_headliner or "the concert"
    listing_claim = (
        f"This URL is an official event listing for {listing_event_name} at {venue_name_str} on {extracted_date} in Chicago, IL."
    )
    await evaluator.verify(
        claim=listing_claim,
        node=official_listing_leaf,
        sources=official_listing_source if official_listing_source else all_event_sources,
        additional_instruction=(
            "Confirm that the page is an official event listing either on the venue's own website or on an official ticketing platform (primary). "
            "It should clearly show the event’s date/time and venue in Chicago, IL."
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
    Evaluate an answer for the Chicago concerts (Dec 2025) task.
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_concerts(),
        template_class=ConcertsExtraction,
        extraction_name="concerts_extraction"
    )

    # Normalize to exactly two concerts (pad with empty if fewer)
    items: List[ConcertItem] = list(extracted.concerts or [])
    if len(items) > 2:
        items = items[:2]
    while len(items) < 2:
        items.append(ConcertItem())

    # Build and verify for each concert
    await verify_concert(evaluator, root, items[0], 0)
    await verify_concert(evaluator, root, items[1], 1)

    return evaluator.get_summary()