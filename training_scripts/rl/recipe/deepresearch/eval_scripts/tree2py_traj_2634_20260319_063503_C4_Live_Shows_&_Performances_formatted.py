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
TASK_ID = "chicago_concert_venue_2026"
TASK_DESCRIPTION = """
Find a performance venue in Chicago, Illinois, that can host concerts with a seating capacity between 1,500 and 5,000 people and has at least one concert scheduled for April 2026. Provide the venue's name, verified Chicago address, exact seating capacity, the name and date of at least one scheduled April 2026 concert, the venue's official website URL, a link to where tickets can be purchased (through platforms such as Ticketmaster, SeatGeek, AXS, or the venue's own site), and information about wheelchair accessible seating availability.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    # Core identity
    venue_name: Optional[str] = None

    # Address + sources
    address: Optional[str] = None
    address_sources: List[str] = Field(default_factory=list)

    # Capacity + sources
    capacity: Optional[str] = None
    capacity_sources: List[str] = Field(default_factory=list)

    # At least one April 2026 concert
    april_2026_event_title: Optional[str] = None
    april_2026_event_date: Optional[str] = None
    april_2026_event_urls: List[str] = Field(default_factory=list)

    # Official website
    official_website_url: Optional[str] = None

    # Ticket purchase
    tickets_url: Optional[str] = None

    # Accessibility info
    accessibility_info_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    Extract all required fields for a single Chicago concert venue mentioned in the answer. Only extract information explicitly present in the answer text. Do not infer.

    Required fields to extract:
    - venue_name: The name of the venue.
    - address: The full street address as written (should indicate Chicago, IL/Illinois).
    - address_sources: Array of URL(s) cited in the answer that directly show or confirm the address (e.g., venue site contact page, Google Maps link, venue info page). If none provided, return [].
    - capacity: The exact stated seating capacity value or phrase as written (e.g., "3,600", "approximately 3,900", "3,500 seats"). Extract as a string.
    - capacity_sources: Array of URL(s) that explicitly state the capacity. If none provided, return [].
    - april_2026_event_title: The name/title of at least one concert scheduled for April 2026 provided in the answer. If not provided, set to null.
    - april_2026_event_date: The date for that April 2026 concert as written (e.g., "April 12, 2026"). If not provided, set to null.
    - april_2026_event_urls: Array of URL(s) that show the April 2026 event at this venue (e.g., event listing on the venue site or a ticketing page). If none provided, return [].
    - official_website_url: The venue's official website URL (home page or a relevant subpage). If not provided, set to null.
    - tickets_url: A URL to a page where tickets can be purchased, hosted by a recognized platform (e.g., Ticketmaster, SeatGeek, AXS) or the venue's own ticketing page. If not provided, set to null.
    - accessibility_info_url: A URL that provides information about wheelchair accessible seating or ADA accommodations (could be on the venue site or a ticketing platform page). If not provided, set to null.

    Notes:
    - For all URL fields or arrays, extract only explicit URLs shown in the answer text (plain or markdown). If the answer cites a source name without a URL, do not fabricate a URL; use null or [] accordingly.
    - Keep all values as strings when possible, including capacity and dates.
    """


# --------------------------------------------------------------------------- #
# Utility                                                                     #
# --------------------------------------------------------------------------- #
def collect_sources(*items: Optional[Any]) -> List[str]:
    """
    Collect and de-duplicate URLs from a mixture of strings, lists, or None.
    """
    urls: List[str] = []
    for it in items:
        if not it:
            continue
        if isinstance(it, list):
            for u in it:
                if isinstance(u, str) and u.strip():
                    urls.append(u.strip())
        elif isinstance(it, str):
            if it.strip():
                urls.append(it.strip())
    # Deduplicate preserving order
    seen = set()
    out: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_concert_venue_requirements(
    evaluator: Evaluator,
    parent_node,
    data: VenueExtraction,
) -> None:
    """
    Build and execute verification checks according to the rubric.
    """
    # Create the top-level requirements node (critical, parallel)
    req_node = evaluator.add_parallel(
        id="Concert_Venue_Requirements",
        desc="Find and verify a mid-sized concert venue in Chicago meeting all specified requirements",
        parent=parent_node,
        critical=True
    )

    # 1) Venue_Name_Provided (existence check)
    evaluator.add_custom_node(
        result=bool(data.venue_name and data.venue_name.strip()),
        id="Venue_Name_Provided",
        desc="The venue name is clearly identified",
        parent=req_node,
        critical=True
    )

    # 2) Chicago_Location_Verified
    chicago_node = evaluator.add_leaf(
        id="Chicago_Location_Verified",
        desc="The venue is located in Chicago with a verifiable address",
        parent=req_node,
        critical=True
    )
    chicago_sources = collect_sources(data.address_sources, data.official_website_url)
    venue_name = data.venue_name or "the venue"
    address_str = data.address or "(address not provided)"
    chicago_claim = (
        f"The venue '{venue_name}' has the address '{address_str}', and this address is located in Chicago, Illinois (USA)."
    )
    await evaluator.verify(
        claim=chicago_claim,
        node=chicago_node,
        sources=chicago_sources,
        additional_instruction=(
            "Confirm from the cited page(s) that the given address is in Chicago, Illinois. "
            "Allow reasonable formatting variations (e.g., 'Chicago, IL'). "
            "If the page lists multiple locations, verify that the one associated with this venue is in Chicago."
        ),
    )

    # 3) Capacity_In_Range
    capacity_node = evaluator.add_leaf(
        id="Capacity_In_Range",
        desc="The venue's seating capacity is between 1,500 and 5,000",
        parent=req_node,
        critical=True
    )
    cap_sources = collect_sources(data.capacity_sources, data.official_website_url)
    capacity_text = data.capacity or "(capacity not provided)"
    cap_claim = (
        f"The official or authoritative page(s) state that the seating capacity of '{venue_name}' is {capacity_text}, "
        "and this capacity is between 1,500 and 5,000 inclusive."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=capacity_node,
        sources=cap_sources,
        additional_instruction=(
            "Verify the stated seating capacity for concerts. Prefer explicit phrases like 'seating capacity', "
            "'capacity', or 'seats'. If multiple capacities are shown for different configurations, "
            "accept if any primary concert seating capacity falls within 1,500–5,000. "
            "Reject unrelated numbers (e.g., attendance figures)."
        ),
    )

    # 4) April_2026_Concert_Scheduled
    april_node = evaluator.add_leaf(
        id="April_2026_Concert_Scheduled",
        desc="The venue has at least one concert scheduled for April 2026",
        parent=req_node,
        critical=True
    )
    april_sources = collect_sources(data.april_2026_event_urls, data.tickets_url, data.official_website_url)
    event_title = data.april_2026_event_title or "(event title not provided)"
    event_date = data.april_2026_event_date or "(event date not provided)"
    april_claim = (
        f"There is at least one concert scheduled at '{venue_name}' in April 2026. "
        f"For example: '{event_title}' on {event_date} (the date should fall within April 2026)."
    )
    await evaluator.verify(
        claim=april_claim,
        node=april_node,
        sources=april_sources,
        additional_instruction=(
            "Use the cited event or ticketing page(s) to verify that at least one concert occurs at this venue in April 2026. "
            "Ensure the event is at the same venue and the date is in April 2026. "
            "If multiple dates are shown, it's sufficient that one scheduled date is in April 2026."
        ),
    )

    # 5) Official_Website
    official_node = evaluator.add_leaf(
        id="Official_Website",
        desc="The venue's official website URL is provided",
        parent=req_node,
        critical=True
    )
    official_url = data.official_website_url
    official_claim = (
        f"This website is the official site of the venue '{venue_name}' located in Chicago, Illinois."
    )
    await evaluator.verify(
        claim=official_claim,
        node=official_node,
        sources=official_url,
        additional_instruction=(
            "Determine whether the URL is the venue’s official website (not a third-party listing). "
            "Look for venue branding, navigation, contact information, or explicit statements that indicate it's the official site."
        ),
    )

    # 6) Ticket_Purchase_Platform
    tickets_node = evaluator.add_leaf(
        id="Ticket_Purchase_Platform",
        desc="Tickets are available through a recognized platform (Ticketmaster, SeatGeek, AXS, venue website, etc.)",
        parent=req_node,
        critical=True
    )
    tickets_url = data.tickets_url
    tickets_claim = (
        f"This URL is a valid ticket purchase page for a concert at '{venue_name}', "
        "hosted on a recognized platform (e.g., Ticketmaster, SeatGeek, AXS) or the venue's own official ticketing page."
    )
    await evaluator.verify(
        claim=tickets_claim,
        node=tickets_node,
        sources=tickets_url,
        additional_instruction=(
            "Confirm that the page facilitates purchasing tickets for an event at this venue. "
            "Accepted platforms include Ticketmaster, SeatGeek, AXS, or the venue’s official site. "
            "Look for clear purchase options ('Buy Tickets', seat selection, checkout)."
        ),
    )

    # 7) Accessibility_Information
    access_node = evaluator.add_leaf(
        id="Accessibility_Information",
        desc="Information about wheelchair accessible seating is provided",
        parent=req_node,
        critical=True
    )
    access_sources = collect_sources(data.accessibility_info_url, data.official_website_url, data.tickets_url)
    access_claim = (
        f"Wheelchair accessible seating or ADA accommodations are available at '{venue_name}', "
        "and the cited page(s) provide information about this."
    )
    await evaluator.verify(
        claim=access_claim,
        node=access_node,
        sources=access_sources,
        additional_instruction=(
            "Look for phrases like 'wheelchair accessible', 'accessible seating', 'ADA', 'mobility', or 'companion seating'. "
            "Information can be on the venue site or the ticketing page. The page should clearly state availability or policy."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Chicago concert venue task and return a structured evaluation summary.
    """
    # Initialize evaluator (root is a non-critical container)
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

    # Extract structured venue information from the answer
    extracted: VenueExtraction = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Optionally record custom info (e.g., a quick summary)
    evaluator.add_custom_info(
        info={
            "venue_name": extracted.venue_name,
            "address": extracted.address,
            "capacity": extracted.capacity,
            "official_website_url": extracted.official_website_url,
            "tickets_url": extracted.tickets_url,
            "april_2026_event_title": extracted.april_2026_event_title,
            "april_2026_event_date": extracted.april_2026_event_date,
        },
        info_type="extracted_summary",
        info_name="extracted_summary",
    )

    # Build and run verification checks
    await verify_concert_venue_requirements(evaluator, root, extracted)

    # Return final structured summary
    return evaluator.get_summary()