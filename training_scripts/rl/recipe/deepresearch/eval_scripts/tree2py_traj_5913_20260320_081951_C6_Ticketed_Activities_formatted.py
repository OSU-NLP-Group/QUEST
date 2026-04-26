import asyncio
import logging
import re
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "zach_bryan_stadiums_2026"
TASK_DESCRIPTION = """
Identify four stadium concerts from Zach Bryan's 2026 'WITH HEAVEN ON TOUR' U.S. tour schedule where the venue has a seating capacity of at least 60,000 and the concert is scheduled between March 1, 2026 and August 31, 2026. For each concert, provide the following information: stadium name, city and state, exact concert date, stadium's concert seating capacity with documentation, official ticket purchase URL, and reference URL to Zach Bryan's official tour page or venue information.
"""

DATE_RANGE_START = date(2026, 3, 1)
DATE_RANGE_END = date(2026, 8, 31)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ConcertItem(BaseModel):
    stadium_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    date: Optional[str] = None  # Keep as string for flexible matching
    capacity: Optional[str] = None  # Free-form (e.g., "80,000", "~70k for concerts")
    capacity_reference_urls: List[str] = Field(default_factory=list)
    ticket_url: Optional[str] = None
    tour_reference_url: Optional[str] = None  # Official Zach Bryan tour page or the venue's event page


class ConcertsExtraction(BaseModel):
    concerts: List[ConcertItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_concerts() -> str:
    return """
    Extract up to four (4) Zach Bryan 2026 'WITH HEAVEN ON TOUR' U.S. stadium concerts from the answer text.
    For each concert, extract the following fields exactly as provided:
    - stadium_name: Name of the stadium venue (not arena/theater).
    - city: City where the stadium is located.
    - state: U.S. state (use standard state name or two-letter code).
    - date: The exact concert date as given in the answer (e.g., "July 12, 2026" or "2026-07-12").
    - capacity: The stadium's seating capacity for concerts as stated (free-form string; do NOT convert to number).
    - capacity_reference_urls: An array of URL(s) that document the stadium's capacity (official venue page, Wikipedia, or credible sources). Include all URLs mentioned; if none, return an empty array.
    - ticket_url: Official ticket purchase URL (e.g., Ticketmaster, AXS, SeatGeek, venue's own ticketing, or the artist's official ticketing page). If not present, null.
    - tour_reference_url: A URL to Zach Bryan's official tour page that lists the date, or the official venue event page confirming the concert. If not present, null.

    Rules:
    - Only include concerts in the United States.
    - If the answer provides more than four items, return only the first four.
    - If some field is missing for an item, set it to null (or empty array for URLs).
    - Do not invent or infer any URLs or values not present in the answer.

    Return a JSON object with a single key "concerts" that is an array of objects with the fields above.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
_ORDINAL_SUFFIX_RE = re.compile(r'(\d+)(st|nd|rd|th)\b', flags=re.IGNORECASE)
_MONTH_FIXES = {
    "Sept": "Sep",
    "Sept.": "Sep",
    "Jun.": "Jun",
    "Jul.": "Jul",
    "Aug.": "Aug",
    "Mar.": "Mar",
    "Apr.": "Apr",
    "Jan.": "Jan",
    "Feb.": "Feb",
    "Oct.": "Oct",
    "Nov.": "Nov",
    "Dec.": "Dec",
}


def _normalize_date_string(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    # Remove ordinal suffixes
    s = _ORDINAL_SUFFIX_RE.sub(r'\1', s)
    # Normalize multiple spaces and commas
    s = re.sub(r'\s+', ' ', s)
    s = s.replace(" ,", ",")
    # Normalize abbreviated months with trailing periods
    for k, v in _MONTH_FIXES.items():
        s = s.replace(k, v)
    return s


def _try_parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    s = _normalize_date_string(s)
    # Common formats to try
    fmts = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d %Y",
        "%B %d %Y",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    # Fallback: try to extract "Month Day, Year" pattern roughly
    m = re.search(r'([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(2026)', s)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)}, {m.group(3)}", "%B %d, %Y").date()
        except Exception:
            try:
                return datetime.strptime(f"{m.group(1)} {m.group(2)}, {m.group(3)}", "%b %d, %Y").date()
            except Exception:
                pass
    return None


def _date_in_required_window(d: Optional[date]) -> bool:
    if d is None:
        return False
    return DATE_RANGE_START <= d <= DATE_RANGE_END


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _gather_membership_sources(item: ConcertItem) -> List[str]:
    urls: List[str] = []
    if _nonempty(item.tour_reference_url):
        urls.append(item.tour_reference_url)  # official tour page or venue event page
    if _nonempty(item.ticket_url):
        urls.append(item.ticket_url)  # official ticketing often confirms date/venue
    if item.capacity_reference_urls:
        urls.extend([u for u in item.capacity_reference_urls if _nonempty(u)])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification for a single concert                                           #
# --------------------------------------------------------------------------- #
async def verify_one_concert(
    evaluator: Evaluator,
    parent_node,
    item: ConcertItem,
    idx: int,
) -> None:
    # Sequential node for this concert
    concert_node = evaluator.add_sequential(
        id=f"concert_{idx+1}",
        desc=f"{['First','Second','Third','Fourth'][idx]} qualifying stadium concert",
        parent=parent_node,
        critical=False,  # allow partial credit across different concerts
    )

    # 1) Tour Membership (critical)
    tour_node = evaluator.add_leaf(
        id=f"tour_membership_{idx+1}",
        desc="Concert is part of Zach Bryan's official 'WITH HEAVEN ON TOUR' 2026 schedule",
        parent=concert_node,
        critical=True,
    )
    membership_claim = (
        f"Zach Bryan's 2026 'WITH HEAVEN ON TOUR' includes a concert at "
        f"{item.stadium_name or '[stadium]'} in {item.city or '[city]'}, {item.state or '[state]'} "
        f"on {item.date or '[date]'}."
    )
    await evaluator.verify(
        claim=membership_claim,
        node=tour_node,
        sources=_gather_membership_sources(item),
        additional_instruction=(
            "Verify on the provided official tour page and/or official venue/ticketing page that Zach Bryan is scheduled "
            "to perform at the specified stadium, city, state, and date in 2026. Allow minor formatting differences in "
            "names and date formats (e.g., 'July 1, 2026' vs '2026-07-01'). If any provided page clearly and explicitly "
            "lists this event, consider this supported."
        ),
    )

    # 2) Venue Requirements (critical, parallel group)
    venue_req_node = evaluator.add_parallel(
        id=f"venue_requirements_{idx+1}",
        desc="Venue meets all stadium and capacity requirements",
        parent=concert_node,
        critical=True,
    )

    # 2.a) Stadium type
    stadium_type_node = evaluator.add_leaf(
        id=f"stadium_type_{idx+1}",
        desc="Venue is an outdoor stadium (not an indoor arena or theater)",
        parent=venue_req_node,
        critical=True,
    )
    stadium_type_claim = (
        f"{item.stadium_name or '[stadium]'} is an outdoor/open-air stadium (not an indoor arena or theater)."
    )
    await evaluator.verify(
        claim=stadium_type_claim,
        node=stadium_type_node,
        sources=_gather_membership_sources(item),
        additional_instruction=(
            "Use the venue's official page or credible sources (e.g., Wikipedia) to confirm the venue is a stadium and is "
            "outdoor/open-air or has a retractable/open roof typically used as an outdoor stadium. If the venue is clearly "
            "an indoor arena or theater, this should be considered not supported."
        ),
    )

    # 2.b) US Location
    us_location_node = evaluator.add_leaf(
        id=f"us_location_{idx+1}",
        desc="Stadium is located in the United States",
        parent=venue_req_node,
        critical=True,
    )
    us_location_claim = (
        f"The stadium {item.stadium_name or '[stadium]'} is located in {item.city or '[city]'}, "
        f"{item.state or '[state]'}, United States."
    )
    await evaluator.verify(
        claim=us_location_claim,
        node=us_location_node,
        sources=_gather_membership_sources(item),
        additional_instruction=(
            "Confirm that the stadium's location is in the United States (including the District of Columbia). "
            "Use any of the provided official or credible pages."
        ),
    )

    # 2.c) Capacity threshold >= 60,000 (verified by sources)
    capacity_threshold_node = evaluator.add_leaf(
        id=f"capacity_threshold_{idx+1}",
        desc="Stadium has a documented seating capacity of at least 60,000 for concerts",
        parent=venue_req_node,
        critical=True,
    )
    capacity_claim = (
        f"The stadium {item.stadium_name or '[stadium]'} has a documented seating capacity of at least 60,000 "
        f"for concerts (or general seating) according to the provided source(s)."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_threshold_node,
        sources=item.capacity_reference_urls if item.capacity_reference_urls else _gather_membership_sources(item),
        additional_instruction=(
            "Look for an explicit capacity figure on the provided source(s). Prefer concert capacity when available. "
            "If concert-specific capacity isn't stated, accept overall stadium seating capacity if it is ≥ 60,000. "
            "Numbers like 'over 60,000', 'approx. 80,000', or exact counts ≥ 60,000 satisfy this check."
        ),
    )

    # 2.d) Reference URL documenting capacity is provided (existence)
    capacity_ref_exist = evaluator.add_custom_node(
        result=bool(item.capacity_reference_urls and len(item.capacity_reference_urls) > 0),
        id=f"reference_url_{idx+1}a",
        desc="Provide a reference URL documenting the stadium's capacity",
        parent=venue_req_node,
        critical=True,
    )

    # 3) Date compliance (critical)
    parsed_dt = _try_parse_date(item.date)
    date_ok = _date_in_required_window(parsed_dt)
    evaluator.add_custom_node(
        result=date_ok,
        id=f"date_compliance_{idx+1}",
        desc="Concert date falls between March 1, 2026 and August 31, 2026",
        parent=concert_node,
        critical=True,
    )

    # 4) Information completeness (critical, parallel)
    info_node = evaluator.add_parallel(
        id=f"information_completeness_{idx+1}",
        desc="All required information is provided",
        parent=concert_node,
        critical=True,
    )

    # 4.a) Basic details provided
    basic_details_ok = _nonempty(item.stadium_name) and _nonempty(item.city) and _nonempty(item.state) and _nonempty(item.date)
    evaluator.add_custom_node(
        result=basic_details_ok,
        id=f"basic_details_{idx+1}",
        desc="Stadium name, city, state, and exact date are provided",
        parent=info_node,
        critical=True,
    )

    # 4.b) Ticket URL is an official purchase page (verified by URL)
    ticket_node = evaluator.add_leaf(
        id=f"ticket_url_{idx+1}",
        desc="Official ticket purchase URL from an authorized platform is provided",
        parent=info_node,
        critical=True,
    )
    ticket_claim = (
        f"This URL is an official ticket purchase page for Zach Bryan's concert on {item.date or '[date]'} "
        f"at {item.stadium_name or '[stadium]'} in {item.city or '[city]'}, {item.state or '[state]'}."
    )
    await evaluator.verify(
        claim=ticket_claim,
        node=ticket_node,
        sources=item.ticket_url if _nonempty(item.ticket_url) else None,
        additional_instruction=(
            "The page should be an official purchase page (e.g., Ticketmaster, AXS, SeatGeek, the venue's own ticketing, "
            "or the artist's official ticketing). Look for a purchase flow or clear 'Buy Tickets' functionality for the "
            "specific event. Generic articles, press releases, or third‑party resale aggregators without official purchase "
            "are not acceptable."
        ),
    )

    # 4.c) Reference URL to tour page or venue confirmation is provided (existence)
    evaluator.add_custom_node(
        result=_nonempty(item.tour_reference_url),
        id=f"reference_url_{idx+1}b",
        desc="Reference URL to Zach Bryan's official tour page or venue confirmation is provided",
        parent=info_node,
        critical=True,
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # four concerts evaluated independently
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

    # Extract concerts data from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_concerts(),
        template_class=ConcertsExtraction,
        extraction_name="extracted_concerts",
    )

    # Ensure exactly four items (pad with empty if fewer; trim if more)
    items = list(extracted.concerts[:4])
    while len(items) < 4:
        items.append(ConcertItem())

    # Build verification tree per concert
    for i in range(4):
        await verify_one_concert(evaluator, root, items[i], i)

    return evaluator.get_summary()