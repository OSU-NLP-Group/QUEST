import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "arena_concerts_2026"
TASK_DESCRIPTION = (
    "Find three major arena concerts scheduled between June 1 and September 30, 2026, "
    "that meet specific venue, tour, and ticketing requirements, and provide verification URLs."
)

DATE_RANGE_START = datetime(2026, 6, 1)
DATE_RANGE_END = datetime(2026, 9, 30)
ALLOWED_TICKET_PLATFORMS = ["ticketmaster.com", "livenation.com", "seatgeek.com"]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ConcertItem(BaseModel):
    artist_name: Optional[str] = None
    tour_name: Optional[str] = None
    concert_date: Optional[str] = None
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    venue_concert_capacity: Optional[str] = None
    vip_confirmation: Optional[str] = None

    # URLs
    event_details_urls: List[str] = Field(default_factory=list)  # Official tour pages or ticketing platforms
    venue_capacity_urls: List[str] = Field(default_factory=list)  # Venue official page/Wikipedia/other reliable sources
    tour_page_urls: List[str] = Field(default_factory=list)  # Official tour pages that list dates
    ticketing_urls: List[str] = Field(default_factory=list)  # Direct ticketing pages (Ticketmaster/Live Nation/SeatGeek)
    city_population_urls: List[str] = Field(default_factory=list)  # City page (Wikipedia/Census) if provided


class ConcertsExtraction(BaseModel):
    concerts: List[ConcertItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_concerts() -> str:
    return (
        "Extract up to three concerts mentioned in the answer that the user intends to propose. "
        "For each concert, return the following fields:\n"
        "1. artist_name: The artist performing as headliner.\n"
        "2. tour_name: The tour name for this set of dates.\n"
        "3. concert_date: The specific concert date in any clearly readable format (e.g., 'June 12, 2026' or '2026-06-12').\n"
        "4. venue_name: The venue name.\n"
        "5. city: The city where the venue is located.\n"
        "6. state: The US state abbreviation or full name.\n"
        "7. venue_concert_capacity: The stated venue seating capacity for concerts (as quoted in the answer), if provided; else null.\n"
        "8. vip_confirmation: A short phrase or sentence from the answer confirming VIP packages/upgrades availability (e.g., 'VIP available', 'VIP packages offered'); if not present, set to null.\n"
        "9. event_details_urls: URLs from official tour pages or major ticketing platforms (Ticketmaster, Live Nation, SeatGeek) that show event details.\n"
        "10. venue_capacity_urls: URLs from the venue website or other reliable sources that state capacity and/or indoor arena type.\n"
        "11. tour_page_urls: URLs that list tour dates and can be used to verify that the tour has at least 10 announced dates.\n"
        "12. ticketing_urls: Direct ticketing page URLs for the concert, ideally from Ticketmaster, Live Nation, or SeatGeek.\n"
        "13. city_population_urls: URLs (e.g., Wikipedia/Census) cited in the answer to support that the city population is over 500,000, if any.\n\n"
        "Return a JSON object with a top-level 'concerts' array containing at most three items in the order they appear in the answer. "
        "For any field missing, return null (for strings) or an empty list (for URLs). "
        "Extract only URLs explicitly present in the answer (plain or markdown). Do not invent or infer URLs."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _normalize_state(s: Optional[str]) -> Optional[str]:
    if not _non_empty(s):
        return None
    return s.strip().lower()


def _extract_largest_int(text: Optional[str]) -> Optional[int]:
    if not _non_empty(text):
        return None
    digits = re.findall(r"\d{1,3}(?:,\d{3})+|\d{4,6}", text.replace("+", ""))
    if not digits:
        digits = re.findall(r"\d{4,6}", text)  # fallback plain long numbers
    if not digits:
        digits = re.findall(r"\d{2,3}", text)  # fallback shorter numbers
    if not digits:
        return None
    values = []
    for d in digits:
        try:
            values.append(int(d.replace(",", "")))
        except Exception:
            continue
    if not values:
        return None
    return max(values)


def _parse_date_any(date_str: Optional[str]) -> Optional[datetime]:
    if not _non_empty(date_str):
        return None
    s = date_str.strip()
    fmts = [
        "%B %d, %Y",   # June 12, 2026
        "%b %d, %Y",   # Jun 12, 2026
        "%Y-%m-%d",    # 2026-06-12
        "%m/%d/%Y",    # 06/12/2026
        "%m-%d-%Y",    # 06-12-2026
        "%A, %B %d, %Y",  # Friday, June 12, 2026
        "%a, %b %d, %Y",  # Fri, Jun 12, 2026
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    # Remove day-of-week prefix if present and retry
    if "," in s:
        try:
            cut = s[s.index(",") + 1 :].strip()
            for fmt in fmts:
                try:
                    return datetime.strptime(cut, fmt)
                except Exception:
                    pass
        except Exception:
            pass
    return None


def _date_in_required_range(date_str: Optional[str]) -> bool:
    d = _parse_date_any(date_str)
    if not d:
        return False
    return DATE_RANGE_START <= d <= DATE_RANGE_END


def _merge_urls(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for u in lst:
            if u and u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification for one concert                                                #
# --------------------------------------------------------------------------- #
async def verify_concert(
    evaluator: Evaluator,
    parent_node,
    item: ConcertItem,
    idx: int,
) -> None:
    """
    Build verification sub-tree and run checks for a single concert.
    """
    i = idx + 1
    concert_node = evaluator.add_parallel(
        id=f"concert_{i}",
        desc=f"Concert {i} (a qualifying concert meeting all requirements)",
        parent=parent_node,
        critical=False,
    )

    # 1) Required fields
    req_node = evaluator.add_parallel(
        id=f"concert_{i}_required_fields",
        desc=f"All required fields for concert {i} are provided",
        parent=concert_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(item.artist_name),
        id=f"concert_{i}_artist_name_provided",
        desc=f"Artist name is provided (concert {i})",
        parent=req_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(item.tour_name),
        id=f"concert_{i}_tour_name_provided",
        desc=f"Tour name is provided (concert {i})",
        parent=req_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(item.concert_date),
        id=f"concert_{i}_concert_date_provided",
        desc=f"Concert date is provided (concert {i})",
        parent=req_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(item.venue_name),
        id=f"concert_{i}_venue_name_provided",
        desc=f"Venue name is provided (concert {i})",
        parent=req_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(item.city) and _non_empty(item.state),
        id=f"concert_{i}_city_state_provided",
        desc=f"Venue city and state are provided (concert {i})",
        parent=req_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(item.venue_concert_capacity),
        id=f"concert_{i}_venue_concert_capacity_provided",
        desc=f"Venue seating capacity for concerts is provided (concert {i})",
        parent=req_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(item.vip_confirmation),
        id=f"concert_{i}_vip_confirmation_provided",
        desc=f"Confirmation that VIP packages/upgrades are available is provided (concert {i})",
        parent=req_node,
        critical=True,
    )

    # 2) Date constraint
    date_node = evaluator.add_parallel(
        id=f"concert_{i}_date_constraint",
        desc=f"Date constraint is satisfied (concert {i})",
        parent=concert_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_date_in_required_range(item.concert_date),
        id=f"concert_{i}_date_in_range",
        desc="Concert date falls between June 1, 2026 and September 30, 2026 (inclusive)",
        parent=date_node,
        critical=True,
    )

    # 3) Venue constraints
    venue_node = evaluator.add_parallel(
        id=f"concert_{i}_venue_constraints",
        desc=f"Venue requirements are satisfied (concert {i})",
        parent=concert_node,
        critical=True,
    )

    indoor_leaf = evaluator.add_leaf(
        id=f"concert_{i}_indoor_arena",
        desc="Venue is an indoor arena",
        parent=venue_node,
        critical=True,
    )
    indoor_sources = item.venue_capacity_urls if item.venue_capacity_urls else item.event_details_urls
    await evaluator.verify(
        claim=f"The venue '{item.venue_name or ''}' is an indoor arena.",
        node=indoor_leaf,
        sources=indoor_sources if indoor_sources else None,
        additional_instruction=(
            "Confirm 'indoor arena' using venue official page or a reliable source (e.g., Wikipedia). "
            "If the page clearly indicates the venue is an arena with a roof or explicitly indoor, mark as supported."
        ),
    )

    capacity_leaf = evaluator.add_leaf(
        id=f"concert_{i}_capacity_at_least_18000",
        desc="Venue has seating capacity of at least 18,000 for concerts",
        parent=venue_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The venue '{item.venue_name or ''}' has a concert seating capacity of at least 18,000.",
        node=capacity_leaf,
        sources=item.venue_capacity_urls if item.venue_capacity_urls else None,
        additional_instruction=(
            "Look specifically for concert capacity. If only a general capacity is listed but exceeds 18,000, "
            "that is acceptable. If multiple configurations exist, use the concert capacity number."
        ),
    )

    major_city_leaf = evaluator.add_leaf(
        id=f"concert_{i}_major_us_city_over_500k",
        desc="Concert takes place in a major US city with population over 500,000",
        parent=venue_node,
        critical=True,
    )
    pop_sources = item.city_population_urls if item.city_population_urls else None
    await evaluator.verify(
        claim=f"The city '{item.city or ''}, {item.state or ''}' has a population over 500,000 (city proper).",
        node=major_city_leaf,
        sources=pop_sources,
        additional_instruction=(
            "Use reliable sources (e.g., Wikipedia or US Census) when available. "
            "If no URL is provided, rely on widely known figures for major US cities."
        ),
    )

    # 4) Tour constraints
    tour_node = evaluator.add_parallel(
        id=f"concert_{i}_tour_constraints",
        desc=f"Tour requirements are satisfied (concert {i})",
        parent=concert_node,
        critical=True,
    )

    multi_city_leaf = evaluator.add_leaf(
        id=f"concert_{i}_multi_city_tour_10_dates",
        desc="Concert is part of a multi-city arena tour with at least 10 announced tour dates",
        parent=tour_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The tour '{item.tour_name or ''}' by '{item.artist_name or ''}' has at least 10 announced dates across multiple cities."
        ),
        node=multi_city_leaf,
        sources=item.tour_page_urls if item.tour_page_urls else None,
        additional_instruction=(
            "Check the official tour page or verified listings. Counting of dates can be approximate; "
            "if the page clearly lists >= 10 dates, mark supported."
        ),
    )

    headliner_leaf = evaluator.add_leaf(
        id=f"concert_{i}_headliner_not_festival_or_opener",
        desc="Artist is the headliner (not a festival appearance and not an opening act)",
        parent=tour_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"For the concert on '{item.concert_date or ''}' at '{item.venue_name or ''}' in '{item.city or ''}, {item.state or ''}', "
            f"the artist '{item.artist_name or ''}' is the headliner (not part of a festival and not an opening act)."
        ),
        node=headliner_leaf,
        sources=item.event_details_urls if item.event_details_urls else None,
        additional_instruction=(
            "On event/ticketing pages, confirm the artist is the main bill/headliner. "
            "If the page indicates a festival or that the artist is supporting another headliner, mark as not supported."
        ),
    )

    # 5) Ticketing constraints
    ticket_node = evaluator.add_parallel(
        id=f"concert_{i}_ticketing_constraints",
        desc=f"Ticketing requirements are satisfied (concert {i})",
        parent=concert_node,
        critical=True,
    )

    vip_leaf = evaluator.add_leaf(
        id=f"concert_{i}_vip_available",
        desc="VIP ticket packages or upgrades are available for this concert",
        parent=ticket_node,
        critical=True,
    )
    vip_sources = _merge_urls(item.event_details_urls, item.tour_page_urls)
    await evaluator.verify(
        claim=f"VIP ticket packages or upgrades are available for the concert of '{item.artist_name or ''}' at '{item.venue_name or ''}'.",
        node=vip_leaf,
        sources=vip_sources if vip_sources else None,
        additional_instruction=(
            "Look for 'VIP', 'VIP Packages', 'VIP Upgrade', 'Platinum', 'Meet & Greet', or similar on the provided pages."
        ),
    )

    allowed_platform_leaf = evaluator.add_leaf(
        id=f"concert_{i}_allowed_ticket_platform",
        desc="Tickets are sold through Ticketmaster, Live Nation, or SeatGeek",
        parent=ticket_node,
        critical=True,
    )
    platform_sources = _merge_urls(item.ticketing_urls, item.event_details_urls)
    await evaluator.verify(
        claim=(
            f"Tickets for this concert are sold through one of the allowed platforms: Ticketmaster, Live Nation, or SeatGeek."
        ),
        node=allowed_platform_leaf,
        sources=platform_sources if platform_sources else None,
        additional_instruction=(
            "Confirm that at least one provided ticket link is hosted on 'ticketmaster.com', 'livenation.com', or 'seatgeek.com'. "
            "Domain and page content should indicate a purchasable ticket interface."
        ),
    )

    # 6) Verification sources presence/appropriateness
    sources_node = evaluator.add_parallel(
        id=f"concert_{i}_verification_sources",
        desc=f"Required verification URLs are provided and appropriate (concert {i})",
        parent=concert_node,
        critical=True,
    )

    event_details_leaf = evaluator.add_leaf(
        id=f"concert_{i}_event_details_verifiable_url",
        desc="Event/ticket URLs collectively verify artist, tour, date, venue, and city/state",
        parent=sources_node,
        critical=True,
    )
    event_sources = _merge_urls(item.event_details_urls, item.ticketing_urls)
    await evaluator.verify(
        claim=(
            f"The provided event/ticket URLs confirm the artist '{item.artist_name or ''}', the concert date '{item.concert_date or ''}', "
            f"the venue '{item.venue_name or ''}', and the city/state '{(item.city or '')}, {(item.state or '')}'. "
            f"The tour name '{item.tour_name or ''}' should be confirmed if present."
        ),
        node=event_details_leaf,
        sources=event_sources if event_sources else None,
        additional_instruction=(
            "A single URL may not include all fields; use any page that explicitly confirms these details on its own to satisfy this check."
        ),
    )

    venue_capacity_leaf = evaluator.add_leaf(
        id=f"concert_{i}_venue_capacity_verifiable_url",
        desc="Venue/capacity URLs verify venue concert capacity and/or arena type",
        parent=sources_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The provided venue or reliable source URLs verify that '{item.venue_name or ''}' is an indoor arena and/or provide its concert capacity figure."
        ),
        node=venue_capacity_leaf,
        sources=item.venue_capacity_urls if item.venue_capacity_urls else None,
        additional_instruction=(
            "Either an explicit capacity figure or an explicit indoor-arena confirmation is acceptable for this check."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the arena concerts 2026 task.
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

    # Extract concert items
    extracted = await evaluator.extract(
        prompt=prompt_extract_concerts(),
        template_class=ConcertsExtraction,
        extraction_name="concerts_extraction",
    )

    # Normalize count to exactly 3 entries (pad with empty if needed)
    concerts: List[ConcertItem] = list(extracted.concerts[:3])
    while len(concerts) < 3:
        concerts.append(ConcertItem())

    # Add ground truth/task constants
    evaluator.add_ground_truth(
        {
            "date_range_start": DATE_RANGE_START.strftime("%Y-%m-%d"),
            "date_range_end": DATE_RANGE_END.strftime("%Y-%m-%d"),
            "allowed_ticket_platforms": ALLOWED_TICKET_PLATFORMS,
            "require_indoor_arena": True,
            "min_concert_capacity": 18000,
            "require_major_city_over_500k": True,
            "require_headliner": True,
            "require_multi_city_tour_min_dates": 10,
            "require_state_diversity": True,
        },
        gt_type="task_requirements",
    )

    # Verify each concert
    for i, item in enumerate(concerts):
        await verify_concert(evaluator, root, item, i)

    # Global check: state diversity
    states = [(_normalize_state(c.state) or "") for c in concerts]
    unique_states = set([s for s in states if _non_empty(s)])
    state_diversity_result = len(unique_states) == 3

    evaluator.add_custom_node(
        result=state_diversity_result,
        id="state_diversity",
        desc="The three concerts are in three different US states",
        parent=root,
        critical=True,
    )

    # Optional custom info: extracted states and dates
    evaluator.add_custom_info(
        {
            "extracted_states": states,
            "concert_dates": [c.concert_date for c in concerts],
            "venues": [c.venue_name for c in concerts],
        },
        info_type="extraction_summary",
    )

    return evaluator.get_summary()