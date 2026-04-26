import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "concert_venues_2026"
TASK_DESCRIPTION = (
    "I'm researching mid-sized concert venues across the United States for a potential tour. "
    "Find 4 dedicated concert or music venues that meet the following requirements:\n\n"
    "1. Each venue must be located in a different major US city with a population over 500,000\n"
    "2. Each venue must have a total capacity (seated and/or standing) between 2,000 and 8,000 people\n"
    "3. The venues must be primarily dedicated to concerts and music performances (not multi-purpose sports arenas or stadiums)\n"
    "4. Each venue must be currently operational and accepting bookings as of February 2026\n\n"
    "For each venue, provide:\n"
    "- The venue name\n"
    "- The city and state where it's located\n"
    "- The venue's total capacity\n"
    "- The official website URL where the capacity can be verified"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None  # keep as string to be robust to ranges/notes
    website_url: Optional[str] = None


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to 4 concert or music venues from the answer in order of appearance. For each venue, extract:
    - name: The venue name (string exactly as written in the answer)
    - city: The city name (string)
    - state: The state (string, if available; do not invent; use 2-letter abbreviation if provided, otherwise the exact string from the answer)
    - capacity: The total capacity as stated in the answer (string exactly as appears; may include commas, ranges, words like 'approx.')
    - website_url: The official venue website URL cited in the answer that can be used to verify capacity (single URL as a string; if multiple are shown, pick the one most likely to be the official site)

    Rules:
    - Do not invent any fields. If a field is missing in the answer, set it to null.
    - If the answer lists more than 4 venues, only extract the first 4.
    - If fewer than 4 venues are provided, return what is available. Do NOT create extra ones.

    Return a JSON object with a single field "venues" as an array of venue objects.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(s: Optional[str]) -> str:
    return s.strip() if isinstance(s, str) else ""


def parse_capacity_numbers(capacity_text: Optional[str]) -> List[int]:
    """
    Extract integer-like numbers from a capacity string.
    Examples:
    - "2,500 seats" -> [2500]
    - "2,000 - 3,500" -> [2000, 3500]
    - "approx 5k" -> try to catch 5, treat '5k' as 5000 (handle 'k' suffix)
    """
    if not capacity_text:
        return []
    text = capacity_text.lower().strip()

    # Handle 'k' notation like '5k', '7.5k'
    def expand_k_notation(t: str) -> str:
        return re.sub(r'(\d+(?:\.\d+)?)\s*k\b', lambda m: str(int(float(m.group(1)) * 1000)), t)

    text = expand_k_notation(text)

    # Remove commas for consistency
    text = text.replace(",", " ")

    # Extract integer tokens
    nums = re.findall(r'\b\d{1,6}\b', text)
    ints = []
    for n in nums:
        try:
            val = int(n)
            ints.append(val)
        except Exception:
            continue
    return ints


def capacity_in_range(capacity_text: Optional[str], low: int = 2000, high: int = 8000) -> bool:
    """
    Determine if the capacity text indicates any number within [low, high].
    This is a permissive check—if any extracted integer lies in the range, return True.
    """
    values = parse_capacity_numbers(capacity_text)
    return any(low <= v <= high for v in values)


def cities_unique(venues: List[VenueItem]) -> bool:
    """
    Check that all provided venues (first 4 slots) are in different US cities (city+state pairs).
    - If any missing city or state for a venue, treat as not unique to be conservative.
    """
    seen: set[Tuple[str, str]] = set()
    for v in venues[:4]:
        c = _safe_str(v.city).lower()
        s = _safe_str(v.state).lower()
        if not c or not s:
            return False
        pair = (c, s)
        if pair in seen:
            return False
        seen.add(pair)
    # Must have exactly 4 distinct pairs
    return len(seen) == 4


# --------------------------------------------------------------------------- #
# Verification logic for one venue                                            #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    idx_zero_based: int
) -> None:
    """
    Build verification subtree for a single venue.
    """
    idx = idx_zero_based + 1
    venue_node = evaluator.add_parallel(
        id=f"Venue_{idx}",
        desc=f"{['First','Second','Third','Fourth'][idx_zero_based]} concert venue meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # Subnode: Core Requirements
    core_node = evaluator.add_parallel(
        id=f"Venue_{idx}_Core_Requirements",
        desc=f"Core requirements for the {['first','second','third','fourth'][idx_zero_based]} venue",
        parent=venue_node,
        critical=False
    )

    # City population > 500,000 (critical)
    city_pop_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_City_Population",
        desc="Venue is located in a major US city with population over 500,000",
        parent=core_node,
        critical=True
    )
    city_name = _safe_str(venue.city)
    state_name = _safe_str(venue.state)
    if city_name and state_name:
        city_claim = (
            f"The city {city_name}, {state_name} has a population over 500,000 "
            f"(consider city proper using recent estimates around 2020–2025)."
        )
    elif city_name:
        city_claim = (
            f"The city {city_name} has a population over 500,000 "
            f"(consider city proper using recent estimates around 2020–2025)."
        )
    else:
        city_claim = "The provided city has a population over 500,000."

    await evaluator.verify(
        claim=city_claim,
        node=city_pop_leaf,
        sources=None,
        additional_instruction=(
            "Use general knowledge or widely accepted public data. "
            "Interpret 'major US city' as the incorporated city proper, not the metro area. "
            "If the city likely has fewer than 500,000 residents, mark as incorrect."
        )
    )

    # Capacity range 2,000–8,000 (critical) – local check from provided capacity text
    capacity_range_ok = capacity_in_range(venue.capacity, 2000, 8000)
    evaluator.add_custom_node(
        result=capacity_range_ok,
        id=f"Venue_{idx}_Capacity_Range",
        desc="Venue has a total capacity between 2,000 and 8,000 people",
        parent=core_node,
        critical=True
    )

    # Venue type dedicated to concerts/music (critical) – verify by website URL
    venue_type_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Venue_Type",
        desc="Venue is primarily dedicated to concerts and music performances (not a multi-purpose sports arena or stadium)",
        parent=core_node,
        critical=True
    )
    # We'll require website URL existence as precondition to avoid futile verification
    website_exists = bool(_safe_str(venue.website_url))
    # Still attempt verify; Evaluator will handle missing sources (should fail)
    await evaluator.verify(
        claim=f"The venue '{_safe_str(venue.name)}' is primarily a concert/music venue (not a multi-purpose sports arena or stadium).",
        node=venue_type_leaf,
        sources=_safe_str(venue.website_url) if website_exists else None,
        additional_instruction=(
            "Rely on the official website content: look for terms like 'music venue', 'concert hall', "
            "'live music', 'performing arts center', or event calendars populated with concerts. "
            "If the page indicates it's a sports arena/stadium or primarily used for sports teams, mark as not dedicated."
        )
    )

    # Operational status as of Feb 2026 (critical) – verify by website URL
    operational_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Operational_Status",
        desc="Venue is currently operational and accepting bookings as of February 2026",
        parent=core_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"As of February 2026, the venue '{_safe_str(venue.name)}' is operational and accepting bookings or has upcoming events."
        ),
        node=operational_leaf,
        sources=_safe_str(venue.website_url) if website_exists else None,
        additional_instruction=(
            "Use the official website page provided. Evidence can include an events calendar with upcoming shows "
            "in 2026, 'Book now' or 'Buy tickets' calls-to-action, or rental/booking information indicating availability. "
            "If the website indicates closure or no events/booking info, mark as not operational."
        )
    )

    # Subnode: Output Information
    out_node = evaluator.add_parallel(
        id=f"Venue_{idx}_Output_Information",
        desc=f"Required output information for the {['first','second','third','fourth'][idx_zero_based]} venue",
        parent=venue_node,
        critical=False
    )

    # Name provided (critical)
    evaluator.add_custom_node(
        result=bool(_safe_str(venue.name)),
        id=f"Venue_{idx}_Name_Provided",
        desc="Venue name is provided",
        parent=out_node,
        critical=True
    )

    # City & state provided (critical)
    evaluator.add_custom_node(
        result=bool(_safe_str(venue.city)) and bool(_safe_str(venue.state)),
        id=f"Venue_{idx}_City_State_Provided",
        desc="City and state location are provided",
        parent=out_node,
        critical=True
    )

    # Capacity provided (critical)
    evaluator.add_custom_node(
        result=bool(_safe_str(venue.capacity)),
        id=f"Venue_{idx}_Capacity_Provided",
        desc="Venue's total capacity number is provided",
        parent=out_node,
        critical=True
    )

    # Website URL provided (critical)
    website_exists_node = evaluator.add_custom_node(
        result=website_exists,
        id=f"Venue_{idx}_Website_URL",
        desc="Official website URL is provided that confirms the venue's capacity and operational status",
        parent=out_node,
        critical=True
    )

    # Capacity supported by website (critical) – explicit verification against URL
    capacity_supported_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Capacity_Supported",
        desc="The provided official website includes the venue capacity information matching or reasonably supporting the stated capacity",
        parent=out_node,
        critical=True
    )
    capacity_claim = (
        f"The official website for '{_safe_str(venue.name)}' provides the venue capacity information "
        f"matching or reasonably consistent with '{_safe_str(venue.capacity)}'."
        if _safe_str(venue.capacity) else
        f"The official website for '{_safe_str(venue.name)}' provides the venue capacity information."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_supported_leaf,
        sources=_safe_str(venue.website_url) if website_exists else None,
        additional_instruction=(
            "On the official website, look for words like 'capacity', 'seating capacity', 'standing room', 'specs', 'technical'. "
            "Allow minor formatting differences (commas, tildes, 'approx'). If capacity is clearly present, accept. "
            "If no capacity information is visible, mark as unsupported."
        )
    )

    # Operational supported by website (critical) – explicit verification against URL
    operational_supported_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Operational_Supported",
        desc="The provided official website includes clear signs that the venue is operational/booking (events, tickets, rentals) as of Feb 2026",
        parent=out_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official website for '{_safe_str(venue.name)}' shows that the venue is operational and accepting bookings or hosting events in 2026.",
        node=operational_supported_leaf,
        sources=_safe_str(venue.website_url) if website_exists else None,
        additional_instruction=(
            "Evidence includes an events calendar listing upcoming shows in 2026, ticket purchase links, "
            "or a rental/booking page indicating current availability. If evidence of ongoing operations is absent, mark unsupported."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 'mid-sized concert venues in major US cities' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # per rubric
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

    # 1) Extract up to 4 venues
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    venues: List[VenueItem] = extraction.venues[:4] if extraction.venues else []
    # Pad to exactly 4 entries with empty placeholders for consistent tree shape
    while len(venues) < 4:
        venues.append(VenueItem())

    # 2) Global critical check: all 4 venues in different cities (city+state pairs)
    uniqueness_ok = cities_unique(venues)
    evaluator.add_custom_node(
        result=uniqueness_ok,
        id="City_Uniqueness",
        desc="All 4 venues are located in different US cities (no two venues in the same city)",
        parent=root,
        critical=True
    )

    # 3) Per-venue verification trees
    for i in range(4):
        await verify_single_venue(evaluator, root, venues[i], i)

    # 4) Return structured result
    return evaluator.get_summary()