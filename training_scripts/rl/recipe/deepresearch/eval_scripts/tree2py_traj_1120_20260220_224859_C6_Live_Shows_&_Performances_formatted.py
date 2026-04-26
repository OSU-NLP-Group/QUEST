import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_tour_venues"
TASK_DESCRIPTION = (
    "A touring production company is planning a multi-city entertainment tour across the United States and needs to "
    "identify appropriate venues for different types of performances. For each of the following four requirements, "
    "identify the venue name, its specific location, and its capacity:\n\n"
    "1. The largest Broadway theater in New York City by seating capacity\n"
    "2. A major indoor concert arena in Chicago, Illinois, with a concert seating capacity of at least 20,000\n"
    "3. A major annual music festival held in California with a daily attendance capacity of at least 100,000 people\n"
    "4. A major annual music festival held in Illinois with a daily attendance capacity of at least 100,000 people\n\n"
    "For each venue, provide the official name, the specific city and state (and venue name where applicable for "
    "festivals), and the exact capacity figure with supporting reference URLs."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    """
    Generic extracted info for a venue/festival item.
    - official_name: venue name (theater/arena) or festival name
    - city/state: specific city and state claimed in the answer
    - venue_site_name: for festivals, the specific venue/grounds name (e.g., 'Empire Polo Club')
    - capacity: capacity figure exactly as presented in the answer (string)
    - identification_urls: URLs cited to describe/name/location/classification
    - capacity_urls: URLs cited to support the capacity (and status like 'largest' where applicable)
    """
    official_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    venue_site_name: Optional[str] = None
    capacity: Optional[str] = None
    identification_urls: List[str] = Field(default_factory=list)
    capacity_urls: List[str] = Field(default_factory=list)


class TourVenuesExtraction(BaseModel):
    """All four required items extracted from the answer."""
    broadway_theater: Optional[VenueItem] = None
    chicago_arena: Optional[VenueItem] = None
    california_festival: Optional[VenueItem] = None
    illinois_festival: Optional[VenueItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract structured information for exactly four items from the answer text:

    1) Largest Broadway theater in New York City by seating capacity
    2) A major indoor concert arena in Chicago, Illinois, with a concert capacity of at least 20,000
    3) A major annual music festival in California with daily capacity ≥ 100,000
    4) A major annual music festival in Illinois with daily capacity ≥ 100,000

    For each item, extract the following fields:
    - official_name: The official venue name (for theater/arena) or the festival's official name.
    - city: The specific city name stated in the answer.
    - state: The U.S. state stated in the answer (e.g., 'New York', 'Illinois', 'California').
    - venue_site_name: For festivals only, the specific grounds or venue name where the festival is held (e.g., 'Empire Polo Club'); if not applicable, set to null.
    - capacity: The capacity figure exactly as written in the answer (e.g., '1,933 seats', '100,000 per day', '20,000+').
    - identification_urls: All URLs cited that describe/name/classify the venue/festival (e.g., official site or Wikipedia).
    - capacity_urls: All URLs cited that specifically support the seating/attendance capacity or claims like 'largest'.

    Return a JSON object with keys:
    - broadway_theater: VenueItem for item 1
    - chicago_arena: VenueItem for item 2
    - california_festival: VenueItem for item 3
    - illinois_festival: VenueItem for item 4

    Rules:
    - Only include URLs that are explicitly present in the answer text. Do not invent or infer URLs.
    - If a field is missing in the answer, set it to null (or [] for URL lists).
    - Keep capacity as a string exactly as presented (do not normalize to numbers).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _parse_capacity_to_int(cap: Optional[str]) -> Optional[int]:
    """
    Try to parse a capacity string into an integer, extracting the most plausible numeric value.

    Examples:
    - '1,933 seats' -> 1933
    - '100,000 per day' -> 100000
    - '20k' or '20 K' -> 20000
    - '100,000+' -> 100000
    - '80,000–100,000' -> 100000 (largest number found)
    """
    if not cap:
        return None

    nums: List[int] = []

    # Handle patterns like "20k", "20 k", "20.5k"
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*[kK]\b', cap):
        try:
            val = float(m.group(1)) * 1000
            nums.append(int(round(val)))
        except Exception:
            pass

    # Handle plain digit groups with commas/spaces, e.g., "100,000", "1 933"
    for m in re.finditer(r'\d[\d,\s]*', cap):
        raw = re.sub(r'[\s,]', '', m.group(0))
        if raw.isdigit():
            try:
                nums.append(int(raw))
            except Exception:
                pass

    if not nums:
        return None
    return max(nums)


def _has_any_digit(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'\d', text))


def _combine_urls(*url_lists: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in url_lists:
        for u in lst:
            u = (u or "").strip()
            if u and u not in combined:
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_broadway_theater(evaluator: Evaluator, parent_node, info: Optional[VenueItem]) -> None:
    """
    Venue 1: Largest Broadway theater in NYC by seating capacity.
    """
    venue_node = evaluator.add_sequential(
        id="venue_1_broadway_theater",
        desc="Identify the largest Broadway theater in New York City",
        parent=parent_node,
        critical=False  # Allow partial across top-level venues
    )

    # Identification block (critical)
    ident_node = evaluator.add_parallel(
        id="venue_1_identification",
        desc="Provide the name of a Broadway theater",
        parent=venue_node,
        critical=True
    )

    # Name provided (existence)
    name_ok = bool(info and info.official_name and info.official_name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id="venue_1_name",
        desc="Theater name is provided",
        parent=ident_node,
        critical=True
    )

    # Reference URL provided (existence)
    ident_urls_ok = bool(info and info.identification_urls and len(info.identification_urls) > 0)
    evaluator.add_custom_node(
        result=ident_urls_ok,
        id="venue_1_reference",
        desc="Reference URL provided for the theater",
        parent=ident_node,
        critical=True
    )

    # Broadway classification (verify with sources)
    class_leaf = evaluator.add_leaf(
        id="venue_1_broadway_classification",
        desc="Theater is classified as a Broadway theater (located in Theater District or Lincoln Center, Manhattan)",
        parent=ident_node,
        critical=True
    )
    name = info.official_name if info and info.official_name else ""
    claim = f"{name} is a Broadway theater located in Manhattan (Theater District or Lincoln Center) in New York City."
    await evaluator.verify(
        claim=claim,
        node=class_leaf,
        sources=(info.identification_urls if info else []),
        additional_instruction=(
            "Accept if the page clearly states it is a Broadway theater (Broadway house) in Manhattan. "
            "Minor phrasing differences are OK."
        )
    )

    # Capacity verification block (critical)
    capacity_node = evaluator.add_parallel(
        id="venue_1_capacity_verification",
        desc="Verify the theater has the highest seating capacity among Broadway theaters",
        parent=venue_node,
        critical=True
    )

    # Capacity value provided (existence)
    capacity_value_ok = bool(info and info.capacity and _has_any_digit(info.capacity))
    evaluator.add_custom_node(
        result=capacity_value_ok,
        id="venue_1_capacity_value",
        desc="Specific seating capacity number is provided",
        parent=capacity_node,
        critical=True
    )

    # Broadway minimum threshold ≥ 500 (logic check)
    cap_num = _parse_capacity_to_int(info.capacity if info else None)
    evaluator.add_custom_node(
        result=(cap_num is not None and cap_num >= 500),
        id="venue_1_minimum_threshold",
        desc="Seating capacity meets Broadway minimum of 500 seats",
        parent=capacity_node,
        critical=True
    )

    # Largest status among Broadway theaters (verify with sources)
    largest_leaf = evaluator.add_leaf(
        id="venue_1_largest_status",
        desc="Theater is identified as having the largest capacity among Broadway theaters",
        parent=capacity_node,
        critical=True
    )
    largest_claim = f"{name} has the largest seating capacity among Broadway theaters in New York City."
    await evaluator.verify(
        claim=largest_claim,
        node=largest_leaf,
        sources=_combine_urls(*(info.identification_urls if info else []), *(info.capacity_urls if info else [])),
        additional_instruction=(
            "Verify that among Broadway theaters, this one has the largest seating capacity. "
            "Rely on lists or authoritative sources comparing Broadway house capacities."
        )
    )

    # Capacity reference provided (existence)
    cap_urls_ok = bool(info and info.capacity_urls and len(info.capacity_urls) > 0)
    evaluator.add_custom_node(
        result=cap_urls_ok,
        id="venue_1_capacity_reference",
        desc="Reference URL provided for capacity information",
        parent=capacity_node,
        critical=True
    )


async def verify_chicago_arena(evaluator: Evaluator, parent_node, info: Optional[VenueItem]) -> None:
    """
    Venue 2: Major indoor concert arena in Chicago, IL with capacity ≥ 20,000.
    """
    venue_node = evaluator.add_sequential(
        id="venue_2_chicago_arena",
        desc="Identify a major concert arena in Chicago with capacity of 20,000 or more",
        parent=parent_node,
        critical=False
    )

    # Identification block (critical)
    ident_node = evaluator.add_parallel(
        id="venue_2_identification",
        desc="Provide the name of an arena in Chicago",
        parent=venue_node,
        critical=True
    )

    # Arena name provided (existence)
    name_ok = bool(info and info.official_name and info.official_name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id="venue_2_name",
        desc="Arena name is provided",
        parent=ident_node,
        critical=True
    )

    # Location verification: Chicago, Illinois (verify with sources)
    loc_leaf = evaluator.add_leaf(
        id="venue_2_location",
        desc="Arena is located in Chicago, Illinois",
        parent=ident_node,
        critical=True
    )
    name = info.official_name if info and info.official_name else ""
    loc_claim = f"{name} is located in Chicago, Illinois."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=(info.identification_urls if info else []),
        additional_instruction="Verify that the arena's location is Chicago, Illinois (Chicago, IL)."
    )

    # Reference URL provided (existence)
    ident_urls_ok = bool(info and info.identification_urls and len(info.identification_urls) > 0)
    evaluator.add_custom_node(
        result=ident_urls_ok,
        id="venue_2_reference",
        desc="Reference URL provided for the arena",
        parent=ident_node,
        critical=True
    )

    # Capacity verification block (critical)
    capacity_node = evaluator.add_parallel(
        id="venue_2_capacity_verification",
        desc="Verify the arena has concert capacity of 20,000 or more",
        parent=venue_node,
        critical=True
    )

    # Specific concert capacity number provided (existence)
    capacity_value_ok = bool(info and info.capacity and _has_any_digit(info.capacity))
    evaluator.add_custom_node(
        result=capacity_value_ok,
        id="venue_2_capacity_value",
        desc="Specific concert capacity number is provided",
        parent=capacity_node,
        critical=True
    )

    # Concert capacity threshold ≥ 20,000 (logic check)
    cap_num = _parse_capacity_to_int(info.capacity if info else None)
    evaluator.add_custom_node(
        result=(cap_num is not None and cap_num >= 20000),
        id="venue_2_capacity_threshold",
        desc="Concert capacity meets or exceeds 20,000",
        parent=capacity_node,
        critical=True
    )

    # Capacity reference provided (existence)
    cap_urls_ok = bool(info and info.capacity_urls and len(info.capacity_urls) > 0)
    evaluator.add_custom_node(
        result=cap_urls_ok,
        id="venue_2_capacity_reference",
        desc="Reference URL provided for capacity information",
        parent=capacity_node,
        critical=True
    )


async def verify_festival(
    evaluator: Evaluator,
    parent_node,
    info: Optional[VenueItem],
    venue_id_prefix: str,
    state_name: str,
    threshold: int = 100000
) -> None:
    """
    Venue 3/4: Major annual music festival with daily capacity threshold (state_name ∈ {California, Illinois}).
    """
    venue_node = evaluator.add_sequential(
        id=f"{venue_id_prefix}",
        desc=f"Identify a major music festival in {state_name} with daily capacity of {threshold} or more",
        parent=parent_node,
        critical=False
    )

    # Identification block (critical)
    ident_node = evaluator.add_parallel(
        id=f"{venue_id_prefix}_identification",
        desc=f"Provide the name of a music festival in {state_name}",
        parent=venue_node,
        critical=True
    )

    # Festival name provided (existence)
    name_ok = bool(info and info.official_name and info.official_name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id=f"{venue_id_prefix}_festival_name",
        desc="Music festival name is provided",
        parent=ident_node,
        critical=True
    )

    # Reference URL provided (existence)
    ident_urls_ok = bool(info and info.identification_urls and len(info.identification_urls) > 0)
    evaluator.add_custom_node(
        result=ident_urls_ok,
        id=f"{venue_id_prefix}_reference",
        desc="Reference URL provided for the festival",
        parent=ident_node,
        critical=True
    )

    # Specific venue name and city in state provided and verified with sources
    loc_leaf = evaluator.add_leaf(
        id=f"{venue_id_prefix}_venue_location",
        desc=f"Specific venue name and city in {state_name} are provided",
        parent=ident_node,
        critical=True
    )
    fest_name = info.official_name if info and info.official_name else ""
    venue_site = info.venue_site_name if info and info.venue_site_name else ""
    city = info.city if info and info.city else ""
    state = info.state if info and info.state else ""
    loc_claim = f"{fest_name} is held at {venue_site} in {city}, {state_name}."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=(info.identification_urls if info else []),
        additional_instruction=(
            f"Verify that the festival takes place in {state_name}, and that the provided venue/grounds name and "
            f"city are correct. Minor naming variants are acceptable."
        )
    )

    # Capacity verification block (critical)
    capacity_node = evaluator.add_parallel(
        id=f"{venue_id_prefix}_capacity_verification",
        desc=f"Verify the festival has daily capacity of {threshold} or more",
        parent=venue_node,
        critical=True
    )

    # Specific daily capacity number provided (existence)
    capacity_value_ok = bool(info and info.capacity and _has_any_digit(info.capacity))
    evaluator.add_custom_node(
        result=capacity_value_ok,
        id=f"{venue_id_prefix}_capacity_value",
        desc="Specific daily capacity number is provided",
        parent=capacity_node,
        critical=True
    )

    # Daily capacity threshold ≥ threshold (logic check)
    cap_num = _parse_capacity_to_int(info.capacity if info else None)
    evaluator.add_custom_node(
        result=(cap_num is not None and cap_num >= threshold),
        id=f"{venue_id_prefix}_capacity_threshold",
        desc=f"Daily capacity meets or exceeds {threshold:,}",
        parent=capacity_node,
        critical=True
    )

    # Capacity reference provided (existence)
    cap_urls_ok = bool(info and info.capacity_urls and len(info.capacity_urls) > 0)
    evaluator.add_custom_node(
        result=cap_urls_ok,
        id=f"{venue_id_prefix}_capacity_reference",
        desc="Reference URL provided for capacity information",
        parent=capacity_node,
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
) -> Dict[str, Any]:
    """
    Build the verification tree and evaluate the answer according to the rubric.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates venues independently
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

    # Extract all four items from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=TourVenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Build subtrees for each required item
    await verify_broadway_theater(evaluator, root, extracted.broadway_theater)
    await verify_chicago_arena(evaluator, root, extracted.chicago_arena)
    await verify_festival(evaluator, root, extracted.california_festival, "venue_3_california_festival", "California", threshold=100000)
    await verify_festival(evaluator, root, extracted.illinois_festival, "venue_4_illinois_festival", "Illinois", threshold=100000)

    # Optional: record simple parsed capacities for transparency/debugging
    try:
        debug_info = {
            "broadway_capacity_num": _parse_capacity_to_int(extracted.broadway_theater.capacity) if extracted.broadway_theater else None,
            "chicago_arena_capacity_num": _parse_capacity_to_int(extracted.chicago_arena.capacity) if extracted.chicago_arena else None,
            "california_festival_capacity_num": _parse_capacity_to_int(extracted.california_festival.capacity) if extracted.california_festival else None,
            "illinois_festival_capacity_num": _parse_capacity_to_int(extracted.illinois_festival.capacity) if extracted.illinois_festival else None,
        }
        evaluator.add_custom_info(debug_info, info_type="parsed_capacities")
    except Exception:
        pass

    return evaluator.get_summary()