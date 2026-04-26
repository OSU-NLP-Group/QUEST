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
TASK_ID = "ca_amphitheaters_by_capacity"
TASK_DESCRIPTION = (
    "I am planning a concert tour across California and need to identify outdoor amphitheaters in four different "
    "capacity ranges to accommodate various audience sizes. For each of the following capacity ranges, identify one "
    "outdoor amphitheater located in California and provide its official name, the city where it is located, its exact "
    "seating capacity, and a reference URL verifying this information:\n\n"
    "1. Small venue: seating capacity between 5,000 and 6,500\n"
    "2. Medium-small venue: seating capacity between 8,000 and 9,000\n"
    "3. Medium-large venue: seating capacity between 17,000 and 18,000\n"
    "4. Large venue: seating capacity of 20,000 or more\n\n"
    "All venues must be outdoor amphitheaters (not indoor arenas, stadiums, or other venue types)."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    """
    One venue item extracted for a capacity category.
    """
    official_name: Optional[str] = None
    city: Optional[str] = None
    capacity_text: Optional[str] = None
    capacity_number: Optional[int] = None
    reference_urls: List[str] = Field(default_factory=list)


class TourVenuesExtraction(BaseModel):
    """
    Extraction result encompassing all four capacity categories.
    """
    small: Optional[VenueItem] = None           # 5,000–6,500
    medium_small: Optional[VenueItem] = None    # 8,000–9,000
    medium_large: Optional[VenueItem] = None    # 17,000–18,000
    large: Optional[VenueItem] = None           # >= 20,000


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract exactly one California outdoor amphitheater from the answer for each capacity category below. For each category, extract:
- official_name: the official venue name as given in the answer
- city: the California city where the venue is located (city name only, no state)
- capacity_text: the exact seating-capacity phrase as written in the answer (e.g., "17,500" or "approximately 8,000")
- capacity_number: the exact seating capacity as an integer if explicitly provided (e.g., 17500 for "17,500"). If the answer only gives an approximate capacity, extract the closest integer shown. If no number is given, set to null.
- reference_urls: all URL(s) mentioned in the answer that are intended to verify this venue and its capacity. Extract only valid URLs explicitly present in the answer.

Map them to:
- small:      Small venue (capacity 5,000–6,500)
- medium_small: Medium-small venue (capacity 8,000–9,000)
- medium_large: Medium-large venue (capacity 17,000–18,000)
- large:      Large venue (capacity 20,000+)

Important:
- Only extract URLs that are explicitly present in the answer (plain or markdown). Do not invent URLs.
- If multiple venues are mentioned for a category, choose the first one in the answer.
- If a category is missing, set that category to null.
- If a field is missing for a chosen venue, set that field to null or [] appropriately.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_capacity_from_text(text: Optional[str]) -> Optional[int]:
    """
    Attempt to parse an integer seating capacity from a capacity_text string.
    Handles formats like: "17,500", "17500", "20,000+", "about 8,000", "17k", "17.5k".
    Returns None if parsing is not possible.
    """
    if not text:
        return None

    s = text.lower().strip()

    # Handle patterns like "17.5k" or "17k"
    m_k = re.search(r'(\d+(?:\.\d+)?)\s*k\b', s)
    if m_k:
        try:
            val = float(m_k.group(1))
            return int(round(val * 1000))
        except Exception:
            pass

    # General integer with commas and optional plus sign
    m_int = re.search(r'(\d{1,3}(?:,\d{3})+|\d+)\s*\+?', s)
    if m_int:
        try:
            digits = m_int.group(1).replace(',', '')
            return int(digits)
        except Exception:
            pass

    return None


def effective_capacity(item: Optional[VenueItem]) -> Optional[int]:
    """
    Decide the effective numeric capacity to use for checks:
    1) Prefer capacity_number from extraction (if not None)
    2) Otherwise parse from capacity_text (if possible)
    """
    if not item:
        return None
    if item.capacity_number is not None:
        try:
            return int(item.capacity_number)
        except Exception:
            pass
    return parse_capacity_from_text(item.capacity_text)


def urls_or_empty(item: Optional[VenueItem]) -> List[str]:
    """
    Return the item's reference URLs list, or [] if None.
    """
    return item.reference_urls if (item and item.reference_urls) else []


def capacity_in_range(cap: Optional[int], min_inclusive: Optional[int], max_inclusive: Optional[int]) -> bool:
    if cap is None:
        return False
    if min_inclusive is not None and cap < min_inclusive:
        return False
    if max_inclusive is not None and cap > max_inclusive:
        return False
    return True


# --------------------------------------------------------------------------- #
# Venue verification logic                                                    #
# --------------------------------------------------------------------------- #
async def verify_venue_category(
    evaluator: Evaluator,
    parent_node,
    category_id: str,
    category_desc: str,
    item: Optional[VenueItem],
    range_bounds: Tuple[Optional[int], Optional[int]],  # (min_inclusive, max_inclusive); use (20000, None) for 20k+
) -> None:
    """
    Build verification subtree for one capacity category and run checks.

    Leaves created (critical unless noted):
    - official_name_provided (custom, critical)
    - city_provided (custom, critical)
    - exact_capacity_provided (custom, critical)
    - capacity_in_range (custom, critical)
    - reference_url_provided (custom, critical)  [extra existence gate]
    - is_outdoor_amphitheater (leaf verify against URL, critical)
    - located_in_california (leaf verify against URL, critical)
    - reference_url_verifies_info (leaf verify by URLs, critical)
    """
    cat_node = evaluator.add_parallel(
        id=category_id,
        desc=category_desc,
        parent=parent_node,
        critical=False
    )

    name_ok = bool(item and item.official_name and item.official_name.strip())
    city_ok = bool(item and item.city and item.city.strip())
    cap_val = effective_capacity(item)
    cap_ok = cap_val is not None
    url_list = urls_or_empty(item)
    url_ok = len(url_list) > 0

    # 1) Official name provided
    evaluator.add_custom_node(
        result=name_ok,
        id=f"{category_id}_official_name_provided",
        desc="The response provides the venue’s official name.",
        parent=cat_node,
        critical=True
    )

    # 2) City provided
    evaluator.add_custom_node(
        result=city_ok,
        id=f"{category_id}_city_provided",
        desc="The response provides the city where the venue is located (in California).",
        parent=cat_node,
        critical=True
    )

    # 3) Exact capacity provided (numeric)
    evaluator.add_custom_node(
        result=cap_ok,
        id=f"{category_id}_exact_capacity_provided",
        desc="The response provides an exact seating capacity as a numeric value.",
        parent=cat_node,
        critical=True
    )

    # 4) Capacity in the required range (numeric gate)
    min_inc, max_inc = range_bounds
    range_text = ""
    if min_inc is not None and max_inc is not None:
        range_text = f"between {min_inc:,} and {max_inc:,} (inclusive)"
    elif min_inc is not None and max_inc is None:
        range_text = f"{min_inc:,} or more"
    elif min_inc is None and max_inc is not None:
        range_text = f"{max_inc:,} or less"
    else:
        range_text = "in the specified range"

    evaluator.add_custom_node(
        result=capacity_in_range(cap_val, min_inc, max_inc),
        id=f"{category_id}_capacity_in_range",
        desc=f"The venue's seating capacity is {range_text}.",
        parent=cat_node,
        critical=True
    )

    # 5) Reference URL existence gate (extra precondition to avoid meaningless URL checks)
    evaluator.add_custom_node(
        result=url_ok,
        id=f"{category_id}_reference_url_provided",
        desc="At least one reference URL is provided.",
        parent=cat_node,
        critical=True
    )

    # Prepare strings for claims
    venue_name = item.official_name if item and item.official_name else "the venue"
    city_name = item.city if item and item.city else "a city in"
    cap_display = f"{cap_val:,}" if cap_val is not None else (item.capacity_text if item else "unknown")

    # 6) Outdoor amphitheater check (verify by URL)
    leaf_outdoor = evaluator.add_leaf(
        id=f"{category_id}_is_outdoor_amphitheater",
        desc="The venue is an outdoor amphitheater (not an indoor arena, stadium, or other venue type).",
        parent=cat_node,
        critical=True
    )
    claim_outdoor = (
        f"The venue named '{venue_name}' is an outdoor amphitheater (open-air), "
        f"not an indoor arena, stadium, or other venue type."
    )
    await evaluator.verify(
        claim=claim_outdoor,
        node=leaf_outdoor,
        sources=url_list,
        additional_instruction=(
            "Accept reasonable wording variants like 'amphitheatre' (British spelling) or 'open-air amphitheater'. "
            "If the page clearly indicates the venue is an outdoor amphitheater, mark as supported. "
            "If it is primarily an indoor arena or stadium, mark as not supported."
        )
    )

    # 7) Located in California (verify by URL)
    leaf_location = evaluator.add_leaf(
        id=f"{category_id}_located_in_california",
        desc="The venue is located in California, United States.",
        parent=cat_node,
        critical=True
    )
    # If city name exists, include it in claim to be specific; otherwise general California check.
    if city_ok:
        claim_location = f"The venue named '{venue_name}' is located in {city_name}, California, United States."
        add_ins_loc = "If the page shows 'CA' for the state, treat it as 'California'. Minor formatting differences are acceptable."
    else:
        claim_location = f"The venue named '{venue_name}' is located in California, United States."
        add_ins_loc = "If the page shows 'CA' for the state, treat it as 'California'."
    await evaluator.verify(
        claim=claim_location,
        node=leaf_location,
        sources=url_list,
        additional_instruction=add_ins_loc
    )

    # 8) Reference URL verifies identity and capacity (verify by URL(s))
    leaf_ref = evaluator.add_leaf(
        id=f"{category_id}_reference_url_verifies_info",
        desc="The response provides at least one reference URL that corroborates the venue identification and the stated seating capacity (and supports that it is an outdoor amphitheater in California).",
        parent=cat_node,
        critical=True
    )
    claim_ref = (
        f"At least one of the provided reference pages explicitly confirms that the venue named '{venue_name}' "
        f"has a seating capacity of {cap_display}."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=leaf_ref,
        sources=url_list,
        additional_instruction=(
            "Look for explicit statements of seating capacity on the page. Consider minor variants such as 'about' or "
            "'approximately' (e.g., 'about 8,000') as acceptable if they clearly indicate the same figure. "
            "Also ensure the page is about this venue (name match allowing minor formatting variations)."
        )
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
    Evaluate an answer for the California amphitheaters by capacity task.
    Returns an evaluation summary dict with the verification tree and final score.
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
        default_model=model
    )

    # Extract venues info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=TourVenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Build subtrees for each category
    # Small: 5,000–6,500
    await verify_venue_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="small_capacity_venue",
        category_desc="Small venue (5,000–6,500): provide one qualifying outdoor amphitheater in California with required details.",
        item=extraction.small,
        range_bounds=(5000, 6500)
    )

    # Medium-small: 8,000–9,000
    await verify_venue_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="medium_small_capacity_venue",
        category_desc="Medium-small venue (8,000–9,000): provide one qualifying outdoor amphitheater in California with required details.",
        item=extraction.medium_small,
        range_bounds=(8000, 9000)
    )

    # Medium-large: 17,000–18,000
    await verify_venue_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="medium_large_capacity_venue",
        category_desc="Medium-large venue (17,000–18,000): provide one qualifying outdoor amphitheater in California with required details.",
        item=extraction.medium_large,
        range_bounds=(17000, 18000)
    )

    # Large: 20,000+
    await verify_venue_category(
        evaluator=evaluator,
        parent_node=root,
        category_id="large_capacity_venue",
        category_desc="Large venue (20,000+): provide one qualifying outdoor amphitheater in California with required details.",
        item=extraction.large,
        range_bounds=(20000, None)
    )

    # Return result
    return evaluator.get_summary()