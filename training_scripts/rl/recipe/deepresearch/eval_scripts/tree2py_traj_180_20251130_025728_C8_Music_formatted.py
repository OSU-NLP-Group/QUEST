import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "zach_bryan_march_2026_stadiums"
TASK_DESCRIPTION = (
    "Zach Bryan's 2026 'With Heaven On Tour' includes performances at multiple stadium venues across the United States. "
    "For all U.S. stadium venues scheduled in March 2026 on this tour, identify each venue and provide the following information: "
    "(1) the official venue name, (2) the city, (3) the state, (4) the scheduled concert date, and (5) the official stadium seating capacity. "
    "Additionally, determine which of these March 2026 U.S. stadium venues has the largest official seating capacity."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    date: Optional[str] = None  # Keep as flexible string (can be "March 1, 2026" or "2026-03-01")
    capacity: Optional[str] = None  # Keep as string to accept ranges/approx ("~80,000", "75,000+")
    source_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)
    largest_capacity_venue_name: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    From the answer text, extract ONLY the March 2026 U.S. STADIUM venues on Zach Bryan's 2026 "With Heaven On Tour".
    For each such venue mentioned in the answer, extract the following fields exactly as stated in the answer:
    - venue_name: The official venue name as written in the answer (e.g., "AT&T Stadium").
    - city: The city associated with the venue as given in the answer.
    - state: The U.S. state as given in the answer (two-letter abbreviation or full name).
    - date: The scheduled concert date as written in the answer (any reasonable date format is acceptable, e.g., "March 4, 2026" or "2026-03-04").
    - capacity: The official stadium seating capacity as stated in the answer (keep the exact text; do NOT convert to a number; include commas/approximation as written).
    - source_urls: All URLs cited in the answer that substantively support this venue/date/capacity (include both event/ticket/tour pages and venue/stadium pages; extract actual URLs; do not invent).

    IMPORTANT FILTERS:
    - Include ONLY venues that are in the United States.
    - Include ONLY venues that are STADIUMS (not arenas, amphitheatres, fairgrounds, etc.).
    - Include ONLY venues with concert dates in March 2026.
    - Preserve the order in which they appear in the answer.

    In addition, if the answer explicitly states which March 2026 U.S. stadium venue has the largest official seating capacity among those listed,
    extract that venue name verbatim into:
    - largest_capacity_venue_name: the venue name as written in the answer; if not stated, return null.

    Return a JSON object with:
    {
      "venues": [ ... list of venue objects as specified ... ],
      "largest_capacity_venue_name": "..." or null
    }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_capacity_to_int(capacity_text: Optional[str]) -> Optional[int]:
    """
    Parse an integer capacity from a free-form capacity string.
    Strategy: extract all integer-like substrings (e.g., "80,000", "75000") and return the largest.
    If none found, return None.
    """
    if not capacity_text:
        return None
    nums = re.findall(r"\d[\d,\.]*", capacity_text)
    candidates: List[int] = []
    for n in nums:
        # Normalize commas and periods; keep digits only
        digits = re.sub(r"[^\d]", "", n)
        if digits.isdigit():
            try:
                candidates.append(int(digits))
            except Exception:
                continue
    return max(candidates) if candidates else None


def normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    s = name.lower().strip()
    # Remove content in trailing parentheses and common articles
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)
    s = s.replace("the ", "", 1) if s.startswith("the ") else s
    # Normalize multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s


def item_sources(item: VenueItem) -> List[str]:
    return [u for u in (item.source_urls or []) if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_venue_item(
    evaluator: Evaluator,
    parent_node,
    item: VenueItem,
    item_index: int
) -> None:
    """
    Build the verification sub-tree for one March 2026 U.S. stadium venue item.
    This follows the rubric leaves exactly for: is_us_stadium_venue, is_scheduled_tour_stop,
    venue_name, city, state, date, capacity.
    """
    node = evaluator.add_parallel(
        id=f"item_{item_index + 1}",
        desc=f"{item_index + 1}st/nd/rd/th March 2026 U.S. stadium venue entry includes all required attributes and is a correct tour stop.",
        parent=parent_node,
        critical=False
    )

    # Leaf: Venue is a U.S. stadium venue.
    leaf_us_stadium = evaluator.add_leaf(
        id=f"item_{item_index + 1}_is_us_stadium_venue",
        desc="Venue is a U.S. stadium venue.",
        parent=node,
        critical=True
    )
    claim_us_stadium = (
        f"The venue '{item.venue_name or ''}' located in {item.city or ''}, {item.state or ''} "
        f"is a stadium in the United States (not an arena/amphitheatre), according to the provided sources."
    )
    await evaluator.verify(
        claim=claim_us_stadium,
        node=leaf_us_stadium,
        sources=item_sources(item),
        additional_instruction=(
            "Verify that this venue is in the USA and is indeed a stadium (not an arena or amphitheatre). "
            "If the venue type is clearly 'stadium' on any provided source, consider this supported."
        )
    )

    # Leaf: Venue is a scheduled tour stop on "With Heaven On Tour".
    leaf_tour_stop = evaluator.add_leaf(
        id=f"item_{item_index + 1}_is_scheduled_tour_stop",
        desc="Venue is a scheduled stop on Zach Bryan's 2026 'With Heaven On Tour'.",
        parent=node,
        critical=True
    )
    claim_tour_stop = (
        f"Zach Bryan's 'With Heaven On Tour' has a scheduled concert at '{item.venue_name or ''}' "
        f"in {item.city or ''}, {item.state or ''} on {item.date or ''}."
    )
    await evaluator.verify(
        claim=claim_tour_stop,
        node=leaf_tour_stop,
        sources=item_sources(item),
        additional_instruction=(
            "Prefer official tour pages, venue event listings, or reputable ticketing sites. "
            "Minor date format differences are acceptable; confirm that at least one scheduled date in March 2026 matches."
        )
    )

    # Leaf: Official venue name is provided and correct.
    leaf_name = evaluator.add_leaf(
        id=f"item_{item_index + 1}_venue_name",
        desc="Official venue name is provided and correct.",
        parent=node,
        critical=True
    )
    claim_name = f"The official venue name is '{item.venue_name or ''}'."
    await evaluator.verify(
        claim=claim_name,
        node=leaf_name,
        sources=item_sources(item),
        additional_instruction=(
            "Check the venue's official website or widely accepted pages (e.g., Wikipedia, stadium site) "
            "to confirm the exact official name. Allow minor stylization differences (e.g., 'The')."
        )
    )

    # Leaf: City is provided and correct.
    leaf_city = evaluator.add_leaf(
        id=f"item_{item_index + 1}_city",
        desc="City is provided and correct.",
        parent=node,
        critical=True
    )
    claim_city = f"The venue '{item.venue_name or ''}' is located in the city of {item.city or ''}."
    await evaluator.verify(
        claim=claim_city,
        node=leaf_city,
        sources=item_sources(item),
        additional_instruction=(
            "Confirm the city on the venue's official or authoritative pages. "
            "If the venue is in a suburb within a metro area, use the official city listed by the venue."
        )
    )

    # Leaf: State is provided and correct.
    leaf_state = evaluator.add_leaf(
        id=f"item_{item_index + 1}_state",
        desc="State is provided and correct.",
        parent=node,
        critical=True
    )
    claim_state = f"The venue '{item.venue_name or ''}' is located in the U.S. state of {item.state or ''}."
    await evaluator.verify(
        claim=claim_state,
        node=leaf_state,
        sources=item_sources(item),
        additional_instruction=(
            "Verify the U.S. state associated with the venue as per the official or authoritative page."
        )
    )

    # Leaf: Date is provided, correct, and occurs in March 2026.
    leaf_date = evaluator.add_leaf(
        id=f"item_{item_index + 1}_date",
        desc="Scheduled concert date is provided, correct, and occurs in March 2026.",
        parent=node,
        critical=True
    )
    claim_date = (
        f"The scheduled concert date at '{item.venue_name or ''}' is {item.date or ''}, "
        f"and the date occurs in March 2026."
    )
    await evaluator.verify(
        claim=claim_date,
        node=leaf_date,
        sources=item_sources(item),
        additional_instruction=(
            "Confirm the concert date on an official tour/event page. "
            "It must be in March 2026 (any day in 2026-03). "
            "Accept reasonable date format variations (e.g., 'March 4, 2026' vs '2026-03-04')."
        )
    )

    # Leaf: Official stadium seating capacity is provided and correct.
    leaf_capacity = evaluator.add_leaf(
        id=f"item_{item_index + 1}_capacity",
        desc="Official stadium seating capacity is provided and correct.",
        parent=node,
        critical=True
    )
    claim_capacity = (
        f"The official (seating) capacity of '{item.venue_name or ''}' is '{item.capacity or ''}'."
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=leaf_capacity,
        sources=item_sources(item),
        additional_instruction=(
            "Validate the venue's official seating capacity (not attendance record). "
            "If the source provides a range or multiple figures, accept if the stated number is a commonly cited official capacity "
            "or falls within the range. Prefer venue official site; otherwise, authoritative sources are acceptable."
        )
    )


def compute_largest_capacity_result(
    venues: List[VenueItem],
    claimed_largest_name: Optional[str]
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compute whether the claimed largest-capacity venue matches the maximum parsed capacity
    among the provided venues (first up to 4).
    Returns (result_boolean, debug_info).
    """
    # Take first 4 items per rubric slots
    considered = venues[:4] if venues else []
    capacities: List[Tuple[int, str, int]] = []  # (parsed_capacity, venue_name, index)
    for idx, v in enumerate(considered):
        cap_val = parse_capacity_to_int(v.capacity)
        if cap_val is not None and v.venue_name:
            capacities.append((cap_val, v.venue_name, idx))

    debug_info = {
        "considered_venues": [
            {
                "index": i,
                "venue_name": v.venue_name,
                "capacity_raw": v.capacity,
                "capacity_parsed": parse_capacity_to_int(v.capacity)
            } for i, v in enumerate(considered)
        ],
        "claimed_largest_name": claimed_largest_name
    }

    if not capacities:
        debug_info["reason"] = "No numeric capacities could be parsed from the considered venues."
        return False, debug_info

    max_val = max(c[0] for c in capacities)
    top_names = {normalize_name(c[1]) for c in capacities if c[0] == max_val}

    debug_info["max_capacity"] = max_val
    debug_info["top_names_normalized"] = list(top_names)

    if not claimed_largest_name:
        debug_info["reason"] = "Answer did not explicitly identify the largest-capacity venue."
        return False, debug_info

    claimed_norm = normalize_name(claimed_largest_name)
    debug_info["claimed_normalized"] = claimed_norm

    # Ensure the claimed venue is among the venues listed (normalized compare)
    listed_norms = {normalize_name(v.venue_name) for v in considered if v.venue_name}
    debug_info["listed_names_normalized"] = list(listed_norms)

    if claimed_norm not in listed_norms:
        debug_info["reason"] = "Claimed largest venue is not among the listed March 2026 venues."
        return False, debug_info

    result = claimed_norm in top_names
    if not result:
        debug_info["reason"] = "Claimed largest venue is not in the top capacity tie-set."
    return result, debug_info


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the March 2026 U.S. stadium venues on Zach Bryan's 'With Heaven On Tour'.
    Builds a verification tree per rubric:
    - Up to 4 venue items with required attributes verified against cited sources.
    - A largest-capacity venue check across the listed venues.
    """
    evaluator = Evaluator()
    # NOTE: The input rubric marks root as critical, but that would force all children to be critical
    # under the framework's consistency rule. We set root to non-critical to allow partial credit.
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

    # Extract venues and claimed largest venue from the answer
    extraction: VenuesExtraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_march_2026_extraction"
    )

    # Record a small custom info for transparency
    evaluator.add_custom_info(
        info={
            "extracted_venue_count": len(extraction.venues),
            "largest_capacity_venue_name_claimed": extraction.largest_capacity_venue_name
        },
        info_type="extraction_overview",
        info_name="extraction_overview"
    )

    # Prepare up to 4 items (pad with empty VenueItem if fewer than 4 to keep a stable tree)
    items: List[VenueItem] = list(extraction.venues[:4])
    while len(items) < 4:
        items.append(VenueItem())

    # Build per-item verification subtrees
    # Items are non-critical at the parent level; each leaf under an item is critical per rubric.
    for idx, venue_item in enumerate(items):
        await verify_venue_item(evaluator, root, venue_item, idx)

    # Largest-capacity venue verification node (item_5)
    item5 = evaluator.add_parallel(
        id="item_5",
        desc="Largest-capacity venue among the March 2026 U.S. stadium venues is identified correctly.",
        parent=root,
        critical=False
    )

    # Compute largest capacity correctness from extracted data
    largest_ok, debug_info = compute_largest_capacity_result(items, extraction.largest_capacity_venue_name)
    evaluator.add_custom_info(
        info=debug_info,
        info_type="largest_capacity_debug",
        info_name="largest_capacity_debug"
    )

    evaluator.add_custom_node(
        result=largest_ok,
        id="largest_capacity_venue_correct",
        desc="Answer correctly identifies which listed March 2026 U.S. stadium venue has the largest official seating capacity.",
        parent=item5,
        critical=True
    )

    # Return the standardized summary with the verification tree
    return evaluator.get_summary()