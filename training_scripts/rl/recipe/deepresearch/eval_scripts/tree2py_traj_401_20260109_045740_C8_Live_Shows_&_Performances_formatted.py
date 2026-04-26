import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "lv_indoor_venues_4k_6k"
TASK_DESCRIPTION = (
    "Identify 3 indoor concert venues in Las Vegas, Nevada, that meet all of the following requirements: "
    "(1) Each venue must have a seated capacity between 4,000 and 6,000 people; "
    "(2) Each venue must be an indoor, climate-controlled performance space, not an outdoor amphitheater; "
    "(3) Each venue must be currently operational and actively hosting live performances in 2025-2026. "
    "For each venue, provide the venue name, its location in Las Vegas, the exact seated capacity, confirmation that "
    "it is indoor and climate-controlled, and evidence of current operational status with scheduled performances. "
    "Include reference URLs supporting all provided information."
)
YEAR_START = 2025
YEAR_END = 2026
CAPACITY_MIN = 4000
CAPACITY_MAX = 6000


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None  # Can be a street address or property/hotel name in Las Vegas
    capacity: Optional[str] = None  # Keep as string; we'll parse integer later
    identification_urls: List[str] = Field(default_factory=list)  # for name + location/address
    capacity_urls: List[str] = Field(default_factory=list)        # for seated capacity
    type_urls: List[str] = Field(default_factory=list)            # for indoor/climate-controlled confirmation
    operational_urls: List[str] = Field(default_factory=list)     # for schedules/residencies in 2025–2026


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract all the concert venues that the answer claims satisfy the task. For each venue, extract the following fields:

- name: The official venue name exactly as stated in the answer.
- location: The specific Las Vegas location or address (this can be a street address or the casino/resort property name, e.g., "Caesars Palace, Las Vegas, NV").
- capacity: The exact seated capacity value as stated in the answer (keep the string exactly as written, including commas or qualifiers).
- identification_urls: All URLs cited in the answer that support the venue's identity and its Las Vegas location/address.
- capacity_urls: All URLs cited in the answer that support the stated seated capacity.
- type_urls: All URLs cited in the answer that support the claim that the venue is indoor and climate-controlled (and not an outdoor amphitheater).
- operational_urls: All URLs cited in the answer that show the venue is operational and has scheduled live performances/residencies in 2025 or 2026 (e.g., an official calendar or listing page).

IMPORTANT:
- Extract only URLs explicitly present in the answer. Do not invent or infer URLs.
- If a single URL supports multiple claims, include it in each corresponding URL list.
- If any field is missing in the answer for a venue, set it to null (for strings) or an empty list (for the URL arrays).
- Extract ALL venues mentioned in the answer (not just three).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_name(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def parse_capacity_int(cap_str: Optional[str]) -> Optional[int]:
    if not cap_str:
        return None
    # extract first integer-like number
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)", cap_str)
    if not m:
        return None
    num = m.group(1).replace(",", "")
    try:
        return int(num)
    except Exception:
        return None


def unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def all_urls_for_venue(v: VenueItem) -> List[str]:
    return unique_urls(
        (v.identification_urls or []) +
        (v.capacity_urls or []) +
        (v.type_urls or []) +
        (v.operational_urls or [])
    )


# --------------------------------------------------------------------------- #
# Venue verification subtrees                                                 #
# --------------------------------------------------------------------------- #
async def verify_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    index_one_based: int,
) -> None:
    """
    Build verification sub-tree for a single venue.
    For each of the four rubric categories, we create a sequential group containing:
      - an existence/prerequisite check (custom node), then
      - an evidence-backed verification leaf.
    All leaves inside each category are critical to that category.
    The overall Venue_i group is non-critical at the root-level to allow partial credit across venues.
    """
    venue_node = evaluator.add_parallel(
        id=f"Venue_{index_one_based}",
        desc=f"Venue {index_one_based} requirements (allow partial credit across venues).",
        parent=parent_node,
        critical=False
    )

    # ---------------------- Identification ---------------------- #
    ident_group = evaluator.add_sequential(
        id=f"V{index_one_based}_Identification",
        desc="Provides the official venue name AND a specific Las Vegas, NV location/address, supported by at least one reference URL from a reliable source.",
        parent=venue_node,
        critical=True
    )
    ident_has_info = evaluator.add_custom_node(
        result=bool(venue and venue.name and venue.name.strip()) and bool(venue.location and venue.location.strip()) and (
            len(venue.identification_urls) > 0 or len(all_urls_for_venue(venue)) > 0
        ),
        id=f"V{index_one_based}_Identification_Provided",
        desc=f"V{index_one_based}: Name + specific Las Vegas location provided and at least one supporting URL listed.",
        parent=ident_group,
        critical=True
    )
    ident_supported = evaluator.add_leaf(
        id=f"V{index_one_based}_Identification_Supported",
        desc=f"V{index_one_based}: Identification supported by cited sources.",
        parent=ident_group,
        critical=True
    )
    ident_sources = venue.identification_urls if venue.identification_urls else all_urls_for_venue(venue)
    ident_claim = (
        f"The page shows that the venue named {venue.name} is located in the Las Vegas area (Las Vegas or Paradise, Nevada), "
        f"with location/address information consistent with: {venue.location}."
    )
    await evaluator.verify(
        claim=ident_claim,
        node=ident_supported,
        sources=ident_sources,
        additional_instruction=(
            "Verify that the venue name is present and the location is in Las Vegas, Nevada. "
            "Treat addresses within Paradise, NV (e.g., on the Las Vegas Strip) as valid for 'Las Vegas area'. "
            "Casino/resort property locations (e.g., Caesars Palace, Park MGM, Resorts World Las Vegas) should count as "
            "specific Las Vegas locations. Minor formatting differences in the location string are acceptable."
        )
    )

    # ---------------------- Capacity ---------------------- #
    capacity_group = evaluator.add_sequential(
        id=f"V{index_one_based}_Capacity",
        desc="States the exact seated capacity value and it is between 4,000 and 6,000 inclusive, supported by at least one reference URL from a reliable source.",
        parent=venue_node,
        critical=True
    )
    cap_int = parse_capacity_int(venue.capacity)
    cap_value_present = evaluator.add_custom_node(
        result=cap_int is not None,
        id=f"V{index_one_based}_Capacity_Value_Present",
        desc=f"V{index_one_based}: Capacity value is provided in the answer and parseable as an integer.",
        parent=capacity_group,
        critical=True
    )
    in_range = evaluator.add_custom_node(
        result=(cap_int is not None) and (CAPACITY_MIN <= cap_int <= CAPACITY_MAX),
        id=f"V{index_one_based}_Capacity_In_Range",
        desc=f"V{index_one_based}: Capacity integer is between {CAPACITY_MIN} and {CAPACITY_MAX} inclusive.",
        parent=capacity_group,
        critical=True
    )
    capacity_supported = evaluator.add_leaf(
        id=f"V{index_one_based}_Capacity_Supported",
        desc=f"V{index_one_based}: Capacity supported by cited sources.",
        parent=capacity_group,
        critical=True
    )
    cap_sources = venue.capacity_urls if venue.capacity_urls else all_urls_for_venue(venue)
    cap_claim = (
        f"The seated capacity of the venue {venue.name} is {cap_int}, which is between {CAPACITY_MIN} and {CAPACITY_MAX} inclusive."
        if cap_int is not None else
        f"The venue {venue.name} has a seated capacity between {CAPACITY_MIN} and {CAPACITY_MAX} inclusive."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=capacity_supported,
        sources=cap_sources,
        additional_instruction=(
            "Confirm the exact seated capacity from the cited page(s). "
            "If multiple capacities are shown, prefer the seated capacity for the venue when configured for concerts. "
            "Use numeric reasoning to determine if the stated capacity falls within the 4,000–6,000 range."
        )
    )

    # ---------------------- Venue Type (Indoor/Climate-Controlled) ---------------------- #
    type_group = evaluator.add_sequential(
        id=f"V{index_one_based}_Venue_Type",
        desc="Confirms the venue is indoor and climate-controlled AND not an outdoor amphitheater, supported by at least one reference URL from a reliable source.",
        parent=venue_node,
        critical=True
    )
    type_has_source = evaluator.add_custom_node(
        result=len(all_urls_for_venue(venue)) > 0,
        id=f"V{index_one_based}_Venue_Type_Source_Present",
        desc=f"V{index_one_based}: At least one source URL is provided to judge the venue type.",
        parent=type_group,
        critical=True
    )
    type_supported = evaluator.add_leaf(
        id=f"V{index_one_based}_Venue_Type_Supported",
        desc=f"V{index_one_based}: Indoor, climate-controlled (not outdoor amphitheater) supported by cited sources.",
        parent=type_group,
        critical=True
    )
    type_sources = venue.type_urls if venue.type_urls else all_urls_for_venue(venue)
    type_claim = (
        f"The venue {venue.name} is an indoor, climate-controlled concert venue (not an outdoor amphitheater)."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_supported,
        sources=type_sources,
        additional_instruction=(
            "Support the claim using the cited page(s). "
            "If the venue is located inside a casino/hotel property or is described as a 'theatre'/'theater', "
            "it should be considered indoor and climate-controlled. "
            "If the page clearly indicates 'outdoor', 'open-air', or 'amphitheater', the claim is not supported."
        )
    )

    # ---------------------- Operational & Scheduled in 2025–2026 ---------------------- #
    op_group = evaluator.add_sequential(
        id=f"V{index_one_based}_Operational_And_Scheduled_2025_2026",
        desc="Provides evidence the venue is operational and actively hosting scheduled live performances/concerts/residencies in 2025–2026 (e.g., an official calendar/listing with 2025/2026 dates), supported by at least one reference URL from a reliable source.",
        parent=venue_node,
        critical=True
    )
    op_has_source = evaluator.add_custom_node(
        result=len(all_urls_for_venue(venue)) > 0,
        id=f"V{index_one_based}_Operational_Source_Present",
        desc=f"V{index_one_based}: At least one source URL is provided to verify 2025/2026 schedules/operation.",
        parent=op_group,
        critical=True
    )
    op_supported = evaluator.add_leaf(
        id=f"V{index_one_based}_Operational_Supported",
        desc=f"V{index_one_based}: Operational with scheduled performances in {YEAR_START} or {YEAR_END} supported by cited sources.",
        parent=op_group,
        critical=True
    )
    op_sources = venue.operational_urls if venue.operational_urls else all_urls_for_venue(venue)
    op_claim = (
        f"The venue {venue.name} is operational and has scheduled live performances in {YEAR_START} or {YEAR_END}."
    )
    await evaluator.verify(
        claim=op_claim,
        node=op_supported,
        sources=op_sources,
        additional_instruction=(
            f"Look for an events calendar, show listings, or residency pages that explicitly display dates in {YEAR_START} or {YEAR_END}. "
            "Month names or numeric dates are acceptable as long as the year is 2025 or 2026. "
            "If the page lists upcoming events at this venue with 2025/2026 dates, the claim is supported."
        )
    )


# --------------------------------------------------------------------------- #
# Root-level checks                                                           #
# --------------------------------------------------------------------------- #
def evaluate_count_and_distinctness(evaluator: Evaluator, parent_node, extracted: VenuesExtraction) -> None:
    # Determine how many venues the answer actually provided (by name)
    provided_names = [v.name for v in extracted.venues if v.name and v.name.strip()]
    normalized = [normalize_name(n) for n in provided_names if n]
    exact_three = (len(provided_names) == 3)
    all_distinct = len(set(normalized)) == len(normalized) if normalized else False

    evaluator.add_custom_node(
        result=(exact_three and all_distinct),
        id="Venue_Count_And_Distinctness",
        desc="Response provides exactly 3 distinct venues (no duplicates).",
        parent=parent_node,
        critical=True  # Failing this should fail the entire task
    )


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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Las Vegas indoor venues (4k–6k capacity) task.
    """

    # Initialize evaluator - root non-critical to allow partial credit; use parallel aggregation
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

    # Extract all venues mentioned in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Add ground truth context (constraints)
    evaluator.add_ground_truth({
        "capacity_range_inclusive": [CAPACITY_MIN, CAPACITY_MAX],
        "required_years": [YEAR_START, YEAR_END],
        "must_be_indoor_climate_controlled": True,
        "must_not_be_outdoor_amphitheater": True,
        "required_city": "Las Vegas area (Las Vegas or Paradise, NV)"
    }, gt_type="constraints")

    # Root-level critical check: exactly 3 distinct venues
    evaluate_count_and_distinctness(evaluator, root, extracted)

    # Prepare exactly three venues for detailed checks (pad with empty items if fewer)
    venues_list: List[VenueItem] = list(extracted.venues or [])
    # If more than three provided, we only evaluate the first three (count/distinctness still checked above)
    selected = venues_list[:3]
    while len(selected) < 3:
        selected.append(VenueItem())

    # Build verification subtree for each selected venue (non-critical blocks to allow partial credit)
    for idx in range(3):
        await verify_venue(
            evaluator=evaluator,
            parent_node=root,
            venue=selected[idx],
            index_one_based=idx + 1
        )

    # Return the structured summary
    return evaluator.get_summary()