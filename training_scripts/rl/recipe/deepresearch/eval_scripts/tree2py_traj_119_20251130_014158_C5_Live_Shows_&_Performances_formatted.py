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
TASK_ID = "venues_national_tour_4_cities"
TASK_DESCRIPTION = (
    "A Broadway touring production company is planning a national tour and needs to identify suitable theater venues in four major U.S. cities. "
    "For each of the following cities, identify a dedicated performing arts theater or concert hall venue that meets these requirements: "
    "(1) the venue must have a seating capacity between 1,000 and 3,000 seats, "
    "(2) the venue must be a legitimate performing arts theater or concert hall (not a sports arena or multi-purpose stadium), and "
    "(3) you must provide a reference URL documenting the venue's seating capacity. "
    "The four cities are: Washington, D.C., New York City, Minneapolis, and Los Angeles. "
    "For each city, provide the venue name, its seating capacity, confirmation that it is a dedicated performing arts venue, and a valid reference URL."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    venue_name: Optional[str] = None
    location_text: Optional[str] = None
    capacity_text: Optional[str] = None
    capacity_number: Optional[int] = None
    venue_type_text: Optional[str] = None
    capacity_urls: List[str] = Field(default_factory=list)


class CityVenuesExtraction(BaseModel):
    washington_dc: Optional[VenueItem] = None
    new_york_city: Optional[VenueItem] = None
    minneapolis: Optional[VenueItem] = None
    los_angeles: Optional[VenueItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt builder                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_city_venues() -> str:
    return """
Extract, from the answer, one venue per required city (select the first relevant venue if multiple are listed). For each of the following cities, extract the following fields if they explicitly appear in the answer:
- venue_name: The venue’s proper name.
- location_text: How the answer phrases the location/city for that venue (e.g., "Washington, DC", "New York, NY", "Minneapolis, MN", "Los Angeles, CA").
- capacity_text: The seating capacity text/value stated in the answer (e.g., "2,400 seats" or "1,800").
- capacity_number: The numeric seating capacity if a single clear integer is given in the answer (strip commas; if a range or multiple capacities are given, or it isn't a single clear integer, return null).
- venue_type_text: The type/classification of the venue as stated in the answer (e.g., "theater", "concert hall", "opera house", "performing arts center").
- capacity_urls: An array of URL(s) in the answer that document seating capacity (e.g., official venue site, Wikipedia, credible venue directory). Only include valid URLs explicitly present in the answer.

Cities to extract (keys must match exactly):
- washington_dc
- new_york_city
- minneapolis
- los_angeles

Rules:
- Do not invent or infer any data not present in the answer text.
- If a field is missing, set it to null (or an empty array for capacity_urls).
- For each city, return only one venue (the first listed in the answer for that city).
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _get_sources_from_item(item: Optional[VenueItem]) -> List[str]:
    if item and item.capacity_urls:
        # Deduplicate while preserving order
        seen = set()
        urls = []
        for u in item.capacity_urls:
            if isinstance(u, str) and u and (u not in seen):
                seen.add(u)
                urls.append(u)
        return urls
    return []


def _normalize_city_display(city_idx: int) -> str:
    if city_idx == 1:
        return "Washington, D.C."
    if city_idx == 2:
        return "New York City"
    if city_idx == 3:
        return "Minneapolis"
    if city_idx == 4:
        return "Los Angeles"
    return "Unknown City"


# --------------------------------------------------------------------------- #
# Verification builder per city                                               #
# --------------------------------------------------------------------------- #
async def verify_city(
    evaluator: Evaluator,
    parent_node,
    city_idx: int,
    city_desc: str,
    item: Optional[VenueItem],
) -> None:
    """
    Build verification subtree for a single city. All five checks in the rubric are created as leaves.
    Critical children ensure meaningful gating: if venue name is missing, other checks will be auto-skipped.
    """
    city_node = evaluator.add_parallel(
        id=f"city_{city_idx}_venue",
        desc=city_desc,
        parent=parent_node,
        critical=False,
    )

    # 1) Provide the venue name (existence check)
    name_exists = bool(item and item.venue_name and item.venue_name.strip())
    evaluator.add_custom_node(
        result=name_exists,
        id=f"city_{city_idx}_venue_name",
        desc="Provide the venue name",
        parent=city_node,
        critical=True,
    )

    # Prepare shared data for subsequent verifications
    venue_name = (item.venue_name.strip() if item and item.venue_name else "").strip()
    sources = _get_sources_from_item(item)
    city_display = _normalize_city_display(city_idx)

    # 2) Venue is located in the specified city
    location_node = evaluator.add_leaf(
        id=f"city_{city_idx}_location",
        desc=f"Venue is located in {city_display}",
        parent=city_node,
        critical=True,
    )
    location_claim = f"The venue '{venue_name}' is located in {city_display}."
    await evaluator.verify(
        claim=location_claim,
        node=location_node,
        sources=sources if sources else None,
        additional_instruction=(
            f"Verify on the provided webpage(s) that the venue is in {city_display}. "
            "Allow reasonable variants (e.g., 'Washington, DC' vs 'Washington, D.C.'; "
            "'New York, NY' vs 'New York City'; 'Los Angeles, CA' vs 'Los Angeles')."
        ),
    )

    # 3) Provide the venue seating capacity and it is between 1,000 and 3,000 seats (inclusive)
    capacity_node = evaluator.add_leaf(
        id=f"city_{city_idx}_capacity",
        desc="Provide the venue seating capacity and it is between 1,000 and 3,000 seats (inclusive)",
        parent=city_node,
        critical=True,
    )
    # Build a claim that checks both: a capacity is stated on the page, and it falls within range
    if item and item.capacity_text:
        capacity_claim = (
            f"The webpage(s) document that '{venue_name}' has a seating capacity described as '{item.capacity_text}', "
            "and the documented seating capacity is between 1,000 and 3,000 seats inclusive."
        )
    else:
        capacity_claim = (
            f"The webpage(s) document the seating capacity for '{venue_name}', and it is between 1,000 and 3,000 seats inclusive."
        )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_node,
        sources=sources if sources else None,
        additional_instruction=(
            "Judge based on the page(s): you should find an explicit number of 'seats' or 'seating capacity'. "
            "If multiple capacities are noted (e.g., different halls or configurations), focus on the main seated capacity of the primary performance space. "
            "If a range is given, ensure it falls entirely within 1,000–3,000. "
            "If no capacity is documented on the page(s), mark as not supported."
        ),
    )

    # 4) Venue is a dedicated performing arts theater or concert hall (not sports arena/multi-purpose stadium)
    type_node = evaluator.add_leaf(
        id=f"city_{city_idx}_venue_type",
        desc="Venue is a dedicated performing arts theater or concert hall (not a sports arena or multi-purpose stadium)",
        parent=city_node,
        critical=True,
    )
    venue_type_claim = (
        f"'{venue_name}' is a dedicated performing arts theater or concert hall (not a sports arena or multi-purpose stadium)."
    )
    await evaluator.verify(
        claim=venue_type_claim,
        node=type_node,
        sources=sources if sources else None,
        additional_instruction=(
            "Use the page(s) to confirm the venue is a legitimate performing arts venue (theater, concert hall, opera house, performing arts center). "
            "If the venue is primarily a sports arena, stadium, or multi-purpose sports facility, the claim is not supported."
        ),
    )

    # 5) Provide a valid URL that documents the venue's seating capacity
    reference_node = evaluator.add_leaf(
        id=f"city_{city_idx}_reference",
        desc="Provide a valid URL that documents the venue's seating capacity",
        parent=city_node,
        critical=True,
    )
    # Form a claim that will only be supported if at least one provided URL actually documents capacity
    if sources:
        ref_claim = (
            f"At least one of the provided URLs documents the seating capacity of '{venue_name}'."
        )
    else:
        ref_claim = (
            f"The answer provides at least one valid URL that documents the seating capacity of '{venue_name}'."
        )
    await evaluator.verify(
        claim=ref_claim,
        node=reference_node,
        sources=sources if sources else None,
        additional_instruction=(
            "A valid URL must be explicitly present in the answer and the webpage must clearly show a seating capacity number for the venue. "
            "If no URL is provided in the answer, or the page does not mention seating capacity, mark as not supported."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 4-city venues task and return a structured result dictionary.
    """
    # Initialize evaluator with a parallel root as the rubric checks per city independently
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

    # Extract structured venue info from the answer
    venues_data = await evaluator.extract(
        prompt=prompt_extract_city_venues(),
        template_class=CityVenuesExtraction,
        extraction_name="city_venues_extraction",
    )

    # Build verification subtrees for each city
    city_plan = [
        (1, "Washington, D.C. venue", venues_data.washington_dc),
        (2, "New York City venue", venues_data.new_york_city),
        (3, "Minneapolis venue", venues_data.minneapolis),
        (4, "Los Angeles venue", venues_data.los_angeles),
    ]

    for idx, desc, item in city_plan:
        await verify_city(
            evaluator=evaluator,
            parent_node=root,
            city_idx=idx,
            city_desc=desc,
            item=item,
        )

    # Return structured summary including the verification tree
    return evaluator.get_summary()