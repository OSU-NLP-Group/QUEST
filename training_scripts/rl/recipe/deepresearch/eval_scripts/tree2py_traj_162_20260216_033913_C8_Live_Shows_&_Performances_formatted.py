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
TASK_ID = "medium_scale_theater_tour_venues"
TASK_DESCRIPTION = (
    "I am planning a multi-city theater tour for a medium-scale Broadway production across major U.S. performing arts venues. "
    "I need to identify 4 suitable performance venues that meet the following criteria: "
    "(1) Located in one of these cities: New York City, Washington DC, Chicago, or Los Angeles; "
    "(2) Classified as a theater or performing arts center (not primarily a sports arena or stadium); "
    "(3) Seating capacity between 1,000 and 4,000 seats (medium-sized venue classification); "
    "(4) Must be ADA accessible with wheelchair seating provisions; "
    "(5) Currently operational as a performance venue. "
    "For each venue, provide: venue name, city location, exact seating capacity, primary venue type/classification, "
    "confirmation of ADA accessibility, and a reference URL that verifies this information."
)

ALLOWED_CITIES = [
    "New York City",
    "Washington DC",
    "Chicago",
    "Los Angeles",
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    seating_capacity: Optional[str] = None
    primary_type: Optional[str] = None
    ada_accessibility: Optional[str] = None
    reference_url: Optional[str] = None


class VenuesExtraction(BaseModel):
    venues: List[VenueInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return (
        "Extract up to four performance venues mentioned in the answer text that are intended for a medium-scale Broadway/theater tour. "
        "For each venue, return an object with the following fields:\n"
        "1) name: The venue name as stated in the answer (string).\n"
        "2) city: The city location as stated (string). Prefer 'New York City', 'Washington DC', 'Chicago', or 'Los Angeles' if applicable. "
        "   If the answer uses synonyms like 'NYC', 'Washington, D.C.', 'LA', normalize to the closest common form but still return exactly what appears if clear.\n"
        "3) seating_capacity: The exact seating capacity as provided in the answer (string). If a range or approximate value is given, return it as-is.\n"
        "4) primary_type: The venue type/classification as stated (e.g., 'theater', 'performing arts center', 'opera house', 'concert hall').\n"
        "5) ada_accessibility: The ADA accessibility confirmation text (e.g., 'ADA compliant', 'wheelchair accessible seating', or similar). "
        "   If no ADA remark is present, return null.\n"
        "6) reference_url: A single URL that is explicitly provided in the answer to verify the venue’s information. "
        "   Extract only URLs explicitly present in the answer (including plain URLs or markdown links). "
        "   Include the full protocol (http/https). If no URL is provided for a venue, return null.\n\n"
        "Important:\n"
        "- Do not invent or infer any information not explicitly present in the answer.\n"
        "- If the answer lists more than four venues, extract all and the evaluator will consider only the first four.\n"
        "- If fewer than four venues are present, return only those available.\n"
        "- Use strings for all fields; do not convert seating capacity into numbers.\n"
        "- If any field is missing for a venue, set it to null.\n"
        "Return a JSON object with a single key 'venues' mapping to an array of venue objects."
    )


# --------------------------------------------------------------------------- #
# Helper functions for verification                                           #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(n, f"#{n}")


def build_missing_source_instruction(reference_url: Optional[str]) -> str:
    if reference_url and reference_url.strip():
        return "Use the provided URL as the authoritative source to evaluate the claim."
    return (
        "No source URL was provided in the answer. According to the evaluation policy, claims without web evidence must be judged as not supported."
    )


def city_synonyms_instruction() -> str:
    return (
        "Treat common synonyms as equivalent: 'NYC' ≈ 'New York City'; 'Los Angeles' ≈ 'LA'; "
        "'Washington, D.C.' ≈ 'Washington DC'. Accept reasonable variants like including state abbreviations (e.g., 'Los Angeles, CA'). "
        "However, the location must clearly be in one of the allowed cities."
    )


def type_synonyms_instruction() -> str:
    return (
        "Classify as valid if the page clearly indicates the venue is primarily a theater or performing arts center, "
        "including synonyms such as 'theatre', 'opera house', 'playhouse', 'performing arts center', or 'concert hall'. "
        "If the venue is primarily a sports arena or stadium (e.g., hosting professional sports as its main purpose), mark as not valid."
    )


def accessibility_synonyms_instruction() -> str:
    return (
        "Consider 'ADA accessible', 'ADA compliant', 'accessibility', 'accessible seating', 'wheelchair seating', "
        "or similar phrasing as acceptable evidence of ADA accessibility with wheelchair seating provisions."
    )


def operational_instruction() -> str:
    return (
        "To determine current operational status as a performance venue, look for signs of ongoing operations such as upcoming events, "
        "active schedules, current season info, ticket links, or recent updates. If the venue is permanently closed or indefinitely under renovation, "
        "mark the claim as not supported."
    )


def capacity_range_instruction() -> str:
    return (
        "Verify from the webpage whether the venue’s seating capacity is between 1,000 and 4,000 seats. "
        "If the exact capacity is stated in the answer as a number or a string (e.g., '1,800' or 'about 2,000'), confirm the capacity and range. "
        "If the page lists multiple halls/rooms, consider the main auditorium if clearly indicated."
    )


# --------------------------------------------------------------------------- #
# Verification logic per venue                                                #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueInfo,
    index_1_based: int,
) -> None:
    """
    Build verification subtree and run checks for a single venue.
    """
    # Create a parallel node for this venue
    venue_node = evaluator.add_parallel(
        id=f"venue_{index_1_based}",
        desc=f"{ordinal(index_1_based)} qualifying performance venue meeting all specified criteria",
        parent=parent_node,
        critical=False,
    )

    # Prepare shared data
    url = venue.reference_url or None
    name_for_claim = venue.name or ""
    city_for_claim = venue.city or ""

    # ---- Reference URL check (critical) ----
    ref_leaf = evaluator.add_leaf(
        id=f"venue_{index_1_based}_reference",
        desc="A reference URL is provided that verifies the venue information",
        parent=venue_node,
        critical=True,
    )
    ref_claim = (
        f"This webpage corresponds to the venue named '{name_for_claim}'. "
        f"If the page obviously refers to the same venue by name or branding, treat as supported."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=url,
        additional_instruction=(
            build_missing_source_instruction(url)
            + " Allow reasonable name variants (e.g., casing differences, 'Theatre' vs 'Theater')."
        ),
    )

    # ---- Location check (critical) ----
    loc_leaf = evaluator.add_leaf(
        id=f"venue_{index_1_based}_location",
        desc="Venue is located in one of the specified cities: New York City, Washington DC, Chicago, or Los Angeles",
        parent=venue_node,
        critical=True,
    )
    if city_for_claim:
        loc_claim = (
            f"The venue is located in '{city_for_claim}', and this city is one of: New York City, Washington DC, Chicago, Los Angeles."
        )
    else:
        loc_claim = (
            "The venue is located in one of: New York City, Washington DC, Chicago, or Los Angeles."
        )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=url,
        additional_instruction=build_missing_source_instruction(url) + " " + city_synonyms_instruction(),
    )

    # ---- Type/classification check (critical) ----
    type_leaf = evaluator.add_leaf(
        id=f"venue_{index_1_based}_type",
        desc="Venue is classified as a theater or performing arts center, not primarily a sports arena or stadium",
        parent=venue_node,
        critical=True,
    )
    type_claim = (
        "The venue is primarily a theater or performing arts center (e.g., theatre, opera house, concert hall), "
        "and not primarily a sports arena or stadium."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=url,
        additional_instruction=build_missing_source_instruction(url) + " " + type_synonyms_instruction(),
    )

    # ---- Capacity range check (critical) ----
    cap_leaf = evaluator.add_leaf(
        id=f"venue_{index_1_based}_capacity",
        desc="Seating capacity is between 1,000 and 4,000 seats (medium-sized venue classification)",
        parent=venue_node,
        critical=True,
    )
    if venue.seating_capacity:
        cap_claim = (
            f"The venue’s seating capacity is '{venue.seating_capacity}' and this falls within 1,000–4,000 seats."
        )
    else:
        cap_claim = (
            "The venue’s seating capacity falls within 1,000–4,000 seats."
        )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=url,
        additional_instruction=build_missing_source_instruction(url) + " " + capacity_range_instruction(),
    )

    # ---- ADA accessibility check (critical) ----
    ada_leaf = evaluator.add_leaf(
        id=f"venue_{index_1_based}_accessibility",
        desc="Venue is ADA accessible with wheelchair seating provisions",
        parent=venue_node,
        critical=True,
    )
    ada_claim = (
        "The venue is ADA accessible with wheelchair seating provisions."
    )
    await evaluator.verify(
        claim=ada_claim,
        node=ada_leaf,
        sources=url,
        additional_instruction=build_missing_source_instruction(url) + " " + accessibility_synonyms_instruction(),
    )

    # ---- Operational status check (critical) ----
    op_leaf = evaluator.add_leaf(
        id=f"venue_{index_1_based}_operational",
        desc="Venue is currently operational as a performance venue",
        parent=venue_node,
        critical=True,
    )
    op_claim = (
        "The venue is currently operational as a performance venue (open and hosting or scheduling performances)."
    )
    await evaluator.verify(
        claim=op_claim,
        node=op_leaf,
        sources=url,
        additional_instruction=build_missing_source_instruction(url) + " " + operational_instruction(),
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
    Evaluate an answer for the medium-scale theater tour venues task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent venue checks, partial credit allowed
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

    # Record ground truth constraints for transparency
    evaluator.add_ground_truth({
        "allowed_cities": ALLOWED_CITIES,
        "capacity_range_seats": "1000-4000",
        "required_classification": "Theater or Performing Arts Center (not primarily a sports arena/stadium)",
        "must_have": ["ADA accessibility with wheelchair seating", "Currently operational", "Reference URL evidence"]
    }, gt_type="constraints")

    # Extract venue entries from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Prepare up to 4 venues; pad with empty if fewer
    venues_list: List[VenueInfo] = list(extracted.venues[:4])
    while len(venues_list) < 4:
        venues_list.append(VenueInfo())

    # Build verification subtree for each of the 4 venues
    for i in range(1, 5):
        await verify_single_venue(
            evaluator=evaluator,
            parent_node=root,
            venue=venues_list[i - 1],
            index_1_based=i,
        )

    # Return the evaluation summary
    return evaluator.get_summary()