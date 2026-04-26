import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "tx_vr_free_roam_7p"
TASK_DESCRIPTION = (
    "You are coordinating a team-building event for a group of 7 colleagues visiting Texas. "
    "The group wants to experience a free-roam virtual reality gaming session where all 7 members "
    "can play together simultaneously in the same VR arena (not split into separate sessions or groups).\n\n"
    "Your task is to:\n\n"
    "1. Identify the fourth most populous city in Texas based on 2024-2025 population data.\n\n"
    "2. Find the free-roam VR arcade venue(s) in that city capable of accommodating all 7 players simultaneously in a single multiplayer session.\n\n"
    "3. For each qualifying venue, provide:\n"
    "   - The venue name\n"
    "   - The complete street address\n"
    "   - The maximum number of players who can participate simultaneously\n"
    "   - Reference URL(s) to verify the information\n\n"
    "Ensure all information is current, accurate, and supported by reliable sources."
)

# -----------------------------------------------------------------------------
# Data Models for Extraction
# -----------------------------------------------------------------------------
class CityInfo(BaseModel):
    name: Optional[str] = None
    population_reference_urls: List[str] = Field(default_factory=list)


class VenueInfo(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    max_players: Optional[str] = None  # Keep as string to handle ranges or text like "up to 8"
    reference_urls: List[str] = Field(default_factory=list)


class TaskExtraction(BaseModel):
    city: Optional[CityInfo] = None
    venues: List[VenueInfo] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_task_info() -> str:
    return (
        "From the provided answer, extract the following structured information:\n\n"
        "1) city:\n"
        "   - name: The specific Texas city identified as the 4th most populous based on 2024–2025 population data.\n"
        "   - population_reference_urls: URL(s) cited that substantiate the population ranking claim for 2024–2025. "
        "     Extract actual URLs only; include all distinct URLs mentioned.\n\n"
        "2) venues: Extract up to 5 venue entries that the answer presents as candidates in the identified city. "
        "Each venue should include:\n"
        "   - name: Venue name.\n"
        "   - address: Complete street address as provided in the answer.\n"
        "   - max_players: The stated maximum number of simultaneous players (string is acceptable; do not coerce to number).\n"
        "   - reference_urls: URL(s) cited to verify the venue’s nature (free-roam), location, and capacity/details. "
        "     Extract actual URLs only; include all distinct URLs mentioned for that venue.\n\n"
        "Rules:\n"
        "- Extract exactly what is written in the answer; do not invent or infer missing data.\n"
        "- If any field is missing for a venue, set it to null (or empty list for URLs).\n"
        "- If the answer lists more than 5 venues, include only the first 5.\n"
        "- For URLs, allow plain links or markdown; return the actual URL string. If a URL lacks protocol, prepend http://.\n"
    )


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def _nonempty_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())

def _first_k(items: List[Any], k: int) -> List[Any]:
    return items[:k] if items else []

# -----------------------------------------------------------------------------
# Verification Subtrees
# -----------------------------------------------------------------------------
async def build_city_identification(
    evaluator: Evaluator,
    parent_node,
    extraction: TaskExtraction,
) -> str:
    """
    Build and verify the city identification subtree.
    Returns the extracted city name (or empty string if missing).
    """
    city_node = evaluator.add_sequential(
        id="city_identification",
        desc="Identify and verify the fourth most populous city in Texas based on 2024–2025 population data, with supporting sources.",
        parent=parent_node,
        critical=True  # City identification is essential
    )

    city_name = extraction.city.name if extraction.city else None
    pop_urls = extraction.city.population_reference_urls if extraction.city else []

    # Leaf: City name provided (existence)
    evaluator.add_custom_node(
        result=_nonempty_str(city_name),
        id="city_name_provided",
        desc="Provide a specific city name in Texas.",
        parent=city_node,
        critical=True
    )

    # Parallel: Rank + Source verification group
    rank_group = evaluator.add_parallel(
        id="city_rank_and_source",
        desc="Support that the provided city is the 4th most populous in Texas using 2024–2025 population data.",
        parent=city_node,
        critical=True
    )

    # Existence of population reference URLs
    pop_urls_node = evaluator.add_custom_node(
        result=(isinstance(pop_urls, list) and len(pop_urls) > 0),
        id="population_reference_urls",
        desc="Provide reference URL(s) that substantiate the 2024–2025 population ranking claim.",
        parent=rank_group,
        critical=True
    )

    # Verification: Fourth rank supported by provided URLs
    rank_leaf = evaluator.add_leaf(
        id="fourth_rank_verification",
        desc="Evidence shows the provided city ranks 4th by population in Texas for 2024–2025 data.",
        parent=rank_group,
        critical=True
    )
    city_str = city_name or ""
    claim_rank = (
        f"{city_str} is the 4th most populous city in Texas based on 2024–2025 population data."
    )
    await evaluator.verify(
        claim=claim_rank,
        node=rank_leaf,
        sources=pop_urls,
        additional_instruction=(
            "Check the provided URLs for credible 2024 or 2025 Texas city population ranking (e.g., US Census estimates, "
            "state demography, reputable ranking articles). Confirm that the cited city is ranked 4th for 2024–2025."
        ),
    )

    return city_str


async def build_venue_entry(
    evaluator: Evaluator,
    parent_node,
    venue: VenueInfo,
    city_name: str,
    idx: int,
):
    """
    Build verification subtree for a single venue entry.
    """
    v_node = evaluator.add_parallel(
        id=f"venue_{idx+1}",
        desc=f"Evaluate venue #{idx+1} (if provided).",
        parent=parent_node,
        critical=False  # Each venue independently contributes partial credit
    )

    # Details group (critical): presence of required fields + reference URLs
    details_group = evaluator.add_parallel(
        id=f"venue_{idx+1}_details",
        desc=f"Venue #{idx+1} includes all required reporting fields with verification URLs.",
        parent=v_node,
        critical=True
    )

    name_node = evaluator.add_custom_node(
        result=_nonempty_str(venue.name),
        id=f"venue_{idx+1}_name",
        desc=f"Provide the venue name for venue #{idx+1}.",
        parent=details_group,
        critical=True
    )

    address_node = evaluator.add_custom_node(
        result=_nonempty_str(venue.address),
        id=f"venue_{idx+1}_address",
        desc=f"Provide the complete street address for venue #{idx+1}.",
        parent=details_group,
        critical=True
    )

    maxcap_node = evaluator.add_custom_node(
        result=_nonempty_str(venue.max_players),
        id=f"venue_{idx+1}_max_capacity",
        desc=f"State the maximum simultaneous player capacity for venue #{idx+1}.",
        parent=details_group,
        critical=True
    )

    refurls_node = evaluator.add_custom_node(
        result=(isinstance(venue.reference_urls, list) and len(venue.reference_urls) > 0),
        id=f"venue_{idx+1}_reference_urls",
        desc=f"Provide reference URL(s) that verify venue #{idx+1}’s free-roam nature, location, and capacity/details.",
        parent=details_group,
        critical=True
    )

    # Qualification group (critical): free-roam, in-city, capacity >= 7 simultaneously
    qual_group = evaluator.add_parallel(
        id=f"venue_{idx+1}_qualification",
        desc=f"Venue #{idx+1} meets all qualification criteria.",
        parent=v_node,
        critical=True
    )

    # Free-roam verification
    freeroam_leaf = evaluator.add_leaf(
        id=f"venue_{idx+1}_free_roam",
        desc=f"Venue #{idx+1} is a free-roam VR arcade (location-based VR entertainment).",
        parent=qual_group,
        critical=True
    )
    venue_name_str = venue.name or ""
    claim_free_roam = (
        f"The venue '{venue_name_str}' offers free-roam VR experiences in a shared arena where players physically move "
        f"without being tethered to a fixed booth (warehouse-scale/arena-scale LBE VR)."
    )
    await evaluator.verify(
        claim=claim_free_roam,
        node=freeroam_leaf,
        sources=venue.reference_urls,
        additional_instruction=(
            "From the provided URLs, verify that the venue explicitly offers free-roam VR (arena-scale/warehouse-scale), "
            "not stationary booth-only VR. Look for terms like 'free roam', 'arena', 'warehouse-scale', 'untethered multiplayer'."
        ),
        extra_prerequisites=[refurls_node],
    )

    # In-city verification
    in_city_leaf = evaluator.add_leaf(
        id=f"venue_{idx+1}_in_city",
        desc=f"Venue #{idx+1} is physically located in the identified city.",
        parent=qual_group,
        critical=True
    )
    addr_str = venue.address or ""
    claim_in_city = (
        f"The venue '{venue_name_str}' is located in {city_name}, Texas. "
        f"The address provided is '{addr_str}'."
    )
    await evaluator.verify(
        claim=claim_in_city,
        node=in_city_leaf,
        sources=venue.reference_urls,
        additional_instruction=(
            "Verify via the provided URLs that the venue's address or location is within the city specified, "
            "i.e., it should indicate '{city}, TX' for the identified city. Minor variations like full state name "
            "or neighborhoods within the city are acceptable."
        ).replace("{city}", city_name or ""),
        extra_prerequisites=[refurls_node],
    )

    # Capacity verification (>=7 simultaneous in one session)
    capacity_leaf = evaluator.add_leaf(
        id=f"venue_{idx+1}_capacity_7_single_session",
        desc=f"Venue #{idx+1} can accommodate at least 7 players simultaneously in a single multiplayer session (not split).",
        parent=qual_group,
        critical=True
    )
    max_players_str = venue.max_players or ""
    claim_capacity = (
        f"The venue '{venue_name_str}' can host 7 players simultaneously in a single multiplayer session (same arena), "
        f"and its stated maximum simultaneous player capacity is '{max_players_str}'."
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_leaf,
        sources=venue.reference_urls,
        additional_instruction=(
            "Confirm that at least one game or the arena configuration explicitly supports 7 players at the same time "
            "in a single session (not split into multiple groups). If multiple capacities are listed per game, "
            "ensure 7 simultaneous players is supported."
        ),
        extra_prerequisites=[refurls_node],
    )


async def build_venue_search_and_reporting(
    evaluator: Evaluator,
    parent_node,
    extraction: TaskExtraction,
    city_name: str,
):
    """
    Build venue search and reporting subtree (evaluate up to 5 venues).
    """
    venues_node = evaluator.add_sequential(
        id="venue_search_and_reporting",
        desc="Provide qualifying free-roam VR venue(s) in the identified city that can host all 7 players simultaneously in one session, including required details and verification links (evaluate up to 5 venues).",
        parent=parent_node,
        critical=False  # Keep non-critical to allow non-critical children without violating critical constraints
    )

    venues = _first_k(extraction.venues, 5)

    # At least one venue provided (existence)
    evaluator.add_custom_node(
        result=(len([v for v in venues if _nonempty_str(v.name)]) >= 1),
        id="at_least_one_venue_provided",
        desc="Provide at least one candidate venue in the identified city (i.e., at least one venue entry is present to evaluate).",
        parent=venues_node,
        critical=True
    )

    # Parallel evaluation of up to 5 venue entries
    venues_group = evaluator.add_parallel(
        id="venues_up_to_5",
        desc="Evaluate up to 5 provided venue entries; each venue is scored independently for partial credit.",
        parent=venues_node,
        critical=False  # Must be non-critical to allow non-critical children
    )

    for idx, venue in enumerate(venues):
        await build_venue_entry(evaluator, venues_group, venue, city_name, idx)


# -----------------------------------------------------------------------------
# Main Evaluation Entry Point
# -----------------------------------------------------------------------------
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the agent's answer for the Texas free-roam VR (7 players) task.
    """
    # Initialize evaluator with sequential root to enforce phase ordering
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured information (city + venues)
    extraction = await evaluator.extract(
        prompt=prompt_extract_task_info(),
        template_class=TaskExtraction,
        extraction_name="structured_task_info",
    )

    # Phase 1: City identification
    city_name = await build_city_identification(evaluator, root, extraction)

    # Phase 2: Venue search & reporting (depends on phase 1 via sequential root)
    await build_venue_search_and_reporting(evaluator, root, extraction, city_name)

    # Return structured summary
    return evaluator.get_summary()