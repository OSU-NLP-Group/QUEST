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
TASK_ID = "nba_arenas_2025_4_cities"
TASK_DESCRIPTION = (
    "For a 2025 concert tour planning project, research the NBA arenas in the following four cities: "
    "Chicago (Illinois), New York (New York), Los Angeles (California), and Miami (Florida). For each city, "
    "identify the NBA arena where the city's NBA team plays, and provide the following information: "
    "(1) The official name of the arena, (2) The arena's seating capacity specifically for basketball games, "
    "(3) The minimum number of wheelchair accessible seats required by ADA regulations (calculated as 1% of the total capacity, showing your calculation), "
    "(4) Confirmation of the city and state where the arena is located, and "
    "(5) Reference URLs from reliable sources to verify: (a) the arena's identity, (b) the capacity figure, "
    "(c) the ADA 1% wheelchair seating requirement, and (d) the location. Present your findings organized by city, "
    "with all required information and supporting reference URLs for each arena."
)

CITY_EXPECTATIONS = {
    "chicago": {"city": "Chicago", "state": "Illinois"},
    "newyork": {"city": "New York", "state": "New York"},
    "losangeles": {"city": "Los Angeles", "state": "California"},
    "miami": {"city": "Miami", "state": "Florida"},
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CityArenaInfo(BaseModel):
    # Arena identification
    arena_name: Optional[str] = None
    identification_urls: List[str] = Field(default_factory=list)

    # Capacity (basketball-specific)
    capacity_basketball: Optional[str] = None
    capacity_urls: List[str] = Field(default_factory=list)

    # ADA 1% wheelchair calculation
    ada_calc_expression: Optional[str] = None  # e.g., "1% of 20,917 = ~209"
    ada_result: Optional[str] = None           # e.g., "209"
    ada_reference_urls: List[str] = Field(default_factory=list)

    # Location verification
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)


class AllCitiesExtraction(BaseModel):
    chicago: Optional[CityArenaInfo] = None
    newyork: Optional[CityArenaInfo] = None
    losangeles: Optional[CityArenaInfo] = None
    miami: Optional[CityArenaInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract, from the provided answer, structured information for four cities' NBA arenas: Chicago (Illinois), New York (New York), Los Angeles (California), and Miami (Florida).

For each city, return an object with the following fields:
- arena_name: The official current name of the NBA arena where the city's NBA team plays its home games.
- identification_urls: A list of URL(s) that confirm the arena identity (at least one URL if provided in the answer).
- capacity_basketball: The seating capacity specifically for basketball games (as stated in the answer; extract as a string exactly as written).
- capacity_urls: A list of URL(s) that support the basketball seating capacity figure.
- ada_calc_expression: The explicit calculation the answer shows for “1% of capacity” (e.g., “1% of 20,917 = 209.17 ≈ 210”), if present.
- ada_result: The numerical result the answer provides for the 1% requirement (e.g., “209” or “210”), if present.
- ada_reference_urls: A list of URL(s) used to reference the ADA 1% wheelchair seating requirement, if any were provided in the answer. If the answer uses a single ADA reference for multiple cities, repeat it for each relevant city.
- location_city: The city name stated for the arena location.
- location_state: The state name stated for the arena location.
- location_urls: A list of URL(s) that support the arena’s location.

Important:
- Extract only what is explicitly present in the answer. Do not invent or infer.
- All URL fields must be actual URLs present in the answer (plain URLs or markdown links). If none are provided in the answer for a field, return an empty list.
- Keep all values as strings exactly as they appear in the answer (do not normalize numbers).
- Organize the top-level JSON with keys chicago, newyork, losangeles, miami, each mapped to their respective object. If a city is missing from the answer, return null for that city.
"""


# --------------------------------------------------------------------------- #
# Verification helper                                                         #
# --------------------------------------------------------------------------- #
def _safe_city_info(ci: Optional[CityArenaInfo]) -> CityArenaInfo:
    return ci if ci is not None else CityArenaInfo()


# --------------------------------------------------------------------------- #
# City verification subtrees                                                  #
# --------------------------------------------------------------------------- #
async def verify_city(
    evaluator: Evaluator,
    parent,
    city_key: str,
    city_desc: str,
    info: CityArenaInfo,
    expected_city: str,
    expected_state: str,
) -> None:
    """
    Build and execute verification nodes for one city according to the rubric.
    """

    # Top-level city node (sequential): if early critical block fails, later parts are skipped
    city_node = evaluator.add_sequential(
        id=f"{city_key}_arena",
        desc=city_desc,
        parent=parent,
        critical=False
    )

    # 1) Arena identification (parallel, critical)
    ident_node = evaluator.add_parallel(
        id=f"{city_key}_arena_identification",
        desc=f"Identify the correct NBA arena located in {expected_city}",
        parent=city_node,
        critical=True
    )

    # 1.a) identification_url (critical leaf — existence check)
    evaluator.add_custom_node(
        result=bool(info.identification_urls),
        id=f"{city_key}_identification_url",
        desc="Provide a reference URL confirming the arena identity",
        parent=ident_node,
        critical=True
    )

    # 1.b) arena_name (critical leaf — verify by identification URLs)
    arena_name_leaf = evaluator.add_leaf(
        id=f"{city_key}_arena_name",
        desc=f"Provide the official name of {expected_city}'s NBA arena",
        parent=ident_node,
        critical=True
    )
    arena_name_val = info.arena_name or ""
    await evaluator.verify(
        claim=(
            f"The NBA arena where the city's NBA team plays in {expected_city}, {expected_state} is named "
            f"'{arena_name_val}'."
        ),
        node=arena_name_leaf,
        sources=info.identification_urls,
        additional_instruction=(
            "Check that the provided page(s) clearly identify the arena by its official/current name and "
            "that it is the NBA home venue for the city's team. Allow minor name variations (e.g., punctuation, "
            "sponsorship naming) if clearly the same arena."
        )
    )

    # 2) Capacity details (parallel, critical)
    cap_node = evaluator.add_parallel(
        id=f"{city_key}_capacity_details",
        desc=f"Provide accurate capacity information for the {expected_city} arena",
        parent=city_node,
        critical=True
    )

    # 2.a) capacity_url (critical leaf — existence check)
    evaluator.add_custom_node(
        result=bool(info.capacity_urls),
        id=f"{city_key}_capacity_url",
        desc="Provide a reference URL confirming the capacity figure",
        parent=cap_node,
        critical=True
    )

    # 2.b) capacity_value (critical leaf — verify by capacity URLs)
    cap_value_leaf = evaluator.add_leaf(
        id=f"{city_key}_capacity_value",
        desc="State the arena's seating capacity for basketball games",
        parent=cap_node,
        critical=True
    )
    cap_val = info.capacity_basketball or ""
    await evaluator.verify(
        claim=(
            f"The seating capacity for basketball games at '{arena_name_val}' is '{cap_val}'."
        ),
        node=cap_value_leaf,
        sources=info.capacity_urls,
        additional_instruction=(
            "Confirm the basketball-specific seating capacity. Do not use concert or hockey capacities. "
            "Allow small formatting differences (commas, approximate markers like '~' or 'about') if they represent "
            "the same value."
        )
    )

    # 3) Accessibility requirement (parallel, critical)
    access_node = evaluator.add_parallel(
        id=f"{city_key}_accessibility_requirement",
        desc="Calculate and provide the minimum wheelchair accessible seating requirement based on ADA 1% rule",
        parent=city_node,
        critical=True
    )

    # 3.a) wheelchair_calculation (parallel, critical)
    calc_node = evaluator.add_parallel(
        id=f"{city_key}_wheelchair_calculation",
        desc="Calculate 1% of the arena's total capacity for wheelchair accessible seats",
        parent=access_node,
        critical=True
    )

    # 3.a.i) calculation_shown (critical leaf — verify from the answer)
    calc_shown_leaf = evaluator.add_leaf(
        id=f"{city_key}_calculation_shown",
        desc="Show the calculation of 1% of stated capacity",
        parent=calc_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The answer explicitly shows the calculation of 1% of the stated basketball capacity ('{cap_val}') "
            f"for the {expected_city} arena."
        ),
        node=calc_shown_leaf,
        # Use simple verification against the answer text
        sources=None,
        additional_instruction=(
            "Look for an explicit calculation, e.g., '1% of 20,917 = 209.17 ≈ 209' or '20,917 * 0.01 = 209.17'. "
            "It must show the operation, not just the final number."
        )
    )

    # 3.a.ii) result_provided (critical leaf — verify result presence from the answer)
    result_provided_leaf = evaluator.add_leaf(
        id=f"{city_key}_result_provided",
        desc="Provide the numerical result of the wheelchair seat requirement",
        parent=calc_node,
        critical=True
    )
    ada_result_val = info.ada_result or ""
    await evaluator.verify(
        claim=(
            f"The answer provides a numerical result for the 1% wheelchair seating requirement: '{ada_result_val}'."
        ),
        node=result_provided_leaf,
        sources=None,
        additional_instruction=(
            "Verify that the answer includes a concrete numeric result (e.g., 209) for the 1% calculation. "
            "It must be a number, not just text."
        )
    )

    # 3.b) ada_reference_url (critical leaf — existence check of ADA reference URL)
    evaluator.add_custom_node(
        result=bool(info.ada_reference_urls),
        id=f"{city_key}_ada_reference_url",
        desc="Provide a reference URL for the ADA 1% wheelchair seating requirement",
        parent=access_node,
        critical=True
    )

    # 4) Location verification (parallel, critical)
    loc_node = evaluator.add_parallel(
        id=f"{city_key}_location_verification",
        desc="Verify and confirm the arena's location details",
        parent=city_node,
        critical=True
    )

    # 4.a) location_url (critical leaf — existence check)
    evaluator.add_custom_node(
        result=bool(info.location_urls),
        id=f"{city_key}_location_url",
        desc="Provide a reference URL confirming the location",
        parent=loc_node,
        critical=True
    )

    # 4.b) city_state (critical leaf — verify location by URL(s))
    city_state_leaf = evaluator.add_leaf(
        id=f"{city_key}_city_state",
        desc=f"Confirm the arena is located in {expected_city}, {expected_state}",
        parent=loc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{arena_name_val}' is located in {expected_city}, {expected_state}.",
        node=city_state_leaf,
        sources=info.location_urls,
        additional_instruction=(
            "Confirm the arena's physical location. Allow minor formatting variants (e.g., boroughs/districts) "
            "but it must be within the stated city and state."
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
) -> Dict:
    """
    Build the verification tree and run checks for the NBA arena research task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Four cities are independent
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AllCitiesExtraction,
        extraction_name="arenas_by_city"
    )

    # Optionally record expected city/state mapping as ground truth context
    evaluator.add_ground_truth(
        {
            "expected_cities": CITY_EXPECTATIONS,
            "notes": "Expected city/state for each verification branch."
        },
        gt_type="expected_city_state"
    )

    # Build subtrees for each city (sequential within each city)
    cities_plan = [
        ("chicago", "Identify and provide complete information about the NBA arena in Chicago, Illinois"),
        ("newyork", "Identify and provide complete information about the NBA arena in New York, New York"),
        ("losangeles", "Identify and provide complete information about the NBA arena in Los Angeles, California"),
        ("miami", "Identify and provide complete information about the NBA arena in Miami, Florida"),
    ]

    # Prepare city info mapping from extraction (with safe defaults)
    extraction_map: Dict[str, CityArenaInfo] = {
        "chicago": _safe_city_info(extraction.chicago),
        "newyork": _safe_city_info(extraction.newyork),
        "losangeles": _safe_city_info(extraction.losangeles),
        "miami": _safe_city_info(extraction.miami),
    }

    # Create city nodes and perform verifications
    for ckey, cdesc in cities_plan:
        exp = CITY_EXPECTATIONS[ckey]
        await verify_city(
            evaluator=evaluator,
            parent=root,
            city_key=ckey,
            city_desc=cdesc,
            info=extraction_map[ckey],
            expected_city=exp["city"],
            expected_state=exp["state"],
        )

    # Return the full evaluation summary
    return evaluator.get_summary()