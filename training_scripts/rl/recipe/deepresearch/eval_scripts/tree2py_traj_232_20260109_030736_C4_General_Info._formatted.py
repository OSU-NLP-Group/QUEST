import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "empire_state_building_specs"
TASK_DESCRIPTION = """
Provide the following architectural and visitor information about the Empire State Building in New York City:

1. The total height including the antenna/spire
2. The roof height (without antenna)
3. The total number of stories
4. Which floor the primary observation deck is located on
5. The height of the primary observation deck
6. Which floor the top observation deck is located on
7. The total number of observation decks in the building

Include reference URLs from reliable sources to support each piece of information.
"""

# Canonical ground truth values per rubric
GROUND_TRUTH = {
    "total_height_with_antenna_ft": "1,454 feet",
    "total_height_with_antenna_m": "443.2 meters",
    "roof_height_ft": "1,250 feet",
    "roof_height_m": "381 meters",
    "total_stories": "102",
    "primary_observation_deck_floor": "86",
    "primary_observation_deck_height_ft": "1,050 feet",
    "primary_observation_deck_height_m": "320 meters",
    "top_observation_deck_floor": "102",
    "observation_deck_count": "3",
    "observation_deck_floors_expected": ["80", "86", "102"],
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ESBExtraction(BaseModel):
    # Entity confirmation (for sanity)
    entity_name: Optional[str] = None
    entity_location: Optional[str] = None
    entity_sources: List[str] = Field(default_factory=list)

    # 1. Total height with antenna/spire
    total_height_with_antenna: Optional[str] = None
    total_height_with_antenna_sources: List[str] = Field(default_factory=list)

    # 2. Roof height (without antenna)
    roof_height: Optional[str] = None
    roof_height_sources: List[str] = Field(default_factory=list)

    # 3. Total number of stories
    total_stories: Optional[str] = None
    total_stories_sources: List[str] = Field(default_factory=list)

    # 4. Primary/main observation deck floor
    primary_observation_deck_floor: Optional[str] = None
    primary_observation_deck_floor_sources: List[str] = Field(default_factory=list)

    # 5. Primary observation deck height
    primary_observation_deck_height: Optional[str] = None
    primary_observation_deck_height_sources: List[str] = Field(default_factory=list)

    # 6. Top observation deck floor
    top_observation_deck_floor: Optional[str] = None
    top_observation_deck_floor_sources: List[str] = Field(default_factory=list)

    # 7. Number of observation decks and their floors
    total_observation_decks: Optional[str] = None
    observation_deck_floors: List[str] = Field(default_factory=list)
    observation_decks_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_esb_info() -> str:
    return """
    Extract structured information from the answer about the Empire State Building (ESB). Return exactly the following fields:

    - entity_name: The building name mentioned (e.g., "Empire State Building")
    - entity_location: The location mentioned (e.g., "Manhattan, New York City" or similar)
    - entity_sources: URL(s) cited that support the entity/location (if any)

    - total_height_with_antenna: The total height including antenna/spire as stated in the answer (keep original wording with units)
    - total_height_with_antenna_sources: All URL(s) cited that support the total height including antenna/spire

    - roof_height: The roof height (without antenna) as stated in the answer (keep original wording with units)
    - roof_height_sources: All URL(s) cited that support the roof height

    - total_stories: Total number of stories as stated in the answer (as a string; do not convert to number)
    - total_stories_sources: All URL(s) cited that support the total stories

    - primary_observation_deck_floor: The floor number for the primary/main observation deck as stated (as a string; e.g., "86" or "86th floor")
    - primary_observation_deck_floor_sources: URL(s) cited that support the primary observation deck floor

    - primary_observation_deck_height: The height of the primary (86th) observation deck as stated (keep units, e.g., "1,050 feet (320 meters)")
    - primary_observation_deck_height_sources: URL(s) cited that support the primary deck height

    - top_observation_deck_floor: The floor number for the top observation deck as stated (as a string; e.g., "102" or "102nd floor")
    - top_observation_deck_floor_sources: URL(s) cited that support the top observation deck floor

    - total_observation_decks: The total number of observation decks as stated (string; e.g., "3")
    - observation_deck_floors: A list of the floor numbers for the observation decks as stated (each item as a string; e.g., ["80", "86", "102"] or ["80th", "86th", "102nd"])
    - observation_decks_sources: URL(s) cited that support the number and floors of the observation decks

    IMPORTANT:
    - Extract only what is explicitly provided in the answer.
    - For URL fields, include only actual URLs present in the answer (markdown links should be resolved to their URL).
    - If any field is missing in the answer, set it to null (for strings) or an empty list (for lists).
    - Do not invent or infer values or URLs that are not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions for verification                                           #
# --------------------------------------------------------------------------- #
def _normalize_floor_list_str(floors: List[str]) -> str:
    if not floors:
        return ""
    return ", ".join(floors)


async def _add_entity_check(evaluator: Evaluator, parent, data: ESBExtraction) -> None:
    node = evaluator.add_leaf(
        id="Correct_Entity_and_Location",
        desc="Answer is specifically about the Empire State Building located in Manhattan, New York City.",
        parent=parent,
        critical=True,
    )
    claim = "The building discussed in the answer is the Empire State Building located in Manhattan, New York City."
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Check the answer text context to ensure it clearly and explicitly refers to the Empire State Building in Manhattan, NYC."
    )


async def _add_height_with_antenna_checks(evaluator: Evaluator, parent, data: ESBExtraction) -> None:
    group = evaluator.add_sequential(
        id="Total_Height_with_Antenna",
        desc="Provides total height including antenna/spire as 1,454 feet (443.2 meters).",
        parent=parent,
        critical=True,
    )
    provided = evaluator.add_custom_node(
        result=bool(data.total_height_with_antenna and data.total_height_with_antenna.strip()),
        id="Total_Height_with_Antenna_Provided",
        desc="Total height including antenna/spire is provided in the answer.",
        parent=group,
        critical=True,
    )
    match_expected = evaluator.add_leaf(
        id="Total_Height_with_Antenna_Matches_Expected",
        desc="Provided total height including antenna equals 1,454 feet (443.2 meters).",
        parent=group,
        critical=True,
    )
    exp_ft = GROUND_TRUTH["total_height_with_antenna_ft"]
    exp_m = GROUND_TRUTH["total_height_with_antenna_m"]
    claim_match = f"The extracted total height with antenna '{data.total_height_with_antenna}' equals the canonical value '{exp_ft} ({exp_m})' allowing only formatting differences (e.g., 'ft' vs 'feet', comma usage, parentheses). The numeric feet value must be 1,454."
    await evaluator.verify(
        claim=claim_match,
        node=match_expected,
        additional_instruction="Do not allow 1,453 or 1,455. Minor formatting variations are OK."
    )
    src_exist = evaluator.add_custom_node(
        result=len(data.total_height_with_antenna_sources) > 0,
        id="Total_Height_with_Antenna_Sources_Provided",
        desc="Source URLs are provided for total height including antenna/spire.",
        parent=group,
        critical=True,
    )
    src_support = evaluator.add_leaf(
        id="Total_Height_with_Antenna_Supported_By_Sources",
        desc="Total height including antenna/spire is supported by cited sources.",
        parent=group,
        critical=True,
    )
    claim_support = f"The total height including the antenna/spire is {data.total_height_with_antenna}."
    await evaluator.verify(
        claim=claim_support,
        node=src_support,
        sources=data.total_height_with_antenna_sources,
        additional_instruction="Confirm the page explicitly states the total height including antenna/spire. Allow minor formatting differences; the numeric feet value should correspond to 1,454."
    )


async def _add_roof_height_checks(evaluator: Evaluator, parent, data: ESBExtraction) -> None:
    group = evaluator.add_sequential(
        id="Roof_Height",
        desc="Provides roof height (without antenna) as 1,250 feet (381 meters).",
        parent=parent,
        critical=True,
    )
    provided = evaluator.add_custom_node(
        result=bool(data.roof_height and data.roof_height.strip()),
        id="Roof_Height_Provided",
        desc="Roof height (without antenna) is provided in the answer.",
        parent=group,
        critical=True,
    )
    match_expected = evaluator.add_leaf(
        id="Roof_Height_Matches_Expected",
        desc="Provided roof height equals 1,250 feet (381 meters).",
        parent=group,
        critical=True,
    )
    exp_ft = GROUND_TRUTH["roof_height_ft"]
    exp_m = GROUND_TRUTH["roof_height_m"]
    claim_match = f"The extracted roof height '{data.roof_height}' equals the canonical value '{exp_ft} ({exp_m})', allowing only formatting differences. The numeric feet value must be 1,250."
    await evaluator.verify(
        claim=claim_match,
        node=match_expected,
        additional_instruction="Do not allow 1,249 or 1,251. Minor formatting differences are OK."
    )
    src_exist = evaluator.add_custom_node(
        result=len(data.roof_height_sources) > 0,
        id="Roof_Height_Sources_Provided",
        desc="Source URLs are provided for roof height.",
        parent=group,
        critical=True,
    )
    src_support = evaluator.add_leaf(
        id="Roof_Height_Supported_By_Sources",
        desc="Roof height is supported by cited sources.",
        parent=group,
        critical=True,
    )
    claim_support = f"The roof height (without antenna) is {data.roof_height}."
    await evaluator.verify(
        claim=claim_support,
        node=src_support,
        sources=data.roof_height_sources,
        additional_instruction="Confirm the page explicitly states the roof height (without antenna)."
    )


async def _add_total_stories_checks(evaluator: Evaluator, parent, data: ESBExtraction) -> None:
    group = evaluator.add_sequential(
        id="Total_Number_of_Stories",
        desc="States the total number of stories as 102.",
        parent=parent,
        critical=True,
    )
    provided = evaluator.add_custom_node(
        result=bool(data.total_stories and data.total_stories.strip()),
        id="Total_Stories_Provided",
        desc="Total number of stories is provided in the answer.",
        parent=group,
        critical=True,
    )
    match_expected = evaluator.add_leaf(
        id="Total_Stories_Matches_Expected",
        desc="Provided total number of stories equals 102.",
        parent=group,
        critical=True,
    )
    exp = GROUND_TRUTH["total_stories"]
    claim_match = f"The extracted total number of stories '{data.total_stories}' equals the canonical value '{exp}'."
    await evaluator.verify(
        claim=claim_match,
        node=match_expected,
        additional_instruction="Treat '102', '102 stories', or '102 floors' as equivalent when clearly meaning story count."
    )
    src_exist = evaluator.add_custom_node(
        result=len(data.total_stories_sources) > 0,
        id="Total_Stories_Sources_Provided",
        desc="Source URLs are provided for total number of stories.",
        parent=group,
        critical=True,
    )
    src_support = evaluator.add_leaf(
        id="Total_Stories_Supported_By_Sources",
        desc="Total number of stories is supported by cited sources.",
        parent=group,
        critical=True,
    )
    claim_support = f"The building has {data.total_stories} stories."
    await evaluator.verify(
        claim=claim_support,
        node=src_support,
        sources=data.total_stories_sources,
        additional_instruction="Confirm the page explicitly states the total number of stories."
    )


async def _add_primary_deck_floor_checks(evaluator: Evaluator, parent, data: ESBExtraction) -> None:
    group = evaluator.add_sequential(
        id="Primary_Observation_Deck_Floor",
        desc="Identifies the primary/main observation deck as located on the 86th floor.",
        parent=parent,
        critical=True,
    )
    provided = evaluator.add_custom_node(
        result=bool(data.primary_observation_deck_floor and data.primary_observation_deck_floor.strip()),
        id="Primary_Deck_Floor_Provided",
        desc="Primary observation deck floor is provided.",
        parent=group,
        critical=True,
    )
    match_expected = evaluator.add_leaf(
        id="Primary_Deck_Floor_Matches_Expected",
        desc="Primary observation deck floor equals 86th floor.",
        parent=group,
        critical=True,
    )
    exp = GROUND_TRUTH["primary_observation_deck_floor"]
    claim_match = f"The extracted primary observation deck floor '{data.primary_observation_deck_floor}' is equivalent to the 86th floor (e.g., '86', '86th', or 'floor 86')."
    await evaluator.verify(
        claim=claim_match,
        node=match_expected,
        additional_instruction="Allow minor phrasing variations; ensure it clearly denotes floor 86."
    )
    src_exist = evaluator.add_custom_node(
        result=len(data.primary_observation_deck_floor_sources) > 0,
        id="Primary_Deck_Floor_Sources_Provided",
        desc="Source URLs are provided for the primary observation deck floor.",
        parent=group,
        critical=True,
    )
    src_support = evaluator.add_leaf(
        id="Primary_Deck_Floor_Supported_By_Sources",
        desc="Primary observation deck floor is supported by cited sources.",
        parent=group,
        critical=True,
    )
    claim_support = f"The primary (main) observation deck is located on the {data.primary_observation_deck_floor} floor."
    await evaluator.verify(
        claim=claim_support,
        node=src_support,
        sources=data.primary_observation_deck_floor_sources,
        additional_instruction="Confirm the page explicitly states the main/primary observatory is at floor 86."
    )


async def _add_primary_deck_height_checks(evaluator: Evaluator, parent, data: ESBExtraction) -> None:
    group = evaluator.add_sequential(
        id="Primary_Observation_Deck_Height",
        desc="Provides the height of the primary (86th floor) observation deck as 1,050 feet (320 meters).",
        parent=parent,
        critical=True,
    )
    provided = evaluator.add_custom_node(
        result=bool(data.primary_observation_deck_height and data.primary_observation_deck_height.strip()),
        id="Primary_Deck_Height_Provided",
        desc="Primary observation deck height is provided.",
        parent=group,
        critical=True,
    )
    match_expected = evaluator.add_leaf(
        id="Primary_Deck_Height_Matches_Expected",
        desc="Primary observation deck height equals 1,050 feet (320 meters).",
        parent=group,
        critical=True,
    )
    exp_ft = GROUND_TRUTH["primary_observation_deck_height_ft"]
    exp_m = GROUND_TRUTH["primary_observation_deck_height_m"]
    claim_match = f"The extracted primary observation deck height '{data.primary_observation_deck_height}' equals '{exp_ft} ({exp_m})', allowing only formatting differences. The numeric feet value must be 1,050."
    await evaluator.verify(
        claim=claim_match,
        node=match_expected,
        additional_instruction="Do not allow 1,049 or 1,051. Minor formatting differences OK."
    )
    src_exist = evaluator.add_custom_node(
        result=len(data.primary_observation_deck_height_sources) > 0,
        id="Primary_Deck_Height_Sources_Provided",
        desc="Source URLs are provided for primary observation deck height.",
        parent=group,
        critical=True,
    )
    src_support = evaluator.add_leaf(
        id="Primary_Deck_Height_Supported_By_Sources",
        desc="Primary observation deck height is supported by cited sources.",
        parent=group,
        critical=True,
    )
    claim_support = f"The height of the primary (86th floor) observation deck is {data.primary_observation_deck_height}."
    await evaluator.verify(
        claim=claim_support,
        node=src_support,
        sources=data.primary_observation_deck_height_sources,
        additional_instruction="Confirm the page explicitly states the height of the 86th-floor observatory."
    )


async def _add_top_deck_floor_checks(evaluator: Evaluator, parent, data: ESBExtraction) -> None:
    group = evaluator.add_sequential(
        id="Top_Observation_Deck_Floor",
        desc="Identifies the top observation deck as located on the 102nd floor.",
        parent=parent,
        critical=True,
    )
    provided = evaluator.add_custom_node(
        result=bool(data.top_observation_deck_floor and data.top_observation_deck_floor.strip()),
        id="Top_Deck_Floor_Provided",
        desc="Top observation deck floor is provided.",
        parent=group,
        critical=True,
    )
    match_expected = evaluator.add_leaf(
        id="Top_Deck_Floor_Matches_Expected",
        desc="Top observation deck floor equals 102nd floor.",
        parent=group,
        critical=True,
    )
    exp = GROUND_TRUTH["top_observation_deck_floor"]
    claim_match = f"The extracted top observation deck floor '{data.top_observation_deck_floor}' is equivalent to the 102nd floor (e.g., '102', '102nd', 'floor 102')."
    await evaluator.verify(
        claim=claim_match,
        node=match_expected,
        additional_instruction="Allow minor phrasing variations; ensure it clearly denotes floor 102."
    )
    src_exist = evaluator.add_custom_node(
        result=len(data.top_observation_deck_floor_sources) > 0,
        id="Top_Deck_Floor_Sources_Provided",
        desc="Source URLs are provided for the top observation deck floor.",
        parent=group,
        critical=True,
    )
    src_support = evaluator.add_leaf(
        id="Top_Deck_Floor_Supported_By_Sources",
        desc="Top observation deck floor is supported by cited sources.",
        parent=group,
        critical=True,
    )
    claim_support = f"The top observation deck is located on the {data.top_observation_deck_floor} floor."
    await evaluator.verify(
        claim=claim_support,
        node=src_support,
        sources=data.top_observation_deck_floor_sources,
        additional_instruction="Confirm the page explicitly states the top observatory is at floor 102."
    )


async def _add_num_and_locations_decks_checks(evaluator: Evaluator, parent, data: ESBExtraction) -> None:
    group = evaluator.add_sequential(
        id="Number_and_Locations_of_Observation_Decks",
        desc="States there are three observation decks and specifies they are on the 80th, 86th, and 102nd floors.",
        parent=parent,
        critical=True,
    )
    provided = evaluator.add_custom_node(
        result=bool(data.total_observation_decks and data.total_observation_decks.strip()) and len(data.observation_deck_floors) > 0,
        id="Observation_Decks_Provided",
        desc="Number of observation decks and their floor locations are provided.",
        parent=group,
        critical=True,
    )
    count_matches = evaluator.add_leaf(
        id="Observation_Deck_Count_Matches_Expected",
        desc="Total number of observation decks equals 3.",
        parent=group,
        critical=True,
    )
    exp_count = GROUND_TRUTH["observation_deck_count"]
    claim_count = f"The extracted total observation deck count '{data.total_observation_decks}' equals '{exp_count}'."
    await evaluator.verify(
        claim=claim_count,
        node=count_matches,
        additional_instruction="Allow phrasing like 'three observation decks' or '3 decks' as equivalent."
    )
    floors_match = evaluator.add_leaf(
        id="Observation_Deck_Floors_Match_Expected",
        desc="Observation deck floors equal 80th, 86th, and 102nd (exactly these three).",
        parent=group,
        critical=True,
    )
    exp_floors = GROUND_TRUTH["observation_deck_floors_expected"]
    extracted_floors_str = _normalize_floor_list_str(data.observation_deck_floors)
    claim_floors = f"The extracted observation deck floors '{extracted_floors_str}' denote exactly these three floors: 80th, 86th, and 102nd (allowing equivalent phrasings like '80', '80th floor')."
    await evaluator.verify(
        claim=claim_floors,
        node=floors_match,
        additional_instruction="Do not allow any extra or missing floors beyond 80, 86, 102."
    )
    src_exist = evaluator.add_custom_node(
        result=len(data.observation_decks_sources) > 0,
        id="Observation_Decks_Sources_Provided",
        desc="Source URLs are provided for number and locations of observation decks.",
        parent=group,
        critical=True,
    )
    src_support = evaluator.add_leaf(
        id="Observation_Decks_Supported_By_Sources",
        desc="Number and locations of observation decks are supported by cited sources.",
        parent=group,
        critical=True,
    )
    claim_support = f"There are {data.total_observation_decks} observation decks located on {extracted_floors_str}."
    await evaluator.verify(
        claim=claim_support,
        node=src_support,
        sources=data.observation_decks_sources,
        additional_instruction="Confirm the page explicitly states the number of decks and their floors."
    )


def _compute_all_citations_present(data: ESBExtraction) -> bool:
    checks = [
        len(data.total_height_with_antenna_sources) > 0,
        len(data.roof_height_sources) > 0,
        len(data.total_stories_sources) > 0,
        len(data.primary_observation_deck_floor_sources) > 0,
        len(data.primary_observation_deck_height_sources) > 0,
        len(data.top_observation_deck_floor_sources) > 0,
        len(data.observation_decks_sources) > 0,
    ]
    return all(checks)


async def _add_citations_overall_check(evaluator: Evaluator, parent, data: ESBExtraction) -> None:
    evaluator.add_custom_node(
        result=_compute_all_citations_present(data),
        id="Citations_For_Each_Attribute",
        desc="Includes verifiable reference URL(s) supporting each required attribute provided.",
        parent=parent,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    # Initialize evaluator with a parallel root
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

    # Extract structured information from the answer
    data: ESBExtraction = await evaluator.extract(
        prompt=prompt_extract_esb_info(),
        template_class=ESBExtraction,
        extraction_name="esb_extraction",
    )

    # Add ground truth info for transparency
    evaluator.add_ground_truth({
        "expected_values": GROUND_TRUTH,
        "notes": "Canonical values per rubric; leaves allow minor formatting variations but enforce exact numeric feet values for core heights."
    })

    # Build the critical verification group
    specs_node = evaluator.add_parallel(
        id="Empire_State_Building_Specifications",
        desc="Verify all required Empire State Building specifications are present, accurate, and properly cited per the stated constraints.",
        parent=root,
        critical=True,
    )

    # 0) Correct entity and location
    await _add_entity_check(evaluator, specs_node, data)

    # 1) Total height with antenna
    await _add_height_with_antenna_checks(evaluator, specs_node, data)

    # 2) Roof height
    await _add_roof_height_checks(evaluator, specs_node, data)

    # 3) Total number of stories
    await _add_total_stories_checks(evaluator, specs_node, data)

    # 4) Primary observation deck floor
    await _add_primary_deck_floor_checks(evaluator, specs_node, data)

    # 5) Primary observation deck height
    await _add_primary_deck_height_checks(evaluator, specs_node, data)

    # 6) Top observation deck floor
    await _add_top_deck_floor_checks(evaluator, specs_node, data)

    # 7) Number and locations of observation decks
    await _add_num_and_locations_decks_checks(evaluator, specs_node, data)

    # 8) Overall citations-for-each-attribute check
    await _add_citations_overall_check(evaluator, specs_node, data)

    # Return the structured evaluation summary
    return evaluator.get_summary()