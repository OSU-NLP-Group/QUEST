import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_4nm_q4_2024_facility"
TASK_DESCRIPTION = (
    "Which semiconductor manufacturing facility in the United States started high-volume production of "
    "4-nanometer process technology in Q4 2024? Provide the facility name and its location (city and state)."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacilityExtraction(BaseModel):
    """
    Extracted facility identification and attribution from the agent's answer.
    """
    facility_name: Optional[str] = None
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_facility() -> str:
    return (
        "From the provided answer, extract the following fields:\n"
        "1. facility_name: The specific semiconductor manufacturing facility name (e.g., 'TSMC Arizona Fab 21'). "
        "   Do not just return a company name. If only a company name is given in the answer and no specific facility "
        "   is named, return null.\n"
        "2. location_city: The city of the facility's location as stated in the answer. If not provided, return null.\n"
        "3. location_state: The U.S. state of the facility's location as stated in the answer. This may be the full "
        "   state name (e.g., 'Arizona') or the postal abbreviation (e.g., 'AZ'). If not provided, return null.\n"
        "4. sources: A list of all URLs explicitly mentioned in the answer (including markdown links). "
        "   Only include valid URLs; if none are provided, return an empty list.\n"
        "Return a single JSON object with these fields. Do not infer or add information not present in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def format_location(city: Optional[str], state: Optional[str]) -> str:
    """
    Construct a 'City, State' string if available; fallback to empty string.
    """
    city = (city or "").strip()
    state = (state or "").strip()
    if city and state:
        return f"{city}, {state}"
    elif city:
        return city
    elif state:
        return state
    return ""


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extraction: FacilityExtraction,
) -> None:
    """
    Build the verification nodes based on the rubric and run verifications.
    """
    # Create the critical parent node for this task
    parent_node = evaluator.add_parallel(
        id="Semiconductor_Facility_Identification",
        desc=(
            "Check whether the answer identifies the correct U.S. semiconductor manufacturing facility and provides "
            "the required name and location, consistent with the Q4 2024 high-volume 4nm production constraint."
        ),
        parent=evaluator.root,
        critical=True,  # Critical parent: all children beneath must be critical
    )

    # 1) Facility_Name_Included (existence of a specific facility name)
    has_facility_name = bool(extraction.facility_name and extraction.facility_name.strip())
    evaluator.add_custom_node(
        result=has_facility_name,
        id="Facility_Name_Included",
        desc=(
            "Answer includes a facility name (not just a company name) identifying a specific semiconductor "
            "manufacturing facility."
        ),
        parent=parent_node,
        critical=True,
    )

    # 2) Location_Included_City_State_US (presence and US location)
    location_leaf = evaluator.add_leaf(
        id="Location_Included_City_State_US",
        desc=(
            "Answer includes the facility location with both city and state, and the stated location is in the United States."
        ),
        parent=parent_node,
        critical=True,
    )
    loc_str = format_location(extraction.location_city, extraction.location_state)
    location_claim = (
        f"The answer includes both the city and state for the facility location — specifically '{loc_str}' — "
        f"and this location is in the United States."
    )
    await evaluator.verify(
        claim=location_claim,
        node=location_leaf,
        additional_instruction=(
            "Verify that the answer text contains both a city and a state for the facility location. "
            "Accept either full state names (e.g., 'Arizona') or postal abbreviations (e.g., 'AZ'). "
            "Also confirm that the stated city/state combination corresponds to a location in the United States."
        ),
    )

    # Prepare sources for evidence-based checks
    source_urls: List[str] = extraction.sources or []

    # 3) Facility_Is_Semiconductor_Fab (facility type is a wafer fab / manufacturing plant)
    fab_leaf = evaluator.add_leaf(
        id="Facility_Is_Semiconductor_Fab",
        desc=(
            "The named facility is in fact a semiconductor manufacturing facility (fab), not merely an office/R&D site/headquarters."
        ),
        parent=parent_node,
        critical=True,
    )
    fab_claim = (
        f"The named facility '{extraction.facility_name or ''}' is a semiconductor wafer fabrication plant or "
        f"manufacturing facility (a 'fab'), not just an office, R&D center, or headquarters site."
    )
    await evaluator.verify(
        claim=fab_claim,
        node=fab_leaf,
        sources=source_urls,
        additional_instruction=(
            "Examine the cited webpages to confirm the facility is a wafer fabrication plant or semiconductor "
            "manufacturing site. Look for terms like 'fab', 'wafer fab', 'semiconductor manufacturing plant', "
            "'production line', or similar indications of manufacturing capability."
        ),
    )

    # 4) High_Volume_4nm_Production (started HVM of 4nm/N4 technology)
    hv_4nm_leaf = evaluator.add_leaf(
        id="High_Volume_4nm_Production",
        desc="The named facility started high-volume production of 4-nanometer (4nm/N4) process technology.",
        parent=parent_node,
        critical=True,
    )
    hv_4nm_claim = (
        f"The facility '{extraction.facility_name or ''}' started high-volume manufacturing (HVM / mass production) "
        f"of 4-nanometer (N4) process technology."
    )
    await evaluator.verify(
        claim=hv_4nm_claim,
        node=hv_4nm_leaf,
        sources=source_urls,
        additional_instruction=(
            "Verify that the evidence explicitly indicates high-volume or mass production (HVM) of 4nm/N4 at the facility. "
            "Accept synonyms like 'mass production', 'volume production', 'HVM'. Ensure the node size is clearly 4nm or N4."
        ),
    )

    # 5) Production_Start_In_Q4_2024 (timing verification)
    q4_leaf = evaluator.add_leaf(
        id="Production_Start_In_Q4_2024",
        desc="The start of high-volume 4nm production occurred in Q4 2024 (October–December 2024).",
        parent=parent_node,
        critical=True,
    )
    q4_claim = (
        f"The start of high-volume 4nm production at '{extraction.facility_name or ''}' occurred in Q4 2024 "
        f"(between October and December 2024)."
    )
    await evaluator.verify(
        claim=q4_claim,
        node=q4_leaf,
        sources=source_urls,
        additional_instruction=(
            "Confirm that the production start date is in Q4 2024. Accept explicit 'Q4 2024' or individual months "
            "October 2024, November 2024, or December 2024 as valid evidence."
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
) -> Dict:
    """
    Evaluate an agent's answer for identifying the U.S. semiconductor manufacturing facility
    that started high-volume 4nm production in Q4 2024.
    """
    # Initialize evaluator (root is non-critical by framework default)
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_facility(),
        template_class=FacilityExtraction,
        extraction_name="facility_extraction",
    )

    # Build and run verifications
    await build_verification_tree(evaluator, extraction)

    # Return summary
    return evaluator.get_summary()