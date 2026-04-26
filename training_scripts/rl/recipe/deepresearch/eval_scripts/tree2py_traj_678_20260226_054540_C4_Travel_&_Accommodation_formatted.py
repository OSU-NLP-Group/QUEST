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
TASK_ID = "old_town_hotel"
TASK_DESCRIPTION = (
    "Identify a hotel located in the Old Town San Diego area that satisfies ALL of the following requirements: "
    "(1) The hotel must be within 0.5 miles walking distance of Old Town San Diego State Historic Park, "
    "(2) The hotel must offer complimentary self-parking to guests, "
    "(3) The hotel must offer complimentary continental breakfast, "
    "(4) Provide the specific name of the hotel, and "
    "(5) Provide a reference URL that verifies these amenities and location details."
)

PARK_OFFICIAL_NAME = "Old Town San Diego State Historic Park"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelSelectionExtraction(BaseModel):
    """
    Extracted information about the selected hotel from the agent's answer.
    """
    hotel_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_info() -> str:
    return """
    Extract the hotel information the answer identifies as meeting all requirements.
    You must return:
    - hotel_name: The specific name of the hotel that the answer ultimately recommends.
    - reference_urls: An array of all URLs explicitly cited in the answer that are intended to verify the hotel's location or amenities.
    
    Rules:
    - If multiple hotels are mentioned, pick the single hotel the answer ultimately recommends. If unclear, choose the first one presented as meeting the criteria.
    - Only include valid, complete URLs explicitly present in the answer text (plain links or markdown links).
    - Do not invent or infer any URLs.
    - If hotel_name is not mentioned, set it to null.
    - If no reference URLs are provided, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_hotel_selection(
    evaluator: Evaluator,
    parent_node,
    extracted: HotelSelectionExtraction,
) -> None:
    """
    Build and execute the verification tree for the hotel selection.
    """
    # Create the critical Hotel_Selection node (parallel aggregation over independent requirements)
    hotel_sel_node = evaluator.add_parallel(
        id="Hotel_Selection",
        desc="Find a hotel in Old Town San Diego that satisfies all specified amenity and location requirements",
        parent=parent_node,
        critical=True,
    )

    # Critical existence checks (gate subsequent verifications through automatic preconditions)
    name_provided = bool(extracted.hotel_name and extracted.hotel_name.strip())
    urls_provided = bool(extracted.reference_urls and len(extracted.reference_urls) > 0)

    evaluator.add_custom_node(
        result=name_provided,
        id="Hotel_Name_Provided",
        desc="Provide the specific name of the hotel that meets all requirements",
        parent=hotel_sel_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=urls_provided,
        id="Reference_URL",
        desc="Provide a reference URL that verifies the hotel's amenities and location",
        parent=hotel_sel_node,
        critical=True,
    )

    # Prepare claims for verification leaves
    hotel_name = extracted.hotel_name or ""

    # 1) Location in Old Town San Diego
    node_loc = evaluator.add_leaf(
        id="Location_in_Old_Town",
        desc="The hotel must be located in the Old Town San Diego area",
        parent=hotel_sel_node,
        critical=True,
    )
    claim_loc = f"The hotel '{hotel_name}' is located in the Old Town San Diego area."
    add_ins_loc = (
        "Confirm that the referenced page(s) explicitly indicate the hotel is in the Old Town San Diego area—"
        "e.g., phrases like 'Old Town', 'Old Town San Diego', 'Old Town district'. "
        "An address or location description that clearly places the hotel within Old Town is acceptable. "
        "Generic 'San Diego' without 'Old Town' is not sufficient."
    )

    # 2) Walking distance within 0.5 miles to Old Town San Diego State Historic Park
    node_dist = evaluator.add_leaf(
        id="Walking_Distance_to_Park",
        desc=f"The hotel must be within 0.5 miles (walking distance) of {PARK_OFFICIAL_NAME}",
        parent=hotel_sel_node,
        critical=True,
    )
    claim_dist = (
        f"The hotel '{hotel_name}' is within 0.5 miles walking distance of {PARK_OFFICIAL_NAME}."
    )
    add_ins_dist = (
        f"Verify that the referenced page(s) explicitly support a walking distance of 0.5 miles or less to '{PARK_OFFICIAL_NAME}'. "
        "Accept statements like '0.5 miles', '0.4 mi', 'half-mile', or equivalent. "
        "If only drive times are shown or if distance is greater than 0.5 miles, treat as not supported. "
        "Minor wording variations of the park name (e.g., 'Old Town State Historic Park') are acceptable."
    )

    # 3) Complimentary self-parking
    node_parking = evaluator.add_leaf(
        id="Free_Parking",
        desc="The hotel must offer complimentary self-parking to guests",
        parent=hotel_sel_node,
        critical=True,
    )
    claim_parking = f"The hotel '{hotel_name}' offers complimentary self-parking to guests."
    add_ins_parking = (
        "Confirm the amenity is 'complimentary self-parking' or equivalent wording such as 'free parking', 'complimentary parking', "
        "'free self-parking'. If the parking is paid, valet-only, or only available via a special package, treat it as not supported."
    )

    # 4) Complimentary continental breakfast
    node_breakfast = evaluator.add_leaf(
        id="Free_Breakfast",
        desc="The hotel must offer complimentary continental breakfast",
        parent=hotel_sel_node,
        critical=True,
    )
    claim_breakfast = f"The hotel '{hotel_name}' offers complimentary continental breakfast."
    add_ins_breakfast = (
        "Look for explicit mention of 'continental breakfast' being complimentary/free. "
        "Phrases like 'complimentary breakfast' are acceptable only if 'continental' is clearly indicated. "
        "If the page only mentions 'hot breakfast' without 'continental', treat as not supported."
    )

    # Execute all URL-grounded verifications concurrently
    # Automatic preconditions will skip these if name/urls checks fail.
    await evaluator.batch_verify(
        [
            (claim_loc, extracted.reference_urls, node_loc, add_ins_loc),
            (claim_dist, extracted.reference_urls, node_dist, add_ins_dist),
            (claim_parking, extracted.reference_urls, node_parking, add_ins_parking),
            (claim_breakfast, extracted.reference_urls, node_breakfast, add_ins_breakfast),
        ]
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
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point to evaluate an agent's answer for the Old Town San Diego hotel selection task.
    """
    # Initialize evaluator
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

    # Extract hotel information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotel_info(),
        template_class=HotelSelectionExtraction,
        extraction_name="hotel_selection_extraction",
    )

    # Build verification tree and perform checks
    await verify_hotel_selection(evaluator, root, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()