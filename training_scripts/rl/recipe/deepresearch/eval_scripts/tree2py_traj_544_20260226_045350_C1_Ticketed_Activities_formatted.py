import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "la28_inglewood_largest_venue_capacity"
TASK_DESCRIPTION = "What is the seating capacity of the largest official Olympic venue in the Inglewood Zone for the LA28 Olympics?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class InglewoodLargestVenueExtraction(BaseModel):
    """
    Extracted info from the answer:
    - venue_name: The venue the answer identifies as the largest official LA28 Olympic venue in the Inglewood Zone
    - capacity: The seating capacity value provided for that venue
    - urls: All URLs explicitly mentioned in the answer (if any)
    """
    venue_name: Optional[str] = None
    capacity: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_inglewood_largest_venue() -> str:
    return """
    Extract from the answer the venue the author identifies as the largest official LA28 Olympic venue in the Inglewood Zone and the seating capacity they provide for that venue.

    Return the following fields:
    - venue_name: The venue name as written in the answer (e.g., "2028 Stadium", "SoFi Stadium", etc.).
    - capacity: The seating capacity value as written in the answer (e.g., "70,240", "70240", "70 240"). Keep it as a string exactly as shown.
    - urls: A list of all URLs explicitly mentioned anywhere in the answer (in any format such as plain URLs or markdown links).

    If any field is missing, set it to null (for strings) or an empty list (for urls).
    """


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
    Evaluate the answer for the LA28 Inglewood largest venue capacity task.
    Checks:
    - Venue identified by the answer matches "2028 Stadium" (also known as "SoFi Stadium").
    - Seating capacity provided is 70,240 (allowing minor formatting differences like commas or spaces).
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

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_inglewood_largest_venue(),
        template_class=InglewoodLargestVenueExtraction,
        extraction_name="inglewood_largest_venue_extraction"
    )

    # Add ground truth/reference info for transparency
    evaluator.add_ground_truth(
        {
            "expected_venue": "2028 Stadium (SoFi Stadium)",
            "expected_capacity": "70,240",
            "zone": "Inglewood Zone",
            "note": "2028 Stadium is LA28’s designation for SoFi Stadium; capacity for LA28 is 70,240."
        },
        gt_type="ground_truth"
    )

    # Build the rubric tree: critical parent with two critical leaf checks
    critical_group = evaluator.add_parallel(
        id="LA28_Inglewood_Largest_Venue",
        desc="Correctly identify the largest official LA28 Olympic venue in the Inglewood Zone and provide its seating capacity",
        parent=root,
        critical=True
    )

    # Leaf 1: Venue Identification
    venue_leaf = evaluator.add_leaf(
        id="Venue_Identification",
        desc="The venue identified is 2028 Stadium (also referred to as SoFi Stadium)",
        parent=critical_group,
        critical=True
    )

    extracted_venue = extracted.venue_name or ""
    venue_claim = (
        f"The answer identifies '{extracted_venue}' as the largest official LA28 Olympic venue in the Inglewood Zone, "
        f"and this venue is the same as '2028 Stadium' (also known as 'SoFi Stadium')."
    )
    await evaluator.verify(
        claim=venue_claim,
        node=venue_leaf,
        additional_instruction=(
            "Judge whether the answer’s named venue refers to the same venue as '2028 Stadium' / 'SoFi Stadium'. "
            "Allow minor variations in naming and casing (e.g., 'SoFi', 'SoFi Stadium, Inglewood', '2028 Stadium'). "
            "Only pass if the answer clearly identifies this venue as the largest official LA28 venue in the Inglewood Zone."
        )
    )

    # Leaf 2: Capacity Value
    capacity_leaf = evaluator.add_leaf(
        id="Capacity_Value",
        desc="The seating capacity provided is 70,240",
        parent=critical_group,
        critical=True
    )

    extracted_capacity = extracted.capacity or ""
    capacity_claim = (
        "In the answer, the seating capacity given for the largest official LA28 Olympic venue in the Inglewood Zone "
        "is 70,240 (treat '70,240', '70240', or '70 240' as equivalent)."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        additional_instruction=(
            f"Use the full answer text to decide if the stated capacity equals 70,240. "
            f"Ignore thousand separators and whitespace. The extracted capacity text is '{extracted_capacity}'. "
            f"Do not accept vague approximations like 'about 70k' unless it explicitly equals 70,240."
        )
    )

    return evaluator.get_summary()