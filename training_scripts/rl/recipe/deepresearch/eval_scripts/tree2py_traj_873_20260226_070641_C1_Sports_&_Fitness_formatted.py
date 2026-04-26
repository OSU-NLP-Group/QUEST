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
TASK_ID = "super_bowl_lx_capacity"
TASK_DESCRIPTION = "What is the standard seating capacity of the stadium that will host Super Bowl LX in February 2026?"

HOST_STADIUM = "Levi's Stadium"
ACCEPTABLE_STANDARD_CAPACITIES = ["68,500", "68500", "70,000", "70000"]  # textual variants


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueCapacityExtraction(BaseModel):
    """
    Structured fields extracted from the agent's answer.
    """
    stadium: Optional[str] = None
    capacity_text: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_capacity() -> str:
    return """
    From the answer text:
    - stadium: Extract the name of the stadium that the answer claims will host Super Bowl LX (2026). Return just the stadium name (e.g., "Levi's Stadium"). If not explicitly stated, return null.
    - capacity_text: Extract the seating capacity figure exactly as stated in the answer for that stadium, focusing on the standard (non-expanded) seating capacity. If multiple capacities are mentioned, prefer the standard capacity (e.g., 68,500 or 70,000 for Levi's Stadium). If no capacity is stated, return null. Keep the original formatting (e.g., "68,500", "70,000", "about 70,000").
    - source_urls: Extract any URLs present in the answer (if any). Return an array of the URLs. If none, return an empty array.

    Notes:
    - Do not invent values. Only extract what is explicitly present.
    - If both a standard capacity and an expanded/event capacity are mentioned, choose the standard capacity for capacity_text.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _capacity_verification_instruction() -> str:
    return (
        "Check the answer text itself (not external sources). Determine whether the answer provides the standard "
        "seating capacity for Levi's Stadium as either 68,500 or 70,000 seats. Accept minor textual variants such as "
        "'68500', '70,000', 'about 70,000', or 'around 70k', and allow commas or spacing differences. "
        "Do NOT accept only expanded/event capacity numbers (e.g., 75,000) unless the standard figure (68,500 or 70,000) "
        "is also clearly stated. If the answer does not include 68,500 or 70,000 (or a clear minor variant), judge it as incorrect."
    )


def _stadium_identification_instruction() -> str:
    return (
        "Verify the answer text itself (do not rely on external web evidence) identifies the host stadium for Super Bowl LX "
        "as Levi's Stadium. Minor naming variants (e.g., spacing, apostrophe style) are acceptable. "
        "Mention of the city ('Santa Clara, California') or the exact date ('February 8, 2026') is not required for this check; "
        "focus on whether the answer names Levi's Stadium as the venue for Super Bowl LX (2026)."
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
    Evaluate an answer for the Super Bowl LX host venue capacity task.
    """
    # 1) Initialize evaluator and root
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

    # 2) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_capacity(),
        template_class=VenueCapacityExtraction,
        extraction_name="venue_capacity_extraction",
    )

    # 3) Add ground truth info for transparency
    evaluator.add_ground_truth(
        {
            "host_stadium_expected": HOST_STADIUM,
            "acceptable_standard_capacities": ACCEPTABLE_STANDARD_CAPACITIES,
            "notes": "Either 68,500 or 70,000 is acceptable as the standard seating capacity for Levi's Stadium.",
        },
        gt_type="ground_truth",
    )

    # 4) Build verification tree according to rubric
    top_node = evaluator.add_parallel(
        id="Super_Bowl_LX_Venue_Capacity",
        desc="The answer correctly identifies the stadium hosting Super Bowl LX in February 2026 and provides its standard seating capacity",
        parent=root,
        critical=False,
    )

    # Leaf A: Correct stadium identified (critical)
    stadium_node = evaluator.add_leaf(
        id="Correct_Stadium_Identified",
        desc="The answer identifies Levi's Stadium in Santa Clara, California as the venue for Super Bowl LX (February 8, 2026)",
        parent=top_node,
        critical=True,
    )
    stadium_claim = (
        "The answer identifies Levi's Stadium as the stadium that will host Super Bowl LX (2026). "
        f"Extracted stadium (if any): '{extracted.stadium}'."
    )

    # Leaf B: Correct capacity provided (critical)
    capacity_node = evaluator.add_leaf(
        id="Correct_Capacity_Provided",
        desc="The answer provides the standard seating capacity as either 68,500 or 70,000 seats (both figures appear in official sources and are acceptable)",
        parent=top_node,
        critical=True,
    )
    capacity_claim = (
        "The answer provides the standard seating capacity for Levi's Stadium as either 68,500 or 70,000 seats. "
        f"Extracted capacity text (if any): '{extracted.capacity_text}'."
    )

    # 5) Run verifications (simple checks against the answer text)
    await evaluator.batch_verify(
        [
            (
                stadium_claim,
                None,  # No external URLs; this is a content-compliance check against the answer
                stadium_node,
                _stadium_identification_instruction(),
            ),
            (
                capacity_claim,
                None,  # No external URLs; this is a content-compliance check against the answer
                capacity_node,
                _capacity_verification_instruction(),
            ),
        ]
    )

    # 6) Return result summary
    return evaluator.get_summary()