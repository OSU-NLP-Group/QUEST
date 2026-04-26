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
TASK_ID = "fredagain_foresthills_capacity_2023"
TASK_DESCRIPTION = (
    "In 2023, the British electronic music artist Fred Again performed three sold-out shows at a historic outdoor "
    "venue in Queens, New York. What is the seating capacity of this venue for concerts?"
)

EXPECTED_VENUE = "Forest Hills Stadium"
EXPECTED_CAPACITY_DISPLAY = "13,000"
EXPECTED_CAPACITY_NUMERIC = "13000"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueCapacityExtraction(BaseModel):
    """Information explicitly stated in the answer about the venue and capacity."""
    venue_name: Optional[str] = None
    capacity_value: Optional[str] = None  # Keep as free text, e.g., "13,000", "13k", "about 13,000"
    capacity_unit: Optional[str] = None   # e.g., "people", "seats"
    source_urls: List[str] = Field(default_factory=list)  # any URLs mentioned in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_capacity() -> str:
    return """
    Extract from the answer:
    - venue_name: The name of the venue identified for Fred Again's 2023 shows in Queens, NY.
    - capacity_value: The value the answer gives for the venue's concert capacity (keep exactly as written, e.g., "13,000", "13k", "about 13,000").
    - capacity_unit: The unit used with the capacity if present (e.g., "people", "seats", "attendees"); otherwise null.
    - source_urls: All URLs explicitly shown in the answer (if any).
    If any field is missing, return null (or empty array for source_urls).
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, extracted: VenueCapacityExtraction) -> None:
    """
    Build the verification tree based on the rubric and run the checks.
    """

    # Top-level parallel node (critical) as per rubric: "venue_capacity_answer"
    venue_node = evaluator.add_parallel(
        id="venue_capacity_answer",
        desc="Answer identifies the intended venue and provides its concert capacity per the given constraints.",
        parent=evaluator.root,
        critical=True
    )

    # Leaf 1: Identify the venue as Forest Hills Stadium (Queens, NYC)
    identify_leaf = evaluator.add_leaf(
        id="identify_venue",
        desc="Identifies the venue as Forest Hills Stadium (Queens, New York City).",
        parent=venue_node,
        critical=True
    )

    # Construct claim using extracted info if available
    if extracted and extracted.venue_name:
        identify_claim = (
            f"The identified venue in the answer is '{extracted.venue_name}', and this refers to "
            f"'{EXPECTED_VENUE}' (a historic outdoor venue in Queens, New York City). They refer to the same place."
        )
    else:
        identify_claim = (
            f"The answer identifies the venue as '{EXPECTED_VENUE}' (a historic outdoor venue in Queens, New York City)."
        )

    await evaluator.verify(
        claim=identify_claim,
        node=identify_leaf,
        additional_instruction=(
            "Judge solely from the answer text. Approve if the venue is clearly Forest Hills Stadium in Queens, NYC. "
            "Allow reasonable naming variants such as 'Forest Hills Stadium at the West Side Tennis Club', "
            "'Forest Hills Stadium (Queens)', or similar phrasing."
        )
    )

    # Leaf 2: State the concert capacity as 13,000
    capacity_leaf = evaluator.add_leaf(
        id="state_concert_capacity",
        desc="States the venue's concert capacity as 13,000 people/seats (i.e., 13,000 for concerts).",
        parent=venue_node,
        critical=True
    )

    # Construct claim using extracted info if available
    if extracted and extracted.capacity_value:
        capacity_claim = (
            f"The answer states the venue's concert capacity for concerts as {EXPECTED_CAPACITY_DISPLAY} "
            f"(approximately {EXPECTED_CAPACITY_NUMERIC}). The extracted capacity string from the answer is "
            f"'{extracted.capacity_value}', and it conveys the same quantity."
        )
    else:
        capacity_claim = (
            f"The answer states that the venue's concert capacity for concerts is {EXPECTED_CAPACITY_DISPLAY}."
        )

    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        additional_instruction=(
            "Judge solely from the answer text. Approve if the answer specifies 13,000 for concerts, including variants "
            "like '13k', '13,000 people', 'about 13,000', or 'up to 13,000 for concerts'. "
            "Reject if the answer gives a different number or only mentions a different context (e.g., tennis capacity)."
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
    Evaluate an answer for the Forest Hills Stadium concert capacity task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root strategy (wrapper); rubric root is the child node below
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_capacity(),
        template_class=VenueCapacityExtraction,
        extraction_name="venue_capacity_extraction"
    )

    # Ground truth reference (for reporting)
    evaluator.add_ground_truth({
        "expected_venue": EXPECTED_VENUE,
        "expected_concert_capacity": EXPECTED_CAPACITY_DISPLAY
    }, gt_type="ground_truth")

    # Build verification tree and run checks
    await build_and_verify(evaluator, extracted)

    # Return unified summary
    return evaluator.get_summary()