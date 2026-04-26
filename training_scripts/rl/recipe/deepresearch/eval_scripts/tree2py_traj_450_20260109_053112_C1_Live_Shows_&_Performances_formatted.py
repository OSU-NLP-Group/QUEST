import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "billboard_boxscore_2025_top_venues_15001_plus"
TASK_DESCRIPTION = (
    "According to Billboard's 2025 year-end Boxscore rankings (covering shows from October 1, 2024, "
    "to September 30, 2025), what is the #1 highest-grossing venue in the Top Venues (15,001+ capacity) "
    "category, where is it located, and what is its seating capacity?"
)

# Ground truth constraints derived from the rubric
GROUND_TRUTH = {
    "expected_location": "Las Vegas",
    "expected_capacity": "18,600",
    "category": "Top Venues (15,001+ capacity)",
    "timeframe": "Oct 1, 2024 to Sep 30, 2025",
    "year_end": "Billboard 2025 Year-End Boxscore"
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """
    Extracted information from the agent's answer.
    - venue_name: The name of the #1 ranked venue for the specified Billboard category and timeframe.
    - location: The venue location string as stated in the answer (e.g., "Las Vegas, NV").
    - seating_capacity: The seating capacity string as stated in the answer (e.g., "18,600").
    - sources: All URLs the answer cites to support the ranking, location, or capacity claims.
    """
    venue_name: Optional[str] = None
    location: Optional[str] = None
    seating_capacity: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    From the provided answer, extract the information specifically about Billboard's 2025 year-end Boxscore rankings
    for the "Top Venues (15,001+ capacity)" category covering shows from Oct 1, 2024 to Sep 30, 2025.

    You must extract:
    1) venue_name: The name of the venue that the answer claims is ranked #1 (highest-grossing) in the specified category/timeframe.
    2) location: The venue's location as stated in the answer (e.g., "Las Vegas", "Las Vegas, NV", or "Las Vegas, Nevada").
    3) seating_capacity: The seating capacity value as stated in the answer (e.g., "18,600"). Preserve the formatting from the answer (commas, spaces).
    4) sources: An array of all URLs mentioned in the answer that purport to support this ranking, venue details, location, or capacity.
       Only include actual URLs (plain or within markdown); do not invent or infer any URLs.

    If multiple venues are mentioned, choose the one the answer presents as #1.
    If any item is missing in the answer, set it to null. For sources, return an empty list if none are present.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    parent_node,
    extracted: VenueExtraction
) -> None:
    """
    Build and execute the verification tree according to the rubric.

    The rubric has one critical sequential node "answer_correctness" with three critical leaf checks:
    1) identify_number_1_venue
    2) provide_constrained_location
    3) provide_constrained_seating_capacity
    """
    # Create the critical sequential node for overall correctness
    ac_node = evaluator.add_sequential(
        id="answer_correctness",
        desc=(
            "Answer identifies the #1 highest-grossing venue on Billboard's 2025 year-end Boxscore "
            "Top Venues (15,001+ capacity) ranking (covering shows from Oct 1, 2024 to Sep 30, 2025) "
            "and provides the required location and seating capacity per the given constraints."
        ),
        parent=parent_node,
        critical=True
    )

    # 1) Identify the #1 venue node (critical leaf)
    identify_node = evaluator.add_leaf(
        id="identify_number_1_venue",
        desc=(
            "Names the venue ranked #1 on Billboard's 2025 year-end Boxscore 'Top Venues (15,001+ capacity)' "
            "list for the stated show-date window (Oct 1, 2024 to Sep 30, 2025)."
        ),
        parent=ac_node,
        critical=True
    )
    venue_name = (extracted.venue_name or "").strip()
    claim_identify = (
        f"According to Billboard's 2025 year-end Boxscore for '{GROUND_TRUTH['category']}' "
        f"(shows spanning {GROUND_TRUTH['timeframe']}), the #1 highest-grossing venue is '{venue_name}'."
    )
    await evaluator.verify(
        claim=claim_identify,
        node=identify_node,
        sources=extracted.sources,
        additional_instruction=(
            "Use the provided URLs (ideally a Billboard year-end Boxscore page or equivalent credible source) "
            "to confirm the #1 venue for the specified category and timeframe. "
            "Allow minor naming variations (e.g., inclusion or omission of sponsor/complex names). "
            "If URLs are missing or irrelevant, judge correctness based on the answer context but prefer explicit source support."
        ),
    )

    # 2) Provide constrained location node (critical leaf)
    location_node = evaluator.add_leaf(
        id="provide_constrained_location",
        desc="States the venue's location, and the stated location is Las Vegas (accepting unambiguous variants like 'Las Vegas, NV').",
        parent=ac_node,
        critical=True
    )
    location_val = (extracted.location or "").strip()
    claim_location = (
        f"The venue's location, as stated in the answer ('{location_val}'), is equivalent to Las Vegas, Nevada."
    )
    await evaluator.verify(
        claim=claim_location,
        node=location_node,
        sources=None,  # Check equivalence with what the answer states; do not require external sources
        additional_instruction=(
            "Judge whether the stated location is clearly 'Las Vegas'—accepting variants like 'Las Vegas', 'Las Vegas, NV', or "
            "'Las Vegas, Nevada'. Do not consider nearby municipalities (e.g., Paradise, NV) equivalent unless the answer explicitly "
            "frames them as Las Vegas in common usage. Focus on the answer's stated location and equivalence to 'Las Vegas'."
        ),
    )

    # 3) Provide constrained seating capacity node (critical leaf)
    capacity_node = evaluator.add_leaf(
        id="provide_constrained_seating_capacity",
        desc="States the venue's seating capacity, and the value is 18,600 (accepting equivalent numeric formatting).",
        parent=ac_node,
        critical=True
    )
    capacity_val = (extracted.seating_capacity or "").strip()
    claim_capacity = (
        f"The stated seating capacity ('{capacity_val}') is equivalent to 18,600."
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_node,
        sources=None,  # Validate against the answer content; equivalence in formatting
        additional_instruction=(
            "Accept equivalent numeric formatting for 18,600 (e.g., '18,600', '18600', '18 600'). "
            "Minor descriptors like 'approx.' are acceptable if they clearly refer to 18,600. "
            "Focus on whether the answer's stated capacity amounts to 18,600."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate the agent's answer against the Billboard 2025 year-end Boxscore Top Venues (15,001+ capacity) rubric.

    Returns a structured summary including the verification tree and final score.
    """
    # Initialize evaluator
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
        default_model=model
    )

    # Extract structured venue info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_info"
    )

    # Add ground truth constraints to summary (for transparency)
    evaluator.add_ground_truth({
        "category": GROUND_TRUTH["category"],
        "timeframe": GROUND_TRUTH["timeframe"],
        "expected_location": GROUND_TRUTH["expected_location"],
        "expected_capacity": GROUND_TRUTH["expected_capacity"],
        "year_end": GROUND_TRUTH["year_end"]
    }, gt_type="constraints")

    # Build and run verification tree
    await build_verification_tree(evaluator, root, extracted)

    # Return structured result
    return evaluator.get_summary()