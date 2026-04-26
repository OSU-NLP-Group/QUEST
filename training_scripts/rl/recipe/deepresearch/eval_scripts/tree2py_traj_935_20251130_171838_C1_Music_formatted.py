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
TASK_ID = "burna_atlanta_venue_opening_date"
TASK_DESCRIPTION = "What is the opening date of the concert venue where Burna Boy performed in Atlanta, Georgia during March 2024?"

MAIN_NODE_DESC = "Determine the opening date of the concert venue where Burna Boy performed in Atlanta, Georgia during March 2024"

GROUND_TRUTH = {
    "expected_venue": "State Farm Arena",
    "expected_opening_date": "September 18, 1999",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueOpeningExtraction(BaseModel):
    """
    Information extracted from the answer about the venue and opening date.
    """
    venue_name: Optional[str] = None
    opening_date: Optional[str] = None
    venue_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_opening_info() -> str:
    return """
    From the answer, extract the following details strictly as presented:
    1) venue_name: The name of the concert venue in Atlanta, Georgia where Burna Boy performed during March 2024.
    2) opening_date: The venue's opening date stated in the answer (as a string exactly as written in the answer).
    3) venue_urls: A list of all URLs explicitly provided in the answer that relate to the venue (e.g., official venue site, Wikipedia page for State Farm Arena/Philips Arena, event page, news article).
    
    Rules:
    - Extract only what is explicitly stated in the answer; do not invent or infer.
    - If a field is missing, set it to null (for venue_name/opening_date) or an empty list (for venue_urls).
    - For URLs, include only valid URLs explicitly present in the answer text (including markdown links).
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extraction: VenueOpeningExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """
    # Create the main sequential, critical node under root to mirror rubric
    main_node = evaluator.add_sequential(
        id="venue_opening_date",
        desc=MAIN_NODE_DESC,
        parent=evaluator.root,
        critical=True,  # Critical parent; all children must be critical
    )

    # Leaf 1: Venue identification check (critical)
    venue_leaf = evaluator.add_leaf(
        id="venue_is_state_farm_arena",
        desc="Answer identifies the relevant venue as State Farm Arena (the venue where Burna Boy performed in Atlanta during March 2024).",
        parent=main_node,
        critical=True,
    )

    venue_claim = (
        "The answer identifies the relevant concert venue as State Farm Arena in Atlanta, Georgia "
        "for Burna Boy's Atlanta performance context."
    )
    await evaluator.verify(
        claim=venue_claim,
        node=venue_leaf,
        additional_instruction=(
            "Judge based only on the content of the provided answer. Consider reasonable variants like "
            "'State Farm Arena (Atlanta)' or 'Atlanta’s State Farm Arena'. Do not require the answer to restate 'March 2024'; "
            "focus on whether State Farm Arena is identified as the relevant venue."
        ),
    )

    # Leaf 2: Opening date check (critical)
    opening_leaf = evaluator.add_leaf(
        id="opening_date_is_sep_18_1999",
        desc="Answer provides the venue's official opening date as September 18, 1999.",
        parent=main_node,
        critical=True,
    )

    opening_claim = (
        "The answer explicitly states the venue's official opening date as September 18, 1999."
    )
    await evaluator.verify(
        claim=opening_claim,
        node=opening_leaf,
        additional_instruction=(
            "Judge strictly by the answer text. Count as correct only if the answer provides the explicit date "
            "'September 18, 1999' (allow minor formatting variations such as 'Sept 18, 1999' or '1999-09-18'). "
            "If the answer only gives the year (e.g., '1999') or a different date, mark incorrect."
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
    Evaluate an answer for the venue opening date task.
    """
    # Initialize evaluator with a sequential root to match rubric flow
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract venue and opening-date related info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_venue_opening_info(),
        template_class=VenueOpeningExtraction,
        extraction_name="venue_opening_extraction",
    )

    # Record ground truth for transparency
    evaluator.add_ground_truth(
        {
            "expected_venue": GROUND_TRUTH["expected_venue"],
            "expected_opening_date": GROUND_TRUTH["expected_opening_date"],
        },
        gt_type="ground_truth",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extraction)

    # Return standard summary
    return evaluator.get_summary()