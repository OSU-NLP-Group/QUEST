import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "statue_crown_advance_booking"
TASK_DESCRIPTION = (
    "How far in advance can tickets for Crown access to the Statue of Liberty be purchased? "
    "Please provide the maximum advance booking window and include a reference URL from an official source."
)

# Ground truth target (for transparency in summary only; verification is evidence-based)
GROUND_TRUTH_TARGET = "Up to 6 months (approximately 180 days) in advance"


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class CrownBookingExtraction(BaseModel):
    """
    Information we need from the agent's answer.
    """
    stated_max_window_text: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_crown_booking() -> str:
    return """
    Extract from the answer the information related to the maximum advance booking window for Statue of Liberty Crown tickets and any cited reference URLs.

    Required fields:
    - stated_max_window_text: A short quote or phrase as it appears in the answer that describes how far in advance Crown tickets can be purchased (e.g., "up to 6 months in advance", "180 days in advance", "six months ahead"). If no such statement is present, return null.
    - reference_urls: An array of all URLs cited as sources or references in the answer (including inline links or a sources section). Extract the actual URLs (handle markdown links). Only include valid URLs. If no URLs are present, return an empty array.

    Do not infer or invent any URLs or statements that are not explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification Logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_crown_booking_window(
    evaluator: Evaluator,
    parent_node,
    extracted: CrownBookingExtraction,
) -> None:
    """
    Build the verification tree and perform checks according to the rubric.
    """
    # Create the critical parallel parent node as specified by rubric
    main_node = evaluator.add_parallel(
        id="Crown_Ticket_Advance_Booking_Window",
        desc="Verify the advance booking window for Statue of Liberty Crown tickets",
        parent=parent_node,
        critical=True
    )

    # 1) Booking_Time_Period (critical leaf)
    # Verify that the answer explicitly states that Crown tickets can be purchased up to 6 months (or ~180 days) in advance.
    booking_time_leaf = evaluator.add_leaf(
        id="Booking_Time_Period",
        desc="The answer states that Crown tickets can be purchased up to 6 months (or 180 days) in advance",
        parent=main_node,
        critical=True
    )

    booking_claim = (
        "The answer explicitly states that Statue of Liberty Crown tickets can be purchased up to six months "
        "(about 180 days) in advance."
    )
    await evaluator.verify(
        claim=booking_claim,
        node=booking_time_leaf,
        additional_instruction=(
            "Judge by reading the answer text. Accept reasonable variants like 'up to 6 months', "
            "'six months in advance', '180 days in advance', 'half a year in advance', or similar expressions. "
            "If the answer states a different maximum window (e.g., 4 or 5 months) or does not clearly state 6 months "
            "or ~180 days, mark this as Incorrect."
        )
    )

    # 2) Reference_URL (critical leaf)
    # Must provide a valid official reference URL (NPS or Statue City Cruises) that supports the 6-month window.
    urls = extracted.reference_urls or []

    if not urls:
        # No URLs provided -> immediate failure for this critical leaf
        evaluator.add_custom_node(
            result=False,
            id="Reference_URL",
            desc="A valid reference URL from an official source (NPS or Statue City Cruises) is provided to support the booking window information",
            parent=main_node,
            critical=True
        )
    else:
        reference_leaf = evaluator.add_leaf(
            id="Reference_URL",
            desc="A valid reference URL from an official source (NPS or Statue City Cruises) is provided to support the booking window information",
            parent=main_node,
            critical=True
        )

        # Claim to verify against the provided URLs
        # The multi-URL verification will pass if at least one URL satisfies the claim
        reference_claim = (
            "This webpage is an official page from the National Park Service (nps.gov) or Statue City Cruises "
            "(statuecitycruises.com) and it explicitly states that Statue of Liberty Crown tickets can be "
            "purchased up to six months (approximately 180 days) in advance."
        )
        await evaluator.verify(
            claim=reference_claim,
            node=reference_leaf,
            sources=urls,
            additional_instruction=(
                "Only mark as Supported if BOTH conditions are met for this specific URL: "
                "1) It is an official source page: its domain is nps.gov or statuecitycruises.com (do not accept other domains). "
                "2) The page clearly states or strongly implies that Crown tickets can be reserved up to 6 months "
                "(around 180 days) in advance. If either condition fails, mark as Not Supported for this URL. "
                "Remember, the verification passes overall if at least one of the provided URLs meets both criteria."
            )
        )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point                                                 #
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
    Evaluate an answer for the Statue of Liberty Crown ticket advance booking window task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator; specific rubric node will be added beneath
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_crown_booking(),
        template_class=CrownBookingExtraction,
        extraction_name="crown_booking_extraction"
    )

    # Add transparency info (non-scoring)
    evaluator.add_ground_truth(
        {
            "target_window": GROUND_TRUTH_TARGET,
            "official_sources_required": ["nps.gov", "statuecitycruises.com"]
        },
        gt_type="ground_truth"
    )
    evaluator.add_custom_info(
        info={"extracted_reference_urls": extracted.reference_urls},
        info_type="extraction_details",
        info_name="extracted_urls"
    )
    evaluator.add_custom_info(
        info={"stated_max_window_text": extracted.stated_max_window_text},
        info_type="extraction_details",
        info_name="stated_window_text"
    )

    # Build verification tree and run checks
    await verify_crown_booking_window(evaluator, root, extracted)

    # Return structured result
    return evaluator.get_summary()