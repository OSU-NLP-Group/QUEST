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
TASK_ID = "houston_rodeo_2026_march12_performer"
TASK_DESCRIPTION = "Who is the performer scheduled to perform at the Houston Rodeo on March 12, 2026, at 6:45 PM at NRG Stadium in Houston, Texas?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PerformerExtraction(BaseModel):
    performer: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_performer() -> str:
    return """
    From the answer, extract:
    - performer: the single performer name that the answer claims is scheduled for the specified event/time (Houston Rodeo on March 12, 2026 at 6:45 PM at NRG Stadium). Return exactly the name string as written in the answer. If multiple performers are mentioned, choose the one the answer most clearly asserts for that specific date/time; otherwise choose the first such performer mentioned. If no performer is explicitly claimed, return null.
    - source_urls: an array of all explicit URLs cited in the answer that purportedly support this claim (including markdown links). Only include URLs that appear in the answer. Do not fabricate any URLs.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree_and_verify(
    evaluator: Evaluator,
    extracted: PerformerExtraction,
) -> None:
    """
    Build the verification tree based on the rubric and run verifications.
    """

    # Create the rubric's main node under root (critical, parallel)
    performer_node = evaluator.add_parallel(
        id="PerformerIdentification",
        desc="Correctly identify the performer scheduled at Houston Rodeo on March 12, 2026 at 6:45 PM at NRG Stadium",
        parent=evaluator.root,
        critical=True,
    )

    # Existence checks (critical) to gate subsequent verification
    name_provided_node = evaluator.add_custom_node(
        result=bool(extracted.performer and extracted.performer.strip()),
        id="PerformerNameProvided",
        desc="The performer name is provided in the answer",
        parent=performer_node,
        critical=True
    )
    sources_provided_node = evaluator.add_custom_node(
        result=bool(extracted.source_urls and len(extracted.source_urls) > 0),
        id="SourcesProvided",
        desc="At least one source URL is provided",
        parent=performer_node,
        critical=True
    )

    # Prepare variables
    performer = extracted.performer or ""
    urls = extracted.source_urls

    # EventAffiliation leaf
    event_node = evaluator.add_leaf(
        id="EventAffiliation",
        desc="The performance is part of the Houston Rodeo 2026 event",
        parent=performer_node,
        critical=True
    )
    event_claim = (
        f"{performer} is scheduled to perform at the Houston Livestock Show and Rodeo "
        f"(also known as RodeoHouston) in 2026."
    )
    await evaluator.verify(
        claim=event_claim,
        node=event_node,
        sources=urls,
        additional_instruction=(
            "Confirm that the page explicitly associates the performer with RodeoHouston (Houston Livestock Show and Rodeo) "
            "for the 2026 season. Accept reasonable variants such as 'RodeoHouston' or 'Houston Livestock Show and Rodeo'."
        ),
    )

    # PerformanceDate leaf
    date_node = evaluator.add_leaf(
        id="PerformanceDate",
        desc="The performer is scheduled on March 12, 2026",
        parent=performer_node,
        critical=True
    )
    date_claim = f"{performer} is scheduled to perform on March 12, 2026."
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=urls,
        additional_instruction=(
            "Verify that the date is explicitly March 12, 2026 (e.g., 'Thu, Mar 12, 2026' or similar formatting) for this performer."
        ),
    )

    # PerformanceTime leaf
    time_node = evaluator.add_leaf(
        id="PerformanceTime",
        desc="The performance starts at 6:45 PM",
        parent=performer_node,
        critical=True
    )
    time_claim = f"The start time for {performer}'s performance is 6:45 PM."
    await evaluator.verify(
        claim=time_claim,
        node=time_node,
        sources=urls,
        additional_instruction=(
            "Check that the page states the performance start time as 6:45 PM. "
            "Allow reasonable formatting variants like '6:45 pm', '6:45 p.m.', or inclusion of time zone (e.g., CT). "
            "Assume times are local to Houston (Central Time) unless otherwise specified."
        ),
    )

    # PerformanceVenue leaf
    venue_node = evaluator.add_leaf(
        id="PerformanceVenue",
        desc="The performance is at NRG Stadium in Houston, Texas",
        parent=performer_node,
        critical=True
    )
    venue_claim = f"The venue for {performer}'s performance is NRG Stadium in Houston, Texas."
    await evaluator.verify(
        claim=venue_claim,
        node=venue_node,
        sources=urls,
        additional_instruction=(
            "Confirm that the venue is explicitly 'NRG Stadium' and that it is in Houston, Texas. "
            "Accept reasonable variants like 'NRG Stadium, Houston TX'."
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
    Evaluate an answer for the Houston Rodeo performer on March 12, 2026 at 6:45 PM task.
    """
    # Initialize evaluator
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

    # Extract performer and sources from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_performer(),
        template_class=PerformerExtraction,
        extraction_name="performer_extraction",
    )

    # Build verification tree and run verification
    await build_verification_tree_and_verify(evaluator, extracted)

    # Return structured summary
    return evaluator.get_summary()