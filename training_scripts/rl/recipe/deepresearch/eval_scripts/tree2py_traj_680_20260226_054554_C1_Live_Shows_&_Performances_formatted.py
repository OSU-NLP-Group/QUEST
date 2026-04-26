import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "TrevorNoah_Nashville_Feb2026_VenueCapacity"
TASK_DESCRIPTION = """
What is the seating capacity of the venue where Trevor Noah is scheduled to perform in Nashville, Tennessee in February 2026?
We expect the answer to:
- Identify the venue (expected: Ryman Auditorium) with reliable event/ticketing or official announcement evidence that confirms: performer (Trevor Noah), location (Nashville, TN), and date (February 2026).
- Provide the venue’s official seating capacity with an acceptable source (official venue website preferred; reliable event/ticketing websites acceptable).
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AnswerExtraction(BaseModel):
    venue_name: Optional[str] = None
    event_source_urls: List[str] = Field(default_factory=list)
    capacity_value: Optional[str] = None
    capacity_source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_answer_info() -> str:
    return """
    Extract the following fields from the answer text:

    1) venue_name: The explicitly named venue where Trevor Noah is scheduled to perform in Nashville, Tennessee in February 2026.
    2) event_source_urls: A list of URL(s) that the answer cites to support the event info (performer Trevor Noah, location Nashville, Tennessee, and date in February 2026). These URLs should be event/ticketing listings or official announcements (e.g., venue or official artist/event pages). Return all such URLs mentioned in the answer. If none are provided, return an empty list.
    3) capacity_value: The numeric seating capacity stated in the answer for the identified venue. Extract it exactly as written in the answer (e.g., "2,362" or "2362"). If none is provided, return null.
    4) capacity_source_urls: A list of URL(s) that the answer cites to support the capacity figure. Prefer official venue pages; reliable ticketing/event sites are acceptable too. Return all such URLs mentioned in the answer for the capacity claim. If none are provided, return an empty list.

    Follow URL extraction rules strictly: extract only URLs explicitly mentioned (plain or in markdown). Do not invent URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def looks_numeric_capacity(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept if the string contains a number with at least 3 digits (e.g., 500, 2362), allowing separators
    m = re.search(r"\d[\d,\.]{2,}", text)
    return m is not None


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extraction: AnswerExtraction) -> None:
    # Root is initialized by caller; we set it critical and sequential behavior per rubric
    root = evaluator.root
    assert root is not None
    root.desc = "Evaluate whether the answer identifies the venue and provides its official seating capacity with verifiable sourcing, satisfying all constraints."
    root.strategy = AggregationStrategy.SEQUENTIAL
    root.critical = True  # All children must be critical

    # ------------------ Node 1: Identify Event Venue With Event Evidence ------------------ #
    node_identify = evaluator.add_parallel(
        id="Identify_Event_Venue_With_Event_Evidence",
        desc="The answer identifies the venue for Trevor Noah's Nashville, TN performance in February 2026, supported by reliable event/ticketing or official announcement evidence.",
        parent=root,
        critical=True
    )

    # Leaf: Venue_Name_Provided (custom existence check)
    _venue_name_present = extraction.venue_name is not None and extraction.venue_name.strip() != ""
    evaluator.add_custom_node(
        result=_venue_name_present,
        id="Venue_Name_Provided",
        desc="The answer explicitly names the venue.",
        parent=node_identify,
        critical=True
    )

    # Leaf: Venue_Is_Ryman_Auditorium (simple verify: name match)
    node_venue_match = evaluator.add_leaf(
        id="Venue_Is_Ryman_Auditorium",
        desc="The venue named in the answer is Ryman Auditorium (matches the stated constraint).",
        parent=node_identify,
        critical=True
    )
    venue_answer_name = extraction.venue_name or ""
    await evaluator.verify(
        claim=f"The named venue '{venue_answer_name}' and 'Ryman Auditorium' refer to the same venue.",
        node=node_venue_match,
        additional_instruction="Allow minor variations such as casing or the presence/absence of the word 'The' or 'Auditorium'. Consider 'Ryman' vs 'Ryman Auditorium' equivalent if they clearly refer to the same venue."
    )

    # Leaf: Event_Evidence_Supports_Performer_Date_Location (verify by URLs)
    node_event_evidence = evaluator.add_leaf(
        id="Event_Evidence_Supports_Performer_Date_Location",
        desc="Event evidence indicates performer Trevor Noah, location Nashville, TN, and a date in February 2026.",
        parent=node_identify,
        critical=True
    )
    event_urls = extraction.event_source_urls or []
    if event_urls:
        await evaluator.verify(
            claim="At least one of the provided URLs shows an event where Trevor Noah is performing in Nashville, Tennessee in February 2026.",
            node=node_event_evidence,
            sources=event_urls,
            additional_instruction=(
                "Accept reliable event/ticketing or official announcements (e.g., ryman.com, ticketmaster.com, livenation.com, axs.com, official artist or venue sites). "
                "Verify all three: performer is Trevor Noah, the location is Nashville, TN, and the event date falls in February 2026 (any day in that month). "
                "If none of the URLs show all three clearly, mark as not supported."
            )
        )
    else:
        # No URLs provided -> fail this critical leaf
        node_event_evidence.score = 0.0
        node_event_evidence.status = "failed"

    # ------------------ Node 2: Provide Official Venue Capacity With Evidence ------------------ #
    node_capacity = evaluator.add_parallel(
        id="Provide_Official_Venue_Capacity_With_Evidence",
        desc="The answer provides the official seating capacity of the identified venue and supports it with an acceptable verifiable source.",
        parent=root,
        critical=True
    )

    # Leaf: Numeric_Seating_Capacity_Provided (custom check)
    _capacity_numeric_like = looks_numeric_capacity(extraction.capacity_value)
    evaluator.add_custom_node(
        result=_capacity_numeric_like,
        id="Numeric_Seating_Capacity_Provided",
        desc="The answer states a numeric seating capacity value (in seats) for the venue.",
        parent=node_capacity,
        critical=True
    )

    # Leaf: Capacity_Source_Is_Acceptable (verify by URLs)
    node_capacity_source = evaluator.add_leaf(
        id="Capacity_Source_Is_Acceptable",
        desc="The cited source is acceptable (official venue or reliable event/ticketing) and supports the stated seating capacity.",
        parent=node_capacity,
        critical=True
    )
    capacity_urls = extraction.capacity_source_urls or []
    capacity_value_str = extraction.capacity_value or ""
    if capacity_urls:
        await evaluator.verify(
            claim=f"The provided source explicitly supports that the seating capacity of Ryman Auditorium is {capacity_value_str}.",
            node=node_capacity_source,
            sources=capacity_urls,
            additional_instruction=(
                "First, check that the page clearly pertains to Ryman Auditorium (the Nashville venue). "
                "Then verify that it explicitly states the venue's seating capacity matching the stated value (allow minor formatting differences like commas or the word 'seats'). "
                "Acceptable sources include the official venue website (e.g., ryman.com) or reputable ticketing/event sites (e.g., ticketmaster.com, livenation.com, axs.com). "
                "Wikipedia, random blogs, or low-credibility pages are NOT acceptable. If none of the URLs both (a) are acceptable and (b) clearly support the number, mark as not supported."
            )
        )
    else:
        node_capacity_source.score = 0.0
        node_capacity_source.status = "failed"


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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root will be sequential
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

    # Make root critical before adding children (so all children must be critical)
    root.critical = True

    # Extract structured fields from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_answer_info(),
        template_class=AnswerExtraction,
        extraction_name="answer_extraction"
    )

    # Add expected-venue ground truth info (for reference only)
    evaluator.add_ground_truth({
        "expected_venue": "Ryman Auditorium",
        "expected_city_state": "Nashville, TN",
        "expected_month_year": "February 2026"
    })

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extraction)

    # Return evaluation summary
    return evaluator.get_summary()