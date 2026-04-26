import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "jack_white_detroit_capacity_2025_04_12"
TASK_DESCRIPTION = "What was the seating capacity of the venue where Jack White performed in Detroit on April 12, 2025?"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueCapacityExtraction(BaseModel):
    """
    Structured info extracted from the agent's answer for the Jack White Detroit event (2025-04-12).
    """
    venue_name: Optional[str] = None
    event_city: Optional[str] = None
    event_date: Optional[str] = None
    capacity_value: Optional[str] = None
    capacity_statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_capacity() -> str:
    return """
    Extract the key information reported by the answer about Jack White’s Detroit performance on April 12, 2025.

    You must extract exactly what the answer states (do not invent or normalize beyond what the answer explicitly says).

    Return the following fields:
    - venue_name: The name of the venue the answer claims hosted the Detroit performance on April 12, 2025.
    - event_city: The city stated for the performance (should be Detroit if present).
    - event_date: The date stated for the performance as written in the answer (e.g., "April 12, 2025" or "2025-04-12").
    - capacity_value: The seating capacity value the answer reports for the venue (return the number/phrase as it appears, e.g., "5,000", "about 5,000", "~5k", "5,100–5,300").
    - capacity_statement: The exact sentence or phrase from the answer where the capacity is stated (to clarify that it refers to seating capacity and not attendance).
    - sources: All URLs mentioned anywhere in the answer (including markdown links). Extract only valid URLs explicitly present in the answer.

    If any field is not present in the answer, return null for that field. If no URLs are present, return an empty list for sources.

    If multiple venues or dates are mentioned, select the venue and date specifically associated with the Detroit, Michigan performance on April 12, 2025.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extracted: VenueCapacityExtraction
) -> None:
    """
    Build the verification tree according to the rubric:
    1) Identify the correct venue for Jack White’s Detroit performance on April 12, 2025.
    2) Provide (and verify) the venue's seating capacity value.
    3) Ensure the reported value explicitly refers to seating capacity, not attendance.
    All three are critical under a sequential parent.
    """
    # Parent node (critical, sequential)
    parent_node = evaluator.add_sequential(
        id="Jack_White_Detroit_April_12_2025_Seating_Capacity",
        desc="Determine the seating capacity of the venue where Jack White performed in Detroit on April 12, 2025.",
        parent=evaluator.root,
        critical=True
    )

    # Prepare extracted values (safe fallbacks)
    venue_name = (extracted.venue_name or "").strip()
    event_city = (extracted.event_city or "Detroit").strip()
    event_date = (extracted.event_date or "April 12, 2025").strip()
    capacity_value = (extracted.capacity_value or "").strip()
    capacity_statement = (extracted.capacity_statement or "").strip()
    sources = extracted.sources if extracted.sources else None

    # 1) Identify correct venue for the specified event (leaf, critical)
    node_venue = evaluator.add_leaf(
        id="Identify_Correct_Venue_For_Specified_Event",
        desc="Correctly identifies the venue corresponding to Jack White’s performance in Detroit, Michigan on April 12, 2025.",
        parent=parent_node,
        critical=True
    )
    claim_venue = (
        f"Jack White performed in {event_city}, Michigan on {event_date}, and the venue was '{venue_name}'."
    )
    await evaluator.verify(
        claim=claim_venue,
        node=node_venue,
        sources=sources,
        additional_instruction=(
            "Verify that the provided source(s) explicitly indicate a Jack White performance occurred on the given date "
            "in Detroit (Michigan) and that the venue matches the stated name. Allow minor formatting differences in dates "
            "(e.g., 2025-04-12 vs April 12, 2025) and venue name variants (official naming vs common alias). "
            "If the URLs are irrelevant or do not mention this specific event and venue, mark as not supported."
        )
    )

    # 2) Provide venue seating capacity (leaf, critical)
    node_capacity_value = evaluator.add_leaf(
        id="Provide_Venue_Seating_Capacity",
        desc="States the seating capacity of the identified venue (as a numeric value, optionally noting an approximate figure if sources vary).",
        parent=parent_node,
        critical=True
    )
    claim_capacity = (
        f"The seating capacity of the venue '{venue_name}' is {capacity_value}."
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=node_capacity_value,
        sources=sources,
        additional_instruction=(
            "Determine whether any of the provided source(s) explicitly support the stated seating capacity for the venue. "
            "Accept reasonable approximations (e.g., 'about', '~', ranges) and minor rounding differences. If a source lists multiple configurations "
            "(seated vs standing), focus on seated capacity; it's acceptable if the stated value approximates the seated capacity. "
            "If no source supports the capacity, mark as not supported."
        )
    )

    # 3) Ensure value refers to seating capacity, not attendance (leaf, critical)
    node_capacity_is_seating = evaluator.add_leaf(
        id="Capacity_Is_Seating_Capacity_Not_Attendance",
        desc="The reported value is explicitly the venue’s seating capacity (not attendance, tickets sold, or another metric).",
        parent=parent_node,
        critical=True
    )
    # If we have an exact quoted statement from the answer, use it to aid verification.
    if capacity_statement:
        claim_capacity_semantics = (
            f"In the answer, the statement \"{capacity_statement}\" indicates that the number {capacity_value} "
            f"refers to the venue's seating capacity, not attendance, tickets sold, or any other metric."
        )
    else:
        claim_capacity_semantics = (
            f"In the answer, the number {capacity_value} is explicitly presented as the venue's seating capacity "
            f"(and not attendance, tickets sold, or any other metric)."
        )
    # This is a semantic check based on the answer text; use simple verification (no URLs).
    await evaluator.verify(
        claim=claim_capacity_semantics,
        node=node_capacity_is_seating,
        sources=None,
        additional_instruction=(
            "Judge solely from the phrasing in the provided answer text. "
            "It must clearly convey that the number is a seating capacity (e.g., 'seating capacity', 'seats', 'seated capacity'). "
            "If the phrasing indicates attendance (e.g., 'sold out crowd of X', 'attendance of X', 'tickets sold'), "
            "or is ambiguous/not clearly seating capacity, mark as incorrect."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Entry point for evaluating an agent's answer to:
    'What was the seating capacity of the venue where Jack White performed in Detroit on April 12, 2025?'
    """
    # Initialize evaluator (root node is non-critical wrapper)
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
        default_model=model
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_capacity(),
        template_class=VenueCapacityExtraction,
        extraction_name="venue_capacity_extraction"
    )

    # Build verification tree per rubric and run verifications
    await build_verification_tree(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()