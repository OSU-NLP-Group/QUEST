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
TASK_ID = "ewc_2026_details"
TASK_DESCRIPTION = "What is the location, date range, and total prize pool for the Esports World Cup 2026?"


EXPECTED_LOCATION = "Riyadh, Saudi Arabia"
EXPECTED_DATE_START = "July 6, 2026"
EXPECTED_DATE_END = "August 23, 2026"
EXPECTED_PRIZE_POOL = "$75,000,000 USD"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EventDetailsExtraction(BaseModel):
    """
    Structured extraction of event details from the answer.
    """
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    date_range: Optional[str] = None
    prize_pool: Optional[str] = None

    # Sources
    all_sources: List[str] = Field(default_factory=list)
    location_sources: List[str] = Field(default_factory=list)
    date_sources: List[str] = Field(default_factory=list)
    prize_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_event_details() -> str:
    return """
    Extract the Esports World Cup 2026 event details as stated in the answer.

    Required fields:
    - location: The event location string exactly as written in the answer (e.g., "Riyadh, Saudi Arabia").
    - start_date: The event start date string if explicitly mentioned (e.g., "July 6, 2026"). If not present, return null.
    - end_date: The event end date string if explicitly mentioned (e.g., "August 23, 2026"). If not present, return null.
    - date_range: If the answer provides the full date range in one combined string (e.g., "July 6, 2026 – August 23, 2026"), extract that combined range. Otherwise, return null.
    - prize_pool: The total prize pool string exactly as written in the answer (e.g., "$75,000,000 USD"). If not present, return null.

    Sources (URLs):
    - all_sources: Extract ALL URLs present anywhere in the answer (including markdown links). These are general sources.
    - location_sources: Extract URLs that are specifically associated with or placed near the location statement.
    - date_sources: Extract URLs that are specifically associated with or placed near the date statement.
    - prize_sources: Extract URLs that are specifically associated with or placed near the prize pool statement.

    Rules:
    - Extract only URLs explicitly present in the answer (plain URLs or markdown links). Do not invent any URLs.
    - Normalize markdown links and extract the underlying URL; store only the URL string.
    - If a field is missing, return null. If a sources list is not provided, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def merge_sources(primary: Optional[List[str]], fallback: Optional[List[str]]) -> List[str]:
    """
    Merge two URL lists, preserving order and deduplicating.
    """
    seen = set()
    merged: List[str] = []
    for lst in (primary or []):
        u = (lst or "").strip()
        if u and u not in seen:
            merged.append(u)
            seen.add(u)
    for u in (fallback or []):
        u = (u or "").strip()
        if u and u not in seen:
            merged.append(u)
            seen.add(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root_node,
    extracted: EventDetailsExtraction
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """
    # Create the rubric's top-level node (parallel aggregation, non-critical)
    details_node = evaluator.add_parallel(
        id="Esports_World_Cup_2026_Details",
        desc="Verify the basic details of the Esports World Cup 2026 event",
        parent=root_node,
        critical=False
    )

    # ---------------------- Location ---------------------- #
    loc_sources = merge_sources(extracted.location_sources, extracted.all_sources)

    # Non-critical gating prerequisites (local to this leaf)
    loc_value_provided = evaluator.add_custom_node(
        result=bool(extracted.location and extracted.location.strip()),
        id="Location_value_provided",
        desc="Location value is provided in the answer",
        parent=details_node,
        critical=False
    )
    loc_sources_present = evaluator.add_custom_node(
        result=bool(loc_sources),
        id="Location_sources_present",
        desc="Location has at least one cited source URL",
        parent=details_node,
        critical=False
    )

    # Critical leaf from rubric
    location_leaf = evaluator.add_leaf(
        id="Location",
        desc="The event location must be Riyadh, Saudi Arabia",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Esports World Cup 2026 location is Riyadh, Saudi Arabia.",
        node=location_leaf,
        sources=loc_sources if loc_sources else None,
        additional_instruction=(
            "Confirm the webpage explicitly indicates that the 2026 Esports World Cup is held in Riyadh, Saudi Arabia. "
            "Allow phrasing variants like 'hosted in Riyadh' or 'takes place in Riyadh, Saudi Arabia'. "
            "The page must be relevant to the 2026 Esports World Cup. If the URL is irrelevant or does not support the claim, mark as not supported."
        ),
        extra_prerequisites=[loc_value_provided, loc_sources_present]
    )

    # ---------------------- Date Range ---------------------- #
    date_sources = merge_sources(extracted.date_sources, extracted.all_sources)

    date_value_provided = evaluator.add_custom_node(
        result=bool(extracted.date_range and extracted.date_range.strip()) or
               (bool(extracted.start_date and extracted.start_date.strip()) and
                bool(extracted.end_date and extracted.end_date.strip())),
        id="Date_Range_value_provided",
        desc="Date range value is provided in the answer (either combined or as start and end dates)",
        parent=details_node,
        critical=False
    )
    date_sources_present = evaluator.add_custom_node(
        result=bool(date_sources),
        id="Date_Range_sources_present",
        desc="Date range has at least one cited source URL",
        parent=details_node,
        critical=False
    )

    date_leaf = evaluator.add_leaf(
        id="Date_Range",
        desc="The event must run from July 6, 2026 to August 23, 2026",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Esports World Cup 2026 runs from July 6, 2026 to August 23, 2026.",
        node=date_leaf,
        sources=date_sources if date_sources else None,
        additional_instruction=(
            "Verify that the webpage states both the start and end dates for the 2026 Esports World Cup. "
            "Allow minor formatting variants (e.g., 'July 6 – Aug 23, 2026', '6 July 2026 to 23 August 2026'). "
            "Ensure the dates correspond specifically to the 2026 event."
        ),
        extra_prerequisites=[date_value_provided, date_sources_present]
    )

    # ---------------------- Prize Pool ---------------------- #
    prize_sources = merge_sources(extracted.prize_sources, extracted.all_sources)

    prize_value_provided = evaluator.add_custom_node(
        result=bool(extracted.prize_pool and extracted.prize_pool.strip()),
        id="Prize_Pool_value_provided",
        desc="Prize pool value is provided in the answer",
        parent=details_node,
        critical=False
    )
    prize_sources_present = evaluator.add_custom_node(
        result=bool(prize_sources),
        id="Prize_Pool_sources_present",
        desc="Prize pool has at least one cited source URL",
        parent=details_node,
        critical=False
    )

    prize_leaf = evaluator.add_leaf(
        id="Prize_Pool",
        desc="The total prize pool must be $75,000,000 USD",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim="The total prize pool for the Esports World Cup 2026 is $75,000,000 USD.",
        node=prize_leaf,
        sources=prize_sources if prize_sources else None,
        additional_instruction=(
            "Confirm that the page states the total prize pool as approximately seventy-five million US dollars. "
            "Accept equivalent representations such as 'US$ 75 million', '$75M', or '75,000,000 USD'. "
            "Ensure the prize pool figure is clearly tied to the 2026 Esports World Cup."
        ),
        extra_prerequisites=[prize_value_provided, prize_sources_present]
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
    Evaluate an answer for Esports World Cup 2026 details.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent checks for location, dates, prize
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

    # Extract event details from the answer
    extracted_details = await evaluator.extract(
        prompt=prompt_extract_event_details(),
        template_class=EventDetailsExtraction,
        extraction_name="event_details"
    )

    # Add ground truth info for transparency
    evaluator.add_ground_truth({
        "expected_location": EXPECTED_LOCATION,
        "expected_start_date": EXPECTED_DATE_START,
        "expected_end_date": EXPECTED_DATE_END,
        "expected_prize_pool": EXPECTED_PRIZE_POOL
    }, gt_type="expected_values")

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extracted_details)

    # Return summary
    return evaluator.get_summary()