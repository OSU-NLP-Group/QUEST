import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "broadway_theater_shubert_musical_2026"
TASK_DESCRIPTION = """
Identify a Broadway theater that meets all of the following criteria: (1) The theater has a seating capacity between 1,000 and 1,500 seats, (2) The theater has a show currently running as of February 26, 2026, (3) The theater is operated by The Shubert Organization, (4) The current show is a musical production (not a play or other performance type), (5) The show will continue running through at least May 1, 2026 (i.e., it is not scheduled to close before this date), and (6) The theater is located in Manhattan's Theater District. Provide the name of the theater that satisfies all these criteria.
"""

AS_OF_DATE = "February 26, 2026"
THROUGH_DATE = "May 1, 2026"


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class TheaterSelection(BaseModel):
    theater_name: Optional[str] = None
    current_show_title: Optional[str] = None
    operator_name: Optional[str] = None
    show_type: Optional[str] = None
    closing_date: Optional[str] = None
    seating_capacity: Optional[str] = None
    location_text: Optional[str] = None

    theater_urls: List[str] = Field(default_factory=list)
    show_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_theater_selection() -> str:
    return f"""
    From the answer, extract the single Broadway theater proposed as meeting the criteria and key details about its current show. If multiple theaters are mentioned, choose the primary or final recommended theater.

    Return a JSON object with the following fields:
    - theater_name: The full name of the theater selected (e.g., "Majestic Theatre").
    - current_show_title: The title of the show claimed to be currently running at this theater (if provided).
    - operator_name: The name of the operating organization for the theater (if mentioned).
    - show_type: The performance type of the current show (e.g., "musical", "play"). Use the wording from the answer if present.
    - closing_date: The listed closing date for the current show if the answer mentions one; otherwise null.
    - seating_capacity: The seating capacity value or description as written in the answer; otherwise null.
    - location_text: The location or neighborhood description of the theater as written in the answer (e.g., "Theater District, Midtown Manhattan"); otherwise null.

    Also extract URLs explicitly present in the answer and group them by relevance:
    - theater_urls: URLs specifically about the theater (operator page, official theater page, IBDB, Playbill, Wikipedia, etc.).
    - show_urls: URLs specifically about the show (official site, Broadway league page, ticketing/performance calendar).
    - other_urls: Any additional URLs cited that support the claims.

    Rules:
    - Extract only URLs explicitly present in the answer (including markdown links). Do not invent URLs.
    - Include full URLs with protocol.
    - Do not deduplicate or filter beyond obvious invalid URLs.

    If any field is not present in the answer text, set it to null (or empty list for URL arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_preserve(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _combined_sources(sel: TheaterSelection) -> List[str]:
    return _dedup_preserve((sel.theater_urls or []) + (sel.show_urls or []) + (sel.other_urls or []))


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def verify_requirements(evaluator: Evaluator, root_node, sel: TheaterSelection) -> None:
    """
    Build verification leaves per rubric and trigger LLM-as-a-judge checks.
    Adds two minimal gating checks (name and sources) as critical leaves to enforce quality and source-grounding.
    """

    all_sources = _combined_sources(sel)
    theater_name = sel.theater_name or "the theater"
    show_title = sel.current_show_title

    # Gating leaves (critical): require theater name and at least one source URL
    evaluator.add_custom_node(
        result=(sel.theater_name is not None and sel.theater_name.strip() != ""),
        id="Theater_Name_Provided",
        desc="A specific theater name is provided in the answer",
        parent=root_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(len(all_sources) > 0),
        id="Sources_Provided",
        desc="At least one supporting source URL is provided in the answer",
        parent=root_node,
        critical=True
    )

    # 1) Seating capacity between 1,000 and 1,500
    cap_node = evaluator.add_leaf(
        id="Seating_Capacity_Requirement",
        desc="The theater's seating capacity must be between 1,000 and 1,500 seats",
        parent=root_node,
        critical=True
    )
    cap_claim = f"The Broadway theatre '{theater_name}' has a seating capacity between 1,000 and 1,500 seats."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_node,
        sources=all_sources,
        additional_instruction=(
            "Use the provided pages (operator/theatre official page, IBDB, Playbill, or Wikipedia) to determine the "
            "theatre's total seating capacity. If multiple numbers/sections are listed, consider the total capacity. "
            "Minor discrepancies within a few seats are acceptable as long as the total clearly falls within 1000–1500."
        ),
    )

    # 2) Show currently running as of Feb 26, 2026
    current_node = evaluator.add_leaf(
        id="Current_Show_Requirement",
        desc=f"The theater must have a show currently running as of {AS_OF_DATE}",
        parent=root_node,
        critical=True
    )
    if show_title and show_title.strip():
        current_claim = (
            f"As of {AS_OF_DATE}, the theatre '{theater_name}' has a production currently running: '{show_title}'."
        )
    else:
        current_claim = (
            f"As of {AS_OF_DATE}, the theatre '{theater_name}' has a production currently running (with performances "
            f"scheduled and not closed)."
        )
    await evaluator.verify(
        claim=current_claim,
        node=current_node,
        sources=all_sources,
        additional_instruction=(
            f"Look for a performance calendar, 'Now Playing', 'On Sale', or ticketing pages indicating performances on "
            f"or after {AS_OF_DATE}. If performances are scheduled on or beyond that date, consider this satisfied."
        ),
    )

    # 3) Operated by The Shubert Organization
    operator_node = evaluator.add_leaf(
        id="Operator_Requirement",
        desc="The theater must be operated by The Shubert Organization",
        parent=root_node,
        critical=True
    )
    operator_claim = f"The theatre '{theater_name}' is operated by The Shubert Organization."
    await evaluator.verify(
        claim=operator_claim,
        node=operator_node,
        sources=all_sources,
        additional_instruction=(
            "Prefer the operator's official site (The Shubert Organization). If a reliable source states the theatre is "
            "a Shubert theatre, that suffices."
        ),
    )

    # 4) Current show is a musical (not a play or other type)
    musical_node = evaluator.add_leaf(
        id="Musical_Type_Requirement",
        desc="The current show must be a musical production (not a play or other performance type)",
        parent=root_node,
        critical=True
    )
    if show_title and show_title.strip():
        musical_claim = f"The production '{show_title}' at '{theater_name}' is a musical (not a straight play)."
    else:
        musical_claim = f"The current production at '{theater_name}' is a musical (not a straight play)."
    await evaluator.verify(
        claim=musical_claim,
        node=musical_node,
        sources=all_sources,
        additional_instruction=(
            "Check the show's official page, Broadway League/IBDB/Playbill listings, or reliable sources for the genre. "
            "Accept clear evidence that the production is a musical (including descriptors like 'Broadway musical')."
        ),
    )

    # 5) Show continues through at least May 1, 2026
    schedule_node = evaluator.add_leaf(
        id="Show_Schedule_Requirement",
        desc=f"The show must not be scheduled to close before {THROUGH_DATE}",
        parent=root_node,
        critical=True
    )
    if show_title and show_title.strip():
        schedule_claim = (
            f"The production '{show_title}' at '{theater_name}' is scheduled to run through at least {THROUGH_DATE} "
            f"(i.e., it is not scheduled to close before that date)."
        )
    else:
        schedule_claim = (
            f"The current production at '{theater_name}' is scheduled to run through at least {THROUGH_DATE} "
            f"(i.e., it is not scheduled to close before that date)."
        )
    await evaluator.verify(
        claim=schedule_claim,
        node=schedule_node,
        sources=all_sources,
        additional_instruction=(
            f"From the provided sources, verify that the run extends to on/after {THROUGH_DATE} or is open-ended with "
            f"performances scheduled in May 2026. If a closing date is listed and it is earlier than {THROUGH_DATE}, "
            f"then this requirement is not met."
        ),
    )

    # 6) Located in Manhattan's Theater District
    location_node = evaluator.add_leaf(
        id="Theater_District_Location",
        desc="The theater must be located in Manhattan's Theater District",
        parent=root_node,
        critical=True
    )
    location_claim = f"The theatre '{theater_name}' is located in Manhattan's Theater District (Midtown Manhattan)."
    await evaluator.verify(
        claim=location_claim,
        node=location_node,
        sources=all_sources,
        additional_instruction=(
            "Accept statements that the theatre is in the Theater District or in Midtown Manhattan's Theater District. "
            "An address that is a known Theater District location also suffices if the page explicitly says so."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Broadway theater identification task.
    """
    # Initialize evaluator and root node
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel requirements; all critical must pass
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

    # Create the rubric-aligned root grouping node
    rubric_root = evaluator.add_parallel(
        id="Broadway_Theater_Identification",
        desc="Identify a Broadway theater that meets all specified criteria for attending a musical performance",
        parent=root,
        critical=False
    )

    # Extract structured information from the answer
    selection = await evaluator.extract(
        prompt=prompt_extract_theater_selection(),
        template_class=TheaterSelection,
        extraction_name="theater_selection"
    )

    # Build and run verification leaves
    await verify_requirements(evaluator, rubric_root, selection)

    # Return structured evaluation summary
    return evaluator.get_summary()