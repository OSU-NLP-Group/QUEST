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
TASK_ID = "whole_foods_thanksgiving_hours_2025"
TASK_DESCRIPTION = (
    "What are the operating hours for Whole Foods stores on Thanksgiving 2025, "
    "and in which U.S. states are Whole Foods stores closed on this holiday?"
)

EXPECTED_HOURS_TEXT = "7 a.m. to 1 p.m."
EXPECTED_CLOSED_STATES = ["Massachusetts", "Maine", "Rhode Island"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HoursStatesExtraction(BaseModel):
    """
    Extracted fields from the agent's answer:
    - thanksgiving_hours: The general operating hours the answer claims for Whole Foods on Thanksgiving 2025
    - closed_states: The list of U.S. states the answer claims are closed on Thanksgiving
    - source_urls: All URLs cited/mentioned in the answer
    """
    thanksgiving_hours: Optional[str] = None
    closed_states: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hours_states() -> str:
    return """
    Extract the following from the answer:

    1) thanksgiving_hours:
       - The general operating hours stated for Whole Foods stores on Thanksgiving 2025.
       - Return a single string exactly as written in the answer (e.g., "7 a.m. to 1 p.m.", "7 AM–1 PM", "7am-1pm").
       - If the answer mentions multiple hours for different locations, extract the general nationwide hours that the answer claims apply by default.
       - If the answer does not present a clear general hours window for Thanksgiving 2025, return null.

    2) closed_states:
       - A list of U.S. state names that the answer claims Whole Foods stores are closed on Thanksgiving.
       - If the answer uses abbreviations (e.g., MA, ME, RI), convert to full state names if you can infer them unambiguously; otherwise, include the abbreviations as they appear.
       - If none are mentioned, return an empty array.

    3) source_urls:
       - Extract all URLs mentioned anywhere in the answer (including markdown links).
       - Return them as a flat array of full URLs.
       - Do not invent URLs. If none are present, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def add_operating_hours_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: HoursStatesExtraction,
) -> None:
    """
    Build the 'Operating_Hours' sub-tree:
    - Check the answer explicitly states the expected hours (simple verification against the answer).
    - Check sources (if any) support the expected hours for Thanksgiving 2025.
    """
    op_node = evaluator.add_sequential(
        id="Operating_Hours",
        desc="States that Whole Foods stores are open from 7 a.m. to 1 p.m. on Thanksgiving 2025",
        parent=parent_node,
        critical=True,  # Child of a critical node must also be critical
    )

    # Leaf 1: The answer states the expected hours
    hours_stated_leaf = evaluator.add_leaf(
        id="Operating_Hours_stated",
        desc="The answer explicitly states Whole Foods stores are open from 7 a.m. to 1 p.m. on Thanksgiving 2025",
        parent=op_node,
        critical=True,
    )
    hours_stated_claim = (
        "The answer explicitly states that Whole Foods stores are open from 7 a.m. to 1 p.m. on Thanksgiving 2025."
    )
    await evaluator.verify(
        claim=hours_stated_claim,
        node=hours_stated_leaf,
        additional_instruction=(
            "Look for an explicit statement in the answer text that mentions the Thanksgiving 2025 hours for Whole Foods "
            "as '7 a.m. to 1 p.m.' Minor formatting differences like '7 AM–1 PM', '7am-1pm', or spacing/typography variations "
            "should be considered equivalent."
        ),
    )

    # Leaf 2: The expected hours are supported by the cited sources (if any)
    hours_supported_leaf = evaluator.add_leaf(
        id="Operating_Hours_supported",
        desc="Cited sources support 7 a.m. to 1 p.m. Thanksgiving hours for Whole Foods in 2025",
        parent=op_node,
        critical=True,
    )
    hours_supported_claim = (
        "Whole Foods Market stores are open from 7 a.m. to 1 p.m. on Thanksgiving Day 2025."
    )
    await evaluator.verify(
        claim=hours_supported_claim,
        node=hours_supported_leaf,
        sources=extracted.source_urls,
        additional_instruction=(
            "Check the provided webpages to see if they explicitly support or clearly state Whole Foods' Thanksgiving Day "
            "hours as 7 a.m. to 1 p.m. for 2025. Minor formatting variations (e.g., '7 AM–1 PM', '7am-1pm') are acceptable. "
            "If the page references a different year without indicating 2025, or references another retailer, it should not be considered supporting evidence."
        ),
    )


async def add_state_exceptions_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: HoursStatesExtraction,
) -> None:
    """
    Build the 'State_Exceptions' sub-tree:
    - Check the answer explicitly identifies MA, ME, and RI as closed on Thanksgiving.
    - Check sources (if any) support closures in these states.
    """
    st_node = evaluator.add_sequential(
        id="State_Exceptions",
        desc="Identifies that Whole Foods stores in Massachusetts, Maine, and Rhode Island are closed on Thanksgiving",
        parent=parent_node,
        critical=True,  # Child of a critical node must also be critical
    )

    # Leaf 1: The answer states MA, ME, and RI are closed
    states_stated_leaf = evaluator.add_leaf(
        id="State_Exceptions_stated",
        desc="The answer states that Whole Foods stores in Massachusetts, Maine, and Rhode Island are closed on Thanksgiving",
        parent=st_node,
        critical=True,
    )
    states_stated_claim = (
        "The answer states that Whole Foods stores in Massachusetts, Maine, and Rhode Island are closed on Thanksgiving."
    )
    await evaluator.verify(
        claim=states_stated_claim,
        node=states_stated_leaf,
        additional_instruction=(
            "Confirm that the answer text includes all three of these states: Massachusetts, Maine, and Rhode Island. "
            "Abbreviations (MA, ME, RI) are acceptable. The statement can reference 'blue laws' or similar reasons, "
            "but the key is that closure in all three states is explicitly indicated."
        ),
    )

    # Leaf 2: Sources support closures in those states
    states_supported_leaf = evaluator.add_leaf(
        id="State_Exceptions_supported",
        desc="Cited sources support that Whole Foods stores in MA, ME, and RI are closed on Thanksgiving",
        parent=st_node,
        critical=True,
    )
    states_supported_claim = (
        "Whole Foods stores in Massachusetts, Maine, and Rhode Island are closed on Thanksgiving Day (including 2025)."
    )
    await evaluator.verify(
        claim=states_supported_claim,
        node=states_supported_leaf,
        sources=extracted.source_urls,
        additional_instruction=(
            "Verify that the webpages indicate that grocery/retail stores (including Whole Foods) are closed in "
            "Massachusetts, Maine, and Rhode Island on Thanksgiving, commonly due to state laws (often called 'blue laws'). "
            "A page that clearly states Thanksgiving Day closures for grocery retailers in these states should be considered supporting evidence."
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
    Evaluate an answer for Whole Foods Thanksgiving 2025 hours and state closures.
    Returns the standard evaluation summary dict produced by the Evaluator.
    """
    # Initialize
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hours_states(),
        template_class=HoursStatesExtraction,
        extraction_name="hours_states_extraction",
    )

    # Record ground truth expectations for transparency (not used for scoring directly)
    evaluator.add_ground_truth({
        "expected_hours_text": EXPECTED_HOURS_TEXT,
        "expected_closed_states": EXPECTED_CLOSED_STATES,
    }, gt_type="ground_truth")

    # Build the rubric tree (as described in the provided JSON)
    top_node = evaluator.add_parallel(
        id="Whole_Foods_Thanksgiving_Hours",
        desc="Provides accurate information about Whole Foods operating hours on Thanksgiving 2025, including both general hours and state-specific exceptions",
        parent=root,
        critical=True,  # Root criteria is critical; all children must be critical
    )

    # Operating hours checks
    await add_operating_hours_checks(evaluator, top_node, extracted)

    # State exceptions checks
    await add_state_exceptions_checks(evaluator, top_node, extracted)

    # Return the full summary (includes extraction info and verification tree)
    return evaluator.get_summary()