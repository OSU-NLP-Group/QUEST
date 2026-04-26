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
TASK_ID = "walmart_holiday_hours_2025"
TASK_DESCRIPTION = "Is Walmart open on Thanksgiving Day 2025 (November 27), and if not, what time does Walmart open on Black Friday 2025 (November 28)?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WalmartHoursExtraction(BaseModel):
    """
    Structured extraction of the agent's answer for Walmart holiday hours.
    """
    thanksgiving_status: Optional[str] = None  # e.g., "closed", "open", or a phrase like "closed on Thanksgiving Day"
    black_friday_open_time: Optional[str] = None  # e.g., "6 a.m. local time"
    thanksgiving_sources: List[str] = Field(default_factory=list)  # URLs cited for Thanksgiving status
    black_friday_sources: List[str] = Field(default_factory=list)  # URLs cited for Black Friday opening time


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_walmart_hours() -> str:
    return """
    Extract the Walmart holiday hours information specifically for the year 2025 from the provided answer.

    Required fields:
    1) thanksgiving_status: The exact textual phrase used in the answer to describe Walmart's operational status on Thanksgiving Day 2025 (November 27, 2025).
       - Examples: "closed", "not open", "closed on Thanksgiving Day", etc.
       - If the answer does not explicitly state the status for Thanksgiving 2025, return null.

    2) black_friday_open_time: The exact textual phrase used in the answer to describe when Walmart opens on Black Friday 2025 (November 28, 2025).
       - Examples: "6 a.m. local time", "opens at 6:00 AM", etc.
       - If the answer does not explicitly provide the opening time for Black Friday 2025, return null.

    3) thanksgiving_sources: A list of all URLs cited in the answer that support the Thanksgiving 2025 status claim.
       - Include only actual URLs explicitly present in the answer (plain URLs or markdown links).
       - If no URLs are provided, return an empty list.

    4) black_friday_sources: A list of all URLs cited in the answer that support the Black Friday 2025 opening time claim.
       - Include only actual URLs explicitly present in the answer (plain URLs or markdown links).
       - If no URLs are provided, return an empty list.

    Notes:
    - Thanksgiving Day 2025 is November 27, 2025; Black Friday 2025 is November 28, 2025.
    - Do not invent any URLs or information. Extract exactly what appears in the answer.
    - If the answer provides a consolidated "Sources" section without explicit mapping, include any URLs that reasonably apply to each claim in both lists.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_thanksgiving_status(
    evaluator: Evaluator,
    parent_node,
    extracted: WalmartHoursExtraction,
) -> None:
    """
    Build and run verification nodes for Thanksgiving 2025 status.
    """

    # Create a critical sequential node for Thanksgiving verification
    tg_node = evaluator.add_sequential(
        id="Thanksgiving_Status",
        desc="Correctly identifies that Walmart is closed on Thanksgiving Day (November 27, 2025)",
        parent=parent_node,
        critical=True,
    )

    # 1) Verify the answer explicitly states "closed on Thanksgiving Day 2025"
    tg_answer_leaf = evaluator.add_leaf(
        id="thanksgiving_answer_states_closed",
        desc="Answer explicitly states Walmart is closed on Thanksgiving Day 2025",
        parent=tg_node,
        critical=True,
    )
    tg_answer_claim = (
        "The answer explicitly states that Walmart is closed on Thanksgiving Day 2025 (November 27, 2025)."
    )
    await evaluator.verify(
        claim=tg_answer_claim,
        node=tg_answer_leaf,
        additional_instruction=(
            "Check the answer text to see if it clearly claims Walmart is closed on Thanksgiving Day 2025. "
            "Allow reasonable phrasing variants like 'closed on Thanksgiving', 'not open on Thanksgiving Day', "
            "or 'closed on Thursday, Nov 27, 2025'."
        ),
    )

    # 2) Existence of sources for Thanksgiving claim (critical gate)
    tg_sources_exist = evaluator.add_custom_node(
        result=(len(extracted.thanksgiving_sources) > 0),
        id="thanksgiving_sources_provided",
        desc="Sources for Thanksgiving 2025 status are provided in the answer",
        parent=tg_node,
        critical=True,
    )

    # 3) Verify cited sources support "closed on Thanksgiving Day 2025"
    tg_support_leaf = evaluator.add_leaf(
        id="thanksgiving_sources_support_closed",
        desc="Cited sources support that Walmart is closed on Thanksgiving Day 2025",
        parent=tg_node,
        critical=True,
    )
    tg_support_claim = "Walmart stores are closed on Thanksgiving Day 2025 (November 27, 2025)."
    await evaluator.verify(
        claim=tg_support_claim,
        node=tg_support_leaf,
        sources=extracted.thanksgiving_sources,
        additional_instruction=(
            "Use the provided URLs to verify the claim. The page should clearly state that Walmart is closed on "
            "Thanksgiving Day 2025. Accept reasonable phrasing such as 'closed on Thanksgiving' when the page context "
            "clearly pertains to 2025. If the page is about a different year (e.g., 2024) or is ambiguous/misleading, "
            "return not supported."
        ),
    )


async def verify_black_friday_opening(
    evaluator: Evaluator,
    parent_node,
    extracted: WalmartHoursExtraction,
) -> None:
    """
    Build and run verification nodes for Black Friday 2025 opening time.
    """

    # Create a critical sequential node for Black Friday verification
    bf_node = evaluator.add_sequential(
        id="Black_Friday_Opening",
        desc="Correctly states that Walmart opens at 6 a.m. local time on Black Friday (November 28, 2025)",
        parent=parent_node,
        critical=True,
    )

    # 1) Verify the answer explicitly states "opens at 6 a.m. local time on Black Friday 2025"
    bf_answer_leaf = evaluator.add_leaf(
        id="black_friday_answer_states_6am",
        desc="Answer explicitly states Walmart opens at 6 a.m. local time on Black Friday 2025",
        parent=bf_node,
        critical=True,
    )
    bf_answer_claim = (
        "The answer explicitly states that Walmart opens at 6 a.m. local time on Black Friday 2025 (November 28, 2025)."
    )
    await evaluator.verify(
        claim=bf_answer_claim,
        node=bf_answer_leaf,
        additional_instruction=(
            "Check the answer text to see if it clearly claims Walmart opens at 6 a.m. local time on Black Friday 2025. "
            "Allow formatting variants like '6:00 AM', '6 am', or 'six a.m.'."
        ),
    )

    # 2) Existence of sources for Black Friday opening time (critical gate)
    bf_sources_exist = evaluator.add_custom_node(
        result=(len(extracted.black_friday_sources) > 0),
        id="black_friday_sources_provided",
        desc="Sources for Black Friday 2025 opening time are provided in the answer",
        parent=bf_node,
        critical=True,
    )

    # 3) Verify cited sources support "opens at 6 a.m. local time on Black Friday 2025"
    bf_support_leaf = evaluator.add_leaf(
        id="black_friday_sources_support_6am",
        desc="Cited sources support that Walmart opens at 6 a.m. local time on Black Friday 2025",
        parent=bf_node,
        critical=True,
    )
    bf_support_claim = "Walmart opens at 6 a.m. local time on Black Friday 2025 (November 28, 2025)."
    await evaluator.verify(
        claim=bf_support_claim,
        node=bf_support_leaf,
        sources=extracted.black_friday_sources,
        additional_instruction=(
            "Use the provided URLs to verify the claim. The page should clearly state that Walmart opens at "
            "6 a.m. local time on Black Friday 2025. Accept minor formatting variants of the time. "
            "If the page refers to a different year or says a different time, return not supported."
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
    Evaluate the agent's answer to determine correctness regarding Walmart's Thanksgiving and Black Friday 2025 hours.
    """

    # Initialize evaluator (root is non-critical by framework design)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall checks are independent
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_walmart_hours(),
        template_class=WalmartHoursExtraction,
        extraction_name="walmart_holiday_hours_extraction",
    )

    # Add ground truth info for reference
    evaluator.add_ground_truth({
        "expected": {
            "thanksgiving_2025_status": "closed",
            "black_friday_2025_open_time": "6 a.m. local time",
            "dates": {
                "thanksgiving_2025": "November 27, 2025",
                "black_friday_2025": "November 28, 2025"
            }
        }
    }, gt_type="ground_truth")

    # Create a critical parent node mirroring the rubric root
    main_node = evaluator.add_parallel(
        id="Walmart_Holiday_Hours",
        desc="Verify Walmart's Thanksgiving and Black Friday 2025 hours",
        parent=root,
        critical=True,
    )

    # Build verification subtrees
    await verify_thanksgiving_status(evaluator, main_node, extracted)
    await verify_black_friday_opening(evaluator, main_node, extracted)

    # Return structured summary
    return evaluator.get_summary()