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
TASK_ID = "nyse_black_friday_hours"
TASK_DESCRIPTION = "What time does the New York Stock Exchange (NYSE) close on the day after Thanksgiving (Black Friday)?"


# --------------------------------------------------------------------------- #
# Optional extraction model (for recording any URLs mentioned in the answer)  #
# --------------------------------------------------------------------------- #
class URLExtraction(BaseModel):
    urls: List[str] = Field(default_factory=list)


def prompt_extract_urls() -> str:
    return """
    Extract all URLs explicitly mentioned in the answer text. Include URLs whether they are plain links,
    markdown links, or embedded within text. Return them in a field named 'urls'.
    If no URLs are present, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def verify_black_friday_hours(evaluator: Evaluator, parent_node) -> None:
    """
    Build the verification tree according to the rubric and run all checks.
    Each leaf checks whether the answer itself states the required information.
    """
    # Group node (critical, parallel)
    group_node = evaluator.add_parallel(
        id="NYSE_Black_Friday_Hours",
        desc="Provides required NYSE/U.S. equity market hours details for the day after Thanksgiving (Black Friday) per the given constraints.",
        parent=parent_node,
        critical=True,
    )

    # Leaf nodes (all critical)
    node_open_status = evaluator.add_leaf(
        id="Market_Open_Status",
        desc="States that the NYSE/U.S. equity markets are open on the day after Thanksgiving (Black Friday).",
        parent=group_node,
        critical=True,
    )
    node_regular_open = evaluator.add_leaf(
        id="Regular_Open_Time",
        desc="States that markets open at their regular time of 9:30 AM ET on Black Friday.",
        parent=group_node,
        critical=True,
    )
    node_early_close = evaluator.add_leaf(
        id="Early_Close_Time",
        desc="States that U.S. equity markets (including NYSE) close early at 1:00 PM ET on Black Friday.",
        parent=group_node,
        critical=True,
    )
    node_options_close = evaluator.add_leaf(
        id="Eligible_Options_Close_Time",
        desc="States that for eligible options, the close time is 1:15 PM ET on Black Friday.",
        parent=group_node,
        critical=True,
    )
    node_timezone = evaluator.add_leaf(
        id="Time_Zone_Specification",
        desc="Specifies that all reported times are in Eastern Time (ET).",
        parent=group_node,
        critical=True,
    )

    # Prepare claims and run verifications in parallel
    claims_and_sources = [
        (
            "The answer explicitly states that the NYSE or U.S. equity markets are open on the day after Thanksgiving (Black Friday), or clearly implies they are open by providing trading hours for that day.",
            None,
            node_open_status,
            "Accept clear implication of being open if specific Black Friday hours (e.g., opening and/or early closing times) are provided, even if the exact word 'open' is not used. The check is purely about what the answer states."
        ),
        (
            "The answer states that markets open at 9:30 AM ET on Black Friday (their regular opening time).",
            None,
            node_regular_open,
            "Accept reasonable variants such as '9:30 am ET', '9:30 a.m. ET', or '09:30 ET'. The statement must connect to Black Friday opening time. If the answer only says 'regular time' without specifying 9:30, consider this insufficient."
        ),
        (
            "The answer states that U.S. equity markets (including the NYSE) close early at 1:00 PM ET on Black Friday.",
            None,
            node_early_close,
            "Accept reasonable variants such as '1 pm ET', '1:00 p.m. ET', or '13:00 ET'. Ensure the statement pertains to Black Friday (the day after Thanksgiving)."
        ),
        (
            "The answer states that for eligible options, the close time on Black Friday is 1:15 PM ET.",
            None,
            node_options_close,
            "Accept reasonable variants such as '1:15 pm ET' or '13:15 ET'. Phrasings like 'eligible options' or 'options market' are acceptable if they clearly refer to options close time."
        ),
        (
            "The answer explicitly specifies that the times are in Eastern Time (ET).",
            None,
            node_timezone,
            "Accept variants such as 'ET', 'Eastern Time', 'Eastern', 'EST', or 'EDT'. As long as the answer clearly indicates the Eastern time zone for the provided times, it satisfies this check."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the NYSE Black Friday hours task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation strategy
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

    # Optional: extract any URLs mentioned in the answer (for record/debugging)
    _urls = await evaluator.extract(
        prompt=prompt_extract_urls(),
        template_class=URLExtraction,
        extraction_name="extracted_urls_from_answer"
    )

    # Add ground truth expectation info for reference in summary (not used for gating)
    evaluator.add_ground_truth({
        "expected_open_status": "Open on Black Friday",
        "expected_open_time": "9:30 AM ET (regular open)",
        "expected_close_time": "1:00 PM ET (early close for U.S. equities, including NYSE)",
        "expected_options_close_time": "1:15 PM ET (eligible options)",
        "expected_time_zone": "Eastern Time (ET)"
    }, gt_type="nyse_black_friday_expected_hours")

    # Build verification tree and run checks
    await verify_black_friday_hours(evaluator, root)

    # Return structured summary
    return evaluator.get_summary()