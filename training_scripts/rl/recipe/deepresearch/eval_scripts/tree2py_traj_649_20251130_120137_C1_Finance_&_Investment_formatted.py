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
TASK_ID = "nyse_close_2025_11_28"
TASK_DESCRIPTION = """
What time does the New York Stock Exchange (NYSE) close on Friday, November 28, 2025? Provide the time in Eastern Time.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class NYSEClosingExtraction(BaseModel):
    """
    Extract from the answer:
    - closing_time: the stated closing time for the NYSE on the specified date, as written (e.g., "1:00 p.m.", "1 PM", "13:00").
    - time_zone: the stated time zone string if provided (e.g., "ET", "EST", "Eastern Time", "Eastern Standard Time").
    - date_mentioned: the explicit date string if the answer repeats it (e.g., "November 28, 2025").
    """
    closing_time: Optional[str] = None
    time_zone: Optional[str] = None
    date_mentioned: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_nyse_closing_info() -> str:
    return """
    From the provided answer text, extract the NYSE closing details for the date in question.

    Return:
    - closing_time: The exact closing time string mentioned for the NYSE on the asked date (e.g., "1:00 p.m.", "1 PM", "1pm", "13:00"). If multiple times are mentioned, choose the one that is presented as the final answer for the NYSE closing time.
    - time_zone: The time zone expression associated with the closing time if present (e.g., "Eastern Time", "ET", "EST", "Eastern Standard Time"). If not specified, return null.
    - date_mentioned: The explicit date string if the answer repeats it (e.g., "November 28, 2025"). If the date is not explicitly repeated in the answer, return null.

    Do not infer or add information that is not explicitly in the answer text.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extraction: NYSEClosingExtraction,
) -> None:
    """
    Build the verification tree based on the rubric:
      - Parent critical node: "Verify the correct closing time for the NYSE on Friday, November 28, 2025"
        - Leaf (critical): "The answer states that the NYSE closes at 1:00 p.m. on November 28, 2025"
        - Leaf (critical): "The answer specifies Eastern Time (ET) or Eastern Standard Time (EST) for the closing time"
    """
    # Create the rubric's main node under the evaluator root.
    main_node = evaluator.add_parallel(
        id="NYSE_Closing_Time_November_28_2025",
        desc="Verify the correct closing time for the NYSE on Friday, November 28, 2025",
        parent=evaluator.root,
        critical=True
    )

    # Leaf 1: Closing Time Accuracy (critical)
    closing_time_leaf = evaluator.add_leaf(
        id="Closing_Time_Accuracy",
        desc="The answer states that the NYSE closes at 1:00 p.m. on November 28, 2025",
        parent=main_node,
        critical=True
    )
    # Use simple verification against the answer content only.
    # Allow common textual variations (1 PM, 1 p.m., 1pm, 1:00 PM, 13:00).
    await evaluator.verify(
        claim=(
            "The answer explicitly states that on Friday, November 28, 2025, "
            "the New York Stock Exchange (NYSE) closes at 1:00 PM."
        ),
        node=closing_time_leaf,
        additional_instruction=(
            "Judge only based on the provided answer text. Consider common variations equivalent, such as "
            "'1 PM', '1 p.m.', '1pm', '1:00 PM', or '13:00'. The statement must clearly attribute this time "
            "to the NYSE closing time for that date."
        ),
    )

    # Leaf 2: Time Zone Specification (critical)
    tz_leaf = evaluator.add_leaf(
        id="Time_Zone_Specification",
        desc="The answer specifies Eastern Time (ET) or Eastern Standard Time (EST) for the closing time",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The answer specifies that the closing time is in Eastern Time, for example using 'Eastern Time', "
            "'ET', or 'EST' (Eastern Standard Time)."
        ),
        node=tz_leaf,
        additional_instruction=(
            "Judge only based on the provided answer text. Accept any clear indicator of the Eastern time zone, "
            "including 'Eastern Time', 'ET', or 'EST'. The time zone mention should reasonably apply to the stated "
            "closing time, not an unrelated context."
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
    Evaluate an answer to: 'What time does the NYSE close on Friday, November 28, 2025? Provide the time in Eastern Time.'
    """
    # Initialize the evaluator
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

    # Ground truth (for reporting only; verification is against the answer text per rubric)
    evaluator.add_ground_truth(
        {
            "expected_closing_time": "1:00 PM",
            "expected_time_zone": "Eastern Time (ET/EST)",
            "date": "Friday, November 28, 2025",
            "note": "The Friday after U.S. Thanksgiving (commonly an early close at 1:00 PM ET)."
        },
        gt_type="expected_answer"
    )

    # Extract structured info from the answer (for transparency/debugging)
    extraction = await evaluator.extract(
        prompt=prompt_extract_nyse_closing_info(),
        template_class=NYSEClosingExtraction,
        extraction_name="closing_time_extraction"
    )

    # Build and run verification checks according to the rubric
    await build_verification_tree(evaluator, extraction)

    # Return the evaluation summary
    return evaluator.get_summary()