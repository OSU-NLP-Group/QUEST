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
TASK_ID = "hobby_lobby_earliest_time_after_thanksgiving_2025"
TASK_DESCRIPTION = "What is the earliest time you can purchase craft supplies from Hobby Lobby after Thanksgiving Day 2025? Provide both the specific time and date."

EXPECTED_TIME = "8:00 AM"
EXPECTED_DATE = "Friday, November 28, 2025"
THANKSGIVING_2025 = "Thursday, November 27, 2025"  # Context info


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EarliestShoppingInfo(BaseModel):
    """
    Extracted earliest shopping info from the agent's answer.
    """
    earliest_time: Optional[str] = None
    earliest_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_earliest_shopping_info() -> str:
    return """
    From the answer, extract the earliest time and date at which one can purchase craft supplies from Hobby Lobby after Thanksgiving Day 2025.

    Return a JSON object with:
    - earliest_time: The time explicitly stated as the earliest opening or shopping time after Thanksgiving Day 2025 (e.g., "8:00 AM", "8 AM").
    - earliest_date: The date explicitly stated for that earliest time (e.g., "Friday, November 28, 2025", "Nov 28, 2025", "11/28/2025").
    - sources: A list of all URLs explicitly cited in the answer that support the stated hours or date (include any Hobby Lobby official pages, store hours pages, or news/articles referenced). If no URLs are provided, return an empty list.

    Notes:
    - If either time or date is not mentioned, set it to null.
    - Accept reasonable formatting variants (e.g., "8 AM" vs "8:00 AM", "Fri Nov 28, 2025" vs "Friday, November 28, 2025").
    - Do not invent information—only extract what is explicitly in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extracted: EarliestShoppingInfo,
    parent_node_desc: str
) -> None:
    """
    Build the verification tree according to the rubric and perform verifications.
    """
    # Create main node as critical parallel aggregator (matches rubric root)
    main_node = evaluator.add_parallel(
        id="Earliest_Shopping_Time",
        desc=parent_node_desc,
        parent=evaluator.root,
        critical=True
    )

    # Leaf 1: Correct_Time (Critical)
    time_leaf = evaluator.add_leaf(
        id="Correct_Time",
        desc="Specifies 8:00 AM as the opening time",
        parent=main_node,
        critical=True
    )
    # Claim focuses on whether the answer explicitly specifies 8:00 AM (or equivalent) as the earliest time
    time_claim = (
        "The answer explicitly specifies 8:00 AM (or equivalent formatting such as '8 AM', '8am', '8:00am') "
        "as the earliest time when Hobby Lobby opens for shopping after Thanksgiving Day 2025 (i.e., on Black Friday)."
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_leaf,
        sources=extracted.sources if extracted and extracted.sources else None,
        additional_instruction=(
            "Focus on whether the answer states 8:00 AM as the earliest opening time after Thanksgiving Day 2025 for Hobby Lobby. "
            "Allow minor formatting variants (e.g., '8 AM', '8am', '8:00am'). "
            "If multiple times are mentioned, judge based on the earliest stated time after Thanksgiving Day. "
            "If URLs are provided, use them to confirm that 8:00 AM is correct for Black Friday 2025; "
            "otherwise, judge based on the answer text alone."
        )
    )

    # Leaf 2: Correct_Date (Critical)
    date_leaf = evaluator.add_leaf(
        id="Correct_Date",
        desc="Specifies Friday, November 28, 2025 (Black Friday) as the date",
        parent=main_node,
        critical=True
    )
    # Claim focuses on whether the answer explicitly specifies the correct date
    date_claim = (
        "The answer explicitly specifies Friday, November 28, 2025 (Black Friday 2025) as the date of the earliest time "
        "to shop at Hobby Lobby after Thanksgiving Day 2025."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=extracted.sources if extracted and extracted.sources else None,
        additional_instruction=(
            "Allow minor formatting variants for the date (e.g., 'Nov 28, 2025', '11/28/2025', 'Friday Nov 28, 2025'). "
            "The key is that the answer indicates Black Friday 2025, which is the day after Thanksgiving Day 2025, "
            "and specifically matches Friday, November 28, 2025."
        )
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
    Evaluate the agent's answer for the Hobby Lobby earliest shopping time after Thanksgiving Day 2025.
    """
    # Initialize evaluator with a parallel root (we'll add a critical child node that matches the rubric root)
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
        default_model=model
    )

    # Extract structured info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_earliest_shopping_info(),
        template_class=EarliestShoppingInfo,
        extraction_name="earliest_shopping_info"
    )

    # Add ground truth contextual info (used for transparency in summary)
    evaluator.add_ground_truth({
        "expected_time": EXPECTED_TIME,
        "expected_date": EXPECTED_DATE,
        "thanksgiving_2025": THANKSGIVING_2025,
        "note": "Expected values reflect typical Black Friday timing and calendar date. Verification prioritizes the agent's stated answer and any cited sources."
    }, gt_type="expected_values")

    # Build verification tree and run checks
    await build_and_verify_tree(
        evaluator=evaluator,
        extracted=extracted_info,
        parent_node_desc="Provides the earliest time to purchase craft supplies from Hobby Lobby after Thanksgiving Day 2025"
    )

    # Return the evaluation summary
    return evaluator.get_summary()