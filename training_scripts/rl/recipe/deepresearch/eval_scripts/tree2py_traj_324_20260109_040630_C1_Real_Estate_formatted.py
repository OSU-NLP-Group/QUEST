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
TASK_ID = "us_property_tax_highest_2025_2026"
TASK_DESCRIPTION = (
    "I'm considering relocating to a different state in the USA and want to understand property tax implications. "
    "Which U.S. state has the highest effective property tax rate as of 2025-2026, and what is that rate? "
    "Please provide the state name and the specific percentage rate."
)

# Ground truth context for reference in summary
GROUND_TRUTH = {
    "expected_state": "New Jersey",
    "expected_effective_rate": "2.23%",
    "expected_timeframe": "2025–2026"
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PropertyTaxAnswerExtraction(BaseModel):
    """Structured extraction from the agent's answer regarding highest effective property tax rate."""
    state: Optional[str] = None
    rate_text: Optional[str] = None
    metric_text: Optional[str] = None
    timeframe_text: Optional[str] = None
    years_mentioned: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_property_tax_info() -> str:
    return """
    Extract the specific details the answer provides about the highest effective property tax rate in the U.S.:
    1. state: The state the answer claims has the highest effective property tax rate.
    2. rate_text: The exact rate text as shown in the answer (e.g., "2.23%" or "2.23 percent"). Include any symbols or formatting.
    3. metric_text: The description of the metric used for the rate (e.g., "effective property tax rate as a percentage of home value", "median annual tax bill", "average tax per $1,000 of value", etc.). Use the wording from the answer.
    4. timeframe_text: Any explicit timeframe or year references that the answer associates with the rate (e.g., "as of 2025", "2026", "2025–2026", "for tax year 2025").
    5. years_mentioned: A list of individual years explicitly mentioned in the answer (e.g., ["2025", "2026"]).
    6. source_urls: All URLs explicitly present in the answer that relate to the claim (including citations or references). Extract valid URLs only.

    If a field is not present in the answer, return null for single-value fields and an empty list for arrays.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root_node,
    extraction: PropertyTaxAnswerExtraction,
) -> None:
    """
    Build and execute the verification tree according to the rubric:
      - state_name_correct
      - rate_value_correct
      - rate_is_percentage_effective
      - timeframe_alignment

    All are critical leaf checks under a parallel root aggregation.
    """

    # 1) The answer names New Jersey as the state with the highest effective property tax rate
    state_leaf = evaluator.add_leaf(
        id="state_name_correct",
        desc="The answer names New Jersey as the state with the highest effective property tax rate (per the provided constraints).",
        parent=root_node,
        critical=True,
    )
    claim_state = (
        "The answer explicitly states that New Jersey is the U.S. state with the highest effective property tax rate."
    )
    await evaluator.verify(
        claim=claim_state,
        node=state_leaf,
        additional_instruction=(
            "Focus on whether the answer names 'New Jersey' as the top state for effective property tax rate. "
            "Do not judge the external truth of the claim here—only whether the answer itself asserts New Jersey as highest."
        ),
    )

    # 2) The answer provides the specific effective property tax rate as 2.23%
    rate_leaf = evaluator.add_leaf(
        id="rate_value_correct",
        desc="The answer gives the effective property tax rate as 2.23%.",
        parent=root_node,
        critical=True,
    )
    claim_rate = "The answer provides the effective property tax rate value as exactly 2.23%."
    await evaluator.verify(
        claim=claim_rate,
        node=rate_leaf,
        additional_instruction=(
            "Verify that the answer includes the exact rate '2.23%'. "
            "Treat '2.23 percent' as equivalent to '2.23%'. "
            "Do not accept approximations such as 'about 2.2%' or 'roughly 2.23' without the percent sign."
        ),
    )

    # 3) The rate is presented as an effective property tax rate as a percentage of home value
    metric_leaf = evaluator.add_leaf(
        id="rate_is_percentage_effective",
        desc="The rate is presented as an effective property tax rate expressed as a percentage of home value (not another metric like median bill).",
        parent=root_node,
        critical=True,
    )
    claim_metric = (
        "The answer frames the rate as an effective property tax rate expressed as a percentage of home value, "
        "and not as a dollar amount (e.g., median or average annual bill) or some other non-percentage metric."
    )
    await evaluator.verify(
        claim=claim_metric,
        node=metric_leaf,
        additional_instruction=(
            "Check the phrasing surrounding the rate. It must indicate a percentage of home value (effective rate), "
            "not a median/average bill in dollars, tax per $1,000, or a ranking without specifying percentage of home value."
        ),
    )

    # 4) The timeframe aligns with 2025–2026
    timeframe_leaf = evaluator.add_leaf(
        id="timeframe_alignment",
        desc="The answer indicates the rate is for the requested 2025–2026 timeframe (e.g., cites 2025/2026 data consistent with the constraints).",
        parent=root_node,
        critical=True,
    )
    claim_timeframe = (
        "The answer clearly indicates that the provided rate pertains to the 2025–2026 timeframe "
        "(e.g., explicitly mentions 2025 or 2026 or '2025–2026')."
    )
    await evaluator.verify(
        claim=claim_timeframe,
        node=timeframe_leaf,
        additional_instruction=(
            "Accept if the answer explicitly references 2025 or 2026 in connection with the rate, "
            "or states '2025–2026'. If no timeframe is stated or only older years are mentioned, this should be marked incorrect."
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
    Evaluate an answer for identifying the U.S. state with the highest effective property tax rate (2025–2026),
    and confirming the specific rate and timeframe details.

    Returns a structured evaluation summary with verification tree and extraction info.
    """
    # Initialize evaluator (root is a parallel aggregator by default)
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_property_tax_info(),
        template_class=PropertyTaxAnswerExtraction,
        extraction_name="property_tax_answer_extraction",
    )

    # Add ground truth info for context in summary
    evaluator.add_ground_truth(
        {
            "expected_state": GROUND_TRUTH["expected_state"],
            "expected_effective_rate": GROUND_TRUTH["expected_effective_rate"],
            "expected_timeframe": GROUND_TRUTH["expected_timeframe"],
        },
        gt_type="ground_truth",
    )

    # Optionally record the extracted fields as custom info for easier debugging
    evaluator.add_custom_info(
        info={
            "extracted_state": extraction.state,
            "extracted_rate_text": extraction.rate_text,
            "extracted_metric_text": extraction.metric_text,
            "extracted_timeframe_text": extraction.timeframe_text,
            "years_mentioned": extraction.years_mentioned,
            "source_urls": extraction.source_urls,
        },
        info_type="extraction_summary",
        info_name="extraction_overview",
    )

    # Build and run verification checks according to rubric
    await build_verification_tree(evaluator, root, extraction)

    # Return the final summary
    return evaluator.get_summary()