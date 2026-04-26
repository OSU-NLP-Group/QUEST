import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_cpa_cpe_min_annual_hours"
TASK_DESCRIPTION = """
What is the minimum number of continuing professional education (CPE) hours that a licensed CPA in California must complete annually to maintain their license?
"""


# --------------------------------------------------------------------------- #
# Extraction model                                                            #
# --------------------------------------------------------------------------- #
class CPACaliforniaCPEExtraction(BaseModel):
    """
    Extracts how the answer states the California CPA minimum annual CPE requirement.
    """
    min_annual_hours_text: Optional[str] = None  # e.g., "20 hours per year", "minimum 20 annually"
    min_annual_hours_number: Optional[str] = None  # e.g., "20"
    sources: List[str] = Field(default_factory=list)  # any URLs cited in the answer related to California CPE


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cpe_info() -> str:
    return """
    From the answer text, extract how it states the minimum annual CPE hours for a California-licensed CPA.

    Return:
    - min_annual_hours_text: The exact phrase or sentence indicating the minimum CPE hours required each year in California (e.g., "minimum 20 hours per year"). If not explicitly stated, return null.
    - min_annual_hours_number: The numeric value of the minimum hours annually if explicitly given (e.g., "20"). If not explicitly stated, return null.
    - sources: Any URLs included in the answer that are cited to support CPE requirements for California. Extract actual URLs only. If none are present, return an empty list.

    Notes:
    - Focus on California and "per year"/"annually". If the answer mentions a biennial total (e.g., 80 hours every two years) BUT also mentions a minimum per year (e.g., "at least 20 per year"), still capture the per-year minimum.
    - If only a biennial total is given and no annual minimum is explicitly stated, set min_annual_hours_text and min_annual_hours_number to null.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extraction: CPACaliforniaCPEExtraction,
) -> None:
    """
    Build the verification tree per rubric and run the checks.
    Rubric:
      - Parent node (critical, parallel): "Correctly identifies the minimum annual CPE hours required for a CA CPA."
      - Leaf node (critical): "States that the minimum annual CPE requirement ... is 20 hours per year."
    """
    # Parent node matching rubric root (critical, parallel)
    parent_node = evaluator.add_parallel(
        id="Minimum_Annual_CPE_Hours_California_CPA",
        desc="Correctly identifies the minimum annual CPE hours required for a licensed CPA in California.",
        parent=evaluator.root,
        critical=True,
    )

    # Single critical leaf per rubric
    leaf_node = evaluator.add_leaf(
        id="Minimum_Annual_CPE_Is_20_Hours",
        desc="States that the minimum annual CPE requirement for a California-licensed CPA is 20 hours per year.",
        parent=parent_node,
        critical=True,
    )

    # We verify whether the answer itself explicitly asserts "20 hours per year" (or equivalent wording).
    # We do NOT fact-check against external sources here (rubric does not require it).
    claim = (
        "According to the provided answer text, the minimum annual continuing professional education (CPE) requirement "
        "for a California-licensed CPA is 20 hours per year (accept equivalent phrasings like '20 hrs annually', "
        "'minimum 20 hours per year', etc.). "
        "Judge solely based on what the answer states."
    )

    await evaluator.verify(
        claim=claim,
        node=leaf_node,
        additional_instruction=(
            "Your task is to check whether the statement is explicitly asserted in the answer. "
            "Do not rely on external knowledge or fact-check the number; only determine if the answer claims "
            "a 20-hours-per-year minimum for California. "
            "Accept minor textual variations (e.g., '20 hrs/year', 'at least 20 per year'). "
            "If the answer only mentions a biennial total (e.g., 80 hours every two years) without stating "
            "a per-year minimum of 20, mark as incorrect."
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
    Evaluate an answer for the California CPA minimum annual CPE requirement task.
    """
    # Initialize evaluator (root is a non-critical wrapper node created internally)
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall aggregation strategy for the wrapper root
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

    # Extraction step (records structured info from the answer)
    extraction = await evaluator.extract(
        prompt=prompt_extract_cpe_info(),
        template_class=CPACaliforniaCPEExtraction,
        extraction_name="cpa_ca_cpe_extraction",
    )

    # Optional: record ground truth info for transparency (not used to auto-grade)
    evaluator.add_ground_truth(
        {
            "expected_min_annual_hours": "20",
            "expected_text": "20 hours per year",
            "jurisdiction": "California",
            "note": "Commonly, CA requires 80 hours per two-year period with a minimum of 20 per year.",
        },
        gt_type="reference_info",
    )

    # Build and verify according to rubric
    await build_and_verify_tree(evaluator, extraction)

    # Return the structured evaluation summary
    return evaluator.get_summary()