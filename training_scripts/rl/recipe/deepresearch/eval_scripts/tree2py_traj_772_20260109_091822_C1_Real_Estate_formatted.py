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
TASK_ID = "wi_salesperson_prelicense_hours"
TASK_DESCRIPTION = (
    "What is the minimum number of pre-licensing education hours required to qualify for a real estate salesperson license in Wisconsin?"
)

EXPECTED_HOURS = "72 hours"
EXPECTED_PROVIDER_REQUIREMENT = "DSPS-approved provider"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WiSalespersonRequirementsExtraction(BaseModel):
    """
    Extraction of what the answer states about WI salesperson pre-licensing requirements.
    """
    hours_mentioned: Optional[str] = None
    mentions_dsps_approved_provider: Optional[bool] = None
    dsps_phrase: Optional[str] = None
    cited_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    From the answer text, extract what it explicitly states about Wisconsin real estate salesperson pre-licensing education.

    Return a JSON object with:
    - hours_mentioned: The minimum number of pre-licensing education hours stated (extract exactly as written; examples: "72 hours", "72-hr", "seventy-two hours"). If none is stated, return null.
    - mentions_dsps_approved_provider: true if and only if the answer explicitly mentions that the pre-licensing education must be from a DSPS-approved provider or uses an equivalent phrase (e.g., "approved by the Wisconsin DSPS", "state-approved provider" clearly referring to Wisconsin DSPS); otherwise return null (do not infer).
    - dsps_phrase: The exact phrase used in the answer that indicates DSPS/state approval, if any; otherwise null.
    - cited_urls: All URLs present in the answer (if any). If none, return an empty list.

    Follow the rules strictly and do not infer anything not explicitly stated in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_wi_salesperson_requirements(
    evaluator: Evaluator,
    parent_node,
    extraction: WiSalespersonRequirementsExtraction,
) -> None:
    """
    Build and execute verification checks according to the rubric:
    - Pre_Licensing_Education_Hours: The answer states that 72 hours are required.
    - DSPS_Approved_Provider: The answer states that the education must be from a DSPS-approved provider.
    """

    # Leaf 1: The answer states 72 hours are required
    hours_node = evaluator.add_leaf(
        id="Pre_Licensing_Education_Hours",
        desc="The answer states that 72 hours of pre-licensing education is required.",
        parent=parent_node,
        critical=True,
    )
    claim_hours = (
        "The answer explicitly states that the minimum pre-licensing education requirement for a Wisconsin real estate "
        "salesperson license is 72 hours."
    )
    add_ins_hours = (
        "Judge ONLY based on the answer text. Accept reasonable variants like '72 hr', '72-hr', or 'seventy-two hours'. "
        "If the answer is ambiguous, mentions a different number, or does not specify hours, mark this as Incorrect."
    )

    # Leaf 2: The answer states DSPS-approved provider is required
    dsps_node = evaluator.add_leaf(
        id="DSPS_Approved_Provider",
        desc="The answer states that the pre-licensing education must be from a DSPS-approved provider.",
        parent=parent_node,
        critical=True,
    )
    claim_dsps = (
        "The answer explicitly states that the required pre-licensing education must be from a provider approved by the "
        "Wisconsin Department of Safety and Professional Services (DSPS), i.e., a DSPS-approved or state-approved provider."
    )
    add_ins_dsps = (
        "Judge ONLY based on the answer text. Accept synonymous phrasing like 'DSPS-approved', "
        "'approved by Wisconsin DSPS', or 'state-approved provider' clearly referring to Wisconsin DSPS. "
        "If not clearly stated, mark as Incorrect."
    )

    # Execute both verifications in parallel (root is parallel)
    await evaluator.batch_verify(
        [
            (claim_hours, None, hours_node, add_ins_hours),
            (claim_dsps, None, dsps_node, add_ins_dsps),
        ]
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
    Evaluate an answer for the Wisconsin real estate salesperson pre-licensing education question.
    """
    # Initialize evaluator with a parallel root to match rubric aggregation
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

    # Record ground truth context (for transparency, not used directly in scoring)
    evaluator.add_ground_truth(
        {
            "expected_minimum_hours": EXPECTED_HOURS,
            "expected_provider_requirement": EXPECTED_PROVIDER_REQUIREMENT,
        },
        gt_type="ground_truth_requirements",
    )

    # Extract structured info from the answer (for summary/debugging)
    extraction = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=WiSalespersonRequirementsExtraction,
        extraction_name="wi_salesperson_requirements_extraction",
    )

    # Build and run verifications as per rubric
    await verify_wi_salesperson_requirements(evaluator, root, extraction)

    # Return structured summary
    return evaluator.get_summary()