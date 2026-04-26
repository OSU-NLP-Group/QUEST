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
TASK_ID = "fha_loan_lowest_down_payment_2026"
TASK_DESCRIPTION = (
    "What is the minimum credit score required to qualify for an FHA loan in 2026 "
    "with the lowest possible down payment, and what is that minimum down payment percentage?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FHARequirementsExtraction(BaseModel):
    """
    Extract the agent's stated FHA minimum credit score and minimum down payment for the
    lowest down payment option, along with any URLs cited for each claim.
    """
    min_credit_score: Optional[str] = None
    min_down_payment_percent: Optional[str] = None
    credit_score_sources: List[str] = Field(default_factory=list)
    down_payment_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_fha_requirements() -> str:
    return (
        "Extract the FHA loan requirements stated in the answer specifically for the lowest possible down payment option.\n"
        "Return a JSON object with the following fields:\n"
        "1) min_credit_score: The minimum credit score the answer states is required to qualify for the lowest FHA down payment option (typically the 3.5% down tier). "
        "If multiple numbers are mentioned, select the one tied to qualifying for the lowest down payment option.\n"
        "2) min_down_payment_percent: The minimum down payment percentage the answer states applies to borrowers who meet the minimum credit score threshold (for FHA, typically 580+). "
        "Preserve formatting such as '%' or 'percent' if present.\n"
        "3) credit_score_sources: An array of URLs that the answer cites for the credit score requirement.\n"
        "4) down_payment_sources: An array of URLs that the answer cites for the down payment percentage.\n"
        "If any item is not present in the answer, set it to null (for strings) or an empty array (for URL arrays).\n"
        "Extract only URLs explicitly shown in the answer (including markdown links)."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(extracted: FHARequirementsExtraction) -> List[str]:
    """Combine and deduplicate all sources from both claims."""
    combined = list({*(extracted.credit_score_sources or []), *(extracted.down_payment_sources or [])})
    return combined


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extracted: FHARequirementsExtraction
) -> None:
    """
    Build the verification tree following the rubric:
    Root critical parallel node with two critical leaf checks:
      - Minimum credit score requirement for lowest down payment tier
      - Minimum down payment percentage for borrowers with 580+ credit score
    """
    # Create the rubric root node (critical, parallel aggregation)
    fha_root = evaluator.add_parallel(
        id="FHA_Loan_Requirements_2026",
        desc="Verify the required FHA loan minimum credit score and minimum down payment for the lowest down payment option in 2026.",
        parent=evaluator.root,
        critical=True
    )

    # Prepare sources
    all_sources = combine_sources(extracted)
    credit_sources = extracted.credit_score_sources or all_sources
    down_payment_sources = extracted.down_payment_sources or all_sources

    # Leaf 1: Minimum Credit Score Requirement
    leaf_credit = evaluator.add_leaf(
        id="Minimum_Credit_Score_Requirement",
        desc="States that the minimum credit score to qualify for the lowest down payment FHA option (3.5%) is 580.",
        parent=fha_root,
        critical=True
    )

    # Formulate claim using the agent's stated value (if any) to ensure we are judging the answer's claim against evidence.
    stated_credit = extracted.min_credit_score or "unknown"
    claim_credit = (
        f"The minimum credit score to qualify for the lowest FHA down payment option (3.5%) is {stated_credit}."
    )
    additional_instruction_credit = (
        "Verify whether the provided URL(s) explicitly support the statement for FHA loans as of 2026. "
        "For FHA, borrowers with credit scores of 580 or higher typically qualify for the minimum 3.5% down payment. "
        "Equivalent phrasing such as '580+' or 'minimum 580' should be treated as the same. "
        "If the URLs indicate that 500–579 require 10% down, that implies 580+ for 3.5%, which supports the statement. "
        "If no URLs are provided, judge based on the statement itself but be conservative."
    )
    await evaluator.verify(
        claim=claim_credit,
        node=leaf_credit,
        sources=credit_sources if credit_sources else None,
        additional_instruction=additional_instruction_credit
    )

    # Leaf 2: Minimum Down Payment Percentage
    leaf_down_payment = evaluator.add_leaf(
        id="Minimum_Down_Payment_Percentage",
        desc="States that the minimum down payment percentage for borrowers with a credit score of 580 or higher is 3.5% of the purchase price.",
        parent=fha_root,
        critical=True
    )

    stated_down_payment = extracted.min_down_payment_percent or "unknown"
    claim_down_payment = (
        f"The minimum down payment percentage for FHA borrowers with a credit score of 580 or higher is {stated_down_payment} of the purchase price."
    )
    additional_instruction_down = (
        "Verify whether the provided URL(s) explicitly state that borrowers with credit scores of 580 or higher qualify for a 3.5% minimum down payment on FHA loans (as of 2026). "
        "Equivalent phrasing such as '3.5 percent' or 'minimum 3.5%' should be treated as the same. "
        "If the sources show 3.5% at 580+, consider that supportive. If no URLs are provided, judge based on the statement itself but be conservative."
    )
    await evaluator.verify(
        claim=claim_down_payment,
        node=leaf_down_payment,
        sources=down_payment_sources if down_payment_sources else None,
        additional_instruction=additional_instruction_down
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
    Evaluate an answer for FHA loan requirements in 2026 (lowest down payment tier).
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator is parallel by rubric
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

    # Extract the agent's stated values and sources
    extracted = await evaluator.extract(
        prompt=prompt_extract_fha_requirements(),
        template_class=FHARequirementsExtraction,
        extraction_name="fha_requirements_extraction"
    )

    # Add ground truth context (for reporting only; verification is evidence-based)
    evaluator.add_ground_truth({
        "expected_min_credit_score_for_3_5_percent_down": "580",
        "expected_min_down_payment_percent_at_580_plus": "3.5%"
    }, gt_type="ground_truth_fha_2026")

    # Build and run verification tree
    await build_verification_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()