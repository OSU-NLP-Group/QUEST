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
TASK_ID = "verizon_outage_compensation_2026_01_14"
TASK_DESCRIPTION = "What compensation did Verizon offer to customers affected by the nationwide outage on January 14, 2026, and how can affected customers redeem this compensation?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OutageCompensationExtraction(BaseModel):
    """
    Structured extraction from the agent's answer:
    - compensation_amount_text: e.g., "$20 account credit", "a $20 bill credit"
    - redemption_method_text: e.g., "through the My Verizon app", "via myVerizon"
    - source_urls: URLs cited in the answer as evidence
    """
    compensation_amount_text: Optional[str] = None
    redemption_method_text: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_outage_compensation() -> str:
    return (
        "Extract the compensation and redemption information for Verizon's nationwide outage on January 14, 2026 as stated in the answer.\n"
        "Return a JSON object with the following fields:\n"
        "1) compensation_amount_text: The exact phrase used in the answer to describe the compensation amount (e.g., \"$20 account credit\", \"a $20 bill credit\"). "
        "Do not paraphrase—copy the phrase verbatim from the answer.\n"
        "2) redemption_method_text: The exact phrase used in the answer to describe how affected customers can redeem the compensation "
        "(e.g., \"through the My Verizon app\", \"via myVerizon\"). Do not paraphrase—copy verbatim.\n"
        "3) source_urls: An array of all URLs explicitly cited in the answer to support the information. "
        "Extract actual URLs even if they appear as markdown links. If the answer does not provide any URLs, return an empty array.\n"
        "If any of the requested fields are not clearly stated in the answer, set that field to null (or empty array for source_urls)."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_compensation_tree(
    evaluator: Evaluator,
    parent_node,
    extracted: OutageCompensationExtraction,
) -> None:
    """
    Build the verification sub-tree under the critical node and perform evidence-based checks.
    """

    # Create leaf nodes for the two critical checks under the critical parent
    amount_leaf = evaluator.add_leaf(
        id="CompensationAmount",
        desc="The answer states that Verizon offered a $20 account credit to affected customers",
        parent=parent_node,
        critical=True,
    )

    redemption_leaf = evaluator.add_leaf(
        id="RedemptionMethod",
        desc="The answer states that the credit can be redeemed through the myVerizon app",
        parent=parent_node,
        critical=True,
    )

    # Prepare claims based on extracted answer content
    comp_phrase = (extracted.compensation_amount_text or "").strip()
    red_phrase = (extracted.redemption_method_text or "").strip()
    sources = extracted.source_urls if extracted.source_urls else []

    # Claims are built from the answer text and verified against cited sources
    # Include context of the specific outage date
    comp_claim = (
        f"Following the nationwide outage on January 14, 2026, Verizon offered {comp_phrase} to affected customers."
        if comp_phrase else
        "Following the nationwide outage on January 14, 2026, Verizon offered a specific compensation to affected customers."
    )
    comp_instruction = (
        f"Verify whether the provided sources explicitly confirm the compensation amount described in the answer ('{comp_phrase}') "
        f"for the January 14, 2026 outage. Treat 'bill credit' and 'account credit' as equivalent naming for the same benefit, "
        f"but the dollar amount must match exactly."
    )

    red_claim = (
        f"Affected customers can redeem the compensation through {red_phrase}."
        if red_phrase else
        "Affected customers can redeem the compensation through the My Verizon app."
    )
    red_instruction = (
        f"Verify whether the provided sources explicitly confirm the redemption method described in the answer ('{red_phrase}'). "
        f"Accept variants like 'My Verizon app', 'My Verizon', or 'myVerizon app' as equivalent. "
        f"If sources indicate a different process (e.g., automatic credit without redemption or web form), mark as not supported."
    )

    # Execute verifications (in parallel where possible)
    await evaluator.batch_verify([
        (comp_claim, sources, amount_leaf, comp_instruction),
        (red_claim, sources, redemption_leaf, red_instruction),
    ])


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
    Evaluate an answer to the Verizon outage compensation task.
    """

    # Initialize evaluator (root is non-critical by design; create a critical child node for the rubric root)
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

    # Create the rubric's critical root node under the evaluator's root
    rubric_root = evaluator.add_parallel(
        id="VerizonOutageCompensation",
        desc="Evaluates whether the answer correctly identifies both the compensation amount and redemption method for customers affected by the Verizon outage on January 14, 2026",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_outage_compensation(),
        template_class=OutageCompensationExtraction,
        extraction_name="outage_compensation_extraction"
    )

    # Build verification tree and run checks
    await build_and_verify_compensation_tree(
        evaluator=evaluator,
        parent_node=rubric_root,
        extracted=extracted
    )

    # Return final structured summary
    return evaluator.get_summary()