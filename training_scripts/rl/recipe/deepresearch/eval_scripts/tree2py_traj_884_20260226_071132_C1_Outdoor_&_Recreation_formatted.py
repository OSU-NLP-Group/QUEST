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
TASK_ID = "tsa_confirmid_feb2026"
TASK_DESCRIPTION = (
    "Starting in February 2026, what paid alternative identification verification option does the Transportation "
    "Security Administration (TSA) offer to travelers at domestic airport security checkpoints who do not have a REAL ID "
    "or other acceptable form of identification? Provide the official name of this program, the fee amount, and the "
    "specific date this policy took effect."
)

# Optional ground-truth expectations (for reporting only; verification is evidence-based)
GROUND_TRUTH = {
    "expected_program_name": "TSA ConfirmID",
    "expected_fee_amount": "$45",
    "expected_effective_date": "February 1, 2026",
    "expected_context": "For travelers at domestic airport TSA security checkpoints who do not have a REAL ID-compliant driver’s license or other acceptable form of identification."
}

# --------------------------------------------------------------------------- #
# Data model for extraction                                                   #
# --------------------------------------------------------------------------- #
class TSAProgramExtraction(BaseModel):
    """Structured data extracted from the agent's answer."""
    program_name: Optional[str] = None
    fee_amount: Optional[str] = None
    effective_date: Optional[str] = None
    context_statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tsa_program_info() -> str:
    return """
    Extract the information the answer provides about TSA's paid alternative identification verification option starting in February 2026.
    Return a JSON object with these fields:
      - program_name: The official name of the TSA program/option as written in the answer (e.g., "TSA ConfirmID"). If not explicitly named, return null.
      - fee_amount: The fee amount as written in the answer (e.g., "$45", "45 dollars"). If not provided, return null.
      - effective_date: The specific date the policy took effect as written in the answer (e.g., "February 1, 2026"). If only a month/year is provided without a specific date, extract that text; if absent, return null.
      - context_statement: A concise sentence from the answer describing who this option is for and where (e.g., for travelers at domestic airport TSA security checkpoints who do not have a REAL ID or other acceptable identification). If the answer lacks this, return null.
      - sources: An array of all URLs the answer cites that support any of the above information. Extract actual URLs (including markdown link targets). If the answer provides no URLs, return an empty array.

    Rules:
    - Do not invent or infer values; extract exactly what the answer states.
    - Preserve formatting for names, dates, and fees (e.g., keep "$45" if present).
    - Include every URL that appears to support this program, its fee, effective date, or who/where it applies.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_text(x: Optional[str], default: str = "") -> str:
    return (x or "").strip() or default


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tsa_nodes(
    evaluator: Evaluator,
    root: Any,
    extracted: TSAProgramExtraction
) -> None:
    """
    Build the verification tree under a critical parent node and run four critical leaf checks.
    """
    parent_node = evaluator.add_parallel(
        id="TSA_Paid_Alternative_ID_Verification_Option",
        desc=("Answer identifies TSA's paid alternative identification verification option starting in February 2026 "
              "for domestic airport checkpoints for travelers without REAL ID/acceptable ID, and provides the requested details."),
        parent=root,
        critical=True  # All children must be critical under this parent
    )

    # Normalize extracted values
    prog_name = _safe_text(extracted.program_name, default="")
    fee_amt = _safe_text(extracted.fee_amount, default="")
    eff_date = _safe_text(extracted.effective_date, default="")
    ctx_stmt = _safe_text(extracted.context_statement, default="")
    sources_list = extracted.sources or []

    # 1) Eligibility and Context
    # Claim focuses on applicability: domestic TSA checkpoints, for travelers without REAL ID or other acceptable ID.
    node_context = evaluator.add_leaf(
        id="Eligibility_and_Context",
        desc=("States the option is for travelers at domestic airport TSA security checkpoints who do not have a REAL ID-compliant "
              "driver's license or other acceptable form of identification."),
        parent=parent_node,
        critical=True
    )

    # Build a claim leveraging the program name if available
    prog_phrase = prog_name if prog_name else "the paid TSA alternative identification verification option"
    claim_context = (
        f"{prog_phrase} is offered at domestic airport TSA security checkpoints and is intended for travelers who do not have "
        f"a REAL ID-compliant driver's license or other acceptable form of identification."
    )

    await evaluator.verify(
        claim=claim_context,
        node=node_context,
        sources=sources_list,
        additional_instruction=(
            "Verify the scope and eligibility: The option must be provided at domestic airport TSA security checkpoints, "
            "and explicitly for travelers lacking a REAL ID-compliant driver's license or other acceptable ID. "
            "Use the cited URLs. If no valid URLs are provided, mark this claim as not supported."
        ),
    )

    # 2) Official Program Name
    node_prog_name = evaluator.add_leaf(
        id="Official_Program_Name",
        desc="Provides the official name of the TSA program/option (TSA ConfirmID).",
        parent=parent_node,
        critical=True
    )

    claim_prog_name = (
        f"The official name of TSA's paid alternative identification verification option is '{prog_name}'."
        if prog_name else
        "TSA's paid alternative identification verification option has an official name explicitly stated by TSA."
    )

    await evaluator.verify(
        claim=claim_prog_name,
        node=node_prog_name,
        sources=sources_list,
        additional_instruction=(
            "Confirm the official branding as shown on TSA's webpage or press release. The claim should match the exact "
            "name used by TSA (allowing minor whitespace or casing variations only if clearly equivalent). "
            "If no URL sources are provided, mark as not supported."
        ),
    )

    # 3) Fee Amount
    node_fee = evaluator.add_leaf(
        id="Fee_Amount",
        desc="Provides the fee amount for using the paid option ($45).",
        parent=parent_node,
        critical=True
    )

    claim_fee = (
        f"The fee amount to use {prog_phrase} is '{fee_amt}'."
        if fee_amt else
        f"{prog_phrase} requires payment of a specific fee amount."
    )

    await evaluator.verify(
        claim=claim_fee,
        node=node_fee,
        sources=sources_list,
        additional_instruction=(
            "Verify the exact fee amount as stated on the cited sources (e.g., '$45', '45 dollars'). "
            "If the claim omits a fee value or URLs are missing, mark as not supported."
        ),
    )

    # 4) Effective Date
    node_date = evaluator.add_leaf(
        id="Effective_Date",
        desc="Provides the specific policy effective date (February 1, 2026).",
        parent=parent_node,
        critical=True
    )

    claim_date = (
        f"The policy took effect on '{eff_date}'."
        if eff_date else
        "The policy has a specific effective date in February 2026."
    )

    await evaluator.verify(
        claim=claim_date,
        node=node_date,
        sources=sources_list,
        additional_instruction=(
            "Confirm the exact effective date on the cited sources. Prefer a fully specified date (Month Day, Year), "
            "e.g., 'February 1, 2026'. If only a vague timeframe is provided or no URLs are cited, mark as not supported."
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
    Evaluate the agent's answer for the TSA paid alternative identification verification option task.
    """
    # Initialize evaluator
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

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_tsa_program_info(),
        template_class=TSAProgramExtraction,
        extraction_name="tsa_paid_alt_id_option"
    )

    # Record ground truth expectations for reporting
    evaluator.add_ground_truth(GROUND_TRUTH, gt_type="expected_policy_details")

    # Build and verify nodes
    await build_and_verify_tsa_nodes(evaluator, root, extracted)

    # Return summarized evaluation result
    return evaluator.get_summary()