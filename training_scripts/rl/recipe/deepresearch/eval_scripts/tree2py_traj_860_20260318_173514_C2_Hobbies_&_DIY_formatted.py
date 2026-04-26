import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "hd_miter_saw_rental_requirements"
TASK_DESCRIPTION = (
    "What are the rental requirements at The Home Depot for renting a miter saw? "
    "Your answer should specifically include: (1) identification requirements, "
    "(2) payment method requirements, and (3) deposit amount information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SectionEvidence(BaseModel):
    """Holds a requirement statement and the URLs that support it (as cited in the answer)."""
    statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RequirementsExtraction(BaseModel):
    """Structured extraction of answer content for verification."""
    identification: Optional[SectionEvidence] = None  # ID requirement to rent a tool/miter saw
    payment_credit_card: Optional[SectionEvidence] = None  # Credit card required for deposit
    payment_cash_prohibition: Optional[SectionEvidence] = None  # Cash not accepted for deposit
    deposit_range: Optional[SectionEvidence] = None  # General tools deposit range ($25-$300)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract from the provided answer the specific claims and their cited URLs related to The Home Depot tool rental requirements
    (particularly applicable to renting a miter saw). For each category below, extract:
    - statement: The exact or closest paraphrase of what the answer claims for that category.
    - sources: All explicit URLs the answer provides to support that specific category. Include plain URLs and URLs inside markdown links.

    Categories to extract (return null for any category not addressed in the answer):
    1) identification:
       - statement: e.g., "Valid (government-issued) photo ID is required" or similar.
       - sources: URLs cited in the answer that support the ID requirement.
    2) payment_credit_card:
       - statement: e.g., "A credit card is required for the deposit."
       - sources: URLs cited in the answer that support this "credit card for deposit" policy.
    3) payment_cash_prohibition:
       - statement: e.g., "Cash is not accepted for the deposit."
       - sources: URLs cited in the answer that support this "no cash for deposit" policy.
    4) deposit_range:
       - statement: e.g., "General tools require a deposit ranging from $25 to $300."
       - sources: URLs cited in the answer that support the deposit range ($25–$300) for general/small tools.

    Important:
    - Do not invent URLs. Only include URLs explicitly present in the answer.
    - If a URL is missing a protocol, prepend http://
    - If a category is mentioned without any URL in the answer, set sources to an empty array.
    - Return a single JSON object with fields: identification, payment_credit_card, payment_cash_prohibition, deposit_range.
    """


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
    Evaluate an answer for The Home Depot miter saw rental requirements task.
    The evaluation enforces source-grounded verification for each requirement.
    """
    # 1) Initialize evaluator
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

    # 2) Extract structured information from the answer
    extracted: RequirementsExtraction = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=RequirementsExtraction,
        extraction_name="requirements_extraction",
    )

    # 3) Build verification tree according to rubric (with explicit source checks)
    #    Create a critical top-level (parallel) node under root to honor the rubric's "critical" root intent.
    main = evaluator.add_parallel(
        id="Tool_Rental_Requirements_Verification",
        desc="Verifies that all rental requirements for a miter saw from The Home Depot are correctly identified, including identification, payment methods, and deposit information",
        parent=root,
        critical=True,
    )

    # Helper accessors
    ident = extracted.identification or SectionEvidence()
    cc = extracted.payment_credit_card or SectionEvidence()
    cash = extracted.payment_cash_prohibition or SectionEvidence()
    depo = extracted.deposit_range or SectionEvidence()

    # ---------------- Identification Requirement ----------------
    # Source presence (critical; blocks verification if no URLs cited)
    evaluator.add_custom_node(
        result=bool(ident.sources),
        id="Identification_Sources_Provided",
        desc="Sources are provided in the answer for the identification requirement",
        parent=main,
        critical=True,
    )

    # Actual verification against URLs
    id_req_node = evaluator.add_leaf(
        id="Identification_Requirement",
        desc="Verifies that the answer states valid identification is required to rent a miter saw from The Home Depot",
        parent=main,
        critical=True,
    )
    id_claim = (
        "The Home Depot requires valid identification to rent tools (including a miter saw). "
        "Accept phrasings like 'valid (government-issued) photo ID' or 'driver's license' as equivalent."
    )
    await evaluator.verify(
        claim=id_claim,
        node=id_req_node,
        sources=ident.sources,
        additional_instruction=(
            "Use the provided URL(s) to confirm that customers must present valid ID to rent tools. "
            "It is acceptable if the policy is stated for general tool rentals rather than naming 'miter saw' explicitly."
        ),
    )

    # ---------------- Payment Method Requirements (parallel group) ----------------
    pay_group = evaluator.add_parallel(
        id="Payment_Method_Requirements",
        desc="Verifies that all payment method requirements are correctly stated",
        parent=main,
        critical=True,
    )

    # Credit card sources
    evaluator.add_custom_node(
        result=bool(cc.sources),
        id="Credit_Card_Sources_Provided",
        desc="Sources are provided in the answer for the 'credit card required for deposit' policy",
        parent=pay_group,
        critical=True,
    )

    # Credit card required for deposit
    cc_node = evaluator.add_leaf(
        id="Credit_Card_Requirement",
        desc="Verifies that the answer states a credit card is required for the deposit",
        parent=pay_group,
        critical=True,
    )
    cc_claim = (
        "A credit card is required for the deposit when renting tools from The Home Depot (e.g., a miter saw). "
        "Treat 'credit card hold' or 'credit card authorization' as equivalent to requiring a credit card for the deposit."
    )
    await evaluator.verify(
        claim=cc_claim,
        node=cc_node,
        sources=cc.sources,
        additional_instruction=(
            "Focus specifically on the deposit method. Some pages may allow other payment methods for rental charges, "
            "but the deposit itself must be on a credit card."
        ),
    )

    # Cash prohibition sources
    evaluator.add_custom_node(
        result=bool(cash.sources),
        id="Cash_Prohibition_Sources_Provided",
        desc="Sources are provided in the answer for the 'cash not accepted for deposits' policy",
        parent=pay_group,
        critical=True,
    )

    # Cash not accepted for deposits
    cash_node = evaluator.add_leaf(
        id="Cash_Prohibition",
        desc="Verifies that the answer states cash is not accepted for deposits",
        parent=pay_group,
        critical=True,
    )
    cash_claim = (
        "Cash is not accepted for tool rental deposits at The Home Depot. "
        "The deposit cannot be provided in cash."
    )
    await evaluator.verify(
        claim=cash_claim,
        node=cash_node,
        sources=cash.sources,
        additional_instruction=(
            "Confirm that the source explicitly states cash cannot be used for the deposit. "
            "Ignore statements about paying rental charges in cash; we are only checking the deposit method."
        ),
    )

    # ---------------- Deposit Amount Range ----------------
    # Source presence (critical)
    evaluator.add_custom_node(
        result=bool(depo.sources),
        id="Deposit_Sources_Provided",
        desc="Sources are provided in the answer for the deposit amount range",
        parent=main,
        critical=True,
    )

    deposit_node = evaluator.add_leaf(
        id="Deposit_Amount_Range",
        desc="Verifies that the answer provides the deposit amount range for general tools ($25-$300)",
        parent=main,
        critical=True,
    )
    deposit_claim = (
        "The deposit amount range for general (small) tools at The Home Depot is $25 to $300. "
        "Accept variants like '$25–$300' or '$25-$300'."
    )
    await evaluator.verify(
        claim=deposit_claim,
        node=deposit_node,
        sources=depo.sources,
        additional_instruction=(
            "Verify that the provided URL(s) explicitly state the deposit range for general/small tools is $25–$300. "
            "Do not confuse this with truck rental deposits or other categories."
        ),
    )

    # 4) Return standardized summary
    return evaluator.get_summary()