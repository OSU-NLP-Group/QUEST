import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "passport_renewal_cost_2025_exp_12day"
TASK_DESCRIPTION = """
What is the total cost in U.S. dollars for an adult to renew their passport book by mail using Form DS-82 with expedited processing and 1-2 day delivery service in 2025?
""".strip()

# Ground truth fee components per rubric
EXPECTED_BASE_RENEWAL_FEE = 130.00
EXPECTED_EXPEDITE_FEE = 60.00
EXPECTED_12DAY_DELIVERY_FEE = 22.05
EXPECTED_TOTAL = EXPECTED_BASE_RENEWAL_FEE + EXPECTED_EXPEDITE_FEE + EXPECTED_12DAY_DELIVERY_FEE


# --------------------------------------------------------------------------- #
# Extraction model                                                            #
# --------------------------------------------------------------------------- #
class PassportCostExtraction(BaseModel):
    """
    Extract any cost numbers the answer explicitly mentions for the specified scenario.
    All fields are strings to maximize robustness to different formatting (e.g., '$130', 'USD 130', '130.00').
    """
    base_fee: Optional[str] = None
    expedite_fee: Optional[str] = None
    delivery_fee_1_2_day: Optional[str] = None
    total_cost: Optional[str] = None
    # Optional raw lines or notes (free-form, helpful for debugging)
    breakdown_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cost_breakdown() -> str:
    return """
    You are given an answer to the question about the total cost for an adult to renew a passport book by mail using Form DS‑82 with expedited processing and 1–2 day delivery service (for 2025).
    Extract the cost breakdown only if it is explicitly present in the answer. Return the following fields:

    - base_fee: The base adult passport book renewal fee mentioned in the answer (string). If not present, return null.
    - expedite_fee: The expedited processing fee mentioned (string). If not present, return null.
    - delivery_fee_1_2_day: The 1–2 day delivery (Priority Mail Express) fee mentioned (string). If not present, return null.
    - total_cost: The final total cost mentioned (string). If not present, return null.
    - breakdown_text: The exact cost breakdown lines or sentences copied verbatim from the answer that mention these fees. If not available, return null.

    Guidelines:
    - Extract only values explicitly provided in the answer. Do not infer or compute any numbers yourself.
    - Preserve the formatting of money values as shown (e.g., "$130", "$130.00", "USD 130").
    - If the answer does not explicitly list a component, set that field to null.
    - If multiple values for the same component are present, pick the most relevant one for the renewal-by-mail DS‑82 scenario with expedited processing and 1–2 day delivery.
    """


# --------------------------------------------------------------------------- #
# Verification helper                                                         #
# --------------------------------------------------------------------------- #
def _fmt_usd(amount: float) -> str:
    # Use $xxx or $xxx.xx depending on cents
    return f"${amount:,.2f}".rstrip('0').rstrip('.') if amount % 1 != 0 else f"${int(amount):,}"


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    Evaluate whether the answer correctly includes each required fee component for the
    adult passport book renewal by mail (Form DS-82) with expedited processing and 1–2 day delivery in 2025.

    The rubric requires three critical checks:
      - Base adult passport book renewal fee of $130 is included
      - Expedited processing fee of $60 is included
      - 1–2 day delivery fee of $22.05 is included
    """
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

    # Extraction (helpful for debugging; verification does not strictly depend on these values)
    extracted = await evaluator.extract(
        prompt=prompt_extract_cost_breakdown(),
        template_class=PassportCostExtraction,
        extraction_name="cost_breakdown_extraction",
    )

    # Record ground truth components for transparency
    evaluator.add_ground_truth({
        "expected_components": {
            "base_fee": _fmt_usd(EXPECTED_BASE_RENEWAL_FEE),
            "expedite_fee": _fmt_usd(EXPECTED_EXPEDITE_FEE),
            "delivery_fee_1_2_day": _fmt_usd(EXPECTED_12DAY_DELIVERY_FEE),
            "expected_total": _fmt_usd(EXPECTED_TOTAL),
        },
        "scenario": "Adult passport book renewal by mail (Form DS-82) with expedited processing and 1–2 day delivery, 2025",
    })

    # Build rubric tree: Total_Cost_Accuracy (critical, parallel) with 3 critical leaves
    total_cost_node = evaluator.add_parallel(
        id="Total_Cost_Accuracy",
        desc="Verifies that the total cost for expedited adult passport book renewal with 1-2 day delivery is correctly calculated",
        parent=root,
        critical=True,
    )

    # Leaf: Base_Renewal_Fee
    base_leaf = evaluator.add_leaf(
        id="Base_Renewal_Fee",
        desc="The base adult passport book renewal fee of $130 is included",
        parent=total_cost_node,
        critical=True,
    )
    # Leaf: Expedite_Fee
    expedite_leaf = evaluator.add_leaf(
        id="Expedite_Fee",
        desc="The expedited processing fee of $60 is included",
        parent=total_cost_node,
        critical=True,
    )
    # Leaf: Delivery_Fee
    delivery_leaf = evaluator.add_leaf(
        id="Delivery_Fee",
        desc="The 1-2 day delivery fee of $22.05 is included",
        parent=total_cost_node,
        critical=True,
    )

    common_instruction = (
        "Judge solely based on the provided answer text for the scenario: adult passport book renewal by mail (Form DS-82) "
        "with expedited processing and 1–2 day delivery in 2025. Consider the check passed if the answer explicitly lists or "
        "clearly uses the specified fee in its breakdown/calculation. Accept minor formatting variations in money values "
        "($, USD, US$; $130 vs $130.00) and in wording for '1–2 day' (e.g., '1 to 2 day', '1–2 day', '1-2 day')."
    )

    claims_and_sources = [
        (
            f"The answer explicitly includes a base adult passport book renewal fee of {_fmt_usd(EXPECTED_BASE_RENEWAL_FEE)}.",
            None,
            base_leaf,
            common_instruction
        ),
        (
            f"The answer explicitly includes an expedited processing fee of {_fmt_usd(EXPECTED_EXPEDITE_FEE)}.",
            None,
            expedite_leaf,
            common_instruction
        ),
        (
            f"The answer explicitly includes a 1–2 day delivery fee of {_fmt_usd(EXPECTED_12DAY_DELIVERY_FEE)}.",
            None,
            delivery_leaf,
            common_instruction
        ),
    ]

    # Run the three verifications in parallel
    await evaluator.batch_verify(claims_and_sources)

    return evaluator.get_summary()