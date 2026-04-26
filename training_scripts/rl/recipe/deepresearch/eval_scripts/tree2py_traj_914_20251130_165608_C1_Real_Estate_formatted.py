import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "down_payment_affordability_2025"
TASK_DESCRIPTION = (
    "A real estate investor has $50,000 available in savings to use as a down payment for purchasing an investment property. "
    "According to current lending standards, conventional loans for investment properties require a minimum down payment of 15% of the purchase price.\n\n"
    "Based on the following median home prices for 2025:\n"
    "- Mississippi: $186,446 (average home value per Zillow)\n"
    "- Texas: $297,000 (average home value per Zillow)\n"
    "- Florida: $374,697 (average home value per Zillow)\n\n"
    "In which of these three states can the investor afford to meet the minimum 15% down payment requirement for an investment property purchased at the state's median home price? List all states that qualify."
)

# State prices and affordability parameters
STATE_PRICES = {
    "Mississippi": 186_446.0,
    "Texas": 297_000.0,
    "Florida": 374_697.0,
}
DOWN_PAYMENT_RATE = 0.15
BUDGET = 50_000.0


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def compute_required_down_payment(price: float) -> float:
    return round(price * DOWN_PAYMENT_RATE, 2)


def format_currency(amount: float) -> str:
    return f"${amount:,.2f}"


def normalize_state_name(name: str) -> Optional[str]:
    """Normalize various forms to canonical state names."""
    if not name:
        return None
    s = name.strip().lower()
    if s in {"mississippi", "ms", "miss.", "miss"}:
        return "Mississippi"
    if s in {"texas", "tx", "tex.", "tex"}:
        return "Texas"
    if s in {"florida", "fl", "fla.", "fla"}:
        return "Florida"
    return None


def expected_affordability_map() -> Dict[str, bool]:
    """Compute affordability truth for each state based on given parameters."""
    return {
        state: compute_required_down_payment(price) <= BUDGET
        for state, price in STATE_PRICES.items()
    }


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class QualifyingStatesExtraction(BaseModel):
    """Extracted list of states (from Mississippi, Texas, Florida) the answer claims as qualifying."""
    states: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_qualifying_states() -> str:
    return (
        "From the answer, extract which of the three states (Mississippi, Texas, Florida) are claimed to qualify under the "
        "$50,000 budget and 15% minimum down payment rule.\n"
        "Return a JSON object with one field:\n"
        "- states: an array listing the qualifying states using exactly these canonical names: "
        "\"Mississippi\", \"Texas\", \"Florida\".\n"
        "Mapping rules:\n"
        "- If the answer uses abbreviations (e.g., MS, TX, FL) or nicknames, convert them to the canonical names.\n"
        "- Extract only among these three states. Ignore any other states mentioned.\n"
        "- If the answer implies 'none' qualify, return an empty array.\n"
        "- If the answer implies 'all three' qualify, return all three canonical names in the array.\n"
        "Do not invent information; extract only what the answer asserts."
    )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_state_affordability(
    evaluator: Evaluator,
    parent_node,
    state: str,
    extracted_states: List[str],
) -> None:
    """
    Build verification sub-tree for a single state:
    - A simple math correctness claim (non-critical).
    - A critical check that the agent's listed qualifying states correctly include/exclude this state.
    """
    # Compute affordability truth
    price = STATE_PRICES[state]
    required_dp = compute_required_down_payment(price)
    affordable = required_dp <= BUDGET

    # Create per-state node (non-critical, parallel)
    state_node = evaluator.add_parallel(
        id=f"{state}_Affordability",
        desc=(
            f"Uses {state} median price and the 15% minimum down payment rule to determine affordability "
            "and whether the state should be in the qualifying list."
        ),
        parent=parent_node,
        critical=False,
    )

    # Leaf 1: Math correctness verification (non-critical)
    math_leaf = evaluator.add_leaf(
        id=f"{state}_Math_Check",
        desc=(
            f"{state}: At the 2025 median home price {format_currency(price)}, the 15% down payment is "
            f"{format_currency(required_dp)}, which is "
            f"{'less than or equal to' if affordable else 'greater than'} {format_currency(BUDGET)}."
        ),
        parent=state_node,
        critical=False,
    )

    claim = (
        f"For {state}, 15% of {format_currency(price)} is {format_currency(required_dp)}, which is "
        f"{'<= ' if affordable else '> '}{format_currency(BUDGET)}."
    )
    await evaluator.verify(
        claim=claim,
        node=math_leaf,
        additional_instruction=(
            "Compute 0.15 × price and compare to $50,000. Allow rounding to the nearest cent. "
            "Confirm the inequality direction in the claim."
        ),
    )

    # Leaf 2: Agent's inclusion/exclusion correctness (critical)
    # Normalize extracted states to canonical names for robust comparison
    normalized_extracted = []
    for s in extracted_states:
        canon = normalize_state_name(s)
        if canon:
            normalized_extracted.append(canon)
    extracted_set = set(normalized_extracted)

    # Expected presence in the agent's list
    expected_present = affordable
    actually_present = state in extracted_set
    result = (actually_present == expected_present)

    evaluator.add_custom_node(
        result=result,
        id=f"{state}_Conclusion_Correct",
        desc=(
            f"{state}: Agent {'correctly' if result else 'incorrectly'} "
            f"{'includes' if actually_present else 'excludes'} {state} in the qualifying-state list "
            f"given the 15% rule and {state}'s median price."
        ),
        parent=state_node,
        critical=True,
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
    Evaluate the agent's answer for the 2025 investment-property down payment affordability task.
    """
    # Initialize evaluator and root node
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

    # Add top-level assessment node
    assessment_node = evaluator.add_parallel(
        id="Affordability_Assessment",
        desc="Evaluate whether $50,000 is sufficient to meet a 15% minimum down payment at each state's given 2025 median home price, and list all qualifying states.",
        parent=root,
        critical=False,
    )

    # Extract states the answer claims as qualifying
    extraction = await evaluator.extract(
        prompt=prompt_extract_qualifying_states(),
        template_class=QualifyingStatesExtraction,
        extraction_name="qualifying_states",
    )

    # Compute ground-truth affordability and expected qualifying list
    truth_map = expected_affordability_map()
    expected_states = [state for state, ok in truth_map.items() if ok]

    # Record ground truth info and computed amounts
    evaluator.add_ground_truth({
        "down_payment_rate": DOWN_PAYMENT_RATE,
        "budget": BUDGET,
        "state_prices_2025": STATE_PRICES,
        "required_down_payments": {
            state: compute_required_down_payment(price) for state, price in STATE_PRICES.items()
        },
        "affordability_truth": truth_map,
        "expected_qualifying_states": expected_states,
    })

    # Also record the extracted states as custom info
    evaluator.add_custom_info(
        info={"extracted_qualifying_states": extraction.states},
        info_type="extraction_summary",
        info_name="agent_claimed_states",
    )

    # Build per-state verification nodes
    for state in ["Mississippi", "Texas", "Florida"]:
        await verify_state_affordability(
            evaluator=evaluator,
            parent_node=assessment_node,
            state=state,
            extracted_states=extraction.states or [],
        )

    # Return standardized summary
    return evaluator.get_summary()