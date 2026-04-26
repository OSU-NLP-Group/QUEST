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
TASK_ID = "costco_executive_upgrade_4200"
TASK_DESCRIPTION = (
    "A family currently has a standard Costco Gold Star Membership and spends approximately $4,200 annually on "
    "eligible purchases at Costco warehouses and Costco.com. They are considering upgrading to an Executive Membership, "
    "which costs an additional $65 per year (for a total annual fee of $130) and provides a 2% annual reward on "
    "qualified Costco purchases, up to a maximum of $1,250 per year. Based on their current annual spending of $4,200, "
    "should they upgrade to the Executive Membership? Calculate whether the 2% reward they would earn exceeds the $65 "
    "upgrade fee, and determine the net benefit or loss from upgrading."
)

GIVEN_SPENDING = 4200.0
UPGRADE_FEE = 65.0
REWARD_RATE = 0.02
REWARD_CAP = 1250.0

EXPECTED_REWARD = min(GIVEN_SPENDING * REWARD_RATE, REWARD_CAP)  # $84.00
EXPECTED_NET = EXPECTED_REWARD - UPGRADE_FEE                      # $19.00
EXPECTED_RECOMMENDATION = "upgrade"  # Because net >= 0


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class UpgradeExtraction(BaseModel):
    """
    Extract information the answer used/reported for the Costco Executive Membership analysis.
    All fields should be extracted exactly as presented in the answer text. If not present, leave as null.
    """
    spending_used: Optional[str] = None                # e.g., "$4,200" or "4200"
    upgrade_fee_used: Optional[str] = None             # e.g., "$65"
    reward_rate_used: Optional[str] = None             # e.g., "2%", "0.02"
    reward_cap_used: Optional[str] = None              # e.g., "$1,250"
    computed_reward: Optional[str] = None              # e.g., "$84"
    net_benefit: Optional[str] = None                  # e.g., "$19"
    recommendation: Optional[str] = None               # e.g., "Upgrade", "Do not upgrade", "Yes, upgrade"
    break_even_spending: Optional[str] = None          # e.g., "$3,250"
    notes: Optional[str] = None                        # any extra commentary captured by the model


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_upgrade_info() -> str:
    return """
    Extract the specific values and conclusions that the answer uses regarding the Costco Executive Membership calculation.
    Return a JSON object with the following fields (use the exact strings found in the answer; do not invent anything):
    - spending_used: the annual eligible spending amount explicitly used/calculated with (e.g., "$4,200" or "4200").
    - upgrade_fee_used: the incremental upgrade fee amount used (should be $65 if used correctly).
    - reward_rate_used: the reward rate used (e.g., "2%" or "0.02").
    - reward_cap_used: the maximum annual reward cap mentioned (e.g., "$1,250").
    - computed_reward: the computed 2% annual reward dollar amount (e.g., "$84"), if provided.
    - net_benefit: the net benefit or loss amount after subtracting the $65 upgrade fee (e.g., "$19"), if provided.
    - recommendation: the final recommendation wording (e.g., "Upgrade", "Do not upgrade", "Yes, upgrade").
    - break_even_spending: a stated break-even spending number, if provided (e.g., "$3,250").
    - notes: any short extra notes if the answer includes relevant caveats or justification.

    Rules:
    - If an item is not explicitly present in the answer, set it to null.
    - Keep currency symbols and percent symbols if they appear in the answer.
    - Do not normalize or compute anything yourself; only extract from the answer text.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_membership_upgrade_analysis(
    evaluator: Evaluator,
    parent_node,
    extracted: UpgradeExtraction
) -> None:
    """
    Construct the verification tree following the rubric and perform verifications.
    """
    # Create the main sequential analysis node as critical
    analysis_node = evaluator.add_sequential(
        id="Membership_Upgrade_Analysis",
        desc="Evaluates whether upgrading to Costco Executive Membership is worthwhile based on the given spending, fees, reward rate, and cap.",
        parent=parent_node,
        critical=True
    )

    # 1) Use_Given_Inputs (parallel, critical) with 4 child leaves (all critical)
    inputs_node = evaluator.add_parallel(
        id="Use_Given_Inputs",
        desc="Uses the scenario's stated inputs (spending, upgrade fee, reward rate, reward cap) as the basis for calculations.",
        parent=analysis_node,
        critical=True
    )

    # Create leaves
    spending_leaf = evaluator.add_leaf(
        id="Annual_Spending_Input",
        desc="Uses the given annual eligible spending amount of $4,200.",
        parent=inputs_node,
        critical=True
    )
    fee_leaf = evaluator.add_leaf(
        id="Upgrade_Fee_Input",
        desc="Uses the given incremental upgrade fee of $65 (Executive vs. Gold Star).",
        parent=inputs_node,
        critical=True
    )
    rate_leaf = evaluator.add_leaf(
        id="Reward_Rate_Input",
        desc="Uses the given 2% reward rate on qualified purchases.",
        parent=inputs_node,
        critical=True
    )
    cap_leaf = evaluator.add_leaf(
        id="Reward_Cap_Input",
        desc="Accounts for the given maximum annual reward cap of $1,250 (i.e., reward is capped if applicable).",
        parent=inputs_node,
        critical=True
    )

    # Prepare claims for the four inputs and verify them in parallel
    input_claims = [
        (
            "The answer uses the given annual eligible spending amount of $4,200 as the spending used in its calculation.",
            None,
            spending_leaf,
            "Judge only whether the answer bases its computation on $4,200 (format variations like 4200 vs $4,200 are acceptable). "
            "If the answer substitutes a different spending amount for the calculation, mark this as incorrect."
        ),
        (
            "The answer uses the incremental upgrade fee of $65 (the additional cost to upgrade from Gold Star to Executive) in its calculation, not the full $130 total membership fee.",
            None,
            fee_leaf,
            "Accept if the answer clearly uses $65 as the incremental upgrade fee for comparison, even if it also mentions the total $130 annual fee. "
            "If the answer compares the reward to $130 instead of the $65 increment, mark this as incorrect."
        ),
        (
            "The answer uses the 2% annual reward rate on qualified Costco purchases.",
            None,
            rate_leaf,
            "Allow small phrasing variations like '2 percent' or '0.02'. "
            "If a different reward rate is used, mark this as incorrect."
        ),
        (
            "The answer accounts for the maximum annual reward cap of $1,250 (acknowledging the cap and/or noting that it is not reached for $4,200 in spending).",
            None,
            cap_leaf,
            "Accept if the answer explicitly mentions the $1,250 cap, or clearly reasons that the cap is not triggered for $4,200. "
            "If the answer omits any mention of the cap entirely or uses the wrong cap, mark this as incorrect."
        ),
    ]
    await evaluator.batch_verify(input_claims)

    # 2) Compute_Annual_Reward (leaf, critical)
    reward_leaf = evaluator.add_leaf(
        id="Compute_Annual_Reward",
        desc="Correctly computes the annual reward as 2% of the given spending and applies the $1,250 cap if relevant (without requiring a pre-specified numeric result).",
        parent=analysis_node,
        critical=True
    )
    reward_claim = "The answer correctly computes the annual reward as $84 (which is 2% of $4,200)."
    await evaluator.verify(
        claim=reward_claim,
        node=reward_leaf,
        additional_instruction=(
            "Check the answer's calculation of the 2% reward specifically for the $4,200 spending. "
            "Accept reasonable numeric formatting (e.g., $84, $84.00). "
            "For this scenario the cap is irrelevant because $84 << $1,250."
        ),
    )

    # 3) Compare_Reward_To_Upgrade_Fee (leaf, critical)
    compare_leaf = evaluator.add_leaf(
        id="Compare_Reward_To_Upgrade_Fee",
        desc="Correctly determines whether the computed reward is greater than or equal to the $65 upgrade fee.",
        parent=analysis_node,
        critical=True
    )
    compare_claim = "The answer correctly determines that the computed reward ($84) is greater than the $65 upgrade fee."
    await evaluator.verify(
        claim=compare_claim,
        node=compare_leaf,
        additional_instruction=(
            "Focus on the logical comparison. Accept equivalent wordings like 'exceeds' or 'is greater than or equal to'. "
            "If the answer implies the reward is less than $65, mark as incorrect."
        ),
    )

    # 4) Compute_Net_Benefit_Or_Loss (leaf, critical)
    net_leaf = evaluator.add_leaf(
        id="Compute_Net_Benefit_Or_Loss",
        desc="Correctly computes net benefit/loss as (computed reward − $65 upgrade fee).",
        parent=analysis_node,
        critical=True
    )
    net_claim = "The answer correctly computes the net benefit from upgrading as $19, calculated as $84 − $65."
    await evaluator.verify(
        claim=net_claim,
        node=net_leaf,
        additional_instruction=(
            "Accept minor formatting variations (e.g., '+$19', '$19.00'). "
            "If the computed net is any value other than $19 for this scenario, mark as incorrect."
        ),
    )

    # 5) Recommendation_Consistent_With_Result (leaf, critical)
    rec_leaf = evaluator.add_leaf(
        id="Recommendation_Consistent_With_Result",
        desc="Provides an upgrade/not-upgrade recommendation that is consistent with the computed net benefit/loss (upgrade if net >= 0; otherwise do not upgrade).",
        parent=analysis_node,
        critical=True
    )
    rec_claim = "The final recommendation in the answer is to upgrade to the Executive Membership, which is consistent with having a non-negative net benefit."
    await evaluator.verify(
        claim=rec_claim,
        node=rec_leaf,
        additional_instruction=(
            "Check for both presence and consistency of the recommendation. "
            "For this scenario, because the net benefit is positive ($19), the consistent recommendation is to upgrade. "
            "Accept equivalent phrasings like 'Yes, upgrade' or 'Upgrading is worthwhile'."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Costco Executive Membership upgrade analysis task.
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

    # Extract structured info from the answer (for logging and transparency)
    extracted = await evaluator.extract(
        prompt=prompt_extract_upgrade_info(),
        template_class=UpgradeExtraction,
        extraction_name="extracted_upgrade_info"
    )

    # Add ground truth information for reference
    evaluator.add_ground_truth({
        "given_inputs": {
            "annual_spending": "$4,200",
            "upgrade_fee_increment": "$65",
            "reward_rate": "2%",
            "reward_cap": "$1,250"
        },
        "expected_computation": {
            "reward": f"${EXPECTED_REWARD:.0f}",
            "net_benefit": f"${EXPECTED_NET:.0f}",
            "recommendation": EXPECTED_RECOMMENDATION
        }
    })

    # Build verification tree and perform checks
    await build_membership_upgrade_analysis(evaluator, root, extracted)

    # Return the final structured summary
    return evaluator.get_summary()