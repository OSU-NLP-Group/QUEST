import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "robo_advisors_three_meet_criteria"
TASK_DESCRIPTION = """
I'm looking to open an investment account with a robo-advisor that meets the following requirements:

1. The account minimum requirement must be $500 or less for investment accounts
2. The standard digital advisory fee must be 0.25% annually or less
3. The average expense ratio of the underlying portfolio ETFs must be 0.20% or less
4. The platform must offer automated tax-loss harvesting as a standard feature
5. The platform must provide access to human financial advisors (either included in the base service or available through a premium tier)
6. The platform must offer socially responsible investing (SRI) or ESG portfolio options
7. The platform must provide automatic portfolio rebalancing
8. The platform must offer at least one additional unique feature beyond the basics, such as cryptocurrency exposure, a portfolio line of credit, an integrated high-yield cash or checking account, or comprehensive goal planning tools

Please identify three distinct robo-advisors that meet all of these criteria. For each robo-advisor, provide:
- The name of the robo-advisor
- Confirmation that it meets each of the eight requirements listed above
- At least one reference URL that supports the information about its features and costs
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AdvisorEntry(BaseModel):
    name: Optional[str] = None
    account_minimum: Optional[str] = None
    advisory_fee: Optional[str] = None
    expense_ratio: Optional[str] = None
    tax_loss_harvesting: Optional[str] = None  # textual confirmation as stated in answer
    human_advisors: Optional[str] = None       # textual confirmation as stated in answer
    sri_esg: Optional[str] = None              # textual confirmation as stated in answer
    auto_rebalancing: Optional[str] = None     # textual confirmation as stated in answer
    unique_feature: Optional[str] = None       # e.g., "crypto exposure", "portfolio line of credit"
    sources: List[str] = Field(default_factory=list)


class AdvisorsExtraction(BaseModel):
    advisors: List[AdvisorEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_robo_advisors() -> str:
    return """
    Extract up to the first three distinct robo-advisors mentioned in the answer, preserving the answer's order.
    For each robo-advisor, return the following fields gathered from the answer text:
    - name: The robo-advisor’s name.
    - account_minimum: The stated investment account minimum (e.g., "$0", "$100", "$500"). If unclear or not stated, set null.
    - advisory_fee: The stated standard digital advisory fee (e.g., "0.25% annually", "0.15%"). If multiple tiers exist, extract the standard/base digital tier's fee. If missing, set null.
    - expense_ratio: The average ETF expense ratio for the core/standard portfolios as stated (e.g., "0.08%", "0.15%–0.19%"). If not stated, set null.
    - tax_loss_harvesting: Any text from the answer indicating whether automated tax-loss harvesting is provided as a standard feature for taxable accounts. If missing, set null.
    - human_advisors: Any text from the answer indicating that human financial advisors are accessible (included or via premium tier). If missing, set null.
    - sri_esg: Any text from the answer indicating SRI/ESG portfolio options are available. If missing, set null.
    - auto_rebalancing: Any text from the answer indicating automatic portfolio rebalancing is provided. If missing, set null.
    - unique_feature: One additional differentiating feature beyond the basics (e.g., cryptocurrency exposure, portfolio line of credit, integrated high-yield cash/checking account, comprehensive goal planning tools). If present in the answer, extract a concise phrase (e.g., "cryptocurrency exposure"). If missing, set null.
    - sources: A list of all reference URLs explicitly provided in the answer that support this robo-advisor’s features or costs. Include any official pages, pricing pages, feature pages, etc. Only include valid URLs explicitly present in the answer.

    Rules:
    - Only extract what is explicitly stated in the answer. Do not infer or fabricate any information.
    - Ensure URLs are valid and include the protocol (http/https). If the answer omits protocol, prepend http://.
    - If a field is not present in the answer, set it to null (or [] for sources).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_robo_advisor(
    evaluator: Evaluator,
    parent_node,
    advisor: AdvisorEntry,
    idx: int
) -> None:
    """
    Build and verify the subtree for a single robo-advisor. Follows the rubric leaf-by-leaf.
    Leaves are evaluated in an order that allows 'Reference_URLs' to gate subsequent checks.
    """
    # Create per-advisor parallel node (non-critical to allow partial per advisor)
    advisor_label = ["First", "Second", "Third"][idx] if idx < 3 else f"Advisor_{idx+1}"
    advisor_node = evaluator.add_parallel(
        id=f"advisor_{idx}",
        desc=f"{advisor_label} robo-advisor meeting all criteria",
        parent=parent_node,
        critical=False
    )

    name = advisor.name or f"Advisor #{idx+1}"
    urls = advisor.sources or []

    # 1) Reference URLs existence (critical, existence check)
    ref_urls_node = evaluator.add_custom_node(
        result=bool(urls),
        id=f"advisor_{idx}_Reference_URLs",
        desc="Provides at least one reference URL that supports the information about features and costs",
        parent=advisor_node,
        critical=True
    )

    # 2) Account Minimum <= $500 (critical)
    acc_min_node = evaluator.add_leaf(
        id=f"advisor_{idx}_Account_Minimum",
        desc="Account minimum requirement is $500 or less for investment accounts",
        parent=advisor_node,
        critical=True
    )
    acc_min_claim = (
        f"The minimum required to open an investment account with {name} is $500 or less. "
        f"If {name} has a $0 minimum, that also satisfies the requirement. "
        f"Ignore cash-only accounts; focus on investing accounts."
    )
    await evaluator.verify(
        claim=acc_min_claim,
        node=acc_min_node,
        sources=urls,
        additional_instruction=(
            "Verify on official pricing or account pages. If multiple account types exist, "
            "consider the standard taxable or retirement investing account minimum. "
            "Values like $0, $10, $100, or $500 qualify."
        ),
    )

    # 3) Advisory Fee <= 0.25% (critical)
    advisory_fee_node = evaluator.add_leaf(
        id=f"advisor_{idx}_Advisory_Fee",
        desc="Standard digital advisory fee is 0.25% annually or less",
        parent=advisor_node,
        critical=True
    )
    advisory_fee_claim = (
        f"The standard base digital advisory fee for {name} is 0.25% annually or less. "
        f"If fees vary by balance tiers, the base tier should be at or below 0.25%. "
        f"Exclude premium, add-on, or human-advised upgrade fees."
    )
    await evaluator.verify(
        claim=advisory_fee_claim,
        node=advisory_fee_node,
        sources=urls,
        additional_instruction=(
            "Check official pricing/fees pages for the standard robo-advisory service. "
            "Do not consider premium human-advisor tiers for this check."
        ),
    )

    # 4) Average ETF Expense Ratio <= 0.20% (critical)
    expense_ratio_node = evaluator.add_leaf(
        id=f"advisor_{idx}_Portfolio_Expense_Ratio",
        desc="Average expense ratio of underlying portfolio ETFs is 0.20% or less",
        parent=advisor_node,
        critical=True
    )
    expense_ratio_claim = (
        f"The average expense ratio (i.e., average ETF fund fees) for {name}'s standard/core portfolios is 0.20% or less. "
        f"This refers to underlying fund fees, not advisory fees."
    )
    await evaluator.verify(
        claim=expense_ratio_claim,
        node=expense_ratio_node,
        sources=urls,
        additional_instruction=(
            "Look for language such as 'average ETF expense ratio', 'average fund fees', or similar. "
            "Ranges like 0.08%–0.15% qualify if the average is within 0.20%. "
            "Exclude advisory/platform fees from this metric."
        ),
    )

    # 5) Automated Tax-Loss Harvesting as Standard Feature (critical)
    tlh_node = evaluator.add_leaf(
        id=f"advisor_{idx}_Tax_Loss_Harvesting",
        desc="Offers automated tax-loss harvesting as a standard feature",
        parent=advisor_node,
        critical=True
    )
    tlh_claim = (
        f"{name} offers automated tax-loss harvesting as a standard feature for taxable investment accounts "
        f"(i.e., included for eligible taxable accounts without a separate paid upgrade)."
    )
    await evaluator.verify(
        claim=tlh_claim,
        node=tlh_node,
        sources=urls,
        additional_instruction=(
            "Confirm it's a core feature for taxable accounts. If it is only in a paid premium tier, do not consider it standard."
        ),
    )

    # 6) Access to Human Financial Advisors (critical)
    human_node = evaluator.add_leaf(
        id=f"advisor_{idx}_Human_Advisor_Access",
        desc="Provides access to human financial advisors (either included in the base service or available through a premium tier)",
        parent=advisor_node,
        critical=True
    )
    human_claim = (
        f"{name} provides access to human financial advisors—either included in the base service or available via "
        f"an optional premium tier."
    )
    await evaluator.verify(
        claim=human_claim,
        node=human_node,
        sources=urls,
        additional_instruction=(
            "Accept either included-access (e.g., via chat/call) or a clearly offered optional upgrade tier that provides human advisors."
        ),
    )

    # 7) SRI/ESG Portfolio Options (critical)
    sri_node = evaluator.add_leaf(
        id=f"advisor_{idx}_SRI_ESG_Options",
        desc="Offers socially responsible investing (SRI) or ESG portfolio options",
        parent=advisor_node,
        critical=True
    )
    sri_claim = f"{name} offers SRI and/or ESG portfolio options."
    await evaluator.verify(
        claim=sri_claim,
        node=sri_node,
        sources=urls,
        additional_instruction=(
            "Look for terms like 'SRI', 'ESG', 'socially responsible investing', or 'sustainable portfolios'."
        ),
    )

    # 8) Automatic Portfolio Rebalancing (critical)
    rebalance_node = evaluator.add_leaf(
        id=f"advisor_{idx}_Auto_Rebalancing",
        desc="Provides automatic portfolio rebalancing",
        parent=advisor_node,
        critical=True
    )
    rebalance_claim = f"{name} provides automatic portfolio rebalancing."
    await evaluator.verify(
        claim=rebalance_claim,
        node=rebalance_node,
        sources=urls,
        additional_instruction=(
            "Confirm that the platform automatically rebalances portfolios (periodic or drift-based)."
        ),
    )

    # 9) Additional Unique Feature (critical)
    unique_node = evaluator.add_leaf(
        id=f"advisor_{idx}_Additional_Unique_Feature",
        desc="Offers at least one additional unique feature (e.g., cryptocurrency exposure, portfolio line of credit, integrated high-yield cash or checking account, or comprehensive goal planning tools)",
        parent=advisor_node,
        critical=True
    )
    if advisor.unique_feature and advisor.unique_feature.strip():
        unique_claim = (
            f"{name} offers '{advisor.unique_feature.strip()}', which is an additional feature beyond basic robo-advisory, "
            f"such as crypto exposure, a portfolio line of credit, integrated cash/checking, or comprehensive goal planning."
        )
    else:
        unique_claim = (
            f"{name} offers at least one additional unique feature beyond basic robo-advisory (e.g., cryptocurrency access, "
            f"portfolio line of credit, integrated high-yield cash or checking, or comprehensive goal planning tools)."
        )
    await evaluator.verify(
        claim=unique_claim,
        node=unique_node,
        sources=urls,
        additional_instruction=(
            "Identify and confirm at least one differentiating feature from official pages. "
            "Common examples include cryptocurrency exposure, portfolio line of credit, integrated high-yield cash/checking, "
            "or robust goals planning tools."
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
    Evaluate an answer for the robo-advisor selection task using the Mind2Web2 evaluation framework.
    """
    # Initialize evaluator with a parallel root (three advisors evaluated independently)
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

    # Extract up to three advisors from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_robo_advisors(),
        template_class=AdvisorsExtraction,
        extraction_name="robo_advisors_extraction"
    )

    # Normalize to exactly 3 entries (pad if fewer)
    advisors: List[AdvisorEntry] = (extracted.advisors or [])[:3]
    while len(advisors) < 3:
        advisors.append(AdvisorEntry())

    # Build top-level nodes for the three advisors (parallel aggregation)
    labels = ["First robo-advisor meeting all criteria", "Second robo-advisor meeting all criteria", "Third robo-advisor meeting all criteria"]
    advisor_parent_nodes = []
    for i in range(3):
        node = evaluator.add_parallel(
            id=f"advisor_group_{i}",
            desc=labels[i],
            parent=root,
            critical=False
        )
        advisor_parent_nodes.append(node)

    # For each advisor, construct and verify all leaf checks
    for i in range(3):
        await verify_robo_advisor(
            evaluator=evaluator,
            parent_node=advisor_parent_nodes[i],
            advisor=advisors[i],
            idx=i
        )

    # Return the evaluation summary
    return evaluator.get_summary()