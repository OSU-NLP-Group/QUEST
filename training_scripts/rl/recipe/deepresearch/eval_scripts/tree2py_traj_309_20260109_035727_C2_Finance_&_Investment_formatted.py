import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mutual_fund_5m_institutional"
TASK_DESCRIPTION = (
    "An institutional investor has exactly $5 million available to invest and wants to access an institutional share class of a U.S. mutual fund. "
    "Identify which major mutual fund company offers an institutional share class with a minimum investment requirement that exactly matches this $5 million amount, "
    "and provide the specific name of that institutional share class."
)

GROUND_TRUTH = {
    "company": "Vanguard",
    "share_class": "Institutional Shares",
    "minimum_investment": "$5,000,000"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FundIdentification(BaseModel):
    """Information extracted from the agent's answer regarding the fund/company/share class."""
    company_name: Optional[str] = None
    share_class_name: Optional[str] = None
    minimum_investment_text: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_fund_identification() -> str:
    return """
    Extract the following fields from the answer:

    1) company_name: The mutual fund company identified by the answer (e.g., "Vanguard").
    2) share_class_name: The exact institutional share class name mentioned in the answer (e.g., "Institutional Shares").
    3) minimum_investment_text: The minimum investment requirement mentioned for the identified institutional share class, as stated in the answer (e.g., "$5 million", "5,000,000 USD").
    4) source_urls: All URLs explicitly cited in the answer that are intended to support the identification (company, share class, and minimum investment). Include only actual URLs present in the answer (plain URLs or markdown links).

    Rules:
    - Return null for any missing field.
    - Do not invent or infer information that is not explicitly present.
    - For URLs, extract only valid URLs; if none are present, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: FundIdentification) -> None:
    """
    Build the verification tree according to the rubric and run checks.
    """

    # Create critical root-level task node
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Complete identification of the mutual fund company and institutional share class that has a minimum investment requirement of exactly $5 million",
        parent=evaluator.root,
        critical=True
    )

    # Fund company subtree (critical, parallel)
    fund_company_node = evaluator.add_parallel(
        id="Fund_Company",
        desc="Correctly identify the mutual fund company offering an institutional share class at the $5 million minimum threshold",
        parent=task_node,
        critical=True
    )

    # Leaf: Company_Name (critical)
    company_leaf = evaluator.add_leaf(
        id="Company_Name",
        desc="The identified fund company is Vanguard",
        parent=fund_company_node,
        critical=True
    )

    # Leaf: Minimum_Investment_Amount (critical)
    min_invest_leaf = evaluator.add_leaf(
        id="Minimum_Investment_Amount",
        desc="The minimum investment requirement is stated as $5 million",
        parent=fund_company_node,
        critical=True
    )

    # Share class subtree (critical, parallel)
    share_class_node = evaluator.add_parallel(
        id="Share_Class",
        desc="Correctly specify the name of the institutional share class offered at this minimum threshold",
        parent=task_node,
        critical=True
    )

    # Leaf: Share_Class_Name (critical)
    share_class_leaf = evaluator.add_leaf(
        id="Share_Class_Name",
        desc="The share class name is identified as Institutional Shares",
        parent=share_class_node,
        critical=True
    )

    # Prepare claims (simple verification against the answer text)
    claim_company = "The answer identifies the mutual fund company as Vanguard."
    claim_minimum = "The answer states that the minimum investment requirement is exactly $5 million (i.e., 5,000,000 USD)."
    claim_share_class = "The answer identifies the institutional share class name as 'Institutional Shares'."

    # Batch verify all leaves
    await evaluator.batch_verify(
        [
            (
                claim_company,
                None,
                company_leaf,
                "Allow minor variations such as 'The Vanguard Group' or 'Vanguard Group' to be considered equivalent to 'Vanguard'. Focus on whether the answer explicitly identifies Vanguard."
            ),
            (
                claim_minimum,
                None,
                min_invest_leaf,
                "Accept variations like '$5 million', 'five million dollars', or '5,000,000 USD'. It must be an exact minimum threshold of 5 million, not a range or a different minimum amount."
            ),
            (
                claim_share_class,
                None,
                share_class_leaf,
                "Allow minor formatting variants (e.g., capitalization). The phrase 'Institutional Shares' should appear explicitly in the answer as the share class name."
            ),
        ]
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the institutional share class identification task.
    """
    # Initialize evaluator with parallel root
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_fund_identification(),
        template_class=FundIdentification,
        extraction_name="fund_identification"
    )

    # Add ground truth info to summary
    evaluator.add_ground_truth(
        {
            "expected_company": GROUND_TRUTH["company"],
            "expected_share_class": GROUND_TRUTH["share_class"],
            "expected_minimum_investment": GROUND_TRUTH["minimum_investment"]
        },
        gt_type="ground_truth"
    )

    # Build tree and verify
    await build_and_verify_tree(evaluator, extracted)

    # Return final structured summary
    return evaluator.get_summary()