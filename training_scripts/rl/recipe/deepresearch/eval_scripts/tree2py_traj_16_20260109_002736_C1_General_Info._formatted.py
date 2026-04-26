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
TASK_ID = "irs_2026_business_mileage_rate"
TASK_DESCRIPTION = "What is the IRS standard mileage rate for business use of a personal vehicle in 2026, and when does this rate take effect?"

EXPECTED_RATE_TEXT = "72.5 cents per mile"
EXPECTED_EFFECTIVE_DATE = "January 1, 2026"
EXAMPLE_OFFICIAL_SOURCES = [
    "IRS Newsroom announcement IR-2025-128",
    "IRS Notice 2026-10"
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MileageInfoExtraction(BaseModel):
    """
    Structured information extracted from the agent's answer.
    """
    stated_business_rate_text: Optional[str] = None
    stated_effective_date_text: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_mileage_info() -> str:
    return """
    Extract the specific information about the IRS standard mileage rate for business use for tax year 2026 from the provided answer.

    Return a JSON object with the following fields:
    1) stated_business_rate_text: The exact text in the answer that states the IRS standard mileage rate for business use in 2026 (e.g., "72.5 cents per mile", "72.5¢ per mile", or "$0.725 per mile"). If not stated, return null.
    2) stated_effective_date_text: The exact text in the answer that states when the 2026 business mileage rate takes effect (e.g., "January 1, 2026", "Jan. 1, 2026", "1/1/2026"). If not stated, return null.
    3) source_urls: All explicit URLs the answer cites as sources for this information. Include only actual URLs present in the answer (including markdown links). If no URLs are cited, return an empty list.

    Important:
    - Focus only on the BUSINESS use rate for 2026.
    - Preserve the exact phrasing found in the answer for the two text fields.
    - Do not invent any URLs; include only those explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_mileage_answer(
    evaluator: Evaluator,
    parent_node,
    extracted: MileageInfoExtraction
) -> None:
    """
    Build and verify all required checks under the critical parent node.
    """
    # 1) Business mileage rate must be 72.5 cents per mile
    rate_node = evaluator.add_leaf(
        id="Business_Mileage_Rate",
        desc="The answer must state that the IRS standard mileage rate for business use in 2026 is 72.5 cents per mile",
        parent=parent_node,
        critical=True
    )
    claim_rate = (
        "Within the provided answer text, the IRS standard mileage rate for business use in 2026 is stated as 72.5 cents per mile."
    )
    await evaluator.verify(
        claim=claim_rate,
        node=rate_node,
        additional_instruction=(
            "Judge ONLY whether the answer text explicitly states the 2026 BUSINESS mileage rate as 72.5 cents per mile. "
            "Accept equivalent formatting such as '72.5¢ per mile' or '$0.725 per mile'. "
            "Do not rely on your own knowledge; base the decision strictly on the answer content."
        ),
    )

    # 2) Effective date must be January 1, 2026
    effective_date_node = evaluator.add_leaf(
        id="Effective_Date",
        desc="The answer must state that the rate takes effect on January 1, 2026",
        parent=parent_node,
        critical=True
    )
    claim_effective = (
        "Within the provided answer text, the effective date for the 2026 business mileage rate is January 1, 2026."
    )
    await evaluator.verify(
        claim=claim_effective,
        node=effective_date_node,
        additional_instruction=(
            "Judge ONLY whether the answer text explicitly indicates the rate takes effect on January 1, 2026. "
            "Accept reasonable variants such as 'Jan. 1, 2026' or '1/1/2026'. "
            "Do not rely on external knowledge; base the decision strictly on the answer content."
        ),
    )

    # 3) Official IRS source reference (e.g., IR-2025-128 or Notice 2026-10)
    official_src_node = evaluator.add_leaf(
        id="Official_Source_Reference",
        desc="The answer must reference an official IRS source (such as the IRS newsroom announcement IR-2025-128 or Notice-2026-10)",
        parent=parent_node,
        critical=True
    )

    # If no sources are provided in the answer, mark this leaf as failed without calling verify_by_urls.
    if not extracted.source_urls:
        official_src_node.score = 0.0
        official_src_node.status = "failed"
    else:
        claim_official = (
            "This webpage is an official IRS source (domain irs.gov) that announces or specifies the IRS standard mileage rate "
            "for business use for tax year 2026—examples include the IRS Newsroom release IR-2025-128 or IRS Notice 2026-10."
        )
        await evaluator.verify(
            claim=claim_official,
            node=official_src_node,
            sources=extracted.source_urls,
            additional_instruction=(
                "Pass if at least one provided URL is on the irs.gov domain AND is specifically about the 2026 standard mileage rates "
                "for business use (e.g., IRS Newsroom IR-2025-128 or Notice 2026-10). "
                "Use the webpage content and the URL to determine whether it is an official IRS announcement or notice for 2026 mileage rates."
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
    Evaluate the agent's answer to:
    'What is the IRS standard mileage rate for business use of a personal vehicle in 2026, and when does this rate take effect?'
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
        default_model=model,
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_mileage_info(),
        template_class=MileageInfoExtraction,
        extraction_name="mileage_info_2026",
    )

    # Add a critical top-level node per rubric
    main_node = evaluator.add_parallel(
        id="2026_IRS_Business_Mileage_Rate",
        desc="Evaluation of the complete answer regarding the 2026 IRS standard mileage rate for business use",
        parent=root,
        critical=True
    )

    # Add ground truth and reference info
    evaluator.add_ground_truth({
        "expected_business_rate_text": EXPECTED_RATE_TEXT,
        "expected_effective_date": EXPECTED_EFFECTIVE_DATE,
        "example_official_sources": EXAMPLE_OFFICIAL_SOURCES
    })

    # Perform verifications under the critical parent node
    await verify_mileage_answer(evaluator, main_node, extracted)

    # Return final structured summary
    return evaluator.get_summary()