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
TASK_ID = "atb_pass_pricing_2026"
TASK_DESCRIPTION = (
    "Starting in 2026, the U.S. Department of the Interior implemented a new pricing structure for the America the "
    "Beautiful Annual Pass, which provides access to national parks and federal recreational lands. What is the cost "
    "of this annual pass for U.S. residents, what is the cost for non-residents, and on what date did this new pricing "
    "take effect?"
)

EXPECTED_INFO = {
    "resident_price": "$80",
    "non_resident_price": "$250",
    "effective_date": "January 1, 2026"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PricingExtraction(BaseModel):
    """
    Structured extraction of pricing and effective date information from the agent's answer.
    """
    resident_price: Optional[str] = None
    non_resident_price: Optional[str] = None
    effective_date: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pricing_info() -> str:
    return (
        "Extract from the answer the three specific pieces of information about the America the Beautiful Annual Pass "
        "pricing structure starting in 2026:\n"
        "1) resident_price: The stated cost for U.S. residents. Return exactly as presented (e.g., \"$80\", \"80 dollars\").\n"
        "2) non_resident_price: The stated cost for non-residents. Return exactly as presented.\n"
        "3) effective_date: The stated date when the new pricing took effect (e.g., \"January 1, 2026\", \"01/01/2026\").\n"
        "4) source_urls: Extract all URLs explicitly included in the answer that are cited as sources or references for these prices/date. "
        "Include plain URLs or URLs in markdown links. If none are provided, return an empty array.\n"
        "If any of the above are not mentioned, return null for that field."
    )


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_pricing_facts(
    evaluator: Evaluator,
    root_node,
    extracted: PricingExtraction,
) -> None:
    """
    Build the verification tree and run checks against the agent's answer.
    This rubric checks that the answer explicitly states the three correct facts.
    Verification is performed against the answer content (simple verification).
    """
    # Create the critical root node with parallel aggregation
    main_node = evaluator.add_parallel(
        id="America_the_Beautiful_Annual_Pass_Pricing_2026",
        desc="Answer provides the resident price, non-resident price, and the effective date for the 2026 America the Beautiful Annual Pass pricing structure.",
        parent=root_node,
        critical=True,
    )

    # Leaf 1: U.S. Resident price is $80
    resident_leaf = evaluator.add_leaf(
        id="US_Resident_Pass_Price",
        desc="States that the America the Beautiful Annual Pass costs $80 for U.S. residents.",
        parent=main_node,
        critical=True,
    )
    resident_claim = "The answer states that the America the Beautiful Annual Pass costs $80 for U.S. residents."
    resident_instruction = (
        "Judge solely based on the answer text. Allow reasonable variations in phrasing and formatting (e.g., '$80', "
        "'80 dollars', 'USD 80'). Accept synonyms like 'U.S. residents', 'US residents', 'U.S. customers' if clearly "
        "referring to residents. Do not infer from your own knowledge; rely only on the provided answer content."
    )

    # Leaf 2: Non-resident price is $250
    nonresident_leaf = evaluator.add_leaf(
        id="Non_Resident_Pass_Price",
        desc="States that the America the Beautiful Non-Resident Annual Pass costs $250 for non-residents.",
        parent=main_node,
        critical=True,
    )
    nonresident_claim = "The answer states that the America the Beautiful Annual Pass costs $250 for non-residents."
    nonresident_instruction = (
        "Judge solely based on the answer text. Allow reasonable formatting variations (e.g., '$250', '250 dollars', "
        "'USD 250') and minor wording variations referring to non-residents. Do not infer beyond the answer content."
    )

    # Leaf 3: Effective date is January 1, 2026
    effective_leaf = evaluator.add_leaf(
        id="Effective_Date",
        desc="States that the new pricing structure took effect on January 1, 2026.",
        parent=main_node,
        critical=True,
    )
    effective_claim = "The answer states that the new pricing structure took effect on January 1, 2026."
    effective_instruction = (
        "Judge solely based on the answer text. Accept reasonable date format variations that correspond to the same date "
        "(e.g., 'January 1, 2026', 'Jan 1, 2026', '01/01/2026'). Do not infer beyond the answer content."
    )

    # Run the three simple verifications in parallel
    await evaluator.batch_verify(
        [
            (resident_claim, None, resident_leaf, resident_instruction),
            (nonresident_claim, None, nonresident_leaf, nonresident_instruction),
            (effective_claim, None, effective_leaf, effective_instruction),
        ]
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
    Evaluate an answer for the 2026 America the Beautiful Annual Pass pricing task.
    This script checks whether the answer explicitly states the correct resident price ($80),
    non-resident price ($250), and effective date (January 1, 2026).
    """
    # Initialize evaluator with parallel root node
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

    # Extract structured pricing info from the answer (recorded for summary/debugging)
    extracted_pricing = await evaluator.extract(
        prompt=prompt_extract_pricing_info(),
        template_class=PricingExtraction,
        extraction_name="pricing_extraction_2026",
    )

    # Add ground truth information (for transparency in summary)
    evaluator.add_ground_truth(
        {
            "expected_resident_price": EXPECTED_INFO["resident_price"],
            "expected_non_resident_price": EXPECTED_INFO["non_resident_price"],
            "expected_effective_date": EXPECTED_INFO["effective_date"],
        },
        gt_type="expected_values"
    )

    # Build verification tree and verify claims
    await verify_pricing_facts(evaluator, root, extracted_pricing)

    # Return standardized summary
    return evaluator.get_summary()