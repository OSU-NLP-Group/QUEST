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
TASK_ID = "america_the_beautiful_annual_pass_2026"
TASK_DESCRIPTION = (
    "What are the current annual pass prices for the America the Beautiful – The National Parks and Federal Recreational Lands Pass "
    "for both U.S. residents and non-residents in 2026, and when did this new pricing structure take effect? Provide the specific dollar "
    "amounts for each category and the exact date the new pricing began."
)

# Ground truth expectations for evaluation context
GROUND_TRUTH_EXPECTED = {
    "resident_price_usd": 80,
    "non_resident_price_usd": 250,
    "effective_date": "January 1, 2026"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PassPricingExtraction(BaseModel):
    resident_price: Optional[str] = None
    non_resident_price: Optional[str] = None
    effective_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pass_pricing() -> str:
    return """
    Extract the 2026 information about the America the Beautiful – The National Parks and Federal Recreational Lands Annual Pass from the provided answer text.

    Required fields:
    1) resident_price: The stated dollar price for U.S. residents (Annual Pass, general public) in 2026, as presented in the answer. If missing, return null. Keep it exactly as written in the answer (e.g., "$80", "80 USD").
    2) non_resident_price: The stated dollar price for non-residents in 2026, as presented in the answer. If missing, return null. Keep it exactly as written in the answer (e.g., "$250", "250 USD").
    3) effective_date: The exact date when the new pricing took effect (e.g., "January 1, 2026"). If missing, return null.
    4) sources: Extract all URLs the answer cites that are intended to support these pricing details. Include any official pages (e.g., NPS, DOI) or credible pages mentioned. Return an array of URLs. If no URLs are provided, return an empty array.

    Notes:
    - Only extract what is explicitly stated in the answer; do not infer or invent information.
    - The "Annual Pass" refers to the standard America the Beautiful Annual Pass, not special categories (Senior, Military, 4th Grade, etc.). Ignore those special passes.
    - If the answer uses different formatting (like "US$80"), still capture it as-is in resident_price/non_resident_price.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _extract_numeric_amount(text: Optional[str]) -> Optional[int]:
    """
    Extract a numeric dollar amount from a free-form price string.
    Examples:
        "$80" -> 80
        "US$80" -> 80
        "80 USD" -> 80
    If not found, return None.
    """
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def _matches_effective_date_2026_01_01(text: Optional[str]) -> bool:
    """
    Return True if the provided text clearly indicates January 1, 2026.
    Accept common variants like:
      - "January 1, 2026"
      - "Jan 1, 2026"
      - "2026-01-01"
    Case-insensitive match.
    """
    if not text:
        return False
    t = text.strip().lower()
    if "2026-01-01" in t:
        return True
    if "january 1, 2026" in t:
        return True
    if "jan 1, 2026" in t or "jan. 1, 2026" in t:
        return True
    # Also accept formats without comma
    if "january 1 2026" in t or "jan 1 2026" in t:
        return True
    return False


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_pricing_tree(
    evaluator: Evaluator,
    root_node,
    extraction: PassPricingExtraction
) -> None:
    """
    Build the verification tree per rubric and perform verifications:
    - Create a critical parallel node for the 2026 pricing structure.
    - Add a critical gating node to ensure sources are provided.
    - Add three critical leaf verifications (resident price $80, non-resident price $250, effective date Jan 1, 2026) grounded by URLs.
    - Add three critical custom checks to ensure the answer text itself states the correct values.
    """
    # Parent critical node (JSON parent is critical parallel)
    main_node = evaluator.add_parallel(
        id="America_the_Beautiful_Pass_2026_Information",
        desc="Verify that the answer correctly provides the 2026 pricing structure for the America the Beautiful Annual Pass, including both resident and non-resident prices, and the effective date of the new pricing",
        parent=root_node,
        critical=True
    )

    # Gating: sources must be present for factual verification
    sources_present = bool(extraction.sources)
    evaluator.add_custom_node(
        result=sources_present,
        id="Sources_Provided",
        desc="At least one supporting source URL is provided in the answer",
        parent=main_node,
        critical=True
    )

    # Custom checks to ensure the answer text states the expected values
    resident_amount = _extract_numeric_amount(extraction.resident_price)
    non_resident_amount = _extract_numeric_amount(extraction.non_resident_price)
    date_is_jan_1_2026 = _matches_effective_date_2026_01_01(extraction.effective_date)

    evaluator.add_custom_node(
        result=(resident_amount == GROUND_TRUTH_EXPECTED["resident_price_usd"]),
        id="Resident_Pass_Price_Answer_Correct",
        desc="Answer text states resident (U.S.) annual pass price as $80",
        parent=main_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(non_resident_amount == GROUND_TRUTH_EXPECTED["non_resident_price_usd"]),
        id="Non_Resident_Pass_Price_Answer_Correct",
        desc="Answer text states non-resident annual pass price as $250",
        parent=main_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=date_is_jan_1_2026,
        id="Effective_Date_Answer_Correct",
        desc="Answer text states the effective date as January 1, 2026",
        parent=main_node,
        critical=True
    )

    # Leaf: Resident price supported by sources
    resident_leaf = evaluator.add_leaf(
        id="Resident_Pass_Price",
        desc="The answer states that the U.S. resident annual pass costs $80",
        parent=main_node,
        critical=True
    )
    resident_claim = (
        "As of 2026, the America the Beautiful Annual Pass price for U.S. residents (general public, standard Annual Pass) is $80."
    )
    await evaluator.verify(
        claim=resident_claim,
        node=resident_leaf,
        sources=extraction.sources,
        additional_instruction=(
            "Verify the price for the standard America the Beautiful Annual Pass (not special categories like Senior, Military, 4th Grade). "
            "Confirm that the webpage explicitly shows $80 as the resident price for 2026. Ignore service fees, taxes, shipping, or non-annual products."
        )
    )

    # Leaf: Non-resident price supported by sources
    non_resident_leaf = evaluator.add_leaf(
        id="Non_Resident_Pass_Price",
        desc="The answer states that the non-resident annual pass costs $250",
        parent=main_node,
        critical=True
    )
    non_resident_claim = (
        "As of 2026, the America the Beautiful Annual Pass price for non-residents is $250."
    )
    await evaluator.verify(
        claim=non_resident_claim,
        node=non_resident_leaf,
        sources=extraction.sources,
        additional_instruction=(
            "Verify the price for the standard Annual Pass specifically for non-residents in 2026. "
            "Pass only if the page clearly states $250 for non-residents. Ignore discounted categories or other pass types."
        )
    )

    # Leaf: Effective date supported by sources
    effective_date_leaf = evaluator.add_leaf(
        id="Effective_Date",
        desc="The answer states that the new pricing structure took effect on January 1, 2026",
        parent=main_node,
        critical=True
    )
    effective_date_claim = (
        "The new pricing structure for the America the Beautiful Annual Pass took effect on January 1, 2026."
    )
    await evaluator.verify(
        claim=effective_date_claim,
        node=effective_date_leaf,
        sources=extraction.sources,
        additional_instruction=(
            "Verify the effective date of the 2026 pricing structure and confirm it is exactly January 1, 2026."
        )
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
    Evaluate an answer for the 2026 America the Beautiful Annual Pass pricing task.
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_pass_pricing(),
        template_class=PassPricingExtraction,
        extraction_name="pass_pricing_2026_extraction",
    )

    # Add ground truth context for transparency
    evaluator.add_ground_truth(
        {
            "expected_resident_price_usd": GROUND_TRUTH_EXPECTED["resident_price_usd"],
            "expected_non_resident_price_usd": GROUND_TRUTH_EXPECTED["non_resident_price_usd"],
            "expected_effective_date": GROUND_TRUTH_EXPECTED["effective_date"]
        },
        gt_type="expected_values"
    )

    # Build verification tree and run checks
    await build_and_verify_pricing_tree(evaluator, root, extraction)

    # Return summary
    return evaluator.get_summary()