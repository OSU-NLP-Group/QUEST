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
TASK_ID = "atb_pass_2026_pricing"
TASK_DESCRIPTION = (
    "What are the prices for the America the Beautiful Annual Pass for U.S. residents and non-residents "
    "starting January 1, 2026? Additionally, what is the per-person fee that non-residents age 16 and older "
    "without an annual pass must pay at certain national parks, on top of the standard entrance fee?"
)

# Ground-truth expectations (for reporting convenience)
GROUND_TRUTH = {
    "effective_date": "January 1, 2026",
    "resident_price_usd": "$80",
    "non_resident_price_usd": "$250",
    "additional_non_resident_fee_usd": "$100",
    "additional_fee_scope": "11 of the most visited national parks",
    "age_rule": "non-residents age 16 and older",
    "context": "on top of standard entrance fees"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PricingExtraction(BaseModel):
    """
    Extract the pricing statements from the answer along with the cited URLs.
    All monetary fields should be strings exactly as the answer presents (e.g., '$80', 'USD 250', '100 dollars').
    URL fields should contain only valid URLs explicitly present in the answer.
    """
    resident_price: Optional[str] = None
    resident_sources: List[str] = Field(default_factory=list)

    non_resident_price: Optional[str] = None
    non_resident_sources: List[str] = Field(default_factory=list)

    additional_fee: Optional[str] = None  # The additional per-person fee amount for non-residents 16+ without a pass
    additional_fee_sources: List[str] = Field(default_factory=list)

    # Any URLs cited in the answer relevant to this topic, not already listed above
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pricing() -> str:
    return """
    Extract the answer's stated pricing for the America the Beautiful Annual Pass effective starting January 1, 2026, and collect the URLs cited as sources.

    Required fields to extract (use null if not mentioned):
    - resident_price: The price the answer claims for a U.S. resident America the Beautiful Annual Pass (string, keep currency formatting as-is, e.g., "$80").
    - resident_sources: All URLs explicitly cited in the answer that support the resident pass price (array of URLs).

    - non_resident_price: The price the answer claims for a non-U.S. resident America the Beautiful Annual Pass (string, as written).
    - non_resident_sources: All URLs explicitly cited in the answer that support the non-resident pass price (array of URLs).

    - additional_fee: The per-person additional fee amount that non-residents age 16 and older without an annual pass must pay at certain national parks, on top of standard entrance fees (string, as written, e.g., "$100").
    - additional_fee_sources: All URLs explicitly cited in the answer that support this additional fee (array of URLs).

    - general_sources: Any other URLs cited in the answer that pertain to this topic but are not already included in the above source arrays (array of URLs).

    Rules:
    - Extract prices and fee amounts exactly as shown in the answer; do not normalize or add currency symbols.
    - Only include URLs that are explicitly present in the answer text. If the answer uses Markdown links [text](url), extract the underlying URLs.
    - If a field is not mentioned in the answer, set it to null (for strings) or an empty list (for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper: union of sources                                                    #
# --------------------------------------------------------------------------- #
def _merge_sources(*url_lists: List[str]) -> List[str]:
    """Merge and de-duplicate multiple lists of URLs while preserving order."""
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            if isinstance(u, str):
                u_str = u.strip()
                if u_str and u_str not in seen:
                    seen.add(u_str)
                    merged.append(u_str)
    return merged


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_pricing(
    evaluator: Evaluator,
    parent_node,
    extraction: PricingExtraction
) -> None:
    """
    Build the verification tree according to the rubric and run the three critical checks.
    """

    # 1) U.S. Resident Annual Pass Price
    node_resident = evaluator.add_leaf(
        id="US_Resident_Annual_Pass_Price",
        desc="The price for a U.S. resident America the Beautiful Annual Pass starting January 1, 2026, is $80.",
        parent=parent_node,
        critical=True
    )

    resident_claim = (
        f"Starting January 1, 2026, the price for a U.S. resident America the Beautiful Annual Pass is $80."
    )
    resident_sources = _merge_sources(extraction.resident_sources, extraction.general_sources)

    await evaluator.verify(
        claim=resident_claim,
        node=node_resident,
        sources=resident_sources if resident_sources else None,
        additional_instruction=(
            "Verify that the cited page(s) explicitly state the America the Beautiful Annual Pass (National Parks and "
            "Federal Recreational Lands Pass) price for U.S. residents is $80 effective January 1, 2026. "
            "Do not rely on older pricing unless the page clearly announces the new 2026 rate. "
            "Allow minor wording variations (e.g., 'America the Beautiful—the National Parks and Federal Recreational Lands Pass'). "
            "If the page does not clearly reference the 2026 effective date or $80, the claim is not supported."
        )
    )

    # 2) Non-U.S. Resident Annual Pass Price
    node_nonresident = evaluator.add_leaf(
        id="Non_Resident_Annual_Pass_Price",
        desc="The price for a non-U.S. resident America the Beautiful Annual Pass starting January 1, 2026, is $250.",
        parent=parent_node,
        critical=True
    )

    nonresident_claim = (
        f"Starting January 1, 2026, the price for a non-U.S. resident America the Beautiful Annual Pass is $250."
    )
    nonresident_sources = _merge_sources(extraction.non_resident_sources, extraction.general_sources)

    await evaluator.verify(
        claim=nonresident_claim,
        node=node_nonresident,
        sources=nonresident_sources if nonresident_sources else None,
        additional_instruction=(
            "Verify that the cited page(s) explicitly state a distinct non-U.S. resident price of $250 for the "
            "America the Beautiful Annual Pass effective January 1, 2026. "
            "If a page only lists the standard (resident) price or does not clearly distinguish non-resident pricing, "
            "the claim is not supported."
        )
    )

    # 3) Additional Non-Resident Fee at 11 Parks
    node_additional_fee = evaluator.add_leaf(
        id="Additional_Non_Resident_Fee",
        desc=(
            "Non-residents age 16 and older without an annual pass must pay an additional $100 per-person fee "
            "(on top of standard entrance fees) at 11 of the most visited national parks, starting January 1, 2026."
        ),
        parent=parent_node,
        critical=True
    )

    additional_fee_claim = (
        "Starting January 1, 2026, non-residents age 16 and older who do not have an annual pass must pay an additional "
        "$100 per person on top of the standard entrance fee at 11 of the most visited national parks."
    )
    additional_fee_sources = _merge_sources(extraction.additional_fee_sources, extraction.general_sources)

    await evaluator.verify(
        claim=additional_fee_claim,
        node=node_additional_fee,
        sources=additional_fee_sources if additional_fee_sources else None,
        additional_instruction=(
            "Check the cited page(s) for explicit statements that: "
            "(1) the policy applies to non-residents age 16+ without an annual pass, "
            "(2) the extra fee is $100 per person, "
            "(3) the fee is in addition to the standard entrance fee, "
            "(4) it applies at 11 of the most visited national parks, "
            "and (5) the effective date is January 1, 2026. "
            "All these conditions must be supported by the source(s)."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the America the Beautiful Annual Pass 2026 pricing task.
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

    # Add top-level critical node reflecting the rubric's main category
    main_node = evaluator.add_parallel(
        id="America_the_Beautiful_Pass_2026_Pricing",
        desc="Verify the correct pricing information for the America the Beautiful Annual Pass effective January 1, 2026, including both resident and non-resident prices, as well as the additional fee structure.",
        parent=root,
        critical=True
    )

    # Extract structured pricing info and cited sources from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_pricing(),
        template_class=PricingExtraction,
        extraction_name="pricing_extraction"
    )

    # Record ground truth (for transparency in the final summary)
    evaluator.add_ground_truth(
        {
            "expected_effective_date": GROUND_TRUTH["effective_date"],
            "expected_resident_price": GROUND_TRUTH["resident_price_usd"],
            "expected_non_resident_price": GROUND_TRUTH["non_resident_price_usd"],
            "expected_additional_non_resident_fee": GROUND_TRUTH["additional_non_resident_fee_usd"],
            "expected_additional_fee_scope": GROUND_TRUTH["additional_fee_scope"],
            "expected_age_rule": GROUND_TRUTH["age_rule"],
            "expected_context": GROUND_TRUTH["context"],
        },
        gt_type="ground_truth_pricing"
    )

    # Build tree and run verifications
    await build_and_verify_pricing(evaluator, main_node, extraction)

    # Return evaluation summary
    return evaluator.get_summary()