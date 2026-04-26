import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "yellowstone_entrance_fee_2026"
TASK_DESCRIPTION = (
    "A family of non-US residents is planning a trip to Yellowstone National Park in February 2026. "
    "The family consists of 2 adults (both 25 years old or older) and 2 children (ages 14 and 10). "
    "They will be traveling by private vehicle and plan to enter the park on 3 separate days during their "
    "10-day stay in the Wyoming region. Based on the 2026 entrance fee structure for Yellowstone National Park, "
    "what is the most cost-effective entrance fee payment method for this family, and what is the total entrance "
    "cost they should budget?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FamilyExtraction(BaseModel):
    """
    Extract family composition and plan details exactly as stated in the answer.
    """
    adults_count: Optional[str] = None
    adults_ages: List[str] = Field(default_factory=list)
    children_count: Optional[str] = None
    children_ages: List[str] = Field(default_factory=list)
    entry_days_planned: Optional[str] = None       # e.g., "3"
    stay_days_planned: Optional[str] = None        # e.g., "10"
    adult_fee_requirement_statement: Optional[str] = None  # any explicit statement in answer
    children_fee_status_statement: Optional[str] = None    # e.g., "children under 16 are free"
    policy_urls: List[str] = Field(default_factory=list)   # any URLs cited about fee policy (ages, etc.)


class FeeOptionsExtraction(BaseModel):
    """
    Extract fee option details, costs, and sources exactly as stated in the answer.
    """
    # Standard/vehicle pass details
    standard_vehicle_pass_price: Optional[str] = None      # e.g., "$35"
    standard_vehicle_validity: Optional[str] = None        # e.g., "7 days"
    nonresident_fee_per_person: Optional[str] = None       # e.g., "$100"
    standard_pass_periods_assumed: Optional[str] = None    # e.g., "2" if answer assumes two 7-day periods

    # Annual pass details
    annual_pass_name: Optional[str] = None                 # e.g., "America the Beautiful Non-Resident Annual Pass"
    annual_pass_price: Optional[str] = None                # e.g., "$250"
    annual_pass_covers_vehicle_statement: Optional[str] = None  # e.g., "covers all passengers in the vehicle"
    annual_pass_waives_nonresident_fee_statement: Optional[str] = None  # e.g., "waives per-person nonresident fee"

    # Decision and total
    chosen_payment_method: Optional[str] = None            # e.g., "Non-Resident Annual Pass"
    total_entrance_cost: Optional[str] = None              # e.g., "$250"

    # Sources
    standard_pass_urls: List[str] = Field(default_factory=list)
    annual_pass_urls: List[str] = Field(default_factory=list)
    other_fee_urls: List[str] = Field(default_factory=list)  # other URLs (policy/2026 fee references)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_family_info() -> str:
    return """
    Extract the family composition and visit plan details exactly as presented in the answer.

    Required fields:
    - adults_count: number of adults explicitly mentioned in the answer (as a string). If missing, return null.
    - adults_ages: list of ages for the adults exactly as described (e.g., ["25", "27"]). If ages are not listed, return an empty list.
    - children_count: number of children explicitly mentioned in the answer (as a string). If missing, return null.
    - children_ages: list of ages for the children exactly as described (e.g., ["14", "10"]). If ages are not listed, return an empty list.
    - entry_days_planned: the number of separate entry days into the park mentioned in the answer (as a string). If missing, return null.
    - stay_days_planned: the number of stay days in the region mentioned in the answer (as a string). If missing, return null.
    - adult_fee_requirement_statement: if the answer explicitly states whether adults are subject to entrance fees (or nonresident fees), copy that sentence or phrase verbatim; otherwise null.
    - children_fee_status_statement: if the answer explicitly states that children under 16 are not subject to entrance fees, copy that sentence or phrase verbatim; otherwise null.
    - policy_urls: extract all URLs in the answer that appear to relate to age-based entrance fee policy or who pays fees. Include valid URLs only.

    Rules:
    - Extract only what appears in the answer; do not infer.
    - If any field is missing from the answer, set it to null or empty as instructed.
    """


def prompt_extract_fee_options() -> str:
    return """
    Extract fee option details, costs, and payment decision exactly as presented in the answer.

    Required fields:
    STANDARD / VEHICLE PASS:
    - standard_vehicle_pass_price: the stated price for a standard private vehicle pass (string, e.g., "$35").
    - standard_vehicle_validity: the stated validity period (string, e.g., "7 days").
    - nonresident_fee_per_person: if the answer claims an additional per-person nonresident fee applies with standard passes, extract the amount (string, e.g., "$100"); else null.
    - standard_pass_periods_assumed: how many 7-day standard pass periods the answer assumes for the described 10-day stay and 3 entries (string integer like "1" or "2"), if explicitly stated or clearly implied; else null.

    ANNUAL PASS:
    - annual_pass_name: the name of the annual pass (string).
    - annual_pass_price: the stated price (string, e.g., "$250").
    - annual_pass_covers_vehicle_statement: if the answer states it covers all passengers in the vehicle, copy the exact phrase; else null.
    - annual_pass_waives_nonresident_fee_statement: if the answer states it waives per-person nonresident fees, copy the phrase; else null.

    DECISION & TOTAL:
    - chosen_payment_method: which payment option the answer recommends as most cost-effective (string).
    - total_entrance_cost: the total entrance cost the answer says to budget (string, e.g., "$250").

    SOURCES:
    - standard_pass_urls: all URLs the answer cites specifically about standard/private vehicle pass prices and validity.
    - annual_pass_urls: all URLs the answer cites specifically about the annual pass (price/coverage).
    - other_fee_urls: all other URLs that the answer cites about fee policy or 2026-specific information.

    Rules:
    - Extract only what appears in the answer; do not invent or infer.
    - If any field is missing, set it to null or empty list as instructed.
    - Extract complete, valid URLs only (markdown links ok; return the actual URL).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _parse_usd_amount(value: Optional[str]) -> Optional[float]:
    """Parse a USD amount like '$250' or '250 USD' to a float; return None if not parseable."""
    if not value:
        return None
    # Find first number (with optional decimal)
    m = re.search(r"(\d+(?:\.\d+)?)", value.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    """Parse a simple integer from a string; return None if not parseable."""
    if not value:
        return None
    m = re.search(r"(-?\d+)", value)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _union_urls(*lists: List[str]) -> List[str]:
    """Deduplicate and preserve order for multiple URL lists."""
    seen = set()
    result = []
    for lst in lists:
        for u in lst:
            if u and u not in seen:
                seen.add(u)
                result.append(u)
    return result


def _compute_costs(
    adults: int,
    periods: int,
    std_price: Optional[float],
    nonresident_per_person: Optional[float],
    annual_price: Optional[float],
) -> Dict[str, Optional[float]]:
    """
    Compute total costs for:
      - Standard vehicle passes (with per-person nonresident fee if provided), multiplied by periods.
      - Annual pass price.
    Any missing inputs yield None for that option.
    """
    standard_total = None
    if std_price is not None:
        # If nonresident fee per person applies, add for each adult per period
        extra = (nonresident_per_person or 0.0) * adults
        standard_total = periods * (std_price + extra)

    annual_total = annual_price if annual_price is not None else None
    return {"standard_total": standard_total, "annual_total": annual_total}


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_family_composition(
    evaluator: Evaluator,
    parent_node,
    fam: FamilyExtraction,
    fee: FeeOptionsExtraction,
) -> None:
    """
    Build 'Family_Composition_Analysis' parallel subtree.
    """
    fam_node = evaluator.add_parallel(
        id="Family_Composition_Analysis",
        desc="Correctly identifies the number of adults and children subject to entrance fees based on the given family composition",
        parent=parent_node,
        critical=True,
    )

    # Adults fee requirement leaf
    adults_leaf = evaluator.add_leaf(
        id="Adults_Fee_Requirement",
        desc="Correctly identifies that both adults (age 25+) are subject to the nonresident fee as they are 16 and over",
        parent=fam_node,
        critical=True,
    )
    # Claim: Verify policy that visitors 16+ pay entrance/are subject to fees (and as framed in the answer, nonresident fee applies if claimed)
    adults_claim = (
        "Visitors who are 16 years of age or older are subject to entrance fees at Yellowstone National Park. "
        "Therefore, the two adults (each 25+) in this family are subject to entrance fee requirements. "
        "If the answer claims a nonresident per-person fee applies for non-US visitors aged 16+, verify that as well."
    )
    adults_sources = _union_urls(fam.policy_urls, fee.other_fee_urls, fee.standard_pass_urls)
    await evaluator.verify(
        claim=adults_claim,
        node=adults_leaf,
        sources=adults_sources if adults_sources else None,
        additional_instruction="Focus on official fee policy statements. Confirm age-based applicability (16+) and, if claimed in the answer, the existence of a nonresident per-person fee policy.",
    )

    # Children fee status leaf
    children_leaf = evaluator.add_leaf(
        id="Children_Fee_Status",
        desc="Correctly identifies that both children (ages 14 and 10) are under 16 and therefore not subject to entrance fees",
        parent=fam_node,
        critical=True,
    )
    children_claim = (
        "Children under 16 are not subject to entrance fees at Yellowstone National Park. "
        "Therefore, the children ages 14 and 10 are not charged entrance fees."
    )
    children_sources = _union_urls(fam.policy_urls, fee.other_fee_urls)
    await evaluator.verify(
        claim=children_claim,
        node=children_leaf,
        sources=children_sources if children_sources else None,
        additional_instruction="Verify explicit policy language that children under 16 are free (no entrance fee).",
    )


async def verify_payment_options_understanding(
    evaluator: Evaluator,
    parent_node,
    fee: FeeOptionsExtraction,
) -> None:
    """
    Build 'Payment_Options_Understanding' parallel subtree.
    We split multi-fact checks into granular leaves for clarity.
    """
    pay_node = evaluator.add_parallel(
        id="Payment_Options_Understanding",
        desc="Correctly understands the pricing structure and applicability of both standard entrance passes and the annual pass for this family's situation",
        parent=parent_node,
        critical=True,
    )

    # Standard pass details (sub-parallel)
    std_node = evaluator.add_parallel(
        id="Standard_Pass_Cost_Structure",
        desc="Standard private vehicle passes cost per 7-day period, and (if claimed) a nonresident per-person fee applies for 16+ with standard passes",
        parent=pay_node,
        critical=True,
    )
    # Leaf: base price and validity
    std_price_leaf = evaluator.add_leaf(
        id="Standard_Pass_Base_Price_Validity",
        desc="Correctly understands that standard private vehicle passes cost {price} per 7-day period",
        parent=std_node,
        critical=True,
    )
    std_price = fee.standard_vehicle_pass_price or ""
    std_validity = fee.standard_vehicle_validity or ""
    std_price_claim = (
        f"The standard private vehicle pass for Yellowstone National Park costs '{std_price}' and is valid for '{std_validity}' per pass."
    )
    await evaluator.verify(
        claim=std_price_claim,
        node=std_price_leaf,
        sources=fee.standard_pass_urls if fee.standard_pass_urls else None,
        additional_instruction="Verify the price and validity duration (typically 7 days per vehicle) as explicitly stated for 2026 if available.",
    )

    # Leaf: nonresident fee per person (only if the answer claims it)
    std_nonresident_leaf = evaluator.add_leaf(
        id="Standard_Pass_Nonresident_Fee",
        desc="Correctly understands that non-US residents aged 16+ must pay an additional per-person nonresident fee with standard passes (as claimed)",
        parent=std_node,
        critical=True,
    )
    nr_fee = fee.nonresident_fee_per_person or ""
    std_nonresident_claim = (
        f"The answer claims a nonresident per-person fee of '{nr_fee}' applies to non-US residents (aged 16+) when using standard passes; verify that such a policy exists and applies."
    )
    await evaluator.verify(
        claim=std_nonresident_claim,
        node=std_nonresident_leaf,
        sources=_union_urls(fee.other_fee_urls, fee.standard_pass_urls),
        additional_instruction="Confirm whether an official 2026 policy specifies an additional per-person nonresident fee for standard vehicle pass holders.",
    )

    # Annual pass details (sub-parallel)
    annual_node = evaluator.add_parallel(
        id="Annual_Pass_Cost_and_Coverage",
        desc="Correctly identifies that the Non-Resident Annual Pass cost and coverage details (price, coverage of all passengers, fee waiver) as claimed",
        parent=pay_node,
        critical=True,
    )

    # Leaf: annual pass price
    annual_price_leaf = evaluator.add_leaf(
        id="Annual_Pass_Price",
        desc="Correctly identifies the annual pass price as claimed",
        parent=annual_node,
        critical=True,
    )
    annual_price = fee.annual_pass_price or ""
    annual_price_claim = f"The non-resident annual pass price is stated as '{annual_price}'; verify that this price is correct."
    await evaluator.verify(
        claim=annual_price_claim,
        node=annual_price_leaf,
        sources=fee.annual_pass_urls if fee.annual_pass_urls else None,
        additional_instruction="Confirm the quoted annual pass price for 2026 from official sources cited in the answer.",
    )

    # Leaf: covers all passengers
    annual_cover_leaf = evaluator.add_leaf(
        id="Annual_Pass_Covers_All",
        desc="Correctly identifies that the annual pass covers all passengers in the vehicle",
        parent=annual_node,
        critical=True,
    )
    cover_stmt = fee.annual_pass_covers_vehicle_statement or ""
    annual_cover_claim = (
        f"The answer claims the annual pass covers all passengers in the vehicle (statement: '{cover_stmt}'); verify this coverage."
    )
    await evaluator.verify(
        claim=annual_cover_claim,
        node=annual_cover_leaf,
        sources=fee.annual_pass_urls if fee.annual_pass_urls else None,
        additional_instruction="Verify coverage details from official annual pass documentation cited in the answer.",
    )

    # Leaf: waives nonresident per-person fee
    annual_waive_leaf = evaluator.add_leaf(
        id="Annual_Pass_Waives_Nonresident_Fee",
        desc="Correctly identifies that the annual pass waives the per-person nonresident fee",
        parent=annual_node,
        critical=True,
    )
    waive_stmt = fee.annual_pass_waives_nonresident_fee_statement or ""
    annual_waive_claim = (
        f"The answer claims the annual pass waives per-person nonresident fees (statement: '{waive_stmt}'); verify that this waiver is accurate."
    )
    await evaluator.verify(
        claim=annual_waive_claim,
        node=annual_waive_leaf,
        sources=fee.annual_pass_urls if fee.annual_pass_urls else None,
        additional_instruction="Confirm any stated waiver of per-person nonresident fees for annual pass holders from official 2026 sources.",
    )


async def verify_costs_and_total(
    evaluator: Evaluator,
    parent_node,
    fam: FamilyExtraction,
    fee: FeeOptionsExtraction,
) -> None:
    """
    Build the sequential section:
      - Data sufficiency gate
      - Cost effectiveness comparison
      - Total cost amount correctness
    """
    seq_node = parent_node  # Already a sequential container per rubric

    # Gate: data sufficiency custom node (critical)
    adults_count = _parse_int(fam.adults_count) or 2  # fallback to scenario's 2 adults
    std_price_num = _parse_usd_amount(fee.standard_vehicle_pass_price)
    nr_fee_num = _parse_usd_amount(fee.nonresident_fee_per_person)
    annual_price_num = _parse_usd_amount(fee.annual_pass_price)
    periods = _parse_int(fee.standard_pass_periods_assumed)
    chosen_method = (fee.chosen_payment_method or "").strip()
    total_cost_str = (fee.total_entrance_cost or "").strip()
    total_cost_num = _parse_usd_amount(total_cost_str)

    # If periods is missing, we cannot reliably compute; require explicit assumption from the answer
    data_sufficient = (
        std_price_num is not None
        and annual_price_num is not None
        and periods is not None
        and chosen_method != ""
        and total_cost_num is not None
        and adults_count is not None
    )

    evaluator.add_custom_node(
        result=data_sufficient,
        id="Cost_Data_Sufficiency",
        desc="Sufficient cost data extracted to validate cost effectiveness and total budget (prices, periods, choice, total)",
        parent=seq_node,
        critical=True,
    )

    # Cost effectiveness comparison leaf (critical)
    compare_leaf = evaluator.add_leaf(
        id="Cost_Effectiveness_Comparison",
        desc="Correctly compares the total costs of different payment options for the family's planned 3 visits and identifies which option results in lower total cost",
        parent=seq_node,
        critical=True,
    )

    costs = _compute_costs(adults_count, periods or 0, std_price_num, nr_fee_num, annual_price_num)
    standard_total = costs["standard_total"]
    annual_total = costs["annual_total"]

    # Build claim based on available costs
    if standard_total is not None and annual_total is not None:
        # Determine which is lower
        lower_option = "Annual Pass" if annual_total <= standard_total else "Standard Vehicle Pass"
        chosen_clean = chosen_method.lower()
        lower_clean = lower_option.lower()
        compare_claim = (
            f"Based on the extracted prices and the assumed {periods} 7-day pass period(s), "
            f"the {lower_option} yields the lower total entrance cost for this family "
            f"(Standard total: ${standard_total:.2f}; Annual total: ${annual_total:.2f}). "
            f"The answer's recommended option ('{chosen_method}') should match the lower-cost option."
        )
    else:
        compare_claim = (
            "Insufficient data to compute a definitive cost comparison. "
            "If sufficient data was provided, the recommended option should correspond to the lower total cost."
        )

    await evaluator.verify(
        claim=compare_claim,
        node=compare_leaf,
        sources=None,
        additional_instruction=(
            "Use arithmetic with the extracted numeric values. "
            "Standard total = periods * (standard_price + nonresident_fee_per_person * adults). "
            "Annual total = annual_price. "
            "Judge if the recommended option matches the lower-cost option. Allow minor rounding."
        ),
    )

    # Total cost amount correctness leaf (critical)
    total_leaf = evaluator.add_leaf(
        id="Total_Cost_Amount",
        desc="Provides the specific total entrance cost amount that the family should budget based on the most cost-effective payment option identified",
        parent=seq_node,
        critical=True,
    )

    # Compute the expected total based on the lower option
    expected_total = None
    if standard_total is not None and annual_total is not None:
        expected_total = annual_total if annual_total <= standard_total else standard_total
    elif annual_total is not None:
        expected_total = annual_total
    elif standard_total is not None:
        expected_total = standard_total

    if expected_total is not None:
        total_claim = (
            f"The correct total entrance cost to budget, following the most cost-effective option, is approximately ${expected_total:.2f}. "
            f"Verify that this matches the total cost stated in the answer ('{total_cost_str}')."
        )
    else:
        total_claim = (
            f"The answer provides a total cost ('{total_cost_str}'), but insufficient data was extracted to compute an expected total for verification."
        )

    await evaluator.verify(
        claim=total_claim,
        node=total_leaf,
        sources=None,
        additional_instruction=(
            "Compare the stated total cost to the computed expected total from the cost comparison. "
            "Allow minor rounding differences (e.g., $250 vs $250.00)."
        ),
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Yellowstone 2026 entrance fee task.
    """
    # Initialize evaluator with root node as critical sequential
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Set root as the critical task node (all children must be critical by design)
    # The Evaluator.initialize creates a non-critical root by default; to fit rubric, we add a critical wrapper.
    # Instead, we make the first child our critical task container and attach all checks under it.
    task_node = evaluator.add_sequential(
        id="Yellowstone_Entrance_Fee_Task",
        desc="Correctly identifies the most cost-effective entrance fee option and provides the total entrance cost for a non-US resident family visiting Yellowstone National Park in 2026",
        parent=root,
        critical=True,
    )

    # Extract structured information from the answer
    fam_task = evaluator.extract(
        prompt=prompt_extract_family_info(),
        template_class=FamilyExtraction,
        extraction_name="family_info",
    )
    fee_task = evaluator.extract(
        prompt=prompt_extract_fee_options(),
        template_class=FeeOptionsExtraction,
        extraction_name="fee_options",
    )
    fam, fee = await asyncio.gather(fam_task, fee_task)

    # Subtree 1: Family composition analysis (parallel, critical)
    await verify_family_composition(evaluator, task_node, fam, fee)

    # Subtree 2: Cost options analysis and comparison (sequential, critical)
    analysis_node = evaluator.add_sequential(
        id="Cost_Options_Analysis_and_Comparison",
        desc="Correctly analyzes available payment options, calculates or determines their costs, and identifies which option is most cost-effective",
        parent=task_node,
        critical=True,
    )

    # Payment options understanding (parallel, critical)
    await verify_payment_options_understanding(evaluator, analysis_node, fee)

    # Cost effectiveness comparison and total cost amount (sequential continuation)
    await verify_costs_and_total(evaluator, analysis_node, fam, fee)

    # Return the evaluator summary
    return evaluator.get_summary()