import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "utah_trip_cost_planning_2025"
TASK_DESCRIPTION = """A retired couple, both age 65, is planning an 8-day road trip through Utah from July 10-17, 2025. They will be traveling in their personal vehicle and plan to visit three of Utah's "Mighty 5" national parks: Arches National Park, Zion National Park, and Bryce Canyon National Park. They also plan to camp at national forest campgrounds for 4 nights total during their trip.

Please provide a comprehensive cost analysis and planning guide that includes:

1. Cost Comparison: Calculate the total cost of entrance fees if they pay individually at each park versus purchasing a Senior Pass (either Annual or Lifetime). Based on this comparison, recommend whether they should purchase a Senior Pass and specify which type (Annual $20 or Lifetime $80), with clear justification showing the cost savings.

2. Park-Specific Requirements: For each of the three national parks, provide:
   - The entrance fee per vehicle
   - Any timed entry reservation requirements that apply during their July 10-17 travel dates
   - Any additional costs for required reservations

3. Camping Cost Information: Provide:
   - The typical nightly rate range for national forest developed campgrounds
   - The discount percentage that Senior Pass holders receive on campsite fees
   - Information about how far in advance campground reservations can be made on Recreation.gov and any associated reservation fees

Your answer should demonstrate clear cost calculations and help them understand all advance planning requirements for their July 2025 trip.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkInfo(BaseModel):
    vehicle_entrance_fee: Optional[str] = None
    timed_entry_required: Optional[str] = None  # "yes" / "no" / None
    reservation_cost_amount: Optional[str] = None  # e.g., "$2" or "None"
    reservation_cost_description: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SeniorPassInfo(BaseModel):
    annual_price: Optional[str] = None  # e.g., "$20"
    lifetime_price: Optional[str] = None  # e.g., "$80"
    sources: List[str] = Field(default_factory=list)


class CostComparisonInfo(BaseModel):
    individual_total: Optional[str] = None  # e.g., "$100"
    annual_total: Optional[str] = None  # e.g., "$22" if includes $2 reservation fee
    lifetime_total: Optional[str] = None  # e.g., "$82" if includes $2 reservation fee
    included_reservation_fees: Optional[str] = None  # "yes"/"no" if the totals include required reservation fees
    sources: List[str] = Field(default_factory=list)


class RecommendationInfo(BaseModel):
    recommended_pass_type: Optional[str] = None  # e.g., "Senior Annual Pass" or "Senior Lifetime Pass"
    justification: Optional[str] = None  # free text justification


class CampingInfo(BaseModel):
    nightly_rate_range: Optional[str] = None  # e.g., "$14–$30"
    senior_discount_percent: Optional[str] = None  # e.g., "50%"
    advance_window: Optional[str] = None  # e.g., "6 months"
    reservation_fee_amount: Optional[str] = None  # e.g., "$8"
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_park_info(park_name: str) -> str:
    return f"""
    Extract the specific planning details for {park_name} as presented in the answer.

    Required fields:
    1) vehicle_entrance_fee: The stated per-vehicle (private non-commercial vehicle) entrance fee for {park_name}. Return the exact text (e.g., "$30").
    2) timed_entry_required: During the travel dates July 10–17, 2025, does a timed-entry reservation apply for {park_name}? Return "yes" or "no" based on the answer's claim.
    3) reservation_cost_amount: The stated dollar amount of any required reservation fee for those dates (e.g., "$2"). Return null if none or not applicable.
    4) reservation_cost_description: A short text describing the reservation and fee (e.g., "Timed-entry ticket $2 per reservation").
    5) sources: All URLs the answer cites for {park_name} (NPS pages, Recreation.gov pages, etc.).

    If any item is missing from the answer, set it to null.
    """


def prompt_extract_senior_pass_info() -> str:
    return """
    Extract the Senior Pass pricing information stated in the answer.

    Required fields:
    1) annual_price: The stated Senior Annual Pass price (e.g., "$20").
    2) lifetime_price: The stated Senior Lifetime Pass price (e.g., "$80").
    3) sources: All URLs cited for Senior Pass pricing (e.g., NPS, USGS store).

    If any item is missing from the answer, set it to null.
    """


def prompt_extract_cost_comparison() -> str:
    return """
    Extract the computed entrance-fee totals and comparison as presented in the answer.

    Required fields:
    1) individual_total: The total cost of entrance fees if paying individually at each park (e.g., "$100").
    2) annual_total: The total trip entrance-fee cost under the Senior Annual Pass (e.g., "$22" if $20 pass + $2 reservation fee).
    3) lifetime_total: The total trip entrance-fee cost under the Senior Lifetime Pass (e.g., "$82" if $80 pass + $2 reservation fee).
    4) included_reservation_fees: "yes" or "no" to indicate whether the above totals include any required reservation fees applicable to July 10–17, 2025.
    5) sources: All URLs cited for cost comparison or fee policies (if any).

    If any item is missing from the answer, set it to null.
    """


def prompt_extract_recommendation() -> str:
    return """
    Extract the recommendation and justification from the answer.

    Required fields:
    1) recommended_pass_type: Which Senior Pass type is recommended (e.g., "Senior Annual Pass" or "Senior Lifetime Pass").
    2) justification: A short summary of the stated justification (e.g., savings compared to individual fees, trip frequency, etc.).

    If any item is missing from the answer, set it to null.
    """


def prompt_extract_camping_info() -> str:
    return """
    Extract camping cost and reservation policy details for national forest developed campgrounds as stated in the answer.

    Required fields:
    1) nightly_rate_range: The typical nightly rate range (e.g., "$14–$30 per night for single-family sites").
    2) senior_discount_percent: The Senior Pass camping discount percentage (e.g., "50%").
    3) advance_window: How far in advance reservations can be made on Recreation.gov (e.g., "6 months").
    4) reservation_fee_amount: The Recreation.gov online reservation fee per reservation (e.g., "$8").
    5) sources: All URLs cited for the above policies.

    If any item is missing from the answer, set it to null.
    """


# --------------------------------------------------------------------------- #
# Helper parsing functions                                                    #
# --------------------------------------------------------------------------- #
def _money_to_float(s: Optional[str]) -> Optional[float]:
    """Extract first monetary amount as float from a string like '$35', '35 USD', or 'about $2'."""
    if not s:
        return None
    # Replace common en-dash/range separators for cleanliness
    s_norm = s.replace("–", "-").replace("—", "-")
    # Find first number token
    m = re.search(r"(\d+(?:[.,]\d+)?)", s_norm)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except Exception:
        return None


def _normalize_yes_no(s: Optional[str]) -> Optional[bool]:
    """Interpret yes/no from free text."""
    if s is None:
        return None
    t = s.strip().lower()
    positives = ["yes", "true", "required", "applies"]
    negatives = ["no", "false", "not required", "does not apply", "none"]
    if any(p in t for p in positives) and not any(n in t for n in negatives):
        return True
    if any(n in t for n in negatives) and not any(p in t for p in positives):
        return False
    return None


def _approx_equal(a: Optional[float], b: Optional[float], tol: float = 1.0) -> bool:
    """Compare two floats with tolerance."""
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_cost_comparison(
    evaluator: Evaluator,
    parent_node,
    arches: ParkInfo,
    zion: ParkInfo,
    bryce: ParkInfo,
    senior_pass: SeniorPassInfo,
    cost_comp: CostComparisonInfo,
    recommendation: RecommendationInfo,
) -> None:
    """
    Build the 'cost_comparison' subtree and perform checks.
    """
    comp_node = evaluator.add_parallel(
        id="cost_comparison",
        desc="Compare paying per-park entrance fees vs buying a Senior Pass (Annual or Lifetime) and provide a justified recommendation",
        parent=parent_node,
        critical=True,
    )

    # 1) individual_entrance_fee_total_calculation (custom check)
    node_sum = evaluator.add_custom_node(
        result=False,  # temporary; will set below after calculation
        id="individual_entrance_fee_total_calculation",
        desc="Correctly identifies and sums the per-vehicle entrance fees for Arches, Zion, and Bryce Canyon to compute the total cost if paying individually",
        parent=comp_node,
        critical=True,
    )
    arches_fee = _money_to_float(arches.vehicle_entrance_fee)
    zion_fee = _money_to_float(zion.vehicle_entrance_fee)
    bryce_fee = _money_to_float(bryce.vehicle_entrance_fee)
    indiv_total_answer = _money_to_float(cost_comp.individual_total)

    if None not in (arches_fee, zion_fee, bryce_fee, indiv_total_answer):
        expected_indiv = arches_fee + zion_fee + bryce_fee
        node_sum.score = 1.0 if _approx_equal(expected_indiv, indiv_total_answer) else 0.0
        node_sum.status = "passed" if node_sum.score == 1.0 else "failed"
    else:
        node_sum.score = 0.0
        node_sum.status = "failed"

    # 2) senior_pass_price_options (verify against sources if provided)
    node_prices = evaluator.add_leaf(
        id="senior_pass_price_options",
        desc="States the Senior Annual Pass price ($20) and Senior Lifetime Pass price ($80)",
        parent=comp_node,
        critical=True,
    )
    claim_prices = "The Senior Annual Pass price is $20 and the Senior Lifetime Pass price is $80."
    await evaluator.verify(
        claim=claim_prices,
        node=node_prices,
        sources=senior_pass.sources if senior_pass.sources else None,
        additional_instruction="Verify the stated Senior Pass prices using official NPS or USGS sources if available."
    )

    # 3) pass_vs_individual_cost_comparison (custom numerical consistency incl. reservation fees)
    node_compare = evaluator.add_custom_node(
        result=False,  # temporary; will compute below
        id="pass_vs_individual_cost_comparison",
        desc="Computes and compares total trip entrance-fee cost under (a) paying individually, (b) Senior Annual Pass, and (c) Senior Lifetime Pass, including any required reservation fees applicable to the travel dates",
        parent=comp_node,
        critical=True,
    )
    annual_price = _money_to_float(senior_pass.annual_price)
    lifetime_price = _money_to_float(senior_pass.lifetime_price)
    annual_total_answer = _money_to_float(cost_comp.annual_total)
    lifetime_total_answer = _money_to_float(cost_comp.lifetime_total)
    # Arches timed-entry is the only commonly required reservation fee among the three; include if applicable
    arches_timed_required = _normalize_yes_no(arches.timed_entry_required)
    arches_res_fee = _money_to_float(arches.reservation_cost_amount) if arches_timed_required else 0.0

    if None not in (annual_price, lifetime_price, indiv_total_answer, annual_total_answer, lifetime_total_answer):
        expected_indiv_total = (arches_fee or 0.0) + (zion_fee or 0.0) + (bryce_fee or 0.0) + (arches_res_fee or 0.0)
        expected_annual_total = annual_price + (arches_res_fee or 0.0)
        expected_lifetime_total = lifetime_price + (arches_res_fee or 0.0)

        ok_indiv = _approx_equal(expected_indiv_total, indiv_total_answer)
        ok_annual = _approx_equal(expected_annual_total, annual_total_answer)
        ok_life = _approx_equal(expected_lifetime_total, lifetime_total_answer)

        node_compare.score = 1.0 if (ok_indiv and ok_annual and ok_life) else 0.0
        node_compare.status = "passed" if node_compare.score == 1.0 else "failed"
    else:
        node_compare.score = 0.0
        node_compare.status = "failed"

    # 4) recommendation_with_justification (verify presence and justification referencing savings)
    node_reco = evaluator.add_leaf(
        id="recommendation_with_justification",
        desc="Recommends whether to buy a Senior Pass and which type (Annual vs Lifetime), with clear justification based on the computed cost differences/savings",
        parent=comp_node,
        critical=True,
    )
    reco_type = recommendation.recommended_pass_type or "a Senior Pass"
    claim_reco = f"The answer recommends purchasing {reco_type} and provides clear justification showing cost savings compared to paying individual park fees."
    await evaluator.verify(
        claim=claim_reco,
        node=node_reco,
        sources=cost_comp.sources if cost_comp.sources else None,
        additional_instruction="Check that the recommendation explicitly references the computed totals and savings, and clearly names the pass type (Annual $20 vs Lifetime $80)."
    )


async def verify_single_park(
    evaluator: Evaluator,
    parent_node,
    park_id: str,
    park_desc: str,
    park_info: ParkInfo,
    expected_fee_hint: Optional[str] = None
) -> None:
    """
    Build the park-specific subtree and perform checks.
    """
    park_node = evaluator.add_parallel(
        id=park_id,
        desc=park_desc,
        parent=parent_node,
        critical=True,
    )

    # Entrance fee leaf
    fee_node = evaluator.add_leaf(
        id=f"{park_id}_entrance_fee",
        desc=f"States {park_id.replace('_', ' ').title()} entrance fee per vehicle ({expected_fee_hint or 'expected amount'} for the relevant pass period)",
        parent=park_node,
        critical=True,
    )
    fee_val = park_info.vehicle_entrance_fee or "the correct per-vehicle entrance fee"
    claim_fee = f"The per-vehicle (private vehicle) entrance fee at {park_desc.split(' requirements')[0]} is {fee_val}."
    await evaluator.verify(
        claim=claim_fee,
        node=fee_node,
        sources=park_info.sources if park_info.sources else None,
        additional_instruction="Verify the non-commercial private vehicle entrance fee (typically valid for 7 days)."
    )

    # Timed-entry applicability leaf
    timed_node = evaluator.add_leaf(
        id=f"{park_id}_timed_entry_applicability",
        desc=f"Correctly states whether {park_id.replace('_', ' ').title()} timed-entry reservations apply during July 10–17, 2025",
        parent=park_node,
        critical=True,
    )
    timed_required = _normalize_yes_no(park_info.timed_entry_required)
    if timed_required is True:
        claim_timed = f"During July 10–17, 2025, a timed-entry reservation is required to enter {park_desc.split(' requirements')[0]}."
    elif timed_required is False:
        claim_timed = f"During July 10–17, 2025, no timed-entry reservation is required to enter {park_desc.split(' requirements')[0]}."
    else:
        claim_timed = f"The answer clearly states the applicability of timed-entry reservations for {park_desc.split(' requirements')[0]} during July 10–17, 2025."
    await evaluator.verify(
        claim=claim_timed,
        node=timed_node,
        sources=park_info.sources if park_info.sources else None,
        additional_instruction="Check the park's official timed-entry program date windows and determine if July 10–17, 2025 falls within them."
    )

    # Required reservation costs leaf
    res_cost_node = evaluator.add_leaf(
        id=f"{park_id}_required_reservation_costs",
        desc=f"States the additional cost for any required {park_id.replace('_', ' ').title()} reservation(s) during those dates (or indicates none)",
        parent=park_node,
        critical=True,
    )
    if timed_required:
        res_amt = park_info.reservation_cost_amount or "the correct fee"
        claim_cost = f"If a timed-entry reservation is required during July 10–17, 2025 at {park_desc.split(' requirements')[0]}, the reservation fee is {res_amt} per reservation."
    else:
        claim_cost = f"There are no additional reservation costs required during July 10–17, 2025 at {park_desc.split(' requirements')[0]}."
    await evaluator.verify(
        claim=claim_cost,
        node=res_cost_node,
        sources=park_info.sources if park_info.sources else None,
        additional_instruction="Verify whether a timed-entry (or similar) reservation fee applies for those dates, and confirm the stated amount if applicable."
    )


async def verify_camping_info(
    evaluator: Evaluator,
    parent_node,
    camping: CampingInfo,
) -> None:
    """
    Build the 'camping_cost_information' subtree and perform checks.
    """
    camp_node = evaluator.add_parallel(
        id="camping_cost_information",
        desc="Camping cost and reservation details for national forest developed campgrounds, including Senior Pass discount and Recreation.gov policies",
        parent=parent_node,
        critical=True,
    )

    # Nightly rate range
    rate_node = evaluator.add_leaf(
        id="nightly_rate_range",
        desc="Provides the typical nightly rate range for national forest developed campgrounds ($14–$30 per night for single-family sites)",
        parent=camp_node,
        critical=True,
    )
    rate_rng = camping.nightly_rate_range or "a typical $14–$30 per night range for single-family sites"
    claim_rate = f"The typical nightly rate range for national forest developed campgrounds is {rate_rng} per night for single-family sites."
    await evaluator.verify(
        claim=claim_rate,
        node=rate_node,
        sources=camping.sources if camping.sources else None,
        additional_instruction="Confirm typical price ranges for developed campgrounds in national forests; allow reasonable regional variation but ensure the claimed range matches cited sources."
    )

    # Senior pass camping discount
    disc_node = evaluator.add_leaf(
        id="senior_pass_camping_discount",
        desc="States the Senior Pass camping discount percentage (50% off eligible amenity/campsite fees as specified)",
        parent=camp_node,
        critical=True,
    )
    disc_pct = camping.senior_discount_percent or "50%"
    claim_disc = f"Senior Pass holders receive {disc_pct} off eligible camping or amenity fees at participating federal recreation sites."
    await evaluator.verify(
        claim=claim_disc,
        node=disc_node,
        sources=camping.sources if camping.sources else None,
        additional_instruction="Verify that the Interagency Senior Pass provides a 50% discount on eligible camping fees (not all fees may be discounted)."
    )

    # Recreation.gov advance window
    adv_node = evaluator.add_leaf(
        id="recreation_gov_advance_window",
        desc="States how far in advance campground reservations can be made on Recreation.gov (typically 6 months in advance)",
        parent=camp_node,
        critical=True,
    )
    adv_window = camping.advance_window or "6 months"
    claim_adv = f"Campground reservations on Recreation.gov can typically be made up to {adv_window} in advance for individual campsites."
    await evaluator.verify(
        claim=claim_adv,
        node=adv_node,
        sources=camping.sources if camping.sources else None,
        additional_instruction="Confirm the typical advance reservation window (commonly 6 months for individual campsites; some facilities differ)."
    )

    # Recreation.gov reservation fee
    fee_node = evaluator.add_leaf(
        id="recreation_gov_reservation_fee",
        desc="States the Recreation.gov online reservation fee ($8 per reservation)",
        parent=camp_node,
        critical=True,
    )
    res_fee_amt = camping.reservation_fee_amount or "$8"
    claim_res_fee = f"Recreation.gov charges an {res_fee_amt} online reservation fee per reservation."
    await evaluator.verify(
        claim=claim_res_fee,
        node=fee_node,
        sources=camping.sources if camping.sources else None,
        additional_instruction="Verify the standard Recreation.gov reservation service fee (commonly $8 per reservation) as stated in the answer."
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
    Evaluate the comprehensive cost analysis and planning guide for the Utah trip in July 2025.
    """
    # Initialize evaluator with a parallel root (allow partial credit across sub-areas)
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

    # Parallelize extractions
    arches_task = evaluator.extract(
        prompt=prompt_extract_park_info("Arches National Park"),
        template_class=ParkInfo,
        extraction_name="arches_info",
    )
    zion_task = evaluator.extract(
        prompt=prompt_extract_park_info("Zion National Park"),
        template_class=ParkInfo,
        extraction_name="zion_info",
    )
    bryce_task = evaluator.extract(
        prompt=prompt_extract_park_info("Bryce Canyon National Park"),
        template_class=ParkInfo,
        extraction_name="bryce_info",
    )
    senior_task = evaluator.extract(
        prompt=prompt_extract_senior_pass_info(),
        template_class=SeniorPassInfo,
        extraction_name="senior_pass_info",
    )
    cost_task = evaluator.extract(
        prompt=prompt_extract_cost_comparison(),
        template_class=CostComparisonInfo,
        extraction_name="cost_comparison_info",
    )
    reco_task = evaluator.extract(
        prompt=prompt_extract_recommendation(),
        template_class=RecommendationInfo,
        extraction_name="recommendation_info",
    )
    camp_task = evaluator.extract(
        prompt=prompt_extract_camping_info(),
        template_class=CampingInfo,
        extraction_name="camping_info",
    )

    arches, zion, bryce, senior_pass, cost_comp, recommendation, camping = await asyncio.gather(
        arches_task, zion_task, bryce_task, senior_task, cost_task, reco_task, camp_task
    )

    # Optional: Add ground truth info for reference in summary (not used for verification)
    evaluator.add_ground_truth({
        "expected_vehicle_fees": {
            "Arches": "$30",
            "Zion": "$35",
            "Bryce Canyon": "$35"
        },
        "senior_pass_prices": {"annual": "$20", "lifetime": "$80"},
        "common_timed_entry": {"Arches": "Yes (summer window)", "Zion": "No", "Bryce Canyon": "No"},
        "camping_reference": {
            "nightly_rate_range": "$14–$30",
            "senior_discount_percent": "50%",
            "rec_gov_advance_window": "6 months (typical)",
            "rec_gov_reservation_fee": "$8 per reservation"
        }
    }, gt_type="reference_expectations")

    # 1) Cost Comparison subtree
    await verify_cost_comparison(
        evaluator=evaluator,
        parent_node=root,
        arches=arches,
        zion=zion,
        bryce=bryce,
        senior_pass=senior_pass,
        cost_comp=cost_comp,
        recommendation=recommendation
    )

    # 2) Park-Specific Requirements subtree
    psr_node = evaluator.add_parallel(
        id="park_specific_requirements",
        desc="For each of the three parks, provide entrance fee per vehicle, timed-entry reservation applicability during July 10–17, 2025, and any required reservation costs",
        parent=root,
        critical=True,
    )

    await verify_single_park(
        evaluator=evaluator,
        parent_node=psr_node,
        park_id="arches",
        park_desc="Arches National Park requirements and costs for July 10–17, 2025",
        park_info=arches,
        expected_fee_hint="$30"
    )
    await verify_single_park(
        evaluator=evaluator,
        parent_node=psr_node,
        park_id="zion",
        park_desc="Zion National Park requirements and costs for July 10–17, 2025",
        park_info=zion,
        expected_fee_hint="$35"
    )
    await verify_single_park(
        evaluator=evaluator,
        parent_node=psr_node,
        park_id="bryce_canyon",
        park_desc="Bryce Canyon National Park requirements and costs for July 10–17, 2025",
        park_info=bryce,
        expected_fee_hint="$35"
    )

    # 3) Camping Cost Information subtree
    await verify_camping_info(
        evaluator=evaluator,
        parent_node=root,
        camping=camping
    )

    # Return structured summary
    return evaluator.get_summary()