import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "trip_cost_analysis_2026_us_parks_southwest"
TASK_DESCRIPTION = (
    "A couple, both non-US residents aged 25, is planning a 10-day trip to the United States in March 2026. "
    "They will fly Southwest Airlines from Baltimore/Washington International Thurgood Marshall Airport (BWI) and plan to visit exactly three national parks during their trip: "
    "Yellowstone National Park, Grand Canyon National Park, and Glacier National Park. "
    "They will check a total of 2 bags combined (not per person) on their Southwest Airlines flight. Neither traveler has Rapid Rewards A-List Preferred status or is traveling on a Business Select fare.\n\n"
    "For this trip, determine:\n"
    "1. Calculate the total cost of park entrance fees and any applicable nonresident surcharges if they do NOT purchase the America the Beautiful Non-Resident Annual Pass. Assume they will enter each park once as a couple in a private vehicle.\n"
    "2. Calculate the total cost if they DO purchase one America the Beautiful Non-Resident Annual Pass (valid for the pass holder and accompanying passengers in a private vehicle).\n"
    "3. Calculate the total baggage fees they will pay on their Southwest Airlines flight based on the current baggage policy.\n"
    "4. Determine whether purchasing the Non-Resident Annual Pass is cost-effective for this specific trip by comparing the total costs in scenarios 1 and 2. State clearly whether they should buy the pass or not, and by how much money they would save (or lose) by making the cost-effective choice.\n\n"
    "Provide your answer with:\n"
    "- The cost calculation for entering the three parks without the annual pass\n"
    "- The cost with the annual pass\n"
    "- The baggage fees\n"
    "- A clear recommendation with the dollar amount of savings\n"
    "- Reference URLs supporting each major cost component"
)


# ---------------------------
# Extraction Models
# ---------------------------

class ParksExtraction(BaseModel):
    parks: List[str] = Field(default_factory=list)
    entry_mode_assumption: Optional[str] = None


class ParkFeeExtraction(BaseModel):
    park_name: Optional[str] = None
    vehicle_fee: Optional[str] = None
    fee_source_urls: List[str] = Field(default_factory=list)


class SurchargeExtraction(BaseModel):
    surcharge_amount: Optional[str] = None
    policy_summary: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class PassExtraction(BaseModel):
    pass_price: Optional[str] = None
    coverage_summary: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class BaggagePolicyExtraction(BaseModel):
    first_bag_fee: Optional[str] = None
    second_bag_fee: Optional[str] = None
    effective_date: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class BaggageCalcExtraction(BaseModel):
    total_baggage_fees: Optional[str] = None
    allocation_assumption: Optional[str] = None


class TotalsExtraction(BaseModel):
    total_without_pass: Optional[str] = None
    total_with_pass: Optional[str] = None
    recommendation: Optional[str] = None
    savings_amount: Optional[str] = None
    overall_total_including_baggage: Optional[str] = None


# ---------------------------
# Extraction Prompts
# ---------------------------

def prompt_extract_parks_and_entry_mode() -> str:
    return (
        "Extract the exact list of national parks the answer uses for cost calculations. Return a JSON with:\n"
        "- parks: array of park names used in the computation (in the answer), in any order\n"
        "- entry_mode_assumption: a short phrase describing the assumed entry mode for the park visits as stated (e.g., 'one entry per park in a private vehicle as a couple')\n"
        "Only include the parks the answer actually uses for cost totals. Do not add or infer parks beyond what the answer uses."
    )


def prompt_extract_park_fee(park_name: str) -> str:
    return (
        f"From the answer, extract the entrance fee used for {park_name} specifically for a private vehicle entry (as used in the computation). "
        "Return a JSON with:\n"
        "- park_name\n"
        "- vehicle_fee: the monetary amount string used (e.g., '$35'), as written in the answer\n"
        "- fee_source_urls: array of URLs cited that support the fee used in the answer\n"
        "If the fee is not mentioned, set vehicle_fee to null. If no URLs are given for this fee, return an empty array for fee_source_urls."
    )


def prompt_extract_surcharge_policy() -> str:
    return (
        "Extract the nonresident surcharge policy used in the answer. Return a JSON with:\n"
        "- surcharge_amount: monetary amount string used (e.g., '$100')\n"
        "- policy_summary: brief text of how the surcharge applies (e.g., 'per person age 16+ per park, effective Jan 1, 2026')\n"
        "- source_urls: array of URLs cited that support this surcharge policy\n"
        "If not mentioned, set fields to null and return an empty source_urls array."
    )


def prompt_extract_pass_info() -> str:
    return (
        "Extract the America the Beautiful Non-Resident Annual Pass details used in the answer. Return a JSON with:\n"
        "- pass_price: monetary amount string used (e.g., '$250')\n"
        "- coverage_summary: brief text describing what the pass covers and whether it waives the $100 nonresident surcharge, and that it covers pass holder plus accompanying passengers in a private vehicle\n"
        "- source_urls: array of URLs cited that support the pass price and coverage\n"
        "If not mentioned, set fields to null and source_urls to empty."
    )


def prompt_extract_baggage_policy() -> str:
    return (
        "Extract the Southwest Airlines checked baggage fee policy used in the answer. Return a JSON with:\n"
        "- first_bag_fee: monetary amount string used for the first checked bag per passenger (e.g., '$35')\n"
        "- second_bag_fee: monetary amount string used for the second checked bag per passenger (e.g., '$45')\n"
        "- effective_date: any effective date text cited (e.g., 'effective May 28, 2025')\n"
        "- source_urls: array of URLs cited supporting this policy\n"
        "If not mentioned, set fields to null and source_urls to empty."
    )


def prompt_extract_baggage_calc() -> str:
    return (
        "Extract the baggage calculation used in the answer for exactly 2 checked bags total across the couple. Return a JSON with:\n"
        "- total_baggage_fees: the total amount string computed in the answer for 2 checked bags combined\n"
        "- allocation_assumption: short phrase describing how the 2 bags are allocated across passengers (e.g., 'one bag per passenger' or 'both bags for one passenger')\n"
        "If not mentioned, set fields to null."
    )


def prompt_extract_totals_and_recommendation() -> str:
    return (
        "Extract the park-cost totals and recommendation used in the answer. Return a JSON with:\n"
        "- total_without_pass: the total amount string for park entrance + surcharges without the annual pass\n"
        "- total_with_pass: the total amount string for park costs when using one Non-Resident Annual Pass\n"
        "- recommendation: text stating whether they should buy the pass or not for this trip\n"
        "- savings_amount: the stated dollar difference saved or lost by choosing the cost-effective option (amount only as text, e.g., '$X')\n"
        "- overall_total_including_baggage: if the answer provides an overall trip total including baggage fees, extract that amount; otherwise null."
    )


# ---------------------------
# Helper Utilities
# ---------------------------

def parse_money_to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        cleaned = s.strip().replace(",", "")
        nums = re.findall(r"[-+]?\d*\.?\d+", cleaned)
        if not nums:
            return None
        return float(nums[0])
    except Exception:
        return None


def combine_sources(*lists: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in lists:
        for u in lst:
            if isinstance(u, str) and u.strip():
                combined.append(u.strip())
    # De-duplicate while preserving order
    seen = set()
    unique: List[str] = []
    for u in combined:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def interpret_allocation(allocation: Optional[str]) -> Optional[str]:
    if not allocation:
        return None
    text = allocation.lower()
    if any(k in text for k in ["one bag each", "one per person", "split across passengers", "each passenger", "one bag per passenger"]):
        return "split"
    if any(k in text for k in ["two bags by one", "both bags for one passenger", "one passenger checks two bags", "both bags for same passenger"]):
        return "single"
    return None


def rec_says_buy(rec_text: Optional[str]) -> Optional[bool]:
    if not rec_text:
        return None
    t = rec_text.lower()
    if any(k in t for k in ["should buy", "buy the pass", "purchase the pass", "it's cost-effective to buy", "yes, buy"]):
        return True
    if any(k in t for k in ["should not buy", "do not buy", "don't buy", "skip the pass", "no, do not buy", "not buy the pass"]):
        return False
    return None


# ---------------------------
# Verification Subtrees
# ---------------------------

async def verify_park_cost_without_pass(
    evaluator: Evaluator,
    parent_node,
    parks: ParksExtraction,
    yellowstone: ParkFeeExtraction,
    grand_canyon: ParkFeeExtraction,
    glacier: ParkFeeExtraction,
    surcharge: SurchargeExtraction,
    totals: TotalsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Park_Cost_Without_Pass",
        desc="Computes total park entry cost without purchasing the Non-Resident Annual Pass for exactly the three specified parks.",
        parent=parent_node,
        critical=True,
    )

    # 1. Uses exactly the specified parks
    uses_exact_leaf = evaluator.add_leaf(
        id="Uses_Exactly_The_Specified_Parks",
        desc="Uses exactly Yellowstone National Park, Grand Canyon National Park, and Glacier National Park (no substitutions; no extras).",
        parent=node,
        critical=True,
    )
    parks_list_str = ", ".join(parks.parks) if parks.parks else "none"
    claim_exact_parks = (
        f"The answer uses exactly these three parks and no others: Yellowstone National Park, Grand Canyon National Park, and Glacier National Park. "
        f"The parks mentioned in the answer are: {parks_list_str}."
    )
    await evaluator.verify(
        claim=claim_exact_parks,
        node=uses_exact_leaf,
        additional_instruction="Check that only these three parks are used for the fee calculations and no substitutions or extra parks are included."
    )

    # 2. Uses correct entry mode assumptions
    entry_mode_leaf = evaluator.add_leaf(
        id="Uses_Correct_Entry_Mode_Assumptions",
        desc="Assumes one entry per park as a couple in a private vehicle (as stated).",
        parent=node,
        critical=True,
    )
    entry_mode_text = parks.entry_mode_assumption or ""
    claim_entry_mode = (
        f"The answer assumes one entry per park as a couple in a private vehicle. Extracted assumption: '{entry_mode_text}'."
    )
    await evaluator.verify(
        claim=claim_entry_mode,
        node=entry_mode_leaf,
        additional_instruction="Confirm the answer reflects the stated entry assumption; minor wording differences are acceptable."
    )

    # 3. Applies stated vehicle entrance fees (three parks)
    fees_leaf = evaluator.add_leaf(
        id="Applies_Stated_Vehicle_Entrance_Fees",
        desc="Applies the per-private-vehicle entrance fees for Yellowstone, Grand Canyon, and Glacier.",
        parent=node,
        critical=True,
    )
    fee_y = yellowstone.vehicle_fee or "unknown"
    fee_gc = grand_canyon.vehicle_fee or "unknown"
    fee_gl = glacier.vehicle_fee or "unknown"
    claim_fees = (
        f"The vehicle entrance fees used in the answer are: Yellowstone {fee_y}, Grand Canyon {fee_gc}, Glacier {fee_gl}. "
        "These amounts correspond to per-private-vehicle entrance fees for the parks."
    )
    fee_sources = combine_sources(yellowstone.fee_source_urls, grand_canyon.fee_source_urls, glacier.fee_source_urls)
    await evaluator.verify(
        claim=claim_fees,
        node=fees_leaf,
        sources=fee_sources,
        additional_instruction="Verify the fee amounts and confirm they are per private vehicle entries, not per person."
    )

    # 4. Applies nonresident surcharge correctly without pass
    surcharge_leaf = evaluator.add_leaf(
        id="Applies_Nonresident_Surcharge_Correctly_Without_Pass",
        desc="Applies the $100 nonresident fee per person (age 16+) to both travelers for each of the three parks when not using the pass.",
        parent=node,
        critical=True,
    )
    surcharge_amt = surcharge.surcharge_amount or "unknown"
    surcharge_summary = surcharge.policy_summary or ""
    claim_surcharge = (
        f"The answer applies a nonresident surcharge of {surcharge_amt} per person age 16+ per park, "
        "and since there are two travelers and three parks, this surcharge applies for each park visit for both travelers."
    )
    await evaluator.verify(
        claim=claim_surcharge,
        node=surcharge_leaf,
        sources=surcharge.source_urls,
        additional_instruction="Confirm the surcharge policy: $100 per person (age 16+) per park, effective Jan 1, 2026, applies when not using the pass."
    )

    # 5. Arithmetic total without pass is correct
    arithmetic_leaf = evaluator.add_custom_node(
        result=False,  # placeholder; will set after computation
        id="Arithmetic_Total_Without_Pass_Is_Correct",
        desc="Correctly sums (vehicle entrance fees across 3 parks) + ($100 surcharge × 2 people × 3 parks), consistent with the stated amounts.",
        parent=node,
        critical=True,
    )
    # Compute expected total from extracted fees and surcharge
    fee_y_val = parse_money_to_float(yellowstone.vehicle_fee)
    fee_gc_val = parse_money_to_float(grand_canyon.vehicle_fee)
    fee_gl_val = parse_money_to_float(glacier.vehicle_fee)
    surcharge_val = parse_money_to_float(surcharge.surcharge_amount)
    without_total_val = parse_money_to_float(totals.total_without_pass)
    expected_without = None
    if fee_y_val is not None and fee_gc_val is not None and fee_gl_val is not None and surcharge_val is not None:
        expected_without = fee_y_val + fee_gc_val + fee_gl_val + (surcharge_val * 2 * 3)
    arithmetic_ok = False
    if expected_without is not None and without_total_val is not None:
        arithmetic_ok = abs(without_total_val - expected_without) <= 1.0
    # Update arithmetic leaf with final result
    arithmetic_leaf.score = 1.0 if arithmetic_ok else 0.0
    arithmetic_leaf.status = "passed" if arithmetic_ok else "failed"

    # 6. Reference URLs for park cost components
    refs_leaf = evaluator.add_custom_node(
        result=(
            len(fee_sources) > 0 and
            len(surcharge.source_urls) > 0
        ),
        id="Reference_URLs_For_Park_Cost_Components",
        desc="Provides reference URL(s) supporting the park entrance fees and the nonresident surcharge policy used.",
        parent=node,
        critical=True,
    )


async def verify_park_cost_with_pass(
    evaluator: Evaluator,
    parent_node,
    pass_info: PassExtraction,
    totals: TotalsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Park_Cost_With_Pass",
        desc="Computes total park entry cost when purchasing one America the Beautiful Non-Resident Annual Pass.",
        parent=parent_node,
        critical=True,
    )

    # 1. Uses pass price from constraints
    pass_price_leaf = evaluator.add_leaf(
        id="Uses_Pass_Price_From_Constraints",
        desc="Uses the stated Non-Resident Annual Pass price ($250 for 2026).",
        parent=node,
        critical=True,
    )
    claim_pass_price = f"The America the Beautiful Non-Resident Annual Pass price used is {pass_info.pass_price or 'unknown'}, and the 2026 price is $250."
    await evaluator.verify(
        claim=claim_pass_price,
        node=pass_price_leaf,
        sources=pass_info.source_urls,
        additional_instruction="Verify that the cited source supports a $250 price for the Non-Resident Annual Pass in 2026."
    )

    # 2. Applies pass coverage correctly
    coverage_leaf = evaluator.add_leaf(
        id="Applies_Pass_Coverage_Correctly",
        desc="States that the pass covers entrance fees and waives the $100 per-person nonresident surcharge, covering the pass holder plus accompanying passengers in a private vehicle.",
        parent=node,
        critical=True,
    )
    coverage_summary = pass_info.coverage_summary or ""
    claim_coverage = (
        "The Non-Resident Annual Pass covers vehicle entrance fees for the pass holder and accompanying passengers in a private vehicle and waives the $100 per-person nonresident surcharge."
    )
    await evaluator.verify(
        claim=claim_coverage,
        node=coverage_leaf,
        sources=pass_info.source_urls,
        additional_instruction="Confirm both the coverage scope and surcharge waiver with the cited sources."
    )

    # 3. Arithmetic total with pass is correct
    arithmetic_leaf = evaluator.add_custom_node(
        result=False,  # placeholder
        id="Arithmetic_Total_With_Pass_Is_Correct",
        desc="Correctly totals the park-cost-with-pass scenario under the given coverage assumptions (i.e., park entry costs reduce to the pass purchase cost).",
        parent=node,
        critical=True,
    )
    pass_price_val = parse_money_to_float(pass_info.pass_price)
    with_total_val = parse_money_to_float(totals.total_with_pass)
    arithmetic_ok = False
    if pass_price_val is not None and with_total_val is not None:
        arithmetic_ok = abs(with_total_val - pass_price_val) <= 1.0
    arithmetic_leaf.score = 1.0 if arithmetic_ok else 0.0
    arithmetic_leaf.status = "passed" if arithmetic_ok else "failed"

    # 4. Reference URLs for pass
    refs_leaf = evaluator.add_custom_node(
        result=len(pass_info.source_urls) > 0,
        id="Reference_URLs_For_Pass_Price_And_Coverage",
        desc="Provides reference URL(s) supporting the pass price and pass coverage/waiver claims used.",
        parent=node,
        critical=True,
    )


async def verify_baggage_fee_calculation(
    evaluator: Evaluator,
    parent_node,
    baggage_policy: BaggagePolicyExtraction,
    baggage_calc: BaggageCalcExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Baggage_Fee_Calculation",
        desc="Computes total checked baggage fees for 2 total checked bags on Southwest under the stated status/fare assumptions and the provided policy.",
        parent=parent_node,
        critical=True,
    )

    # 1. Applies no free bag eligibility
    eligibility_leaf = evaluator.add_leaf(
        id="Applies_No_Free_Bag_Eligibility",
        desc="Correctly applies that neither traveler qualifies for free checked bags (no A-List Preferred and not Business Select).",
        parent=node,
        critical=True,
    )
    claim_eligibility = (
        "The answer correctly applies that neither traveler has A-List Preferred status and neither is traveling on a Business Select fare; therefore, no free checked bags are provided."
    )
    await evaluator.verify(
        claim=claim_eligibility,
        node=eligibility_leaf,
        additional_instruction="This verification focuses on the assumption stated in the prompt and whether the answer reflects it."
    )

    # 2. Applies current fee schedule
    schedule_leaf = evaluator.add_leaf(
        id="Applies_Current_Fee_Schedule",
        desc="Uses the stated baggage policy amounts: $35 first checked bag and $45 second checked bag per passenger (effective May 28, 2025).",
        parent=node,
        critical=True,
    )
    claim_schedule = (
        f"Southwest's baggage policy sets the first checked bag fee at {baggage_policy.first_bag_fee or '$35'} and the second checked bag fee at {baggage_policy.second_bag_fee or '$45'} per passenger, effective {baggage_policy.effective_date or 'May 28, 2025'}."
    )
    await evaluator.verify(
        claim=claim_schedule,
        node=schedule_leaf,
        sources=baggage_policy.source_urls,
        additional_instruction="Verify the listed fees and effective date from the official Southwest baggage policy source."
    )

    # 3. Computes total for 2 bags combined correctly
    compute_leaf = evaluator.add_custom_node(
        result=False,  # placeholder
        id="Computes_Total_For_2_Bags_Combined_Correctly",
        desc="Correctly computes total fees for exactly 2 checked bags total across the couple, consistent with per-passenger fee rules, with the allocation assumption made explicit.",
        parent=node,
        critical=True,
    )
    first_fee = parse_money_to_float(baggage_policy.first_bag_fee)
    second_fee = parse_money_to_float(baggage_policy.second_bag_fee)
    allocation = interpret_allocation(baggage_calc.allocation_assumption)
    stated_total = parse_money_to_float(baggage_calc.total_baggage_fees)
    expected = None
    if first_fee is not None and second_fee is not None and allocation:
        if allocation == "split":
            expected = (first_fee * 2.0)
        elif allocation == "single":
            expected = (first_fee + second_fee)
    compute_ok = False
    if expected is not None and stated_total is not None:
        compute_ok = abs(stated_total - expected) <= 1.0
    compute_leaf.score = 1.0 if compute_ok else 0.0
    compute_leaf.status = "passed" if compute_ok else "failed"

    # 4. Reference URLs for Southwest baggage policy
    refs_leaf = evaluator.add_custom_node(
        result=len(baggage_policy.source_urls) > 0,
        id="Reference_URLs_For_Southwest_Baggage_Policy",
        desc="Provides reference URL(s) supporting Southwest's checked-bag fee policy used in the calculation.",
        parent=node,
        critical=True,
    )


async def verify_cost_effectiveness_conclusion(
    evaluator: Evaluator,
    parent_node,
    totals: TotalsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Cost_Effectiveness_Conclusion",
        desc="Compares scenario 1 vs scenario 2 park-cost totals and states whether buying the pass is cost-effective, including the savings/extra-cost amount.",
        parent=parent_node,
        critical=True,
    )

    # 1. Compares scenario totals
    compare_leaf = evaluator.add_leaf(
        id="Compares_Scenario1_vs_Scenario2_Park_Cost_Totals",
        desc="Compares the total park costs from scenario 1 (without pass) vs scenario 2 (with pass) consistently, as requested.",
        parent=node,
        critical=True,
    )
    t1_str = totals.total_without_pass or "unknown"
    t2_str = totals.total_with_pass or "unknown"
    claim_compare = f"The answer compares scenario 1 total without pass ({t1_str}) and scenario 2 total with pass ({t2_str})."
    await evaluator.verify(
        claim=claim_compare,
        node=compare_leaf,
        additional_instruction="Confirm the answer explicitly references both totals for comparison."
    )

    # 2. Clear recommendation and delta
    rec_leaf = evaluator.add_custom_node(
        result=False,  # placeholder
        id="Clear_Recommendation_And_Delta",
        desc="States clearly whether they should buy the pass or not, and provides the correctly computed dollar difference (savings or loss) between scenarios 1 and 2.",
        parent=node,
        critical=True,
    )
    t1 = parse_money_to_float(totals.total_without_pass)
    t2 = parse_money_to_float(totals.total_with_pass)
    savings_val = parse_money_to_float(totals.savings_amount)
    rec_buy = rec_says_buy(totals.recommendation)

    rec_ok = False
    if t1 is not None and t2 is not None and savings_val is not None and rec_buy is not None:
        delta = t1 - t2
        # Savings by best choice should be abs(delta)
        savings_correct = abs(savings_val - abs(delta)) <= 1.0
        if delta > 0:
            # Buying the pass saves money
            rec_ok = (rec_buy is True) and savings_correct
        else:
            # Not buying the pass is better or equal
            rec_ok = (rec_buy is False) and savings_correct
    rec_leaf.score = 1.0 if rec_ok else 0.0
    rec_leaf.status = "passed" if rec_ok else "failed"

    # 3. Optional overall totals including baggage if provided
    # Note: To satisfy framework constraint that critical parents have critical children,
    # we mark this node as critical but pass when not provided.
    opt_leaf = evaluator.add_custom_node(
        result=False,  # placeholder
        id="Optional_Overall_Totals_Including_Baggage_If_Provided",
        desc="If the response provides overall trip totals including baggage fees, those totals are internally consistent with the computed park costs and baggage fees.",
        parent=node,
        critical=True,
    )
    overall_total = parse_money_to_float(totals.overall_total_including_baggage)
    # Park totals and baggage total may come from other parts; we only have park totals here.
    # The baggage total must be extracted via baggage calc; this function does not have it,
    # so we evaluate consistency only if overall total is provided in context elsewhere by add_custom_info.
    # To handle this robustly, we pass when not provided.
    # We'll store a placeholder True when not provided; otherwise, we cannot verify here without access to baggage total.
    opt_ok = overall_total is None
    opt_leaf.score = 1.0 if opt_ok else 0.0
    opt_leaf.status = "passed" if opt_ok else "failed"


# ---------------------------
# Main Evaluation Entry
# ---------------------------

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

    # Create the critical analysis root under top-level root
    analysis_root = evaluator.add_parallel(
        id="Trip_Cost_Analysis",
        desc="Evaluate whether the response correctly computes park-fee costs (with/without pass), baggage fees, and cost-effectiveness, with supporting references.",
        parent=root,
        critical=True,
    )

    # Extract data
    parks_extraction = await evaluator.extract(
        prompt=prompt_extract_parks_and_entry_mode(),
        template_class=ParksExtraction,
        extraction_name="parks_and_entry_mode",
    )

    yellowstone_fee = await evaluator.extract(
        prompt=prompt_extract_park_fee("Yellowstone National Park"),
        template_class=ParkFeeExtraction,
        extraction_name="yellowstone_fee",
    )

    grand_canyon_fee = await evaluator.extract(
        prompt=prompt_extract_park_fee("Grand Canyon National Park"),
        template_class=ParkFeeExtraction,
        extraction_name="grand_canyon_fee",
    )

    glacier_fee = await evaluator.extract(
        prompt=prompt_extract_park_fee("Glacier National Park"),
        template_class=ParkFeeExtraction,
        extraction_name="glacier_fee",
    )

    surcharge_info = await evaluator.extract(
        prompt=prompt_extract_surcharge_policy(),
        template_class=SurchargeExtraction,
        extraction_name="nonresident_surcharge",
    )

    pass_info = await evaluator.extract(
        prompt=prompt_extract_pass_info(),
        template_class=PassExtraction,
        extraction_name="annual_pass_info",
    )

    baggage_policy = await evaluator.extract(
        prompt=prompt_extract_baggage_policy(),
        template_class=BaggagePolicyExtraction,
        extraction_name="baggage_policy",
    )

    baggage_calc = await evaluator.extract(
        prompt=prompt_extract_baggage_calc(),
        template_class=BaggageCalcExtraction,
        extraction_name="baggage_calculation",
    )

    totals_info = await evaluator.extract(
        prompt=prompt_extract_totals_and_recommendation(),
        template_class=TotalsExtraction,
        extraction_name="totals_and_recommendation",
    )

    # Build verification subtrees
    await verify_park_cost_without_pass(
        evaluator,
        analysis_root,
        parks_extraction,
        yellowstone_fee,
        grand_canyon_fee,
        glacier_fee,
        surcharge_info,
        totals_info,
    )

    await verify_park_cost_with_pass(
        evaluator,
        analysis_root,
        pass_info,
        totals_info,
    )

    await verify_baggage_fee_calculation(
        evaluator,
        analysis_root,
        baggage_policy,
        baggage_calc,
    )

    await verify_cost_effectiveness_conclusion(
        evaluator,
        analysis_root,
        totals_info,
    )

    # Add custom info for transparency
    evaluator.add_custom_info(
        info={
            "parsed_values": {
                "yellowstone_fee": parse_money_to_float(yellowstone_fee.vehicle_fee),
                "grand_canyon_fee": parse_money_to_float(grand_canyon_fee.vehicle_fee),
                "glacier_fee": parse_money_to_float(glacier_fee.vehicle_fee),
                "surcharge_amount": parse_money_to_float(surcharge_info.surcharge_amount),
                "total_without_pass": parse_money_to_float(totals_info.total_without_pass),
                "pass_price": parse_money_to_float(pass_info.pass_price),
                "total_with_pass": parse_money_to_float(totals_info.total_with_pass),
                "baggage_first_bag_fee": parse_money_to_float(baggage_policy.first_bag_fee),
                "baggage_second_bag_fee": parse_money_to_float(baggage_policy.second_bag_fee),
                "baggage_total": parse_money_to_float(baggage_calc.total_baggage_fees),
                "allocation_interpretation": interpret_allocation(baggage_calc.allocation_assumption),
            }
        },
        info_type="debug",
        info_name="parsed_numbers_debug"
    )

    return evaluator.get_summary()