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
TASK_ID = "nonresident_park_fees_2026_three_parks"
TASK_DESCRIPTION = (
    "A Canadian family (2 adults, 2 children ages 10 and 14) will visit Grand Canyon, Yellowstone, "
    "and Yosemite in July 2026 in one private vehicle. Compute and compare: "
    "1) total cost paying individual entrance fees (with any 2026 nonresident surcharges), "
    "2) total cost purchasing America the Beautiful annual pass(es); then recommend the most cost-effective option "
    "with justification and valid source URLs. Include how the new 2026 nonresident fee structure affects costs."
)

# Expected policy details per rubric (used for reasoning/claims; not as ground truth enforcement)
EXPECTED_NONRESIDENT_SURCHARGE = 100.0  # per paying person
EXPECTED_NONRESIDENT_PASS_PRICE = 250.0  # nonresident ATB pass (Jan 1, 2026)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ParkFeeBreakdown(BaseModel):
    park_name: Optional[str] = None
    base_fee_value: Optional[float] = None  # numeric USD without $ sign
    fee_basis: Optional[str] = None  # e.g., "per vehicle (private vehicle)", "per person", etc.
    surcharge_per_person_value: Optional[float] = None  # expected 100.0 for 2026 nonresident surcharge
    paying_persons_count: Optional[int] = None  # should reflect children under 16 free
    subtotal_value: Optional[float] = None  # per-park subtotal the answer computed
    source_urls: List[str] = Field(default_factory=list)  # URLs supporting this park's base fee


class Scenario1Extraction(BaseModel):
    # Per-park details
    grand_canyon: Optional[ParkFeeBreakdown] = None
    yellowstone: Optional[ParkFeeBreakdown] = None
    yosemite: Optional[ParkFeeBreakdown] = None

    # Policy source(s) for 2026 nonresident surcharge
    surcharge_policy_urls: List[str] = Field(default_factory=list)

    # Scenario 1 total across all three parks
    total_value: Optional[float] = None


class Scenario2Extraction(BaseModel):
    pass_type: Optional[str] = None  # e.g., "America the Beautiful annual pass (nonresident)"
    pass_unit_price_value: Optional[float] = None  # expected 250.0
    pass_count: Optional[int] = None
    total_value: Optional[float] = None

    # URLs supporting price, coverage rules, surcharge avoidance
    pass_price_urls: List[str] = Field(default_factory=list)
    pass_rules_urls: List[str] = Field(default_factory=list)
    surcharge_avoidance_urls: List[str] = Field(default_factory=list)


class ComparisonExtraction(BaseModel):
    # Use canonical labels: "scenario_1" or "scenario_2"
    recommended_option: Optional[str] = None
    savings_value: Optional[float] = None  # explicit numeric savings stated in the answer

    # Presence flags for considerations
    pass_validity_explained: Optional[bool] = None  # 12 months from month of purchase
    coverage_rules_explained: Optional[bool] = None  # one non-commercial vehicle + passengers; children under 16 free
    additional_visits_considered: Optional[bool] = None  # effect of additional visits on cost-effectiveness


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_scenario1() -> str:
    return """
Extract the Scenario 1 (individual entrance fees) details used in the answer for each of the three parks: Grand Canyon, Yellowstone, Yosemite.
For each park, extract:
- park_name: the park name string
- base_fee_value: the numeric USD amount of the standard entrance fee used for this calculation (without $ sign; a float)
- fee_basis: how the base fee is charged (e.g., "per vehicle (private vehicle)", "per person")
- surcharge_per_person_value: the numeric USD amount of the 2026 nonresident surcharge per paying person used in the answer (expected 100.0)
- paying_persons_count: integer number of people in this party to whom the surcharge is applied in the answer (children under 16 should be free)
- subtotal_value: the numeric USD subtotal reported for this park in Scenario 1
- source_urls: a list of URLs that the answer cites for this park’s base entrance fee (extract all provided URLs)

Also extract:
- surcharge_policy_urls: a list of URLs that support the 2026 nonresident surcharge policy used in Scenario 1 (e.g., that it is $100 per paying person and that children under 16 are free)
- total_value: the numeric Scenario 1 total across all three parks

Return numbers as plain floats (no currency symbols). If any item is missing, set it to null. Ensure URL fields contain only valid URLs explicitly present in the answer.
    """.strip()


def prompt_extract_scenario2() -> str:
    return """
Extract the Scenario 2 (annual pass) details used in the answer.
Fields to extract:
- pass_type: the type/name of pass (e.g., "America the Beautiful annual pass (nonresident)")
- pass_unit_price_value: the numeric USD price per pass used (expected 250.0 for nonresidents in 2026; return as a float without $)
- pass_count: integer number of passes the answer proposes to purchase for this party
- total_value: the numeric USD Scenario 2 total in the answer

Also extract URL lists cited for each aspect:
- pass_price_urls: URLs supporting the $250 nonresident pass price starting Jan 1, 2026
- pass_rules_urls: URLs supporting coverage rules (e.g., covers one non-commercial vehicle and its passengers at per-vehicle fee areas; children under 16 free)
- surcharge_avoidance_urls: URLs or official references supporting that the nonresident pass avoids the $100 per-person nonresident surcharge for covered entries

Return numbers as floats (no currency symbols). If any item is missing, set it to null. Extract only URLs explicitly present in the answer.
    """.strip()


def prompt_extract_comparison() -> str:
    return """
From the comparison/recommendation section of the answer, extract:
- recommended_option: choose and return one of ["scenario_1", "scenario_2"] indicating which the answer claims is most cost-effective
- savings_value: the numeric USD savings explicitly stated by the answer (absolute difference); return as float (no currency symbol), or null if not stated
- pass_validity_explained: boolean, whether the answer explicitly explains that the annual pass is valid for 12 months from the month of purchase
- coverage_rules_explained: boolean, whether the answer explicitly explains key pass coverage rules relevant to this trip (coverage for one non-commercial vehicle and its passengers; children under 16 free)
- additional_visits_considered: boolean, whether the answer explicitly addresses how additional park visits within the pass validity period could affect cost-effectiveness

Try to normalize the recommendation to "scenario_1" if they recommend paying individual entrance fees, or "scenario_2" if they recommend buying the annual pass. Return booleans as true/false.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def approx_equal(a: Optional[float], b: Optional[float], tol: float = 2.0) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def all_nonempty_url_lists(lists: List[List[str]]) -> bool:
    return all(isinstance(lst, list) and len([u for u in lst if isinstance(u, str) and u.strip()]) > 0 for lst in lists)


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_scenario1_sources(evaluator: Evaluator, parent_node, s1: Scenario1Extraction) -> None:
    """
    Build 'scenario_1_sources' as a custom leaf that passes only if:
    - Each park's base fee amount is supported by its provided URLs
    - The 2026 nonresident surcharge policy (per paying person, children under 16 free) is supported by provided URLs
    This function uses multiple standalone verifications (node=None) and combines results.
    """
    # Prepare node (as leaf by custom result)
    # We'll compute combined result first, then add the node with that result.
    gc = s1.grand_canyon or ParkFeeBreakdown()
    ys = s1.yellowstone or ParkFeeBreakdown()
    yo = s1.yosemite or ParkFeeBreakdown()

    # Presence checks for URLs and base fees
    url_presence_ok = all_nonempty_url_lists([
        gc.source_urls or [],
        ys.source_urls or [],
        yo.source_urls or [],
        s1.surcharge_policy_urls or [],
    ])
    values_present = (
        (gc.base_fee_value is not None) and
        (ys.base_fee_value is not None) and
        (yo.base_fee_value is not None)
    )

    # If basics missing, we'll still add the node but it will fail
    result_ok = url_presence_ok and values_present

    verify_results: List[bool] = []

    # Only attempt verifications if we have minimal inputs
    if result_ok:
        # Park base fees supported by provided URLs
        # Grand Canyon
        gc_claim = f"For Grand Canyon National Park, the standard entrance fee for a private vehicle is ${gc.base_fee_value:.2f}."
        verify_results.append(await evaluator.verify(
            claim=gc_claim,
            node=None,
            sources=gc.source_urls,
            additional_instruction="Verify the per-vehicle entrance fee amount. Allow minor rounding differences."
        ))

        # Yellowstone
        ys_claim = f"For Yellowstone National Park, the standard entrance fee for a private vehicle is ${ys.base_fee_value:.2f}."
        verify_results.append(await evaluator.verify(
            claim=ys_claim,
            node=None,
            sources=ys.source_urls,
            additional_instruction="Verify the per-vehicle entrance fee amount. Allow minor rounding differences."
        ))

        # Yosemite
        yo_claim = f"For Yosemite National Park, the standard entrance fee for a private vehicle is ${yo.base_fee_value:.2f}."
        verify_results.append(await evaluator.verify(
            claim=yo_claim,
            node=None,
            sources=yo.source_urls,
            additional_instruction="Verify the per-vehicle entrance fee amount. Allow minor rounding differences."
        ))

        # 2026 nonresident surcharge policy support
        surcharge_claim = (
            "Starting in 2026, there is a $100 nonresident surcharge per paying person at U.S. national parks, "
            "and children under 16 are exempt (do not pay entrance fees)."
        )
        verify_results.append(await evaluator.verify(
            claim=surcharge_claim,
            node=None,
            sources=s1.surcharge_policy_urls,
            additional_instruction="Check that the policy states a $100 surcharge per paying person for nonresidents, "
                                   "and that children under 16 are free."
        ))

        result_ok = result_ok and all(verify_results)

    # Now add the leaf node with the combined result
    evaluator.add_custom_node(
        result=result_ok,
        id="scenario_1_sources",
        desc="Provide valid reference URLs that support the entrance-fee inputs for Grand Canyon, Yellowstone, and Yosemite and the 2026 nonresident surcharge policy as applied in Scenario 1.",
        parent=parent_node,
        critical=True
    )


async def verify_park_subtotal(
    evaluator: Evaluator,
    parent_node,
    leaf_id: str,
    leaf_desc: str,
    pb: Optional[ParkFeeBreakdown],
) -> None:
    """
    Add a leaf and verify (via simple verification) that:
    subtotal_value == base_fee_value + paying_persons_count * surcharge_per_person_value
    and that the fee basis is correctly per-vehicle (for this party traveling together) and surcharge applies to paying persons only.
    """
    node = evaluator.add_leaf(
        id=leaf_id,
        desc=leaf_desc,
        parent=parent_node,
        critical=True
    )

    base = pb.base_fee_value if pb and pb.base_fee_value is not None else None
    surcharge = pb.surcharge_per_person_value if pb and pb.surcharge_per_person_value is not None else None
    payers = pb.paying_persons_count if pb and pb.paying_persons_count is not None else None
    subtotal = pb.subtotal_value if pb and pb.subtotal_value is not None else None
    basis = pb.fee_basis if pb and pb.fee_basis else ""

    # Construct a clear arithmetic claim using answer numbers
    claim = (
        f"In Scenario 1, for this park, the base entrance fee is ${base} (basis: {basis}), "
        f"the nonresident surcharge is ${surcharge} per paying person, applied to {payers} paying persons "
        f"(children under 16 are free). Therefore the park subtotal should equal base + payers*surcharge. "
        f"The answer's subtotal ${subtotal} equals that computed sum (allow rounding to nearest dollar)."
    )

    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Perform the arithmetic check using the provided numbers from the answer. "
                               "Accept small rounding differences (within a couple of dollars). "
                               "Also check logic: fee basis is per-vehicle (one vehicle), and surcharge applies only to paying persons (adults; children under 16 free)."
    )


async def verify_scenario1_total(evaluator: Evaluator, parent_node, s1: Scenario1Extraction) -> None:
    node = evaluator.add_leaf(
        id="scenario_1_total",
        desc="Scenario 1 total equals the sum of the three per-park subtotals and is presented clearly as a single total.",
        parent=parent_node,
        critical=True
    )

    gc = s1.grand_canyon or ParkFeeBreakdown()
    ys = s1.yellowstone or ParkFeeBreakdown()
    yo = s1.yosemite or ParkFeeBreakdown()
    total = s1.total_value

    claim = (
        f"For Scenario 1, the answer's per-park subtotals are: Grand Canyon ${gc.subtotal_value}, "
        f"Yellowstone ${ys.subtotal_value}, Yosemite ${yo.subtotal_value}. "
        f"The Scenario 1 total ${total} equals the sum of these three subtotals "
        f"(allow small rounding differences)."
    )

    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Compute sum of the three per-park subtotals and check that it matches the stated Scenario 1 total. "
                               "Allow small rounding differences within a couple of dollars."
    )


async def verify_scenario2_sources(evaluator: Evaluator, parent_node, s2: Scenario2Extraction) -> None:
    """
    Build 'scenario_2_sources' as a custom leaf that passes only if:
    - $250 nonresident pass price (from Jan 1, 2026) is supported by pass_price_urls
    - coverage rules (one non-commercial vehicle and passengers; children under 16 free) are supported by pass_rules_urls
    - pass avoids the $100 per-person nonresident surcharge (supported by surcharge_avoidance_urls)
    """
    price_urls_ok = len(s2.pass_price_urls) > 0
    rules_urls_ok = len(s2.pass_rules_urls) > 0
    avoid_urls_ok = len(s2.surcharge_avoidance_urls) > 0

    all_urls_present = price_urls_ok and rules_urls_ok and avoid_urls_ok
    result_ok = all_urls_present

    verify_results = []

    if result_ok:
        # (a) Price $250 for nonresident pass (2026)
        price_claim = (
            "Starting January 1, 2026, the America the Beautiful annual pass price for nonresidents is $250."
        )
        verify_results.append(await evaluator.verify(
            claim=price_claim,
            node=None,
            sources=s2.pass_price_urls,
            additional_instruction="Verify that $250 is the applicable nonresident annual pass price from 2026 onward."
        ))

        # (b) Coverage rules for one non-commercial vehicle (and passengers)
        rules_claim = (
            "The America the Beautiful annual pass covers entrance fees for one non-commercial vehicle and all its passengers "
            "at per-vehicle fee areas; at per-person fee areas, it covers the pass holder(s), and children under 16 are free."
        )
        verify_results.append(await evaluator.verify(
            claim=rules_claim,
            node=None,
            sources=s2.pass_rules_urls,
            additional_instruction="Verify the official coverage rules for the America the Beautiful annual pass."
        ))

        # (c) Pass avoids the $100 per-person nonresident surcharge for covered entries
        avoid_claim = (
            "Holders of the nonresident $250 America the Beautiful annual pass do not have to pay the $100 per-person nonresident surcharge for covered entries."
        )
        verify_results.append(await evaluator.verify(
            claim=avoid_claim,
            node=None,
            sources=s2.surcharge_avoidance_urls,
            additional_instruction="Verify that the nonresident annual pass obviates the separate nonresident surcharge when using the pass for covered entries."
        ))

        result_ok = result_ok and all(verify_results)

    evaluator.add_custom_node(
        result=result_ok,
        id="scenario_2_sources",
        desc="Provide valid reference URLs supporting: (a) the $250 nonresident America the Beautiful annual pass price starting Jan 1, 2026, (b) pass coverage rules for one non-commercial vehicle, and (c) that the $250 nonresident pass avoids the $100 per-person nonresident surcharge.",
        parent=parent_node,
        critical=True
    )


async def verify_scenario2_pass_count(evaluator: Evaluator, parent_node, s2: Scenario2Extraction) -> None:
    """
    Add a custom node to check pass count correctness for one private vehicle traveling together.
    Expected: one pass is sufficient for this family traveling together in one private vehicle.
    """
    correct_count = (s2.pass_count == 1)
    evaluator.add_custom_node(
        result=bool(correct_count),
        id="scenario_2_pass_count",
        desc="Correctly determine the number of America the Beautiful annual passes needed for this party given one private vehicle and pass coverage rules.",
        parent=parent_node,
        critical=True
    )


async def verify_scenario2_total(evaluator: Evaluator, parent_node, s2: Scenario2Extraction) -> None:
    node = evaluator.add_leaf(
        id="scenario_2_total",
        desc="Scenario 2 total correctly accounts for pass purchase cost(s) and does not add entrance fees or nonresident surcharges that are covered/avoided under the pass rules.",
        parent=parent_node,
        critical=True
    )

    price = s2.pass_unit_price_value
    count = s2.pass_count
    total = s2.total_value

    claim = (
        f"In Scenario 2, the answer uses a pass price of ${price} and {count} pass(es). "
        f"The Scenario 2 total ${total} equals pass_price × pass_count, and entrance fees or nonresident surcharges "
        f"covered/avoided by the pass are not added again."
    )

    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Check that the numeric total equals price × count (allow small rounding differences), "
                               "and also that no additional entrance fees or nonresident surcharges are added on top of the pass coverage."
    )


async def verify_comparison_and_recommendation(
    evaluator: Evaluator,
    parent_node,
    s1: Scenario1Extraction,
    s2: Scenario2Extraction,
    comp: ComparisonExtraction
) -> None:
    # comparison_correct (custom logic check with extracted totals)
    comp_node_1 = evaluator.add_custom_node(
        result=(
            (s1.total_value is not None and s2.total_value is not None and comp.recommended_option in ("scenario_1", "scenario_2"))
            and (
                (s1.total_value <= s2.total_value and comp.recommended_option == "scenario_1") or
                (s2.total_value < s1.total_value and comp.recommended_option == "scenario_2")
            )
        ),
        id="comparison_correct",
        desc="Correctly compare the computed Scenario 1 and Scenario 2 totals and identify which option is cheaper for this family and itinerary.",
        parent=parent_node,
        critical=True
    )

    # recommendation_justification_numeric (custom numeric savings check)
    expected_diff = None
    if s1.total_value is not None and s2.total_value is not None:
        try:
            expected_diff = abs(float(s1.total_value) - float(s2.total_value))
        except Exception:
            expected_diff = None

    comp_node_2 = evaluator.add_custom_node(
        result=(
            expected_diff is not None and comp.savings_value is not None and approx_equal(expected_diff, comp.savings_value, tol=5.0)
        ),
        id="recommendation_justification_numeric",
        desc="Justify the recommendation using the numeric totals (e.g., explicit savings/difference).",
        parent=parent_node,
        critical=True
    )

    # consideration_pass_validity (ensure the answer explicitly explains the 12-month validity from month of purchase)
    validity_node = evaluator.add_leaf(
        id="consideration_pass_validity",
        desc="Explain the annual pass validity period rule (valid for 12 months from the month of purchase) as a consideration for timing/value.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly explains that the America the Beautiful annual pass is valid for 12 months from the month of purchase.",
        node=validity_node,
        additional_instruction="Look for explicit mention of 'valid for 12 months from (the) month of purchase' or equivalent wording."
    )

    # consideration_coverage_rules (ensure the answer explains coverage rules)
    coverage_node = evaluator.add_leaf(
        id="consideration_coverage_rules",
        desc="Explain key pass coverage rules relevant to this trip (coverage for one non-commercial vehicle/passengers; children under 16 free) as a consideration for interpreting the totals.",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly explains that the pass covers one non-commercial vehicle and its passengers at per-vehicle fee areas, and that children under 16 are free.",
        node=coverage_node,
        additional_instruction="Look for explicit explanation of both aspects: vehicle coverage and under-16 free."
    )

    # consideration_additional_visits_value (ensure the answer addresses value if more parks visited)
    extra_visits_node = evaluator.add_leaf(
        id="consideration_additional_visits_value",
        desc="Address how additional park visits within the pass validity period could affect cost-effectiveness (even if not planned).",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer discusses how additional park visits within the pass validity period would increase the value of the annual pass and potentially improve cost-effectiveness.",
        node=extra_visits_node,
        additional_instruction="Look for explicit mention that additional visits during the pass validity could make the pass more worthwhile."
    )


# --------------------------------------------------------------------------- #
# Scenario builders                                                           #
# --------------------------------------------------------------------------- #
async def build_scenario_1(evaluator: Evaluator, parent_node, s1: Scenario1Extraction) -> None:
    """
    Scenario 1: sequential, critical. Children:
      - scenario_1_sources (custom)
      - scenario_1_per_park_subtotals (parallel, critical) with three park subtotal leaves
      - scenario_1_total (leaf)
    """
    # 1) Sources verification (combined)
    await verify_scenario1_sources(evaluator, parent_node, s1)

    # 2) Per-park subtotals (parallel)
    per_park_node = evaluator.add_parallel(
        id="scenario_1_per_park_subtotals",
        desc="Compute correct per-park subtotals for all three parks, reflecting the correct fee basis (e.g., per-vehicle) and applying the nonresident surcharge to the correct paying persons (children under 16 free).",
        parent=parent_node,
        critical=True
    )

    # Grand Canyon
    await verify_park_subtotal(
        evaluator=evaluator,
        parent_node=per_park_node,
        leaf_id="scenario_1_grand_canyon_subtotal",
        leaf_desc="Grand Canyon subtotal correctly equals (sourced standard entrance fee) + (applicable $100-per-paying-person nonresident surcharge for this party).",
        pb=s1.grand_canyon
    )

    # Yellowstone
    await verify_park_subtotal(
        evaluator=evaluator,
        parent_node=per_park_node,
        leaf_id="scenario_1_yellowstone_subtotal",
        leaf_desc="Yellowstone subtotal correctly equals ($35 per private vehicle standard entrance fee, per constraints) + (applicable $100-per-paying-person nonresident surcharge for this party).",
        pb=s1.yellowstone
    )

    # Yosemite
    await verify_park_subtotal(
        evaluator=evaluator,
        parent_node=per_park_node,
        leaf_id="scenario_1_yosemite_subtotal",
        leaf_desc="Yosemite subtotal correctly equals (sourced standard entrance fee) + (applicable $100-per-paying-person nonresident surcharge for this party).",
        pb=s1.yosemite
    )

    # 3) Scenario 1 total
    await verify_scenario1_total(evaluator, parent_node, s1)


async def build_scenario_2(evaluator: Evaluator, parent_node, s2: Scenario2Extraction) -> None:
    """
    Scenario 2: sequential, critical. Children:
      - scenario_2_sources (custom combined)
      - scenario_2_pass_count (custom check)
      - scenario_2_total (leaf check)
    """
    await verify_scenario2_sources(evaluator, parent_node, s2)
    await verify_scenario2_pass_count(evaluator, parent_node, s2)
    await verify_scenario2_total(evaluator, parent_node, s2)


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
    Entry point for evaluating an answer to the 2026 nonresident national parks cost-comparison task.
    """
    # Initialize evaluator with a root node
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root: sequential pipeline
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

    # Add a task-level critical sequential node (acts as the true critical root for rubric)
    task_node = evaluator.add_sequential(
        id="task_main",
        desc="Compute and compare the total entrance-cost options for the specified nonresident family (2 adults, 2 children under 16) visiting Grand Canyon, Yellowstone, and Yosemite in July 2026: (1) paying individual entrance fees (plus required nonresident surcharges) vs (2) purchasing America the Beautiful annual pass(es), then recommend the most cost-effective option with justification and sources.",
        parent=root,
        critical=True
    )

    # Extract structured info (in parallel)
    scenario1_task = evaluator.extract(
        prompt=prompt_extract_scenario1(),
        template_class=Scenario1Extraction,
        extraction_name="scenario_1_extraction"
    )
    scenario2_task = evaluator.extract(
        prompt=prompt_extract_scenario2(),
        template_class=Scenario2Extraction,
        extraction_name="scenario_2_extraction"
    )
    comparison_task = evaluator.extract(
        prompt=prompt_extract_comparison(),
        template_class=ComparisonExtraction,
        extraction_name="comparison_extraction"
    )

    s1, s2, comp = await asyncio.gather(scenario1_task, scenario2_task, comparison_task)

    # Build 'compute_two_scenarios' (parallel, critical)
    compute_node = evaluator.add_parallel(
        id="compute_two_scenarios",
        desc="Correctly compute totals (with breakdowns) for Scenario 1 and Scenario 2 using the given 2026 nonresident rules and the trip context.",
        parent=task_node,
        critical=True
    )

    # Scenario 1 branch (sequential, critical)
    scenario1_node = evaluator.add_sequential(
        id="scenario_1_individual_fees",
        desc="Scenario 1: Total cost paying individual entrance fees at each park in July 2026, including any applicable nonresident surcharges, with a detailed breakdown and supporting URLs.",
        parent=compute_node,
        critical=True
    )
    await build_scenario_1(evaluator, scenario1_node, s1)

    # Scenario 2 branch (sequential, critical)
    scenario2_node = evaluator.add_sequential(
        id="scenario_2_annual_pass",
        desc="Scenario 2: Total cost purchasing America the Beautiful annual pass(es) for a nonresident family traveling together in one private vehicle in July 2026, with a detailed breakdown and supporting URLs.",
        parent=compute_node,
        critical=True
    )
    await build_scenario_2(evaluator, scenario2_node, s2)

    # Comparison and recommendation (parallel, critical)
    compare_node = evaluator.add_parallel(
        id="comparison_and_recommendation",
        desc="Compare Scenario 1 vs Scenario 2 totals and provide a clear most-cost-effective recommendation with justification and requested additional considerations.",
        parent=task_node,
        critical=True
    )
    await verify_comparison_and_recommendation(evaluator, compare_node, s1, s2, comp)

    # Optional: add some custom info for transparency
    evaluator.add_custom_info(
        info={
            "expected_policy_notes": {
                "nonresident_surcharge_per_paying_person": EXPECTED_NONRESIDENT_SURCHARGE,
                "nonresident_annual_pass_price": EXPECTED_NONRESIDENT_PASS_PRICE,
                "assumptions": [
                    "Children under 16 are free for entrance fees.",
                    "Party travels together in one private non-commercial vehicle."
                ]
            }
        },
        info_type="policy_context",
        info_name="policy_context_notes"
    )

    # Return final structured summary
    return evaluator.get_summary()