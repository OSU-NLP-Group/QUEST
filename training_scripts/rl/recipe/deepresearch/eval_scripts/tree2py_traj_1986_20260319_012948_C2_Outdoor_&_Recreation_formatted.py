import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "maine_pass_strategy_2026"
TASK_DESCRIPTION = (
    "A U.S. resident is planning a 7-day trip to Maine in July 2026 and wants to visit both Acadia National Park "
    "and Baxter State Park. During their visit to Acadia, they plan to drive the Cadillac Summit Road once. "
    "They need to determine the most cost-effective strategy for park access.\n\n"
    "Analyze two strategies and provide costs, coverage, URLs, and recommendation."
)

# EXPECTED VALUES PER RUBRIC
ABP_COST_EXPECTED = 80
ACADIA_FEE_EXPECTED = 35
CADILLAC_FEE_EXPECTED = 6
BAXTER_FEE_EXPECTED = 50
TOTAL_A_EXPECTED = ABP_COST_EXPECTED + CADILLAC_FEE_EXPECTED + BAXTER_FEE_EXPECTED  # 136
TOTAL_B_EXPECTED = ACADIA_FEE_EXPECTED + CADILLAC_FEE_EXPECTED + BAXTER_FEE_EXPECTED  # 91


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class TripCostExtraction(BaseModel):
    # Pass coverage/applicability
    acadia_pass_coverage_statement: Optional[str] = None
    acadia_pass_coverage_urls: List[str] = Field(default_factory=list)

    # Cadillac Summit Road reservation
    cadillac_reservation_required_statement: Optional[str] = None
    cadillac_reservation_fee: Optional[str] = None
    cadillac_not_covered_by_pass_statement: Optional[str] = None
    cadillac_urls: List[str] = Field(default_factory=list)

    # Baxter State Park acceptance & fee
    baxter_pass_acceptance_statement: Optional[str] = None
    baxter_acceptance_urls: List[str] = Field(default_factory=list)
    baxter_entrance_fee: Optional[str] = None
    baxter_fee_urls: List[str] = Field(default_factory=list)

    # Fees used in strategies
    abp_cost: Optional[str] = None
    abp_cost_urls: List[str] = Field(default_factory=list)

    acadia_individual_fee: Optional[str] = None
    acadia_fee_urls: List[str] = Field(default_factory=list)

    # Totals and recommendation
    strategy_a_total: Optional[str] = None
    strategy_b_total: Optional[str] = None
    recommendation: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_costs() -> str:
    return """
    Extract the fee amounts, coverage statements, URLs, totals, and recommendation exactly as presented in the answer.
    Return null for any missing information. Use arrays for URLs. Do not invent values.

    Fields to extract:
    - acadia_pass_coverage_statement: The textual statement (if any) that the America the Beautiful (Annual) Pass covers Acadia National Park entrance.
    - acadia_pass_coverage_urls: All URLs that the answer cites to support that pass coverage at Acadia.

    - cadillac_reservation_required_statement: The textual statement (if any) about needing a vehicle reservation to drive Cadillac Summit Road.
    - cadillac_reservation_fee: The amount stated for the Cadillac Summit Road vehicle reservation fee (as text, e.g., "$6", "6 USD").
    - cadillac_not_covered_by_pass_statement: The textual statement (if any) that the Cadillac reservation is NOT covered by entrance passes.
    - cadillac_urls: All URLs that support/describe the Cadillac reservation requirement and its fee.

    - baxter_pass_acceptance_statement: The textual statement (if any) about whether Baxter State Park accepts or does not accept the America the Beautiful Pass or Maine State Park Pass, and that a separate entrance fee is required.
    - baxter_acceptance_urls: All URLs that the answer cites for Baxter pass acceptance/non-acceptance.
    - baxter_entrance_fee: The amount stated for Baxter State Park entrance fee used in the calculations (as text).
    - baxter_fee_urls: All URLs that the answer cites for the Baxter entrance fee used.

    - abp_cost: The stated cost of the America the Beautiful Annual Pass used (as text, e.g., "$80").
    - abp_cost_urls: All URLs that support the America the Beautiful Annual Pass price used.

    - acadia_individual_fee: The stated individual Acadia NP entrance fee used for calculations (as text).
    - acadia_fee_urls: All URLs that support the Acadia individual entrance fee used.

    - strategy_a_total: The final total the answer presents for Strategy A (America the Beautiful pass + any required extra fees), as provided in the answer (as text).
    - strategy_b_total: The final total the answer presents for Strategy B (individual fees + any required extra fees), as provided in the answer (as text).
    - recommendation: The final recommendation text (e.g., "Strategy B is cheaper", "Choose individual fees", etc.).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def parse_amount_to_float(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    # Extract first numeric token (allow $ and commas)
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", value.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def urls_non_empty(*url_lists: List[str]) -> bool:
    for lst in url_lists:
        if not lst or len([u for u in lst if isinstance(u, str) and u.strip()]) == 0:
            return False
    return True


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_pass_applicability_checks(evaluator: Evaluator, parent) -> None:
    """
    Build the 'Pass_Applicability_Analysis' subtree (critical).
    """
    node = evaluator.add_parallel(
        id="Pass_Applicability_Analysis",
        desc="Evaluates understanding of which passes are valid at each park",
        parent=parent,
        critical=True
    )

    # Get last extraction result (we recorded only one, so it's safe)
    # More robust approach: pass the extracted object around; here we fetch from evaluator summary store is not available directly.
    # We'll rely on closure variable in calling context to pass extraction; Instead, we attach to evaluator.custom_info for retrieval.
    # Simpler: We'll ask caller to pass extraction. To keep signature consistent, we will fetch from evaluator._extraction_results[-1].
    extraction_dict = evaluator._extraction_results[-1]["result"]
    ext = TripCostExtraction(**extraction_dict)

    # 1) Acadia_Pass_Coverage
    acadia_sources_exist = evaluator.add_custom_node(
        result=urls_non_empty(ext.acadia_pass_coverage_urls),
        id="Acadia_Pass_Coverage_sources_exist",
        desc="Acadia pass coverage URLs provided",
        parent=node,
        critical=True
    )
    acadia_cov_leaf = evaluator.add_leaf(
        id="Acadia_Pass_Coverage",
        desc="Correctly identifies that America the Beautiful Pass covers Acadia National Park entrance fee",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The America the Beautiful Annual Pass is accepted for Acadia National Park and covers its entrance fee.",
        node=acadia_cov_leaf,
        sources=ext.acadia_pass_coverage_urls,
        additional_instruction="Confirm that Acadia NP entrance fees are covered by the America the Beautiful (Annual) Pass. "
                               "Allow phrasing variants. Focus on entrance fee coverage."
    )

    # 2) Cadillac_Reservation_Requirement: split into three focused checks
    cadillac_sources_exist = evaluator.add_custom_node(
        result=urls_non_empty(ext.cadillac_urls),
        id="Cadillac_sources_exist",
        desc="Cadillac Summit Road reservation URLs provided",
        parent=node,
        critical=True
    )
    cadillac_required_leaf = evaluator.add_leaf(
        id="Cadillac_Reservation_Requirement_required",
        desc="Cadillac Summit Road requires a timed vehicle reservation",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Driving Cadillac Summit Road requires a timed vehicle reservation.",
        node=cadillac_required_leaf,
        sources=ext.cadillac_urls,
        additional_instruction="Verify from official sources (e.g., NPS or Recreation.gov) that a reservation is required."
    )

    cadillac_fee_leaf = evaluator.add_leaf(
        id="Cadillac_Reservation_Requirement_fee",
        desc="Cadillac Summit Road reservation fee is $6",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The vehicle reservation fee for Cadillac Summit Road is $6.",
        node=cadillac_fee_leaf,
        sources=ext.cadillac_urls,
        additional_instruction="Confirm the exact fee amount ($6)."
    )

    cadillac_not_covered_leaf = evaluator.add_leaf(
        id="Cadillac_Reservation_Requirement_not_covered",
        desc="Cadillac reservation is not covered by any entrance pass",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Cadillac Summit Road vehicle reservation is not covered by entrance passes such as the America the Beautiful Annual Pass.",
        node=cadillac_not_covered_leaf,
        sources=ext.cadillac_urls,
        additional_instruction="Look for explicit statements indicating that the timed vehicle reservation is separate from, and not covered by, entrance passes."
    )

    # 3) Baxter_Pass_Requirements
    baxter_sources_exist = evaluator.add_custom_node(
        result=urls_non_empty(ext.baxter_acceptance_urls),
        id="Baxter_Pass_Requirements_sources_exist",
        desc="Baxter pass acceptance/requirements URLs provided",
        parent=node,
        critical=True
    )

    baxter_not_accepts_leaf = evaluator.add_leaf(
        id="Baxter_Pass_Requirements_not_accept_ABP",
        desc="Baxter does not accept America the Beautiful Pass or Maine State Park Pass",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Baxter State Park does not accept the America the Beautiful Annual Pass or the Maine State Park Pass.",
        node=baxter_not_accepts_leaf,
        sources=ext.baxter_acceptance_urls,
        additional_instruction="Confirm that neither the America the Beautiful Annual Pass nor the Maine State Park Pass is accepted at Baxter."
    )

    baxter_separate_fee_leaf = evaluator.add_leaf(
        id="Baxter_Pass_Requirements_separate_fee",
        desc="Baxter requires a separate entrance fee",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="A separate entrance fee is required to visit Baxter State Park.",
        node=baxter_separate_fee_leaf,
        sources=ext.baxter_acceptance_urls + ext.baxter_fee_urls,
        additional_instruction="Confirm existence of an entrance fee requirement at Baxter, independent of national/state passes."
    )

    # 4) Pass_Coverage_Reference (URLs present for pass acceptance at each park)
    pass_cov_refs = evaluator.add_custom_node(
        result=urls_non_empty(ext.acadia_pass_coverage_urls, ext.baxter_acceptance_urls),
        id="Pass_Coverage_Reference",
        desc="Provides URL reference confirming which passes are accepted at each park",
        parent=node,
        critical=True
    )
    _ = pass_cov_refs  # silence linter


async def build_cost_calculation_checks(evaluator: Evaluator, parent) -> None:
    """
    Build the 'Cost_Calculation' subtree (critical).
    """
    node = evaluator.add_parallel(
        id="Cost_Calculation",
        desc="Evaluates the accuracy of cost calculations for both pass strategies",
        parent=parent,
        critical=True
    )

    extraction_dict = evaluator._extraction_results[-1]["result"]
    ext = TripCostExtraction(**extraction_dict)

    # America_Beautiful_Strategy_Cost - break into detailed critical checks
    a_node = evaluator.add_parallel(
        id="America_Beautiful_Strategy_Cost",
        desc="Calculates total cost using America the Beautiful Annual Pass ($80) plus Cadillac reservation ($6) plus Baxter entrance fee ($50)",
        parent=node,
        critical=True
    )

    # Existence of URLs for each referenced fee (ABP, Cadillac, Baxter)
    a_urls_exist = evaluator.add_custom_node(
        result=urls_non_empty(ext.abp_cost_urls, ext.cadillac_urls, ext.baxter_fee_urls),
        id="A_URLs_exist",
        desc="Strategy A fee URLs provided (ABP, Cadillac, Baxter)",
        parent=a_node,
        critical=True
    )
    _ = a_urls_exist

    # Verify specific fee amounts per rubric
    a_abp_price_leaf = evaluator.add_leaf(
        id="A_ABP_price_80",
        desc="ABP price is $80",
        parent=a_node,
        critical=True
    )
    await evaluator.verify(
        claim="The America the Beautiful Annual Pass price is $80.",
        node=a_abp_price_leaf,
        sources=ext.abp_cost_urls,
        additional_instruction="Verify the standard America the Beautiful Annual Pass price ($80) from an authoritative source."
    )

    a_cadillac_fee_leaf = evaluator.add_leaf(
        id="A_Cadillac_fee_6",
        desc="Cadillac reservation fee is $6",
        parent=a_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Cadillac Summit Road vehicle reservation fee is $6.",
        node=a_cadillac_fee_leaf,
        sources=ext.cadillac_urls,
        additional_instruction="Confirm the exact fee amount ($6)."
    )

    a_baxter_fee_leaf = evaluator.add_leaf(
        id="A_Baxter_fee_50",
        desc="Baxter entrance fee is $50",
        parent=a_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Baxter State Park entrance fee used for this calculation is $50.",
        node=a_baxter_fee_leaf,
        sources=ext.baxter_fee_urls,
        additional_instruction="Confirm that the entrance fee amount used is $50."
    )

    a_total_leaf = evaluator.add_leaf(
        id="A_Total_136",
        desc="Strategy A total is $136",
        parent=a_node,
        critical=True
    )
    await evaluator.verify(
        claim="The stated total for Strategy A (America the Beautiful pass strategy) is $136.",
        node=a_total_leaf,
        additional_instruction="Treat minor formatting variants (e.g., '$136.00') as equivalent."
    )

    # Individual_Fees_Strategy_Cost - break into detailed critical checks
    b_node = evaluator.add_parallel(
        id="Individual_Fees_Strategy_Cost",
        desc="Calculates total cost using individual Acadia entrance fee ($35) plus Cadillac reservation ($6) plus Baxter entrance fee ($50)",
        parent=node,
        critical=True
    )

    b_urls_exist = evaluator.add_custom_node(
        result=urls_non_empty(ext.acadia_fee_urls, ext.cadillac_urls, ext.baxter_fee_urls),
        id="B_URLs_exist",
        desc="Strategy B fee URLs provided (Acadia, Cadillac, Baxter)",
        parent=b_node,
        critical=True
    )
    _ = b_urls_exist

    b_acadia_fee_leaf = evaluator.add_leaf(
        id="B_Acadia_fee_35",
        desc="Acadia entrance fee is $35",
        parent=b_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Acadia National Park private-vehicle entrance fee used for this calculation is $35.",
        node=b_acadia_fee_leaf,
        sources=ext.acadia_fee_urls,
        additional_instruction="Confirm the $35 Acadia NP entrance fee (7-day vehicle pass or equivalent as used)."
    )

    b_cadillac_fee_leaf = evaluator.add_leaf(
        id="B_Cadillac_fee_6",
        desc="Cadillac reservation fee is $6",
        parent=b_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Cadillac Summit Road vehicle reservation fee is $6.",
        node=b_cadillac_fee_leaf,
        sources=ext.cadillac_urls,
        additional_instruction="Confirm the exact fee amount ($6)."
    )

    b_baxter_fee_leaf = evaluator.add_leaf(
        id="B_Baxter_fee_50",
        desc="Baxter entrance fee is $50",
        parent=b_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Baxter State Park entrance fee used for this calculation is $50.",
        node=b_baxter_fee_leaf,
        sources=ext.baxter_fee_urls,
        additional_instruction="Confirm that the entrance fee amount used is $50."
    )

    b_total_leaf = evaluator.add_leaf(
        id="B_Total_91",
        desc="Strategy B total is $91",
        parent=b_node,
        critical=True
    )
    await evaluator.verify(
        claim="The stated total for Strategy B (individual fees strategy) is $91.",
        node=b_total_leaf,
        additional_instruction="Treat minor formatting variants (e.g., '$91.00') as equivalent."
    )

    # Cost_Calculation_Reference (all specific fees used have URLs)
    refs_leaf = evaluator.add_custom_node(
        result=urls_non_empty(ext.abp_cost_urls, ext.acadia_fee_urls, ext.cadillac_urls, ext.baxter_fee_urls),
        id="Cost_Calculation_Reference",
        desc="Provides URL references for the specific fees used in calculations",
        parent=node,
        critical=True
    )
    _ = refs_leaf


async def build_recommendation_check(evaluator: Evaluator, parent) -> None:
    """
    Build the 'Optimal_Strategy_Recommendation' check (non-critical).
    """
    extraction_dict = evaluator._extraction_results[-1]["result"]
    ext = TripCostExtraction(**extraction_dict)

    # Simple verification: ensure the answer recommends the cheaper strategy per rubric totals (A=$136 vs B=$91 → B cheaper)
    rec_leaf = evaluator.add_leaf(
        id="Optimal_Strategy_Recommendation",
        desc="Identifies which strategy is more cost-effective based on calculated totals",
        parent=parent,
        critical=False
    )
    await evaluator.verify(
        claim="Based on the totals ($136 for Strategy A vs $91 for Strategy B), the more cost-effective option for this trip is Strategy B (paying individual entrance fees).",
        node=rec_leaf,
        additional_instruction=(
            "Allow synonymous wording such as 'individual entrance fees', 'pay fees separately', 'Strategy B', etc. "
            "This check focuses only on whether the recommendation aligns with the cheaper option."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Maine 2026 pass strategy task.
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
        default_model=model
    )

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_trip_costs(),
        template_class=TripCostExtraction,
        extraction_name="trip_cost_extraction"
    )

    # Record ground truth expectations per rubric for transparency
    evaluator.add_ground_truth({
        "expected_costs": {
            "America_the_Beautiful_Annual_Pass": f"${ABP_COST_EXPECTED}",
            "Acadia_Individual_Entrance": f"${ACADIA_FEE_EXPECTED}",
            "Cadillac_Summit_Reservation": f"${CADILLAC_FEE_EXPECTED}",
            "Baxter_Entrance": f"${BAXTER_FEE_EXPECTED}"
        },
        "expected_totals": {
            "Strategy_A": f"${TOTAL_A_EXPECTED}",
            "Strategy_B": f"${TOTAL_B_EXPECTED}"
        },
        "expected_recommendation": "Strategy B (individual fees) is cheaper"
    }, gt_type="rubric_expected_values")

    # Build verification subtrees
    await build_pass_applicability_checks(evaluator, root)
    await build_cost_calculation_checks(evaluator, root)
    await build_recommendation_check(evaluator, root)

    # Return the final structured summary
    return evaluator.get_summary()