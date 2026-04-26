import asyncio
import logging
import math
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sc_property_transaction_analysis"
TASK_DESCRIPTION = (
    "For a property sale in South Carolina with a purchase price of $265,000, where the buyer is a 66-year-old "
    "individual who has been a legal resident of South Carolina for 3 years and is purchasing the property as their "
    "primary residence, provide the following information: (1) Calculate the total deed recording fee (transfer tax) "
    "that must be paid at closing, including the breakdown of state and county portions. (2) Determine the property "
    "assessment ratio that the buyer will qualify for as an owner-occupant of their primary residence. (3) Determine "
    "whether the buyer qualifies for the homestead exemption and, if so, state the fair market value amount that will "
    "be exempt from property taxation."
)

PURCHASE_PRICE = 265000
FEE_RATE_TOTAL_PER_500 = 1.85
FEE_RATE_STATE_PER_500 = 1.30
FEE_RATE_COUNTY_PER_500 = 0.55
EXPECTED_ASSESSMENT_RATIO_STR = "4%"  # Owner-occupied legal residence (primary) assessment ratio in SC
EXPECTED_HOMESTEAD_EXEMPT_AMOUNT_STR = "$50,000"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DeedFeeExtraction(BaseModel):
    total_fee: Optional[str] = None
    state_fee: Optional[str] = None
    county_fee: Optional[str] = None
    explanation: Optional[str] = None


class AssessmentExtraction(BaseModel):
    assessment_ratio: Optional[str] = None
    explanation: Optional[str] = None


class HomesteadExtraction(BaseModel):
    eligibility_statement: Optional[str] = None
    uses_conditional_language: Optional[bool] = None
    exempt_amount: Optional[str] = None
    explanation: Optional[str] = None


class SCTransactionExtraction(BaseModel):
    deed_fee: Optional[DeedFeeExtraction] = None
    assessment: Optional[AssessmentExtraction] = None
    homestead: Optional[HomesteadExtraction] = None


# --------------------------------------------------------------------------- #
# Helper computations                                                         #
# --------------------------------------------------------------------------- #
def compute_deed_fee_breakdown(purchase_price: float) -> Dict[str, Any]:
    """
    Compute SC deed recording fee totals and breakdown using:
      - $1.85 per $500 (or fractional part thereof) total
      - State portion: $1.30 per $500
      - County portion: $0.55 per $500
    """
    increments = math.ceil(purchase_price / 500.0)
    total_fee = increments * FEE_RATE_TOTAL_PER_500
    state_fee = increments * FEE_RATE_STATE_PER_500
    county_fee = increments * FEE_RATE_COUNTY_PER_500
    return {
        "increments": increments,
        "total_fee": total_fee,
        "state_fee": state_fee,
        "county_fee": county_fee,
        "total_fee_fmt": f"${total_fee:,.2f}",
        "state_fee_fmt": f"${state_fee:,.2f}",
        "county_fee_fmt": f"${county_fee:,.2f}",
    }


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_sc_transaction() -> str:
    return """
    Extract the specific information the answer provides about this South Carolina property transaction.

    1) Deed Recording Fee:
       - total_fee: The total deed recording fee (transfer tax) amount stated in the answer (as a string, include currency formatting if present).
       - state_fee: The state portion amount stated (as a string).
       - county_fee: The county portion amount stated (as a string).
       - explanation: Any brief explanation or rate description used (optional).

    2) Assessment Ratio:
       - assessment_ratio: The ratio stated for an owner-occupied primary residence (as a string, e.g., "4%" or "0.04").
       - explanation: Any brief explanation the answer provides (optional).

    3) Homestead Exemption:
       - eligibility_statement: The answer's statement regarding eligibility (quote or paraphrase from the answer).
       - uses_conditional_language: Return true if the answer uses conditional phrasing (e.g., "if approved", "subject to", "assuming") acknowledging prerequisites/conditions; return false if it asserts eligibility unconditionally; if not applicable or unknown, return false.
       - exempt_amount: The stated fair market value amount that is exempt under the homestead exemption (as a string, e.g., "$50,000"), if the answer provides it.
       - explanation: Any brief explanation or prerequisites mentioned (optional).

    Rules:
    - Extract only what is explicitly in the answer; do not invent values.
    - If a requested field is missing, set it to null.
    """


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_deed_recording_fee_checks(
    evaluator: Evaluator,
    parent_node,
    extraction: SCTransactionExtraction,
) -> None:
    """
    Build the Deed Recording Fee verification subtree:
      - Fee_Total (critical leaf)
      - Fee_Breakdown (critical leaf)
    """
    expected = compute_deed_fee_breakdown(PURCHASE_PRICE)

    drf_node = evaluator.add_parallel(
        id="Deed_Recording_Fee",
        desc="Compute the deed recording fee (transfer tax) for the $265,000 transfer using the given SC rate and provide state/county portions.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Fee_Total
    fee_total_leaf = evaluator.add_leaf(
        id="Fee_Total",
        desc="Provides the correct total deed recording fee for $265,000 using $1.85 per $500 (or fractional part thereof).",
        parent=drf_node,
        critical=True,
    )

    extracted_total = (extraction.deed_fee.total_fee if extraction and extraction.deed_fee else None)
    extracted_total_str = extracted_total if extracted_total else "not provided"
    claim_total = (
        f"According to the answer, the total deed recording fee for a ${PURCHASE_PRICE:,} transfer is '{extracted_total_str}'. "
        f"Using South Carolina's rate of $1.85 per $500 (or fractional part thereof), the correct total is {expected['total_fee_fmt']}. "
        f"Judge whether the answer's number matches the correct total (allow ±$0.01 tolerance and minor formatting differences). "
        f"If the answer does not provide a total amount, judge this as incorrect."
    )
    await evaluator.verify(
        claim=claim_total,
        node=fee_total_leaf,
        additional_instruction=(
            f"Computation details: increments = ceil({PURCHASE_PRICE}/500) = {expected['increments']}, "
            f"total = increments × 1.85 = {expected['total_fee_fmt']}."
        ),
    )

    # Leaf: Fee_Breakdown
    fee_breakdown_leaf = evaluator.add_leaf(
        id="Fee_Breakdown",
        desc="Provides a correct breakdown into state portion ($1.30 per $500) and county portion ($0.55 per $500), consistent with the total.",
        parent=drf_node,
        critical=True,
    )

    extracted_state = (extraction.deed_fee.state_fee if extraction and extraction.deed_fee else None)
    extracted_county = (extraction.deed_fee.county_fee if extraction and extraction.deed_fee else None)
    extracted_state_str = extracted_state if extracted_state else "not provided"
    extracted_county_str = extracted_county if extracted_county else "not provided"

    claim_breakdown = (
        f"The answer breaks down the deed recording fee as: state portion '{extracted_state_str}' and county portion '{extracted_county_str}'. "
        f"Verify whether these match South Carolina's breakdown for a ${PURCHASE_PRICE:,} transfer: "
        f"state = {expected['state_fee_fmt']} (using $1.30 per $500) and county = {expected['county_fee_fmt']} (using $0.55 per $500). "
        f"Also verify that the sum of the stated state and county portions equals the correct total {expected['total_fee_fmt']} "
        f"(allow ±$0.01 tolerance and minor formatting differences). If either portion is missing, judge as incorrect."
    )
    await evaluator.verify(
        claim=claim_breakdown,
        node=fee_breakdown_leaf,
        additional_instruction=(
            f"Computation details: increments = {expected['increments']}; "
            f"state = increments × 1.30 = {expected['state_fee_fmt']}; "
            f"county = increments × 0.55 = {expected['county_fee_fmt']}; "
            f"sum must equal total {expected['total_fee_fmt']}."
        ),
    )


async def build_assessment_ratio_checks(
    evaluator: Evaluator,
    parent_node,
    extraction: SCTransactionExtraction,
) -> None:
    """
    Build the Assessment Ratio verification subtree:
      - Ratio_For_Primary_Residence (critical leaf)
    """
    assess_node = evaluator.add_parallel(
        id="Assessment_Ratio",
        desc="Identify the assessment ratio applicable to the buyer as an owner-occupant of their primary residence.",
        parent=parent_node,
        critical=True,
    )

    ratio_leaf = evaluator.add_leaf(
        id="Ratio_For_Primary_Residence",
        desc="Correctly states that an owner-occupied primary residence (legal residence) qualifies for the 4% assessment ratio (and does not incorrectly apply the 6% non-primary ratio).",
        parent=assess_node,
        critical=True,
    )

    extracted_ratio = (extraction.assessment.assessment_ratio if extraction and extraction.assessment else None)
    extracted_ratio_str = extracted_ratio if extracted_ratio else "not provided"

    claim_ratio = (
        f"The answer states the assessment ratio for an owner-occupied primary residence as '{extracted_ratio_str}'. "
        f"Verify that in South Carolina the legal residence (primary residence) assessment ratio is 4% (i.e., '{EXPECTED_ASSESSMENT_RATIO_STR}' or '0.04'), "
        f"and that 6% applies to non-primary residences. If the answer fails to state 4% or states 6% for primary, judge as incorrect."
    )
    await evaluator.verify(
        claim=claim_ratio,
        node=ratio_leaf,
        additional_instruction="Accept equivalent forms such as '4%' or '0.04'. Focus on primary residence vs non-primary distinction.",
    )


async def build_homestead_exemption_checks(
    evaluator: Evaluator,
    parent_node,
    extraction: SCTransactionExtraction,
) -> None:
    """
    Build the Homestead Exemption verification subtree:
      - Eligibility_Determination (critical leaf)
      - Exempt_Amount (critical leaf, sequentially after eligibility)
    """
    homestead_node = evaluator.add_sequential(
        id="Homestead_Exemption",
        desc="Determine whether the buyer qualifies for the SC Homestead Exemption under the provided criteria and, if eligible, state the exempt fair market value amount.",
        parent=parent_node,
        critical=True,
    )

    # Eligibility determination leaf
    eligibility_leaf = evaluator.add_leaf(
        id="Eligibility_Determination",
        desc="Determines eligibility using the provided criteria: age 65+ (buyer is 66) and SC legal residency for at least one calendar year (buyer has 3 years), and acknowledges required prerequisites/conditions from constraints (legal residence approval by county assessor; title must be solely to applicant or jointly with spouse only). If any prerequisite cannot be confirmed from the given facts, the answer is stated conditionally rather than asserting unconditional qualification.",
        parent=homestead_node,
        critical=True,
    )

    eligibility_stmt = (extraction.homestead.eligibility_statement if extraction and extraction.homestead else None)
    conditional_flag = (extraction.homestead.uses_conditional_language if extraction and extraction.homestead else None)
    eligibility_stmt_str = eligibility_stmt if eligibility_stmt else "not provided"
    conditional_flag_str = "true" if conditional_flag else "false"

    claim_eligibility = (
        f"The buyer is 66 years old and has been a South Carolina legal resident for 3 years, and intends the property as a primary residence. "
        f"These meet the age and residency criteria for SC Homestead Exemption. However, full eligibility also requires that the property be approved as the legal residence "
        f"by the county assessor and that title is held solely by the applicant or jointly with the spouse only (no other co-owners). "
        f"The answer's eligibility statement is: '{eligibility_stmt_str}', and it uses conditional phrasing: {conditional_flag_str}. "
        f"Verify that the answer appropriately acknowledges these prerequisites and does not unconditionally assert eligibility if such prerequisites are not confirmed. "
        f"If the answer is unconditional without acknowledging prerequisites, judge as incorrect."
    )
    await evaluator.verify(
        claim=claim_eligibility,
        node=eligibility_leaf,
        additional_instruction=(
            "Look for wording such as 'if approved', 'subject to', 'assuming legal residence is granted', "
            "or mention that the title must be solely to the applicant or jointly with spouse, and application with the county assessor is required. "
            "If such conditions are not confirmed in the given facts, the answer should be conditional."
        ),
    )

    # Exempt amount leaf (sequentially depends on eligibility)
    exempt_leaf = evaluator.add_leaf(
        id="Exempt_Amount",
        desc="If eligible, states that the first $50,000 of fair market value is exempt from property taxation under the Homestead Exemption.",
        parent=homestead_node,
        critical=True,
    )

    exempt_amount = (extraction.homestead.exempt_amount if extraction and extraction.homestead else None)
    exempt_amount_str = exempt_amount if exempt_amount else "not provided"
    claim_exempt = (
        f"The answer states the homestead exemption amount as '{exempt_amount_str}'. "
        f"Verify that under the South Carolina Homestead Exemption, the first $50,000 of fair market value is exempt from property taxation. "
        f"If the answer provides a different amount or none, judge as incorrect."
    )
    await evaluator.verify(
        claim=claim_exempt,
        node=exempt_leaf,
        additional_instruction="Accept '$50,000' or equivalent formatting; amounts must match exactly (±$0.01 tolerance for formatting).",
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
    Evaluate an answer for South Carolina property transaction analysis.
    """
    # Initialize evaluator (root is non-critical by framework design)
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_sc_transaction(),
        template_class=SCTransactionExtraction,
        extraction_name="sc_transaction_extraction",
    )

    # Compute ground-truth expected values for display/reference
    expected = compute_deed_fee_breakdown(PURCHASE_PRICE)
    evaluator.add_ground_truth({
        "purchase_price": f"${PURCHASE_PRICE:,.2f}",
        "deed_fee_expected": {
            "increments": expected["increments"],
            "total_fee": expected["total_fee_fmt"],
            "state_fee": expected["state_fee_fmt"],
            "county_fee": expected["county_fee_fmt"],
            "rates_per_500": {
                "total": f"${FEE_RATE_TOTAL_PER_500:.2f}",
                "state": f"${FEE_RATE_STATE_PER_500:.2f}",
                "county": f"${FEE_RATE_COUNTY_PER_500:.2f}",
            },
        },
        "assessment_ratio_expected": EXPECTED_ASSESSMENT_RATIO_STR,
        "homestead_exempt_amount_expected": EXPECTED_HOMESTEAD_EXEMPT_AMOUNT_STR,
        "homestead_prerequisites": [
            "Legal residence approval by county assessor",
            "Title held solely by applicant or jointly with spouse only",
        ],
    })

    # Build the strict critical analysis subtree under a critical aggregator
    sc_root = evaluator.add_parallel(
        id="SC_Property_Transaction_Analysis",
        desc="Accurate analysis for the described South Carolina property sale: deed recording fee (with state/county breakdown), assessment ratio for owner-occupied primary residence, and homestead exemption qualification and exempt amount.",
        parent=root,
        critical=True,
    )

    # Deed Recording Fee checks
    await build_deed_recording_fee_checks(evaluator, sc_root, extraction)

    # Assessment Ratio checks
    await build_assessment_ratio_checks(evaluator, sc_root, extraction)

    # Homestead Exemption checks
    await build_homestead_exemption_checks(evaluator, sc_root, extraction)

    # Return standardized summary
    return evaluator.get_summary()