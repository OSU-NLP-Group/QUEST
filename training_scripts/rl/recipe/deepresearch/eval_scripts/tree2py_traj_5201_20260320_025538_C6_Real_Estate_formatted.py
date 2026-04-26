import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fha_condo_miami_2026"
TASK_DESCRIPTION = """
You are evaluating a condominium purchase in Miami-Dade County, Florida, for a buyer planning to use FHA financing. The specific property details are as follows:

- The condominium building is 4 habitable stories tall
- The building's certificate of occupancy was issued in January 1992 (making it 34 years old as of 2026)
- The building is located 2 miles from the Atlantic Ocean coastline
- Purchase price: $450,000
- The buyer plans a 5% down payment ($22,500)
- Buyer's credit score: 625
- Buyer's gross monthly income: $7,500
- Buyer's existing monthly debt obligations: $850 (car loan and credit card minimum payments)
- Estimated new monthly mortgage payment (PITI): $2,400
- Buyer intends to use property as primary residence

Association details:
- The association's 2026 annual budget allocates 12% to replacement reserves
- Owner occupancy rate: 62%
- Current delinquency rate: 9%
- The association completed its initial SIRS on November 15, 2025, and submitted it on December 10, 2025
- The building received written notice for milestone inspection on May 1, 2025, and completed Phase 1 inspection on August 15, 2025

Property condition:
- FHA appraisal confirmed foundation, roof, and structural elements meet standards
- All mechanical systems (heating, plumbing, electrical) are functional and code-compliant

Determine whether this condominium purchase meets all necessary requirements for FHA loan approval. Your answer should identify any compliance issues or confirm that all requirements are satisfied. For each major requirement category (building/association compliance, property/loan eligibility, and buyer qualification), provide specific verification of compliance or note any deficiencies.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CondoFHAExtraction(BaseModel):
    # Building/Association facts or the answer's stated values
    building_stories: Optional[str] = None
    certificate_of_occupancy_date: Optional[str] = None
    distance_to_coast_miles: Optional[str] = None
    milestone_notice_date: Optional[str] = None
    milestone_phase1_completion_date: Optional[str] = None
    sirs_completion_date: Optional[str] = None
    sirs_submission_date: Optional[str] = None
    reserves_percent: Optional[str] = None
    owner_occupancy_percent: Optional[str] = None
    delinquency_percent: Optional[str] = None
    milestone_sirs_urls: List[str] = Field(default_factory=list)

    # Property standards as stated in the answer
    structural_ok_statement: Optional[str] = None
    mechanical_ok_statement: Optional[str] = None

    # Loan details as stated in the answer
    purchase_price: Optional[str] = None
    down_payment_amount: Optional[str] = None
    down_payment_percent: Optional[str] = None
    computed_loan_amount: Optional[str] = None
    fha_limit_value_stated: Optional[str] = None
    fha_limit_context: Optional[str] = None  # e.g., "baseline", "high-cost"
    fha_limit_urls: List[str] = Field(default_factory=list)

    # Buyer and DTI details as stated in the answer
    credit_score: Optional[str] = None
    monthly_income: Optional[str] = None
    monthly_debts: Optional[str] = None
    piti: Optional[str] = None
    dti_value_stated: Optional[str] = None
    dti_calculation_statement: Optional[str] = None
    primary_residence_statement: Optional[str] = None

    # Homestead references (non-FHA informational)
    homestead_jan1_mentioned: Optional[bool] = None
    homestead_mar1_mentioned: Optional[bool] = None
    tax_urls: List[str] = Field(default_factory=list)

    # Overall conclusion section from the answer, if any
    overall_conclusion: Optional[str] = None
    general_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_fha_condo_info() -> str:
    return """
    Extract, from the answer (not from external knowledge), the key statements, values, and any cited URLs related to FHA condominium eligibility for this scenario. Keep all numeric values as strings exactly as written in the answer (do not normalize). If an item is not explicitly present, return null for that field. For URL arrays, return the actual URLs present in the answer; return an empty list if none.

    Fields to extract:
    - building_stories: The number of habitable stories referenced for the building (string).
    - certificate_of_occupancy_date: The date referenced for the building's CO (string as written).
    - distance_to_coast_miles: The distance to the coastline in miles (string as written).
    - milestone_notice_date: The date the building received the milestone inspection notice (string as written).
    - milestone_phase1_completion_date: The date Phase 1 was completed (string as written).
    - sirs_completion_date: The date the initial SIRS was completed (string as written).
    - sirs_submission_date: The date the SIRS was submitted (string as written).
    - reserves_percent: The association reserve allocation percent (string as written, e.g., "12%").
    - owner_occupancy_percent: The owner-occupancy rate percent (string as written).
    - delinquency_percent: The delinquency rate percent (string as written).
    - milestone_sirs_urls: Any URLs the answer cites regarding Florida milestone/SIRS rules (array).

    - structural_ok_statement: The exact sentence/phrase where the answer states the structure/foundation/roof meets FHA standards (string).
    - mechanical_ok_statement: The exact sentence/phrase where the answer states mechanical systems are functional and code-compliant (string).

    - purchase_price: The stated purchase price (string as written, e.g., "$450,000").
    - down_payment_amount: The down payment dollar amount (string as written).
    - down_payment_percent: The down payment percent (string as written, e.g., "5%").
    - computed_loan_amount: The loan amount the answer computes/states (string as written).
    - fha_limit_value_stated: The FHA loan limit value the answer cites (string, e.g., "$832,750" or "$1,249,125").
    - fha_limit_context: Any context label stated around the limit (e.g., "baseline", "high-cost", "Miami-Dade limit").
    - fha_limit_urls: Any URLs the answer cites for FHA loan limits (array).

    - credit_score: The buyer credit score as stated (string).
    - monthly_income: The gross monthly income as stated (string).
    - monthly_debts: The existing monthly debt as stated (string).
    - piti: The PITI as stated (string).
    - dti_value_stated: The DTI percent the answer computes/states (string as written, e.g., "43%" or "43.3%").
    - dti_calculation_statement: The sentence/phrase where the answer shows or describes the DTI calculation (string).
    - primary_residence_statement: The statement confirming primary residence intent/requirement (string).

    - homestead_jan1_mentioned: true if the answer explicitly mentions that Florida homestead requires primary residence as of January 1 for that tax year; false if it says the opposite; null if not mentioned.
    - homestead_mar1_mentioned: true if the answer explicitly mentions the March 1 homestead application deadline; false if it says the opposite; null if not mentioned.
    - tax_urls: Any URLs the answer cites for Florida homestead info (array).

    - overall_conclusion: The overall eligibility conclusion from the answer (string).
    - general_urls: Any other URLs the answer cites (array).
    """


# --------------------------------------------------------------------------- #
# Helper computations (ground-truth based on scenario details)                #
# --------------------------------------------------------------------------- #
def compute_scenario_ground_truth() -> dict:
    # Scenario constants
    purchase_price = 450_000
    down_payment_amount = 22_500
    loan_amount = purchase_price - down_payment_amount  # 427,500
    fha_baseline_limit = 832_750  # per rubric constraint for 2026 baseline
    fha_high_cost_cap = 1_249_125  # per rubric constraint
    income = 7_500
    existing_debts = 850
    piti = 2_400
    dti = (existing_debts + piti) / income  # 0.4333...

    # Florida Milestone/SIRS applicability and timing checks
    building_stories = 4
    distance_miles = 2.0
    # Trigger (>= 3 stories and within 3 miles uses 25-year threshold); building is 34 years old by 2026
    milestone_notice = datetime(2025, 5, 1)
    phase1_done = datetime(2025, 8, 15)
    days_to_phase1 = (phase1_done - milestone_notice).days  # should be <= 180

    sirs_completed = datetime(2025, 11, 15)
    sirs_submitted = datetime(2025, 12, 10)
    days_to_submit = (sirs_submitted - sirs_completed).days  # should be <= 45

    return {
        "purchase_price": purchase_price,
        "down_payment_amount": down_payment_amount,
        "loan_amount": loan_amount,
        "fha_baseline_limit": fha_baseline_limit,
        "fha_high_cost_cap": fha_high_cost_cap,
        "income": income,
        "existing_debts": existing_debts,
        "piti": piti,
        "dti": dti,
        "dti_percent_str": f"{round(dti * 100, 1)}%",  # e.g., "43.3%"
        "building_stories": building_stories,
        "distance_miles": distance_miles,
        "milestone_notice": "2025-05-01",
        "phase1_done": "2025-08-15",
        "days_to_phase1": days_to_phase1,
        "sirs_completed": "2025-11-15",
        "sirs_submitted": "2025-12-10",
        "days_to_submit": days_to_submit,
        # Association metrics (given)
        "reserves_pct": 12,
        "owner_occ_pct": 62,
        "delinq_pct": 9,
        # Property standards (given as satisfied)
        "structural_ok": True,
        "mechanical_ok": True,
        # Buyer core facts
        "credit_score": 625,
        "down_payment_percent": 5,  # %
        "primary_residence": True,
    }


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_building_and_association_checks(evaluator: Evaluator, parent, extracted: CondoFHAExtraction, gt: dict):
    """
    Build the Building & Association Compliance subtree:
    - Florida milestone/SIRS conditional compliance
    - FHA association budget/occupancy/delinquency checks
    """
    building_node = evaluator.add_parallel(
        id="Building_And_Association_Compliance",
        desc="Verify Florida building regulatory compliance (milestone/SIRS, if applicable) and FHA condo association criteria.",
        parent=parent,
        critical=True
    )

    # 1) Florida Milestone & SIRS compliance (conditional)
    milestone_leaf = evaluator.add_leaf(
        id="Florida_Milestone_And_SIRS_Compliance_If_Applicable",
        desc="Florida milestone & SIRS compliance (trigger applies and deadlines satisfied).",
        parent=building_node,
        critical=True
    )

    milestone_claim = (
        "Given the scenario facts, the Florida milestone inspection and SIRS requirements apply and are satisfied: "
        "the building has 4 habitable stories and is 2 miles from the coastline (so the 25-year trigger applies), "
        "Phase 1 was completed on 2025-08-15 within 180 days of the 2025-05-01 written notice, the initial SIRS was "
        "completed on 2025-11-15 (by the 2025-12-31 deadline), and it was submitted on 2025-12-10 (within 45 days). "
        "The answer reaches and states this compliant conclusion."
    )
    await evaluator.verify(
        claim=milestone_claim,
        node=milestone_leaf,
        additional_instruction=(
            "Focus on whether the answer explicitly recognizes applicability (4 stories, within ~3 miles) AND "
            "confirms timing compliance (≤180 days for Phase 1; SIRS done by Dec 31, 2025 and submitted within 45 days). "
            "Minor wording differences are fine if the conclusion is clearly 'compliant'."
        )
    )

    # 2) FHA association requirements (parallel criticals)
    assoc_node = evaluator.add_parallel(
        id="FHA_Association_Requirements",
        desc="Verify the condominium association meets FHA condo approval criteria.",
        parent=building_node,
        critical=True
    )

    # 2a) Replacement reserves ≥ 10%
    reserves_leaf = evaluator.add_leaf(
        id="Replacement_Reserves_At_Least_10pct",
        desc="Association allocates ≥ 10% of annual budget to replacement reserves.",
        parent=assoc_node,
        critical=True
    )
    reserves_claim = (
        "The association allocates 12% of its annual budget to replacement reserves, which meets or exceeds the FHA "
        "minimum requirement of 10%."
    )
    await evaluator.verify(
        claim=reserves_claim,
        node=reserves_leaf,
        additional_instruction="Check the answer acknowledges that 12% satisfies the ≥10% reserve allocation requirement."
    )

    # 2b) Owner occupancy ≥ 50%
    occ_leaf = evaluator.add_leaf(
        id="Owner_Occupancy_At_Least_50pct",
        desc="Owner-occupancy rate is at least 50%.",
        parent=assoc_node,
        critical=True
    )
    occ_claim = "The owner-occupancy rate is 62%, which is at least the FHA-required 50%."
    await evaluator.verify(
        claim=occ_claim,
        node=occ_leaf,
        additional_instruction="Verify the answer makes or supports the conclusion that 62% owner occupancy satisfies ≥50%."
    )

    # 2c) Delinquency < 15%
    delinq_leaf = evaluator.add_leaf(
        id="Delinquency_Below_15pct",
        desc="Association delinquency rate is below 15%.",
        parent=assoc_node,
        critical=True
    )
    delinq_claim = "The association delinquency rate is 9%, which is below the FHA threshold of 15%."
    await evaluator.verify(
        claim=delinq_claim,
        node=delinq_leaf,
        additional_instruction="Confirm the answer recognizes 9% is below the 15% cap."
    )


async def build_property_and_loan_checks(evaluator: Evaluator, parent, extracted: CondoFHAExtraction, gt: dict):
    """
    Build Property & Loan Eligibility subtree:
    - FHA Minimum Property Standards (structural + mechanical)
    - Loan limit compliance (sequential: compute loan amount → check against limit)
    """
    pl_node = evaluator.add_parallel(
        id="Property_And_Loan_Eligibility",
        desc="Verify property meets FHA MPS and loan amount is within applicable FHA limit.",
        parent=parent,
        critical=True
    )

    # 1) FHA Minimum Property Standards
    mps_node = evaluator.add_parallel(
        id="FHA_Minimum_Property_Standards",
        desc="FHA Minimum Property Standards verification.",
        parent=pl_node,
        critical=True
    )

    structural_leaf = evaluator.add_leaf(
        id="Structural_Soundness",
        desc="Foundation/roof/structural elements meet FHA standards.",
        parent=mps_node,
        critical=True
    )
    structural_claim = (
        "The answer confirms that the FHA appraisal found the foundation, roof, and structural elements meet FHA standards."
    )
    await evaluator.verify(
        claim=structural_claim,
        node=structural_leaf,
        additional_instruction="Accept equivalent wording such as 'meets FHA standards' or 'no material deficiencies' for structural elements."
    )

    mechanical_leaf = evaluator.add_leaf(
        id="Mechanical_Systems_Functional",
        desc="Mechanical systems (heating, plumbing, electrical) functional and code-compliant.",
        parent=mps_node,
        critical=True
    )
    mechanical_claim = (
        "The answer confirms that heating, plumbing, and electrical systems are functional and code-compliant."
    )
    await evaluator.verify(
        claim=mechanical_claim,
        node=mechanical_leaf,
        additional_instruction="Equivalent statements indicating proper operation and code compliance are acceptable."
    )

    # 2) Loan Limit Compliance (sequential)
    loan_seq = evaluator.add_sequential(
        id="FHA_Loan_Limit_Compliance",
        desc="Loan amount computation and limit compliance.",
        parent=pl_node,
        critical=True
    )

    # 2a) Compute loan amount
    compute_leaf = evaluator.add_leaf(
        id="Loan_Amount_Computed_From_Price_And_Down_Payment",
        desc="Loan amount equals purchase price minus down payment.",
        parent=loan_seq,
        critical=True
    )
    compute_claim = (
        "The answer correctly computes the base FHA loan amount as $427,500, calculated as $450,000 − $22,500."
    )
    await evaluator.verify(
        claim=compute_claim,
        node=compute_leaf,
        additional_instruction="Allow minor rounding and currency formatting differences; the math must align with price − down payment."
    )

    # 2b) Amount within limit
    limit_leaf = evaluator.add_leaf(
        id="Loan_Amount_Within_Constraint_Limit_Range",
        desc="Computed loan amount does not exceed applicable FHA limit.",
        parent=loan_seq,
        critical=True
    )
    limit_claim = (
        "At $427,500, the loan amount is within the FHA 2026 single-unit baseline limit of $832,750 "
        "and below any high-cost cap (e.g., $1,249,125). The answer reaches the same 'within limit' conclusion."
    )
    await evaluator.verify(
        claim=limit_claim,
        node=limit_leaf,
        additional_instruction="Check that the answer states or clearly implies the computed loan amount does not exceed the FHA limit."
    )


async def build_buyer_checks(evaluator: Evaluator, parent, extracted: CondoFHAExtraction, gt: dict):
    """
    Build Buyer Financial Qualification subtree:
    - Credit/Down payment eligibility
    - DTI calculation correctness (treated critical here to satisfy framework constraints)
    - DTI assessed against guidance
    - Primary residence requirement
    """
    buyer_node = evaluator.add_parallel(
        id="Buyer_Financial_Qualification",
        desc="Verify buyer credit/down payment, DTI, and primary occupancy requirements.",
        parent=parent,
        critical=True  # Made critical per rubric; all direct children must also be critical (framework constraint)
    )

    # Credit and down payment eligibility
    credit_dp_leaf = evaluator.add_leaf(
        id="Credit_And_Down_Payment_Eligibility",
        desc="Credit score and down payment satisfy FHA tiered minimums.",
        parent=buyer_node,
        critical=True
    )
    credit_dp_claim = (
        "With a 625 credit score (≥580) and a 5% down payment, the buyer meets FHA minimums "
        "(≥580 allows 3.5% down). The answer confirms this eligibility."
    )
    await evaluator.verify(
        claim=credit_dp_claim,
        node=credit_dp_leaf,
        additional_instruction="Look for an explicit tie between 625 score, ≥580 threshold, and ≥3.5% minimum down payment."
    )

    # DTI calculation correctness (treated as critical due to framework child-critical constraint)
    dti_calc_leaf = evaluator.add_leaf(
        id="DTI_Calculation_Correct",
        desc="DTI calculated correctly as (existing debts + PITI) / gross monthly income.",
        parent=buyer_node,
        critical=True
    )
    dti_calc_claim = (
        "The answer correctly calculates DTI as approximately 43.3% using (850 + 2,400) / 7,500."
    )
    await evaluator.verify(
        claim=dti_calc_claim,
        node=dti_calc_leaf,
        additional_instruction="Accept minor rounding (e.g., 43%–44%) if the formula and inputs are correct."
    )

    # DTI assessed against guidance
    dti_assess_leaf = evaluator.add_leaf(
        id="DTI_Assessed_Against_Constraint_Guidance",
        desc="DTI assessed against typical FHA guidance (≤43% typical; 43–50% may be acceptable with compensating factors; >50% flagged).",
        parent=buyer_node,
        critical=True
    )
    dti_assess_claim = (
        "The answer evaluates the ~43.3% DTI against FHA-style guidance (≤43% typical; 43–50% potentially acceptable with "
        "compensating factors; >50% a deficiency) and appropriately characterizes ~43.3% as borderline but potentially acceptable."
    )
    await evaluator.verify(
        claim=dti_assess_claim,
        node=dti_assess_leaf,
        additional_instruction="Look for explicit discussion aligning 43.3% with typical ≤43%/43–50% thresholds and compensating factors."
    )

    # Primary residence requirement
    primary_leaf = evaluator.add_leaf(
        id="Primary_Residence_Requirement",
        desc="Property will be used as the buyer's primary residence (FHA requirement).",
        parent=buyer_node,
        critical=True
    )
    primary_claim = "The answer confirms the property will be used as the buyer's primary residence, satisfying FHA occupancy."
    await evaluator.verify(
        claim=primary_claim,
        node=primary_leaf,
        additional_instruction="Confirm the answer clearly states primary residence use/intent."
    )


async def build_tax_benefit_checks(evaluator: Evaluator, parent, extracted: CondoFHAExtraction):
    """
    Build additional, non-FHA tax-benefit context subtree (homestead).
    This is non-critical and intentionally separated to satisfy framework critical-child constraints.
    """
    tax_node = evaluator.add_parallel(
        id="Tax_Benefit_Eligibility",
        desc="Additional (non-FHA) check: Florida homestead exemption timing requirements.",
        parent=parent,
        critical=False
    )

    jan1_leaf = evaluator.add_leaf(
        id="Homestead_Primary_Residence_As_Of_Jan1",
        desc="Answer mentions primary residence required as of January 1 for homestead eligibility.",
        parent=tax_node,
        critical=False
    )
    jan1_claim = (
        "The answer mentions that to claim Florida homestead exemption for a given tax year, "
        "the property must be the primary residence as of January 1 of that year."
    )
    await evaluator.verify(
        claim=jan1_claim,
        node=jan1_leaf,
        additional_instruction="Pass if the answer explicitly references this January 1 requirement; otherwise fail."
    )

    mar1_leaf = evaluator.add_leaf(
        id="Homestead_Application_Deadline_Mar1",
        desc="Answer mentions March 1 homestead exemption application deadline.",
        parent=tax_node,
        critical=False
    )
    mar1_claim = "The answer mentions the March 1 deadline to apply for the Florida homestead exemption."
    await evaluator.verify(
        claim=mar1_claim,
        node=mar1_leaf,
        additional_instruction="Pass if the answer explicitly references the March 1 homestead application deadline."
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
    Evaluate an answer for the FHA condo eligibility scenario in Miami-Dade (2026).
    Returns a structured summary including the verification tree and final score.
    """
    # Initialize evaluator (root is always non-critical; we'll add critical subtrees as needed)
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

    # Extract structured info from the answer (used to assist verification and recordkeeping)
    extracted = await evaluator.extract(
        prompt=prompt_extract_fha_condo_info(),
        template_class=CondoFHAExtraction,
        extraction_name="fha_condo_extraction"
    )

    # Compute scenario ground truth to record and for composing claims
    gt = compute_scenario_ground_truth()
    evaluator.add_ground_truth({
        "scenario_computed": gt,
        "expected_association": {
            "reserves_pct_at_least_10": True,
            "owner_occ_at_least_50": True,
            "delinq_below_15": True
        },
        "expected_property_MPS": {
            "structural_ok": True,
            "mechanical_ok": True
        },
        "expected_buyer_core": {
            "credit_downpayment_ok": True,
            "dti_about": gt["dti_percent_str"],
            "primary_residence": True
        },
        "fl_milestone_sirs_expected": {
            "trigger_applies": True,
            "phase1_within_180_days": gt["days_to_phase1"] <= 180,
            "sirs_completed_by_2025_12_31": True,
            "sirs_submitted_within_45_days": gt["days_to_submit"] <= 45
        },
        "loan_limit_expected": {
            "loan_amount": gt["loan_amount"],
            "within_baseline_limit": gt["loan_amount"] <= gt["fha_baseline_limit"]
        }
    })

    # Create the main critical eligibility aggregator (excluding the non-critical tax context)
    core_node = evaluator.add_parallel(
        id="Condominium_Purchase_Eligibility",
        desc="Evaluate whether the condominium purchase meets necessary FHA approval requirements.",
        parent=root,
        critical=True
    )

    # Build subtrees
    await build_building_and_association_checks(evaluator, core_node, extracted, gt)
    await build_property_and_loan_checks(evaluator, core_node, extracted, gt)
    await build_buyer_checks(evaluator, core_node, extracted, gt)

    # Add non-critical homestead context as a separate sibling subtree (to satisfy framework constraints)
    await build_tax_benefit_checks(evaluator, root, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()