import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants and computed expectations                           #
# --------------------------------------------------------------------------- #
TASK_ID = "home_purchase_financial_2026"
TASK_DESCRIPTION = (
    "A single filer with an annual income of $95,000 is purchasing a home for $400,000 in a standard U.S. county "
    "(with baseline 2026 conforming loan limits) and plans to make a $40,000 down payment. The buyer has a 3-person "
    "household and current monthly debt obligations of $800. The estimated total monthly housing payment (including "
    "principal, interest, property taxes, homeowners insurance, and PMI) will be approximately $2,500.\n\n"
    "For this home purchase transaction, provide the following information:\n\n"
    "1. What is the mortgage loan amount?\n"
    "2. What is the down payment as a percentage of the purchase price?\n"
    "3. Is private mortgage insurance (PMI) required for this loan?\n"
    "4. What is the estimated closing costs range in dollars (based on the typical 2-5% of loan amount)?\n"
    "5. What is the recommended earnest money deposit range in dollars (based on the typical 1-3% of purchase price)?\n"
    "6. Is the loan amount within the 2026 conforming loan limit for standard counties? State the baseline conforming loan limit.\n"
    "7. Does the household income qualify for a USDA loan based on 2026 income limits for a 3-person household? State the applicable income limit.\n"
    "8. Does the buyer's debt-to-income ratio meet the FHA maximum requirement? State the FHA DTI limit and the buyer's calculated DTI ratio.\n"
    "9. Does the buyer's debt-to-income ratio meet the Conventional loan maximum requirement? State the Conventional DTI limit and the buyer's calculated DTI ratio.\n"
    "10. What is the applicable long-term capital gains tax rate (0%, 15%, or 20%) for this single filer based on their income?\n"
    "11. What is the SALT deduction cap applicable for tax years 2025-2026?\n"
    "12. If the property were converted to a rental, what is the depreciation period for residential rental property per IRS guidelines?\n"
    "13. At what loan balance amount (in dollars) can PMI be automatically removed, based on the 78% threshold of the original property value?"
)

# Given problem inputs
PURCHASE_PRICE = 400_000.0
DOWN_PAYMENT = 40_000.0
LOAN_AMOUNT_EXPECTED = PURCHASE_PRICE - DOWN_PAYMENT  # 360,000
DOWN_PCT_EXPECTED = DOWN_PAYMENT / PURCHASE_PRICE  # 0.10
CLOSING_LOW = 0.02 * LOAN_AMOUNT_EXPECTED  # 7,200
CLOSING_HIGH = 0.05 * LOAN_AMOUNT_EXPECTED  # 18,000
EARNEST_LOW = 0.01 * PURCHASE_PRICE  # 4,000
EARNEST_HIGH = 0.03 * PURCHASE_PRICE  # 12,000
PMI_REQUIRED_EXPECTED = True  # Down < 20%
ANNUAL_INCOME = 95_000.0
GROSS_MONTHLY_INCOME = ANNUAL_INCOME / 12.0  # 7,916.666...
MONTHLY_OTHER_DEBT = 800.0
MONTHLY_HOUSING_PAYMENT = 2_500.0
DTI_DECIMAL = (MONTHLY_HOUSING_PAYMENT + MONTHLY_OTHER_DEBT) / GROSS_MONTHLY_INCOME  # ~0.4167
PMI_AUTO_REMOVE_BAL_EXPECTED = 0.78 * PURCHASE_PRICE  # 312,000

# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def fmt_money(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)

def fmt_percent_decimal(p: float, decimals: int = 1) -> str:
    try:
        return f"{round(p * 100, decimals)}%"
    except Exception:
        return f"{p*100}%"

EXPECTED_VALUES = {
    "expected_loan_amount": fmt_money(LOAN_AMOUNT_EXPECTED),
    "expected_down_payment_percentage": fmt_percent_decimal(DOWN_PCT_EXPECTED),
    "expected_pmi_required": "Yes" if PMI_REQUIRED_EXPECTED else "No",
    "expected_closing_costs_range": f"{fmt_money(CLOSING_LOW)} - {fmt_money(CLOSING_HIGH)}",
    "expected_earnest_money_range": f"{fmt_money(EARNEST_LOW)} - {fmt_money(EARNEST_HIGH)}",
    "expected_dti_ratio": fmt_percent_decimal(DTI_DECIMAL),
    "expected_pmi_auto_remove_balance": fmt_money(PMI_AUTO_REMOVE_BAL_EXPECTED),
    "constants": {
        "purchase_price": fmt_money(PURCHASE_PRICE),
        "down_payment": fmt_money(DOWN_PAYMENT),
        "gross_monthly_income": f"${GROSS_MONTHLY_INCOME:,.2f}",
        "monthly_housing": fmt_money(MONTHLY_HOUSING_PAYMENT),
        "monthly_other_debt": fmt_money(MONTHLY_OTHER_DEBT),
    }
}

# --------------------------------------------------------------------------- #
# Extraction model                                                            #
# --------------------------------------------------------------------------- #
class HomePurchaseExtraction(BaseModel):
    # Core answers
    loan_amount: Optional[str] = None
    down_payment_percentage: Optional[str] = None
    pmi_required: Optional[str] = None  # Expect 'Yes'/'No' or equivalent
    closing_costs_range: Optional[str] = None  # e.g., "$7,200 - $18,000"
    earnest_money_range: Optional[str] = None  # e.g., "$4,000 - $12,000"

    # Conforming loan limit (2026 baseline)
    conforming_baseline_limit_2026: Optional[str] = None
    conforming_limit_sources: List[str] = Field(default_factory=list)

    # USDA income eligibility (2026)
    usda_income_limit_2026: Optional[str] = None
    usda_sources: List[str] = Field(default_factory=list)

    # DTI & program limits
    buyer_dti_ratio: Optional[str] = None  # e.g., "41.7%"
    fha_max_dti_limit: Optional[str] = None
    fha_dti_sources: List[str] = Field(default_factory=list)
    conventional_max_dti_limit: Optional[str] = None
    conventional_dti_sources: List[str] = Field(default_factory=list)

    # Tax items
    ltcg_tax_rate: Optional[str] = None  # e.g., "15%"
    ltcg_sources: List[str] = Field(default_factory=list)
    salt_cap_2025_2026: Optional[str] = None  # e.g., "$10,000"
    salt_sources: List[str] = Field(default_factory=list)

    # Rental depreciation
    rental_depreciation_period: Optional[str] = None  # e.g., "27.5 years"
    rental_depreciation_method: Optional[str] = None  # e.g., "straight-line under MACRS"
    rental_depreciation_sources: List[str] = Field(default_factory=list)

    # PMI removal threshold balance
    pmi_auto_remove_balance: Optional[str] = None  # e.g., "$312,000"

# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_home_purchase() -> str:
    return """
Extract the following items as they are explicitly stated in the answer text. If a field is not stated, return null for that field. For any fact that the answer attributes to external sources (webpages), extract the actual URLs into the corresponding sources list (if provided). Do not invent URLs.

Required fields to extract:
1) loan_amount: The mortgage loan amount stated in the answer (string).
2) down_payment_percentage: The down payment as a percent of the purchase price as stated (e.g., "10%") (string).
3) pmi_required: Whether PMI is required, as stated (e.g., "Yes" or "No") (string).
4) closing_costs_range: A dollar range for estimated closing costs (e.g., "$7,200 - $18,000") (string).
5) earnest_money_range: A dollar range for recommended earnest money (e.g., "$4,000 - $12,000") (string).

Conforming loan limit (2026):
6) conforming_baseline_limit_2026: The 2026 baseline conforming loan limit for standard counties as stated (string).
7) conforming_limit_sources: All URLs cited that support the 2026 baseline limit (array of strings).

USDA:
8) usda_income_limit_2026: The 2026 USDA income limit for a household size band that covers 3-person households, as stated (string).
9) usda_sources: All URLs cited that support the USDA income limit (array of strings).

DTI & program limits:
10) buyer_dti_ratio: The buyer’s calculated DTI ratio as stated (e.g., "41.7%") (string).
11) fha_max_dti_limit: The FHA maximum DTI limit as stated (string).
12) fha_dti_sources: All URLs cited that support the FHA DTI limit (array of strings).
13) conventional_max_dti_limit: The Conventional maximum DTI limit as stated (string).
14) conventional_dti_sources: All URLs cited that support the Conventional DTI limit (array of strings).

Taxes:
15) ltcg_tax_rate: The applicable long-term capital gains tax rate (0%, 15%, or 20%) as stated for the single filer (string).
16) ltcg_sources: All URLs cited that support that capital gains rate (array of strings).
17) salt_cap_2025_2026: The SALT deduction cap applicable for tax years 2025–2026 as stated (string).
18) salt_sources: All URLs cited that support the SALT cap (array of strings).

Rental depreciation:
19) rental_depreciation_period: The depreciation period stated for residential rental property (string, e.g., "27.5 years").
20) rental_depreciation_method: The method stated (e.g., "straight-line under MACRS") (string).
21) rental_depreciation_sources: URLs cited that support the rental depreciation period/method (array of strings).

PMI removal threshold:
22) pmi_auto_remove_balance: The loan balance amount at which PMI is automatically removed based on the 78% of original value threshold, as stated in the answer (string, e.g., "$312,000").

IMPORTANT:
- Return exactly the fields above in JSON. Use null for any missing value.
- For sources fields, only include URLs explicitly shown in the answer (plain URLs or in markdown links). If none, return an empty list.
- Preserve units and symbols (like %, $) as present in the answer for value fields.
"""

# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
ROUNDING_TOLERANCE_INSTRUCTION = (
    "Allow minor rounding differences and formatting variations (currency symbols, commas, and whitespace). "
    "Treat values as matching if they are approximately equal within ordinary rounding tolerance."
)

def _safe(val: Optional[str]) -> str:
    return val if (val is not None and str(val).strip() != "") else "not stated"

# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, extraction: HomePurchaseExtraction) -> None:
    # Create top-level analysis node (critical; all children must be critical)
    analysis = evaluator.add_parallel(
        id="Home_Purchase_Financial_Analysis",
        desc="Complete financial analysis and eligibility verification for the home purchase transaction (answers all required sub-questions).",
        parent=evaluator.root,
        critical=True
    )

    # 1) Loan Amount Calculation (leaf)
    loan_node = evaluator.add_leaf(
        id="Loan_Amount_Calculation",
        desc="State the mortgage loan amount, computed as purchase price minus down payment.",
        parent=analysis,
        critical=True
    )
    expected_loan_txt = fmt_money(LOAN_AMOUNT_EXPECTED)
    await evaluator.verify(
        claim=(
            f"The answer states the mortgage loan amount as {_safe(extraction.loan_amount)}. "
            f"Based on a $400,000 purchase price and a $40,000 down payment, the correct loan amount is about {expected_loan_txt}. "
            "These should match within normal rounding/formatting tolerance."
        ),
        node=loan_node,
        additional_instruction=ROUNDING_TOLERANCE_INSTRUCTION
    )

    # 2) Down Payment Percentage (leaf)
    dpp_node = evaluator.add_leaf(
        id="Down_Payment_Percentage",
        desc="State the down payment as a percentage of the purchase price.",
        parent=analysis,
        critical=True
    )
    expected_pct_txt = fmt_percent_decimal(DOWN_PCT_EXPECTED)
    await evaluator.verify(
        claim=(
            f"The answer states the down payment percentage as {_safe(extraction.down_payment_percentage)}. "
            f"From a $40,000 down payment on a $400,000 purchase, it is approximately {expected_pct_txt}. "
            "These should match within ordinary rounding tolerance."
        ),
        node=dpp_node,
        additional_instruction=ROUNDING_TOLERANCE_INSTRUCTION
    )

    # 3) PMI Requirement Status (leaf)
    pmi_req_node = evaluator.add_leaf(
        id="PMI_Requirement_Status",
        desc="State whether PMI is required, based on whether the down payment is less than 20% of the purchase price.",
        parent=analysis,
        critical=True
    )
    expected_pmi_txt = "Yes" if PMI_REQUIRED_EXPECTED else "No"
    await evaluator.verify(
        claim=(
            f"The answer states whether PMI is required as {_safe(extraction.pmi_required)}. "
            "With a 10% down payment (less than 20%), PMI should be required (i.e., 'Yes'). "
            f"This should match the expected result '{expected_pmi_txt}'."
        ),
        node=pmi_req_node,
        additional_instruction=(
            "Focus on the logical rule: PMI is typically required when down payment < 20%. "
            "Minor wording variations like 'PMI applies' or 'PMI needed' should count as 'Yes'."
        )
    )

    # 4) Closing Costs Range 2–5% of loan amount (leaf)
    closing_node = evaluator.add_leaf(
        id="Closing_Costs_Range",
        desc="Provide estimated closing costs range in dollars using 2% to 5% of the loan amount.",
        parent=analysis,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The answer provides an estimated closing costs range as {_safe(extraction.closing_costs_range)}. "
            f"Using 2% to 5% of the loan amount {expected_loan_txt}, the range is about "
            f"{fmt_money(CLOSING_LOW)} to {fmt_money(CLOSING_HIGH)}. "
            "The provided range should align with this computation allowing normal rounding."
        ),
        node=closing_node,
        additional_instruction=ROUNDING_TOLERANCE_INSTRUCTION
    )

    # 5) Earnest Money 1–3% of purchase price (leaf)
    earnest_node = evaluator.add_leaf(
        id="Earnest_Money_Deposit_Range",
        desc="Provide recommended earnest money deposit range in dollars using 1% to 3% of the purchase price.",
        parent=analysis,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The answer provides a recommended earnest money range as {_safe(extraction.earnest_money_range)}. "
            f"Using 1% to 3% of the purchase price {fmt_money(PURCHASE_PRICE)}, the range is about "
            f"{fmt_money(EARNEST_LOW)} to {fmt_money(EARNEST_HIGH)}. "
            "The stated range should match this within ordinary rounding tolerance."
        ),
        node=earnest_node,
        additional_instruction=ROUNDING_TOLERANCE_INSTRUCTION
    )

    # 6) Conforming Loan Limit Check (sequential)
    conforming_seq = evaluator.add_sequential(
        id="Conforming_Loan_Limit_Check",
        desc="Assess whether the loan amount is within the 2026 baseline conforming loan limit for standard counties, and state that baseline limit.",
        parent=analysis,
        critical=True
    )
    # 6.1 State baseline (verify by URLs)
    baseline_leaf = evaluator.add_leaf(
        id="State_Baseline_Conforming_Loan_Limit_2026",
        desc="State the 2026 baseline conforming loan limit for standard counties (per constraints).",
        parent=conforming_seq,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The 2026 baseline conforming loan limit for standard counties is stated as "
            f"{_safe(extraction.conforming_baseline_limit_2026)}."
        ),
        node=baseline_leaf,
        sources=extraction.conforming_limit_sources,
        additional_instruction=(
            "Verify the page explicitly states the 2026 baseline conforming loan limit for standard counties (1-unit). "
            "Minor formatting differences are fine."
        )
    )
    # 6.2 Verify loan amount within limit (simple logic)
    within_limit_leaf = evaluator.add_leaf(
        id="Verify_Loan_Amount_Within_Conforming_Limit",
        desc="Determine whether the computed loan amount is within the stated 2026 baseline conforming loan limit.",
        parent=conforming_seq,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"With a mortgage loan amount of about {expected_loan_txt} and a stated 2026 baseline limit of "
            f"{_safe(extraction.conforming_baseline_limit_2026)}, the loan amount is within the limit."
        ),
        node=within_limit_leaf,
        additional_instruction=(
            "Treat the inequality check logically using the numbers given; rounding/formatting differences are acceptable."
        )
    )

    # 7) USDA Income Eligibility Check (sequential)
    usda_seq = evaluator.add_sequential(
        id="USDA_Income_Eligibility_Check",
        desc="Assess whether the household income qualifies for a USDA loan for a 3-person household under 2026 limits, and state the applicable income limit.",
        parent=analysis,
        critical=True
    )
    # 7.1 State USDA limit (verify by URLs)
    usda_limit_leaf = evaluator.add_leaf(
        id="State_USDA_Income_Limit_2026",
        desc="State the applicable 2026 USDA income limit for the relevant household size band covering a 3-person household (per constraints).",
        parent=usda_seq,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The 2026 USDA income limit applicable to a 3-person household (covered by the relevant 1–4 person band if applicable) "
            f"is stated as {_safe(extraction.usda_income_limit_2026)}."
        ),
        node=usda_limit_leaf,
        sources=extraction.usda_sources,
        additional_instruction=(
            "Confirm the page provides the applicable USDA income limit for 2026 for a 3-person household (or the 1–4 person category)."
        )
    )
    # 7.2 Determine eligibility (simple logic)
    usda_elig_leaf = evaluator.add_leaf(
        id="Verify_Income_Qualifies_For_USDA",
        desc="Determine whether the buyer's annual income is at or below the stated USDA income limit.",
        parent=usda_seq,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"Given the buyer's annual income of ${ANNUAL_INCOME:,.0f} and a stated USDA income limit of "
            f"{_safe(extraction.usda_income_limit_2026)}, the buyer's income is at or below the limit (i.e., qualifies)."
        ),
        node=usda_elig_leaf,
        additional_instruction=(
            "Apply straightforward comparison logic based on the numbers. Allow for typical formatting (commas, currency)."
        )
    )

    # 8–9) DTI Requirements (parallel)
    dti_par = evaluator.add_parallel(
        id="DTI_Requirements",
        desc="Compute the buyer's DTI ratio from the provided inputs and assess FHA and Conventional DTI compliance (including stating each program's DTI limit).",
        parent=analysis,
        critical=True
    )
    # 8) Buyer DTI ratio (leaf)
    buyer_dti_leaf = evaluator.add_leaf(
        id="Buyer_DTI_Ratio",
        desc="State the buyer's calculated DTI ratio using the provided DTI formula and the problem's given income and monthly obligations (including housing payment + other monthly debts as applicable to the stated calculation).",
        parent=dti_par,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The answer states the buyer's DTI ratio as {_safe(extraction.buyer_dti_ratio)}. "
            f"Using (${MONTHLY_HOUSING_PAYMENT:,.0f} housing + ${MONTHLY_OTHER_DEBT:,.0f} other debt) / "
            f"${GROSS_MONTHLY_INCOME:,.2f} gross monthly income ≈ {fmt_percent_decimal(DTI_DECIMAL)}. "
            "These should match within ordinary rounding tolerance."
        ),
        node=buyer_dti_leaf,
        additional_instruction=ROUNDING_TOLERANCE_INSTRUCTION
    )

    # 8) FHA DTI Compliance (sequential)
    fha_seq = evaluator.add_sequential(
        id="FHA_DTI_Compliance",
        desc="State the FHA DTI limit and whether the buyer's calculated DTI ratio meets it.",
        parent=dti_par,
        critical=True
    )
    # 8.1) State FHA Max (by URLs)
    fha_limit_leaf = evaluator.add_leaf(
        id="State_FHA_Max_DTI_Limit",
        desc="State the FHA maximum DTI limit (per constraints).",
        parent=fha_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"The FHA maximum DTI limit is stated as {_safe(extraction.fha_max_dti_limit)}.",
        node=fha_limit_leaf,
        sources=extraction.fha_dti_sources,
        additional_instruction="Verify the page states the FHA maximum total DTI limit. Minor wording/formatting differences are fine."
    )
    # 8.2) Determine FHA compliance (simple logic)
    fha_comp_leaf = evaluator.add_leaf(
        id="Determine_FHA_DTI_Compliance",
        desc="Determine whether the buyer's calculated DTI ratio is at or below the FHA DTI limit.",
        parent=fha_seq,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"Given the buyer's DTI of approximately {fmt_percent_decimal(DTI_DECIMAL)} and an FHA maximum DTI of "
            f"{_safe(extraction.fha_max_dti_limit)}, the buyer's DTI is at or below the FHA limit."
        ),
        node=fha_comp_leaf,
        additional_instruction="Treat this as a simple comparison; allow rounding differences."
    )

    # 9) Conventional DTI Compliance (sequential)
    conv_seq = evaluator.add_sequential(
        id="Conventional_DTI_Compliance",
        desc="State the Conventional DTI limit and whether the buyer's calculated DTI ratio meets it.",
        parent=dti_par,
        critical=True
    )
    # 9.1) State Conventional Max (by URLs)
    conv_limit_leaf = evaluator.add_leaf(
        id="State_Conventional_Max_DTI_Limit",
        desc="State the Conventional maximum DTI limit (per constraints).",
        parent=conv_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Conventional maximum DTI limit is stated as {_safe(extraction.conventional_max_dti_limit)}.",
        node=conv_limit_leaf,
        sources=extraction.conventional_dti_sources,
        additional_instruction="Verify the page states the Conventional (conforming) maximum total DTI limit."
    )
    # 9.2) Determine Conventional compliance (simple logic)
    conv_comp_leaf = evaluator.add_leaf(
        id="Determine_Conventional_DTI_Compliance",
        desc="Determine whether the buyer's calculated DTI ratio is at or below the Conventional DTI limit.",
        parent=conv_seq,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"Given the buyer's DTI of approximately {fmt_percent_decimal(DTI_DECIMAL)} and a Conventional maximum DTI of "
            f"{_safe(extraction.conventional_max_dti_limit)}, the buyer's DTI is at or below the Conventional limit."
        ),
        node=conv_comp_leaf,
        additional_instruction="Treat this as a simple comparison; allow rounding differences."
    )

    # 10) Capital Gains Tax Rate (leaf, verify by URLs if available)
    ltcg_leaf = evaluator.add_leaf(
        id="Capital_Gains_Tax_Rate",
        desc="State the applicable long-term capital gains tax rate (0%, 15%, or 20%) for the single filer based on the provided thresholds.",
        parent=analysis,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"For a single filer with income of ${ANNUAL_INCOME:,.0f}, the applicable long-term capital gains tax rate is "
            f"{_safe(extraction.ltcg_tax_rate)}."
        ),
        node=ltcg_leaf,
        sources=extraction.ltcg_sources,
        additional_instruction=(
            "Verify via the provided source(s) that $95,000 falls into the stated long-term capital gains bracket for 2026 or the relevant tax year. "
            "If the source provides thresholds, confirm that the quoted rate matches those thresholds."
        )
    )

    # 11) SALT deduction cap (leaf, verify by URLs)
    salt_leaf = evaluator.add_leaf(
        id="SALT_Deduction_Limit",
        desc="State the SALT deduction cap applicable for tax years 2025–2026 (per constraints).",
        parent=analysis,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The SALT deduction cap applicable for tax years 2025–2026 is stated as {_safe(extraction.salt_cap_2025_2026)}."
        ),
        node=salt_leaf,
        sources=extraction.salt_sources,
        additional_instruction="Verify the source explicitly states the SALT cap applicable for 2025–2026."
    )

    # 12) Residential Rental Depreciation (leaf, verify by URLs)
    rental_dep_leaf = evaluator.add_leaf(
        id="Residential_Rental_Depreciation",
        desc="State the depreciation rule for residential rental property as specified in the constraints (depreciation period and any required method detail).",
        parent=analysis,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The answer states residential rental property is depreciated over {_safe(extraction.rental_depreciation_period)} "
            f"using {_safe(extraction.rental_depreciation_method)}."
        ),
        node=rental_dep_leaf,
        sources=extraction.rental_depreciation_sources,
        additional_instruction=(
            "Verify that the source states the residential rental depreciation period (e.g., 27.5 years) and method (e.g., straight-line under MACRS)."
        )
    )

    # 13) PMI removal threshold loan balance (leaf, simple arithmetic)
    pmi_remove_leaf = evaluator.add_leaf(
        id="PMI_Removal_Threshold",
        desc="State the loan balance amount (in dollars) at which PMI is automatically removed based on the 78% of original property value threshold.",
        parent=analysis,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The answer states the PMI automatic removal balance as {_safe(extraction.pmi_auto_remove_balance)}. "
            f"Using 78% of the original value ${PURCHASE_PRICE:,.0f}, the threshold balance is approximately "
            f"{fmt_money(PMI_AUTO_REMOVE_BAL_EXPECTED)}. "
            "These should match allowing minor rounding/formatting differences."
        ),
        node=pmi_remove_leaf,
        additional_instruction=ROUNDING_TOLERANCE_INSTRUCTION
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
    Evaluate an answer for the 2026 home purchase financial analysis task.
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

    # Record ground truth/expected computations to help interpret results
    evaluator.add_ground_truth(EXPECTED_VALUES, gt_type="expected_computations")

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_home_purchase(),
        template_class=HomePurchaseExtraction,
        extraction_name="home_purchase_extraction"
    )

    # Build verification tree and run checks
    await build_and_verify(evaluator, extraction)

    # Return summarized results
    return evaluator.get_summary()