import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "commercial_reit_acquisition_va_fairfax_2026_03"
TASK_DESCRIPTION = (
    "A Real Estate Investment Trust (REIT) is planning to acquire a commercial office building in Fairfax County, "
    "Virginia, with a purchase price of $3,500,000. The REIT will finance 75% of the purchase price with a "
    "commercial mortgage at current market rates (March 2026). As the REIT's financial analyst, prepare a complete "
    "acquisition cost analysis that includes: (1) Detailed calculation of ALL buyer closing costs broken down into "
    "individual line items with estimated amounts, (2) Financing terms including loan amount and expected interest "
    "rate range based on March 2026 commercial mortgage rates, (3) Required due diligence assessments for a commercial "
    "property of this size and type, and (4) Verification that the REIT structure meets all four primary IRS "
    "qualification tests. Provide specific dollar amounts or percentages for each cost component, and include "
    "reference URLs supporting your data."
)

PURCHASE_PRICE = 3_500_000
EXPECTED_LOAN_AMOUNT = 0.75 * PURCHASE_PRICE  # 2,625,000
EXPECTED_DOWN_PAYMENT = 0.25 * PURCHASE_PRICE  # 875,000

# For closing costs reasonableness
CLOSING_COST_MIN_PCT = 2.0
CLOSING_COST_MAX_PCT = 6.0

# Category labels for extraction and validation
CATEGORY_LENDER = "lender_financing_fees"
CATEGORY_TITLE = "title_settlement_services"
CATEGORY_GOVT = "government_taxes_fees"
CATEGORY_THIRD_PARTY = "third_party_reports_inspections"
CATEGORY_ESCROWS = "escrows_prorations_reserves"
REQUIRED_CATEGORIES = [
    CATEGORY_LENDER,
    CATEGORY_TITLE,
    CATEGORY_GOVT,
    CATEGORY_THIRD_PARTY,
    CATEGORY_ESCROWS,
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FinancingInfo(BaseModel):
    loan_amount: Optional[str] = None
    down_payment: Optional[str] = None
    interest_rate_range: Optional[str] = None  # e.g., "6.5% - 7.5%"
    rate_type: Optional[str] = None            # e.g., "30-year fixed commercial"
    time_reference: Optional[str] = None       # e.g., "March 2026"
    references: List[str] = Field(default_factory=list)


class ClosingCostItem(BaseModel):
    category: Optional[str] = None  # Must be one of REQUIRED_CATEGORIES
    label: Optional[str] = None
    amount: Optional[str] = None  # e.g., "$12,500" or "$0.00"
    percent_of_price: Optional[str] = None  # e.g., "0.5%"
    source_urls: List[str] = Field(default_factory=list)


class ClosingCostsInfo(BaseModel):
    items: List[ClosingCostItem] = Field(default_factory=list)
    total_amount: Optional[str] = None          # single stated total, if provided
    total_low: Optional[str] = None             # range low
    total_high: Optional[str] = None            # range high
    total_percent: Optional[str] = None         # e.g., "3.2%"
    total_percent_low: Optional[str] = None     # e.g., "2.5%"
    total_percent_high: Optional[str] = None    # e.g., "4.0%"
    references: List[str] = Field(default_factory=list)


class DueDiligenceInfo(BaseModel):
    items: List[str] = Field(default_factory=list)
    phase_i_esa_included: Optional[bool] = None
    professional_appraisal_included: Optional[bool] = None
    ada_compliance_assessment_included: Optional[bool] = None
    references: List[str] = Field(default_factory=list)


class REITTestsInfo(BaseModel):
    distribution_90_percent: Optional[bool] = None
    shareholder_100_335_days: Optional[bool] = None
    income_95_percent_qualifying: Optional[bool] = None
    income_75_percent_real_estate: Optional[bool] = None
    references: List[str] = Field(default_factory=list)


class AcquisitionAnalysisExtraction(BaseModel):
    financing: FinancingInfo = FinancingInfo()
    closing_costs: ClosingCostsInfo = ClosingCostsInfo()
    due_diligence: DueDiligenceInfo = DueDiligenceInfo()
    reit_tests: REITTestsInfo = REITTestsInfo()


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_acquisition_analysis() -> str:
    return f"""
You will extract structured information from the provided answer related to this task:
- A REIT acquires a $3,500,000 commercial office property in Fairfax County, VA.
- Financing at 75% LTV with current market rates as of March 2026.
- Provide detailed buyer closing costs, financing terms, due diligence, and IRS REIT tests, with reference URLs.

OUTPUT RULES:
- Return JSON exactly following the specified schema fields below.
- Extract values exactly as they appear in the answer. Do not invent data.
- For any missing field, return null (for scalars) or [] (for lists).
- For all URLs, include full URLs with protocol (http/https).
- Normalize categories to the exact slugs provided.

SCHEMA:
- financing:
  - loan_amount: string currency as in answer (e.g., "$2,625,000", "2.625M USD"). If a range is given, include the main figure used in calculations.
  - down_payment: string currency (e.g., "$875,000").
  - interest_rate_range: string (e.g., "6.5%–7.5%" or "6.5%-7.5%").
  - rate_type: string describing the product, explicitly if present (e.g., "30-year fixed commercial").
  - time_reference: string indicating timing (should include "March 2026" or equivalent if present).
  - references: list of URL strings specifically cited to support the March 2026 commercial mortgage rate range.

- closing_costs:
  - items: array of objects. For each line item include:
    - category: one of the EXACT slugs:
        "{CATEGORY_LENDER}", "{CATEGORY_TITLE}", "{CATEGORY_GOVT}", "{CATEGORY_THIRD_PARTY}", "{CATEGORY_ESCROWS}"
    - label: short name of the line item (e.g., "Origination fee (1 point)")
    - amount: string currency as in answer (e.g., "$12,500"), if present; else null
    - percent_of_price: string percent as in answer (e.g., "0.5%"), if present; else null
    - source_urls: list of URL strings for this line item, if any were cited inline.
  - total_amount: string currency total if a single total is stated; else null
  - total_low: string currency lower bound if a range total is stated; else null
  - total_high: string currency upper bound if a range total is stated; else null
  - total_percent: string percent total if a single percent is stated; else null
  - total_percent_low: string percent lower bound of total percent range if stated; else null
  - total_percent_high: string percent upper bound of total percent range if stated; else null
  - references: list of URL strings supporting typical commercial closing cost components and/or typical total ranges (e.g., "2–6%").

- due_diligence:
  - items: list of strings (verbatim items mentioned).
  - phase_i_esa_included: boolean indicating whether "Phase I Environmental Site Assessment (ESA)" is explicitly included.
  - professional_appraisal_included: boolean indicating whether a professional appraisal is explicitly included.
  - ada_compliance_assessment_included: boolean indicating whether ADA compliance assessment is explicitly included.
  - references: list of URL strings citing standards/requirements for ESA/appraisals/ADA checks.

- reit_tests:
  - distribution_90_percent: boolean indicating whether the answer states the 90% distribution test.
  - shareholder_100_335_days: boolean indicating whether the answer states "at least 100 shareholders for at least 335 days of the taxable year".
  - income_95_percent_qualifying: boolean indicating whether the answer states the 95% gross income test.
  - income_75_percent_real_estate: boolean indicating whether the answer states the 75% gross income test.
  - references: list of URL strings (preferably IRS/Code sections) supporting these tests.

Ensure categories use the exact slugs above, and extract every reference URL that the answer provides for each section.
"""


# --------------------------------------------------------------------------- #
# Helper parsing utilities                                                    #
# --------------------------------------------------------------------------- #
_money_re = re.compile(r"([-+]?\d[\d,]*(?:\.\d+)?)")
_percent_re = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*%")

def parse_money_to_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    txt = s.strip().lower()
    if not txt:
        return None
    # Handle million notation
    if "million" in txt or "m" in txt:
        # Try to capture explicit "X million"
        m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:m|million)", txt)
        if m:
            try:
                return float(m.group(1)) * 1_000_000.0
            except:
                pass
        # Fallback to digits
    # Strip currency symbols/commas and parse first number
    m2 = _money_re.search(txt.replace(",", ""))
    if m2:
        try:
            return float(m2.group(1))
        except:
            return None
    return None


def parse_percent_to_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = _percent_re.search(s)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    return None


def in_range_overlaps(low: float, high: float, rng_low: float, rng_high: float) -> bool:
    return not (high < rng_low or low > rng_high)


def compute_closing_cost_percent_candidates(cc: ClosingCostsInfo) -> Dict[str, Optional[float]]:
    """
    Compute best-effort percent candidates from extracted totals relative to PURCHASE_PRICE.
    Returns a dict including:
      - amount_single_pct
      - amount_low_pct
      - amount_high_pct
      - percent_single
      - percent_low
      - percent_high
    """
    out: Dict[str, Optional[float]] = {
        "amount_single_pct": None,
        "amount_low_pct": None,
        "amount_high_pct": None,
        "percent_single": None,
        "percent_low": None,
        "percent_high": None,
    }

    # From amounts
    amt_single = parse_money_to_float(cc.total_amount)
    if amt_single and PURCHASE_PRICE > 0:
        out["amount_single_pct"] = 100.0 * amt_single / PURCHASE_PRICE

    amt_low = parse_money_to_float(cc.total_low)
    amt_high = parse_money_to_float(cc.total_high)
    if amt_low and PURCHASE_PRICE > 0:
        out["amount_low_pct"] = 100.0 * amt_low / PURCHASE_PRICE
    if amt_high and PURCHASE_PRICE > 0:
        out["amount_high_pct"] = 100.0 * amt_high / PURCHASE_PRICE

    # From percents stated directly
    out["percent_single"] = parse_percent_to_float(cc.total_percent)
    out["percent_low"] = parse_percent_to_float(cc.total_percent_low)
    out["percent_high"] = parse_percent_to_float(cc.total_percent_high)

    return out


def coverage_check_for_five_categories(items: List[ClosingCostItem]) -> bool:
    """
    True if we have at least one quantified (amount or percent) line item in each required category.
    """
    present: Dict[str, bool] = {c: False for c in REQUIRED_CATEGORIES}
    for it in items:
        if not it or not it.category:
            continue
        if it.category not in present:
            continue
        quantified = (it.amount is not None and str(it.amount).strip() != "") or \
                     (it.percent_of_price is not None and str(it.percent_of_price).strip() != "")
        if quantified:
            present[it.category] = True
    return all(present.values())


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_financing_terms(evaluator: Evaluator, parent_node, data: AcquisitionAnalysisExtraction) -> None:
    """
    Build and verify the Financing_Terms subtree.
    """
    fin_node = evaluator.add_parallel(
        id="Financing_Terms",
        desc="Financing terms based on 75% LTV and March 2026 market rates for 30-year fixed commercial mortgages.",
        parent=parent_node,
        critical=True
    )

    # 1) Loan_Amount_And_Down_Payment_Calculation (leaf)
    loan_down_leaf = evaluator.add_leaf(
        id="Loan_Amount_And_Down_Payment_Calculation",
        desc="Correctly calculates loan amount (75% of $3,500,000) and down payment (25% of $3,500,000).",
        parent=fin_node,
        critical=True
    )
    loan_str = data.financing.loan_amount or ""
    down_str = data.financing.down_payment or ""
    claim_calc = (
        f"The answer correctly calculates the loan amount as {loan_str} (75% of $3,500,000) and the "
        f"down payment/equity as {down_str} (25% of $3,500,000)."
    )
    await evaluator.verify(
        claim=claim_calc,
        node=loan_down_leaf,
        additional_instruction=(
            "Check the answer text: confirm it shows loan = 75% of $3,500,000 (~$2,625,000) and down payment = 25% "
            "(~$875,000). Accept minor rounding and formatting variants like '2.625M' or '$2,625,000', "
            "and '0.875M' or '$875,000'."
        ),
    )

    # 2) Interest_Rate_Range_March_2026_30yr_Fixed_Commercial (leaf)
    rate_leaf = evaluator.add_leaf(
        id="Interest_Rate_Range_March_2026_30yr_Fixed_Commercial",
        desc="Provides an expected interest-rate range explicitly tied to March 2026 market data and explicitly for 30-year fixed commercial mortgages.",
        parent=fin_node,
        critical=True
    )
    ir_range = data.financing.interest_rate_range or ""
    rate_type = data.financing.rate_type or ""
    time_ref = data.financing.time_reference or ""
    claim_rate = (
        f"The answer provides an expected interest-rate range '{ir_range}' explicitly tied to March 2026 and "
        f"explicitly for 30-year fixed commercial mortgages (rate type noted as '{rate_type}', time reference '{time_ref}')."
    )
    await evaluator.verify(
        claim=claim_rate,
        node=rate_leaf,
        additional_instruction=(
            "Verify the answer explicitly anchors the rate range to March 2026, includes percent values (e.g., X%–Y%), "
            "and states that the product is 30-year fixed commercial. Accept minor formatting like 'Mar 2026'."
        ),
    )

    # 3) Financing_References (leaf - verify by cited URLs)
    fin_ref_leaf = evaluator.add_leaf(
        id="Financing_References",
        desc="Includes at least one reference URL supporting the March 2026 commercial mortgage rate range used.",
        parent=fin_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "This source provides commercial mortgage rate information or benchmarks around March 2026 (e.g., 'current rates' "
            "in March 2026) that can justify an expected range for 30-year fixed commercial mortgages or similar CRE loan products."
        ),
        node=fin_ref_leaf,
        sources=data.financing.references,
        additional_instruction=(
            "PASS if at least one URL clearly presents commercial mortgage/CRE rate data or benchmarks near March 2026. "
            "It can include SBA 504/7(a) or CMBS rates if clearly applicable to CRE lending benchmarks. "
            "FAIL if no relevant rate information or only residential mortgage content."
        ),
    )


async def verify_closing_costs(evaluator: Evaluator, parent_node, data: AcquisitionAnalysisExtraction) -> None:
    """
    Build and verify the Buyer_Closing_Costs subtree.
    """
    cc_node = evaluator.add_parallel(
        id="Buyer_Closing_Costs",
        desc="Buyer closing-cost analysis with quantified components, totals, 2–6% reasonableness check, and references.",
        parent=parent_node,
        critical=True
    )

    # 1) Standard_Closing_Cost_Component_Coverage_Quantified_Max5 (leaf via custom boolean)
    coverage_ok = coverage_check_for_five_categories(data.closing_costs.items)
    coverage_leaf = evaluator.add_custom_node(
        result=coverage_ok,
        id="Standard_Closing_Cost_Component_Coverage_Quantified_Max5",
        desc="Covers all standard buyer closing cost components by providing at least one distinct line item WITH a specific estimated dollar amount or percentage in each of the five categories.",
        parent=cc_node,
        critical=True
    )

    # 2) Closing_Cost_Total_Computed (leaf - simple verify presence and derivation claim)
    total_leaf = evaluator.add_leaf(
        id="Closing_Cost_Total_Computed",
        desc="Computes and reports a total buyer closing-cost estimate (or clearly stated total range) derived from the listed component amounts.",
        parent=cc_node,
        critical=True
    )
    total_parts = []
    if data.closing_costs.total_amount:
        total_parts.append(f"total amount {data.closing_costs.total_amount}")
    if data.closing_costs.total_low and data.closing_costs.total_high:
        total_parts.append(f"range {data.closing_costs.total_low}–{data.closing_costs.total_high}")
    if data.closing_costs.total_percent:
        total_parts.append(f"total percent {data.closing_costs.total_percent}")
    if data.closing_costs.total_percent_low and data.closing_costs.total_percent_high:
        total_parts.append(f"percent range {data.closing_costs.total_percent_low}–{data.closing_costs.total_percent_high}")
    total_desc = "; ".join(total_parts) if total_parts else "no total stated"

    claim_total = (
        f"The answer reports an overall buyer closing-cost total ({total_desc}) that is presented as a sum or range "
        "derived from the listed component line items."
    )
    await evaluator.verify(
        claim=claim_total,
        node=total_leaf,
        additional_instruction=(
            "PASS if the answer clearly shows a total (single number) or a total range, and it is logically presented as the sum "
            "or aggregation of the listed components. The total may be shown as a currency figure and/or as a percentage of price."
        ),
    )

    # 3) Total_Within_2_to_6_Percent_Constraint (leaf via custom boolean using parsed candidates)
    pct_candidates = compute_closing_cost_percent_candidates(data.closing_costs)

    # Build an overlap test:
    # If we have both low and high percents (by amount or by direct percent), check range overlap with [2,6].
    # Else if we only have single percent, ensure it's within 2..6.
    within_ok = False

    # Check ranges by amount:
    if pct_candidates["amount_low_pct"] is not None and pct_candidates["amount_high_pct"] is not None:
        within_ok = in_range_overlaps(pct_candidates["amount_low_pct"], pct_candidates["amount_high_pct"],
                                      CLOSING_COST_MIN_PCT, CLOSING_COST_MAX_PCT)

    # Check ranges by percent fields:
    if not within_ok and pct_candidates["percent_low"] is not None and pct_candidates["percent_high"] is not None:
        within_ok = in_range_overlaps(pct_candidates["percent_low"], pct_candidates["percent_high"],
                                      CLOSING_COST_MIN_PCT, CLOSING_COST_MAX_PCT)

    # Check single percent by amount:
    if not within_ok and pct_candidates["amount_single_pct"] is not None:
        v = pct_candidates["amount_single_pct"]
        within_ok = (CLOSING_COST_MIN_PCT <= v <= CLOSING_COST_MAX_PCT)

    # Check single direct percent:
    if not within_ok and pct_candidates["percent_single"] is not None:
        v = pct_candidates["percent_single"]
        within_ok = (CLOSING_COST_MIN_PCT <= v <= CLOSING_COST_MAX_PCT)

    # Record calculation info for transparency
    evaluator.add_custom_info(
        info={
            "closing_cost_percent_candidates": pct_candidates,
            "purchase_price": PURCHASE_PRICE,
            "min_pct": CLOSING_COST_MIN_PCT,
            "max_pct": CLOSING_COST_MAX_PCT,
        },
        info_type="calc",
        info_name="closing_costs_percent_check"
    )

    within_leaf = evaluator.add_custom_node(
        result=within_ok,
        id="Total_Within_2_to_6_Percent_Constraint",
        desc="Verifies that the computed total closing costs are within the typical 2–6% range of the $3,500,000 purchase price.",
        parent=cc_node,
        critical=True
    )

    # 4) Closing_Cost_References (leaf - verify by URLs)
    cc_ref_leaf = evaluator.add_leaf(
        id="Closing_Cost_References",
        desc="Includes at least one reference URL supporting typical buyer closing cost components and/or typical closing cost ranges.",
        parent=cc_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "This source describes typical commercial real estate buyer closing cost components and/or states typical total "
            "closing cost ranges (for example around 2–6% of the purchase price)."
        ),
        node=cc_ref_leaf,
        sources=data.closing_costs.references,
        additional_instruction=(
            "PASS if the page provides credible information about commercial closing cost components or typical total ranges. "
            "General CRE references, lender or title/escrow industry pages are acceptable if relevant."
        ),
    )


async def verify_due_diligence(evaluator: Evaluator, parent_node, data: AcquisitionAnalysisExtraction) -> None:
    """
    Build and verify the Due_Diligence_Assessments subtree.
    """
    dd_node = evaluator.add_parallel(
        id="Due_Diligence_Assessments",
        desc="Required due diligence assessments for a commercial office property of this size/type, including the explicitly constrained minimum set.",
        parent=parent_node,
        critical=True
    )

    # 1) Phase I ESA explicitly included (leaf)
    phase_leaf = evaluator.add_leaf(
        id="Phase_I_ESA_Included",
        desc="Explicitly includes a Phase I Environmental Site Assessment (ESA) as due diligence.",
        parent=dd_node,
        critical=True
    )
    await evaluator.verify(
        claim="The due diligence list explicitly includes a Phase I Environmental Site Assessment (ESA).",
        node=phase_leaf,
        additional_instruction="Check the answer text and PASS only if 'Phase I ESA' or an equivalent explicit phrase is present."
    )

    # 2) Professional Appraisal included (leaf)
    app_leaf = evaluator.add_leaf(
        id="Professional_Appraisal_Included",
        desc="Explicitly includes a professional appraisal (required by constraints for commercial transactions over $500,000).",
        parent=dd_node,
        critical=True
    )
    await evaluator.verify(
        claim="The due diligence list explicitly includes a professional appraisal for the property.",
        node=app_leaf,
        additional_instruction="PASS only if the answer clearly lists a professional appraisal."
    )

    # 3) ADA Compliance assessment included (leaf)
    ada_leaf = evaluator.add_leaf(
        id="ADA_Compliance_Assessment_Included",
        desc="Explicitly includes an ADA compliance assessment (required by constraints).",
        parent=dd_node,
        critical=True
    )
    await evaluator.verify(
        claim="The due diligence list explicitly includes an ADA compliance assessment.",
        node=ada_leaf,
        additional_instruction="PASS only if ADA compliance assessment is explicitly included."
    )

    # 4) Due_Diligence_References (leaf - verify by URLs)
    dd_ref_leaf = evaluator.add_leaf(
        id="Due_Diligence_References",
        desc="Includes at least one reference URL supporting the due diligence requirements/standards referenced (ESA, appraisal requirement, ADA compliance assessment).",
        parent=dd_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "This source describes standards or requirements relevant to Phase I ESA (e.g., ASTM E1527), commercial property "
            "appraisals, or ADA compliance assessments for buildings."
        ),
        node=dd_ref_leaf,
        sources=data.due_diligence.references,
        additional_instruction=(
            "PASS if at least one URL provides credible guidance/standards for ESA/appraisals/ADA compliance in commercial contexts."
        ),
    )


async def verify_reit_tests(evaluator: Evaluator, parent_node, data: AcquisitionAnalysisExtraction) -> None:
    """
    Build and verify the REIT_IRS_Qualification_Tests subtree.
    """
    reit_node = evaluator.add_parallel(
        id="REIT_IRS_Qualification_Tests",
        desc="Verification that the REIT structure meets all four IRS qualification tests specified in constraints, supported by references.",
        parent=parent_node,
        critical=True
    )

    # 1) Distribution test 90%
    dist_leaf = evaluator.add_leaf(
        id="Distribution_Test_90_Percent",
        desc="Verifies the REIT distribution requirement: at least 90% of taxable income distributed annually.",
        parent=reit_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states the REIT 90% distribution requirement (at least 90% of taxable income distributed annually).",
        node=dist_leaf,
        additional_instruction="PASS only if the answer clearly mentions the 90% distribution requirement."
    )

    # 2) Shareholder test 100/335
    sh_leaf = evaluator.add_leaf(
        id="Shareholder_Test_100_and_335_Days",
        desc="Verifies the shareholder requirement: at least 100 shareholders for at least 335 days of the taxable year.",
        parent=reit_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states the shareholder requirement: at least 100 shareholders for at least 335 days of the taxable year.",
        node=sh_leaf,
        additional_instruction="PASS only if both '100 shareholders' and '335 days' (or equivalent description) are present."
    )

    # 3) Income test 95%
    inc95_leaf = evaluator.add_leaf(
        id="Income_Test_95_Percent_Qualifying",
        desc="Verifies the 95% gross income test from qualifying sources.",
        parent=reit_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states the REIT 95% gross income test from qualifying sources.",
        node=inc95_leaf,
        additional_instruction="PASS only if the 95% gross income test is stated clearly."
    )

    # 4) Income test 75%
    inc75_leaf = evaluator.add_leaf(
        id="Income_Test_75_Percent_Real_Estate",
        desc="Verifies the 75% gross income test from real estate-related sources.",
        parent=reit_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states the REIT 75% gross income test from real estate-related sources.",
        node=inc75_leaf,
        additional_instruction="PASS only if the 75% gross income test is stated clearly."
    )

    # 5) REIT qualification references by URL(s)
    reit_ref_leaf = evaluator.add_leaf(
        id="REIT_Qualification_References",
        desc="Includes at least one reference URL supporting the four REIT qualification tests (preferably IRS/IRC sources).",
        parent=reit_node,
        critical=True
    )
    await evaluator.verify(
        claim="This source states one or more of the REIT qualification tests: 90% distribution, 100 shareholders/335 days, 95% gross income, 75% gross income.",
        node=reit_ref_leaf,
        sources=data.reit_tests.references,
        additional_instruction=(
            "Prefer IRS/Code sections or reputable summaries that clearly state these tests. PASS if at least one URL explicitly "
            "describes these REIT qualification tests."
        ),
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
    Evaluate an answer for the commercial REIT acquisition analysis task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator
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

    # Create top-level critical node reflecting the rubric root
    main_node = evaluator.add_parallel(
        id="Commercial_Property_Acquisition_Analysis",
        desc="Complete acquisition cost analysis for a REIT purchasing a $3,500,000 commercial office building in Fairfax County, Virginia, including closing costs, financing terms (March 2026), due diligence, and REIT IRS qualification tests, with supporting reference URLs.",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_acquisition_analysis(),
        template_class=AcquisitionAnalysisExtraction,
        extraction_name="acquisition_analysis"
    )

    # Add key ground-truth constants for transparency
    evaluator.add_ground_truth({
        "purchase_price": PURCHASE_PRICE,
        "expected_loan_amount_75pct": EXPECTED_LOAN_AMOUNT,
        "expected_down_payment_25pct": EXPECTED_DOWN_PAYMENT,
        "closing_cost_reasonable_range_percent": [CLOSING_COST_MIN_PCT, CLOSING_COST_MAX_PCT],
        "rate_time_reference_required": "March 2026",
        "required_closing_cost_categories": REQUIRED_CATEGORIES
    })

    # Build and verify each major rubric section
    await verify_financing_terms(evaluator, main_node, extracted)
    await verify_closing_costs(evaluator, main_node, extracted)
    await verify_due_diligence(evaluator, main_node, extracted)
    await verify_reit_tests(evaluator, main_node, extracted)

    # Return summary
    return evaluator.get_summary()