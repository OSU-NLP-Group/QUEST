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
TASK_ID = "loan_analysis_ca_615_450k"
TASK_DESCRIPTION = (
    "A first-time homebuyer in California has a credit score of 615 and wants to purchase a home priced at $450,000. "
    "They have saved $22,500 for a down payment, which is 5% of the purchase price.\n\n"
    "Determine the following:\n\n"
    "1. Loan Eligibility: Based on standard credit score requirements, identify whether this borrower qualifies for:\n"
    "- An FHA loan with 3.5% down payment\n"
    "- A conventional loan with 5% down payment\n\n"
    "For each loan type, state whether they are eligible or not eligible, and explain why based on the minimum credit score requirements.\n\n"
    "2. Mortgage Insurance Costs: For any loan type(s) the borrower qualifies for, calculate or describe:\n"
    "- For FHA loans: The upfront Mortgage Insurance Premium (MIP) amount as a percentage and dollar amount, and whether annual MIP can be removed\n"
    "- For conventional loans: Whether Private Mortgage Insurance (PMI) would be required, and at what loan-to-value ratio it can be removed based on the original property value\n\n"
    "3. Recommendation: Based on your eligibility assessment and mortgage insurance analysis, which loan type would you recommend for this borrower and why? "
    "Consider the long-term implications of mortgage insurance requirements in your recommendation."
)

# Scenario constants
PURCHASE_PRICE = 450_000.0
DOWN_PAYMENT_DOLLARS = 22_500.0  # 5% of purchase price
DOWN_PAYMENT_PCT = DOWN_PAYMENT_DOLLARS / PURCHASE_PRICE  # 0.05
CREDIT_SCORE = 615

# Standard eligibility thresholds (industry norms)
FHA_MIN_SCORE_3_5_DOWN = 580
FHA_MIN_SCORE_10_DOWN = 500
CONV_MIN_SCORE_5_DOWN = 620

# Derived expected eligibility
EXPECTED_FHA_ELIGIBLE = CREDIT_SCORE >= FHA_MIN_SCORE_3_5_DOWN  # True for 615
EXPECTED_CONV_ELIGIBLE = CREDIT_SCORE >= CONV_MIN_SCORE_5_DOWN  # False for 615

# FHA loan amount options (commonly referenced scenarios)
FHA_BASE_LOAN_3_5_DOWN = PURCHASE_PRICE * (1.0 - 0.035)  # 96.5% LTV base
FHA_BASE_LOAN_5_DOWN = PURCHASE_PRICE * (1.0 - 0.05)     # 95% LTV base
UFMIP_RATE = 0.0175
EXPECTED_UFMIP_3_5_DOWN = FHA_BASE_LOAN_3_5_DOWN * UFMIP_RATE  # ≈ $7,599.38
EXPECTED_UFMIP_5_DOWN = FHA_BASE_LOAN_5_DOWN * UFMIP_RATE      # ≈ $7,481.25
# Acceptable numeric tolerance band for UFMIP dollar check (to allow rounding)
UFMIP_ACCEPT_MIN = min(EXPECTED_UFMIP_3_5_DOWN, EXPECTED_UFMIP_5_DOWN) - 75.0
UFMIP_ACCEPT_MAX = max(EXPECTED_UFMIP_3_5_DOWN, EXPECTED_UFMIP_5_DOWN) + 75.0

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EligibilityExtraction(BaseModel):
    # FHA eligibility assessment and reasoning based on the answer
    fha_status: Optional[str] = None  # e.g., "eligible", "not eligible", "eligible for 3.5% down"
    fha_reason: Optional[str] = None  # explanation text
    fha_min_score_cited: Optional[str] = None  # any cited minimum score for FHA (e.g., "580", "500")

    # Conventional eligibility assessment and reasoning based on the answer
    conventional_status: Optional[str] = None  # e.g., "eligible", "not eligible"
    conventional_reason: Optional[str] = None  # explanation text
    conventional_min_score_cited: Optional[str] = None  # cited minimum score for conventional (e.g., "620")


class InsuranceExtraction(BaseModel):
    # FHA insurance details
    fha_upfront_mip_percent: Optional[str] = None  # e.g., "1.75%"
    fha_upfront_mip_dollar: Optional[str] = None   # e.g., "$7,599"
    fha_annual_mip_removal_statement: Optional[str] = None  # e.g., "cannot be removed with <10% down"

    # Conventional PMI details
    conventional_pmi_required_statement: Optional[str] = None  # e.g., "PMI required with <20% down"
    conventional_pmi_removal_ltv_statement: Optional[str] = None  # e.g., "removed at 80% LTV"


class RecommendationExtraction(BaseModel):
    recommended_loan_type: Optional[str] = None  # e.g., "FHA", "Conventional", "Wait/Improve credit"
    recommendation_reason: Optional[str] = None  # explanation text


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_eligibility() -> str:
    return (
        "Extract the loan eligibility determinations and their explanations from the answer. "
        "Return:\n"
        "- fha_status: Whether the borrower is eligible for an FHA loan with 3.5% down (use 'eligible' or 'not eligible'; "
        "you may include qualifiers like 'eligible for 3.5% down').\n"
        "- fha_reason: The explanation, ideally citing minimum credit score requirements (e.g., 580 for 3.5% down).\n"
        "- fha_min_score_cited: The minimum score cited for FHA in the explanation (e.g., '580', '500'); if multiple, include the most relevant.\n"
        "- conventional_status: Whether the borrower is eligible for a conventional loan with 5% down (use 'eligible' or 'not eligible').\n"
        "- conventional_reason: The explanation, ideally citing minimum credit score requirements (e.g., 620).\n"
        "- conventional_min_score_cited: The minimum score cited for conventional in the explanation (e.g., '620').\n"
        "If a field is not present in the answer, return null."
    )


def prompt_extract_insurance() -> str:
    return (
        "Extract the mortgage insurance details described in the answer. Return:\n"
        "- fha_upfront_mip_percent: The FHA upfront MIP percentage (e.g., '1.75%').\n"
        "- fha_upfront_mip_dollar: The upfront MIP dollar amount, as stated (e.g., '$7,599').\n"
        "- fha_annual_mip_removal_statement: The statement about annual MIP removal for FHA "
        "(e.g., 'cannot be removed with <10% down', 'removable after 11 years with >=10% down').\n"
        "- conventional_pmi_required_statement: Whether the answer states PMI is required for conventional with <20% down.\n"
        "- conventional_pmi_removal_ltv_statement: The LTV threshold for removing PMI "
        "(e.g., 'can be removed at 80% LTV based on original property value').\n"
        "If an item is missing, return null."
    )


def prompt_extract_recommendation() -> str:
    return (
        "Extract the final recommendation and reasoning. Return:\n"
        "- recommended_loan_type: The recommended option (prefer a concise label like 'FHA', 'Conventional', 'Wait/Improve credit').\n"
        "- recommendation_reason: The explanation for the recommendation, ideally considering long-term mortgage insurance implications.\n"
        "If either is missing, return null."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_status(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = text.strip().lower()
    if "eligible" in t and "not" not in t:
        return "eligible"
    if "ineligible" in t or "not eligible" in t or "does not qualify" in t or "unqualified" in t:
        return "not eligible"
    # Try exact words
    if t == "eligible":
        return "eligible"
    if t == "not eligible":
        return "not eligible"
    return None


def parse_dollar_amount(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    t = text.strip()
    # Remove currency symbols and commas
    for ch in ["$", ","]:
        t = t.replace(ch, "")
    t = t.strip()
    try:
        # Some answers might include words like "approx" or "~"
        t = t.replace("approx", "").replace("approximately", "").replace("~", "").strip()
        return float(t)
    except Exception:
        return None


def contains_number(text: Optional[str], target: str) -> bool:
    if not text or not target:
        return False
    return target in text


def mentions_175_percent(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower().replace(" ", "")
    return ("1.75%" in text) or ("1.75percent" in t) or ("1.75" in t and "%of" in t)


def mentions_pmi_required_under_20(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return ("pmi" in t and ("<20%" in t or "less than 20%" in t or "under 20%" in t or "below 20%" in t))


def mentions_pmi_remove_at_80_ltv(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return ("80% ltv" in t) or ("at 80% ltv" in t) or ("ltv of 80%" in t) or ("remove at 80%" in t)


def recommendation_is_consistent(rec_type: Optional[str]) -> bool:
    if not rec_type:
        return False
    t = rec_type.strip().lower()
    # Given conventional is NOT eligible and FHA IS eligible, consistent recommendations are:
    # - FHA (now), or
    # - Wait/Improve credit (to become conventional-eligible), or similar phrasing
    return ("fha" in t) or ("wait" in t) or ("improve" in t) or ("raise" in t) or ("increase" in t) or ("later" in t) or ("refinance" in t)


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_eligibility_assessment(
    evaluator: Evaluator,
    parent_node,
    elig: EligibilityExtraction
) -> None:
    """
    Build and verify the eligibility assessment sub-tree.
    """
    eligibility_node = evaluator.add_parallel(
        id="eligibility_assessment",
        desc="Correctly determine eligibility for both FHA and conventional loan options based on the borrower's credit score",
        parent=parent_node,
        critical=True
    )

    # FHA eligibility subtree (sequential, critical)
    fha_seq = evaluator.add_sequential(
        id="fha_eligibility",
        desc="FHA eligibility correctness and explanation (3.5% down min 580; 10% down min 500)",
        parent=eligibility_node,
        critical=True
    )

    fha_status_norm = normalize_status(elig.fha_status)
    fha_correct_custom = evaluator.add_custom_node(
        result=(EXPECTED_FHA_ELIGIBLE and fha_status_norm == "eligible"),
        id="fha_eligibility_correct",
        desc="The answer's FHA eligibility determination is correct for a 615 credit score (>= 580 for 3.5% down)",
        parent=fha_seq,
        critical=True
    )

    fha_min_leaf = evaluator.add_leaf(
        id="fha_min_score_cited",
        desc="The explanation cites the correct FHA minimum score (580 for 3.5% down)",
        parent=fha_seq,
        critical=True
    )
    await evaluator.verify(
        claim="The explanation in the answer mentions that the minimum credit score for FHA with 3.5% down is 580.",
        node=fha_min_leaf,
        additional_instruction="Pass if the answer clearly cites '580' as the FHA minimum for 3.5% down; mentioning 500 as the minimum for 10% down is fine but not sufficient alone."
    )

    # Conventional eligibility subtree (sequential, critical)
    conv_seq = evaluator.add_sequential(
        id="conventional_eligibility",
        desc="Conventional eligibility correctness and explanation (5% down min 620)",
        parent=eligibility_node,
        critical=True
    )

    conv_status_norm = normalize_status(elig.conventional_status)
    conv_correct_custom = evaluator.add_custom_node(
        result=(not EXPECTED_CONV_ELIGIBLE and conv_status_norm == "not eligible"),
        id="conventional_eligibility_correct",
        desc="The answer's conventional eligibility determination is correct (615 < 620 => not eligible)",
        parent=conv_seq,
        critical=True
    )

    conv_min_leaf = evaluator.add_leaf(
        id="conventional_min_score_cited",
        desc="The explanation cites the correct conventional minimum score (620 for 5% down)",
        parent=conv_seq,
        critical=True
    )
    await evaluator.verify(
        claim="The explanation in the answer mentions that the minimum credit score for conventional loans with 5% down is 620.",
        node=conv_min_leaf,
        additional_instruction="Pass if the answer clearly cites '620' as the conventional minimum for this scenario."
    )


async def build_mortgage_insurance_analysis(
    evaluator: Evaluator,
    parent_node,
    ins: InsuranceExtraction
) -> None:
    """
    Build and verify the mortgage insurance analysis sub-tree.
    """
    mi_node = evaluator.add_parallel(
        id="mortgage_insurance_analysis",
        desc="Accurately calculate and compare mortgage insurance requirements for eligible loan types",
        parent=parent_node,
        critical=True
    )

    # FHA MIP analysis: gated by actual eligibility (scenario-based)
    fha_seq = evaluator.add_sequential(
        id="fha_mip_main",
        desc="FHA mortgage insurance analysis (upfront MIP 1.75%, annual MIP life-of-loan with <10% down)",
        parent=mi_node,
        critical=True
    )

    fha_gate = evaluator.add_custom_node(
        result=EXPECTED_FHA_ELIGIBLE,
        id="fha_mip_gate",
        desc="FHA is eligible for a 615 credit score (gate for FHA MIP analysis)",
        parent=fha_seq,
        critical=True
    )

    # Ensure the answer provides both percent and dollar
    fha_fields_provided = evaluator.add_custom_node(
        result=(ins.fha_upfront_mip_percent is not None and ins.fha_upfront_mip_dollar is not None),
        id="fha_mip_fields_provided",
        desc="FHA upfront MIP percent and dollar amount are provided in the answer",
        parent=fha_seq,
        critical=True
    )

    # Percent correctness via LLM verification (textual presence of 1.75%)
    fha_percent_leaf = evaluator.add_leaf(
        id="fha_mip_percent_correct",
        desc="The answer states the FHA upfront MIP is 1.75% of the loan amount",
        parent=fha_seq,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the FHA upfront Mortgage Insurance Premium (MIP) is 1.75% of the loan amount.",
        node=fha_percent_leaf,
        additional_instruction="Pass only if the answer clearly mentions 1.75% and ties it to the loan amount (not purchase price)."
    )

    # Dollar correctness via custom numeric check within a reasonable band
    fha_mip_value = parse_dollar_amount(ins.fha_upfront_mip_dollar)
    fha_dollar_ok = (fha_mip_value is not None) and (UFMIP_ACCEPT_MIN <= fha_mip_value <= UFMIP_ACCEPT_MAX)
    fha_dollar_leaf = evaluator.add_custom_node(
        result=fha_dollar_ok,
        id="fha_mip_dollar_correct",
        desc=f"Upfront MIP dollar amount is approximately correct (expected between ${UFMIP_ACCEPT_MIN:,.0f} and ${UFMIP_ACCEPT_MAX:,.0f})",
        parent=fha_seq,
        critical=True
    )

    # Annual MIP removal rule correctness
    fha_annual_leaf = evaluator.add_leaf(
        id="fha_annual_mip_removal_rule",
        desc="The answer correctly states annual MIP lasts for the life of the loan with less than 10% down",
        parent=fha_seq,
        critical=True
    )
    await evaluator.verify(
        claim="For FHA with less than 10% down, the annual MIP cannot be removed and lasts for the life of the loan.",
        node=fha_annual_leaf,
        additional_instruction="Pass if the answer clearly states life-of-loan MIP for <10% down; mentioning 11-year removal for >=10% down is acceptable context."
    )

    # Conventional PMI analysis: gated by actual eligibility (scenario-based)
    conv_seq = evaluator.add_sequential(
        id="conventional_pmi_main",
        desc="Conventional PMI analysis (PMI required with <20% down; removable at 80% LTV based on original value)",
        parent=mi_node,
        critical=True
    )

    conv_gate = evaluator.add_custom_node(
        result=EXPECTED_CONV_ELIGIBLE,
        id="conventional_pmi_gate",
        desc="Conventional is eligible (gate for PMI analysis) — in this scenario it is not",
        parent=conv_seq,
        critical=True
    )

    conv_pmi_req_leaf = evaluator.add_leaf(
        id="conventional_pmi_required",
        desc="The answer states PMI is required for conventional loans with less than 20% down",
        parent=conv_seq,
        critical=True
    )
    await evaluator.verify(
        claim="For conventional loans, Private Mortgage Insurance (PMI) is required when the down payment is less than 20%.",
        node=conv_pmi_req_leaf,
        additional_instruction="Pass if the answer clearly states PMI is required under 20% down. This leaf will be skipped if conventional is not eligible (gate fails)."
    )

    conv_pmi_remove_leaf = evaluator.add_leaf(
        id="conventional_pmi_removal_80_ltv",
        desc="The answer states PMI can be removed at 80% LTV based on the original property value",
        parent=conv_seq,
        critical=True
    )
    await evaluator.verify(
        claim="PMI for a conventional loan can be removed when the loan-to-value ratio reaches 80%, typically based on the original property value.",
        node=conv_pmi_remove_leaf,
        additional_instruction="Pass if the answer mentions removal at 80% LTV; note that 78% is the automatic cancellation threshold, but 80% is commonly cited for borrower-requested removal."
    )


async def build_optimal_recommendation(
    evaluator: Evaluator,
    parent_node,
    rec: RecommendationExtraction
) -> None:
    """
    Build and verify the optimal recommendation sub-tree (non-critical, partial credit allowed).
    """
    rec_node = evaluator.add_parallel(
        id="optimal_recommendation",
        desc="Provide a well-justified recommendation based on the analysis",
        parent=parent_node,
        critical=False
    )

    # Check presence of cost comparison (MI implications over time)
    cost_leaf = evaluator.add_leaf(
        id="cost_comparison",
        desc="Compare the financial implications of eligible loan options, considering mortgage insurance costs over time",
        parent=rec_node,
        critical=False
    )
    await evaluator.verify(
        claim="The recommendation discusses or compares long-term mortgage insurance implications (e.g., FHA MIP life-of-loan vs conventional PMI being cancellable) to justify the choice.",
        node=cost_leaf,
        additional_instruction="Pass if the answer meaningfully contrasts FHA vs Conventional MI over time; superficial mentions without comparison should not pass."
    )

    # Recommendation consistency (custom logic) + justification quality via simple check
    # Custom consistency: given conventional not eligible and FHA eligible, consistent if recommending FHA or clearly advising to wait/improve to become conventional-eligible.
    consistency_result = recommendation_is_consistent(rec.recommended_loan_type)
    justified_custom = evaluator.add_custom_node(
        result=consistency_result,
        id="justified_conclusion",
        desc="Recommendation is logically consistent with eligibility (FHA eligible; conventional not) and considers MI removal potential",
        parent=rec_node,
        critical=False
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
    Evaluate an answer for the loan comparison analysis task.
    """
    # Initialize evaluator (root should be non-critical to allow non-critical recommendation subtree)
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
        default_model=model
    )

    # Extract all components (run concurrently)
    elig_task = evaluator.extract(
        prompt=prompt_extract_eligibility(),
        template_class=EligibilityExtraction,
        extraction_name="eligibility_extraction"
    )
    ins_task = evaluator.extract(
        prompt=prompt_extract_insurance(),
        template_class=InsuranceExtraction,
        extraction_name="insurance_extraction"
    )
    rec_task = evaluator.extract(
        prompt=prompt_extract_recommendation(),
        template_class=RecommendationExtraction,
        extraction_name="recommendation_extraction"
    )

    elig_res, ins_res, rec_res = await asyncio.gather(elig_task, ins_task, rec_task)

    # Ground truth / scenario info
    evaluator.add_ground_truth({
        "scenario": {
            "purchase_price": PURCHASE_PRICE,
            "down_payment_dollars": DOWN_PAYMENT_DOLLARS,
            "down_payment_percent": DOWN_PAYMENT_PCT,
            "credit_score": CREDIT_SCORE
        },
        "thresholds": {
            "fha_min_score_3_5_down": FHA_MIN_SCORE_3_5_DOWN,
            "fha_min_score_10_down": FHA_MIN_SCORE_10_DOWN,
            "conventional_min_score_5_down": CONV_MIN_SCORE_5_DOWN
        },
        "expected": {
            "fha_eligible": EXPECTED_FHA_ELIGIBLE,
            "conventional_eligible": EXPECTED_CONV_ELIGIBLE,
            "ufmip_expected_3_5_down": round(EXPECTED_UFMIP_3_5_DOWN, 2),
            "ufmip_expected_5_down": round(EXPECTED_UFMIP_5_DOWN, 2),
            "ufmip_accept_range": [round(UFMIP_ACCEPT_MIN, 2), round(UFMIP_ACCEPT_MAX, 2)]
        }
    })

    # Build verification tree according to rubric structure
    await build_eligibility_assessment(evaluator, root, elig_res)
    await build_mortgage_insurance_analysis(evaluator, root, ins_res)
    await build_optimal_recommendation(evaluator, root, rec_res)

    # Return structured evaluation summary
    return evaluator.get_summary()