import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "loan_upfront_cash_minimization_tx_veteran_600"
TASK_DESCRIPTION = (
    "A military veteran with a credit score of 600 is planning to purchase a primary residence in Texas for $350,000. "
    "They want to minimize their upfront cash requirement (down payment plus closing costs). Considering FHA loans, "
    "conventional loans, and VA loans, which loan type would require the lowest total upfront cash payment? Provide the "
    "specific loan type name and calculate the total upfront cash required for that loan option, showing both the down "
    "payment amount and estimated closing costs separately."
)

PURCHASE_PRICE = 350000.0
CLOSING_COST_MIN_PCT = 2.0
CLOSING_COST_MAX_PCT = 5.0

# Minimum down payment rules used in this evaluation
FHA_MIN_DOWN_PCT = 3.5  # for credit score >= 580
CONV_MIN_DOWN_PCT = 3.0
VA_MIN_DOWN_PCT = 0.0


class SelectedOptionExtraction(BaseModel):
    """Structured extraction of the final selected option and key numbers used in the answer."""
    # Scenario use
    purchase_price_used: Optional[str] = None
    credit_score_used: Optional[str] = None
    veteran_status_considered: Optional[bool] = None
    considered_loan_types: List[str] = Field(default_factory=list)

    # Selection
    selected_loan_type: Optional[str] = None

    # Down payment for selected option
    selected_down_payment_percent: Optional[str] = None
    selected_down_payment_dollars: Optional[str] = None

    # Closing costs used in final breakdown
    selected_closing_costs_percent: Optional[str] = None
    selected_closing_costs_dollars: Optional[str] = None

    # Total upfront cash used in final breakdown
    selected_total_upfront_cash_dollars: Optional[str] = None


def prompt_extract_selected_option() -> str:
    return (
        "Extract the final selected loan option and the numeric breakdown used in the answer.\n"
        "Return the following fields:\n"
        "1) purchase_price_used: the purchase price dollar amount the answer uses in any calculations (e.g., \"$350,000\"). If not mentioned, return null.\n"
        "2) credit_score_used: the credit score the answer references when applying FHA rules (e.g., \"600\"). If not mentioned, return null.\n"
        "3) veteran_status_considered: true if the answer treats the buyer as VA-eligible (due to veteran status), false if it explicitly disqualifies, null if unclear.\n"
        "4) considered_loan_types: an array listing each of FHA, conventional, and/or VA that the answer explicitly considers (strings such as \"FHA\", \"Conventional\", \"VA\").\n"
        "5) selected_loan_type: the final chosen loan type (one of \"FHA\", \"Conventional\", or \"VA\"). If no final selection is given, return null.\n"
        "6) selected_down_payment_percent: the percent used to compute the down payment for the selected loan type (e.g., \"3.5%\", \"3%\", \"0%\"), if present; otherwise null.\n"
        "7) selected_down_payment_dollars: the down payment dollar amount stated for the selected loan type (e.g., \"$12,250\"), if present; otherwise null.\n"
        "8) selected_closing_costs_percent: the closing cost percent used (e.g., \"3%\") if present; otherwise null.\n"
        "9) selected_closing_costs_dollars: the closing costs dollar amount used (e.g., \"$10,500\"), if present; otherwise null.\n"
        "10) selected_total_upfront_cash_dollars: the total upfront cash dollar amount (down payment + closing costs) stated for the selected option; if present, otherwise null.\n"
        "Rules:\n"
        "- Extract only what is explicitly stated in the answer. If any field is not mentioned, return null.\n"
        "- For arrays, return empty array if nothing is listed.\n"
        "- Keep monetary amounts and percents exactly as written (do not compute or normalize)."
    )


def normalize_loan_type(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = s.strip().lower()
    if "va" in t or "veterans affairs" in t or "u.s. department of veterans affairs" in t:
        return "VA"
    if "fha" in t or "federal housing administration" in t:
        return "FHA"
    if "conventional" in t or "conforming" in t:
        return "Conventional"
    return None


def parse_money(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    # Extract first numeric token like 12,250.50
    m = re.findall(r"([0-9][0-9,]*(?:\.[0-9]+)?)", s.replace(" ", ""))
    if not m:
        return None
    try:
        return float(m[0].replace(",", ""))
    except Exception:
        return None


def parse_percent_value(s: Optional[str]) -> Optional[float]:
    """Parse a single percent value like '3.5%' -> 3.5. If a range is found, return the first number; use parse_percent_range for ranges."""
    if not s:
        return None
    # Handle in case of range, we'll just extract first numeric hit as value
    nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", s)
    if not nums:
        return None
    try:
        return float(nums[0])
    except Exception:
        return None


def parse_percent_range(s: Optional[str]) -> Optional[Tuple[float, float]]:
    """Parse percent ranges like '2-5%' or '2%–5%' and return (min, max) as floats."""
    if not s:
        return None
    # Normalize dashes
    s_norm = s.replace("–", "-").replace("—", "-").lower()
    # Find two numbers in sequence
    nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", s_norm)
    if len(nums) >= 2:
        try:
            lo = float(nums[0])
            hi = float(nums[1])
            if lo <= hi:
                return lo, hi
        except Exception:
            return None
    return None


def approx_equal(a: float, b: float, tol: float = 50.0) -> bool:
    return abs(a - b) <= tol


def compute_expected_down_payment(loan_type: Optional[str]) -> Optional[float]:
    lt = normalize_loan_type(loan_type)
    if not lt:
        return None
    if lt == "VA":
        return PURCHASE_PRICE * (VA_MIN_DOWN_PCT / 100.0)
    if lt == "FHA":
        return PURCHASE_PRICE * (FHA_MIN_DOWN_PCT / 100.0)
    if lt == "Conventional":
        return PURCHASE_PRICE * (CONV_MIN_DOWN_PCT / 100.0)
    return None


def build_and_verify_tree(evaluator: Evaluator, extraction: SelectedOptionExtraction):
    """
    Build the verification tree based on the rubric and run all verifications/custom checks.
    """
    # ---------------- Root (sequential, critical in rubric; root is non-critical in framework) ----------------
    root = evaluator.root

    # ---------------- 1) Uses given scenario inputs ----------------
    scenario_node = evaluator.add_parallel(
        id="use_given_scenario_inputs",
        desc="Uses the given scenario inputs needed for calculation (purchase price $350,000; buyer is a military veteran; credit score 600).",
        parent=root,
        critical=True
    )

    # 1.a Uses $350,000 purchase price
    n_uses_price = evaluator.add_leaf(
        id="uses_purchase_price",
        desc="Uses $350,000 as the purchase price when computing dollar amounts.",
        parent=scenario_node,
        critical=True
    )
    claim_price = "The answer uses $350,000 as the purchase price when computing dollar amounts."
    asyncio.create_task(evaluator.verify(
        claim=claim_price,
        node=n_uses_price,
        additional_instruction="Confirm the computations (down payment or closing costs) are based on $350,000; mark incorrect if a different price was used."
    ))

    # 1.b Uses credit score 600 when applying FHA constraints
    n_uses_credit = evaluator.add_leaf(
        id="uses_credit_score_condition",
        desc="Uses the stated credit score (600) when applying FHA down-payment constraints.",
        parent=scenario_node,
        critical=True
    )
    claim_credit = "The answer uses the stated credit score of 600 when applying FHA down-payment constraints."
    asyncio.create_task(evaluator.verify(
        claim=claim_credit,
        node=n_uses_credit,
        additional_instruction="Check that FHA minimum down payment logic references a credit score of 600 (i.e., 580+ qualifies for 3.5% minimum)."
    ))

    # 1.c Treats buyer as VA-eligible due to veteran status
    n_uses_veteran = evaluator.add_leaf(
        id="uses_veteran_status_condition",
        desc="Treats the buyer as VA-eligible per the provided constraint that the buyer is a military veteran (i.e., does not incorrectly disqualify VA).",
        parent=scenario_node,
        critical=True
    )
    claim_veteran = "The answer treats the buyer as VA-eligible due to the stated veteran status (does not incorrectly disqualify VA)."
    asyncio.create_task(evaluator.verify(
        claim=claim_veteran,
        node=n_uses_veteran,
        additional_instruction="Verify that VA is considered as an eligible option because the buyer is a military veteran."
    ))

    # ---------------- 2) Closing cost estimate is valid ----------------
    closing_node = evaluator.add_parallel(
        id="closing_cost_estimate_is_valid",
        desc="Provides a closing-cost estimate suitable for computing upfront cash, consistent with the stated Texas range.",
        parent=root,
        critical=True
    )

    # Derive closing cost checks from extraction
    cc_dollars = parse_money(extraction.selected_closing_costs_dollars)
    cc_pct_value = parse_percent_value(extraction.selected_closing_costs_percent)
    cc_pct_range = parse_percent_range(extraction.selected_closing_costs_percent)

    # 2.a Closing costs within 2–5% range
    def _closing_within_range() -> bool:
        if cc_pct_range is not None:
            lo, hi = cc_pct_range
            return (lo >= CLOSING_COST_MIN_PCT) and (hi <= CLOSING_COST_MAX_PCT)
        if cc_pct_value is not None:
            return (CLOSING_COST_MIN_PCT <= cc_pct_value <= CLOSING_COST_MAX_PCT)
        if cc_dollars is not None:
            implied_pct = (cc_dollars / PURCHASE_PRICE) * 100.0
            return (CLOSING_COST_MIN_PCT <= implied_pct <= CLOSING_COST_MAX_PCT)
        return False

    evaluator.add_custom_node(
        result=_closing_within_range(),
        id="closing_costs_within_range",
        desc="Estimated closing costs correspond to 2%–5% of $350,000 (either explicitly as a percent in that range or as a dollar amount whose implied percent is in that range).",
        parent=closing_node,
        critical=True
    )

    # 2.b Closing costs math correct
    def _closing_math_correct() -> bool:
        # If a % is stated, they should also provide a dollar conversion that matches.
        if cc_pct_value is not None and cc_dollars is not None:
            calc = PURCHASE_PRICE * (cc_pct_value / 100.0)
            return approx_equal(calc, cc_dollars, tol=50.0)
        # If only dollars are stated, implied percent must be in the 2–5% range (handled here as math-consistency check).
        if cc_pct_value is None and cc_dollars is not None:
            implied_pct = (cc_dollars / PURCHASE_PRICE) * 100.0
            return (CLOSING_COST_MIN_PCT <= implied_pct <= CLOSING_COST_MAX_PCT)
        # If only percent is stated with no dollar conversion, we cannot confirm arithmetic conversion; treat as not correct.
        return False

    evaluator.add_custom_node(
        result=_closing_math_correct(),
        id="closing_costs_math_correct",
        desc="If a % is stated, the conversion to dollars for $350,000 is arithmetically correct; if dollars are stated, the implied % is consistent with the range.",
        parent=closing_node,
        critical=True
    )

    # ---------------- 3) Apply minimum down payment rules ----------------
    rules_node = evaluator.add_parallel(
        id="apply_minimum_down_payment_rules",
        desc="Applies the minimum down-payment rules from the constraints for each loan type considered.",
        parent=root,
        critical=True
    )

    n_fha_rule = evaluator.add_leaf(
        id="fha_min_down_rule_applied",
        desc="Applies FHA minimum down payment as 3.5% when credit score is 580+ (credit score here is 600).",
        parent=rules_node,
        critical=True
    )
    claim_fha_rule = (
        "The answer applies FHA minimum down payment as 3.5% given the 600 credit score (580+ qualifies for 3.5%)."
    )
    asyncio.create_task(evaluator.verify(
        claim=claim_fha_rule,
        node=n_fha_rule,
        additional_instruction="Ensure the FHA minimum down rule is stated or used correctly; do not accept values lower than 3.5% for 580+ scores."
    ))

    n_conv_rule = evaluator.add_leaf(
        id="conventional_min_down_rule_applied",
        desc="Applies conventional minimum down payment of 3% (or higher, but must not claim <3% is allowed).",
        parent=rules_node,
        critical=True
    )
    claim_conv_rule = (
        "The answer applies conventional minimum down payment of 3% (does not claim less than 3% is allowed)."
    )
    asyncio.create_task(evaluator.verify(
        claim=claim_conv_rule,
        node=n_conv_rule,
        additional_instruction="Accept 3% or higher; mark incorrect if the answer claims less than 3% down is allowed for conventional."
    ))

    n_va_rule = evaluator.add_leaf(
        id="va_min_down_rule_applied",
        desc="Applies VA down payment as 0% for an eligible veteran.",
        parent=rules_node,
        critical=True
    )
    claim_va_rule = "The answer applies VA down payment as 0% because the buyer is an eligible veteran."
    asyncio.create_task(evaluator.verify(
        claim=claim_va_rule,
        node=n_va_rule,
        additional_instruction="Do not consider VA funding fees for this check; strictly check down payment is 0% for eligible veterans."
    ))

    # ---------------- 4) Selects true lowest upfront cash option ----------------
    select_node = evaluator.add_parallel(
        id="selects_true_lowest_upfront_cash_option",
        desc="Selects the loan type that yields the lowest total upfront cash (down payment + the stated closing-cost estimate), consistent with the constraints and the calculations.",
        parent=root,
        critical=True
    )

    n_cover_all = evaluator.add_leaf(
        id="comparison_covers_all_three_options",
        desc="The selection rationale considers FHA, conventional, and VA (does not omit one of the specified options).",
        parent=select_node,
        critical=True
    )
    claim_cover_all = "The answer explicitly considers FHA, conventional, and VA loan options when determining the lowest upfront cash."
    asyncio.create_task(evaluator.verify(
        claim=claim_cover_all,
        node=n_cover_all,
        additional_instruction="Check that all three are part of the comparison; mentioning them in the rationale is sufficient."
    ))

    n_selection_consistent = evaluator.add_leaf(
        id="selection_is_consistent_with_minimum_down_payments_and_closing_cost_estimate",
        desc="Given the stated closing-cost estimate and the minimum down-payment rules, the chosen loan type is indeed the minimum-upfront-cash option (ties acceptable only if explicitly identified as a tie and still answers with a specific choice).",
        parent=select_node,
        critical=True
    )
    sel_type_norm = normalize_loan_type(extraction.selected_loan_type) or "UNKNOWN"
    # Build claim dynamically
    if cc_dollars is not None:
        claim_selection = (
            f"Given closing costs of approximately ${round(cc_dollars, 2)}, "
            f"and minimum down payments (VA 0%, Conventional 3%, FHA 3.5%), "
            f"the selected loan type '{sel_type_norm}' yields the lowest total upfront cash."
        )
    else:
        claim_selection = (
            f"Given minimum down payments (VA 0%, Conventional 3%, FHA 3.5%) and a typical Texas closing-cost estimate (2–5%), "
            f"the selected loan type '{sel_type_norm}' yields the lowest total upfront cash."
        )
    asyncio.create_task(evaluator.verify(
        claim=claim_selection,
        node=n_selection_consistent,
        additional_instruction=(
            "Assume closing costs are similar across loan types for this comparison and ignore VA funding fees. "
            "Evaluate strictly on down payment plus closing costs."
        )
    ))

    # ---------------- 5) Compute selected option breakdown ----------------
    breakdown_node = evaluator.add_sequential(
        id="compute_selected_option_breakdown",
        desc="Computes and presents the upfront cash breakdown for the selected loan type using the chosen closing-cost estimate.",
        parent=root,
        critical=True
    )

    # 5.a Down payment amount matches minimum rule for selected loan type, computed from $350,000
    dp_expected = compute_expected_down_payment(extraction.selected_loan_type)
    dp_stated = parse_money(extraction.selected_down_payment_dollars)

    evaluator.add_custom_node(
        result=(
            dp_expected is not None and dp_stated is not None and approx_equal(dp_expected, dp_stated, tol=50.0)
        ),
        id="selected_down_payment_amount_correct",
        desc="Down payment is stated as a dollar amount and matches the minimum down-payment rule for the selected loan type, computed from $350,000.",
        parent=breakdown_node,
        critical=True
    )

    # 5.b Closing costs stated and correspond to validated estimate earlier (same number used)
    evaluator.add_custom_node(
        result=(cc_dollars is not None),
        id="selected_closing_costs_amount_stated",
        desc="Closing costs are stated as a dollar amount (or a clearly defined estimate derived from the stated percent) and correspond to the closing-cost estimate validated earlier.",
        parent=breakdown_node,
        critical=True
    )

    # 5.c Total upfront cash equals down payment + closing costs
    total_stated = parse_money(extraction.selected_total_upfront_cash_dollars)
    sum_calc = None
    if dp_stated is not None and cc_dollars is not None:
        sum_calc = dp_stated + cc_dollars

    evaluator.add_custom_node(
        result=(total_stated is not None and sum_calc is not None and approx_equal(total_stated, sum_calc, tol=50.0)),
        id="selected_total_upfront_cash_math_correct",
        desc="Total upfront cash equals down payment + closing costs, with correct arithmetic.",
        parent=breakdown_node,
        critical=True
    )

    # ---------------- 6) Final answer format ----------------
    format_node = evaluator.add_parallel(
        id="final_answer_format",
        desc="Final answer includes the required outputs clearly.",
        parent=root,
        critical=True
    )

    n_names_selected = evaluator.add_leaf(
        id="names_selected_loan_type",
        desc="Explicitly names the selected loan type in the final answer.",
        parent=format_node,
        critical=True
    )
    claim_names_selected = "The final answer explicitly names the selected loan type (FHA, Conventional, or VA)."
    asyncio.create_task(evaluator.verify(
        claim=claim_names_selected,
        node=n_names_selected,
        additional_instruction="Look for a clear statement like 'Selected: VA loan' or equivalent phrasing."
    ))

    n_shows_items = evaluator.add_leaf(
        id="shows_required_line_items",
        desc="Shows down payment and estimated closing costs as separate line items and provides the total upfront cash.",
        parent=format_node,
        critical=True
    )
    claim_shows_items = (
        "The final answer shows down payment and estimated closing costs separately and also provides the total upfront cash."
    )
    asyncio.create_task(evaluator.verify(
        claim=claim_shows_items,
        node=n_shows_items,
        additional_instruction="Accept any clear presentation with the three values separated (DP, closing costs, total)."
    ))

    # Add custom info for transparency
    evaluator.add_custom_info(
        info={
            "normalized_selected_type": sel_type_norm,
            "parsed_down_payment_dollars": dp_stated,
            "parsed_closing_costs_dollars": cc_dollars,
            "parsed_total_upfront_dollars": total_stated,
            "expected_minimum_dp_by_type": {
                "VA": PURCHASE_PRICE * (VA_MIN_DOWN_PCT / 100.0),
                "Conventional": PURCHASE_PRICE * (CONV_MIN_DOWN_PCT / 100.0),
                "FHA": PURCHASE_PRICE * (FHA_MIN_DOWN_PCT / 100.0),
            },
            "closing_costs_percent_value": cc_pct_value,
            "closing_costs_percent_range": cc_pct_range,
        },
        info_type="computed_numbers",
        info_name="computed_numbers"
    )


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
    Entry point for evaluation: builds extraction, verification tree, runs checks, and returns summary.
    """
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Ground truth context (added as info)
    evaluator.add_ground_truth({
        "purchase_price": PURCHASE_PRICE,
        "credit_score": 600,
        "veteran_status": True,
        "state": "Texas",
        "closing_cost_range_percent": [CLOSING_COST_MIN_PCT, CLOSING_COST_MAX_PCT],
        "min_down_payment_rules": {
            "FHA": f"{FHA_MIN_DOWN_PCT}%",
            "Conventional": f"{CONV_MIN_DOWN_PCT}%",
            "VA": f"{VA_MIN_DOWN_PCT}%"
        }
    }, gt_type="constraints")

    # Extract selected option and key numbers
    extraction = await evaluator.extract(
        prompt=prompt_extract_selected_option(),
        template_class=SelectedOptionExtraction,
        extraction_name="selected_option_extraction"
    )

    # Build tree and schedule verifications
    build_and_verify_tree(evaluator, extraction)

    # Wait briefly to ensure all scheduled verifications complete before computing summary
    # (Evaluator.verify returns tasks; we need to wait for the event loop to process them)
    await asyncio.sleep(0.05)

    # Compute final scores/statuses by traversing the tree
    if evaluator.root:
        evaluator.root.compute_score(mutate=True)

    return evaluator.get_summary()