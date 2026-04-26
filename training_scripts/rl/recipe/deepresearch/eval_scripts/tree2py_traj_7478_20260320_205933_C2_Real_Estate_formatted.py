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
TASK_ID = "ga_down_payment_assistance_eval"
TASK_DESCRIPTION = """
Sarah is a first-time homebuyer looking to purchase a home in Atlanta, Georgia. She has been a Georgia resident for 10 months and has the following circumstances:

- Home purchase price: $300,000
- Credit score: 670
- Has not owned a home in the past 5 years
- Current liquid assets: $15,000
- Planning to use an FHA loan (which typically requires 3.5% minimum down payment)

Sarah wants to apply for a down payment assistance program to help with her upfront costs.

Please provide the following information:

1. Identify which Georgia down payment assistance program Sarah qualifies for. Provide the program name and the official website URL for the program.

2. State the amount of down payment assistance Sarah would receive from this program.

3. Calculate Sarah's total upfront costs, including:
   - The net down payment she needs to pay (after applying the assistance)
   - Estimated closing costs based on typical Georgia ranges
   - Total upfront cost amount

Include the official source URLs for all program information you reference.
"""

# Problem constants for calculations (based on the prompt/rubric)
PURCHASE_PRICE = 300_000.0
FHA_DOWN_PAYMENT_RATE = 0.035  # 3.5%
GD_ASSIST_PCT = 0.05           # 5%
GD_ASSIST_CAP = 10_000.0
GD_MIN_BORR_DOWN = 1_000.0
GD_MAX_SALES_PRICE_LIMIT = 550_000.0  # As specified in rubric note
CLOSING_COST_MIN_RATE = 0.02
CLOSING_COST_MAX_RATE = 0.05

SARAH_CREDIT_SCORE = 670
SARAH_FTHB_YEARS = 5
SARAH_LIQUID_ASSETS = 15_000.0


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramExtraction(BaseModel):
    program_name: Optional[str] = None
    program_official_url: Optional[str] = None

    # Eligibility support URLs (official program/agency sources)
    eligibility_credit_score_urls: List[str] = Field(default_factory=list)
    eligibility_first_time_homebuyer_urls: List[str] = Field(default_factory=list)
    georgia_dream_constraints_urls: List[str] = Field(default_factory=list)  # liquid assets, max sales price, borrower min down payment, etc.

    # Assistance rule and amount (as stated in the answer)
    assistance_rule_description: Optional[str] = None
    assistance_rule_source_urls: List[str] = Field(default_factory=list)
    assistance_amount_stated: Optional[str] = None  # e.g., "$10,000"

    # Calculations presented in the answer (strings; may include $ or ranges)
    fha_min_down_payment_stated: Optional[str] = None

    # Net down payment after applying assistance (before any specific program min borrower rule)
    net_down_payment_after_assistance_stated: Optional[str] = None

    # Final net down payment after any program-specific borrower minimum (e.g., GA Dream $1,000)
    final_net_down_payment_stated: Optional[str] = None

    # Closing costs (either value or range)
    closing_costs_estimate_value: Optional[str] = None
    closing_costs_estimate_low: Optional[str] = None
    closing_costs_estimate_high: Optional[str] = None

    # Total upfront cost (either value or range)
    total_upfront_cost_stated: Optional[str] = None
    total_upfront_cost_low: Optional[str] = None
    total_upfront_cost_high: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program_info() -> str:
    return """
Extract the following fields strictly from the provided answer text (do not invent anything):

1) Program identification
- program_name: The specific Georgia down payment assistance program named in the answer.
- program_official_url: The official program/agency website URL for that program (not a news/blog/third-party site).

2) Eligibility official sources (URLs)
- eligibility_credit_score_urls: Official source URL(s) cited in the answer that describe the minimum credit score requirement for the identified program.
- eligibility_first_time_homebuyer_urls: Official source URL(s) cited that define "first-time homebuyer" (e.g., has not owned a home in the past 3 years) for the program.
- georgia_dream_constraints_urls: Official source URL(s) cited for Georgia Dream–specific constraints (if and only if the identified program is Georgia Dream). These should include pages that mention rules such as max sales price, liquid assets limit, minimum borrower down payment, etc. If the program is not Georgia Dream, return an empty array.

3) Assistance rule and amount
- assistance_rule_description: The rule text or summary stated in the answer describing how the down payment assistance amount is determined (e.g., “min(5% of purchase price, $10,000)”).
- assistance_rule_source_urls: Official source URL(s) cited that support the assistance rule described.
- assistance_amount_stated: The numeric down payment assistance amount Sarah would receive as stated in the answer (e.g., "$10,000"). If the answer states a range or not a single value, return the most definitive numeric dollar value referenced or null if not clearly stated.

4) Calculations presented in the answer
- fha_min_down_payment_stated: The computed FHA minimum down payment from the answer (ideally 3.5% of $300,000).
- net_down_payment_after_assistance_stated: The net down payment after subtracting the assistance from the FHA minimum down payment, BEFORE enforcing any program-specific minimum borrower down payment rule (e.g., before GA Dream’s $1,000 min). If not shown, return null.
- final_net_down_payment_stated: The final net down payment after applying any program-specific minimum borrower rule (e.g., GA Dream’s $1,000 minimum). If not shown, return null.

5) Closing costs and total upfront cost from the answer
- closing_costs_estimate_value: If the answer provides a single-dollar estimate for closing costs, extract it (e.g., "$8,000").
- closing_costs_estimate_low: If the answer provides a range for closing costs, extract the low end (e.g., "$6,000"). Otherwise return null.
- closing_costs_estimate_high: If the answer provides a range, extract the high end (e.g., "$15,000"). Otherwise return null.
- total_upfront_cost_stated: If the answer provides a single-dollar total upfront cost, extract it.
- total_upfront_cost_low: If a range total is provided, extract the low end. Otherwise null.
- total_upfront_cost_high: If a range total is provided, extract the high end. Otherwise null.

Notes:
- All URLs must be explicitly present in the answer text. Extract the actual URLs (not just mentions of site names).
- Keep all numeric values as strings exactly as written in the answer (including $ and commas) if possible.
- If any field is missing from the answer, set it to null or an empty array as specified.
"""


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def is_ga_dream(program_name: Optional[str]) -> bool:
    if not program_name:
        return False
    return "georgia dream" in program_name.lower()


def to_float_money(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    s = val.strip()
    if s == "":
        return None
    # Handle text like "$6,000 - $15,000" -> not a single value, return None
    if re.search(r"[-–]\s*\$", s):
        return None
    # Extract first numeric token
    m = re.search(r"([0-9][0-9,]*(?:\.[0-9]+)?)", s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def to_float_money_strict(val: Optional[str]) -> Optional[float]:
    # Strict money parser: allows $, commas, decimals, but no other words
    if val is None:
        return None
    s = val.strip()
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def parse_range_low_high(low: Optional[str], high: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    lo = to_float_money(low)
    hi = to_float_money(high)
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    return lo, hi


def approx_equal(a: Optional[float], b: Optional[float], tol: float = 5.0) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def ensure_urls(primary: Optional[str], extras: List[str]) -> List[str]:
    urls = []
    if extras:
        urls.extend(extras)
    if primary:
        urls.append(primary)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def add_program_identification_nodes(evaluator: Evaluator, parent, extr: ProgramExtraction):
    node = evaluator.add_parallel(
        id="Program_Identification_And_Official_URL",
        desc="Provide the program identity and its official website URL.",
        parent=parent,
        critical=True,
    )

    # Program name existence
    evaluator.add_custom_node(
        result=bool(extr.program_name and extr.program_name.strip()),
        id="Program_Name_Provided",
        desc="States the name of a specific Georgia down payment assistance program being used for the answer.",
        parent=node,
        critical=True,
    )

    # Official URL existence
    evaluator.add_custom_node(
        result=bool(extr.program_official_url and extr.program_official_url.strip()),
        id="Official_Program_Website_URL_Provided",
        desc="Provides the official website URL for the identified program (official program/agency site).",
        parent=node,
        critical=True,
    )


async def add_eligibility_nodes(evaluator: Evaluator, parent, extr: ProgramExtraction):
    node = evaluator.add_parallel(
        id="Eligibility_Verification_With_Official_Sources",
        desc="Demonstrate (with official program/agency source URLs) that Sarah meets the applicable eligibility constraints for the identified program.",
        parent=parent,
        critical=True,
    )

    urls_credit = ensure_urls(extr.program_official_url, extr.eligibility_credit_score_urls)
    urls_fthb = ensure_urls(extr.program_official_url, extr.eligibility_first_time_homebuyer_urls)
    urls_gd = ensure_urls(extr.program_official_url, extr.georgia_dream_constraints_urls)

    # Credit score
    leaf_cs = evaluator.add_leaf(
        id="Meets_Min_Credit_Score",
        desc="Cites an official program/agency source showing the program minimum credit score is <= 670 or explicitly that 670 meets the requirement.",
        parent=node,
        critical=True,
    )
    claim_cs = (
        f"According to the provided official source(s), Sarah's 670 credit score meets the program's "
        f"minimum credit score requirement (i.e., the minimum is 670 or lower, or the page says 670 qualifies)."
    )
    await evaluator.verify(
        claim=claim_cs,
        node=leaf_cs,
        sources=urls_credit,
        additional_instruction="Focus on whether the page shows a minimum credit score at or below 670, or explicitly states that a 670 score qualifies.",
    )

    # First-time homebuyer definition (3 years) and application to Sarah (5 years)
    leaf_fthb = evaluator.add_leaf(
        id="Meets_First_Time_Homebuyer_Definition",
        desc="Cites an official program/agency source defining first-time homebuyer as not owning a home in the past 3 years, and applies it to Sarah (no home in 5 years).",
        parent=node,
        critical=True,
    )
    claim_fthb = (
        "The official program/agency source defines a first-time homebuyer as someone who has not owned a home in the "
        "past 3 years (or an equivalent phrasing). Sarah has not owned a home in the past 5 years, so she satisfies this definition."
    )
    await evaluator.verify(
        claim=claim_fthb,
        node=leaf_fthb,
        sources=urls_fthb,
        additional_instruction="Check the page for a definition similar to 'no homeownership in the last 3 years' and confirm that 5 years satisfies it.",
    )

    # Georgia Dream specifics (conditional)
    if is_ga_dream(extr.program_name):
        # Liquid assets constraint
        leaf_liq = evaluator.add_leaf(
            id="Georgia_Dream_Liquid_Assets_Constraint_If_Applicable",
            desc="Passes if the identified program is not Georgia Dream; if it is Georgia Dream, cites an official source and verifies Sarah meets the liquid assets rule (<= $20,000 or 20% of sales price, whichever is higher).",
            parent=node,
            critical=True,
        )
        claim_liq = (
            "According to the official Georgia Dream source(s), the liquid assets rule requires borrower liquid assets "
            "to be less than or equal to the greater of $20,000 or 20% of the sales price. For a $300,000 price, that cap is $60,000. "
            "Sarah's liquid assets are $15,000, so she meets this rule."
        )
        await evaluator.verify(
            claim=claim_liq,
            node=leaf_liq,
            sources=urls_gd,
            additional_instruction="Verify both the rule text and that $15,000 is below the limit ($60,000) for a $300,000 purchase.",
        )

        # Max sales price
        leaf_max = evaluator.add_leaf(
            id="Georgia_Dream_Max_Sales_Price_Constraint_If_Applicable",
            desc="Passes if the identified program is not Georgia Dream; if it is Georgia Dream, cites an official source and verifies the purchase price $300,000 is within the $550,000 max sales price limit.",
            parent=node,
            critical=True,
        )
        claim_max = (
            "According to the official Georgia Dream source(s), the maximum sales price limit is $550,000. "
            "Therefore, a $300,000 purchase is within the limit."
        )
        await evaluator.verify(
            claim=claim_max,
            node=leaf_max,
            sources=urls_gd,
            additional_instruction="Confirm the page indicates a $550,000 purchase price cap (or equivalent) and that $300,000 is within the limit.",
        )
    else:
        # Not Georgia Dream -> automatically pass both Georgia Dream–specific checks
        evaluator.add_custom_node(
            result=True,
            id="Georgia_Dream_Liquid_Assets_Constraint_If_Applicable",
            desc="Not Georgia Dream program, so liquid assets constraint is not applicable and passes by rubric.",
            parent=node,
            critical=True,
        )
        evaluator.add_custom_node(
            result=True,
            id="Georgia_Dream_Max_Sales_Price_Constraint_If_Applicable",
            desc="Not Georgia Dream program, so max sales price constraint is not applicable and passes by rubric.",
            parent=node,
            critical=True,
        )


async def add_assistance_nodes(evaluator: Evaluator, parent, extr: ProgramExtraction):
    node = evaluator.add_parallel(
        id="Down_Payment_Assistance_Amount",
        desc="Compute and report the assistance amount, supported by official program/agency sources (as program information).",
        parent=parent,
        critical=True,
    )

    urls_rule = ensure_urls(extr.program_official_url, extr.assistance_rule_source_urls)

    # Assistance rule sourced
    leaf_rule = evaluator.add_leaf(
        id="Assistance_Rule_Sourced",
        desc="Provides an official program/agency source URL stating the rule used to compute the assistance amount.",
        parent=node,
        critical=True,
    )

    if is_ga_dream(extr.program_name):
        claim_rule = (
            "The official Georgia Dream source(s) state that down payment assistance equals the lesser of 5% of the purchase "
            "price or $10,000."
        )
    else:
        # Use the program's own assistance rule as described in the answer (must be supported by official source)
        rule_text = extr.assistance_rule_description or "the assistance rule described in the answer"
        claim_rule = (
            f"The provided official source(s) support the program's down payment assistance rule as described: {rule_text}."
        )

    await evaluator.verify(
        claim=claim_rule,
        node=leaf_rule,
        sources=urls_rule,
        additional_instruction="Verify that the cited page(s) explicitly describe the stated assistance rule.",
    )

    # Assistance amount calculated
    leaf_amt = evaluator.add_leaf(
        id="Assistance_Amount_Calculated",
        desc="States the numeric down payment assistance amount and shows the calculation per the program’s sourced rule(s). If the program is Georgia Dream, applies assistance = min(5% of purchase price, $10,000).",
        parent=node,
        critical=True,
    )

    stated_assist = to_float_money(extr.assistance_amount_stated)

    if is_ga_dream(extr.program_name):
        expected = min(GD_ASSIST_PCT * PURCHASE_PRICE, GD_ASSIST_CAP)  # min(5% of 300k, 10k) -> 10k
        claim_amt = (
            f"For a $300,000 purchase under Georgia Dream, assistance is min(5% of price, $10,000) = $10,000. "
            f"The answer's stated assistance amount is ${stated_assist:.0f}."
            if stated_assist is not None else
            "For a $300,000 purchase under Georgia Dream, assistance is min(5% of price, $10,000) = $10,000."
        )
        await evaluator.verify(
            claim=claim_amt,
            node=leaf_amt,
            additional_instruction="Accept minor rounding; verify that the stated amount matches $10,000.",
        )
    else:
        # Non-Georgia Dream: verify that the stated assistance aligns with the sourced rule
        amt_text = extr.assistance_amount_stated or "an amount stated in the answer"
        claim_amt = (
            f"According to the official source(s) and the program's assistance rule, the assistance amount Sarah would "
            f"receive for a $300,000 purchase is {amt_text}."
        )
        await evaluator.verify(
            claim=claim_amt,
            node=leaf_amt,
            sources=urls_rule,
            additional_instruction="Check that the amount stated matches the program rule on the cited official page(s).",
        )


async def add_upfront_cost_nodes(evaluator: Evaluator, parent, extr: ProgramExtraction):
    node = evaluator.add_parallel(
        id="Upfront_Cost_Calculations",
        desc="Compute net down payment after assistance, estimate closing costs using the required Georgia range, and compute total upfront costs.",
        parent=parent,
        critical=True,
    )

    # FHA minimum down payment
    fha_leaf = evaluator.add_leaf(
        id="FHA_Minimum_Down_Payment_Computed",
        desc="Computes FHA minimum down payment as 3.5% of $300,000.",
        parent=node,
        critical=True,
    )
    expected_fha = round(FHA_DOWN_PAYMENT_RATE * PURCHASE_PRICE, 2)  # 10500.00
    claim_fha = (
        f"The FHA minimum down payment for a $300,000 purchase at 3.5% is ${expected_fha:,.0f}."
    )
    await evaluator.verify(
        claim=claim_fha,
        node=fha_leaf,
        additional_instruction="This is a straightforward calculation. Accept reasonable rounding."
    )

    # Net down payment after assistance (sequential sub-node)
    net_seq = evaluator.add_sequential(
        id="Net_Down_Payment_After_Assistance",
        desc="Computes the borrower’s net down payment after applying assistance (and any applicable program minimum borrower down payment rule).",
        parent=node,
        critical=True,
    )

    # Child 1: subtraction (raw net)
    raw_net_leaf = evaluator.add_leaf(
        id="Down_Payment_Minus_Assistance_Computed",
        desc="Calculates down payment due after subtracting the assistance amount from the FHA minimum down payment.",
        parent=net_seq,
        critical=True,
    )

    stated_assist = to_float_money(extr.assistance_amount_stated)
    raw_net_calc = None
    if stated_assist is not None:
        raw_net_calc = max(expected_fha - stated_assist, 0.0)

    # Prefer explicit 'net after assistance' value from answer (pre-program-minimum)
    stated_raw_net = to_float_money(extr.net_down_payment_after_assistance_stated)
    if stated_raw_net is None:
        # If not explicitly provided, we still attempt to verify the implied arithmetic step using the stated assistance
        if raw_net_calc is not None:
            claim_raw = (
                f"Subtracting the assistance amount (${stated_assist:,.0f}) from the FHA minimum down payment "
                f"(${expected_fha:,.0f}) gives ${raw_net_calc:,.0f} as the down payment due after assistance."
            )
        else:
            # If no assistance amount was extracted, make a generic arithmetic claim that references the formula
            claim_raw = (
                "The net down payment after assistance equals the FHA minimum down payment ($10,500) minus the "
                "assistance amount stated in the answer."
            )
    else:
        # Verify the answer shows or implies the correct subtraction result
        # We compare against the computed value if possible; otherwise just assert their number follows the formula.
        if raw_net_calc is not None:
            claim_raw = (
                f"In the answer, the down payment after subtracting assistance is correctly computed as "
                f"${stated_raw_net:,.0f}, matching ${expected_fha:,.0f} - ${stated_assist:,.0f} = ${raw_net_calc:,.0f}."
            )
        else:
            claim_raw = (
                f"In the answer, the down payment after subtracting assistance is computed as ${stated_raw_net:,.0f}."
            )

    await evaluator.verify(
        claim=claim_raw,
        node=raw_net_leaf,
        additional_instruction="Accept minor rounding; the intent is to verify the subtraction step."
    )

    # Child 2: enforce GA Dream min borrower down payment if applicable
    if is_ga_dream(extr.program_name):
        enforce_leaf = evaluator.add_leaf(
            id="Georgia_Dream_Min_Borrower_Down_Payment_Enforced_If_Applicable",
            desc="Passes if the identified program is not Georgia Dream; if it is Georgia Dream, enforces the minimum $1,000 borrower down payment requirement in the final net down payment.",
            parent=net_seq,
            critical=True,
        )
        final_net_stated = to_float_money(extr.final_net_down_payment_stated)
        # If we have raw_net_calc, expected final = max(raw_net_calc, $1,000)
        expected_final = None
        if raw_net_calc is not None:
            expected_final = max(raw_net_calc, GD_MIN_BORR_DOWN)

        if expected_final is not None and final_net_stated is not None:
            claim_enf = (
                f"For Georgia Dream, the borrower must pay at least ${GD_MIN_BORR_DOWN:,.0f} out of pocket. "
                f"Given the raw net of ${raw_net_calc:,.0f}, the final net down payment should be "
                f"${expected_final:,.0f}. The answer's final net down payment is ${final_net_stated:,.0f}."
            )
        elif expected_final is not None:
            claim_enf = (
                f"For Georgia Dream, the borrower must pay at least ${GD_MIN_BORR_DOWN:,.0f}. "
                f"Given the raw net of ${raw_net_calc:,.0f}, the final net down payment should be ${expected_final:,.0f}."
            )
        else:
            claim_enf = (
                f"For Georgia Dream, the borrower must pay at least ${GD_MIN_BORR_DOWN:,.0f} as a minimum down payment."
            )

        await evaluator.verify(
            claim=claim_enf,
            node=enforce_leaf,
            additional_instruction="Check that the final net down payment shown in the answer is at least $1,000 for Georgia Dream (or equals the raw net if it exceeds $1,000)."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="Georgia_Dream_Min_Borrower_Down_Payment_Enforced_If_Applicable",
            desc="Not Georgia Dream program; GA Dream $1,000 minimum borrower down payment is not applicable and passes by rubric.",
            parent=net_seq,
            critical=True,
        )

    # Closing costs estimate in 2%–5% range of $300,000
    cc_leaf_ok = evaluator.add_custom_node(
        result=False,  # placeholder; will set true after computing
        id="Closing_Costs_Estimate_In_Range",
        desc="Estimates buyer closing costs using a value or a range consistent with 2%–5% of $300,000 (i.e., between $6,000 and $15,000).",
        parent=node,
        critical=True,
    )
    cc_min = PURCHASE_PRICE * CLOSING_COST_MIN_RATE  # 6,000
    cc_max = PURCHASE_PRICE * CLOSING_COST_MAX_RATE  # 15,000
    cc_value = to_float_money(extr.closing_costs_estimate_value)
    cc_lo, cc_hi = parse_range_low_high(extr.closing_costs_estimate_low, extr.closing_costs_estimate_high)

    cc_in_range = False
    if cc_value is not None:
        cc_in_range = (cc_min - 5.0) <= cc_value <= (cc_max + 5.0)  # minor tolerance
    elif cc_lo is not None and cc_hi is not None:
        cc_in_range = (cc_min - 5.0) <= cc_lo <= (cc_max + 5.0) and (cc_min - 5.0) <= cc_hi <= (cc_max + 5.0)

    # Overwrite the custom node with actual result (create a new node id automatically if duplicate)
    evaluator.add_custom_node(
        result=cc_in_range,
        id="Closing_Costs_Estimate_In_Range",
        desc=f"Closing costs within 2%–5% of ${PURCHASE_PRICE:,.0f} (i.e., ${cc_min:,.0f}–${cc_max:,.0f}).",
        parent=node,
        critical=True,
    )

    # Total upfront cost totaled = final net down payment + closing costs
    total_leaf = evaluator.add_custom_node(
        result=False,  # placeholder; will add correct result next
        id="Total_Upfront_Cost_Totaled",
        desc="Computes total upfront costs as (net down payment after assistance) + (estimated closing costs).",
        parent=node,
        critical=True,
    )

    # Build expected totals based on extracted/stated values
    # Determine final net down payment number to use
    final_net = to_float_money(extr.final_net_down_payment_stated)
    if final_net is None:
        # If final net not provided, fall back to raw net (may be acceptable for non-GD programs)
        final_net = to_float_money(extr.net_down_payment_after_assistance_stated)

    # Determine closing costs numeric(s)
    closing_single = to_float_money(extr.closing_costs_estimate_value)
    closing_lo, closing_hi = parse_range_low_high(extr.closing_costs_estimate_low, extr.closing_costs_estimate_high)

    total_ok = False
    # Extract provided totals
    total_single = to_float_money(extr.total_upfront_cost_stated)
    total_low = to_float_money(extr.total_upfront_cost_low)
    total_high = to_float_money(extr.total_upfront_cost_high)

    if final_net is not None:
        if closing_single is not None and total_single is not None:
            expected_total = final_net + closing_single
            total_ok = approx_equal(expected_total, total_single, tol=5.0)
        elif closing_lo is not None and closing_hi is not None and total_low is not None and total_high is not None:
            expected_low = final_net + closing_lo
            expected_high = final_net + closing_hi
            total_ok = approx_equal(expected_low, total_low, tol=5.0) and approx_equal(expected_high, total_high, tol=5.0)
        elif closing_single is not None and total_low is not None and total_high is not None:
            # If answer gave total as range but closing as single, compute range by +/- small tolerance
            expected_low = final_net + closing_single
            expected_high = final_net + closing_single
            total_ok = approx_equal(expected_low, total_low, tol=5.0) and approx_equal(expected_high, total_high, tol=5.0)
        elif closing_lo is not None and closing_hi is not None and total_single is not None:
            # If answer gave closing as range but total as single, accept if total_single is between expected_low and expected_high
            expected_low = final_net + closing_lo
            expected_high = final_net + closing_hi
            total_ok = (expected_low - 5.0) <= total_single <= (expected_high + 5.0)

    evaluator.add_custom_node(
        result=total_ok,
        id="Total_Upfront_Cost_Totaled",
        desc="Total upfront cost correctly totals final net down payment and closing costs (allowing minor rounding).",
        parent=node,
        critical=True,
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
    Evaluate an answer for the Georgia Down Payment Assistance task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root is sequential per rubric
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

    # Extract structured information from the answer
    extr: ProgramExtraction = await evaluator.extract(
        prompt=prompt_extract_program_info(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction",
    )

    # Build the rubric tree under the critical root "Complete_Task"
    complete_task = evaluator.add_sequential(
        id="Complete_Task",
        desc="Identify a Georgia down payment assistance program Sarah qualifies for, provide required official URLs/sources for program information, compute assistance, and calculate total upfront costs (net down payment + closing costs).",
        parent=root,
        critical=True,
    )

    # 1) Program identification
    await add_program_identification_nodes(evaluator, complete_task, extr)

    # 2) Eligibility verification with official sources
    await add_eligibility_nodes(evaluator, complete_task, extr)

    # 3) Assistance rule + amount
    await add_assistance_nodes(evaluator, complete_task, extr)

    # 4) Upfront cost calculations
    await add_upfront_cost_nodes(evaluator, complete_task, extr)

    # Add helpful computed info to the summary
    expected_fha = round(FHA_DOWN_PAYMENT_RATE * PURCHASE_PRICE, 2)
    expected_gd_assist = min(GD_ASSIST_PCT * PURCHASE_PRICE, GD_ASSIST_CAP)
    evaluator.add_custom_info(
        info={
            "purchase_price": PURCHASE_PRICE,
            "expected_fha_min_down": expected_fha,
            "ga_dream_rule_min(5%,$10k)_for_300k": expected_gd_assist,
            "closing_cost_expected_range": [PURCHASE_PRICE * CLOSING_COST_MIN_RATE, PURCHASE_PRICE * CLOSING_COST_MAX_RATE],
            "sarah_profile": {
                "credit_score": SARAH_CREDIT_SCORE,
                "first_time_homebuyer_years_no_home": SARAH_FTHB_YEARS,
                "liquid_assets": SARAH_LIQUID_ASSETS,
                "loan_program": "FHA",
            },
            "program_detected": extr.program_name,
            "program_official_url": extr.program_official_url,
        },
        info_type="computed_context",
    )

    return evaluator.get_summary()