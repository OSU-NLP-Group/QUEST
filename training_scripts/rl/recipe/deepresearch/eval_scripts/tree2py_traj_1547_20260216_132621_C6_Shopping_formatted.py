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
TASK_ID = "shopping_strategy_optimization_2026"
TASK_DESCRIPTION = """
You are planning to make several household purchases over the next 2 months with a total value of approximately $550-600. Your purchases will include: (1) Electronics/appliances worth approximately $300, (2) Home goods and household items worth approximately $150, and (3) Grocery and pantry items worth approximately $100-150. You want to minimize your total out-of-pocket costs while maximizing purchase protection benefits and return flexibility. You are considering shopping at major retailers (Walmart, Target, or Amazon) and are open to joining a membership program if it provides net savings. Determine the optimal shopping strategy by addressing the following: (1) Retailer Selection: Which retailer(s) should you use for your purchases to minimize shipping costs while meeting free shipping thresholds? Identify the minimum purchase amount required for free standard shipping at these retailers. (2) Membership Program Decision: Should you join a retail membership program (Walmart+, Amazon Prime, or Target Circle 360)? Calculate whether the annual membership cost would be offset by shipping savings and other benefits based on your planned purchases. (3) Payment Method: What payment method should you use to maximize rewards and ensure purchase protection? If using a store credit card, identify the rewards rate and APR, and explain whether the rewards justify the potential interest costs. Also verify what credit card protection benefits (purchase protection and extended warranty) would apply. (4) Purchase Timing: When should you make these purchases to take advantage of seasonal sales or promotional events? Identify any relevant sale periods within your 2-month timeframe. (5) Return Policy: What is the standard return window at your selected retailer(s), and would any purchases qualify for extended holiday return periods? (6) Total Cost Analysis: Calculate the net total cost of your shopping strategy, including gross purchase total, applied discounts or rewards, membership fees (if applicable), shipping fees (if any), and final net cost. Provide specific numbers, percentages, and policy details based on current information available as of early 2026, and include reference URLs to support your recommendations.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RetailerThreshold(BaseModel):
    retailer: Optional[str] = None
    free_shipping_threshold: Optional[str] = None
    under_min_fee_policy: Optional[str] = None  # e.g., "Walmart charges a fee for orders under $35"


class RetailerSelection(BaseModel):
    selected_retailers: List[str] = Field(default_factory=list)
    compared_retailers: List[str] = Field(default_factory=list)
    thresholds: List[RetailerThreshold] = Field(default_factory=list)
    shipping_policy_urls: List[str] = Field(default_factory=list)


class OrderAssumptions(BaseModel):
    num_orders: Optional[str] = None  # keep as string to allow ranges/phrases
    per_order_totals: List[str] = Field(default_factory=list)
    shipping_cost_impact_calc: Optional[str] = None  # text explanation with numbers if any


class MembershipProgram(BaseModel):
    name: Optional[str] = None  # e.g., "Walmart+", "Amazon Prime", "Target Circle 360"
    fee: Optional[str] = None   # e.g., "$98", "$139", "$99", "$49 (Target Circle Card holders)"
    join_decision: Optional[str] = None  # e.g., "join", "do not join", "not necessary"
    break_even_calc: Optional[str] = None  # explanation/numbers comparing savings vs. fee
    urls: List[str] = Field(default_factory=list)


class MembershipAnalysis(BaseModel):
    programs: List[MembershipProgram] = Field(default_factory=list)
    membership_urls: List[str] = Field(default_factory=list)


class PaymentProtectionInfo(BaseModel):
    method_name: Optional[str] = None  # e.g., "Chase Sapphire Preferred", "Amex", "Target Circle Card"
    rewards_rate: Optional[str] = None  # e.g., "5%", "3% on groceries", etc.
    apr: Optional[str] = None           # APR number/range for store card if mentioned
    apr_explanation: Optional[str] = None  # text explaining interest tradeoff (e.g., pay in full)
    purchase_protection_term: Optional[str] = None  # e.g., "90 days", "120 days", etc.
    extended_warranty_term: Optional[str] = None    # e.g., "+1 year on warranties of ≤3 years"
    urls: List[str] = Field(default_factory=list)   # card/issuer policy pages


class SalePeriod(BaseModel):
    name: Optional[str] = None           # e.g., "Presidents' Day sale"
    date_range: Optional[str] = None     # e.g., "mid-Feb 2026", "Feb 14–17, 2026"
    applicable_categories: List[str] = Field(default_factory=list)


class PurchaseTiming(BaseModel):
    two_month_window: Optional[str] = None  # e.g., "Jan–Feb 2026" or "Feb–Mar 2026"
    sale_periods: List[SalePeriod] = Field(default_factory=list)
    sale_timing_urls: List[str] = Field(default_factory=list)


class ReturnPolicyItem(BaseModel):
    retailer: Optional[str] = None
    standard_return_window: Optional[str] = None  # e.g., "90 days", "30 days"
    holiday_extension_applicability: Optional[str] = None  # text if applicable or "not applicable"
    urls: List[str] = Field(default_factory=list)


class ReturnPolicies(BaseModel):
    items: List[ReturnPolicyItem] = Field(default_factory=list)


class TotalCostAnalysis(BaseModel):
    gross_total: Optional[str] = None  # e.g., "$580", or "$550–$600"
    category_totals: Dict[str, Optional[str]] = Field(default_factory=dict)  # keys: electronics, home_goods, grocery
    discounts_rewards_total: Optional[str] = None  # e.g., "$35 in rewards", "5% off"
    fees_total: Optional[str] = None  # membership fee + shipping/min-order/delivery fees if any
    net_total: Optional[str] = None   # final net out-of-pocket
    formula: Optional[str] = None     # explicit formula text used (e.g., Net = Gross − Discounts/Rewards + Fees)


class Recency(BaseModel):
    recency_statement: Optional[str] = None  # e.g., "as of early 2026"
    dates_mentioned: List[str] = Field(default_factory=list)  # e.g., "February 2026", "Jan 2026"


class Citations(BaseModel):
    shipping_policy_urls: List[str] = Field(default_factory=list)
    membership_pricing_urls: List[str] = Field(default_factory=list)
    return_policy_urls: List[str] = Field(default_factory=list)
    card_protections_urls: List[str] = Field(default_factory=list)
    sale_timing_urls: List[str] = Field(default_factory=list)


class ShoppingStrategyExtraction(BaseModel):
    recency: Optional[Recency] = None
    retailer_selection: Optional[RetailerSelection] = None
    order_assumptions: Optional[OrderAssumptions] = None
    memberships: Optional[MembershipAnalysis] = None
    payment_methods: List[PaymentProtectionInfo] = Field(default_factory=list)
    timing: Optional[PurchaseTiming] = None
    returns: Optional[ReturnPolicies] = None
    total_cost: Optional[TotalCostAnalysis] = None
    citations: Optional[Citations] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_strategy() -> str:
    return """
    Extract structured information from the answer to evaluate a shopping strategy using current information as of early 2026. Return a single JSON object with the following fields. If any item is missing in the answer, set it to null or an empty list as appropriate.

    recency:
      - recency_statement: A sentence or phrase explicitly indicating information is current "as of early 2026" or equivalent.
      - dates_mentioned: Any retrieval or verification dates mentioned that fall in early 2026 (e.g., Jan–Mar 2026).

    retailer_selection:
      - selected_retailers: Retailers explicitly chosen (subset of Walmart, Target, Amazon).
      - compared_retailers: Any retailers explicitly compared even if not chosen.
      - thresholds: Array of {retailer, free_shipping_threshold, under_min_fee_policy} values stated in the answer (free standard shipping thresholds and any minimum-order fee notes like Walmart under-$35 fee).
      - shipping_policy_urls: All URLs cited that support shipping policy or free-shipping threshold statements.

    order_assumptions:
      - num_orders: The assumed number of orders (text or number) the plan uses.
      - per_order_totals: The approximate per-order totals (array of strings) used in the plan.
      - shipping_cost_impact_calc: A short text explaining whether shipping/minimum-order fees apply under the assumptions, including any Walmart under-$35 fee if Walmart is used.

    memberships:
      - programs: Array of membership analyses; each element:
          { name, fee, join_decision, break_even_calc, urls }
        Include entries for Walmart+, Amazon Prime, and Target Circle 360 if the answer mentions them.
      - membership_urls: All URLs cited for membership pricing pages.

    payment_methods:
      - Array where each element contains:
        { method_name, rewards_rate, apr, apr_explanation, purchase_protection_term, extended_warranty_term, urls }
        If recommending a store card, include rewards rate and APR and any explanation regarding paying in full.

    timing:
      - two_month_window: The assumed two-month calendar window in early 2026 used for timing (e.g., "Jan–Feb 2026").
      - sale_periods: Array of { name, date_range, applicable_categories } for any relevant sale or promo periods mentioned.
      - sale_timing_urls: All URLs cited that support sale/promotional timing.

    returns:
      - items: Array of return policy entries; each element:
        { retailer, standard_return_window, holiday_extension_applicability, urls }
      (retailer from selected retailers if present).

    total_cost:
      - gross_total: The gross planned purchase total (point estimate within $550–$600 or the range itself).
      - category_totals: Object with keys like electronics, home_goods, grocery and string values (e.g., "$300").
      - discounts_rewards_total: Discounts/rewards in dollars or percentage text.
      - fees_total: Sum of applicable fees under the plan (membership fee if joined; shipping/min-order/delivery fees if any).
      - net_total: Final net out-of-pocket cost.
      - formula: The explicit formula text used (e.g., "Net = Gross − Discounts/Rewards + Fees").

    citations:
      - shipping_policy_urls: Duplicate list of shipping policy URLs (for convenience).
      - membership_pricing_urls: Duplicate list of membership pricing URLs.
      - return_policy_urls: Duplicate list of return policy URLs.
      - card_protections_urls: URLs that support purchase protection or extended warranty claims.
      - sale_timing_urls: URLs that support sale/promotional periods.

    Special URL extraction rules:
      - Extract only URLs explicitly present in the answer. Include plain URLs and markdown-formatted links.
      - If a URL is missing a protocol, prepend "http://".
      - Return valid URLs only.
    """


# --------------------------------------------------------------------------- #
# Helper selection functions                                                  #
# --------------------------------------------------------------------------- #
def _find_threshold_for_any_selected(extr: ShoppingStrategyExtraction) -> Optional[RetailerThreshold]:
    rs = extr.retailer_selection or RetailerSelection()
    selected = [r.lower() for r in rs.selected_retailers]
    for th in rs.thresholds:
        if th and th.retailer and th.free_shipping_threshold:
            if not selected or th.retailer.lower() in selected:
                return th
    return None


def _find_membership_with_fee(extr: ShoppingStrategyExtraction) -> Optional[MembershipProgram]:
    ma = extr.memberships or MembershipAnalysis()
    for prog in ma.programs:
        if prog and prog.name and prog.fee:
            return prog
    return None


def _find_return_policy_item(extr: ShoppingStrategyExtraction) -> Optional[ReturnPolicyItem]:
    rp = extr.returns or ReturnPolicies()
    for item in rp.items:
        if item and item.retailer and item.standard_return_window:
            return item
    return None


def _find_payment_with_purchase_protection(extr: ShoppingStrategyExtraction) -> Optional[PaymentProtectionInfo]:
    for pm in extr.payment_methods:
        if pm and pm.method_name and pm.purchase_protection_term:
            return pm
    return None


def _find_payment_with_extended_warranty(extr: ShoppingStrategyExtraction) -> Optional[PaymentProtectionInfo]:
    for pm in extr.payment_methods:
        if pm and pm.method_name and pm.extended_warranty_term:
            return pm
    return None


def _find_sale_period(extr: ShoppingStrategyExtraction) -> Optional[SalePeriod]:
    timing = extr.timing or PurchaseTiming()
    for sp in timing.sale_periods:
        if sp and sp.name and sp.date_range:
            return sp
    return None


def _has_all_membership_fees(extr: ShoppingStrategyExtraction) -> bool:
    ma = extr.memberships or MembershipAnalysis()
    names_to_check = {"walmart+": False, "amazon prime": False, "target circle 360": False}
    for prog in ma.programs:
        if prog and prog.name and prog.fee:
            key = prog.name.lower().strip()
            if key in names_to_check:
                names_to_check[key] = True
    return all(names_to_check.values())


def _all_selected_retailers_have_thresholds(extr: ShoppingStrategyExtraction) -> bool:
    rs = extr.retailer_selection or RetailerSelection()
    if not rs.selected_retailers:
        return False
    selected_lower = {r.lower() for r in rs.selected_retailers}
    found = set()
    for th in rs.thresholds:
        if th and th.retailer and th.free_shipping_threshold:
            retailer_lower = th.retailer.lower().strip()
            if retailer_lower in selected_lower:
                found.add(retailer_lower)
    return found == selected_lower


def _has_shipping_cost_impact_calc(extr: ShoppingStrategyExtraction) -> bool:
    oa = extr.order_assumptions or OrderAssumptions()
    return bool(oa.shipping_cost_impact_calc and oa.shipping_cost_impact_calc.strip())


def _has_apr_explanation(extr: ShoppingStrategyExtraction) -> bool:
    for pm in extr.payment_methods:
        if pm and pm.apr_explanation and pm.apr_explanation.strip():
            return True
    return False


# --------------------------------------------------------------------------- #
# Subtree builders                                                            #
# --------------------------------------------------------------------------- #
async def build_recency_assumption(evaluator: Evaluator, parent, extr: ShoppingStrategyExtraction):
    node = evaluator.add_leaf(
        id="Recency_Assumption",
        desc="Uses information current as of early 2026 (e.g., states 'as of early 2026' or provides a retrieval/verification date in early 2026).",
        parent=parent,
        critical=True,
    )
    rec = extr.recency or Recency()
    result = bool((rec.recency_statement and "2026" in rec.recency_statement) or any("2026" in d for d in rec.dates_mentioned))
    # Use simple verification to check presence-in-answer
    claim = "The answer explicitly indicates that information is current as of early 2026 or includes a retrieval/verification date in early 2026."
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Look for phrases like 'as of early 2026' or explicit dates in Jan–Mar 2026 within the answer text."
    )


async def build_citations_subtree(evaluator: Evaluator, parent, extr: ShoppingStrategyExtraction):
    # Parent citations node
    cit_node = evaluator.add_parallel(
        id="Citations",
        desc="Provides reference URLs supporting key numeric/policy claims used in the recommendation.",
        parent=parent,
        critical=True
    )
    citations = extr.citations or Citations()

    evaluator.add_custom_node(
        result=bool(citations.shipping_policy_urls),
        id="Shipping_Policy_URL",
        desc="Includes at least one URL supporting free-shipping threshold and/or shipping/minimum-order fee policy for at least one selected retailer.",
        parent=cit_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(citations.membership_pricing_urls),
        id="Membership_Pricing_URL",
        desc="Includes at least one URL supporting membership pricing for any membership program recommended or explicitly analyzed.",
        parent=cit_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(citations.return_policy_urls),
        id="Return_Policy_URL",
        desc="Includes at least one URL supporting return-window policy for at least one selected retailer.",
        parent=cit_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(citations.card_protections_urls),
        id="Card_Protections_URL",
        desc="Includes at least one URL supporting purchase protection and/or extended warranty claims for the recommended payment method.",
        parent=cit_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(citations.sale_timing_urls),
        id="Sale_Timing_URL",
        desc="Includes at least one URL supporting any cited sale/promotional period used to justify purchase timing.",
        parent=cit_node,
        critical=True
    )


async def build_retailer_selection_and_shipping_subtree(evaluator: Evaluator, parent, extr: ShoppingStrategyExtraction):
    rs_node = evaluator.add_parallel(
        id="Retailer_Selection_and_Shipping",
        desc="Select retailer(s) (Walmart/Target/Amazon) and minimize shipping costs while meeting free-shipping thresholds.",
        parent=parent,
        critical=True
    )
    rs = extr.retailer_selection or RetailerSelection()
    oa = extr.order_assumptions or OrderAssumptions()

    evaluator.add_custom_node(
        result=bool(rs.selected_retailers),
        id="Retailer_Choice_Stated",
        desc="Clearly identifies which retailer(s) will be used for the purchases (may be 1–3).",
        parent=rs_node,
        critical=True
    )

    # Verify at least one threshold claim via URLs (source-grounded)
    th = _find_threshold_for_any_selected(extr)
    fs_leaf = evaluator.add_leaf(
        id="Free_Shipping_Thresholds_Identified",
        desc="States the minimum purchase amount required for free standard shipping for each selected retailer (and any retailer explicitly compared).",
        parent=rs_node,
        critical=True
    )
    claim = "The answer states the minimum purchase amount required for free standard shipping for the selected retailer(s), and this threshold is supported by the cited shipping policy URL(s)."
    sources = rs.shipping_policy_urls if rs.shipping_policy_urls else (extr.citations.shipping_policy_urls if extr.citations else None)
    await evaluator.verify(
        claim=claim if not th else f"The minimum purchase amount required for free standard shipping at {th.retailer} is {th.free_shipping_threshold}.",
        node=fs_leaf,
        sources=sources,
        additional_instruction="Check the free standard shipping threshold per the retailer policy page(s). Allow minor wording variations such as 'orders $35+ ship free'."
    )

    evaluator.add_custom_node(
        result=bool(oa.num_orders or oa.per_order_totals),
        id="Order_Structure_Assumptions",
        desc="States assumptions about how purchases are grouped into orders (e.g., number of orders and approximate per-order totals) to evaluate shipping/fees.",
        parent=rs_node,
        critical=True
    )

    ship_calc_leaf = evaluator.add_leaf(
        id="Shipping_Cost_Impact_Calculated",
        desc="Calculates whether shipping fees/minimum-order fees apply under the stated order assumptions for selected retailer(s) using explicit numbers (including Walmart under-$35 fee if Walmart is used, per constraints).",
        parent=rs_node,
        critical=True
    )
    calc_claim = "The answer calculates whether shipping or minimum-order fees apply under the stated order assumptions, using explicit numbers; if Walmart is used, it considers the under-$35 fee."
    await evaluator.verify(
        claim=calc_claim,
        node=ship_calc_leaf,
        additional_instruction="Look for explicit numeric examples tying per-order totals to shipping thresholds and fees."
    )


async def build_membership_program_decision_subtree(evaluator: Evaluator, parent, extr: ShoppingStrategyExtraction):
    mem_node = evaluator.add_parallel(
        id="Membership_Program_Decision",
        desc="Decide whether to join Walmart+, Amazon Prime, and/or Target Circle 360 and quantify net value under the plan.",
        parent=parent,
        critical=True
    )
    ma = extr.memberships or MembershipAnalysis()

    evaluator.add_custom_node(
        result=_has_all_membership_fees(extr),
        id="Membership_Fees_Identified",
        desc="States annual costs for Walmart+ ($98), Amazon Prime ($139), and Target Circle 360 ($99; $49 for Target Circle Card holders).",
        parent=mem_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=any(prog.join_decision for prog in ma.programs),
        id="Membership_Join_or_Not_Decision",
        desc="Makes an explicit recommendation for each membership (join/not join) or a clearly justified subset, consistent with the stated shopping plan.",
        parent=mem_node,
        critical=True
    )

    be_leaf = evaluator.add_leaf(
        id="Break_Even_or_Net_Value_Calculation",
        desc="Quantifies membership net impact using stated assumptions (e.g., shipping savings and/or delivery-fee avoidance and/or monetized benefits) and compares against the annual fee.",
        parent=mem_node,
        critical=True
    )
    prog_for_calc = _find_membership_with_fee(extr)
    be_claim = "The answer quantifies membership net impact by comparing shipping savings and/or delivery-fee avoidance and other monetized benefits against the annual membership fee."
    # Use membership URLs if available
    mem_sources = ma.membership_urls if ma.membership_urls else (extr.citations.membership_pricing_urls if extr.citations else None)
    await evaluator.verify(
        claim=be_claim if not prog_for_calc else f"The annual cost for {prog_for_calc.name} is {prog_for_calc.fee}, and the answer compares this fee to estimated savings.",
        node=be_leaf,
        sources=mem_sources,
        additional_instruction="Look for numeric or clearly quantified comparisons versus the annual fee."
    )


async def build_payment_method_and_protections_subtree(evaluator: Evaluator, parent, extr: ShoppingStrategyExtraction):
    pay_node = evaluator.add_parallel(
        id="Payment_Method_and_Protections",
        desc="Recommend a payment method that maximizes rewards and addresses purchase protection/extended warranty and store-card APR vs rewards tradeoff.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extr.payment_methods),
        id="Payment_Method_Recommendation",
        desc="States the recommended payment method(s) for the purchases (may split by category such as electronics vs groceries).",
        parent=pay_node,
        critical=True
    )

    # Rewards rate disclosed for at least one recommended method
    rewards_present = any(pm.rewards_rate for pm in extr.payment_methods)
    evaluator.add_custom_node(
        result=rewards_present,
        id="Rewards_Rate_Disclosed",
        desc="Discloses rewards rate(s) for the recommended method(s). If recommending a store card, includes the applicable store-card reward rate per constraints.",
        parent=pay_node,
        critical=True
    )

    apr_leaf = evaluator.add_leaf(
        id="APR_and_Interest_Tradeoff_Explained_For_Store_Card_If_Used",
        desc="If a store credit card is recommended/used, states APR and explains whether rewards justify potential interest costs (e.g., pay in full).",
        parent=pay_node,
        critical=True
    )
    apr_claim = "If any store credit card is recommended, the answer states APR and explains that interest costs can outweigh rewards if not paid in full; it recommends paying in full to avoid interest."
    await evaluator.verify(
        claim=apr_claim,
        node=apr_leaf,
        additional_instruction="Look for explicit APR mention and a pay-in-full recommendation to avoid interest costs."
    )

    # Purchase protection verification via URLs
    pp_info = _find_payment_with_purchase_protection(extr)
    pp_leaf = evaluator.add_leaf(
        id="Purchase_Protection_Explained",
        desc="Explains applicable purchase protection coverage terms consistent with constraints (typically 90–180 days) and ties it to the plan.",
        parent=pay_node,
        critical=True
    )
    pp_sources = []
    if extr.citations and extr.citations.card_protections_urls:
        pp_sources.extend(extr.citations.card_protections_urls)
    if pp_info and pp_info.urls:
        pp_sources.extend(pp_info.urls)
    pp_claim = "The recommended payment method offers purchase protection within the 90–180 day range."
    if pp_info and pp_info.purchase_protection_term:
        pp_claim = f"The recommended payment method '{pp_info.method_name}' offers purchase protection of {pp_info.purchase_protection_term}."
    await evaluator.verify(
        claim=pp_claim,
        node=pp_leaf,
        sources=pp_sources or None,
        additional_instruction="Verify purchase protection duration from issuer/card policy pages."
    )

    # Extended warranty verification via URLs
    ew_info = _find_payment_with_extended_warranty(extr)
    ew_leaf = evaluator.add_leaf(
        id="Extended_Warranty_Explained",
        desc="Explains applicable extended-warranty coverage consistent with constraints (typically +1 year on warranties of 3 years or less) and ties it to the plan.",
        parent=pay_node,
        critical=True
    )
    ew_sources = []
    if extr.citations and extr.citations.card_protections_urls:
        ew_sources.extend(extr.citations.card_protections_urls)
    if ew_info and ew_info.urls:
        ew_sources.extend(ew_info.urls)
    ew_claim = "The recommended payment method provides extended warranty, typically +1 year on warranties of 3 years or less."
    if ew_info and ew_info.extended_warranty_term:
        ew_claim = f"The recommended payment method '{ew_info.method_name}' provides an extended warranty of {ew_info.extended_warranty_term}."
    await evaluator.verify(
        claim=ew_claim,
        node=ew_leaf,
        sources=ew_sources or None,
        additional_instruction="Verify extended warranty terms from issuer/card policy pages."
    )


async def build_purchase_timing_subtree(evaluator: Evaluator, parent, extr: ShoppingStrategyExtraction):
    timing_node = evaluator.add_parallel(
        id="Purchase_Timing",
        desc="Recommend when to buy within a stated 2-month window to capture relevant sales/promotions.",
        parent=parent,
        critical=True
    )
    timing = extr.timing or PurchaseTiming()

    evaluator.add_custom_node(
        result=bool(timing.two_month_window),
        id="Two_Month_Window_Assumption_Stated",
        desc="States the assumed 2-month calendar window in early 2026 used for timing recommendations (since the start date is not explicit).",
        parent=timing_node,
        critical=True
    )

    sale_leaf = evaluator.add_leaf(
        id="Relevant_Sale_Periods_Identified_Within_Window",
        desc="Identifies at least one relevant sale/promotional period within the assumed window and connects it to product categories.",
        parent=timing_node,
        critical=True
    )
    sp = _find_sale_period(extr)
    sale_claim = "The answer identifies at least one relevant sale or promotional period within the two-month window."
    if sp:
        cats = ", ".join(sp.applicable_categories) if sp.applicable_categories else "relevant categories"
        sale_claim = f"The '{sp.name}' sale occurs around {sp.date_range} and is relevant to {cats}."
    await evaluator.verify(
        claim=sale_claim,
        node=sale_leaf,
        sources=timing.sale_timing_urls or (extr.citations.sale_timing_urls if extr.citations else None),
        additional_instruction="Verify that the cited sale period falls within the stated two-month window and is relevant to the purchase categories."
    )


async def build_return_policy_subtree(evaluator: Evaluator, parent, extr: ShoppingStrategyExtraction):
    ret_node = evaluator.add_parallel(
        id="Return_Policy",
        desc="Provide standard return windows at selected retailer(s) and evaluate holiday extension applicability.",
        parent=parent,
        critical=True
    )
    rp_item = _find_return_policy_item(extr)
    ret_urls = []
    if extr.citations and extr.citations.return_policy_urls:
        ret_urls.extend(extr.citations.return_policy_urls)
    if rp_item and rp_item.urls:
        ret_urls.extend(rp_item.urls)

    std_leaf = evaluator.add_leaf(
        id="Standard_Return_Windows_Stated",
        desc="States standard return windows for the selected retailer(s).",
        parent=ret_node,
        critical=True
    )
    std_claim = "The answer states the standard return window for at least one of the selected retailers."
    if rp_item and rp_item.retailer and rp_item.standard_return_window:
        std_claim = f"The standard return window at {rp_item.retailer} is {rp_item.standard_return_window}."
    await evaluator.verify(
        claim=std_claim,
        node=std_leaf,
        sources=ret_urls or None,
        additional_instruction="Confirm the return window length from the retailer's policy page."
    )

    evaluator.add_custom_node(
        result=bool(rp_item and rp_item.holiday_extension_applicability),
        id="Holiday_Return_Extension_Evaluated",
        desc="Determines whether planned purchases qualify for any extended holiday return period at selected retailer(s).",
        parent=ret_node,
        critical=True
    )


async def build_total_cost_analysis_subtree(evaluator: Evaluator, parent, extr: ShoppingStrategyExtraction):
    tca_node = evaluator.add_sequential(
        id="Total_Cost_Analysis",
        desc="Compute net total cost using explicit numbers: gross total, discounts/rewards, membership fees (if applicable), shipping/other fees, and final net cost.",
        parent=parent,
        critical=True
    )
    tc = extr.total_cost or TotalCostAnalysis()

    evaluator.add_custom_node(
        result=bool(tc.gross_total or tc.category_totals),
        id="Gross_Purchase_Total_Stated",
        desc="States the gross planned purchase total (using the given ~$550–600 range or a clearly stated point estimate within it) and/or category breakdown used in calculations.",
        parent=tca_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(tc.discounts_rewards_total),
        id="Discounts_and_Rewards_Calculated",
        desc="Quantifies discounts/rewards in dollars (and/or %) with assumptions stated.",
        parent=tca_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(tc.fees_total),
        id="Fees_Calculated",
        desc="Quantifies applicable fees under the plan (membership fee if joined; shipping/minimum-order fees if applicable; delivery fees if applicable).",
        parent=tca_node,
        critical=True
    )

    net_leaf = evaluator.add_leaf(
        id="Net_Total_Computed",
        desc="Computes final net out-of-pocket cost with a clear formula (e.g., Net = Gross − Discounts/Rewards + Fees).",
        parent=tca_node,
        critical=True
    )
    formula_text = tc.formula or "Net = Gross − Discounts/Rewards + Fees"
    net_claim = f"The answer computes the final net cost using a clear formula such as '{formula_text}' and provides a net total value."
    await evaluator.verify(
        claim=net_claim,
        node=net_leaf,
        additional_instruction="Confirm that the answer explicitly shows the formula and a numeric net total."
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
    Evaluate the shopping strategy optimization answer using a hierarchical verification tree.
    """
    # Initialize evaluator with a critical root (parallel aggregation)
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_strategy(),
        template_class=ShoppingStrategyExtraction,
        extraction_name="shopping_strategy_extraction"
    )

    # Build verification tree according to rubric
    # Root is critical; all children under root must be critical per framework constraint
    await build_recency_assumption(evaluator, root, extracted)
    await build_citations_subtree(evaluator, root, extracted)
    await build_retailer_selection_and_shipping_subtree(evaluator, root, extracted)
    await build_membership_program_decision_subtree(evaluator, root, extracted)
    await build_payment_method_and_protections_subtree(evaluator, root, extracted)
    await build_purchase_timing_subtree(evaluator, root, extracted)
    await build_return_policy_subtree(evaluator, root, extracted)
    await build_total_cost_analysis_subtree(evaluator, root, extracted)

    # Add custom info for transparency
    evaluator.add_custom_info(
        info={"current_date_context": "early 2026", "task_focus": "minimize net cost while maximizing protections and returns"},
        info_type="context",
        info_name="evaluation_context"
    )

    # Return summary
    return evaluator.get_summary()