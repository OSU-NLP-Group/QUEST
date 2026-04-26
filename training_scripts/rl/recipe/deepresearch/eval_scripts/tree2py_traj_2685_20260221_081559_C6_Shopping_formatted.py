import asyncio
import logging
import re
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# ------------------------------------------------------------
# Task Constants
# ------------------------------------------------------------
TASK_ID = "pharmacy_chain_selection"
TASK_DESCRIPTION = (
    "A 60-year-old individual is looking to optimize their healthcare and shopping expenses by selecting a pharmacy that offers "
    "the best combination of senior discounts and convenient services. They require a pharmacy chain that provides: "
    "(1) A senior discount program available on the first Tuesday of every month for customers aged 55 or older, and "
    "(2) Immunization services (such as flu shots) that accept walk-in visits without requiring appointments. "
    "Additionally, they are interested in learning about: "
    "(3) Whether the pharmacy has any loyalty or membership programs with annual costs under $100, "
    "(4) For comparison purposes, the membership structure of a major warehouse club (specifically one that offers a basic tier around $60-70 annually and a premium tier that provides cashback rewards), "
    "(5) Information about extended holiday return policies at major retailers (specifically for items purchased between November and December 2025), and "
    "(6) The minimum order requirements for free store pickup service at a major grocery retailer. "
    "Please identify the pharmacy chain that meets the required criteria (items 1-2) and provide information about the additional comparison points (items 3-6), including specific details such as discount percentages, age requirements, membership costs, break-even points for premium memberships, and return deadlines. "
    "Include reference URLs to support your findings."
)


# ------------------------------------------------------------
# Helper parsing utilities
# ------------------------------------------------------------
def parse_dollar_amount(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Extract the first numeric token which likely represents dollars
    m = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    try:
        return float(m.group(1)) if m else None
    except Exception:
        return None


def parse_percentage(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    try:
        return float(m.group(1)) if m else None
    except Exception:
        return None


def join_list(items: Optional[List[str]]) -> str:
    if not items:
        return ""
    return "; ".join([i.strip() for i in items if i and i.strip()])


# ------------------------------------------------------------
# Data Models (Extraction)
# ------------------------------------------------------------
class SeniorDiscountInfo(BaseModel):
    day_policy_text: Optional[str] = None
    age_requirement_text: Optional[str] = None
    discount_percentage_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ImmunizationInfo(BaseModel):
    walk_in_policy_text: Optional[str] = None
    service_types: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class MembershipProgramInfo(BaseModel):
    program_name: Optional[str] = None
    annual_cost_text: Optional[str] = None
    benefits: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class PharmacyExtraction(BaseModel):
    chain_name: Optional[str] = None
    senior_discount: Optional[SeniorDiscountInfo] = None
    immunization: Optional[ImmunizationInfo] = None
    membership_program: Optional[MembershipProgramInfo] = None


class WarehouseClubExtraction(BaseModel):
    club_name: Optional[str] = None
    basic_tier_name: Optional[str] = None
    basic_tier_cost_text: Optional[str] = None
    premium_tier_name: Optional[str] = None
    premium_tier_cost_text: Optional[str] = None
    reward_percentage_text: Optional[str] = None
    annual_reward_cap_text: Optional[str] = None
    break_even_spend_estimate_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HolidayReturnPolicyExtraction(BaseModel):
    retailer_name: Optional[str] = None
    purchase_window_start_text: Optional[str] = None
    purchase_window_end_text: Optional[str] = None
    return_deadline_in_store_text: Optional[str] = None
    url: Optional[str] = None


class StorePickupPolicyExtraction(BaseModel):
    retailer_name: Optional[str] = None
    minimum_amount_text: Optional[str] = None
    under_minimum_fee_text: Optional[str] = None
    url: Optional[str] = None


# ------------------------------------------------------------
# Extraction Prompts
# ------------------------------------------------------------
def prompt_extract_pharmacy() -> str:
    return (
        "Extract the pharmacy chain and key policy details directly from the answer. "
        "Return a JSON with fields: "
        "chain_name, senior_discount, immunization, membership_program. "
        "For senior_discount, include: day_policy_text (e.g., 'first Tuesday of every month'), "
        "age_requirement_text (e.g., 'age 55+'), discount_percentage_text (e.g., '20% off'), "
        "and urls (array of URLs supporting the senior discount). "
        "For immunization, include: walk_in_policy_text (e.g., 'walk-ins accepted'), "
        "service_types (array, e.g., ['flu', 'COVID-19']), and urls (array of URLs supporting immunization info). "
        "For membership_program, include: program_name, annual_cost_text, benefits (array), and urls (array). "
        "If any field is not stated, return null or an empty array accordingly."
    )


def prompt_extract_warehouse() -> str:
    return (
        "Extract membership details for a major warehouse club mentioned in the answer. "
        "Return fields: club_name, basic_tier_name, basic_tier_cost_text, premium_tier_name, premium_tier_cost_text, "
        "reward_percentage_text (e.g., '2%'), annual_reward_cap_text (e.g., '$1,000 cap'), "
        "break_even_spend_estimate_text (if the answer provides an estimate), and urls (array of references). "
        "If any field is not stated, return null."
    )


def prompt_extract_holiday_return() -> str:
    return (
        "Extract an extended holiday return policy from a major retailer mentioned in the answer. "
        "Return fields: retailer_name, purchase_window_start_text, purchase_window_end_text, "
        "return_deadline_in_store_text, and url (single URL reference). "
        "If any field is not stated, return null."
    )


def prompt_extract_store_pickup() -> str:
    return (
        "Extract store pickup requirements for a major grocery retailer mentioned in the answer. "
        "Return fields: retailer_name, minimum_amount_text (e.g., '$35'), under_minimum_fee_text (e.g., '$6.99 fee'), "
        "and url (single URL reference). "
        "If any field is not stated, return null."
    )


# ------------------------------------------------------------
# Verification Builders
# ------------------------------------------------------------
async def build_pharmacy_chain_verification(
    evaluator: Evaluator,
    parent_node,
    pharm: PharmacyExtraction,
) -> None:
    # Pharmacy Chain Identification (non-critical container)
    pc_node = evaluator.add_parallel(
        id="Pharmacy_Chain_Identification",
        desc="Correctly identify a major pharmacy chain meeting all specified criteria",
        parent=parent_node,
        critical=False,
    )

    chain_name = pharm.chain_name or "the pharmacy"

    # Senior Discount Core (critical child with only critical leaves)
    senior = pharm.senior_discount or SeniorDiscountInfo()
    sd_core = evaluator.add_parallel(
        id="Senior_Discount_Core",
        desc="Verify the pharmacy offers senior discounts on the first Tuesday of each month",
        parent=pc_node,
        critical=True,
    )

    sd_sources_exist = evaluator.add_custom_node(
        result=bool(senior.urls),
        id="Reference_URL_Senior_Discount",
        desc="Provide a valid URL reference supporting the senior discount information",
        parent=sd_core,
        critical=True,
    )

    sd_day_node = evaluator.add_leaf(
        id="Discount_Day_Accuracy",
        desc="Confirm the discount is available on the first Tuesday of each month",
        parent=sd_core,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{chain_name} offers a seniors discount on the first Tuesday of each month.",
        node=sd_day_node,
        sources=senior.urls,
        additional_instruction="Consider phrasing variants like 'first Tuesday of every month'. Verify the page explicitly states the timing.",
    )

    sd_age_node = evaluator.add_leaf(
        id="Age_Requirement_Verification",
        desc="Verify the minimum age requirement for the senior discount (must be 55 or older)",
        parent=sd_core,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The seniors discount at {chain_name} requires age 55 or older.",
        node=sd_age_node,
        sources=senior.urls,
        additional_instruction="Allow variants like '55+' or 'ages 55 and up'.",
    )

    # Senior Discount Extras (non-critical)
    sd_extras = evaluator.add_parallel(
        id="Senior_Discount_Extras",
        desc="Additional details about the senior discount (non-critical)",
        parent=pc_node,
        critical=False,
    )
    sd_pct_node = evaluator.add_leaf(
        id="Discount_Percentage",
        desc="State the percentage or amount of discount offered",
        parent=sd_extras,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The seniors discount amount is {senior.discount_percentage_text or ''}.",
        node=sd_pct_node,
        sources=senior.urls if senior.urls else None,
        additional_instruction="Verify the discount amount/percentage if specified; allow text variants like 'up to 20%'.",
    )

    # Immunization Service Core (critical)
    immun = pharm.immunization or ImmunizationInfo()
    imm_core = evaluator.add_parallel(
        id="Immunization_Service_Core",
        desc="Verify the pharmacy provides immunization services with walk-in availability",
        parent=pc_node,
        critical=True,
    )

    imm_sources_exist = evaluator.add_custom_node(
        result=bool(immun.urls),
        id="Reference_URL_Immunization",
        desc="Provide a valid URL reference supporting immunization service information",
        parent=imm_core,
        critical=True,
    )

    walkin_node = evaluator.add_leaf(
        id="Walk_In_Availability",
        desc="Confirm walk-in immunization services are available without appointment",
        parent=imm_core,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{chain_name} offers immunization services with walk-ins accepted without required appointments.",
        node=walkin_node,
        sources=immun.urls,
        additional_instruction="Look for phrasing like 'walk-in available', 'no appointment needed', or 'same-day'.",
    )

    # Immunization Extras (non-critical)
    imm_extras = evaluator.add_parallel(
        id="Immunization_Extras",
        desc="List types of immunizations offered (non-critical)",
        parent=pc_node,
        critical=False,
    )
    svc_types_text = join_list(immun.service_types)
    svc_types_node = evaluator.add_leaf(
        id="Service_Types",
        desc="List types of immunizations offered (e.g., flu shots, COVID-19)",
        parent=imm_extras,
        critical=False,
    )
    await evaluator.verify(
        claim=f"Immunization services include: {svc_types_text}.",
        node=svc_types_node,
        sources=immun.urls if immun.urls else None,
        additional_instruction="Partial matches acceptable (e.g., flu or COVID-19).",
    )

    # Membership Program Verification (non-critical)
    mem = pharm.membership_program or MembershipProgramInfo()
    mem_node = evaluator.add_parallel(
        id="Membership_Program_Verification",
        desc="Verify the pharmacy has a loyalty or membership program",
        parent=pc_node,
        critical=False,
    )

    mem_sources_exist = evaluator.add_custom_node(
        result=bool(mem.urls),
        id="Membership_URL_Presence",
        desc="Reference URL for pharmacy membership/loyalty program is present",
        parent=mem_node,
        critical=False,
    )

    prog_name_node = evaluator.add_leaf(
        id="Program_Name",
        desc="Identify the name of the loyalty or membership program",
        parent=mem_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The pharmacy's loyalty/membership program is called '{mem.program_name or ''}'.",
        node=prog_name_node,
        sources=mem.urls if mem.urls else None,
        additional_instruction="Verify naming (e.g., 'myWalgreens', 'Extracare'). Allow reasonable variants.",
    )

    annual_cost_node = evaluator.add_leaf(
        id="Annual_Cost_Verification",
        desc="Verify the annual membership cost is under $100",
        parent=mem_node,
        critical=False,
    )
    await evaluator.verify(
        claim="The annual membership cost for the pharmacy's program is under $100.",
        node=annual_cost_node,
        sources=mem.urls if mem.urls else None,
        additional_instruction="If multiple tiers exist, verify whether at least one annual tier is below $100.",
    )

    benefits_node = evaluator.add_leaf(
        id="Membership_Benefits",
        desc="List key benefits of the membership program",
        parent=mem_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"Key membership benefits include: {join_list(mem.benefits)}.",
        node=benefits_node,
        sources=mem.urls if mem.urls else None,
        additional_instruction="Benefits can include rewards, percent-off, cashback, pharmacy savings, etc.",
    )


async def build_warehouse_verification(
    evaluator: Evaluator,
    parent_node,
    wh: WarehouseClubExtraction,
) -> None:
    alt_node = evaluator.add_parallel(
        id="Alternative_Shopping_Analysis",
        desc="Analyze an alternative retail chain's membership program for comparison",
        parent=parent_node,
        critical=False,
    )

    wh_main = evaluator.add_parallel(
        id="Warehouse_Club_Identification",
        desc="Identify a warehouse club that offers two membership tiers",
        parent=alt_node,
        critical=False,
    )

    wh_sources_exist = evaluator.add_custom_node(
        result=bool(wh.urls),
        id="Reference_URL_Warehouse",
        desc="Provide valid URL references supporting warehouse club membership information",
        parent=wh_main,
        critical=False,
    )

    # Basic tier details (non-critical)
    basic_node = evaluator.add_parallel(
        id="Basic_Membership_Details",
        desc="Provide details about the basic membership tier",
        parent=wh_main,
        critical=False,
    )

    basic_cost_leaf = evaluator.add_leaf(
        id="Basic_Tier_Cost",
        desc="State the annual cost of the basic membership",
        parent=basic_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The basic membership annual cost is {wh.basic_tier_cost_text or ''}.",
        node=basic_cost_leaf,
        sources=wh.urls if wh.urls else None,
        additional_instruction="Verify cost of the entry/basic tier (e.g., $60-$70).",
    )

    basic_name_leaf = evaluator.add_leaf(
        id="Basic_Tier_Name",
        desc="Provide the name of the basic membership tier",
        parent=basic_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The basic tier is named '{wh.basic_tier_name or ''}'.",
        node=basic_name_leaf,
        sources=wh.urls if wh.urls else None,
        additional_instruction="Verify the name (e.g., Gold Star, Club).",
    )

    # Premium tier details (non-critical)
    premium_node = evaluator.add_parallel(
        id="Premium_Membership_Details",
        desc="Provide details about the premium membership tier",
        parent=wh_main,
        critical=False,
    )

    premium_cost_leaf = evaluator.add_leaf(
        id="Premium_Tier_Cost",
        desc="State the annual cost of the premium membership",
        parent=premium_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The premium membership annual cost is {wh.premium_tier_cost_text or ''}.",
        node=premium_cost_leaf,
        sources=wh.urls if wh.urls else None,
        additional_instruction="Verify cost for the premium tier (e.g., $110-$120).",
    )

    reward_pct_leaf = evaluator.add_leaf(
        id="Reward_Percentage",
        desc="State the cashback or reward percentage for premium members",
        parent=premium_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The premium membership reward percentage is {wh.reward_percentage_text or ''}.",
        node=reward_pct_leaf,
        sources=wh.urls if wh.urls else None,
        additional_instruction="Verify cashback/reward percent (e.g., 2%).",
    )

    reward_cap_leaf = evaluator.add_leaf(
        id="Annual_Reward_Cap",
        desc="State the maximum annual reward amount",
        parent=premium_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The maximum annual reward amount is {wh.annual_reward_cap_text or ''}.",
        node=reward_cap_leaf,
        sources=wh.urls if wh.urls else None,
        additional_instruction="Verify any stated cap (e.g., $1,000 annual cap).",
    )

    # Break-even calculation (non-critical, simple verify)
    break_even_leaf = evaluator.add_leaf(
        id="Break_Even_Calculation",
        desc="Calculate or state the annual spending needed to break even on premium membership upgrade cost",
        parent=premium_node,
        critical=False,
    )

    basic_cost = parse_dollar_amount(wh.basic_tier_cost_text)
    premium_cost = parse_dollar_amount(wh.premium_tier_cost_text)
    reward_pct = parse_percentage(wh.reward_percentage_text)
    if basic_cost is not None and premium_cost is not None and reward_pct:
        upgrade_diff = max(0.0, premium_cost - basic_cost)
        # reward_pct is like 2 -> convert to 0.02
        pct_decimal = reward_pct / 100.0
        break_even_spend = int(round(upgrade_diff / pct_decimal)) if pct_decimal > 0 else None
    else:
        break_even_spend = None

    be_text = (
        f"approximately ${break_even_spend:,} per year"
        if break_even_spend is not None else
        (wh.break_even_spend_estimate_text or "an appropriate annual spend to break even")
    )

    await evaluator.verify(
        claim=f"The break-even spending for the premium tier is {be_text}.",
        node=break_even_leaf,
        sources=None,
        additional_instruction=(
            "Judge the arithmetic/logic: break-even ≈ (premium - basic) / (reward %). Accept reasonable rounding."
        ),
    )


async def build_holiday_return_verification(
    evaluator: Evaluator,
    parent_node,
    rp: HolidayReturnPolicyExtraction,
) -> None:
    holiday_node = evaluator.add_parallel(
        id="Holiday_Return_Policy_Analysis",
        desc="Analyze extended holiday return policies at major retailers",
        parent=parent_node,
        critical=False,
    )

    window_node = evaluator.add_parallel(
        id="Retailer_Return_Window",
        desc="Identify a major retailer with extended holiday return policy",
        parent=holiday_node,
        critical=False,
    )

    rp_sources_exist = evaluator.add_custom_node(
        result=bool(rp.url),
        id="Reference_URL_Return_Policy",
        desc="Provide a valid URL reference supporting the return policy information",
        parent=window_node,
        critical=False,
    )

    start_leaf = evaluator.add_leaf(
        id="Purchase_Window_Start",
        desc="State the start date of the eligible purchase window for extended returns",
        parent=window_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The eligible holiday purchase window starts on {rp.purchase_window_start_text or ''}.",
        node=start_leaf,
        sources=rp.url if rp.url else None,
        additional_instruction="Verify the start date applies to the extended holiday return policy.",
    )

    end_leaf = evaluator.add_leaf(
        id="Purchase_Window_End",
        desc="State the end date of the eligible purchase window for extended returns",
        parent=window_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The eligible holiday purchase window ends on {rp.purchase_window_end_text or ''}.",
        node=end_leaf,
        sources=rp.url if rp.url else None,
        additional_instruction="Verify the end date applies to the extended holiday return policy.",
    )

    deadline_leaf = evaluator.add_leaf(
        id="Return_Deadline_In_Store",
        desc="State the deadline for in-store returns for holiday purchases",
        parent=window_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The in-store return deadline for holiday purchases is {rp.return_deadline_in_store_text or ''}.",
        node=deadline_leaf,
        sources=rp.url if rp.url else None,
        additional_instruction="Verify that the deadline applies to purchases made in Nov-Dec 2025.",
    )

    retailer_name_leaf = evaluator.add_leaf(
        id="Retailer_Name",
        desc="Identify the specific retailer offering this extended return policy",
        parent=window_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The extended holiday return policy is offered by {rp.retailer_name or ''}.",
        node=retailer_name_leaf,
        sources=rp.url if rp.url else None,
        additional_instruction="Verify the retailer name as listed on the policy page.",
    )


async def build_store_pickup_verification(
    evaluator: Evaluator,
    parent_node,
    sp: StorePickupPolicyExtraction,
) -> None:
    pickup_node = evaluator.add_parallel(
        id="Store_Pickup_Policy_Verification",
        desc="Verify store pickup requirements at a major grocery retailer",
        parent=parent_node,
        critical=False,
    )

    min_req_node = evaluator.add_parallel(
        id="Minimum_Order_Requirement",
        desc="Identify the minimum order amount for free store pickup at a major retailer",
        parent=pickup_node,
        critical=False,
    )

    sp_sources_exist = evaluator.add_custom_node(
        result=bool(sp.url),
        id="Reference_URL_Pickup",
        desc="Provide a valid URL reference supporting the pickup policy",
        parent=min_req_node,
        critical=False,
    )

    min_amount_leaf = evaluator.add_leaf(
        id="Minimum_Amount",
        desc="State the minimum order amount required (e.g., $35)",
        parent=min_req_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The minimum order amount for free store pickup is {sp.minimum_amount_text or ''}.",
        node=min_amount_leaf,
        sources=sp.url if sp.url else None,
        additional_instruction="Verify the minimum order threshold for free pickup, if present.",
    )

    under_min_fee_leaf = evaluator.add_leaf(
        id="Under_Minimum_Fee",
        desc="State the fee charged for orders below the minimum amount",
        parent=min_req_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The fee for orders below the minimum is {sp.under_minimum_fee_text or ''}.",
        node=under_min_fee_leaf,
        sources=sp.url if sp.url else None,
        additional_instruction="Verify the fee (pickup surcharge) if stated.",
    )

    retailer_name_pickup_leaf = evaluator.add_leaf(
        id="Retailer_Name_Pickup",
        desc="Identify which major retailer has this pickup policy",
        parent=min_req_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The pickup policy applies to {sp.retailer_name or ''}.",
        node=retailer_name_pickup_leaf,
        sources=sp.url if sp.url else None,
        additional_instruction="Verify retailer name on the pickup policy page.",
    )


# ------------------------------------------------------------
# Main Evaluation Entry
# ------------------------------------------------------------
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root non-critical, parallel aggregation
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

    # Extract data groups
    pharm_info = await evaluator.extract(
        prompt=prompt_extract_pharmacy(),
        template_class=PharmacyExtraction,
        extraction_name="pharmacy_chain_info",
    )

    warehouse_info = await evaluator.extract(
        prompt=prompt_extract_warehouse(),
        template_class=WarehouseClubExtraction,
        extraction_name="warehouse_club_info",
    )

    holiday_return_info = await evaluator.extract(
        prompt=prompt_extract_holiday_return(),
        template_class=HolidayReturnPolicyExtraction,
        extraction_name="holiday_return_policy",
    )

    store_pickup_info = await evaluator.extract(
        prompt=prompt_extract_store_pickup(),
        template_class=StorePickupPolicyExtraction,
        extraction_name="store_pickup_policy",
    )

    # Build verification tree per rubric
    await build_pharmacy_chain_verification(evaluator, root, pharm_info)
    await build_warehouse_verification(evaluator, root, warehouse_info)
    await build_holiday_return_verification(evaluator, root, holiday_return_info)
    await build_store_pickup_verification(evaluator, root, store_pickup_info)

    # Return evaluation summary
    return evaluator.get_summary()