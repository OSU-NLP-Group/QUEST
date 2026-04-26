import asyncio
import logging
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "grand_canyon_may2026_trip_planning"
TASK_DESCRIPTION = (
    "A group of 4 hikers is planning a Grand Canyon backpacking trip starting on May 15, 2026. "
    "Their planned itinerary includes 2 consecutive nights at Bright Angel Campground followed by 1 night at Cottonwood Campground (all below-rim camping). "
    "They want to apply for the early access lottery through Recreation.gov.\n\n"
    "Provide the following information:\n\n"
    "1. Lottery Application Window: Exact opening date and closing date/time (with timezone) for the early access lottery application for this May 2026 trip.\n"
    "2. Total Permit Cost: Total cost for this backcountry permit, including the basic permit fee and all nightly charges for the 4-person group staying 3 nights below the rim.\n"
    "3. Latest Refund Deadline: Latest date to cancel the permit and receive a refund of the nightly charges (assuming the permit has not been printed).\n\n"
    "For each answer, provide dates, costs, or deadlines along with reference URL(s) from official Grand Canyon National Park or Recreation.gov sources."
)

# Ground truth / constraints used for verification
EXPECTED_LOTTERY_OPENING_DATE = "December 16, 2025"
EXPECTED_LOTTERY_CLOSING_DATETIME_TZ = "January 1, 2026 at 5:00 PM MST"

EXPECTED_GROUP_SIZE = 4
EXPECTED_NIGHTS_BELOW_RIM = 3
EXPECTED_BRIGHT_ANGEL_NIGHTS = 2
EXPECTED_COTTONWOOD_NIGHTS = 1
EXPECTED_BELOW_RIM_RATE_USD = 15.0
EXPECTED_BASIC_PERMIT_FEE_USD = 10.0  # subject to waiver for early-access lottery winners in their timeslot

TRIP_START_DATE = date(2026, 5, 15)
REFUND_RULE_DAYS_BEFORE = 30
EXPECTED_REFUND_DEADLINE = TRIP_START_DATE - timedelta(days=REFUND_RULE_DAYS_BEFORE)  # April 15, 2026


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LotteryInfo(BaseModel):
    opening_date: Optional[str] = None
    closing_date_time: Optional[str] = None  # include both date and time, e.g., "Jan 1, 2026 at 5:00 PM MST"
    timezone: Optional[str] = None  # e.g., "MST"
    urls: List[str] = Field(default_factory=list)  # official references


class CostInfo(BaseModel):
    group_size: Optional[str] = None  # e.g., "4"
    nights_below_rim_total: Optional[str] = None  # e.g., "3"
    bright_angel_nights: Optional[str] = None  # e.g., "2"
    cottonwood_nights: Optional[str] = None  # e.g., "1"
    nightly_rate_below_rim: Optional[str] = None  # e.g., "$15 per person per night"
    basic_permit_fee: Optional[str] = None  # e.g., "$10"
    basic_fee_waiver_mentioned: Optional[bool] = None  # True if they mention waiver condition
    entrance_fees_included_in_total: Optional[bool] = None  # True if they included entrance fees in the total
    total_permit_cost_reported: Optional[str] = None  # e.g., "$180" or "$190"
    urls: List[str] = Field(default_factory=list)  # official references for fee policies


class RefundInfo(BaseModel):
    refund_deadline_date: Optional[str] = None  # e.g., "April 15, 2026"
    printed_condition_acknowledged: Optional[bool] = None  # True if they mention "not printed" condition
    urls: List[str] = Field(default_factory=list)  # official references for refund policy


class TripPlanningExtraction(BaseModel):
    lottery: Optional[LotteryInfo] = None
    cost: Optional[CostInfo] = None
    refund: Optional[RefundInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_planning() -> str:
    return """
    Extract the trip-planning information exactly as presented in the answer. Structure the output into three sections: lottery, cost, and refund.

    lottery:
      - opening_date: The stated early access lottery opening date for May 2026 trips (string).
      - closing_date_time: The stated early access lottery closing date and time with timezone (string).
      - timezone: The timezone abbreviation provided for the closing time (string).
      - urls: An array of official reference URLs (NPS or Recreation.gov) that the answer cites for the lottery window. Extract actual URLs only.

    cost:
      - group_size: The number of people used by the answer to compute nightly charges (string, e.g., "4").
      - nights_below_rim_total: The total number of below-rim nights used (string, e.g., "3").
      - bright_angel_nights: The number of Bright Angel Campground nights used (string, e.g., "2").
      - cottonwood_nights: The number of Cottonwood Campground nights used (string, e.g., "1").
      - nightly_rate_below_rim: The below-rim nightly rate used (string, e.g., "$15 per person per night").
      - basic_permit_fee: The basic permit fee amount the answer uses or cites (string, e.g., "$10").
      - basic_fee_waiver_mentioned: A boolean indicating whether the answer acknowledges the waiver condition for early-access lottery winners booking in their timeslot.
      - entrance_fees_included_in_total: A boolean indicating whether the answer includes park entrance fees in the total permit cost (should generally be false if excluded).
      - total_permit_cost_reported: The total permit cost reported in the answer (string, e.g., "$180" or "$190").
      - urls: An array of official reference URLs (NPS or Recreation.gov) used to justify fee amounts/rules. Extract actual URLs only.

    refund:
      - refund_deadline_date: The latest cancellation date stated to receive a refund of nightly charges (string).
      - printed_condition_acknowledged: A boolean indicating whether the answer acknowledges the condition that the permit has not been printed (printing affects cancellation/refund).
      - urls: An array of official reference URLs (NPS or Recreation.gov) for the refund timing rule and printing-related restriction. Extract actual URLs only.

    RULES:
    - Extract only what is explicitly stated in the answer.
    - Return null for any missing field.
    - For URLs, extract only valid complete URLs; include markdown-linked URLs by extracting the target.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_official_url(url: str) -> bool:
    if not url:
        return False
    u = url.lower()
    return ("nps.gov" in u) or ("recreation.gov" in u)


def are_official_urls(urls: List[str]) -> bool:
    if not urls:
        return False
    return all(is_official_url(u) for u in urls)


def parse_int_from_str(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def parse_currency_to_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    # Find first number (allow decimals)
    m = re.search(r"(\d+(?:\.\d+)?)", s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def format_mmddyyyy(d: date) -> str:
    # e.g., "April 15, 2026"
    return d.strftime("%B %-d, %Y") if hasattr(d, "strftime") else str(d)


def sources_or_none(urls: List[str]) -> Optional[List[str]]:
    return urls if urls else None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_lottery_window(
    evaluator: Evaluator,
    parent_node,
    lottery: Optional[LotteryInfo],
) -> None:
    # Create parent aggregator node for Lottery Application Window (critical)
    lot_node = evaluator.add_parallel(
        id="Lottery_Application_Window",
        desc="Provides the exact early access lottery opening date and closing date/time with timezone for a May 2026 trip, supported by official sources.",
        parent=parent_node,
        critical=True,
    )

    urls = lottery.urls if lottery and lottery.urls else []

    # Leaf: Lottery Opening Date (must match Dec 16, 2025)
    open_leaf = evaluator.add_leaf(
        id="Lottery_Opening_Date",
        desc="States the lottery application opening date for May 2026 trips",
        parent=lot_node,
        critical=True,
    )
    open_claim = f"For May 2026 Grand Canyon backcountry trips, the early access lottery application opens on {EXPECTED_LOTTERY_OPENING_DATE}."
    await evaluator.verify(
        claim=open_claim,
        node=open_leaf,
        sources=sources_or_none(urls),
        additional_instruction="Verify using the official NPS or Recreation.gov sources that this opening date is correct for May 2026 trips."
    )

    # Leaf: Lottery Closing Date/Time/Timezone (must match Jan 1, 2026 at 5:00 PM MST)
    close_leaf = evaluator.add_leaf(
        id="Lottery_Closing_DateTime_Timezone",
        desc="States the lottery application closing date/time/timezone for May 2026 trips",
        parent=lot_node,
        critical=True,
    )
    close_claim = (
        f"For May 2026 trips, the early access lottery application closes on {EXPECTED_LOTTERY_CLOSING_DATETIME_TZ}."
    )
    await evaluator.verify(
        claim=close_claim,
        node=close_leaf,
        sources=sources_or_none(urls),
        additional_instruction="Confirm the exact closing date, time, and timezone from official sources."
    )

    # Leaf: Reference URLs provided and official
    lot_urls_leaf = evaluator.add_custom_node(
        result=are_official_urls(urls),
        id="Lottery_Window_Reference_URLs",
        desc="Provides official Grand Canyon NPS or Recreation.gov URL(s) supporting the stated lottery window",
        parent=lot_node,
        critical=True,
    )


async def verify_total_permit_cost(
    evaluator: Evaluator,
    parent_node,
    cost: Optional[CostInfo],
) -> None:
    cost_node = evaluator.add_parallel(
        id="Total_Permit_Cost",
        desc="Computes the total backcountry permit cost for 4 people, 3 nights below the rim, consistent with fee rules and supported by official sources.",
        parent=parent_node,
        critical=True,
    )

    urls = cost.urls if cost and cost.urls else []

    # Leaf: Nightly charges inputs correct (4 people, 3 nights, below-rim, 2 BA + 1 Cottonwood)
    inputs_leaf = evaluator.add_leaf(
        id="Nightly_Charges_Inputs_Correct",
        desc="Uses the correct nightly-charge inputs: 4 people, 3 nights total (2 Bright Angel + 1 Cottonwood), and below-rim status for all nights.",
        parent=cost_node,
        critical=True,
    )
    inputs_claim = (
        "The nightly charges are calculated for 4 people staying 3 nights all below the rim "
        "(specifically 2 nights at Bright Angel Campground and 1 night at Cottonwood Campground)."
    )
    await evaluator.verify(
        claim=inputs_claim,
        node=inputs_leaf,
        additional_instruction="Verify the answer explicitly uses these inputs in the cost calculation."
    )

    # Leaf: Nightly rate correct ($15 pppn below rim) and supported by official sources
    rate_leaf = evaluator.add_leaf(
        id="Nightly_Rate_Correct",
        desc="Applies the correct below-rim nightly rate ($15 per person per night).",
        parent=cost_node,
        critical=True,
    )
    rate_claim = "The below-rim nightly rate is $15 per person per night according to official policy."
    await evaluator.verify(
        claim=rate_claim,
        node=rate_leaf,
        sources=sources_or_none(urls),
        additional_instruction="Check official Grand Canyon NPS or Recreation.gov pages for the below-rim nightly rate."
    )

    # Parallel sub-node: Basic permit fee handled per policy (split into two critical leaves)
    fee_policy_node = evaluator.add_parallel(
        id="Basic_Permit_Fee_Handled_Per_Policy",
        desc="Handles the basic permit fee per policy ($10) and acknowledges the waiver for early-access lottery winners booking in their timeslot.",
        parent=cost_node,
        critical=True,
    )

    # Leaf: Basic permit fee amount correct ($10)
    fee_amount_leaf = evaluator.add_leaf(
        id="Basic_Permit_Fee_Amount_Correct",
        desc="Recognizes the basic permit fee is $10 per permit.",
        parent=fee_policy_node,
        critical=True,
    )
    fee_amount_claim = "The basic Grand Canyon backcountry permit fee is $10 per permit."
    await evaluator.verify(
        claim=fee_amount_claim,
        node=fee_amount_leaf,
        sources=sources_or_none(urls),
        additional_instruction="Verify the basic permit fee amount from official sources."
    )

    # Leaf: Waiver condition acknowledged for early-access lottery winners booking in timeslot
    fee_waiver_leaf = evaluator.add_leaf(
        id="Basic_Permit_Fee_Waiver_Acknowledged",
        desc="Acknowledges the waiver condition for early-access lottery winners booking in their timeslot.",
        parent=fee_policy_node,
        critical=True,
    )
    waiver_claim = (
        "Early-access lottery winners who book within their assigned timeslot have the $10 basic permit fee waived."
    )
    await evaluator.verify(
        claim=waiver_claim,
        node=fee_waiver_leaf,
        sources=sources_or_none(urls),
        additional_instruction="Check official sources for the waiver rule tied to the early-access lottery timeslot."
    )

    # Leaf: Excludes entrance fees from total
    excludes_leaf = evaluator.add_leaf(
        id="Excludes_Entrance_Fees",
        desc="Does not include park entrance fees in the backcountry permit total.",
        parent=cost_node,
        critical=True,
    )
    excludes_claim = "The backcountry permit total reported excludes park entrance fees (entrance fees are separate)."
    await evaluator.verify(
        claim=excludes_claim,
        node=excludes_leaf,
        additional_instruction="Verify that the answer excludes entrance fees from the permit total calculation."
    )

    # Leaf: Arithmetic total correct
    # Compute expected totals from extracted values (accept either with fee or waived fee)
    def compute_expected_totals() -> Optional[List[float]]:
        if not cost:
            return None
        group_size = parse_int_from_str(cost.group_size) or EXPECTED_GROUP_SIZE
        nights_total = (
            parse_int_from_str(cost.nights_below_rim_total)
            or (
                (parse_int_from_str(cost.bright_angel_nights) or 0)
                + (parse_int_from_str(cost.cottonwood_nights) or 0)
            )
            or EXPECTED_NIGHTS_BELOW_RIM
        )
        nightly_rate = parse_currency_to_float(cost.nightly_rate_below_rim) or EXPECTED_BELOW_RIM_RATE_USD
        base_fee = parse_currency_to_float(cost.basic_permit_fee) or EXPECTED_BASIC_PERMIT_FEE_USD

        nightly_charges = group_size * nights_total * nightly_rate
        # Accept either case: waiver applied or not applied (depending on scenario stated)
        total_with_fee = base_fee + nightly_charges
        total_waived = nightly_charges
        return [round(total_with_fee, 2), round(total_waived, 2)]

    expected_totals = compute_expected_totals()
    reported_total = parse_currency_to_float(cost.total_permit_cost_reported if cost else None)
    arithmetic_pass = False
    breakdown_info = {}
    if expected_totals and reported_total is not None:
        # Allow small tolerance for rounding
        tolerance = 0.5
        arithmetic_pass = any(abs(reported_total - et) <= tolerance for et in expected_totals)
        breakdown_info = {
            "reported_total": reported_total,
            "expected_totals_allowed": expected_totals,
            "tolerance": tolerance,
            "details": {
                "assumed_group_size": parse_int_from_str(cost.group_size) if cost else None,
                "assumed_nights_total": parse_int_from_str(cost.nights_below_rim_total) if cost else None,
                "assumed_bright_angel_nights": parse_int_from_str(cost.bright_angel_nights) if cost else None,
                "assumed_cottonwood_nights": parse_int_from_str(cost.cottonwood_nights) if cost else None,
                "assumed_nightly_rate": parse_currency_to_float(cost.nightly_rate_below_rim) if cost else None,
                "assumed_basic_fee": parse_currency_to_float(cost.basic_permit_fee) if cost else None,
            },
        }

    # Record arithmetic breakdown for transparency
    evaluator.add_custom_info(breakdown_info, info_type="arithmetic_breakdown", info_name="permit_cost_math")

    arithmetic_leaf = evaluator.add_custom_node(
        result=bool(arithmetic_pass),
        id="Arithmetic_Total_Correct",
        desc="The reported total equals (basic permit fee as applicable) + (people × nights × below-rim nightly rate).",
        parent=cost_node,
        critical=True,
    )

    # Leaf: Cost reference URLs official
    cost_urls_leaf = evaluator.add_custom_node(
        result=are_official_urls(urls),
        id="Cost_Reference_URLs",
        desc="Provides official NPS or Recreation.gov URL(s) supporting fee amounts/policies (basic fee/waiver, nightly rate).",
        parent=cost_node,
        critical=True,
    )


async def verify_refund_deadline(
    evaluator: Evaluator,
    parent_node,
    refund: Optional[RefundInfo],
) -> None:
    refund_node = evaluator.add_parallel(
        id="Latest_Refund_Deadline",
        desc="Provides the latest cancellation date to receive a refund of nightly charges, supported by official sources.",
        parent=parent_node,
        critical=True,
    )

    urls = refund.urls if refund and refund.urls else []

    # Leaf: Refund deadline date correct (30 days before May 15, 2026 => April 15, 2026)
    refund_date_leaf = evaluator.add_leaf(
        id="Refund_Deadline_Date_Correct",
        desc="States the latest date to cancel and receive a refund of nightly charges (30 days before start).",
        parent=refund_node,
        critical=True,
    )
    expected_refund_str = EXPECTED_REFUND_DEADLINE.strftime("%B %d, %Y")
    refund_claim = (
        f"The latest date to cancel and receive a refund of nightly charges is {expected_refund_str} "
        f"(at least {REFUND_RULE_DAYS_BEFORE} days before the permit start date of May 15, 2026)."
    )
    await evaluator.verify(
        claim=refund_claim,
        node=refund_date_leaf,
        sources=sources_or_none(urls),
        additional_instruction="Verify the timing rule and apply it to the given start date to confirm the stated deadline."
    )

    # Leaf: Refund conditions acknowledged (not printed)
    refund_conditions_leaf = evaluator.add_leaf(
        id="Refund_Conditions_Acknowledged",
        desc="Acknowledges that the refund of nightly charges requires the permit has not been printed; printing affects cancellation.",
        parent=refund_node,
        critical=True,
    )
    conditions_claim = "A refund of nightly charges is only available if the permit has not been printed; printing timing affects cancellation and refund eligibility."
    await evaluator.verify(
        claim=conditions_claim,
        node=refund_conditions_leaf,
        sources=sources_or_none(urls),
        additional_instruction="Confirm the 'not printed' condition and any printing-related restrictions from official sources."
    )

    # Leaf: Refund policy reference URLs official
    refund_urls_leaf = evaluator.add_custom_node(
        result=are_official_urls(urls),
        id="Refund_Policy_Reference_URLs",
        desc="Provides official NPS or Recreation.gov URL(s) supporting the refund timing rule and printing-related restriction.",
        parent=refund_node,
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
    Evaluate the answer for Grand Canyon May 2026 trip planning (lottery window, permit cost, refund deadline).
    """
    # Initialize evaluator (root non-critical aggregation; we add a critical top-level aggregator under root)
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

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_trip_planning(),
        template_class=TripPlanningExtraction,
        extraction_name="trip_planning_extraction",
    )

    # Add ground truth info to the report
    evaluator.add_ground_truth({
        "lottery_opening_expected": EXPECTED_LOTTERY_OPENING_DATE,
        "lottery_closing_expected": EXPECTED_LOTTERY_CLOSING_DATETIME_TZ,
        "cost_expected_group_size": EXPECTED_GROUP_SIZE,
        "cost_expected_nights_below_rim": EXPECTED_NIGHTS_BELOW_RIM,
        "cost_expected_bright_angel_nights": EXPECTED_BRIGHT_ANGEL_NIGHTS,
        "cost_expected_cottonwood_nights": EXPECTED_COTTONWOOD_NIGHTS,
        "cost_expected_nightly_rate_usd": EXPECTED_BELOW_RIM_RATE_USD,
        "cost_expected_basic_permit_fee_usd": EXPECTED_BASIC_PERMIT_FEE_USD,
        "refund_rule_days_before": REFUND_RULE_DAYS_BEFORE,
        "refund_deadline_expected": EXPECTED_REFUND_DEADLINE.strftime("%B %d, %Y"),
    }, gt_type="ground_truth")

    # Top-level critical aggregator representing "Trip_Planning_Complete"
    trip_node = evaluator.add_parallel(
        id="Trip_Planning_Complete",
        desc="All required trip planning information is provided for the May 15, 2026 Grand Canyon backpacking trip, with official supporting URLs.",
        parent=root,
        critical=True,
    )

    # Build and verify each sub-tree
    await verify_lottery_window(evaluator, trip_node, extracted.lottery or LotteryInfo())
    await verify_total_permit_cost(evaluator, trip_node, extracted.cost or CostInfo())
    await verify_refund_deadline(evaluator, trip_node, extracted.refund or RefundInfo())

    # Return summary with aggregated score and verification tree
    return evaluator.get_summary()