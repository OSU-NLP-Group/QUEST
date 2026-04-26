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
TASK_ID = "yellowstone_universal_planning_2026"
TASK_DESCRIPTION = (
    "A family is planning a summer 2026 vacation with two major components: (1) a stay at Yellowstone National Park "
    "checking in on June 15, 2026, for 5 consecutive nights, and (2) a 3-day visit to Universal Orlando Resort theme parks. "
    "Today's date is November 30, 2025. They need to make optimal booking and planning decisions.\n\n"
    "Provide the following information:\n\n"
    "1. YELLOWSTONE BOOKING WINDOW: On what specific date and time (Mountain Time) will online reservations first open for "
    "their June 15-19, 2026 Yellowstone lodging, and explain how Yellowstone's rolling 13-month advance reservation window works?\n\n"
    "2. UNIVERSAL ORLANDO PREMIER HOTELS: Identify all Universal Orlando premier hotels that include complimentary unlimited "
    "Express Pass for guests, and explain the validity period of the Express Pass benefit relative to check-in and check-out dates.\n\n"
    "3. EXPRESS PASS OPTIMIZATION: If the family books one night at a Universal Orlando premier hotel, how many full days of "
    "Express Pass access will each guest receive, specifically identifying which days the passes are valid?\n\n"
    "4. US AIRLINE PASSENGER RIGHTS: If their airline cancels their flight to Orlando, what are they automatically entitled to "
    "under current US Department of Transportation rules that became effective in October 2024, including the timeframe for processing refunds?\n\n"
    "5. RESERVATION POLICIES: For their Yellowstone lodging reservation arriving on June 15, 2026, how many additional consecutive "
    "nights (if any) can they extend into July 2026 at the same lodge/campground and room type when booking on the reservation opening date?"
)

# Helpful constant: expected Universal Orlando Premier hotels with Express Unlimited
EXPECTED_PREMIER_HOTELS = {
    "hard rock hotel",
    "loews portofino bay hotel",
    "loews royal pacific resort",
}

# Official sources (used in verification)
YELLOWSTONE_RESERVATIONS_URLS = [
    "https://www.yellowstonenationalparklodges.com/reservations/",
    "https://www.yellowstonenationalparklodges.com/planning/reservations/",
]

UNIVERSAL_ROYAL_PACIFIC_URL = "https://www.universalorlando.com/web/en/us/places-to-stay/loews-royal-pacific-resort"
UNIVERSAL_PORTOFINO_URL = "https://www.universalorlando.com/web/en/us/places-to-stay/loews-portofino-bay-hotel"
UNIVERSAL_HARD_ROCK_URL = "https://www.universalorlando.com/web/en/us/places-to-stay/hard-rock-hotel"

DOT_RULE_URLS = [
    # Multiple DOT-related pages to maximize verification support
    "https://www.transportation.gov/briefing-room/usdot-issues-final-rule-on-automatic-airline-refunds",  # press release-style URL (may vary)
    "https://www.transportation.gov/airconsumer/refunds",  # general refunds page
    "https://www.transportation.gov/office-of-aviation-consumer-protection/refunds",  # OACP page on refunds
    "https://www.transportation.gov/airconsumer/fly-rights",  # consumer fly rights overview
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class YellowstoneWindow(BaseModel):
    opening_datetime_mt: Optional[str] = None
    rolling_window_explanation: Optional[str] = None


class UniversalPremierHotels(BaseModel):
    hotel_names: List[str] = Field(default_factory=list)
    validity_rule_text: Optional[str] = None


class ExpressOneNight(BaseModel):
    days_count: Optional[str] = None
    days_identified_text: Optional[str] = None


class USDOTRefunds(BaseModel):
    automatic_refund_statement: Optional[str] = None
    refund_processing_timeframe: Optional[str] = None
    rule_effective_date: Optional[str] = None


class YellowstoneExtension(BaseModel):
    july_extension_nights: Optional[str] = None
    same_property_type_condition_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_yellowstone_window() -> str:
    return (
        "Extract the Yellowstone booking window details that the answer provides.\n"
        "- opening_datetime_mt: The specific date AND time in Mountain Time when online reservations first open for their June 15–19, 2026 lodging.\n"
        "- rolling_window_explanation: A concise explanation of Yellowstone's rolling 13-month window (e.g., opens on the 5th of each month at midnight MT for the entire same month of the following year).\n"
        "Return both fields even if one is missing (use null for missing)."
    )


def prompt_extract_universal_hotels() -> str:
    return (
        "Extract Universal Orlando premier hotel information from the answer.\n"
        "- hotel_names: List all hotel names that the answer claims include complimentary Unlimited Express Pass for guests.\n"
        "- validity_rule_text: The answer's statement explaining when the Express Pass benefit is valid relative to check-in and check-out dates.\n"
        "Do not include non-premier hotels. If a field is missing, provide an empty list or null."
    )


def prompt_extract_express_one_night() -> str:
    return (
        "Extract the Express Pass optimization statement for a one-night stay at a Universal Orlando premier hotel.\n"
        "- days_count: The stated number of full park days of Express Pass access (e.g., '2', 'two').\n"
        "- days_identified_text: The specific days identified (e.g., 'check-in day and check-out day').\n"
        "Return null for any missing field."
    )


def prompt_extract_usdot_rights() -> str:
    return (
        "Extract the US airline passenger rights (DOT rules effective Oct 2024) from the answer.\n"
        "- automatic_refund_statement: The answer's claim that a cancellation triggers an automatic refund (not a voucher).\n"
        "- refund_processing_timeframe: The stated timeframe to process refunds (e.g., 'credit card refunds within 7 business days of becoming due').\n"
        "- rule_effective_date: The effective date or month/year (e.g., 'October 2024' or 'Oct 28, 2024').\n"
        "Return null for any missing field."
    )


def prompt_extract_yellowstone_extension() -> str:
    return (
        "Extract the extension policy details for Yellowstone lodging into July 2026 from the answer.\n"
        "- july_extension_nights: The number of additional consecutive nights that can be extended into July 2026 (e.g., '4').\n"
        "- same_property_type_condition_text: The answer's statement that any extension must be at the same lodge/campground and room/site type.\n"
        "Return null for any missing field."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_hotel_name(name: str) -> str:
    return name.strip().lower()


def _matches_expected_premier(name: str) -> bool:
    n = _normalize_hotel_name(name)
    # Flexible token-based checks
    return (
        ("hard rock" in n and "hotel" in n)
        or ("portofino" in n)
        or ("royal pacific" in n and ("resort" in n or "hotel" in n))
        or (n in EXPECTED_PREMIER_HOTELS)
    )


def _includes_all_expected(hotels: List[str]) -> bool:
    normalized = {_normalize_hotel_name(h) for h in hotels}
    found = set()
    for h in normalized:
        if ("hard rock" in h and "hotel" in h):
            found.add("hard rock hotel")
        if "portofino" in h:
            found.add("loews portofino bay hotel")
        if "royal pacific" in h and ("resort" in h or "hotel" in h):
            found.add("loews royal pacific resort")
        if h in EXPECTED_PREMIER_HOTELS:
            found.add(h)
    return EXPECTED_PREMIER_HOTELS.issubset(found)


def _no_extra_hotels(hotels: List[str]) -> bool:
    # Ensure listed hotels are limited to expected set (allow flexible matching)
    for h in hotels:
        if not _matches_expected_premier(h):
            return False
    return True


def _expected_yellowstone_opening_for_june_2026() -> str:
    # Yellowstone rolling window: opens the 5th of the month for the entire same month of the following year at midnight MT.
    # For June 2026, opening date is May 5, 2025 at 12:00 AM MT.
    # Return a normalized string for comparison.
    return "May 5, 2025 12:00 AM MT"


def _string_contains_any(text: Optional[str], keywords: List[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return all(k.lower() in t for k in keywords)


def _is_two_days(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    return t in {"2", "two", "2 days", "two days", "two full days", "2 full days"}


def _parse_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    t = text.strip().lower()
    try:
        return int(t)
    except Exception:
        # Try mapping common words
        if t in {"one"}:
            return 1
        if t in {"two"}:
            return 2
        if t in {"three"}:
            return 3
        if t in {"four"}:
            return 4
        if t in {"five"}:
            return 5
    return None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_yellowstone_booking_window_nodes(
    evaluator: Evaluator,
    parent_node,
    yz: YellowstoneWindow,
) -> None:
    node = evaluator.add_parallel(
        id="yellowstone_booking_window",
        desc="Determines when Yellowstone online reservations first open for the June 15–19, 2026 stay and explains the rolling window",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Opening date/time correctness (custom logic comparing to expected)
    expected_dt = _expected_yellowstone_opening_for_june_2026()
    opening_dt_text = yz.opening_datetime_mt or ""
    # Normalize simple variants such as "MT" vs "Mountain Time"
    normalized_opening = opening_dt_text.strip().replace("Mountain Time", "MT")

    evaluator.add_custom_node(
        result=(normalized_opening.lower() == expected_dt.lower()),
        id="opening_datetime_mt",
        desc=f"States the correct reservation opening date AND time in Mountain Time (expected: {expected_dt})",
        parent=node,
        critical=True,
    )

    # Leaf: Rolling window explanation (verify by official Yellowstone Lodges pages)
    expl_leaf = evaluator.add_leaf(
        id="rolling_window_explanation",
        desc="Explains how the rolling 13-month window works (opens on the 5th; opens the entire same month of the following year; at midnight MT)",
        parent=node,
        critical=True,
    )
    claim = (
        "Yellowstone National Park Lodges uses a rolling 13‑month advance window: on the 5th of each month at 12:00 a.m. Mountain Time, "
        "online reservations open for the entire same month of the following year."
    )
    await evaluator.verify(
        claim=claim,
        node=expl_leaf,
        sources=YELLOWSTONE_RESERVATIONS_URLS,
        additional_instruction="Look for language indicating the 13‑month rolling window, the 5th‑of‑month release, midnight MT, and that the entire same month of the following year opens.",
    )


async def build_universal_premier_hotels_nodes(
    evaluator: Evaluator,
    parent_node,
    uni: UniversalPremierHotels,
) -> None:
    node = evaluator.add_parallel(
        id="universal_premier_hotels_express",
        desc="Identifies Universal Orlando premier hotels that include complimentary unlimited Express Pass and explains benefit validity period",
        parent=parent_node,
        critical=True,
    )

    # Sub-aggregation for list completeness and correctness
    list_node = evaluator.add_parallel(
        id="hotel_list_complete_and_correct",
        desc="Lists all and only the premier hotels that include complimentary unlimited Express Pass (Hard Rock Hotel, Loews Portofino Bay Hotel, Loews Royal Pacific Resort)",
        parent=node,
        critical=True,
    )
    hotels = uni.hotel_names or []

    # Leaf: Contains all expected hotels
    evaluator.add_custom_node(
        result=_includes_all_expected(hotels),
        id="hotel_list_includes_all_three",
        desc="List includes Hard Rock Hotel, Loews Portofino Bay Hotel, and Loews Royal Pacific Resort",
        parent=list_node,
        critical=True,
    )

    # Leaf: No extras beyond the three expected
    evaluator.add_custom_node(
        result=_no_extra_hotels(hotels),
        id="hotel_list_no_extras",
        desc="List contains no additional hotels beyond the three premier properties that include complimentary Unlimited Express Pass",
        parent=list_node,
        critical=True,
    )

    # Verify each hotel's Express Unlimited benefit via official pages
    royal_leaf = evaluator.add_leaf(
        id="royal_pacific_express_included",
        desc="Loews Royal Pacific Resort includes complimentary Unlimited Express Pass for hotel guests",
        parent=list_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Loews Royal Pacific Resort includes complimentary Unlimited Express Pass for each registered hotel guest for the length of stay.",
        node=royal_leaf,
        sources=UNIVERSAL_ROYAL_PACIFIC_URL,
        additional_instruction="Confirm the benefit description on the official resort page. Allow phrasing variations like 'Universal Express Unlimited' and 'skip the regular lines'.",
    )

    port_leaf = evaluator.add_leaf(
        id="portofino_express_included",
        desc="Loews Portofino Bay Hotel includes complimentary Unlimited Express Pass for hotel guests",
        parent=list_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Loews Portofino Bay Hotel includes complimentary Unlimited Express Pass for each registered hotel guest for the length of stay.",
        node=port_leaf,
        sources=UNIVERSAL_PORTOFINO_URL,
        additional_instruction="Confirm the benefit description on the official hotel page.",
    )

    hard_leaf = evaluator.add_leaf(
        id="hard_rock_express_included",
        desc="Hard Rock Hotel includes complimentary Unlimited Express Pass for hotel guests",
        parent=list_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Hard Rock Hotel includes complimentary Unlimited Express Pass for each registered hotel guest for the length of stay.",
        node=hard_leaf,
        sources=UNIVERSAL_HARD_ROCK_URL,
        additional_instruction="Confirm the benefit description on the official hotel page.",
    )

    # Validity rule leaf
    validity_leaf = evaluator.add_leaf(
        id="express_pass_validity_rule",
        desc="Explains Express Pass validity relative to stay dates (valid starting the morning of check-in day through park close on check-out day)",
        parent=node,
        critical=True,
    )
    validity_claim = (
        "For Universal Orlando premier hotels, the complimentary Universal Express Unlimited benefit is valid starting the day of check-in "
        "and remains valid through park close on the day of check‑out (i.e., both check‑in and check‑out days are included)."
    )
    await evaluator.verify(
        claim=validity_claim,
        node=validity_leaf,
        sources=[UNIVERSAL_ROYAL_PACIFIC_URL, UNIVERSAL_PORTOFINO_URL, UNIVERSAL_HARD_ROCK_URL],
        additional_instruction="Look for phrasing such as 'valid for length of stay including day of check-in and check-out'. Treat 'morning of check-in through park close on check-out' as equivalent.",
    )


async def build_express_one_night_nodes(
    evaluator: Evaluator,
    parent_node,
    expr: ExpressOneNight,
) -> None:
    node = evaluator.add_parallel(
        id="express_pass_optimization_one_night",
        desc="Optimizes Express Pass use for a one-night premier-hotel stay",
        parent=parent_node,
        critical=True,
    )

    # Leaf: must state 2 full days and identify check-in and check-out days
    # Use custom check based on extracted text (robust and precise)
    has_two_days = _is_two_days(expr.days_count)
    identifies_days = _string_contains_any(expr.days_identified_text, ["check-in", "check-out"])

    evaluator.add_custom_node(
        result=(has_two_days and identifies_days),
        id="days_of_access_for_one_night",
        desc="Correctly states 2 full days and identifies them as the check-in day and the check-out day",
        parent=node,
        critical=True,
    )


async def build_usdot_rights_nodes(
    evaluator: Evaluator,
    parent_node,
    dot: USDOTRefunds,
) -> None:
    node = evaluator.add_parallel(
        id="us_dot_airline_cancellation_rights",
        desc="Explains passenger entitlements if the airline cancels the flight under the stated Oct 2024 DOT automatic-refund rules",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Automatic refund entitlement (verify by DOT pages)
    auto_leaf = evaluator.add_leaf(
        id="automatic_refund_entitlement",
        desc="States that a cancellation triggers an automatic refund entitlement under the referenced DOT rule (refund rather than a forced voucher/credit)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Under US DOT rules effective October 2024, if an airline cancels a flight or makes a significant change, passengers are entitled to an automatic cash refund rather than being forced to accept a voucher or credit.",
        node=auto_leaf,
        sources=DOT_RULE_URLS,
        additional_instruction="Confirm the rule mandates automatic refunds for cancellations/significant changes and does not allow forcing vouchers.",
    )

    # Leaf: Refund processing timeframe (7 business days for credit card refunds)
    timeframe_leaf = evaluator.add_leaf(
        id="refund_processing_timeframe",
        desc="Includes the required refund processing timeframe (credit card refunds within 7 business days of becoming due)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="US DOT rules require airlines to process refunds to credit card accounts within 7 business days of the refund becoming due (and other payment methods generally within 20 calendar days).",
        node=timeframe_leaf,
        sources=DOT_RULE_URLS,
        additional_instruction="Look specifically for '7 business days' for credit card refunds. Mention of 20 calendar days for other payment methods is acceptable but not required.",
    )

    # Leaf: Rule effective date (October 2024 acceptable; Oct 28, 2024 acceptable)
    effective_leaf = evaluator.add_leaf(
        id="rule_effective_oct_2024",
        desc="Indicates the rule became effective in October 2024 (Oct 28, 2024 acceptable)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The US DOT automatic airline refunds rule became effective in October 2024 (around October 28, 2024).",
        node=effective_leaf,
        sources=DOT_RULE_URLS,
        additional_instruction="Confirm the effective month/date around late October 2024; allow 'October 28, 2024' or similar 'late October' statements.",
    )


async def build_yellowstone_extension_nodes(
    evaluator: Evaluator,
    parent_node,
    ext: YellowstoneExtension,
) -> None:
    node = evaluator.add_parallel(
        id="yellowstone_extension_into_july",
        desc="Determines how many additional consecutive nights (if any) the June 15–19, 2026 Yellowstone stay can be extended into July 2026 at the same property and room/site type when booking on the opening date",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Correctly computes number of July extension nights given policy (up to 4 consecutive nights into next month)
    # For a June 15–19 stay, maximum nights that actually fall in July when booking on opening date for June 2026 is 4 (assuming continuous extension across end of June).
    parsed = _parse_int(ext.july_extension_nights)
    evaluator.add_custom_node(
        result=(parsed == 4),
        id="july_extension_nights_computed",
        desc="Correctly computes the number of added nights that fall in July 2026: 4 nights (with continuous extension across end-of-month)",
        parent=node,
        critical=True,
    )

    # Leaf: States same property and room/site type condition
    evaluator.add_custom_node(
        result=_string_contains_any(ext.same_property_type_condition_text, ["same", "property"]) and
               _string_contains_any(ext.same_property_type_condition_text, ["room", "site"]),
        id="same_property_and_type_condition",
        desc="States that any extension must be at the same lodge/campground and same room/site type",
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

    # Create a critical task aggregator under the root (root created by initialize is non-critical)
    task_node = evaluator.add_parallel(
        id="task_core",
        desc="Provides all requested planning/booking information for the specified Yellowstone + Universal Orlando trip",
        parent=root,
        critical=True,
    )

    # Concurrent extractions
    yz_task = evaluator.extract(
        prompt=prompt_extract_yellowstone_window(),
        template_class=YellowstoneWindow,
        extraction_name="yellowstone_booking_window",
    )
    uni_task = evaluator.extract(
        prompt=prompt_extract_universal_hotels(),
        template_class=UniversalPremierHotels,
        extraction_name="universal_premier_hotels",
    )
    expr_task = evaluator.extract(
        prompt=prompt_extract_express_one_night(),
        template_class=ExpressOneNight,
        extraction_name="express_one_night",
    )
    dot_task = evaluator.extract(
        prompt=prompt_extract_usdot_rights(),
        template_class=USDOTRefunds,
        extraction_name="usdot_rights",
    )
    ext_task = evaluator.extract(
        prompt=prompt_extract_yellowstone_extension(),
        template_class=YellowstoneExtension,
        extraction_name="yellowstone_extension",
    )

    yz, uni, expr, dot, ext = await asyncio.gather(yz_task, uni_task, expr_task, dot_task, ext_task)

    # Ground truth / computed reference info for transparency
    evaluator.add_ground_truth({
        "expected_yellowstone_opening_datetime_mt": _expected_yellowstone_opening_for_june_2026(),
        "expected_premier_hotels_with_express_unlimited": sorted(list(EXPECTED_PREMIER_HOTELS)),
        "express_one_night_expected_days": 2,
        "dot_refund_credit_card_days": 7,
        "dot_rule_effective_month_year": "October 2024",
        "yellowstone_max_july_extension_nights": 4
    }, gt_type="reference_expectations")

    # Build verification subtrees
    await build_yellowstone_booking_window_nodes(evaluator, task_node, yz)
    await build_universal_premier_hotels_nodes(evaluator, task_node, uni)
    await build_express_one_night_nodes(evaluator, task_node, expr)
    await build_usdot_rights_nodes(evaluator, task_node, dot)
    await build_yellowstone_extension_nodes(evaluator, task_node, ext)

    # Return final structured summary
    return evaluator.get_summary()