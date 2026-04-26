import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "travel_planning_yellowstone_breeze_mia_rccl_2026"
TASK_DESCRIPTION = (
    "A family is planning a summer vacation to Yellowstone National Park, departing from Miami International Airport "
    "(MIA) on Breeze Airways for a 7-day trip. They are also considering adding a 7-night Royal Caribbean cruise to "
    "their travel plans. To make informed booking decisions, they need a comprehensive travel planning analysis that "
    "covers: (1) Breeze Airways Baggage Policy - What are the maximum dimensions and weight limits for carry-on bags "
    "and checked bags? What is the cost structure for checked bags when purchased at initial booking versus at the "
    "airport? (2) MIA Airport Parking Cost Comparison - For their 7-day trip, what is the total cost of parking in the "
    "MIA garage versus the Economy Park & Ride lot? Which option is more economical? (3) Yellowstone Lodging "
    "Reservation Policies - If they want to stay at Yellowstone National Park lodges in July 2026, when is the earliest "
    "they can make a reservation? What are the cancellation policy tiers and associated fees or penalties? What deposit "
    "is required at booking? (4) Royal Caribbean Cruise Cancellation Policy - For a 7-night cruise, what are the "
    "cancellation deadlines and associated refund percentages or cancellation charges? Please provide a comprehensive "
    "analysis with specific dimensions, costs, dates, timelines, and policy details, supported by reference URLs from "
    "official sources."
)


# --------------------------------------------------------------------------- #
# Helper parsing functions                                                    #
# --------------------------------------------------------------------------- #
def parse_money(value: Optional[str]) -> Optional[float]:
    """
    Parse a monetary string into a float. Accepts formats like "$25", "25.00", "USD 25", etc.
    Returns None if parsing fails.
    """
    if value is None:
        return None
    try:
        # Extract first occurrence of a number (integer or decimal)
        match = re.search(r"([-+]?\d+(?:\.\d+)?)", value.replace(",", ""))
        if match:
            return float(match.group(1))
        return None
    except Exception:
        return None


def parse_int(value: Optional[str]) -> Optional[int]:
    """
    Parse an integer from a string (e.g., "1 transaction", "Assume 2 payments").
    Returns None if parsing fails.
    """
    if value is None:
        return None
    try:
        match = re.search(r"([-+]?\d+)", value.replace(",", ""))
        if match:
            return int(match.group(1))
        return None
    except Exception:
        return None


def approx_equal(a: Optional[float], b: Optional[float], tol: float = 1e-2) -> bool:
    """
    Compare two floats for approximate equality within a tolerance.
    """
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BreezeBaggageExtraction(BaseModel):
    personal_item_max_dimensions: Optional[str] = None  # e.g., "17 x 13 x 8 inches"
    carry_on_max_dimensions: Optional[str] = None       # e.g., "22 x 14 x 9 inches"
    carry_on_max_weight: Optional[str] = None           # e.g., "35 pounds"
    checked_bag_max_size: Optional[str] = None          # e.g., "62 linear inches"
    checked_bag_max_weight: Optional[str] = None        # e.g., "50 pounds"
    checked_bag_price_initial_booking: Optional[str] = None  # e.g., "$35"
    checked_bag_price_at_airport: Optional[str] = None       # e.g., "$75"
    reference_urls_baggage: List[str] = Field(default_factory=list)


class MIAParkingExtraction(BaseModel):
    garage_rate_per_20_min: Optional[str] = None             # e.g., "$2.00 per 20 minutes"
    garage_max_daily_rate: Optional[str] = None              # e.g., "$25.00"
    garage_7_day_total: Optional[str] = None                # e.g., "$175.00"
    economy_max_daily_rate: Optional[str] = None            # e.g., "$12.00"
    economy_convenience_fee_per_transaction: Optional[str] = None  # e.g., "$0.27"
    economy_transaction_count_assumption: Optional[str] = None     # e.g., "1 transaction"
    economy_7_day_total: Optional[str] = None               # e.g., "$84.27"
    more_economical_option: Optional[str] = None            # e.g., "Economy Park & Ride" or "MIA Garage"
    reference_urls_parking: List[str] = Field(default_factory=list)


class YellowstoneLodgingExtraction(BaseModel):
    earliest_reservation_date_for_july_2026: Optional[str] = None  # e.g., "July 5, 2025"
    cancellation_tiers_and_fees_text: Optional[str] = None         # free-form summary
    deposit_requirement_text: Optional[str] = None                 # e.g., "First night’s rate"
    reference_urls_yellowstone: List[str] = Field(default_factory=list)


class RoyalCaribbeanCancellationExtraction(BaseModel):
    category_label: Optional[str] = None  # e.g., "5–14 nights"
    window_90_plus_days_charge: Optional[str] = None        # e.g., "No charge except deposit"
    window_89_to_75_days_charge: Optional[str] = None       # e.g., "25%"
    window_74_to_61_days_charge: Optional[str] = None       # e.g., "50%"
    window_60_to_31_days_charge: Optional[str] = None       # e.g., "75%"
    window_30_or_less_days_charge: Optional[str] = None     # e.g., "100%"
    reference_urls_cruise: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_breeze_baggage() -> str:
    return """
    Extract Breeze Airways baggage policy details as explicitly stated in the answer.
    Required fields:
    - personal_item_max_dimensions: the maximum dimensions of a personal item (e.g., "17 x 13 x 8 inches").
    - carry_on_max_dimensions: the maximum dimensions of a carry-on (e.g., "22 x 14 x 9 inches", including handles and wheels).
    - carry_on_max_weight: the maximum weight allowed for a carry-on (e.g., "35 pounds").
    - checked_bag_max_size: the maximum size of a checked bag stated as linear inches (L+W+H) (e.g., "62 linear inches").
    - checked_bag_max_weight: the maximum weight allowed for a checked bag (e.g., "50 pounds").
    - checked_bag_price_initial_booking: the stated first checked bag price at initial booking (e.g., "$35").
    - checked_bag_price_at_airport: the stated first checked bag price at the airport (e.g., "$75").
    - reference_urls_baggage: an array of official Breeze Airways URLs that support baggage limits and/or bag-fee timing. Extract only URLs visibly present in the answer. If none, return an empty array.
    If any field is not present in the answer, set it to null (or empty array for URLs).
    """


def prompt_extract_mia_parking() -> str:
    return """
    Extract Miami International Airport (MIA) parking rate details and the computed totals for a 7-day stay as stated in the answer.
    Required fields:
    - garage_rate_per_20_min: the MIA garage rate per 20-minute increment (e.g., "$2.00 per 20 minutes").
    - garage_max_daily_rate: the MIA garage maximum daily rate (e.g., "$25.00").
    - garage_7_day_total: the numeric total for 7 days in the MIA garage provided in the answer (e.g., "$175.00").
    - economy_max_daily_rate: the Economy Park & Ride maximum daily rate (e.g., "$12.00").
    - economy_convenience_fee_per_transaction: the Economy Park & Ride convenience fee (e.g., "$0.27 per transaction").
    - economy_transaction_count_assumption: the assumed number of payment transactions used in the answer for the Economy lot (e.g., "1 transaction").
    - economy_7_day_total: the numeric total for 7 days in the Economy lot as provided in the answer (e.g., "$84.27").
    - more_economical_option: which option the answer states is cheaper ("MIA Garage" or "Economy Park & Ride").
    - reference_urls_parking: an array of official Miami International Airport URLs that support the parking rates/fees. Extract only URLs present in the answer; if none, return an empty array.
    If any field is not present in the answer, set it to null (or empty array for URLs).
    """


def prompt_extract_yellowstone() -> str:
    return """
    Extract Yellowstone National Park lodges reservation and cancellation policy details as stated in the answer.
    Required fields:
    - earliest_reservation_date_for_july_2026: the earliest date reservations open for a July 2026 stay (e.g., "July 5, 2025").
    - cancellation_tiers_and_fees_text: the stated tiers and associated penalties/fees for cancellations (e.g., '30+ days: full deposit refund; 7–30 days: $25 non-refundable fee; within 7 days of summer arrival: forfeit full deposit').
    - deposit_requirement_text: the stated deposit required at booking (e.g., 'First night’s rate').
    - reference_urls_yellowstone: one or more official Yellowstone lodging/reservation policy URLs (e.g., yellowstonenationalparklodges.com or nps.gov). Extract only URLs present in the answer; if none, return an empty array.
    If any field is not present in the answer, set it to null (or empty array for URLs).
    """


def prompt_extract_rccl() -> str:
    return """
    Extract Royal Caribbean cruise cancellation policy details for a 7-night cruise as stated in the answer.
    Required fields:
    - category_label: the category applied to a 7-night cruise (e.g., "5–14 nights").
    - window_90_plus_days_charge: the stated charge/refund policy at 90 or more days prior (e.g., "No charge except deposit").
    - window_89_to_75_days_charge: the stated charge/refund policy for 89–75 days prior (e.g., "25%").
    - window_74_to_61_days_charge: the stated charge/refund policy for 74–61 days prior (e.g., "50%").
    - window_60_to_31_days_charge: the stated charge/refund policy for 60–31 days prior (e.g., "75%").
    - window_30_or_less_days_charge: the stated charge/refund policy for 30 or fewer days prior (e.g., "100%").
    - reference_urls_cruise: official Royal Caribbean URLs supporting the cancellation policy. Extract only URLs present in the answer; if none, return an empty array.
    If any field is not present in the answer, set it to null (or empty array for URLs).
    """


# --------------------------------------------------------------------------- #
# Verification branch builders                                                #
# --------------------------------------------------------------------------- #
async def build_baggage_branch(
    evaluator: Evaluator,
    parent_node,
    baggage: BreezeBaggageExtraction,
) -> None:
    section_node = evaluator.add_parallel(
        id="Breeze_Airways_Baggage_Policy",
        desc="Breeze Airways baggage size/weight limits and checked-bag pricing timing.",
        parent=parent_node,
        critical=True,
    )

    urls = baggage.reference_urls_baggage

    # Personal Item Max Dimensions
    if baggage.personal_item_max_dimensions and urls:
        node = evaluator.add_leaf(
            id="Personal_Item_Max_Dimensions",
            desc="States Breeze personal item maximum dimensions (17x13x8 inches).",
            parent=section_node,
            critical=True,
        )
        claim = f"Breeze Airways personal item maximum dimensions are {baggage.personal_item_max_dimensions}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Verify against Breeze's official baggage policy page. Allow minor formatting variants (e.g., '17 x 13 x 8 in').",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Personal_Item_Max_Dimensions",
            desc="States Breeze personal item maximum dimensions (17x13x8 inches).",
            parent=section_node,
            critical=True,
        )

    # Carry-On Max Dimensions
    if baggage.carry_on_max_dimensions and urls:
        node = evaluator.add_leaf(
            id="Carry_On_Max_Dimensions",
            desc="States Breeze carry-on maximum dimensions (22x14x9 inches, including handles and wheels).",
            parent=section_node,
            critical=True,
        )
        claim = f"Breeze Airways carry-on maximum dimensions are {baggage.carry_on_max_dimensions}, including handles and wheels."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Verify carry-on dimensional limits including handles and wheels from Breeze official policy.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Carry_On_Max_Dimensions",
            desc="States Breeze carry-on maximum dimensions (22x14x9 inches, including handles and wheels).",
            parent=section_node,
            critical=True,
        )

    # Carry-On Max Weight
    if baggage.carry_on_max_weight and urls:
        node = evaluator.add_leaf(
            id="Carry_On_Max_Weight",
            desc="States Breeze carry-on maximum weight (35 pounds).",
            parent=section_node,
            critical=True,
        )
        claim = f"Breeze Airways carry-on maximum weight is {baggage.carry_on_max_weight}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Verify carry-on weight limit from Breeze official policy. Accept reasonable units like 'lbs', 'pounds'.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Carry_On_Max_Weight",
            desc="States Breeze carry-on maximum weight (35 pounds).",
            parent=section_node,
            critical=True,
        )

    # Checked Bag Max Size
    if baggage.checked_bag_max_size and urls:
        node = evaluator.add_leaf(
            id="Checked_Bag_Max_Size",
            desc="States Breeze checked-bag maximum size (62 linear inches, L+W+H).",
            parent=section_node,
            critical=True,
        )
        claim = f"Breeze Airways checked bag maximum size is {baggage.checked_bag_max_size} (linear inches, L+W+H)."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Verify checked bag size limit (linear inches) on Breeze official baggage page.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Checked_Bag_Max_Size",
            desc="States Breeze checked-bag maximum size (62 linear inches, L+W+H).",
            parent=section_node,
            critical=True,
        )

    # Checked Bag Max Weight
    if baggage.checked_bag_max_weight and urls:
        node = evaluator.add_leaf(
            id="Checked_Bag_Max_Weight",
            desc="States Breeze checked-bag maximum weight (50 pounds).",
            parent=section_node,
            critical=True,
        )
        claim = f"Breeze Airways checked bag maximum weight is {baggage.checked_bag_max_weight}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Verify checked bag weight limit from the official Breeze policy.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Checked_Bag_Max_Weight",
            desc="States Breeze checked-bag maximum weight (50 pounds).",
            parent=section_node,
            critical=True,
        )

    # Checked Bag Price at Initial Booking
    if baggage.checked_bag_price_initial_booking and urls:
        node = evaluator.add_leaf(
            id="Checked_Bag_Price_At_Initial_Booking",
            desc="States first checked bag price at initial booking ($35).",
            parent=section_node,
            critical=True,
        )
        claim = f"The first checked bag price at initial booking is {baggage.checked_bag_price_initial_booking}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Verify the 'at initial booking' bag fee from Breeze official fees/policy page.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Checked_Bag_Price_At_Initial_Booking",
            desc="States first checked bag price at initial booking ($35).",
            parent=section_node,
            critical=True,
        )

    # Checked Bag Price at Airport
    if baggage.checked_bag_price_at_airport and urls:
        node = evaluator.add_leaf(
            id="Checked_Bag_Price_At_Airport",
            desc="States first checked bag price at the airport ($75).",
            parent=section_node,
            critical=True,
        )
        claim = f"The first checked bag price at the airport is {baggage.checked_bag_price_at_airport}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Verify the airport bag fee from Breeze official baggage/fees page.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Checked_Bag_Price_At_Airport",
            desc="States first checked bag price at the airport ($75).",
            parent=section_node,
            critical=True,
        )

    # Reference URL presence/support
    if urls:
        node = evaluator.add_leaf(
            id="Reference_URL_Baggage",
            desc="Provides at least one official Breeze Airways URL supporting baggage limits and/or bag-fee timing.",
            parent=section_node,
            critical=True,
        )
        claim = "This referenced page is an official Breeze Airways resource that includes baggage limits and/or bag-fee timing details."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Confirm the page is Breeze official and contains baggage policy or fee timing information.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Reference_URL_Baggage",
            desc="Provides at least one official Breeze Airways URL supporting baggage limits and/or bag-fee timing.",
            parent=section_node,
            critical=True,
        )


async def build_mia_parking_branch(
    evaluator: Evaluator,
    parent_node,
    parking: MIAParkingExtraction,
) -> None:
    section_node = evaluator.add_parallel(
        id="MIA_Parking_Cost_Comparison",
        desc="Computes and compares total parking costs for 7 days at MIA garage vs Economy Park & Ride using the provided rate rules.",
        parent=parent_node,
        critical=True,
    )

    urls = parking.reference_urls_parking

    # Garage Rate Per 20 Min
    if parking.garage_rate_per_20_min and urls:
        node = evaluator.add_leaf(
            id="Garage_Rate_Per_20_Min",
            desc="Uses/reflects MIA garage rate: $2.00 per 20-minute increment.",
            parent=section_node,
            critical=True,
        )
        claim = f"MIA garage rate is {parking.garage_rate_per_20_min}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Verify the per 20-minute garage rate on the official MIA parking information page.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Garage_Rate_Per_20_Min",
            desc="Uses/reflects MIA garage rate: $2.00 per 20-minute increment.",
            parent=section_node,
            critical=True,
        )

    # Garage Max Daily Rate
    if parking.garage_max_daily_rate and urls:
        node = evaluator.add_leaf(
            id="Garage_Max_Daily_Rate",
            desc="Uses/reflects MIA garage maximum daily rate: $25.00 (after 4 hours).",
            parent=section_node,
            critical=True,
        )
        claim = f"MIA garage maximum daily rate is {parking.garage_max_daily_rate} (after 4 hours)."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Verify the garage maximum daily rate and the 'after 4 hours' rule on the official MIA site.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Garage_Max_Daily_Rate",
            desc="Uses/reflects MIA garage maximum daily rate: $25.00 (after 4 hours).",
            parent=section_node,
            critical=True,
        )

    # Garage 7-Day Total Correct (logical check)
    expected_garage_total = None
    stated_garage_total = None
    daily_rate = parse_money(parking.garage_max_daily_rate)
    if daily_rate is not None:
        expected_garage_total = daily_rate * 7.0
    stated_garage_total = parse_money(parking.garage_7_day_total)
    evaluator.add_custom_node(
        result=approx_equal(expected_garage_total, stated_garage_total),
        id="Garage_7_Day_Total_Correct",
        desc="Provides a numeric total cost for 7 days of MIA garage parking consistent with the stated rate rules.",
        parent=section_node,
        critical=True,
    )

    # Economy Max Daily Rate
    if parking.economy_max_daily_rate and urls:
        node = evaluator.add_leaf(
            id="Economy_Max_Daily_Rate",
            desc="Uses/reflects Economy Park & Ride maximum daily rate: $12.00.",
            parent=section_node,
            critical=True,
        )
        claim = f"Economy Park & Ride maximum daily rate is {parking.economy_max_daily_rate}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Verify the Economy Park & Ride daily maximum on the official MIA page.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Economy_Max_Daily_Rate",
            desc="Uses/reflects Economy Park & Ride maximum daily rate: $12.00.",
            parent=section_node,
            critical=True,
        )

    # Economy Convenience Fee
    if parking.economy_convenience_fee_per_transaction and urls:
        node = evaluator.add_leaf(
            id="Economy_Convenience_Fee",
            desc="Uses/reflects Economy Park & Ride convenience fee: $0.27 per transaction.",
            parent=section_node,
            critical=True,
        )
        claim = f"Economy Park & Ride convenience fee is {parking.economy_convenience_fee_per_transaction} per transaction."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Verify the per-transaction convenience fee from official MIA sources.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Economy_Convenience_Fee",
            desc="Uses/reflects Economy Park & Ride convenience fee: $0.27 per transaction.",
            parent=section_node,
            critical=True,
        )

    # Economy Transaction Count Assumption - verify explicitly stated in answer
    if parking.economy_transaction_count_assumption:
        node = evaluator.add_leaf(
            id="Economy_Transaction_Count_Assumption",
            desc="Explicitly states the assumed number of payment transactions for the Economy Park & Ride stay (needed to apply the per-transaction convenience fee).",
            parent=section_node,
            critical=True,
        )
        claim = (
            f"The answer explicitly states the assumed number of payment transactions for the Economy Park & Ride stay "
            f"as '{parking.economy_transaction_count_assumption}'."
        )
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=None,
            additional_instruction="Check the answer text to confirm the transaction-count assumption is explicitly stated.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Economy_Transaction_Count_Assumption",
            desc="Explicitly states the assumed number of payment transactions for the Economy Park & Ride stay (needed to apply the per-transaction convenience fee).",
            parent=section_node,
            critical=True,
        )

    # Economy 7-Day Total Correct (logical check)
    econ_daily = parse_money(parking.economy_max_daily_rate)
    fee = parse_money(parking.economy_convenience_fee_per_transaction)
    txn_count = parse_int(parking.economy_transaction_count_assumption)
    expected_econ_total = None
    if econ_daily is not None:
        expected_econ_total = econ_daily * 7.0
        if fee is not None and txn_count is not None:
            expected_econ_total += fee * float(txn_count)
    stated_econ_total = parse_money(parking.economy_7_day_total)
    evaluator.add_custom_node(
        result=approx_equal(expected_econ_total, stated_econ_total),
        id="Economy_7_Day_Total_Correct",
        desc="Provides a numeric total cost for 7 days of Economy Park & Ride consistent with the daily maximum and the stated transaction-count assumption.",
        parent=section_node,
        critical=True,
    )

    # More Economical Option (logical check)
    cheaper_ok = False
    if expected_garage_total is not None and expected_econ_total is not None and parking.more_economical_option:
        cheaper_option = "Economy Park & Ride" if expected_econ_total <= expected_garage_total else "MIA Garage"
        # Normalize comparison by lowercasing
        cheaper_ok = parking.more_economical_option.strip().lower() == cheaper_option.strip().lower()
    evaluator.add_custom_node(
        result=cheaper_ok,
        id="More_Economical_Option",
        desc="Correctly identifies which option is more economical based on the two computed totals.",
        parent=section_node,
        critical=True,
    )

    # Reference URL presence/support
    if urls:
        node = evaluator.add_leaf(
            id="Reference_URL_Parking",
            desc="Provides at least one official Miami International Airport URL supporting the parking rates/fees used.",
            parent=section_node,
            critical=True,
        )
        claim = "This referenced page is an official MIA resource that lists parking rates and/or fees used in the computation."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Confirm the page is official (MIA) and includes parking rate and/or fee details.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Reference_URL_Parking",
            desc="Provides at least one official Miami International Airport URL supporting the parking rates/fees used.",
            parent=section_node,
            critical=True,
        )


async def build_yellowstone_branch(
    evaluator: Evaluator,
    parent_node,
    ys: YellowstoneLodgingExtraction,
) -> None:
    section_node = evaluator.add_parallel(
        id="Yellowstone_Lodging_Policies",
        desc="Yellowstone National Park lodging reservation opening timing for July 2026 and cancellation/deposit policies.",
        parent=parent_node,
        critical=True,
    )

    urls = ys.reference_urls_yellowstone

    # Earliest Reservation Date for July 2026
    if ys.earliest_reservation_date_for_july_2026 and urls:
        node = evaluator.add_leaf(
            id="Earliest_Reservation_Date_For_July_2026",
            desc="Gives the earliest reservation opening date for a July 2026 lodge stay consistent with the provided reservation-opening rules (13 months in advance; opens on the 5th of the prior year's same month).",
            parent=section_node,
            critical=True,
        )
        claim = f"The earliest reservation opening date for a July 2026 lodge stay is {ys.earliest_reservation_date_for_july_2026}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Verify on Yellowstone Lodges (Xanterra) or NPS official pages the reservation opening timeline for July 2026.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Earliest_Reservation_Date_For_July_2026",
            desc="Gives the earliest reservation opening date for a July 2026 lodge stay consistent with the provided reservation-opening rules (13 months in advance; opens on the 5th of the prior year's same month).",
            parent=section_node,
            critical=True,
        )

    # Cancellation Tiers and Fees
    if ys.cancellation_tiers_and_fees_text and urls:
        node = evaluator.add_leaf(
            id="Cancellation_Tiers_And_Fees",
            desc="States cancellation policy tiers and penalties: 30+ days (full deposit refund), 7–30 days ($25 non-refundable fee), within 7 days of summer arrival (forfeit full deposit).",
            parent=section_node,
            critical=True,
        )
        claim = f"The Yellowstone lodging cancellation policy is described as: {ys.cancellation_tiers_and_fees_text}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Check official Yellowstone Lodges/NPS policy pages to confirm the tiers and fees align with the answer.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Cancellation_Tiers_And_Fees",
            desc="States cancellation policy tiers and penalties: 30+ days (full deposit refund), 7–30 days ($25 non-refundable fee), within 7 days of summer arrival (forfeit full deposit).",
            parent=section_node,
            critical=True,
        )

    # Deposit Requirement
    if ys.deposit_requirement_text and urls:
        node = evaluator.add_leaf(
            id="Deposit_Requirement",
            desc="States deposit required at booking: deposit equals first night’s rate.",
            parent=section_node,
            critical=True,
        )
        claim = f"The deposit required at booking is stated as: {ys.deposit_requirement_text}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Verify deposit policy (e.g., first night’s rate) on official Yellowstone Lodges/NPS pages.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Deposit_Requirement",
            desc="States deposit required at booking: deposit equals first night’s rate.",
            parent=section_node,
            critical=True,
        )

    # Reference URL presence/support
    if urls:
        node = evaluator.add_leaf(
            id="Reference_URL_Yellowstone",
            desc="Provides at least one official Yellowstone lodging/reservations policy URL supporting reservation timing, deposit, and cancellation terms.",
            parent=section_node,
            critical=True,
        )
        claim = "This referenced page is an official Yellowstone Lodges (Xanterra) or NPS resource that includes reservation timing, deposit, and cancellation policy details."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Confirm official status and coverage of reservation opening, deposit, and cancellation policy details.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Reference_URL_Yellowstone",
            desc="Provides at least one official Yellowstone lodging/reservations policy URL supporting reservation timing, deposit, and cancellation terms.",
            parent=section_node,
            critical=True,
        )


async def build_rccl_branch(
    evaluator: Evaluator,
    parent_node,
    rccl: RoyalCaribbeanCancellationExtraction,
) -> None:
    section_node = evaluator.add_parallel(
        id="Royal_Caribbean_Cancellation_Policy",
        desc="Royal Caribbean cancellation deadlines and charges for a 7-night (5–14 night) cruise.",
        parent=parent_node,
        critical=True,
    )

    urls = rccl.reference_urls_cruise

    # Correct Category and Windows
    if urls:
        node = evaluator.add_leaf(
            id="Correct_Category_And_Windows",
            desc="Correctly treats a 7-night cruise as a 5–14 night cruise and lists the relevant cancellation windows (90+ days; 89–75; 74–61; 60–31; ≤30).",
            parent=section_node,
            critical=True,
        )
        stated_category = rccl.category_label or "unknown category"
        claim = (
            f"A 7-night Royal Caribbean cruise falls under the '{stated_category}' cancellation schedule, "
            f"with relevant windows: 90+ days; 89–75; 74–61; 60–31; 30 or less."
        )
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Verify the correct category (5–14 nights) and the listed windows on the official Royal Caribbean policy page.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Correct_Category_And_Windows",
            desc="Correctly treats a 7-night cruise as a 5–14 night cruise and lists the relevant cancellation windows (90+ days; 89–75; 74–61; 60–31; ≤30).",
            parent=section_node,
            critical=True,
        )

    # Correct Charges Per Window
    if urls and any([
        rccl.window_90_plus_days_charge,
        rccl.window_89_to_75_days_charge,
        rccl.window_74_to_61_days_charge,
        rccl.window_60_to_31_days_charge,
        rccl.window_30_or_less_days_charge,
    ]):
        node = evaluator.add_leaf(
            id="Correct_Charges_Per_Window",
            desc="Correctly provides the charge/refund structure for each window: 90+ (no charge except deposit), 89–75 (25%), 74–61 (50%), 60–31 (75%), ≤30 (100%).",
            parent=section_node,
            critical=True,
        )
        claim = (
            f"For 5–14 night cruises, the answer states the cancellation charges/refund structure per window as: "
            f"90+ days: {rccl.window_90_plus_days_charge}; "
            f"89–75 days: {rccl.window_89_to_75_days_charge}; "
            f"74–61 days: {rccl.window_74_to_61_days_charge}; "
            f"60–31 days: {rccl.window_60_to_31_days_charge}; "
            f"30 or less: {rccl.window_30_or_less_days_charge}."
        )
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Verify each window's cancellation charge/refund percentage on official Royal Caribbean policy pages.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Correct_Charges_Per_Window",
            desc="Correctly provides the charge/refund structure for each window: 90+ (no charge except deposit), 89–75 (25%), 74–61 (50%), 60–31 (75%), ≤30 (100%).",
            parent=section_node,
            critical=True,
        )

    # Reference URL presence/support
    if urls:
        node = evaluator.add_leaf(
            id="Reference_URL_Cruise",
            desc="Provides at least one official Royal Caribbean URL supporting the cancellation policy used.",
            parent=section_node,
            critical=True,
        )
        claim = "This referenced page is an official Royal Caribbean policy page listing cancellation charges/refunds for 5–14 night cruises."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Confirm official status (Royal Caribbean) and that the page includes cancellation charges/refund schedules.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Reference_URL_Cruise",
            desc="Provides at least one official Royal Caribbean URL supporting the cancellation policy used.",
            parent=section_node,
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
) -> Dict[str, Any]:
    """
    Evaluate the comprehensive travel planning analysis according to the provided rubric.
    """
    # Initialize evaluator
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

    # Create top-level critical analysis node (to mirror rubric root's critical nature)
    analysis_node = evaluator.add_parallel(
        id="Travel_Planning_Analysis",
        desc="Evaluates a comprehensive travel planning analysis covering Breeze baggage policies, MIA parking costs for a 7-day trip, Yellowstone lodging reservation/cancellation policies for July 2026, and Royal Caribbean 7-night cruise cancellation terms, with official-source URLs.",
        parent=root,
        critical=True,
    )

    # Extract information for each section
    baggage_extraction = await evaluator.extract(
        prompt=prompt_extract_breeze_baggage(),
        template_class=BreezeBaggageExtraction,
        extraction_name="breeze_baggage",
    )

    mia_parking_extraction = await evaluator.extract(
        prompt=prompt_extract_mia_parking(),
        template_class=MIAParkingExtraction,
        extraction_name="mia_parking",
    )

    yellowstone_extraction = await evaluator.extract(
        prompt=prompt_extract_yellowstone(),
        template_class=YellowstoneLodgingExtraction,
        extraction_name="yellowstone_lodging",
    )

    rccl_extraction = await evaluator.extract(
        prompt=prompt_extract_rccl(),
        template_class=RoyalCaribbeanCancellationExtraction,
        extraction_name="rccl_cancellation",
    )

    # Build verification subtrees
    await build_baggage_branch(evaluator, analysis_node, baggage_extraction)
    await build_mia_parking_branch(evaluator, analysis_node, mia_parking_extraction)
    await build_yellowstone_branch(evaluator, analysis_node, yellowstone_extraction)
    await build_rccl_branch(evaluator, analysis_node, rccl_extraction)

    # Optionally record custom info about trip parameters
    evaluator.add_custom_info(
        info={
            "trip_length_days": 7,
            "cruise_length_nights": 7,
            "origin_airport": "MIA",
            "airline": "Breeze Airways",
            "destination": "Yellowstone National Park",
        },
        info_type="trip_parameters",
    )

    # Return evaluation summary
    return evaluator.get_summary()