import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import re

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mlk_2026_gulf_coast_allegiant_trip"
TASK_DESCRIPTION = """
A family of four (two adults, one teenager aged 16, and one child aged 10) from the Philadelphia area wants to take a trip to Florida's Gulf Coast during the Martin Luther King Day 2026 long weekend. They want to use Allegiant Air's newly announced service from a Northeast airport to minimize travel costs.

Plan their complete trip including:

1. Flight Details: Identify the correct departure airport in the Northeast where Allegiant Air operates with service available for January 2026 travel to Punta Gorda, Florida. Specify the airline, departure airport code, destination airport code, and travel dates (outbound and return) that align with the MLK Day 2026 weekend (noting that MLK Day 2026 is a Monday).

2. TSA Compliance: For each of the 4 family members, specify what identification documents are required to board the flight, considering the TSA REAL ID requirements that took effect in May 2025.

3. Baggage Planning: For each passenger, decide whether to bring one carry-on bag or one checked bag, and calculate the total baggage fees for the entire family (round-trip). Use Allegiant's current baggage fee structure and specify whether fees are paid at booking or at the airport.

4. Accommodation: Select either a hotel or a licensed vacation rental in the Punta Gorda/Fort Myers area that can accommodate the family for 3 nights (covering the long weekend). Provide the accommodation name, nightly rate, and specify the cancellation policy. Calculate the total accommodation cost including Florida's 6% sales tax and an estimated 4% local tourist development tax.

5. Total Cost: Calculate the complete trip cost breakdown:
   - Base airfare for 4 round-trip tickets
   - Total baggage fees
   - Total accommodation cost (3 nights + taxes)
   - Grand total for the trip

Provide reference URLs supporting: (a) the Allegiant airport/route information, (b) the MLK Day 2026 date, (c) TSA REAL ID requirements, (d) Allegiant's baggage policy, and (e) the selected accommodation.
"""

# Ground truth contextual info used for validation where applicable
GROUND_TRUTH = {
    "mlk_day_2026": "Monday, January 19, 2026",
    "mlk_day_2026_weekend_outbound": "2026-01-16",
    "mlk_day_2026_weekend_return": "2026-01-19",
    "required_departure_airport_code": "PHL",
    "required_destination_airport_code": "PGD",
    "allegiant_airline_name": "Allegiant Air",
    "nights_expected": 3,
    "combined_tax_rate": 0.10,  # 6% Florida sales tax + 4% local tourist development tax
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FlightPlan(BaseModel):
    airline: Optional[str] = None
    departure_airport_name: Optional[str] = None
    departure_iata: Optional[str] = None
    destination_airport_name: Optional[str] = None
    destination_iata: Optional[str] = None
    outbound_date: Optional[str] = None    # Keep string format from answer; we'll parse if needed
    return_date: Optional[str] = None
    route_urls: List[str] = Field(default_factory=list)


class TSATravelerItem(BaseModel):
    label: Optional[str] = None  # e.g., Adult 1, Adult 2, Teen 16, Child 10
    age: Optional[str] = None
    id_guidance: Optional[str] = None


class TSACompliance(BaseModel):
    travelers: List[TSATravelerItem] = Field(default_factory=list)
    tsa_urls: List[str] = Field(default_factory=list)


class BaggagePassengerChoice(BaseModel):
    label: Optional[str] = None  # e.g., Adult 1, Teen 16, etc.
    bag_type: Optional[str] = None  # "carry-on" or "checked"
    per_one_way_fee: Optional[str] = None  # e.g., "$30"
    round_trip_fee: Optional[str] = None   # e.g., "$60"
    where_paid: Optional[str] = None       # e.g., "at booking", "at airport"


class BaggagePlan(BaseModel):
    passengers: List[BaggagePassengerChoice] = Field(default_factory=list)
    total_baggage_fees: Optional[str] = None
    payment_timing_text: Optional[str] = None  # statement about paid at booking vs airport
    baggage_policy_urls: List[str] = Field(default_factory=list)


class AccommodationPlan(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None  # "hotel" or "vacation_rental"
    location: Optional[str] = None  # city/area (e.g., Punta Gorda, Fort Myers)
    capacity_text: Optional[str] = None  # sleeps 4, two queen beds, etc.
    nightly_rate: Optional[str] = None
    nights: Optional[str] = None
    checkin_date: Optional[str] = None
    checkout_date: Optional[str] = None
    cancellation_policy: Optional[str] = None
    license_number: Optional[str] = None  # if vacation_rental
    license_url: Optional[str] = None     # if vacation_rental: official registry URL (if provided)
    accommodation_urls: List[str] = Field(default_factory=list)
    total_with_taxes: Optional[str] = None
    drive_time_or_distance_to_pgd: Optional[str] = None


class CostBreakdown(BaseModel):
    base_airfare_total: Optional[str] = None
    baggage_fees_total: Optional[str] = None
    accommodation_total: Optional[str] = None
    grand_total: Optional[str] = None


class AdditionalReferences(BaseModel):
    mlk_date_urls: List[str] = Field(default_factory=list)


class TripExtraction(BaseModel):
    flight: FlightPlan = Field(default_factory=FlightPlan)
    tsa: TSACompliance = Field(default_factory=TSACompliance)
    baggage: BaggagePlan = Field(default_factory=BaggagePlan)
    lodging: AccommodationPlan = Field(default_factory=AccommodationPlan)
    costs: CostBreakdown = Field(default_factory=CostBreakdown)
    refs: AdditionalReferences = Field(default_factory=AdditionalReferences)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip() -> str:
    return """
    Extract the complete trip plan as structured JSON. Pull values exactly as stated in the answer (do not invent). If something is missing, set it to null or empty list as appropriate.

    1) flight:
       - airline: Name of the airline used for the flight (e.g., "Allegiant Air").
       - departure_airport_name: Name of departure airport.
       - departure_iata: 3-letter IATA code of the departure airport (uppercase, e.g., "PHL").
       - destination_airport_name: Name of destination airport.
       - destination_iata: 3-letter IATA code of the destination airport (uppercase, e.g., "PGD").
       - outbound_date: Outbound flight date as written in the answer (e.g., "Fri Jan 16, 2026" or "2026-01-16").
       - return_date: Return flight date as written in the answer (e.g., "Mon Jan 19, 2026" or "2026-01-19").
       - route_urls: All URLs the answer cites for Allegiant's airports/routes.

    2) tsa:
       - travelers: An array of exactly the travelers for this family (two adults, one 16-year-old, one 10-year-old). For each traveler, extract:
           • label (e.g., "Adult 1", "Adult 2", "Teen 16", "Child 10")
           • age (as written, keep as string)
           • id_guidance (verbatim guidance from the answer for what ID the traveler needs)
       - tsa_urls: All URLs cited for TSA REAL ID rules.

    3) baggage:
       - passengers: For each of the 4 passengers, extract:
           • label (match the traveler labeling if present)
           • bag_type ("carry-on" or "checked"; use exactly these two values if answer implies one or the other)
           • per_one_way_fee: the fee used per one-way for that bag, as written (string, e.g., "$35")
           • round_trip_fee: the fee used for round-trip for that bag, as written (string)
           • where_paid: where the answer says the fee will be paid (e.g., "at booking", "at airport")
       - total_baggage_fees: The total baggage fees for the whole family round-trip, as a single amount string (e.g., "$240")
       - payment_timing_text: The answer's specific statement about whether baggage fees are paid at booking vs at the airport.
       - baggage_policy_urls: All URLs cited for Allegiant’s baggage policy/fees.

    4) lodging:
       - name: Accommodation name (hotel or vacation rental).
       - type: "hotel" or "vacation_rental" (pick one if clearly indicated).
       - location: City/area string as presented (e.g., "Punta Gorda" or "Fort Myers").
       - capacity_text: Exact wording indicating it can accommodate the family of 4 (e.g., "sleeps 4", "two queen beds", etc.) if present.
       - nightly_rate: Nightly rate used for calculation (string, like "$150").
       - nights: Number of nights (keep as string).
       - checkin_date: Check-in date as written.
       - checkout_date: Check-out date as written.
       - cancellation_policy: The stated cancellation policy (string).
       - license_number: If it is a vacation rental and a license number is provided, extract it; otherwise null.
       - license_url: If a registry or official license URL is provided, extract it; otherwise null.
       - accommodation_urls: All URLs supporting the property’s rate/policy/location.
       - total_with_taxes: Total accommodation cost for 3 nights including taxes as used in the answer (string).
       - drive_time_or_distance_to_pgd: If provided, the stated drive time or distance to PGD.

    5) costs:
       - base_airfare_total: Total base airfare for 4 round-trip tickets (string).
       - baggage_fees_total: Baggage fees total line item used in the final cost breakdown (string).
       - accommodation_total: Accommodation total line item used in the final cost breakdown (string).
       - grand_total: Grand total for the trip (string).

    6) refs:
       - mlk_date_urls: All URLs cited to support "MLK Day 2026 is Monday, January 19, 2026".

    Important:
    - Keep all money values as strings with the symbols if present (e.g., "$250"). Do not convert to numbers.
    - For URLs: Only include explicit URLs shown in the answer.
    - If a field is missing in the answer, return null (for a single value) or [] (for list).
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
_MONEY_RE = re.compile(r"[-+]?\d*[\.,]?\d+")

def parse_money_to_float(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # Remove currency symbols and commas
    s_clean = s.replace("$", "").replace(",", "").strip()
    # Extract first number-like token
    m = _MONEY_RE.search(s_clean)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except Exception:
        return None


def normalize_iata(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    return code.strip().upper()


def try_parse_date(d: Optional[str]) -> Optional[datetime]:
    if not d or not d.strip():
        return None
    s = d.strip()
    fmts = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%a %b %d, %Y",
        "%a, %b %d, %Y",
        "%b %d %Y",
        "%B %d %Y",
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f)
        except Exception:
            continue
    # Try to find yyyy-mm-dd like
    m = re.search(r"(20\d{2})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])", s)
    if m:
        y, mo, da = m.group(1), m.group(2), m.group(3)
        try:
            return datetime(int(y), int(mo), int(da))
        except Exception:
            pass
    return None


def dates_match_mlk_weekend(outbound: Optional[str], return_d: Optional[str]) -> bool:
    od = try_parse_date(outbound)
    rd = try_parse_date(return_d)
    if not od or not rd:
        return False
    target_out = try_parse_date(GROUND_TRUTH["mlk_day_2026_weekend_outbound"])
    target_ret = try_parse_date(GROUND_TRUTH["mlk_day_2026_weekend_return"])
    if not target_out or not target_ret:
        return False
    return (od.date() == target_out.date()) and (rd.date() == target_ret.date())


def str_contains_any(s: Optional[str], keywords: List[str]) -> bool:
    if not s:
        return False
    low = s.lower()
    return any(k.lower() in low for k in keywords)


def extract_numeric(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def approx_equal(a: Optional[float], b: Optional[float], tol: float = 1.5) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def get_traveler_buckets(travelers: List[TSATravelerItem]) -> Tuple[int, bool, bool]:
    """
    Return:
        adult_count (int),
        has_teen_16 (bool),
        has_child_10 (bool)
    """
    adult_count = 0
    has_teen_16 = False
    has_child_10 = False

    for t in travelers:
        label = (t.label or "").lower()
        age_text = (t.age or "").lower()

        # adults
        if "adult" in label or "18" in age_text or "19" in age_text or "20" in age_text or "21" in age_text:
            adult_count += 1

        # teen 16
        if "16" in label or "16" in age_text:
            has_teen_16 = True

        # child 10
        if "10" in label or ("child" in label and "10" in age_text):
            has_child_10 = True

    return adult_count, has_teen_16, has_child_10


def lodging_total_expected(nightly_rate: Optional[str], nights: Optional[str]) -> Optional[float]:
    rate = parse_money_to_float(nightly_rate)
    n = extract_numeric(nights) or GROUND_TRUTH["nights_expected"]
    if rate is None:
        return None
    return round(rate * n * (1.0 + GROUND_TRUTH["combined_tax_rate"]), 2)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_flight_details(evaluator: Evaluator, parent, trip: TripExtraction) -> None:
    node = evaluator.add_parallel(
        id="Flight_Details",
        desc="Provide a correct Allegiant flight plan to PGD for the MLK Day 2026 long weekend using the required Northeast departure airport per constraints.",
        parent=parent,
        critical=False
    )

    # Airline_Is_Allegiant (simple verify)
    airline_leaf = evaluator.add_leaf(
        id="Airline_Is_Allegiant",
        desc="Flight plan specifies Allegiant Air as the airline.",
        parent=node,
        critical=True
    )
    airline_claim = f"The stated airline '{trip.flight.airline or ''}' is Allegiant Air."
    await evaluator.verify(
        claim=airline_claim,
        node=airline_leaf,
        additional_instruction="Pass if the airline mentioned is Allegiant Air (allow minor formatting/case variations)."
    )

    # Departure_Airport_Matches_Constraint_With_Code (PHL)
    dep_code = normalize_iata(trip.flight.departure_iata)
    dep_custom = evaluator.add_custom_node(
        result=(dep_code == GROUND_TRUTH["required_departure_airport_code"]),
        id="Departure_Airport_Matches_Constraint_With_Code",
        desc="Departure airport matches the constraint-specified Northeast airport (PHL) and includes the departure IATA code.",
        parent=node,
        critical=True
    )

    # Destination_Is_PGD_With_Code
    dest_code = normalize_iata(trip.flight.destination_iata)
    dest_custom = evaluator.add_custom_node(
        result=(dest_code == GROUND_TRUTH["required_destination_airport_code"]),
        id="Destination_Is_PGD_With_Code",
        desc="Destination airport is Punta Gorda (PGD) and includes the destination IATA code.",
        parent=node,
        critical=True
    )

    # Travel_Dates_Match_MLK_Weekend_And_3_Nights (custom check)
    date_ok = dates_match_mlk_weekend(trip.flight.outbound_date, trip.flight.return_date)
    date_custom = evaluator.add_custom_node(
        result=date_ok,
        id="Travel_Dates_Match_MLK_Weekend_And_3_Nights",
        desc="Outbound/return dates align with MLK Day 2026 long weekend and the 3-night duration in constraints (Fri Jan 16, 2026 to Mon Jan 19, 2026; MLK Day is Monday).",
        parent=node,
        critical=True
    )


async def verify_tsa_compliance(evaluator: Evaluator, parent, trip: TripExtraction) -> None:
    node = evaluator.add_parallel(
        id="TSA_Compliance_All_4_Members",
        desc="Specify required identification to board for each of the 4 family members under TSA REAL ID rules effective May 2025.",
        parent=parent,
        critical=False
    )

    # Lists_ID_For_Each_Of_4_Travelers
    travelers = trip.tsa.travelers or []
    adult_count, has_teen_16, has_child_10 = get_traveler_buckets(travelers)
    list_all = evaluator.add_custom_node(
        result=(len(travelers) >= 4 and adult_count >= 2 and has_teen_16 and has_child_10),
        id="Lists_ID_For_Each_Of_4_Travelers",
        desc="Explicitly provides ID/document guidance for each traveler (two adults, age 16, age 10), not only generic group guidance.",
        parent=node,
        critical=True
    )

    # Adult_ID_Rule_Correct (simple verify)
    adult_rule = evaluator.add_leaf(
        id="Adult_ID_Rule_Correct",
        desc="States that adults (18+) must present REAL ID-compliant ID or an acceptable TSA alternative (e.g., passport) under the May 2025 enforcement.",
        parent=node,
        critical=True
    )
    adult_guidance_combined = " ".join([(t.id_guidance or "") for t in travelers if t and (t.label or "").lower().find("adult") >= 0])
    adult_claim = "Adults aged 18 or older must present a REAL ID–compliant ID or an acceptable TSA alternative such as a U.S. passport to fly domestically (enforced starting May 2025)."
    await evaluator.verify(
        claim=adult_claim,
        node=adult_rule,
        additional_instruction="Judge based on the answer text: pass if the guidance for adults states REAL ID-compliant ID or acceptable alternatives (e.g., passport) are required under May 2025 enforcement."
    )

    # Minor_ID_Rule_Correct (simple verify)
    minor_rule = evaluator.add_leaf(
        id="Minor_ID_Rule_Correct",
        desc="States that minors under 18 do not need ID for domestic flights when traveling with a parent/guardian.",
        parent=node,
        critical=True
    )
    minor_claim = "Minors under 18 do not need ID for domestic flights when traveling with a parent or guardian."
    await evaluator.verify(
        claim=minor_claim,
        node=minor_rule,
        additional_instruction="Judge based on the answer text: pass if the answer states minors under 18 do not need ID when traveling domestically with a parent/guardian."
    )


async def verify_baggage_plan(evaluator: Evaluator, parent, trip: TripExtraction) -> None:
    node = evaluator.add_parallel(
        id="Baggage_Planning_All_4_Passengers",
        desc="Baggage choice per passenger and correct total baggage fees (round-trip) using Allegiant policy including when fees are paid.",
        parent=parent,
        critical=False
    )

    # Bag_Choice_Per_Passenger
    pax = trip.baggage.passengers or []
    all_have_choice = (len(pax) >= 4) and all(
        (p is not None) and (p.bag_type or "").strip().lower() in ("carry-on", "checked")
        for p in pax
    )
    bag_choice_node = evaluator.add_custom_node(
        result=all_have_choice,
        id="Bag_Choice_Per_Passenger",
        desc="For each passenger, specifies a baggage choice as requested (one carry-on OR one checked bag).",
        parent=node,
        critical=True
    )

    # Total_Baggage_Fees_RoundTrip_Calculated (check existence of total)
    total_baggage_present = trip.baggage.total_baggage_fees is not None and str(trip.baggage.total_baggage_fees).strip() != ""
    total_baggage_node = evaluator.add_custom_node(
        result=total_baggage_present,
        id="Total_Baggage_Fees_RoundTrip_Calculated",
        desc="Calculates the total baggage fees for the entire family for round-trip travel using Allegiant’s fee structure.",
        parent=node,
        critical=True
    )

    # Payment_Timing_Stated (simple verify using provided statement)
    payment_leaf = evaluator.add_leaf(
        id="Payment_Timing_Stated",
        desc="States whether baggage fees are paid at booking vs at the airport (as requested).",
        parent=node,
        critical=True
    )
    timing_text = trip.baggage.payment_timing_text or ""
    timing_claim = f"The answer explicitly states where baggage fees are paid (e.g., at booking vs at the airport). Provided statement: {timing_text}"
    await evaluator.verify(
        claim=timing_claim,
        node=payment_leaf,
        additional_instruction="Pass if the statement clearly indicates payment timing (e.g., paid at booking, or higher at airport). Do not require exact fee amounts here."
    )


async def verify_accommodation(evaluator: Evaluator, parent, trip: TripExtraction) -> None:
    node = evaluator.add_parallel(
        id="Accommodation",
        desc="Select a hotel or licensed vacation rental meeting the 3-night MLK weekend stay and compute lodging costs with taxes.",
        parent=parent,
        critical=False
    )

    # Accommodation_Name_Provided
    name_ok = bool((trip.lodging.name or "").strip())
    evaluator.add_custom_node(
        result=name_ok,
        id="Accommodation_Name_Provided",
        desc="Provides the accommodation name (hotel or vacation rental).",
        parent=node,
        critical=True
    )

    # Accommodation_Location_Matches_Area_Constraint (verify by URL(s) if provided, otherwise simple)
    loc_leaf = evaluator.add_leaf(
        id="Accommodation_Location_Matches_Area_Constraint",
        desc="Accommodation is in the Punta Gorda/Fort Myers area (per constraints).",
        parent=node,
        critical=True
    )
    loc_claim = "This property is located in the Punta Gorda or Fort Myers area in Florida."
    loc_sources = trip.lodging.accommodation_urls if trip.lodging.accommodation_urls else None
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=loc_sources,
        additional_instruction="Look for city/area on the page. Pass if it indicates Punta Gorda, Port Charlotte, or Fort Myers area (Lee/Charlotte County)."
    )

    # Accommodation_Capacity_And_Stay_Length (custom logic)
    # Check can accommodate 4 and planned for 3 nights (check-in Jan 16, checkout Jan 19 if available)
    capacity_ok = str_contains_any(trip.lodging.capacity_text, ["4", "sleeps 4", "two queen", "two double", "two beds", "sofa bed", "family of 4"])
    nights_n = extract_numeric(trip.lodging.nights) or GROUND_TRUTH["nights_expected"]
    stay_dates_ok = dates_match_mlk_weekend(trip.lodging.checkin_date, trip.lodging.checkout_date)
    evaluator.add_custom_node(
        result=(capacity_ok and nights_n == GROUND_TRUTH["nights_expected"] and stay_dates_ok),
        id="Accommodation_Capacity_And_Stay_Length",
        desc="Accommodation can accommodate 4 people and is planned for 3 nights covering the long weekend.",
        parent=node,
        critical=True
    )

    # Nightly_Rate_Provided
    evaluator.add_custom_node(
        result=(trip.lodging.nightly_rate is not None and str(trip.lodging.nightly_rate).strip() != ""),
        id="Nightly_Rate_Provided",
        desc="Provides the nightly rate used for the cost calculation.",
        parent=node,
        critical=True
    )

    # Cancellation_Policy_Provided
    evaluator.add_custom_node(
        result=(trip.lodging.cancellation_policy is not None and str(trip.lodging.cancellation_policy).strip() != ""),
        id="Cancellation_Policy_Provided",
        desc="States the cancellation policy for the selected accommodation.",
        parent=node,
        critical=True
    )

    # Vacation_Rental_License_Evidence_If_Applicable
    is_vr = (trip.lodging.type or "").strip().lower() == "vacation_rental"
    has_license_evidence = bool((trip.lodging.license_number or "").strip() or (trip.lodging.license_url or "").strip())
    evaluator.add_custom_node(
        result=(not is_vr) or (is_vr and has_license_evidence),
        id="Vacation_Rental_License_Evidence_If_Applicable",
        desc="If (and only if) a vacation rental is selected, provides verifiable evidence it is licensed (e.g., license number and/or official registry URL).",
        parent=node,
        critical=True
    )

    # Accommodation_Total_With_Taxes_Calculated (recompute and compare)
    expected_total = lodging_total_expected(trip.lodging.nightly_rate, trip.lodging.nights)
    stated_total = parse_money_to_float(trip.lodging.total_with_taxes)
    evaluator.add_custom_node(
        result=approx_equal(expected_total, stated_total, tol=2.0),
        id="Accommodation_Total_With_Taxes_Calculated",
        desc="Calculates total accommodation cost for 3 nights including 6% Florida sales tax plus estimated 4% local tourist development tax.",
        parent=node,
        critical=True
    )

    # Distance_Or_Drive_Time_To_PGD_Disclosed (non-critical)
    evaluator.add_custom_node(
        result=bool((trip.lodging.drive_time_or_distance_to_pgd or "").strip()),
        id="Distance_Or_Drive_Time_To_PGD_Disclosed",
        desc="States distance or drive time to PGD to support the 'reasonable proximity' constraint (without imposing an invented threshold).",
        parent=node,
        critical=False
    )


async def verify_total_costs(evaluator: Evaluator, parent, trip: TripExtraction) -> None:
    node = evaluator.add_parallel(
        id="Total_Cost_Breakdown",
        desc="Compute and present the required total trip cost components and the grand total.",
        parent=parent,
        critical=False
    )

    # Parse amounts
    base_air = parse_money_to_float(trip.costs.base_airfare_total)
    bag_cost_line = parse_money_to_float(trip.costs.baggage_fees_total)
    lodg_cost_line = parse_money_to_float(trip.costs.accommodation_total)
    grand = parse_money_to_float(trip.costs.grand_total)
    bag_section_total = parse_money_to_float(trip.baggage.total_baggage_fees)
    lodging_section_total = parse_money_to_float(trip.lodging.total_with_taxes)

    # Base_Airfare_Total_4_RoundTrip
    evaluator.add_custom_node(
        result=(base_air is not None and base_air > 0),
        id="Base_Airfare_Total_4_RoundTrip",
        desc="States the base airfare total for 4 round-trip tickets.",
        parent=node,
        critical=True
    )

    # Baggage_Fees_Line_Item_Included_And_Consistent
    evaluator.add_custom_node(
        result=(bag_cost_line is not None and approx_equal(bag_cost_line, bag_section_total, tol=2.0)),
        id="Baggage_Fees_Line_Item_Included_And_Consistent",
        desc="Includes a baggage-fees line item in the total cost breakdown and it matches the baggage total computed in the baggage section.",
        parent=node,
        critical=True
    )

    # Accommodation_Cost_Line_Item_Included_And_Consistent
    evaluator.add_custom_node(
        result=(lodg_cost_line is not None and approx_equal(lodg_cost_line, lodging_section_total, tol=2.0)),
        id="Accommodation_Cost_Line_Item_Included_And_Consistent",
        desc="Includes an accommodation-cost line item in the total cost breakdown and it matches the accommodation total computed in the lodging section (3 nights + taxes).",
        parent=node,
        critical=True
    )

    # Grand_Total_Sum_Correct
    sum_components = None
    if base_air is not None and bag_cost_line is not None and lodg_cost_line is not None:
        sum_components = base_air + bag_cost_line + lodg_cost_line
        sum_components = round(sum_components, 2)
    evaluator.add_custom_node(
        result=approx_equal(sum_components, grand, tol=2.0),
        id="Grand_Total_Sum_Correct",
        desc="Grand total equals base airfare total + total baggage fees + total accommodation cost.",
        parent=node,
        critical=True
    )


async def verify_references(evaluator: Evaluator, parent, trip: TripExtraction) -> None:
    node = evaluator.add_parallel(
        id="Reference_URLs_All_Required",
        desc="Provide reference URLs supporting required facts/policies and selected lodging details.",
        parent=parent,
        critical=False
    )

    # URL_Allegiant_Route_And_Airport_Info
    allegiant_route_leaf = evaluator.add_leaf(
        id="URL_Allegiant_Route_And_Airport_Info",
        desc="Provides at least one reference URL supporting the Allegiant airport/route information used.",
        parent=node,
        critical=True
    )
    dep = normalize_iata(trip.flight.departure_iata) or "PHL"
    dst = normalize_iata(trip.flight.destination_iata) or "PGD"
    route_claim = f"Allegiant operates service between {dep} and {dst} in or by January 2026."
    await evaluator.verify(
        claim=route_claim,
        node=allegiant_route_leaf,
        sources=trip.flight.route_urls,
        additional_instruction="Pass if the page(s) indicate Allegiant service on the stated route or from the stated departure airport to Punta Gorda (PGD) around Jan 2026; official route pages, news releases, or schedules count."
    )

    # URL_MLK_Day_2026_Date
    mlk_leaf = evaluator.add_leaf(
        id="URL_MLK_Day_2026_Date",
        desc="Provides at least one reference URL supporting the MLK Day 2026 date (Monday).",
        parent=node,
        critical=True
    )
    mlk_claim = "In 2026, Martin Luther King Jr. Day is Monday, January 19, 2026."
    await evaluator.verify(
        claim=mlk_claim,
        node=mlk_leaf,
        sources=trip.refs.mlk_date_urls,
        additional_instruction="Verify the federal holiday date shown on the page matches Monday, January 19, 2026."
    )

    # URL_TSA_REAL_ID_Requirements
    tsa_leaf = evaluator.add_leaf(
        id="URL_TSA_REAL_ID_Requirements",
        desc="Provides at least one reference URL supporting TSA REAL ID requirements effective May 2025.",
        parent=node,
        critical=True
    )
    tsa_claim = "Beginning May 7, 2025, adult airline passengers (18+) must present a REAL ID–compliant license or acceptable alternative to fly domestically."
    await evaluator.verify(
        claim=tsa_claim,
        node=tsa_leaf,
        sources=trip.tsa.tsa_urls,
        additional_instruction="Look for TSA or DHS pages describing REAL ID enforcement beginning May 7, 2025 and ID requirements for adults."
    )

    # URL_Allegiant_Baggage_Policy
    bag_leaf = evaluator.add_leaf(
        id="URL_Allegiant_Baggage_Policy",
        desc="Provides at least one reference URL supporting Allegiant’s baggage policy/fees used.",
        parent=node,
        critical=True
    )
    bag_claim = "This page presents Allegiant’s baggage policy and fees information for carry-on and checked baggage, including any differences by purchase timing."
    await evaluator.verify(
        claim=bag_claim,
        node=bag_leaf,
        sources=trip.baggage.baggage_policy_urls,
        additional_instruction="Pass if the page covers Allegiant baggage policy and fee information; exact amounts may vary by route/date."
    )

    # URL_Accommodation_Rate_And_Cancellation
    lodge_leaf = evaluator.add_leaf(
        id="URL_Accommodation_Rate_And_Cancellation",
        desc="Provides at least one reference URL supporting the selected accommodation’s nightly rate and cancellation policy.",
        parent=node,
        critical=True
    )
    lodge_claim = "This page displays the accommodation’s nightly rate and states the cancellation policy."
    await evaluator.verify(
        claim=lodge_claim,
        node=lodge_leaf,
        sources=trip.lodging.accommodation_urls,
        additional_instruction="Look for room rate/pricing and cancellation policy on the page; dynamic pricing display is acceptable if the page indicates rate and cancellation terms."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate the provided answer against the MLK 2026 Gulf Coast Allegiant trip planning rubric.
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

    # Extraction
    trip = await evaluator.extract(
        prompt=prompt_extract_trip(),
        template_class=TripExtraction,
        extraction_name="trip_plan_extraction"
    )

    # Ground truth info (for transparency)
    evaluator.add_ground_truth({
        "mlk_day_2026": GROUND_TRUTH["mlk_day_2026"],
        "required_dates": {
            "outbound": GROUND_TRUTH["mlk_day_2026_weekend_outbound"],
            "return": GROUND_TRUTH["mlk_day_2026_weekend_return"]
        },
        "required_airports": {
            "departure": GROUND_TRUTH["required_departure_airport_code"],
            "destination": GROUND_TRUTH["required_destination_airport_code"]
        },
        "expected_nights": GROUND_TRUTH["nights_expected"],
        "combined_tax_rate": "10% (6% FL state + 4% local)"
    }, gt_type="ground_truth")

    # Build the rubric tree under a top non-critical aggregator to allow mixed criticalities
    task_root = evaluator.add_parallel(
        id="Complete_MLK_Weekend_Trip_Planning",
        desc="Plan the complete MLK Day 2026 long-weekend trip (flights, TSA IDs, baggage, lodging, total cost, references) per the question + constraints.",
        parent=root,
        critical=False
    )

    # Sub-sections
    await verify_flight_details(evaluator, task_root, trip)
    await verify_tsa_compliance(evaluator, task_root, trip)
    await verify_baggage_plan(evaluator, task_root, trip)
    await verify_accommodation(evaluator, task_root, trip)
    await verify_total_costs(evaluator, task_root, trip)
    await verify_references(evaluator, task_root, trip)

    return evaluator.get_summary()