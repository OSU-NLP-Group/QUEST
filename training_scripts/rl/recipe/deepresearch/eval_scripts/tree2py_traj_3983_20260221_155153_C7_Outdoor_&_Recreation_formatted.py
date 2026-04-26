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
TASK_ID = "family_disney_world_trip_planning"
TASK_DESCRIPTION = """A family of four (2 adults and 2 children ages 6 and 8) is planning a 4-day vacation to Walt Disney World from Bangor, Maine. They want to minimize their travel costs while ensuring a convenient trip.

Please provide a comprehensive transportation plan that includes:

1. Flight Information: Identify which airline offers direct (non-stop) flights from Bangor International Airport to a Florida airport that serves the Disney World area. Specify both the departure airport code and the arrival airport code.

2. Seating Requirements: Determine how many airline seats the family needs to purchase, considering airline policies for children of these ages.

3. Airport-to-Disney Distance: Provide the approximate driving distance (in miles) from the arrival airport to Walt Disney World.

4. Baggage Allowances and Fees: 
   - Identify how many free personal items the family receives on this airline
   - Explain the carry-on baggage fee policy for this airline
   - Calculate the total cost if the family chooses to bring 4 carry-on bags (using pre-booked pricing)

5. Ground Transportation: Identify at least one ground transportation service available from the arrival airport to Disney World hotels, and provide the approximate cost for transporting the family of 4.

6. Airport Parking: Identify the parking options at Bangor International Airport, provide the daily parking rate, and calculate the total parking cost for their 4-day trip.

7. Travel Policy Compliance: Confirm that the family composition (2 adults with children ages 6 and 8) complies with the airline's child travel supervision policies.

8. Cost Summary: Provide an estimated breakdown of the total transportation costs including ground transportation and parking (flight ticket prices may be excluded from the total).

All information must be grounded in current (2026) policies and publicly available information.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FlightInfo(BaseModel):
    airline: Optional[str] = None
    depart_code: Optional[str] = None
    arrive_code: Optional[str] = None
    flight_source_urls: List[str] = Field(default_factory=list)


class SeatingInfo(BaseModel):
    seats_to_purchase: Optional[str] = None
    seat_policy_urls: List[str] = Field(default_factory=list)


class DistanceInfo(BaseModel):
    distance_miles: Optional[str] = None
    distance_source_urls: List[str] = Field(default_factory=list)


class BaggageInfo(BaseModel):
    free_personal_item_policy: Optional[str] = None
    free_personal_items_for_family: Optional[str] = None
    carry_on_fee_policy: Optional[str] = None
    carry_on_prebook_price_per_bag: Optional[str] = None
    carry_on_at_airport_price_per_bag: Optional[str] = None
    carry_on_total_cost_for_4: Optional[str] = None
    baggage_policy_urls: List[str] = Field(default_factory=list)


class GroundTransportInfo(BaseModel):
    service_name: Optional[str] = None
    cost_for_family: Optional[str] = None
    transport_urls: List[str] = Field(default_factory=list)


class ParkingInfo(BaseModel):
    parking_options: List[str] = Field(default_factory=list)
    daily_rate: Optional[str] = None
    total_cost_4_days: Optional[str] = None
    parking_urls: List[str] = Field(default_factory=list)


class ChildPolicyInfo(BaseModel):
    compliance_statement: Optional[str] = None
    child_policy_urls: List[str] = Field(default_factory=list)


class CostSummary(BaseModel):
    includes_ground_transport: Optional[bool] = None
    ground_transport_cost: Optional[str] = None
    includes_parking: Optional[bool] = None
    parking_cost: Optional[str] = None
    baggage_carry_on_cost: Optional[str] = None
    total_estimated_cost: Optional[str] = None


class PlanExtraction(BaseModel):
    flight: Optional[FlightInfo] = None
    seating: Optional[SeatingInfo] = None
    distance: Optional[DistanceInfo] = None
    baggage: Optional[BaggageInfo] = None
    transport: Optional[GroundTransportInfo] = None
    parking: Optional[ParkingInfo] = None
    child_policy: Optional[ChildPolicyInfo] = None
    cost_summary: Optional[CostSummary] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
    Extract the transportation plan details for the family trip exactly as presented in the answer. Return a JSON object with the following nested structure (return null for fields that are not mentioned):

    flight:
      airline: The named airline for the direct (non-stop) flight
      depart_code: The departure airport code (expected "BGR" if provided)
      arrive_code: The arrival Florida airport code serving the Disney area (e.g., MCO, SFB)
      flight_source_urls: All URLs cited that support the airline/route/flight information

    seating:
      seats_to_purchase: The number of airline seats the family should buy as stated (e.g., "4")
      seat_policy_urls: URL(s) cited for the airline's child seating policy

    distance:
      distance_miles: The approximate driving distance in miles from the arrival airport to Walt Disney World, as stated
      distance_source_urls: URL(s) used to support the driving distance (e.g., map or airport/destination info)

    baggage:
      free_personal_item_policy: The stated policy text about free personal items (e.g., "one free personal item per passenger")
      free_personal_items_for_family: The stated total number of free personal items the family gets (e.g., "4")
      carry_on_fee_policy: The stated explanation of carry-on fee policy, including any difference between pre-booked and at-airport
      carry_on_prebook_price_per_bag: The stated pre-booked price per carry-on bag (e.g., "$30")
      carry_on_at_airport_price_per_bag: The stated at-airport price per carry-on bag if mentioned (e.g., "$50")
      carry_on_total_cost_for_4: The stated total cost for 4 carry-on bags using pre-booked pricing (e.g., "$120")
      baggage_policy_urls: URL(s) cited for baggage policy and pricing

    transport:
      service_name: The named ground transportation service from the arrival airport to Disney hotels (e.g., "Mears Connect")
      cost_for_family: The approximate cost for a family of 4 (specify one-way or round trip only if stated in the answer)
      transport_urls: URL(s) cited that support availability and/or pricing

    parking:
      parking_options: A list of the named parking options at Bangor International Airport as stated (e.g., ["Short-Term", "Long-Term"])
      daily_rate: The daily parking rate used in the calculation as stated (e.g., "$12/day")
      total_cost_4_days: The stated total parking cost for the 4-day trip (e.g., "$48")
      parking_urls: URL(s) cited for BGR parking information and rates

    child_policy:
      compliance_statement: The stated confirmation that 2 adults with children ages 6 and 8 comply with the airline's child travel policy
      child_policy_urls: URL(s) cited for the airline's child travel policy

    cost_summary:
      includes_ground_transport: true/false whether ground transportation cost is included in the total cost breakdown
      ground_transport_cost: The cost included for ground transportation in the summary
      includes_parking: true/false whether airport parking cost is included in the total cost breakdown
      parking_cost: The cost included for parking in the summary
      baggage_carry_on_cost: The carry-on bag cost included (if included) in the summary
      total_estimated_cost: The stated grand total of the included components in the summary

    SPECIAL RULES:
    - Extract values exactly as presented in the answer (use strings for prices or distances, including symbols).
    - For URL lists, extract all URLs explicitly mentioned; include both plain and markdown links.
    - If any section or field is missing, return null for it.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _parse_money(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        m = re.findall(r"[-+]?\d*\.?\d+", value.replace(",", ""))
        return float(m[0]) if m else None
    except Exception:
        return None


def _parse_number(value: Optional[str]) -> Optional[float]:
    return _parse_money(value)


def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


def _nonempty_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Verification tree construction & checks                                     #
# --------------------------------------------------------------------------- #
async def verify_source_and_year_grounding(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Source_And_Year_Grounding",
        desc="Information and cost calculations are grounded in publicly available 2026 policies/pricing as required.",
        parent=parent,
        critical=True
    )

    # Leaf 1: The answer indicates information/prices are for 2026 (simple check within the answer text)
    year_leaf = evaluator.add_leaf(
        id="Year_2026_Mentioned",
        desc="Answer indicates information/pricing is current for 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that information or prices are current as of 2026 (e.g., mentions '2026' or 'current in 2026').",
        node=year_leaf,
        additional_instruction="Look for explicit mentions of '2026' or statements that policies/pricing are current in 2026. Accept if any section clearly refers to 2026."
    )

    # Leaf 2: Key sources provided for all critical components (existence check)
    flight_urls = _safe_urls(plan.flight.flight_source_urls) if plan.flight else []
    baggage_urls = _safe_urls(plan.baggage.baggage_policy_urls) if plan.baggage else []
    transport_urls = _safe_urls(plan.transport.transport_urls) if plan.transport else []
    parking_urls = _safe_urls(plan.parking.parking_urls) if plan.parking else []
    child_urls = _safe_urls(plan.child_policy.child_policy_urls) if plan.child_policy else []

    all_present = all([
        len(flight_urls) > 0,
        len(baggage_urls) > 0,
        len(transport_urls) > 0,
        len(parking_urls) > 0,
        len(child_urls) > 0
    ])
    evaluator.add_custom_node(
        result=all_present,
        id="Key_Sources_Provided",
        desc="Key sources provided for flight, baggage policy, ground transport, parking, and child policy.",
        parent=node,
        critical=True
    )


async def verify_flight_information(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Flight_Information",
        desc="Provides required direct-flight information from BGR to a Florida airport serving the Disney World area.",
        parent=parent,
        critical=True
    )

    airline = plan.flight.airline if plan.flight else None
    depart_code = plan.flight.depart_code if plan.flight else None
    arrive_code = plan.flight.arrive_code if plan.flight else None
    flight_urls = _safe_urls(plan.flight.flight_source_urls) if plan.flight else []

    # Direct flight airline check (with source)
    direct_leaf = evaluator.add_leaf(
        id="Direct_Flight_Airline",
        desc="Identifies an airline that offers a direct (non-stop) flight meeting the route constraint (BGR to a Florida airport serving Disney World area).",
        parent=node,
        critical=True
    )
    claim_direct = f"The airline {airline or '[airline not specified]'} offers a direct (non-stop) flight from BGR (Bangor) to {arrive_code or 'a Florida airport serving the Disney World area'}."
    await evaluator.verify(
        claim=claim_direct,
        node=direct_leaf,
        sources=flight_urls,
        additional_instruction="Verify that the airline operates (or has operated) a non-stop route from Bangor (BGR) to a Florida airport serving the Disney World area (e.g., Orlando MCO or Sanford SFB). Seasonal service counts."
    )

    # Departure airport code must be BGR (existence/accuracy check)
    evaluator.add_custom_node(
        result=(depart_code is not None and depart_code.strip().upper() == "BGR"),
        id="Departure_Airport_Code",
        desc="Specifies the departure airport code as Bangor International Airport (BGR).",
        parent=node,
        critical=True
    )

    # Arrival airport code existence (3-letter code)
    evaluator.add_custom_node(
        result=(arrive_code is not None and len(arrive_code.strip()) == 3),
        id="Arrival_Airport_Code",
        desc="Specifies the arrival Florida airport code that serves the Disney World area for the selected direct flight.",
        parent=node,
        critical=True
    )


async def verify_seating_requirements(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Seating_Requirements",
        desc="Determines how many seats must be purchased for 2 adults and children ages 6 and 8; children age 2+ require their own seat.",
        parent=parent,
        critical=True
    )

    # Seat count correctness (simple logical verification)
    count_leaf = evaluator.add_leaf(
        id="Seat_Count_Correct",
        desc="Correctly states the number of seats to purchase (4 seats for 2 adults + ages 6 and 8).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="For a family with 2 adults and children ages 6 and 8, the number of airline seats to purchase is exactly 4.",
        node=count_leaf,
        additional_instruction="This is a simple logical check based on standard airline policy that children age 2 and over require their own seat."
    )

    # Policy support: children 2+ require their own seat (source-grounded)
    policy_leaf = evaluator.add_leaf(
        id="Seat_Policy_Supported",
        desc="Child seating policy supported by cited source(s).",
        parent=node,
        critical=True
    )
    seat_policy_urls = _safe_urls(plan.seating.seat_policy_urls) if plan.seating else []
    airline_name = plan.flight.airline if plan.flight else "the airline"
    await evaluator.verify(
        claim=f"According to the cited policy, children age 2 and older require their own ticketed seat on {airline_name}.",
        node=policy_leaf,
        sources=seat_policy_urls,
        additional_instruction="Verify that the airline's child travel policy indicates children aged 2+ must occupy their own seat (lap infants under 2 are the usual exception)."
    )


async def verify_airport_to_disney_distance(evaluator: Evaluator, parent, plan: PlanExtraction):
    # Single critical leaf at root level
    dist_leaf = evaluator.add_leaf(
        id="Airport_to_Disney_Distance",
        desc="Provides the approximate driving distance (miles) from the arrival airport to Walt Disney World.",
        parent=parent,
        critical=True
    )
    distance_str = plan.distance.distance_miles if plan.distance else None
    arrive_code = plan.flight.arrive_code if plan.flight else None
    dist_urls = _safe_urls(plan.distance.distance_source_urls) if plan.distance else []
    claim_dist = f"The driving distance from {arrive_code or 'the arrival airport'} to Walt Disney World is approximately {distance_str or '[distance not specified]'} miles."
    await evaluator.verify(
        claim=claim_dist,
        node=dist_leaf,
        sources=dist_urls,
        additional_instruction="Allow reasonable rounding (e.g., within ~10%). Confirm via the cited map or information source."
    )


async def verify_baggage_allowances_and_fees(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Baggage_Allowances_And_Fees",
        desc="Provides personal-item allowance, carry-on fee policy, and total cost for 4 carry-on bags using pre-booked pricing.",
        parent=parent,
        critical=True
    )
    bag = plan.baggage or BaggageInfo()
    airline_name = plan.flight.airline if plan.flight else "the airline"
    bag_urls = _safe_urls(bag.baggage_policy_urls)

    # Free personal item allowance (policy-supported)
    personal_leaf = evaluator.add_leaf(
        id="Free_Personal_Item_Allowance",
        desc="Identifies how many free personal items the family receives on this airline.",
        parent=node,
        critical=True
    )
    if _nonempty_str(bag.free_personal_items_for_family):
        claim_pi = f"A family of four receives {bag.free_personal_items_for_family} free personal items on {airline_name}, consistent with the airline's policy (typically one free personal item per passenger)."
    else:
        claim_pi = f"{airline_name} allows one free personal item per passenger, implying 4 free personal items for a family of four."
    await evaluator.verify(
        claim=claim_pi,
        node=personal_leaf,
        sources=bag_urls,
        additional_instruction="Verify that the policy allows a personal item free of charge; if one per passenger, it implies 4 free personal items for a family of 4."
    )

    # Carry-on fee policy explanation (policy-supported)
    carry_leaf = evaluator.add_leaf(
        id="Carry_On_Fee_Policy",
        desc="Explains the airline’s carry-on baggage fee policy, including pre-booked vs at-airport pricing.",
        parent=node,
        critical=True
    )
    if _nonempty_str(bag.carry_on_prebook_price_per_bag) and _nonempty_str(bag.carry_on_at_airport_price_per_bag):
        claim_carry = (
            f"{airline_name} charges for carry-on bags; the pre-booked price is about "
            f"{bag.carry_on_prebook_price_per_bag}, and the at-airport price is higher at about "
            f"{bag.carry_on_at_airport_price_per_bag}."
        )
    else:
        claim_carry = (
            f"{airline_name} charges a fee for carry-on bags; pre-booked pricing is cheaper than at-airport pricing."
        )
    await evaluator.verify(
        claim=claim_carry,
        node=carry_leaf,
        sources=bag_urls,
        additional_instruction="Confirm that carry-on is not free and that pre-booked pricing is lower than at-airport pricing. If numeric values are provided, verify them."
    )

    # Total carry-on cost for 4 (math consistency check)
    total_leaf = evaluator.add_leaf(
        id="Carry_On_Total_Cost_For_4",
        desc="Calculates total carry-on cost for 4 bags using pre-booked pricing; math consistent with stated policy.",
        parent=node,
        critical=True
    )
    prebook_price = bag.carry_on_prebook_price_per_bag or ""
    total_for_4 = bag.carry_on_total_cost_for_4 or ""
    claim_total = (
        f"Given a pre-booked carry-on price of {prebook_price} per bag, the total for 4 carry-on bags is {total_for_4}."
    )
    await evaluator.verify(
        claim=claim_total,
        node=total_leaf,
        additional_instruction="Verify arithmetic consistency: total should be 4 times the per-bag pre-book price (allow minor rounding)."
    )


async def verify_ground_transportation(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Ground_Transportation",
        desc="Identifies at least one ground transportation service from the arrival airport to Disney hotels and provides an approximate cost for a family of 4.",
        parent=parent,
        critical=True
    )

    gt = plan.transport or GroundTransportInfo()
    arrive_code = plan.flight.arrive_code if plan.flight else None
    gt_urls = _safe_urls(gt.transport_urls)

    # Service availability
    service_leaf = evaluator.add_leaf(
        id="Ground_Transportation_Service",
        desc="Names at least one ground transportation service available from the arrival airport to Disney World area hotels.",
        parent=node,
        critical=True
    )
    claim_service = f"{gt.service_name or '[service not specified]'} provides transportation from {arrive_code or 'the arrival airport'} to Walt Disney World area hotels."
    await evaluator.verify(
        claim=claim_service,
        node=service_leaf,
        sources=gt_urls,
        additional_instruction="Verify that the named service offers transfers from the specified arrival airport to Disney or nearby hotels."
    )

    # Cost for family of 4
    cost_leaf = evaluator.add_leaf(
        id="Ground_Transportation_Cost",
        desc="Provides an approximate cost to transport the family of 4.",
        parent=node,
        critical=True
    )
    claim_cost = f"The approximate cost for a family of 4 using {gt.service_name or 'the service'} is {gt.cost_for_family or '[amount not specified]'}."
    await evaluator.verify(
        claim=claim_cost,
        node=cost_leaf,
        sources=gt_urls,
        additional_instruction="Verify the stated price or range for 4 passengers (one-way or round trip as stated). Allow reasonable approximation."
    )


async def verify_airport_parking(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Airport_Parking",
        desc="Identifies Bangor (BGR) parking options, provides daily rate, and calculates total parking cost for 4 days.",
        parent=parent,
        critical=True
    )

    pk = plan.parking or ParkingInfo()
    pk_urls = _safe_urls(pk.parking_urls)

    # Parking options identification (source-supported)
    options_leaf = evaluator.add_leaf(
        id="Parking_Options_At_BGR",
        desc="Identifies the parking options at Bangor International Airport.",
        parent=node,
        critical=True
    )
    if pk.parking_options:
        options_str = ", ".join(pk.parking_options)
        claim_opts = f"This page lists parking options at Bangor International Airport (BGR), such as {options_str}."
    else:
        claim_opts = "This page lists parking options and rates at Bangor International Airport (BGR)."
    await evaluator.verify(
        claim=claim_opts,
        node=options_leaf,
        sources=pk_urls,
        additional_instruction="Confirm that the cited page is the official BGR parking information page (or equivalent) that enumerates options/rates."
    )

    # Daily rate (source-supported)
    rate_leaf = evaluator.add_leaf(
        id="Parking_Daily_Rate",
        desc="Provides the daily parking rate at Bangor International Airport used for the calculation.",
        parent=node,
        critical=True
    )
    claim_rate = f"The daily parking rate used is {pk.daily_rate or '[rate not specified]'} at Bangor International Airport (BGR)."
    await evaluator.verify(
        claim=claim_rate,
        node=rate_leaf,
        sources=pk_urls,
        additional_instruction="Verify that the daily rate stated matches the cited BGR parking information."
    )

    # 4-day total math (simple verify)
    total_leaf = evaluator.add_leaf(
        id="Total_Parking_Cost_4_Days",
        desc="Calculates the total parking cost for the 4-day trip duration (math consistent with the stated daily rate).",
        parent=node,
        critical=True
    )
    claim_total = f"Given a daily parking rate of {pk.daily_rate or '[rate not specified]'}, the total for 4 days is {pk.total_cost_4_days or '[total not specified]'}."
    await evaluator.verify(
        claim=claim_total,
        node=total_leaf,
        additional_instruction="Verify arithmetic consistency: total should be 4 times the daily rate (allow minor rounding)."
    )


async def verify_child_travel_policy(evaluator: Evaluator, parent, plan: PlanExtraction):
    leaf = evaluator.add_leaf(
        id="Child_Travel_Policy_Compliance",
        desc="Confirms the family composition (2 adults with children ages 6 and 8) complies with the airline's child travel supervision policies.",
        parent=parent,
        critical=True
    )
    airline_name = plan.flight.airline if plan.flight else "the airline"
    child_urls = _safe_urls(plan.child_policy.child_policy_urls) if plan.child_policy else []
    claim = f"Two adults traveling with children ages 6 and 8 comply with {airline_name}'s child travel supervision policy."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=child_urls,
        additional_instruction="Verify that children ages 6 and 8 are allowed to travel when accompanied by adults per the airline's child travel policy."
    )


async def verify_cost_summary(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Cost_Summary",
        desc="Provides an estimated breakdown including ground transportation and parking; may exclude flight ticket prices.",
        parent=parent,
        critical=True
    )
    cs = plan.cost_summary or CostSummary()

    # Include ground transportation
    leaf_gt_inc = evaluator.add_leaf(
        id="Cost_Summary_Includes_Ground_Transport",
        desc="Includes ground transportation cost in the cost breakdown.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The cost summary includes ground transportation cost.",
        node=leaf_gt_inc,
        additional_instruction="Check the answer's cost summary section to see if a ground transportation cost line item is included."
    )

    # Include parking
    leaf_pk_inc = evaluator.add_leaf(
        id="Cost_Summary_Includes_Parking",
        desc="Includes airport parking cost in the cost breakdown.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The cost summary includes airport parking cost.",
        node=leaf_pk_inc,
        additional_instruction="Check the answer's cost summary section to see if a parking cost line item is included."
    )

    # Total math check
    leaf_total = evaluator.add_leaf(
        id="Cost_Summary_Total_Math",
        desc="Provides a total and arithmetic matches included components (excluding flight tickets if omitted).",
        parent=node,
        critical=True
    )

    # Construct a math consistency claim with the numbers
    gt_cost = cs.ground_transport_cost or ""
    pk_cost = cs.parking_cost or ""
    bag_cost = cs.baggage_carry_on_cost or ""
    total_cost = cs.total_estimated_cost or ""

    parts = []
    if cs.includes_ground_transport and _nonempty_str(gt_cost):
        parts.append(gt_cost)
    if cs.includes_parking and _nonempty_str(pk_cost):
        parts.append(pk_cost)
    if _nonempty_str(bag_cost):
        parts.append(bag_cost)

    if parts:
        parts_str = " + ".join(parts)
        claim_math = f"The sum of included components ({parts_str}) equals the stated total {total_cost} (allow small rounding differences)."
    else:
        claim_math = f"The cost summary total {total_cost} equals the sum of the included components listed in the answer (allow small rounding differences)."

    await evaluator.verify(
        claim=claim_math,
        node=leaf_total,
        additional_instruction="Check arithmetic consistency. Allow small rounding differences (e.g., within $1). Focus on components explicitly included."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the family Disney World trip planning task (transportation plan).
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level criteria evaluated independently but critical gating applies
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

    # Extract structured plan data from the answer
    plan: PlanExtraction = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction"
    )

    # Build verification tree according to rubric
    # 1) Source & Year Grounding (critical)
    await verify_source_and_year_grounding(evaluator, root, plan)

    # 2) Flight Information (critical)
    await verify_flight_information(evaluator, root, plan)

    # 3) Seating Requirements (critical)
    await verify_seating_requirements(evaluator, root, plan)

    # 4) Airport-to-Disney Distance (critical)
    await verify_airport_to_disney_distance(evaluator, root, plan)

    # 5) Baggage Allowances & Fees (critical)
    await verify_baggage_allowances_and_fees(evaluator, root, plan)

    # 6) Ground Transportation (critical)
    await verify_ground_transportation(evaluator, root, plan)

    # 7) Airport Parking (critical)
    await verify_airport_parking(evaluator, root, plan)

    # 8) Child Travel Policy Compliance (critical)
    await verify_child_travel_policy(evaluator, root, plan)

    # 9) Cost Summary (critical)
    await verify_cost_summary(evaluator, root, plan)

    # Return evaluation summary
    return evaluator.get_summary()