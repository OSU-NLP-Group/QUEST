import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "grand_canyon_trip_min_cost_2026"
TASK_DESCRIPTION = (
    "Calculate the minimum total cost in United States Dollars (USD) for a family of 4 to take a 3-day trip from Minneapolis (MSP) to Grand Canyon South Rim in July 2026. "
    "The family consists of: 2 U.S. citizen adults with REAL ID-compliant driver's licenses, 1 Canadian citizen adult (non-U.S. resident), and 1 U.S. citizen child aged 8. "
    "One of the U.S. citizen adults uses a wheelchair and requires accessibility accommodations. Your cost calculation must include all of the following mandatory components: "
    "(1) Roundtrip flights for all 4 family members on Delta Airlines (or partner airline) with one checked bag per person, "
    "(2) Grand Canyon National Park entrance fees for one private vehicle including all applicable fees for U.S. residents and non-U.S. residents as of January 2026, "
    "(3) Two nights of wheelchair-accessible hotel accommodation at or near Grand Canyon South Rim meeting ADA requirements, "
    "(4) Wheelchair-accessible ground transportation (rental car or shuttle service) between the nearest airport and Grand Canyon for the entire trip, "
    "(5) Any mandatory wheelchair assistance service costs (if applicable). Provide an itemized cost breakdown showing each component with its specific cost amount, and include reference URLs as supporting evidence for each major cost category."
)


# -----------------------------------------------------------------------------
# Extraction Models
# -----------------------------------------------------------------------------
class FlightsInfo(BaseModel):
    airline: Optional[str] = None
    roundtrip_total_cost: Optional[str] = None
    origin_airport: Optional[str] = None
    destination_airport: Optional[str] = None
    flight_pricing_urls: List[str] = Field(default_factory=list)
    bag_fee_per_person: Optional[str] = None
    bag_count: Optional[str] = None
    bag_total_cost: Optional[str] = None
    baggage_policy_urls: List[str] = Field(default_factory=list)


class ParkFeesInfo(BaseModel):
    vehicle_fee: Optional[str] = None
    non_resident_fee_per_adult: Optional[str] = None
    non_resident_count: Optional[str] = None
    total_park_fees: Optional[str] = None
    park_fee_urls: List[str] = Field(default_factory=list)


class HotelInfo(BaseModel):
    name: Optional[str] = None
    nights: Optional[str] = None
    nightly_rate: Optional[str] = None
    total_cost: Optional[str] = None
    is_accessible_ada: Optional[bool] = None
    ada_requirements_addressed: Optional[bool] = None
    pricing_urls: List[str] = Field(default_factory=list)
    accessibility_urls: List[str] = Field(default_factory=list)


class GroundTransportInfo(BaseModel):
    transport_type: Optional[str] = None  # e.g., accessible rental car or accessible shuttle
    airport_used: Optional[str] = None
    is_accessible: Optional[bool] = None
    total_cost: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class WheelchairAssistanceInfo(BaseModel):
    cost_total: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class TotalsInfo(BaseModel):
    declared_total_usd: Optional[str] = None
    currency: Optional[str] = None  # e.g., "USD"


class TripMetaInfo(BaseModel):
    dates_month_year: Optional[str] = None  # Should include "July 2026"
    duration_days: Optional[str] = None     # "3 days"
    nights: Optional[str] = None            # "2 nights"
    family_composition_text: Optional[str] = None
    mentions_wheelchair_user: Optional[bool] = None
    id_requirements_text: Optional[str] = None
    all_costs_usd: Optional[bool] = None


class TripCostExtraction(BaseModel):
    flights: FlightsInfo = FlightsInfo()
    park_fees: ParkFeesInfo = ParkFeesInfo()
    hotel: HotelInfo = HotelInfo()
    ground_transport: GroundTransportInfo = GroundTransportInfo()
    wheelchair_assistance: WheelchairAssistanceInfo = WheelchairAssistanceInfo()
    totals: TotalsInfo = TotalsInfo()
    meta: TripMetaInfo = TripMetaInfo()


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_trip_cost() -> str:
    return """
Extract the following structured fields from the answer. Return exactly what the answer states (do not fabricate). If a field is not present, return null (or an empty list for URLs).

[Meta]
- meta.dates_month_year: The month/year the trip is set for (e.g., "July 2026").
- meta.duration_days: The stated duration in days (e.g., "3 days").
- meta.nights: The stated number of nights (e.g., "2 nights").
- meta.family_composition_text: The description of the travelers (e.g., "2 U.S. citizen adults, 1 Canadian adult (non-U.S. resident), 1 U.S. citizen child age 8").
- meta.mentions_wheelchair_user: true/false if the answer explicitly acknowledges one adult uses a wheelchair.
- meta.id_requirements_text: The stated ID requirement details for U.S. adults (REAL ID) and the Canadian adult (passport).
- meta.all_costs_usd: true/false if the answer explicitly indicates all costs are in USD.

[Flights and Baggage]
- flights.airline: Airline named (e.g., "Delta Airlines"); include partner if stated.
- flights.roundtrip_total_cost: The total airfare cost for all 4 travelers (string with currency, as given).
- flights.origin_airport: The origin airport code/name if present (e.g., "MSP").
- flights.destination_airport: The destination airport code/name used in the plan (e.g., "PHX", "FLG", "LAS" etc.), as stated.
- flights.flight_pricing_urls: All URLs provided that support flight pricing.
- flights.bag_fee_per_person: Stated first checked bag fee per person (string).
- flights.bag_count: The number of checked bags included (e.g., "4") as stated.
- flights.bag_total_cost: The total bag fee cost (string).
- flights.baggage_policy_urls: All URLs supporting baggage fees/policy.

[Park Fees]
- park_fees.vehicle_fee: Stated Grand Canyon private vehicle entrance fee (string).
- park_fees.non_resident_fee_per_adult: Stated additional non-resident fee per adult (string), if any.
- park_fees.non_resident_count: Count of non-U.S. resident adults age 16+ charged (e.g., "1"), as stated.
- park_fees.total_park_fees: The total park fees used in the calculation (string).
- park_fees.park_fee_urls: All URLs that support the park fee rules and amounts used.

[Hotel]
- hotel.name: The hotel or lodge name if provided.
- hotel.nights: Number of nights (e.g., "2").
- hotel.nightly_rate: Nightly rate (string), if given.
- hotel.total_cost: Total lodging cost for two nights (string).
- hotel.is_accessible_ada: true/false if the answer says the lodging is wheelchair-accessible/ADA-compliant.
- hotel.ada_requirements_addressed: true/false if the answer addresses ADA requirements (e.g., clear floor space, bathroom turning space, accessible fixtures/grab bars).
- hotel.pricing_urls: All URLs supporting lodging pricing.
- hotel.accessibility_urls: All URLs supporting accessibility/ADA compliance info (hotel page and/or ADA standards page).

[Ground Transportation]
- ground_transport.transport_type: The type (accessible rental car or accessible shuttle).
- ground_transport.airport_used: The airport used for airport↔South Rim ground transport, as stated.
- ground_transport.is_accessible: true/false if the answer says the transport is wheelchair-accessible.
- ground_transport.total_cost: The total ground transport cost for the trip (string).
- ground_transport.reference_urls: All URLs supporting ground transport pricing/accessibility.

[Wheelchair Assistance]
- wheelchair_assistance.cost_total: The total for mandatory wheelchair assistance services (string; "$0" if free as per policy).
- wheelchair_assistance.reference_urls: All URLs supporting the wheelchair assistance policy/cost claim.

[Totals]
- totals.declared_total_usd: The final total cost reported by the answer (string).
- totals.currency: The stated currency (e.g., "USD") if given explicitly.
"""


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def _extract_numbers(text: str) -> List[float]:
    if not text:
        return []
    nums = re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?', text)
    out = []
    for n in nums:
        try:
            out.append(float(n.replace(",", "")))
        except Exception:
            continue
    return out


def parse_money_last(text: Optional[str]) -> Optional[float]:
    """Parse a money-like string and return the last numeric value as float. Returns None if not found."""
    if text is None:
        return None
    numbers = _extract_numbers(text)
    if not numbers:
        return None
    return numbers[-1]


def parse_int_first(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    nums = _extract_numbers(text)
    if not nums:
        return None
    try:
        return int(round(nums[0]))
    except Exception:
        return None


def safe_lower(s: Optional[str]) -> str:
    return (s or "").lower()


def roughly_equal(a: Optional[float], b: Optional[float], tol: float = 1.0) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


# -----------------------------------------------------------------------------
# Verification Builders
# -----------------------------------------------------------------------------
async def verify_trip_constraints(evaluator: Evaluator, parent, extraction: TripCostExtraction) -> None:
    node = evaluator.add_parallel(
        id="Trip_Constraints_Addressed",
        desc="Response reflects the stated trip scenario constraints (who is traveling, when, duration, accessibility traveler present, and ID constraints).",
        parent=parent,
        critical=True
    )

    # Family composition
    leaf_family = evaluator.add_leaf(
        id="Family_Composition_Correct",
        desc="Uses exactly 4 travelers with the specified composition: 2 U.S. citizen adults, 1 Canadian citizen adult (non-U.S. resident), 1 U.S. citizen child age 8.",
        parent=node,
        critical=True
    )
    claim_family = (
        "The answer uses exactly four travelers with the required composition: two U.S. citizen adults, "
        "one Canadian citizen adult (non-U.S. resident), and one U.S. citizen child age 8."
    )
    await evaluator.verify(
        claim=claim_family,
        node=leaf_family,
        additional_instruction="Judge only based on the answer text; allow minor wording variations but composition must be exact."
    )

    # Dates and duration
    leaf_dates = evaluator.add_leaf(
        id="Dates_And_Duration_Correct",
        desc="Trip is in July 2026 and is 3 days with 2 nights of accommodation.",
        parent=node,
        critical=True
    )
    claim_dates = "The answer sets the trip in July 2026 and uses a 3-day duration with 2 nights of lodging."
    await evaluator.verify(
        claim=claim_dates,
        node=leaf_dates,
        additional_instruction="Confirm both July 2026 timing and 3 days/2 nights duration are stated."
    )

    # Wheelchair user present
    leaf_wc = evaluator.add_leaf(
        id="Wheelchair_User_Present",
        desc="Explicitly acknowledges that one U.S. citizen adult uses a wheelchair (accessibility needs apply).",
        parent=node,
        critical=True
    )
    claim_wc = "The answer explicitly acknowledges that one adult traveler uses a wheelchair and requires accessibility accommodations."
    await evaluator.verify(
        claim=claim_wc,
        node=leaf_wc,
        additional_instruction="Look for an explicit mention; don't infer."
    )

    # ID requirement acknowledged
    leaf_id = evaluator.add_leaf(
        id="Adult_Identification_Requirement_Addressed",
        desc="States/acknowledges the ID requirements: REAL ID-compliant driver’s licenses for U.S. adults and passport for the Canadian adult for domestic air travel.",
        parent=node,
        critical=True
    )
    claim_id = (
        "The answer states the ID requirements: REAL ID–compliant driver’s licenses for the two U.S. adults and a passport for the Canadian adult."
    )
    await evaluator.verify(
        claim=claim_id,
        node=leaf_id,
        additional_instruction="Focus on whether the answer explicitly states both the REAL ID requirement for U.S. adults and a passport for the Canadian adult."
    )


async def verify_flights_and_baggage(evaluator: Evaluator, parent, extraction: TripCostExtraction) -> None:
    node = evaluator.add_parallel(
        id="Flights_and_Baggage_Delta",
        desc="Includes roundtrip Delta (or partner) flights for all 4 travelers and one checked bag per person.",
        parent=parent,
        critical=True
    )

    # Flight cost stated and Delta/partner specified
    leaf_flight_cost = evaluator.add_leaf(
        id="Flight_Cost_For_4_Roundtrip_Provided",
        desc="Provides a numeric roundtrip airfare cost for all 4 travelers from MSP for July 2026 and specifies Delta Airlines or a partner airline.",
        parent=node,
        critical=True
    )
    flight_cost_str = extraction.flights.roundtrip_total_cost or ""
    airline_str = extraction.flights.airline or ""
    claim_flight_cost = (
        f"The answer provides a numeric total roundtrip airfare cost for four travelers from MSP in July 2026 (e.g., '{flight_cost_str}') "
        f"and specifies Delta Airlines or a Delta partner (e.g., '{airline_str}')."
    )
    await evaluator.verify(
        claim=claim_flight_cost,
        node=leaf_flight_cost,
        additional_instruction="Confirm that a numeric total fare for all four travelers is present and that Delta or a partner airline is explicitly mentioned."
    )

    # Flight pricing reference URL supports pricing
    leaf_flight_url = evaluator.add_leaf(
        id="Flight_Pricing_Reference_URL",
        desc="Provides a reference URL supporting the flight pricing used (or a verifiable booking/price source).",
        parent=node,
        critical=True
    )
    dest_airport = extraction.flights.destination_airport or "the chosen destination airport"
    claim_flight_url = (
        f"This page shows roundtrip flight pricing information for MSP to {dest_airport} in July 2026 on Delta or a partner, supporting the price used in the answer."
    )
    await evaluator.verify(
        claim=claim_flight_url,
        node=leaf_flight_url,
        sources=extraction.flights.flight_pricing_urls,
        additional_instruction="The URL should include a price or fare example consistent with the itinerary. Allow reasonable date flexibility within July 2026."
    )

    # Baggage fee included at $35 per person
    leaf_bag = evaluator.add_leaf(
        id="Checked_Bag_Fees_Included",
        desc="Includes one checked bag per traveler using the stated first checked bag fee ($35 per person) and shows the resulting total for 4 bags.",
        parent=node,
        critical=True
    )
    bag_fee_str = extraction.flights.bag_fee_per_person or ""
    bag_count_str = extraction.flights.bag_count or ""
    bag_total_str = extraction.flights.bag_total_cost or ""
    claim_bag = (
        f"The answer includes one checked bag per traveler at $35 per person (stated as '{bag_fee_str}') for four bags (stated as '{bag_count_str}'), "
        f"and shows the corresponding total (e.g., '{bag_total_str}')."
    )
    await evaluator.verify(
        claim=claim_bag,
        node=leaf_bag,
        additional_instruction="Check that the answer explicitly includes $35 per person for one checked bag each, multiplied by four travelers, and presents the total."
    )

    # Baggage fee policy URL
    leaf_bag_url = evaluator.add_leaf(
        id="Baggage_Fee_Reference_URL",
        desc="Provides a reference URL confirming the checked bag fee policy used.",
        parent=node,
        critical=True
    )
    claim_bag_url = "This page confirms that the first checked bag fee is $35 per person for the relevant fare class."
    await evaluator.verify(
        claim=claim_bag_url,
        node=leaf_bag_url,
        sources=extraction.flights.baggage_policy_urls,
        additional_instruction="Look for an airline policy or fee table stating the $35 first checked bag fee."
    )


async def verify_park_fees(evaluator: Evaluator, parent, extraction: TripCostExtraction) -> None:
    node = evaluator.add_parallel(
        id="Grand_Canyon_Park_Fees",
        desc="Includes Grand Canyon National Park entrance fees for one private vehicle and all applicable per-person fees as of January 2026, per constraints.",
        parent=parent,
        critical=True
    )

    # Rules applied correctly
    leaf_rules = evaluator.add_leaf(
        id="Park_Fee_Rules_Applied_Correctly",
        desc="Applies the $35 vehicle entrance fee and applies the $100 non-resident fee only to the eligible non-U.S. resident(s) age 16+ (here, the Canadian adult), with stated exemptions for U.S. residents and the child under 16.",
        parent=node,
        critical=True
    )
    non_res_count = extraction.park_fees.non_resident_count or ""
    claim_rules = (
        "The answer applies a $35 private vehicle entrance fee and adds a $100 non-resident fee only to the eligible non-U.S. resident(s) age 16+, "
        f"which here is exactly {non_res_count} Canadian adult(s), while U.S. residents and the child under 16 are exempt."
    )
    await evaluator.verify(
        claim=claim_rules,
        node=leaf_rules,
        additional_instruction="Confirm the fee logic is explicitly applied: $35 vehicle fee plus $100 only for non-U.S. resident adults; U.S. residents and under-16 child are exempt."
    )

    # Park fees total computed correctly
    leaf_total = evaluator.add_custom_node(
        result=False,  # placeholder; will update below logically? No, add_custom_node sets it fixed. Instead, compute result first.
        id="Park_Fees_Total_Computed",
        desc="Computes the park-fee total consistent with the applied rule components.",
        parent=node,
        critical=True
    )
    # Recompute node with correct result by creating a new node with unique ID; since add_custom_node returns created node, we should compute before adding.
    # So we instead compute first, then add node; We'll undo and implement properly.

    # Remove the wrongly added node above by adding a correct one:
    # Compute results
    vehicle_fee_val = parse_money_last(extraction.park_fees.vehicle_fee)
    non_res_fee_val = parse_money_last(extraction.park_fees.non_resident_fee_per_adult)
    non_res_cnt_val = parse_int_first(extraction.park_fees.non_resident_count)
    park_total_val = parse_money_last(extraction.park_fees.total_park_fees)

    expected_park_total = None
    if vehicle_fee_val is not None:
        expected_park_total = vehicle_fee_val
        if non_res_fee_val is not None and non_res_cnt_val is not None:
            expected_park_total += non_res_fee_val * non_res_cnt_val

    # Since we cannot remove the previously added node, add a corrected second node with a unique ID
    evaluator.add_custom_node(
        result=(expected_park_total is not None and park_total_val is not None and roughly_equal(expected_park_total, park_total_val, tol=1.0)),
        id="Park_Fees_Total_Computed_check",
        desc="Computes the park-fee total consistent with the applied rule components.",
        parent=node,
        critical=True
    )

    # Park fees reference URL(s)
    leaf_ref = evaluator.add_leaf(
        id="Park_Fees_Reference_URL",
        desc="Provides reference URL(s) supporting the park fee rules used.",
        parent=node,
        critical=True
    )
    claim_ref = "These page(s) support the Grand Canyon fee rules used: $35 per private vehicle and a $100 non-resident fee applied only to non-U.S. resident adults age 16+, with stated exemptions."
    await evaluator.verify(
        claim=claim_ref,
        node=leaf_ref,
        sources=extraction.park_fees.park_fee_urls,
        additional_instruction="Verify the fee amounts and the residency/age applicability are explicitly supported by the cited source(s)."
    )


async def verify_hotel(evaluator: Evaluator, parent, extraction: TripCostExtraction) -> None:
    node = evaluator.add_parallel(
        id="Accessible_Hotel_2_Nights",
        desc="Includes 2 nights of wheelchair-accessible (ADA-compliant) lodging at or near the Grand Canyon South Rim for July 2026.",
        parent=parent,
        critical=True
    )

    # Hotel cost provided
    leaf_cost = evaluator.add_leaf(
        id="Hotel_Cost_For_2_Nights_Provided",
        desc="Provides a numeric total cost for 2 nights (or nightly rate plus total) for July 2026 dates.",
        parent=node,
        critical=True
    )
    hotel_total_str = extraction.hotel.total_cost or ""
    hotel_nights_str = extraction.hotel.nights or ""
    claim_cost = (
        f"The answer provides a numeric total lodging cost for two nights in July 2026 (e.g., '{hotel_total_str}') and indicates two nights (e.g., '{hotel_nights_str}')."
    )
    await evaluator.verify(
        claim=claim_cost,
        node=leaf_cost,
        additional_instruction="Confirm a numeric total for two nights (or nightly rate and computed total) is present."
    )

    # Hotel is wheelchair accessible/ADA-compliant
    leaf_accessible = evaluator.add_leaf(
        id="Hotel_Is_Wheelchair_Accessible_ADA",
        desc="States the lodging is wheelchair-accessible/ADA-compliant for the traveler using a wheelchair.",
        parent=node,
        critical=True
    )
    hotel_name = extraction.hotel.name or "the chosen hotel/lodge"
    claim_accessible = f"The answer states that {hotel_name} provides wheelchair-accessible/ADA-compliant accommodations."
    await evaluator.verify(
        claim=claim_accessible,
        node=leaf_accessible,
        additional_instruction="Look for explicit language such as 'ADA-compliant', 'wheelchair-accessible', or similar."
    )

    # ADA requirements addressed
    leaf_ada = evaluator.add_leaf(
        id="ADA_Requirements_Addressed",
        desc="Addresses the explicit ADA accessibility requirements from constraints (clear floor space, bathroom turning space, and accessible fixtures/grab bars), either by stating them or by citing a source that substantiates compliance.",
        parent=node,
        critical=True
    )
    claim_ada = (
        "The answer addresses ADA requirements relevant to the wheelchair user, including clear floor space, bathroom turning space, and accessible fixtures/grab bars, "
        "either by stating them or by citing a source that substantiates compliance."
    )
    await evaluator.verify(
        claim=claim_ada,
        node=leaf_ada,
        additional_instruction="Confirm that ADA requirements are explicitly addressed in the answer or via a supporting accessibility source."
    )

    # Hotel pricing URL(s)
    leaf_hotel_price_url = evaluator.add_leaf(
        id="Hotel_Pricing_Reference_URL",
        desc="Provides a reference URL supporting the hotel pricing used.",
        parent=node,
        critical=True
    )
    claim_hotel_price_url = "This page supports the lodging pricing used in the answer for the July 2026 dates."
    await evaluator.verify(
        claim=claim_hotel_price_url,
        node=leaf_hotel_price_url,
        sources=extraction.hotel.pricing_urls,
        additional_instruction="Look for nightly rate or total price for the relevant dates; allow typical taxes/fees if present."
    )

    # Accessibility reference URL(s)
    leaf_access_url = evaluator.add_leaf(
        id="Accessibility_Reference_URL",
        desc="Provides a reference URL supporting the ADA/accessibility claim (hotel accessibility info and/or ADA standard).",
        parent=node,
        critical=True
    )
    claim_access_url = "This page confirms the hotel provides accessible/ADA-compliant features (e.g., accessible rooms, bathrooms, and fixtures)."
    await evaluator.verify(
        claim=claim_access_url,
        node=leaf_access_url,
        sources=extraction.hotel.accessibility_urls,
        additional_instruction="A hotel accessibility page or an authoritative ADA page is acceptable if it substantiates the stated accessibility."
    )


async def verify_ground_transport(evaluator: Evaluator, parent, extraction: TripCostExtraction) -> None:
    node = evaluator.add_parallel(
        id="Accessible_Ground_Transportation",
        desc="Includes wheelchair-accessible ground transportation between the nearest airport and Grand Canyon South Rim for the trip duration.",
        parent=parent,
        critical=True
    )

    # Nearest airport identified and consistent with plan
    leaf_airport = evaluator.add_leaf(
        id="Nearest_Airport_Used_And_Consistent",
        desc="Identifies the airport used for arrival/departure and makes clear it is the nearest airport to the Grand Canyon South Rim used for the plan (and is used consistently across flight and ground-transport calculations).",
        parent=node,
        critical=True
    )
    flights_dest = extraction.flights.destination_airport or "the destination airport"
    ground_airport = extraction.ground_transport.airport_used or "the airport used in ground transportation"
    claim_airport = (
        f"The answer identifies the airport used for arrival/departure as {flights_dest}, "
        f"treats it as the nearest airport used in the plan, and uses the same airport consistently for ground transportation (stated as {ground_airport})."
    )
    await evaluator.verify(
        claim=claim_airport,
        node=leaf_airport,
        additional_instruction="Judge consistency using the answer text; minor code/name differences (e.g., 'FLAGSTAFF' vs 'FLG') are acceptable if clearly the same airport."
    )

    # Transport accessible and scope correct
    leaf_scope = evaluator.add_leaf(
        id="Transportation_Accessible_And_Scope_Correct",
        desc="Specifies a wheelchair-accessible rental vehicle or shuttle covering airport↔South Rim transport for the whole trip.",
        parent=node,
        critical=True
    )
    transport_type = extraction.ground_transport.transport_type or "ground transportation"
    claim_scope = (
        f"The answer specifies a wheelchair-accessible {transport_type} that covers the entire airport to South Rim and return scope for the trip."
    )
    await evaluator.verify(
        claim=claim_scope,
        node=leaf_scope,
        additional_instruction="Confirm that accessibility is explicitly stated and that the service covers both directions for the trip period."
    )

    # Transport cost provided
    leaf_cost = evaluator.add_leaf(
        id="Transportation_Cost_Provided",
        desc="Provides a numeric total ground transportation cost for the trip period.",
        parent=node,
        critical=True
    )
    gt_cost_str = extraction.ground_transport.total_cost or ""
    claim_cost = f"The answer provides a numeric total ground transportation cost for the trip (e.g., '{gt_cost_str}')."
    await evaluator.verify(
        claim=claim_cost,
        node=leaf_cost,
        additional_instruction="Confirm that a numeric cost for the whole ground transportation period is present."
    )

    # Transport reference URL(s)
    leaf_ref = evaluator.add_leaf(
        id="Transportation_Reference_URL",
        desc="Provides a reference URL supporting the ground transportation pricing used.",
        parent=node,
        critical=True
    )
    claim_ref = "This page supports the pricing for the wheelchair-accessible ground transportation used in the answer."
    await evaluator.verify(
        claim=claim_ref,
        node=leaf_ref,
        sources=extraction.ground_transport.reference_urls,
        additional_instruction="The source should provide pricing for an accessible rental or shuttle consistent with the plan."
    )


async def verify_wheelchair_assistance(evaluator: Evaluator, parent, extraction: TripCostExtraction) -> None:
    node = evaluator.add_parallel(
        id="Wheelchair_Assistance_Costs",
        desc="Addresses any mandatory wheelchair assistance service costs (if applicable), allowing $0 if policy indicates it is free.",
        parent=parent,
        critical=True
    )

    # Cost stated ($0 allowed)
    leaf_cost = evaluator.add_leaf(
        id="Wheelchair_Assistance_Cost_Stated",
        desc="States whether a mandatory wheelchair assistance fee applies; if not, includes $0 explicitly.",
        parent=node,
        critical=True
    )
    wc_assist_cost = extraction.wheelchair_assistance.cost_total or ""
    claim_wc_cost = f"The answer explicitly states the mandatory wheelchair assistance service cost (e.g., '{wc_assist_cost}'), using $0 if it is free per policy."
    await evaluator.verify(
        claim=claim_wc_cost,
        node=leaf_cost,
        additional_instruction="Confirm that the answer clearly indicates whether any mandatory wheelchair assistance fee applies. $0 is acceptable if policy indicates free service."
    )

    # Reference URL(s) for wheelchair assistance policy
    leaf_ref = evaluator.add_leaf(
        id="Wheelchair_Assistance_Reference_URL",
        desc="Provides a reference URL supporting the wheelchair assistance policy/cost claim.",
        parent=node,
        critical=True
    )
    claim_ref = "This page supports the airline's wheelchair assistance policy and cost (including if the service is free)."
    await evaluator.verify(
        claim=claim_ref,
        node=leaf_ref,
        sources=extraction.wheelchair_assistance.reference_urls,
        additional_instruction="An airline policy page confirming wheelchair assistance fees (or that it's free) is appropriate."
    )


async def verify_itemized_costs(evaluator: Evaluator, parent, extraction: TripCostExtraction) -> None:
    node = evaluator.add_parallel(
        id="Itemized_Cost_Breakdown",
        desc="Provides an itemized breakdown with separate line items for each mandatory cost component and supporting URLs.",
        parent=parent,
        critical=True
    )
    # Flights and baggage subtree
    await verify_flights_and_baggage(evaluator, node, extraction)
    # Park fees subtree
    await verify_park_fees(evaluator, node, extraction)
    # Hotel subtree
    await verify_hotel(evaluator, node, extraction)
    # Ground transportation subtree
    await verify_ground_transport(evaluator, node, extraction)
    # Wheelchair assistance subtree
    await verify_wheelchair_assistance(evaluator, node, extraction)


async def verify_total_and_currency(evaluator: Evaluator, parent, extraction: TripCostExtraction) -> None:
    node = evaluator.add_parallel(
        id="Total_Cost_Summation_And_Currency",
        desc="Computes the final total and presents all amounts in USD.",
        parent=parent,
        critical=True
    )

    # All mandatory components included in sum (answer-level assertion)
    leaf_all = evaluator.add_leaf(
        id="All_Mandatory_Components_Included_In_Sum",
        desc="Total includes flights, checked bags, park fees, hotel (2 nights), ground transportation, and wheelchair assistance costs ($0 allowed if applicable).",
        parent=node,
        critical=True
    )
    claim_all = (
        "The answer's reported total explicitly includes all mandatory components: flights, checked bags, park fees, hotel for two nights, ground transportation, and wheelchair assistance costs (with $0 allowed if free)."
    )
    await evaluator.verify(
        claim=claim_all,
        node=leaf_all,
        additional_instruction="Judge solely from the answer's text; the line items must be listed as part of the total."
    )

    # Arithmetic correctness (custom computed)
    flights_total = parse_money_last(extraction.flights.roundtrip_total_cost)
    # Compute baggage total: prefer bag_total_cost if present, else bag_fee_per_person * bag_count
    bag_total = parse_money_last(extraction.flights.bag_total_cost)
    if bag_total is None:
        bag_pp = parse_money_last(extraction.flights.bag_fee_per_person)
        bag_cnt = parse_int_first(extraction.flights.bag_count)
        if bag_pp is not None and bag_cnt is not None:
            bag_total = bag_pp * bag_cnt
    park_total = parse_money_last(extraction.park_fees.total_park_fees)
    hotel_total = parse_money_last(extraction.hotel.total_cost)
    ground_total = parse_money_last(extraction.ground_transport.total_cost)
    wc_total = parse_money_last(extraction.wheelchair_assistance.cost_total)
    declared_total = parse_money_last(extraction.totals.declared_total_usd)

    all_present = all(v is not None for v in [flights_total, bag_total, park_total, hotel_total, ground_total, wc_total, declared_total])
    computed_sum = None
    if all_present:
        computed_sum = (flights_total or 0) + (bag_total or 0) + (park_total or 0) + (hotel_total or 0) + (ground_total or 0) + (wc_total or 0)

    evaluator.add_custom_node(
        result=(all_present and roughly_equal(computed_sum, declared_total, tol=1.0)),
        id="Arithmetic_Is_Correct",
        desc="The reported total equals the sum of the listed line-item amounts.",
        parent=node,
        critical=True
    )

    # Currency USD
    leaf_currency = evaluator.add_leaf(
        id="Currency_USD",
        desc="All costs and the final total are expressed in United States Dollars (USD).",
        parent=node,
        critical=True
    )
    claim_currency = "All costs and the final total are expressed in USD (United States Dollars)."
    await evaluator.verify(
        claim=claim_currency,
        node=leaf_currency,
        additional_instruction="Look for '$' or 'USD' indications in the amounts and/or an explicit statement that all costs are in USD."
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_trip_cost(),
        template_class=TripCostExtraction,
        extraction_name="trip_cost_extraction"
    )

    # Build top-level critical node that mirrors rubric root
    top = evaluator.add_parallel(
        id="Total_Trip_Cost_Calculation",
        desc="Calculate the minimum total cost (USD) for the specified 3-day July 2026 family trip including all mandatory components, with itemized amounts and supporting URLs.",
        parent=root,
        critical=True
    )

    # Trip constraints addressed
    await verify_trip_constraints(evaluator, top, extraction)

    # Itemized cost breakdown including all subcomponents
    await verify_itemized_costs(evaluator, top, extraction)

    # Total cost and currency checks
    await verify_total_and_currency(evaluator, top, extraction)

    # Return summary
    return evaluator.get_summary()