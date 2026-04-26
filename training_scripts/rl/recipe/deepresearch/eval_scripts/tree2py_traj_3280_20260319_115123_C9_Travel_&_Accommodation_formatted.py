import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "complete_cruise_vacation_plan"
TASK_DESCRIPTION = """
A family of cruise enthusiasts from Houston, Texas is planning a Caribbean cruise vacation and needs help assembling a complete travel plan. They have the following specific requirements:

Cruise Requirements:
- Must be a 7-night Royal Caribbean cruise
- Must depart from Port Canaveral (Orlando), Florida
- The itinerary must include Grand Turk, Turks & Caicos as a port of call
- They want to participate in the formal nights (they know 7-night cruises have 2 formal nights)

Pre-Cruise Accommodation:
- Need a hotel near Port Canaveral for the night before embarkation
- Hotel must offer a park-and-cruise package with free parking for the cruise duration (7 days)
- Hotel must provide free shuttle service to the cruise terminal

Flight Arrangements:
- Must fly Frontier Airlines from Houston IAH to Orlando
- Personal item dimensions must be within Frontier's limits (14"H x 18"W x 8"D)
- Need to verify carry-on bag requirements if they choose to bring one (24"H x 16"W x 10"D, 35 lbs max)

Travel Program:
- They want to enroll in TSA PreCheck at Houston IAH before their trip
- Need to know the enrollment fee range for new applicants ($76.75-$85)
- Want confirmation that enrollment locations are available at IAH airport

Onboard Planning:
- Interested in purchasing Royal Caribbean's Deluxe Beverage Package
- Need to know the daily per-person price range (before gratuity: $56-$105)
- Want confirmation that 18% gratuity is added

Shore Excursions:
- Want to pre-book a shore excursion for Grand Turk through Royal Caribbean
- Interested in activities such as snorkeling, beach clubs, or island sightseeing

Provide a comprehensive cruise vacation plan that satisfies all of the above requirements, including specific details for each component with supporting reference URLs.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CruiseInfo(BaseModel):
    cruise_line: Optional[str] = None
    duration_nights: Optional[str] = None
    departure_port: Optional[str] = None
    includes_grand_turk_statement: Optional[str] = None
    formal_nights_statement: Optional[str] = None
    itinerary_urls: List[str] = Field(default_factory=list)
    formal_nights_urls: List[str] = Field(default_factory=list)


class HotelInfo(BaseModel):
    hotel_name: Optional[str] = None
    hotel_location: Optional[str] = None
    package_name: Optional[str] = None
    one_night_before_stay_statement: Optional[str] = None
    free_parking_duration_statement: Optional[str] = None
    free_shuttle_statement: Optional[str] = None
    accommodation_urls: List[str] = Field(default_factory=list)


class FlightInfo(BaseModel):
    airline: Optional[str] = None
    origin_airport: Optional[str] = None
    destination_airport: Optional[str] = None
    personal_item_dims: Optional[str] = None
    carry_on_dims: Optional[str] = None
    carry_on_weight_limit: Optional[str] = None
    policy_urls: List[str] = Field(default_factory=list)


class TSAInfo(BaseModel):
    enrollment_locations: List[str] = Field(default_factory=list)
    fee_range: Optional[str] = None
    membership_validity: Optional[str] = None
    tsa_urls: List[str] = Field(default_factory=list)


class BeverageInfo(BaseModel):
    package_type: Optional[str] = None
    daily_price_range: Optional[str] = None
    gratuity_percent: Optional[str] = None
    beverage_urls: List[str] = Field(default_factory=list)


class ExcursionInfo(BaseModel):
    excursion_name: Optional[str] = None
    destination: Optional[str] = None
    activity_types: List[str] = Field(default_factory=list)
    booking_urls: List[str] = Field(default_factory=list)


class PlanExtraction(BaseModel):
    cruise: Optional[CruiseInfo] = None
    accommodation: Optional[HotelInfo] = None
    flight: Optional[FlightInfo] = None
    tsa: Optional[TSAInfo] = None
    beverage: Optional[BeverageInfo] = None
    excursion: Optional[ExcursionInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
Extract the structured details for the requested comprehensive cruise vacation plan from the answer text. Only extract information explicitly stated in the answer and the URLs actually provided in the answer. Return null for any field not present.

For each section, extract the following fields:

1) cruise (object):
- cruise_line: The cruise line/operator name as written (e.g., "Royal Caribbean" or "Royal Caribbean International").
- duration_nights: The cruise length as written (e.g., "7-night", "7 nights").
- departure_port: The departure port as written (e.g., "Port Canaveral (Orlando), Florida").
- includes_grand_turk_statement: The text snippet stating or implying the itinerary includes "Grand Turk" or "Grand Turk, Turks & Caicos".
- formal_nights_statement: The text snippet stating that a 7-night cruise has exactly 2 formal nights and ideally noting the typical schedule (e.g., Day 2 and Day 6).
- itinerary_urls: All URLs provided that describe the selected cruise itinerary/details (may include the specific sailing, ship, or Royal Caribbean site pages).
- formal_nights_urls: All URLs provided that support the statement about formal nights (e.g., Royal Caribbean, official FAQs, or credible sources).

2) accommodation (object):
- hotel_name: The hotel name as written.
- hotel_location: The city/area near Port Canaveral as written (e.g., "Cape Canaveral", "Cocoa Beach").
- package_name: The package name as written (e.g., "Park and Cruise", "Snooze & Cruise").
- one_night_before_stay_statement: The text that indicates a 1-night pre-cruise stay is included/assumed.
- free_parking_duration_statement: The text that indicates free parking for about 7 days (the cruise duration).
- free_shuttle_statement: The text that indicates free shuttle service to the cruise terminal.
- accommodation_urls: All URLs provided that describe the hotel/package details.

3) flight (object):
- airline: Airline name as written (should be "Frontier Airlines").
- origin_airport: The origin airport as written (e.g., "Houston IAH").
- destination_airport: The destination airport as written (e.g., "Orlando", "MCO").
- personal_item_dims: The personal item size limit as written (e.g., '14" x 18" x 8"').
- carry_on_dims: The carry-on size limit as written (e.g., '24" x 16" x 10"').
- carry_on_weight_limit: The carry-on weight limit as written (e.g., "35 lbs").
- policy_urls: All URLs provided that support Frontier flight/baggage policy details (e.g., Frontier baggage policy pages).

4) tsa (object):
- enrollment_locations: A list of location labels as written that indicate TSA PreCheck enrollment availability at IAH (e.g., "Terminal A North", "Terminal E").
- fee_range: The fee range as written (e.g., "$76.75-$85").
- membership_validity: The membership validity period as written (e.g., "valid for 5 years").
- tsa_urls: All URLs provided that support TSA PreCheck enrollment/fees/locations (e.g., tsa.gov, IAH airport, IdentoGO).

5) beverage (object):
- package_type: The package type as written (e.g., "Deluxe Beverage Package").
- daily_price_range: The daily per-person price range before gratuity as written (e.g., "$56-$105").
- gratuity_percent: The gratuity/service charge percent as written (e.g., "18%").
- beverage_urls: All URLs provided that support beverage package pricing/details (prefer Royal Caribbean official sources).

6) excursion (object):
- excursion_name: The selected Grand Turk excursion name/title as written (if any).
- destination: The destination name as written (should indicate "Grand Turk").
- activity_types: A list of activity types mentioned (e.g., "snorkeling", "beach club", "island sightseeing").
- booking_urls: All URLs provided that show the excursion is pre-bookable via Royal Caribbean (e.g., Cruise Planner pages, Royal Caribbean shore excursions pages).

Output a single JSON object with keys: cruise, accommodation, flight, tsa, beverage, excursion, each mapping to the corresponding object as described above.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_cruise_selection(evaluator: Evaluator, parent_node, cruise: Optional[CruiseInfo]) -> None:
    node = evaluator.add_parallel(
        id="Cruise_Selection",
        desc="Select a Royal Caribbean cruise meeting duration, departure port, itinerary, and formal night requirements, with a supporting URL.",
        parent=parent_node,
        critical=True,
    )

    itinerary_urls = _safe_urls(cruise.itinerary_urls if cruise else [])
    formal_urls = _safe_urls(cruise.formal_nights_urls if cruise else [])

    # Cruise_Line_Requirement
    line_leaf = evaluator.add_leaf(
        id="Cruise_Line_Requirement",
        desc="The selected cruise must be operated by Royal Caribbean.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The selected cruise is operated by Royal Caribbean (Royal Caribbean International).",
        node=line_leaf,
        sources=itinerary_urls,
        additional_instruction="Confirm the operator on the itinerary/ship page is Royal Caribbean. Allow minor naming variants like 'Royal Caribbean International'.",
    )

    # Cruise_Duration_Requirement
    duration_leaf = evaluator.add_leaf(
        id="Cruise_Duration_Requirement",
        desc="The cruise must be exactly 7 nights in duration.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The cruise duration is exactly 7 nights (7-night or 7 nights).",
        node=duration_leaf,
        sources=itinerary_urls,
        additional_instruction="Check the itinerary page for the length. Accept phrasing like '7-night' or '7 nights'.",
    )

    # Departure_Port_Requirement
    depart_leaf = evaluator.add_leaf(
        id="Departure_Port_Requirement",
        desc="The cruise must depart from Port Canaveral (Orlando), Florida.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The cruise departs from Port Canaveral (Orlando), Florida.",
        node=depart_leaf,
        sources=itinerary_urls,
        additional_instruction="Accept 'Port Canaveral' or 'Orlando (Port Canaveral)' or 'Cape Canaveral' as the departure port.",
    )

    # Port_Of_Call_Requirement
    poc_leaf = evaluator.add_leaf(
        id="Port_Of_Call_Requirement",
        desc="The cruise itinerary must include Grand Turk, Turks & Caicos as a port of call.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The cruise itinerary includes Grand Turk, Turks and Caicos (often written as 'Grand Turk, Turks & Caicos').",
        node=poc_leaf,
        sources=itinerary_urls,
        additional_instruction="Look for 'Grand Turk' in the listed ports of call. Minor punctuation/spelling variants are acceptable.",
    )

    # Formal_Nights_Requirement
    formal_leaf = evaluator.add_leaf(
        id="Formal_Nights_Requirement",
        desc="The plan must state that the 7-night cruise has exactly 2 formal nights and note the typical schedule (Day 2 and Day 6) per the constraints.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="A 7-night Royal Caribbean cruise typically has exactly two formal nights, commonly on Day 2 and Day 6.",
        node=formal_leaf,
        sources=formal_urls if formal_urls else itinerary_urls,  # fallback if formal-specific URL not given
        additional_instruction="Verify that the provided source(s) explicitly state there are 2 formal nights on a 7-night cruise, and ideally mention the usual schedule Day 2 and Day 6 (allowing slight variations/wording).",
    )

    # Cruise_Selection_Reference (existence of at least one itinerary URL)
    evaluator.add_custom_node(
        result=len(itinerary_urls) > 0,
        id="Cruise_Selection_Reference",
        desc="Provide at least one supporting reference URL for the chosen cruise itinerary/details.",
        parent=node,
        critical=True,
    )


async def verify_accommodation(evaluator: Evaluator, parent_node, hotel: Optional[HotelInfo]) -> None:
    node = evaluator.add_parallel(
        id="Pre_Cruise_Accommodation",
        desc="Provide a hotel plan near Port Canaveral for the night before embarkation that includes park-and-cruise, free parking, and shuttle, with a supporting URL.",
        parent=parent_node,
        critical=True,
    )

    acc_urls = _safe_urls(hotel.accommodation_urls if hotel else [])

    # Hotel_Location_Requirement
    loc_leaf = evaluator.add_leaf(
        id="Hotel_Location_Requirement",
        desc="Hotel must be located near Port Canaveral.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The hotel is located near Port Canaveral (e.g., in Cape Canaveral or Cocoa Beach, within a short driving distance of the cruise terminals).",
        node=loc_leaf,
        sources=acc_urls,
        additional_instruction="Accept addresses in Cape Canaveral/Cocoa Beach or explicit mentions of proximity to Port Canaveral cruise port.",
    )

    # Park_And_Cruise_Package_Requirement
    pac_leaf = evaluator.add_leaf(
        id="Park_And_Cruise_Package_Requirement",
        desc="Hotel must offer a park-and-cruise package.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The hotel offers a 'Park and Cruise' (or similar) package specifically for cruise travelers.",
        node=pac_leaf,
        sources=acc_urls,
        additional_instruction="Accept synonyms like 'Park & Cruise', 'Snooze & Cruise', or pages that clearly bundle parking with a pre-cruise stay.",
    )

    # Night_Stay_Requirement
    night_leaf = evaluator.add_leaf(
        id="Night_Stay_Requirement",
        desc="Hotel package must include a 1-night stay before cruise departure.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The package includes a 1-night stay prior to the cruise embarkation.",
        node=night_leaf,
        sources=acc_urls,
        additional_instruction="Look for wording like 'one-night hotel stay included' or clear inclusion of a pre-cruise night.",
    )

    # Parking_Duration_Requirement
    park_leaf = evaluator.add_leaf(
        id="Parking_Duration_Requirement",
        desc="Hotel package must include free parking for the cruise duration (7 days minimum).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The package includes free parking for approximately 7 days (the duration of a typical 7-night cruise).",
        node=park_leaf,
        sources=acc_urls,
        additional_instruction="Pass if the page offers free cruise parking for at least 7 days or 'for the length of the cruise'.",
    )

    # Shuttle_Service_Requirement
    shuttle_leaf = evaluator.add_leaf(
        id="Shuttle_Service_Requirement",
        desc="Hotel must provide free shuttle service to the cruise terminal.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The hotel provides free shuttle service to the Port Canaveral cruise terminal.",
        node=shuttle_leaf,
        sources=acc_urls,
        additional_instruction="Look for explicit mention of 'free shuttle' to the cruise port; do not accept paid-only shuttles.",
    )

    # Accommodation_Reference
    evaluator.add_custom_node(
        result=len(acc_urls) > 0,
        id="Accommodation_Reference",
        desc="Provide at least one supporting reference URL for the hotel/park-and-cruise package details.",
        parent=node,
        critical=True,
    )


async def verify_flight(evaluator: Evaluator, parent_node, flight: Optional[FlightInfo]) -> None:
    node = evaluator.add_parallel(
        id="Flight_Arrangements",
        desc="Provide Frontier flight planning from Houston IAH to Orlando with personal-item (and carry-on) compliance and a supporting URL.",
        parent=parent_node,
        critical=True,
    )

    policy_urls = _safe_urls(flight.policy_urls if flight else [])

    # Airline_Selection_Requirement (simple check from the answer)
    airline_leaf = evaluator.add_leaf(
        id="Airline_Selection_Requirement",
        desc="Flight must be operated by Frontier Airlines.",
        parent=node,
        critical=True,
    )
    airline_val = (flight.airline if flight else "") or ""
    await evaluator.verify(
        claim=f"The planned airline is Frontier Airlines (answer states airline='{airline_val}').",
        node=airline_leaf,
        additional_instruction="Judge based on the answer text; accept 'Frontier' or 'Frontier Airlines' as matching.",
    )

    # Route_Requirement (simple check from the answer)
    route_leaf = evaluator.add_leaf(
        id="Route_Requirement",
        desc="Flight plan must be from Houston IAH to Orlando.",
        parent=node,
        critical=True,
    )
    origin_val = (flight.origin_airport if flight else "") or ""
    dest_val = (flight.destination_airport if flight else "") or ""
    await evaluator.verify(
        claim=f"The planned route is from Houston IAH to Orlando (answer indicates origin='{origin_val}', destination='{dest_val}').",
        node=route_leaf,
        additional_instruction="Judge based on the answer text; accept 'Orlando', 'MCO', or 'Orlando International' for destination.",
    )

    # Personal_Item_Compliance (verify against policy URLs)
    personal_leaf = evaluator.add_leaf(
        id="Personal_Item_Compliance",
        desc="Personal item dimensions must comply with Frontier's maximum: 14 inches height x 18 inches width x 8 inches depth.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Frontier Airlines' personal item maximum size is 14 inches (H) x 18 inches (W) x 8 inches (D).",
        node=personal_leaf,
        sources=policy_urls,
        additional_instruction="Verify on official Frontier pages (or equivalent authoritative sources). Accept minor formatting differences (e.g., 14\" x 18\" x 8\").",
    )

    # Carry_On_Compliance (verify against policy URLs)
    carry_leaf = evaluator.add_leaf(
        id="Carry_On_Compliance",
        desc="The plan must verify/state that if bringing a carry-on bag, it must comply with Frontier's maximum: 24 inches height x 16 inches width x 10 inches depth, and weigh no more than 35 pounds.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Frontier carry-on maximum size is 24 inches (H) x 16 inches (W) x 10 inches (D), and the carry-on weight limit is 35 pounds.",
        node=carry_leaf,
        sources=policy_urls,
        additional_instruction="Confirm both the size and 35 lb weight limit; allow reasonable unit variations (e.g., 'lbs').",
    )

    # Flight_Reference
    evaluator.add_custom_node(
        result=len(policy_urls) > 0,
        id="Flight_Reference",
        desc="Provide at least one supporting reference URL for Frontier flight/baggage policy information used in the plan.",
        parent=node,
        critical=True,
    )


async def verify_tsa(evaluator: Evaluator, parent_node, tsa: Optional[TSAInfo]) -> None:
    node = evaluator.add_parallel(
        id="TSA_PreCheck_Enrollment",
        desc="Provide TSA PreCheck enrollment plan at Houston IAH including fee range and availability, with supporting URL(s).",
        parent=parent_node,
        critical=True,
    )

    tsa_urls = _safe_urls(tsa.tsa_urls if tsa else [])

    # Enrollment_Location_Requirement
    enroll_leaf = evaluator.add_leaf(
        id="Enrollment_Location_Requirement",
        desc="Confirm TSA PreCheck enrollment is available at Houston IAH and specify that enrollment is available in Terminal A North and/or Terminal E (per constraints).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="TSA PreCheck enrollment is available at Houston IAH, specifically with an enrollment center in Terminal A North and/or Terminal E.",
        node=enroll_leaf,
        sources=tsa_urls,
        additional_instruction="Pass if any provided official source confirms a PreCheck enrollment center at IAH in either Terminal A North or Terminal E (naming variants acceptable).",
    )

    # Enrollment_Fee_Requirement
    fee_leaf = evaluator.add_leaf(
        id="Enrollment_Fee_Requirement",
        desc="State the enrollment fee range for new applicants ($76.75–$85).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The TSA PreCheck enrollment fee for new applicants falls within the range $76.75 to $85.",
        node=fee_leaf,
        sources=tsa_urls,
        additional_instruction="Accept official sources listing fees that fall anywhere within $76.75–$85 (providers may quote $78 or $85; treat these as within range).",
    )

    # Membership_Validity_Requirement
    validity_leaf = evaluator.add_leaf(
        id="Membership_Validity_Requirement",
        desc="State that TSA PreCheck membership is valid for 5 years.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="TSA PreCheck membership is valid for 5 years.",
        node=validity_leaf,
        sources=tsa_urls,
        additional_instruction="Verify on tsa.gov or equivalent official sources that membership validity is 5 years.",
    )

    # TSA_PreCheck_Reference
    evaluator.add_custom_node(
        result=len(tsa_urls) > 0,
        id="TSA_PreCheck_Reference",
        desc="Provide at least one supporting reference URL for TSA PreCheck enrollment details referenced in the plan.",
        parent=node,
        critical=True,
    )


async def verify_beverage(evaluator: Evaluator, parent_node, bev: Optional[BeverageInfo]) -> None:
    node = evaluator.add_parallel(
        id="Onboard_Beverage_Package",
        desc="Provide plan details for Royal Caribbean Deluxe Beverage Package including price range and gratuity, with supporting URL(s).",
        parent=parent_node,
        critical=True,
    )

    bev_urls = _safe_urls(bev.beverage_urls if bev else [])

    # Package_Type_Requirement
    pkg_leaf = evaluator.add_leaf(
        id="Package_Type_Requirement",
        desc="The beverage package must be Royal Caribbean's Deluxe Beverage Package.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The package in question is Royal Caribbean's 'Deluxe Beverage Package'.",
        node=pkg_leaf,
        sources=bev_urls,
        additional_instruction="Confirm the page refers specifically to Royal Caribbean's Deluxe Beverage Package (not other lines or packages).",
    )

    # Price_Range_Requirement
    price_leaf = evaluator.add_leaf(
        id="Price_Range_Requirement",
        desc="State the daily per-person price range (before gratuity: $56–$105).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The typical daily per-person price (before gratuity) for Royal Caribbean's Deluxe Beverage Package falls within $56 to $105.",
        node=price_leaf,
        sources=bev_urls,
        additional_instruction="RC pricing is dynamic; pass if the cited source(s) show a typical advertised/sale range within $56–$105.",
    )

    # Gratuity_Requirement
    grat_leaf = evaluator.add_leaf(
        id="Gratuity_Requirement",
        desc="Confirm that 18% gratuity is added to the package pricing.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="An 18% gratuity (service charge) is added to the Royal Caribbean Deluxe Beverage Package price.",
        node=grat_leaf,
        sources=bev_urls,
        additional_instruction="Look for explicit mention of 18% service charge or gratuity on the official RC page or equivalent authoritative sources.",
    )

    # Beverage_Package_Reference
    evaluator.add_custom_node(
        result=len(bev_urls) > 0,
        id="Beverage_Package_Reference",
        desc="Provide at least one supporting reference URL for the beverage package pricing/details.",
        parent=node,
        critical=True,
    )


async def verify_excursion(evaluator: Evaluator, parent_node, exc: Optional[ExcursionInfo]) -> None:
    node = evaluator.add_parallel(
        id="Shore_Excursion_Planning",
        desc="Provide a Grand Turk shore excursion plan that is pre-bookable via Royal Caribbean and matches desired activity types, with supporting URL(s).",
        parent=parent_node,
        critical=True,
    )

    exc_urls = _safe_urls(exc.booking_urls if exc else [])

    # Booking_Method_Requirement
    booking_leaf = evaluator.add_leaf(
        id="Booking_Method_Requirement",
        desc="Shore excursion must be available for pre-booking through Royal Caribbean (e.g., Cruise Planner).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The specified Grand Turk shore excursion can be pre-booked through Royal Caribbean (e.g., via Cruise Planner).",
        node=booking_leaf,
        sources=exc_urls,
        additional_instruction="Accept clear RC pages indicating 'Book now', 'Pre-reserve', or Cruise Planner access for the excursion.",
    )

    # Excursion_Type_Requirement
    type_leaf = evaluator.add_leaf(
        id="Excursion_Type_Requirement",
        desc="Excursion options must include at least one of: snorkeling, beach club access, or island sightseeing.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The selected Grand Turk excursion includes at least one of these activity types: snorkeling, beach club access, or island sightseeing.",
        node=type_leaf,
        sources=exc_urls,
        additional_instruction="Pass if any one of the three activity types is clearly supported by the cited RC excursion page(s).",
    )

    # Shore_Excursion_Reference
    evaluator.add_custom_node(
        result=len(exc_urls) > 0,
        id="Shore_Excursion_Reference",
        desc="Provide at least one supporting reference URL for Royal Caribbean Grand Turk shore excursions/pre-booking.",
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
    Evaluate an answer for the complete cruise vacation plan task.
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
        default_model=model,
    )

    # Add ground-truth style constraints for transparency in summary (not for grading directly)
    evaluator.add_ground_truth({
        "required_cruise": {
            "line": "Royal Caribbean",
            "duration_nights": "7",
            "departure_port": "Port Canaveral (Orlando), FL",
            "must_include": "Grand Turk, Turks & Caicos",
            "formal_nights": "2 (typically Day 2 & Day 6)"
        },
        "required_accommodation": {
            "location": "Near Port Canaveral",
            "package": "Park-and-cruise",
            "parking": "Free for duration (~7 days)",
            "shuttle": "Free to cruise terminal"
        },
        "flight": {
            "airline": "Frontier Airlines",
            "route": "IAH -> Orlando",
            "personal_item": '14" x 18" x 8"',
            "carry_on": '24" x 16" x 10", 35 lbs max'
        },
        "tsa_precheck": {
            "iah_locations": "Terminal A North and/or Terminal E",
            "fee_range": "$76.75-$85",
            "validity": "5 years"
        },
        "beverage": {
            "package": "Royal Caribbean Deluxe Beverage Package",
            "price_range_per_day_before_gratuity": "$56-$105",
            "gratuity": "18%"
        },
        "shore_excursion": {
            "destination": "Grand Turk",
            "activity_types_any_of": ["snorkeling", "beach club", "island sightseeing"],
            "booking": "Pre-book via Royal Caribbean"
        }
    })

    # Build the rubric-tree root (critical child under the framework's non-critical root)
    plan_root = evaluator.add_parallel(
        id="Complete_Cruise_Vacation_Plan",
        desc="A comprehensive cruise vacation plan from Houston meeting all specified requirements for cruise selection, accommodation, flights, TSA PreCheck, onboard beverage package, and shore excursion planning, with supporting reference URLs.",
        parent=root,
        critical=True,
    )

    # Extract plan data
    extracted = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction",
    )

    # Run verifications per section
    await verify_cruise_selection(evaluator, plan_root, extracted.cruise)
    await verify_accommodation(evaluator, plan_root, extracted.accommodation)
    await verify_flight(evaluator, plan_root, extracted.flight)
    await verify_tsa(evaluator, plan_root, extracted.tsa)
    await verify_beverage(evaluator, plan_root, extracted.beverage)
    await verify_excursion(evaluator, plan_root, extracted.excursion)

    # Return summary
    return evaluator.get_summary()