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
TASK_ID = "multi_city_orlando_cruise_accessibility"
TASK_DESCRIPTION = """You are planning a comprehensive multi-city travel itinerary for two groups that will meet in Orlando before embarking on a Caribbean cruise. The trip involves coordinating flights, hotel accommodations, and cruise arrangements with multiple accessibility and special needs requirements.

Nashville Group (3 travelers):
- Passenger 1: Requires wheelchair assistance and is traveling with a service animal
- Passenger 2: Travels with medical equipment requiring special handling
- Passenger 3: Is bringing a pet dog (at least 8 weeks old)

Bangor Group (2 travelers):
- Passenger 1: Requires priority boarding and accessible seating
- Passenger 2: Traveling with carry-on baggage only

Requirements:

1. Flights: Both groups must take direct/nonstop flights to Orlando. The Nashville group departs from Nashville International Airport (BNA), and the Bangor group departs from Bangor International Airport (BGR).

2. Orlando Hotel: Book a hotel in the Orlando area that provides:
   - At least two ADA-accessible rooms (each with minimum 32-inch door width, 36-inch wide route to bed, and accessible bathroom with grab bars and roll-in shower or accessible tub)
   - At least one pet-friendly room (accepting dogs, with the pet under typical weight limits)
   - A conference/meeting room that can accommodate the entire group (using the standard calculation of 20-25 square feet per person)
   - All conference spaces must be wheelchair accessible

3. Caribbean Cruise: Book a Carnival Cruise that:
   - Departs from Port Canaveral (Orlando area)
   - Includes Curaçao as a port of call
   - Provides at least one wheelchair-accessible stateroom
   - Accommodates special dietary requests (submitted at least 45 days in advance)
   - Has licensed medical staff (minimum 1 doctor and 2 nurses) and a medical center
   - All passengers must complete online check-in before midnight prior to sailing and arrive at least 60 minutes before departure

4. Curaçao Entry: All US citizen travelers must have:
   - Valid passports for the duration of stay
   - Completed Curaçao Digital Immigration Card (DI card) within 7 days prior to departure

5. Travel Insurance: Purchase comprehensive travel insurance that includes:
   - Medical evacuation coverage of at least $100,000
   - Trip cancellation coverage for non-refundable prepaid costs
   - Pre-existing condition waiver (requires purchasing insurance within 15 days of initial trip deposit and insuring 100% of trip costs)

Provide a detailed travel plan that addresses all these requirements, including:
- Specific airline/flight information for both departure cities
- Hotel name and confirmation that it meets all accessibility, pet, and meeting space requirements
- Cruise line confirmation, itinerary details, and accessibility features
- All required documentation for international travel
- Travel insurance policy details with appropriate coverage levels

Each component of your answer should include reference URLs from reliable sources that verify the information meets the stated requirements.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FlightInfo(BaseModel):
    airline: Optional[str] = None
    flight_number: Optional[str] = None
    depart_airport: Optional[str] = None
    arrival_airport: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class NashvilleFlightChecks(BaseModel):
    wheelchair_assistance_addressed: Optional[bool] = None
    service_animal_docs_addressed: Optional[bool] = None
    medical_equipment_handling_addressed: Optional[bool] = None
    medical_equipment_not_count_baggage_addressed: Optional[bool] = None
    pet_min_age_addressed: Optional[bool] = None


class BangorFlightChecks(BaseModel):
    priority_boarding_addressed: Optional[bool] = None
    accessible_seating_addressed: Optional[bool] = None
    carryon_only_addressed: Optional[bool] = None
    carryon_size_text: Optional[str] = None


class FlightsExtraction(BaseModel):
    nashville: Optional[FlightInfo] = None
    nashville_checks: Optional[NashvilleFlightChecks] = None
    bangor: Optional[FlightInfo] = None
    bangor_checks: Optional[BangorFlightChecks] = None
    flights_urls: List[str] = Field(default_factory=list)


class HotelExtraction(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    meeting_room_name: Optional[str] = None
    meeting_room_capacity_people: Optional[str] = None
    meeting_room_sqft: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CruiseExtraction(BaseModel):
    cruise_line: Optional[str] = None
    ship_name: Optional[str] = None
    itinerary_url: Optional[str] = None
    departure_port: Optional[str] = None
    accessible_stateroom_url: Optional[str] = None
    dietary_policy_url: Optional[str] = None
    medical_staff_policy_url: Optional[str] = None
    medical_center_policy_url: Optional[str] = None
    checkin_policy_url: Optional[str] = None
    arrival_time_policy_url: Optional[str] = None
    embarkation_docs_url: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CuracaoExtraction(BaseModel):
    passport_requirement_url: Optional[str] = None
    di_card_url: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class InsuranceExtraction(BaseModel):
    provider_name: Optional[str] = None
    policy_name: Optional[str] = None
    policy_urls: List[str] = Field(default_factory=list)
    med_evac_coverage_text: Optional[str] = None
    trip_cancellation_text: Optional[str] = None
    preexisting_waiver_text: Optional[str] = None


class TravelPlanExtraction(BaseModel):
    flights: Optional[FlightsExtraction] = None
    hotel: Optional[HotelExtraction] = None
    cruise: Optional[CruiseExtraction] = None
    curacao: Optional[CuracaoExtraction] = None
    insurance: Optional[InsuranceExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_travel_plan() -> str:
    return """
Extract a structured summary of the travel plan as presented in the answer. Only extract what is explicitly mentioned in the answer text and URLs that are actually included. Provide null when information is missing.

1) flights:
- nashville:
  - airline (string)
  - flight_number (string)
  - depart_airport (string, e.g., "BNA")
  - arrival_airport (string, e.g., "MCO" or "SFB")
  - urls (array of URLs specific to the Nashville flight, airline policy pages, booking pages, or route/schedule pages)
- nashville_checks:
  - wheelchair_assistance_addressed (boolean: does the answer state wheelchair assistance will be requested/arranged?)
  - service_animal_docs_addressed (boolean: does the answer mention airline service animal documentation requirements?)
  - medical_equipment_handling_addressed (boolean: does the answer mention special handling of medical equipment?)
  - medical_equipment_not_count_baggage_addressed (boolean: does the answer note assistive/medical devices do not count toward baggage?)
  - pet_min_age_addressed (boolean: does the answer confirm the pet dog is at least 8 weeks old and notes the airline's minimum age?)
- bangor:
  - airline (string)
  - flight_number (string)
  - depart_airport (string, e.g., "BGR")
  - arrival_airport (string, e.g., "MCO" or "SFB")
  - urls (array of URLs specific to the Bangor flight, airline policy pages, booking pages, or route/schedule pages)
- bangor_checks:
  - priority_boarding_addressed (boolean)
  - accessible_seating_addressed (boolean)
  - carryon_only_addressed (boolean)
  - carryon_size_text (string, if the answer mentions specific carry-on size limits like "22 x 14 x 9", otherwise null)
- flights_urls (array of additional URLs in the answer that support direct-flight availability and policies relevant to accessibility/service animals/medical equipment/pets)

2) hotel:
- name (string)
- location (string, as described in the answer; e.g., "Orlando, FL" or a nearby Orlando-area location like Lake Buena Vista/Kissimmee)
- meeting_room_name (string, if any specific room is cited)
- meeting_room_capacity_people (string, if the answer gives a person-capacity)
- meeting_room_sqft (string, if the answer gives a square-foot number)
- urls (array of URLs verifying the hotel’s ADA room features, pet policy, and meeting/conference spaces)

3) cruise:
- cruise_line (string; e.g., "Carnival Cruise Line")
- ship_name (string, if present)
- itinerary_url (string, URL to the itinerary/booking page that shows departure port and ports of call)
- departure_port (string; e.g., "Port Canaveral")
- accessible_stateroom_url (string, URL where accessible cabins are described)
- dietary_policy_url (string, URL for dietary requests policy)
- medical_staff_policy_url (string, URL for medical staff info)
- medical_center_policy_url (string, URL for onboard medical center info)
- checkin_policy_url (string, URL for online check-in policy)
- arrival_time_policy_url (string, URL for arrival time/boarding requirements)
- embarkation_docs_url (string, URL describing required embarkation documents)
- urls (array of any other cruise-related URLs included in the answer)

4) curacao:
- passport_requirement_url (string, official or authoritative URL about passport validity requirements)
- di_card_url (string, official or authoritative URL describing the Curaçao Digital Immigration Card timing)
- urls (array of any other Curaçao entry requirement URLs cited)

5) insurance:
- provider_name (string)
- policy_name (string)
- policy_urls (array of URL(s) to the policy or official plan page)
- med_evac_coverage_text (string as stated in the answer for med-evac coverage; e.g., "$100,000")
- trip_cancellation_text (string phrase indicating trip cancellation coverage for non-refundable prepaid costs, if stated)
- preexisting_waiver_text (string describing the pre-existing condition waiver terms; should include timing relative to initial deposit and insuring 100% of trip costs, if stated)
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_urls(urls: List[Optional[str]]) -> List[str]:
    clean = [u.strip() for u in urls if u and isinstance(u, str) and u.strip()]
    # Deduplicate preserving order
    seen = set()
    result = []
    for u in clean:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _extract_first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.findall(r"\d+", text.replace(",", ""))
    if not m:
        return None
    try:
        return int(m[0])
    except Exception:
        return None


def _contains_size_22_14_9(text: Optional[str]) -> bool:
    if not text:
        return True  # If not specified, consider consistent per rubric wording "If specified ..."
    t = text.lower().replace("inches", "in").replace("in.", "in")
    # Accept any order 22,14,9 present
    return ("22" in t and "14" in t and "9" in t)


def _group_size() -> int:
    # 3 Nashville + 2 Bangor
    return 5


def _required_sqft_range_for_group(n_people: int) -> (int, int):
    return 20 * n_people, 25 * n_people


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_flights_verification(evaluator: Evaluator, parent_node, extracted: TravelPlanExtraction):
    flights_node = evaluator.add_parallel(
        id="Flights_To_Orlando",
        desc="Flight plan for both groups to reach Orlando via direct/nonstop flights, addressing stated passenger needs and providing verifying URLs.",
        parent=parent_node,
        critical=True
    )

    flights: FlightsExtraction = extracted.flights or FlightsExtraction()
    nash: FlightInfo = flights.nashville or FlightInfo()
    bgr: FlightInfo = flights.bangor or FlightInfo()
    nash_checks: NashvilleFlightChecks = flights.nashville_checks or NashvilleFlightChecks()
    bgr_checks: BangorFlightChecks = flights.bangor_checks or BangorFlightChecks()

    global_flt_urls = flights.flights_urls or []
    nash_urls = _dedupe_urls((nash.urls or []) + global_flt_urls)
    bgr_urls = _dedupe_urls((bgr.urls or []) + global_flt_urls)

    # Nashville Flights (critical group)
    nashville_node = evaluator.add_parallel(
        id="Nashville_Flights",
        desc="Nashville group flight details and special needs coverage.",
        parent=flights_node,
        critical=True
    )

    # Specific flight info provided
    nash_specific = evaluator.add_custom_node(
        result=bool((nash.airline and nash.airline.strip()) or (nash.flight_number and nash.flight_number.strip())),
        id="Nashville_Specific_Flight_Info_Provided",
        desc="Answer includes specific airline/flight information for the Nashville (BNA) group.",
        parent=nashville_node,
        critical=True
    )

    # Direct flight from BNA to Orlando area
    nash_direct_leaf = evaluator.add_leaf(
        id="Nashville_Flight_Direct_From_BNA",
        desc="Nashville group takes a direct/nonstop flight departing from BNA and arriving to Orlando (Orlando-area airport).",
        parent=nashville_node,
        critical=True
    )
    nash_direct_claim = "There is a direct (nonstop) flight from Nashville International Airport (BNA) to an Orlando-area airport (e.g., Orlando International Airport [MCO] or Orlando Sanford International Airport [SFB]) that aligns with the proposed plan."
    await evaluator.verify(
        claim=nash_direct_claim,
        node=nash_direct_leaf,
        sources=nash_urls,
        additional_instruction="Accept 'direct' and 'nonstop' as equivalent. Treat MCO and SFB as Orlando-area airports."
    )

    # Wheelchair assistance addressed
    evaluator.add_custom_node(
        result=bool(nash_checks.wheelchair_assistance_addressed),
        id="Wheelchair_Assistance_Process_Addressed",
        desc="Plan states wheelchair assistance is requested in advance through the airline (for Nashville passenger 1).",
        parent=nashville_node,
        critical=True
    )

    # Service animal documentation addressed
    evaluator.add_custom_node(
        result=bool(nash_checks.service_animal_docs_addressed),
        id="Service_Animal_Documentation_Addressed",
        desc="Plan addresses service-animal documentation requirements for airline travel (for Nashville passenger 1).",
        parent=nashville_node,
        critical=True
    )

    # Medical equipment special handling addressed
    evaluator.add_custom_node(
        result=bool(nash_checks.medical_equipment_handling_addressed),
        id="Medical_Equipment_Special_Handling_Addressed",
        desc="Plan addresses special handling needs for medical equipment (for Nashville passenger 2).",
        parent=nashville_node,
        critical=True
    )

    # Medical equipment not counted toward baggage allowance addressed
    evaluator.add_custom_node(
        result=bool(nash_checks.medical_equipment_not_count_baggage_addressed),
        id="Medical_Equipment_Not_Count_Toward_Baggage_Addressed",
        desc="Plan states medical equipment does not count toward standard baggage allowance (for Nashville passenger 2).",
        parent=nashville_node,
        critical=True
    )

    # Pet dog minimum age addressed
    evaluator.add_custom_node(
        result=bool(nash_checks.pet_min_age_addressed),
        id="Pet_Dog_Minimum_Age_Addressed",
        desc="Plan confirms pet dog meets minimum age requirement (at least 8 weeks old) (for Nashville passenger 3).",
        parent=nashville_node,
        critical=True
    )

    # Bangor Flights (critical group)
    bangor_node = evaluator.add_parallel(
        id="Bangor_Flights",
        desc="Bangor group flight details and special needs coverage.",
        parent=flights_node,
        critical=True
    )

    # Specific flight info provided
    evaluator.add_custom_node(
        result=bool((bgr.airline and bgr.airline.strip()) or (bgr.flight_number and bgr.flight_number.strip())),
        id="Bangor_Specific_Flight_Info_Provided",
        desc="Answer includes specific airline/flight information for the Bangor (BGR) group.",
        parent=bangor_node,
        critical=True
    )

    # Direct flight from BGR to Orlando area
    bangor_direct_leaf = evaluator.add_leaf(
        id="Bangor_Flight_Direct_From_BGR",
        desc="Bangor group takes a direct/nonstop flight departing from BGR and arriving to Orlando (Orlando-area airport).",
        parent=bangor_node,
        critical=True
    )
    bangor_direct_claim = "There is a direct (nonstop) flight from Bangor International Airport (BGR) to an Orlando-area airport (e.g., Orlando International Airport [MCO] or Orlando Sanford International Airport [SFB]) that aligns with the proposed plan."
    await evaluator.verify(
        claim=bangor_direct_claim,
        node=bangor_direct_leaf,
        sources=bgr_urls,
        additional_instruction="Accept 'direct' and 'nonstop' as equivalent. Treat MCO and SFB as Orlando-area airports."
    )

    # Priority boarding addressed
    evaluator.add_custom_node(
        result=bool(bgr_checks.priority_boarding_addressed),
        id="Priority_Boarding_Addressed",
        desc="Plan addresses priority boarding for Bangor passenger 1.",
        parent=bangor_node,
        critical=True
    )

    # Accessible seating addressed
    evaluator.add_custom_node(
        result=bool(bgr_checks.accessible_seating_addressed),
        id="Accessible_Seating_Addressed",
        desc="Plan addresses accessible seating for Bangor passenger 1.",
        parent=bangor_node,
        critical=True
    )

    # Carry-on only addressed
    evaluator.add_custom_node(
        result=bool(bgr_checks.carryon_only_addressed),
        id="Carryon_Only_Addressed",
        desc="Plan reflects carry-on-only travel for Bangor passenger 2.",
        parent=bangor_node,
        critical=True
    )

    # Carry-on size limit consistency (treated as critical due to framework constraints)
    evaluator.add_custom_node(
        result=_contains_size_22_14_9(bgr_checks.carryon_size_text),
        id="Carryon_Size_Limit_Addressed",
        desc="If carry-on sizing is specified, it is consistent with the typical carry-on size limit (22 inches x 14 inches x 9 inches).",
        parent=bangor_node,
        critical=True
    )

    # Flights reference URLs exist
    all_flight_urls = _dedupe_urls((flights.flights_urls or []) + (nash.urls or []) + (bgr.urls or []))
    evaluator.add_custom_node(
        result=len(all_flight_urls) > 0,
        id="Flights_Reference_URLs",
        desc="Answer provides reliable source URL(s) supporting direct-flight availability/policies and the stated accommodations.",
        parent=flights_node,
        critical=True
    )


async def build_hotel_verification(evaluator: Evaluator, parent_node, extracted: TravelPlanExtraction):
    hotel_node = evaluator.add_parallel(
        id="Orlando_Hotel_Accommodation",
        desc="Book a hotel in the Orlando area meeting ADA-accessible room, pet-friendly, and meeting-space requirements with verifying URLs.",
        parent=parent_node,
        critical=True
    )

    hotel: HotelExtraction = extracted.hotel or HotelExtraction()
    hotel_urls = hotel.urls or []

    # Hotel name provided
    evaluator.add_custom_node(
        result=bool(hotel.name and hotel.name.strip()),
        id="Hotel_Name_Provided",
        desc="Answer specifies the hotel name.",
        parent=hotel_node,
        critical=True
    )

    # Hotel in Orlando area (verify via URLs)
    hotel_loc_leaf = evaluator.add_leaf(
        id="Hotel_In_Orlando_Area",
        desc="Hotel is in the Orlando area.",
        parent=hotel_node,
        critical=True
    )
    hotel_loc_claim = "This hotel is located in the Orlando, Florida area (including Orlando proper or adjacent areas commonly considered 'Orlando area' such as Lake Buena Vista or Kissimmee)."
    await evaluator.verify(
        claim=hotel_loc_claim,
        node=hotel_loc_leaf,
        sources=hotel_urls,
        additional_instruction="Accept 'Orlando', 'Lake Buena Vista', or 'Kissimmee' as Orlando-area."
    )

    # ADA-accessible rooms with required features
    ada_rooms_leaf = evaluator.add_leaf(
        id="At_Least_Two_ADA_Accessible_Rooms_With_Required_Features",
        desc="Hotel provides at least two ADA-accessible rooms, each with minimum 32-inch door width, 36-inch wide route to bed, and an accessible bathroom with grab bars and roll-in shower or accessible tub.",
        parent=hotel_node,
        critical=True
    )
    ada_rooms_claim = "The hotel offers ADA-accessible guestrooms with 32-inch minimum door width, a 36-inch wide accessible route to the bed, and accessible bathrooms with grab bars and either a roll-in shower or an accessible tub, with at least two such accessible rooms available."
    await evaluator.verify(
        claim=ada_rooms_claim,
        node=ada_rooms_leaf,
        sources=hotel_urls,
        additional_instruction="Look for the hotel's accessibility/ADA page or room descriptions. Consider it satisfied if multiple accessible room types are listed or text clearly indicates multiple accessible rooms with these features."
    )

    # Pet-friendly room accepts dogs
    pet_leaf = evaluator.add_leaf(
        id="At_Least_One_Pet_Friendly_Room_Accepts_Dogs",
        desc="Hotel offers at least one pet-friendly room that accepts dogs (with the pet under the hotel's weight/size limits).",
        parent=hotel_node,
        critical=True
    )
    pet_claim = "The hotel is pet-friendly and accepts dogs, subject to typical hotel weight/size limits."
    await evaluator.verify(
        claim=pet_claim,
        node=pet_leaf,
        sources=hotel_urls,
        additional_instruction="Verify via the hotel's pet policy page; acceptance of dogs is required. If weight limits are shown, treat the plan's dog as within typical limits unless the page states otherwise."
    )

    # Meeting room capacity sufficient for group using 20–25 sq ft per person
    meet_leaf = evaluator.add_leaf(
        id="Meeting_Room_Capacity_Sufficient",
        desc="Hotel provides a conference/meeting room that accommodates the entire group using the 20–25 sq ft per person calculation.",
        parent=hotel_node,
        critical=True
    )
    n_people = _group_size()
    min_sqft, max_sqft = _required_sqft_range_for_group(n_people)
    meet_claim = f"The hotel has at least one meeting/conference room suitable for a group of {n_people} people—either with capacity ≥ {n_people} persons or with at least {min_sqft}–{max_sqft} square feet based on 20–25 sq ft per person."
    await evaluator.verify(
        claim=meet_claim,
        node=meet_leaf,
        sources=hotel_urls,
        additional_instruction=f"Pass if the hotel's meeting space page indicates capacity ≥ {n_people} people or square footage ≥ {min_sqft} sq ft."
    )

    # Conference spaces wheelchair accessible
    conf_access_leaf = evaluator.add_leaf(
        id="Conference_Spaces_Wheelchair_Accessible",
        desc="All conference/meeting spaces are wheelchair accessible.",
        parent=hotel_node,
        critical=True
    )
    conf_access_claim = "The hotel's conference/meeting spaces are wheelchair accessible."
    await evaluator.verify(
        claim=conf_access_claim,
        node=conf_access_leaf,
        sources=hotel_urls,
        additional_instruction="Look for accessibility statements about meeting spaces, public areas, or 'accessible meeting rooms/spaces'."
    )

    # Hotel reference URLs exist
    evaluator.add_custom_node(
        result=len(hotel_urls) > 0,
        id="Hotel_Reference_URLs",
        desc="Answer includes reliable URL(s) verifying the hotel’s ADA-room features, pet policy, and meeting/conference space accessibility/capacity basis.",
        parent=hotel_node,
        critical=True
    )


async def build_cruise_verification(evaluator: Evaluator, parent_node, extracted: TravelPlanExtraction):
    cruise_node = evaluator.add_parallel(
        id="Carnival_Cruise_Booking",
        desc="Book a Carnival cruise departing Port Canaveral, calling at Curaçao, meeting accessibility/medical/dietary/embarkation requirements with verifying URLs.",
        parent=parent_node,
        critical=True
    )

    cruise: CruiseExtraction = extracted.cruise or CruiseExtraction()
    # Aggregate cruise URLs
    cruise_urls = _dedupe_urls(
        [cruise.itinerary_url, cruise.accessible_stateroom_url, cruise.dietary_policy_url,
         cruise.medical_staff_policy_url, cruise.medical_center_policy_url, cruise.checkin_policy_url,
         cruise.arrival_time_policy_url, cruise.embarkation_docs_url] + (cruise.urls or [])
    )

    # Cruise line is Carnival
    is_carnival_leaf = evaluator.add_leaf(
        id="Cruise_Is_Carnival",
        desc="Cruise line is Carnival Cruise Line.",
        parent=cruise_node,
        critical=True
    )
    is_carnival_claim = "The cruise is operated by Carnival Cruise Line."
    await evaluator.verify(
        claim=is_carnival_claim,
        node=is_carnival_leaf,
        sources=cruise_urls or cruise.itinerary_url,
        additional_instruction="The itinerary/booking page or official Carnival pages should clearly show the brand 'Carnival'."
    )

    # Departs from Port Canaveral
    dep_pc_leaf = evaluator.add_leaf(
        id="Departs_From_Port_Canaveral",
        desc="Cruise departs from Port Canaveral (Orlando area).",
        parent=cruise_node,
        critical=True
    )
    dep_pc_claim = "The cruise itinerary departs from Port Canaveral, Florida."
    await evaluator.verify(
        claim=dep_pc_claim,
        node=dep_pc_leaf,
        sources=cruise.itinerary_url or cruise_urls,
        additional_instruction="Verify on the itinerary page that the departure port is Port Canaveral (near Orlando)."
    )

    # Includes Curaçao as a port of call
    curacao_leaf = evaluator.add_leaf(
        id="Includes_Curacao_Port_Of_Call",
        desc="Cruise includes Curaçao as a port of call.",
        parent=cruise_node,
        critical=True
    )
    curacao_claim = "The cruise itinerary includes Curaçao as a port of call."
    await evaluator.verify(
        claim=curacao_claim,
        node=curacao_leaf,
        sources=cruise.itinerary_url or cruise_urls,
        additional_instruction="The itinerary should list 'Curaçao' (often Willemstad) among its ports."
    )

    # Accessible stateroom available
    access_cabin_leaf = evaluator.add_leaf(
        id="Wheelchair_Accessible_Stateroom_Available",
        desc="Cruise provides at least one wheelchair-accessible stateroom.",
        parent=cruise_node,
        critical=True
    )
    access_cabin_claim = "The ship offers at least one wheelchair-accessible stateroom."
    await evaluator.verify(
        claim=access_cabin_claim,
        node=access_cabin_leaf,
        sources=cruise.accessible_stateroom_url or cruise_urls,
        additional_instruction="Use the ship's accessibility page or Carnival accessibility resources for accessible accommodation details."
    )

    # Dietary requests with 45+ days advance
    diet_leaf = evaluator.add_leaf(
        id="Dietary_Request_45_Days_Advance",
        desc="Plan indicates special dietary requests are accommodated when submitted at least 45 days in advance.",
        parent=cruise_node,
        critical=True
    )
    diet_claim = "Carnival accommodates special dietary requests when submitted at least 45 days in advance of sailing."
    await evaluator.verify(
        claim=diet_claim,
        node=diet_leaf,
        sources=cruise.dietary_policy_url or cruise_urls,
        additional_instruction="Check official Carnival FAQs/policy pages for timing requirements for dietary requests."
    )

    # Licensed medical staff minimums
    staff_leaf = evaluator.add_leaf(
        id="Licensed_Medical_Staff_Minimums",
        desc="Cruise has licensed medical staff (minimum 1 doctor and 2 nurses).",
        parent=cruise_node,
        critical=True
    )
    staff_claim = "The ship carries licensed medical staff including at least one doctor and two nurses."
    await evaluator.verify(
        claim=staff_claim,
        node=staff_leaf,
        sources=cruise.medical_staff_policy_url or cruise_urls,
        additional_instruction="Verify via Carnival's medical center or health services policy pages."
    )

    # Medical center available
    med_center_leaf = evaluator.add_leaf(
        id="Medical_Center_Available",
        desc="Cruise has a medical center (available 24 hours, per constraints).",
        parent=cruise_node,
        critical=True
    )
    med_center_claim = "The ship has an onboard medical center available 24 hours a day (or with 24-hour emergency availability)."
    await evaluator.verify(
        claim=med_center_claim,
        node=med_center_leaf,
        sources=cruise.medical_center_policy_url or cruise_urls,
        additional_instruction="Accept phrasing indicating 24-hour emergency availability or round-the-clock access."
    )

    # Online check-in before midnight
    checkin_leaf = evaluator.add_leaf(
        id="Online_Checkin_Before_Midnight",
        desc="Plan states all passengers must complete online check-in before midnight prior to sailing.",
        parent=cruise_node,
        critical=True
    )
    checkin_claim = "All passengers must complete online check-in no later than midnight prior to sailing."
    await evaluator.verify(
        claim=checkin_claim,
        node=checkin_leaf,
        sources=cruise.checkin_policy_url or cruise_urls,
        additional_instruction="Look for online check-in cutoff timing on Carnival's official check-in policy pages."
    )

    # Arrive at least 60 minutes before departure
    arrive_leaf = evaluator.add_leaf(
        id="Arrive_At_Least_60_Min_Before_Departure",
        desc="Plan states passengers must arrive at least 60 minutes before departure.",
        parent=cruise_node,
        critical=True
    )
    arrive_claim = "Passengers must arrive at the cruise port at least 60 minutes before departure."
    await evaluator.verify(
        claim=arrive_claim,
        node=arrive_leaf,
        sources=cruise.arrival_time_policy_url or cruise_urls,
        additional_instruction="Verify via Carnival's boarding/arrival time policy pages."
    )

    # Embarkation documentation
    docs_leaf = evaluator.add_leaf(
        id="Embarkation_Documentation",
        desc="Plan includes required embarkation documents: boarding pass, photo ID, and citizenship documents.",
        parent=cruise_node,
        critical=True
    )
    docs_claim = "Required embarkation documents include the boarding pass, a government-issued photo ID, and citizenship documents (such as a passport)."
    await evaluator.verify(
        claim=docs_claim,
        node=docs_leaf,
        sources=cruise.embarkation_docs_url or cruise_urls,
        additional_instruction="Verify with Carnival's official embarkation/boarding document policy pages."
    )

    # Cruise reference URLs exist
    evaluator.add_custom_node(
        result=len(cruise_urls) > 0,
        id="Cruise_Reference_URLs",
        desc="Answer includes reliable URL(s) verifying itinerary/ports and the stated accessibility, dietary, medical, and embarkation requirements.",
        parent=cruise_node,
        critical=True
    )


async def build_curacao_verification(evaluator: Evaluator, parent_node, extracted: TravelPlanExtraction):
    cur_node = evaluator.add_parallel(
        id="Curacao_Entry_Requirements",
        desc="Ensure US citizen travelers meet Curaçao entry requirements with verifying URLs.",
        parent=parent_node,
        critical=True
    )

    cur: CuracaoExtraction = extracted.curacao or CuracaoExtraction()
    cur_urls = _dedupe_urls([cur.passport_requirement_url, cur.di_card_url] + (cur.urls or []))

    # Valid passports for duration of stay
    passport_leaf = evaluator.add_leaf(
        id="Valid_Passport_Duration",
        desc="All US citizen travelers have valid passports for the duration of stay.",
        parent=cur_node,
        critical=True
    )
    passport_claim = "US citizens entering Curaçao must have a valid passport for the duration of their stay."
    await evaluator.verify(
        claim=passport_claim,
        node=passport_leaf,
        sources=cur.passport_requirement_url or cur_urls,
        additional_instruction="Use official Curaçao tourism/immigration or US government travel resources."
    )

    # DI card within 7 days prior to departure
    di_card_leaf = evaluator.add_leaf(
        id="DI_Card_Within_7_Days",
        desc="Curaçao Digital Immigration Card (DI card) is completed within 7 days prior to departure.",
        parent=cur_node,
        critical=True
    )
    di_card_claim = "Travelers to Curaçao must complete the Digital Immigration (DI) Card within 7 days prior to departure."
    await evaluator.verify(
        claim=di_card_claim,
        node=di_card_leaf,
        sources=cur.di_card_url or cur_urls,
        additional_instruction="Use the official DI card site or official Curaçao entry requirement pages that specify the 7-day timing."
    )

    # Reference URLs exist
    evaluator.add_custom_node(
        result=len(cur_urls) > 0,
        id="Curacao_Entry_Reference_URLs",
        desc="Answer includes reliable URL(s) verifying Curaçao passport and DI-card timing requirements.",
        parent=cur_node,
        critical=True
    )


async def build_insurance_verification(evaluator: Evaluator, parent_node, extracted: TravelPlanExtraction):
    ins_node = evaluator.add_parallel(
        id="Travel_Insurance",
        desc="Purchase comprehensive travel insurance meeting the stated coverage and waiver requirements, with verifying URLs.",
        parent=parent_node,
        critical=True
    )

    ins: InsuranceExtraction = extracted.insurance or InsuranceExtraction()
    policy_urls = ins.policy_urls or []

    # Medical evacuation >= $100,000
    medevac_leaf = evaluator.add_leaf(
        id="Medical_Evacuation_At_Least_100k",
        desc="Policy includes medical evacuation coverage of at least $100,000.",
        parent=ins_node,
        critical=True
    )
    medevac_claim = "This travel insurance policy includes at least $100,000 in medical evacuation coverage."
    await evaluator.verify(
        claim=medevac_claim,
        node=medevac_leaf,
        sources=policy_urls,
        additional_instruction="Verify on the policy or plan detail page that medical evacuation or emergency medical transportation coverage is ≥ $100,000."
    )

    # Trip cancellation for non-refundable prepaids
    trip_cancel_leaf = evaluator.add_leaf(
        id="Trip_Cancellation_For_Nonrefundable_Prepaids",
        desc="Policy includes trip cancellation coverage for non-refundable prepaid costs.",
        parent=ins_node,
        critical=True
    )
    trip_cancel_claim = "This policy includes trip cancellation coverage for non-refundable prepaid trip costs."
    await evaluator.verify(
        claim=trip_cancel_claim,
        node=trip_cancel_leaf,
        sources=policy_urls,
        additional_instruction="Verify that trip cancellation coverage applies to non-refundable prepaid expenses; plan brochure or certificate should state this."
    )

    # Pre-existing condition waiver terms
    waiver_leaf = evaluator.add_leaf(
        id="Preexisting_Condition_Waiver_Requirements",
        desc="Policy includes a pre-existing condition waiver, requiring purchase within 15 days of initial trip deposit and insuring 100% of trip costs.",
        parent=ins_node,
        critical=True
    )
    waiver_claim = "The policy offers a pre-existing condition waiver that requires purchasing the insurance within 15 days of the initial trip deposit and insuring 100% of prepaid trip costs."
    await evaluator.verify(
        claim=waiver_claim,
        node=waiver_leaf,
        sources=policy_urls,
        additional_instruction="Verify the specific timing window and the requirement to insure 100% of trip costs in the plan wording."
    )

    # Insurance policy details provided (existence check)
    evaluator.add_custom_node(
        result=bool((ins.provider_name and ins.provider_name.strip()) and len(policy_urls) > 0),
        id="Insurance_Policy_Details_Provided",
        desc="Answer provides travel insurance policy details sufficient to confirm coverages and waiver terms.",
        parent=ins_node,
        critical=True
    )

    # Insurance reference URLs exist
    evaluator.add_custom_node(
        result=len(policy_urls) > 0,
        id="Insurance_Reference_URLs",
        desc="Answer includes reliable URL(s) verifying the insurance coverage levels and waiver requirements.",
        parent=ins_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the comprehensive travel plan with accessibility and special needs requirements.
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

    # Extract the structured plan from the answer
    extracted: TravelPlanExtraction = await evaluator.extract(
        prompt=prompt_extract_travel_plan(),
        template_class=TravelPlanExtraction,
        extraction_name="travel_plan_extraction"
    )

    # Build top-level critical node (parallel aggregation)
    complete_plan = evaluator.add_parallel(
        id="Complete_Travel_Plan",
        desc="Provide a detailed travel plan covering flights (both groups), Orlando hotel, Carnival cruise to Curaçao, Curaçao entry documentation, and travel insurance; include reliable reference URLs verifying requirements.",
        parent=root,
        critical=True
    )

    # Build subtrees
    await build_flights_verification(evaluator, complete_plan, extracted)
    await build_hotel_verification(evaluator, complete_plan, extracted)
    await build_cruise_verification(evaluator, complete_plan, extracted)
    await build_curacao_verification(evaluator, complete_plan, extracted)
    await build_insurance_verification(evaluator, complete_plan, extracted)

    # Add custom info for clarity
    evaluator.add_custom_info(
        info={
            "total_travelers": _group_size(),
            "meeting_room_required_sqft_range": _required_sqft_range_for_group(_group_size())
        },
        info_type="computed_requirements",
        info_name="computed_requirements"
    )

    return evaluator.get_summary()