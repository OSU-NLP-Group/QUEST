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
TASK_ID = "family_vacation_plan_feb2026"
TASK_DESCRIPTION = (
    "A family of four (2 adults, 1 child aged 5, and 1 child aged 8) along with their small dog is planning a two-week vacation departing from the United States in February 2026. "
    "Their travel plans include:\n\n"
    "Week 1: Caribbean Cruise\n"
    "- They want to take an MSC cruise that visits Grenada\n"
    "- They will fly to Grenada to board the cruise ship\n"
    "- All family members hold US passports\n\n"
    "Week 2: Edinburgh, Scotland\n"
    "- They will fly directly from a US city to Edinburgh\n"
    "- They plan to spend one full day visiting Edinburgh Zoo on a weekday in February\n"
    "- They need pet-friendly accommodation that can house their family of 4 plus their dog\n"
    "- They plan to use public transportation in Edinburgh\n\n"
    "Your task is to create a comprehensive travel plan that addresses all of the following requirements:\n"
    "1. Travel Documentation: Verify passport validity requirements for Caribbean cruise travel and confirm visa requirements for US citizens traveling to Grenada for tourism.\n"
    "2. Caribbean Flight: Identify the correct airport code for Grenada and confirm flight availability to this destination.\n"
    "3. MSC Cruise: Confirm that MSC Cruises operates ships that visit St. George's, Grenada, and name at least one specific ship that serves this route.\n"
    "4. Edinburgh Flight: Identify at least one US city that offers direct flights to Edinburgh and name the airline(s) operating these direct routes.\n"
    "5. Edinburgh Zoo Visit: For their visit on a weekday in February, provide hours, last entry, admission prices, total family cost, parking cost, and confirm the zoo is open.\n"
    "6. Edinburgh Accommodation: Identify accommodation requirements including family room capacity for 4 people and pet-friendly policy for their dog.\n"
    "7. Edinburgh Transportation: Identify at least one bus route number that stops at Edinburgh Zoo, and provide the bus fares for adults and children traveling from the city center to the zoo.\n"
    "Provide specific details including airport codes, ship names, airline names, prices, times, and bus route numbers where applicable. All information must be grounded in verifiable sources."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TravelDocsExtraction(BaseModel):
    passport_validity_statement: Optional[str] = None
    passport_validity_sources: List[str] = Field(default_factory=list)
    visa_requirement_statement: Optional[str] = None
    visa_requirement_sources: List[str] = Field(default_factory=list)
    cruise_passport_validity_statement: Optional[str] = None
    cruise_passport_validity_sources: List[str] = Field(default_factory=list)


class CaribbeanFlightExtraction(BaseModel):
    grenada_airport_code: Optional[str] = None
    airport_code_sources: List[str] = Field(default_factory=list)
    flight_availability_sources: List[str] = Field(default_factory=list)


class MSCCruiseExtraction(BaseModel):
    msc_grenada_statement: Optional[str] = None
    msc_grenada_sources: List[str] = Field(default_factory=list)
    ship_names: List[str] = Field(default_factory=list)
    ship_sources: List[str] = Field(default_factory=list)


class EdinburghFlightExtraction(BaseModel):
    direct_cities: List[str] = Field(default_factory=list)
    airlines: List[str] = Field(default_factory=list)
    flight_sources: List[str] = Field(default_factory=list)


class ZooExtraction(BaseModel):
    feb_opening_time: Optional[str] = None
    feb_closing_time: Optional[str] = None
    last_entry_time: Optional[str] = None
    adult_price: Optional[str] = None
    child_price: Optional[str] = None
    family_total_price: Optional[str] = None
    parking_cost: Optional[str] = None
    open_on_weekdays_feb: Optional[bool] = None
    zoo_sources: List[str] = Field(default_factory=list)


class AccommodationExtraction(BaseModel):
    hotel_name: Optional[str] = None
    hotel_url: Optional[str] = None
    family_room_capacity: Optional[str] = None
    pet_policy: Optional[str] = None
    pet_fee: Optional[str] = None
    accommodation_sources: List[str] = Field(default_factory=list)


class TransportationExtraction(BaseModel):
    bus_routes: List[str] = Field(default_factory=list)
    adult_bus_fare: Optional[str] = None
    child_bus_fare: Optional[str] = None
    transport_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_travel_docs() -> str:
    return (
        "Extract the travel documentation details mentioned in the answer. "
        "Return the following fields:\n"
        "- passport_validity_statement: the stated requirement for US passport validity for Grenada tourism or Caribbean cruise travel (verbatim from answer)\n"
        "- passport_validity_sources: all URLs in the answer that support the passport validity requirement\n"
        "- visa_requirement_statement: the stated visa policy for US citizens visiting Grenada for tourism (verbatim)\n"
        "- visa_requirement_sources: all URLs in the answer that support the visa policy\n"
        "- cruise_passport_validity_statement: the stated MSC cruise passport validity requirement (verbatim)\n"
        "- cruise_passport_validity_sources: all URLs in the answer that support the MSC passport rule\n"
        "If any field is not present in the answer, set it to null (or empty list for sources)."
    )


def prompt_extract_caribbean_flight() -> str:
    return (
        "Extract the Caribbean flight details from the answer. Return:\n"
        "- grenada_airport_code: the airport code given for Grenada (e.g., GND)\n"
        "- airport_code_sources: URLs supporting the airport code identification\n"
        "- flight_availability_sources: URLs supporting the existence of scheduled flights to Grenada (Maurice Bishop International Airport)\n"
        "Use only URLs present in the answer; if not provided, return empty lists."
    )


def prompt_extract_msc_cruise() -> str:
    return (
        "Extract the MSC cruise details mentioned in the answer. Return:\n"
        "- msc_grenada_statement: a short statement indicating MSC Cruises operates ships that visit St. George's, Grenada (verbatim)\n"
        "- msc_grenada_sources: URLs supporting that MSC calls at St. George's, Grenada\n"
        "- ship_names: list of specific MSC ship names stated to visit Grenada\n"
        "- ship_sources: URLs supporting that the listed ship(s) call at St. George's, Grenada\n"
        "Set fields to null or empty list if not in the answer."
    )


def prompt_extract_edinburgh_flight() -> str:
    return (
        "Extract the US-to-Edinburgh direct flight details. Return:\n"
        "- direct_cities: list of US cities or airports claimed to have direct/nonstop flights to Edinburgh\n"
        "- airlines: list of airlines named as operating direct US-Edinburgh routes\n"
        "- flight_sources: URLs supporting the direct routes and operating airlines\n"
        "Use only URLs present in the answer."
    )


def prompt_extract_zoo() -> str:
    return (
        "Extract the Edinburgh Zoo visit details for a weekday in February. Return:\n"
        "- feb_opening_time: the opening time in February (e.g., 10am)\n"
        "- feb_closing_time: the closing time in February (e.g., 4pm)\n"
        "- last_entry_time: last entry time before closing (e.g., 3pm)\n"
        "- adult_price: adult admission price or range string (e.g., '£30.00-£32.50')\n"
        "- child_price: child admission price or range string for ages 3-15 (e.g., '£22.00-£24.50')\n"
        "- family_total_price: the total admission cost for 2 adults and 2 children as stated in the answer (string)\n"
        "- parking_cost: the parking cost at Edinburgh Zoo (string like '£4')\n"
        "- open_on_weekdays_feb: boolean indicating zoo is open on weekdays in February (true/false)\n"
        "- zoo_sources: URLs supporting all the above details\n"
        "Use only URLs present in the answer; if not present, set fields to null or empty list."
    )


def prompt_extract_accommodation() -> str:
    return (
        "Extract the Edinburgh accommodation details. Return:\n"
        "- hotel_name: the hotel or accommodation name\n"
        "- hotel_url: the hotel's official URL if provided\n"
        "- family_room_capacity: the stated capacity for a family room (e.g., 'sleeps 4')\n"
        "- pet_policy: the stated pet/dog policy (verbatim summary)\n"
        "- pet_fee: the stated pet/dog fee (string, if any)\n"
        "- accommodation_sources: URLs supporting capacity and pet policy\n"
        "Use only URLs present in the answer."
    )


def prompt_extract_transportation() -> str:
    return (
        "Extract the Edinburgh transportation details for getting to the zoo. Return:\n"
        "- bus_routes: list of bus route numbers that stop at or serve Edinburgh Zoo (e.g., ['12', '26', '31'])\n"
        "- adult_bus_fare: adult single fare from city to zoo (e.g., '£3.50')\n"
        "- child_bus_fare: child single fare from city to zoo (e.g., '£1.75')\n"
        "- transport_sources: URLs supporting the routes and fares\n"
        "Use only URLs present in the answer."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_travel_documentation(
    evaluator: Evaluator,
    parent_node,
    docs: TravelDocsExtraction,
) -> None:
    section = evaluator.add_parallel(
        id="Travel_Documentation",
        desc="Verify all travel documentation requirements are met for international travel",
        parent=parent_node,
        critical=False
    )

    # Passport validity (Grenada tourism)
    node_passport_validity = evaluator.add_leaf(
        id="Passport_Validity_Outbound",
        desc="Confirm passport is valid for at least 6 months beyond the return date from Grenada",
        parent=section,
        critical=True
    )
    claim_passport_validity = (
        "For US citizens traveling to Grenada for tourism, passports should be valid for at least six months "
        "beyond the end of travel (or date of departure from Grenada)."
    )
    await evaluator.verify(
        claim=claim_passport_validity,
        node=node_passport_validity,
        sources=docs.passport_validity_sources,
        additional_instruction="Use official sources (e.g., U.S. government or Grenada government/tourism) to confirm the 6-month validity guidance."
    )

    # Visa requirement (US citizens, < 90 days)
    node_visa = evaluator.add_leaf(
        id="Visa_Requirement_Check",
        desc="Confirm that US citizens do not require a visa for tourist travel to Grenada (stays under 90 days)",
        parent=section,
        critical=True
    )
    claim_visa = "U.S. citizens do not require a visa for tourist visits to Grenada for stays under 90 days."
    await evaluator.verify(
        claim=claim_visa,
        node=node_visa,
        sources=docs.visa_requirement_sources,
        additional_instruction="Verify the visa policy for U.S. citizens visiting Grenada for tourism; stays under 90 days should be visa-free."
    )

    # Cruise passport validity (MSC)
    node_cruise_passport = evaluator.add_leaf(
        id="Cruise_Passport_Validity",
        desc="Verify passport validity meets cruise line requirement of 6 months after cruise ends",
        parent=section,
        critical=True
    )
    claim_cruise_passport = "MSC Cruises requires that passengers' passports be valid for at least six months after the cruise ends."
    await evaluator.verify(
        claim=claim_cruise_passport,
        node=node_cruise_passport,
        sources=docs.cruise_passport_validity_sources,
        additional_instruction="Confirm MSC's documented passport validity policy for international cruises regarding six months of validity after the cruise."
    )


async def verify_caribbean_flight(
    evaluator: Evaluator,
    parent_node,
    cf: CaribbeanFlightExtraction,
) -> None:
    section = evaluator.add_parallel(
        id="Caribbean_Flight_Logistics",
        desc="Verify flight arrangements to Caribbean destination for cruise departure",
        parent=parent_node,
        critical=False
    )

    # Airport code GND
    node_code = evaluator.add_leaf(
        id="Grenada_Airport_Code",
        desc="Identify the correct airport code for Grenada as GND (Maurice Bishop International Airport)",
        parent=section,
        critical=True
    )
    code_used = (cf.grenada_airport_code or "GND").strip()
    claim_code = f"The IATA airport code for Maurice Bishop International Airport in Grenada is '{code_used}'."
    await evaluator.verify(
        claim=claim_code,
        node=node_code,
        sources=cf.airport_code_sources,
        additional_instruction="Confirm the IATA code 'GND' for Maurice Bishop International Airport (St. George's, Grenada)."
    )

    # Flight route availability (non-critical)
    node_avail = evaluator.add_leaf(
        id="Flight_Route_Availability",
        desc="Confirm availability of flights to Grenada from the departure city",
        parent=section,
        critical=False
    )
    claim_avail = "There are scheduled commercial flights to Maurice Bishop International Airport (GND), Grenada."
    await evaluator.verify(
        claim=claim_avail,
        node=node_avail,
        sources=cf.flight_availability_sources,
        additional_instruction="Any reputable source showing scheduled service to GND (airline route map, airport destinations list, flight aggregators) is acceptable."
    )


async def verify_msc_cruise(
    evaluator: Evaluator,
    parent_node,
    cruise: MSCCruiseExtraction,
) -> None:
    section = evaluator.add_parallel(
        id="MSC_Cruise_Selection",
        desc="Verify appropriate MSC cruise selection that visits Grenada",
        parent=parent_node,
        critical=False
    )

    # MSC calls at St. George's, Grenada
    node_cruise_line = evaluator.add_leaf(
        id="Cruise_Line_Confirmation",
        desc="Confirm that MSC Cruises operates ships visiting St. George's, Grenada",
        parent=section,
        critical=True
    )
    claim_cruise_line = "MSC Cruises operates itineraries that call at St. George's, Grenada."
    await evaluator.verify(
        claim=claim_cruise_line,
        node=node_cruise_line,
        sources=cruise.msc_grenada_sources,
        additional_instruction="Use MSC's official site or reputable cruise itinerary listings that explicitly show St. George's, Grenada as a port call."
    )

    # Specific ship identification
    node_ship = evaluator.add_leaf(
        id="Ship_Identification",
        desc="Identify at least one MSC ship that visits Grenada (e.g., MSC Seaview, MSC Virtuosa, or MSC Meraviglia)",
        parent=section,
        critical=True
    )
    ship_name = cruise.ship_names[0] if cruise.ship_names else "an MSC ship"
    claim_ship = f"The MSC ship '{ship_name}' has an itinerary that includes a call at St. George's, Grenada."
    await evaluator.verify(
        claim=claim_ship,
        node=node_ship,
        sources=cruise.ship_sources or cruise.msc_grenada_sources,
        additional_instruction="Verify via official itinerary pages or authoritative listings that the named MSC ship calls at St. George's, Grenada."
    )


async def verify_edinburgh_flight(
    evaluator: Evaluator,
    parent_node,
    ef: EdinburghFlightExtraction,
) -> None:
    section = evaluator.add_parallel(
        id="Edinburgh_Flight_Arrangements",
        desc="Verify direct flight availability from US to Edinburgh",
        parent=parent_node,
        critical=False
    )

    # Direct flight cities
    node_city = evaluator.add_leaf(
        id="Direct_Flight_Cities",
        desc="Identify at least one US city offering direct flights to Edinburgh (New York Newark, New York JFK, Chicago O'Hare, or Philadelphia)",
        parent=section,
        critical=True
    )
    city = ef.direct_cities[0] if ef.direct_cities else "a U.S. city (e.g., Newark, JFK, Chicago O'Hare, Philadelphia)"
    claim_city = f"There are direct (nonstop) flights from {city} to Edinburgh (EDI)."
    await evaluator.verify(
        claim=claim_city,
        node=node_city,
        sources=ef.flight_sources,
        additional_instruction="Confirm nonstop service exists; acceptable sources include airline schedules, airport route maps, or authoritative travel listings."
    )

    # Airline identification
    node_airline = evaluator.add_leaf(
        id="Airline_Identification",
        desc="Identify airlines operating direct US-Edinburgh routes (United Airlines or American Airlines)",
        parent=section,
        critical=True
    )
    airline = ef.airlines[0] if ef.airlines else "a U.S. airline (e.g., United Airlines or American Airlines)"
    claim_airline = f"{airline} operates direct (nonstop) flights between a U.S. city and Edinburgh (EDI)."
    await evaluator.verify(
        claim=claim_airline,
        node=node_airline,
        sources=ef.flight_sources,
        additional_instruction="Prefer airline official schedules or airport route listings to verify nonstop operation."
    )


async def verify_zoo_visit(
    evaluator: Evaluator,
    parent_node,
    zoo: ZooExtraction,
) -> None:
    section = evaluator.add_parallel(
        id="Edinburgh_Zoo_Visit",
        desc="Plan and validate all aspects of Edinburgh Zoo visit for family",
        parent=parent_node,
        critical=False
    )

    # Visit date validation (closed only Dec 25)
    node_date = evaluator.add_leaf(
        id="Visit_Date_Validation",
        desc="Confirm visit date is not December 25 (Christmas Day - only closure day)",
        parent=section,
        critical=True
    )
    claim_date = "Edinburgh Zoo is closed only on Christmas Day (25 December) and is open on weekdays in February."
    await evaluator.verify(
        claim=claim_date,
        node=node_date,
        sources=zoo.zoo_sources,
        additional_instruction="Check the official opening times or visitor information pages for closure policy."
    )

    # Opening hours for February
    node_hours = evaluator.add_leaf(
        id="Opening_Hours_Verification",
        desc="Verify opening hours for February visit date (10am-4pm for January-February period)",
        parent=section,
        critical=True
    )
    claim_hours = "In February, Edinburgh Zoo's opening hours are approximately 10:00 to 16:00."
    await evaluator.verify(
        claim=claim_hours,
        node=node_hours,
        sources=zoo.zoo_sources,
        additional_instruction="Confirm the published seasonal opening hours for January–February."
    )

    # Last entry one hour before closing
    node_last_entry = evaluator.add_leaf(
        id="Last_Entry_Time",
        desc="Confirm awareness that last entry is one hour before closing time (3pm for February)",
        parent=section,
        critical=True
    )
    claim_last_entry = "Last entry to Edinburgh Zoo is one hour before closing (so 15:00 if closing at 16:00 in February)."
    await evaluator.verify(
        claim=claim_last_entry,
        node=node_last_entry,
        sources=zoo.zoo_sources,
        additional_instruction="Use the official visit information that states last entry policy."
    )

    # Adult admission cost range
    node_adult_price = evaluator.add_leaf(
        id="Adult_Admission_Cost",
        desc="Provide correct adult admission price range (£30.00-£32.50 per adult)",
        parent=section,
        critical=True
    )
    claim_adult_price = "The standard adult admission price for Edinburgh Zoo is between £30.00 and £32.50."
    await evaluator.verify(
        claim=claim_adult_price,
        node=node_adult_price,
        sources=zoo.zoo_sources,
        additional_instruction="Verify the general public/adult price band; ranges or seasonal variation within this band are acceptable."
    )

    # Child admission cost range
    node_child_price = evaluator.add_leaf(
        id="Child_Admission_Cost",
        desc="Provide correct child admission price for ages 3-15 (£22.00-£24.50 per child)",
        parent=section,
        critical=True
    )
    claim_child_price = "The standard child admission price (ages 3–15) for Edinburgh Zoo is between £22.00 and £24.50."
    await evaluator.verify(
        claim=claim_child_price,
        node=node_child_price,
        sources=zoo.zoo_sources,
        additional_instruction="Confirm pricing for ages 3–15; ranges due to peak/off-peak tickets are acceptable."
    )

    # Total family admission (2 adults + 2 children)
    node_family_total = evaluator.add_leaf(
        id="Total_Family_Admission",
        desc="Calculate total admission cost for family of 2 adults and 2 children (ages 5 and 8)",
        parent=section,
        critical=True
    )
    claim_family_total = (
        "Based on the stated adult and child prices, the total admission for a family of two adults and two children "
        "would fall between £104.00 and £114.00."
    )
    await evaluator.verify(
        claim=claim_family_total,
        node=node_family_total,
        sources=zoo.zoo_sources,
        additional_instruction="Compute using two adult tickets within £30–£32.50 and two child tickets within £22–£24.50."
    )

    # Parking cost
    node_parking = evaluator.add_leaf(
        id="Parking_Cost",
        desc="Identify parking cost at Edinburgh Zoo (£4 for visitors)",
        parent=section,
        critical=True
    )
    claim_parking = "Visitor parking at Edinburgh Zoo costs £4."
    await evaluator.verify(
        claim=claim_parking,
        node=node_parking,
        sources=zoo.zoo_sources,
        additional_instruction="Use official parking information."
    )


async def verify_accommodation(
    evaluator: Evaluator,
    parent_node,
    acc: AccommodationExtraction,
) -> None:
    section = evaluator.add_parallel(
        id="Edinburgh_Accommodation",
        desc="Verify hotel accommodation meets all family requirements",
        parent=parent_node,
        critical=False
    )

    # Family room capacity for 4
    node_capacity = evaluator.add_leaf(
        id="Family_Room_Capacity",
        desc="Confirm hotel offers family rooms that can accommodate 4 people (2 adults, 2 children)",
        parent=section,
        critical=True
    )
    hotel_display = acc.hotel_name or "The selected hotel"
    claim_capacity = f"{hotel_display} offers a room or configuration that accommodates 4 people."
    capacity_sources = [acc.hotel_url] if acc.hotel_url else []
    capacity_sources += acc.accommodation_sources
    await evaluator.verify(
        claim=claim_capacity,
        node=node_capacity,
        sources=capacity_sources,
        additional_instruction="Confirm via official hotel page or reputable booking site that a room type sleeping 4 is offered."
    )

    # Pet-friendly policy (dogs) and any fees
    node_pet = evaluator.add_leaf(
        id="Pet_Friendly_Policy",
        desc="Verify hotel accepts pets, specifically dogs, and identify any associated fees",
        parent=section,
        critical=True
    )
    if acc.pet_fee:
        claim_pet = f"{hotel_display} accepts dogs (pet-friendly). The pet fee policy includes {acc.pet_fee}."
    else:
        claim_pet = f"{hotel_display} accepts dogs (pet-friendly)."
    pet_sources = [acc.hotel_url] if acc.hotel_url else []
    pet_sources += acc.accommodation_sources
    await evaluator.verify(
        claim=claim_pet,
        node=node_pet,
        sources=pet_sources,
        additional_instruction="Look for pet policy text (dog acceptance) and any stated fee or deposit."
    )

    # Hotel location / access to zoo (non-critical)
    node_location = evaluator.add_leaf(
        id="Hotel_Location",
        desc="Confirm hotel has reasonable access to Edinburgh Zoo via public transportation or is near the zoo",
        parent=section,
        critical=False
    )
    claim_location = (
        f"{hotel_display} has reasonable access to Edinburgh Zoo via public transport or is within a short travel distance."
    )
    await evaluator.verify(
        claim=claim_location,
        node=node_location,
        sources=pet_sources,
        additional_instruction="Accept evidence showing proximity on map or transit guidance indicating an easy bus route to the zoo."
    )


async def verify_transportation(
    evaluator: Evaluator,
    parent_node,
    trans: TransportationExtraction,
) -> None:
    section = evaluator.add_parallel(
        id="Edinburgh_Transportation",
        desc="Plan transportation logistics in Edinburgh for zoo visit",
        parent=parent_node,
        critical=False
    )

    # Bus routes to zoo (buses 12, 26, or 31)
    node_routes = evaluator.add_leaf(
        id="Bus_Routes_to_Zoo",
        desc="Identify bus routes that stop at Edinburgh Zoo entrance (buses 12, 26, or 31)",
        parent=section,
        critical=True
    )
    route = trans.bus_routes[0] if trans.bus_routes else "12"
    claim_route = f"Bus route {route} stops at or serves Edinburgh Zoo."
    await evaluator.verify(
        claim=claim_route,
        node=node_routes,
        sources=trans.transport_sources,
        additional_instruction="Check Lothian Buses or official route info for routes 12, 26, or 31 serving the zoo stop on Corstorphine Road."
    )

    # Adult bus fare
    node_fare_adult = evaluator.add_leaf(
        id="Bus_Fare_Adult",
        desc="Provide adult bus fare from city to zoo (£3.50)",
        parent=section,
        critical=True
    )
    claim_fare_adult = "The adult single fare on Lothian Buses is £3.50."
    await evaluator.verify(
        claim=claim_fare_adult,
        node=node_fare_adult,
        sources=trans.transport_sources,
        additional_instruction="Verify the current adult single fare on Lothian Buses; fare applies city center to zoo."
    )

    # Child bus fare
    node_fare_child = evaluator.add_leaf(
        id="Bus_Fare_Child",
        desc="Provide child bus fare from city to zoo (£1.75)",
        parent=section,
        critical=True
    )
    claim_fare_child = "The child single fare on Lothian Buses is £1.75."
    await evaluator.verify(
        claim=claim_fare_child,
        node=node_fare_child,
        sources=trans.transport_sources,
        additional_instruction="Verify the current child single fare on Lothian Buses; fare applies city center to zoo."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the agent's travel plan answer for the family vacation task.
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

    # Concurrent extractions
    docs_task = evaluator.extract(
        prompt=prompt_extract_travel_docs(),
        template_class=TravelDocsExtraction,
        extraction_name="travel_documentation"
    )
    carib_task = evaluator.extract(
        prompt=prompt_extract_caribbean_flight(),
        template_class=CaribbeanFlightExtraction,
        extraction_name="caribbean_flight"
    )
    cruise_task = evaluator.extract(
        prompt=prompt_extract_msc_cruise(),
        template_class=MSCCruiseExtraction,
        extraction_name="msc_cruise"
    )
    edi_flight_task = evaluator.extract(
        prompt=prompt_extract_edinburgh_flight(),
        template_class=EdinburghFlightExtraction,
        extraction_name="edinburgh_flight"
    )
    zoo_task = evaluator.extract(
        prompt=prompt_extract_zoo(),
        template_class=ZooExtraction,
        extraction_name="edinburgh_zoo"
    )
    acc_task = evaluator.extract(
        prompt=prompt_extract_accommodation(),
        template_class=AccommodationExtraction,
        extraction_name="edinburgh_accommodation"
    )
    trans_task = evaluator.extract(
        prompt=prompt_extract_transportation(),
        template_class=TransportationExtraction,
        extraction_name="edinburgh_transportation"
    )

    (
        docs_info,
        carib_info,
        cruise_info,
        edi_info,
        zoo_info,
        acc_info,
        trans_info
    ) = await asyncio.gather(
        docs_task, carib_task, cruise_task, edi_flight_task, zoo_task, acc_task, trans_task
    )

    # Build top-level plan node (non-critical to allow partial scoring across sections)
    plan_node = evaluator.add_parallel(
        id="Family_Vacation_Plan",
        desc="Complete validation of a two-week family vacation plan including Caribbean cruise and Edinburgh visit",
        parent=root,
        critical=False
    )

    # Verify each section
    await verify_travel_documentation(evaluator, plan_node, docs_info)
    await verify_caribbean_flight(evaluator, plan_node, carib_info)
    await verify_msc_cruise(evaluator, plan_node, cruise_info)
    await verify_edinburgh_flight(evaluator, plan_node, edi_info)
    await verify_zoo_visit(evaluator, plan_node, zoo_info)
    await verify_accommodation(evaluator, plan_node, acc_info)
    await verify_transportation(evaluator, plan_node, trans_info)

    return evaluator.get_summary()