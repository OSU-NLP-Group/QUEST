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
TASK_ID = "complete_travel_plan"
TASK_DESCRIPTION = """Plan a complete travel itinerary for a trip from Bangor, Maine to San Diego, California with the following specific requirements:

Flight Routing:
- Your journey must depart from Bangor International Airport (BGR)
- You must connect through John F. Kennedy International Airport (JFK) in New York
- Your connection time at JFK must be at least 2 hours to accommodate international-to-domestic transfer procedures
- Your final destination must be San Diego International Airport (SAN)

JFK Layover:
- During your layover at JFK, identify one Priority Pass accessible lounge located in Terminal 4 that you can use
- Provide the lounge name and its specific location within Terminal 4

San Diego Accommodation:
- Select a hotel near San Diego International Airport that meets ALL of the following criteria:
  - Located within 2 miles of SAN Airport
  - Offers free shuttle service to and from the airport
- Provide the hotel name, address, and confirm both the distance requirement and shuttle service availability

National Park Visit:
- Identify one national park within day-trip distance from San Diego that you plan to visit
- The park must accept the America the Beautiful Pass for entrance
- Provide the park name, location, and confirm it accepts the America the Beautiful Pass

For each component of your itinerary (flights, lounge, hotel, and national park), provide supporting reference URLs that verify your selections meet all stated requirements.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FlightItineraryExtraction(BaseModel):
    depart_airport_code: Optional[str] = None
    depart_airport_name: Optional[str] = None
    connection_airports: List[str] = Field(default_factory=list)  # list of IATA codes, e.g., ["JFK"]
    jfk_layover_str: Optional[str] = None  # e.g., "2h 30m"
    arrival_airport_code: Optional[str] = None
    arrival_airport_name: Optional[str] = None
    flight_numbers: List[str] = Field(default_factory=list)  # e.g., ["DL123", "B61234"]
    airlines: List[str] = Field(default_factory=list)        # e.g., ["Delta Air Lines", "JetBlue"]
    flight_urls: List[str] = Field(default_factory=list)     # URLs that support routing/times


class LoungeInfoExtraction(BaseModel):
    lounge_name: Optional[str] = None
    terminal: Optional[str] = None  # Should be "Terminal 4" or "T4"
    location_details: Optional[str] = None  # e.g., "Concourse B near Gate B39"
    lounge_urls: List[str] = Field(default_factory=list)  # Include Priority Pass page or official listings


class HotelInfoExtraction(BaseModel):
    hotel_name: Optional[str] = None
    hotel_address: Optional[str] = None
    distance_to_san_miles: Optional[str] = None  # e.g., "1.8 miles"
    free_shuttle_service_description: Optional[str] = None  # e.g., "Complimentary airport shuttle"
    hotel_urls: List[str] = Field(default_factory=list)  # hotel official page and/or map/evidence links


class ParkInfoExtraction(BaseModel):
    park_name: Optional[str] = None
    park_location: Optional[str] = None  # city/region or address
    day_trip_distance_description: Optional[str] = None  # e.g., "2 hours drive from SAN"
    park_urls: List[str] = Field(default_factory=list)  # NPS or official pages, evidence of pass acceptance


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_flights() -> str:
    return """
Extract the flight routing information from the answer. Focus only on what is explicitly stated in the answer text.

Fields to extract:
- depart_airport_code: IATA code for the departure airport (e.g., "BGR")
- depart_airport_name: Full name of the departure airport
- connection_airports: An array of IATA codes for connection airports in the itinerary (e.g., ["JFK"])
- jfk_layover_str: The stated layover duration at JFK if provided (e.g., "2h 15m"); otherwise null
- arrival_airport_code: IATA code for the final destination airport (e.g., "SAN")
- arrival_airport_name: Full name of the final destination airport
- flight_numbers: An array of any flight numbers mentioned (e.g., ["DL123", "B61234"])
- airlines: An array of airline names mentioned
- flight_urls: An array of URLs provided in the answer that support the flight routing or schedule (e.g., airline pages, booking pages, aggregators)

Rules:
- Do not infer or add airports or codes not explicitly stated.
- If a field is missing in the answer, set it to null (for single fields) or an empty array (for arrays).
- For URLs, extract only actual URLs that appear in the answer text (including markdown links).
"""


def prompt_extract_lounge() -> str:
    return """
Extract the JFK lounge information from the answer as presented.

Fields to extract:
- lounge_name: The lounge name identified
- terminal: The terminal designation (e.g., "Terminal 4", "T4")
- location_details: The specific location within Terminal 4 as described in the answer (e.g., near a certain gate)
- lounge_urls: An array of URLs that support lounge access and its location; include Priority Pass page if present

Rules:
- Only extract what is explicitly mentioned in the answer.
- If an item is missing, return null (or empty array for lounge_urls).
"""


def prompt_extract_hotel() -> str:
    return """
Extract the San Diego accommodation information as stated in the answer.

Fields to extract:
- hotel_name: The hotel name
- hotel_address: The hotel address
- distance_to_san_miles: A distance string indicating proximity to SAN (e.g., "1.6 miles"), if provided
- free_shuttle_service_description: The exact phrase or description in the answer indicating free airport shuttle, if provided
- hotel_urls: An array of URLs that support the hotel's location and/or shuttle service

Rules:
- Only extract information explicitly given in the answer.
- If any field is missing, set to null (or empty array for hotel_urls).
"""


def prompt_extract_park() -> str:
    return """
Extract the national park visit information from the answer as stated.

Fields to extract:
- park_name: The chosen national park's name
- park_location: The park's location (city/region/state or specific area)
- day_trip_distance_description: Any description indicating the travel time/distance from San Diego (e.g., "about 2 hours drive"), if provided
- park_urls: An array of URLs that support the park identification and/or acceptance of the America the Beautiful Pass

Rules:
- Extract only what is explicitly present in the answer.
- If a field is missing, set it to null (or empty array for park_urls).
"""

# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_flight_routing(evaluator: Evaluator, parent_node, flights: FlightItineraryExtraction) -> None:
    # Parent node for flight routing (critical)
    flight_node = evaluator.add_parallel(
        id="Flight_Routing",
        desc="Flight itinerary must depart from Bangor International Airport (BGR), connect through JFK Airport with adequate connection time, and arrive at San Diego International Airport (SAN).",
        parent=parent_node,
        critical=True
    )

    # Gate: ensure some routing details and at least one supporting URL exist
    has_basic_details = bool(
        (flights.depart_airport_code or flights.depart_airport_name or flights.arrival_airport_code or flights.arrival_airport_name or flights.connection_airports)
    )
    has_flight_urls = bool(flights.flight_urls)
    evaluator.add_custom_node(
        result=has_basic_details and has_flight_urls,
        id="Flight_Details_Provided",
        desc="Specific flight numbers, airlines, or routing details are provided with supporting reference URLs.",
        parent=flight_node,
        critical=True  # Adjusted to satisfy critical parent consistency and gate other checks
    )

    # Leaf: Departure airport is BGR
    dep_leaf = evaluator.add_leaf(
        id="Departure_Airport",
        desc="The departure airport must be Bangor International Airport (BGR).",
        parent=flight_node,
        critical=True
    )
    dep_claim = "The itinerary departs from Bangor International Airport (BGR)."
    await evaluator.verify(
        claim=dep_claim,
        node=dep_leaf,
        sources=flights.flight_urls,
        additional_instruction="Verify that the departure airport for the trip is BGR (Bangor International Airport)."
    )

    # Leaf: Connection at JFK
    conn_leaf = evaluator.add_leaf(
        id="Connection_Airport",
        desc="The itinerary must include a connection through John F. Kennedy International Airport (JFK) in New York.",
        parent=flight_node,
        critical=True
    )
    conn_claim = "The itinerary includes a connection at John F. Kennedy International Airport (JFK) in New York."
    await evaluator.verify(
        claim=conn_claim,
        node=conn_leaf,
        sources=flights.flight_urls,
        additional_instruction="Confirm that one of the connections is at JFK (IATA: JFK)."
    )

    # Leaf: Connection time at least 2 hours
    conn_time_leaf = evaluator.add_leaf(
        id="Connection_Time",
        desc="The connection time at JFK must be at least 2 hours to allow for international to domestic transfer including customs and immigration.",
        parent=flight_node,
        critical=True
    )
    layover_note = f"The stated JFK layover is {flights.jfk_layover_str}." if flights.jfk_layover_str else "No layover duration string was extracted from the answer."
    conn_time_claim = f"The layover at JFK is at least 2 hours. {layover_note}"
    await evaluator.verify(
        claim=conn_time_claim,
        node=conn_time_leaf,
        sources=flights.flight_urls,
        additional_instruction=(
            "Use the cited flight pages or itinerary page to check the scheduled arrival time of the inbound flight to JFK and the departure time of the onward flight from JFK. "
            "Confirm that the connection time at JFK is >= 2 hours (120 minutes). If the answer states a layover duration, you may use it if the cited page supports it."
        )
    )

    # Leaf: Arrival airport is SAN
    arr_leaf = evaluator.add_leaf(
        id="Arrival_Airport",
        desc="The final destination must be San Diego International Airport (SAN).",
        parent=flight_node,
        critical=True
    )
    arr_claim = "The final destination of the itinerary is San Diego International Airport (SAN)."
    await evaluator.verify(
        claim=arr_claim,
        node=arr_leaf,
        sources=flights.flight_urls,
        additional_instruction="Verify that the final arrival airport is SAN (San Diego International Airport)."
    )


async def verify_jfk_lounge(evaluator: Evaluator, parent_node, lounge: LoungeInfoExtraction) -> None:
    lounge_node = evaluator.add_parallel(
        id="JFK_Lounge_Access",
        desc="During the layover at JFK, the traveler must access a Priority Pass lounge located in Terminal 4.",
        parent=parent_node,
        critical=True
    )

    # Leaf 1: Verify lounge is in Terminal 4 and accessible via Priority Pass (URL-grounded)
    pp_t4_leaf = evaluator.add_leaf(
        id="Priority_Pass_Lounge_Terminal_4",
        desc="Identifies a specific Priority Pass accessible lounge located in JFK Terminal 4.",
        parent=lounge_node,
        critical=True
    )
    lounge_name_txt = lounge.lounge_name or "the lounge"
    term_txt = lounge.terminal or "Terminal 4"
    claim_pp_t4 = f"{lounge_name_txt} is located in {term_txt} at JFK and is accessible via Priority Pass."
    await evaluator.verify(
        claim=claim_pp_t4,
        node=pp_t4_leaf,
        sources=lounge.lounge_urls,
        additional_instruction="Confirm both: (1) the lounge is in Terminal 4 at JFK; (2) it is accessible via Priority Pass membership (including any usage conditions)."
    )

    # Leaf 2: Verify details (name and specific location) are provided and referenced (existence check)
    details_exist = bool(lounge.lounge_name and lounge.location_details and lounge.lounge_urls)
    evaluator.add_custom_node(
        result=details_exist,
        id="Lounge_Details_And_Reference",
        desc="Provides the lounge name, location details within Terminal 4, and a supporting reference URL confirming it is accessible via Priority Pass.",
        parent=lounge_node,
        critical=True
    )


async def verify_san_diego_hotel(evaluator: Evaluator, parent_node, hotel: HotelInfoExtraction) -> None:
    hotel_node = evaluator.add_parallel(
        id="San_Diego_Accommodation",
        desc="Hotel accommodation near San Diego Airport that is within 2 miles and offers free shuttle service to/from the airport.",
        parent=parent_node,
        critical=True
    )

    # Gate: hotel identification present (name + address)
    hotel_identified = bool(hotel.hotel_name and hotel.hotel_address)
    evaluator.add_custom_node(
        result=hotel_identified,
        id="Hotel_Identification",
        desc="Provides the specific hotel name and address.",
        parent=hotel_node,
        critical=True
    )

    # Leaf: distance within 2 miles (URL-grounded)
    dist_leaf = evaluator.add_leaf(
        id="Hotel_Distance",
        desc="The hotel must be located within 2 miles of San Diego International Airport (SAN).",
        parent=hotel_node,
        critical=True
    )
    dist_claim = f"The hotel '{hotel.hotel_name or 'the selected hotel'}' is within 2 miles of San Diego International Airport (SAN)."
    await evaluator.verify(
        claim=dist_claim,
        node=dist_leaf,
        sources=hotel.hotel_urls,
        additional_instruction=(
            "Verify that the hotel is within 2.0 miles of SAN. Accept straight-line or driving distance if explicitly stated. "
            "If a distance value is shown (e.g., '1.8 miles'), that counts. SAN address reference for context: 3225 N Harbor Dr, San Diego, CA 92101."
        )
    )

    # Leaf: free shuttle (URL-grounded)
    shuttle_leaf = evaluator.add_leaf(
        id="Free_Shuttle_Service",
        desc="The hotel must offer free shuttle service to and from San Diego International Airport.",
        parent=hotel_node,
        critical=True
    )
    shuttle_claim = f"The hotel '{hotel.hotel_name or 'the selected hotel'}' offers free airport shuttle service to and from SAN."
    await evaluator.verify(
        claim=shuttle_claim,
        node=shuttle_leaf,
        sources=hotel.hotel_urls,
        additional_instruction="Look for 'complimentary airport shuttle', 'free airport shuttle', or equivalent wording on the hotel's official page or cited references."
    )

    # Leaf: reference URL confirms both distance and shuttle (URL-grounded)
    ref_leaf = evaluator.add_leaf(
        id="Hotel_Reference_URL",
        desc="Provides a reference URL confirming the hotel's location, distance from airport, and free shuttle service availability.",
        parent=hotel_node,
        critical=True
    )
    ref_claim = (
        f"The provided reference page(s) confirm that '{hotel.hotel_name or 'the selected hotel'}' is within 2 miles of SAN and offers free airport shuttle service."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=hotel.hotel_urls,
        additional_instruction="If multiple URLs are given, any single official or credible page confirming both distance/proximity to SAN and free airport shuttle is sufficient."
    )


async def verify_national_park(evaluator: Evaluator, parent_node, park: ParkInfoExtraction) -> None:
    park_node = evaluator.add_parallel(
        id="National_Park_Visit",
        desc="Identifies a national park within day-trip distance from San Diego that accepts the America the Beautiful Pass for entrance.",
        parent=parent_node,
        critical=True
    )

    # Gate: park identification present (name + location)
    park_identified = bool(park.park_name and park.park_location)
    evaluator.add_custom_node(
        result=park_identified,
        id="Park_Identification",
        desc="Provides the specific national park name and location.",
        parent=park_node,
        critical=True
    )

    # Leaf: within day-trip distance (URL-grounded)
    distance_leaf = evaluator.add_leaf(
        id="Park_Within_Day_Trip_Distance",
        desc="The national park must be located within reasonable day-trip distance from San Diego (generally within San Diego County or adjacent areas accessible within a few hours' drive).",
        parent=park_node,
        critical=True
    )
    distance_claim = (
        f"The national park '{park.park_name or 'the selected park'}' is within reasonable day-trip distance from San Diego (approximately ≤ 3 hours one-way by car)."
    )
    await evaluator.verify(
        claim=distance_claim,
        node=distance_leaf,
        sources=park.park_urls,
        additional_instruction=(
            "Confirm that the park can be visited as a day trip from San Diego (≈ 3 hours or less one-way drive). "
            "Cited pages indicating travel time/distance or commonly accepted day-trip guidance are acceptable."
        )
    )

    # Leaf: accepts America the Beautiful Pass (URL-grounded)
    pass_leaf = evaluator.add_leaf(
        id="Accepts_America_Beautiful_Pass",
        desc="The national park must accept the America the Beautiful Pass for entrance (must be a site managed by National Park Service or other federal agencies that honor the pass).",
        parent=park_node,
        critical=True
    )
    pass_claim = (
        f"The national park '{park.park_name or 'the selected park'}' accepts the America the Beautiful Pass for entrance."
    )
    await evaluator.verify(
        claim=pass_claim,
        node=pass_leaf,
        sources=park.park_urls,
        additional_instruction="Use official NPS or federal site pages when available; confirm that America the Beautiful (Interagency) Pass is accepted."
    )

    # Leaf: reference URL confirms location and pass acceptance (URL-grounded)
    ref_leaf = evaluator.add_leaf(
        id="Park_Reference_URL",
        desc="Provides a reference URL confirming the park's location, entrance fee structure, and acceptance of America the Beautiful Pass.",
        parent=park_node,
        critical=True
    )
    ref_claim = (
        f"The provided reference page(s) confirm the location of '{park.park_name or 'the selected park'}' and that it accepts the America the Beautiful Pass."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=park.park_urls,
        additional_instruction="Any official page clearly confirming the park's location and pass acceptance is sufficient."
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

    # Run extractions (can be parallelized)
    flights_task = evaluator.extract(
        prompt=prompt_extract_flights(),
        template_class=FlightItineraryExtraction,
        extraction_name="flight_itinerary"
    )
    lounge_task = evaluator.extract(
        prompt=prompt_extract_lounge(),
        template_class=LoungeInfoExtraction,
        extraction_name="jfk_lounge"
    )
    hotel_task = evaluator.extract(
        prompt=prompt_extract_hotel(),
        template_class=HotelInfoExtraction,
        extraction_name="san_diego_hotel"
    )
    park_task = evaluator.extract(
        prompt=prompt_extract_park(),
        template_class=ParkInfoExtraction,
        extraction_name="national_park"
    )

    flights, lounge, hotel, park = await asyncio.gather(flights_task, lounge_task, hotel_task, park_task)

    # Build top-level critical node as per rubric (root is non-critical in framework, but child node will be critical)
    complete_plan = evaluator.add_parallel(
        id="Complete_Travel_Plan",
        desc="A complete travel itinerary from Bangor, Maine to San Diego, California via JFK Airport, including flight routing, lounge access during layover, hotel accommodation with shuttle service, and national park visit planning.",
        parent=root,
        critical=True
    )

    # Verify each component under the critical complete plan node
    await verify_flight_routing(evaluator, complete_plan, flights)
    await verify_jfk_lounge(evaluator, complete_plan, lounge)
    await verify_san_diego_hotel(evaluator, complete_plan, hotel)
    await verify_national_park(evaluator, complete_plan, park)

    return evaluator.get_summary()