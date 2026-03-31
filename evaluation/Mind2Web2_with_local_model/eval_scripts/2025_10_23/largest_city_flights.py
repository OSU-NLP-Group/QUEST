import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import Evaluator, AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "largest_city_flights"
TASK_DESCRIPTION = """
Find a list of the largest U.S. cities ranked by population, and identify the top 3 cities on that list. 
Then, plan the following round-trip itinerary with non-stop economy flights, all arriving in the afternoon:
1. From the largest city to the second largest on the 13th, two months from now.
2. After a 3-night stay, fly from the second largest to the third largest city.
3. After another 3-night stay, return from the third largest city to the largest city.
For each flight, provide the flight number, scheduled departure and arrival times, and airports.
"""

# Configuration constants
REQUIRED_CITIES_COUNT = 3
MONTHS_AHEAD = 2
DEPARTURE_DAY = 13
NIGHTS_STAY = 3
AFTERNOON_START_TIME = "12:00 PM"
AFTERNOON_END_TIME = "8:59 PM"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                      #
# --------------------------------------------------------------------------- #
class City(BaseModel):
    name: Optional[str] = None
    rank: Optional[int] = None


class CitiesSource(BaseModel):
    source_url: Optional[str] = None
    source_description: Optional[str] = None
    cities: List[City] = Field(default_factory=list)


class Flight(BaseModel):
    departure_city: Optional[str] = None
    arrival_city: Optional[str] = None
    flight_number: Optional[str] = None
    departure_time: Optional[str] = None
    arrival_time: Optional[str] = None
    departure_airport: Optional[str] = None
    arrival_airport: Optional[str] = None
    price: Optional[str] = None
    departure_date: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class FlightItinerary(BaseModel):
    cities_source: Optional[CitiesSource]
    flights: List[Flight] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_cities_source() -> str:
    return f"""
    Extract information about the source used to identify the largest U.S. cities by population, and the top {REQUIRED_CITIES_COUNT} cities identified.

    Extract:
    1. The URL or reference used as the source for the list of largest cities (source_url)
    2. A brief description of the source if mentioned (source_description)
    3. The top {REQUIRED_CITIES_COUNT} largest cities mentioned, with:
       - name: The name of the city
       - rank: Its rank (1, 2, or 3)

    If the answer doesn't explicitly mention a source URL or description, return null for those fields.
    """


def prompt_extract_flight(from_city: str, to_city: str) -> str:
    return f"""
    Extract details about the flight from {from_city} to {to_city} mentioned in the answer.

    Extract the following information:
    - departure_city: The name of the departure city
    - arrival_city: The name of the arrival city
    - flight_number: The flight number (e.g., "AA123")
    - departure_time: The scheduled departure time
    - arrival_time: The scheduled arrival time
    - departure_airport: The departure airport code
    - arrival_airport: The arrival airport code
    - price: The economy class price
    - departure_date: The date of the flight
    - urls: List of URLs referenced when providing information about this specific flight

    If any information is missing, return null for that field.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                           #
# --------------------------------------------------------------------------- #
def calculate_expected_dates():
    """Calculate the expected dates for the three flights based on current date."""
    today = datetime.now()

    # First flight: the 13th day, MONTHS_AHEAD months from now
    first_flight_month = (today.month + MONTHS_AHEAD) % 12
    first_flight_year = today.year + ((today.month + MONTHS_AHEAD) // 12)
    if first_flight_month == 0:  # Handle December + MONTHS_AHEAD months
        first_flight_month = 12
        first_flight_year -= 1

    first_flight = datetime(first_flight_year, first_flight_month, DEPARTURE_DAY)

    # Second flight: after NIGHTS_STAY-night stay
    second_flight = first_flight + timedelta(days=NIGHTS_STAY)

    # Third flight: after another NIGHTS_STAY-night stay
    third_flight = second_flight + timedelta(days=NIGHTS_STAY)

    return [first_flight, second_flight, third_flight]


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_cities_source(
        evaluator: Evaluator,
        parent_node,
        itinerary: FlightItinerary,
) -> bool:
    """
    Verify the source for the list of largest cities and the correctness of the top 3 cities.
    Returns True if cities are valid, False otherwise.
    """
    cities_node = evaluator.add_sequential(
        id="cities_source_verification",
        desc=f"Verification of the source for largest US cities and the top {REQUIRED_CITIES_COUNT} cities identified",
        parent=parent_node,
        critical=True
    )

    # 1. Source existence check
    source_exists = bool(
        itinerary.cities_source and 
        (itinerary.cities_source.source_url or itinerary.cities_source.source_description)
    )
    
    source_node = evaluator.add_custom_node(
        result=source_exists,
        id="cities_source_exists",
        desc="Verification that a source for the list of largest US cities is provided",
        parent=cities_node,
        critical=True
    )

    # 2. Top 3 cities provided check
    cities_provided_node = evaluator.add_leaf(
        id="top_3_cities_provided",
        desc=f"Verification that the top {REQUIRED_CITIES_COUNT} largest US cities are identified",
        parent=cities_node,
        critical=True
    )

    claim = f"The answer identifies the top {REQUIRED_CITIES_COUNT} largest US cities by population."
    cities_provided = await evaluator.verify(
        claim=claim,
        node=cities_provided_node,
    )

    # 3. Cities correctness verification
    cities_correctness_node = evaluator.add_leaf(
        id="cities_match_source",
        desc="Verification that the identified cities correctly match their claimed rank according to the source",
        parent=cities_node,
        critical=True
    )

    sorted_cities = []
    cities_claim = ""
    if cities_provided and itinerary.cities_source and len(itinerary.cities_source.cities) >= REQUIRED_CITIES_COUNT:
        # Sort cities by rank and create claim
        sorted_cities = sorted([c for c in itinerary.cities_source.cities if c.rank and c.name],
                               key=lambda c: c.rank if c.rank else 999)

        if len(sorted_cities) >= REQUIRED_CITIES_COUNT:
            cities_claim = "According to population rankings, the top three largest US cities are: "
            for i, city in enumerate(sorted_cities[:REQUIRED_CITIES_COUNT]):
                cities_claim += f"{i + 1}. {city.name}"
                if i < REQUIRED_CITIES_COUNT - 1:
                    cities_claim += ", "

    cities_correct = await evaluator.verify(
        claim=cities_claim,
        node=cities_correctness_node,
        sources=itinerary.cities_source.source_url,
    )

    return cities_correct


async def verify_flight(
        evaluator: Evaluator,
        parent_node,
        flight: Flight,
        flight_index: int,
        expected_from_city: str,
        expected_to_city: str,
        expected_date: datetime,
) -> None:
    """
    Verify details of a specific flight.
    """
    flight_node = evaluator.add_sequential(
        id=f"flight_{flight_index}_verification",
        desc=f"Verification of flight {flight_index}: from {expected_from_city} to {expected_to_city}",
        parent=parent_node,
        critical=False
    )

    # 1. Check if flight data and URLs exist
    flight_exists = bool(flight and flight.urls and len(flight.urls) > 0)
    existence_node = evaluator.add_custom_node(
        result=flight_exists,
        id=f"flight_{flight_index}_provided",
        desc=f"Verification that flight info is provided for flight {flight_index}",
        parent=flight_node,
        critical=True
    )

    # 2. Verify cities
    city_exists = bool(flight_exists and flight.departure_city and flight.arrival_city)
    city_exists = evaluator.add_custom_node(
        result=city_exists,
        id=f"flight_{flight_index}_city_provided",
        desc=f"Verification that cities are provided for flight {flight_index}",
        parent=flight_node,
        critical=True
    )

    cities_node = evaluator.add_leaf(
        id=f"flight_{flight_index}_cities",
        desc=f"Verification that flight {flight_index} correctly connects {expected_from_city} to {expected_to_city}",
        parent=flight_node,
        critical=True
    )

    claim = f"The city '{flight.departure_city}' matches '{expected_from_city}' and the extracted arrival city '{flight.arrival_city}' matches '{expected_to_city}'."
    await evaluator.verify(
        claim=claim,
        node=cities_node,
        additional_instruction="Since this is to verify the city of the FROM and TO for a flight, allow reasonable variations as long as they are usually considered as the same place in common sense when choosing flights or if one include another."
    )

    # 3. Verify flight details (provided + substantiated)
    details_fields = [
        ("departure_time", "Departure time"),
        ("arrival_time", "Arrival time"),
        ("departure_airport", "Departure airport"),
        ("arrival_airport", "Arrival airport"),
        ("departure_date", "Departure date"),
    ]

    for field, description in details_fields:
        # Detail provided check
        detail_provided = bool(flight_exists and getattr(flight, field, None))
        provided_node = evaluator.add_custom_node(
            result=detail_provided,
            id=f"flight_{flight_index}_{field}_provided",
            desc=f"{description} is provided for flight {flight_index}",
            parent=flight_node,
            critical=True
        )

        # Detail substantiated verification
        substantiated_node = evaluator.add_leaf(
            id=f"flight_{flight_index}_{field}_substantiated",
            desc=f"{description} for flight {flight_index} is substantiated by provided URLs",
            parent=flight_node,
            critical=True
        )

        value = getattr(flight, field, None)
        claim = f"Flight {flight.flight_number} from {flight.departure_city} to {flight.arrival_city} has {description.lower()} of {value}."
        additional_instruction = ""
        if "airport" in field:
            additional_instruction = "If this is to verify the departure or arrival airport, it is possible that the page does not show the exact airport code, but the city name or the abbreviation for the city name instead (e.g., NYC instead of JFK), then, as long as the page shows the same city to the airport name/code provided here, treat it as a correct. In other words, as long as the airport code here corresponds to the correct departure or arrival city, treat it as a correct. However, if there is airport code of the corresponding airport in the webpage and the airport code in the webpage is explicitly different, of course it would still be a failure."
        
        await evaluator.verify(
            claim=claim,
            node=substantiated_node,
            sources=flight.urls,
            additional_instruction=additional_instruction
        )

    # 4. Verify afternoon arrival
    # Verify afternoon timing if time exists
    afternoon_time_node = evaluator.add_leaf(
        id=f"flight_{flight_index}_afternoon_timing",
        desc=f"Verification that arrival time {flight.arrival_time if flight else 'N/A'} is in afternoon",
        parent=flight_node,
        critical=True
    )

    claim = f"The time {flight.arrival_time} is between {AFTERNOON_START_TIME} and {AFTERNOON_END_TIME}."
    await evaluator.verify(
        claim=claim,
        node=afternoon_time_node,
    )

    # Afternoon substantiated verification
    afternoon_substantiated_node = evaluator.add_leaf(
        id=f"flight_{flight_index}_afternoon_arrival_substantiated",
        desc=f"Verification that afternoon arrival for flight {flight_index} is substantiated by provided URLs",
        parent=flight_node,
        critical=True
    )

    claim = f"Flight {flight.flight_number} from {expected_from_city} to {expected_to_city} arrives in the afternoon."
    await evaluator.verify(
        claim=claim,
        node=afternoon_substantiated_node,
        sources=flight.urls,
    )

    # 5. Verify non-stop is substantiated by URLs
    nonstop_substantiated_node = evaluator.add_leaf(
        id=f"flight_{flight_index}_nonstop_substantiated",
        desc=f"Verification that non-stop status for flight {flight_index} is substantiated by provided URLs",
        parent=flight_node,
        critical=True
    )

    claim = f"Flight {flight.flight_number} from {expected_from_city} to {expected_to_city} is a non-stop flight."
    await evaluator.verify(
        claim=claim,
        node=nonstop_substantiated_node,
        sources=flight.urls,
    )

    # 6. Verify correct date
    # Verify date matches expected
    date_verification_node = evaluator.add_leaf(
        id=f"flight_{flight_index}_correct_date",
        desc=f"Verification that flight {flight_index} occurs on the correct date according to task requirements",
        parent=flight_node,
        critical=True
    )

    expected_month = expected_date.strftime("%B")
    expected_day = expected_date.day
    claim = f"The date {flight.departure_date} matches the date {expected_month} {expected_day}."
    await evaluator.verify(
        claim=claim,
        node=date_verification_node,
    )

    # Date substantiated verification
    date_substantiated_node = evaluator.add_leaf(
        id=f"flight_{flight_index}_date_substantiated",
        desc=f"Verification that date for flight {flight_index} is substantiated by provided URLs",
        parent=flight_node,
        critical=True
    )

    expected_month = expected_date.strftime("%B")
    expected_day = expected_date.day
    claim = f"Flight {flight.flight_number} departs on {expected_month} {expected_day}."
    await evaluator.verify(
        claim=claim,
        node=date_substantiated_node,
        sources=flight.urls,
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
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract information about cities source and top 3 cities
    cities_source = await evaluator.extract(
        prompt=prompt_extract_cities_source(),
        template_class=CitiesSource,
        extraction_name="cities_source"
    )

    # Initialize the itinerary with the cities source
    itinerary = FlightItinerary(cities_source=cities_source)

    # Verify cities source first
    cities_valid = await verify_cities_source(evaluator, root, itinerary)

    # Flight verification
    flights_node = evaluator.add_parallel(
        id="flights_verification",
        desc="Verification of the round-trip itinerary with three flights",
        parent=root,
        critical=False
    )

    # Extract and verify flight information if cities are correct
    if cities_valid and cities_source:
        # Sort cities by rank
        sorted_cities = sorted([c for c in cities_source.cities if c.rank and c.name],
                               key=lambda c: c.rank if c.rank else 999)

        if len(sorted_cities) >= REQUIRED_CITIES_COUNT:
            city_names = [city.name for city in sorted_cities[:REQUIRED_CITIES_COUNT]]

            # Calculate expected dates
            expected_dates = calculate_expected_dates()

            # Define the three required flights
            flight_routes = [
                (1, 0, 1, "largest to second largest"),
                (2, 1, 2, "second largest to third largest"),
                (3, 2, 0, "third largest back to largest")
            ]

            # Extract and verify each flight
            for flight_index, (flight_num, from_idx, to_idx, description) in enumerate(flight_routes, 1):
                # Extract this specific flight
                flight = await evaluator.extract(
                    prompt=prompt_extract_flight(
                        city_names[from_idx],
                        city_names[to_idx]
                    ),
                    template_class=Flight,
                    extraction_name=f"flight_{flight_index}"
                )

                # Add to itinerary
                itinerary.flights.append(flight)

                # Verify this flight
                await verify_flight(
                    evaluator,
                    flights_node,
                    flight,
                    flight_index,
                    city_names[from_idx],
                    city_names[to_idx],
                    expected_dates[flight_index - 1],
                )
        else:
            # Pad missing flights to maintain consistent structure
            for flight_index in range(3):
                flight = Flight()
                # Verify this flight
                await verify_flight(
                    evaluator,
                    flights_node,
                    flight,
                    flight_index,
                    None, None, None,
                )

    else:
        # Pad missing flights to maintain consistent structure
        for flight_index in range(3):
            flight = Flight()
            # Verify this flight
            await verify_flight(
                evaluator,
                flights_node,
                flight,
                flight_index,
                None, None, None,
            )

    # Add flight data to evaluator for summary
    evaluator.add_custom_info({
        "itinerary": {
            "cities_source": itinerary.cities_source.dict() if itinerary.cities_source else None,
            "flights": [flight.dict() for flight in itinerary.flights]
        }
    }, "flight_data")

    # Return the summary in the new format
    return evaluator.get_summary()