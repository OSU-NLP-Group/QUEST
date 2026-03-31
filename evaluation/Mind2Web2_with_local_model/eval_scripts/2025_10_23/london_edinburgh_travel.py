import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient

TASK_ID = "london_edinburgh_travel"
TASK_DESCRIPTION = """I am traveling from London to Edinburgh on the 13th, exactly two months from today. Help me compare these three transportation methods:

1. Train: LNER train, direct service from London King's Cross, using a Standard Fixed ticket.
2. Bus: FlixBus, direct service departing from London Victoria Coach Station.
3. Flight: Ryanair direct flight, Basic fare, departing from London Stansted Airport.

For each transportation method, select two departures from the provider's official website—one clearly before noon and one clearly after noon—on the specified date. Clearly provide the departure time, the exact one-way adult fare, and the journey duration.

Please also provide direct links to the official website's search results pages, clearly showing each selected departure in the search results. I want to use these links to directly start the booking process."""

# Number of months ahead for the travel date calculation
MONTHS_AHEAD = 2


# Dynamically calculate travel date
def calculate_travel_date():
    """Dynamically calculate the 13th of N months from now"""
    current_date = datetime.utcnow()
    # Use relativedelta to correctly handle year boundaries
    future_date = current_date + relativedelta(months=MONTHS_AHEAD)
    # Set to the 13th
    travel_date = future_date.replace(day=13, hour=0, minute=0, second=0, microsecond=0)
    return travel_date


TRAVEL_DATE_OBJ = calculate_travel_date()
TRAVEL_DATE = TRAVEL_DATE_OBJ.strftime("%B %d, %Y")
TRAVEL_DATE_FORMATS = list(dict.fromkeys([
    TRAVEL_DATE_OBJ.strftime("%B %d, %Y"),
    TRAVEL_DATE_OBJ.strftime("%d %B %Y"),
    TRAVEL_DATE_OBJ.strftime("%d %b %Y"),
    TRAVEL_DATE_OBJ.strftime("%d/%m/%Y"),
    TRAVEL_DATE_OBJ.strftime("%m/%d/%Y"),
    TRAVEL_DATE_OBJ.strftime("%Y-%m-%d"),
]))
TRAVEL_DATE_FORMATS_TEXT = " or ".join(TRAVEL_DATE_FORMATS)

TASK_DESCRIPTION = f"""
I am traveling from London to Edinburgh on the 13th, exactly two months from today. Help me compare these three transportation methods:

1. Train: LNER train, direct service from London King's Cross, using a Standard Fixed ticket.
2. Bus: FlixBus, direct service departing from London Victoria Coach Station.
3. Flight: Ryanair direct flight, Basic fare, departing from London Stansted Airport.

For each transportation method, select two departures from the provider's official website—one clearly before noon and one clearly after noon—on the specified date. Clearly provide the departure time, the exact one-way adult fare, and the journey duration.

Please also provide direct links to the official website's search results pages, clearly showing each selected departure in the search results. I want to use these links to directly start the booking process.
"""

EVAL_NOTES = ""
GROUND_TRUTH = {}


class SingleDepartureInfo(BaseModel):
    """Information for a single departure (morning or afternoon)"""
    departure_time: Optional[str] = Field(default=None, description="Departure time")
    fare: Optional[str] = Field(default=None, description="One-way adult fare")
    duration: Optional[str] = Field(default=None, description="Journey duration")
    urls: Optional[List[str]] = Field(default_factory=list, description="All URLs for this departure")


def prompt_extract_train_morning() -> str:
    """Extract LNER train morning departure"""
    return f"""
    Extract LNER train morning (before 12:00 noon) departure information from the answer for the journey from London King's Cross to Edinburgh on {TRAVEL_DATE}.

    Please extract:
    - departure_time: The exact departure time (must be before 12:00 noon)
    - fare: The exact one-way adult Standard Fixed fare with currency symbol
    - duration: Journey duration
    - urls: All relevant webpage links (including search results page links)

    Extract information exactly as it appears in the text. If any information is missing, set it to null.
    """


def prompt_extract_train_afternoon() -> str:
    """Extract LNER train afternoon departure"""
    return f"""
    Extract LNER train afternoon (after 12:00 noon) departure information from the answer for the journey from London King's Cross to Edinburgh on {TRAVEL_DATE}.

    Please extract:
    - departure_time: The exact departure time (must be after 12:00 noon)
    - fare: The exact one-way adult Standard Fixed fare with currency symbol
    - duration: Journey duration
    - urls: All relevant webpage links (including search results page links)

    Extract information exactly as it appears in the text. If any information is missing, set it to null.
    """


def prompt_extract_bus_morning() -> str:
    """Extract FlixBus morning departure"""
    return f"""
    Extract FlixBus morning (before 12:00 noon) departure information from the answer for the journey from London Victoria Coach Station to Edinburgh on {TRAVEL_DATE}.

    Please extract:
    - departure_time: The exact departure time (must be before 12:00 noon)
    - fare: The exact one-way adult fare with currency symbol
    - duration: Journey duration
    - urls: All relevant webpage links (including search results page links)

    Extract information exactly as it appears in the text. If any information is missing, set it to null.
    """


def prompt_extract_bus_afternoon() -> str:
    """Extract FlixBus afternoon departure"""
    return f"""
    Extract FlixBus afternoon (after 12:00 noon) departure information from the answer for the journey from London Victoria Coach Station to Edinburgh on {TRAVEL_DATE}.

    Please extract:
    - departure_time: The exact departure time (must be after 12:00 noon)
    - fare: The exact one-way adult fare with currency symbol
    - duration: Journey duration
    - urls: All relevant webpage links (including search results page links)

    Extract information exactly as it appears in the text. If any information is missing, set it to null.
    """


def prompt_extract_flight_morning() -> str:
    """Extract Ryanair flight morning departure"""
    return f"""
    Extract Ryanair flight morning (before 12:00 noon) departure information from the answer for the journey from London Stansted Airport to Edinburgh on {TRAVEL_DATE}.

    Please extract:
    - departure_time: The exact departure time (must be before 12:00 noon)
    - fare: The exact one-way adult Basic fare with currency symbol
    - duration: Flight duration
    - urls: All relevant webpage links (including search results page links)

    Extract information exactly as it appears in the text. If any information is missing, set it to null.
    """


def prompt_extract_flight_afternoon() -> str:
    """Extract Ryanair flight afternoon departure"""
    return f"""
    Extract Ryanair flight afternoon (after 12:00 noon) departure information from the answer for the journey from London Stansted Airport to Edinburgh on {TRAVEL_DATE}.

    Please extract:
    - departure_time: The exact departure time (must be after 12:00 noon)
    - fare: The exact one-way adult Basic fare with currency symbol
    - duration: Flight duration
    - urls: All relevant webpage links (including search results page links)

    Extract information exactly as it appears in the text. If any information is missing, set it to null.
    """


async def verify_search_results_and_departure(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        departure_info: SingleDepartureInfo,
        transport_type: str,
        provider_name: str,
        departure_location: str,
        destination: str = "Edinburgh"
) -> None:
    """Verify that the webpage is a search results page and the claimed departure appears in it"""

    # Create a node for search results verification
    search_results_node = evaluator.add_leaf(
        id=f"{parent_node.id}_search_results",
        desc="Webpage shows search results with the claimed departure",
        parent=parent_node,
        critical=True
    )

    # Construct the claim based on transport type
    if transport_type == "train":
        claim = (
            f"The webpage is a search results page showing {provider_name} train services "
            f"from {departure_location} to {destination} on {TRAVEL_DATE}, "
            f"and includes a departure at {departure_info.departure_time} "
        )
    elif transport_type == "bus":
        claim = (
            f"The webpage is a search results page showing {provider_name} bus services "
            f"from {departure_location} to {destination} on {TRAVEL_DATE}, "
            f"and includes a departure at {departure_info.departure_time} "
        )
    elif transport_type == "flight":
        claim = (
            f"The webpage is a search results page showing {provider_name} flights "
            f"from {departure_location} to {destination} on {TRAVEL_DATE}, "
            f"and includes a departure at {departure_info.departure_time} "
        )

    await evaluator.verify(
        claim=claim,
        node=search_results_node,
        sources=departure_info.urls,
        additional_instruction=(
            f"Verify: 1) This is a search results page, "
            f"2) It shows {transport_type} options for {TRAVEL_DATE} (may appear as other formats, such as {TRAVEL_DATE_FORMATS_TEXT}), "
            f"3) The specific departure at {departure_info.departure_time} is visible in the results, "
            f"4) The fare {departure_info.fare} and duration {departure_info.duration} match what's shown"
        )
    )


async def verify_transport_requirements(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        transport_type: str,
        provider_name: str,
        departure_location: str,
        departure_info: SingleDepartureInfo,
        fare_type: Optional[str],
        is_direct: bool,
        time_period: str
) -> None:
    """Verify transport-specific requirements (without URL verification since provenance was already verified)"""

    if transport_type == "train":
        # Verify LNER provider
        provider_node = evaluator.add_leaf(
            id=f"{parent_node.id}_lner_provider",
            desc="Service is provided by LNER",
            parent=parent_node,
            critical=True
        )

        await evaluator.verify(
            claim=f"The departure at {departure_info.departure_time} is an LNER (London North Eastern Railway) train service",
            node=provider_node,
            sources=departure_info.urls
        )

        # Verify King's Cross departure
        station_node = evaluator.add_leaf(
            id=f"{parent_node.id}_kings_cross",
            desc="Departs from London King's Cross station",
            parent=parent_node,
            critical=True
        )

        await evaluator.verify(
            claim=f"The train at {departure_info.departure_time} departs from London King's Cross station",
            node=station_node,
            sources=departure_info.urls
        )

        # Verify direct service
        direct_node = evaluator.add_leaf(
            id=f"{parent_node.id}_direct_train",
            desc="Direct train service without changes",
            parent=parent_node,
            critical=True
        )

        await evaluator.verify(
            claim=f"The train at {departure_info.departure_time} is a direct train service from London to Edinburgh without changes",
            node=direct_node,
            sources=departure_info.urls
        )

        # Verify Standard Fixed ticket
        ticket_node = evaluator.add_leaf(
            id=f"{parent_node.id}_standard_fixed",
            desc="Standard Fixed ticket type",
            parent=parent_node,
            critical=True
        )

        await evaluator.verify(
            claim=f"The fare of the train at {departure_info.departure_time} is {departure_info.fare}.",
            node=ticket_node,
            sources=departure_info.urls,
            additional_instruction="Verify the ticket fare of the specific departure. If multiple fares are shown (e.g., different classees), compare the fare with the Standard Fixed ticket fare."
        )

        # Verify duration
        duration_node = evaluator.add_leaf(
            id=f"{parent_node.id}_duration",
            desc="Check duration",
            parent=parent_node,
            critical=True
        )

        await evaluator.verify(
            claim=f"The duration of the train at {departure_info.departure_time} is {departure_info.duration}",
            node=duration_node,
            sources=departure_info.urls
        )

    elif transport_type == "bus":
        # Verify FlixBus provider
        provider_node = evaluator.add_leaf(
            id=f"{parent_node.id}_flixbus_provider",
            desc="Service is provided by FlixBus",
            parent=parent_node,
            critical=True
        )

        await evaluator.verify(
            claim=f"The departure at {departure_info.departure_time} is a FlixBus service",
            node=provider_node,
            sources=departure_info.urls
        )

        # Verify Victoria Coach Station departure
        station_node = evaluator.add_leaf(
            id=f"{parent_node.id}_victoria_coach",
            desc="Departs from London Victoria Coach Station",
            parent=parent_node,
            critical=True
        )

        await evaluator.verify(
            claim=f"The bus at {departure_info.departure_time} departs from London Victoria Coach Station",
            node=station_node,
            sources=departure_info.urls
        )

        # Verify direct service
        direct_node = evaluator.add_leaf(
            id=f"{parent_node.id}_direct_bus",
            desc="Direct bus service without changes",
            parent=parent_node,
            critical=True
        )

        await evaluator.verify(
            claim=f"The bus at {departure_info.departure_time} is a direct bus service from London to Edinburgh without stops requiring transfer",
            node=direct_node,
            sources=departure_info.urls
        )

        # Verify fare
        fare_node = evaluator.add_leaf(
            id=f"{parent_node.id}_fare",
            desc="Fare check",
            parent=parent_node,
            critical=True
        )

        await evaluator.verify(
            claim=f"The fare {departure_info.fare} is for the bus at {departure_info.departure_time}",
            node=fare_node,
            sources=departure_info.urls
        )

        # Verify duration
        duration_node = evaluator.add_leaf(
            id=f"{parent_node.id}_duration",
            desc="Check duration",
            parent=parent_node,
            critical=True
        )

        await evaluator.verify(
            claim=f"The duration of the bus at {departure_info.departure_time} is {departure_info.duration}",
            node=duration_node,
            sources=departure_info.urls
        )

    elif transport_type == "flight":
        # Verify Ryanair provider
        provider_node = evaluator.add_leaf(
            id=f"{parent_node.id}_ryanair_provider",
            desc="Flight is operated by Ryanair",
            parent=parent_node,
            critical=True
        )

        await evaluator.verify(
            claim=f"The departure at {departure_info.departure_time} is a Ryanair flight",
            node=provider_node,
            sources=departure_info.urls
        )

        # Verify Stansted Airport departure
        airport_node = evaluator.add_leaf(
            id=f"{parent_node.id}_stansted_airport",
            desc="Departs from London Stansted Airport",
            parent=parent_node,
            critical=True
        )

        await evaluator.verify(
            claim=f"The flight at {departure_info.departure_time} departs from London Stansted Airport (STN)",
            node=airport_node,
            sources=departure_info.urls
        )

        # Verify direct flight
        direct_node = evaluator.add_leaf(
            id=f"{parent_node.id}_direct_flight",
            desc="Direct flight without stops",
            parent=parent_node,
            critical=True
        )

        await evaluator.verify(
            claim=f"The flight at {departure_info.departure_time} is a direct/non-stop flight from London to Edinburgh",
            node=direct_node,
            sources=departure_info.urls
        )

        # Verify Basic fare
        fare_node = evaluator.add_leaf(
            id=f"{parent_node.id}_basic_fare",
            desc="Basic fare type",
            parent=parent_node,
            critical=True
        )

        await evaluator.verify(
            claim=f"The fare of the flight at {departure_info.departure_time} is {departure_info.fare}.",
            node=fare_node,
            sources=departure_info.urls,
            additional_instruction="Verify the ticket fare of the specific departure. If multiple fares are shown (e.g., different classees), compare the fare with the Basic fare."
        )

        # Verify duration
        duration_node = evaluator.add_leaf(
            id=f"{parent_node.id}_duration",
            desc="Check duration",
            parent=parent_node,
            critical=True
        )

        await evaluator.verify(
            claim=f"The duration of the flight at {departure_info.departure_time} is {departure_info.duration}",
            node=duration_node,
            sources=departure_info.urls
        )

    # Verify time period constraint (morning/afternoon)
    time_node = evaluator.add_leaf(
        id=f"{parent_node.id}_time_constraint",
        desc=f"Departure time is in the {time_period}",
        parent=parent_node,
        critical=True
    )

    time_constraint = "before 12:00 (noon)" if time_period == "morning" else "after 12:00 (noon)"
    await evaluator.verify(
        claim=f"The departure time {departure_info.departure_time} is {time_constraint}",
        node=time_node,
        sources=departure_info.urls
    )


async def verify_single_departure(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        departure_info: SingleDepartureInfo,
        transport_type: str,  # "train", "bus", "flight"
        time_period: str,  # "morning", "afternoon"
        provider_name: str,  # "LNER", "FlixBus", "Ryanair"
        departure_location: str,
        fare_type: Optional[str] = None,
        is_direct: bool = True  # For checking direct service
) -> None:
    """Verify a single departure (morning or afternoon)"""

    # Create departure node
    departure_node = evaluator.add_parallel(
        id=f"{transport_type}_{time_period}",
        desc=f"{provider_name} {time_period} departure",
        parent=parent_node,
        critical=False  # Allow partial scoring
    )

    # Merged existence check for all required information
    all_info_exists = (
            departure_info is not None and
            departure_info.departure_time and departure_info.departure_time.strip() and
            departure_info.fare and departure_info.fare.strip() and
            departure_info.duration and departure_info.duration.strip() and
            departure_info.urls and len(departure_info.urls) > 0
    )

    existence_node = evaluator.add_custom_node(
        result=all_info_exists,
        id=f"{transport_type}_{time_period}_all_exists",
        desc=f"All required information exists (departure time, fare, duration, URLs)",
        parent=departure_node,
        critical=True
    )

    # First verify the webpage is a search results page and contains the claimed departure
    await verify_search_results_and_departure(
        evaluator, departure_node, departure_info,
        transport_type, provider_name, departure_location
    )

    # Then verify transport-specific requirements (without URL since provenance was verified)
    await verify_transport_requirements(
        evaluator, departure_node, transport_type, provider_name,
        departure_location, departure_info, fare_type, is_direct, time_period
    )


async def verify_transport_method(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        transport_type: str,
        provider_name: str,
        departure_location: str,
        morning_info: SingleDepartureInfo,
        afternoon_info: SingleDepartureInfo,
        fare_type: Optional[str] = None,
        is_direct: bool = True
) -> None:
    """Verify one transport method with morning and afternoon departures"""

    # Create transport method node
    method_node = evaluator.add_parallel(
        id=f"{transport_type}_method",
        desc=f"{provider_name} from {departure_location}",
        parent=parent_node,
        critical=False  # Allow partial scoring between methods
    )

    # Verify morning departure
    await verify_single_departure(
        evaluator, method_node, morning_info,
        transport_type, "morning", provider_name,
        departure_location, fare_type, is_direct
    )

    # Verify afternoon departure
    await verify_single_departure(
        evaluator, method_node, afternoon_info,
        transport_type, "afternoon", provider_name,
        departure_location, fare_type, is_direct
    )


async def evaluate_answer(
        client: LLMClient,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict[str, Any]:
    """Main evaluation function for London to Edinburgh travel comparison"""

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

    # Extract train information
    train_morning = await evaluator.extract(
        prompt=prompt_extract_train_morning(),
        template_class=SingleDepartureInfo,
        extraction_name="train_morning"
    )

    train_afternoon = await evaluator.extract(
        prompt=prompt_extract_train_afternoon(),
        template_class=SingleDepartureInfo,
        extraction_name="train_afternoon"
    )

    # Extract bus information
    bus_morning = await evaluator.extract(
        prompt=prompt_extract_bus_morning(),
        template_class=SingleDepartureInfo,
        extraction_name="bus_morning"
    )

    bus_afternoon = await evaluator.extract(
        prompt=prompt_extract_bus_afternoon(),
        template_class=SingleDepartureInfo,
        extraction_name="bus_afternoon"
    )

    # Extract flight information
    flight_morning = await evaluator.extract(
        prompt=prompt_extract_flight_morning(),
        template_class=SingleDepartureInfo,
        extraction_name="flight_morning"
    )

    flight_afternoon = await evaluator.extract(
        prompt=prompt_extract_flight_afternoon(),
        template_class=SingleDepartureInfo,
        extraction_name="flight_afternoon"
    )

    # Verify each transport method
    await verify_transport_method(
        evaluator, root, "train", "LNER",
        "London King's Cross", train_morning, train_afternoon,
        "Standard Fixed", is_direct=True
    )

    await verify_transport_method(
        evaluator, root, "bus", "FlixBus",
        "London Victoria Coach Station", bus_morning, bus_afternoon,
        is_direct=True
    )

    await verify_transport_method(
        evaluator, root, "flight", "Ryanair",
        "London Stansted Airport", flight_morning, flight_afternoon,
        "Basic", is_direct=True
    )

    # Add metadata
    evaluator.add_custom_info({
        "travel_date": TRAVEL_DATE,
        "months_ahead": MONTHS_AHEAD,
        "transport_methods": ["train", "bus", "flight"],
        "departures_per_method": 2,
        "total_departures": 6
    }, "evaluation_metadata")

    return evaluator.get_summary()
