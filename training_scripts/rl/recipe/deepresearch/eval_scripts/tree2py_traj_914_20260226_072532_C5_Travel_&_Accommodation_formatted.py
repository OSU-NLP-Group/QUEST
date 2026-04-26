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
TASK_ID = "trip_planning_airlines_airport_cruise"
TASK_DESCRIPTION = (
    "You are planning a comprehensive travel itinerary and need to identify specific airlines, an airport, "
    "and a cruise ship that meet precise criteria for your trip.\n\n"
    "Please identify the following four items:\n\n"
    "1. First Airline: Identify the airline that was founded on May 27, 2021, by David Neeleman, operates both "
    "Airbus A220-300 and Embraer 190 aircraft in its fleet, and offers three fare bundle types specifically named "
    "\"Nice,\" \"Nicer,\" and \"Nicest.\"\n\n"
    "2. Second Airline: Identify the airline that is based at Minneapolis-Saint Paul International Airport (MSP) as "
    "its hub and operates an all-Boeing 737-800 fleet for its passenger service operations.\n\n"
    "3. Nashville Airport: Identify the Nashville International Airport by providing its three-letter airport code "
    "(which is BNA) and confirming it has a rooftop lounge facility named \"BNA Sky Pavilion.\"\n\n"
    "4. Disney Cruise Ship: Identify the Disney cruise ship whose maiden voyage departed on November 20, 2025, sails "
    "from Port Everglades in Fort Lauderdale, Florida, and offers both 4-night and 5-night Bahamas cruise itineraries.\n\n"
    "For each identified item, provide the name and at least one reference URL that verifies the information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ItemWithSources(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AirlineOneInfo(ItemWithSources):
    founding_date: Optional[str] = None
    founder: Optional[str] = None
    fleet_aircraft: List[str] = Field(default_factory=list)
    fare_bundles: List[str] = Field(default_factory=list)


class AirlineTwoInfo(ItemWithSources):
    hub: Optional[str] = None
    fleet_types: List[str] = Field(default_factory=list)


class AirportInfo(ItemWithSources):
    code: Optional[str] = None
    facility_names: List[str] = Field(default_factory=list)


class CruiseShipInfo(ItemWithSources):
    maiden_voyage_date: Optional[str] = None
    departure_port: Optional[str] = None
    itineraries: List[str] = Field(default_factory=list)


class TripExtraction(BaseModel):
    airline1: Optional[AirlineOneInfo] = None
    airline2: Optional[AirlineTwoInfo] = None
    airport: Optional[AirportInfo] = None
    cruise_ship: Optional[CruiseShipInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_info() -> str:
    return """
    Extract structured information for four items mentioned in the answer. For each item, return the name identified
    by the answer and all reference URLs cited in the answer that support the information. If an item is not present
    in the answer, return null for that item. If URLs are not provided for an item, return an empty array for sources.

    Items and fields to extract:

    1) airline1 (First Airline: founded 2021 by David Neeleman; operates A220-300 & Embraer 190; fare bundles Nice/Nicer/Nicest)
       - name: The airline's name presented in the answer
       - sources: Array of URLs cited for this airline
       - founding_date: The founding date mentioned in the answer (string or null)
       - founder: Founder name mentioned (string or null)
       - fleet_aircraft: Array of aircraft model names mentioned (e.g., ["Airbus A220-300", "Embraer 190"])
       - fare_bundles: Array of fare bundle names mentioned (e.g., ["Nice","Nicer","Nicest"])

    2) airline2 (Second Airline: MSP hub; all-Boeing 737-800 passenger fleet)
       - name: The airline's name presented in the answer
       - sources: Array of URLs cited for this airline
       - hub: Hub location mentioned (string or null)
       - fleet_types: Array of aircraft types/models mentioned (e.g., ["Boeing 737-800"])

    3) airport (Nashville International Airport: code BNA; rooftop lounge "BNA Sky Pavilion")
       - name: The airport's name presented in the answer (e.g., "Nashville International Airport")
       - sources: Array of URLs cited for this airport
       - code: Airport code mentioned (string or null)
       - facility_names: Array of facility or lounge names mentioned (e.g., ["BNA Sky Pavilion"])

    4) cruise_ship (Disney cruise ship: maiden voyage Nov 20, 2025; sails from Port Everglades; offers 4/5-night Bahamas)
       - name: The ship's name presented in the answer
       - sources: Array of URLs cited for this ship
       - maiden_voyage_date: Date string mentioned (e.g., "November 20, 2025")
       - departure_port: Port name mentioned (e.g., "Port Everglades, Fort Lauderdale, Florida")
       - itineraries: Array of itinerary descriptions mentioned (e.g., ["4-night Bahamas","5-night Bahamas"])

    IMPORTANT:
    - Extract only what is explicitly present in the answer.
    - For sources, include only actual URLs from the answer. If none are provided for an item, return an empty list.
    - If multiple candidates are mentioned, choose the first or primary one for each item.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_item_name(name: Optional[str], fallback: str) -> str:
    return name.strip() if (name and name.strip()) else fallback


def _sources_or_empty(item_sources: Optional[List[str]]) -> List[str]:
    return item_sources if item_sources else []


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_airline_one(
    evaluator: Evaluator,
    parent_node,
    info: Optional[AirlineOneInfo],
) -> None:
    node = evaluator.add_parallel(
        id="airline_founded_2021",
        desc="Identify the airline founded on May 27, 2021, by David Neeleman that operates Airbus A220-300 and Embraer 190 aircraft and offers Nice, Nicer, and Nicest fare bundles",
        parent=parent_node,
        critical=False
    )
    name = _safe_item_name(info.name if info else None, "the airline identified in the answer")
    sources = _sources_or_empty(info.sources if info else None)

    # Founding info
    founding_leaf = evaluator.add_leaf(
        id="founding_info",
        desc="The airline was founded on May 27, 2021, by David Neeleman",
        parent=node,
        critical=True
    )
    # Fleet composition
    fleet_leaf = evaluator.add_leaf(
        id="fleet_composition",
        desc="The airline operates both Airbus A220-300 and Embraer 190 aircraft",
        parent=node,
        critical=True
    )
    # Fare structure
    fare_leaf = evaluator.add_leaf(
        id="fare_structure",
        desc="The airline offers three fare bundle types named Nice, Nicer, and Nicest",
        parent=node,
        critical=True
    )
    # Reference URL existence (custom check)
    evaluator.add_custom_node(
        result=(len(sources) > 0),
        id="reference_url_airline1",
        desc="Provide a reference URL supporting the airline identification",
        parent=node,
        critical=True
    )

    claims = [
        (
            f"{name} was founded on May 27, 2021 by David Neeleman.",
            sources,
            founding_leaf,
            "Verify the page explicitly states the founding date (May 27, 2021) and names David Neeleman as the founder."
        ),
        (
            f"{name} operates both Airbus A220-300 and Embraer 190 aircraft as part of its fleet.",
            sources,
            fleet_leaf,
            "Confirm that the fleet section mentions both Airbus A220-300 and Embraer 190 (or reasonable naming variants) as aircraft operated by the airline."
        ),
        (
            f"{name} offers three fare bundle types named 'Nice', 'Nicer', and 'Nicest'.",
            sources,
            fare_leaf,
            "Confirm the fare bundles are explicitly named Nice, Nicer, and Nicest (allow minor punctuation or case variations)."
        ),
    ]
    await evaluator.batch_verify(claims)


async def verify_airline_two(
    evaluator: Evaluator,
    parent_node,
    info: Optional[AirlineTwoInfo],
) -> None:
    node = evaluator.add_parallel(
        id="airline_msp_hub",
        desc="Identify the airline based at Minneapolis-Saint Paul International Airport (MSP) as its hub that operates an all-Boeing 737-800 passenger fleet",
        parent=parent_node,
        critical=False
    )
    name = _safe_item_name(info.name if info else None, "the airline identified in the answer")
    sources = _sources_or_empty(info.sources if info else None)

    # Hub location
    hub_leaf = evaluator.add_leaf(
        id="hub_location",
        desc="The airline's hub is Minneapolis-Saint Paul International Airport (MSP)",
        parent=node,
        critical=True
    )
    # Fleet type
    fleet_type_leaf = evaluator.add_leaf(
        id="fleet_type",
        desc="The airline operates an all-Boeing 737-800 fleet for passenger service",
        parent=node,
        critical=True
    )
    # Reference URL existence (custom check)
    evaluator.add_custom_node(
        result=(len(sources) > 0),
        id="reference_url_airline2",
        desc="Provide a reference URL supporting the airline identification",
        parent=node,
        critical=True
    )

    claims = [
        (
            f"{name}'s hub is Minneapolis-Saint Paul International Airport (MSP).",
            sources,
            hub_leaf,
            "Verify the page clearly indicates MSP (Minneapolis–Saint Paul International Airport) as the hub for the airline."
        ),
        (
            f"{name} operates an all-Boeing 737-800 fleet for passenger service operations.",
            sources,
            fleet_type_leaf,
            "Verify that for passenger service, the airline exclusively uses Boeing 737-800 aircraft (allow historical or cargo differences to be ignored)."
        ),
    ]
    await evaluator.batch_verify(claims)


async def verify_airport(
    evaluator: Evaluator,
    parent_node,
    info: Optional[AirportInfo],
) -> None:
    node = evaluator.add_parallel(
        id="nashville_airport",
        desc="Identify the airport in Nashville with code BNA that has a rooftop lounge called BNA Sky Pavilion",
        parent=parent_node,
        critical=False
    )
    airport_name = _safe_item_name(info.name if info else None, "Nashville International Airport")
    sources = _sources_or_empty(info.sources if info else None)

    # Airport code
    code_leaf = evaluator.add_leaf(
        id="airport_code",
        desc="The airport code is BNA",
        parent=node,
        critical=True
    )
    # Rooftop lounge
    lounge_leaf = evaluator.add_leaf(
        id="rooftop_lounge",
        desc="The airport has a rooftop lounge named BNA Sky Pavilion",
        parent=node,
        critical=True
    )
    # Reference URL existence (custom check)
    evaluator.add_custom_node(
        result=(len(sources) > 0),
        id="reference_url_airport",
        desc="Provide a reference URL supporting the airport identification",
        parent=node,
        critical=True
    )

    claims = [
        (
            f"{airport_name} has the airport code BNA.",
            sources,
            code_leaf,
            "Confirm that the page explicitly maps Nashville International Airport to the IATA code BNA."
        ),
        (
            f"{airport_name} has a rooftop lounge facility named 'BNA Sky Pavilion'.",
            sources,
            lounge_leaf,
            "Confirm the existence of a rooftop lounge named 'BNA Sky Pavilion' at the airport."
        ),
    ]
    await evaluator.batch_verify(claims)


async def verify_cruise_ship(
    evaluator: Evaluator,
    parent_node,
    info: Optional[CruiseShipInfo],
) -> None:
    node = evaluator.add_parallel(
        id="cruise_ship_2025",
        desc="Identify the Disney cruise ship whose maiden voyage departed on November 20, 2025, sailing from Port Everglades in Fort Lauderdale with 4-night and 5-night Bahamas itineraries",
        parent=parent_node,
        critical=False
    )
    name = _safe_item_name(info.name if info else None, "the Disney cruise ship identified in the answer")
    sources = _sources_or_empty(info.sources if info else None)

    # Maiden voyage date
    maiden_leaf = evaluator.add_leaf(
        id="maiden_voyage_date",
        desc="The cruise ship's maiden voyage departed on November 20, 2025",
        parent=node,
        critical=True
    )
    # Departure port
    port_leaf = evaluator.add_leaf(
        id="departure_port",
        desc="The cruise ship sails from Port Everglades in Fort Lauderdale, Florida",
        parent=node,
        critical=True
    )
    # Itinerary duration
    itinerary_leaf = evaluator.add_leaf(
        id="itinerary_duration",
        desc="The cruise ship offers 4-night and 5-night Bahamas cruise itineraries",
        parent=node,
        critical=True
    )
    # Reference URL existence (custom check)
    evaluator.add_custom_node(
        result=(len(sources) > 0),
        id="reference_url_cruise",
        desc="Provide a reference URL supporting the cruise ship identification",
        parent=node,
        critical=True
    )

    claims = [
        (
            f"{name}'s maiden voyage departed on November 20, 2025.",
            sources,
            maiden_leaf,
            "Verify the page specifies a maiden voyage date of November 20, 2025 for this Disney ship."
        ),
        (
            f"{name} sails from Port Everglades in Fort Lauderdale, Florida.",
            sources,
            port_leaf,
            "Confirm Port Everglades (Fort Lauderdale, FL) is listed as a departure port or homeport for this ship."
        ),
        (
            f"{name} offers Bahamas itineraries that are both 4-night and 5-night in duration.",
            sources,
            itinerary_leaf,
            "Confirm that the itineraries include both 4-night and 5-night Bahamas cruises."
        ),
    ]
    await evaluator.batch_verify(claims)


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
) -> Dict[str, Any]:
    """
    Evaluate the answer for the comprehensive trip planning task covering airlines, the airport, and the Disney cruise ship.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel: items evaluated independently
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_trip_info(),
        template_class=TripExtraction,
        extraction_name="trip_items"
    )

    # Build verification subtrees for each item
    await verify_airline_one(evaluator, root, extracted.airline1 or AirlineOneInfo())
    await verify_airline_two(evaluator, root, extracted.airline2 or AirlineTwoInfo())
    await verify_airport(evaluator, root, extracted.airport or AirportInfo())
    await verify_cruise_ship(evaluator, root, extracted.cruise_ship or CruiseShipInfo())

    # Return final summary
    return evaluator.get_summary()