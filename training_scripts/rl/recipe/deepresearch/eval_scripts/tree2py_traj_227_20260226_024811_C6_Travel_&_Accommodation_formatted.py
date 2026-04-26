import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dollywood_cruise_2026"
TASK_DESCRIPTION = (
    "A family of six is planning a European cruise vacation in 2026 and needs help coordinating their travel arrangements. "
    "They require:\n\n"
    "1. Pre-trip accommodation: A room at one of Dollywood's on-site resorts in Pigeon Forge, Tennessee, that can "
    "accommodate all 6 family members in a single room. The room must include standard amenities.\n\n"
    "2. Outbound flight: A nonstop flight from Denver International Airport (DEN) to a European city, operated by United "
    "Airlines with daily service. The arrival city must enable a connection to Rotterdam, Netherlands, where their cruise departs.\n\n"
    "3. Cruise selection: A Holland America Line cruise that meets the following criteria:\n"
    "   - The ship must be from Holland America's Pinnacle class\n"
    "   - The cruise must depart from Rotterdam, Netherlands\n"
    "   - Provide the ship's name, passenger capacity, and gross tonnage\n\n"
    "For each component, provide:\n"
    "- The specific name of the resort and room type\n"
    "- The European arrival city for the flight\n"
    "- The name of the cruise ship and its specifications\n\n"
    "Include reference URLs supporting each selection."
)

ALLOWED_ARRIVAL_CITIES = ["Frankfurt", "Munich", "London"]  # City-level allowance list


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AccommodationInfo(BaseModel):
    resort_name: Optional[str] = None
    room_type: Optional[str] = None
    accommodates_six: Optional[bool] = None
    # Amenities mentioned in the answer (free-form, extracted as strings). This is not used for verification directly,
    # but recorded for context. Verification will rely on URL evidence.
    amenities: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class FlightInfo(BaseModel):
    departure_airport: Optional[str] = None
    arrival_city: Optional[str] = None
    carrier: Optional[str] = None
    is_nonstop: Optional[bool] = None
    is_daily_service: Optional[bool] = None
    urls: List[str] = Field(default_factory=list)


class CruiseInfo(BaseModel):
    operator: Optional[str] = None
    ship_name: Optional[str] = None
    pinnacle_class: Optional[bool] = None
    departure_port: Optional[str] = None
    passenger_capacity: Optional[str] = None  # use string for flexibility
    gross_tonnage: Optional[str] = None       # use string for flexibility
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_accommodation() -> str:
    return """
    Extract the pre-trip accommodation details from the answer.

    Required fields:
    - resort_name: The exact name of the Dollywood on-site resort selected (e.g., "Dollywood's DreamMore Resort & Spa" or "Dollywood's HeartSong Lodge & Resort").
    - room_type: The specific room type name (e.g., "Family Suite", "King Bunk Room", "Corner Suite", etc.).
    - accommodates_six: A boolean indicating whether the selected room type is stated (in the answer) to accommodate 6 guests in a single room.
    - amenities: List of amenity phrases mentioned for the selected room (free-form strings; include items like Wi-Fi, flat-screen TV, mini refrigerator, coffee maker).
    - urls: All URLs provided in the answer that support the resort's on-site status and/or the room details/amenities.

    Rules:
    - Only use information explicitly present in the answer text.
    - For any missing field, return null (or empty list for arrays).
    - Extract the URLs exactly as given (including markdown links); ensure they are valid-looking URLs.
    """


def prompt_extract_flight() -> str:
    return """
    Extract the outbound flight details from the answer.

    Required fields:
    - departure_airport: Departure airport code or name (should be DEN / Denver International Airport).
    - arrival_city: The European arrival city name (e.g., "Frankfurt", "Munich", or "London").
    - carrier: The airline operating the flight (should be "United Airlines" or equivalent phrasing).
    - is_nonstop: A boolean indicating the flight is nonstop (no intermediate stops).
    - is_daily_service: A boolean indicating the service is daily.
    - urls: All URLs provided in the answer to support the flight claim(s) (route, schedule, nonstop/daily, carrier).

    Rules:
    - Only use information explicitly present in the answer text.
    - For any missing field, return null (or empty list for arrays).
    - Extract the URLs exactly as given (including markdown links); ensure they are valid-looking URLs.
    """


def prompt_extract_cruise() -> str:
    return """
    Extract the cruise selection details from the answer.

    Required fields:
    - operator: The cruise operator name (should be "Holland America Line").
    - ship_name: The exact cruise ship name.
    - pinnacle_class: A boolean indicating the ship is stated to be in Holland America's Pinnacle class.
    - departure_port: The departure port city (should be Rotterdam, Netherlands).
    - passenger_capacity: The passenger capacity value as stated.
    - gross_tonnage: The gross tonnage value as stated.
    - urls: All URLs provided in the answer to support the cruise/ship facts and its specifications.

    Rules:
    - Only use information explicitly present in the answer text.
    - For any missing field, return null (or empty list for arrays).
    - Extract the URLs exactly as given (including markdown links); ensure they are valid-looking URLs.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_accommodation(evaluator: Evaluator, parent_node, info: AccommodationInfo) -> None:
    """
    Build and verify the Pre_Trip_Accommodation subtree (critical, parallel).
    """
    node = evaluator.add_parallel(
        id="Pre_Trip_Accommodation",
        desc="Select a Dollywood on-site resort room type that fits 6 in one room and includes required standard amenities, with supporting URL(s).",
        parent=parent_node,
        critical=True
    )

    # Existence / support prerequisites (critical siblings)
    evaluator.add_custom_node(
        result=(info.room_type is not None and info.room_type.strip() != ""),
        id="Room_Type_Specified",
        desc="Provides the specific room type name.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(len(info.urls) > 0),
        id="Accommodation_URL_Support",
        desc="Provides at least one reference URL supporting the accommodation selection details (resort/on-site status and/or room details).",
        parent=node,
        critical=True
    )

    # Resort is Dollywood on-site
    resort_leaf = evaluator.add_leaf(
        id="Resort_Is_Dollywood_On_Site",
        desc="Resort is one of Dollywood's two on-site resorts (DreamMore Resort & Spa or HeartSong Lodge & Resort).",
        parent=node,
        critical=True
    )
    resort_name = info.resort_name or ""
    await evaluator.verify(
        claim=f"The selected resort '{resort_name}' is one of Dollywood's two on-site resorts: DreamMore Resort & Spa or HeartSong Lodge & Resort.",
        node=resort_leaf,
        sources=info.urls,
        additional_instruction=(
            "Confirm via the provided URL(s) that the resort is an official Dollywood on-site property. "
            "Accept variations like 'Dollywood's DreamMore Resort' or 'HeartSong Lodge & Resort'."
        )
    )

    # Room accommodates 6 in one room
    accommodates_leaf = evaluator.add_leaf(
        id="Room_Accommodates_6_In_One_Room",
        desc="Selected room type is capable of accommodating 6 guests in a single room.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The room type '{info.room_type or ''}' at '{resort_name}' can accommodate six guests in a single room (not two separate rooms).",
        node=accommodates_leaf,
        sources=info.urls,
        additional_instruction=(
            "Verify the occupancy/capacity for the specific room type. Accept phrases like 'sleeps up to 6' or 'maximum 6 guests'. "
            "Do not count configurations requiring booking multiple rooms."
        )
    )

    # Required amenities verification
    amenities_leaf = evaluator.add_leaf(
        id="Room_Includes_Required_Amenities",
        desc="Room includes the required standard amenities: complimentary Wi‑Fi, flat-screen TV, mini refrigerator, and coffee maker.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The '{info.room_type or ''}' at '{resort_name}' includes complimentary Wi‑Fi, a flat-screen TV, a mini refrigerator, and a coffee maker.",
        node=amenities_leaf,
        sources=info.urls,
        additional_instruction=(
            "Confirm each amenity via the provided URL(s). Accept reasonable synonyms: 'Wi-Fi'/'wireless internet', "
            "'TV'/'smart TV', 'mini fridge'/'refrigerator', 'coffee maker'/'Keurig'. All four must be present."
        )
    )


async def verify_flight(evaluator: Evaluator, parent_node, info: FlightInfo) -> None:
    """
    Build and verify the Outbound_Flight subtree (critical, parallel).
    """
    node = evaluator.add_parallel(
        id="Outbound_Flight",
        desc="Identify a United Airlines daily nonstop flight from DEN to an allowed European arrival city that enables onward travel to Rotterdam, with supporting URL(s).",
        parent=parent_node,
        critical=True
    )

    # Existence / support prerequisite
    evaluator.add_custom_node(
        result=(len(info.urls) > 0),
        id="Flight_URL_Support",
        desc="Provides at least one reference URL supporting the flight claim(s) (nonstop, daily service, United, route).",
        parent=node,
        critical=True
    )

    # Departs from DEN
    dep_leaf = evaluator.add_leaf(
        id="Departs_From_DEN",
        desc="Flight departs from Denver International Airport (DEN).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The flight departs from Denver International Airport (DEN).",
        node=dep_leaf,
        sources=info.urls,
        additional_instruction=(
            "Confirm the origin airport code/name is DEN / Denver International Airport on the referenced page(s)."
        )
    )

    # Operated by United
    united_leaf = evaluator.add_leaf(
        id="Operated_By_United",
        desc="Flight is operated by United Airlines.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The flight is operated by United Airlines.",
        node=united_leaf,
        sources=info.urls,
        additional_instruction=(
            "Confirm the operating carrier is United Airlines; accept 'UA' or 'United' as valid references."
        )
    )

    # Nonstop daily to allowed European city (Frankfurt, Munich, or London)
    nonstop_daily_leaf = evaluator.add_leaf(
        id="Nonstop_Daily_To_Allowed_European_City",
        desc="Flight provides daily nonstop service from DEN to one of the allowed European arrival cities (Frankfurt, Munich, or London).",
        parent=node,
        critical=True
    )
    arrival_city = (info.arrival_city or "").strip()
    await evaluator.verify(
        claim=(
            f"United Airlines offers daily nonstop service from DEN to {arrival_city}, "
            f"and {arrival_city} is one of the allowed cities: Frankfurt, Munich, or London."
        ),
        node=nonstop_daily_leaf,
        sources=info.urls,
        additional_instruction=(
            "Confirm both 'nonstop' and 'daily' service for the DEN→arrival city route on the referenced page(s). "
            "Also ensure the arrival city is one of: Frankfurt, Munich, London. "
            "Accept airport-specific variants such as FRA (Frankfurt), MUC (Munich), LHR (London Heathrow)."
        )
    )

    # Enables connection to Rotterdam (rail or flight)
    connection_leaf = evaluator.add_leaf(
        id="Enables_Connection_To_Rotterdam",
        desc="Arrival city enables a connection to Rotterdam, Netherlands (e.g., via onward flight/rail), consistent with the requirement.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"From {arrival_city}, there is established air or rail service enabling onward travel to Rotterdam, Netherlands."
        ),
        node=connection_leaf,
        sources=info.urls,
        additional_instruction=(
            "Confirm using the provided URL(s) that onward travel to Rotterdam is feasible (e.g., rail services like Eurostar/NS/DB "
            "or connecting flights to RTM). The evidence should explicitly or clearly imply connectivity to Rotterdam."
        )
    )


async def verify_cruise(evaluator: Evaluator, parent_node, info: CruiseInfo) -> None:
    """
    Build and verify the Cruise_Selection subtree (critical, parallel).
    """
    node = evaluator.add_parallel(
        id="Cruise_Selection",
        desc="Select a Holland America Line Pinnacle-class cruise departing from Rotterdam and provide required ship details, with supporting URL(s).",
        parent=parent_node,
        critical=True
    )

    # Existence / support prerequisites (critical siblings)
    evaluator.add_custom_node(
        result=(info.ship_name is not None and info.ship_name.strip() != ""),
        id="Ship_Name_Provided",
        desc="Provides the cruise ship name.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(info.passenger_capacity is not None and info.passenger_capacity.strip() != ""),
        id="Passenger_Capacity_Provided",
        desc="Provides the ship's passenger capacity.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(info.gross_tonnage is not None and info.gross_tonnage.strip() != ""),
        id="Gross_Tonnage_Provided",
        desc="Provides the ship's gross tonnage.",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(len(info.urls) > 0),
        id="Cruise_URL_Support",
        desc="Provides at least one reference URL supporting the cruise/ship selection and its specifications.",
        parent=node,
        critical=True
    )

    # Operated by Holland America Line
    op_leaf = evaluator.add_leaf(
        id="Operated_By_Holland_America",
        desc="Cruise is operated by Holland America Line.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The cruise is operated by Holland America Line.",
        node=op_leaf,
        sources=info.urls,
        additional_instruction="Confirm via the provided URL(s) that the operator is Holland America Line (HAL)."
    )

    # Ship is Pinnacle class
    class_leaf = evaluator.add_leaf(
        id="Ship_Is_Pinnacle_Class",
        desc="Selected ship belongs to Holland America Line's Pinnacle class.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ship '{info.ship_name or ''}' belongs to Holland America Line's Pinnacle class.",
        node=class_leaf,
        sources=info.urls,
        additional_instruction=(
            "Confirm Pinnacle class membership via the provided URL(s). "
            "Accept known Pinnacle class ships if clearly indicated (e.g., Koningsdam, Nieuw Statendam, Rotterdam)."
        )
    )

    # Departs from Rotterdam
    depart_leaf = evaluator.add_leaf(
        id="Departs_From_Rotterdam",
        desc="Cruise departs from Rotterdam, Netherlands.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The selected cruise itinerary departs from Rotterdam, Netherlands.",
        node=depart_leaf,
        sources=info.urls,
        additional_instruction=(
            "Confirm that the cruise/itinerary explicitly lists Rotterdam, Netherlands as the departure port. "
            "Accept synonymous phrasing like 'embarkation: Rotterdam'."
        )
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
) -> Dict[str, Any]:
    """
    Evaluate a single answer for the Dollywood + United + HAL vacation package task.
    """
    # Initialize evaluator and root
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

    # Create critical package node under root
    package_node = evaluator.add_parallel(
        id="Vacation_Package",
        desc="Complete vacation package meeting all specified requirements (lodging, flight, and cruise), each supported by reference URL(s).",
        parent=root,
        critical=True
    )

    # Run extractions concurrently
    accommodation_task = evaluator.extract(
        prompt=prompt_extract_accommodation(),
        template_class=AccommodationInfo,
        extraction_name="accommodation_info"
    )
    flight_task = evaluator.extract(
        prompt=prompt_extract_flight(),
        template_class=FlightInfo,
        extraction_name="flight_info"
    )
    cruise_task = evaluator.extract(
        prompt=prompt_extract_cruise(),
        template_class=CruiseInfo,
        extraction_name="cruise_info"
    )

    accommodation_info, flight_info, cruise_info = await asyncio.gather(
        accommodation_task, flight_task, cruise_task
    )

    # Add custom info for context
    evaluator.add_custom_info(
        info={"allowed_arrival_cities": ALLOWED_ARRIVAL_CITIES},
        info_type="constraints",
        info_name="flight_constraints"
    )

    # Build and verify subtrees
    await verify_accommodation(evaluator, package_node, accommodation_info)
    await verify_flight(evaluator, package_node, flight_info)
    await verify_cruise(evaluator, package_node, cruise_info)

    # Return structured summary
    return evaluator.get_summary()