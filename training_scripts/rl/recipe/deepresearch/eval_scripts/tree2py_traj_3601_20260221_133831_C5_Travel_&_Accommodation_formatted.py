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
TASK_ID = "spain_eclipse_trip_plan"
TASK_DESCRIPTION = (
    "I am planning a trip to Spain to view the total solar eclipse on August 12, 2026. "
    "I would like to fly from Boston to Barcelona using JetBlue's new transatlantic route that begins service in April 2026. "
    "Please provide a complete trip plan that includes: (1) Round-trip flight information on JetBlue from Boston to Barcelona, "
    "with travel dates that span August 12, 2026 (arriving before the eclipse and departing after); "
    "(2) Hotel accommodation for at least one night in a Spanish city that falls within the path of totality for the August 12, 2026 solar eclipse, "
    "with the stay covering the eclipse date; (3) For the flight: provide the booking URL or reference; "
    "(4) For the hotel: provide the hotel name, the city name, the booking URL, and a reference URL documenting that the city is within the eclipse totality path."
)

ECLIPSE_DATE = "2026-08-12"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FlightInfo(BaseModel):
    """Round-trip flight information."""
    airline_name: Optional[str] = None
    departure_city: Optional[str] = None
    departure_airport_code: Optional[str] = None
    arrival_city: Optional[str] = None
    arrival_airport_code: Optional[str] = None
    outbound_date: Optional[str] = None  # Prefer ISO-like strings; allow any reasonable format
    return_date: Optional[str] = None
    booking_urls: List[str] = Field(default_factory=list)  # Accept multiple booking/reference URLs


class HotelInfo(BaseModel):
    """Hotel accommodation information."""
    hotel_name: Optional[str] = None
    city_name: Optional[str] = None
    check_in_date: Optional[str] = None
    check_out_date: Optional[str] = None
    booking_urls: List[str] = Field(default_factory=list)
    totality_reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_flight_info() -> str:
    return (
        "Extract the round-trip flight details mentioned in the answer for travel between Boston and Barcelona. "
        "We expect JetBlue to be the operating airline. Extract the following fields:\n"
        "1. airline_name: The operating airline name as stated (e.g., 'JetBlue', 'JetBlue Airways').\n"
        "2. departure_city: The departure city name (e.g., 'Boston').\n"
        "3. departure_airport_code: The departure airport code if present (e.g., 'BOS'); return null if not stated.\n"
        "4. arrival_city: The arrival city name (e.g., 'Barcelona').\n"
        "5. arrival_airport_code: The arrival airport code if present (e.g., 'BCN'); return null if not stated.\n"
        "6. outbound_date: The outbound flight date (format as shown in the answer).\n"
        "7. return_date: The return flight date (format as shown in the answer).\n"
        "8. booking_urls: All booking or reference URLs provided for the flight (array). Include any JetBlue, OTA, or airline booking pages. "
        "If none are given, return an empty array.\n"
        "Only extract what is explicitly present in the answer. Do not invent or infer missing details."
    )


def prompt_extract_hotel_info() -> str:
    return (
        "Extract the hotel accommodation information for a stay in Spain that includes the eclipse date of August 12, 2026. "
        "Extract the following fields:\n"
        "1. hotel_name: The name of the hotel.\n"
        "2. city_name: The city where the hotel is located.\n"
        "3. check_in_date: The check-in date.\n"
        "4. check_out_date: The check-out date.\n"
        "5. booking_urls: All booking or reference URLs provided for the hotel (array). If none are given, return an empty array.\n"
        "6. totality_reference_urls: All URLs that document the chosen city is within the path of totality for the August 12, 2026 solar eclipse (array). "
        "If none are provided, return an empty array.\n"
        "Only extract what is explicitly present in the answer. Do not invent or infer missing details."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    """Normalize URL list: ensure a list and filter obvious empties."""
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_round_trip_flight(
    evaluator: Evaluator,
    plan_parent_node,
    flight: FlightInfo
) -> None:
    """
    Build and verify the 'Round_Trip_Flight' subtree with critical, independent checks.
    """
    flight_node = evaluator.add_parallel(
        id="Round_Trip_Flight",
        desc="Round-trip flight booking from Boston to Barcelona on JetBlue that spans the eclipse date",
        parent=plan_parent_node,
        critical=True
    )

    # 1) Booking URL existence (critical)
    flight_booking_urls = _safe_urls(flight.booking_urls)
    evaluator.add_custom_node(
        result=len(flight_booking_urls) > 0,
        id="Booking_URL_Provided",
        desc="A booking URL or reference link for the flights is provided",
        parent=flight_node,
        critical=True
    )

    # 2) Airline must be JetBlue (critical, verify by URLs when available)
    airline_leaf = evaluator.add_leaf(
        id="Airline_is_JetBlue",
        desc="The flight is operated by JetBlue Airways",
        parent=flight_node,
        critical=True
    )
    airline_claim = (
        "The round-trip flights shown on the provided booking/reference page(s) are operated by JetBlue Airways (JetBlue). "
        "Minor naming variations like 'JetBlue' or 'JetBlue Airways' should be considered equivalent."
    )
    await evaluator.verify(
        claim=airline_claim,
        node=airline_leaf,
        sources=flight_booking_urls,
        additional_instruction=(
            "Use the booking/reference page(s) to determine the operating airline. "
            "If codeshare/marketing carrier nuances appear, consider the primary operating carrier as the airline listed."
        )
    )

    # 3) Route must connect Boston and Barcelona (critical)
    route_leaf = evaluator.add_leaf(
        id="Route_Boston_Barcelona",
        desc="The flight route connects Boston (BOS) and Barcelona (BCN)",
        parent=flight_node,
        critical=True
    )
    route_claim = (
        "The itinerary on the provided booking/reference page(s) is a round trip between Boston (BOS) and Barcelona (BCN). "
        "If airport codes are not displayed, the page should clearly indicate 'Boston' to 'Barcelona' and back."
    )
    await evaluator.verify(
        claim=route_claim,
        node=route_leaf,
        sources=flight_booking_urls,
        additional_instruction=(
            "Verify both outbound and return segments connect Boston and Barcelona. "
            "Allow reasonable format differences (city names vs airport codes)."
        )
    )

    # 4) Dates must span the eclipse date (critical)
    dates_leaf = evaluator.add_leaf(
        id="Dates_Span_Eclipse",
        desc="The travel dates span August 12, 2026 (outbound before and return after)",
        parent=flight_node,
        critical=True
    )
    outbound_str = flight.outbound_date or "UNKNOWN"
    return_str = flight.return_date or "UNKNOWN"
    dates_claim = (
        f"The flight itinerary shows an outbound date '{outbound_str}' and a return date '{return_str}', "
        f"and these dates span {ECLIPSE_DATE} (outbound earlier than {ECLIPSE_DATE}, return later than {ECLIPSE_DATE})."
    )
    await evaluator.verify(
        claim=dates_claim,
        node=dates_leaf,
        sources=flight_booking_urls,
        additional_instruction=(
            "Confirm the itinerary dates on the provided page(s). "
            "Treat date formatting differences leniently. "
            "The requirement is that arrival is before the eclipse date and departure is after."
        )
    )


async def verify_hotel_accommodation(
    evaluator: Evaluator,
    plan_parent_node,
    hotel: HotelInfo
) -> None:
    """
    Build and verify the 'Hotel_Accommodation' subtree with critical checks.
    """
    hotel_node = evaluator.add_parallel(
        id="Hotel_Accommodation",
        desc="Hotel accommodation in a Spanish city within the eclipse totality path for dates including August 12, 2026",
        parent=plan_parent_node,
        critical=True
    )

    hotel_booking_urls = _safe_urls(hotel.booking_urls)
    totality_urls = _safe_urls(hotel.totality_reference_urls)

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=(hotel.hotel_name is not None and hotel.hotel_name.strip() != ""),
        id="Hotel_Name_Provided",
        desc="The name of the hotel is provided",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(hotel_booking_urls) > 0,
        id="Hotel_Booking_URL",
        desc="A booking URL or reference link for the hotel is provided",
        parent=hotel_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(totality_urls) > 0,
        id="Totality_Reference_URL",
        desc="A URL reference documenting that the chosen city is in the totality path is provided",
        parent=hotel_node,
        critical=True
    )

    # Located in a Spanish city (critical)
    located_leaf = evaluator.add_leaf(
        id="Located_in_Spanish_City",
        desc="The hotel is located in a city in Spain",
        parent=hotel_node,
        critical=True
    )
    city_str = hotel.city_name or "UNKNOWN CITY"
    hotel_name_str = hotel.hotel_name or "UNKNOWN HOTEL"
    located_claim = (
        f"The hotel '{hotel_name_str}' is located in {city_str}, Spain, as shown on the provided booking/reference page(s)."
    )
    await evaluator.verify(
        claim=located_claim,
        node=located_leaf,
        sources=hotel_booking_urls,
        additional_instruction=(
            "Use the hotel booking/reference page(s) to confirm the city and country. "
            "Allow minor naming variations (e.g., language/local spellings)."
        )
    )

    # City in totality path (critical)
    totality_leaf = evaluator.add_leaf(
        id="City_in_Totality_Path",
        desc="The city where the hotel is located falls within the path of totality for the August 12, 2026 solar eclipse",
        parent=hotel_node,
        critical=True
    )
    totality_claim = (
        f"The city {city_str} is within the path of totality for the August 12, 2026 solar eclipse in Spain."
    )
    await evaluator.verify(
        claim=totality_claim,
        node=totality_leaf,
        sources=totality_urls,
        additional_instruction=(
            "Use eclipse path maps or credible references to confirm that the city is inside the totality track on 2026-08-12."
        )
    )

    # Stay includes eclipse date (critical)
    stay_leaf = evaluator.add_leaf(
        id="Stay_Includes_Eclipse_Date",
        desc="The hotel stay dates include August 12, 2026",
        parent=hotel_node,
        critical=True
    )
    check_in_str = hotel.check_in_date or "UNKNOWN"
    check_out_str = hotel.check_out_date or "UNKNOWN"
    stay_claim = (
        f"The hotel stay from '{check_in_str}' to '{check_out_str}' includes {ECLIPSE_DATE}."
    )
    await evaluator.verify(
        claim=stay_claim,
        node=stay_leaf,
        sources=hotel_booking_urls,
        additional_instruction=(
            "Confirm the stay window on the hotel booking/reference page(s). "
            "Treat date formatting differences leniently; the requirement is that the stay covers the day of 2026-08-12."
        )
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
    Evaluate an agent's trip plan answer for the Spain eclipse trip.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root wrapper; actual plan node will be critical
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

    # Extract flight and hotel info
    flight_info, hotel_info = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_flight_info(),
            template_class=FlightInfo,
            extraction_name="flight_info"
        ),
        evaluator.extract(
            prompt=prompt_extract_hotel_info(),
            template_class=HotelInfo,
            extraction_name="hotel_info"
        )
    )

    # Add a critical plan node mirroring the rubric's top-level requirement
    plan_node = evaluator.add_parallel(
        id="Spain_Eclipse_Trip_Plan",
        desc="Complete trip plan for viewing the August 12, 2026 solar eclipse in Spain, including round-trip flights from Boston and hotel accommodation in a city within the totality path",
        parent=root,
        critical=True
    )

    # Build and verify subtrees
    await verify_round_trip_flight(evaluator, plan_node, flight_info)
    await verify_hotel_accommodation(evaluator, plan_node, hotel_info)

    # Add custom info for context
    evaluator.add_custom_info(
        info={
            "eclipse_date": ECLIPSE_DATE,
            "notes": "All critical checks must pass for the plan to be considered valid."
        },
        info_type="context",
        info_name="task_context"
    )

    return evaluator.get_summary()