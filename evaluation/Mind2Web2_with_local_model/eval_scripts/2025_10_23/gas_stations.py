import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient
from mind2web2.api_tools import tool_googlemap

TASK_ID = "gas_stations"
TASK_DESCRIPTION = """
I'm planning a road trip from Atlanta, GA to Los Angeles, CA, and my car can travel approximately 500-ish miles per full tank of gas. At the start of my trip, the fuel tank will be completely full. Your task is to help me identify a series of gas stations along the primary highway route from Atlanta to Los Angeles.

The selected gas stations should meet these criteria:

- The driving distance between each consecutive gas station should be between 300 and 500 miles.
- The first gas station should be within 300-500 miles from Atlanta, and the last one should be within 500 miles of Los Angeles.
- All gas stations should be located close to the main highways to minimize additional driving time.

Please provide a sequential list of gas stations, including:

- The name and address of each gas station

Ensure the entire route remains realistic and efficient without significant detours.
"""

EVAL_NOTES = """
1. Considering the possible small variance in driving distances, though we require the distance between two consecutive gas station should be between 300 and 500 miles, we can relax the requirement to 290 and 510 miles for evaluation.

2. To evaluate whether the gas stations are close to the main highways, we can compare the total distance to the distance without gas stations.
"""

GROUND_TRUTH = {}

# Constants for evaluation
MIN_DISTANCE_MILES = 290  # Relaxed from 300
MAX_DISTANCE_MILES = 510  # Relaxed from 500
ATLANTA_GA = "Atlanta, GA"
LOS_ANGELES_CA = "Los Angeles, CA"
METERS_PER_MILE = 1609.34
MAX_DETOUR_PERCENTAGE = 5  # Maximum 5% detour for highway proximity check


class GasStation(BaseModel):
    """Information about a single gas station"""
    name: Optional[str] = Field(default=None, description="Name of the gas station")
    address: Optional[str] = Field(default=None, description="Full address of the gas station")
    url: Optional[str] = Field(default=None, description="Source URL if provided")


class GasStationList(BaseModel):
    """List of gas stations for the route"""
    stations: List[GasStation] = Field(
        default_factory=list,
        description="Sequential list of gas stations from Atlanta to Los Angeles"
    )
    urls: List[str] = Field(
        default_factory=list,
        description="All URLs mentioned in the answer"
    )


def prompt_extract_gas_stations() -> str:
    """Extraction prompt for getting gas station information from the answer"""
    return """
    Extract the list of gas stations from the answer for a road trip from Atlanta, GA to Los Angeles, CA.

    Look for:
    - stations: A sequential list of gas stations in order from Atlanta to Los Angeles
      - For each station, extract:
        - name: The name of the gas station (e.g., "Shell", "Chevron", "Exxon")
        - address: The full address including street, city, and state
        - url: The source URL if one is provided for this specific station
    - urls: All URLs mentioned anywhere in the answer

    Extract information exactly as it appears in the text.
    The stations should be in sequential order from Atlanta to Los Angeles.
    If any field is not mentioned, set it to null.
    """


async def verify_gas_station_route(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        gas_stations: GasStationList,
        gmaps_tool: tool_googlemap.GoogleMapsTool,
) -> None:
    """Verify the entire gas station route meets all criteria"""

    try:
        direct_distance_meters = await gmaps_tool.calculate_distance(
            ATLANTA_GA,
            LOS_ANGELES_CA,
            mode="driving",
        )
        if isinstance(direct_distance_meters, str):
            direct_distance_miles: Optional[float] = None
            direct_distance_error = direct_distance_meters
        else:
            direct_distance_miles = direct_distance_meters / METERS_PER_MILE
            direct_distance_error = None
    except Exception as exc:
        direct_distance_miles = None
        direct_distance_error = str(exc)
    
    if direct_distance_miles == None:
        direct_distance_miles = 2182 # a default estimated distance
    
    MIN_STATIONS = direct_distance_miles // MAX_DISTANCE_MILES
    MAX_STATIONS = direct_distance_miles // MIN_DISTANCE_MILES

    if direct_distance_miles is not None:
        max_total_route_distance = direct_distance_miles * (1 + MAX_DETOUR_PERCENTAGE / 100)
    else:
        max_total_route_distance = None

    evaluator.add_custom_info(
        {
            "direct_distance_miles": direct_distance_miles,
            "direct_distance_error": direct_distance_error,
        },
        "atl_to_la_direct_distance",
    )

    stations_to_evaluate = list(gas_stations.stations[:MAX_STATIONS])

    # Pad with placeholder entries if there are fewer than the minimum required stations
    if len(stations_to_evaluate) < MIN_STATIONS:
        stations_to_evaluate.extend(
            GasStation(name="empty", address=None, url=None)
            for _ in range(MIN_STATIONS - len(stations_to_evaluate))
        )

    cumulative_distance_miles: Optional[float] = 0.0
    previous_address: Optional[str] = ATLANTA_GA
    total_nodes = len(stations_to_evaluate)

    for idx, station in enumerate(stations_to_evaluate):
        station_label = station.name.strip() if station.name else "Unnamed station"
        station_node = evaluator.add_parallel(
            id=f"station_{idx + 1}",
            desc=f"Station {idx + 1}: {station_label}",
            parent=parent_node,
            critical=False,
        )

        # 1. Address must be provided
        address_present = bool(station.address and station.address.strip())
        evaluator.add_custom_node(
            result=address_present,
            id=f"station_{idx + 1}_address_present",
            desc=f"Station {idx + 1} includes an address",
            parent=station_node,
            critical=True,
        )

        # 2. Distance from previous location (and to destination if last station) within limits
        distance_prev_miles: Optional[float] = None
        prev_distance_error: Optional[str] = None

        if previous_address and station.address:
            try:
                distance_prev_meters = await gmaps_tool.calculate_distance(
                    previous_address,
                    station.address,
                    mode="driving",
                )
                if isinstance(distance_prev_meters, str):
                    prev_distance_error = distance_prev_meters
                else:
                    distance_prev_miles = distance_prev_meters / METERS_PER_MILE
            except Exception as exc:
                prev_distance_error = str(exc)
        else:
            prev_distance_error = "Missing address for previous or current waypoint"

        prev_in_range = (
            distance_prev_miles is not None
            and MIN_DISTANCE_MILES <= distance_prev_miles <= MAX_DISTANCE_MILES
        )

        # Distance from current station to destination (needed for last station and detour check)
        distance_to_end_miles: Optional[float] = None
        end_distance_error: Optional[str] = None

        if station.address:
            try:
                distance_to_end_meters = await gmaps_tool.calculate_distance(
                    station.address,
                    LOS_ANGELES_CA,
                    mode="driving",
                )
                if isinstance(distance_to_end_meters, str):
                    end_distance_error = distance_to_end_meters
                else:
                    distance_to_end_miles = distance_to_end_meters / METERS_PER_MILE
            except Exception as exc:
                end_distance_error = str(exc)
        else:
            end_distance_error = "Missing address for current station"

        is_last_station = idx == total_nodes - 1
        end_in_range = True
        if is_last_station:
            end_in_range = (
                distance_to_end_miles is not None
                and MIN_DISTANCE_MILES <= distance_to_end_miles <= MAX_DISTANCE_MILES
            )

        distance_desc_parts = []
        if distance_prev_miles is not None:
            distance_desc_parts.append(
                f"Previous segment: {distance_prev_miles:.1f} miles (allowed {MIN_DISTANCE_MILES}-{MAX_DISTANCE_MILES})"
            )
        else:
            distance_desc_parts.append(
                f"Previous segment unavailable ({prev_distance_error})"
            )

        if is_last_station:
            if distance_to_end_miles is not None:
                distance_desc_parts.append(
                    f"Station to Los Angeles: {distance_to_end_miles:.1f} miles "
                    f"(allowed {MIN_DISTANCE_MILES}-{MAX_DISTANCE_MILES})"
                )
            else:
                distance_desc_parts.append(
                    f"Station to Los Angeles unavailable ({end_distance_error})"
                )

        evaluator.add_custom_node(
            result=prev_in_range and end_in_range,
            id=f"station_{idx + 1}_distance_check",
            desc="; ".join(distance_desc_parts),
            parent=station_node,
            critical=True,
        )

        # 3. Total path respecting detour allowance
        if cumulative_distance_miles is None or distance_prev_miles is None:
            cumulative_distance_miles = None
        else:
            cumulative_distance_miles += distance_prev_miles

        total_route_miles: Optional[float] = None
        detour_percentage: Optional[float] = None
        route_within_limit = False

        if (
            cumulative_distance_miles is not None
            and distance_to_end_miles is not None
            and direct_distance_miles is not None
            and max_total_route_distance is not None
        ):
            total_route_miles = cumulative_distance_miles + distance_to_end_miles
            detour_percentage = (
                (total_route_miles - direct_distance_miles) / direct_distance_miles
            ) * 100
            route_within_limit = total_route_miles <= max_total_route_distance

        if (
            cumulative_distance_miles is None
            or distance_to_end_miles is None
            or direct_distance_miles is None
        ):
            detour_desc = "Unable to compute total route distance"
        else:
            detour_desc = (
                f"Start→Station distance: {cumulative_distance_miles:.1f} miles; "
                f"Station→Los Angeles: {distance_to_end_miles:.1f} miles; "
                f"Total: {total_route_miles:.1f} miles "
                f"(detour {detour_percentage:.1f}% vs. {MAX_DETOUR_PERCENTAGE}% limit)"
            )

        evaluator.add_custom_node(
            result=route_within_limit,
            id=f"station_{idx + 1}_detour_limit",
            desc=detour_desc,
            parent=station_node,
            critical=True,
        )

        # Update previous address for next iteration
        previous_address = station.address if station.address else None


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                               #
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
    Main evaluation function for gas station route verification.
    """

    # -------- 1. Initialize evaluator ----------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
        agent_name=agent_name,
        answer_name=answer_name,
        # Evaluator creation parameters
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Initialize Google Maps tool
    gmaps_tool = tool_googlemap.GoogleMapsTool()

    # -------- 2. Extract structured information ------------------- #
    gas_stations = await evaluator.extract(
        prompt=prompt_extract_gas_stations(),
        template_class=GasStationList,
        extraction_name="gas_stations_extraction",
    )

    # -------- 3. Build verification tree -------------------------- #
    await verify_gas_station_route(evaluator, root, gas_stations, gmaps_tool)

    # -------- 4. Return evaluation results ------------------------ #
    return evaluator.get_summary()
