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
TASK_ID = "breeze_to_marsala_feasibility"
TASK_DESCRIPTION = """
Breeze Airways has announced plans for future transatlantic expansion to Europe. Considering Breeze Airways' current fleet of Airbus A220-300 aircraft and their existing crew base locations, evaluate the feasibility of establishing a route that would allow passengers to travel from a Breeze crew base in the United States to Marsala, Sicily.
Your analysis must:
(1) Verify that the Airbus A220-300 has sufficient range to reach a European airport from a US location,
(2) Identify which Breeze Airways crew base could support this operation based on distance to Europe being within the aircraft's range,
(3) Identify a European airport that could serve as a connection point with available onward flights to the airport nearest to Marsala,
(4) Confirm the distance from that nearest airport to Marsala city center, and
(5) Verify that regular connecting flights exist between your identified European connection point and the airport serving Marsala (Trapani–Birgi, IATA: TPS).
Provide specific airport codes, distances in appropriate units, and reference URLs supporting each component of your analysis.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class AircraftSpec(BaseModel):
    a220_range_nm: Optional[str] = None
    range_source_urls: List[str] = Field(default_factory=list)


class CrewBaseInfo(BaseModel):
    base_city: Optional[str] = None
    base_airport_code: Optional[str] = None  # IATA code (e.g., ORF, TPA)
    crew_base_source_urls: List[str] = Field(default_factory=list)


class BaseDistanceInfo(BaseModel):
    european_airport_name: Optional[str] = None
    european_airport_code: Optional[str] = None  # IATA (e.g., SNN, KEF, DUB, LIS, etc.)
    distance_nm: Optional[str] = None
    distance_source_urls: List[str] = Field(default_factory=list)


class ConnectionAirportInfo(BaseModel):
    connection_airport_name: Optional[str] = None
    connection_airport_code: Optional[str] = None  # IATA
    flight_connection_urls: List[str] = Field(default_factory=list)


class TrapaniAccessInfo(BaseModel):
    nearest_airport_name: Optional[str] = None  # Typically Trapani–Birgi Airport
    nearest_airport_code: Optional[str] = None  # TPS
    distance_km_to_marsala: Optional[str] = None
    distance_source_urls: List[str] = Field(default_factory=list)


class FlightServiceInfo(BaseModel):
    airline_names: List[str] = Field(default_factory=list)
    conn_to_tps_schedule_urls: List[str] = Field(default_factory=list)


class FeasibilityExtraction(BaseModel):
    aircraft: Optional[AircraftSpec] = None
    crew_base: Optional[CrewBaseInfo] = None
    base_distance: Optional[BaseDistanceInfo] = None
    connection: Optional[ConnectionAirportInfo] = None
    trapani_access: Optional[TrapaniAccessInfo] = None
    flight_service: Optional[FlightServiceInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_feasibility() -> str:
    return """
Extract the following fields exactly as stated in the answer. Do not infer or invent values. Use strings for all distances and ranges, include units if present.

Return a JSON object with these keys:
- aircraft:
  - a220_range_nm: The Airbus A220-300 maximum range as stated in the answer (prefer nautical miles; include numeric and unit as shown, e.g., "3450 nm").
  - range_source_urls: An array of URLs cited in the answer that support the A220-300 range figure.
- crew_base:
  - base_city: The Breeze Airways crew base city used for this analysis (e.g., "Norfolk").
  - base_airport_code: The IATA code for that base airport (e.g., "ORF").
  - crew_base_source_urls: An array of URLs cited in the answer that confirm this location is a Breeze crew base.
- base_distance:
  - european_airport_name: The European airport name used for the transatlantic reach analysis (e.g., "Shannon Airport").
  - european_airport_code: The IATA code for that European airport (e.g., "SNN").
  - distance_nm: The great-circle distance from the selected US base airport to this European airport, as stated in the answer (e.g., "2990 nm").
  - distance_source_urls: An array of URLs cited that support this distance (e.g., GCMap, distance calculator).
- connection:
  - connection_airport_name: The European connection airport used to reach Trapani (TPS).
  - connection_airport_code: The IATA code of that airport.
  - flight_connection_urls: An array of URLs cited that show flights from this airport to Trapani (TPS) (e.g., airline route map, booking page, airport destinations page).
- trapani_access:
  - nearest_airport_name: The airport serving Marsala (e.g., "Trapani–Birgi Airport").
  - nearest_airport_code: The IATA code (e.g., "TPS").
  - distance_km_to_marsala: The distance from that airport to Marsala city center, as stated in the answer (e.g., "15 km").
  - distance_source_urls: An array of URLs cited that support this distance (e.g., airport ground transport page, tourism site).
- flight_service:
  - airline_names: Array of airline names that operate the connection between the identified European airport and Trapani (TPS) as stated in the answer (e.g., ["Ryanair"]).
  - conn_to_tps_schedule_urls: An array of URLs cited that confirm the schedule/route operation for flights between the connection airport and Trapani (TPS).

If any field is not present in the answer, set it to null for single values or an empty list for URLs.
""".strip()


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_aircraft_capability(evaluator: Evaluator, parent_node, data: FeasibilityExtraction) -> None:
    """
    Build and verify the 'Aircraft_Capability' subtree.
    """
    spec = data.aircraft or AircraftSpec()
    aircraft_node = evaluator.add_parallel(
        id="Aircraft_Capability",
        desc="Verify the Airbus A220-300 aircraft specifications",
        parent=parent_node,
        critical=True
    )

    # A220_Range_Check: factual claim grounded by URL(s)
    range_leaf = evaluator.add_leaf(
        id="A220_Range_Check",
        desc="Confirm the maximum range of the Airbus A220-300 in nautical miles",
        parent=aircraft_node,
        critical=True
    )
    range_claim = f"The maximum range of the Airbus A220-300 is {spec.a220_range_nm}."
    await evaluator.verify(
        claim=range_claim,
        node=range_leaf,
        sources=spec.range_source_urls,
        additional_instruction=(
            "Verify the maximum range figure for the Airbus A220-300 as shown on the provided source(s). "
            "Minor formatting differences or inclusion of metric conversion are acceptable as long as the range matches."
        )
    )

    # Reference_URL_Aircraft_Range: ensure a range source URL is provided
    evaluator.add_custom_node(
        result=bool(spec.range_source_urls),
        id="Reference_URL_Aircraft_Range",
        desc="Provide a reference URL supporting the A220-300 range specification",
        parent=aircraft_node,
        critical=True
    )


async def verify_crew_base_selection(evaluator: Evaluator, parent_node, data: FeasibilityExtraction) -> None:
    """
    Build and verify the 'Crew_Base_Selection' subtree.
    """
    crew = data.crew_base or CrewBaseInfo()
    dist = data.base_distance or BaseDistanceInfo()

    base_node = evaluator.add_parallel(
        id="Crew_Base_Selection",
        desc="Identify which Breeze Airways crew base has a distance to Europe within the A220-300's operational range",
        parent=parent_node,
        critical=True
    )

    # Base_Distance_Analysis: verify the great-circle distance claim via provided URL(s)
    distance_leaf = evaluator.add_leaf(
        id="Base_Distance_Analysis",
        desc="Determine which Breeze crew base is closest to Europe and verify its distance is within aircraft range",
        parent=base_node,
        critical=True
    )
    distance_claim = (
        f"The great-circle distance from {crew.base_airport_code} to {dist.european_airport_code} is approximately "
        f"{dist.distance_nm}."
    )
    await evaluator.verify(
        claim=distance_claim,
        node=distance_leaf,
        sources=dist.distance_source_urls,
        additional_instruction=(
            "Check that the page provides or implies the great-circle distance between the two airports. "
            "Allow reasonable rounding differences. If multiple values are presented due to routing choices, accept the "
            "great-circle or typical direct distance that aligns with the claimed figure."
        )
    )

    # Reference_URL_Crew_Base: verify the selected location is a Breeze crew base
    crew_base_leaf = evaluator.add_leaf(
        id="Reference_URL_Crew_Base",
        desc="Provide a reference URL confirming this location as a Breeze Airways crew base",
        parent=base_node,
        critical=True
    )
    crew_base_claim = (
        f"{crew.base_city} ({crew.base_airport_code}) is a Breeze Airways crew base."
    )
    await evaluator.verify(
        claim=crew_base_claim,
        node=crew_base_leaf,
        sources=crew.crew_base_source_urls,
        additional_instruction=(
            "Confirm that the provided source explicitly indicates the location is a Breeze Airways crew base. "
            "Accept official Breeze pages, press releases, or credible news articles that clearly state 'crew base'."
        )
    )


async def verify_european_connection_feasibility(evaluator: Evaluator, parent_node, data: FeasibilityExtraction) -> None:
    """
    Build and verify the 'European_Connection_Feasibility' subtree.
    """
    conn = data.connection or ConnectionAirportInfo()
    trap = data.trapani_access or TrapaniAccessInfo()

    euro_node = evaluator.add_sequential(
        id="European_Connection_Feasibility",
        desc="Verify that a viable European connection point exists with access to Marsala",
        parent=parent_node,
        critical=True
    )

    # Connection_Airport_Verification
    conn_airport_node = evaluator.add_parallel(
        id="Connection_Airport_Verification",
        desc="Identify and verify a European airport that can serve as connection point to Trapani",
        parent=euro_node,
        critical=True
    )

    # Connection_Airport_Selection: verify flights to Trapani (TPS) from connection airport
    conn_select_leaf = evaluator.add_leaf(
        id="Connection_Airport_Selection",
        desc="Identify a European airport with flights to Trapani that is within range from the identified US crew base",
        parent=conn_airport_node,
        critical=True
    )
    conn_select_claim = (
        f"The European airport {conn.connection_airport_name} ({conn.connection_airport_code}) offers flights to Trapani (TPS)."
    )
    await evaluator.verify(
        claim=conn_select_claim,
        node=conn_select_leaf,
        sources=conn.flight_connection_urls,
        additional_instruction=(
            "Verify that the provided page(s) show scheduled or regular flights between the connection airport and Trapani (TPS). "
            "Accept airline route maps, airport destinations pages, or booking/schedule listings indicating service."
        )
    )

    # Reference_URL_European_Airport: ensure URLs for the connection flights are present
    evaluator.add_custom_node(
        result=bool(conn.flight_connection_urls),
        id="Reference_URL_European_Airport",
        desc="Provide a reference URL confirming flight connections from this European airport to Trapani",
        parent=conn_airport_node,
        critical=True
    )

    # Trapani_Marsala_Access
    trapani_node = evaluator.add_parallel(
        id="Trapani_Marsala_Access",
        desc="Verify that Trapani Airport provides access to Marsala",
        parent=euro_node,
        critical=True
    )

    # Distance_Verification: verify the distance from Trapani (TPS) to Marsala city center
    distance_leaf = evaluator.add_leaf(
        id="Distance_Verification",
        desc="Confirm the distance from Trapani Airport to Marsala city center in kilometers",
        parent=trapani_node,
        critical=True
    )
    trapani_distance_claim = (
        f"The distance from {trap.nearest_airport_name} ({trap.nearest_airport_code}) to Marsala city center is "
        f"approximately {trap.distance_km_to_marsala}."
    )
    await evaluator.verify(
        claim=trapani_distance_claim,
        node=distance_leaf,
        sources=trap.distance_source_urls,
        additional_instruction=(
            "Confirm that the provided page gives or implies the approximate distance between Trapani–Birgi Airport (TPS) and Marsala. "
            "Allow reasonable rounding differences."
        )
    )

    # Reference_URL_Trapani_Distance: ensure a distance source URL exists
    evaluator.add_custom_node(
        result=bool(trap.distance_source_urls),
        id="Reference_URL_Trapani_Distance",
        desc="Provide a reference URL supporting the Trapani-Marsala distance",
        parent=trapani_node,
        critical=True
    )


async def verify_flight_service(evaluator: Evaluator, parent_node, data: FeasibilityExtraction) -> None:
    """
    Build and verify the 'Flight_Service_Verification' subtree.
    """
    conn = data.connection or ConnectionAirportInfo()
    svc = data.flight_service or FlightServiceInfo()

    flight_node = evaluator.add_parallel(
        id="Flight_Service_Verification",
        desc="Confirm that regular connecting flights exist between the identified European airport and Trapani",
        parent=parent_node,
        critical=True
    )

    # Flight_Service_Confirmation: verify regular flights and airline(s)
    confirm_leaf = evaluator.add_leaf(
        id="Flight_Service_Confirmation",
        desc="Identify the airline(s) operating the connection and verify regular service exists",
        parent=flight_node,
        critical=True
    )
    airline_list_str = ", ".join(svc.airline_names) if svc.airline_names else "the listed airline(s)"
    service_claim = (
        f"Regular scheduled flights are operated by {airline_list_str} between {conn.connection_airport_code} and Trapani (TPS)."
    )
    await evaluator.verify(
        claim=service_claim,
        node=confirm_leaf,
        sources=svc.conn_to_tps_schedule_urls or conn.flight_connection_urls,
        additional_instruction=(
            "Confirm that the page(s) indicate scheduled/regular service (current or seasonal) between the specified airports. "
            "Accept airline/airport route maps or schedules that clearly show the route."
        )
    )

    # Reference_URL_Flight_Schedule: ensure a schedule/route URL exists
    evaluator.add_custom_node(
        result=bool(svc.conn_to_tps_schedule_urls or conn.flight_connection_urls),
        id="Reference_URL_Flight_Schedule",
        desc="Provide a reference URL confirming the flight schedule or route operation",
        parent=flight_node,
        critical=True
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
    Entry point to evaluate an answer for the Breeze-to-Marsala feasibility task.
    Builds the verification tree and returns a structured evaluation summary.
    """
    # Initialize evaluator (root is non-critical by framework design)
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_feasibility(),
        template_class=FeasibilityExtraction,
        extraction_name="feasibility_extraction"
    )

    # Build top-level critical node mirroring the rubric
    feasibility_node = evaluator.add_sequential(
        id="Feasibility_Assessment",
        desc="Evaluate whether Breeze Airways could feasibly operate a route that would allow passengers to reach Marsala, Sicily, considering aircraft capabilities, operational bases, and available connections",
        parent=root,
        critical=True
    )

    # US_Operations_Feasibility (critical + sequential)
    us_ops_node = evaluator.add_sequential(
        id="US_Operations_Feasibility",
        desc="Verify that Breeze Airways has the operational capability to reach Europe from a US crew base",
        parent=feasibility_node,
        critical=True
    )

    # Aircraft capability subtree
    await verify_aircraft_capability(evaluator, us_ops_node, extracted)

    # Crew base selection subtree
    await verify_crew_base_selection(evaluator, us_ops_node, extracted)

    # European connection feasibility (critical + sequential)
    await verify_european_connection_feasibility(evaluator, feasibility_node, extracted)

    # Flight service verification (critical + parallel)
    await verify_flight_service(evaluator, feasibility_node, extracted)

    # Optional: record some custom info for debugging
    evaluator.add_custom_info(
        info={
            "selected_base": {
                "city": (extracted.crew_base.base_city if extracted.crew_base else None),
                "iata": (extracted.crew_base.base_airport_code if extracted.crew_base else None),
            },
            "connection_airport": {
                "name": (extracted.connection.connection_airport_name if extracted.connection else None),
                "iata": (extracted.connection.connection_airport_code if extracted.connection else None),
            },
            "trapani_access": {
                "nearest_airport": (extracted.trapani_access.nearest_airport_name if extracted.trapani_access else None),
                "nearest_iata": (extracted.trapani_access.nearest_airport_code if extracted.trapani_access else None),
                "distance_km_to_marsala": (extracted.trapani_access.distance_km_to_marsala if extracted.trapani_access else None),
            }
        },
        info_type="extracted_summary"
    )

    # Return the aggregated evaluation summary
    return evaluator.get_summary()