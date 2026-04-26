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
TASK_ID = "bangor_flight_vs_boston_cruise"
TASK_DESCRIPTION = """A family from Bangor, Maine is planning a week-long vacation and comparing two types of trips: flying to Florida versus driving to Boston to board a cruise ship.

For the Florida flight option, identify two different airlines that offer non-stop flights from Bangor International Airport to Florida destinations. The two airlines must be distinct from each other, and they must serve different Florida airports. For each of the two airline routes, provide:
1. The airline's name
2. The destination airport's complete official name and three-letter IATA code
3. Whether the service operates year-round or is seasonal only
4. A URL from an official airline or airport website that documents this specific route

For the Boston cruise option, provide:
5. The driving distance in miles from Bangor, Maine to Boston's Black Falcon Cruise Terminal
6. The approximate driving time in hours and minutes under normal traffic conditions
7. A URL that documents this distance and driving time information

Please ensure all information is current as of February 2026 and sourced from official airline, airport, or reliable travel information websites.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RouteInfo(BaseModel):
    airline_name: Optional[str] = None
    destination_airport_full_name: Optional[str] = None
    destination_airport_iata: Optional[str] = None
    service_schedule: Optional[str] = None  # "year-round" or "seasonal"
    reference_url: Optional[str] = None


class CruiseDriveInfo(BaseModel):
    driving_distance_miles: Optional[str] = None  # keep as string to allow "approx. 230 mi"
    driving_time_h_m: Optional[str] = None
    reference_url: Optional[str] = None


class VacationExtraction(BaseModel):
    route1: Optional[RouteInfo] = None
    route2: Optional[RouteInfo] = None
    boston_drive: Optional[CruiseDriveInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_vacation() -> str:
    return """
Extract the following structured information from the answer text for two distinct non-stop flight routes from Bangor International Airport (BGR) to different Florida airports, and for the Boston cruise driving option.

Rules:
- Only extract what is explicitly mentioned in the answer.
- For URLs, extract the full URL string if present. If not present, set to null.
- If the answer provides more than two flight routes, pick the first two that are clearly non-stop and go to Florida. If non-stop isn't explicitly stated but implied (e.g., "direct" or route maps listing "nonstop"), still extract as provided in the answer text.
- If any field is missing, return null for that field.

Return a JSON with the following fields:

route1:
  airline_name: string or null
  destination_airport_full_name: string or null  (e.g., "Orlando International Airport")
  destination_airport_iata: string or null       (e.g., "MCO", 3-letter code)
  service_schedule: string or null               (e.g., "year-round" or "seasonal")
  reference_url: string or null                  (official airline or airport page documenting the specific BGR→Florida route)

route2:
  airline_name: string or null                   (must be a different airline from route1 per the task)
  destination_airport_full_name: string or null  (must be a different Florida airport from route1 per the task)
  destination_airport_iata: string or null
  service_schedule: string or null
  reference_url: string or null

boston_drive:
  driving_distance_miles: string or null         (e.g., "231 miles", "approx. 235 mi")
  driving_time_h_m: string or null               (e.g., "3 hr 45 min")
  reference_url: string or null                  (page that lists distance and time, e.g., Google Maps, Massport, etc.)
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def norm(s: Optional[str]) -> str:
    return (s or "").strip()


def upper_or_empty(s: Optional[str]) -> str:
    return (s or "").strip().upper()


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_flight_route(
    evaluator: Evaluator,
    parent_node,
    route: RouteInfo,
    idx: int
) -> None:
    """
    Build and verify the subtree for a single Florida flight route from BGR.
    All four checks are critical as per rubric.
    """
    rid = idx  # 1 or 2
    airline = norm(route.airline_name)
    dest_name = norm(route.destination_airport_full_name)
    dest_iata = upper_or_empty(route.destination_airport_iata)
    schedule = norm(route.service_schedule)
    ref_url = norm(route.reference_url)

    # Parent node (parallel), non-critical group container
    route_node = evaluator.add_parallel(
        id=f"florida_route_{rid}",
        desc=f"{'First' if rid == 1 else 'Second'} airline and route option from Bangor International Airport to a Florida destination",
        parent=evaluator.root,
        critical=False
    )

    # 1) Airline name leaf (critical)
    airline_leaf = evaluator.add_leaf(
        id=f"airline_name_{rid}",
        desc=f"Name of the airline operating this route from Bangor to Florida",
        parent=route_node,
        critical=True
    )
    claim_airline = (
        f"The webpage shows that the airline '{airline}' operates a non-stop (direct) route between "
        f"Bangor International Airport (BGR) and {dest_name} ({dest_iata}) in Florida."
    )
    await evaluator.verify(
        claim=claim_airline,
        node=airline_leaf,
        sources=ref_url if ref_url else None,
        additional_instruction=(
            "Only pass if the provided webpage explicitly supports this non-stop route and carrier. "
            "Treat 'nonstop' and 'direct' as equivalent. If no valid webpage is provided, mark as Incorrect."
        )
    )

    # 2) Destination airport leaf (critical)
    dest_leaf = evaluator.add_leaf(
        id=f"destination_airport_{rid}",
        desc="Complete destination airport information including the full official airport name and three-letter IATA code",
        parent=route_node,
        critical=True
    )
    claim_dest = (
        f"The webpage shows that the Florida destination airport for the Bangor (BGR) non-stop flight is "
        f"'{dest_name}' with IATA code '{dest_iata}', and that this airport is located in Florida."
    )
    await evaluator.verify(
        claim=claim_dest,
        node=dest_leaf,
        sources=ref_url if ref_url else None,
        additional_instruction=(
            "Allow minor variations in the official airport name (e.g., inclusion/omission of 'International'). "
            "The IATA code must match the claimed code, and the airport must be in Florida. "
            "If no valid webpage is provided, mark as Incorrect."
        )
    )

    # 3) Service frequency leaf (critical)
    freq_leaf = evaluator.add_leaf(
        id=f"service_frequency_{rid}",
        desc="Service operating schedule: whether the route operates year-round or seasonal only",
        parent=route_node,
        critical=True
    )
    claim_freq = (
        f"The webpage indicates that the Bangor (BGR) – {dest_name} ({dest_iata}) service is '{schedule}'."
    )
    await evaluator.verify(
        claim=claim_freq,
        node=freq_leaf,
        sources=ref_url if ref_url else None,
        additional_instruction=(
            "Interpretation: 'year-round' means the service is scheduled throughout the year; "
            "'seasonal' means it only operates for certain months or a defined seasonal period. "
            "If the page lists months or indicates 'seasonal', treat it as seasonal. "
            "If no valid webpage is provided, mark as Incorrect."
        )
    )

    # 4) Reference URL leaf (critical)
    ref_leaf = evaluator.add_leaf(
        id=f"reference_url_{rid}",
        desc="URL reference from an official airline or airport source documenting this specific route",
        parent=route_node,
        critical=True
    )
    claim_ref = (
        f"The URL '{ref_url}' is from an official airline or airport website and documents the specific non-stop "
        f"Bangor (BGR) to {dest_name} ({dest_iata}) route operated by {airline}."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=ref_url if ref_url else None,
        additional_instruction=(
            "Official sources include airline or airport domains (e.g., aa.com, delta.com, united.com, "
            "bangorinternationalairport.com, flymco.com). Press releases or route pages on official domains count. "
            "Third-party aggregators or blogs do not count. The page must clearly document this route. "
            "If the URL is missing or not official/relevant, mark as Incorrect."
        )
    )


async def verify_routes_distinctness(
    evaluator: Evaluator,
    parent_node,
    route1: Optional[RouteInfo],
    route2: Optional[RouteInfo]
) -> None:
    """
    Cross-route critical checks:
    - Airlines must be distinct.
    - Destination airports must be different Florida airports.
    """
    r1_air = norm(route1.airline_name if route1 else None)
    r2_air = norm(route2.airline_name if route2 else None)
    r1_code = upper_or_empty(route1.destination_airport_iata if route1 else None)
    r2_code = upper_or_empty(route2.destination_airport_iata if route2 else None)
    r1_name = norm(route1.destination_airport_full_name if route1 else None)
    r2_name = norm(route2.destination_airport_full_name if route2 else None)

    # Critical leaf: airlines distinct (simple logical check)
    distinct_airlines_leaf = evaluator.add_leaf(
        id="distinct_airlines",
        desc="The two airlines for the Florida routes must be distinct from each other",
        parent=parent_node,
        critical=True
    )
    claim_airlines = (
        f"The two airline names refer to different companies: '{r1_air}' vs '{r2_air}'. "
        f"Consider common abbreviations and branding (e.g., 'AA' vs 'American Airlines' should be treated as the same)."
    )
    await evaluator.verify(
        claim=claim_airlines,
        node=distinct_airlines_leaf,
        additional_instruction=(
            "Judge whether these two airline names refer to different carriers. "
            "Treat abbreviations, subsidiaries, or brand variations (e.g., 'American' vs 'AA', 'United Express' vs 'United Airlines') "
            "as the same operator unless clearly a different airline. "
            "If either name is missing, mark as Incorrect."
        )
    )

    # Critical custom node: destinations are different airports (IATA code comparison when available)
    # Prefer a strict code inequality check if both codes present; otherwise fall back to names
    codes_present = bool(r1_code) and bool(r2_code)
    if codes_present:
        result_diff = (r1_code != r2_code)
        evaluator.add_custom_node(
            result=result_diff,
            id="distinct_destinations_codes",
            desc=f"The two Florida destination airports are different (IATA codes compared: '{r1_code}' vs '{r2_code}')",
            parent=parent_node,
            critical=True
        )
    else:
        # Fallback on names if codes missing
        result_diff_names = bool(r1_name) and bool(r2_name) and (r1_name.lower() != r2_name.lower())
        evaluator.add_custom_node(
            result=result_diff_names,
            id="distinct_destinations_names",
            desc=f"The two Florida destination airports are different (names compared: '{r1_name}' vs '{r2_name}')",
            parent=parent_node,
            critical=True
        )


async def verify_boston_cruise_logistics(
    evaluator: Evaluator,
    parent_node,
    drive: CruiseDriveInfo
) -> None:
    """
    Verify Boston cruise driving information with three critical leaves.
    """
    dist = norm(drive.driving_distance_miles)
    time_hm = norm(drive.driving_time_h_m)
    ref_url = norm(drive.reference_url)

    drive_node = evaluator.add_parallel(
        id="boston_cruise_logistics",
        desc="Driving information from Bangor to Boston's cruise terminal",
        parent=parent_node,
        critical=False
    )

    # Driving distance leaf (critical)
    dist_leaf = evaluator.add_leaf(
        id="driving_distance",
        desc="Total driving distance in miles from Bangor, Maine to Boston's Black Falcon Cruise Terminal",
        parent=drive_node,
        critical=True
    )
    claim_dist = (
        f"The webpage shows the driving distance from Bangor, Maine to Flynn Cruiseport Boston "
        f"(Black Falcon Cruise Terminal) is approximately '{dist}'."
    )
    await evaluator.verify(
        claim=claim_dist,
        node=dist_leaf,
        sources=ref_url if ref_url else None,
        additional_instruction=(
            "Verify that the page shows a route between Bangor, ME and the cruise terminal "
            "(also known as Flynn Cruiseport Boston or Black Falcon Cruise Terminal, 1 Black Falcon Ave, Boston, MA). "
            "Because routes vary, accept reasonable approximations (±~10%). "
            "If no valid webpage is provided, mark as Incorrect."
        )
    )

    # Driving time leaf (critical)
    time_leaf = evaluator.add_leaf(
        id="driving_time",
        desc="Approximate driving time in hours and minutes under normal traffic conditions",
        parent=drive_node,
        critical=True
    )
    claim_time = (
        f"The webpage shows the approximate driving time (normal traffic) for this route is '{time_hm}'."
    )
    await evaluator.verify(
        claim=claim_time,
        node=time_leaf,
        sources=ref_url if ref_url else None,
        additional_instruction=(
            "Verify the page shows an expected driving time for the route between Bangor, ME and Flynn Cruiseport Boston/Black Falcon. "
            "Minor variations are acceptable; we are checking plausibility and consistency. "
            "If no valid webpage is provided, mark as Incorrect."
        )
    )

    # Reference URL reliability leaf (critical)
    ref_leaf = evaluator.add_leaf(
        id="reference_url_3",
        desc="URL reference documenting the distance and driving time between these locations",
        parent=drive_node,
        critical=True
    )
    claim_ref = (
        f"The URL '{ref_url}' is from a reliable travel information source (e.g., Google Maps, Bing Maps, Apple Maps, "
        f"Massport or official port/city sites) and documents driving directions between Bangor, ME and "
        f"Flynn Cruiseport Boston/Black Falcon, including both distance and time."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=ref_url if ref_url else None,
        additional_instruction=(
            "Accept reliable sources such as Google Maps, Bing Maps, Apple Maps, Massport/official port/city websites, "
            "or well-known mapping services. The page must include both distance and time for the driving route. "
            "If the URL is missing or not reliable/relevant, mark as Incorrect."
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
    Evaluate an answer for the Bangor flights vs Boston cruise task.
    """
    # Initialize evaluator with parallel root as per rubric
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

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_vacation(),
        template_class=VacationExtraction,
        extraction_name="vacation_extraction"
    )

    route1 = extracted.route1 or RouteInfo()
    route2 = extracted.route2 or RouteInfo()
    drive = extracted.boston_drive or CruiseDriveInfo()

    # Build flight route verifications
    await verify_flight_route(evaluator, root, route1, idx=1)
    await verify_flight_route(evaluator, root, route2, idx=2)

    # Cross-route distinctness checks (critical at root level)
    await verify_routes_distinctness(evaluator, root, route1, route2)

    # Driving logistics verifications (Boston cruise option)
    await verify_boston_cruise_logistics(evaluator, root, drive)

    # Return structured summary
    return evaluator.get_summary()