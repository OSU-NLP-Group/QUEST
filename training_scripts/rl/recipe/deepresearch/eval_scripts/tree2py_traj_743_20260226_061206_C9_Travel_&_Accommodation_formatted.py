import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "amtrak_ld_routes_np_ski_hub_ferry_v1"
TASK_DESCRIPTION = (
    "Identify four distinct Amtrak long-distance passenger train routes in the United States, where each route must "
    "satisfy all of the following criteria:\n\n"
    "1. National Park Access with In-Park Lodging: The route must have at least one official Amtrak station stop that "
    "provides access to a U.S. national park, and that national park must have at least one lodge or accommodation "
    "facility located within the park boundaries and operated by an official park concessionaire.\n"
    "2. Ski Resort Access with On-Mountain Lodging: The route must have at least one official Amtrak station stop that "
    "provides access to a ski resort, and that ski resort must have at least one lodge or hotel located within the "
    "resort boundaries that offers ski-in/ski-out access or direct slope access.\n"
    "3. Major Airport Hub Connection: The route must stop at a city that is served by a major U.S. airport hub, defined "
    "as an airport with nonstop service to 100 or more destinations.\n"
    "4. Lake Michigan Ferry Connection: The route must have at least one official Amtrak station stop in a city with "
    "operational Lake Michigan cross-lake ferry service (either SS Badger operating between Ludington, MI and Manitowoc, "
    "WI, or Lake Express operating between Milwaukee, WI and Muskegon, MI).\n\n"
    "For each of the four routes, provide:\n"
    "- The official name of the Amtrak long-distance route\n"
    "- The station providing national park access, the name of the national park, and the name of at least one in-park "
    "lodge with its concessionaire operator\n"
    "- The station providing ski resort access, the name of the ski resort, and the name of at least one on-mountain lodge "
    "with ski-in/ski-out or direct slope access\n"
    "- The station connecting to the major airport hub, the name and airport code of the hub airport, and confirmation that "
    "it serves 100+ nonstop destinations\n"
    "- The station with Lake Michigan ferry service and the name of the ferry service\n\n"
    "Additionally, provide a direct URL reference from an official source for each piece of information to verify that "
    "the stations are on the specified routes, that the lodges are within park/resort boundaries, that the airports meet "
    "the hub criteria, and that the ferry services operate from the specified cities."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ParkInfo(BaseModel):
    station: Optional[str] = None
    park_name: Optional[str] = None
    lodge_name: Optional[str] = None
    concessionaire: Optional[str] = None
    station_on_route_urls: List[str] = Field(default_factory=list)
    park_access_urls: List[str] = Field(default_factory=list)
    lodge_location_urls: List[str] = Field(default_factory=list)
    lodge_operator_urls: List[str] = Field(default_factory=list)


class SkiInfo(BaseModel):
    station: Optional[str] = None
    resort_name: Optional[str] = None
    lodge_name: Optional[str] = None
    station_on_route_urls: List[str] = Field(default_factory=list)
    resort_access_urls: List[str] = Field(default_factory=list)
    lodge_location_urls: List[str] = Field(default_factory=list)
    lodge_access_urls: List[str] = Field(default_factory=list)


class AirportInfo(BaseModel):
    station: Optional[str] = None
    hub_city: Optional[str] = None
    airport_name: Optional[str] = None
    airport_code: Optional[str] = None
    station_on_route_urls: List[str] = Field(default_factory=list)
    airport_city_service_urls: List[str] = Field(default_factory=list)
    nonstop_destinations_urls: List[str] = Field(default_factory=list)


class FerryInfo(BaseModel):
    station: Optional[str] = None
    ferry_city: Optional[str] = None
    ferry_service_name: Optional[str] = None
    station_on_route_urls: List[str] = Field(default_factory=list)
    ferry_city_urls: List[str] = Field(default_factory=list)
    ferry_operation_urls: List[str] = Field(default_factory=list)


class RouteCandidate(BaseModel):
    route_name: Optional[str] = None
    route_url: Optional[str] = None
    park: ParkInfo = Field(default_factory=ParkInfo)
    ski: SkiInfo = Field(default_factory=SkiInfo)
    airport: AirportInfo = Field(default_factory=AirportInfo)
    ferry: FerryInfo = Field(default_factory=FerryInfo)


class RoutesExtraction(BaseModel):
    routes: List[RouteCandidate] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_routes() -> str:
    return (
        "Extract up to four Amtrak long-distance routes described in the answer. For each route, return a structured "
        "JSON object with the following fields, using ONLY the exact URLs explicitly present in the answer text:\n\n"
        "Route fields:\n"
        "- route_name: The official name of the Amtrak long-distance passenger train route\n"
        "- route_url: A URL (preferably an official Amtrak page) that confirms the route name/designation\n\n"
        "National Park section (park):\n"
        "- station: Name of the Amtrak station that provides access to the national park\n"
        "- park_name: Name of the U.S. national park\n"
        "- lodge_name: Name of at least one lodge inside the park\n"
        "- concessionaire: Name of the lodge operator (e.g., Xanterra, Delaware North, Aramark, Forever Resorts)\n"
        "- station_on_route_urls: Array of URLs that show this station is an official stop on the specified route "
        "(Amtrak route page, schedule, or station list)\n"
        "- park_access_urls: Array of URLs that confirm the station provides access to the national park "
        "(NPS, official shuttle provider, or official visitor info)\n"
        "- lodge_location_urls: Array of URLs that confirm the lodge is located within park boundaries "
        "(NPS, concessionaire official site)\n"
        "- lodge_operator_urls: Array of URLs that confirm the concessionaire operates the lodge\n\n"
        "Ski section (ski):\n"
        "- station: Name of the Amtrak station providing access to the ski resort\n"
        "- resort_name: Name of the ski resort\n"
        "- lodge_name: Name of an on-mountain lodge/hotel\n"
        "- station_on_route_urls: Array of URLs confirming the station is on the route\n"
        "- resort_access_urls: Array of URLs confirming the station provides access to the resort "
        "(resort website, local transit/shuttle, official tourism)\n"
        "- lodge_location_urls: Array of URLs confirming the lodge is located within resort boundaries or is on-mountain\n"
        "- lodge_access_urls: Array of URLs confirming ski-in/ski-out or direct slope access\n\n"
        "Airport hub section (airport):\n"
        "- station: Name of the Amtrak station where the route stops in the hub city\n"
        "- hub_city: Name of the city\n"
        "- airport_name: Name of the major airport\n"
        "- airport_code: Airport IATA code (e.g., ORD)\n"
        "- station_on_route_urls: Array of URLs confirming the station is on the route\n"
        "- airport_city_service_urls: Array of URLs confirming the airport serves the specified city\n"
        "- nonstop_destinations_urls: Array of URLs confirming the airport has nonstop service to 100+ destinations\n\n"
        "Ferry section (ferry):\n"
        "- station: Name of the Amtrak station in the ferry city\n"
        "- ferry_city: Name of the city with Lake Michigan cross-lake ferry service\n"
        "- ferry_service_name: 'SS Badger' or 'Lake Express'\n"
        "- station_on_route_urls: Array of URLs confirming the station is on the route\n"
        "- ferry_city_urls: Array of URLs confirming the ferry operates from this city (official ferry website preferred)\n"
        "- ferry_operation_urls: Array of URLs confirming the service is a cross-lake ferry across Lake Michigan\n\n"
        "Return a JSON object with a single field 'routes' which is an array of up to four route objects as defined above. "
        "If the answer includes more than four routes, extract only the first four. If any field is missing, set it to null; "
        "for URL arrays, return an empty list if none are provided."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> Optional[str]:
    return s.strip() if isinstance(s, str) and s.strip() != "" else None


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(_norm(u) for u in urls)


def _first_n_routes(extracted: RoutesExtraction, n: int = 4) -> List[RouteCandidate]:
    routes = extracted.routes[:n] if extracted and extracted.routes else []
    # pad to length n
    while len(routes) < n:
        routes.append(RouteCandidate())
    return routes


# --------------------------------------------------------------------------- #
# Verification per-route                                                      #
# --------------------------------------------------------------------------- #
async def verify_route(
    evaluator: Evaluator,
    parent_route_node,
    idx: int,
    route: RouteCandidate,
) -> None:
    """
    Build verification tree for a single route and run checks.
    All children are critical under a critical route node.
    """
    rid = idx + 1
    route_name = _norm(route.route_name)

    # 1) Route name verification (sequential)
    name_node = evaluator.add_sequential(
        id=f"Route_{rid}_Name_Verification",
        desc=f"Route {rid}: Official Amtrak long-distance route name verification",
        parent=parent_route_node,
        critical=True,
    )

    name_exists = evaluator.add_custom_node(
        result=bool(route_name) and bool(_norm(route.route_url)),
        id=f"Route_{rid}_Name_Exists",
        desc=f"Route {rid}: Route name and verifying URL are provided",
        parent=name_node,
        critical=True,
    )

    name_url_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_Name_URL",
        desc=f"Route {rid}: URL confirms the route '{route_name or 'UNKNOWN'}' is an officially designated Amtrak long-distance passenger train",
        parent=name_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{route_name or 'UNKNOWN'}' is an officially designated Amtrak long-distance passenger train route.",
        node=name_url_leaf,
        sources=_norm(route.route_url),
        additional_instruction=(
            "Confirm on the provided page that the named route is an Amtrak route operating long-distance service. "
            "Accept an official Amtrak route page, timetable, or route description in the amtrak.com domain (or equivalent official source). "
            "Minor naming variations are acceptable."
        ),
    )

    # 2) National Park section (sequential)
    np_node = evaluator.add_sequential(
        id=f"Route_{rid}_National_Park",
        desc=f"Route {rid}: National Park access with in-park concessionaire-operated lodging",
        parent=parent_route_node,
        critical=True,
    )

    # Station on route
    np_station_exists = evaluator.add_custom_node(
        result=bool(_norm(route.park.station)) and _has_urls(route.park.station_on_route_urls),
        id=f"Route_{rid}_NP_Station_Exists",
        desc=f"Route {rid}: National Park station and station-on-route URLs are provided",
        parent=np_node,
        critical=True,
    )
    np_station_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_NP_Station_URL",
        desc=f"Route {rid}: The station is an official stop on this Amtrak route",
        parent=np_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The station '{route.park.station or 'UNKNOWN'}' is an official stop on the Amtrak route '{route_name or 'UNKNOWN'}'.",
        node=np_station_leaf,
        sources=route.park.station_on_route_urls,
        additional_instruction="Verify the route's official stop list or timetable confirms this station for the named route.",
    )

    # Park access
    np_park_exists = evaluator.add_custom_node(
        result=bool(_norm(route.park.park_name)) and _has_urls(route.park.park_access_urls),
        id=f"Route_{rid}_NP_Park_Exists",
        desc=f"Route {rid}: National Park name and park-access URLs are provided",
        parent=np_node,
        critical=True,
    )
    np_park_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_NP_Park_URL",
        desc=f"Route {rid}: The station provides access to the specified national park",
        parent=np_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The station '{route.park.station or 'UNKNOWN'}' provides access to {route.park.park_name or 'UNKNOWN'} National Park.",
        node=np_park_leaf,
        sources=route.park.park_access_urls,
        additional_instruction=(
            "Confirm that the station is commonly used to access the park via official NPS pages, park transportation/shuttle information, "
            "or trusted official visitor resources."
        ),
    )

    # Lodge details (parallel)
    np_lodge_node = evaluator.add_parallel(
        id=f"Route_{rid}_NP_Lodge",
        desc=f"Route {rid}: Park lodge within boundaries and operated by official concessionaire",
        parent=np_node,
        critical=True,
    )

    # Lodge location
    np_lodge_loc_exists = evaluator.add_custom_node(
        result=bool(_norm(route.park.lodge_name)) and _has_urls(route.park.lodge_location_urls),
        id=f"Route_{rid}_NP_Lodge_Location_Exists",
        desc=f"Route {rid}: Lodge name and lodge-location URLs are provided",
        parent=np_lodge_node,
        critical=True,
    )
    np_lodge_loc_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_NP_Lodge_Location_URL",
        desc=f"Route {rid}: The lodge is located within the national park boundaries",
        parent=np_lodge_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The lodge '{route.park.lodge_name or 'UNKNOWN'}' is located within the boundaries of {route.park.park_name or 'UNKNOWN'} National Park.",
        node=np_lodge_loc_leaf,
        sources=route.park.lodge_location_urls,
        additional_instruction="Confirm using NPS or concessionaire official sources that the lodge is inside the park.",
    )

    # Lodge operator (concessionaire)
    np_lodge_op_exists = evaluator.add_custom_node(
        result=bool(_norm(route.park.lodge_name)) and bool(_norm(route.park.concessionaire)) and _has_urls(route.park.lodge_operator_urls),
        id=f"Route_{rid}_NP_Lodge_Operator_Exists",
        desc=f"Route {rid}: Lodge operator (official concessionaire) and URLs are provided",
        parent=np_lodge_node,
        critical=True,
    )
    np_lodge_op_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_NP_Lodge_Operator_URL",
        desc=f"Route {rid}: The lodge is operated by an official park concessionaire",
        parent=np_lodge_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The lodge '{route.park.lodge_name or 'UNKNOWN'}' is operated by official park concessionaire '{route.park.concessionaire or 'UNKNOWN'}'.",
        node=np_lodge_op_leaf,
        sources=route.park.lodge_operator_urls,
        additional_instruction=(
            "Confirm that the named operator is an official NPS concessionaire for the lodge (e.g., Xanterra, Delaware North, Aramark, Forever Resorts)."
        ),
    )

    # 3) Ski section (sequential)
    ski_node = evaluator.add_sequential(
        id=f"Route_{rid}_Ski_Resort",
        desc=f"Route {rid}: Ski resort access with on-mountain lodge offering ski-in/ski-out or direct slope access",
        parent=parent_route_node,
        critical=True,
    )

    # Ski station on route
    ski_station_exists = evaluator.add_custom_node(
        result=bool(_norm(route.ski.station)) and _has_urls(route.ski.station_on_route_urls),
        id=f"Route_{rid}_Ski_Station_Exists",
        desc=f"Route {rid}: Ski station and station-on-route URLs are provided",
        parent=ski_node,
        critical=True,
    )
    ski_station_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_Ski_Station_URL",
        desc=f"Route {rid}: The ski station is an official stop on this Amtrak route",
        parent=ski_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The station '{route.ski.station or 'UNKNOWN'}' is an official stop on the Amtrak route '{route_name or 'UNKNOWN'}'.",
        node=ski_station_leaf,
        sources=route.ski.station_on_route_urls,
        additional_instruction="Confirm the station appears in the official stop list or timetable for the named route.",
    )

    # Resort access
    ski_resort_exists = evaluator.add_custom_node(
        result=bool(_norm(route.ski.resort_name)) and _has_urls(route.ski.resort_access_urls),
        id=f"Route_{rid}_Ski_Resort_Exists",
        desc=f"Route {rid}: Ski resort name and resort-access URLs are provided",
        parent=ski_node,
        critical=True,
    )
    ski_resort_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_Ski_Resort_URL",
        desc=f"Route {rid}: The station provides access to the specified ski resort",
        parent=ski_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The station '{route.ski.station or 'UNKNOWN'}' provides access to the ski resort '{route.ski.resort_name or 'UNKNOWN'}'.",
        node=ski_resort_leaf,
        sources=route.ski.resort_access_urls,
        additional_instruction="Confirm via resort website or official transportation/shuttle info that the station is a recognized access point.",
    )

    # Ski lodge (parallel)
    ski_lodge_node = evaluator.add_parallel(
        id=f"Route_{rid}_Ski_Lodge",
        desc=f"Route {rid}: On-mountain lodge location and ski-in/ski-out or direct slope access",
        parent=ski_node,
        critical=True,
    )

    ski_lodge_loc_exists = evaluator.add_custom_node(
        result=bool(_norm(route.ski.lodge_name)) and _has_urls(route.ski.lodge_location_urls),
        id=f"Route_{rid}_Ski_Lodge_Location_Exists",
        desc=f"Route {rid}: Ski lodge name and lodge-location URLs are provided",
        parent=ski_lodge_node,
        critical=True,
    )
    ski_lodge_loc_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_Ski_Lodge_Location_URL",
        desc=f"Route {rid}: The ski lodge is located within resort boundaries or on-mountain",
        parent=ski_lodge_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The lodge '{route.ski.lodge_name or 'UNKNOWN'}' is located within the boundaries of the ski resort '{route.ski.resort_name or 'UNKNOWN'}' or is on-mountain.",
        node=ski_lodge_loc_leaf,
        sources=route.ski.lodge_location_urls,
        additional_instruction="Confirm via the resort or lodge official site that the lodge is on-mountain or within resort property.",
    )

    ski_lodge_access_exists = evaluator.add_custom_node(
        result=bool(_norm(route.ski.lodge_name)) and _has_urls(route.ski.lodge_access_urls),
        id=f"Route_{rid}_Ski_Lodge_Access_Exists",
        desc=f"Route {rid}: Ski lodge access URLs (ski-in/ski-out or direct slope access) are provided",
        parent=ski_lodge_node,
        critical=True,
    )
    ski_lodge_access_leaf = evaluator.add_leaf(
            id=f"Route_{rid}_Ski_Lodge_Access_URL",
            desc=f"Route {rid}: The lodge provides ski-in/ski-out or direct slope access",
            parent=ski_lodge_node,
            critical=True,
        )
    await evaluator.verify(
        claim=f"The lodge '{route.ski.lodge_name or 'UNKNOWN'}' offers ski-in/ski-out or direct slope access.",
        node=ski_lodge_access_leaf,
        sources=route.ski.lodge_access_urls,
        additional_instruction="Confirm any explicit statement such as 'ski-in/ski-out', 'slopeside', or direct lift/slope access on official sources.",
    )

    # 4) Airport hub section (sequential)
    hub_node = evaluator.add_sequential(
        id=f"Route_{rid}_Airport_Hub",
        desc=f"Route {rid}: Major airport hub connection (100+ nonstop destinations)",
        parent=parent_route_node,
        critical=True,
    )

    hub_station_exists = evaluator.add_custom_node(
        result=bool(_norm(route.airport.station)) and _has_urls(route.airport.station_on_route_urls),
        id=f"Route_{rid}_Hub_Station_Exists",
        desc=f"Route {rid}: Hub station and station-on-route URLs are provided",
        parent=hub_node,
        critical=True,
    )
    hub_station_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_Hub_Station_URL",
        desc=f"Route {rid}: The hub station is an official stop on this Amtrak route",
        parent=hub_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The station '{route.airport.station or 'UNKNOWN'}' is an official stop on the Amtrak route '{route_name or 'UNKNOWN'}'.",
        node=hub_station_leaf,
        sources=route.airport.station_on_route_urls,
        additional_instruction="Check route stop list or timetable for confirmation.",
    )

    hub_airport_node = evaluator.add_parallel(
        id=f"Route_{rid}_Hub_Airport",
        desc=f"Route {rid}: Airport serves the city and has 100+ nonstop destinations",
        parent=hub_node,
        critical=True,
    )

    hub_airport_ident_exists = evaluator.add_custom_node(
        result=bool(_norm(route.airport.hub_city)) and bool(_norm(route.airport.airport_name)) and bool(_norm(route.airport.airport_code)) and _has_urls(route.airport.airport_city_service_urls),
        id=f"Route_{rid}_Hub_Airport_Ident_Exists",
        desc=f"Route {rid}: Airport identification fields and city-service URLs are provided",
        parent=hub_airport_node,
        critical=True,
    )
    hub_airport_ident_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_Hub_Airport_URL",
        desc=f"Route {rid}: The airport serves the specified hub city",
        parent=hub_airport_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The airport '{route.airport.airport_name or 'UNKNOWN'}' ({route.airport.airport_code or 'UNK'}) serves the city '{route.airport.hub_city or 'UNKNOWN'}'.",
        node=hub_airport_ident_leaf,
        sources=route.airport.airport_city_service_urls,
        additional_instruction="Confirm via airport official site or authoritative source that the airport serves the specified city.",
    )

    hub_dest_exists = evaluator.add_custom_node(
        result=_has_urls(route.airport.nonstop_destinations_urls),
        id=f"Route_{rid}_Hub_Destinations_Exists",
        desc=f"Route {rid}: URLs confirming 100+ nonstop destinations are provided",
        parent=hub_airport_node,
        critical=True,
    )
    hub_dest_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_Hub_Destinations_URL",
        desc=f"Route {rid}: The airport offers nonstop service to 100 or more destinations",
        parent=hub_airport_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The airport '{route.airport.airport_name or 'UNKNOWN'}' offers nonstop service to 100 or more destinations.",
        node=hub_dest_leaf,
        sources=route.airport.nonstop_destinations_urls,
        additional_instruction="Confirm using airport statistics or official route maps; seasonal destinations count toward the total.",
    )

    # 5) Ferry section (sequential)
    ferry_node = evaluator.add_sequential(
        id=f"Route_{rid}_Ferry",
        desc=f"Route {rid}: Lake Michigan cross-lake ferry connection (SS Badger or Lake Express)",
        parent=parent_route_node,
        critical=True,
    )

    ferry_station_exists = evaluator.add_custom_node(
        result=bool(_norm(route.ferry.station)) and _has_urls(route.ferry.station_on_route_urls),
        id=f"Route_{rid}_Ferry_Station_Exists",
        desc=f"Route {rid}: Ferry city station and station-on-route URLs are provided",
        parent=ferry_node,
        critical=True,
    )
    ferry_station_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_Ferry_Station_URL",
        desc=f"Route {rid}: The ferry city station is an official stop on this Amtrak route",
        parent=ferry_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The station '{route.ferry.station or 'UNKNOWN'}' is an official stop on the Amtrak route '{route_name or 'UNKNOWN'}'.",
        node=ferry_station_leaf,
        sources=route.ferry.station_on_route_urls,
        additional_instruction="Check route stop list or timetable for confirmation.",
    )

    ferry_service_node = evaluator.add_parallel(
        id=f"Route_{rid}_Ferry_Service",
        desc=f"Route {rid}: Ferry operates from the specified city and is a Lake Michigan cross-lake service",
        parent=ferry_node,
        critical=True,
    )

    ferry_location_exists = evaluator.add_custom_node(
        result=bool(_norm(route.ferry.ferry_city)) and bool(_norm(route.ferry.ferry_service_name)) and _has_urls(route.ferry.ferry_city_urls),
        id=f"Route_{rid}_Ferry_Location_Exists",
        desc=f"Route {rid}: Ferry city and city-operation URLs are provided",
        parent=ferry_service_node,
        critical=True,
    )
    ferry_location_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_Ferry_Location_URL",
        desc=f"Route {rid}: The ferry service operates from the specified city",
        parent=ferry_service_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The ferry service '{route.ferry.ferry_service_name or 'UNKNOWN'}' operates from the city '{route.ferry.ferry_city or 'UNKNOWN'}'.",
        node=ferry_location_leaf,
        sources=route.ferry.ferry_city_urls,
        additional_instruction="Confirm via official ferry website pages indicating the departure port city.",
    )

    ferry_operation_exists = evaluator.add_custom_node(
        result=_has_urls(route.ferry.ferry_operation_urls),
        id=f"Route_{rid}_Ferry_Operation_Exists",
        desc=f"Route {rid}: URLs confirming cross-lake operation across Lake Michigan are provided",
        parent=ferry_service_node,
        critical=True,
    )
    ferry_operation_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_Ferry_Operation_URL",
        desc=f"Route {rid}: The ferry provides cross-lake service across Lake Michigan",
        parent=ferry_service_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The ferry service '{route.ferry.ferry_service_name or 'UNKNOWN'}' provides cross-lake service across Lake Michigan.",
        node=ferry_operation_leaf,
        sources=route.ferry.ferry_operation_urls,
        additional_instruction="Confirm endpoints span Lake Michigan (e.g., Milwaukee–Muskegon for Lake Express or Ludington–Manitowoc for SS Badger).",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Amtrak long-distance routes task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # routes evaluated independently
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

    # Extract structured info
    extracted_routes = await evaluator.extract(
        prompt=prompt_extract_routes(),
        template_class=RoutesExtraction,
        extraction_name="routes_extraction",
    )

    # Prepare first 4 routes
    routes = _first_n_routes(extracted_routes, n=4)

    # Add a global critical uniqueness check (sibling precondition for all route subtrees)
    norm_names = [(_norm(r.route_name) or f"__missing_{i}") for i, r in enumerate(routes)]
    # Only consider non-missing names for uniqueness; duplicates among provided names should fail
    provided_names = [n for n in norm_names if not n.startswith("__missing_")]
    distinct_result = len(set(provided_names)) == len(provided_names) and len(provided_names) == 4

    evaluator.add_custom_node(
        result=distinct_result,
        id="Routes_Distinct",
        desc="All four identified routes are distinct (no duplicates among provided route names)",
        parent=root,
        critical=True,
    )

    # Build critical route nodes and verify each route
    for i, route in enumerate(routes):
        route_node = evaluator.add_parallel(
            id=f"Route_{i+1}",
            desc=f"Route {i+1}: Must satisfy all criteria (name, national park, ski, airport hub, ferry)",
            parent=root,
            critical=True,  # each route must satisfy all criteria
        )
        await verify_route(evaluator, route_node, i, route)

    # Return summary
    return evaluator.get_summary()