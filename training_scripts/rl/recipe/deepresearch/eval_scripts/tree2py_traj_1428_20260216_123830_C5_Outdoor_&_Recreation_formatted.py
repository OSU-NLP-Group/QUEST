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
TASK_ID = "whistler_ski_trip_planning_2026_mlk"
TASK_DESCRIPTION = (
    "I'm planning a ski trip to Whistler, British Columbia for Martin Luther King Jr. Day weekend 2026 "
    "(January 17-19, 2026), departing from Nashville International Airport (BNA). I need help with the following:\n"
    "1) Identify a viable flight routing from BNA to YVR, including at least one connection hub, operating airlines, "
    "and approximate total travel time.\n"
    "2) Identify at least one shuttle/bus service between YVR and Whistler with distance, travel time, and reference URLs.\n"
    "3) Confirm Whistler Blackcomb will be open during MLK weekend 2026, and state typical base lift opening time in mid/late January.\n"
    "4) For Amex Platinum, identify at least two BNA airport lounges with locations and access policies.\n"
    "Additionally, if available, provide info on early morning ski access programs (e.g., First Tracks) including cost and upload times."
)

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class FlightRoutingItem(BaseModel):
    connection_hub: Optional[str] = None
    airlines: List[str] = Field(default_factory=list)
    total_travel_time: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class GroundTransportService(BaseModel):
    name: Optional[str] = None
    distance: Optional[str] = None  # Accept strings like "120 km (75 miles)"
    travel_time: Optional[str] = None  # Accept strings like "2-3 hours"
    urls: List[str] = Field(default_factory=list)


class ResortOperationsInfo(BaseModel):
    operating_status_statement: Optional[str] = None  # e.g., "Open during Jan 17-19, 2026"
    base_lift_open_time: Optional[str] = None  # e.g., "8:15 AM" or "8:30 AM"
    early_access_program_name: Optional[str] = None  # e.g., "First Tracks"
    early_access_cost: Optional[str] = None  # e.g., "$25 CAD" or similar
    early_access_upload_time: Optional[str] = None  # e.g., "7:30 AM"
    urls_ops: List[str] = Field(default_factory=list)   # URLs confirming operating season/dates
    urls_hours: List[str] = Field(default_factory=list) # URLs for lift hours
    urls_early: List[str] = Field(default_factory=list) # URLs for early access program info


class LoungeInfo(BaseModel):
    name: Optional[str] = None
    terminal_location: Optional[str] = None  # e.g., "Concourse B, near Gate B3"
    access_policy: Optional[str] = None  # e.g., "Amex Platinum with same-day Delta boarding pass" etc.
    urls: List[str] = Field(default_factory=list)


class TripPlanExtraction(BaseModel):
    flight_routings: List[FlightRoutingItem] = Field(default_factory=list)
    ground_transports: List[GroundTransportService] = Field(default_factory=list)
    resort_ops: Optional[ResortOperationsInfo] = None
    lounges: List[LoungeInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_plan() -> str:
    return """
    Extract structured details from the answer for the Whistler MLK 2026 trip plan. Only extract data that appears in the answer text. Do not invent anything. Follow strictly:

    1) flight_routings: an array of possible routings from BNA to YVR in January 2026.
       Each item fields:
       - connection_hub: name/code of a major connection city/airport (e.g., "SEA", "DEN", "SLC", "DFW", etc.) if mentioned.
       - airlines: array of airline names or codes operating the legs in the cited routing (e.g., ["Delta", "Alaska"]).
       - total_travel_time: approximate total travel time string (e.g., "7h 45m", "8–10 hours").
       - urls: array of URL(s) in the answer that specifically reference or support this routing (e.g., flight search result, airline route page). If none provided, return an empty array.

    2) ground_transports: an array of shuttle/bus services between YVR and Whistler.
       Each item fields:
       - name: service/company name (e.g., "SkyLynx", "Epic Rides", "Whistler Shuttle").
       - distance: approximate distance string (e.g., "120 km" or "75 miles" or both).
       - travel_time: approximate travel time string (e.g., "2–3 hours").
       - urls: array of reference URL(s) cited in the answer for the service.

    3) resort_ops: Whistler Blackcomb operations info.
       Fields:
       - operating_status_statement: a short statement from the answer about being open during Jan 17–19, 2026 (or season dates).
       - base_lift_open_time: typical base lift opening time string for mid-to-late January (e.g., "8:15 AM").
       - early_access_program_name: program name if mentioned (e.g., "First Tracks"), else null.
       - early_access_cost: program price string if mentioned (e.g., "$25 CAD"), else null.
       - early_access_upload_time: upload time string if mentioned (e.g., "7:30 AM"), else null.
       - urls_ops: URLs cited for operating season/dates confirmation.
       - urls_hours: URLs cited for lift hours.
       - urls_early: URLs cited for early access details.
       If any of these URL sets are missing in the answer, return an empty array for that field.

    4) lounges: an array of airport lounges at BNA accessible to Amex Platinum, as described by the answer.
       Each item fields:
       - name: lounge name (e.g., "Delta Sky Club", "The Club at BNA", "Centurion Lounge").
       - terminal_location: specific terminal/concourse/gate area location string.
       - access_policy: summarized policy relevant to Amex Platinum access as described (e.g., "Amex Platinum + same-day Delta boarding pass" or "Priority Pass via Amex Platinum", etc.).
       - urls: array of URL(s) cited for this lounge.

    Return a single JSON with these top-level fields: flight_routings, ground_transports, resort_ops, lounges.
    For any missing field, use null (for strings) or empty arrays (for lists).
    """


# --------------------------------------------------------------------------- #
# Helper selection functions                                                  #
# --------------------------------------------------------------------------- #
def pick_first_valid_route(items: List[FlightRoutingItem]) -> Optional[FlightRoutingItem]:
    for r in items:
        if r and (r.connection_hub or r.airlines or r.total_travel_time) and (r.urls and len(r.urls) > 0):
            return r
    # fallback to first if exists even without urls (will fail existence later)
    return items[0] if items else None


def pick_first_valid_service(items: List[GroundTransportService]) -> Optional[GroundTransportService]:
    for s in items:
        if s and s.name and (s.urls and len(s.urls) > 0):
            return s
    return items[0] if items else None


def ensure_list(x: Optional[List[str]]) -> List[str]:
    return x if x else []


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_flight_routing(
    evaluator: Evaluator,
    parent_node,
    extracted: TripPlanExtraction
) -> None:
    """
    Build and verify the Flight Routing subtree.
    """
    node = evaluator.add_parallel(
        id="Flight_Routing",
        desc="Identify a viable flight routing from Nashville (BNA) to Vancouver (YVR) for MLK weekend dates, including connection city, airlines, and total travel time",
        parent=parent_node,
        critical=True
    )

    route = pick_first_valid_route(extracted.flight_routings)
    connection_hub = (route.connection_hub or "").strip() if route else ""
    airlines_list = route.airlines if route else []
    total_time = (route.total_travel_time or "").strip() if route else ""
    urls = route.urls if route else []

    # Existence gate (critical)
    evaluator.add_custom_node(
        result=bool(route and connection_hub and airlines_list and total_time and urls),
        id="Flight_Routing_Exists",
        desc="Flight routing info with sources is provided",
        parent=node,
        critical=True
    )

    # Connection hub verification
    hub_leaf = evaluator.add_leaf(
        id="Connection_Hub",
        desc="Identify at least one major hub airport where connecting flights from BNA to YVR are available during January 2026",
        parent=node,
        critical=True
    )
    hub_claim = f"The provided itinerary shows a connection at {connection_hub} between Nashville (BNA) and Vancouver (YVR)."
    await evaluator.verify(
        claim=hub_claim,
        node=hub_leaf,
        sources=urls,
        additional_instruction="Confirm the page shows an itinerary or routing BNA → (connection) → YVR, with the connection specifically at the stated hub."
    )

    # Flight duration verification
    duration_leaf = evaluator.add_leaf(
        id="Flight_Duration",
        desc="Provide total estimated travel time from BNA to YVR including connection time",
        parent=node,
        critical=True
    )
    duration_claim = f"The total travel time from BNA to YVR is approximately {total_time}."
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=urls,
        additional_instruction="Accept reasonable approximations (e.g., 7h 45m vs 8h). Verify the total duration on the cited itinerary page."
    )

    # Operating airlines verification
    airlines_leaf = evaluator.add_leaf(
        id="Operating_Airlines",
        desc="Identify airlines operating on the chosen routing",
        parent=node,
        critical=True
    )
    airlines_text = ", ".join(airlines_list) if airlines_list else ""
    airlines_claim = f"The itinerary includes flights operated by {airlines_text}."
    await evaluator.verify(
        claim=airlines_claim,
        node=airlines_leaf,
        sources=urls,
        additional_instruction="Check the operating carriers listed on the itinerary or airline pages. Minor naming variations are acceptable."
    )


async def verify_ground_transport(
    evaluator: Evaluator,
    parent_node,
    extracted: TripPlanExtraction
) -> None:
    """
    Build and verify the Ground Transportation subtree.
    """
    node = evaluator.add_parallel(
        id="Ground_Transportation",
        desc="Identify ground transportation options from Vancouver International Airport (YVR) to Whistler",
        parent=parent_node,
        critical=True
    )

    svc = pick_first_valid_service(extracted.ground_transports)
    svc_name = (svc.name or "").strip() if svc else ""
    svc_distance = (svc.distance or "").strip() if svc else ""
    svc_time = (svc.travel_time or "").strip() if svc else ""
    urls = svc.urls if svc else []

    # Existence gate (critical)
    evaluator.add_custom_node(
        result=bool(svc and svc_name and urls),
        id="Ground_Transport_Exists",
        desc="At least one YVR ↔ Whistler shuttle or bus service with sources is provided",
        parent=node,
        critical=True
    )

    # Shuttle service verification
    shuttle_leaf = evaluator.add_leaf(
        id="Shuttle_Service",
        desc="Identify at least one shuttle or bus service operating between YVR and Whistler",
        parent=node,
        critical=True
    )
    shuttle_claim = f"The company {svc_name} operates a shuttle or bus service between Vancouver International Airport (YVR) and Whistler."
    await evaluator.verify(
        claim=shuttle_claim,
        node=shuttle_leaf,
        sources=urls,
        additional_instruction="Confirm the page clearly indicates service between YVR (or Vancouver Airport) and Whistler."
    )

    # Travel details verification
    details_leaf = evaluator.add_leaf(
        id="Travel_Details",
        desc="Provide the approximate distance and travel time from YVR to Whistler",
        parent=node,
        critical=True
    )
    details_claim = f"The distance from YVR to Whistler is approximately {svc_distance} and the travel time is approximately {svc_time}."
    await evaluator.verify(
        claim=details_claim,
        node=details_leaf,
        sources=urls,
        additional_instruction="Allow approximate values (e.g., ~120 km / ~75 miles, 2–3 hours). If only one unit is shown, verify that unit."
    )

    # Service reference verification
    ref_leaf = evaluator.add_leaf(
        id="Service_Reference",
        desc="Provide reference URL(s) for the shuttle service(s) mentioned",
        parent=node,
        critical=True
    )
    ref_claim = "This page provides official or authoritative information about a shuttle or bus service between Vancouver (YVR) and Whistler."
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=urls,
        additional_instruction="The page should be a service provider or authoritative source describing the YVR–Whistler shuttle/bus."
    )


async def verify_resort_operations(
    evaluator: Evaluator,
    parent_node,
    extracted: TripPlanExtraction
) -> None:
    """
    Build and verify the Resort Operations subtree.
    Note: This subtree is non-critical to allow partial credit; early access is optional.
    """
    node = evaluator.add_parallel(
        id="Resort_Operations",
        desc="Provide information about Whistler Blackcomb operations during MLK weekend 2026",
        parent=parent_node,
        critical=False  # Non-critical to allow optional early access without failing the whole section
    )

    ops = extracted.resort_ops or ResortOperationsInfo()
    urls_ops = ensure_list(ops.urls_ops)
    urls_hours = ensure_list(ops.urls_hours)
    urls_early = ensure_list(ops.urls_early)

    # Require at least some resort URLs to ground claims (critical precondition for web-backed leaves)
    resort_sources_leaf = evaluator.add_custom_node(
        result=bool(urls_ops or urls_hours),
        id="Resort_Sources_Provided",
        desc="Resort operations/hours URLs are provided to support claims",
        parent=node,
        critical=True  # Gate the two critical leaves below
    )

    # Operating status (critical within this subtree)
    open_leaf = evaluator.add_leaf(
        id="Operating_Status",
        desc="Confirm that Whistler Blackcomb is open and operating during January 17-19, 2026",
        parent=node,
        critical=True
    )
    open_claim = "Whistler Blackcomb is open and operating on January 17–19, 2026."
    await evaluator.verify(
        claim=open_claim,
        node=open_leaf,
        sources=(urls_ops if urls_ops else urls_hours),
        additional_instruction="Verify via season dates or operating calendar/hours that mid-January 2026 is within the operating season."
    )

    # Lift hours (critical within this subtree)
    lift_time_str = (ops.base_lift_open_time or "").strip()
    lift_hours_leaf = evaluator.add_leaf(
        id="Lift_Hours",
        desc="Provide the typical opening time for base area lifts in mid-to-late January 2026",
        parent=node,
        critical=True
    )
    lift_claim = f"Base area lifts at Whistler Blackcomb typically open at {lift_time_str} in mid-to-late January."
    await evaluator.verify(
        claim=lift_claim,
        node=lift_hours_leaf,
        sources=(urls_hours if urls_hours else urls_ops),
        additional_instruction="Consult official hours pages; accept small variations (e.g., 8:15 vs 8:30) if clearly described as typical."
    )

    # Early access program (optional, non-critical)
    early_leaf = evaluator.add_leaf(
        id="Early_Access",
        desc="Identify if an early access program (First Tracks) is available and provide cost and upload time information",
        parent=node,
        critical=False
    )
    early_program = (ops.early_access_program_name or "an early morning ski access program").strip()
    early_cost = (ops.early_access_cost or "").strip()
    early_upload = (ops.early_access_upload_time or "").strip()
    early_claim = f"Whistler offers {early_program} with a cost of {early_cost} and an upload time around {early_upload}."
    await evaluator.verify(
        claim=early_claim,
        node=early_leaf,
        sources=(urls_early if urls_early else urls_hours),
        additional_instruction="Verify if any early-access/First Tracks program exists and states cost and upload (lift) time. If unclear or not available, mark as not supported."
    )


async def verify_airport_lounges(
    evaluator: Evaluator,
    parent_node,
    extracted: TripPlanExtraction
) -> None:
    """
    Build and verify the Airport Lounge Access subtree.
    """
    node = evaluator.add_parallel(
        id="Airport_Lounge_Access",
        desc="Identify airport lounge options at Nashville International Airport (BNA) for American Express Platinum Card holders",
        parent=parent_node,
        critical=True
    )

    lounges = extracted.lounges or []
    # Must have at least two lounges with URLs
    at_least_two = sum(1 for l in lounges if l and l.name and l.urls) >= 2
    evaluator.add_custom_node(
        result=at_least_two,
        id="Available_Lounges",
        desc="Identify at least two specific airport lounges accessible at BNA",
        parent=node,
        critical=True
    )

    # Add detail checks for first two lounges found with URLs
    details_node = evaluator.add_parallel(
        id="Lounge_Details",
        desc="Verify locations and access requirements for at least two lounges at BNA",
        parent=node,
        critical=True
    )

    # Pick first two lounges with names and URLs
    valid_lounges = [l for l in lounges if l and l.name and (l.urls and len(l.urls) > 0)]
    while len(valid_lounges) < 2:
        valid_lounges.append(LoungeInfo())  # padding to avoid index errors

    for idx in range(2):
        lounge = valid_lounges[idx]
        lname = (lounge.name or "").strip()
        lloc = (lounge.terminal_location or "").strip()
        lpolicy = (lounge.access_policy or "").strip()
        lurls = lounge.urls if lounge.urls else []

        sub = evaluator.add_parallel(
            id=f"Lounge_{idx+1}",
            desc=f"Lounge #{idx+1}: {lname}",
            parent=details_node,
            critical=True
        )

        # Location verification
        loc_leaf = evaluator.add_leaf(
            id=f"Lounge_{idx+1}_Location",
            desc="Provide the terminal and gate area locations for the identified lounges at BNA",
            parent=sub,
            critical=True
        )
        loc_claim = f"The lounge '{lname}' at BNA is located at {lloc}."
        await evaluator.verify(
            claim=loc_claim,
            node=loc_leaf,
            sources=lurls,
            additional_instruction="Verify the stated concourse/terminal and nearby gate area on the cited page. Accept minor wording differences."
        )

        # Access policy verification
        access_leaf = evaluator.add_leaf(
            id=f"Lounge_{idx+1}_Access_Requirements",
            desc="Describe the access requirements or policies for Amex Platinum cardholders at BNA lounges",
            parent=sub,
            critical=True
        )
        access_claim = f"American Express Platinum cardholders can access the '{lname}' lounge at BNA under the following policy: {lpolicy}."
        await evaluator.verify(
            claim=access_claim,
            node=access_leaf,
            sources=lurls,
            additional_instruction="Verify that the cited page indicates access for Amex Platinum (directly or via partner rules such as Delta Sky Club rules, Priority Pass eligibility, etc.)."
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
    Evaluate an answer for the Whistler MLK 2026 trip planning task using the obj_task_eval framework.
    """
    # Initialize evaluator with a parallel root (root is always non-critical in the framework)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_trip_plan(),
        template_class=TripPlanExtraction,
        extraction_name="trip_plan_extraction"
    )

    # Build and verify each sub-tree
    await verify_flight_routing(evaluator, root, extracted)
    await verify_ground_transport(evaluator, root, extracted)
    await verify_resort_operations(evaluator, root, extracted)
    await verify_airport_lounges(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()