import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bdl_budget_routes"
TASK_DESCRIPTION = (
    "Identify three direct budget airline routes currently operated from Hartford Bradley International Airport (BDL) in Connecticut to three different U.S. states. "
    "For each route, provide: (1) the specific budget airline operating the service (must be one of: Avelo Airlines, Breeze Airways, Frontier Airlines, Spirit Airlines, or Sun Country Airlines), "
    "(2) the destination airport's three-letter IATA code, (3) the destination state, and (4) an official reference URL from the airline's or airport's website confirming that this route is currently operational or officially scheduled."
)
AS_OF_TEXT = "February 2026"

ALLOWED_AIRLINES = {
    "Avelo Airlines",
    "Breeze Airways",
    "Frontier Airlines",
    "Spirit Airlines",
    "Sun Country Airlines",
}

# US States (2-letter code to full name map)
US_STATES_MAP = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}
US_STATE_NAMES_TO_CODE = {v.lower(): k for k, v in US_STATES_MAP.items()}


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def normalize_airline_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.strip().lower()
    if "avelo" in n:
        return "Avelo Airlines"
    if "breeze" in n:
        return "Breeze Airways"
    if "frontier" in n:
        return "Frontier Airlines"
    if "spirit" in n:
        return "Spirit Airlines"
    if "sun country" in n:
        return "Sun Country Airlines"
    return None  # Unknown / not in allowed list


def is_iata_code(candidate: Optional[str]) -> bool:
    if not candidate:
        return False
    return re.fullmatch(r"[A-Za-z]{3}", candidate.strip()) is not None


def normalize_state_to_code(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip()
    # Try code first
    if len(s) == 2 and s.upper() in US_STATES_MAP:
        return s.upper()
    # Try full name
    key = s.lower()
    # Handle common punctuation removal
    key = re.sub(r"[^\w\s]", "", key).strip()
    if key in US_STATE_NAMES_TO_CODE:
        return US_STATE_NAMES_TO_CODE[key]
    return None


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class RouteItem(BaseModel):
    airline: Optional[str] = None
    destination_iata: Optional[str] = None
    destination_state: Optional[str] = None
    official_url: Optional[str] = None
    # Optional: if the answer explicitly mentions departure IATA, extract it (may be null)
    departure_iata: Optional[str] = None


class RoutesExtraction(BaseModel):
    routes: List[RouteItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_routes() -> str:
    return """
    Extract all budget airline route entries mentioned in the answer text that involve Hartford Bradley International Airport (BDL).
    For EACH route mentioned, return an object with the following fields:
    1) airline: the airline operating the route, as written in the answer (e.g., "Avelo Airlines", "Breeze Airways", "Frontier Airlines", "Spirit Airlines", or "Sun Country Airlines"). If only a short brand form is used (e.g., "Frontier"), extract it exactly as written.
    2) destination_iata: the destination airport’s three-letter IATA code as provided in the answer (e.g., "MCO", "TPA"). If not provided, set to null.
    3) destination_state: the destination state (full name like "Florida" or two-letter code like "FL") as provided in the answer. If not provided, set to null.
    4) official_url: the URL from the airline’s or airport’s official website provided in the answer that confirms the route (e.g., airline route page, airport destinations page, or press release). If multiple are provided, pick the most relevant single URL. If none is provided, set to null.
    5) departure_iata: the departure airport IATA code if explicitly stated in the answer for this route (e.g., "BDL"); else null.

    Important extraction rules:
    - Extract only what is explicitly present in the answer. Do not infer missing fields.
    - For URLs, only extract full valid URLs that are explicitly present in the answer.
    - Include ALL routes mentioned in the answer, even if there are more than three. We will filter later.
    """


# --------------------------------------------------------------------------- #
# Verification for each route                                                 #
# --------------------------------------------------------------------------- #
async def verify_single_route(
    evaluator: Evaluator,
    parent_node,
    route: RouteItem,
    index: int,
) -> None:
    """
    Build verification sub-tree for a single route.
    """
    rid = index + 1
    route_node = evaluator.add_parallel(
        id=f"Route_{rid}",
        desc=f"Route {rid} details and verification.",
        parent=parent_node,
        critical=False  # Keep route-level non-critical to allow partial credit across three routes
    )

    # Precompute normalized elements
    airline_norm = normalize_airline_name(route.airline) if route.airline else None
    state_code = normalize_state_to_code(route.destination_state)
    dest_iata = (route.destination_iata or "").upper().strip() if route.destination_iata else None
    url = route.official_url.strip() if route.official_url else None

    # 1) Official reference URL presence (critical prerequisite for many URL-grounded checks)
    url_present_node = evaluator.add_custom_node(
        result=bool(url),
        id=f"Route_{rid}_Official_URL_Provided",
        desc="Official reference URL is provided in the answer.",
        parent=route_node,
        critical=True
    )

    # 2) Departs from BDL (URL-grounded)
    dep_bdl_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_Departure_BDL",
        desc="Departs from Hartford Bradley International Airport (BDL).",
        parent=route_node,
        critical=True
    )
    await evaluator.verify(
        claim="This official page confirms a route between Hartford Bradley International Airport (BDL) and the stated destination airport. "
              "It should show BDL either as origin or as part of the route pair (BDL ↔ destination).",
        node=dep_bdl_leaf,
        sources=url,
        additional_instruction="Accept if the page shows a route pair including BDL and the destination airport (or clearly states service from Hartford/BDL). "
                               "Check both text and screenshot. Minor naming variants (e.g., 'Hartford' for BDL) are acceptable.",
        extra_prerequisites=[url_present_node]
    )

    # 3) Airline is one of the budget carriers (logic check)
    budget_leaf = evaluator.add_custom_node(
        result=(airline_norm in ALLOWED_AIRLINES) if airline_norm else False,
        id=f"Route_{rid}_Budget_Airline",
        desc="Operated by one of: Avelo Airlines, Breeze Airways, Frontier Airlines, Spirit Airlines, or Sun Country Airlines.",
        parent=route_node,
        critical=True
    )

    # 4) Non-stop / direct (URL-grounded)
    direct_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_Direct_Flight",
        desc="Is a direct/non-stop flight (no connections).",
        parent=route_node,
        critical=True
    )
    await evaluator.verify(
        claim="This route between BDL and the destination is a non-stop (direct) flight.",
        node=direct_leaf,
        sources=url,
        additional_instruction="Look for keywords like 'Nonstop', 'Direct', 'No connections', or route listings that imply non-stop service. "
                               "If the official airline route page lists it as nonstop or direct, consider it supported.",
        extra_prerequisites=[url_present_node]
    )

    # 5) Operational or officially scheduled as of February 2026 (URL-grounded)
    operational_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_Operational_Status",
        desc=f"Currently operational or officially scheduled to begin service as of {AS_OF_TEXT}.",
        parent=route_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"As of {AS_OF_TEXT}, this route is currently operational or officially scheduled according to the linked official page.",
        node=operational_leaf,
        sources=url,
        additional_instruction=f"Accept if the airline or airport official page explicitly lists the route in current destinations or shows a schedule/press release indicating ongoing or upcoming service around {AS_OF_TEXT}. "
                               f"If the page clearly indicates the route has ended or no longer operates, mark as not supported.",
        extra_prerequisites=[url_present_node]
    )

    # 6) Destination airport IATA provided (existence/format check)
    iata_leaf = evaluator.add_custom_node(
        result=is_iata_code(dest_iata),
        id=f"Route_{rid}_Destination_Airport_IATA",
        desc="Destination airport's three-letter IATA code is provided.",
        parent=route_node,
        critical=True
    )

    # 7) Destination state provided (existence) – split from validity
    state_provided_leaf = evaluator.add_custom_node(
        result=bool(route.destination_state and route.destination_state.strip()),
        id=f"Route_{rid}_Destination_State_Provided",
        desc="Destination state is provided in the answer.",
        parent=route_node,
        critical=True
    )

    # 8) Destination state is a U.S. state (validity)
    state_valid_leaf = evaluator.add_custom_node(
        result=bool(state_code and state_code in US_STATES_MAP),
        id=f"Route_{rid}_Destination_State_Is_US_State",
        desc="Destination state is a valid U.S. state.",
        parent=route_node,
        critical=True
    )

    # 9) Destination state is not Connecticut
    not_ct_leaf = evaluator.add_custom_node(
        result=(state_code is not None and state_code != "CT"),
        id=f"Route_{rid}_Destination_State_Not_CT",
        desc="Destination state is not Connecticut.",
        parent=route_node,
        critical=True
    )

    # 10) Destination is a commercial airport with TSA checkpoint (URL-grounded; infer from scheduled airline service)
    tsa_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_TSA_Commercial_Airport",
        desc="Destination is a commercial airport with a TSA security checkpoint.",
        parent=route_node,
        critical=True
    )
    tsa_claim = (
        "The destination airport is a commercial U.S. airport with TSA passenger security screening."
    )
    await evaluator.verify(
        claim=tsa_claim,
        node=tsa_leaf,
        sources=url,
        additional_instruction="If this is an airline or airport official page listing scheduled passenger service to the destination airport, "
                               "it's reasonable to conclude the destination is a TSA-screened commercial airport. "
                               "Accept if the page clearly shows scheduled air service to that airport.",
        extra_prerequisites=[url_present_node]
    )

    # 11) Official reference URL: verify it's from the airline's or airport's official website (URL-grounded)
    official_site_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_Official_Reference_URL_Is_Official",
        desc="Reference URL is from the airline's or airport's official website.",
        parent=route_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This URL is an official website of either the airline ('{route.airline or ''}') or Hartford Bradley International Airport.",
        node=official_site_leaf,
        sources=url,
        additional_instruction="Check the domain and branding to ensure it is an official site (e.g., airline brand domain or the airport's official site). "
                               "Reject third-party travel blogs or booking aggregators.",
        extra_prerequisites=[url_present_node]
    )

    # 12) Official reference URL confirms the route specifically (URL-grounded)
    url_confirms_route_leaf = evaluator.add_leaf(
        id=f"Route_{rid}_Official_Reference_URL_Confirms_Route",
        desc="Official reference URL confirms this specific BDL → destination route.",
        parent=route_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This official page explicitly confirms a route between BDL and the destination airport{(' (' + dest_iata + ')') if dest_iata else ''}.",
        node=url_confirms_route_leaf,
        sources=url,
        additional_instruction="Look for the destination listed on a BDL route list, a destinations page showing BDL ↔ destination, or a press release/schedule confirming service.",
        extra_prerequisites=[url_present_node]
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
    Evaluate an answer for the Hartford BDL budget routes task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Routes are evaluated independently
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

    # Extract routes from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_routes(),
        template_class=RoutesExtraction,
        extraction_name="routes_extraction",
    )

    # Record helpful reference info
    evaluator.add_custom_info(
        info={"allowed_budget_airlines": sorted(list(ALLOWED_AIRLINES)), "as_of": AS_OF_TEXT},
        info_type="reference_info",
        info_name="constraints_reference"
    )

    # --------------------- Global set-level constraints --------------------- #
    # We keep this node critical per rubric to gate the overall result.
    set_constraints_node = evaluator.add_parallel(
        id="Global_Set_Constraints",
        desc="Constraints that apply to the set of all three routes.",
        parent=root,
        critical=True
    )

    # Exactly three routes provided in the answer (no more, no fewer).
    total_routes_reported = len(extracted.routes)
    exactly_three_leaf = evaluator.add_custom_node(
        result=(total_routes_reported == 3),
        id="Exactly_Three_Routes_Provided",
        desc="Response provides exactly three routes (no more, no fewer).",
        parent=set_constraints_node,
        critical=True
    )

    # Select first three for detailed verification (per general guidance)
    selected_routes: List[RouteItem] = extracted.routes[:3]

    # All destination states are distinct across the three (checking normalized codes)
    if len(selected_routes) == 3:
        codes = [normalize_state_to_code(rt.destination_state) for rt in selected_routes]
        all_three_present = all(c is not None for c in codes)
        are_distinct = len(set(codes)) == 3 if all_three_present else False
    else:
        all_three_present = False
        are_distinct = False

    states_distinct_leaf = evaluator.add_custom_node(
        result=(len(selected_routes) == 3 and all_three_present and are_distinct),
        id="All_Destination_States_Are_Distinct",
        desc="The three destination states are all different from each other.",
        parent=set_constraints_node,
        critical=True
    )

    # --------------------- Per-route verification --------------------------- #
    # Create placeholders if fewer than 3 to build a uniform tree (those routes will fail)
    while len(selected_routes) < 3:
        selected_routes.append(RouteItem())

    # Build verification for each of the three routes
    await verify_single_route(evaluator, root, selected_routes[0], 0)
    await verify_single_route(evaluator, root, selected_routes[1], 1)
    await verify_single_route(evaluator, root, selected_routes[2], 2)

    # Return the structured evaluation summary
    return evaluator.get_summary()