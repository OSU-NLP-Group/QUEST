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
TASK_ID = "ga_to_orlando_avelo_digitalid_stadium"
TASK_DESCRIPTION = (
    "A traveler residing in Atlanta, Georgia holds a Georgia-issued mobile driver's license (digital ID) and wants to attend a major sporting event at Camping World Stadium in Orlando, Florida. "
    "They prefer to fly with a low-cost carrier and are considering Avelo Airlines. Please provide the following information: "
    "(1) Route Information: Identify the specific Avelo Airlines nonstop route (departure city and arrival airport code) that connects Georgia to the Orlando area. "
    "(2) Digital ID Verification: Confirm whether Georgia residents can use their state-issued digital ID for TSA screening at airports, and identify which digital wallet platforms (Apple Wallet, Google Wallet, and/or Samsung Wallet) support Georgia's digital ID. "
    "(3) Venue Details: Provide the exact location (city and state) of Camping World Stadium and its seating capacity. "
    "(4) Airline Information: Verify whether Avelo Airlines is classified as a low-cost carrier and confirm whether the arrival airport is one of Avelo's hub airports. "
    "For each piece of information, provide supporting reference URLs from official or reliable sources."
)

# Ground-truth expectations (as implied by rubric)
EXPECTED_DEPARTURE_CITY = "Atlanta, Georgia"
EXPECTED_ARRIVAL_AIRPORT_CODE = "LAL"
EXPECTED_STADIUM_CITY = "Orlando"
EXPECTED_STADIUM_STATE = "Florida"
EXPECTED_CAPACITY_RANGE = "60,000-66,000 (approx. 65,000)"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RouteInfo(BaseModel):
    departure_city: Optional[str] = None
    arrival_airport_code: Optional[str] = None
    arrival_airport_name: Optional[str] = None
    route_reference_urls: List[str] = Field(default_factory=list)


class DigitalIDInfo(BaseModel):
    tsa_acceptance_claim: Optional[str] = None
    platforms: List[str] = Field(default_factory=list)
    tsa_reference_urls: List[str] = Field(default_factory=list)
    platform_reference_urls: List[str] = Field(default_factory=list)


class VenueInfo(BaseModel):
    stadium_city: Optional[str] = None
    stadium_state: Optional[str] = None
    seating_capacity: Optional[str] = None
    venue_reference_urls: List[str] = Field(default_factory=list)
    proximity_reference_urls: List[str] = Field(default_factory=list)


class AirlineInfo(BaseModel):
    low_cost_carrier_claim: Optional[str] = None
    hub_claim: Optional[str] = None
    airline_reference_urls: List[str] = Field(default_factory=list)


class TravelPlanExtraction(BaseModel):
    route: Optional[RouteInfo] = None
    digital_id: Optional[DigitalIDInfo] = None
    venue: Optional[VenueInfo] = None
    airline: Optional[AirlineInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_travel_plan() -> str:
    return """
    Extract the following structured information exactly as presented in the answer. Do not infer any information that is not explicitly stated.

    1) Route Information (Avelo Airlines Georgia→Orlando area)
       - route.departure_city: The departure city in Georgia (include state if present).
       - route.arrival_airport_code: The 3-letter IATA code for the arrival airport (e.g., LAL, MCO, SFB).
       - route.arrival_airport_name: The name of the arrival airport if provided.
       - route.route_reference_urls: All URLs cited that support the Avelo route operation (official airline pages, news, schedules, airport pages, etc.).

    2) Digital ID Verification (Georgia mobile driver's license)
       - digital_id.tsa_acceptance_claim: The statement or conclusion about Georgia digital IDs being accepted by TSA for airport screening (verbatim or summarized).
       - digital_id.platforms: A list of the wallet platforms named as supporting Georgia digital ID (e.g., "Apple Wallet", "Google Wallet", "Samsung Wallet"). Use the platform names exactly as mentioned in the answer.
       - digital_id.tsa_reference_urls: All TSA.gov URLs (or other cited official sources) provided to support TSA acceptance of Georgia’s digital ID.
       - digital_id.platform_reference_urls: All URLs provided that support the platform support claim(s) (Apple, Google, Samsung, or Georgia DDS).

    3) Venue Details (Camping World Stadium)
       - venue.stadium_city: The city of Camping World Stadium.
       - venue.stadium_state: The state of Camping World Stadium.
       - venue.seating_capacity: The seating capacity as stated (keep it as a string).
       - venue.venue_reference_urls: All URLs cited that confirm stadium location and/or capacity.
       - venue.proximity_reference_urls: Any URLs provided that support proximity or service relevance between the arrival airport and Orlando.

    4) Airline Information (Avelo)
       - airline.low_cost_carrier_claim: The statement about Avelo Airlines classification as a low-cost carrier.
       - airline.hub_claim: The statement about whether the arrival airport is one of Avelo’s hub airports.
       - airline.airline_reference_urls: All URLs cited to support Avelo’s classification and hubs.

    Return a JSON object with keys: route, digital_id, venue, airline.
    For any missing data, return null or empty lists accordingly.
    Ensure URLs are valid and include the protocol (prepend http:// if missing).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _filter_urls_by_domain(urls: List[str], domain_substr: str) -> List[str]:
    if not urls:
        return []
    domain_substr_lower = domain_substr.lower()
    return [u for u in urls if isinstance(u, str) and domain_substr_lower in u.lower()]


def _pick_supported_platform(platforms: List[str]) -> Optional[str]:
    """
    Pick the first recognized platform among Apple Wallet, Google Wallet, Samsung Wallet.
    Returns the normalized platform name or None if not found.
    """
    if not platforms:
        return None
    for p in platforms:
        if not isinstance(p, str):
            continue
        pl = p.strip().lower()
        if "apple" in pl:
            return "Apple Wallet"
        if "google" in pl or "android" in pl:
            return "Google Wallet"
        if "samsung" in pl:
            return "Samsung Wallet"
    return None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_route(evaluator: Evaluator, parent, data: TravelPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Route_Identification",
        desc="Correct identification of the Avelo Airlines route from Georgia to the Orlando area",
        parent=parent,
        critical=True  # All children under here must be critical
    )

    route = data.route or RouteInfo()

    # Departure City: must be Atlanta, GA
    dep_city_node = evaluator.add_leaf(
        id="Departure_City",
        desc="The departure city is Atlanta, Georgia",
        parent=node,
        critical=True
    )
    dep_city = route.departure_city or ""
    dep_claim = f"The departure city provided in the answer ('{dep_city}') refers to Atlanta, Georgia."
    await evaluator.verify(
        claim=dep_claim,
        node=dep_city_node,
        additional_instruction="Treat 'Atlanta' or 'ATL' as Atlanta, Georgia; be case-insensitive. Fail if it is a different Georgia city."
    )

    # Arrival Airport: must be LAL (Lakeland)
    arr_airport_node = evaluator.add_leaf(
        id="Arrival_Airport",
        desc="The arrival airport is Lakeland International Airport (LAL)",
        parent=node,
        critical=True
    )
    arr_code = (route.arrival_airport_code or "").strip()
    arr_name = route.arrival_airport_name or ""
    arr_claim = (
        f"The arrival airport provided in the answer corresponds to Lakeland (IATA code 'LAL'). "
        f"Extracted code: '{arr_code}'. Extracted name: '{arr_name}'."
    )
    await evaluator.verify(
        claim=arr_claim,
        node=arr_airport_node,
        additional_instruction="Pass if the arrival airport code equals 'LAL', allowing name variants like 'Lakeland Linder International Airport'."
    )

    # Airline operates this nonstop route - verify via provided route URLs
    operates_node = evaluator.add_leaf(
        id="Airline_Operates_Route",
        desc="Avelo Airlines operates this nonstop route",
        parent=node,
        critical=True
    )
    operates_claim = (
        f"Avelo Airlines operates a nonstop route between {dep_city if dep_city else 'the stated Georgia city'} "
        f"and {arr_code if arr_code else (arr_name or 'the stated arrival airport')}."
    )
    await evaluator.verify(
        claim=operates_claim,
        node=operates_node,
        sources=route.route_reference_urls,
        additional_instruction="Accept evidence from official airline pages, schedules, airport pages, or reliable news confirming Avelo operates this nonstop route."
    )

    # Existence of at least one route reference URL
    evaluator.add_custom_node(
        result=bool(route.route_reference_urls),
        id="Route_Reference_URL",
        desc="Provide a reference URL confirming the route operation",
        parent=node,
        critical=True
    )


async def verify_digital_id(evaluator: Evaluator, parent, data: TravelPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Digital_ID_Eligibility",
        desc="Verification of digital ID eligibility for the traveler",
        parent=parent,
        critical=False  # Allow non-critical children under this group
    )
    di = data.digital_id or DigitalIDInfo()

    # Home State Acceptance (critical): Georgia issues TSA-accepted digital IDs
    home_accept_node = evaluator.add_leaf(
        id="Home_State_Acceptance",
        desc="Georgia is confirmed as a state that issues TSA-accepted digital IDs",
        parent=node,
        critical=True
    )
    acceptance_claim = (
        "Georgia's state-issued digital ID (mobile driver's license) is accepted by TSA for screening at participating airports."
    )
    acceptance_sources = []
    if di.tsa_reference_urls:
        acceptance_sources.extend(di.tsa_reference_urls)
    if di.platform_reference_urls:
        # Sometimes platform pages or GA DDS pages also explicitly mention TSA acceptance
        acceptance_sources.extend(di.platform_reference_urls)
    await evaluator.verify(
        claim=acceptance_claim,
        node=home_accept_node,
        sources=acceptance_sources,
        additional_instruction="Pass if the source(s) explicitly indicate Georgia's mobile driver's license is TSA-accepted at participating airports."
    )

    # Digital ID Platform (non-critical): At least one wallet platform supports Georgia
    platform_node = evaluator.add_leaf(
        id="Digital_ID_Platform",
        desc="Identification of at least one platform (Apple Wallet, Google Wallet, or Samsung Wallet) that supports Georgia digital IDs",
        parent=node,
        critical=False
    )
    chosen_platform = _pick_supported_platform(di.platforms)
    if chosen_platform:
        platform_claim = f"Georgia's digital ID is supported in {chosen_platform}."
    else:
        platform_claim = "At least one of Apple Wallet, Google Wallet, or Samsung Wallet supports Georgia's digital ID."
    await evaluator.verify(
        claim=platform_claim,
        node=platform_node,
        sources=di.platform_reference_urls,
        additional_instruction="Pass only if the cited page(s) explicitly show Georgia as supported for the specified wallet platform."
    )

    # TSA Acceptance Reference URL from TSA.gov (critical)
    tsa_url_node = evaluator.add_leaf(
        id="TSA_Acceptance_Reference_URL",
        desc="Provide a reference URL from TSA.gov confirming Georgia's participation in digital ID program",
        parent=node,
        critical=True
    )
    tsa_urls = _filter_urls_by_domain(di.tsa_reference_urls, "tsa.gov")
    tsa_claim = "This TSA.gov page confirms that Georgia participates in the digital ID program accepted for TSA screening."
    await evaluator.verify(
        claim=tsa_claim,
        node=tsa_url_node,
        sources=tsa_urls,
        additional_instruction="Pass only if the TSA.gov page clearly references Georgia's participation or acceptance of Georgia mobile IDs."
    )


async def verify_venue(evaluator: Evaluator, parent, data: TravelPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Stadium_Venue_Information",
        desc="Accurate information about the destination venue",
        parent=parent,
        critical=False  # Allow a non-critical child (proximity)
    )
    venue = data.venue or VenueInfo()

    # Stadium Location (critical)
    location_node = evaluator.add_leaf(
        id="Stadium_Location",
        desc="Camping World Stadium is located in Orlando, Florida",
        parent=node,
        critical=True
    )
    city = venue.stadium_city or ""
    state = venue.stadium_state or ""
    location_claim = (
        f"The answer correctly states that Camping World Stadium is located in Orlando, Florida. "
        f"Extracted location: {city}, {state}."
    )
    await evaluator.verify(
        claim=location_claim,
        node=location_node,
        sources=venue.venue_reference_urls,
        additional_instruction="Pass if the cited source(s) clearly confirm the stadium is in Orlando, Florida."
    )

    # Stadium Capacity (critical) – approx 65,000 acceptable range 60k–66k
    capacity_node = evaluator.add_leaf(
        id="Stadium_Capacity",
        desc="The stadium's seating capacity is approximately 65,000 (acceptable range: 60,000-66,000)",
        parent=node,
        critical=True
    )
    capacity_str = venue.seating_capacity or ""
    capacity_claim = (
        f"Camping World Stadium's seating capacity is approximately 65,000 (within 60,000–66,000). "
        f"Extracted capacity: '{capacity_str}'."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_node,
        sources=venue.venue_reference_urls,
        additional_instruction="Pass if the provided source(s) show capacity about 65,000 (within 60,000–66,000). Minor variations are acceptable."
    )

    # Proximity to Airport (non-critical)
    proximity_node = evaluator.add_leaf(
        id="Proximity_to_Airport",
        desc="Acknowledgment that Lakeland Airport (LAL) serves the Orlando area or is within reasonable distance",
        parent=node,
        critical=False
    )
    prox_claim = (
        "Lakeland Linder International Airport (LAL) is within reasonable driving distance of the Orlando area or is described as serving the Orlando area."
    )
    prox_sources = venue.proximity_reference_urls if venue.proximity_reference_urls else venue.venue_reference_urls
    await evaluator.verify(
        claim=prox_claim,
        node=proximity_node,
        sources=prox_sources,
        additional_instruction="Pass if the source(s) explicitly describe LAL as serving Greater Orlando or show reasonable driving proximity to Orlando."
    )

    # Venue reference URL existence (critical)
    evaluator.add_custom_node(
        result=bool(venue.venue_reference_urls),
        id="Venue_Reference_URL",
        desc="Provide a reference URL confirming stadium location and capacity details",
        parent=node,
        critical=True
    )


async def verify_airline(evaluator: Evaluator, parent, data: TravelPlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Airline_Classification",
        desc="Correct classification of Avelo Airlines",
        parent=parent,
        critical=False
    )
    air = data.airline or AirlineInfo()

    # Low-cost carrier identification (non-critical)
    lcc_node = evaluator.add_leaf(
        id="Low_Cost_Carrier",
        desc="Avelo Airlines is identified as a low-cost carrier",
        parent=node,
        critical=False
    )
    lcc_claim = "Avelo Airlines is a low-cost carrier (LCC)."
    await evaluator.verify(
        claim=lcc_claim,
        node=lcc_node,
        sources=air.airline_reference_urls,
        additional_instruction="Pass if the source(s) classify Avelo as a low-cost carrier or ultra-low-cost carrier."
    )

    # Hub confirmation for arrival airport (non-critical)
    hub_node = evaluator.add_leaf(
        id="Hub_Confirmation",
        desc="Lakeland (LAL) is confirmed as one of Avelo's four hub airports",
        parent=node,
        critical=False
    )
    hub_claim = "Lakeland (LAL) is one of Avelo Airlines’ hub (base) airports."
    await evaluator.verify(
        claim=hub_claim,
        node=hub_node,
        sources=air.airline_reference_urls,
        additional_instruction="Pass only if the source(s) explicitly list LAL as an Avelo hub/base among its hubs."
    )

    # Airline reference URL existence (critical)
    evaluator.add_custom_node(
        result=bool(air.airline_reference_urls),
        id="Airline_Reference_URL",
        desc="Provide a reference URL confirming Avelo's classification and hub information",
        parent=node,
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
    Evaluate the answer for the Georgia-to-Orlando Avelo/Digital ID/Stadium task.
    """
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

    # Add top-level logical node (non-critical to allow partial credit across categories)
    top = evaluator.add_parallel(
        id="Travel_Plan_Verification",
        desc="Verification of a complete travel plan for a Georgia resident with digital ID traveling to an event at Camping World Stadium via Avelo Airlines",
        parent=root,
        critical=False
    )

    # Perform extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_travel_plan(),
        template_class=TravelPlanExtraction,
        extraction_name="travel_plan_extraction"
    )

    # Add ground truth expectations (from rubric)
    evaluator.add_ground_truth({
        "expected_departure_city": EXPECTED_DEPARTURE_CITY,
        "expected_arrival_airport_code": EXPECTED_ARRIVAL_AIRPORT_CODE,
        "expected_stadium_location": f"{EXPECTED_STADIUM_CITY}, {EXPECTED_STADIUM_STATE}",
        "expected_capacity_range": EXPECTED_CAPACITY_RANGE
    }, gt_type="ground_truth")

    # Build verification subtrees
    await verify_route(evaluator, top, extraction)
    await verify_digital_id(evaluator, top, extraction)
    await verify_venue(evaluator, top, extraction)
    await verify_airline(evaluator, top, extraction)

    return evaluator.get_summary()