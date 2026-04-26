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
TASK_ID = "allegiant_md_wdw_plan"
TASK_DESCRIPTION = """
A family of four from the Baltimore, Maryland area is planning a budget-friendly vacation to Walt Disney World in Florida. Due to Allegiant Air's recent service changes, they need to identify a complete travel solution that meets the following requirements:

1. Flight: They must use Allegiant Air departing from a Maryland airport (given that Allegiant ended service at BWI in December 2024) to reach a Florida destination that provides access to Walt Disney World.

2. Pre-flight Accommodation: They need to stay one night at a hotel near their departure airport the evening before their early morning flight. This hotel must offer:
   - 24-hour complimentary airport shuttle service
   - Complimentary breakfast
   - Free or included parking for their vehicle while they travel

3. Disney Accommodation: At Walt Disney World, they require a Disney Value Resort that:
   - Provides Disney Skyliner access for convenient park transportation
   - Can accommodate their family of 4 in a standard room

Please provide: (a) the specific Allegiant Air route (departure airport code and Florida arrival airport code), (b) the name of one hotel near the departure airport that meets all the specified amenities, and (c) the name of one Disney World Value Resort that satisfies the Skyliner access and capacity requirements. Include reference URLs confirming each selection.
"""


# --------------------------------------------------------------------------- #
# Data model for structured extraction                                        #
# --------------------------------------------------------------------------- #
class PlanExtraction(BaseModel):
    # Flight/Route
    departure_airport_code: Optional[str] = None  # e.g., "HGR"
    arrival_airport_code: Optional[str] = None    # e.g., "SFB" or "PIE"
    route_support_urls: List[str] = Field(default_factory=list)  # URLs that show Allegiant operates the specified route
    wdw_access_statement: Optional[str] = None    # Answer text snippet that claims WDW access from arrival airport
    wdw_access_urls: List[str] = Field(default_factory=list)     # URLs supporting that the arrival airport provides access to WDW

    # Pre-flight hotel (near HGR)
    hotel_name: Optional[str] = None
    hotel_location: Optional[str] = None  # Address/city/area as stated in the answer
    one_night_pre_flight_statement: Optional[str] = None  # Text from the answer indicating one night before early flight
    near_hgr_statement: Optional[str] = None  # Text from the answer indicating near HGR / reasonable driving distance
    shuttle_24h_urls: List[str] = Field(default_factory=list)  # URLs supporting 24-hour complimentary airport shuttle
    breakfast_urls: List[str] = Field(default_factory=list)    # URLs supporting complimentary breakfast
    parking_urls: List[str] = Field(default_factory=list)      # URLs supporting free/included parking while traveling (park & fly)

    # Disney Value Resort (Skyliner + sleeps 4)
    resort_name: Optional[str] = None
    resort_value_category_urls: List[str] = Field(default_factory=list)  # URLs supporting "Value" resort category
    resort_skyliner_urls: List[str] = Field(default_factory=list)        # URLs supporting Skyliner access
    resort_sleeps4_urls: List[str] = Field(default_factory=list)         # URLs supporting standard room sleeps 4


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
    Extract exactly and only what is explicitly provided in the answer for the following fields.

    FLIGHT / ROUTE (Allegiant):
    - departure_airport_code: 3-letter IATA code for the Allegiant departure airport (as stated in the answer).
    - arrival_airport_code: 3-letter IATA code for the Allegiant arrival airport in Florida (as stated).
    - route_support_urls: all URL(s) cited that directly support that Allegiant Air operates the specified route between the two stated airport codes. Return an empty array if none are present.
    - wdw_access_statement: the sentence or short phrase in the answer that claims the arrival airport provides access to Walt Disney World (via ground transport/driving). Null if not present.
    - wdw_access_urls: all URL(s) that support access from the arrival airport to Walt Disney World (e.g., airport ground transportation pages, distance/drive info). Return empty array if none are present.

    PRE-FLIGHT HOTEL (near HGR):
    - hotel_name: the specific hotel’s name selected in the answer.
    - hotel_location: the location/address/city for the hotel as provided in the answer text.
    - one_night_pre_flight_statement: the text in the answer that explicitly states the stay is for one night before an early morning flight. Null if not present.
    - near_hgr_statement: the text in the answer that explicitly says the hotel is near HGR or within reasonable driving distance of HGR. Null if not present.
    - shuttle_24h_urls: URL(s) cited that support a 24-hour complimentary airport shuttle. Empty array if none.
    - breakfast_urls: URL(s) cited that support complimentary breakfast. Empty array if none.
    - parking_urls: URL(s) cited that support free or included parking during the trip (park-and-fly/long-term). Empty array if none.

    DISNEY VALUE RESORT (Skyliner + sleeps 4):
    - resort_name: the Disney resort selected in the answer.
    - resort_value_category_urls: URL(s) cited that support that the resort is a Walt Disney World "Value" category resort. Empty if none.
    - resort_skyliner_urls: URL(s) cited that support that the resort has Disney Skyliner access. Empty if none.
    - resort_sleeps4_urls: URL(s) cited that support that a standard room sleeps at least 4 guests. Empty if none.

    Rules:
    - Extract only from the provided answer text. Do not invent or infer any information or URLs.
    - For URLs, return only valid, explicit URLs mentioned (including those inside markdown links).
    - If a requested field is not present in the answer, return null for strings and empty arrays for URL lists.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_iata(code: Optional[str]) -> bool:
    if not code:
        return False
    c = code.strip().upper()
    return len(c) == 3 and c.isalpha()


def _norm(code: Optional[str]) -> str:
    return (code or "").strip().upper()


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_flight_route_checks(evaluator: Evaluator, parent_node, data: PlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Flight_Route",
        desc="Provide a valid Allegiant Air route from HGR to an allowed Florida airport and show it provides access to Walt Disney World, with citation(s).",
        parent=parent_node,
        critical=True
    )

    dep = _norm(data.departure_airport_code)
    arr = _norm(data.arrival_airport_code)

    # Route specified as airport codes (existence + format)
    evaluator.add_custom_node(
        result=(_is_iata(dep) and _is_iata(arr)),
        id="Route_Specified_As_Airport_Codes",
        desc="Provides the route as (departure airport code, arrival airport code).",
        parent=node,
        critical=True
    )

    # Departure must be HGR
    evaluator.add_custom_node(
        result=(dep == "HGR"),
        id="Departure_Airport_Is_HGR",
        desc="Departure airport code is HGR (per constraints).",
        parent=node,
        critical=True
    )

    # Arrival must be SFB or PIE
    evaluator.add_custom_node(
        result=(arr in {"SFB", "PIE"}),
        id="Arrival_Airport_Is_SFB_or_PIE",
        desc="Arrival airport code is either SFB or PIE (per constraints).",
        parent=node,
        critical=True
    )

    # Source presence gate for Allegiant route support
    evaluator.add_custom_node(
        result=_has_urls(data.route_support_urls),
        id="Route_Sources_Provided",
        desc="At least one source URL is provided for Allegiant operating the specified route.",
        parent=node,
        critical=True
    )

    # Verify Allegiant operates the route
    allegiant_leaf = evaluator.add_leaf(
        id="Cited_Evidence_Allegiant_Operates_Route",
        desc="Provides at least one reference URL that supports that Allegiant operates the specified HGR→(SFB/PIE) route.",
        parent=node,
        critical=True
    )
    route_claim = f"Allegiant Air operates (or has operated) flights between Hagerstown (HGR) and {arr}."
    await evaluator.verify(
        claim=route_claim,
        node=allegiant_leaf,
        sources=data.route_support_urls,
        additional_instruction="Accept seasonal or limited service if the source indicates Allegiant serves or has served HGR↔SFB/PIE. Valid sources include Allegiant's official site, airport route pages, or reputable news/airport pages."
    )

    # WDW access source presence gate
    evaluator.add_custom_node(
        result=_has_urls(data.wdw_access_urls),
        id="WDW_Access_Sources_Provided",
        desc="At least one source URL is provided indicating the arrival airport provides access to Walt Disney World.",
        parent=node,
        critical=True
    )

    # Verify that chosen arrival airport provides access to WDW
    wdw_access_leaf = evaluator.add_leaf(
        id="WDW_Access_Explained",
        desc="States that the chosen arrival airport provides access to Walt Disney World (e.g., via ground transport/driving).",
        parent=node,
        critical=True
    )
    access_claim = f"From {arr}, travelers can reach Walt Disney World Resort via typical ground transportation (e.g., shuttles, ride share, rental car, or reasonable driving distance)."
    await evaluator.verify(
        claim=access_claim,
        node=wdw_access_leaf,
        sources=data.wdw_access_urls,
        additional_instruction="The cited page(s) should mention transportation options or distance/drive times from the airport to Walt Disney World/Lake Buena Vista/Disney area."
    )


async def build_pre_flight_hotel_checks(evaluator: Evaluator, parent_node, data: PlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Pre_Flight_Hotel",
        desc="Select one hotel for the night before the flight that is near HGR (or the answer explicitly claims it is within reasonable driving distance) and meets shuttle/breakfast/parking needs, with citations.",
        parent=parent_node,
        critical=True
    )

    # Hotel name provided
    evaluator.add_custom_node(
        result=bool(data.hotel_name and data.hotel_name.strip()),
        id="Hotel_Name_Provided",
        desc="Provides the name of one specific pre-flight hotel.",
        parent=node,
        critical=True
    )

    # One night pre-flight stated (verify against the answer text)
    one_night_leaf = evaluator.add_leaf(
        id="One_Night_Pre_Flight_Stated",
        desc="States this hotel stay is for one night on the evening before the early morning flight.",
        parent=node,
        critical=True
    )
    one_night_claim = "The answer explicitly states that the hotel stay is for one night on the evening before an early morning flight."
    await evaluator.verify(
        claim=one_night_claim,
        node=one_night_leaf,
        additional_instruction="Scan the answer text for wording such as 'one night' and 'before the early morning flight'. Minor paraphrases are acceptable."
    )

    # Hotel location provided AND near HGR claimed (verify against the answer text)
    location_near_leaf = evaluator.add_leaf(
        id="Hotel_Location_Provided_And_Near_HGR_Claimed",
        desc="Provides the hotel location (address/city) and explicitly indicates it is near HGR or within reasonable driving distance (as stated in constraints).",
        parent=node,
        critical=True
    )
    loc_text = (data.hotel_location or "").strip()
    near_text = (data.near_hgr_statement or "").strip()
    location_claim = (
        f"The answer provides the hotel's location (e.g., '{loc_text}') and explicitly states it is near HGR "
        f"(Hagerstown Regional Airport) or within a reasonable driving distance (e.g., '{near_text}')."
    )
    await evaluator.verify(
        claim=location_claim,
        node=location_near_leaf,
        additional_instruction="This is an answer-text check. Confirm both elements: (1) a location/address/city is provided for the hotel and (2) the answer explicitly mentions 'near HGR' or 'within reasonable driving distance of HGR'."
    )

    hotel_name = data.hotel_name or "the selected hotel"

    # Shuttle sources gate and verification
    evaluator.add_custom_node(
        result=_has_urls(data.shuttle_24h_urls),
        id="Shuttle_Sources_Provided",
        desc="At least one source URL is provided for 24-hour complimentary airport shuttle.",
        parent=node,
        critical=True
    )
    shuttle_leaf = evaluator.add_leaf(
        id="Cited_24h_Complimentary_Shuttle",
        desc="Provides reference URL(s) indicating the hotel offers a 24-hour complimentary airport shuttle service.",
        parent=node,
        critical=True
    )
    shuttle_claim = f"{hotel_name} offers a 24-hour complimentary airport shuttle service."
    await evaluator.verify(
        claim=shuttle_claim,
        node=shuttle_leaf,
        sources=data.shuttle_24h_urls,
        additional_instruction="Accept synonyms like '24-hour shuttle', '24/7 airport shuttle', provided it is complimentary/free and airport-related."
    )

    # Breakfast sources gate and verification
    evaluator.add_custom_node(
        result=_has_urls(data.breakfast_urls),
        id="Breakfast_Sources_Provided",
        desc="At least one source URL is provided for complimentary breakfast.",
        parent=node,
        critical=True
    )
    breakfast_leaf = evaluator.add_leaf(
        id="Cited_Complimentary_Breakfast",
        desc="Provides reference URL(s) indicating the hotel provides complimentary breakfast.",
        parent=node,
        critical=True
    )
    breakfast_claim = f"{hotel_name} provides complimentary breakfast (free or included)."
    await evaluator.verify(
        claim=breakfast_claim,
        node=breakfast_leaf,
        sources=data.breakfast_urls,
        additional_instruction="Accept language like 'complimentary breakfast', 'free breakfast', or 'breakfast included'. Continental or hot breakfast both count."
    )

    # Parking sources gate and verification (park-and-fly / included parking while traveling)
    evaluator.add_custom_node(
        result=_has_urls(data.parking_urls),
        id="Parking_Sources_Provided",
        desc="At least one source URL is provided for free/included parking during the trip.",
        parent=node,
        critical=True
    )
    parking_leaf = evaluator.add_leaf(
        id="Cited_Free_or_Included_Parking_During_Trip",
        desc="Provides reference URL(s) indicating the hotel offers free parking or parking included in the rate for leaving the vehicle during the trip.",
        parent=node,
        critical=True
    )
    parking_claim = f"{hotel_name} offers free or included parking for guests to leave their vehicle during their trip (e.g., park-and-fly/long-term parking)."
    await evaluator.verify(
        claim=parking_claim,
        node=parking_leaf,
        sources=data.parking_urls,
        additional_instruction="Look for 'park & fly', 'park sleep fly', 'free long-term parking', or similar language that clearly allows leaving the car during the trip."
    )


async def build_disney_value_resort_checks(evaluator: Evaluator, parent_node, data: PlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Disney_Value_Resort",
        desc="Select one Disney Value Resort with Disney Skyliner access that can sleep 4 in a standard room, with citations.",
        parent=parent_node,
        critical=True
    )

    resort = data.resort_name or ""

    # Resort name provided
    evaluator.add_custom_node(
        result=bool(resort.strip()),
        id="Resort_Name_Provided",
        desc="Provides the name of one specific Disney resort.",
        parent=node,
        critical=True
    )

    # Value category
    evaluator.add_custom_node(
        result=_has_urls(data.resort_value_category_urls),
        id="Value_Category_Sources_Provided",
        desc="At least one source URL is provided indicating the resort is in the Value category.",
        parent=node,
        critical=True
    )
    value_leaf = evaluator.add_leaf(
        id="Cited_Value_Category",
        desc="Provides reference URL(s) indicating the resort is in Disney World's Value Resort category.",
        parent=node,
        critical=True
    )
    value_claim = f"{resort} is a Walt Disney World Value Resort."
    await evaluator.verify(
        claim=value_claim,
        node=value_leaf,
        sources=data.resort_value_category_urls,
        additional_instruction="Prefer official Disney site pages; reputable planning sites are acceptable if clear."
    )

    # Skyliner access
    evaluator.add_custom_node(
        result=_has_urls(data.resort_skyliner_urls),
        id="Skyliner_Sources_Provided",
        desc="At least one source URL is provided indicating the resort has Disney Skyliner access.",
        parent=node,
        critical=True
    )
    skyliner_leaf = evaluator.add_leaf(
        id="Cited_Skyliner_Access",
        desc="Provides reference URL(s) indicating the resort has Disney Skyliner access.",
        parent=node,
        critical=True
    )
    skyliner_claim = f"{resort} has access to the Disney Skyliner gondola system."
    await evaluator.verify(
        claim=skyliner_claim,
        node=skyliner_leaf,
        sources=data.resort_skyliner_urls,
        additional_instruction="The page should state that the resort is on or connected to the Skyliner line or station."
    )

    # Standard room sleeps 4
    evaluator.add_custom_node(
        result=_has_urls(data.resort_sleeps4_urls),
        id="Sleeps4_Sources_Provided",
        desc="At least one source URL is provided indicating a standard room sleeps 4 guests.",
        parent=node,
        critical=True
    )
    sleeps4_leaf = evaluator.add_leaf(
        id="Cited_Standard_Room_Sleeps_4",
        desc="Provides reference URL(s) indicating the resort can accommodate at least 4 guests in a standard room configuration.",
        parent=node,
        critical=True
    )
    sleeps4_claim = f"A standard room at {resort} accommodates at least 4 guests."
    await evaluator.verify(
        claim=sleeps4_claim,
        node=sleeps4_leaf,
        sources=data.resort_sleeps4_urls,
        additional_instruction="Verify that a standard room configuration (non-suite) at this resort sleeps 4 guests (e.g., 2 queen/double beds)."
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
    Evaluate an answer for the Allegiant Maryland to WDW travel planning task.

    Returns:
        A structured dict containing the final score and the full verification tree.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level: parallel aggregation of the 3 major parts
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

    # Create a top-level critical node to enforce all three components
    task_node = evaluator.add_parallel(
        id="Travel_Planning_Task",
        desc="Provide a complete travel solution: (a) an Allegiant route from the required Maryland airport to an allowed Florida airport, (b) one qualifying pre-flight hotel, and (c) one qualifying Disney Value Resort with Skyliner access; include reference URL(s) that substantiate each required property.",
        parent=root,
        critical=True
    )

    # 1) Extract structured info from the answer
    plan_data = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction"
    )

    # Optional: record quick summary of extracted key fields
    evaluator.add_custom_info(
        {
            "departure_airport_code": plan_data.departure_airport_code,
            "arrival_airport_code": plan_data.arrival_airport_code,
            "hotel_name": plan_data.hotel_name,
            "resort_name": plan_data.resort_name
        },
        info_type="extracted_overview",
        info_name="extracted_overview"
    )

    # 2) Build verification subtrees
    await build_flight_route_checks(evaluator, task_node, plan_data)
    await build_pre_flight_hotel_checks(evaluator, task_node, plan_data)
    await build_disney_value_resort_checks(evaluator, task_node, plan_data)

    # 3) Return evaluation summary
    return evaluator.get_summary()