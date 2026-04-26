import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fall_trip_orlando_maine_2026"
TASK_DESCRIPTION = (
    "You are planning a fall outdoor recreation trip in 2026 that combines attending a major outdoor music festival "
    "at Camping World Stadium in Orlando, Florida with outdoor activities in Maine during peak fall foliage season. "
    "Your trip must satisfy several requirements related to festival selection, Maine activities (including Acadia National Park), "
    "travel logistics, equipment and baggage planning for American Airlines, and Orlando accommodation near Camping World Stadium."
)

ALLOWED_FESTIVALS = [
    {"name": "Rolling Loud", "dates": "May 8–10, 2026"},
    {"name": "EDC Orlando", "dates": "November 6–8, 2026"},
    {"name": "Vans Warped Tour", "dates": "November 14–15, 2026"},
]
STADIUM_CAPACITY_APPROX = "approximately 65,000"

BAGGAGE_FEES = {
    "first_bag_usd": 40,
    "second_bag_usd": 45
}


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class FestivalInfo(BaseModel):
    festival_name: Optional[str] = None
    festival_dates: Optional[str] = None
    timing_justification: Optional[str] = None
    venue_capacity: Optional[str] = None
    ticket_category: Optional[str] = None  # e.g., "GA", "GA+", "VIP"
    ticket_price: Optional[str] = None     # keep as string to allow ranges like "$199+" or "from $299"
    official_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)  # any extra URLs cited for the festival/stadium


class MainePlan(BaseModel):
    acadia_visit_details: Optional[str] = None
    wildlife_timing_details: Optional[str] = None
    bangor_to_acadia_distance: Optional[str] = None  # e.g., "49 miles"
    bangor_to_acadia_drive_time: Optional[str] = None  # e.g., "1 hour"
    camping_accommodation_details: Optional[str] = None  # mention state park or Acadia campground + Oct 15 note
    activities_url: Optional[str] = None


class TravelInfo(BaseModel):
    airline: Optional[str] = None
    depart_airport_code: Optional[str] = None  # e.g., "BGR"
    arrival_airport_code: Optional[str] = None  # e.g., "MCO"
    travel_timeline_summary: Optional[str] = None


class EquipmentBaggage(BaseModel):
    equipment_list: List[str] = Field(default_factory=list)  # e.g., ["tent", "sleeping bag", "backpack", "hiking boots"]
    total_checked_bags_count: Optional[int] = None  # explicit count if the answer provides it
    total_baggage_fee_amount_usd: Optional[float] = None  # numeric fee amount if provided


class OrlandoHotel(BaseModel):
    hotel_name: Optional[str] = None
    distance_to_stadium: Optional[str] = None  # e.g., "2.3 miles"
    distance_to_stadium_miles_numeric: Optional[float] = None  # optional numeric parse if present
    hotel_url: Optional[str] = None


class TripExtraction(BaseModel):
    festival: Optional[FestivalInfo] = None
    maine: Optional[MainePlan] = None
    travel: Optional[TravelInfo] = None
    equipment: Optional[EquipmentBaggage] = None
    hotel: Optional[OrlandoHotel] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip() -> str:
    return """
    Extract structured information for this combined Orlando festival + Maine fall outdoor recreation trip plan from the answer text.

    FESTIVAL:
    - festival_name: the selected festival name (must be one of Rolling Loud, EDC Orlando, or Vans Warped Tour)
    - festival_dates: the dates given for the selected festival (e.g., "May 8–10, 2026")
    - timing_justification: the explanation of why the selected festival allows a mid-October Maine visit
    - venue_capacity: the stated seating capacity of Camping World Stadium (as text, e.g., "about 65,000" or "approximately 65,000")
    - ticket_category: the ticket category chosen (GA, GA+, or VIP)
    - ticket_price: the current price as stated (keep as string to allow ranges or symbols, e.g., "$199+", "from $299")
    - official_url: the official festival website URL
    - additional_urls: any other URLs cited for festival/stadium information (array)

    MAINE PLAN:
    - acadia_visit_details: text indicating a visit to Acadia National Park in mid-October
    - wildlife_timing_details: text describing October wildlife viewing timing (migratory birds, prep for winter, etc.)
    - bangor_to_acadia_distance: the distance from Bangor to Acadia National Park (text, e.g., "49 miles")
    - bangor_to_acadia_drive_time: the driving time (text, e.g., "1 hour")
    - camping_accommodation_details: camping/accommodation (state park or Acadia campground) with October 15 closure awareness
    - activities_url: a URL reference supporting Maine activities or Acadia information

    TRAVEL LOGISTICS:
    - airline: the airline for flights (should be American Airlines)
    - depart_airport_code: departure airport code (BGR)
    - arrival_airport_code: arrival airport code (MCO)
    - travel_timeline_summary: text showing dates coordination for both festival attendance and Maine mid-October activities

    EQUIPMENT & BAGGAGE:
    - equipment_list: list of camping/outdoor equipment items to bring (e.g., tent, sleeping bag, hiking gear, backpack)
    - total_checked_bags_count: integer number of checked bags if provided; otherwise null
    - total_baggage_fee_amount_usd: numeric total baggage fee USD if provided; otherwise null

    ORLANDO HOTEL:
    - hotel_name: the selected hotel name (within ~2–3 miles of Camping World Stadium)
    - distance_to_stadium: stated approximate distance to the stadium (text, e.g., "2.1 miles")
    - distance_to_stadium_miles_numeric: numeric miles if provided; otherwise null
    - hotel_url: URL for the hotel/accommodation information

    Rules:
    - Extract exactly what the answer states. Do not invent or infer missing data.
    - For any missing field, return null.
    - For URLs, return actual URLs that appear in the answer. If none, return null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _filter_valid_urls(urls: List[Optional[str]]) -> List[str]:
    out = []
    for u in urls:
        if isinstance(u, str) and ("http://" in u or "https://" in u):
            out.append(u.strip())
    return out


def _parse_amount_from_text(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Extract the first number like 85, 85.00, $85, USD 85, etc.
    m = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _compute_expected_baggage_fee_usd(num_bags: int) -> float:
    """
    Given the number of checked bags, compute expected fee using:
    - $40 for the first checked bag
    - $45 for the second checked bag
    For bags beyond 2, this rubric provides no explicit fee; we compute up to the second bag only.
    """
    fee = 0.0
    if num_bags >= 1:
        fee += BAGGAGE_FEES["first_bag_usd"]
    if num_bags >= 2:
        fee += BAGGAGE_FEES["second_bag_usd"]
    return fee


def _contains_key_equipment(items: List[str]) -> bool:
    """
    Check the list includes at least two typical camping/outdoor items among:
    tent, sleeping bag, backpack, hiking gear/boots, pad, stove.
    """
    if not items:
        return False
    normalized = [i.lower() for i in items]
    keys = ["tent", "sleeping bag", "backpack", "hiking gear", "hiking boots", "sleeping pad", "camp stove", "stove"]
    found = sum(any(k in x for k in keys) for x in normalized)
    return found >= 2


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_festival_details(
    evaluator: Evaluator,
    parent_node,
    fest: FestivalInfo,
) -> Dict[str, Any]:
    """
    Build and verify the 'Festival_Selection_and_Details' subtree.
    Returns dict of useful nodes for downstream prerequisites.
    """
    node = evaluator.add_parallel(
        id="Festival_Selection_and_Details",
        desc="Selection of appropriate outdoor music festival at Camping World Stadium and associated details",
        parent=parent_node,
        critical=False
    )

    # Festival reference URL existence (critical)
    fest_url_present = fest is not None and fest.official_url is not None and fest.official_url.strip() != ""
    url_node = evaluator.add_custom_node(
        result=fest_url_present,
        id="Festival_Reference_URL",
        desc="Provides official URL reference for the selected festival",
        parent=node,
        critical=True
    )

    # Identification (critical)
    ident_leaf = evaluator.add_leaf(
        id="Festival_Identification",
        desc="Identifies one of the three major outdoor music festivals at Camping World Stadium in Orlando in 2026",
        parent=node,
        critical=True
    )
    allowed_names = ", ".join([f["name"] for f in ALLOWED_FESTIVALS])
    allowed_dates = "; ".join([f'{f["name"]}: {f["dates"]}' for f in ALLOWED_FESTIVALS])
    claim_ident = (
        f"The selected festival '{fest.festival_name}' is one of the allowed options ({allowed_names}) "
        f"held in 2026 at Camping World Stadium in Orlando, with dates '{fest.festival_dates}'."
    )
    await evaluator.verify(
        claim=claim_ident,
        node=ident_leaf,
        additional_instruction=f"Allowed festivals and dates: {allowed_dates}. "
                               f"Verify the answer clearly selects one of these and associates it with Camping World Stadium."
    )

    # Timing justification (critical)
    timing_leaf = evaluator.add_leaf(
        id="Festival_Timing_Justification",
        desc="Explains why the selected festival allows for Maine fall foliage viewing during peak season (mid-October)",
        parent=node,
        critical=True
    )
    claim_timing = (
        "The itinerary explains that Maine activities are scheduled in mid-October for peak foliage, and the selected "
        f"festival dates ('{fest.festival_dates}') allow attending the festival and visiting Maine in mid-October."
    )
    await evaluator.verify(
        claim=claim_timing,
        node=timing_leaf,
        additional_instruction="Verify the answer provides a rationale connecting the festival date (May or November) with availability for a mid-October Maine trip."
    )

    # Venue capacity information (critical) – verify via provided festival/stadium URLs
    capacity_leaf = evaluator.add_leaf(
        id="Venue_Capacity_Information",
        desc="Provides the seating capacity of Camping World Stadium (approximately 65,000)",
        parent=node,
        critical=True
    )
    sources_capacity = _filter_valid_urls([fest.official_url] + (fest.additional_urls or []))
    claim_capacity = "Camping World Stadium seating capacity is approximately 65,000."
    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_leaf,
        sources=sources_capacity if sources_capacity else None,
        additional_instruction="Use the provided official festival/stadium URL(s). Allow approximate phrasing such as 'about 65,000' or 'approximately 65,000'."
    )

    # Ticket category and pricing (critical) – verify by official festival URL(s)
    ticket_leaf = evaluator.add_leaf(
        id="Ticket_Category_and_Pricing",
        desc="Specifies ticket category (GA, GA+, or VIP) with accurate pricing information for the selected festival",
        parent=node,
        critical=True
    )
    claim_ticket = (
        f"The ticket category '{fest.ticket_category}' has the stated current pricing '{fest.ticket_price}' for {fest.festival_name}."
    )
    sources_ticket = _filter_valid_urls([fest.official_url] + (fest.additional_urls or []))
    await evaluator.verify(
        claim=claim_ticket,
        node=ticket_leaf,
        sources=sources_ticket if sources_ticket else None,
        additional_instruction="Verify that the official festival page confirms the selected ticket tier and the given price or price range."
    )

    # Return nodes useful for prerequisites in other subtrees
    return {
        "festival_timing_leaf": timing_leaf,
        "festival_url_node": url_node
    }


async def verify_maine_plan(
    evaluator: Evaluator,
    parent_node,
    maine: MainePlan,
) -> Dict[str, Any]:
    """
    Build and verify the 'Maine_Outdoor_Activities_Plan' subtree.
    Returns dict of useful nodes for downstream prerequisites.
    """
    node = evaluator.add_parallel(
        id="Maine_Outdoor_Activities_Plan",
        desc="Detailed plan for outdoor recreation activities in Maine during peak fall foliage season",
        parent=parent_node,
        critical=False
    )

    # Maine activities reference URL existence (critical)
    maine_url_present = maine is not None and maine.activities_url is not None and maine.activities_url.strip() != ""
    maine_url_node = evaluator.add_custom_node(
        result=maine_url_present,
        id="Maine_Activities_Reference_URL",
        desc="Provides URL reference supporting Maine outdoor activities or Acadia information",
        parent=node,
        critical=True
    )

    # Acadia visit in mid-October (critical)
    acadia_leaf = evaluator.add_leaf(
        id="Acadia_National_Park_Visit",
        desc="Includes visit to Acadia National Park during mid-October for peak fall foliage",
        parent=node,
        critical=True
    )
    claim_acadia = (
        "The itinerary explicitly includes a visit to Acadia National Park in mid-October to experience peak fall foliage."
    )
    await evaluator.verify(
        claim=claim_acadia,
        node=acadia_leaf,
        additional_instruction="Verify the answer mentions visiting Acadia National Park and that the timing is mid-October."
    )

    # Wildlife viewing timing in October (critical)
    wildlife_leaf = evaluator.add_leaf(
        id="Wildlife_Viewing_Timing",
        desc="Schedules activities in October for optimal wildlife viewing opportunities",
        parent=node,
        critical=True
    )
    claim_wildlife = (
        "The plan schedules outdoor activities in October to optimize wildlife viewing opportunities, "
        "including migratory birds and animals preparing for winter."
    )
    await evaluator.verify(
        claim=claim_wildlife,
        node=wildlife_leaf,
        additional_instruction="Verify the answer explicitly notes October as prime time for wildlife viewing."
    )

    # Bangor -> Acadia distance/time (critical) – verify via Maine activities URL
    distance_leaf = evaluator.add_leaf(
        id="Bangor_to_Acadia_Distance",
        desc="Accurately states the distance from Bangor to Acadia National Park (approximately 49 miles, 1 hour drive)",
        parent=node,
        critical=True
    )
    claim_distance = (
        f"The driving distance from Bangor to Acadia National Park is stated as '{maine.bangor_to_acadia_distance}' "
        f"and the drive time as '{maine.bangor_to_acadia_drive_time}', which should be around 49 miles (~1 hour)."
    )
    await evaluator.verify(
        claim=claim_distance,
        node=distance_leaf,
        sources=maine.activities_url if (maine and maine.activities_url) else None,
        additional_instruction="Use the provided Maine activities or Acadia URL to confirm the approximate distance/time. Allow minor variations (e.g., 47–52 miles, ~1 hour)."
    )

    # Camping accommodation and Oct 15 closure awareness (critical)
    camping_leaf = evaluator.add_leaf(
        id="Camping_Accommodation_Details",
        desc="Specifies camping accommodation in Maine with awareness of October 15 closure date",
        parent=node,
        critical=True
    )
    claim_camping = (
        "The plan specifies camping accommodations in Maine (state park or Acadia campground) and notes that most Maine campgrounds close by October 15."
    )
    await evaluator.verify(
        claim=claim_camping,
        node=camping_leaf,
        additional_instruction="Verify the answer mentions a specific camping location and references the October 15 closure timing."
    )

    return {
        "acadia_leaf": acadia_leaf,
        "maine_url_node": maine_url_node
    }


async def verify_travel_logistics(
    evaluator: Evaluator,
    parent_node,
    travel: TravelInfo,
    prerequisites: Dict[str, Any]
) -> None:
    """
    Build and verify the 'Travel_Logistics' subtree.
    """
    node = evaluator.add_parallel(
        id="Travel_Logistics",
        desc="Complete travel arrangements between Bangor, Maine and Orlando, Florida",
        parent=parent_node,
        critical=False
    )

    # Flight route specification (critical) – simple verify against the answer
    flight_leaf = evaluator.add_leaf(
        id="Flight_Route_Specification",
        desc="Specifies American Airlines flight route from Bangor International Airport (BGR) to Orlando International Airport (MCO)",
        parent=node,
        critical=True
    )
    claim_route = (
        f"The plan books flights on American Airlines from Bangor International Airport (BGR) to Orlando International Airport (MCO), "
        f"clearly specifying departure and arrival airports."
    )
    await evaluator.verify(
        claim=claim_route,
        node=flight_leaf,
        additional_instruction="Verify the answer explicitly mentions American Airlines and the BGR → MCO routing with both airport codes."
    )

    # Travel timeline coordination (critical) – simple verify; depend on festival timing & Acadia visit
    timeline_leaf = evaluator.add_leaf(
        id="Travel_Timeline_Coordination",
        desc="Coordinates travel dates to accommodate both festival attendance and Maine activities during peak foliage season",
        parent=node,
        critical=True
    )
    claim_timeline = (
        "The travel dates are coordinated to allow attending the selected Orlando festival and conducting Maine activities in mid-October during peak foliage."
    )
    await evaluator.verify(
        claim=claim_timeline,
        node=timeline_leaf,
        additional_instruction="Verify the answer shows coherent timing for both trip components (festival and mid-October Maine).",
        extra_prerequisites=[
            prerequisites.get("festival_timing_leaf"),
            prerequisites.get("acadia_leaf")
        ]
    )


async def verify_equipment_and_baggage(
    evaluator: Evaluator,
    parent_node,
    equipment: EquipmentBaggage
) -> None:
    """
    Build and verify the 'Equipment_and_Baggage_Planning' subtree.
    """
    node = evaluator.add_parallel(
        id="Equipment_and_Baggage_Planning",
        desc="Planning for outdoor recreation equipment transport and associated costs",
        parent=parent_node,
        critical=False
    )

    # Camping equipment list provided (critical)
    equipment_present = equipment is not None and _contains_key_equipment(equipment.equipment_list)
    equip_leaf = evaluator.add_custom_node(
        result=equipment_present,
        id="Camping_Equipment_List",
        desc="Lists specific camping/outdoor equipment needed (e.g., tent, sleeping bag, hiking gear)",
        parent=node,
        critical=True
    )

    # We split 'Baggage_Fee_Calculation' into two leaves to follow one-check-per-leaf best practice:
    # 1) Provided
    # 2) Correctness

    # 1) Baggage fee provided (critical)
    fee_provided = equipment is not None and (equipment.total_baggage_fee_amount_usd is not None)
    fee_provided_leaf = evaluator.add_custom_node(
        result=fee_provided,
        id="Baggage_Fee_Provided",
        desc="Total American Airlines baggage fee is provided",
        parent=node,
        critical=True
    )

    # 2) Baggage fee correctness (critical) – compute expected fee using count of bags
    num_bags = (
        equipment.total_checked_bags_count if (equipment and equipment.total_checked_bags_count is not None)
        else (len(equipment.equipment_list) if equipment else 0)
    )
    expected_fee = _compute_expected_baggage_fee_usd(num_bags)
    provided_fee = equipment.total_baggage_fee_amount_usd
    fee_correct = (provided_fee is not None) and (abs(provided_fee - expected_fee) <= 0.5)  # allow small rounding tolerance

    # Record custom info for transparency
    evaluator.add_custom_info(
        info={
            "bags_count_used": num_bags,
            "expected_fee_usd": expected_fee,
            "provided_fee_usd": provided_fee,
            "first_bag_usd": BAGGAGE_FEES["first_bag_usd"],
            "second_bag_usd": BAGGAGE_FEES["second_bag_usd"]
        },
        info_type="baggage_fee_check",
        info_name="baggage_fee_calculation"
    )

    fee_correct_leaf = evaluator.add_custom_node(
        result=fee_correct,
        id="Baggage_Fee_Calculation",
        desc="Calculates American Airlines baggage fees correctly (first bag $40, second bag $45; each piece as a separate checked bag)",
        parent=node,
        critical=True
    )


async def verify_orlando_accommodation(
    evaluator: Evaluator,
    parent_node,
    hotel: OrlandoHotel
) -> None:
    """
    Build and verify the 'Orlando_Accommodation' subtree.
    """
    node = evaluator.add_parallel(
        id="Orlando_Accommodation",
        desc="Hotel accommodation near Camping World Stadium in Orlando",
        parent=parent_node,
        critical=False
    )

    # Hotel URL provided (critical)
    hotel_url_present = hotel is not None and hotel.hotel_url is not None and hotel.hotel_url.strip() != ""
    hotel_url_leaf = evaluator.add_custom_node(
        result=hotel_url_present,
        id="Orlando_Hotel_Reference_URL",
        desc="Provides URL reference for hotel information or Orlando accommodation options",
        parent=node,
        critical=True
    )

    # Hotel name and within ~2–3 miles (critical) – simple verify; optionally pass hotel URL
    name_loc_leaf = evaluator.add_leaf(
        id="Hotel_Name_and_Location",
        desc="Specifies a hotel within approximately 2–3 miles of Camping World Stadium",
        parent=node,
        critical=True
    )
    claim_hotel = (
        f"The plan specifies hotel '{hotel.hotel_name}' located within approximately 2–3 miles of Camping World Stadium."
    )
    await evaluator.verify(
        claim=claim_hotel,
        node=name_loc_leaf,
        sources=hotel.hotel_url if (hotel and hotel.hotel_url) else None,
        additional_instruction="Verify the answer states the hotel name and that it is within ~2–3 miles of Camping World Stadium."
    )

    # Distance to venue (critical) – simple verify of the stated distance; optionally pass hotel URL
    distance_leaf = evaluator.add_leaf(
        id="Distance_to_Venue",
        desc="Provides approximate distance from hotel to Camping World Stadium",
        parent=node,
        critical=True
    )
    claim_distance = (
        f"The answer provides the approximate distance from the selected hotel to Camping World Stadium as '{hotel.distance_to_stadium}'."
    )
    await evaluator.verify(
        claim=claim_distance,
        node=distance_leaf,
        sources=hotel.hotel_url if (hotel and hotel.hotel_url) else None,
        additional_instruction="Verify the stated approximate distance from the hotel to the stadium as provided in the answer; allow reasonable rounding."
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
    Evaluate a single answer for the 2026 Orlando festival + Maine fall trip plan.
    Returns a structured summary dict including the verification tree and final score.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root as parallel to allow partial credit across main sections
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

    # Add helpful ground truth/context info to summary
    evaluator.add_ground_truth({
        "allowed_festivals": ALLOWED_FESTIVALS,
        "stadium_capacity_expectation": STADIUM_CAPACITY_APPROX,
        "aa_baggage_fees_usd": BAGGAGE_FEES
    }, gt_type="trip_requirements_reference")

    # Extract structured info from the answer
    trip = await evaluator.extract(
        prompt=prompt_extract_trip(),
        template_class=TripExtraction,
        extraction_name="trip_extraction"
    )

    # Build main root node (marked critical in rubric, but root must be non-critical to avoid child-critical constraint)
    plan_root = evaluator.add_parallel(
        id="Complete_Outdoor_Recreation_Trip_Plan",
        desc="A comprehensive fall outdoor recreation trip plan combining an Orlando music festival with Maine outdoor activities",
        parent=root,
        critical=False
    )

    # Subtrees
    fest_nodes = await verify_festival_details(
        evaluator=evaluator,
        parent_node=plan_root,
        fest=trip.festival or FestivalInfo()
    )

    maine_nodes = await verify_maine_plan(
        evaluator=evaluator,
        parent_node=plan_root,
        maine=trip.maine or MainePlan()
    )

    await verify_travel_logistics(
        evaluator=evaluator,
        parent_node=plan_root,
        travel=trip.travel or TravelInfo(),
        prerequisites={
            "festival_timing_leaf": fest_nodes.get("festival_timing_leaf"),
            "acadia_leaf": maine_nodes.get("acadia_leaf")
        }
    )

    await verify_equipment_and_baggage(
        evaluator=evaluator,
        parent_node=plan_root,
        equipment=trip.equipment or EquipmentBaggage()
    )

    await verify_orlando_accommodation(
        evaluator=evaluator,
        parent_node=plan_root,
        hotel=trip.hotel or OrlandoHotel()
    )

    # Return summary
    return evaluator.get_summary()