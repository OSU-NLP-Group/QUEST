import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "mauritius_travel_planning"
TASK_DESCRIPTION = """You are planning a vacation to Mauritius departing from Nashville International Airport (BNA). Research and provide the following information:

1. Travel Requirements: Confirm whether US citizens need a visa for tourist travel to Mauritius and document the standard duration allowed.

2. Flight Constraints: Verify whether direct flights exist from the United States to Mauritius. If connections are required, document the minimum recommended layover time for international connections.

3. Airport Parking: Identify at least one economy or off-site parking option at Nashville BNA Airport with a daily rate under $20.

4. Hotel Selection: Find FOUR different beachfront hotels in Mauritius that meet ALL of the following criteria:
   - Located directly on the beach (beachfront, within 100 meters of the beach with direct access)
   - Minimum 4-star rating from a recognized source (hotel website, major booking platform, or tourism authority)
   - Offers "Deluxe" room category (or equivalent upgraded room type) that is at least 500 square feet or described as notably larger than standard rooms
   - For each hotel, provide: hotel name, specific location in Mauritius, booking/official website URL, confirmation of beachfront status, star rating with source, and deluxe room availability with size confirmation
   - Additionally, note whether each hotel offers: swimming pool, on-site restaurant, and spa services

For each piece of information, provide supporting reference URLs from your research.
"""


# -----------------------------------------------------------------------------
# Pydantic models for extraction
# -----------------------------------------------------------------------------
class VisaInfo(BaseModel):
    visa_reference_urls: List[str] = Field(default_factory=list)
    visa_policy_summary: Optional[str] = None
    visa_duration_text: Optional[str] = None  # e.g., "90 days"
    no_visa_required: Optional[bool] = None


class FlightConstraints(BaseModel):
    flight_reference_urls: List[str] = Field(default_factory=list)
    no_direct_flights_statement: Optional[str] = None
    no_direct_flights: Optional[bool] = None


class LayoverInfo(BaseModel):
    layover_reference_urls: List[str] = Field(default_factory=list)
    recommended_min_layover_text: Optional[str] = None  # e.g., "2-3 hours"


class ParkingOption(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    daily_rate_text: Optional[str] = None     # e.g., "$15/day"
    daily_rate_usd: Optional[str] = None      # e.g., "15"
    is_on_site: Optional[bool] = None
    is_off_site: Optional[bool] = None
    qualifies_under_20: Optional[bool] = None


class HotelInfo(BaseModel):
    # Basic
    name: Optional[str] = None
    location: Optional[str] = None
    booking_url: Optional[str] = None

    # Beachfront
    beachfront_sources: List[str] = Field(default_factory=list)
    direct_beach_access: Optional[bool] = None
    distance_within_100m: Optional[bool] = None

    # Rating
    rating_value: Optional[str] = None
    rating_sources: List[str] = Field(default_factory=list)

    # Room type (Deluxe or equivalent)
    room_sources: List[str] = Field(default_factory=list)
    deluxe_available: Optional[bool] = None
    room_size_sqft: Optional[str] = None
    room_size_text: Optional[str] = None
    larger_than_standard: Optional[bool] = None

    # Amenities
    amenities_sources: List[str] = Field(default_factory=list)
    pool: Optional[bool] = None
    restaurant: Optional[bool] = None
    spa: Optional[bool] = None


class TravelPlanExtraction(BaseModel):
    visa_info: Optional[VisaInfo] = None
    flight_info: Optional[FlightConstraints] = None
    layover_info: Optional[LayoverInfo] = None
    parking_options: List[ParkingOption] = Field(default_factory=list)
    hotels: List[HotelInfo] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_travel_plan() -> str:
    return """
Extract structured information from the answer for a Mauritius vacation plan from Nashville (BNA). Follow these rules:
- Only extract what is explicitly present in the answer.
- For URLs, extract full valid URLs (include protocol). If none are provided, use an empty list.
- If a field is not present, set it to null (for strings/booleans) or [] (for lists).

Return a JSON object with this schema:

{
  "visa_info": {
    "visa_reference_urls": [string, ...],
    "visa_policy_summary": string|null,
    "visa_duration_text": string|null,
    "no_visa_required": boolean|null
  },
  "flight_info": {
    "flight_reference_urls": [string, ...],
    "no_direct_flights_statement": string|null,
    "no_direct_flights": boolean|null
  },
  "layover_info": {
    "layover_reference_urls": [string, ...],
    "recommended_min_layover_text": string|null
  },
  "parking_options": [
    {
      "name": string|null,
      "url": string|null,
      "daily_rate_text": string|null,
      "daily_rate_usd": string|null,
      "is_on_site": boolean|null,
      "is_off_site": boolean|null,
      "qualifies_under_20": boolean|null
    },
    ...
  ],
  "hotels": [
    {
      "name": string|null,
      "location": string|null,
      "booking_url": string|null,

      "beachfront_sources": [string, ...],
      "direct_beach_access": boolean|null,
      "distance_within_100m": boolean|null,

      "rating_value": string|null,
      "rating_sources": [string, ...],

      "room_sources": [string, ...],
      "deluxe_available": boolean|null,
      "room_size_sqft": string|null,
      "room_size_text": string|null,
      "larger_than_standard": boolean|null,

      "amenities_sources": [string, ...],
      "pool": boolean|null,
      "restaurant": boolean|null,
      "spa": boolean|null
    },
    ...
  ]
}

Notes:
- visa_reference_urls: pages like official government sites, embassy, tourism board, IATA, or recognized visa resources cited in the answer.
- flight_reference_urls: pages cited in the answer supporting whether there are (or are not) direct flights US→Mauritius (e.g., airline route maps, MRU airport, flight search results referenced).
- layover_reference_urls: cited sources for minimum recommended layover times for international connections.
- parking_options: include any economy or off-site options near BNA with the cited daily rate and URL. If the answer explicitly says a daily rate under $20, set qualifies_under_20 true.
- hotels: extract up to 6 hotels if present; include all the specified fields and URL lists from the answer.
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    return url.startswith("http://") or url.startswith("https://")


def parse_usd_amount_to_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Find a number like 15, 15.00, $15, $15.50
    m = re.search(r"\$?\s*([0-9]+(?:\.[0-9]+)?)", text.replace(",", ""))
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def first_n(lst: List[Any], n: int) -> List[Any]:
    return lst[:n] if lst else []


def non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if is_valid_url(u)]


# -----------------------------------------------------------------------------
# Verification subroutines
# -----------------------------------------------------------------------------
async def verify_visa_requirements(evaluator: Evaluator, parent_node, data: TravelPlanExtraction):
    node = evaluator.add_sequential(
        id="Visa_Requirements",
        desc="Verification of visa requirements for US citizens traveling to Mauritius",
        parent=parent_node,
        critical=True  # Critical section
    )

    visa = data.visa_info or VisaInfo()

    # Existence of visa references
    has_refs = len(non_empty_urls(visa.visa_reference_urls)) > 0
    evaluator.add_custom_node(
        result=has_refs,
        id="Visa_Requirement_Reference_URLs",
        desc="URLs supporting visa requirement information",
        parent=node,
        critical=True
    )

    # No visa required confirmation
    leaf_no_visa = evaluator.add_leaf(
        id="No_Visa_Required_Confirmation",
        desc="Confirmation that US citizens do not need a visa for Mauritius for tourist stays",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="US citizens traveling to Mauritius for tourism do not need to obtain a visa in advance (visa‑exempt/visa‑free entry).",
        node=leaf_no_visa,
        sources=visa.visa_reference_urls,
        additional_instruction="Focus specifically on tourist travel. Phrases like 'visa-free', 'no visa required', or 'visa not required for 90 days' should be treated as supporting evidence."
    )

    # Standard duration documented (set as critical to satisfy framework constraints)
    leaf_duration = evaluator.add_leaf(
        id="Visa_Duration_Documented",
        desc="Documentation of the standard duration allowed (90 days) for US citizens in Mauritius without a visa",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The standard visa‑free stay duration for US citizens in Mauritius is 90 days (about 3 months).",
        node=leaf_duration,
        sources=visa.visa_reference_urls,
        additional_instruction="Accept reasonable phrasing like 'up to 90 days', 'max 90 days', or '3 months'."
    )


async def verify_flight_constraints(evaluator: Evaluator, parent_node, data: TravelPlanExtraction):
    node = evaluator.add_sequential(
        id="Flight_Constraints_Verification",
        desc="Verification that no direct flights exist from the US to Mauritius",
        parent=parent_node,
        critical=True
    )

    flight = data.flight_info or FlightConstraints()

    # Existence of flight references
    has_refs = len(non_empty_urls(flight.flight_reference_urls)) > 0
    evaluator.add_custom_node(
        result=has_refs,
        id="Flight_Constraints_Reference_URLs",
        desc="URLs supporting flight availability information",
        parent=node,
        critical=True
    )

    # No direct flights confirmation
    leaf_no_direct = evaluator.add_leaf(
        id="No_Direct_Flights_Confirmed",
        desc="Confirmation that no direct flights exist between the United States and Mauritius, requiring at least one connection",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="There are no nonstop/direct flights from any US airport to Mauritius (MRU); at least one connection is required.",
        node=leaf_no_direct,
        sources=flight.flight_reference_urls,
        additional_instruction="Pages like airline route maps, MRU airport flight information, or credible flight search references cited in the answer can support this."
    )


async def verify_minimum_layover(evaluator: Evaluator, parent_node, data: TravelPlanExtraction):
    node = evaluator.add_sequential(
        id="Minimum_Layover_Time",
        desc="Documentation of minimum recommended layover time for international connections",
        parent=parent_node,
        critical=True
    )

    layover = data.layover_info or LayoverInfo()

    # Existence of layover references
    has_refs = len(non_empty_urls(layover.layover_reference_urls)) > 0
    evaluator.add_custom_node(
        result=has_refs,
        id="Layover_Reference_URLs",
        desc="URLs supporting layover time recommendations",
        parent=node,
        critical=True
    )

    # 2–3 hour minimum documented
    leaf_min = evaluator.add_leaf(
        id="Two_Hour_Minimum_Documented",
        desc="Documentation that minimum recommended layover time for international connections is 2-3 hours",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="A minimum recommended layover time for international flight connections is about 2–3 hours.",
        node=leaf_min,
        sources=layover.layover_reference_urls,
        additional_instruction="Look for guidance from airlines, airports, or credible travel resources cited in the answer. Accept ranges like '2 hours minimum' or '2–3 hours recommended'."
    )


async def verify_bna_parking(evaluator: Evaluator, parent_node, data: TravelPlanExtraction):
    node = evaluator.add_sequential(
        id="BNA_Parking_Information",
        desc="Identification of affordable parking options at Nashville BNA Airport",
        parent=parent_node,
        critical=True
    )

    # Gather all parking URLs from options
    all_parking_urls = [p.url for p in (data.parking_options or []) if is_valid_url(p.url)]
    has_parking_refs = len(all_parking_urls) > 0

    evaluator.add_custom_node(
        result=has_parking_refs,
        id="Parking_Reference_URLs",
        desc="URLs supporting BNA parking rate information",
        parent=node,
        critical=True
    )

    # Under $20/day confirmation: verify using any provided parking URLs.
    # The verifier will pass if any one of the URLs supports the claim.
    leaf_under_20 = evaluator.add_leaf(
        id="Economy_Parking_Rate_Under_Twenty_Dollars",
        desc="Identification of at least one economy or off-site parking option at BNA with daily rate under $20",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This page shows a Nashville International Airport (BNA) economy or off-site parking option with a daily rate under $20.",
        node=leaf_under_20,
        sources=all_parking_urls,
        additional_instruction="Accept either an official BNA economy lot or an off-site provider near BNA if the page clearly shows a daily price under $20."
    )


async def verify_single_hotel(evaluator: Evaluator, parent_node, hotel: HotelInfo, idx: int):
    """
    Verify a single hotel with all required criteria.
    """
    hotel_id = f"Hotel_{idx}"
    hotel_node = evaluator.add_parallel(
        id=hotel_id,
        desc=f"{['First','Second','Third','Fourth','Fifth','Sixth'][idx-1]} qualifying beachfront hotel in Mauritius meeting all specified criteria" if 1 <= idx <= 6 else f"Hotel {idx} verification",
        parent=parent_node,
        critical=False  # Overall hotel node allows partial credit within
    )

    # 1) Basic Information (critical)
    basic_node = evaluator.add_parallel(
        id=f"{hotel_id}_Basic_Information",
        desc=f"Basic identifying information for {hotel_id}",
        parent=hotel_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(hotel.name and hotel.name.strip()),
        id=f"{hotel_id}_Name",
        desc="The name of the hotel",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(hotel.location and hotel.location.strip()),
        id=f"{hotel_id}_Location",
        desc="The city or specific area in Mauritius where the hotel is located",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=is_valid_url(hotel.booking_url),
        id=f"{hotel_id}_Booking_URL",
        desc="A valid URL where the hotel can be viewed or booked",
        parent=basic_node,
        critical=True
    )

    # 2) Beachfront Status (critical)
    beach_node = evaluator.add_parallel(
        id=f"{hotel_id}_Beachfront_Status",
        desc=f"Verification that {hotel_id} is a true beachfront property",
        parent=hotel_node,
        critical=True
    )

    beachfront_urls = non_empty_urls(hotel.beachfront_sources)
    evaluator.add_custom_node(
        result=len(beachfront_urls) > 0,
        id=f"{hotel_id}_Beachfront_Reference_URLs",
        desc=f"URLs supporting beachfront status of {hotel_id}",
        parent=beach_node,
        critical=True
    )

    leaf_direct_access = evaluator.add_leaf(
        id=f"{hotel_id}_Direct_Beach_Access",
        desc=f"Confirmation that {hotel_id} provides direct access to the beach with no buildings between hotel and beach",
        parent=beach_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel '{hotel.name or 'the hotel'}' is directly on the beach with direct beach access (no road/buildings in between).",
        node=leaf_direct_access,
        sources=beachfront_urls,
        additional_instruction="Accept phrases like 'beachfront', 'direct beach access', 'on the beach', or map/location evidence showing the property touches the beach."
    )

    leaf_within_100m = evaluator.add_leaf(
        id=f"{hotel_id}_Beach_Distance_Within_100m",
        desc=f"Confirmation that {hotel_id} is within 100 meters (1-minute walk) of the beach",
        parent=beach_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel '{hotel.name or 'the hotel'}' is located on the beach or within about 100 meters (~1-minute walk) of the beach.",
        node=leaf_within_100m,
        sources=beachfront_urls,
        additional_instruction="Allow reasonable language indicating immediate beachfront location or a very short walk (≈100 m)."
    )

    # 3) Star Rating (critical, sequential)
    rating_node = evaluator.add_sequential(
        id=f"{hotel_id}_Star_Rating",
        desc=f"Verification that {hotel_id} meets minimum star rating requirement",
        parent=hotel_node,
        critical=True
    )

    rating_urls = non_empty_urls(hotel.rating_sources)
    evaluator.add_custom_node(
        result=len(rating_urls) > 0,
        id=f"{hotel_id}_Star_Rating_Reference_URLs",
        desc=f"URLs supporting star rating of {hotel_id}",
        parent=rating_node,
        critical=True
    )

    leaf_min_four = evaluator.add_leaf(
        id=f"{hotel_id}_Minimum_Four_Star",
        desc=f"Confirmation that {hotel_id} has a minimum 4-star rating",
        parent=rating_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel '{hotel.name or 'the hotel'}' is rated at least 4 stars.",
        node=leaf_min_four,
        sources=rating_urls,
        additional_instruction="Accept evidence from recognized sources (official hotel site, Booking.com, Expedia, Hotels.com, Agoda, Google Hotels, or an official tourism board). The rating may be shown as text or icons."
    )

    leaf_rating_src = evaluator.add_leaf(
        id=f"{hotel_id}_Rating_Source_Verified",
        desc=f"Verification that the star rating comes from a recognized source (hotel website, major booking platform, or tourism board)",
        parent=rating_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page is a recognized source for hotel ratings (official hotel website, major booking platform, or an official tourism/tourism board site).",
        node=leaf_rating_src,
        sources=rating_urls,
        additional_instruction="Check the nature of the site: official domain for the hotel, major OTAs (Booking, Expedia, Hotels.com, Agoda), Google Hotels, or official tourism authority."
    )

    # 4) Room Type (critical, parallel)
    room_node = evaluator.add_parallel(
        id=f"{hotel_id}_Room_Type",
        desc=f"Verification that {hotel_id} offers deluxe room category meeting size requirements - at least one of the size criteria must be satisfied",
        parent=hotel_node,
        critical=True
    )

    room_urls = non_empty_urls(hotel.room_sources)
    evaluator.add_custom_node(
        result=len(room_urls) > 0,
        id=f"{hotel_id}_Room_Type_Reference_URLs",
        desc=f"URLs supporting room type information for {hotel_id}",
        parent=room_node,
        critical=True
    )

    leaf_deluxe = evaluator.add_leaf(
        id=f"{hotel_id}_Deluxe_Room_Available",
        desc=f"Confirmation that {hotel_id} offers rooms explicitly designated as 'Deluxe' or equivalent upgrade category",
        parent=room_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page shows that the hotel offers a 'Deluxe' room or an equivalent upgraded room category (e.g., Premium/Club/Junior Suite).",
        node=leaf_deluxe,
        sources=room_urls,
        additional_instruction="Synonyms like 'Deluxe Ocean View', 'Premium', 'Club', 'Executive', or 'Junior Suite' count as upgraded categories."
    )

    leaf_room_size = evaluator.add_leaf(
        id=f"{hotel_id}_Room_Size_Specification",
        desc="Documentation of room size - either specific measurement of at least 500 square feet OR description as notably larger than standard rooms",
        parent=room_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page shows that the Deluxe/upgrade room is at least 500 square feet (≈46–50 sqm) OR explicitly described as notably larger than standard rooms.",
        node=leaf_room_size,
        sources=room_urls,
        additional_instruction="Accept sizes in square meters if they convert to roughly ≥46.5 sqm (≈500 sq ft). Alternatively, accept clear textual statements that the room is significantly larger than the hotel's standard rooms."
    )

    # 5) Amenities (non-critical, parallel)
    amen_node = evaluator.add_parallel(
        id=f"{hotel_id}_Amenities",
        desc=f"Documentation of available amenities at {hotel_id}",
        parent=hotel_node,
        critical=False
    )

    amen_urls = non_empty_urls(hotel.amenities_sources)
    evaluator.add_custom_node(
        result=len(amen_urls) > 0,
        id=f"{hotel_id}_Amenities_Reference_URLs",
        desc=f"URLs supporting amenity information for {hotel_id}",
        parent=amen_node,
        critical=True  # reference existence should be true to proceed
    )

    leaf_pool = evaluator.add_leaf(
        id=f"{hotel_id}_Pool_Available",
        desc=f"Indication of whether {hotel_id} has a swimming pool",
        parent=amen_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The hotel '{hotel.name or 'the hotel'}' has a swimming pool.",
        node=leaf_pool,
        sources=amen_urls,
        additional_instruction="Look for features/amenities sections or descriptions confirming the presence of a pool."
    )

    leaf_rest = evaluator.add_leaf(
        id=f"{hotel_id}_Restaurant_OnSite",
        desc=f"Indication of whether {hotel_id} has an on-site restaurant",
        parent=amen_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The hotel '{hotel.name or 'the hotel'}' has an on-site restaurant.",
        node=leaf_rest,
        sources=amen_urls,
        additional_instruction="Confirm that dining options or an on-site restaurant are mentioned."
    )

    leaf_spa = evaluator.add_leaf(
        id=f"{hotel_id}_Spa_Services",
        desc=f"Indication of whether {hotel_id} offers spa services",
        parent=amen_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The hotel '{hotel.name or 'the hotel'}' offers a spa or spa/wellness services.",
        node=leaf_spa,
        sources=amen_urls,
        additional_instruction="Accept 'spa', 'wellness center', 'massage treatments', or similar wording indicating spa services."
    )


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the Mauritius travel planning task using the Mind2Web2 framework.
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

    # Extraction
    extracted: TravelPlanExtraction = await evaluator.extract(
        prompt=prompt_extract_travel_plan(),
        template_class=TravelPlanExtraction,
        extraction_name="travel_plan_extraction"
    )

    # Top-level planning node (non-critical to allow partial credit while inner sections enforce critical logic)
    top = evaluator.add_parallel(
        id="Mauritius_Travel_Planning",
        desc="Complete travel planning package for a Mauritius vacation from Nashville, including verification of travel requirements, hotel selection, and logistics",
        parent=root,
        critical=False
    )

    # Core verifications
    await verify_visa_requirements(evaluator, top, extracted)
    await verify_flight_constraints(evaluator, top, extracted)
    await verify_minimum_layover(evaluator, top, extracted)
    await verify_bna_parking(evaluator, top, extracted)

    # Hotels: take the first 4 (pad with empty if fewer)
    hotels = list(extracted.hotels or [])
    while len(hotels) < 4:
        hotels.append(HotelInfo())

    hotel_nodes = []
    for i in range(4):
        # Hotel indexing in rubric uses Hotel_1..Hotel_4
        await verify_single_hotel(evaluator, top, hotels[i], idx=i + 1)
        hotel_nodes.append(i + 1)

    # Return evaluation summary
    return evaluator.get_summary()