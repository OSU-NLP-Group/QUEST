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
TASK_ID = "miami_beach_family_hotels"
TASK_DESCRIPTION = """
A family of five (2 adults and 3 children aged 3, 7, and 10) is planning a beach vacation to Miami Beach, Florida, for July 2026. They need to identify 3 hotels that meet all of the following requirements:

1. Room Configuration: The hotel must offer accommodations that can house all 5 family members, either through a two-bedroom suite (minimum 900 square feet) or two connecting rooms.

2. Location: The hotel must be located in Miami Beach with direct beach access or beachfront location, and be within the Miami Beach area (approximately 10-15 miles from Miami International Airport).

3. Swimming Pool: The hotel must have an on-site swimming pool accessible to all guests.

4. Check-in Policy: The hotel must allow adults (18 years or older) to check in with children.

5. Room Amenities: The hotel must be able to provide cribs or rollaway beds upon request for young children.

6. Transportation Access: The hotel should have access to public transportation options such as the Miami Beach Trolley or be within walkable distance to trolley stops.

For each of the 3 hotels identified, provide:
- Hotel name and location in Miami Beach
- Confirmation that the room configuration meets the family size requirement
- Verification of beachfront or beach access
- Confirmation of on-site swimming pool
- Check-in age policy
- Supporting URL references for each major requirement category (basic info, rooms, location, amenities, and policies)

Each hotel must satisfy all critical requirements to be considered a valid option.
"""


# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class HotelBasic(BaseModel):
    name: Optional[str] = None
    location_text: Optional[str] = None
    basic_urls: List[str] = Field(default_factory=list)


class HotelRooms(BaseModel):
    supports_family_config: Optional[str] = None  # e.g., "two-bedroom suite >=900 sq ft" or "connecting rooms"
    suite_sqft: Optional[str] = None              # e.g., "900 sq ft", "1,000 square feet"
    connecting_rooms_available: Optional[str] = None  # e.g., "Yes", "Available on request"
    occupancy_statement: Optional[str] = None         # e.g., "Sleeps 5", "Max occupancy 5"
    cribs_or_rollaway: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HotelLocation(BaseModel):
    beachfront: Optional[str] = None              # e.g., "Beachfront", "Oceanfront"
    direct_beach_access: Optional[str] = None     # e.g., "Direct beach access"
    in_miami_beach: Optional[str] = None          # e.g., "Miami Beach, FL"
    mia_distance_statement: Optional[str] = None  # optional note, if provided
    urls: List[str] = Field(default_factory=list)


class HotelAmenities(BaseModel):
    pool: Optional[str] = None                    # e.g., "Outdoor pool", "Pool on site"
    urls: List[str] = Field(default_factory=list)


class HotelPolicies(BaseModel):
    checkin_age: Optional[str] = None             # e.g., "18+", "21+"
    checkin_time: Optional[str] = None            # e.g., "3:00 PM", "4:00 PM"
    children_stay_free: Optional[str] = None      # e.g., "Children under 12 stay free"
    urls: List[str] = Field(default_factory=list)


class HotelADA(BaseModel):
    ada_compliance: Optional[str] = None          # e.g., "ADA compliant", "Accessible rooms available"
    urls: List[str] = Field(default_factory=list)


class HotelTransport(BaseModel):
    trolley_access: Optional[str] = None          # e.g., "Near Miami Beach Trolley stop"
    urls: List[str] = Field(default_factory=list)


class HotelItem(BaseModel):
    basic: HotelBasic = Field(default_factory=HotelBasic)
    rooms: HotelRooms = Field(default_factory=HotelRooms)
    location: HotelLocation = Field(default_factory=HotelLocation)
    amenities: HotelAmenities = Field(default_factory=HotelAmenities)
    policies: HotelPolicies = Field(default_factory=HotelPolicies)
    ada: HotelADA = Field(default_factory=HotelADA)
    transport: HotelTransport = Field(default_factory=HotelTransport)


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    Extract up to 3 hotels mentioned in the answer that are proposed for a family of five visiting Miami Beach, Florida.
    For each hotel, extract the following fields exactly as stated in the answer (use null for missing fields; include all URLs explicitly cited in the answer):

    Hotel object fields:
    - basic:
      - name: The hotel name
      - location_text: The location/address text as presented (should indicate Miami Beach if available)
      - basic_urls: URLs that identify the hotel's official page or listing and indicate the location (homepage, contact/location page, etc.)

    - rooms:
      - supports_family_config: A statement indicating either a two-bedroom suite (at least 900 sq ft) OR two connecting rooms are available to accommodate 5
      - suite_sqft: The square footage value for the two-bedroom suite if stated (e.g., "900 sq ft")
      - connecting_rooms_available: A statement indicating connecting/adjoining rooms availability
      - occupancy_statement: Any occupancy/maximum persons statement showing capacity of 5
      - cribs_or_rollaway: A statement confirming cribs or rollaway beds can be provided on request
      - urls: URLs specific to room configuration/occupancy/cribs/rollaway/connecting rooms information

    - location:
      - beachfront: A statement indicating beachfront/oceanfront/on the beach
      - direct_beach_access: A statement indicating direct beach access
      - in_miami_beach: A statement confirming the hotel is in Miami Beach, Florida
      - mia_distance_statement: If the answer mentions proximity/distance to MIA, include it verbatim (optional)
      - urls: URLs specific to beachfront/beach access/location in Miami Beach

    - amenities:
      - pool: A statement confirming an on-site pool accessible to guests
      - urls: URLs specific to pool/amenities

    - policies:
      - checkin_age: A statement of the minimum check-in age (e.g., "18+" or "21+")
      - checkin_time: A statement of check-in time (should be between 3:00 PM and 4:00 PM if stated)
      - children_stay_free: A statement that children under a specified age can stay free when sharing a room with adults
      - urls: URLs specific to hotel policies (check-in, children, etc.)

    - ada:
      - ada_compliance: A statement indicating ADA compliance or accessible features/rooms
      - urls: URLs specific to ADA/accessibility information

    - transport:
      - trolley_access: A statement indicating access to the Miami Beach Trolley or walkable distance to trolley stops
      - urls: URLs specific to transportation/trolley access (if cited)

    Return a JSON object with a 'hotels' array containing up to 3 such hotel objects in the order presented in the answer. If the answer includes more than 3 hotels, include only the first 3. If fewer than 3 hotels are provided, include the available ones and leave missing fields as null or empty lists.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def merge_urls(*lists: Optional[List[str]]) -> List[str]:
    """Merge multiple URL lists, deduplicate while preserving order."""
    result: List[str] = []
    seen: set = set()
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                result.append(url)
    return result


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_hotel(
    evaluator: Evaluator,
    parent_node,
    hotel: HotelItem,
    idx: int,
) -> None:
    """
    Build verification tree and run checks for a single hotel.
    """
    hotel_node = evaluator.add_parallel(
        id=f"hotel_{idx}",
        desc=f"Hotel option #{idx + 1}",
        parent=parent_node,
        critical=False  # allow partial credit across multiple hotels
    )

    # ----------------- Basic Info -----------------
    basic_node = evaluator.add_parallel(
        id=f"hotel_{idx}_basic_info",
        desc="Hotel is clearly identified and located in Miami Beach",
        parent=hotel_node,
        critical=True
    )

    # Existence: Hotel name
    evaluator.add_custom_node(
        result=bool(hotel.basic.name and hotel.basic.name.strip()),
        id=f"hotel_{idx}_name",
        desc="Hotel name is provided",
        parent=basic_node,
        critical=True
    )

    # Existence: Basic info URL(s)
    basic_sources = merge_urls(hotel.basic.basic_urls, hotel.location.urls)
    evaluator.add_custom_node(
        result=len(basic_sources) > 0,
        id=f"hotel_{idx}_basic_info_url",
        desc="Supporting URL provided for basic hotel identification/location",
        parent=basic_node,
        critical=True
    )

    # Verify: Located in Miami Beach
    loc_leaf = evaluator.add_leaf(
        id=f"hotel_{idx}_miami_beach_location",
        desc="Hotel location/address indicates it is in Miami Beach, Florida",
        parent=basic_node,
        critical=True
    )
    loc_claim = "This hotel is located in Miami Beach, Florida."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=basic_sources,
        additional_instruction="Confirm the hotel address or location text shows 'Miami Beach, FL' or equivalent."
    )

    # ----------------- Room Configuration -----------------
    rooms_node = evaluator.add_parallel(
        id=f"hotel_{idx}_room_configuration",
        desc="Room configuration supports a family of 5 per constraints",
        parent=hotel_node,
        critical=True
    )

    rooms_sources = merge_urls(hotel.rooms.urls, hotel.basic.basic_urls)

    # Existence: Rooms URL(s)
    evaluator.add_custom_node(
        result=len(hotel.rooms.urls) > 0,
        id=f"hotel_{idx}_rooms_url",
        desc="Supporting URL provided for room configuration/occupancy/crib-or-rollaway information",
        parent=rooms_node,
        critical=True
    )

    # Verify: Accommodates 5 via allowed configuration
    accommodates_leaf = evaluator.add_leaf(
        id=f"hotel_{idx}_accommodates_5_via_allowed_config",
        desc="Hotel offers accommodations for 5 guests within stated occupancy limits via (a) a two-bedroom suite of at least 900 sq ft OR (b) two connecting rooms",
        parent=rooms_node,
        critical=True
    )

    # Build claim based on extracted signals
    if (hotel.rooms.supports_family_config or hotel.rooms.suite_sqft) and (hotel.rooms.suite_sqft or "suite" in (hotel.rooms.supports_family_config or "").lower()):
        claim_accom = (
            f"The hotel offers a two-bedroom suite of at least 900 square feet and it can accommodate a family of 5."
        )
    elif hotel.rooms.connecting_rooms_available:
        claim_accom = (
            "The hotel offers two connecting/adjoining rooms that can be booked to accommodate a family of 5."
        )
    else:
        claim_accom = (
            "The hotel can accommodate a family of 5 via either a two-bedroom suite (>=900 sq ft) or two connecting rooms."
        )

    await evaluator.verify(
        claim=claim_accom,
        node=accommodates_leaf,
        sources=rooms_sources,
        additional_instruction=(
            "Look for phrases like 'two-bedroom suite', '900 sq ft', 'connecting rooms', 'adjoining rooms', "
            "'occupancy 5', or 'sleeps 5'. If any of these explicitly support the allowed configurations, mark as supported."
        )
    )

    # Verify: Cribs or rollaway availability
    cribs_leaf = evaluator.add_leaf(
        id=f"hotel_{idx}_cribs_or_rollaway_available",
        desc="Hotel can provide cribs or rollaway beds upon request",
        parent=rooms_node,
        critical=True
    )
    cribs_sources = merge_urls(hotel.rooms.urls, hotel.policies.urls, hotel.amenities.urls, hotel.basic.basic_urls)
    await evaluator.verify(
        claim="Cribs or rollaway beds are available upon request at this hotel.",
        node=cribs_leaf,
        sources=cribs_sources,
        additional_instruction="Confirm explicit mention of 'cribs' or 'rollaway beds' being available (often on room details or policies pages)."
    )

    # ----------------- Location Requirements -----------------
    location_node = evaluator.add_parallel(
        id=f"hotel_{idx}_location_requirements",
        desc="Hotel meets beach access and proximity constraints",
        parent=hotel_node,
        critical=True
    )

    location_sources = merge_urls(hotel.location.urls, hotel.basic.basic_urls)

    # Existence: Location URL(s)
    evaluator.add_custom_node(
        result=len(hotel.location.urls) > 0,
        id=f"hotel_{idx}_location_url",
        desc="Supporting URL provided for beachfront/beach-access and Miami Beach proximity information",
        parent=location_node,
        critical=True
    )

    # Verify: Beachfront or direct beach access
    beach_leaf = evaluator.add_leaf(
        id=f"hotel_{idx}_beachfront_or_direct_beach_access",
        desc="Hotel is beachfront or has direct beach access",
        parent=location_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel is beachfront (oceanfront) or has direct beach access.",
        node=beach_leaf,
        sources=location_sources,
        additional_instruction="Accept synonyms like 'oceanfront', 'on the beach', 'beach access', 'steps to beach'."
    )

    # Verify: Within Miami Beach area (~10–15 miles from MIA) – operationalized as 'in Miami Beach'
    mia_leaf = evaluator.add_leaf(
        id=f"hotel_{idx}_within_10_15_miles_mia",
        desc="Hotel is within the Miami Beach area (approximately 10–15 miles from Miami International Airport)",
        parent=location_node,
        critical=True
    )
    await evaluator.verify(
        claim="This hotel is located in Miami Beach, Florida.",
        node=mia_leaf,
        sources=location_sources,
        additional_instruction="Verifying location in 'Miami Beach' suffices for this proximity requirement; focus on the address/location."
    )

    # ----------------- Amenities -----------------
    amenities_node = evaluator.add_parallel(
        id=f"hotel_{idx}_amenities",
        desc="Hotel provides required on-site amenities",
        parent=hotel_node,
        critical=True
    )

    amenities_sources = merge_urls(hotel.amenities.urls, hotel.basic.basic_urls)

    # Existence: Amenities URL(s)
    evaluator.add_custom_node(
        result=len(hotel.amenities.urls) > 0,
        id=f"hotel_{idx}_amenities_url",
        desc="Supporting URL provided for pool/amenities information",
        parent=amenities_node,
        critical=True
    )

    # Verify: On-site swimming pool
    pool_leaf = evaluator.add_leaf(
        id=f"hotel_{idx}_onsite_pool",
        desc="Hotel has an on-site swimming pool accessible to guests",
        parent=amenities_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel has an on-site swimming pool accessible to guests.",
        node=pool_leaf,
        sources=amenities_sources,
        additional_instruction="Look for explicit mention of 'pool' among amenities/features."
    )

    # ----------------- Booking Policies -----------------
    policies_node = evaluator.add_parallel(
        id=f"hotel_{idx}_booking_policies",
        desc="Hotel policies meet stated constraints",
        parent=hotel_node,
        critical=True
    )

    policy_sources = merge_urls(hotel.policies.urls, hotel.basic.basic_urls)

    # Existence: Policies URL(s)
    evaluator.add_custom_node(
        result=len(hotel.policies.urls) > 0,
        id=f"hotel_{idx}_policies_url",
        desc="Supporting URL provided for check-in age/time and children policy information",
        parent=policies_node,
        critical=True
    )

    # Verify: Check-in age 18+
    checkin_age_leaf = evaluator.add_leaf(
        id=f"hotel_{idx}_checkin_age_18_plus",
        desc="Hotel allows check-in by an adult aged 18 or older when traveling with children",
        parent=policies_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel allows check-in by adults aged 18 or older (min check-in age is 18).",
        node=checkin_age_leaf,
        sources=policy_sources,
        additional_instruction="Confirm minimum check-in age policy; specifically verify '18+' allowance."
    )

    # Verify: Check-in time between 3:00 PM and 4:00 PM
    checkin_time_leaf = evaluator.add_leaf(
        id=f"hotel_{idx}_checkin_time_3_to_4_pm",
        desc="Hotel check-in time is stated as between 3:00 PM and 4:00 PM",
        parent=policies_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel's check-in time is between 3:00 PM and 4:00 PM.",
        node=checkin_time_leaf,
        sources=policy_sources,
        additional_instruction="Accept common phrasing like 'Check-in: 3 PM' or 'Check-in from 4 PM'; variations like 'after 3 PM' qualify."
    )

    # Verify: Children stay free policy
    children_free_leaf = evaluator.add_leaf(
        id=f"hotel_{idx}_children_stay_free_policy",
        desc="Hotel states a policy that children under a specified age (typically 12–18) can stay free when sharing a room with adults",
        parent=policies_node,
        critical=True
    )
    await evaluator.verify(
        claim="Children under a specified age can stay free when sharing a room with adults at this hotel.",
        node=children_free_leaf,
        sources=policy_sources,
        additional_instruction="Look for 'children stay free' or equivalent policy language on the hotel site."
    )

    # ----------------- Accessibility -----------------
    ada_node = evaluator.add_parallel(
        id=f"hotel_{idx}_accessibility",
        desc="Hotel meets accessibility constraint",
        parent=hotel_node,
        critical=True
    )

    ada_sources = merge_urls(hotel.ada.urls, hotel.policies.urls, hotel.basic.basic_urls)

    # Existence: ADA URL(s)
    evaluator.add_custom_node(
        result=len(hotel.ada.urls) > 0,
        id=f"hotel_{idx}_ada_url",
        desc="Supporting URL provided for ADA/accessibility information",
        parent=ada_node,
        critical=True
    )

    # Verify: ADA compliance
    ada_leaf = evaluator.add_leaf(
        id=f"hotel_{idx}_ada_compliance",
        desc="Hotel meets ADA requirements for accessibility features",
        parent=ada_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel meets ADA requirements or provides ADA-compliant accessible rooms/features.",
        node=ada_leaf,
        sources=ada_sources,
        additional_instruction="Confirm explicit mention of ADA compliance, accessible rooms, or accessibility features per ADA."
    )

    # ----------------- Transportation Access (Non-Critical) -----------------
    transport_node = evaluator.add_parallel(
        id=f"hotel_{idx}_transportation_access",
        desc="Preferred (non-mandatory) public transportation access",
        parent=hotel_node,
        critical=False
    )

    transport_sources = merge_urls(hotel.transport.urls, hotel.location.urls, hotel.basic.basic_urls)

    trolley_leaf = evaluator.add_leaf(
        id=f"hotel_{idx}_trolley_access",
        desc="Hotel has access to public transportation such as the Miami Beach Trolley or is within walkable distance to trolley stops",
        parent=transport_node,
        critical=False
    )
    await evaluator.verify(
        claim="The hotel has access to the Miami Beach Trolley or is within walking distance to trolley stops.",
        node=trolley_leaf,
        sources=transport_sources,
        additional_instruction="Accept mentions of nearby trolley stops, trolley routes, or hotel listings referencing the Miami Beach Trolley."
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
    Evaluate an answer for Miami Beach hotels suitability for a family of five with required constraints.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # hotels are evaluated independently
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

    # Extract structured hotel data
    extracted_hotels = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction"
    )

    # Normalize to exactly 3 hotels (pad if fewer)
    hotels_list: List[HotelItem] = list(extracted_hotels.hotels[:3])
    while len(hotels_list) < 3:
        hotels_list.append(HotelItem())

    # Record trip context as custom info
    evaluator.add_custom_info(
        info={
            "trip_month": "July 2026",
            "party": "2 adults + 3 children (ages 3, 7, 10)",
            "destination": "Miami Beach, Florida",
            "constraints_summary": [
                "Accommodations for 5 via two-bedroom suite (>=900 sq ft) or two connecting rooms",
                "Located in Miami Beach; beachfront or direct beach access",
                "On-site swimming pool",
                "Adults 18+ can check in with children",
                "Cribs or rollaway beds available upon request",
                "Preferred: access to Miami Beach Trolley or walkable to trolley stops"
            ]
        },
        info_type="context",
        info_name="trip_context"
    )

    # Build verification tree per hotel
    for i, hotel in enumerate(hotels_list):
        await verify_hotel(evaluator, root, hotel, i)

    return evaluator.get_summary()