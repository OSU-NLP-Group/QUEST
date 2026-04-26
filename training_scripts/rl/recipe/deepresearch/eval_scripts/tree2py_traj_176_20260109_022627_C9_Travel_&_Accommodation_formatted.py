import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "minneapolis_conference_hotels"
TASK_DESCRIPTION = """
A corporate event planning company is organizing a major technology conference in Minneapolis, Minnesota, expecting approximately 500 attendees. They need to identify 4 hotels in downtown Minneapolis that can serve as official conference hotels and meet all of the following requirements:

Conference Facilities:
- Must have a ballroom or event space with a minimum capacity of 500 guests for seated events
- Must have at least 10,000 square feet of total meeting space
- Must have at least 10 separate meeting rooms available for breakout sessions

Accessibility:
- Must offer ADA-compliant accessible guest rooms
- Accessible rooms must have doorways with a minimum of 32 inches clear width for wheelchair access
- Must have wheelchair-accessible bathrooms with roll-in showers or other accessible features

Amenities:
- Must have a fitness center with 24-hour access for hotel guests
- Must have an indoor heated swimming pool
- Must offer complimentary breakfast to guests

Parking:
- Must provide on-site parking facilities (either valet or self-parking)
- Overnight parking rate must not exceed $60 per night

Location:
- Must be located in downtown Minneapolis, Minnesota
- Must be within walking distance (1 mile or less) of major downtown attractions

Room Configurations:
- Must offer rooms with king bed options
- Must offer rooms with queen bed options
- Must have suite accommodations available

Service Policies:
- Standard check-in time must be 4:00 PM or earlier
- Must offer flexible cancellation policy allowing free cancellation if notice is given at least 24 hours before check-in

For each hotel identified, provide the hotel name, and for each requirement category, provide the specific information that satisfies the requirement along with a reference URL from the hotel's official website, booking platform, or reputable hotel information source that confirms each specification.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RequirementField(BaseModel):
    info: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HotelItem(BaseModel):
    name: Optional[str] = None

    ballroom_capacity: Optional[RequirementField] = None
    meeting_space_total_sqft: Optional[RequirementField] = None
    meeting_rooms_count: Optional[RequirementField] = None

    ada_accessible_guest_rooms: Optional[RequirementField] = None
    accessible_doorway_width: Optional[RequirementField] = None
    accessible_bathroom_features: Optional[RequirementField] = None

    fitness_center_24h: Optional[RequirementField] = None
    indoor_heated_pool: Optional[RequirementField] = None
    complimentary_breakfast: Optional[RequirementField] = None

    onsite_parking: Optional[RequirementField] = None
    overnight_parking_rate: Optional[RequirementField] = None

    location_downtown: Optional[RequirementField] = None
    distance_to_attractions: Optional[RequirementField] = None

    king_bed_option: Optional[RequirementField] = None
    queen_bed_option: Optional[RequirementField] = None
    suites_available: Optional[RequirementField] = None

    checkin_time: Optional[RequirementField] = None
    cancellation_policy: Optional[RequirementField] = None


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    Identify up to 6 hotels mentioned in the answer that are proposed for the downtown Minneapolis conference.

    For each hotel, extract the following data structure exactly with these keys:

    {
      "hotels": [
        {
          "name": string or null,

          "ballroom_capacity": { "info": string or null, "urls": [urls...] },
          "meeting_space_total_sqft": { "info": string or null, "urls": [urls...] },
          "meeting_rooms_count": { "info": string or null, "urls": [urls...] },

          "ada_accessible_guest_rooms": { "info": string or null, "urls": [urls...] },
          "accessible_doorway_width": { "info": string or null, "urls": [urls...] },
          "accessible_bathroom_features": { "info": string or null, "urls": [urls...] },

          "fitness_center_24h": { "info": string or null, "urls": [urls...] },
          "indoor_heated_pool": { "info": string or null, "urls": [urls...] },
          "complimentary_breakfast": { "info": string or null, "urls": [urls...] },

          "onsite_parking": { "info": string or null, "urls": [urls...] },
          "overnight_parking_rate": { "info": string or null, "urls": [urls...] },

          "location_downtown": { "info": string or null, "urls": [urls...] },
          "distance_to_attractions": { "info": string or null, "urls": [urls...] },

          "king_bed_option": { "info": string or null, "urls": [urls...] },
          "queen_bed_option": { "info": string or null, "urls": [urls...] },
          "suites_available": { "info": string or null, "urls": [urls...] },

          "checkin_time": { "info": string or null, "urls": [urls...] },
          "cancellation_policy": { "info": string or null, "urls": [urls...] }
        },
        ...
      ]
    }

    RULES:
    - info must be the specific detail stated in the answer (e.g., "Grand Ballroom 600 banquet", "12 meeting rooms", "11,000 sq ft", "Check-in 3:00 PM", "Free cancellation until 24 hrs prior", etc.). If missing, set to null.
    - urls must be a list of actual URLs explicitly present in the answer (official hotel site, booking platform, or reputable info source). If none provided, return an empty list.
    - Do not invent any data; extract only what appears in the answer.
    - Keep numbers and units exactly as written (e.g., "32 inches", "$55", "10,500 sq ft").
    - If the answer mentions more than 4 hotels, still extract them, but we will only use the first 4 in evaluation.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _field_exists(field: Optional[RequirementField]) -> bool:
    return bool(
        field is not None and
        field.info is not None and str(field.info).strip() != "" and
        field.urls is not None and len(field.urls) > 0
    )


async def _verify_requirement_with_urls(
    evaluator: Evaluator,
    parent_node,
    hotel_index: int,
    hotel_name: Optional[str],
    field: Optional[RequirementField],
    id_base: str,
    seq_desc: str,
    exist_desc: str,
    verify_desc: str,
    claim: str,
    add_ins: str,
) -> None:
    """
    Build a sequential two-step requirement:
      1) existence of specific info + at least one URL
      2) verification of the claim against the provided URLs
    """
    seq_node = evaluator.add_sequential(
        id=f"hotel_{hotel_index}_{id_base}",
        desc=seq_desc,
        parent=parent_node,
        critical=True
    )

    # Existence check (critical)
    exists = _field_exists(field)
    evaluator.add_custom_node(
        result=exists,
        id=f"hotel_{hotel_index}_{id_base}_exists",
        desc=exist_desc,
        parent=seq_node,
        critical=True
    )

    # Verification leaf (critical)
    verify_leaf = evaluator.add_leaf(
        id=f"hotel_{hotel_index}_{id_base}_supported",
        desc=verify_desc,
        parent=seq_node,
        critical=True
    )

    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=(field.urls if field else None),
        additional_instruction=add_ins
    )


async def verify_single_hotel(
    evaluator: Evaluator,
    parent_node,
    hotel: HotelItem,
    hotel_index: int
) -> None:
    """
    Build verification sub-tree for a single proposed hotel.
    All children requirements are critical under this hotel node to enforce "must meet all requirements".
    """
    hotel_node = evaluator.add_parallel(
        id=f"Hotel_{hotel_index + 1}",
        desc=f"Hotel #{hotel_index + 1} verification - {hotel.name or 'Unnamed'}",
        parent=parent_node,
        critical=True
    )

    # Hotel name provided (critical existence check)
    name_exists = hotel.name is not None and hotel.name.strip() != ""
    evaluator.add_custom_node(
        result=name_exists,
        id=f"hotel_{hotel_index}_name_provided",
        desc="Provide the hotel name.",
        parent=hotel_node,
        critical=True
    )

    # Conference Facilities
    await _verify_requirement_with_urls(
        evaluator, hotel_node, hotel_index, hotel.name, hotel.ballroom_capacity,
        id_base="ballroom_capacity_500", seq_desc="Ballroom / event space capacity requirement",
        exist_desc="Specific ballroom/event capacity info with at least one confirming URL is provided",
        verify_desc="Ballroom/event space supports ≥ 500 seated guests (supported by cited source)",
        claim="The hotel's ballroom or event space can accommodate at least 500 seated guests.",
        add_ins="Use capacity charts or event space pages; treat 'banquet' as seated. A single room or a combined configuration meeting ≥500 counts."
    )

    await _verify_requirement_with_urls(
        evaluator, hotel_node, hotel_index, hotel.name, hotel.meeting_space_total_sqft,
        id_base="meeting_space_total_10000sqft", seq_desc="Total meeting space requirement",
        exist_desc="Total meeting space info with at least one confirming URL is provided",
        verify_desc="Total meeting space is ≥ 10,000 sq ft (supported by cited source)",
        claim="The hotel has at least 10,000 square feet of total meeting space.",
        add_ins="Check meeting space overview or factsheet; accept 'over 10,000', 'approximately 10,000+', etc. Numeric comparisons ≥ 10,000 sq ft."
    )

    await _verify_requirement_with_urls(
        evaluator, hotel_node, hotel_index, hotel.name, hotel.meeting_rooms_count,
        id_base="meeting_rooms_10plus", seq_desc="Breakout rooms count requirement",
        exist_desc="Number of separate meeting rooms info with at least one confirming URL is provided",
        verify_desc="There are at least 10 separate meeting rooms (supported by cited source)",
        claim="The hotel offers at least 10 separate meeting rooms suitable for breakout sessions.",
        add_ins="Look for 'meeting rooms', 'breakout rooms', or capacity charts listing room counts. Threshold is ≥10."
    )

    # Accessibility
    await _verify_requirement_with_urls(
        evaluator, hotel_node, hotel_index, hotel.name, hotel.ada_accessible_guest_rooms,
        id_base="ada_accessible_rooms", seq_desc="ADA accessible guest rooms requirement",
        exist_desc="ADA-compliant accessible guest rooms info with at least one confirming URL is provided",
        verify_desc="ADA-compliant accessible guest rooms are offered (supported by cited source)",
        claim="The hotel offers ADA-compliant accessible guest rooms.",
        add_ins="Accept phrasing such as 'accessible rooms', 'ADA rooms', 'mobility accessible'. The page must explicitly indicate ADA/accessible rooms offered."
    )

    await _verify_requirement_with_urls(
        evaluator, hotel_node, hotel_index, hotel.name, hotel.accessible_doorway_width,
        id_base="doorway_32in_clear_width", seq_desc="Accessible doorway width requirement",
        exist_desc="Accessible room doorway clear width info (≥ 32 inches) with at least one confirming URL is provided",
        verify_desc="Accessible guest room doorways provide ≥ 32 inches clear width (supported by cited source)",
        claim="Accessible guest room doorways provide at least 32 inches of clear width.",
        add_ins="Look for ADA specifications for room doorways. Accept 'minimum 32 inches of clear opening'."
    )

    await _verify_requirement_with_urls(
        evaluator, hotel_node, hotel_index, hotel.name, hotel.accessible_bathroom_features,
        id_base="accessible_bathrooms_rollin", seq_desc="Accessible bathroom features requirement",
        exist_desc="Accessible bathroom features (e.g., roll-in showers) info with at least one confirming URL is provided",
        verify_desc="Accessible bathrooms with roll-in showers or equivalent features are available (supported by cited source)",
        claim="The hotel offers accessible bathrooms such as roll-in showers or equivalent accessible features.",
        add_ins="Accept phrasing like 'roll-in shower', 'accessible tub with grab bars', 'wheelchair-accessible bathroom'."
    )

    # Amenities
    await _verify_requirement_with_urls(
        evaluator, hotel_node, hotel_index, hotel.name, hotel.fitness_center_24h,
        id_base="fitness_24h", seq_desc="Fitness center 24-hour access requirement",
        exist_desc="Fitness center 24-hour access info with at least one confirming URL is provided",
        verify_desc="Fitness center provides 24-hour access (supported by cited source)",
        claim="The hotel provides a fitness center with 24-hour access for guests.",
        add_ins="Page should indicate 24-hour or 'open 24/7'. If hours are stated, confirm coverage 00:00–24:00."
    )

    await _verify_requirement_with_urls(
        evaluator, hotel_node, hotel_index, hotel.name, hotel.indoor_heated_pool,
        id_base="indoor_heated_pool", seq_desc="Indoor heated swimming pool requirement",
        exist_desc="Indoor heated pool info with at least one confirming URL is provided",
        verify_desc="An indoor heated swimming pool is available (supported by cited source)",
        claim="The hotel has an indoor heated swimming pool.",
        add_ins="Look for amenities or pool page indicating both 'indoor' and 'heated'."
    )

    await _verify_requirement_with_urls(
        evaluator, hotel_node, hotel_index, hotel.name, hotel.complimentary_breakfast,
        id_base="complimentary_breakfast", seq_desc="Complimentary breakfast requirement",
        exist_desc="Complimentary breakfast info with at least one confirming URL is provided",
        verify_desc="Complimentary breakfast is offered (supported by cited source)",
        claim="The hotel offers complimentary breakfast to guests.",
        add_ins="Accept 'complimentary', 'free breakfast', or 'breakfast included' with standard rates; exclude paid-only breakfast."
    )

    # Parking
    await _verify_requirement_with_urls(
        evaluator, hotel_node, hotel_index, hotel.name, hotel.onsite_parking,
        id_base="onsite_parking", seq_desc="On-site parking requirement",
        exist_desc="On-site parking (valet or self) info with at least one confirming URL is provided",
        verify_desc="On-site parking (valet or self) is provided (supported by cited source)",
        claim="The hotel provides on-site parking (valet or self-parking).",
        add_ins="Accept valet or self-parking on property; off-site only does not meet requirement."
    )

    await _verify_requirement_with_urls(
        evaluator, hotel_node, hotel_index, hotel.name, hotel.overnight_parking_rate,
        id_base="overnight_parking_rate_60max", seq_desc="Overnight parking rate requirement",
        exist_desc="Overnight parking rate info with at least one confirming URL is provided",
        verify_desc="Overnight parking rate is $60/night or less (supported by cited source)",
        claim="The overnight parking rate is $60 per night or less.",
        add_ins="Confirm the nightly rate for overnight parking (valet or self). Consider most common/standard rate. Threshold ≤ $60."
    )

    # Location
    await _verify_requirement_with_urls(
        evaluator, hotel_node, hotel_index, hotel.name, hotel.location_downtown,
        id_base="located_downtown_mpls", seq_desc="Downtown Minneapolis location requirement",
        exist_desc="Downtown Minneapolis location info with at least one confirming URL is provided",
        verify_desc="Hotel is located in downtown Minneapolis, Minnesota (supported by cited source)",
        claim="The hotel is located in downtown Minneapolis, Minnesota.",
        add_ins="Accept official address or page explicitly stating 'downtown Minneapolis'."
    )

    await _verify_requirement_with_urls(
        evaluator, hotel_node, hotel_index, hotel.name, hotel.distance_to_attractions,
        id_base="within_1mile_attractions", seq_desc="Proximity to major downtown attractions requirement",
        exist_desc="Within 1 mile of major downtown attractions info with at least one confirming URL is provided",
        verify_desc="Hotel is within 1 mile (walking distance) of major downtown attractions (supported by cited source)",
        claim="The hotel is within 1 mile (walking distance) of major downtown attractions such as Nicollet Mall, Target Center, U.S. Bank Stadium, or the Minneapolis Convention Center.",
        add_ins="Prefer explicit distances ≤ 1 mile or statements 'within 1 mile'. If only 'walking distance' is claimed, ensure listed attraction(s) are typical major downtown venues."
    )

    # Room Configurations
    await _verify_requirement_with_urls(
        evaluator, hotel_node, hotel_index, hotel.name, hotel.king_bed_option,
        id_base="king_bed_option", seq_desc="King bed option requirement",
        exist_desc="King bed room option info with at least one confirming URL is provided",
        verify_desc="Rooms with king bed options are available (supported by cited source)",
        claim="Rooms with king bed options are available at the hotel.",
        add_ins="Accept 'King Room', '1 King Bed', 'King suite', etc. Must be explicitly stated."
    )

    await _verify_requirement_with_urls(
        evaluator, hotel_node, hotel_index, hotel.name, hotel.queen_bed_option,
        id_base="queen_bed_option", seq_desc="Queen bed option requirement",
        exist_desc="Queen bed room option info with at least one confirming URL is provided",
        verify_desc="Rooms with queen bed options are available (supported by cited source)",
        claim="Rooms with queen bed options are available at the hotel.",
        add_ins="Accept 'Queen Room', '2 Queen Beds', '1 Queen Bed', etc. Must be explicitly stated."
    )

    await _verify_requirement_with_urls(
        evaluator, hotel_node, hotel_index, hotel.name, hotel.suites_available,
        id_base="suites_available", seq_desc="Suite accommodations requirement",
        exist_desc="Suites availability info with at least one confirming URL is provided",
        verify_desc="Suite accommodations are available (supported by cited source)",
        claim="Suite accommodations are available at the hotel.",
        add_ins="Accept 'suite', 'king suite', 'executive suite', etc."
    )

    # Service Policies
    await _verify_requirement_with_urls(
        evaluator, hotel_node, hotel_index, hotel.name, hotel.checkin_time,
        id_base="checkin_time_4pm_or_earlier", seq_desc="Standard check-in time requirement",
        exist_desc="Standard check-in time info with at least one confirming URL is provided",
        verify_desc="Standard check-in time is 4:00 PM or earlier (supported by cited source)",
        claim="The hotel's standard check-in time is 4:00 PM or earlier.",
        add_ins="If multiple times are listed, use the standard/default. Accept 3:00 PM, 4:00 PM, or any time earlier than 4:00 PM."
    )

    await _verify_requirement_with_urls(
        evaluator, hotel_node, hotel_index, hotel.name, hotel.cancellation_policy,
        id_base="cancellation_free_24h_before", seq_desc="Flexible cancellation policy requirement",
        exist_desc="Cancellation policy info with at least one confirming URL is provided",
        verify_desc="Free cancellation is allowed with at least 24 hours notice before check-in (supported by cited source)",
        claim="The hotel offers free cancellation with at least 24 hours notice before check-in.",
        add_ins="Look for flexible or standard rates allowing free cancellation until ≥24 hours prior. Non-refundable rates do not count."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Minneapolis conference hotels task.
    """
    # Initialize evaluator
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
        default_model=model,
    )

    # Create a critical wrapper node to enforce that all four hotels must pass
    task_node = evaluator.add_parallel(
        id="Find_4_Suitable_Hotels",
        desc="Identify 4 hotels in downtown Minneapolis that meet all specified requirements with confirming URLs.",
        parent=root,
        critical=True
    )

    # Extract hotels data
    extracted_hotels = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction",
    )

    # Prepare the set of hotels to verify: first 4, pad with empty if fewer
    hotels_to_check: List[HotelItem] = list(extracted_hotels.hotels[:4])
    while len(hotels_to_check) < 4:
        hotels_to_check.append(HotelItem())

    # Build verification subtrees for each of the 4 hotels
    for i, hotel in enumerate(hotels_to_check):
        await verify_single_hotel(evaluator, task_node, hotel, i)

    # Return structured summary
    return evaluator.get_summary()