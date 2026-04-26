import asyncio
import logging
from typing import Any, List, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "orlando_conference_hotel"
TASK_DESCRIPTION = (
    "Identify a hotel in Orlando, Florida that is suitable for hosting a corporate conference with 50 attendees "
    "requiring overnight accommodations for 30 participants. The hotel must meet all of the following requirements: "
    "(1) be located within walking distance (0.5 miles or less) of downtown Orlando or the Orange County Convention Center; "
    "(2) provide high-speed, reliable Wi-Fi in both guest rooms and common areas suitable for business use; "
    "(3) have flexible meeting spaces that support multiple seating configurations such as classroom, boardroom, or theater arrangements; "
    "(4) provide shuttle service to/from Orlando International Airport or major transportation hubs, or offer concierge-arranged transportation assistance; "
    "(5) have a business center accessible 24 hours a day with printing, scanning, and copying services; "
    "(6) have on-site wellness and fitness facilities such as a gym or fitness center; "
    "(7) have on-site dining options including a restaurant or café that serves meals throughout the day; "
    "(8) provide or have access to audiovisual equipment including projectors, screens, and sound systems for meetings; "
    "(9) have parking facilities available on-site or nearby for guest vehicles; "
    "(10) comply with ADA accessibility requirements including accessible guest rooms and common areas; "
    "(11) accept group bookings of 10 or more rooms; "
    "(12) provide on-site catering services for meetings and events; "
    "(13) have at least one meeting room that can accommodate a minimum of 50 attendees; and "
    "(14) be able to accommodate a room block of at least 30 guest rooms for the conference dates. "
    "Provide the hotel's name, location details, and a reference URL confirming it meets these requirements."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelInfo(BaseModel):
    hotel_name: Optional[str] = None
    location_details: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_info() -> str:
    return (
        "Extract the single hotel identified in the answer that is recommended for the described corporate conference. "
        "Return the following fields:\n"
        "1. hotel_name: The hotel's official name.\n"
        "2. location_details: Any location/address details, including references to downtown Orlando or the Orange County Convention Center proximity.\n"
        "3. source_urls: An array of all reference URLs explicitly provided in the answer that substantiate the hotel's features/requirements. "
        "Include the hotel's official pages (meetings/events, amenities, dining, accessibility) and any external verification URLs mentioned. "
        "Do not invent URLs; only extract those explicitly present in the answer. If a field is missing, return null (or empty array for source_urls)."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _name_or_generic(name: Optional[str]) -> str:
    return name.strip() if (name and name.strip()) else "the hotel"


def _normalize_sources(urls: List[str]) -> List[str]:
    # Deduplicate and keep non-empty strings
    seen = set()
    result: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if not uu:
            continue
        if uu not in seen:
            seen.add(uu)
            result.append(uu)
    return result


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_hotel_requirements(
    evaluator: Evaluator,
    parent_node,
    info: HotelInfo,
) -> None:
    """
    Build the verification tree and run checks for all hotel requirements.
    """
    hotel_name = _name_or_generic(info.hotel_name)
    sources = _normalize_sources(info.source_urls)

    # Create a critical parent node for all requirements
    hotel_req_node = evaluator.add_parallel(
        id="Hotel_Meeting_Requirements",
        desc="The identified hotel meets all specified requirements for hosting the corporate conference and the response includes the required identifying information.",
        parent=parent_node,
        critical=True,
    )

    # 0) Existence check: response provides name, location details, and at least one reference URL
    existence_ok = bool(info.hotel_name and info.hotel_name.strip()) and bool(info.location_details and info.location_details.strip()) and bool(sources)
    evaluator.add_custom_node(
        result=existence_ok,
        id="Response_Provides_Hotel_Name_Location_And_Reference_URL",
        desc="The response provides the hotel's name, location details, and at least one reference URL that can be used to verify the stated requirements.",
        parent=hotel_req_node,
        critical=True,
    )

    # Prepare all leaf nodes and associated verification tuples for batch verification
    claims_and_sources: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    def add_leaf_and_claim(node_id: str, desc: str, claim: str, add_ins: str) -> None:
        node = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=hotel_req_node,
            critical=True,
        )
        claims_and_sources.append((
            claim,
            sources if sources else None,
            node,
            add_ins
        ))

    # 1) Location proximity (Orlando, and within 0.5 miles walking distance of downtown Orlando or OCCC)
    add_leaf_and_claim(
        "Location_Proximity_Constraint",
        "The hotel is located in Orlando, Florida within walking distance (0.5 miles or less) of downtown Orlando or the Orange County Convention Center.",
        f"{hotel_name} is located in Orlando, Florida and is within walking distance (0.5 miles or less) of downtown Orlando or the Orange County Convention Center.",
        (
            "Check the hotel's official site or cited sources for explicit statements of proximity such as: "
            "'0.5 miles', 'walking distance', 'short walk', 'adjacent', 'connected to', 'across the street', 'on convention center property'. "
            "If exact distance is not stated but the text clearly indicates adjacency/connection to OCCC or downtown (e.g., 'connected to the convention center' or 'on property'), "
            "consider it within 0.5 miles. If the sources do not clearly support Orlando location and proximity, mark as not supported."
        )
    )

    # 2) Wi-Fi infrastructure (guest rooms and common areas, suitable for business use)
    add_leaf_and_claim(
        "WiFi_Infrastructure",
        "The hotel provides high-speed, reliable Wi-Fi in both guest rooms and common areas, suitable for business use.",
        f"{hotel_name} provides high-speed, reliable Wi‑Fi (internet) in guest rooms and common/public areas suitable for business use.",
        (
            "Look for amenity descriptions like 'high-speed Wi-Fi', 'high-speed internet', 'complimentary Wi-Fi', "
            "'Wi-Fi in guest rooms and public areas', or business-friendly internet features."
        )
    )

    # 3) Flexible meeting spaces and multiple seating configurations
    add_leaf_and_claim(
        "Flexible_Meeting_Spaces",
        "The hotel has flexible meeting spaces that support multiple seating configurations such as classroom, boardroom, or theater arrangements.",
        f"{hotel_name} offers flexible meeting/event spaces with multiple seating configurations (e.g., classroom, boardroom, theater).",
        (
            "Check meeting and events pages for 'room setups' or 'configurations' mentioning classroom, theater, boardroom, banquet, U-shape, etc. "
            "Any clear mention of multiple configurations suffices."
        )
    )

    # 4) Transportation services (airport shuttle or concierge-arranged transportation)
    add_leaf_and_claim(
        "Transportation_Services",
        "The hotel provides shuttle service to/from Orlando International Airport or major transportation hubs, or offers concierge-arranged transportation assistance.",
        f"{hotel_name} provides airport shuttle or concierge-arranged transportation assistance to Orlando International Airport (MCO) or major hubs.",
        (
            "Look for 'airport shuttle', 'transportation assistance', 'concierge can arrange rides', references to 'MCO', "
            "or partnerships with transportation providers."
        )
    )

    # 5) Business center 24/7 with printing, scanning, copying
    add_leaf_and_claim(
        "Business_Center_Access",
        "The hotel has a business center accessible 24 hours a day with printing, scanning, and copying services.",
        f"{hotel_name} has a 24-hour business center providing printing, scanning, and copying services.",
        (
            "Look for '24-hour business center' or specific services like printing, scanning, copying. "
            "If the business center hours or services are clearly stated, pass."
        )
    )

    # 6) Fitness facilities on-site
    add_leaf_and_claim(
        "Fitness_Facilities",
        "The hotel has on-site wellness and fitness facilities such as a gym, fitness center, or exercise room.",
        f"{hotel_name} has on-site fitness facilities (e.g., gym or fitness center).",
        "Check amenities pages for 'fitness center', 'gym', or similar on-site wellness facilities."
    )

    # 7) On-site dining options serving meals throughout the day
    add_leaf_and_claim(
        "Dining_Options",
        "The hotel has on-site dining options including a restaurant or café that serves meals throughout the day.",
        f"{hotel_name} offers on-site dining (restaurant/café) serving meals throughout the day (breakfast, lunch, and dinner).",
        (
            "Look for on-site restaurant or café information and references to breakfast/lunch/dinner availability or all-day dining."
        )
    )

    # 8) AV equipment (projectors, screens, sound systems)
    add_leaf_and_claim(
        "AV_Equipment",
        "The hotel provides or has access to audiovisual equipment including projectors, screens, and sound systems for meetings.",
        f"{hotel_name} provides or has access to audiovisual equipment for meetings, including projectors, screens, and sound systems.",
        (
            "Check meetings/events or catering/AV services pages for 'AV equipment', 'projectors', 'screens', 'sound system', "
            "or on-site/partner AV provider availability."
        )
    )

    # 9) Parking availability
    add_leaf_and_claim(
        "Parking_Availability",
        "The hotel has parking facilities available on-site or nearby for guest vehicles.",
        f"{hotel_name} provides on-site or nearby parking for guest vehicles.",
        "Look for 'parking', 'on-site parking', 'self-parking', or directions/parking info pages."
    )

    # 10) ADA accessibility (accessible rooms and common areas)
    add_leaf_and_claim(
        "ADA_Accessibility",
        "The hotel complies with ADA accessibility requirements, including accessible guest rooms and common areas.",
        f"{hotel_name} complies with ADA accessibility requirements, with accessible guest rooms and common areas.",
        (
            "Look for 'ADA accessible', 'accessible rooms', 'accessible public areas', or accessibility statements/policies "
            "on the hotel site."
        )
    )

    # 11) Group booking capability (10+ rooms)
    add_leaf_and_claim(
        "Group_Booking_Capability",
        "The hotel accepts group bookings of 10 or more rooms.",
        f"{hotel_name} accepts group bookings of 10 or more rooms (room blocks).",
        (
            "Check group sales/meetings pages for references to 'group bookings', 'room blocks', 'group rates', or minimums like 10 rooms. "
            "If explicitly stated or implied by group block policies, pass."
        )
    )

    # 12) On-site catering services
    add_leaf_and_claim(
        "Catering_Services",
        "The hotel provides on-site catering services for meetings and events.",
        f"{hotel_name} provides on-site catering services for meetings and events.",
        (
            "Check meetings/events or dining pages for 'catering', 'banquet services', 'event catering', or menus."
        )
    )

    # 13) Meeting room capacity minimum 50 attendees
    add_leaf_and_claim(
        "Meeting_Room_Capacity",
        "The hotel has at least one meeting room that can accommodate a minimum of 50 attendees.",
        f"{hotel_name} has at least one meeting room that can accommodate 50 or more attendees.",
        (
            "Look for capacity charts or specs listing room capacities by setup (e.g., theater/classroom/banquet) with numbers >= 50."
        )
    )

    # 14) Room block capacity of at least 30 guest rooms
    add_leaf_and_claim(
        "Room_Block_Capacity",
        "The hotel can accommodate a room block of at least 30 guest rooms for the conference dates.",
        f"{hotel_name} can accommodate a room block of at least 30 guest rooms for the conference dates.",
        (
            "Check group booking or sales pages for 'room blocks', 'block of rooms', or minimum/maximum block sizes. "
            "If the site indicates capacity for large room blocks (>=30) or explicitly states 30+ rooms, pass. "
            "If unclear or not supported, fail."
        )
    )

    # Execute all verifications in parallel for efficiency
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the Orlando corporate conference hotel task.
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
        default_model=model,
    )

    # Extract the hotel's basic information and cited sources from the answer
    hotel_info = await evaluator.extract(
        prompt=prompt_extract_hotel_info(),
        template_class=HotelInfo,
        extraction_name="hotel_info",
    )

    # Build tree and verify requirements
    await verify_hotel_requirements(evaluator, root, hotel_info)

    # Return standardized summary
    return evaluator.get_summary()