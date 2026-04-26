import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "family_beach_vacation_florida_gulf_swa"
TASK_DESCRIPTION = """
I am planning a family beach vacation from Phoenix, Arizona to Florida's Gulf Coast. We are a family of five (2 adults and 3 children) and will be flying Southwest Airlines. I need to find three beachfront or oceanfront hotels that meet the following requirements:

1. The hotel must be located in a Florida Gulf Coast destination that is served by Southwest Airlines with flights from Phoenix Sky Harbor International Airport (PHX).

2. The hotel must offer family rooms or suites that can accommodate all 5 of us in one room. We prefer configurations like two queen beds plus a sofa bed.

3. The room must include both a refrigerator and a microwave as in-room amenities, as we like to store snacks and prepare simple meals.

4. The hotel must provide complimentary airport shuttle service to/from the local airport.

For each of the three hotels, please provide:
- The hotel name and its beachfront location (city and airport code)
- A direct link to the hotel's website or a major booking platform (such as Expedia, Hotels.com, Booking.com, or TripAdvisor) showing the hotel and confirming it meets these requirements
- Specific details about the room type that accommodates 5 people, and confirmation of the refrigerator and microwave amenities
- Confirmation of the free airport shuttle service

Additionally, please provide current information about Southwest Airlines' baggage policy for flights from Phoenix, including:
- The fees for checked bags (first and second bag)
- Any Southwest status tiers or fare classes that receive free checked bags
"""

ALLOWED_BOOKING_DOMAINS = [
    "expedia.", "hotels.com", "booking.com", "tripadvisor.",
    "marriott.", "hilton.", "hyatt.", "ihg.", "omni", "omnihotels",
    "wyndham", "choicehotels", "bestwestern.", "accor.", "radisson.",
    "kimpton.", "sonesta.", "intercontinental."
]

# Southwest fallback information sources (for destination service verification)
SWA_FALLBACK_URLS = [
    "https://www.southwest.com/destinations/",
    "https://www.southwest.com/route-map/",
    "https://www.southwest.com/help/baggage/baggage-policies",
    "https://en.wikipedia.org/wiki/List_of_Southwest_Airlines_destinations",
    "https://en.wikipedia.org/wiki/Phoenix_Sky_Harbor_International_Airport"
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    # Core identification
    name: Optional[str] = None
    city: Optional[str] = None
    airport_code: Optional[str] = None  # IATA code like TPA, RSW, SRQ, PNS, ECP, etc.

    # URLs
    url: Optional[str] = None  # main hotel or booking link cited in the answer
    additional_urls: List[str] = Field(default_factory=list)  # any other cited links for the hotel
    southwest_urls: List[str] = Field(default_factory=list)  # any cited links referencing Southwest service

    # Property/location descriptors extracted from the answer
    beachfront_text: Optional[str] = None  # phrases like beachfront/oceanfront/Gulf-front

    # Room information
    room_type: Optional[str] = None
    capacity_text: Optional[str] = None  # e.g., "Sleeps 5", "fits 5 in one room"
    bed_configuration: Optional[str] = None  # e.g., "2 queen beds + sofa bed"

    # Amenities and shuttle
    refrigerator_in_room: Optional[str] = None
    microwave_in_room: Optional[str] = None
    shuttle_text: Optional[str] = None  # should mention complimentary/free airport shuttle


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


class BaggageInfo(BaseModel):
    first_bag_fee: Optional[str] = None
    second_bag_fee: Optional[str] = None
    free_bags_tiers_or_fares: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    Extract up to the first three (3) beachfront or oceanfront hotels that the answer proposes as meeting the user's requirements.

    For each hotel, extract these fields:
    - name: The hotel name as written in the answer.
    - city: The Florida city (Gulf Coast destination) for the hotel.
    - airport_code: The local airport's 3-letter IATA code (e.g., TPA, RSW, SRQ, PNS, ECP) as stated in the answer.
    - url: A direct link to the hotel's official site or a major booking platform cited in the answer (Expedia, Hotels.com, Booking.com, TripAdvisor, or major brand websites such as Marriott/Hilton/Hyatt/IHG/Wyndham/etc.). If multiple links are present, pick the most primary one that demonstrates the hotel's details.
    - additional_urls: Any other URLs cited in the answer that relate to this hotel (room details page, amenities page, etc.).
    - southwest_urls: Any URLs cited in the answer that reference Southwest Airlines routes/destinations/airport service for this destination.
    - beachfront_text: Any explicit phrasing in the answer that the property is beachfront, oceanfront, Gulf-front, or directly on the beach/waterfront.
    - room_type: The specific room/suite name that is claimed to fit 5 people in one room.
    - capacity_text: The exact occupancy wording (e.g., "Sleeps 5", "fits 5", etc.) if present in the answer.
    - bed_configuration: The described bedding arrangement (e.g., "two queen beds plus a sofa bed").
    - refrigerator_in_room: The answer's statement or wording confirming an in-room refrigerator is included.
    - microwave_in_room: The answer's statement or wording confirming an in-room microwave is included.
    - shuttle_text: The answer's statement or wording confirming complimentary/free airport shuttle is provided.

    Important:
    - Extract ONLY information explicitly present in the answer text verbatim.
    - For any field missing in the answer, return null or an empty list as appropriate.
    - For URLs, extract only valid URLs explicitly mentioned. Do not invent URLs.
    - Maintain the original order from the answer; if more than 3 hotels are present, keep only the first 3.
    """


def prompt_extract_baggage() -> str:
    return """
    Extract the CURRENT Southwest Airlines checked baggage policy information as stated in the answer, specifically for flights from Phoenix (PHX).

    Return:
    - first_bag_fee: The fee (e.g., "$0", "free", "$X") for the first checked bag as stated.
    - second_bag_fee: The fee for the second checked bag as stated.
    - free_bags_tiers_or_fares: A list of any Southwest status tiers or fare classes mentioned that receive free checked bags (e.g., "all passengers", "Business Select", "military", etc.). If the answer says everyone gets the first two bags free, include "all passengers".
    - source_urls: Any URLs the answer cites that support this baggage policy.

    Notes:
    - Extract EXACTLY as stated in the answer. If not mentioned, return null or an empty list.
    - Only extract URLs explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip().lower()
    return u.startswith("http://") or u.startswith("https://")


def is_allowed_booking_or_brand_url(url: Optional[str]) -> bool:
    if not is_valid_url(url):
        return False
    u = url.strip().lower()
    return any(domain in u for domain in ALLOWED_BOOKING_DOMAINS) or True  # allow general hotel domains by default


def hotel_sources(h: HotelItem) -> List[str]:
    urls: List[str] = []
    if is_valid_url(h.url):
        urls.append(h.url)  # main first
    for u in h.additional_urls:
        if is_valid_url(u):
            urls.append(u)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def swa_service_sources(h: HotelItem) -> List[str]:
    urls: List[str] = []
    for u in h.southwest_urls:
        if is_valid_url(u):
            urls.append(u)
    # Always append some robust fallbacks to help verification
    urls.extend(SWA_FALLBACK_URLS)
    # Deduplicate
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_single_hotel(evaluator: Evaluator, parent_node, idx: int, hotel: HotelItem) -> None:
    """
    Build verification subtree and run checks for one hotel.
    """
    hotel_node = evaluator.add_parallel(
        id=f"Hotel_{idx+1}",
        desc=f"{['First','Second','Third'][idx]} qualifying beachfront hotel meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # 1) URL Reference (critical existence/validity check)
    url_ok = is_valid_url(hotel.url)
    evaluator.add_custom_node(
        result=url_ok,
        id=f"Hotel_{idx+1}_URL_Reference",
        desc=f"Provide verifiable URL to Hotel {idx+1}'s official website or major booking platform listing (Expedia, Hotels.com, Booking.com, or TripAdvisor)",
        parent=hotel_node,
        critical=True
    )

    # 2) Property requirements (parallel, non-critical group)
    prop_node = evaluator.add_parallel(
        id=f"Hotel_{idx+1}_Property_Requirements",
        desc=f"Hotel {idx+1} must satisfy all property-level requirements",
        parent=hotel_node,
        critical=False
    )

    # 2.1 Identification (critical existence of basic info)
    id_ok = (hotel.name is not None and hotel.name.strip() != "") and \
            (hotel.city is not None and hotel.city.strip() != "") and \
            (hotel.airport_code is not None and hotel.airport_code.strip() != "")

    evaluator.add_custom_node(
        result=id_ok,
        id=f"Hotel_{idx+1}_Identification",
        desc=f"Provide the hotel name, city location, and airport code for the destination",
        parent=prop_node,
        critical=True
    )

    # 2.2 Location parent node (critical) → split into two concrete leaves
    loc_parent = evaluator.add_parallel(
        id=f"Hotel_{idx+1}_Location",
        desc="Hotel must be beachfront or oceanfront property located in a Florida Gulf Coast destination that is served by Southwest Airlines from PHX",
        parent=prop_node,
        critical=True
    )

    # 2.2.a Beachfront/oceanfront in a Florida Gulf Coast city
    beachfront_leaf = evaluator.add_leaf(
        id=f"Hotel_{idx+1}_Beachfront",
        desc=f"Hotel {idx+1} is beachfront or oceanfront in a Florida Gulf Coast city",
        parent=loc_parent,
        critical=True
    )
    beachfront_claim = (
        f"The property named '{hotel.name or 'the hotel'}' is beachfront or oceanfront and is located in {hotel.city or 'a Florida Gulf Coast city'}."
        " Accept related phrases such as 'on the beach', 'Gulf-front', 'on the Gulf of Mexico', or 'oceanfront'."
    )
    await evaluator.verify(
        claim=beachfront_claim,
        node=beachfront_leaf,
        sources=hotel_sources(hotel),
        additional_instruction="Use the provided hotel or booking URLs. The page should explicitly indicate beachfront/oceanfront or an equivalent phrase."
    )

    # 2.2.b Southwest service (destination served + PHX served)
    swa_service_leaf = evaluator.add_leaf(
        id=f"Hotel_{idx+1}_Southwest_Service",
        desc=f"Destination airport {hotel.airport_code or ''} is served by Southwest; PHX is also a Southwest airport (implying PHX→destination possible)",
        parent=loc_parent,
        critical=True
    )
    swa_claim = (
        f"Southwest Airlines serves the destination airport {hotel.airport_code or '[airport code]'} in/near {hotel.city or '[city]'}, Florida, "
        f"and Southwest also operates from Phoenix Sky Harbor (PHX), so it is possible to book Southwest flights from PHX to {hotel.airport_code or '[airport code]'} (with or without connections)."
    )
    await evaluator.verify(
        claim=swa_claim,
        node=swa_service_leaf,
        sources=swa_service_sources(hotel),
        additional_instruction="You may use official Southwest pages or reliable listings (including Wikipedia's Southwest destinations page). It's sufficient to verify that Southwest serves both airports (PHX and the destination airport)."
    )

    # 2.3 Complimentary airport shuttle
    shuttle_leaf = evaluator.add_leaf(
        id=f"Hotel_{idx+1}_Airport_Shuttle",
        desc=f"Hotel must provide complimentary airport shuttle service to/from the local airport",
        parent=prop_node,
        critical=True
    )
    shuttle_claim = (
        f"The hotel '{hotel.name or 'the hotel'}' provides a complimentary (free) airport shuttle service to/from the local airport."
    )
    await evaluator.verify(
        claim=shuttle_claim,
        node=shuttle_leaf,
        sources=hotel_sources(hotel),
        additional_instruction="Look for explicit language such as 'complimentary airport shuttle' or 'free airport shuttle'."
    )

    # 2.4 Room capacity (sleeps 5 in one room, e.g., two queens + sofa bed)
    capacity_leaf = evaluator.add_leaf(
        id=f"Hotel_{idx+1}_Room_Capacity",
        desc=f"Hotel must offer family rooms/suites that accommodate 5 people in one room (e.g., 2 queens + sofa bed)",
        parent=prop_node,
        critical=True
    )
    capacity_claim = (
        f"The room type '{hotel.room_type or 'a suitable family room/suite'}' at this hotel accommodates at least five (5) guests in one room, "
        f"for example a configuration such as two queen beds plus a sofa bed or an equivalent combination. "
        f"Accept clear occupancy labels like 'Sleeps 5' on the page."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=hotel_sources(hotel),
        additional_instruction="Confirm the occupancy on the hotel's or booking page. It must clearly indicate capacity for 5 guests in a single room/suite."
    )

    # 2.5 In-room amenities: both refrigerator and microwave
    amenities_leaf = evaluator.add_leaf(
        id=f"Hotel_{idx+1}_In_Room_Amenities",
        desc=f"Room must include both a refrigerator and a microwave as in-room amenities",
        parent=prop_node,
        critical=True
    )
    amenities_claim = (
        f"The specified room includes BOTH an in-room refrigerator and an in-room microwave (not only a shared or lobby microwave)."
    )
    await evaluator.verify(
        claim=amenities_claim,
        node=amenities_leaf,
        sources=hotel_sources(hotel),
        additional_instruction="Look for room amenities listing. It must clearly mention both 'refrigerator' (or 'mini-fridge') and 'microwave' in the room."
    )


async def verify_baggage_policy(evaluator: Evaluator, parent_node, baggage: BaggageInfo) -> None:
    """
    Build verification for Southwest baggage policy info.
    """
    bag_node = evaluator.add_parallel(
        id="Baggage_Policy_Information",
        desc="Provide current information about Southwest Airlines' baggage policy for flights from Phoenix: fees for 1st/2nd checked bags and tiers/fare classes with free checked bags",
        parent=parent_node,
        critical=False
    )

    # Fees leaf (critical)
    fees_leaf = evaluator.add_leaf(
        id="Baggage_Fees",
        desc="Southwest checked baggage fees for first and second bag, as stated",
        parent=bag_node,
        critical=True
    )
    fees_text = (
        f"The Southwest Airlines checked baggage fees for flights from Phoenix (PHX) are: "
        f"first checked bag: {baggage.first_bag_fee or '[unspecified]'}; "
        f"second checked bag: {baggage.second_bag_fee or '[unspecified]'}."
    )
    fees_sources = baggage.source_urls[:] if baggage.source_urls else []
    # Ensure an official fallback source is included
    if "https://www.southwest.com/help/baggage/baggage-policies" not in fees_sources:
        fees_sources.append("https://www.southwest.com/help/baggage/baggage-policies")

    await evaluator.verify(
        claim=fees_text,
        node=fees_leaf,
        sources=fees_sources,
        additional_instruction="Rely on official Southwest baggage policy page(s) if available. Verify the current fees or policy language exactly as claimed."
    )

    # Free tiers/fare classes leaf (critical)
    tiers_leaf = evaluator.add_leaf(
        id="Baggage_Free_Tiers",
        desc="Southwest tiers or fare classes receiving free checked bags",
        parent=bag_node,
        critical=True
    )
    tiers_list = baggage.free_bags_tiers_or_fares or []
    tiers_text = "The following Southwest tiers or fare classes receive free checked bags: " + (
        ", ".join(tiers_list) if tiers_list else "[unspecified]"
    ) + ". If all passengers receive two free checked bags, it is acceptable to state 'all passengers'."
    tiers_sources = baggage.source_urls[:] if baggage.source_urls else []
    if "https://www.southwest.com/help/baggage/baggage-policies" not in tiers_sources:
        tiers_sources.append("https://www.southwest.com/help/baggage/baggage-policies")

    await evaluator.verify(
        claim=tiers_text,
        node=tiers_leaf,
        sources=tiers_sources,
        additional_instruction="Confirm whether free checked bags apply to everyone or specific fare classes/status. Use official Southwest baggage page(s)."
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
    Evaluate an answer for the Florida Gulf Coast family beach vacation task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # parallel across hotels + baggage section
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

    # Parallel extractions
    hotels_extraction_task = evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction"
    )
    baggage_extraction_task = evaluator.extract(
        prompt=prompt_extract_baggage(),
        template_class=BaggageInfo,
        extraction_name="baggage_extraction"
    )

    hotels_extraction, baggage_info = await asyncio.gather(
        hotels_extraction_task, baggage_extraction_task
    )

    # Record some custom diagnostics
    evaluator.add_custom_info(
        {"allowed_booking_domains": ALLOWED_BOOKING_DOMAINS},
        info_type="settings",
        info_name="booking_domain_whitelist"
    )

    # Ensure exactly 3 hotel slots (pad with empty items if fewer)
    hotels: List[HotelItem] = (hotels_extraction.hotels or [])[:3]
    while len(hotels) < 3:
        hotels.append(HotelItem())

    # Build verification tree for 3 hotels
    family_node = evaluator.add_parallel(
        id="Family_Beach_Vacation_Planning",
        desc="Find three beachfront hotels in Florida Gulf Coast destinations accessible from Phoenix via Southwest Airlines that meet all family accommodation requirements",
        parent=root,
        critical=False
    )

    for i in range(3):
        await verify_single_hotel(evaluator, family_node, i, hotels[i])

    # Baggage policy verification
    await verify_baggage_policy(evaluator, family_node, baggage_info or BaggageInfo())

    # Return standard summary
    return evaluator.get_summary()