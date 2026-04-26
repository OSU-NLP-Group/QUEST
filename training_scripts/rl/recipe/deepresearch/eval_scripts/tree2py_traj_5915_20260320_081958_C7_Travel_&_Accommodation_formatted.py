import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "orlando_dog_hotels_mlk_2026"
TASK_DESCRIPTION = """
I am planning a family trip to Orlando, Florida for the MLK Day 2026 long weekend (Friday, January 16 through Monday, January 19, 2026 - a 3-night stay). We will be bringing our 55-pound dog and want to stay close to Disney Springs for easy access to shopping and entertainment.

Please identify exactly 4 hotels that meet ALL of the following requirements:

1. Located within 2 miles of Disney Springs in Orlando, Florida
2. Have availability for January 16-19, 2026 (3 nights)
3. Accept dogs with a weight of at least 50 pounds per dog
4. Offer on-site parking facilities (either complimentary or paid)
5. Provide breakfast service to guests (can be complimentary breakfast, continental breakfast, or on-site breakfast restaurant)
6. Have a fitness center available for guest use
7. Have a swimming pool on the property
8. Provide WiFi to guests (complimentary or paid)
9. Have a check-in time of 4:00 PM or earlier

For each hotel, provide:
- The hotel name
- A brief description confirming it meets the requirements
- A reference URL from the hotel's official website or a major booking platform (such as Booking.com, Hotels.com, Expedia, Marriott.com, Hilton.com, Disney's official website, etc.) that verifies the amenities and policies
"""

WEEKEND_CHECK_IN = "Friday, January 16, 2026"
WEEKEND_CHECK_OUT = "Monday, January 19, 2026"
NIGHTS = 3


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    availability_urls: List[str] = Field(default_factory=list)


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    Extract up to the first 4 hotels that the answer claims meet the user's requirements.
    For each hotel, extract:
    - name: the hotel's name
    - description: the brief description the answer gives to confirm it meets the requirements
    - reference_urls: all URLs (official site and/or major booking platforms) cited for this specific hotel that verify amenities/policies
    - availability_urls: any URL(s) in the answer that specifically show availability or rates for the dates Jan 16–19, 2026 for this hotel. If none are provided, return an empty list.
    
    Rules:
    - Only extract URLs explicitly present in the answer text (including markdown links).
    - Keep full URLs including protocol.
    - If more than 4 hotels are mentioned, include only the first 4.
    - If fewer than 4 hotels are mentioned, return as many as are present.
    - If a field is missing, return null (for strings) or [] (for URL lists).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
ALLOWED_REFERENCE_DOMAINS = [
    # Major OTAs / booking platforms
    "booking.com",
    "hotels.com",
    "expedia.com",
    "priceline.com",
    "agoda.com",
    "orbitz.com",
    "travelocity.com",
    "hotwire.com",
    # Major brand official sites
    "marriott.com",
    "hilton.com",
    "hyatt.com",
    "ihg.com",
    "holidayinn.com",
    "choicehotels.com",
    "wyndhamhotels.com",
    "bestwestern.com",
    "sonesta.com",
    "omnihotels.com",
    "druryhotels.com",
    "accor.com",
    "radissonhotels.com",
    "loewshotels.com",
    "fourseasons.com",
    # Disney official
    "disney.go.com",
    "disneyworld.disney.go.com",
    # Disney Springs resort area hotels site
    "disneyspringshotels.com",
]


def _normalize_domain(host: str) -> str:
    if host.startswith("www."):
        return host[4:]
    return host


def is_allowed_reference_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = _normalize_domain(parsed.netloc.lower())
        if not host:
            return False
        for allow in ALLOWED_REFERENCE_DOMAINS:
            allow_norm = allow.lower()
            if host == allow_norm or host.endswith("." + allow_norm):
                return True
        # Heuristic: brand official sites often have brand domain substring before TLD; be conservative
        return False
    except Exception:
        return False


def unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def combine_all_sources(h: HotelItem) -> List[str]:
    return unique_urls(list(h.availability_urls) + list(h.reference_urls))


# --------------------------------------------------------------------------- #
# Verification logic per hotel                                                #
# --------------------------------------------------------------------------- #
async def verify_hotel(evaluator: Evaluator, parent: VerificationNode, hotel: HotelItem, index_1b: int) -> None:
    hotel_node = evaluator.add_parallel(
        id=f"hotel_{index_1b}",
        desc=f"Hotel #{index_1b} verification: {hotel.name or 'Unnamed hotel'}",
        parent=parent,
        critical=False  # allow partial across hotels
    )

    # Required info (name + at least one ref URL)
    required_ok = bool(hotel.name and hotel.name.strip()) and len(hotel.reference_urls) > 0
    evaluator.add_custom_node(
        result=required_ok,
        id=f"hotel_{index_1b}_required_info",
        desc=f"Hotel #{index_1b} has required information (name and at least one reference URL)",
        parent=hotel_node,
        critical=True
    )

    all_urls = combine_all_sources(hotel)

    # Group: Location + Availability
    loc_dates_node = evaluator.add_parallel(
        id=f"hotel_{index_1b}_location_dates",
        desc=f"Hotel #{index_1b}: within 2 miles of Disney Springs AND availability for {WEEKEND_CHECK_IN}–{WEEKEND_CHECK_OUT} ({NIGHTS} nights)",
        parent=hotel_node,
        critical=True
    )

    # 1) Within 2 miles of Disney Springs
    n_within_2 = evaluator.add_leaf(
        id=f"hotel_{index_1b}_within_2_miles",
        desc=f"Hotel #{index_1b} is within 2 miles of Disney Springs",
        parent=loc_dates_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The hotel '{hotel.name or ''}' is located within 2 miles of Disney Springs in Orlando / Lake Buena Vista, "
            f"Florida (evidence can include an explicit distance like '0.5 mile', '1.2 miles', 'walking distance', "
            f"'Disney Springs Resort Area', 'adjacent to Disney Springs', a map clearly showing proximity, or similar)."
        ),
        node=n_within_2,
        sources=all_urls,
        additional_instruction=(
            "Pass if the page explicitly indicates distance ≤ 2 miles, walking distance, on-property/adjacent to Disney Springs, "
            "or is clearly a Disney Springs Resort Area hotel. Use reasonable judgment based on page text or maps."
        )
    )

    # 2) Availability for specific dates
    n_avail = evaluator.add_leaf(
        id=f"hotel_{index_1b}_availability",
        desc=f"Hotel #{index_1b} has availability for {WEEKEND_CHECK_IN}–{WEEKEND_CHECK_OUT} ({NIGHTS} nights)",
        parent=loc_dates_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The hotel has at least one available room for check-in {WEEKEND_CHECK_IN} and check-out {WEEKEND_CHECK_OUT} "
            f"({NIGHTS} nights)."
        ),
        node=n_avail,
        sources=all_urls if all_urls else None,
        additional_instruction=(
            "Prefer booking search result pages that explicitly include those dates and show available rates/rooms. "
            "Pass only if the page indicates availability or shows bookable rates for those dates. "
            "If the page clearly indicates 'sold out' or 'no availability', fail. If dates are not shown, fail."
        )
    )

    # Group: Pet + Parking
    pet_parking_node = evaluator.add_parallel(
        id=f"hotel_{index_1b}_pet_parking",
        desc=f"Hotel #{index_1b}: pet policy ≥ 50 lb dogs and on-site parking",
        parent=hotel_node,
        critical=True
    )

    n_pet = evaluator.add_leaf(
        id=f"hotel_{index_1b}_pet_50lb",
        desc=f"Hotel #{index_1b} accepts dogs with weight limit ≥ 50 lb per dog",
        parent=pet_parking_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The hotel accepts dogs with a weight allowance of at least 50 pounds per dog (or no weight limit)."
        ),
        node=n_pet,
        sources=all_urls,
        additional_instruction=(
            "Pass if the pet policy states a maximum weight ≥ 50 lb, or says 'no weight limit', "
            "or is clearly large-dog friendly at or above 50 lb. If weight limit < 50 lb or 'service animals only', fail."
        )
    )

    n_parking = evaluator.add_leaf(
        id=f"hotel_{index_1b}_parking",
        desc=f"Hotel #{index_1b} offers on-site parking (complimentary or paid)",
        parent=pet_parking_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel provides on-site parking for guests (complimentary or paid).",
        node=n_parking,
        sources=all_urls,
        additional_instruction="Accept paid, valet, or self-parking if it is on-site or at the property."
    )

    # Group: Dining (breakfast) + Fitness
    dining_fitness_node = evaluator.add_parallel(
        id=f"hotel_{index_1b}_dining_fitness",
        desc=f"Hotel #{index_1b}: breakfast service and fitness center",
        parent=hotel_node,
        critical=True
    )

    n_breakfast = evaluator.add_leaf(
        id=f"hotel_{index_1b}_breakfast",
        desc=f"Hotel #{index_1b} provides breakfast service (complimentary or on-site restaurant)",
        parent=dining_fitness_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The hotel provides breakfast service to guests (complimentary/continental breakfast or an on-site breakfast restaurant)."
        ),
        node=n_breakfast,
        sources=all_urls,
        additional_instruction=(
            "Pass if breakfast is complimentary OR if there is an on-site restaurant offering breakfast service. "
            "Room service breakfast also counts as on-site breakfast."
        )
    )

    n_fitness = evaluator.add_leaf(
        id=f"hotel_{index_1b}_fitness",
        desc=f"Hotel #{index_1b} has a fitness center for guest use",
        parent=dining_fitness_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel has a fitness center available to guests.",
        node=n_fitness,
        sources=all_urls,
        additional_instruction="Look for amenities lists or facilities pages mentioning 'fitness center', 'gym', or 'health club'."
    )

    # Group: Pool + WiFi
    pool_wifi_node = evaluator.add_parallel(
        id=f"hotel_{index_1b}_pool_wifi",
        desc=f"Hotel #{index_1b}: swimming pool and WiFi",
        parent=hotel_node,
        critical=True
    )

    n_pool = evaluator.add_leaf(
        id=f"hotel_{index_1b}_pool",
        desc=f"Hotel #{index_1b} has a swimming pool on property",
        parent=pool_wifi_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel has a swimming pool on the property.",
        node=n_pool,
        sources=all_urls,
        additional_instruction="Indoor or outdoor pools count. Splash pools/lazy rivers count if clearly a pool facility."
    )

    n_wifi = evaluator.add_leaf(
        id=f"hotel_{index_1b}_wifi",
        desc=f"Hotel #{index_1b} provides WiFi to guests (complimentary or paid)",
        parent=pool_wifi_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel provides WiFi (complimentary or paid) to guests.",
        node=n_wifi,
        sources=all_urls,
        additional_instruction="Amenity lists frequently mention WiFi or internet access; either complimentary or paid is acceptable."
    )

    # Group: Check-in time
    checkin_node = evaluator.add_parallel(
        id=f"hotel_{index_1b}_checkin",
        desc=f"Hotel #{index_1b}: check-in time 4:00 PM or earlier",
        parent=hotel_node,
        critical=True
    )

    n_checkin = evaluator.add_leaf(
        id=f"hotel_{index_1b}_checkin_time",
        desc=f"Hotel #{index_1b} check-in time is 4:00 PM or earlier",
        parent=checkin_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel's standard check-in time is at or before 4:00 PM (16:00).",
        node=n_checkin,
        sources=all_urls,
        additional_instruction=(
            "Pass if the page shows check-in time ≤ 4:00 PM (e.g., '3 PM', '4 PM', 'Anytime after 2 PM'). "
            "If check-in is later than 4 PM (e.g., '5 PM'), fail. Be careful not to confuse with check-out time."
        )
    )

    # Group: Reference domain validity (at least one acceptable reference)
    ref_node = evaluator.add_parallel(
        id=f"hotel_{index_1b}_reference",
        desc=f"Hotel #{index_1b}: has a valid reference URL from the official site or a major booking platform",
        parent=hotel_node,
        critical=True
    )

    # Programmatic check to avoid ambiguity and ensure determinism
    has_allowed_ref = any(is_allowed_reference_url(u) for u in hotel.reference_urls)
    evaluator.add_custom_node(
        result=has_allowed_ref,
        id=f"hotel_{index_1b}_reference_valid",
        desc=f"Hotel #{index_1b} provides at least one valid reference URL (official or major platform)",
        parent=ref_node,
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
    model: str = "o4-mini",
) -> Dict:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent hotel evaluations
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Find 4 hotels within 2 miles of Disney Springs, Orlando, Florida, that meet all specified requirements for the MLK Day 2026 weekend (January 16-19, 2026).",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # IMPORTANT: set root as non-critical to allow partial credit if fewer than 4 succeed
    root.critical = False

    # Record contextual info
    evaluator.add_custom_info(
        {
            "weekend": {
                "check_in": WEEKEND_CHECK_IN,
                "check_out": WEEKEND_CHECK_OUT,
                "nights": NIGHTS,
            },
            "location_focus": "Within 2 miles of Disney Springs (Orlando / Lake Buena Vista, FL)",
            "pet_weight_requirement_lb": 50,
            "amenities_required": [
                "on-site parking",
                "breakfast service",
                "fitness center",
                "swimming pool",
                "WiFi",
                "check-in time <= 4:00 PM",
            ],
        },
        info_type="task_context",
    )

    # Extract hotel list from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction",
    )

    # Normalize to exactly 4 slots (pad with placeholders if needed; truncate extras)
    hotels: List[HotelItem] = list(extracted.hotels or [])
    if len(hotels) > 4:
        hotels = hotels[:4]
    while len(hotels) < 4:
        hotels.append(HotelItem())

    # Build tree and run verifications
    for i in range(4):
        await verify_hotel(evaluator, root, hotels[i], i + 1)

    # Return structured evaluation summary
    return evaluator.get_summary()