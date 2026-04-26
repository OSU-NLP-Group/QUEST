import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "spring_break_2026_hotels"
TASK_DESCRIPTION = (
    "A family is planning a multi-city spring break trip across the United States during March 16-22, 2026 "
    "(peak spring break week). They need to book 4 different family-friendly hotels, one in each of the following cities, "
    "with each hotel meeting specific requirements:\n\n"
    "1. Orlando, Florida: Find a hotel that has an outdoor swimming pool and on-site dining options (restaurant, café, or bar).\n\n"
    "2. Tampa, Florida: Find a hotel that is located within 5 miles of ZooTampa at Lowry Park, as they plan to visit the new "
    "Florida Waters expansion with the Straz Family Manatee Rescue exhibit.\n\n"
    "3. Washington DC Metropolitan Area: Find a hotel that has direct walking access to a WMATA Metro station (within 0.5 miles), "
    "since they want to use public transportation to explore the city.\n\n"
    "4. Chicago Metropolitan Area: Find a hotel that is located within 10 miles of Chicago O'Hare International Airport (ORD), "
    "as they have a connecting flight to catch.\n\n"
    "For each of the 4 hotels, provide: (1) the hotel name, (2) a brief description of how it meets the requirements, and "
    "(3) a verifiable website URL or booking page. All hotels must be available for booking during March 16-22, 2026, and "
    "must explicitly welcome families with children."
)

DATE_RANGE_HUMAN = "March 16–22, 2026"
CHECK_IN_ISO = "2026-03-16"
CHECK_OUT_ISO = "2026-03-22"

ZOOTAMPA_NAME = "ZooTampa at Lowry Park"
ZOOTAMPA_ADDRESS = "1101 W Sligh Ave, Tampa, FL 33604"

ORD_NAME = "Chicago O'Hare International Airport (ORD)"
ORD_ADDRESS = "10000 W O'Hare Ave, Chicago, IL 60666"


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class HotelEntry(BaseModel):
    name: Optional[str] = None
    hotel_url: Optional[str] = None
    extra_urls: List[str] = Field(default_factory=list)
    description: Optional[str] = None

    # Helpful extracted snippets (all optional strings for robustness)
    location_text: Optional[str] = None
    availability_text: Optional[str] = None
    family_friendly_text: Optional[str] = None

    # Orlando-specific signals
    pool_text: Optional[str] = None           # e.g., "outdoor pool", "resort-style outdoor pool"
    dining_text: Optional[str] = None         # e.g., "on-site restaurant", "bar & grill", "café"

    # Tampa-specific
    distance_to_zootampa_miles: Optional[str] = None  # e.g., "3.1 miles", "4 mi", "≈ 5 miles"

    # DC-specific
    metro_station_name: Optional[str] = None
    metro_distance_miles: Optional[str] = None  # e.g., "0.3 miles", "0.4 mi", "500 meters"

    # Chicago-specific
    distance_to_ord_miles: Optional[str] = None  # e.g., "6.5 mi", "9.8 miles"


class HotelsExtraction(BaseModel):
    orlando: Optional[HotelEntry] = None
    tampa: Optional[HotelEntry] = None
    dc_area: Optional[HotelEntry] = None
    chicago_area: Optional[HotelEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return f"""
Extract the four hotels presented in the answer for the specified cities/areas and return a structured JSON with these keys:
- orlando
- tampa
- dc_area
- chicago_area

For each key, extract an object with the following fields (use null if missing):
- name: The hotel name exactly as written in the answer.
- hotel_url: A verifiable official hotel website or a booking page URL for the hotel.
- extra_urls: A list of any additional source URLs or maps links the answer cites for verification (e.g., Google Maps, WMATA station page).
- description: The brief explanation (as written) for how the hotel meets the requirement.
- location_text: Any location/address/city/area text provided.
- availability_text: Any mention of availability for the dates {DATE_RANGE_HUMAN} (or "available those dates", "rooms available", etc.).
- family_friendly_text: Any statement that the property is family-friendly or explicitly welcomes families/children (e.g., "family-friendly", "kids stay free").
- pool_text: (Orlando only) Any mention showing an outdoor pool, e.g., "outdoor pool", "heated outdoor pool".
- dining_text: (Orlando only) Any mention of on-site dining, e.g., "on-site restaurant", "bar & grill", "café".
- distance_to_zootampa_miles: (Tampa only) The distance to ZooTampa if the answer mentions/derives it (keep as a string, do not convert).
- metro_station_name: (DC only) The nearest WMATA Metro station name if mentioned.
- metro_distance_miles: (DC only) The distance in miles to the nearest WMATA Metro station if mentioned (string).
- distance_to_ord_miles: (Chicago only) The distance in miles to {ORD_NAME} if mentioned (string).

Rules:
- Do not infer or create URLs; only extract URLs explicitly present in the answer.
- If multiple hotels are listed for a city, pick the one the answer ultimately uses. If unclear, pick the first valid one.
- For extra_urls, include any map links or other supporting links the answer cited.
- Keep distances as raw strings (e.g., "4 mi", "3.2 miles").
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def gather_sources(entry: Optional[HotelEntry]) -> List[str]:
    urls: List[str] = []
    if entry and entry.hotel_url:
        urls.append(entry.hotel_url)
    if entry and entry.extra_urls:
        for u in entry.extra_urls:
            if u and isinstance(u, str) and u not in urls:
                urls.append(u)
    return urls


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_orlando(
    evaluator: Evaluator,
    parent,
    entry: Optional[HotelEntry],
) -> None:
    node = evaluator.add_parallel(
        id="Hotel_1_Orlando",
        desc="Identify a family-friendly hotel in Orlando, Florida with pool and dining",
        parent=parent,
        critical=False,
    )

    # URL presence (critical gate)
    url_ok = bool(entry and entry.hotel_url and entry.hotel_url.strip())
    evaluator.add_custom_node(
        result=url_ok,
        id="URL_Reference_Hotel1",
        desc="Provide verifiable website URL or booking page for the hotel",
        parent=node,
        critical=True,
    )

    sources = gather_sources(entry)

    # Location: Orlando, FL
    loc_leaf = evaluator.add_leaf(
        id="Hotel_1_Orlando_Location_Orlando",
        desc="Hotel must be located in Orlando, Florida",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This hotel's official page indicates the property is located in Orlando, Florida.",
        node=loc_leaf,
        sources=sources,
        additional_instruction="Check the address/location section. Accept 'Orlando, FL' or an Orlando neighborhood clearly within Orlando city.",
    )

    # Availability for the specified dates
    avail_leaf = evaluator.add_leaf(
        id="Hotel_1_Orlando_Date_Availability",
        desc="Hotel must be available for booking/stay during March 16-22, 2026",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The hotel shows availability for a stay from {CHECK_IN_ISO} (check-in) to {CHECK_OUT_ISO} (check-out).",
        node=avail_leaf,
        sources=sources,
        additional_instruction=(
            f"Verify via the booking engine or rates page on the provided URL(s). "
            f"Evidence could be 'rooms available', 'select room' for those dates, or visible rates for {DATE_RANGE_HUMAN}. "
            f"If the page clearly shows 'sold out' or does not support those dates, mark as not supported."
        ),
    )

    # Family-friendly
    family_leaf = evaluator.add_leaf(
        id="Hotel_1_Orlando_Family_Friendly",
        desc="Hotel must explicitly welcome families with children",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The hotel's official website explicitly welcomes families with children (family-friendly).",
        node=family_leaf,
        sources=sources,
        additional_instruction=(
            "Look for phrases like 'family-friendly', 'families', 'kids', 'children welcome', 'family rooms/suites', or similar explicit statements/policies."
        ),
    )

    # Outdoor pool
    pool_leaf = evaluator.add_leaf(
        id="Hotel_1_Orlando_Outdoor_Pool",
        desc="Hotel must have an outdoor swimming pool",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The hotel has an outdoor swimming pool.",
        node=pool_leaf,
        sources=sources,
        additional_instruction="Accept 'outdoor pool', 'heated outdoor pool', or images/facilities explicitly indicating an outdoor swimming pool (not solely indoor).",
    )

    # On-site dining
    dining_leaf = evaluator.add_leaf(
        id="Hotel_1_Orlando_Onsite_Dining",
        desc="Hotel must have on-site dining options (restaurant, café, or bar)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The hotel has on-site dining options such as a restaurant, café, lounge, or bar.",
        node=dining_leaf,
        sources=sources,
        additional_instruction="Accept on-site 'restaurant', 'bar', 'café', 'bistro', 'grill', 'lounge', or similar clearly on property (not exclusively off-site).",
    )


async def verify_tampa(
    evaluator: Evaluator,
    parent,
    entry: Optional[HotelEntry],
) -> None:
    node = evaluator.add_parallel(
        id="Hotel_2_Tampa",
        desc="Identify a family-friendly hotel in Tampa, Florida near ZooTampa",
        parent=parent,
        critical=False,
    )

    # URL presence (critical gate)
    url_ok = bool(entry and entry.hotel_url and entry.hotel_url.strip())
    evaluator.add_custom_node(
        result=url_ok,
        id="URL_Reference_Hotel2",
        desc="Provide verifiable website URL or booking page for the hotel",
        parent=node,
        critical=True,
    )

    sources = gather_sources(entry)

    # Location: Tampa, FL
    loc_leaf = evaluator.add_leaf(
        id="Hotel_2_Tampa_Location_Tampa",
        desc="Hotel must be located in Tampa, Florida",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This hotel's official page indicates the property is located in Tampa, Florida.",
        node=loc_leaf,
        sources=sources,
        additional_instruction="Check the address/location section. Accept 'Tampa, FL' or a clearly Tampa city address.",
    )

    # Availability for the specified dates
    avail_leaf = evaluator.add_leaf(
        id="Hotel_2_Tampa_Date_Availability",
        desc="Hotel must be available for booking/stay during March 16-22, 2026",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The hotel shows availability for a stay from {CHECK_IN_ISO} to {CHECK_OUT_ISO}.",
        node=avail_leaf,
        sources=sources,
        additional_instruction=(
            f"Verify via the booking engine or rates page. Look for clear availability indicators covering {DATE_RANGE_HUMAN}."
        ),
    )

    # Family-friendly
    family_leaf = evaluator.add_leaf(
        id="Hotel_2_Tampa_Family_Friendly",
        desc="Hotel must explicitly welcome families with children",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The hotel's official website explicitly welcomes families with children (family-friendly).",
        node=family_leaf,
        sources=sources,
        additional_instruction="Look for 'family-friendly' language, 'kids', 'children welcome', or family suites/policies.",
    )

    # Proximity to ZooTampa within 5 miles
    proximity_leaf = evaluator.add_leaf(
        id="Hotel_2_Tampa_Proximity_ZooTampa",
        desc="Hotel must be within 5 miles of ZooTampa at Lowry Park",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The hotel is within 5 miles of {ZOOTAMPA_NAME} located at {ZOOTAMPA_ADDRESS}.",
        node=proximity_leaf,
        sources=sources,
        additional_instruction=(
            "Use any provided hotel map, Google Maps, or distance reference on the hotel page/linked sources. "
            "If a distance is explicitly stated under 5 miles (<= 5 mi), accept. If the evidence suggests > 5 miles, reject."
        ),
    )


async def verify_dc(
    evaluator: Evaluator,
    parent,
    entry: Optional[HotelEntry],
) -> None:
    node = evaluator.add_parallel(
        id="Hotel_3_Washington_DC",
        desc="Identify a family-friendly hotel in Washington DC area with Metro access",
        parent=parent,
        critical=False,
    )

    # URL presence (critical gate)
    url_ok = bool(entry and entry.hotel_url and entry.hotel_url.strip())
    evaluator.add_custom_node(
        result=url_ok,
        id="URL_Reference_Hotel3",
        desc="Provide verifiable website URL or booking page for the hotel",
        parent=node,
        critical=True,
    )

    sources = gather_sources(entry)

    # Location: DC metro area (DC/MD/VA)
    loc_leaf = evaluator.add_leaf(
        id="Hotel_3_DC_Location_DC_Area",
        desc="Hotel must be located in Washington DC metropolitan area (DC, Maryland, or Virginia)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This hotel is within the Washington DC metropolitan area (District of Columbia, Maryland suburbs, or Virginia suburbs).",
        node=loc_leaf,
        sources=sources,
        additional_instruction=(
            "Accept addresses in DC proper or nearby MD/VA jurisdictions commonly considered DC metro (e.g., Arlington, Alexandria, Fairfax, "
            "Falls Church, Bethesda, Silver Spring, National Landing/Crystal City, Rosslyn, etc.)."
        ),
    )

    # Availability for the specified dates
    avail_leaf = evaluator.add_leaf(
        id="Hotel_3_DC_Date_Availability",
        desc="Hotel must be available for booking/stay during March 16-22, 2026",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The hotel shows availability for a stay from {CHECK_IN_ISO} to {CHECK_OUT_ISO}.",
        node=avail_leaf,
        sources=sources,
        additional_instruction=f"Verify availability across {DATE_RANGE_HUMAN} via the hotel's booking/rates page.",
    )

    # Family-friendly
    family_leaf = evaluator.add_leaf(
        id="Hotel_3_DC_Family_Friendly",
        desc="Hotel must explicitly welcome families with children",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The hotel's official website explicitly welcomes families with children (family-friendly).",
        node=family_leaf,
        sources=sources,
        additional_instruction="Look for 'family-friendly', 'kids', 'children welcome', or family rooms/suites.",
    )

    # Metro access within 0.5 miles
    metro_leaf = evaluator.add_leaf(
        id="Hotel_3_DC_Metro_Access",
        desc="Hotel must have direct walking access to a WMATA Metro station (within 0.5 miles)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The hotel has direct walking access to a WMATA Metro station within 0.5 miles (0.8 km).",
        node=metro_leaf,
        sources=sources,
        additional_instruction=(
            "Accept explicit statements like 'steps from [Station Name]' or distances <= 0.5 miles to a WMATA Metrorail station. "
            "Use linked maps or station pages if provided."
        ),
    )


async def verify_chicago(
    evaluator: Evaluator,
    parent,
    entry: Optional[HotelEntry],
) -> None:
    node = evaluator.add_parallel(
        id="Hotel_4_Chicago",
        desc="Identify a family-friendly hotel in Chicago area near O'Hare Airport",
        parent=parent,
        critical=False,
    )

    # URL presence (critical gate)
    url_ok = bool(entry and entry.hotel_url and entry.hotel_url.strip())
    evaluator.add_custom_node(
        result=url_ok,
        id="URL_Reference_Hotel4",
        desc="Provide verifiable website URL or booking page for the hotel",
        parent=node,
        critical=True,
    )

    sources = gather_sources(entry)

    # Location: Chicago metropolitan area
    loc_leaf = evaluator.add_leaf(
        id="Hotel_4_Chicago_Location_Chicago_Area",
        desc="Hotel must be located in Chicago metropolitan area",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="This hotel is located in the Chicago metropolitan area.",
        node=loc_leaf,
        sources=sources,
        additional_instruction="Accept addresses in Chicago proper or nearby suburbs that are part of the Chicago metro.",
    )

    # Availability for the specified dates
    avail_leaf = evaluator.add_leaf(
        id="Hotel_4_Chicago_Date_Availability",
        desc="Hotel must be available for booking/stay during March 16-22, 2026",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The hotel shows availability for a stay from {CHECK_IN_ISO} to {CHECK_OUT_ISO}.",
        node=avail_leaf,
        sources=sources,
        additional_instruction=f"Verify visible availability or rates for {DATE_RANGE_HUMAN} via the hotel's booking engine.",
    )

    # Family-friendly
    family_leaf = evaluator.add_leaf(
        id="Hotel_4_Chicago_Family_Friendly",
        desc="Hotel must explicitly welcome families with children",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The hotel's official website explicitly welcomes families with children (family-friendly).",
        node=family_leaf,
        sources=sources,
        additional_instruction="Look for 'family-friendly', 'kids', 'children welcome', or similar language.",
    )

    # Proximity to ORD within 10 miles
    ord_leaf = evaluator.add_leaf(
        id="Hotel_4_Chicago_Airport_Proximity",
        desc="Hotel must be within 10 miles of Chicago O'Hare International Airport (ORD)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The hotel is within 10 miles of {ORD_NAME} at {ORD_ADDRESS}.",
        node=ord_leaf,
        sources=sources,
        additional_instruction=(
            "Use any maps or distance references provided. Accept explicit distances <= 10 miles."
        ),
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
    Evaluate an agent's answer for the Spring Break 2026 multi-city hotel search task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=(
            "Find 4 family-friendly hotels in specified U.S. cities available during spring break 2026 "
            f"({DATE_RANGE_HUMAN}), each meeting specific location and amenity requirements."
        ),
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured hotel info
    extracted: HotelsExtraction = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction",
    )

    # Optionally record task/date info as custom info
    evaluator.add_custom_info(
        info={
            "required_dates": {"check_in": CHECK_IN_ISO, "check_out": CHECK_OUT_ISO, "human_range": DATE_RANGE_HUMAN},
            "special_landmarks": {"ZooTampa": ZOOTAMPA_ADDRESS, "ORD": ORD_ADDRESS},
        },
        info_type="task_context",
        info_name="task_context",
    )

    # Build four hotel verification subtrees
    await verify_orlando(evaluator, root, extracted.orlando)
    await verify_tampa(evaluator, root, extracted.tampa)
    await verify_dc(evaluator, root, extracted.dc_area)
    await verify_chicago(evaluator, root, extracted.chicago_area)

    # Return structured evaluation summary
    return evaluator.get_summary()