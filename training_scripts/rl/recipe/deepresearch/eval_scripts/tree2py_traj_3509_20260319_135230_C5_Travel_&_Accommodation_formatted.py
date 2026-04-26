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
TASK_ID = "sixflags_valencia_hotel"
TASK_DESCRIPTION = """
You are planning a family trip to Six Flags Magic Mountain in Valencia, California. Identify a hotel that meets ALL of the following requirements: (1) Located within 1 mile of Six Flags Magic Mountain, (2) Part of a major hotel chain brand (such as Hilton, Marriott, IHG, Hyatt, etc.), (3) Offers an outdoor swimming pool, (4) Has a fitness center, (5) Provides complimentary breakfast included with the room rate, (6) Offers free parking, (7) Currently operational and accepting reservations. Provide the specific hotel name and supporting reference URLs that verify the hotel meets these requirements.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelExtraction(BaseModel):
    # Identity
    name: Optional[str] = None
    brand: Optional[str] = None

    # Location/address
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    distance_to_six_flags: Optional[str] = None  # e.g., "0.8 mi", "1 mile"

    # Amenities as mentioned in the answer (strings preferred for flexibility)
    outdoor_pool_mentioned: Optional[str] = None
    fitness_center_mentioned: Optional[str] = None
    complimentary_breakfast_mentioned: Optional[str] = None
    free_parking_mentioned: Optional[str] = None

    # URLs
    location_urls: List[str] = Field(default_factory=list)     # pages confirming address/proximity
    amenities_urls: List[str] = Field(default_factory=list)    # pages listing amenities
    official_url: Optional[str] = None                         # hotel's official property page
    booking_url: Optional[str] = None                          # booking/reservations page (official if possible)
    brand_url: Optional[str] = None                            # brand/property listing page


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel() -> str:
    return """
    Extract the single hotel proposed in the answer that is intended to meet ALL requirements. Return fields exactly as they appear in the answer text; do not invent information.

    Required fields:
    - name: The specific hotel name (e.g., "Hampton Inn by Hilton Los Angeles/Santa Clarita")
    - brand: The brand or chain name (e.g., "Hilton", "Hampton by Hilton", "Marriott", "Hyatt", "IHG", "Wyndham", "Choice", "Best Western", etc.)
    - address: The street address if provided
    - city: The city stated for the hotel (e.g., "Valencia" or "Santa Clarita")
    - state: The state abbreviation or full state name (e.g., "CA" or "California")
    - distance_to_six_flags: Any explicit textual distance to "Six Flags Magic Mountain" if mentioned (e.g., "0.8 mi", "1 mile")
    - outdoor_pool_mentioned: Copy the phrase that indicates the hotel has an outdoor pool if present (e.g., "outdoor pool", "heated outdoor pool")
    - fitness_center_mentioned: Copy the phrase that indicates the hotel has a fitness center or gym if present
    - complimentary_breakfast_mentioned: Copy the phrase that indicates complimentary or free breakfast is included with the room rate (e.g., "free hot breakfast", "complimentary breakfast")
    - free_parking_mentioned: Copy the phrase that indicates free parking for guests if present

    URL fields (extract actual URLs mentioned in the answer; do not infer):
    - location_urls: Array of URL(s) that substantiate the hotel's address and/or its distance/proximity to Six Flags Magic Mountain
    - amenities_urls: Array of URL(s) that substantiate the hotel's amenities (pool, gym, breakfast, parking)
    - official_url: The hotel's official property or brand page URL, if present
    - booking_url: A reservations/booking page URL that shows availability for this hotel, if present
    - brand_url: A brand/property listing URL that clearly shows the hotel's brand affiliation, if present

    Rules:
    - If any field is not present in the answer, set it to null (or empty array for URL lists).
    - For URL lists, include all relevant URLs that the answer associates with that check.
    - Do not add or infer URLs that are not explicitly provided in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _merge_urls(*candidates: Optional[List[str] | str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for c in candidates:
        if not c:
            continue
        if isinstance(c, str):
            items = [c]
        else:
            items = c
        for u in items:
            if not _non_empty(u):
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: HotelExtraction) -> None:
    # Root wrapper (non-critical, high-level description)
    root = evaluator.find_node("root")

    # Main critical node: all requirements must be satisfied
    main_node = evaluator.add_parallel(
        id="Hotel_Selection_Verification",
        desc="Verify that the identified hotel meets all specified requirements for location and amenities",
        parent=root,
        critical=True,
    )

    # ----------------------- Identity & Location (Critical) -------------------
    id_loc_node = evaluator.add_parallel(
        id="Hotel_Identity_and_Location",
        desc="Verify hotel identification and location requirements",
        parent=main_node,
        critical=True,
    )

    # Reference URL presence for location/proximity (gatekeeper)
    ref_loc_presence = evaluator.add_custom_node(
        result=bool(extracted.location_urls and len(extracted.location_urls) > 0),
        id="Reference_URL_Location",
        desc="A valid reference URL is provided that confirms the hotel's location and proximity to Six Flags Magic Mountain",
        parent=id_loc_node,
        critical=True,
    )

    # Hotel name and brand container (Critical)
    name_brand_container = evaluator.add_parallel(
        id="Hotel_Name_and_Brand",
        desc="The hotel name is provided and it is affiliated with a recognized major hotel chain brand",
        parent=id_loc_node,
        critical=True,
    )

    # 1) Name provided (existence)
    evaluator.add_custom_node(
        result=_non_empty(extracted.name),
        id="Hotel_Name_Provided",
        desc="Hotel name is provided in the answer",
        parent=name_brand_container,
        critical=True,
    )

    # 2) Brand affiliation with major chain (verify via URLs if available)
    brand_leaf = evaluator.add_leaf(
        id="Brand_Major_Chain",
        desc="The hotel is affiliated with a recognized major hotel chain brand",
        parent=name_brand_container,
        critical=True,
    )
    brand_claim = (
        f"The hotel named '{extracted.name or 'the hotel'}' is part of the '{extracted.brand}' brand, "
        "which is a recognized major hotel chain brand."
        if _non_empty(extracted.brand)
        else "This hotel is part of a recognized major hotel chain brand."
    )
    brand_sources = _merge_urls(extracted.brand_url, extracted.official_url, extracted.location_urls)
    await evaluator.verify(
        claim=brand_claim,
        node=brand_leaf,
        sources=brand_sources if brand_sources else None,
        additional_instruction=(
            "Consider brands like Hilton (Hampton, Homewood, DoubleTree, etc.), Marriott (Courtyard, Fairfield, Residence Inn, etc.), "
            "IHG (Holiday Inn, Holiday Inn Express, Staybridge Suites, etc.), Hyatt (Hyatt Place, Hyatt House, etc.), "
            "Wyndham (La Quinta, Ramada, etc.), Choice (Comfort, Quality Inn, Sleep Inn, etc.), Best Western, Accor, Radisson as major hotel chains. "
            "The page should clearly show brand affiliation (brand name/logo or loyalty program)."
        ),
    )

    # Location city check: Valencia, California (Critical)
    loc_city_leaf = evaluator.add_leaf(
        id="Location_Valencia_CA",
        desc="The hotel is located in Valencia, California",
        parent=id_loc_node,
        critical=True,
    )
    loc_sources = _merge_urls(extracted.location_urls, extracted.official_url)
    city_name = "Valencia, California"
    loc_city_claim = (
        f"The hotel '{extracted.name or 'the hotel'}' is located in Valencia, California."
    )
    await evaluator.verify(
        claim=loc_city_claim,
        node=loc_city_leaf,
        sources=loc_sources if loc_sources else None,
        additional_instruction=(
            "Pass if the address explicitly shows 'Valencia, CA'. "
            "Also accept cases where the address is 'Santa Clarita, CA' but the page explicitly associates the property with the Valencia area "
            "(e.g., 'Valencia/Santa Clarita', 'in Valencia neighborhood of Santa Clarita')."
        ),
    )

    # Proximity to Six Flags (<= 1 mile) (Critical)
    prox_leaf = evaluator.add_leaf(
        id="Proximity_to_Six_Flags",
        desc="The hotel is within 1 mile of Six Flags Magic Mountain",
        parent=id_loc_node,
        critical=True,
    )
    prox_sources = _merge_urls(extracted.location_urls, extracted.official_url)
    prox_claim = (
        "The hotel is within 1.0 mile of Six Flags Magic Mountain in Valencia, California."
    )
    await evaluator.verify(
        claim=prox_claim,
        node=prox_leaf,
        sources=prox_sources if prox_sources else None,
        additional_instruction=(
            "Pass if the page states a distance to Six Flags Magic Mountain of 1.0 mile or less "
            "(e.g., 0.2 mi, 0.8 miles, ~1 mile). If the page only implies close proximity without a numeric distance, "
            "it must be unambiguous (e.g., 'across the street from Six Flags Magic Mountain')."
        ),
    )

    # Operational status / accepting reservations (Critical)
    op_leaf = evaluator.add_leaf(
        id="Currently_Operational",
        desc="The hotel is currently operational and accepting reservations",
        parent=id_loc_node,
        critical=True,
    )
    op_sources = _merge_urls(extracted.booking_url, extracted.official_url, extracted.location_urls)
    op_claim = (
        "The hotel is currently open and accepting reservations (bookable rooms are available for upcoming dates)."
    )
    await evaluator.verify(
        claim=op_claim,
        node=op_leaf,
        sources=op_sources if op_sources else None,
        additional_instruction=(
            "Look for signals like 'Book Now', date pickers, availability calendars, or rate selection on an official/brand booking page. "
            "Fail if the page indicates 'temporarily closed', 'permanently closed', or otherwise not accepting reservations."
        ),
    )

    # ----------------------- Amenities (Critical) -----------------------------
    amen_node = evaluator.add_parallel(
        id="Required_Amenities_Verification",
        desc="Verify all required amenities are available at the hotel",
        parent=main_node,
        critical=True,
    )

    # Reference URL presence for amenities (gatekeeper)
    evaluator.add_custom_node(
        result=bool(extracted.amenities_urls and len(extracted.amenities_urls) > 0),
        id="Reference_URL_Amenities",
        desc="A valid reference URL is provided that confirms the hotel's amenities",
        parent=amen_node,
        critical=True,
    )

    # Outdoor swimming pool (Critical)
    pool_leaf = evaluator.add_leaf(
        id="Outdoor_Swimming_Pool",
        desc="The hotel has an outdoor swimming pool available for guests",
        parent=amen_node,
        critical=True,
    )
    pool_sources = _merge_urls(extracted.amenities_urls, extracted.official_url)
    pool_claim = "The hotel offers an outdoor swimming pool for guests."
    await evaluator.verify(
        claim=pool_claim,
        node=pool_leaf,
        sources=pool_sources if pool_sources else None,
        additional_instruction=(
            "The page must indicate 'outdoor pool' (or synonyms like 'heated outdoor pool'). "
            "If it only mentions an indoor pool, do not pass."
        ),
    )

    # Fitness center (Critical)
    gym_leaf = evaluator.add_leaf(
        id="Fitness_Center",
        desc="The hotel has a fitness center available for guests",
        parent=amen_node,
        critical=True,
    )
    gym_sources = _merge_urls(extracted.amenities_urls, extracted.official_url)
    gym_claim = "The hotel has a fitness center or gym available for guests."
    await evaluator.verify(
        claim=gym_claim,
        node=gym_leaf,
        sources=gym_sources if gym_sources else None,
        additional_instruction="Look for 'fitness center', 'gym', or similar terminology clearly listed as an amenity.",
    )

    # Complimentary breakfast included (Critical)
    breakfast_leaf = evaluator.add_leaf(
        id="Complimentary_Breakfast",
        desc="The hotel provides complimentary breakfast included with the room rate",
        parent=amen_node,
        critical=True,
    )
    breakfast_sources = _merge_urls(extracted.amenities_urls, extracted.official_url)
    breakfast_claim = "The hotel includes complimentary (free) breakfast with the room rate."
    await evaluator.verify(
        claim=breakfast_claim,
        node=breakfast_leaf,
        sources=breakfast_sources if breakfast_sources else None,
        additional_instruction=(
            "Pass only if the page states 'free breakfast', 'complimentary breakfast', or equivalent. "
            "If it merely says 'breakfast available' or references a paid restaurant without indicating it is free/included, do not pass."
        ),
    )

    # Free parking (Critical)
    parking_leaf = evaluator.add_leaf(
        id="Free_Parking",
        desc="The hotel offers free parking for guests",
        parent=amen_node,
        critical=True,
    )
    parking_sources = _merge_urls(extracted.amenities_urls, extracted.official_url)
    parking_claim = "The hotel offers free parking for guests."
    await evaluator.verify(
        claim=parking_claim,
        node=parking_leaf,
        sources=parking_sources if parking_sources else None,
        additional_instruction="Look specifically for 'free parking' or 'complimentary parking'. If parking is paid, do not pass.",
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
    # Initialize evaluator with a neutral root
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured hotel information from the answer
    extracted: HotelExtraction = await evaluator.extract(
        prompt=prompt_extract_hotel(),
        template_class=HotelExtraction,
        extraction_name="hotel_extraction",
    )

    # Build and run verification
    await build_verification_tree(evaluator, extracted)

    # Return evaluator summary
    return evaluator.get_summary()