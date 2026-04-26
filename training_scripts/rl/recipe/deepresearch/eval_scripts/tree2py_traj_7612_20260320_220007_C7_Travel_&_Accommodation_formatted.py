import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "lake_michigan_mlk_weekend_hotels_2026"
TASK_DESCRIPTION = """
I'm planning a long weekend trip to Lake Michigan for Martin Luther King Jr. Day weekend 2026 (January 16-19) and need to identify 4 distinct hotels that cater to different needs. Please find:

Hotel 1 (Beach-focused family hotel): Must offer direct beach access or be located on a beach, have an on-site swimming pool, and have availability for check-in on Friday, January 16, 2026 (specify the standard check-in time).

Hotel 2 (Business-friendly waterfront hotel): Must be a waterfront property with lake views, offer free WiFi for guests, and have a business center or business amenities available.

Hotel 3 (Full-service resort): Must have an on-site restaurant, a fitness center or gym facility, and clearly state its cancellation policy including the deadline for free cancellation.

Hotel 4 (Pet-friendly hotel): Must be pet-friendly and accept dogs, offer parking (specify if free or paid with daily rate if applicable), and specify its distance to the nearest Lake Michigan beach.

For each hotel, provide the hotel name, location (city, state), and a reference URL supporting the information.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelBase(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    lake_michigan_relevance: Optional[str] = None  # statement/justification from the answer


class Hotel1Model(HotelBase):
    beach_access: Optional[str] = None
    pool: Optional[str] = None
    availability_jan16_2026: Optional[str] = None  # statement that availability exists
    checkin_time: Optional[str] = None


class Hotel2Model(HotelBase):
    waterfront_lake_views: Optional[str] = None
    free_wifi: Optional[str] = None
    business_amenities: Optional[str] = None  # e.g., business center, meeting rooms, workspaces


class Hotel3Model(HotelBase):
    onsite_restaurant: Optional[str] = None
    fitness_center: Optional[str] = None
    cancellation_policy_deadline: Optional[str] = None  # free-cancellation deadline statement


class Hotel4Model(HotelBase):
    dog_friendly: Optional[str] = None
    parking_details: Optional[str] = None  # include free/paid and daily rate if applicable
    distance_to_lake_michigan_beach: Optional[str] = None


class FourHotelsExtraction(BaseModel):
    hotel1: Optional[Hotel1Model] = None
    hotel2: Optional[Hotel2Model] = None
    hotel3: Optional[Hotel3Model] = None
    hotel4: Optional[Hotel4Model] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
Extract exactly four hotels from the provided answer, mapping each to its category and extracting the following fields.
The four hotels correspond to:
- hotel1: Beach-focused family hotel
- hotel2: Business-friendly waterfront hotel
- hotel3: Full-service resort
- hotel4: Pet-friendly hotel

GENERAL RULES:
- Extract only information explicitly present in the answer text; do not invent.
- For every hotel, extract any and all reference URLs that the answer cites to support its claims (include full URLs; if none are provided, return an empty list).
- Use strings for fields; keep formatting as in the answer when possible.
- If a field is not provided in the answer, return null for that field (or [] for lists).

For each hotel, extract:

Common fields for all hotels:
- name: Hotel name
- city: City
- state: State (use the postal abbreviation if the answer provides it that way; otherwise use the form present in the answer)
- reference_urls: Array of URLs cited in the answer for this hotel
- lake_michigan_relevance: A short phrase from the answer indicating the hotel is on Lake Michigan or in a Lake Michigan shoreline community

Category-specific fields:

Hotel 1 (Beach-focused family hotel):
- beach_access: Phrase indicating direct beach access or beachfront location
- pool: Phrase indicating an on-site swimming pool exists
- availability_jan16_2026: Phrase indicating availability for check-in on Friday, January 16, 2026
- checkin_time: The standard check-in time as stated in the answer

Hotel 2 (Business-friendly waterfront):
- waterfront_lake_views: Phrase indicating the property is waterfront with lake views (Lake Michigan)
- free_wifi: Phrase indicating free WiFi for guests
- business_amenities: Phrase indicating a business center or business amenities

Hotel 3 (Full-service resort):
- onsite_restaurant: Phrase indicating an on-site restaurant
- fitness_center: Phrase indicating a fitness center or gym
- cancellation_policy_deadline: The stated cancellation policy including a free-cancellation deadline (as a short quote or summary)

Hotel 4 (Pet-friendly hotel):
- dog_friendly: Phrase indicating the hotel is pet-friendly and accepts dogs
- parking_details: Phrase indicating if parking is free or paid; include daily rate if present
- distance_to_lake_michigan_beach: Phrase specifying distance to the nearest Lake Michigan beach

Return JSON with top-level keys: hotel1, hotel2, hotel3, hotel4, each an object per the above.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0 and any(_nonempty_str(u) for u in urls))


def _distinct_nonempty_names(hotels: List[Optional[HotelBase]]) -> bool:
    names = [h.name.strip().lower() for h in hotels if h and _nonempty_str(h.name)]
    return len(names) == 4 and len(set(names)) == 4


def _fmt_loc(city: Optional[str], state: Optional[str]) -> str:
    c = city.strip() if city else ""
    s = state.strip() if state else ""
    return f"{c}, {s}".strip(", ")


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_hotel_1(evaluator: Evaluator, parent_node, h: Optional[Hotel1Model]) -> None:
    node = evaluator.add_parallel(
        id="Hotel_1_Beach_Family",
        desc="Hotel 1: Beach-focused family hotel requirements.",
        parent=parent_node,
        critical=False
    )

    # Existence checks (critical gates)
    evaluator.add_custom_node(
        result=_nonempty_str(h.name) if h else False,
        id="H1_Name",
        desc="Provide the hotel name.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(_nonempty_str(h.city) and _nonempty_str(h.state)) if h else False,
        id="H1_Location",
        desc="Provide the location (city, state).",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_any_url(h.reference_urls) if h else False,
        id="H1_Reference_URL",
        desc="Provide at least one reference URL supporting the stated information for this hotel.",
        parent=node,
        critical=True
    )

    # Prepare claims
    name = h.name if h else ""
    loc = _fmt_loc(h.city if h else None, h.state if h else None)
    urls = h.reference_urls if h else []

    # Add verification leaves
    lm_rel_node = evaluator.add_leaf(
        id="H1_Lake_Michigan_Relevance",
        desc="Hotel is on/at Lake Michigan (or in a Lake Michigan shoreline community), supported by the provided reference.",
        parent=node,
        critical=True
    )
    beach_node = evaluator.add_leaf(
        id="H1_Beach_Access",
        desc="Offers direct beach access or is located on a beach (beachfront).",
        parent=node,
        critical=True
    )
    pool_node = evaluator.add_leaf(
        id="H1_Pool",
        desc="Has an on-site swimming pool.",
        parent=node,
        critical=True
    )
    avail_node = evaluator.add_leaf(
        id="H1_Availability_Jan16_2026",
        desc="Shows availability (bookable) for check-in on Friday, January 16, 2026, supported by the provided reference.",
        parent=node,
        critical=True
    )
    checkin_node = evaluator.add_leaf(
        id="H1_CheckIn_Time",
        desc="Specifies the standard check-in time.",
        parent=node,
        critical=True
    )

    # Build claims
    claim_lm = f"The property '{name}' is on Lake Michigan or in a Lake Michigan shoreline community; it is in {loc}."
    claim_beach = "This hotel offers direct beach access or is located on a beach (beachfront)."
    claim_pool = "This hotel has an on-site swimming pool."
    claim_avail = "This property shows bookable availability for check-in on Friday, January 16, 2026."
    ci_val = h.checkin_time if (h and _nonempty_str(h.checkin_time)) else "a specific standard check-in time"
    claim_checkin = f"The hotel's standard check-in time is {ci_val}."

    # Batch verify
    await evaluator.batch_verify([
        (claim_lm, urls, lm_rel_node,
         "Accept if the page indicates the hotel is on Lake Michigan or the city is on Lake Michigan's shore."),
        (claim_beach, urls, beach_node,
         "Confirm that the page states beachfront location or direct beach access (synonyms accepted)."),
        (claim_pool, urls, pool_node,
         "Verify that the hotel lists a swimming pool among its amenities (indoor or outdoor both acceptable)."),
        (claim_avail, urls, avail_node,
         "Look for booking calendar or availability text for Jan 16, 2026; if clearly shown, mark supported."),
        (claim_checkin, urls, checkin_node,
         "Find the standard check-in time on the page (accept common variants like 'Check-in from 3 PM')."),
    ])


async def verify_hotel_2(evaluator: Evaluator, parent_node, h: Optional[Hotel2Model]) -> None:
    node = evaluator.add_parallel(
        id="Hotel_2_Business_Waterfront",
        desc="Hotel 2: Business-friendly waterfront hotel requirements.",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=_nonempty_str(h.name) if h else False,
        id="H2_Name",
        desc="Provide the hotel name.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(_nonempty_str(h.city) and _nonempty_str(h.state)) if h else False,
        id="H2_Location",
        desc="Provide the location (city, state).",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_any_url(h.reference_urls) if h else False,
        id="H2_Reference_URL",
        desc="Provide at least one reference URL supporting the stated information for this hotel.",
        parent=node,
        critical=True
    )

    name = h.name if h else ""
    loc = _fmt_loc(h.city if h else None, h.state if h else None)
    urls = h.reference_urls if h else []

    lm_rel_node = evaluator.add_leaf(
        id="H2_Lake_Michigan_Relevance",
        desc="Hotel is on/at Lake Michigan (or in a Lake Michigan shoreline community), supported by the provided reference.",
        parent=node,
        critical=True
    )
    waterfront_node = evaluator.add_leaf(
        id="H2_Waterfront_Lake_Views",
        desc="Is a waterfront property with lake views (Lake Michigan views).",
        parent=node,
        critical=True
    )
    wifi_node = evaluator.add_leaf(
        id="H2_Free_WiFi",
        desc="Offers free WiFi for guests.",
        parent=node,
        critical=True
    )
    biz_node = evaluator.add_leaf(
        id="H2_Business_Amenities",
        desc="Has a business center or business amenities available.",
        parent=node,
        critical=True
    )

    claim_lm = f"The property '{name}' is on Lake Michigan or in a Lake Michigan shoreline community; it is in {loc}."
    claim_waterfront = "This hotel is waterfront on Lake Michigan and offers lake views."
    claim_wifi = "This hotel offers free WiFi for guests."
    claim_biz = "This hotel provides business amenities, such as a business center, meeting rooms, or workspaces."

    await evaluator.batch_verify([
        (claim_lm, urls, lm_rel_node,
         "Accept if the page indicates location on Lake Michigan or a shoreline city on Lake Michigan."),
        (claim_waterfront, urls, waterfront_node,
         "Look for waterfront claims and explicit 'lake view' or 'Lake Michigan view' phrasing."),
        (claim_wifi, urls, wifi_node,
         "Confirm free WiFi is listed as a complimentary amenity."),
        (claim_biz, urls, biz_node,
         "Confirm presence of a business center or business-friendly amenities (meeting rooms, co-working, etc.)."),
    ])


async def verify_hotel_3(evaluator: Evaluator, parent_node, h: Optional[Hotel3Model]) -> None:
    node = evaluator.add_parallel(
        id="Hotel_3_Full_Service_Resort",
        desc="Hotel 3: Full-service resort requirements.",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=_nonempty_str(h.name) if h else False,
        id="H3_Name",
        desc="Provide the hotel name.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(_nonempty_str(h.city) and _nonempty_str(h.state)) if h else False,
        id="H3_Location",
        desc="Provide the location (city, state).",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_any_url(h.reference_urls) if h else False,
        id="H3_Reference_URL",
        desc="Provide at least one reference URL supporting the stated information for this hotel.",
        parent=node,
        critical=True
    )

    name = h.name if h else ""
    loc = _fmt_loc(h.city if h else None, h.state if h else None)
    urls = h.reference_urls if h else []

    lm_rel_node = evaluator.add_leaf(
        id="H3_Lake_Michigan_Relevance",
        desc="Hotel is on/at Lake Michigan (or in a Lake Michigan shoreline community), supported by the provided reference.",
        parent=node,
        critical=True
    )
    rest_node = evaluator.add_leaf(
        id="H3_Onsite_Restaurant",
        desc="Has an on-site restaurant.",
        parent=node,
        critical=True
    )
    fit_node = evaluator.add_leaf(
        id="H3_Fitness_Center",
        desc="Has a fitness center or gym facility.",
        parent=node,
        critical=True
    )
    cancel_node = evaluator.add_leaf(
        id="H3_Cancellation_Policy_Deadline",
        desc="Clearly states its cancellation policy including the deadline for free cancellation.",
        parent=node,
        critical=True
    )

    claim_lm = f"The property '{name}' is on Lake Michigan or in a Lake Michigan shoreline community; it is in {loc}."
    claim_rest = "This hotel has an on-site restaurant."
    claim_fit = "This hotel has a fitness center or gym facility."
    deadline_txt = h.cancellation_policy_deadline if (h and _nonempty_str(h.cancellation_policy_deadline)) else "a free-cancellation deadline"
    claim_cancel = f"The page clearly states a cancellation policy that includes {deadline_txt}."

    await evaluator.batch_verify([
        (claim_lm, urls, lm_rel_node,
         "Accept if the page indicates the hotel is on Lake Michigan or within a shoreline city on Lake Michigan."),
        (claim_rest, urls, rest_node,
         "Verify there is an on-site restaurant listed among the amenities or dining options."),
        (claim_fit, urls, fit_node,
         "Verify that a fitness center or gym is available on-site."),
        (claim_cancel, urls, cancel_node,
         "Look for a free-cancellation window or explicit deadline (e.g., 'free cancellation until 48 hours before arrival')."),
    ])


async def verify_hotel_4(evaluator: Evaluator, parent_node, h: Optional[Hotel4Model]) -> None:
    node = evaluator.add_parallel(
        id="Hotel_4_Pet_Friendly",
        desc="Hotel 4: Pet-friendly hotel requirements.",
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=_nonempty_str(h.name) if h else False,
        id="H4_Name",
        desc="Provide the hotel name.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(_nonempty_str(h.city) and _nonempty_str(h.state)) if h else False,
        id="H4_Location",
        desc="Provide the location (city, state).",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_any_url(h.reference_urls) if h else False,
        id="H4_Reference_URL",
        desc="Provide at least one reference URL supporting the stated information for this hotel.",
        parent=node,
        critical=True
    )

    name = h.name if h else ""
    loc = _fmt_loc(h.city if h else None, h.state if h else None)
    urls = h.reference_urls if h else []

    lm_rel_node = evaluator.add_leaf(
        id="H4_Lake_Michigan_Relevance",
        desc="Hotel is on/at Lake Michigan (or in a Lake Michigan shoreline community), supported by the provided reference.",
        parent=node,
        critical=True
    )
    dog_node = evaluator.add_leaf(
        id="H4_Dog_Friendly",
        desc="Is pet-friendly and explicitly accepts dogs.",
        parent=node,
        critical=True
    )
    parking_node = evaluator.add_leaf(
        id="H4_Parking_With_Details",
        desc="Offers parking and specifies whether it is free or paid (and provides the daily rate if paid).",
        parent=node,
        critical=True
    )
    distance_node = evaluator.add_leaf(
        id="H4_Distance_To_Nearest_Lake_Michigan_Beach",
        desc="Specifies its distance to the nearest Lake Michigan beach.",
        parent=node,
        critical=True
    )

    claim_lm = f"The property '{name}' is on Lake Michigan or in a Lake Michigan shoreline community; it is in {loc}."
    claim_dog = "This hotel is pet-friendly and accepts dogs."
    pd = h.parking_details if (h and _nonempty_str(h.parking_details)) else "parking details indicating free or paid (with daily rate if paid)"
    claim_parking = f"The page provides {pd}."
    dist = h.distance_to_lake_michigan_beach if (h and _nonempty_str(h.distance_to_lake_michigan_beach)) else "a stated distance to the nearest Lake Michigan beach"
    claim_distance = f"The page specifies {dist}."

    await evaluator.batch_verify([
        (claim_lm, urls, lm_rel_node,
         "Accept if the page indicates the hotel is within a Lake Michigan shoreline city or directly along Lake Michigan."),
        (claim_dog, urls, dog_node,
         "Verify dogs are explicitly allowed. 'Pet-friendly' alone counts only if dogs are clearly included."),
        (claim_parking, urls, parking_node,
         "Verify that parking is mentioned and whether it is free or paid; if paid, a daily rate should be shown."),
        (claim_distance, urls, distance_node,
         "Verify the page states the distance to a Lake Michigan beach; accept approximate values (e.g., 0.5 mi)."),
    ])


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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Hotels evaluated independently
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

    # Extract structured hotel info
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=FourHotelsExtraction,
        extraction_name="four_hotels_extraction",
    )

    # Distinct hotels check (critical)
    distinct_node = evaluator.add_custom_node(
        result=_distinct_nonempty_names([extracted.hotel1, extracted.hotel2, extracted.hotel3, extracted.hotel4]),
        id="Distinct_Hotels",
        desc="All 4 hotels listed are distinct properties (no duplicates).",
        parent=root,
        critical=True,
    )

    # Build category subtrees (non-critical groups as per rubric)
    await verify_hotel_1(evaluator, root, extracted.hotel1)
    await verify_hotel_2(evaluator, root, extracted.hotel2)
    await verify_hotel_3(evaluator, root, extracted.hotel3)
    await verify_hotel_4(evaluator, root, extracted.hotel4)

    # Return evaluation summary
    return evaluator.get_summary()