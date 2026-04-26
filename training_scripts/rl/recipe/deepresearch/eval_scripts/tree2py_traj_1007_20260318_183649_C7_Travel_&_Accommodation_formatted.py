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
TASK_ID = "luxury_chain_hotel_mexico_city"
TASK_DESCRIPTION = """
Identify a luxury hotel in Mexico City that belongs to one of the following major international hotel chains: Marriott International (including brands such as St. Regis, Ritz-Carlton, JW Marriott), Hilton (including brands such as Waldorf Astoria, Conrad), Hyatt, IHG (InterContinental Hotels Group), or Accor (including brands such as Sofitel). For the hotel you identify, provide the following comprehensive information: (1) The complete official name of the hotel, (2) The parent hotel chain and specific luxury brand, (3) The complete street address including neighborhood/colonia, (4) Confirmation of luxury or 5-star classification, (5) The name of the on-site spa facility, (6) Description of the swimming pool (indoor/outdoor, location), (7) Details about the fitness center availability, (8) Names of at least one on-site restaurant, (9) Room service hours, (10) Confirmation of concierge services, (11) Description of meeting or event facilities (capacity or number of rooms), (12) Airport transportation options available (shuttle, car service, or arranged transport), (13) Verification that the hotel is located within Mexico City proper, (14) A booking link from the hotel's official website or a major booking platform (Booking.com, Expedia, Hotels.com) showing 2026 availability. All information must be verifiable through publicly accessible sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FieldWithSources(BaseModel):
    text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ChainBrand(BaseModel):
    chain: Optional[str] = None
    brand: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RestaurantItem(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BookingLink(BaseModel):
    booking_url: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class HotelInfoExtraction(BaseModel):
    hotel_name: Optional[FieldWithSources] = None
    chain_brand: Optional[ChainBrand] = None
    complete_address: Optional[FieldWithSources] = None
    luxury_classification: Optional[FieldWithSources] = None
    spa_facility: Optional[FieldWithSources] = None
    pool_description: Optional[FieldWithSources] = None
    fitness_center: Optional[FieldWithSources] = None
    restaurant_list: List[RestaurantItem] = Field(default_factory=list)
    room_service_hours: Optional[FieldWithSources] = None
    concierge_services: Optional[FieldWithSources] = None
    meeting_facilities: Optional[FieldWithSources] = None
    airport_transportation: Optional[FieldWithSources] = None
    location_verification: Optional[FieldWithSources] = None
    booking: Optional[BookingLink] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_info() -> str:
    return """
    Extract structured details for a single luxury hotel in Mexico City (CDMX) that belongs to ONE of these chains: Marriott International, Hilton, Hyatt, IHG, or Accor. Extract ONLY what is explicitly provided in the answer.

    For each field below, extract the value and all URLs cited in the answer that support that field (as an array of URLs). Do not invent any URLs. If the answer does not provide a value or a URL for a field, set it to null (or [] for arrays).

    Fields to extract:
    - hotel_name:
        - text: the complete official name of the hotel
        - sources: URLs cited that support the official name
    - chain_brand:
        - chain: parent company name (e.g., Marriott International, Hilton, Hyatt, IHG, Accor)
        - brand: specific luxury brand (e.g., St. Regis, Ritz-Carlton, JW Marriott, Waldorf Astoria, Conrad, Park Hyatt, InterContinental, Sofitel)
        - sources: URLs cited that support chain/brand
    - complete_address:
        - text: full street address including neighborhood/colonia if mentioned
        - sources: URLs cited that show the address
    - luxury_classification:
        - text: evidence text such as "5-star", "luxury", "Forbes rating", etc.
        - sources: URLs cited that support luxury/5-star classification (official site or major platforms)
    - spa_facility:
        - text: name of the on-site spa or a clear statement that a spa exists (include the name if given)
        - sources: URLs cited that mention the spa
    - pool_description:
        - text: description of the swimming pool (indoor/outdoor, rooftop/ground-level, etc.)
        - sources: URLs cited that mention the pool
    - fitness_center:
        - text: description confirming a fitness center/gym and any details (24h, equipment, etc.)
        - sources: URLs cited that mention the fitness center
    - restaurant_list: an array where each item has:
        - name: the name of an on-site restaurant (extract at least one if available)
        - sources: URLs cited that mention the restaurant
    - room_service_hours:
        - text: statement like "24-hour room service" or specific hours
        - sources: URLs cited that show room service and hours
    - concierge_services:
        - text: confirmation of concierge services (include any detail if present)
        - sources: URLs cited that mention concierge
    - meeting_facilities:
        - text: description of meeting/event facilities (capacity, number of rooms, or general description)
        - sources: URLs cited that mention meeting/event spaces
    - airport_transportation:
        - text: available airport transportation options (shuttle, car service, arranged transfer, etc.)
        - sources: URLs cited that mention airport transport
    - location_verification:
        - text: explicit confirmation that the hotel is in Mexico City/CDMX (not Estado de México)
        - sources: URLs cited that confirm Mexico City proper location
    - booking:
        - booking_url: a booking link from the official hotel/brand site or a major platform (Booking.com, Expedia, Hotels.com) that shows 2026 availability
        - sources: URLs cited for booking (include the booking_url here as well)

    Rules:
    - Extract only what the answer explicitly provides.
    - For all 'sources' fields, return only valid URLs that appear in the answer (plain URLs or from markdown links).
    - If the answer lists more than one hotel, concentrate on the first valid one that fits the requirement.
    - If a field has no value or no URLs in the answer, set the field to null (or [] for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            # Be conservative: only accept full URLs
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def collect_all_sources(ex: HotelInfoExtraction) -> List[str]:
    urls: List[str] = []
    if not ex:
        return []
    fields_with_sources = [
        ex.hotel_name, ex.complete_address, ex.luxury_classification, ex.spa_facility,
        ex.pool_description, ex.fitness_center, ex.room_service_hours, ex.concierge_services,
        ex.meeting_facilities, ex.airport_transportation, ex.location_verification
    ]
    for fld in fields_with_sources:
        if fld and fld.sources:
            urls.extend(fld.sources)

    if ex.chain_brand and ex.chain_brand.sources:
        urls.extend(ex.chain_brand.sources)

    if ex.restaurant_list:
        for r in ex.restaurant_list:
            if r and r.sources:
                urls.extend(r.sources)

    if ex.booking:
        if ex.booking.sources:
            urls.extend(ex.booking.sources)
        if ex.booking.booking_url:
            urls.append(ex.booking.booking_url)

    return _dedup_urls(urls)


def sources_or_fallback(primary: Optional[List[str]], fallback: List[str]) -> List[str]:
    p = primary or []
    p = _dedup_urls(p)
    if p:
        return p
    return fallback


def restaurant_names_list(ex: HotelInfoExtraction) -> List[str]:
    names: List[str] = []
    for r in ex.restaurant_list or []:
        if r and r.name:
            n = r.name.strip()
            if n:
                names.append(n)
    return names


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_hotel_info(evaluator: Evaluator, root_node, ex: HotelInfoExtraction) -> None:
    # Top-level critical parallel node (as per rubric)
    main_node = evaluator.add_parallel(
        id="luxury_chain_hotel_mexico_city",
        desc="Identify a luxury hotel in Mexico City from a major international chain and provide comprehensive information",
        parent=root_node,
        critical=True
    )

    # Prepare a global fallback pool of sources gathered from the answer
    global_sources = collect_all_sources(ex)

    # 1) Hotel name
    hotel_name_text = ex.hotel_name.text if (ex and ex.hotel_name and ex.hotel_name.text) else ""
    hotel_name_node = evaluator.add_leaf(
        id="hotel_name",
        desc="Provide the complete official name of the hotel",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel's official complete name is '{hotel_name_text}'.",
        node=hotel_name_node,
        sources=sources_or_fallback(ex.hotel_name.sources if ex and ex.hotel_name else None, global_sources),
        additional_instruction="Verify that the provided pages explicitly show the hotel's official name. Allow minor punctuation/casing variations."
    )

    # 2) Chain and brand
    chain = ex.chain_brand.chain if (ex and ex.chain_brand and ex.chain_brand.chain) else ""
    brand = ex.chain_brand.brand if (ex and ex.chain_brand and ex.chain_brand.brand) else ""
    chain_brand_node = evaluator.add_leaf(
        id="hotel_chain_and_brand",
        desc="The hotel must belong to Marriott International, Hilton, Hyatt, IHG, or Accor, and specify the luxury brand (e.g., St. Regis, Ritz-Carlton, Waldorf Astoria, Conrad, Park Hyatt, InterContinental, Sofitel)",
        parent=main_node,
        critical=True
    )
    chain_brand_claim = f"The hotel belongs to '{chain}' and is branded as '{brand}'."
    await evaluator.verify(
        claim=chain_brand_claim,
        node=chain_brand_node,
        sources=sources_or_fallback(ex.chain_brand.sources if ex and ex.chain_brand else None, global_sources),
        additional_instruction=(
            "Confirm that the hotel is part of one of the ALLOWED parent chains: Marriott International, Hilton, Hyatt, IHG, or Accor. "
            "Also verify the specific luxury brand (examples: St. Regis, Ritz-Carlton, JW Marriott, Waldorf Astoria, Conrad, Park Hyatt, InterContinental, Sofitel). "
            "Use the provided sources to confirm both the chain and the brand (brand pages or the hotel's official page are acceptable)."
        )
    )

    # 3) Complete address
    addr_text = ex.complete_address.text if (ex and ex.complete_address and ex.complete_address.text) else ""
    address_node = evaluator.add_leaf(
        id="complete_address",
        desc="Provide the complete street address including street name, number, neighborhood/colonia",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel's address is '{addr_text}' and corresponds to the property in Mexico City.",
        node=address_node,
        sources=sources_or_fallback(ex.complete_address.sources if ex and ex.complete_address else None, global_sources),
        additional_instruction="Verify that the page shows a full street address for the hotel, ideally including colonia/neighborhood if present. Allow small formatting variants."
    )

    # 4) Luxury classification
    lux_text = ex.luxury_classification.text if (ex and ex.luxury_classification and ex.luxury_classification.text) else ""
    luxury_node = evaluator.add_leaf(
        id="luxury_classification",
        desc="The hotel must be verified as luxury or 5-star rated through official hotel website or major booking platforms (Booking.com, Expedia, Hotels.com, Forbes Travel Guide, etc.)",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel is a luxury or 5-star property.",
        node=luxury_node,
        sources=sources_or_fallback(ex.luxury_classification.sources if ex and ex.luxury_classification else None, global_sources),
        additional_instruction=(
            "Accept clear evidence such as '5-star', 'luxury hotel', Forbes Travel Guide rating, or membership in a well-known luxury brand (e.g., St. Regis, Ritz-Carlton, Waldorf Astoria, Park Hyatt, InterContinental, Sofitel) as sufficient proof."
        )
    )

    # 5) Spa facility name
    spa_text = ex.spa_facility.text if (ex and ex.spa_facility and ex.spa_facility.text) else ""
    spa_node = evaluator.add_leaf(
        id="spa_facility_name",
        desc="The hotel must have an on-site spa facility, and provide the spa name or description",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel has an on-site spa facility{(' named ' + spa_text) if spa_text else ''}.",
        node=spa_node,
        sources=sources_or_fallback(ex.spa_facility.sources if ex and ex.spa_facility else None, global_sources),
        additional_instruction="Look for terms like 'spa', 'wellness', 'spa by', etc., on the provided pages."
    )

    # 6) Pool description
    pool_text = ex.pool_description.text if (ex and ex.pool_description and ex.pool_description.text) else ""
    pool_node = evaluator.add_leaf(
        id="pool_description",
        desc="The hotel must have a swimming pool, and specify whether it is indoor or outdoor and its location (e.g., rooftop, ground level)",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel has a swimming pool. Details: {pool_text}".strip(),
        node=pool_node,
        sources=sources_or_fallback(ex.pool_description.sources if ex and ex.pool_description else None, global_sources),
        additional_instruction="Accept synonyms like 'pool', 'piscina', or 'alberca'. If location (rooftop/indoor/outdoor) is mentioned, confirm it."
    )

    # 7) Fitness center details
    fit_text = ex.fitness_center.text if (ex and ex.fitness_center and ex.fitness_center.text) else ""
    fitness_node = evaluator.add_leaf(
        id="fitness_center_details",
        desc="The hotel must have a fitness center or gym, and provide details about its availability and features",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel has a fitness center or gym. Details: {fit_text}".strip(),
        node=fitness_node,
        sources=sources_or_fallback(ex.fitness_center.sources if ex and ex.fitness_center else None, global_sources),
        additional_instruction="Look for 'fitness center', 'gym', 'gimnasio', and any hours/equipment details."
    )

    # 8) Restaurant names (at least one)
    rest_names = restaurant_names_list(ex)
    restaurant_node = evaluator.add_leaf(
        id="restaurant_names",
        desc="The hotel must have at least one on-site restaurant, and provide the name(s) of the restaurant(s)",
        parent=main_node,
        critical=True
    )
    rest_sources: List[str] = []
    for r in ex.restaurant_list or []:
        rest_sources.extend(r.sources or [])
    rest_sources = _dedup_urls(rest_sources)
    await evaluator.verify(
        claim=f"At least one of the following is an on-site restaurant at the hotel: {', '.join(rest_names)}.",
        node=restaurant_node,
        sources=sources_or_fallback(rest_sources, global_sources),
        additional_instruction="Confirm that at least one listed restaurant name appears as an on-property venue on the provided pages. Bars/cafés count if they are on-site."
    )

    # 9) Room service hours
    rs_text = ex.room_service_hours.text if (ex and ex.room_service_hours and ex.room_service_hours.text) else ""
    room_service_node = evaluator.add_leaf(
        id="room_service_hours",
        desc="The hotel must offer room service, and specify the service hours (e.g., 24-hour, limited hours)",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel offers room service; hours: {rs_text}".strip(),
        node=room_service_node,
        sources=sources_or_fallback(ex.room_service_hours.sources if ex and ex.room_service_hours else None, global_sources),
        additional_instruction="Look for 'room service', 'in-room dining', or 'servicio a la habitación', including indications like '24-hour' or time ranges."
    )

    # 10) Concierge services
    concierge_text = ex.concierge_services.text if (ex and ex.concierge_services and ex.concierge_services.text) else ""
    concierge_node = evaluator.add_leaf(
        id="concierge_services",
        desc="The hotel must provide concierge services",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel provides concierge services.",
        node=concierge_node,
        sources=sources_or_fallback(ex.concierge_services.sources if ex and ex.concierge_services else None, global_sources),
        additional_instruction="Terms like 'concierge', 'Les Clefs d'Or', 'butler service', or front-desk concierge descriptions are acceptable evidence."
    )

    # 11) Meeting/event facilities
    meeting_text = ex.meeting_facilities.text if (ex and ex.meeting_facilities and ex.meeting_facilities.text) else ""
    meeting_node = evaluator.add_leaf(
        id="meeting_facilities",
        desc="The hotel must have meeting rooms or event facilities, and provide information about capacity, number of rooms, or general description",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel offers meeting or event facilities. Details: {meeting_text}".strip(),
        node=meeting_node,
        sources=sources_or_fallback(ex.meeting_facilities.sources if ex and ex.meeting_facilities else None, global_sources),
        additional_instruction="Look for 'meeting rooms', 'event spaces', 'ballrooms', 'capacity', 'square meters/feet', or number of rooms."
    )

    # 12) Airport transportation
    airport_text = ex.airport_transportation.text if (ex and ex.airport_transportation and ex.airport_transportation.text) else ""
    airport_node = evaluator.add_leaf(
        id="airport_transportation",
        desc="The hotel must offer airport transportation options, and specify the type (shuttle service, car service, or arranged transportation)",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel offers airport transportation options (e.g., shuttle, car service, arranged transfers). Details: {airport_text}".strip(),
        node=airport_node,
        sources=sources_or_fallback(ex.airport_transportation.sources if ex and ex.airport_transportation else None, global_sources),
        additional_instruction="Accept mentions of 'airport transfer', 'airport shuttle', 'private car service', or 'arranged transportation'."
    )

    # 13) Location within Mexico City proper
    loc_text = ex.location_verification.text if (ex and ex.location_verification and ex.location_verification.text) else ""
    location_node = evaluator.add_leaf(
        id="location_verification",
        desc="Verify the hotel is located within Mexico City proper (not in suburbs or surrounding municipalities)",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel is located within Mexico City (CDMX) proper, not in Estado de México or surrounding municipalities.",
        node=location_node,
        sources=sources_or_fallback(ex.location_verification.sources if ex and ex.location_verification else None, sources_or_fallback(ex.complete_address.sources if ex and ex.complete_address else None, global_sources)),
        additional_instruction="Accept 'Mexico City', 'Ciudad de México', 'CDMX', or boroughs like 'Miguel Hidalgo', 'Cuauhtémoc', 'Polanco', 'Roma', 'Condesa' as evidence of being within Mexico City proper. Reject Naucalpan, Tlalnepantla, Nezahualcóyotl, etc."
    )

    # 14) Booking link (shows 2026 availability)
    booking_url = ex.booking.booking_url if (ex and ex.booking and ex.booking.booking_url) else None
    booking_node = evaluator.add_leaf(
        id="booking_link",
        desc="Provide a verified booking link from the hotel's official website or a major booking platform (Booking.com, Expedia, Hotels.com) that shows the hotel can be booked for 2026 stays",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="This is a booking page for the hotel on its official site or a major platform, and it shows dates/availability for the year 2026.",
        node=booking_node,
        sources=booking_url if booking_url else sources_or_fallback(ex.booking.sources if ex and ex.booking else None, global_sources),
        additional_instruction=(
            "Accept official/brand booking engines (e.g., marriott.com, hyatt.com, hilton.com, ihg.com, accor.com/sofitel), or major OTAs (booking.com, expedia.com, hotels.com). "
            "The page should indicate 2026 availability explicitly (e.g., '2026' in the calendar, date selectors, or URL query parameters reflecting 2026). "
            "Fail if the link is unrelated or does not clearly show 2026 booking."
        )
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
    Evaluate an answer for the luxury chain hotel in Mexico City task.
    """
    # Initialize evaluator with a parallel root (as rubric root is parallel)
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

    # Extract hotel information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotel_info(),
        template_class=HotelInfoExtraction,
        extraction_name="hotel_info_extraction",
    )

    # Build verification tree and run checks
    await verify_hotel_info(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()