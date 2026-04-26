import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "paris_eiffel_hotels_apr2026"
TASK_DESCRIPTION = (
    "I am planning a trip to Paris and need to book accommodation near the Eiffel Tower for a weekend visit in April 2026. "
    "Please find three different hotels that meet ALL of the following requirements:\n"
    "1. Walking Distance: The hotel must be within a 15-minute walk from the Eiffel Tower\n"
    "2. Price Range: The nightly rate must be under €200 per night for a standard room\n"
    "3. Advance Booking: The hotel must accept reservations made at least 60 days in advance\n"
    "4. Free WiFi: The hotel must offer complimentary wireless internet access\n\n"
    "For each of the three hotels, please provide:\n"
    "- The hotel name\n"
    "- The exact walking distance (in minutes) from the hotel to the Eiffel Tower\n"
    "- The price per night for a standard room\n"
    "- A direct link to the hotel's official website or a reputable booking platform showing the property details\n"
    "- Confirmation that free WiFi is included"
)


class HotelItem(BaseModel):
    name: Optional[str] = None
    walking_distance_minutes: Optional[str] = None
    price_per_night_eur: Optional[str] = None
    url: Optional[str] = None
    extra_urls: List[str] = Field(default_factory=list)
    wifi_included: Optional[str] = None
    advance_booking_policy: Optional[str] = None


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


def prompt_extract_hotels() -> str:
    return (
        "Extract up to three distinct hotels mentioned in the answer that are near the Eiffel Tower. "
        "For each hotel, return a JSON object with the following fields:\n"
        "1) name: The hotel name exactly as stated in the answer.\n"
        "2) walking_distance_minutes: The walking time in minutes to the Eiffel Tower if stated (string). If only distance is given, include the distance with units in this field (e.g., '1.0 km'). If not stated, set to null.\n"
        "3) price_per_night_eur: The price per night for a standard room as stated in the answer (string). Include the currency symbol or code if present (e.g., '€180', 'EUR 175'). If not stated, set to null.\n"
        "4) url: A single primary URL for the hotel's official site or a reputable booking platform property page, explicitly mentioned in the answer. Use the actual URL string; do not invent URLs. If missing, set to null.\n"
        "5) extra_urls: An array of any additional property detail URLs explicitly included in the answer (e.g., other booking-platform pages). If none, return an empty array.\n"
        "6) wifi_included: A short text snippet indicating free WiFi if mentioned (e.g., 'Free WiFi included', 'complimentary WiFi'), otherwise null.\n"
        "7) advance_booking_policy: Any text in the answer indicating booking can be made 60+ days in advance (e.g., 'book months ahead', 'accepts reservations in advance'), otherwise null.\n\n"
        "Return the hotels in an array field named 'hotels'. Extract only what is explicitly stated in the answer; do not infer.\n"
        "For any URLs missing a protocol, prepend 'http://'."
    )


def _collect_sources(h: HotelItem) -> List[str]:
    sources: List[str] = []
    if h.url and h.url.strip():
        sources.append(h.url.strip())
    for u in h.extra_urls:
        if u and u.strip():
            sources.append(u.strip())
    return sources


async def verify_single_hotel(evaluator: Evaluator, parent_node, hotel: HotelItem, index: int) -> None:
    hotel_node = evaluator.add_parallel(
        id=f"hotel_{index}",
        desc=f"Hotel #{index + 1} meeting all requirements",
        parent=parent_node,
        critical=False,
    )

    has_min_info = bool(hotel.name and hotel.name.strip()) and bool(hotel.url and hotel.url.strip())
    existence_node = evaluator.add_custom_node(
        result=has_min_info,
        id=f"hotel_{index}_info_present",
        desc=f"Hotel #{index + 1} has a name and a reference URL in the answer",
        parent=hotel_node,
        critical=True,
    )

    sources = _collect_sources(hotel)

    ref_node = evaluator.add_leaf(
        id=f"hotel_{index}_reference_url",
        desc="Valid reference URL provided from reputable source",
        parent=hotel_node,
        critical=True,
    )
    ref_claim = (
        f"The provided URL corresponds to the stated hotel '{hotel.name}' and is either the official hotel website "
        f"or a reputable booking platform/property page that shows detailed information about this hotel."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_node,
        sources=sources,
        additional_instruction=(
            "Consider reputable platforms such as Booking.com, Expedia, Hotels.com, TripAdvisor, Accor, Marriott, Hilton, Hyatt, IHG, or the official hotel site. "
            "The page should clearly show the property details and the hotel name matching or equivalent to the extracted name."
        ),
    )

    walk_node = evaluator.add_leaf(
        id=f"hotel_{index}_walking_distance",
        desc="Hotel is within 15-minute walk from Eiffel Tower",
        parent=hotel_node,
        critical=True,
    )
    walk_claim = (
        "The hotel's referenced page indicates the property is within a 15-minute walk of the Eiffel Tower (Tour Eiffel), "
        "or provides a distance consistent with that walking time (e.g., roughly 1.2 km or less)."
    )
    await evaluator.verify(
        claim=walk_claim,
        node=walk_node,
        sources=sources,
        additional_instruction=(
            "Look for text stating walking minutes or approximate distance to 'Eiffel Tower' or 'Tour Eiffel'. "
            "If only distance is given, treat about ≤1.2 km as within 15 minutes for typical walking pace. "
            "If neither walking time nor distance is provided, consider the claim not supported."
        ),
    )

    price_node = evaluator.add_leaf(
        id=f"hotel_{index}_price_range",
        desc="Nightly rate is under €200 for a standard room",
        parent=hotel_node,
        critical=True,
    )
    price_claim = (
        "The referenced page shows or indicates that the standard room nightly rate is under €200."
    )
    await evaluator.verify(
        claim=price_claim,
        node=price_node,
        sources=sources,
        additional_instruction=(
            "Focus on prices for typical/standard rooms shown on the property's page. "
            "Accept phrasing like 'from €X' if X < 200. Ignore taxes/fees for this check. "
            "If prices are shown in a different currency, consider whether the displayed base rate appears to be clearly under €200."
        ),
    )

    advance_node = evaluator.add_leaf(
        id=f"hotel_{index}_advance_booking",
        desc="Hotel accepts reservations at least 60 days in advance",
        parent=hotel_node,
        critical=True,
    )
    advance_claim = (
        "The referenced page indicates that the hotel accepts reservations at least 60 days in advance (e.g., months ahead or a booking calendar showing future availability)."
    )
    await evaluator.verify(
        claim=advance_claim,
        node=advance_node,
        sources=sources,
        additional_instruction=(
            "Look for any booking policy text that mentions advance reservations or check whether the booking interface shows future dates 60+ days ahead (e.g., several months out). "
            "If the page provides no indication of advance booking capability, consider the claim not supported."
        ),
    )

    wifi_node = evaluator.add_leaf(
        id=f"hotel_{index}_free_wifi",
        desc="Hotel offers complimentary wireless internet access",
        parent=hotel_node,
        critical=True,
    )
    wifi_claim = (
        "The referenced page clearly indicates that the hotel offers complimentary (free) WiFi to guests."
    )
    await evaluator.verify(
        claim=wifi_claim,
        node=wifi_node,
        sources=sources,
        additional_instruction=(
            "Check the amenities list or property description for 'Free WiFi', 'complimentary WiFi', 'Wi-Fi included', "
            "either in rooms or common areas. If only paid WiFi is mentioned, or WiFi is not mentioned, the claim is not supported."
        ),
    )


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
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Find three hotels near the Eiffel Tower that meet all specified requirements",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    extracted = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction",
    )

    hotels: List[HotelItem] = list(extracted.hotels[:3])
    while len(hotels) < 3:
        hotels.append(HotelItem())

    for i in range(3):
        await verify_single_hotel(evaluator, root, hotels[i], i)

    evaluator.add_custom_info(
        info={"target_month": "April 2026", "trip_type": "Weekend near Eiffel Tower"},
        info_type="planning_context",
    )

    return evaluator.get_summary()