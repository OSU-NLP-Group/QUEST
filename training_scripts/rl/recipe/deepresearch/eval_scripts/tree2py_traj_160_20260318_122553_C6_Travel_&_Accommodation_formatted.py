import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cr_sjo_hotels"
TASK_DESCRIPTION = (
    "Find three hotels located near Juan Santamaría International Airport (SJO) in San José, Costa Rica, "
    "that meet all of the specified requirements (location/transportation, room/pricing, amenities/policies) for a family of four. "
    "Provide for each hotel: name, a URL showing location/distance/shuttle details, a URL showing current room rates/configurations, "
    "and a URL listing amenities and cancellation policy."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    # Identifiers
    hotel_name: Optional[str] = None

    # URLs (one each, if available)
    location_url: Optional[str] = None
    room_url: Optional[str] = None
    amenities_url: Optional[str] = None

    # Location & Transportation (as text from answer; free-form strings to maximize compatibility)
    distance_to_sjo_text: Optional[str] = None
    travel_time_text: Optional[str] = None
    free_shuttle_text: Optional[str] = None
    shuttle_hours_text: Optional[str] = None

    # Room & Pricing (free-form)
    family_accommodation_text: Optional[str] = None
    price_text_usd: Optional[str] = None

    # Amenities & Policies (free-form)
    wifi_text: Optional[str] = None
    breakfast_text: Optional[str] = None
    pool_text: Optional[str] = None
    cancellation_policy_text: Optional[str] = None


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
Extract from the answer up to three hotels that the answer proposes near Juan Santamaría International Airport (SJO) in Costa Rica.

For each hotel, return the following fields exactly as presented in the answer text (do not invent values):
- hotel_name: The hotel's name.
- location_url: A single URL (full URL with protocol) from the hotel's official website or a major booking platform (Booking.com, Hotels.com, Expedia, TripAdvisor) that shows the hotel's location, mentions distance or travel time to SJO, and ideally mentions airport shuttle details. If multiple are provided, pick the one that best shows location/distance/shuttle. If none present, return null.
- room_url: A single URL that shows current room rates and the room configurations/occupancy. If multiple, pick the best one; if none, return null.
- amenities_url: A single URL that lists amenities and cancellation policy details. If multiple, pick the best one; if none, return null.

Also extract the following descriptive strings, as-is from the answer when present:
- distance_to_sjo_text: Any distance to SJO or proximity wording (e.g., "3 km from SJO", "6 miles", etc.).
- travel_time_text: Any travel time statements to SJO by car (e.g., "10 minutes to SJO").
- free_shuttle_text: Text indicating free/complimentary airport shuttle is offered.
- shuttle_hours_text: Shuttle hours text if mentioned (e.g., "5 AM–10 PM", "24/7", etc.).
- family_accommodation_text: Text indicating a room accommodates at least 4 (2 adults + 2 children), occupancy "sleeps 4", "2 queen beds", etc.
- price_text_usd: The stated nightly rate (ideally in USD) for a room for a family of four if the answer gives a number (e.g., "$145/night"). If not provided, return null. Keep text as-is.
- wifi_text: Any text indicating complimentary/free WiFi throughout the property.
- breakfast_text: Any text indicating complimentary breakfast included with the rate.
- pool_text: Any text indicating the presence of an on-site swimming pool.
- cancellation_policy_text: Any text indicating free cancellation rules/timing (e.g., "free cancellation up to 24 hours before check-in").

Rules:
- Only extract values explicitly present in the answer. If a field is missing in the answer for a hotel, set it to null.
- Return at most three hotels in an array named 'hotels', in the same order as in the answer.
- Do not include any URLs not explicitly provided in the answer.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(*urls: Optional[str]) -> List[str]:
    """Return a list of non-empty URLs with simple de-duplication."""
    seen = set()
    res: List[str] = []
    for u in urls:
        if u and isinstance(u, str):
            u2 = u.strip()
            if u2 and u2 not in seen:
                seen.add(u2)
                res.append(u2)
    return res


async def _verify_with_urls_or_fail(
    evaluator: Evaluator,
    *,
    claim: str,
    node_id: str,
    desc: str,
    parent,
    critical: bool,
    sources: List[str],
    additional_instruction: str = "None",
) -> None:
    """
    Create a leaf node for a by-URL verification when sources available; otherwise directly fail a custom node.
    """
    if sources:
        node = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=parent,
            critical=critical
        )
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=sources,
            additional_instruction=additional_instruction
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=desc + " (no supporting URL provided in the answer)",
            parent=parent,
            critical=critical
        )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_hotel(
    evaluator: Evaluator,
    parent,
    hotel: HotelItem,
    index: int
) -> None:
    """
    Build and verify the subtree for one hotel.
    Note on criticality: To allow preferred-but-not-mandatory checks under the same parent,
    we set parent containers as non-critical and mark essential children as critical.
    This avoids violating the framework rule that 'critical parents cannot have non-critical children'.
    """
    i = index + 1
    hotel_node = evaluator.add_parallel(
        id=f"Hotel_{i}",
        desc=f"Hotel #{i} meeting all requirements",
        parent=parent,
        critical=False
    )

    # Location & Transportation
    loc_node = evaluator.add_parallel(
        id=f"Hotel_{i}_Location_and_Transportation",
        desc="Hotel location, distance, and airport shuttle services",
        parent=hotel_node,
        critical=False  # Mixed criticality children
    )

    # Distance Check
    dist_node = evaluator.add_parallel(
        id=f"Hotel_{i}_Distance_Check",
        desc="Verify hotel distance and travel time to airport",
        parent=loc_node,
        critical=False
    )

    loc_sources = _safe_sources(hotel.location_url)
    loc_or_amen_sources = _safe_sources(hotel.location_url, hotel.amenities_url)

    # Within 10km (critical)
    await _verify_with_urls_or_fail(
        evaluator,
        claim=(
            "This page shows that the hotel is located within 10 kilometers (approximately 6.2 miles) "
            "of Juan Santamaría International Airport (SJO). Accept distance in miles if ≤ 6.3."
        ),
        node_id=f"Hotel_{i}_Within_10km",
        desc="Hotel is located within 10 kilometers of Juan Santamaría International Airport",
        parent=dist_node,
        critical=True,
        sources=loc_sources,
        additional_instruction=(
            "Look for distance statements or maps referencing 'Juan Santamaría International Airport', "
            "'SJO', or 'San Jose International'. If the page states distance in miles, treat 6.2–6.3 miles as within 10 km."
        )
    )

    # Travel time ≤ 20 min by car (critical)
    await _verify_with_urls_or_fail(
        evaluator,
        claim=(
            "This page states that travel time by car from the hotel to SJO is 20 minutes or less."
        ),
        node_id=f"Hotel_{i}_Travel_Time_20min",
        desc="Stated travel time to airport is 20 minutes or less by car",
        parent=dist_node,
        critical=True,
        sources=loc_sources,
        additional_instruction=(
            "Accept phrasing such as 'X minutes to the airport', 'short 10-min drive', etc.; "
            "Must be ≤ 20 minutes."
        )
    )

    # Free shuttle service (critical)
    await _verify_with_urls_or_fail(
        evaluator,
        claim="This page states that the hotel offers a complimentary (free) airport shuttle service.",
        node_id=f"Hotel_{i}_Free_Shuttle_Service",
        desc="Hotel offers complimentary airport shuttle service",
        parent=loc_node,
        critical=True,
        sources=loc_or_amen_sources,
        additional_instruction=(
            "Look for 'complimentary', 'free airport shuttle', or similar. "
            "Paid shuttles or third-party arrangements are not acceptable."
        )
    )

    # Shuttle hours (non-critical, preferred)
    await _verify_with_urls_or_fail(
        evaluator,
        claim=(
            "The airport shuttle operates at least from 5:00 AM to 10:00 PM daily, or for a longer window (e.g., 24/7)."
        ),
        node_id=f"Hotel_{i}_Shuttle_Hours",
        desc="Shuttle operates at least from 5 AM to 10 PM daily",
        parent=loc_node,
        critical=False,
        sources=loc_or_amen_sources,
        additional_instruction=(
            "Accept if service window includes 05:00–22:00 daily or is 24/7. "
            "If the page lacks clear hours, this should fail."
        )
    )

    # Location Reference URL (critical)
    await _verify_with_urls_or_fail(
        evaluator,
        claim=(
            "This page provides the hotel's location (address or map), mentions distance or car travel time to SJO, "
            "and includes airport shuttle details."
        ),
        node_id=f"Hotel_{i}_Location_Reference_URL",
        desc=("Provide URL from hotel's official website or major booking platform showing location, "
              "distance to SJO, and shuttle service details"),
        parent=loc_node,
        critical=True,
        sources=loc_sources,
        additional_instruction=(
            "Major platforms allowed: Booking.com, Hotels.com, Expedia, TripAdvisor, or the hotel's official website. "
            "Verify that the page itself includes the location and mentions distance/travel time and shuttle."
        )
    )

    # Room & Pricing
    rp_node = evaluator.add_parallel(
        id=f"Hotel_{i}_Room_and_Pricing",
        desc="Room configuration and nightly rate requirements",
        parent=hotel_node,
        critical=False  # Mixed criticality children
    )

    room_sources = _safe_sources(hotel.room_url)

    # Family accommodation (critical)
    await _verify_with_urls_or_fail(
        evaluator,
        claim=(
            "This page shows at least one room option that accommodates 4 people (e.g., 2 adults and 2 children) "
            "or explicitly 'sleeps 4' or occupancy ≥ 4."
        ),
        node_id=f"Hotel_{i}_Family_Accommodation",
        desc="Hotel offers rooms that can accommodate at least 4 people (2 adults and 2 children)",
        parent=rp_node,
        critical=True,
        sources=room_sources,
        additional_instruction=(
            "Look for occupancy labels like 'Sleeps 4', '4 guests', or specific family room descriptions. "
            "Bed configurations implying 4 guests are acceptable if occupancy is clear."
        )
    )

    # Price verification (critical)
    price_hint = f" Price noted in the answer: {hotel.price_text_usd}" if hotel.price_text_usd else ""
    await _verify_with_urls_or_fail(
        evaluator,
        claim=(
            "This page shows at least one room option that accommodates 4 people priced at $150 USD per night or less "
            "(before taxes and fees)."
        ),
        node_id=f"Hotel_{i}_Price_Verification",
        desc="Standard family room rate is $150 USD or less per night",
        parent=rp_node,
        critical=True,
        sources=room_sources,
        additional_instruction=(
            "Confirm that a room suitable for 4 guests is available at $150/night USD or less before taxes/fees."
            " If prices vary, any qualifying option counts." + price_hint
        )
    )

    # Room reference URL (critical)
    await _verify_with_urls_or_fail(
        evaluator,
        claim=(
            "This page shows current room rates and room configurations/occupancy for the hotel."
        ),
        node_id=f"Hotel_{i}_Room_Reference_URL",
        desc="Provide URL showing current room rates and configurations from hotel website or major booking platform",
        parent=rp_node,
        critical=True,
        sources=room_sources,
        additional_instruction=(
            "Accept if the page contains rate listings and occupancy or bed-type information. "
            "Major booking platforms or the official site are acceptable."
        )
    )

    # Amenities & Policies
    ap_node = evaluator.add_parallel(
        id=f"Hotel_{i}_Amenities_and_Policies",
        desc="Hotel amenities and cancellation policy",
        parent=hotel_node,
        critical=False  # Mixed criticality children
    )

    # Essential amenities (container)
    ess_node = evaluator.add_parallel(
        id=f"Hotel_{i}_Essential_Amenities",
        desc="Hotel provides essential amenities for family comfort",
        parent=ap_node,
        critical=False
    )

    amen_sources = _safe_sources(hotel.amenities_url)
    amen_or_room_sources = _safe_sources(hotel.amenities_url, hotel.room_url)

    # Free WiFi (critical)
    await _verify_with_urls_or_fail(
        evaluator,
        claim="This page indicates complimentary/free WiFi is available throughout the property.",
        node_id=f"Hotel_{i}_Free_WiFi",
        desc="Complimentary WiFi is available throughout the property",
        parent=ess_node,
        critical=True,
        sources=amen_or_room_sources,
        additional_instruction=(
            "Accept synonyms: 'free Wi-Fi', 'complimentary wireless internet', etc. "
            "Property-wide or in-room WiFi acceptable if free is indicated."
        )
    )

    # Complimentary breakfast (non-critical, preferred)
    await _verify_with_urls_or_fail(
        evaluator,
        claim="This page indicates complimentary/included breakfast with the room rate.",
        node_id=f"Hotel_{i}_Breakfast_Service",
        desc="Complimentary breakfast is included with the room rate",
        parent=ess_node,
        critical=False,
        sources=amen_or_room_sources,
        additional_instruction=(
            "Look for 'Breakfast included' or equivalent. Do not accept if breakfast is available only for an extra fee."
        )
    )

    # Pool facility (non-critical, preferred)
    await _verify_with_urls_or_fail(
        evaluator,
        claim="This page indicates the hotel has an on-site swimming pool.",
        node_id=f"Hotel_{i}_Pool_Facility",
        desc="Hotel has an on-site swimming pool",
        parent=ess_node,
        critical=False,
        sources=amen_or_room_sources,
        additional_instruction=(
            "Accept mentions of 'pool', 'swimming pool', 'outdoor pool', etc., on property."
        )
    )

    # Cancellation policy (critical)
    await _verify_with_urls_or_fail(
        evaluator,
        claim=(
            "This page indicates free cancellation is allowed at least 24 hours before the check-in date "
            "(e.g., 'free cancellation up to 1 day before arrival')."
        ),
        node_id=f"Hotel_{i}_Cancellation_Policy",
        desc="Hotel allows free cancellation at least 24 hours before check-in date",
        parent=ap_node,
        critical=True,
        sources=amen_or_room_sources,
        additional_instruction=(
            "Accept policy phrasing like 'free cancellation until 1 day before arrival', '24 hours prior', or later cutoffs; "
            "Booking-platform 'Free cancellation until [date/time]' relative to a typical check-in date is acceptable if it "
            "corresponds to ≥ 24 hours before check-in."
        )
    )

    # Amenities reference URL (critical)
    await _verify_with_urls_or_fail(
        evaluator,
        claim="This page lists the hotel's amenities and also describes the cancellation policy details.",
        node_id=f"Hotel_{i}_Amenities_Reference_URL",
        desc="Provide URL showing hotel amenities list and cancellation policy details",
        parent=ap_node,
        critical=True,
        sources=amen_sources,
        additional_instruction=(
            "The page should include an amenities section and contain cancellation policy text."
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
    Evaluate an answer for the Costa Rica SJO airport hotels task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel across hotels
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

    # Extract up to 3 hotels from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction",
    )

    hotels: List[HotelItem] = list(extracted.hotels) if extracted and extracted.hotels else []
    # Keep only first 3; pad if fewer
    hotels = hotels[:3]
    while len(hotels) < 3:
        hotels.append(HotelItem())

    # Create a top-level container (non-critical to allow partial credit)
    cr_container = evaluator.add_parallel(
        id="Costa_Rica_Hotel_Search",
        desc="Find three hotels near SJO in Costa Rica that meet all specified requirements for a family stay",
        parent=root,
        critical=False
    )

    # Verify each hotel subtree
    for idx in range(3):
        await verify_hotel(evaluator, cr_container, hotels[idx], idx)

    # Optionally include a compact summary of extracted hotel names and URLs
    try:
        evaluator.add_custom_info(
            {
                "hotels_extracted": [
                    {
                        "name": h.hotel_name,
                        "location_url": h.location_url,
                        "room_url": h.room_url,
                        "amenities_url": h.amenities_url,
                        "price_text_usd": h.price_text_usd,
                    }
                    for h in hotels
                ]
            },
            info_type="extracted_overview",
            info_name="extracted_hotels_overview"
        )
    except Exception:
        pass

    return evaluator.get_summary()