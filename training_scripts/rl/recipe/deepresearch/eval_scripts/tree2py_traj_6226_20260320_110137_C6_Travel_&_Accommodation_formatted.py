import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "dollywood_pf_hotels_4_2026"
TASK_DESCRIPTION = """
I am planning a family trip to Dollywood in Pigeon Forge, Tennessee, in spring 2026. I need to find exactly 4 hotels in Pigeon Forge that meet all of the following requirements:

1. Location: The hotel must be located in Pigeon Forge, Tennessee, within 5 miles of Dollywood theme park.

2. Free Parking: The hotel must offer free on-site parking to guests (no daily parking fees).

3. Free Breakfast: The hotel must include complimentary breakfast for guests as part of the room rate.

4. Indoor Pool: The hotel must have an indoor pool facility available for guest use.

5. Check-in Time: The hotel must offer standard check-in at or before 3:00 PM.

For each of the 4 hotels, please provide:
- The hotel name
- A brief description confirming it meets all five requirements
- The hotel's official website URL or a link to its listing on a major booking platform (such as Expedia, Hotels.com, Booking.com, or the hotel's own website)
""".strip()


# --------------------------- Data Models -------------------------------------
class HotelItem(BaseModel):
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    location_text: Optional[str] = None
    distance_to_dollywood: Optional[str] = None
    free_parking: Optional[str] = None
    free_breakfast: Optional[str] = None
    indoor_pool: Optional[str] = None
    checkin_time: Optional[str] = None
    description: Optional[str] = None


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


# ------------------------ Extraction Prompt ----------------------------------
def prompt_extract_hotels() -> str:
    return """
    Extract up to 8 hotels mentioned in the answer, preserving their order of appearance.
    For each hotel, return the following fields:
    - name: The hotel's name as written in the answer.
    - reference_urls: A list of all explicit URLs (http/https) in the answer that correspond to the hotel's official site or major booking platforms (e.g., expedia.com, hotels.com, booking.com, hilton.com, marriott.com, ihg.com, choicehotels.com, wyndhamhotels.com, hyatt.com, bestwestern.com). Extract only actual URLs that appear in the answer (including within markdown links).
    - location_text: Any location/city text stated in the answer (e.g., "Pigeon Forge, TN").
    - distance_to_dollywood: Any distance to Dollywood explicitly stated in the answer (e.g., "2.3 miles").
    - free_parking: The exact phrase from the answer indicating parking is free, if present; otherwise null.
    - free_breakfast: The exact phrase from the answer indicating breakfast is complimentary/included, if present; otherwise null.
    - indoor_pool: The exact phrase from the answer indicating there is an indoor pool, if present; otherwise null.
    - checkin_time: Any stated check-in time in the answer (e.g., "3:00 PM", "2 PM"), if present; otherwise null.
    - description: The brief confirming description provided in the answer for this hotel, if any; otherwise null.

    Rules:
    - Do not infer or add information not explicitly present in the answer.
    - For reference_urls, return every valid URL included in the answer for that hotel (official site or major booking platforms). If none are present, return an empty list.
    - Keep strings exactly as they appear in the answer where possible.
    """


# ------------------------ Verification Helpers -------------------------------
def _first_n_hotels(extracted: HotelsExtraction, n: int = 4) -> List[HotelItem]:
    hotels = extracted.hotels[:n]
    # Pad with empty placeholders if fewer than n
    while len(hotels) < n:
        hotels.append(HotelItem())
    return hotels


async def verify_one_hotel(evaluator: Evaluator, parent_node, hotel: HotelItem, index_zero_based: int) -> None:
    i = index_zero_based + 1
    hotel_node = evaluator.add_parallel(
        id=f"Hotel_{i}",
        desc=f"{['First','Second','Third','Fourth','Fifth','Sixth','Seventh','Eighth'][index_zero_based] if index_zero_based < 8 else f'Hotel #{i}'} qualifying hotel identified with complete information",
        parent=parent_node,
        critical=False
    )

    # Reference URL existence (critical gate)
    urls = [u for u in (hotel.reference_urls or []) if isinstance(u, str) and u.strip()]
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"Hotel_{i}_Reference_URL",
        desc=f"Official website or verified booking platform URL for Hotel {i} provided",
        parent=hotel_node,
        critical=True
    )

    # Requirements group (critical) - all children under must be critical
    req_node = evaluator.add_parallel(
        id=f"Hotel_{i}_Requirements_Met",
        desc=f"Hotel {i} meets all specified criteria",
        parent=hotel_node,
        critical=True
    )

    # Create all five leaf nodes
    loc_node = evaluator.add_leaf(
        id=f"Hotel_{i}_Location",
        desc=f"Hotel {i} is located in Pigeon Forge, TN, and is within 5 miles of Dollywood",
        parent=req_node,
        critical=True
    )
    parking_node = evaluator.add_leaf(
        id=f"Hotel_{i}_Free_Parking",
        desc=f"Hotel {i} offers free on-site parking to guests with no daily parking fee",
        parent=req_node,
        critical=True
    )
    breakfast_node = evaluator.add_leaf(
        id=f"Hotel_{i}_Free_Breakfast",
        desc=f"Hotel {i} includes complimentary breakfast for guests",
        parent=req_node,
        critical=True
    )
    pool_node = evaluator.add_leaf(
        id=f"Hotel_{i}_Indoor_Pool",
        desc=f"Hotel {i} has an indoor pool facility available to guests",
        parent=req_node,
        critical=True
    )
    checkin_node = evaluator.add_leaf(
        id=f"Hotel_{i}_Check_In_Policy",
        desc=f"Hotel {i} offers check-in at or before 3:00 PM",
        parent=req_node,
        critical=True
    )

    hotel_name = hotel.name or f"Hotel {i}"

    claims_and_sources: List[tuple[str, List[str], Any, Optional[str]]] = []

    # Location + distance within 5 miles
    loc_claim = (
        f"The property named '{hotel_name}' is located in Pigeon Forge, Tennessee, "
        f"and is within 5 miles (8 km) of the Dollywood theme park in Pigeon Forge."
    )
    loc_ai = (
        "Use the provided webpage(s) to confirm two things: "
        "(1) the city is Pigeon Forge, TN (accept address lines or page text that clearly show Pigeon Forge, TN), and "
        "(2) the hotel is within 5 miles (8 km) of Dollywood. "
        "For (2), accept explicit distance to Dollywood or a map/screenshot on the page that clearly indicates a distance ≤ 5 miles. "
        "If distance is not explicitly stated but the page shows a distance to Dollywood (e.g., '2.3 mi to Dollywood') or an attractions section confirming proximity under 5 miles, consider it supported. "
        "If the page is irrelevant or does not support either condition, mark as not supported."
    )
    claims_and_sources.append((loc_claim, urls, loc_node, loc_ai))

    # Free parking
    parking_claim = (
        f"'{hotel_name}' offers free on-site self-parking for guests and does not charge a daily parking fee."
    )
    parking_ai = (
        "Look for 'free parking', 'complimentary self-parking', or equivalent language. "
        "If the page states parking is available only for a fee or does not clearly indicate it's complimentary, mark as not supported. "
        "Valet fees do not disqualify if self-parking is free."
    )
    claims_and_sources.append((parking_claim, urls, parking_node, parking_ai))

    # Free breakfast
    breakfast_claim = (
        f"'{hotel_name}' includes complimentary breakfast for guests as part of the room rate."
    )
    breakfast_ai = (
        "Look for 'free hot breakfast', 'complimentary breakfast', 'breakfast included', or similar language. "
        "If breakfast is 'available for a fee' or only via paid packages, not included for standard stays, mark as not supported."
    )
    claims_and_sources.append((breakfast_claim, urls, breakfast_node, breakfast_ai))

    # Indoor pool
    pool_claim = (
        f"'{hotel_name}' has an indoor pool available for guest use."
    )
    pool_ai = (
        "Confirm the amenity is an indoor pool specifically (accept 'indoor heated pool' or 'indoor/outdoor pool'). "
        "If the page only shows an outdoor pool or does not mention an indoor pool, mark as not supported."
    )
    claims_and_sources.append((pool_claim, urls, pool_node, pool_ai))

    # Check-in time at or before 3:00 PM
    checkin_claim = (
        f"'{hotel_name}' has a standard check-in time at or before 3:00 PM local time."
    )
    checkin_ai = (
        "Verify the stated check-in policy on the page. "
        "Accept 'check-in 3:00 PM' or any time earlier (e.g., 2:00 PM). "
        "If the page says 'check-in 4:00 PM' (or later), mark as not supported. "
        "Phrases like 'check-in after 3 PM' are acceptable."
    )
    claims_and_sources.append((checkin_claim, urls, checkin_node, checkin_ai))

    # Run verifications (auto-preconditions will skip if reference URL failed)
    await evaluator.batch_verify(claims_and_sources)


# ------------------------ Main Evaluation Entry ------------------------------
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
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extraction
    extracted_hotels = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction",
    )

    # Keep exactly 4 for verification (padding with placeholders if needed)
    hotels = _first_n_hotels(extracted_hotels, 4)

    # Optional: record simple meta info
    evaluator.add_custom_info(
        {
            "total_hotels_in_answer": len(extracted_hotels.hotels),
            "used_for_verification": min(4, len(extracted_hotels.hotels)),
        },
        info_type="extraction_stats",
        info_name="extraction_statistics",
    )

    # Build tree per hotel
    for idx in range(4):
        await verify_one_hotel(evaluator, root, hotels[idx], idx)

    return evaluator.get_summary()