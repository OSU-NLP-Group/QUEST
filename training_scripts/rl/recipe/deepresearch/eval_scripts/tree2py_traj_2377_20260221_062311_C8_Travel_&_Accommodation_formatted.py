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
TASK_ID = "orlando_universal_hotels_mlk_2026"
TASK_DESCRIPTION = (
    "Find 4 hotels in the Orlando, Florida area that meet the following criteria for a family vacation during "
    "Martin Luther King Jr. Day weekend 2026 (January 17-19, 2026):\n\n"
    "Each hotel must satisfy ALL of the following requirements:\n"
    "1. Be located within walking distance (1 mile or less) of Universal Orlando theme parks (Universal Studios Florida or Universal Islands of Adventure)\n"
    "2. Have confirmed availability for the dates January 17-19, 2026\n"
    "3. Feature a swimming pool (indoor or outdoor)\n"
    "4. Offer complimentary breakfast or breakfast included with the room rate\n"
    "5. Provide family-friendly room configurations (such as suites with multiple bedrooms, connecting rooms, or standard rooms that can accommodate 4 or more guests)\n"
    "6. Offer complimentary shuttle service to Universal Orlando theme parks or other area attractions\n"
    "7. Be located in Orlando, Florida\n\n"
    "For each hotel, provide:\n"
    "- The hotel name\n"
    "- Confirmation of its location in Orlando, Florida\n"
    "- Verification that it meets each of the 6 specific criteria listed above\n"
    "- A reference URL to the hotel's official website or a reputable booking site confirming the information"
)

MLK_START_DATE = "January 17, 2026"
MLK_END_DATE = "January 19, 2026"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    """Basic information for one hotel, extracted from the agent's answer."""
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class HotelsExtraction(BaseModel):
    """Extraction result: a list of hotels provided in the answer."""
    hotels: List[HotelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    Extract up to 6 candidate hotels mentioned in the answer. For each hotel, return:
    1) name: The hotel's name as stated in the answer.
    2) reference_urls: One or more URLs explicitly cited in the answer that correspond to the hotel's official website or a reputable booking site page (e.g., Marriott, Hilton, Hyatt, IHG, Booking.com, Expedia, Hotels.com). These should be pages that plausibly confirm the hotel's features and policies. Extract the actual URLs (from plain text or markdown links). If a URL is missing a protocol, prepend http://.

    Important:
    - Extract only URLs explicitly present in the answer. Do not invent or infer any URLs.
    - Prefer the hotel's official site if available; otherwise include reputable booking sites.
    - If the answer mentions multiple URLs for a hotel, include all of them.
    - If a hotel lacks any URL in the answer, return an empty list for reference_urls.

    Return a JSON object with a 'hotels' array, each element having 'name' and 'reference_urls'.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _valid_urls(urls: List[str]) -> List[str]:
    """Filter URLs to those that look valid with http/https protocol."""
    return [u for u in urls if isinstance(u, str) and (u.strip().startswith("http://") or u.strip().startswith("https://"))]


def _display_hotel_name(name: Optional[str]) -> str:
    return name.strip() if isinstance(name, str) and name.strip() else "the referenced hotel"


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_hotel(
    evaluator: Evaluator,
    parent_node,
    hotel: HotelItem,
    idx: int,
) -> None:
    """
    Build the verification subtree and run checks for a single hotel.
    """
    # Create hotel main node (non-critical to allow partial credit across hotels)
    hotel_node = evaluator.add_parallel(
        id=f"hotel_{idx+1}",
        desc=[
            "First hotel meeting all specified criteria",
            "Second hotel meeting all specified criteria",
            "Third hotel meeting all specified criteria",
            "Fourth hotel meeting all specified criteria",
        ][idx],
        parent=parent_node,
        critical=False,
    )

    # Prepare sources and reference existence gate
    sources_list = _valid_urls(hotel.reference_urls)
    has_reference = len(sources_list) > 0

    # Reference presence node (critical gate)
    reference_node = evaluator.add_custom_node(
        result=has_reference,
        id=f"hotel_{idx+1}_reference",
        desc=f"Valid reference URL provided for the hotel",
        parent=hotel_node,
        critical=True,
    )

    # Leaf: Name and Orlando location (critical)
    name_loc_leaf = evaluator.add_leaf(
        id=f"hotel_{idx+1}_name_and_location",
        desc=f"Hotel name and confirmation it is located in Orlando, Florida area",
        parent=hotel_node,
        critical=True,
    )
    name_text = _display_hotel_name(hotel.name)
    claim_name_loc = (
        f"The referenced page is for the hotel named '{name_text}', and it clearly indicates the hotel is located in Orlando, Florida."
    )
    await evaluator.verify(
        claim=claim_name_loc,
        node=name_loc_leaf,
        sources=sources_list,
        additional_instruction=(
            "Verify both the hotel's identity (name/brand) and that the location is Orlando, Florida. "
            "Minor formatting or naming variations are acceptable. "
            "Location references like 'Orlando, FL' or specific Orlando ZIP codes (e.g., 32819) count as Orlando. "
            "If the page is not about this hotel or does not state Orlando, fail."
        ),
        extra_prerequisites=[reference_node],
    )

    # Leaf: Proximity to Universal (<= 1 mile walking distance) (critical)
    proximity_leaf = evaluator.add_leaf(
        id=f"hotel_{idx+1}_universal_proximity",
        desc=f"Hotel is within walking distance (1 mile or less) of Universal Orlando theme parks",
        parent=hotel_node,
        critical=True,
    )
    claim_proximity = (
        "The hotel is within 1 mile walking distance of Universal Orlando theme parks (Universal Studios Florida or Universal Islands of Adventure)."
    )
    await evaluator.verify(
        claim=claim_proximity,
        node=proximity_leaf,
        sources=sources_list,
        additional_instruction=(
            "Look for explicit walking-distance language (e.g., 'walking distance to Universal') or a distance "
            "≤ 1 mile to 'Universal Orlando Resort', 'Universal Studios Florida', 'Islands of Adventure', or 'CityWalk'. "
            "Distances like 0.8 mi, 0.9 mi, 1.0 mi count as pass. If no distance evidence or the distance exceeds 1 mile, fail."
        ),
        extra_prerequisites=[reference_node],
    )

    # Leaf: Availability for MLK weekend (Jan 17–19, 2026) (critical)
    availability_leaf = evaluator.add_leaf(
        id=f"hotel_{idx+1}_mlk_availability",
        desc=f"Hotel has availability for January 17-19, 2026",
        parent=hotel_node,
        critical=True,
    )
    claim_availability = (
        f"The hotel shows available rooms for a stay from {MLK_START_DATE} to {MLK_END_DATE}."
    )
    await evaluator.verify(
        claim=claim_availability,
        node=availability_leaf,
        sources=sources_list,
        additional_instruction=(
            "Check the booking page or listing for explicit evidence of availability on January 17–19, 2026 (2 nights). "
            "This may appear as calendar/date selection showing rates or 'rooms available' for those dates. "
            "If dates are not present or availability is unclear/unspecified, fail."
        ),
        extra_prerequisites=[reference_node],
    )

    # Leaf: Swimming pool (critical)
    pool_leaf = evaluator.add_leaf(
        id=f"hotel_{idx+1}_pool",
        desc=f"Hotel has a swimming pool (indoor or outdoor)",
        parent=hotel_node,
        critical=True,
    )
    claim_pool = "The hotel has at least one swimming pool (indoor or outdoor)."
    await evaluator.verify(
        claim=claim_pool,
        node=pool_leaf,
        sources=sources_list,
        additional_instruction=(
            "Confirm mentions of pool amenities: 'outdoor pool', 'indoor pool', 'swimming pool'. "
            "Images or amenity lists count as evidence. If the page does not indicate any pool, fail."
        ),
        extra_prerequisites=[reference_node],
    )

    # Leaf: Complimentary/included breakfast (critical)
    breakfast_leaf = evaluator.add_leaf(
        id=f"hotel_{idx+1}_breakfast",
        desc=f"Hotel offers complimentary breakfast or breakfast included with stay",
        parent=hotel_node,
        critical=True,
    )
    claim_breakfast = "The hotel offers complimentary breakfast or breakfast included in the room rate."
    await evaluator.verify(
        claim=claim_breakfast,
        node=breakfast_leaf,
        sources=sources_list,
        additional_instruction=(
            "Look for terms like 'free breakfast', 'complimentary breakfast', or 'breakfast included'. "
            "If breakfast is paid separately or not clearly included/complimentary, fail."
        ),
        extra_prerequisites=[reference_node],
    )

    # Leaf: Family-friendly room configurations (critical)
    family_leaf = evaluator.add_leaf(
        id=f"hotel_{idx+1}_family_rooms",
        desc=f"Hotel offers family-friendly room configurations (suites, connecting rooms, or rooms accommodating 4+ guests)",
        parent=hotel_node,
        critical=True,
    )
    claim_family = (
        "The hotel offers family-friendly room configurations such as suites with multiple bedrooms, connecting rooms, "
        "or standard rooms that can accommodate four or more guests."
    )
    await evaluator.verify(
        claim=claim_family,
        node=family_leaf,
        sources=sources_list,
        additional_instruction=(
            "Accept evidence like 'sleeps 4', 'two queen beds', 'family suites', 'connecting rooms available', or explicit "
            "occupancy for 4+ guests. If occupancy appears limited to 2–3 or no family options are indicated, fail."
        ),
        extra_prerequisites=[reference_node],
    )

    # Leaf: Complimentary shuttle to Universal or attractions (critical)
    shuttle_leaf = evaluator.add_leaf(
        id=f"hotel_{idx+1}_shuttle",
        desc=f"Hotel provides complimentary shuttle service to Universal Orlando or theme parks",
        parent=hotel_node,
        critical=True,
    )
    claim_shuttle = "The hotel provides complimentary shuttle service to Universal Orlando theme parks or nearby attractions."
    await evaluator.verify(
        claim=claim_shuttle,
        node=shuttle_leaf,
        sources=sources_list,
        additional_instruction=(
            "Look for 'free shuttle', 'complimentary shuttle', or explicit shuttle to Universal Orlando/area theme parks. "
            "If shuttle exists but appears to be paid or not to Universal/attractions, fail."
        ),
        extra_prerequisites=[reference_node],
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
    Evaluate an answer for the Orlando Universal hotels during MLK weekend 2026 task.
    """
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

    # Extract hotels from the answer
    extracted_hotels = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction",
    )

    # Normalize: take first 4 hotels, pad if fewer
    hotels_list: List[HotelItem] = list(extracted_hotels.hotels[:4])
    while len(hotels_list) < 4:
        hotels_list.append(HotelItem())

    # Add custom info for date range
    evaluator.add_custom_info(
        info={"mlk_weekend": {"start": MLK_START_DATE, "end": MLK_END_DATE}},
        info_type="date_range",
        info_name="mlk_weekend_2026",
    )

    # Build verification subtrees for each hotel
    for i, hotel in enumerate(hotels_list):
        await verify_single_hotel(evaluator, root, hotel, i)

    # Return structured evaluation summary
    return evaluator.get_summary()