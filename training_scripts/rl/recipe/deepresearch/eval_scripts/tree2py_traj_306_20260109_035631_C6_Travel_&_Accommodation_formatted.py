import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nyc_madison_hotels_accessible_pets_meeting_breakfast"
TASK_DESCRIPTION = (
    "A music band is traveling to New York City to perform at Madison Square Garden and needs accommodation for their crew. "
    "Find two hotels in Manhattan that meet all of the following requirements:\n\n"
    "1. The hotel must be located within 1 mile walking distance of Madison Square Garden\n"
    "2. The hotel must have wheelchair accessible rooms that are ADA-compliant, including:\n"
    "   - Minimum 32-inch door width in accessible rooms\n"
    "   - Wheelchair turning space (60-inch diameter or T-turn) in accessible rooms\n"
    "   - Grab bars in accessible bathrooms (on side and back wall near toilet)\n"
    "   - Roll-in shower or accessible bathtub with seat in accessible bathrooms\n"
    "3. The hotel must accept dogs with no weight restrictions (able to accommodate dogs over 50 pounds)\n"
    "4. The hotel must have a meeting room or event space that can accommodate at least 30 people\n"
    "5. The hotel must offer complimentary breakfast included in the room rate\n\n"
    "For each hotel, provide:\n"
    "- Hotel name\n"
    "- Hotel address or location in Manhattan\n"
    "- Reference URL(s) that verify each of the five requirements above"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    proximity_urls: List[str] = Field(default_factory=list)
    accessibility_urls: List[str] = Field(default_factory=list)
    pet_policy_urls: List[str] = Field(default_factory=list)
    meeting_space_urls: List[str] = Field(default_factory=list)
    breakfast_urls: List[str] = Field(default_factory=list)


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
Extract exactly two distinct hotels from the answer that the agent proposed to satisfy the task. For each hotel, extract the following fields strictly from the answer text:

- name: The hotel name as stated.
- address: The address or location text as stated (Manhattan, NYC).
- proximity_urls: URL(s) specifically used to substantiate that the hotel is within 1 mile walking distance of Madison Square Garden. These are typically Google Maps walking directions links or pages explicitly stating a walking distance ≤ 1.0 mile.
- accessibility_urls: URL(s) that substantiate ADA-compliant wheelchair-accessible room features, including ALL of:
    • Minimum 32-inch door width
    • Wheelchair turning space (60-inch diameter circle or a T-turn)
    • Grab bars in accessible bathroom (side and back wall near toilet)
    • Roll-in shower OR accessible bathtub with seat
- pet_policy_urls: URL(s) that substantiate dogs are accepted with no weight restrictions (can accommodate dogs over 50 lbs).
- meeting_space_urls: URL(s) that substantiate the existence of a meeting or event space with capacity for at least 30 people.
- breakfast_urls: URL(s) that substantiate complimentary breakfast is included in the room rate.

Important extraction rules:
1) Only extract URLs explicitly present in the answer; do not invent or infer new URLs.
2) Keep URLs exactly as they appear (markdown or plain). If in markdown, extract the URL target.
3) If a category has no URLs provided in the answer, return an empty array for that category.
4) If more than two hotels are listed, return only the first two as they appear in the answer.
5) If fewer than two hotels are provided, return all available.

Return a JSON object with a field "hotels": an array of up to two HotelItem objects.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _filter_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out: List[str] = []
    seen: set = set()
    for u in urls:
        if not u:
            continue
        s = str(u).strip()
        if not s:
            continue
        # some answers might miss protocol; allow but keep as-is as the framework normalizes later if possible
        # basic validation: must contain at least a dot or be a valid map/domain-like link
        if "." not in s and "http" not in s and "maps" not in s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _union_all_urls(h: HotelItem) -> List[str]:
    """Union of all URL categories for a hotel (deduped)."""
    combined = (
        h.proximity_urls
        + h.accessibility_urls
        + h.pet_policy_urls
        + h.meeting_space_urls
        + h.breakfast_urls
    )
    return _filter_urls(combined)


def _normalize_name(name: Optional[str]) -> str:
    return (name or "").strip().lower()


def _exactly_two_distinct(hotels: List[HotelItem]) -> bool:
    if len(hotels) != 2:
        return False
    n1 = _normalize_name(hotels[0].name)
    n2 = _normalize_name(hotels[1].name)
    return bool(n1) and bool(n2) and (n1 != n2)


# --------------------------------------------------------------------------- #
# Verification for a single hotel                                             #
# --------------------------------------------------------------------------- #
async def verify_hotel(
    evaluator: Evaluator,
    parent_node,
    hotel: HotelItem,
    hotel_idx: int,
) -> None:
    """
    Build and verify the subtree for a single hotel according to the rubric.
    """
    hotel_node = evaluator.add_parallel(
        id=f"hotel_{hotel_idx+1}",
        desc=f"Hotel {hotel_idx+1} (evaluated independently for partial credit)",
        parent=parent_node,
        critical=False
    )

    # 1) Hotel name provided (critical existence check)
    name_exists = bool(hotel.name and hotel.name.strip())
    evaluator.add_custom_node(
        result=name_exists,
        id=f"hotel_{hotel_idx+1}_name",
        desc="Hotel name is provided",
        parent=hotel_node,
        critical=True
    )

    # Prepare URL lists (filtered) early; these will be used both for existence checks and content verification
    proximity_urls = _filter_urls(hotel.proximity_urls)
    accessibility_urls = _filter_urls(hotel.accessibility_urls)
    pet_urls = _filter_urls(hotel.pet_policy_urls)
    meeting_urls = _filter_urls(hotel.meeting_space_urls)
    breakfast_urls = _filter_urls(hotel.breakfast_urls)
    all_urls = _union_all_urls(hotel)

    # 2) Hotel Manhattan location (critical check via sources if available)
    manhattan_leaf = evaluator.add_leaf(
        id=f"hotel_{hotel_idx+1}_manhattan_location",
        desc="Hotel address/location is provided and is in Manhattan, NYC",
        parent=hotel_node,
        critical=True
    )
    addr_part = f" with address '{hotel.address}'" if hotel.address else ""
    manhattan_claim = f"The hotel '{hotel.name}'{addr_part} is located in Manhattan, New York City."
    await evaluator.verify(
        claim=manhattan_claim,
        node=manhattan_leaf,
        sources=all_urls,
        additional_instruction=(
            "Verify from the provided URL(s) that the property is in Manhattan, NYC (New York County). "
            "Accept addresses explicitly in Manhattan borough (e.g., neighborhoods like Midtown, Chelsea, SoHo, etc.)."
        )
    )

    # 3) Proximity URL existence (critical sibling)
    evaluator.add_custom_node(
        result=len(proximity_urls) > 0,
        id=f"hotel_{hotel_idx+1}_proximity_url",
        desc="Provides reference URL(s) verifying the within-1-mile walking distance to Madison Square Garden",
        parent=hotel_node,
        critical=True
    )

    # 4) Proximity content verification (critical)
    proximity_leaf = evaluator.add_leaf(
        id=f"hotel_{hotel_idx+1}_proximity",
        desc="Hotel is within 1 mile walking distance of Madison Square Garden",
        parent=hotel_node,
        critical=True
    )
    proximity_claim = (
        f"The hotel '{hotel.name}' is within 1 mile walking distance to Madison Square Garden in New York City."
    )
    await evaluator.verify(
        claim=proximity_claim,
        node=proximity_leaf,
        sources=proximity_urls,
        additional_instruction=(
            "Use only the provided proximity URL(s). Prefer Google Maps walking directions or explicit statements "
            "on distance. Confirm that the walking distance is ≤ 1.0 mile."
        )
    )

    # 5) Accessibility URL existence (critical sibling before content)
    evaluator.add_custom_node(
        result=len(accessibility_urls) > 0,
        id=f"hotel_{hotel_idx+1}_accessibility_url",
        desc="Provides reference URL(s) verifying the required ADA accessible room/bathroom features",
        parent=hotel_node,
        critical=True
    )

    # 6) Accessibility content verification (critical group with 4 required sub-features)
    access_node = evaluator.add_parallel(
        id=f"hotel_{hotel_idx+1}_accessibility",
        desc="Hotel offers ADA-compliant wheelchair-accessible rooms that meet all specified accessibility features (door width, turning space, grab bars, roll-in shower or accessible tub with seat)",
        parent=hotel_node,
        critical=True
    )

    # 6.1) 32-inch door width
    door_leaf = evaluator.add_leaf(
        id=f"hotel_{hotel_idx+1}_accessible_door_width",
        desc="Accessible rooms have minimum 32-inch door width",
        parent=access_node,
        critical=True
    )
    door_claim = (
        f"The accessible guest rooms at '{hotel.name}' have door clear widths of at least 32 inches."
    )
    # 6.2) Turning space
    turn_leaf = evaluator.add_leaf(
        id=f"hotel_{hotel_idx+1}_accessible_turning_space",
        desc="Accessible rooms have wheelchair turning space (60-inch diameter or T-turn)",
        parent=access_node,
        critical=True
    )
    turn_claim = (
        f"The accessible guest rooms at '{hotel.name}' provide wheelchair turning space (a 60-inch diameter circle or a T-turn)."
    )
    # 6.3) Grab bars
    grab_leaf = evaluator.add_leaf(
        id=f"hotel_{hotel_idx+1}_accessible_grab_bars",
        desc="Accessible bathroom has grab bars on side and back wall near toilet",
        parent=access_node,
        critical=True
    )
    grab_claim = (
        f"The accessible bathrooms at '{hotel.name}' include grab bars on both the side wall and back wall near the toilet."
    )
    # 6.4) Roll-in shower or accessible tub with seat
    shower_leaf = evaluator.add_leaf(
        id=f"hotel_{hotel_idx+1}_accessible_shower_or_tub",
        desc="Accessible bathroom has roll-in shower OR accessible bathtub with seat",
        parent=access_node,
        critical=True
    )
    shower_claim = (
        f"The accessible bathrooms at '{hotel.name}' provide either a roll-in shower or an accessible bathtub with a seat."
    )

    await evaluator.verify(
        claim=door_claim,
        node=door_leaf,
        sources=accessibility_urls,
        additional_instruction=(
            "Verify that the source explicitly states door widths are at least 32 inches (clear width) for accessible rooms."
        )
    )
    await evaluator.verify(
        claim=turn_claim,
        node=turn_leaf,
        sources=accessibility_urls,
        additional_instruction=(
            "Verify that the source explicitly states wheelchair turning space of either a 60-inch diameter circle or a T-turn in accessible rooms."
        )
    )
    await evaluator.verify(
        claim=grab_claim,
        node=grab_leaf,
        sources=accessibility_urls,
        additional_instruction=(
            "Verify that the source explicitly mentions grab bars on the side and back wall near the toilet in accessible bathrooms."
        )
    )
    await evaluator.verify(
        claim=shower_claim,
        node=shower_leaf,
        sources=accessibility_urls,
        additional_instruction=(
            "Verify that the source explicitly mentions either roll-in shower availability OR an accessible bathtub with a seat in accessible bathrooms."
        )
    )

    # 7) Pet policy URL existence (critical sibling)
    evaluator.add_custom_node(
        result=len(pet_urls) > 0,
        id=f"hotel_{hotel_idx+1}_pet_policy_url",
        desc="Provides reference URL(s) verifying dog acceptance with no weight restriction",
        parent=hotel_node,
        critical=True
    )

    # 8) Pet policy content verification (critical)
    pet_leaf = evaluator.add_leaf(
        id=f"hotel_{hotel_idx+1}_pet_policy",
        desc="Hotel accepts dogs with no weight restrictions (able to accommodate dogs over 50 pounds)",
        parent=hotel_node,
        critical=True
    )
    pet_claim = (
        f"The hotel '{hotel.name}' accepts dogs with no weight limit (can accommodate dogs over 50 pounds)."
    )
    await evaluator.verify(
        claim=pet_claim,
        node=pet_leaf,
        sources=pet_urls,
        additional_instruction=(
            "Confirm there is no stated weight limit for dogs, or that large dogs (over 50 lbs) are allowed. "
            "If the source lists a weight limit, this claim is not supported."
        )
    )

    # 9) Meeting space URL existence (critical sibling)
    evaluator.add_custom_node(
        result=len(meeting_urls) > 0,
        id=f"hotel_{hotel_idx+1}_meeting_space_url",
        desc="Provides reference URL(s) verifying meeting/event space capacity of at least 30 people",
        parent=hotel_node,
        critical=True
    )

    # 10) Meeting space content verification (critical)
    meeting_leaf = evaluator.add_leaf(
        id=f"hotel_{hotel_idx+1}_meeting_space",
        desc="Hotel has a meeting room or event space that can accommodate at least 30 people",
        parent=hotel_node,
        critical=True
    )
    meeting_claim = (
        f"The hotel '{hotel.name}' has a meeting room or event space with capacity for at least 30 people."
    )
    await evaluator.verify(
        claim=meeting_claim,
        node=meeting_leaf,
        sources=meeting_urls,
        additional_instruction=(
            "Look for room capacity charts, floor plans, venue pages, or descriptions indicating a minimum capacity of 30 attendees in a single room."
        )
    )

    # 11) Breakfast URL existence (critical sibling)
    evaluator.add_custom_node(
        result=len(breakfast_urls) > 0,
        id=f"hotel_{hotel_idx+1}_breakfast_url",
        desc="Provides reference URL(s) verifying complimentary breakfast included in the room rate",
        parent=hotel_node,
        critical=True
    )

    # 12) Breakfast content verification (critical)
    breakfast_leaf = evaluator.add_leaf(
        id=f"hotel_{hotel_idx+1}_breakfast",
        desc="Hotel offers complimentary breakfast included in the room rate",
        parent=hotel_node,
        critical=True
    )
    breakfast_claim = (
        f"The hotel '{hotel.name}' includes complimentary breakfast in the room rate (not an extra paid add-on)."
    )
    await evaluator.verify(
        claim=breakfast_claim,
        node=breakfast_leaf,
        sources=breakfast_urls,
        additional_instruction=(
            "Confirm breakfast is complimentary and included in the rate. Statements like 'free breakfast' or 'complimentary breakfast included' are acceptable; "
            "if breakfast is paid/separately charged, this claim is not supported."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the NYC Madison Square Garden hotel requirements.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel; critical gating handled by child nodes
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

    # 1) Extract structured hotel info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction"
    )

    # 2) Normalize to exactly two hotels for evaluation (pad with empty if fewer)
    hotels: List[HotelItem] = list(extracted.hotels[:2])
    while len(hotels) < 2:
        hotels.append(HotelItem())

    # 3) Critical check: exactly two distinct hotels provided
    evaluator.add_custom_node(
        result=_exactly_two_distinct(hotels),
        id="two_distinct_hotels_provided",
        desc="Exactly two distinct hotels are provided",
        parent=root,
        critical=True
    )

    # 4) Build and verify each hotel subtree
    await verify_hotel(evaluator, root, hotels[0], 0)
    await verify_hotel(evaluator, root, hotels[1], 1)

    # 5) Return evaluation summary
    return evaluator.get_summary()