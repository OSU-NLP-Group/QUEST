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
TASK_ID = "chatt_aquarium_hotels"
TASK_DESCRIPTION = (
    "I am planning a family trip to visit the Tennessee Aquarium in Chattanooga, Tennessee, and I am traveling with my pet dog. "
    "I need to find at least 4 different hotels in downtown Chattanooga that meet ALL of the following requirements: "
    "(1) The hotel must be within walking distance of the Tennessee Aquarium, "
    "(2) The hotel must offer complimentary breakfast (free breakfast included), "
    "(3) The hotel must provide free parking OR have parking available nearby, "
    "(4) The hotel must be pet-friendly (allow pets/dogs). "
    "For each hotel you identify, please provide: the official hotel name, the street address, a URL to the hotel's official website or booking platform page, "
    "and confirmation that all four requirements are met."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    url: Optional[str] = None
    extra_urls: List[str] = Field(default_factory=list)


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    From the provided answer text, extract up to 8 hotels that the answer claims satisfy the user's requirements.
    For each hotel, extract the following fields (exactly as written in the answer text; do not invent anything):
    - name: the official hotel name (string)
    - address: the street address (string)
    - url: a single URL to the hotel's official website or a booking platform page (string)
    - extra_urls: an array of any additional URLs specifically associated with the same hotel entry in the answer (array of strings). If none, return an empty array.
    
    Rules:
    - Only extract hotels mentioned in the answer. Do not add hotels that are not in the answer.
    - If a field is missing in the answer, set it to null (for name/address/url) or [] for extra_urls.
    - Preserve the order that the hotels appear in the answer.
    - Ensure URLs are valid-looking; if a URL is missing a protocol, prepend http://
    
    Return JSON with a single key 'hotels' that is an array of hotel objects as specified.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return "".join(ch.lower() for ch in name if ch.isalnum() or ch.isspace()).strip()


def _combine_sources(h: HotelItem) -> List[str]:
    urls: List[str] = []
    if h.url and h.url.strip():
        urls.append(h.url.strip())
    if h.extra_urls:
        for u in h.extra_urls:
            if isinstance(u, str) and u.strip():
                urls.append(u.strip())
    # De-duplicate while preserving order
    seen = set()
    ordered_unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            ordered_unique.append(u)
    return ordered_unique


def _select_first_k_unique_hotels(extraction: HotelsExtraction, k: int = 4) -> List[HotelItem]:
    selected: List[HotelItem] = []
    seen_names = set()
    for h in extraction.hotels:
        norm = _normalize_name(h.name)
        if not norm:
            # allow entries without names only if we still need to pad later; skip for now
            continue
        if norm in seen_names:
            continue
        seen_names.add(norm)
        selected.append(h)
        if len(selected) >= k:
            break
    # If fewer than k, pad with empty placeholders
    while len(selected) < k:
        selected.append(HotelItem())
    return selected


# --------------------------------------------------------------------------- #
# Verification for a single hotel                                             #
# --------------------------------------------------------------------------- #
async def verify_hotel(
    evaluator: Evaluator,
    parent_node,
    hotel: HotelItem,
    hotel_index_1based: int,
) -> None:
    """
    Build verification nodes for one hotel and run verifications.
    """
    # Parent node for this hotel
    hotel_node = evaluator.add_parallel(
        id=f"hotel_{hotel_index_1based}",
        desc=f"Hotel {hotel_index_1based} (must satisfy all constraints and provide required fields).",
        parent=parent_node,
        critical=False,  # Each hotel contributes partially; the global minimum node enforces >=4 distinct listings
    )

    # Critical URL existence node to gate other URL-based checks
    url_exists = bool(hotel.url and hotel.url.strip())
    evaluator.add_custom_node(
        result=url_exists,
        id=f"hotel_{hotel_index_1based}_url",
        desc="Provide a URL to the hotel's official website or a booking platform page.",
        parent=hotel_node,
        critical=True,
    )

    # Prepare sources (may be empty if URL missing; other nodes will auto-skip due to critical sibling 'url' failing)
    sources = _combine_sources(hotel)

    # Add leaf nodes for other required checks (all critical)
    # Name
    name_node = evaluator.add_leaf(
        id=f"hotel_{hotel_index_1based}_name",
        desc="Provide the official hotel name.",
        parent=hotel_node,
        critical=True,
    )
    name_claim = f"The official name of the hotel is '{hotel.name}'." if hotel.name else "The official name of the hotel is provided."
    name_instruction = (
        "Verify on the provided webpage(s) that the property's official name matches the stated name. "
        "Allow minor differences in punctuation, capitalization, or brand qualifiers (e.g., 'Downtown' suffix)."
    )

    # Address
    address_node = evaluator.add_leaf(
        id=f"hotel_{hotel_index_1based}_address",
        desc="Provide the street address.",
        parent=hotel_node,
        critical=True,
    )
    address_claim = (
        f"The hotel's street address is '{hotel.address}' in Chattanooga, TN."
        if hotel.address else
        "The hotel's street address is provided in Chattanooga, TN."
    )
    address_instruction = (
        "Check the webpage(s) for a street address that matches the stated address. "
        "Allow minor abbreviations like 'St' vs 'Street', 'Ave' vs 'Avenue', punctuation, or formatting variations. "
        "It should clearly be a Chattanooga, Tennessee address."
    )

    # Downtown
    downtown_node = evaluator.add_leaf(
        id=f"hotel_{hotel_index_1based}_downtown",
        desc="Hotel is located in downtown Chattanooga, Tennessee.",
        parent=hotel_node,
        critical=True,
    )
    downtown_claim = "This hotel is located in downtown Chattanooga, Tennessee."
    downtown_instruction = (
        "Use the hotel's page to determine if it is in 'Downtown Chattanooga'. "
        "Accept explicit phrases like 'Downtown Chattanooga' or 'in the heart of downtown'. "
        "If the address, page text, map, or screenshot strongly indicates a downtown location (e.g., near the Tennessee Aquarium or downtown landmarks), count it as downtown."
    )

    # Walking distance to Tennessee Aquarium
    walk_node = evaluator.add_leaf(
        id=f"hotel_{hotel_index_1based}_walking_distance",
        desc="Hotel is within walking distance of the Tennessee Aquarium.",
        parent=hotel_node,
        critical=True,
    )
    walk_claim = "This hotel is within walking distance of the Tennessee Aquarium in Chattanooga."
    walk_instruction = (
        "Verify from the hotel page that the Tennessee Aquarium is close enough to walk. "
        "Accept phrases like 'walking distance', 'steps from', 'a short walk', or explicit distances (e.g., under ~1 mile). "
        "If the page clearly states proximity to the Tennessee Aquarium consistent with a walkable distance, count it as supported."
    )

    # Complimentary breakfast
    breakfast_node = evaluator.add_leaf(
        id=f"hotel_{hotel_index_1based}_breakfast",
        desc="Hotel offers complimentary breakfast.",
        parent=hotel_node,
        critical=True,
    )
    breakfast_claim = "This hotel offers complimentary (free) breakfast included for guests."
    breakfast_instruction = (
        "Check the amenities or description for 'free breakfast', 'complimentary breakfast', 'breakfast included', or similar. "
        "Do not count phrases like 'breakfast available' if it implies paid breakfast without inclusion."
    )

    # Parking free or available nearby
    parking_node = evaluator.add_leaf(
        id=f"hotel_{hotel_index_1based}_parking",
        desc="Hotel provides free parking OR has parking available nearby.",
        parent=hotel_node,
        critical=True,
    )
    parking_claim = (
        "This hotel provides free parking OR has parking available on-site or nearby (even if fees apply)."
    )
    parking_instruction = (
        "Support this if the page mentions 'free parking', 'complimentary parking', 'self-parking', 'valet parking', "
        "or indicates a nearby parking garage/lot. Either free parking OR available (paid) parking satisfies the requirement."
    )

    # Pet-friendly
    pet_node = evaluator.add_leaf(
        id=f"hotel_{hotel_index_1based}_pet_friendly",
        desc="Hotel is pet-friendly (allows pets/dogs).",
        parent=hotel_node,
        critical=True,
    )
    pet_claim = "This hotel is pet-friendly and allows pets/dogs (fees or restrictions may apply)."
    pet_instruction = (
        "Look for 'pet-friendly', 'pets allowed', 'dogs allowed', or a 'pet policy'. "
        "Any explicit allowance of pets (even with fees) supports the claim. If the page says 'no pets', it does not support."
    )

    # Batch verify all URL-based leaves (auto-skipped if URL precondition fails)
    claims_and_sources = [
        (name_claim, sources, name_node, name_instruction),
        (address_claim, sources, address_node, address_instruction),
        (downtown_claim, sources, downtown_node, downtown_instruction),
        (walk_claim, sources, walk_node, walk_instruction),
        (breakfast_claim, sources, breakfast_node, breakfast_instruction),
        (parking_claim, sources, parking_node, parking_instruction),
        (pet_claim, sources, pet_node, pet_instruction),
    ]
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the Chattanooga downtown hotels near Tennessee Aquarium task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel: per-hotel checks independent; global minimum is a critical child
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

    # 1) Extract hotel candidates from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction",
    )

    # 2) Select first 4 unique-by-name hotels (pad with empty placeholders if needed)
    selected_hotels = _select_first_k_unique_hotels(extraction, k=4)
    provided_names = [h.name for h in extraction.hotels if h.name]
    unique_name_set = {_normalize_name(n) for n in provided_names if n}
    unique_count_all = len(unique_name_set)

    # Record custom info for debugging
    evaluator.add_custom_info(
        info={
            "total_extracted_hotels": len(extraction.hotels),
            "unique_name_count_all": unique_count_all,
            "selected_hotels_count": len(selected_hotels),
            "selected_names": [h.name for h in selected_hotels],
        },
        info_type="extraction_stats",
        info_name="extraction_stats",
    )

    # 3) Global minimum and distinctness check (critical at root)
    #    Requirement: at least 4 distinct hotels provided in the answer (by name).
    at_least_four_distinct = unique_count_all >= 4
    evaluator.add_custom_node(
        result=at_least_four_distinct,
        id="global_minimum_and_distinctness",
        desc="Provide at least 4 hotels and they must be distinct (not the same hotel repeated).",
        parent=root,
        critical=True,
    )

    # 4) Build verification subtrees for exactly 4 hotels (in order)
    for i in range(4):
        await verify_hotel(
            evaluator=evaluator,
            parent_node=root,
            hotel=selected_hotels[i],
            hotel_index_1based=i + 1,
        )

    # 5) Return structured summary
    return evaluator.get_summary()