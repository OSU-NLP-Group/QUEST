import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "sf_hotels_accessibility_loyalty"
TASK_DESCRIPTION = (
    "I'm planning a trip to San Francisco and need to find two hotels that meet very specific requirements for my needs. "
    "Each hotel must: (1) Have an AAA Four Diamond rating or higher, "
    "(2) Offer accessible rooms that include all of the following ADA-compliant features: doors with at least 32 inches of clear width, "
    "a 60-inch diameter wheelchair turning space (or T-shaped turning area), and roll-in showers measuring at least 30 inches by 60 inches, "
    "(3) Be pet-friendly with a policy that allows dogs weighing at least 50 pounds and permits at least 1 dog per room, "
    "(4) Be located in either downtown San Francisco or the Union Square area, "
    "(5) Belong to a hotel chain with a loyalty program where elite status can be achieved with 60 nights or fewer per year, "
    "and where that elite status includes either complimentary breakfast or lounge access as a benefit. "
    "For each hotel, please provide the hotel name and a reference URL that confirms these details."
)


# -----------------------------
# Extraction Models
# -----------------------------
class HotelItem(BaseModel):
    name: Optional[str] = None
    chain_name: Optional[str] = None
    aaa_urls: List[str] = Field(default_factory=list)
    accessibility_urls: List[str] = Field(default_factory=list)
    pet_policy_urls: List[str] = Field(default_factory=list)
    location_urls: List[str] = Field(default_factory=list)
    loyalty_program_urls: List[str] = Field(default_factory=list)
    elite_tier_name: Optional[str] = None  # optional, if the answer mentions which tier has breakfast/lounge


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


# -----------------------------
# Extraction Prompt
# -----------------------------
def prompt_extract_hotels() -> str:
    return """
    Extract structured information for up to TWO hotels mentioned in the answer that are claimed to meet all criteria.

    For each hotel, extract the following fields:
    - name: The hotel's name (string).
    - chain_name: The hotel chain/brand (e.g., Marriott, Hilton, Hyatt), if given (string or null).
    - aaa_urls: An array of URLs that directly support the hotel's AAA Four Diamond (or Five Diamond) rating.
    - accessibility_urls: An array of URLs that directly support ADA accessibility specifics for rooms, including:
        • doors with at least 32 inches of clear width,
        • a 60-inch diameter wheelchair turning space (or T-shaped turning area),
        • roll-in showers measuring at least 30 inches by 60 inches.
    - pet_policy_urls: An array of URLs that directly support the pet policy allowing dogs weighing at least 50 pounds and permitting at least 1 dog per room.
    - location_urls: An array of URLs that directly support the hotel's location in downtown San Francisco or the Union Square area.
    - loyalty_program_urls: An array of URLs that directly support the loyalty program criteria: elite status reachable with ≤60 nights per year AND that elite status includes complimentary breakfast OR lounge access.
    - elite_tier_name: If the answer mentions which elite tier has breakfast or lounge access (e.g., "Globalist", "Diamond", "Platinum"), extract it as a string; otherwise null.

    SPECIAL RULES:
    - Only extract URLs explicitly present in the answer text. If a source is mentioned but no URL is provided, leave the corresponding array empty.
    - Accept URLs given in plain form or in markdown link format; extract the actual URL.
    - If more than two hotels are mentioned, return only the first two. If fewer than two are mentioned, return what is available.
    - If any field is missing, set it to null or an empty array as appropriate.
    """


# -----------------------------
# Helper
# -----------------------------
def _safe_list(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and len(u.strip()) > 0]


def _union_sources(h: HotelItem) -> List[str]:
    # union of all known sources to help verify chain membership if needed
    combined = []
    for arr in [
        h.location_urls,
        h.aaa_urls,
        h.accessibility_urls,
        h.pet_policy_urls,
        h.loyalty_program_urls,
    ]:
        combined.extend(_safe_list(arr))
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for u in combined:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


# -----------------------------
# Verification per hotel
# -----------------------------
async def verify_hotel(
    evaluator: Evaluator,
    root_parent,
    hotel: HotelItem,
    index: int,
) -> None:
    # Parent node for this hotel (parallel, non-critical to allow partial credit per hotel)
    hotel_node = evaluator.add_parallel(
        id=f"hotel_{index + 1}",
        desc=("First hotel meeting all specified requirements" if index == 0
              else "Second hotel meeting all specified requirements"),
        parent=root_parent,
        critical=False
    )

    # 1) Hotel name provided (existence check)
    name_exists = bool(hotel and hotel.name and hotel.name.strip())
    evaluator.add_custom_node(
        result=name_exists,
        id=f"hotel_{index + 1}_name_provided",
        desc="Provide the hotel name",
        parent=hotel_node,
        critical=True
    )

    hotel_name = hotel.name or ""

    # 2) AAA rating supported (critical leaf)
    aaa_leaf = evaluator.add_leaf(
        id=f"hotel_{index + 1}_aaa_rating_supported",
        desc="Hotel has AAA Four Diamond rating or higher, supported by a provided reference URL",
        parent=hotel_node,
        critical=True
    )
    aaa_claim = f"The hotel '{hotel_name}' has an AAA Four Diamond rating or higher (Four Diamond or Five Diamond)."
    await evaluator.verify(
        claim=aaa_claim,
        node=aaa_leaf,
        sources=_safe_list(hotel.aaa_urls),
        additional_instruction=(
            "Confirm that the page explicitly states an AAA Four Diamond or AAA Five Diamond rating for the hotel. "
            "Do not accept generic star ratings or non-AAA ratings."
        )
    )

    # 3) Accessibility supported: break into 3 critical checks under one critical parent
    access_parent = evaluator.add_parallel(
        id=f"hotel_{index + 1}_accessibility_supported",
        desc=("Accessible rooms include all of the following: (1) doors with at least 32-inch clear width, "
              "(2) 60-inch diameter wheelchair turning space or T-shaped turning area, and "
              "(3) roll-in showers measuring at least 30 inches by 60 inches; supported by a provided reference URL"),
        parent=hotel_node,
        critical=True
    )

    # 3a) Door width >= 32"
    door_leaf = evaluator.add_leaf(
        id=f"hotel_{index + 1}_ada_door_width_32in",
        desc="Accessible room doors have at least 32 inches of clear width",
        parent=access_parent,
        critical=True
    )
    door_claim = f"The accessible rooms at '{hotel_name}' have doors with at least 32 inches of clear width."
    await evaluator.verify(
        claim=door_claim,
        node=door_leaf,
        sources=_safe_list(hotel.accessibility_urls),
        additional_instruction=(
            "Verify ADA-compliant door clear width for accessible rooms (≥32 inches). "
            "Confirm the measurement or specification is explicitly stated."
        )
    )

    # 3b) Turning space 60" diameter or T-shaped turning area
    turn_leaf = evaluator.add_leaf(
        id=f"hotel_{index + 1}_ada_turning_space_60in",
        desc="Accessible rooms provide a 60-inch diameter turning space or T-shaped turning area",
        parent=access_parent,
        critical=True
    )
    turn_claim = (f"The accessible rooms at '{hotel_name}' provide either a 60-inch diameter wheelchair turning space "
                  "or a T-shaped turning area.")
    await evaluator.verify(
        claim=turn_claim,
        node=turn_leaf,
        sources=_safe_list(hotel.accessibility_urls),
        additional_instruction=(
            "Verify that the accessible room layout includes either a 60-inch diameter turning space "
            "or a T-shaped turning area."
        )
    )

    # 3c) Roll-in showers at least 30x60 inches
    shower_leaf = evaluator.add_leaf(
        id=f"hotel_{index + 1}_ada_roll_in_shower_30x60",
        desc="Roll-in showers measure at least 30 inches by 60 inches",
        parent=access_parent,
        critical=True
    )
    shower_claim = f"The accessible rooms at '{hotel_name}' feature roll-in showers of at least 30 inches by 60 inches."
    await evaluator.verify(
        claim=shower_claim,
        node=shower_leaf,
        sources=_safe_list(hotel.accessibility_urls),
        additional_instruction=(
            "Verify that roll-in showers meet or exceed 30 inches by 60 inches. "
            "Look for explicit dimensions or ADA-compliant specs."
        )
    )

    # 4) Pet policy supported: break into 2 critical checks under one critical parent
    pet_parent = evaluator.add_parallel(
        id=f"hotel_{index + 1}_pet_policy_supported",
        desc="Hotel is pet-friendly and allows dogs weighing at least 50 pounds and permits at least 1 dog per room; supported by a provided reference URL",
        parent=hotel_node,
        critical=True
    )

    pet_weight_leaf = evaluator.add_leaf(
        id=f"hotel_{index + 1}_pet_weight_50lb",
        desc="Allows dogs weighing at least 50 pounds",
        parent=pet_parent,
        critical=True
    )
    pet_weight_claim = f"The hotel '{hotel_name}' allows dogs weighing at least 50 pounds."
    await evaluator.verify(
        claim=pet_weight_claim,
        node=pet_weight_leaf,
        sources=_safe_list(hotel.pet_policy_urls),
        additional_instruction=(
            "Verify the stated weight limit or allowance in the pet policy page. "
            "The policy should allow dogs of at least 50 lbs."
        )
    )

    pet_count_leaf = evaluator.add_leaf(
        id=f"hotel_{index + 1}_pet_min_one_dog",
        desc="Permits at least 1 dog per room",
        parent=pet_parent,
        critical=True
    )
    pet_count_claim = f"The hotel '{hotel_name}' permits at least 1 dog per room."
    await evaluator.verify(
        claim=pet_count_claim,
        node=pet_count_leaf,
        sources=_safe_list(hotel.pet_policy_urls),
        additional_instruction=(
            "Verify that the pet policy allows at least one dog per room. "
            "Look for per-room pet limits."
        )
    )

    # 5) Location supported (critical leaf)
    location_leaf = evaluator.add_leaf(
        id=f"hotel_{index + 1}_location_supported",
        desc="Hotel is located in downtown San Francisco or the Union Square area; supported by a provided reference URL",
        parent=hotel_node,
        critical=True
    )
    location_claim = (f"The hotel '{hotel_name}' is located in either downtown San Francisco or the Union Square area "
                      "of San Francisco.")
    await evaluator.verify(
        claim=location_claim,
        node=location_leaf,
        sources=_safe_list(hotel.location_urls),
        additional_instruction=(
            "Confirm the hotel is in downtown San Francisco or the Union Square area. "
            "Accept mentions such as 'Union Square', 'Downtown San Francisco', 'steps from Union Square', etc."
        )
    )

    # 6) Loyalty program supported: break into subchecks under one critical parent
    loyalty_parent = evaluator.add_parallel(
        id=f"hotel_{index + 1}_elite_program_supported",
        desc=("Hotel belongs to a chain with a loyalty program where elite status is achievable with 60 nights or fewer per year, "
              "and that elite status includes either complimentary breakfast or lounge access; supported by a provided reference URL"),
        parent=hotel_node,
        critical=True
    )

    # 6a) Chain membership (critical)
    chain_leaf = evaluator.add_leaf(
        id=f"hotel_{index + 1}_chain_membership",
        desc="Hotel belongs to the stated chain/brand",
        parent=loyalty_parent,
        critical=True
    )
    chain_name = hotel.chain_name or "the stated hotel chain/brand"
    chain_claim = f"The hotel '{hotel_name}' belongs to {chain_name}."
    await evaluator.verify(
        claim=chain_claim,
        node=chain_leaf,
        sources=_union_sources(hotel),
        additional_instruction=(
            "Use any provided official hotel or brand page to confirm chain affiliation. "
            "Look for the brand name or chain explicitly associated with the hotel."
        )
    )

    # 6b) Elite nights threshold ≤ 60 (critical)
    nights_leaf = evaluator.add_leaf(
        id=f"hotel_{index + 1}_elite_status_nights_threshold",
        desc="Elite status is achievable with 60 nights or fewer per year",
        parent=loyalty_parent,
        critical=True
    )
    tier_txt = f" ({hotel.elite_tier_name})" if hotel.elite_tier_name else ""
    nights_claim = (f"For {chain_name}'s loyalty program{tier_txt}, an elite status tier can be reached with 60 nights or fewer "
                    "in a year.")
    await evaluator.verify(
        claim=nights_claim,
        node=nights_leaf,
        sources=_safe_list(hotel.loyalty_program_urls),
        additional_instruction=(
            "Verify an elite status threshold of ≤60 nights per year for the program/tier. "
            "If thresholds are given in stays, confirm equivalence that meets ≤60 nights where applicable; prefer explicit nights."
        )
    )

    # 6c) Elite benefits include breakfast or lounge access (critical)
    benefits_leaf = evaluator.add_leaf(
        id=f"hotel_{index + 1}_elite_benefits_breakfast_or_lounge",
        desc="Elite status includes complimentary breakfast or lounge access",
        parent=loyalty_parent,
        critical=True
    )
    benefits_claim = (f"In {chain_name}'s loyalty program{tier_txt}, the elite status tier includes either complimentary breakfast "
                      "or lounge access as a benefit.")
    await evaluator.verify(
        claim=benefits_claim,
        node=benefits_leaf,
        sources=_safe_list(hotel.loyalty_program_urls),
        additional_instruction=(
            "Confirm that at least one of the following is explicitly included at the elite tier: "
            "complimentary breakfast OR lounge access."
        )
    )


# -----------------------------
# Main Evaluation Entry
# -----------------------------
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Two hotels evaluated independently
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=(
            "Find two hotels in San Francisco that meet the specified rating, accessibility, pet policy, location, "
            "and loyalty-program criteria; provide hotel name and supporting reference URL(s) for each required criterion"
        ),
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract hotels and their supporting URLs
    extraction = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction"
    )

    # Keep only the first two hotels; pad with empty if fewer than two
    hotels: List[HotelItem] = list(extraction.hotels[:2])
    while len(hotels) < 2:
        hotels.append(HotelItem())

    # Build and verify for each hotel
    for idx, hotel in enumerate(hotels):
        await verify_hotel(evaluator, root, hotel, idx)

    return evaluator.get_summary()