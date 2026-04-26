import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "airport_hotel_one"
TASK_DESCRIPTION = (
    "Identify one hotel at a major U.S. airport that meets ALL of the following criteria:\n"
    "- The hotel is directly connected to an airport terminal via skybridge or covered walkway, OR is located within a short walk (described as \"steps away\" or similar) from the main terminal\n"
    "- The hotel offers at least 30,000 square feet of meeting and event space\n"
    "- The hotel is pet-friendly and accepts dogs\n"
    "- The hotel has at least 280 guest rooms\n\n"
    "For the hotel you identify, provide the following information with reference URL(s):\n"
    "1. The name of the hotel\n"
    "2. The airport it serves (full airport name)\n"
    "3. The exact square footage of meeting/event space\n"
    "4. The total number of guest rooms\n"
    "5. The pet policy regarding weight limits (specify if there's a maximum weight limit or if there's no weight limit)\n"
    "6. Whether the hotel has an indoor pool (yes or no)\n"
    "7. The number of on-site restaurants\n"
    "8. A description of how guests access the terminal from the hotel"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    hotel_name: Optional[str] = None
    airport_full_name: Optional[str] = None

    meeting_space_sqft: Optional[str] = None
    guest_rooms_total: Optional[str] = None

    pet_friendly_dogs: Optional[str] = None  # e.g., "yes", "dogs allowed", "no"
    pet_policy_weight_limit: Optional[str] = None  # e.g., "up to 50 lbs", "no weight limit"

    indoor_pool: Optional[str] = None  # "yes" or "no"
    onsite_restaurants_count: Optional[str] = None

    terminal_access_description: Optional[str] = None

    # Sources per item/constraint
    sources_hotel_name: List[str] = Field(default_factory=list)
    sources_airport_served: List[str] = Field(default_factory=list)
    sources_major_airport_status: List[str] = Field(default_factory=list)
    sources_meeting_space: List[str] = Field(default_factory=list)
    sources_guest_rooms: List[str] = Field(default_factory=list)
    sources_pet_friendly_dogs: List[str] = Field(default_factory=list)
    sources_pet_weight_limit_policy: List[str] = Field(default_factory=list)
    sources_indoor_pool: List[str] = Field(default_factory=list)
    sources_restaurants_count: List[str] = Field(default_factory=list)
    sources_terminal_access: List[str] = Field(default_factory=list)


class HotelCountExtraction(BaseModel):
    hotel_names_mentioned: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_item() -> str:
    return (
        "Extract exactly one candidate hotel and all required fields as they appear in the answer text. If multiple hotels are mentioned, pick the one the answer ultimately recommends or focuses on and still extract all fields; however, we will separately count total hotels mentioned in another extraction.\n\n"
        "You must extract these fields:\n"
        "- hotel_name: The property's official hotel name.\n"
        "- airport_full_name: The full airport name the hotel serves (e.g., \"Hartsfield–Jackson Atlanta International Airport\").\n"
        "- meeting_space_sqft: The exact total square footage of meeting/event space (as written, keep units and formatting).\n"
        "- guest_rooms_total: The hotel's total number of guest rooms (as written).\n"
        "- pet_friendly_dogs: Whether the hotel accepts dogs (e.g., \"yes\", \"dogs allowed\").\n"
        "- pet_policy_weight_limit: The stated pet weight limit policy (e.g., \"up to 50 lbs\", \"no weight limit\").\n"
        "- indoor_pool: Whether the hotel has an indoor pool (\"yes\" or \"no\").\n"
        "- onsite_restaurants_count: The number of on-site restaurants (as written).\n"
        "- terminal_access_description: A description of how guests access the terminal (e.g., \"via skybridge\", \"covered walkway\", \"steps away\", \"short walk\").\n\n"
        "For each of the following, also extract one or more reference URL(s) explicitly cited in the answer that support the claim:\n"
        "- sources_hotel_name\n"
        "- sources_airport_served\n"
        "- sources_major_airport_status (e.g., FAA hub classification page, airport page explicitly calling it major)\n"
        "- sources_meeting_space\n"
        "- sources_guest_rooms\n"
        "- sources_pet_friendly_dogs\n"
        "- sources_pet_weight_limit_policy\n"
        "- sources_indoor_pool\n"
        "- sources_restaurants_count\n"
        "- sources_terminal_access\n\n"
        "Rules:\n"
        "1) Return only URLs that are explicitly present in the answer (plain URLs or markdown links). If no URL is given for an item, return an empty list for that sources_* field.\n"
        "2) Do not infer or invent numbers or URLs.\n"
        "3) Use the answer text exactly; if a field is missing, set it to null.\n"
    )


def prompt_extract_hotel_count() -> str:
    return (
        "List all distinct hotel property names mentioned anywhere in the answer. Deduplicate synonymous references if they point to the same property. "
        "Return them in 'hotel_names_mentioned' as an array. If none are present, return an empty array."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_int_first_number(text: Optional[str]) -> Optional[int]:
    """Extract the first integer from text; returns None if not found."""
    if not text:
        return None
    # Remove commas, capture first integer-like sequence
    m = re.search(r"(\d[\d,]*)", text)
    if not m:
        return None
    cleaned = m.group(1).replace(",", "")
    try:
        return int(cleaned)
    except Exception:
        return None


def str_truthy(text: Optional[str]) -> bool:
    """Return True if text appears to affirm (e.g., 'yes', 'dogs allowed')."""
    if not text:
        return False
    t = text.strip().lower()
    positives = {"yes", "y", "true", "dogs allowed", "dog-friendly", "pet-friendly", "accepts dogs"}
    return any(p in t for p in positives)


def normalize_yes_no(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = text.strip().lower()
    if t in {"yes", "y", "true"}:
        return "yes"
    if t in {"no", "n", "false"}:
        return "no"
    # attempt heuristic
    if "no" in t and "pool" in t:
        return "no"
    if "indoor" in t and ("pool" in t or "swimming" in t):
        return "yes"
    return None


def combine_sources(*lists: List[str]) -> List[str]:
    """Combine and deduplicate source URL lists."""
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for u in lst:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification tree builder                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root_node,
    count_info: HotelCountExtraction,
    item: HotelItem,
) -> None:
    # Create the critical evaluation parent node
    eval_parent = evaluator.add_parallel(
        id="Airport_Hotel_Evaluation",
        desc="Evaluate that exactly one identified hotel satisfies all required selection criteria and that all requested fields are provided with supporting reference URL(s).",
        parent=root_node,
        critical=True,
    )

    # Exactly One Hotel Identified
    exactly_one = evaluator.add_custom_node(
        result=(len(count_info.hotel_names_mentioned) == 1),
        id="Exactly_One_Hotel_Identified",
        desc="Identifies exactly one hotel (not multiple candidate hotels).",
        parent=eval_parent,
        critical=True,
    )

    # Hotel Name Provided
    hotel_name_ok = evaluator.add_custom_node(
        result=bool(item.hotel_name and item.hotel_name.strip()),
        id="Hotel_Name_Provided",
        desc="Provides the hotel name.",
        parent=eval_parent,
        critical=True,
    )

    # Airport Served - Provided and Supported
    airport_provided = evaluator.add_custom_node(
        result=bool(item.airport_full_name and item.airport_full_name.strip()),
        id="Airport_Served_Full_Name_Provided",
        desc="Provides the full airport name the hotel serves.",
        parent=eval_parent,
        critical=True,
    )

    # Major U.S. Airport Requirement (composite checks)
    major_airport_node = evaluator.add_parallel(
        id="Major_US_Airport_Requirement",
        desc="Verifies the hotel serves a U.S. airport and provides a citation that supports the airport qualifies as a 'major' airport.",
        parent=eval_parent,
        critical=True,
    )

    # Airport Served supported by sources
    airport_served_supported = evaluator.add_leaf(
        id="Airport_Served_Supported",
        desc="The hotel serves the stated airport (supported by sources).",
        parent=major_airport_node,
        critical=True,
    )
    claim_airport_served = f"The hotel '{item.hotel_name or ''}' serves or is located at '{item.airport_full_name or ''}'."
    await evaluator.verify(
        claim=claim_airport_served,
        node=airport_served_supported,
        sources=combine_sources(item.sources_airport_served, item.sources_terminal_access),
        additional_instruction="Accept formulations such as 'connected to Terminal X', 'located at the airport', or 'on airport grounds' as serving the airport.",
    )

    # U.S. location of airport
    is_us_airport_leaf = evaluator.add_leaf(
        id="US_Airport_Verified",
        desc="The stated airport is located in the United States.",
        parent=major_airport_node,
        critical=True,
    )
    claim_us_airport = f"The airport '{item.airport_full_name or ''}' is located in the United States."
    await evaluator.verify(
        claim=claim_us_airport,
        node=is_us_airport_leaf,
        sources=combine_sources(item.sources_major_airport_status, item.sources_airport_served),
        additional_instruction="Verify the airport's country/location; typical sources include the airport's official page or Wikipedia indicating it's a U.S. airport.",
    )

    # Major airport status verification
    major_status_leaf = evaluator.add_leaf(
        id="Major_Airport_Status_Supported",
        desc="The airport qualifies as a 'major' U.S. airport per cited source (e.g., FAA hub classification or explicit 'major' description).",
        parent=major_airport_node,
        critical=True,
    )
    claim_major_status = (
        f"The airport '{item.airport_full_name or ''}' is a 'major' U.S. airport, "
        "as indicated by a recognized classification (e.g., FAA large or medium hub) or an explicit 'major' designation in the cited source."
    )
    await evaluator.verify(
        claim=claim_major_status,
        node=major_status_leaf,
        sources=item.sources_major_airport_status,
        additional_instruction=(
            "Treat FAA hub classifications 'Large Hub' and 'Medium Hub' as major airports. "
            "If the source explicitly calls it 'major', that also suffices."
        ),
    )

    # Terminal Access Requirement (description + support)
    terminal_access_node = evaluator.add_parallel(
        id="Terminal_Access_Requirement",
        desc="Provides a description of terminal access (direct connection via skybridge/covered walkway or short walk/steps away).",
        parent=eval_parent,
        critical=True,
    )
    terminal_desc_provided = evaluator.add_custom_node(
        result=bool(item.terminal_access_description and item.terminal_access_description.strip()),
        id="Terminal_Access_Description_Provided",
        desc="Terminal access description is provided.",
        parent=terminal_access_node,
        critical=True,
    )
    terminal_access_supported = evaluator.add_leaf(
        id="Terminal_Access_Supported",
        desc="Terminal access mode is supported by sources.",
        parent=terminal_access_node,
        critical=True,
    )
    claim_terminal_access = (
        "Guests can access the airport terminal via a skybridge or covered walkway, "
        "or the hotel is described as within a short walk (e.g., 'steps away') from the main terminal."
    )
    await evaluator.verify(
        claim=claim_terminal_access,
        node=terminal_access_supported,
        sources=item.sources_terminal_access,
        additional_instruction=(
            "Confirm the source explicitly mentions a connected skybridge/covered walkway or phrases like 'steps away', "
            "'short walk', or 'adjacent to terminal'."
        ),
    )

    # Meeting Space Requirement (provided, supported, threshold)
    meeting_space_node = evaluator.add_parallel(
        id="Meeting_Space_Requirement",
        desc="Provides the exact meeting/event space square footage and verifies it is at least 30,000 sq ft.",
        parent=eval_parent,
        critical=True,
    )
    meeting_space_provided = evaluator.add_custom_node(
        result=bool(item.meeting_space_sqft and item.meeting_space_sqft.strip()),
        id="Meeting_Space_Provided",
        desc="Meeting/event space square footage is provided.",
        parent=meeting_space_node,
        critical=True,
    )
    meeting_space_supported = evaluator.add_leaf(
        id="Meeting_Space_Supported",
        desc="Stated meeting/event space square footage is supported by sources.",
        parent=meeting_space_node,
        critical=True,
    )
    claim_meeting_sqft = f"The hotel has {item.meeting_space_sqft or ''} of total meeting and event space."
    await evaluator.verify(
        claim=claim_meeting_sqft,
        node=meeting_space_supported,
        sources=item.sources_meeting_space,
        additional_instruction="Verify the total meeting/event space square footage matches the source content.",
    )
    meeting_sqft_val = parse_int_first_number(item.meeting_space_sqft)
    meeting_space_threshold = evaluator.add_custom_node(
        result=(meeting_sqft_val is not None and meeting_sqft_val >= 30000),
        id="Meeting_Space_At_Least_30000",
        desc=f"Meeting/event space is at least 30,000 sq ft (parsed: {meeting_sqft_val if meeting_sqft_val is not None else 'None'}).",
        parent=meeting_space_node,
        critical=True,
    )

    # Guest Rooms Requirement (provided, supported, threshold)
    guest_rooms_node = evaluator.add_parallel(
        id="Guest_Rooms_Requirement",
        desc="Provides total guest rooms and verifies it is at least 280.",
        parent=eval_parent,
        critical=True,
    )
    guest_rooms_provided = evaluator.add_custom_node(
        result=bool(item.guest_rooms_total and item.guest_rooms_total.strip()),
        id="Guest_Rooms_Provided",
        desc="Guest rooms total is provided.",
        parent=guest_rooms_node,
        critical=True,
    )
    guest_rooms_supported = evaluator.add_leaf(
        id="Guest_Rooms_Supported",
        desc="Stated total guest rooms is supported by sources.",
        parent=guest_rooms_node,
        critical=True,
    )
    claim_guest_rooms = f"The hotel has {item.guest_rooms_total or ''} total guest rooms."
    await evaluator.verify(
        claim=claim_guest_rooms,
        node=guest_rooms_supported,
        sources=item.sources_guest_rooms,
        additional_instruction="Verify the total number of guest rooms matches the source content.",
    )
    guest_rooms_val = parse_int_first_number(item.guest_rooms_total)
    guest_rooms_threshold = evaluator.add_custom_node(
        result=(guest_rooms_val is not None and guest_rooms_val >= 280),
        id="Guest_Rooms_At_Least_280",
        desc=f"Guest rooms total is at least 280 (parsed: {guest_rooms_val if guest_rooms_val is not None else 'None'}).",
        parent=guest_rooms_node,
        critical=True,
    )

    # Pet Friendly: accepts dogs
    pet_dogs_node = evaluator.add_parallel(
        id="Pet_Friendly_Accepts_Dogs",
        desc="Verifies the hotel is pet-friendly and accepts dogs.",
        parent=eval_parent,
        critical=True,
    )
    pet_dogs_provided = evaluator.add_custom_node(
        result=str_truthy(item.pet_friendly_dogs),
        id="Pet_Friendly_Dogs_Stated",
        desc="Pet-friendly / accepts dogs is stated.",
        parent=pet_dogs_node,
        critical=True,
    )
    pet_dogs_supported = evaluator.add_leaf(
        id="Pet_Friendly_Dogs_Supported",
        desc="Pet-friendly / accepts dogs is supported by sources.",
        parent=pet_dogs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The hotel accepts dogs.",
        node=pet_dogs_supported,
        sources=item.sources_pet_friendly_dogs,
        additional_instruction="Confirm the source indicates dogs are allowed (pet-friendly).",
    )

    # Pet Weight Limit Policy: provided + supported
    pet_weight_node = evaluator.add_parallel(
        id="Pet_Weight_Limit_Policy_Provided",
        desc="Specifies the pet weight-limit policy (maximum weight limit or no weight limit).",
        parent=eval_parent,
        critical=True,
    )
    pet_weight_provided = evaluator.add_custom_node(
        result=bool(item.pet_policy_weight_limit and item.pet_policy_weight_limit.strip()),
        id="Pet_Weight_Policy_Stated",
        desc="Pet weight-limit policy is stated.",
        parent=pet_weight_node,
        critical=True,
    )
    pet_weight_supported = evaluator.add_leaf(
        id="Pet_Weight_Policy_Supported",
        desc="Pet weight-limit policy is supported by sources.",
        parent=pet_weight_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The hotel's pet policy states: {item.pet_policy_weight_limit or ''}.",
        node=pet_weight_supported,
        sources=item.sources_pet_weight_limit_policy,
        additional_instruction="Confirm that the source includes the stated pet weight limit or explicitly indicates no weight limit.",
    )

    # Indoor Pool: stated + supported
    indoor_pool_node = evaluator.add_parallel(
        id="Indoor_Pool_Stated",
        desc="States whether the hotel has an indoor pool (yes/no).",
        parent=eval_parent,
        critical=True,
    )
    indoor_pool_norm = normalize_yes_no(item.indoor_pool)
    indoor_pool_provided = evaluator.add_custom_node(
        result=bool(indoor_pool_norm),
        id="Indoor_Pool_Provided",
        desc="Indoor pool status (yes/no) is stated.",
        parent=indoor_pool_node,
        critical=True,
    )
    indoor_pool_supported = evaluator.add_leaf(
        id="Indoor_Pool_Supported",
        desc="Indoor pool status is supported by sources.",
        parent=indoor_pool_node,
        critical=True,
    )
    if indoor_pool_norm == "yes":
        claim_pool = "The hotel has an indoor pool."
    elif indoor_pool_norm == "no":
        claim_pool = "The hotel does not have an indoor pool."
    else:
        claim_pool = "The hotel's indoor pool status is specified."
    await evaluator.verify(
        claim=claim_pool,
        node=indoor_pool_supported,
        sources=item.sources_indoor_pool,
        additional_instruction="Verify specifically 'indoor pool' versus generic 'pool'.",
    )

    # On-site Restaurants Count: provided + supported
    restaurants_node = evaluator.add_parallel(
        id="Onsite_Restaurants_Count_Provided",
        desc="Provides the number of on-site restaurants.",
        parent=eval_parent,
        critical=True,
    )
    restaurants_provided = evaluator.add_custom_node(
        result=bool(item.onsite_restaurants_count and item.onsite_restaurants_count.strip()),
        id="Restaurants_Count_Stated",
        desc="On-site restaurants count is stated.",
        parent=restaurants_node,
        critical=True,
    )
    restaurants_supported = evaluator.add_leaf(
        id="Restaurants_Count_Supported",
        desc="On-site restaurants count is supported by sources.",
        parent=restaurants_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The hotel has {item.onsite_restaurants_count or ''} on-site restaurants.",
        node=restaurants_supported,
        sources=item.sources_restaurants_count,
        additional_instruction="Confirm the number of on-site restaurants stated in the source matches the answer.",
    )

    # Reference URLs Provided (existence checks for each category)
    refs_parent = evaluator.add_parallel(
        id="Reference_URLs_Provided",
        desc="Provides reference URL(s) that support each required claim/information item and each selection constraint.",
        parent=eval_parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(item.sources_hotel_name) > 0,
        id="Citations_For_Hotel_Name",
        desc="Includes at least one reference URL supporting the hotel identity/name.",
        parent=refs_parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(item.sources_airport_served) > 0,
        id="Citations_For_Airport_Served",
        desc="Includes at least one reference URL supporting which airport the hotel serves.",
        parent=refs_parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(item.sources_major_airport_status) > 0,
        id="Citations_For_Major_Airport_Status",
        desc="Includes at least one reference URL supporting that the airport qualifies as a 'major' U.S. airport.",
        parent=refs_parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(item.sources_meeting_space) > 0,
        id="Citations_For_Meeting_Space",
        desc="Includes at least one reference URL supporting the stated meeting/event space square footage.",
        parent=refs_parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(item.sources_guest_rooms) > 0,
        id="Citations_For_Guest_Rooms",
        desc="Includes at least one reference URL supporting the stated total number of guest rooms.",
        parent=refs_parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(item.sources_pet_friendly_dogs) > 0,
        id="Citations_For_Pet_Friendly_Dogs",
        desc="Includes at least one reference URL supporting that the hotel accepts dogs.",
        parent=refs_parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(item.sources_pet_weight_limit_policy) > 0,
        id="Citations_For_Pet_Weight_Limit_Policy",
        desc="Includes at least one reference URL supporting the stated pet weight-limit policy.",
        parent=refs_parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(item.sources_indoor_pool) > 0,
        id="Citations_For_Indoor_Pool_Status",
        desc="Includes at least one reference URL supporting the stated indoor pool status (yes/no).",
        parent=refs_parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(item.sources_restaurants_count) > 0,
        id="Citations_For_Restaurants_Count",
        desc="Includes at least one reference URL supporting the stated number of on-site restaurants.",
        parent=refs_parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(item.sources_terminal_access) > 0,
        id="Citations_For_Terminal_Access",
        desc="Includes at least one reference URL supporting the described terminal access method.",
        parent=refs_parent,
        critical=True,
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

    # Perform extractions (can be parallelized)
    count_task = evaluator.extract(
        prompt=prompt_extract_hotel_count(),
        template_class=HotelCountExtraction,
        extraction_name="hotel_count",
    )
    item_task = evaluator.extract(
        prompt=prompt_extract_hotel_item(),
        template_class=HotelItem,
        extraction_name="hotel_item",
    )
    count_info, item = await asyncio.gather(count_task, item_task)

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, count_info, item)

    # Return the evaluation summary
    return evaluator.get_summary()