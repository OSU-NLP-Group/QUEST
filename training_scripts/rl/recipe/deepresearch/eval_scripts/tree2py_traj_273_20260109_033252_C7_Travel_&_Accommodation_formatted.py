import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "la_5star_accessible_hotel"
TASK_DESCRIPTION = (
    "I am planning to stay at a five-star hotel in Los Angeles, California for an upcoming trip. I am a wheelchair user traveling with my service dog, and I need to ensure the hotel meets all necessary requirements. "
    "Please identify a hotel that satisfies ALL of the following criteria: "
    "(1) The hotel must have a five-star rating; "
    "(2) The hotel must be located in Los Angeles, California; "
    "(3) The hotel must provide 24-hour reception service; "
    "(4) The hotel must offer valet parking service; "
    "(5) The hotel must have an on-site fitness center or gym; "
    "(6) The hotel must have at least one on-site restaurant or dining facility; "
    "(7) The hotel must allow service animals with no additional fees (per ADA requirements); "
    "(8) The hotel must have accessible rooms with a minimum door width of 32 inches; "
    "(9) The hotel must have accessible rooms equipped with grab bars in the bathroom; "
    "(10) The hotel must have an automatic fire sprinkler system in guest rooms; "
    "(11) The hotel must post emergency evacuation plans in each guest room; "
    "(12) The hotel's standard check-in time must be between 3:00 PM and 4:00 PM; "
    "(13) The hotel must offer a free cancellation policy allowing cancellation at least 24 hours before check-in; "
    "(14) The hotel must allow guests aged 21 or younger to check in. "
    "What is the name of a hotel that meets all these requirements?"
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class HotelExtraction(BaseModel):
    """
    Extract a single hotel candidate (prefer the first one if multiple are provided)
    along with categorized URLs explicitly mentioned in the answer.
    """
    hotel_name: Optional[str] = None

    # A single official site URL if present
    official_url: Optional[str] = None

    # General URLs and category-specific URLs (only URLs explicitly present in the answer)
    general_urls: List[str] = Field(default_factory=list)
    rating_urls: List[str] = Field(default_factory=list)
    amenities_urls: List[str] = Field(default_factory=list)               # e.g., fitness, restaurant, 24h front desk, valet
    accessibility_urls: List[str] = Field(default_factory=list)          # e.g., ADA accessibility statement, room accessibility details
    pet_policy_urls: List[str] = Field(default_factory=list)             # e.g., pet/service animal policy pages
    dining_urls: List[str] = Field(default_factory=list)                 # e.g., on-site restaurant pages
    parking_urls: List[str] = Field(default_factory=list)                # e.g., valet/parking details
    checkin_policy_urls: List[str] = Field(default_factory=list)         # e.g., hotel policies page listing check-in time and age
    cancellation_policy_urls: List[str] = Field(default_factory=list)    # e.g., cancellation policy or rate T&C
    safety_urls: List[str] = Field(default_factory=list)                 # e.g., safety features, fire sprinklers, evacuation plans

    # Optional textual data explicitly stated in the answer (if any)
    location_text: Optional[str] = None
    checkin_time_text: Optional[str] = None
    cancellation_policy_text: Optional[str] = None
    minimum_checkin_age_text: Optional[str] = None
    service_animals_policy_text: Optional[str] = None
    door_width_text: Optional[str] = None
    grab_bars_text: Optional[str] = None
    fire_sprinkler_text: Optional[str] = None
    evacuation_plan_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_candidate() -> str:
    return """
    From the answer text, extract details for exactly one specific hotel candidate that the answer claims satisfies the requirements. If the answer lists multiple hotels, extract the first one only.

    Required fields:
    - hotel_name: The name of the hotel property (as stated in the answer).
    - official_url: A single official hotel website URL if explicitly given (e.g., brand or property's official site). If multiple, pick the most direct property URL.
    - general_urls: A list of any other URLs mentioned that reference the hotel or its info.
    - rating_urls: URLs that support the hotel's five-star rating (e.g., Forbes Travel Guide, AAA, other credible rating pages). Only include if explicitly provided.
    - amenities_urls: URLs that indicate amenities such as 24-hour front desk, valet parking, gym/fitness, etc.
    - accessibility_urls: URLs with accessibility details, ADA page, accessible room features (door width, grab bars, etc.).
    - pet_policy_urls: URLs that state pet policies and explicitly mention service animals or ADA compliance.
    - dining_urls: URLs for on-site restaurant(s) or dining facilities.
    - parking_urls: URLs describing valet parking or parking services.
    - checkin_policy_urls: URLs that state check-in time and minimum check-in age.
    - cancellation_policy_urls: URLs that state cancellation policy terms (e.g., free cancellation window).
    - safety_urls: URLs indicating safety features like automatic fire sprinklers and emergency evacuation plan posting.

    Optional text fields (only if explicitly stated in the answer; otherwise null):
    - location_text
    - checkin_time_text
    - cancellation_policy_text
    - minimum_checkin_age_text
    - service_animals_policy_text
    - door_width_text
    - grab_bars_text
    - fire_sprinkler_text
    - evacuation_plan_text

    Rules:
    - Extract only URLs explicitly present in the answer text (including markdown links).
    - If a field is missing, set it to null (or an empty list where appropriate).
    - Do not invent or infer anything not stated in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions for sources                                                #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def get_all_sources(info: HotelExtraction) -> List[str]:
    urls: List[str] = []
    # Prioritize official URL first if present
    if info.official_url:
        urls.append(info.official_url)

    # Then add all category lists
    urls.extend(info.rating_urls)
    urls.extend(info.amenities_urls)
    urls.extend(info.accessibility_urls)
    urls.extend(info.pet_policy_urls)
    urls.extend(info.dining_urls)
    urls.extend(info.parking_urls)
    urls.extend(info.checkin_policy_urls)
    urls.extend(info.cancellation_policy_urls)
    urls.extend(info.safety_urls)
    urls.extend(info.general_urls)
    return _dedup_urls(urls)


def preferred_sources(info: HotelExtraction, prioritized_categories: List[str]) -> List[str]:
    """
    Build a sources list starting with prioritized category lists, then include all other known URLs.
    prioritized_categories: list of attribute names on HotelExtraction (e.g., ["rating_urls", "amenities_urls"])
    """
    urls: List[str] = []
    # Start with prioritized categories
    for cat in prioritized_categories:
        if hasattr(info, cat):
            urls.extend(getattr(info, cat) or [])
    # Include official URL early
    if info.official_url:
        urls.append(info.official_url)
    # Finish with the whole set to ensure we don't miss any link
    urls.extend(get_all_sources(info))
    return _dedup_urls(urls)


def hotel_label(info: HotelExtraction) -> str:
    return f"'{info.hotel_name}'" if (info and info.hotel_name) else "the hotel"


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_and_verify_hotel_requirements(
    evaluator: Evaluator,
    root_node,
    info: HotelExtraction,
) -> None:
    """
    Construct the rubric verification tree and run the checks for the hotel.
    All requirement children are critical; failing any results in overall failure.
    """
    # Create the rubric root (critical, parallel) under the global root
    hotel_req_node = evaluator.add_parallel(
        id="Hotel_Requirements",
        desc="Root node evaluating whether the identified hotel meets all specified requirements for a wheelchair user with a service dog",
        parent=root_node,
        critical=True,
    )

    # Gate: Ensure a hotel is identified and at least one source URL exists
    all_urls = get_all_sources(info)
    has_hotel_and_sources = (info.hotel_name is not None and str(info.hotel_name).strip() != "") and (len(all_urls) > 0)
    evaluator.add_custom_node(
        result=has_hotel_and_sources,
        id="Hotel_Identified",
        desc="A specific hotel is identified and at least one source URL is provided in the answer",
        parent=hotel_req_node,
        critical=True,
    )

    # 1) Five-star rating
    node_5star = evaluator.add_leaf(
        id="Five_Star_Rating",
        desc="The hotel must have a verified five-star rating",
        parent=hotel_req_node,
        critical=True,
    )
    claim_5star = f"The hotel {hotel_label(info)} has a verified five-star rating (e.g., Forbes 5-Star, AAA Five Diamond, or clearly stated 5-star rating on a credible page)."
    await evaluator.verify(
        claim=claim_5star,
        node=node_5star,
        sources=preferred_sources(info, ["rating_urls"]),
        additional_instruction=(
            "Verify that the provided page(s) clearly state a 5-star rating or a recognized equivalent (e.g., Forbes 5-Star or AAA Five Diamond). "
            "Marketing language alone without a formal rating should not count. The support must be explicit."
        ),
    )

    # 2) Location in Los Angeles, California
    node_loc = evaluator.add_leaf(
        id="Location_Los_Angeles",
        desc="The hotel must be located in Los Angeles, California",
        parent=hotel_req_node,
        critical=True,
    )
    claim_loc = f"The hotel {hotel_label(info)} is located within the City of Los Angeles, California."
    await evaluator.verify(
        claim=claim_loc,
        node=node_loc,
        sources=preferred_sources(info, ["general_urls"]),
        additional_instruction=(
            "Confirm that the hotel's address is in 'Los Angeles, CA' (the city), not just Los Angeles County. "
            "Cities like Beverly Hills, Santa Monica, and West Hollywood are separate and should not be accepted as Los Angeles. "
            "Allow neighborhood names (e.g., Hollywood, Downtown LA) if clearly within the City of Los Angeles."
        ),
    )

    # 3) 24-hour reception/front desk
    node_24h = evaluator.add_leaf(
        id="24_Hour_Reception",
        desc="The hotel must provide 24-hour reception or front desk service",
        parent=hotel_req_node,
        critical=True,
    )
    claim_24h = f"The hotel {hotel_label(info)} provides a 24-hour front desk or reception."
    await evaluator.verify(
        claim=claim_24h,
        node=node_24h,
        sources=preferred_sources(info, ["amenities_urls", "checkin_policy_urls"]),
        additional_instruction=(
            "Look for wording like '24-hour front desk', '24-hour reception', or equivalent. "
            "If a source explicitly states 24-hour service at the front desk, it satisfies this requirement."
        ),
    )

    # 4) Valet parking
    node_valet = evaluator.add_leaf(
        id="Valet_Parking_Service",
        desc="The hotel must offer valet parking service",
        parent=hotel_req_node,
        critical=True,
    )
    claim_valet = f"The hotel {hotel_label(info)} offers valet parking service."
    await evaluator.verify(
        claim=claim_valet,
        node=node_valet,
        sources=preferred_sources(info, ["parking_urls", "amenities_urls"]),
        additional_instruction="Confirm the page clearly mentions 'valet parking' on-site (not just nearby).",
    )

    # 5) Fitness center
    node_gym = evaluator.add_leaf(
        id="Fitness_Center",
        desc="The hotel must have an on-site fitness center or gym facility",
        parent=hotel_req_node,
        critical=True,
    )
    claim_gym = f"The hotel {hotel_label(info)} has an on-site fitness center or gym."
    await evaluator.verify(
        claim=claim_gym,
        node=node_gym,
        sources=preferred_sources(info, ["amenities_urls"]),
        additional_instruction="Look for 'fitness center', 'gym', or similar wording indicating on-site availability.",
    )

    # 6) On-site restaurant
    node_rest = evaluator.add_leaf(
        id="On_Site_Restaurant",
        desc="The hotel must have at least one on-site restaurant or dining facility",
        parent=hotel_req_node,
        critical=True,
    )
    claim_rest = f"The hotel {hotel_label(info)} has at least one on-site restaurant or dining facility."
    await evaluator.verify(
        claim=claim_rest,
        node=node_rest,
        sources=preferred_sources(info, ["dining_urls", "amenities_urls"]),
        additional_instruction="Confirm there is at least one restaurant or dining venue located on the hotel premises.",
    )

    # 7) Service animals allowed with no fees
    node_service_animals = evaluator.add_leaf(
        id="Service_Animals_Policy",
        desc="The hotel must allow service animals with no additional fees in compliance with ADA requirements",
        parent=hotel_req_node,
        critical=True,
    )
    claim_service_animals = f"Service animals are allowed at {hotel_label(info)} with no additional fees or deposits (ADA compliant)."
    await evaluator.verify(
        claim=claim_service_animals,
        node=node_service_animals,
        sources=preferred_sources(info, ["pet_policy_urls", "accessibility_urls"]),
        additional_instruction=(
            "Accept statements such as 'service animals welcome at no charge', 'service animals are exempt from pet fees', "
            "or 'ADA service animals allowed without fee'. Generic 'pets allowed' with fees does NOT satisfy this unless it clearly exempts service animals."
        ),
    )

    # 8) Accessible door width min 32 inches
    node_door = evaluator.add_leaf(
        id="Accessible_Door_Width",
        desc="The hotel's accessible rooms must have doors with a minimum clear width of 32 inches",
        parent=hotel_req_node,
        critical=True,
    )
    claim_door = f"Accessible guest room doors at {hotel_label(info)} have a clear width of at least 32 inches."
    await evaluator.verify(
        claim=claim_door,
        node=node_door,
        sources=preferred_sources(info, ["accessibility_urls", "safety_urls"]),
        additional_instruction=(
            "Look for explicit door width measurements like '32 inches' (also acceptable: 32 in, 32\", or 813 mm approx). "
            "If the page does not state door width explicitly, do not assume; the requirement is not satisfied."
        ),
    )

    # 9) Bathroom grab bars
    node_grab = evaluator.add_leaf(
        id="Bathroom_Grab_Bars",
        desc="The hotel's accessible rooms must be equipped with grab bars in the bathroom",
        parent=hotel_req_node,
        critical=True,
    )
    claim_grab = f"Accessible guest rooms at {hotel_label(info)} include grab bars in the bathroom."
    await evaluator.verify(
        claim=claim_grab,
        node=node_grab,
        sources=preferred_sources(info, ["accessibility_urls"]),
        additional_instruction="Accept 'grab bars', 'handrails', or explicit ADA bathroom features indicating grab bars in tub/shower/toilet areas.",
    )

    # 10) Fire sprinklers in guest rooms
    node_sprinkler = evaluator.add_leaf(
        id="Fire_Sprinkler_System",
        desc="The hotel must have an automatic fire sprinkler system installed in guest rooms",
        parent=hotel_req_node,
        critical=True,
    )
    claim_sprinkler = f"Guest rooms at {hotel_label(info)} are equipped with an automatic fire sprinkler system."
    await evaluator.verify(
        claim=claim_sprinkler,
        node=node_sprinkler,
        sources=preferred_sources(info, ["safety_urls", "accessibility_urls"]),
        additional_instruction=(
            "Look for explicit mentions of 'automatic fire sprinklers' or 'sprinkler systems' in guest rooms. "
            "General references to alarms or smoke detectors alone are not sufficient."
        ),
    )

    # 11) Emergency evacuation plans posted in each room
    node_evac = evaluator.add_leaf(
        id="Emergency_Evacuation_Plans",
        desc="The hotel must post emergency evacuation plans in each guest room",
        parent=hotel_req_node,
        critical=True,
    )
    claim_evac = f"Emergency evacuation plans are posted in each guest room at {hotel_label(info)}."
    await evaluator.verify(
        claim=claim_evac,
        node=node_evac,
        sources=preferred_sources(info, ["safety_urls"]),
        additional_instruction=(
            "Look for statements such as 'evacuation plan/map posted on the back of each guest room door' or similar. "
            "If the presence of evacuation plans in each guest room is not explicit, do not assume."
        ),
    )

    # 12) Standard check-in time between 3:00 PM and 4:00 PM
    node_checkin_time = evaluator.add_leaf(
        id="Check_In_Time",
        desc="The hotel's standard check-in time must be between 3:00 PM and 4:00 PM",
        parent=hotel_req_node,
        critical=True,
    )
    claim_checkin_time = f"The standard check-in time at {hotel_label(info)} is between 3:00 PM and 4:00 PM (inclusive)."
    await evaluator.verify(
        claim=claim_checkin_time,
        node=node_checkin_time,
        sources=preferred_sources(info, ["checkin_policy_urls"]),
        additional_instruction=(
            "Accept explicit check-in times such as '3:00 PM', '3 PM', or '4 PM'. "
            "A time range including 3 PM or 4 PM also counts, as long as the standard check-in is within 3–4 PM inclusive."
        ),
    )

    # 13) Free cancellation at least 24 hours before check-in
    node_cancel = evaluator.add_leaf(
        id="Cancellation_Policy",
        desc="The hotel must offer a free cancellation policy allowing cancellation at least 24 hours before check-in",
        parent=hotel_req_node,
        critical=True,
    )
    claim_cancel = f"Guests can cancel for free at least 24 hours before check-in at {hotel_label(info)}."
    await evaluator.verify(
        claim=claim_cancel,
        node=node_cancel,
        sources=preferred_sources(info, ["cancellation_policy_urls", "checkin_policy_urls"]),
        additional_instruction=(
            "Look for wording like 'free cancellation up to 24 hours before arrival/check-in' or 'by 4 PM the day before arrival'. "
            "Note: If the policy is rate-dependent and the cited source clearly specifies a free 24-hour cancellation option, that is acceptable."
        ),
    )

    # 14) Minimum check-in age allows 21 or younger
    node_age = evaluator.add_leaf(
        id="Minimum_Check_In_Age",
        desc="The hotel must allow guests aged 21 or younger to check in",
        parent=hotel_req_node,
        critical=True,
    )
    claim_age = f"The minimum check-in age at {hotel_label(info)} is 21 or lower (<= 21)."
    await evaluator.verify(
        claim=claim_age,
        node=node_age,
        sources=preferred_sources(info, ["checkin_policy_urls"]),
        additional_instruction=(
            "Pass if the policy states minimum age is 21, 20, 19, or 18 (i.e., <= 21). "
            "Fail if it requires 22+ or 25+, or if the page states a higher minimum age requirement."
        ),
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
    Evaluate an answer for the LA five-star accessible hotel task using the Mind2Web2 evaluator.
    """
    # Initialize evaluator (root is non-critical by framework design)
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

    # Extract a single hotel candidate with categorized URLs
    hotel_info = await evaluator.extract(
        prompt=prompt_extract_hotel_candidate(),
        template_class=HotelExtraction,
        extraction_name="hotel_candidate",
    )

    # Build the verification tree and run checks
    await build_and_verify_hotel_requirements(evaluator, root, hotel_info)

    # Return final structured summary
    return evaluator.get_summary()