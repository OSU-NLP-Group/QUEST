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
TASK_ID = "seattle_hotels_2026_trip"
TASK_DESCRIPTION = (
    "I'm planning a group trip to Seattle, Washington in 2026 for colleagues and their families. "
    "I need to find 4 different hotels in Seattle that each serve a different purpose:\n\n"
    "1. A hotel suitable for business travelers that provides high-speed Wi-Fi throughout the property and offers "
    "workspaces with desks and ergonomic seating (either in rooms or common areas). It should start serving breakfast "
    "no later than 7:00 AM.\n\n"
    "2. A family-friendly hotel that has a swimming pool or water play area, provides cribs or playpens for infants "
    "upon request, and offers family room configurations or suites.\n\n"
    "3. A pet-friendly hotel that explicitly allows dogs or cats, permits at least 2 pets per room, and specifies a "
    "pet weight limit or allows pets up to 80 pounds.\n\n"
    "4. An accessible hotel that offers ADA-compliant or wheelchair-accessible rooms with wheelchair-accessible "
    "bathrooms and provides additional accessibility features such as roll-in showers, grab bars, or accessible parking.\n\n"
    "All 4 hotels must allow check-in for guests who are 18 years or older. For each hotel, provide the hotel name, a "
    "brief description of how it meets the specified requirements, and a reference URL from the hotel's official "
    "website or a reputable booking platform."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelRef(BaseModel):
    # Core identity and source
    name: Optional[str] = None
    url: Optional[str] = None
    location_text: Optional[str] = None  # As stated in the answer, if any

    # Common/baseline
    checkin_age_desc: Optional[str] = None

    # Business-specific
    wifi_desc: Optional[str] = None
    workspace_desc: Optional[str] = None
    breakfast_start_time: Optional[str] = None  # e.g., "6:30 AM", "7 AM"

    # Family-specific
    pool_desc: Optional[str] = None
    cribs_desc: Optional[str] = None
    family_room_desc: Optional[str] = None

    # Pet-specific
    pets_allowed_desc: Optional[str] = None
    pet_count_desc: Optional[str] = None
    pet_weight_desc: Optional[str] = None

    # Accessible-specific
    accessible_room_desc: Optional[str] = None
    accessible_bathroom_desc: Optional[str] = None
    accessibility_features_desc: Optional[str] = None


class HotelsExtraction(BaseModel):
    business: Optional[HotelRef] = None
    family: Optional[HotelRef] = None
    pet: Optional[HotelRef] = None
    accessible: Optional[HotelRef] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    From the provided answer, extract structured information for exactly four distinct hotels in Seattle, Washington,
    each mapped to one category: business, family, pet, and accessible. If multiple candidates are provided for any
    category, extract only the first one mentioned for that category. If a field is not explicitly present in the answer,
    return null for that field. Do not infer or fabricate.

    For each category, extract an object with the following fields:
    - name: The hotel's name as written in the answer.
    - url: A single reference URL from the hotel's official site or a reputable booking platform mentioned in the answer.
           If no URL is given in the answer text, return null.
    - location_text: Any textual location information provided in the answer (e.g., "Seattle, WA" or neighborhood).
    
    Common/baseline:
    - checkin_age_desc: The stated minimum check-in age policy if mentioned (e.g., "Minimum age 18", "21+").

    Business-specific (business):
    - wifi_desc: Text from the answer that indicates high-speed or fast Wi-Fi and ideally coverage across the property.
    - workspace_desc: Text stating there are workspaces with desks and ergonomic seating (in-room or shared areas).
    - breakfast_start_time: The exact breakfast start time if specified (e.g., "6:30 AM", "7:00 AM").

    Family-specific (family):
    - pool_desc: Text indicating a pool or water play area.
    - cribs_desc: Text indicating cribs/pack-and-plays on request.
    - family_room_desc: Text indicating family rooms, suites, or multi-room configurations.

    Pet-specific (pet):
    - pets_allowed_desc: Text indicating pets (dogs/cats) are allowed.
    - pet_count_desc: Text indicating at least two pets per room are permitted (e.g., "up to 2 pets").
    - pet_weight_desc: Text indicating a pet weight limit or text such as "up to 80 pounds".

    Accessible-specific (accessible):
    - accessible_room_desc: Text indicating ADA-compliant or wheelchair-accessible rooms.
    - accessible_bathroom_desc: Text indicating wheelchair-accessible bathrooms in accessible rooms.
    - accessibility_features_desc: Text indicating features like roll-in showers, grab bars, or accessible parking.

    Return a JSON object with keys: business, family, pet, accessible; each maps to the object specified above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip()
    return (u.startswith("http://") or u.startswith("https://")) and "." in u


def norm_name(n: Optional[str]) -> Optional[str]:
    if not n:
        return None
    import re
    return re.sub(r"[^a-z0-9]+", "", n.lower())


def extract_names_for_uniqueness(extracted: HotelsExtraction) -> List[str]:
    names = []
    for item in [extracted.business, extracted.family, extracted.pet, extracted.accessible]:
        if item and item.name:
            names.append(item.name.strip())
        else:
            names.append("")  # placeholder to later detect missing names
    return names


# Generic instruction appended to all URL-based verifications
URL_EVIDENCE_REQUIRED = (
    "Base your judgment solely on the provided webpage content (text/screenshot). "
    "If the URL is missing, irrelevant, or the page does not state or clearly imply the claim, judge it as Incorrect."
)

# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def add_ref_url_check(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    url: Optional[str],
) -> Any:
    """Add a critical existence/validity check for the reference URL as a custom node."""
    return evaluator.add_custom_node(
        result=is_valid_url(url),
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=True
    )


async def verify_business_hotel(evaluator: Evaluator, root_node, hotel: Optional[HotelRef]) -> None:
    node = evaluator.add_parallel(
        id="Hotel_1_Business_Traveler",
        desc="Hotel suitable for business travelers with required amenities",
        parent=root_node,
        critical=False
    )

    name = (hotel.name if hotel and hotel.name else "the hotel").strip()
    url = hotel.url if hotel else None

    # Reference URL check (critical, used as prerequisite for all other checks)
    ref_node = await add_ref_url_check(
        evaluator, node, "BT_Reference_URL",
        "Valid reference URL provided for this hotel",
        url
    )

    # Location: Seattle, Washington
    bt_loc = evaluator.add_leaf(
        id="BT_Location",
        desc="Hotel is located in Seattle, Washington",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel named '{name}' is located in Seattle, Washington.",
        node=bt_loc,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED,
        extra_prerequisites=[ref_node]
    )

    # High-speed Wi-Fi throughout property
    bt_wifi = evaluator.add_leaf(
        id="BT_WiFi",
        desc="Hotel provides high-speed Wi-Fi throughout the property",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel provides high-speed Wi‑Fi available across the property, including guest rooms and public areas.",
        node=bt_wifi,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED + " Look for terms like 'high-speed', 'fast', or coverage in rooms and public areas.",
        extra_prerequisites=[ref_node]
    )

    # Workspace with desks and ergonomic seating
    bt_work = evaluator.add_leaf(
        id="BT_Workspace",
        desc="Hotel offers workspace with desk and ergonomic seating in rooms or common areas",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel offers workspaces with desks and ergonomic seating either in guest rooms or in shared/common areas (e.g., business center, coworking lounge).",
        node=bt_work,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED + " Accept in-room desks with ergonomic chairs or documented workspaces in common areas.",
        extra_prerequisites=[ref_node]
    )

    # Breakfast start time no later than 7:00 AM
    bt_bfast = evaluator.add_leaf(
        id="BT_Breakfast_Hours",
        desc="Hotel provides breakfast service starting no later than 7:00 AM",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Breakfast service at this hotel starts at or before 7:00 AM.",
        node=bt_bfast,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED + " Verify explicit breakfast hours and ensure the start time is 7:00 AM or earlier.",
        extra_prerequisites=[ref_node]
    )

    # Check-in age 18+
    bt_age = evaluator.add_leaf(
        id="BT_CheckIn_Age",
        desc="Hotel allows check-in for guests 18 years or older",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel's minimum check-in age is 18 years or older.",
        node=bt_age,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED + " If the page states 21+ or does not specify the minimum age, judge Incorrect.",
        extra_prerequisites=[ref_node]
    )


async def verify_family_hotel(evaluator: Evaluator, root_node, hotel: Optional[HotelRef]) -> None:
    node = evaluator.add_parallel(
        id="Hotel_2_Family_Friendly",
        desc="Hotel suitable for families with children, providing family-oriented amenities",
        parent=root_node,
        critical=False
    )

    name = (hotel.name if hotel and hotel.name else "the hotel").strip()
    url = hotel.url if hotel else None

    ref_node = await add_ref_url_check(
        evaluator, node, "FF_Reference_URL",
        "Valid reference URL provided for this hotel",
        url
    )

    # Location
    ff_loc = evaluator.add_leaf(
        id="FF_Location",
        desc="Hotel is located in Seattle, Washington",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel named '{name}' is located in Seattle, Washington.",
        node=ff_loc,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED,
        extra_prerequisites=[ref_node]
    )

    # Pool / water play
    ff_pool = evaluator.add_leaf(
        id="FF_Pool",
        desc="Hotel has a swimming pool or water play area",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel has a swimming pool or a water play/splash area.",
        node=ff_pool,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED + " Accept indoor/outdoor pools or kids' splash areas.",
        extra_prerequisites=[ref_node]
    )

    # Cribs or playpens upon request
    ff_cribs = evaluator.add_leaf(
        id="FF_Cribs",
        desc="Hotel provides cribs or playpens for infants upon request",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel provides cribs, pack-and-plays, or playpens for infants upon request.",
        node=ff_cribs,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED + " Look for 'cribs available', 'infant beds', or 'pack and play' verbiage.",
        extra_prerequisites=[ref_node]
    )

    # Family rooms / suites
    ff_rooms = evaluator.add_leaf(
        id="FF_Family_Rooms",
        desc="Hotel offers family room configurations or suites",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel offers family-oriented room configurations or suites (e.g., family rooms, multi-bedroom suites, or connecting rooms).",
        node=ff_rooms,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED,
        extra_prerequisites=[ref_node]
    )

    # Check-in age 18+
    ff_age = evaluator.add_leaf(
        id="FF_CheckIn_Age",
        desc="Hotel allows check-in for guests 18 years or older",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel's minimum check-in age is 18 years or older.",
        node=ff_age,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED + " If the page states 21+ or does not specify the minimum age, judge Incorrect.",
        extra_prerequisites=[ref_node]
    )


async def verify_pet_hotel(evaluator: Evaluator, root_node, hotel: Optional[HotelRef]) -> None:
    node = evaluator.add_parallel(
        id="Hotel_3_Pet_Friendly",
        desc="Hotel that accepts pets with clear pet policy",
        parent=root_node,
        critical=False
    )

    name = (hotel.name if hotel and hotel.name else "the hotel").strip()
    url = hotel.url if hotel else None

    ref_node = await add_ref_url_check(
        evaluator, node, "PF_Reference_URL",
        "Valid reference URL provided for this hotel",
        url
    )

    # Location
    pf_loc = evaluator.add_leaf(
        id="PF_Location",
        desc="Hotel is located in Seattle, Washington",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel named '{name}' is located in Seattle, Washington.",
        node=pf_loc,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED,
        extra_prerequisites=[ref_node]
    )

    # Accepts dogs or cats
    pf_accepts = evaluator.add_leaf(
        id="PF_Accepts_Pets",
        desc="Hotel explicitly allows dogs or cats",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel explicitly allows dogs and/or cats.",
        node=pf_accepts,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED + " Look for 'pet-friendly', 'dogs allowed', or 'cats allowed' in policy details.",
        extra_prerequisites=[ref_node]
    )

    # At least 2 pets per room
    pf_count = evaluator.add_leaf(
        id="PF_Pet_Count",
        desc="Hotel allows at least 2 pets per room",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel permits at least two pets per room.",
        node=pf_count,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED + " Look for 'up to 2 pets per room' or similar wording.",
        extra_prerequisites=[ref_node]
    )

    # Weight limit or allows up to 80 pounds
    pf_weight = evaluator.add_leaf(
        id="PF_Weight_Limit",
        desc="Hotel specifies pet weight limit or allows pets up to 80 pounds",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel's pet policy either specifies a pet weight limit or explicitly allows pets up to 80 pounds.",
        node=pf_weight,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED + " Accept if any maximum weight is stated (e.g., 50/75/80 lbs) or if the policy clearly allows pets up to 80 lbs.",
        extra_prerequisites=[ref_node]
    )

    # Check-in age 18+
    pf_age = evaluator.add_leaf(
        id="PF_CheckIn_Age",
        desc="Hotel allows check-in for guests 18 years or older",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel's minimum check-in age is 18 years or older.",
        node=pf_age,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED + " If the page states 21+ or does not specify the minimum age, judge Incorrect.",
        extra_prerequisites=[ref_node]
    )


async def verify_accessible_hotel(evaluator: Evaluator, root_node, hotel: Optional[HotelRef]) -> None:
    node = evaluator.add_parallel(
        id="Hotel_4_Accessible",
        desc="Hotel with ADA-compliant accessible accommodations",
        parent=root_node,
        critical=False
    )

    name = (hotel.name if hotel and hotel.name else "the hotel").strip()
    url = hotel.url if hotel else None

    ref_node = await add_ref_url_check(
        evaluator, node, "AC_Reference_URL",
        "Valid reference URL provided for this hotel",
        url
    )

    # Location
    ac_loc = evaluator.add_leaf(
        id="AC_Location",
        desc="Hotel is located in Seattle, Washington",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel named '{name}' is located in Seattle, Washington.",
        node=ac_loc,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED,
        extra_prerequisites=[ref_node]
    )

    # Accessible rooms (ADA / wheelchair-accessible)
    ac_rooms = evaluator.add_leaf(
        id="AC_Accessible_Rooms",
        desc="Hotel offers ADA-compliant or wheelchair-accessible rooms",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel offers ADA-compliant or wheelchair-accessible guest rooms.",
        node=ac_rooms,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED + " Look for 'ADA compliant', 'accessible rooms', or 'wheelchair accessible rooms'.",
        extra_prerequisites=[ref_node]
    )

    # Accessible bathrooms in accessible rooms
    ac_bath = evaluator.add_leaf(
        id="AC_Accessible_Bathrooms",
        desc="Accessible rooms include wheelchair-accessible bathrooms",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel's accessible rooms include wheelchair-accessible bathrooms.",
        node=ac_bath,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED + " Accept terms such as 'accessible bathrooms', 'wheelchair-accessible bathroom', etc.",
        extra_prerequisites=[ref_node]
    )

    # Additional accessibility features: roll-in showers, grab bars, accessible parking
    ac_features = evaluator.add_leaf(
        id="AC_Accessible_Features",
        desc="Hotel provides additional accessibility features (e.g., roll-in showers, grab bars, accessible parking)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel provides additional accessibility features such as roll-in showers, grab bars, or accessible parking.",
        node=ac_features,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED + " At least one of the listed features should be explicitly mentioned.",
        extra_prerequisites=[ref_node]
    )

    # Check-in age 18+
    ac_age = evaluator.add_leaf(
        id="AC_CheckIn_Age",
        desc="Hotel allows check-in for guests 18 years or older",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel's minimum check-in age is 18 years or older.",
        node=ac_age,
        sources=url,
        additional_instruction=URL_EVIDENCE_REQUIRED + " If the page states 21+ or does not specify the minimum age, judge Incorrect.",
        extra_prerequisites=[ref_node]
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
    Evaluate an answer for the Seattle hotels task using the Mind2Web2 framework.
    """
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
        default_model=model
    )

    # IMPORTANT: Root should be non-critical to allow partial credit across hotels
    root.critical = False

    # 1) Extract structured hotel information from the answer
    extracted: HotelsExtraction = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_structured"
    )

    # 2) Add uniqueness check (critical under root)
    names = extract_names_for_uniqueness(extracted)
    normed = [norm_name(n) for n in names if n]
    uniqueness_ok = (len(normed) == 4) and (len(set(normed)) == 4)
    evaluator.add_custom_node(
        result=uniqueness_ok,
        id="Hotels_Are_Unique",
        desc="All 4 hotels provided are different properties (not the same hotel used for multiple categories)",
        parent=root,
        critical=True
    )

    # Optionally, record extracted names for debugging
    evaluator.add_custom_info(
        info={"business": names[0], "family": names[1], "pet": names[2], "accessible": names[3]},
        info_type="extracted_hotels",
        info_name="extracted_hotels_summary"
    )

    # 3) Build verification subtrees for each category
    await verify_business_hotel(evaluator, root, extracted.business)
    await verify_family_hotel(evaluator, root, extracted.family)
    await verify_pet_hotel(evaluator, root, extracted.pet)
    await verify_accessible_hotel(evaluator, root, extracted.accessible)

    # 4) Return evaluator summary
    return evaluator.get_summary()