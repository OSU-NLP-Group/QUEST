import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "fort_lauderdale_pet_cruise_hotel_2025"
TASK_DESCRIPTION = (
    "I am planning to take the Disney Destiny cruise from Fort Lauderdale in 2025 and need to book a hotel for the night before my cruise departs. "
    "I am flying into Fort Lauderdale-Hollywood International Airport (FLL) and will be traveling with my 80-pound dog.\n\n"
    "Please identify a hotel in Fort Lauderdale, Florida that meets ALL of the following requirements:\n\n"
    "1. The hotel must be located in Fort Lauderdale, Florida\n"
    "2. The hotel must accept dogs as pets\n"
    "3. The hotel must allow dogs weighing at least 75 pounds\n"
    "4. The pet fee must not exceed $50 per night\n"
    "5. The hotel must offer a park-stay-cruise or park-and-cruise package that includes parking for the duration of a cruise\n"
    "6. The parking package must allow at least 7 consecutive days of parking\n"
    "7. The hotel must provide or arrange shuttle transportation to Port Everglades cruise terminal\n"
    "8. The hotel must be within 5 miles of Fort Lauderdale-Hollywood International Airport (FLL)\n"
    "9. The hotel must be within 5 miles of Port Everglades\n\n"
    "For your answer, please provide:\n"
    "- The complete hotel name\n"
    "- The hotel chain or brand (if applicable)\n"
    "- The hotel's standard check-in time\n"
    "- The hotel's contact information (phone number or website)\n"
    "- A reference URL from the hotel's official website or a major booking platform that confirms these requirements"
)

# ---------------------------
# Data Models for Extraction
# ---------------------------
class HotelCoreInfo(BaseModel):
    hotel_name: Optional[str] = None
    brand: Optional[str] = None
    check_in_time: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_website: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class PetPolicyInfo(BaseModel):
    accepts_dogs: Optional[str] = None  # e.g., "pets allowed", "dogs permitted"
    max_dog_weight_allowed: Optional[str] = None  # e.g., "No weight limit", "up to 80 lbs"
    pet_fee_per_night: Optional[str] = None  # e.g., "$25 per night", "free", "per stay $40"


class CruisePackageInfo(BaseModel):
    package_name: Optional[str] = None  # e.g., "Park and Cruise", "Stay, Park & Cruise"
    includes_parking_for_cruise_duration: Optional[str] = None  # e.g., "Parking included for duration of cruise"
    parking_days_allowed: Optional[str] = None  # e.g., "7 days", "10 nights"
    shuttle_to_port_everglades: Optional[str] = None  # e.g., "Shuttle to Port Everglades available"


class LocationInfo(BaseModel):
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    distance_to_fll_miles: Optional[str] = None  # e.g., "3.2 miles"
    distance_to_port_everglades_miles: Optional[str] = None  # e.g., "4.5 miles"


class HotelExtraction(BaseModel):
    hotel_core: Optional[HotelCoreInfo] = None
    pet_policy: Optional[PetPolicyInfo] = None
    cruise_package: Optional[CruisePackageInfo] = None
    location: Optional[LocationInfo] = None


# ---------------------------
# Extraction Prompt
# ---------------------------
def prompt_extract_hotel_info() -> str:
    return (
        "Extract structured information about the single hotel proposed in the answer. "
        "Return the following fields (use strings; if not explicitly mentioned, use null):\n\n"
        "hotel_core:\n"
        "- hotel_name: The complete hotel name exactly as stated.\n"
        "- brand: The hotel chain or brand if explicitly stated (e.g., Marriott, Hilton). If not present, null.\n"
        "- check_in_time: The standard check-in time if mentioned (e.g., '3:00 PM').\n"
        "- contact_phone: A phone number if provided; else null.\n"
        "- contact_website: The hotel's official website URL if provided; else null.\n"
        "- reference_urls: All URLs from the hotel's official site or major booking platforms (e.g., marriott.com, hilton.com, hyatt.com, ihg.com, booking.com, expedia.com, hotels.com, tripadvisor.com) that the answer cites to support constraints. Extract only URLs explicitly present in the answer.\n\n"
        "pet_policy:\n"
        "- accepts_dogs: Text stating whether dogs are accepted.\n"
        "- max_dog_weight_allowed: Text about dog weight limit (e.g., 'no weight limit', 'up to 80 lbs').\n"
        "- pet_fee_per_night: The pet fee wording as stated (e.g., '$25 per night', 'per stay $40', 'free').\n\n"
        "cruise_package:\n"
        "- package_name: Name or label of a park-stay-cruise / park-and-cruise (or similar) package.\n"
        "- includes_parking_for_cruise_duration: Text indicating parking included for cruise duration.\n"
        "- parking_days_allowed: The number of consecutive parking days included (text as given).\n"
        "- shuttle_to_port_everglades: Text indicating shuttle availability/arrangement to Port Everglades.\n\n"
        "location:\n"
        "- address: Street address if provided.\n"
        "- city: City name if provided.\n"
        "- state: State abbreviation/name if provided.\n"
        "- distance_to_fll_miles: Distance to FLL in miles if stated (text as given).\n"
        "- distance_to_port_everglades_miles: Distance to Port Everglades in miles if stated (text as given).\n\n"
        "Important:\n"
        "- Extract only what is explicitly present in the answer text.\n"
        "- For URLs, include full URLs as presented (including protocol if available). If missing protocol, prepend 'http://'.\n"
        "- If any field is missing, return null. For lists, return empty list.\n"
    )


# ---------------------------
# Helper Functions
# ---------------------------
MAJOR_BOOKING_DOMAINS = {
    "marriott.com", "hilton.com", "hyatt.com", "ihg.com", "choicehotels.com", "bestwestern.com", "wyndhamhotels.com",
    "radissonhotels.com", "sonesta.com", "omnihotels.com", "accor.com",
    "booking.com", "expedia.com", "hotels.com", "priceline.com", "agoda.com", "tripadvisor.com", "kayak.com"
}

KNOWN_BRAND_TOKENS = {
    "Marriott", "Courtyard", "Residence Inn", "SpringHill Suites", "Fairfield", "TownePlace Suites", "Aloft",
    "Four Points", "Sheraton", "Westin",
    "Hilton", "DoubleTree", "Hampton", "Embassy Suites", "Homewood Suites", "Home2 Suites", "Tru",
    "Hyatt", "Hyatt Place", "Hyatt House",
    "IHG", "Holiday Inn", "Holiday Inn Express", "Crowne Plaza", "Staybridge Suites", "Candlewood Suites",
    "Best Western", "Wyndham", "La Quinta", "Days Inn", "Super 8", "Comfort", "Quality Inn", "Sleep Inn", "Cambria",
    "Radisson", "Sonesta"
}


def extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        # strip subdomains
        parts = host.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return host
    except Exception:
        return ""


def is_major_or_official_url(url: str, brand: Optional[str], hotel_name: Optional[str]) -> bool:
    domain = extract_domain(url)
    if domain in MAJOR_BOOKING_DOMAINS:
        return True
    # Simple heuristic: if brand token appears in domain (e.g., marriott.com), consider official
    lower = domain.lower()
    tokens = []
    if brand:
        tokens.append(brand.lower())
    if hotel_name:
        # add common chain tokens from hotel name
        for token in KNOWN_BRAND_TOKENS:
            if token.lower() in hotel_name.lower():
                tokens.append(token.lower())
    for t in tokens:
        if t and t in lower:
            return True
    return False


def has_any_valid_reference_url(refs: List[str], brand: Optional[str], hotel_name: Optional[str]) -> bool:
    valid_refs = [u for u in refs if isinstance(u, str) and len(u.strip()) > 0]
    if not valid_refs:
        return False
    for u in valid_refs:
        if is_major_or_official_url(u, brand, hotel_name):
            return True
    return False


def infer_brand_from_name(hotel_name: Optional[str]) -> bool:
    if not hotel_name:
        return False
    name_lower = hotel_name.lower()
    for token in KNOWN_BRAND_TOKENS:
        if token.lower() in name_lower:
            return True
    return False


def pick_reference_sources(core: Optional[HotelCoreInfo], limit: int = 6) -> List[str]:
    sources = []
    if core:
        # reference_urls from the answer
        sources.extend(core.reference_urls or [])
        # include contact website if provided
        if core.contact_website and core.contact_website.strip():
            sources.append(core.contact_website.strip())
    # Deduplicate while preserving order
    seen = set()
    result = []
    for u in sources:
        if u and u not in seen:
            result.append(u)
            seen.add(u)
        if len(result) >= limit:
            break
    return result


# ---------------------------
# Verification Builder
# ---------------------------
async def build_and_verify_constraints(
    evaluator: Evaluator,
    parent_node,
    extraction: HotelExtraction,
    reference_url_leaf: Optional[Any],  # Leaf node to use as prerequisite (reference_url_provided)
) -> List[Any]:
    core = extraction.hotel_core or HotelCoreInfo()
    cruise = extraction.cruise_package or CruisePackageInfo()
    pet = extraction.pet_policy or PetPolicyInfo()
    loc = extraction.location or LocationInfo()

    hotel_name = core.hotel_name or "the hotel"
    sources = pick_reference_sources(core, limit=6)

    constraints_parent = evaluator.add_parallel(
        id="hotel_meets_all_constraints",
        desc="The identified hotel satisfies all stated constraints (pet, location, parking package, shuttle).",
        parent=parent_node,
        critical=True
    )

    # Create all constraint leaves
    node_loc_ftl = evaluator.add_leaf(
        id="located_in_fort_lauderdale",
        desc="Hotel is located in Fort Lauderdale, Florida.",
        parent=constraints_parent,
        critical=True
    )
    node_accepts_dogs = evaluator.add_leaf(
        id="accepts_dogs",
        desc="Hotel accepts dogs as pets.",
        parent=constraints_parent,
        critical=True
    )
    node_weight = evaluator.add_leaf(
        id="allows_dogs_75lbs_or_more",
        desc="Hotel allows dogs weighing at least 75 pounds.",
        parent=constraints_parent,
        critical=True
    )
    node_pet_fee = evaluator.add_leaf(
        id="pet_fee_max_50_per_night",
        desc="Pet fee does not exceed $50 per night.",
        parent=constraints_parent,
        critical=True
    )
    node_pkg_parking = evaluator.add_leaf(
        id="park_and_cruise_package_includes_cruise_duration_parking",
        desc="Hotel offers a park-stay-cruise/park-and-cruise (or similar) package that includes parking for the duration of the cruise.",
        parent=constraints_parent,
        critical=True
    )
    node_parking_7 = evaluator.add_leaf(
        id="parking_allows_at_least_7_days",
        desc="The parking package allows at least 7 consecutive days of parking.",
        parent=constraints_parent,
        critical=True
    )
    node_shuttle = evaluator.add_leaf(
        id="shuttle_to_port_everglades",
        desc="Hotel provides or arranges shuttle transportation to Port Everglades cruise terminal.",
        parent=constraints_parent,
        critical=True
    )
    node_within_fll = evaluator.add_leaf(
        id="within_5_miles_of_fll",
        desc="Hotel is within 5 miles of Fort Lauderdale-Hollywood International Airport (FLL).",
        parent=constraints_parent,
        critical=True
    )
    node_within_port = evaluator.add_leaf(
        id="within_5_miles_of_port_everglades",
        desc="Hotel is within 5 miles of Port Everglades.",
        parent=constraints_parent,
        critical=True
    )

    # Prepare claims and additional instructions for batch verification
    claims_and_sources = [
        (
            f"The hotel '{hotel_name}' is located in Fort Lauderdale, Florida.",
            sources,
            node_loc_ftl,
            "Check the page for the hotel's city/state or address indicating Fort Lauderdale, FL (allow reasonable variants). If no valid reference URL is provided, treat as unsupported."
        ),
        (
            f"The hotel '{hotel_name}' accepts dogs (pet friendly).",
            sources,
            node_accepts_dogs,
            "Verify pet policy text indicating dogs are allowed (e.g., 'pets allowed', 'dog-friendly'). Service animals alone are not sufficient."
        ),
        (
            f"The hotel '{hotel_name}' allows dogs weighing at least 75 pounds (no weight limit or explicit limit >= 75 lbs).",
            sources,
            node_weight,
            "Accept wording such as 'no weight limit', 'large dogs allowed', or explicit limits like 'up to 80 lbs'. If weight limit < 75 lbs, fail."
        ),
        (
            f"The pet fee at '{hotel_name}' does not exceed $50 per night.",
            sources,
            node_pet_fee,
            "Consider phrases like 'pet fee $X per night' or 'per stay'. If the fee is per stay and ≤ $50, it does not exceed $50 per night. Ignore refundable deposits. If no valid reference URL, treat as unsupported."
        ),
        (
            f"The hotel '{hotel_name}' offers a park-stay-cruise or park-and-cruise (or similar) package that includes parking for the duration of the cruise.",
            sources,
            node_pkg_parking,
            "Look for 'Park and Cruise', 'Stay, Park & Cruise', or similar wording indicating parking coverage for the cruise duration."
        ),
        (
            f"The parking included for '{hotel_name}' allows at least 7 consecutive days.",
            sources,
            node_parking_7,
            "Verify text explicitly stating parking duration of at least 7 days/nights included with the package."
        ),
        (
            f"The hotel '{hotel_name}' provides or arranges shuttle transportation to Port Everglades.",
            sources,
            node_shuttle,
            "Look for shuttle/transportation mentions specifically referencing Port Everglades or cruise terminal."
        ),
        (
            f"The hotel '{hotel_name}' is within 5 miles of Fort Lauderdale-Hollywood International Airport (FLL).",
            sources,
            node_within_fll,
            "Use any stated distance or proximity info on the page (miles or km; convert if needed). If only address is present without distance, treat as unsupported."
        ),
        (
            f"The hotel '{hotel_name}' is within 5 miles of Port Everglades.",
            sources,
            node_within_port,
            "Use any stated distance or proximity info on the page (miles or km; convert if needed). If only address is present without distance, treat as unsupported."
        ),
    ]

    # Run batch verification with prerequisite on reference URL presence
    await evaluator.batch_verify(
        [
            (claim, srcs, node, add_ins)
            for (claim, srcs, node, add_ins) in claims_and_sources
        ],
        extra_prerequisites=[reference_url_leaf] if reference_url_leaf else None,
        majority_vote=True,
        num_trials=3,
        use_screenshot=True
    )

    return [
        node_loc_ftl, node_accepts_dogs, node_weight, node_pet_fee,
        node_pkg_parking, node_parking_7, node_shuttle, node_within_fll, node_within_port
    ]


# ---------------------------
# Main Evaluation Function
# ---------------------------
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
        default_model=model
    )

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_hotel_info(),
        template_class=HotelExtraction,
        extraction_name="hotel_extraction"
    )

    core = extraction.hotel_core or HotelCoreInfo()

    # Build top-level critical evaluation node
    top_eval = evaluator.add_parallel(
        id="hotel_answer_evaluation",
        desc="Evaluate whether the response identifies one hotel that meets all constraints and provides all required booking details and evidence.",
        parent=root,
        critical=True
    )

    # Required response fields present (critical)
    required_fields_parent = evaluator.add_parallel(
        id="required_response_fields_present",
        desc="Response includes all required fields requested in the question.",
        parent=top_eval,
        critical=True
    )

    # Field existence checks (critical custom nodes)
    name_provided_node = evaluator.add_custom_node(
        result=bool(core.hotel_name and core.hotel_name.strip()),
        id="complete_hotel_name_provided",
        desc="Complete hotel name is provided.",
        parent=required_fields_parent,
        critical=True
    )

    brand_present_or_in_name = (bool(core.brand and core.brand.strip())) or infer_brand_from_name(core.hotel_name)
    brand_provided_node = evaluator.add_custom_node(
        result=brand_present_or_in_name,
        id="hotel_chain_or_brand_provided",
        desc="Hotel chain/brand (if applicable) is identified.",
        parent=required_fields_parent,
        critical=True
    )

    checkin_provided_node = evaluator.add_custom_node(
        result=bool(core.check_in_time and core.check_in_time.strip()),
        id="standard_check_in_time_provided",
        desc="Hotel's standard check-in time is provided.",
        parent=required_fields_parent,
        critical=True
    )

    contact_provided = bool((core.contact_phone and core.contact_phone.strip()) or (core.contact_website and core.contact_website.strip()))
    contact_provided_node = evaluator.add_custom_node(
        result=contact_provided,
        id="contact_information_provided",
        desc="Hotel contact information (phone number or website) is provided.",
        parent=required_fields_parent,
        critical=True
    )

    # Reference URL check: at least one official or major booking platform URL
    refs = core.reference_urls or []
    reference_url_ok = has_any_valid_reference_url(refs, core.brand, core.hotel_name)
    reference_url_provided_node = evaluator.add_custom_node(
        result=reference_url_ok,
        id="reference_url_provided",
        desc="At least one reference URL from the hotel's official website or a major booking platform is provided.",
        parent=required_fields_parent,
        critical=True
    )

    # Constraints verification (critical)
    constraint_leaf_nodes = await build_and_verify_constraints(
        evaluator=evaluator,
        parent_node=top_eval,
        extraction=extraction,
        reference_url_leaf=reference_url_provided_node
    )

    # Reference URL confirms constraints (critical). We set it based on previous leaf outcomes.
    all_constraints_passed = all(n.status == "passed" for n in constraint_leaf_nodes)
    evaluator.add_custom_node(
        result=reference_url_ok and all_constraints_passed,
        id="reference_url_confirms_constraints",
        desc="The provided reference URL(s) collectively substantiate the key stated constraints (pet policy/fee/weight, park-and-cruise parking duration, shuttle, and proximity/location claims).",
        parent=top_eval,
        critical=True
    )

    return evaluator.get_summary()