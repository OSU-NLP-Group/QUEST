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
TASK_ID = "ca_beachfront_pet_hotel"
TASK_DESCRIPTION = (
    "I'm planning a beach vacation in California with my two large dogs (both around 80 pounds each). "
    "I need to find a beachfront hotel that can accommodate both of my dogs and allows me to take them for walks directly on the beach.\n\n"
    "Please identify a pet-friendly beachfront hotel in California that meets the following requirements:\n"
    "- Must be located directly on the beachfront/oceanfront\n"
    "- Must accept dogs weighing at least 75 pounds\n"
    "- Must allow at least 2 dogs per room\n"
    "- Must have a clearly stated pet fee policy\n"
    "- Must provide pet-specific amenities (such as pet beds, bowls, treats, or waste bags)\n"
    "- Must offer direct beach access for walking dogs\n"
    "- Must have a publicly accessible website where I can learn more or book\n\n"
    "For the hotel you identify, please provide:\n"
    "- The hotel name and location (city/region)\n"
    "- The specific pet weight limit or policy\n"
    "- The maximum number of pets allowed per room\n"
    "- The pet fee amount and structure (per night or per stay)\n"
    "- What pet amenities are provided\n"
    "- A link to the hotel's website or pet policy page"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelExtraction(BaseModel):
    # Identification
    hotel_name: Optional[str] = None
    location: Optional[str] = None  # city/region/state

    # Beach context (answer-stated; free text)
    beachfront_or_oceanfront: Optional[str] = None
    direct_beach_access: Optional[str] = None

    # Pet policies (answer-stated; free text)
    weight_policy_text: Optional[str] = None  # e.g., "No weight limit", "Up to 100 lbs per dog"
    allows_two_or_more_dogs_text: Optional[str] = None  # e.g., "2 dogs per room", "up to two pets"
    pet_fee_amount: Optional[str] = None  # e.g., "$150", "$75"
    pet_fee_structure: Optional[str] = None  # e.g., "per night", "per stay", "per dog per stay"
    pet_fee_policy_text: Optional[str] = None  # free-form policy text if present

    # Amenities
    pet_amenities: List[str] = Field(default_factory=list)

    # URLs explicitly present in the answer (official site, property page, pet policy page, booking page, etc.)
    website_urls: List[str] = Field(default_factory=list)
    pet_policy_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotel_info() -> str:
    return """
    Extract details for a single identified hotel from the answer. If multiple hotels are mentioned, extract the first one only.
    Return fields exactly as stated in the answer text (do not infer).

    Required fields to extract:
    - hotel_name: The specific hotel/property name.
    - location: The location as written (e.g., city/region/state).
    - beachfront_or_oceanfront: The exact wording (if the answer claims beachfront/oceanfront).
    - direct_beach_access: The exact wording (if the answer claims direct beach access).
    - weight_policy_text: The dog weight policy wording (e.g., 'No weight limit' or 'Up to 100 lbs per pet'); null if not stated.
    - allows_two_or_more_dogs_text: The wording about how many pets are allowed per room (e.g., '2 dogs per room'); null if not stated.
    - pet_fee_amount: The fee amount text (e.g., '$150'); null if not stated.
    - pet_fee_structure: The structure text (e.g., 'per night', 'per stay', 'per pet per night'); null if not stated.
    - pet_fee_policy_text: If the answer provides a single combined fee statement, include it here; otherwise null.
    - pet_amenities: List at least the pet-specific amenities mentioned (e.g., pet beds, bowls, treats, waste bags). If none are explicitly mentioned, return an empty list.
    - website_urls: All URLs in the answer that appear to be official property/hotel or booking/policy pages. Extract actual URLs (including those in markdown links).
    - pet_policy_urls: All URLs in the answer that specifically look like pet policy pages (if any). Extract actual URLs.

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer.
    - Include full URLs with protocol. If a URL misses protocol, prepend http://.
    - Do not invent or infer URLs.

    If any field is not present in the answer, set it to null (or empty array for pet_amenities).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(extracted: HotelExtraction) -> List[str]:
    """Combine all extracted URLs into a unique list for verification."""
    urls: List[str] = []
    if extracted.website_urls:
        urls.extend([u for u in extracted.website_urls if isinstance(u, str) and u.strip() != ""])
    if extracted.pet_policy_urls:
        urls.extend([u for u in extracted.pet_policy_urls if isinstance(u, str) and u.strip() != ""])
    # Deduplicate while preserving order
    seen = set()
    unique_urls: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)
    return unique_urls


def _safe(val: Optional[str], fallback: str = "") -> str:
    return val if isinstance(val, str) and val.strip() else fallback


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_pet_friendly_beachfront_hotel(
    evaluator: Evaluator,
    parent_node,
    extracted: HotelExtraction
) -> None:
    """
    Build verification nodes according to the rubric and run URL-backed checks.
    All eight rubric checks are modeled as critical leaf nodes under a single critical parallel node.
    """
    # Critical parent aggregator mirroring rubric root
    response_node = evaluator.add_parallel(
        id="Pet_Friendly_Beachfront_Hotel_Response",
        desc="Evaluate whether the response identifies a qualifying pet-friendly beachfront hotel in California and provides all required hotel/pet-policy details.",
        parent=parent_node,
        critical=True
    )

    # Prepare common materials
    sources_list = _combine_sources(extracted)
    name = _safe(extracted.hotel_name, "[missing name]")
    location = _safe(extracted.location, "[missing location]")

    # 1) Hotel identified in CA with name and location
    node_hotel_id = evaluator.add_leaf(
        id="Hotel_Identified_In_California_With_Name_And_Location",
        desc="Response provides a specific hotel name and its location (city/region) in California.",
        parent=response_node,
        critical=True
    )
    claim_hotel_id = (
        f"The answer identifies a specific hotel named '{name}' with location '{location}', and the hotel is located in California (CA). "
        f"Use the provided URLs to confirm the property is indeed in California."
    )
    add_ins_hotel_id = (
        "Confirm via the linked page(s) that the hotel's address is in California (CA). "
        "If the answer omits either a clear hotel name or a clear California location, consider this incorrect. "
        "If the URL points to a chain, ensure the page corresponds to a California property."
    )

    # 2) Direct beachfront/oceanfront location
    node_beachfront = evaluator.add_leaf(
        id="Direct_Beachfront_Oceanfront_Location",
        desc="Response states the hotel is located directly on the beachfront/oceanfront.",
        parent=response_node,
        critical=True
    )
    claim_beachfront = (
        "This hotel is directly beachfront/oceanfront (i.e., on the beach or immediately adjacent to the beach without a public road in between)."
    )
    add_ins_beachfront = (
        "Look for terms such as 'beachfront', 'oceanfront', 'on the beach', 'steps from the sand', 'private beach', or 'direct beach access'. "
        "Do not accept phrasing like 'near the beach' or 'short walk to the beach'."
    )

    # 3) Direct beach access suitable for walking dogs
    node_direct_access = evaluator.add_leaf(
        id="Dog_Walks_Direct_Beach_Access",
        desc="Response states the hotel offers direct beach access suitable for walking dogs on the beach.",
        parent=response_node,
        critical=True
    )
    claim_direct_access = (
        "The property offers direct access from the hotel premises to the beach (e.g., private/onsite path or steps to the sand), "
        "making it possible to walk dogs directly on the beach from the hotel."
    )
    add_ins_direct_access = (
        "You do not need explicit wording 'dogs allowed on the beach' to pass; however, the hotel must have direct beach access. "
        "If the hotel's page explicitly states there is no direct access or dogs are prohibited on the adjacent beach, mark as incorrect."
    )

    # 4) Dog weight policy allows at least 75 lbs
    node_weight = evaluator.add_leaf(
        id="Dog_Weight_Policy_Allows_At_Least_75_Lbs",
        desc="Response provides the hotel’s dog weight limit/policy and it allows dogs weighing at least 75 pounds (e.g., explicit limit ≥75 or no stated weight restriction).",
        parent=response_node,
        critical=True
    )
    weight_text = _safe(extracted.weight_policy_text)
    if weight_text:
        claim_weight = (
            f"The hotel's pet policy (e.g., '{weight_text}') allows dogs weighing at least 75 pounds per dog "
            f"(either explicitly setting a limit of 75 lb or more, or stating there is no weight restriction)."
        )
    else:
        claim_weight = (
            "The hotel's pet policy allows dogs weighing at least 75 pounds per dog, either by explicitly stating a weight limit of 75 lb or more, "
            "or by stating there is no weight restriction."
        )
    add_ins_weight = (
        "Look for phrases like 'no weight limit' or explicit maximums (e.g., 'up to 100 lb per dog'). "
        "If the policy caps weight below 75 lb or restricts to small/medium dogs only, this is incorrect."
    )

    # 5) Allows at least two dogs per room
    node_dogs_count = evaluator.add_leaf(
        id="Allows_At_Least_Two_Dogs_Per_Room",
        desc="Response provides the maximum number of dogs/pets allowed per room and it is at least 2.",
        parent=response_node,
        critical=True
    )
    dogs_text = _safe(extracted.allows_two_or_more_dogs_text)
    if dogs_text:
        claim_dogs_count = (
            f"The hotel's pet policy (e.g., '{dogs_text}') allows at least two dogs (or pets) per room."
        )
    else:
        claim_dogs_count = (
            "The hotel's pet policy allows at least two dogs (or pets) per room."
        )
    add_ins_dogs_count = (
        "Accept wording like 'up to 2 pets', '2 dogs allowed', or any phrasing clearly allowing at least two pets per room. "
        "If it only allows 1 pet, mark incorrect."
    )

    # 6) Pet fee policy and amount provided
    node_fee = evaluator.add_leaf(
        id="Pet_Fee_Policy_And_Amount_Provided",
        desc="Response provides a clearly stated pet fee policy including the fee amount and whether it is charged per night or per stay (or equivalent structure).",
        parent=response_node,
        critical=True
    )
    fee_amount = _safe(extracted.pet_fee_amount)
    fee_structure = _safe(extracted.pet_fee_structure)
    fee_text = _safe(extracted.pet_fee_policy_text)

    if fee_amount and fee_structure:
        claim_fee = (
            f"The answer provides a clear pet fee policy including the amount '{fee_amount}' and structure '{fee_structure}', "
            f"and the linked page(s) confirm these details."
        )
    elif fee_text:
        claim_fee = (
            f"The answer includes a clear pet fee policy statement (e.g., '{fee_text}') that includes both a fee amount and whether it is charged "
            f"per night or per stay, and the linked page(s) confirm these details."
        )
    else:
        claim_fee = (
            "The answer includes a clear pet fee policy with a specific amount and an explicit structure (per night vs per stay), "
            "and the linked page(s) confirm these details."
        )
    add_ins_fee = (
        "To pass, the answer must state both the fee amount and the fee structure (e.g., per night/per stay/per pet per stay). "
        "Verify on the linked page(s) that these details are supported. If the answer omits either the amount or the structure, "
        "or the website contradicts it, mark incorrect."
    )

    # 7) Pet-specific amenities described
    node_amenities = evaluator.add_leaf(
        id="Pet_Specific_Amenities_Described",
        desc="Response describes at least one pet-specific amenity offered by the hotel (e.g., pet beds, bowls, treats, waste bags).",
        parent=response_node,
        critical=True
    )
    amenities_list = extracted.pet_amenities or []
    if amenities_list:
        amenities_str = ", ".join(amenities_list[:5])
        claim_amenities = (
            f"The answer lists at least one pet-specific amenity (e.g., {amenities_str}), "
            f"and the linked page(s) show that the hotel offers at least one of these pet-specific amenities."
        )
    else:
        claim_amenities = (
            "The hotel offers at least one pet-specific amenity (e.g., pet bed, water bowl, treats, waste bags), "
            "and at least one such amenity is supported by the linked page(s)."
        )
    add_ins_amenities = (
        "The answer must describe at least one pet-specific amenity; confirm via the linked page(s) that the hotel offers "
        "at least one such amenity. Generic 'pet-friendly' without an amenity does not pass."
    )

    # 8) Public website link provided
    node_public_site = evaluator.add_leaf(
        id="Public_Website_Link_Provided",
        desc="Response provides a publicly accessible website link (hotel site or pet policy page) where the user can learn more and/or book and that supports/verifies the pet policy information.",
        parent=response_node,
        critical=True
    )
    claim_public_site = (
        "The answer includes at least one publicly accessible link to the hotel's official site, property page, booking page, "
        "or pet policy page for the identified hotel."
    )
    add_ins_public_site = (
        "Check that at least one URL is present in the answer and opens as a publicly accessible page (not requiring login). "
        "Prefer official hotel site or brand-owned property/policy page, but an accessible booking/property page is acceptable."
    )

    # Batch verify all leaves with the combined sources
    claims_and_sources = [
        (claim_hotel_id, sources_list, node_hotel_id, add_ins_hotel_id),
        (claim_beachfront, sources_list, node_beachfront, add_ins_beachfront),
        (claim_direct_access, sources_list, node_direct_access, add_ins_direct_access),
        (claim_weight, sources_list, node_weight, add_ins_weight),
        (claim_dogs_count, sources_list, node_dogs_count, add_ins_dogs_count),
        (claim_fee, sources_list, node_fee, add_ins_fee),
        (claim_amenities, sources_list, node_amenities, add_ins_amenities),
        (claim_public_site, sources_list, node_public_site, add_ins_public_site),
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the California beachfront pet-friendly hotel task.
    """
    # Initialize evaluator (root is a non-critical container)
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

    # Extract structured hotel information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotel_info(),
        template_class=HotelExtraction,
        extraction_name="hotel_extraction"
    )

    # Record task requirements as ground truth metadata for transparency
    evaluator.add_ground_truth({
        "requirements": [
            "Located directly on the beachfront/oceanfront",
            "Accepts dogs at least 75 lbs (or no weight limit)",
            "Allows at least two dogs per room",
            "Clearly stated pet fee policy with amount and structure",
            "Provides at least one pet-specific amenity",
            "Direct beach access for walking dogs",
            "Publicly accessible website link provided",
            "Hotel in California with name and location"
        ]
    }, gt_type="rubric_requirements")

    # Build verification tree and run checks
    await verify_pet_friendly_beachfront_hotel(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()