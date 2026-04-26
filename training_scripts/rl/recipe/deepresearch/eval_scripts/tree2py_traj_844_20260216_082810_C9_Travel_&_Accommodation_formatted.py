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
TASK_ID = "wisconsin_dells_waterpark_resorts"
TASK_DESCRIPTION = (
    "I'm planning a family vacation to Wisconsin Dells, Wisconsin, and need to compare waterpark resort options. "
    "Please identify three different waterpark resorts in Wisconsin Dells that meet the following specific requirements:\n\n"
    "For each resort, provide:\n"
    "1. Resort name and confirmation that it is located in Wisconsin Dells, Wisconsin\n"
    "2. Official website URL for the resort\n"
    "3. Indoor waterpark amenities: Confirm the resort has an indoor waterpark that includes BOTH of the following features:\n"
    "   - A lazy river\n"
    "   - A wave pool\n"
    "4. Pet policy: State whether the resort allows pets (dogs). If pets ARE allowed, provide:\n"
    "   - The pet fee amount (per night, per pet)\n"
    "   - The weight limit per pet\n"
    "   - The maximum number of pets allowed per room\n"
    "   If pets are NOT allowed, clearly state that pets are not permitted.\n"
    "5. Resort fee: Disclose whether the resort charges a mandatory daily resort fee, and if so, state the amount (per day, before tax)\n"
    "6. Additional amenities: List at least two amenities or services offered by the resort beyond waterpark access (examples: fitness center, spa, golf course, arcade, restaurants, etc.)\n\n"
    "For each piece of information, include a reference URL from the resort's official website or a reputable booking/review site that supports your answer.\n\n"
    "Please ensure all three resorts are distinct properties and provide complete information for each one."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ResortItem(BaseModel):
    name: Optional[str] = None
    official_website: Optional[str] = None

    # Location support
    location_urls: List[str] = Field(default_factory=list)

    # Waterpark amenities support (indoor + lazy river + wave pool)
    waterpark_urls: List[str] = Field(default_factory=list)

    # Pet policy and details (if allowed)
    pet_policy: Optional[str] = None  # Expected values: "allowed", "not allowed", "service animals only", or similar
    pet_fee_amount: Optional[str] = None  # e.g., "$50 per night per pet"
    pet_weight_limit: Optional[str] = None  # e.g., "up to 50 lbs"
    max_pets_per_room: Optional[str] = None  # e.g., "2 pets per room"
    pet_policy_urls: List[str] = Field(default_factory=list)

    # Resort fee disclosure
    resort_fee_amount: Optional[str] = None  # e.g., "$29.95 per day" or "none"
    resort_fee_urls: List[str] = Field(default_factory=list)

    # Additional amenities beyond waterpark
    additional_amenities: List[str] = Field(default_factory=list)
    amenities_urls: List[str] = Field(default_factory=list)


class ResortsExtraction(BaseModel):
    resorts: List[ResortItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_resorts() -> str:
    return (
        "Extract up to five waterpark resorts mentioned in the answer that are in Wisconsin Dells, Wisconsin, "
        "and return structured details for each. Only extract what is explicitly stated in the answer. "
        "For each resort, return the following fields:\n"
        "- name: the resort's name (string)\n"
        "- official_website: the official website URL of the resort (string URL). If not provided, return null.\n"
        "- location_urls: array of URLs from either the official site or reputable booking/review sites that explicitly support "
        "  that the resort is located in Wisconsin Dells, Wisconsin. Return an empty array if none are provided.\n"
        "- waterpark_urls: array of URLs supporting the indoor waterpark features (must be used to verify that there is an indoor waterpark "
        "  and that it includes BOTH a lazy river and a wave pool). Return an empty array if none are provided.\n"
        "- pet_policy: a concise status string extracted from the answer, one of: 'allowed', 'not allowed', or 'service animals only'. "
        "  If ambiguous or not provided, return null.\n"
        "- pet_fee_amount: if pets are allowed, the stated pet fee amount per night per pet (string as presented). Else null.\n"
        "- pet_weight_limit: if pets are allowed, the maximum weight limit per pet (string as presented). Else null.\n"
        "- max_pets_per_room: if pets are allowed, the maximum number of pets allowed per room (string as presented). Else null.\n"
        "- pet_policy_urls: array of URLs supporting the pet policy information. Return an empty array if none are provided.\n"
        "- resort_fee_amount: the mandatory daily resort fee amount (string as presented), or 'none' if the answer explicitly states no mandatory daily resort fee. "
        "  If missing or unclear, return null.\n"
        "- resort_fee_urls: array of URLs supporting the resort fee information. Return an empty array if none are provided.\n"
        "- additional_amenities: an array of at least two amenities beyond waterpark access (e.g., fitness center, spa, golf course, arcade, restaurants, etc.). "
        "  If fewer than two are provided, return whatever is given (including empty array).\n"
        "- amenities_urls: array of URLs supporting the additional amenities (official site or reputable booking/review sites). "
        "  Return an empty array if none are provided.\n\n"
        "Important rules:\n"
        "1) Extract only URLs explicitly present in the answer; do not invent URLs.\n"
        "2) Prefer the resort's official website; otherwise use reputable booking/review sites.\n"
        "3) Do not merge different resorts' information; keep each resort separate.\n"
        "4) If any field is missing in the answer, set it to null (or empty array for URL fields).\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*lists: Optional[List[str]]) -> List[str]:
    combined: List[str] = []
    for lst in lists:
        if lst:
            combined.extend([u for u in lst if isinstance(u, str) and len(u.strip()) > 0])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in combined:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _normalize_pet_status(status: Optional[str]) -> Optional[str]:
    if not status:
        return None
    s = status.strip().lower()
    if s in {"allowed", "pet-friendly", "pets allowed", "dogs allowed"}:
        return "allowed"
    if s in {"not allowed", "no pets", "pets not allowed", "no dogs", "not pet-friendly"}:
        return "not allowed"
    if "service" in s and "animal" in s:
        return "service animals only"
    return None


# --------------------------------------------------------------------------- #
# Verification sub-tree for one resort                                        #
# --------------------------------------------------------------------------- #
async def verify_resort(
    evaluator: Evaluator,
    parent_node,
    resort: ResortItem,
    index: int,
) -> None:
    i = index + 1
    display_name = resort.name or f"Resort #{i}"

    # Resort node (non-critical to allow partial credit per-resort)
    resort_node = evaluator.add_parallel(
        id=f"Resort_{i}",
        desc=f"Resort {i}: Wisconsin Dells waterpark resort meeting specified criteria",
        parent=parent_node,
        critical=False,
    )

    # ---------------- Basic Information ----------------
    basic_node = evaluator.add_parallel(
        id=f"Basic_Information_{i}",
        desc="Resort name, location, and official website",
        parent=resort_node,
        critical=True,  # All children must be critical
    )

    # Existence gate for basic info
    basic_required = evaluator.add_custom_node(
        result=bool(resort.name) and bool(resort.official_website) and (len(resort.location_urls) > 0 or bool(resort.official_website)),
        id=f"Basic_Info_Required_{i}",
        desc="Basic info provided: resort name, official website, and at least one location-supporting URL",
        parent=basic_node,
        critical=True,
    )

    # Leaf: Name and Location verification
    name_loc_leaf = evaluator.add_leaf(
        id=f"Resort_Name_And_Location_{i}",
        desc=f"Resort is correctly identified as '{display_name}' in Wisconsin Dells, Wisconsin",
        parent=basic_node,
        critical=True,
    )
    name_loc_sources = _combine_sources(resort.location_urls, [resort.official_website] if resort.official_website else [])
    await evaluator.verify(
        claim=f"The resort named '{display_name}' is located in Wisconsin Dells, Wisconsin.",
        node=name_loc_leaf,
        sources=name_loc_sources,
        additional_instruction="Confirm the resort's location is Wisconsin Dells, WI. Minor variations like 'Wisconsin Dells, WI' are acceptable.",
    )

    # Leaf: Official website verification
    website_leaf = evaluator.add_leaf(
        id=f"Official_Website_{i}",
        desc="Official website URL is provided and valid",
        parent=basic_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This URL is the official website of the resort '{display_name}'.",
        node=website_leaf,
        sources=resort.official_website,
        additional_instruction="Check branding, logo, and on-page cues (e.g., 'Official Site'). Third-party booking sites should not be considered official.",
    )

    # ---------------- Waterpark Amenities ----------------
    water_node = evaluator.add_parallel(
        id=f"Waterpark_Amenities_{i}",
        desc="Indoor waterpark with lazy river and wave pool",
        parent=resort_node,
        critical=True,  # All children must be critical
    )

    # Existence gate for waterpark sources
    water_sources_exist = evaluator.add_custom_node(
        result=len(resort.waterpark_urls) > 0,
        id=f"Waterpark_Sources_Exist_{i}",
        desc="URLs supporting indoor waterpark features are provided",
        parent=water_node,
        critical=True,
    )

    indoor_leaf = evaluator.add_leaf(
        id=f"Indoor_Waterpark_{i}",
        desc="Resort has indoor waterpark facilities",
        parent=water_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The resort '{display_name}' has an indoor waterpark.",
        node=indoor_leaf,
        sources=resort.waterpark_urls,
        additional_instruction="Look for explicit mention of 'indoor waterpark' or 'indoor pools' as part of the waterpark.",
    )

    lazy_leaf = evaluator.add_leaf(
        id=f"Lazy_River_{i}",
        desc="Indoor waterpark includes a lazy river",
        parent=water_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The indoor waterpark at '{display_name}' includes a lazy river.",
        node=lazy_leaf,
        sources=resort.waterpark_urls,
        additional_instruction="Confirm that a lazy river is listed as a feature of the indoor waterpark.",
    )

    wave_leaf = evaluator.add_leaf(
        id=f"Wave_Pool_{i}",
        desc="Indoor waterpark includes a wave pool",
        parent=water_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The indoor waterpark at '{display_name}' includes a wave pool.",
        node=wave_leaf,
        sources=resort.waterpark_urls,
        additional_instruction="Confirm that a wave pool is listed as a feature of the indoor waterpark.",
    )

    amenities_ref_leaf = evaluator.add_leaf(
        id=f"Amenities_URL_Reference_{i}",
        desc="URL reference supporting waterpark amenities claims",
        parent=water_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"At least one of the provided URLs explicitly supports that '{display_name}' has an indoor waterpark with a lazy river and a wave pool.",
        node=amenities_ref_leaf,
        sources=resort.waterpark_urls,
        additional_instruction="Check the resort's official site or reputable booking/review sites for explicit features.",
    )

    # ---------------- Pet Policy ----------------
    pet_node = evaluator.add_parallel(
        id=f"Pet_Policy_{i}",
        desc="Clear pet policy statement with specific details if pets allowed",
        parent=resort_node,
        critical=False,  # Mixed critical/non-critical children; parent must be non-critical
    )

    # Existence gate for pet policy sources (critical under this node)
    pet_sources_exist = evaluator.add_custom_node(
        result=len(resort.pet_policy_urls) > 0,
        id=f"Pet_Policy_Sources_Exist_{i}",
        desc="URLs supporting pet policy information are provided",
        parent=pet_node,
        critical=True,
    )

    normalized_pet = _normalize_pet_status(resort.pet_policy)
    pet_allowed_flag = evaluator.add_custom_node(
        result=(normalized_pet == "allowed"),
        id=f"Pets_Allowed_Flag_{i}",
        desc="Pets allowed flag based on the extracted policy",
        parent=pet_node,
        critical=False,  # Used as a precondition for details
    )

    pet_policy_leaf = evaluator.add_leaf(
        id=f"Pet_Policy_Stated_{i}",
        desc="Resort's pet policy is clearly stated (whether pets are allowed or not)",
        parent=pet_node,
        critical=True,
    )

    if normalized_pet == "allowed":
        pet_claim = f"Pets (dogs) are allowed at '{display_name}'."
    elif normalized_pet == "service animals only":
        pet_claim = f"Only service animals are permitted at '{display_name}'; pets (non-service) are not allowed."
    else:
        pet_claim = f"Pets are not permitted at '{display_name}'."

    await evaluator.verify(
        claim=pet_claim,
        node=pet_policy_leaf,
        sources=resort.pet_policy_urls,
        additional_instruction="Interpret 'pet-friendly' or 'dogs permitted' as allowed. If it says 'service animals only', treat as not allowed for pets.",
    )

    pet_details_node = evaluator.add_parallel(
        id=f"Pet_Policy_Details_{i}",
        desc="If pets are allowed, specific fee, weight limit, and maximum count are provided",
        parent=pet_node,
        critical=False,  # Non-critical details
    )

    # Pet Fee Amount
    pet_fee_leaf = evaluator.add_leaf(
        id=f"Pet_Fee_Amount_{i}",
        desc="Pet fee amount per night per pet is specified",
        parent=pet_details_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The pet fee at '{display_name}' is {resort.pet_fee_amount} per night per pet.",
        node=pet_fee_leaf,
        sources=resort.pet_policy_urls,
        additional_instruction="Verify 'per night per pet' fee amount. If fee is per stay or cleaning fee, it should not be considered equivalent.",
        extra_prerequisites=[pet_allowed_flag],
    )

    # Pet Weight Limit
    pet_weight_leaf = evaluator.add_leaf(
        id=f"Pet_Weight_Limit_{i}",
        desc="Maximum weight limit per pet is specified",
        parent=pet_details_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The maximum weight limit per pet at '{display_name}' is {resort.pet_weight_limit}.",
        node=pet_weight_leaf,
        sources=resort.pet_policy_urls,
        additional_instruction="Verify a numeric weight limit (e.g., 'up to 50 lbs').",
        extra_prerequisites=[pet_allowed_flag],
    )

    # Max Pets Per Room
    max_pets_leaf = evaluator.add_leaf(
        id=f"Max_Pets_Per_Room_{i}",
        desc="Maximum number of pets allowed per room is specified",
        parent=pet_details_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"The maximum number of pets allowed per room at '{display_name}' is {resort.max_pets_per_room}.",
        node=max_pets_leaf,
        sources=resort.pet_policy_urls,
        additional_instruction="Verify a clear maximum count (e.g., '2 pets per room').",
        extra_prerequisites=[pet_allowed_flag],
    )

    pet_ref_leaf = evaluator.add_leaf(
        id=f"Pet_Policy_URL_Reference_{i}",
        desc="URL reference supporting pet policy information",
        parent=pet_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"At least one of the provided URLs clearly states the pet policy for '{display_name}'.",
        node=pet_ref_leaf,
        sources=resort.pet_policy_urls,
        additional_instruction="Check official site policies or reputable booking/review sites.",
    )

    # ---------------- Resort Fee and Additional Amenities ----------------
    fees_node = evaluator.add_parallel(
        id=f"Resort_Fee_And_Additional_Amenities_{i}",
        desc="Resort fee disclosure and at least two additional amenities",
        parent=resort_node,
        critical=True,  # All children must be critical
    )

    # Existence gate for fees/amenities sources
    fees_amenities_sources_exist = evaluator.add_custom_node(
        result=(len(resort.resort_fee_urls) > 0 or len(resort.amenities_urls) > 0),
        id=f"Fees_Amenities_Sources_Exist_{i}",
        desc="At least one URL exists to support resort fee or amenities information",
        parent=fees_node,
        critical=True,
    )

    # Resort Fee Disclosure leaf
    fee_leaf = evaluator.add_leaf(
        id=f"Resort_Fee_Disclosure_{i}",
        desc="Mandatory resort fee amount (if any) is clearly disclosed",
        parent=fees_node,
        critical=True,
    )
    fee_sources = resort.resort_fee_urls
    if resort.resort_fee_amount and resort.resort_fee_amount.strip().lower() != "none":
        fee_claim = f"The resort '{display_name}' charges a mandatory daily resort fee of {resort.resort_fee_amount} (before tax)."
    else:
        fee_claim = f"The resort '{display_name}' does not charge a mandatory daily resort fee."
    await evaluator.verify(
        claim=fee_claim,
        node=fee_leaf,
        sources=fee_sources,
        additional_instruction="Confirm whether a daily mandatory resort fee is charged and its amount (before tax). If none, verify explicit 'no resort fee' statement.",
    )

    # Additional amenities existence gate
    amenities_exist = evaluator.add_custom_node(
        result=(len(resort.additional_amenities) >= 2 and len(resort.amenities_urls) > 0),
        id=f"Two_Amenities_Listed_Exist_{i}",
        desc="At least two additional amenities listed and supporting URLs provided",
        parent=fees_node,
        critical=True,
    )

    amenities_leaf = evaluator.add_leaf(
        id=f"Additional_Amenities_{i}",
        desc="At least two additional amenities beyond waterpark are listed",
        parent=fees_node,
        critical=True,
    )
    amenities_text = ", ".join(resort.additional_amenities) if resort.additional_amenities else "none listed"
    await evaluator.verify(
        claim=f"The resort '{display_name}' offers additional amenities beyond waterpark access, including: {amenities_text}. At least two of these are offered.",
        node=amenities_leaf,
        sources=resort.amenities_urls,
        additional_instruction="Verify that at least two listed amenities appear across the provided URLs (e.g., fitness center, spa, golf, arcade, restaurants).",
    )

    fees_ref_leaf = evaluator.add_leaf(
        id=f"Fees_And_Amenities_URL_Reference_{i}",
        desc="URL reference supporting resort fee and amenities information",
        parent=fees_node,
        critical=True,
    )
    combined_fee_amenities_sources = _combine_sources(resort.resort_fee_urls, resort.amenities_urls)
    await evaluator.verify(
        claim=f"At least one of the provided URLs supports the resort fee or the listed additional amenities for '{display_name}'.",
        node=fees_ref_leaf,
        sources=combined_fee_amenities_sources,
        additional_instruction="Use the official site or reputable booking/review pages to confirm fee and amenities.",
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Wisconsin Dells waterpark resorts task.
    """
    # Initialize evaluator
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

    # Extract resorts
    extraction: ResortsExtraction = await evaluator.extract(
        prompt=prompt_extract_resorts(),
        template_class=ResortsExtraction,
        extraction_name="resorts_extraction",
    )

    # Filter to first three resorts; pad with empty entries if fewer
    resorts: List[ResortItem] = list(extraction.resorts[:3])
    while len(resorts) < 3:
        resorts.append(ResortItem())

    # Top-level task node (non-critical to allow partial credit across resorts)
    top_node = evaluator.add_parallel(
        id="Find_Three_Wisconsin_Dells_Waterpark_Resorts",
        desc="Task requires finding three Wisconsin Dells waterpark resorts with specific amenities and policies",
        parent=root,
        critical=False,
    )

    # Global checks: at least three resorts provided and distinct properties
    provided_names = [r.name for r in resorts if r.name]
    three_provided = evaluator.add_custom_node(
        result=len([n for n in provided_names if n and n.strip()]) >= 3,
        id="Three_Resorts_Provided",
        desc="At least three resorts are provided with names",
        parent=top_node,
        critical=True,
    )

    lower_names = [n.strip().lower() for n in provided_names]
    distinct_resorts = evaluator.add_custom_node(
        result=(len(lower_names) == 3 and len(set(lower_names)) == 3),
        id="Distinct_Resorts",
        desc="All three resorts are distinct properties",
        parent=top_node,
        critical=True,
    )

    # Build verification sub-trees for each resort
    for idx, resort in enumerate(resorts):
        await verify_resort(evaluator, top_node, resort, idx)

    # Return evaluation summary
    return evaluator.get_summary()