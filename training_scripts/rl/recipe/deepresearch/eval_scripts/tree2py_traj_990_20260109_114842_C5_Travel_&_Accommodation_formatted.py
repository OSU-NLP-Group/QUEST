import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "accessible_florida_beach_resorts"
TASK_DESCRIPTION = (
    "I'm planning a beach vacation in Florida for my family, and I need to find three beach resorts that meet our "
    "specific accessibility and amenity needs. For each resort, please provide:\n\n"
    "1. The resort name and location (city) in Florida\n"
    "2. Confirmation that it has ADA-compliant accessible guest rooms with roll-in showers\n"
    "3. Confirmation of accessible parking availability\n"
    "4. Confirmation of accessible pool entry (lift, ramp, or zero-entry)\n"
    "5. The name of at least one on-site restaurant\n"
    "6. Confirmation of direct beach access (ground-level access, ramps, or beach wheelchair availability)\n"
    "7. Pet policy details (maximum weight limit and nightly fee)\n"
    "8. A reference URL to the resort's official website or accessibility information page\n\n"
    "All three resorts must be beachfront properties in Florida with complete ADA accessibility features."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ResortInfo(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    beachfront_statement: Optional[str] = None  # e.g., "beachfront", "oceanfront", "on the beach"
    accessible_guest_rooms_rollin_statement: Optional[str] = None  # statement mentioning roll-in showers
    rollin_shower_dimensions_statement: Optional[str] = None  # statement mentioning 30x60 or explicit dimensions
    accessible_parking_statement: Optional[str] = None
    accessible_pool_entry_type: Optional[str] = None  # e.g., "pool lift", "ramp", "zero-entry"
    onsite_restaurants: List[str] = Field(default_factory=list)  # at least one name
    direct_beach_access_statement: Optional[str] = None  # ground-level, ramp, beach wheelchair availability
    pet_policy_max_weight: Optional[str] = None  # e.g., "50 lbs", "25 kilograms"
    pet_policy_nightly_fee: Optional[str] = None  # e.g., "$100/night"
    verification_url: Optional[str] = None  # official site or accessibility page
    source_urls: List[str] = Field(default_factory=list)  # any additional official URLs provided


class ResortsExtraction(BaseModel):
    resorts: List[ResortInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_resorts() -> str:
    return """
    Extract up to the first three resort entries mentioned in the answer. For each resort, return the following fields:

    - name: The resort's name exactly as provided.
    - city: The Florida city for the resort.
    - beachfront_statement: Text explicitly indicating the resort is beachfront/on the beach/oceanfront/Gulf-front.
    - accessible_guest_rooms_rollin_statement: Text confirming ADA-compliant accessible guest rooms with roll-in showers.
    - rollin_shower_dimensions_statement: Text confirming roll-in showers meet ADA minimum dimensions (30 inches wide by 60 inches deep). If no dimensions are stated, set to null.
    - accessible_parking_statement: Text confirming ADA-accessible parking availability.
    - accessible_pool_entry_type: The accessible pool entry type (e.g., "pool lift", "ramp", "zero-entry"). If unspecified, set to null.
    - onsite_restaurants: An array of the names of on-site restaurant(s) (include at least one if provided).
    - direct_beach_access_statement: Text confirming direct beach access (ground-level access, ramps, or beach wheelchair availability).
    - pet_policy_max_weight: The maximum pet weight limit if provided (as text, e.g., "50 lbs").
    - pet_policy_nightly_fee: The nightly pet fee if provided (as text, e.g., "$100 per night").
    - verification_url: A URL to the resort's official website or official accessibility page.
    - source_urls: Any additional URLs explicitly provided for this resort; must be actual URLs listed in the answer.

    Rules:
    - Extract only information explicitly present in the answer.
    - If a field is missing, set it to null (or empty array for onsite_restaurants/source_urls).
    - If more than three resorts are listed, include only the first three.
    - Do not invent data; do not infer URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _all_sources(resort: ResortInfo) -> List[str]:
    urls: List[str] = []
    if resort.verification_url and resort.verification_url.strip():
        urls.append(resort.verification_url.strip())
    if resort.source_urls:
        urls.extend([u for u in resort.source_urls if isinstance(u, str) and u.strip()])
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


def _no_sources_instruction(urls: List[str]) -> str:
    if urls:
        return "None"
    return ("No verification URLs are provided in the answer for this resort. "
            "If there are no URLs to check, treat the claim as NOT SUPPORTED.")


def _first_restaurant(resort: ResortInfo) -> Optional[str]:
    return resort.onsite_restaurants[0] if resort.onsite_restaurants else None


# --------------------------------------------------------------------------- #
# Verification logic per resort                                               #
# --------------------------------------------------------------------------- #
async def verify_resort(
    evaluator: Evaluator,
    parent_node,
    resort: ResortInfo,
    index: int,
) -> None:
    """
    Build the verification subtree for a single resort and execute the leaf verifications.
    All leaves are created first and then verified in parallel via batch_verify to avoid
    order-dependent skipping due to sibling critical failures.
    """
    resort_node = evaluator.add_parallel(
        id=f"Resort_{index + 1}",
        desc=f"{['First','Second','Third'][index]} resort entry meets all constraints and includes all required fields.",
        parent=parent_node,
        critical=False
    )

    sources = _all_sources(resort)
    missing_src_ins = _no_sources_instruction(sources)

    # Create leaf nodes for each criterion (all critical per rubric)
    name_city_node = evaluator.add_leaf(
        id=f"resort_{index + 1}_name_city_in_florida",
        desc="Provides the resort name and Florida city, and the property is beachfront/on a beach in Florida.",
        parent=resort_node,
        critical=True
    )

    rooms_rollin_node = evaluator.add_leaf(
        id=f"resort_{index + 1}_rooms_rollin",
        desc="Confirms ADA-compliant accessible guest rooms with roll-in showers.",
        parent=resort_node,
        critical=True
    )

    rollin_dims_node = evaluator.add_leaf(
        id=f"resort_{index + 1}_rollin_dimensions",
        desc="Confirms roll-in showers meet the ADA minimum dimensions of 30 inches wide by 60 inches deep.",
        parent=resort_node,
        critical=True
    )

    accessible_parking_node = evaluator.add_leaf(
        id=f"resort_{index + 1}_accessible_parking",
        desc="Confirms accessible parking availability compliant with ADA standards.",
        parent=resort_node,
        critical=True
    )

    accessible_pool_node = evaluator.add_leaf(
        id=f"resort_{index + 1}_accessible_pool_entry",
        desc="Confirms accessible pool entry via lift, ramp, or zero-entry.",
        parent=resort_node,
        critical=True
    )

    onsite_rest_node = evaluator.add_leaf(
        id=f"resort_{index + 1}_onsite_restaurant_name",
        desc="Provides the name of at least one on-site restaurant.",
        parent=resort_node,
        critical=True
    )

    beach_access_node = evaluator.add_leaf(
        id=f"resort_{index + 1}_direct_beach_access",
        desc="Confirms direct beach access via ground-level access, ramps, or beach wheelchair availability.",
        parent=resort_node,
        critical=True
    )

    pet_policy_node = evaluator.add_leaf(
        id=f"resort_{index + 1}_pet_policy_weight_fee",
        desc="Provides pet policy details including maximum weight limit and nightly fee.",
        parent=resort_node,
        critical=True
    )

    verification_url_node = evaluator.add_leaf(
        id=f"resort_{index + 1}_verification_url",
        desc="Provides a reference URL to the resort's official website or official accessibility information page.",
        parent=resort_node,
        critical=True
    )

    # Build claims
    # 1. Name and City in Florida + Beachfront
    name = resort.name or ""
    city = resort.city or ""
    claim_name_city = (
        f"The resort named '{name}' is located in {city}, Florida and is a beachfront property (on or directly adjacent to the beach)."
    )
    add_ins_name_city = (
        "Use the provided URL(s) to verify both the Florida city location and that the property is beachfront/oceanfront/"
        "Gulf-front/on the beach. Accept synonyms such as 'oceanfront', 'Gulf-front', 'on the beach', or 'beachfront'. "
        + missing_src_ins
    )

    # 2. Accessible guest rooms with roll-in showers
    claim_rooms_rollin = "The resort offers ADA-compliant accessible guest rooms with roll-in showers."
    add_ins_rooms_rollin = (
        "Verify that the official site/accessibility page explicitly mentions 'roll-in showers' in accessible guest rooms "
        "or equivalent phrasing indicating ADA compliance. " + missing_src_ins
    )

    # 3. Roll-in shower ADA minimum dimensions
    claim_rollin_dims = (
        "The resort's roll-in showers meet the ADA minimum dimensions of at least 30 inches wide by 60 inches deep."
    )
    add_ins_rollin_dims = (
        "Only pass if the page explicitly states the roll-in shower dimensions as 30 inches wide by 60 inches deep "
        "(e.g., '30\" x 60\"', '30 inches by 60 inches'). If such dimensions cannot be found, treat as NOT SUPPORTED. "
        + missing_src_ins
    )

    # 4. Accessible parking
    claim_accessible_parking = "ADA-accessible parking is available at this property."
    add_ins_accessible_parking = (
        "Accept phrases like 'accessible parking', 'ADA parking', or explicit mention of ADA-compliant parking. "
        + missing_src_ins
    )

    # 5. Accessible pool entry
    claim_accessible_pool = (
        "The resort's pool(s) have accessible entry via a pool lift or accessible ramp or zero-entry (beach-entry)."
    )
    add_ins_accessible_pool = (
        "Look for 'pool lift', 'lift', 'accessible ramp', or 'zero-entry' references for pool accessibility. "
        + missing_src_ins
    )

    # 6. On-site restaurant name
    restaurant_name = _first_restaurant(resort) or ""
    claim_onsite_rest = f"The resort has an on-site restaurant named '{restaurant_name}'."
    add_ins_onsite_rest = (
        "Verify that the named restaurant is on-site (located at the resort). If the restaurant name is missing or cannot "
        "be corroborated on the official site, treat as NOT SUPPORTED. " + missing_src_ins
    )

    # 7. Direct beach access
    claim_beach_access = (
        "The resort provides direct beach access via ground-level access or accessible ramps or offers beach wheelchairs."
    )
    add_ins_beach_access = (
        "Confirm direct beach access as described (ground-level access, ramps, or beach wheelchair availability). Accept "
        "equivalent phrasing indicating direct access from the property to the beach. " + missing_src_ins
    )

    # 8. Pet policy (weight and nightly fee)
    max_w = resort.pet_policy_max_weight or ""
    night_fee = resort.pet_policy_nightly_fee or ""
    claim_pet_policy = (
        f"The resort's pet policy allows pets up to {max_w} and charges a nightly fee of {night_fee}."
    )
    add_ins_pet_policy = (
        "Verify the maximum pet weight and the nightly fee on the official page. Allow reasonable formatting variations "
        "(e.g., '$100 per night', 'USD 100/night'). If these details cannot be found, treat as NOT SUPPORTED. "
        + missing_src_ins
    )

    # 9. Verification URL (official site or accessibility page)
    url_for_check = resort.verification_url or ""
    claim_verification_url = (
        f"This URL is the resort's official website or official accessibility information page for '{name}'."
    )
    add_ins_verification_url = (
        "Check domain and content to confirm this is the official resort website (brand/hotel domain) or the resort's "
        "official accessibility page. Third-party aggregators (e.g., Expedia, Booking) do NOT count as official. "
        + ("No URL provided; treat as NOT SUPPORTED." if not url_for_check.strip() else "None")
    )

    # Prepare batch verification list
    verifications: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = [
        (claim_name_city, sources if sources else None, name_city_node, add_ins_name_city),
        (claim_rooms_rollin, sources if sources else None, rooms_rollin_node, add_ins_rooms_rollin),
        (claim_rollin_dims, sources if sources else None, rollin_dims_node, add_ins_rollin_dims),
        (claim_accessible_parking, sources if sources else None, accessible_parking_node, add_ins_accessible_parking),
        (claim_accessible_pool, sources if sources else None, accessible_pool_node, add_ins_accessible_pool),
        (claim_onsite_rest, sources if sources else None, onsite_rest_node, add_ins_onsite_rest),
        (claim_beach_access, sources if sources else None, beach_access_node, add_ins_beach_access),
        (claim_pet_policy, sources if sources else None, pet_policy_node, add_ins_pet_policy),
        (claim_verification_url, url_for_check if url_for_check.strip() else None, verification_url_node, add_ins_verification_url),
    ]

    # Execute all leaf verifications in parallel
    await evaluator.batch_verify(verifications)


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
    Evaluate an answer for the Florida accessible beachfront resorts task.
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
        default_model=model
    )

    # Extract resorts provided in the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_resorts(),
        template_class=ResortsExtraction,
        extraction_name="resorts_extraction"
    )

    # Normalize to exactly 3 resorts (pad with empty if fewer; keep only first 3 if more)
    resorts: List[ResortInfo] = list(extraction.resorts[:3])
    while len(resorts) < 3:
        resorts.append(ResortInfo())

    # Build verification tree for the top-level task node
    task_node = evaluator.add_parallel(
        id="Find_Three_Accessible_Beach_Resorts",
        desc="Identify three Florida beachfront resorts that meet all specified accessibility and amenity requirements, and provide the requested details for each.",
        parent=root,
        critical=False
    )

    # Verify each of the three resort entries
    for i in range(3):
        await verify_resort(evaluator, task_node, resorts[i], i)

    # Return unified summary
    return evaluator.get_summary()