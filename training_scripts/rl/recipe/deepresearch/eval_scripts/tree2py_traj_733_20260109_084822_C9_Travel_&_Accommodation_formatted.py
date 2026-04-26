import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "luxury_pacific_mexico_resorts"
TASK_DESCRIPTION = (
    "Identify four distinct luxury beachfront resorts located on Mexico's Pacific coast "
    "(specifically in either the Riviera Nayarit or Los Cabos region) that each meet ALL of the following comprehensive requirements:\n\n"
    "Location & Property Type:\n"
    "- Must be a luxury beachfront resort with direct beach access and a private or semi-private beach area\n"
    "- Must be operated by or affiliated with a recognized luxury hotel brand\n\n"
    "Accommodations:\n"
    "- Minimum of 100 guest rooms or suites\n"
    "- Must offer suite accommodations with private terraces or balconies\n"
    "- Must provide villas or residences with private plunge pools\n\n"
    "Dining:\n"
    "- At least 4 on-site restaurants\n"
    "- 24-hour in-room dining service\n"
    "- At least one beachfront dining venue\n\n"
    "Pool Facilities:\n"
    "- At least 3 separate swimming pools on property\n"
    "- At least one adults-only pool\n"
    "- At least one infinity-edge pool with ocean views\n\n"
    "Spa & Wellness:\n"
    "- Full-service spa facility with a minimum of 10 treatment rooms\n"
    "- Fitness center equipped with cardiovascular and strength training equipment\n\n"
    "Family Amenities:\n"
    "- Supervised kids club program that accepts children starting at age 4 or younger\n"
    "- At least one family-friendly pool area\n\n"
    "Water Sports:\n"
    "- Water sports equipment and activities available\n"
    "- Snorkeling equipment provided\n"
    "- On-site dive center or PADI-certified diving program\n\n"
    "Business Facilities:\n"
    "- Meeting or conference facilities with at least 10,000 square feet of total space\n\n"
    "Service Standards:\n"
    "- 24-hour concierge services\n"
    "- Butler service for suite or villa guests\n"
    "- AAA Four Diamond rating or higher, OR Forbes Travel Guide 4-Star rating or higher\n\n"
    "For each of the four resorts, provide the resort name, luxury brand affiliation, specific location (city/area and region), "
    "and reference URLs documenting that all requirements are met."
)

ALLOWED_REGIONS = ["Riviera Nayarit", "Los Cabos"]
# Region synonyms to help the verifier contextualize locations in allowed regions
REGION_SYNONYMS = {
    "Riviera Nayarit": ["Punta Mita", "Litibu", "Nuevo Vallarta", "Bucerías", "La Cruz de Huanacaxtle", "Sayulita", "San Francisco (San Pancho)", "Nayarit"],
    "Los Cabos": ["Cabo San Lucas", "San José del Cabo", "Corridor", "Baja California Sur", "Cabo", "Palmilla"]
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ResortEntry(BaseModel):
    name: Optional[str] = None
    brand: Optional[str] = None
    location: Optional[str] = None  # City/area
    region: Optional[str] = None    # Preferably 'Riviera Nayarit' or 'Los Cabos' if stated; otherwise null
    urls: List[str] = Field(default_factory=list)


class ResortsExtraction(BaseModel):
    resorts: List[ResortEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_resorts() -> str:
    return (
        "Extract up to four resort entries exactly as presented in the answer. For each entry, return:\n"
        "- name: the resort name (string)\n"
        "- brand: the luxury brand/operator affiliation (string)\n"
        "- location: the specific city/area (string)\n"
        "- region: the broader region, ideally 'Riviera Nayarit' or 'Los Cabos' if explicitly stated; otherwise null\n"
        "- urls: all reference URLs associated with this resort in the answer (array of strings). Include only valid URLs that appear in the answer.\n\n"
        "Rules:\n"
        "1) Do not invent or infer anything; extract only what is explicitly present in the answer.\n"
        "2) If a field is missing, return null (for strings) or an empty array (for urls).\n"
        "3) Preserve the order resorts appear in the answer. If more than four are mentioned, include only the first four.\n"
        "4) For urls, include any official resort brand sites, fact sheets, brochures, press pages, credible third-party sources, or listings included in the answer.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def first_four_distinct_names(resorts: List[ResortEntry]) -> Tuple[bool, List[str]]:
    selected = resorts[:4]
    names = [r.name.strip().lower() for r in selected if r.name]
    distinct = len(names) == 4 and len(set(names)) == 4
    return distinct, [r.name or "" for r in selected]


def safe_urls(resort: ResortEntry) -> List[str]:
    return [u for u in resort.urls if isinstance(u, str) and u.strip()]


def build_location_instruction() -> str:
    syns_rn = ", ".join(REGION_SYNONYMS["Riviera Nayarit"])
    syns_lc = ", ".join(REGION_SYNONYMS["Los Cabos"])
    return (
        "Confirm the resort is on Mexico's Pacific coast specifically in either Riviera Nayarit or Los Cabos.\n"
        "Helpful context for Riviera Nayarit: " + syns_rn + ".\n"
        "Helpful context for Los Cabos: " + syns_lc + ".\n"
        "Allow reasonable naming variants (e.g., local area within the region). The claim should be explicitly supported by the provided URLs."
    )


def general_verification_instruction() -> str:
    return (
        "Focus on the provided URLs to verify the claim. Accept minor wording variations and synonyms (e.g., '24-hour room service' for '24-hour in-room dining', "
        "'infinity pool' for 'infinity-edge pool', 'kids club' starting at 'age 3 or 4', etc.). "
        "If the evidence is not clearly present or the URLs are irrelevant, mark as not supported."
    )


# --------------------------------------------------------------------------- #
# Verification for a single resort                                            #
# --------------------------------------------------------------------------- #
async def verify_resort(
    evaluator: Evaluator,
    parent_node,
    resort: ResortEntry,
    index: int
) -> None:
    name = resort.name or ""
    brand = resort.brand or ""
    location = resort.location or ""
    region = resort.region or ""

    # Container node for a single resort (non-critical to allow partial credit across resorts)
    resort_node = evaluator.add_parallel(
        id=f"resort_{index+1}",
        desc=f"Resort #{index+1} entry meets all constraints and includes required fields.",
        parent=parent_node,
        critical=False
    )

    # Required identifying fields (critical gate)
    req_node = evaluator.add_parallel(
        id=f"resort_{index+1}_required_fields",
        desc="Required identifying fields are present for this resort.",
        parent=resort_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(name.strip()),
        id=f"resort_{index+1}_name_provided",
        desc="Resort name is provided.",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(brand.strip()),
        id=f"resort_{index+1}_brand_provided",
        desc="Luxury brand affiliation/operator is provided.",
        parent=req_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(location.strip()),
        id=f"resort_{index+1}_location_provided",
        desc="Specific location is provided (city/area and region).",
        parent=req_node,
        critical=True
    )

    urls_list = safe_urls(resort)
    evaluator.add_custom_node(
        result=len(urls_list) >= 1,
        id=f"resort_{index+1}_urls_provided",
        desc="One or more reference URLs are provided that collectively substantiate that this resort meets the stated constraints.",
        parent=req_node,
        critical=True
    )

    # Constraints aggregator (critical)
    cons_node = evaluator.add_parallel(
        id=f"resort_{index+1}_constraints",
        desc="This resort satisfies all stated location, accommodation, dining, pool, spa, family, water sports, business, and service/rating constraints.",
        parent=resort_node,
        critical=True
    )

    # Prepare leaf nodes and verification tuples for batch execution
    claims_and_sources: List[Tuple[str, List[str], Any, Optional[str]]] = []

    def add_leaf_and_claim(node_id: str, desc: str, claim: str, add_ins: Optional[str] = None):
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=cons_node,
            critical=True
        )
        claims_and_sources.append((claim, urls_list, leaf, add_ins or general_verification_instruction()))
        return leaf

    # Location and property type
    add_leaf_and_claim(
        node_id=f"resort_{index+1}_location_allowed_region",
        desc="Located on Mexico's Pacific coast and specifically in either Riviera Nayarit or Los Cabos.",
        claim=f"The resort '{name}' is located on Mexico's Pacific coast in either Riviera Nayarit or Los Cabos.",
        add_ins=build_location_instruction()
    )

    add_leaf_and_claim(
        node_id=f"resort_{index+1}_beachfront_private_access",
        desc="Luxury beachfront resort with direct beach access and a private or semi-private beach area.",
        claim=f"The resort '{name}' is a luxury beachfront property with direct beach access and has a private or semi-private beach area."
    )

    # Brand affiliation
    add_leaf_and_claim(
        node_id=f"resort_{index+1}_luxury_brand_affiliation",
        desc="Operated by or affiliated with a recognized luxury hotel brand.",
        claim=f"The resort '{name}' is operated by or affiliated with the luxury brand '{brand}'.",
        add_ins="Confirm that the resort is part of or operated by the named luxury brand or collection. Accept synonyms like 'managed by', 'a member of', or 'part of'."
    )

    # Accommodations
    add_leaf_and_claim(
        node_id=f"resort_{index+1}_min_100_rooms",
        desc="Has at least 100 guest rooms and/or suites.",
        claim=f"The resort '{name}' has at least 100 guest rooms and/or suites."
    )

    add_leaf_and_claim(
        node_id=f"resort_{index+1}_suites_private_terrace_balcony",
        desc="Offers suite accommodations with private terraces or balconies.",
        claim=f"The resort '{name}' offers suite accommodations with private terraces or balconies."
    )

    add_leaf_and_claim(
        node_id=f"resort_{index+1}_villas_private_plunge_pools",
        desc="Provides villas or residences with private plunge pools.",
        claim=f"The resort '{name}' provides villas or residences with private plunge pools."
    )

    # Dining
    add_leaf_and_claim(
        node_id=f"resort_{index+1}_at_least_4_restaurants",
        desc="Has at least 4 on-site restaurants.",
        claim=f"The resort '{name}' has at least four on-site restaurants."
    )

    add_leaf_and_claim(
        node_id=f"resort_{index+1}_24hr_inroom_dining",
        desc="Provides 24-hour in-room dining service.",
        claim=f"The resort '{name}' provides 24-hour in-room dining service.",
        add_ins="Look for '24-hour room service', 'in-room dining available 24/7', or equivalent phrasing."
    )

    add_leaf_and_claim(
        node_id=f"resort_{index+1}_beachfront_dining_venue",
        desc="Has at least one beachfront dining venue.",
        claim=f"The resort '{name}' offers at least one beachfront dining venue.",
        add_ins="Check for a restaurant or bar that is explicitly described as beachfront, on the beach, or directly on the shore."
    )

    # Pools
    add_leaf_and_claim(
        node_id=f"resort_{index+1}_at_least_3_pools",
        desc="Has at least 3 separate swimming pools on property.",
        claim=f"The resort '{name}' has at least three separate swimming pools on property."
    )

    add_leaf_and_claim(
        node_id=f"resort_{index+1}_adults_only_pool",
        desc="Has at least one adults-only pool.",
        claim=f"The resort '{name}' has at least one adults-only pool.",
        add_ins="Look for phrasing like 'adults-only pool', 'adult pool', or 'quiet adults pool'."
    )

    add_leaf_and_claim(
        node_id=f"resort_{index+1}_infinity_oceanview_pool",
        desc="Has at least one infinity-edge pool with ocean views.",
        claim=f"The resort '{name}' has at least one infinity-edge pool with ocean views.",
        add_ins="Accept 'infinity pool' or 'infinity-edge' and confirm ocean/sea views."
    )

    # Spa & Wellness
    add_leaf_and_claim(
        node_id=f"resort_{index+1}_spa_10_treatment_rooms",
        desc="Has a full-service spa facility with at least 10 treatment rooms.",
        claim=f"The resort '{name}' has a full-service spa with at least ten treatment rooms.",
        add_ins="Look for a spa description, floor plan, or brochure indicating number of treatment rooms."
    )

    add_leaf_and_claim(
        node_id=f"resort_{index+1}_fitness_cardio_strength",
        desc="Has a fitness center equipped with cardiovascular and strength training equipment.",
        claim=f"The resort '{name}' has a fitness center equipped with both cardiovascular and strength training equipment."
    )

    # Family amenities
    add_leaf_and_claim(
        node_id=f"resort_{index+1}_kids_club_age_4_or_younger",
        desc="Offers a supervised kids club program that accepts children starting at age 4 or younger.",
        claim=f"The resort '{name}' has a supervised kids club that accepts children starting at age four or younger.",
        add_ins="Accept age 3 or 4 as satisfying the requirement. Verify minimum age explicitly."
    )

    add_leaf_and_claim(
        node_id=f"resort_{index+1}_family_friendly_pool_area",
        desc="Has at least one family-friendly pool area.",
        claim=f"The resort '{name}' has at least one family-friendly pool area.",
        add_ins="Look for 'family pool', 'children's pool', 'kid-friendly pool', or similar language."
    )

    # Water sports
    add_leaf_and_claim(
        node_id=f"resort_{index+1}_water_sports_available",
        desc="Water sports equipment and activities are available.",
        claim=f"The resort '{name}' offers water sports equipment and activities."
    )

    add_leaf_and_claim(
        node_id=f"resort_{index+1}_snorkeling_equipment_provided",
        desc="Snorkeling equipment is provided.",
        claim=f"The resort '{name}' provides snorkeling equipment."
    )

    add_leaf_and_claim(
        node_id=f"resort_{index+1}_onsite_dive_or_padi",
        desc="Has an on-site dive center or a PADI-certified diving program.",
        claim=f"The resort '{name}' has an on-site dive center or offers a PADI-certified diving program.",
        add_ins="Accept a fully on-site dive center or an official PADI program operated directly from the resort."
    )

    # Business facilities
    add_leaf_and_claim(
        node_id=f"resort_{index+1}_meeting_10000_sqft",
        desc="Has meeting/conference facilities with at least 10,000 square feet of total space.",
        claim=f"The resort '{name}' has meeting or conference facilities totaling at least 10,000 square feet.",
        add_ins="Look for meeting/event fact sheets or floor plans indicating total meeting space >= 10,000 sq ft."
    )

    # Service standards
    add_leaf_and_claim(
        node_id=f"resort_{index+1}_24hr_concierge",
        desc="Provides 24-hour concierge services.",
        claim=f"The resort '{name}' provides 24-hour concierge services.",
        add_ins="Look for '24-hour concierge', 'round-the-clock concierge', or equivalent."
    )

    add_leaf_and_claim(
        node_id=f"resort_{index+1}_butler_service",
        desc="Offers butler service for suite or villa guests.",
        claim=f"The resort '{name}' offers butler service for suite or villa guests.",
        add_ins="Look for 'butler', 'personal butler', 'butler service' associated with suites or villas."
    )

    add_leaf_and_claim(
        node_id=f"resort_{index+1}_aaa_or_forbes_rating",
        desc="Has AAA Four Diamond or higher, OR Forbes Travel Guide 4-Star or higher rating.",
        claim=f"The resort '{name}' holds either AAA Four Diamond (or higher) or Forbes Travel Guide 4-Star (or higher) rating.",
        add_ins="Evidence may be on AAA or Forbes websites, or official resort/brand pages citing the rating. Accept either AAA ≥ Four Diamond or Forbes ≥ 4-Star."
    )

    # Execute all constraint verifications in parallel
    if claims_and_sources:
        await evaluator.batch_verify(claims_and_sources)


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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

    task_node = evaluator.add_parallel(
        id="Four_Luxury_Pacific_Coast_Resorts",
        desc="Identify four distinct luxury beachfront resorts in either Riviera Nayarit or Los Cabos that each satisfy all stated amenity/service constraints, and provide the required identifying fields and supporting URLs.",
        parent=root,
        critical=False  # Allow partial credit across the four resorts
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_resorts(),
        template_class=ResortsExtraction,
        extraction_name="resorts_extraction"
    )

    # Ensure we only consider first four entries; pad if fewer
    resorts = extracted.resorts[:4]
    while len(resorts) < 4:
        resorts.append(ResortEntry())

    # Check "Exactly four resorts are provided and distinct"
    distinct_ok, provided_names = first_four_distinct_names(resorts)
    evaluator.add_custom_node(
        result=distinct_ok,
        id="Four_Distinct_Resorts_Provided",
        desc="Exactly four resorts are provided and they are distinct (no duplicates).",
        parent=task_node,
        critical=True
    )

    # Verify each resort
    for i, resort in enumerate(resorts):
        await verify_resort(evaluator, task_node, resort, i)

    return evaluator.get_summary()