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
TASK_ID = "ca_state_park_campground"
TASK_DESCRIPTION = """
Identify a California State Park that offers year-round camping and meets ALL of the following requirements: 
(1) part of the California State Parks system; 
(2) all campsites include fire rings and picnic tables; 
(3) maximum campsite capacity is 8 people per site; 
(4) maximum 2 vehicles per campsite (including towed vehicles); 
(5) maximum 2 tents per campsite; 
(6) at least 2 ADA accessible campsites; 
(7) accessible toilets and showers; 
(8) check-in time is 2:00 PM; 
(9) check-out time is 12:00 PM (noon); 
(10) accepts reservations through ReserveCalifornia (phone: 1-800-444-7275 or online); 
(11) dogs are allowed with a maximum 6-foot leash requirement; 
(12) dogs must be kept in a tent or enclosed vehicle at night; 
(13) token-operated or coin-operated showers are available. 
What is the name of this park?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkCampgroundExtraction(BaseModel):
    """Structured extraction of the identified park/campground and all referenced URLs per requirement."""
    park_name: Optional[str] = None
    campground_name: Optional[str] = None

    # General official URLs for this park or campground (e.g., CA State Parks official page, ReserveCalifornia page)
    official_urls: List[str] = Field(default_factory=list)

    # Requirement-specific URLs cited in the answer (extract only explicit URLs)
    state_parks_system_urls: List[str] = Field(default_factory=list)
    year_round_urls: List[str] = Field(default_factory=list)
    fire_rings_urls: List[str] = Field(default_factory=list)
    picnic_tables_urls: List[str] = Field(default_factory=list)
    max_occupancy_urls: List[str] = Field(default_factory=list)
    max_vehicles_urls: List[str] = Field(default_factory=list)
    max_tents_urls: List[str] = Field(default_factory=list)
    ada_campsites_urls: List[str] = Field(default_factory=list)
    accessible_facilities_urls: List[str] = Field(default_factory=list)
    check_in_urls: List[str] = Field(default_factory=list)
    check_out_urls: List[str] = Field(default_factory=list)
    reserve_ca_urls: List[str] = Field(default_factory=list)
    dogs_leash_urls: List[str] = Field(default_factory=list)
    dogs_night_urls: List[str] = Field(default_factory=list)
    token_showers_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_park_campground_info() -> str:
    return """
    From the provided answer, extract the identified California State Park and/or the specific campground name.
    Additionally, extract all URLs explicitly mentioned in the answer that support each of the listed requirements.
    Only extract URLs that are explicitly present in the answer (plain text URLs or markdown links). Do not invent URLs.

    Return a JSON object with the following fields:
    - park_name: The name of the California State Park (if explicitly mentioned in the answer); else null.
    - campground_name: The name of the specific campground (if explicitly mentioned); else null.
    - official_urls: An array of general official URLs for the park or campground, such as a California State Parks unit page or a ReserveCalifornia page, if present in the answer.

    For each requirement below, extract the URLs cited in the answer that support that requirement. 
    If none are provided in the answer, return an empty array for that field.

    - state_parks_system_urls: URLs supporting that the park/campground is part of the California State Parks system.
    - year_round_urls: URLs supporting that camping is available year-round (e.g., "open all year").
    - fire_rings_urls: URLs supporting that all campsites have fire rings (e.g., "fire rings at each site").
    - picnic_tables_urls: URLs supporting that all campsites have picnic tables.
    - max_occupancy_urls: URLs supporting maximum campsite capacity is 8 people per site.
    - max_vehicles_urls: URLs supporting maximum 2 vehicles per campsite (including towed vehicles).
    - max_tents_urls: URLs supporting maximum 2 tents per campsite.
    - ada_campsites_urls: URLs supporting at least 2 ADA accessible campsites exist.
    - accessible_facilities_urls: URLs supporting accessible toilets and accessible showers.
    - check_in_urls: URLs supporting the check-in time is 2:00 PM.
    - check_out_urls: URLs supporting the check-out time is 12:00 PM (noon).
    - reserve_ca_urls: URLs supporting that reservations are accepted through ReserveCalifornia (online).
    - dogs_leash_urls: URLs supporting that dogs are allowed and must be on a leash no longer than 6 feet.
    - dogs_night_urls: URLs supporting that dogs must be kept in a tent or enclosed vehicle at night.
    - token_showers_urls: URLs supporting that showers are token-operated or coin-operated.

    If any field is not present in the answer, return null for string fields and an empty array for URL arrays.
    Remember:
    - Only extract URLs that are explicitly present in the answer text.
    - Do not infer or create URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _merge_urls(*lists: List[List[str]]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        merged.extend(lst)
    return _dedup_preserve_order(merged)


def _entity_name(info: ParkCampgroundExtraction) -> str:
    return (info.campground_name or info.park_name or "the identified park or campground").strip()


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, info: ParkCampgroundExtraction) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """
    # Create the rubric root node (critical, parallel aggregation)
    root_node = evaluator.add_parallel(
        id="California_State_Park_Campground_Verification",
        desc="Verify the identified California State Park campground meets all specified requirements",
        parent=evaluator.root,
        critical=True
    )

    # Existence check: ensure a park/campground name is identified
    name_exists = bool(info.park_name or info.campground_name)
    evaluator.add_custom_node(
        result=name_exists,
        id="Park_Name_Identified",
        desc="Park or campground name is identified in the answer",
        parent=root_node,
        critical=True
    )

    name = _entity_name(info)

    # Prepare per-requirement leaves, claims, and sources
    claims_and_sources: List[tuple[str, List[str] | None, Any, Optional[str]]] = []

    # 1) State Parks System
    node_sps = evaluator.add_leaf(
        id="State_Parks_System",
        desc="The campground must be part of the California State Parks system",
        parent=root_node,
        critical=True
    )
    claim_sps = f"{name} is part of the California State Parks system (a unit managed by California State Parks)."
    sources_sps = _merge_urls(info.state_parks_system_urls, info.official_urls)
    add_ins_sps = "Confirm that this park/campground is a California State Parks unit. The official CA State Parks unit page counts as sufficient evidence."
    claims_and_sources.append((claim_sps, sources_sps if sources_sps else None, node_sps, add_ins_sps))

    # 2) Year-round camping
    node_yr = evaluator.add_leaf(
        id="Year_Round_Camping",
        desc="The campground must offer year-round camping availability",
        parent=root_node,
        critical=True
    )
    claim_yr = f"{name} offers year-round camping (open all year)."
    sources_yr = _merge_urls(info.year_round_urls, info.official_urls, info.reserve_ca_urls)
    add_ins_yr = "Look for phrases such as 'open year-round', 'open all year', or equivalent wording specific to this campground."
    claims_and_sources.append((claim_yr, sources_yr if sources_yr else None, node_yr, add_ins_yr))

    # 3) Fire rings
    node_fr = evaluator.add_leaf(
        id="Fire_Rings_Provided",
        desc="All campsites must include fire rings",
        parent=root_node,
        critical=True
    )
    claim_fr = f"All campsites at {name} include fire rings (or fire pits)."
    sources_fr = _merge_urls(info.fire_rings_urls, info.official_urls, info.reserve_ca_urls)
    add_ins_fr = "Verify 'fire rings at each site' or equivalent (e.g., 'fire pit provided at every campsite')."
    claims_and_sources.append((claim_fr, sources_fr if sources_fr else None, node_fr, add_ins_fr))

    # 4) Picnic tables
    node_pt = evaluator.add_leaf(
        id="Picnic_Tables_Provided",
        desc="All campsites must include picnic tables",
        parent=root_node,
        critical=True
    )
    claim_pt = f"All campsites at {name} include picnic tables."
    sources_pt = _merge_urls(info.picnic_tables_urls, info.official_urls, info.reserve_ca_urls)
    add_ins_pt = "Confirm a statement equivalent to 'picnic table at each site' or 'all sites have picnic tables'."
    claims_and_sources.append((claim_pt, sources_pt if sources_pt else None, node_pt, add_ins_pt))

    # 5) Max occupancy 8 people
    node_occ = evaluator.add_leaf(
        id="Maximum_Occupancy_Eight",
        desc="Maximum campsite capacity must be 8 people per site",
        parent=root_node,
        critical=True
    )
    claim_occ = "The maximum campsite capacity is 8 people per site."
    sources_occ = _merge_urls(info.max_occupancy_urls, info.official_urls, info.reserve_ca_urls)
    add_ins_occ = "Accept phrasing like 'up to 8 persons per campsite' or 'max occupancy 8'."
    claims_and_sources.append((claim_occ, sources_occ if sources_occ else None, node_occ, add_ins_occ))

    # 6) Max two vehicles (including towed)
    node_veh = evaluator.add_leaf(
        id="Maximum_Two_Vehicles",
        desc="Maximum vehicle limit must be 2 vehicles per campsite (including towed vehicles)",
        parent=root_node,
        critical=True
    )
    claim_veh = "The maximum vehicles per campsite is 2, including towed vehicles/trailers."
    sources_veh = _merge_urls(info.max_vehicles_urls, info.official_urls, info.reserve_ca_urls)
    add_ins_veh = "Phrasing like '2 vehicles max per campsite (including towed)' or '2 vehicle limit' should be treated as equivalent."
    claims_and_sources.append((claim_veh, sources_veh if sources_veh else None, node_veh, add_ins_veh))

    # 7) Max two tents
    node_tents = evaluator.add_leaf(
        id="Maximum_Two_Tents",
        desc="Maximum tent limit must be 2 tents per campsite",
        parent=root_node,
        critical=True
    )
    claim_tents = "A maximum of 2 tents are allowed per campsite."
    sources_tents = _merge_urls(info.max_tents_urls, info.official_urls, info.reserve_ca_urls)
    add_ins_tents = "Treat 'limit 2 tents per site' as equivalent."
    claims_and_sources.append((claim_tents, sources_tents if sources_tents else None, node_tents, add_ins_tents))

    # 8) At least 2 ADA accessible campsites
    node_ada = evaluator.add_leaf(
        id="ADA_Accessible_Campsites",
        desc="Must have at least 2 ADA accessible campsites",
        parent=root_node,
        critical=True
    )
    claim_ada = f"{name} provides at least two ADA-accessible campsites."
    sources_ada = _merge_urls(info.ada_campsites_urls, info.official_urls, info.reserve_ca_urls)
    add_ins_ada = "Confirm ADA/accessible sites count is 2 or more."
    claims_and_sources.append((claim_ada, sources_ada if sources_ada else None, node_ada, add_ins_ada))

    # 9) Accessible toilets and showers
    node_access_fac = evaluator.add_leaf(
        id="Accessible_Facilities",
        desc="Must have accessible toilets and showers",
        parent=root_node,
        critical=True
    )
    claim_access_fac = f"{name} has accessible toilets and accessible showers."
    sources_access_fac = _merge_urls(info.accessible_facilities_urls, info.official_urls, info.reserve_ca_urls)
    add_ins_access_fac = "Look for accessibility indications for restrooms and showers; both should be accessible."
    claims_and_sources.append((claim_access_fac, sources_access_fac if sources_access_fac else None, node_access_fac, add_ins_access_fac))

    # 10) Check-in 2:00 PM
    node_check_in = evaluator.add_leaf(
        id="Check_In_Time_2PM",
        desc="Check-in time must be 2:00 PM",
        parent=root_node,
        critical=True
    )
    claim_check_in = "The check-in time is 2:00 PM."
    sources_check_in = _merge_urls(info.check_in_urls, info.official_urls, info.reserve_ca_urls)
    add_ins_check_in = "Confirm check-in time equals 2:00 PM; accept equivalent formatting (e.g., 2 PM)."
    claims_and_sources.append((claim_check_in, sources_check_in if sources_check_in else None, node_check_in, add_ins_check_in))

    # 11) Check-out 12:00 PM (noon)
    node_check_out = evaluator.add_leaf(
        id="Check_Out_Time_Noon",
        desc="Check-out time must be 12:00 PM (noon)",
        parent=root_node,
        critical=True
    )
    claim_check_out = "The check-out time is 12:00 PM (noon)."
    sources_check_out = _merge_urls(info.check_out_urls, info.official_urls, info.reserve_ca_urls)
    add_ins_check_out = "Confirm '12:00 PM', '12 PM', or 'noon' for check-out."
    claims_and_sources.append((claim_check_out, sources_check_out if sources_check_out else None, node_check_out, add_ins_check_out))

    # 12) ReserveCalifornia reservations
    node_reserve = evaluator.add_leaf(
        id="ReserveCalifornia_System",
        desc="Must accept reservations through ReserveCalifornia (phone: 1-800-444-7275 or online)",
        parent=root_node,
        critical=True
    )
    claim_reserve = f"{name} accepts reservations through ReserveCalifornia."
    sources_reserve = _merge_urls(info.reserve_ca_urls, info.official_urls)
    add_ins_reserve = "It is sufficient if the page indicates reservations are made via ReserveCalifornia (online or phone)."
    claims_and_sources.append((claim_reserve, sources_reserve if sources_reserve else None, node_reserve, add_ins_reserve))

    # 13) Dogs allowed, leash <= 6 feet
    node_leash = evaluator.add_leaf(
        id="Dogs_Allowed_Six_Foot_Leash",
        desc="Dogs must be allowed with a maximum 6-foot leash requirement",
        parent=root_node,
        critical=True
    )
    claim_leash = "Dogs are allowed and must be kept on a leash no longer than 6 feet."
    sources_leash = _merge_urls(info.dogs_leash_urls, info.official_urls, info.reserve_ca_urls)
    add_ins_leash = "Verify dogs are permitted and the leash length restriction is 6 feet maximum."
    claims_and_sources.append((claim_leash, sources_leash if sources_leash else None, node_leash, add_ins_leash))

    # 14) Dogs night containment
    node_dog_night = evaluator.add_leaf(
        id="Dog_Night_Containment",
        desc="Dogs must be kept in a tent or enclosed vehicle at night",
        parent=root_node,
        critical=True
    )
    claim_dog_night = "At night, dogs must be kept in a tent or an enclosed vehicle."
    sources_dog_night = _merge_urls(info.dogs_night_urls, info.official_urls, info.reserve_ca_urls)
    add_ins_dog_night = "Confirm an explicit rule about dogs being contained in a tent or enclosed vehicle during nighttime."
    claims_and_sources.append((claim_dog_night, sources_dog_night if sources_dog_night else None, node_dog_night, add_ins_dog_night))

    # 15) Token/coin-operated showers
    node_token = evaluator.add_leaf(
        id="Token_Operated_Showers",
        desc="Must have token-operated or coin-operated showers available",
        parent=root_node,
        critical=True
    )
    claim_token = f"{name} has token-operated or coin-operated showers."
    sources_token = _merge_urls(info.token_showers_urls, info.official_urls, info.reserve_ca_urls)
    add_ins_token = "Accept either 'coin-operated' or 'token-operated' showers as satisfying this requirement."
    claims_and_sources.append((claim_token, sources_token if sources_token else None, node_token, add_ins_token))

    # Execute verifications (parallelized)
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
    Evaluate an answer for the California State Park campground verification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured info from the answer
    extracted_info: ParkCampgroundExtraction = await evaluator.extract(
        prompt=prompt_extract_park_campground_info(),
        template_class=ParkCampgroundExtraction,
        extraction_name="park_campground_info"
    )

    # Add custom info for transparency
    entity = _entity_name(extracted_info)
    total_urls = sum([
        len(extracted_info.official_urls),
        len(extracted_info.state_parks_system_urls),
        len(extracted_info.year_round_urls),
        len(extracted_info.fire_rings_urls),
        len(extracted_info.picnic_tables_urls),
        len(extracted_info.max_occupancy_urls),
        len(extracted_info.max_vehicles_urls),
        len(extracted_info.max_tents_urls),
        len(extracted_info.ada_campsites_urls),
        len(extracted_info.accessible_facilities_urls),
        len(extracted_info.check_in_urls),
        len(extracted_info.check_out_urls),
        len(extracted_info.reserve_ca_urls),
        len(extracted_info.dogs_leash_urls),
        len(extracted_info.dogs_night_urls),
        len(extracted_info.token_showers_urls),
    ])
    evaluator.add_custom_info(
        info={"entity_name": entity, "total_extracted_urls": total_urls},
        info_type="extraction_stats",
        info_name="extraction_statistics"
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extracted_info)

    # Return structured evaluation summary
    return evaluator.get_summary()