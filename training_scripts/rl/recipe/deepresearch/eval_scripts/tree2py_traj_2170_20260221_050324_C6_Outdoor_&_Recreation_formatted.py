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
TASK_ID = "backcountry_western_np"
TASK_DESCRIPTION = """
Identify one backcountry camping location in a western United States national park that satisfies all requirements:
1) wilderness/backcountry permit via recreation.gov, 2) group capacity of at least 8 people, 3) within a western U.S. national park, 
4) campfires permitted during summer season (not prohibited by elevation/area), 5) stay ≥2 consecutive nights allowed, 
6) access via maintained trails, 7) designated site OR camping complies with water proximity rules.
Provide: location name, national park, state, and official reference URLs (recreation.gov and/or NPS) verifying each requirement.
"""

WESTERN_STATES = {
    "WA", "OR", "CA", "NV", "AZ", "UT", "CO", "NM",
    "WY", "MT", "ID", "AK", "HI"
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RequirementURLs(BaseModel):
    location_reference_urls: List[str] = Field(default_factory=list)
    recreation_gov_permit_url: Optional[str] = None
    permit_type_reference_urls: List[str] = Field(default_factory=list)
    advance_reservation_urls: List[str] = Field(default_factory=list)
    group_capacity_reference_urls: List[str] = Field(default_factory=list)
    trail_access_reference_urls: List[str] = Field(default_factory=list)
    campfire_reference_urls: List[str] = Field(default_factory=list)
    stay_duration_reference_urls: List[str] = Field(default_factory=list)
    water_proximity_reference_urls: List[str] = Field(default_factory=list)


class CampingLocationExtraction(BaseModel):
    location_name: Optional[str] = None
    park_name: Optional[str] = None
    state: Optional[str] = None
    urls: RequirementURLs = Field(default_factory=RequirementURLs)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_location() -> str:
    return """
    Extract the single backcountry camping location and the official reference URLs provided in the answer.

    Fields to return:
    - location_name: The specific name of the backcountry camping location (campsite, zone, or area)
    - park_name: The name of the U.S. National Park where the location is situated
    - state: The U.S. state for the park (use the standard two-letter state abbreviation if available; otherwise the state name)
    - urls:
        - location_reference_urls: All official URLs confirming the location’s existence within the park (prefer nps.gov or recreation.gov)
        - recreation_gov_permit_url: The recreation.gov URL for the wilderness/backcountry permit listing or reservation page, if present
        - permit_type_reference_urls: Official URLs confirming this is a wilderness/backcountry overnight camping permit (NPS or recreation.gov)
        - advance_reservation_urls: Official URL(s) confirming that advance reservations are available (not walk-up only)
        - group_capacity_reference_urls: Official URL(s) confirming the group size allowance (must allow ≥8 people)
        - trail_access_reference_urls: Official URL(s) confirming access via maintained trails
        - campfire_reference_urls: Official URL(s) confirming campfire allowance during summer (not prohibited by elevation/area restrictions)
        - stay_duration_reference_urls: Official URL(s) confirming that at least 2 consecutive nights are allowed at the same zone/site
        - water_proximity_reference_urls: Official URL(s) confirming compliance with water proximity rules (designated campsite or distance rules)

    Extraction rules:
    - Only extract URLs that appear explicitly in the answer.
    - Prefer official sources (recreation.gov or National Park Service: nps.gov).
    - If any field is not present in the answer, set it to null or an empty list as appropriate.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_official_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.lower().strip()
    return ("recreation.gov" in u) or ("nps.gov" in u)


def official_urls_only(urls: Optional[List[str]]) -> List[str]:
    urls = urls or []
    return [u for u in urls if is_official_url(u)]


def merge_official_sources(*url_lists: List[str], extra_single: Optional[str] = None) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        merged.extend(official_urls_only(lst))
    if extra_single and is_official_url(extra_single):
        merged.append(extra_single)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for u in merged:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_location_identification(
    evaluator: Evaluator,
    parent_node,
    info: CampingLocationExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Location_Identification",
        desc="Provide basic identifying information about the backcountry camping location",
        parent=parent_node,
        critical=True
    )

    # Basic info presence + western state check
    state = (info.state or "").strip()
    name_ok = bool(info.location_name and info.location_name.strip())
    park_ok = bool(info.park_name and info.park_name.strip())
    state_ok = bool(state) and (
        state.upper() in WESTERN_STATES or state.title() in {
            "Washington", "Oregon", "California", "Nevada", "Arizona", "Utah",
            "Colorado", "New Mexico", "Wyoming", "Montana", "Idaho", "Alaska", "Hawaii"
        }
    )

    evaluator.add_custom_node(
        result=(name_ok and park_ok and state_ok),
        id="Basic_Location_Information",
        desc="The location name, national park name, and western U.S. state are provided",
        parent=node,
        critical=True
    )

    # Verify existence and park-state association via official references
    loc_ref_leaf = evaluator.add_leaf(
        id="Location_Reference_URL",
        desc="A reference URL from an official park or recreation source confirming the location's existence",
        parent=node,
        critical=True
    )

    loc_sources = official_urls_only(info.urls.location_reference_urls)
    claim = (
        f"The backcountry location '{info.location_name}' exists within {info.park_name} "
        f"in {info.state}, according to official sources."
    )
    await evaluator.verify(
        claim=claim,
        node=loc_ref_leaf,
        sources=loc_sources,
        additional_instruction=(
            "Verify that the provided official page(s) (NPS or recreation.gov) explicitly mention "
            "this location (zone/site/area) within the stated national park."
        )
    )


async def build_permit_system_verification(
    evaluator: Evaluator,
    parent_node,
    info: CampingLocationExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Permit_System_Verification",
        desc="Verify that the location has proper wilderness permit requirements and reservation system",
        parent=parent_node,
        critical=True
    )

    # Recreation.gov listing
    rg_node = evaluator.add_parallel(
        id="Recreation_Gov_Listing",
        desc="The location's wilderness permit is available through recreation.gov",
        parent=node,
        critical=True
    )

    # Existence of recreation.gov URL (official)
    rec_url_present = bool(info.urls.recreation_gov_permit_url) and is_official_url(info.urls.recreation_gov_permit_url)
    evaluator.add_custom_node(
        result=rec_url_present,
        id="Recreation_Gov_Reference_URL",
        desc="A reference URL from recreation.gov showing the permit listing",
        parent=rg_node,
        critical=True
    )

    listed_leaf = evaluator.add_leaf(
        id="Listed_On_Recreation_Gov",
        desc="The specific wilderness permit or backcountry camping reservation is listed on recreation.gov",
        parent=rg_node,
        critical=True
    )
    listed_claim = (
        f"The wilderness/backcountry permit for {info.park_name} covering '{info.location_name}' "
        f"is available on recreation.gov."
    )
    await evaluator.verify(
        claim=listed_claim,
        node=listed_leaf,
        sources=info.urls.recreation_gov_permit_url,
        additional_instruction=(
            "Confirm this recreation.gov page is a permit/reservation listing related to wilderness/backcountry overnight use. "
            "It may be park-wide or zone-based, but should cover the cited location."
        )
    )

    # Wilderness permit type
    wpt_node = evaluator.add_parallel(
        id="Wilderness_Permit_Type",
        desc="The permit type is identified as a wilderness or backcountry camping permit",
        parent=node,
        critical=True
    )

    type_sources = merge_official_sources(
        info.urls.permit_type_reference_urls,
        extra_single=info.urls.recreation_gov_permit_url
    )

    evaluator.add_custom_node(
        result=bool(type_sources),
        id="Permit_Type_Reference_URL",
        desc="A reference URL confirming the wilderness/backcountry permit requirement",
        parent=wpt_node,
        critical=True
    )

    type_leaf = evaluator.add_leaf(
        id="Wilderness_Permit_Confirmed",
        desc="The permit is confirmed to be for wilderness or backcountry overnight camping",
        parent=wpt_node,
        critical=True
    )
    type_claim = (
        f"The permit for {info.park_name} is a wilderness/backcountry permit for overnight camping."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=type_sources,
        additional_instruction=(
            "Check the official page(s) to confirm the permit pertains to wilderness/backcountry overnight camping, not day-use only."
        )
    )

    # Advance reservation
    advance_leaf = evaluator.add_leaf(
        id="Advance_Reservation_Available",
        desc="Advance reservations are available for this location (not walk-up only)",
        parent=node,
        critical=True
    )
    advance_sources = merge_official_sources(
        info.urls.advance_reservation_urls,
        extra_single=info.urls.recreation_gov_permit_url
    )
    advance_claim = (
        "Advance reservations for the wilderness/backcountry permit are available (not restricted to walk-up only)."
    )
    await evaluator.verify(
        claim=advance_claim,
        node=advance_leaf,
        sources=advance_sources,
        additional_instruction=(
            "Look for booking windows, reservation instructions, or language indicating advance reservation capability on official pages."
        )
    )


async def build_physical_requirements(
    evaluator: Evaluator,
    parent_node,
    info: CampingLocationExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Physical_Requirements",
        desc="Verify that the location meets physical access and group capacity requirements",
        parent=parent_node,
        critical=True
    )

    # Group capacity
    gc_node = evaluator.add_parallel(
        id="Group_Capacity",
        desc="The location can accommodate a group of at least 8 people for overnight camping",
        parent=node,
        critical=True
    )

    gc_sources = official_urls_only(info.urls.group_capacity_reference_urls)
    evaluator.add_custom_node(
        result=bool(gc_sources),
        id="Group_Capacity_Reference_URL",
        desc="A reference URL from official park regulations confirming the group size allowance",
        parent=gc_node,
        critical=True
    )

    gc_leaf = evaluator.add_leaf(
        id="Eight_Plus_People_Accommodation",
        desc="The location's group size regulations allow for groups of 8 or more people",
        parent=gc_node,
        critical=True
    )
    gc_claim = (
        f"The backcountry camping rules for '{info.location_name}' in {info.park_name} allow groups of at least 8 people overnight."
    )
    await evaluator.verify(
        claim=gc_claim,
        node=gc_leaf,
        sources=gc_sources,
        additional_instruction=(
            "Verify group size limits in official regulations or zone/site rules; confirm that ≥8 people are permitted for overnight camping."
        )
    )

    # Trail access
    ta_node = evaluator.add_parallel(
        id="Trail_Access",
        desc="The location is accessible via maintained trails",
        parent=node,
        critical=True
    )

    ta_sources = official_urls_only(info.urls.trail_access_reference_urls)
    evaluator.add_custom_node(
        result=bool(ta_sources),
        id="Trail_Access_Reference_URL",
        desc="A reference URL confirming the trail access information",
        parent=ta_node,
        critical=True
    )

    ta_leaf = evaluator.add_leaf(
        id="Maintained_Trail_Route",
        desc="The location is reached by maintained trails (not requiring off-trail navigation for primary approach)",
        parent=ta_node,
        critical=True
    )
    ta_claim = (
        f"The primary approach route to '{info.location_name}' uses maintained trails."
    )
    await evaluator.verify(
        claim=ta_claim,
        node=ta_leaf,
        sources=ta_sources,
        additional_instruction=(
            "Look for official trail descriptions, maps, or guidance indicating maintained trails are the primary access."
        )
    )


async def build_regulations_compliance(
    evaluator: Evaluator,
    parent_node,
    info: CampingLocationExtraction
) -> None:
    node = evaluator.add_parallel(
        id="Camping_Regulations_Compliance",
        desc="Verify that the location meets all camping regulation requirements",
        parent=parent_node,
        critical=True
    )

    # Fire and duration rules
    fd_node = evaluator.add_parallel(
        id="Fire_And_Duration_Rules",
        desc="Verify compliance with campfire and stay duration regulations",
        parent=node,
        critical=True
    )

    # Campfire regulations
    cf_node = evaluator.add_parallel(
        id="Campfire_Regulations",
        desc="Campfires are permitted at the camping location and elevation",
        parent=fd_node,
        critical=True
    )

    cf_sources = official_urls_only(info.urls.campfire_reference_urls)
    evaluator.add_custom_node(
        result=bool(cf_sources),
        id="Campfire_Reference_URL",
        desc="A reference URL from official park regulations confirming campfire allowance",
        parent=cf_node,
        critical=True
    )

    cf_leaf = evaluator.add_leaf(
        id="Campfire_Permitted",
        desc="Campfires are allowed at this location during summer season (not prohibited by elevation or area-specific restrictions)",
        parent=cf_node,
        critical=True
    )
    cf_claim = (
        f"Campfires are permitted at '{info.location_name}' during the summer season, "
        "and not prohibited by elevation-based or area-specific restrictions."
    )
    await evaluator.verify(
        claim=cf_claim,
        node=cf_leaf,
        sources=cf_sources,
        additional_instruction=(
            "Confirm summer-season campfire allowance in official regulations; "
            "if elevation/area restrictions apply, the location should still be permitted."
        )
    )

    # Stay duration regulations
    sd_node = evaluator.add_parallel(
        id="Stay_Duration_Regulations",
        desc="The location allows camping for at least 2 consecutive nights",
        parent=fd_node,
        critical=True
    )

    sd_sources = official_urls_only(info.urls.stay_duration_reference_urls)
    evaluator.add_custom_node(
        result=bool(sd_sources),
        id="Duration_Reference_URL",
        desc="A reference URL from official park regulations confirming the stay duration allowance",
        parent=sd_node,
        critical=True
    )

    sd_leaf = evaluator.add_leaf(
        id="Two_Plus_Nights_Allowed",
        desc="The location or zone allows camping for 2 or more consecutive nights (no 1-night maximum restriction)",
        parent=sd_node,
        critical=True
    )
    sd_claim = (
        f"{info.park_name} allows camping for at least two consecutive nights at '{info.location_name}' "
        "in the same zone/designated site (no 1-night maximum)."
    )
    await evaluator.verify(
        claim=sd_claim,
        node=sd_leaf,
        sources=sd_sources,
        additional_instruction=(
            "Find official rules indicating a minimum stay allowance of ≥2 consecutive nights for the zone/site."
        )
    )

    # Water proximity rules
    wp_node = evaluator.add_parallel(
        id="Water_Proximity_Rules",
        desc="Verify compliance with water proximity camping regulations",
        parent=node,
        critical=True
    )

    wdc_node = evaluator.add_parallel(
        id="Water_Distance_Compliance",
        desc="The camping location complies with water proximity regulations",
        parent=wp_node,
        critical=True
    )

    wp_sources = official_urls_only(info.urls.water_proximity_reference_urls)
    evaluator.add_custom_node(
        result=bool(wp_sources),
        id="Water_Proximity_Reference_URL",
        desc="A reference URL from official park regulations confirming water proximity compliance",
        parent=wdc_node,
        critical=True
    )

    wp_leaf = evaluator.add_leaf(
        id="Water_Distance_Rules_Met",
        desc="The location is either a designated campsite or allows camping in compliance with water distance rules (e.g., not within prohibited distance of water sources)",
        parent=wdc_node,
        critical=True
    )
    wp_claim = (
        f"Camping at '{info.location_name}' complies with the park's water proximity regulations "
        "(either a designated campsite or campsites must be at the required distance from water)."
    )
    await evaluator.verify(
        claim=wp_claim,
        node=wp_leaf,
        sources=wp_sources,
        additional_instruction=(
            "Confirm designated site status OR explicit water-distance rules that apply to the location."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the backcountry camping location task.
    """
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

    # Extract structured information from the answer
    extracted: CampingLocationExtraction = await evaluator.extract(
        prompt=prompt_extract_location(),
        template_class=CampingLocationExtraction,
        extraction_name="camping_location_extraction"
    )

    # Top-level critical node representing the entire rubric
    top = evaluator.add_parallel(
        id="Backcountry_Camping_Location_Identification",
        desc="Identify one suitable backcountry camping location in a western U.S. national park that meets all specified requirements for group camping",
        parent=root,
        critical=True
    )

    # Record helpful custom info
    evaluator.add_custom_info(
        info={"western_states_set": sorted(list(WESTERN_STATES))},
        info_type="context",
        info_name="western_states_reference"
    )

    # Build all critical subtrees
    await build_location_identification(evaluator, top, extracted)
    await build_permit_system_verification(evaluator, top, extracted)
    await build_physical_requirements(evaluator, top, extracted)
    await build_regulations_compliance(evaluator, top, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()