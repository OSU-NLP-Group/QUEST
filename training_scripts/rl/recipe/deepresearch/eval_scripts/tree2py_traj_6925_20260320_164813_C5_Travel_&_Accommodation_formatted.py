import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "aruba_eagle_beach_family_ai_2026"
TASK_DESCRIPTION = """
Identify an all-inclusive resort on Eagle Beach in Aruba that meets the following requirements for a family reunion during Martin Luther King Jr. Day weekend 2026 (January 17-19, 2026): (1) the resort must be a beachfront property with direct beach access, (2) the resort must have an on-site kids club that accepts children starting at age 4 or younger, (3) the resort must feature multiple pools or special water features (such as lazy river, water slides, or children's pool), and (4) the resort must offer family suites or room configurations that can accommodate at least 6 guests. Provide the resort name and reference URLs supporting each requirement.
"""

MLK_2026_START = "January 17, 2026"
MLK_2026_END = "January 19, 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ResortExtraction(BaseModel):
    resort_name: Optional[str] = None

    # General / identity
    general_info_urls: List[str] = Field(default_factory=list)

    # Location & property type
    location_urls: List[str] = Field(default_factory=list)            # pages mentioning Eagle Beach location
    beachfront_urls: List[str] = Field(default_factory=list)          # pages stating beachfront / direct beach access
    all_inclusive_urls: List[str] = Field(default_factory=list)       # pages stating all-inclusive plans/options

    # Family amenities
    kids_club_urls: List[str] = Field(default_factory=list)           # pages stating kids club and age details
    kids_club_min_age_text: Optional[str] = None                      # any stated minimum age text from the answer
    pool_feature_urls: List[str] = Field(default_factory=list)        # pages for pools/water features
    pool_features_text: Optional[str] = None

    # Accommodation capacity
    capacity_urls: List[str] = Field(default_factory=list)            # pages stating family suites / sleep 6 config
    capacity_text: Optional[str] = None

    # Availability/booking
    availability_urls: List[str] = Field(default_factory=list)        # booking/calendar pages
    availability_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_resort() -> str:
    return """
    Extract structured information about the recommended Aruba resort from the answer.

    Required fields:
    - resort_name: The exact resort name as written in the answer (string or null if missing).
    - general_info_urls: URLs that directly identify the resort (official site, profile page, Google Maps, OTA, reputable travel sites). Array of URLs, can be empty.
    - location_urls: URLs that explicitly mention the resort is on Eagle Beach in Aruba. Array, can be empty.
    - beachfront_urls: URLs that explicitly say the resort is beachfront or has direct beach access. Array, can be empty.
    - all_inclusive_urls: URLs that explicitly say the resort offers all-inclusive packages (optional or full). Array, can be empty.
    - kids_club_urls: URLs that mention the on-site kids club and its age policy. Array, can be empty.
    - kids_club_min_age_text: The minimum starting age for the kids club as written in the answer (e.g., "ages 4–12", "from age 3", "ages 5-12"). Return null if not stated.
    - pool_feature_urls: URLs that show multiple pools or special water features (e.g., lazy river, water slides, or children's pool). Array, can be empty.
    - pool_features_text: Any wording from the answer describing pool/water features. Null if not stated.
    - capacity_urls: URLs that show family suites or room configurations that can accommodate at least 6 guests (e.g., 2-bedroom suite, villa, guaranteed connecting rooms for 6). Array, can be empty.
    - capacity_text: Any wording in the answer describing how 6 guests can be accommodated. Null if not stated.
    - availability_urls: URLs to the resort's official booking engine, calendar, or reliable OTA page where January 2026 dates can be checked. Array, can be empty.
    - availability_text: Any wording in the answer about MLK weekend 2026 availability. Null if not stated.

    Important:
    - Only extract URLs explicitly present in the answer text. Do not invent or infer URLs.
    - If a URL appears relevant to multiple categories, include it in all applicable arrays (duplication across arrays is OK).
    - If a field is not present in the answer, set it to null (for string fields) or [] (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if isinstance(u, str) and u.strip() and u.strip() not in seen:
                seen.add(u.strip())
                merged.append(u.strip())
    return merged


async def _verify_with_sources_or_fail(
    evaluator: Evaluator,
    node_id: str,
    desc: str,
    parent,
    claim: str,
    sources: List[str],
    critical: bool = True,
    additional_instruction: str = "None",
):
    """
    Create a leaf node and verify by URLs if available; if sources are missing, fail the node immediately.
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )

    # If any blocking preconditions already failed, evaluator.verify will skip automatically.
    # But if sources are missing, this should be treated as a failure (not a skip).
    if not sources:
        leaf.score = 0.0
        leaf.status = "failed"
        return False

    return await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def _build_resort_identification(
    evaluator: Evaluator,
    parent,
    ext: ResortExtraction,
):
    """
    Section: Resort_Exists_and_Identified (parallel, critical)
    """
    sect = evaluator.add_parallel(
        id="Resort_Exists_and_Identified",
        desc="A specific resort name is provided and can be verified",
        parent=parent,
        critical=True,
    )

    # Resort_Name_Provided (critical, custom check)
    evaluator.add_custom_node(
        result=bool(ext.resort_name and ext.resort_name.strip()),
        id="Resort_Name_Provided",
        desc="The answer includes the specific name of an Aruba resort",
        parent=sect,
        critical=True,
    )

    # Resort_Reference_URL (critical, verify)
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="Resort_Reference_URL",
        desc="A reference URL is provided that confirms the resort's existence and basic information",
        parent=sect,
        claim=f"This webpage is about the resort named '{ext.resort_name}' in Aruba and confirms basic resort information (e.g., overview, address, or amenities).",
        sources=_non_empty_urls(ext.general_info_urls),
        critical=True,
        additional_instruction="Accept official site pages, reputable travel sites, or OTA listings that clearly identify the resort by name.",
    )


async def _build_location_property_type(
    evaluator: Evaluator,
    parent,
    ext: ResortExtraction,
):
    """
    Section: Location_and_Property_Type (parallel, critical)
    """
    sect = evaluator.add_parallel(
        id="Location_and_Property_Type",
        desc="The resort meets location and property type requirements",
        parent=parent,
        critical=True,
    )

    # Eagle_Beach_Location
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="Eagle_Beach_Location",
        desc="The resort is located on Eagle Beach in Aruba",
        parent=sect,
        claim=f"The resort '{ext.resort_name}' is located on Eagle Beach in Aruba.",
        sources=_non_empty_urls(ext.location_urls or ext.general_info_urls),
        critical=True,
        additional_instruction="The page should explicitly mention 'Eagle Beach' for this resort (allow minor variants like 'on Eagle Beach' or 'at Eagle Beach').",
    )

    # Beachfront_Property
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="Beachfront_Property",
        desc="The resort is a beachfront property with direct beach access",
        parent=sect,
        claim="The resort is a beachfront property with direct beach access (the property borders the beach; not separated by a public road or long walk).",
        sources=_non_empty_urls(ext.beachfront_urls or ext.location_urls or ext.general_info_urls),
        critical=True,
        additional_instruction="Confirm that the property is directly on the beach. If a public road separates the property and beach, do not count as 'direct beach access'.",
    )

    # All_Inclusive_Packages
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="All_Inclusive_Packages",
        desc="The resort offers all-inclusive packages",
        parent=sect,
        claim="The resort offers all-inclusive packages (full AI or optional AI meal plan qualifies).",
        sources=_non_empty_urls(ext.all_inclusive_urls or ext.general_info_urls),
        critical=True,
        additional_instruction="Look for wording such as 'all-inclusive', 'AI plan', 'all-inclusive option', or 'all-inclusive resort'. Optional AI packages are acceptable.",
    )

    # Location_Reference_URL
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="Location_Reference_URL",
        desc="A reference URL confirms the location and property type details",
        parent=sect,
        claim="This page provides authoritative details about the resort's location (Eagle Beach) and/or property type (beachfront, all-inclusive offering).",
        sources=_non_empty_urls(ext.location_urls, ext.beachfront_urls, ext.all_inclusive_urls),
        critical=True,
        additional_instruction="Accept official site pages or reputable sources that explicitly mention Eagle Beach, beachfront nature, or all-inclusive packages for this resort.",
    )


async def _build_family_amenities(
    evaluator: Evaluator,
    parent,
    ext: ResortExtraction,
):
    """
    Section: Family_Amenities (parallel, critical)
    """
    sect = evaluator.add_parallel(
        id="Family_Amenities",
        desc="The resort provides necessary family-friendly amenities",
        parent=parent,
        critical=True,
    )

    # Kids_Club_Age_Requirement
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="Kids_Club_Age_Requirement",
        desc="The resort has a kids club that accepts children starting at age 4 or younger",
        parent=sect,
        claim="The resort has an on-site kids club that accepts children starting at age 4 or younger.",
        sources=_non_empty_urls(ext.kids_club_urls or ext.general_info_urls),
        critical=True,
        additional_instruction="Look for kids club/children's club/miniclub. Accept if the minimum age is 4, 3, 2, 1, or no minimum (e.g., 'ages 4–12', '3–12', 'from age 4'). If minimum age is 5+, this should fail.",
    )

    # Multiple_Pool_Facilities
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="Multiple_Pool_Facilities",
        desc="The resort features multiple pools or special water features (such as lazy river, water slides, or children's pool)",
        parent=sect,
        claim="The resort features multiple pools or notable water features such as a lazy river, waterslides, splash pad, or a children's pool.",
        sources=_non_empty_urls(ext.pool_feature_urls or ext.general_info_urls),
        critical=True,
        additional_instruction="Pass if there are two or more pools OR special features (slides, splash areas, lazy river).",
    )

    # Amenities_Reference_URL
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="Amenities_Reference_URL",
        desc="A reference URL confirms the kids club and pool facility details",
        parent=sect,
        claim="This page confirms kids club details (including minimum age) and/or pool/water features at the resort.",
        sources=_non_empty_urls(ext.kids_club_urls, ext.pool_feature_urls),
        critical=True,
        additional_instruction="Accept official site pages or reputable sources listing family amenities like kids club and pool facilities.",
    )


async def _build_accommodation_capacity(
    evaluator: Evaluator,
    parent,
    ext: ResortExtraction,
):
    """
    Section: Accommodation_Capacity (parallel, critical)
    """
    sect = evaluator.add_parallel(
        id="Accommodation_Capacity",
        desc="The resort offers appropriate family accommodation",
        parent=parent,
        critical=True,
    )

    # Family_Suite_Capacity
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="Family_Suite_Capacity",
        desc="The resort offers family suites or room configurations that can accommodate at least 6 guests",
        parent=sect,
        claim="The resort offers a family suite or room configuration that can accommodate at least 6 guests (e.g., 2-bedroom suite, large villa, or guaranteed connecting rooms for 6).",
        sources=_non_empty_urls(ext.capacity_urls or ext.general_info_urls),
        critical=True,
        additional_instruction="Look for max occupancy of six (6) or more in a single bookable unit (suite/villa), or an explicit policy offering guaranteed connecting rooms that together sleep at least 6.",
    )

    # Accommodation_Reference_URL
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="Accommodation_Reference_URL",
        desc="A reference URL confirms the room capacity information",
        parent=sect,
        claim="This page confirms that at least one room type or configuration can sleep 6+ guests.",
        sources=_non_empty_urls(ext.capacity_urls),
        critical=True,
        additional_instruction="Accept official accommodation/room pages or reputable booking pages showing occupancy for 6 or more.",
    )


async def _build_availability(
    evaluator: Evaluator,
    parent,
    ext: ResortExtraction,
):
    """
    Section: MLK_Weekend_Availability (parallel, critical)
    """
    sect = evaluator.add_parallel(
        id="MLK_Weekend_Availability",
        desc="The resort is available for booking during the specified dates",
        parent=parent,
        critical=True,
    )

    # January_2026_Availability
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="January_2026_Availability",
        desc=f"The resort accepts bookings for {MLK_2026_START}–{MLK_2026_END} (MLK Day weekend)",
        parent=sect,
        claim=f"The resort's booking engine or calendar allows selecting a stay that includes {MLK_2026_START} to {MLK_2026_END} (MLK Day weekend 2026).",
        sources=_non_empty_urls(ext.availability_urls),
        critical=True,
        additional_instruction="Pass if the booking tool/calendar shows 2026 and permits selecting the MLK weekend dates for this property (even if specific rooms are sold out). Fail if the site blocks 2026 or does not allow selecting those dates for this resort.",
    )

    # Availability_Reference_URL
    await _verify_with_sources_or_fail(
        evaluator=evaluator,
        node_id="Availability_Reference_URL",
        desc="A reference URL confirms availability or booking capability for the specified period",
        parent=sect,
        claim="This page is the resort's official booking engine, calendar, or a reputable booking source that covers January 2026 for this resort.",
        sources=_non_empty_urls(ext.availability_urls),
        critical=True,
        additional_instruction="Prefer the official booking engine; reputable OTA pages with date selectors are acceptable.",
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
    Evaluate an answer for the Aruba Eagle Beach all-inclusive family resort task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # overall flow is sequential per rubric
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

    # Extraction
    extracted: ResortExtraction = await evaluator.extract(
        prompt=prompt_extract_resort(),
        template_class=ResortExtraction,
        extraction_name="resort_extraction",
    )

    # Add a critical main node (as per rubric) under the root
    main = evaluator.add_sequential(
        id="Resort_Identification_and_Verification",
        desc="The identified resort meets all specified requirements for the family reunion",
        parent=root,
        critical=True,
    )

    # Build all rubric sections under the critical main node
    await _build_resort_identification(evaluator, main, extracted)
    await _build_location_property_type(evaluator, main, extracted)
    await _build_family_amenities(evaluator, main, extracted)
    await _build_accommodation_capacity(evaluator, main, extracted)
    await _build_availability(evaluator, main, extracted)

    # Optionally record helpful custom info
    evaluator.add_custom_info(
        info={
            "resort_name_extracted": extracted.resort_name,
            "url_counts": {
                "general_info": len(extracted.general_info_urls),
                "location": len(extracted.location_urls),
                "beachfront": len(extracted.beachfront_urls),
                "all_inclusive": len(extracted.all_inclusive_urls),
                "kids_club": len(extracted.kids_club_urls),
                "pools": len(extracted.pool_feature_urls),
                "capacity": len(extracted.capacity_urls),
                "availability": len(extracted.availability_urls),
            },
        },
        info_type="extraction_summary",
        info_name="extraction_summary",
    )

    return evaluator.get_summary()