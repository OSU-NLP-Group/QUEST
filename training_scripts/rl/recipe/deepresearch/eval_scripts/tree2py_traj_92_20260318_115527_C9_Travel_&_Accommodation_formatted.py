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
TASK_ID = "orlando_resort_mlk_2026"
TASK_DESCRIPTION = """
Identify a resort in the Orlando, Florida area that meets the following requirements for a large multi-generational family reunion during MLK Weekend 2026 (January 17-19, 2026):

The resort must have:
- At least 300 guest rooms (qualifying as a large property)
- At least 2 swimming pools on property
- An on-site water park
- An on-site golf course
- A full-service spa
- At least 2 on-site restaurants or dining venues
- At least 10,000 square feet of meeting/event space
- Suite-style accommodations available (not just standard rooms)
- ADA-compliant accessible rooms

The resort must be located in or near Orlando, Florida, and must be open and accepting reservations for the MLK Weekend dates (January 17-19, 2026).

For your answer, provide the name of the resort and include URL references that verify each of the required amenities and features.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FeatureEvidence(BaseModel):
    value_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ResortExtraction(BaseModel):
    resort_name: Optional[str] = None

    # Location evidence
    location_text: Optional[str] = None
    location_sources: List[str] = Field(default_factory=list)

    # Booking evidence (for Jan 17–19, 2026)
    booking_sources: List[str] = Field(default_factory=list)

    # Feature evidences (each with value_text and sources)
    rooms: FeatureEvidence = Field(default_factory=FeatureEvidence)          # At least 300 rooms
    pools: FeatureEvidence = Field(default_factory=FeatureEvidence)          # At least 2 pools
    water_park: FeatureEvidence = Field(default_factory=FeatureEvidence)     # On-site water park with features
    golf_course: FeatureEvidence = Field(default_factory=FeatureEvidence)    # On-site 18-hole championship
    spa: FeatureEvidence = Field(default_factory=FeatureEvidence)            # Full-service spa with treatments
    dining: FeatureEvidence = Field(default_factory=FeatureEvidence)         # 2+ on-site restaurants/venues
    meeting_space: FeatureEvidence = Field(default_factory=FeatureEvidence)  # 10,000+ sq ft
    suites: FeatureEvidence = Field(default_factory=FeatureEvidence)         # Suite-style accommodations
    ada: FeatureEvidence = Field(default_factory=FeatureEvidence)            # ADA-compliant accessible rooms


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_resort_and_evidence() -> str:
    return """
    Your goal is to extract exactly one resort recommendation and the evidence URLs for each required criterion from the provided answer.

    Extract the following fields:
    - resort_name: The exact name of the single resort recommended.

    - location_text: Any text in the answer describing the location (e.g., "Orlando", "Lake Buena Vista", "Kissimmee", etc.).
    - location_sources: All URLs cited that help confirm the resort is in or near Orlando, Florida. Include resort official pages or reputable listings that clearly denote location.

    - booking_sources: All URLs cited that demonstrate the resort is open and accepting reservations for the MLK Weekend 2026 dates (Jan 17–19, 2026). Prefer URLs that show a booking engine page with those dates or a search result for those dates.

    For each of the following features, extract an object with:
      - value_text: The number/description as stated (e.g., "1,000 guest rooms", "three pools", "18-hole course", "120,000 sq ft of event space", "suites and villas", "ADA accessible rooms").
      - sources: All URLs cited that support this feature.

    - rooms: At least 300 guest rooms.
    - pools: At least 2 swimming pools on property.
    - water_park: On-site water park and at least one explicit water-attraction feature (e.g., slides, lazy river).
    - golf_course: On-site golf course; ideally 18-hole championship or similar wording.
    - spa: Full-service spa offering massages and/or treatments.
    - dining: At least 2 on-site restaurants or dining venues (bars & grills, cafes, food halls acceptable if on-site).
    - meeting_space: At least 10,000 sq ft of meeting/event space.
    - suites: Suite-style accommodations (e.g., suites, villas, condos, multi-bedroom units) are available.
    - ada: ADA-compliant accessible guest rooms are available.

    IMPORTANT RULES:
    - Extract only what is explicitly present in the answer.
    - For each "sources" field, include only valid URLs that appear in the answer. If none are present, return an empty list for that sources field.
    - If a value (like room count) is mentioned in the answer, place it in value_text exactly as written.
    - If any field is missing from the answer, return null for strings or an empty list for sources as appropriate.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


async def _add_url_supported_check(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    urls: List[str],
    additional_instruction: str,
) -> None:
    """
    Build a critical parallel sub-tree that requires:
      1) Source URLs are provided.
      2) The claim is supported by the cited sources.
    """
    group = evaluator.add_parallel(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=True
    )

    # (1) Sources provided (critical)
    sources_exist = evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"{node_id}_sources_provided",
        desc=f"{desc} - Source URLs are provided",
        parent=group,
        critical=True
    )

    # (2) Claim supported by cited sources (critical)
    supported_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported",
        desc=f"{desc} - Supported by cited sources",
        parent=group,
        critical=True
    )
    await evaluator.verify(
        claim=claim,
        node=supported_leaf,
        sources=urls if urls else None,
        additional_instruction=additional_instruction,
        extra_prerequisites=[sources_exist]
    )


# --------------------------------------------------------------------------- #
# Main verification logic                                                     #
# --------------------------------------------------------------------------- #
async def _build_and_verify(evaluator: Evaluator, root, ext: ResortExtraction) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """
    # Create a main grouping node (non-critical) to aggregate all critical checks under it
    task_group = evaluator.add_parallel(
        id="task_verification",
        desc="Identify one resort in/near Orlando, Florida that meets all stated requirements for MLK Weekend 2026 (Jan 17–19, 2026) with URL evidence for each requirement.",
        parent=root,
        critical=False
    )

    # 1) Resort name is provided (single resort) - critical
    _ = evaluator.add_custom_node(
        result=bool(ext.resort_name and ext.resort_name.strip()),
        id="resort_name_provided",
        desc="Resort name is provided (single resort).",
        parent=task_group,
        critical=True
    )

    # 2) Location in/near Orlando, Florida - critical with URL support
    await _add_url_supported_check(
        evaluator,
        task_group,
        node_id="location_orlando_area",
        desc="Resort is located in or near Orlando, Florida.",
        claim=(
            f"The resort '{ext.resort_name or 'the recommended resort'}' is located in or near Orlando, Florida "
            "(including nearby areas commonly considered part of the Orlando area, such as Lake Buena Vista, "
            "Kissimmee, Celebration, Bay Lake, ChampionsGate, Four Corners, Davenport, or Winter Garden)."
        ),
        urls=_safe_urls(ext.location_sources),
        additional_instruction=(
            "Verify that the webpage explicitly states the resort's location as Orlando or an immediately adjacent "
            "Orlando-area community (e.g., Lake Buena Vista, Kissimmee, Celebration, Bay Lake, ChampionsGate, "
            "Four Corners, Davenport, Winter Garden). Minor naming variations are acceptable. If the page is "
            "irrelevant or does not clearly state Orlando-area location, mark as not supported."
        )
    )

    # 3) Booking availability for Jan 17–19, 2026 - critical with URL support
    await _add_url_supported_check(
        evaluator,
        task_group,
        node_id="booking_availability_mlk_weekend",
        desc="Resort is operational and accepting reservations for Jan 17-19, 2026.",
        claim=(
            f"The resort '{ext.resort_name or 'the recommended resort'}' is accepting reservations for check-in on "
            "Saturday, January 17, 2026, and check-out on Monday, January 19, 2026 (2-night stay over MLK Weekend 2026)."
        ),
        urls=_safe_urls(ext.booking_sources),
        additional_instruction=(
            "Prefer evidence showing the booking engine or availability search results explicitly set to "
            "Jan 17, 2026 (check-in) to Jan 19, 2026 (check-out), with rates or room options displayed. "
            "If the booking page shows those exact dates with available rooms/rates or a 'Select room' / 'Choose room' flow, "
            "consider it supported. If it clearly indicates 'sold out' or 'no availability' for those dates, then it is not supported. "
            "If the page is a generic booking page without any date evidence, do not consider it sufficient."
        )
    )

    # Amenity/feature checks (each critical with URL support)
    # 4) Rooms: 300+
    await _add_url_supported_check(
        evaluator,
        task_group,
        node_id="room_count_300_plus_with_url",
        desc="URL reference(s) provided that verify the resort has at least 300 guest rooms.",
        claim=(
            f"The resort '{ext.resort_name or 'the recommended resort'}' has at least 300 guest rooms."
        ),
        urls=_safe_urls(ext.rooms.sources),
        additional_instruction=(
            "Look for explicit statements like '300 rooms', '300+ rooms', 'over 500 rooms', '1,000 guestrooms', or "
            "similar. Phrases like 'hundreds of rooms' are acceptable if clearly indicating 300 or more. "
            "Suites that are counted as rooms are acceptable. The page must make it clear that total rooms are at least 300."
        )
    )

    # 5) Pools: 2+
    await _add_url_supported_check(
        evaluator,
        task_group,
        node_id="two_or_more_pools_with_url",
        desc="URL reference(s) provided that verify the resort has at least 2 swimming pools on property.",
        claim=(
            f"The resort '{ext.resort_name or 'the recommended resort'}' has at least two swimming pools on property."
        ),
        urls=_safe_urls(ext.pools.sources),
        additional_instruction=(
            "Verify language like 'two pools', 'multiple pools', 'several pools', or naming at least two distinct pools. "
            "Pool complexes count as long as there are at least two pools. Lazy rivers or slides alone do not count as a pool."
        )
    )

    # 6) On-site water park with features
    await _add_url_supported_check(
        evaluator,
        task_group,
        node_id="onsite_water_park_with_features_with_url",
        desc="URL reference(s) provided that verify an on-site water park exists and indicates water-attraction features (e.g., slides, lazy river, or similar).",
        claim=(
            f"The resort '{ext.resort_name or 'the recommended resort'}' has an on-site water park with notable water attractions "
            "such as slides, a lazy river, splash zones, or similar features."
        ),
        urls=_safe_urls(ext.water_park.sources),
        additional_instruction=(
            "The page should refer to an on-site water park (i.e., part of the resort complex) and mention at least one "
            "distinct feature such as slides, a lazy river, wave pool, or splash area. Third-party offsite parks do not count."
        )
    )

    # 7) On-site 18-hole championship golf course
    await _add_url_supported_check(
        evaluator,
        task_group,
        node_id="onsite_golf_course_18_hole_championship_with_url",
        desc="URL reference(s) provided that verify an on-site 18-hole championship golf course exists.",
        claim=(
            f"The resort '{ext.resort_name or 'the recommended resort'}' has an on-site 18-hole (or multiple 18-hole) championship golf course(s)."
        ),
        urls=_safe_urls(ext.golf_course.sources),
        additional_instruction=(
            "Confirm the golf course is on-site (part of the resort property/complex) and 18 holes (or more). "
            "Wording like 'signature 18-hole course' or 'championship course' is sufficient if clearly associated with the resort."
        )
    )

    # 8) Full-service spa with treatments
    await _add_url_supported_check(
        evaluator,
        task_group,
        node_id="full_service_spa_with_treatments_with_url",
        desc="URL reference(s) provided that verify a full-service spa exists with massage and/or treatment services.",
        claim=(
            f"The resort '{ext.resort_name or 'the recommended resort'}' has a full-service spa offering massages and/or treatment services."
        ),
        urls=_safe_urls(ext.spa.sources),
        additional_instruction=(
            "Accept if the spa menu or description explicitly mentions massages, facials, body treatments, or comparable services. "
            "Basic fitness centers or saunas alone do not qualify."
        )
    )

    # 9) Two or more dining venues
    await _add_url_supported_check(
        evaluator,
        task_group,
        node_id="two_or_more_dining_venues_with_url",
        desc="URL reference(s) provided that verify at least 2 on-site restaurants or dining venues.",
        claim=(
            f"The resort '{ext.resort_name or 'the recommended resort'}' has at least two on-site restaurants or dining venues."
        ),
        urls=_safe_urls(ext.dining.sources),
        additional_instruction=(
            "Restaurants, cafes, bars & grills, markets/food halls located on property all count as dining venues. "
            "Confirm at least two distinct on-site options are listed."
        )
    )

    # 10) Meeting/event space: 10,000+ sq ft
    await _add_url_supported_check(
        evaluator,
        task_group,
        node_id="meeting_space_10000_sqft_plus_with_url",
        desc="URL reference(s) provided that verify at least 10,000 square feet of meeting/event space.",
        claim=(
            f"The resort '{ext.resort_name or 'the recommended resort'}' offers at least 10,000 square feet of meeting or event space on-site."
        ),
        urls=_safe_urls(ext.meeting_space.sources),
        additional_instruction=(
            "Accept explicit statements like '10,000 sq ft', 'over 10,000 square feet', '100,000 sq. ft.', etc. "
            "If only square meters are given, ≥ 929 square meters is equivalent to ≥ 10,000 sq ft. "
            "Ballroom or total event space counts if it clearly meets or exceeds 10,000 sq ft."
        )
    )

    # 11) Suite-style accommodations
    await _add_url_supported_check(
        evaluator,
        task_group,
        node_id="suite_style_accommodations_with_url",
        desc="URL reference(s) provided that verify suite-style accommodations are offered (not only standard rooms).",
        claim=(
            f"The resort '{ext.resort_name or 'the recommended resort'}' offers suite-style accommodations (e.g., suites, villas, condos, or multi-bedroom units), not just standard rooms."
        ),
        urls=_safe_urls(ext.suites.sources),
        additional_instruction=(
            "Look for room categories named 'suite', 'villa', 'condo', 'residence', or multi-bedroom units. "
            "Descriptions emphasizing separate living areas or kitchens also support suite-style accommodations."
        )
    )

    # 12) ADA-compliant accessible rooms
    await _add_url_supported_check(
        evaluator,
        task_group,
        node_id="ada_accessible_rooms_with_url",
        desc="URL reference(s) provided that verify ADA-compliant accessible guest rooms are available.",
        claim=(
            f"The resort '{ext.resort_name or 'the recommended resort'}' provides ADA-compliant accessible guest rooms."
        ),
        urls=_safe_urls(ext.ada.sources),
        additional_instruction=(
            "Verify terms such as 'ADA compliant', 'Accessible room', 'Mobility accessible', 'Hearing accessible', "
            "or descriptions mentioning roll-in showers, grab bars, visual alarms, etc. The evidence must clearly "
            "indicate accessible/ADA rooms are available."
        )
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
    Evaluate an answer for the Orlando-area resort MLK Weekend 2026 task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel: each requirement is checked independently
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

    # Extract structured details from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_resort_and_evidence(),
        template_class=ResortExtraction,
        extraction_name="resort_extraction",
    )

    # Record key context info
    evaluator.add_custom_info(
        info={
            "mlk_weekend_dates": {"check_in": "2026-01-17", "check_out": "2026-01-19"},
            "resort_name_extracted": extracted.resort_name
        },
        info_type="context",
        info_name="task_context"
    )

    # Build verification tree and run checks
    await _build_and_verify(evaluator, root, extracted)

    # Return the structured evaluation summary
    return evaluator.get_summary()