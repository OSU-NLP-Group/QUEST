import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mexico_city_concert_venues_2026"
TASK_DESCRIPTION = (
    "A major international music production company is planning a concert tour stop in Mexico City for 2026 and needs to identify suitable indoor concert venues. "
    "Identify three indoor concert venues located in Mexico City, Mexico, that meet the following requirements:\n\n"
    "1. The venue must have a concert seating capacity of at least 9,000 people.\n"
    "2. The venue must be an indoor arena or auditorium suitable for hosting major concert productions.\n"
    "3. The venue must have a large stage with minimum dimensions of 20 meters wide (or equivalent large-scale staging capability) suitable for major productions.\n"
    "4. The venue must have modern sound and lighting systems suitable for major concert productions.\n"
    "5. The venue must be currently operational and available for booking concerts in 2026.\n"
    "6. The venue must have an established history of hosting concerts and major events.\n"
    "7. The venue should include wheelchair-accessible seating or accessibility features.\n"
    "8. The venue should have substantial parking facilities.\n"
    "9. The venue should include luxury suites or VIP seating areas where applicable.\n\n"
    "For each venue, provide the venue name, a description of how it meets the requirements, and URL references that confirm the venue's specifications."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    sources: List[str] = Field(default_factory=list)

    # Optional structured fields that may appear in the answer
    location: Optional[str] = None
    venue_type: Optional[str] = None
    capacity: Optional[str] = None
    stage: Optional[str] = None
    sound_lighting: Optional[str] = None
    operational_2026: Optional[str] = None
    history: Optional[str] = None

    # Non-critical preference signals
    accessibility: Optional[str] = None
    parking: Optional[str] = None
    vip: Optional[str] = None


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to three indoor concert venues located in Mexico City, Mexico, as presented in the answer. 
    For each venue, return an object containing at least:
      - name: The venue name exactly as given in the answer (string; null if not provided).
      - description: A short description of how the venue meets the requirements (string; null if not provided).
      - sources: An array of URL(s) cited in the answer that support the venue specifications (list of strings). Include only valid URLs and those explicitly present in the answer.
    Optionally, if the answer provides them, also extract:
      - location: City/area text mentioned (string; null if not provided).
      - venue_type: e.g., "indoor arena" or "auditorium" (string; null if not provided).
      - capacity: any stated capacity text (e.g., "10,000 seats") (string; null if not provided).
      - stage: any stage dimension or capability text (string; null if not provided).
      - sound_lighting: any text indicating modern sound/lighting systems (string; null if not provided).
      - operational_2026: any text indicating operational/booking availability in 2026 (string; null if not provided).
      - history: any text indicating a history of hosting concerts/events (string; null if not provided).
      - accessibility: any text indicating wheelchair-accessible seating or accessibility features (string; null if not provided).
      - parking: any text indicating substantial parking facilities (string; null if not provided).
      - vip: any text indicating luxury suites or VIP seating areas (string; null if not provided).
    
    Return a JSON object:
      {
        "venues": [
          {venue_1_object},
          {venue_2_object},
          {venue_3_object}
        ]
      }
    If more than three venues appear in the answer, include only the first three in the array.
    If fewer than three appear, include only those found.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_text(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _valid_sources(sources: Optional[List[str]]) -> List[str]:
    if not sources:
        return []
    # Filter out obvious non-URLs or empty
    cleaned = []
    for u in sources:
        if not u:
            continue
        u = u.strip()
        if len(u) < 4:
            continue
        # Accept any that contain typical URL patterns
        if "http://" in u or "https://" in u or "." in u:
            cleaned.append(u)
    return cleaned


# --------------------------------------------------------------------------- #
# Verification for a single venue                                             #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    venue_index: int,
) -> None:
    vid = f"venue_{venue_index}"
    venue_name_display = venue.name or f"Venue #{venue_index + 1}"
    # Each venue is a sequential node: Deliverables -> MUST -> SHOULD
    venue_node = evaluator.add_sequential(
        id=f"{vid}",
        desc=f"{venue_name_display} verification (sequential: deliverables, must, should)",
        parent=parent_node,
        critical=False  # Each venue yields partial credit
    )

    # ------------------ Deliverables (critical) ------------------ #
    deliverables_node = evaluator.add_parallel(
        id=f"{vid}_deliverables",
        desc="Required fields are provided for this venue.",
        parent=venue_node,
        critical=True
    )
    # 1) Venue name provided
    evaluator.add_custom_node(
        result=_non_empty_text(venue.name),
        id=f"{vid}_name_provided",
        desc="Provides the venue name.",
        parent=deliverables_node,
        critical=True
    )
    # 2) Description provided
    evaluator.add_custom_node(
        result=_non_empty_text(venue.description),
        id=f"{vid}_requirements_description",
        desc="Provides a description explaining how the venue meets the stated requirements.",
        parent=deliverables_node,
        critical=True
    )
    # 3) Supporting URLs provided
    srcs = _valid_sources(venue.sources)
    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id=f"{vid}_supporting_urls",
        desc="Provides URL reference(s) that confirm the venue's specifications claimed.",
        parent=deliverables_node,
        critical=True
    )

    # ------------------ MUST Requirements (critical) ------------------ #
    must_node = evaluator.add_parallel(
        id=f"{vid}_must_requirements",
        desc="All MUST constraints from the question are satisfied for this venue.",
        parent=venue_node,
        critical=True
    )

    # Build leaf nodes for MUST items
    loc_leaf = evaluator.add_leaf(
        id=f"{vid}_location_mexico_city",
        desc=f"{venue_name_display}: Venue is located in Mexico City, Mexico.",
        parent=must_node,
        critical=True
    )
    indoor_leaf = evaluator.add_leaf(
        id=f"{vid}_indoor_arena_or_auditorium",
        desc=f"{venue_name_display}: Venue is an indoor arena or auditorium suitable for major concert productions.",
        parent=must_node,
        critical=True
    )
    capacity_leaf = evaluator.add_leaf(
        id=f"{vid}_capacity_at_least_9000",
        desc=f"{venue_name_display}: Concert seating capacity is at least 9,000 people.",
        parent=must_node,
        critical=True
    )
    stage_leaf = evaluator.add_leaf(
        id=f"{vid}_stage_at_least_20m_or_equiv",
        desc=f"{venue_name_display}: Has a large stage ≥20m wide or equivalent large-scale staging capability.",
        parent=must_node,
        critical=True
    )
    sound_light_leaf = evaluator.add_leaf(
        id=f"{vid}_modern_sound_and_lighting",
        desc=f"{venue_name_display}: Has modern sound and lighting systems suitable for major concert productions.",
        parent=must_node,
        critical=True
    )
    operational_leaf = evaluator.add_leaf(
        id=f"{vid}_operational_and_bookable_2026",
        desc=f"{venue_name_display}: Currently operational and available for booking concerts in 2026.",
        parent=must_node,
        critical=True
    )
    history_leaf = evaluator.add_leaf(
        id=f"{vid}_history_of_concerts_and_events",
        desc=f"{venue_name_display}: Has an established history of hosting concerts and major events.",
        parent=must_node,
        critical=True
    )

    # Prepare MUST claims with additional instructions
    must_claims = [
        (
            f"The venue '{venue_name_display}' is located in Mexico City, Mexico (CDMX).",
            srcs,
            loc_leaf,
            "Accept boroughs/neighborhoods within Mexico City (e.g., Azcapotzalco, Benito Juárez, Miguel Hidalgo). "
            "If the page states the venue is in Mexico City or CDMX, consider this supported."
        ),
        (
            f"The venue '{venue_name_display}' is an indoor arena or auditorium suitable for hosting major concert productions.",
            srcs,
            indoor_leaf,
            "Look for terms like 'indoor arena', 'covered arena', 'auditorium', or evidence of hosting arena-scale concerts."
        ),
        (
            f"The venue '{venue_name_display}' has a concert seating capacity of at least 9,000 people.",
            srcs,
            capacity_leaf,
            "Accept explicit capacity numbers ≥ 9,000 or ranges/wording indicating capacity above 9,000 (e.g., 10,000, 15,000)."
        ),
        (
            f"The venue '{venue_name_display}' has a large stage that is at least 20 meters wide OR otherwise supports equivalent large-scale staging capability.",
            srcs,
            stage_leaf,
            "20 meters ≈ 65.6 feet. Accept stage width in feet ≥ 65, or explicit mentions of large-scale staging capability "
            "(e.g., modular/expandable stage systems supporting arena-scale productions)."
        ),
        (
            f"The venue '{venue_name_display}' has modern sound and lighting systems suitable for major concert productions.",
            srcs,
            sound_light_leaf,
            "Accept phrases like 'state-of-the-art sound', 'modern lighting rigs', 'professional audio/visual systems', "
            "or technical specs indicating suitability for major concerts."
        ),
        (
            f"The venue '{venue_name_display}' is currently operational and can be booked for concerts in 2026.",
            srcs,
            operational_leaf,
            "Evidence can include 2026 event schedules, active booking pages referencing 2026 availability, or official notices "
            "indicating operations and bookings in 2026."
        ),
        (
            f"The venue '{venue_name_display}' has an established history of hosting concerts and major events.",
            srcs,
            history_leaf,
            "Accept event calendars, past events listings, or press coverage showing multiple concerts/major events over time."
        ),
    ]

    # Run MUST verifications in parallel (they will be auto-skipped if deliverables fail due to sequential gating)
    await evaluator.batch_verify(must_claims)

    # ------------------ SHOULD Requirements (non-critical) ------------------ #
    should_node = evaluator.add_parallel(
        id=f"{vid}_should_requirements",
        desc="Non-critical preferences (\"should\" constraints) for this venue.",
        parent=venue_node,
        critical=False
    )

    access_leaf = evaluator.add_leaf(
        id=f"{vid}_accessibility_features",
        desc=f"{venue_name_display}: Provides wheelchair-accessible seating or accessibility features.",
        parent=should_node,
        critical=False
    )
    parking_leaf = evaluator.add_leaf(
        id=f"{vid}_substantial_parking",
        desc=f"{venue_name_display}: Provides substantial parking facilities.",
        parent=should_node,
        critical=False
    )
    vip_leaf = evaluator.add_leaf(
        id=f"{vid}_luxury_or_vip_areas",
        desc=f"{venue_name_display}: Includes luxury suites or VIP seating areas where applicable.",
        parent=should_node,
        critical=False
    )

    should_claims = [
        (
            f"The venue '{venue_name_display}' provides wheelchair-accessible seating or accessibility features.",
            srcs,
            access_leaf,
            "Accept ADA/accessible seating statements, wheelchair access mentions, ramps/elevators for guests, or official accessibility information."
        ),
        (
            f"The venue '{venue_name_display}' offers substantial parking facilities.",
            srcs,
            parking_leaf,
            "Accept multi-level parking structures, large lots, or explicit parking capacity indicating substantial availability."
        ),
        (
            f"The venue '{venue_name_display}' includes luxury suites or VIP seating areas.",
            srcs,
            vip_leaf,
            "Accept references to suites, VIP boxes, premium seating, hospitality areas, or similar luxury sections."
        ),
    ]

    await evaluator.batch_verify(should_claims)


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
    Evaluate an answer for the Mexico City indoor concert venues task.
    Builds a verification tree:
      - Root (parallel, non-critical): Three venue subtrees
      - Venue subtree (sequential): Deliverables (critical) -> MUST (critical) -> SHOULD (non-critical)
      - MUST and SHOULD contain per-criterion leaf verifications grounded by URLs.
    """
    # Initialize evaluator with a non-critical parallel root to allow partial credit across venues
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

    # Extract venue information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    venues = list(extracted.venues or [])
    extracted_count = len(venues)

    # Limit to first 3 venues, pad with empty entries if fewer than 3
    if extracted_count > 3:
        venues = venues[:3]
    while len(venues) < 3:
        venues.append(VenueItem())

    # Add custom info about extraction
    evaluator.add_custom_info(
        info={
            "extracted_venue_count": extracted_count,
            "used_venues": min(extracted_count, 3),
            "padded_venues": max(0, 3 - extracted_count)
        },
        info_type="extraction_stats",
        info_name="venues_extraction_stats"
    )

    # Build and verify each of the 3 venues under the root
    for i in range(3):
        await verify_single_venue(evaluator, root, venues[i], i)

    # Return evaluation summary
    return evaluator.get_summary()