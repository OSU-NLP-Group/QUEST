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
TASK_ID = "co_nrt_accessible_high_loop_paved_multiuse_multith"
TASK_DESCRIPTION = (
    "Identify a National Recreation Trail in Colorado that is located at an elevation above 10,000 feet, "
    "is wheelchair accessible with ADA-compliant features, has a paved surface, forms a loop configuration "
    "(not an out-and-back route), allows both biking and walking as recreational uses, and has multiple "
    "trailhead access points."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TrailExtraction(BaseModel):
    """
    Structured information for a single identified trail.
    Keep fields as strings/lists to be robust to various answer formats.
    """
    name: Optional[str] = None
    state: Optional[str] = None
    elevation_text: Optional[str] = None
    accessibility_text: Optional[str] = None  # e.g., "ADA accessible", "wheelchair-accessible"
    surface_text: Optional[str] = None        # e.g., "paved", "asphalt", "concrete"
    configuration_text: Optional[str] = None  # e.g., "loop", "out-and-back"
    uses: List[str] = Field(default_factory=list)  # e.g., ["biking", "walking"]
    trailheads: List[str] = Field(default_factory=list)  # names/locations of trailheads or access points
    sources: List[str] = Field(default_factory=list)     # all URLs cited for this trail in the answer


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_trail() -> str:
    return """
    Extract details about the single primary trail identified in the answer that the author intends to recommend for this task.
    If multiple trails are mentioned, choose the first one that appears to satisfy the constraints; if unclear which satisfies them, pick the first trail named.

    Return the following fields:
    - name: The trail's name, exactly as stated in the answer.
    - state: The U.S. state for the trail (e.g., "Colorado" or "CO") if provided.
    - elevation_text: Any elevation-related text associated with the trail (e.g., "10,300 feet", "3,150 m", "elevation ~10,500 ft").
    - accessibility_text: Any phrase indicating wheelchair accessibility or ADA features.
    - surface_text: The trail surface type (e.g., "paved", "asphalt", "concrete") if mentioned.
    - configuration_text: The route configuration if stated (e.g., "loop", "out-and-back", "lollipop loop").
    - uses: A list of recreational uses explicitly mentioned for the trail (e.g., "biking", "walking", "hiking", "cycling").
    - trailheads: A list of trailhead or access point names/locations if the answer mentions multiple access points.
    - sources: A list of all URLs cited in the answer for this trail. Include URLs in plain form or from markdown links. If none are present, return an empty list.

    Do not invent any values. If a field is not available in the answer, set it to null (for strings) or an empty list (for lists).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _safe_trail_name(trail: TrailExtraction) -> str:
    return trail.name if trail and trail.name else "the identified trail"


def _base_additional_instruction() -> str:
    return (
        "Use only the provided URL sources to evaluate the claim. "
        "If the provided sources list is empty, invalid, irrelevant, or inaccessible, "
        "treat the claim as not supported (Incorrect). "
        "Allow reasonable synonymy and minor lexical variations. "
    )


def _instruction_nrt() -> str:
    return _base_additional_instruction() + (
        "Specifically check whether the page explicitly indicates the trail is designated as a "
        "National Recreation Trail (NRT) under the U.S. National Recreation Trails Program. "
        "Synonyms/phrases like 'National Recreation Trail' or 'NRT' count."
    )


def _instruction_co() -> str:
    return _base_additional_instruction() + (
        "Verify that the trail is located in Colorado (CO). Accept mentions of cities/parks in Colorado. "
        "Abbreviations like 'CO' or 'Colorado, USA' count."
    )


def _instruction_elevation() -> str:
    return _base_additional_instruction() + (
        "Verify that the trail's location/elevation is above or equal to 10,000 feet. "
        "Accept metric equivalents (>= 3,048 meters). Distinguish 'elevation' from 'elevation gain'. "
        "Evidence should indicate altitude of the trail, its typical elevation, or clear elevation range "
        "that is >= 10,000 ft for the route."
    )


def _instruction_accessibility() -> str:
    return _base_additional_instruction() + (
        "Verify wheelchair accessibility and ADA-compliant features. "
        "Accept language such as 'ADA accessible', 'wheelchair accessible', 'barrier-free', "
        "or descriptions of compliant grade/width/surface consistent with ADA."
    )


def _instruction_paved() -> str:
    return _base_additional_instruction() + (
        "Verify that the surface is paved (e.g., asphalt or concrete). "
        "Do not accept 'compacted gravel' as paved."
    )


def _instruction_loop() -> str:
    return _base_additional_instruction() + (
        "Verify that the route is a loop (closed circuit). "
        "Terms like 'loop', 'circuit', or 'lollipop loop' that return to the start are acceptable. "
        "Do not accept a simple out-and-back as a loop."
    )


def _instruction_multiuse() -> str:
    return _base_additional_instruction() + (
        "Verify that both biking (cycling) and walking (or hiking) are allowed recreational uses on this trail. "
        "Synonyms like 'bicycle', 'cycling', 'walk', 'hike' are acceptable."
    )


def _instruction_multith() -> str:
    return _base_additional_instruction() + (
        "Verify that there are multiple (two or more) distinct trailheads or access points for the trail. "
        "Accept multiple named trailheads, access points, or clearly distinct parking/entrance locations that function as trailheads."
    )


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_trail(
    evaluator: Evaluator,
    parent_node,
    trail: TrailExtraction,
) -> None:
    """
    Build the verification tree and run checks for the identified trail.
    Follows the rubric: a single critical parallel node with eight critical leaves.
    """
    # Create the main critical node
    main = evaluator.add_parallel(
        id="TrailIdentification",
        desc="Identify a National Recreation Trail in Colorado meeting the specified elevation, accessibility, surface, configuration, use, and access-point constraints.",
        parent=parent_node,
        critical=True,
    )

    # Prepare leaves
    # 1. National Recreation Trail designation
    nrt_node = evaluator.add_leaf(
        id="NationalRecreationTrailDesignation",
        desc="The trail is officially designated as a National Recreation Trail.",
        parent=main,
        critical=True,
    )

    # 2. Colorado location
    co_node = evaluator.add_leaf(
        id="ColoradoLocation",
        desc="The trail is located in Colorado.",
        parent=main,
        critical=True,
    )

    # 3. High elevation (>= 10,000 ft)
    elev_node = evaluator.add_leaf(
        id="HighElevationLocation",
        desc="The trail is at an elevation above 10,000 feet.",
        parent=main,
        critical=True,
    )

    # 4. Wheelchair/ADA accessibility
    ada_node = evaluator.add_leaf(
        id="WheelchairAccessibility",
        desc="The trail is wheelchair accessible and meets ADA accessibility standards.",
        parent=main,
        critical=True,
    )

    # 5. Paved surface
    paved_node = evaluator.add_leaf(
        id="PavedSurface",
        desc="The trail has a paved surface.",
        parent=main,
        critical=True,
    )

    # 6. Loop configuration
    loop_node = evaluator.add_leaf(
        id="LoopConfiguration",
        desc="The trail forms a loop configuration (not an out-and-back route).",
        parent=main,
        critical=True,
    )

    # 7. Multiple recreational uses (biking and walking)
    uses_node = evaluator.add_leaf(
        id="MultipleRecreationalUses",
        desc="The trail allows both biking and walking as recreational uses.",
        parent=main,
        critical=True,
    )

    # 8. Multiple trailheads
    mth_node = evaluator.add_leaf(
        id="MultipleTrailheads",
        desc="The trail has multiple trailhead access points.",
        parent=main,
        critical=True,
    )

    trail_name = _safe_trail_name(trail)
    sources = trail.sources if trail and trail.sources else []

    # Build claims and run in parallel using batch_verify
    claims_and_sources: List[tuple[str, List[str], Any, Optional[str]]] = [
        (
            f"{trail_name} is officially designated as a National Recreation Trail (NRT).",
            sources,
            nrt_node,
            _instruction_nrt(),
        ),
        (
            f"{trail_name} is located in Colorado.",
            sources,
            co_node,
            _instruction_co(),
        ),
        (
            f"{trail_name} is located at an elevation above or equal to 10,000 feet (or >= 3,048 meters).",
            sources,
            elev_node,
            _instruction_elevation(),
        ),
        (
            f"{trail_name} is wheelchair accessible and has ADA-compliant features.",
            sources,
            ada_node,
            _instruction_accessibility(),
        ),
        (
            f"{trail_name} has a paved surface (asphalt or concrete).",
            sources,
            paved_node,
            _instruction_paved(),
        ),
        (
            f"{trail_name} is configured as a loop route (closed circuit), not a simple out-and-back.",
            sources,
            loop_node,
            _instruction_loop(),
        ),
        (
            f"Both biking (cycling) and walking (hiking) are allowed uses on {trail_name}.",
            sources,
            uses_node,
            _instruction_multiuse(),
        ),
        (
            f"{trail_name} has multiple trailheads or access points (two or more).",
            sources,
            mth_node,
            _instruction_multith(),
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Colorado National Recreation Trail task with multiple constraints.
    """
    # Initialize evaluator and root
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

    # Extraction
    trail_data = await evaluator.extract(
        prompt=prompt_extract_trail(),
        template_class=TrailExtraction,
        extraction_name="trail_extraction",
    )

    # Optional logging info block
    evaluator.add_custom_info(
        info={
            "selected_trail_name": trail_data.name,
            "sources_count": len(trail_data.sources) if trail_data.sources else 0,
            "noted_uses": trail_data.uses,
            "noted_configuration": trail_data.configuration_text,
            "noted_surface": trail_data.surface_text,
            "noted_accessibility": trail_data.accessibility_text,
            "noted_elevation": trail_data.elevation_text,
            "noted_trailheads_count": len(trail_data.trailheads) if trail_data.trailheads else 0,
        },
        info_type="extraction_summary",
        info_name="trail_extraction_summary",
    )

    # Build verification tree and run checks
    await build_and_verify_trail(evaluator, root, trail_data)

    # Return final structured summary
    return evaluator.get_summary()