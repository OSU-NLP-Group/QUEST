import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "yosemite_trip_2026"
TASK_DESCRIPTION = (
    "Plan a trip to Yosemite National Park for July 2026 for a group of 5 adults, where one member uses a wheelchair "
    "and requires accessible facilities. Your plan must include: (1) One wheelchair-accessible, paved day hike located "
    "in Yosemite Valley that takes approximately 30 minutes to 1 hour to complete. Provide the trail name and distance. "
    "(2) One overnight backpacking trail that is accessible from Glacier Point Road and is suitable for an easy 2-day trip. "
    "Provide the trail name, distance, and difficulty rating. (3) Complete information about the Yosemite wilderness permit "
    "pickup process, including where permits must be picked up, the operating hours of permit stations, and whether after-hours "
    "pickup is available. (4) The total wilderness permit cost for your 5-person group, showing the calculation breakdown including "
    "the per-permit fee and per-person fee. For each item, provide supporting reference URLs."
)

GROUP_SIZE = 5
EXPECTED_PERMIT_FEE = 10.0
EXPECTED_PER_PERSON_FEE = 5.0
EXPECTED_TOTAL_COST = EXPECTED_PERMIT_FEE + GROUP_SIZE * EXPECTED_PER_PERSON_FEE  # 35.0


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AccessibleHikeExtraction(BaseModel):
    trail_name: Optional[str] = None
    trail_location: Optional[str] = None
    wheelchair_accessible_text: Optional[str] = None
    paved_surface_text: Optional[str] = None
    duration_text: Optional[str] = None
    distance_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class OvernightTrailExtraction(BaseModel):
    trail_name: Optional[str] = None
    trailhead_location_text: Optional[str] = None
    accessible_from_glacier_point_text: Optional[str] = None
    difficulty_text: Optional[str] = None
    distance_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PermitProcessExtraction(BaseModel):
    permit_required_text: Optional[str] = None
    pickup_location_type_text: Optional[str] = None
    hours_text: Optional[str] = None
    after_hours_policy_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PermitCostExtraction(BaseModel):
    per_permit_fee_text: Optional[str] = None
    per_person_fee_text: Optional[str] = None
    total_cost_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_accessible_hike() -> str:
    return """
    From the answer, extract details for ONE wheelchair-accessible, paved day hike located in Yosemite Valley
    that takes approximately 30 minutes to 1 hour. If multiple are provided, extract the FIRST one mentioned.

    Return fields:
    - trail_name: The trail name as written in the answer.
    - trail_location: The stated location (should indicate Yosemite Valley if present).
    - wheelchair_accessible_text: Exact phrase(s) indicating wheelchair accessibility (e.g., "wheelchair accessible", "ADA accessible").
    - paved_surface_text: Exact phrase(s) indicating the trail is paved (e.g., "paved path", "asphalt", "paved sidewalk").
    - duration_text: The stated approximate time to complete (e.g., "30 minutes", "about 1 hour", "30–60 minutes").
    - distance_text: The stated distance (any format, e.g., "1 mile", "1.6 km (1 mile)", "0.5–1 mi").
    - sources: List of all URLs the answer cites for THIS accessible hike (extract exactly as URLs).

    If an item is missing in the answer, set it to null (or [] for sources).
    """


def prompt_extract_overnight_trail() -> str:
    return """
    From the answer, extract details for ONE overnight backpacking trail that is accessible from Glacier Point Road
    and suitable for an EASY 2-day trip. If multiple are provided, extract the FIRST one mentioned.

    Return fields:
    - trail_name: The trail name as written in the answer.
    - trailhead_location_text: Text indicating the trailhead location (if provided).
    - accessible_from_glacier_point_text: Exact phrase(s) indicating it is accessible from Glacier Point Road (or synonyms).
    - difficulty_text: The stated difficulty (e.g., "Easy", "Moderate", "Suitable for an easy 2-day trip").
    - distance_text: The stated total distance (any format).
    - sources: List of all URLs the answer cites for THIS overnight trail.

    If an item is missing in the answer, set it to null (or [] for sources).
    """


def prompt_extract_permit_process() -> str:
    return """
    From the answer, extract the Yosemite wilderness permit pickup process details.

    Return fields:
    - permit_required_text: Text indicating that wilderness permits are required for overnight stays in the Yosemite Wilderness.
    - pickup_location_type_text: Text indicating permits must be picked up IN PERSON at a Yosemite Wilderness Permit Station.
    - hours_text: Stated operating hours for permit stations (e.g., "8 am to 5 pm").
    - after_hours_policy_text: Text indicating whether after-hours pickup is allowed or not.
    - sources: List of all URLs the answer cites for the wilderness permit process.

    If an item is missing in the answer, set it to null (or [] for sources).
    """


def prompt_extract_permit_cost() -> str:
    return """
    From the answer, extract the wilderness permit cost details and the total for a 5-person group.

    Return fields:
    - per_permit_fee_text: The per-permit fee as text (e.g., "$10 reservation fee per permit").
    - per_person_fee_text: The per-person fee as text (e.g., "$5 per person").
    - total_cost_text: The total cost for 5 people as text (e.g., "$35").
    - sources: List of all URLs the answer cites for the permit fee information.

    If an item is missing in the answer, set it to null (or [] for sources).
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    return any(isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")) for u in urls)


def _first_amount(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d{1,2})?)", text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _is_close(a: Optional[float], b: float, tol: float = 1e-6) -> bool:
    if a is None:
        return False
    return abs(a - b) <= tol


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_accessible_day_hike_verification(
    evaluator: Evaluator,
    parent,
    data: AccessibleHikeExtraction,
):
    node = evaluator.add_parallel(
        id="accessible_day_hike",
        desc="Wheelchair-accessible day hike information in Yosemite Valley",
        parent=parent,
        critical=True,
    )

    # Trail Name provided (existence)
    evaluator.add_custom_node(
        result=_non_empty(data.trail_name),
        id="accessible_trail_name",
        desc="Trail name must be provided",
        parent=node,
        critical=True,
    )

    # Trail located in Yosemite Valley (verify with sources)
    trail_loc_node = evaluator.add_leaf(
        id="accessible_trail_location_valley",
        desc="Trail must be located in Yosemite Valley",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The trail '{data.trail_name or ''}' is located in Yosemite Valley in Yosemite National Park.",
        node=trail_loc_node,
        sources=data.sources,
        additional_instruction=(
            "Accept reasonable wording that the trail is on the valley floor or in 'Yosemite Valley'. "
            "Do not accept locations clearly outside Yosemite Valley."
        ),
    )

    # Wheelchair accessible and paved (verify with sources)
    access_node = evaluator.add_leaf(
        id="accessible_trail_wheelchair_paved",
        desc="Trail must be wheelchair accessible and paved",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The trail '{data.trail_name or ''}' is wheelchair-accessible and has a paved surface.",
        node=access_node,
        sources=data.sources,
        additional_instruction=(
            "Look for phrases like 'wheelchair accessible', 'ADA accessible', 'accessible path', and for surface type like "
            "'paved', 'asphalt', 'paved sidewalk'. Both accessibility and paved surface must be supported."
        ),
    )

    # Duration requirement 30–60 minutes (verify with sources)
    duration_node = evaluator.add_leaf(
        id="accessible_trail_duration",
        desc="Trail duration must be approximately 30 minutes to 1 hour",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Completing the trail '{data.trail_name or ''}' typically takes approximately 30 minutes to 1 hour.",
        node=duration_node,
        sources=data.sources,
        additional_instruction=(
            "Accept equivalents such as '0.5–1 hour', 'about 30–60 minutes', or 'under an hour'. "
            "It's fine if the stated time is for a typical/average visitor or round-trip on a flat, paved path."
        ),
    )

    # Trail distance specified (existence)
    evaluator.add_custom_node(
        result=_non_empty(data.distance_text),
        id="accessible_trail_distance_specified",
        desc="Trail distance should be specified",
        parent=node,
        critical=True,
    )

    # Reference URLs present (existence)
    evaluator.add_custom_node(
        result=_has_urls(data.sources),
        id="accessible_trail_reference",
        desc="URL reference for the accessible trail information",
        parent=node,
        critical=True,
    )


async def build_overnight_trail_verification(
    evaluator: Evaluator,
    parent,
    data: OvernightTrailExtraction,
):
    node = evaluator.add_parallel(
        id="overnight_backpacking_trail",
        desc="Overnight backpacking trail accessible from Glacier Point Road",
        parent=parent,
        critical=True,
    )

    # Trail Name provided (existence)
    evaluator.add_custom_node(
        result=_non_empty(data.trail_name),
        id="overnight_trail_name",
        desc="Trail name must be provided",
        parent=node,
        critical=True,
    )

    # Trailhead accessible from Glacier Point Road (verify with sources)
    gp_access_node = evaluator.add_leaf(
        id="overnight_trailhead_glacier_point_rd",
        desc="Trailhead must be accessible from Glacier Point Road",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The trail '{data.trail_name or ''}' is accessible from Glacier Point Road.",
        node=gp_access_node,
        sources=data.sources,
        additional_instruction=(
            "Support includes wording like 'trailhead on Glacier Point Road', 'access via Glacier Point Road', or "
            "'parking on Glacier Point Road'. Synonyms or nearby named points along Glacier Point Road are acceptable."
        ),
    )

    # Difficulty specified as Easy / suitable for easy 2-day trip (verify with sources)
    difficulty_node = evaluator.add_leaf(
        id="overnight_trail_difficulty",
        desc="Trail difficulty must be specified as Easy or compatible with a 2-day easy trip",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The trail '{data.trail_name or ''}' is easy and suitable for a 2-day backpacking trip.",
        node=difficulty_node,
        sources=data.sources,
        additional_instruction=(
            "Accept 'Easy' difficulty or language clearly indicating the route is suitable for a beginner-friendly "
            "two-day backpacking itinerary. Phrases like 'gentle', 'family-friendly overnight', or 'low elevation gain' "
            "can support this if the overall tone is 'easy'."
        ),
    )

    # Distance specified (existence)
    evaluator.add_custom_node(
        result=_non_empty(data.distance_text),
        id="overnight_trail_distance_info",
        desc="Trail distance should be specified",
        parent=node,
        critical=True,
    )

    # Reference URLs present (existence)
    evaluator.add_custom_node(
        result=_has_urls(data.sources),
        id="overnight_trail_reference",
        desc="URL reference for the overnight trail information",
        parent=node,
        critical=True,
    )


async def build_permit_process_verification(
    evaluator: Evaluator,
    parent,
    data: PermitProcessExtraction,
):
    node = evaluator.add_parallel(
        id="wilderness_permit_process",
        desc="Information about obtaining Yosemite wilderness permits",
        parent=parent,
        critical=True,
    )

    # Wilderness permits required for overnight (verify with sources)
    required_node = evaluator.add_leaf(
        id="permit_requirement_required",
        desc="Must state that wilderness permits are required for overnight stays in Yosemite Wilderness",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Wilderness permits are required for all overnight stays in the Yosemite Wilderness.",
        node=required_node,
        sources=data.sources,
        additional_instruction="Look for explicit statements of requirement for any overnight backpacking/camping in the Wilderness.",
    )

    # Pickup location type is in-person at a Wilderness Permit Station (verify with sources)
    pickup_loc_node = evaluator.add_leaf(
        id="permit_pickup_location_type",
        desc="Must specify that permits must be picked up in person at a Yosemite Wilderness Permit Station",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Yosemite wilderness permits must be picked up in person at a Yosemite Wilderness Permit Station.",
        node=pickup_loc_node,
        sources=data.sources,
        additional_instruction="Accept wording like 'in person at a permit station', 'Wilderness Center', or 'permit office'.",
    )

    # Operating hours 8 am - 5 pm (verify with sources)
    hours_node = evaluator.add_leaf(
        id="permit_pickup_hours",
        desc="Must specify the permit station operating hours (8 am - 5 pm)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Yosemite wilderness permit stations operate from 8 am to 5 pm.",
        node=hours_node,
        sources=data.sources,
        additional_instruction="Accept formats like '8:00 am–5:00 pm' or '8 to 5'.",
    )

    # After-hours pickup NOT available (verify with sources)
    after_hours_node = evaluator.add_leaf(
        id="permit_no_after_hours",
        desc="Must state that after-hours pickup is not available",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="After-hours pickup of Yosemite wilderness permits is not available.",
        node=after_hours_node,
        sources=data.sources,
        additional_instruction="Accept language like 'no night box', 'no after-hours pickup', or 'must pick up during business hours'.",
    )

    # Reference URLs present (existence)
    evaluator.add_custom_node(
        result=_has_urls(data.sources),
        id="permit_process_reference",
        desc="URL reference for wilderness permit process information",
        parent=node,
        critical=True,
    )


async def build_permit_cost_verification(
    evaluator: Evaluator,
    parent,
    data: PermitCostExtraction,
):
    node = evaluator.add_parallel(
        id="total_permit_cost",
        desc="Total wilderness permit cost for 5-person group",
        parent=parent,
        critical=True,
    )

    # Per-permit fee is $10 (verify with sources)
    per_permit_node = evaluator.add_leaf(
        id="permit_fee_per_permit",
        desc="Must include the $10 per permit fee component",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The wilderness permit reservation fee is $10 per permit.",
        node=per_permit_node,
        sources=data.sources,
        additional_instruction="Accept wording like 'reservation fee $10 per permit'.",
    )

    # Per-person fee is $5 (verify with sources)
    per_person_node = evaluator.add_leaf(
        id="permit_fee_per_person",
        desc="Must include the $5 per person fee component",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The wilderness permit fee is $5 per person.",
        node=per_person_node,
        sources=data.sources,
        additional_instruction="Accept wording like '$5 per person' or '$5 per person charge'.",
    )

    # Correct total calculation = $35 (custom math check using extracted values if available)
    per_permit_amt = _first_amount(data.per_permit_fee_text)
    per_person_amt = _first_amount(data.per_person_fee_text)
    total_amt = _first_amount(data.total_cost_text)

    correct_total = (
        _is_close(per_permit_amt, EXPECTED_PERMIT_FEE)
        and _is_close(per_person_amt, EXPECTED_PER_PERSON_FEE)
        and _is_close(total_amt, EXPECTED_TOTAL_COST)
        and _is_close(EXPECTED_PERMIT_FEE + GROUP_SIZE * EXPECTED_PER_PERSON_FEE, EXPECTED_TOTAL_COST)
    )

    evaluator.add_custom_node(
        result=correct_total,
        id="permit_correct_total_calculation",
        desc=f"Total cost must be correctly calculated as ${int(EXPECTED_TOTAL_COST)} ($10 permit fee + $5 × {GROUP_SIZE} people)",
        parent=node,
        critical=True,
    )

    # Reference URLs present (existence)
    evaluator.add_custom_node(
        result=_has_urls(data.sources),
        id="permit_cost_reference",
        desc="URL reference for permit fee information",
        parent=node,
        critical=True,
    )


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
    model: str = "o4-mini",
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
        default_model=model,
    )

    # Add a critical top-level aggregator to mirror the rubric's root
    overall = evaluator.add_parallel(
        id="yosemite_trip_planning",
        desc="Complete trip planning for Yosemite National Park in July 2026 with accessibility and wilderness camping requirements",
        parent=root,
        critical=True,
    )

    # Parallelize extractions
    accessible_task = evaluator.extract(
        prompt=prompt_extract_accessible_hike(),
        template_class=AccessibleHikeExtraction,
        extraction_name="accessible_day_hike",
    )
    overnight_task = evaluator.extract(
        prompt=prompt_extract_overnight_trail(),
        template_class=OvernightTrailExtraction,
        extraction_name="overnight_backpacking_trail",
    )
    permit_process_task = evaluator.extract(
        prompt=prompt_extract_permit_process(),
        template_class=PermitProcessExtraction,
        extraction_name="permit_process",
    )
    permit_cost_task = evaluator.extract(
        prompt=prompt_extract_permit_cost(),
        template_class=PermitCostExtraction,
        extraction_name="permit_cost",
    )

    accessible, overnight, permit_process, permit_cost = await asyncio.gather(
        accessible_task, overnight_task, permit_process_task, permit_cost_task
    )

    # Add ground truth/expected constants (for transparency)
    evaluator.add_ground_truth(
        {
            "group_size": GROUP_SIZE,
            "expected_per_permit_fee": EXPECTED_PERMIT_FEE,
            "expected_per_person_fee": EXPECTED_PER_PERSON_FEE,
            "expected_total_cost": EXPECTED_TOTAL_COST,
        },
        gt_type="expected_fee_breakdown",
    )

    # Build verification trees for each major requirement
    await build_accessible_day_hike_verification(evaluator, overall, accessible)
    await build_overnight_trail_verification(evaluator, overall, overnight)
    await build_permit_process_verification(evaluator, overall, permit_process)
    await build_permit_cost_verification(evaluator, overall, permit_cost)

    return evaluator.get_summary()