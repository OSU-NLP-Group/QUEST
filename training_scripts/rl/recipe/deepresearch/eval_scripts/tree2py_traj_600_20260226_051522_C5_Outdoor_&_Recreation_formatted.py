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
TASK_ID = "maine_trip_2026"
TASK_DESCRIPTION = """Validates the complete camping trip plan for a July 2026 weekend at a Maine State Park with Acadia National Park visit."""

# Ground Truth facts expected (used for reference; not directly scored)
GROUND_TRUTH_INFO = {
    "maine_state_parks_opening": "February 5, 2026 at 9:00 AM EST",
    "maine_state_parks_advance_booking": "Online reservations must be made by 4:00 PM EST at least 1 day ahead of arrival date",
    "maine_state_parks_weekend_min_stay": "2-night minimum for weekend reservations",
    "cadillac_reservation_required_period": "May 20 through October 25, 2026",
    "cadillac_reservation_cost": "$6",
    "bangor_to_acadia_distance_miles": "approximately 50 miles",
    "summer_activities_examples": ["mountain biking", "scenic lift rides", "hiking", "golf"]
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CampgroundReservationInfo(BaseModel):
    reservation_system_urls: List[str] = Field(default_factory=list)
    opening_datetime: Optional[str] = None  # e.g., "February 5, 2026 at 9:00 AM EST"
    advance_booking_requirement: Optional[str] = None  # e.g., "by 4:00 PM EST at least 1 day ahead"
    weekend_minimum_stay_nights: Optional[str] = None  # e.g., "2 nights"


class AcadiaCadillacInfo(BaseModel):
    cadillac_reservation_urls: List[str] = Field(default_factory=list)
    vehicle_reservation_required_july2026: Optional[str] = None  # "required" or "not required"
    reservation_period: Optional[str] = None  # e.g., "May 20 through October 25, 2026"
    reservation_cost_usd: Optional[str] = None  # e.g., "$6"


class LocationContext(BaseModel):
    bangor_to_acadia_distance_miles: Optional[str] = None  # e.g., "approximately 50 miles"
    location_sources: List[str] = Field(default_factory=list)


class SummerActivitiesInfo(BaseModel):
    ski_resort_summer_activities: List[str] = Field(default_factory=list)  # list of activities mentioned
    activity_sources: List[str] = Field(default_factory=list)


class TripPlanExtraction(BaseModel):
    campground: Optional[CampgroundReservationInfo] = None
    acadia: Optional[AcadiaCadillacInfo] = None
    location: Optional[LocationContext] = None
    summer_activities: Optional[SummerActivitiesInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_plan() -> str:
    return """
    Extract structured details from the answer related to:
    1) Maine State Park campground reservation information,
    2) Acadia National Park Cadillac Summit Road requirements,
    3) Location context (distance from Bangor),
    4) Summer activities at Maine ski resorts.

    Return a JSON object with keys: campground, acadia, location, summer_activities.

    For "campground":
    - reservation_system_urls: all URLs provided that point to official Maine State Park campground reservation information or the official reservation system (e.g., maine.gov DACF parks pages, CampWithME). Only include URLs explicitly present in the answer.
    - opening_datetime: the stated opening date/time for 2026 reservations (verbatim as in the answer, e.g., "February 5, 2026 at 9:00 AM EST").
    - advance_booking_requirement: the stated rule for advance online booking (e.g., "by 4:00 PM EST at least 1 day ahead of arrival").
    - weekend_minimum_stay_nights: the stated minimum nights required for weekend reservations (e.g., "2 nights").

    For "acadia":
    - cadillac_reservation_urls: all URLs provided that point to official NPS or recreation.gov pages explaining Cadillac Summit Road vehicle reservations.
    - vehicle_reservation_required_july2026: normalize to "required" or "not required" based solely on the answer's statement.
    - reservation_period: the stated period during which vehicle reservations are required (e.g., "May 20 through October 25, 2026").
    - reservation_cost_usd: the stated cost for the Cadillac Summit Road vehicle reservation (e.g., "$6").

    For "location":
    - bangor_to_acadia_distance_miles: the stated approximate distance (e.g., "about 50 miles", "approximately 50 miles").
    - location_sources: any URLs provided to support the distance (if any).

    For "summer_activities":
    - ski_resort_summer_activities: list of activities at Maine ski resorts mentioned in the answer; include only from this set if present or obvious synonyms: ["mountain biking", "scenic lift rides", "hiking", "golf"]. If synonyms like "lift rides" or "bike park" appear, map them reasonably to the canonical terms.
    - activity_sources: any URLs provided to support the activities (if any).

    Rules:
    - Do not invent information. If an item is missing in the answer, set it to null (for single fields) or empty array (for list/urls).
    - Extract full URLs only; for markdown links, include just the URL.
    """


# --------------------------------------------------------------------------- #
# Helper verification utilities                                               #
# --------------------------------------------------------------------------- #
async def verify_text_claim_with_urls(
    evaluator: Evaluator,
    parent,
    leaf_id: str,
    desc: str,
    claim_text: Optional[str],
    sources: List[str],
    additional_instruction: str,
    critical: bool = True
) -> None:
    """
    Create a leaf node and verify a text claim against one or more URLs.
    If claim_text is missing/empty, mark the node as failed without calling the verifier.
    """
    node = evaluator.add_leaf(
        id=leaf_id,
        desc=desc,
        parent=parent,
        critical=critical
    )

    if not claim_text or not str(claim_text).strip():
        node.score = 0.0
        node.status = "failed"
        return

    await evaluator.verify(
        claim=claim_text,
        node=node,
        sources=sources if sources else None,
        additional_instruction=additional_instruction
    )


async def verify_answer_mentions(
    evaluator: Evaluator,
    parent,
    leaf_id: str,
    desc: str,
    mention_text: Optional[str],
    additional_instruction: str,
    critical: bool = False
) -> None:
    """
    Verify that the answer includes a stated piece of information (simple verification).
    If mention_text is missing/empty, mark failed.
    """
    node = evaluator.add_leaf(
        id=leaf_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    if not mention_text or not str(mention_text).strip():
        node.score = 0.0
        node.status = "failed"
        return

    claim = f"The answer states: {mention_text}"
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=None,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_campground_reservation_timing(
    evaluator: Evaluator,
    root_node,
    data: TripPlanExtraction
) -> None:
    camp_node = evaluator.add_parallel(
        id="Campground_Reservation_Timing",
        desc="Verifies correct identification of when and how to make Maine State Park campground reservations",
        parent=root_node,
        critical=True
    )

    campground = data.campground or CampgroundReservationInfo()

    # 1) Reservation System URL existence (critical)
    evaluator.add_custom_node(
        result=bool(campground.reservation_system_urls),
        id="Reservation_System_URL",
        desc="Provides URL reference for Maine State Park campground reservation system or official information",
        parent=camp_node,
        critical=True
    )

    # 2) Opening date/time (critical, verify against the provided official URLs)
    await verify_text_claim_with_urls(
        evaluator=evaluator,
        parent=camp_node,
        leaf_id="Reservation_Opening_Date",
        desc="Identifies that Maine State Park campground reservations open February 5, 2026 at 9:00 AM EST",
        claim_text=f"Maine State Park campground reservations open on {campground.opening_datetime}." if campground.opening_datetime else None,
        sources=campground.reservation_system_urls,
        additional_instruction="Verify that the official Maine State Parks reservation information page explicitly states the opening date and time for 2026 as February 5, 2026 at 9:00 AM EST. Allow 'ET' or 'Eastern Time' variants.",
        critical=True
    )

    # 3) Advance booking requirement (critical)
    await verify_text_claim_with_urls(
        evaluator=evaluator,
        parent=camp_node,
        leaf_id="Advance_Booking_Requirement",
        desc="States that online reservations must be made by 4:00 PM EST at least 1 day ahead of arrival date",
        claim_text=f"Online reservations must be made by {campground.advance_booking_requirement} for Maine State Park campgrounds." if campground.advance_booking_requirement else None,
        sources=campground.reservation_system_urls,
        additional_instruction="Verify that the official page states online reservations must be made by 4:00 PM EST (or 4 PM ET) at least 1 day prior to arrival.",
        critical=True
    )

    # 4) Weekend minimum stay (critical)
    await verify_text_claim_with_urls(
        evaluator=evaluator,
        parent=camp_node,
        leaf_id="Weekend_Minimum_Stay",
        desc="Identifies the 2-night minimum stay requirement for weekend reservations at Maine State Parks",
        claim_text=f"Weekend reservations at Maine State Park campgrounds require a minimum of {campground.weekend_minimum_stay_nights}." if campground.weekend_minimum_stay_nights else None,
        sources=campground.reservation_system_urls,
        additional_instruction="Verify that weekend reservations specify a 2-night minimum stay requirement.",
        critical=True
    )


async def verify_acadia_cadillac_requirements(
    evaluator: Evaluator,
    root_node,
    data: TripPlanExtraction
) -> None:
    acadia_node = evaluator.add_parallel(
        id="Acadia_Cadillac_Requirements",
        desc="Verifies correct identification of requirements to drive Cadillac Summit Road in Acadia National Park, including cost information",
        parent=root_node,
        critical=True
    )

    acadia = data.acadia or AcadiaCadillacInfo()

    # 1) Cadillac Reservation URL existence (critical)
    evaluator.add_custom_node(
        result=bool(acadia.cadillac_reservation_urls),
        id="Cadillac_Reservation_URL",
        desc="Provides URL reference for Cadillac Summit Road vehicle reservations (nps.gov or recreation.gov)",
        parent=acadia_node,
        critical=True
    )

    # 2) Vehicle reservation requirement in July 2026 (critical)
    vr_value = acadia.vehicle_reservation_required_july2026
    claim_vr = None
    if vr_value and vr_value.strip():
        normalized = vr_value.strip().lower()
        if normalized == "required":
            claim_vr = "A vehicle reservation is required to drive Cadillac Summit Road during July 2026."
        elif normalized == "not required":
            claim_vr = "A vehicle reservation is not required to drive Cadillac Summit Road during July 2026."
        else:
            claim_vr = f"Vehicle reservation status for Cadillac Summit Road in July 2026 is: {vr_value}."

    await verify_text_claim_with_urls(
        evaluator=evaluator,
        parent=acadia_node,
        leaf_id="Vehicle_Reservation_Required",
        desc="States that vehicle reservation is required for Cadillac Summit Road during July 2026",
        claim_text=claim_vr,
        sources=acadia.cadillac_reservation_urls,
        additional_instruction="Confirm the page explicitly states whether a vehicle reservation is required to drive Cadillac Summit Road in July 2026. Focus on official NPS or Recreation.gov pages.",
        critical=True
    )

    # 3) Reservation period (critical)
    await verify_text_claim_with_urls(
        evaluator=evaluator,
        parent=acadia_node,
        leaf_id="Reservation_Period",
        desc="Identifies that vehicle reservations are required from May 20 through October 25, 2026",
        claim_text=f"Vehicle reservations for Cadillac Summit Road are required from {acadia.reservation_period}." if acadia.reservation_period else None,
        sources=acadia.cadillac_reservation_urls,
        additional_instruction="Verify the official pages state the vehicle reservation requirement applies from May 20 through October 25, 2026.",
        critical=True
    )

    # 4) Reservation cost (critical)
    await verify_text_claim_with_urls(
        evaluator=evaluator,
        parent=acadia_node,
        leaf_id="Reservation_Cost",
        desc="States that Cadillac Summit Road vehicle reservation costs $6 and provides this as the total cost for required Acadia reservations",
        claim_text=f"The Cadillac Summit Road vehicle reservation costs {acadia.reservation_cost_usd}." if acadia.reservation_cost_usd else None,
        sources=acadia.cadillac_reservation_urls,
        additional_instruction="Verify that the reservation fee for Cadillac Summit Road (vehicle reservation) is $6 on the official sources.",
        critical=True
    )


async def verify_park_location_context(
    evaluator: Evaluator,
    root_node,
    data: TripPlanExtraction
) -> None:
    loc_node = evaluator.add_parallel(
        id="Park_Location_Context",
        desc="Provides geographical context about Acadia National Park location",
        parent=root_node,
        critical=False
    )

    location = data.location or LocationContext()
    # Verify the answer mentions an approximate distance (non-critical, simple verify)
    await verify_answer_mentions(
        evaluator=evaluator,
        parent=loc_node,
        leaf_id="Distance_from_Bangor",
        desc="Identifies that Acadia National Park is approximately 50 miles from Bangor, Maine",
        mention_text=f"Acadia National Park is {location.bangor_to_acadia_distance_miles} from Bangor, Maine." if location.bangor_to_acadia_distance_miles else None,
        additional_instruction="This check verifies that the answer mentions the approximate distance (around 50 miles). Allow phrases like 'about' or 'approximately'.",
        critical=False
    )


async def verify_summer_activity_options(
    evaluator: Evaluator,
    root_node,
    data: TripPlanExtraction
) -> None:
    act_node = evaluator.add_parallel(
        id="Summer_Activity_Options",
        desc="Provides information about additional summer outdoor activities available in Maine",
        parent=root_node,
        critical=False
    )

    activities = data.summer_activities or SummerActivitiesInfo()
    # Build a concise mention text of activities extracted
    mention = None
    if activities.ski_resort_summer_activities:
        listed = ", ".join(activities.ski_resort_summer_activities)
        mention = f"Maine ski resorts offer summer activities such as {listed}."

    await verify_answer_mentions(
        evaluator=evaluator,
        parent=act_node,
        leaf_id="Ski_Resort_Summer_Activities",
        desc="Mentions that Maine ski resorts offer summer activities such as mountain biking, scenic lift rides, hiking, or golf",
        mention_text=mention,
        additional_instruction="Verify that the answer text mentions one or more of these activities. Treat reasonable synonyms (e.g., 'lift rides' for 'scenic lift rides', 'bike park' for 'mountain biking') as matches.",
        critical=False
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate the answer for the Maine trip planning task using the Mind2Web2 framework.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root: parallel aggregation across categories
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Validates the complete camping trip plan for a July 2026 weekend at a Maine State Park with Acadia National Park visit",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Record ground truth info
    evaluator.add_ground_truth(
        gt_info=GROUND_TRUTH_INFO,
        gt_type="expected_facts"
    )

    # Extract structured trip plan info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_trip_plan(),
        template_class=TripPlanExtraction,
        extraction_name="trip_plan_extraction"
    )

    # Build verification tree according to rubric
    await verify_campground_reservation_timing(evaluator, root, extracted)
    await verify_acadia_cadillac_requirements(evaluator, root, extracted)
    await verify_park_location_context(evaluator, root, extracted)
    await verify_summer_activity_options(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()