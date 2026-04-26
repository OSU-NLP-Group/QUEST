import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gc_rim_to_rim_plan_2026"
TASK_DESCRIPTION = (
    "Plan a complete 3-day/2-night rim-to-rim backpacking trip in Grand Canyon National Park for a group of 4 people, "
    "starting from the North Rim on May 20, 2026, and ending at the South Rim. Your plan must include: "
    "1. The complete hiking route specifying which trail(s) will be used each day and the distance covered each day, "
    "2. The specific named campground where the group will stay each of the 2 nights, "
    "3. The total permit cost for the entire group (including all fees), "
    "4. A list of water refill locations available along your chosen route, "
    "5. The park's 24-hour emergency phone number, "
    "6. At least one key safety recommendation for hiking in the Grand Canyon during late spring/early summer"
)

EXPECTED_EMERGENCY_PHONE = "(928) 638-7805"
EXPECTED_TOTAL_COST = 10 + (15 * 4 * 2)  # $10 application fee + $15 per person per night × 4 people × 2 nights = $130


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TripBasicsExtraction(BaseModel):
    start_rim: Optional[str] = None
    start_date: Optional[str] = None
    end_rim: Optional[str] = None
    duration_days: Optional[str] = None
    duration_nights: Optional[str] = None
    group_size: Optional[str] = None


class DailyPlan(BaseModel):
    trails_used: List[str] = Field(default_factory=list)
    distance_text: Optional[str] = None


class RoutePlanExtraction(BaseModel):
    day1: Optional[DailyPlan] = None
    day2: Optional[DailyPlan] = None
    day3: Optional[DailyPlan] = None


class CampgroundsExtraction(BaseModel):
    night1_name: Optional[str] = None
    night2_name: Optional[str] = None


class PermitCostExtraction(BaseModel):
    total_cost_text: Optional[str] = None


class WaterRefillExtraction(BaseModel):
    water_locations: List[str] = Field(default_factory=list)


class EmergencyExtraction(BaseModel):
    emergency_phone: Optional[str] = None


class SafetyExtraction(BaseModel):
    safety_tips: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_basics() -> str:
    return """
    Extract the trip basics from the answer exactly as stated:
    - start_rim: the starting rim mentioned (e.g., "North Rim")
    - start_date: the start date mentioned (e.g., "May 20, 2026" or "5/20/2026")
    - end_rim: the ending rim mentioned (e.g., "South Rim")
    - duration_days: the number of days (e.g., "3", "3 days")
    - duration_nights: the number of nights (e.g., "2", "2 nights")
    - group_size: the stated group size (e.g., "4", "four people", "group of 4")
    If any field is not explicitly mentioned in the answer, return null for that field.
    """


def prompt_extract_route_plan() -> str:
    return """
    Extract the daily route plan details for each of Day 1, Day 2, and Day 3 exactly as stated in the answer.
    For each day, return:
    - trails_used: a list of named trail(s) used that day (e.g., "North Kaibab Trail", "Bright Angel Trail").
      Extract the names exactly as they appear; if multiple trails are named, include all.
    - distance_text: the stated distance for that day as text (e.g., "14 miles", "13–14 miles", "~7 mi").
    If a day is missing, return null for that day. If trails or distance are not mentioned for a day, set those fields to null or empty list accordingly.
    """


def prompt_extract_campgrounds() -> str:
    return """
    Extract the named campground for each night exactly as stated:
    - night1_name: the specific named campground for Night 1 (e.g., "Cottonwood Campground")
    - night2_name: the specific named campground for Night 2 (e.g., "Bright Angel Campground" or "Indian Garden Campground")
    If a night does not mention a specific named campground, return null for that field.
    """


def prompt_extract_permit_cost() -> str:
    return """
    Extract the total permit cost for the entire group including all fees exactly as stated in the answer:
    - total_cost_text: a single total cost string (e.g., "$130", "USD 130", "Total permit cost: $130").
    If the answer does not provide a single total cost figure, return null.
    """


def prompt_extract_water_refill() -> str:
    return """
    Extract the list of water refill locations explicitly mentioned in the answer along the chosen route:
    - water_locations: an array of named refill points (e.g., "Supai Tunnel", "Roaring Springs", "Cottonwood Campground", "Phantom Ranch", "Indian Garden").
    Only include names explicitly listed in the answer. If none are listed, return an empty array.
    """


def prompt_extract_emergency_phone() -> str:
    return """
    Extract the park's 24-hour emergency phone number if stated in the answer:
    - emergency_phone: the phone number string exactly as given in the answer.
    If not provided, return null.
    """


def prompt_extract_safety_tips() -> str:
    return """
    Extract safety recommendations provided for late spring/early summer hiking in the Grand Canyon:
    - safety_tips: an array of the tips or recommendations as short text strings, exactly as stated.
    If none are provided, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def has_numeric(text: Optional[str]) -> bool:
    if not text:
        return False
    return re.search(r"\d", text) is not None


def parse_first_number(text: Optional[str]) -> Optional[float]:
    """
    Parse the first numeric value from a string like "$130", "USD 130", "Total: 130.00".
    Returns float if found, else None.
    """
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_trip_basics_checks(
    evaluator: Evaluator,
    parent_node,
) -> None:
    """
    Add trip basics verification leaves under parent_node and run simple verifications
    against the answer text.
    """
    basics_node = evaluator.add_parallel(
        id="trip_basics",
        desc="Trip matches the requested start/end rims, start date, duration, and group size.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: start_date_and_rim
    start_leaf = evaluator.add_leaf(
        id="start_date_and_rim",
        desc="States the trip starts at the North Rim on May 20, 2026.",
        parent=basics_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The plan states the trip starts at the North Rim on May 20, 2026.",
        node=start_leaf,
        additional_instruction=(
            "Allow minor variations in date formats (e.g., 5/20/2026) and phrasing (e.g., 'start from North Kaibab trailhead (North Rim)'). "
            "The key is that North Rim is the start and the date is May 20, 2026."
        ),
    )

    # Leaf: end_rim
    end_leaf = evaluator.add_leaf(
        id="end_rim",
        desc="States the trip ends at the South Rim.",
        parent=basics_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The plan states the trip ends at the South Rim.",
        node=end_leaf,
        additional_instruction="Accept phrasing variants like 'finish at South Rim' or 'end at Bright Angel Trailhead (South Rim)'.",
    )

    # Leaf: duration
    duration_leaf = evaluator.add_leaf(
        id="duration",
        desc="Plan is explicitly 3 days / 2 nights.",
        parent=basics_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The plan explicitly states it is a 3-day, 2-night trip.",
        node=duration_leaf,
        additional_instruction="Accept '3 days/2 nights' or equivalent phrasing.",
    )

    # Leaf: group_size
    group_leaf = evaluator.add_leaf(
        id="group_size",
        desc="Plan is for a group of 4 people.",
        parent=basics_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The plan is for a group of 4 people.",
        node=group_leaf,
        additional_instruction="Allow phrases like 'party of four', 'group of 4'.",
    )


async def build_route_plan_checks(
    evaluator: Evaluator,
    parent_node,
    route: RoutePlanExtraction,
) -> None:
    """
    Add route plan verification nodes under parent_node.
    For each day, we separate the checks into:
    - trails specified (presence of named trail(s))
    - distance numeric (presence of a numeric distance)
    """
    route_node = evaluator.add_parallel(
        id="route_plan",
        desc="Includes the complete hiking route specifying trail(s) used each day and the distance covered each day (3 days).",
        parent=parent_node,
        critical=True,
    )

    for day_idx in (1, 2, 3):
        day_plan: Optional[DailyPlan] = getattr(route, f"day{day_idx}", None)
        day_node = evaluator.add_parallel(
            id=f"day{day_idx}",
            desc=f"Day {day_idx} specifies trail(s) used and a numeric distance for that day.",
            parent=route_node,
            critical=True,
        )

        # Trails specified (custom existence check)
        trails_ok = bool(day_plan and day_plan.trails_used and len(day_plan.trails_used) > 0)
        evaluator.add_custom_node(
            result=trails_ok,
            id=f"day{day_idx}_trails_specified",
            desc=f"Day {day_idx} names trail(s) used that day.",
            parent=day_node,
            critical=True,
        )

        # Distance numeric (custom numeric presence check)
        dist_ok = has_numeric(day_plan.distance_text if day_plan else None)
        evaluator.add_custom_node(
            result=dist_ok,
            id=f"day{day_idx}_distance_numeric",
            desc=f"Day {day_idx} includes a numeric distance.",
            parent=day_node,
            critical=True,
        )


async def build_campground_checks(
    evaluator: Evaluator,
    parent_node,
    camps: CampgroundsExtraction,
) -> None:
    camp_node = evaluator.add_parallel(
        id="campgrounds",
        desc="Specifies the specific named campground for each of the 2 nights.",
        parent=parent_node,
        critical=True,
    )

    night1_ok = nonempty(camps.night1_name)
    evaluator.add_custom_node(
        result=night1_ok,
        id="night1",
        desc="Night 1 names a specific campground.",
        parent=camp_node,
        critical=True,
    )

    night2_ok = nonempty(camps.night2_name)
    evaluator.add_custom_node(
        result=night2_ok,
        id="night2",
        desc="Night 2 names a specific campground.",
        parent=camp_node,
        critical=True,
    )


async def build_permit_cost_checks(
    evaluator: Evaluator,
    parent_node,
    cost: PermitCostExtraction,
) -> None:
    permit_node = evaluator.add_parallel(
        id="permit_cost",
        desc="Provides the total permit cost for the entire group including all fees, consistent with the given fee constraints.",
        parent=parent_node,
        critical=True,
    )

    # total_cost_provided: numeric provided
    total_num = parse_first_number(cost.total_cost_text)
    evaluator.add_custom_node(
        result=(total_num is not None),
        id="total_cost_provided",
        desc="States a single total permit cost for the entire group (numeric).",
        parent=permit_node,
        critical=True,
    )

    # total_cost_correct_per_constraints: equals expected $130
    evaluator.add_custom_node(
        result=(total_num == EXPECTED_TOTAL_COST),
        id="total_cost_correct_per_constraints",
        desc="Total cost matches the constraint-defined fee structure: $10 application fee + ($15 per person per night × 4 people × 2 nights).",
        parent=permit_node,
        critical=True,
    )


async def build_water_refill_checks(
    evaluator: Evaluator,
    parent_node,
    water: WaterRefillExtraction,
) -> None:
    water_node = evaluator.add_parallel(
        id="water_refill_locations",
        desc="Lists water refill locations available along the chosen route.",
        parent=parent_node,
        critical=True,
    )

    have_water_list = bool(water.water_locations and len(water.water_locations) > 0)
    evaluator.add_custom_node(
        result=have_water_list,
        id="water_locations_listed",
        desc="Provides a list of one or more water refill locations (named locations).",
        parent=water_node,
        critical=True,
    )


async def build_emergency_phone_checks(
    evaluator: Evaluator,
    parent_node,
) -> None:
    phone_node = evaluator.add_parallel(
        id="emergency_phone",
        desc="Provides the park's 24-hour emergency phone number.",
        parent=parent_node,
        critical=True,
    )

    phone_leaf = evaluator.add_leaf(
        id="correct_number",
        desc="Emergency phone number is (928) 638-7805.",
        parent=phone_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The plan provides the Grand Canyon National Park 24-hour emergency phone number as (928) 638-7805.",
        node=phone_leaf,
        additional_instruction="Accept minor formatting variations (spaces, hyphens, parentheses) but the digits must match 928-638-7805.",
    )


async def build_safety_recommendation_checks(
    evaluator: Evaluator,
    parent_node,
) -> None:
    safety_node = evaluator.add_parallel(
        id="safety_recommendation",
        desc="Provides at least one key safety recommendation for late spring/early summer hiking in the Grand Canyon, consistent with constraints.",
        parent=parent_node,
        critical=True,
    )

    safety_leaf = evaluator.add_leaf(
        id="at_least_one_heat_safety_measure",
        desc="Includes at least one heat-related safety recommendation consistent with constraints (e.g., avoid hiking 10 AM–4 PM during excessive heat warnings and/or acknowledges extreme inner-canyon temperatures).",
        parent=safety_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The plan includes at least one heat-related safety recommendation appropriate for late spring/early summer in the Grand Canyon."
        ),
        node=safety_leaf,
        additional_instruction=(
            "Examples include: start early/finish late, avoid hiking between ~10am and 4pm, carry electrolyte drinks, hydrate frequently, "
            "acknowledge extreme inner-canyon temperatures, and adjust plans during excessive heat warnings. "
            "At least one such measure must be present."
        ),
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
    Evaluate the rim-to-rim backpacking plan answer for completeness and correctness
    based on the critical rubric tree.
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
        default_model=model,
    )

    # Extract all needed information concurrently
    trip_task = evaluator.extract(
        prompt=prompt_extract_trip_basics(),
        template_class=TripBasicsExtraction,
        extraction_name="trip_basics",
    )
    route_task = evaluator.extract(
        prompt=prompt_extract_route_plan(),
        template_class=RoutePlanExtraction,
        extraction_name="route_plan",
    )
    camps_task = evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundsExtraction,
        extraction_name="campgrounds",
    )
    cost_task = evaluator.extract(
        prompt=prompt_extract_permit_cost(),
        template_class=PermitCostExtraction,
        extraction_name="permit_cost",
    )
    water_task = evaluator.extract(
        prompt=prompt_extract_water_refill(),
        template_class=WaterRefillExtraction,
        extraction_name="water_refill_locations",
    )
    emergency_task = evaluator.extract(
        prompt=prompt_extract_emergency_phone(),
        template_class=EmergencyExtraction,
        extraction_name="emergency_phone",
    )
    safety_task = evaluator.extract(
        prompt=prompt_extract_safety_tips(),
        template_class=SafetyExtraction,
        extraction_name="safety_recommendations",
    )

    trip, route, camps, cost, water, emergency, safety = await asyncio.gather(
        trip_task, route_task, camps_task, cost_task, water_task, emergency_task, safety_task
    )

    # Record ground truth expectations for transparency
    evaluator.add_ground_truth({
        "expected_emergency_phone": EXPECTED_EMERGENCY_PHONE,
        "permit_fee_formula": "$10 application + $15 per person per night × 4 people × 2 nights",
        "expected_total_cost": EXPECTED_TOTAL_COST
    })

    # Build a single critical "overall" node under root (root is non-critical by framework design)
    overall = evaluator.add_parallel(
        id="overall",
        desc="Provide a complete 3-day/2-night rim-to-rim backpacking plan for 4 people starting North Rim on May 20, 2026 and ending South Rim, including all requested elements.",
        parent=root,
        critical=True,
    )

    # Build and run checks
    await build_trip_basics_checks(evaluator, overall)
    await build_route_plan_checks(evaluator, overall, route)
    await build_campground_checks(evaluator, overall, camps)
    await build_permit_cost_checks(evaluator, overall, cost)
    await build_water_refill_checks(evaluator, overall, water)
    await build_emergency_phone_checks(evaluator, overall)
    await build_safety_recommendation_checks(evaluator, overall)

    # Return structured evaluation summary
    return evaluator.get_summary()