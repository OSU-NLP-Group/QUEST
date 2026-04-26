import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wilderness_ca_aug2026_4n"
TASK_DESCRIPTION = (
    "Plan a 4-night wilderness backpacking trip in California for a group of 8 adults during August 2026. "
    "Your plan must include:\n\n"
    "1. Wilderness Area Identification: official name + URL confirming existence/location.\n"
    "2. Trailhead Selection: specific entry trailhead + URL.\n"
    "3. Permit Information: whether permits required, reservation method, advance window, fee structure + URL.\n"
    "4. Group Size Regulations: max group size, verify 8 within limit + URL.\n"
    "5. Bear Canister Requirements: whether required + URL.\n"
    "6. Camping Regulations: designated vs dispersed, setbacks from water and trails + URLs.\n"
    "7. Fire Restrictions: elevation/distance restrictions if any + URL.\n"
    "8. Stay Duration Limits: max consecutive nights, verify 4 nights allowed + URL.\n"
    "9. Seasonal Access: accessible during August 2026 + URL.\n"
    "10. Camping Locations: for each of 4 nights, campsite/zone, approx distance info, reference URL.\n"
    "11. Route Description: loop vs shuttle, approx total distance, route/trail map URL.\n\n"
    "All claims must be supported by URLs from official park/forest service sites or established outdoor recreation resources."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class NightInfo(BaseModel):
    campsite_or_zone: Optional[str] = None
    distance_info: Optional[str] = None
    url: Optional[str] = None


class TripPlanExtraction(BaseModel):
    # Trip basics
    trip_month_year: Optional[str] = None  # e.g., "August 2026"
    duration_nights: Optional[str] = None  # e.g., "4 nights" or "4-night (5-day)"
    group_size: Optional[str] = None       # e.g., "8 adults"

    # Wilderness area
    wilderness_area_name: Optional[str] = None
    wilderness_area_url: Optional[str] = None

    # Trailhead
    trailhead_name: Optional[str] = None
    trailhead_url: Optional[str] = None

    # Permit info
    permit_required: Optional[str] = None
    permit_reservation_method: Optional[str] = None
    permit_advance_window: Optional[str] = None
    permit_fee_structure: Optional[str] = None
    permit_info_url: Optional[str] = None

    # Group size regulations
    max_group_size: Optional[str] = None
    group_size_regulations_url: Optional[str] = None

    # Bear canister rules
    bear_canister_required: Optional[str] = None
    bear_canister_rules_url: Optional[str] = None

    # Camping regulations
    camping_style: Optional[str] = None  # e.g., "designated sites only" or "dispersed camping with setbacks"
    camping_style_url: Optional[str] = None
    water_setback_distance_feet: Optional[str] = None
    water_setback_url: Optional[str] = None
    trail_setback_distance_feet: Optional[str] = None
    trail_setback_url: Optional[str] = None

    # Fire restrictions
    fire_elevation_limit_info: Optional[str] = None  # e.g., "No campfires above 9,000 feet"
    fire_distance_restrictions_info: Optional[str] = None
    fire_rules_url: Optional[str] = None

    # Stay duration limits
    max_consecutive_nights: Optional[str] = None
    stay_duration_rules_url: Optional[str] = None

    # Seasonal access
    seasonal_access_confirmed: Optional[str] = None  # e.g., "Accessible in August"
    seasonal_access_url: Optional[str] = None

    # Camping locations for 4 nights
    night1: Optional[NightInfo] = None
    night2: Optional[NightInfo] = None
    night3: Optional[NightInfo] = None
    night4: Optional[NightInfo] = None

    # Route description
    route_loop_or_shuttle: Optional[str] = None  # e.g., "loop" or "requires shuttle"
    approx_total_distance: Optional[str] = None  # e.g., "40 miles"
    route_map_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_plan() -> str:
    return """
    Extract the structured plan information exactly as stated in the answer. Return values as strings when applicable. 
    If any item is missing, return null for that field. For URLs, return full valid URLs explicitly mentioned in the answer.

    Required fields:
    - trip_month_year: The month and year for the trip timing (e.g., "August 2026").
    - duration_nights: The itinerary duration (e.g., "4 nights" or "4-night (5-day)").
    - group_size: The group size (e.g., "8 adults").

    Wilderness area:
    - wilderness_area_name
    - wilderness_area_url

    Trailhead:
    - trailhead_name
    - trailhead_url

    Permit information:
    - permit_required (e.g., "permits required" or "no permit required")
    - permit_reservation_method (e.g., "advance reservation via Recreation.gov", "walk-up")
    - permit_advance_window (e.g., "up to 6 months in advance")
    - permit_fee_structure (e.g., "$6 reservation fee + $5/person")
    - permit_info_url

    Group size regulations:
    - max_group_size (e.g., "12", "15")
    - group_size_regulations_url

    Bear canister rules:
    - bear_canister_required (e.g., "required", "recommended")
    - bear_canister_rules_url

    Camping regulations:
    - camping_style (e.g., "designated sites only", "dispersed camping with setbacks")
    - camping_style_url
    - water_setback_distance_feet (e.g., "200 feet")
    - water_setback_url
    - trail_setback_distance_feet (e.g., "100 feet")
    - trail_setback_url

    Fire restrictions:
    - fire_elevation_limit_info (state the elevation limit if any, otherwise "none specified")
    - fire_distance_restrictions_info (state distance restrictions if any, otherwise "none specified")
    - fire_rules_url

    Stay duration limits:
    - max_consecutive_nights
    - stay_duration_rules_url

    Seasonal access:
    - seasonal_access_confirmed (e.g., "accessible in August 2026")
    - seasonal_access_url

    Camping locations for nights 1 through 4:
    For each night, extract a NightInfo object:
    - campsite_or_zone
    - distance_info (for night 1: distance from trailhead; for nights 2-4: distance from previous night)
    - url

    Route description:
    - route_loop_or_shuttle (e.g., "loop", "requires shuttle")
    - approx_total_distance (e.g., "45 miles", "~38 mi")
    - route_map_url
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def safe_str(x: Optional[str]) -> str:
    return x.strip() if isinstance(x, str) else ""


def parse_first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\d+", text.replace(",", ""))
    return int(m.group()) if m else None


def gather_all_urls(plan: TripPlanExtraction) -> List[str]:
    urls: List[Optional[str]] = [
        plan.wilderness_area_url,
        plan.trailhead_url,
        plan.permit_info_url,
        plan.group_size_regulations_url,
        plan.bear_canister_rules_url,
        plan.camping_style_url,
        plan.water_setback_url,
        plan.trail_setback_url,
        plan.fire_rules_url,
        plan.stay_duration_rules_url,
        plan.seasonal_access_url,
        plan.route_map_url,
        plan.night1.url if plan.night1 else None,
        plan.night2.url if plan.night2 else None,
        plan.night3.url if plan.night3 else None,
        plan.night4.url if plan.night4 else None,
    ]
    seen = set()
    unique: List[str] = []
    for u in urls:
        if u and u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


async def verify_text_leaf(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    sources: Optional[str | List[str]] = None,
    add_ins: Optional[str] = None,
    critical: bool = True,
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=add_ins or "None",
    )


def add_url_existence_node(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    url: Optional[str],
    critical: bool = True,
):
    return evaluator.add_custom_node(
        result=bool(url) and bool(safe_str(url)),
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )


def add_text_existence_node(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    text: Optional[str],
    critical: bool = True,
):
    return evaluator.add_custom_node(
        result=bool(text) and bool(safe_str(text)),
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )


# --------------------------------------------------------------------------- #
# Verification section builders                                               #
# --------------------------------------------------------------------------- #
async def build_trip_basics(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    basics = evaluator.add_parallel(
        id="Trip_Basics",
        desc="Trip matches the requested timing, duration, and group size.",
        parent=parent,
        critical=True,
    )
    # Trip timing
    await verify_text_leaf(
        evaluator,
        basics,
        "Trip_Timing_August_2026",
        "Plan is explicitly for August 2026.",
        sources=None,
        add_ins="Verify the answer text explicitly states the trip timing in August 2026.",
        critical=True,
    )
    # Duration
    await verify_text_leaf(
        evaluator,
        basics,
        "Trip_Duration_4_Nights",
        "The itinerary duration is 4 nights.",
        sources=None,
        add_ins="Verify the answer explicitly provides a 4-night (approximately 5-day) plan.",
        critical=True,
    )
    # Group size
    await verify_text_leaf(
        evaluator,
        basics,
        "Group_Size_8_Adults",
        "The plan is explicitly for a group of 8 adults.",
        sources=None,
        add_ins="Verify the answer states the group size is 8 adults.",
        critical=True,
    )


async def build_source_constraints(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    src_node = evaluator.add_parallel(
        id="Source_Constraints",
        desc="References comply with the allowed-source requirement.",
        parent=parent,
        critical=True,
    )
    all_urls = gather_all_urls(plan)
    claim = (
        f"The following reference URLs appear to be from allowed sources "
        f"(official park service websites, forest service websites, or established outdoor recreation resources): {all_urls}."
    )
    add_ins = (
        "Allowed sources include domains like nps.gov, fs.usda.gov, blm.gov, parks.ca.gov, recreation.gov, "
        "other official state/municipal park sites, and established outdoor resources (e.g., AllTrails, CalTopo, SummitPost). "
        "Judge based on the domain provenance. Minor subdomains are fine."
    )
    await verify_text_leaf(
        evaluator,
        src_node,
        "Allowed_Source_URLs_Only",
        "All provided reference URLs comply with the allowed-source requirement.",
        sources=None,
        add_ins=add_ins,
        critical=True,
    )


async def build_wilderness_area_identification(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    w_node = evaluator.add_parallel(
        id="Wilderness_Area_Identification",
        desc="Identify the wilderness area and support its existence and California location with a URL.",
        parent=parent,
        critical=True,
    )
    # Official name provided
    add_text_existence_node(
        evaluator,
        w_node,
        "Wilderness_Area_Official_Name",
        "Provide the official name of the selected wilderness area.",
        plan.wilderness_area_name,
        critical=True,
    )
    # URL existence first
    add_url_existence_node(
        evaluator,
        w_node,
        "Wilderness_Area_Existence_Location_URL_presence",
        "A reference URL is provided confirming the wilderness area’s existence/location.",
        plan.wilderness_area_url,
        critical=True,
    )
    # Verify existence and CA location via URL
    name = safe_str(plan.wilderness_area_name)
    await verify_text_leaf(
        evaluator,
        w_node,
        "Wilderness_Area_Existence_Location_URL",
        f"The wilderness area named '{name}' exists and is located in California.",
        sources=plan.wilderness_area_url,
        add_ins="Verify the page mentions the wilderness area and that it is in California.",
        critical=True,
    )


async def build_trailhead_selection(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    t_node = evaluator.add_parallel(
        id="Trailhead_Selection",
        desc="Identify the entry trailhead and support it with a URL.",
        parent=parent,
        critical=True,
    )
    # Entry trailhead name
    add_text_existence_node(
        evaluator,
        t_node,
        "Entry_Trailhead_Name",
        "Provide the official name of the specific entry trailhead for the chosen route.",
        plan.trailhead_name,
        critical=True,
    )
    # URL presence
    add_url_existence_node(
        evaluator,
        t_node,
        "Entry_Trailhead_URL_presence",
        "A reference URL is provided for the entry trailhead information.",
        plan.trailhead_url,
        critical=True,
    )
    # Verify trailhead existence via URL
    th_name = safe_str(plan.trailhead_name)
    await verify_text_leaf(
        evaluator,
        t_node,
        "Entry_Trailhead_URL",
        f"The entry trailhead '{th_name}' is valid for the chosen route.",
        sources=plan.trailhead_url,
        add_ins="Verify that the trailhead name appears on the referenced page and is appropriate for access to the described route.",
        critical=True,
    )


async def build_permit_information(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    p_node = evaluator.add_parallel(
        id="Permit_Information",
        desc="Provide complete wilderness permit system details with a supporting URL.",
        parent=parent,
        critical=True,
    )
    # URL presence first
    add_url_existence_node(
        evaluator,
        p_node,
        "Permit_Info_URL",
        "Provide a reference URL for permit information.",
        plan.permit_info_url,
        critical=True,
    )
    # Key fields
    await verify_text_leaf(
        evaluator,
        p_node,
        "Permits_Required_Status",
        f"Permits required status: {safe_str(plan.permit_required)}.",
        sources=plan.permit_info_url,
        add_ins="Verify whether overnight permits are required per the permit info page.",
        critical=True,
    )
    await verify_text_leaf(
        evaluator,
        p_node,
        "Permit_Reservation_Method",
        f"Permit reservation method: {safe_str(plan.permit_reservation_method)}.",
        sources=plan.permit_info_url,
        add_ins="Verify how permits can be reserved (advance/walk-up/lottery/etc.) per the permit info page.",
        critical=True,
    )
    await verify_text_leaf(
        evaluator,
        p_node,
        "Permit_Advance_Window",
        f"Permit advance reservation window: {safe_str(plan.permit_advance_window)}.",
        sources=plan.permit_info_url,
        add_ins="Verify how far in advance permits can be reserved.",
        critical=True,
    )
    await verify_text_leaf(
        evaluator,
        p_node,
        "Permit_Fee_Structure",
        f"Permit fee structure: {safe_str(plan.permit_fee_structure)}.",
        sources=plan.permit_info_url,
        add_ins="Verify reservation fees and per-person costs as stated.",
        critical=True,
    )


async def build_group_size_regulations(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    g_node = evaluator.add_parallel(
        id="Group_Size_Regulations",
        desc="Provide max group size, verify 8 is within limit, and include URL.",
        parent=parent,
        critical=True,
    )
    # URL presence first
    add_url_existence_node(
        evaluator,
        g_node,
        "Group_Size_Regulations_URL",
        "Provide a reference URL for group size regulations.",
        plan.group_size_regulations_url,
        critical=True,
    )
    # Max group size
    await verify_text_leaf(
        evaluator,
        g_node,
        "Max_Group_Size",
        f"Maximum group size allowed: {safe_str(plan.max_group_size)}.",
        sources=plan.group_size_regulations_url,
        add_ins="Verify the stated maximum overnight party size for the chosen wilderness area.",
        critical=True,
    )
    # Verify 8 within limit (logical check supported by the same URL)
    max_text = safe_str(plan.max_group_size)
    await verify_text_leaf(
        evaluator,
        g_node,
        "Group_Size_Within_Limit_Verification",
        f"A group of 8 is within the stated maximum group size ('{max_text}').",
        sources=plan.group_size_regulations_url,
        add_ins="Use the maximum group size stated on the page to confirm that 8 people is within the limit.",
        critical=True,
    )


async def build_bear_canister_rules(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    b_node = evaluator.add_parallel(
        id="Bear_Canister_Rules",
        desc="State bear canister requirement and include URL.",
        parent=parent,
        critical=True,
    )
    # URL presence first
    add_url_existence_node(
        evaluator,
        b_node,
        "Bear_Canister_Rules_URL",
        "Provide a reference URL for bear canister requirements.",
        plan.bear_canister_rules_url,
        critical=True,
    )
    await verify_text_leaf(
        evaluator,
        b_node,
        "Bear_Canister_Required_Status",
        f"Bear-resistant food storage containers requirement: {safe_str(plan.bear_canister_required)}.",
        sources=plan.bear_canister_rules_url,
        add_ins="Verify if bear canisters are required or recommended for the area.",
        critical=True,
    )


async def build_camping_regulations(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    c_node = evaluator.add_parallel(
        id="Camping_Regulations",
        desc="Provide camping style and setback requirements (water, trails) with URLs.",
        parent=parent,
        critical=True,
    )
    # Camping style URL presence first
    add_url_existence_node(
        evaluator,
        c_node,
        "Designated_vs_Dispersed_URL",
        "Provide a reference URL supporting the camping style rule.",
        plan.camping_style_url,
        critical=True,
    )
    await verify_text_leaf(
        evaluator,
        c_node,
        "Designated_vs_Dispersed_Camping",
        f"Camping style: {safe_str(plan.camping_style)}.",
        sources=plan.camping_style_url,
        add_ins="Verify whether camping is limited to designated sites or dispersed camping is allowed (with setbacks).",
        critical=True,
    )
    # Water setback
    add_url_existence_node(
        evaluator,
        c_node,
        "Water_Setback_URL",
        "Provide a reference URL for the water setback requirement.",
        plan.water_setback_url,
        critical=True,
    )
    await verify_text_leaf(
        evaluator,
        c_node,
        "Water_Setback_Distance_Feet",
        f"Minimum campsite distance from water sources: {safe_str(plan.water_setback_distance_feet)}.",
        sources=plan.water_setback_url,
        add_ins="Verify the minimum required distance from lakes/streams/rivers for camping.",
        critical=True,
    )
    # Trail setback
    add_url_existence_node(
        evaluator,
        c_node,
        "Trail_Setback_URL",
        "Provide a reference URL for the trail setback requirement.",
        plan.trail_setback_url,
        critical=True,
    )
    await verify_text_leaf(
        evaluator,
        c_node,
        "Trail_Setback_Distance_Feet",
        f"Minimum campsite distance from trails: {safe_str(plan.trail_setback_distance_feet)}.",
        sources=plan.trail_setback_url,
        add_ins="Verify the minimum required distance from established trails for camping.",
        critical=True,
    )


async def build_fire_restrictions(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    f_node = evaluator.add_parallel(
        id="Fire_Restrictions",
        desc="Provide campfire rules including elevation limits and distance restrictions (or explicitly none), with URL.",
        parent=parent,
        critical=True,
    )
    # URL presence first
    add_url_existence_node(
        evaluator,
        f_node,
        "Fire_Rules_URL",
        "Provide a reference URL for fire regulations.",
        plan.fire_rules_url,
        critical=True,
    )
    await verify_text_leaf(
        evaluator,
        f_node,
        "Fire_Elevation_Limit_Info",
        f"Campfire elevation limit info: {safe_str(plan.fire_elevation_limit_info)}.",
        sources=plan.fire_rules_url,
        add_ins="Verify any elevation threshold above which campfires are not allowed; if none specified, confirm that.",
        critical=True,
    )
    await verify_text_leaf(
        evaluator,
        f_node,
        "Fire_Distance_Restrictions_Info",
        f"Campfire distance restrictions info: {safe_str(plan.fire_distance_restrictions_info)}.",
        sources=plan.fire_rules_url,
        add_ins="Verify any stated distance restrictions (e.g., distance from water/trails); if none specified, confirm that.",
        critical=True,
    )


async def build_stay_duration_limits(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    s_node = evaluator.add_parallel(
        id="Stay_Duration_Limits",
        desc="Provide consecutive-night limits, verify 4 nights is allowed, with URL.",
        parent=parent,
        critical=True,
    )
    # URL presence first
    add_url_existence_node(
        evaluator,
        s_node,
        "Stay_Duration_Rules_URL",
        "Provide a reference URL for stay duration regulations.",
        plan.stay_duration_rules_url,
        critical=True,
    )
    await verify_text_leaf(
        evaluator,
        s_node,
        "Max_Consecutive_Nights",
        f"Maximum consecutive nights allowed: {safe_str(plan.max_consecutive_nights)}.",
        sources=plan.stay_duration_rules_url,
        add_ins="Verify the stated maximum number of consecutive nights permitted.",
        critical=True,
    )
    max_text = safe_str(plan.max_consecutive_nights)
    await verify_text_leaf(
        evaluator,
        s_node,
        "Four_Nights_Within_Limit_Verification",
        f"A 4-night itinerary is within the allowed limit ('{max_text}').",
        sources=plan.stay_duration_rules_url,
        add_ins="Use the max consecutive nights stated on the page to confirm that 4 nights is allowed.",
        critical=True,
    )


async def build_seasonal_access(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    a_node = evaluator.add_parallel(
        id="Seasonal_Access",
        desc="Confirm accessibility during August 2026 with URL.",
        parent=parent,
        critical=True,
    )
    # URL presence first
    add_url_existence_node(
        evaluator,
        a_node,
        "Seasonal_Access_URL",
        "Provide a reference URL supporting the seasonal access claim.",
        plan.seasonal_access_url,
        critical=True,
    )
    await verify_text_leaf(
        evaluator,
        a_node,
        "August_2026_Access_Confirmed",
        "The wilderness area and relevant trails are accessible during August 2026.",
        sources=plan.seasonal_access_url,
        add_ins="Confirm typical summer accessibility and any seasonal closures; August should be accessible unless otherwise stated.",
        critical=True,
    )


async def build_camping_locations(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    def night_node_builder(night_id: str, night_label: str, night: Optional[NightInfo]):
        n = evaluator.add_parallel(
            id=night_id,
            desc=f"{night_label} camping details.",
            parent=camp_node,
            critical=True,
        )
        # URL presence first
        add_url_existence_node(
            evaluator,
            n,
            f"{night_id}_Camping_URL",
            f"Provide a reference URL for {night_label.lower()} camping location information.",
            night.url if night else None,
            critical=True,
        )
        # Campsite/zone verification
        name = safe_str(night.campsite_or_zone) if night else ""
        # Existence of name (logic gate as custom)
        add_text_existence_node(
            evaluator,
            n,
            f"{night_id}_Campsite_or_Zone_presence",
            f"Provide the specific campsite name or camping zone designation for {night_label.lower()}.",
            night.campsite_or_zone if night else None,
            critical=True,
        )
        # Verify name against URL
        asyncio.create_task(verify_text_leaf(
            evaluator,
            n,
            f"{night_id}_Campsite_or_Zone",
            f"The campsite/zone '{name}' is valid for {night_label.lower()}.",
            sources=night.url if night else None,
            add_ins="Verify that the campsite/zone name appears or is clearly indicated as a valid camping location.",
            critical=True,
        ))
        # Distance verification
        dist = safe_str(night.distance_info) if night else ""
        asyncio.create_task(verify_text_leaf(
            evaluator,
            n,
            f"{night_id}_Distance",
            f"Approximate distance information is provided: {dist}.",
            sources=night.url if night else None,
            add_ins="Verify that the distance description (from trailhead for night 1, or between nights) is supported or reasonably indicated on the page or map.",
            critical=True,
        ))

    camp_node = evaluator.add_parallel(
        id="Camping_Locations_4_Nights",
        desc="Provide required camping location details for each of the four nights, including distances and URLs.",
        parent=parent,
        critical=True,
    )
    night_node_builder("Night_1", "Night 1", plan.night1)
    night_node_builder("Night_2", "Night 2", plan.night2)
    night_node_builder("Night_3", "Night 3", plan.night3)
    night_node_builder("Night_4", "Night 4", plan.night4)


async def build_route_description(evaluator: Evaluator, parent, plan: TripPlanExtraction) -> None:
    r_node = evaluator.add_parallel(
        id="Route_Description",
        desc="Describe the route configuration and total distance, and provide a URL to a map/route source.",
        parent=parent,
        critical=True,
    )
    # URL presence first
    add_url_existence_node(
        evaluator,
        r_node,
        "Route_Map_or_Trail_Map_URL",
        "Provide a reference URL showing the complete route or trail map.",
        plan.route_map_url,
        critical=True,
    )
    # Loop or shuttle
    await verify_text_leaf(
        evaluator,
        r_node,
        "Loop_or_Shuttle",
        f"Route configuration: {safe_str(plan.route_loop_or_shuttle)}.",
        sources=plan.route_map_url,
        add_ins="Verify whether the route is a loop or requires a shuttle per the map/source.",
        critical=True,
    )
    # Approx total distance
    await verify_text_leaf(
        evaluator,
        r_node,
        "Approx_Total_Distance",
        f"Approximate total distance: {safe_str(plan.approx_total_distance)}.",
        sources=plan.route_map_url,
        add_ins="Verify the approximate total distance for the complete route.",
        critical=True,
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
    Evaluate an answer for the California wilderness 4-night backpacking plan in August 2026.
    """
    # Initialize evaluator (root is non-critical by design; we add a critical top-level plan node)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract plan information
    plan: TripPlanExtraction = await evaluator.extract(
        prompt=prompt_extract_trip_plan(),
        template_class=TripPlanExtraction,
        extraction_name="trip_plan_extraction",
    )

    # Add custom info: aggregated URLs
    all_urls = gather_all_urls(plan)
    evaluator.add_custom_info(
        {"total_reference_urls": len(all_urls), "urls": all_urls},
        info_type="url_aggregation",
        info_name="aggregated_reference_urls",
    )

    # Build critical sequential plan node
    plan_node = evaluator.add_sequential(
        id="Wilderness_Backpacking_Route_Plan",
        desc="Complete plan for a 4-night wilderness backpacking trip in California for 8 adults during August 2026, satisfying all specified requirements with URL support from allowed sources.",
        parent=root,
        critical=True,
    )

    # Sequentially evaluate sections (later ones auto-skip if earlier critical fails)
    await build_trip_basics(evaluator, plan_node, plan)
    await build_source_constraints(evaluator, plan_node, plan)
    await build_wilderness_area_identification(evaluator, plan_node, plan)
    await build_trailhead_selection(evaluator, plan_node, plan)

    # Permit and regulations container node (critical, parallel)
    regs_node = evaluator.add_parallel(
        id="Permit_And_Regulations",
        desc="Provide required permit and regulation details with supporting URLs.",
        parent=plan_node,
        critical=True,
    )
    await build_permit_information(evaluator, regs_node, plan)
    await build_group_size_regulations(evaluator, regs_node, plan)
    await build_bear_canister_rules(evaluator, regs_node, plan)
    await build_camping_regulations(evaluator, regs_node, plan)
    await build_fire_restrictions(evaluator, regs_node, plan)
    await build_stay_duration_limits(evaluator, regs_node, plan)
    await build_seasonal_access(evaluator, regs_node, plan)

    # Camping locations and route description (critical)
    await build_camping_locations(evaluator, plan_node, plan)
    await build_route_description(evaluator, plan_node, plan)

    # Return evaluation summary
    return evaluator.get_summary()