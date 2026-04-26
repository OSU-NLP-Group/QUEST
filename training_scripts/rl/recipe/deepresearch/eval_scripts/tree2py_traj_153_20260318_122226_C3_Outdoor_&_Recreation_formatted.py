import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "gc_backpacking_permit_may2026"
TASK_DESCRIPTION = """You are planning a 2-night backpacking trip to Grand Canyon National Park from May 15-17, 2026. You and a friend (both U.S. residents) want to apply for an early access lottery to secure a backcountry permit.

Given the current trail closures and conditions as of late 2025:
- The North Kaibab Trail remains completely closed due to post-fire impacts
- The South Kaibab Trail is open only from the trailhead to the Tipoff (not to the Colorado River)
- The River Trail and Silver Bridge were scheduled to reopen in late 2025
- Bright Angel Campground reopened on November 1, 2025

Provide the following information:

1. Permit Lottery Timeline: What are the exact start and end dates (including time and time zone) of the early access lottery application window for your May 2026 trip? Provide a reference URL from an official NPS or Recreation.gov source.

2. Hiking Route and Campground: Describe a feasible overnight hiking route that respects current trail accessibility, and identify which specific campground you would stay at. Your route must only use trail segments that are confirmed to be open.

3. Water Availability: For your chosen route, indicate whether water sources are available along the trail or at your selected campground, and if hikers need to carry all their water.
"""


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
class PermitTimelineExtraction(BaseModel):
    start_datetime_str: Optional[str] = None
    end_datetime_str: Optional[str] = None
    timezone_str: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class ItineraryDay(BaseModel):
    date: Optional[str] = None
    description: Optional[str] = None
    route_segments: List[str] = Field(default_factory=list)
    overnight_campground: Optional[str] = None


class ItineraryExtraction(BaseModel):
    days: List[ItineraryDay] = Field(default_factory=list)
    named_trails: List[str] = Field(default_factory=list)
    named_campgrounds: List[str] = Field(default_factory=list)
    overall_route_summary: Optional[str] = None


class WaterExtraction(BaseModel):
    trail_water_statement: Optional[str] = None
    campground_water_statement: Optional[str] = None
    carry_all_water_answer: Optional[str] = None
    water_source_urls: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction Prompts
# -----------------------------------------------------------------------------
def prompt_extract_permit_timeline() -> str:
    return """
    From the answer, extract the early-access lottery application window details that the answer claims apply to a May 2026 Grand Canyon backcountry trip.

    Return:
    - start_datetime_str: the exact start date-time string including any time and time zone notation, exactly as written in the answer (e.g., "Dec 15, 2025 at 8:00 a.m. Arizona time").
    - end_datetime_str: the exact end date-time string including any time and time zone notation, exactly as written in the answer.
    - timezone_str: the explicit time zone string mentioned by the answer (e.g., "Arizona time", "MST", "MDT", "MT", "UTC-7"); if not clearly mentioned, return null.
    - source_urls: list of all official reference URLs the answer cites for this lottery timing that are from nps.gov or recreation.gov only. Exclude any other domains.

    Important:
    - Do not invent dates/times. If the answer does not provide a field, set it to null.
    - source_urls must be actual URLs present in the answer and must be on nps.gov or recreation.gov.
    """


def prompt_extract_itinerary() -> str:
    return """
    Extract the described 2-night itinerary for May 15–17, 2026.

    Return:
    - days: an array (up to 3 items) where each item has:
        - date: the calendar date string as written in the answer for that day's plan (e.g., "May 15, 2026"), or null if not stated.
        - description: brief textual description of the plan for that day (paraphrase allowed but keep key details).
        - route_segments: list of trail names or segments explicitly mentioned for that day's travel (e.g., "Bright Angel Trail to Pipe Creek Resthouse", "South Kaibab Trail to Tipoff").
        - overnight_campground: the named campground for the overnight following that day, if any (e.g., "Havasupai Gardens", "Bright Angel Campground"); else null.
    - named_trails: a flat list of distinct trail names/segments explicitly named anywhere in the route (e.g., "South Kaibab Trail", "Bright Angel Trail", "River Trail", "Silver Bridge", "North Kaibab Trail", "Tonto Trail", "Pipe Creek Resthouse", "Tipoff").
    - named_campgrounds: a flat list of the specific campground names explicitly identified for overnights (e.g., "Havasupai Gardens", "Bright Angel Campground").
    - overall_route_summary: a one-sentence condensed route summary as given in the answer (if present).

    Rules:
    - Do not infer or add trails/campgrounds not mentioned in the answer.
    - If details are missing, return null or empty lists appropriately.
    """


def prompt_extract_water() -> str:
    return """
    Extract the water availability statements for the chosen itinerary.

    Return:
    - trail_water_statement: a concise statement from the answer about water availability along the specific trails/segments used (e.g., "No potable water on South Kaibab; water at Havasupai Gardens spigot").
    - campground_water_statement: a concise statement from the answer about water availability at the selected campground(s) (e.g., "Potable water available at Havasupai Gardens").
    - carry_all_water_answer: explicitly "yes" or "no" if the answer states whether hikers must carry all their water for this plan; if the answer says it depends or is conditional, put "depends"; otherwise null.
    - water_source_urls: any URLs the answer cites specifically to support water availability (if any). Can be from any domain. If none are provided, return an empty list.

    Rules:
    - Quote or closely paraphrase only what the answer actually states.
    - Do not invent water details not mentioned in the answer.
    """


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _is_official_url(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        return ("nps.gov" in netloc) or ("recreation.gov" in netloc)
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Verification Builders
# -----------------------------------------------------------------------------
async def verify_permit_timeline(
    evaluator: Evaluator,
    parent_node,
    permit: PermitTimelineExtraction
) -> None:
    # Parent: Permit_Lottery_Timeline (parallel, critical)
    tl_node = evaluator.add_parallel(
        id="Permit_Lottery_Timeline",
        desc="Provide the early access lottery application window details for a May 2026 trip, including an official reference.",
        parent=parent_node,
        critical=True
    )

    start_str = permit.start_datetime_str or ""
    end_str = permit.end_datetime_str or ""
    tz_str = permit.timezone_str or ""
    src_urls = permit.source_urls or []

    # 1) Window_Start_End_With_Time_And_Timezone (leaf, critical)
    leaf_time_tz = evaluator.add_leaf(
        id="Window_Start_End_With_Time_And_Timezone",
        desc="Gives the exact application window start and end; each includes a time and a time zone.",
        parent=tl_node,
        critical=True
    )
    claim_time_tz = (
        f"Both of the following include an explicit clock time and a time zone: "
        f"START='{start_str}' and END='{end_str}'. "
        f"Accept timezone labels like 'Arizona time', MST/MDT/MT, UTC offsets, etc."
    )
    await evaluator.verify(
        claim=claim_time_tz,
        node=leaf_time_tz,
        additional_instruction="Judge strictly: if either start or end is missing an explicit time or an explicit time zone, mark incorrect."
    )

    # 2) Window_Matches_Stated_Rule (leaf, critical)
    leaf_rule = evaluator.add_leaf(
        id="Window_Matches_Stated_Rule",
        desc="The stated window is consistent with the provided rule: ~2 weeks long and ends on the 1st or 2nd of the month exactly four months prior to the desired start month (May 2026 -> January 2026).",
        parent=tl_node,
        critical=True
    )
    claim_rule = (
        f"Given START='{start_str}' and END='{end_str}', the application window spans roughly two weeks "
        f"(about 12–16 days) and the END date falls on January 1 or January 2, 2026 (four months before May 2026)."
    )
    await evaluator.verify(
        claim=claim_rule,
        node=leaf_rule,
        additional_instruction="Only use the provided start/end strings. If you cannot clearly conclude this from those strings, mark as incorrect."
    )

    # 3) Official_Reference_URL (leaf/converted to custom check to enforce presence)
    # Enforce at least one official URL is provided
    has_official = any(_is_official_url(u) for u in src_urls)
    evaluator.add_custom_node(
        result=has_official,
        id="Official_Reference_URL",
        desc="Provides at least one supporting reference URL from an official NPS.gov or Recreation.gov source for the lottery timeline.",
        parent=tl_node,
        critical=True
    )


async def verify_hiking_route_and_campground(
    evaluator: Evaluator,
    parent_node,
    itin: ItineraryExtraction
) -> None:
    # Parent: Hiking_Route_And_Campground (sequential, critical)
    route_node = evaluator.add_sequential(
        id="Hiking_Route_And_Campground",
        desc="Feasible 2-night itinerary (May 15–17, 2026) with specific campground(s) using only open trail segments.",
        parent=parent_node,
        critical=True
    )

    # 1) Itinerary_Is_2_Nights_On_Given_Dates (leaf)
    dates_leaf = evaluator.add_leaf(
        id="Itinerary_Is_2_Nights_On_Given_Dates",
        desc="Provides a day-by-day plan that clearly corresponds to a 2-night trip spanning May 15–17, 2026.",
        parent=route_node,
        critical=True
    )
    # Build a minimal description for context
    day_summaries = []
    for d in itin.days:
        date_txt = d.date or ""
        cg_txt = d.overnight_campground or ""
        segs = ", ".join(d.route_segments) if d.route_segments else ""
        day_summaries.append(f"[{date_txt}] segments: {segs}; overnight: {cg_txt}")
    context_snippet = " | ".join(day_summaries) if day_summaries else "no structured days extracted"
    claim_dates = (
        "The itinerary is clearly a 2-night backpacking plan spanning May 15–17, 2026, "
        "i.e., two overnights on the nights of May 15 and May 16, finishing on May 17."
    )
    await evaluator.verify(
        claim=claim_dates,
        node=dates_leaf,
        additional_instruction=f"Use only the answer text. Structured extraction: {context_snippet}"
    )

    # 2) Trail_Segments_Are_Named (custom existence)
    trails_named = bool(itin.named_trails)
    evaluator.add_custom_node(
        result=trails_named,
        id="Trail_Segments_Are_Named",
        desc="Names the trail(s) and/or trail segment(s) used in the route description.",
        parent=route_node,
        critical=True
    )

    # 3) Campground_Specified (custom existence)
    campground_named = bool(itin.named_campgrounds)
    evaluator.add_custom_node(
        result=campground_named,
        id="Campground_Specified",
        desc="Names the specific campground(s) intended for the overnight stay(s).",
        parent=route_node,
        critical=True
    )

    # 4) Closure_And_Access_Compliance (parallel group)
    compliance_node = evaluator.add_parallel(
        id="Closure_And_Access_Compliance",
        desc="The itinerary does not use any trail segment in a way that contradicts the provided closure/access constraints.",
        parent=route_node,
        critical=True
    )

    # 4.a) Does_Not_Use_North_Kaibab
    nk_leaf = evaluator.add_leaf(
        id="Does_Not_Use_North_Kaibab",
        desc="Does not use the North Kaibab Trail (given as completely closed).",
        parent=compliance_node,
        critical=True
    )
    await evaluator.verify(
        claim="The described itinerary does not include hiking on the North Kaibab Trail.",
        node=nk_leaf,
        additional_instruction="If the answer mentions North Kaibab only to say it is closed or that it will NOT be used, that is compliant."
    )

    # 4.b) South_Kaibab_Not_Beyond_Tipoff
    sk_leaf = evaluator.add_leaf(
        id="South_Kaibab_Not_Beyond_Tipoff",
        desc="If the South Kaibab Trail is used, the itinerary does not go beyond the Tipoff.",
        parent=compliance_node,
        critical=True
    )
    await evaluator.verify(
        claim="If the itinerary uses the South Kaibab Trail, it only goes as far as the Tipoff and does not continue beyond the Tipoff toward the river.",
        node=sk_leaf,
        additional_instruction="Pass if South Kaibab is not used, or if it is used only to the Tipoff. Fail if it proceeds beyond the Tipoff."
    )

    # 4.c) Bright_Angel_Not_Beyond_Pipe_Creek
    ba_leaf = evaluator.add_leaf(
        id="Bright_Angel_Not_Beyond_Pipe_Creek",
        desc="If the Bright Angel Trail is used, the itinerary does not go beyond Pipe Creek Resthouse.",
        parent=compliance_node,
        critical=True
    )
    await evaluator.verify(
        claim="If the itinerary uses the Bright Angel Trail, it does not go past (beyond) Pipe Creek Resthouse.",
        node=ba_leaf,
        additional_instruction="Pass if Bright Angel is not used, or if used only up to Pipe Creek Resthouse; fail if it goes beyond (e.g., to Havasupai Gardens or the river)."
    )

    # 4.d) River_Trail_And_Silver_Bridge_Use_Consistent_With_Closure_Dates
    rt_leaf = evaluator.add_leaf(
        id="River_Trail_And_Silver_Bridge_Use_Consistent_With_Closure_Dates",
        desc="If River Trail and/or Silver Bridge are used, usage aligns with the provided late-2025 reopening (i.e., not during closed periods).",
        parent=compliance_node,
        critical=True
    )
    await evaluator.verify(
        claim="Any described use of the River Trail or Silver Bridge occurs after their late-2025 reopening (e.g., as part of the May 2026 itinerary) and not during a stated closure period.",
        node=rt_leaf,
        additional_instruction="If the itinerary date is May 2026, this should be consistent. Fail only if the answer implies using them during their closed period."
    )

    # 5) Campground_Is_Reachable_From_Stated_Route (leaf)
    reachable_leaf = evaluator.add_leaf(
        id="Campground_Is_Reachable_From_Stated_Route",
        desc="The described route plausibly reaches the named campground(s) via the stated trail segments while respecting the provided access constraints.",
        parent=route_node,
        critical=True
    )
    route_summary = itin.overall_route_summary or ""
    trails_list = ", ".join(itin.named_trails) if itin.named_trails else ""
    camps_list = ", ".join(itin.named_campgrounds) if itin.named_campgrounds else ""
    claim_reach = (
        f"The route as described (summary: '{route_summary}'; trails: {trails_list}) plausibly reaches "
        f"the named campground(s) ({camps_list}) without violating the provided access constraints."
    )
    await evaluator.verify(
        claim=claim_reach,
        node=reachable_leaf,
        additional_instruction="Base your judgment only on the content of the answer; check for internal plausibility and consistency with the closure rules."
    )


async def verify_water_availability(
    evaluator: Evaluator,
    parent_node,
    water: WaterExtraction,
    itin: ItineraryExtraction
) -> None:
    # Parent: Water_Availability (parallel, critical)
    water_node = evaluator.add_parallel(
        id="Water_Availability",
        desc="State water availability along the chosen route and at the selected campground, and whether hikers must carry all their water.",
        parent=parent_node,
        critical=True
    )

    trails_used = ", ".join(itin.named_trails) if itin.named_trails else "no trails extracted"
    camps_used = ", ".join(itin.named_campgrounds) if itin.named_campgrounds else "no campgrounds extracted"

    # 1) Trail_Water_Availability_Stated
    trail_water_leaf = evaluator.add_leaf(
        id="Trail_Water_Availability_Stated",
        desc="States whether water sources are available along the trail(s)/segment(s) used in the itinerary (including explicitly noting that South Kaibab has no water if it is part of the route).",
        parent=water_node,
        critical=True
    )
    claim_trail_water = (
        f"The answer explicitly states water availability along the actual trails/segments used ({trails_used}); "
        f"if South Kaibab is part of the route, it explicitly notes that there is no potable water on South Kaibab."
    )
    await evaluator.verify(
        claim=claim_trail_water,
        node=trail_water_leaf,
        additional_instruction=f"Use the answer content only. Provided extraction: '{water.trail_water_statement or ''}'."
    )

    # 2) Campground_Water_Availability_Stated
    cg_water_leaf = evaluator.add_leaf(
        id="Campground_Water_Availability_Stated",
        desc="States whether water is available at the selected campground.",
        parent=water_node,
        critical=True
    )
    claim_cg_water = (
        f"The answer explicitly states whether potable water is available at the named campground(s) ({camps_used})."
    )
    await evaluator.verify(
        claim=claim_cg_water,
        node=cg_water_leaf,
        additional_instruction=f"Use the answer content only. Provided extraction: '{water.campground_water_statement or ''}'."
    )

    # 3) Carry_All_Water_Answered (custom yes/no/depends presence)
    carry_answer = (water.carry_all_water_answer or "").strip().lower()
    carry_present = carry_answer in {"yes", "no", "depends"}
    evaluator.add_custom_node(
        result=carry_present,
        id="Carry_All_Water_Answered",
        desc="Explicitly answers whether hikers need to carry all their water for the chosen plan (yes/no).",
        parent=water_node,
        critical=True
    )


# -----------------------------------------------------------------------------
# Main Evaluation Entry Point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the Grand Canyon May 15–17, 2026 backpacking permit and itinerary task.
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
        default_model=model
    )

    # Record the constraints for transparency in the final report
    evaluator.add_custom_info(
        info={
            "given_constraints_late_2025": {
                "north_kaibab": "closed",
                "south_kaibab": "open only to Tipoff",
                "river_trail_and_silver_bridge": "scheduled to reopen in late 2025",
                "bright_angel_campground": "reopened Nov 1, 2025"
            }
        },
        info_type="constraints",
        info_name="provided_closures_and_access"
    )

    # Extract required info (in parallel)
    permit_task = evaluator.extract(
        prompt=prompt_extract_permit_timeline(),
        template_class=PermitTimelineExtraction,
        extraction_name="permit_timeline"
    )
    itin_task = evaluator.extract(
        prompt=prompt_extract_itinerary(),
        template_class=ItineraryExtraction,
        extraction_name="itinerary"
    )
    water_task = evaluator.extract(
        prompt=prompt_extract_water(),
        template_class=WaterExtraction,
        extraction_name="water"
    )

    permit, itinerary, water = await asyncio.gather(permit_task, itin_task, water_task)

    # Build critical parent node: Complete_Trip_Plan
    plan_root = evaluator.add_parallel(
        id="Complete_Trip_Plan",
        desc="Provide (1) the early-access permit lottery timeline with an official URL, (2) a feasible 2-night itinerary and specific campground using only trail segments consistent with the provided access constraints, and (3) water availability guidance for the chosen route/campground.",
        parent=root,
        critical=True
    )

    # Sub-verifications
    await verify_permit_timeline(evaluator, plan_root, permit)
    await verify_hiking_route_and_campground(evaluator, plan_root, itinerary)
    await verify_water_availability(evaluator, plan_root, water, itinerary)

    # Return structured summary
    return evaluator.get_summary()