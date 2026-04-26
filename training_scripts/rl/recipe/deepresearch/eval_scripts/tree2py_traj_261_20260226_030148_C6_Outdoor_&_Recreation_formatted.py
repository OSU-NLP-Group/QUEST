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
TASK_ID = "whiting_ranch_march_2026_plan"
TASK_DESCRIPTION = (
    "Plan a morning group hiking trip to Whiting Ranch Wilderness Park in Orange County, California for a family of 4 "
    "(2 adults, 2 children ages 8 and 10) on a weekday in March 2026. The family wants to complete an easy-to-moderate "
    "difficulty hike that is 4-6 miles round trip and includes viewing the Red Rock Canyon scenic feature. Your hiking "
    "plan must: (1) Identify a specific primary trail route that meets the difficulty and distance requirements, "
    "(2) Ensure all current mountain lion safety protocols (established after the November 2025 incidents) are followed, "
    "(3) Propose appropriate timing that complies with park operating hours and avoids peak mountain lion activity periods, "
    "(4) Confirm the group composition meets safety requirements, and (5) Include the park office emergency contact number "
    "for wildlife sightings. Provide your plan with the following components: primary trail name and route description, "
    "distance and difficulty rating of the selected trail, estimated start and end times, group safety protocols to be followed, "
    "emergency contact information, and supporting reference URLs for trail information, park hours, and safety protocols."
)

REQUIRED_PARK_OFFICE_NUMBER = "949-923-2245"
PARK_HOURS_CANONICAL = "7 AM to sunset"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PlanExtraction(BaseModel):
    # Route and trail
    primary_trail_name: Optional[str] = None
    route_description: Optional[str] = None
    red_rock_statement: Optional[str] = None
    distance_round_trip: Optional[str] = None
    difficulty: Optional[str] = None

    # Timing
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    within_park_hours_statement: Optional[str] = None
    avoids_dawn_dusk_statement: Optional[str] = None
    rain_closure_statement: Optional[str] = None

    # Group composition and behavior
    group_composition_statement: Optional[str] = None
    do_not_hike_alone_statement: Optional[str] = None
    group_stays_together_statement: Optional[str] = None

    # Safety protocols
    make_noise_statement: Optional[str] = None
    maintain_25_yards_statement: Optional[str] = None
    encounter_protocol_text: Optional[str] = None
    emergency_phone_number: Optional[str] = None

    # Other park rules / logistics
    dogs_not_permitted_statement: Optional[str] = None
    parking_fee_statement: Optional[str] = None

    # URLs
    trail_info_urls: List[str] = Field(default_factory=list)
    park_hours_urls: List[str] = Field(default_factory=list)
    safety_protocol_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan_fields() -> str:
    return """
    Extract the following fields exactly as presented in the answer. Do not invent any content. Use null for any field that is not explicitly provided.

    ROUTE AND TRAIL
    - primary_trail_name: The single named primary trail or named route for the hike (e.g., "Red Rock Canyon via Borrego Trail").
    - route_description: A prose description of the intended route (including trailhead/parking, trail names/junctions in order, turnaround point or loop completion).
    - red_rock_statement: The sentence/phrase where the plan mentions visiting/viewing Red Rock Canyon; if not present, return null.
    - distance_round_trip: The round-trip distance as stated (e.g., "4.8 miles", "4–6 miles", "about 5 miles").
    - difficulty: The difficulty label or phrase as stated (e.g., "easy", "easy to moderate", "moderate").

    TIMING
    - start_time: The proposed trail start time, as text (e.g., "8:00 AM").
    - end_time: The proposed planned end/finish time, as text (e.g., "11:30 AM").
    - within_park_hours_statement: The sentence/phrase asserting the hike is within park hours (7 AM to sunset); if not present, return null.
    - avoids_dawn_dusk_statement: The sentence/phrase asserting that dawn/dusk are avoided; if not present, return null.
    - rain_closure_statement: The sentence/phrase acknowledging trails may be closed up to 3 days after rain and/or asking to check status/reschedule; if not present, return null.

    GROUP COMPOSITION AND BEHAVIOR
    - group_composition_statement: The sentence/phrase confirming group composition (2 adults, 2 children ages 8 and 10); if not present, return null.
    - do_not_hike_alone_statement: The sentence/phrase explicitly confirming 'do NOT hike alone' or hiking with 2+ people; if absent, return null.
    - group_stays_together_statement: The sentence/phrase confirming the group stays together, especially children kept close; if absent, return null.

    SAFETY PROTOCOLS
    - make_noise_statement: The sentence/phrase instructing to make noise while hiking; if absent, return null.
    - maintain_25_yards_statement: The sentence/phrase instructing to stay at least 25 yards (or ~75 feet) away from wildlife; if absent, return null.
    - encounter_protocol_text: The exact text in the plan that lays out the mountain lion encounter steps (do not run; hold ground; make noise; wave arms to appear larger; do not crouch or turn your back; throw items/rocks to scare it away). If multiple lines, include them; if absent, return null.
    - emergency_phone_number: Extract the phone number provided as the park office emergency contact for wildlife (digits+separators). If multiple numbers, extract the one identified for park office wildlife reporting.

    OTHER PARK RULES / LOGISTICS
    - dogs_not_permitted_statement: The sentence/phrase acknowledging dogs are not permitted; if absent, return null.
    - parking_fee_statement: The sentence/phrase acknowledging the $3 daily parking fee; if absent, return null.

    URLS
    - trail_info_urls: All URLs provided in the answer that are for trail information in Whiting Ranch or the chosen route.
    - park_hours_urls: All URLs provided for park hours.
    - safety_protocol_urls: All URLs provided for safety protocols (e.g., mountain lion guidance).
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _merge_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            if isinstance(u, str):
                u2 = u.strip()
                if u2 and u2 not in seen:
                    merged.append(u2)
                    seen.add(u2)
    return merged


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_route_selection_checks(evaluator: Evaluator, parent_node, ex: PlanExtraction) -> None:
    route_node = evaluator.add_parallel(
        id="Route_Selection_And_Red_Rock_Canyon",
        desc="Plan identifies a specific primary trail route that meets distance/difficulty requirements and includes Red Rock Canyon",
        parent=parent_node,
        critical=True,
    )

    # Primary trail name provided (existence check)
    evaluator.add_custom_node(
        result=_non_empty(ex.primary_trail_name),
        id="Primary_Trail_Name_Provided",
        desc="Provides a specific primary trail name (or named route) for the hike",
        parent=route_node,
        critical=True,
    )

    # Route description has key elements
    node_desc = "Provides a route description that includes (a) start/trailhead or parking area, (b) named trail(s)/junction(s) in the order taken, and (c) turnaround point or loop completion description"
    route_desc_node = evaluator.add_leaf(
        id="Route_Description_Includes_Key_Elements",
        desc=node_desc,
        parent=route_node,
        critical=True,
    )
    claim = (
        "The hiking plan's route description includes all of the following: "
        "(a) a start/trailhead or parking area, "
        "(b) named trail(s)/junction(s) in the order taken, and "
        "(c) a turnaround point or loop completion description."
    )
    await evaluator.verify(
        claim=claim,
        node=route_desc_node,
        additional_instruction="Check the full answer for these three components in the route description. Accept equivalent phrasing and synonyms."
    )

    # Includes Red Rock Canyon feature
    red_rock_node = evaluator.add_leaf(
        id="Includes_Red_Rock_Canyon_Feature",
        desc="Route explicitly includes viewing/visiting the Red Rock Canyon scenic feature",
        parent=route_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The planned route explicitly includes visiting or viewing the Red Rock Canyon scenic feature in Whiting Ranch Wilderness Park.",
        node=red_rock_node,
        additional_instruction="Look for explicit inclusion of 'Red Rock Canyon' in the plan."
    )

    # Distance within 4–6 miles RT
    distance_node = evaluator.add_leaf(
        id="Distance_Within_4_to_6_Round_Trip",
        desc="States a round-trip distance that is within 4–6 miles",
        parent=route_node,
        critical=True,
    )
    dist_text = ex.distance_round_trip or "not provided"
    await evaluator.verify(
        claim=f"The plan states a round-trip distance within 4 to 6 miles. The stated distance is: '{dist_text}'.",
        node=distance_node,
        additional_instruction="Allow reasonable phrasing such as 'about 5 miles', 'roughly 4.8 miles', or a range contained within 4–6 miles. If no distance is stated, this should be incorrect."
    )

    # Difficulty easy-to-moderate
    difficulty_node = evaluator.add_leaf(
        id="Difficulty_Easy_to_Moderate",
        desc="States a difficulty rating/category for the selected hike and it is easy-to-moderate",
        parent=route_node,
        critical=True,
    )
    diff_text = ex.difficulty or "not provided"
    await evaluator.verify(
        claim=f"The plan states the selected hike is 'easy-to-moderate' (or equivalent). The difficulty text is: '{diff_text}'.",
        node=difficulty_node,
        additional_instruction="Accept 'easy', 'easy to moderate', 'easy-moderate', 'easy/moderate', or 'moderate' if clearly framed as easy-to-moderate. Reject 'strenuous' or 'hard'."
    )

    # Trail info reference URL provided (existence)
    evaluator.add_custom_node(
        result=len(ex.trail_info_urls) > 0,
        id="Trail_Info_Reference_URL",
        desc="Provides at least one supporting reference URL for trail information",
        parent=route_node,
        critical=True,
    )


async def build_timing_and_operations_checks(evaluator: Evaluator, parent_node, ex: PlanExtraction) -> None:
    timing_node = evaluator.add_parallel(
        id="Timing_And_Park_Operations",
        desc="Plan provides timing that complies with park hours and avoids peak mountain lion activity times; accounts for rain-related closures",
        parent=parent_node,
        critical=True,
    )

    # Start and end times provided (existence)
    evaluator.add_custom_node(
        result=_non_empty(ex.start_time) and _non_empty(ex.end_time),
        id="Start_And_End_Times_Provided",
        desc="Provides estimated start and end times for the hike",
        parent=timing_node,
        critical=True,
    )

    # Within park hours 7AM to sunset
    within_hours_node = evaluator.add_leaf(
        id="Within_Park_Hours_7AM_to_Sunset",
        desc="Proposed hiking window is stated to be within park operating hours (7 AM to sunset)",
        parent=timing_node,
        critical=True,
    )
    st = ex.start_time or "not provided"
    et = ex.end_time or "not provided"
    await evaluator.verify(
        claim=f"The plan explicitly schedules the hike within park operating hours (7 AM to sunset). The proposed start/end times are {st} to {et}, and/or the plan states it is within park hours.",
        node=within_hours_node,
        additional_instruction="Do not check the actual sunset time. Only verify the plan's explicit compliance statement or that the provided window is framed as within '7 AM to sunset'."
    )

    # Explicitly avoids dawn and dusk
    avoids_node = evaluator.add_leaf(
        id="Explicitly_Avoids_Dawn_And_Dusk",
        desc="Plan explicitly states the hike is scheduled in daytime and avoids dawn/dusk (peak mountain lion activity periods)",
        parent=timing_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The plan explicitly states the hike will be in daytime and avoids dawn and dusk (peak mountain lion activity periods).",
        node=avoids_node,
        additional_instruction="Look for phrases such as 'avoid dawn/dusk', 'daytime hours', 'not during crepuscular times'."
    )

    # Rain closure contingency
    rain_node = evaluator.add_leaf(
        id="Rain_Closure_Contingency",
        desc="Acknowledges/handles the constraint that trails may be closed for up to 3 days following rain (e.g., check status/reschedule)",
        parent=timing_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The plan acknowledges that Whiting Ranch trails may be closed for up to 3 days after rain and suggests checking status or rescheduling.",
        node=rain_node,
        additional_instruction="Accept wording like 'closed up to 3 days after rain', 'check status after rain', or equivalent."
    )

    # Park hours reference URL provided (existence)
    evaluator.add_custom_node(
        result=len(ex.park_hours_urls) > 0,
        id="Park_Hours_Reference_URL",
        desc="Provides at least one supporting reference URL for park hours",
        parent=timing_node,
        critical=True,
    )


async def build_group_composition_checks(evaluator: Evaluator, parent_node, ex: PlanExtraction) -> None:
    group_node = evaluator.add_parallel(
        id="Group_Composition_Requirements",
        desc="Plan confirms the family group composition and behavior meets stated safety requirements",
        parent=parent_node,
        critical=True,
    )

    # Family of 4 matches prompt
    fam_node = evaluator.add_leaf(
        id="Family_Of_4_Matches_Prompt",
        desc="Plan reflects the stated group: 2 adults and 2 children ages 8 and 10",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The plan reflects the group as 2 adults and 2 children ages 8 and 10.",
        node=fam_node,
        additional_instruction="Look for explicit mention of two adults and two children with ages 8 and 10."
    )

    # Do not hike alone
    no_alone_node = evaluator.add_leaf(
        id="Do_Not_Hike_Alone",
        desc="Explicitly confirms compliance with the 'Do NOT hike alone' mandatory group requirement (2+ people)",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The plan explicitly confirms compliance with the 'Do NOT hike alone' requirement (hike in a group of at least two).",
        node=no_alone_node,
        additional_instruction="Look for explicit text like 'Do not hike alone', 'always hike with at least one other person', or equivalent."
    )

    # Group stays together; children kept close
    together_node = evaluator.add_leaf(
        id="Group_Stays_Together_Children_Kept_Close",
        desc="Includes the requirement that the group stays together, especially children",
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The plan includes the requirement that the group stays together and that children are kept close to adults.",
        node=together_node,
        additional_instruction="Look for explicit phrasing that children stay close and the group remains together."
    )


async def build_mountain_lion_safety_checks(evaluator: Evaluator, parent_node, ex: PlanExtraction) -> None:
    safety_node = evaluator.add_parallel(
        id="Mountain_Lion_And_Wildlife_Safety",
        desc="Plan includes all required mountain lion safety protocols and wildlife distancing/reporting requirements",
        parent=parent_node,
        critical=True,
    )

    # Make noise while hiking
    noise_node = evaluator.add_leaf(
        id="Make_Noise_While_Hiking",
        desc="Includes making noise while hiking to alert mountain lions",
        parent=safety_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The plan instructs hikers to make noise while hiking to alert wildlife/mountain lions.",
        node=noise_node,
        additional_instruction="Look for 'make noise', 'talk', 'clap', 'announce presence', or equivalent."
    )

    # Maintain wildlife distance 25 yards
    dist_node = evaluator.add_leaf(
        id="Maintain_Wildlife_Distance_25_Yards",
        desc="Includes staying at least 25 yards away from most wildlife",
        parent=safety_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The plan includes staying at least 25 yards (approximately 75 feet) away from wildlife.",
        node=dist_node,
        additional_instruction="Accept '25 yards' or approximations like '75 feet'."
    )

    # Full encounter protocol steps
    encounter_node = evaluator.add_leaf(
        id="Encounter_Protocol_Includes_All_Steps",
        desc="Includes the full mountain lion encounter protocol: do not run; hold your ground; make noise; wave arms to appear larger; do not crouch or turn your back; throw items to scare it away",
        parent=safety_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The plan includes the full mountain lion encounter protocol: do not run; hold your ground; make noise; "
            "wave arms to appear larger; do not crouch or turn your back; and throw items (e.g., rocks) to scare it away."
        ),
        node=encounter_node,
        additional_instruction="All listed steps should be present or clearly implied by equivalent phrasing."
    )

    # Report sightings to park office number (inclusion of exact number)
    phone_node = evaluator.add_leaf(
        id="Report_Sightings_To_Park_Office_Number",
        desc="Includes the park office emergency contact number for wildlife sightings and matches the required number (949-923-2245)",
        parent=safety_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The plan includes the park office emergency contact number for wildlife sightings as {REQUIRED_PARK_OFFICE_NUMBER}."
        ),
        node=phone_node,
        additional_instruction=(
            f"Verify that the exact number {REQUIRED_PARK_OFFICE_NUMBER} appears in the plan and is identified as the park office contact for wildlife sightings."
        )
    )

    # Safety protocols reference URL provided (existence)
    evaluator.add_custom_node(
        result=len(ex.safety_protocol_urls) > 0,
        id="Safety_Protocols_Reference_URL",
        desc="Provides at least one supporting reference URL for safety protocols",
        parent=safety_node,
        critical=True,
    )


async def build_other_rules_checks(evaluator: Evaluator, parent_node, ex: PlanExtraction) -> None:
    rules_node = evaluator.add_parallel(
        id="Other_Park_Rules_And_Logistics",
        desc="Plan complies with other stated park rules/logistics constraints",
        parent=parent_node,
        critical=True,
    )

    all_urls = _merge_urls(ex.trail_info_urls, ex.park_hours_urls, ex.safety_protocol_urls)

    # Dogs not permitted
    dogs_node = evaluator.add_leaf(
        id="Dogs_Not_Permitted",
        desc="Acknowledges dogs are not permitted in the park",
        parent=rules_node,
        critical=True,
    )
    # Prefer verifying presence in the plan; optionally support with URLs if available.
    if all_urls:
        await evaluator.verify(
            claim="The plan explicitly acknowledges that dogs are not permitted in Whiting Ranch Wilderness Park.",
            node=dogs_node,
            sources=all_urls,
            additional_instruction="Confirm both that the plan states 'no dogs' and that at least one provided reference page supports that rule."
        )
    else:
        await evaluator.verify(
            claim="The plan explicitly acknowledges that dogs are not permitted in Whiting Ranch Wilderness Park.",
            node=dogs_node,
            additional_instruction="Rely on the plan text only since no URLs were provided."
        )

    # Parking fee $3
    fee_node = evaluator.add_leaf(
        id="Parking_Fee_3_Dollars",
        desc="Acknowledges the $3 daily parking fee",
        parent=rules_node,
        critical=True,
    )
    if all_urls:
        await evaluator.verify(
            claim="The plan acknowledges there is a $3 daily parking fee at Whiting Ranch Wilderness Park, and this fee is accurate per the provided references.",
            node=fee_node,
            sources=all_urls,
            additional_instruction="Verify that the plan mentions the $3 fee and that at least one provided reference supports the $3 daily parking fee."
        )
    else:
        await evaluator.verify(
            claim="The plan acknowledges there is a $3 daily parking fee at Whiting Ranch Wilderness Park.",
            node=fee_node,
            additional_instruction="Rely on the plan text only since no URLs were provided."
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
    Evaluate an answer for the Whiting Ranch morning hiking plan task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Overall categories are independent; all are required
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

    # Extract structured fields from the answer
    extracted: PlanExtraction = await evaluator.extract(
        prompt=prompt_extract_plan_fields(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction",
    )

    # Add ground truth/constraining info for transparency (not used for scoring)
    evaluator.add_ground_truth({
        "required_phone_number": REQUIRED_PARK_OFFICE_NUMBER,
        "park_hours_canonical": PARK_HOURS_CANONICAL,
        "distance_requirement_round_trip_miles": "4–6",
        "difficulty_requirement": "easy-to-moderate",
        "must_include_feature": "Red Rock Canyon",
        "group_requirement": "Do not hike alone; group stays together; children kept close",
        "rain_closure_policy": "Trails may be closed up to 3 days after rain",
    })

    # Build top-level critical node container (since the framework root is non-critical by default)
    main_node = evaluator.add_parallel(
        id="Root",
        desc="Complete hiking trip plan for Whiting Ranch Wilderness Park that satisfies all stated route, timing, safety, park-rule, and reference-URL requirements",
        parent=root,
        critical=True,
    )

    # Build subtrees
    await build_route_selection_checks(evaluator, main_node, extracted)
    await build_timing_and_operations_checks(evaluator, main_node, extracted)
    await build_group_composition_checks(evaluator, main_node, extracted)
    await build_mountain_lion_safety_checks(evaluator, main_node, extracted)
    await build_other_rules_checks(evaluator, main_node, extracted)

    # Return the summary with verification tree and scores
    return evaluator.get_summary()