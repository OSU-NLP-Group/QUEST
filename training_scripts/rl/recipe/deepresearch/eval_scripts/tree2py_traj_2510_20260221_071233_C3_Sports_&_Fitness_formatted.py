import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "alex_honnold_taipei101_tmd_2026"
TASK_DESCRIPTION = (
    "In January 2026, renowned climber Alex Honnold completed a live-broadcast free solo climb "
    "(without ropes or safety equipment) of a famous skyscraper in Taiwan. This building is equipped "
    "with a primary earthquake and wind protection system. Identify this building and its protection system, "
    "then provide the following specifications of this system: (1) The weight in metric tons, (2) The diameter in meters, "
    "(3) The floor range between which this system is suspended (specify both the upper and lower floors). "
    "For each piece of information, provide supporting reference URLs that verify your answer."
)

# Expected ground truths for verification (used to check the answer states these facts)
EXPECTED = {
    "building_name": "Taipei 101",
    "building_country": "Taiwan",
    "building_height_m": "508",
    "building_height_ft": "1,667",
    "building_floors_above_ground": "101",
    "climb_date_text": "January 24, 2026",
    "climb_duration_text": "1 hour, 31 minutes, and 34 seconds",
    "broadcast_service": "Netflix",
    "broadcast_title": "Skyscraper Live",
    "tallest_urban": "tallest urban free solo in history",
    "system_name": "tuned mass damper",
    "system_function": "earthquake and wind protection",
    "tmd_weight_metric_tons": "660",
    "tmd_diameter_meters": "5.5",
    "tmd_floor_upper": "92",
    "tmd_floor_lower": "87",
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AnswerExtraction(BaseModel):
    # Building identity and location
    building_name: Optional[str] = None
    building_name_sources: List[str] = Field(default_factory=list)

    building_location_text: Optional[str] = None
    building_in_taiwan_sources: List[str] = Field(default_factory=list)

    # Building height
    building_height_m: Optional[str] = None
    building_height_ft: Optional[str] = None
    building_height_sources: List[str] = Field(default_factory=list)

    # Floors above ground
    floors_above_ground: Optional[str] = None
    floors_above_ground_sources: List[str] = Field(default_factory=list)

    # Honnold free solo occurred
    free_solo_statement_text: Optional[str] = None
    honnold_free_solo_sources: List[str] = Field(default_factory=list)

    # Date of climb
    climb_date_text: Optional[str] = None
    climb_date_sources: List[str] = Field(default_factory=list)

    # Duration
    climb_duration_text: Optional[str] = None
    climb_duration_sources: List[str] = Field(default_factory=list)

    # Broadcast
    broadcast_text: Optional[str] = None
    broadcast_sources: List[str] = Field(default_factory=list)

    # Tallest claim
    tallest_urban_claim_text: Optional[str] = None
    tallest_urban_sources: List[str] = Field(default_factory=list)

    # Protection system identification
    system_name_text: Optional[str] = None
    system_name_sources: List[str] = Field(default_factory=list)

    # System function (earthquake & wind)
    system_function_text: Optional[str] = None
    system_function_sources: List[str] = Field(default_factory=list)

    # TMD specifications
    tmd_weight_metric_tons_text: Optional[str] = None
    tmd_weight_sources: List[str] = Field(default_factory=list)

    tmd_diameter_meters_text: Optional[str] = None
    tmd_diameter_sources: List[str] = Field(default_factory=list)

    tmd_floor_upper_text: Optional[str] = None
    tmd_floor_lower_text: Optional[str] = None
    tmd_floors_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_answer_fields() -> str:
    return """
Extract the following fields from the provided answer text. For every claim, also extract the supporting URL(s) that the answer explicitly cites for that specific claim. If a field is not present in the answer, set it to null, and for sources, return an empty array.

Building and climb:
- building_name: the building identified (e.g., "Taipei 101")
- building_name_sources: URL(s) the answer cites to support the building identification
- building_location_text: where the building is (e.g., "Taiwan", "Taipei, Taiwan")
- building_in_taiwan_sources: URL(s) the answer cites to support that the building is in Taiwan
- building_height_m: height in meters (text as presented, e.g., "508 m" or "508")
- building_height_ft: height in feet (text as presented, e.g., "1,667 ft" or "1667")
- building_height_sources: URL(s) the answer cites for the height
- floors_above_ground: floors above ground (text as presented, e.g., "101")
- floors_above_ground_sources: URL(s) the answer cites for the floors
- free_solo_statement_text: text indicating Alex Honnold completed a free solo (no ropes/safety) climb
- honnold_free_solo_sources: URL(s) the answer cites that support the free solo occurrence
- climb_date_text: the date the climb occurred (e.g., "January 24, 2026", "Jan 24, 2026")
- climb_date_sources: URL(s) the answer cites that support the date
- climb_duration_text: the duration of the climb (e.g., "1 hour, 31 minutes, and 34 seconds", "1:31:34")
- climb_duration_sources: URL(s) the answer cites that support the duration
- broadcast_text: text indicating the live broadcast platform and title (e.g., "broadcast live on Netflix as 'Skyscraper Live'")
- broadcast_sources: URL(s) the answer cites that support the broadcast claim
- tallest_urban_claim_text: text indicating this was the tallest urban free solo in history
- tallest_urban_sources: URL(s) the answer cites that support the "tallest urban free solo" claim

Protection system:
- system_name_text: name of the primary protection system (e.g., "tuned mass damper", "TMD")
- system_name_sources: URL(s) the answer cites for the system name
- system_function_text: text indicating the system is for earthquake and wind protection
- system_function_sources: URL(s) the answer cites that support the function

TMD specifications:
- tmd_weight_metric_tons_text: TMD weight in metric tons as stated in the answer (e.g., "660 metric tons", "660 t")
- tmd_weight_sources: URL(s) the answer cites for TMD weight
- tmd_diameter_meters_text: TMD diameter in meters as stated (e.g., "5.5 meters", "5.5 m")
- tmd_diameter_sources: URL(s) the answer cites for TMD diameter
- tmd_floor_upper_text: the upper floor number between which the TMD is suspended (e.g., "92")
- tmd_floor_lower_text: the lower floor number (e.g., "87")
- tmd_floors_sources: URL(s) the answer cites for the floor range

IMPORTANT FOR URL FIELDS:
- Extract only valid URLs explicitly present in the answer (plain or Markdown links).
- If a field lacks dedicated supporting URLs, return an empty array for its sources.
- Do not invent URLs; do not copy unrelated URLs.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(urls: Optional[List[str]]) -> List[str]:
    return urls if (urls and isinstance(urls, list)) else []


async def add_stated_and_supported_check(
    evaluator: Evaluator,
    parent,
    base_id: str,
    overall_desc: str,
    stated_claim: str,
    sources: List[str],
    supported_claim: str,
    stated_additional_instruction: Optional[str] = None,
    supported_additional_instruction: Optional[str] = None,
) -> None:
    """
    Build a critical sequential sub-tree that enforces:
      1) At least one supporting URL is provided
      2) The answer explicitly states the claim
      3) The provided URL(s) support the claim
    """
    # Group node (critical & sequential to enforce order)
    group = evaluator.add_sequential(
        id=f"{base_id}_group",
        desc=overall_desc,
        parent=parent,
        critical=True,
    )

    # 1) Has sources
    evaluator.add_custom_node(
        result=len(_safe_sources(sources)) > 0,
        id=f"{base_id}_sources_present",
        desc=f"{overall_desc} — sources provided",
        parent=group,
        critical=True,
    )

    # 2) Stated in the answer (simple verify; checks the answer text)
    stated_leaf = evaluator.add_leaf(
        id=f"{base_id}_stated_in_answer",
        desc=f"{overall_desc} — stated in the answer",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=stated_claim,
        node=stated_leaf,
        additional_instruction=stated_additional_instruction
        or "Check the provided answer text for this exact claim. Allow reasonable wording variations.",
    )

    # 3) Supported by the cited source(s)
    supported_leaf = evaluator.add_leaf(
        id=f"{base_id}_source_supported",
        desc=f"{overall_desc} — supported by cited sources",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=supported_claim,
        node=supported_leaf,
        sources=_safe_sources(sources),
        additional_instruction=supported_additional_instruction
        or "Verify the claim using the provided webpage(s). Allow minor wording/formatting variations but ensure explicit support.",
    )


async def add_supported_only_check(
    evaluator: Evaluator,
    parent,
    base_id: str,
    overall_desc: str,
    sources: List[str],
    supported_claim: str,
    supported_additional_instruction: Optional[str] = None,
) -> None:
    """
    Build a critical sequential sub-tree that enforces:
      1) At least one supporting URL is provided
      2) The provided URL(s) support the claim
    This is used where a separate 'stated in answer' check is unnecessary.
    """
    group = evaluator.add_sequential(
        id=f"{base_id}_group",
        desc=overall_desc,
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(_safe_sources(sources)) > 0,
        id=f"{base_id}_sources_present",
        desc=f"{overall_desc} — sources provided",
        parent=group,
        critical=True,
    )

    supported_leaf = evaluator.add_leaf(
        id=f"{base_id}_source_supported",
        desc=f"{overall_desc} — supported by cited sources",
        parent=group,
        critical=True,
    )
    await evaluator.verify(
        claim=supported_claim,
        node=supported_leaf,
        sources=_safe_sources(sources),
        additional_instruction=supported_additional_instruction
        or "Verify the claim using the provided webpage(s). Allow minor wording/formatting variations but ensure explicit support.",
    )


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_building_and_climb_constraints(
    evaluator: Evaluator,
    parent,
    extracted: AnswerExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="building_and_climb_constraints",
        desc="Verify the building identity and all climb/building constraints, each supported by reference URL(s).",
        parent=parent,
        critical=True,
    )

    # 1) Building is Taipei 101
    await add_stated_and_supported_check(
        evaluator,
        node,
        base_id="building_is_taipei_101",
        overall_desc="Answer identifies the building specifically as 'Taipei 101' AND provides at least one supporting reference URL.",
        stated_claim="The provided answer identifies the building as 'Taipei 101' (case-insensitive, allow minor variations like 'TAIPEI 101').",
        sources=_safe_sources(extracted.building_name_sources),
        supported_claim="The building being referred to is named 'Taipei 101'.",
        stated_additional_instruction="Check in the answer that 'Taipei 101' is identified as the building (case-insensitive).",
        supported_additional_instruction="Verify that the cited source(s) explicitly identify the building as 'Taipei 101'.",
    )

    # 2) Building is in Taiwan
    await add_stated_and_supported_check(
        evaluator,
        node,
        base_id="building_in_taiwan",
        overall_desc="Answer states the building is in Taiwan AND provides at least one supporting reference URL.",
        stated_claim="The provided answer explicitly states that the building is in Taiwan.",
        sources=_safe_sources(extracted.building_in_taiwan_sources),
        supported_claim="Taipei 101 is located in Taiwan (e.g., Taipei, Taiwan).",
        supported_additional_instruction="Verify that the cited source(s) explicitly indicate that Taipei 101 is in Taiwan.",
    )

    # 3) Building height: 508 m (1,667 ft)
    await add_stated_and_supported_check(
        evaluator,
        node,
        base_id="building_height_508m_1667ft",
        overall_desc="Answer states the building height is 508 meters (1,667 feet) AND provides at least one supporting reference URL.",
        stated_claim="The answer states the building height is 508 meters (1,667 feet). Allow minor formatting differences (e.g., '508 m', '1,667 ft', or '1667 feet').",
        sources=_safe_sources(extracted.building_height_sources),
        supported_claim="Taipei 101 has a height of 508 meters (1,667 feet).",
        supported_additional_instruction="Verify that the source(s) explicitly report the height as 508 m (1,667 ft); minor formatting differences are acceptable.",
    )

    # 4) Building 101 floors above ground
    await add_stated_and_supported_check(
        evaluator,
        node,
        base_id="building_101_floors_above_ground",
        overall_desc="Answer states the building has 101 floors above ground AND provides at least one supporting reference URL.",
        stated_claim="The answer states the building has 101 floors above ground.",
        sources=_safe_sources(extracted.floors_above_ground_sources),
        supported_claim="Taipei 101 has 101 floors above ground.",
    )

    # 5) Honnold free solo climb occurred (no ropes)
    await add_stated_and_supported_check(
        evaluator,
        node,
        base_id="honnold_free_solo_occurred",
        overall_desc="Answer states Alex Honnold completed a free solo climb (no ropes/safety equipment) of the building AND provides at least one supporting reference URL.",
        stated_claim="The answer states Alex Honnold completed a free solo climb (without ropes or safety equipment) of the building.",
        sources=_safe_sources(extracted.honnold_free_solo_sources),
        supported_claim="Alex Honnold completed a free solo (no ropes/safety equipment) climb of Taipei 101.",
        supported_additional_instruction="Verify that the source(s) explicitly describe Honnold's climb of Taipei 101 as free solo (no ropes/safety equipment).",
    )

    # 6) Climb date: January 24, 2026
    await add_stated_and_supported_check(
        evaluator,
        node,
        base_id="climb_date_jan_24_2026",
        overall_desc="Answer states the climb occurred on January 24, 2026 AND provides at least one supporting reference URL.",
        stated_claim="The answer states the climb occurred on January 24, 2026 (allow variants like 'Jan 24, 2026').",
        sources=_safe_sources(extracted.climb_date_sources),
        supported_claim="Alex Honnold's Taipei 101 climb occurred on January 24, 2026.",
    )

    # 7) Climb duration: approximately 1h 31m 34s
    await add_stated_and_supported_check(
        evaluator,
        node,
        base_id="climb_duration_1h31m34s",
        overall_desc="Answer states the climb duration was approximately 1 hour, 31 minutes, and 34 seconds AND provides at least one supporting reference URL.",
        stated_claim="The answer states the climb duration was about 1 hour, 31 minutes, and 34 seconds (accept format variations like '1:31:34').",
        sources=_safe_sources(extracted.climb_duration_sources),
        supported_claim="The climb duration was approximately 1 hour, 31 minutes, and 34 seconds.",
        supported_additional_instruction="Allow minor formatting or rounding variations as long as the duration is clearly about 1h 31m 34s.",
    )

    # 8) Broadcast live on Netflix as "Skyscraper Live"
    await add_stated_and_supported_check(
        evaluator,
        node,
        base_id="broadcast_live_on_netflix_skyscraper_live",
        overall_desc="Answer states the climb was broadcast live on Netflix as 'Skyscraper Live' AND provides at least one supporting reference URL.",
        stated_claim="The answer states the climb was broadcast live on Netflix as 'Skyscraper Live'.",
        sources=_safe_sources(extracted.broadcast_sources),
        supported_claim="The climb was broadcast live on Netflix as 'Skyscraper Live'.",
    )

    # 9) Tallest urban free solo in history
    await add_stated_and_supported_check(
        evaluator,
        node,
        base_id="tallest_urban_free_solo_in_history",
        overall_desc="Answer states this was the tallest urban free solo in history AND provides at least one supporting reference URL.",
        stated_claim="The answer states this was the tallest urban free solo in history.",
        sources=_safe_sources(extracted.tallest_urban_sources),
        supported_claim="This was the tallest urban free solo climb in history.",
    )


async def build_system_identification(
    evaluator: Evaluator,
    parent,
    extracted: AnswerExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="protection_system_identification",
        desc="Verify the building’s primary earthquake/wind protection system identification, supported by reference URL(s).",
        parent=parent,
        critical=True,
    )

    # System is a tuned mass damper (TMD)
    await add_stated_and_supported_check(
        evaluator,
        node,
        base_id="system_is_tmd",
        overall_desc="Answer identifies the protection system as a tuned mass damper (TMD) AND provides at least one supporting reference URL.",
        stated_claim="The answer identifies the primary protection system as a tuned mass damper (TMD).",
        sources=_safe_sources(extracted.system_name_sources),
        supported_claim="Taipei 101 uses a tuned mass damper (TMD) as a primary protection system.",
    )

    # System function: earthquake and wind protection
    await add_stated_and_supported_check(
        evaluator,
        node,
        base_id="system_for_quake_and_wind",
        overall_desc="Answer states the tuned mass damper is used for earthquake and wind protection in Taipei 101 AND provides at least one supporting reference URL.",
        stated_claim="The answer states that the TMD is used for earthquake and wind protection.",
        sources=_safe_sources(extracted.system_function_sources),
        supported_claim="The tuned mass damper in Taipei 101 is used for earthquake and wind protection.",
    )


async def build_tmd_specifications(
    evaluator: Evaluator,
    parent,
    extracted: AnswerExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="tmd_specifications",
        desc="Verify the required tuned mass damper specifications (weight, diameter, suspension floor range), each supported by reference URL(s).",
        parent=parent,
        critical=True,
    )

    # TMD Weight: 660 metric tons
    await add_stated_and_supported_check(
        evaluator,
        node,
        base_id="tmd_weight_660_tons",
        overall_desc="Answer states the TMD weighs 660 metric tons (728 short tons) AND provides at least one supporting reference URL.",
        stated_claim="The answer states the tuned mass damper weighs 660 metric tons (accept minor format variants like '660 t').",
        sources=_safe_sources(extracted.tmd_weight_sources),
        supported_claim="Taipei 101's tuned mass damper weighs 660 metric tons (about 728 short tons).",
        supported_additional_instruction="Verify that the source(s) explicitly indicate a TMD weight of 660 metric tons; minor unit/format variations acceptable.",
    )

    # TMD Diameter: 5.5 meters
    await add_stated_and_supported_check(
        evaluator,
        node,
        base_id="tmd_diameter_5_5_meters",
        overall_desc="Answer states the TMD diameter is 5.5 meters AND provides at least one supporting reference URL.",
        stated_claim="The answer states the tuned mass damper has a diameter of 5.5 meters (accept '5.5 m').",
        sources=_safe_sources(extracted.tmd_diameter_sources),
        supported_claim="Taipei 101's tuned mass damper has a diameter of 5.5 meters.",
    )

    # TMD suspended between 92nd and 87th floors
    await add_stated_and_supported_check(
        evaluator,
        node,
        base_id="tmd_suspended_92_to_87",
        overall_desc="Answer states the TMD is suspended between the 92nd and 87th floors (upper and lower floors both specified) AND provides at least one supporting reference URL.",
        stated_claim="The answer states the tuned mass damper is suspended between the 92nd and 87th floors (upper and lower floors specified).",
        sources=_safe_sources(extracted.tmd_floors_sources),
        supported_claim="Taipei 101's tuned mass damper is suspended between the 92nd and 87th floors (upper and lower floors specified).",
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # overall gating: later parts depend on earlier
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_answer_fields(),
        template_class=AnswerExtraction,
        extraction_name="extracted_answer_fields",
    )

    # Record expected ground truths for transparency/debugging
    evaluator.add_ground_truth(
        {
            "expected_values": EXPECTED,
            "notes": "These are the specific values the answer is expected to state; separate URL-based verifications ensure source support.",
        },
        gt_type="expected_specs",
    )

    # Build the top-level critical evaluation node
    answer_eval = evaluator.add_sequential(
        id="answer_evaluation",
        desc="Evaluate whether the answer satisfies all constraints about the climb/building, identifies the protection system, and provides the required TMD specifications with supporting reference URLs.",
        parent=root,
        critical=True,
    )

    # Subsections
    await build_building_and_climb_constraints(evaluator, answer_eval, extracted)
    await build_system_identification(evaluator, answer_eval, extracted)
    await build_tmd_specifications(evaluator, answer_eval, extracted)

    return evaluator.get_summary()