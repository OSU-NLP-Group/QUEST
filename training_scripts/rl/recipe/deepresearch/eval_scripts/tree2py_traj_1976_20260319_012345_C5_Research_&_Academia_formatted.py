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
TASK_ID = "eclipse_2026_03_03"
TASK_DESCRIPTION = """
Provide comprehensive information about the March 3, 2026 total lunar eclipse for educational and reference purposes. Your response must include:

1. Eclipse Timing Details: Provide the duration of totality (in minutes), the start time of totality (in UTC), the end time of totality (in UTC), and the overall duration of all eclipse phases combined (in hours and minutes).

2. Geographic Visibility: Identify which major geographic regions (continents or broad areas) where the eclipse was visible, which major regions where it was NOT visible, and approximately how many people worldwide could see at least some portion of the total phase.

3. Griffith Observatory Broadcast Program: Provide information about Griffith Observatory's live online broadcast of this eclipse, including the observatory's location (city and state), and the start and end times of the broadcast (in PST).

4. Scientific Observation Capabilities: Describe the temperature change observable on the lunar surface during the eclipse (in Kelvin), and identify at least two other types of scientific measurements or observations that were possible during this lunar eclipse.

For each category of information, provide at least one reference URL that supports your answers.
"""

# Ground truth expectations used for simple equality/fuzzy matching checks
EXPECTED = {
    "totality_duration_minutes": "58 minutes",
    "totality_start_utc": "11:04 UTC on March 3, 2026",
    "totality_end_utc": "12:02 UTC on March 3, 2026",
    "overall_eclipse_duration": "5 hours 39 minutes",
    "visible_regions_text": "eastern Asia; Australia; the Pacific; North and Central America; far western South America",
    "non_visible_regions_text": "Africa and Europe",
    "population_reach": "approximately 3.34 billion",
    "observatory_location": "Los Angeles, California",
    "broadcast_start_pst": "12:47 a.m. PST on March 3, 2026",
    "broadcast_end_pst": "6:25 a.m. PST on March 3, 2026",
    "temp_change_range": "147–220 K cooling",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EclipseInfoExtraction(BaseModel):
    # 1) Timing
    totality_duration_minutes: Optional[str] = None
    totality_start_utc: Optional[str] = None
    totality_end_utc: Optional[str] = None
    overall_eclipse_duration: Optional[str] = None
    timing_sources: List[str] = Field(default_factory=list)

    # 2) Geographic visibility
    visible_regions: List[str] = Field(default_factory=list)
    non_visible_regions: List[str] = Field(default_factory=list)
    population_reach: Optional[str] = None
    visibility_sources: List[str] = Field(default_factory=list)

    # 3) Griffith Observatory broadcast
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    broadcast_start_pst: Optional[str] = None
    broadcast_end_pst: Optional[str] = None
    observatory_sources: List[str] = Field(default_factory=list)

    # 4) Scientific observation capabilities
    temp_change_kelvin: Optional[str] = None
    other_measurements: List[str] = Field(default_factory=list)
    science_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_eclipse_info() -> str:
    return """
    Extract the requested information about the March 3, 2026 total lunar eclipse exactly as stated in the provided answer text. Do NOT invent data.

    Return a JSON object with the following fields. If any field is not explicitly stated in the answer, set it to null (for strings) or an empty array (for lists).

    1) Timing details:
       - totality_duration_minutes: string (e.g., "58 minutes")
       - totality_start_utc: string (e.g., "11:04 UTC on March 3, 2026" or "2026-03-03 11:04 UTC")
       - totality_end_utc: string (e.g., "12:02 UTC on March 3, 2026" or "2026-03-03 12:02 UTC")
       - overall_eclipse_duration: string (e.g., "5 hours 39 minutes")
       - timing_sources: array of URLs cited in the answer specifically for timing

    2) Geographic visibility:
       - visible_regions: array of strings listing the broad regions/continents where visible (e.g., ["eastern Asia", "Australia", "Pacific", "North and Central America", "far western South America"])
       - non_visible_regions: array of strings listing broad regions/continents where NOT visible (e.g., ["Africa", "Europe"])
       - population_reach: string for approximate number of people who could see at least some of the total phase (e.g., "approximately 3.34 billion")
       - visibility_sources: array of URLs cited for visibility/population info

    3) Griffith Observatory broadcast:
       - location_city: string (e.g., "Los Angeles")
       - location_state: string (e.g., "California")
       - broadcast_start_pst: string (e.g., "12:47 a.m. PST on March 3, 2026")
       - broadcast_end_pst: string (e.g., "6:25 a.m. PST on March 3, 2026")
       - observatory_sources: array of URLs cited for the broadcast/program info

    4) Scientific observation capabilities:
       - temp_change_kelvin: string describing the temperature change on the lunar surface during the eclipse as stated (e.g., "about 200 K cooling" or "147–220 K")
       - other_measurements: array of at least two other types of scientific observations or measurements that were possible (e.g., ["infrared observations", "radio observations"])
       - science_sources: array of URLs cited for the scientific measurements

    SPECIAL RULES FOR URL FIELDS:
    - Extract only URLs that are explicitly present in the answer.
    - Include full URLs; if a URL is missing protocol, prepend "http://".
    - Do not infer or fabricate URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _fmt_list(items: List[str]) -> str:
    return "; ".join([s.strip() for s in items if isinstance(s, str) and s.strip()]) if items else ""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_timing_checks(evaluator: Evaluator, parent, data: EclipseInfoExtraction) -> None:
    node = evaluator.add_parallel(
        id="Eclipse_Timing_Details",
        desc="Accurate timing information for the total lunar eclipse",
        parent=parent,
        critical=False
    )

    # 1) Totality duration equals 58 minutes
    leaf_duration = evaluator.add_leaf(
        id="Totality_Duration",
        desc="The duration of totality is provided and equals 58 minutes",
        parent=node,
        critical=True
    )
    claim_duration = (
        f"The stated totality duration '{data.totality_duration_minutes}' matches the correct value 58 minutes "
        f"(allowing minor formatting or rounding)."
    )
    await evaluator.verify(
        claim=claim_duration,
        node=leaf_duration,
        additional_instruction="Focus on whether the two durations represent the same amount of time. Treat '58 min' or '0:58' as equivalent to '58 minutes'."
    )

    # 2) Totality start equals 11:04 UTC on March 3, 2026
    leaf_start = evaluator.add_leaf(
        id="Totality_Start_UTC",
        desc="The start time of totality in UTC is provided and equals 11:04 UTC on March 3, 2026",
        parent=node,
        critical=True
    )
    claim_start = (
        f"The stated totality start time '{data.totality_start_utc}' (UTC) matches 11:04 UTC on March 3, 2026 "
        f"(allowing minor formatting variations like '2026-03-03 11:04 UTC')."
    )
    await evaluator.verify(
        claim=claim_start,
        node=leaf_start,
        additional_instruction="Judge equivalence robustly (ignore letter casing, allow presence/absence of seconds)."
    )

    # 3) Totality end equals 12:02 UTC on March 3, 2026
    leaf_end = evaluator.add_leaf(
        id="Totality_End_UTC",
        desc="The end time of totality in UTC is provided and equals 12:02 UTC on March 3, 2026",
        parent=node,
        critical=True
    )
    claim_end = (
        f"The stated totality end time '{data.totality_end_utc}' (UTC) matches 12:02 UTC on March 3, 2026 "
        f"(allowing minor formatting variations)."
    )
    await evaluator.verify(
        claim=claim_end,
        node=leaf_end,
        additional_instruction="Judge equivalence robustly (ignore letter casing, allow presence/absence of seconds)."
    )

    # 4) Overall eclipse duration equals 5 hours 39 minutes
    leaf_overall = evaluator.add_leaf(
        id="Overall_Eclipse_Duration",
        desc="The overall duration (all phases) is provided and equals 5 hours 39 minutes",
        parent=node,
        critical=True
    )
    claim_overall = (
        f"The stated overall eclipse duration '{data.overall_eclipse_duration}' matches 5 hours 39 minutes "
        f"(allowing minor formatting like '5h 39m')."
    )
    await evaluator.verify(
        claim=claim_overall,
        node=leaf_overall,
        additional_instruction="Treat '5h 39m' as equivalent to '5 hours 39 minutes'."
    )

    # 5) Timing reference URL(s) support the provided timing info
    if not data.timing_sources:
        evaluator.add_custom_node(
            result=False,
            id="Timing_Reference_URL",
            desc="A valid reference URL is provided that supports the timing information",
            parent=node,
            critical=True
        )
    else:
        leaf_timing_src = evaluator.add_leaf(
            id="Timing_Reference_URL",
            desc="A valid reference URL is provided that supports the timing information",
            parent=node,
            critical=True
        )
        claim_timing_src = (
            "This page provides timing details for the March 3, 2026 total lunar eclipse, including the totality "
            f"duration '{data.totality_duration_minutes}', the totality window from '{data.totality_start_utc}' "
            f"to '{data.totality_end_utc}' UTC, and the overall eclipse duration '{data.overall_eclipse_duration}'."
        )
        await evaluator.verify(
            claim=claim_timing_src,
            node=leaf_timing_src,
            sources=data.timing_sources,
            additional_instruction="Confirm the page is about the March 3, 2026 total lunar eclipse and explicitly supports the stated durations/times. Minor differences like including seconds are acceptable."
        )


async def build_visibility_checks(evaluator: Evaluator, parent, data: EclipseInfoExtraction) -> None:
    node = evaluator.add_parallel(
        id="Geographic_Visibility",
        desc="Geographic visibility information for the eclipse",
        parent=parent,
        critical=False
    )

    expected_visible = EXPECTED["visible_regions_text"]
    expected_non_visible = EXPECTED["non_visible_regions_text"]

    # 1) Visible regions match expected set (fuzzy)
    leaf_visible = evaluator.add_leaf(
        id="Visible_Regions",
        desc="The visible regions are accurately identified (eastern Asia, Australia, Pacific, North and Central America, far western South America)",
        parent=node,
        critical=True
    )
    claim_visible = (
        f"The answer's listed visible regions '{_fmt_list(data.visible_regions)}' match the expected set: "
        f"{expected_visible}. Allow fuzzy matching and synonymous phrasing."
    )
    await evaluator.verify(
        claim=claim_visible,
        node=leaf_visible,
        additional_instruction="Accept reasonable synonyms and phrasing (e.g., 'the Pacific' vs 'Pacific')."
    )

    # 2) Non-visible regions match expected set (Africa and Europe)
    leaf_non_visible = evaluator.add_leaf(
        id="Non_Visible_Regions",
        desc="The non-visible regions are accurately identified (Africa and Europe)",
        parent=node,
        critical=True
    )
    claim_non_visible = (
        f"The answer's listed non-visible regions '{_fmt_list(data.non_visible_regions)}' match the expected set: "
        f"{expected_non_visible}. Allow minor phrasing variants."
    )
    await evaluator.verify(
        claim=claim_non_visible,
        node=leaf_non_visible,
        additional_instruction="Treat 'Africa & Europe' as equivalent to 'Africa and Europe'."
    )

    # 3) Population reach approximately 3.34 billion
    leaf_population = evaluator.add_leaf(
        id="Population_Reach",
        desc="The approximate number of people who can see at least some of the total phase is provided (approximately 3.34 billion)",
        parent=node,
        critical=True
    )
    claim_population = (
        f"The stated population reach '{data.population_reach}' is approximately equal to 3.34 billion "
        f"(allow modest rounding and formatting differences)."
    )
    await evaluator.verify(
        claim=claim_population,
        node=leaf_population,
        additional_instruction="Accept 'about 3.3 billion' or '≈3.34B' as equivalent."
    )

    # 4) Visibility reference URL(s) support the visibility info
    if not data.visibility_sources:
        evaluator.add_custom_node(
            result=False,
            id="Visibility_Reference_URL",
            desc="A valid reference URL is provided that supports the visibility information",
            parent=node,
            critical=True
        )
    else:
        leaf_vis_src = evaluator.add_leaf(
            id="Visibility_Reference_URL",
            desc="A valid reference URL is provided that supports the visibility information",
            parent=node,
            critical=True
        )
        claim_vis_src = (
            "This page provides visibility details for the March 3, 2026 total lunar eclipse, including broad regions "
            f"where it was visible (e.g., {_fmt_list(data.visible_regions)}) and not visible (e.g., {_fmt_list(data.non_visible_regions)}), "
            f"and/or mentions the approximate population reach '{data.population_reach}'."
        )
        await evaluator.verify(
            claim=claim_vis_src,
            node=leaf_vis_src,
            sources=data.visibility_sources,
            additional_instruction="The page should clearly be about visibility for the March 3, 2026 total lunar eclipse. Fuzzy matches for regional phrasing are acceptable."
        )


async def build_griffith_checks(evaluator: Evaluator, parent, data: EclipseInfoExtraction) -> None:
    node = evaluator.add_parallel(
        id="Griffith_Observatory_Program",
        desc="Information about Griffith Observatory's live broadcast program for the eclipse",
        parent=parent,
        critical=False
    )

    # 1) Observatory location: Los Angeles, California
    leaf_location = evaluator.add_leaf(
        id="Observatory_Location",
        desc="The observatory location is correctly identified as Los Angeles, California",
        parent=node,
        critical=True
    )
    location_text = f"{(data.location_city or '').strip()}, {(data.location_state or '').strip()}".strip(", ").strip()
    claim_location = (
        f"The stated observatory location '{location_text}' matches 'Los Angeles, California' "
        f"(allowing minor formatting)."
    )
    await evaluator.verify(
        claim=claim_location,
        node=leaf_location,
        additional_instruction="Equate 'LA, CA' or 'Los Angeles CA' with 'Los Angeles, California'."
    )

    # 2) Broadcast start time equals 12:47 a.m. PST on March 3, 2026
    leaf_beg = evaluator.add_leaf(
        id="Broadcast_Start_Time",
        desc="The broadcast start time is provided and equals 12:47 a.m. PST on March 3, 2026",
        parent=node,
        critical=True
    )
    claim_beg = (
        f"The stated broadcast start time '{data.broadcast_start_pst}' matches 12:47 a.m. PST on March 3, 2026 "
        f"(allowing minor formatting variants)."
    )
    await evaluator.verify(
        claim=claim_beg,
        node=leaf_beg,
        additional_instruction="Focus on PST local time; formatting variations like '12:47 AM PST' are acceptable."
    )

    # 3) Broadcast end time equals 6:25 a.m. PST on March 3, 2026
    leaf_end = evaluator.add_leaf(
        id="Broadcast_End_Time",
        desc="The broadcast end time is provided and equals 6:25 a.m. PST on March 3, 2026",
        parent=node,
        critical=True
    )
    claim_end = (
        f"The stated broadcast end time '{data.broadcast_end_pst}' matches 6:25 a.m. PST on March 3, 2026 "
        f"(allowing minor formatting variants)."
    )
    await evaluator.verify(
        claim=claim_end,
        node=leaf_end,
        additional_instruction="Focus on PST local time; formatting variations like '6:25 AM PST' are acceptable."
    )

    # 4) Observatory reference URL(s) support the program info
    if not data.observatory_sources:
        evaluator.add_custom_node(
            result=False,
            id="Observatory_Reference_URL",
            desc="A valid reference URL is provided that supports the Griffith Observatory program information",
            parent=node,
            critical=True
        )
    else:
        leaf_obs_src = evaluator.add_leaf(
            id="Observatory_Reference_URL",
            desc="A valid reference URL is provided that supports the Griffith Observatory program information",
            parent=node,
            critical=True
        )
        claim_obs_src = (
            "This page describes Griffith Observatory's live broadcast for the March 3, 2026 total lunar eclipse, "
            f"confirms the observatory location as Los Angeles, California, and lists broadcast times from "
            f"'{data.broadcast_start_pst}' to '{data.broadcast_end_pst}' PST."
        )
        await evaluator.verify(
            claim=claim_obs_src,
            node=leaf_obs_src,
            sources=data.observatory_sources,
            additional_instruction="The page can be on griffithobservatory.org or an official channel hosting the program details."
        )


async def build_science_checks(evaluator: Evaluator, parent, data: EclipseInfoExtraction) -> None:
    node = evaluator.add_parallel(
        id="Scientific_Observation_Capabilities",
        desc="Scientific measurements and observations possible during the lunar eclipse",
        parent=parent,
        critical=False
    )

    # 1) Temperature change: within 147–220 K cooling range
    leaf_temp = evaluator.add_leaf(
        id="Temperature_Change_Measurement",
        desc="The temperature change on the lunar surface during eclipse is provided (147-220 Kelvin cooling)",
        parent=node,
        critical=True
    )
    claim_temp = (
        f"The stated lunar surface temperature change during the eclipse ('{data.temp_change_kelvin}') indicates a "
        f"cooling between 147 K and 220 K (inclusive), allowing approximate phrasing like 'about 200 K'."
    )
    await evaluator.verify(
        claim=claim_temp,
        node=leaf_temp,
        additional_instruction="Accept equivalent range notations such as '147–220 K', '147-220K', or approximate single values around ~200 K explicitly described as cooling."
    )

    # 2) At least two other measurement types identified
    has_two_or_more = isinstance(data.other_measurements, list) and len(
        [m for m in data.other_measurements if isinstance(m, str) and m.strip()]
    ) >= 2
    evaluator.add_custom_node(
        result=has_two_or_more,
        id="Other_Measurement_Types",
        desc="At least two other types of scientific measurements are identified (e.g., infrared observations, radio observations, laser ranging)",
        parent=node,
        critical=True
    )

    # 3) Science reference URL(s) support the scientific info
    if not data.science_sources:
        evaluator.add_custom_node(
            result=False,
            id="Science_Reference_URL",
            desc="A valid reference URL is provided that supports the scientific observation information",
            parent=node,
            critical=True
        )
    else:
        leaf_sci_src = evaluator.add_leaf(
            id="Science_Reference_URL",
            desc="A valid reference URL is provided that supports the scientific observation information",
            parent=node,
            critical=True
        )
        claim_sci_src = (
            "This page discusses scientific observations possible during the March 3, 2026 total lunar eclipse, "
            f"including a lunar surface temperature change around '{data.temp_change_kelvin}' (within ~147–220 K cooling), "
            f"and mentions at least two observation types such as {_fmt_list(data.other_measurements)}."
        )
        await evaluator.verify(
            claim=claim_sci_src,
            node=leaf_sci_src,
            sources=data.science_sources,
            additional_instruction="Look for explicit statements about thermal drop magnitude and example observation modalities (e.g., IR, radio, photometry, spectroscopy, laser ranging)."
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
    Evaluate an answer for the March 3, 2026 total lunar eclipse information task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Categories are independent
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
        prompt=prompt_extract_eclipse_info(),
        template_class=EclipseInfoExtraction,
        extraction_name="eclipse_info_extraction",
    )

    # Add ground truth info for transparency
    evaluator.add_ground_truth({
        "expected": EXPECTED
    })

    # Build verification subtrees
    await build_timing_checks(evaluator, root, extracted)
    await build_visibility_checks(evaluator, root, extracted)
    await build_griffith_checks(evaluator, root, extracted)
    await build_science_checks(evaluator, root, extracted)

    # Return standard evaluation summary
    return evaluator.get_summary()