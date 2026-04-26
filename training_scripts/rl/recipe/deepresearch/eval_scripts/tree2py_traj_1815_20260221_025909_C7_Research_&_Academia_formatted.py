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
TASK_ID = "march_2026_factsheet"
TASK_DESCRIPTION = (
    "You are compiling a comprehensive reference factsheet for astronomical observers and space weather researchers "
    "documenting key phenomena occurring during March 2026. Your factsheet must include the following specific data points, "
    "each with supporting reference URLs:\n\n"
    "1. The exact date of the total lunar eclipse occurring in March 2026\n"
    "2. The duration of totality (in minutes) for this lunar eclipse\n"
    "3. At least three major regions or continents from which the total lunar eclipse will be visible\n"
    "4. The UTC time when totality begins for the lunar eclipse\n"
    "5. The UTC time when totality ends for the lunar eclipse\n"
    "6. The year when the next total lunar eclipse after March 2026 will occur\n"
    "7. The traditional name for the March full moon\n"
    "8. The date of the new moon in March 2026\n"
    "9. The total number of full moons occurring in the year 2026\n"
    "10. The discovery date (month, day, and year) of the interstellar comet 3I/ATLAS\n"
    "11. The country where the ATLAS observatory that discovered comet 3I/ATLAS is located\n"
    "12. The sequence number of 3I/ATLAS among confirmed interstellar objects (i.e., whether it is the 1st, 2nd, 3rd, etc., confirmed interstellar object)\n"
    "13. According to NOAA Space Weather Prediction Center scales, the Kp index value that corresponds to a G3 (Strong) geomagnetic storm\n"
    "14. The month and specific date of the closest full supermoon in 2026\n\n"
    "For each data point, provide the factual answer and at least one authoritative reference URL that supports your answer."
)

# Ground truth values used for matching and guidance
GROUND_TRUTH = {
    "lunar_eclipse_date": "March 3, 2026",
    "totality_duration_minutes": "58 minutes",
    "required_visibility_regions": ["Asia", "Australia", "North America"],
    "totality_start_utc": "11:04 UTC",  # Allow 11:04:34 or similar rounding
    "totality_end_utc": "12:02–12:03 UTC",  # Allow 12:02:49 or rounded 12:02 / 12:03
    "next_total_lunar_eclipse_year": "2028",  # Specifically 31 Dec 2028
    "march_full_moon_name": "Worm Moon",
    "march_new_moon_date": "March 18, 2026",  # Some sources may list March 18–19 UTC
    "total_full_moons_2026": "13",
    "comet_3I_ATLAS_discovery_date": "July 1, 2025",
    "comet_3I_ATLAS_discovery_country": "Chile",  # Specifically Río Hurtado
    "comet_3I_ATLAS_sequence_number": "3rd",
    "noaa_g3_kp_index": "7",
    "closest_supermoon_2026_date": "December 23, 2026",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ValueWithSources(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RegionsWithSources(BaseModel):
    regions: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class FactsheetExtraction(BaseModel):
    lunar_eclipse_date: Optional[ValueWithSources] = None
    totality_duration_minutes: Optional[ValueWithSources] = None
    visibility_regions: Optional[RegionsWithSources] = None
    totality_start_utc: Optional[ValueWithSources] = None
    totality_end_utc: Optional[ValueWithSources] = None
    next_total_lunar_eclipse_year: Optional[ValueWithSources] = None
    march_full_moon_name: Optional[ValueWithSources] = None
    march_new_moon_date: Optional[ValueWithSources] = None
    total_full_moons_2026: Optional[ValueWithSources] = None
    comet_3I_ATLAS_discovery_date: Optional[ValueWithSources] = None
    comet_3I_ATLAS_discovery_country: Optional[ValueWithSources] = None
    comet_3I_ATLAS_sequence_number: Optional[ValueWithSources] = None
    noaa_g3_kp_index: Optional[ValueWithSources] = None
    closest_supermoon_2026_date: Optional[ValueWithSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_factsheet() -> str:
    return """
    Extract the specific facts for the March 2026 astronomical events factsheet exactly as stated in the answer, 
    along with the authoritative reference URLs cited for each fact. For each item below, return the value from 
    the answer text and an array of URLs explicitly cited in the answer that support that value. If a value or URLs 
    are missing for an item, set the value to null and/or return an empty array for sources.

    Return a single JSON object with the following fields:
    - lunar_eclipse_date: { value: string|null, sources: string[] } 
      The exact date of the total lunar eclipse in March 2026 (e.g., "March 3, 2026").
    - totality_duration_minutes: { value: string|null, sources: string[] }
      The duration of totality in minutes (e.g., "58 minutes", "≈58 min").
    - visibility_regions: { regions: string[], sources: string[] }
      At least three major regions or continents where totality is visible. Extract the list exactly as presented 
      (e.g., ["Asia","Australia","North America"]) and the supporting URLs.
    - totality_start_utc: { value: string|null, sources: string[] }
      The UTC time when totality begins (e.g., "11:04 UTC", "11:04:34 UTC").
    - totality_end_utc: { value: string|null, sources: string[] }
      The UTC time when totality ends (e.g., "12:02:49 UTC", "12:02 UTC", "12:03 UTC").
    - next_total_lunar_eclipse_year: { value: string|null, sources: string[] }
      The year of the next total lunar eclipse after March 2026 (e.g., "2028").
    - march_full_moon_name: { value: string|null, sources: string[] }
      The traditional name for the March full moon (e.g., "Worm Moon").
    - march_new_moon_date: { value: string|null, sources: string[] }
      The date of the new moon in March 2026 (e.g., "March 18, 2026", or "March 18–19, 2026").
    - total_full_moons_2026: { value: string|null, sources: string[] }
      The total number of full moons in 2026 (e.g., "13").
    - comet_3I_ATLAS_discovery_date: { value: string|null, sources: string[] }
      The discovery date of 3I/ATLAS (e.g., "July 1, 2025").
    - comet_3I_ATLAS_discovery_country: { value: string|null, sources: string[] }
      The country of the ATLAS telescope that discovered 3I/ATLAS (e.g., "Chile").
    - comet_3I_ATLAS_sequence_number: { value: string|null, sources: string[] }
      The sequence number among confirmed interstellar objects (e.g., "3rd", "third").
    - noaa_g3_kp_index: { value: string|null, sources: string[] }
      The Kp index that corresponds to a G3 (Strong) geomagnetic storm per NOAA SWPC (e.g., "7").
    - closest_supermoon_2026_date: { value: string|null, sources: string[] }
      The month and specific date of the closest full supermoon in 2026 (e.g., "December 23, 2026").

    Rules:
    - Only extract URLs explicitly present in the answer text. Do not invent or infer any URLs.
    - Dates, times, and numerical values should be extracted as strings exactly as shown in the answer. 
      If ranges or approximations are used (e.g., "12:02–12:03 UTC", "≈58 minutes"), extract them verbatim.
    - If an item is not mentioned, set its value to null and sources to [].
    """


# --------------------------------------------------------------------------- #
# Helper functions for verification                                           #
# --------------------------------------------------------------------------- #
def _has_value(vws: Optional[ValueWithSources]) -> bool:
    return bool(vws and vws.value and vws.value.strip())


def _has_sources(vws: Optional[ValueWithSources]) -> bool:
    return bool(vws and vws.sources and len(vws.sources) > 0)


def _has_regions(regs: Optional[RegionsWithSources]) -> bool:
    return bool(regs and regs.regions and len(regs.regions) >= 1)


def _has_regions_min3(regs: Optional[RegionsWithSources]) -> bool:
    return bool(regs and regs.regions and len(regs.regions) >= 3)


async def verify_value_item(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    extracted: Optional[ValueWithSources],
    expected_text: str,
    support_claim: str,
    match_add_ins: Optional[str] = None,
    support_add_ins: Optional[str] = None,
) -> None:
    """
    Build a sequential item node with:
      - presence of value (critical)
      - presence of sources (critical)
      - match extracted value to expected (critical, simple verify)
      - verify claim against cited sources (critical, verify by URLs)
    """
    item = evaluator.add_sequential(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=_has_value(extracted),
        id=f"{node_id}_value_present",
        desc=f"{node_id}: Value is present in the answer",
        parent=item,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_sources(extracted),
        id=f"{node_id}_sources_present",
        desc=f"{node_id}: At least one reference URL is provided",
        parent=item,
        critical=True
    )

    # Match extracted value to expected text
    match_node = evaluator.add_leaf(
        id=f"{node_id}_match_expected",
        desc=f"{node_id}: Extracted value matches the expected fact",
        parent=item,
        critical=True
    )
    extracted_value = extracted.value if extracted and extracted.value else ""
    match_claim = (
        f"The provided answer's value '{extracted_value}' is equivalent to the expected fact '{expected_text}'. "
        f"Treat minor formatting differences, capitalization, and reasonable rounding as acceptable."
    )
    await evaluator.verify(
        claim=match_claim,
        node=match_node,
        additional_instruction=match_add_ins or "None"
    )

    # Source-supported claim
    support_node = evaluator.add_leaf(
        id=f"{node_id}_source_support",
        desc=f"{node_id}: Claim is supported by cited reference URL(s)",
        parent=item,
        critical=True
    )
    sources_list = extracted.sources if extracted else []
    await evaluator.verify(
        claim=support_claim,
        node=support_node,
        sources=sources_list,
        additional_instruction=support_add_ins or "None"
    )


async def verify_visibility_regions_item(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    extracted: Optional[RegionsWithSources],
    required_regions: List[str],
) -> None:
    """
    Special handling for visibility regions:
      - presence of at least 3 regions (critical)
      - presence of sources (critical)
      - simple check: list includes Asia, Australia, North America (critical)
      - source-supported claim about those regions (critical)
    """
    item = evaluator.add_sequential(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=_has_regions_min3(extracted),
        id=f"{node_id}_regions_present",
        desc=f"{node_id}: At least three visibility regions are listed in the answer",
        parent=item,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted and extracted.sources and len(extracted.sources) > 0),
        id=f"{node_id}_sources_present",
        desc=f"{node_id}: At least one reference URL is provided",
        parent=item,
        critical=True
    )

    # Regions include required continents
    include_node = evaluator.add_leaf(
        id=f"{node_id}_includes_required",
        desc=f"{node_id}: Listed regions include Asia, Australia, and North America",
        parent=item,
        critical=True
    )
    regions_list = extracted.regions if extracted else []
    regions_text = ", ".join(regions_list)
    include_claim = (
        f"Based on the answer's listed regions [{regions_text}], the visibility list includes Asia, Australia, and North America. "
        f"Treat reasonable synonyms or umbrella terms as acceptable (e.g., 'Oceania' ≈ Australia, 'Americas' includes North America)."
    )
    await evaluator.verify(
        claim=include_claim,
        node=include_node,
        additional_instruction="Allow minor naming variations and synonyms for continents and regions."
    )

    # Source-supported claim
    support_node = evaluator.add_leaf(
        id=f"{node_id}_source_support",
        desc=f"{node_id}: Visibility in Asia, Australia, and North America is supported by cited URL(s)",
        parent=item,
        critical=True
    )
    support_claim = (
        "The total lunar eclipse on March 3, 2026 is visible from Asia, Australia, and North America."
    )
    await evaluator.verify(
        claim=support_claim,
        node=support_node,
        sources=extracted.sources if extracted else [],
        additional_instruction="Focus on confirmation that these three continents have visibility of totality."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate the March 2026 astronomical events factsheet answer and return a structured summary.
    """
    # Initialize evaluator (root is non-critical by default)
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

    # Extract structured facts from the answer
    facts = await evaluator.extract(
        prompt=prompt_extract_factsheet(),
        template_class=FactsheetExtraction,
        extraction_name="factsheet_extraction"
    )

    # Add ground truth info for transparency
    evaluator.add_ground_truth({"expected_facts": GROUND_TRUTH}, gt_type="ground_truth_facts")

    # Create a main parallel node representing the factsheet
    factsheet_node = evaluator.add_parallel(
        id="March_2026_Astronomical_Events_Factsheet",
        desc="Complete factsheet documenting specific astronomical and space weather phenomena for March 2026, with each data point supported by reference URLs",
        parent=root,
        critical=False
    )

    # 1. Lunar Eclipse Date
    await verify_value_item(
        evaluator=evaluator,
        parent_node=factsheet_node,
        node_id="Lunar_Eclipse_Date",
        node_desc="The date of the total lunar eclipse occurring in March 2026 is correctly stated as March 3, 2026, with at least one authoritative reference URL",
        extracted=facts.lunar_eclipse_date,
        expected_text=GROUND_TRUTH["lunar_eclipse_date"],
        support_claim="The total lunar eclipse occurs on March 3, 2026.",
        match_add_ins="Accept date formatting variations (e.g., '3 March 2026').",
        support_add_ins="Verify from credible astronomy sources (e.g., NASA, timeanddate.com)."
    )

    # 2. Totality Duration
    await verify_value_item(
        evaluator=evaluator,
        parent_node=factsheet_node,
        node_id="Totality_Duration",
        node_desc="The duration of totality for the March 3, 2026 lunar eclipse is correctly stated as 58 minutes, with at least one authoritative reference URL",
        extracted=facts.totality_duration_minutes,
        expected_text=GROUND_TRUTH["totality_duration_minutes"],
        support_claim="The totality duration for the March 3, 2026 lunar eclipse is about 58 minutes.",
        match_add_ins="Allow '58 min' or '≈58 minutes' as equivalent.",
        support_add_ins="Minor rounding is acceptable; confirm the stated duration."
    )

    # 3. Visibility Regions
    await verify_visibility_regions_item(
        evaluator=evaluator,
        parent_node=factsheet_node,
        node_id="Eclipse_Visibility_Regions",
        node_desc="At least three major regions where the March 3, 2026 total lunar eclipse is visible are correctly identified (must include Asia, Australia, and North America), with at least one authoritative reference URL",
        extracted=facts.visibility_regions,
        required_regions=GROUND_TRUTH["required_visibility_regions"]
    )

    # 4. Totality Start Time (UTC)
    await verify_value_item(
        evaluator=evaluator,
        parent_node=factsheet_node,
        node_id="Totality_Start_Time_UTC",
        node_desc="The UTC time when totality begins is correctly stated as ~11:04 UTC for the March 3, 2026 lunar eclipse, with at least one authoritative reference URL",
        extracted=facts.totality_start_utc,
        expected_text=GROUND_TRUTH["totality_start_utc"],
        support_claim="Totality for the March 3, 2026 lunar eclipse begins at approximately 11:04 UTC.",
        match_add_ins="Treat '11:04:34 UTC' as equivalent to '11:04 UTC'; ±1 minute tolerance is acceptable.",
        support_add_ins="Minor rounding is acceptable; confirm the start time."
    )

    # 5. Totality End Time (UTC)
    await verify_value_item(
        evaluator=evaluator,
        parent_node=factsheet_node,
        node_id="Totality_End_Time_UTC",
        node_desc="The UTC time when totality ends is correctly stated as ~12:02–12:03 UTC for the March 3, 2026 lunar eclipse, with at least one authoritative reference URL",
        extracted=facts.totality_end_utc,
        expected_text=GROUND_TRUTH["totality_end_utc"],
        support_claim="Totality for the March 3, 2026 lunar eclipse ends at approximately 12:02–12:03 UTC.",
        match_add_ins="Accept '12:02:49 UTC' and rounding to '12:02 UTC' or '12:03 UTC'.",
        support_add_ins="Minor rounding is acceptable; confirm the end time."
    )

    # 6. Next Total Lunar Eclipse Year
    await verify_value_item(
        evaluator=evaluator,
        parent_node=factsheet_node,
        node_id="Next_Total_Lunar_Eclipse_Year",
        node_desc="The year of the next total lunar eclipse after March 3, 2026 is correctly stated as 2028 (Dec 31, 2028), with at least one authoritative reference URL",
        extracted=facts.next_total_lunar_eclipse_year,
        expected_text=GROUND_TRUTH["next_total_lunar_eclipse_year"],
        support_claim="The next total lunar eclipse after March 3, 2026 occurs in 2028, specifically on December 31, 2028.",
        match_add_ins="Value '2028' is correct even if the specific date is included.",
        support_add_ins="Confirm from credible eclipse catalogs or NASA sources."
    )

    # 7. March Full Moon Name
    await verify_value_item(
        evaluator=evaluator,
        parent_node=factsheet_node,
        node_id="March_Full_Moon_Name",
        node_desc="The traditional name for the March full moon is correctly stated as the Worm Moon, with at least one authoritative reference URL",
        extracted=facts.march_full_moon_name,
        expected_text=GROUND_TRUTH["march_full_moon_name"],
        support_claim="The traditional name for the March full moon is 'Worm Moon'.",
        match_add_ins="Allow capitalization differences or 'Full Worm Moon'.",
        support_add_ins="Confirm from reputable astronomical or almanac sources."
    )

    # 8. March 2026 New Moon Date
    await verify_value_item(
        evaluator=evaluator,
        parent_node=factsheet_node,
        node_id="March_2026_New_Moon_Date",
        node_desc="The date of the new moon in March 2026 is correctly stated as March 18, 2026 (some sources may indicate March 18–19), with at least one authoritative reference URL",
        extracted=facts.march_new_moon_date,
        expected_text=GROUND_TRUTH["march_new_moon_date"],
        support_claim="The new moon in March 2026 occurs on March 18, 2026 (some sources may show March 18–19 depending on UTC time).",
        match_add_ins="Treat 'March 18–19, 2026' as acceptable if UTC straddles midnight.",
        support_add_ins="Confirm UTC timing; minor date boundary differences are acceptable."
    )

    # 9. Total Full Moons in 2026
    await verify_value_item(
        evaluator=evaluator,
        parent_node=factsheet_node,
        node_id="Total_Full_Moons_2026",
        node_desc="The total number of full moons in 2026 is correctly stated as 13, with at least one authoritative reference URL",
        extracted=facts.total_full_moons_2026,
        expected_text=GROUND_TRUTH["total_full_moons_2026"],
        support_claim="There are a total of 13 full moons in the year 2026.",
        match_add_ins="Allow '13' or 'thirteen' equivalence.",
        support_add_ins="Confirm using reliable lunar phase calendars."
    )

    # 10. 3I/ATLAS Discovery Date
    await verify_value_item(
        evaluator=evaluator,
        parent_node=factsheet_node,
        node_id="3I_ATLAS_Discovery_Date",
        node_desc="The discovery date of interstellar comet 3I/ATLAS is correctly stated as July 1, 2025, with at least one authoritative reference URL",
        extracted=facts.comet_3I_ATLAS_discovery_date,
        expected_text=GROUND_TRUTH["comet_3I_ATLAS_discovery_date"],
        support_claim="Interstellar comet 3I/ATLAS was discovered on July 1, 2025.",
        match_add_ins="Accept formatting variations like '1 July 2025'.",
        support_add_ins="Prefer authoritative sources (MPEC circulars, official project pages)."
    )

    # 11. 3I/ATLAS Discovery Location (Country)
    await verify_value_item(
        evaluator=evaluator,
        parent_node=factsheet_node,
        node_id="3I_ATLAS_Discovery_Location",
        node_desc="The country where the ATLAS telescope that discovered 3I/ATLAS is located is correctly stated as Chile (specifically at Río Hurtado), with at least one authoritative reference URL",
        extracted=facts.comet_3I_ATLAS_discovery_country,
        expected_text=GROUND_TRUTH["comet_3I_ATLAS_discovery_country"],
        support_claim="The ATLAS observatory that discovered 3I/ATLAS is located in Chile (specifically Río Hurtado).",
        match_add_ins="Allow inclusion of the site 'Río Hurtado' while matching country 'Chile'.",
        support_add_ins="Confirm using ATLAS project pages or reputable observatory references."
    )

    # 12. 3I/ATLAS Sequence Number among interstellar objects
    await verify_value_item(
        evaluator=evaluator,
        parent_node=factsheet_node,
        node_id="3I_ATLAS_Sequence_Number",
        node_desc="3I/ATLAS is correctly identified as the third (3rd) confirmed interstellar object, with at least one authoritative reference URL",
        extracted=facts.comet_3I_ATLAS_sequence_number,
        expected_text=GROUND_TRUTH["comet_3I_ATLAS_sequence_number"],
        support_claim="3I/ATLAS is the third confirmed interstellar object.",
        match_add_ins="Treat '3rd' and 'third' as equivalent.",
        support_add_ins="Cross-check with authoritative listings of interstellar objects (e.g., MPC)."
    )

    # 13. NOAA G3 Kp Index
    await verify_value_item(
        evaluator=evaluator,
        parent_node=factsheet_node,
        node_id="NOAA_G3_Kp_Index",
        node_desc="According to NOAA Space Weather Scales, a G3 (Strong) geomagnetic storm corresponds to Kp = 7, with at least one authoritative reference URL",
        extracted=facts.noaa_g3_kp_index,
        expected_text=GROUND_TRUTH["noaa_g3_kp_index"],
        support_claim="According to NOAA SWPC scales, a G3 (Strong) geomagnetic storm corresponds to Kp = 7.",
        match_add_ins="Value '7' is correct for G3.",
        support_add_ins="Prefer NOAA SWPC pages or documents for verification."
    )

    # 14. Closest Full Supermoon in 2026
    await verify_value_item(
        evaluator=evaluator,
        parent_node=factsheet_node,
        node_id="Closest_Supermoon_2026_Date",
        node_desc="The closest full supermoon of 2026 is correctly stated to occur on December 23, 2026, with at least one authoritative reference URL",
        extracted=facts.closest_supermoon_2026_date,
        expected_text=GROUND_TRUTH["closest_supermoon_2026_date"],
        support_claim="The closest full supermoon in 2026 occurs on December 23, 2026.",
        match_add_ins="Allow variants like 'Dec 23, 2026' or '23 December 2026'.",
        support_add_ins="Confirm with reliable lunar perigee/full moon resources (e.g., timeanddate.com, reputable astronomy sites)."
    )

    # Return final summary
    return evaluator.get_summary()