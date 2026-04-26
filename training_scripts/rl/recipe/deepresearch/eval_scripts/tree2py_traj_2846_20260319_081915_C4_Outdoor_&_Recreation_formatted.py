import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "largest_indoor_waterpark_resort_us"
TASK_DESCRIPTION = (
    "You are planning a family vacation and want to visit the largest indoor waterpark resort in the United States.\n\n"
    "Find this resort and provide the following information:\n\n"
    "1. The name and location (city and state) of the resort\n"
    "2. The total square footage of the indoor waterpark\n"
    "3. The temperature maintained inside the waterpark\n"
    "4. The type of surf/wave simulation attraction available (if any)\n"
    "5. The number of swim-up bars available\n"
    "6. Whether the resort offers cabanas or bungalows for private rental\n"
    "7. Whether the resort offers day passes for non-overnight guests\n"
    "8. An official website URL reference that supports your findings"
)

OFFICIAL_SOURCE_INSTRUCTION = (
    "Use only official resort/company pages as valid evidence (e.g., the resort's own website or its operator's official pages). "
    "Do NOT accept third‑party aggregator, news, blog, wiki, or review sites as sufficient evidence. "
    "Reject the claim if no provided URL is an official page or if the URL content does not explicitly support the statement. "
    "Allow minor wording variations and focus on whether the official page clearly supports the claim."
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class WaterparkInfo(BaseModel):
    # Core fields (mostly strings to allow flexible formats)
    resort_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    square_footage: Optional[str] = None
    maintained_temperature: Optional[str] = None
    surf_wave_attraction: Optional[str] = None  # e.g., "FlowRider Double", "Wave pool", "None", "No"
    swim_up_bars_count: Optional[str] = None    # keep as string to allow "1", "one", etc.
    cabanas_or_bungalows: Optional[str] = None  # yes/no or description
    day_passes_availability: Optional[str] = None  # yes/no or description

    # Source URLs (attribute-specific + global)
    sources_overall: List[str] = Field(default_factory=list)
    sources_name_loc: List[str] = Field(default_factory=list)
    sources_sqft: List[str] = Field(default_factory=list)
    sources_largest: List[str] = Field(default_factory=list)
    sources_temperature: List[str] = Field(default_factory=list)
    sources_surf_wave: List[str] = Field(default_factory=list)
    sources_swim_up_bars: List[str] = Field(default_factory=list)
    sources_cabanas: List[str] = Field(default_factory=list)
    sources_day_passes: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_waterpark_info() -> str:
    return """
    Extract the requested details about the single resort the answer claims is the largest indoor waterpark resort in the United States.
    Return a single JSON object with the following fields (use null for any missing field):

    - resort_name: The resort's name, exactly as written in the answer.
    - city: The city of the resort.
    - state: The U.S. state of the resort (use the state name or standard 2-letter abbreviation exactly as shown in the answer).
    - square_footage: The total indoor waterpark square footage (keep any commas, decimals, units or qualifiers exactly as written, e.g., "223,000 sq ft").
    - maintained_temperature: The maintained indoor temperature of the waterpark (e.g., "84°F", "82–84 °F", "29°C").
    - surf_wave_attraction: The type/name of any surf or wave simulation attraction if mentioned (e.g., "FlowRider", "FlowRider Double", "wave pool"). If the answer explicitly states none, put "None" or "No".
    - swim_up_bars_count: The number of swim-up bars in the INDOOR waterpark, exactly as written (e.g., "1", "two").
    - cabanas_or_bungalows: Whether the resort offers private cabanas/bungalows for rental; if provided, capture the answer's exact wording (e.g., "Yes—private cabanas available", "No").
    - day_passes_availability: Whether day passes are offered to non-overnight guests; capture the answer's exact wording (e.g., "Yes (limited; buy online)", "No—resort guests only").

    Also extract URLs mentioned in the answer. Only include actual URLs that appear in the answer (plain links or markdown links):
    - sources_overall: URLs cited as overall/primary references.
    - sources_name_loc: URLs supporting the resort name and US location (city, state).
    - sources_sqft: URLs supporting the total indoor waterpark square footage.
    - sources_largest: URLs supporting that it is the largest indoor waterpark RESORT in the United States (by indoor waterpark square footage).
    - sources_temperature: URLs supporting the maintained indoor temperature.
    - sources_surf_wave: URLs supporting the surf/wave simulation information (presence or absence).
    - sources_swim_up_bars: URLs supporting the count of swim-up bars.
    - sources_cabanas: URLs supporting cabana/bungalow private rentals.
    - sources_day_passes: URLs supporting day-pass availability.

    Rules for URL extraction:
    - Extract only URLs explicitly shown in the answer text. Do not invent or infer URLs.
    - Include full URLs with protocol. If a URL lacks a protocol, prepend "http://".
    - If the answer provides a general "Sources" section for all claims, put those links into sources_overall.
    - Deduplicate URLs within each list and remove obviously malformed links.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls or []:
        if not u:
            continue
        u_stripped = u.strip()
        if not u_stripped:
            continue
        if u_stripped not in seen:
            seen.add(u_stripped)
            out.append(u_stripped)
    return out


def _pick_sources(info: WaterparkInfo, specific: List[str]) -> List[str]:
    specific_clean = _dedup_urls(specific or [])
    if specific_clean:
        return specific_clean
    return _dedup_urls(info.sources_overall or [])


def _has_any_source(info: WaterparkInfo, specific: List[str]) -> bool:
    return len(_pick_sources(info, specific)) > 0


def _normalize_text(s: Optional[str]) -> str:
    return (s or "").strip()


def _to_bool_from_text(s: Optional[str]) -> Optional[bool]:
    if s is None:
        return None
    t = s.strip().lower()
    if not t:
        return None
    negatives = ["no", "not", "does not", "doesn't", "no day", "no day pass", "resort guests only", "overnight guests only", "hotel guests only", "must stay overnight", "no cabana", "no bungalow"]
    positives = ["yes", "offer", "offers", "available", "day pass", "day-pass", "cabanas available", "rentals available", "book a cabana"]
    if any(neg in t for neg in negatives):
        return False
    if any(pos in t for pos in positives):
        return True
    return None


# --------------------------------------------------------------------------- #
# Verification node builders                                                  #
# --------------------------------------------------------------------------- #
async def add_name_location_checks(evaluator: Evaluator, parent, info: WaterparkInfo) -> None:
    group = evaluator.add_sequential(
        id="Resort_Name_and_US_Location_with_Official_Citation",
        desc="Provides the resort name and location (city and state) and the location is in the United States, supported by an official URL.",
        parent=parent,
        critical=True,
    )
    has_values = all([
        _normalize_text(info.resort_name),
        _normalize_text(info.city),
        _normalize_text(info.state),
    ])
    has_sources = _has_any_source(info, info.sources_name_loc)

    evaluator.add_custom_node(
        result=has_values and has_sources,
        id="name_loc_presence",
        desc="Name and US location are provided with at least one official URL source",
        parent=group,
        critical=True
    )

    verify_node = evaluator.add_leaf(
        id="name_loc_supported",
        desc="Resort name and US location are supported by official source(s)",
        parent=group,
        critical=True
    )

    claim = f"The resort named '{_normalize_text(info.resort_name)}' is located in {_normalize_text(info.city)}, {_normalize_text(info.state)}, United States."
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=_pick_sources(info, info.sources_name_loc),
        additional_instruction=OFFICIAL_SOURCE_INSTRUCTION + " Verify that the official page clearly shows the resort name and the city/state within the United States."
    )


async def add_square_footage_checks(evaluator: Evaluator, parent, info: WaterparkInfo) -> None:
    group = evaluator.add_sequential(
        id="Indoor_Waterpark_Square_Footage_with_Official_Citation",
        desc="Provides the total indoor waterpark square footage, supported by an official URL.",
        parent=parent,
        critical=True
    )
    has_value = bool(_normalize_text(info.square_footage))
    has_sources = _has_any_source(info, info.sources_sqft)

    evaluator.add_custom_node(
        result=has_value and has_sources,
        id="sqft_presence",
        desc="Indoor waterpark square footage value is provided with at least one official URL source",
        parent=group,
        critical=True
    )

    verify_node = evaluator.add_leaf(
        id="sqft_supported",
        desc="Indoor waterpark square footage is supported by official source(s)",
        parent=group,
        critical=True
    )

    claim = f"The total indoor waterpark square footage is '{_normalize_text(info.square_footage)}'."
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=_pick_sources(info, info.sources_sqft),
        additional_instruction=OFFICIAL_SOURCE_INSTRUCTION + " Ensure that the cited figure specifically refers to the INDOOR waterpark area (not the entire resort, hotel, or outdoor spaces). Allow minor rounding/formatting differences."
    )


async def add_largest_by_sqft_checks(evaluator: Evaluator, parent, info: WaterparkInfo) -> None:
    group = evaluator.add_sequential(
        id="Largest_By_Square_Footage_Justification",
        desc="Justifies that the chosen resort is the largest in the U.S. by indoor waterpark square footage, using official-source evidence.",
        parent=parent,
        critical=True
    )
    # For largest claim, prefer dedicated 'largest' sources, else fall back to overall
    candidate_sources = info.sources_largest if info.sources_largest else info.sources_overall
    has_sources = _has_any_source(info, candidate_sources)

    evaluator.add_custom_node(
        result=bool(_normalize_text(info.resort_name)) and has_sources,
        id="largest_sources_present",
        desc="At least one official URL is provided to justify 'largest in the U.S. by indoor waterpark square footage'",
        parent=group,
        critical=True
    )

    verify_node = evaluator.add_leaf(
        id="largest_supported",
        desc="Largest-by-square-footage claim is supported by official source(s)",
        parent=group,
        critical=True
    )

    claim = f"The resort '{_normalize_text(info.resort_name)}' is the largest indoor waterpark resort in the United States by indoor waterpark square footage."
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=_pick_sources(info, candidate_sources),
        additional_instruction=OFFICIAL_SOURCE_INSTRUCTION + " The page should explicitly claim 'largest' (e.g., 'America's Largest Indoor Waterpark') OR otherwise provide clear official evidence to support the superlative. If unclear or only third‑party pages are provided, mark as not supported."
    )


async def add_temperature_checks(evaluator: Evaluator, parent, info: WaterparkInfo) -> None:
    group = evaluator.add_sequential(
        id="Maintained_Indoor_Temperature_with_Official_Citation",
        desc="Provides the maintained indoor waterpark temperature, supported by an official URL.",
        parent=parent,
        critical=True
    )
    has_value = bool(_normalize_text(info.maintained_temperature))
    has_sources = _has_any_source(info, info.sources_temperature)

    evaluator.add_custom_node(
        result=has_value and has_sources,
        id="temp_presence",
        desc="Maintained indoor temperature is provided with at least one official URL source",
        parent=group,
        critical=True
    )

    verify_node = evaluator.add_leaf(
        id="temp_supported",
        desc="Maintained indoor temperature is supported by official source(s)",
        parent=group,
        critical=True
    )

    claim = f"The indoor waterpark is maintained at around '{_normalize_text(info.maintained_temperature)}'."
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=_pick_sources(info, info.sources_temperature),
        additional_instruction=OFFICIAL_SOURCE_INSTRUCTION + " Verify the temperature value or range (e.g., 84°F). Allow small wording variations such as 'kept at' or 'maintained at'."
    )


async def add_surf_wave_checks(evaluator: Evaluator, parent, info: WaterparkInfo) -> None:
    group = evaluator.add_sequential(
        id="Surf_Wave_Simulation_Info_with_Official_Citation",
        desc="States whether a surf/wave simulation attraction is available; if available, provides the type/name; supported by an official URL.",
        parent=parent,
        critical=True
    )
    has_value = bool(_normalize_text(info.surf_wave_attraction))
    has_sources = _has_any_source(info, info.sources_surf_wave)

    evaluator.add_custom_node(
        result=has_value and has_sources,
        id="surf_presence",
        desc="Surf/wave simulation info is provided with at least one official URL source",
        parent=group,
        critical=True
    )

    verify_node = evaluator.add_leaf(
        id="surf_supported",
        desc="Surf/wave simulation info is supported by official source(s)",
        parent=group,
        critical=True
    )

    val = _normalize_text(info.surf_wave_attraction)
    if val.lower() in ("none", "no", "n/a"):
        claim = "The resort's INDOOR waterpark does NOT have a surf or wave simulation attraction (such as FlowRider or similar)."
    else:
        claim = f"The resort's INDOOR waterpark offers a surf/wave simulation attraction: '{val}'."

    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=_pick_sources(info, info.sources_surf_wave),
        additional_instruction=OFFICIAL_SOURCE_INSTRUCTION + " Check the official attractions page(s). If verifying absence, ensure the official content indicates no such surf simulator; otherwise, list presence and the attraction name/type."
    )


async def add_swimup_bars_checks(evaluator: Evaluator, parent, info: WaterparkInfo) -> None:
    group = evaluator.add_sequential(
        id="Swim_Up_Bars_Count_with_Official_Citation",
        desc="Provides the number of swim-up bars available, supported by an official URL.",
        parent=parent,
        critical=True
    )
    has_value = bool(_normalize_text(info.swim_up_bars_count))
    has_sources = _has_any_source(info, info.sources_swim_up_bars)

    evaluator.add_custom_node(
        result=has_value and has_sources,
        id="swimup_presence",
        desc="Swim-up bars count is provided with at least one official URL source",
        parent=group,
        critical=True
    )

    verify_node = evaluator.add_leaf(
        id="swimup_supported",
        desc="Swim-up bars count is supported by official source(s)",
        parent=group,
        critical=True
    )

    claim = f"There are '{_normalize_text(info.swim_up_bars_count)}' swim-up bars in the INDOOR waterpark."
    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=_pick_sources(info, info.sources_swim_up_bars),
        additional_instruction=OFFICIAL_SOURCE_INSTRUCTION + " Confirm the count of swim-up bars specifically within the INDOOR waterpark."
    )


async def add_cabanas_checks(evaluator: Evaluator, parent, info: WaterparkInfo) -> None:
    group = evaluator.add_sequential(
        id="Cabana_or_Bungalow_Private_Rental_with_Official_Citation",
        desc="States whether cabanas or bungalows are offered for private rental, supported by an official URL.",
        parent=parent,
        critical=True
    )
    has_value = bool(_normalize_text(info.cabanas_or_bungalows))
    has_sources = _has_any_source(info, info.sources_cabanas)

    evaluator.add_custom_node(
        result=has_value and has_sources,
        id="cabanas_presence",
        desc="Cabana/bungalow private rental info is provided with at least one official URL source",
        parent=group,
        critical=True
    )

    verify_node = evaluator.add_leaf(
        id="cabanas_supported",
        desc="Cabana/bungalow private rental info is supported by official source(s)",
        parent=group,
        critical=True
    )

    yn = _to_bool_from_text(info.cabanas_or_bungalows)
    if yn is False:
        claim = "The resort does NOT offer private cabanas or bungalows for rental at the INDOOR waterpark."
    else:
        claim = "The resort offers private cabanas or bungalows for rental at the INDOOR waterpark."

    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=_pick_sources(info, info.sources_cabanas),
        additional_instruction=OFFICIAL_SOURCE_INSTRUCTION + " Look for an official rentals page indicating private cabanas/bungalows for the INDOOR waterpark."
    )


async def add_daypasses_checks(evaluator: Evaluator, parent, info: WaterparkInfo) -> None:
    group = evaluator.add_sequential(
        id="Day_Pass_Availability_with_Official_Citation",
        desc="States whether day passes for non-overnight guests are offered, supported by an official URL.",
        parent=parent,
        critical=True
    )
    has_value = bool(_normalize_text(info.day_passes_availability))
    has_sources = _has_any_source(info, info.sources_day_passes)

    evaluator.add_custom_node(
        result=has_value and has_sources,
        id="daypass_presence",
        desc="Day-pass availability info is provided with at least one official URL source",
        parent=group,
        critical=True
    )

    verify_node = evaluator.add_leaf(
        id="daypass_supported",
        desc="Day-pass availability info is supported by official source(s)",
        parent=group,
        critical=True
    )

    yn = _to_bool_from_text(info.day_passes_availability)
    if yn is False:
        claim = "The resort does NOT offer day passes for non-overnight guests to the INDOOR waterpark."
    else:
        claim = "The resort offers day passes for non-overnight guests to the INDOOR waterpark (even if limited or on select dates)."

    await evaluator.verify(
        claim=claim,
        node=verify_node,
        sources=_pick_sources(info, info.sources_day_passes),
        additional_instruction=OFFICIAL_SOURCE_INSTRUCTION + " Verify explicitly whether non‑overnight guests can purchase day passes. Accept notes about limited availability or select dates as 'offers'."
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

    # Extract structured info
    extracted: WaterparkInfo = await evaluator.extract(
        prompt=prompt_extract_waterpark_info(),
        template_class=WaterparkInfo,
        extraction_name="waterpark_info"
    )

    # Build rubric root (critical)
    rubric_root = evaluator.add_parallel(
        id="Complete_Indoor_Waterpark_Information",
        desc="Correctly identifies the largest indoor waterpark resort in the United States (by indoor waterpark square footage) and provides all requested attributes, each supported by at least one official URL citation.",
        parent=root,
        critical=True
    )

    # Add all verification subtrees (each subtree is critical per rubric)
    await add_name_location_checks(evaluator, rubric_root, extracted)
    await add_square_footage_checks(evaluator, rubric_root, extracted)
    await add_largest_by_sqft_checks(evaluator, rubric_root, extracted)
    await add_temperature_checks(evaluator, rubric_root, extracted)
    await add_surf_wave_checks(evaluator, rubric_root, extracted)
    await add_swimup_bars_checks(evaluator, rubric_root, extracted)
    await add_cabanas_checks(evaluator, rubric_root, extracted)
    await add_daypasses_checks(evaluator, rubric_root, extracted)

    # Return standard summary
    return evaluator.get_summary()