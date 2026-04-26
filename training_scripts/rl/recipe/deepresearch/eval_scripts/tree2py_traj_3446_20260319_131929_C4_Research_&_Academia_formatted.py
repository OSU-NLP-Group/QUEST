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
TASK_ID = "optimal_eclipse_city_2026_03_03"
TASK_DESCRIPTION = (
    "For the total lunar eclipse occurring on March 3, 2026, identify a major city that offers optimal viewing conditions. "
    "The city must meet the following criteria: (1) the entire totality phase (lasting 58-59 minutes) must be visible with "
    "the moon above the horizon throughout, (2) it must be located in one of the primary visibility regions (Western North America, "
    "Australia, New Zealand, East Asia, or Pacific), (3) it must be in an area with statistically favorable weather conditions "
    "for clear skies (specifically northwest Mexico, southwest United States, or inland Australia), (4) it must NOT be in the "
    "Eastern time zone of North America where the moon sets during totality, and (5) the viewing time should occur during reasonable "
    "observation hours (between 6:00 PM and 8:00 AM local time). Provide the city name, the local time range when totality occurs "
    "at that location, and verify that this timing corresponds correctly to the global totality window of 11:04-12:02 UTC."
)
GLOBAL_TOTALITY_UTC_WINDOW = "11:04–12:02 UTC on 2026-03-03"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EclipseCityExtraction(BaseModel):
    """
    Extracted information for the first proposed city mentioned in the answer for viewing the
    March 3, 2026 total lunar eclipse.
    """
    city: Optional[str] = None
    state_or_province: Optional[str] = None
    country_or_region: Optional[str] = None

    # Optional high-level region category if explicitly stated in the answer
    # (e.g., 'Western North America', 'Australia', 'New Zealand', 'East Asia', 'Pacific')
    region_category: Optional[str] = None

    # Local totality timing (as stated in the answer; keep strings as-is)
    local_totality_start: Optional[str] = None
    local_totality_end: Optional[str] = None
    local_totality_range: Optional[str] = None  # e.g., "3:04–4:02 am local" or "03:04–04:02 CST"
    timezone: Optional[str] = None  # e.g., "MST", "PST", "ACST", etc., if stated

    # URLs explicitly provided in the answer
    sources: List[str] = Field(default_factory=list)           # all URLs mentioned
    eclipse_sources: List[str] = Field(default_factory=list)   # URLs about the 2026-03-03 eclipse for this city
    city_info_sources: List[str] = Field(default_factory=list) # URLs about the city profile/timezone/population
    weather_sources: List[str] = Field(default_factory=list)   # URLs about climate/clear-sky statistics/region suitability


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_city_candidate() -> str:
    return """
    Extract information for the first proposed city that the answer recommends for optimally viewing
    the March 3, 2026 total lunar eclipse. Return:
    - city: City name (string)
    - state_or_province: State/Province/Region if provided (string or null)
    - country_or_region: Country or broader region if provided (string or null)
    - region_category: If the answer explicitly categorizes the city as located in ANY of:
        ["Western North America", "Australia", "New Zealand", "East Asia", "Pacific"], extract that exact label;
        otherwise return null. Do NOT invent this tag if not present.
    - local_totality_start: The local time when totality begins at the city (string as written, or null if absent)
    - local_totality_end: The local time when totality ends at the city (string as written, or null if absent)
    - local_totality_range: A single string representing the local totality time range as written in the answer
        (e.g., "3:04–4:02 am local", "03:04–04:02 CST"); if not given as a single range, return null.
    - timezone: The local timezone abbreviation/name if explicitly stated (string or null)
    - sources: An array of ALL URLs mentioned anywhere in the answer
    - eclipse_sources: An array of URLs that specifically discuss the March 3, 2026 lunar eclipse timing/visibility
        for this city (e.g., timeanddate.com/eclipse in that city, NASA/EclipseWise city page). Only include if explicitly cited.
    - city_info_sources: An array of URLs about the city's profile (e.g., Wikipedia city page or official city site)
        and/or timezone info. Only include if explicitly cited.
    - weather_sources: An array of URLs specifically about clear-sky/climate suitability for the location or the broader
        subregion (northwest Mexico, southwest U.S., or inland Australia). Only include if explicitly cited.

    IMPORTANT:
    - Extract only what is explicitly present in the provided answer text. Do not invent or infer new URLs or fields.
    - Preserve the original formatting of times, including AM/PM or 24-hour format and timezone abbreviations if present.
    - If a requested field is not present, set it to null (or empty list for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _make_local_range_text(ex: EclipseCityExtraction) -> Optional[str]:
    """Create a human-readable local totality range string using the extracted fields."""
    if ex.local_totality_range and ex.local_totality_range.strip():
        return ex.local_totality_range.strip()
    if (ex.local_totality_start and ex.local_totality_start.strip()) and (ex.local_totality_end and ex.local_totality_end.strip()):
        return f"{ex.local_totality_start.strip()} – {ex.local_totality_end.strip()}"
    return None


def _pick_urls(preferred: List[str], fallback: List[str]) -> List[str]:
    """Return preferred if non-empty, else fallback (may still be empty)."""
    return preferred if preferred else fallback


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_for_city(
    evaluator: Evaluator,
    parent_node,
    ex: EclipseCityExtraction
) -> None:
    """
    Build the verification tree for the proposed city and perform verifications according to the rubric.
    We structure the root as non-critical and split into:
      - A critical parallel group "must_criteria" for all mandatory checks
      - A non-critical parallel group "nice_to_have" for ReasonableViewingHours
    """
    must_node = evaluator.add_parallel(
        id="must_criteria",
        desc="All mandatory criteria for the optimal eclipse viewing city",
        parent=parent_node,
        critical=True
    )
    nice_node = evaluator.add_parallel(
        id="nice_to_have",
        desc="Non-critical but relevant viewing hour criterion",
        parent=parent_node,
        critical=False
    )

    # 1) CityNameProvided (existence)
    city_exists = bool(ex.city and ex.city.strip())
    evaluator.add_custom_node(
        result=city_exists,
        id="CityNameProvided",
        desc="The candidate must provide a specific city name as requested in the question",
        parent=must_node,
        critical=True
    )

    # 2) MajorCityStatus (verify via city info sources if possible)
    major_city_leaf = evaluator.add_leaf(
        id="MajorCityStatus",
        desc="The location must be a recognized major city with significant population and infrastructure for astronomical observation",
        parent=must_node,
        critical=True
    )
    city_sources = _pick_urls(ex.city_info_sources, ex.sources)
    city_name_for_claim = ex.city or "the proposed city"
    await evaluator.verify(
        claim=f"{city_name_for_claim} is widely recognized as a major city with significant population and urban infrastructure.",
        node=major_city_leaf,
        sources=city_sources,
        additional_instruction="Use the cited city-profile sources (e.g., Wikipedia/official pages) to determine if this is commonly regarded as a major city (large population or significant metro/infrastructure)."
    )

    # 3) CompleteVisibility (entire totality visible above the horizon)
    complete_visibility_leaf = evaluator.add_leaf(
        id="CompleteVisibility",
        desc="The entire totality phase (58-59 minutes) must be visible while the moon is above the horizon at the proposed city",
        parent=must_node,
        critical=True
    )
    eclipse_sources = _pick_urls(ex.eclipse_sources, ex.sources)
    await evaluator.verify(
        claim=(
            f"At {city_name_for_claim}, the entire totality phase of the March 3, 2026 total lunar eclipse is visible above the horizon "
            f"from start to end (approximately 58–59 minutes)."
        ),
        node=complete_visibility_leaf,
        sources=eclipse_sources,
        additional_instruction="Look for city-specific eclipse pages (e.g., timeanddate, NASA/EclipseWise) indicating that totality is fully visible and the Moon remains above the horizon throughout totality."
    )

    # 4) GeographicRegion (allowed primary visibility regions)
    geographic_region_leaf = evaluator.add_leaf(
        id="GeographicRegion",
        desc="The city must be located in one of the regions where totality is visible: Western North America, Australia, New Zealand, East Asia, or Pacific",
        parent=must_node,
        critical=True
    )
    if ex.region_category and ex.region_category.strip():
        region_phrase = ex.region_category.strip()
        region_claim = (
            f"{city_name_for_claim} is located in the region '{region_phrase}', "
            f"which is one of the allowed primary visibility regions for the 2026-03-03 total lunar eclipse "
            f"(Western North America, Australia, New Zealand, East Asia, or Pacific)."
        )
    else:
        region_claim = (
            f"{city_name_for_claim} is located in one of the allowed primary visibility regions for the 2026-03-03 total lunar eclipse "
            f"(Western North America, Australia, New Zealand, East Asia, or Pacific)."
        )
    await evaluator.verify(
        claim=region_claim,
        node=geographic_region_leaf,
        sources=_pick_urls(eclipse_sources, city_sources),
        additional_instruction="Use the cited sources to judge whether the city's geographic location fits any of the listed allowed regions."
    )

    # 5) WeatherConditions (favorable subregions)
    weather_leaf = evaluator.add_leaf(
        id="WeatherConditions",
        desc="The city must be in a region with statistically favorable weather conditions (northwest Mexico, southwest U.S., or inland Australia)",
        parent=must_node,
        critical=True
    )
    weather_sources = _pick_urls(ex.weather_sources, _pick_urls(ex.city_info_sources, ex.sources))
    await evaluator.verify(
        claim=(
            f"{city_name_for_claim} lies within one of the favorable clear-sky subregions for early March: "
            f"northwest Mexico, the southwestern United States, or inland Australia."
        ),
        node=weather_leaf,
        sources=weather_sources,
        additional_instruction="This is a location-based check. Consider the city's geographic placement relative to these subregions; explicit climate stats are helpful but not strictly required if the location clearly falls within one of the named subregions."
    )

    # 6) ExcludeEasternTimeZone (not in North American ET)
    not_et_leaf = evaluator.add_leaf(
        id="ExcludeEasternTimeZone",
        desc="The city must not be in the Eastern time zone of North America where the moon sets during totality",
        parent=must_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{city_name_for_claim} is not located in North America's Eastern Time Zone (ET/EST/EDT).",
        node=not_et_leaf,
        sources=city_sources,
        additional_instruction="Use the cited city/region/timezone sources to confirm the city is not in the Eastern Time Zone of North America."
    )

    # 7) LocalTimeProvided (existence)
    local_range_text = _make_local_range_text(ex)
    local_time_provided = bool(local_range_text and local_range_text.strip())
    evaluator.add_custom_node(
        result=local_time_provided,
        id="LocalTimeProvided",
        desc="The candidate must provide the local time range when totality occurs at the proposed location",
        parent=must_node,
        critical=True
    )

    # 8) TimingWithinTotality (local timing matches global 11:04–12:02 UTC window)
    timing_leaf = evaluator.add_leaf(
        id="TimingWithinTotality",
        desc="The provided timing information must correctly fall within the totality window of 11:04-12:02 UTC on March 3, 2026",
        parent=must_node,
        critical=True
    )
    # Provide a safe default if local_range_text is missing (the node will get skipped if LocalTimeProvided failed)
    claimed_local_range = local_range_text or "(no local range provided)"
    await evaluator.verify(
        claim=(
            f"The local totality timing in {city_name_for_claim} ({claimed_local_range}) correctly corresponds to the global totality window of "
            f"{GLOBAL_TOTALITY_UTC_WINDOW} (allowing up to a 1–2 minute tolerance)."
        ),
        node=timing_leaf,
        sources=eclipse_sources,
        additional_instruction=(
            "Cross-check the city's local totality times against the known global totality window (11:04–12:02 UTC on 2026-03-03). "
            "Account for local timezone and daylight-saving (if applicable) and allow a small rounding tolerance."
        )
    )

    # 9) ReasonableViewingHours (non-critical)
    viewing_hours_leaf = evaluator.add_leaf(
        id="ReasonableViewingHours",
        desc="The totality viewing time should occur during hours when the general public can reasonably observe (between 6:00 PM and 8:00 AM local time)",
        parent=nice_node,
        critical=False
    )
    claimed_range_for_hours = local_range_text or "(no local range provided)"
    await evaluator.verify(
        claim=(
            f"The local totality time range {claimed_range_for_hours} occurs between 6:00 PM and 8:00 AM local time "
            f"(considering that crossing midnight is still within this window)."
        ),
        node=viewing_hours_leaf,
        sources=None,  # This is a pure logical/time-window check based on the provided text
        additional_instruction=(
            "Interpret the provided local time range. If both endpoints (or the entire interval) fall within 18:00–08:00 local time, "
            "or the range crosses midnight but remains within that window, then this criterion is satisfied."
        )
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
    Evaluate an answer for the optimal city to view the March 3, 2026 total lunar eclipse.
    Returns a structured summary with the verification tree and score.
    """
    # Initialize evaluator (root is non-critical to allow a non-critical child; criticality handled in subgroups)
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

    # Add ground truth reference (UTC totality window)
    evaluator.add_ground_truth({
        "global_totality_window_utc": GLOBAL_TOTALITY_UTC_WINDOW
    })

    # Extract the proposed city and related info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_city_candidate(),
        template_class=EclipseCityExtraction,
        extraction_name="eclipse_city_candidate"
    )

    # Build verification nodes and run checks
    await build_verification_for_city(evaluator, root, extraction)

    # Return evaluation summary
    return evaluator.get_summary()