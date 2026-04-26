import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "outdoor_recreation_trip_2026"
TASK_DESCRIPTION = (
    "You are based in San Diego, California, and planning an outdoor recreation trip for summer 2026. "
    "Your itinerary includes: (1) Visiting both Acadia National Park and Great Smoky Mountains National Park using "
    "Breeze Airways nonstop flights from San Diego, (2) Viewing the total solar eclipse in Spain in August 2026, and "
    "(3) Hiking in the Canary Islands. For this trip, provide the following information: National Park Access via "
    "Breeze Airways; Solar Eclipse Viewing in Mallorca, Spain; Hiking in the Canary Islands."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class RouteInfo(BaseModel):
    destination_city: Optional[str] = None
    destination_airport_code: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ClosestAirportInfo(BaseModel):
    name: Optional[str] = None
    airport_code: Optional[str] = None
    approx_distance_miles: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BreezeAccessInfo(BaseModel):
    great_smoky_mountains_nonstop: Optional[RouteInfo] = None
    acadia_nonstop_route_1: Optional[RouteInfo] = None
    acadia_nonstop_route_2: Optional[RouteInfo] = None
    acadia_closest_airport: Optional[ClosestAirportInfo] = None


class EclipseMallorcaInfo(BaseModel):
    eclipse_date: Optional[str] = None
    eclipse_date_sources: List[str] = Field(default_factory=list)

    totality_crosses_mallorca_statement: Optional[str] = None
    crosses_sources: List[str] = Field(default_factory=list)

    palma_totality_start_time_local: Optional[str] = None
    palma_totality_start_sources: List[str] = Field(default_factory=list)

    palma_totality_duration: Optional[str] = None
    palma_totality_duration_sources: List[str] = Field(default_factory=list)

    palma_partial_eclipse_start_time_local: Optional[str] = None
    palma_partial_start_sources: List[str] = Field(default_factory=list)

    eclipse_glasses_safety_standard: Optional[str] = None
    eclipse_glasses_safety_sources: List[str] = Field(default_factory=list)

    alternative_location_name: Optional[str] = None
    alternative_location_totality_duration: Optional[str] = None
    alternative_location_sources: List[str] = Field(default_factory=list)


class HikingCanaryInfo(BaseModel):
    island: Optional[str] = None
    island_sources: List[str] = Field(default_factory=list)

    approx_trail_count: Optional[str] = None
    trail_count_sources: List[str] = Field(default_factory=list)

    notable_trail_name: Optional[str] = None
    notable_trail_sources: List[str] = Field(default_factory=list)


class TripPlanExtraction(BaseModel):
    breeze: Optional[BreezeAccessInfo] = None
    eclipse: Optional[EclipseMallorcaInfo] = None
    hiking: Optional[HikingCanaryInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_plan() -> str:
    return """
Extract the following structured information from the provided answer text exactly as it appears, without inventing details. Only extract URLs explicitly present in the answer.

1) National Park Access via Breeze Airways (from San Diego - airport code SAN):
   a) Great Smoky Mountains NP access:
      - great_smoky_mountains_nonstop:
        • destination_city (e.g., "Knoxville, TN")
        • destination_airport_code (e.g., "TYS")
        • sources: all URLs cited in the answer that support the Breeze Airways nonstop SAN→destination route
   b) Acadia NP access:
      - acadia_nonstop_route_1:
        • destination_city (e.g., "Portland, ME")
        • destination_airport_code (e.g., "PWM")
        • sources: URLs supporting Breeze nonstop SAN→destination
      - acadia_nonstop_route_2:
        • destination_city
        • destination_airport_code
        • sources: URLs supporting Breeze nonstop SAN→destination
        Note: If more than two candidate routes are listed, select the first two distinct destination airports as they appear in the answer.
      - acadia_closest_airport:
        • name (closest airport to Acadia National Park)
        • airport_code
        • approx_distance_miles (string; e.g., "10", "12–15", "about 50")
        • sources: URLs supporting that this is the closest airport and the approximate distance

2) Solar Eclipse Viewing in Mallorca (August 2026):
   - eclipse_date (prefer "YYYY-MM-DD" or a clear date string, e.g., "2026-08-12" or "August 12, 2026")
   - eclipse_date_sources: URLs supporting the date
   - totality_crosses_mallorca_statement (e.g., "yes" or "confirmed" if answer claims totality crosses Mallorca; otherwise "no" or the exact text provided)
   - crosses_sources: URLs supporting whether totality crosses Mallorca
   - palma_totality_start_time_local: local start time of totality in Palma de Mallorca (string)
   - palma_totality_start_sources: URLs supporting Palma totality start time
   - palma_totality_duration: duration of totality in Palma (string like "1m 42s" or "1 minute 42 seconds")
   - palma_totality_duration_sources: URLs supporting Palma totality duration
   - palma_partial_eclipse_start_time_local: local start time of partial eclipse in Palma (string)
   - palma_partial_start_sources: URLs supporting Palma partial start time
   - eclipse_glasses_safety_standard (e.g., "ISO 12312-2")
   - eclipse_glasses_safety_sources: URLs supporting the safety standard requirement
   - alternative_location_name: a Mallorca location other than Palma mentioned for viewing totality
   - alternative_location_totality_duration: duration of totality at that alternative location (string)
   - alternative_location_sources: URLs supporting the alternative location and its duration

3) Hiking in the Canary Islands:
   - island: the Canary Island identified (e.g., "Tenerife", "Gran Canaria", "La Gomera", "La Palma", "Lanzarote")
   - island_sources: URLs supporting that the island has extensive hiking networks
   - approx_trail_count: approximate number of hiking trails on that island (string like "100", "200+", "over 1,000")
   - trail_count_sources: URLs supporting the approximate trail count
   - notable_trail_name: one notable hiking trail on that island
   - notable_trail_sources: URLs supporting that the trail exists on that island

Rules:
- Only use URLs explicitly present in the answer. If no URL is given for a field, return an empty list for its sources.
- Maintain strings for times, durations, and distances exactly as provided (do not convert or reformat).
- If multiple items are present, choose the first ones that fit the requirements in the order they appear.
- If some required field is not mentioned in the answer, set it to null and the corresponding sources to an empty array.

Return a single JSON object matching this Pydantic schema:
TripPlanExtraction (with nested BreezeAccessInfo, EclipseMallorcaInfo, HikingCanaryInfo).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def safe(s: Optional[str]) -> str:
    return s or ""


def sources_or_none(urls: Optional[List[str]]) -> Optional[List[str]]:
    if not urls:
        return None
    return urls


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_breeze_subtree(evaluator: Evaluator, parent, breeze: Optional[BreezeAccessInfo]) -> None:
    # Parent node for Breeze routes (critical, parallel)
    breeze_node = evaluator.add_parallel(
        id="breeze_airways_national_park_access",
        desc="Provide Breeze Airways nonstop route information from San Diego that supports access to Great Smoky Mountains NP and Acadia NP, plus closest-airport info for Acadia.",
        parent=parent,
        critical=True,
    )

    # Great Smoky Mountains route
    gsm = breeze.great_smoky_mountains_nonstop if breeze else None
    gsm_leaf = evaluator.add_leaf(
        id="great_smoky_mountains_nonstop_route_from_san",
        desc="Identifies a Breeze Airways nonstop route from San Diego that can serve as an access point to Great Smoky Mountains National Park, and specifies the destination city and airport code.",
        parent=breeze_node,
        critical=True,
    )
    gsm_claim = (
        f"Breeze Airways operates a nonstop flight from San Diego (SAN) to "
        f"{safe(gsm.destination_city)} ({safe(gsm.destination_airport_code)})."
        if gsm else
        "The provided answer identifies a Breeze Airways nonstop route from San Diego (SAN) to access Great Smoky Mountains National Park, including destination city and airport code."
    )
    await evaluator.verify(
        claim=gsm_claim,
        node=gsm_leaf,
        sources=sources_or_none(gsm.sources if gsm else []),
        additional_instruction=(
            "Verify that the URL(s) explicitly support a Breeze Airways nonstop route originating from San Diego (SAN) "
            "to the specified destination city and airport code. If nonstop is not supported or airline/route mismatches, mark as unsupported. "
            "If information is missing or ambiguous, judge as unsupported."
        ),
    )

    # Acadia route 1
    ar1 = breeze.acadia_nonstop_route_1 if breeze else None
    ar1_leaf = evaluator.add_leaf(
        id="acadia_nonstop_route_1_from_san",
        desc="Identifies one Breeze Airways nonstop route from San Diego that can serve as an access point to Acadia National Park, and specifies the destination city and airport code.",
        parent=breeze_node,
        critical=True,
    )
    ar1_claim = (
        f"Breeze Airways operates a nonstop flight from San Diego (SAN) to "
        f"{safe(ar1.destination_city)} ({safe(ar1.destination_airport_code)})."
        if ar1 else
        "The provided answer identifies one Breeze Airways nonstop SAN route that can access Acadia National Park, including destination city and airport code."
    )
    await evaluator.verify(
        claim=ar1_claim,
        node=ar1_leaf,
        sources=sources_or_none(ar1.sources if ar1 else []),
        additional_instruction=(
            "Check that the page(s) confirm Breeze Airways nonstop service from SAN to the stated destination with the stated airport code. "
            "If nonstop is not confirmed or airline is different, return unsupported."
        ),
    )

    # Acadia route 2
    ar2 = breeze.acadia_nonstop_route_2 if breeze else None
    ar2_leaf = evaluator.add_leaf(
        id="acadia_nonstop_route_2_from_san",
        desc="Identifies a second (distinct) Breeze Airways nonstop route from San Diego that can serve as an access point to Acadia National Park, and specifies the destination city and airport code.",
        parent=breeze_node,
        critical=True,
    )
    ar2_claim = (
        f"Breeze Airways operates a nonstop flight from San Diego (SAN) to "
        f"{safe(ar2.destination_city)} ({safe(ar2.destination_airport_code)})."
        if ar2 else
        "The provided answer identifies a second Breeze Airways nonstop SAN route for accessing Acadia National Park, including destination city and airport code."
    )
    await evaluator.verify(
        claim=ar2_claim,
        node=ar2_leaf,
        sources=sources_or_none(ar2.sources if ar2 else []),
        additional_instruction=(
            "Verify that this is a Breeze Airways nonstop SAN route to the specified destination/airport code. "
            "Do not assume distinctness from the first route based on the URL alone; focus on the correctness of this route claim against the provided page(s)."
        ),
    )

    # Acadia closest airport + distance
    aca_closest = breeze.acadia_closest_airport if breeze else None
    aca_leaf = evaluator.add_leaf(
        id="acadia_closest_airport_code_and_distance",
        desc="Specifies the closest airport to Acadia National Park, including its airport code and an approximate distance from the park in miles.",
        parent=breeze_node,
        critical=True,
    )
    aca_claim = (
        f"The closest airport to Acadia National Park is {safe(aca_closest.name)} "
        f"({safe(aca_closest.airport_code)}), approximately {safe(aca_closest.approx_distance_miles)} miles from the park."
        if aca_closest else
        "The answer specifies the closest airport to Acadia National Park with its airport code and an approximate distance in miles."
    )
    await evaluator.verify(
        claim=aca_claim,
        node=aca_leaf,
        sources=sources_or_none(aca_closest.sources if aca_closest else []),
        additional_instruction=(
            "Confirm that the cited source(s) explicitly state that this is the closest airport to Acadia National Park "
            "and indicate a distance matching the approximate value given (allow minor rounding differences). "
            "If multiple airports are discussed, ensure the page indicates this one is the closest."
        ),
    )


async def build_eclipse_subtree(evaluator: Evaluator, parent, eclipse: Optional[EclipseMallorcaInfo]) -> None:
    # Parent node for eclipse info (critical, parallel)
    eclipse_node = evaluator.add_parallel(
        id="solar_eclipse_viewing_mallorca",
        desc="Provide the required August 2026 total solar eclipse details for Mallorca (including Palma-specific timing/safety and one alternative Mallorca location with duration).",
        parent=parent,
        critical=True,
    )

    # Date of eclipse
    date_leaf = evaluator.add_leaf(
        id="eclipse_date",
        desc="Correctly states the date of the total solar eclipse in August 2026.",
        parent=eclipse_node,
        critical=True,
    )
    date_claim = (
        f"The total solar eclipse date in August 2026 is {safe(eclipse.eclipse_date)}."
        if eclipse else
        "The answer states the correct date of the total solar eclipse in August 2026."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=sources_or_none(eclipse.eclipse_date_sources if eclipse else []),
        additional_instruction="Verify the exact eclipse date using the cited source(s). Accept common date formats. If unsupported, return False.",
    )

    # Totality crosses Mallorca
    crosses_leaf = evaluator.add_leaf(
        id="totality_crosses_mallorca",
        desc="Correctly confirms that the eclipse path of totality crosses Mallorca.",
        parent=eclipse_node,
        critical=True,
    )
    crosses_claim = "The path of totality crosses Mallorca (Majorca) island in Spain."
    await evaluator.verify(
        claim=crosses_claim,
        node=crosses_leaf,
        sources=sources_or_none(eclipse.crosses_sources if eclipse else []),
        additional_instruction="Confirm from the cited map or authoritative source that Mallorca lies within the path of totality.",
    )

    # Palma totality start time (local)
    palma_start_leaf = evaluator.add_leaf(
        id="palma_totality_start_time_local",
        desc="Provides the correct local start time of totality for Palma de Mallorca.",
        parent=eclipse_node,
        critical=True,
    )
    palma_start_claim = (
        f"In Palma de Mallorca, totality starts at approximately {safe(eclipse.palma_totality_start_time_local)} local time."
        if eclipse else
        "The answer provides the correct local start time of totality for Palma de Mallorca."
    )
    await evaluator.verify(
        claim=palma_start_claim,
        node=palma_start_leaf,
        sources=sources_or_none(eclipse.palma_totality_start_sources if eclipse else []),
        additional_instruction="Times should be local to Palma de Mallorca in August 2026 (CEST, UTC+2). Allow minor rounding or formatting differences.",
    )

    # Palma totality duration
    palma_dur_leaf = evaluator.add_leaf(
        id="palma_totality_duration",
        desc="Provides the correct duration of totality for Palma de Mallorca (in minutes and seconds).",
        parent=eclipse_node,
        critical=True,
    )
    palma_dur_claim = (
        f"In Palma de Mallorca, the duration of totality is approximately {safe(eclipse.palma_totality_duration)}."
        if eclipse else
        "The answer provides the correct totality duration for Palma de Mallorca."
    )
    await evaluator.verify(
        claim=palma_dur_claim,
        node=palma_dur_leaf,
        sources=sources_or_none(eclipse.palma_totality_duration_sources if eclipse else []),
        additional_instruction="Allow minor rounding differences (e.g., 1m 41s vs 1m 42s).",
    )

    # Palma partial eclipse start time (local)
    palma_partial_leaf = evaluator.add_leaf(
        id="palma_partial_eclipse_start_time_local",
        desc="Provides the correct local start time of the partial eclipse for Palma de Mallorca.",
        parent=eclipse_node,
        critical=True,
    )
    palma_partial_claim = (
        f"In Palma de Mallorca, the partial eclipse begins at approximately {safe(eclipse.palma_partial_eclipse_start_time_local)} local time."
        if eclipse else
        "The answer provides the correct local start time of the partial eclipse for Palma de Mallorca."
    )
    await evaluator.verify(
        claim=palma_partial_claim,
        node=palma_partial_leaf,
        sources=sources_or_none(eclipse.palma_partial_start_sources if eclipse else []),
        additional_instruction="Times should be local to Palma de Mallorca (CEST) and supported by the cited source(s).",
    )

    # Eclipse glasses safety standard
    glasses_leaf = evaluator.add_leaf(
        id="eclipse_glasses_safety_standard",
        desc="States the correct international safety standard for eclipse glasses required during partial phases.",
        parent=eclipse_node,
        critical=True,
    )
    glasses_claim = (
        f"The required international safety standard for eclipse glasses during partial phases is {safe(eclipse.eclipse_glasses_safety_standard)}."
        if eclipse else
        "The answer states the correct international safety standard for eclipse glasses used during partial phases."
    )
    await evaluator.verify(
        claim=glasses_claim,
        node=glasses_leaf,
        sources=sources_or_none(eclipse.eclipse_glasses_safety_sources if eclipse else []),
        additional_instruction="Commonly referenced standard is ISO 12312-2; verify the exact wording per the cited source(s).",
    )

    # Alternative Mallorca location with totality duration
    alt_leaf = evaluator.add_leaf(
        id="alternative_mallorca_location_with_totality_duration",
        desc="Identifies one alternative eclipse viewing location in Mallorca besides Palma and includes the correct totality duration for that location.",
        parent=eclipse_node,
        critical=True,
    )
    alt_claim = (
        f"In {safe(eclipse.alternative_location_name)}, Mallorca, totality lasts approximately {safe(eclipse.alternative_location_totality_duration)}."
        if eclipse else
        "The answer identifies an alternate Mallorca location besides Palma and provides its totality duration."
    )
    await evaluator.verify(
        claim=alt_claim,
        node=alt_leaf,
        sources=sources_or_none(eclipse.alternative_location_sources if eclipse else []),
        additional_instruction="Verify that the cited source supports the totality duration for the specified alternative Mallorca location.",
    )


async def build_hiking_subtree(evaluator: Evaluator, parent, hiking: Optional[HikingCanaryInfo]) -> None:
    # Parent node for hiking (critical, sequential)
    hiking_node = evaluator.add_sequential(
        id="canary_islands_hiking",
        desc="Provide the Canary Islands hiking destination plus trail-count and one notable trail name.",
        parent=parent,
        critical=True,
    )

    # Island destination
    island_leaf = evaluator.add_leaf(
        id="canary_island_destination",
        desc="Identifies a valid Canary Island destination with extensive hiking trail networks.",
        parent=hiking_node,
        critical=True,
    )
    island_claim = (
        f"{safe(hiking.island)} is a Canary Island destination known for extensive hiking trail networks."
        if hiking else
        "The answer identifies a valid Canary Island destination with extensive hiking trail networks."
    )
    await evaluator.verify(
        claim=island_claim,
        node=island_leaf,
        sources=sources_or_none(hiking.island_sources if hiking else []),
        additional_instruction="Confirm that the cited page(s) support the island being well-known for hiking (trail networks, routes, or similar).",
    )

    # Approximate trail count
    trail_count_leaf = evaluator.add_leaf(
        id="approx_trail_count",
        desc="Provides an approximate count of hiking trails for the identified island (and the count corresponds to that same island).",
        parent=hiking_node,
        critical=True,
    )
    trail_count_claim = (
        f"{safe(hiking.island)} has approximately {safe(hiking.approx_trail_count)} hiking trails."
        if hiking else
        "The answer provides an approximate trail count for the identified Canary Island."
    )
    await evaluator.verify(
        claim=trail_count_claim,
        node=trail_count_leaf,
        sources=sources_or_none(hiking.trail_count_sources if hiking else []),
        additional_instruction="Accept approximate counts and ranges (e.g., 100+, ~200). Ensure the count pertains to the same island.",
    )

    # Notable trail on island
    notable_leaf = evaluator.add_leaf(
        id="notable_trail_on_island",
        desc="Names one notable hiking trail located on the identified island.",
        parent=hiking_node,
        critical=True,
    )
    notable_claim = (
        f"{safe(hiking.notable_trail_name)} is a notable hiking trail on {safe(hiking.island)}."
        if hiking else
        "The answer names one notable hiking trail located on the identified island."
    )
    await evaluator.verify(
        claim=notable_claim,
        node=notable_leaf,
        sources=sources_or_none(hiking.notable_trail_sources if hiking else []),
        additional_instruction="Confirm the trail exists and is located on the same island identified.",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point for evaluating the answer for the 2026 outdoor recreation trip planning task.
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
        default_model=model,
    )

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_trip_plan(),
        template_class=TripPlanExtraction,
        extraction_name="trip_plan_extraction",
    )

    # Build the verification tree per rubric
    main_node = evaluator.add_parallel(
        id="outdoor_recreation_trip_planning",
        desc="Verify all required information for the trip: Breeze nonstop access to the two national parks, Mallorca eclipse details, and Canary Islands hiking info.",
        parent=root,
        critical=True,
    )

    # Breeze routes subtree
    await build_breeze_subtree(evaluator, main_node, extraction.breeze if extraction else None)

    # Eclipse subtree
    await build_eclipse_subtree(evaluator, main_node, extraction.eclipse if extraction else None)

    # Hiking subtree
    await build_hiking_subtree(evaluator, main_node, extraction.hiking if extraction else None)

    # Return the evaluation summary
    return evaluator.get_summary()