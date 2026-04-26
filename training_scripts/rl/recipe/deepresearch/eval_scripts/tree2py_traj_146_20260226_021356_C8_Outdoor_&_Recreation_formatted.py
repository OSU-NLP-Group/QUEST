import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "lapland_ski_resort_2025_2026"
TASK_DESCRIPTION = (
    "Identify the ski resort in Finnish Lapland, Finland, that has at least 40 slopes and at least 25 lifts operating during the winter 2025-2026 season. "
    "For this resort, provide the following verified information: (1) Exact location (region and country confirmation), (2) Exact number of slopes, "
    "(3) Exact number of lifts, (4) Total slope length in kilometers (must be at least 35 km), (5) Maximum vertical drop in meters (must be at least 300 meters), "
    "(6) Name of the nearest airport, (7) Distance from the resort to the nearest airport in kilometers (must be within 20 km), (8) Confirmation of winter 2025-2026 operation, "
    "(9) Total lift capacity in passengers per hour (must be at least 25,000 passengers/hour), (10) Availability of beginner-suitable slopes (blue slopes), "
    "(11) Availability of advanced-suitable slopes (black slopes), (12) Availability of cross-country skiing trails. "
    "Additionally, for the nearest airport you identified, determine which of the following four airlines operate direct commercial flights to that airport during the winter 2025-2026 season: "
    "Frontier Airlines, British Airways, Lufthansa, and JetBlue. For each airline, state whether they operate direct flights to the identified airport (Yes/No) and provide reference URLs supporting your answer. "
    "Include reference URLs for all resort specifications and airport information."
)


# -------------------------- Data Models -------------------------- #
class AirlineInfo(BaseModel):
    operates_direct: Optional[str] = None  # "Yes" or "No"
    urls: List[str] = Field(default_factory=list)


class ResortAirlineExtraction(BaseModel):
    resort_name: Optional[str] = None
    resort_reference_urls: List[str] = Field(default_factory=list)

    location_region: Optional[str] = None
    country: Optional[str] = None

    slope_count: Optional[str] = None
    lift_count: Optional[str] = None
    total_slope_km: Optional[str] = None
    vertical_drop_m: Optional[str] = None

    winter_2025_2026_operates: Optional[str] = None

    nearest_airport_name: Optional[str] = None
    airport_iata_code: Optional[str] = None
    nearest_airport_distance_km: Optional[str] = None
    airport_reference_urls: List[str] = Field(default_factory=list)

    lift_capacity_pax_per_hour: Optional[str] = None
    beginner_slopes_available: Optional[str] = None
    advanced_slopes_available: Optional[str] = None
    cross_country_trails_available: Optional[str] = None

    european_connection_urls: List[str] = Field(default_factory=list)

    airline_frontier: Optional[AirlineInfo] = None
    airline_british_airways: Optional[AirlineInfo] = None
    airline_lufthansa: Optional[AirlineInfo] = None
    airline_jetblue: Optional[AirlineInfo] = None


# -------------------------- Extraction Prompt -------------------------- #
def prompt_extract_resort_and_airline_info() -> str:
    return """
Extract the ski resort identification and all requested specifications, plus airline accessibility information, exactly as presented in the answer text. Do not invent or infer anything not explicitly stated.

Return a single JSON with these fields:
- resort_name: string | null
- resort_reference_urls: array of strings (valid URLs) for the resort specs and operations; if none are present, return an empty array
- location_region: string | null (should be something like "Lapland" or "Finnish Lapland")
- country: string | null (should be "Finland" if stated)
- slope_count: string | null (exact count as stated in the answer)
- lift_count: string | null (exact count as stated)
- total_slope_km: string | null (e.g., "43 km" or "43")
- vertical_drop_m: string | null (e.g., "325 m" or "325")
- winter_2025_2026_operates: string | null ("Yes" or "No" as stated)
- nearest_airport_name: string | null (e.g., "Kittilä Airport")
- airport_iata_code: string | null (e.g., "KTT" if given)
- nearest_airport_distance_km: string | null (e.g., "15 km" or "15")
- airport_reference_urls: array of strings (valid URLs) used to support nearest airport identity and distance; empty array if not given
- lift_capacity_pax_per_hour: string | null (e.g., "27,000" or "27000")
- beginner_slopes_available: string | null ("Yes" or "No" as stated; accept "blue slopes available" as "Yes")
- advanced_slopes_available: string | null ("Yes" or "No" as stated; accept "black slopes available" as "Yes")
- cross_country_trails_available: string | null ("Yes" or "No" as stated)
- european_connection_urls: array of strings (valid URLs) that support the claim that the airport has direct commercial flights from at least one major European city during winter 2025-2026; empty array if none are given

- airline_frontier: object | null
  - operates_direct: string | null ("Yes" or "No" as stated for winter 2025-2026)
  - urls: array of strings (valid URLs) that support the stated Yes/No for Frontier; empty array if none given
- airline_british_airways: object | null
  - operates_direct: string | null ("Yes" or "No")
  - urls: array of strings
- airline_lufthansa: object | null
  - operates_direct: string | null ("Yes" or "No")
  - urls: array of strings
- airline_jetblue: object | null
  - operates_direct: string | null ("Yes" or "No")
  - urls: array of strings

Special rules:
- Only extract URLs explicitly written in the answer (plain URLs or markdown links). If the answer mentions a source without a URL, do not invent one; leave the corresponding list empty.
- Preserve numbers as strings if they appear that way (e.g., "43 km", "~15 km"). Do not normalize or compute.
- For yes/no style fields, use exactly "Yes" or "No" when the answer implies it; otherwise return null.
- If the answer lists multiple resorts or airports, extract only the one the answer ultimately chooses as the primary identified resort and nearest airport.
"""


# -------------------------- Helper Utilities -------------------------- #
def _combine_sources(*lists: Optional[List[str]]) -> List[str]:
    urls: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str) and u.strip() and u not in urls:
                urls.append(u.strip())
    return urls


def _yn(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"yes", "y", "true"}:
        return "Yes"
    if v in {"no", "n", "false"}:
        return "No"
    return value


def _airline_info_or_default(info: Optional[AirlineInfo]) -> AirlineInfo:
    return info if info is not None else AirlineInfo(operates_direct=None, urls=[])


# -------------------------- Verification Builders -------------------------- #
async def build_step1_resort_identification(evaluator: Evaluator, root, data: ResortAirlineExtraction):
    node = evaluator.add_leaf(
        id="step1_resort_identification",
        desc="Identify the ski resort in Finnish Lapland that has at least 40 slopes and at least 25 lifts",
        parent=root,
        critical=True,
    )

    resort_name = data.resort_name or "the identified resort"
    claim = (
        f"There exists a ski resort in Finnish Lapland, Finland named '{resort_name}' that has at least 40 slopes and at least 25 lifts."
    )
    sources = data.resort_reference_urls
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction=(
            "Verify that the named resort is in the Lapland region of Finland and that its published statistics meet both thresholds: "
            "≥40 slopes and ≥25 lifts. Minor naming variants are acceptable (e.g., Finnish/English forms)."
        ),
    )


async def build_step2_resort_specifications(evaluator: Evaluator, root, data: ResortAirlineExtraction):
    step2 = evaluator.add_parallel(
        id="step2_resort_specifications",
        desc="Verify all required specifications for the identified ski resort",
        parent=root,
        critical=False,
    )

    resort_name = data.resort_name or "the resort"
    airport_name = data.nearest_airport_name or "the nearest airport"
    resort_sources = data.resort_reference_urls
    airport_sources = _combine_sources(data.airport_reference_urls, resort_sources)

    # spec1_location
    n = evaluator.add_leaf(
        id="spec1_location",
        desc="The resort is confirmed to be located in Finnish Lapland, Finland",
        parent=step2,
        critical=True,
    )
    claim = f"The ski resort '{resort_name}' is located in the Lapland region of Finland (i.e., Finnish Lapland)."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=resort_sources,
        additional_instruction="Accept variants like 'Lapland', 'Finnish Lapland', or 'Lapin maakunta' as the same region.",
    )

    # spec2_slope_count
    n = evaluator.add_leaf(
        id="spec2_slope_count",
        desc="The resort has at least 40 slopes",
        parent=step2,
        critical=True,
    )
    claim = f"The ski resort '{resort_name}' has at least 40 slopes."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=resort_sources,
        additional_instruction=(
            "If the page lists an exact number (e.g., 43 slopes), consider the 'at least 40' requirement satisfied. "
            "Ignore minor textual variations."
        ),
    )

    # spec3_lift_count
    n = evaluator.add_leaf(
        id="spec3_lift_count",
        desc="The resort has at least 25 lifts",
        parent=step2,
        critical=True,
    )
    claim = f"The ski resort '{resort_name}' has at least 25 lifts."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=resort_sources,
        additional_instruction="If the page lists an exact number (e.g., 27 lifts), treat this as meeting the 'at least 25' requirement.",
    )

    # spec4_total_slope_length
    n = evaluator.add_leaf(
        id="spec4_total_slope_length",
        desc="The resort offers at least 35 kilometers of total slope length",
        parent=step2,
        critical=True,
    )
    claim = f"The ski resort '{resort_name}' offers at least 35 kilometers of total slope length."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=resort_sources,
        additional_instruction="If the page lists an exact number ≥35 km (e.g., 38 km), this satisfies the requirement; allow small rounding.",
    )

    # spec5_vertical_drop
    n = evaluator.add_leaf(
        id="spec5_vertical_drop",
        desc="The resort provides at least 300 meters of vertical drop",
        parent=step2,
        critical=True,
    )
    claim = f"The ski resort '{resort_name}' has a maximum vertical drop of at least 300 meters."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=resort_sources,
        additional_instruction="If the page lists an exact figure ≥300 m, consider it satisfying the requirement.",
    )

    # spec6_nearest_airport
    n = evaluator.add_leaf(
        id="spec6_nearest_airport",
        desc="Identify the nearest airport to the resort",
        parent=step2,
        critical=True,
    )
    claim = f"The nearest airport to {resort_name} is {airport_name}."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=airport_sources,
        additional_instruction=(
            "Verify the named nearest airport as stated in the provided sources. "
            "Accept authoritative resort or airport pages that explicitly identify the nearest airport."
        ),
    )

    # spec7_airport_distance
    n = evaluator.add_leaf(
        id="spec7_airport_distance",
        desc="The nearest airport is located within 20 kilometers of the resort",
        parent=step2,
        critical=True,
    )
    dist_txt = data.nearest_airport_distance_km or "the stated distance"
    claim = (
        f"The distance from {resort_name} to {airport_name} is {dist_txt}, which is within 20 km."
    )
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=airport_sources,
        additional_instruction=(
            "Distance may be approximate and given as road or straight-line; accept reasonable rounding if the stated value is ≤20 km."
        ),
    )

    # spec8_winter_operations
    n = evaluator.add_leaf(
        id="spec8_winter_operations",
        desc="The resort operates during winter season 2025-2026",
        parent=step2,
        critical=True,
    )
    claim = f"The ski resort '{resort_name}' operates during the winter 2025–2026 season."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=resort_sources,
        additional_instruction=(
            "Look for season calendars, operating dates or announcements explicitly including 2025–2026 (or 'winter 25/26')."
        ),
    )

    # spec9_lift_capacity
    n = evaluator.add_leaf(
        id="spec9_lift_capacity",
        desc="The resort's total lift capacity is at least 25,000 passengers per hour",
        parent=step2,
        critical=True,
    )
    claim = f"The ski resort '{resort_name}' has a total lift capacity of at least 25,000 passengers per hour."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=resort_sources,
        additional_instruction="Accept if the page states an exact capacity ≥25,000 pph; allow minor formatting differences in numbers.",
    )

    # spec10_beginner_slopes
    n = evaluator.add_leaf(
        id="spec10_beginner_slopes",
        desc="The resort offers slopes suitable for beginners (blue slopes)",
        parent=step2,
        critical=True,
    )
    claim = f"The ski resort '{resort_name}' offers beginner-suitable 'blue' slopes."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=resort_sources,
        additional_instruction="Look for 'blue runs' or equivalent beginner difficulty classification indicating availability.",
    )

    # spec11_advanced_slopes
    n = evaluator.add_leaf(
        id="spec11_advanced_slopes",
        desc="The resort offers slopes suitable for advanced skiers (black slopes)",
        parent=step2,
        critical=True,
    )
    claim = f"The ski resort '{resort_name}' offers advanced-suitable 'black' slopes."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=resort_sources,
        additional_instruction="Look for 'black runs' or equivalent advanced difficulty classification indicating availability.",
    )

    # spec12_cross_country
    n = evaluator.add_leaf(
        id="spec12_cross_country",
        desc="The resort provides access to cross-country skiing trails",
        parent=step2,
        critical=True,
    )
    claim = f"The ski resort '{resort_name}' provides access to cross-country skiing trails."
    await evaluator.verify(
        claim=claim,
        node=n,
        sources=resort_sources,
        additional_instruction="Accept trails listed on the resort site or local official tourism pages linked in the answer.",
    )

    # spec13_resort_reference (existence/validity check)
    has_resort_ref = any(isinstance(u, str) and u.strip().startswith(("http://", "https://")) for u in resort_sources)
    evaluator.add_custom_node(
        result=has_resort_ref,
        id="spec13_resort_reference",
        desc="Provide a valid reference URL for the ski resort's official information",
        parent=step2,
        critical=True,
    )

    # spec14_airport_reference (existence/validity check)
    has_airport_ref = any(isinstance(u, str) and u.strip().startswith(("http://", "https://")) for u in data.airport_reference_urls)
    evaluator.add_custom_node(
        result=has_airport_ref,
        id="spec14_airport_reference",
        desc="Provide a valid reference URL confirming the airport location and distance",
        parent=step2,
        critical=True,
    )


async def build_step3_airline_verification(evaluator: Evaluator, root, data: ResortAirlineExtraction):
    step3 = evaluator.add_parallel(
        id="step3_airline_verification",
        desc="Verify which of the four specified airlines operate direct flights to the nearest airport during winter 2025-2026",
        parent=root,
        critical=False,
    )

    airport_name = data.nearest_airport_name or "the identified airport"
    iata = data.airport_iata_code or ""

    # Airline reference existence check (critical)
    frontier = _airline_info_or_default(data.airline_frontier)
    ba = _airline_info_or_default(data.airline_british_airways)
    lh = _airline_info_or_default(data.airline_lufthansa)
    jb = _airline_info_or_default(data.airline_jetblue)

    all_have_urls = all([
        isinstance(frontier.urls, list) and len(frontier.urls) > 0,
        isinstance(ba.urls, list) and len(ba.urls) > 0,
        isinstance(lh.urls, list) and len(lh.urls) > 0,
        isinstance(jb.urls, list) and len(jb.urls) > 0,
    ])
    evaluator.add_custom_node(
        result=all_have_urls,
        id="airline_reference",
        desc="Provide valid reference URLs confirming the airline service information for the identified airport",
        parent=step3,
        critical=True,
    )

    # European connection (critical)
    european_urls = _combine_sources(
        data.european_connection_urls,
        frontier.urls, ba.urls, lh.urls, jb.urls,
        data.airport_reference_urls
    )
    node = evaluator.add_leaf(
        id="european_connection",
        desc="The identified airport has direct commercial flight connections from at least one major European city during winter 2025-2026",
        parent=step3,
        critical=True,
    )
    claim = (
        f"The airport '{airport_name}' has direct commercial flights from at least one major European city during the winter 2025–2026 season."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=european_urls,
        additional_instruction=(
            "Accept authoritative airline or airport pages, schedules, or route maps that show a direct connection from a European city "
            "(e.g., London, Frankfurt, Munich, Oslo, Stockholm, etc.) during winter 2025–2026."
        ),
    )

    # Helper for each airline verification
    async def _verify_airline(node_id: str, airline_name: str, info: AirlineInfo):
        operates = _yn(info.operates_direct)
        # If operates is Yes/No, craft the corresponding claim
        if operates == "Yes":
            txt = f"{airline_name} operates direct flights to {airport_name} {f'({iata})' if iata else ''} during the winter 2025–2026 season."
        elif operates == "No":
            txt = f"{airline_name} does not operate direct flights to {airport_name} {f'({iata})' if iata else ''} during the winter 2025–2026 season."
        else:
            txt = (
                f"The answer asserts whether {airline_name} operates direct flights to {airport_name} during winter 2025–2026."
            )

        leaf = evaluator.add_leaf(
            id=node_id,
            desc=f"Verify whether {airline_name} operates direct flights to the identified airport during winter 2025-2026",
            parent=step3,
            critical=True,
        )
        await evaluator.verify(
            claim=txt,
            node=leaf,
            sources=info.urls if info.urls else None,
            additional_instruction=(
                "Use only the provided URLs. Acceptable evidence includes airline route maps, schedules, booking engine pages, official announcements, or airport pages. "
                "If the claim is 'No', confirm that the airline's route map/schedule does not list this airport as a served destination for that season, or that an authoritative page indicates no direct service."
            ),
        )

    await _verify_airline("airline1_frontier", "Frontier Airlines", frontier)
    await _verify_airline("airline2_british_airways", "British Airways", ba)
    await _verify_airline("airline3_lufthansa", "Lufthansa", lh)
    await _verify_airline("airline4_jetblue", "JetBlue", jb)


# -------------------------- Main Evaluation Entry -------------------------- #
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

    data = await evaluator.extract(
        prompt=prompt_extract_resort_and_airline_info(),
        template_class=ResortAirlineExtraction,
        extraction_name="resort_airline_extraction",
    )

    await build_step1_resort_identification(evaluator, root, data)
    await build_step2_resort_specifications(evaluator, root, data)
    await build_step3_airline_verification(evaluator, root, data)

    return evaluator.get_summary()