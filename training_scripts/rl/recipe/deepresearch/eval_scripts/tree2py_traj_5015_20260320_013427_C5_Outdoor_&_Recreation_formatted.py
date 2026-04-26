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
TASK_ID = "winter_ski_resort_selection"
TASK_DESCRIPTION = """
You are planning an international winter sports trip for the 2026-2027 ski season and need to identify two ski resorts that meet specific accessibility and performance criteria.

Requirements:

1. North American Resort: Identify one ski resort in North America that:
   - Is accessible from an airport served by direct British Airways flights from London Heathrow
   - Has a minimum vertical drop of at least 400 meters (approximately 1,312 feet)
   - Has its ski season opening scheduled for November or earlier in the 2025-2026 or 2026-2027 season

2. Japanese Resort: Identify one ski resort in Japan that:
   - Is accessible from the airport served by the new Air Canada direct flight from Vancouver starting December 2026
   - Has a minimum vertical drop of at least 400 meters
   - Opens in November or earlier for the 2026-2027 winter season

For each resort, provide:
- Resort name
- Base elevation
- Summit elevation
- Vertical drop
- Scheduled opening date for the relevant season
- Airport access information (which airport and airline route)

Additional requirement: Include a note about British Airways' ski equipment baggage policy dimensions for the North American resort.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AirportAccess(BaseModel):
    airport_name: Optional[str] = None            # e.g., Denver International Airport
    airport_iata: Optional[str] = None            # e.g., DEN
    airline: Optional[str] = None                 # e.g., British Airways / Air Canada
    route_description: Optional[str] = None       # e.g., "BA non-stop LHR–DEN", "Air Canada YVR–CTS"
    verification_urls: List[str] = Field(default_factory=list)  # Airline route/schedule/destination page URLs
    ground_access_urls: List[str] = Field(default_factory=list) # Resort "Getting here"/transport page URLs showing access from the airport


class ResortStats(BaseModel):
    base_elevation: Optional[str] = None          # Prefer units included (e.g., "2,760 m" or "9,055 ft")
    summit_elevation: Optional[str] = None        # Prefer units included
    vertical_drop_m: Optional[str] = None         # Vertical drop in meters as written in the answer (string okay)
    stats_urls: List[str] = Field(default_factory=list)  # URLs that list base/summit/vertical stats (official or trusted sources)


class OpeningInfo(BaseModel):
    season: Optional[str] = None                  # e.g., "2025–2026" or "2026–2027"
    scheduled_opening_date: Optional[str] = None  # e.g., "November 10, 2026" or "Early November 2026"
    opening_urls: List[str] = Field(default_factory=list)  # URLs that mention scheduled opening date


class BAGearPolicy(BaseModel):
    note: Optional[str] = None                    # e.g., "BA ski equipment up to 190 x 75 x 65 cm"
    policy_urls: List[str] = Field(default_factory=list)   # URLs to BA official policy pages


class ResortItem(BaseModel):
    name: Optional[str] = None
    country_or_region: Optional[str] = None
    stats: ResortStats = ResortStats()
    opening: OpeningInfo = OpeningInfo()
    airport_access: AirportAccess = AirportAccess()
    resort_urls: List[str] = Field(default_factory=list)   # General resort page URLs
    baggage_policy: Optional[BAGearPolicy] = None          # Only used for North American resort


class SkiPlanExtraction(BaseModel):
    north_america: ResortItem = ResortItem()
    japan: ResortItem = ResortItem()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ski_plan() -> str:
    return """
    Extract exactly two resorts from the answer:
    1) A North American ski resort that is reachable from an airport with direct British Airways flights from London Heathrow (LHR).
    2) A Japanese ski resort that is reachable from Sapporo New Chitose (CTS), which is served by the new Air Canada direct flight from Vancouver (YVR) starting December 2026.

    For each resort, populate the following JSON fields. Extract values exactly as they appear in the answer text. For all URL fields, extract only fully-qualified URLs explicitly present in the answer (including URLs in markdown links). Do not invent or infer any URLs.

    For "north_america":
    - name
    - country_or_region
    - stats:
        - base_elevation
        - summit_elevation
        - vertical_drop_m
        - stats_urls[]           # pages showing mountain stats (official or trusted)
    - opening:
        - season                 # e.g., "2025–2026" or "2026–2027"
        - scheduled_opening_date # e.g., "November 10, 2026" or "Early November 2026"
        - opening_urls[]         # pages mentioning the scheduled opening
    - airport_access:
        - airport_name
        - airport_iata
        - airline                # should indicate British Airways
        - route_description      # e.g., "BA non-stop LHR–DEN"
        - verification_urls[]    # airline route/schedule pages proving direct BA LHR -> airport
        - ground_access_urls[]   # resort transit pages showing access from the airport
    - resort_urls[]              # general resort info pages
    - baggage_policy:
        - note                   # Include the note from the answer about BA ski equipment policy (dimensions target: 190 x 75 x 65 cm)
        - policy_urls[]          # BA official policy URL(s)

    For "japan":
    - name
    - country_or_region
    - stats:
        - base_elevation
        - summit_elevation
        - vertical_drop_m
        - stats_urls[]
    - opening:
        - season
        - scheduled_opening_date
        - opening_urls[]
    - airport_access:
        - airport_name           # Ideally "New Chitose Airport" / "Sapporo (CTS)"
        - airport_iata           # e.g., "CTS"
        - airline                # should indicate Air Canada
        - route_description      # e.g., "Air Canada non-stop YVR–CTS (from Dec 2026)"
        - verification_urls[]    # airline page(s) proving YVR–CTS direct service starting Dec 2026
        - ground_access_urls[]   # resort transit pages showing access from CTS
    - resort_urls[]

    Rules:
    - If any field is missing from the answer, set it to null (or an empty list for URL arrays).
    - Do not perform any external lookup; only use what’s in the answer.
    - Keep numbers as strings if the answer uses ranges or different units.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def merge_sources(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for arr in lists:
        if not arr:
            continue
        for url in arr:
            if not url:
                continue
            u = url.strip()
            if u and u not in seen:
                seen.add(u)
                result.append(u)
    return result


def has_digits(s: Optional[str]) -> bool:
    if not s:
        return False
    return any(ch.isdigit() for ch in s)


# --------------------------------------------------------------------------- #
# Verification: North American resort                                         #
# --------------------------------------------------------------------------- #
async def verify_north_american_resort(evaluator: Evaluator, parent_node, na: ResortItem) -> None:
    na_node = evaluator.add_parallel(
        id="North_American_Resort",
        desc="Identify a ski resort in North America that meets all specified accessibility and specification requirements, and provide complete information about it.",
        parent=parent_node,
        critical=False
    )

    # 1) Resort name provided (non-critical)
    evaluator.add_custom_node(
        result=bool(na and na.name and na.name.strip()),
        id="na_resort_name_provided",
        desc="The name of the North American resort must be provided.",
        parent=na_node,
        critical=False
    )

    # Prepare common fields
    resort_name = na.name or "the resort"
    a = na.airport_access or AirportAccess()
    dest_text = a.airport_name or (a.airport_iata or "the gateway airport")
    dest_with_iata = f"{a.airport_name} ({a.airport_iata})" if (a.airport_name and a.airport_iata) else dest_text
    airline_sources = merge_sources(a.verification_urls, a.ground_access_urls)

    # 2) BA Heathrow accessibility (critical)
    ba_access_node = evaluator.add_leaf(
        id="na_ba_heathrow_accessibility",
        desc="The resort must be accessible from an airport that has direct British Airways flights from London Heathrow.",
        parent=na_node,
        critical=True
    )
    ba_claim = (
        f"British Airways operates a non-stop direct flight from London Heathrow (LHR) to {dest_with_iata}. "
        f"This airport serves as a practical gateway to reach {resort_name} via ground transport (e.g., shuttle, bus, or car)."
    )
    await evaluator.verify(
        claim=ba_claim,
        node=ba_access_node,
        sources=airline_sources,
        additional_instruction=(
            "Verify two things from the provided URLs: "
            "1) BA (British Airways) has direct LHR → the specified airport (IATA if provided). Seasonal service counts as direct. "
            "2) The resort is accessible from that airport per resort or transport pages. "
            "If either is not clearly supported by the URLs, mark as not supported."
        )
    )

    # 3) Airport access info explicitly provided (non-critical)
    route_text = " ".join(filter(None, [a.route_description or "", a.airline or ""])).lower()
    airport_info_provided = bool(
        (a.airport_name and (a.route_description or a.airline)) and
        (("british airways" in route_text) and (("heathrow" in route_text) or ("lhr" in route_text)))
    )
    evaluator.add_custom_node(
        result=airport_info_provided,
        id="na_airport_access_info_provided",
        desc="The airport name and airline route information (British Airways from Heathrow) must be explicitly provided.",
        parent=na_node,
        critical=False
    )

    # Stats sources
    stats_sources = merge_sources(na.stats.stats_urls, na.resort_urls)

    # 4) Vertical drop minimum 400m (critical)
    vd_min_node = evaluator.add_leaf(
        id="na_vertical_drop_minimum",
        desc="The resort must have a vertical drop of at least 400 meters (approximately 1,312 feet).",
        parent=na_node,
        critical=True
    )
    vd_str = na.stats.vertical_drop_m or ""
    vd_claim = (
        f"{resort_name} has a vertical drop of at least 400 meters. "
        f"The referenced page(s) show the resort's vertical drop (noted as '{vd_str}' in the answer)."
    )
    await evaluator.verify(
        claim=vd_claim,
        node=vd_min_node,
        sources=stats_sources,
        additional_instruction=(
            "Check the resort's official stats or trusted ski data pages for vertical drop. "
            "Accept values ≥ 400 m. If only feet are shown, ~1,312 ft ≈ 400 m. "
            "Minor rounding differences are acceptable."
        )
    )

    # 5) Vertical drop value provided (non-critical)
    evaluator.add_custom_node(
        result=has_digits(na.stats.vertical_drop_m),
        id="na_vertical_drop_value_provided",
        desc="The actual vertical drop value of the resort must be stated in the answer.",
        parent=na_node,
        critical=False
    )

    # 6) Season opening timing Nov or earlier for 2025–26 or 2026–27 (critical)
    opening_sources = merge_sources(na.opening.opening_urls)
    opening_timing_node = evaluator.add_leaf(
        id="na_season_opening_timing",
        desc="The resort's ski season must open in November or earlier for the 2025-2026 or 2026-2027 season.",
        parent=na_node,
        critical=True
    )
    season_str = na.opening.season or "2026–2027"
    open_str = na.opening.scheduled_opening_date or "(opening date not specified)"
    opening_claim = (
        f"For the {season_str} ski season, {resort_name} is scheduled to open in November or earlier "
        f"(October or November). The answer states: '{open_str}'."
    )
    await evaluator.verify(
        claim=opening_claim,
        node=opening_timing_node,
        sources=opening_sources,
        additional_instruction=(
            "Look for an official resort announcement, events page, or season calendar that states the scheduled "
            "season opening in November or earlier (Oct/Nov) for either 2025–2026 or 2026–2027. "
            "Approximate phrasing such as 'early November' or 'late October' is acceptable."
        )
    )

    # 7) Opening date provided (non-critical)
    evaluator.add_custom_node(
        result=bool(na.opening.scheduled_opening_date and na.opening.scheduled_opening_date.strip()),
        id="na_opening_date_provided",
        desc="The scheduled opening date for the relevant season must be stated in the answer.",
        parent=na_node,
        critical=False
    )

    # 8) Base elevation provided (non-critical)
    evaluator.add_custom_node(
        result=bool(na.stats.base_elevation and na.stats.base_elevation.strip()),
        id="na_base_elevation_provided",
        desc="The base elevation of the resort must be provided in the answer.",
        parent=na_node,
        critical=False
    )

    # 9) Summit elevation provided (non-critical)
    evaluator.add_custom_node(
        result=bool(na.stats.summit_elevation and na.stats.summit_elevation.strip()),
        id="na_summit_elevation_provided",
        desc="The summit/top elevation of the resort must be provided in the answer.",
        parent=na_node,
        critical=False
    )

    # 10) BA baggage policy reference (non-critical, but source-grounded)
    baggage = na.baggage_policy or BAGearPolicy()
    bag_sources = merge_sources(baggage.policy_urls)
    baggage_node = evaluator.add_leaf(
        id="na_baggage_policy_reference",
        desc="A note about British Airways ski equipment baggage policy compliance (dimensions within 190 x 75 x 65cm) must be included.",
        parent=na_node,
        critical=False
    )
    baggage_claim = (
        "British Airways' sports equipment policy allows ski/snowboard items up to 190 x 75 x 65 cm to be checked "
        "as a piece of baggage (subject to standard weight limits or fees as applicable). The answer includes this note."
    )
    await evaluator.verify(
        claim=baggage_claim,
        node=baggage_node,
        sources=bag_sources,
        additional_instruction=(
            "Use BA's official sports equipment/baggage policy pages to verify the maximum dimensions "
            "of 190 x 75 x 65 cm for ski/snowboard equipment. If the provided URL(s) do not confirm this, mark as not supported."
        )
    )


# --------------------------------------------------------------------------- #
# Verification: Japanese resort                                               #
# --------------------------------------------------------------------------- #
async def verify_japanese_resort(evaluator: Evaluator, parent_node, jp: ResortItem) -> None:
    jp_node = evaluator.add_parallel(
        id="Japanese_Resort",
        desc="Identify a ski resort in Japan that meets all specified accessibility and specification requirements, and provide complete information about it.",
        parent=parent_node,
        critical=False
    )

    # 1) Resort name provided (non-critical)
    evaluator.add_custom_node(
        result=bool(jp and jp.name and jp.name.strip()),
        id="jp_resort_name_provided",
        desc="The name of the Japanese resort must be provided.",
        parent=jp_node,
        critical=False
    )

    # Prepare common fields
    resort_name = jp.name or "the resort"
    a = jp.airport_access or AirportAccess()
    dest_text = a.airport_name or (a.airport_iata or "the gateway airport")
    dest_with_iata = f"{a.airport_name} ({a.airport_iata})" if (a.airport_name and a.airport_iata) else dest_text
    airline_sources = merge_sources(a.verification_urls, a.ground_access_urls)

    # 2) Air Canada accessibility via YVR->CTS starting Dec 2026 (critical)
    ac_access_node = evaluator.add_leaf(
        id="jp_air_canada_accessibility",
        desc="The resort must be accessible from the airport served by the new Air Canada direct flight from Vancouver starting December 2026 (Sapporo Chitose Airport).",
        parent=jp_node,
        critical=True
    )
    ac_claim = (
        f"Air Canada operates a new direct (non-stop) flight from Vancouver (YVR) to Sapporo New Chitose (CTS) starting December 2026, "
        f"and {resort_name} is accessible from {dest_with_iata} (e.g., by shuttle, bus, train, or car)."
    )
    await evaluator.verify(
        claim=ac_claim,
        node=ac_access_node,
        sources=airline_sources,
        additional_instruction=(
            "Confirm two points using the provided URLs: "
            "1) Air Canada's new nonstop service YVR ↔ CTS begins in December 2026 (press release/schedule accepted). "
            "2) The resort is reachable from New Chitose Airport (CTS) via ground transport per resort/transport pages. "
            "If either is missing or unsupported by the URLs, mark as not supported."
        )
    )

    # 3) Airport access info explicitly provided (non-critical)
    route_text = " ".join(filter(None, [a.route_description or "", a.airline or ""])).lower()
    airport_info_provided = bool(
        (a.airport_name and (a.route_description or a.airline)) and
        (("air canada" in route_text) and ("vancouver" in route_text) and (("sapporo" in route_text) or ("chitose" in route_text) or ("cts" in route_text)))
    )
    evaluator.add_custom_node(
        result=airport_info_provided,
        id="jp_airport_access_info_provided",
        desc="The airport name and airline route information (Air Canada from Vancouver to Sapporo Chitose) must be explicitly provided.",
        parent=jp_node,
        critical=False
    )

    # Stats sources
    stats_sources = merge_sources(jp.stats.stats_urls, jp.resort_urls)

    # 4) Vertical drop minimum 400m (critical)
    vd_min_node = evaluator.add_leaf(
        id="jp_vertical_drop_minimum",
        desc="The resort must have a vertical drop of at least 400 meters.",
        parent=jp_node,
        critical=True
    )
    vd_str = jp.stats.vertical_drop_m or ""
    vd_claim = (
        f"{resort_name} has a vertical drop of at least 400 meters. "
        f"The referenced page(s) show the resort's vertical drop (noted as '{vd_str}' in the answer)."
    )
    await evaluator.verify(
        claim=vd_claim,
        node=vd_min_node,
        sources=stats_sources,
        additional_instruction=(
            "Check the resort's official stats or trusted ski data pages for vertical drop. "
            "Accept values ≥ 400 m. If only feet are shown, ~1,312 ft ≈ 400 m. "
            "Minor rounding differences are acceptable."
        )
    )

    # 5) Vertical drop value provided (non-critical)
    evaluator.add_custom_node(
        result=has_digits(jp.stats.vertical_drop_m),
        id="jp_vertical_drop_value_provided",
        desc="The actual vertical drop value of the resort must be stated in the answer.",
        parent=jp_node,
        critical=False
    )

    # 6) Season opening timing Nov or earlier for 2026–27 (critical)
    opening_sources = merge_sources(jp.opening.opening_urls)
    opening_timing_node = evaluator.add_leaf(
        id="jp_season_opening_timing",
        desc="The resort must open in November or earlier for the 2026-2027 winter season.",
        parent=jp_node,
        critical=True
    )
    season_str = jp.opening.season or "2026–2027"
    open_str = jp.opening.scheduled_opening_date or "(opening date not specified)"
    opening_claim = (
        f"For the {season_str} ski season, {resort_name} is scheduled to open in November or earlier "
        f"(October or November). The answer states: '{open_str}'."
    )
    await evaluator.verify(
        claim=opening_claim,
        node=opening_timing_node,
        sources=opening_sources,
        additional_instruction=(
            "Look for an official resort announcement, events page, or season calendar that states the scheduled "
            "season opening in November or earlier (Oct/Nov) for 2026–2027 specifically. "
            "Approximate phrasing such as 'early November' or 'late October' is acceptable."
        )
    )

    # 7) Opening date provided (non-critical)
    evaluator.add_custom_node(
        result=bool(jp.opening.scheduled_opening_date and jp.opening.scheduled_opening_date.strip()),
        id="jp_opening_date_provided",
        desc="The scheduled opening date for the 2026-2027 season must be stated in the answer.",
        parent=jp_node,
        critical=False
    )

    # 8) Base elevation provided (non-critical)
    evaluator.add_custom_node(
        result=bool(jp.stats.base_elevation and jp.stats.base_elevation.strip()),
        id="jp_base_elevation_provided",
        desc="The base elevation of the resort must be provided in the answer.",
        parent=jp_node,
        critical=False
    )

    # 9) Summit elevation provided (non-critical)
    evaluator.add_custom_node(
        result=bool(jp.stats.summit_elevation and jp.stats.summit_elevation.strip()),
        id="jp_summit_elevation_provided",
        desc="The summit/top elevation of the resort must be provided in the answer.",
        parent=jp_node,
        critical=False
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Two sub-resorts evaluated independently
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

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_ski_plan(),
        template_class=SkiPlanExtraction,
        extraction_name="resort_extraction",
    )

    # Optional: record threshold policy used
    evaluator.add_custom_info(
        info={
            "vertical_drop_min_m": 400,
            "opening_month_cutoff": "November (Nov) or earlier",
            "na_opening_seasons_allowed": ["2025–2026", "2026–2027"],
            "jp_opening_season_required": "2026–2027",
            "ba_baggage_target_dimensions_cm": "190 x 75 x 65",
        },
        info_type="policy",
        info_name="evaluation_policy"
    )

    # Build top-level node from rubric
    selection_node = evaluator.add_parallel(
        id="Winter_Ski_Resort_Selection",
        desc="Identify two ski resorts meeting specified criteria: one in North America accessible via British Airways from London Heathrow, and one in Japan accessible via Air Canada from Vancouver.",
        parent=root,
        critical=False
    )

    # Verify NA resort
    await verify_north_american_resort(evaluator, selection_node, extracted.north_america or ResortItem())

    # Verify JP resort
    await verify_japanese_resort(evaluator, selection_node, extracted.japan or ResortItem())

    # Return structured result
    return evaluator.get_summary()