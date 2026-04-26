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
TASK_ID = "denver_rmnp_acadia_2025_planning"
TASK_DESCRIPTION = """
A family of four from Denver, Colorado is planning a two-week outdoor recreation trip in July 2025. Their itinerary includes: visiting Rocky Mountain National Park for 4 days (including the Bear Lake area during peak daytime hours), flying from Denver International Airport to Providence, Rhode Island via Breeze Airways, driving to and visiting Acadia National Park in Maine for 5 days, watching the sunrise from Cadillac Mountain at Acadia National Park, camping at both Rocky Mountain National Park and Acadia National Park, and visiting one Colorado state park near Denver for one day before departing. Provide a comprehensive planning guide that includes: (1) the days of the week when Breeze Airways operates direct flights from Denver to Providence, (2) all required entrance passes, permits, and vehicle reservations needed for Rocky Mountain National Park, including the standard 7-day vehicle pass cost, the date range and hours when timed entry is required, the types of timed entry reservations available, and the cost per timed entry reservation, (3) the names of available campgrounds at Rocky Mountain National Park and the reservation system used for booking, (4) all required entrance passes and vehicle reservations needed for Acadia National Park, including the standard 7-day vehicle pass cost, the date range when Cadillac Summit Road reservations are required, the types of Cadillac Summit Road vehicle reservations available, and the cost per vehicle reservation, (5) the names of the three campgrounds at Acadia National Park and the reservation system used for booking, (6) the required pass or daily fee for visiting Colorado state parks, including the cost of a daily vehicle pass and at least two annual pass options with their costs, (7) information about the America the Beautiful Annual Pass, including its cost and what it covers, (8) a cost-optimized pass purchase strategy that compares the total cost of buying individual park entrance passes for both national parks versus purchasing an America the Beautiful Annual Pass, with a clear recommendation, (9) the distance in miles and approximate drive time from Providence, Rhode Island to Acadia National Park in Maine, and (10) the general booking window (how many months in advance) and standard release time for camping reservations on Recreation.gov. For each piece of information, provide specific details with supporting reference URLs.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FlightInfo(BaseModel):
    days_of_week: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class RMNPEntrance(BaseModel):
    vehicle_pass_cost: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RMNPTimedEntry(BaseModel):
    date_range: Optional[str] = None
    hours: Optional[str] = None
    types: List[str] = Field(default_factory=list)
    reservation_cost: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RMNPCamping(BaseModel):
    campgrounds: List[str] = Field(default_factory=list)
    booking_system: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AcadiaEntrance(BaseModel):
    vehicle_pass_cost: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AcadiaCadillac(BaseModel):
    date_range: Optional[str] = None
    reservation_types: List[str] = Field(default_factory=list)
    reservation_cost: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AcadiaCamping(BaseModel):
    campgrounds: List[str] = Field(default_factory=list)
    booking_system: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class COStateParks(BaseModel):
    daily_vehicle_pass_cost: Optional[str] = None
    annual_pass_options: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class AmericaBeautifulPass(BaseModel):
    cost: Optional[str] = None
    coverage: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CostOptimization(BaseModel):
    individual_total: Optional[str] = None
    atb_total: Optional[str] = None
    recommendation: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DistanceInfo(BaseModel):
    miles: Optional[str] = None
    hours: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RecGovBooking(BaseModel):
    booking_window: Optional[str] = None
    release_time: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TripExtraction(BaseModel):
    flight_info: Optional[FlightInfo] = None
    rmnp_entrance: Optional[RMNPEntrance] = None
    rmnp_timed: Optional[RMNPTimedEntry] = None
    rmnp_camping: Optional[RMNPCamping] = None
    acadia_entrance: Optional[AcadiaEntrance] = None
    acadia_cadillac: Optional[AcadiaCadillac] = None
    acadia_camping: Optional[AcadiaCamping] = None
    co_state_parks: Optional[COStateParks] = None
    atb_pass: Optional[AmericaBeautifulPass] = None
    cost_optimization: Optional[CostOptimization] = None
    distance: Optional[DistanceInfo] = None
    recgov: Optional[RecGovBooking] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_flight_info() -> str:
    return """
    From the answer, extract Breeze Airways direct (nonstop) flight information from Denver (DEN) to Providence (PVD).
    Return:
    - days_of_week: list of day names when direct DEN→PVD flights operate (e.g., ["Thursday","Sunday"])
    - sources: all URLs cited that support these operating days or the Breeze schedule/route
    Only include days explicitly stated in the answer text and URLs explicitly present.
    """


def prompt_extract_rmnp_entrance() -> str:
    return """
    From the answer, extract Rocky Mountain National Park entrance fee info.
    Return:
    - vehicle_pass_cost: the standard 7-day private vehicle entrance pass cost (as written, e.g., "$35")
    - sources: URLs supporting this fee
    """


def prompt_extract_rmnp_timed_entry() -> str:
    return """
    From the answer, extract Rocky Mountain National Park timed entry details.
    Return:
    - date_range: the date range when timed entry is required (as written)
    - hours: the hours during which timed entry is required (as written)
    - types: list of reservation types (e.g., ["Bear Lake Road","General park entry"])
    - reservation_cost: cost per timed entry reservation (e.g., "$2")
    - sources: URLs supporting timed entry details
    """


def prompt_extract_rmnp_camping() -> str:
    return """
    From the answer, extract RMNP campground and booking info.
    Return:
    - campgrounds: list of named campgrounds mentioned (e.g., ["Moraine Park","Aspenglen"])
    - booking_system: the official reservation/booking system named (e.g., "Recreation.gov")
    - sources: URLs supporting the campground info and booking system
    """


def prompt_extract_acadia_entrance() -> str:
    return """
    From the answer, extract Acadia National Park entrance fee info.
    Return:
    - vehicle_pass_cost: the standard 7-day private vehicle entrance pass cost (as written, e.g., "$35")
    - sources: URLs supporting this fee
    """


def prompt_extract_acadia_cadillac() -> str:
    return """
    From the answer, extract Cadillac Summit Road vehicle reservation details at Acadia.
    Return:
    - date_range: the date range when reservations are required (as written)
    - reservation_types: list of types (e.g., ["Sunrise","Daytime"])
    - reservation_cost: cost per vehicle reservation (e.g., "$6")
    - sources: URLs supporting Cadillac Summit Road reservation info
    """


def prompt_extract_acadia_camping() -> str:
    return """
    From the answer, extract Acadia campground and booking info.
    Return:
    - campgrounds: list of named campgrounds mentioned (e.g., ["Blackwoods","Seawall","Schoodic Woods"])
    - booking_system: the official reservation/booking system named (e.g., "Recreation.gov")
    - sources: URLs supporting the campground info and booking system
    """


def prompt_extract_co_state_parks() -> str:
    return """
    From the answer, extract Colorado state parks fee/pass details.
    Return:
    - daily_vehicle_pass_cost: the daily vehicle pass cost as written (e.g., "$10" or "$10–$11")
    - annual_pass_options: list including at least two named annual passes with costs, e.g., ["Vehicle Annual Pass $80","Keep Colorado Wild Pass $29"]
    - sources: URLs supporting these fees/passes
    """


def prompt_extract_atb_pass() -> str:
    return """
    From the answer, extract America the Beautiful Annual Pass details.
    Return:
    - cost: the annual pass cost (e.g., "$80")
    - coverage: what the pass covers (as written)
    - sources: URLs supporting the cost and coverage
    """


def prompt_extract_cost_optimization() -> str:
    return """
    From the answer, extract the cost optimization comparison and recommendation.
    Return:
    - individual_total: the total cost the answer computed for buying both national park entrance passes individually (as written)
    - atb_total: the total cost the answer used for the America the Beautiful pass (as written)
    - recommendation: the clear recommendation the answer makes based on the comparison (as written)
    - sources: any URLs the answer cites for the costs used in the comparison (e.g., RMNP fee, Acadia fee, ATB pass)
    """


def prompt_extract_distance() -> str:
    return """
    From the answer, extract the driving distance and time from Providence, Rhode Island to Acadia National Park.
    Return:
    - miles: the distance in miles as written (e.g., "327 miles")
    - hours: the approximate driving time as written (e.g., "5–6 hours")
    - sources: URLs supporting the distance/time (e.g., Google Maps)
    """


def prompt_extract_recgov() -> str:
    return """
    From the answer, extract Recreation.gov booking timing info.
    Return:
    - booking_window: the general booking window (e.g., "up to 6 months in advance")
    - release_time: the standard release time (e.g., "10 AM ET")
    - sources: URLs supporting these Recreation.gov booking details
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


async def _add_and_verify_leaf(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    claim: str,
    *,
    critical: bool = True,
    sources: Optional[List[str]] = None,
    additional_instruction: Optional[str] = None,
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction or "None",
    )


async def _verify_with_reference_support(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    factual_claim: str,
    sources: Optional[List[str]],
    *,
    url_instruction: Optional[str] = None,
    fallback_when_no_url: Optional[str] = None,
) -> None:
    """
    If sources provided -> verify factual claim against URLs.
    If no sources -> verify that the answer provides at least one supporting URL (fail if not).
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=True,
    )
    if sources and len(sources) > 0:
        await evaluator.verify(
            claim=factual_claim,
            node=leaf,
            sources=sources,
            additional_instruction=url_instruction or "Verify the claim using the provided webpage(s). Allow minor wording differences.",
        )
    else:
        # Fall back: require that the answer actually provided at least one URL
        await evaluator.verify(
            claim="The answer provides at least one supporting reference URL for this information.",
            node=leaf,
            sources=None,
            additional_instruction=fallback_when_no_url
            or "Examine the answer text. If it does not include at least one URL supporting this item, mark this incorrect.",
        )


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_breeze_airways_section(
    evaluator: Evaluator,
    parent,
    flight: FlightInfo,
):
    section = evaluator.add_parallel(
        id="breeze_airways_flight_info",
        desc="Breeze Airways direct flight schedule from Denver to Providence",
        parent=parent,
        critical=True,
    )

    # flight_days
    await _add_and_verify_leaf(
        evaluator,
        section,
        "flight_days",
        "Days of week Breeze operates direct DEN→PVD flights (Thursdays and Sundays)",
        claim=(
            "The answer states that Breeze Airways operates direct (nonstop) flights from Denver (DEN) to "
            "Providence (PVD) on Thursdays and Sundays."
        ),
        additional_instruction=(
            "Check only the answer content for the stated operating days. Allow minor wording like 'Thu'/'Sun'."
        ),
    )

    # reference_url_flights
    await _verify_with_reference_support(
        evaluator,
        section,
        "reference_url_flights",
        "Reference URL supporting the Breeze DEN→PVD direct-flight operating days",
        factual_claim=(
            "Breeze Airways operates direct (nonstop) flights from Denver (DEN) to Providence (PVD) on "
            "Thursdays and Sundays."
        ),
        sources=flight.sources if flight else [],
        url_instruction=(
            "Look for a Breeze schedule, route map, or booking calendar indicating nonstop DEN→PVD service on "
            "Thursdays and Sundays. Accept reasonable equivalents."
        ),
        fallback_when_no_url="If no link is included in the answer, mark this as incorrect.",
    )


async def build_rmnp_entrance_section(
    evaluator: Evaluator,
    parent,
    rmnp_fee: RMNPEntrance,
):
    section = evaluator.add_parallel(
        id="rmnp_entrance_requirement",
        desc="Rocky Mountain National Park entrance pass requirement",
        parent=parent,
        critical=True,
    )

    # rmnp_entrance_fee
    await _add_and_verify_leaf(
        evaluator,
        section,
        "rmnp_entrance_fee",
        "Standard 7-day vehicle pass cost for RMNP ($35)",
        claim="The answer states that the standard 7-day private vehicle entrance pass for Rocky Mountain National Park costs $35.",
        additional_instruction="Check only the answer content for the stated fee.",
    )

    # reference_url_rmnp_fee
    await _verify_with_reference_support(
        evaluator,
        section,
        "reference_url_rmnp_fee",
        "Reference URL supporting RMNP entrance fee",
        factual_claim="The standard 7-day private vehicle entrance pass for Rocky Mountain National Park costs $35.",
        sources=rmnp_fee.sources if rmnp_fee else [],
        url_instruction="Verify the exact fee on an official NPS or Recreation.gov page.",
        fallback_when_no_url="If the answer provides no URL for this fee, mark this incorrect.",
    )


async def build_rmnp_timed_entry_section(
    evaluator: Evaluator,
    parent,
    rmnp_timed: RMNPTimedEntry,
):
    section = evaluator.add_parallel(
        id="rmnp_timed_entry",
        desc="RMNP timed entry permit requirements",
        parent=parent,
        critical=True,
    )

    # timed_entry_dates
    await _add_and_verify_leaf(
        evaluator,
        section,
        "timed_entry_dates",
        "Timed entry required date range (May 23–October 13, 2025)",
        claim="The answer states that RMNP timed entry is required from May 23, 2025 through October 13, 2025.",
        additional_instruction="Check only the answer content; allow minor formatting variations of the dates.",
    )

    # timed_entry_hours
    await _add_and_verify_leaf(
        evaluator,
        section,
        "timed_entry_hours",
        "Timed entry required hours (9 AM–2 PM daily)",
        claim="The answer states that timed entry applies daily from 9 AM to 2 PM.",
        additional_instruction="Check only the answer content; allow minor formatting like '9am–2pm'.",
    )

    # timed_entry_types
    await _add_and_verify_leaf(
        evaluator,
        section,
        "timed_entry_types",
        "Timed entry reservation types (Bear Lake Road permit; general park entry permit)",
        claim=(
            "The answer states there are two timed-entry reservation types at RMNP: a Bear Lake Road (Bear Lake Road Corridor) permit "
            "and a general park entry (Park Access) permit."
        ),
        additional_instruction=(
            "Accept synonyms like 'Park Access + Bear Lake Road' for the Bear Lake Road option and 'Park Access' for general park entry."
        ),
    )

    # timed_entry_cost
    await _add_and_verify_leaf(
        evaluator,
        section,
        "timed_entry_cost",
        "Cost per RMNP timed entry reservation ($2)",
        claim="The answer states that each RMNP timed entry reservation costs $2.",
        additional_instruction="Check only the answer content for the stated reservation cost.",
    )

    # reference_url_timed_entry
    await _verify_with_reference_support(
        evaluator,
        section,
        "reference_url_timed_entry",
        "Reference URL supporting RMNP timed entry details",
        factual_claim=(
            "For 2025, RMNP requires timed entry from May 23 to October 13, applies from 9 AM to 2 PM daily, "
            "offers Bear Lake Road and general Park Access permit types, and each reservation costs $2."
        ),
        sources=rmnp_timed.sources if rmnp_timed else [],
        url_instruction=(
            "Verify all elements (date range, hours, types, cost) from an official NPS or Recreation.gov page. "
            "Allow minor wording differences."
        ),
        fallback_when_no_url="If no URLs are provided in the answer, mark this as incorrect.",
    )


async def build_rmnp_camping_section(
    evaluator: Evaluator,
    parent,
    rmnp_camping: RMNPCamping,
):
    section = evaluator.add_parallel(
        id="rmnp_camping",
        desc="RMNP campground names and how to book them",
        parent=parent,
        critical=True,
    )

    # rmnp_campground_moraine_park
    await _add_and_verify_leaf(
        evaluator,
        section,
        "rmnp_campground_moraine_park",
        "Includes Moraine Park Campground as an available RMNP campground",
        claim="The answer includes Moraine Park Campground as an available campground at Rocky Mountain National Park.",
        additional_instruction="Check only the answer content; allow minor wording differences.",
    )

    # rmnp_campground_aspenglen
    await _add_and_verify_leaf(
        evaluator,
        section,
        "rmnp_campground_aspenglen",
        "Includes Aspenglen Campground as an available RMNP campground",
        claim="The answer includes Aspenglen Campground as an available campground at Rocky Mountain National Park.",
        additional_instruction="Check only the answer content; allow minor wording differences.",
    )

    # rmnp_reservation_system
    await _add_and_verify_leaf(
        evaluator,
        section,
        "rmnp_reservation_system",
        "Correctly identifies the official reservation/booking system used for RMNP campgrounds",
        claim="The answer states that RMNP campground reservations are made on Recreation.gov.",
        additional_instruction="Check only the answer content.",
    )

    # reference_url_rmnp_camping
    await _verify_with_reference_support(
        evaluator,
        section,
        "reference_url_rmnp_camping",
        "Reference URL supporting RMNP camping/campground information and the stated booking system",
        factual_claim=(
            "Moraine Park and Aspenglen are official campgrounds in Rocky Mountain National Park, and campground reservations "
            "are handled through Recreation.gov."
        ),
        sources=rmnp_camping.sources if rmnp_camping else [],
        url_instruction="Confirm campground names and that reservations are via Recreation.gov from official pages.",
        fallback_when_no_url="If the answer includes no supporting campground/booking URL, mark this as incorrect.",
    )


async def build_acadia_entrance_section(
    evaluator: Evaluator,
    parent,
    acadia_fee: AcadiaEntrance,
):
    section = evaluator.add_parallel(
        id="acadia_entrance_requirement",
        desc="Acadia National Park entrance pass requirement",
        parent=parent,
        critical=True,
    )

    # acadia_entrance_fee
    await _add_and_verify_leaf(
        evaluator,
        section,
        "acadia_entrance_fee",
        "Standard 7-day vehicle pass cost for Acadia NP ($35)",
        claim="The answer states that the standard 7-day private vehicle entrance pass for Acadia National Park costs $35.",
        additional_instruction="Check only the answer content.",
    )

    # reference_url_acadia_fee
    await _verify_with_reference_support(
        evaluator,
        section,
        "reference_url_acadia_fee",
        "Reference URL supporting Acadia entrance fee",
        factual_claim="The standard 7-day private vehicle entrance pass for Acadia National Park costs $35.",
        sources=acadia_fee.sources if acadia_fee else [],
        url_instruction="Verify the fee on an official NPS or Recreation.gov page.",
        fallback_when_no_url="If the answer includes no supporting fee URL, mark this incorrect.",
    )


async def build_acadia_cadillac_section(
    evaluator: Evaluator,
    parent,
    cad: AcadiaCadillac,
):
    section = evaluator.add_parallel(
        id="acadia_cadillac_reservation",
        desc="Cadillac Summit Road vehicle reservation requirements at Acadia",
        parent=parent,
        critical=True,
    )

    # cadillac_dates
    await _add_and_verify_leaf(
        evaluator,
        section,
        "cadillac_dates",
        "Date range when Cadillac Summit Road reservations are required (May 21–October 26, 2025)",
        claim="The answer states that Cadillac Summit Road vehicle reservations are required from May 21 to October 26, 2025.",
        additional_instruction="Check only the answer content.",
    )

    # cadillac_reservation_types
    await _add_and_verify_leaf(
        evaluator,
        section,
        "cadillac_reservation_types",
        "Cadillac Summit Road reservation types (Sunrise and Daytime)",
        claim="The answer states that Cadillac Summit Road has two reservation types: Sunrise and Daytime.",
        additional_instruction="Check only the answer content.",
    )

    # cadillac_cost
    await _add_and_verify_leaf(
        evaluator,
        section,
        "cadillac_cost",
        "Cost per Cadillac Summit Road vehicle reservation ($6)",
        claim="The answer states that each Cadillac Summit Road vehicle reservation costs $6.",
        additional_instruction="Check only the answer content.",
    )

    # reference_url_cadillac
    await _verify_with_reference_support(
        evaluator,
        section,
        "reference_url_cadillac",
        "Reference URL supporting Cadillac Summit Road reservation requirements",
        factual_claim=(
            "At Acadia in 2025, Cadillac Summit Road requires vehicle reservations from May 21 to October 26, "
            "with Sunrise and Daytime reservation types, and each reservation costs $6."
        ),
        sources=cad.sources if cad else [],
        url_instruction="Verify the dates, reservation types, and fee from official NPS/Recreation.gov pages.",
        fallback_when_no_url="If the answer includes no supporting URL, mark this incorrect.",
    )


async def build_acadia_camping_section(
    evaluator: Evaluator,
    parent,
    acadia_camping: AcadiaCamping,
):
    section = evaluator.add_parallel(
        id="acadia_camping",
        desc="Acadia campground names and how to book them",
        parent=parent,
        critical=True,
    )

    # acadia_campgrounds
    await _add_and_verify_leaf(
        evaluator,
        section,
        "acadia_campgrounds",
        "Names the three Acadia campgrounds (Blackwoods, Seawall, Schoodic Woods)",
        claim="The answer lists all three Acadia campgrounds by name: Blackwoods, Seawall, and Schoodic Woods.",
        additional_instruction="Check only the answer content; all three must be listed.",
    )

    # acadia_reservation_system
    await _add_and_verify_leaf(
        evaluator,
        section,
        "acadia_reservation_system",
        "Correctly identifies the official reservation/booking system used for Acadia campgrounds",
        claim="The answer states that Acadia campground reservations are made on Recreation.gov.",
        additional_instruction="Check only the answer content.",
    )

    # reference_url_acadia_camping
    await _verify_with_reference_support(
        evaluator,
        section,
        "reference_url_acadia_camping",
        "Reference URL supporting Acadia camping/campground information and the stated booking system",
        factual_claim=(
            "Blackwoods, Seawall, and Schoodic Woods are official campgrounds at Acadia National Park and reservations are on Recreation.gov."
        ),
        sources=acadia_camping.sources if acadia_camping else [],
        url_instruction="Confirm campground names and booking system on official NPS/Recreation.gov pages.",
        fallback_when_no_url="If the answer includes no supporting URL, mark this incorrect.",
    )


async def build_colorado_state_parks_section(
    evaluator: Evaluator,
    parent,
    co: COStateParks,
):
    section = evaluator.add_parallel(
        id="colorado_state_park_pass",
        desc="Colorado state park fee/pass requirements",
        parent=parent,
        critical=True,
    )

    # daily_pass_cost
    await _add_and_verify_leaf(
        evaluator,
        section,
        "daily_pass_cost",
        "Daily vehicle pass cost for Colorado state parks ($10–$11)",
        claim="The answer states that the daily vehicle pass for Colorado state parks is approximately $10–$11, depending on the park.",
        additional_instruction="Check only the answer content; allow $10 or $11 and wording like 'varies by park'.",
    )

    # annual_pass_options
    await _add_and_verify_leaf(
        evaluator,
        section,
        "annual_pass_options",
        "At least two annual pass options with costs (Vehicle Annual Pass $80; Keep Colorado Wild Pass $29)",
        claim=(
            "The answer lists at least two annual Colorado state park pass options with costs, including the Vehicle Annual Pass "
            "at $80 and the Keep Colorado Wild Pass at $29."
        ),
        additional_instruction="Check only the answer content; minor wording variations allowed.",
    )

    # reference_url_co_parks
    await _verify_with_reference_support(
        evaluator,
        section,
        "reference_url_co_parks",
        "Reference URL supporting Colorado state parks fees/passes",
        factual_claim=(
            "Colorado state parks daily vehicle pass is about $10–$11, the Vehicle Annual Pass costs $80, and the Keep Colorado Wild Pass costs $29."
        ),
        sources=co.sources if co else [],
        url_instruction="Verify fees and pass costs on official Colorado Parks & Wildlife or state sources.",
        fallback_when_no_url="If the answer includes no supporting URL, mark this incorrect.",
    )


async def build_atb_section(
    evaluator: Evaluator,
    parent,
    atb: AmericaBeautifulPass,
):
    section = evaluator.add_parallel(
        id="america_beautiful_pass",
        desc="America the Beautiful Annual Pass details",
        parent=parent,
        critical=True,
    )

    # atb_cost
    await _add_and_verify_leaf(
        evaluator,
        section,
        "atb_cost",
        "America the Beautiful Annual Pass cost ($80)",
        claim="The answer states that the America the Beautiful Annual Pass costs $80.",
        additional_instruction="Check only the answer content.",
    )

    # atb_coverage
    await _add_and_verify_leaf(
        evaluator,
        section,
        "atb_coverage",
        "States what the pass covers (entrance to national parks and federal recreation sites for one year)",
        claim=(
            "The answer states that the America the Beautiful Annual Pass covers entrance fees at national parks and other federal recreational sites for one year."
        ),
        additional_instruction="Check only the answer content; minor wording differences allowed.",
    )

    # reference_url_atb
    await _verify_with_reference_support(
        evaluator,
        section,
        "reference_url_atb",
        "Reference URL supporting America the Beautiful pass cost and coverage",
        factual_claim="The America the Beautiful Annual Pass costs $80 and covers entrance fees at national parks and federal recreation sites for one year.",
        sources=atb.sources if atb else [],
        url_instruction="Verify cost and coverage on official USGS/NPS pages.",
        fallback_when_no_url="If the answer includes no supporting URL, mark this incorrect.",
    )


async def build_cost_optimization_section(
    evaluator: Evaluator,
    parent,
    cost_opt: CostOptimization,
    rmnp_fee: RMNPEntrance,
    acadia_fee: AcadiaEntrance,
    atb: AmericaBeautifulPass,
):
    section = evaluator.add_parallel(
        id="pass_cost_optimization",
        desc="Cost-optimized pass purchase strategy and recommendation",
        parent=parent,
        critical=True,
    )

    # cost_comparison
    await _add_and_verify_leaf(
        evaluator,
        section,
        "cost_comparison",
        "Compares total cost of buying individual national park entrance passes vs buying America the Beautiful Annual Pass",
        claim=(
            "The answer compares the total cost of buying both national park entrance passes individually (e.g., $35 + $35 = about $70) "
            "against the $80 America the Beautiful Annual Pass."
        ),
        additional_instruction=(
            "Check only the answer content for a clear numeric comparison. Minor rounding and wording differences are acceptable."
        ),
    )

    # recommendation
    await _add_and_verify_leaf(
        evaluator,
        section,
        "recommendation",
        "Provides a clear recommendation based on the comparison",
        claim=(
            "The answer provides a clear recommendation on whether to purchase individual park passes or the America the Beautiful Annual Pass based on the cost comparison."
        ),
        additional_instruction="Check only the answer content. Either recommendation is acceptable if clearly justified.",
    )

    # reference_url_cost_optimization
    combined_sources = _merge_sources(
        (rmnp_fee.sources if rmnp_fee else []),
        (acadia_fee.sources if acadia_fee else []),
        (atb.sources if atb else []),
        (cost_opt.sources if cost_opt else []),
    )
    await _verify_with_reference_support(
        evaluator,
        section,
        "reference_url_cost_optimization",
        "Reference URL(s) supporting the costs used in the comparison (RMNP fee, Acadia fee, and/or ATB pass cost)",
        factual_claim="RMNP standard vehicle pass is $35, Acadia standard vehicle pass is $35, and the America the Beautiful Annual Pass costs $80.",
        sources=combined_sources,
        url_instruction="Verify each stated cost on official NPS/USGS/Recreation.gov pages.",
        fallback_when_no_url="If the answer provides no URLs supporting any of the costs, mark this incorrect.",
    )


async def build_distance_section(
    evaluator: Evaluator,
    parent,
    dist: DistanceInfo,
):
    section = evaluator.add_parallel(
        id="providence_to_acadia_distance",
        desc="Distance and drive time from Providence, RI to Acadia NP",
        parent=parent,
        critical=True,
    )

    # distance_miles
    await _add_and_verify_leaf(
        evaluator,
        section,
        "distance_miles",
        "Distance in miles (approximately 327 miles)",
        claim=(
            "The answer states that the driving distance from Providence, Rhode Island to Acadia National Park is approximately 327 miles."
        ),
        additional_instruction="Check only the answer content; accept reasonable approximations (e.g., 300–360 miles).",
    )

    # drive_time
    await _add_and_verify_leaf(
        evaluator,
        section,
        "drive_time",
        "Approximate drive time (5–6 hours)",
        claim="The answer states that the approximate drive time from Providence, Rhode Island to Acadia National Park is 5–6 hours.",
        additional_instruction="Check only the answer content; accept reasonable approximations (e.g., 4.5–7 hours).",
    )

    # reference_url_distance
    # Prefer verifying exactly what the answer said, if extracted
    miles_text = dist.miles if (dist and dist.miles) else "approximately 327 miles"
    hours_text = dist.hours if (dist and dist.hours) else "about 5–6 hours"
    await _verify_with_reference_support(
        evaluator,
        section,
        "reference_url_distance",
        "Reference URL supporting distance/drive-time estimate",
        factual_claim=f"The driving distance from Providence, RI to Acadia National Park is {miles_text} and the drive time is {hours_text}.",
        sources=dist.sources if dist else [],
        url_instruction=(
            "Use a reliable mapping source to support the approximate distance and driving time. Allow typical variations due to route and traffic."
        ),
        fallback_when_no_url="If the answer includes no URL supporting the distance/time, mark this incorrect.",
    )


async def build_recgov_section(
    evaluator: Evaluator,
    parent,
    recgov: RecGovBooking,
):
    section = evaluator.add_parallel(
        id="recreation_gov_booking",
        desc="General Recreation.gov camping booking window and release time",
        parent=parent,
        critical=True,
    )

    # booking_window
    await _add_and_verify_leaf(
        evaluator,
        section,
        "booking_window",
        "General booking window (up to 6 months in advance)",
        claim="The answer states that the general booking window on Recreation.gov is up to 6 months in advance.",
        additional_instruction=(
            "Check only the answer content. Accept wording that 'most campgrounds' or 'many sites' open 6 months in advance."
        ),
    )

    # release_time
    await _add_and_verify_leaf(
        evaluator,
        section,
        "release_time",
        "Standard release time (most reservations released at 10 AM ET)",
        claim="The answer states that most Recreation.gov reservations are released at 10 AM Eastern Time.",
        additional_instruction="Check only the answer content; allow 'ET' in place of 'Eastern Time'.",
    )

    # reference_url_recgov
    await _verify_with_reference_support(
        evaluator,
        section,
        "reference_url_recgov",
        "Reference URL supporting Recreation.gov booking window and release time",
        factual_claim="On Recreation.gov, most inventory opens up to 6 months in advance and is released at 10 AM ET.",
        sources=recgov.sources if recgov else [],
        url_instruction="Verify the general booking window and typical release time from official Recreation.gov support pages.",
        fallback_when_no_url="If the answer includes no supporting URL, mark this incorrect.",
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
    Evaluate an answer for the Denver → RMNP/Acadia July 2025 planning guide.
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

    # Extract all sections (can run concurrently)
    flight_task = evaluator.extract(
        prompt_extract_flight_info(), FlightInfo, extraction_name="flight_info"
    )
    rmnp_fee_task = evaluator.extract(
        prompt_extract_rmnp_entrance(), RMNPEntrance, extraction_name="rmnp_entrance"
    )
    rmnp_timed_task = evaluator.extract(
        prompt_extract_rmnp_timed_entry(), RMNPTimedEntry, extraction_name="rmnp_timed_entry"
    )
    rmnp_camping_task = evaluator.extract(
        prompt_extract_rmnp_camping(), RMNPCamping, extraction_name="rmnp_camping"
    )
    acadia_fee_task = evaluator.extract(
        prompt_extract_acadia_entrance(), AcadiaEntrance, extraction_name="acadia_entrance"
    )
    acadia_cad_task = evaluator.extract(
        prompt_extract_acadia_cadillac(), AcadiaCadillac, extraction_name="acadia_cadillac"
    )
    acadia_camping_task = evaluator.extract(
        prompt_extract_acadia_camping(), AcadiaCamping, extraction_name="acadia_camping"
    )
    co_state_parks_task = evaluator.extract(
        prompt_extract_co_state_parks(), COStateParks, extraction_name="co_state_parks"
    )
    atb_task = evaluator.extract(
        prompt_extract_atb_pass(), AmericaBeautifulPass, extraction_name="atb_pass"
    )
    cost_opt_task = evaluator.extract(
        prompt_extract_cost_optimization(), CostOptimization, extraction_name="cost_optimization"
    )
    distance_task = evaluator.extract(
        prompt_extract_distance(), DistanceInfo, extraction_name="distance_info"
    )
    recgov_task = evaluator.extract(
        prompt_extract_recgov(), RecGovBooking, extraction_name="recgov_info"
    )

    (
        flight_info,
        rmnp_fee,
        rmnp_timed,
        rmnp_camping,
        acadia_fee,
        acadia_cad,
        acadia_camping,
        co_state_parks,
        atb_pass,
        cost_opt,
        dist_info,
        recgov_info,
    ) = await asyncio.gather(
        flight_task,
        rmnp_fee_task,
        rmnp_timed_task,
        rmnp_camping_task,
        acadia_fee_task,
        acadia_cad_task,
        acadia_camping_task,
        co_state_parks_task,
        atb_task,
        cost_opt_task,
        distance_task,
        recgov_task,
    )

    # Build the top-level critical trip node
    trip_node = evaluator.add_parallel(
        id="trip_planning",
        desc="Complete planning requirements for a Denver to Rocky Mountain NP and Acadia NP trip via Breeze Airways",
        parent=root,
        critical=True,
    )

    # Build each critical section
    await build_breeze_airways_section(evaluator, trip_node, flight_info)
    await build_rmnp_entrance_section(evaluator, trip_node, rmnp_fee)
    await build_rmnp_timed_entry_section(evaluator, trip_node, rmnp_timed)
    await build_rmnp_camping_section(evaluator, trip_node, rmnp_camping)
    await build_acadia_entrance_section(evaluator, trip_node, acadia_fee)
    await build_acadia_cadillac_section(evaluator, trip_node, acadia_cad)
    await build_acadia_camping_section(evaluator, trip_node, acadia_camping)
    await build_colorado_state_parks_section(evaluator, trip_node, co_state_parks)
    await build_atb_section(evaluator, trip_node, atb_pass)
    await build_cost_optimization_section(evaluator, trip_node, cost_opt, rmnp_fee, acadia_fee, atb_pass)
    await build_distance_section(evaluator, trip_node, dist_info)
    await build_recgov_section(evaluator, trip_node, recgov_info)

    return evaluator.get_summary()