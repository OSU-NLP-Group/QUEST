import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "family_outdoor_10day_itinerary"
TASK_DESCRIPTION = (
    "Plan a comprehensive 10-day outdoor recreation itinerary for a family of four (including two children, ages 8 and 11) traveling from Bangor, Maine, visiting three major outdoor destinations in North America. "
    "The itinerary must include:\n\n"
    "1. A theme park visit in Tennessee that features the longest roller coaster at that park, with all family members meeting the height requirement for that specific ride.\n\n"
    "2. A ski resort visit in British Columbia, Canada, that holds the record for being North America's largest ski resort by skiable terrain, where the family can experience the aerial lift system connecting the two mountains.\n\n"
    "3. A wilderness backpacking trip in a California national park requiring an overnight wilderness permit obtained through the federal recreation reservation system, with the permit reserved during the advance booking window (not the last-minute window).\n\n"
    "For transportation, all flights must connect through major airline hubs. Specifically:\n"
    "- Outbound from Bangor to the first destination must connect through an American Airlines hub\n"
    "- At least one flight segment must connect through a Delta Airlines hub\n\n"
    "Provide for each destination:\n"
    "- The specific venue/park name and its location (city/state or city/province)\n"
    "- The featured attraction or facility name that meets the specified criteria\n"
    "- Key specifications that verify the criteria are met (e.g., measurements, rankings, records)\n"
    "- Ground transportation options available from the nearest airport\n"
    "- For the wilderness destination: the permit reservation system name, advance booking timeline (how many weeks ahead), and per-person fee\n"
    "- Flight routing showing the hub connections for at least the outbound journey from Bangor"
)


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class GroundTransportInfo(BaseModel):
    nearest_airport: Optional[str] = None
    options: List[str] = Field(default_factory=list)


class ThemeParkSegment(BaseModel):
    park_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    featured_coaster_name: Optional[str] = None
    longest_claim: Optional[str] = None
    longest_specifications: List[str] = Field(default_factory=list)
    longest_sources: List[str] = Field(default_factory=list)
    height_requirement_text: Optional[str] = None
    height_requirement_inches: Optional[str] = None
    height_requirement_sources: List[str] = Field(default_factory=list)
    family_meets_height_statement: Optional[str] = None
    operating_2026_sources: List[str] = Field(default_factory=list)
    ground_transport: GroundTransportInfo = Field(default_factory=GroundTransportInfo)


class SkiResortSegment(BaseModel):
    resort_name: Optional[str] = None
    city: Optional[str] = None
    province: Optional[str] = None
    largest_claim: Optional[str] = None
    acreage_or_hectares: Optional[str] = None
    largest_sources: List[str] = Field(default_factory=list)
    two_mountains_named: List[str] = Field(default_factory=list)
    connecting_lift_name: Optional[str] = None
    enclosed_gondola_statement: Optional[str] = None
    gondola_sources: List[str] = Field(default_factory=list)
    operating_2026_sources: List[str] = Field(default_factory=list)
    ground_transport: GroundTransportInfo = Field(default_factory=GroundTransportInfo)


class WildernessSegment(BaseModel):
    park_name: Optional[str] = None
    location_city_or_region: Optional[str] = None
    state: Optional[str] = None
    trip_name: Optional[str] = None
    year_round_permit_statement: Optional[str] = None
    permit_sources: List[str] = Field(default_factory=list)
    recreation_gov_named: Optional[str] = None
    recreation_gov_sources: List[str] = Field(default_factory=list)
    advance_window_statement: Optional[str] = None
    advance_window_weeks: Optional[str] = None
    advance_window_sources: List[str] = Field(default_factory=list)
    last_minute_window_avoided_statement: Optional[str] = None
    fee_details_text: Optional[str] = None
    per_person_fee_amount: Optional[str] = None
    fee_sources: List[str] = Field(default_factory=list)
    operating_2026_sources: List[str] = Field(default_factory=list)
    ground_transport: GroundTransportInfo = Field(default_factory=GroundTransportInfo)


class FlightRouting(BaseModel):
    first_destination_name: Optional[str] = None
    outbound_segments: List[str] = Field(default_factory=list)  # e.g., "BGR -> CLT -> TYS"
    outbound_hubs: List[str] = Field(default_factory=list)      # hub airport codes used on outbound
    aa_hub_used: Optional[str] = None                           # e.g., CLT, PHL, DFW, ORD, MIA, etc.
    aa_hub_sources: List[str] = Field(default_factory=list)
    aa_nonstop_claim_text: Optional[str] = None                 # "AA operates nonstop BGR–CLT"
    aa_nonstop_sources: List[str] = Field(default_factory=list)
    delta_hub_used: Optional[str] = None                        # e.g., ATL, DTW, MSP, SLC, JFK, LGA
    delta_hub_sources: List[str] = Field(default_factory=list)
    all_hub_airports: List[str] = Field(default_factory=list)   # combined list of all connection airports
    all_hub_sources: List[str] = Field(default_factory=list)


class ItineraryCore(BaseModel):
    day_labels: List[str] = Field(default_factory=list)  # e.g., ["Day 1", ..., "Day 10"]
    has_day_by_day: Optional[bool] = None
    duration_text: Optional[str] = None                  # e.g., "10 days", "Day 1–Day 10"


# --------------------------------------------------------------------------- #
# Extraction Prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_itinerary_core() -> str:
    return (
        "Extract the itinerary structure specifics from the answer.\n"
        "Return:\n"
        "- day_labels: an array of strings that identify each day labels present (e.g., 'Day 1', 'Day 2', ..., 'Day 10'). Include exactly those present in the answer.\n"
        "- has_day_by_day: a boolean, true if the answer presents a clear day-by-day plan (each day has specific travel/activity content), false otherwise.\n"
        "- duration_text: any explicit duration phrasing (e.g., '10 days', 'Day 1–Day 10').\n"
        "If any field is missing, set to null or empty list as appropriate."
    )


def prompt_extract_tennessee_theme_park() -> str:
    return (
        "Extract the Tennessee theme park visit details from the answer.\n"
        "Return fields:\n"
        "- park_name\n"
        "- city\n"
        "- state\n"
        "- featured_coaster_name\n"
        "- longest_claim: text asserting the featured coaster is the park's longest\n"
        "- longest_specifications: list of measurements/specs (e.g., length in feet/meters) used to support 'longest at the park'\n"
        "- longest_sources: list of URLs cited that support the 'longest coaster at the park' claim\n"
        "- height_requirement_text: text of minimum height requirement\n"
        "- height_requirement_inches: the minimum height (in inches) if provided (as text, e.g., '39')\n"
        "- height_requirement_sources: URLs that support the height requirement\n"
        "- family_meets_height_statement: text in the answer explicitly asserting all family members meet the height requirement\n"
        "- operating_2026_sources: URLs that indicate the park is publicly accessible and operating/open in 2026\n"
        "- ground_transport: { nearest_airport, options: array of ground transportation options (rideshare, taxi, shuttle, rental car, etc.) }\n"
        "For all URL fields, only extract actual URLs explicitly present in the answer."
    )


def prompt_extract_bc_ski_resort() -> str:
    return (
        "Extract the British Columbia ski resort visit details from the answer.\n"
        "Return fields:\n"
        "- resort_name\n"
        "- city\n"
        "- province\n"
        "- largest_claim: text asserting the resort is North America's largest by skiable terrain\n"
        "- acreage_or_hectares: area figure (as text) referenced for skiable terrain\n"
        "- largest_sources: URLs supporting the 'largest by skiable terrain' claim\n"
        "- two_mountains_named: array of the two mountain names at the resort\n"
        "- connecting_lift_name: name of the aerial lift system connecting the two mountains\n"
        "- enclosed_gondola_statement: text asserting the connecting lift is an enclosed gondola\n"
        "- gondola_sources: URLs supporting the enclosed gondola connection between the two mountains\n"
        "- operating_2026_sources: URLs that indicate the resort is publicly accessible and operating/open in 2026\n"
        "- ground_transport: { nearest_airport, options: array of ground transportation options }\n"
        "Extract only URLs explicitly present in the answer."
    )


def prompt_extract_california_wilderness() -> str:
    return (
        "Extract the California national park wilderness backpacking trip details from the answer.\n"
        "Return fields:\n"
        "- park_name\n"
        "- location_city_or_region\n"
        "- state\n"
        "- trip_name: the specific wilderness route/area planned\n"
        "- year_round_permit_statement: text asserting overnight wilderness/backcountry travel requires a permit year-round\n"
        "- permit_sources: URLs supporting permit requirement statements\n"
        "- recreation_gov_named: text stating that the permit is reservable via Recreation.gov\n"
        "- recreation_gov_sources: URLs supporting Recreation.gov usage\n"
        "- advance_window_statement: text asserting permit reserved during the advance booking window (not last-minute)\n"
        "- advance_window_weeks: advance booking window length (as text, e.g., '24 weeks')\n"
        "- advance_window_sources: URLs supporting the advance window policy\n"
        "- last_minute_window_avoided_statement: text asserting last-minute window is not used\n"
        "- fee_details_text: text describing permit fees\n"
        "- per_person_fee_amount: the per-person fee amount as text (e.g., '$5 per person per night')\n"
        "- fee_sources: URLs supporting fee details\n"
        "- operating_2026_sources: URLs that indicate the park is publicly accessible and operating/open in 2026\n"
        "- ground_transport: { nearest_airport, options: array of ground transportation options }\n"
        "Extract only URLs explicitly present in the answer."
    )


def prompt_extract_flight_routing() -> str:
    return (
        "Extract the flight routing and hub constraints from the answer.\n"
        "Return fields:\n"
        "- first_destination_name: name of the first destination visited after departing Bangor\n"
        "- outbound_segments: array describing the outbound routing (e.g., 'BGR -> CLT -> TYS')\n"
        "- outbound_hubs: array of hub airport codes used on the outbound journey\n"
        "- aa_hub_used: the American Airlines hub airport code used on the outbound connection (e.g., CLT, DFW, ORD, PHL, MIA)\n"
        "- aa_hub_sources: URLs supporting that this airport is an AA hub\n"
        "- aa_nonstop_claim_text: text asserting AA operates nonstop service from Bangor (BGR) to the AA hub used\n"
        "- aa_nonstop_sources: URLs supporting the nonstop claim\n"
        "- delta_hub_used: a Delta Air Lines hub airport code used somewhere in the itinerary\n"
        "- delta_hub_sources: URLs supporting that this airport is a Delta hub\n"
        "- all_hub_airports: array of all connection airports mentioned in the routing that are claimed as major hubs\n"
        "- all_hub_sources: URLs supporting that each listed airport is a major airline hub\n"
        "Extract only URLs explicitly present in the answer."
    )


# --------------------------------------------------------------------------- #
# Verification Subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_tennessee_theme_park(evaluator: Evaluator, parent_node, tp: ThemeParkSegment) -> None:
    node = evaluator.add_parallel(
        id="tennessee_theme_park",
        desc="Includes a Tennessee theme park visit featuring the park’s longest roller coaster and satisfying the coaster height requirement constraints",
        parent=parent_node,
        critical=True
    )

    # Park name and Tennessee location provided
    evaluator.add_custom_node(
        result=(bool(tp.park_name) and bool(tp.city) and bool(tp.state) and ("tennessee" in tp.state.lower())),
        id="park_name_and_location",
        desc="Provides the theme park name and its city/state location, and the state is Tennessee",
        parent=node,
        critical=True
    )

    # Featured coaster named
    evaluator.add_custom_node(
        result=bool(tp.featured_coaster_name),
        id="featured_coaster_name",
        desc="Names the specific roller coaster used to satisfy the requirement",
        parent=node,
        critical=True
    )

    # Existence of sources for longest claim (precondition)
    evaluator.add_custom_node(
        result=bool(tp.longest_sources),
        id="tenn_longest_sources_provided",
        desc="Sources are provided to support 'longest coaster at the park' claim",
        parent=node,
        critical=True
    )

    # Longest coaster evidence from sources
    longest_leaf = evaluator.add_leaf(
        id="longest_coaster_evidence",
        desc="Provides verifiable specification(s)/claim showing the named coaster is the longest roller coaster at that specific park",
        parent=node,
        critical=True
    )
    longest_claim = f"The roller coaster '{tp.featured_coaster_name}' is the longest roller coaster at {tp.park_name}."
    await evaluator.verify(
        claim=longest_claim,
        node=longest_leaf,
        sources=tp.longest_sources,
        additional_instruction="Confirm the claim using the provided URLs; allow reasonable phrasing variants. Look for explicit 'longest at the park' statements or tables with lengths."
    )

    # Height requirement sources provided (precondition)
    evaluator.add_custom_node(
        result=bool(tp.height_requirement_sources),
        id="tenn_height_sources_provided",
        desc="Sources are provided to support the coaster's height requirement",
        parent=node,
        critical=True
    )

    # Height requirement <= 39 inches
    height_leaf = evaluator.add_leaf(
        id="height_requirement_max_39",
        desc="States the coaster’s minimum height requirement and it is ≤ 39 inches",
        parent=node,
        critical=True
    )
    height_claim = f"The minimum height requirement for '{tp.featured_coaster_name}' is less than or equal to 39 inches."
    await evaluator.verify(
        claim=height_claim,
        node=height_leaf,
        sources=tp.height_requirement_sources,
        additional_instruction="Focus on the minimum height requirement. Consider ≤39 inches as satisfying the constraint; minor formatting differences are acceptable."
    )

    # Family meets height requirement (answer states explicitly)
    family_leaf = evaluator.add_leaf(
        id="all_family_meets_height_requirement",
        desc="Explicitly states that all family members (including both children) meet the minimum height requirement for the featured coaster",
        parent=node,
        critical=True
    )
    family_claim = "The answer explicitly states that all family members, including both children ages 8 and 11, meet the minimum height requirement for the featured coaster."
    await evaluator.verify(
        claim=family_claim,
        node=family_leaf,
        additional_instruction="Verify this statement based on the answer text only."
    )

    # Ground transportation options from nearest airport
    evaluator.add_custom_node(
        result=bool(tp.ground_transport and tp.ground_transport.options),
        id="ground_transportation_from_nearest_airport",
        desc="Lists at least one ground transportation option from the nearest airport to the destination (rideshare, taxi, shuttle, and/or rental car)",
        parent=node,
        critical=True
    )


async def verify_bc_ski_resort(evaluator: Evaluator, parent_node, sr: SkiResortSegment) -> None:
    node = evaluator.add_parallel(
        id="british_columbia_ski_resort",
        desc="Includes a British Columbia ski resort visit that is North America’s largest by skiable terrain and uses an enclosed gondola connecting two mountains",
        parent=parent_node,
        critical=True
    )

    # Resort name and BC location provided
    evaluator.add_custom_node(
        result=(bool(sr.resort_name) and bool(sr.city) and bool(sr.province) and ("british columbia" in sr.province.lower())),
        id="resort_name_and_location",
        desc="Provides the ski resort name and its city/province location, and the province is British Columbia, Canada",
        parent=node,
        critical=True
    )

    # Sources for largest claim provided (precondition)
    evaluator.add_custom_node(
        result=bool(sr.largest_sources),
        id="bc_largest_sources_provided",
        desc="Sources are provided to support 'largest by skiable terrain' claim",
        parent=node,
        critical=True
    )

    # Largest by skiable terrain evidence
    largest_leaf = evaluator.add_leaf(
        id="largest_by_skiable_terrain_evidence",
        desc="Provides verifiable specification(s) supporting the claim that the resort is North America’s largest ski resort by skiable terrain (e.g., acreage/hectares plus the record/ranking claim)",
        parent=node,
        critical=True
    )
    largest_claim = f"{sr.resort_name} is North America's largest ski resort by skiable terrain."
    await evaluator.verify(
        claim=largest_claim,
        node=largest_leaf,
        sources=sr.largest_sources,
        additional_instruction="Confirm the record using the provided URLs; look for explicit statements like 'largest in North America' and acreage figures."
    )

    # Names of two mountains and connecting lift provided (existence)
    evaluator.add_custom_node(
        result=(len(sr.two_mountains_named) >= 2 and bool(sr.connecting_lift_name)),
        id="two_mountains_and_lift_named",
        desc="States the resort consists of two separate mountains and names the aerial lift system that connects them",
        parent=node,
        critical=True
    )

    # Sources for gondola provided (precondition)
    evaluator.add_custom_node(
        result=bool(sr.gondola_sources),
        id="bc_gondola_sources_provided",
        desc="Sources are provided to support the enclosed gondola statement",
        parent=node,
        critical=True
    )

    # Enclosed gondola confirmed
    gondola_leaf = evaluator.add_leaf(
        id="enclosed_gondola_confirmed",
        desc="Explicitly states that the connecting lift system is an enclosed gondola system",
        parent=node,
        critical=True
    )
    gondola_claim = f"The connecting lift system '{sr.connecting_lift_name}' is an enclosed gondola that connects the two mountains."
    await evaluator.verify(
        claim=gondola_claim,
        node=gondola_leaf,
        sources=sr.gondola_sources,
        additional_instruction="Confirm that the named lift is an enclosed gondola connecting the two mountains; wording variants are acceptable."
    )

    # Ground transportation options from nearest airport
    evaluator.add_custom_node(
        result=bool(sr.ground_transport and sr.ground_transport.options),
        id="ground_transportation_from_nearest_airport",
        desc="Lists at least one ground transportation option from the nearest airport to the destination (rideshare, taxi, shuttle, and/or rental car)",
        parent=node,
        critical=True
    )


async def verify_california_wilderness(evaluator: Evaluator, parent_node, wl: WildernessSegment) -> None:
    node = evaluator.add_parallel(
        id="california_wilderness_destination",
        desc="Includes a California national park wilderness backpacking trip requiring year-round overnight permits reserved via Recreation.gov during an advance window, with required fee details",
        parent=parent_node,
        critical=True
    )

    # Park name and California location
    park_loc_leaf = evaluator.add_leaf(
        id="park_name_and_location",
        desc="Provides the national park name and its location, and the park is in California",
        parent=node,
        critical=True
    )
    park_loc_claim = f"The national park '{wl.park_name}' is located in California."
    # Use any available permit or recreation.gov sources for park location; if none, simple verify based on answer
    park_loc_sources = wl.permit_sources or wl.recreation_gov_sources or []
    await evaluator.verify(
        claim=park_loc_claim,
        node=park_loc_leaf,
        sources=park_loc_sources if park_loc_sources else None,
        additional_instruction="If sources are provided, confirm the park's location is California; otherwise judge based on the answer text."
    )

    # Backpacking trip named (existence)
    evaluator.add_custom_node(
        result=bool(wl.trip_name),
        id="backpacking_trip_named",
        desc="Names the specific wilderness backpacking trip/route/area planned within the park (i.e., the featured wilderness activity for this destination)",
        parent=node,
        critical=True
    )

    # Permit sources provided (precondition)
    evaluator.add_custom_node(
        result=bool(wl.permit_sources),
        id="permit_sources_provided",
        desc="Sources provided for permit requirement",
        parent=node,
        critical=True
    )

    # Year-round overnight permit required
    permit_req_leaf = evaluator.add_leaf(
        id="year_round_overnight_permit_required",
        desc="States that overnight wilderness/backcountry travel requires a wilderness permit year-round at this park",
        parent=node,
        critical=True
    )
    permit_req_claim = f"Overnight wilderness/backcountry travel at {wl.park_name} requires a wilderness permit year-round."
    await evaluator.verify(
        claim=permit_req_claim,
        node=permit_req_leaf,
        sources=wl.permit_sources,
        additional_instruction="Confirm the year-round overnight permit requirement using the official sources provided."
    )

    # Recreation.gov sources provided (precondition)
    evaluator.add_custom_node(
        result=bool(wl.recreation_gov_sources),
        id="recreation_gov_sources_provided",
        desc="Sources provided for Recreation.gov reservation system usage",
        parent=node,
        critical=True
    )

    # Recreation.gov used and named
    rgov_leaf = evaluator.add_leaf(
        id="recreation_gov_used_and_named",
        desc="States that the permit is reservable through Recreation.gov and names that system/platform",
        parent=node,
        critical=True
    )
    rgov_claim = f"The wilderness permit at {wl.park_name} is reservable through Recreation.gov."
    await evaluator.verify(
        claim=rgov_claim,
        node=rgov_leaf,
        sources=wl.recreation_gov_sources,
        additional_instruction="Confirm the reservation platform is Recreation.gov using the provided URLs."
    )

    # Advance window used (answer statement only)
    adv_used_leaf = evaluator.add_leaf(
        id="advance_booking_window_used_not_last_minute",
        desc="States the permit is reserved during the advance booking window (not the last-minute window)",
        parent=node,
        critical=True
    )
    adv_used_claim = "The answer explicitly states the permit is reserved during the advance booking window and not the last-minute window."
    await evaluator.verify(
        claim=adv_used_claim,
        node=adv_used_leaf,
        additional_instruction="Judge based on the answer text only."
    )

    # Advance window sources provided (precondition)
    evaluator.add_custom_node(
        result=bool(wl.advance_window_sources),
        id="advance_window_sources_provided",
        desc="Sources provided for advance booking window timeline",
        parent=node,
        critical=True
    )

    # Advance window at least 20 weeks
    adv_weeks_leaf = evaluator.add_leaf(
        id="advance_booking_window_at_least_20_weeks",
        desc="Specifies an advance booking timeline of at least 20 weeks before the trip date (in weeks or an equivalent)",
        parent=node,
        critical=True
    )
    adv_weeks_claim = f"The advance booking window for wilderness permits at {wl.park_name} opens at least 20 weeks before the trip date."
    await evaluator.verify(
        claim=adv_weeks_claim,
        node=adv_weeks_leaf,
        sources=wl.advance_window_sources,
        additional_instruction="Confirm the policy indicates ≥20 weeks (e.g., 24 weeks). Minor phrasing variations are acceptable."
    )

    # Fee sources provided (precondition)
    evaluator.add_custom_node(
        result=bool(wl.fee_sources),
        id="fee_sources_provided",
        desc="Sources provided for permit fee details",
        parent=node,
        critical=True
    )

    # Permit fees include per-person fee
    fee_leaf = evaluator.add_leaf(
        id="permit_fees_include_per_person_fee",
        desc="Provides permit fee details including a per-person fee in addition to any base/transaction fee (with amounts given)",
        parent=node,
        critical=True
    )
    fee_claim = f"The wilderness permit fee structure at {wl.park_name} includes a per-person fee (e.g., {wl.per_person_fee_amount}) in addition to any base or transaction fee."
    await evaluator.verify(
        claim=fee_claim,
        node=fee_leaf,
        sources=wl.fee_sources,
        additional_instruction="Confirm that a per-person fee exists in addition to base/transaction fees. Variants like 'per person per night' are acceptable."
    )

    # Ground transportation options from nearest airport
    evaluator.add_custom_node(
        result=bool(wl.ground_transport and wl.ground_transport.options),
        id="ground_transportation_from_nearest_airport",
        desc="Lists at least one ground transportation option from the nearest airport to the destination (rideshare, taxi, shuttle, and/or rental car)",
        parent=node,
        critical=True
    )


async def verify_flight_routing(evaluator: Evaluator, parent_node, fr: FlightRouting) -> None:
    node = evaluator.add_parallel(
        id="flight_routing",
        desc="Provides flight routing from Bangor that satisfies the hub-connection constraints and shows at least the outbound routing",
        parent=parent_node,
        critical=True
    )

    # Outbound routing shown
    evaluator.add_custom_node(
        result=(bool(fr.outbound_segments) and any(("BGR" in s or "Bangor" in s) for s in fr.outbound_segments)),
        id="outbound_routing_shown",
        desc="Shows the outbound routing from Bangor to the first destination including the connection hub airport(s)",
        parent=node,
        critical=True
    )

    # AA hub sources provided (precondition)
    evaluator.add_custom_node(
        result=bool(fr.aa_hub_used) and bool(fr.aa_hub_sources),
        id="aa_hub_sources_provided",
        desc="Sources provided confirming AA hub classification",
        parent=node,
        critical=True
    )

    # Outbound via American Airlines hub
    aa_hub_leaf = evaluator.add_leaf(
        id="outbound_via_american_hub",
        desc="Outbound routing from Bangor to the first destination connects through an American Airlines hub (hub airport is identified as an AA hub)",
        parent=node,
        critical=True
    )
    aa_hub_claim = f"The outbound routing connects through {fr.aa_hub_used}, which is an American Airlines hub."
    await evaluator.verify(
        claim=aa_hub_claim,
        node=aa_hub_leaf,
        sources=fr.aa_hub_sources,
        additional_instruction="Confirm the airport is an AA hub using the provided URLs (official airline pages or authoritative references)."
    )

    # AA nonstop sources provided (precondition)
    evaluator.add_custom_node(
        result=bool(fr.aa_nonstop_sources),
        id="aa_nonstop_sources_provided",
        desc="Sources provided for AA nonstop from Bangor to the AA hub used",
        parent=node,
        critical=True
    )

    # AA operates nonstop from BGR to hub
    aa_nonstop_leaf = evaluator.add_leaf(
        id="aa_nonstop_bgr_to_hub_claim",
        desc="States that American Airlines operates nonstop service from Bangor (BGR) to at least one AA hub used for the itinerary’s outbound connection",
        parent=node,
        critical=True
    )
    aa_nonstop_claim = f"American Airlines operates nonstop service from Bangor (BGR) to {fr.aa_hub_used}."
    await evaluator.verify(
        claim=aa_nonstop_claim,
        node=aa_nonstop_leaf,
        sources=fr.aa_nonstop_sources,
        additional_instruction="Confirm nonstop service exists (seasonal or year-round) between BGR and the specified AA hub."
    )

    # Delta hub sources provided (precondition)
    evaluator.add_custom_node(
        result=bool(fr.delta_hub_used) and bool(fr.delta_hub_sources),
        id="delta_hub_sources_provided",
        desc="Sources provided confirming Delta hub classification",
        parent=node,
        critical=True
    )

    # At least one Delta hub connection
    dl_hub_leaf = evaluator.add_leaf(
        id="at_least_one_delta_hub_connection",
        desc="Includes at least one flight segment in the itinerary that connects through a Delta Air Lines hub (hub airport is identified as a Delta hub)",
        parent=node,
        critical=True
    )
    dl_hub_claim = f"The itinerary includes a flight segment connecting through {fr.delta_hub_used}, which is a Delta Air Lines hub."
    await evaluator.verify(
        claim=dl_hub_claim,
        node=dl_hub_leaf,
        sources=fr.delta_hub_sources,
        additional_instruction="Confirm the airport is a Delta hub using the provided URLs."
    )

    # All hubs sources provided (precondition)
    evaluator.add_custom_node(
        result=bool(fr.all_hub_airports) and bool(fr.all_hub_sources),
        id="all_hub_sources_provided",
        desc="Sources provided confirming all connection airports are major airline hubs",
        parent=node,
        critical=True
    )

    # All flights use major hubs
    all_hubs_leaf = evaluator.add_leaf(
        id="all_flights_use_major_hubs",
        desc="All described flight connections are through major airline hub airports (the hubs are identified in the routing)",
        parent=node,
        critical=True
    )
    all_hubs_claim = (
        f"All described flight connections are through major airline hub airports: {', '.join(fr.all_hub_airports)}."
    )
    await evaluator.verify(
        claim=all_hubs_claim,
        node=all_hubs_leaf,
        sources=fr.all_hub_sources,
        additional_instruction="Confirm hub status for each listed airport using the provided URLs."
    )


async def verify_venues_operating_2026(evaluator: Evaluator, parent_node, tp: ThemeParkSegment, sr: SkiResortSegment, wl: WildernessSegment) -> None:
    venues_node = evaluator.add_parallel(
        id="venues_operating_2026",
        desc="For each of the three destinations, indicates the venue is publicly accessible and operating/open in 2026 (not permanently closed)",
        parent=parent_node,
        critical=True
    )

    # Tennessee theme park open 2026
    tenn_open_leaf = evaluator.add_leaf(
        id="tennessee_venue_open_2026",
        desc="Tennessee theme park is operating/open in 2026",
        parent=venues_node,
        critical=True
    )
    tenn_open_claim = f"As of 2026, {tp.park_name} is publicly accessible and operating."
    await evaluator.verify(
        claim=tenn_open_claim,
        node=tenn_open_leaf,
        sources=tp.operating_2026_sources if tp.operating_2026_sources else None,
        additional_instruction="Use the provided URLs to confirm the park is open/operating in 2026. If sources are missing, judge based on answer text."
    )

    # BC ski resort open 2026
    bc_open_leaf = evaluator.add_leaf(
        id="bc_resort_open_2026",
        desc="British Columbia ski resort is operating/open in 2026",
        parent=venues_node,
        critical=True
    )
    bc_open_claim = f"As of 2026, {sr.resort_name} is publicly accessible and operating."
    await evaluator.verify(
        claim=bc_open_claim,
        node=bc_open_leaf,
        sources=sr.operating_2026_sources if sr.operating_2026_sources else None,
        additional_instruction="Use the provided URLs to confirm the resort is open/operating in 2026. If sources are missing, judge based on answer text."
    )

    # California national park open 2026
    ca_open_leaf = evaluator.add_leaf(
        id="ca_park_open_2026",
        desc="California national park is operating/open in 2026",
        parent=venues_node,
        critical=True
    )
    ca_open_claim = f"As of 2026, {wl.park_name} is publicly accessible and operating."
    await evaluator.verify(
        claim=ca_open_claim,
        node=ca_open_leaf,
        sources=wl.operating_2026_sources if wl.operating_2026_sources else None,
        additional_instruction="Use the provided URLs to confirm the park is open/operating in 2026. If sources are missing, judge based on answer text."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate the provided itinerary answer against the rubric using the obj_task_eval framework.
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
        default_model=model
    )

    # Parallel extractions
    core_extraction_task = evaluator.extract(
        prompt=prompt_extract_itinerary_core(),
        template_class=ItineraryCore,
        extraction_name="itinerary_core"
    )
    tennessee_task = evaluator.extract(
        prompt=prompt_extract_tennessee_theme_park(),
        template_class=ThemeParkSegment,
        extraction_name="tennessee_theme_park"
    )
    bc_task = evaluator.extract(
        prompt=prompt_extract_bc_ski_resort(),
        template_class=SkiResortSegment,
        extraction_name="bc_ski_resort"
    )
    ca_task = evaluator.extract(
        prompt=prompt_extract_california_wilderness(),
        template_class=WildernessSegment,
        extraction_name="california_wilderness"
    )
    flight_task = evaluator.extract(
        prompt=prompt_extract_flight_routing(),
        template_class=FlightRouting,
        extraction_name="flight_routing"
    )

    core, tp, sr, wl, fr = await asyncio.gather(core_extraction_task, tennessee_task, bc_task, ca_task, flight_task)

    # Build overall critical node to align with rubric root criticality
    overall = evaluator.add_parallel(
        id="overall",
        desc="Produces a 10-day outdoor recreation itinerary for a family of four starting from Bangor that satisfies all destination/activity/permit/flight constraints and required per-destination details",
        parent=root,
        critical=True
    )

    # Top-level itinerary checks
    # 1) Duration exactly 10 days
    duration_is_10 = (len(core.day_labels) == 10) or (core.duration_text is not None and "10" in core.duration_text)
    evaluator.add_custom_node(
        result=duration_is_10,
        id="itinerary_duration_exactly_10_days",
        desc="Itinerary explicitly covers exactly 10 days (e.g., Day 1–Day 10)",
        parent=overall,
        critical=True
    )

    # 2) Day-by-day plan clearly provided
    day_by_day_leaf = evaluator.add_leaf(
        id="itinerary_day_by_day_plan",
        desc="Provides a clear plan for each day (travel and/or activities) rather than only high-level bullets",
        parent=overall,
        critical=True
    )
    day_by_day_claim = "The answer provides a clear day-by-day plan covering each of the 10 days (with specific travel and/or activity details)."
    await evaluator.verify(
        claim=day_by_day_claim,
        node=day_by_day_leaf,
        additional_instruction="Judge based on the answer text only; verify that each day has specific content rather than just high-level bullets."
    )

    # 3) Family party specified
    family_party_leaf = evaluator.add_leaf(
        id="family_party_specified",
        desc="Explicitly states the travelers are a family of four including two children ages 8 and 11",
        parent=overall,
        critical=True
    )
    family_party_claim = "The answer explicitly states the travelers are a family of four including two children ages 8 and 11."
    await evaluator.verify(
        claim=family_party_claim,
        node=family_party_leaf,
        additional_instruction="Judge based on the answer text only."
    )

    # 4) Venues operating in 2026
    await verify_venues_operating_2026(evaluator, overall, tp, sr, wl)

    # Sub-destinations
    await verify_tennessee_theme_park(evaluator, overall, tp)
    await verify_bc_ski_resort(evaluator, overall, sr)
    await verify_california_wilderness(evaluator, overall, wl)

    # Flight routing subtree
    await verify_flight_routing(evaluator, overall, fr)

    # Return summary
    return evaluator.get_summary()