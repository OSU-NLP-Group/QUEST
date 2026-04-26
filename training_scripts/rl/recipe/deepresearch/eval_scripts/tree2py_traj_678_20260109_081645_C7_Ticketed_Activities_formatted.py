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
TASK_ID = "nfl_stadium_ticketing_2026"
TASK_DESCRIPTION = (
    "I'm planning to attend a major sporting event in 2026 and need complete ticketing information for "
    "one of the largest NFL stadiums in the United States. Identify an NFL stadium with a seating capacity of at least 80,000, "
    "and provide the following details: the stadium's official name, exact seating capacity, city and state location, which NFL team(s) "
    "play there, whether the box office is open on non-event days and its operating hours, what mobile ticketing platform is used, "
    "if on-site parking is available and its cost, whether luxury suites are offered and their typical capacity, if season tickets are available, "
    "what other types of events (besides NFL games) are hosted there, and which company is the official ticketing partner."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StadiumInfo(BaseModel):
    stadium_official_name: Optional[str] = None
    seating_capacity: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    nfl_teams: List[str] = Field(default_factory=list)

    # Sources for identity/overview facts (name, capacity, location, teams, official status, partner if applicable)
    identity_source_urls: List[str] = Field(default_factory=list)

    # Eligibility and status
    operational_in_2026: Optional[str] = None  # "yes" / "no" / "unknown"
    operational_source_urls: List[str] = Field(default_factory=list)

    # Box office
    box_office_exists: Optional[str] = None  # "yes" / "no" / "unknown"
    box_office_open_non_event_days: Optional[str] = None  # "yes" / "no" / "unknown"
    box_office_hours: Optional[str] = None
    box_office_source_urls: List[str] = Field(default_factory=list)

    # Mobile/digital ticketing
    mobile_ticketing_platform: Optional[str] = None
    mobile_ticketing_source_urls: List[str] = Field(default_factory=list)

    # Parking
    parking_available: Optional[str] = None  # "yes" / "no" / "unknown"
    parking_cost_info: Optional[str] = None  # Specific price/range or "varies by event"
    parking_source_urls: List[str] = Field(default_factory=list)

    # Luxury suites
    luxury_suites_available: Optional[str] = None  # "yes" / "no" / "unknown"
    luxury_suite_typical_capacity: Optional[str] = None  # e.g., "16–24", "20", etc.
    suites_source_urls: List[str] = Field(default_factory=list)

    # Season tickets
    season_tickets_available: Optional[str] = None  # "yes" / "no" / "unknown"
    season_tickets_source_urls: List[str] = Field(default_factory=list)

    # Other events
    other_events: List[str] = Field(default_factory=list)  # e.g., ["concerts", "college football", "soccer"]
    other_events_source_urls: List[str] = Field(default_factory=list)

    # Official ticketing partner
    official_ticketing_partner: Optional[str] = None
    ticketing_partner_source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stadium_info() -> str:
    return """
    You must extract complete ticketing and venue information for a single NFL stadium described in the answer.

    Selection rule:
    - Choose the first stadium mentioned that has a seating capacity of at least 80,000 as presented in the answer.
    - If multiple stadiums are mentioned, select the first one that meets the capacity threshold.
    - Extract only what is explicitly stated in the answer; do not invent or infer.

    For the selected stadium, extract the following fields exactly as stated (use strings as-is; allow ranges and qualifiers):
    1) stadium_official_name: The official name of the stadium.
    2) seating_capacity: The exact seating capacity string as given (e.g., "80,000", "82,500", "around 82k").
    3) city: City where the stadium is located.
    4) state: U.S. state abbreviation or full name.
    5) nfl_teams: List of NFL teams that play home games at the stadium.
    6) identity_source_urls: ALL URLs cited in the answer that support the official name, capacity, location, and team tenancy. Extract only valid URLs presented in the answer (plain or markdown).
    7) operational_in_2026: "yes", "no", or "unknown" depending on whether the answer claims the stadium is operational in 2026.
    8) operational_source_urls: URLs cited that indicate operational status / 2026 events/schedules.
    9) box_office_exists: "yes", "no", or "unknown" as stated.
    10) box_office_open_non_event_days: "yes", "no", or "unknown" as stated.
    11) box_office_hours: Operating days/hours string as stated (e.g., "Mon–Fri 10am–5pm; event days starting 3 hours prior").
    12) box_office_source_urls: URLs cited for box office information.
    13) mobile_ticketing_platform: The mobile/digital ticketing platform/app used (e.g., "Ticketmaster", "SeatGeek", "SafeTix").
    14) mobile_ticketing_source_urls: URLs cited for digital ticketing/mobile platform.
    15) parking_available: "yes", "no", or "unknown" as stated for on-site/dedicated event parking.
    16) parking_cost_info: A specific price, a price range, or an official statement like "pricing varies by event".
    17) parking_source_urls: URLs cited for parking availability and cost.
    18) luxury_suites_available: "yes", "no", or "unknown".
    19) luxury_suite_typical_capacity: Typical guest capacity or capacity range for suites.
    20) suites_source_urls: URLs cited for suites info.
    21) season_tickets_available: "yes", "no", or "unknown".
    22) season_tickets_source_urls: URLs cited for season tickets.
    23) other_events: List of other event types hosted (besides NFL games), as stated (e.g., ["concerts","college football","soccer"]).
    24) other_events_source_urls: URLs cited for other events.
    25) official_ticketing_partner: The official ticketing partner/company/platform name.
    26) ticketing_partner_source_urls: URLs cited for the official ticketing partner.

    IMPORTANT:
    - Extract only URLs explicitly present in the answer (plain or markdown). If no URLs are cited for a field, return an empty list for that field's URLs.
    - If a text field is not mentioned, return null for that field. If a list field is not mentioned, return an empty list.
    - Do not infer any values not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _join_list(items: List[str]) -> str:
    return ", ".join([s for s in items if s and s.strip()]) if items else ""


def _pick_sources(primary: List[str], fallback: Optional[List[str]] = None) -> List[str]:
    if primary and len(primary) > 0:
        return primary
    return fallback or []


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_for_stadium(
    evaluator: Evaluator,
    parent_node,
    info: StadiumInfo,
) -> None:
    """
    Build verification leaves under the critical parallel node for the stadium ticketing details.
    Each leaf represents one verification step or a single existence check.
    """

    # 1) Official NFL Stadium (verification against sources)
    node_official = evaluator.add_leaf(
        id="Official_NFL_Stadium",
        desc="The identified venue is an official NFL stadium (i.e., a recognized NFL home venue for at least one NFL team).",
        parent=parent_node,
        critical=True,
    )
    teams_list = _join_list(info.nfl_teams)
    stadium_name = info.stadium_official_name or ""
    claim_official = (
        f"The stadium '{stadium_name}' is an official NFL stadium and is a recognized home venue for these NFL team(s): {teams_list}."
    )
    await evaluator.verify(
        claim=claim_official,
        node=node_official,
        sources=_pick_sources(info.identity_source_urls),
        additional_instruction="Confirm the stadium is a recognized home venue for at least one NFL team using the provided sources. Minor naming variations are acceptable."
    )

    # 2) Stadium Official Name (existence check - 'provided')
    node_name_provided = evaluator.add_custom_node(
        result=bool(info.stadium_official_name and info.stadium_official_name.strip()),
        id="Stadium_Official_Name",
        desc="The stadium's official name is provided.",
        parent=parent_node,
        critical=True
    )

    # 3) Seating Capacity ≥ 80,000 (verification)
    node_capacity = evaluator.add_leaf(
        id="Seating_Capacity_At_Least_80000",
        desc="The exact seating capacity is stated and is at least 80,000.",
        parent=parent_node,
        critical=True,
    )
    capacity_str = info.seating_capacity or ""
    claim_capacity = (
        f"The stadium '{stadium_name}' has a seating capacity of {capacity_str}, and this capacity is at least 80,000."
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=node_capacity,
        sources=_pick_sources(info.identity_source_urls),
        additional_instruction="Verify the stated capacity from the sources and confirm it meets or exceeds 80,000. Allow reasonable rounding differences."
    )

    # 4) Location City, State, US (verification)
    node_location = evaluator.add_leaf(
        id="Location_City_State_US",
        desc="The stadium's city and U.S. state location are provided (confirming the stadium is located in the United States).",
        parent=parent_node,
        critical=True,
    )
    city = info.city or ""
    state = info.state or ""
    claim_location = (
        f"The stadium '{stadium_name}' is located in {city}, {state}, United States."
    )
    await evaluator.verify(
        claim=claim_location,
        node=node_location,
        sources=_pick_sources(info.identity_source_urls),
        additional_instruction="Confirm the stadium's city and state location in the USA from the provided sources."
    )

    # 5) NFL Team(s) (verification)
    node_teams = evaluator.add_leaf(
        id="NFL_Team_S",
        desc="The NFL team(s) that play at the stadium are identified.",
        parent=parent_node,
        critical=True,
    )
    claim_teams = (
        f"The NFL team(s) that play home games at '{stadium_name}' are: {teams_list}."
    )
    await evaluator.verify(
        claim=claim_teams,
        node=node_teams,
        sources=_pick_sources(info.identity_source_urls),
        additional_instruction="Verify the listed NFL team(s) tenancy using the provided sources."
    )

    # 6) Operational in 2026 (verification)
    node_operational = evaluator.add_leaf(
        id="Operational_In_2026",
        desc="Provides evidence the stadium is operational during 2026 (e.g., not permanently closed before/during 2026).",
        parent=parent_node,
        critical=True,
    )
    op_val = info.operational_in_2026 or "unknown"
    claim_operational = (
        f"The stadium '{stadium_name}' is operational in 2026 (status: {op_val})."
    )
    await evaluator.verify(
        claim=claim_operational,
        node=node_operational,
        sources=_pick_sources(info.operational_source_urls, info.identity_source_urls),
        additional_instruction="Use event calendars, schedules, or official pages indicating ongoing operations or 2026 events to confirm operational status."
    )

    # 7) Public Box Office: non-event days and hours (verification)
    node_box_office = evaluator.add_leaf(
        id="Public_Box_Office_Non_Event_Days_And_Hours",
        desc="Confirms there is a publicly accessible box office, indicates whether it is open on non-event days, and provides its operating days/hours (publicly available and verifiable).",
        parent=parent_node,
        critical=True,
    )
    exists_val = info.box_office_exists or "unknown"
    non_event_val = info.box_office_open_non_event_days or "unknown"
    hours_val = info.box_office_hours or ""
    claim_box = (
        f"There is a publicly accessible box office for '{stadium_name}' (exists: {exists_val}); "
        f"it is {non_event_val} open on non-event days; operating days/hours: {hours_val}."
    )
    await evaluator.verify(
        claim=claim_box,
        node=node_box_office,
        sources=_pick_sources(info.box_office_source_urls),
        additional_instruction="Check official 'box office' or 'ticket office' pages for stated hours and whether it operates on non-event days."
    )

    # 8) Mobile Ticketing Platform (verification)
    node_mobile = evaluator.add_leaf(
        id="Mobile_Ticketing_Platform",
        desc="Identifies the mobile/digital ticketing platform or app used by the stadium.",
        parent=parent_node,
        critical=True,
    )
    mobile_platform = info.mobile_ticketing_platform or ""
    claim_mobile = (
        f"The stadium '{stadium_name}' uses '{mobile_platform}' as its mobile/digital ticketing platform or app."
    )
    await evaluator.verify(
        claim=claim_mobile,
        node=node_mobile,
        sources=_pick_sources(info.mobile_ticketing_source_urls, info.ticketing_partner_source_urls),
        additional_instruction="Confirm the mobile/digital ticketing platform/app (e.g., Ticketmaster, SeatGeek, SafeTix) via official digital ticketing pages."
    )

    # 9) On-site Parking Available (verification)
    node_parking_available = evaluator.add_leaf(
        id="On_Site_Parking_Available",
        desc="Confirms on-site/dedicated event parking is available.",
        parent=parent_node,
        critical=True,
    )
    parking_avail = info.parking_available or "unknown"
    claim_parking_avail = (
        f"On-site or dedicated event parking is available for '{stadium_name}' (status: {parking_avail})."
    )
    await evaluator.verify(
        claim=claim_parking_avail,
        node=node_parking_available,
        sources=_pick_sources(info.parking_source_urls),
        additional_instruction="Use official stadium parking or event logistics pages to confirm the availability of on-site/dedicated parking."
    )

    # 10) Parking Cost Info (verification)
    node_parking_cost = evaluator.add_leaf(
        id="Parking_Cost_Info",
        desc="Provides parking cost information (a specific price, a price range, or an official statement that pricing varies by event), with a verifiable source.",
        parent=parent_node,
        critical=True,
    )
    parking_cost = info.parking_cost_info or ""
    claim_parking_cost = (
        f"Parking cost information for '{stadium_name}' is: {parking_cost}."
    )
    await evaluator.verify(
        claim=claim_parking_cost,
        node=node_parking_cost,
        sources=_pick_sources(info.parking_source_urls),
        additional_instruction="Verify the official parking cost details. Accept specific prices, ranges, or an official statement that pricing varies by event."
    )

    # 11) Luxury Suites Available (verification)
    node_suites_available = evaluator.add_leaf(
        id="Luxury_Suites_Available",
        desc="Confirms luxury suite options are offered.",
        parent=parent_node,
        critical=True,
    )
    suites_avail = info.luxury_suites_available or "unknown"
    claim_suites_avail = (
        f"Luxury suites are offered at '{stadium_name}' (status: {suites_avail})."
    )
    await evaluator.verify(
        claim=claim_suites_avail,
        node=node_suites_available,
        sources=_pick_sources(info.suites_source_urls),
        additional_instruction="Confirm the availability of luxury suites via official premium seating or suites pages."
    )

    # 12) Luxury Suite Typical Capacity (verification)
    node_suite_capacity = evaluator.add_leaf(
        id="Luxury_Suite_Typical_Capacity",
        desc="Provides the typical guest capacity (or capacity range) for luxury suites.",
        parent=parent_node,
        critical=True,
    )
    suite_cap = info.luxury_suite_typical_capacity or ""
    claim_suite_cap = (
        f"The typical luxury suite capacity at '{stadium_name}' is {suite_cap}."
    )
    await evaluator.verify(
        claim=claim_suite_cap,
        node=node_suite_capacity,
        sources=_pick_sources(info.suites_source_urls),
        additional_instruction="Verify typical suite capacities or capacity ranges from official suites/premium seating information."
    )

    # 13) Season Tickets Available (verification)
    node_season = evaluator.add_leaf(
        id="Season_Tickets_Available",
        desc="Confirms season ticket packages are available.",
        parent=parent_node,
        critical=True,
    )
    season_avail = info.season_tickets_available or "unknown"
    claim_season = (
        f"Season ticket packages are available for events at '{stadium_name}' or for the NFL team(s) that play there (status: {season_avail})."
    )
    await evaluator.verify(
        claim=claim_season,
        node=node_season,
        sources=_pick_sources(info.season_tickets_source_urls, info.identity_source_urls),
        additional_instruction="Confirm availability of season tickets via team/stadium official ticketing pages."
    )

    # 14) Non-NFL Events Hosted (verification)
    node_other_events = evaluator.add_leaf(
        id="Non_NFL_Events_Hosted",
        desc="Identifies other types of events hosted at the stadium besides NFL games.",
        parent=parent_node,
        critical=True,
    )
    other_events_list = _join_list(info.other_events)
    claim_other_events = (
        f"The stadium '{stadium_name}' hosts other types of events besides NFL games, including: {other_events_list}."
    )
    await evaluator.verify(
        claim=claim_other_events,
        node=node_other_events,
        sources=_pick_sources(info.other_events_source_urls, info.identity_source_urls),
        additional_instruction="Use events calendars or official venue booking pages to confirm non-NFL event types (e.g., concerts, soccer, college football)."
    )

    # 15) Official Ticketing Partner (verification)
    node_partner = evaluator.add_leaf(
        id="Official_Ticketing_Partner",
        desc="Identifies the official ticketing partner/company/platform.",
        parent=parent_node,
        critical=True,
    )
    partner_name = info.official_ticketing_partner or ""
    claim_partner = (
        f"The official ticketing partner/company/platform for '{stadium_name}' is '{partner_name}'."
    )
    await evaluator.verify(
        claim=claim_partner,
        node=node_partner,
        sources=_pick_sources(info.ticketing_partner_source_urls, info.identity_source_urls),
        additional_instruction="Confirm the official ticketing partner via official stadium or team ticketing pages."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the NFL stadium ticketing details task.
    """

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

    # Add the critical top-level node to respect the rubric's root critical constraint
    main_node = evaluator.add_parallel(
        id="NFL_Stadium_Ticketing_Details",
        desc="Comprehensive ticketing and venue information for an eligible (official, US-based) NFL stadium with seating capacity ≥ 80,000 and operational in 2026.",
        parent=root,
        critical=True
    )

    # Extract structured stadium information from the agent's answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_stadium_info(),
        template_class=StadiumInfo,
        extraction_name="stadium_ticketing_details",
    )

    # Add a brief custom info summary to help downstream inspection
    evaluator.add_custom_info(
        info={
            "stadium_official_name": extracted_info.stadium_official_name,
            "seating_capacity": extracted_info.seating_capacity,
            "city": extracted_info.city,
            "state": extracted_info.state,
            "nfl_teams": extracted_info.nfl_teams,
            "mobile_ticketing_platform": extracted_info.mobile_ticketing_platform,
            "official_ticketing_partner": extracted_info.official_ticketing_partner
        },
        info_type="extraction_summary",
        info_name="selected_stadium_summary"
    )

    # Build verification tree and run checks
    await build_verification_for_stadium(evaluator, main_node, extracted_info)

    # Return structured result summary
    return evaluator.get_summary()