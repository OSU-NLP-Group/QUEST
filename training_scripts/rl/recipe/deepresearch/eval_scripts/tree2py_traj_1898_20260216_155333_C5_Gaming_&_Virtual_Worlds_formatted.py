import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "rlcs_2026_france_major"
TASK_DESCRIPTION = """A competitive Rocket League esports organization is planning their 2026 European tournament schedule and needs detailed information about the major RLCS event taking place in France during the summer months.

Identify the RLCS major tournament held in Europe between May and August 2026, and provide the following information:
1. The official tournament name and exact dates
2. The venue name, specific city/location, and seating capacity for sporting events
3. Tournament format details including number of teams, game mode, and competition structure
4. Prize pool information including total amount, first place prize, and second place prize

All information must be supported by reference URLs from official sources.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TournamentIdentity(BaseModel):
    tournament_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    public_start_date: Optional[str] = None
    public_end_date: Optional[str] = None
    identity_urls: List[str] = Field(default_factory=list)
    classification_urls: List[str] = Field(default_factory=list)


class VenueInfo(BaseModel):
    venue_name: Optional[str] = None
    city_or_area: Optional[str] = None
    country: Optional[str] = None
    sporting_capacity: Optional[str] = None
    venue_urls: List[str] = Field(default_factory=list)


class FormatInfo(BaseModel):
    team_count: Optional[str] = None
    game_mode: Optional[str] = None
    group_stage_format: Optional[str] = None
    group_stage_start: Optional[str] = None
    group_stage_end: Optional[str] = None
    playoff_format: Optional[str] = None
    playoff_start: Optional[str] = None
    playoff_end: Optional[str] = None
    format_urls: List[str] = Field(default_factory=list)


class PrizeInfo(BaseModel):
    total_prize: Optional[str] = None
    first_prize: Optional[str] = None
    second_prize: Optional[str] = None
    prize_urls: List[str] = Field(default_factory=list)


class RLCSReportExtraction(BaseModel):
    tournament: Optional[TournamentIdentity] = None
    venue: Optional[VenueInfo] = None
    format: Optional[FormatInfo] = None
    prize: Optional[PrizeInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_rlcs_report() -> str:
    return """
    Extract the RLCS 2026 European Major tournament details as presented in the answer. Focus on the event that takes place in France in the summer months (May–August) of 2026.
    Return a JSON object with the following nested structure and fields. Use strings for all textual/number fields. If any field is missing in the answer, return null for that field or an empty list for URL arrays.

    {
      "tournament": {
        "tournament_name": string or null,
        "start_date": string or null,              // e.g., "May 20, 2026" or "May 20 2026"
        "end_date": string or null,                // e.g., "May 24, 2026"
        "public_start_date": string or null,       // start of public spectator days, if provided
        "public_end_date": string or null,         // end of public spectator days, if provided
        "identity_urls": [urls...],                // URL(s) cited for tournament identity and dates
        "classification_urls": [urls...]           // URL(s) cited for RLCS/Major classification (if separate)
      },
      "venue": {
        "venue_name": string or null,              // e.g., "Paris La Défense Arena"
        "city_or_area": string or null,            // e.g., "Nanterre"
        "country": string or null,                 // e.g., "France"
        "sporting_capacity": string or null,       // seating capacity for sporting events (not concerts)
        "venue_urls": [urls...]                    // URL(s) cited for venue info (official venue or RLCS pages)
      },
      "format": {
        "team_count": string or null,              // e.g., "16"
        "game_mode": string or null,               // e.g., "3v3"
        "group_stage_format": string or null,      // e.g., "single round robin with 4 groups of 4"
        "group_stage_start": string or null,       // dates for group stage start (if given)
        "group_stage_end": string or null,         // dates for group stage end (if given)
        "playoff_format": string or null,          // e.g., "12-team hybrid elimination bracket"
        "playoff_start": string or null,           // playoffs start date
        "playoff_end": string or null,             // playoffs end date
        "format_urls": [urls...]                   // URL(s) cited for format info
      },
      "prize": {
        "total_prize": string or null,             // e.g., "$354,000 USD"
        "first_prize": string or null,             // e.g., "$102,000 USD"
        "second_prize": string or null,            // e.g., "$51,000 USD"
        "prize_urls": [urls...]                    // URL(s) cited for prize info
      }
    }

    SPECIAL RULES FOR URL EXTRACTION:
    - Only include URLs explicitly present in the answer text.
    - Keep full URLs (with protocol). If a URL is missing protocol, prepend "http://".
    - For each category, include all URLs that the answer associates with that category.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return lst if lst else []


def _choose_sources(primary: List[str], fallback: List[str]) -> List[str]:
    return primary if primary else fallback


# --------------------------------------------------------------------------- #
# Verification section builders                                               #
# --------------------------------------------------------------------------- #
async def build_official_urls_section(
    evaluator: Evaluator,
    parent: VerificationNode,
    extraction: RLCSReportExtraction
) -> Dict[str, VerificationNode]:
    """
    Build the 'Official_Source_URLs' section and return key 'provided' prerequisite nodes
    for other sections to depend on.
    """
    official_node = evaluator.add_parallel(
        id="Official_Source_URLs",
        desc="All requested information is supported by reference URLs from official sources.",
        parent=parent,
        critical=True
    )

    # Identity & Dates
    identity_urls = _safe_list(extraction.tournament.identity_urls if extraction.tournament else [])
    classification_urls = _safe_list(extraction.tournament.classification_urls if extraction.tournament else [])
    id_sources = identity_urls + [u for u in classification_urls if u not in identity_urls]

    id_group = evaluator.add_parallel(
        id="Official_URLs_For_Tournament_Identity_and_Dates",
        desc="Provides official-source URL(s) supporting tournament name and dates.",
        parent=official_node,
        critical=True
    )
    id_provided = evaluator.add_custom_node(
        result=len(id_sources) > 0,
        id="Identity_URLs_Provided",
        desc="At least one URL is provided for tournament identity and dates",
        parent=id_group,
        critical=True
    )
    id_official = evaluator.add_leaf(
        id="Identity_URLs_Official",
        desc="At least one provided URL is an official RLCS/Rocket League source for tournament name/dates",
        parent=id_group,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of these URLs is an official Rocket League Esports or Psyonix source that states the tournament name and/or dates.",
        node=id_official,
        sources=id_sources,
        additional_instruction=(
            "Consider official if the domain belongs to Rocket League/Psyonix (e.g., rocketleague.com, "
            "esports.rocketleague.com, rocketleagueesports.com, blog.rocketleague.com) or the official "
            "Rocket League Esports social media accounts (e.g., x.com/RLEsports). Do not count wikis "
            "(Liquipedia), third-party news, or fan forums as official."
        )
    )

    # Venue details
    venue_urls = _safe_list(extraction.venue.venue_urls if extraction.venue else [])
    venue_group = evaluator.add_parallel(
        id="Official_URLs_For_Venue_Details",
        desc="Provides official-source URL(s) supporting venue name, location, and sporting-event capacity.",
        parent=official_node,
        critical=True
    )
    venue_provided = evaluator.add_custom_node(
        result=len(venue_urls) > 0,
        id="Venue_URLs_Provided",
        desc="At least one URL is provided for venue details",
        parent=venue_group,
        critical=True
    )
    venue_official = evaluator.add_leaf(
        id="Venue_URLs_Official",
        desc="At least one provided URL is an official venue or RLCS source with venue details",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of these URLs is an official venue website or an official RLCS page that states the venue name and location and/or capacity.",
        node=venue_official,
        sources=venue_urls,
        additional_instruction=(
            "Consider official if the domain belongs to the venue (e.g., parisladefense-arena.com, ledefensearena.com) "
            "or Rocket League/Psyonix domains. Do not count wikis or ticketing aggregators unless it's the official "
            "venue ticketing portal directly on the venue's domain."
        )
    )

    # Format
    format_urls = _safe_list(extraction.format.format_urls if extraction.format else [])
    format_group = evaluator.add_parallel(
        id="Official_URLs_For_Format",
        desc="Provides official-source URL(s) supporting team count, mode, and structure (group stage + playoffs).",
        parent=official_node,
        critical=True
    )
    format_provided = evaluator.add_custom_node(
        result=len(format_urls) > 0,
        id="Format_URLs_Provided",
        desc="At least one URL is provided for tournament format",
        parent=format_group,
        critical=True
    )
    format_official = evaluator.add_leaf(
        id="Format_URLs_Official",
        desc="At least one provided URL is an official RLCS/Rocket League source with format details",
        parent=format_group,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of these URLs is an official Rocket League Esports or Psyonix source that states the format details (teams, mode, structure).",
        node=format_official,
        sources=format_urls,
        additional_instruction=(
            "Consider official Rocket League/Psyonix domains or official RLCS social channels. Ignore wikis and third-party sites."
        )
    )

    # Prize info
    prize_urls = _safe_list(extraction.prize.prize_urls if extraction.prize else [])
    prize_group = evaluator.add_parallel(
        id="Official_URLs_For_Prize_Info",
        desc="Provides official-source URL(s) supporting total prize pool and 1st/2nd prize amounts.",
        parent=official_node,
        critical=True
    )
    prize_provided = evaluator.add_custom_node(
        result=len(prize_urls) > 0,
        id="Prize_URLs_Provided",
        desc="At least one URL is provided for prize information",
        parent=prize_group,
        critical=True
    )
    prize_official = evaluator.add_leaf(
        id="Prize_URLs_Official",
        desc="At least one provided URL is an official RLCS/Rocket League source with prize distribution",
        parent=prize_group,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of these URLs is an official Rocket League Esports or Psyonix source that states the total prize and prize distribution.",
        node=prize_official,
        sources=prize_urls,
        additional_instruction=(
            "Consider Rocket League/Psyonix official domains or sanctioned RLCS pages. Ignore wikis, news aggregators, and fan sites."
        )
    )

    return {
        "identity_provided": id_provided,
        "venue_provided": venue_provided,
        "format_provided": format_provided,
        "prize_provided": prize_provided
    }


async def build_tournament_section(
    evaluator: Evaluator,
    parent: VerificationNode,
    extraction: RLCSReportExtraction,
    prereq_nodes: Dict[str, VerificationNode]
) -> None:
    node = evaluator.add_parallel(
        id="Tournament_Identification_and_Dates",
        desc="Correctly identifies the tournament and satisfies required event classification and dates.",
        parent=parent,
        critical=True
    )

    # Official tournament name provided
    tournament_name = extraction.tournament.tournament_name if extraction.tournament else None
    evaluator.add_custom_node(
        result=bool(tournament_name and tournament_name.strip()),
        id="Official_Tournament_Name_Provided",
        desc="Provides the official tournament name (as stated by official sources).",
        parent=node,
        critical=True
    )

    # Prepare sources (identity + classification)
    identity_urls = _safe_list(extraction.tournament.identity_urls if extraction.tournament else [])
    classification_urls = _safe_list(extraction.tournament.classification_urls if extraction.tournament else [])
    id_sources = identity_urls + [u for u in classification_urls if u not in identity_urls]

    # Is official RLCS event
    is_rlcs_event = evaluator.add_leaf(
        id="Is_Official_RLCS_Event",
        desc="Tournament is an official Rocket League Championship Series (RLCS) event.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This tournament is an official Rocket League Championship Series (RLCS) event.",
        node=is_rlcs_event,
        sources=id_sources,
        additional_instruction="Verify that the page explicitly indicates RLCS affiliation (e.g., 'RLCS').",
        extra_prerequisites=[prereq_nodes["identity_provided"]]
    )

    # Is RLCS Major event
    is_major = evaluator.add_leaf(
        id="Is_RLCS_Major_Event",
        desc="Tournament is an RLCS Major tournament (i.e., classified as a Major by official sources).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This tournament is classified as an RLCS Major (Major event) by official sources.",
        node=is_major,
        sources=id_sources,
        additional_instruction="Look for the word 'Major' in the official RLCS materials.",
        extra_prerequisites=[prereq_nodes["identity_provided"]]
    )

    # Event dates exact: May 20–24, 2026
    dates_exact = evaluator.add_leaf(
        id="Event_Dates_Exact",
        desc="Tournament runs from May 20–24, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The tournament runs from May 20, 2026 to May 24, 2026.",
        node=dates_exact,
        sources=id_sources,
        additional_instruction="Accept variations like 'May 20–24, 2026' or '20-24 May 2026'.",
        extra_prerequisites=[prereq_nodes["identity_provided"]]
    )

    # Public attendance dates: May 22–24, 2026
    public_dates = evaluator.add_leaf(
        id="Public_Attendance_Dates",
        desc="Public attendance (crowd days) are May 22–24, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Public attendance (spectator/crowd days) are May 22, 2026 through May 24, 2026.",
        node=public_dates,
        sources=id_sources,
        additional_instruction="Look for phrasing like 'live audience', 'spectator days', 'crowd days'.",
        extra_prerequisites=[prereq_nodes["identity_provided"]]
    )


async def build_venue_section(
    evaluator: Evaluator,
    parent: VerificationNode,
    extraction: RLCSReportExtraction,
    prereq_nodes: Dict[str, VerificationNode]
) -> None:
    node = evaluator.add_parallel(
        id="Venue_Information",
        desc="Venue satisfies all required venue constraints and requested venue details are provided.",
        parent=parent,
        critical=True
    )

    venue = extraction.venue or VenueInfo()
    venue_urls = _safe_list(venue.venue_urls)

    # Venue name exact
    venue_name_leaf = evaluator.add_leaf(
        id="Venue_Name_Exact",
        desc="Venue is Paris La Défense Arena.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue is Paris La Défense Arena.",
        node=venue_name_leaf,
        sources=venue_urls,
        additional_instruction="Allow minor spelling/diacritic variations like 'Paris La Defense Arena'.",
        extra_prerequisites=[prereq_nodes["venue_provided"]]
    )

    # Venue location constraint (Nanterre, France)
    venue_loc_leaf = evaluator.add_leaf(
        id="Venue_Location_Constraint",
        desc="Venue is located in France, specifically in the Nanterre area (city/location provided accordingly).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Paris La Défense Arena is located in Nanterre, France.",
        node=venue_loc_leaf,
        sources=venue_urls,
        additional_instruction="Verify that the venue is in Nanterre (a commune in the western suburbs of Paris), France.",
        extra_prerequisites=[prereq_nodes["venue_provided"]]
    )

    # Venue capacity - split into provided + minimum check under the same rubric node
    capacity_group = evaluator.add_parallel(
        id="Venue_Capacity_Stated_and_Minimum",
        desc="States the venue seating capacity for sporting events, and it is at least 30,000.",
        parent=node,
        critical=True
    )
    capacity_provided = evaluator.add_custom_node(
        result=bool(venue.sporting_capacity and venue.sporting_capacity.strip()),
        id="Venue_Capacity_Provided",
        desc="Venue seating capacity for sporting events is provided in the answer",
        parent=capacity_group,
        critical=True
    )
    capacity_min_leaf = evaluator.add_leaf(
        id="Venue_Capacity_Min_30000",
        desc="Venue sporting-event seating capacity is at least 30,000.",
        parent=capacity_group,
        critical=True
    )
    await evaluator.verify(
        claim="The seating capacity for sporting events at Paris La Défense Arena is at least 30,000.",
        node=capacity_min_leaf,
        sources=venue_urls,
        additional_instruction="Use the capacity for sporting events (e.g., rugby configuration), not the maximum for concerts. Accept any figure ≥ 30,000.",
        extra_prerequisites=[prereq_nodes["venue_provided"], capacity_provided]
    )


async def build_format_section(
    evaluator: Evaluator,
    parent: VerificationNode,
    extraction: RLCSReportExtraction,
    prereq_nodes: Dict[str, VerificationNode]
) -> None:
    node = evaluator.add_parallel(
        id="Tournament_Format",
        desc="Format satisfies all required format constraints.",
        parent=parent,
        critical=True
    )

    fmt = extraction.format or FormatInfo()
    format_urls = _safe_list(fmt.format_urls)

    # Team count
    team_count_leaf = evaluator.add_leaf(
        id="Team_Count",
        desc="Tournament features 16 teams.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The tournament features 16 teams.",
        node=team_count_leaf,
        sources=format_urls,
        additional_instruction="Look for explicit team count or group structure implying 16 teams.",
        extra_prerequisites=[prereq_nodes["format_provided"]]
    )

    # Game mode
    game_mode_leaf = evaluator.add_leaf(
        id="Game_Mode",
        desc="Tournament is played in 3v3 format.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The tournament is played in a 3v3 format.",
        node=game_mode_leaf,
        sources=format_urls,
        additional_instruction="RLCS standard is 3v3 unless explicitly stated otherwise.",
        extra_prerequisites=[prereq_nodes["format_provided"]]
    )

    # Competition structure & timing (split into group and playoffs)
    comp_group = evaluator.add_parallel(
        id="Competition_Structure_Timing",
        desc="Tournament has a group stage (May 20–21, 2026) followed by playoffs (May 22–24, 2026).",
        parent=node,
        critical=True
    )

    group_timing = evaluator.add_leaf(
        id="Group_Stage_Timing",
        desc="Group stage takes place May 20–21, 2026.",
        parent=comp_group,
        critical=True
    )
    await evaluator.verify(
        claim="The group stage takes place on May 20–21, 2026.",
        node=group_timing,
        sources=format_urls,
        additional_instruction="Accept variations like 'May 20-21, 2026'.",
        extra_prerequisites=[prereq_nodes["format_provided"]]
    )

    playoffs_timing = evaluator.add_leaf(
        id="Playoff_Timing",
        desc="Playoffs take place May 22–24, 2026.",
        parent=comp_group,
        critical=True
    )
    await evaluator.verify(
        claim="The playoffs take place on May 22–24, 2026.",
        node=playoffs_timing,
        sources=format_urls,
        additional_instruction="Accept variations like 'May 22-24, 2026'.",
        extra_prerequisites=[prereq_nodes["format_provided"]]
    )

    # Group stage format
    group_format_leaf = evaluator.add_leaf(
        id="Group_Stage_Format",
        desc="Group stage uses single round robin format with 4 groups of 4 teams.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The group stage uses a single round robin format with 4 groups of 4 teams.",
        node=group_format_leaf,
        sources=format_urls,
        additional_instruction="Look for 'single round robin' and '4 groups of 4'.",
        extra_prerequisites=[prereq_nodes["format_provided"]]
    )

    # Playoff format
    playoff_format_leaf = evaluator.add_leaf(
        id="Playoff_Format",
        desc="Playoffs use a 12-team hybrid elimination bracket.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The playoffs use a 12-team hybrid elimination bracket.",
        node=playoff_format_leaf,
        sources=format_urls,
        additional_instruction="A hybrid elimination bracket may combine double/single elimination elements; verify the '12-team' and 'hybrid' aspects.",
        extra_prerequisites=[prereq_nodes["format_provided"]]
    )


async def build_prize_section(
    evaluator: Evaluator,
    parent: VerificationNode,
    extraction: RLCSReportExtraction,
    prereq_nodes: Dict[str, VerificationNode]
) -> None:
    node = evaluator.add_parallel(
        id="Prize_Information",
        desc="Prize pool satisfies all required prize constraints.",
        parent=parent,
        critical=True
    )

    prize = extraction.prize or PrizeInfo()
    prize_urls = _safe_list(prize.prize_urls)

    total_leaf = evaluator.add_leaf(
        id="Total_Prize_Pool",
        desc="Total prize pool is $354,000 USD.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The total prize pool is $354,000 USD.",
        node=total_leaf,
        sources=prize_urls,
        additional_instruction="Verify exact amount; currency should be USD.",
        extra_prerequisites=[prereq_nodes["prize_provided"]]
    )

    first_leaf = evaluator.add_leaf(
        id="First_Place_Prize",
        desc="First place prize is $102,000 USD.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The first place prize is $102,000 USD.",
        node=first_leaf,
        sources=prize_urls,
        additional_instruction="Verify the 1st place prize allocation in the official prize distribution.",
        extra_prerequisites=[prereq_nodes["prize_provided"]]
    )

    second_leaf = evaluator.add_leaf(
        id="Second_Place_Prize",
        desc="Second place prize is $51,000 USD.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The second place prize is $51,000 USD.",
        node=second_leaf,
        sources=prize_urls,
        additional_instruction="Verify the 2nd place prize allocation in the official prize distribution.",
        extra_prerequisites=[prereq_nodes["prize_provided"]]
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the RLCS 2026 European Major tournament task.
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_rlcs_report(),
        template_class=RLCSReportExtraction,
        extraction_name="rlcs_2026_france_major_report"
    )

    # Top-level critical report node
    report_node = evaluator.add_parallel(
        id="RLCS_2026_European_Major_Tournament_Report",
        desc="Evaluate whether the response identifies the correct RLCS Major event matching the given constraints and provides all required details with official-source reference URLs.",
        parent=root,
        critical=True
    )

    # Build Official Source URLs section first to provide gating prerequisites for other verifications
    prereq_nodes = await build_official_urls_section(evaluator, report_node, extraction)

    # Build remaining sections
    await build_tournament_section(evaluator, report_node, extraction, prereq_nodes)
    await build_venue_section(evaluator, report_node, extraction, prereq_nodes)
    await build_format_section(evaluator, report_node, extraction, prereq_nodes)
    await build_prize_section(evaluator, report_node, extraction, prereq_nodes)

    return evaluator.get_summary()