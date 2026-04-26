import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "venue_info_compilation"
TASK_DESCRIPTION = (
    "Compile comprehensive venue information for major live performance spaces across the following categories:\n\n"
    "1. NFL Thanksgiving 2024 Venues: Identify the three NFL stadiums that hosted Thanksgiving Day games on November 27, 2024 (Detroit Lions, Dallas Cowboys, and Baltimore Ravens games). For each stadium, provide:\n"
    "   - Stadium name\n"
    "   - Seating capacity\n"
    "   - The artist who performed the halftime show\n"
    "   - The broadcast network that aired the game\n"
    "   - The game start time (ET)\n\n"
    "2. Largest Broadway Theater: Identify the largest Broadway theater in New York City by seating capacity. Provide:\n"
    "   - Theater name\n"
    "   - Seating capacity\n"
    "   - Current show or production running at the theater\n"
    "   - Street address\n\n"
    "3. Radio City Christmas Spectacular Venue: Identify the New York City venue that hosts the Radio City Christmas Spectacular. Provide:\n"
    "   - Venue name\n"
    "   - Seating capacity\n"
    "   - The date range for the 2025 season\n"
    "   - Full street address\n"
    "   - Show runtime\n\n"
    "For each piece of information, include a supporting URL reference from an official or reputable source."
)


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class FieldWithSources(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class NFLVenueFields(BaseModel):
    stadium_name: Optional[FieldWithSources] = None
    seating_capacity: Optional[FieldWithSources] = None
    halftime_artist: Optional[FieldWithSources] = None
    broadcast_network: Optional[FieldWithSources] = None
    start_time_et: Optional[FieldWithSources] = None


class VenuesNFL(BaseModel):
    detroit: Optional[NFLVenueFields] = None
    dallas: Optional[NFLVenueFields] = None
    baltimore: Optional[NFLVenueFields] = None


class BroadwayLargest(BaseModel):
    theater_name: Optional[FieldWithSources] = None
    seating_capacity: Optional[FieldWithSources] = None
    current_show: Optional[FieldWithSources] = None
    street_address: Optional[FieldWithSources] = None


class RadioCityVenue(BaseModel):
    venue_name: Optional[FieldWithSources] = None
    seating_capacity: Optional[FieldWithSources] = None
    season_2025_date_range: Optional[FieldWithSources] = None
    full_street_address: Optional[FieldWithSources] = None
    runtime: Optional[FieldWithSources] = None


class VenuesExtraction(BaseModel):
    nfl: Optional[VenuesNFL] = None
    broadway_largest: Optional[BroadwayLargest] = None
    radio_city: Optional[RadioCityVenue] = None


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract structured information exactly as presented in the answer. For every requested field, also extract all supporting URLs explicitly cited in the answer that correspond to that field.

    Return a single JSON object with this exact structure (use null for any missing sub-objects, and use null for any missing values; use [] for missing sources):

    {
      "nfl": {
        "detroit": {
          "stadium_name": {"value": string|null, "sources": [string, ...]},
          "seating_capacity": {"value": string|null, "sources": [string, ...]},
          "halftime_artist": {"value": string|null, "sources": [string, ...]},
          "broadcast_network": {"value": string|null, "sources": [string, ...]},
          "start_time_et": {"value": string|null, "sources": [string, ...]}
        },
        "dallas": {
          "stadium_name": {"value": string|null, "sources": [string, ...]},
          "seating_capacity": {"value": string|null, "sources": [string, ...]},
          "halftime_artist": {"value": string|null, "sources": [string, ...]},
          "broadcast_network": {"value": string|null, "sources": [string, ...]},
          "start_time_et": {"value": string|null, "sources": [string, ...]}
        },
        "baltimore": {
          "stadium_name": {"value": string|null, "sources": [string, ...]},
          "seating_capacity": {"value": string|null, "sources": [string, ...]},
          "halftime_artist": {"value": string|null, "sources": [string, ...]},
          "broadcast_network": {"value": string|null, "sources": [string, ...]},
          "start_time_et": {"value": string|null, "sources": [string, ...]}
        }
      },
      "broadway_largest": {
        "thater_name": {"value": string|null, "sources": [string, ...]},
        "seating_capacity": {"value": string|null, "sources": [string, ...]},
        "current_show": {"value": string|null, "sources": [string, ...]},
        "street_address": {"value": string|null, "sources": [string, ...]}
      },
      "radio_city": {
        "venue_name": {"value": string|null, "sources": [string, ...]},
        "seating_capacity": {"value": string|null, "sources": [string, ...]},
        "season_2025_date_range": {"value": string|null, "sources": [string, ...]},
        "full_street_address": {"value": string|null, "sources": [string, ...]},
        "runtime": {"value": string|null, "sources": [string, ...]}
      }
    }

    IMPORTANT CLARIFICATIONS:
    - The NFL portion targets the Thanksgiving 2024 games for the Detroit Lions, Dallas Cowboys, and Baltimore Ravens. Extract the five fields for each, and include all URLs cited in the answer that support those specific facts.
    - For the Broadway portion, the "theater_name" URLs should support the claim that it is the largest Broadway theater by seating capacity (not just a large theatre in general).
    - For Radio City, the "season_2025_date_range" refers specifically to the 2025 season of the Christmas Spectacular.
    - Always extract URLs exactly as written in the answer (including markdown links).
    - For seating capacities, keep them as strings exactly as presented (e.g., "80,000", "approx. 65,000", "5,960").
    - If the answer does not provide a URL for a field, return an empty array for that field's "sources".
    - If the answer provides multiple URLs for a field, include all of them in the "sources" array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_value(field: Optional[FieldWithSources]) -> str:
    return field.value if (field and field.value) else ""


def _safe_sources(field: Optional[FieldWithSources]) -> List[str]:
    return field.sources if (field and field.sources) else []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _verify_field_group(
    evaluator: Evaluator,
    parent_node,
    group_id: str,
    group_desc: str,
    field: Optional[FieldWithSources],
    claim: str,
    additional_instruction: str
) -> None:
    """
    Build a critical field group that requires:
      1) Existence with at least one supporting URL
      2) Claim supported by the cited URL(s)
    """
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent_node,
        critical=True
    )

    # 1) Existence check
    exists = (field is not None) and (field.value is not None and str(field.value).strip() != "") and (len(_safe_sources(field)) > 0)
    evaluator.add_custom_node(
        result=exists,
        id=f"{group_id}_exists",
        desc=f"{group_desc} — value present and at least one supporting URL provided",
        parent=group_node,
        critical=True
    )

    # 2) Support check via URLs
    support_leaf = evaluator.add_leaf(
        id=f"{group_id}_supported",
        desc=f"{group_desc} — claim is supported by cited source(s)",
        parent=group_node,
        critical=True
    )

    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=_safe_sources(field),
        additional_instruction=additional_instruction
    )


async def _verify_nfl_category(
    evaluator: Evaluator,
    root,
    cat_id: str,
    cat_desc: str,
    team_context_label: str,
    data: Optional[NFLVenueFields]
) -> None:
    """
    Verify all fields for one NFL Thanksgiving 2024 venue category.
    """
    node = evaluator.add_parallel(
        id=cat_id,
        desc=cat_desc,
        parent=root,
        critical=False
    )

    if data is None:
        # Add placeholder failed children to reflect missing category gracefully via critical children
        # Each required field group will be built with None => existence fails, support skipped
        data = NFLVenueFields()

    # Stadium name
    stadium_name_claim = f"The stadium that hosted the {team_context_label} Thanksgiving 2024 game is '{_safe_value(data.stadium_name)}'."
    stadium_name_ins = (
        "Verify the venue identity (stadium name). It is acceptable if the source confirms the team's home stadium "
        "or the specific Thanksgiving game venue. Allow minor formatting variations (e.g., punctuation or abbreviations). "
        "Ignore any exact date mismatch; focus on whether the given stadium name is shown to be the venue for the team/game."
    )
    await _verify_field_group(
        evaluator,
        node,
        f"{cat_id}__stadium_name_with_url",
        f"{team_context_label}: stadium name with supporting URL",
        data.stadium_name,
        stadium_name_claim,
        stadium_name_ins
    )

    # Seating capacity
    stadium_name_for_capacity = _safe_value(data.stadium_name) or "the stadium"
    capacity_claim = f"The seating capacity of {stadium_name_for_capacity} is '{_safe_value(data.seating_capacity)}'."
    capacity_ins = (
        "Verify the standard listed seating capacity for the stadium. Minor variations or qualifiers (e.g., 'about', 'expandable') "
        "are acceptable as long as the figure aligns with the source. Do not require exact match formatting for commas or spaces."
    )
    await _verify_field_group(
        evaluator,
        node,
        f"{cat_id}__seating_capacity_with_url",
        f"{team_context_label}: seating capacity with supporting URL",
        data.seating_capacity,
        capacity_claim,
        capacity_ins
    )

    # Halftime artist
    halftime_claim = f"The halftime show performer for the {team_context_label} Thanksgiving 2024 game was '{_safe_value(data.halftime_artist)}'."
    halftime_ins = (
        "Verify the halftime performer for that specific Thanksgiving game. Accept reputable sources such as NFL, team websites, "
        "broadcasters' announcements, or major news outlets."
    )
    await _verify_field_group(
        evaluator,
        node,
        f"{cat_id}__halftime_artist_with_url",
        f"{team_context_label}: halftime show performer with supporting URL",
        data.halftime_artist,
        halftime_claim,
        halftime_ins
    )

    # Broadcast network
    network_claim = f"The broadcast network for the {team_context_label} Thanksgiving 2024 game was '{_safe_value(data.broadcast_network)}'."
    network_ins = (
        "Verify the broadcasting network (e.g., CBS, FOX, NBC, Prime Video). Focus on the network identity; "
        "ignore simulcasts or streaming specifics unless they contradict the main claim."
    )
    await _verify_field_group(
        evaluator,
        node,
        f"{cat_id}__broadcast_network_with_url",
        f"{team_context_label}: broadcast network with supporting URL",
        data.broadcast_network,
        network_claim,
        network_ins
    )

    # Start time ET
    kickoff_claim = f"The kickoff time (Eastern Time) for the {team_context_label} Thanksgiving 2024 game was '{_safe_value(data.start_time_et)}'."
    kickoff_ins = (
        "Verify the listed kickoff time in ET/Eastern Time. Minor formatting differences (e.g., 'p.m. ET' vs 'PM ET') are acceptable. "
        "Focus on the time-of-day value."
    )
    await _verify_field_group(
        evaluator,
        node,
        f"{cat_id}__start_time_et_with_url",
        f"{team_context_label}: game start time (ET) with supporting URL",
        data.start_time_et,
        kickoff_claim,
        kickoff_ins
    )


async def _verify_broadway_largest(
    evaluator: Evaluator,
    root,
    data: Optional[BroadwayLargest]
) -> None:
    node = evaluator.add_parallel(
        id="largest_broadway_theater",
        desc="Largest Broadway theater by seating capacity in NYC: provide all required fields; each field must include a supporting URL from an official or reputable source.",
        parent=root,
        critical=False
    )

    if data is None:
        data = BroadwayLargest()

    # Theater name (must support largest-by-capacity claim)
    name_val = _safe_value(data.theater_name)
    name_claim = f"The largest Broadway theater by seating capacity is '{name_val}'."
    name_ins = (
        "Verify that the cited source explicitly supports that this is the largest Broadway theater by seating capacity (not merely one of the largest). "
        "Focus on 'Broadway' theaters specifically (Broadway League classification)."
    )
    await _verify_field_group(
        evaluator,
        node,
        "largest_broadway_theater__theater_name_with_url",
        "Largest Broadway theater: theater name with supporting URL (supports 'largest by seating capacity' claim)",
        data.theater_name,
        name_claim,
        name_ins
    )

    # Seating capacity
    capacity_claim = f"The seating capacity of '{name_val if name_val else 'the theater'}' is '{_safe_value(data.seating_capacity)}'."
    capacity_ins = (
        "Verify the listed seating capacity. Minor formatting differences and qualifiers (e.g., 'about') are acceptable."
    )
    await _verify_field_group(
        evaluator,
        node,
        "largest_broadway_theater__seating_capacity_with_url",
        "Largest Broadway theater: seating capacity with supporting URL",
        data.seating_capacity,
        capacity_claim,
        capacity_ins
    )

    # Current show
    show_claim = f"The current show or production running at '{name_val if name_val else 'the theater'}' is '{_safe_value(data.current_show)}'."
    show_ins = (
        "Verify that the cited source indicates this show is currently running at the theater (e.g., theater's official site, show's official site, Broadway League). "
        "Some variations (e.g., limited engagement) are acceptable if the page implies 'current'."
    )
    await _verify_field_group(
        evaluator,
        node,
        "largest_broadway_theater__current_show_with_url",
        "Largest Broadway theater: current show with supporting URL",
        data.current_show,
        show_claim,
        show_ins
    )

    # Street address
    addr_claim = f"The street address of '{name_val if name_val else 'the theater'}' is '{_safe_value(data.street_address)}'."
    addr_ins = (
        "Verify the theater's official street address. Formatting variations like 'W.' vs 'West' are acceptable."
    )
    await _verify_field_group(
        evaluator,
        node,
        "largest_broadway_theater__street_address_with_url",
        "Largest Broadway theater: street address with supporting URL",
        data.street_address,
        addr_claim,
        addr_ins
    )


async def _verify_radio_city(
    evaluator: Evaluator,
    root,
    data: Optional[RadioCityVenue]
) -> None:
    node = evaluator.add_parallel(
        id="radio_city_christmas_spectacular_venue",
        desc="NYC venue hosting the Radio City Christmas Spectacular: provide all required fields; each field must include a supporting URL from an official or reputable source.",
        parent=root,
        critical=False
    )

    if data is None:
        data = RadioCityVenue()

    venue_name = _safe_value(data.venue_name)

    # Venue name
    name_claim = f"The New York City venue that hosts the Radio City Christmas Spectacular is '{venue_name}'."
    name_ins = (
        "Verify that the cited source explicitly shows this is the venue hosting the Radio City Christmas Spectacular."
    )
    await _verify_field_group(
        evaluator,
        node,
        "radio_city_christmas_spectacular_venue__venue_name_with_url",
        "Radio City Christmas Spectacular: venue name with supporting URL",
        data.venue_name,
        name_claim,
        name_ins
    )

    # Seating capacity
    cap_claim = f"The seating capacity of '{venue_name if venue_name else 'the venue'}' is '{_safe_value(data.seating_capacity)}'."
    cap_ins = (
        "Verify the listed seating capacity for the venue (e.g., Radio City Music Hall). Minor formatting differences or typical approximations are acceptable."
    )
    await _verify_field_group(
        evaluator,
        node,
        "radio_city_christmas_spectacular_venue__seating_capacity_with_url",
        "Radio City Christmas Spectacular: venue seating capacity with supporting URL",
        data.seating_capacity,
        cap_claim,
        cap_ins
    )

    # 2025 date range
    dr_claim = f"The date range for the 2025 Radio City Christmas Spectacular season is '{_safe_value(data.season_2025_date_range)}'."
    dr_ins = (
        "Verify that the cited source lists the 2025 season schedule or an explicit date range for the 2025 Christmas Spectacular. "
        "If the source shows a start and end date that match the claim, it is acceptable."
    )
    await _verify_field_group(
        evaluator,
        node,
        "radio_city_christmas_spectacular_venue__2025_date_range_with_url",
        "Radio City Christmas Spectacular: 2025 season date range with supporting URL",
        data.season_2025_date_range,
        dr_claim,
        dr_ins
    )

    # Full street address
    full_addr_claim = f"The full street address of '{venue_name if venue_name else 'the venue'}' is '{_safe_value(data.full_street_address)}'."
    full_addr_ins = (
        "Verify the official full street address. Formatting variations are acceptable as long as content matches."
    )
    await _verify_field_group(
        evaluator,
        node,
        "radio_city_christmas_spectacular_venue__full_street_address_with_url",
        "Radio City Christmas Spectacular: full street address with supporting URL",
        data.full_street_address,
        full_addr_claim,
        full_addr_ins
    )

    # Runtime
    runtime_claim = f"The show runtime for the Radio City Christmas Spectacular is '{_safe_value(data.runtime)}'."
    runtime_ins = (
        "Verify the show's runtime/duration. Accept reasonable variants (e.g., 'about 90 minutes')."
    )
    await _verify_field_group(
        evaluator,
        node,
        "radio_city_christmas_spectacular_venue__runtime_with_url",
        "Radio City Christmas Spectacular: show runtime with supporting URL",
        data.runtime,
        runtime_claim,
        runtime_ins
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
    Evaluate an answer for the multi-venue information task.
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

    # Extract all structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # NFL Detroit
    await _verify_nfl_category(
        evaluator,
        root,
        "nfl_detroit_thanksgiving_venue",
        "Detroit Lions Thanksgiving 2024 venue: provide all required fields; each field must include a supporting URL from an official or reputable source.",
        "Detroit Lions",
        extraction.nfl.detroit if (extraction.nfl and extraction.nfl.detroit) else None
    )

    # NFL Dallas
    await _verify_nfl_category(
        evaluator,
        root,
        "nfl_dallas_thanksgiving_venue",
        "Dallas Cowboys Thanksgiving 2024 venue: provide all required fields; each field must include a supporting URL from an official or reputable source.",
        "Dallas Cowboys",
        extraction.nfl.dallas if (extraction.nfl and extraction.nfl.dallas) else None
    )

    # NFL Baltimore
    await _verify_nfl_category(
        evaluator,
        root,
        "nfl_baltimore_thanksgiving_venue",
        "Baltimore Ravens Thanksgiving 2024 venue: provide all required fields; each field must include a supporting URL from an official or reputable source.",
        "Baltimore Ravens",
        extraction.nfl.baltimore if (extraction.nfl and extraction.nfl.baltimore) else None
    )

    # Largest Broadway Theater
    await _verify_broadway_largest(
        evaluator,
        root,
        extraction.broadway_largest if extraction.broadway_largest else None
    )

    # Radio City Christmas Spectacular Venue
    await _verify_radio_city(
        evaluator,
        root,
        extraction.radio_city if extraction.radio_city else None
    )

    return evaluator.get_summary()