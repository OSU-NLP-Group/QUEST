import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "entertainment_2025_2026"
TASK_DESCRIPTION = (
    "Provide detailed information about the following four distinct items from the entertainment industry in 2025-2026:\n\n"
    "1. A Music Festival: Identify a multi-day music festival held in the United States between September 2025 and June 2026. "
    "The festival must span at least two consecutive days and be held at a venue with a capacity of at least 15,000 people. "
    "The lineup must have been publicly announced before February 20, 2026. Provide: (a) the official festival name, (b) exact dates "
    "(month, day, year), (c) venue name, city, and state, (d) venue capacity, and (e) at least three confirmed performers.\n\n"
    "2. An International Music Awards Ceremony: Identify an international music awards ceremony that took place in 2025 over at least two days. "
    "Provide: (a) the official ceremony name, (b) exact dates (month, day, year), (c) venue name and city, and (d) the winners of at least three "
    "major award categories with the specific category names and winner names.\n\n"
    "3. A Television Host Achievement: Identify a television host who hosts a long-running show on a major U.S. network and has been with that network "
    "for at least 30 years. The show must have been on air for at least 15 years. Provide: (a) the host's full name, (b) the official show title, "
    "(c) the network name, (d) the year the host joined the network, and (e) the year the show premiered.\n\n"
    "4. A Nonprofit Foundation: Identify a nonprofit foundation founded by a public figure that focuses on supporting survivors of violence or abuse. "
    "The foundation must have been operating for at least 10 years as of 2026 and must operate at least three distinct programs. Provide: "
    "(a) the official foundation name, (b) the founder's name, (c) the founding year, (d) a description of the foundation's mission, "
    "(e) the names and descriptions of at least three programs, and (f) quantifiable impact metrics for at least one program.\n\n"
    "For each item, include reference URLs that verify the information provided."
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class FestivalInfo(BaseModel):
    name: Optional[str] = None
    start_date: Optional[str] = None  # Allow flexible formats
    end_date: Optional[str] = None
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None
    performers: List[str] = Field(default_factory=list)
    lineup_announcement_date: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class AwardsWinner(BaseModel):
    category: Optional[str] = None
    winner: Optional[str] = None


class AwardsInfo(BaseModel):
    name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    venue_name: Optional[str] = None
    city: Optional[str] = None
    winners: List[AwardsWinner] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


class TVHostInfo(BaseModel):
    host_name: Optional[str] = None
    show_title: Optional[str] = None
    network: Optional[str] = None
    host_join_year: Optional[str] = None
    show_premiere_year: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class ProgramInfo(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    metrics: Optional[str] = None  # Prefer string to allow flexible numeric phrasing


class FoundationInfo(BaseModel):
    name: Optional[str] = None
    founder: Optional[str] = None
    founding_year: Optional[str] = None
    mission: Optional[str] = None
    programs: List[ProgramInfo] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


class AllItemsExtraction(BaseModel):
    festival: Optional[FestivalInfo] = None
    awards: Optional[AwardsInfo] = None
    tv_host: Optional[TVHostInfo] = None
    foundation: Optional[FoundationInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all_items() -> str:
    return (
        "Extract four distinct items and their fields exactly as presented in the answer. Do not invent data.\n\n"
        "Item 1: Music Festival (U.S., between Sep 2025 and Jun 2026)\n"
        "- name: official festival name\n"
        "- start_date: exact start date (month day, year or yyyy-mm-dd)\n"
        "- end_date: exact end date\n"
        "- venue_name: venue name\n"
        "- city: city\n"
        "- state: state (must be U.S.)\n"
        "- capacity: venue capacity (number or descriptive string including numbers)\n"
        "- performers: array of at least 3 confirmed performers\n"
        "- lineup_announcement_date: date when lineup was publicly announced (if provided)\n"
        "- reference_urls: array of URLs cited for this festival\n\n"
        "Item 2: International Music Awards Ceremony (held in 2025, multi-day)\n"
        "- name: official ceremony name\n"
        "- start_date: exact start date (month day, year or yyyy-mm-dd)\n"
        "- end_date: exact end date\n"
        "- venue_name: venue name\n"
        "- city: city\n"
        "- winners: array of objects, each with 'category' and 'winner' strings, for at least 3 major award categories\n"
        "- reference_urls: array of URLs cited for this ceremony\n\n"
        "Item 3: Television Host Achievement (major U.S. network, tenure ≥30 years; show ≥15 years)\n"
        "- host_name: full name\n"
        "- show_title: official show title\n"
        "- network: network name\n"
        "- host_join_year: year the host joined the network\n"
        "- show_premiere_year: year the show premiered\n"
        "- reference_urls: array of URLs cited for this host/show\n\n"
        "Item 4: Nonprofit Foundation (supports survivors of violence/abuse; operating ≥10 years as of 2026; ≥3 programs)\n"
        "- name: official foundation name\n"
        "- founder: founder's name\n"
        "- founding_year: founding year\n"
        "- mission: description of mission (should clearly relate to survivor/victim support)\n"
        "- programs: array of objects, each with 'name', 'description', and optional 'metrics' (include numbers/percentages if available)\n"
        "- reference_urls: array of URLs cited for this foundation\n\n"
        "If any field is missing in the answer, return null for that field or an empty array where appropriate. Extract only explicitly stated data."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(x: Optional[List[str]]) -> List[str]:
    return x if x else []


def _join_date_range(start: Optional[str], end: Optional[str]) -> str:
    if start and end:
        return f"{start} to {end}"
    return (start or "") or (end or "")


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_music_festival(evaluator: Evaluator, parent_node, fest: Optional[FestivalInfo]) -> None:
    item_node = evaluator.add_parallel(
        id="item_1_music_festival",
        desc="A multi-day music festival held in the United States between September 2025 and June 2026 at a major venue",
        parent=parent_node,
        critical=False
    )
    sources = _safe_list(fest.reference_urls if fest else [])

    # Basic info
    basic_node = evaluator.add_parallel(
        id="festival_basic_info",
        desc="Basic information about the festival including name, dates, and location",
        parent=item_node,
        critical=True
    )

    # Name
    name_node = evaluator.add_parallel(
        id="festival_name",
        desc="The official name of the festival is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fest and fest.name and fest.name.strip()),
        id="festival_name_present",
        desc="Festival name is present in the answer",
        parent=name_node,
        critical=True
    )
    name_ref_leaf = evaluator.add_leaf(
        id="festival_name_reference",
        desc="A reference URL confirming the festival name is provided",
        parent=name_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official festival name is '{fest.name if fest and fest.name else ''}'.",
        node=name_ref_leaf,
        sources=sources,
        additional_instruction="Confirm that the provided sources explicitly mention the same official festival name."
    )

    # Dates
    dates_node = evaluator.add_parallel(
        id="festival_dates",
        desc="The exact dates of the festival (month, day, and year) are provided and fall between September 2025 and June 2026",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fest and fest.start_date and fest.end_date),
        id="festival_dates_provided",
        desc="Festival start and end dates are provided",
        parent=dates_node,
        critical=True
    )
    dates_in_range_leaf = evaluator.add_leaf(
        id="festival_dates_in_window",
        desc="Festival dates fall between Sep 1, 2025 and Jun 30, 2026 (inclusive)",
        parent=dates_node,
        critical=True
    )
    await evaluator.verify(
        claim=(f"The festival dates '{_join_date_range(fest.start_date if fest else None, fest.end_date if fest else None)}' "
               "fall between September 1, 2025 and June 30, 2026, inclusive."),
        node=dates_in_range_leaf,
        additional_instruction="Use reasonable date parsing. Treat the window as inclusive. If only month/day order differs, normalize."
    )
    dates_ref_leaf = evaluator.add_leaf(
        id="festival_dates_reference",
        desc="A reference URL confirming the festival dates is provided",
        parent=dates_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The festival takes place from {fest.start_date if fest else ''} to {fest.end_date if fest else ''}.",
        node=dates_ref_leaf,
        sources=sources,
        additional_instruction="Verify that sources explicitly list the same exact festival dates."
    )

    # Location
    loc_node = evaluator.add_parallel(
        id="festival_location",
        desc="The specific venue name, city, and state in the United States are provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fest and fest.venue_name and fest.city and fest.state),
        id="festival_location_provided",
        desc="Festival venue name, city, and state are provided",
        parent=loc_node,
        critical=True
    )
    loc_ref_leaf = evaluator.add_leaf(
        id="festival_location_reference",
        desc="A reference URL confirming the festival location is provided",
        parent=loc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The festival is held at {fest.venue_name if fest else ''}, {fest.city if fest else ''}, {fest.state if fest else ''}, United States.",
        node=loc_ref_leaf,
        sources=sources,
        additional_instruction="Verify the venue, city, and state are explicitly mentioned and match the claim."
    )

    # Duration (multi-day)
    duration_node = evaluator.add_parallel(
        id="festival_duration",
        desc="The festival spans at least two consecutive days",
        parent=item_node,
        critical=True
    )
    consecutive_leaf = evaluator.add_leaf(
        id="consecutive_days_verification",
        desc="The provided dates show the festival occurs over at least two consecutive days",
        parent=duration_node,
        critical=True
    )
    await evaluator.verify(
        claim=(f"The festival spans at least two consecutive days from {fest.start_date if fest else ''} "
               f"to {fest.end_date if fest else ''}."),
        node=consecutive_leaf,
        additional_instruction="Confirm that end_date is at least one day after start_date and dates are consecutive (no gaps)."
    )
    duration_ref_leaf = evaluator.add_leaf(
        id="duration_reference",
        desc="A reference URL confirming the festival duration is provided",
        parent=duration_node,
        critical=True
    )
    await evaluator.verify(
        claim="The festival is a multi-day event (two or more consecutive days).",
        node=duration_ref_leaf,
        sources=sources,
        additional_instruction="The source should indicate multiple days or a schedule spanning multiple consecutive dates."
    )

    # Venue capacity
    capacity_node = evaluator.add_parallel(
        id="festival_venue_capacity",
        desc="The venue has a capacity of at least 15,000 people",
        parent=item_node,
        critical=True
    )
    capacity_spec_node = evaluator.add_parallel(
        id="capacity_specification",
        desc="The capacity of the venue is stated",
        parent=capacity_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fest and fest.capacity and fest.capacity.strip()),
        id="capacity_present",
        desc="Venue capacity value is provided",
        parent=capacity_spec_node,
        critical=True
    )
    capacity_value_leaf = evaluator.add_leaf(
        id="capacity_value",
        desc="The stated capacity meets or exceeds 15,000 people",
        parent=capacity_spec_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue capacity '{fest.capacity if fest else ''}' meets or exceeds 15,000 people.",
        node=capacity_value_leaf,
        additional_instruction="Interpret capacity strings. If a range or 'approximately', judge if 15,000 threshold is reasonably met."
    )
    capacity_ref_leaf = evaluator.add_leaf(
        id="capacity_reference",
        desc="A reference URL confirming the venue capacity is provided",
        parent=capacity_spec_node,
        critical=True
    )
    await evaluator.verify(
        claim=(f"The capacity of {fest.venue_name if fest else ''} is {fest.capacity if fest else ''} "
               "or at least 15,000."),
        node=capacity_ref_leaf,
        sources=sources,
        additional_instruction="Confirm the venue capacity on official venue pages or reliable sources."
    )

    # Lineup
    lineup_node = evaluator.add_parallel(
        id="festival_lineup",
        desc="At least three confirmed performers for the festival are listed",
        parent=item_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fest and fest.performers and len(fest.performers) >= 3),
        id="performer_count",
        desc="At least three specific performer names are provided",
        parent=lineup_node,
        critical=True
    )
    performer_ref_leaf = evaluator.add_leaf(
        id="performer_reference",
        desc="A reference URL confirming the festival lineup is provided",
        parent=lineup_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The festival lineup includes at least these performers: {', '.join(fest.performers[:5]) if fest and fest.performers else ''}.",
        node=performer_ref_leaf,
        sources=sources,
        additional_instruction="Sources (festival site or reputable press) should confirm these performers are part of the lineup."
    )
    announcement_leaf = evaluator.add_leaf(
        id="lineup_announcement",
        desc="The lineup was publicly announced before February 20, 2026",
        parent=lineup_node,
        critical=True
    )
    await evaluator.verify(
        claim=(f"The lineup announcement date '{fest.lineup_announcement_date if fest else ''}' "
               "is on or before February 20, 2026."),
        node=announcement_leaf,
        additional_instruction="If the date is unclear, infer from article timestamps if clearly before Feb 20, 2026. Use inclusive comparison."
    )
    announcement_ref_leaf = evaluator.add_leaf(
        id="announcement_reference",
        desc="A reference URL confirming the announcement timing is provided",
        parent=lineup_node,
        critical=True
    )
    await evaluator.verify(
        claim="The festival lineup announcement occurred on or before February 20, 2026.",
        node=announcement_ref_leaf,
        sources=sources,
        additional_instruction="Verify announcement articles or official posts have timestamps before or on Feb 20, 2026."
    )


async def verify_awards_ceremony(evaluator: Evaluator, parent_node, awards: Optional[AwardsInfo]) -> None:
    item_node = evaluator.add_parallel(
        id="item_2_awards_ceremony",
        desc="An international music awards ceremony that took place in 2025 with specific winner information",
        parent=parent_node,
        critical=False
    )
    sources = _safe_list(awards.reference_urls if awards else [])

    # Basic info
    basic_node = evaluator.add_parallel(
        id="ceremony_basic_info",
        desc="Basic information about the awards ceremony including name, dates, and location",
        parent=item_node,
        critical=True
    )

    # Name
    name_node = evaluator.add_parallel(
        id="ceremony_name",
        desc="The official name of the awards ceremony is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(awards and awards.name and awards.name.strip()),
        id="ceremony_name_present",
        desc="Awards ceremony name is present",
        parent=name_node,
        critical=True
    )
    name_ref_leaf = evaluator.add_leaf(
        id="ceremony_name_reference",
        desc="A reference URL confirming the ceremony name is provided",
        parent=name_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official awards ceremony name is '{awards.name if awards else ''}'.",
        node=name_ref_leaf,
        sources=sources,
        additional_instruction="Confirm the ceremony name through official pages or reputable coverage."
    )

    # Dates
    dates_node = evaluator.add_parallel(
        id="ceremony_dates",
        desc="The exact dates of the ceremony (month, day, and year in 2025) are provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(awards and awards.start_date and awards.end_date),
        id="ceremony_dates_provided",
        desc="Ceremony start and end dates are provided",
        parent=dates_node,
        critical=True
    )
    dates_ref_leaf = evaluator.add_leaf(
        id="ceremony_dates_reference",
        desc="A reference URL confirming the ceremony dates is provided",
        parent=dates_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The awards ceremony takes place from {awards.start_date if awards else ''} to {awards.end_date if awards else ''} in 2025.",
        node=dates_ref_leaf,
        sources=sources,
        additional_instruction="Verify both dates and that the event happened in 2025."
    )

    # Location
    loc_node = evaluator.add_parallel(
        id="ceremony_location",
        desc="The specific venue name and city are provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(awards and awards.venue_name and awards.city),
        id="ceremony_location_provided",
        desc="Awards ceremony venue and city are provided",
        parent=loc_node,
        critical=True
    )
    loc_ref_leaf = evaluator.add_leaf(
        id="ceremony_location_reference",
        desc="A reference URL confirming the ceremony location is provided",
        parent=loc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The awards ceremony is held at {awards.venue_name if awards else ''} in {awards.city if awards else ''}.",
        node=loc_ref_leaf,
        sources=sources,
        additional_instruction="Confirm venue and city details in reliable sources."
    )

    # Duration (multi-day)
    duration_node = evaluator.add_parallel(
        id="ceremony_duration",
        desc="The ceremony took place over at least two days",
        parent=item_node,
        critical=True
    )
    multi_day_leaf = evaluator.add_leaf(
        id="multi_day_verification",
        desc="The provided dates show the ceremony occurred over at least two days",
        parent=duration_node,
        critical=True
    )
    await evaluator.verify(
        claim=(f"The awards ceremony spans at least two days from {awards.start_date if awards else ''} "
               f"to {awards.end_date if awards else ''}."),
        node=multi_day_leaf,
        additional_instruction="Confirm end_date is at least one day after start_date. Treat consecutive dates appropriately."
    )
    duration_ref_leaf = evaluator.add_leaf(
        id="duration_reference",
        desc="A reference URL confirming the ceremony duration is provided",
        parent=duration_node,
        critical=True
    )
    await evaluator.verify(
        claim="The awards ceremony takes place over multiple days (two or more).",
        node=duration_ref_leaf,
        sources=sources,
        additional_instruction="Sources should explicitly indicate multi-day scheduling or date ranges."
    )

    # Winners
    winners_node = evaluator.add_parallel(
        id="award_winners",
        desc="Winners of at least three major award categories are provided",
        parent=item_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(awards and awards.winners and len([w for w in awards.winners if w.category and w.winner]) >= 3),
        id="winner_count",
        desc="Winners for at least three distinct award categories are listed",
        parent=winners_node,
        critical=True
    )
    details_node = evaluator.add_parallel(
        id="category_and_winner_details",
        desc="For each category, both the category name and winner name are provided",
        parent=winners_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(awards and all((w.category and w.winner) for w in awards.winners[:3])),
        id="winner_details_provided",
        desc="At least three winners have both category and winner names",
        parent=details_node,
        critical=True
    )
    winners_ref_leaf = evaluator.add_leaf(
        id="winners_reference",
        desc="A reference URL confirming the award winners is provided",
        parent=details_node,
        critical=True
    )
    winners_summary = ""
    if awards and awards.winners:
        triplet = awards.winners[:3]
        winners_summary = "; ".join([f"{w.category}: {w.winner}" for w in triplet if w.category and w.winner])
    await evaluator.verify(
        claim=f"The awards ceremony winners include: {winners_summary}.",
        node=winners_ref_leaf,
        sources=sources,
        additional_instruction="Verify that the listed winners for their categories match official or reputable sources."
    )


async def verify_tv_host_achievement(evaluator: Evaluator, parent_node, tv: Optional[TVHostInfo]) -> None:
    item_node = evaluator.add_parallel(
        id="item_3_tv_host_achievement",
        desc="A television host with a long-running show on a major U.S. network who has been with that network for at least 30 years",
        parent=parent_node,
        critical=False
    )
    sources = _safe_list(tv.reference_urls if tv else [])

    # Basic info
    basic_node = evaluator.add_parallel(
        id="host_basic_info",
        desc="Basic information about the television host including name and show title",
        parent=item_node,
        critical=True
    )

    # Host name
    host_name_node = evaluator.add_parallel(
        id="host_name",
        desc="The full name of the television host is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(tv and tv.host_name and tv.host_name.strip()),
        id="host_name_provided",
        desc="Host name is present",
        parent=host_name_node,
        critical=True
    )
    host_name_ref_leaf = evaluator.add_leaf(
        id="host_name_reference",
        desc="A reference URL confirming the host name is provided",
        parent=host_name_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The television host's full name is '{tv.host_name if tv else ''}'.",
        node=host_name_ref_leaf,
        sources=sources,
        additional_instruction="Confirm the host's full name via network or official bios."
    )

    # Show title
    show_node = evaluator.add_parallel(
        id="show_title",
        desc="The official title of the host's show is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(tv and tv.show_title and tv.show_title.strip()),
        id="show_title_provided",
        desc="Show title is present",
        parent=show_node,
        critical=True
    )
    show_ref_leaf = evaluator.add_leaf(
        id="show_title_reference",
        desc="A reference URL confirming the show title is provided",
        parent=show_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official show title is '{tv.show_title if tv else ''}'.",
        node=show_ref_leaf,
        sources=sources,
        additional_instruction="Confirm the show title via network pages or official listings."
    )

    # Network
    network_node = evaluator.add_parallel(
        id="network",
        desc="The network that airs the show is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(tv and tv.network and tv.network.strip()),
        id="network_provided",
        desc="Network name is present",
        parent=network_node,
        critical=True
    )
    network_ref_leaf = evaluator.add_leaf(
        id="network_reference",
        desc="A reference URL confirming the network is provided",
        parent=network_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The show '{tv.show_title if tv else ''}' airs on {tv.network if tv else ''}.",
        node=network_ref_leaf,
        sources=sources,
        additional_instruction="Verify that the show is aired on the stated major U.S. network."
    )

    # Network tenure
    tenure_node = evaluator.add_parallel(
        id="network_tenure",
        desc="The host has been with the network for at least 30 years",
        parent=item_node,
        critical=True
    )
    start_year_seq = evaluator.add_sequential(
        id="start_year",
        desc="The year the host joined the network is provided",
        parent=tenure_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(tv and tv.host_join_year and tv.host_join_year.strip()),
        id="start_year_provided",
        desc="Host's network join year is present",
        parent=start_year_seq,
        critical=True
    )
    tenure_calc_leaf = evaluator.add_leaf(
        id="tenure_calculation",
        desc="The tenure from the start year to 2026 is at least 30 years",
        parent=start_year_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"The host's tenure from {tv.host_join_year if tv else ''} to 2026 is at least 30 years.",
        node=tenure_calc_leaf,
        additional_instruction="Subtract start year from 2026; result must be ≥ 30. Handle reasonable formatting."
    )
    tenure_ref_leaf = evaluator.add_leaf(
        id="tenure_reference",
        desc="A reference URL confirming the host's network tenure is provided",
        parent=start_year_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"The host joined {tv.network if tv else ''} in {tv.host_join_year if tv else ''}.",
        node=tenure_ref_leaf,
        sources=sources,
        additional_instruction="Confirm the host's join year or tenure duration on network or official bios."
    )

    # Show longevity
    longevity_node = evaluator.add_parallel(
        id="show_longevity",
        desc="The show has been on air for at least 15 years",
        parent=item_node,
        critical=True
    )
    premiere_seq = evaluator.add_sequential(
        id="show_premiere",
        desc="The year the show premiered is provided",
        parent=longevity_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(tv and tv.show_premiere_year and tv.show_premiere_year.strip()),
        id="show_premiere_provided",
        desc="Show premiere year is present",
        parent=premiere_seq,
        critical=True
    )
    show_duration_leaf = evaluator.add_leaf(
        id="show_duration",
        desc="The show has been running for at least 15 years as of 2026",
        parent=premiere_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"The show has been running for at least 15 years since its premiere year {tv.show_premiere_year if tv else ''} through 2026.",
        node=show_duration_leaf,
        additional_instruction="Subtract premiere year from 2026; result must be ≥ 15."
    )
    longevity_ref_leaf = evaluator.add_leaf(
        id="longevity_reference",
        desc="A reference URL confirming the show's longevity is provided",
        parent=premiere_seq,
        critical=True
    )
    await evaluator.verify(
        claim=f"The show '{tv.show_title if tv else ''}' premiered in {tv.show_premiere_year if tv else ''}.",
        node=longevity_ref_leaf,
        sources=sources,
        additional_instruction="Confirm the premiere year via official network pages or reputable coverage."
    )


async def verify_foundation(evaluator: Evaluator, parent_node, fnd: Optional[FoundationInfo]) -> None:
    item_node = evaluator.add_parallel(
        id="item_4_nonprofit_foundation",
        desc="A nonprofit foundation founded by a public figure focused on victim advocacy with multiple established programs",
        parent=parent_node,
        critical=False
    )
    sources = _safe_list(fnd.reference_urls if fnd else [])

    # Basic info
    basic_node = evaluator.add_parallel(
        id="foundation_basic_info",
        desc="Basic information about the foundation including name, founder, and founding year",
        parent=item_node,
        critical=True
    )

    # Foundation name
    fname_node = evaluator.add_parallel(
        id="foundation_name",
        desc="The official name of the foundation is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fnd and fnd.name and fnd.name.strip()),
        id="foundation_name_provided",
        desc="Foundation name is present",
        parent=fname_node,
        critical=True
    )
    fname_ref_leaf = evaluator.add_leaf(
        id="foundation_name_reference",
        desc="A reference URL confirming the foundation name is provided",
        parent=fname_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official foundation name is '{fnd.name if fnd else ''}'.",
        node=fname_ref_leaf,
        sources=sources,
        additional_instruction="Confirm the foundation's official name via its site or official registry pages."
    )

    # Founder name
    founder_node = evaluator.add_parallel(
        id="founder_name",
        desc="The name of the founder is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fnd and fnd.founder and fnd.founder.strip()),
        id="founder_provided",
        desc="Founder name is present",
        parent=founder_node,
        critical=True
    )
    founder_ref_leaf = evaluator.add_leaf(
        id="founder_reference",
        desc="A reference URL confirming the founder is provided",
        parent=founder_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The founder of the foundation is '{fnd.founder if fnd else ''}'.",
        node=founder_ref_leaf,
        sources=sources,
        additional_instruction="Confirm founder identity via official foundation pages."
    )

    # Founding year
    fyear_node = evaluator.add_parallel(
        id="founding_year",
        desc="The year the foundation was founded is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fnd and fnd.founding_year and fnd.founding_year.strip()),
        id="founding_year_provided",
        desc="Founding year is present",
        parent=fyear_node,
        critical=True
    )
    fyear_ref_leaf = evaluator.add_leaf(
        id="founding_year_reference",
        desc="A reference URL confirming the founding year is provided",
        parent=fyear_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The foundation was founded in {fnd.founding_year if fnd else ''}.",
        node=fyear_ref_leaf,
        sources=sources,
        additional_instruction="Confirm the founding year via reliable sources."
    )

    # Mission
    mission_node = evaluator.add_parallel(
        id="foundation_mission",
        desc="The foundation's primary mission focuses on supporting survivors of violence or abuse",
        parent=item_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fnd and fnd.mission and fnd.mission.strip()),
        id="mission_statement",
        desc="A description of the foundation's mission related to victim/survivor support is provided",
        parent=mission_node,
        critical=True
    )
    mission_ref_leaf = evaluator.add_leaf(
        id="mission_reference",
        desc="A reference URL confirming the foundation's mission is provided",
        parent=mission_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The foundation's mission focuses on supporting survivors of violence or abuse: {fnd.mission if fnd and fnd.mission else ''}.",
        node=mission_ref_leaf,
        sources=sources,
        additional_instruction="Confirm mission statements explicitly mention survivor/victim support or related services."
    )

    # Programs
    programs_node = evaluator.add_parallel(
        id="foundation_programs",
        desc="The foundation operates at least three distinct programs with documented impact",
        parent=item_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fnd and fnd.programs and len([p for p in fnd.programs if p.name]) >= 3),
        id="program_count",
        desc="At least three distinct program names are provided",
        parent=programs_node,
        critical=True
    )
    program_details_node = evaluator.add_parallel(
        id="program_details",
        desc="For each program, a brief description of its purpose or activities is provided",
        parent=programs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(
            fnd and all((p.name and p.description and p.description.strip()) for p in fnd.programs[:3])
        ),
        id="program_details_provided",
        desc="At least three programs include names and descriptions",
        parent=program_details_node,
        critical=True
    )
    metrics_leaf = evaluator.add_leaf(
        id="program_metrics",
        desc="For at least one program, quantifiable impact metrics (such as participants served, satisfaction rates, or reach) are provided",
        parent=program_details_node,
        critical=True
    )
    metrics_text = ""
    if fnd and fnd.programs:
        for p in fnd.programs:
            if p.metrics:
                metrics_text = p.metrics
                break
    await evaluator.verify(
        claim=f"At least one program includes quantifiable metrics: '{metrics_text}'.",
        node=metrics_leaf,
        additional_instruction="Judge quantifiable if it contains numbers, percentages, counts, or measurable figures."
    )
    programs_ref_leaf = evaluator.add_leaf(
        id="programs_reference",
        desc="A reference URL confirming the foundation's programs is provided",
        parent=program_details_node,
        critical=True
    )
    program_summary = ""
    if fnd and fnd.programs:
        program_summary = "; ".join([f"{(p.name or '').strip()}: {(p.description or '').strip()}" for p in fnd.programs[:3]])
    await evaluator.verify(
        claim=f"The foundation operates programs such as: {program_summary}.",
        node=programs_ref_leaf,
        sources=sources,
        additional_instruction="Confirm program names and descriptions on the foundation's site or official materials."
    )

    # Age
    age_node = evaluator.add_parallel(
        id="foundation_age",
        desc="The foundation has been operating for at least 10 years as of 2026",
        parent=item_node,
        critical=True
    )
    age_calc_leaf = evaluator.add_leaf(
        id="age_calculation",
        desc="The time from founding year to 2026 is at least 10 years",
        parent=age_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The foundation age from {fnd.founding_year if fnd else ''} to 2026 is at least 10 years.",
        node=age_calc_leaf,
        additional_instruction="Compute 2026 - founding_year; result must be ≥ 10."
    )
    age_ref_leaf = evaluator.add_leaf(
        id="age_reference",
        desc="A reference URL confirming the foundation's age is provided",
        parent=age_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The foundation was founded in {fnd.founding_year if fnd else ''}.",
        node=age_ref_leaf,
        sources=sources,
        additional_instruction="Confirm founding year or historical timeline in reliable sources."
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
    Evaluate the provided answer for the entertainment industry items (2025-2026).
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Items evaluated independently
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

    # IMPORTANT: Root must be non-critical to allow partial credit across items
    root.critical = False

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_all_items(),
        template_class=AllItemsExtraction,
        extraction_name="items_extraction"
    )

    # Build verification subtrees
    await verify_music_festival(evaluator, root, extracted.festival)
    await verify_awards_ceremony(evaluator, root, extracted.awards)
    await verify_tv_host_achievement(evaluator, root, extracted.tv_host)
    await verify_foundation(evaluator, root, extracted.foundation)

    # Summary
    return evaluator.get_summary()