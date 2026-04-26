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
TASK_ID = "live_performance_planning_2026"
TASK_DESCRIPTION = """You are planning to attend three different types of live performances in 2026: a hip-hop concert, a country music concert, and a Broadway show. For each performance, provide comprehensive information to help with planning your attendance.

Requirements:

1. Hip-Hop Concert: Identify one concert from J. Cole's "The Fall-Off Tour" 2026 scheduled after February 20, 2026. For this concert, provide:
   - The venue name, complete street address, and confirmation that it is an indoor arena with capacity for at least 15,000 people
   - The specific performance date (month, day, year) and confirmation it is part of "The Fall-Off Tour"
   - An official ticketing platform name and a direct link to purchase tickets
   - A URL reference from an official source (tour website, venue website, Ticketmaster, or Live Nation) confirming the venue, date, and ticketing information

2. Country Concert: Identify one concert from Ella Langley's "The Dandelion Tour" 2026 scheduled after February 20, 2026. For this concert, provide:
   - The venue name, complete street address, and venue type
   - The specific performance date (month, day, year) and confirmation it is part of "The Dandelion Tour"
   - An official ticketing platform name and a direct link to purchase tickets
   - The names of any opening acts or special guests scheduled for that specific performance date (if applicable)
   - A URL reference from an official source confirming the venue, date, and ticketing information

3. Broadway Show: Identify one Broadway show performance in New York City scheduled after February 20, 2026. For this show, provide:
   - The show title and the theater name
   - The theater's complete street address in Manhattan (including street, Manhattan/New York, NY, and zip code)
   - Confirmation that the theater is located in Manhattan's Theater District (between West 41st-54th Streets and 6th-8th Avenues)
   - At least one specific performance date available after February 20, 2026
   - An official ticketing platform name and a direct link to purchase tickets
   - The approximate seating capacity of the theater
   - A URL reference from an official source confirming the show, theater, schedule, and ticketing information

For each performance, all information must be verifiable through official sources such as tour websites, venue websites, Ticketmaster, Live Nation, Broadway.com, or other official ticketing platforms.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HipHopConcertExtraction(BaseModel):
    # Venue
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    venue_address: Optional[str] = None
    venue_type: Optional[str] = None  # e.g., "indoor arena"
    venue_capacity: Optional[str] = None  # e.g., "18,000"
    venue_urls: List[str] = Field(default_factory=list)

    # Date / Tour
    performance_date: Optional[str] = None  # e.g., "March 5, 2026"
    tour_name: Optional[str] = None  # e.g., "The Fall-Off Tour"
    date_urls: List[str] = Field(default_factory=list)

    # Ticketing
    ticket_platform: Optional[str] = None  # e.g., Ticketmaster
    ticket_url: Optional[str] = None
    ticket_urls: List[str] = Field(default_factory=list)  # Optional extra ticket refs


class CountryConcertExtraction(BaseModel):
    # Venue
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    venue_address: Optional[str] = None
    venue_type: Optional[str] = None  # arena/theater/other
    venue_urls: List[str] = Field(default_factory=list)

    # Date / Tour
    performance_date: Optional[str] = None
    tour_name: Optional[str] = None  # e.g., "The Dandelion Tour"
    date_urls: List[str] = Field(default_factory=list)

    # Ticketing
    ticket_platform: Optional[str] = None
    ticket_url: Optional[str] = None
    ticket_urls: List[str] = Field(default_factory=list)

    # Opening acts
    opening_acts: List[str] = Field(default_factory=list)
    opening_acts_urls: List[str] = Field(default_factory=list)


class BroadwayShowExtraction(BaseModel):
    # Show and Theater
    show_title: Optional[str] = None
    theater_name: Optional[str] = None
    theater_info_urls: List[str] = Field(default_factory=list)

    # Theater Address and Location
    theater_address: Optional[str] = None  # Complete address including Manhattan/New York, NY and zip
    address_urls: List[str] = Field(default_factory=list)
    location_urls: List[str] = Field(default_factory=list)  # optional refs for district confirmation

    # Performance Schedule
    performance_date: Optional[str] = None
    schedule_urls: List[str] = Field(default_factory=list)

    # Ticketing
    ticket_platform: Optional[str] = None  # Ticketmaster/Broadway.com/TodayTix/Telecharge, etc.
    ticket_url: Optional[str] = None
    ticket_urls: List[str] = Field(default_factory=list)

    # Capacity
    theater_capacity: Optional[str] = None
    capacity_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hiphop_concert() -> str:
    return """
    Extract details for ONE hip-hop concert from J. Cole's "The Fall-Off Tour" 2026 that is scheduled after February 20, 2026, as presented in the answer.
    If multiple are provided, select the FIRST one after the specified date.
    Fields to extract:
    - venue_name: Official venue name
    - venue_city: City of the venue
    - venue_address: Full street address (street, city, state, zip)
    - venue_type: Venue type description (e.g., "indoor arena")
    - venue_capacity: The stated capacity (approximate) if included
    - venue_urls: All official URL(s) confirming venue info (venue site, Ticketmaster, Live Nation, or tour website)
    - performance_date: Specific date (month day, year) for this concert
    - tour_name: Tour name associated with this concert (expect "The Fall-Off Tour")
    - date_urls: All official URL(s) confirming date and tour info
    - ticket_platform: Official ticketing platform name (e.g., Ticketmaster, Live Nation, SeatGeek, venue box office)
    - ticket_url: Direct URL to purchase tickets for this specific concert
    - ticket_urls: Any additional official ticketing URL(s) if provided

    Rules:
    - Extract only what is explicitly present in the answer.
    - For all URL lists, include only valid URLs; return empty lists if none.
    - If any field is missing, return null for that field.
    """


def prompt_extract_country_concert() -> str:
    return """
    Extract details for ONE country concert from Ella Langley's "The Dandelion Tour" 2026 that is scheduled after February 20, 2026, as presented in the answer.
    If multiple are provided, select the FIRST one after the specified date.
    Fields to extract:
    - venue_name
    - venue_city
    - venue_address: Full street address (street, city, state, zip)
    - venue_type: Venue type specification (arena, theater, other)
    - venue_urls: Official URL(s) confirming venue info
    - performance_date: Specific date (month day, year)
    - tour_name: Tour name (expect "The Dandelion Tour")
    - date_urls: Official URL(s) confirming date and tour info
    - ticket_platform: Official ticketing platform name
    - ticket_url: Direct ticket purchase URL
    - ticket_urls: Any additional official ticketing URL(s)
    - opening_acts: Names of opening acts or special guests for that date (if provided)
    - opening_acts_urls: Official URL(s) confirming opening acts for that date

    Rules:
    - Extract only explicit info from the answer.
    - For URLs, include only valid ones. If none, return empty lists.
    - If a field is missing, return null (or empty list for lists).
    """


def prompt_extract_broadway_show() -> str:
    return """
    Extract details for ONE Broadway show performance in New York City scheduled after February 20, 2026, as presented in the answer.
    If multiple are provided, select the FIRST one after the specified date.
    Fields to extract:
    - show_title: Official title of the Broadway show
    - theater_name: Official name of the theater
    - theater_info_urls: Official URL(s) confirming show and theater info (e.g., Broadway.com, official show site, Ticketmaster, Telecharge)
    - theater_address: Complete street address including Manhattan/New York, NY and zip code
    - address_urls: Official URL(s) confirming the theater's address
    - location_urls: Official URL(s) that help confirm the theater is in Manhattan’s Theater District
    - performance_date: At least one specific performance date after Feb 20, 2026
    - schedule_urls: Official URL(s) confirming the performance schedule
    - ticket_platform: Official ticketing platform name (Ticketmaster, Broadway.com, TodayTix, Telecharge, etc.)
    - ticket_url: Direct ticket purchase URL
    - ticket_urls: Any additional official ticketing URL(s)
    - theater_capacity: Approximate seating capacity value if provided
    - capacity_urls: Official URL(s) confirming the theater's seating capacity

    Rules:
    - Extract only explicit info from the answer.
    - Return null for missing scalar fields and empty lists for missing URL lists.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


# --------------------------------------------------------------------------- #
# Verification functions: Hip-Hop Concert                                     #
# --------------------------------------------------------------------------- #
async def verify_hip_hop_concert(evaluator: Evaluator, parent_node, info: HipHopConcertExtraction) -> None:
    # Hip-Hop Concert root
    hip_node = evaluator.add_parallel(
        id="Hip_Hop_Concert",
        desc="A hip-hop concert from J. Cole's 'The Fall-Off Tour' 2026 scheduled after February 20, 2026",
        parent=parent_node,
        critical=False
    )

    # Venue Information (Critical)
    venue_group = evaluator.add_parallel(
        id="hiphop_venue_information",
        desc="Complete venue details for the hip-hop concert",
        parent=hip_node,
        critical=True
    )
    venue_exists = evaluator.add_custom_node(
        result=_non_empty_str(info.venue_name) and _non_empty_str(info.venue_city) and _non_empty_str(info.venue_address) and _has_urls(info.venue_urls),
        id="hiphop_venue_existence",
        desc="Hip-hop venue fields and at least one official venue URL are provided",
        parent=venue_group,
        critical=True
    )

    # Venue name and city
    vn_city = evaluator.add_leaf(
        id="hiphop_venue_name_and_city",
        desc="The official name of the venue and the city where it is located",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The concert's venue is {info.venue_name} located in {info.venue_city}.",
        node=vn_city,
        sources=info.venue_urls,
        additional_instruction="Verify on official venue/ticketing/tour pages. Minor formatting differences in names are acceptable."
    )

    # Full venue address
    vn_addr = evaluator.add_leaf(
        id="hiphop_full_venue_address",
        desc="The complete street address of the venue including street, city, state, and zip code",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue's complete address is '{info.venue_address}'.",
        node=vn_addr,
        sources=info.venue_urls,
        additional_instruction="Confirm the full address string matches the official source(s)."
    )

    # Arena type and capacity
    vn_type_cap = evaluator.add_leaf(
        id="hiphop_arena_type_and_capacity",
        desc="The venue is an indoor arena with capacity for at least 15,000 people",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue {info.venue_name} is an indoor arena and has a capacity of at least 15,000.",
        node=vn_type_cap,
        sources=info.venue_urls,
        additional_instruction="Accept if the venue type indicates indoor arena and capacity stated is ≥ 15,000 on an official page (venue site, Ticketmaster, Live Nation, or tour site)."
    )

    # Venue URL reference (official)
    vn_url_ref = evaluator.add_leaf(
        id="hiphop_venue_url_reference",
        desc="A valid URL from an official source (venue website, Ticketmaster, Live Nation, or tour website) that confirms the venue information",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of these URLs is an official source for the venue information (venue site, Ticketmaster, Live Nation, or official tour website).",
        node=vn_url_ref,
        sources=info.venue_urls,
        additional_instruction="Judge officialness by domain (e.g., ticketmaster.com, livenation.com, official venue domain, or official tour domain)."
    )

    # Performance Date and Time (Critical)
    date_group = evaluator.add_parallel(
        id="hiphop_date_and_tour",
        desc="The scheduled date and confirmation of tour participation",
        parent=hip_node,
        critical=True
    )
    date_exists = evaluator.add_custom_node(
        result=_non_empty_str(info.performance_date) and _has_urls(info.date_urls),
        id="hiphop_date_existence",
        desc="Performance date and at least one official date/tour URL are provided",
        parent=date_group,
        critical=True
    )

    # Specific performance date
    date_specific = evaluator.add_leaf(
        id="hiphop_specific_performance_date",
        desc="The exact date (month, day, and year) when the concert is scheduled",
        parent=date_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The concert is scheduled on {info.performance_date}.",
        node=date_specific,
        sources=info.date_urls,
        additional_instruction="Confirm the date on official date/tour/venue/ticketing sources for this specific event."
    )

    # Date is after Feb 20, 2026 (simple logic)
    date_after = evaluator.add_leaf(
        id="hiphop_date_after_2026_02_20",
        desc="The performance date must be after February 20, 2026",
        parent=date_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The date '{info.performance_date}' is after February 20, 2026.",
        node=date_after,
        additional_instruction="Treat months spelled out. Compare calendar dates (MM/DD/YYYY equivalently)."
    )

    # Confirm part of Fall-Off Tour
    tour_confirm = evaluator.add_leaf(
        id="hiphop_part_of_fall_off_tour",
        desc="The concert is confirmed to be part of J. Cole's 'The Fall-Off Tour' 2026",
        parent=date_group,
        critical=True
    )
    await evaluator.verify(
        claim="This concert is part of J. Cole's 'The Fall-Off Tour' (2026).",
        node=tour_confirm,
        sources=info.date_urls,
        additional_instruction="Look for explicit mention of 'The Fall-Off Tour' on official pages."
    )

    # Date URL reference officialness
    date_url_ref = evaluator.add_leaf(
        id="hiphop_date_url_reference",
        desc="A valid URL from an official source that confirms the performance date and tour information",
        parent=date_group,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of these URLs is an official source for date and tour info.",
        node=date_url_ref,
        sources=info.date_urls,
        additional_instruction="Judge officialness by domain (venue site, Ticketmaster, Live Nation, or official tour website)."
    )

    # Ticketing Information (Critical)
    ticket_group = evaluator.add_parallel(
        id="hiphop_ticketing_information",
        desc="Information about where and how to purchase tickets",
        parent=hip_node,
        critical=True
    )
    ticket_exists = evaluator.add_custom_node(
        result=_non_empty_str(info.ticket_platform) and _non_empty_str(info.ticket_url),
        id="hiphop_ticket_existence",
        desc="Ticketing platform and direct purchase URL are provided",
        parent=ticket_group,
        critical=True
    )

    # Official ticketing platform
    ticket_platform_node = evaluator.add_leaf(
        id="hiphop_official_ticketing_platform",
        desc="Name of an official ticketing platform (e.g., Ticketmaster, Live Nation, SeatGeek, venue box office)",
        parent=ticket_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official ticketing platform is {info.ticket_platform}.",
        node=ticket_platform_node,
        sources=info.ticket_url,
        additional_instruction="Confirm the platform name from the ticket purchase page. Accept recognized official platforms (Ticketmaster, Live Nation, SeatGeek, venue box office)."
    )

    # Direct ticket purchase link
    ticket_link_node = evaluator.add_leaf(
        id="hiphop_direct_ticket_purchase_link",
        desc="A direct URL link from an official ticketing platform or venue website to purchase tickets for this specific concert",
        parent=ticket_group,
        critical=True
    )
    await evaluator.verify(
        claim="This URL sells tickets for this specific concert date/venue.",
        node=ticket_link_node,
        sources=info.ticket_url,
        additional_instruction="The page should be a purchase flow for the concert (not a generic information page)."
    )


# --------------------------------------------------------------------------- #
# Verification functions: Country Concert                                     #
# --------------------------------------------------------------------------- #
ALLOWED_OPENING_ACTS = {
    "Kaitlin Butts", "Gabriella Rose", "Kameron Marlowe", "Dylan Marlowe", "Laci Kaye Booth"
}


async def verify_country_concert(evaluator: Evaluator, parent_node, info: CountryConcertExtraction) -> None:
    country_node = evaluator.add_parallel(
        id="Country_Concert",
        desc="A country music concert from Ella Langley's 'The Dandelion Tour' 2026 scheduled after February 20, 2026",
        parent=parent_node,
        critical=False
    )

    # Venue Information (Critical)
    venue_group = evaluator.add_parallel(
        id="country_venue_information",
        desc="Complete venue details for the country concert",
        parent=country_node,
        critical=True
    )
    venue_exists = evaluator.add_custom_node(
        result=_non_empty_str(info.venue_name) and _non_empty_str(info.venue_city) and _non_empty_str(info.venue_address) and _non_empty_str(info.venue_type) and _has_urls(info.venue_urls),
        id="country_venue_existence",
        desc="Country venue fields and at least one official venue URL are provided",
        parent=venue_group,
        critical=True
    )

    vn_city = evaluator.add_leaf(
        id="country_venue_name_and_city",
        desc="The official name of the venue and the city where it is located",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The concert's venue is {info.venue_name} located in {info.venue_city}.",
        node=vn_city,
        sources=info.venue_urls,
        additional_instruction="Verify on official venue/ticketing/tour pages."
    )

    vn_addr = evaluator.add_leaf(
        id="country_full_venue_address",
        desc="The complete street address of the venue including street, city, state, and zip code",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue's complete address is '{info.venue_address}'.",
        node=vn_addr,
        sources=info.venue_urls,
        additional_instruction="Confirm the address string matches the official source(s)."
    )

    vn_type = evaluator.add_leaf(
        id="country_venue_type_specification",
        desc="Identification of the venue type (arena, theater, or other appropriate concert venue)",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue type is '{info.venue_type}'.",
        node=vn_type,
        sources=info.venue_urls,
        additional_instruction="Confirm the venue type on official pages. Minor wording variations are acceptable."
    )

    vn_url_ref = evaluator.add_leaf(
        id="country_venue_url_reference",
        desc="A valid URL from an official source that confirms the venue information",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of these URLs is an official source for the venue information.",
        node=vn_url_ref,
        sources=info.venue_urls,
        additional_instruction="Judge officialness by domain (venue site, Ticketmaster, Live Nation, or official tour website)."
    )

    # Performance Date and Time (Critical)
    date_group = evaluator.add_parallel(
        id="country_date_and_tour",
        desc="The scheduled date and confirmation of tour participation",
        parent=country_node,
        critical=True
    )
    date_exists = evaluator.add_custom_node(
        result=_non_empty_str(info.performance_date) and _has_urls(info.date_urls),
        id="country_date_existence",
        desc="Performance date and at least one official date/tour URL are provided",
        parent=date_group,
        critical=True
    )

    date_specific = evaluator.add_leaf(
        id="country_specific_performance_date",
        desc="The exact date (month, day, and year) when the concert is scheduled",
        parent=date_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The concert is scheduled on {info.performance_date}.",
        node=date_specific,
        sources=info.date_urls,
        additional_instruction="Confirm the date on official date/tour/venue/ticketing sources for this specific event."
    )

    date_after = evaluator.add_leaf(
        id="country_date_after_2026_02_20",
        desc="The performance date must be after February 20, 2026",
        parent=date_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The date '{info.performance_date}' is after February 20, 2026.",
        node=date_after,
        additional_instruction="Treat months spelled out. Compare calendar dates (MM/DD/YYYY equivalently)."
    )

    tour_confirm = evaluator.add_leaf(
        id="country_part_of_dandelion_tour",
        desc="The concert is confirmed to be part of Ella Langley's 'The Dandelion Tour' 2026",
        parent=date_group,
        critical=True
    )
    await evaluator.verify(
        claim="This concert is part of Ella Langley's 'The Dandelion Tour' (2026).",
        node=tour_confirm,
        sources=info.date_urls,
        additional_instruction="Look for explicit mention of 'The Dandelion Tour' on official pages."
    )

    date_url_ref = evaluator.add_leaf(
        id="country_date_url_reference",
        desc="A valid URL from an official source that confirms the performance date and tour information",
        parent=date_group,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of these URLs is an official source for date and tour info.",
        node=date_url_ref,
        sources=info.date_urls,
        additional_instruction="Judge officialness by domain (venue site, Ticketmaster, Live Nation, or official tour website)."
    )

    # Ticketing Information (Critical)
    ticket_group = evaluator.add_parallel(
        id="country_ticketing_information",
        desc="Information about where and how to purchase tickets",
        parent=country_node,
        critical=True
    )
    ticket_exists = evaluator.add_custom_node(
        result=_non_empty_str(info.ticket_platform) and _non_empty_str(info.ticket_url),
        id="country_ticket_existence",
        desc="Ticketing platform and direct purchase URL are provided",
        parent=ticket_group,
        critical=True
    )

    ticket_platform_node = evaluator.add_leaf(
        id="country_official_ticketing_platform",
        desc="Name of an official ticketing platform",
        parent=ticket_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official ticketing platform is {info.ticket_platform}.",
        node=ticket_platform_node,
        sources=info.ticket_url,
        additional_instruction="Confirm the platform name from the ticket purchase page. Accept recognized official platforms."
    )

    ticket_link_node = evaluator.add_leaf(
        id="country_direct_ticket_purchase_link",
        desc="A direct URL link from an official ticketing platform or venue website to purchase tickets for this specific concert",
        parent=ticket_group,
        critical=True
    )
    await evaluator.verify(
        claim="This URL sells tickets for this specific concert date/venue.",
        node=ticket_link_node,
        sources=info.ticket_url,
        additional_instruction="The page should be a purchase flow for the concert (not a generic information page)."
    )

    # Opening Act Information (Non-Critical)
    opening_group = evaluator.add_parallel(
        id="country_opening_act_information",
        desc="Information about special guests or opening acts if applicable",
        parent=country_node,
        critical=False
    )
    opening_exists = evaluator.add_custom_node(
        result=bool(info.opening_acts) and len(info.opening_acts) > 0,
        id="country_opening_act_exists",
        desc="Opening acts are provided in the answer",
        parent=opening_group,
        critical=True  # Gate following leaves; allowed although parent is non-critical
    )

    opening_names = evaluator.add_leaf(
        id="country_opening_act_names",
        desc="Names of opening acts or special guests scheduled for this specific performance date",
        parent=opening_group,
        critical=False
    )
    await evaluator.verify(
        claim=f"The opening acts for this date include: {', '.join(info.opening_acts)}.",
        node=opening_names,
        sources=info.opening_acts_urls,
        additional_instruction="Confirm the listed opening acts on an official source for the specific date."
    )

    opening_verify = evaluator.add_leaf(
        id="country_opening_act_verification",
        desc="If opening acts are provided, they must be from the confirmed list: Kaitlin Butts, Gabriella Rose, Kameron Marlowe, Dylan Marlowe, or Laci Kaye Booth",
        parent=opening_group,
        critical=False
    )
    await evaluator.verify(
        claim=f"All provided opening acts are among the allowed list: {', '.join(sorted(ALLOWED_OPENING_ACTS))}. Provided acts: {', '.join(info.opening_acts)}.",
        node=opening_verify,
        additional_instruction="Evaluate set inclusion logically; minor name formatting differences are acceptable."
    )

    opening_url_ref = evaluator.add_leaf(
        id="country_opening_act_url_reference",
        desc="A URL from an official source confirming the opening acts for this specific date",
        parent=opening_group,
        critical=False
    )
    await evaluator.verify(
        claim="At least one of these URLs is an official source confirming the opening acts for this date.",
        node=opening_url_ref,
        sources=info.opening_acts_urls,
        additional_instruction="Judge officialness by domain (venue site, Ticketmaster, Live Nation, official artist/tour site)."
    )


# --------------------------------------------------------------------------- #
# Verification functions: Broadway Show                                       #
# --------------------------------------------------------------------------- #
async def verify_broadway_show(evaluator: Evaluator, parent_node, info: BroadwayShowExtraction) -> None:
    broadway_node = evaluator.add_parallel(
        id="Broadway_Show",
        desc="A Broadway show performance in New York City scheduled after February 20, 2026",
        parent=parent_node,
        critical=False
    )

    # Show and Theater Information (Critical)
    show_group = evaluator.add_parallel(
        id="broadway_show_and_theater_info",
        desc="Details about the Broadway show and its theater",
        parent=broadway_node,
        critical=True
    )
    show_exists = evaluator.add_custom_node(
        result=_non_empty_str(info.show_title) and _non_empty_str(info.theater_name) and _has_urls(info.theater_info_urls),
        id="broadway_show_info_existence",
        desc="Show title, theater name, and at least one official info URL are provided",
        parent=show_group,
        critical=True
    )

    show_title_node = evaluator.add_leaf(
        id="broadway_show_title",
        desc="The official title of the Broadway show",
        parent=show_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The show title is '{info.show_title}'.",
        node=show_title_node,
        sources=info.theater_info_urls,
        additional_instruction="Confirm on official sources (Broadway.com, official show site, Ticketmaster, Telecharge)."
    )

    theater_name_node = evaluator.add_leaf(
        id="broadway_theater_name",
        desc="The official name of the Broadway theater where the show is performed",
        parent=show_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater name is '{info.theater_name}'.",
        node=theater_name_node,
        sources=info.theater_info_urls,
        additional_instruction="Confirm theater name on official sources."
    )

    theater_info_url_ref = evaluator.add_leaf(
        id="broadway_theater_info_url_reference",
        desc="A valid URL from an official source confirming the show and theater information",
        parent=show_group,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of these URLs is an official source confirming the show and theater information.",
        node=theater_info_url_ref,
        sources=info.theater_info_urls,
        additional_instruction="Judge officialness by domain (Broadway.com, official show site, Ticketmaster, Telecharge, TodayTix, etc.)."
    )

    # Theater Address and Location (Critical)
    addr_group = evaluator.add_parallel(
        id="broadway_theater_address_and_location",
        desc="Complete address and location verification for the theater",
        parent=broadway_node,
        critical=True
    )
    addr_exists = evaluator.add_custom_node(
        result=_non_empty_str(info.theater_address) and _has_urls(info.address_urls),
        id="broadway_address_existence",
        desc="Theater address and at least one official address URL are provided",
        parent=addr_group,
        critical=True
    )

    complete_addr = evaluator.add_leaf(
        id="broadway_complete_street_address",
        desc="The complete street address of the theater including street number, street name, Manhattan/New York, NY, and zip code",
        parent=addr_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater's complete address is '{info.theater_address}'.",
        node=complete_addr,
        sources=info.address_urls,
        additional_instruction="Confirm the full address on official sources."
    )

    theater_district = evaluator.add_leaf(
        id="broadway_theater_district_location",
        desc="The theater is located in Manhattan's Theater District (between West 41st-54th Streets and 6th-8th Avenues)",
        parent=addr_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater at address '{info.theater_address}' is located in Manhattan's Theater District (between West 41st–54th Streets and 6th–8th Avenues).",
        node=theater_district,
        sources=(info.location_urls if _has_urls(info.location_urls) else info.address_urls),
        additional_instruction="If the official page lists a W 41st–54th Street address near 6th–8th Avenues, consider it within the Theater District. Minor inference from address is acceptable."
    )

    address_url_ref = evaluator.add_leaf(
        id="broadway_address_url_reference",
        desc="A URL from an official source confirming the theater's address",
        parent=addr_group,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of these URLs is an official source confirming the theater's address.",
        node=address_url_ref,
        sources=info.address_urls,
        additional_instruction="Judge officialness by domain (official theater site, Broadway.com, Ticketmaster, Telecharge)."
    )

    # Performance Schedule (Critical)
    schedule_group = evaluator.add_parallel(
        id="broadway_performance_schedule",
        desc="Information about show dates and current run status",
        parent=broadway_node,
        critical=True
    )
    sched_exists = evaluator.add_custom_node(
        result=_non_empty_str(info.performance_date) and _has_urls(info.schedule_urls),
        id="broadway_schedule_existence",
        desc="At least one performance date and schedule URL provided",
        parent=schedule_group,
        critical=True
    )

    avail_date = evaluator.add_leaf(
        id="broadway_available_performance_date",
        desc="At least one specific performance date after February 20, 2026",
        parent=schedule_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"There is a performance on {info.performance_date}.",
        node=avail_date,
        sources=info.schedule_urls,
        additional_instruction="Confirm the specific date on the official schedule or ticketing page."
    )

    after_check = evaluator.add_leaf(
        id="broadway_date_after_2026_02_20",
        desc="At least one specific performance date after February 20, 2026",
        parent=schedule_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The date '{info.performance_date}' is after February 20, 2026.",
        node=after_check,
        additional_instruction="Treat months spelled out. Compare calendar dates (MM/DD/YYYY equivalently)."
    )

    running_2026 = evaluator.add_leaf(
        id="broadway_currently_running_show",
        desc="The show is confirmed to be currently running or scheduled to run in 2026",
        parent=schedule_group,
        critical=True
    )
    await evaluator.verify(
        claim="The show is running or scheduled in 2026.",
        node=running_2026,
        sources=(info.schedule_urls if _has_urls(info.schedule_urls) else info.theater_info_urls),
        additional_instruction="Confirm the schedule shows 2026 performance dates or indicates an ongoing run into 2026."
    )

    schedule_url_ref = evaluator.add_leaf(
        id="broadway_schedule_url_reference",
        desc="A URL from an official source confirming the performance schedule",
        parent=schedule_group,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of these URLs is an official source confirming the performance schedule.",
        node=schedule_url_ref,
        sources=info.schedule_urls,
        additional_instruction="Judge officialness by domain (official show site, Broadway.com, Ticketmaster, Telecharge, TodayTix)."
    )

    # Ticketing Information (Critical)
    ticket_group = evaluator.add_parallel(
        id="broadway_ticketing_information",
        desc="Information about purchasing Broadway tickets",
        parent=broadway_node,
        critical=True
    )
    ticket_exists = evaluator.add_custom_node(
        result=_non_empty_str(info.ticket_platform) and _non_empty_str(info.ticket_url),
        id="broadway_ticket_existence",
        desc="Ticketing platform and direct purchase URL provided",
        parent=ticket_group,
        critical=True
    )

    ticket_platform_node = evaluator.add_leaf(
        id="broadway_official_ticketing_platform",
        desc="Name of an official Broadway ticketing platform (e.g., Ticketmaster, Broadway.com, TodayTix, Telecharge)",
        parent=ticket_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official ticketing platform is {info.ticket_platform}.",
        node=ticket_platform_node,
        sources=info.ticket_url,
        additional_instruction="Confirm the platform name from the ticket purchase page. Accept recognized Broadway ticketing platforms."
    )

    ticket_link_node = evaluator.add_leaf(
        id="broadway_direct_ticket_purchase_link",
        desc="A direct URL link from an official Broadway ticketing source to purchase tickets for this show",
        parent=ticket_group,
        critical=True
    )
    await evaluator.verify(
        claim="This URL sells tickets for this Broadway show.",
        node=ticket_link_node,
        sources=info.ticket_url,
        additional_instruction="The page should be a purchase flow for the show (not a generic info page)."
    )

    # Theater Capacity Information (Non-Critical)
    capacity_group = evaluator.add_parallel(
        id="broadway_theater_capacity_information",
        desc="Seating capacity details for the theater",
        parent=broadway_node,
        critical=False
    )
    capacity_exists = evaluator.add_custom_node(
        result=_non_empty_str(info.theater_capacity),
        id="broadway_capacity_existence",
        desc="The theater capacity is provided in the answer",
        parent=capacity_group,
        critical=True  # Gate sub-checks within this optional group
    )

    stated_capacity = evaluator.add_leaf(
        id="broadway_stated_capacity",
        desc="The approximate seating capacity of the theater",
        parent=capacity_group,
        critical=False
    )
    await evaluator.verify(
        claim=f"The theater's seating capacity is approximately {info.theater_capacity}.",
        node=stated_capacity,
        sources=(info.capacity_urls if _has_urls(info.capacity_urls) else info.theater_info_urls),
        additional_instruction="Confirm a capacity figure on an official source. Approximate numbers are acceptable."
    )

    capacity_range = evaluator.add_leaf(
        id="broadway_theater_range",
        desc="The capacity falls within typical Broadway theater range (approximately 500-2,000 seats)",
        parent=capacity_group,
        critical=False
    )
    await evaluator.verify(
        claim=f"The stated capacity '{info.theater_capacity}' falls within the typical Broadway range of approximately 500–2,000 seats.",
        node=capacity_range,
        additional_instruction="Treat approximate values generously (e.g., 'about 1,500')."
    )

    capacity_url_ref = evaluator.add_leaf(
        id="broadway_capacity_url_reference",
        desc="A URL confirming the theater's seating capacity",
        parent=capacity_group,
        critical=False
    )
    await evaluator.verify(
        claim="At least one of these URLs confirms the theater's seating capacity.",
        node=capacity_url_ref,
        sources=(info.capacity_urls if _has_urls(info.capacity_urls) else info.theater_info_urls),
        additional_instruction="Prefer official theater site, Broadway.com, or official show site if capacity is listed."
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
) -> Dict:
    """
    Evaluate an answer for the live performance attendance planning task (2026).
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

    # Extract structured info for each performance type
    hiphop_info = await evaluator.extract(
        prompt=prompt_extract_hiphop_concert(),
        template_class=HipHopConcertExtraction,
        extraction_name="hiphop_concert"
    )
    country_info = await evaluator.extract(
        prompt=prompt_extract_country_concert(),
        template_class=CountryConcertExtraction,
        extraction_name="country_concert"
    )
    broadway_info = await evaluator.extract(
        prompt=prompt_extract_broadway_show(),
        template_class=BroadwayShowExtraction,
        extraction_name="broadway_show"
    )

    # Build subtree root
    live_root = evaluator.add_parallel(
        id="Live_Performance_Attendance_Planning",
        desc="Identify three different live performance events (one hip-hop concert, one country concert, and one Broadway show) scheduled after February 20, 2026, with complete venue, date, ticketing, and performance information for each",
        parent=root,
        critical=False
    )

    # Verify each category
    await verify_hip_hop_concert(evaluator, live_root, hiphop_info)
    await verify_country_concert(evaluator, live_root, country_info)
    await verify_broadway_show(evaluator, live_root, broadway_info)

    # Return structured summary
    return evaluator.get_summary()