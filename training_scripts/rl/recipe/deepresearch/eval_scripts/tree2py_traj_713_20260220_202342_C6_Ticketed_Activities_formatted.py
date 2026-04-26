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
TASK_ID = "events_2025_2026"
TASK_DESCRIPTION = """
Identify four specific ticketed entertainment events and venues from 2025-2026 that meet the following criteria:

Event 1 - Comedy Show:
- Takes place at Red Rocks Amphitheatre in Morrison, Colorado (capacity: 9,525 seats)
- Occurs on April 29, 2025, with a show start time of 7:30 PM
- Is presented by SeriesFest
- Features a phone-free policy using Yondr pouches
- Has tickets starting at $74 or higher

Event 2 - Comic Convention:
- Planet Comicon Kansas City 2026, March 27-29, 2026
- Held at Kansas City Convention Center (Bartle Hall), 301 West 13th Street, Kansas City, MO
- A celebrity guest offers photo ops in a Ghostbuster jumpsuit priced at $135
- The same celebrity offers a combo package (autograph + selfie) priced at $160
- Identify the celebrity name

Event 3 - Broadway Show:
- Performs at Minskoff Theatre (200 West 45th Street, New York, NY)
- Theater capacity between 1,621 and 1,710 seats
- Show runs through at least June 2026
- Runtime ~2 hours 30 minutes including one intermission
- Digital lottery tickets priced at $35
- Identify the show name

Event 4 - Film Festival Venue:
- Sie FilmCenter at 2510 E Colfax Ave, Denver, CO 80206
- Serves as the festival hub for SeriesFest Season 11 (April 29-May 4, 2025)
- Has three theaters; largest seats 178 guests

For each event/venue, provide all required specific details and valid URL references supporting each piece of information.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ComedyEventInfo(BaseModel):
    venue_name: Optional[str] = None
    venue_city_state: Optional[str] = None  # Expected: "Morrison, Colorado"
    venue_capacity: Optional[str] = None    # Keep as string to accommodate variants
    venue_urls: List[str] = Field(default_factory=list)

    event_date: Optional[str] = None        # Expected: "April 29, 2025"
    show_start_time: Optional[str] = None   # Expected: "7:30 PM"
    date_urls: List[str] = Field(default_factory=list)

    presenter: Optional[str] = None         # Expected: "SeriesFest"
    phone_policy: Optional[str] = None      # Expected mention of "Yondr" pouches
    feature_urls: List[str] = Field(default_factory=list)

    minimum_ticket_price: Optional[str] = None  # e.g., "$74", "$74+ fees"
    pricing_urls: List[str] = Field(default_factory=list)


class ConventionInfo(BaseModel):
    convention_name: Optional[str] = None   # Expected: "Planet Comicon Kansas City 2026"
    convention_dates: Optional[str] = None  # Expected: "March 27-29, 2026"
    info_urls: List[str] = Field(default_factory=list)

    venue_name: Optional[str] = None        # Expected: "Kansas City Convention Center (Bartle Hall)"
    venue_address: Optional[str] = None     # Expected: "301 West 13th Street, Kansas City, MO"
    venue_urls: List[str] = Field(default_factory=list)

    celebrity_name: Optional[str] = None
    celebrity_costume: Optional[str] = None # Expected mention of "Ghostbuster jumpsuit"
    celebrity_urls: List[str] = Field(default_factory=list)

    photo_op_price: Optional[str] = None    # Expected: "$135"
    combo_package_price: Optional[str] = None  # Expected: "$160"
    pricing_urls: List[str] = Field(default_factory=list)


class BroadwayInfo(BaseModel):
    theater_name: Optional[str] = None      # Expected: "Minskoff Theatre"
    address: Optional[str] = None           # Expected: "200 West 45th Street, New York, NY"
    capacity: Optional[str] = None          # e.g., "1,710 seats"
    venue_urls: List[str] = Field(default_factory=list)

    show_name: Optional[str] = None
    show_urls: List[str] = Field(default_factory=list)

    run_through: Optional[str] = None       # e.g., "through June 2026"
    runtime: Optional[str] = None           # e.g., "2 hours 30 minutes including one intermission"
    schedule_urls: List[str] = Field(default_factory=list)

    lottery_price: Optional[str] = None     # Expected: "$35"
    tickets_urls: List[str] = Field(default_factory=list)


class FestivalVenueInfo(BaseModel):
    venue_name: Optional[str] = None        # Expected: "Sie FilmCenter"
    address: Optional[str] = None           # Expected: "2510 E Colfax Ave, Denver, CO 80206"
    venue_urls: List[str] = Field(default_factory=list)

    festival_name: Optional[str] = None     # Expected: "SeriesFest Season 11"
    festival_dates: Optional[str] = None    # Expected: "April 29-May 4, 2025"
    festival_urls: List[str] = Field(default_factory=list)

    total_theaters: Optional[str] = None    # Expected: "3"
    largest_theater_capacity: Optional[str] = None  # Expected: "178"
    capacity_urls: List[str] = Field(default_factory=list)


class EventsExtraction(BaseModel):
    event1_comedy: Optional[ComedyEventInfo] = None
    event2_convention: Optional[ConventionInfo] = None
    event3_broadway: Optional[BroadwayInfo] = None
    event4_festival_venue: Optional[FestivalVenueInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract four groups of structured information from the answer, corresponding to the specified events. For each field, extract exactly what the answer states (do not infer). Also extract URL references explicitly mentioned in the answer for each group; URLs can be plain or markdown links—return the actual URL strings. If a field is missing, set it to null; if no URLs are provided for a group, return an empty list.

    event1_comedy:
      - venue_name
      - venue_city_state
      - venue_capacity
      - venue_urls (URLs that support venue name/location/capacity)
      - event_date
      - show_start_time
      - date_urls (URLs that support date/time)
      - presenter
      - phone_policy (e.g., mentions Yondr pouches / phone-free)
      - feature_urls (URLs that support presenter & phone policy)
      - minimum_ticket_price (e.g., "$74", "$74+ fees")
      - pricing_urls (URLs that support pricing)

    event2_convention:
      - convention_name
      - convention_dates
      - info_urls (URLs that support name & dates)
      - venue_name
      - venue_address
      - venue_urls (URLs that support venue & address)
      - celebrity_name (the guest offering Ghostbuster jumpsuit photo ops)
      - celebrity_costume (e.g., "Ghostbuster jumpsuit")
      - celebrity_urls (URLs that support the guest & costume offering)
      - photo_op_price (e.g., "$135")
      - combo_package_price (e.g., "$160")
      - pricing_urls (URLs that support pricing for photo ops and combo)

    event3_broadway:
      - theater_name
      - address
      - capacity (state as written; a number or phrase is fine)
      - venue_urls (URLs that support theater name/address/capacity)
      - show_name
      - show_urls (URLs that confirm the show at this venue)
      - run_through (e.g., "through June 2026")
      - runtime (e.g., "2 hours 30 minutes including one intermission")
      - schedule_urls (URLs that support run duration and runtime)
      - lottery_price (e.g., "$35")
      - tickets_urls (URLs that support lottery pricing)

    event4_festival_venue:
      - venue_name
      - address
      - venue_urls (URLs that support venue & address)
      - festival_name (e.g., "SeriesFest Season 11")
      - festival_dates (e.g., "April 29-May 4, 2025")
      - festival_urls (URLs that support festival & dates and indicate Sie FilmCenter is the festival hub)
      - total_theaters (e.g., "3")
      - largest_theater_capacity (e.g., "178")
      - capacity_urls (URLs that support total theaters and largest capacity)
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


def normalize_list(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_event_1_comedy(evaluator: Evaluator, parent_node, info: Optional[ComedyEventInfo]) -> None:
    node_event = evaluator.add_parallel(
        id="Event_1_Comedy_Show",
        desc="Identify the comedy event at Red Rocks Amphitheatre on April 29, 2025, presented by SeriesFest",
        parent=parent_node,
        critical=False
    )

    info = info or ComedyEventInfo()

    # Comedy_Venue_Info group
    venue_group = evaluator.add_parallel(
        id="Comedy_Venue_Info",
        desc="Provide correct venue name, location, and capacity for the comedy event",
        parent=node_event,
        critical=False
    )
    urls_present_node = evaluator.add_custom_node(
        result=has_urls(info.venue_urls),
        id="Comedy_Venue_URL",
        desc="Provide a valid URL reference for the venue information",
        parent=venue_group,
        critical=True
    )

    leaf_venue_name = evaluator.add_leaf(
        id="Comedy_Venue_Name",
        desc="The venue must be Red Rocks Amphitheatre in Morrison, Colorado",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim="The venue is Red Rocks Amphitheatre located in Morrison, Colorado.",
        node=leaf_venue_name,
        sources=normalize_list(info.venue_urls),
        additional_instruction="Confirm the venue name and city/state. Minor formatting variations are acceptable."
    )

    leaf_venue_capacity = evaluator.add_leaf(
        id="Comedy_Venue_Capacity",
        desc="The venue capacity must be stated as 9,525 seats",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim="Red Rocks Amphitheatre has a capacity of 9,525 seats.",
        node=leaf_venue_capacity,
        sources=normalize_list(info.venue_urls),
        additional_instruction="Accept equivalent phrasing such as 'capacity 9525' or '9,525 attendees'."
    )

    # Comedy_Date_Time group
    date_group = evaluator.add_parallel(
        id="Comedy_Date_Time",
        desc="Provide correct event date and show start time",
        parent=node_event,
        critical=False
    )
    date_urls_present = evaluator.add_custom_node(
        result=has_urls(info.date_urls),
        id="Comedy_Date_URL",
        desc="Provide a valid URL reference for the date and time information",
        parent=date_group,
        critical=True
    )

    leaf_date = evaluator.add_leaf(
        id="Comedy_Date",
        desc="The event date must be April 29, 2025 (Tuesday)",
        parent=date_group,
        critical=True
    )
    await evaluator.verify(
        claim="The event date is April 29, 2025 (Tuesday).",
        node=leaf_date,
        sources=normalize_list(info.date_urls),
        additional_instruction="Day-of-week should be Tuesday for April 29, 2025; allow minor formatting variants."
    )

    leaf_time = evaluator.add_leaf(
        id="Comedy_Show_Time",
        desc="The show start time must be 7:30 PM",
        parent=date_group,
        critical=True
    )
    await evaluator.verify(
        claim="The show start time is 7:30 PM.",
        node=leaf_time,
        sources=normalize_list(info.date_urls),
        additional_instruction="Accept variants like '7:30pm' or '7:30 PM MT'."
    )

    # Comedy_Presenter_Features group
    features_group = evaluator.add_parallel(
        id="Comedy_Presenter_Features",
        desc="Identify the presenting organization and special event features",
        parent=node_event,
        critical=False
    )
    features_urls_present = evaluator.add_custom_node(
        result=has_urls(info.feature_urls),
        id="Comedy_Features_URL",
        desc="Provide a valid URL reference for presenter and special features information",
        parent=features_group,
        critical=True
    )

    leaf_presenter = evaluator.add_leaf(
        id="Comedy_Presenter",
        desc="The event must be presented by SeriesFest",
        parent=features_group,
        critical=True
    )
    await evaluator.verify(
        claim="The event is presented by SeriesFest.",
        node=leaf_presenter,
        sources=normalize_list(info.feature_urls),
        additional_instruction="Look for explicit mention of 'presented by SeriesFest'."
    )

    leaf_phone_policy = evaluator.add_leaf(
        id="Comedy_Phone_Policy",
        desc="The event must have a phone-free policy using Yondr pouches",
        parent=features_group,
        critical=True
    )
    await evaluator.verify(
        claim="The event uses a phone-free policy with Yondr pouches.",
        node=leaf_phone_policy,
        sources=normalize_list(info.feature_urls),
        additional_instruction="Confirm the use of Yondr pouches or equivalent phone-free enforcement."
    )

    # Comedy_Pricing group
    pricing_group = evaluator.add_parallel(
        id="Comedy_Pricing",
        desc="Provide correct ticket pricing information",
        parent=node_event,
        critical=False
    )
    pricing_urls_present = evaluator.add_custom_node(
        result=has_urls(info.pricing_urls),
        id="Comedy_Pricing_URL",
        desc="Provide a valid URL reference for ticket pricing information",
        parent=pricing_group,
        critical=True
    )

    leaf_min_price = evaluator.add_leaf(
        id="Comedy_Minimum_Price",
        desc="Tickets must start at $74 or higher",
        parent=pricing_group,
        critical=True
    )
    await evaluator.verify(
        claim="Tickets start at $74 or higher.",
        node=leaf_min_price,
        sources=normalize_list(info.pricing_urls),
        additional_instruction="Accept variants like 'starting at $74', '$74+ fees', or higher amounts meeting the threshold."
    )


async def verify_event_2_convention(evaluator: Evaluator, parent_node, info: Optional[ConventionInfo]) -> None:
    node_event = evaluator.add_parallel(
        id="Event_2_Comic_Convention",
        desc="Identify the comic convention event where a celebrity offers Ghostbuster jumpsuit photo ops in March 2026",
        parent=parent_node,
        critical=False
    )

    info = info or ConventionInfo()

    # Convention_Info group
    info_group = evaluator.add_parallel(
        id="Convention_Info",
        desc="Provide correct convention name and dates",
        parent=node_event,
        critical=False
    )
    info_urls_present = evaluator.add_custom_node(
        result=has_urls(info.info_urls),
        id="Convention_Info_URL",
        desc="Provide a valid URL reference for convention information",
        parent=info_group,
        critical=True
    )

    leaf_name = evaluator.add_leaf(
        id="Convention_Name",
        desc="The convention must be Planet Comicon Kansas City 2026",
        parent=info_group,
        critical=True
    )
    await evaluator.verify(
        claim="The convention is Planet Comicon Kansas City 2026.",
        node=leaf_name,
        sources=normalize_list(info.info_urls),
        additional_instruction="Confirm the exact event branding for the 2026 edition."
    )

    leaf_dates = evaluator.add_leaf(
        id="Convention_Dates",
        desc="The convention dates must be March 27-29, 2026",
        parent=info_group,
        critical=True
    )
    await evaluator.verify(
        claim="The convention dates are March 27–29, 2026.",
        node=leaf_dates,
        sources=normalize_list(info.info_urls),
        additional_instruction="Accept M/D/YYYY or 'March 27-29, 2026' style formatting."
    )

    # Convention_Venue group
    venue_group = evaluator.add_parallel(
        id="Convention_Venue",
        desc="Provide correct venue name and complete address",
        parent=node_event,
        critical=False
    )
    venue_urls_present = evaluator.add_custom_node(
        result=has_urls(info.venue_urls),
        id="Convention_Venue_URL",
        desc="Provide a valid URL reference for venue information",
        parent=venue_group,
        critical=True
    )

    leaf_venue_name = evaluator.add_leaf(
        id="Convention_Venue_Name",
        desc="The venue must be Kansas City Convention Center (Bartle Hall)",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim="The venue is Kansas City Convention Center (Bartle Hall).",
        node=leaf_venue_name,
        sources=normalize_list(info.venue_urls),
        additional_instruction="Accept synonyms like 'Bartle Hall within the Kansas City Convention Center'."
    )

    leaf_address = evaluator.add_leaf(
        id="Convention_Address",
        desc="The address must be 301 West 13th Street, Kansas City, MO",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim="The venue address is 301 West 13th Street, Kansas City, MO.",
        node=leaf_address,
        sources=normalize_list(info.venue_urls),
        additional_instruction="Accept variants like '301 W 13th St' and inclusion of ZIP code."
    )

    # Celebrity_Guest group
    celebrity_group = evaluator.add_parallel(
        id="Celebrity_Guest",
        desc="Identify the celebrity guest and their photo op costume offering",
        parent=node_event,
        critical=False
    )
    celeb_urls_present = evaluator.add_custom_node(
        result=has_urls(info.celebrity_urls),
        id="Celebrity_Guest_URL",
        desc="Provide a valid URL reference for celebrity guest information",
        parent=celebrity_group,
        critical=True
    )

    leaf_celeb_name = evaluator.add_leaf(
        id="Celebrity_Name",
        desc="Identify the celebrity offering Ghostbuster jumpsuit photo ops",
        parent=celebrity_group,
        critical=True
    )
    celeb_name = info.celebrity_name or ""
    await evaluator.verify(
        claim=f"The celebrity guest offering Ghostbuster jumpsuit photo ops is {celeb_name}.",
        node=leaf_celeb_name,
        sources=normalize_list(info.celebrity_urls),
        additional_instruction="Confirm the same celebrity is explicitly tied to the Ghostbuster jumpsuit photo op offering."
    )

    leaf_celeb_costume = evaluator.add_leaf(
        id="Celebrity_Photo_Costume",
        desc="The celebrity must offer photo ops in a Ghostbuster jumpsuit",
        parent=celebrity_group,
        critical=True
    )
    await evaluator.verify(
        claim="The celebrity offers photo ops in a Ghostbuster jumpsuit.",
        node=leaf_celeb_costume,
        sources=normalize_list(info.celebrity_urls),
        additional_instruction="Look for explicit mention of the costume (Ghostbuster jumpsuit) as part of photo ops."
    )

    # Convention_Pricing group
    pricing_group = evaluator.add_parallel(
        id="Convention_Pricing",
        desc="Provide correct pricing for photo ops and combo packages",
        parent=node_event,
        critical=False
    )
    pricing_urls_present = evaluator.add_custom_node(
        result=has_urls(info.pricing_urls),
        id="Convention_Pricing_URL",
        desc="Provide a valid URL reference for pricing information",
        parent=pricing_group,
        critical=True
    )

    leaf_photo_price = evaluator.add_leaf(
        id="Photo_Op_Price",
        desc="The Ghostbuster jumpsuit photo op price must be $135",
        parent=pricing_group,
        critical=True
    )
    await evaluator.verify(
        claim="The price for the Ghostbuster jumpsuit photo op is $135.",
        node=leaf_photo_price,
        sources=normalize_list(info.pricing_urls),
        additional_instruction="Price must be $135 for the specific Ghostbuster jumpsuit photo op."
    )

    leaf_combo_price = evaluator.add_leaf(
        id="Combo_Package_Price",
        desc="The combo package (autograph + selfie) price must be $160",
        parent=pricing_group,
        critical=True
    )
    await evaluator.verify(
        claim="The combo package (autograph + selfie) price is $160.",
        node=leaf_combo_price,
        sources=normalize_list(info.pricing_urls),
        additional_instruction="Confirm the combo that includes autograph + selfie is priced at $160."
    )


async def verify_event_3_broadway(evaluator: Evaluator, parent_node, info: Optional[BroadwayInfo]) -> None:
    node_event = evaluator.add_parallel(
        id="Event_3_Broadway_Show",
        desc="Identify the Broadway show at Minskoff Theatre with digital lottery tickets and long-running schedule through 2026",
        parent=parent_node,
        critical=False
    )

    info = info or BroadwayInfo()

    # Broadway_Venue group
    venue_group = evaluator.add_parallel(
        id="Broadway_Venue",
        desc="Provide correct theater name, address, and seating capacity",
        parent=node_event,
        critical=False
    )
    venue_urls_present = evaluator.add_custom_node(
        result=has_urls(info.venue_urls),
        id="Broadway_Venue_URL",
        desc="Provide a valid URL reference for theater information",
        parent=venue_group,
        critical=True
    )

    leaf_theatre_name = evaluator.add_leaf(
        id="Broadway_Theater_Name",
        desc="The theater must be Minskoff Theatre",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim="The theater is Minskoff Theatre.",
        node=leaf_theatre_name,
        sources=normalize_list(info.venue_urls),
        additional_instruction="Confirm the venue name is Minskoff Theatre."
    )

    leaf_address = evaluator.add_leaf(
        id="Broadway_Address",
        desc="The address must be 200 West 45th Street, New York, NY",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim="The theater address is 200 West 45th Street, New York, NY.",
        node=leaf_address,
        sources=normalize_list(info.venue_urls),
        additional_instruction="Accept variants like '200 W 45th St' and inclusion of ZIP code."
    )

    leaf_capacity = evaluator.add_leaf(
        id="Broadway_Capacity",
        desc="The seating capacity must be between 1,621 and 1,710 seats",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim="The seating capacity of Minskoff Theatre is within the range 1,621 to 1,710 seats.",
        node=leaf_capacity,
        sources=normalize_list(info.venue_urls),
        additional_instruction="If a specific number within the range (e.g., 1,710) is stated, consider it as satisfying the range condition."
    )

    # Broadway_Show_Identity group
    show_group = evaluator.add_parallel(
        id="Broadway_Show_Identity",
        desc="Identify the specific show running at the venue",
        parent=node_event,
        critical=False
    )
    show_urls_present = evaluator.add_custom_node(
        result=has_urls(info.show_urls),
        id="Broadway_Show_URL",
        desc="Provide a valid URL reference for show information",
        parent=show_group,
        critical=True
    )

    leaf_show_name = evaluator.add_leaf(
        id="Broadway_Show_Name",
        desc="Identify the name of the Broadway show",
        parent=show_group,
        critical=True
    )
    show_name = info.show_name or ""
    await evaluator.verify(
        claim=f"The Broadway show at the Minskoff Theatre is '{show_name}'.",
        node=leaf_show_name,
        sources=normalize_list(info.show_urls),
        additional_instruction="Confirm the show name explicitly associated with Minskoff Theatre."
    )

    # Broadway_Schedule group
    schedule_group = evaluator.add_parallel(
        id="Broadway_Schedule",
        desc="Provide correct show run duration and performance length",
        parent=node_event,
        critical=False
    )
    schedule_urls_present = evaluator.add_custom_node(
        result=has_urls(info.schedule_urls),
        id="Broadway_Schedule_URL",
        desc="Provide a valid URL reference for schedule information",
        parent=schedule_group,
        critical=True
    )

    leaf_run = evaluator.add_leaf(
        id="Broadway_Run_Duration",
        desc="The show must run through at least June 2026",
        parent=schedule_group,
        critical=True
    )
    await evaluator.verify(
        claim="The show's schedule indicates performances through at least June 2026.",
        node=leaf_run,
        sources=normalize_list(info.schedule_urls),
        additional_instruction="Pages listing schedule or calendar with dates reaching or beyond June 2026 should be accepted."
    )

    leaf_runtime = evaluator.add_leaf(
        id="Broadway_Performance_Length",
        desc="The show runtime must be approximately 2 hours 30 minutes including one intermission",
        parent=schedule_group,
        critical=True
    )
    await evaluator.verify(
        claim="The show runtime is approximately 2 hours and 30 minutes including one intermission.",
        node=leaf_runtime,
        sources=normalize_list(info.schedule_urls),
        additional_instruction="Accept variants like '2h 30m' and phrasing indicating one intermission."
    )

    # Broadway_Tickets group
    tickets_group = evaluator.add_parallel(
        id="Broadway_Tickets",
        desc="Provide correct digital lottery ticket price",
        parent=node_event,
        critical=False
    )
    tickets_urls_present = evaluator.add_custom_node(
        result=has_urls(info.tickets_urls),
        id="Broadway_Tickets_URL",
        desc="Provide a valid URL reference for ticket lottery information",
        parent=tickets_group,
        critical=True
    )

    leaf_lottery = evaluator.add_leaf(
        id="Broadway_Lottery_Price",
        desc="The digital lottery ticket price must be $35",
        parent=tickets_group,
        critical=True
    )
    await evaluator.verify(
        claim="Digital lottery tickets are priced at $35.",
        node=leaf_lottery,
        sources=normalize_list(info.tickets_urls),
        additional_instruction="Confirm the specific lottery price; accept variants like '$35 per ticket' or '$35 digital lottery'."
    )


async def verify_event_4_festival_venue(evaluator: Evaluator, parent_node, info: Optional[FestivalVenueInfo]) -> None:
    node_event = evaluator.add_parallel(
        id="Event_4_Film_Festival_Venue",
        desc="Identify the film festival venue serving as the hub for SeriesFest Season 11 screenings in Denver",
        parent=parent_node,
        critical=False
    )

    info = info or FestivalVenueInfo()

    # Festival_Venue_Info group
    venue_group = evaluator.add_parallel(
        id="Festival_Venue_Info",
        desc="Provide correct venue name and complete address",
        parent=node_event,
        critical=False
    )
    venue_urls_present = evaluator.add_custom_node(
        result=has_urls(info.venue_urls),
        id="Festival_Venue_URL",
        desc="Provide a valid URL reference for venue information",
        parent=venue_group,
        critical=True
    )

    leaf_venue_name = evaluator.add_leaf(
        id="Festival_Venue_Name",
        desc="The venue must be Sie FilmCenter",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim="The venue is Sie FilmCenter.",
        node=leaf_venue_name,
        sources=normalize_list(info.venue_urls),
        additional_instruction="Confirm the venue name."
    )

    leaf_venue_address = evaluator.add_leaf(
        id="Festival_Venue_Address",
        desc="The address must be 2510 E Colfax Ave, Denver, CO 80206",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim="The venue address is 2510 E Colfax Ave, Denver, CO 80206.",
        node=leaf_venue_address,
        sources=normalize_list(info.venue_urls),
        additional_instruction="Accept 'E' vs 'East' and minor formatting variations; ZIP must be 80206."
    )

    # Festival_Details group
    details_group = evaluator.add_parallel(
        id="Festival_Details",
        desc="Provide correct festival name and date range",
        parent=node_event,
        critical=False
    )
    details_urls_present = evaluator.add_custom_node(
        result=has_urls(info.festival_urls),
        id="Festival_Details_URL",
        desc="Provide a valid URL reference for festival information",
        parent=details_group,
        critical=True
    )

    leaf_festival_name = evaluator.add_leaf(
        id="Festival_Name",
        desc="The festival must be SeriesFest Season 11",
        parent=details_group,
        critical=True
    )
    await evaluator.verify(
        claim="The festival is SeriesFest Season 11.",
        node=leaf_festival_name,
        sources=normalize_list(info.festival_urls),
        additional_instruction="Prefer pages indicating Sie FilmCenter as the festival hub for SeriesFest Season 11."
    )

    leaf_festival_dates = evaluator.add_leaf(
        id="Festival_Dates",
        desc="The festival dates must be April 29-May 4, 2025",
        parent=details_group,
        critical=True
    )
    await evaluator.verify(
        claim="The festival dates are April 29–May 4, 2025.",
        node=leaf_festival_dates,
        sources=normalize_list(info.festival_urls),
        additional_instruction="Accept en-dash or hyphen; ensure year 2025."
    )

    # Venue_Capacity group
    capacity_group = evaluator.add_parallel(
        id="Venue_Capacity",
        desc="Provide correct information about the venue's theater capacity",
        parent=node_event,
        critical=False
    )
    capacity_urls_present = evaluator.add_custom_node(
        result=has_urls(info.capacity_urls),
        id="Venue_Capacity_URL",
        desc="Provide a valid URL reference for capacity information",
        parent=capacity_group,
        critical=True
    )

    leaf_total_theaters = evaluator.add_leaf(
        id="Total_Theaters",
        desc="The venue must have three theaters",
        parent=capacity_group,
        critical=True
    )
    await evaluator.verify(
        claim="Sie FilmCenter has three theaters.",
        node=leaf_total_theaters,
        sources=normalize_list(info.capacity_urls),
        additional_instruction="Look for venue overview or specs stating theater count."
    )

    leaf_largest_capacity = evaluator.add_leaf(
        id="Largest_Theater_Capacity",
        desc="The largest theater must seat 178 guests",
        parent=capacity_group,
        critical=True
    )
    await evaluator.verify(
        claim="The largest theater at Sie FilmCenter seats 178 guests.",
        node=leaf_largest_capacity,
        sources=normalize_list(info.capacity_urls),
        additional_instruction="Accept '178 seats' phrasing or equivalent."
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
    Evaluate an answer for the events_2025_2026 task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel aggregation
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

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction"
    )

    # Ground truth / constraints for reference in summary (not used for scoring directly)
    evaluator.add_ground_truth({
        "event_1_expected": {
            "venue": "Red Rocks Amphitheatre, Morrison, CO",
            "capacity": "9,525",
            "date": "April 29, 2025",
            "time": "7:30 PM",
            "presenter": "SeriesFest",
            "phone_policy": "Yondr pouches",
            "min_price": ">= $74"
        },
        "event_2_expected": {
            "name": "Planet Comicon Kansas City 2026",
            "dates": "March 27–29, 2026",
            "venue": "Kansas City Convention Center (Bartle Hall)",
            "address": "301 West 13th Street, Kansas City, MO",
            "photo_op_price": "$135",
            "combo_price": "$160"
        },
        "event_3_expected": {
            "theatre": "Minskoff Theatre",
            "address": "200 West 45th Street, New York, NY",
            "capacity_range": "1,621–1,710",
            "run_through": "≥ June 2026",
            "runtime": "≈ 2h30m incl. 1 intermission",
            "lottery_price": "$35"
        },
        "event_4_expected": {
            "venue": "Sie FilmCenter",
            "address": "2510 E Colfax Ave, Denver, CO 80206",
            "festival": "SeriesFest Season 11",
            "dates": "April 29–May 4, 2025",
            "theaters": "3",
            "largest_capacity": "178"
        }
    }, gt_type="constraints")

    # Build and verify the four event subtrees
    await verify_event_1_comedy(evaluator, root, extracted.event1_comedy)
    await verify_event_2_convention(evaluator, root, extracted.event2_convention)
    await verify_event_3_broadway(evaluator, root, extracted.event3_broadway)
    await verify_event_4_festival_venue(evaluator, root, extracted.event4_festival_venue)

    return evaluator.get_summary()