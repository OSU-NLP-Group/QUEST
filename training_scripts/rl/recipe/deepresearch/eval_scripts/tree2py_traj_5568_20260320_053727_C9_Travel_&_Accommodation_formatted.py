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
TASK_ID = "family_cruise_2026_plan"
TASK_DESCRIPTION = """
You are planning a multi-destination family vacation in 2026 for 6 people that combines a Philippine cruise experience, a Tennessee theme park visit, and UK heritage tourism. Provide detailed travel arrangements for the following four components:

1. Subic Bay Pre-Cruise Hotel (Philippines)
Identify a hotel located within the Subic Bay Freeport Zone that meets these requirements:
- Accessible from the cruise ship terminal (Alava Wharf or Rivera Wharf)
- Within 2 km of Ocean Adventure marine theme park OR offers shuttle service
- Provides family-appropriate accommodations (minimum 2 beds or family room)
- Has standard amenities (WiFi, air conditioning, breakfast options)

2. Dollywood Resort Accommodation (Tennessee, USA)
Identify an official Dollywood resort property in Pigeon Forge, Tennessee that meets these requirements:
- Must be either Dollywood's DreamMore Resort & Spa OR Dollywood's HeartSong Lodge & Resort
- Room must sleep 6 people (Family Suite, Junior Suite, or equivalent)
- Must include complimentary TimeSaver passes for all registered guests
- Provide the name of Dollywood's longest roller coaster and its track length

3. Newark to Caribbean Flights (USA)
Identify a round-trip JetBlue flight route from Newark Liberty International Airport that meets these requirements:
- Must depart from Terminal A at Newark airport (where JetBlue operates)
- Must be a direct/nonstop flight to a Caribbean destination
- Caribbean destination must be served by JetBlue from Newark

4. Stonehenge Special Access (United Kingdom)
Provide details for booking the Stonehenge Stone Circle Experience that meets these requirements:
- Must be the Stone Circle Experience (inner circle access, not standard admission)
- Provide the correct adult ticket price (age 18+)
- Provide the correct child ticket price (age 5-17)
- Confirm the access timing occurs outside normal visiting hours (before 9:30am or after 5pm/7pm)

For each component, provide:
- The specific hotel name / resort name / destination / experience name
- Key specifications that satisfy the stated requirements
- Official reference URL(s) supporting your answer
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SubicHotel(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    near_cruise_terminal_note: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)

    ocean_adventure_distance_km: Optional[str] = None
    has_shuttle_service: Optional[str] = None  # "yes"/"no"/"unknown"
    transportation_option: Optional[str] = None  # e.g., "walking", "shuttle", "taxi"
    proximity_urls: List[str] = Field(default_factory=list)

    room_type: Optional[str] = None
    bed_configuration: Optional[str] = None
    amenities_wifi: Optional[str] = None  # "yes"/"no"/"unknown"
    amenities_aircon: Optional[str] = None
    breakfast_available: Optional[str] = None
    amenities_urls: List[str] = Field(default_factory=list)


class DollywoodResort(BaseModel):
    name: Optional[str] = None  # DreamMore Resort & Spa OR HeartSong Lodge & Resort
    location_city: Optional[str] = None  # Expect "Pigeon Forge"
    resort_urls: List[str] = Field(default_factory=list)  # dollywood.com resort landing page(s)

    room_type: Optional[str] = None  # Family Suite / Junior Suite / equivalent
    sleeps_count: Optional[str] = None  # textual, e.g., "sleeps 6"
    bed_configuration: Optional[str] = None
    room_urls: List[str] = Field(default_factory=list)  # specific room page(s)

    timesaver_included_text: Optional[str] = None
    perks_urls: List[str] = Field(default_factory=list)  # perks/benefits page(s) on dollywood.com

    longest_coaster_name: Optional[str] = None  # Expect "Big Bear Mountain"
    longest_coaster_length: Optional[str] = None  # Expect "3,990 feet"
    coaster_urls: List[str] = Field(default_factory=list)  # Big Bear Mountain page at dollywood.com


class NewarkFlight(BaseModel):
    airline: Optional[str] = None  # JetBlue
    departure_airport: Optional[str] = None  # Newark Liberty International Airport (EWR)
    terminal: Optional[str] = None  # Terminal A
    destination_city: Optional[str] = None  # e.g., San Juan, Aruba, Montego Bay
    is_caribbean_destination: Optional[str] = None  # "yes"/"no"/"unknown"
    is_nonstop: Optional[str] = None  # "yes"/"no"/"unknown"

    terminal_urls: List[str] = Field(default_factory=list)  # jetblue.com or newarkairport.com page
    route_urls: List[str] = Field(default_factory=list)  # jetblue.com route/schedule/destination page(s)


class StonehengeAccess(BaseModel):
    experience_name: Optional[str] = None  # Expect "Stone Circle Experience" or "Stone Circle Access"
    inner_circle_access_text: Optional[str] = None
    experience_urls: List[str] = Field(default_factory=list)  # english-heritage.org.uk experience page(s)

    adult_price: Optional[str] = None  # Expect "£70"
    child_price: Optional[str] = None  # Expect "£40"
    pricing_urls: List[str] = Field(default_factory=list)  # english-heritage.org.uk pricing page(s)

    special_hours_text: Optional[str] = None
    normal_hours_text: Optional[str] = None
    timing_urls: List[str] = Field(default_factory=list)  # english-heritage.org.uk timing/hours page(s)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_subic_hotel() -> str:
    return """
    Extract details for the Subic Bay pre-cruise hotel from the answer. Return fields strictly as mentioned in the answer:

    Required fields:
    - name: exact hotel name used in the answer
    - address: full or partial address as stated in the answer
    - near_cruise_terminal_note: any note that indicates accessibility from Alava Wharf or Rivera Wharf
    - location_urls: array of official hotel website URLs or reputable booking platform URLs cited for the hotel's location/address (extract only URLs explicitly present in the answer)

    Ocean Adventure proximity & transport:
    - ocean_adventure_distance_km: distance value (if the answer gives a number or phrasing like '1.5 km')
    - has_shuttle_service: 'yes'/'no'/'unknown' depending on whether the answer explicitly states a shuttle to Ocean Adventure
    - transportation_option: one of ['walking', 'shuttle', 'taxi', 'rideshare', 'driving'] if mentioned
    - proximity_urls: array of URLs used to support distance or transportation

    Family room & amenities:
    - room_type: the family-appropriate room type mentioned (e.g., 'Family Room', 'Deluxe Room', 'Suite')
    - bed_configuration: text mentioning beds/sleeping arrangements (e.g., '2 queen beds + sofa bed')
    - amenities_wifi: 'yes'/'no'/'unknown' based on whether WiFi is mentioned
    - amenities_aircon: 'yes'/'no'/'unknown' based on whether air conditioning is mentioned
    - breakfast_available: 'yes'/'no'/'unknown' based on whether breakfast options are mentioned
    - amenities_urls: array of URLs cited that show the room/amenities
    """


def prompt_extract_dollywood_resort() -> str:
    return """
    Extract details for the Dollywood resort accommodation from the answer. Use only information explicitly in the answer:

    Resort identity:
    - name: resort name as stated (must be either "Dollywood's DreamMore Resort & Spa" or "Dollywood's HeartSong Lodge & Resort")
    - location_city: the city as stated (should be Pigeon Forge)
    - resort_urls: array of dollywood.com URLs cited for the resort details

    Room capacity:
    - room_type: stated room category (e.g., 'Family Suite', 'Junior Suite', or equivalent)
    - sleeps_count: text that indicates sleeping capacity (e.g., 'sleeps 6')
    - bed_configuration: textual bed layout (e.g., '2 queens + sleeper sofa')
    - room_urls: array of dollywood.com URLs showing this room type

    Guest perks:
    - timesaver_included_text: text confirming TimeSaver passes included for all registered guests
    - perks_urls: array of dollywood.com URLs listing guest benefits/perks

    Theme park facts:
    - longest_coaster_name: stated as Dollywood's longest roller coaster (expected 'Big Bear Mountain')
    - longest_coaster_length: stated track length (expected '3,990 feet' or similar)
    - coaster_urls: array of dollywood.com URLs for the coaster information
    """


def prompt_extract_newark_flight() -> str:
    return """
    Extract details for the JetBlue Newark to Caribbean flight route from the answer. Use only information explicitly in the answer:

    - airline: airline name (should be JetBlue)
    - departure_airport: airport name (should be Newark Liberty International Airport or EWR)
    - terminal: terminal at Newark (should be Terminal A)
    - destination_city: selected Caribbean destination city (e.g., 'San Juan', 'Aruba', 'Montego Bay')
    - is_caribbean_destination: 'yes'/'no'/'unknown' based on the answer's statement
    - is_nonstop: 'yes'/'no'/'unknown' based on the answer's statement about nonstop/direct service

    - terminal_urls: array of jetblue.com or newarkairport.com URLs cited for JetBlue's Newark terminal info
    - route_urls: array of jetblue.com URLs cited that show the Newark-to-destination route and/or nonstop service
    """


def prompt_extract_stonehenge() -> str:
    return """
    Extract details for the Stonehenge Stone Circle Experience from the answer. Use only information explicitly in the answer:

    Access & experience:
    - experience_name: name as stated (should be 'Stone Circle Experience' or 'Stone Circle Access')
    - inner_circle_access_text: text confirming walking among the stones/inner circle access
    - experience_urls: array of english-heritage.org.uk URLs cited that describe the experience

    Pricing:
    - adult_price: the adult (18+) ticket price (expected '£70')
    - child_price: the child (5–17) ticket price (expected '£40')
    - pricing_urls: array of english-heritage.org.uk URLs showing pricing

    Timing:
    - special_hours_text: text confirming the access is outside normal visiting hours (before 9:30am or after 5/7pm)
    - normal_hours_text: text stating normal hours (e.g., 9:30am–5pm winter / 9:30am–7pm summer)
    - timing_urls: array of english-heritage.org.uk URLs confirming the timing information
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_affirmative(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    return any(x in t for x in ["yes", "y", "true", "available", "provided", "included", "offers", "offer"])


def _first_nonempty(*lists: List[str]) -> List[str]:
    for ls in lists:
        if ls:
            return ls
    return []


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_subic_hotel(evaluator: Evaluator, parent_node, hotel: SubicHotel) -> None:
    # Main node for Subic Bay accommodation (non-critical to allow partial credit across components)
    subic_node = evaluator.add_parallel(
        id="subic_bay_accommodation",
        desc="Identify a hotel in Subic Bay Freeport Zone suitable for cruise passengers with Ocean Adventure park access",
        parent=parent_node,
        critical=False
    )

    # 1) Location verification (critical group)
    location_node = evaluator.add_parallel(
        id="location_verification",
        desc="Verify the hotel is located within Subic Bay Freeport Zone and accessible from cruise terminal",
        parent=subic_node,
        critical=True
    )

    # 1.c) URL reference (verify first to serve as prerequisite)
    loc_url_ref_leaf = evaluator.add_leaf(
        id="location_url_reference",
        desc="Provide official website or booking platform URL showing hotel location",
        parent=location_node,
        critical=True
    )
    await evaluator.verify(
        claim="At least one provided page is the hotel's official website or a reputable booking platform listing that shows the hotel's address/location in the Subic Bay area.",
        node=loc_url_ref_leaf,
        sources=hotel.location_urls,
        additional_instruction="Accept domains that are the hotel's official site or major booking platforms (e.g., Booking, Agoda, Expedia, Hotels.com), and ensure the page displays the hotel's location/address."
    )

    # 1.a) Freeport Zone location
    freeport_leaf = evaluator.add_leaf(
        id="freeport_zone_location",
        desc="Confirm hotel address is within Subic Bay Freeport Zone boundaries",
        parent=location_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel's address is located within the Subic Bay Freeport Zone (SBMA) in Olongapo/Subic Bay, Philippines.",
        node=freeport_leaf,
        sources=hotel.location_urls,
        additional_instruction="Look for address strings explicitly containing 'Subic Bay Freeport Zone' or 'SBMA'. If a map/address on the page clearly places the hotel inside SBMA, consider it supported."
    )

    # 1.b) Cruise terminal access (Alava Wharf or Rivera Wharf)
    cruise_access_leaf = evaluator.add_leaf(
        id="cruise_terminal_access",
        desc="Verify hotel is accessible from Alava Wharf or Rivera Wharf where cruise ships dock",
        parent=location_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel is reasonably accessible from Alava Wharf or Rivera Wharf (Subic Bay cruise terminals) by a short drive or walk.",
        node=cruise_access_leaf,
        sources=hotel.location_urls,
        additional_instruction="Accessibility can be inferred if the hotel is located inside Subic Bay Freeport Zone near Olongapo and within a few kilometers (or ~5–15 min drive) to the wharfs; a map or directions on the page are acceptable evidence."
    )

    # 2) Ocean Adventure proximity (critical group)
    prox_node = evaluator.add_parallel(
        id="ocean_adventure_proximity",
        desc="Verify hotel's proximity to Ocean Adventure marine theme park",
        parent=subic_node,
        critical=True
    )

    # 2.c) Proximity URL reference first
    prox_urls = _first_nonempty(hotel.proximity_urls, hotel.location_urls)
    prox_url_ref_leaf = evaluator.add_leaf(
        id="proximity_url_reference",
        desc="Provide URL source confirming distance or transportation availability",
        parent=prox_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page states either the distance between the hotel and Ocean Adventure or indicates shuttle/transportation availability between the hotel and Ocean Adventure.",
        node=prox_url_ref_leaf,
        sources=prox_urls,
        additional_instruction="Look for explicit distance metrics to Ocean Adventure or mentions of shuttle service or practical transport to Ocean Adventure."
    )

    # 2.a) Distance specification (≤2 km OR shuttle service)
    distance_leaf = evaluator.add_leaf(
        id="distance_specification",
        desc="Confirm distance to Ocean Adventure is ≤2 km or shuttle service is available",
        parent=prox_node,
        critical=True
    )
    if _has_affirmative(hotel.has_shuttle_service) or (hotel.transportation_option and "shuttle" in hotel.transportation_option.lower()):
        dist_claim = "The hotel offers a shuttle service to Ocean Adventure marine theme park."
        dist_add_ins = "Confirm that the page mentions a shuttle (or equivalent dedicated transport) between the hotel and Ocean Adventure."
    else:
        reported = hotel.ocean_adventure_distance_km or "unknown"
        dist_claim = f"The distance from the hotel to Ocean Adventure is at most 2 km. Reported distance in the answer: {reported}."
        dist_add_ins = "If a specific distance is shown (e.g., 1.5 km), accept it if ≤ 2 km. If only a walking time is given, accept if it reasonably implies ≤ 2 km."
    await evaluator.verify(
        claim=dist_claim,
        node=distance_leaf,
        sources=prox_urls,
        additional_instruction=dist_add_ins
    )

    # 2.b) Transportation option
    transport_leaf = evaluator.add_leaf(
        id="transportation_option",
        desc="Identify available transportation method (walking, shuttle, or taxi)",
        parent=prox_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"A practical transportation method between the hotel and Ocean Adventure is available as stated (e.g., walking, shuttle, taxi). Stated method: {hotel.transportation_option or 'unspecified'}.",
        node=transport_leaf,
        sources=prox_urls,
        additional_instruction="Accept if the page indicates any reasonable transport such as walking distance, shuttle, taxi, rideshare, or short drive."
    )

    # 3) Hotel amenities (critical group)
    amen_node = evaluator.add_parallel(
        id="hotel_amenities",
        desc="Verify hotel provides essential amenities for cruise passengers",
        parent=subic_node,
        critical=True
    )

    # 3.c) Amenities URL reference first
    amen_url_ref_leaf = evaluator.add_leaf(
        id="amenities_url_reference",
        desc="Provide hotel website or booking platform URL showing amenities",
        parent=amen_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page lists the hotel's room details and/or amenities.",
        node=amen_url_ref_leaf,
        sources=_first_nonempty(hotel.amenities_urls, hotel.location_urls),
        additional_instruction="Accept pages that show room features/amenities tables or bullet lists indicating services and facilities."
    )

    # 3.a) Accommodation type (family-suitable)
    accom_leaf = evaluator.add_leaf(
        id="accommodation_type",
        desc="Specify room type suitable for a family (minimum 2 beds or family room)",
        parent=amen_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel offers a family-appropriate room with at least two beds or a family room. Example cited room: {hotel.room_type or 'unspecified'} with bed configuration: {hotel.bed_configuration or 'unspecified'}.",
        node=accom_leaf,
        sources=_first_nonempty(hotel.amenities_urls, hotel.location_urls),
        additional_instruction="Accept if the room details clearly show at least two beds (e.g., 2 queens) or a family-labeled room/suite suitable for families."
    )

    # 3.b) Standard amenities
    std_amen_leaf = evaluator.add_leaf(
        id="standard_amenities",
        desc="Confirm hotel has WiFi, air conditioning, and breakfast options",
        parent=amen_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel provides WiFi, air conditioning, and breakfast options.",
        node=std_amen_leaf,
        sources=_first_nonempty(hotel.amenities_urls, hotel.location_urls),
        additional_instruction="All three should be present on the page: WiFi (free or paid ok), air conditioning (AC in rooms), and breakfast options (included or available). Minor wording variants acceptable."
    )


async def verify_dollywood_resort(evaluator: Evaluator, parent_node, resort: DollywoodResort) -> None:
    d_node = evaluator.add_parallel(
        id="dollywood_resort_booking",
        desc="Identify appropriate Dollywood resort accommodation for a family of 6 with theme park access perks",
        parent=parent_node,
        critical=False
    )

    # 1) Resort property verification (critical)
    resort_node = evaluator.add_parallel(
        id="resort_property_verification",
        desc="Verify the accommodation is an official Dollywood resort property",
        parent=d_node,
        critical=True
    )

    # 1.c) official website reference first
    off_web_leaf = evaluator.add_leaf(
        id="official_website_reference",
        desc="Provide dollywood.com URL confirming resort details",
        parent=resort_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page on dollywood.com provides official resort information.",
        node=off_web_leaf,
        sources=resort.resort_urls,
        additional_instruction="Ensure the domain is dollywood.com and the page clearly refers to the resort property."
    )

    # 1.a) official resort name check
    official_name_leaf = evaluator.add_leaf(
        id="official_resort_name",
        desc="Confirm resort is either DreamMore Resort & Spa or HeartSong Lodge & Resort",
        parent=resort_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The selected resort '{resort.name or 'unspecified'}' is either Dollywood's DreamMore Resort & Spa or Dollywood's HeartSong Lodge & Resort.",
        node=official_name_leaf,
        sources=resort.resort_urls,
        additional_instruction="Accept if the page shows the resort name exactly or with minor punctuation/casing differences."
    )

    # 1.b) resort location
    resort_loc_leaf = evaluator.add_leaf(
        id="resort_location",
        desc="Verify resort is located in Pigeon Forge, Tennessee",
        parent=resort_node,
        critical=True
    )
    await evaluator.verify(
        claim="The resort is located in Pigeon Forge, Tennessee.",
        node=resort_loc_leaf,
        sources=resort.resort_urls,
        additional_instruction="The page should state 'Pigeon Forge, Tennessee' or similar phrasing confirming the location."
    )

    # 2) Room capacity requirements (critical)
    room_node = evaluator.add_parallel(
        id="room_capacity_requirements",
        desc="Verify room can accommodate family of 6 people",
        parent=d_node,
        critical=True
    )

    # 2.c) capacity URL reference first
    capacity_ref_leaf = evaluator.add_leaf(
        id="capacity_url_reference",
        desc="Provide URL from dollywood.com showing room capacity and configuration",
        parent=room_node,
        critical=True
    )
    await evaluator.verify(
        claim="This dollywood.com page lists the room’s capacity and/or bed configuration.",
        node=capacity_ref_leaf,
        sources=resort.room_urls,
        additional_instruction="The room page should mention 'sleeps' count and/or list bed types, e.g., 2 queens + sleeper sofa."
    )

    # 2.a) room category identification
    room_cat_leaf = evaluator.add_leaf(
        id="room_category_identification",
        desc="Identify specific room type (must be Family Suite, Junior Suite, or equivalent sleeping ≥5-6)",
        parent=room_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The room type '{resort.room_type or 'unspecified'}' is a suite suitable for families (Family Suite, Junior Suite, or equivalent) offered by this resort.",
        node=room_cat_leaf,
        sources=resort.room_urls,
        additional_instruction="Accept if the page indicates the room is a family-oriented suite or equivalent category."
    )

    # 2.b) sleeping arrangement details (explicitly sleeps 6)
    sleep_leaf = evaluator.add_leaf(
        id="sleeping_arrangement_details",
        desc="Specify bed configuration (e.g., 2 queens + sleeper sofa, or king + bunks + sleeper)",
        parent=room_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The room sleeps at least 6 guests and has the stated bed configuration: {resort.bed_configuration or 'unspecified'}. Reported capacity: {resort.sleeps_count or 'unspecified'}.",
        node=sleep_leaf,
        sources=resort.room_urls,
        additional_instruction="The page should state 'sleeps up to 6' (or 6+) or make it clear via the bed configuration that six guests can be accommodated."
    )

    # 3) Guest perks verification (critical)
    perks_node = evaluator.add_parallel(
        id="guest_perks_verification",
        desc="Verify resort guests receive complimentary TimeSaver passes",
        parent=d_node,
        critical=True
    )

    # 3.b) perks URL first
    perks_ref_leaf = evaluator.add_leaf(
        id="perks_url_reference",
        desc="Provide dollywood.com URL listing resort guest benefits and perks",
        parent=perks_node,
        critical=True
    )
    await evaluator.verify(
        claim="This dollywood.com page lists resort guest benefits/perks.",
        node=perks_ref_leaf,
        sources=resort.perks_urls,
        additional_instruction="The page should be a resort benefits/perks page on dollywood.com."
    )

    # 3.a) TimeSaver pass inclusion
    timesaver_leaf = evaluator.add_leaf(
        id="timesaver_pass_inclusion",
        desc="Confirm complimentary TimeSaver passes are included for all registered guests",
        parent=perks_node,
        critical=True
    )
    await evaluator.verify(
        claim="Complimentary TimeSaver passes are included for all registered guests at the selected Dollywood resort.",
        node=timesaver_leaf,
        sources=resort.perks_urls or resort.resort_urls,
        additional_instruction="Focus on perks language indicating complimentary TimeSaver included for all registered guests (park admission may still be required separately)."
    )

    # 4) Theme park facts (critical)
    facts_node = evaluator.add_parallel(
        id="theme_park_facts",
        desc="Provide factual information about Dollywood's roller coasters",
        parent=d_node,
        critical=True
    )

    # 4.c) coaster info URL first
    coaster_ref_leaf = evaluator.add_leaf(
        id="coaster_info_url_reference",
        desc="Provide dollywood.com URL with roller coaster specifications",
        parent=facts_node,
        critical=True
    )
    await evaluator.verify(
        claim="This dollywood.com page provides specifications for Big Bear Mountain (roller coaster).",
        node=coaster_ref_leaf,
        sources=resort.coaster_urls,
        additional_instruction="Ensure the domain is dollywood.com and that the page is about Big Bear Mountain with ride stats/specifications."
    )

    # 4.a) longest coaster identification
    longest_leaf = evaluator.add_leaf(
        id="longest_coaster_identification",
        desc="Identify Dollywood's longest roller coaster by track length",
        parent=facts_node,
        critical=True
    )
    await evaluator.verify(
        claim="Dollywood's longest roller coaster is Big Bear Mountain.",
        node=longest_leaf,
        sources=resort.coaster_urls,
        additional_instruction="Verify that the page states Big Bear Mountain is the park’s longest coaster."
    )

    # 4.b) track length
    length_leaf = evaluator.add_leaf(
        id="coaster_track_length",
        desc="Provide track length of 3,990 feet for Big Bear Mountain",
        parent=facts_node,
        critical=True
    )
    await evaluator.verify(
        claim="Big Bear Mountain's track length is 3,990 feet.",
        node=length_leaf,
        sources=resort.coaster_urls,
        additional_instruction="Allow minor formatting variations (e.g., commas, feet symbol)."
    )


async def verify_newark_flight(evaluator: Evaluator, parent_node, flight: NewarkFlight) -> None:
    f_node = evaluator.add_parallel(
        id="newark_flight_arrangement",
        desc="Book round-trip JetBlue flights from Newark to a Caribbean destination",
        parent=parent_node,
        critical=False
    )

    # 1) Departure airport verification (critical)
    dep_node = evaluator.add_parallel(
        id="departure_airport_verification",
        desc="Verify flights depart from Newark Liberty International Airport Terminal A",
        parent=f_node,
        critical=True
    )

    # 1.b) airport URL reference first
    airport_ref_leaf = evaluator.add_leaf(
        id="airport_url_reference",
        desc="Provide jetblue.com or newarkairport.com URL confirming terminal location",
        parent=dep_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page on jetblue.com or newarkairport.com shows that JetBlue operates from Terminal A at Newark (EWR).",
        node=airport_ref_leaf,
        sources=flight.terminal_urls,
        additional_instruction="Ensure the page clearly lists JetBlue at EWR Terminal A."
    )

    # 1.a) terminal identification
    terminal_leaf = evaluator.add_leaf(
        id="terminal_identification",
        desc="Confirm JetBlue operates from Terminal A at Newark airport",
        parent=dep_node,
        critical=True
    )
    await evaluator.verify(
        claim="JetBlue operates from Terminal A at Newark Liberty International Airport (EWR).",
        node=terminal_leaf,
        sources=flight.terminal_urls,
        additional_instruction="Accept if the page clearly lists JetBlue at Terminal A; minor wording differences ok."
    )

    # 2) Airline route verification (critical)
    route_node = evaluator.add_parallel(
        id="airline_route_verification",
        desc="Verify JetBlue operates direct flights from Newark to chosen Caribbean destination",
        parent=f_node,
        critical=True
    )

    # 2.c) route URL reference first
    route_ref_leaf = evaluator.add_leaf(
        id="route_url_reference",
        desc="Provide jetblue.com URL showing Newark to Caribbean destination route",
        parent=route_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This jetblue.com page shows service between Newark (EWR) and the Caribbean destination {flight.destination_city or 'selected destination'}.",
        node=route_ref_leaf,
        sources=flight.route_urls,
        additional_instruction="Prefer a JetBlue route map, destination page, or schedule page for the Newark to destination route."
    )

    # 2.a) destination selection (served by JetBlue from EWR)
    dest_leaf = evaluator.add_leaf(
        id="destination_selection",
        desc="Select a Caribbean destination served by JetBlue from Newark (e.g., San Juan, Aruba, Jamaica)",
        parent=route_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"JetBlue serves the Caribbean destination {flight.destination_city or 'the selected destination'} from Newark (EWR).",
        node=dest_leaf,
        sources=flight.route_urls,
        additional_instruction="The page should indicate that the chosen destination is served from EWR by JetBlue."
    )

    # 2.b) direct flight confirmation (nonstop)
    nonstop_leaf = evaluator.add_leaf(
        id="direct_flight_confirmation",
        desc="Confirm JetBlue offers nonstop service on this route",
        parent=route_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"There is nonstop/direct JetBlue service between Newark (EWR) and {flight.destination_city or 'the selected Caribbean destination'}.",
        node=nonstop_leaf,
        sources=flight.route_urls,
        additional_instruction="The page should show 'nonstop' or clearly indicate direct service for this route."
    )


async def verify_stonehenge(evaluator: Evaluator, parent_node, st: StonehengeAccess) -> None:
    s_node = evaluator.add_parallel(
        id="stonehenge_special_access",
        desc="Book Stonehenge Stone Circle Experience with inner circle access",
        parent=parent_node,
        critical=False
    )

    # 1) Access type verification (critical)
    access_node = evaluator.add_parallel(
        id="access_type_verification",
        desc="Verify booking is for Stone Circle Experience (inner circle access)",
        parent=s_node,
        critical=True
    )

    # 1.c) experience URL reference first
    exp_url_leaf = evaluator.add_leaf(
        id="experience_url_reference",
        desc="Provide english-heritage.org.uk URL describing Stone Circle Experience",
        parent=access_node,
        critical=True
    )
    await evaluator.verify(
        claim="This english-heritage.org.uk page describes the Stone Circle Experience (Stone Circle Access).",
        node=exp_url_leaf,
        sources=st.experience_urls,
        additional_instruction="Ensure the page is on english-heritage.org.uk and is specifically about Stone Circle Experience."
    )

    # 1.a) experience name confirmation
    exp_name_leaf = evaluator.add_leaf(
        id="experience_name_confirmation",
        desc="Confirm booking is specifically for 'Stone Circle Experience' or 'Stone Circle Access'",
        parent=access_node,
        critical=True
    )
    await evaluator.verify(
        claim="The product is the 'Stone Circle Experience' (also called Stone Circle Access) at Stonehenge.",
        node=exp_name_leaf,
        sources=st.experience_urls,
        additional_instruction="Accept minor naming variants like 'Stone Circle Access' if clearly the same product."
    )

    # 1.b) inner circle access
    inner_access_leaf = evaluator.add_leaf(
        id="inner_circle_access",
        desc="Verify visitors can walk among the stones inside the roped area",
        parent=access_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Stone Circle Experience allows visitors to enter the inner circle and walk among the stones inside the roped area.",
        node=inner_access_leaf,
        sources=st.experience_urls,
        additional_instruction="The page should clearly mention inner circle access or walking among the stones; not standard admission."
    )

    # 2) Pricing verification (critical)
    price_node = evaluator.add_parallel(
        id="pricing_verification",
        desc="Verify correct pricing for Stone Circle Experience tickets",
        parent=s_node,
        critical=True
    )

    # 2.c) pricing URL reference first
    price_ref_leaf = evaluator.add_leaf(
        id="pricing_url_reference",
        desc="Provide english-heritage.org.uk URL showing Stone Circle Experience pricing",
        parent=price_node,
        critical=True
    )
    await evaluator.verify(
        claim="This english-heritage.org.uk page shows Stone Circle Experience pricing.",
        node=price_ref_leaf,
        sources=st.pricing_urls,
        additional_instruction="Ensure the page lists ticket prices for the Stone Circle Experience."
    )

    # 2.a) adult price £70
    adult_leaf = evaluator.add_leaf(
        id="adult_ticket_price",
        desc="Confirm adult (18+) Stone Circle Experience price is £70",
        parent=price_node,
        critical=True
    )
    await evaluator.verify(
        claim="The adult (18+) Stone Circle Experience ticket price is £70.",
        node=adult_leaf,
        sources=st.pricing_urls,
        additional_instruction="Verify currency and amount; accept minor formatting variants (e.g., with/without space)."
    )

    # 2.b) child price £40
    child_leaf = evaluator.add_leaf(
        id="child_ticket_price",
        desc="Confirm child (5-17) Stone Circle Experience price is £40",
        parent=price_node,
        critical=True
    )
    await evaluator.verify(
        claim="The child (ages 5–17) Stone Circle Experience ticket price is £40.",
        node=child_leaf,
        sources=st.pricing_urls,
        additional_instruction="Verify the age band and price; allow minor formatting variants."
    )

    # 3) Access timing details (critical; all children critical to satisfy framework constraints)
    timing_node = evaluator.add_parallel(
        id="access_timing_details",
        desc="Verify special access occurs outside normal visitor hours",
        parent=s_node,
        critical=True
    )

    # 3.c) timing URL reference first
    timing_ref_leaf = evaluator.add_leaf(
        id="timing_url_reference",
        desc="Provide english-heritage.org.uk URL confirming special access timing",
        parent=timing_node,
        critical=True
    )
    timing_sources = _first_nonempty(st.timing_urls, st.experience_urls)
    await evaluator.verify(
        claim="This english-heritage.org.uk page confirms the Stone Circle Experience timing relative to normal opening hours.",
        node=timing_ref_leaf,
        sources=timing_sources,
        additional_instruction="Prefer a page that mentions the experience occurs outside standard opening hours."
    )

    # 3.a) special hours confirmation
    special_leaf = evaluator.add_leaf(
        id="special_hours_confirmation",
        desc="Confirm access is before 9:30am or after normal closing time (5pm winter/7pm summer)",
        parent=timing_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Stone Circle Experience occurs outside normal visiting hours (before 9:30am or after normal closing time such as 5pm in winter or 7pm in summer).",
        node=special_leaf,
        sources=timing_sources,
        additional_instruction="Look for wording that the experience runs early morning or in the evening outside public opening times."
    )

    # 3.b) normal hours reference (set to critical to align with framework constraints)
    normal_hours_leaf = evaluator.add_leaf(
        id="normal_hours_reference",
        desc="Confirm normal Stonehenge visiting hours are 9:30am-5pm (winter) or 9:30am-7pm (summer)",
        parent=timing_node,
        critical=True
    )
    await evaluator.verify(
        claim="Normal Stonehenge visiting hours are approximately 9:30am–5pm (winter) and 9:30am–7pm (summer).",
        node=normal_hours_leaf,
        sources=timing_sources,
        additional_instruction="Minor seasonal variations or specific dates acceptable; the general pattern 9:30–5 (winter) and 9:30–7 (summer) should be evident."
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
    Evaluate an answer for the multi-destination family vacation plan (2026) task.
    """
    evaluator = Evaluator()
    # Note: Root is set to non-critical to satisfy framework constraint (critical parents cannot have non-critical children)
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

    # Extract all components in parallel
    subic_task = evaluator.extract(
        prompt=prompt_extract_subic_hotel(),
        template_class=SubicHotel,
        extraction_name="subic_bay_hotel"
    )
    dolly_task = evaluator.extract(
        prompt=prompt_extract_dollywood_resort(),
        template_class=DollywoodResort,
        extraction_name="dollywood_resort"
    )
    flight_task = evaluator.extract(
        prompt=prompt_extract_newark_flight(),
        template_class=NewarkFlight,
        extraction_name="newark_flight"
    )
    stone_task = evaluator.extract(
        prompt=prompt_extract_stonehenge(),
        template_class=StonehengeAccess,
        extraction_name="stonehenge_experience"
    )

    subic, dolly, flight, stone = await asyncio.gather(subic_task, dolly_task, flight_task, stone_task)

    # Build four main component subtrees (parallel)
    # 1) Subic hotel
    await verify_subic_hotel(evaluator, root, subic or SubicHotel())

    # 2) Dollywood resort
    await verify_dollywood_resort(evaluator, root, dolly or DollywoodResort())

    # 3) Newark flights
    await verify_newark_flight(evaluator, root, flight or NewarkFlight())

    # 4) Stonehenge special access
    await verify_stonehenge(evaluator, root, stone or StonehengeAccess())

    # Return summary
    return evaluator.get_summary()