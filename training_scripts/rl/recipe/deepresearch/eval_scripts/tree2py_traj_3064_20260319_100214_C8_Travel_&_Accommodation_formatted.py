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
TASK_ID = "us_hotels_multi_requirements"
TASK_DESCRIPTION = """
I am planning a comprehensive travel guide and need to identify 4 specific hotels/resorts across the United States, each serving different travel purposes and meeting detailed requirements. Please provide the name of one hotel/resort for each category below:

1. Florida Gulf Coast Beachfront Resort:
- Must be located on St. Pete Beach, Florida, with direct Gulf of Mexico beachfront access
- Must span at least 20 tropical acres along the beach
- Must have at least 3 outdoor heated swimming pools
- Must feature a multi-story waterslide (at least 3 stories tall)
- Must have at least 5 on-site food and beverage outlets
- Must provide at least 200 beach loungers for guests
- Must have at least 1,000 feet of beachfront shoreline
- Must have at least 300 guest rooms and accommodations
- Must offer water-based activities such as paddleboats or floating water park features

2. Orlando Disney Springs Resort Area Hotel:
- Must be located in the Disney Springs Resort Area in Orlando, Florida
- Must be within walking distance (half mile or less) of Disney Springs
- Must provide free shuttle transportation to Walt Disney World theme parks
- Must offer early theme park entry as a guest benefit
- Must provide complimentary standard parking at Disney theme parks as a benefit
- Must have at least 150 guest rooms
- Must offer early access to book Disney dining reservations
- Must have an outdoor swimming pool and fitness center
- Standard check-in time must be 3:00 PM or later

3. Whistler Ski Resort Conference Hotel:
- Must be located in Whistler, British Columbia, at the base of Whistler or Blackcomb Mountain
- Must have at least 15,000 square feet of total meeting and event space
- Must have at least 10 separate meeting rooms or function spaces
- The hotel's largest ballroom must accommodate at least 500 people
- Must have at least 300 guest rooms
- Must be within walking distance of ski lift access
- Must have at least 2 on-site restaurants
- Must have an on-site spa facility
- Should ideally hold a sustainability certification such as Green Key

4. Southern California Six Flags Magic Mountain Hotel:
- Must be located in Valencia or Santa Clarita, California
- Must be within 1.5 miles of Six Flags Magic Mountain's entrance
- Should be within walking distance (10-15 minute walk) of Six Flags Magic Mountain
- Must be easily accessible from Interstate 5
- Must have at least 100 guest rooms
- Must have an outdoor swimming pool
- Must have a fitness center on-site
- Must have an on-site restaurant or dining option
- Check-in time must be 3:00 PM

For each hotel, provide the complete hotel name and a reference URL that confirms the hotel meets the specified criteria.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FloridaResortInfo(BaseModel):
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    acres: Optional[str] = None
    outdoor_heated_pools_count: Optional[str] = None
    waterslide_description: Optional[str] = None
    dining_outlets_count: Optional[str] = None
    beach_loungers_count: Optional[str] = None
    shoreline_length_feet: Optional[str] = None
    room_count: Optional[str] = None
    water_activities: Optional[str] = None


class OrlandoDisneyHotelInfo(BaseModel):
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    distance_to_disney_springs: Optional[str] = None
    free_shuttle: Optional[str] = None
    early_theme_park_entry: Optional[str] = None
    complimentary_theme_park_parking: Optional[str] = None
    room_count: Optional[str] = None
    early_dining_booking: Optional[str] = None
    has_pool: Optional[str] = None
    has_fitness_center: Optional[str] = None
    check_in_time: Optional[str] = None


class WhistlerConferenceHotelInfo(BaseModel):
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    at_base_of_mountain: Optional[str] = None
    meeting_space_total_sqft: Optional[str] = None
    meeting_rooms_count: Optional[str] = None
    largest_ballroom_capacity: Optional[str] = None
    room_count: Optional[str] = None
    walking_distance_to_lifts: Optional[str] = None
    onsite_restaurants_count: Optional[str] = None
    has_spa: Optional[str] = None
    sustainability_certification: Optional[str] = None


class SixFlagsHotelInfo(BaseModel):
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    city: Optional[str] = None
    distance_to_six_flags_miles: Optional[str] = None
    walking_distance: Optional[str] = None
    interstate5_access: Optional[str] = None
    room_count: Optional[str] = None
    has_pool: Optional[str] = None
    has_fitness_center: Optional[str] = None
    has_restaurant: Optional[str] = None
    check_in_time: Optional[str] = None


class TravelHotelsExtraction(BaseModel):
    florida_beachfront: Optional[FloridaResortInfo] = None
    orlando_disney: Optional[OrlandoDisneyHotelInfo] = None
    whistler_conference: Optional[WhistlerConferenceHotelInfo] = None
    sixflags_hotel: Optional[SixFlagsHotelInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    Extract the four requested hotel/resort candidates and the specific data points mentioned for each category as they appear in the answer. Do NOT invent any data. If a field is not present, set it to null. Always extract any URLs explicitly mentioned in the answer text for each category.

    The output must be a JSON object with keys:
    - florida_beachfront
    - orlando_disney
    - whistler_conference
    - sixflags_hotel

    For each key, extract the following fields:
    Common:
      - name: full hotel/resort name (string)
      - reference_urls: list of one or more URLs that the answer cites for this hotel

    florida_beachfront (St. Pete Beach resort):
      - location: textual location description if present
      - acres: stated acreage of property (string as written)
      - outdoor_heated_pools_count: number if stated or text fragment
      - waterslide_description: description mentioning multi-story slide if present
      - dining_outlets_count: number of on-site food and beverage outlets if stated or text fragment
      - beach_loungers_count: stated number of loungers/cabanas/chairs if given
      - shoreline_length_feet: shoreline length in feet as stated (string)
      - room_count: stated number of rooms/accommodations (string)
      - water_activities: text about paddleboats, floating water park, or similar

    orlando_disney (Disney Springs Resort Area hotel):
      - location: textual area description if present
      - distance_to_disney_springs: distance or phrase like "across the street" (string)
      - free_shuttle: text describing free shuttle to Disney theme parks if present
      - early_theme_park_entry: text confirming Early Theme Park Entry if present
      - complimentary_theme_park_parking: text about complimentary standard parking at Disney theme parks, if stated
      - room_count: stated rooms count (string)
      - early_dining_booking: text about early access to book Disney dining reservations if present
      - has_pool: text about outdoor swimming pool if present
      - has_fitness_center: text about fitness center if present
      - check_in_time: stated standard check-in time (string, e.g., "3:00 PM")

    whistler_conference:
      - location: textual location description
      - at_base_of_mountain: text indicating base of Whistler or Blackcomb (e.g., "ski-in/ski-out")
      - meeting_space_total_sqft: stated total meeting space (string, e.g., "32,000 sq. ft.")
      - meeting_rooms_count: stated count of meeting rooms (string)
      - largest_ballroom_capacity: stated capacity of largest ballroom (string)
      - room_count: stated rooms count (string)
      - walking_distance_to_lifts: text indicating proximity to ski lifts by walking
      - onsite_restaurants_count: number of on-site restaurants if stated or text
      - has_spa: text indicating on-site spa
      - sustainability_certification: text mentioning certifications like Green Key, etc.

    sixflags_hotel (Valencia/Santa Clarita near Six Flags Magic Mountain):
      - city: stated city (Valencia or Santa Clarita)
      - distance_to_six_flags_miles: stated distance in miles or textual equivalent
      - walking_distance: text indicating 10–15 minute walk if stated
      - interstate5_access: text indicating easy access from I-5
      - room_count: stated rooms count (string)
      - has_pool: text indicating outdoor pool
      - has_fitness_center: text indicating fitness center
      - has_restaurant: text indicating on-site restaurant/dining
      - check_in_time: stated check-in time (string, e.g., "3:00 PM")

    Special rules:
    - For any URL fields, extract only actual URLs present in the answer (plain URLs or markdown links).
    - Keep numbers as strings if they appear with units (e.g., "25 acres", "1,000 feet", "3:00 PM").
    - If multiple candidate URLs are provided for a hotel, include all of them in reference_urls (deduplicate if repeated).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


def _has_required(info_name: Optional[str], urls: Optional[List[str]]) -> bool:
    return bool(info_name and info_name.strip() and urls and len(urls) > 0)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_florida_resort(
    evaluator: Evaluator,
    parent_node,
    info: Optional[FloridaResortInfo],
):
    node = evaluator.add_parallel(
        id="hotel_1_florida_beachfront",
        desc="A beachfront resort on Florida's Gulf Coast with extensive amenities",
        parent=parent_node,
        critical=False,
    )

    # Existence gate
    exists = _has_required(info.name if info else None, info.reference_urls if info else [])
    evaluator.add_custom_node(
        result=exists,
        id="hotel_1_required_info",
        desc="Hotel #1: name provided and at least one reference URL",
        parent=node,
        critical=True,
    )

    sources = _safe_sources(info.reference_urls if info else [])
    name = info.name if info and info.name else "the resort"

    # Reference page about the property
    ref_leaf = evaluator.add_leaf(
        id="hotel_1_reference",
        desc="Provide a URL reference confirming the resort meets the specified criteria",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This webpage is about {name} and describes the resort/property.",
        node=ref_leaf,
        sources=sources,
        additional_instruction="Accept official hotel site or reputable sources (e.g., brand, tourism board, major OTAs). The page should clearly be about the named property.",
    )

    # Location
    loc_leaf = evaluator.add_leaf(
        id="hotel_1_location",
        desc="The resort must be located on St. Pete Beach, Florida, with direct Gulf of Mexico beachfront access",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} is located on St. Pete Beach, Florida and has direct Gulf of Mexico beachfront access.",
        node=loc_leaf,
        sources=sources,
        additional_instruction="Look for language like 'beachfront', 'on St. Pete Beach', 'Gulf of Mexico frontage', or similar explicit wording.",
    )

    # Property size
    acres_leaf = evaluator.add_leaf(
        id="hotel_1_property_size",
        desc="The resort must span at least 20 tropical acres along the beach",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} spans at least 20 acres along the beachfront.",
        node=acres_leaf,
        sources=sources,
        additional_instruction="Support should show acreage of 20+ acres (e.g., '25 acres'); synonyms like 'tropical acres' acceptable.",
    )

    # Pools
    pools_leaf = evaluator.add_leaf(
        id="hotel_1_pool_facilities",
        desc="The resort must have at least 3 outdoor heated swimming pools",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} has at least 3 outdoor heated swimming pools.",
        node=pools_leaf,
        sources=sources,
        additional_instruction="Explicit mention of 'heated' and 'outdoor' pools is preferred. If counts vary by season, accept clear evidence of 3 or more.",
    )

    # Waterslide
    slide_leaf = evaluator.add_leaf(
        id="hotel_1_waterslide",
        desc="The resort must feature a multi-story waterslide (at least 3 stories)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} features a multi-story waterslide that is at least three stories tall.",
        node=slide_leaf,
        sources=sources,
        additional_instruction="Accept phrasing like '3-story waterslide' or 'three-story slide'; synonyms OK.",
    )

    # Dining
    dining_leaf = evaluator.add_leaf(
        id="hotel_1_dining_options",
        desc="The resort must have multiple on-site food and beverage outlets (at least 5)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} has at least 5 on-site food and beverage outlets.",
        node=dining_leaf,
        sources=sources,
        additional_instruction="Count restaurants, bars, cafés, lounges, or grab-and-go that are on-site.",
    )

    # Beach loungers
    loungers_leaf = evaluator.add_leaf(
        id="hotel_1_beach_loungers",
        desc="The resort must provide at least 200 beach loungers for guests",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} provides at least 200 beach loungers, chairs, or cabanas for guests on its beach.",
        node=loungers_leaf,
        sources=sources,
        additional_instruction="Accept synonyms like cabanas/hooded chairs if they are functionally beach loungers; the page should imply ≥200 available.",
    )

    # Shoreline length
    shoreline_leaf = evaluator.add_leaf(
        id="hotel_1_shoreline_length",
        desc="The resort must have at least 1,000 feet of beachfront shoreline",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} has at least 1,000 feet of beachfront shoreline.",
        node=shoreline_leaf,
        sources=sources,
        additional_instruction="Look for explicit numbers like '1,000 feet' or more of private/continuous beach.",
    )

    # Room count
    rooms_leaf = evaluator.add_leaf(
        id="hotel_1_room_count",
        desc="The resort must have at least 300 guest rooms and accommodations",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} has at least 300 guest rooms or accommodations.",
        node=rooms_leaf,
        sources=sources,
        additional_instruction="Suites count as accommodations; accept total keys/units if stated.",
    )

    # Activities
    activities_leaf = evaluator.add_leaf(
        id="hotel_1_activities",
        desc="The resort must offer water-based activities such as paddleboats or floating water park features",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} offers water-based activities such as paddleboats or floating water-park features.",
        node=activities_leaf,
        sources=sources,
        additional_instruction="Accept explicit offerings like paddleboats, inflatable water park, floating slides, or similar.",
    )


async def verify_orlando_disney(
    evaluator: Evaluator,
    parent_node,
    info: Optional[OrlandoDisneyHotelInfo],
):
    node = evaluator.add_parallel(
        id="hotel_2_orlando_theme_park",
        desc="A hotel in the Disney Springs Resort Area with official Disney benefits",
        parent=parent_node,
        critical=False,
    )

    exists = _has_required(info.name if info else None, info.reference_urls if info else [])
    evaluator.add_custom_node(
        result=exists,
        id="hotel_2_required_info",
        desc="Hotel #2: name provided and at least one reference URL",
        parent=node,
        critical=True,
    )

    sources = _safe_sources(info.reference_urls if info else [])
    name = info.name if info and info.name else "the hotel"

    # Reference page about the property
    ref_leaf = evaluator.add_leaf(
        id="hotel_2_reference",
        desc="Provide a URL reference confirming the hotel meets the specified criteria",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This webpage is about {name} (a Disney Springs Resort Area hotel).",
        node=ref_leaf,
        sources=sources,
        additional_instruction="Accept official hotel site, Disney partner listing, or reputable OTA page clearly about the named property.",
    )

    # Location in Disney Springs Resort Area
    loc_leaf = evaluator.add_leaf(
        id="hotel_2_location",
        desc="The hotel must be located in the Disney Springs Resort Area in Orlando, Florida",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} is located in the Disney Springs Resort Area in Orlando, Florida.",
        node=loc_leaf,
        sources=sources,
        additional_instruction="Look for explicit mention of 'Disney Springs Resort Area' affiliation.",
    )

    # Distance to Disney Springs (<= 0.5 miles)
    dist_leaf = evaluator.add_leaf(
        id="hotel_2_disney_springs_distance",
        desc="The hotel must be within walking distance (half mile or less) of Disney Springs",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} is within 0.5 miles walking distance of Disney Springs (roughly 10 minutes or less on foot).",
        node=dist_leaf,
        sources=sources,
        additional_instruction="Accept phrases like 'across the street' or 'steps from Disney Springs' implying ≤0.5 miles.",
    )

    # Free shuttle to theme parks
    shuttle_leaf = evaluator.add_leaf(
        id="hotel_2_theme_park_shuttle",
        desc="The hotel must provide free shuttle transportation to Walt Disney World theme parks",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} provides free shuttle transportation to Walt Disney World theme parks.",
        node=shuttle_leaf,
        sources=sources,
        additional_instruction="Confirm that transportation is complimentary and serves Disney theme parks.",
    )

    # Early Theme Park Entry
    ete_leaf = evaluator.add_leaf(
        id="hotel_2_early_entry",
        desc="The hotel must offer early theme park entry as a guest benefit",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Guests at {name} receive Early Theme Park Entry at Walt Disney World.",
        node=ete_leaf,
        sources=sources,
        additional_instruction="The page should explicitly state 'Early Theme Park Entry' or equivalent official benefit.",
    )

    # Complimentary standard parking at theme parks
    parking_leaf = evaluator.add_leaf(
        id="hotel_2_theme_park_parking",
        desc="The hotel must provide complimentary standard parking at Disney theme parks as a benefit",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Guests at {name} receive complimentary standard parking at Disney theme parks as a benefit.",
        node=parking_leaf,
        sources=sources,
        additional_instruction="This is distinct from free hotel parking; the page must indicate free parking at the Disney theme parks.",
    )

    # Room count >= 150
    rooms_leaf = evaluator.add_leaf(
        id="hotel_2_room_count",
        desc="The hotel must have at least 150 guest rooms",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} has at least 150 guest rooms.",
        node=rooms_leaf,
        sources=sources,
        additional_instruction="Accept 'rooms and suites' as total room count when stated.",
    )

    # Early access to book Disney dining reservations
    dining_res_leaf = evaluator.add_leaf(
        id="hotel_2_dining_reservations",
        desc="The hotel must offer early access to book Disney dining reservations",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Guests at {name} receive early access to book Disney dining reservations.",
        node=dining_res_leaf,
        sources=sources,
        additional_instruction="Look for language like 'early access to book dining' or similar official benefit description.",
    )

    # Amenities: outdoor pool and fitness center
    amenities_leaf = evaluator.add_leaf(
        id="hotel_2_amenities",
        desc="The hotel must have an outdoor swimming pool and fitness center",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} has an outdoor swimming pool and an on-site fitness center.",
        node=amenities_leaf,
        sources=sources,
        additional_instruction="Both amenities must be present on-site.",
    )

    # Check-in time >= 3:00 PM
    checkin_leaf = evaluator.add_leaf(
        id="hotel_2_check_in_time",
        desc="The hotel's standard check-in time must be 3:00 PM or later",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The standard check-in time at {name} is 3:00 PM or later.",
        node=checkin_leaf,
        sources=sources,
        additional_instruction="Accept 3:00 PM or a later time such as 4:00 PM as satisfying the requirement.",
    )


async def verify_whistler_conference(
    evaluator: Evaluator,
    parent_node,
    info: Optional[WhistlerConferenceHotelInfo],
):
    node = evaluator.add_parallel(
        id="hotel_3_whistler_ski",
        desc="A ski resort hotel in Whistler, British Columbia with extensive meeting facilities",
        parent=parent_node,
        critical=False,
    )

    exists = _has_required(info.name if info else None, info.reference_urls if info else [])
    evaluator.add_custom_node(
        result=exists,
        id="hotel_3_required_info",
        desc="Hotel #3: name provided and at least one reference URL",
        parent=node,
        critical=True,
    )

    sources = _safe_sources(info.reference_urls if info else [])
    name = info.name if info and info.name else "the hotel"

    # Reference page
    ref_leaf = evaluator.add_leaf(
        id="hotel_3_reference",
        desc="Provide a URL reference confirming the hotel meets the specified criteria",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This webpage is about {name} in Whistler, BC.",
        node=ref_leaf,
        sources=sources,
        additional_instruction="Accept official hotel/group site or reputable meeting/OTA listing clearly about the property.",
    )

    # Location at base of Whistler/Blackcomb
    loc_leaf = evaluator.add_leaf(
        id="hotel_3_location",
        desc="The hotel must be located in Whistler, British Columbia, at the base of Whistler or Blackcomb Mountain",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} is located in Whistler, British Columbia at the base of Whistler or Blackcomb Mountain (e.g., ski-in/ski-out).",
        node=loc_leaf,
        sources=sources,
        additional_instruction="Look for 'at the base', 'ski-in/ski-out', or explicit proximity to Whistler/Blackcomb base areas.",
    )

    # Meeting space total >= 15,000 sqft
    meet_total_leaf = evaluator.add_leaf(
        id="hotel_3_meeting_space_total",
        desc="The hotel must have at least 15,000 square feet of total meeting and event space",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} offers at least 15,000 square feet of total meeting and event space.",
        node=meet_total_leaf,
        sources=sources,
        additional_instruction="Accept totals stated in sq ft (≥15,000).",
    )

    # Meeting rooms count >= 10
    meet_rooms_leaf = evaluator.add_leaf(
        id="hotel_3_meeting_rooms",
        desc="The hotel must have at least 10 separate meeting rooms or function spaces",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} has at least 10 separate meeting rooms or function spaces.",
        node=meet_rooms_leaf,
        sources=sources,
        additional_instruction="Accept named rooms, salons, boardrooms; combine divisible rooms appropriately if page treats them as separate spaces.",
    )

    # Ballroom capacity >= 500
    ballroom_leaf = evaluator.add_leaf(
        id="hotel_3_ballroom_capacity",
        desc="The hotel's largest ballroom must accommodate at least 500 people",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The largest ballroom at {name} accommodates at least 500 people.",
        node=ballroom_leaf,
        sources=sources,
        additional_instruction="Capacity may be shown as theater/banquet; any clear configuration with capacity ≥500 qualifies.",
    )

    # Room count >= 300
    rooms_leaf = evaluator.add_leaf(
        id="hotel_3_room_count",
        desc="The hotel must have at least 300 guest rooms",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} has at least 300 guest rooms.",
        node=rooms_leaf,
        sources=sources,
        additional_instruction="Suites count as rooms.",
    )

    # Walking distance to ski lifts
    lift_leaf = evaluator.add_leaf(
        id="hotel_3_ski_access",
        desc="The hotel must be within walking distance of ski lift access",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} is within walking distance of ski lift access.",
        node=lift_leaf,
        sources=sources,
        additional_instruction="Phrases like 'steps from the lifts' or 'short walk to the gondola' qualify.",
    )

    # On-site restaurants count >= 2
    dining_leaf = evaluator.add_leaf(
        id="hotel_3_dining",
        desc="The hotel must have multiple on-site restaurants (at least 2)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} has at least two on-site restaurants.",
        node=dining_leaf,
        sources=sources,
        additional_instruction="Count distinct on-site restaurants; bars/lounges can count if serving food as restaurants.",
    )

    # On-site spa
    spa_leaf = evaluator.add_leaf(
        id="hotel_3_spa",
        desc="The hotel must have an on-site spa facility",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} has an on-site spa facility.",
        node=spa_leaf,
        sources=sources,
        additional_instruction="Spa brand names and in-hotel wellness centers qualify if clearly on-site.",
    )

    # Sustainability (non-critical)
    sustain_leaf = evaluator.add_leaf(
        id="hotel_3_sustainability",
        desc="The hotel should hold a sustainability certification such as Green Key",
        parent=node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"{name} holds a recognized sustainability certification such as Green Key, Green Key Global, or similar.",
        node=sustain_leaf,
        sources=sources,
        additional_instruction="Accept reputable third-party certifications (e.g., Green Key, LEED, BOMA, EarthCheck).",
    )


async def verify_sixflags_hotel(
    evaluator: Evaluator,
    parent_node,
    info: Optional[SixFlagsHotelInfo],
):
    node = evaluator.add_parallel(
        id="hotel_4_six_flags",
        desc="A hotel near Six Flags Magic Mountain in Valencia, California with convenient theme park access",
        parent=parent_node,
        critical=False,
    )

    exists = _has_required(info.name if info else None, info.reference_urls if info else [])
    evaluator.add_custom_node(
        result=exists,
        id="hotel_4_required_info",
        desc="Hotel #4: name provided and at least one reference URL",
        parent=node,
        critical=True,
    )

    sources = _safe_sources(info.reference_urls if info else [])
    name = info.name if info and info.name else "the hotel"

    # Reference page
    ref_leaf = evaluator.add_leaf(
        id="hotel_4_reference",
        desc="Provide a URL reference confirming the hotel meets the specified criteria",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This webpage is about {name} in Valencia or Santa Clarita, CA.",
        node=ref_leaf,
        sources=sources,
        additional_instruction="Accept official hotel site or reputable listing clearly about the property.",
    )

    # Location (Valencia or Santa Clarita)
    loc_leaf = evaluator.add_leaf(
        id="hotel_4_location",
        desc="The hotel must be located in Valencia or Santa Clarita, California",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} is located in either Valencia, California or Santa Clarita, California.",
        node=loc_leaf,
        sources=sources,
        additional_instruction="The page should clearly state the city as Valencia or Santa Clarita.",
    )

    # Distance to Six Flags <= 1.5 miles
    dist_leaf = evaluator.add_leaf(
        id="hotel_4_six_flags_distance",
        desc="The hotel must be within 1.5 miles of Six Flags Magic Mountain's entrance",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} is within 1.5 miles of the Six Flags Magic Mountain entrance.",
        node=dist_leaf,
        sources=sources,
        additional_instruction="Accept distances stated as ≤1.5 miles; mentions like 'adjacent' or exact mileages within threshold qualify.",
    )

    # Walking distance 10–15 min (non-critical)
    walk_leaf = evaluator.add_leaf(
        id="hotel_4_walking_distance",
        desc="The hotel should be within walking distance (10-15 minute walk) of Six Flags Magic Mountain",
        parent=node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"{name} is within a typical 10–15 minute walk of Six Flags Magic Mountain.",
        node=walk_leaf,
        sources=sources,
        additional_instruction="Accept approximate walk times or short-walk phrasing consistent with 10–15 minutes.",
    )

    # Interstate 5 access
    i5_leaf = evaluator.add_leaf(
        id="hotel_4_interstate_access",
        desc="The hotel must be easily accessible from Interstate 5",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} is easily accessible from Interstate 5 (I‑5).",
        node=i5_leaf,
        sources=sources,
        additional_instruction="Look for driving directions or proximity statements referencing I‑5.",
    )

    # Room count >= 100
    rooms_leaf = evaluator.add_leaf(
        id="hotel_4_room_count",
        desc="The hotel must have at least 100 guest rooms",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} has at least 100 guest rooms.",
        node=rooms_leaf,
        sources=sources,
        additional_instruction="Suites count toward total.",
    )

    # Outdoor pool
    pool_leaf = evaluator.add_leaf(
        id="hotel_4_pool",
        desc="The hotel must have an outdoor swimming pool",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} has an outdoor swimming pool.",
        node=pool_leaf,
        sources=sources,
        additional_instruction="Pool must be outdoors.",
    )

    # Fitness center
    fitness_leaf = evaluator.add_leaf(
        id="hotel_4_fitness_center",
        desc="The hotel must have a fitness center on-site",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} has an on-site fitness center.",
        node=fitness_leaf,
        sources=sources,
        additional_instruction="Look for 'fitness center' or 'gym' on the amenities page.",
    )

    # On-site restaurant
    rest_leaf = evaluator.add_leaf(
        id="hotel_4_restaurant",
        desc="The hotel must have an on-site restaurant or dining option",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{name} has an on-site restaurant or dining option.",
        node=rest_leaf,
        sources=sources,
        additional_instruction="Breakfast-only rooms without a dining venue do not qualify; on-site dining is required.",
    )

    # Check-in time exactly 3:00 PM
    checkin_leaf = evaluator.add_leaf(
        id="hotel_4_check_in_time",
        desc="The hotel's check-in time must be 3:00 PM",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The standard check-in time at {name} is 3:00 PM.",
        node=checkin_leaf,
        sources=sources,
        additional_instruction="Must be 3:00 PM (not earlier or later) to satisfy this exact requirement.",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the multi-hotel requirements task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel across the four hotel categories
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=TravelHotelsExtraction,
        extraction_name="hotels_extraction",
    )

    # Build verification subtrees for each hotel category
    await verify_florida_resort(evaluator, root, extracted.florida_beachfront or FloridaResortInfo())
    await verify_orlando_disney(evaluator, root, extracted.orlando_disney or OrlandoDisneyHotelInfo())
    await verify_whistler_conference(evaluator, root, extracted.whistler_conference or WhistlerConferenceHotelInfo())
    await verify_sixflags_hotel(evaluator, root, extracted.sixflags_hotel or SixFlagsHotelInfo())

    # Return evaluation summary
    return evaluator.get_summary()