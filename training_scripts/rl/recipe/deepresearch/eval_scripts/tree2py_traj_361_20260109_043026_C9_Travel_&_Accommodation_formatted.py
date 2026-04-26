import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "airport_hotels_business_accessible_pets"
TASK_DESCRIPTION = """
I'm planning business travel across the United States and need to identify 4 suitable airport hotels at different major U.S. airports that meet comprehensive requirements for business travelers with accessibility needs and pets.

For each of the 4 hotels (each at a different U.S. airport), please provide:

1. Hotel Identification: Official hotel name, airport name with airport code, and complete street address
2. Terminal Access: Specify whether the hotel has direct terminal connection via indoor walkway or AirTrain, OR provides complimentary 24-hour shuttle service running every 10-20 minutes to terminals
3. Accessibility: Confirmation of ADA-compliant wheelchair-accessible rooms with at least 3 specific accessibility features (such as roll-in shower, grab bars, accessible toilet height, bed height, or clear space requirements)
4. Pet Policy: Pet acceptance confirmation with specific weight limit per pet (in pounds), maximum number of pets per room, and pet fee amount
5. Business Amenities:
   - 24-hour business center with at least 2 types of equipment (computers, printers, or high-speed internet)
   - Conference/meeting room availability with capacity information and at least 1 equipment type
6. Room Types: At least 3 distinct room categories (such as standard, deluxe, executive, or suite) with descriptions of size or amenity differences
7. Policies:
   - Check-in time (must fall within 2:00 PM - 4:00 PM range)
   - Check-out time (must fall within 11:00 AM - 12:00 PM range)
   - Flexible cancellation policy allowing free cancellation 24-48 hours before check-in
8. Facilities:
   - 24-hour fitness center with equipment details
   - Indoor swimming pool
   - On-site parking with EV charging stations available
   - Complimentary high-speed WiFi suitable for business needs
   - Breakfast service availability (specify if complimentary or paid, and type)

For each piece of information provided, include a reference URL from the hotel's official website or a reputable booking/travel site that confirms the details.
"""


# =========================
# Data Models
# =========================
class RoomCategory(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None  # differences in size/amenities


class Identification(BaseModel):
    hotel_name: Optional[str] = None
    airport_name: Optional[str] = None
    airport_code: Optional[str] = None
    address: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class TerminalAccess(BaseModel):
    # Use strings to maximize compatibility
    direct_connection: Optional[str] = None  # e.g., "indoor walkway", "AirTrain", "skybridge"
    shuttle_complimentary: Optional[str] = None  # "yes"/"no"
    shuttle_24h: Optional[str] = None  # "yes"/"no"
    shuttle_frequency: Optional[str] = None  # e.g., "every 10–20 minutes", "every 15 minutes"
    shuttle_distance_miles: Optional[str] = None  # e.g., "0.7", "1 mile", "less than 1 mile"
    urls: List[str] = Field(default_factory=list)


class Accessibility(BaseModel):
    accessible_rooms_confirmed: Optional[str] = None  # "yes"/"no"
    features: List[str] = Field(default_factory=list)  # e.g., ["roll-in shower", "grab bars"]
    urls: List[str] = Field(default_factory=list)


class PetPolicy(BaseModel):
    pets_accepted: Optional[str] = None  # "yes"/"no"
    weight_limit_lbs: Optional[str] = None  # e.g., "50", "75 lbs", "two pets up to 40 lb each"
    max_pets_per_room: Optional[str] = None  # e.g., "2", "2 pets"
    pet_fee: Optional[str] = None  # e.g., "$100 per stay", "$75 per night"
    urls: List[str] = Field(default_factory=list)


class BusinessCenter(BaseModel):
    open_24h: Optional[str] = None  # "yes"/"no"
    equipment: List[str] = Field(default_factory=list)  # e.g., ["computers", "printers", "high-speed internet"]
    urls: List[str] = Field(default_factory=list)


class MeetingFacilities(BaseModel):
    rooms_available: Optional[str] = None  # "yes"/"no"
    capacity_info: Optional[str] = None  # e.g., "Up to 120 guests", "500 sq ft room", etc.
    equipment: List[str] = Field(default_factory=list)  # e.g., ["projection screens", "audio equipment"]
    urls: List[str] = Field(default_factory=list)


class RoomTypes(BaseModel):
    categories: List[RoomCategory] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class Policies(BaseModel):
    checkin_time: Optional[str] = None  # e.g., "3:00 PM"
    checkout_time: Optional[str] = None  # e.g., "11:00 AM"
    cancellation_policy: Optional[str] = None  # e.g., "Free cancellation up to 48 hours before"
    urls: List[str] = Field(default_factory=list)


class Facilities(BaseModel):
    fitness_24h: Optional[str] = None  # "yes"/"no"
    fitness_equipment: List[str] = Field(default_factory=list)  # e.g., ["treadmills", "free weights"]
    indoor_pool: Optional[str] = None  # "yes"/"no"
    parking_on_site: Optional[str] = None  # "yes"/"no"
    ev_charging: Optional[str] = None  # "yes"/"no"
    wifi_complimentary_high_speed: Optional[str] = None  # "yes"/"no"
    breakfast_service: Optional[str] = None  # e.g., "complimentary buffet", "paid continental"
    urls: List[str] = Field(default_factory=list)


class FireSafety(BaseModel):
    sprinkler_system: Optional[str] = None  # "yes"/"no"
    audible_visual_alarms: Optional[str] = None  # "yes"/"no"
    exit_signs_visible: Optional[str] = None  # "yes"/"no"
    urls: List[str] = Field(default_factory=list)


class HotelItem(BaseModel):
    identification: Identification = Field(default_factory=Identification)
    terminal_access: TerminalAccess = Field(default_factory=TerminalAccess)
    accessibility: Accessibility = Field(default_factory=Accessibility)
    pet_policy: PetPolicy = Field(default_factory=PetPolicy)
    business_center: BusinessCenter = Field(default_factory=BusinessCenter)
    meeting_facilities: MeetingFacilities = Field(default_factory=MeetingFacilities)
    room_types: RoomTypes = Field(default_factory=RoomTypes)
    policies: Policies = Field(default_factory=Policies)
    facilities: Facilities = Field(default_factory=Facilities)
    fire_safety: FireSafety = Field(default_factory=FireSafety)


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


# =========================
# Extraction Prompt
# =========================
def prompt_extract_hotels() -> str:
    return """
    Extract structured information for up to 4 airport hotels (each at a different U.S. airport) mentioned in the answer. For each hotel, return the following fields:

    identification:
      - hotel_name (string)
      - airport_name (string)
      - airport_code (string, e.g., "JFK", "LAX")
      - address (string, complete street address)
      - urls (array of URLs that support name/airport/address)

    terminal_access:
      - direct_connection (string or null, e.g., "indoor walkway", "AirTrain", "skybridge")
      - shuttle_complimentary (string "yes"/"no"/null)
      - shuttle_24h (string "yes"/"no"/null)
      - shuttle_frequency (string or null, e.g., "every 10–20 minutes", "every 15 minutes")
      - shuttle_distance_miles (string or null, e.g., "0.7", "1 mile", "less than 1 mile")
      - urls (array of URLs)

    accessibility:
      - accessible_rooms_confirmed (string "yes"/"no"/null indicating ADA wheelchair-accessible rooms)
      - features (array of strings; list specific features such as "roll-in shower", "grab bars", "toilet height 17–19 inches", "bed height 20–23 inches", "≥36 inches clear space each side of bed")
      - urls (array of URLs)

    pet_policy:
      - pets_accepted (string "yes"/"no"/null)
      - weight_limit_lbs (string or null; include numeric limit and unit if available)
      - max_pets_per_room (string or null)
      - pet_fee (string or null; amount and whether per stay or per night)
      - urls (array of URLs)

    business_center:
      - open_24h (string "yes"/"no"/null)
      - equipment (array of strings among ["computers", "printers", "high-speed internet"]; include any others too)
      - urls (array of URLs)

    meeting_facilities:
      - rooms_available (string "yes"/"no"/null)
      - capacity_info (string or null; e.g., capacities or room sizes)
      - equipment (array of strings; e.g., "projection screens", "audio equipment")
      - urls (array of URLs)

    room_types:
      - categories (array of objects: {name, description})
      - urls (array of URLs)

    policies:
      - checkin_time (string, e.g., "3:00 PM")
      - checkout_time (string, e.g., "11:00 AM")
      - cancellation_policy (string)
      - urls (array of URLs)

    facilities:
      - fitness_24h (string "yes"/"no"/null)
      - fitness_equipment (array of strings, include cardio and strength equipment details)
      - indoor_pool (string "yes"/"no"/null)
      - parking_on_site (string "yes"/"no"/null)
      - ev_charging (string "yes"/"no"/null)
      - wifi_complimentary_high_speed (string "yes"/"no"/null)
      - breakfast_service (string describing whether complimentary or paid, and type)
      - urls (array of URLs)

    fire_safety:
      - sprinkler_system (string "yes"/"no"/null)
      - audible_visual_alarms (string "yes"/"no"/null)
      - exit_signs_visible (string "yes"/"no"/null)
      - urls (array of URLs)

    Return a JSON object with "hotels": [ ... up to 4 items ... ].
    Only include URLs that are explicitly mentioned in the answer. If a field is not mentioned, return null or empty array accordingly.
    """


# =========================
# Helper Functions
# =========================
def _truthy(s: Optional[str]) -> bool:
    if not s:
        return False
    return str(s).strip().lower() in {"yes", "true", "y", "1"}


def _parse_time_to_minutes(t: Optional[str]) -> Optional[int]:
    if not t:
        return None
    s = t.strip().lower()
    # Normalize unicode dash
    s = s.replace("–", "-").replace("—", "-")
    # Remove words like "around", "approximately"
    s = re.sub(r"\b(around|approximately|about|after)\b", "", s).strip()

    # Patterns: "3:00 pm", "3 pm", "15:00"
    m = re.match(r"^(\d{1,2}):?(\d{2})?\s*([ap]\.?m\.?)?$", s)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        ampm = m.group(3)
        if ampm:
            ampm = ampm.replace(".", "")
            if ampm.startswith("p") and hour != 12:
                hour += 12
            if ampm.startswith("a") and hour == 12:
                hour = 0
        return hour * 60 + minute

    # Words like "3 pm"
    m2 = re.match(r"^(\d{1,2})\s*([ap]m)$", s)
    if m2:
        hour = int(m2.group(1))
        ampm = m2.group(2)
        if ampm == "pm" and hour != 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        return hour * 60

    # 24h like "15:00"
    m3 = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if m3:
        hour = int(m3.group(1))
        minute = int(m3.group(2))
        return hour * 60 + minute

    # Only hour like "15"
    m4 = re.match(r"^(\d{1,2})$", s)
    if m4:
        hour = int(m4.group(1))
        return hour * 60

    return None


def _time_in_range(t: Optional[str], start_min: int, end_min: int) -> bool:
    """
    Inclusive range [start_min, end_min]. Returns False if parse fails.
    """
    mins = _parse_time_to_minutes(t)
    if mins is None:
        return False
    return start_min <= mins <= end_min


def _first_n_hotels(extraction: HotelsExtraction, n: int = 4) -> List[HotelItem]:
    hotels = extraction.hotels[:n]
    # pad if fewer
    while len(hotels) < n:
        hotels.append(HotelItem())
    return hotels


# =========================
# Verification Builders
# =========================
async def build_identification_nodes(evaluator: Evaluator, parent, hotel: HotelItem, idx: int):
    ident = hotel.identification
    node = evaluator.add_parallel(
        id=f"hotel_{idx}_identification",
        desc="Provide official hotel name, airport name + airport code, and complete street address, with a supporting URL",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(ident.hotel_name and ident.hotel_name.strip()),
        id=f"hotel_{idx}_hotel_name",
        desc="Official hotel name is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(ident.airport_name and ident.airport_name.strip() and ident.airport_code and ident.airport_code.strip()),
        id=f"hotel_{idx}_airport_name_code",
        desc="Airport name and airport code are provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(ident.address and ident.address.strip()),
        id=f"hotel_{idx}_street_address",
        desc="Complete street address is provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=f"hotel_{idx}_identification_url",
        desc="At least one URL supports the identification details (hotel/airport/address)",
        parent=node,
        critical=True
    )
    claim = f"This page confirms the hotel's official name '{ident.hotel_name}', the airport '{ident.airport_name}' ({ident.airport_code}), and the complete street address '{ident.address}'. Allow formatting variations."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ident.urls,
        additional_instruction="Accept pages from the hotel's official site or reputable booking/travel sites. Minor formatting differences are acceptable."
    )


async def build_terminal_access_nodes(evaluator: Evaluator, parent, hotel: HotelItem, idx: int):
    ta = hotel.terminal_access
    node = evaluator.add_parallel(
        id=f"hotel_{idx}_terminal_access",
        desc="Hotel satisfies terminal access requirement (direct connection OR qualifying shuttle), with a supporting URL",
        parent=parent,
        critical=True
    )

    cond_leaf = evaluator.add_leaf(
        id=f"hotel_{idx}_terminal_access_condition",
        desc="Either (A) direct indoor walkway/AirTrain connection to terminals is stated, OR (B) hotel is within 1 mile and offers complimentary 24-hour shuttle running every 10–20 minutes",
        parent=node,
        critical=True
    )
    if ta.direct_connection and ta.direct_connection.strip():
        claim = f"The hotel has a direct connection to airport terminals via {ta.direct_connection}."
    else:
        claim = (
            f"The hotel is within {ta.shuttle_distance_miles} of the airport and offers a complimentary 24-hour shuttle "
            f"running {ta.shuttle_frequency} to the terminals."
        )
    await evaluator.verify(
        claim=claim,
        node=cond_leaf,
        sources=ta.urls,
        additional_instruction="Satisfy A or B. For B, require (distance ≤ 1 mile), complimentary, 24-hour, and frequency approximately every 10–20 minutes (±5 minutes acceptable)."
    )

    evaluator.add_custom_node(
        result=bool(ta.urls),
        id=f"hotel_{idx}_terminal_access_url",
        desc="At least one URL supports the terminal access details (direct connection or shuttle/distance/frequency/hours)",
        parent=node,
        critical=True
    )


async def build_accessibility_nodes(evaluator: Evaluator, parent, hotel: HotelItem, idx: int):
    acc = hotel.accessibility
    node = evaluator.add_parallel(
        id=f"hotel_{idx}_accessibility",
        desc="Hotel satisfies ADA accessibility requirements for wheelchair-accessible rooms, with a supporting URL",
        parent=parent,
        critical=True
    )

    leaf_rooms = evaluator.add_leaf(
        id=f"hotel_{idx}_ada_accessible_rooms",
        desc="Hotel offers ADA-compliant wheelchair-accessible rooms (explicit confirmation)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel explicitly offers ADA-compliant wheelchair-accessible rooms.",
        node=leaf_rooms,
        sources=acc.urls,
        additional_instruction="Verify explicit ADA/wheelchair-accessible room availability on the provided page."
    )

    features_text = "; ".join(acc.features) if acc.features else ""
    leaf_features = evaluator.add_leaf(
        id=f"hotel_{idx}_ada_required_features",
        desc="Accessibility details include roll-in shower, grab bars, accessible toilet height (17–19 inches), bed height (20–23 inches), and ≥36 inches clear space on both sides of bed",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page's accessibility details include at least three of the following with clear specifications: roll-in shower; grab bars; toilet height 17–19 inches; bed height 20–23 inches; ≥36 inches clear space on both sides of bed. Listed features: {features_text}.",
        node=leaf_features,
        sources=acc.urls,
        additional_instruction="Pass if at least three of the specified features are present and clearly described (heights/clear space where applicable)."
    )

    evaluator.add_custom_node(
        result=bool(acc.urls),
        id=f"hotel_{idx}_accessibility_url",
        desc="At least one URL supports the accessibility/ADA room and feature details",
        parent=node,
        critical=True
    )


async def build_pet_policy_nodes(evaluator: Evaluator, parent, hotel: HotelItem, idx: int):
    pp = hotel.pet_policy
    node = evaluator.add_parallel(
        id=f"hotel_{idx}_pet_policy",
        desc="Hotel pet policy is fully specified, with a supporting URL",
        parent=parent,
        critical=True
    )

    leaf_accept = evaluator.add_leaf(
        id=f"hotel_{idx}_pets_accepted",
        desc="Hotel explicitly accepts pets",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel explicitly accepts pets.",
        node=leaf_accept,
        sources=pp.urls,
        additional_instruction="Verify pet acceptance; service animals alone do not count as general pet acceptance."
    )

    leaf_weight = evaluator.add_leaf(
        id=f"hotel_{idx}_pet_weight_limit",
        desc="Specific weight limit per pet (in pounds) is provided",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"A specific weight limit per pet (in pounds) is provided: '{pp.weight_limit_lbs}'.",
        node=leaf_weight,
        sources=pp.urls,
        additional_instruction="Confirm the numeric weight limit per pet in pounds (or a clearly equivalent statement)."
    )

    leaf_max = evaluator.add_leaf(
        id=f"hotel_{idx}_max_pets_per_room",
        desc="Maximum number of pets per room is provided",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The maximum number of pets per room is specified: '{pp.max_pets_per_room}'.",
        node=leaf_max,
        sources=pp.urls,
        additional_instruction="Confirm the maximum number of pets per room."
    )

    leaf_fee = evaluator.add_leaf(
        id=f"hotel_{idx}_pet_fee",
        desc="Pet fee amount is provided (one-time or per-night as applicable)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The pet fee amount is provided: '{pp.pet_fee}'.",
        node=leaf_fee,
        sources=pp.urls,
        additional_instruction="Confirm the pet fee amount and whether per stay or per night."
    )

    evaluator.add_custom_node(
        result=bool(pp.urls),
        id=f"hotel_{idx}_pet_policy_url",
        desc="At least one URL supports the pet acceptance, limits, and fees",
        parent=node,
        critical=True
    )


async def build_business_amenities_nodes(evaluator: Evaluator, parent, hotel: HotelItem, idx: int):
    bc = hotel.business_center
    mf = hotel.meeting_facilities

    node = evaluator.add_parallel(
        id=f"hotel_{idx}_business_amenities",
        desc="Hotel provides required business amenities, with supporting URLs",
        parent=parent,
        critical=True
    )

    # Business Center
    bc_node = evaluator.add_parallel(
        id=f"hotel_{idx}_business_center",
        desc="24-hour business center with required equipment is confirmed, with a supporting URL",
        parent=node,
        critical=True
    )

    leaf_24h = evaluator.add_leaf(
        id=f"hotel_{idx}_business_center_24h",
        desc="Business center is explicitly 24-hour",
        parent=bc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The business center is explicitly open 24 hours.",
        node=leaf_24h,
        sources=bc.urls,
        additional_instruction="Confirm wording indicating 24-hour access."
    )

    eq_list = ", ".join(bc.equipment) if bc.equipment else ""
    leaf_eq = evaluator.add_leaf(
        id=f"hotel_{idx}_business_center_equipment",
        desc="Business center includes high-speed internet, computers, and printers",
        parent=bc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The business center provides at least two equipment types among high-speed internet, computers, and printers. Listed in answer: {eq_list}.",
        node=leaf_eq,
        sources=bc.urls,
        additional_instruction="Pass if the page confirms at least two of the following: high-speed internet, computers, printers."
    )

    evaluator.add_custom_node(
        result=bool(bc.urls),
        id=f"hotel_{idx}_business_center_url",
        desc="At least one URL supports business center hours and equipment",
        parent=bc_node,
        critical=True
    )

    # Meeting Facilities
    mf_node = evaluator.add_parallel(
        id=f"hotel_{idx}_meeting_facilities",
        desc="Meeting/conference facilities meet the stated requirements, with a supporting URL",
        parent=node,
        critical=True
    )

    leaf_rooms_avail = evaluator.add_leaf(
        id=f"hotel_{idx}_meeting_rooms_available",
        desc="Conference/meeting rooms are available (explicit confirmation)",
        parent=mf_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel provides conference/meeting rooms.",
        node=leaf_rooms_avail,
        sources=mf.urls,
        additional_instruction="Confirm meeting rooms availability on the page."
    )

    leaf_capacity = evaluator.add_leaf(
        id=f"hotel_{idx}_meeting_capacity_info",
        desc="Capacity information is provided for meeting/conference space",
        parent=mf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Capacity information is provided for the meeting/conference space: '{mf.capacity_info}'.",
        node=leaf_capacity,
        sources=mf.urls,
        additional_instruction="Look for capacity (people) and/or room size in sq ft."
    )

    leaf_sqft_pp = evaluator.add_leaf(
        id=f"hotel_{idx}_meeting_space_per_person_constraint",
        desc="Provided meeting size/capacity information supports at least 20–25 square feet per person (or explicitly states meeting this standard)",
        parent=mf_node,
        critical=True
    )
    await evaluator.verify(
        claim="The meeting space information supports at least ~20 square feet per person (either by explicit statement or by room size vs capacity).",
        node=leaf_sqft_pp,
        sources=mf.urls,
        additional_instruction="If both capacity and square footage are present, verify sq ft per person ≥ 20. Otherwise, accept explicit compliance statements."
    )

    leaf_mf_eq = evaluator.add_leaf(
        id=f"hotel_{idx}_meeting_equipment",
        desc="Meeting space equipment includes projection screens and audio equipment",
        parent=mf_node,
        critical=True
    )
    eq_txt = ", ".join(mf.equipment) if mf.equipment else ""
    await evaluator.verify(
        claim=f"The meeting facilities include equipment such as projection screens and audio equipment. Listed equipment: {eq_txt}.",
        node=leaf_mf_eq,
        sources=mf.urls,
        additional_instruction="Pass if at least one relevant equipment type (e.g., projection screen or audio equipment) is confirmed; both preferred."
    )

    evaluator.add_custom_node(
        result=bool(mf.urls),
        id=f"hotel_{idx}_meeting_facilities_url",
        desc="At least one URL supports meeting room availability, capacity/size, and equipment",
        parent=mf_node,
        critical=True
    )


async def build_room_types_nodes(evaluator: Evaluator, parent, hotel: HotelItem, idx: int):
    rt = hotel.room_types
    node = evaluator.add_parallel(
        id=f"hotel_{idx}_room_types",
        desc="Hotel offers at least 3 room categories with described differences, with a supporting URL",
        parent=parent,
        critical=True
    )

    categories_names = [c.name for c in rt.categories if c and c.name]
    categories_diffs = [c.description for c in rt.categories if c and c.description]
    cat_list_text = ", ".join([n for n in categories_names if n]) if categories_names else ""

    leaf_three = evaluator.add_leaf(
        id=f"hotel_{idx}_three_room_categories",
        desc="At least 3 distinct room categories are listed",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hotel offers at least three distinct room categories. Categories listed: {cat_list_text}.",
        node=leaf_three,
        sources=rt.urls,
        additional_instruction="Confirm at least three distinct categories (e.g., standard, deluxe, executive, suite)."
    )

    diff_text = "; ".join([d for d in categories_diffs if d]) if categories_diffs else ""
    leaf_diffs = evaluator.add_leaf(
        id=f"hotel_{idx}_room_category_differences",
        desc="Differences between room categories are described (size and/or amenity differences)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Differences between categories (size/amenities) are described. Examples: {diff_text}.",
        node=leaf_diffs,
        sources=rt.urls,
        additional_instruction="Look for size differences (sq ft) or amenities (e.g., lounge access, upgraded bath)."
    )

    evaluator.add_custom_node(
        result=bool(rt.urls),
        id=f"hotel_{idx}_room_types_url",
        desc="At least one URL supports room category and difference details",
        parent=node,
        critical=True
    )


async def build_policies_nodes(evaluator: Evaluator, parent, hotel: HotelItem, idx: int):
    pol = hotel.policies
    node = evaluator.add_parallel(
        id=f"hotel_{idx}_policies",
        desc="Hotel policies meet check-in/check-out and cancellation requirements, with supporting URLs",
        parent=parent,
        critical=True
    )

    # Custom checks for time windows
    evaluator.add_custom_node(
        result=_time_in_range(pol.checkin_time, 14 * 60, 16 * 60),
        id=f"hotel_{idx}_checkin_time_window",
        desc="Check-in time is within 2:00 PM–4:00 PM",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_time_in_range(pol.checkout_time, 11 * 60, 12 * 60),
        id=f"hotel_{idx}_checkout_time_window",
        desc="Check-out time is within 11:00 AM–12:00 PM",
        parent=node,
        critical=True
    )

    leaf_cancel = evaluator.add_leaf(
        id=f"hotel_{idx}_cancellation_policy",
        desc="Flexible cancellation allows free cancellation 24–48 hours before check-in",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel's policy allows free cancellation within 24–48 hours before check-in.",
        node=leaf_cancel,
        sources=pol.urls,
        additional_instruction="Verify language indicating free cancellation window ≥24 hours and ≤48 hours prior to check-in."
    )

    evaluator.add_custom_node(
        result=bool(pol.urls),
        id=f"hotel_{idx}_policies_url",
        desc="At least one URL supports check-in, check-out, and cancellation policy details",
        parent=node,
        critical=True
    )


async def build_facilities_nodes(evaluator: Evaluator, parent, hotel: HotelItem, idx: int):
    fa = hotel.facilities
    node = evaluator.add_parallel(
        id=f"hotel_{idx}_facilities",
        desc="Hotel facilities meet fitness/pool/parking/EV/WiFi/breakfast requirements, with supporting URLs",
        parent=parent,
        critical=True
    )

    leaf_fit = evaluator.add_leaf(
        id=f"hotel_{idx}_fitness_center",
        desc="24-hour fitness center with cardio and strength equipment details is provided",
        parent=node,
        critical=True
    )
    fit_eq = ", ".join(fa.fitness_equipment) if fa.fitness_equipment else ""
    await evaluator.verify(
        claim=f"The hotel provides a 24-hour fitness center with cardio and strength equipment. Equipment details: {fit_eq}.",
        node=leaf_fit,
        sources=fa.urls,
        additional_instruction="Confirm 24-hour access and list of equipment types (e.g., treadmills, elliptical, free weights)."
    )

    leaf_pool = evaluator.add_leaf(
        id=f"hotel_{idx}_indoor_pool",
        desc="Indoor swimming pool is available",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="An indoor swimming pool is available.",
        node=leaf_pool,
        sources=fa.urls,
        additional_instruction="Confirm that the pool is indoor."
    )

    leaf_parking_ev = evaluator.add_leaf(
        id=f"hotel_{idx}_parking_and_ev",
        desc="On-site parking is available and EV charging stations are available",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="On-site parking is available and EV charging stations are available on the property.",
        node=leaf_parking_ev,
        sources=fa.urls,
        additional_instruction="Confirm both on-site parking and the availability of EV charging."
    )

    leaf_wifi = evaluator.add_leaf(
        id=f"hotel_{idx}_wifi",
        desc="Complimentary high-speed WiFi suitable for business needs is provided",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Complimentary high-speed WiFi suitable for business needs is provided.",
        node=leaf_wifi,
        sources=fa.urls,
        additional_instruction="Confirm complimentary (free) and high-speed suitable for business use."
    )

    leaf_breakfast = evaluator.add_leaf(
        id=f"hotel_{idx}_breakfast",
        desc="Breakfast service availability is provided, including whether complimentary or paid and the type",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Breakfast service is available; details include whether complimentary or paid and the type: '{fa.breakfast_service}'.",
        node=leaf_breakfast,
        sources=fa.urls,
        additional_instruction="Confirm breakfast availability and specify whether complimentary or paid, and the type (buffet, continental, etc.)."
    )

    evaluator.add_custom_node(
        result=bool(fa.urls),
        id=f"hotel_{idx}_facilities_url",
        desc="At least one URL supports the facilities (fitness/pool/parking+EV/WiFi/breakfast) details",
        parent=node,
        critical=True
    )


async def build_fire_safety_nodes(evaluator: Evaluator, parent, hotel: HotelItem, idx: int):
    fs = hotel.fire_safety
    node = evaluator.add_parallel(
        id=f"hotel_{idx}_fire_safety",
        desc="Hotel meets stated fire safety requirements, with a supporting URL",
        parent=parent,
        critical=True
    )

    leaf_sprinkler = evaluator.add_leaf(
        id=f"hotel_{idx}_sprinkler_system",
        desc="Sprinkler system is present/confirmed",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel's fire safety includes a sprinkler system.",
        node=leaf_sprinkler,
        sources=fs.urls,
        additional_instruction="Confirm presence of sprinklers."
    )

    leaf_alarms = evaluator.add_leaf(
        id=f"hotel_{idx}_audible_visual_alarms",
        desc="Fire alarms include audible and visual signals",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Fire alarms include both audible and visual signals.",
        node=leaf_alarms,
        sources=fs.urls,
        additional_instruction="Confirm audible and visual alarm features."
    )

    leaf_exit = evaluator.add_leaf(
        id=f"hotel_{idx}_exit_signs",
        desc="Exit signs are visible from all areas (or equivalently stated)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Exit signage is clearly visible throughout the property (or equivalently stated).",
        node=leaf_exit,
        sources=fs.urls,
        additional_instruction="Confirm visibility of exit signs or equivalent compliance statements."
    )

    evaluator.add_custom_node(
        result=bool(fs.urls),
        id=f"hotel_{idx}_fire_safety_url",
        desc="At least one URL supports the fire safety details",
        parent=node,
        critical=True
    )


async def verify_hotel(evaluator: Evaluator, parent, hotel: HotelItem, idx_zero_based: int, prior_airport_codes: List[str]):
    idx = idx_zero_based + 1
    hotel_node = evaluator.add_parallel(
        id=f"hotel_{idx}",
        desc=f"Hotel #{idx} (airport-hotel item {idx}) meets all required conditions",
        parent=parent,
        critical=False
    )

    # For hotels 2–4: airport uniqueness check
    if idx >= 2:
        current_code = (hotel.identification.airport_code or "").strip().upper() if hotel.identification.airport_code else ""
        is_unique = bool(current_code) and current_code not in [c.strip().upper() for c in prior_airport_codes if c]
        evaluator.add_custom_node(
            result=is_unique,
            id=f"hotel_{idx}_airport_uniqueness",
            desc=f"Hotel #{idx} airport code differs from previous hotel(s) airport code(s)",
            parent=hotel_node,
            critical=True
        )
        group_parent = evaluator.add_parallel(
            id=f"hotel_{idx}_same_requirements_as_hotel_1",
            desc=f"Hotel #{idx} satisfies all the same requirement groups as Hotel #1 (identification, terminal access, accessibility, pet policy, business amenities, room types, policies, facilities, fire safety)",
            parent=hotel_node,
            critical=True
        )
    else:
        group_parent = hotel_node

    # Build all requirement groups
    await build_identification_nodes(evaluator, group_parent, hotel, idx)
    await build_terminal_access_nodes(evaluator, group_parent, hotel, idx)
    await build_accessibility_nodes(evaluator, group_parent, hotel, idx)
    await build_pet_policy_nodes(evaluator, group_parent, hotel, idx)
    await build_business_amenities_nodes(evaluator, group_parent, hotel, idx)
    await build_room_types_nodes(evaluator, group_parent, hotel, idx)
    await build_policies_nodes(evaluator, group_parent, hotel, idx)
    await build_facilities_nodes(evaluator, group_parent, hotel, idx)
    await build_fire_safety_nodes(evaluator, group_parent, hotel, idx)


# =========================
# Main Evaluation Entry
# =========================
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root should be non-critical to satisfy framework constraints
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

    # Extract hotels info
    extraction = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction"
    )

    hotels = _first_n_hotels(extraction, 4)
    prior_codes: List[str] = []

    # Add root description node (non-critical, parallel aggregation already set)
    # Build verification for each hotel
    for i, hotel in enumerate(hotels):
        await verify_hotel(evaluator, root, hotel, i, prior_codes)
        # Track airport codes for uniqueness checks
        code = hotel.identification.airport_code or ""
        if code.strip():
            prior_codes.append(code.strip())

    # Custom info: number of hotels extracted and airport codes
    evaluator.add_custom_info(
        info={"extracted_hotels_count": len(extraction.hotels), "used_hotels_count": len(hotels),
              "airport_codes": prior_codes},
        info_type="extraction_stats",
        info_name="hotels_stats"
    )

    return evaluator.get_summary()