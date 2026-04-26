import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

TASK_ID = "Multi_City_Accommodation_Search"
TASK_DESCRIPTION = """Find four hotels across different US cities, each meeting comprehensive specified requirements for distinct purposes: family resort in Phoenix, pet-friendly hotel in San Antonio, business conference hotel in Nashville, and accessible beachfront hotel in Montauk."""

# =========================
# Data Models
# =========================

class HotelCore(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    star_rating: Optional[str] = None  # e.g., "3-star", "4 stars"
    url: Optional[str] = None  # Official website or authoritative booking platform URL
    additional_urls: List[str] = Field(default_factory=list)  # Any other cited URLs


class PhoenixFeatures(BaseModel):
    resort_classification: Optional[bool] = None
    family_friendly: Optional[bool] = None
    kids_club: Optional[bool] = None
    pool_waterslide: Optional[bool] = None
    two_bedroom_suite_available: Optional[bool] = None
    suite_size_min_sqft: Optional[str] = None
    breakfast_included: Optional[bool] = None
    parking_on_site: Optional[bool] = None
    check_in_age: Optional[str] = None  # e.g., "18", "21"
    cancellation_policy_24hr_full_refund: Optional[bool] = None


class SanAntonioFeatures(BaseModel):
    near_six_flags_within_5_miles: Optional[bool] = None
    dogs_allowed: Optional[bool] = None
    dog_weight_limit_per_dog_lbs: Optional[str] = None  # e.g., "75", "80", "No limit"
    no_breed_restrictions: Optional[bool] = None
    standard_rooms_available: Optional[bool] = None
    fitness_center_on_site: Optional[bool] = None
    dining_on_site_or_walkable: Optional[bool] = None
    standard_check_in_time_range: Optional[str] = None  # e.g., "3:00 PM", "4:00 PM"
    cc_hold_per_night_usd: Optional[str] = None  # e.g., "$150", "$200", "None"


class NashvilleFeatures(BaseModel):
    downtown_or_within_3_miles: Optional[bool] = None
    meeting_room_min_sqft: Optional[str] = None
    meeting_room_capacity_min: Optional[str] = None
    business_center_available: Optional[bool] = None
    double_occupancy_rooms_available: Optional[bool] = None
    on_site_restaurant: Optional[bool] = None
    wifi_guest_and_meeting: Optional[bool] = None
    cancellation_48hr_full_refund: Optional[bool] = None


class MontaukFeatures(BaseModel):
    beachfront_or_oceanfront: Optional[bool] = None
    ada_accessible_rooms: Optional[bool] = None
    accessible_doorway_width_in: Optional[str] = None
    hallway_width_in: Optional[str] = None
    is_multi_story: Optional[bool] = None
    elevator_to_all_floors_or_ground_floor_accessible: Optional[bool] = None
    parking_on_site: Optional[bool] = None
    on_site_restaurant: Optional[bool] = None
    standard_check_in_time: Optional[str] = None  # e.g., "3:00 PM"


class PhoenixHotel(BaseModel):
    core: HotelCore = HotelCore()
    features: PhoenixFeatures = PhoenixFeatures()


class SanAntonioHotel(BaseModel):
    core: HotelCore = HotelCore()
    features: SanAntonioFeatures = SanAntonioFeatures()


class NashvilleHotel(BaseModel):
    core: HotelCore = HotelCore()
    features: NashvilleFeatures = NashvilleFeatures()


class MontaukHotel(BaseModel):
    core: HotelCore = HotelCore()
    features: MontaukFeatures = MontaukFeatures()


class MultiCityExtraction(BaseModel):
    phoenix: Optional[PhoenixHotel] = None
    san_antonio: Optional[SanAntonioHotel] = None
    nashville: Optional[NashvilleHotel] = None
    montauk: Optional[MontaukHotel] = None


# =========================
# Extraction Prompt
# =========================

def prompt_extract_multi_city_hotels() -> str:
    return """
    Extract the hotel details presented in the answer for each of the four required cities: Phoenix/Scottsdale (AZ), San Antonio (TX), Nashville (TN), and Montauk (NY).
    For each city, return a structured object with:
    core:
      - name: Official hotel name
      - address: Full street address including city, state, ZIP
      - city: City name
      - state: State abbreviation or name
      - zip_code: ZIP code
      - star_rating: Star rating classification as a string (e.g., "3-star", "4 stars"). If not mentioned, return null.
      - url: The official hotel website URL or a verified authoritative booking platform URL mentioned in the answer
      - additional_urls: An array of any other cited URLs for the hotel in the answer (can be empty)
    features: City-specific boolean or text fields directly extracted from the answer (do not invent; if not mentioned, return null):
      For phoenix:
        - resort_classification (true/false)
        - family_friendly (true/false)
        - kids_club (true/false)
        - pool_waterslide (true/false)
        - two_bedroom_suite_available (true/false)
        - suite_size_min_sqft (string, e.g., "700", "750")
        - breakfast_included (true/false)
        - parking_on_site (true/false)
        - check_in_age (string, e.g., "18", "21")
        - cancellation_policy_24hr_full_refund (true/false)
      For san_antonio:
        - near_six_flags_within_5_miles (true/false)
        - dogs_allowed (true/false)
        - dog_weight_limit_per_dog_lbs (string, e.g., "75", "No limit")
        - no_breed_restrictions (true/false)
        - standard_rooms_available (true/false)
        - fitness_center_on_site (true/false)
        - dining_on_site_or_walkable (true/false)
        - standard_check_in_time_range (string, e.g., "3:00 PM", "4:00 PM")
        - cc_hold_per_night_usd (string, e.g., "$150", "$200", "None")
      For nashville:
        - downtown_or_within_3_miles (true/false)
        - meeting_room_min_sqft (string, e.g., "500", "600")
        - meeting_room_capacity_min (string, e.g., "20", "50")
        - business_center_available (true/false)
        - double_occupancy_rooms_available (true/false)
        - on_site_restaurant (true/false)
        - wifi_guest_and_meeting (true/false)
        - cancellation_48hr_full_refund (true/false)
      For montauk:
        - beachfront_or_oceanfront (true/false)
        - ada_accessible_rooms (true/false)
        - accessible_doorway_width_in (string, e.g., "32", "36")
        - hallway_width_in (string, e.g., "36", "40")
        - is_multi_story (true/false)
        - elevator_to_all_floors_or_ground_floor_accessible (true/false)
        - parking_on_site (true/false)
        - on_site_restaurant (true/false)
        - standard_check_in_time (string, e.g., "3:00 PM")

    If multiple hotels are listed for a city, extract the first one only. If a value is not provided in the answer text, set it to null (or empty list for additional_urls). Do not infer or add information not present in the answer.
    Return a JSON object with keys: phoenix, san_antonio, nashville, montauk.
    """


# =========================
# Helper Functions
# =========================

def gather_sources(core: HotelCore) -> List[str]:
    urls: List[str] = []
    if core and core.url:
        urls.append(core.url)
    if core and core.additional_urls:
        urls.extend([u for u in core.additional_urls if isinstance(u, str) and u.strip() != ""])
    return urls


def normalize_city_state(city: Optional[str], state: Optional[str]) -> str:
    c = (city or "").strip()
    s = (state or "").strip()
    return f"{c}, {s}".strip(", ")


# =========================
# Verification Subtrees
# =========================

async def verify_phoenix(evaluator: Evaluator, parent: VerificationNode, data: Optional[PhoenixHotel]) -> None:
    node = evaluator.add_parallel(
        id="Phoenix_Family_Resort",
        desc="Identify a family-friendly resort in Phoenix or Scottsdale, Arizona that meets all specified requirements for a spring break family vacation",
        parent=parent,
        critical=False
    )

    core = data.core if data else HotelCore()
    features = data.features if data else PhoenixFeatures()
    sources = gather_sources(core)

    # URL Reference - critical
    url_leaf = evaluator.add_leaf(
        id="Phoenix_URL_Reference",
        desc="Provide the official hotel website URL or verified booking platform URL for the Phoenix family resort",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This webpage is the official hotel site or an authoritative booking platform page for {core.name or 'the selected resort'}. Recognized platforms include brand sites (Marriott, Hilton, Hyatt, IHG, Wyndham, Choice, Omni, Loews) or major OTAs (Booking.com, Expedia, Hotels.com).",
        node=url_leaf,
        sources=sources,
        additional_instruction="Confirm that the page corresponds to the selected resort and is either the official site or an established booking platform."
    )

    # Location group - critical
    loc = evaluator.add_parallel(
        id="Phoenix_Location",
        desc="Verify the resort is located in Phoenix or Scottsdale, Arizona and is classified as a family-friendly resort property",
        parent=node,
        critical=True
    )

    city_state_leaf = evaluator.add_leaf(
        id="Phoenix_City_State",
        desc="The property is located in Phoenix or Scottsdale, Arizona",
        parent=loc,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel is located in Phoenix or Scottsdale, Arizona.",
        node=city_state_leaf,
        sources=sources,
        additional_instruction="Check the address on the page; either Phoenix, AZ or Scottsdale, AZ satisfies the requirement.",
        extra_prerequisites=[url_leaf]
    )

    resort_class_leaf = evaluator.add_leaf(
        id="Phoenix_Resort_Classification",
        desc="The property is classified or marketed as a resort (not just a standard hotel)",
        parent=loc,
        critical=True
    )
    await evaluator.verify(
        claim="This property is marketed and classified as a resort.",
        node=resort_class_leaf,
        sources=sources,
        additional_instruction="Look for 'Resort' in the property type or marketing description. Timeshares/villas operated as resorts also count.",
        extra_prerequisites=[url_leaf]
    )

    family_friendly_leaf = evaluator.add_leaf(
        id="Phoenix_Family_Friendly_Designation",
        desc="The property explicitly markets itself as family-friendly or suitable for families with children",
        parent=loc,
        critical=True
    )
    await evaluator.verify(
        claim="The resort explicitly markets itself as family-friendly or suitable for families with children.",
        node=family_friendly_leaf,
        sources=sources,
        additional_instruction="Look for phrases such as 'family-friendly', 'great for families', 'kids', 'children', 'family activities'.",
        extra_prerequisites=[url_leaf]
    )

    # Room configuration - sequential, critical
    room_cfg = evaluator.add_sequential(
        id="Phoenix_Room_Configuration",
        desc="Verify the resort offers two-bedroom suite accommodations meeting the minimum size requirement",
        parent=node,
        critical=True
    )

    suite_avail_leaf = evaluator.add_leaf(
        id="Phoenix_Two_Bedroom_Suite_Available",
        desc="The resort offers two-bedroom suite configurations",
        parent=room_cfg,
        critical=True
    )
    await evaluator.verify(
        claim="The resort offers two-bedroom suite configurations.",
        node=suite_avail_leaf,
        sources=sources,
        additional_instruction="Check accommodations listings for 'Two Bedroom Suite', '2-Bedroom Suite', 'Two-Bedroom Villa', etc.",
        extra_prerequisites=[url_leaf]
    )

    suite_size_leaf = evaluator.add_leaf(
        id="Phoenix_Suite_Size_Minimum",
        desc="The two-bedroom suites are at least 700 square feet in size",
        parent=room_cfg,
        critical=True
    )
    size_text = features.suite_size_min_sqft or "700"
    await evaluator.verify(
        claim=f"The two-bedroom suites are at least 700 square feet.",
        node=suite_size_leaf,
        sources=sources,
        additional_instruction="Confirm suite size details; allow minor variations (e.g., 'approx 700 sq ft', '700+ sq ft').",
        extra_prerequisites=[url_leaf, suite_avail_leaf]
    )

    # Family amenities - parallel, critical
    fam_amen = evaluator.add_parallel(
        id="Phoenix_Family_Amenities",
        desc="Verify the resort provides essential family-oriented amenities including kids club and water recreation",
        parent=node,
        critical=True
    )

    kids_club_leaf = evaluator.add_leaf(
        id="Phoenix_Kids_Club",
        desc="The resort has a kids club that offers supervised activities for children",
        parent=fam_amen,
        critical=True
    )
    await evaluator.verify(
        claim="The resort has a kids club offering supervised activities for children.",
        node=kids_club_leaf,
        sources=sources,
        additional_instruction="Look for 'Kids Club', 'Children’s Club', or supervised kids programs.",
        extra_prerequisites=[url_leaf]
    )

    waterslide_leaf = evaluator.add_leaf(
        id="Phoenix_Pool_Waterslide",
        desc="The resort has a swimming pool with a waterslide or water features suitable for children",
        parent=fam_amen,
        critical=True
    )
    await evaluator.verify(
        claim="The resort has a swimming pool with a waterslide or notable water features suitable for children.",
        node=waterslide_leaf,
        sources=sources,
        additional_instruction="Look for a waterslide, splash pad, lazy river, or similar family water features.",
        extra_prerequisites=[url_leaf]
    )

    # Policies & services - parallel, critical
    policies = evaluator.add_parallel(
        id="Phoenix_Policies_Services",
        desc="Verify the resort meets policy requirements for check-in age and cancellation flexibility",
        parent=node,
        critical=True
    )

    age_leaf = evaluator.add_leaf(
        id="Phoenix_Check_In_Age_18",
        desc="The check-in age requirement is 18 years old (not 21 or higher)",
        parent=policies,
        critical=True
    )
    await evaluator.verify(
        claim="The minimum check-in age is 18 years old (not 21 or higher).",
        node=age_leaf,
        sources=sources,
        additional_instruction="Check policy/FAQ; verify that 18-year-olds can check in.",
        extra_prerequisites=[url_leaf]
    )

    cancel_leaf = evaluator.add_leaf(
        id="Phoenix_Flexible_Cancellation_24hrs",
        desc="The resort offers flexible cancellation with full refund if cancelled at least 24 hours before check-in",
        parent=policies,
        critical=True
    )
    await evaluator.verify(
        claim="Cancellation with full refund is allowed if cancelled at least 24 hours before check-in.",
        node=cancel_leaf,
        sources=sources,
        additional_instruction="Check rate/cancellation policy; flexible cancellation windows should include a 24-hour full refund option.",
        extra_prerequisites=[url_leaf]
    )

    # General amenities - parallel, critical
    general = evaluator.add_parallel(
        id="Phoenix_General_Amenities",
        desc="Verify the resort provides breakfast, parking, and meets star rating standards",
        parent=node,
        critical=True
    )

    breakfast_leaf = evaluator.add_leaf(
        id="Phoenix_Complimentary_Breakfast",
        desc="The resort includes complimentary breakfast (continental or buffet style)",
        parent=general,
        critical=True
    )
    await evaluator.verify(
        claim="Complimentary breakfast (continental or buffet) is included.",
        node=breakfast_leaf,
        sources=sources,
        additional_instruction="Confirm breakfast is complimentary; room-rate dependent is acceptable if clearly stated.",
        extra_prerequisites=[url_leaf]
    )

    parking_leaf = evaluator.add_leaf(
        id="Phoenix_On_Site_Parking",
        desc="The resort has on-site parking available for guests",
        parent=general,
        critical=True
    )
    await evaluator.verify(
        claim="On-site parking is available for guests.",
        node=parking_leaf,
        sources=sources,
        additional_instruction="Look for parking details (garage/lot/valet); paid or complimentary both satisfy availability.",
        extra_prerequisites=[url_leaf]
    )

    star_leaf = evaluator.add_leaf(
        id="Phoenix_Star_Rating_3_Plus",
        desc="The resort is rated 3-star or higher",
        parent=general,
        critical=True
    )
    await evaluator.verify(
        claim="The resort has a rating of at least 3 stars.",
        node=star_leaf,
        sources=sources,
        additional_instruction="Use booking platform star classifications if official site does not state stars.",
        extra_prerequisites=[url_leaf]
    )


async def verify_san_antonio(evaluator: Evaluator, parent: VerificationNode, data: Optional[SanAntonioHotel]) -> None:
    node = evaluator.add_parallel(
        id="San_Antonio_Pet_Friendly_Hotel",
        desc="Identify a pet-friendly hotel in San Antonio, Texas near Six Flags Fiesta Texas that accommodates large dogs",
        parent=parent,
        critical=False
    )

    core = data.core if data else HotelCore()
    features = data.features if data else SanAntonioFeatures()
    sources = gather_sources(core)

    url_leaf = evaluator.add_leaf(
        id="San_Antonio_URL_Reference",
        desc="Provide the official hotel website URL or verified booking platform URL for the San Antonio pet-friendly hotel",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This webpage is the official hotel site or an authoritative booking platform page for {core.name or 'the selected hotel'}.",
        node=url_leaf,
        sources=sources,
        additional_instruction="Confirm it is either an official brand site or a major OTA page."
    )

    loc = evaluator.add_parallel(
        id="San_Antonio_Location",
        desc="Verify the hotel is located within 5 miles of Six Flags Fiesta Texas in San Antonio, Texas",
        parent=node,
        critical=True
    )

    sa_city_leaf = evaluator.add_leaf(
        id="San_Antonio_City_State",
        desc="The hotel is located in San Antonio, Texas",
        parent=loc,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel is located in San Antonio, Texas.",
        node=sa_city_leaf,
        sources=sources,
        additional_instruction="Confirm via the address.",
        extra_prerequisites=[url_leaf]
    )

    sa_sixflags_leaf = evaluator.add_leaf(
        id="San_Antonio_Six_Flags_Proximity",
        desc="The hotel is within 5 miles of Six Flags Fiesta Texas",
        parent=loc,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel is within 5 miles of Six Flags Fiesta Texas (17000 W I-10, San Antonio, TX 78257).",
        node=sa_sixflags_leaf,
        sources=sources,
        additional_instruction="Look for stated distances on the page or hotel description; explicit mention of distance ≤ 5 miles satisfies this requirement.",
        extra_prerequisites=[url_leaf]
    )

    pet = evaluator.add_parallel(
        id="San_Antonio_Pet_Policy",
        desc="Verify the hotel accepts dogs with appropriate weight limits and no breed restrictions",
        parent=node,
        critical=True
    )

    dogs_leaf = evaluator.add_leaf(
        id="San_Antonio_Dogs_Allowed",
        desc="The hotel explicitly allows dogs as pets",
        parent=pet,
        critical=True
    )
    await evaluator.verify(
        claim="Dogs are explicitly allowed at the hotel.",
        node=dogs_leaf,
        sources=sources,
        additional_instruction="Check pet policy; cats allowed is irrelevant; dogs must be allowed.",
        extra_prerequisites=[url_leaf]
    )

    weight_leaf = evaluator.add_leaf(
        id="San_Antonio_Weight_Limit_75lbs",
        desc="The hotel allows dogs with a maximum weight limit of at least 75 pounds per dog",
        parent=pet,
        critical=True
    )
    await evaluator.verify(
        claim="The pet policy allows dogs of at least 75 pounds per dog (or has no maximum weight limit).",
        node=weight_leaf,
        sources=sources,
        additional_instruction="If policy states 'no weight limit', treat as meeting ≥75 lbs.",
        extra_prerequisites=[url_leaf]
    )

    breed_leaf = evaluator.add_leaf(
        id="San_Antonio_No_Breed_Restrictions",
        desc="The hotel has no breed restrictions for dogs",
        parent=pet,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel has no breed restrictions for dogs.",
        node=breed_leaf,
        sources=sources,
        additional_instruction="Policy must explicitly state no breed restrictions or imply all breeds allowed.",
        extra_prerequisites=[url_leaf]
    )

    room_amen = evaluator.add_parallel(
        id="San_Antonio_Room_Amenities",
        desc="Verify the hotel offers standard rooms and required amenities",
        parent=node,
        critical=True
    )

    std_rooms_leaf = evaluator.add_leaf(
        id="San_Antonio_Standard_Rooms",
        desc="The hotel has standard rooms available (suites not required)",
        parent=room_amen,
        critical=True
    )
    await evaluator.verify(
        claim="Standard rooms (not only suites) are available.",
        node=std_rooms_leaf,
        sources=sources,
        additional_instruction="Look for 'Standard Room', 'Double', 'King' room types.",
        extra_prerequisites=[url_leaf]
    )

    fitness_leaf = evaluator.add_leaf(
        id="San_Antonio_Fitness_Center",
        desc="The hotel has a fitness center on-site",
        parent=room_amen,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel has an on-site fitness center.",
        node=fitness_leaf,
        sources=sources,
        additional_instruction="Look for 'Fitness Center', 'Gym'.",
        extra_prerequisites=[url_leaf]
    )

    dining_leaf = evaluator.add_leaf(
        id="San_Antonio_Dining_Options",
        desc="The hotel has an on-site restaurant or dining options within walking distance",
        parent=room_amen,
        critical=True
    )
    await evaluator.verify(
        claim="There is either an on-site restaurant or dining options within walking distance.",
        node=dining_leaf,
        sources=sources,
        additional_instruction="Walking distance is reasonable; look for 'on-site dining', 'restaurant next door', etc.",
        extra_prerequisites=[url_leaf]
    )

    policies = evaluator.add_parallel(
        id="San_Antonio_Policies",
        desc="Verify the hotel meets check-in timing and credit card hold requirements",
        parent=node,
        critical=True
    )

    checkin_leaf = evaluator.add_leaf(
        id="San_Antonio_Standard_Check_In_Time",
        desc="The hotel allows check-in during standard hours (3:00-4:00 PM)",
        parent=policies,
        critical=True
    )
    await evaluator.verify(
        claim="Standard check-in time is between 3:00 PM and 4:00 PM.",
        node=checkin_leaf,
        sources=sources,
        additional_instruction="If multiple times are given, confirm that 3pm or 4pm is a standard check-in time.",
        extra_prerequisites=[url_leaf]
    )

    hold_leaf = evaluator.add_leaf(
        id="San_Antonio_Hold_Limit_200",
        desc="The credit card authorization hold does not exceed $200 per night (excluding room rate and taxes)",
        parent=policies,
        critical=True
    )
    await evaluator.verify(
        claim="The credit card authorization hold (incidental deposit/pre-authorization) is no more than $200 per night.",
        node=hold_leaf,
        sources=sources,
        additional_instruction="Look for 'incidental hold', 'deposit', 'pre-authorization'; if the page states $200 or less, pass; if unspecified, treat as not supported.",
        extra_prerequisites=[url_leaf]
    )


async def verify_nashville(evaluator: Evaluator, parent: VerificationNode, data: Optional[NashvilleHotel]) -> None:
    node = evaluator.add_parallel(
        id="Nashville_Business_Hotel",
        desc="Identify a business-conference hotel in Nashville, Tennessee with adequate meeting facilities for corporate events",
        parent=parent,
        critical=False
    )

    core = data.core if data else HotelCore()
    features = data.features if data else NashvilleFeatures()
    sources = gather_sources(core)

    url_leaf = evaluator.add_leaf(
        id="Nashville_URL_Reference",
        desc="Provide the official hotel website URL or verified booking platform URL for the Nashville business hotel",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This webpage is the official hotel site or an authoritative booking platform page for {core.name or 'the selected hotel'}.",
        node=url_leaf,
        sources=sources,
        additional_instruction="Confirm correspondence to the selected property and page type."
    )

    loc = evaluator.add_parallel(
        id="Nashville_Location",
        desc="Verify the hotel is located in downtown Nashville or within 3 miles of downtown",
        parent=node,
        critical=True
    )

    n_city_leaf = evaluator.add_leaf(
        id="Nashville_City_State",
        desc="The hotel is located in Nashville, Tennessee",
        parent=loc,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel is located in Nashville, Tennessee.",
        node=n_city_leaf,
        sources=sources,
        additional_instruction="Confirm via address.",
        extra_prerequisites=[url_leaf]
    )

    n_downtown_leaf = evaluator.add_leaf(
        id="Nashville_Downtown_Proximity",
        desc="The hotel is in downtown Nashville or within 3 miles of downtown",
        parent=loc,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel is in downtown Nashville or within 3 miles of downtown.",
        node=n_downtown_leaf,
        sources=sources,
        additional_instruction="Look for distance to downtown or neighborhood description indicating downtown proximity.",
        extra_prerequisites=[url_leaf]
    )

    mtg = evaluator.add_parallel(
        id="Nashville_Meeting_Facilities",
        desc="Verify the hotel has appropriate meeting room facilities for business conferences",
        parent=node,
        critical=True
    )

    mtg_size_leaf = evaluator.add_leaf(
        id="Nashville_Meeting_Room_500sqft",
        desc="The hotel has at least one meeting room with a minimum of 500 square feet",
        parent=mtg,
        critical=True
    )
    await evaluator.verify(
        claim="There is at least one meeting room with a minimum of 500 square feet.",
        node=mtg_size_leaf,
        sources=sources,
        additional_instruction="Check event/meetings page for room specs; allow 'approx' or '500+ sq ft'.",
        extra_prerequisites=[url_leaf]
    )

    mtg_capacity_leaf = evaluator.add_leaf(
        id="Nashville_Capacity_20_People",
        desc="The meeting room can accommodate 20 or more people",
        parent=mtg,
        critical=True
    )
    await evaluator.verify(
        claim="At least one meeting room accommodates 20 or more people.",
        node=mtg_capacity_leaf,
        sources=sources,
        additional_instruction="Look for capacity charts, room setup options listed ≥ 20 attendees.",
        extra_prerequisites=[url_leaf]
    )

    biz_center_leaf = evaluator.add_leaf(
        id="Nashville_Business_Center",
        desc="The hotel has a business center available to guests",
        parent=mtg,
        critical=True
    )
    await evaluator.verify(
        claim="A business center is available to guests.",
        node=biz_center_leaf,
        sources=sources,
        additional_instruction="Check amenities list for 'Business Center'.",
        extra_prerequisites=[url_leaf]
    )

    room_biz = evaluator.add_parallel(
        id="Nashville_Room_Business_Amenities",
        desc="Verify the hotel offers appropriate room configurations and business amenities",
        parent=node,
        critical=True
    )

    double_occ_leaf = evaluator.add_leaf(
        id="Nashville_Double_Occupancy_Rooms",
        desc="The hotel offers standard double occupancy rooms",
        parent=room_biz,
        critical=True
    )
    await evaluator.verify(
        claim="Standard double occupancy rooms are offered.",
        node=double_occ_leaf,
        sources=sources,
        additional_instruction="Look for 'Double' or 'Two beds' standard rooms.",
        extra_prerequisites=[url_leaf]
    )

    onsite_dining_leaf = evaluator.add_leaf(
        id="Nashville_On_Site_Dining",
        desc="The hotel has an on-site restaurant or dining facility",
        parent=room_biz,
        critical=True
    )
    await evaluator.verify(
        claim="There is an on-site restaurant or dining facility.",
        node=onsite_dining_leaf,
        sources=sources,
        additional_instruction="Look for 'Restaurant', 'Bar & Grill', 'Dining'.",
        extra_prerequisites=[url_leaf]
    )

    wifi_leaf = evaluator.add_leaf(
        id="Nashville_WiFi_Access",
        desc="The hotel provides WiFi access in guest rooms and meeting spaces",
        parent=room_biz,
        critical=True
    )
    await evaluator.verify(
        claim="WiFi is provided in guest rooms and meeting spaces.",
        node=wifi_leaf,
        sources=sources,
        additional_instruction="Confirm WiFi availability across both guestrooms and meeting facilities.",
        extra_prerequisites=[url_leaf]
    )

    rating_policy = evaluator.add_parallel(
        id="Nashville_Rating_Policy",
        desc="Verify the hotel meets star rating and cancellation policy requirements",
        parent=node,
        critical=True
    )

    star_leaf = evaluator.add_leaf(
        id="Nashville_Star_Rating_3_Plus",
        desc="The hotel is rated 3-star or higher",
        parent=rating_policy,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel has a rating of at least 3 stars.",
        node=star_leaf,
        sources=sources,
        additional_instruction="Use booking platform star ratings if official site omits stars.",
        extra_prerequisites=[url_leaf]
    )

    cancel_leaf = evaluator.add_leaf(
        id="Nashville_Cancellation_48hrs",
        desc="The hotel allows cancellation up to 48 hours before check-in for a full refund",
        parent=rating_policy,
        critical=True
    )
    await evaluator.verify(
        claim="Cancellation up to 48 hours before check-in allows a full refund.",
        node=cancel_leaf,
        sources=sources,
        additional_instruction="Check rate/cancellation policy for a 48-hour full refund window.",
        extra_prerequisites=[url_leaf]
    )


async def verify_montauk(evaluator: Evaluator, parent: VerificationNode, data: Optional[MontaukHotel]) -> None:
    node = evaluator.add_parallel(
        id="Montauk_Accessible_Hotel",
        desc="Identify an ADA-compliant beachfront hotel in Montauk, New York with wheelchair accessible facilities",
        parent=parent,
        critical=False
    )

    core = data.core if data else HotelCore()
    features = data.features if data else MontaukFeatures()
    sources = gather_sources(core)

    url_leaf = evaluator.add_leaf(
        id="Montauk_URL_Reference",
        desc="Provide the official hotel website URL or verified booking platform URL for the Montauk accessible beachfront hotel",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This webpage is the official hotel site or an authoritative booking platform page for {core.name or 'the selected hotel'}.",
        node=url_leaf,
        sources=sources,
        additional_instruction="Confirm page type and correspondence."
    )

    loc = evaluator.add_parallel(
        id="Montauk_Location",
        desc="Verify the hotel is a beachfront or oceanfront property in Montauk, New York",
        parent=node,
        critical=True
    )

    m_city_leaf = evaluator.add_leaf(
        id="Montauk_City_State",
        desc="The hotel is located in Montauk, New York",
        parent=loc,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel is located in Montauk, New York.",
        node=m_city_leaf,
        sources=sources,
        additional_instruction="Confirm via address.",
        extra_prerequisites=[url_leaf]
    )

    beachfront_leaf = evaluator.add_leaf(
        id="Montauk_Beachfront_Oceanfront",
        desc="The hotel is a beachfront or oceanfront property with direct beach or ocean access or views",
        parent=loc,
        critical=True
    )
    await evaluator.verify(
        claim="The property is beachfront or oceanfront, with direct beach or ocean access or views.",
        node=beachfront_leaf,
        sources=sources,
        additional_instruction="Look for 'beachfront', 'oceanfront', direct beach access, or ocean views.",
        extra_prerequisites=[url_leaf]
    )

    access = evaluator.add_parallel(
        id="Montauk_Accessibility_Compliance",
        desc="Verify the hotel meets ADA accessibility requirements for wheelchair users",
        parent=node,
        critical=True
    )

    ada_rooms_leaf = evaluator.add_leaf(
        id="Montauk_ADA_Accessible_Rooms",
        desc="The hotel has ADA-compliant wheelchair accessible guest rooms",
        parent=access,
        critical=True
    )
    await evaluator.verify(
        claim="ADA-compliant wheelchair accessible guest rooms are available.",
        node=ada_rooms_leaf,
        sources=sources,
        additional_instruction="Look for 'ADA Accessible', 'Wheelchair accessible rooms'.",
        extra_prerequisites=[url_leaf]
    )

    doorway_leaf = evaluator.add_leaf(
        id="Montauk_Doorway_Width_32in",
        desc="Accessible room doorways are at least 32 inches wide",
        parent=access,
        critical=True
    )
    await evaluator.verify(
        claim="Accessible guest room doorways are at least 32 inches wide.",
        node=doorway_leaf,
        sources=sources,
        additional_instruction="Look for accessibility features listing doorway width ≥ 32 inches.",
        extra_prerequisites=[url_leaf]
    )

    hallway_leaf = evaluator.add_leaf(
        id="Montauk_Hallway_Width_36in",
        desc="Hallways are at least 36 inches wide to accommodate wheelchairs",
        parent=access,
        critical=True
    )
    await evaluator.verify(
        claim="Hallways are at least 36 inches wide to accommodate wheelchairs.",
        node=hallway_leaf,
        sources=sources,
        additional_instruction="Look for common area accessibility specs; hallways width ≥ 36 inches.",
        extra_prerequisites=[url_leaf]
    )

    # Vertical access - sequential, set parent non-critical to satisfy framework constraints
    vert = evaluator.add_sequential(
        id="Montauk_Vertical_Access",
        desc="Verify the hotel provides appropriate elevator or ground floor access for wheelchair users",
        parent=node,
        critical=False
    )

    multi_story_leaf = evaluator.add_leaf(
        id="Montauk_Multi_Story_Check",
        desc="Determine if the property is multi-story (more than one floor with guest rooms)",
        parent=vert,
        critical=False
    )
    # Verify the multi-story status if stated; if not stated, likely fail but non-critical
    if features.is_multi_story is True:
        ms_claim = "The property is multi-story with guest rooms on multiple floors."
    elif features.is_multi_story is False:
        ms_claim = "The property is a single-story with guest rooms on one floor."
    else:
        ms_claim = "The property’s number of guest room floors is stated on the page."
    await evaluator.verify(
        claim=ms_claim,
        node=multi_story_leaf,
        sources=sources,
        additional_instruction="Confirm whether the hotel has multiple guest room floors or is single-story when such information is available.",
        extra_prerequisites=[url_leaf]
    )

    elevator_leaf = evaluator.add_leaf(
        id="Montauk_Elevator_or_Ground_Floor",
        desc="If multi-story, the hotel has elevator access to all floors OR has ground floor accessible rooms available",
        parent=vert,
        critical=True
    )
    await evaluator.verify(
        claim="If the property is multi-story, it provides elevator access to all floors or has ground floor accessible rooms available; if single-story, this requirement is satisfied.",
        node=elevator_leaf,
        sources=sources,
        additional_instruction="Treat single-story properties as satisfying the requirement; otherwise, verify elevator access or ground-floor accessible rooms.",
        extra_prerequisites=[url_leaf, multi_story_leaf]
    )

    addl = evaluator.add_parallel(
        id="Montauk_Additional_Requirements",
        desc="Verify the hotel provides parking, dining, and appropriate check-in timing",
        parent=node,
        critical=True
    )

    m_parking_leaf = evaluator.add_leaf(
        id="Montauk_On_Site_Parking",
        desc="The hotel has on-site parking facilities",
        parent=addl,
        critical=True
    )
    await evaluator.verify(
        claim="On-site parking facilities are available.",
        node=m_parking_leaf,
        sources=sources,
        additional_instruction="Any on-site parking (lot, garage, valet) satisfies.",
        extra_prerequisites=[url_leaf]
    )

    m_dining_leaf = evaluator.add_leaf(
        id="Montauk_On_Site_Dining",
        desc="The hotel has an on-site restaurant or dining facility",
        parent=addl,
        critical=True
    )
    await evaluator.verify(
        claim="There is an on-site restaurant or dining facility.",
        node=m_dining_leaf,
        sources=sources,
        additional_instruction="Check amenities/dining pages.",
        extra_prerequisites=[url_leaf]
    )

    m_checkin_leaf = evaluator.add_leaf(
        id="Montauk_Check_In_3PM_Earlier",
        desc="Standard check-in time is 3:00 PM or earlier",
        parent=addl,
        critical=True
    )
    await evaluator.verify(
        claim="Standard check-in time is 3:00 PM or earlier.",
        node=m_checkin_leaf,
        sources=sources,
        additional_instruction="Confirm check-in time listed on policies; 3pm or earlier satisfies.",
        extra_prerequisites=[url_leaf]
    )


# =========================
# Main Evaluation
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
) -> Dict:
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_multi_city_hotels(),
        template_class=MultiCityExtraction,
        extraction_name="multi_city_hotels"
    )

    # Top-level nodes for each city (parallel under root)
    # Phoenix
    await verify_phoenix(evaluator, root, extracted.phoenix if extracted and extracted.phoenix else PhoenixHotel())
    # San Antonio
    await verify_san_antonio(evaluator, root, extracted.san_antonio if extracted and extracted.san_antonio else SanAntonioHotel())
    # Nashville
    await verify_nashville(evaluator, root, extracted.nashville if extracted and extracted.nashville else NashvilleHotel())
    # Montauk
    await verify_montauk(evaluator, root, extracted.montauk if extracted and extracted.montauk else MontaukHotel())

    return evaluator.get_summary()