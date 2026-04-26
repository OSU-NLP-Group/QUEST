import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_wildlife_vacation_apr2026"
TASK_DESCRIPTION = (
    "Comprehensive 5-day wildlife-focused California vacation plan for Apr 12–16, 2026 for 2 adults + 1 child (age 10) "
    "traveling with a small dog (18 lb), including required Safari Park activities, nearby pet-friendly hotel, one-night "
    "pet-friendly camping on Apr 15, and round-trip travel from Bangor, ME, with required reservation/policy compliance "
    "details and supporting URLs."
)

# Ground truth/reference facts to record in summary (not used for automated scoring directly)
GROUND_TRUTH_INFO = {
    "elephant_valley_opening": "Denny Sanford Elephant Valley opens March 5, 2026 at 12:00 PM (noon).",
    "safari_park_hours_winter_spring": "San Diego Zoo Safari Park operating hours for Jan 5–Jun 12, 2026 are typically 9:00 AM–5:00 PM.",
    "wildlife_safari_duration": "Wildlife Safari tour is 90 minutes.",
    "ultimate_safari_reservation": "Ultimate Safari requires at least 72-hour advance reservation via phone (619-718-3000). Separate park admission required.",
    "tours_child_supervision": "Ages 15 and younger must be accompanied by a paid adult for Safari Park tours.",
    "pets_not_allowed_in_park": "Pets are not allowed inside the Safari Park.",
    "breeze_pet_policy": "Breeze in-cabin pets: combined pet + carrier must be under 25 lb; $99 per one-way segment.",
    "ca_state_parks_reserve_window": "California State Parks reservations via ReserveCalifornia: rolling 6-month window, new inventory releases at 8:00 AM PST.",
    "ca_state_parks_online_fee": "ReserveCalifornia imposes an $8 online camping reservation fee.",
    "campground_pet_rules": "Leash length ≤ 6 feet; pets must not be left unattended."
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TripOverviewExtraction(BaseModel):
    trip_start_date: Optional[str] = None
    trip_end_date: Optional[str] = None
    adults_count: Optional[str] = None
    child_count: Optional[str] = None
    child_age: Optional[str] = None
    dog_weight_lb: Optional[str] = None


class TravelExtraction(BaseModel):
    breeze_note_statement: Optional[str] = None
    itinerary_routing: Optional[str] = None
    airlines_used: List[str] = Field(default_factory=list)
    pet_policy_handling_statement: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)          # general travel support (routes/policies)
    breeze_route_urls: List[str] = Field(default_factory=list)        # specific Breeze route info
    breeze_policy_urls: List[str] = Field(default_factory=list)       # specific Breeze pet policy pages


class SafariExtraction(BaseModel):
    visit_date: Optional[str] = None
    elephant_valley_included_statement: Optional[str] = None
    scheduled_times: List[str] = Field(default_factory=list)  # e.g., ["10:00 AM Wildlife Safari", "1:30 PM Ultimate Safari"]
    wildlife_safari_90min_statement: Optional[str] = None
    ultimate_safari_booking_statement: Optional[str] = None  # include "72 hours", "619-718-3000"
    ultimate_safari_separate_admission_ack: Optional[str] = None
    child_supervision_statement: Optional[str] = None
    pets_not_allowed_ack: Optional[str] = None
    dog_care_arrangement: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)  # generic Safari Park refs
    elephant_valley_urls: List[str] = Field(default_factory=list)
    hours_urls: List[str] = Field(default_factory=list)
    tours_urls: List[str] = Field(default_factory=list)
    pet_policy_urls: List[str] = Field(default_factory=list)


class HotelExtraction(BaseModel):
    hotel_name: Optional[str] = None
    hotel_distance_miles_to_safari_park: Optional[str] = None
    pet_weight_limit_lb: Optional[str] = None
    stay_dates: Optional[str] = None
    booking_details: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)  # hotel pet policy page and/or proximity info


class CampingExtraction(BaseModel):
    camping_date: Optional[str] = None
    campground_name: Optional[str] = None
    park_type: Optional[str] = None  # "California State Park" or "National Park"
    pet_friendly_ack: Optional[str] = None
    reservation_platform: Optional[str] = None  # "ReserveCalifornia.com" or "Recreation.gov"
    reservation_platform_urls: List[str] = Field(default_factory=list)
    ca_state_parks_window_ack: Optional[str] = None
    ca_state_parks_release_time_ack: Optional[str] = None
    ca_state_parks_online_fee_ack: Optional[str] = None
    min_advance_hours_ack: Optional[str] = None  # "48 hours"
    pet_rules_leash_length_ack: Optional[str] = None
    pet_rules_no_unattended_ack: Optional[str] = None
    campground_pet_policy_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_overview() -> str:
    return """
    Extract the trip overview details exactly as stated in the answer:
    - trip_start_date: The stated start date of the trip (e.g., "April 12, 2026" or "Apr 12, 2026")
    - trip_end_date: The stated end date of the trip (e.g., "April 16, 2026" or "Apr 16, 2026")
    - adults_count: Stated number of adults (as a string, e.g., "2")
    - child_count: Stated number of children (as a string, e.g., "1")
    - child_age: Stated child age (as a string, e.g., "10")
    - dog_weight_lb: Stated dog weight in pounds (as a string, e.g., "18 pounds")
    If any field is not explicitly stated, return null for that field.
    """


def prompt_extract_travel() -> str:
    return """
    Extract the round-trip travel details and supporting references exactly as stated in the answer:
    - breeze_note_statement: The sentence noting Breeze operates from Bangor to Orlando/Tampa/Raleigh but not directly to California (if present).
    - itinerary_routing: A concise summary of the stated round-trip routing from Bangor, ME to California and back.
    - airlines_used: A list of airline names explicitly mentioned in the itinerary (e.g., ["Delta", "United", "Breeze"]).
    - pet_policy_handling_statement: The sentence describing how the dog will be handled under airline pet policies (eligibility, carrier/weight constraints, and fees).
    - supporting_urls: URLs provided for travel-related claims (routes or airline pet policy pages).
    - breeze_route_urls: URLs specifically supporting the Breeze route limitation from Bangor (if any).
    - breeze_policy_urls: URLs specifically supporting Breeze's pet policy (if any).
    Return null for any missing field; return empty lists for missing URL groups.
    """


def prompt_extract_safari() -> str:
    return """
    Extract San Diego Zoo Safari Park plan details, requirements, and supporting references exactly as stated in the answer:
    - visit_date: The planned Safari Park visit date.
    - elephant_valley_included_statement: Statement indicating Denny Sanford Elephant Valley will be visited.
    - scheduled_times: A list of planned times for activities at the Safari Park (e.g., "10:00 AM Wildlife Safari").
    - wildlife_safari_90min_statement: Statement confirming the Wildlife Safari tour is 90 minutes.
    - ultimate_safari_booking_statement: Statement confirming Ultimate Safari will be reserved at least 72 hours in advance via phone 619-718-3000.
    - ultimate_safari_separate_admission_ack: Statement acknowledging separate park admission is required for Ultimate Safari.
    - child_supervision_statement: Statement confirming the 10-year-old will be accompanied by a paid adult for tours (applies to ages 15 and younger).
    - pets_not_allowed_ack: Statement acknowledging pets are not allowed inside the Safari Park.
    - dog_care_arrangement: The arrangement for caring for the dog during the Safari Park visit/tours.
    - supporting_urls: General Safari Park supporting URLs provided in the answer.
    - elephant_valley_urls: URLs specifically about Elephant Valley (opening info).
    - hours_urls: URLs about Safari Park operating hours.
    - tours_urls: URLs about Wildlife Safari / Ultimate Safari policies/details.
    - pet_policy_urls: URLs about Safari Park pet restrictions.
    Return null for missing scalar fields; return empty lists for missing URL groups.
    """


def prompt_extract_hotel() -> str:
    return """
    Extract pet-friendly hotel details exactly as stated in the answer:
    - hotel_name: The selected hotel's name.
    - hotel_distance_miles_to_safari_park: The stated distance (or clear wording indicating within 10 miles) between the hotel and Safari Park.
    - pet_weight_limit_lb: The stated pet weight limit in the hotel's policy (e.g., "25 pounds").
    - stay_dates: The stated hotel stay dates (ensure they align within Apr 12–16, 2026 and coordinate with Apr 15 camping).
    - booking_details: How/where to book (e.g., site link or direct).
    - supporting_urls: URL(s) to the hotel's pet policy and/or page indicating proximity to Safari Park.
    Return null for missing scalar fields; return empty list for missing URLs.
    """


def prompt_extract_camping() -> str:
    return """
    Extract the camping plan details exactly as stated in the answer:
    - camping_date: The camping night date (should be Apr 15, 2026).
    - campground_name: The selected campground's name.
    - park_type: Either "California State Park" or "National Park".
    - pet_friendly_ack: Statement confirming pet-friendly campground/park.
    - reservation_platform: Either "ReserveCalifornia.com" (for CA State Parks) or "Recreation.gov" (for National Parks).
    - reservation_platform_urls: URLs for the reservation platform page used/referenced.
    - ca_state_parks_window_ack: Statement about booking within rolling 6-month window (only relevant for CA State Parks).
    - ca_state_parks_release_time_ack: Statement about 8:00 AM PST release time (only relevant for CA State Parks).
    - ca_state_parks_online_fee_ack: Statement about $8 online camping reservation fee (only relevant for CA State Parks).
    - min_advance_hours_ack: Statement confirming reservation made at least 48 hours in advance.
    - pet_rules_leash_length_ack: Statement about leash length ≤ 6 feet.
    - pet_rules_no_unattended_ack: Statement about pets must not be left unattended.
    - campground_pet_policy_urls: URLs for campground/park pet policy page relied upon.
    Return null for missing scalar fields; return empty lists for missing URL groups.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def parse_first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(\d+)", text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def normalize_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                result.append(u)
    return result


def str_contains(s: Optional[str], token: str) -> bool:
    return (s or "").lower().find(token.lower()) >= 0


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_trip_overview_subtree(evaluator: Evaluator, parent_node, trip: TripOverviewExtraction) -> None:
    node = evaluator.add_parallel(
        id="Trip_Overview",
        desc="Plan states trip dates and travelers (family + dog).",
        parent=parent_node,
        critical=True
    )

    # Trip dates specified
    dates_leaf = evaluator.add_leaf(
        id="Trip_Dates_Specified",
        desc="Trip dates are specified as Apr 12–16, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan states the trip dates as Apr 12–16, 2026.",
        node=dates_leaf,
        additional_instruction="Verify from the answer text only; check for clear mention of Apr 12–16, 2026."
    )

    # Family composition specified
    fam_leaf = evaluator.add_leaf(
        id="Family_Composition_Specified",
        desc="Plan specifies 2 adults and 1 child age 10.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies a family of 2 adults and 1 child age 10.",
        node=fam_leaf,
        additional_instruction="Verify from the answer text only."
    )

    # Dog size specified
    dog_leaf = evaluator.add_leaf(
        id="Dog_Size_Specified",
        desc="Plan specifies traveling with a small dog weighing 18 pounds.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies traveling with a small dog weighing 18 pounds.",
        node=dog_leaf,
        additional_instruction="Verify from the answer text only."
    )


async def build_travel_subtree(evaluator: Evaluator, parent_node, travel: TravelExtraction, trip: TripOverviewExtraction) -> None:
    node = evaluator.add_parallel(
        id="Round_Trip_Travel_From_Bangor",
        desc="Round-trip travel arrangements from Bangor, ME to California and back, including the required Breeze route limitation note and pet-policy handling, with supporting URLs.",
        parent=parent_node,
        critical=True
    )

    # Breeze not direct to California noted (verify via URLs)
    breeze_urls = normalize_urls(travel.supporting_urls, travel.breeze_route_urls)
    breeze_note_leaf = evaluator.add_leaf(
        id="Breeze_Not_Direct_To_California_Noted",
        desc="Plan notes Breeze operates from Bangor to Orlando/Tampa/Raleigh but not directly to California.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Breeze Airways operates from Bangor (BGR) to Orlando, Tampa, and Raleigh, but does not operate directly to California.",
        node=breeze_note_leaf,
        sources=breeze_urls if breeze_urls else None,
        additional_instruction="Use official Breeze route/schedule pages or credible references among the provided URLs."
    )

    # Round-trip itinerary provided (answer text)
    itin_leaf = evaluator.add_leaf(
        id="Round_Trip_Itinerary_Provided",
        desc="Plan provides a specific round-trip travel routing from Bangor, ME to California and back (may include connections/other airlines).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan provides a specific round-trip routing from Bangor, ME to California and back.",
        node=itin_leaf,
        additional_instruction="Verify from the answer text only; look for concrete routing/airlines/connection details."
    )

    # Air travel pet policy addressed (answer text)
    pet_policy_leaf = evaluator.add_leaf(
        id="Air_Travel_Pet_Policy_Addressed",
        desc="Plan addresses how the dog will be transported on the chosen air itinerary (pet eligibility/weight+carrier constraints and fees for the airline(s) used).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan addresses airline pet policy details for the dog, including eligibility, carrier/weight constraints, and applicable fees.",
        node=pet_policy_leaf,
        additional_instruction="Verify from the answer text only."
    )

    # Breeze pet policy compliance if used (custom: pass when not used)
    airlines_used_lower = [a.lower() for a in (travel.airlines_used or [])]
    breeze_used = any("breeze" in a for a in airlines_used_lower) or str_contains(travel.itinerary_routing, "Breeze")
    # Compute a conservative compliance check:
    dog_weight_int = parse_first_int(trip.dog_weight_lb)
    compliance = True if not breeze_used else (dog_weight_int is not None and dog_weight_int <= 25)
    evaluator.add_custom_node(
        result=compliance,
        id="Breeze_Pet_Policy_Compliance_If_Used",
        desc="If Breeze is used for any segment, plan confirms compliance with Breeze pet policy: combined pet+carrier under 25 lb and $99 per one-way journey. If Breeze is not used, this requirement is considered satisfied.",
        parent=node,
        critical=True
    )

    # Travel supporting URLs provided (existence check)
    urls_exist = any([travel.supporting_urls, travel.breeze_route_urls, travel.breeze_policy_urls])
    evaluator.add_custom_node(
        result=bool(urls_exist),
        id="Travel_Supporting_URLs_Provided",
        desc="Plan provides at least one supporting reference URL covering the Breeze route limitation note and/or airline pet policy information relied upon.",
        parent=node,
        critical=True
    )


async def build_safari_subtree(evaluator: Evaluator, parent_node, safari: SafariExtraction) -> None:
    node = evaluator.add_parallel(
        id="Safari_Park_Visit_And_Required_Tours",
        desc="Safari Park visit during Apr 12–16, 2026 including Elephant Valley and both required safari experiences, meeting booking/timing, child-supervision, and pet-exclusion requirements, with supporting URLs.",
        parent=parent_node,
        critical=True
    )

    # Safari Park visit date within trip (Apr 12–16)
    visit_date_leaf = evaluator.add_leaf(
        id="Safari_Park_Visit_Date_Within_Trip",
        desc="Safari Park visit date is specified and falls within Apr 12–16, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Safari Park visit date is specified and falls within Apr 12–16, 2026.",
        node=visit_date_leaf,
        additional_instruction="Verify from the answer text only."
    )

    # Elephant Valley included and post opening (answer text check)
    ev_leaf = evaluator.add_leaf(
        id="Elephant_Valley_Included_And_Post_Opening",
        desc="Plan includes visiting Denny Sanford Elephant Valley and confirms the visit occurs after its opening (Mar 5, 2026 at noon).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes visiting Denny Sanford Elephant Valley and confirms the visit occurs after its opening (March 5, 2026 at noon).",
        node=ev_leaf,
        additional_instruction="Verify from the answer text only."
    )

    # Operating hours respected (answer text)
    hours_leaf = evaluator.add_leaf(
        id="Operating_Hours_Respected",
        desc="Safari Park activities are scheduled within operating hours (9:00 AM–5:00 PM for Jan 5–Jun 12, 2026).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The scheduled Safari Park activities are within the operating hours (9:00 AM–5:00 PM for Jan 5–Jun 12, 2026).",
        node=hours_leaf,
        additional_instruction="Verify from the answer text only."
    )

    # Wildlife Safari booked and 90 minutes
    ws_leaf = evaluator.add_leaf(
        id="Wildlife_Safari_Booked_And_90_Min",
        desc="Plan includes booking the Wildlife Safari tour and reflects that it is 90 minutes.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes booking the Wildlife Safari tour and notes that it is 90 minutes.",
        node=ws_leaf,
        additional_instruction="Verify from the answer text only."
    )

    # Ultimate Safari booking compliance (72+ hours via phone)
    us_book_leaf = evaluator.add_leaf(
        id="Ultimate_Safari_Booking_Compliance",
        desc="Plan includes booking Ultimate Safari and states reservation is made at least 72 hours in advance via phone at 619-718-3000.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan states the Ultimate Safari reservation will be made at least 72 hours in advance via phone at 619-718-3000.",
        node=us_book_leaf,
        additional_instruction="Verify from the answer text only."
    )

    # Ultimate Safari separate admission acknowledged
    us_adm_leaf = evaluator.add_leaf(
        id="Ultimate_Safari_Separate_Admission_Acknowledged",
        desc="Plan acknowledges Ultimate Safari requires separate Safari Park admission (in addition to the tour booking).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan acknowledges that Ultimate Safari requires separate Safari Park admission.",
        node=us_adm_leaf,
        additional_instruction="Verify from the answer text only."
    )

    # Child supervision compliance
    child_leaf = evaluator.add_leaf(
        id="Child_Supervision_For_Tours_Compliance",
        desc="Plan confirms the child (age 10) is accompanied by a paid adult for Safari Park tours (rule applies to ages 15 and younger).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan confirms that the 10-year-old child will be accompanied by a paid adult for Safari Park tours.",
        node=child_leaf,
        additional_instruction="Verify from the answer text only."
    )

    # Pets not allowed in park acknowledged
    pets_leaf = evaluator.add_leaf(
        id="Pets_Not_Allowed_In_Park_Acknowledged",
        desc="Plan acknowledges pets are not allowed inside the Safari Park.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan acknowledges pets are not allowed inside the Safari Park.",
        node=pets_leaf,
        additional_instruction="Verify from the answer text only."
    )

    # Dog care arrangement during park visit
    care_leaf = evaluator.add_leaf(
        id="Dog_Care_Arrangement_During_Park_Visit",
        desc="Plan provides a concrete arrangement for the dog during the Safari Park visit/tours (since the dog cannot enter).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan provides a concrete arrangement for caring for the dog during the Safari Park visit/tours.",
        node=care_leaf,
        additional_instruction="Verify from the answer text only."
    )

    # Safari Park supporting URLs provided (existence)
    safari_urls_exist = any([safari.supporting_urls, safari.elephant_valley_urls, safari.hours_urls, safari.tours_urls, safari.pet_policy_urls])
    evaluator.add_custom_node(
        result=bool(safari_urls_exist),
        id="Safari_Park_Supporting_URLs_Provided",
        desc="Plan provides supporting reference URL(s) for Safari Park requirements (Elephant Valley opening, hours, tour reservation rules, age supervision rule, and pet restriction).",
        parent=node,
        critical=True
    )


async def build_hotel_subtree(evaluator: Evaluator, parent_node, hotel: HotelExtraction) -> None:
    node = evaluator.add_parallel(
        id="Pet_Friendly_Hotel_Within_10_Miles",
        desc="Pet-friendly hotel within 10 miles of the Safari Park accommodating dogs up to 25 lb, with stay dates, booking details, and supporting URLs.",
        parent=parent_node,
        critical=True
    )

    # Hotel within 10 miles (answer text check)
    dist_leaf = evaluator.add_leaf(
        id="Hotel_Within_10_Miles",
        desc="Selected hotel is within 10 miles of the Safari Park.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan states the selected hotel is within 10 miles of the Safari Park.",
        node=dist_leaf,
        additional_instruction="Verify from the answer text only."
    )

    # Hotel allows dogs up to 25 lb (verify via URLs)
    pet_weight_leaf = evaluator.add_leaf(
        id="Hotel_Allows_Dogs_Up_To_25lb",
        desc="Hotel pet policy accommodates dogs up to 25 pounds (covers the 18-lb dog).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The hotel's pet policy allows dogs up to 25 pounds.",
        node=pet_weight_leaf,
        sources=hotel.supporting_urls if hotel.supporting_urls else None,
        additional_instruction="Use the hotel's official pet policy page among the provided URLs."
    )

    # Hotel stay dates specified
    stay_leaf = evaluator.add_leaf(
        id="Hotel_Stay_Dates_Specified",
        desc="Hotel stay dates within Apr 12–16, 2026 are specified and coordinated with the Apr 15 camping night.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies hotel stay dates within Apr 12–16, 2026 and coordinates around the Apr 15 camping night.",
        node=stay_leaf,
        additional_instruction="Verify from the answer text only."
    )

    # Booking details provided
    booking_leaf = evaluator.add_leaf(
        id="Hotel_Booking_Details_Provided",
        desc="Plan provides booking details sufficient to execute the booking (property name, dates, and how/where to book).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan provides booking details sufficient to execute the hotel booking (property name, dates, and how/where to book).",
        node=booking_leaf,
        additional_instruction="Verify from the answer text only."
    )

    # Hotel supporting URLs provided (existence)
    evaluator.add_custom_node(
        result=bool(hotel.supporting_urls),
        id="Hotel_Supporting_URLs_Provided",
        desc="Plan provides supporting URL(s) for the selected hotel’s pet policy (including weight limit) and/or the hotel’s proximity to the Safari Park.",
        parent=node,
        critical=True
    )


async def build_camping_subtree(evaluator: Evaluator, parent_node, camp: CampingExtraction) -> None:
    node = evaluator.add_parallel(
        id="One_Night_Camping_April_15",
        desc="One-night camping for Apr 15, 2026 at a pet-friendly campground in a CA State Park or National Park in California, using the correct system and respecting booking windows and pet rules, with supporting URLs.",
        parent=parent_node,
        critical=True
    )

    # Camping night is April 15
    date_leaf = evaluator.add_leaf(
        id="Camping_Night_Is_April_15_2026",
        desc="Camping is scheduled for the night of Apr 15, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan schedules camping for the night of April 15, 2026.",
        node=date_leaf,
        additional_instruction="Verify from the answer text only."
    )

    # Campground is CA State or National Park and pet-friendly (verify via URLs if available)
    pet_friendly_leaf = evaluator.add_leaf(
        id="Campground_Is_CA_State_Or_National_Park_And_Pet_Friendly",
        desc="Selected campground is in either a California State Park or a National Park in California and is pet-friendly.",
        parent=node,
        critical=True
    )
    urls_pf = normalize_urls(camp.campground_pet_policy_urls, camp.reservation_platform_urls)
    await evaluator.verify(
        claim="The selected campground is in a California State Park or National Park in California and is pet-friendly.",
        node=pet_friendly_leaf,
        sources=urls_pf if urls_pf else None,
        additional_instruction="Use campground/park official pages provided."
    )

    # Correct reservation system
    res_sys_leaf = evaluator.add_leaf(
        id="Correct_Reservation_System",
        desc="Plan uses ReserveCalifornia.com for CA State Parks or Recreation.gov for National Park campgrounds (as applicable).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan uses the correct reservation platform: ReserveCalifornia.com for CA State Parks or Recreation.gov for National Park campgrounds.",
        node=res_sys_leaf,
        sources=camp.reservation_platform_urls if camp.reservation_platform_urls else None,
        additional_instruction="Use the official reservation platform URL(s) provided."
    )

    # CA State Parks reservation window + release time (conditional; pass if not state parks)
    is_state_park = str_contains(camp.park_type, "state")
    if is_state_park:
        window_leaf = evaluator.add_leaf(
            id="CA_State_Parks_Reservation_Window_If_Applicable",
            desc="If a CA State Park campground is selected, plan confirms booking within the rolling 6-month window and notes the 8:00 AM PST release time.",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim="ReserveCalifornia uses a rolling 6-month booking window with new inventory released at 8:00 AM PST.",
            node=window_leaf,
            sources=camp.reservation_platform_urls if camp.reservation_platform_urls else None,
            additional_instruction="Verify both the 6-month window and the 8:00 AM PST release time using official ReserveCalifornia info among the provided URLs."
        )
        fee_leaf = evaluator.add_leaf(
            id="CA_State_Parks_Online_Fee_If_Applicable",
            desc="If a CA State Park campground is selected, plan notes the $8 online camping reservation fee.",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim="ReserveCalifornia charges an $8 online camping reservation fee.",
            node=fee_leaf,
            sources=camp.reservation_platform_urls if camp.reservation_platform_urls else None,
            additional_instruction="Verify using official ReserveCalifornia fee information among the provided URLs."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="CA_State_Parks_Reservation_Window_If_Applicable",
            desc="Not applicable (National Park selected): CA State Parks 6-month window and 8:00 AM PST release time requirement considered satisfied.",
            parent=node,
            critical=True
        )
        evaluator.add_custom_node(
            result=True,
            id="CA_State_Parks_Online_Fee_If_Applicable",
            desc="Not applicable (National Park selected): $8 ReserveCalifornia online camping reservation fee requirement considered satisfied.",
            parent=node,
            critical=True
        )

    # Minimum 48 hours advance (answer text)
    adv_leaf = evaluator.add_leaf(
        id="Minimum_48_Hours_Advance",
        desc="Plan confirms the camping reservation is made at least 48 hours in advance.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan confirms the camping reservation is made at least 48 hours in advance.",
        node=adv_leaf,
        additional_instruction="Verify from the answer text only."
    )

    # Campground pet rules leash length (verify via URLs)
    leash_leaf = evaluator.add_leaf(
        id="Campground_Pet_Rules_Leash_Length",
        desc="Plan states and complies with campground pet rule: leash length not more than 6 feet.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The campground/park pet rules require leash length not more than 6 feet.",
        node=leash_leaf,
        sources=camp.campground_pet_policy_urls if camp.campground_pet_policy_urls else None,
        additional_instruction="Verify using official campground/park pet policy URL(s)."
    )

    # Campground pet rules no unattended pets (verify via URLs)
    unattended_leaf = evaluator.add_leaf(
        id="Campground_Pet_Rules_No_Unattended_Pets",
        desc="Plan states and complies with campground pet rule: pets must not be left unattended.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The campground/park pet rules state pets must not be left unattended.",
        node=unattended_leaf,
        sources=camp.campground_pet_policy_urls if camp.campground_pet_policy_urls else None,
        additional_instruction="Verify using official campground/park pet policy URL(s)."
    )

    # Camping supporting URLs provided (existence)
    urls_exist = any([camp.reservation_platform_urls, camp.campground_pet_policy_urls])
    evaluator.add_custom_node(
        result=bool(urls_exist),
        id="Camping_Supporting_URLs_Provided",
        desc="Plan provides supporting URL(s) for the reservation platform used and the campground/park pet policy relied upon.",
        parent=node,
        critical=True
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
    Evaluate the answer for the comprehensive California wildlife vacation plan (Apr 12–16, 2026).
    """
    # Initialize evaluator (root is non-critical)
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

    # Record ground truth reference info (for human-readable context)
    evaluator.add_ground_truth(GROUND_TRUTH_INFO, gt_type="reference_facts")

    # Extract structured info (run concurrently)
    trip_task = evaluator.extract(
        prompt=prompt_extract_trip_overview(),
        template_class=TripOverviewExtraction,
        extraction_name="trip_overview"
    )
    travel_task = evaluator.extract(
        prompt=prompt_extract_travel(),
        template_class=TravelExtraction,
        extraction_name="travel_plan"
    )
    safari_task = evaluator.extract(
        prompt=prompt_extract_safari(),
        template_class=SafariExtraction,
        extraction_name="safari_plan"
    )
    hotel_task = evaluator.extract(
        prompt=prompt_extract_hotel(),
        template_class=HotelExtraction,
        extraction_name="hotel_plan"
    )
    camping_task = evaluator.extract(
        prompt=prompt_extract_camping(),
        template_class=CampingExtraction,
        extraction_name="camping_plan"
    )

    trip, travel, safari, hotel, camping = await asyncio.gather(
        trip_task, travel_task, safari_task, hotel_task, camping_task
    )

    # Create the top-level critical node representing the full plan
    plan_node = evaluator.add_parallel(
        id="Complete_California_Wildlife_Vacation_Plan",
        desc=("Comprehensive 5-day wildlife-focused California vacation plan for Apr 12–16, 2026 for 2 adults + 1 child "
              "(age 10) with a small dog (18 lb), including required Safari Park activities, nearby pet-friendly hotel, "
              "one-night pet-friendly camping on Apr 15, and round-trip travel from Bangor, ME, with required reservation/"
              "policy compliance details and supporting URLs."),
        parent=root,
        critical=True
    )

    # Build all subtrees
    await build_trip_overview_subtree(evaluator, plan_node, trip)
    await build_travel_subtree(evaluator, plan_node, travel, trip)
    await build_safari_subtree(evaluator, plan_node, safari)
    await build_hotel_subtree(evaluator, plan_node, hotel)
    await build_camping_subtree(evaluator, plan_node, camping)

    # Return evaluation summary
    return evaluator.get_summary()