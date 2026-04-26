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
TASK_ID = "pet_friendly_tennessee_trip_2026"
TASK_DESCRIPTION = (
    "A family from Hartford, Connecticut is planning a 3-day trip to Tennessee in April 2026 with their 20-pound "
    "Labrador puppy (4 months old). They want to fly on Breeze Airways, visit Dollywood during the special April 3-12, "
    "2026 period when extended operating hours are available, and include at least one additional pet-friendly outdoor "
    "activity during their stay. They also need pet-friendly accommodation in the Pigeon Forge area.\n\n"
    "Create a comprehensive travel plan that addresses:\n"
    "1) Flight on Breeze from Hartford, CT to Tennessee (route availability), verify puppy meets Breeze pet rules, and pet fees range;\n"
    "2) Dollywood visit (select a date within April 3–12, 2026; provide operating hours during that period; explain dog care since pets aren't allowed);\n"
    "3) One pet-friendly outdoor activity near Pigeon Forge with pet-allowed areas;\n"
    "4) A pet-friendly hotel in/near Pigeon Forge with convenient access to Dollywood.\n"
    "Provide supporting evidence with relevant URLs for each component."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FlightSection(BaseModel):
    tn_destination: Optional[str] = None
    route_statement: Optional[str] = None
    route_urls: List[str] = Field(default_factory=list)
    pet_weight_rule_statement: Optional[str] = None
    pet_age_rule_statement: Optional[str] = None
    pet_max_per_passenger_statement: Optional[str] = None
    pet_fee_range: Optional[str] = None
    pet_policy_urls: List[str] = Field(default_factory=list)
    pet_fee_urls: List[str] = Field(default_factory=list)
    dog_weight_mention: Optional[str] = None
    dog_age_mention: Optional[str] = None


class DollywoodSection(BaseModel):
    visit_date: Optional[str] = None
    operating_hours: Optional[str] = None
    hours_urls: List[str] = Field(default_factory=list)
    dog_care_arrangement: Optional[str] = None
    dog_care_urls: List[str] = Field(default_factory=list)


class ActivitySection(BaseModel):
    name: Optional[str] = None
    distance_or_time: Optional[str] = None
    allowed_areas: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HotelSection(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    pet_friendly_statement: Optional[str] = None
    dollywood_access_statement: Optional[str] = None
    extra_urls: List[str] = Field(default_factory=list)


class ItinerarySection(BaseModel):
    timeframe: Optional[str] = None
    days: List[str] = Field(default_factory=list)


class TripPlanExtraction(BaseModel):
    flight: FlightSection = Field(default_factory=FlightSection)
    dollywood: DollywoodSection = Field(default_factory=DollywoodSection)
    activity: ActivitySection = Field(default_factory=ActivitySection)
    hotel: HotelSection = Field(default_factory=HotelSection)
    itinerary: ItinerarySection = Field(default_factory=ItinerarySection)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_plan() -> str:
    return """
Extract the key facts from the answer for the four required components (Flight on Breeze, Dollywood visit in April 2026 with extended hours window, Pet-friendly outdoor activity, and Pet-friendly hotel near Pigeon Forge) plus the 3-day itinerary timing. Populate the JSON fields below with EXACT text from the answer where appropriate and collect all relevant URLs cited for each component.

Return a single JSON object with this shape:

{
  "flight": {
    "tn_destination": string or null,                         // The Tennessee city/airport named in the plan for Breeze flights
    "route_statement": string or null,                        // The plan's statement about Breeze route availability
    "route_urls": [urls...],                                  // URLs supporting Breeze route availability (Breeze route map, booking page, etc.)
    "pet_weight_rule_statement": string or null,              // The plan's statement about Breeze in-cabin pet weight rule
    "pet_age_rule_statement": string or null,                 // The plan's statement about Breeze minimum pet age rule
    "pet_max_per_passenger_statement": string or null,        // The plan's statement on max 1 pet per passenger rule
    "pet_fee_range": string or null,                          // The plan's stated pet fee range per flight
    "pet_policy_urls": [urls...],                             // URLs supporting Breeze pet policy details (weight/age/max-per-passenger)
    "pet_fee_urls": [urls...],                                // URLs specifically supporting the pet fee range, if separate
    "dog_weight_mention": string or null,                     // The plan's explicit mention of the puppy's weight (if present)
    "dog_age_mention": string or null                         // The plan's explicit mention of the puppy's age (if present)
  },
  "dollywood": {
    "visit_date": string or null,                             // The specific Dollywood visit date chosen in the window (as written in the answer)
    "operating_hours": string or null,                        // The operating hours stated for the April 3–12, 2026 period (as written)
    "hours_urls": [urls...],                                  // URLs supporting the hours (e.g., Dollywood calendar)
    "dog_care_arrangement": string or null,                   // The plan's dog-care arrangement explanation (e.g., Doggywood, local kennel)
    "dog_care_urls": [urls...]                                // URLs supporting no-pets policy and/or Doggywood/kennel info
  },
  "activity": {
    "name": string or null,                                   // The identified pet-friendly attraction (national park or major outdoor site)
    "distance_or_time": string or null,                       // The plan's stated driving distance/time or justification of proximity
    "allowed_areas": string or null,                          // The plan's stated pet-allowed areas/trails/sections
    "urls": [urls...]                                         // URLs supporting pet-allowed areas and attraction details
  },
  "hotel": {
    "name": string or null,                                   // The recommended hotel name
    "url": string or null,                                    // A URL for the hotel page (if provided)
    "pet_friendly_statement": string or null,                 // The plan's statement that the hotel is pet-friendly
    "dollywood_access_statement": string or null,             // The plan's statement of convenient access to Dollywood (e.g., '5 minutes away')
    "extra_urls": [urls...]                                   // Any additional URLs supporting hotel details
  },
  "itinerary": {
    "timeframe": string or null,                              // The plan’s overall timeframe (e.g., 'April 2026') if explicitly stated
    "days": [strings...]                                      // A list of day-by-day itinerary items; include one string for each day mentioned
  }
}

Rules:
- Do NOT invent any facts or URLs; only extract what appears in the answer.
- For all URL arrays, include only valid URLs that appear in the answer (plain or markdown).
- If multiple hotels/activities are listed, extract only the primary or first one.
- If some fields are not in the answer, use null (for strings) or [] (for lists).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _contains_april_2026(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return ("april" in t or "apr " in t or "apr." in t) and "2026" in t


def _extract_day_from_date_str(date_str: str) -> Optional[int]:
    # Try patterns like "April 7, 2026", "Apr 7 2026", "April 07", "Apr-7"
    patterns = [
        r"(?:april|apr\.?)\s*(\d{1,2})(?:\D|$)",
        r"(\d{1,2})\s*(?:april|apr\.?)"  # reverse order just in case
    ]
    s = date_str.lower()
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            try:
                day = int(m.group(1))
                return day
            except:
                pass
    # Try explicit window string like "April 3–12" (endash or hyphen)
    rng_patterns = [
        r"(?:april|apr\.?)\s*3\s*[-–—]\s*12",
        r"(?:april|apr\.?)\s*(?:3rd|third)\s*[-–—]\s*(?:12th|twelfth)"
    ]
    for pat in rng_patterns:
        if re.search(pat, s):
            # Not specific day, but implies within window; return a sentinel
            return 7  # some day in the window for acceptance
    return None


def _visit_date_within_april_3_12_2026(date_str: Optional[str]) -> bool:
    if not date_str:
        return False
    if not _contains_april_2026(date_str):
        return False
    day = _extract_day_from_date_str(date_str)
    if day is None:
        return False
    return 3 <= day <= 12


def _non_empty(lst: Optional[List[str]]) -> bool:
    return bool(lst and len([x for x in lst if x and x.strip()]) > 0)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_supporting_urls_nodes(
    evaluator: Evaluator,
    parent_node,
    extracted: TripPlanExtraction
) -> Dict[str, Any]:
    """
    Build and evaluate the Supporting_Evidence_URLs critical group first.
    Returns a dict with references to individual leaf nodes to be used as explicit prerequisites.
    """
    support_node = evaluator.add_parallel(
        id="supporting_evidence_urls",
        desc="Provides relevant supporting evidence URLs for each required component.",
        parent=parent_node,
        critical=True
    )

    # Flight URLs: any of route_urls, pet_policy_urls, pet_fee_urls
    flight_urls_all = _dedup_urls(
        (extracted.flight.route_urls or []) +
        (extracted.flight.pet_policy_urls or []) +
        (extracted.flight.pet_fee_urls or [])
    )
    flight_urls_leaf = evaluator.add_custom_node(
        result=_non_empty(flight_urls_all),
        id="Flight_URLs_Provided",
        desc="Includes at least one relevant URL supporting the Flight component claims.",
        parent=support_node,
        critical=True
    )

    # Dollywood URLs: any of hours_urls or dog_care_urls
    dolly_urls_all = _dedup_urls(
        (extracted.dollywood.hours_urls or []) +
        (extracted.dollywood.dog_care_urls or [])
    )
    dolly_urls_leaf = evaluator.add_custom_node(
        result=_non_empty(dolly_urls_all),
        id="Dollywood_URLs_Provided",
        desc="Includes at least one relevant URL supporting the Dollywood component claims.",
        parent=support_node,
        critical=True
    )

    # Outdoor Activity URLs
    outdoor_urls_all = _dedup_urls(extracted.activity.urls or [])
    outdoor_urls_leaf = evaluator.add_custom_node(
        result=_non_empty(outdoor_urls_all),
        id="Outdoor_Activity_URLs_Provided",
        desc="Includes at least one relevant URL supporting the Outdoor Activity component claims.",
        parent=support_node,
        critical=True
    )

    # Hotel URLs
    hotel_urls_all = _dedup_urls(([extracted.hotel.url] if extracted.hotel.url else []) + (extracted.hotel.extra_urls or []))
    hotel_urls_leaf = evaluator.add_custom_node(
        result=_non_empty(hotel_urls_all),
        id="Hotel_URLs_Provided",
        desc="Includes at least one relevant URL supporting the Accommodation component claims.",
        parent=support_node,
        critical=True
    )

    return {
        "flight_urls_leaf": flight_urls_leaf,
        "flight_urls": flight_urls_all,
        "dolly_urls_leaf": dolly_urls_leaf,
        "dolly_urls": dolly_urls_all,
        "outdoor_urls_leaf": outdoor_urls_leaf,
        "outdoor_urls": outdoor_urls_all,
        "hotel_urls_leaf": hotel_urls_leaf,
        "hotel_urls": hotel_urls_all,
    }


async def verify_trip_timing(
    evaluator: Evaluator,
    parent_node,
    extracted: TripPlanExtraction
) -> None:
    timing_node = evaluator.add_parallel(
        id="Trip_Length_And_Timing",
        desc="Plan reflects a 3-day trip in April 2026.",
        parent=parent_node,
        critical=True
    )

    # April_2026_Timing (existence/logic check)
    in_april_2026 = (
        _contains_april_2026(extracted.itinerary.timeframe) or
        _contains_april_2026(extracted.dollywood.visit_date)
    )
    evaluator.add_custom_node(
        result=in_april_2026,
        id="April_2026_Timing",
        desc="The plan’s dates/timeframe are in April 2026.",
        parent=timing_node,
        critical=True
    )

    # Three_Day_Itinerary (existence/logic check)
    has_three_days = bool(extracted.itinerary.days and len(extracted.itinerary.days) >= 3)
    evaluator.add_custom_node(
        result=has_three_days,
        id="Three_Day_Itinerary",
        desc="Provides a day-by-day itinerary covering 3 days.",
        parent=timing_node,
        critical=True
    )


async def verify_flight(
    evaluator: Evaluator,
    parent_node,
    extracted: TripPlanExtraction,
    supports: Dict[str, Any]
) -> None:
    flight_node = evaluator.add_parallel(
        id="Flight_Arrangements",
        desc="Flight plan elements required by the prompt for Breeze Airways travel with a puppy.",
        parent=parent_node,
        critical=True
    )

    # Breeze_Route_Available
    route_leaf = evaluator.add_leaf(
        id="Breeze_Route_Available",
        desc="Confirms whether Breeze Airways operates a route from Hartford, CT to a Tennessee destination.",
        parent=flight_node,
        critical=True
    )
    dest_text = extracted.flight.tn_destination or "a Tennessee airport"
    claim_route = (
        f"Breeze Airways operates scheduled service enabling travel between Hartford, CT (BDL) and {dest_text} in Tennessee, "
        f"as indicated on Breeze's route/network or booking pages."
    )
    await evaluator.verify(
        claim=claim_route,
        node=route_leaf,
        sources=supports.get("flight_urls", []),
        additional_instruction="Verify that the provided Breeze webpage(s) indicate service involving Hartford (BDL) and the specified Tennessee destination (direct or via Breeze network).",
        extra_prerequisites=[supports["flight_urls_leaf"]]
    )

    # Pet_Eligibility_Weight (verify the policy rule itself with URLs)
    weight_leaf = evaluator.add_leaf(
        id="Pet_Eligibility_Weight",
        desc="Verifies the puppy meets Breeze’s in-cabin weight rule: combined weight (pet + carrier) under 25 lbs.",
        parent=flight_node,
        critical=True
    )
    claim_weight = "Breeze Airways' in-cabin pet policy sets a maximum combined weight of 25 pounds for the pet plus carrier."
    await evaluator.verify(
        claim=claim_weight,
        node=weight_leaf,
        sources=supports.get("flight_urls", []),
        additional_instruction="Focus on confirming the pet+carrier 25 lb (11.3 kg) limit as stated on Breeze's official policy pages.",
        extra_prerequisites=[supports["flight_urls_leaf"]]
    )

    # Pet_Eligibility_Age (policy)
    age_leaf = evaluator.add_leaf(
        id="Pet_Eligibility_Age",
        desc="Verifies the puppy meets Breeze’s minimum age rule: at least 8 weeks old.",
        parent=flight_node,
        critical=True
    )
    claim_age = "Breeze Airways requires pets to be at least 8 weeks old to travel in cabin."
    await evaluator.verify(
        claim=claim_age,
        node=age_leaf,
        sources=supports.get("flight_urls", []),
        additional_instruction="Verify the minimum age requirement (8 weeks) on Breeze's pet policy page.",
        extra_prerequisites=[supports["flight_urls_leaf"]]
    )

    # Pet_Eligibility_Max_Per_Passenger (policy)
    max_per_leaf = evaluator.add_leaf(
        id="Pet_Eligibility_Max_Per_Passenger",
        desc="States/addresses Breeze’s limit of maximum 1 pet per passenger and aligns the plan accordingly.",
        parent=flight_node,
        critical=True
    )
    claim_max_per = "Breeze Airways allows a maximum of one pet per passenger for in-cabin travel."
    await evaluator.verify(
        claim=claim_max_per,
        node=max_per_leaf,
        sources=supports.get("flight_urls", []),
        additional_instruction="Verify Breeze's in-cabin pet limit is one pet per passenger.",
        extra_prerequisites=[supports["flight_urls_leaf"]]
    )

    # Pet_Fee_Range_Provided (policy/fee page)
    fee_leaf = evaluator.add_leaf(
        id="Pet_Fee_Range_Provided",
        desc="Provides the range of pet fees charged per flight (per one-way flight per pet carrier): $75–$99.",
        parent=flight_node,
        critical=True
    )
    claim_fee = "Breeze Airways charges a pet fee in the range of $75 to $99 per one-way flight per pet carrier."
    await evaluator.verify(
        claim=claim_fee,
        node=fee_leaf,
        sources=supports.get("flight_urls", []),
        additional_instruction="Confirm the pet fee amount range ($75–$99) from Breeze's official fees or pet policy pages.",
        extra_prerequisites=[supports["flight_urls_leaf"]]
    )


async def verify_dollywood(
    evaluator: Evaluator,
    parent_node,
    extracted: TripPlanExtraction,
    supports: Dict[str, Any]
) -> None:
    dolly_node = evaluator.add_parallel(
        id="Dollywood_Visit_Planning",
        desc="Dollywood plan elements for the Apr 3–12, 2026 extended-hours window, including dog-care arrangements.",
        parent=parent_node,
        critical=True
    )

    # Specific_Visit_Date_Selected (within Apr 3–12, 2026) - logic check based on answer text
    selected_within_window = _visit_date_within_april_3_12_2026(extracted.dollywood.visit_date)
    evaluator.add_custom_node(
        result=selected_within_window,
        id="Specific_Visit_Date_Selected",
        desc="Selects a specific Dollywood visit date that falls within April 3–12, 2026.",
        parent=dolly_node,
        critical=True
    )

    # Operating_Hours_Provided (verify with URLs)
    hours_leaf = evaluator.add_leaf(
        id="Operating_Hours_Provided",
        desc="Provides Dollywood operating hours during April 3–12, 2026 (extended hours window).",
        parent=dolly_node,
        critical=True
    )
    hours_text = extracted.dollywood.operating_hours or "the stated extended operating hours"
    claim_hours = f"During April 3–12, 2026, Dollywood's posted operating hours are {hours_text} (extended hours period)."
    await evaluator.verify(
        claim=claim_hours,
        node=hours_leaf,
        sources=supports.get("dolly_urls", []),
        additional_instruction="Use Dollywood's official calendar or hours page for the Apr 3–12, 2026 window to verify the stated hours.",
        extra_prerequisites=[supports["dolly_urls_leaf"]]
    )

    # Dog_Care_Arrangement_Explained (existence check — explanation provided)
    has_dog_care_expl = bool(extracted.dollywood.dog_care_arrangement and extracted.dollywood.dog_care_arrangement.strip())
    evaluator.add_custom_node(
        result=has_dog_care_expl,
        id="Dog_Care_Arrangement_Explained",
        desc="Explains how the family will arrange care for the dog because pets are not allowed inside Dollywood (except service animals).",
        parent=dolly_node,
        critical=True
    )


async def verify_outdoor_activity(
    evaluator: Evaluator,
    parent_node,
    extracted: TripPlanExtraction,
    supports: Dict[str, Any]
) -> None:
    activity_node = evaluator.add_parallel(
        id="Pet-Friendly_Outdoor_Activity",
        desc="Additional pet-friendly outdoor activity requirement near Pigeon Forge with pet access details.",
        parent=parent_node,
        critical=True
    )

    # Attraction_Identified (verify with URLs it is a pet-friendly outdoor attraction near Pigeon Forge)
    attraction_leaf = evaluator.add_leaf(
        id="Attraction_Identified",
        desc="Identifies a specific pet-friendly national park or major outdoor attraction within reasonable driving distance of Pigeon Forge.",
        parent=activity_node,
        critical=True
    )
    activity_name = extracted.activity.name or "the specified attraction"
    claim_attraction = (
        f"{activity_name} is a pet-friendly outdoor attraction (e.g., national park or major outdoor site) in the Pigeon Forge area."
    )
    await evaluator.verify(
        claim=claim_attraction,
        node=attraction_leaf,
        sources=supports.get("outdoor_urls", []),
        additional_instruction=(
            "Confirm the attraction is a national park or major outdoor destination and allows dogs (at least in some areas)."
        ),
        extra_prerequisites=[supports["outdoor_urls_leaf"]]
    )

    # Reasonable_Driving_Distance_Addressed (existence/logic check)
    has_distance_info = bool(extracted.activity.distance_or_time and extracted.activity.distance_or_time.strip())
    evaluator.add_custom_node(
        result=has_distance_info,
        id="Reasonable_Driving_Distance_Addressed",
        desc="Provides driving time/distance or justification that the attraction is within reasonable driving distance of Pigeon Forge.",
        parent=activity_node,
        critical=True
    )

    # Pet_Allowed_Areas_Specified (verify areas with URLs)
    areas_leaf = evaluator.add_leaf(
        id="Pet_Allowed_Areas_Specified",
        desc="Specifies which areas/trails/sections allow pets at the identified attraction.",
        parent=activity_node,
        critical=True
    )
    allowed_text = extracted.activity.allowed_areas or "the specified areas or trails"
    claim_areas = f"At {activity_name}, pets are allowed in {allowed_text}."
    await evaluator.verify(
        claim=claim_areas,
        node=areas_leaf,
        sources=supports.get("outdoor_urls", []),
        additional_instruction="Verify the pet-allowed areas/trails from the attraction's official page or authoritative resource.",
        extra_prerequisites=[supports["outdoor_urls_leaf"]]
    )


async def verify_hotel(
    evaluator: Evaluator,
    parent_node,
    extracted: TripPlanExtraction,
    supports: Dict[str, Any]
) -> None:
    hotel_node = evaluator.add_parallel(
        id="Hotel_Accommodation",
        desc="Accommodation requirement in/near Pigeon Forge with pet-friendly status and Dollywood access.",
        parent=parent_node,
        critical=True
    )

    # Hotel_Recommended (existence check)
    has_hotel = bool(extracted.hotel.name and extracted.hotel.name.strip())
    evaluator.add_custom_node(
        result=has_hotel,
        id="Hotel_Recommended",
        desc="Recommends at least one specific hotel in or near Pigeon Forge.",
        parent=hotel_node,
        critical=True
    )

    # Hotel_Pet_Friendly_Confirmed (verify with hotel URLs)
    pet_friendly_leaf = evaluator.add_leaf(
        id="Hotel_Pet_Friendly_Confirmed",
        desc="Confirms the recommended hotel is pet-friendly.",
        parent=hotel_node,
        critical=True
    )
    hotel_name = extracted.hotel.name or "the selected hotel"
    claim_pet_friendly = f"{hotel_name} is pet-friendly."
    await evaluator.verify(
        claim=claim_pet_friendly,
        node=pet_friendly_leaf,
        sources=supports.get("hotel_urls", []),
        additional_instruction="Verify on the hotel's official page (or booking page) that pets are allowed.",
        extra_prerequisites=[supports["hotel_urls_leaf"]]
    )

    # Access_To_Dollywood_Addressed (verify proximity using hotel URL, if stated)
    access_leaf = evaluator.add_leaf(
        id="Access_To_Dollywood_Addressed",
        desc="Explains/indicates the hotel provides convenient access to Dollywood (e.g., proximity/location).",
        parent=hotel_node,
        critical=True
    )
    claim_access = (
        f"{hotel_name} provides convenient access to Dollywood (e.g., close proximity or short travel time)."
    )
    await evaluator.verify(
        claim=claim_access,
        node=access_leaf,
        sources=supports.get("hotel_urls", []),
        additional_instruction="Accept if the hotel page indicates proximity to Dollywood or location within Pigeon Forge close to the park.",
        extra_prerequisites=[supports["hotel_urls_leaf"]]
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
    Evaluate an answer for the pet-friendly Tennessee trip plan in April 2026.
    """
    # Initialize evaluator (root is non-critical by design; we'll add a critical wrapper)
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

    # Extract structured plan info
    extracted: TripPlanExtraction = await evaluator.extract(
        prompt=prompt_extract_trip_plan(),
        template_class=TripPlanExtraction,
        extraction_name="trip_plan_extraction"
    )

    # Top-level critical wrapper node to reflect rubric root
    trip_main = evaluator.add_parallel(
        id="Pet-Friendly_Tennessee_Trip",
        desc="Comprehensive 3-day Tennessee trip plan in April 2026 with Breeze flight, Dollywood (Apr 3–12, 2026), pet-friendly outdoor activity, and pet-friendly lodging near Pigeon Forge, with supporting URLs.",
        parent=root,
        critical=True
    )

    # Build Supporting Evidence URLs node first so we can use as explicit prerequisites
    supports = await build_supporting_urls_nodes(evaluator, trip_main, extracted)

    # Trip length and timing
    await verify_trip_timing(evaluator, trip_main, extracted)

    # Flight arrangements (Breeze + pet policy)
    await verify_flight(evaluator, trip_main, extracted, supports)

    # Dollywood visit planning
    await verify_dollywood(evaluator, trip_main, extracted, supports)

    # Pet-friendly outdoor activity
    await verify_outdoor_activity(evaluator, trip_main, extracted, supports)

    # Hotel accommodation
    await verify_hotel(evaluator, trip_main, extracted, supports)

    # Return the evaluation summary
    return evaluator.get_summary()