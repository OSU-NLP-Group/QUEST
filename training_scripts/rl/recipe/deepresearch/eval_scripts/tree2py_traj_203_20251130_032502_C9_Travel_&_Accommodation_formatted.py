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
TASK_ID = "orlando_vacation_2025"
TASK_DESCRIPTION = (
    "A large extended family group of 14 people from Denver is planning a week-long Orlando vacation in 2025. "
    "They will be staying at a Universal Orlando Resort hotel. Based on current 2025 information, provide the following:\n\n"
    "1. Universal Orlando Hotel Selection: Identify one Universal Orlando Resort hotel property that meets ALL of the requirements.\n"
    "2. Airport Distance: What is the distance in miles from Orlando International Airport (MCO) to Universal Studios?\n"
    "3. Denver Hub Airline: Which airline serves as a primary hub carrier at Denver International Airport? (one of the two main hub airlines)\n"
    "4. Disney Destiny Cruise Information: What is the departure date and the departure port city for the Disney Destiny's maiden voyage?"
)


# Ground-truth and constraint info used for checks (not to enforce specific hotel)
EXPECTED_INFO = {
    "distance_mco_to_universal_approx_miles": 16.0,
    "distance_tolerance_miles": 3.0,  # Allow approx ±3 miles
    "den_hub_airlines": ["united airlines", "frontier airlines", "united", "frontier"],
    "disney_destiny_maiden_date": "2025-11-20",  # normalized target
    "disney_destiny_maiden_port_city": "fort lauderdale",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HotelInfo(BaseModel):
    """
    Information about the selected Universal Orlando Resort hotel.
    """
    hotel_name: Optional[str] = None
    hotel_urls: List[str] = Field(default_factory=list)


class DistanceInfo(BaseModel):
    """
    Distance information from MCO to Universal Studios.
    """
    distance_miles_text: Optional[str] = None
    distance_urls: List[str] = Field(default_factory=list)


class HubAirlineInfo(BaseModel):
    """
    DEN hub airline information.
    """
    airline_name: Optional[str] = None
    airline_urls: List[str] = Field(default_factory=list)


class DisneyDestinyInfo(BaseModel):
    """
    Disney Destiny maiden voyage info.
    """
    maiden_departure_date_text: Optional[str] = None
    maiden_departure_port_city: Optional[str] = None
    disney_urls: List[str] = Field(default_factory=list)


class VacationPlanExtraction(BaseModel):
    """
    Overall extraction structure for the vacation planning answer.
    """
    hotel: Optional[HotelInfo] = None
    airport_distance: Optional[DistanceInfo] = None
    den_hub_airline: Optional[HubAirlineInfo] = None
    disney_destiny: Optional[DisneyDestinyInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_vacation_plan() -> str:
    return """
    Extract the structured information from the answer for the Orlando vacation planning task.

    Required fields:

    hotel:
      - hotel_name: The specific Universal Orlando Resort hotel property named in the answer (string).
      - hotel_urls: All URLs cited in the answer that refer to this hotel or its official information (array of URLs).

    airport_distance:
      - distance_miles_text: The stated distance in miles from Orlando International Airport (MCO) to Universal Studios (string as written).
      - distance_urls: All URLs (if any) cited for this distance (array of URLs).

    den_hub_airline:
      - airline_name: The named primary hub airline at Denver International Airport (DEN). If multiple are mentioned, pick the first one (string).
      - airline_urls: Any URLs cited for this claim (array of URLs).

    disney_destiny:
      - maiden_departure_date_text: The stated departure date for Disney Destiny's maiden voyage (string as written).
      - maiden_departure_port_city: The stated departure port city for Disney Destiny's maiden voyage (string as written).
      - disney_urls: Any URLs cited that support Disney Destiny maiden voyage details (array of URLs).

    IMPORTANT:
    - Return null for any field not present in the answer.
    - For URL fields, only include explicit URLs present in the answer (plain or markdown link). Do not invent URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_whitespace(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return re.sub(r"\s+", " ", s).strip()


def parse_miles_value(text: Optional[str]) -> Optional[float]:
    """
    Try to extract a miles value from free-form text.
    Accept formats like "16 miles", "approx 16 mi", "15.8mi", "about 17 miles".
    """
    if not text:
        return None
    s = text.lower()
    # Extract possible float before 'mi' or 'mile'
    match = re.search(r"(\d+(\.\d+)?)\s*(mi|mile|miles)\b", s)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    # Fallback: first numeric token
    match2 = re.search(r"(\d+(\.\d+)?)", s)
    if match2:
        try:
            return float(match2.group(1))
        except Exception:
            return None
    return None


def miles_is_approximately(value: Optional[float], target: float, tolerance: float) -> bool:
    if value is None:
        return False
    return abs(value - target) <= tolerance


def normalize_date_string(s: Optional[str]) -> Optional[str]:
    """
    Normalize a date string to YYYY-MM-DD if possible.
    Handle common formats: 'November 20, 2025', 'Nov 20 2025', '11/20/2025', '2025-11-20'.
    """
    if not s:
        return None
    s = s.strip()

    # Direct ISO
    iso_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if iso_match:
        return s

    # MM/DD/YYYY
    mdY = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if mdY:
        mm = int(mdY.group(1))
        dd = int(mdY.group(2))
        yyyy = int(mdY.group(3))
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{yyyy:04d}-{mm:02d}-{dd:02d}"

    # Month name formats
    months = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }

    # e.g., "November 20, 2025" or "Nov 20 2025"
    s2 = s.replace(",", " ")
    tokens = [t for t in s2.split() if t]
    if len(tokens) >= 3:
        # Try Month Day Year
        month_token = tokens[0].lower()
        if month_token in months:
            try:
                day_token = re.sub(r"[^\d]", "", tokens[1])
                day = int(day_token)
                year = int(tokens[2])
                month = months[month_token]
                if 1 <= day <= 31:
                    return f"{year:04d}-{month:02d}-{day:02d}"
            except Exception:
                pass

        # Try Day Month Year (rare, but just in case)
        if tokens[1].lower() in months:
            try:
                day_token = re.sub(r"[^\d]", "", tokens[0])
                day = int(day_token)
                year = int(tokens[2])
                month = months[tokens[1].lower()]
                if 1 <= day <= 31:
                    return f"{year:04d}-{month:02d}-{day:02d}"
            except Exception:
                pass

    return None


def city_matches(target_city: str, provided: Optional[str]) -> bool:
    """
    Case-insensitive check that provided city refers to the target city.
    Accept variants like "Fort Lauderdale, FL" or "Port Everglades (Fort Lauderdale)".
    """
    if not provided:
        return False
    p = provided.lower().strip()
    t = target_city.lower().strip()
    if t in p:
        return True
    # Allow abbreviations or state suffixes
    simplified = re.sub(r"[^\w\s]", " ", p)
    simplified = re.sub(r"\s+", " ", simplified)
    return t in simplified


def choose_sources(*url_lists: List[str]) -> List[str]:
    """
    Combine multiple url lists, deduplicate while preserving order.
    """
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        for u in lst:
            if u and (u not in seen):
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_universal_hotel_selection(
    evaluator: Evaluator,
    parent_node,
    hotel: Optional[HotelInfo],
) -> None:
    """
    Build verification for 'Universal_Hotel_Selection' and check all required features.
    """
    sel_node = evaluator.add_parallel(
        id="Universal_Hotel_Selection",
        desc="Identifies one Universal Orlando Resort hotel that satisfies all listed constraints.",
        parent=parent_node,
        critical=True
    )

    hotel_name = _normalize_whitespace(hotel.hotel_name if hotel else None)
    hotel_sources = hotel.hotel_urls if hotel else []

    # Existence of specific property name (Critical)
    evaluator.add_custom_node(
        result=bool(hotel_name),
        id="Hotel_Property_Identified",
        desc="A specific Universal Orlando Resort hotel property name is provided.",
        parent=sel_node,
        critical=True
    )

    # Helper to add a feature verification leaf with URL support and auto-preconditions
    async def add_feature_leaf(node_id: str, description: str, claim_text: str, add_ins: str) -> None:
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=description,
            parent=sel_node,
            critical=True
        )
        await evaluator.verify(
            claim=claim_text,
            node=leaf,
            sources=hotel_sources if hotel_sources else None,
            additional_instruction=add_ins
        )

    # Features verification claims
    if hotel_name:
        # Water slide
        await add_feature_leaf(
            "Has_Water_Slide",
            "The identified hotel has a water slide feature at the pool area.",
            f"The hotel {hotel_name} has a water slide at its pool area.",
            "Look for terms like 'water slide', 'waterslide', or a slide structure in the pool area."
        )

        # Lazy river
        await add_feature_leaf(
            "Has_Lazy_River",
            "The identified hotel has a lazy river.",
            f"The hotel {hotel_name} offers a lazy river.",
            "Accept synonyms such as 'river-style pool' clearly indicating a lazy river."
        )

        # Multiple pools (two or more)
        await add_feature_leaf(
            "Has_Multiple_Pools",
            "The identified hotel has two or more pools.",
            f"The hotel {hotel_name} has at least two pools on property.",
            "Verify that the property explicitly lists multiple distinct pool areas."
        )

        # Dive tower feature at the pool
        await add_feature_leaf(
            "Has_Dive_Tower_Feature",
            "The identified hotel has a dive tower feature at the pool.",
            f"The pool area at {hotel_name} includes a 'dive tower' feature.",
            "A 'dive tower' may be described as a themed tower or structure by the pool; accept equivalent naming on official pages."
        )

        # Rooms sleep six people
        await add_feature_leaf(
            "Rooms_Sleep_Six",
            "Rooms at the identified hotel can sleep six people.",
            f"Standard accommodations (e.g., family suites) at {hotel_name} can sleep up to six people.",
            "Look for phrases like 'sleeps 6' or 'up to 6' in room descriptions."
        )

        # Rooms include a privacy partition feature
        await add_feature_leaf(
            "Rooms_Have_Privacy_Partition",
            "Rooms include a privacy partition feature.",
            f"Rooms at {hotel_name} include a privacy partition feature.",
            "Accept 'privacy partition', 'privacy divider', or equivalents in room descriptions."
        )

        # On-site bowling alley
        await add_feature_leaf(
            "Has_Onsite_Bowling_Alley",
            "The identified hotel has a bowling alley on-site.",
            f"There is an on-site bowling alley at {hotel_name}.",
            "Look for named venues like 'Galaxy Bowl' or similar on property pages."
        )

        # Poolside bar
        await add_feature_leaf(
            "Has_Poolside_Bar",
            "The identified hotel has a poolside bar.",
            f"There is a poolside bar at {hotel_name}.",
            "Accept venues clearly indicated as a bar by the pool (e.g., 'poolside bar')."
        )

        # On-site dining options
        await add_feature_leaf(
            "Has_Onsite_Dining",
            "The identified hotel has on-site dining options.",
            f"There are on-site dining options available at {hotel_name}.",
            "Confirm the presence of restaurants or food venues located at the hotel."
        )

        # Tier classification: Prime Value or Preferred
        await add_feature_leaf(
            "Tier_Is_PrimeValue_Or_Preferred",
            "The identified hotel is classified in either the Prime Value tier or Preferred tier (Universal tier system).",
            f"The hotel {hotel_name} is classified as either 'Prime Value' or 'Preferred' tier in Universal's system.",
            "Verify Universal's tier naming on the hotel's official pages or Universal's tier overview."
        )
    else:
        # If hotel name is missing, still create leaves (verify() will auto-skip due to failed prerequisite)
        for node_id, desc in [
            ("Has_Water_Slide", "The identified hotel has a water slide feature at the pool area."),
            ("Has_Lazy_River", "The identified hotel has a lazy river."),
            ("Has_Multiple_Pools", "The identified hotel has two or more pools."),
            ("Has_Dive_Tower_Feature", "The identified hotel has a dive tower feature at the pool."),
            ("Rooms_Sleep_Six", "Rooms at the identified hotel can sleep six people."),
            ("Rooms_Have_Privacy_Partition", "Rooms include a privacy partition feature."),
            ("Has_Onsite_Bowling_Alley", "The identified hotel has a bowling alley on-site."),
            ("Has_Poolside_Bar", "The identified hotel has a poolside bar."),
            ("Has_Onsite_Dining", "The identified hotel has on-site dining options."),
            ("Tier_Is_PrimeValue_Or_Preferred", "The identified hotel is classified in either the Prime Value tier or Preferred tier (Universal tier system)."),
        ]:
            leaf = evaluator.add_leaf(id=node_id, desc=desc, parent=sel_node, critical=True)
            await evaluator.verify(
                claim="Hotel not identified; prerequisites failed.",
                node=leaf,
                sources=None,
                additional_instruction="Skip due to missing hotel identification."
            )


async def verify_airport_distance_info(
    evaluator: Evaluator,
    parent_node,
    distance_info: Optional[DistanceInfo]
) -> None:
    """
    Build verification for 'Airport_Distance_Info'.
    """
    dist_node = evaluator.add_parallel(
        id="Airport_Distance_Info",
        desc="Provides the distance in miles from Orlando International Airport (MCO) to Universal Studios.",
        parent=parent_node,
        critical=True
    )

    dist_text = _normalize_whitespace(distance_info.distance_miles_text if distance_info else None)

    # Distance value provided (Critical)
    # Check presence of a numeric miles value or mention.
    miles_value = parse_miles_value(dist_text)
    stated_in_miles = bool(dist_text) and bool(re.search(r"\b(mi|mile|miles)\b", dist_text.lower())) if dist_text else False
    evaluator.add_custom_node(
        result=bool(dist_text) and (bool(miles_value) or stated_in_miles),
        id="Distance_Miles_Provided",
        desc="A distance value in miles is stated.",
        parent=dist_node,
        critical=True
    )

    # Approximately 16 miles (Critical) — numeric proximity check
    approx_ok = miles_is_approximately(miles_value, EXPECTED_INFO["distance_mco_to_universal_approx_miles"], EXPECTED_INFO["distance_tolerance_miles"])
    evaluator.add_custom_node(
        result=approx_ok,
        id="Distance_Approximately_16_Miles",
        desc="Distance is approximately 16 miles (per constraints).",
        parent=dist_node,
        critical=True
    )


async def verify_hub_airline_info(
    evaluator: Evaluator,
    parent_node,
    hub: Optional[HubAirlineInfo]
) -> None:
    """
    Build verification for 'Hub_Airline_Info'.
    """
    hub_node = evaluator.add_parallel(
        id="Hub_Airline_Info",
        desc="Identifies one primary hub airline at Denver International Airport (DEN).",
        parent=parent_node,
        critical=True
    )

    airline = _normalize_whitespace(hub.airline_name if hub else None)
    airline_lc = airline.lower() if airline else ""

    valid_names = EXPECTED_INFO["den_hub_airlines"]
    # Accept any string containing 'united' or 'frontier'
    is_valid = any(name in airline_lc for name in valid_names)

    evaluator.add_custom_node(
        result=is_valid,
        id="Names_Valid_DEN_Hub_Airline",
        desc="Names either United Airlines or Frontier Airlines as a primary hub carrier at DEN (per constraints).",
        parent=hub_node,
        critical=True
    )


async def verify_disney_destiny_information(
    evaluator: Evaluator,
    parent_node,
    dd: Optional[DisneyDestinyInfo]
) -> None:
    """
    Build verification for 'Disney_Destiny_Information'.
    """
    dd_node = evaluator.add_parallel(
        id="Disney_Destiny_Information",
        desc="Provides Disney Destiny maiden voyage departure date and departure port city.",
        parent=parent_node,
        critical=True
    )

    date_text = _normalize_whitespace(dd.maiden_departure_date_text if dd else None)
    normalized_date = normalize_date_string(date_text)
    expected_date = EXPECTED_INFO["disney_destiny_maiden_date"]
    evaluator.add_custom_node(
        result=(normalized_date == expected_date),
        id="Maiden_Voyage_Departure_Date",
        desc="Departure date is provided as November 20, 2025 (per constraints).",
        parent=dd_node,
        critical=True
    )

    port_city = _normalize_whitespace(dd.maiden_departure_port_city if dd else None)
    city_ok = city_matches(EXPECTED_INFO["disney_destiny_maiden_port_city"], port_city)
    evaluator.add_custom_node(
        result=city_ok,
        id="Maiden_Voyage_Departure_Port_City",
        desc="Departure port city is provided as Fort Lauderdale (per constraints).",
        parent=dd_node,
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
    Evaluate the answer according to the Orlando Vacation Planning rubric tree.
    """
    # Initialize evaluator (root is non-critical placeholder)
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

    # Add top-level rubric node as critical (to gate all children)
    top_node = evaluator.add_parallel(
        id="Orlando_Vacation_Planning_Complete",
        desc="Checks that all requested outputs are provided: a qualifying Universal hotel, MCO-to-Universal distance, a DEN hub airline, and Disney Destiny maiden voyage date + departure port city.",
        parent=root,
        critical=True
    )

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_vacation_plan(),
        template_class=VacationPlanExtraction,
        extraction_name="vacation_plan_extraction"
    )

    # Record ground-truth info (for reference in summary)
    evaluator.add_ground_truth({
        "distance_target_miles": EXPECTED_INFO["distance_mco_to_universal_approx_miles"],
        "distance_tolerance_miles": EXPECTED_INFO["distance_tolerance_miles"],
        "valid_den_hub_airlines": ["United Airlines", "Frontier Airlines"],
        "disney_destiny_maiden_date_expected": "November 20, 2025",
        "disney_destiny_maiden_date_expected_iso": EXPECTED_INFO["disney_destiny_maiden_date"],
        "disney_destiny_maiden_port_expected": "Fort Lauderdale"
    })

    # Build and run verification subtrees
    await verify_universal_hotel_selection(evaluator, top_node, extraction.hotel)
    await verify_airport_distance_info(evaluator, top_node, extraction.airport_distance)
    await verify_hub_airline_info(evaluator, top_node, extraction.den_hub_airline)
    await verify_disney_destiny_information(evaluator, top_node, extraction.disney_destiny)

    # Return aggregated summary
    return evaluator.get_summary()