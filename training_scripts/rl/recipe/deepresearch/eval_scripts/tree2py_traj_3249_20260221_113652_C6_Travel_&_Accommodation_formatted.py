import asyncio
import logging
import re
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mlk_weekend_cruise_pets_2026"
TASK_DESCRIPTION = (
    "You are planning a long weekend cruise vacation to celebrate Martin Luther King Jr. Day 2026 (January 19, 2026) "
    "and want to depart from the New York area. You will be flying into JFK Airport with your 50-pound dog and need to "
    "stay at a pet-friendly hotel near the airport the night before your cruise. You are specifically interested in sailing "
    "with Royal Caribbean from their Cape Liberty cruise port in New Jersey. Your requirements are: (1) Identify a Royal "
    "Caribbean cruise departing from Cape Liberty (Bayonne, New Jersey) during MLK weekend 2026 that is either 4 or 5 nights "
    "in duration; (2) Find TWO different pet-friendly hotels located near JFK Airport (within 5 miles or at the airport) "
    "that accept dogs weighing at least 50 pounds; (3) For each hotel, provide the pet fee charged per stay; "
    "(4) Calculate the cost of parking at a cruise terminal for your cruise duration (using Brooklyn or Manhattan cruise terminal "
    "rates as a comparable reference); (5) State the OMNY weekly fare cap amount for NYC public transportation; "
    "(6) Confirm whether Royal Caribbean allows pets to board their cruise ships. Provide your answer with all relevant details "
    "including hotel names, cruise details, costs, and URL references to verify each piece of information."
)

MLK_WINDOW_START = date(2026, 1, 16)
MLK_WINDOW_END = date(2026, 1, 20)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CruiseInfo(BaseModel):
    operator: Optional[str] = None  # e.g., Royal Caribbean
    ship_name: Optional[str] = None
    cruise_name: Optional[str] = None
    departure_port: Optional[str] = None  # e.g., "Cape Liberty"
    departure_city: Optional[str] = None  # e.g., "Bayonne"
    departure_state: Optional[str] = None  # e.g., "NJ" or "New Jersey"
    departure_date: Optional[str] = None  # e.g., "Jan 17, 2026" or "2026-01-17"
    nights: Optional[str] = None  # e.g., "4 nights" or "5-night"
    cruise_urls: List[str] = Field(default_factory=list)  # primary cruise link(s)
    port_urls: List[str] = Field(default_factory=list)  # port info link(s)
    duration_urls: List[str] = Field(default_factory=list)  # duration/date link(s)
    distance_from_jfk_miles: Optional[str] = None  # e.g., "27 miles"


class Hotel(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    distance_miles: Optional[str] = None  # distance to JFK stated, if any
    at_airport: Optional[bool] = None  # true if explicitly at JFK (on-airport)
    pet_fee_text: Optional[str] = None  # the fee text (e.g., "$150 per stay")
    weight_limit_text: Optional[str] = None  # weight policy text (e.g., "up to 50 lb")
    pet_policy_urls: List[str] = Field(default_factory=list)  # policy URL(s)
    location_urls: List[str] = Field(default_factory=list)  # location/details URL(s)
    primary_urls: List[str] = Field(default_factory=list)  # main hotel URL(s)


class ParkingInfo(BaseModel):
    terminal_name: Optional[str] = None  # Brooklyn or Manhattan Cruise Terminal
    rate_per_day_text: Optional[str] = None  # e.g., "$45 per day"
    nights_used_for_calc: Optional[str] = None  # e.g., "4 nights" or "5 nights"
    total_cost_text: Optional[str] = None  # e.g., "$180"
    parking_urls: List[str] = Field(default_factory=list)  # parking rate URL(s)


class OMNYInfo(BaseModel):
    weekly_cap_amount_text: Optional[str] = None  # e.g., "$35"
    ride_policy_text: Optional[str] = None  # e.g., "12 rides in 7 days"
    omny_urls: List[str] = Field(default_factory=list)  # OMNY URL(s)


class PetPolicyRC(BaseModel):
    policy_summary_text: Optional[str] = None  # e.g., "No pets; service animals only"
    policy_urls: List[str] = Field(default_factory=list)  # RC pet policy URL(s)


class TripExtraction(BaseModel):
    cruise: Optional[CruiseInfo] = None
    hotel1: Optional[Hotel] = None
    hotel2: Optional[Hotel] = None
    parking: Optional[ParkingInfo] = None
    omny: Optional[OMNYInfo] = None
    rc_pet_policy: Optional[PetPolicyRC] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip() -> str:
    return """
    Extract the trip planning details as presented in the answer. Return a single JSON object with these top-level keys:
    - cruise: details for the identified cruise (or null if not provided)
    - hotel1: details for the first pet-friendly hotel near JFK (or null if not provided)
    - hotel2: details for the second distinct pet-friendly hotel near JFK (or null if not provided)
    - parking: parking cost details (or null if not provided)
    - omny: NYC OMNY fare cap details (or null if not provided)
    - rc_pet_policy: Royal Caribbean pet policy summary and sources (or null if not provided)

    Field specifications:

    cruise:
      - operator: the cruise line/operator name (e.g., "Royal Caribbean"), as stated
      - ship_name: the ship's name if provided
      - cruise_name: the itinerary or product name if provided
      - departure_port: the port name text (e.g., "Cape Liberty")
      - departure_city: the city (e.g., "Bayonne")
      - departure_state: the state (e.g., "NJ" or "New Jersey")
      - departure_date: the stated departure date string (e.g., "Jan 17, 2026" or "2026-01-17")
      - nights: the stated duration string (e.g., "4 nights", "5-night")
      - cruise_urls: array of URLs that are the primary cruise page(s) cited in the answer
      - port_urls: array of URLs used to support the port/location information
      - duration_urls: array of URLs used to support the specific duration and dates
      - distance_from_jfk_miles: the stated distance from JFK to the cruise port (string; e.g., "27 miles")

    hotel1 and hotel2 (select the first two distinct hotels mentioned in the answer; if more are mentioned, use the first two only):
      - name: hotel name
      - address: hotel address text if given
      - distance_miles: the stated distance to JFK in miles (string), if any
      - at_airport: true if the hotel is explicitly at JFK Airport or on-airport; false otherwise; null if not clear
      - pet_fee_text: the stated pet fee amount (e.g., "$150 per stay")
      - weight_limit_text: the stated pet weight policy text (e.g., "up to 50 lb", "max 75 lbs")
      - pet_policy_urls: array of URLs that directly support the pet policy/fee/weight information
      - location_urls: array of URLs that support the hotel's location relative to JFK (can be the same as primary if relevant)
      - primary_urls: array of the hotel's primary URL(s) cited

    parking:
      - terminal_name: the terminal name used for the comparable parking rate (e.g., "Brooklyn Cruise Terminal" or "Manhattan Cruise Terminal")
      - rate_per_day_text: the stated parking rate per day (e.g., "$45 per day")
      - nights_used_for_calc: number of nights used for total parking calculation, if stated (e.g., "4 nights")
      - total_cost_text: the stated total parking cost for the cruise duration (e.g., "$180")
      - parking_urls: array of URL(s) that support the parking rate information

    omny:
      - weekly_cap_amount_text: the stated OMNY weekly fare cap amount (e.g., "$35")
      - ride_policy_text: the text that explains how the cap is achieved (e.g., "after 12 paid rides within 7 days")
      - omny_urls: array of OMNY official/source URL(s)

    rc_pet_policy:
      - policy_summary_text: the stated summary of Royal Caribbean pet policy (e.g., "No pets; only service animals allowed")
      - policy_urls: array of Royal Caribbean policy URL(s) supporting the statement

    Rules:
    - Return exactly these fields. If any field is not mentioned in the answer, set it to null (for strings/bools) or [] for arrays.
    - Do not invent or infer values; extract exactly as stated in the answer.
    - Extract only URLs explicitly present in the answer text.
    - If multiple URLs are given for the same thing, include all of them (deduplicated if exact duplicates).
    """


# --------------------------------------------------------------------------- #
# Helper utility functions                                                    #
# --------------------------------------------------------------------------- #
def dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def combine_sources(*lists: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in lists:
        combined.extend(lst or [])
    return dedup_urls(combined)


def parse_first_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Extract numbers like 50, 50.0, $50, 50 lbs, etc.
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_first_money_value(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # find dollar amount, e.g. $35.00 or 35
    m = re.search(r"\$?\s*([0-9]+(?:\.[0-9]{1,2})?)", text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_nights(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(\d+)\s*[- ]?\s*night", text.lower())
    if not m:
        # fallback: any integer
        m2 = re.search(r"(\d+)", text)
        if not m2:
            return None
        try:
            return int(m2.group(1))
        except Exception:
            return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def approx_equal(a: float, b: float, tol: float = 1.0) -> bool:
    return abs(a - b) <= tol


def try_parse_date(text: Optional[str]) -> Optional[date]:
    if not text:
        return None
    candidates = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%b %d, %Y",
        "%B %d, %Y",
        "%m/%d/%Y",
        "%m-%d-%Y",
    ]
    cleaned = text.strip()
    for fmt in candidates:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except Exception:
            continue
    # Sometimes text may include weekday or other text, try to extract Month Day, Year
    m = re.search(r"([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})", cleaned)
    if m:
        for fmt in ["%b %d, %Y", "%B %d, %Y"]:
            try:
                return datetime.strptime(m.group(1), fmt).date()
            except Exception:
                continue
    # Try ISO-like in text
    m2 = re.search(r"(\d{4})[-/](\d{2})[-/](\d{2})", cleaned)
    if m2:
        try:
            return date(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
        except Exception:
            return None
    return None


def distance_in_range_approx(miles_text: Optional[str], low: float = 22.0, high: float = 35.0) -> bool:
    val = parse_first_float(miles_text)
    if val is None:
        return False
    return low <= val <= high


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_cruise(evaluator: Evaluator, parent_node, cruise: Optional[CruiseInfo]) -> None:
    # Parent aggregate for Cruise Identification (keep non-critical to avoid strict child-critical constraint)
    cruise_node = evaluator.add_parallel(
        id="Cruise_Identification",
        desc="Identify a suitable Royal Caribbean cruise departing from Cape Liberty during MLK weekend 2026",
        parent=parent_node,
        critical=False
    )

    # Sources
    cruise_sources = combine_sources(
        cruise.cruise_urls if cruise else [],
        cruise.duration_urls if cruise else [],
        cruise.port_urls if cruise else []
    )

    # Cruise URL reference existence (critical leaf)
    evaluator.add_custom_node(
        result=bool(cruise and cruise.cruise_urls and len(cruise.cruise_urls) > 0),
        id="Cruise_URL_Reference",
        desc="Provide URL reference for the identified cruise",
        parent=cruise_node,
        critical=True
    )

    # Cruise line leaf (critical)
    cruise_line_leaf = evaluator.add_leaf(
        id="Cruise_Line",
        desc="Verify the cruise is operated by Royal Caribbean",
        parent=cruise_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page shows a cruise operated by Royal Caribbean.",
        node=cruise_line_leaf,
        sources=cruise.cruise_urls if cruise else [],
        additional_instruction="Confirm the brand/operator is Royal Caribbean; accept if the page clearly indicates Royal Caribbean as the cruise line."
    )

    # Departure Port group (non-critical aggregator here; its leaves enforce critical checks)
    dep_port_node = evaluator.add_parallel(
        id="Departure_Port",
        desc="Verify the cruise departs from Cape Liberty (Bayonne, NJ)",
        parent=cruise_node,
        critical=False
    )

    # Port Location (critical)
    port_location_leaf = evaluator.add_leaf(
        id="Port_Location",
        desc="Confirm port is located at Bayonne, New Jersey",
        parent=dep_port_node,
        critical=True
    )
    await evaluator.verify(
        claim="The cruise departs from Cape Liberty cruise port in Bayonne, New Jersey.",
        node=port_location_leaf,
        sources=cruise_sources,
        additional_instruction="Verify that the departure port is Cape Liberty in Bayonne, NJ. Accept equivalent phrasings like 'Cape Liberty Cruise Port (Bayonne, NJ)'."
    )

    # Port distance from JFK (non-critical; internal consistency check only)
    evaluator.add_custom_node(
        result=distance_in_range_approx(cruise.distance_from_jfk_miles, 25.0, 30.0),
        id="Port_Distance_from_JFK",
        desc="Verify distance from JFK Airport is approximately 25-30 miles",
        parent=dep_port_node,
        critical=False
    )

    # Port URL Reference (critical existence of a URL to verify location)
    evaluator.add_custom_node(
        result=bool(cruise and (cruise.port_urls or cruise.cruise_urls)),
        id="Port_URL_Reference",
        desc="Provide URL reference for port location verification",
        parent=dep_port_node,
        critical=True
    )

    # Cruise Duration group (non-critical aggregator; leaves enforce details)
    duration_node = evaluator.add_parallel(
        id="Cruise_Duration",
        desc="Verify cruise is 4-5 nights in duration to align with a long weekend",
        parent=cruise_node,
        critical=False
    )

    # Duration Range verification (critical)
    nights_int = parse_nights(cruise.nights if cruise else None)
    if nights_int in (4, 5):
        duration_claim = f"This cruise is {nights_int} nights long."
    else:
        duration_claim = "This cruise has a duration of either 4 or 5 nights."

    duration_leaf = evaluator.add_leaf(
        id="Duration_Range",
        desc="Confirm cruise duration is either 4 nights or 5 nights",
        parent=duration_node,
        critical=True
    )
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=(cruise.duration_urls if cruise else []) or (cruise.cruise_urls if cruise else []),
        additional_instruction="Verify whether the page shows either 4-night or 5-night duration. If the page shows 4-night or 5-night, the claim is considered correct."
    )

    # MLK weekend coverage (critical)
    dep_date_text = cruise.departure_date if cruise else None
    parsed_dep = try_parse_date(dep_date_text)
    if parsed_dep:
        mlk_claim = f"The cruise departs on {parsed_dep.isoformat()}, which falls within the Martin Luther King Jr. Day weekend window (Jan 16–20, 2026)."
    else:
        mlk_claim = "This cruise departs during the Martin Luther King Jr. Day weekend window (Jan 16–20, 2026)."

    mlk_leaf = evaluator.add_leaf(
        id="MLK_Weekend_Coverage",
        desc="Verify cruise dates align with MLK Day weekend (January 17-20, 2026 or similar)",
        parent=duration_node,
        critical=True
    )
    await evaluator.verify(
        claim=mlk_claim,
        node=mlk_leaf,
        sources=(cruise.duration_urls if cruise else []) or (cruise.cruise_urls if cruise else []),
        additional_instruction="Judge this true if the departure date shown on the page is between Jan 16 and Jan 20, 2026 (inclusive), even if the page does not explicitly mention MLK Day."
    )

    # Duration URL reference (critical existence)
    evaluator.add_custom_node(
        result=bool(cruise and (cruise.duration_urls or cruise.cruise_urls)),
        id="Duration_URL_Reference",
        desc="Provide URL reference for cruise duration and dates",
        parent=duration_node,
        critical=True
    )


async def verify_hotel(evaluator: Evaluator, parent_node, hotel: Optional[Hotel], which: int) -> None:
    # Parent aggregate for the hotel (non-critical; evaluation will average)
    hotel_node = evaluator.add_parallel(
        id=f"Hotel_{which}",
        desc=f"{'First' if which == 1 else 'Second'} pet-friendly hotel meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # Policy group
    policy_node = evaluator.add_parallel(
        id=f"H{which}_Pet_Policy",
        desc="Verify hotel accepts pets and meets weight requirements",
        parent=hotel_node,
        critical=False
    )

    # Weight limit (critical)
    weight_leaf = evaluator.add_leaf(
        id=f"H{which}_Weight_Limit",
        desc="Confirm hotel accepts dogs of at least 50 pounds",
        parent=policy_node,
        critical=True
    )
    await evaluator.verify(
        claim="A 50-pound dog is allowed under this hotel's pet policy (weight limit is at least 50 lb, and 'up to 50 lb' is acceptable).",
        node=weight_leaf,
        sources=(hotel.pet_policy_urls if hotel else []) or (hotel.primary_urls if hotel else []),
        additional_instruction="Verify from the pet policy page. If the page says 'up to 50 lb' or '50 lb max', consider a 50-pound dog acceptable."
    )

    # Pet fee (critical)
    pet_fee_text = hotel.pet_fee_text if hotel else None
    pet_fee_leaf = evaluator.add_leaf(
        id=f"H{which}_Pet_Fee",
        desc="Provide the pet fee amount charged by the hotel",
        parent=policy_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The pet fee charged by this hotel is: {pet_fee_text}.",
        node=pet_fee_leaf,
        sources=hotel.pet_policy_urls if hotel else [],
        additional_instruction="Confirm the pet fee amount and unit (per stay/per night/per pet as applicable). The claim is correct only if the page clearly supports the same amount and terms."
    )

    # Pet policy URL reference (critical existence)
    evaluator.add_custom_node(
        result=bool(hotel and hotel.pet_policy_urls and len(hotel.pet_policy_urls) > 0),
        id=f"H{which}_Pet_Policy_URL",
        desc="Provide URL reference for pet policy verification",
        parent=policy_node,
        critical=True
    )

    # Location group
    location_node = evaluator.add_parallel(
        id=f"H{which}_Location",
        desc="Verify hotel location meets proximity requirements",
        parent=hotel_node,
        critical=False
    )

    # Airport proximity (critical)
    proximity_leaf = evaluator.add_leaf(
        id=f"H{which}_Airport_Proximity",
        desc="Confirm hotel is near JFK Airport (within 5 miles or at the airport)",
        parent=location_node,
        critical=True
    )
    await evaluator.verify(
        claim="This hotel is at JFK Airport or within 5 miles of JFK.",
        node=proximity_leaf,
        sources=(hotel.location_urls if hotel else []) or (hotel.primary_urls if hotel else []),
        additional_instruction="Use the hotel's official page: if an on-airport address or clear statement indicates 'JFK Airport' or a stated distance ≤ 5 miles, accept as true. Otherwise, do not assume."
    )

    # Location URL reference (critical existence)
    evaluator.add_custom_node(
        result=bool(hotel and (hotel.location_urls or hotel.primary_urls)),
        id=f"H{which}_Location_URL",
        desc="Provide URL reference for hotel location verification",
        parent=location_node,
        critical=True
    )

    # Primary URL reference (critical existence)
    evaluator.add_custom_node(
        result=bool(hotel and hotel.primary_urls and len(hotel.primary_urls) > 0),
        id=f"H{which}_URL_Reference",
        desc=f"Provide primary URL reference for Hotel {which}",
        parent=hotel_node,
        critical=True
    )


async def verify_transportation(evaluator: Evaluator, parent_node, parking: Optional[ParkingInfo], omny: Optional[OMNYInfo], cruise: Optional[CruiseInfo]) -> None:
    transport_node = evaluator.add_parallel(
        id="Transportation_Cost_Analysis",
        desc="Calculate and compare transportation costs for traveling from JFK to Cape Liberty",
        parent=parent_node,
        critical=False
    )

    # Parking group
    parking_node = evaluator.add_parallel(
        id="Cruise_Terminal_Parking",
        desc="Determine parking cost at cruise terminal for cruise duration",
        parent=transport_node,
        critical=False
    )

    # Parking rate (critical)
    parking_rate_leaf = evaluator.add_leaf(
        id="Parking_Rate",
        desc="Provide the per-night parking rate at comparable cruise terminal (Brooklyn or Manhattan)",
        parent=parking_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The published parking rate is {parking.rate_per_day_text if parking else None} per day at the cited cruise terminal.",
        node=parking_rate_leaf,
        sources=parking.parking_urls if parking else [],
        additional_instruction="Verify the per-day parking rate from the Brooklyn or Manhattan Cruise Terminal official or authoritative page. The page should clearly show the same rate."
    )

    # Total parking cost calculation (non-critical; arithmetic consistency check)
    # Accept either nights or nights+1 days, as different terminals charge per calendar day.
    rate = parse_first_money_value(parking.rate_per_day_text if parking else None)
    total = parse_first_money_value(parking.total_cost_text if parking else None)
    nights_for_calc = parse_nights(parking.nights_used_for_calc if parking else None)
    if nights_for_calc is None:
        # fall back to cruise duration
        nights_for_calc = parse_nights(cruise.nights if cruise else None)
    valid_total = False
    if rate is not None and total is not None and nights_for_calc in (4, 5):
        cost_nights = rate * nights_for_calc
        cost_days = rate * (nights_for_calc + 1)
        if approx_equal(total, cost_nights, tol=2.0) or approx_equal(total, cost_days, tol=2.0):
            valid_total = True

    evaluator.add_custom_node(
        result=valid_total,
        id="Total_Parking_Cost",
        desc="Calculate total parking cost for a 4-5 night cruise",
        parent=parking_node,
        critical=False
    )

    # Parking URL reference (critical existence)
    evaluator.add_custom_node(
        result=bool(parking and parking.parking_urls and len(parking.parking_urls) > 0),
        id="Parking_URL_Reference",
        desc="Provide URL reference for parking rate information",
        parent=parking_node,
        critical=True
    )

    # Public Transit / OMNY group
    omny_node = evaluator.add_parallel(
        id="Public_Transit_Option",
        desc="State the OMNY weekly fare cap for NYC public transportation",
        parent=transport_node,
        critical=False
    )

    # OMNY $35 weekly cap (critical)
    omny_cap_leaf = evaluator.add_leaf(
        id="OMNY_Fare_Cap",
        desc="State the OMNY weekly fare cap amount ($35)",
        parent=omny_node,
        critical=True
    )
    await evaluator.verify(
        claim="The OMNY weekly fare cap amount is $35.",
        node=omny_cap_leaf,
        sources=omny.omny_urls if omny else [],
        additional_instruction="Confirm on an official OMNY or MTA page that the weekly fare cap is $35 for subway and local buses."
    )

    # 12-ride policy (non-critical detail)
    fare_policy_leaf = evaluator.add_leaf(
        id="Fare_Cap_Details",
        desc="Explain the 12-ride policy for achieving the weekly cap",
        parent=omny_node,
        critical=False
    )
    await evaluator.verify(
        claim="The OMNY weekly cap is reached after 12 paid rides within a 7-day period when using the same payment method.",
        node=fare_policy_leaf,
        sources=omny.omny_urls if omny else [],
        additional_instruction="Verify that the page explains 12 paid rides within 7 days triggers the weekly cap."
    )

    # OMNY URL reference (critical existence)
    evaluator.add_custom_node(
        result=bool(omny and omny.omny_urls and len(omny.omny_urls) > 0),
        id="OMNY_URL_Reference",
        desc="Provide URL reference for OMNY fare cap information",
        parent=omny_node,
        critical=True
    )


async def verify_rc_pet_policy(evaluator: Evaluator, parent_node, rc: Optional[PetPolicyRC]) -> None:
    pet_node = evaluator.add_parallel(
        id="Pet_Accommodation_Plan",
        desc="Confirm Royal Caribbean's policy regarding pets on cruises",
        parent=parent_node,
        critical=False
    )

    # No pets policy (critical)
    no_pets_leaf = evaluator.add_leaf(
        id="No_Pets_Policy",
        desc="State that Royal Caribbean does not accept pets on board (only service animals)",
        parent=pet_node,
        critical=True
    )
    await evaluator.verify(
        claim="Royal Caribbean does not accept pets onboard; only trained service animals are permitted.",
        node=no_pets_leaf,
        sources=rc.policy_urls if rc else [],
        additional_instruction="Verify on an official Royal Caribbean policy page that pets are not allowed, except for service animals."
    )

    # Pet policy URL reference (critical existence)
    evaluator.add_custom_node(
        result=bool(rc and rc.policy_urls and len(rc.policy_urls) > 0),
        id="Pet_Policy_URL_Reference",
        desc="Provide URL reference for Royal Caribbean pet policy",
        parent=pet_node,
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
    Evaluate an answer for the MLK weekend cruise + pet planning task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation at root
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

    # Extraction
    trip_data: TripExtraction = await evaluator.extract(
        prompt=prompt_extract_trip(),
        template_class=TripExtraction,
        extraction_name="trip_extraction"
    )

    # Build main planning node (non-critical aggregator; leaves inside enforce critical checks)
    plan_node = evaluator.add_parallel(
        id="Trip_Planning",
        desc="Complete planning for a long-weekend cruise vacation from New York area with a pet, departing around MLK Day 2026",
        parent=root,
        critical=False
    )

    # 1) Cruise Identification
    await verify_cruise(evaluator, plan_node, trip_data.cruise)

    # 2) Pet-Friendly Hotels (two different hotels)
    hotels_node = evaluator.add_parallel(
        id="Pet_Friendly_Hotels",
        desc="Identify TWO suitable pet-friendly hotels near JFK Airport that accommodate a 50-pound dog",
        parent=plan_node,
        critical=False
    )
    await verify_hotel(evaluator, hotels_node, trip_data.hotel1, which=1)
    await verify_hotel(evaluator, hotels_node, trip_data.hotel2, which=2)

    # 3) Transportation Cost Analysis (parking + OMNY)
    await verify_transportation(evaluator, plan_node, trip_data.parking, trip_data.omny, trip_data.cruise)

    # 4) Pet Accommodation Plan (Royal Caribbean pet policy)
    await verify_rc_pet_policy(evaluator, plan_node, trip_data.rc_pet_policy)

    return evaluator.get_summary()