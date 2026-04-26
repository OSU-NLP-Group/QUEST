import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tampa_cruise_hotels_mar_2026"
TASK_DESCRIPTION = """
I am planning a 7-day cruise departing from Port Tampa Bay in March 2026 and need to arrive the night before. Please identify 4 different hotels that meet all of the following requirements:

1. Located within 5 miles of Port Tampa Bay cruise terminal
2. Offer free shuttle service to the cruise port that operates on cruise departure days
3. Provide cruise parking packages that allow parking for the full 7-day cruise duration
4. Have a check-in time of 3:00 PM or earlier
5. Have an official website where reservations can be made

For each hotel, provide:
- Hotel name
- Complete street address
- Distance from the hotel to Port Tampa Bay cruise terminal (in miles)
- Official website URL
- Phone number for reservations or shuttle arrangements
- Confirmation that free shuttle service to cruise port is available
- Confirmation that shuttle operates on cruise departure days
- Confirmation that cruise parking is available for 7+ days
- Daily parking rate (in USD)
- Approximate nightly room rate (in USD)
- Check-in time
- Total cost for 1 night stay plus 7 days of parking (calculated as: room rate + (parking rate × 7 days))
"""

# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class HotelItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    distance_to_port_miles: Optional[str] = None
    website_url: Optional[str] = None
    phone: Optional[str] = None

    free_shuttle_to_port: Optional[str] = None
    shuttle_operates_on_departure_days: Optional[str] = None
    cruise_parking_available_7_plus_days: Optional[str] = None

    parking_rate_usd_per_day: Optional[str] = None
    nightly_room_rate_usd: Optional[str] = None
    checkin_time: Optional[str] = None
    total_cost_room_plus_7days_parking_usd: Optional[str] = None

    # Additional official evidence links (ideally specific official pages such as "Park & Cruise", "Transportation", or FAQs)
    evidence_urls: List[str] = Field(default_factory=list)


class HotelsExtraction(BaseModel):
    hotels: List[HotelItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels() -> str:
    return """
    Extract structured information about up to 6 hotels mentioned in the answer that are related to Port Tampa Bay cruises.

    For each hotel, extract the following fields exactly as presented in the answer:
    - name: Hotel name
    - address: Complete street address
    - distance_to_port_miles: Distance in miles from the hotel to the Port Tampa Bay cruise terminal, as a string (e.g., "2.5", "approx. 3 miles")
    - website_url: Official hotel website URL where reservations can be made (not third-party OTAs)
    - phone: A reservations or shuttle/transportation contact phone number

    Confirmations (store the answer's stated confirmation text, such as "yes", "complimentary shuttle", "no", or a short phrase):
    - free_shuttle_to_port: Confirmation text that a free/complimentary shuttle to the Port Tampa Bay cruise terminal is available
    - shuttle_operates_on_departure_days: Confirmation text that this shuttle operates on cruise departure days
    - cruise_parking_available_7_plus_days: Confirmation text that cruise parking packages allow parking for at least 7 days

    Pricing and timing (as strings exactly as stated, do not calculate or normalize):
    - parking_rate_usd_per_day: Daily parking rate in USD (e.g., "$10", "10 USD/day")
    - nightly_room_rate_usd: Approximate nightly room rate in USD (e.g., "$149", "about 160")
    - checkin_time: Check-in time string (e.g., "3:00 PM", "2 PM", "15:00")
    - total_cost_room_plus_7days_parking_usd: Total cost for 1 night + 7 days of parking (room rate + 7 × daily parking rate), as given by the answer

    - evidence_urls: A list of 1–5 official hotel website URLs relevant for verifying cruise shuttle/parking info (examples: "Park & Cruise", "Transportation", "Shuttle", "Cruise Parking", or relevant FAQ pages). Only include official hotel or brand URLs. Do not include maps or third-party sites.

    Return a JSON object with a single key "hotels" that is an array of hotel objects with these fields.
    If a field is missing in the answer for a hotel, set it to null (or empty array for evidence_urls).
    """


# --------------------------------------------------------------------------- #
# Helper parsing utilities                                                    #
# --------------------------------------------------------------------------- #
def _extract_first_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Replace commas in numbers like 1,234.56
    norm = text.replace(",", "")
    m = re.search(r"(-?\d+(?:\.\d+)?)", norm)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_distance_miles(text: Optional[str]) -> Optional[float]:
    return _extract_first_number(text)


def parse_usd_amount(text: Optional[str]) -> Optional[float]:
    # Accept formats like "$10", "10 USD", "USD 12.50", "about 160", "150-170" -> take the first number
    return _extract_first_number(text)


def parse_time_to_minutes(text: Optional[str]) -> Optional[int]:
    if not text:
        return None

    t = text.strip().lower()
    # Handle special words
    if "noon" in t:
        return 12 * 60
    if "midnight" in t:
        return 0

    # 24-hour format: HH:MM or HH.MM
    m24 = re.search(r"\b([01]?\d|2[0-3])[:\.]([0-5]\d)\b", t)
    if m24:
        hh = int(m24.group(1))
        mm = int(m24.group(2))
        return hh * 60 + mm

    m24_h = re.search(r"\b([01]?\d|2[0-3])\b", t)
    ampm = re.search(r"\b(am|pm)\b", t)
    # 12-hour with minutes: h:mm am/pm
    m12 = re.search(r"\b(1[0-2]|0?\d)[:\.]([0-5]\d)\s*(am|pm)\b", t)
    if m12:
        hh = int(m12.group(1))
        mm = int(m12.group(2))
        ap = m12.group(3)
        if ap == "pm" and hh != 12:
            hh += 12
        if ap == "am" and hh == 12:
            hh = 0
        return hh * 60 + mm

    # 12-hour without minutes: h am/pm
    m12_h = re.search(r"\b(1[0-2]|0?\d)\s*(am|pm)\b", t)
    if m12_h:
        hh = int(m12_h.group(1))
        ap = m12_h.group(2)
        if ap == "pm" and hh != 12:
            hh += 12
        if ap == "am" and hh == 12:
            hh = 0
        return hh * 60

    # If only hour present with no am/pm but common forms like "3 pm" handled above; If bare "15", assume 24h
    if m24_h and not ampm:
        hh = int(m24_h.group(1))
        return hh * 60

    return None


def looks_like_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip()
    return u.startswith("http://") or u.startswith("https://")


def has_phone_digits(phone: Optional[str]) -> bool:
    if not phone:
        return False
    digits = re.sub(r"\D", "", phone)
    return len(digits) >= 7


def build_sources_list(hotel: HotelItem) -> List[str]:
    urls = []
    if hotel.website_url and looks_like_url(hotel.website_url):
        urls.append(hotel.website_url.strip())
    for u in hotel.evidence_urls or []:
        if looks_like_url(u):
            urls.append(u.strip())
    # Deduplicate preserving order
    seen = set()
    dedup = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    return dedup


# --------------------------------------------------------------------------- #
# Verification for a single hotel                                             #
# --------------------------------------------------------------------------- #
async def verify_hotel(evaluator: Evaluator, parent_node, hotel: HotelItem, idx: int) -> None:
    """
    Build the subtree for one hotel and run necessary verifications.
    All leaf nodes here are critical per rubric.
    """
    hotel_node = evaluator.add_parallel(
        id=f"hotel_{idx + 1}",
        desc=f"{['First','Second','Third','Fourth','Fifth','Sixth'][idx] if idx < 6 else f'Hotel #{idx+1}'} hotel meeting all requirements",
        parent=parent_node,
        critical=False,
    )

    # 1) Name provided
    evaluator.add_custom_node(
        result=bool(hotel.name and hotel.name.strip()),
        id=f"hotel_{idx + 1}_name",
        desc="Hotel name is provided",
        parent=hotel_node,
        critical=True
    )

    # 2) Address provided
    evaluator.add_custom_node(
        result=bool(hotel.address and hotel.address.strip()),
        id=f"hotel_{idx + 1}_address",
        desc="Complete street address is provided",
        parent=hotel_node,
        critical=True
    )

    # 3) Distance provided and within 5 miles
    distance_val = parse_distance_miles(hotel.distance_to_port_miles)
    evaluator.add_custom_node(
        result=(distance_val is not None and distance_val <= 5.0),
        id=f"hotel_{idx + 1}_distance",
        desc="Distance to Port Tampa Bay cruise terminal is stated (must be within 5 miles)",
        parent=hotel_node,
        critical=True
    )

    # 4) Official website URL provided
    evaluator.add_custom_node(
        result=looks_like_url(hotel.website_url),
        id=f"hotel_{idx + 1}_website",
        desc="Official website URL is provided",
        parent=hotel_node,
        critical=True
    )

    # 5) Phone number provided
    evaluator.add_custom_node(
        result=has_phone_digits(hotel.phone),
        id=f"hotel_{idx + 1}_phone",
        desc="Phone number is provided",
        parent=hotel_node,
        critical=True
    )

    # Build sources for URL-grounded checks
    sources = build_sources_list(hotel)

    # 6) Free shuttle to cruise port is confirmed (URL-verified)
    leaf_shuttle_free = evaluator.add_leaf(
        id=f"hotel_{idx + 1}_free_shuttle",
        desc="Free shuttle service to cruise port is confirmed",
        parent=hotel_node,
        critical=True
    )
    claim_shuttle_free = (
        f"The hotel's official website confirms that the hotel offers a free or complimentary shuttle "
        f"service to the Port Tampa Bay cruise terminal for cruise guests."
    )

    # 7) Shuttle operates on cruise departure days (URL-verified)
    leaf_shuttle_sched = evaluator.add_leaf(
        id=f"hotel_{idx + 1}_shuttle_schedule",
        desc="Shuttle operates on cruise departure days is confirmed",
        parent=hotel_node,
        critical=True
    )
    claim_shuttle_sched = (
        "The hotel's official website confirms that the shuttle operates on cruise departure days "
        "(i.e., it runs when cruises are departing from Port Tampa Bay)."
    )

    # 8) Cruise parking for 7+ days is confirmed available (URL-verified)
    leaf_parking = evaluator.add_leaf(
        id=f"hotel_{idx + 1}_parking_available",
        desc="Cruise parking for 7+ days is confirmed as available",
        parent=hotel_node,
        critical=True
    )
    claim_parking = (
        "The hotel's official website confirms that it offers cruise parking packages that allow guests "
        "to leave their car parked for at least 7 days (covering a 7-day cruise)."
    )

    # Batch verify the URL-grounded leaves
    await evaluator.batch_verify(
        [
            (
                claim_shuttle_free,
                sources,
                leaf_shuttle_free,
                "Rely only on official hotel or brand webpages. Accept synonyms like 'complimentary', 'included', or 'no charge'. "
                "If the shuttle is paid-only or airport-only, mark as not supported."
            ),
            (
                claim_shuttle_sched,
                sources,
                leaf_shuttle_sched,
                "Look for schedule notes that explicitly or implicitly indicate service on cruise departure days "
                "(e.g., 'on cruise days', 'on embarkation mornings', or similar). If unclear or only airport service is mentioned, mark as not supported."
            ),
            (
                claim_parking,
                sources,
                leaf_parking,
                "Look for 'Park & Cruise', 'Cruise Parking', or package details that indicate parking duration for the length of the cruise. "
                "If wording clearly implies 'duration of cruise' or '7+ days', accept. If day count is insufficient or unclear, mark as not supported."
            ),
        ]
    )

    # 9) Daily parking rate is provided
    parking_rate_val = parse_usd_amount(hotel.parking_rate_usd_per_day)
    evaluator.add_custom_node(
        result=(parking_rate_val is not None),
        id=f"hotel_{idx + 1}_parking_rate",
        desc="Daily parking rate in USD is provided",
        parent=hotel_node,
        critical=True
    )

    # 10) Nightly room rate is provided
    room_rate_val = parse_usd_amount(hotel.nightly_room_rate_usd)
    evaluator.add_custom_node(
        result=(room_rate_val is not None),
        id=f"hotel_{idx + 1}_room_rate",
        desc="Approximate nightly room rate in USD is provided",
        parent=hotel_node,
        critical=True
    )

    # 11) Check-in time provided and is 3:00 PM or earlier
    checkin_minutes = parse_time_to_minutes(hotel.checkin_time)
    evaluator.add_custom_node(
        result=(checkin_minutes is not None and checkin_minutes <= 15 * 60),
        id=f"hotel_{idx + 1}_checkin",
        desc="Check-in time is provided and is 3:00 PM or earlier",
        parent=hotel_node,
        critical=True
    )

    # 12) Total cost correctly calculated: total = room + 7 * parking
    total_cost_val = parse_usd_amount(hotel.total_cost_room_plus_7days_parking_usd)
    expected_total = None
    if room_rate_val is not None and parking_rate_val is not None:
        expected_total = room_rate_val + 7 * parking_rate_val
    tolerance = 1.0  # Allow small rounding tolerance
    total_ok = (
        total_cost_val is not None and
        expected_total is not None and
        abs(total_cost_val - expected_total) <= tolerance
    )
    evaluator.add_custom_node(
        result=total_ok,
        id=f"hotel_{idx + 1}_total_cost",
        desc="Total cost is correctly calculated as: room rate + (parking rate × 7)",
        parent=hotel_node,
        critical=True
    )

    # Record some parsed info for debugging/traceability
    evaluator.add_custom_info(
        info={
            "hotel_index": idx + 1,
            "name": hotel.name,
            "parsed": {
                "distance_miles": distance_val,
                "parking_rate_usd_per_day": parking_rate_val,
                "nightly_room_rate_usd": room_rate_val,
                "checkin_minutes_after_midnight": checkin_minutes,
                "total_cost_reported": total_cost_val,
                "total_cost_expected": expected_total,
                "sources_used": sources,
            }
        },
        info_type="hotel_parsed_values",
        info_name=f"hotel_{idx + 1}_parsed_values"
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Tampa Port Cruise Hotels task.
    """
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

    # Extract hotels from the answer
    extracted: HotelsExtraction = await evaluator.extract(
        prompt=prompt_extract_hotels(),
        template_class=HotelsExtraction,
        extraction_name="hotels_extraction",
    )

    # Normalize to exactly 4 hotels (pad with empty items if fewer)
    hotels: List[HotelItem] = list(extracted.hotels[:4])
    while len(hotels) < 4:
        hotels.append(HotelItem())

    # Build verification subtrees for each of the 4 hotels
    for i in range(4):
        await verify_hotel(evaluator, root, hotels[i], i)

    # Return summary with verification tree and scores
    return evaluator.get_summary()