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
TASK_ID = "memorial_breeze_2026"
TASK_DESCRIPTION = (
    "I am planning an outdoor recreation getaway for Memorial Day 2026 weekend and want to use Breeze Airways for "
    "budget-friendly nonstop flights. I also have an American Express Platinum Card and would like to use a Centurion Lounge "
    "at my destination airport.\n\nPlease identify a round-trip flight itinerary that meets all of the following requirements:\n\n"
    "1. Origin: The departure city must be located on the East Coast of the United States (states: Maine, New Hampshire, Vermont, Massachusetts, Rhode Island, Connecticut, New York, New Jersey, Pennsylvania, Delaware, Maryland, Virginia, North Carolina, South Carolina, Georgia, or Florida).\n\n"
    "2. Destination: The arrival city must be in the West Coast or Mountain West region of the United States and offer outdoor recreation opportunities (such as hiking, beaches, mountains, deserts, or national parks).\n\n"
    "3. Outbound Flight: Must depart on Friday, May 23, 2026 or Saturday, May 24, 2026, be operated by Breeze Airways, and be a nonstop flight.\n\n"
    "4. Return Flight: Must depart on Sunday, May 25, 2026 or Monday, May 26, 2026, be operated by Breeze Airways, and be a nonstop flight.\n\n"
    "5. Airport Amenity: The destination airport must have an American Express Centurion Lounge.\n\n"
    "For your selected itinerary, please provide:\n"
    "- Origin city name and airport code (3-letter IATA code)\n"
    "- Destination city name and airport code (3-letter IATA code)\n"
    "- A URL from Breeze Airways' website confirming that they operate nonstop flights on this route\n"
    "- A URL confirming the Centurion Lounge location at the destination airport"
)

EAST_COAST_STATES = [
    "Maine", "New Hampshire", "Vermont", "Massachusetts", "Rhode Island", "Connecticut",
    "New York", "New Jersey", "Pennsylvania", "Delaware", "Maryland", "Virginia",
    "North Carolina", "South Carolina", "Georgia", "Florida"
]

WEST_COAST_STATES = ["California", "Oregon", "Washington"]
MOUNTAIN_WEST_STATES = ["Montana", "Idaho", "Wyoming", "Utah", "Nevada", "Colorado", "Arizona", "New Mexico"]

ALLOWED_OUTBOUND_DATES = [(2026, 5, 23), (2026, 5, 24)]
ALLOWED_RETURN_DATES = [(2026, 5, 25), (2026, 5, 26)]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ItineraryExtraction(BaseModel):
    origin_city_name: Optional[str] = None
    origin_airport_code: Optional[str] = None
    destination_city_name: Optional[str] = None
    destination_airport_code: Optional[str] = None
    outbound_date: Optional[str] = None
    return_date: Optional[str] = None
    outbound_airline: Optional[str] = None
    return_airline: Optional[str] = None
    breeze_route_url: Optional[str] = None
    centurion_lounge_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_itinerary() -> str:
    return """
    Extract the round-trip flight itinerary details presented in the answer. Return the following fields:

    - origin_city_name: The origin city name (string)
    - origin_airport_code: The origin airport IATA code (3 letters). Extract exactly the 3-letter code.
    - destination_city_name: The destination city name (string)
    - destination_airport_code: The destination airport IATA code (3 letters). Extract exactly the 3-letter code.
    - outbound_date: The outbound departure date string as written in the answer (e.g., 'May 23, 2026', '2026-05-23', '5/23/2026'). Do not transform; extract verbatim.
    - return_date: The return departure date string as written in the answer.
    - outbound_airline: The operating airline for the outbound flight (string)
    - return_airline: The operating airline for the return flight (string)
    - breeze_route_url: A URL from Breeze Airways' website confirming they operate nonstop flights on this route. Only extract if explicitly provided. If multiple URLs are present, pick the most route-specific page related to this corridor. If none are provided, return null.
    - centurion_lounge_url: A URL confirming the Centurion Lounge location at the destination airport. Only extract if explicitly provided in the answer. If multiple URLs are present, pick the most specific page for the destination airport's lounge. If none are provided, return null.

    Rules:
    - Do NOT invent or infer details. Extract only what is explicitly present in the answer text.
    - For URLs, extract full valid URLs. If a URL is missing a protocol, prepend http://.
    - If any field is missing in the answer, return null for that field.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_iata(code: Optional[str]) -> str:
    return (code or "").strip().upper()


def is_valid_iata(code: Optional[str]) -> bool:
    c = _normalize_iata(code)
    return len(c) == 3 and c.isalpha()


def _contains_domain(url: Optional[str], substrings: List[str]) -> bool:
    if not url:
        return False
    url_l = url.lower()
    return any(s.lower() in url_l for s in substrings)


def _match_date_str(date_str: Optional[str], allowed_ymd: List[tuple]) -> bool:
    """
    Robustly check if a free-form date string matches any allowed (year, month, day).
    Supports formats like:
      - '2026-05-23'
      - '05/23/2026' or '5/23/2026'
      - 'May 23, 2026' (case-insensitive, with optional ordinal suffixes)
      - 'Fri, May 23 2026' etc.
    """
    if not date_str or not date_str.strip():
        return False

    s = date_str.strip()
    s_lower = s.lower()

    # Numeric ISO-like: YYYY-MM-DD
    m_iso = re.search(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b", s)
    if m_iso:
        y, mo, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        return (y, mo, d) in allowed_ymd

    # US style: M/D/YYYY
    m_us = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](20\d{2})\b", s)
    if m_us:
        mo, d, y = int(m_us.group(1)), int(m_us.group(2)), int(m_us.group(3))
        return (y, mo, d) in allowed_ymd

    # Month name formats: "May 23, 2026" / "May 23 2026" / with ordinal suffixes
    month_names = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }
    m_text = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+)(20\d{2})\b",
        s_lower
    )
    if m_text:
        mon = month_names[m_text.group(1)]
        day = int(m_text.group(2))
        year = int(m_text.group(3))
        return (year, mon, day) in allowed_ymd

    # Last fallback: only match on presence of "May", day, and 2026
    # Accept patterns like "May 23 '26" won't be recognized; keep strict year = 2026
    for (year, mon, day) in allowed_ymd:
        if year == 2026 and mon == 5:
            # Accept variations like "May 23" anywhere plus "2026"
            if re.search(r"\bmay\b", s_lower) and re.search(rf"\b{day}\b", s_lower) and re.search(r"\b2026\b", s_lower):
                return True

    return False


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_itinerary(
    evaluator: Evaluator,
    extraction: ItineraryExtraction,
    logger: logging.Logger
) -> None:
    """
    Build the verification tree under a critical main node and execute verifications.
    """
    # Critical main node (parallel aggregation). All children must be critical per framework constraint.
    main = evaluator.add_parallel(
        id="Memorial_Day_2026_Trip_Plan",
        desc="Complete round-trip flight itinerary for Memorial Day 2026 weekend from East Coast to West Coast/Mountain West destination with outdoor recreation, using Breeze Airways nonstop flights to an airport with Centurion Lounge",
        parent=evaluator.root,
        critical=True
    )

    # Existence and format checks (critical custom nodes)
    origin_city_ok = evaluator.add_custom_node(
        result=bool(extraction.origin_city_name and extraction.origin_city_name.strip()),
        id="Origin_City_Name",
        desc="Origin city name is provided",
        parent=main,
        critical=True
    )

    origin_code_ok = evaluator.add_custom_node(
        result=is_valid_iata(extraction.origin_airport_code),
        id="Origin_Airport_Code",
        desc="Origin airport code (3-letter IATA code) is provided",
        parent=main,
        critical=True
    )

    dest_city_ok = evaluator.add_custom_node(
        result=bool(extraction.destination_city_name and extraction.destination_city_name.strip()),
        id="Destination_City_Name",
        desc="Destination city name is provided",
        parent=main,
        critical=True
    )

    dest_code_ok = evaluator.add_custom_node(
        result=is_valid_iata(extraction.destination_airport_code),
        id="Destination_Airport_Code",
        desc="Destination airport code (3-letter IATA code) is provided",
        parent=main,
        critical=True
    )

    breeze_url_ok = evaluator.add_custom_node(
        result=bool(extraction.breeze_route_url and _contains_domain(extraction.breeze_route_url, ["flybreeze.com", "breezeairways.com"])),
        id="Breeze_Route_URL",
        desc="URL reference confirming Breeze Airways operates nonstop flights on this route is provided",
        parent=main,
        critical=True
    )

    lounge_url_ok = evaluator.add_custom_node(
        result=bool(extraction.centurion_lounge_url and _contains_domain(extraction.centurion_lounge_url, ["americanexpress.com", "centurionlounge.com", "thecenturionlounge.com"])),
        id="Centurion_Lounge_URL",
        desc="URL reference confirming the Centurion Lounge location at the destination airport is provided",
        parent=main,
        critical=True
    )

    # Date range checks (critical custom nodes)
    outbound_date_ok = evaluator.add_custom_node(
        result=_match_date_str(extraction.outbound_date, ALLOWED_OUTBOUND_DATES),
        id="Outbound_Date_Range",
        desc="Outbound flight departs on May 23 or May 24, 2026",
        parent=main,
        critical=True
    )

    return_date_ok = evaluator.add_custom_node(
        result=_match_date_str(extraction.return_date, ALLOWED_RETURN_DATES),
        id="Return_Date_Range",
        desc="Return flight departs on May 25 or May 26, 2026",
        parent=main,
        critical=True
    )

    # Region and recreation checks (critical leaves via simple verification)
    origin_region_node = evaluator.add_leaf(
        id="Origin_East_Coast",
        desc="Origin city is located on the East Coast of the United States (states: ME, NH, VT, MA, RI, CT, NY, NJ, PA, DE, MD, VA, NC, SC, GA, FL)",
        parent=main,
        critical=True
    )
    origin_city = (extraction.origin_city_name or "").strip()
    origin_code = _normalize_iata(extraction.origin_airport_code)
    origin_claim = (
        f"The origin city '{origin_city}' (airport code {origin_code}) is located in one of these East Coast states: "
        f"{', '.join(EAST_COAST_STATES)}."
    )
    await evaluator.verify(
        claim=origin_claim,
        node=origin_region_node,
        additional_instruction="Use general U.S. geography knowledge. Consider the city's location and associated airport. If the city/airport is within any listed state, mark as Correct."
    )

    dest_region_node = evaluator.add_leaf(
        id="Destination_West_Region",
        desc="Destination city is located in the West Coast or Mountain West region of the United States",
        parent=main,
        critical=True
    )
    dest_city = (extraction.destination_city_name or "").strip()
    dest_code = _normalize_iata(extraction.destination_airport_code)
    dest_claim = (
        f"The destination city '{dest_city}' (airport code {dest_code}) is located in either the West Coast states "
        f"({', '.join(WEST_COAST_STATES)}) or the Mountain West states ({', '.join(MOUNTAIN_WEST_STATES)})."
    )
    await evaluator.verify(
        claim=dest_claim,
        node=dest_region_node,
        additional_instruction="Use general U.S. geography knowledge. If the city/airport is within any listed state sets, mark as Correct."
    )

    dest_outdoor_node = evaluator.add_leaf(
        id="Destination_Outdoor_Recreation",
        desc="Destination city offers outdoor recreation opportunities",
        parent=main,
        critical=True
    )
    recreation_claim = (
        f"The destination '{dest_city}' offers outdoor recreation opportunities such as hiking, beaches, mountains, deserts, or access to national parks."
    )
    await evaluator.verify(
        claim=recreation_claim,
        node=dest_outdoor_node,
        additional_instruction="Consider broadly known outdoor activities and nearby natural attractions. If the city is known for or near outdoor recreation (mountains, beaches, deserts, parks), mark as Correct."
    )

    # Breeze airline and nonstop checks (critical leaves, verified via Breeze URL)
    outbound_airline_node = evaluator.add_leaf(
        id="Outbound_Airline_Breeze",
        desc="Outbound flight is operated by Breeze Airways",
        parent=main,
        critical=True
    )
    outbound_nonstop_node = evaluator.add_leaf(
        id="Outbound_Nonstop",
        desc="Outbound flight is nonstop (no connections)",
        parent=main,
        critical=True
    )
    return_airline_node = evaluator.add_leaf(
        id="Return_Airline_Breeze",
        desc="Return flight is operated by Breeze Airways",
        parent=main,
        critical=True
    )
    return_nonstop_node = evaluator.add_leaf(
        id="Return_Nonstop",
        desc="Return flight is nonstop (no connections)",
        parent=main,
        critical=True
    )

    # Destination Centurion Lounge presence (critical leaf, verified via lounge URL)
    dest_lounge_node = evaluator.add_leaf(
        id="Destination_Centurion_Lounge",
        desc="Destination airport has an American Express Centurion Lounge",
        parent=main,
        critical=True
    )

    # Prepare batch verifications (these will auto-skip if their critical URL-existence siblings failed)
    breeze_url = extraction.breeze_route_url or None
    lounge_url = extraction.centurion_lounge_url or None

    claims_and_sources: List[tuple[str, Any, Any, Optional[str]]] = [
        (
            f"Breeze Airways operates flights between {origin_code} and {dest_code}.",
            breeze_url,
            outbound_airline_node,
            "Verify that the Breeze Airways page indicates service on this corridor. Focus on the route map or listing showing Breeze service."
        ),
        (
            f"Breeze Airways offers nonstop service between {origin_code} and {dest_code}.",
            breeze_url,
            outbound_nonstop_node,
            "Verify that the Breeze page explicitly shows 'nonstop' or no connections for this route."
        ),
        (
            f"Breeze Airways operates flights between {dest_code} and {origin_code}.",
            breeze_url,
            return_airline_node,
            "Verify that the Breeze page indicates service on the reverse direction as well (typical for a nonstop route)."
        ),
        (
            f"Breeze Airways offers nonstop service between {dest_code} and {origin_code}.",
            breeze_url,
            return_nonstop_node,
            "Verify that the Breeze page explicitly shows 'nonstop' or no connections for the reverse direction."
        ),
        (
            f"The airport with code {dest_code} has an American Express Centurion Lounge (including Centurion Studio Partner locations).",
            lounge_url,
            dest_lounge_node,
            "Confirm that this official page indicates a Centurion Lounge or Centurion Studio Partner at the destination airport."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)

    # Add contextual ground truth and custom info for transparency
    evaluator.add_ground_truth({
        "allowed_outbound_dates": ["2026-05-23", "2026-05-24"],
        "allowed_return_dates": ["2026-05-25", "2026-05-26"],
        "east_coast_states": EAST_COAST_STATES,
        "west_coast_states": WEST_COAST_STATES,
        "mountain_west_states": MOUNTAIN_WEST_STATES
    }, gt_type="requirements")


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
    Evaluate an answer for the Memorial Day 2026 Breeze itinerary task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root remains non-critical; critical gating occurs in the main node
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

    # Extract itinerary details
    extraction = await evaluator.extract(
        prompt=prompt_extract_itinerary(),
        template_class=ItineraryExtraction,
        extraction_name="itinerary_extraction"
    )

    # Build tree and run verifications
    await build_and_verify_itinerary(evaluator, extraction, logger)

    # Return structured summary
    return evaluator.get_summary()