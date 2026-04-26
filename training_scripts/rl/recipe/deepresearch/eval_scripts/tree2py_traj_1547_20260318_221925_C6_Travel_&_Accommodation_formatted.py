import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "caribbean_cruise_2026_trip_package"
TASK_DESCRIPTION = """
I'm planning a Caribbean cruise vacation departing from Florida in spring 2026. I need help researching the following components for my trip:

1. Caribbean Cruise: Find one cruise that meets ALL of these requirements:
   - Departs from a Florida port (Miami, Fort Lauderdale, Tampa, or Port Canaveral)
   - Duration of at least 9 days
   - Departs between April 1, 2026 and May 31, 2026
   - Visits at least one Caribbean destination where US citizens can enter visa-free
   - Include the cruise line name, departure port, departure date, duration, and at least one Caribbean destination visited

2. Pre-Cruise Hotels: Find TWO different hotels near the cruise departure port that meet ALL of these criteria:
   - Must be from a major US hotel chain (Marriott, Hilton, IHG, Hyatt, or Choice Hotels brands)
   - Must be located within 5 miles of the cruise departure port
   - Must have at least a 3-star rating or equivalent quality classification
   - Include the hotel name, brand/chain, distance from port, and star rating for each

3. Airport Lounge Access: For the nearest major international airport to the cruise departure port:
   - Identify the airport name
   - Find one airport lounge accessible to travelers
   - Specify which terminal(s) the lounge is located in
   - Describe the access requirements (such as membership type, day pass availability, or qualifying ticket class)

For each component, please provide a reference URL where I can verify the information.
"""

ALLOWED_FLORIDA_PORTS = ["Miami", "Fort Lauderdale", "Tampa", "Port Canaveral"]
MAJOR_CHAINS = ["Marriott", "Hilton", "IHG", "Hyatt", "Choice"]

EXPECTED_AIRPORTS_BY_PORT = {
    "Miami": [("Miami International Airport", "MIA")],
    "Fort Lauderdale": [("Fort Lauderdale-Hollywood International Airport", "FLL"), ("Fort Lauderdale Intl", "FLL")],
    "Tampa": [("Tampa International Airport", "TPA")],
    "Port Canaveral": [("Orlando International Airport", "MCO")],
}


# --------------------------------------------------------------------------- #
# Utilities and normalization helpers                                         #
# --------------------------------------------------------------------------- #
def _canonical_port_name(port_text: Optional[str]) -> Optional[str]:
    if not port_text:
        return None
    s = port_text.strip().lower()
    if "miami" in s or "portmiami" in s:
        return "Miami"
    if "everglades" in s or "fort lauderdale" in s or "ft. lauderdale" in s or "ft lauderdale" in s:
        return "Fort Lauderdale"
    if "canaveral" in s or "cape canaveral" in s:
        return "Port Canaveral"
    if "tampa" in s:
        return "Tampa"
    return None


def _is_allowed_florida_port(port_text: Optional[str]) -> bool:
    canon = _canonical_port_name(port_text)
    return canon in ALLOWED_FLORIDA_PORTS if canon else False


def _matches_expected_airport(departure_port_text: Optional[str], airport_name_or_code: Optional[str]) -> bool:
    if not departure_port_text or not airport_name_or_code:
        return False
    canon_port = _canonical_port_name(departure_port_text)
    if not canon_port:
        return False
    candidates = EXPECTED_AIRPORTS_BY_PORT.get(canon_port, [])
    s = airport_name_or_code.strip().lower()
    for (full_name, code) in candidates:
        if code and code.lower() in s:
            return True
        if full_name and full_name.lower() in s:
            return True
        # Also allow partial robust matches
        if canon_port == "Fort Lauderdale" and ("fort lauderdale" in s and "airport" in s):
            return True
        if canon_port == "Port Canaveral" and ("orlando" in s and ("mco" in s or "international" in s)):
            return True
    return False


def _merge_urls(*url_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            uu = u.strip()
            if not uu:
                continue
            if uu not in seen:
                seen.add(uu)
                merged.append(uu)
    return merged


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CruiseExtraction(BaseModel):
    cruise_line: Optional[str] = None
    departure_port: Optional[str] = None
    departure_date: Optional[str] = None
    duration: Optional[str] = None  # Keep free-form (e.g., "10 days", "9 nights")
    destinations: List[str] = Field(default_factory=list)  # At least one Caribbean destination
    cruise_url: Optional[str] = None
    extra_source_urls: List[str] = Field(default_factory=list)   # Any additional URLs (itinerary, line page, etc.)
    visa_policy_urls: List[str] = Field(default_factory=list)    # Any URL(s) supporting visa-free for US citizens


class HotelItem(BaseModel):
    name: Optional[str] = None
    chain: Optional[str] = None  # Marriott, Hilton, IHG, Hyatt, Choice (or brand like "Hilton Garden Inn" → chain "Hilton")
    distance_from_port: Optional[str] = None  # e.g., "2.1 miles", free-form
    star_rating: Optional[str] = None  # e.g., "3-star", "4 stars", "Upper-midscale"
    url: Optional[str] = None
    rating_url: Optional[str] = None  # If star rating is from an OTA/aggregator
    additional_urls: List[str] = Field(default_factory=list)


class LoungeExtraction(BaseModel):
    nearest_airport_name: Optional[str] = None  # e.g., "Miami International Airport"
    nearest_airport_code: Optional[str] = None  # e.g., "MIA"
    lounge_name: Optional[str] = None
    terminal_locations: Optional[str] = None  # e.g., "Terminal D, near Gate 20"
    access_requirements: Optional[str] = None  # e.g., "Priority Pass members or day pass $65"
    lounge_url: Optional[str] = None
    airport_info_urls: List[str] = Field(default_factory=list)  # Any URL(s) supporting the airport identification


class TravelPackageExtraction(BaseModel):
    cruise: Optional[CruiseExtraction] = None
    hotels: List[HotelItem] = Field(default_factory=list)  # Expect 2 items (pad if fewer)
    lounge: Optional[LoungeExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_travel_package() -> str:
    return """
You will extract structured information from the provided answer for a Caribbean cruise trip package. Extract only what is explicitly present in the answer. Do not invent anything. If something is missing, return null or an empty list.

CRUISE (choose exactly one cruise mentioned; if multiple, pick the first that appears to meet constraints):
- cruise_line: The cruise line operating the cruise (e.g., "Carnival", "Royal Caribbean").
- departure_port: The named departure port as stated (e.g., "PortMiami", "Miami", "Port Everglades (Fort Lauderdale)", "Tampa", "Port Canaveral").
- departure_date: The stated departure date (any readable string, e.g., "May 10, 2026").
- duration: The stated duration (free-form, e.g., "10 days", "9 nights", "9-day").
- destinations: A list of at least one destination the cruise visits (e.g., ["Nassau, Bahamas", "Cozumel, Mexico"]). Include exactly as stated.
- cruise_url: A primary URL referencing this exact cruise or an official itinerary page.
- extra_source_urls: Any additional URLs cited for the cruise (e.g., cruise line homepage, itinerary info, etc.).
- visa_policy_urls: Any URLs (if provided) that support "U.S. citizens can enter [destination] visa-free" (e.g., gov/consular/official tourism sources). If none are provided, return an empty list.

HOTELS (extract up to two distinct hotels mentioned; if more than two are mentioned, include only the first two):
For each hotel, extract:
- name: The hotel's exact name.
- chain: The chain/brand (e.g., "Marriott", "Hilton", "IHG", "Hyatt", "Choice"; if a sub-brand, map to parent chain if stated, e.g., "Courtyard by Marriott" → "Marriott" or just include as given).
- distance_from_port: The distance to the cruise port if stated (free-form string like "2.3 miles").
- star_rating: The star rating or equivalent quality classification if stated (free-form, e.g., "3-star", "Upper Midscale").
- url: A primary hotel URL (brand site or OTA page) supporting the hotel's identity.
- rating_url: If star rating is supported on another page, include that URL; else null.
- additional_urls: Any other URLs cited for this hotel (e.g., Google Maps link showing distance).

LOUNGE (airport lounge near the cruise departure port):
- nearest_airport_name: The nearest major international airport's name to the cruise departure port (as stated).
- nearest_airport_code: The IATA code if stated (e.g., "MIA"); else null.
- lounge_name: The name of one accessible lounge at that airport (as stated).
- terminal_locations: The stated terminal(s) or location for the lounge.
- access_requirements: The stated access requirements (e.g., memberships, day pass availability, qualifying ticket classes).
- lounge_url: A primary URL page about that lounge (airport, lounge program, airline, or lounge operator).
- airport_info_urls: Any additional URLs (if provided) that support the identification of the nearest major international airport.

Return a JSON object with fields: cruise, hotels (list), lounge.
If some part is missing in the answer, still return the JSON with nulls/empty lists appropriately.
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _cruise_sources(cruise: Optional[CruiseExtraction]) -> List[str]:
    if not cruise:
        return []
    return _merge_urls(
        [cruise.cruise_url] if cruise.cruise_url else [],
        cruise.extra_source_urls
    )


def _cruise_visa_sources(cruise: Optional[CruiseExtraction]) -> List[str]:
    if not cruise:
        return []
    # Prefer explicit visa policy URLs; if absent, fall back to cruise sources (may fail)
    vs = cruise.visa_policy_urls or []
    if vs:
        return _merge_urls(vs)
    return _cruise_sources(cruise)


def _hotel_sources(h: Optional[HotelItem]) -> List[str]:
    if not h:
        return []
    return _merge_urls(
        [h.url] if h.url else [],
        [h.rating_url] if h.rating_url else [],
        h.additional_urls
    )


def _airport_sources(l: Optional[LoungeExtraction]) -> List[str]:
    if not l:
        return []
    # For nearest airport verification, prefer any dedicated airport info URLs, otherwise the lounge page still indicates the airport identity
    return _merge_urls(l.airport_info_urls, [l.lounge_url] if l and l.lounge_url else [])


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_cruise(evaluator: Evaluator, parent_node, cruise: Optional[CruiseExtraction]) -> None:
    cruise_parent = evaluator.add_parallel(
        id="Cruise_Selection",
        desc="Identify one Caribbean cruise meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # URL presence reference (critical presence check)
    evaluator.add_custom_node(
        result=bool(cruise and (_cruise_sources(cruise))),
        id="Cruise_URL_Reference",
        desc="Provide a valid URL reference for the identified cruise",
        parent=cruise_parent,
        critical=True
    )

    details_node = evaluator.add_parallel(
        id="Cruise_Details",
        desc="Provide complete cruise information",
        parent=cruise_parent,
        critical=True
    )

    # Departure Port - supported by URL
    dep_port_node = evaluator.add_leaf(
        id="Departure_Port_supported",
        desc="Cruise departs from the stated departure port",
        parent=details_node,
        critical=True
    )
    dep_port_text = cruise.departure_port if cruise else ""
    await evaluator.verify(
        claim=f"The cruise departs from {dep_port_text}.",
        node=dep_port_node,
        sources=_cruise_sources(cruise),
        additional_instruction="Verify the departure port shown on the referenced cruise/itinerary page. Allow synonyms like PortMiami for Miami, Port Everglades for Fort Lauderdale."
    )

    # Departure Port - allowed Florida port (logic check)
    evaluator.add_custom_node(
        result=_is_allowed_florida_port(dep_port_text),
        id="Departure_Port_allowed",
        desc="Cruise must depart from a Florida port (Miami, Fort Lauderdale, Tampa, or Port Canaveral)",
        parent=details_node,
        critical=True
    )

    # Cruise Duration at least 9 days
    duration_node = evaluator.add_leaf(
        id="Cruise_Duration",
        desc="Cruise must be at least 9 days in duration",
        parent=details_node,
        critical=True
    )
    duration_text = cruise.duration if cruise else ""
    await evaluator.verify(
        claim="The cruise duration is at least 9 days (or 9 nights or more).",
        node=duration_node,
        sources=_cruise_sources(cruise),
        additional_instruction="Check the itinerary page for duration. If shown in nights, treat '9 nights' as satisfying the requirement. Accept equivalent phrasing like '9-day' or '10 days'."
    )

    # Departure Date between April 1, 2026 and May 31, 2026
    dep_date_node = evaluator.add_leaf(
        id="Departure_Date",
        desc="Cruise must depart between April 1, 2026 and May 31, 2026",
        parent=details_node,
        critical=True
    )
    dep_date_text = cruise.departure_date if cruise else ""
    await evaluator.verify(
        claim=f"The cruise departs on {dep_date_text}, and this date falls between April 1, 2026 and May 31, 2026 inclusive.",
        node=dep_date_node,
        sources=_cruise_sources(cruise),
        additional_instruction="Extract the departure date from the page and judge whether it is within the specified window. Allow reasonable date format variations."
    )

    # Caribbean destination visited (itinerary)
    picked_destination = (cruise.destinations[0] if (cruise and cruise.destinations) else "")
    dest_visit_node = evaluator.add_leaf(
        id="Caribbean_Destination_Visited",
        desc="Cruise itinerary includes at least one Caribbean destination",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The cruise itinerary includes a stop at {picked_destination}.",
        node=dest_visit_node,
        sources=_cruise_sources(cruise),
        additional_instruction="Confirm the named destination appears in the port calls/itinerary on the referenced page. Fuzzy-match name variants if minor."
    )

    # Visa-free for US citizens to the picked destination
    visa_node = evaluator.add_leaf(
        id="Caribbean_Destination_Visa_Free",
        desc="At least one visited destination allows U.S. citizens to enter visa-free",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"U.S. citizens can enter {picked_destination} visa-free for typical short-stay tourism.",
        node=visa_node,
        sources=_cruise_visa_sources(cruise),
        additional_instruction="Verify via official or reputable sources (e.g., government/consular/tourism) that U.S. citizens do not require a visa for short tourist stays to this destination. If no relevant source is provided, the claim is not supported."
    )

    # Cruise line name validation
    line_node = evaluator.add_leaf(
        id="Cruise_Line_Name",
        desc="Provide the name of the cruise line operating this cruise",
        parent=details_node,
        critical=True
    )
    line_text = cruise.cruise_line if cruise else ""
    await evaluator.verify(
        claim=f"The cruise is operated by {line_text}.",
        node=line_node,
        sources=_cruise_sources(cruise),
        additional_instruction="Confirm the cruise line/operator name appears on the referenced cruise/itinerary page."
    )


async def verify_hotel(
    evaluator: Evaluator,
    parent_node,
    hotel: Optional[HotelItem],
    hotel_index: int,
    departure_port_text: Optional[str]
) -> None:
    hotel_num = hotel_index + 1
    hotel_parent = evaluator.add_parallel(
        id=f"Hotel_{hotel_num}",
        desc=f"{'First' if hotel_num == 1 else 'Second'} hotel option meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # Hotel URL presence reference (critical presence check)
    evaluator.add_custom_node(
        result=bool(hotel and (_hotel_sources(hotel))),
        id=f"Hotel_{hotel_num}_URL_Reference",
        desc=f"Provide a valid URL reference for the {'first' if hotel_num == 1 else 'second'} hotel",
        parent=hotel_parent,
        critical=True
    )

    details_node = evaluator.add_parallel(
        id=f"Hotel_{hotel_num}_Details",
        desc=f"Complete information for {'first' if hotel_num == 1 else 'second'} hotel",
        parent=hotel_parent,
        critical=True
    )

    # Chain allowed (logic)
    chain_allowed = False
    chain_text = hotel.chain if hotel else None
    if chain_text:
        # Accept if the chain string contains any major chain keyword (e.g., "Courtyard by Marriott" counts for "Marriott")
        low = chain_text.lower()
        chain_allowed = any(c.lower() in low for c in MAJOR_CHAINS)
    evaluator.add_custom_node(
        result=chain_allowed,
        id=f"Hotel_{hotel_num}_Chain_Allowed",
        desc="Hotel must be from a major US hotel chain (Marriott, Hilton, IHG, Hyatt, or Choice Hotels brands)",
        parent=details_node,
        critical=True
    )

    # Chain verification on page
    chain_node = evaluator.add_leaf(
        id=f"Hotel_{hotel_num}_Chain_Verified",
        desc="Hotel brand/chain is accurately cited",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This hotel is part of the {chain_text} brand/chain.",
        node=chain_node,
        sources=_hotel_sources(hotel),
        additional_instruction="Confirm on the page that the hotel belongs to the named chain/brand. Brand variants (e.g., Hilton Garden Inn → Hilton) are acceptable."
    )

    # Distance within 5 miles
    distance_node = evaluator.add_leaf(
        id=f"Hotel_{hotel_num}_Distance",
        desc="Hotel must be within 5 miles of the cruise departure port",
        parent=details_node,
        critical=True
    )
    dep_port_canon = _canonical_port_name(departure_port_text)
    dep_port_display = departure_port_text or "the cruise port"
    await evaluator.verify(
        claim=f"This hotel is within 5 miles of the cruise departure port ({dep_port_display}).",
        node=distance_node,
        sources=_hotel_sources(hotel),
        additional_instruction="Look for explicit distance to the port or a map/directions link. Accept if distance shown is ≤ 5 miles. Treat synonyms: PortMiami=Miami cruise port; Port Everglades=Fort Lauderdale cruise port; Port Canaveral=Cape Canaveral."
    )

    # Star rating at least 3-star or equivalent
    rating_node = evaluator.add_leaf(
        id=f"Hotel_{hotel_num}_Rating",
        desc="Hotel must have at least a 3-star rating or equivalent quality classification",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim="This hotel has at least a 3-star rating or an equivalent quality classification (e.g., midscale or above).",
        node=rating_node,
        sources=_hotel_sources(hotel),
        additional_instruction="Check the hotel/OTA page for '3-star' or higher. If exact stars are absent, accept equivalent category suggesting >=3-star quality."
    )

    # Hotel name verified
    name_node = evaluator.add_leaf(
        id=f"Hotel_{hotel_num}_Name",
        desc="Provide the specific hotel name",
        parent=details_node,
        critical=True
    )
    name_text = hotel.name if hotel else ""
    await evaluator.verify(
        claim=f"The hotel's official name is '{name_text}'.",
        node=name_node,
        sources=_hotel_sources(hotel),
        additional_instruction="Verify the displayed property name on the referenced page."
    )


async def verify_lounge(
    evaluator: Evaluator,
    parent_node,
    lounge: Optional[LoungeExtraction],
    departure_port_text: Optional[str]
) -> None:
    lounge_parent = evaluator.add_parallel(
        id="Airport_Lounge_Access",
        desc="Identify airport lounge access at the nearest international airport to the cruise port",
        parent=parent_node,
        critical=False
    )

    # Lounge URL reference presence (critical)
    evaluator.add_custom_node(
        result=bool(lounge and lounge.lounge_url),
        id="Lounge_URL_Reference",
        desc="Provide a valid URL reference for the airport lounge information",
        parent=lounge_parent,
        critical=True
    )

    details_node = evaluator.add_parallel(
        id="Airport_Lounge_Details",
        desc="Complete information about airport lounge",
        parent=lounge_parent,
        critical=True
    )

    # Nearest airport logic check vs expected mapping
    airport_name_or_code = lounge.nearest_airport_code or lounge.nearest_airport_name if lounge else None
    evaluator.add_custom_node(
        result=_matches_expected_airport(departure_port_text, airport_name_or_code),
        id="Nearest_Airport_Matches_Expected",
        desc="Nearest major international airport matches the expected airport for the departure port",
        parent=details_node,
        critical=True
    )

    # Nearest airport supported by URLs (when available)
    airport_node = evaluator.add_leaf(
        id="Nearest_Airport",
        desc="Identify the nearest major international airport to the cruise departure port",
        parent=details_node,
        critical=True
    )
    dep_port_display = departure_port_text or "the cruise port"
    nearest_airport_display = (lounge.nearest_airport_name or lounge.nearest_airport_code or "") if lounge else ""
    await evaluator.verify(
        claim=f"The nearest major international airport to {dep_port_display} is {nearest_airport_display}.",
        node=airport_node,
        sources=_airport_sources(lounge),
        additional_instruction="Use the provided airport or city information page(s). Accept if the page clearly indicates this is the primary international airport serving the departure port's metro area."
    )

    # Lounge name exists at the airport
    lounge_name_node = evaluator.add_leaf(
        id="Lounge_Name",
        desc="Provide the name of at least one airport lounge accessible to travelers at this airport",
        parent=details_node,
        critical=True
    )
    lounge_name_text = lounge.lounge_name if lounge else ""
    airport_name_text = lounge.nearest_airport_name if lounge else ""
    await evaluator.verify(
        claim=f"There is an airport lounge named '{lounge_name_text}' at {airport_name_text or 'the airport'}.",
        node=lounge_name_node,
        sources=[lounge.lounge_url] if lounge and lounge.lounge_url else [],
        additional_instruction="Verify that the lounge name appears on the referenced lounge or airport page."
    )

    # Terminal location
    terminal_node = evaluator.add_leaf(
        id="Terminal_Location",
        desc="Specify which terminal(s) the lounge is located in",
        parent=details_node,
        critical=True
    )
    term_text = lounge.terminal_locations if lounge else ""
    await evaluator.verify(
        claim=f"The lounge is located in terminal(s): {term_text}.",
        node=terminal_node,
        sources=[lounge.lounge_url] if lounge and lounge.lounge_url else [],
        additional_instruction="Confirm the terminal(s)/location displayed on the lounge's official or airport page."
    )

    # Access requirements
    access_node = evaluator.add_leaf(
        id="Access_Requirements",
        desc="Specify the access requirements (membership type, day pass availability, or qualifying ticket class)",
        parent=details_node,
        critical=True
    )
    access_text = lounge.access_requirements if lounge else ""
    await evaluator.verify(
        claim=f"Access requirements: {access_text}.",
        node=access_node,
        sources=[lounge.lounge_url] if lounge and lounge.lounge_url else [],
        additional_instruction="Confirm the access rules (e.g., Priority Pass, airline status, premium cabin, or day pass availability) on the referenced page."
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
    # Initialize evaluator (root as non-critical parallel to allow partial credit across components)
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

    # Add helpful context as custom info
    evaluator.add_custom_info(
        {
            "allowed_florida_ports": ALLOWED_FLORIDA_PORTS,
            "major_hotel_chains": MAJOR_CHAINS,
            "expected_airports_by_port": EXPECTED_AIRPORTS_BY_PORT,
        },
        info_type="constraints",
        info_name="evaluation_constraints"
    )

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_travel_package(),
        template_class=TravelPackageExtraction,
        extraction_name="travel_package_extraction"
    )

    # Build the top-level research node (set to non-critical to comply with framework; inner nodes enforce critical checks)
    research_node = evaluator.add_parallel(
        id="Travel_Package_Research",
        desc="Complete research for a Caribbean cruise vacation package including cruise, pre-cruise hotels, and airport lounge access",
        parent=root,
        critical=False
    )

    # Cruise verification
    await verify_cruise(evaluator, research_node, extraction.cruise)

    # Hotels verification (ensure exactly two are evaluated; pad with empty if needed)
    hotels = list(extraction.hotels) if extraction.hotels else []
    while len(hotels) < 2:
        hotels.append(HotelItem())
    hotel_accom_node = evaluator.add_parallel(
        id="Hotel_Accommodations",
        desc="Identify two hotels near the cruise departure port meeting all criteria",
        parent=research_node,
        critical=False
    )
    departure_port_text = extraction.cruise.departure_port if extraction.cruise else None
    await verify_hotel(evaluator, hotel_accom_node, hotels[0], 0, departure_port_text)
    await verify_hotel(evaluator, hotel_accom_node, hotels[1], 1, departure_port_text)

    # Lounge verification
    await verify_lounge(evaluator, research_node, extraction.lounge, departure_port_text)

    # Final summary
    return evaluator.get_summary()