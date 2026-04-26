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
TASK_ID = "caribbean_destinations_infrastructure"
TASK_DESCRIPTION = """
A luxury travel company is planning a Caribbean destination program that must accommodate: private business jet operations (specifically Bombardier Challenger 350 aircraft requiring minimum 4,835 feet runway for takeoff), cruise ship arrivals from major international cruise lines, accommodation through major international hotel loyalty programs (Marriott Bonvoy, Hilton Honors, World of Hyatt, or IHG One Rewards), and commercial airline connectivity via at least one major global airline alliance (Star Alliance, SkyTeam, or Oneworld). Identify at least two Caribbean destinations (island/country and primary city/port) that satisfy ALL of the following requirements: (1) Airport Infrastructure - The destination must have an international airport with runway length of at least 4,835 feet to accommodate Bombardier Challenger 350 operations, international designation with customs facilities for business aviation; provide the airport's official name and IATA code. (2) Cruise Port Facilities - The destination must have an operational cruise terminal and service by at least one major cruise line (Viking, Royal Caribbean, Celebrity, Carnival, Norwegian, Princess, or Holland America); provide the specific cruise terminal location or name. (3) Hotel Accommodation - The destination must have at least one property from Marriott Bonvoy, Hilton Honors, World of Hyatt, or IHG One Rewards; provide the specific hotel property name and brand. (4) Airline Connectivity - The destination's airport must be served by at least one airline from Star Alliance, SkyTeam, or Oneworld; provide at least one specific airline name. For each destination, provide supporting reference URLs for: airport runway specifications, cruise port information, hotel property information, and airline service information.
"""

RUNWAY_MIN_FT = 4835

MAJOR_CRUISE_LINES = [
    "Viking", "Royal Caribbean", "Celebrity", "Carnival",
    "Norwegian", "Princess", "Holland America"
]

# Alliance membership pages (used to verify alliance membership when alliance_name provided)
ALLIANCE_MEMBERSHIP_PAGES = {
    "Star Alliance": "https://www.staralliance.com/en/member-airlines",
    "SkyTeam": "https://www.skyteam.com/en/members",
    "Oneworld": "https://www.oneworld.com/members",
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DestinationItem(BaseModel):
    # General destination identification
    country_or_island: Optional[str] = None
    primary_city_or_port: Optional[str] = None

    # Airport information
    airport_name: Optional[str] = None
    iata_code: Optional[str] = None
    icao_code: Optional[str] = None
    runway_length_ft: Optional[str] = None
    international_designation: Optional[str] = None  # e.g., "international" or "customs available"
    runway_specifications_url: Optional[str] = None

    # Cruise port info
    cruise_terminal_name_or_location: Optional[str] = None
    cruise_line: Optional[str] = None  # one of MAJOR_CRUISE_LINES
    cruise_port_info_url: Optional[str] = None

    # Hotel info
    hotel_property_name: Optional[str] = None
    hotel_brand: Optional[str] = None  # brand within Marriott/Hilton/Hyatt/IHG
    hotel_info_url: Optional[str] = None

    # Airline connectivity
    airline_name: Optional[str] = None
    alliance_name: Optional[str] = None  # Star Alliance / SkyTeam / Oneworld
    airline_connectivity_url: Optional[str] = None
    alliance_membership_url: Optional[str] = None  # optional if provided in answer


class DestinationsExtraction(BaseModel):
    destinations: List[DestinationItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_destinations() -> str:
    return """
    Extract up to TWO Caribbean destinations mentioned in the answer that attempt to satisfy all specified infrastructure requirements.
    For each destination, extract the following fields (return null for any missing field):

    General Destination Identification:
    - country_or_island: The island or country name
    - primary_city_or_port: The primary city or port name

    Airport Information:
    - airport_name: The official airport name
    - iata_code: The airport IATA code (3-letter)
    - icao_code: The airport ICAO code (4-letter, if available)
    - runway_length_ft: The runway length value mentioned (include units or keep as string if a range/approximation)
    - international_designation: A phrase indicating international status and customs availability (e.g., "international with customs")
    - runway_specifications_url: A URL reference that contains runway specifications (prefer official or reputable sources)

    Cruise Port Information:
    - cruise_terminal_name_or_location: The specific cruise terminal name or location/address
    - cruise_line: The name of at least one major cruise line (choose one from: Viking, Royal Caribbean, Celebrity, Carnival, Norwegian, Princess, Holland America) reported to serve this port
    - cruise_port_info_url: A URL reference that documents the cruise port facilities or schedules

    Hotel Accommodation:
    - hotel_property_name: The specific hotel property name
    - hotel_brand: The hotel brand (e.g., Marriott, Hilton, Hyatt, IHG sub-brands)
    - hotel_info_url: A URL reference that documents the hotel property (prefer official brand/property site)

    Airline Connectivity:
    - airline_name: The specific airline name that serves the destination airport
    - alliance_name: The airline alliance name (Star Alliance, SkyTeam, or Oneworld) that the airline belongs to
    - airline_connectivity_url: A URL reference showing that the airline serves the destination airport (e.g., airport route map, airline route page, timetable)
    - alliance_membership_url: (Optional) A URL reference that shows alliance membership (e.g., alliance official member list page). If not provided in the answer, set to null.

    RULES:
    - Extract ONLY what is explicitly present in the answer text and its provided sources; do not invent or infer missing items.
    - If the answer lists more than two destinations, extract the first two. If fewer than two, extract whatever is available and set missing fields to null.
    - Keep values as strings; do not convert units to numbers.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def alliance_members_page_url(alliance_name: Optional[str]) -> Optional[str]:
    if not alliance_name:
        return None
    name = alliance_name.strip()
    return ALLIANCE_MEMBERSHIP_PAGES.get(name)


def non_empty_str(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_destination(
    evaluator: Evaluator,
    parent_node,
    dest: DestinationItem,
    idx: int,
    dest_critical: bool,
) -> None:
    """
    Build verification sub-tree and run checks for a single destination.

    Args:
        evaluator: Evaluator instance
        parent_node: Root or grouping node to attach this destination under
        dest: Extracted destination information
        idx: Destination index (0-based)
        dest_critical: Whether this destination is critical to overall success
    """
    # Destination main node
    dest_label = "First" if idx == 0 else "Second"
    dest_node = evaluator.add_parallel(
        id=f"destination_{idx+1}",
        desc=f"{dest_label} destination meeting all requirements",
        parent=parent_node,
        critical=dest_critical
    )

    # 1) Destination Name (existence check)
    evaluator.add_custom_node(
        result=(non_empty_str(dest.country_or_island) and non_empty_str(dest.primary_city_or_port)),
        id=f"d{idx+1}_destination_name",
        desc="Specific Caribbean destination identified (country/island and primary city/port)",
        parent=dest_node,
        critical=True
    )

    # 2) Airport Compatibility
    airport_node = evaluator.add_parallel(
        id=f"d{idx+1}_airport_compatibility",
        desc="Airport serving the destination meets business jet operational requirements",
        parent=dest_node,
        critical=True
    )

    # 2.a) Airport Identification
    airport_id_node = evaluator.add_parallel(
        id=f"d{idx+1}_airport_identification",
        desc="Complete airport identification information provided",
        parent=airport_node,
        critical=True
    )

    # Airport name and IATA/ICAO code verification
    airport_name_code_leaf = evaluator.add_leaf(
        id=f"d{idx+1}_airport_name_code",
        desc="Official airport name and IATA/ICAO code provided and supported",
        parent=airport_id_node,
        critical=True
    )
    name_code_claim_parts = []
    if non_empty_str(dest.airport_name):
        name_code_claim_parts.append(f"official name is '{dest.airport_name}'")
    if non_empty_str(dest.iata_code):
        name_code_claim_parts.append(f"IATA code is '{dest.iata_code}'")
    if non_empty_str(dest.icao_code):
        name_code_claim_parts.append(f"ICAO code is '{dest.icao_code}'")

    name_code_claim = "The airport " + (" ".join(name_code_claim_parts) if name_code_claim_parts else "identification matches the provided details.")

    await evaluator.verify(
        claim=name_code_claim,
        node=airport_name_code_leaf,
        sources=dest.runway_specifications_url,
        additional_instruction="Verify that the airport page explicitly mentions the official name and the codes (IATA/ICAO). Allow minor formatting variations."
    )

    # Runway specifications URL existence (gate preconditions for runway checks)
    evaluator.add_custom_node(
        result=non_empty_str(dest.runway_specifications_url),
        id=f"d{idx+1}_runway_specifications_url",
        desc="URL reference provided for runway specifications",
        parent=airport_id_node,
        critical=True
    )

    # 2.b) Runway Requirements
    runway_req_node = evaluator.add_parallel(
        id=f"d{idx+1}_runway_requirements",
        desc="Airport meets physical runway and operational requirements",
        parent=airport_node,
        critical=True
    )

    # Runway length verification (>= RUNWAY_MIN_FT)
    runway_length_leaf = evaluator.add_leaf(
        id=f"d{idx+1}_runway_length",
        desc=f"Airport runway length meets or exceeds {RUNWAY_MIN_FT} feet (Bombardier Challenger 350 minimum takeoff requirement)",
        parent=runway_req_node,
        critical=True
    )
    runway_claim = f"The airport '{dest.airport_name or 'the destination airport'}' has at least one runway with a length of {RUNWAY_MIN_FT} feet or more."
    await evaluator.verify(
        claim=runway_claim,
        node=runway_length_leaf,
        sources=dest.runway_specifications_url,
        additional_instruction=f"Check runway specifications on the provided page to confirm a runway length >= {RUNWAY_MIN_FT} ft. If multiple runways exist, the longest one should be considered."
    )

    # International status (customs facilities) verification
    intl_status_leaf = evaluator.add_leaf(
        id=f"d{idx+1}_international_status",
        desc="Airport has international designation with customs facilities for business aviation",
        parent=runway_req_node,
        critical=True
    )
    intl_claim = f"The airport '{dest.airport_name or 'the destination airport'}' is an international airport with customs/immigration facilities supporting business aviation."
    await evaluator.verify(
        claim=intl_claim,
        node=intl_status_leaf,
        sources=dest.runway_specifications_url,
        additional_instruction="Verify that the page indicates international status and presence of customs/immigration services suitable for business or private aviation."
    )

    # 3) Cruise Port Access
    cruise_node = evaluator.add_parallel(
        id=f"d{idx+1}_cruise_port_access",
        desc="Destination has a cruise port accessible to major cruise lines",
        parent=dest_node,
        critical=True
    )

    # 3.a) Cruise Documentation
    cruise_doc_node = evaluator.add_parallel(
        id=f"d{idx+1}_cruise_documentation",
        desc="Complete cruise port identification and documentation",
        parent=cruise_node,
        critical=True
    )

    terminal_location_leaf = evaluator.add_leaf(
        id=f"d{idx+1}_terminal_location",
        desc="Specific cruise terminal location or address provided and supported",
        parent=cruise_doc_node,
        critical=True
    )
    terminal_location_claim = f"The cruise terminal/location '{dest.cruise_terminal_name_or_location or 'the specified terminal'}' exists at the destination."
    await evaluator.verify(
        claim=terminal_location_claim,
        node=terminal_location_leaf,
        sources=dest.cruise_port_info_url,
        additional_instruction="Verify the terminal name or location/address is explicitly mentioned on the port information page."
    )

    evaluator.add_custom_node(
        result=non_empty_str(dest.cruise_port_info_url),
        id=f"d{idx+1}_cruise_port_info_url",
        desc="URL reference provided for cruise port information",
        parent=cruise_doc_node,
        critical=True
    )

    # 3.b) Cruise Facilities
    cruise_fac_node = evaluator.add_parallel(
        id=f"d{idx+1}_cruise_facilities",
        desc="Cruise terminal facilities and services available",
        parent=cruise_node,
        critical=True
    )

    cruise_terminal_leaf = evaluator.add_leaf(
        id=f"d{idx+1}_cruise_terminal_exists",
        desc="Cruise terminal facility exists at the destination",
        parent=cruise_fac_node,
        critical=True
    )
    cruise_terminal_claim = f"The destination has an operational cruise terminal named or located as '{dest.cruise_terminal_name_or_location or 'the specified terminal'}'."
    await evaluator.verify(
        claim=cruise_terminal_claim,
        node=cruise_terminal_leaf,
        sources=dest.cruise_port_info_url,
        additional_instruction="Confirm the existence of a cruise terminal at the destination on the provided port information page."
    )

    cruise_line_leaf = evaluator.add_leaf(
        id=f"d{idx+1}_cruise_line_service",
        desc="At least one major cruise line (Viking, Royal Caribbean, Celebrity, Carnival, Norwegian, Princess, or Holland America) serves this port",
        parent=cruise_fac_node,
        critical=True
    )
    line = dest.cruise_line or "one of the major cruise lines listed"
    cruise_line_claim = f"The port is served by {line}, which is among the major cruise lines: {', '.join(MAJOR_CRUISE_LINES)}."
    await evaluator.verify(
        claim=cruise_line_claim,
        node=cruise_line_leaf,
        sources=dest.cruise_port_info_url,
        additional_instruction="Check the port schedules, ship calls, or cruise line listings to confirm that at least one of the specified major cruise lines serves the port."
    )

    # 4) Hotel Accommodation
    hotel_node = evaluator.add_parallel(
        id=f"d{idx+1}_hotel_accommodation",
        desc="Destination has accommodation from a major international hotel chain",
        parent=dest_node,
        critical=True
    )

    # 4.a) Hotel Identification
    hotel_id_node = evaluator.add_parallel(
        id=f"d{idx+1}_hotel_identification",
        desc="Complete hotel property identification and documentation",
        parent=hotel_node,
        critical=True
    )

    # Property details subgroup
    property_details_node = evaluator.add_parallel(
        id=f"d{idx+1}_property_details",
        desc="Specific property name and brand information",
        parent=hotel_id_node,
        critical=True
    )

    property_name_leaf = evaluator.add_leaf(
        id=f"d{idx+1}_property_name",
        desc="Specific hotel property name provided and supported",
        parent=property_details_node,
        critical=True
    )
    property_name_claim = f"The hotel property '{dest.hotel_property_name or 'the specified property'}' exists at the destination."
    await evaluator.verify(
        claim=property_name_claim,
        node=property_name_leaf,
        sources=dest.hotel_info_url,
        additional_instruction="Verify that the property page clearly shows the hotel property name and its association with the destination."
    )

    hotel_brand_leaf = evaluator.add_leaf(
        id=f"d{idx+1}_hotel_brand",
        desc="Specific brand within the hotel chain identified and supported",
        parent=property_details_node,
        critical=True
    )
    hotel_brand_claim = f"The property's brand is '{dest.hotel_brand or 'a listed major brand'}'."
    await evaluator.verify(
        claim=hotel_brand_claim,
        node=hotel_brand_leaf,
        sources=dest.hotel_info_url,
        additional_instruction="Verify the hotel's brand (e.g., Marriott, Hilton, Hyatt, IHG sub-brands) on the property page. Allow minor variations (e.g., 'Marriott Bonvoy' branding vs sub-brand names)."
    )

    evaluator.add_custom_node(
        result=non_empty_str(dest.hotel_info_url),
        id=f"d{idx+1}_hotel_info_url",
        desc="URL reference provided for hotel information",
        parent=hotel_id_node,
        critical=True
    )

    # 4.b) Hotel Availability (major chain presence)
    hotel_avail_node = evaluator.add_parallel(
        id=f"d{idx+1}_hotel_availability",
        desc="Hotel property from specified loyalty programs available",
        parent=hotel_node,
        critical=True
    )

    major_chain_leaf = evaluator.add_leaf(
        id=f"d{idx+1}_major_chain_presence",
        desc="At least one property from Marriott Bonvoy, Hilton Honors, World of Hyatt, or IHG One Rewards exists at the destination",
        parent=hotel_avail_node,
        critical=True
    )
    chain_presence_claim = (
        f"The property '{dest.hotel_property_name or 'the specified property'}' belongs to the "
        f"'{dest.hotel_brand or 'listed major brand'}' family and participates in one of the programs: "
        "Marriott Bonvoy, Hilton Honors, World of Hyatt, or IHG One Rewards."
    )
    await evaluator.verify(
        claim=chain_presence_claim,
        node=major_chain_leaf,
        sources=dest.hotel_info_url,
        additional_instruction="Verify that the property is part of one of the specified major loyalty programs by checking the brand and loyalty mentions on the property/brand page."
    )

    # 5) Airline Connectivity
    airline_node = evaluator.add_parallel(
        id=f"d{idx+1}_airline_connectivity",
        desc="Destination airport is served by at least one major airline alliance member",
        parent=dest_node,
        critical=True
    )

    # 5.a) Airline Documentation
    airline_doc_node = evaluator.add_parallel(
        id=f"d{idx+1}_airline_documentation",
        desc="Complete airline service identification and documentation",
        parent=airline_node,
        critical=True
    )

    specific_airlines_leaf = evaluator.add_leaf(
        id=f"d{idx+1}_specific_airlines",
        desc="At least one specific airline serving the destination identified and supported",
        parent=airline_doc_node,
        critical=True
    )
    specific_airline_claim = f"The airline '{dest.airline_name or 'the specified airline'}' serves the airport '{dest.airport_name or 'the destination airport'}'."
    await evaluator.verify(
        claim=specific_airline_claim,
        node=specific_airlines_leaf,
        sources=dest.airline_connectivity_url,
        additional_instruction="Verify the airport routes/airline page lists this airline serving the destination airport. Allow seasonal or limited-service notes."
    )

    evaluator.add_custom_node(
        result=non_empty_str(dest.airline_connectivity_url),
        id=f"d{idx+1}_airline_connectivity_url",
        desc="URL reference provided for airline service information",
        parent=airline_doc_node,
        critical=True
    )

    # 5.b) Alliance Availability
    alliance_node = evaluator.add_parallel(
        id=f"d{idx+1}_alliance_availability",
        desc="Major airline alliance service available",
        parent=airline_node,
        critical=True
    )

    alliance_service_leaf = evaluator.add_leaf(
        id=f"d{idx+1}_alliance_service",
        desc="Airport is served by at least one airline from Star Alliance, SkyTeam, or Oneworld",
        parent=alliance_node,
        critical=True
    )

    # Prepare sources for alliance verification: airline connectivity URL + alliance membership page (if alliance_name provided)
    alliance_sources: List[str] = []
    if non_empty_str(dest.airline_connectivity_url):
        alliance_sources.append(dest.airline_connectivity_url)
    # Use alliance_membership_url if provided by the answer; otherwise, derive from alliance_name if possible
    membership_url = dest.alliance_membership_url or alliance_members_page_url(dest.alliance_name)
    if non_empty_str(membership_url):
        alliance_sources.append(membership_url)

    alliance_service_claim = (
        f"The airport '{dest.airport_name or 'the destination airport'}' is served by '{dest.airline_name or 'a listed airline'}', "
        f"which is a member of '{dest.alliance_name or 'one of Star Alliance/SkyTeam/Oneworld'}'."
    )
    await evaluator.verify(
        claim=alliance_service_claim,
        node=alliance_service_leaf,
        sources=alliance_sources if alliance_sources else None,
        additional_instruction=(
            "Verify two aspects: (1) the airline serves the destination airport (from airline/airport connectivity page), "
            "(2) the airline is a member of the specified alliance (from official alliance member list page). "
            "If membership evidence is missing, be strict and require explicit membership listing."
        )
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
    Evaluate an answer for Caribbean destinations infrastructure compliance.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel; destinations evaluated independently
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

    # Extract destinations from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_destinations(),
        template_class=DestinationsExtraction,
        extraction_name="caribbean_destinations",
    )

    # Ensure we evaluate exactly two destinations (pad with empty if needed, trim if more)
    destinations: List[DestinationItem] = list(extracted.destinations[:2])
    while len(destinations) < 2:
        destinations.append(DestinationItem())

    # Build the main identification node (root-level grouping, non-critical to allow partial credit if only one destination satisfies)
    main_node = evaluator.add_parallel(
        id="caribbean_destination_identification",
        desc="Identify Caribbean destinations that satisfy all specified travel infrastructure requirements",
        parent=root,
        critical=False  # Non-critical root to allow partial scoring across multiple destinations
    )

    # Destination 1 is critical (expected to be fully correct)
    await verify_destination(
        evaluator=evaluator,
        parent_node=main_node,
        dest=destinations[0],
        idx=0,
        dest_critical=True
    )

    # Destination 2 is non-critical (allows partial credit if only one is fully correct)
    await verify_destination(
        evaluator=evaluator,
        parent_node=main_node,
        dest=destinations[1],
        idx=1,
        dest_critical=False
    )

    # Return evaluation summary
    return evaluator.get_summary()