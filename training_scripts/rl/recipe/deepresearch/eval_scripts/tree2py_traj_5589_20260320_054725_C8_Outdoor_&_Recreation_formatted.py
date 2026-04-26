import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "caribbean_cruise_2026_plan"
TASK_DESCRIPTION = """
A family from Boston is planning a Caribbean cruise vacation in late 2026. They are interested in flying to Florida and then taking a 5-night cruise that includes opportunities for snorkeling in the Turks and Caicos Islands. They have specifically heard about snorkeling at Grand Turk and also about Windsong Resort being located "on the reef."

Based on current flight and cruise options, provide a comprehensive vacation plan that includes:

1. The airline and specific airports for direct flights from Boston to the Florida Panhandle region where they can access cruise departures
2. The cruise line, ship name, and departure port for a 5-night Eastern Caribbean cruise that includes Grand Turk as a port of call
3. The ship's docking hours at Grand Turk and the best beach snorkeling locations on the island, including where to rent snorkeling equipment
4. Clarification about Windsong Resort's actual location and whether it would be accessible during their Grand Turk port stop

Ensure all information is specific, accurate, and based on actual 2026 services and geographic facts.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FlightExtraction(BaseModel):
    airline: Optional[str] = None
    airline_sources: List[str] = Field(default_factory=list)

    departure_airport_name: Optional[str] = None
    departure_airport_code: Optional[str] = None
    departure_airport_sources: List[str] = Field(default_factory=list)

    arrival_airport_name: Optional[str] = None
    arrival_airport_code: Optional[str] = None
    arrival_airport_sources: List[str] = Field(default_factory=list)

    service_start_year: Optional[str] = None
    service_start_sources: List[str] = Field(default_factory=list)


class CruiseExtraction(BaseModel):
    cruise_line: Optional[str] = None
    line_sources: List[str] = Field(default_factory=list)

    ship_name: Optional[str] = None
    ship_sources: List[str] = Field(default_factory=list)

    cruise_duration_nights: Optional[str] = None
    duration_sources: List[str] = Field(default_factory=list)

    departure_port: Optional[str] = None
    departure_port_sources: List[str] = Field(default_factory=list)

    grand_turk_included: Optional[str] = None  # e.g., "yes", "included", or null
    grand_turk_sources: List[str] = Field(default_factory=list)

    additional_port: Optional[str] = None  # e.g., "Perfect Day at CocoCay"
    additional_port_sources: List[str] = Field(default_factory=list)


class SnorkelingExtraction(BaseModel):
    docking_hours: Optional[str] = None  # e.g., "8:00 AM – 5:00 PM"
    docking_sources: List[str] = Field(default_factory=list)

    best_area: Optional[str] = None  # e.g., "west/northwest side"
    best_area_sources: List[str] = Field(default_factory=list)

    top_site: Optional[str] = None  # e.g., "Boaby Rock Point"
    top_site_sources: List[str] = Field(default_factory=list)

    accessible_site: Optional[str] = None  # e.g., "Governor's Beach"
    accessible_site_sources: List[str] = Field(default_factory=list)

    rental_location: Optional[str] = None  # e.g., "dive shops on Front Street in Cockburn Town"
    rental_sources: List[str] = Field(default_factory=list)

    currency: Optional[str] = None  # e.g., "U.S. Dollar"
    currency_sources: List[str] = Field(default_factory=list)


class WindsongExtraction(BaseModel):
    island: Optional[str] = None  # e.g., "Providenciales"
    island_sources: List[str] = Field(default_factory=list)

    area: Optional[str] = None  # e.g., "Grace Bay"
    area_sources: List[str] = Field(default_factory=list)

    reef_name: Optional[str] = None  # e.g., "Bight Reef (Coral Gardens)"
    reef_sources: List[str] = Field(default_factory=list)

    reef_distance: Optional[str] = None  # e.g., "about 20 yards from shore"
    reef_distance_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_flight_info() -> str:
    return """
    Extract flight logistics mentioned in the answer for a direct (nonstop) routing from Boston to Florida Panhandle:

    Required fields:
    - airline: the airline named for the direct service (e.g., "JetBlue")
    - airline_sources: all URLs in the answer that directly support this airline/route claim (e.g., airline route map, airport news, press release, schedules)

    - departure_airport_name: the named Boston airport (e.g., "Boston Logan International Airport")
    - departure_airport_code: the IATA code (e.g., "BOS")
    - departure_airport_sources: URLs supporting the specified Boston airport for this direct routing

    - arrival_airport_name: the named Florida Panhandle airport (e.g., "Destin–Fort Walton Beach Airport")
    - arrival_airport_code: the IATA code (e.g., "VPS")
    - arrival_airport_sources: URLs supporting this specified arrival airport

    - service_start_year: the year the answer states the service began (e.g., "2026"); if not mentioned, return null
    - service_start_sources: URLs supporting the service start-year claim

    Rules:
    - Only extract URLs that are explicitly present in the answer.
    - If a field is not present, return null (or empty list for URLs).
    """


def prompt_extract_cruise_info() -> str:
    return """
    Extract cruise details for a 5-night Eastern Caribbean cruise including Grand Turk:

    Required fields:
    - cruise_line: name of the cruise line (e.g., "Royal Caribbean International")
    - line_sources: URLs supporting the cruise line used for the itinerary

    - ship_name: the ship named (e.g., "Explorer of the Seas")
    - ship_sources: URLs supporting the specific ship and itinerary

    - cruise_duration_nights: text describing the duration (e.g., "5 nights")
    - duration_sources: URLs that show the cruise duration

    - departure_port: text of the departure port (e.g., "Port Canaveral (Orlando)")
    - departure_port_sources: URLs supporting the departure port

    - grand_turk_included: text indicating Grand Turk is a port of call (e.g., "yes", "included")
    - grand_turk_sources: URLs supporting Grand Turk as a port of call

    - additional_port: any other port cited (e.g., "Perfect Day at CocoCay")
    - additional_port_sources: URLs supporting the additional port

    Rules:
    - Extract the URLs exactly as cited in the answer.
    - If any field is not present, set it to null (or empty list for URLs).
    """


def prompt_extract_snorkeling_info() -> str:
    return """
    Extract Grand Turk port and snorkeling information cited in the answer:

    Required fields:
    - docking_hours: the stated time window in port at Grand Turk (e.g., "8:00 AM to 5:00 PM")
    - docking_sources: URLs showing the ship's schedule for Grand Turk

    - best_area: the general coastline area identified for best beach snorkeling (e.g., "west/northwest side")
    - best_area_sources: URLs supporting this

    - top_site: the named top shallow reef snorkeling site (e.g., "Boaby Rock Point")
    - top_site_sources: URLs supporting this

    - accessible_site: the most easily accessible snorkeling site (e.g., "Governor's Beach")
    - accessible_site_sources: URLs supporting this

    - rental_location: where to rent snorkeling gear (e.g., "dive shops on Front Street in Cockburn Town")
    - rental_sources: URLs supporting where to rent

    - currency: the official currency mentioned for Turks & Caicos (e.g., "U.S. Dollar")
    - currency_sources: URLs supporting the currency claim

    Rules:
    - Extract only what is explicitly in the answer.
    - If any field is not present, set it to null (or empty list for URLs).
    """


def prompt_extract_windsong_info() -> str:
    return """
    Extract Windsong Resort location and reef access details as stated in the answer:

    Required fields:
    - island: the island where Windsong Resort is located (e.g., "Providenciales")
    - island_sources: URLs supporting the island location

    - area: the area of the island (e.g., "Grace Bay")
    - area_sources: URLs supporting the area

    - reef_name: the specific reef by the resort (e.g., "Bight Reef (Coral Gardens)")
    - reef_sources: URLs supporting the reef name

    - reef_distance: how far the reef is from shore (e.g., "about 20 yards from the shoreline")
    - reef_distance_sources: URLs supporting the reef distance

    Rules:
    - Use only URLs explicitly provided in the answer.
    - If any field is not present, set it to null (or empty list for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def merge_sources(*lists: Optional[List[str]]) -> List[str]:
    """Merge multiple URL lists into a unique, ordered list."""
    seen = set()
    out: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                out.append(url)
    return out


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_flight_logistics(evaluator: Evaluator, parent_node, flight: FlightExtraction) -> None:
    """
    Build and verify the Flight Logistics subtree.
    JSON mapping:
      - Airline_Identification (critical)
      - Boston_Departure_Airport (critical)
      - Florida_Arrival_Airport (critical)
      - Service_Availability_Reference (non-critical)
    """
    node = evaluator.add_parallel(
        id="Flight_Logistics",
        desc="Verification of flight routing from Boston to Florida for cruise departure",
        parent=parent_node,
        critical=False  # Adjusted to allow mixed critical children
    )

    # 1) Airline Identification
    airline_leaf = evaluator.add_leaf(
        id="Airline_Identification",
        desc="Correctly identifies JetBlue as the airline operating direct service to Destin-Fort Walton Beach",
        parent=node,
        critical=True
    )
    airline_claim = (
        "JetBlue operates nonstop (direct) flights between Boston Logan International Airport (BOS) "
        "and Destin–Fort Walton Beach Airport (VPS) in Florida."
    )
    airline_sources = merge_sources(
        flight.airline_sources,
        flight.departure_airport_sources,
        flight.arrival_airport_sources,
    )

    # 2) Boston Departure Airport
    bos_leaf = evaluator.add_leaf(
        id="Boston_Departure_Airport",
        desc="Specifies Boston Logan International Airport (BOS) as the departure point",
        parent=node,
        critical=True
    )
    bos_claim = "The Boston departure airport for the direct flight is Boston Logan International Airport (BOS)."
    bos_sources = merge_sources(
        flight.departure_airport_sources,
        flight.airline_sources
    )

    # 3) Florida Arrival Airport
    vps_leaf = evaluator.add_leaf(
        id="Florida_Arrival_Airport",
        desc="Identifies Destin-Fort Walton Beach Airport (VPS) as the arrival airport",
        parent=node,
        critical=True
    )
    vps_claim = "The Florida arrival airport is Destin–Fort Walton Beach Airport (VPS)."
    vps_sources = merge_sources(
        flight.arrival_airport_sources,
        flight.airline_sources
    )

    # 4) Service Availability Reference (non-critical)
    svc_leaf = evaluator.add_leaf(
        id="Service_Availability_Reference",
        desc="References that JetBlue service to Destin-Fort Walton Beach began in 2026",
        parent=node,
        critical=False
    )
    stated_year = flight.service_start_year or "2026"
    svc_claim = f"JetBlue service to Destin–Fort Walton Beach (VPS) began in {stated_year}."
    svc_sources = merge_sources(flight.service_start_sources, flight.airline_sources)

    await evaluator.batch_verify([
        (
            airline_claim,
            airline_sources,
            airline_leaf,
            "Treat 'direct' and 'nonstop' as equivalent; verify that the route BOS↔VPS is shown by JetBlue or authoritative sources (airport or airline announcements)."
        ),
        (
            bos_claim,
            bos_sources,
            bos_leaf,
            "Confirm that the cited route explicitly uses Boston Logan (BOS)."
        ),
        (
            vps_claim,
            vps_sources,
            vps_leaf,
            "Confirm that the destination airport is Destin–Fort Walton Beach Airport (VPS) in Florida."
        ),
        (
            svc_claim,
            svc_sources,
            svc_leaf,
            "Look for an announcement or schedule indicating the start of JetBlue service to VPS and confirm the year."
        ),
    ])


async def verify_cruise_details(evaluator: Evaluator, parent_node, cruise: CruiseExtraction) -> None:
    """
    Build and verify the Cruise Details subtree.
    JSON mapping:
      - Cruise_Line (critical)
      - Ship_Name (critical)
      - Cruise_Duration (critical)
      - Departure_Port (critical)
      - Grand_Turk_Inclusion (critical)
      - Additional_Port (non-critical)
    """
    node = evaluator.add_parallel(
        id="Cruise_Details",
        desc="Verification of cruise line, ship, itinerary, and departure port",
        parent=parent_node,
        critical=False  # Adjusted to allow mixed critical children
    )

    itinerary_sources = merge_sources(
        cruise.line_sources,
        cruise.ship_sources,
        cruise.duration_sources,
        cruise.departure_port_sources,
        cruise.grand_turk_sources,
        cruise.additional_port_sources
    )

    # 1) Cruise Line
    cl_leaf = evaluator.add_leaf(
        id="Cruise_Line",
        desc="Correctly identifies Royal Caribbean as the cruise line",
        parent=node,
        critical=True
    )
    cl_claim = "The cruise line for the selected itinerary is Royal Caribbean International."

    # 2) Ship Name
    ship_leaf = evaluator.add_leaf(
        id="Ship_Name",
        desc="Specifies Explorer of the Seas as the ship",
        parent=node,
        critical=True
    )
    ship_claim = "The ship for this itinerary is Explorer of the Seas."

    # 3) Cruise Duration
    dur_leaf = evaluator.add_leaf(
        id="Cruise_Duration",
        desc="States the cruise is 5 nights in duration",
        parent=node,
        critical=True
    )
    dur_claim = "The cruise is 5 nights long."

    # 4) Departure Port
    port_leaf = evaluator.add_leaf(
        id="Departure_Port",
        desc="Identifies Orlando (Port Canaveral) as the departure port",
        parent=node,
        critical=True
    )
    port_claim = "The departure port is Port Canaveral (Orlando), Florida."

    # 5) Grand Turk Inclusion
    gt_leaf = evaluator.add_leaf(
        id="Grand_Turk_Inclusion",
        desc="Confirms Grand Turk, Turks & Caicos is included as a port of call",
        parent=node,
        critical=True
    )
    gt_claim = "Grand Turk (Turks and Caicos Islands) is a port of call on the itinerary."

    # 6) Additional Port (non-critical)
    addp_leaf = evaluator.add_leaf(
        id="Additional_Port",
        desc="Mentions Perfect Day at CocoCay, Bahamas as another port",
        parent=node,
        critical=False
    )
    addp_claim = "Perfect Day at CocoCay (Bahamas) is also included as a port of call on the itinerary."

    await evaluator.batch_verify([
        (
            cl_claim,
            itinerary_sources,
            cl_leaf,
            "Verify on the official Royal Caribbean site or reputable cruise listings that the itinerary is operated by Royal Caribbean."
        ),
        (
            ship_claim,
            itinerary_sources,
            ship_leaf,
            "Confirm that Explorer of the Seas operates the cited 5-night sailing including the listed ports."
        ),
        (
            dur_claim,
            itinerary_sources,
            dur_leaf,
            "Confirm the itinerary length is 5 nights."
        ),
        (
            port_claim,
            itinerary_sources,
            port_leaf,
            "Confirm Port Canaveral (often marketed as Orlando) is the departure port."
        ),
        (
            gt_claim,
            itinerary_sources,
            gt_leaf,
            "Confirm that Grand Turk is listed as a port of call on the itinerary."
        ),
        (
            addp_claim,
            itinerary_sources,
            addp_leaf,
            "Check if Perfect Day at CocoCay is included among the ports of call."
        ),
    ])


async def verify_grand_turk_snorkeling(evaluator: Evaluator, parent_node, snork: SnorkelingExtraction) -> None:
    """
    Build and verify the Grand Turk Snorkeling subtree.
    JSON mapping:
      - Docking_Hours (critical)
      - Best_Snorkeling_Area (critical)
      - Top_Snorkeling_Site (critical)
      - Most_Accessible_Site (critical)
      - Equipment_Rental_Location (critical)
      - Currency_Information (non-critical)
    """
    node = evaluator.add_parallel(
        id="Grand_Turk_Snorkeling",
        desc="Verification of snorkeling information at Grand Turk port",
        parent=parent_node,
        critical=False  # Adjusted to allow mixed critical children
    )

    # 1) Docking Hours
    dock_leaf = evaluator.add_leaf(
        id="Docking_Hours",
        desc="Provides the ship's docking hours at Grand Turk (8:00 AM to 5:00 PM)",
        parent=node,
        critical=True
    )
    # Prefer using extracted text if provided; otherwise use the canonical 8–5 statement.
    docking_text = snork.docking_hours or "8:00 AM to 5:00 PM"
    dock_claim = f"The scheduled time in port at Grand Turk for the referenced itinerary is {docking_text}."
    dock_sources = merge_sources(snork.docking_sources)

    # 2) Best Snorkeling Area
    area_leaf = evaluator.add_leaf(
        id="Best_Snorkeling_Area",
        desc="Identifies west/northwest side as the location of best beach snorkeling",
        parent=node,
        critical=True
    )
    area_claim = (
        "On Grand Turk, the best shore-accessible snorkeling is along the west and northwest side of the island."
    )
    area_sources = merge_sources(snork.best_area_sources)

    # 3) Top Snorkeling Site
    top_leaf = evaluator.add_leaf(
        id="Top_Snorkeling_Site",
        desc="Names Boaby Rock Point as offering the best shallow reef snorkeling when conditions are calm",
        parent=node,
        critical=True
    )
    top_claim = "Boaby Rock Point offers excellent shallow reef snorkeling when sea conditions are calm."
    top_sources = merge_sources(snork.top_site_sources)

    # 4) Most Accessible Site
    access_leaf = evaluator.add_leaf(
        id="Most_Accessible_Site",
        desc="Identifies Governor's Beach as the most easily accessible snorkeling location",
        parent=node,
        critical=True
    )
    access_claim = "Governor's Beach is one of the most easily accessible snorkeling locations on Grand Turk."
    access_sources = merge_sources(snork.accessible_site_sources)

    # 5) Equipment Rental Location
    rent_leaf = evaluator.add_leaf(
        id="Equipment_Rental_Location",
        desc="Specifies that snorkeling equipment can be rented from dive shops on Front Street in Cockburn Town",
        parent=node,
        critical=True
    )
    rent_claim = "Snorkeling equipment can be rented from dive shops on Front Street in Cockburn Town on Grand Turk."
    rent_sources = merge_sources(snork.rental_sources)

    # 6) Currency Information (non-critical)
    curr_leaf = evaluator.add_leaf(
        id="Currency_Information",
        desc="Notes that U.S. Dollar is the official currency in Turks & Caicos",
        parent=node,
        critical=False
    )
    curr_claim = "The official currency of the Turks and Caicos Islands is the U.S. Dollar (USD)."
    curr_sources = merge_sources(snork.currency_sources)

    await evaluator.batch_verify([
        (
            dock_claim,
            dock_sources,
            dock_leaf,
            "Use the itinerary schedule or port schedule page for the specific sailing date. Reasonable formatting variants of times are acceptable."
        ),
        (
            area_claim,
            area_sources,
            area_leaf,
            "Verify via authoritative destination guides (e.g., official tourism or well-regarded snorkeling guides) that the west/northwest side has the best shore snorkeling."
        ),
        (
            top_claim,
            top_sources,
            top_leaf,
            "Confirm that Boaby Rock Point is highlighted for top shallow reef snorkeling during calm conditions."
        ),
        (
            access_claim,
            access_sources,
            access_leaf,
            "Confirm that Governor's Beach is identified as an easily accessible snorkeling spot."
        ),
        (
            rent_claim,
            rent_sources,
            rent_leaf,
            "Confirm that snorkeling gear rentals are available from dive shops on Front Street in Cockburn Town."
        ),
        (
            curr_claim,
            curr_sources,
            curr_leaf,
            "Confirm that the USD is the official currency in the Turks and Caicos Islands."
        ),
    ])


async def verify_windsong_resort(evaluator: Evaluator, parent_node, wind: WindsongExtraction) -> None:
    """
    Build and verify the Windsong Resort Clarification subtree.
    JSON mapping:
      - Resort_Island_Location (critical)
      - Resort_Area_Location (critical)
      - Reef_Name (critical)
      - Reef_Distance (non-critical)
    """
    node = evaluator.add_parallel(
        id="Windsong_Resort_Clarification",
        desc="Clarification of Windsong Resort location and reef access",
        parent=parent_node,
        critical=False  # Adjusted to allow mixed critical children
    )

    # 1) Resort Island Location
    island_leaf = evaluator.add_leaf(
        id="Resort_Island_Location",
        desc="Clarifies that Windsong Resort is located in Providenciales, not Grand Turk",
        parent=node,
        critical=True
    )
    island_claim = "Windsong Resort is located on the island of Providenciales (not on Grand Turk) in Turks and Caicos."
    island_sources = merge_sources(wind.island_sources, wind.area_sources, wind.reef_sources)

    # 2) Resort Area Location
    area_leaf = evaluator.add_leaf(
        id="Resort_Area_Location",
        desc="Specifies that Windsong is located on Grace Bay",
        parent=node,
        critical=True
    )
    area_claim = "Windsong Resort is located on Grace Bay, Providenciales."
    area_sources = merge_sources(wind.area_sources, wind.reef_sources)

    # 3) Reef Name
    reef_leaf = evaluator.add_leaf(
        id="Reef_Name",
        desc="Identifies the resort is positioned on Bight Reef (Coral Gardens)",
        parent=node,
        critical=True
    )
    reef_claim = "Windsong Resort sits by Bight Reef, also known as Coral Gardens, on Providenciales."
    reef_sources = merge_sources(wind.reef_sources)

    # 4) Reef Distance (non-critical)
    dist_leaf = evaluator.add_leaf(
        id="Reef_Distance",
        desc="States the reef is approximately 20 yards from the shoreline",
        parent=node,
        critical=False
    )
    stated_distance = wind.reef_distance or "about 20 yards from the shoreline"
    dist_claim = f"The reef by Windsong Resort is approximately {stated_distance}."
    dist_sources = merge_sources(wind.reef_distance_sources, wind.reef_sources)

    await evaluator.batch_verify([
        (
            island_claim,
            island_sources,
            island_leaf,
            "Verify using the official resort site or reputable travel guides that Windsong is on Providenciales, not Grand Turk."
        ),
        (
            area_claim,
            area_sources,
            area_leaf,
            "Confirm that the resort is on Grace Bay."
        ),
        (
            reef_claim,
            reef_sources,
            reef_leaf,
            "Confirm that the resort fronts Bight Reef (also called Coral Gardens)."
        ),
        (
            dist_claim,
            dist_sources,
            dist_leaf,
            "Confirm an approximate near-shore distance (≈20 yards) is cited; small variations are acceptable."
        ),
    ])


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
    Evaluate an answer for the 2026 Caribbean cruise vacation plan task.
    """
    # Initialize evaluator (root is non-critical to allow mixed-critical children)
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

    # Extract all major components in parallel
    flight_task = evaluator.extract(
        prompt=prompt_extract_flight_info(),
        template_class=FlightExtraction,
        extraction_name="flight_info"
    )
    cruise_task = evaluator.extract(
        prompt=prompt_extract_cruise_info(),
        template_class=CruiseExtraction,
        extraction_name="cruise_info"
    )
    snork_task = evaluator.extract(
        prompt=prompt_extract_snorkeling_info(),
        template_class=SnorkelingExtraction,
        extraction_name="snorkeling_info"
    )
    wind_task = evaluator.extract(
        prompt=prompt_extract_windsong_info(),
        template_class=WindsongExtraction,
        extraction_name="windsong_info"
    )

    flight, cruise, snork, wind = await asyncio.gather(
        flight_task, cruise_task, snork_task, wind_task
    )

    # Build verification subtrees
    await verify_flight_logistics(evaluator, root, flight)
    await verify_cruise_details(evaluator, root, cruise)
    await verify_grand_turk_snorkeling(evaluator, root, snork)
    await verify_windsong_resort(evaluator, root, wind)

    # Return structured result
    return evaluator.get_summary()