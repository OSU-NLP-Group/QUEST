import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "celebrity_eclipse_2027_barcelona_roundtrip"
TASK_DESCRIPTION = (
    "A travel enthusiast wants to book a cruise to view the total solar eclipse occurring on August 2, 2027. "
    "They are specifically interested in a Celebrity Cruises voyage that departs from Barcelona, Spain on July 31, 2027 "
    "and returns to Barcelona. The cruise must be exactly 7 nights in duration and the ship must have a passenger capacity "
    "of at least 3,000 at double occupancy. The traveler requires detailed verification of the itinerary, including: the name "
    "of the specific Celebrity Cruises ship operating this voyage; the exact passenger capacity of the ship; the exact departure "
    "time from Barcelona on July 31, 2027; the exact return date to Barcelona; confirmation that the eclipse viewing occurs on "
    "Day 4 of the cruise (August 2, 2027); and complete details for three specific port stops: Palma de Mallorca, Spain (including "
    "the cruise day number, arrival time, and departure time); Malaga, Spain (including the cruise day number, arrival time, and "
    "departure time); and Ibiza, Balearic Islands (including the cruise day number, arrival time, and departure time). Provide all "
    "of this verified information with supporting URL references from Celebrity Cruises' official website or reliable cruise information sources."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class PortStopExtract(BaseModel):
    day_number: Optional[str] = None
    arrival_time: Optional[str] = None
    departure_time: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class CruiseExtraction(BaseModel):
    # Core cruise/operator/itinerary
    operator: Optional[str] = None
    departure_port: Optional[str] = None
    return_port: Optional[str] = None
    departure_date: Optional[str] = None
    departure_time: Optional[str] = None
    return_date: Optional[str] = None
    duration_nights: Optional[str] = None

    # Eclipse specifics
    eclipse_date: Optional[str] = None
    eclipse_day_number: Optional[str] = None

    # Ship specifics
    ship_name: Optional[str] = None
    passenger_capacity: Optional[str] = None  # exact number as stated (double occupancy)

    # URL buckets
    schedule_urls: List[str] = Field(default_factory=list)         # itinerary/route/dates/ports/depart time
    ship_capacity_urls: List[str] = Field(default_factory=list)    # ship name + capacity
    eclipse_urls: List[str] = Field(default_factory=list)          # eclipse timing + day

    # Port detail blocks + optional per-port URLs (redundant but helpful)
    palma: Optional[PortStopExtract] = None
    malaga: Optional[PortStopExtract] = None
    ibiza: Optional[PortStopExtract] = None

    palma_urls: List[str] = Field(default_factory=list)
    malaga_urls: List[str] = Field(default_factory=list)
    ibiza_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cruise() -> str:
    return """
    Extract all information the answer provides about the specific Celebrity Cruises 7-night Barcelona roundtrip eclipse voyage.

    Required scalar fields (extract exactly as written in the answer text; use strings):
    - operator: the cruise operator/brand (e.g., "Celebrity Cruises")
    - departure_port: embarkation port (e.g., "Barcelona, Spain")
    - return_port: disembarkation/return port (e.g., "Barcelona, Spain")
    - departure_date: the departure date (e.g., "July 31, 2027" or "2027-07-31")
    - departure_time: the exact departure time from Barcelona on departure date (e.g., "5:00 PM" or "17:00")
    - return_date: the final return date to Barcelona (e.g., "August 7, 2027")
    - duration_nights: the cruise duration in nights (e.g., "7" or "7 nights")
    - eclipse_date: the date of the total solar eclipse (e.g., "August 2, 2027")
    - eclipse_day_number: the day number within the cruise on which eclipse viewing occurs (e.g., "4")
    - ship_name: the exact ship name operating the voyage (e.g., "Celebrity Ascent")
    - passenger_capacity: the exact passenger capacity at double occupancy for the ship (e.g., "3,260")

    URL buckets (extract actual URLs mentioned in the answer text):
    - schedule_urls: URL(s) for the itinerary/schedule/route/dates/ports/departure time (e.g., Celebrity itinerary page)
    - ship_capacity_urls: URL(s) that state the ship name and passenger capacity (double occupancy)
    - eclipse_urls: URL(s) confirming eclipse timing and day number for this voyage
    - palma_urls: URL(s) supporting Palma de Mallorca day number and arrival/departure times
    - malaga_urls: URL(s) supporting Malaga day number and arrival/departure times
    - ibiza_urls: URL(s) supporting Ibiza day number and arrival/departure times

    Port detail blocks (if provided):
    - palma: an object with fields day_number, arrival_time, departure_time, and source_urls (list)
    - malaga: same structure
    - ibiza: same structure

    Rules:
    - Return null for any scalar field not mentioned explicitly.
    - For URLs, return empty lists if not provided.
    - Do not fabricate or infer values not present in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return lst if lst else []


def _merge_urls(*lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for lst in lists:
        for u in lst or []:
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_base_cruise_constraints(evaluator: Evaluator, parent, ext: CruiseExtraction) -> None:
    base = evaluator.add_parallel(
        id="Base_Cruise_Constraints",
        desc="Verify cruise line/route/dates/duration/eclipse timing constraints.",
        parent=parent,
        critical=True
    )

    schedule_sources = _safe_list(ext.schedule_urls)
    eclipse_sources = _safe_list(ext.eclipse_urls)

    # Operated by Celebrity Cruises
    node = evaluator.add_leaf(
        id="Operated_By_Celebrity_Cruises",
        desc="Cruise is operated by Celebrity Cruises.",
        parent=base,
        critical=True
    )
    await evaluator.verify(
        claim="The cruise is operated by Celebrity Cruises.",
        node=node,
        sources=schedule_sources,
        additional_instruction="Look for 'Celebrity Cruises' or 'Celebrity' branding on the itinerary/operator section."
    )

    # Departure port is Barcelona
    node = evaluator.add_leaf(
        id="Departs_From_Barcelona",
        desc="Departure port is Barcelona, Spain.",
        parent=base,
        critical=True
    )
    await evaluator.verify(
        claim="The departure/embarkation port is Barcelona, Spain.",
        node=node,
        sources=schedule_sources,
        additional_instruction="Accept minor formatting variants like 'Barcelona (Spain)'."
    )

    # Return port is Barcelona
    node = evaluator.add_leaf(
        id="Returns_To_Barcelona",
        desc="Return port is Barcelona, Spain.",
        parent=base,
        critical=True
    )
    await evaluator.verify(
        claim="The return/disembarkation port is Barcelona, Spain.",
        node=node,
        sources=schedule_sources,
        additional_instruction="Accept minor formatting variants like 'Barcelona (Spain)'."
    )

    # Departure date July 31, 2027
    node = evaluator.add_leaf(
        id="Departure_Date_Is_2027_07_31",
        desc="Departure date is July 31, 2027.",
        parent=base,
        critical=True
    )
    await evaluator.verify(
        claim="The departure date is July 31, 2027.",
        node=node,
        sources=schedule_sources,
        additional_instruction="Allow equivalent formats such as '31 Jul 2027' or '2027-07-31'."
    )

    # Departure time 5:00 PM
    node = evaluator.add_leaf(
        id="Departure_Time_Is_5_00_PM",
        desc="Departure time from Barcelona on July 31, 2027 is exactly 5:00 PM.",
        parent=base,
        critical=True
    )
    await evaluator.verify(
        claim="The departure time from Barcelona on July 31, 2027 is 5:00 PM.",
        node=node,
        sources=schedule_sources,
        additional_instruction="Allow 24-hour format equivalence (e.g., 17:00 = 5:00 PM); ignore seconds/spaces."
    )

    # Duration exactly 7 nights
    node = evaluator.add_leaf(
        id="Duration_Is_Exactly_7_Nights",
        desc="Cruise duration is exactly 7 nights.",
        parent=base,
        critical=True
    )
    await evaluator.verify(
        claim="The cruise duration is exactly 7 nights.",
        node=node,
        sources=schedule_sources,
        additional_instruction="Accept '7-Night' or '7 Nights' as equivalent."
    )

    # Return date Aug 7, 2027
    node = evaluator.add_leaf(
        id="Return_Date_Is_2027_08_07",
        desc="Return date is August 7, 2027.",
        parent=base,
        critical=True
    )
    await evaluator.verify(
        claim="The return date to Barcelona is August 7, 2027.",
        node=node,
        sources=schedule_sources,
        additional_instruction="Allow equivalent formats such as '07 Aug 2027' or '2027-08-07'."
    )

    # Eclipse is Aug 2, 2027
    node = evaluator.add_leaf(
        id="Eclipse_Is_Aug_2_2027",
        desc="The cruise is specifically for viewing the total solar eclipse occurring on August 2, 2027.",
        parent=base,
        critical=True
    )
    await evaluator.verify(
        claim="This cruise is specifically for viewing the total solar eclipse occurring on August 2, 2027.",
        node=node,
        sources=eclipse_sources,
        additional_instruction="Look for explicit mention of the 'total solar eclipse' and the date 'August 2, 2027'."
    )

    # Eclipse occurs on Day 4
    node = evaluator.add_leaf(
        id="Eclipse_Occurs_On_Cruise_Day_4",
        desc="Eclipse viewing occurs on Day 4 of the cruise (August 2, 2027).",
        parent=base,
        critical=True
    )
    await evaluator.verify(
        claim="Eclipse viewing occurs on Day 4 (August 2, 2027) of the cruise.",
        node=node,
        sources=eclipse_sources,
        additional_instruction="Confirm both the day number (Day 4) and the date (Aug 2, 2027) are tied to eclipse viewing."
    )


async def build_ship_requirements(evaluator: Evaluator, parent, ext: CruiseExtraction) -> None:
    ship_group = evaluator.add_parallel(
        id="Ship_Requirements",
        desc="Verify the specific ship and its passenger capacity requirement.",
        parent=parent,
        critical=True
    )

    schedule_sources = _safe_list(ext.schedule_urls)
    ship_sources = _safe_list(ext.ship_capacity_urls)
    ship_name = ext.ship_name or ""

    # Ship name provided (and correct for this voyage)
    node = evaluator.add_leaf(
        id="Ship_Name_Provided",
        desc="Provide the name of the specific Celebrity Cruises ship operating this voyage.",
        parent=ship_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The voyage is operated by the ship {ship_name}.",
        node=node,
        sources=schedule_sources,
        additional_instruction="Match the ship name shown on the itinerary page. Minor variants (with/without 'Celebrity ') are acceptable."
    )

    # Passenger capacity exact (double occupancy)
    node = evaluator.add_leaf(
        id="Passenger_Capacity_Exact_Provided",
        desc="Provide the ship's exact passenger capacity at double occupancy.",
        parent=ship_group,
        critical=True
    )
    capacity_text = ext.passenger_capacity or ""
    await evaluator.verify(
        claim=f"The exact passenger capacity at double occupancy of the ship {ship_name or 'this ship'} is {capacity_text}.",
        node=node,
        sources=ship_sources,
        additional_instruction="Focus on double-occupancy (guests at double occupancy). Accept comma separators in numbers."
    )

    # Passenger capacity at least 3,000
    node = evaluator.add_leaf(
        id="Passenger_Capacity_At_Least_3000",
        desc="Confirm the ship's passenger capacity is at least 3,000 at double occupancy.",
        parent=ship_group,
        critical=True
    )
    await evaluator.verify(
        claim="The ship's passenger capacity at double occupancy is at least 3,000.",
        node=node,
        sources=ship_sources,
        additional_instruction="Verify the numeric capacity stated on the page is >= 3000 (e.g., 3,200; 3,260)."
    )


async def _build_single_port(
    evaluator: Evaluator,
    parent,
    node_id_prefix: str,
    display_port_name: str,
    expected_day: str,
    expected_arrival: str,
    expected_departure: str,
    urls: List[str],
) -> None:
    port_node = evaluator.add_parallel(
        id=node_id_prefix,
        desc=f"Verify {display_port_name} port stop requirements.",
        parent=parent,
        critical=True
    )

    # Included
    leaf = evaluator.add_leaf(
        id=f"{node_id_prefix.split('_Stop')[0]}_Included" if node_id_prefix.endswith("_Stop") else f"{node_id_prefix}_Included",
        desc=f"Itinerary includes a port stop at {display_port_name}.",
        parent=port_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The itinerary includes a port stop at {display_port_name}.",
        node=leaf,
        sources=urls,
        additional_instruction=f"Accept naming variants (e.g., '{display_port_name}' with region in parentheses)."
    )

    # Day number
    leaf = evaluator.add_leaf(
        id=f"{node_id_prefix.split('_Stop')[0]}_Day_Number" if node_id_prefix.endswith("_Stop") else f"{node_id_prefix}_Day_Number",
        desc=f"{display_port_name} stop occurs on Day {expected_day} of the cruise.",
        parent=port_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The {display_port_name} stop occurs on Day {expected_day} of the cruise.",
        node=leaf,
        sources=urls,
        additional_instruction="Itineraries often show 'Day X: <port>'. Minor formatting differences are acceptable."
    )

    # Docking window (arrival/departure)
    leaf = evaluator.add_leaf(
        id=f"{node_id_prefix.split('_Stop')[0]}_Docking_Window" if node_id_prefix.endswith("_Stop") else f"{node_id_prefix}_Docking_Window",
        desc=f"{display_port_name} docking window is {expected_arrival} to {expected_departure}.",
        parent=port_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At {display_port_name}, the published arrival time is {expected_arrival} and the departure time is {expected_departure}.",
        node=leaf,
        sources=urls,
        additional_instruction="Allow 24-hour formatting equivalence (e.g., 07:00 = 7:00 AM; 18:00 = 6:00 PM; 11:59 = 11:59 AM)."
    )


async def build_required_port_stops(evaluator: Evaluator, parent, ext: CruiseExtraction) -> None:
    ports_group = evaluator.add_parallel(
        id="Required_Port_Stops",
        desc="Verify the three required port stops and their constraint-specified day numbers and docking windows.",
        parent=parent,
        critical=True
    )

    # Palma de Mallorca (Day 2, 7:00 AM - 6:00 PM)
    palma_sources = _merge_urls(ext.palma_urls, ext.palma.source_urls if ext.palma else None)
    await _build_single_port(
        evaluator=evaluator,
        parent=ports_group,
        node_id_prefix="Palma_de_Mallorca_Stop",
        display_port_name="Palma de Mallorca, Spain",
        expected_day="2",
        expected_arrival="7:00 AM",
        expected_departure="6:00 PM",
        urls=palma_sources
    )

    # Malaga (Day 6, 7:00 AM - 5:00 PM)
    malaga_sources = _merge_urls(ext.malaga_urls, ext.malaga.source_urls if ext.malaga else None)
    await _build_single_port(
        evaluator=evaluator,
        parent=ports_group,
        node_id_prefix="Malaga_Stop",
        display_port_name="Malaga, Spain",
        expected_day="6",
        expected_arrival="7:00 AM",
        expected_departure="5:00 PM",
        urls=malaga_sources
    )

    # Ibiza (Day 7, 11:59 AM - 7:00 PM)
    ibiza_sources = _merge_urls(ext.ibiza_urls, ext.ibiza.source_urls if ext.ibiza else None)
    await _build_single_port(
        evaluator=evaluator,
        parent=ports_group,
        node_id_prefix="Ibiza_Stop",
        display_port_name="Ibiza, Balearic Islands",
        expected_day="7",
        expected_arrival="11:59 AM",
        expected_departure="7:00 PM",
        urls=ibiza_sources
    )


def build_supporting_url_references(evaluator: Evaluator, parent, ext: CruiseExtraction) -> None:
    refs = evaluator.add_parallel(
        id="Supporting_URL_References",
        desc="Provide supporting URL reference(s) (Celebrity official website or other reliable cruise info sources) that substantiate all required claims.",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(_safe_list(ext.schedule_urls)) > 0,
        id="URL_For_Cruise_Schedule_And_Itinerary",
        desc="Provide URL(s) supporting the cruise operator, embark/disembark ports, dates, duration, and departure time.",
        parent=refs,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(_safe_list(ext.ship_capacity_urls)) > 0,
        id="URL_For_Ship_Name_And_Capacity",
        desc="Provide URL(s) supporting the ship name and passenger capacity (double occupancy).",
        parent=refs,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(_safe_list(ext.eclipse_urls)) > 0,
        id="URL_For_Eclipse_Timing",
        desc="Provide URL(s) supporting that eclipse viewing is on August 2, 2027 and occurs on Day 4 of the cruise.",
        parent=refs,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(_safe_list(ext.palma_urls)) > 0,
        id="URL_For_Palma_Schedule",
        desc="Provide URL(s) supporting Palma de Mallorca day number and arrival/departure (docking) times.",
        parent=refs,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(_safe_list(ext.malaga_urls)) > 0,
        id="URL_For_Malaga_Schedule",
        desc="Provide URL(s) supporting Malaga day number and arrival/departure (docking) times.",
        parent=refs,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(_safe_list(ext.ibiza_urls)) > 0,
        id="URL_For_Ibiza_Schedule",
        desc="Provide URL(s) supporting Ibiza day number and arrival/departure (docking) times.",
        parent=refs,
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
    model: str = "o4-mini",
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
        default_model=model,
    )

    # Extract structured data from the answer
    extraction: CruiseExtraction = await evaluator.extract(
        prompt=prompt_extract_cruise(),
        template_class=CruiseExtraction,
        extraction_name="cruise_extraction"
    )

    # Add ground-truth constraints (for transparency; not used for auto-grading)
    evaluator.add_ground_truth({
        "expected_constraints": {
            "operator": "Celebrity Cruises",
            "departure_port": "Barcelona, Spain",
            "return_port": "Barcelona, Spain",
            "departure_date": "July 31, 2027",
            "departure_time": "5:00 PM",
            "duration_nights": "7",
            "return_date": "August 7, 2027",
            "eclipse_date": "August 2, 2027",
            "eclipse_day_number": "4",
            "ports": {
                "Palma de Mallorca, Spain": {"day": "2", "arrival": "7:00 AM", "departure": "6:00 PM"},
                "Malaga, Spain": {"day": "6", "arrival": "7:00 AM", "departure": "5:00 PM"},
                "Ibiza, Balearic Islands": {"day": "7", "arrival": "11:59 AM", "departure": "7:00 PM"}
            },
            "capacity_min_double_occupancy": "3000"
        }
    })

    # Build top-level critical group node (matches rubric root section)
    top = evaluator.add_parallel(
        id="2027_Solar_Eclipse_Cruise_Verification",
        desc="Verify all constraint-required details of the specified Celebrity Cruises solar eclipse cruise.",
        parent=root,
        critical=True
    )

    # Build subtrees
    await build_base_cruise_constraints(evaluator, top, extraction)
    await build_ship_requirements(evaluator, top, extraction)
    await build_required_port_stops(evaluator, top, extraction)
    build_supporting_url_references(evaluator, top, extraction)

    # Return the full evaluation summary
    return evaluator.get_summary()