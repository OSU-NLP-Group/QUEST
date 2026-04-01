import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.llm_client.base_client import LLMClient
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import Evaluator, VerificationNode, AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "baltimore_marc_station"
TASK_DESCRIPTION = """
I'm moving to Baltimore soon and have heard great things about the MARC Train between Camden Station in Baltimore and Union Station in DC. I'd like to get a place in a suburb close to a MARC train stop.
Could you first find all the MARC train stops between these two stations and clearly provide the name of the town or city where each stop is located, as well as its specific address. Then, for each unique town or city identified, find the Total Crime Index from NeighborhoodScout. 
"""

EVAL_NOTES = "Evaluating MARC train stations between Camden and Union Station with location and crime data"

# Ground truth data for verification - expected station-city pairs
EXPECTED_STATIONS = {
    "St. Denis": "Halethorpe, Maryland",
    "Dorsey": "Dorsey, Maryland",
    "Jessup": "Jessup, Maryland",
    "Savage": "Savage, Maryland",
    "Laurel Race Track": "Laurel, Maryland",
    "Laurel Park": "Laurel, Maryland",
    "Laurel": "Laurel, Maryland",
    "Muirkirk": "Beltsville, Maryland",
    "Greenbelt": "Greenbelt, Maryland",
    "College Park": "College Park, Maryland",
    "Riverdale": "Riverdale Park, Maryland",
    "Riverdale Park": "Riverdale Park, Maryland",
    "Riverdale Park Town Center": "Riverdale Park, Maryland"
}

GROUND_TRUTH = {
    "expected_stations": EXPECTED_STATIONS,
    "total_expected_stations": len(EXPECTED_STATIONS)
}


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class StationNamesList(BaseModel):
    """Extract list of MARC station names from the answer."""
    station_names: List[str] = Field(
        default_factory=list,
        description="List of MARC station names mentioned in the answer"
    )


class SingleStationInfo(BaseModel):
    """Extract detailed information for a single MARC station."""
    station_name: Optional[str] = Field(
        default=None,
        description="The exact name of the MARC station"
    )
    town_city: Optional[str] = Field(
        default=None,
        description="The town or city where this station is located"
    )
    address: Optional[str] = Field(
        default=None,
        description="The specific street address of this station"
    )
    supporting_urls: List[str] = Field(
        default_factory=list,
        description="URLs supporting station location/address information"
    )
    crime_index: Optional[str] = Field(
        default=None,
        description="Total Crime Index value from NeighborhoodScout"
    )
    crime_index_urls: List[str] = Field(
        default_factory=list,
        description="URLs from NeighborhoodScout supporting crime index data"
    )


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_station_names() -> str:
    """Extract all MARC station names between Camden and Union Station."""
    return """
    Extract all MARC train station names mentioned in the answer that are between Camden Station (Baltimore) and Union Station (DC). 

    Important:
    - Do NOT include Camden Station or Union Station themselves
    - Only extract intermediate stations along the route
    - Return them as a list in the order they appear in the answer
    - If no stations are found, return an empty list
    """


def prompt_extract_single_station_info(station_name: str) -> str:
    """Extract comprehensive information for a specific MARC station."""
    return f"""
    Extract comprehensive information for the specific MARC station "{station_name}" from the answer. Look for:

    1. station_name: The exact name of this station (should match "{station_name}")
    2. town_city: The town or city where this station is located
    3. address: The specific street address of this station
    4. supporting_urls: Any URLs that support information about this station's location or address
    5. crime_index: The Total Crime Index value for the town/city where this station is located (from NeighborhoodScout)
    6. crime_index_urls: Any URLs from NeighborhoodScout that support the crime index information

    Extract information only for station "{station_name}". If information is not found or clearly stated, return null for that field.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_station_existence_and_location(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        station_info: SingleStationInfo,
        station_index: int,
) -> None:
    """
    Verify that the station exists in ground truth and location is correct.
    Uses the recommended pattern with parallel nodes and critical existence checks.
    """
    # Create a wrapper for location verification
    location_wrapper = evaluator.add_parallel(
        id=f"station_{station_index}_location_verification",
        desc=f"Location verification for station {station_index + 1}",
        parent=parent_node,
        critical=True
    )
    
    # Station existence and data completeness check (critical)
    has_station_name = bool(station_info.station_name and station_info.station_name.strip())
    has_location = bool(station_info.town_city and station_info.town_city.strip())
    
    existence_node = evaluator.add_custom_node(
        result=has_station_name and has_location,
        id=f"station_{station_index}_location_data_exists",
        desc=f"Station {station_index + 1} has name '{station_info.station_name or 'Not found'}' and location '{station_info.town_city or 'Not found'}'",
        parent=location_wrapper,
        critical=True  # Critical - gates the verification
    )

    # Location verification
    location_verify_node = evaluator.add_leaf(
        id=f"station_{station_index}_location_correct",
        desc=f"Station '{station_info.station_name or 'N/A'}' is correctly located in '{station_info.town_city or 'N/A'}' according to ground truth",
        parent=location_wrapper,
        critical=True
    )

    # Always call verify - let the existence check gate if needed
    ground_truth_pairs = [f"{station}: {city}" for station, city in EXPECTED_STATIONS.items()]
    ground_truth_text = "; ".join(ground_truth_pairs)

    claim = f"The station-city pair '{station_info.station_name}' in '{station_info.town_city}' matches one of the expected MARC station-city pairs from the ground truth: {ground_truth_text}"

    await evaluator.verify(
        claim=claim,
        node=location_verify_node,
        sources=None,  # Ground truth verification
        additional_instruction="Allow for reasonable variations in station names and city/town names (e.g., 'Laurel' vs 'Laurel, Maryland', case differences, etc.). Focus on whether the pairing is substantially correct."
    )


async def verify_station_address(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        station_info: SingleStationInfo,
        station_index: int,
) -> None:
    """Verify that the station address is correct and supported by URLs."""
    # Create a wrapper for address verification
    address_wrapper = evaluator.add_parallel(
        id=f"station_{station_index}_address_verification",
        desc=f"Address verification for station {station_index + 1}",
        parent=parent_node,
        critical=False
    )
    
    # Combined existence check for address and URLs
    has_address = bool(station_info.address and station_info.address.strip())
    has_urls = bool(station_info.supporting_urls)
    
    existence_node = evaluator.add_custom_node(
        result=has_address and has_urls,
        id=f"station_{station_index}_address_data_exists",
        desc=f"Station {station_index + 1} has address '{station_info.address or 'Not provided'}' and {len(station_info.supporting_urls)} supporting URLs",
        parent=address_wrapper,
        critical=True  # Critical - gates the verification
    )

    # Address verification with URL support
    address_verify_node = evaluator.add_leaf(
        id=f"station_{station_index}_address_verified",
        desc=f"Address '{station_info.address or 'N/A'}' for station '{station_info.station_name or 'N/A'}' is accurate and supported by provided URLs",
        parent=address_wrapper,
        critical=True
    )

    # Always call verify
    claim = f"The address '{station_info.address}' is the correct address for MARC station '{station_info.station_name}'. The webpage should clearly show this is the station's address (e.g., on a Google Maps page, transit authority page, or explicitly stating this address belongs to the station)."

    await evaluator.verify(
        claim=claim,
        node=address_verify_node,
        sources=station_info.supporting_urls,  # Pass list directly
        additional_instruction="Look for clear evidence that this address corresponds to the MARC station. Accept various formats of address presentation."
    )


async def verify_station_crime_index(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        station_info: SingleStationInfo,
        station_index: int,
) -> None:
    """Verify crime index data from NeighborhoodScout."""
    # Create a wrapper for crime verification
    crime_wrapper = evaluator.add_parallel(
        id=f"station_{station_index}_crime_verification",
        desc=f"Crime index verification for station {station_index + 1}",
        parent=parent_node,
        critical=False
    )
    
    # Combined existence check for crime index and URLs
    has_crime_index = bool(station_info.crime_index and station_info.crime_index.strip())
    has_crime_urls = bool(station_info.crime_index_urls)
    
    existence_node = evaluator.add_custom_node(
        result=has_crime_index and has_crime_urls,
        id=f"station_{station_index}_crime_data_exists",
        desc=f"Station {station_index + 1} has crime index '{station_info.crime_index or 'Not provided'}' and {len(station_info.crime_index_urls)} NeighborhoodScout URLs",
        parent=crime_wrapper,
        critical=True  # Critical - gates the verification
    )

    # Crime index verification
    crime_verify_node = evaluator.add_leaf(
        id=f"station_{station_index}_crime_verified",
        desc=f"Crime index '{station_info.crime_index or 'N/A'}' for '{station_info.town_city or 'N/A'}' is accurate according to NeighborhoodScout",
        parent=crime_wrapper,
        critical=True
    )

    # Always call verify
    claim = f"According to NeighborhoodScout, the Total Crime Index for {station_info.town_city} is '{station_info.crime_index}'. The webpage should clearly show this is the crime index value for this location."

    await evaluator.verify(
        claim=claim,
        node=crime_verify_node,
        sources=station_info.crime_index_urls,  # Pass list directly
        additional_instruction="Verify that this is a NeighborhoodScout page showing the Total Crime Index for the specified city/town. Allow for reasonable numerical variations (e.g., rounding)."
    )


async def verify_single_station(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        station_info: SingleStationInfo,
        station_index: int,
) -> None:
    """
    Verify all aspects of a single station.
    Uses parallel aggregation since address and crime checks are independent.
    """
    station_desc = f"Station {station_index + 1}: {station_info.station_name or 'Not found'}"

    # Create main station node (non-critical to allow partial scoring across stations)
    station_node = evaluator.add_parallel(
        id=f"station_{station_index}",
        desc=f"Complete verification for {station_desc} - location, address, and crime index",
        parent=parent_node,
        critical=False  # Non-critical - want partial credit across stations
    )

    # Add a general existence check for the station
    has_station = bool(station_info.station_name and station_info.station_name.strip())
    station_exists_node = evaluator.add_custom_node(
        result=has_station,
        id=f"station_{station_index}_exists",
        desc=f"Station {station_index + 1} found in answer: {station_info.station_name or 'Not found'}",
        parent=station_node,
        critical=True  # Critical - if no station found, skip all verifications
    )

    # Create sub-verification nodes for different aspects directly under station_node
    # Perform verifications
    await verify_station_existence_and_location(evaluator, station_node, station_info, station_index)
    await verify_station_address(evaluator, station_node, station_info, station_index)
    await verify_station_crime_index(evaluator, station_node, station_info, station_index)


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: LLMClient,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Main evaluation function for Baltimore MARC station task.

    Evaluates whether the answer correctly identifies MARC stations between Camden and Union Station
    with proper location, address, and crime index information.
    """

    # -------- 1. Initialize evaluator ------------------------------------ #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel - want partial credit across stations
        agent_name=agent_name,
        answer_name=answer_name,
        # Evaluator creation parameters
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # -------- 2. Add ground truth information ----------------------------- #
    evaluator.add_ground_truth(GROUND_TRUTH, "expected_marc_stations")

    # -------- 3. Extract station names ----------------------------------- #
    station_names_info = await evaluator.extract(
        prompt=prompt_extract_station_names(),
        template_class=StationNamesList,
        extraction_name="found_stations",
        source=None  # Extract from answer
    )

    # -------- 4. Process stations (take first N, pad if needed) ---------- #
    max_expected_stations = len(EXPECTED_STATIONS)
    found_stations = station_names_info.station_names[:max_expected_stations]

    # Create empty SingleStationInfo objects for missing stations
    station_infos = []
    for i, station_name in enumerate(found_stations):
        if station_name:
            # Extract detailed information for this station
            station_info = await evaluator.extract(
                prompt=prompt_extract_single_station_info(station_name),
                template_class=SingleStationInfo,
                extraction_name=f"station_{i}_details",
                source=None  # Extract from answer
            )
        else:
            # Create empty object for missing station
            station_info = SingleStationInfo()
        station_infos.append(station_info)
    
    # Pad with empty objects for missing stations
    while len(station_infos) < max_expected_stations:
        station_infos.append(SingleStationInfo())

    # Record extraction summary
    evaluator.add_custom_info(
        {
            "total_found": len(station_names_info.station_names),
            "expected_count": max_expected_stations,
            "found_stations": station_names_info.station_names,
            "evaluated_count": len(station_infos)
        },
        "station_extraction_summary"
    )

    # -------- 5. Verify each station position ----------------------------- #
    for i, station_info in enumerate(station_infos):
        await verify_single_station(evaluator, root, station_info, i)

    # -------- 6. Return evaluation results -------------------------------- #
    return evaluator.get_summary()