import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient
from mind2web2.api_tools import tool_googlemap

TASK_ID = "la_visit_hotel"
TASK_DESCRIPTION = """
I am visiting a friend in Los Angeles and need a hotel for a three-day stay. I want to visit USC, UCLA, and Griffith Observatory, as well as Mission San Gabriel Arcángel in San Gabriel and Seoul International Park in Koreatown. I prefer accommodations within a 15-miles driving distance of all these locations to minimize travel time.

Please provide two suitable hotel options along with their name, complete physical addresses, and driving distance (in miles) to all the locations mentioned above.
"""

EVAL_NOTES = """
https://www.google.com/maps/d/u/0/edit?hl=en&mid=1oyl5bpPuMDvLPU-NLIrZxW7SnDglxKM&ll=34.105728622506405%2C-118.24389137257221&z=11
"""

GROUND_TRUTH = {}  # No ground truth provided

# Define the 5 locations to check distances to
LOCATIONS = {
    "USC": "University of Southern California, Los Angeles, CA",
    "UCLA": "University of California Los Angeles, Los Angeles, CA",
    "Griffith Observatory": "Griffith Observatory, Los Angeles, CA",
    "Mission San Gabriel Arcángel": "Mission San Gabriel Arcángel, San Gabriel, CA",
    "Seoul International Park": "Seoul International Park, Koreatown, Los Angeles, CA"
}

DISTANCE_LIMIT_MILES = 15.0
DISTANCE_TOLERANCE_MILES = 0.5  # Tolerance for distance verification
REPORTED_DISTANCE_TOLERANCE_MILES = 1.0  # Tolerance for comparing with reported distance


class HotelNames(BaseModel):
    """Just the hotel names"""
    hotel_names: List[str] = Field(default_factory=list, description="List of hotel names")


class SingleHotelInfo(BaseModel):
    """Detailed information for a single hotel"""
    name: Optional[str] = Field(default=None, description="Hotel name")
    address: Optional[str] = Field(default=None, description="Complete physical address")
    distance_to_usc: Optional[str] = Field(default=None, description="Driving distance to USC in miles")
    distance_to_ucla: Optional[str] = Field(default=None, description="Driving distance to UCLA in miles")
    distance_to_griffith: Optional[str] = Field(default=None,
                                                description="Driving distance to Griffith Observatory in miles")
    distance_to_mission: Optional[str] = Field(default=None,
                                               description="Driving distance to Mission San Gabriel Arcángel in miles")
    distance_to_seoul_park: Optional[str] = Field(default=None,
                                                  description="Driving distance to Seoul International Park in miles")
    urls: List[str] = Field(default_factory=list, description="All URLs related to this hotel")


def prompt_extract_hotel_names() -> str:
    """First extraction: just get hotel names"""
    return """
    Extract ONLY the names of the hotels recommended in the answer.

    List all hotel names mentioned, in the order they appear.
    Extract only the hotel names, nothing else.
    """


def prompt_extract_single_hotel_info(hotel_name: str) -> str:
    """Extract detailed information for a specific hotel"""
    return f"""
    Extract detailed information for the hotel named "{hotel_name}" from the answer.

    Extract:
    1. name: The exact hotel name (should match "{hotel_name}")
    2. address: The complete physical address including street number, street name, city, state, and zip if available
    3. distance_to_usc: The driving distance to USC in miles (extract as string, e.g., "12.5 miles" or "12.5")
    4. distance_to_ucla: The driving distance to UCLA in miles
    5. distance_to_griffith: The driving distance to Griffith Observatory in miles
    6. distance_to_mission: The driving distance to Mission San Gabriel Arcángel in miles
    7. distance_to_seoul_park: The driving distance to Seoul International Park in miles
    8. urls: ALL URLs that are related to this specific hotel

    Extract information exactly as it appears in the text. If any information is missing, set it to null.
    For distances, extract the numerical value with or without "miles" unit.
    """


def parse_distance(distance_str: Optional[str]) -> Optional[float]:
    """Parse distance string to float value in miles"""
    if not distance_str:
        return None

    try:
        # Remove common variations
        cleaned = distance_str.lower().replace("miles", "").replace("mile", "").replace("mi", "").replace("approx.", "").replace("approximately", "").replace("driving", "").strip().replace("~", "")
        # Handle ranges by taking the first value
        if "-" in cleaned:
            cleaned = cleaned.split("-")[0].strip()
        return float(cleaned)
    except:
        return None


async def verify_single_distance(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        hotel_name: str,
        hotel_address: str,
        location_name: str,
        location_address: str,
        reported_distance_str: Optional[str],
        node_id_suffix: str,
        gmaps_tool: tool_googlemap.GoogleMapsTool,
        continue_flag: bool = True
) -> bool:
    """Verify distance to a single location"""

    # Create verification node
    distance_node = evaluator.add_leaf(
        id=f"{node_id_suffix}_{location_name.lower().replace(' ', '_')}",
        desc=f"Distance to {location_name} is within {DISTANCE_LIMIT_MILES} miles (±{DISTANCE_TOLERANCE_MILES})",
        parent=parent_node,
        critical=True
    )

    if continue_flag is False:
        distance_node.score = 0.0
        distance_node.status = "skipped"
        return False

    if reported_distance_str is None:
        distance_node.score = 0.0
        distance_node.status = "failed"
        return False

    try:
        # Calculate actual distance using Google Maps API
        distance_meters = await gmaps_tool.calculate_distance(
            hotel_address,
            location_address,
            mode="driving"
        )

        if isinstance(distance_meters, int):
            actual_distance_miles = distance_meters * 0.000621371  # Convert meters to miles
            reported_distance_miles = parse_distance(reported_distance_str)

            # Check if within limit (with tolerance)
            within_limit = actual_distance_miles <= (DISTANCE_LIMIT_MILES + DISTANCE_TOLERANCE_MILES)

            # Check if reported distance matches actual (if reported)
            distance_match = True
            if reported_distance_miles is not None:
                distance_diff = abs(actual_distance_miles - reported_distance_miles)
                distance_match = distance_diff <= REPORTED_DISTANCE_TOLERANCE_MILES
            elif reported_distance_str.strip() != "":
                # the parse_distance has something wrong, directly use the reported_distance_str
                claim = f"Reported distance (miles) '{reported_distance_str}' match the actual distance (miles) '{actual_distance_miles:.1f} miles'."
                distance_match = await evaluator.verify(
                    claim=claim,
                    node=None,
                    additional_instruction=f"Verify whether the reported distance falls within the ±{REPORTED_DISTANCE_TOLERANCE_MILES} miles of the actual distance (i.e., between {(actual_distance_miles-REPORTED_DISTANCE_TOLERANCE_MILES):.1f} and {(actual_distance_miles+REPORTED_DISTANCE_TOLERANCE_MILES):.1f} miles)."
                )

            # Both conditions must be met
            result = within_limit and distance_match

            # Set verification result
            distance_node.score = 1.0 if result else 0.0
            distance_node.status = "passed" if result else "failed"

            # Log the verification
            evaluator.verifier.logger.info(
                f"Distance verification for {hotel_name} to {location_name}: "
                f"{'✅ PASSED' if result else '❌ FAILED'} - "
                f"Actual: {actual_distance_miles:.1f} miles, "
                f"Reported: {reported_distance_str or 'N/A'}, "
                f"Within limit: {within_limit}, Distance match: {distance_match}"
            )

        else:
            # API call failed
            distance_node.score = 0.0
            distance_node.status = "failed"
            evaluator.verifier.logger.info(
                f"Failed to calculate distance from {hotel_name} to {location_name}: {distance_meters}"
            )

    except Exception as e:
        # Handle any errors
        distance_node.score = 0.0
        distance_node.status = "failed"
        evaluator.verifier.logger.info(
            f"Error calculating distance from {hotel_name} to {location_name}: {str(e)}"
        )
    
    return bool(distance_node.status == "passed")


async def verify_single_hotel(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        hotel_name: str,
        hotel_index: int,
        gmaps_tool: tool_googlemap.GoogleMapsTool
) -> None:
    """Verify a single hotel with all requirements"""

    # Create hotel container node (non-critical for partial scoring)
    hotel_node = evaluator.add_parallel(
        id=f"hotel_{hotel_index}",
        desc=f"Hotel {hotel_index}: {hotel_name}",
        parent=parent_node,
        critical=False  # Non-critical to allow partial scoring
    )

    # Extract detailed information for this specific hotel
    hotel_info = await evaluator.extract(
        prompt=prompt_extract_single_hotel_info(hotel_name),
        template_class=SingleHotelInfo,
        extraction_name=f"hotel_{hotel_index}_details"
    )

    # Check existence of all required information
    all_info_exists = (
            bool(hotel_info.name and hotel_info.name.strip()) and
            bool(hotel_info.address and hotel_info.address.strip()) and
            hotel_info.distance_to_usc is not None and
            hotel_info.distance_to_ucla is not None and
            hotel_info.distance_to_griffith is not None and
            hotel_info.distance_to_mission is not None and
            hotel_info.distance_to_seoul_park is not None and
            len(hotel_info.urls) > 0
    )

    existence_node = evaluator.add_custom_node(
        result=all_info_exists,
        id=f"hotel_{hotel_index}_all_info_exists",
        desc="All required information exists (name, address, 5 distances, URLs)",
        parent=hotel_node,
        critical=True
    )

    # Verify hotel name and address via URLs
    name_address_node = evaluator.add_leaf(
        id=f"hotel_{hotel_index}_name_address_verified",
        desc="Hotel name and address are verified by source URLs",
        parent=hotel_node,
        critical=True
    )

    if hotel_info.urls and hotel_info.name and hotel_info.address:
        claim = f"The hotel '{hotel_info.name}' is located at the address '{hotel_info.address}'"
        await evaluator.verify(
            claim=claim,
            node=name_address_node,
            sources=hotel_info.urls,
            additional_instruction="Verify that the webpage confirms this hotel exists at the stated address"
        )
    else:
        name_address_node.score = 0.0
        name_address_node.status = "failed"

    # Create parallel container for distance verifications
    distances_node = evaluator.add_parallel(
        id=f"hotel_{hotel_index}_distances",
        desc="All driving distances are within 15 miles",
        parent=hotel_node,
        critical=True
    )

    # Verify each distance
    if hotel_info.address:
        continue_flag = await verify_single_distance(
            evaluator, distances_node, hotel_info.name or hotel_name, hotel_info.address,
            "USC", LOCATIONS["USC"], hotel_info.distance_to_usc,
            f"hotel_{hotel_index}_distance", gmaps_tool
        )

        continue_flag = await verify_single_distance(
            evaluator, distances_node, hotel_info.name or hotel_name, hotel_info.address,
            "UCLA", LOCATIONS["UCLA"], hotel_info.distance_to_ucla,
            f"hotel_{hotel_index}_distance", gmaps_tool, continue_flag
        )

        continue_flag = await verify_single_distance(
            evaluator, distances_node, hotel_info.name or hotel_name, hotel_info.address,
            "Griffith Observatory", LOCATIONS["Griffith Observatory"], hotel_info.distance_to_griffith,
            f"hotel_{hotel_index}_distance", gmaps_tool, continue_flag
        )

        continue_flag = await verify_single_distance(
            evaluator, distances_node, hotel_info.name or hotel_name, hotel_info.address,
            "Mission San Gabriel Arcángel", LOCATIONS["Mission San Gabriel Arcángel"], hotel_info.distance_to_mission,
            f"hotel_{hotel_index}_distance", gmaps_tool, continue_flag
        )

        continue_flag = await verify_single_distance(
            evaluator, distances_node, hotel_info.name or hotel_name, hotel_info.address,
            "Seoul International Park", LOCATIONS["Seoul International Park"], hotel_info.distance_to_seoul_park,
            f"hotel_{hotel_index}_distance", gmaps_tool, continue_flag
        )


async def create_placeholder_hotel(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        hotel_index: int
) -> None:
    """Create placeholder nodes for missing hotel"""

    # Hotel container node
    hotel_node = evaluator.add_parallel(
        id=f"hotel_{hotel_index}",
        desc=f"Hotel {hotel_index}: Not provided",
        parent=parent_node,
        critical=False
    )

    # All sub-nodes are skipped
    evaluator.add_leaf(
        id=f"hotel_{hotel_index}_all_info_exists",
        desc="All required information exists (name, address, 5 distances, URLs)",
        parent=hotel_node,
        critical=True,
        score=0.0,
        status="skipped"
    )

    evaluator.add_leaf(
        id=f"hotel_{hotel_index}_name_address_verified",
        desc="Hotel name and address are verified by source URLs",
        parent=hotel_node,
        critical=True,
        score=0.0,
        status="skipped"
    )

    # Distances container
    distances_node = evaluator.add_parallel(
        id=f"hotel_{hotel_index}_distances",
        desc="All driving distances are within 15 miles",
        parent=hotel_node,
        critical=True
    )

    # Add skipped distance nodes
    for location_name in LOCATIONS.keys():
        evaluator.add_leaf(
            id=f"hotel_{hotel_index}_distance_{location_name.lower().replace(' ', '_')}",
            desc=f"Distance to {location_name} is within {DISTANCE_LIMIT_MILES} miles (±{DISTANCE_TOLERANCE_MILES})",
            parent=distances_node,
            critical=True,
            score=0.0,
            status="skipped"
        )


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
    Main evaluation function for la_visit_hotel task.

    Evaluation process:
    1. First extract just hotel names
    2. For each hotel (max 2), extract detailed info
    3. Verify existence, name/address via URLs, and distances via Google Maps
    """

    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Hotels are evaluated independently
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

    # Initialize Google Maps tool
    gmaps_tool = tool_googlemap.GoogleMapsTool()

    # First, extract just hotel names
    hotel_names_info = await evaluator.extract(
        prompt=prompt_extract_hotel_names(),
        template_class=HotelNames,
        extraction_name="hotel_names_extraction"
    )

    # Take only first 2 hotels if more are provided
    hotels_to_verify = hotel_names_info.hotel_names[:2]

    # Verify each hotel
    for i in range(2):  # Always create nodes for 2 hotels
        if i < len(hotels_to_verify):
            # Verify provided hotel
            await verify_single_hotel(
                evaluator,
                root,
                hotels_to_verify[i],
                i + 1,
                gmaps_tool
            )
        else:
            # Create placeholder for missing hotel
            await create_placeholder_hotel(evaluator, root, i + 1)

    # Add custom info about evaluation
    evaluator.add_custom_info({
        "hotels_found": len(hotel_names_info.hotel_names),
        "hotels_evaluated": 2,
        "distance_limit_miles": DISTANCE_LIMIT_MILES,
        "distance_tolerance_miles": DISTANCE_TOLERANCE_MILES,
        "reported_distance_tolerance_miles": REPORTED_DISTANCE_TOLERANCE_MILES,
        "locations_checked": list(LOCATIONS.keys())
    }, "evaluation_metadata")

    # Return evaluation summary
    return evaluator.get_summary()