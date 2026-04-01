import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient
from mind2web2.api_tools import tool_googlemap

TASK_ID = "find_parks"
TASK_DESCRIPTION = """
I am planning a trip from Dayton, Ohio to Notre Dame, Indiana, and I wish to visit a state park during the trip. Identify three official state parks (designated and managed by state authorities) for which the total driving distance from Dayton to Notre Dame via the state park is less than 260 miles. For each park, provide the name of the official state park, the park's physical address, a direct link to the park's page on the respective state's official website, and the total driving distance of the trip (Dayton → park → Notre Dame).
"""

EVAL_NOTES = ""
GROUND_TRUTH = {}

# Constants
START_LOCATION = "Dayton, Ohio"
END_LOCATION = "Notre Dame, Indiana"
MAX_DISTANCE_MILES = 260
METERS_PER_MILE = 1609.34


class ParkNames(BaseModel):
    """List of park names extracted from the answer"""
    park_names: List[str] = Field(default_factory=list, description="List of state park names mentioned in the answer")


class SingleParkInfo(BaseModel):
    """Information for a single state park"""
    address: Optional[str] = Field(default=None, description="Physical address of the park")
    official_url: Optional[str] = Field(default=None, description="Link to park page on state website")
    total_distance: Optional[str] = Field(default=None, description="Total driving distance for the trip")


def prompt_extract_park_names() -> str:
    """Extraction prompt for getting park names from the answer"""
    return """
    Extract the names of state parks mentioned in the answer.

    Look for:
    - Names of official state parks that are explicitly mentioned as state parks
    - Parks that are described as being designated or managed by state authorities

    Extract the park names exactly as they appear in the text, in the order they are presented.
    Return a list of park names. If no parks are mentioned, return an empty list.
    """


def prompt_extract_single_park_info(park_name: str) -> str:
    """Extraction prompt for getting information about a specific park"""
    return f"""
    Extract the following information specifically for the state park named "{park_name}" from the answer:

    - address: The physical address or location of this specific park
    - official_url: The direct link to this park's page on the state's official website
    - total_distance: The total driving distance for the trip (Dayton → this park → Notre Dame)

    IMPORTANT: Only extract information that is explicitly associated with "{park_name}" in the answer.
    Extract information exactly as it appears in the text.
    If any field is not mentioned for this specific park, set it to null.
    """


def prompt_verify_state_park_designation(park_name: str) -> str:
    """Verification prompt to check if park is indeed a state park"""
    return f"""
    Based on the webpage content, verify whether "{park_name}" is indeed an official state park 
    that is designated and managed by state authorities.

    Look for indications such as:
    - Explicit mention that it's a "State Park"
    - References to state government management or designation
    - Official state park logos or headers
    - Being part of the state park system

    The park must be officially designated as a state park to be considered valid.
    """


def prompt_verify_park_address(park_name: str, address: str) -> str:
    """Verification prompt to check if the park is at the claimed address"""
    return f"""
    Based on the webpage content, verify whether the state park "{park_name}" 
    is indeed located at or near the address: {address}

    Look for:
    - Address information on the official park page
    - Location descriptions that match the provided address
    - Maps or directions that confirm the location

    The address should reasonably match what's on the official website.
    """


async def verify_single_park(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        park_name: str,
        park_info: SingleParkInfo,
        park_index: int,
        gmaps_tool: tool_googlemap.GoogleMapsTool,
) -> None:
    """Verify all information for a single park"""

    # Create park-level node
    park_node = evaluator.add_parallel(
        id=f"park_{park_index}",
        desc=f"Park {park_index + 1}: {park_name or 'Missing'}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial scoring
    )

    # 1. Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(park_name and park_name.strip() and
                    park_info.address and park_info.address.strip() and
                    park_info.official_url and park_info.official_url.strip() and
                    park_info.total_distance and park_info.total_distance.strip()),
        id=f"park_{park_index}_info_exists",
        desc=f"All required information exists for park {park_index + 1}",
        parent=park_node,
        critical=True,
    )


    # 2. Verify it's a state park from official website
    state_park_node = evaluator.add_leaf(
        id=f"park_{park_index}_is_state_park",
        desc=f"{park_name or 'Park'} is confirmed as official state park",
        parent=park_node,
        critical=True,
    )

    # if park_name and park_info.official_url:
    await evaluator.verify(
        claim=prompt_verify_state_park_designation(park_name),
        node=state_park_node,
        sources=park_info.official_url,
    )

    # 3. Verify park address from official website
    address_node = evaluator.add_leaf(
        id=f"park_{park_index}_address_verified",
        desc=f"{park_name or 'Park'} is at claimed address",
        parent=park_node,
        critical=True,
    )

    # if park_name and park_info.address and park_info.official_url:
    await evaluator.verify(
        claim=prompt_verify_park_address(park_name, park_info.address),
        node=address_node,
        sources=park_info.official_url,
        additional_instruction="Verify the park is located at or near the provided address"
    )

    # 4. Verify distance constraint using Google Maps API
    if park_info.address:
        try:
            # Calculate Dayton to Park distance
            distance1_meters = await gmaps_tool.calculate_distance(
                START_LOCATION,
                park_info.address,
                mode="driving"
            )

            # Calculate Park to Notre Dame distance
            distance2_meters = await gmaps_tool.calculate_distance(
                park_info.address,
                END_LOCATION,
                mode="driving"
            )

            if isinstance(distance1_meters, int) and isinstance(distance2_meters, int):
                total_meters = distance1_meters + distance2_meters
                total_miles = total_meters / METERS_PER_MILE

                # Add custom node with the actual verification
                evaluator.add_custom_node(
                    result=(total_miles < MAX_DISTANCE_MILES),
                    id=f"park_{park_index}_distance_valid",
                    desc=f"Total distance {total_miles:.1f} miles is under {MAX_DISTANCE_MILES} miles limit",
                    parent=park_node,
                    critical=True,
                )

                distance_match_node = evaluator.add_leaf(
                    id=f"park_{park_index}_distance_match",
                    desc=f"Claimed total distance {park_info.total_distance} is matched with the actual total distance {total_miles:.1f}",
                    parent=park_node,
                    critical=True
                )

                await evaluator.verify(
                    claim=f"The claimed distance {park_info.total_distance} is approximately equal to {total_miles:.1f}.",
                    node=distance_match_node,
                    additional_instruction=f"A tolerance of ±10 miles around {total_miles:.1f} is permitted."
                )

            else:
                # Google Maps API failed - add failed node
                evaluator.add_custom_node(
                    result=False,
                    id=f"park_{park_index}_distance_valid",
                    desc="Failed to calculate distance via Google Maps API",
                    parent=park_node,
                    critical=True,
                )

                evaluator.add_custom_node(
                    result=False,
                    id=f"park_{park_index}_distance_match",
                    desc="Failed to calculate distance via Google Maps API",
                    parent=park_node,
                    critical=True,
                )


        except Exception as e:
            # Error in distance calculation - add failed node
            evaluator.add_custom_node(
                result=False,
                id=f"park_{park_index}_distance_valid",
                desc=f"Error calculating distance: {str(e)}",
                parent=park_node,
                critical=True,
            )

            evaluator.add_custom_node(
                    result=False,
                    id=f"park_{park_index}_distance_match",
                    desc=f"Error calculating distance: {str(e)}",
                    parent=park_node,
                    critical=True,
                )
    else:
        # No address to verify distance - add failed node
        evaluator.add_custom_node(
            result=False,
            id=f"park_{park_index}_distance_valid",
            desc="Cannot verify distance without park address",
            parent=park_node,
            critical=True,
        )

        evaluator.add_custom_node(
            result=False,
            id=f"park_{park_index}_distance_match",
            desc="Cannot verify distance without park address",
            parent=park_node,
            critical=True,
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
    """Main evaluation function for state parks task"""

    # Initialize evaluator
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

    # Initialize Google Maps tool
    gmaps_tool = tool_googlemap.GoogleMapsTool()

    # Step 1: Extract park names
    park_names_result = await evaluator.extract(
        prompt=prompt_extract_park_names(),
        template_class=ParkNames,
        extraction_name="park_names_extraction",
    )

    # Get up to 3 park names
    park_names = park_names_result.park_names[:3] if park_names_result.park_names else []

    # Add empty placeholders if less than 3 parks
    while len(park_names) < 3:
        park_names.append("")

    # Step 2 & 3: For each park, extract details and verify
    for i, park_name in enumerate(park_names):
        if park_name:  # Only extract details if park name exists
            # Extract details for this specific park
            park_info = await evaluator.extract(
                prompt=prompt_extract_single_park_info(park_name),
                template_class=SingleParkInfo,
                extraction_name=f"park_{i + 1}_details",
            )
        else:
            # Create empty park info for placeholder
            park_info = SingleParkInfo()

        # Verify this park
        await verify_single_park(
            evaluator=evaluator,
            parent_node=root,
            park_name=park_name,
            park_info=park_info,
            park_index=i,
            gmaps_tool=gmaps_tool,
        )

    # Return evaluation results
    return evaluator.get_summary()