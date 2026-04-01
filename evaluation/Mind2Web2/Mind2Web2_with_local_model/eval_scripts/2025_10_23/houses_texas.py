import asyncio
import logging
from typing import Optional, List, Dict, Any
import re

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient
from mind2web2.api_tools import tool_googlemap

TASK_ID = "houses_texas"
TASK_DESCRIPTION = """
I am planning to move to San Antonio, TX, with my family. Identify 5 single-family houses currently for sale, priced between $250,000 and $400,000 USD, listed on Zillow. Each house must be within a 5-mile driving distance of an elementary school. For each house, provide a direct Zillow link for the house (not the search result page), the house's address, the asking price, and the name and address of the corresponding elementary school.
"""

EVAL_NOTES = ""
GROUND_TRUTH = {}

# Constants for the task
REQUIRED_HOUSES = 5
MIN_PRICE = 249000
MAX_PRICE = 401000
MAX_DISTANCE_MILES = 5.4
METERS_PER_MILE = 1609.34


class HouseAddressList(BaseModel):
    """List of house addresses extracted from the answer"""
    addresses: List[str] = Field(default_factory=list, description="List of house addresses in order they appear")


class HouseDetails(BaseModel):
    """Detailed information for a single house"""
    address: str = Field(description="The house address")
    zillow_url: Optional[str] = Field(default=None, description="Direct Zillow URL for the house")
    price: Optional[str] = Field(default=None, description="Asking price of the house")
    school_name: Optional[str] = Field(default=None, description="Name of the elementary school")
    school_address: Optional[str] = Field(default=None, description="Address of the elementary school")
    urls: List[str] = Field(default_factory=list, description="All URLs associated with this house")


def prompt_extract_addresses() -> str:
    """First extraction: just get the list of house addresses"""
    return """
    Extract ONLY the street addresses of single-family houses for sale in San Antonio, TX from the answer.

    Instructions:
    - Extract complete street addresses in the order they appear
    - Include only the house addresses (NOT school addresses)
    - Extract all house addresses mentioned, even if more than 5
    - Each address should be a complete street address (e.g., "123 Main St, San Antonio, TX 78201")

    Return a list of addresses exactly as they appear in the text.
    """


def prompt_extract_house_details(address: str) -> str:
    """Second extraction: get all details for a specific house address"""
    return f"""
    Extract ALL information about the house at the following address from the answer:
    Address: {address}

    Look for and extract:
    - address: The exact address (should match "{address}")
    - zillow_url: The direct Zillow link for this specific house (not a search results page)
    - price: The asking price exactly as written (e.g., "$350,000", "350K", etc.)
    - school_name: The name of the elementary school associated with this house
    - school_address: The complete address of the elementary school
    - urls: ALL URLs mentioned in connection with this house (including Zillow link and any others)

    Important:
    - Extract information ONLY for the house at {address}
    - If multiple URLs are mentioned for this house, include all of them in the urls field
    - Return null for any field not found
    - Be precise - only extract information clearly associated with this specific address
    """


async def verify_house(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        house: HouseDetails,
        house_index: int,
        gmaps_tool: tool_googlemap.GoogleMapsTool,
) -> None:
    """
    Verify a single house meets all requirements

    Args:
        evaluator: The evaluator instance
        parent_node: Parent node to attach verifications to
        house: House details to verify
        house_index: 0-based index of the house
        gmaps_tool: Google Maps API tool for distance verification
    """
    house_num = house_index + 1  # 1-based for display

    # Create container node for this house (non-critical to allow partial scoring)
    house_node = evaluator.add_parallel(
        id=f"house_{house_num}",
        desc=f"House #{house_num} verification",
        parent=parent_node,
        critical=False  # Non-critical to allow partial scoring across houses
    )

    # 1. Single existence check for all required fields
    all_fields_exist = bool(
        house.zillow_url and house.zillow_url.strip() and
        house.address and house.address.strip() and
        house.school_address and house.school_address.strip() and
        house.school_name and house.school_name.strip()
    )

    existence_node = evaluator.add_custom_node(
        result=all_fields_exist,
        id=f"house_{house_num}_all_fields_exist",
        desc=f"House #{house_num} has all required fields (Zillow URL, address, school name, school address)",
        parent=house_node,
        critical=True  # Critical - if any field missing, fail this house
    )

    # 2. Verify Zillow URL (check it's a Zillow page for this address with price in range and single-family)
    zillow_node = evaluator.add_leaf(
        id=f"house_{house_num}_zillow_verification",
        desc=f"House #{house_num} Zillow page verification",
        parent=house_node,
        critical=True
    )

    # if house.zillow_url:
    claim = (
        f"The URL '{house.zillow_url}' is a Zillow listing page for a single-family house "
        f"at the address '{house.address}' with an asking price between $250,000 and $400,000"
    )

    await evaluator.verify(
        claim=claim,
        node=zillow_node,
        sources=house.zillow_url,  # Single URL verification
        additional_instruction=(
            "Verify ALL of the following:\n"
            "1. This is a Zillow.com listing page (not search results)\n"
            "2. The address on the page matches the claimed address\n"
            "3. The asking price is shown and falls between $250,000-$400,000\n"
            "4. Based on the description and/or photos, this is a single-family house (not condo, townhouse, etc.)"
        )
    )

    price_node = evaluator.add_leaf(
        id=f"house_{house_num}_price",
        desc=f"House #{house_num} has the accurate asking price",
        parent=house_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"The house at '{house.address}' has a selling price {house.price} according to the Zillow page.",
        node=price_node,
        sources=house.zillow_url
    )

    # 3. Verify address is in San Antonio
    san_antonio_node = evaluator.add_leaf(
        id=f"house_{house_num}_san_antonio",
        desc=f"House #{house_num} is located in San Antonio, TX",
        parent=house_node,
        critical=True
    )

    # First try simple verification
    claim = f"The address '{house.address}' is located in San Antonio, Texas"

    # Check if address explicitly mentions San Antonio
    # if "san antonio" in house.address.lower() or ", sa " in house.address.lower():
    #     # Address explicitly mentions San Antonio, use simple verify
    #     await evaluator.verify(
    #         claim=claim,
    #         node=san_antonio_node,
    #         sources=None,
    #         additional_instruction="Verify the address is in San Antonio, TX based on the address string"
    #     )
    # else:
        # Address doesn't explicitly mention San Antonio, use Google Maps to verify
        # try:
    try:
        address_info = await gmaps_tool.get_address_information(house.address)
    except Exception as e:
        address_info = ""
        print(e)
        # API call failed, use simple verification


    if address_info and len(address_info) > 0:
        # Check if any component indicates San Antonio
        is_san_antonio = False
        for component in address_info[0].get('address_components', []):
            if 'locality' in component.get('types', []):
                if component.get('long_name', '').lower() == 'san antonio':
                    is_san_antonio = True
                    break

        evaluator.add_custom_node(
            result=is_san_antonio,
            id=f"house_{house_num}_gmaps_san_antonio",
            desc=f"Google Maps confirms house #{house_num} is in San Antonio",
            parent=san_antonio_node,
            critical=True
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"house_{house_num}_gmaps_san_antonio",
            desc=f"Google Maps fails to confirm house #{house_num}: {address_info}",
            parent=san_antonio_node,
            critical=True
        )

        # else:
        #     # Fallback to simple verification if API fails
        #     await evaluator.verify(
        #         claim=claim,
        #         node=san_antonio_node,
        #         sources=None,
        #         additional_instruction="Verify if this address would be in San Antonio, TX area"
        #     )
        # except Exception as e:
        #     # API failed, use simple verification
        #     await evaluator.verify(
        #         claim=claim,
        #         node=san_antonio_node,
        #         sources=None,
        #         additional_instruction="Verify if this address would be in San Antonio, TX area"
        #     )

    # 4. Verify school information via URLs
    school_verification_node = evaluator.add_leaf(
        id=f"house_{house_num}_school_verification",
        desc=f"House #{house_num} school information verification",
        parent=house_node,
        critical=True
    )

    # if house.urls and house.school_name and house.school_address:
    claim = (
        f"The elementary school '{house.school_name}' is located at "
        f"'{house.school_address}' as indicated by the provided page"
    )

    await evaluator.verify(
        claim=claim,
        node=school_verification_node,
        sources=house.urls,  # Check all URLs
        )
    # else:
    #     school_verification_node.score = 0.0
    #     school_verification_node.status = "failed"

    # 5. Verify distance using Google Maps API
    distance_node = evaluator.add_leaf(
        id=f"house_{house_num}_distance",
        desc=f"House #{house_num} is within 5-mile driving distance of the elementary school",
        parent=house_node,
        critical=True
    )

    # if house.address and house.school_address:
    try:
        # Calculate driving distance
        distance_meters = await gmaps_tool.calculate_distance(
            house.address,
            house.school_address,
            mode="driving"
        )

        if isinstance(distance_meters, int):
            distance_miles = distance_meters / METERS_PER_MILE
            within_range = distance_miles <= MAX_DISTANCE_MILES

            evaluator.add_custom_node(
                result=within_range,
                id=f"house_{house_num}_distance_check",
                desc=f"House #{house_num} driving distance: {distance_miles:.1f} miles (limit: {MAX_DISTANCE_MILES} miles)",
                parent=distance_node,
                critical=True
            )
        else:
            # API returned error message
            distance_node.score = 0.0
            distance_node.status = "failed"
            evaluator.add_custom_node(
                result=False,
                id=f"house_{house_num}_distance_api_error",
                desc=f"Google Maps API error: {distance_meters}",
                parent=distance_node,
                critical=True
            )
    except Exception as e:
        # API call failed completely
        distance_node.score = 0.0
        distance_node.status = "failed"
        evaluator.add_custom_node(
            result=False,
            id=f"house_{house_num}_distance_exception",
            desc=f"Distance calculation failed: {str(e)}",
            parent=distance_node,
            critical=True
        )
    else:
        distance_node.score = 0.0
        distance_node.status = "skipped"


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                               #
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
    Main evaluation function for the houses_texas task

    This function:
    1. Initializes the evaluator
    2. First extracts just the house addresses
    3. Then extracts detailed information for each address
    4. Verifies each house meets all requirements
    5. Returns the evaluation summary
    """

    # -------- 1. Initialize evaluator ----------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Houses are evaluated in parallel
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

    # -------- 2. Extract house addresses first -------------------- #
    address_list = await evaluator.extract(
        prompt=prompt_extract_addresses(),
        template_class=HouseAddressList,
        extraction_name="address_extraction",
    )

    # -------- 3. Extract details for each address ---------------- #
    houses_details = []
    addresses_to_process = address_list.addresses[:REQUIRED_HOUSES]  # Only process first 5

    for i, address in enumerate(addresses_to_process):
        house_details = await evaluator.extract(
            prompt=prompt_extract_house_details(address),
            template_class=HouseDetails,
            extraction_name=f"house_{i + 1}_details",
        )
        houses_details.append(house_details)

    # Pad with empty houses if fewer than required
    while len(houses_details) < REQUIRED_HOUSES:
        empty_house = HouseDetails(address="")
        houses_details.append(empty_house)

    # Record extraction statistics
    evaluator.add_custom_info(
        {
            "total_addresses_found": len(address_list.addresses),
            "addresses_processed": len(addresses_to_process),
            "required_houses": REQUIRED_HOUSES,
        },
        "extraction_statistics"
    )

    # -------- 4. Build verification tree -------------------------- #
    for i, house in enumerate(houses_details):
        await verify_house(evaluator, root, house, i, gmaps_tool)

    # -------- 5. Return evaluation results ------------------------ #
    return evaluator.get_summary()