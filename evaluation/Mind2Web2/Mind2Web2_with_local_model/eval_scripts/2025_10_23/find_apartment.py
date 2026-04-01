import asyncio
import logging
from typing import Optional, List, Dict
import re

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.api_tools import tool_googlemap

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "find_apartment"
TASK_DESCRIPTION = """
I'm looking for a one-bedroom apartment near the Michigan League in Ann Arbor, MI, where I'll be living alone. Please find three apartment buildings within 10 miles driving distance of this location, each explicitly offering on-site parking (any form clearly stated on the listing page).

Each apartment must list at least one one-bedroom unit with a total monthly rent of no more than $1,500. If multiple prices are listed (such as a range or different lease terms) for a unit, use the highest monthly rent shown. If pricing is per-person, assume single occupancy. Exclude listings without clearly stated monthly rent.

For each apartment, provide a direct link to its Zillow or Apartments.com listing, its full address, and the monthly rent of one qualifying one-bedroom unit exactly as listed.
"""

# Michigan League address for distance calculation
MICHIGAN_LEAGUE_ADDRESS = "911 N University Ave, Ann Arbor, MI 48109"


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class ApartmentNames(BaseModel):
    """Container for apartment names extracted from the answer."""
    names: List[str] = Field(default_factory=list)


class ApartmentDetails(BaseModel):
    """Detailed information for a specific apartment."""
    address: Optional[str] = Field(default=None)
    monthly_rent: Optional[str] = Field(default=None)
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_apartment_names() -> str:
    return """
    Extract the names of all apartment buildings/complexes mentioned in the answer.
    Return only the names, without addresses or other details.
    If a name is not explicitly provided, use a brief identifying description.
    """


def prompt_extract_apartment_details(apartment_name: str) -> str:
    return f"""
    For the apartment "{apartment_name}", extract the following information:
    - address: The full address exactly as stated
    - monthly_rent: The monthly rent amount exactly as written in the answer, including any floor plan information or details (preserve original formatting)
    - urls: ALL URLs that might be related to this apartment (listing URLs, building websites, etc.)

    If any field is not available, set it to null or empty list for urls.
    """


# --------------------------------------------------------------------------- #
# Verification functions for individual apartments                            #
# --------------------------------------------------------------------------- #
async def verify_apartment_all_requirements(
        evaluator: Evaluator,
        apartment_name: str,
        apartment_details: ApartmentDetails,
        apartment_index: int,
        gmaps_tool: tool_googlemap.GoogleMapsTool,
) -> None:
    """
    Verify all requirements for a single apartment listing.
    All checks are marked as critical.
    """
    apt_node = evaluator.add_parallel(
        id=f"apartment_{apartment_index + 1}",
        desc=f"Apartment {apartment_index + 1} ({apartment_name}) meets all specified requirements",
        critical=False  # Allow partial scoring across apartments
    )

    # Verify all requirements (all critical)
    await verify_basic_info_provided(evaluator, apt_node, apartment_details, apartment_index + 1)
    await verify_platform_listing(evaluator, apt_node, apartment_name, apartment_details.urls, apartment_index + 1)
    await verify_address_substantiation(evaluator, apt_node, apartment_details, apartment_index + 1)
    await verify_parking_availability(evaluator, apt_node, apartment_details.urls, apartment_index + 1)
    await verify_rent_substantiation(evaluator, apt_node, apartment_details, apartment_index + 1)
    await verify_distance_requirement(evaluator, apt_node, apartment_details, gmaps_tool, apartment_index + 1)


async def verify_basic_info_provided(
        evaluator: Evaluator,
        parent_node,
        apartment_details: ApartmentDetails,
        apartment_num: int,
) -> None:
    """Verify that basic required information is provided."""

    has_all_info = bool(
        apartment_details.address and
        apartment_details.monthly_rent and
        apartment_details.urls
    )

    basic_info_node = evaluator.add_custom_node(
        result=has_all_info,
        id=f"apt_{apartment_num}_basic_info",
        desc="Required basic information (rent, address, URLs) is provided",
        parent=parent_node,
        critical=True,
    )


async def verify_platform_listing(
        evaluator: Evaluator,
        parent_node,
        apartment_name: str,
        urls: List[str],
        apartment_num: int,
) -> None:
    """Verify that there's a direct Zillow or Apartments.com page for this apartment."""
    platform_node = evaluator.add_leaf(
        id=f"apt_{apartment_num}_platform",
        desc="Has a direct Zillow or Apartments.com page for this apartment",
        parent=parent_node,
        critical=True,
    )

    claim = f"One of these pages is a direct Zillow or Apartments.com listing page specifically for the apartment '{apartment_name}'"
    await evaluator.verify(
        claim=claim,
        node=platform_node,
        sources=urls,
        additional_instruction="Look for pages on zillow.com or apartments.com domains that show details for this specific apartment building/complex"
    )


async def verify_address_substantiation(
        evaluator: Evaluator,
        parent_node,
        apartment_details: ApartmentDetails,
        apartment_num: int,
) -> None:
    """Verify that the apartment address is substantiated by listing pages."""
    address_node = evaluator.add_leaf(
        id=f"apt_{apartment_num}_address_substantiation",
        desc="Apartment address is substantiated by listing pages",
        parent=parent_node,
        critical=True,
    )

    claim = f"The apartment address '{apartment_details.address}' is confirmed and matches the information on the listing pages"
    await evaluator.verify(
        claim=claim,
        node=address_node,
        sources=apartment_details.urls,
        additional_instruction="Look for the specific address or location information that matches the stated address"
    )


async def verify_parking_availability(
        evaluator: Evaluator,
        parent_node,
        urls: List[str],
        apartment_num: int,
) -> None:
    """Verify that on-site parking is explicitly mentioned."""
    parking_node = evaluator.add_leaf(
        id=f"apt_{apartment_num}_parking",
        desc="On-site parking is explicitly offered (any form)",
        parent=parent_node,
        critical=True,
    )

    claim = "This apartment building explicitly offers on-site parking in any form (parking garage, parking lot, assigned parking, etc.)"
    await evaluator.verify(
        claim=claim,
        node=parking_node,
        sources=urls,
        additional_instruction="Look for any indication of parking availability, including: parking facilities, parking amenities, garage, parking spots, or similar terms. Additionally, the presence of a 'Parking' tab, button, or section anywhere on the page indicates parking is available, even if that section appears collapsed or does not show detailed content. Any parking-related element or mention should be considered valid evidence of on-site parking."
    )

async def verify_rent_substantiation(
        evaluator: Evaluator,
        parent_node,
        apartment_details: ApartmentDetails,
        apartment_num: int,
) -> None:
    """Verify that 1BR monthly rent is substantiated and within budget."""
    rent_node = evaluator.add_leaf(
        id=f"apt_{apartment_num}_rent_substantiation",
        desc="1BR monthly rent is substantiated by listing pages and ≤ $1,500",
        parent=parent_node,
        critical=True,
    )
    
    claim = f"The listing pages show a 1-bedroom unit or floor plan with monthly rent of {apartment_details.monthly_rent} (if multiple floor plans exist, any 1BR unit matching this rent is sufficient; if multiple prices/ranges are shown for a unit/floor plan, the highest price should match)"
    await evaluator.verify(
        claim=claim,
        node=rent_node,
        sources=apartment_details.urls,
        additional_instruction="Look for 1-bedroom (1BR, or 1B1B, or similar) unit pricing that exactly matches the stated rent amount. The rent should be ≤ $1,500. If ranges are shown, use the highest price."
    )


async def verify_distance_requirement(
        evaluator: Evaluator,
        parent_node,
        apartment_details: ApartmentDetails,
        gmaps_tool: tool_googlemap.GoogleMapsTool,
        apartment_num: int,
) -> None:
    """Verify that apartment is within 10.5 miles driving distance of Michigan League."""
    distance_node = evaluator.add_leaf(
        id=f"apt_{apartment_num}_distance",
        desc="Within 10.5 miles driving distance of Michigan League",
        parent=parent_node,
        critical=True,
    )

    try:
        # Get driving distance in meters
        distance_meters = await gmaps_tool.calculate_distance(
            apartment_details.address,
            MICHIGAN_LEAGUE_ADDRESS,
            mode="driving"
        )

        if isinstance(distance_meters, int):
            # Convert meters to miles
            distance_miles = distance_meters / 1609.34

            # Use simple_verify to check if within 10.5 miles
            claim = f"The driving distance of {distance_miles:.2f} miles is within 10.5 miles (allowing for formatting differences and measurement tolerances)"
            await evaluator.verify(
                claim=claim,
                node=distance_node,
                additional_instruction="Consider that 10.5 miles or less is acceptable, accounting for potential formatting differences and measurement tolerances"
            )
        else:
            distance_node.score = 0.0
            distance_node.status = "failed"
    except Exception as e:
        distance_node.score = 0.0
        distance_node.status = "failed"


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Initialize evaluator -------- #
    evaluator = Evaluator()
    evaluator.initialize(
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
        default_model=model
    )

    # -------- 2. Set up Google Maps tool -------- #
    gmaps_tool = tool_googlemap.GoogleMapsTool()

    # -------- 3. Extract apartment names first -------- #
    apartment_names = await evaluator.extract(
        prompt=prompt_extract_apartment_names(),
        template_class=ApartmentNames,
        extraction_name="apartment_names"
    )

    # -------- 4. Process exactly 3 apartments (as requested) -------- #
    apartments_to_verify = apartment_names.names[:3]  # Take first 3 names

    # Pad the list with empty names if we have fewer than 3
    while len(apartments_to_verify) < 3:
        apartments_to_verify.append("")

    # Extract details for each apartment and verify
    for i in range(3):
        apartment_name = apartments_to_verify[i] if apartments_to_verify[i] else f"Apartment {i + 1} (Missing)"

        if apartments_to_verify[i]:  # Only extract if we have a name
            # Extract detailed information for this apartment
            apartment_details = await evaluator.extract(
                prompt=prompt_extract_apartment_details(apartment_name),
                template_class=ApartmentDetails,
                extraction_name=f"apartment_{i + 1}_details"
            )
        else:
            # Create empty details for missing apartment
            apartment_details = ApartmentDetails()
        
        # Verify this apartment (will handle empty data through existence checks)
        await verify_apartment_all_requirements(
            evaluator=evaluator,
            apartment_name=apartment_name,
            apartment_details=apartment_details,
            apartment_index=i,
            gmaps_tool=gmaps_tool,
        )

    # -------- 5. Add custom info -------- #
    evaluator.add_custom_info({
        "total_apartment_names_found": len(apartment_names.names),
        "apartments_processed": min(3, len(apartment_names.names)),
        "michigan_league_address": MICHIGAN_LEAGUE_ADDRESS,
    }, "task_statistics")

    # -------- 6. Return structured result -------- #
    return evaluator.get_summary()