import asyncio
import logging
from typing import Optional, List, Dict

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.api_tools import tool_googlemap

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "find_daycare"
TASK_DESCRIPTION = """
Identify three licensed group childcare centers located specifically in Brookline, MA, within a 10-mile driving distance of 75 Peterborough Street, Boston, MA. For each childcare center, provide its official website, physical address, current Google Maps rating, and a direct link to its licensing profile on the Massachusetts Early Education and Care (EEC) website.
"""

JUDGE_MODEL = "o4-mini"

# Reference address for distance calculation
REFERENCE_ADDRESS = "75 Peterborough Street, Boston, MA"

# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class DaycareCenter(BaseModel):
    name: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    google_maps_rating: Optional[str] = None
    eec_license_link: Optional[str] = None


class ExtractedDaycares(BaseModel):
    daycares: List[DaycareCenter] = Field(default_factory=list)


class ExtractedWebsiteUrls(BaseModel):
    urls: List[str] = Field(default_factory=list)


class ExtractedAddressUrls(BaseModel):
    urls: List[str] = Field(default_factory=list)


class ExtractedRatingUrls(BaseModel):
    urls: List[str] = Field(default_factory=list)


class ExtractedLicenseUrls(BaseModel):
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_daycares() -> str:
    return """
    Extract information about daycare centers from the answer. Look for:
    1. Name of each daycare center
    2. Official website URL
    3. Physical address
    4. Google Maps rating (as a string, e.g., "4.5", "4.2 stars", etc.)
    5. EEC licensing profile link

    Extract all daycare centers mentioned in the answer, even if more than 3 are provided.
    If any field is missing for a daycare center, set it to null.
    """


def prompt_extract_website_urls(daycare_name: str) -> str:
    return f"""
    Extract all URLs from the answer that could potentially support the website information for "{daycare_name}".
    This includes:
    - Direct links to the daycare's official website
    - Links to pages that might contain or verify the website information
    - Any other relevant URLs that could substantiate the website claim
    """


def prompt_extract_address_urls(daycare_name: str) -> str:
    return f"""
    Extract all URLs from the answer that could potentially support the address information for "{daycare_name}".
    This includes:
    - Google Maps links
    - Directory listings that show the address
    - Official website pages that display the address
    - Any other relevant URLs that could substantiate the address claim
    """


def prompt_extract_rating_urls(daycare_name: str) -> str:
    return f"""
    Extract all URLs from the answer that could potentially support the Google Maps rating for "{daycare_name}".
    This includes:
    - Direct Google Maps/Google Business links
    - Review aggregator sites that show Google ratings
    - Any other relevant URLs that could substantiate the rating claim
    """


def prompt_extract_license_urls(daycare_name: str) -> str:
    return f"""
    Extract all URLs from the answer that could potentially support the EEC licensing information for "{daycare_name}".
    This includes:
    - Direct links to Massachusetts EEC website licensing profiles
    - Links to pages that contain or verify licensing information
    - Any other relevant URLs that could substantiate the licensing claim
    """


# --------------------------------------------------------------------------- #
# Individual verification functions                                           #
# --------------------------------------------------------------------------- #
async def verify_website_info(
        evaluator: Evaluator,
        parent_node,
        daycare: DaycareCenter,
        daycare_index: int,
) -> None:
    """Verify website information."""
    website_node = evaluator.add_parallel(
        id=f"daycare_{daycare_index}_website",
        desc=f"Website information for daycare {daycare_index + 1} is accurate",
        parent=parent_node,
        critical=False
    )

    # Existence check
    website_exists = evaluator.add_custom_node(
        result=bool(daycare.website and daycare.website.strip()),
        id=f"daycare_{daycare_index}_website_exists",
        desc=f"Website URL is provided for daycare {daycare_index + 1}",
        parent=website_node,
        critical=True
    )

    # Directly verify if the website is for this daycare
    website_verification_node = evaluator.add_leaf(
        id=f"daycare_{daycare_index}_website_verification",
        desc=f"Website URL leads to the official website for {daycare.name or f'daycare {daycare_index + 1}'}",
        parent=website_node,
        critical=True
    )

    website_claim = f"The URL {daycare.website} is the official website for the daycare center {daycare.name or f'daycare {daycare_index + 1}'}"
    await evaluator.verify(
        claim=website_claim,
        node=website_verification_node,
        sources=daycare.website,
        additional_instruction="Verify that this URL leads to an official website for this specific daycare center. Look for the daycare name, childcare services information, or other indicators that this is the correct official website."
    )


async def verify_address_info(
        evaluator: Evaluator,
        parent_node,
        daycare: DaycareCenter,
        daycare_index: int,
        gmaps_tool: tool_googlemap.GoogleMapsTool,
) -> None:
    """Verify address information with location requirements and distance checking."""
    address_node = evaluator.add_parallel(
        id=f"daycare_{daycare_index}_address",
        desc=f"Address information for daycare {daycare_index + 1} is accurate, in Brookline MA, within 10 miles, and substantiated",
        parent=parent_node,
        critical=True
    )

    # Extract supporting URLs for address claim
    address_urls = await evaluator.extract(
        prompt=prompt_extract_address_urls(daycare.name or f"daycare {daycare_index + 1}"),
        template_class=ExtractedAddressUrls,
        extraction_name=f"address_urls_{daycare_index}"
    )

    # Existence check
    address_exists = evaluator.add_custom_node(
        result=bool(daycare.address and daycare.address.strip()) and bool(address_urls.urls),
        id=f"daycare_{daycare_index}_address_exists",
        desc=f"Address and supporting sources are provided for daycare {daycare_index + 1}",
        parent=address_node,
        critical=True
    )

    # Combined verification: location requirement AND source substantiation
    address_verification_node = evaluator.add_leaf(
        id=f"daycare_{daycare_index}_address_verification",
        desc=f"Address is in Brookline, MA and verified from provided sources",
        parent=address_node,
        critical=True
    )

    address_claim = f"The address for {daycare.name or f'daycare {daycare_index + 1}'} is {daycare.address}, which is located in Brookline, Massachusetts, as shown on the provided source pages"
    await evaluator.verify(
        claim=address_claim,
        node=address_verification_node,
        sources=address_urls.urls,
        additional_instruction="Verify two things: (1) The address is specifically in Brookline, MA, and (2) The provided URLs contain or confirm this address for this daycare."
    )

    # Distance verification using direct calculation
    try:
        # Get driving distance in meters
        distance_meters = await gmaps_tool.calculate_distance(
            daycare.address,
            REFERENCE_ADDRESS,
            mode="driving"
        )

        if isinstance(distance_meters, int):
            # Convert meters to miles
            distance_miles = distance_meters / 1609.34
            
            # Direct comparison - no LLM needed
            distance_within_limit = distance_miles <= 10.0
            
            distance_node = evaluator.add_custom_node(
                result=distance_within_limit,
                id=f"daycare_{daycare_index}_distance",
                desc=f"Driving distance of {distance_miles:.2f} miles from {REFERENCE_ADDRESS} is within 10 miles",
                parent=address_node,
                critical=True
            )
        else:
            # API call failed or returned non-numeric result
            distance_node = evaluator.add_custom_node(
                result=False,
                id=f"daycare_{daycare_index}_distance",
                desc=f"Failed to calculate driving distance from {REFERENCE_ADDRESS}",
                parent=address_node,
                critical=True
            )
    except Exception as e:
        # Handle any exceptions from the API call
        distance_node = evaluator.add_custom_node(
            result=False,
            id=f"daycare_{daycare_index}_distance",
            desc=f"Error calculating driving distance: {str(e)}",
            parent=address_node,
            critical=True
        )


async def verify_rating_info(
        evaluator: Evaluator,
        parent_node,
        daycare: DaycareCenter,
        daycare_index: int,
) -> None:
    """Verify Google Maps rating information with provenance checking."""
    rating_node = evaluator.add_parallel(
        id=f"daycare_{daycare_index}_rating",
        desc=f"Google Maps rating for daycare {daycare_index + 1} is accurate and substantiated",
        parent=parent_node,
        critical=False
    )

    # Extract supporting URLs for rating claim
    rating_urls = await evaluator.extract(
        prompt=prompt_extract_rating_urls(daycare.name or f"daycare {daycare_index + 1}"),
        template_class=ExtractedRatingUrls,
        extraction_name=f"rating_urls_{daycare_index}"
    )

    # Existence check
    rating_exists = evaluator.add_custom_node(
        result=bool(daycare.google_maps_rating and daycare.google_maps_rating.strip()) and bool(rating_urls.urls),
        id=f"daycare_{daycare_index}_rating_exists",
        desc=f"Google Maps rating and supporting sources are provided for daycare {daycare_index + 1}",
        parent=rating_node,
        critical=True
    )

    # Verify the rating is substantiated by the sources
    rating_verification_node = evaluator.add_leaf(
        id=f"daycare_{daycare_index}_rating_verification",
        desc=f"Google Maps rating '{daycare.google_maps_rating}' is verified from the provided sources",
        parent=rating_node,
        critical=True
    )

    rating_claim = f"The Google Maps rating for {daycare.name or f'daycare {daycare_index + 1}'} is {daycare.google_maps_rating} as shown on the provided source pages"
    await evaluator.verify(
        claim=rating_claim,
        node=rating_verification_node,
        sources=rating_urls.urls,
        additional_instruction="Verify that the provided URLs contain or display this specific Google Maps rating for this daycare. The rating should match what is claimed."
    )


async def verify_license_info(
        evaluator: Evaluator,
        parent_node,
        daycare: DaycareCenter,
        daycare_index: int,
) -> None:
    """Verify EEC licensing information."""
    license_node = evaluator.add_parallel(
        id=f"daycare_{daycare_index}_license",
        desc=f"EEC licensing information for daycare {daycare_index + 1} is accurate",
        parent=parent_node,
        critical=True
    )

    # Existence check
    license_exists = evaluator.add_custom_node(
        result=bool(daycare.eec_license_link and daycare.eec_license_link.strip()),
        id=f"daycare_{daycare_index}_license_exists",
        desc=f"EEC license link is provided for daycare {daycare_index + 1}",
        parent=license_node,
        critical=True
    )

    # Directly verify if the link leads to the license profile
    license_verification_node = evaluator.add_leaf(
        id=f"daycare_{daycare_index}_license_verification",
        desc=f"EEC license link leads to the actual licensing profile for {daycare.name or f'daycare {daycare_index + 1}'}",
        parent=license_node,
        critical=True
    )

    license_claim = f"The URL {daycare.eec_license_link} is a valid Massachusetts EEC licensing profile page for {daycare.name or f'daycare {daycare_index + 1}'}"
    await evaluator.verify(
        claim=license_claim,
        node=license_verification_node,
        sources=daycare.eec_license_link,
        additional_instruction="Verify that this URL leads to an actual Massachusetts EEC licensing profile page for this specific daycare. The page should contain licensing information, license numbers, or other official EEC licensing details for this daycare center."
    )


async def verify_single_daycare(
        evaluator: Evaluator,
        parent_node,
        daycare: DaycareCenter,
        daycare_index: int,
        gmaps_tool: tool_googlemap.GoogleMapsTool,
) -> None:
    """Verify all information for a single daycare center."""
    daycare_node = evaluator.add_parallel(
        id=f"daycare_{daycare_index}",
        desc=f"Daycare {daycare_index + 1} ({daycare.name or 'unnamed'}) meets all requirements with proper substantiation",
        parent=parent_node,
        critical=False
    )

    # Check if daycare has a name (completeness check)
    name_exists = evaluator.add_custom_node(
        result=bool(daycare.name and daycare.name.strip()),
        id=f"daycare_{daycare_index}_name_exists",
        desc=f"Daycare {daycare_index + 1} has a valid name provided",
        parent=daycare_node,
        critical=True
    )

    # Verify all required fields
    await verify_website_info(evaluator, daycare_node, daycare, daycare_index)
    await verify_address_info(evaluator, daycare_node, daycare, daycare_index, gmaps_tool)
    await verify_rating_info(evaluator, daycare_node, daycare, daycare_index)
    await verify_license_info(evaluator, daycare_node, daycare, daycare_index)


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: openai.AsyncAzureOpenAI,
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
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()
    
    # Initialize evaluator with parallel strategy for root
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
        default_model=model if model else JUDGE_MODEL
    )

    # -------- 2. Set up Google Maps tool -------------------------------- #
    gmaps_tool = tool_googlemap.GoogleMapsTool()

    # -------- 3. Extract structured info from the answer ----------------- #
    parsed_info = await evaluator.extract(
        prompt=prompt_extract_daycares(),
        template_class=ExtractedDaycares,
        extraction_name="daycares"
    )

    # Ensure we have exactly 3 daycares to verify (pad with empty if needed)
    daycares_to_verify = list(parsed_info.daycares[:3])
    while len(daycares_to_verify) < 3:
        daycares_to_verify.append(DaycareCenter())

    # -------- 4. Build verification tree -------------------------------- #
    # Verify each of the 3 daycare centers
    for i in range(3):
        await verify_single_daycare(evaluator, root, daycares_to_verify[i], i, gmaps_tool)

    # -------- 5. Add custom info ---------------------------------------- #
    evaluator.add_custom_info({
        "total_daycares_found": len(parsed_info.daycares),
        "daycares_processed": min(3, len(parsed_info.daycares)),
        "reference_address": REFERENCE_ADDRESS,
    }, "task_statistics")

    # -------- 6. Return structured result ------------------------------- #
    return evaluator.get_summary()