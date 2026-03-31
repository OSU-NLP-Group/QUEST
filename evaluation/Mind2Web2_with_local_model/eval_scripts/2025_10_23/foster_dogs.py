import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.api_tools import tool_googlemap

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "foster_dogs"
TASK_DESCRIPTION = """
I'm looking to foster a dog in Columbus, OH. Please help me find an animal shelter within a 15-minute drive of Easton Town Center that offers a dog foster program. The shelter's website should list dogs needing foster with their photos and descriptions. What is the physical address of that animal shelter? From their website, please find three dogs currently available for foster who are under six years old, and include links to their individual profiles.
"""

# Easton Town Center address for distance calculation
EASTON_TOWN_CENTER_ADDRESS = "160 Easton Town Center, Columbus, OH 43219"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ShelterInfo(BaseModel):
    """Information about the animal shelter."""
    name: Optional[str] = None
    address: Optional[str] = None
    website_url: Optional[str] = None
    has_foster_program: Optional[bool] = None
    distance_from_easton: Optional[str] = None

class DogProfile(BaseModel):
    """Information about each dog available for fostering."""
    name: Optional[str] = None
    age: Optional[str] = None  # Age as a string (e.g., "3 years", "5 months")
    profile_url: Optional[str] = None
    description: Optional[str] = None
    shelter_name: Optional[str] = None  # Which shelter is this dog from

class FosterDogsExtraction(BaseModel):
    """Complete extraction of shelter and dog information."""
    shelter: Optional[ShelterInfo] = None
    dogs: List[DogProfile] = Field(default_factory=list)

class ProvLink(BaseModel):
    url: str
    description: str

class ProvLinks(BaseModel):
    links: List[ProvLink] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_shelter_info() -> str:
    return """
    Extract information about the animal shelter mentioned in the answer. Include:
    1. The name of the shelter
    2. The physical address of the shelter
    3. The website URL of the shelter
    4. Whether the shelter has a foster program (true/false)
    5. Any information about the distance or drive time from Easton Town Center

    If any information is not mentioned in the answer, return null for that field.
    """

def prompt_extract_dog_profiles() -> str:
    return """
    Extract information about all dogs mentioned in the answer that are available for fostering.
    For each dog, include:
    1. The dog's name
    2. The dog's age (as provided in the text)
    3. The URL to the dog's profile page
    4. A brief description of the dog (if provided)
    5. The name of the shelter this dog is from (if mentioned)

    Extract information for ALL dogs mentioned in the answer, even if there are more than three.
    If any field is not mentioned for a particular dog, return null for that field.
    """

def prompt_extract_shelter_urls(shelter_name: str) -> str:
    return f"""
    Extract all URLs mentioned in the answer that are specifically related to the shelter named "{shelter_name}".
    For each URL, include:
    1. The complete URL
    2. A brief description of what the URL points to (e.g., "shelter homepage", "about page", "adoption page")
    
    Focus on URLs that might be useful for verifying information about this specific shelter,
    such as its location, foster program, or dog listings.
    
    Do NOT include URLs for other shelters that might be mentioned in the answer.
    """

# --------------------------------------------------------------------------- #
# Helper function to collect URLs                                             #
# --------------------------------------------------------------------------- #
def collect_shelter_urls(shelter_info: ShelterInfo, shelter_urls: ProvLinks) -> List[str]:
    """Collect all available URLs for shelter verification."""
    urls = []
    if shelter_info.website_url:
        urls.append(shelter_info.website_url)
    if shelter_urls and shelter_urls.links:
        urls.extend([link.url for link in shelter_urls.links])
    return urls

# --------------------------------------------------------------------------- #
# Verification functions for shelter requirements                             #
# --------------------------------------------------------------------------- #
async def verify_shelter_requirements(
    evaluator: Evaluator,
    parent_node,
    shelter_info: ShelterInfo,
    shelter_urls: ProvLinks,
    gmaps_tool: tool_googlemap.GoogleMapsTool,
) -> None:
    """
    Verify all shelter requirements in a single parallel node.
    """
    # Single existence check for all required shelter information
    existence_check = evaluator.add_custom_node(
        result=bool(shelter_info.name and shelter_info.address and collect_shelter_urls(shelter_info, shelter_urls)),
        id="shelter_complete_info_exists",
        desc="Check if shelter name, address, and URLs were provided",
        parent=parent_node,
        critical=True
    )
    
    # Collect URLs once for all verifications
    urls_to_check = collect_shelter_urls(shelter_info, shelter_urls)
    
    # 1. Verify shelter exists
    shelter_exists_node = evaluator.add_leaf(
        id="shelter_exists_verification",
        desc="Verification that the shelter exists with the provided name and address",
        parent=parent_node,
        critical=True
    )
    
    shelter_claim = f"There is an animal shelter named '{shelter_info.name}' located at '{shelter_info.address}'."
    
    await evaluator.verify(
        claim=shelter_claim,
        node=shelter_exists_node,
        sources=urls_to_check if urls_to_check else None,
    )
    
    # 2. Verify location using Google Maps API
    skip_flag = False if shelter_exists_node.status == "passed" else True
    await verify_location_with_gmaps(evaluator, parent_node, shelter_info, gmaps_tool, skip_flag)
    
    # 3. Verify foster program
    foster_node = evaluator.add_leaf(
        id="foster_program_verification",
        desc="Verification that the shelter offers a dog foster program",
        parent=parent_node,
        critical=True
    )
    
    foster_claim = f"The animal shelter '{shelter_info.name}' offers a dog foster program."
    
    await evaluator.verify(
        claim=foster_claim,
        node=foster_node,
        sources=urls_to_check if urls_to_check else None,
        additional_instruction="Look for explicit mention of a foster program for dogs. This might be on pages about volunteering, fostering, or ways to help."
    )
    
    # 4. Verify dog listings
    listings_node = evaluator.add_leaf(
        id="dog_listings_verification",
        desc="Verification that the shelter website lists dogs with photos and descriptions",
        parent=parent_node,
        critical=True
    )
    
    listings_claim = "The shelter's website lists dogs that need fostering, and includes their photos and descriptions."
    
    await evaluator.verify(
        claim=listings_claim,
        node=listings_node,
        sources=urls_to_check if urls_to_check else None,
        additional_instruction="Look for pages that list dogs available for fostering. Check if these listings include photos and descriptions of the dogs."
    )

async def verify_location_with_gmaps(
    evaluator: Evaluator,
    parent_node,
    shelter_info: ShelterInfo,
    gmaps_tool: tool_googlemap.GoogleMapsTool,
    skip_flag: bool,
) -> None:
    """
    Verify location requirement using Google Maps API.
    """
    location_node = evaluator.add_leaf(
        id="shelter_location_verification",
        desc="Verification that the shelter is within 15-minute drive of Easton Town Center",
        parent=parent_node,
        critical=True
    )

    if skip_flag:
        location_node.status = "skipped"
        location_node.score = 0.0
        return

    try:
        # Get driving time in seconds
        travel_time_seconds = await gmaps_tool.calculate_travel_time(
            shelter_info.address,
            EASTON_TOWN_CENTER_ADDRESS,
            mode="driving"
        )
        
        if isinstance(travel_time_seconds, int):
            # Convert seconds to minutes
            travel_time_minutes = travel_time_seconds / 60
            
            # Direct comparison - no LLM needed
            if travel_time_minutes <= 15.0:
                location_node.score = 1.0
                location_node.status = "passed"
            else:
                location_node.score = 0.0
                location_node.status = "failed"
        else:
            # API returned unexpected format
            location_node.score = 0.0
            location_node.status = "failed"
            
    except Exception as e:
        # If Google Maps fails, mark as failed
        location_node.score = 0.0
        location_node.status = "failed"

# --------------------------------------------------------------------------- #
# Verification functions for dog profile requirements                         #
# --------------------------------------------------------------------------- #
async def verify_dog_profile(
    evaluator: Evaluator,
    parent_node,
    dog: DogProfile,
    index: int,
    shelter_info: ShelterInfo,
    shelter_urls: List[str],
) -> None:
    """
    Verify information about a single dog profile.
    """
    dog_node = evaluator.add_parallel(
        id=f"dog_{index}",
        desc=f"Dog #{index+1}: Verification of dog profile information and eligibility.",
        parent=parent_node,
        critical=False,
    )
    
    # Combined existence check for all required dog information
    existence_check = evaluator.add_custom_node(
        result=bool(dog.name and dog.profile_url),
        id=f"dog_{index}_info_exists",
        desc=f"Check if dog #{index+1} has name and profile URL",
        parent=dog_node,
        critical=True
    )
    
    # Verify the dog is from the correct shelter
    shelter_match_node = evaluator.add_leaf(
        id=f"dog_{index}_shelter_match",
        desc=f"Dog #{index+1} is from the verified shelter '{shelter_info.name}'",
        parent=dog_node,
        critical=True
    )
    
    shelter_claim = f"This dog profile is from the shelter '{shelter_info.name}'."
    
    await evaluator.verify(
        claim=shelter_claim,
        node=shelter_match_node,
        sources=dog.profile_url,
        additional_instruction=f"Verify that this dog profile is from {shelter_info.name}. Look for the shelter name on the page, or check if the URL domain matches the shelter's website domain."
    )
    
    # Verify the dog's age is under six years
    age_node = evaluator.add_leaf(
        id=f"dog_{index}_age",
        desc=f"Dog #{index+1} ({dog.name if dog.name else 'unnamed'}) is under six years old",
        parent=dog_node,
        critical=True
    )
    
    age_claim = f"The dog named {dog.name} is under six years old."
    
    await evaluator.verify(
        claim=age_claim,
        node=age_node,
        sources=dog.profile_url,
        additional_instruction="Look for information about the dog's age. Determine if the dog is under six years old. The age might be expressed in years, months, or described as 'young', 'puppy', etc."
    )
    
    # Verify the dog is available for fostering
    foster_node = evaluator.add_leaf(
        id=f"dog_{index}_foster",
        desc=f"Dog #{index+1} ({dog.name if dog.name else 'unnamed'}) is available for fostering",
        parent=dog_node,
        critical=True
    )
    
    foster_claim = f"The dog named {dog.name if dog.name else 'This dog'} is currently available for fostering."
    
    await evaluator.verify(
        claim=foster_claim,
        node=foster_node,
        sources=dog.profile_url,
        additional_instruction="Look for explicit information indicating that this dog is available for fostering (not just adoption). The information might be on the dog's profile page or might reference a foster program."
    )

# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()
    
    # Initialize evaluator with sequential strategy for root
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # -------- 2. Set up Google Maps tool --------------------------------- #
    gmaps_tool = tool_googlemap.GoogleMapsTool()

    # -------- 3. Extract structured info from the answer ----------------- #
    # First, extract shelter information
    shelter_info = await evaluator.extract(
        prompt=prompt_extract_shelter_info(),
        template_class=ShelterInfo,
        extraction_name="shelter_info"
    )
    
    # Extract URLs specifically for the identified shelter
    if shelter_info.name:
        shelter_urls = await evaluator.extract(
            prompt=prompt_extract_shelter_urls(shelter_info.name),
            template_class=ProvLinks,
            extraction_name="shelter_urls"
        )
    else:
        # No shelter name found, create empty URL list
        shelter_urls = ProvLinks()
    
    # Extract dog profiles
    dog_profiles = await evaluator.extract(
        prompt=prompt_extract_dog_profiles(),
        template_class=FosterDogsExtraction,
        extraction_name="dog_profiles"
    )
    
    # Ensure we consider at most 3 dogs as required by the task
    dogs_to_verify = dog_profiles.dogs[:3] if dog_profiles.dogs else []
    
    # If fewer than 3 dogs are provided, create placeholder entries
    while len(dogs_to_verify) < 3:
        dogs_to_verify.append(DogProfile())

    # -------- 4. Build verification tree -------------------------------- #
    # Create shelter verification stage
    shelter_stage = evaluator.add_parallel(
        id="shelter_verification",
        desc="Verification of shelter requirements: location, website, foster program, and dog listings.",
        critical=False, # false to allow partial score
    )
    
    # Create dog profiles verification stage (direct child of root, no redundant wrapper)
    dog_profiles_node = evaluator.add_parallel(
        id="dog_profiles",
        desc="Verification of three dogs under six years old available for foster.",
        critical=False,  # false to allow partial score
    )
    
    # -------- 5. Verify shelter requirements ---------------------------- #
    await verify_shelter_requirements(evaluator, shelter_stage, shelter_info, shelter_urls, gmaps_tool)
    
    # -------- 6. Verify dog profiles ------------------------------------ #
    # Collect shelter URLs for dog verification
    shelter_url_list = collect_shelter_urls(shelter_info, shelter_urls)
    
    # Verify each dog profile
    for i, dog in enumerate(dogs_to_verify):
        await verify_dog_profile(
            evaluator, 
            dog_profiles_node, 
            dog, 
            i,
            shelter_info,
            shelter_url_list
        )
    
    # -------- 7. Add custom info ---------------------------------------- #
    evaluator.add_custom_info({
        "total_dogs_found": len(dog_profiles.dogs) if dog_profiles.dogs else 0,
        "dogs_processed": min(3, len(dog_profiles.dogs) if dog_profiles.dogs else 0),
        "easton_town_center_address": EASTON_TOWN_CENTER_ADDRESS,
        "shelter_identified": shelter_info.name if shelter_info.name else "None",
    }, "task_statistics")
    
    # -------- 8. Return structured result ------------------------------- #
    return evaluator.get_summary()