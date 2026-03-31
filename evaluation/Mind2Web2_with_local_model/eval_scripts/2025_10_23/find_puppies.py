import asyncio
import logging
from typing import Optional, List, Dict

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "find_puppies"
TASK_DESCRIPTION = """
I'm planning to get a dog from the AKC Marketplace. Please find five female puppies of different breeds that are currently available in Ohio on the marketplace. The breeds should be long-haired and weigh no more than 65 pounds according to information on AKC.

For each puppy, please provide a direct link to that specific puppy on the AKC Marketplace (not the search results), the breed name, a link to the breed description on AKC, and the breed's weight range as listed by the AKC.
"""

JUDGE_MODEL = "o4-mini"

# --------------------------------------------------------------------------- #
# Data-models for extracted info                                              #
# --------------------------------------------------------------------------- #
class PuppyInfo(BaseModel):
    """Information about a single puppy."""
    puppy_url: Optional[str] = None  # Direct link to specific puppy listing
    breed_name: Optional[str] = None  # Name of the breed
    breed_url: Optional[str] = None  # Link to breed description on AKC
    weight_range: Optional[str] = None  # Weight range as listed by AKC


class ExtractedPuppies(BaseModel):
    """Container for all extracted puppies."""
    puppies: List[PuppyInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_puppies() -> str:
    return """
    Extract information about puppies mentioned in the answer. For each puppy, extract:
    
    1. puppy_url: The direct link to the specific puppy on the AKC Marketplace.
    2. breed_name: The name of the breed.
    3. breed_url: The link to the breed description on AKC's website.
    4. weight_range: The weight range for the breed as listed by AKC.
    
    Return a list of puppies with all the information above. If any information is missing, set that field to null.
    If a URL is provided without http:// or https://, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                     #
# --------------------------------------------------------------------------- #
async def verify_puppy(
        evaluator: Evaluator,
        parent_node,
        index: int, 
        puppy: PuppyInfo,
) -> None:
    """
    Verify all requirements for a single puppy.
    """
    # Create the main puppy node with parallel strategy
    puppy_node = evaluator.add_parallel(
        id=f"puppy_{index+1}",
        desc=f"Puppy #{index+1}: {puppy.breed_name if puppy.breed_name else 'Unknown breed'}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit across puppies
    )
    
    # 1. Single comprehensive completeness check for all required fields
    completeness_node = evaluator.add_custom_node(
        result=(
            bool(puppy.puppy_url and "marketplace.akc.org" in puppy.puppy_url) and
            bool(puppy.breed_name and puppy.breed_name.strip()) and
            bool(puppy.breed_url and "akc.org" in puppy.breed_url) and
            bool(puppy.weight_range)
        ),
        id=f"puppy_{index+1}_completeness",
        desc=f"Puppy #{index+1} has all required information (puppy URL, breed name, breed URL, weight range)",
        parent=puppy_node,
        critical=True
    )
    
    # 2. Verify puppy URL is valid direct link
    url_valid_node = evaluator.add_leaf(
        id=f"puppy_{index+1}_url_valid",
        desc=f"Puppy #{index+1} URL is a direct link to a specific puppy",
        parent=puppy_node,
        critical=True
    )
    
    await evaluator.verify(
        claim="The webpage is a valid direct link to a specific puppy on AKC Marketplace.",
        node=url_valid_node,
        sources=puppy.puppy_url,
        additional_instruction="The webpage should link directly to a specific puppy rather than displaying a list of search results."
    )
    
    # 3. Verify puppy availability
    availability_node = evaluator.add_leaf(
        id=f"puppy_{index+1}_is_available",
        desc=f"Puppy #{index+1} is currently available",
        parent=puppy_node,
        critical=True,
    )
    
    await evaluator.verify(
        claim="The puppy is currently available for purchase or adoption.",
        node=availability_node,
        sources=puppy.puppy_url,
        additional_instruction="Look for any indication that the puppy is available, such as 'available', 'for sale', etc. If the puppy is marked as 'sold', 'unavailable', or 'reserved', then it is not available."
    )
    
    # 4. Verify if puppy is female
    female_node = evaluator.add_leaf(
        id=f"puppy_{index+1}_is_female",
        desc=f"Puppy #{index+1} is female",
        parent=puppy_node,
        critical=True,
    )
    
    await evaluator.verify(
        claim="The puppy is female.",
        node=female_node,
        sources=puppy.puppy_url,
        additional_instruction="Look for gender information on the page, such as 'female', 'girl', etc. If the puppy is explicitly marked as 'male' or 'boy', then it is not female."
    )
    
    # 5. Verify if puppy is in Ohio
    ohio_node = evaluator.add_leaf(
        id=f"puppy_{index+1}_in_ohio",
        desc=f"Puppy #{index+1} is located in Ohio",
        parent=puppy_node,
        critical=True,
    )
    
    await evaluator.verify(
        claim="The puppy is located in Ohio.",
        node=ohio_node,
        sources=puppy.puppy_url,
        additional_instruction="Look for location information on the page that indicates the puppy is in Ohio. This might be shown as 'OH', 'Ohio', or mention of a city in Ohio."
    )
    
    # 6. Verify breed name matches
    breed_name_correct = evaluator.add_leaf(
        id=f"puppy_{index+1}_breed_name_correct",
        desc=f"Breed name matches the puppy listing",
        parent=puppy_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The breed of puppy shown in the webpage is {puppy.breed_name}.",
        node=breed_name_correct,
        sources=puppy.puppy_url
    )
    
    # 7. Verify breed URL is valid
    breed_url_valid = evaluator.add_leaf(
        id=f"puppy_{index+1}_breed_url_valid",
        desc=f"Breed URL is valid AKC breed description",
        parent=puppy_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The webpage is breed description of breed {puppy.breed_name} on AKC.",
        node=breed_url_valid,
        sources=puppy.breed_url
    )
    
    # 8. Verify if breed is long-haired
    long_hair_node = evaluator.add_leaf(
        id=f"puppy_{index+1}_long_hair",
        desc=f"Breed of puppy #{index+1} is long-haired",
        parent=puppy_node,
        critical=True,
    )
    
    await evaluator.verify(
        claim=f"The {puppy.breed_name} breed is considered long-haired.",
        node=long_hair_node,
        sources=puppy.breed_url,
        additional_instruction="Look for descriptions of the breed's coat like 'long coat', 'long hair', 'flowing coat', 'abundant coat', etc. Use common sense judgment - breeds with medium to long hair should be considered long-haired."
    )
    
    # 9. Verify weight limit (decoupled into two checks)
    
    # 9a. Check if the extracted weight range is under 65 pounds
    weight_under_65 = evaluator.add_leaf(
        id=f"puppy_{index+1}_weight_under_65",
        desc=f"Extracted weight range for puppy #{index+1} is under 65 pounds",
        parent=puppy_node,
        critical=True,
    )
    
    await evaluator.verify(
        claim=f"The weight range '{puppy.weight_range}' indicates that the breed weighs no more than 65 pounds.",
        node=weight_under_65,
        sources=None,  # This is a simple claim verification, no URL needed
        additional_instruction="Check if the weight range indicates a maximum weight of 65 pounds or less. For ranges like '50-60 pounds' or 'up to 55 pounds', the maximum should not exceed 65. If different weights are given for males and females, focus on the female weight since the task specifically asks about female puppies."
    )
    
    # 9b. Verify the weight range matches what's on the breed URL
    weight_matches_url = evaluator.add_leaf(
        id=f"puppy_{index+1}_weight_matches_breed_page",
        desc=f"Weight range for puppy #{index+1} matches the breed page",
        parent=puppy_node,
        critical=True,
    )
    
    await evaluator.verify(
        claim=f"The breed page shows that {puppy.breed_name} has a weight range of {puppy.weight_range}.",
        node=weight_matches_url,
        sources=puppy.breed_url,
        additional_instruction="Verify that the weight range provided in the answer matches what's shown on the AKC breed description page. Look for weight information in the breed characteristics or specifications section."
    )


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
    
    # Initialize evaluator
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

    # -------- 2. Extract structured puppy info from the answer ----------- #
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_puppies(),
        template_class=ExtractedPuppies,
        extraction_name="extracted_puppies"
    )
    
    # -------- 3. Deduplicate puppies by breed, keeping only the first of each breed -------- #
    breed_to_puppy = {}
    for puppy in extracted_info.puppies:
        if puppy.breed_name and puppy.breed_name.strip():
            breed_name = puppy.breed_name.strip().lower()
            # Only add if this breed hasn't been seen before
            if breed_name not in breed_to_puppy:
                breed_to_puppy[breed_name] = puppy
    
    # Convert to a list of unique puppies (one per breed)
    unique_puppies = list(breed_to_puppy.values())
    
    # Pad the list to ensure we have exactly 5 puppies (empty ones if needed)
    while len(unique_puppies) < 5:
        unique_puppies.append(PuppyInfo())
    
    # -------- 4. Add custom info about deduplication --------------------- #
    evaluator.add_custom_info({
        "total_puppies_extracted": len(extracted_info.puppies),
        "unique_breeds_extracted": len([p for p in unique_puppies if p.breed_name]),
        "puppies_after_deduplication": len(unique_puppies)
    }, "deduplication_stats")
    
    # -------- 5. Verify each puppy --------------------------------------- #
    verification_tasks = []
    for i in range(5):  # We need exactly 5 puppies
        verification_tasks.append(
            verify_puppy(
                evaluator=evaluator,
                parent_node=root,
                index=i,
                puppy=unique_puppies[i]
            )
        )
    
    # Wait for all verification tasks to complete
    await asyncio.gather(*verification_tasks)
    
    # -------- 6. Return structured result -------------------------------- #
    return evaluator.get_summary()