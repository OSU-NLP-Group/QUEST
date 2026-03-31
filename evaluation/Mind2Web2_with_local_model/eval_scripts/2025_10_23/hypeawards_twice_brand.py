import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import Evaluator

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "hypeawards_twice_brand"
TASK_DESCRIPTION = """
Could you find three brands that have each won a Hypebeast HypeAward or Hypebeast100 Award (excluding Best Sneaker and Best Collab) at least twice from 2020 to the present? Please list the years each brand received the award and the categories they won.
"""

EVALUATION_NOTES = """
Note that the Hypebeast Hypeaward is being renamed to Hypebeast100 Award since 2022
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BrandName(BaseModel):
    """Model for storing just a brand name"""
    name: Optional[str] = None


class BrandNames(BaseModel):
    """Model for storing a list of brand names"""
    brands: List[BrandName] = Field(default_factory=list)


class BrandAward(BaseModel):
    """Model for a specific award won by a brand"""
    year: Optional[str] = None
    category: Optional[str] = None
    urls: List[str] = Field(default_factory=list)  # Changed from url to urls to handle multiple URLs per award


class BrandInfo(BaseModel):
    """Model for detailed information about a brand and its awards"""
    awards: List[BrandAward] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_brand_names() -> str:
    return """
    Extract ONLY the names of brands mentioned in the answer as the final answer to the task, where the answer indicate that they have won Hypebeast HypeAwards or Hypebeast100 Awards for multiple times from 2020 to the present.

    Please extract ONLY the brand names, without any additional information about awards or years.
    Include ALL brands mentioned in the answer as the final answer list, if there are more than three.
    If no brands are mentioned, return an empty list.
    """


def prompt_extract_brand_details(brand_name: str) -> str:
    return f"""
    Extract detailed information about the brand "{brand_name}" from the answer.

    For this specific brand, extract:
    1. A list of awards they won, including:
       - The year they received each award
       - The category of each award
       - Any URLs specifically associated with this award (if present)
    2. Any URLs provided in the answer that are specifically associated with this brand and its awards

    IMPORTANT: Do NOT extract awards in the "Best Sneaker" or "Best Collab" categories, as these are specifically excluded from the task.

    If any field is missing, set it to null.
    If no URLs are provided for this brand or its awards, return an empty list.

    Note that the Hypebeast Hypeaward is being renamed to Hypebeast100 Award since 2022. Consider both names as valid.
    Make sure to only extract URLs that are directly related to this specific brand. Do not include general URLs or URLs related to other brands.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_has_multiple_awards(
        evaluator: Evaluator,
        parent_node,
        brand_name: str,
        brand_info: BrandInfo,
        brand_index: int,
) -> bool:
    """
    Verify that a brand has won at least two awards as required by the task.
    This is critical for each brand to be considered valid.
    """
    multiple_awards_node = evaluator.add_leaf(
        id=f"brand_{brand_index}_multiple_awards",
        desc=f"Verify that {brand_name} has won at least two Hypebeast HypeAwards/Hypebeast100 Awards from 2020 to present",
        parent=parent_node,
        critical=True,
    )

    # Construct a claim about this brand having multiple awards
    awards_text = []
    if brand_info and brand_info.awards:
        for award in brand_info.awards:
            if award.year and award.category:
                awards_text.append(f"{brand_name} won a Hypebeast HypeAward/Hypebeast100 Award in the category '{award.category}' in {award.year}")
    
    evaluator.add_custom_node(
        result=bool(brand_info and brand_info.awards and awards_text),
        id=f"brand_{brand_index}_multiple_awards_existence",
        desc=f"Check that {brand_name} has two awards in the answer",
        parent=multiple_awards_node,
        critical=True,
    )

    multiple_awards_verify_node = evaluator.add_leaf(
        id=f"brand_{brand_index}_multiple_awards_verification",
        desc=f"Check at least two awards for {brand_name}",
        parent=multiple_awards_node,
        critical=True
    )

    # Create the verification claim
    verification_claim = f"Check the following text, whether there are two Hypebeast HypeAwards/Hypebeast100 Awards mentioned in the following text (only considered the awards of the years from 2020 inclusive being valid)? Here's the text : {'; '.join(awards_text)}\n\n"

    # Use verify to check if the brand has at least two awards
    has_multiple_awards = await evaluator.verify(
        claim=verification_claim,
        node=multiple_awards_verify_node,
        additional_instruction="Consider both 'HypeAward' and 'Hypebeast100 Award' as valid award names.",
    )

    return has_multiple_awards


async def verify_award_has_category(
        evaluator: Evaluator,
        parent_node,
        award: BrandAward,
        brand_name: str,
        award_index: int,
        brand_index: int,
) -> bool:
    """
    Verify that an award has a category specified using existence check.
    """
    # Use add_custom_node for existence check
    has_category = bool(award.category and award.category.strip())
    
    category_node = evaluator.add_custom_node(
        result=has_category,
        id=f"brand_{brand_index}_award_{award_index}_has_category",
        desc=f"Verify that award {award_index + 1} for {brand_name} has a category specified",
        parent=parent_node,
        critical=True,
    )

    return has_category


async def verify_award_provenance(
        evaluator: Evaluator,
        parent_node,
        brand_name: str,
        award: BrandAward,
        award_index: int,
        brand_index: int,
        brand_urls: List[str],
) -> None:
    """
    Verify that the award information is backed by valid sources.
    Uses only URLs relevant to this brand and award.
    """
    award_year = award.year if award.year else "unknown year"
    award_category = award.category if award.category else "unknown category"

    # Create parent node for award provenance verification
    provenance_parent = evaluator.add_parallel(
        id=f"brand_{brand_index}_award_{award_index}_provenance",
        desc=f"Verify that {brand_name}'s award in {award_year} for category '{award_category}' is substantiated by valid sources",
        parent=parent_node,
        critical=True,
    )

    # Combine URLs from both the brand and this specific award
    check_urls = list(brand_urls)
    if award.urls:
        for url in award.urls:
            if url not in check_urls:
                check_urls.append(url)

    # Add existence check for URLs
    url_exists_node = evaluator.add_custom_node(
        result=bool(check_urls),
        id=f"brand_{brand_index}_award_{award_index}_urls_exist",
        desc=f"Check if URLs are provided for {brand_name}'s award verification",
        parent=provenance_parent,
        critical=True,
    )

    # Add verification node
    verification_node = evaluator.add_leaf(
        id=f"brand_{brand_index}_award_{award_index}_verification",
        desc=f"Verify award details from sources for {brand_name}",
        parent=provenance_parent,
        critical=True,
    )

    # Create the claim for verification
    award_claim = f"{brand_name} won a Hypebeast HypeAward/Hypebeast100 Award in the category '{award_category}' in {award_year}"

    # Always call verify - the gate-then-average logic will handle missing URLs
    await evaluator.verify(
        claim=award_claim,
        node=verification_node,
        sources=check_urls,
        additional_instruction="Note that the Hypebeast Hypeaward is being renamed to Hypebeast100 Award since 2022. Consider both names as valid. Verify that this specific brand won this specific award in this specific category and year.",
    )


async def verify_individual_award(
        evaluator: Evaluator,
        parent_node,
        brand_name: str,
        award: BrandAward,
        award_index: int,
        brand_index: int,
        brand_urls: List[str],
) -> None:
    """
    Verify a single award, including its category and provenance.
    """
    award_year = award.year if award.year else "unknown year"

    award_node = evaluator.add_sequential(
        id=f"brand_{brand_index}_award_{award_index}",
        desc=f"Verification of award {award_index + 1} ({award_year}) for {brand_name}",
        parent=parent_node,
        critical=True,
    )

    # First check: Award has valid category
    has_category = await verify_award_has_category(
        evaluator,
        award_node,
        award,
        brand_name,
        award_index,
        brand_index
    )

    # Always add provenance verification (sequential logic will handle skipping if category check fails)
    await verify_award_provenance(
        evaluator,
        award_node,
        brand_name,
        award,
        award_index,
        brand_index,
        brand_urls
    )


async def verify_individual_brand(
        evaluator: Evaluator,
        parent_node,
        brand_name: Optional[str] ,
        brand_index: int,
) -> None:
    """
    Verify all aspects of a single brand and its awards.
    If brand_name is None, create nodes that will fail existence checks.
    """
    brand_label = brand_name if brand_name else f"Brand {brand_index + 1}"

    brand_node = evaluator.add_parallel(
        id=f"brand_{brand_index}",
        desc=f"Verification of brand {brand_index + 1}: {brand_label}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit for other valid brands
    )

    # Add existence check for brand name
    brand_exists_node = evaluator.add_custom_node(
        result=bool(brand_name and brand_name.strip()),
        id=f"brand_{brand_index}_exists",
        desc=f"Check if brand {brand_index + 1} exists in the answer",
        parent=brand_node,
        critical=True,
    )

    # If we don't have a brand, the remaining verification will be automatically skipped
    # due to the critical existence check failing
    
    # Sequential checks for this brand
    brand_checks = evaluator.add_sequential(
        id=f"brand_{brand_index}_checks",
        desc=f"Sequential checks for {brand_label}",
        parent=brand_node,
        critical=True,
    )

    # Extract detailed information for this specific brand (even if None for consistent structure)
    brand_info = BrandInfo()  # Empty model if no brand name
    if brand_name:
        brand_info = await evaluator.extract(
            prompt=prompt_extract_brand_details(brand_name),
            template_class=BrandInfo,
            extraction_name=f"brand_{brand_index}_details",
            additional_instruction=EVALUATION_NOTES,
        )

    # Check 1: Verify brand has multiple awards
    has_multiple_awards = await verify_has_multiple_awards(
        evaluator,
        brand_checks,
        brand_name or f"Brand {brand_index + 1}",
        brand_info,
        brand_index,
    )

    # Awards verification - always create structure, let gate-then-average handle logic
    awards_verification_node = evaluator.add_parallel(
        id=f"brand_{brand_index}_awards_verification",
        desc=f"Verification of individual awards for {brand_label}",
        parent=brand_checks,
        critical=True,
    )

    # Take only the first two awards to verify (pad if needed for consistent structure)
    awards_to_verify = brand_info.awards[:2] if brand_info.awards else []
    while len(awards_to_verify) < 2:
        awards_to_verify.append(BrandAward())  # Empty award model

    # Verify each award using only the URLs related to this brand
    for i, award in enumerate(awards_to_verify):
        await verify_individual_award(
            evaluator,
            awards_verification_node,
            brand_name or f"Brand {brand_index + 1}",
            award,
            i,
            brand_index,
            brand_info.urls if brand_info else []
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
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        agent_name=agent_name,
        answer_name=answer_name,
        default_model=model,
    )

    # First, extract just the brand names
    extracted_brand_names = await evaluator.extract(
        prompt=prompt_extract_brand_names(),
        template_class=BrandNames,
        extraction_name="extracted_brand_names",
    )

    # Always create exactly 3 verification nodes for consistent structure
    # Pad the brand list if needed
    brand_list = extracted_brand_names.brands[:3] if extracted_brand_names.brands else []
    while len(brand_list) < 3:
        brand_list.append(BrandName(name=None))

    for i in range(3):
        brand_name = brand_list[i].name if brand_list[i].name else None

        # For each brand, extract details and verify (or create structure that will fail existence checks)
        await verify_individual_brand(
            evaluator,
            evaluator.root,
            brand_name,
            i,
        )

    # Return structured result using evaluator's get_summary method
    return evaluator.get_summary()