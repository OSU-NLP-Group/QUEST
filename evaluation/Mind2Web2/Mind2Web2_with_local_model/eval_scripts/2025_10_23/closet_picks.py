import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "closet_picks"
TASK_DESCRIPTION = """
I enjoy watching The Criterion Collection's Closet Picks series, which features artists selecting and discussing their personal favorite films from The Criterion Collection. Please find 8 issues from the Closet Picks series on The Criterion Collection website published within the past 2 years, ensuring that each issue features a director as a guest recommending films. Compile a list of these 8 Closet Picks issues with director names and the links. Additionally, for each issue, select one film and check its availability and price of the cheapest option (if available) on The Criterion Collection website.
"""

CRITERION_CLOSET_PICKS_URL = "https://www.criterion.com/closet-picks/search"
CRITERION_BASE_URL = "https://www.criterion.com"
CURRENT_DATE = datetime.now()
# NOTE: use a hard date now
# CURRENT_DATE = datetime(2025, 5, 1, 23, 7, 18, 890926)
TWO_YEARS_AGO = CURRENT_DATE - timedelta(days=2 * 365)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DirectorsList(BaseModel):
    """Model for extracting the list of directors mentioned in the answer."""
    directors: List[str] = Field(default_factory=list)


class ClosetPickURLs(BaseModel):
    """Model for extracting Closet Pick URLs for a specific director."""
    urls: List[str] = Field(default_factory=list)


class FilmInfo(BaseModel):
    """Model for extracting film information for a specific Closet Pick."""
    title: Optional[str] = None
    availability: Optional[str] = None
    price: Optional[str] = None


class FilmURL(BaseModel):
    """Model for extracting film URL for a specific film."""
    url: Optional[str] = None


class ClosetPickInfo(BaseModel):
    """Complete model for storing all information about a Closet Pick."""
    director_name: str = ""
    closet_pick_url: str = ""
    film_title: Optional[str] = None
    film_availability: Optional[str] = None
    film_price: Optional[str] = None
    film_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_directors() -> str:
    """
    Returns a prompt for extracting the list of directors mentioned in the answer.

    Returns:
        str: The extraction prompt for directors.
    """
    return """
    Extract the names of all directors mentioned in the answer who have Closet Picks in the Criterion Collection.
    Return only the director names in a list format.

    If more than 8 directors are mentioned, extract only the first 8.
    If fewer than 8 directors are mentioned, extract all available ones.
    """


def prompt_extract_closet_pick_url(director_name: str) -> str:
    """
    Returns a prompt for extracting the Closet Pick URL for a specific director.

    Args:
        director_name (str): The name of the director to extract the URL for.

    Returns:
        str: The extraction prompt for Closet Pick URLs.
    """
    return f"""
    Extract the URL to the Criterion Collection Closet Pick featuring {director_name}.
    
    IMPORTANT: 
    - Only extract URLs from the official Criterion Collection website (www.criterion.com or criterion.com)
    
    Return only the complete URL, including the protocol (http:// or https://).
    If the URL is incomplete (missing protocol), prepend 'https://'.
    If multiple URLs are mentioned for this director, extract only the one that leads to their Closet Pick on the official Criterion Collection website.
    If no official Criterion Collection URL is provided for this director, return null.
    """


def prompt_extract_film_info(director_name: str) -> str:
    """
    Returns a prompt for extracting information about a film recommended in a Closet Pick.

    Args:
        director_name (str): The name of the director whose film recommendation to extract.

    Returns:
        str: The extraction prompt for film information.
    """
    return f"""
    Extract information about one film recommended by {director_name} in their Closet Pick:
    - The film title
    - The film's availability status on the Criterion Collection website (if mentioned)
    - The price of the cheapest option for this film (if mentioned)

    Return null for any fields that are not mentioned in the answer.
    If multiple films are mentioned for this director, extract information only for the first one.
    """


def prompt_extract_film_url(director_name: str, film_title: str) -> str:
    """
    Returns a prompt for extracting the URL to a specific film on the Criterion website.

    Args:
        director_name (str): The name of the director who recommended the film.
        film_title (str): The title of the film to extract the URL for.

    Returns:
        str: The extraction prompt for film URLs.
    """
    return f"""
    Extract the URL to the film '{film_title}' recommended by {director_name} on the Criterion Collection website.
    
    IMPORTANT:
    - Only extract URLs from the official Criterion Collection website (www.criterion.com or criterion.com)
    - The URL should lead to the film's page on the Criterion Collection website
    
    Return only the complete URL, including the protocol (http:// or https://).
    If the URL is incomplete (missing protocol), prepend 'https://'.
    If no official Criterion Collection URL is provided for this film, return null.
    """


# --------------------------------------------------------------------------- #
# Information extraction functions                                           #
# --------------------------------------------------------------------------- #
async def extract_closet_picks_info(evaluator: Evaluator) -> List[ClosetPickInfo]:
    """
    Extract all Closet Pick information from the answer.
    Only extracts director names, URLs, and film info - NO DATE EXTRACTION.

    Args:
        evaluator (Evaluator): The evaluator object containing the extractor.

    Returns:
        List[ClosetPickInfo]: List of extracted Closet Pick information.
    """
    # 1. Extract directors first
    directors_info = await evaluator.extract(
        prompt=prompt_extract_directors(),
        template_class=DirectorsList,
        extraction_name="directors_list"
    )

    # Limit to 8 directors
    directors = directors_info.directors[:8]

    # 2. Extract details for each director
    closet_picks = []

    for director_name in directors:
        # Extract Closet Pick URL
        url_info = await evaluator.extract(
            prompt=prompt_extract_closet_pick_url(director_name),
            template_class=ClosetPickURLs,
            extraction_name=f"closet_pick_url_{director_name}"
        )

        closet_pick_url = url_info.urls[0] if url_info.urls else ""

        # Extract film information
        film_info = await evaluator.extract(
            prompt=prompt_extract_film_info(director_name),
            template_class=FilmInfo,
            extraction_name=f"film_info_{director_name}"
        )

        # Extract film URL if we have a film title
        film_url = None
        if film_info.title:
            url_result = await evaluator.extract(
                prompt=prompt_extract_film_url(director_name, film_info.title),
                template_class=FilmURL,
                extraction_name=f"film_url_{director_name}_{film_info.title}"
            )
            film_url = url_result.url

        # Create Closet Pick info object (NO DATE FIELD)
        closet_pick = ClosetPickInfo(
            director_name=director_name,
            closet_pick_url=closet_pick_url,
            film_title=film_info.title,
            film_availability=film_info.availability,
            film_price=film_info.price,
            film_url=film_url,
        )

        closet_picks.append(closet_pick)

    # Pad with empty objects if fewer than 8
    while len(closet_picks) < 8:
        closet_picks.append(ClosetPickInfo())

    return closet_picks


# --------------------------------------------------------------------------- #
# Verification functions for Step 1: Valid Closet Pick                      #
# --------------------------------------------------------------------------- #
async def verify_closet_pick_step1(
        evaluator: Evaluator,
        parent_node,
        closet_pick: ClosetPickInfo,
        index: int,
) -> None:
    """
    Verify Step 1 requirements for a Closet Pick (exists, features director, recent).

    Args:
        evaluator (Evaluator): The evaluator object.
        parent_node: The parent node to attach verifications to.
        closet_pick (ClosetPickInfo): The Closet Pick information to verify.
        index (int): The index of this Closet Pick in the list.
    """
    # Combined existence check for critical fields
    exists_node = evaluator.add_custom_node(
        result=bool(closet_pick.director_name and closet_pick.closet_pick_url),
        id=f"closet_pick_{index}_has_required_fields",
        desc=f"Check if Closet Pick #{index} has both director name and URL",
        parent=parent_node,
        critical=True
    )

    # 1. Verify it's a valid Criterion Collection Closet Pick
    valid_pick_node = evaluator.add_leaf(
        id=f"closet_pick_{index}_valid",
        desc=f"Closet Pick #{index} featuring {closet_pick.director_name or 'Unknown'} is a valid Criterion Collection Closet Pick",
        parent=parent_node,
        critical=True,
    )

    claim = f"This webpage is an authentic Criterion Collection Closet Pick featuring {closet_pick.director_name} as a guest selecting films from the Criterion Collection."
    additional_instruction = "Verify that this is a genuine Closet Pick from the Criterion Collection, where a guest selects films from their closet. The webpage should be from the official Criterion Collection website and should clearly identify this as a Closet Pick."

    await evaluator.verify(
        claim=claim,
        node=valid_pick_node,
        sources=closet_pick.closet_pick_url,
        additional_instruction=additional_instruction,
    )

    # 2. Verify the guest is a director
    director_node = evaluator.add_leaf(
        id=f"closet_pick_{index}_is_director",
        desc=f"The guest {closet_pick.director_name or 'Unknown'} is specifically a film director",
        parent=parent_node,
        critical=True,
    )

    claim = f"{closet_pick.director_name} is a film director who is featured in this Criterion Collection Closet Pick."
    additional_instruction = "Verify that the person is primarily known as a film director, not just an actor, producer, writer, or other type of artist. The person should have directed at least one feature film or documentary."

    await evaluator.verify(
        claim=claim,
        node=director_node,
        sources=closet_pick.closet_pick_url,
        additional_instruction=additional_instruction,
    )

    # 3. Verify it was published within the past 2 years
    recent_node = evaluator.add_leaf(
        id=f"closet_pick_{index}_recent",
        desc=f"Closet Pick #{index} was published within the past 2 years",
        parent=parent_node,
        critical=True,
    )

    claim = f"The {closet_pick.director_name}'s Closet Picks was published within the past 2 years (on or after {TWO_YEARS_AGO.strftime('%B %Y')})."
    additional_instruction = f"""
    Carefully examine the webpage to determine when this Closet Pick was published.
    The Closet Pick must have been published on or after {TWO_YEARS_AGO.strftime('%B %Y')} to qualify as 'within the past 2 years'.
    Today's date for reference is {CURRENT_DATE.strftime('%B %Y')}.
    """

    await evaluator.verify(
        claim=claim,
        node=recent_node,
        sources=CRITERION_CLOSET_PICKS_URL,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Verification functions for Step 2: Film Information                       #
# --------------------------------------------------------------------------- #
async def verify_closet_pick_step2(
        evaluator: Evaluator,
        parent_node,
        closet_pick: ClosetPickInfo,
        index: int,
) -> None:
    """
    Verify Step 2 requirements for a Closet Pick (film recommendation and pricing).

    Args:
        evaluator (Evaluator): The evaluator object.
        parent_node: The parent node to attach verifications to.
        closet_pick (ClosetPickInfo): The Closet Pick information to verify.
        index (int): The index of this Closet Pick in the list.
    """
    # Check if film title exists
    film_exists_node = evaluator.add_custom_node(
        result=bool(closet_pick.film_title),
        id=f"closet_pick_{index}_film_title_exists",
        desc=f"Check if a film title was provided for Closet Pick #{index}",
        parent=parent_node,
        critical=True
    )

    # Verify the director recommends this film
    film_rec_node = evaluator.add_leaf(
        id=f"closet_pick_{index}_film_recommended",
        desc=f"Director {closet_pick.director_name or 'Unknown'} recommends the film '{closet_pick.film_title or 'Unknown'}'",
        parent=parent_node,
        critical=True,
    )

    claim = f"In this Closet Pick, {closet_pick.director_name} recommends or discusses the film '{closet_pick.film_title}'."
    additional_instruction = f"Verify that {closet_pick.director_name} specifically mentions, recommends, or discusses the film '{closet_pick.film_title}' in this Closet Pick."

    await evaluator.verify(
        claim=claim,
        node=film_rec_node,
        sources=closet_pick.closet_pick_url,
        additional_instruction=additional_instruction,
    )

    # Availability and price verification (sequential)
    avail_price_node = evaluator.add_sequential(
        id=f"closet_pick_{index}_availability_price",
        desc=f"Sequential verification of availability and price for film '{closet_pick.film_title or 'Unknown'}'",
        parent=parent_node,
        critical=False,
    )

    # Build sources list for verification
    sources = []
    if closet_pick.film_url:
        sources.append(closet_pick.film_url)

    # Availability verification
    avail_exists_node = evaluator.add_custom_node(
        result=bool(closet_pick.film_availability),
        id=f"closet_pick_{index}_availability_exists",
        desc=f"Check if availability information was provided",
        parent=avail_price_node,
        critical=True
    )

    avail_node = evaluator.add_leaf(
        id=f"closet_pick_{index}_availability_correct",
        desc=f"Film availability status '{closet_pick.film_availability or 'Unknown'}' is accurate",
        parent=avail_price_node,
        critical=True,
    )

    claim = f"The film '{closet_pick.film_title}' has availability status '{closet_pick.film_availability}' on the Criterion Collection website."
    additional_instruction = f"Verify that the availability status '{closet_pick.film_availability}' is accurate for the film '{closet_pick.film_title}' on the Criterion Collection website. Check if the film is currently available in the stated format."

    await evaluator.verify(
        claim=claim,
        node=avail_node,
        sources=sources if sources else None,
        additional_instruction=additional_instruction,
    )

    # Price verification
    # Use the actual price or default to 'N/A'
    price_value = closet_pick.film_price if closet_pick.film_price else "N/A"

    price_exists_node = evaluator.add_custom_node(
        result=True,  # Always true since we default to 'N/A'
        id=f"closet_pick_{index}_price_exists",
        desc=f"Check if price information was provided or defaulted to N/A",
        parent=avail_price_node,
        critical=True
    )

    price_node = evaluator.add_leaf(
        id=f"closet_pick_{index}_price_correct",
        desc=f"Film price '{price_value}' represents the cheapest option",
        parent=avail_price_node,
        critical=True,
    )

    claim = f"The film '{closet_pick.film_title}' has the cheapest option priced at {price_value} on the Criterion Collection website."
    additional_instruction = f"""
    Verify that the price '{price_value}' is accurate for the film '{closet_pick.film_title}' on the Criterion Collection website.

    Specifically check that:
    1. This price exists for this film on the Criterion Collection website
    2. This is indeed the CHEAPEST option available for this film
    3. The price is current and accurate

    If there are multiple formats (Blu-ray, DVD, digital, etc.), ensure this is the lowest price among all available options. A special case is if the film is not available for purchase, in which case the price should be 'N/A' or similar.
    """

    await evaluator.verify(
        claim=claim,
        node=price_node,
        sources=sources if sources else None,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Two-step sequential verification                                           #
# --------------------------------------------------------------------------- #
async def verify_single_closet_pick_two_step(
        evaluator: Evaluator,
        parent_node,
        closet_pick: ClosetPickInfo,
        index: int,
) -> None:
    """
    Two-step sequential verification for a single Closet Pick.

    Args:
        evaluator (Evaluator): The evaluator object.
        parent_node: The parent node to attach this verification to.
        closet_pick (ClosetPickInfo): The Closet Pick information to verify.
        index (int): The index of this Closet Pick in the list.
    """
    pick_node = evaluator.add_sequential(
        id=f"closet_pick_{index}",
        desc=f"Closet Pick #{index}: {closet_pick.director_name or 'Missing'} - Two-step verification",
        parent=parent_node,
    )

    # Step 1: Critical gateway - must pass to proceed to Step 2
    step1 = evaluator.add_parallel(
        id=f"step1_valid_closet_pick_{index}",
        desc=f"Step 1: Verify {closet_pick.director_name or 'Missing'}'s Closet Pick is valid, recent, and URL-supported",
        parent=pick_node,
        critical=True,  # This is the critical gateway
    )

    await verify_closet_pick_step1(evaluator, step1, closet_pick, index)

    # Step 2: Film information - allows partial credit
    step2 = evaluator.add_parallel(
        id=f"step2_film_info_{index}",
        desc=f"Step 2: Verify film recommendation and pricing accuracy",
        parent=pick_node,
        critical=False,  # Non-critical, allows partial scoring
    )

    await verify_closet_pick_step2(evaluator, step2, closet_pick, index)


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
    Evaluate the answer to the Criterion Collection Closet Picks task.

    This function implements a two-step sequential verification:
    1. Verify each Closet Pick is valid, recent, and features a director
    2. Verify film recommendations and pricing accuracy

    Key improvements:
    - No date extraction from answer text
    - Time verification done entirely through URL content checking
    - Two-step sequential logic with proper critical gateway
    - Allows partial scoring for film information

    Args:
        client: The LLM client for making API calls.
        answer (str): The answer text to evaluate.
        agent_name (str): Name of the agent that produced the answer.
        answer_name (str): Name/identifier for this specific answer.
        cache (CacheFileSys): Cache object for storing web content.
        semaphore (asyncio.Semaphore): Semaphore for controlling concurrent API calls.
        logger (logging.Logger): Logger for recording evaluation progress.
        model (str, optional): The model to use for evaluation. Defaults to "o4-mini".

    Returns:
        Dict: Evaluation results including the final score and breakdown.
    """
    # -------- 1. Set up evaluator --------------------------------------- #
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
        default_model=model
    )

    # -------- 2. Extract information (NO DATE EXTRACTION) --------------- #
    closet_picks = await extract_closet_picks_info(evaluator)

    # -------- 3. Build verification tree -------------------------------- #
    # Create a node for the set of 8 Closet Picks (allowing partial credit)
    closet_picks_node = evaluator.add_parallel(
        id="closet_picks",
        desc="Verification of 8 Criterion Collection Closet Picks featuring directors from the past 2 years",
        critical=False,
    )

    # Verify each Closet Pick using the two-step sequential logic
    for i, pick in enumerate(closet_picks, 1):
        await verify_single_closet_pick_two_step(evaluator, closet_picks_node, pick, i)

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()