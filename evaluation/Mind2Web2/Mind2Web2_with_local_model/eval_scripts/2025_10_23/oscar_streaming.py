import asyncio
import logging
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "oscar_streaming"
TASK_DESCRIPTION = """
Find the list of all the nominees for Best Picture at last year's Academy Awards (which honored films released the year before). For each nominated film, determine at least one streaming platform where it is currently available (in the US). Then, for each of the mentioned streaming platforms, find the current lowest monthly price for a basic subscription from their official websites (excluding free trial offers, student plan and promotional offers).
"""

EVALUATION_INSTRUCTIONS = """
From 2021, Best Picture award has a set number of ten nominees.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class StreamingPlatform(BaseModel):
    name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class StreamingPlatformPrice(BaseModel):
    platform: Optional[str] = None
    monthly_price: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class NominatedFilm(BaseModel):
    title: Optional[str] = None
    nomination_urls: List[str] = Field(default_factory=list)
    streaming_platforms: List[StreamingPlatform] = Field(default_factory=list)


class NominatedFilmsData(BaseModel):
    films: List[NominatedFilm] = Field(default_factory=list)


class FilmStreamingInfo(BaseModel):
    platforms: List[StreamingPlatform] = Field(default_factory=list)


class PlatformPricingData(BaseModel):
    pricing_info: List[StreamingPlatformPrice] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_nominees() -> str:
    return """
    Extract all the films that were nominated for Best Picture at last year's Academy Awards according to the answer text.
    For each film, include:
    1. The title of the film
    2. Any URLs provided that potentially could support that this film was nominated (e.g., a link for all the nominees, or dedicated links for each film.)

    Return all films in the 'films' field as a list.
    """


def prompt_extract_film_streaming_platforms(film_title: str) -> str:
    return f"""
    For the film "{film_title}", extract all streaming platforms where it is available according to the answer.
    For each streaming platform, include:
    1. The name of the platform
    2. Any URLs provided that verify the film is available on this platform

    If no URLs are provided for a platform, include an empty list for urls.
    If no streaming platforms are mentioned for this film, return an empty list.
    Return all platforms in the 'platforms' field as a list.
    """


def prompt_extract_platform_prices() -> str:
    return """
    Extract information about the monthly subscription prices for all streaming platforms mentioned in the answer.
    For each platform, include:
    1. The name of the platform
    2. The current lowest monthly price for a basic subscription (excluding free trials, student plans, and promotional offers)
    3. Any URLs provided that potentially can verify this pricing information, or indicated by the answer that are used to show the pricing information.

    It's important to include only the URLs that point to the official website of the platform where the pricing information is shown.

    If the price or URLs are not mentioned for a platform, include an empty list for urls or null for the price.
    Return all pricing information in the 'pricing_info' field as a list.
    """


# --------------------------------------------------------------------------- #
# Verification functions for a single film                                    #
# --------------------------------------------------------------------------- #
async def verify_film(
        evaluator: Evaluator,
        parent_node,
        film: NominatedFilm,
        platform_prices: Dict[str, StreamingPlatformPrice],
        film_index: int,
) -> None:
    """
    Verify all aspects of a single film using sequential logic.
    """
    film_title = film.title if film.title else f"Missing film #{film_index}"

    film_node = evaluator.add_sequential(
        id=f"film_{film_index}",
        desc=f"Sequential verification of film #{film_index}: '{film_title}'",
        parent=parent_node,
        critical=False,
    )

    # Step 1: Nomination verification (non-critical to allow partial credit)
    nomination_node = evaluator.add_parallel(
        id=f"nomination_{film_index}",
        desc=f"Verification that '{film_title}' was nominated for Best Picture with proper source",
        parent=film_node,
        critical=False,  # Non-critical to allow partial credit
    )

    # Combined existence check for nomination
    nomination_exists = evaluator.add_custom_node(
        result=(film.title is not None and film.title.strip() != "") and bool(film.nomination_urls),
        id=f"nomination_exists_{film_index}",
        desc=f"Check if film title and nomination URLs were provided for film #{film_index}",
        parent=nomination_node,
        critical=True
    )

    # Verify nomination accuracy
    nomination_accuracy_node = evaluator.add_leaf(
        id=f"nomination_accuracy_{film_index}",
        desc=f"Verification that '{film.title}' was actually nominated for Best Picture",
        parent=nomination_node,
        critical=True,
    )

    nomination_claim = f"The film '{film.title}' was nominated for Best Picture at last year's Academy Awards"
    await evaluator.verify(
        claim=nomination_claim,
        node=nomination_accuracy_node,
        sources=film.nomination_urls,
        additional_instruction="Verify if this film was nominated for Best Picture at the Academy Awards ceremony that took place last year. Note that 'last year's Academy Awards' refers to the ceremony held in the previous calendar year, which honored films from the year before that."
    )

    # Step 2: Streaming platform verification (non-critical to allow partial credit)
    streaming_node = evaluator.add_parallel(
        id=f"streaming_{film_index}",
        desc=f"Verification that '{film.title}' has at least one streaming platform with proper source",
        parent=film_node,
        critical=False,  # Non-critical to allow partial credit
    )

    # Combined existence check for streaming
    platform = film.streaming_platforms[0] if film.streaming_platforms else StreamingPlatform()
    streaming_exists = evaluator.add_custom_node(
        result=bool(film.streaming_platforms) and bool(platform.name) and bool(platform.urls),
        id=f"streaming_exists_{film_index}",
        desc=f"Check if streaming platform with URLs was provided for '{film.title}'",
        parent=streaming_node,
        critical=True
    )

    # Verify platform availability
    platform_accuracy_node = evaluator.add_leaf(
        id=f"platform_accuracy_{film_index}",
        desc=f"Verification that '{film.title}' is actually available on {platform.name if platform.name else 'the platform'}",
        parent=streaming_node,
        critical=True,
    )

    availability_claim = f"The film '{film.title}' is available to stream on {platform.name if platform.name else 'the mentioned platform'} in the US"
    await evaluator.verify(
        claim=availability_claim,
        node=platform_accuracy_node,
        sources=platform.urls,
        additional_instruction="Verify if this film is currently available to stream on the mentioned platform in the United States. If there's no specific information about the region, just assume it's showing information for the US market by default. But, if you don't find the information from the webpage text, please look carefully from the screenshot. There could be icons of platforms on the page showing its availability. For example, the icons in the 'STREAMING' section on IMDb"
    )

    # Step 3: Pricing verification (non-critical by default)
    pricing_node = evaluator.add_parallel(
        id=f"pricing_{film_index}",
        desc=f"Verification of pricing information for streaming platform",
        parent=film_node,
        critical=False,
    )

    # Find matching price info
    platform_price = None
    if platform.name:
        platform_name_lower = platform.name.lower()
        for price in platform_prices.values():
            if price.platform:
                price_name_lower = price.platform.lower()
                if (price_name_lower == platform_name_lower or
                        platform_name_lower in price_name_lower or
                        price_name_lower in platform_name_lower):
                    platform_price = price
                    break

    # Combined existence check for pricing
    pricing_exists = evaluator.add_custom_node(
        result=(platform_price is not None and 
                platform_price.platform is not None and 
                platform_price.monthly_price is not None and 
                bool(platform_price.urls)),
        id=f"pricing_exists_{film_index}",
        desc=f"Check if pricing information with URLs was provided for {platform.name if platform.name else 'the platform'}",
        parent=pricing_node,
        critical=True
    )

    # Verify price accuracy
    price_accuracy_node = evaluator.add_leaf(
        id=f"price_accuracy_{film_index}",
        desc=f"Verification that the price for {platform.name if platform.name else 'the platform'} is accurate: {platform_price.monthly_price if platform_price else 'N/A'}",
        parent=pricing_node,
        critical=True,
    )

    price_claim = f"The current lowest monthly price for a basic subscription to {platform.name if platform.name else 'the platform'} is {platform_price.monthly_price if platform_price else 'unknown'} (excluding free trials, student plans, and promotional offers)"
    await evaluator.verify(
        claim=price_claim,
        node=price_accuracy_node,
        sources=platform_price.urls if platform_price else [],
        additional_instruction="Verify if the provided monthly price for the streaming service is accurate according to its official website. The source must be the platform's official website, and the price must be for the basic subscription, excluding any free trials, discounts, or promotional offers. Third-party websites like news articles or price comparison sites are not considered valid sources for pricing information. Minor variations in the price (e.g., $9.99 or $9.9 vs $10) are acceptable as long as they are within a reasonable range."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate a single answer to the Academy Awards streaming task.
    """
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
        extract_model=model,
        verify_model=model
    )

    # 1. Extract the nominated films
    nominated_films_data = await evaluator.extract(
        prompt=prompt_extract_nominees(),
        template_class=NominatedFilmsData,
        extraction_name="nominated_films"
    )

    nominated_films = nominated_films_data.films

    # 2. For each film, extract streaming platforms
    for film in nominated_films:
        if not film.title:
            continue

        film_platforms_data = await evaluator.extract(
            prompt=prompt_extract_film_streaming_platforms(film.title),
            template_class=FilmStreamingInfo,
            extraction_name=f"streaming_platforms_for_{film.title}"
        )
        film.streaming_platforms = film_platforms_data.platforms

    # 3. Extract platform pricing information
    platform_pricing_data = await evaluator.extract(
        prompt=prompt_extract_platform_prices(),
        template_class=PlatformPricingData,
        extraction_name="platform_pricing",
        additional_instruction="Make sure to extract only URLs that point to the official websites of the streaming platforms for pricing information. Third-party websites are not valid sources for pricing."
    )

    # Create a dictionary for easy lookup
    platform_prices_dict = {}
    for price in platform_pricing_data.pricing_info:
        if price.platform:
            platform_prices_dict[price.platform.lower()] = price

    # Pad films to ensure we have exactly 10
    while len(nominated_films) < 10:
        nominated_films.append(NominatedFilm())

    # Create film verification nodes for all 10 expected films
    for i in range(10):
        await verify_film(
            evaluator=evaluator,
            parent_node=root,
            film=nominated_films[i],
            platform_prices=platform_prices_dict,
            film_index=i + 1,
        )

    # Return structured result
    return evaluator.get_summary()