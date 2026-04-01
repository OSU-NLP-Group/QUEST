import asyncio
import logging
from typing import Dict, List, Optional

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "music_price"
TASK_DESCRIPTION = """
For each of the following countries—Singapore, United Kingdom, Australia, Canada, New Zealand, and Ireland—visit the official Apple Music and Spotify websites to find the current subscription prices in local currencies for their Student and Individual plans. Exclude free trials, promotional offers, or any other temporary discounts. Include direct links to each country's official Apple Music and Spotify plan pricing pages.
"""

COUNTRIES = [
    "Singapore",
    "United Kingdom", 
    "Australia", 
    "Canada", 
    "New Zealand", 
    "Ireland"
]

SERVICES = ["Apple Music", "Spotify"]
PLANS = ["Student", "Individual"]

# --------------------------------------------------------------------------- #
# Data model for extracted service info                                       #
# --------------------------------------------------------------------------- #
class PlanPrice(BaseModel):
    """Price information for a specific plan"""
    price: Optional[str] = None
    currency: Optional[str] = None
    url: Optional[str] = None

class ServicePrices(BaseModel):
    """Price information for both plans of a service"""
    student: Optional[PlanPrice] = None
    individual: Optional[PlanPrice] = None

class CountryPrices(BaseModel):
    """Price information for both services in a country"""
    apple_music: Optional[ServicePrices] = None
    spotify: Optional[ServicePrices] = None

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_country_prices(country: str) -> str:
    return f"""
    Extract the subscription prices for Apple Music and Spotify for {country} from the given answer.

    For both Apple Music and Spotify in {country}, extract:
    1. Student plan price with currency
    2. Individual plan price with currency
    3. URLs to the official pricing pages

    IMPORTANT RULES FOR URL EXTRACTION:
    - If a specific URL is provided for the Student plan, include it in the Student plan data.
    - If a specific URL is provided for the Individual plan, include it in the Individual plan data.
    - If only a general pricing URL is provided for a service (not specific to any plan), include that same URL in BOTH the Student and Individual plans.
    - If multiple URLs are provided for the same service, extract the most relevant URL for each plan.

    If any information is missing, set the corresponding field to null.
    Extract only the price amounts and currencies exactly as they appear in the text, without attempting to normalize or standardize formats.
    For URLs, extract the complete URLs that link to the official pricing pages.
    """

# --------------------------------------------------------------------------- #
# Country-level verification                                                  #
# --------------------------------------------------------------------------- #
async def verify_country_prices(
        evaluator: Evaluator,
        parent_node,
        country: str,
        country_data: CountryPrices,
) -> None:
    """Verify the price information for a specific country"""
    
    country_node = evaluator.add_parallel(
        id=f"{country.lower().replace(' ', '_')}_verification",
        desc=f"Verification of {country} music subscription prices",
        parent=parent_node,
    )
    
    # Ensure we have service data even if extraction failed
    apple_music_data = country_data.apple_music if country_data else None
    if apple_music_data is None:
        apple_music_data = ServicePrices()
    
    spotify_data = country_data.spotify if country_data else None
    if spotify_data is None:
        spotify_data = ServicePrices()
    
    # Verify Apple Music
    await verify_service_prices(
        evaluator=evaluator,
        parent_node=country_node,
        country=country,
        service="Apple Music",
        service_data=apple_music_data,
    )
    
    # Verify Spotify
    await verify_service_prices(
        evaluator=evaluator,
        parent_node=country_node,
        country=country,
        service="Spotify",
        service_data=spotify_data,
    )

# --------------------------------------------------------------------------- #
# Service-level verification                                                  #
# --------------------------------------------------------------------------- #
async def verify_service_prices(
        evaluator: Evaluator,
        parent_node,
        country: str,
        service: str,
        service_data: ServicePrices,
) -> None:
    """
    Verify the price information for a specific service.
    Uses parallel verification with critical existence checks.
    REQUIRES BOTH Student and Individual plans to be present.
    """
    
    service_id = service.lower().replace(" ", "_")
    service_node = evaluator.add_parallel(
        id=f"{country.lower().replace(' ', '_')}_{service_id}_verification",
        desc=f"Verification of {service} prices for {country}",
        parent=parent_node,
    )
    
    # Collect URLs from service data
    urls = []
    if service_data.student and service_data.student.url:
        urls.append(service_data.student.url)
    if service_data.individual and service_data.individual.url:
        urls.append(service_data.individual.url)
    
    # Remove duplicates while preserving order
    urls = list(dict.fromkeys(urls))
    
    # Critical existence check for BOTH student AND individual data with URLs
    service_exists = evaluator.add_custom_node(
        result=(
            bool(urls) and 
            service_data.student is not None and 
            service_data.student.price is not None and
            service_data.individual is not None and 
            service_data.individual.price is not None
        ),
        id=f"{country.lower().replace(' ', '_')}_{service_id}_exists",
        desc=f"Check if BOTH student and individual {service} data with URLs exist for {country}",
        parent=service_node,
        critical=True
    )
    
    # Verify URL validity - enhanced to check for direct pricing page
    url_node = evaluator.add_leaf(
        id=f"{country.lower().replace(' ', '_')}_{service_id}_url_verification",
        desc=f"Verification of direct {service} pricing page URL for {country}",
        parent=service_node,
        critical=True
    )
    
    url_claim = f"This URL is a direct link to the official {service} pricing page for {country}, where the subscription prices for Student and Individual plans are clearly displayed"
    
    await evaluator.verify(
        claim=url_claim,
        node=url_node,
        sources=urls,
        additional_instruction="Verify that this is a direct pricing page (not a general homepage or support page) where the actual subscription prices for Student and Individual plans are visible. The page should clearly show the pricing information for the specific country."
    )
    
    # Verify student plan price (critical since both plans are required)
    student_node = evaluator.add_leaf(
        id=f"{country.lower().replace(' ', '_')}_{service_id}_student_price",
        desc=f"Verification of {service} Student plan price for {country}",
        parent=service_node,
        critical=True  # Critical because both plans are required
    )
    
    student_price_info = ""
    if service_data.student and service_data.student.price:
        student_price_info = service_data.student.price
        if service_data.student.currency:
            student_price_info = f"{student_price_info} {service_data.student.currency}"
    
    student_claim = f"The {service} Student plan in {country} costs {student_price_info}, expressed in the local currency."
    await evaluator.verify(
        claim=student_claim,
        node=student_node,
        sources=urls,
    )
    
    # Verify individual plan price (critical since both plans are required)
    individual_node = evaluator.add_leaf(
        id=f"{country.lower().replace(' ', '_')}_{service_id}_individual_price",
        desc=f"Verification of {service} Individual plan price for {country}",
        parent=service_node,
        critical=True  # Critical because both plans are required
    )
    
    individual_price_info = ""
    if service_data.individual and service_data.individual.price:
        individual_price_info = service_data.individual.price
        if service_data.individual.currency:
            individual_price_info = f"{individual_price_info} {service_data.individual.currency}"
    
    individual_claim = f"The {service} Individual plan for {country} costs {individual_price_info}, expressed in the local currency."
    await evaluator.verify(
        claim=individual_claim,
        node=individual_node,
        sources=urls,
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
        default_model=model
    )

    # -------- 2. Extract and verify each country separately ------------- #
    for country in COUNTRIES:
        logger.info(f"Extracting and verifying data for {country}")
        
        # Extract data for this specific country only
        country_data = await evaluator.extract(
            prompt=prompt_extract_country_prices(country),
            template_class=CountryPrices,
            extraction_name=f"{country}_prices"
        )
        
        # Verify this country's data
        await verify_country_prices(
            evaluator=evaluator,
            parent_node=root,
            country=country,
            country_data=country_data,
        )

    # -------- 3. Return structured result ------------------------------- #
    return evaluator.get_summary()