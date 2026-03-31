import asyncio
import logging
from typing import List, Dict, Optional

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "moroccan_couscous"
TASK_DESCRIPTION = """
Find three restaurants in Manhattan, NYC that serve couscous with beef, lamb, or chicken, where the estimated cost per person for the restaurant is no more than $30 or two dollar signs ("$$") as shown on Google Maps. Provide the restaurant names, addresses, the eligible couscous dishes offered for each restaurant, a link to their Google Maps page, and a link to their menu.
"""

JUDGE_MODEL = "o4-mini"

# --------------------------------------------------------------------------- #
# Data models for extracted info                                             #
# --------------------------------------------------------------------------- #
class RestaurantName(BaseModel):
    """Simple model for restaurant name."""
    name: Optional[str] = None


class RestaurantNames(BaseModel):
    """List of restaurant names extracted from the answer."""
    restaurants: List[RestaurantName] = Field(default_factory=list)


class CouscousDish(BaseModel):
    """Information about a specific couscous dish."""
    name: Optional[str] = None
    protein: Optional[str] = None  # beef, lamb, or chicken


class Restaurant(BaseModel):
    """Information about a restaurant."""
    name: Optional[str] = None
    address: Optional[str] = None
    couscous_dishes: List[CouscousDish] = Field(default_factory=list)
    google_maps_url: Optional[str] = None
    menu_url: Optional[str] = None
    other_urls: List[str] = Field(default_factory=list)  # Added to store additional URLs


class RestaurantDetails(BaseModel):
    """Detailed information about a specific restaurant."""
    address: Optional[str] = None
    couscous_dishes: List[CouscousDish] = Field(default_factory=list)


class UrlExtraction(BaseModel):
    """URLs extracted for verification of a specific restaurant."""
    google_maps_url: Optional[str] = None
    menu_url: Optional[str] = None
    other_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                         #
# --------------------------------------------------------------------------- #
def prompt_extract_restaurant_names() -> str:
    return """
    Extract the names of all restaurants mentioned in the answer that serve couscous dishes.
    
    Provide only the restaurant names in a list. Do not include any other information.
    Extract information exactly as it appears in the answer.
    Do not invent or infer information not explicitly stated in the answer.
    """


def prompt_extract_restaurant_details(restaurant_name: str) -> str:
    return f"""
    For the restaurant named "{restaurant_name}", extract the following details from the answer:
    
    1. The restaurant address
    2. All mentioned couscous dishes with beef, lamb, or chicken (name of the dish and which protein it contains)
    
    Only extract information that is explicitly mentioned for this specific restaurant.
    If any information is missing, mark it as null.
    """


def prompt_extract_urls_for_restaurant(restaurant_name: str) -> str:
    return f"""
    For the restaurant named "{restaurant_name}", extract all URLs mentioned in the answer that are associated with this restaurant.
    
    Specifically extract:
    1. The Google Maps URL (if present)
    2. The menu URL (if present)
    3. Any other URLs associated with this restaurant (put these in the other_urls list)
    
    Only extract URLs that are explicitly mentioned for this specific restaurant. If no URL is provided for a particular field, mark it as null.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_restaurant(
    evaluator: Evaluator,
    parent_node,
    restaurant: Restaurant,
    idx: int,
) -> None:
    """
    Verify all aspects of a single restaurant.
    """
    # Create a parallel node for this restaurant
    restaurant_node = evaluator.add_parallel(
        id=f"restaurant_{idx}",
        desc=f"Verification of restaurant #{idx}: {restaurant.name or 'Unknown'}",
        parent=parent_node,
        critical=False,  # Allow partial credit across restaurants
    )
    
    # First: Check if basic restaurant info exists
    restaurant_exists = evaluator.add_custom_node(
        result=bool(restaurant.name) and bool(restaurant.google_maps_url) and bool(restaurant.menu_url) and bool(restaurant.address),
        id=f"restaurant_{idx}_basic_info_exists",
        desc=f"Check if restaurant #{idx} has name and both required URLs",
        parent=restaurant_node,
        critical=True
    )
    
    # Second: Verify Google Maps URL
    google_maps_node = evaluator.add_leaf(
        id=f"google_maps_url_{idx}",
        desc=f"Verify that the Google Maps URL for restaurant #{idx} is valid and points to the correct restaurant",
        parent=restaurant_node,
        critical=True,
    )
    
    claim = f"The webpage is a valid Google Maps page for the restaurant '{restaurant.name}'."
    await evaluator.verify(
        claim=claim,
        node=google_maps_node,
        sources=restaurant.google_maps_url,
        additional_instruction="Verify that this URL is a valid Google Maps link and that it points to the correct restaurant.",
    )
    
    # Third: Verify menu URL
    menu_node = evaluator.add_leaf(
        id=f"menu_url_{idx}",
        desc=f"Verify that the menu URL for restaurant #{idx} contains the restaurant's menu",
        parent=restaurant_node,
        critical=True,
    )
    
    claim = f"The webpage contains the menu for the restaurant '{restaurant.name}'."
    await evaluator.verify(
        claim=claim,
        node=menu_node,
        sources=restaurant.menu_url,
        additional_instruction="Verify that this URL is a valid menu link for the restaurant and shows their food offerings.",
    )
    
    # Fifth: Verify Manhattan location
    manhattan_node = evaluator.add_leaf(
        id=f"manhattan_location_{idx}",
        desc=f"Verify that restaurant #{idx} is located in Manhattan, NYC",
        parent=restaurant_node,
        critical=True,
    )
    
    claim = f"The restaurant '{restaurant.name}' with address '{restaurant.address}' is located in Manhattan, NYC."
    await evaluator.verify(
        claim=claim,
        node=manhattan_node,
        sources=restaurant.google_maps_url,
        additional_instruction="Verify from the Google Maps page that this restaurant is indeed located at the given address in Manhattan, NYC.",
    )
    

    # Seventh: Verify price - ONLY from Google Maps as per task requirements
    price_node = evaluator.add_leaf(
        id=f"price_verification_{idx}",
        desc=f"Verify that restaurant #{idx} has a price point no more than $30 or '$$' as shown on Google Maps",
        parent=restaurant_node,
        critical=True,
    )
    
    claim = f"The restaurant '{restaurant.name}' has a price point of no more than $30 or '$$' per person."
    await evaluator.verify(
        claim=claim,
        node=price_node,
        sources=restaurant.google_maps_url,  # Only Google Maps as per task
        additional_instruction="Check the Google Maps page to verify the price point of this restaurant is no more than $30 or '$$' per person. Look for price indicators shown on Google Maps ($ or $$ symbols).",
    )
    
    # Eighth: Check dishes completeness
    dishes_complete = evaluator.add_custom_node(
        result=bool(restaurant.couscous_dishes) and all(
            dish.name and dish.protein for dish in restaurant.couscous_dishes
        ),
        id=f"restaurant_{idx}_dishes_complete",
        desc=f"Check if couscous dishes with proteins are provided for restaurant #{idx}",
        parent=restaurant_node,
        critical=True
    )
    
    # Ninth: Verify couscous dishes - can use multiple sources
    couscous_node = evaluator.add_leaf(
        id=f"couscous_dishes_{idx}",
        desc=f"Verify that restaurant #{idx} serves the mentioned couscous dishes with beef, lamb, or chicken",
        parent=restaurant_node,
        critical=True,
    )

    # Always construct the claim the same way - the critical node handles empty cases
    dishes_text = ", ".join([f"{dish.name} with {dish.protein}" for dish in restaurant.couscous_dishes])
    claim = f"Restaurant '{restaurant.name}' offers the following couscous dishes: {dishes_text}."

    # For dishes, we can check menu URL and other URLs since task doesn't restrict this
    dishes_sources = []
    if restaurant.menu_url:
        dishes_sources.append(restaurant.menu_url)
    if restaurant.other_urls:
        dishes_sources.extend(restaurant.other_urls)

    await evaluator.verify(
        claim=claim,
        node=couscous_node,
        sources=dishes_sources,
        additional_instruction="Check the menu or restaurant website to verify that the restaurant offers the mentioned couscous dishes with beef, lamb, or chicken.",
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
    Evaluate the answer to the Moroccan couscous restaurant task.
    
    The task requires finding three restaurants in Manhattan, NYC that:
    1. Serve couscous with beef, lamb, or chicken
    2. Cost no more than $30 or "$$" per person as shown on Google Maps
    3. Provide restaurant name, address, eligible couscous dishes
    4. Include Google Maps and menu links
    
    We evaluate by:
    1. First extracting restaurant names from the answer
    2. Then extracting detailed information for each restaurant
    3. Verifying each restaurant with independent checks (parallel)
    4. Awarding partial credit based on how many valid restaurants are provided
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
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
        default_model=model if model else JUDGE_MODEL
    )

    # -------- 2. Extract restaurant names first -------------------------- #
    restaurant_names = await evaluator.extract(
        prompt=prompt_extract_restaurant_names(),
        template_class=RestaurantNames,
        extraction_name="restaurant_names"
    )
    
    # -------- 3. Extract detailed info for each restaurant --------------- #
    restaurants = []
    
    # Process each restaurant name and extract its details
    for restaurant_name_obj in restaurant_names.restaurants:
        if not restaurant_name_obj.name:
            continue
            
        restaurant_name = restaurant_name_obj.name
        
        # Create a new restaurant object
        restaurant = Restaurant(name=restaurant_name)
        
        # Extract URLs for this restaurant
        urls = await evaluator.extract(
            prompt=prompt_extract_urls_for_restaurant(restaurant_name),
            template_class=UrlExtraction,
            extraction_name=f"urls_for_{restaurant_name}"
        )
        restaurant.google_maps_url = urls.google_maps_url
        restaurant.menu_url = urls.menu_url
        restaurant.other_urls = urls.other_urls  # Store additional URLs
        
        # Extract other details for this restaurant
        details = await evaluator.extract(
            prompt=prompt_extract_restaurant_details(restaurant_name),
            template_class=RestaurantDetails,
            extraction_name=f"details_for_{restaurant_name}"
        )
        restaurant.address = details.address
        restaurant.couscous_dishes = details.couscous_dishes
        
        restaurants.append(restaurant)
    
    # -------- 4. Build verification tree -------------------------------- #
    # The task asks for three restaurants, so we'll verify up to three
    required_restaurants = 3
    verified_restaurants = restaurants[:required_restaurants]
    
    # If fewer than required restaurants were provided, add placeholders
    while len(verified_restaurants) < required_restaurants:
        verified_restaurants.append(Restaurant())
    
    # Verify each restaurant - directly connected to the root
    for idx, restaurant in enumerate(verified_restaurants, 1):
        await verify_restaurant(evaluator, root, restaurant, idx)

    # -------- 5. Return structured result ------------------------------- #
    return evaluator.get_summary()