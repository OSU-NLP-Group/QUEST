import asyncio
import logging
from typing import Optional, List, Dict, Any
import re
from pydantic import BaseModel, Field

from mind2web2.eval_toolkit import create_evaluator
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import Evaluator
from mind2web2.verification_tree import AggregationStrategy
from mind2web2.api_tools import tool_googlemap


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "manhattan_restaurants"
TASK_DESCRIPTION = """
My friends and I are planning a culinary tour in Manhattan, New York, to experience cuisines from different countries. Your task is to help us find three restaurants on Manhattan, each representing a different country's cuisine. Please include the address, the type of cuisine, and the yelp page for each restaurant.

The restaurants should meet the following criteria:

Each restaurant must have a rating of at least 4 stars on Yelp.
The distance between any two consecutive restaurants in the itinerary should not exceed 3 miles of walking distance.
"""

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class RestaurantName(BaseModel):
    name: Optional[str] = None

class RestaurantNames(BaseModel):
    restaurants: List[RestaurantName] = Field(default_factory=list)

class Restaurant(BaseModel):
    name: Optional[str] = None
    cuisine: Optional[str] = None
    address: Optional[str] = None
    yelp_url: Optional[str] = None

class RestaurantURLs(BaseModel):
    urls: List[str] = Field(default_factory=list)

class DistancePair(BaseModel):
    restaurant1_name: Optional[str]
    restaurant2_name: Optional[str]
    distance_miles: Optional[float]

class DistancePairs(BaseModel):
    pairs: List[DistancePair] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_restaurant_names() -> str:
    return """
    Please extract the names of all restaurants mentioned in the answer.
    
    Create a list of restaurant names in the order they appear in the answer.
    If a restaurant doesn't have a name clearly mentioned, put null for that entry.
    """

def prompt_extract_restaurant_details(restaurant_name: str) -> str:
    return f"""
    Please extract detailed information about the restaurant named '{restaurant_name}' from the answer.
    
    Extract the following information:
    1. Name of the restaurant (should be {restaurant_name})
    2. Cuisine type or country of origin for the cuisine
    3. Full address
    4. Yelp URL (if provided)
    
    If any field is not provided in the answer, set it to null.
    """

def prompt_extract_restaurant_urls(restaurant_name: str) -> str:
    return f"""
    Please extract all URLs mentioned in the answer that are associated with the restaurant '{restaurant_name}'.
    
    This should include:
    1. The Yelp URL for the restaurant
    2. Any other URLs that provide information about this restaurant
    3. Any URLs that might be used to substantiate claims about this restaurant
    
    Return a list of all relevant URLs. If no URLs are mentioned for this restaurant, return an empty list.
    """

def prompt_extract_distance_info() -> str:
    return """
    Please extract information about the walking distances between consecutive restaurants mentioned in the answer.
    
    For each pair of consecutive restaurants, extract:
    1. The name of the first restaurant
    2. The name of the second restaurant
    3. The walking distance in miles between them
    
    If any field is not mentioned in the answer, set it to null.
    If distance is mentioned but not in miles, convert it to miles if possible.
    """

# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_restaurant_criteria(
        evaluator: Evaluator,
        root_node,
        restaurant: Restaurant,
        restaurant_urls: RestaurantURLs,
        restaurant_index: int
) -> None:
    """
    Verify that a restaurant meets all the criteria specified in the task.
    """
    # Create restaurant node directly under root
    restaurant_node = evaluator.add_sequential(
        id=f"restaurant_{restaurant_index}",
        desc=f"Restaurant {restaurant_index+1}: '{restaurant.name or 'Unknown'}' meets all required criteria",
        parent=root_node,
        critical=False  # Each restaurant is non-critical for partial credit
    )
    
    # 1. Check completeness of information
    completeness_node = evaluator.add_custom_node(
        result=(
            bool(restaurant.name) and
            bool(restaurant.address) and
            bool(restaurant.cuisine) and
            bool(restaurant.yelp_url)
        ),
        id=f"restaurant_{restaurant_index}_completeness",
        desc=f"Restaurant {restaurant_index+1} has all required information (name, address, cuisine, Yelp URL)",
        parent=restaurant_node,
        critical=True
    )
    
    # 2. Verify address accuracy
    address_verify = evaluator.add_leaf(
        id=f"restaurant_{restaurant_index}_address_verify",
        desc="Verify address is accurate and in Manhattan",
        parent=restaurant_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The restaurant '{restaurant.name}' is located at {restaurant.address} in Manhattan, New York",
        node=address_verify,
        sources=restaurant_urls.urls,
        additional_instruction="Verify both that the address is accurate for this restaurant AND that it is specifically in Manhattan (not just New York City)."
    )

    # 3. Verify Yelp URL corresponds to the restaurant
    yelp_match_verify = evaluator.add_leaf(
        id=f"restaurant_{restaurant_index}_yelp_match_verify",
        desc="Verify Yelp page is for this restaurant",
        parent=restaurant_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"This is a Yelp page for the restaurant '{restaurant.name}'.",
        node=yelp_match_verify,
        sources=restaurant.yelp_url,
        additional_instruction="Verify that the restaurant name and address on the Yelp page match what was provided."
    )
    
    # 4. Verify Yelp rating is 4+ stars
    rating_verify = evaluator.add_leaf(
        id=f"restaurant_{restaurant_index}_rating_verify",
        desc="Verify 4+ star rating",
        parent=restaurant_node,
        critical=True
    )
    

    claim=f"The restaurant '{restaurant.name}' has a rating of at least 4 stars on Yelp",
    await evaluator.verify(
        claim=claim,
        node=rating_verify,
        sources=restaurant.yelp_url,
        additional_instruction="Look for the star rating on the Yelp page. Accept ratings of 4.0 or higher."
    )
    
    # 5. Verify claimed cuisine type
    cuisine_verify = evaluator.add_leaf(
        id=f"restaurant_{restaurant_index}_cuisine_verify",
        desc="Verify cuisine type",
        parent=restaurant_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The restaurant '{restaurant.name}' serves {restaurant.cuisine} cuisine",
        node=cuisine_verify,
        sources=restaurant_urls.urls,
        additional_instruction="Verify that the restaurant actually serves the claimed type of cuisine."
    )


async def verify_different_cuisines(
        evaluator: Evaluator,
        parent_node,
        restaurants: List[Restaurant]
) -> None:
    """
    Verify that each restaurant represents a different country's cuisine.
    """
    # Extract valid restaurants with cuisine info
    valid_restaurants = [r for r in restaurants if r.name and r.cuisine]
    
    if len(valid_restaurants) <= 1:
        # Only one or no restaurants - automatically pass
        evaluator.add_custom_node(
            result=True,
            id="different_cuisines_auto_pass",
            desc=f"Only {len(valid_restaurants)} restaurant(s) with cuisine info - automatically passes different cuisine check",
            parent=parent_node,
            critical=True
        )
    else:
        # Multiple restaurants - need to verify they're different
        cuisines = [r.cuisine for r in valid_restaurants]
        cuisine_list = ', '.join(cuisines)
        
        different_cuisines_node = evaluator.add_leaf(
            id="different_cuisines_verify",
            desc=f"Verify cuisines are from different countries: {cuisine_list}",
            parent=parent_node,
            critical=True
        )
        
        await evaluator.verify(
            claim=f"The following cuisines represent different countries or cuisine types: {cuisine_list}",
            node=different_cuisines_node,
            sources=None,  # This is a simple logical check
            additional_instruction="Consider cuisines from different countries as different. Regional variations within the same country (e.g., Northern Italian vs Southern Italian) should be considered the same."
        )


async def verify_walking_distance(
        evaluator: Evaluator,
        parent_node,
        distance_pairs: DistancePairs,
        restaurants: List[Restaurant],
        gmaps_tool: tool_googlemap.GoogleMapsTool
) -> None:
    """
    Verify that the walking distance between consecutive restaurants does not exceed 3 miles.
    """
    # Create a map of restaurant names to addresses for lookup
    restaurant_map = {r.name.lower().strip(): r.address for r in restaurants if r.name and r.address}

    distance_node = evaluator.add_parallel(
        id="walking_distance_pairs",
        desc="Check for walking distance pairs",
        parent=parent_node,
        critical=True
    )

    if len(restaurant_map) <= 1:
        for i in range(2):
            evaluator.add_custom_node(
                result=True,
                id=f"distance_pair_{i}",
                desc="Less than 2 restaurant is provided, pass the distance check automatically",
                parent=distance_node,
                critical=True
            )
        return
    
    if len(distance_pairs.pairs) == 0:
        # No distance pairs provided
        for i in range(2):
            evaluator.add_custom_node(
                result=False,
                id=f"distance_pair_{i}",
                desc="No distance information provided between restaurants",
                parent=distance_node,
                critical=True
            )
        return
    
    # Verify each distance pair
    valid_pair_cnt = 0
    for i, pair in enumerate(distance_pairs.pairs):
        if not pair.restaurant1_name or not pair.restaurant2_name:
            # Skip invalid pairs
            continue

        if valid_pair_cnt == 2:
            # only check for two pairs
            break
        
        # Look up addresses for both restaurants
        start_address = restaurant_map.get(pair.restaurant1_name.lower().strip())
        end_address = restaurant_map.get(pair.restaurant2_name.lower().strip())
        
        if not start_address or not end_address:
            evaluator.add_custom_node(
                result=False,
                id=f"distance_pair_{valid_pair_cnt}",
                desc=f"Distance from '{pair.restaurant1_name}' to '{pair.restaurant2_name}' ≤ 3 miles",
                parent=distance_node,
                critical=True
            )
            valid_pair_cnt += 1
            continue
        # Calculate actual distance using Google Maps
        try:
            distance_meters = await gmaps_tool.calculate_distance(
                start_address,
                end_address,
                mode="walking"  # Using walking mode as specified in task
            )
            
            if isinstance(distance_meters, int):
                actual_distance_miles = distance_meters * 0.000621371  # Convert meters to miles

                # If claimed distance was provided, verify it's reasonably accurate
                if pair.distance_miles is not None:
                    # Allow some tolerance (e.g., ±0.5 miles) for claimed vs actual distance
                    tolerance = 0.5
                    claimed_accurate = abs(pair.distance_miles - actual_distance_miles) <= tolerance
                    
                    evaluator.add_custom_node(
                        result=bool(claimed_accurate) and bool(actual_distance_miles <= 3.0),
                        id=f"distance_pair_{valid_pair_cnt}",
                        desc=f"Claimed distance ({pair.distance_miles} miles) matches actual ({actual_distance_miles:.2f} miles), which is <= 3 miles",
                        parent=distance_node,
                        critical=True
                    )
                else:
                    evaluator.add_custom_node(
                        result=bool(actual_distance_miles <= 3.0),
                        id=f"distance_pair_{valid_pair_cnt}",
                        desc=f"Actual ({actual_distance_miles:.2f} miles) <= 3 miles",
                        parent=distance_node,
                        critical=True
                    )
            else:
                # Failed to calculate distance
                evaluator.add_custom_node(
                    result=False,
                    id=f"distance_pair_{valid_pair_cnt}",
                    desc=f"Failed to calculate distance: {distance_meters}",
                    parent=distance_node,
                    critical=True
                )
        except Exception as e:
            # Error calculating distance
            evaluator.add_custom_node(
                result=False,
                id=f"distance_pair_{i}_calculation_error",
                desc=f"Error calculating distance: {str(e)}",
                parent=distance_node,
                critical=True
            )
        valid_pair_cnt += 1
    
    if valid_pair_cnt < 2:
        if len(restaurant_map) == 3:
            while valid_pair_cnt < 2:
                evaluator.add_custom_node(
                    result=False,
                    id=f"distance_pair_{valid_pair_cnt}",
                    desc=f"Missing distance pair",
                    parent=distance_node,
                    critical=True
                )
                valid_pair_cnt += 1
        else:
            while valid_pair_cnt < 2:
                evaluator.add_custom_node(
                    result=True,
                    id=f"distance_pair_{valid_pair_cnt}",
                    desc=f"No need to check due to the missing of restaurant",
                    parent=distance_node,
                    critical=True
                )
                valid_pair_cnt += 1


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: object,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate the answer to the Manhattan restaurants task.
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
        default_model=model
    )
    
    # Initialize Google Maps tool
    gmaps_tool = tool_googlemap.GoogleMapsTool()
    
    # STAGE 1: Extract restaurant names first
    restaurant_names = await evaluator.extract(
        prompt=prompt_extract_restaurant_names(),
        template_class=RestaurantNames,
        extraction_name="restaurant_names"
    )
    
    # STAGE 2: Extract detailed information for each restaurant
    restaurants = []
    restaurant_urls_list = []
    
    for i, restaurant_name in enumerate(restaurant_names.restaurants):
        if not restaurant_name.name:
            continue
            
        # Extract details for this restaurant
        restaurant_details = await evaluator.extract(
            prompt=prompt_extract_restaurant_details(restaurant_name.name),
            template_class=Restaurant,
            extraction_name=f"restaurant_{i}_details"
        )
        restaurants.append(restaurant_details)
        
        # Extract URLs for this restaurant
        urls = await evaluator.extract(
            prompt=prompt_extract_restaurant_urls(restaurant_name.name),
            template_class=RestaurantURLs,
            extraction_name=f"restaurant_{i}_urls"
        )
        restaurant_urls_list.append(urls)
    
    # Extract distance information
    distance_pairs = await evaluator.extract(
        prompt=prompt_extract_distance_info(),
        template_class=DistancePairs,
        extraction_name="distance_pairs"
    )
    
    # Pad to exactly 3 restaurants if needed
    while len(restaurants) < 3:
        restaurants.append(Restaurant())
        restaurant_urls_list.append(RestaurantURLs())
    
    # Limit to first 3 restaurants if more are provided
    if len(restaurants) > 3:
        restaurants = restaurants[:3]
        restaurant_urls_list = restaurant_urls_list[:3]
    
    # Build verification tree
    
    # Verify each restaurant individually (directly under root)
    for i, (restaurant, urls) in enumerate(zip(restaurants, restaurant_urls_list)):
        await verify_restaurant_criteria(evaluator, root, restaurant, urls, i)
    
    # Verify criteria that depend on multiple restaurants
    multi_criteria_node = evaluator.add_sequential(
        id="multi_restaurant_criteria",
        desc="Criteria that depend on multiple restaurants",
        parent=root,
        critical=True  # Critical as requested
    )
    
    # Verify different cuisines
    await verify_different_cuisines(evaluator, multi_criteria_node, restaurants)
    
    # Verify walking distances
    await verify_walking_distance(evaluator, multi_criteria_node, distance_pairs, restaurants, gmaps_tool)
    
    # Get the final evaluation summary
    return evaluator.get_summary()