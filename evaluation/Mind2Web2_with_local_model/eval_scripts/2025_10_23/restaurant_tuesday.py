import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient

TASK_ID = "restaurant_tuesday"
TASK_DESCRIPTION = """
Find restaurants on OpenTable located in Downtown Los Angeles, CA—two for brunch and two for dinner. The brunch restaurants must be open before 12PM on Tuesday, and the dinner restaurants must be open between 6PM and 8PM on Tuesday. All restaurants should have outdoor seating. The four selected restaurants should represent four distinct cuisines: Peruvian, Japanese, Italian, and Spanish. For each restaurant, provide a link to its OpenTable page, the exact physical address, its hours of operation on Tuesday, the cuisine(s) it serves, and a link to its menu on the official website.
"""

EVAL_NOTES = ""
GROUND_TRUTH = {}


class RestaurantBasicInfo(BaseModel):
    """Basic restaurant information for initial extraction"""
    name: Optional[str] = Field(default=None, description="Restaurant name")
    cuisine: Optional[str] = Field(default=None, description="Cuisine type")
    meal_type: Optional[str] = Field(default=None, description="brunch or dinner")


class CuisineRestaurants(BaseModel):
    """Restaurants grouped by cuisine"""
    peruvian_restaurants: List[RestaurantBasicInfo] = Field(default_factory=list)
    japanese_restaurants: List[RestaurantBasicInfo] = Field(default_factory=list)
    italian_restaurants: List[RestaurantBasicInfo] = Field(default_factory=list)
    spanish_restaurants: List[RestaurantBasicInfo] = Field(default_factory=list)


class RestaurantDetailedInfo(BaseModel):
    """Detailed information for a single restaurant"""
    name: Optional[str] = Field(default=None, description="Restaurant name")
    opentable_url: Optional[str] = Field(default=None, description="OpenTable page URL")
    address: Optional[str] = Field(default=None, description="Physical address")
    tuesday_hours: Optional[str] = Field(default=None, description="Hours on Tuesday")
    cuisine: Optional[str] = Field(default=None, description="Cuisine type")
    meal_type: Optional[str] = Field(default=None, description="brunch or dinner")
    other_urls: List[str] = Field(default_factory=list, description="All other URLs mentioned")


def prompt_extract_restaurants_by_cuisine() -> str:
    """Initial extraction prompt for restaurants grouped by cuisine"""
    return """
    Extract all restaurants mentioned in the answer, grouped by their cuisine type. The task requires finding restaurants representing 4 cuisines: Peruvian, Japanese, Italian, and Spanish.

    For each restaurant found, extract:
    - name: The restaurant's name
    - cuisine: The cuisine type (Peruvian, Japanese, Italian, or Spanish)
    - meal_type: Whether this restaurant is mentioned as a "brunch" or "dinner" option. Use exactly "brunch" or "dinner" as the value.

    Group the restaurants by cuisine type:
    - peruvian_restaurants: List of all Peruvian restaurants
    - japanese_restaurants: List of all Japanese restaurants
    - italian_restaurants: List of all Italian restaurants
    - spanish_restaurants: List of all Spanish restaurants

    Extract ALL restaurants mentioned for each cuisine, even if multiple are provided.
    """


def prompt_extract_restaurant_details(restaurant_name: str) -> str:
    """Extraction prompt for detailed information about a specific restaurant"""
    return f"""
    Extract detailed information about the restaurant "{restaurant_name}" from the answer.

    Extract:
    - name: The restaurant's name (should be "{restaurant_name}")
    - opentable_url: The OpenTable page URL (must be a valid OpenTable URL)
    - address: The exact physical address
    - tuesday_hours: Hours of operation on Tuesday (extract the exact hours mentioned, e.g., "11AM-10PM")
    - cuisine: The cuisine type
    - meal_type: Whether this is a "brunch" or "dinner" restaurant
    - other_urls: ALL other URLs mentioned for this restaurant (menu links, official website, etc.)

    Make sure to extract ALL URLs associated with this restaurant.
    For tuesday_hours, extract the exact hours as stated in the answer.
    If any field is not mentioned, set it to null.
    """


async def verify_restaurant_details(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        restaurant: RestaurantDetailedInfo,
        index: int,
        expected_cuisine: str,
        expected_meal_type: str
) -> None:
    """Verify detailed information for a single restaurant"""

    # Create parent node for this restaurant
    restaurant_node = evaluator.add_parallel(
        id=f"restaurant_{index}",
        desc=f"Restaurant {index}: {restaurant.name or 'Not provided'} ({expected_cuisine}, {expected_meal_type})",
        parent=parent_node,
        critical=False  # Non-critical to allow partial scoring
    )

    # 1. Check restaurant exists with all required information
    existence_node = evaluator.add_custom_node(
        result=bool(
            restaurant.name and restaurant.name.strip() and 
            restaurant.opentable_url and restaurant.opentable_url.strip() and 
            restaurant.other_urls and 
            restaurant.meal_type and restaurant.meal_type.strip() and 
            restaurant.address and restaurant.address.strip() and
            restaurant.tuesday_hours and restaurant.tuesday_hours.strip()
        ),
        id=f"restaurant_{index}_exists",
        desc=f"Restaurant {index} has all required information",
        parent=restaurant_node,
        critical=True
    )

    # 2. Verify OpenTable page shows correct hours and name
    opentable_hours_node = evaluator.add_leaf(
        id=f"restaurant_{index}_opentable_hours",
        desc=f"OpenTable page confirms restaurant is open at required times and shows correct name",
        parent=restaurant_node,
        critical=True
    )

    if restaurant.meal_type == "brunch":
        time_requirement = "The restaurant must be open before 12PM on Tuesday"
    else:  # dinner
        time_requirement = "The restaurant must be open between 6PM and 8PM on Tuesday"

    await evaluator.verify(
        claim=f"The OpenTable page shows that {restaurant.name} is open during the required times: {time_requirement}. And the page should show the restaurant name matches '{restaurant.name}'.",
        node=opentable_hours_node,
        sources=restaurant.opentable_url,
        additional_instruction=f"Check the Hours of operation section on the OpenTable page. It may not specify Tuesday specifically, but verify the restaurant would be open on Tuesday at the required times. Also confirm the restaurant name on the page matches the provided name. Allow for reasonable minor variations in wording or formatting."
    )

    # 3. Verify OpenTable indicates Downtown LA location
    opentable_location_node = evaluator.add_leaf(
        id=f"restaurant_{index}_opentable_downtown",
        desc=f"OpenTable page indicates restaurant is in Downtown Los Angeles",
        parent=restaurant_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"The OpenTable page indicates this restaurant {restaurant.name} is located in Downtown Los Angeles",
        node=opentable_location_node,
        sources=restaurant.opentable_url,
        additional_instruction="Look for location information in: 1) The address area (often top left), 2) 'Additional information' section under 'Neighborhood', or 3) The address itself. Downtown LA may be indicated as 'Downtown', 'DTLA', or similar variations."
    )

    # 4. Verify outdoor seating
    outdoor_seating_node = evaluator.add_leaf(
        id=f"restaurant_{index}_outdoor_seating",
        desc=f"Restaurant has outdoor seating",
        parent=restaurant_node,
        critical=True
    )

    all_urls = [url for url in [restaurant.opentable_url] + restaurant.other_urls if url]

    await evaluator.verify(
        claim=f"The restaurant {restaurant.name} has outdoor seating available",
        node=outdoor_seating_node,
        sources=all_urls,
        additional_instruction="Look for mentions of outdoor seating, patio, al fresco dining, or similar terms. This information might be in the additional information, amenities section, description, or special features of the restaurant."
    )

    # 5. Verify address
    address_node = evaluator.add_leaf(
        id=f"restaurant_{index}_address_verified",
        desc=f"Restaurant address '{restaurant.address}' is verified by at least one URL",
        parent=restaurant_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"The restaurant {restaurant.name}'s address is: {restaurant.address}",
        node=address_node,
        sources=all_urls,
        additional_instruction="Verify that at least one of the provided URLs confirms this address for the restaurant."
    )

    # 6. Verify exact hours of operation
    hours_verification_node = evaluator.add_leaf(
        id=f"restaurant_{index}_exact_hours",
        desc=f"Exact Tuesday hours '{restaurant.tuesday_hours}' are verified",
        parent=restaurant_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"The restaurant {restaurant.name} is open on Tuesday during these hours: {restaurant.tuesday_hours}",
        node=hours_verification_node,
        sources=all_urls,
        additional_instruction="Verify that the provided hours match what's shown on at least one of the URLs. The hours should indicate the restaurant is open on Tuesday with specific opening and closing times."
    )

    # 7. Verify cuisine type
    cuisine_node = evaluator.add_leaf(
        id=f"restaurant_{index}_cuisine_verified",
        desc=f"Restaurant serves {expected_cuisine} cuisine as verified by URLs",
        parent=restaurant_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"The restaurant {restaurant.name} serves {expected_cuisine} cuisine",
        node=cuisine_node,
        sources=all_urls,
        additional_instruction=f"Verify that at least one URL confirms this restaurant serves {expected_cuisine} cuisine. Allow reasonable variations (e.g., 'Spanish' vs 'Spanish tapas', 'Japanese' vs 'Japanese sushi')."
    )

    # 8. Verify menu URL
    menu_node = evaluator.add_leaf(
        id=f"restaurant_{index}_menu",
        desc=f"Official restaurant menu is available",
        parent=restaurant_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"This page is from the restaurant {restaurant.name}'s official website. And, this page is a menu of it or the page containing their menu",
        node=menu_node,
        sources=restaurant.other_urls,
        additional_instruction=f"Notice, the restaurant name may not exactly appear on the page, but should likely be inferable from the URL, title, icon, branding, etc."
    )


async def evaluate_answer(
        client: LLMClient,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict[str, Any]:
    """Main evaluation function for restaurant_tuesday task"""

    # 1. Initialize evaluator
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
    )

    # 2. Extract restaurants grouped by cuisine
    cuisine_info = await evaluator.extract(
        prompt=prompt_extract_restaurants_by_cuisine(),
        template_class=CuisineRestaurants,
        extraction_name="cuisine_grouped_extraction",
    )

    # 3. Prepare exactly 4 restaurants (2 brunch, 2 dinner) with proper padding
    final_restaurants = []
    final_cuisines = []
    final_meal_types = []
    
    # Target: 2 brunch, 2 dinner from 4 different cuisines
    target_cuisines = ["Peruvian", "Japanese", "Italian", "Spanish"]
    cuisine_restaurants_map = {
        "Peruvian": cuisine_info.peruvian_restaurants,
        "Japanese": cuisine_info.japanese_restaurants,
        "Italian": cuisine_info.italian_restaurants,
        "Spanish": cuisine_info.spanish_restaurants
    }
    
    # Try to get one restaurant from each cuisine, respecting meal type limits
    brunch_count = 0
    dinner_count = 0
    
    for cuisine in target_cuisines:
        restaurants = cuisine_restaurants_map[cuisine]
        if restaurants:
            restaurant = restaurants[0]
            if restaurant.meal_type == "brunch" and brunch_count < 2:
                final_restaurants.append(restaurant)
                final_cuisines.append(cuisine)
                final_meal_types.append("brunch")
                brunch_count += 1
            elif restaurant.meal_type == "dinner" and dinner_count < 2:
                final_restaurants.append(restaurant)
                final_cuisines.append(cuisine)
                final_meal_types.append("dinner")
                dinner_count += 1

    # Pad to exactly 4 restaurants with empty RestaurantBasicInfo objects
    while len(final_restaurants) < 4:
        final_restaurants.append(RestaurantBasicInfo())
        final_cuisines.append("Unknown")
        final_meal_types.append("brunch" if brunch_count < 2 else "dinner")
        if final_meal_types[-1] == "brunch":
            brunch_count += 1
        else:
            dinner_count += 1

    # Add custom info about selection process
    evaluator.add_custom_info({
        "cuisines_found": {
            "Peruvian": len(cuisine_info.peruvian_restaurants),
            "Japanese": len(cuisine_info.japanese_restaurants),
            "Italian": len(cuisine_info.italian_restaurants),
            "Spanish": len(cuisine_info.spanish_restaurants)
        },
        "final_selection": len([r for r in final_restaurants if r.name]),
        "brunch_count": brunch_count,
        "dinner_count": dinner_count
    }, "selection_process")

    # 4. Create verification structure by meal type
    brunch_node = evaluator.add_parallel(
        id="brunch_restaurants",
        desc="Brunch restaurants (open before 12PM on Tuesday)",
        parent=root,
        critical=False
    )

    dinner_node = evaluator.add_parallel(
        id="dinner_restaurants",
        desc="Dinner restaurants (open 6PM-8PM on Tuesday)",
        parent=root,
        critical=False
    )

    # 5. Extract detailed info and verify each restaurant
    for idx, (basic_info, cuisine, meal_type) in enumerate(zip(final_restaurants, final_cuisines, final_meal_types)):
        if basic_info.name:
            # Extract detailed information for real restaurants
            detailed_info = await evaluator.extract(
                prompt=prompt_extract_restaurant_details(basic_info.name),
                template_class=RestaurantDetailedInfo,
                extraction_name=f"restaurant_{idx + 1}_details",
            )
        else:
            # Create empty detailed info for missing restaurants
            detailed_info = RestaurantDetailedInfo()

        # Assign to appropriate parent based on meal type
        parent = brunch_node if meal_type == "brunch" else dinner_node

        # Verify the restaurant (works for both real and empty restaurants)
        await verify_restaurant_details(
            evaluator=evaluator,
            parent_node=parent,
            restaurant=detailed_info,
            index=idx + 1,
            expected_cuisine=cuisine,
            expected_meal_type=meal_type
        )

    # 6. Return evaluation results
    return evaluator.get_summary()