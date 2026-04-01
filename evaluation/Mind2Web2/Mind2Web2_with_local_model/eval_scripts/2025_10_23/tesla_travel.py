import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy
from mind2web2.api_tools import tool_googlemap

TASK_ID = "tesla_travel"
TASK_DESCRIPTION = """
I am driving my Tesla from Nashville, TN to Indianapolis, IN, and plan to stop in Evansville, IN and Bloomington, IN. For each Tesla Supercharger located in these two cities, please provide the exact address of the Supercharger, along with the names, addresses, and Google Map pages of three restaurants within a mile driving distance of each Supercharger.
"""

EVAL_NOTES = ""
GROUND_TRUTH = {
    "evansville_supercharger": "2622 Menards Dr, Evansville, IN 47715",
    "bloomington_supercharger": "3600 W 3rd St, Bloomington, IN 47404"
}


class SuperchargerInfo(BaseModel):
    """Information about a Tesla Supercharger location."""
    address: Optional[str] = Field(default=None, description="Complete address of the Supercharger")
    source_urls: Optional[List[str]] = Field(default_factory=list, description="URLs supporting this information")


class RestaurantInfo(BaseModel):
    """Information about a restaurant near a Supercharger."""
    name: Optional[str] = Field(default=None, description="Name of the restaurant")
    address: Optional[str] = Field(default=None, description="Address of the restaurant")
    google_maps_urls: Optional[List[str]] = Field(default_factory=list,
                                                  description="Google Maps page URLs for this restaurant")
    source_urls: Optional[List[str]] = Field(default_factory=list, description="URLs supporting this information")


class RestaurantsList(BaseModel):
    """A wrapper class for a list of restaurants to work with OpenAI structured output."""
    restaurants: List[RestaurantInfo] = Field(default_factory=list, description="List of restaurants")


class CityInfo(BaseModel):
    """Complete information for a single city including Supercharger and restaurants."""
    supercharger: Optional[SuperchargerInfo] = Field(default=None, description="Supercharger information")
    restaurants: Optional[List[RestaurantInfo]] = Field(default_factory=list, description="List of restaurants")


def prompt_extract_supercharger_info(city_name: str) -> str:
    """
    Extraction prompt for Tesla Supercharger information in a specific city.
    """
    return f"""
    Extract Tesla Supercharger information for {city_name}, IN from the answer.

    Look for:
    - The complete address of the Tesla Supercharger in {city_name}
    - Any URLs that support or provide this Supercharger address information

    Extract information exactly as it appears in the text.
    If the Supercharger address is not mentioned for {city_name}, set address to null.
    Include all URLs that might support the Supercharger address information.
    """


def prompt_extract_restaurants_info(city_name: str) -> str:
    """
    Extraction prompt for restaurant information in a specific city.
    """
    return f"""
    Extract restaurant information for {city_name}, IN from the answer.

    Look for restaurants near the Tesla Supercharger in {city_name}. The task asks for 3 restaurants.

    For each restaurant, extract:
    - Name of the restaurant
    - Address of the restaurant
    - Google Maps page URLs for the restaurant (extract ALL Google Maps URLs for each restaurant)
    - Any other URLs that support this restaurant information

    Extract information exactly as it appears in the text.
    If any field is not mentioned for a restaurant, set it to null.
    If fewer than 3 restaurants are provided for {city_name}, extract what is available.
    For Google Maps URLs, include ALL URLs that appear to be Google Maps links for each restaurant.
    """


async def verify_supercharger_location(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        city_name: str,
        supercharger_info: Optional[SuperchargerInfo],
        ground_truth_address: str,
) -> None:
    """
    Verify Tesla Supercharger location information for a city.
    """
    # Create existence check as critical node
    existence_node = evaluator.add_custom_node(
        result=bool(supercharger_info and supercharger_info.address and supercharger_info.address.strip()),
        id=f"{city_name.lower()}_supercharger_exists",
        desc=f"Tesla Supercharger address provided for {city_name}",
        parent=parent_node,
        critical=True,
    )

    # Accuracy verification - compare against ground truth
    accuracy_node = evaluator.add_leaf(
        id=f"{city_name.lower()}_supercharger_accuracy",
        desc=f"Supercharger address matches ground truth for {city_name}",
        parent=parent_node,
        critical=True,
    )

    # Verify accuracy against ground truth
    if supercharger_info and supercharger_info.address:
        claim = f"The provided address '{supercharger_info.address}' matches the ground truth address '{ground_truth_address}'"
        additional_instruction = "Allow for minor formatting differences, abbreviations, and reasonable address variations. Focus on whether they refer to the same physical location. For example, if no zip code nor city/state name is provided, it should still be considered accurate if the rest of the address matches."
    else:
        claim = "No address provided for verification"
        additional_instruction = "Mark as failed since no address is available"

    await evaluator.verify(
        claim=claim,
        node=accuracy_node,
        sources=None,
        additional_instruction=additional_instruction
    )

    # URL provenance verification if sources are provided
    if supercharger_info and supercharger_info.source_urls:
        provenance_node = evaluator.add_leaf(
            id=f"{city_name.lower()}_supercharger_provenance",
            desc=f"Supercharger address is supported by provided URLs for {city_name}",
            parent=parent_node,
            critical=True,
        )

        if supercharger_info.address:
            provenance_claim = f"This page shows an address of '{supercharger_info.address}' for a Tesla Supercharger"
            await evaluator.verify(
                claim=provenance_claim,
                node=provenance_node,
                sources=supercharger_info.source_urls,
                additional_instruction="Allow for reasonable variations in address formatting. Verify that the page shows Tesla Supercharger information with the claimed address."
            )


async def verify_restaurant_basic_info(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        city_name: str,
        restaurant: RestaurantInfo,
        restaurant_index: int,
) -> None:
    """
    Verify basic completeness of restaurant information.
    """
    restaurant_id = f"{city_name.lower()}_restaurant_{restaurant_index + 1}"

    # Check if all required fields are present
    has_name = bool(restaurant.name and restaurant.name.strip())
    has_address = bool(restaurant.address and restaurant.address.strip())
    has_google_maps = bool(restaurant.google_maps_urls and len(restaurant.google_maps_urls) > 0)

    # Check if Google Maps URLs are actually Google Maps URLs
    valid_gmaps_urls = []
    if has_google_maps:
        for url in restaurant.google_maps_urls:
            if url and ("maps.google" in url.lower() or "google.com/maps" in url.lower()):
                valid_gmaps_urls.append(url)

    has_valid_google_maps = len(valid_gmaps_urls) > 0

    # Combined existence and validity check
    existence_node = evaluator.add_custom_node(
        result=(has_name and has_address and has_valid_google_maps),
        id=f"{restaurant_id}_complete_info",
        desc=f"Restaurant {restaurant_index + 1} has name, address, and valid Google Maps URLs",
        parent=parent_node,
        critical=True,
    )


async def verify_restaurant_google_maps(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        city_name: str,
        restaurant: RestaurantInfo,
        restaurant_index: int,
) -> None:
    """
    Verify restaurant information against Google Maps pages.
    """
    restaurant_id = f"{city_name.lower()}_restaurant_{restaurant_index + 1}"

    # Filter valid Google Maps URLs
    valid_gmaps_urls = []
    if restaurant.google_maps_urls:
        for url in restaurant.google_maps_urls:
            if url and ("maps.google" in url.lower() or "google.com/maps" in url.lower()):
                valid_gmaps_urls.append(url)

    if not valid_gmaps_urls:
        return  # Skip if no valid Google Maps URLs

    # Verify address consistency with Google Maps
    address_consistency_node = evaluator.add_leaf(
        id=f"{restaurant_id}_gmaps_address_match",
        desc=f"Restaurant {restaurant_index + 1} address matches Google Maps page",
        parent=parent_node,
        critical=True,
    )

    if restaurant.address:
        address_claim = f"The Google Maps page shows the address '{restaurant.address}' for this location"
        await evaluator.verify(
            claim=address_claim,
            node=address_consistency_node,
            sources=valid_gmaps_urls,
            additional_instruction="Verify that the Google Maps page shows the same or equivalent address as claimed. Allow for reasonable formatting differences and address variations."
        )

    # Verify name consistency with Google Maps (if the Maps page shows a clear business name)
    name_consistency_node = evaluator.add_leaf(
        id=f"{restaurant_id}_gmaps_name_match",
        desc=f"Restaurant {restaurant_index + 1} name matches or is reasonable compared to Google Maps page",
        parent=parent_node,
        critical=True,
    )

    if restaurant.name:
        name_claim = f"The Google Maps page shows the business name '{restaurant.name}' or a reasonable variant, OR the page shows location information that could reasonably correspond to a restaurant with this name"
        await evaluator.verify(
            claim=name_claim,
            node=name_consistency_node,
            sources=valid_gmaps_urls,
            additional_instruction="Check if the Google Maps page shows the exact restaurant name, a reasonable variant of it, or at least shows a location that could plausibly be this restaurant. Be flexible with name matching as business names can vary slightly across different sources."
        )


async def verify_restaurant_distance(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        city_name: str,
        restaurant: RestaurantInfo,
        restaurant_index: int,
        supercharger_address: str,
        gmaps_tool: tool_googlemap.GoogleMapsTool,
) -> None:
    """
    Verify that restaurant is within 1 mile driving distance of the Supercharger.
    """
    restaurant_id = f"{city_name.lower()}_restaurant_{restaurant_index + 1}"

    distance_node = evaluator.add_leaf(
        id=f"{restaurant_id}_distance_check",
        desc=f"Restaurant {restaurant_index + 1} is within 1 mile driving distance of Supercharger",
        parent=parent_node,
        critical=True,
    )

    if not restaurant.address:
        # If no address, this will be handled by the existence check
        return

    try:
        # Calculate driving distance using Google Maps API
        distance_meters = await gmaps_tool.calculate_distance(
            supercharger_address,
            restaurant.address,
            mode="driving"
        )

        if isinstance(distance_meters, int):
            # Convert meters to miles (1 mile = 1609.34 meters)
            distance_miles = distance_meters / 1609.34
            is_within_one_mile = distance_miles <= 1.0

            # Create custom node with distance check result
            evaluator.add_custom_node(
                result=is_within_one_mile,
                id=f"{restaurant_id}_distance_result",
                desc=f"Restaurant {restaurant_index + 1} distance: {distance_miles:.2f} miles ({'✓' if is_within_one_mile else '✗'} within 1 mile)",
                parent=distance_node,
                critical=True,
            )
        else:
            # Distance calculation failed
            evaluator.add_custom_node(
                result=False,
                id=f"{restaurant_id}_distance_failed",
                desc=f"Restaurant {restaurant_index + 1} distance calculation failed",
                parent=distance_node,
                critical=True,
            )

    except Exception as e:
        # Handle API errors
        evaluator.add_custom_node(
            result=False,
            id=f"{restaurant_id}_distance_error",
            desc=f"Restaurant {restaurant_index + 1} distance verification error: {str(e)}",
            parent=distance_node,
            critical=True,
        )


async def verify_restaurant_info(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        city_name: str,
        restaurant: RestaurantInfo,
        restaurant_index: int,
        supercharger_address: str,
        gmaps_tool: tool_googlemap.GoogleMapsTool,
) -> None:
    """
    Verify all aspects of restaurant information.
    """
    # Basic info completeness
    await verify_restaurant_basic_info(evaluator, parent_node, city_name, restaurant, restaurant_index)

    # Google Maps verification
    await verify_restaurant_google_maps(evaluator, parent_node, city_name, restaurant, restaurant_index)

    # Distance verification
    await verify_restaurant_distance(evaluator, parent_node, city_name, restaurant, restaurant_index,
                                     supercharger_address, gmaps_tool)


async def verify_city_restaurants(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        city_name: str,
        restaurants: List[RestaurantInfo],
        supercharger_address: str,
        gmaps_tool: tool_googlemap.GoogleMapsTool,
) -> None:
    """
    Verify restaurant information for a city.
    """
    # Ensure we have exactly 3 restaurant slots (pad with empty ones if needed)
    padded_restaurants = restaurants[:3] if restaurants else []
    while len(padded_restaurants) < 3:
        padded_restaurants.append(RestaurantInfo())

    # Create nodes for each restaurant (including missing ones)
    for i, restaurant in enumerate(padded_restaurants):
        restaurant_node = evaluator.add_parallel(
            id=f"{city_name.lower()}_restaurant_{i + 1}",
            desc=f"Restaurant {i + 1} information for {city_name}",
            parent=parent_node,
            critical=False,  # Allow partial credit
        )

        if i < len(restaurants) and restaurants[i].name:  # Restaurant exists and has data
            await verify_restaurant_info(
                evaluator, restaurant_node, city_name, restaurant, i, supercharger_address, gmaps_tool
            )
        else:  # Missing restaurant
            evaluator.add_custom_node(
                result=False,
                id=f"{city_name.lower()}_restaurant_{i + 1}_missing",
                desc=f"Restaurant {i + 1} information missing for {city_name}",
                parent=restaurant_node,
                critical=True,
            )


async def verify_city_info(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        city_name: str,
        city_info: Optional[CityInfo],
        ground_truth_address: str,
        gmaps_tool: tool_googlemap.GoogleMapsTool,
) -> None:
    """
    Verify complete information for a city.
    """
    city_node = evaluator.add_sequential(
        id=f"{city_name.lower()}_city",
        desc=f"Complete information for {city_name}, IN",
        parent=parent_node,
        critical=False,  # Allow partial credit between cities
    )

    # Supercharger verification
    supercharger_node = evaluator.add_parallel(
        id=f"{city_name.lower()}_supercharger",
        desc=f"Tesla Supercharger information for {city_name}",
        parent=city_node,
        critical=False,  # Supercharger info is essential for the task
    )

    supercharger_info = city_info.supercharger if city_info else None
    await verify_supercharger_location(evaluator, supercharger_node, city_name, supercharger_info, ground_truth_address)

    # Restaurants verification
    restaurants_node = evaluator.add_parallel(
        id=f"{city_name.lower()}_restaurants",
        desc=f"Restaurant information for {city_name} (3 restaurants expected)",
        parent=city_node,
        critical=False,  # Restaurant info is essential for the task
    )

    restaurants = city_info.restaurants if city_info else []
    await verify_city_restaurants(
        evaluator, restaurants_node, city_name, restaurants, ground_truth_address, gmaps_tool
    )


async def evaluate_answer(
        client,  # LLMClient type hint removed for compatibility
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Main evaluation function for Tesla travel task.

    Evaluates whether the answer provides:
    1. Correct Tesla Supercharger addresses for Evansville and Bloomington
    2. Three restaurants per city with names, addresses, and Google Maps URLs
    3. Proper source attribution for the information provided
    4. Restaurants within 1 mile driving distance of each Supercharger
    """

    # -------- 1. Initialize evaluator ----------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Cities can be evaluated independently
        agent_name=agent_name,
        answer_name=answer_name,
        # Evaluator creation parameters
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # -------- 2. Record ground truth information ------------------- #
    evaluator.add_ground_truth(GROUND_TRUTH, "supercharger_addresses")

    # -------- 3. Initialize Google Maps tool ---------------------- #
    gmaps_tool = tool_googlemap.GoogleMapsTool()

    # -------- 4. Extract structured information separately --------- #

    # Extract Evansville Supercharger info
    evansville_supercharger = await evaluator.extract(
        prompt=prompt_extract_supercharger_info("Evansville"),
        template_class=SuperchargerInfo,
        extraction_name="evansville_supercharger_extraction",
        source=None,
    )

    # Extract Evansville restaurant info
    evansville_restaurants_result = await evaluator.extract(
        prompt=prompt_extract_restaurants_info("Evansville"),
        template_class=RestaurantsList,
        extraction_name="evansville_restaurants_extraction",
        source=None,
    )

    # Extract Bloomington Supercharger info
    bloomington_supercharger = await evaluator.extract(
        prompt=prompt_extract_supercharger_info("Bloomington"),
        template_class=SuperchargerInfo,
        extraction_name="bloomington_supercharger_extraction",
        source=None,
    )

    # Extract Bloomington restaurant info
    bloomington_restaurants_result = await evaluator.extract(
        prompt=prompt_extract_restaurants_info("Bloomington"),
        template_class=RestaurantsList,
        extraction_name="bloomington_restaurants_extraction",
        source=None,
    )

    # -------- 5. Build city info objects -------------------------- #
    evansville_info = CityInfo(
        supercharger=evansville_supercharger,
        restaurants=evansville_restaurants_result.restaurants if isinstance(evansville_restaurants_result.restaurants, list) else []
    )

    bloomington_info = CityInfo(
        supercharger=bloomington_supercharger,
        restaurants=bloomington_restaurants_result.restaurants if isinstance(bloomington_restaurants_result.restaurants, list) else []
    )

    # -------- 6. Build verification tree -------------------------- #

    # Verify Evansville information
    await verify_city_info(
        evaluator,
        root,
        "Evansville",
        evansville_info,
        GROUND_TRUTH["evansville_supercharger"],
        gmaps_tool
    )

    # Verify Bloomington information
    await verify_city_info(
        evaluator,
        root,
        "Bloomington",
        bloomington_info,
        GROUND_TRUTH["bloomington_supercharger"],
        gmaps_tool
    )

    # -------- 7. Return evaluation results ------------------------ #
    return evaluator.get_summary()