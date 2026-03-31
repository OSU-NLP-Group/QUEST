import asyncio
import logging
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from mind2web2.eval_toolkit import Extractor, Verifier, create_evaluator
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import Evaluator, AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "find_bikes"
TASK_DESCRIPTION = """
Identify one currently available black road bike model from each of these brands: Giant, and Specialized. For each bike, clearly state the frame material (e.g., carbon, aluminum, steel, titanium), brake type (e.g., hydraulic disc, mechanical disc, or rim brakes), and the number of gears (e.g., 11-speed, 12-speed), and also provide a direct link to the product page on the official brand website.
"""

REQUIRED_BRANDS = ["Giant", "Specialized"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BikeInfo(BaseModel):
    """Model for a single bike's information."""
    brand: Optional[str] = None
    model_name: Optional[str] = None
    frame_material: Optional[str] = None
    brake_type: Optional[str] = None
    gears: Optional[str] = None
    url: Optional[str] = None


class ExtractedBikes(BaseModel):
    """Model for all extracted bikes."""
    bikes: List[BikeInfo] = Field(default_factory=list)


def prompt_extract_brand_bike(brand: str) -> str:
    return f"""
    Look for road bikes from the {brand} brand specifically. Extract details about the first {brand} road bike mentioned in the answer:

    1. Confirm the brand is {brand} (case-insensitive matching accepted)
    2. The model name of the bike
    3. The frame material (e.g., carbon, aluminum, steel, titanium, alloy, composite)
    4. The brake type (e.g., hydraulic disc, mechanical disc, rim brakes, disc brakes)
    5. The number of gears (e.g., 11-speed, 12-speed)
    6. The URL to the product page

    If no {brand} bike is found, return an empty bikes array.
    If multiple {brand} bikes are mentioned, only extract the first one.

    Be flexible with brand name matching - accept variations like "{brand.lower()}", "{brand.upper()}", etc.
    """


# --------------------------------------------------------------------------- #
# Verification functions for each bike component                              #
# --------------------------------------------------------------------------- #
async def verify_bike_requirements(
        evaluator: Evaluator,
        parent_node,
        bike: BikeInfo,
        brand: str
) -> None:
    """Verify all requirements for a single bike."""
    
    # Create parallel verification group for this bike
    bike_verification = evaluator.add_parallel(
        id=f"{brand}_bike_verification",
        desc=f"Verification of all requirements for {brand} bike",
        parent=parent_node,
        critical=False
    )
    
    # Existence check - critical gate for all verifications
    bike_exists = evaluator.add_custom_node(
        result=bool(bike.model_name and bike.model_name.strip() != "" and
                    bike.url and bike.url.strip() != "" and 
                    bike.frame_material and bike.frame_material.strip() != "" and
                    bike.brake_type and bike.brake_type.strip() != "" and
                    bike.gears and bike.gears.strip() != ""
                    ),
        id=f"{brand}_bike_complete",
        desc=f"Check if complete {brand} bike information is provided",
        parent=bike_verification,
        critical=True
    )
    
    # URL verification
    url_node = evaluator.add_leaf(
        id=f"{brand}_official_url",
        desc=f"Verify that the URL for the {brand} {bike.model_name or 'bike'} is from the official {brand} website",
        parent=bike_verification,
        critical=True
    )
    
    # Known official domains for the required brands
    official_domains = {
        "Giant": ["giant-bicycles.com", "giant.com", "giant-bikes.com"],
        "Specialized": ["specialized.com"],
    }
    
    brand_domains = official_domains.get(brand, [])
    url_check_instruction = f"""
    Verify that the URL is from the official {brand} website. For reference, 
    these are known official domains for {brand}: {', '.join(brand_domains)}.

    The URL should lead to an official product page, not a third-party retailer, 
    review site, or other non-official source. Check the domain carefully.
    """
    
    await evaluator.verify(
        claim=f"The URL '{bike.url}' is from the official {brand} website domain.",
        node=url_node,
        additional_instruction=url_check_instruction
    )

    if url_node.status == "failed":
        # If simple verification fails, try to load the page and check
        await evaluator.verify(
            claim=f"This webpage is on the official {brand} website.",
            node=url_node,
            sources=bike.url,
            additional_instruction=url_check_instruction
        )
    
    # Road bike verification
    road_bike_node = evaluator.add_leaf(
        id=f"{brand}_is_road_bike",
        desc=f"Verify that the {brand} {bike.model_name or 'bike'} is a road bike",
        parent=bike_verification,
        critical=True
    )
    
    road_bike_instruction = """
    Determine if this is specifically a road bike based on the product information and images. 
    Road bikes typically have drop handlebars, thin tires, and are designed for paved roads 
    and speed. They are distinct from mountain bikes, hybrid bikes, gravel bikes, cruisers, or other types.
    Look for explicit categorization as a "road bike" in the product description, navigation 
    hierarchy, or clear visual indicators that this is a road bike.
    """
    
    await evaluator.verify(
        claim=f"The {brand} {bike.model_name or 'bike'} is categorized as a road bike.",
        node=road_bike_node,
        sources=bike.url,
        additional_instruction=road_bike_instruction
    )
    
    # Color verification
    color_node = evaluator.add_leaf(
        id=f"{brand}_is_black",
        desc=f"Verify that the {brand} {bike.model_name or 'bike'} is available in black color",
        parent=bike_verification,
        critical=True
    )
    
    color_check_instruction = """
    Look at the bike images and product information on this page. Check if the bike is available in black 
    or a very dark color that could reasonably be considered black (such as "deep smoke", 
    "carbon", "charcoal", "midnight", "stealth", etc.). Focus on both the actual bike shown in images 
    and any color options mentioned in the product description.
    """
    
    await evaluator.verify(
        claim=f"The {brand} {bike.model_name or 'bike'} is available in black or a very dark color that could reasonably be considered black.",
        node=color_node,
        sources=bike.url,
        additional_instruction=color_check_instruction
    )
    
    # Availability verification
    availability_node = evaluator.add_leaf(
        id=f"{brand}_availability",
        desc=f"Verify that the {brand} {bike.model_name or 'bike'} is currently available for purchase",
        parent=bike_verification,
        critical=True
    )
    
    availability_instruction = """
    Check if the bike appears to be currently available for purchase. Look for indicators such as:
    1. The ability to add the bike to cart or select size/options
    2. Current pricing information displayed
    3. Size/configuration options available
    4. Absence of "out of stock", "discontinued", "no longer available", or similar messages

    If the bike appears to be a current model that can be purchased, consider it available.
    """
    
    await evaluator.verify(
        claim=f"The {brand} {bike.model_name or 'bike'} appears to be currently available for purchase.",
        node=availability_node,
        sources=bike.url,
        additional_instruction=availability_instruction
    )
    
    # Frame material verification
    frame_node = evaluator.add_leaf(
        id=f"{brand}_frame_material",
        desc=f"Verify that the frame material of the {brand} {bike.model_name or 'bike'} is correctly identified as '{bike.frame_material or 'not specified'}'",
        parent=bike_verification,
        critical=True
    )
    
    frame_instruction = """
    Examine the product page to verify the frame material of this bike. Look for specific 
    mentions of the frame material in the product specifications, features, or description.

    Common frame materials include carbon fiber, aluminum/aluminium, steel, titanium, or 
    composite materials. Some bikes may use terms like "alloy" (usually referring to aluminum alloy)
    or "carbon composite". Verify that the claimed frame material matches what's stated on the 
    official product page. Allow for reasonable variations in terminology (e.g., "carbon" vs "carbon fiber").
    """
    
    await evaluator.verify(
        claim=f"The frame material of the {brand} {bike.model_name or 'bike'} is {bike.frame_material}.",
        node=frame_node,
        sources=bike.url,
        additional_instruction=frame_instruction
    )
    
    # Brake type verification
    brake_node = evaluator.add_leaf(
        id=f"{brand}_brake_type",
        desc=f"Verify that the brake type of the {brand} {bike.model_name or 'bike'} is correctly identified as '{bike.brake_type or 'not specified'}'",
        parent=bike_verification,
        critical=True
    )
    
    brake_instruction = """
    Examine the product page to verify the brake type of this bike. Look for specific 
    mentions of the brake type in the product specifications, features, or description.

    Common brake types include hydraulic disc brakes, mechanical disc brakes, rim brakes,
    or simply disc brakes (when the specific type isn't specified). Verify that the 
    claimed brake type matches what's stated on the official product page. Allow for 
    reasonable variations in terminology.
    """
    
    await evaluator.verify(
        claim=f"The brake type of the {brand} {bike.model_name or 'bike'} is {bike.brake_type}.",
        node=brake_node,
        sources=bike.url,
        additional_instruction=brake_instruction
    )
    
    # Gear count verification
    gear_node = evaluator.add_leaf(
        id=f"{brand}_gears",
        desc=f"Verify that the number of gears on the {brand} {bike.model_name or 'bike'} is correctly identified as '{bike.gears or 'not specified'}'",
        parent=bike_verification,
        critical=True
    )
    
    gears_instruction = """
    Examine the product page to verify the number of gears or speeds for this bike. Look for specific 
    mentions in the product specifications, features, or description.

    Common representations include "11-speed", "12-speed", "22 gears", etc. Sometimes this may be 
    indicated by the cassette range (e.g., "11-32T, 12-speed") or drivetrain specifications.
    Verify that the claimed number of gears matches what's stated on the official product page.
    Allow for reasonable variations in terminology (e.g., "speed" vs "gears").
    """
    
    await evaluator.verify(
        claim=f"The {brand} {bike.model_name or 'bike'} has {bike.gears} gears/speeds.",
        node=gear_node,
        sources=bike.url,
        additional_instruction=gears_instruction
    )


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
    Evaluate a single answer and return a structured result dictionary.
    """
    
    # -------- 1. Initialize evaluator ----------------------------------- #
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

    # -------- 2. Extract structured info from the answer for each brand separately -------- #
    logger.info("Extracting bike information from the answer for each brand...")

    brand_bike_map = {}
    for brand in REQUIRED_BRANDS:
        try:
            brand_bikes = await evaluator.extract(
                prompt=prompt_extract_brand_bike(brand),
                template_class=ExtractedBikes,
                extraction_name=f"{brand}_bike_extraction"
            )

            if brand_bikes.bikes:
                # Take only the first bike for this brand
                brand_bike_map[brand] = brand_bikes.bikes[0]
            else:
                # Create empty bike object to maintain consistent structure
                brand_bike_map[brand] = BikeInfo(brand=brand)

        except Exception as e:
            logger.error(f"Error extracting bike for {brand}: {e}")
            # Create empty bike object for failed extraction
            brand_bike_map[brand] = BikeInfo(brand=brand)

    # -------- 3. Add custom info about extraction summary --------------- #
    evaluator.add_custom_info({
        "extracted_bikes": {brand: bike.dict() if bike.model_name else None for brand, bike in brand_bike_map.items()},
        "required_brands": REQUIRED_BRANDS,
        "extraction_summary": {
            "total_brands_found": len([b for b in brand_bike_map.values() if b.model_name]),
            "total_brands_required": len(REQUIRED_BRANDS),
            "missing_brands": [brand for brand, bike in brand_bike_map.items() if not bike.model_name],
        }
    }, "bike_extraction_results")

    # -------- 4. Build verification tree -------------------------------- #
    # Verify each required brand - this creates consistent tree structure even if bikes are missing
    for brand in REQUIRED_BRANDS:
        bike = brand_bike_map.get(brand, BikeInfo(brand=brand))
        await verify_bike_requirements(evaluator, root, bike, brand)

    # -------- 5. Return structured result ------------------------------- #
    return evaluator.get_summary()