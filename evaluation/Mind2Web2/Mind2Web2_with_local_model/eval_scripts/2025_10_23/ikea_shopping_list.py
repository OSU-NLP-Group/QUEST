import asyncio
import logging
from typing import Optional, Dict, Any

from pydantic import BaseModel

from mind2web2 import Evaluator, AggregationStrategy, CacheFileSys

# Task-specific constants
TASK_ID = "ikea_shopping_list"
TASK_DESCRIPTION = """
I recently moved to a new apartment in the US and I would like to get some furniture for my bedroom from IKEA. My budget is $200-$600 and I need a bed frame, a desk, a chair, a floor lamp, and an two-door wardrobe. Please help me make a shopping list and make sure the total price is within my budget range (do not go over or under). Also, make sure all the furniture in the shopping list are white. For each item, provide the name, price, and purchase link.
"""

# Required furniture types
REQUIRED_TYPES = ["bed frame", "desk", "chair", "floor lamp", "wardrobe"]


# Data models for extraction
class FurnitureItem(BaseModel):
    """Model for a single furniture item"""
    name: Optional[str] = None
    price: Optional[str] = None
    color: Optional[str] = None
    url: Optional[str] = None


class TotalPrice(BaseModel):
    """Model for a single furniture item"""
    total_price: Optional[str] = None


# Extraction prompts
def prompt_extract_furniture_item(furniture_type: str) -> str:
    return f"""
    Extract information about the {furniture_type} mentioned in the answer. Look for:

    1. name: The name of the {furniture_type} (e.g., "MALM", "LINNMON/ADILS", etc.)
    2. price: The price of the {furniture_type} (with or without $ symbol)
    3. color: The color of the {furniture_type}
    4. url: The purchase link/URL for the {furniture_type}

    If there is no {furniture_type} mentioned in the answer, or if any of these details are missing, return null for that field.

    Note: For wardrobe, specifically look for a two-door wardrobe or any wardrobe that has at least 2 doors.
    """


def prompt_extract_total_price() -> str:
    return """
    Extract the total price of all items in the shopping list as mentioned in the answer.
    Return the total price as a string (with or without $ symbol).
    If no total price is explicitly mentioned, return null.
    """


# Helper functions
def extract_numeric_price(price_str: Optional[str]) -> float:
    """Extract numeric price from string, return 0.0 if invalid"""
    if not price_str:
        return 0.0

    try:
        # Remove $ and , characters and convert to float
        return float(price_str.replace("$", "").replace(",", "").strip())
    except ValueError:
        return 0.0


# Verification functions
async def verify_furniture_item(
        evaluator: Evaluator,
        parent_node,
        item: FurnitureItem,
        furniture_type: str,
) -> float:  # Return the price for budget calculation
    """
    Verify a single furniture item meets all requirements.
    Returns the verified price for budget calculation.
    """
    # Create parent node for this item
    item_parent = evaluator.add_parallel(
        id=f"{furniture_type.replace(' ', '_')}_verification",
        desc=f"Verification of {furniture_type} requirements",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )

    # 1. Check if item exists with all required fields (URL and price)
    item_exists = evaluator.add_custom_node(
        result=bool(item and item.url and item.price),
        id=f"{furniture_type.replace(' ', '_')}_exists",
        desc=f"Check if {furniture_type} was found with URL and price",
        parent=item_parent,
        critical=True
    )

    # 2. Verify item is the correct type and color
    type_node = evaluator.add_leaf(
        id=f"{furniture_type.replace(' ', '_')}_correct_type",
        desc=f"Verify the item is a white {furniture_type}",
        parent=item_parent,
        critical=True,
    )

    # Special case for wardrobe
    if furniture_type == "wardrobe":
        claim = f"This is a valid IKEA product page for a {furniture_type}. And, this item in the page is a white two-door wardrobe or has at least 2 doors."
    else:
        claim = f"This is a valid IKEA product page for a {furniture_type}. And, this item in the page is a white {furniture_type}."

    await evaluator.verify(
        claim=claim,
        node=type_node,
        sources=item.url if item else None,
        additional_instruction=f"Check if this IKEA product is actually a {furniture_type}. For wardrobe, verify it has at least 2 doors. Also, verify if it is white in color (at least roughly)."
    )

    # 3. Verify price accuracy
    price_node = evaluator.add_leaf(
        id=f"{furniture_type.replace(' ', '_')}_price_verification",
        desc=f"Verify the price of the {furniture_type} is accurate",
        parent=item_parent,
        critical=True,
    )

    # Since we have a critical existence check, we can assume item.price exists here
    price_claim = f"The price of this {furniture_type} is {item.price}. And the price '{item.price}' is a valid price number (it can be approximated)."

    await evaluator.verify(
        claim=price_claim,
        node=price_node,
        sources=item.url if item else None,
        additional_instruction="minor deviation is acceptable (e.g., 249->250), but the price should be roughly the same as the one in the page."
    )

    # Calculate verified price based on the parent node's aggregated score
    # The parent node will have score > 0 only if all critical nodes passed
    item_parent.compute_score(mutate=True)
    if item_parent.aggregated_score > 0:
        return extract_numeric_price(item.price)
    return 0.0


async def verify_budget_compliance(
        evaluator: Evaluator,
        parent_node,
        calculated_total: float,
        stated_total: Optional[str],
) -> None:
    """
    Verify that the total price is within the budget range ($200-$600)
    """
    budget_parent = evaluator.add_parallel(
        id="budget_compliance",
        desc="Budget requirements verification",
        parent=parent_node,
        critical=True
    )

    # Check if we have a valid total
    has_valid_total = evaluator.add_custom_node(
        result=(calculated_total > 0),
        id="has_valid_total",
        desc="Check if there is a valid calculated total price",
        parent=budget_parent,
        critical=True
    )

    # Verify calculated total is within budget
    within_budget_node = evaluator.add_leaf(
        id="within_budget_range",
        desc=f"Verify the calculated total price (${calculated_total:.2f}) is within budget ($200-$600)",
        parent=budget_parent,
        critical=True,
    )

    claim = f"This number (${calculated_total:.2f}) (the prices of actual valid items after examination) is within the range of 200-600, not going over or under."

    await evaluator.verify(
        claim=claim,
        node=within_budget_node,
        additional_instruction="Check if the number (the actual total price of valid furniture) is between $200 and $600 inclusive. It should not be below $200 or above $600."
    )


# Main evaluation entry point
async def evaluate_answer(
        client,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # 1. Set up evaluator
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

    # 2. Extract structured info from the answer
    # Extract all furniture items and total price in parallel
    extraction_tasks = []

    # Create extraction tasks for each furniture type
    for furniture_type in REQUIRED_TYPES:
        extraction_tasks.append(
            evaluator.extract(
                prompt=prompt_extract_furniture_item(furniture_type),
                template_class=FurnitureItem,
                extraction_name=f"{furniture_type.replace(' ', '_')}_extraction"
            )
        )

    # Also extract the stated total price
    extraction_tasks.append(
        evaluator.extract(
            prompt=prompt_extract_total_price(),
            template_class=TotalPrice,
            extraction_name="total_price_extraction"
        )
    )

    extraction_results = await asyncio.gather(*extraction_tasks)

    # Parse extraction results
    extracted_items = dict(zip(REQUIRED_TYPES, extraction_results[:-1]))
    stated_total = extraction_results[-1].total_price

    # 3. Build verification tree
    # Verify each furniture item and collect verified prices
    verified_prices = []

    for furniture_type in REQUIRED_TYPES:
        price = await verify_furniture_item(
            evaluator,
            root,
            extracted_items[furniture_type],
            furniture_type
        )
        verified_prices.append(price)

    # Calculate total price from verified items
    calculated_total = sum(verified_prices)

    # Add info about prices
    evaluator.add_custom_info(
        {
            "verified_prices": {
                k: verified_prices[i] for i, k in enumerate(REQUIRED_TYPES)
            },
            "calculated_total": calculated_total,
            "stated_total": stated_total,
            "budget_range": "$200-$600"
        },
        "price_summary"
    )

    # Verify budget compliance
    await verify_budget_compliance(
        evaluator,
        root,
        calculated_total,
        stated_total
    )

    # 4. Return structured result
    return evaluator.get_summary()
