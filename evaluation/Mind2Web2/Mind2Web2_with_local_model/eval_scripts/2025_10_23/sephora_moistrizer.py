import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient

TASK_ID = "sephora_moistrizer"
TASK_DESCRIPTION = """
Find three moisturizers on Sephora that contain Squalane and Ceramide, but do not contain Dimethicone. Provide their purchase links on Sephora, brand names and prices.
"""

EVAL_NOTES = ""
GROUND_TRUTH = {}


class MoisturizerInfo(BaseModel):
    """Information for a single moisturizer"""
    brand_name: Optional[str] = Field(default=None, description="Brand name of the moisturizer")
    purchase_link: Optional[str] = Field(default=None, description="Sephora purchase link")
    price: Optional[str] = Field(default=None, description="Price of the moisturizer")


class ExtractedBrands(BaseModel):
    """Extracted brand names"""
    brand_names: List[str] = Field(default_factory=list, description="List of brand names for moisturizers")


class ExtractedDetails(BaseModel):
    """Extracted details for all moisturizers"""
    moisturizers: List[MoisturizerInfo] = Field(default_factory=list, description="List of moisturizer details")


def prompt_extract_brand_names() -> str:
    """Extraction prompt for getting brand names"""
    return """
    Extract the brand names of moisturizers mentioned in the answer.

    Look for:
    - Brand names of moisturizers (e.g., "La Mer", "Clinique", "Drunk Elephant", etc.)
    - Extract brand names in the order they appear in the answer

    Extract brand names exactly as they appear in the text.
    Return a list of brand names.
    """


def prompt_extract_details(brand_names: List[str]) -> str:
    """Extraction prompt for getting purchase links and prices"""
    brands_str = ", ".join(f'"{brand}"' for brand in brand_names)

    return f"""
    Extract the purchase links and prices for moisturizers from these brands: {brands_str}

    For each brand mentioned above, look for:
    - purchase_link: The Sephora URL where the moisturizer can be purchased
    - price: The price of the moisturizer (e.g., "$45", "$120.00", etc.)

    Important:
    - Match the details to the brands in the order provided
    - Extract URLs exactly as they appear, including the full URL
    - Extract prices exactly as they appear in the text
    - If any information is missing for a brand, set that field to null
    - Return details for all brands listed, even if some information is missing
    """


async def verify_moisturizer(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        moisturizer: MoisturizerInfo,
        index: int,
) -> None:
    """
    Verify a single moisturizer meets all requirements

    Args:
        evaluator: The evaluator instance
        parent_node: Parent node in the verification tree
        moisturizer: Moisturizer information to verify
        index: Index of the moisturizer (1-based for display)
    """
    # Create a container node for this moisturizer
    moisturizer_node = evaluator.add_sequential(
        id=f"moisturizer_{index}",
        desc=f"Moisturizer {index}: {moisturizer.brand_name or 'Unknown'}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial scoring
    )

    # 1. Existence check - all required fields must be present
    existence_check = evaluator.add_custom_node(
        result=bool(
            moisturizer.brand_name and moisturizer.brand_name.strip() and
            moisturizer.purchase_link and moisturizer.purchase_link.strip() and
            moisturizer.price and moisturizer.price.strip()
        ),
        id=f"moisturizer_{index}_exists",
        desc=f"Moisturizer {index} has all required information (brand, link, price)",
        parent=moisturizer_node,
        critical=True,  # Critical - if any field is missing, other checks are meaningless
    )

    # 2. Verify it's a valid Sephora moisturizer page with correct details
    sephora_verification_node = evaluator.add_leaf(
        id=f"moisturizer_{index}_sephora_valid",
        desc=f"Moisturizer {index} is from Sephora with correct name and price",
        parent=moisturizer_node,
        critical=True,  # Critical - must be a valid Sephora product
    )

    # Verify the URL is a Sephora moisturizer page with correct details
    claim = f"""The webpage at {moisturizer.purchase_link or 'N/A'} is:
    1. A valid Sephora.com product page
    2. For a specific moisturizer product
    3. From the brand "{moisturizer.brand_name or 'N/A'}"
    4. Priced at {moisturizer.price or 'N/A'}"""

    await evaluator.verify(
        claim=claim,
        node=sephora_verification_node,
        sources=moisturizer.purchase_link,
        additional_instruction="""Verify all four points:
        1. Check if the URL domain is sephora.com
        2. Confirm it's a product page for a specific moisturizer/skincare product (not serum, cleanser, etc.)
        3. Verify the brand name matches (allow reasonable name variations)
        4. Check the price matches (allow for sale/original prices or slight variations). For any price number, allow +-1 dollar variation.
        All four conditions must be met for verification to pass."""
    )

    # 3. Verify ingredients meet requirements
    ingredients_verification_node = evaluator.add_leaf(
        id=f"moisturizer_{index}_ingredients",
        desc=f"Moisturizer {index} contains Squalane and Ceramide, but not Dimethicone",
        parent=moisturizer_node,
        critical=True,  # Critical - must meet ingredient requirements
    )

    # Verify ingredients
    claim = f"""The moisturizer "{moisturizer.brand_name or 'Unknown'}" at {moisturizer.purchase_link or 'N/A'} has ingredients that:
    1. Contains Squalane (or Squalene)
    2. Contains Ceramide (or Ceramides, including variants like Ceramide NP, Ceramide AP, etc.)
    3. Does NOT contain Dimethicone (or any silicone variants ending in -cone or -siloxane)"""

    await evaluator.verify(
        claim=claim,
        node=ingredients_verification_node,
        sources=moisturizer.purchase_link,
        additional_instruction="""Check the ingredients list carefully:
        1. Look for "Squalane" or "Squalene" in the ingredients
        2. Look for any form of Ceramide (Ceramide, Ceramides, Ceramide NP, Ceramide AP, Ceramide EOP, etc.)
        3. Ensure there is NO Dimethicone or similar silicones (Cyclopentasiloxane, Dimethiconol, etc.)
        All three conditions must be met. The ingredients are usually listed on the product page."""
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                               #
# --------------------------------------------------------------------------- #
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
    """
    Main evaluation function for Sephora moisturizer task.

    This function:
    1. Extracts brand names first
    2. Then extracts purchase links and prices for each brand
    3. Verifies each moisturizer meets all requirements
    4. Returns evaluation summary with partial scoring
    """

    # -------- 1. Initialize evaluator ----------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel to allow partial scoring
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

    # -------- 2. Extract brand names first ------------------------ #
    logger.info("Extracting brand names...")
    brands_info = await evaluator.extract(
        prompt=prompt_extract_brand_names(),
        template_class=ExtractedBrands,
        extraction_name="brand_names_extraction",
    )

    # -------- 3. Extract details for found brands ----------------- #
    logger.info(f"Found {len(brands_info.brand_names)} brands, extracting details...")

    if brands_info.brand_names:
        details_info = await evaluator.extract(
            prompt=prompt_extract_details(brands_info.brand_names),
            template_class=ExtractedDetails,
            extraction_name="moisturizer_details_extraction",
        )
    else:
        # No brands found, create empty details
        details_info = ExtractedDetails()

    # -------- 4. Build verification tree -------------------------- #
    logger.info(f"Verifying {len(details_info.moisturizers)} moisturizers...")

    # Ensure we have exactly 3 moisturizers (add placeholders if needed)
    moisturizers = details_info.moisturizers[:3]  # Take first 3 if more than 3
    while len(moisturizers) < 3:
        moisturizers.append(MoisturizerInfo())  # Add empty placeholder

    # Verify each moisturizer
    for i, moisturizer in enumerate(moisturizers, 1):
        await verify_moisturizer(evaluator, root, moisturizer, i)

    # -------- 5. Return evaluation results ------------------------ #
    return evaluator.get_summary()