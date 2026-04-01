import asyncio
import logging
from typing import Optional, List, Dict, Any
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ingredient_safety"
TASK_DESCRIPTION = """
Select one individual fragrance (excluding fragrance sets) from each of the following brands available on Sephora: Chanel, Jo Malone London, and Dior. For each selected fragrance, I would like to investigate the safety of its ingredients using the Environmental Working Group (EWG) Skin Deep Database. Specifically, for each fragrance, identify one ingredient that has a moderate or high rating on any common concern according to EWG, and provide the ingredient's name, the specific concern for which it has a moderate or high rating, and a direct link to its EWG profile.
"""

REQUIRED_BRANDS = ["Chanel", "Jo Malone London", "Dior"]
BRAND_MATCH_THRESHOLD = 0.8  # Similarity threshold for brand matching


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FragranceInfo(BaseModel):
    """Information about a selected fragrance."""
    brand: Optional[str] = None
    name: Optional[str] = None
    sephora_urls: List[str] = Field(default_factory=list)  # Changed to multiple URLs


class IngredientInfo(BaseModel):
    """Information about an ingredient with safety concerns."""
    fragrance_brand: Optional[str] = None  # To link back to the fragrance
    fragrance_name: Optional[str] = None  # To link back to the fragrance
    ingredient_name: Optional[str] = None
    concern: Optional[str] = None
    rating: Optional[str] = None  # Could be "Moderate", "High", etc.
    ewg_urls: List[str] = Field(default_factory=list)  # Changed to multiple URLs


class ExtractedFragrances(BaseModel):
    """List of fragrances extracted from the answer."""
    fragrances: List[FragranceInfo] = Field(default_factory=list)


class ExtractedIngredients(BaseModel):
    """List of ingredients with safety concerns extracted from the answer."""
    ingredients: List[IngredientInfo] = Field(default_factory=list)


class ExtractedUrls(BaseModel):
    """URLs extracted for a specific verification purpose."""
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Utility functions for brand matching                                        #
# --------------------------------------------------------------------------- #
def normalize_brand_name(brand_name: str) -> str:
    """Normalize brand name for comparison."""
    return brand_name.lower().strip().replace(' ', '').replace('-', '').replace('.', '')


def calculate_brand_similarity(brand1: str, brand2: str) -> float:
    """Calculate similarity between two brand names using sequence matching."""
    normalized1 = normalize_brand_name(brand1)
    normalized2 = normalize_brand_name(brand2)
    return SequenceMatcher(None, normalized1, normalized2).ratio()


def find_matching_fragrance(fragrances: List[FragranceInfo], target_brand: str) -> Optional[FragranceInfo]:
    """Find fragrance that matches the target brand with similarity threshold."""
    best_match = None
    best_score = 0.0

    for fragrance in fragrances:
        if fragrance.brand:
            similarity = calculate_brand_similarity(fragrance.brand, target_brand)
            if similarity >= BRAND_MATCH_THRESHOLD and similarity > best_score:
                best_score = similarity
                best_match = fragrance

    return best_match


def find_matching_ingredient(ingredients: List[IngredientInfo], target_brand: str) -> Optional[IngredientInfo]:
    """Find ingredient that matches the target brand with similarity threshold."""
    best_match = None
    best_score = 0.0

    for ingredient in ingredients:
        if ingredient.fragrance_brand:
            similarity = calculate_brand_similarity(ingredient.fragrance_brand, target_brand)
            if similarity >= BRAND_MATCH_THRESHOLD and similarity > best_score:
                best_score = similarity
                best_match = ingredient

    return best_match


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_fragrances() -> str:
    return """
    Extract information about the fragrances mentioned in the answer. For each fragrance, extract:
    1. The brand name (Chanel, Jo Malone London, or Dior)
    2. The full fragrance name
    3. ALL Sephora URLs related to the fragrance (extract all URLs that might contain product information, ingredient lists, or other relevant details about the fragrance)

    Focus only on individual fragrances, not fragrance sets. If a brand is mentioned without a specific fragrance, or if a brand is missing entirely, return null for the corresponding fields.
    For URLs, extract all potentially relevant Sephora links, not just the first one.
    """


def prompt_extract_ingredients() -> str:
    return """
    Extract information about ingredients with safety concerns mentioned in the answer. For each ingredient, extract:
    1. The associated fragrance brand (to link back to the fragrance)
    2. The associated fragrance name
    3. The ingredient name
    4. The specific concern (e.g., "Cancer", "Allergies & Immunotoxicity", "Developmental and Reproductive Toxicity", "Use Restrictions")
    5. The rating level (e.g., "Moderate", "High") 
    6. ALL EWG URLs for the ingredient (extract all URLs that might contain EWG information for this ingredient)

    If any information is missing, return null for the corresponding fields.
    For URLs, extract all potentially relevant EWG links, not just the first one.
    """


def prompt_extract_sephora_urls(brand: str, fragrance_name: str) -> str:
    return f"""
    Extract all URLs that might be associated with the fragrance "{fragrance_name}" from {brand} on Sephora.
    Focus specifically on Sephora URLs that would contain the fragrance's product page or ingredient list.
    Extract only complete and valid URLs. The URLs should be related to Sephora's website and specifically for this fragrance.
    Include ALL potentially relevant Sephora URLs, not just the primary product page.
    """


def prompt_extract_urls_for_ingredient(ingredient_name: str, brand: str) -> str:
    return f"""
    Extract all URLs that might be associated with the ingredient "{ingredient_name}" mentioned for the {brand} fragrance. 
    Focus specifically on Environmental Working Group (EWG) URLs or any URLs that might contain information about this ingredient's safety rating.
    Extract only complete and valid URLs. Include ALL URLs that are explicitly labeled as EWG links and any that appear to be related to the ingredient's safety profile.
    """


# --------------------------------------------------------------------------- #
# Verification functions for individual brands                                #
# --------------------------------------------------------------------------- #
async def verify_brand_fragrance(
        evaluator: Evaluator,
        parent_node,
        brand: str,
        fragrances: List[FragranceInfo],
        ingredients: List[IngredientInfo],
) -> None:
    """
    Verify fragrance selection and ingredient safety information for a specific brand.
    """
    brand_id = brand.lower().replace(' ', '_').replace('.', '')

    # Create a sequential node for this brand
    brand_node = evaluator.add_sequential(
        id=f"brand_{brand_id}",
        desc=f"Verify fragrance selection and ingredient safety information for {brand}",
        parent=parent_node
    )

    # Find matching fragrances for this brand using improved matching
    matching_fragrance = find_matching_fragrance(fragrances, brand)

    # Step 1: Verify fragrance selection using existence check
    fragrance_selection_node = evaluator.add_custom_node(
        result=matching_fragrance is not None,
        id=f"fragrance_selection_{brand_id}",
        desc=f"Verify that an individual {brand} fragrance was selected",
        parent=brand_node,
        critical=True
    )

    # Step 2: Verify ingredient safety information
    ingredient_node = evaluator.add_sequential(
        id=f"ingredient_verification_{brand_id}",
        desc=f"Verify ingredient safety information for the selected {brand} fragrance",
        parent=brand_node,
        critical=True
    )

    # Find matching ingredients for this fragrance using improved matching
    matching_ingredient = find_matching_ingredient(ingredients, brand)

    # 1. Check if ingredient name exists
    name_exists_node = evaluator.add_custom_node(
        result=matching_ingredient is not None and matching_ingredient.ingredient_name is not None,
        id=f"ingredient_name_exists_{brand_id}",
        desc=f"Check if an ingredient name is provided for the {brand} fragrance",
        parent=ingredient_node,
        critical=True
    )

    # 2. Verify the ingredient is actually in the fragrance
    ingredient_in_fragrance_parent = evaluator.add_parallel(
        id=f"ingredient_in_fragrance_parent_{brand_id}",
        desc=f"Verify ingredient presence in fragrance",
        parent=ingredient_node,
        critical=True
    )

    # Get all Sephora URLs for the fragrance
    sephora_urls = []
    if matching_fragrance:
        sephora_urls = matching_fragrance.sephora_urls.copy()

    # If no URLs directly provided, try to extract them
    if not sephora_urls and matching_fragrance and matching_fragrance.name:
        extracted_sephora_urls = await evaluator.extract(
            prompt=prompt_extract_sephora_urls(brand, matching_fragrance.name),
            template_class=ExtractedUrls,
            extraction_name=f"sephora_urls_{brand_id}"
        )
        sephora_urls = [url for url in extracted_sephora_urls.urls if "sephora" in url.lower()]

    # Check existence of URLs and ingredient
    ingredient_urls_exist = evaluator.add_custom_node(
        result=(len(sephora_urls) > 0),
        id=f"ingredient_urls_exist_{brand_id}",
        desc=f"Check if ingredient and Sephora URLs exist for verification",
        parent=ingredient_in_fragrance_parent,
        critical=True
    )

    ingredient_in_fragrance_node = evaluator.add_leaf(
        id=f"ingredient_in_fragrance_{brand_id}",
        desc=f"Verify that the ingredient is present in the {brand} fragrance",
        parent=ingredient_in_fragrance_parent,
        critical=True
    )

    # Always call verify regardless of data existence
    ingredient_claim = ""
    if matching_fragrance and matching_fragrance.name and matching_ingredient and matching_ingredient.ingredient_name:
        ingredient_claim = f"The fragrance '{matching_fragrance.name}' by {brand} contains the ingredient '{matching_ingredient.ingredient_name}'."
    
    await evaluator.verify(
        claim=ingredient_claim,
        node=ingredient_in_fragrance_node,
        sources=sephora_urls,
        additional_instruction="Verify whether the ingredient list for this fragrance on the Sephora page contains the specified ingredient. The ingredient might be listed in the 'Ingredients' section or in the product details. Note that ingredient names can sometimes be complex or have slight variations in spelling."
    )

    # 3. Verify EWG profile link is provided
    # Get all EWG URLs for the ingredient
    ewg_urls = []
    if matching_ingredient:
        ewg_urls = [url for url in matching_ingredient.ewg_urls if "ewg" in url.lower()]

    # If no EWG URLs directly provided, extract potential URLs from the answer
    if not ewg_urls and matching_ingredient and matching_ingredient.ingredient_name:
        extracted_urls = await evaluator.extract(
            prompt=prompt_extract_urls_for_ingredient(matching_ingredient.ingredient_name, brand),
            template_class=ExtractedUrls,
            extraction_name=f"ewg_urls_{brand_id}"
        )
        ewg_urls = [url for url in extracted_urls.urls if "ewg" in url.lower()]

    ewg_link_exists = evaluator.add_custom_node(
        result=len(ewg_urls) > 0,
        id=f"ewg_link_exists_{brand_id}",
        desc=f"Check if EWG profile link is provided for the {brand} fragrance ingredient",
        parent=ingredient_node,
        critical=True
    )


    # 4. Verify EWG link substantiates the identified concern and rating
    concern_rating_parent = evaluator.add_parallel(
        id=f"concern_rating_{brand_id}",
        desc=f"Verify EWG link substantiation",
        parent=ingredient_node,
        critical=True
    )

    concern_rating_exists = evaluator.add_custom_node(
        result=(matching_ingredient is not None and 
                matching_ingredient.ingredient_name is not None and
                matching_ingredient.concern is not None and
                len(ewg_urls) > 0),
        id=f"concern_data_exists_{brand_id}",
        desc=f"Check if data exists for EWG concern rating verification",
        parent=concern_rating_parent,
        critical=True
    )

    provenance_node = evaluator.add_leaf(
        id=f"provenance_{brand_id}",
        desc=f"Verify that the EWG link substantiates the identified concern and rating",
        parent=concern_rating_parent,
        critical=True
    )

    # Construct claim for verification
    claim = ""
    if matching_ingredient and matching_ingredient.ingredient_name and matching_ingredient.concern:
        claim = f"The ingredient '{matching_ingredient.ingredient_name}' has either a moderate or high rating for the concern '{matching_ingredient.concern}' according to EWG's Skin Deep Database."

    await evaluator.verify(
        claim=claim,
        node=provenance_node,
        sources=ewg_urls,
        additional_instruction="Verify whether the webpage contains information confirming that this specific ingredient. The information doesn't need to use exactly the same wording, but it should substantiate the core claim about the ingredient's specific concern and whether there's a moderate or high rating."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: Any,  # OpenAI client
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
    # -------- 1. Initialize evaluator ------------------------------------ #
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
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

    # -------- 2. Extract structured info from the answer ----------------- #
    # Extract fragrances
    fragrances_info = await evaluator.extract(
        prompt=prompt_extract_fragrances(),
        template_class=ExtractedFragrances,
        extraction_name="fragrances"
    )

    # Extract ingredients with safety concerns
    ingredients_info = await evaluator.extract(
        prompt=prompt_extract_ingredients(),
        template_class=ExtractedIngredients,
        extraction_name="ingredients"
    )

    # -------- 3. Build verification tree -------------------------------- #
    # Verify each brand's fragrance and ingredient information
    for brand in REQUIRED_BRANDS:
        await verify_brand_fragrance(
            evaluator=evaluator,
            parent_node=evaluator.root,
            brand=brand,
            fragrances=fragrances_info.fragrances,
            ingredients=ingredients_info.ingredients
        )

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()