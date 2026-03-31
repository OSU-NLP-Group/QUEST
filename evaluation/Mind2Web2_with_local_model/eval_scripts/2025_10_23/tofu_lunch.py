import asyncio
import logging
from typing import Optional, List, Dict, Any, Set
from collections import defaultdict

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient

TASK_ID = "tofu_lunch"
TASK_DESCRIPTION = """
I want to make 3 main dishes and one dessert that use tofu as the primary ingredient. Please provide a link to each recipe along with a list of all ingredients and seasonings used. Across all four recipes, there should be no more than 30 unique ingredients or seasonings in total (excluding tofu). Finally, compile a combined shopping list of all the ingredients and seasonings used (except tofu). For both the ingredient count and the shopping list, exclude common staples such as water and cooking oils. However, basic seasonings like salt and pepper should still be included.
"""

EVAL_NOTES = ""  # No additional evaluation notes provided
GROUND_TRUTH = {}  # No ground truth provided

# Configuration
INGREDIENT_COUNT_LIMIT = 30
INGREDIENT_COUNT_TOLERANCE = 3


class ShoppingList(BaseModel):
    """Combined shopping list from answer"""
    ingredients: List[str] = Field(default_factory=list,
                                   description="Combined shopping list of ingredients (excluding tofu, water, cooking oils)")


class RecipeNames(BaseModel):
    """Names of all recipes"""
    main_dishes: List[str] = Field(default_factory=list, description="Names of main dish recipes")
    dessert: Optional[str] = Field(default=None, description="Name of dessert recipe")


class SingleRecipeInfo(BaseModel):
    """Detailed information for a single recipe"""
    name: Optional[str] = Field(default=None, description="Recipe name")
    url: Optional[str] = Field(default=None, description="Recipe URL")
    ingredients: List[str] = Field(default_factory=list, description="List of ingredients and seasonings")


class UniqueIngredientsList(BaseModel):
    """List of unique ingredients across all recipes"""
    ingredients: List[str] = Field(default_factory=list, 
                                   description="List of unique ingredients excluding tofu, water, and oils")


def prompt_extract_shopping_list() -> str:
    """Extract the combined shopping list from answer"""
    return """
    Extract the combined shopping list mentioned in the answer.
    
    Look for a section that compiles all ingredients across all recipes.
    This is typically presented as a "combined shopping list" or "shopping list".
    
    Extract the list exactly as it appears in the answer.
    If no explicit combined shopping list is provided, return an empty list.
    """


def prompt_extract_recipe_names() -> str:
    """Extract the names of all recipes"""
    return """
    Extract the names of all tofu recipes mentioned in the answer.

    Look for:
    1. Three main dish recipes that use tofu
    2. One dessert recipe that uses tofu

    Extract the recipe names exactly as they appear in the answer.
    If fewer recipes are provided, extract only what is available.
    """


def prompt_extract_single_recipe(recipe_name: str, recipe_type: str) -> str:
    """Extract information for a specific recipe"""
    return f"""
    Extract detailed information for the recipe: "{recipe_name}"

    This is a {recipe_type} recipe.

    Extract:
    1. The recipe name (should match "{recipe_name}")
    2. The URL/link to the recipe
    3. All ingredients and seasonings listed for this specific recipe
       - Include everything mentioned for this recipe
       - Do NOT exclude tofu, water, or oils at this stage (we need the complete list)

    Extract information exactly as it appears in the answer for this specific recipe.
    """


def prompt_extract_unique_ingredients(all_recipe_details: List[SingleRecipeInfo]) -> str:
    """Create prompt for extracting unique ingredients using LLM"""
    # Compile all ingredients from all recipes
    all_ingredients = []
    for recipe in all_recipe_details:
        if recipe.ingredients and recipe.name:
            all_ingredients.append(f"\nRecipe: {recipe.name}")
            for ingredient in recipe.ingredients:
                all_ingredients.append(f"  - {ingredient}")
    
    ingredients_text = "\n".join(all_ingredients) if all_ingredients else "No ingredients found"
    
    return f"""
    Extract a list of UNIQUE ingredients across all recipes below.
    
    Important rules:
    1. EXCLUDE: tofu (any form), water, cooking oils (vegetable oil, canola oil, olive oil, etc.)
    2. INCLUDE: all other ingredients including basic seasonings (salt, pepper, etc.)
    3. List each unique ingredient only ONCE
    4. Merge similar ingredients that are essentially the same:
       - "garlic", "minced garlic", "garlic cloves" -> just include as "garlic"
       - "soy sauce", "low-sodium soy sauce" -> just include as "soy sauce"
       - "onion", "diced onion", "chopped onion" -> just include as "onion"
       - Different forms/preparations of the same ingredient should be merged
    
    Here are all the ingredients from all recipes:
    {ingredients_text}
    
    Return a clean list of unique ingredients (one per item in the list).
    """


async def verify_recipe_comprehensive(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        recipe_info: SingleRecipeInfo,
        recipe_type: str,
        index: int,
) -> None:
    """
    Comprehensive verification for a single recipe
    """
    recipe_id = f"{recipe_type}_{index}"

    # Create container node for this recipe
    recipe_node = evaluator.add_parallel(
        id=f"{recipe_id}_verification",
        desc=f"{recipe_type.replace('_', ' ').title()} {index}: {recipe_info.name or '(Missing)'}",
        parent=parent_node,
        critical=False,  # Non-critical for partial credit
    )

    # Existence check for all required components
    has_name = bool(recipe_info.name and recipe_info.name.strip())
    has_url = bool(recipe_info.url and recipe_info.url.strip())
    has_ingredients = bool(recipe_info.ingredients and len(recipe_info.ingredients) > 0)

    existence_node = evaluator.add_custom_node(
        result=has_name and has_url and has_ingredients,
        id=f"{recipe_id}_exists",
        desc=f"Recipe has name, URL, and ingredients list",
        parent=recipe_node,
        critical=True,  # Critical - if basic info missing, skip verification
    )

    # Comprehensive URL verification
    url_verification_node = evaluator.add_leaf(
        id=f"{recipe_id}_url_verification",
        desc=f"URL verification for {recipe_info.name or 'recipe'}",
        parent=recipe_node,
        critical=True,
    )

    # Create comprehensive claim for URL verification
    ingredients_str = ", ".join(recipe_info.ingredients[:5]) if recipe_info.ingredients else "no ingredients"
    if len(recipe_info.ingredients) > 5:
        ingredients_str += f" and {len(recipe_info.ingredients) - 5} more"

    verification_claim = f"""
    The webpage contains a recipe that satisfies ALL of the following:
    1. The recipe matches "{recipe_info.name or 'unknown recipe'}" (allow reasonable variations)
    2. The recipe uses tofu as a primary/main ingredient
    3. The ingredients listed in the answer ({ingredients_str}) generally match those on the webpage
       - Allow reasonable variations like "tofu" vs "firm tofu" vs "silken tofu"
       - Allow omission of water, oil, or very basic ingredients
       - Allow minor differences in ingredient descriptions
       - The core ingredients should align without major fabrications or omissions
    """

    await evaluator.verify(
        claim=verification_claim.strip(),
        node=url_verification_node,
        sources=recipe_info.url if recipe_info.url else [],
        additional_instruction="Be flexible with ingredient matching. Focus on whether the core ingredients align, not exact wording. Tofu variations, seasoning variations, and minor differences are all acceptable."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                  #
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
    Main evaluation function for the tofu lunch task.

    Evaluation structure:
    1. Critical node: Total unique ingredients ≤ 30 + tolerance
    2. Non-critical container with 4 recipe verifications (3 main dishes + 1 dessert)
    """

    # -------- 1. Initialize evaluator ----------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
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

    # -------- 2. Extract recipe names first ----------------------- #
    recipe_names = await evaluator.extract(
        prompt=prompt_extract_recipe_names(),
        template_class=RecipeNames,
        extraction_name="recipe_names",
    )

    # -------- 3. Extract shopping list from answer ---------------- #
    shopping_list = await evaluator.extract(
        prompt=prompt_extract_shopping_list(),
        template_class=ShoppingList,
        extraction_name="shopping_list",
    )

    # -------- 4. Extract details for all recipes ------------------ #
    all_recipe_details = []
    
    # Process main dishes (up to 3)
    for i in range(3):
        if i < len(recipe_names.main_dishes):
            recipe_info = await evaluator.extract(
                prompt=prompt_extract_single_recipe(recipe_names.main_dishes[i], "main dish"),
                template_class=SingleRecipeInfo,
                extraction_name=f"main_dish_{i + 1}_details",
            )
        else:
            # Create empty recipe for missing main dish
            recipe_info = SingleRecipeInfo()
        all_recipe_details.append(recipe_info)
    
    # Process dessert
    if recipe_names.dessert:
        dessert_info = await evaluator.extract(
            prompt=prompt_extract_single_recipe(recipe_names.dessert, "dessert"),
            template_class=SingleRecipeInfo,
            extraction_name="dessert_details",
        )
    else:
        # Create empty recipe for missing dessert
        dessert_info = SingleRecipeInfo()
    all_recipe_details.append(dessert_info)

    # -------- 5. Extract unique ingredients list using LLM -------- #
    unique_ingredients_info = await evaluator.extract(
        prompt=prompt_extract_unique_ingredients(all_recipe_details),
        template_class=UniqueIngredientsList,
        extraction_name="unique_ingredients_list",
    )
    
    # Count using Python
    unique_count = len(unique_ingredients_info.ingredients)

    # -------- 6. Critical nodes: ingredient count and correctness - #
    critical_container = evaluator.add_parallel(
        id="critical_checks",
        desc="Critical ingredient requirements",
        parent=root,
        critical=True,  # Critical container
    )

    # Check ingredient count
    count_valid = unique_count <= (INGREDIENT_COUNT_LIMIT + INGREDIENT_COUNT_TOLERANCE)
    evaluator.add_custom_node(
        result=count_valid,
        id="ingredient_count_valid",
        desc=f"Actual unique ingredient count ({unique_count}) is within limit of {INGREDIENT_COUNT_LIMIT} + tolerance {INGREDIENT_COUNT_TOLERANCE}",
        parent=critical_container,
        critical=True,
    )

    # Verify shopping list correctness using LLM
    shopping_list_verification = evaluator.add_leaf(
        id="shopping_list_correctness",
        desc="Shopping list matches calculated unique ingredients",
        parent=critical_container,
        critical=True,
    )

    # Create full lists for comparison - no clipping
    unique_ingredients_str = ", ".join(unique_ingredients_info.ingredients)
    shopping_list_str = ", ".join(shopping_list.ingredients)
    
    claim = f"""
    The provided shopping list ({len(shopping_list.ingredients)} items) reasonably corresponds to 
    the unique ingredients needed across all four recipes ({unique_count} unique items).
    
    Unique ingredients from recipes:
    {unique_ingredients_str}
    
    Shopping list provided:
    {shopping_list_str}
    
    The shopping list should:
    1. Contain most or all of the unique ingredients from the recipes
    2. Properly exclude tofu, water, and cooking oils as specified
    3. If basic seanonings (such as salt and pepper) appear in the unique ingredients, they should also be included in the shopping list
    4. Have reasonable consolidation of similar items (e.g., one entry for garlic instead of separate entries for minced/fresh garlic)
    
    The lists should substantially match with allowance for reasonable variations in how ingredients are listed or consolidated.
    """

    await evaluator.verify(
        claim=claim.strip(),
        node=shopping_list_verification,
        additional_instruction="""
        Be flexible with the comparison. Consider these as matching:
        - Different forms of the same ingredient (minced garlic = garlic)
        - Minor spelling variations or different descriptions
        - Reasonable consolidations in the shopping list
        Focus on whether the shopping list captures the essential ingredients needed for all recipes.
        A difference of 1-3 items due to consolidation or interpretation is acceptable.
        The order of items does not matter.
        """
    )

    # -------- 7. Create recipes container (non-critical) ---------- #
    recipes_container = evaluator.add_parallel(
        id="all_recipes",
        desc="All recipe verifications (3 main dishes + 1 dessert)",
        parent=root,
        critical=False,  # Non-critical for partial credit
    )

    # -------- 8. Verify all recipes uniformly --------------------- #
    # Verify main dishes
    for i in range(3):
        await verify_recipe_comprehensive(
            evaluator,
            recipes_container,
            all_recipe_details[i],
            "main_dish",
            i + 1
        )
    
    # Verify dessert
    await verify_recipe_comprehensive(
        evaluator,
        recipes_container,
        all_recipe_details[3],
        "dessert",
        1
    )

    # -------- 9. Add detailed info to summary --------------------- #
    evaluator.add_custom_info(
        {
            "calculated_unique_ingredients": unique_ingredients_info.ingredients,
            "calculated_unique_count": unique_count,
            "provided_shopping_list": shopping_list.ingredients,
            "provided_shopping_list_count": len(shopping_list.ingredients),
            "count_limit": INGREDIENT_COUNT_LIMIT,
            "tolerance": INGREDIENT_COUNT_TOLERANCE,
        },
        info_type="ingredient_analysis",
        info_name="ingredient_summary"
    )

    # -------- 10. Return evaluation results ----------------------- #
    return evaluator.get_summary()