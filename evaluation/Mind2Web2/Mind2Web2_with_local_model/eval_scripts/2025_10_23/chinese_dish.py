import asyncio
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "chinese_dish"
TASK_DESCRIPTION = """
I want to cook two delicious Chinese dishes using what I have in my fridge: potatoes, chicken thighs, cilantro, scallions, and cucumbers. Please help me find two recipes that, together, make full use of all these ingredients. It's okay if the recipes share some ingredients, but they should not be exactly the same.
For each recipe, please:
- List all the required ingredients.
- Specify any additional ingredients and seasonings I need to buy.
Note: The only seasonings I currently have are oil, salt, MSG, soy sauce, and vinegar. And of course, I have water 😊
"""

# Available ingredients
AVAILABLE_INGREDIENTS = ["potatoes", "chicken thighs", "cilantro", "scallions", "cucumbers"]
AVAILABLE_SEASONINGS = ["oil", "salt", "msg", "soy sauce", "vinegar", "water"]


# --------------------------------------------------------------------------- #
# Data models for extracted info                                             #
# --------------------------------------------------------------------------- #

class RecipeName(BaseModel):
    name: Optional[str] = None


class RecipeNameList(BaseModel):
    recipe_names: List[RecipeName] = Field(default_factory=list)


class Ingredient(BaseModel):
    name: str
    is_seasoning: bool


class RecipeInfo(BaseModel):
    name: Optional[str] = None
    ingredients: List[Ingredient] = Field(default_factory=list)
    additional_ingredients: List[Ingredient] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #

def prompt_extract_recipe_names() -> str:
    return """
    Extract the names of all Chinese recipes provided in the answer (even if more than two are present), and only extract their names. Make sure the recipes are distinct and clearly indicated as a chinese recipe. 
    """


def prompt_extract_recipe_info(chinese_recipe: str) -> str:
    return f"""
    Extract information about the recipe "{chinese_recipe}" from the answer. Follow these specific instructions for each field:

    1. **name**: The exact recipe name as mentioned in the answer (should match or be very similar to "{chinese_recipe}")

    2. **ingredients**: Extract ALL ingredients mentioned for this recipe, including:
       - The 5 available ingredients (potatoes, chicken thighs, cilantro, scallions, cucumbers) if used
       - The available seasonings (oil, salt, MSG, soy sauce, vinegar, water) if used
       - Any additional ingredients that need to be purchased
       - Mark each ingredient with is_seasoning=true if it's a seasoning/spice/sauce/condiment, false otherwise

    3. **additional_ingredients**: Extract ONLY the ingredients that need to be purchased (not already available). This should be a subset of the ingredients list above, containing only items that are:
       - NOT in the available ingredients list (potatoes, chicken thighs, cilantro, scallions, cucumbers)
       - NOT in the available seasonings list (oil, salt, MSG, soy sauce, vinegar, water)
       - Again, mark each with is_seasoning=true or false

    4. **urls**: Extract ALL URLs provided in the answer that are related to this specific recipe. Include any links to recipe pages, cooking websites, or sources for this dish.

    Available ingredients: potatoes, chicken thighs, cilantro, scallions, cucumbers
    Available seasonings: oil, salt, MSG, soy sauce, vinegar, water

    Return empty lists if no items found for a field. Return null for name only if the recipe name cannot be determined.
    """

# --------------------------------------------------------------------------- #
# Individual recipe verification functions                                    #
# --------------------------------------------------------------------------- #     
async def verify_recipe(
        evaluator: Evaluator,
        parent_node,
        recipe_info: RecipeInfo,
        other_recipe_info: Optional[RecipeInfo],
        recipe_index: int
) -> None:
    """Verify all aspects of a single recipe."""
    
    # Recipe-level verification node
    recipe_node = evaluator.add_parallel(
        id=f"recipe_{recipe_index}_verification",
        desc=f"Recipe #{recipe_index} verification",
        parent=parent_node,
        critical=False  # Non-critical to allow partial credit
    )
    
    # 1. Completeness check
    recipe_complete = evaluator.add_custom_node(
        result=(
            recipe_info.name is not None and 
            len(recipe_info.ingredients) > 0 and 
            len(recipe_info.urls) > 0
        ),
        id=f"recipe_{recipe_index}_complete",
        desc=f"Recipe #{recipe_index} has name, ingredients, and URLs",
        parent=recipe_node,
        critical=True
    )
    
    # 2. Chinese dish verification
    chinese_node = evaluator.add_leaf(
        id=f"recipe_{recipe_index}_chinese",
        desc=f"Recipe #{recipe_index} is a Chinese dish",
        parent=recipe_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The recipe named '{recipe_info.name or 'Unknown'}' is a Chinese dish, as cited in the recipe pages",
        node=chinese_node,
        sources=recipe_info.urls
    )
    
    # 3. Ingredients verification
    ingredients_node = evaluator.add_leaf(
        id=f"recipe_{recipe_index}_ingredients_verified",
        desc=f"Recipe #{recipe_index} ingredients are accurately listed",
        parent=recipe_node,
        critical=True
    )
    
    ingredients_str = ", ".join([ing.name for ing in recipe_info.ingredients])
    await evaluator.verify(
        claim=f"The recipe named '{recipe_info.name or 'Unknown'}' contains the following ingredients: {ingredients_str}",
        node=ingredients_node,
        sources=recipe_info.urls,
        additional_instruction="Verify that all listed ingredients are actually mentioned in the recipe. Ingredients may be considered the same up to slightly different spellings or synonyms. For example, 'potato' and 'potatoes' are the same ingredient, as are 'cilantro' and 'coriander'."
    )
    
    # 4. Distinctness check (only for recipe 2)
    if recipe_index == 2 and other_recipe_info is not None:
        distinct_node = evaluator.add_leaf(
            id=f"recipe_{recipe_index}_distinct_from_1",
            desc=f"Recipe #{recipe_index} is distinct from Recipe #1",
            parent=recipe_node,
            critical=True
        )
        
        recipe_1_name = other_recipe_info.name or 'Recipe 1'
        recipe_2_name = recipe_info.name or 'Recipe 2'
        recipe_1_ing_str = ", ".join([ing.name for ing in other_recipe_info.ingredients])
        recipe_2_ing_str = ", ".join([ing.name for ing in recipe_info.ingredients])
        
        await evaluator.verify(
            claim=f"""Verify these are two distinctly different Chinese dishes:
            
            Recipe 1: '{recipe_1_name}' with ingredients ({recipe_1_ing_str})
            Recipe 2: '{recipe_2_name}' with ingredients ({recipe_2_ing_str})
            
            These should be two fundamentally different dishes (like 'Mapo Tofu' vs 'Kung Pao Chicken'), not just variations or different names for the same dish (like 'Kung Pao Chicken' vs 'Gong Bao Chicken').""",
            node=distinct_node,
            additional_instruction="Consider the dishes different if they would be listed as separate items on a Chinese restaurant menu. Similar ingredients are fine as long as the dishes themselves are distinct."
        )
    
    # 5. Verify additional ingredients are exactly those needing purchase
    additional_correct_node = evaluator.add_leaf(
        id=f"recipe_{recipe_index}_additional_correct",
        desc=f"Recipe #{recipe_index} additional ingredients are exactly those needing purchase",
        parent=recipe_node,
        critical=False
    )

    # Get the string representations
    all_ingredients_str = ", ".join([ing.name for ing in recipe_info.ingredients])
    additional_ingredients_str = ", ".join([ing.name for ing in recipe_info.additional_ingredients]) if recipe_info.additional_ingredients else "none"
    available_ingredients_str = ", ".join(AVAILABLE_INGREDIENTS)
    available_seasonings_str = ", ".join(AVAILABLE_SEASONINGS)

    # Let the LLM do the fuzzy matching
    if recipe_info.additional_ingredients:
        claim = f"""Given that:
        - The recipe uses these ingredients: {all_ingredients_str}
        - Available ingredients are: {available_ingredients_str}
        - Available seasonings are: {available_seasonings_str}
        - The additional ingredients list contains: {additional_ingredients_str}
        
        Verify that the additional ingredients list contains ALL and ONLY the ingredients from the recipe that are not in the available ingredients or seasonings."""
    else:
        claim = f"""Given that:
        - The recipe uses these ingredients: {all_ingredients_str}
        - Available ingredients are: {available_ingredients_str}
        - Available seasonings are: {available_seasonings_str}
        - The additional ingredients list is empty
        
        Verify that ALL ingredients used in this recipe are already available (either in available ingredients or seasonings), so no additional ingredients need to be purchased."""

    await evaluator.verify(
        claim=claim,
        node=additional_correct_node,
        additional_instruction="Consider ingredients the same despite minor spelling variations (potato/potatoes), different names for the same item (scallion/green onion, cilantro/coriander), or specializations (baby potatoes are still potatoes)."
    )
    
# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # All requirements evaluated independently
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
    
    # Extract recipe names
    recipe_name_list = await evaluator.extract(
        prompt=prompt_extract_recipe_names(),
        template_class=RecipeNameList,
        extraction_name="recipe_names"
    )
    
    # Extract info for each recipe (limit to 2)
    recipe_info_list = []
    for i, recipe_name in enumerate(recipe_name_list.recipe_names[:2]):
        if recipe_name.name:
            recipe_info = await evaluator.extract(
                prompt=prompt_extract_recipe_info(recipe_name.name),
                template_class=RecipeInfo,
                extraction_name=f"recipe_{i+1}_info"
            )
            recipe_info_list.append(recipe_info)
    
    # Pad to ensure we have 2 recipes
    while len(recipe_info_list) < 2:
        recipe_info_list.append(RecipeInfo())
    
    # Verify Recipe 1
    await verify_recipe(
        evaluator=evaluator,
        parent_node=root,
        recipe_info=recipe_info_list[0],
        other_recipe_info=None,  # Recipe 1 doesn't need to check distinctness
        recipe_index=1
    )
    
    # Verify Recipe 2 (including distinctness from Recipe 1)
    await verify_recipe(
        evaluator=evaluator,
        parent_node=root,
        recipe_info=recipe_info_list[1],
        other_recipe_info=recipe_info_list[0],  # Pass Recipe 1 for distinctness check
        recipe_index=2
    )
    
    # Verify all ingredients are used
    all_ingredients_node = evaluator.add_leaf(
        id="all_ingredients_used",
        desc="All available ingredients are used across the two recipes",
        parent=root,
        critical=True 
    )
    
    # Collect all non-seasoning ingredients from both recipes
    all_recipe_ingredients = set()
    for recipe in recipe_info_list:
        for ing in recipe.ingredients:
            if not ing.is_seasoning:
                all_recipe_ingredients.add(ing.name.lower())
    
    combined_ing_str = ", ".join(sorted(all_recipe_ingredients)) if all_recipe_ingredients else "none"
    
    await evaluator.verify(
        claim=f"The combined ingredients from both recipes ({combined_ing_str}) include all of these required ingredients: {', '.join(AVAILABLE_INGREDIENTS)}",
        node=all_ingredients_node,
        additional_instruction="Verify that every available ingredient appears in at least one recipe. Ingredients may be considered the same up to slightly different spellings or synonyms."
    )
    
    # Return result
    return evaluator.get_summary()