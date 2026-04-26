import asyncio
import logging
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "salt_dough_christmas_ornaments_tutorial"
TASK_DESCRIPTION = """
Find a comprehensive online tutorial or guide for making traditional salt dough Christmas ornaments. The tutorial must include all of the following elements: (1) a complete recipe with specific measurements for all three basic ingredients (all-purpose flour, salt, and water); (2) preparation instructions that specify the kneading duration and the thickness to which the dough should be rolled; (3) complete baking instructions with a specific oven temperature and either a specific baking duration or a clear endpoint indicator (such as 'until hard and dry'); and (4) at least one method for preserving or sealing the finished ornaments to ensure long-term durability. Provide the URL of the tutorial and explain how it satisfies each of these requirements.
"""


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class TutorialExtraction(BaseModel):
    # The main tutorial URL the answer cites
    tutorial_url: Optional[str] = None

    # Explanation fields (copied from the answer text; used to check presence)
    explains_ingredients: Optional[str] = None
    explains_measurements: Optional[str] = None
    explains_roles: Optional[str] = None
    explains_kneading: Optional[str] = None
    explains_thickness: Optional[str] = None
    explains_holes: Optional[str] = None
    explains_temp: Optional[str] = None
    explains_duration_or_endpoint: Optional[str] = None
    explains_flipping: Optional[str] = None
    explains_preservation: Optional[str] = None
    explains_durability: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tutorial_info() -> str:
    return """
    Your task is to extract from the ANSWER the single main tutorial URL and the explanation snippets for each required compliance item.

    Extract:
    - tutorial_url: The primary URL to the salt dough ornaments tutorial/guide that the answer cites. If multiple URLs are present, choose the single URL that is the main tutorial/guide; if none are present, set to null.
    - explains_ingredients: The answer’s explanation text showing where/how the tutorial includes the three ingredients (flour, salt, water). If absent, null.
    - explains_measurements: The answer’s explanation text showing where/how the tutorial provides specific measurements for flour, salt, and water. If absent, null.
    - explains_roles: The answer’s explanation text showing where/how the tutorial indicates ingredient roles (flour as primary, salt as key ingredient, water as binder). If absent, null.
    - explains_kneading: The answer’s explanation text showing where/how the tutorial specifies a kneading duration (time). If absent, null.
    - explains_thickness: The answer’s explanation text showing where/how the tutorial specifies rolling thickness. If absent, null.
    - explains_holes: The answer’s explanation text showing where/how the tutorial instructs making holes for hanging before baking. If absent, null.
    - explains_temp: The answer’s explanation text showing where/how the tutorial specifies an oven temperature. If absent, null.
    - explains_duration_or_endpoint: The answer’s explanation text showing where/how the tutorial specifies a baking duration or a clear endpoint indicator (e.g., until hard and dry). If absent, null.
    - explains_flipping: The answer’s explanation text showing where/how the tutorial mentions flipping halfway. If absent, null.
    - explains_preservation: The answer’s explanation text showing where/how the tutorial provides at least one sealing/preservation method. If absent, null.
    - explains_durability: The answer’s explanation text showing why/how the tutorial’s preservation supports long-term durability. If absent, null.

    Important:
    - Only extract URLs explicitly present in the answer text. If no explicit URL is present, tutorial_url must be null.
    - For each explanation field, return the most relevant snippet from the answer text itself (not invented), or null if not present.
    """


# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


# --------------------------------------------------------------------------- #
# Verification Logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: TutorialExtraction) -> None:
    """
    Build the verification tree and run all checks according to the rubric.
    """
    root = evaluator.root

    # Top-level critical node (acts as the rubric root).
    main_node = evaluator.add_parallel(
        id="Tutorial_Meets_All_Requirements",
        desc="A publicly accessible online tutorial/guide for traditional salt dough Christmas ornaments is provided, it satisfies all stated constraints, and the response explains compliance.",
        parent=root,
        critical=True
    )

    # 1) Publicly Accessible URL Provided
    url = extracted.tutorial_url.strip() if extracted.tutorial_url else None
    if not url:
        # If no URL was provided in the answer, directly fail this critical leaf
        evaluator.add_custom_node(
            result=False,
            id="Publicly_Accessible_URL_Provided",
            desc="Provides a valid, publicly accessible URL to the tutorial/guide.",
            parent=main_node,
            critical=True
        )
    else:
        url_leaf = evaluator.add_leaf(
            id="Publicly_Accessible_URL_Provided",
            desc="Provides a valid, publicly accessible URL to the tutorial/guide.",
            parent=main_node,
            critical=True
        )
        await evaluator.verify(
            claim="This is a publicly accessible tutorial/guide webpage (loads without login/paywall).",
            node=url_leaf,
            sources=url,
            additional_instruction="Pass only if the page loads and is clearly a tutorial/guide (not just a generic category or index)."
        )

    # 2) Tutorial is for Salt Dough Christmas Ornaments
    tut_topic_leaf = evaluator.add_leaf(
        id="Tutorial_Is_For_Salt_Dough_Christmas_Ornaments",
        desc="The linked tutorial/guide is specifically about making salt dough ornaments intended as Christmas ornaments.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page is a tutorial/guide for making salt dough ornaments intended for Christmas (or holiday) tree decoration.",
        node=tut_topic_leaf,
        sources=url if url else None,
        additional_instruction="Look for phrases like 'salt dough ornaments' and references to Christmas/holiday ornaments or tree decorations."
    )

    # 3) Recipe Requirements (parallel, all critical)
    recipe_node = evaluator.add_parallel(
        id="Recipe_Requirements",
        desc="Recipe constraints (ingredients and measurements) are satisfied by the tutorial.",
        parent=main_node,
        critical=True
    )

    # 3.1) Includes all three ingredients
    recipe_ing_leaf = evaluator.add_leaf(
        id="Recipe_Includes_All_Prescribed_Ingredients",
        desc="Tutorial recipe includes all-purpose flour, table salt, and water as ingredients.",
        parent=recipe_node,
        critical=True
    )
    await evaluator.verify(
        claim="The recipe lists all-purpose flour, salt (table salt), and water as the three basic ingredients.",
        node=recipe_ing_leaf,
        sources=url if url else None,
        additional_instruction="Accept reasonable wording variants like 'plain flour' for all-purpose flour and 'salt' for table salt."
    )

    # 3.2) Specific measurements for all three
    recipe_meas_leaf = evaluator.add_leaf(
        id="Recipe_Provides_Measurements_For_All_Three",
        desc="Recipe provides specific measurement quantities for all three basic ingredients (all-purpose flour, table salt, and water).",
        parent=recipe_node,
        critical=True
    )
    await evaluator.verify(
        claim="The recipe provides specific quantities for flour, salt, and water (e.g., cups/tablespoons/ml/grams).",
        node=recipe_meas_leaf,
        sources=url if url else None,
        additional_instruction="Pass only if the page shows explicit numeric amounts for each of the three ingredients."
    )

    # 3.3) Ingredient roles stated/indicated
    recipe_roles_leaf = evaluator.add_leaf(
        id="Recipe_Specifies_Ingredient_Roles",
        desc="Tutorial states/indicates the constrained roles: all-purpose flour as the primary ingredient, table salt as a key ingredient, and water as the binding ingredient.",
        parent=recipe_node,
        critical=True
    )
    await evaluator.verify(
        claim="The tutorial indicates that flour is the main bulk/base, salt is a key dry component, and water is the binder for forming the dough.",
        node=recipe_roles_leaf,
        sources=url if url else None,
        additional_instruction="This can be explicit or clearly implied by how the recipe describes or uses each ingredient."
    )

    # 4) Preparation Requirements (parallel, all critical)
    prep_node = evaluator.add_parallel(
        id="Preparation_Requirements",
        desc="Preparation constraints are satisfied by the tutorial.",
        parent=main_node,
        critical=True
    )

    # 4.1) Kneading duration specified
    knead_leaf = evaluator.add_leaf(
        id="Preparation_Specifies_Kneading_Duration",
        desc="Preparation instructions specify the kneading duration (a time duration is stated).",
        parent=prep_node,
        critical=True
    )
    await evaluator.verify(
        claim="The preparation instructions specify a kneading duration (e.g., 'knead for X minutes').",
        node=knead_leaf,
        sources=url if url else None,
        additional_instruction="Require a time value; 'knead until smooth' alone is not sufficient."
    )

    # 4.2) Rolling thickness specified
    thickness_leaf = evaluator.add_leaf(
        id="Preparation_Specifies_Rolling_Thickness",
        desc="Preparation instructions specify the thickness to which the dough should be rolled.",
        parent=prep_node,
        critical=True
    )
    await evaluator.verify(
        claim="The instructions specify a dough thickness to roll to (e.g., 1/4 inch, 3-5 mm).",
        node=thickness_leaf,
        sources=url if url else None,
        additional_instruction="Look for explicit numeric thickness in inches or millimeters."
    )

    # 4.3) Holes for hanging explained
    holes_leaf = evaluator.add_leaf(
        id="Preparation_Explains_Holes_For_Hanging",
        desc="Preparation instructions explain how to create holes for hanging ornaments before baking.",
        parent=prep_node,
        critical=True
    )
    await evaluator.verify(
        claim="Before baking, the tutorial instructs making a hole for hanging (e.g., using a straw or skewer).",
        node=holes_leaf,
        sources=url if url else None,
        additional_instruction="Look for language like 'use a straw/toothpick/skewer to make a hole for ribbon/twine'."
    )

    # 5) Baking Requirements (parallel, all critical)
    baking_node = evaluator.add_parallel(
        id="Baking_Requirements",
        desc="Baking constraints are satisfied by the tutorial.",
        parent=main_node,
        critical=True
    )

    # 5.1) Oven temperature specified
    temp_leaf = evaluator.add_leaf(
        id="Baking_Specifies_Oven_Temperature",
        desc="Baking instructions provide a specific oven temperature.",
        parent=baking_node,
        critical=True
    )
    await evaluator.verify(
        claim="The baking instructions specify an oven temperature (in °F or °C).",
        node=temp_leaf,
        sources=url if url else None,
        additional_instruction="Common values include 200°F–250°F (90°C–120°C), but any specific temperature is acceptable."
    )

    # 5.2) Duration or endpoint specified
    duration_leaf = evaluator.add_leaf(
        id="Baking_Specifies_Duration_Or_Endpoint",
        desc="Baking instructions provide either a specific baking duration or a clear endpoint indicator for doneness/dryness.",
        parent=baking_node,
        critical=True
    )
    await evaluator.verify(
        claim="The baking instructions include either a specific time range/duration or a clear endpoint like 'until hard and dry'.",
        node=duration_leaf,
        sources=url if url else None,
        additional_instruction="Accept 'bake X–Y hours' or endpoint language that clearly indicates doneness."
    )

    # 5.3) Flipping halfway mentioned
    flipping_leaf = evaluator.add_leaf(
        id="Baking_Mentions_Flipping_Halfway",
        desc="Baking instructions mention flipping the ornaments halfway through baking.",
        parent=baking_node,
        critical=True
    )
    await evaluator.verify(
        claim="The baking instructions mention flipping/turning the ornaments halfway through baking.",
        node=flipping_leaf,
        sources=url if url else None,
        additional_instruction="Look for 'flip', 'turn over', 'rotate' at the midpoint of the bake."
    )

    # 6) Preservation Requirements (parallel, all critical)
    pres_node = evaluator.add_parallel(
        id="Preservation_Requirements",
        desc="Preservation/sealing constraints are satisfied by the tutorial.",
        parent=main_node,
        critical=True
    )

    # 6.1) Preservation method described
    pres_method_leaf = evaluator.add_leaf(
        id="Preservation_Method_Described",
        desc="Tutorial describes at least one method for sealing/preserving the finished ornaments.",
        parent=pres_node,
        critical=True
    )
    await evaluator.verify(
        claim="The tutorial recommends at least one sealing/preserving method (e.g., clear varnish, Mod Podge, acrylic sealer, polyurethane, PVA glue).",
        node=pres_method_leaf,
        sources=url if url else None,
        additional_instruction="Accept any credible sealing method intended for craft ornaments."
    )

    # 6.2) Preservation ensures long-term durability
    pres_durability_leaf = evaluator.add_leaf(
        id="Preservation_Ensures_Long_Term_Durability",
        desc="The preservation method is stated or clearly intended to improve long-term durability (i.e., lasting preservation over time).",
        parent=pres_node,
        critical=True
    )
    await evaluator.verify(
        claim="The sealing/preserving method is stated or clearly intended to improve long-term durability and protect the ornaments over time.",
        node=pres_durability_leaf,
        sources=url if url else None,
        additional_instruction="Look for phrases like 'seal to protect', 'helps them last', 'prevents moisture', or similar durability intent."
    )

    # 7) Response Explanation Requirements (parallel, all critical)
    explain_node = evaluator.add_parallel(
        id="Response_Explanation_Requirements",
        desc="The response explains, with evidence from the linked tutorial, how each required constraint is satisfied.",
        parent=main_node,
        critical=True
    )

    # Each explanation check is a binary existence check on the answer text
    evaluator.add_custom_node(
        result=_non_empty(extracted.explains_ingredients),
        id="Explains_Recipe_Ingredients_Compliance",
        desc="Response explains where/how the tutorial includes all-purpose flour, table salt, and water.",
        parent=explain_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(extracted.explains_measurements),
        id="Explains_Recipe_Measurements_Compliance",
        desc="Response explains where/how the tutorial provides specific measurements for flour, salt, and water.",
        parent=explain_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(extracted.explains_roles),
        id="Explains_Ingredient_Roles_Compliance",
        desc="Response explains where/how the tutorial indicates the required ingredient roles (primary/key/binding).",
        parent=explain_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(extracted.explains_kneading),
        id="Explains_Kneading_Duration_Compliance",
        desc="Response explains where/how the tutorial specifies kneading duration.",
        parent=explain_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(extracted.explains_thickness),
        id="Explains_Rolling_Thickness_Compliance",
        desc="Response explains where/how the tutorial specifies rolling thickness.",
        parent=explain_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(extracted.explains_holes),
        id="Explains_Holes_For_Hanging_Compliance",
        desc="Response explains where/how the tutorial instructs making holes for hanging before baking.",
        parent=explain_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(extracted.explains_temp),
        id="Explains_Oven_Temperature_Compliance",
        desc="Response explains where/how the tutorial specifies an oven temperature.",
        parent=explain_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(extracted.explains_duration_or_endpoint),
        id="Explains_Baking_Duration_Or_Endpoint_Compliance",
        desc="Response explains where/how the tutorial provides a baking duration or endpoint indicator.",
        parent=explain_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(extracted.explains_flipping),
        id="Explains_Flipping_Halfway_Compliance",
        desc="Response explains where/how the tutorial mentions flipping halfway through baking.",
        parent=explain_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(extracted.explains_preservation),
        id="Explains_Preservation_Method_Compliance",
        desc="Response explains where/how the tutorial provides at least one sealing/preservation method.",
        parent=explain_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(extracted.explains_durability),
        id="Explains_Long_Term_Durability_Compliance",
        desc="Response explains why/how the tutorial’s preservation method supports long-term durability (as stated or clearly implied by the tutorial).",
        parent=explain_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the salt dough Christmas ornaments tutorial task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract the tutorial URL and explanation snippets from the answer
    extracted: TutorialExtraction = await evaluator.extract(
        prompt=prompt_extract_tutorial_info(),
        template_class=TutorialExtraction,
        extraction_name="tutorial_extraction",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return structured summary
    return evaluator.get_summary()