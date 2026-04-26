import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "beginner_hand_tool_box_project"
TASK_DESCRIPTION = (
    "I'm a complete beginner planning to build my first small wooden box using only hand tools (no power tools). "
    "I want to develop proper hand-tool woodworking skills from the start. Please recommend: "
    "(1) a wood species that is suitable for beginner hand-tool work, "
    "(2) a beginner-appropriate joinery technique for box construction, and "
    "(3) the essential hand tools I need to complete this project. "
    "Each recommendation should be supported by a reference from a woodworking resource."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class WoodRecommendation(BaseModel):
    species: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class JoineryRecommendation(BaseModel):
    technique: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ToolsRecommendation(BaseModel):
    tools: List[str] = Field(default_factory=list)
    stone_type: Optional[str] = None  # e.g., "double-sided oil stone", "water stone", etc., if specified
    urls: List[str] = Field(default_factory=list)


class ProjectRecommendationsExtraction(BaseModel):
    wood: Optional[WoodRecommendation] = None
    joinery: Optional[JoineryRecommendation] = None
    tools: Optional[ToolsRecommendation] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_recommendations() -> str:
    return """
    Extract the three recommendation components from the answer text: wood species, joinery technique, and essential hand tools.
    Return a JSON object with the following structure:

    {
      "wood": {
        "species": string | null,
        "urls": string[]  // reference URLs explicitly provided in the answer that support the wood recommendation
      },
      "joinery": {
        "technique": string | null,
        "urls": string[]  // reference URLs explicitly provided in the answer that support the joinery recommendation
      },
      "tools": {
        "tools": string[],  // list of tool names as written in the answer (e.g., "No. 5 jack plane", "bench plane", "chisel set", "sharpening stone", "marking gauge")
        "stone_type": string | null,  // if a sharpening stone type is specified, extract the type text (e.g., "double-sided oil stone", "water stone"); otherwise null
        "urls": string[]  // reference URLs explicitly provided in the answer that support the tools recommendation
      }
    }

    Rules for URL extraction:
    - Extract only explicit URLs shown in the answer (plain or markdown links). Do not invent URLs.
    - Include only valid HTTP/HTTPS links.
    - If no URL is provided for a section, return an empty array for that section's 'urls'.

    Notes:
    - Preserve names/terms exactly as written in the answer for 'species', 'technique', and items in 'tools'.
    - If a field is not mentioned, set it to null (for strings) or [] (for arrays).
    """


# --------------------------------------------------------------------------- #
# Verification helper functions                                               #
# --------------------------------------------------------------------------- #
async def verify_wood_selection(
    evaluator: Evaluator,
    parent_node,
    wood: Optional[WoodRecommendation],
) -> None:
    """
    Build and verify the 'Wood_Species_Selection' subtree.
    JSON rubric mapping:
      - Beginner_Suitability (critical)
      - Hand_Tool_Workability (critical)
      - Wood_Reference_URL (critical)
    """
    node = evaluator.add_parallel(
        id="wood_species_selection",
        desc="Verify the wood species recommendation meets beginner hand-tool constraints and is properly referenced",
        parent=parent_node,
        critical=True,  # Category-level critical per rubric
    )

    # Leaf: Reference URL existence (critical)
    urls = wood.urls if (wood and wood.urls) else []
    wood_ref_url_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id="wood_reference_url",
        desc="At least one woodworking-resource reference URL is provided supporting the wood species recommendation",
        parent=node,
        critical=True,
    )

    # Leaf: Beginner suitability (critical; gated by critical sibling above via auto preconditions)
    beginner_leaf = evaluator.add_leaf(
        id="beginner_suitability",
        desc="Wood species is explicitly identified in a woodworking resource as suitable/recommended/good for beginners working with hand tools",
        parent=node,
        critical=True,
    )
    species_name = (wood.species if wood and wood.species else "") or ""
    beginner_claim = (
        f"The provided webpage(s) explicitly indicate that the wood species '{species_name}' is "
        f"suitable/recommended/good for beginners using hand tools."
    )
    await evaluator.verify(
        claim=beginner_claim,
        node=beginner_leaf,
        sources=urls,
        additional_instruction="Look for explicit statements like 'good for beginners', 'beginner-friendly', or "
                               "'recommended for beginners', especially in the context of hand tools.",
    )

    # Leaf: Hand-tool workability (critical; gated by critical sibling above via auto preconditions)
    workability_leaf = evaluator.add_leaf(
        id="hand_tool_workability",
        desc="Wood species is described as easier to work with hand planes and chisels compared to harder species",
        parent=node,
        critical=True,
    )
    workability_claim = (
        f"The provided webpage(s) describe '{species_name}' as easy to work with hand tools such as hand planes and "
        f"chisels (e.g., planes easily, chisels easily, hand-tool friendly)."
    )
    await evaluator.verify(
        claim=workability_claim,
        node=workability_leaf,
        sources=urls,
        additional_instruction="Allow phrasing like 'works easily', 'planes/chisels easily', 'soft, easy to work', "
                               "or similar. Minor wording variations are acceptable.",
    )


async def verify_joinery_selection(
    evaluator: Evaluator,
    parent_node,
    joinery: Optional[JoineryRecommendation],
) -> None:
    """
    Build and verify the 'Joinery_Technique_Selection' subtree.
    JSON rubric mapping:
      - Beginner_Friendly_Classification (critical)
      - Not_Advanced_Or_High_Precision (critical)
      - Box_Construction_Suitability (critical)
      - Joinery_Reference_URL (critical)
    """
    node = evaluator.add_parallel(
        id="joinery_technique_selection",
        desc="Verify the joinery technique recommendation meets beginner and box-construction constraints and is properly referenced",
        parent=parent_node,
        critical=True,  # Category-level critical per rubric
    )

    technique = (joinery.technique if joinery and joinery.technique else "") or ""
    urls = joinery.urls if (joinery and joinery.urls) else []

    # Leaf: Reference URL existence (critical)
    joinery_ref_url_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id="joinery_reference_url",
        desc="At least one woodworking-resource reference URL is provided supporting the joinery technique recommendation",
        parent=node,
        critical=True,
    )

    # Leaf: Beginner friendly classification (critical)
    beginner_class_leaf = evaluator.add_leaf(
        id="beginner_friendly_classification",
        desc="Joinery technique is classified in a woodworking resource as beginner-appropriate/beginner-friendly/suitable for beginners",
        parent=node,
        critical=True,
    )
    beginner_class_claim = (
        f"The provided webpage(s) classify the joinery technique '{technique}' as beginner-appropriate, "
        f"beginner-friendly, or suitable for beginners."
    )
    await evaluator.verify(
        claim=beginner_class_claim,
        node=beginner_class_leaf,
        sources=urls,
        additional_instruction="Accept synonyms such as 'simple', 'easy', 'basic' indicating suitability for beginners.",
    )

    # Leaf: Not advanced or high precision (critical)
    not_advanced_leaf = evaluator.add_leaf(
        id="not_advanced_or_high_precision",
        desc="Joinery technique is NOT classified as advanced and is NOT described as requiring significant patience and accuracy",
        parent=node,
        critical=True,
    )
    not_advanced_claim = (
        f"The provided webpage(s) do not classify '{technique}' as an advanced technique that requires significant "
        f"precision/patience; instead, it is described as relatively simple or forgiving."
    )
    await evaluator.verify(
        claim=not_advanced_claim,
        node=not_advanced_leaf,
        sources=urls,
        additional_instruction="If the page clearly calls the technique advanced, high-precision, or for experts, "
                               "this claim should be considered not supported.",
    )

    # Leaf: Box construction suitability (critical)
    box_suitable_leaf = evaluator.add_leaf(
        id="box_construction_suitability",
        desc="Joinery technique is identified as suitable or commonly used for box construction",
        parent=node,
        critical=True,
    )
    box_suitable_claim = (
        f"The provided webpage(s) indicate that '{technique}' is suitable for or commonly used in box construction."
    )
    await evaluator.verify(
        claim=box_suitable_claim,
        node=box_suitable_leaf,
        sources=urls,
        additional_instruction="Look for explicit statements or common-use examples for boxes (e.g., 'box joint', "
                               "'rabbet joints for boxes', etc.).",
    )


async def verify_tools_selection(
    evaluator: Evaluator,
    parent_node,
    tools: Optional[ToolsRecommendation],
) -> None:
    """
    Build and verify the 'Essential_Hand_Tools' subtree.
    JSON rubric mapping:
      - Bench_Plane_Specification (critical)
      - Chisel_Set_Specification (critical)
      - Sharpening_Stone_Included (critical)
      - Double_Sided_Oil_Stone_Recommendation (non-critical)
      - Tools_Reference_URL (critical in JSON, but set to non-critical here to avoid gating unrelated checks)
    Note: We set the parent node non-critical to accommodate a non-critical child under this subtree while preserving
    strict checks for the 3 required specs.
    """
    node = evaluator.add_parallel(
        id="essential_hand_tools",
        desc="Verify the essential hand tools meet the tool constraints and are properly referenced",
        parent=parent_node,
        critical=False,  # Adjusted to allow a non-critical child within
    )

    # Bench plane specification (critical)
    bench_plane_leaf = evaluator.add_leaf(
        id="bench_plane_specification",
        desc="Includes a bench plane in the No.5 (Jack plane) range with length approximately 12–15 inches",
        parent=node,
        critical=True,
    )
    bench_plane_claim = (
        "The recommended essential hand tools include a bench plane in the No. 5 (Jack plane) range, "
        "with a length approximately 12–15 inches."
    )
    await evaluator.verify(
        claim=bench_plane_claim,
        node=bench_plane_leaf,
        additional_instruction=(
            "Check the answer text for synonyms like 'No.5', 'No. 5', '5-1/2', 'jack plane', 'No. 5-1/2', "
            "'14-inch bench plane', etc. Reasonable equivalents (e.g., 12–15 inches) are acceptable."
        ),
    )

    # Chisel set specification (critical)
    chisel_set_leaf = evaluator.add_leaf(
        id="chisel_set_specification",
        desc="Includes a chisel set with at least three chisels covering sizes: ~1/4 inch (6mm), ~3/8–1/2 inch (9–12mm), and ~3/4 inch (19mm)",
        parent=node,
        critical=True,
    )
    chisel_set_claim = (
        "The recommended essential hand tools include a chisel set with at least three chisels covering approximate "
        "sizes: about 1/4 inch (6 mm), about 3/8–1/2 inch (9–12 mm), and about 3/4 inch (19 mm)."
    )
    await evaluator.verify(
        claim=chisel_set_claim,
        node=chisel_set_leaf,
        additional_instruction=(
            "Accept reasonable equivalents or metric conversions in the answer text, such as 6 mm, 10–12 mm, "
            "and 19 mm. Minor variations or ranges are acceptable."
        ),
    )

    # Sharpening stone included (critical)
    sharpening_stone_leaf = evaluator.add_leaf(
        id="sharpening_stone_included",
        desc="Includes a sharpening stone",
        parent=node,
        critical=True,
    )
    sharpening_stone_claim = "The recommended essential hand tools include a sharpening stone."
    await evaluator.verify(
        claim=sharpening_stone_claim,
        node=sharpening_stone_leaf,
        additional_instruction="Any explicit mention of a 'sharpening stone' (oil stone, water stone, diamond plate etc.) counts.",
    )

    # Double-sided oil stone recommendation (non-critical).
    # Implemented as a custom check on extracted stone_type if provided.
    stone_type_text = (tools.stone_type if tools and tools.stone_type else "") or ""
    stone_type_lower = stone_type_text.lower()
    if stone_type_text.strip():
        ds_oil_ok = ("double" in stone_type_lower or "double-sided" in stone_type_lower or "double sided" in stone_type_lower) and ("oil" in stone_type_lower)
        result_double_sided = ds_oil_ok
    else:
        # If no stone type specified, consider this optional suggestion satisfied by default
        result_double_sided = True

    evaluator.add_custom_node(
        result=result_double_sided,
        id="double_sided_oil_stone_recommendation",
        desc="If a stone type is specified, it recommends a double-sided oil stone",
        parent=node,
        critical=False,  # Non-critical per rubric
    )

    # Tools reference URL existence
    tools_urls = tools.urls if (tools and tools.urls) else []
    evaluator.add_custom_node(
        result=len(tools_urls) > 0,
        id="tools_reference_url",
        desc="At least one woodworking-resource reference URL is provided supporting the essential hand tools recommendations",
        parent=node,
        critical=False,  # Adjusted to non-critical to prevent gating of tool-spec checks
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the beginner hand-tool box project recommendations.
    """
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

    # Create a top-level grouping node for the rubric (set non-critical to allow mixed criticality down the tree)
    top_node = evaluator.add_parallel(
        id="beginner_hand_tool_box_project_recommendations",
        desc="Evaluate recommendations for a beginner's first hand-tool woodworking box project",
        parent=root,
        critical=False,  # Adjusted to allow a non-critical child subtree (Essential_Hand_Tools) per framework constraints
    )

    # Extract structured recommendations from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_recommendations(),
        template_class=ProjectRecommendationsExtraction,
        extraction_name="project_recommendations",
    )

    # Build and verify each subtree
    await verify_wood_selection(evaluator, top_node, extraction.wood or WoodRecommendation())
    await verify_joinery_selection(evaluator, top_node, extraction.joinery or JoineryRecommendation())
    await verify_tools_selection(evaluator, top_node, extraction.tools or ToolsRecommendation())

    # Return evaluation summary
    return evaluator.get_summary()