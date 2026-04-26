import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dollar_tree_fall_wreath_plan"
TASK_DESCRIPTION = (
    "Sarah wants to create a fall-themed deco mesh wreath for her front door using only Dollar Tree supplies. "
    "Her door space is limited, so the finished wreath cannot exceed 26 inches in total diameter. She has a budget of $10 "
    "and prefers a wreath-making technique that minimizes mesh fraying to ensure a neat appearance. Based on these constraints, "
    "provide a complete material plan that includes: (1) The specific wire wreath frame size (in inches diameter) she should purchase, "
    "(2) The deco mesh specifications (width in inches and roll length in yards), (3) The number of mesh rolls needed, "
    "(4) The specific deco mesh technique she should use, (5) Any additional essential supplies needed (beyond frame and mesh), "
    "and (6) The total estimated cost. All materials must be available at Dollar Tree, the finished wreath must not exceed 26 inches in diameter, "
    "the total cost must stay under $10, and the technique must be one that minimizes fraying."
)

# Standards for checks
STANDARD_FRAME_SIZES_IN = {"8", "12", "14", "14.25", "18", "20"}
STANDARD_MESH_WIDTHS_IN = {"6", "10", "21"}
STANDARD_MESH_ROLL_LENGTHS_YD = {"5", "10"}
RECOGNIZED_TECHNIQUES = {"poof", "ruffle", "curl", "woodland ruffle", "cruffle"}  # include common synonym "cruffle"
FALL_KEYWORDS = {"fall", "autumn", "thanksgiving", "harvest"}
FALL_MOTIFS = {"leaves", "leaf", "pumpkin", "pumpkins", "maple", "acorn", "corn", "gourd"}
FALL_COLORS = {"orange", "burnt orange", "rust", "gold", "mustard", "burgundy", "brown", "copper"}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class WreathPlanExtraction(BaseModel):
    # Required outputs
    frame_size_in: Optional[str] = None
    mesh_width_in: Optional[str] = None
    mesh_roll_length_yd: Optional[str] = None
    mesh_roll_count: Optional[str] = None
    technique: Optional[str] = None
    additional_supplies: List[str] = Field(default_factory=list)
    total_estimated_cost: Optional[str] = None

    # Constraint-related details
    finished_diameter: Optional[str] = None  # any expression provided by the answer

    # Fall theme indicators explicitly mentioned in the answer
    fall_keywords_found: List[str] = Field(default_factory=list)
    fall_motifs_found: List[str] = Field(default_factory=list)
    fall_color_words_found: List[str] = Field(default_factory=list)

    # Dollar Tree sourcing mentions and URLs explicitly in the answer
    store_mentions: List[str] = Field(default_factory=list)      # e.g., "Dollar Tree"
    frame_url: Optional[str] = None                              # URL if present
    mesh_urls: List[str] = Field(default_factory=list)
    supplies_urls: List[str] = Field(default_factory=list)
    store_urls: List[str] = Field(default_factory=list)          # General store/product URLs mentioned


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_wreath_plan() -> str:
    return """
    Extract the material plan details for the fall-themed deco mesh wreath exactly as stated in the answer.

    REQUIRED FIELDS:
    1) frame_size_in: The wire wreath frame diameter in inches, as a number string if possible (e.g., "14" or "14.25"). If expressed with units or text, extract the text verbatim.
    2) mesh_width_in: The deco mesh width in inches (e.g., "6", "10", "21", or a textual form).
    3) mesh_roll_length_yd: The mesh roll length in yards (e.g., "5", "10", or a textual form).
    4) mesh_roll_count: The number of mesh rolls needed (extract as provided; can be a number or textual range).
    5) technique: The specific deco mesh technique named (e.g., "Poof", "Ruffle", "Curl", "Woodland Ruffle", "Cruffle", etc.).
    6) additional_supplies: List all additional essential supplies beyond frame and mesh mentioned in the answer (e.g., pipe cleaners/chenille stems, zip ties/cable ties, wire cutter, scissors, floral wire).
    7) total_estimated_cost: The single total cost estimate for all materials as stated (e.g., "$8", "about $9.50").
    8) finished_diameter: Any stated expected or estimated finished wreath diameter (e.g., "24 inches", "about 25\"", etc.).

    FALL THEME INDICATORS:
    - fall_keywords_found: Collect any explicit fall/autumn/thanksgiving/harvest words mentioned in the answer.
    - fall_motifs_found: Collect motifs indicating fall (e.g., leaves, pumpkins, maple, acorn, gourd).
    - fall_color_words_found: Collect explicitly mentioned fall color palette words (e.g., orange, burnt orange, rust, gold, mustard, burgundy, brown, copper).

    DOLLAR TREE SOURCING:
    - store_mentions: Collect any store names mentioned for sourcing (e.g., "Dollar Tree").
    - frame_url: If a URL is provided for the frame, extract it.
    - mesh_urls: Extract any URLs for mesh products mentioned.
    - supplies_urls: Extract any URLs for additional supplies mentioned.
    - store_urls: Extract any general store/product URLs mentioned (e.g., Dollar Tree product pages or category pages).
    
    SPECIAL RULES FOR URL EXTRACTION:
    - Only extract URLs explicitly present in the answer text (including markdown links).
    - If no URL is provided, return null for single fields or an empty array for list fields.

    If a field is not present, set it to null or an empty list as appropriate.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_str(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def gather_all_sources(ex: WreathPlanExtraction) -> List[str]:
    urls: List[str] = []
    if _nonempty_str(ex.frame_url):
        urls.append(ex.frame_url)  # type: ignore
    urls.extend([u for u in ex.mesh_urls if _nonempty_str(u)])
    urls.extend([u for u in ex.supplies_urls if _nonempty_str(u)])
    urls.extend([u for u in ex.store_urls if _nonempty_str(u)])
    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_fall_theme_node(evaluator: Evaluator, parent_node, ex: WreathPlanExtraction) -> None:
    # Leaf: Fall_Theme_Element_Included (critical)
    node = evaluator.add_leaf(
        id="Fall_Theme_Element_Included",
        desc="Plan explicitly includes at least one fall-themed element (keyword, motif, or fall color palette)",
        parent=parent_node,
        critical=True,
    )

    claim = (
        "The answer explicitly includes at least one fall-themed element: "
        "either uses words like 'fall', 'autumn', 'thanksgiving', or 'harvest', "
        "or names fall motifs such as leaves or pumpkins, or mentions fall color palette colors "
        "(e.g., orange, burnt orange, rust, gold, mustard, burgundy, brown, copper)."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction=(
            "Check the answer text for any of these indicators. "
            "Minor variations or synonyms are acceptable. "
            "If at least one indicator is present, judge True; otherwise False."
        ),
    )


async def build_outputs_included_node(evaluator: Evaluator, parent_node, ex: WreathPlanExtraction) -> None:
    outputs_node = evaluator.add_parallel(
        id="All_Requested_Outputs_Included",
        desc="Response includes all required parts (frame size, mesh width & roll length, mesh roll count, technique, additional supplies, total cost)",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty_str(ex.frame_size_in),
        id="Frame_Size_Provided",
        desc="Provides the specific wire wreath frame size (diameter in inches) to purchase",
        parent=outputs_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty_str(ex.mesh_width_in) and _nonempty_str(ex.mesh_roll_length_yd),
        id="Mesh_Specs_Provided",
        desc="Provides deco mesh width (in inches) AND roll length (in yards)",
        parent=outputs_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty_str(ex.mesh_roll_count),
        id="Mesh_Roll_Count_Provided",
        desc="Specifies the number of mesh rolls needed",
        parent=outputs_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty_str(ex.technique),
        id="Technique_Provided",
        desc="Names a specific deco mesh technique to use",
        parent=outputs_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(ex.additional_supplies),
        id="Additional_Supplies_Provided",
        desc="Lists additional essential supplies needed beyond the frame and mesh",
        parent=outputs_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_nonempty_str(ex.total_estimated_cost),
        id="Total_Cost_Provided",
        desc="Provides a single total estimated cost for all listed materials",
        parent=outputs_node,
        critical=True,
    )


async def build_constraints_node(evaluator: Evaluator, parent_node, ex: WreathPlanExtraction) -> None:
    cons_node = evaluator.add_parallel(
        id="Constraints_Compliance",
        desc="Plan satisfies all explicit constraints (size <= 26\", budget < $10, Dollar Tree-only sourcing, standards, technique minimizing fraying, essential supplies covered)",
        parent=parent_node,
        critical=True,
    )

    # Finished diameter under 26"
    fin_diam_node = evaluator.add_leaf(
        id="Finished_Diameter_Under_26",
        desc="States an expected/estimated finished wreath diameter and it does not exceed 26 inches",
        parent=cons_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states an expected or estimated finished wreath diameter, and it does not exceed 26 inches.",
        node=fin_diam_node,
        additional_instruction="Explicit mention is required in the answer; if not stated, judge False.",
    )

    # Under $10 total
    under_10_node = evaluator.add_leaf(
        id="Under_10_Dollars",
        desc="Total estimated cost is under $10",
        parent=cons_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The total estimated cost for all listed materials is strictly under $10.",
        node=under_10_node,
        additional_instruction="Look for a single total cost value in the answer. Allow minor rounding; if it is $10 or more, judge False.",
    )

    # Dollar Tree-only sourcing
    dt_only_node = evaluator.add_leaf(
        id="Dollar_Tree_Only_Sourcing",
        desc="All listed materials are explicitly represented as available from Dollar Tree",
        parent=cons_node,
        critical=True,
    )
    dt_sources = gather_all_sources(ex)
    await evaluator.verify(
        claim="All listed materials (frame, mesh, and additional supplies) are indicated in the answer as available from Dollar Tree.",
        node=dt_only_node,
        sources=dt_sources if dt_sources else None,
        additional_instruction=(
            "Judge True only if the answer clearly indicates Dollar Tree for the sourcing of each material "
            "(via explicit 'Dollar Tree' mentions and/or Dollar Tree product links). If any item appears non-Dollar Tree or unspecified, judge False."
        ),
    )

    # Frame size within provided standards
    frame_std_node = evaluator.add_leaf(
        id="Frame_Size_Within_Provided_Standards",
        desc='Chosen frame diameter is one of the provided standard wire wreath frame sizes (8", 12", 14", 14.25", 18", 20")',
        parent=cons_node,
        critical=True,
    )
    claim_frame_std = (
        f'The chosen frame diameter stated in the answer is one of: 8", 12", 14", 14.25", 18", or 20". '
        f'The answer states: {ex.frame_size_in or "N/A"}.'
    )
    await evaluator.verify(
        claim=claim_frame_std,
        node=frame_std_node,
        additional_instruction=(
            "Accept minor formatting variations (e.g., 14 in, 14\", 14-inch). If the stated diameter is not in the set, judge False."
        ),
    )

    # Mesh specs within standards
    mesh_std_node = evaluator.add_leaf(
        id="Mesh_Specs_Within_Provided_Standards",
        desc='Mesh width is one of {6", 10", 21"} AND mesh roll length is one of {5 yd, 10 yd}',
        parent=cons_node,
        critical=True,
    )
    claim_mesh_std = (
        f'The mesh width and roll length stated in the answer are within standards: width ∈ {{6", 10", 21"}} and roll length ∈ {{5 yd, 10 yd}}. '
        f'The answer states width: {ex.mesh_width_in or "N/A"}, roll length: {ex.mesh_roll_length_yd or "N/A"}.'
    )
    await evaluator.verify(
        claim=claim_mesh_std,
        node=mesh_std_node,
        additional_instruction=(
            "Allow reasonable unit formats (inches/in, yards/yd). If either width or roll length is outside the allowed sets, judge False."
        ),
    )

    # Technique recognized
    tech_rec_node = evaluator.add_leaf(
        id="Technique_Is_Recognized",
        desc="Technique is one of the recognized categories (Poof, Ruffle, Curl, Woodland Ruffle, or listed combinations)",
        parent=cons_node,
        critical=True,
    )
    claim_tech_rec = (
        f"The technique named in the answer is a recognized category among: Poof, Ruffle, Curl, Woodland Ruffle, or their combinations/synonyms (e.g., 'cruffle'). "
        f"Stated technique: {ex.technique or 'N/A'}."
    )
    await evaluator.verify(
        claim=claim_tech_rec,
        node=tech_rec_node,
        additional_instruction=(
            "Match the technique name from the answer against the recognized set. Accept common synonyms (e.g., 'cruffle' for curl+ruffle)."
        ),
    )

    # Technique minimizes fraying
    tech_min_fray_node = evaluator.add_leaf(
        id="Technique_Minimizes_Fraying",
        desc="Technique is explicitly described as minimizing fraying (or is the explicitly minimum-fray technique category per constraints)",
        parent=cons_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly indicates that the chosen technique minimizes mesh fraying.",
        node=tech_min_fray_node,
        additional_instruction=(
            "Base judgment on explicit statements in the answer. Look for phrases like 'minimizes fraying', 'reduces fraying', "
            "or explicit recommendation that the chosen technique is selected to keep fraying minimal. If not explicitly stated, judge False."
        ),
    )

    # Essential supplies covered: (a) pipe cleaners OR zip ties AND (b) a wire cutter
    ess_supplies_node = evaluator.add_leaf(
        id="Essential_Supplies_Covered",
        desc="Additional supplies include (a) pipe cleaners OR zip ties, AND (b) a wire cutter",
        parent=cons_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The additional supplies listed in the answer include either pipe cleaners (aka chenille stems) or zip ties (aka cable ties), "
            "and also include a wire cutter (aka wire cutters/diagonal cutters)."
        ),
        node=ess_supplies_node,
        additional_instruction=(
            "Check the 'additional supplies' portion of the answer. Accept reasonable synonyms: 'chenille stems' for pipe cleaners; "
            "'cable ties' for zip ties; 'wire cutters', 'diagonal cutters' for wire cutter."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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

    # Extract structured plan from the answer
    ex: WreathPlanExtraction = await evaluator.extract(
        prompt=prompt_extract_wreath_plan(),
        template_class=WreathPlanExtraction,
        extraction_name="wreath_material_plan",
    )

    # Build a critical plan node under root to enforce overall gating
    plan_node = evaluator.add_parallel(
        id="Wreath_Material_Plan",
        desc="Complete material plan for a fall-themed deco mesh wreath meeting size, budget, source, and minimum-fray technique constraints",
        parent=root,
        critical=True,
    )

    # Fall theme check
    await build_fall_theme_node(evaluator, plan_node, ex)

    # All requested outputs included
    await build_outputs_included_node(evaluator, plan_node, ex)

    # Constraints compliance
    await build_constraints_node(evaluator, plan_node, ex)

    # Return the evaluation summary
    return evaluator.get_summary()