import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "houston_diy_christmas_2026"
TASK_DESCRIPTION = """
You are planning to build a DIY wooden Christmas decoration project in Houston, Texas, and need to acquire all materials by November 30, 2026. Your project requires the following materials: one piece of 1x6 dimensional lumber (at least 8 feet long), at least 5 wooden star cutouts, wood glue suitable for wood-to-wood bonding, and at least two colors of paint appropriate for wood crafts. Create a complete material acquisition plan that specifies: (1) which specific stores in Houston, TX you will shop at for each type of material (noting that lumber must come from home improvement stores, while craft supplies must come from craft stores), (2) the actual dimensions of the 1x6 lumber (not just the nominal size), (3) the specific type of wood glue needed for wood-to-wood bonding, (4) the type of paint appropriate for wood craft projects, and (5) a shopping timeline that ensures all materials are acquired by your November 30, 2026 deadline while accounting for Thanksgiving Day (November 27, 2026) store closures. For each material sourcing decision, provide a reference URL that supports your store selection, material specifications, or timeline planning.
"""

DEADLINE_ISO = "2026-11-30"
THANKSGIVING_2026_ISO = "2026-11-27"


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class TimelineInfo(BaseModel):
    acquisition_deadline: Optional[str] = None
    thanksgiving_accounting_text: Optional[str] = None
    holiday_hours_urls: List[str] = Field(default_factory=list)


class LumberInfo(BaseModel):
    store_name: Optional[str] = None
    store_city: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    nominal_size: Optional[str] = None
    actual_thickness: Optional[str] = None
    actual_width: Optional[str] = None
    length: Optional[str] = None


class CutoutsInfo(BaseModel):
    store_name: Optional[str] = None
    store_city: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    quantity_text: Optional[str] = None
    shape: Optional[str] = None


class GlueInfo(BaseModel):
    store_name: Optional[str] = None
    store_city: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    glue_type: Optional[str] = None
    brand: Optional[str] = None


class PaintInfo(BaseModel):
    store_name: Optional[str] = None
    store_city: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    paint_type: Optional[str] = None
    colors: List[str] = Field(default_factory=list)


class AcquisitionPlanExtraction(BaseModel):
    timeline: Optional[TimelineInfo] = None
    lumber: Optional[LumberInfo] = None
    cutouts: Optional[CutoutsInfo] = None
    glue: Optional[GlueInfo] = None
    paint: Optional[PaintInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_acquisition_plan() -> str:
    return """
Extract the material acquisition plan details from the provided answer. Return a single JSON object with the following structure:

{
  "timeline": {
    "acquisition_deadline": "verbatim deadline date or phrasing for when all materials will be acquired (e.g., 'by Nov 30, 2026')",
    "thanksgiving_accounting_text": "verbatim sentence/phrase showing the plan accounts for Thanksgiving Day closures on Nov 27, 2026 (if present, otherwise null)",
    "holiday_hours_urls": ["list of any URLs the answer cites for store holiday hours or schedule"]
  },
  "lumber": {
    "store_name": "the specific store named for lumber (e.g., 'The Home Depot', 'Lowe's')",
    "store_city": "the city for the store if stated (e.g., 'Houston')",
    "urls": ["all URLs provided for lumber product page(s) or store location(s)"],
    "nominal_size": "the nominal size mentioned (e.g., '1x6')",
    "actual_thickness": "actual thickness mentioned (e.g., '3/4 in', '0.75 in')",
    "actual_width": "actual width mentioned (e.g., '5-1/2 in', '5.5 in')",
    "length": "the length mentioned (e.g., '8 ft', '96 in')"
  },
  "cutouts": {
    "store_name": "the named craft store for wooden star cutouts (e.g., 'Michaels', 'Hobby Lobby')",
    "store_city": "the city for the store if stated",
    "urls": ["all URLs provided for cutouts product page(s) or store location(s)"],
    "quantity_text": "verbatim quantity reference (e.g., 'pack of 6', 'at least 5')",
    "shape": "shape mentioned (e.g., 'star')"
  },
  "glue": {
    "store_name": "the named craft store for wood glue",
    "store_city": "the city for the store if stated",
    "urls": ["all URLs provided for glue product page(s) or store location(s)"],
    "glue_type": "verbatim glue type (e.g., 'PVA wood glue', 'Titebond Original')",
    "brand": "brand if provided (e.g., 'Titebond')"
  },
  "paint": {
    "store_name": "the named craft store for paint",
    "store_city": "the city for the store if stated",
    "urls": ["all URLs provided for paint product page(s) or store location(s)"],
    "paint_type": "verbatim paint type (e.g., 'acrylic craft paint')",
    "colors": ["list of distinct color names mentioned for paint (exclude 'assorted' if not specific)"]
  }
}

Rules:
- Extract exactly what the answer states. Do not infer or invent any data.
- Only include URLs explicitly present in the answer text.
- Keep all numbers and measurements as strings exactly as stated in the answer (e.g., '5-1/2 in', '3/4 in', '8 ft').
- If something is not present in the answer, set the field to null or an empty list as appropriate.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # De-duplicate while preserving order
    seen = set()
    out: List[str] = []
    for u in urls:
        if isinstance(u, str) and u.strip() and u not in seen:
            out.append(u)
            seen.add(u)
    return out


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_timeline_nodes(evaluator: Evaluator, parent, data: AcquisitionPlanExtraction):
    node = evaluator.add_parallel(
        id="Timeline_Compliance",
        desc="Shopping timeline accounts for all relevant constraints including deadline and holiday closures",
        parent=parent,
        critical=False
    )

    # Deadline met (critical)
    deadline_leaf = evaluator.add_leaf(
        id="Deadline_Met",
        desc="All materials are planned to be acquired by November 30, 2026",
        parent=node,
        critical=True
    )
    deadline_text = data.timeline.acquisition_deadline if data and data.timeline else None
    await evaluator.verify(
        claim="According to the plan, all materials will be acquired on or before November 30, 2026 (inclusive). Earlier completion also satisfies the requirement.",
        node=deadline_leaf,
        additional_instruction=f"If the plan clearly states 'by Nov 30, 2026' or an equivalent phrasing, consider it correct. Extracted deadline snippet (if any): {deadline_text!r}"
    )

    # Thanksgiving acknowledged (critical)
    tg_leaf = evaluator.add_leaf(
        id="Thanksgiving_Closure_Acknowledged",
        desc="Shopping plan accounts for Thanksgiving Day (November 27, 2026) store closures",
        parent=node,
        critical=True
    )
    tg_snippet = data.timeline.thanksgiving_accounting_text if data and data.timeline else None
    await evaluator.verify(
        claim="The plan explicitly acknowledges Thanksgiving Day (Nov 27, 2026) store closures and avoids scheduling in-store shopping that day or otherwise accounts for closures.",
        node=tg_leaf,
        additional_instruction=f"Look for explicit mention of Thanksgiving closures or avoidance on Nov 27, 2026. Snippet (if any): {tg_snippet!r}"
    )

    # URL reference provided for holiday hours (non-critical) - existence check
    holiday_urls = _safe_urls(data.timeline.holiday_hours_urls if data and data.timeline else [])
    evaluator.add_custom_node(
        result=len(holiday_urls) > 0,
        id="URL_Reference_Timeline",
        desc="Reference URL provided for store holiday hours",
        parent=node,
        critical=False
    )


async def build_lumber_nodes(evaluator: Evaluator, parent, data: AcquisitionPlanExtraction):
    node = evaluator.add_parallel(
        id="Lumber_Acquisition",
        desc="1x6 lumber is sourced from the correct store type with proper dimensional specifications",
        parent=parent,
        critical=False
    )
    lumber = data.lumber if data else None
    lumber_urls = _safe_urls(lumber.urls if lumber else [])

    # Store type (critical) - simple verification from the plan text
    store_leaf = evaluator.add_leaf(
        id="Lumber_Store_Type",
        desc="Lumber is sourced from a home improvement store (Home Depot or Lowe's) in Houston, TX",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="According to the plan, the lumber will be purchased from either The Home Depot or Lowe's (a home improvement store) in Houston, Texas.",
        node=store_leaf,
        additional_instruction="Accept 'The Home Depot' or 'Home Depot' and 'Lowe's' as valid. The plan text should indicate Houston, TX for the lumber store."
    )

    # Actual dimensions (critical) - verify with URLs if provided
    dims_leaf = evaluator.add_leaf(
        id="Lumber_Actual_Dimensions",
        desc="The actual dimensions of 1x6 lumber are correctly specified as 3/4 inch thick by 5-1/2 inches wide",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The source page explicitly shows that a nominal 1x6 board has actual dimensions approximately 3/4 inch (0.75 in) thick by 5-1/2 inches (5.5 in) wide.",
        node=dims_leaf,
        sources=lumber_urls,
        additional_instruction="Look for phrases like 'Actual: 0.75 in. x 5.5 in.' on a product/spec page from a home improvement store."
    )

    # Length requirement (critical) - simple verification from the plan text
    length_leaf = evaluator.add_leaf(
        id="Lumber_Length",
        desc="At least one piece of 1x6 lumber with minimum 8 ft length is specified",
        parent=node,
        critical=True
    )
    length_text = lumber.length if lumber else None
    await evaluator.verify(
        claim="The plan specifies at least one 1x6 board with a minimum length of 8 feet (>= 96 inches).",
        node=length_leaf,
        additional_instruction=f"Treat '8 ft', '8-foot', '96 in', or anything >= 8 ft as satisfying. Extracted length snippet (if any): {length_text!r}"
    )

    # URL reference provided (non-critical) - existence check
    evaluator.add_custom_node(
        result=len(lumber_urls) > 0,
        id="URL_Reference_Lumber",
        desc="Reference URL provided for lumber dimensional specifications or store location",
        parent=node,
        critical=False
    )


async def build_cutouts_nodes(evaluator: Evaluator, parent, data: AcquisitionPlanExtraction):
    node = evaluator.add_parallel(
        id="Wooden_Cutouts_Acquisition",
        desc="Wooden star cutouts are sourced from the correct store type with proper quantity",
        parent=parent,
        critical=False
    )
    cutouts = data.cutouts if data else None
    cutouts_urls = _safe_urls(cutouts.urls if cutouts else [])

    # Store type (critical) - simple verification from plan text
    store_leaf = evaluator.add_leaf(
        id="Cutouts_Store_Type",
        desc="Wooden cutouts are sourced from a craft store (Hobby Lobby or Michaels) in Houston, TX",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="According to the plan, the wooden star cutouts will be sourced from a craft store: either Michaels or Hobby Lobby, in Houston, Texas.",
        node=store_leaf,
        additional_instruction="Accept 'Michaels' or 'Hobby Lobby'. The plan text should indicate Houston, TX for the cutouts store."
    )

    # Quantity (critical) - simple verification from plan text
    qty_leaf = evaluator.add_leaf(
        id="Cutouts_Quantity",
        desc="At least 5 wooden star cutouts are specified",
        parent=node,
        critical=True
    )
    qty_text = cutouts.quantity_text if cutouts else None
    await evaluator.verify(
        claim="The plan specifies a quantity of wooden star cutouts that is at least 5.",
        node=qty_leaf,
        additional_instruction=f"Look for 'at least 5', 'pack of 5 or more', or any count >= 5. Extracted quantity snippet (if any): {qty_text!r}"
    )

    # URL reference provided (non-critical) - existence check
    evaluator.add_custom_node(
        result=len(cutouts_urls) > 0,
        id="URL_Reference_Cutouts",
        desc="Reference URL provided for wooden cutout availability or store location",
        parent=node,
        critical=False
    )


async def build_glue_nodes(evaluator: Evaluator, parent, data: AcquisitionPlanExtraction):
    node = evaluator.add_parallel(
        id="Wood_Glue_Acquisition",
        desc="Wood glue is sourced from the correct store type and is the appropriate type for wood-to-wood bonding",
        parent=parent,
        critical=False
    )
    glue = data.glue if data else None
    glue_urls = _safe_urls(glue.urls if glue else [])

    # Store type (critical) - simple verification from plan text
    store_leaf = evaluator.add_leaf(
        id="Glue_Store_Type",
        desc="Wood glue is sourced from a craft store (Hobby Lobby or Michaels) in Houston, TX",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="According to the plan, the wood glue will be sourced from a craft store: either Michaels or Hobby Lobby, in Houston, Texas.",
        node=store_leaf,
        additional_instruction="Accept 'Michaels' or 'Hobby Lobby'. The plan text should indicate Houston, TX for the glue store."
    )

    # Glue type (critical) - verify by URLs when available
    type_leaf = evaluator.add_leaf(
        id="Glue_Type",
        desc="Wood glue is specified as PVA-based wood glue (such as Titebond) appropriate for wood-to-wood bonding",
        parent=node,
        critical=True
    )
    glue_type_text = glue.glue_type if glue else None
    await evaluator.verify(
        claim="The source page shows that the selected wood glue is a PVA-based wood glue (e.g., Titebond) suitable for wood-to-wood bonding.",
        node=type_leaf,
        sources=glue_urls,
        additional_instruction=f"Look for terms like 'PVA', 'polyvinyl acetate', or Titebond Original/II/III. Extracted glue type snippet (if any): {glue_type_text!r}"
    )

    # URL reference provided (non-critical) - existence check
    evaluator.add_custom_node(
        result=len(glue_urls) > 0,
        id="URL_Reference_Glue",
        desc="Reference URL provided for wood glue type recommendation or store location",
        parent=node,
        critical=False
    )


async def build_paint_nodes(evaluator: Evaluator, parent, data: AcquisitionPlanExtraction):
    node = evaluator.add_parallel(
        id="Paint_Acquisition",
        desc="Acrylic paint is sourced from the correct store type with at least two colors",
        parent=parent,
        critical=False
    )
    paint = data.paint if data else None
    paint_urls = _safe_urls(paint.urls if paint else [])

    # Store type (critical) - simple verification from plan text
    store_leaf = evaluator.add_leaf(
        id="Paint_Store_Type",
        desc="Acrylic paint is sourced from a craft store (Hobby Lobby or Michaels) in Houston, TX",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="According to the plan, the acrylic paint will be sourced from a craft store: either Michaels or Hobby Lobby, in Houston, Texas.",
        node=store_leaf,
        additional_instruction="Accept 'Michaels' or 'Hobby Lobby'. The plan text should indicate Houston, TX for the paint store."
    )

    # Paint type (critical) - verify by URLs when available
    type_leaf = evaluator.add_leaf(
        id="Paint_Type",
        desc="Paint is specified as acrylic paint, which is appropriate for wood craft projects",
        parent=node,
        critical=True
    )
    paint_type_text = paint.paint_type if paint else None
    await evaluator.verify(
        claim="The source page shows that the selected paint is acrylic craft paint suitable for wood projects.",
        node=type_leaf,
        sources=paint_urls,
        additional_instruction=f"Look for 'acrylic paint', 'craft acrylic', or 'multi-surface acrylic' that mentions suitability for wood. Extracted paint type snippet (if any): {paint_type_text!r}"
    )

    # Paint colors (critical) - check that at least two colors are specified (existence/quantity)
    colors = paint.colors if paint else []
    evaluator.add_custom_node(
        result=bool(colors) and len([c for c in colors if isinstance(c, str) and c.strip()]) >= 2,
        id="Paint_Colors",
        desc="At least two colors of acrylic paint are specified",
        parent=node,
        critical=True
    )

    # URL reference provided (non-critical) - existence check
    evaluator.add_custom_node(
        result=len(paint_urls) > 0,
        id="URL_Reference_Paint",
        desc="Reference URL provided for paint type recommendation or store location",
        parent=node,
        critical=False
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation across major requirement groups
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

    # IMPORTANT: Root node cannot be critical if children are mixed criticality (framework constraint)
    root.critical = False
    root.desc = "All materials for the DIY wooden Christmas decoration project are properly sourced and planned for acquisition"

    # Extract structured plan
    extracted_plan = await evaluator.extract(
        prompt=prompt_extract_acquisition_plan(),
        template_class=AcquisitionPlanExtraction,
        extraction_name="acquisition_plan"
    )

    # Record ground-truth constraints for reference (not for hard matching)
    evaluator.add_ground_truth({
        "deadline_iso": DEADLINE_ISO,
        "thanksgiving_iso": THANKSGIVING_2026_ISO,
        "required_materials": [
            "1 piece of 1x6 dimensional lumber (>= 8 ft long)",
            ">= 5 wooden star cutouts",
            "wood glue for wood-to-wood bonding (PVA-based such as Titebond)",
            ">= 2 colors of acrylic paint suitable for wood crafts"
        ],
        "store_type_rules": {
            "lumber": "home improvement store (Home Depot or Lowe's)",
            "craft_supplies": "craft store (Michaels or Hobby Lobby)"
        }
    }, gt_type="requirements")

    # Build verification subtrees
    await build_timeline_nodes(evaluator, root, extracted_plan)
    await build_lumber_nodes(evaluator, root, extracted_plan)
    await build_cutouts_nodes(evaluator, root, extracted_plan)
    await build_glue_nodes(evaluator, root, extracted_plan)
    await build_paint_nodes(evaluator, root, extracted_plan)

    # Return summary
    return evaluator.get_summary()