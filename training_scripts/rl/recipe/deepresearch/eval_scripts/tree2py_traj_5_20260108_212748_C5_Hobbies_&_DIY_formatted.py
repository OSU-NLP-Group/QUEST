import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "woodshop_beginner_plan"
TASK_DESCRIPTION = (
    "You are setting up a home woodworking workshop in your garage and planning to build your first beginner furniture "
    "project (a simple side table or bookshelf). Provide a comprehensive setup plan that includes: (1) the essential "
    "safety equipment you need, specifying items for eye, hearing, and respiratory protection; (2) specifications for "
    "a properly sized workbench, including height, depth, and length measurements that follow ergonomic guidelines; "
    "(3) a list of core tools required for the project, including measuring, cutting, drilling, and clamping tools; "
    "(4) an appropriate wood type selection for a beginner; and (5) your construction approach, specifying a "
    "beginner-friendly joinery method and the finishing process you will use. Your plan should follow standard "
    "woodworking practices and safety guidelines established in the woodworking community."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SafetyInfo(BaseModel):
    eye_protection: List[str] = Field(default_factory=list)
    hearing_protection: List[str] = Field(default_factory=list)
    respiratory_protection: List[str] = Field(default_factory=list)


class WorkbenchSpecs(BaseModel):
    height_text: Optional[str] = None
    depth_text: Optional[str] = None
    length_text: Optional[str] = None


class CoreTools(BaseModel):
    measuring_tools: List[str] = Field(default_factory=list)
    cutting_tools: List[str] = Field(default_factory=list)
    drill_driver_tools: List[str] = Field(default_factory=list)
    clamps: List[str] = Field(default_factory=list)


class MaterialInfo(BaseModel):
    primary_wood: Optional[str] = None
    additional_woods: List[str] = Field(default_factory=list)


class FinishInfo(BaseModel):
    steps_text: Optional[str] = None
    final_sanding_grit: Optional[int] = None
    finish_types: List[str] = Field(default_factory=list)


class ConstructionInfo(BaseModel):
    joinery_methods: List[str] = Field(default_factory=list)
    finishing: FinishInfo = Field(default_factory=FinishInfo)


class PlanExtraction(BaseModel):
    safety: SafetyInfo = Field(default_factory=SafetyInfo)
    workbench: WorkbenchSpecs = Field(default_factory=WorkbenchSpecs)
    tools: CoreTools = Field(default_factory=CoreTools)
    materials: MaterialInfo = Field(default_factory=MaterialInfo)
    construction: ConstructionInfo = Field(default_factory=ConstructionInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
    Extract a structured woodworking setup plan from the answer. Organize the extracted content into the following fields.
    Return strictly in JSON format matching the schema. Use lists for multiple items. If something is not specified, return null or an empty list appropriately.

    1) safety:
       - eye_protection: list of items for eye safety (e.g., safety glasses, goggles, face shield).
       - hearing_protection: list of items for hearing safety (e.g., earplugs, earmuffs).
       - respiratory_protection: list of items for dust/respiratory safety (e.g., dust mask, N95/respirator).
    
    2) workbench:
       - height_text: the stated height (include units as written, e.g., "36 inches" or "90 cm").
       - depth_text: the stated depth.
       - length_text: the stated length.
       Note: Keep values as text; do not invent numbers. If ranges are given, include the range text (e.g., "34–38 inches").
    
    3) tools:
       - measuring_tools: list of measuring items (e.g., tape measure, combination square, speed square, ruler).
       - cutting_tools: list of cutting tools (e.g., circular saw, jigsaw, hand saw).
       - drill_driver_tools: list of power drill/driver mentions (e.g., drill, impact driver, cordless drill).
       - clamps: list of clamps suitable for assembly/glue-ups (e.g., F-clamps, bar clamps, quick clamps).
    
    4) materials:
       - primary_wood: the main wood type recommended/specifically chosen for the beginner project (e.g., pine, poplar, birch).
       - additional_woods: any other wood types mentioned.
    
    5) construction:
       - joinery_methods: list the named joinery methods proposed (e.g., pocket holes/screws, dowels, butt joints with screws).
       - finishing:
         - steps_text: a summary sentence or phrase of the finishing steps/process as stated.
         - final_sanding_grit: the final grit number explicitly mentioned before finishing (e.g., 220). If not explicitly given, set to null.
         - finish_types: list of protective finish types explicitly named (e.g., polyurethane, shellac, varnish, lacquer, polycrylic).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s) and s.strip() != ""


def _list_nonempty(lst: Optional[List[str]]) -> bool:
    return bool(lst) and len(lst) > 0


def _fmt_items(items: List[str]) -> str:
    return ", ".join(items) if items else "None"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_safety_checks(evaluator: Evaluator, parent_node: Optional[Any], safety: SafetyInfo) -> None:
    node = evaluator.add_parallel(
        id="Safety_Equipment",
        desc="Safety equipment list includes all required PPE categories: eye, hearing, and respiratory protection.",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_list_nonempty(safety.eye_protection),
        id="Safety_Eye_Provided",
        desc=f"Eye protection provided: {_fmt_items(safety.eye_protection)}",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_list_nonempty(safety.hearing_protection),
        id="Safety_Hearing_Provided",
        desc=f"Hearing protection provided: {_fmt_items(safety.hearing_protection)}",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_list_nonempty(safety.respiratory_protection),
        id="Safety_Respiratory_Provided",
        desc=f"Respiratory protection provided: {_fmt_items(safety.respiratory_protection)}",
        parent=node,
        critical=True,
    )


async def build_workbench_checks(evaluator: Evaluator, parent_node: Optional[Any], bench: WorkbenchSpecs) -> None:
    node = evaluator.add_parallel(
        id="Workbench_Specifications",
        desc="Workbench specs include height, depth, and length within ergonomic guidelines.",
        parent=parent_node,
        critical=True,
    )

    # Presence checks
    evaluator.add_custom_node(
        result=_nonempty(bench.height_text),
        id="Workbench_Height_Provided",
        desc=f"Workbench height specified: {bench.height_text or 'None'}",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty(bench.depth_text),
        id="Workbench_Depth_Provided",
        desc=f"Workbench depth specified: {bench.depth_text or 'None'}",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty(bench.length_text),
        id="Workbench_Length_Provided",
        desc=f"Workbench length specified: {bench.length_text or 'None'}",
        parent=node,
        critical=True,
    )

    # Range conformance checks (verified against the answer text)
    height_claim = (
        f"The plan's workbench height ({bench.height_text or 'unspecified'}) falls within 34–38 inches inclusive, "
        "or it states a range that includes or overlaps this interval."
    )
    height_node = evaluator.add_leaf(
        id="Workbench_Height_Range",
        desc="Workbench height is within 34–38 inches.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=height_claim,
        node=height_node,
        additional_instruction=(
            "Judge based solely on the provided plan. If height is a single value, check it lies within 34–38 inches. "
            "If a range is stated, check whether the range covers or lies within 34–38 inches. Accept typical bench "
            "height if explicitly within the interval. Do not infer missing measurements."
        ),
    )

    depth_claim = (
        f"The plan's workbench depth ({bench.depth_text or 'unspecified'}) falls within 18–24 inches inclusive, "
        "or it states a range that includes or overlaps this interval."
    )
    depth_node = evaluator.add_leaf(
        id="Workbench_Depth_Range",
        desc="Workbench depth is within 18–24 inches.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=depth_claim,
        node=depth_node,
        additional_instruction=(
            "Judge using only the plan text. If a single depth is given, it must be 18–24 inches. If a range is given, "
            "ensure it lies within or overlaps 18–24 inches."
        ),
    )

    length_claim = (
        f"The plan's workbench length ({bench.length_text or 'unspecified'}) is at least 4 feet (48 inches)."
    )
    length_node = evaluator.add_leaf(
        id="Workbench_Length_Min",
        desc="Workbench length is at least 4 feet (48 inches).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=length_claim,
        node=length_node,
        additional_instruction=(
            "Confirm that the stated bench length is ≥ 4 ft (48 in). If provided in feet or inches, convert logically. "
            "Ranges are acceptable if the minimum is ≥ 4 ft."
        ),
    )


async def build_tools_checks(evaluator: Evaluator, parent_node: Optional[Any], tools: CoreTools) -> None:
    node = evaluator.add_parallel(
        id="Core_Tools",
        desc="Core tools include measuring, cutting, drill/driver, and clamps.",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_list_nonempty(tools.measuring_tools),
        id="Tools_Measuring_Provided",
        desc=f"Measuring tool(s) provided: {_fmt_items(tools.measuring_tools)}",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_list_nonempty(tools.cutting_tools),
        id="Tools_Cutting_Provided",
        desc=f"Cutting tool(s) provided: {_fmt_items(tools.cutting_tools)}",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_list_nonempty(tools.drill_driver_tools),
        id="Tools_DrillDriver_Provided",
        desc=f"Drill/driver tool(s) provided: {_fmt_items(tools.drill_driver_tools)}",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_list_nonempty(tools.clamps),
        id="Tools_Clamps_Provided",
        desc=f"Clamps provided: {_fmt_items(tools.clamps)}",
        parent=node,
        critical=True,
    )


async def build_material_checks(evaluator: Evaluator, parent_node: Optional[Any], mat: MaterialInfo) -> None:
    node = evaluator.add_parallel(
        id="Material_Selection",
        desc="Beginner-friendly wood selection is specified.",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_nonempty(mat.primary_wood),
        id="Material_Specified",
        desc=f"Primary wood specified: {mat.primary_wood or 'None'}",
        parent=node,
        critical=True,
    )

    # Verify beginner-friendly choice using the plan text context
    mat_claim = (
        f"The specified primary wood ({mat.primary_wood or 'unspecified'}) is beginner-friendly and easy to work, "
        "such as softwoods (pine, poplar, spruce/fir) or beginner-friendly hardwoods (birch, soft maple), or birch plywood."
    )
    mat_node = evaluator.add_leaf(
        id="Material_Beginner_Friendly",
        desc="Primary wood is beginner-friendly (e.g., pine, poplar, birch, soft maple).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=mat_claim,
        node=mat_node,
        additional_instruction=(
            "Determine based only on the plan. Accept woods explicitly known to be beginner-friendly: pine, poplar, "
            "spruce, fir, birch, soft maple, birch plywood, or similar. Reject obviously advanced/dense woods if "
            "they're presented as the primary beginner choice."
        ),
    )


async def build_construction_checks(evaluator: Evaluator, parent_node: Optional[Any], cons: ConstructionInfo) -> None:
    node = evaluator.add_parallel(
        id="Construction_Approach",
        desc="Beginner-friendly joinery and proper finishing (final 220 grit and protective finish).",
        parent=parent_node,
        critical=True,
    )

    # Joinery presence
    evaluator.add_custom_node(
        result=_list_nonempty(cons.joinery_methods),
        id="Joinery_Specified",
        desc=f"Joinery method(s) specified: {_fmt_items(cons.joinery_methods)}",
        parent=node,
        critical=True,
    )

    # Joinery appropriateness
    joinery_list_str = _fmt_items(cons.joinery_methods)
    joinery_claim = (
        f"The specified joinery method(s) ({joinery_list_str}) are beginner-appropriate (pocket holes/screws, dowels, or butt joints with screws)."
    )
    joinery_node = evaluator.add_leaf(
        id="Joinery_Beginner_Appropriate",
        desc="Joinery is beginner-appropriate (pocket holes, dowels, or butt joints with screws).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=joinery_claim,
        node=joinery_node,
        additional_instruction=(
            "Accept pocket holes/pocket screws, dowel joinery, and butt joints with screws as beginner-friendly. "
            "If other joinery is mentioned, judge whether it clearly matches one of these beginner categories."
        ),
    )

    # Finishing presence
    evaluator.add_custom_node(
        result=_nonempty(cons.finishing.steps_text),
        id="Finishing_Steps_Specified",
        desc=f"Finishing process specified: {cons.finishing.steps_text or 'None'}",
        parent=node,
        critical=True,
    )

    # Final sanding check
    sanding_claim = (
        f"The finishing process includes sanding with a final sanding grit of 220 or higher before applying finish. "
        f"Final grit stated: {cons.finishing.final_sanding_grit if cons.finishing.final_sanding_grit is not None else 'unspecified'}."
    )
    sanding_node = evaluator.add_leaf(
        id="Finishing_Final_220",
        desc="Final sanding at 220 grit or higher before finishing.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=sanding_claim,
        node=sanding_node,
        additional_instruction=(
            "Pass if the plan explicitly states final sanding at 220 grit or a higher grit (e.g., 240, 320) before finish. "
            "If the plan does not specify the final grit, mark as incorrect."
        ),
    )

    # Protective finish check
    finish_types_str = _fmt_items(cons.finishing.finish_types)
    finish_claim = (
        f"The plan applies a protective finish (e.g., polyurethane, shellac, varnish, lacquer, polycrylic). "
        f"Named finishes: {finish_types_str}."
    )
    finish_node = evaluator.add_leaf(
        id="Finishing_Protective",
        desc="Protective finish (polyurethane, shellac, varnish, lacquer, or polycrylic) is applied.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=finish_claim,
        node=finish_node,
        additional_instruction=(
            "Accept protective finishes like polyurethane (including wipe-on), polycrylic, shellac, varnish, and lacquer. "
            "Oil-only finishes without a protective film (e.g., mineral oil) do not meet the requirement unless clearly stated as protective."
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
    """
    Evaluate the woodworking setup and beginner project plan.
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

    # Extract plan
    plan: PlanExtraction = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="woodshop_plan_extraction",
    )

    # Build verification tree according to rubric
    # Root is critical; all children must be critical
    await build_safety_checks(evaluator, root, plan.safety)
    await build_workbench_checks(evaluator, root, plan.workbench)
    await build_tools_checks(evaluator, root, plan.tools)
    await build_material_checks(evaluator, root, plan.materials)
    await build_construction_checks(evaluator, root, plan.construction)

    # Return structured summary
    return evaluator.get_summary()