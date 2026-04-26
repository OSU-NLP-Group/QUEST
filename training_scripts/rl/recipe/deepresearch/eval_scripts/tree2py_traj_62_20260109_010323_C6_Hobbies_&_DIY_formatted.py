import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "beginner_bookshelf_project"
TASK_DESCRIPTION = """
You are a beginner woodworker living in Austin, Texas, and want to build your first furniture project: a bookshelf for your home office. You plan to use a local community makerspace rather than building your own workshop. Your home office wall space allows for a bookshelf no wider than 36 inches, and you need at least 4 adjustable shelves to accommodate books of various sizes. The shelves must be sturdy enough to hold a typical book collection (minimum 35 pounds per running foot).

Your project planning must include:

1. Makerspace Selection: Identify a suitable makerspace in Austin, Texas that:
   - Has a monthly membership cost of $80 or less
   - Provides access to woodworking equipment
   - Offers or requires safety certification/training for woodshop access

2. Safety Compliance: Demonstrate your understanding of:
   - The mandatory safety training requirements before accessing the woodshop
   - All required personal protective equipment (PPE) you must use

3. Design Specifications: Create a bookshelf design that:
   - Does not exceed 36 inches in width
   - Includes at least 4 adjustable shelves
   - Can support a minimum of 35 pounds per running foot
   - If using 3/4" plywood for shelves, ensures no unsupported span exceeds 32 inches
   - Specifies an appropriate joinery method suitable for beginners

4. Material and Budget Plan: Develop a plan that:
   - Keeps total material costs at or below $100
   - Uses commonly available lumber (pine dimensional lumber or plywood)
   - Includes a complete materials list with dimensions and estimated costs

5. Finishing Process: Outline a proper finishing procedure that:
   - Follows appropriate sanding grit progression (minimum 3 grits, starting around 120 and ending at 180-220)
   - Applies a protective finish suitable for furniture
   - If using water-based polyurethane: specifies at least 3 coats with proper drying time (2-4 hours) between coats
   - Includes light sanding between finish coats

Provide your complete project plan with all specifications, material lists, and process steps. Include reference URLs to support your makerspace selection, safety requirements, structural design decisions, and finishing techniques.
"""


# ---------------------------
# Data Models for Extraction
# ---------------------------

class Makerspace(BaseModel):
    name: Optional[str] = None
    location_city_state: Optional[str] = None
    monthly_cost: Optional[str] = None
    woodworking_access: Optional[str] = None  # e.g., "yes", "woodshop access", "table saw available"
    safety_training_requirement: Optional[str] = None  # e.g., "orientation required", "safety class required"
    urls: List[str] = Field(default_factory=list)  # official website pages or relevant references


class SafetyInfo(BaseModel):
    mandatory_training_statement: Optional[str] = None
    ppe_eye: bool = False
    ppe_hearing: bool = False
    ppe_respiratory: bool = False
    ppe_urls: List[str] = Field(default_factory=list)


class DimensionalConstraints(BaseModel):
    width_in: Optional[str] = None
    adjustable_shelf_count: Optional[str] = None


class StructuralRequirements(BaseModel):
    load_capacity_lbs_per_ft: Optional[str] = None
    plywood_thickness_in: Optional[str] = None
    max_unsupported_span_in: Optional[str] = None
    structural_urls: List[str] = Field(default_factory=list)


class DesignSpecs(BaseModel):
    dimensions: DimensionalConstraints = DimensionalConstraints()
    structural: StructuralRequirements = StructuralRequirements()
    joinery_method: Optional[str] = None


class MaterialItem(BaseModel):
    name: Optional[str] = None
    dimensions: Optional[str] = None  # e.g., "3/4\" x 12\" x 96\""
    quantity: Optional[str] = None
    unit_cost: Optional[str] = None
    extended_cost: Optional[str] = None


class MaterialPlan(BaseModel):
    uses_common_lumber: bool = False
    total_material_cost: Optional[str] = None
    materials: List[MaterialItem] = Field(default_factory=list)


class FinishingPlan(BaseModel):
    sanding_grits: List[str] = Field(default_factory=list)  # e.g., ["120", "150", "220"]
    sanding_urls: List[str] = Field(default_factory=list)
    finish_type: Optional[str] = None  # e.g., "water-based polyurethane"
    coats_count: Optional[str] = None
    drying_time_hours: Optional[str] = None
    inter_coat_sanding: bool = False
    finishing_urls: List[str] = Field(default_factory=list)


class ProjectPlanExtraction(BaseModel):
    makerspace: Makerspace = Makerspace()
    safety: SafetyInfo = SafetyInfo()
    design: DesignSpecs = DesignSpecs()
    materials: MaterialPlan = MaterialPlan()
    finishing: FinishingPlan = FinishingPlan()


# ---------------------------
# Extraction Prompt
# ---------------------------

def prompt_extract_project_plan() -> str:
    return """
Extract the complete project plan details from the answer. Return a JSON object with the following structure and rules:

makerspace:
  - name: the selected makerspace name (string)
  - location_city_state: the location string as provided (e.g., "Austin, TX" or "Austin, Texas")
  - monthly_cost: the monthly membership cost value or text (string). Include currency symbol if present.
  - woodworking_access: text snippet indicating access to woodworking tools/equipment (e.g., "woodshop access", "woodworking tools available")
  - safety_training_requirement: text indicating safety training/certification requirement or offering for woodshop access
  - urls: ARRAY of URLs explicitly provided in the answer that support makerspace selection/requirements

safety:
  - mandatory_training_statement: text in the answer that states mandatory training before woodshop access
  - ppe_eye: BOOLEAN true if safety glasses or goggles are explicitly mentioned; false otherwise
  - ppe_hearing: BOOLEAN true if hearing protection is explicitly mentioned; false otherwise
  - ppe_respiratory: BOOLEAN true if dust mask or respirator is explicitly mentioned; false otherwise
  - ppe_urls: ARRAY of URLs provided that support PPE requirements

design:
  dimensions:
    - width_in: the width (in inches) as stated (string, do not convert to number; e.g., "34", "36", "approx. 32")
    - adjustable_shelf_count: the count of adjustable shelves stated (string; e.g., "4", "5")
  structural:
    - load_capacity_lbs_per_ft: text stating minimum load capacity requirement (string; e.g., "≥35 lb/ft", "at least 35 pounds per running foot")
    - plywood_thickness_in: text of plywood thickness used for shelves (string; e.g., "3/4")
    - max_unsupported_span_in: text of span limitation (string; e.g., "32")
    - structural_urls: ARRAY of URLs provided that support structural span/load guidance/calculations
  - joinery_method: text describing chosen joinery suitable for beginners (e.g., "pocket holes", "dowels", "reinforced butt joints")

materials:
  - uses_common_lumber: BOOLEAN true if plan uses pine dimensional lumber or plywood
  - total_material_cost: total material cost as stated (string; e.g., "$95", "about $90")
  - materials: ARRAY of objects, each with:
      • name: item name (string)
      • dimensions: item dimensions (string)
      • quantity: quantity (string)
      • unit_cost: per-item cost (string)
      • extended_cost: total cost for that line item (string)

finishing:
  - sanding_grits: ARRAY of grit values or labels in order used (strings; e.g., ["120", "150", "220"])
  - sanding_urls: ARRAY of URLs provided that support sanding progression/technique
  - finish_type: text of the protective finish (e.g., "water-based polyurethane", "oil-based polyurethane", "wipe-on poly", etc.)
  - coats_count: number of coats stated (string; e.g., "3", "three")
  - drying_time_hours: drying time between coats as stated (string; e.g., "2-4 hours")
  - inter_coat_sanding: BOOLEAN true if light sanding between coats is explicitly included
  - finishing_urls: ARRAY of URLs provided that support finishing technique/coat schedule

General extraction rules:
- Extract only what is explicitly in the answer; do not invent.
- If any field is not mentioned, return null for strings and false for booleans. Use empty arrays for URLs when none are provided.
- For URL arrays, extract only valid URLs (plain or markdown link targets). Do not infer any URLs. Include full URLs with protocol.
"""


# ---------------------------
# Helper Functions
# ---------------------------

def _is_water_based_poly(finish_type: Optional[str]) -> bool:
    if not finish_type:
        return False
    ft = finish_type.lower()
    return ("polyurethane" in ft) and ("water" in ft or "water-based" in ft or "waterbased" in ft)


def _list_to_readable(items: List[str]) -> str:
    return ", ".join(items) if items else ""


def _has_materials_with_dims_and_cost(materials: List[MaterialItem]) -> bool:
    if not materials:
        return False
    # Require at least one line item with dimensions and some cost fields
    for m in materials:
        if (m.dimensions and m.dimensions.strip()) and ((m.unit_cost and m.unit_cost.strip()) or (m.extended_cost and m.extended_cost.strip())):
            return True
    return False


# ---------------------------
# Verification Subtrees
# ---------------------------

async def verify_makerspace_selection(evaluator: Evaluator, parent_node, plan: ProjectPlanExtraction) -> None:
    ms_node = evaluator.add_parallel(
        id="Makerspace_Selection",
        desc="Selection of an appropriate makerspace facility that meets location, cost, and capability requirements",
        parent=parent_node,
        critical=True
    )

    core_node = evaluator.add_parallel(
        id="Makerspace_Core_Requirements",
        desc="Core makerspace requirements for location, cost, equipment access, and safety training",
        parent=ms_node,
        critical=True
    )

    urls = plan.makerspace.urls

    # Location requirement
    loc_leaf = evaluator.add_leaf(
        id="Location_Requirement",
        desc="Makerspace must be located in Austin, Texas",
        parent=core_node,
        critical=True
    )
    await evaluator.verify(
        claim="The selected makerspace is located in Austin, Texas.",
        node=loc_leaf,
        sources=urls if urls else None,
        additional_instruction="Use the makerspace page(s) to confirm the location is in Austin, Texas. If no URL is provided, use the answer text; judge incorrect if the location is not clearly Austin, Texas."
    )

    # Cost requirement
    cost_leaf = evaluator.add_leaf(
        id="Cost_Requirement",
        desc="Monthly membership cost must not exceed $80",
        parent=core_node,
        critical=True
    )
    cost_text = plan.makerspace.monthly_cost or "unspecified"
    await evaluator.verify(
        claim=f"The makerspace offers a monthly membership priced at $80 or less (answer states: {cost_text}).",
        node=cost_leaf,
        sources=urls if urls else None,
        additional_instruction="Check membership/pricing page. Passing examples: $80, $75, or cheaper monthly option. If only annual pricing or cost > $80 is given, mark incorrect."
    )

    # Woodworking access
    wood_leaf = evaluator.add_leaf(
        id="Woodworking_Access",
        desc="Makerspace must provide access to woodworking tools and equipment",
        parent=core_node,
        critical=True
    )
    await evaluator.verify(
        claim="The selected makerspace provides access to woodworking equipment (e.g., woodshop, table saw, bandsaw).",
        node=wood_leaf,
        sources=urls if urls else None,
        additional_instruction="Look for 'woodshop', 'woodworking', or a listed set of woodworking tools on the makerspace pages."
    )

    # Safety training availability
    safety_train_leaf = evaluator.add_leaf(
        id="Safety_Training_Availability",
        desc="Makerspace must offer or require safety certification/training for woodshop access",
        parent=core_node,
        critical=True
    )
    await evaluator.verify(
        claim="The makerspace offers or requires orientation/safety certification/class before accessing the woodshop.",
        node=safety_train_leaf,
        sources=urls if urls else None,
        additional_instruction="Look for 'orientation', 'safety class', 'certification' requirements tied to woodshop access."
    )

    # Makerspace reference URLs
    if urls and len(urls) > 0:
        ref_leaf = evaluator.add_leaf(
            id="Makerspace_Reference",
            desc="Provide URL reference(s) supporting the selected makerspace and its stated requirements",
            parent=ms_node,
            critical=True
        )
        await evaluator.verify(
            claim="These URLs support the makerspace selection and confirm location in Austin, monthly membership ≤ $80, woodworking access, and safety training for woodshop.",
            node=ref_leaf,
            sources=urls,
            additional_instruction="Mark incorrect if the URLs are irrelevant or do not substantiate the stated criteria."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Makerspace_Reference",
            desc="Provide URL reference(s) supporting the makerspace and its requirements — none provided",
            parent=ms_node,
            critical=True
        )


async def verify_safety_compliance(evaluator: Evaluator, parent_node, plan: ProjectPlanExtraction) -> None:
    safety_node = evaluator.add_parallel(
        id="Safety_Compliance",
        desc="Demonstration of understanding and commitment to safety requirements before beginning the project",
        parent=parent_node,
        critical=True
    )

    # Mandatory training statement
    mandatory_leaf = evaluator.add_leaf(
        id="Mandatory_Training",
        desc="State the mandatory safety training/orientation requirement before woodshop access",
        parent=safety_node,
        critical=True
    )
    statement = plan.safety.mandatory_training_statement or ""
    await evaluator.verify(
        claim="The plan clearly states that mandatory safety training/orientation is required before woodshop access.",
        node=mandatory_leaf,
        sources=None,
        additional_instruction=f"Check the answer text for a clear statement. Current extracted text: '{statement}'. If unclear or missing, mark incorrect."
    )

    # PPE requirements grouped
    ppe_node = evaluator.add_parallel(
        id="PPE_Requirements",
        desc="Identify all required PPE (safety glasses, hearing protection, dust mask/respirator)",
        parent=safety_node,
        critical=True
    )

    # Eye protection
    eye_leaf = evaluator.add_leaf(
        id="Eye_Protection",
        desc="Include safety glasses or goggles",
        parent=ppe_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes safety glasses or goggles as required PPE.",
        node=eye_leaf,
        sources=None,
        additional_instruction="Allow reasonable variants like 'ANSI Z87 safety glasses', 'goggles'."
    )

    # Hearing protection
    hearing_leaf = evaluator.add_leaf(
        id="Hearing_Protection",
        desc="Include hearing protection",
        parent=ppe_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes hearing protection as required PPE.",
        node=hearing_leaf,
        sources=None,
        additional_instruction="Accept 'ear muffs' or 'ear plugs' as hearing protection."
    )

    # Respiratory protection
    resp_leaf = evaluator.add_leaf(
        id="Respiratory_Protection",
        desc="Include a dust mask or respirator for wood dust",
        parent=ppe_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes a dust mask or respirator for wood dust as required PPE.",
        node=resp_leaf,
        sources=None,
        additional_instruction="Accept terms like 'N95', 'dust mask', or 'respirator'."
    )

    # PPE reference URLs
    ppe_urls = plan.safety.ppe_urls
    if ppe_urls and len(ppe_urls) > 0:
        ppe_ref_leaf = evaluator.add_leaf(
            id="PPE_Reference",
            desc="Provide URL reference(s) supporting the PPE requirement",
            parent=ppe_node,
            critical=True
        )
        await evaluator.verify(
            claim="These references explicitly list required PPE for woodshop work: safety glasses, hearing protection, and dust mask/respirator.",
            node=ppe_ref_leaf,
            sources=ppe_urls,
            additional_instruction="Mark incorrect if the URLs do not substantiate PPE requirements."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="PPE_Reference",
            desc="Provide URL reference(s) supporting the PPE requirement — none provided",
            parent=ppe_node,
            critical=True
        )


async def verify_design_specifications(evaluator: Evaluator, parent_node, plan: ProjectPlanExtraction) -> None:
    design_node = evaluator.add_parallel(
        id="Design_Specifications",
        desc="Bookshelf design must meet dimensional, structural, and functional requirements",
        parent=parent_node,
        critical=True
    )

    # Dimensional constraints
    dim_node = evaluator.add_parallel(
        id="Dimensional_Constraints",
        desc="Overall dimensions must meet specified space requirements",
        parent=design_node,
        critical=True
    )
    # Width ≤ 36
    width_leaf = evaluator.add_leaf(
        id="Width_Specification",
        desc="Bookshelf width must not exceed 36 inches",
        parent=dim_node,
        critical=True
    )
    width_text = plan.design.dimensions.width_in or "unspecified"
    await evaluator.verify(
        claim=f"The bookshelf width specified in the plan is 36 inches or less (answer states: {width_text}).",
        node=width_leaf,
        sources=None,
        additional_instruction="If width is greater than 36 or missing, mark incorrect."
    )

    # At least 4 adjustable shelves
    shelves_leaf = evaluator.add_leaf(
        id="Shelf_Count",
        desc="Design must include at least 4 adjustable shelves",
        parent=dim_node,
        critical=True
    )
    count_text = plan.design.dimensions.adjustable_shelf_count or "unspecified"
    await evaluator.verify(
        claim=f"The design includes at least 4 adjustable shelves (answer indicates: {count_text}).",
        node=shelves_leaf,
        sources=None,
        additional_instruction="Accept '≥4', 'four', 'at least 4', or explicit adjustable shelf count ≥ 4."
    )

    # Structural requirements
    struct_node = evaluator.add_parallel(
        id="Structural_Requirements",
        desc="Design must meet load capacity and span requirements for safe book storage",
        parent=design_node,
        critical=True
    )

    # Load capacity ≥ 35 lb/ft
    load_leaf = evaluator.add_leaf(
        id="Load_Capacity",
        desc="Shelves must support minimum 35 pounds per running foot",
        parent=struct_node,
        critical=True
    )
    load_text = plan.design.structural.load_capacity_lbs_per_ft or "unspecified"
    await evaluator.verify(
        claim=f"The shelves in the plan can support a minimum of 35 pounds per running foot (answer indicates: {load_text}).",
        node=load_leaf,
        sources=None,
        additional_instruction="If the plan indicates a lower capacity or is missing, mark incorrect."
    )

    # Span limitation for 3/4-in plywood
    span_leaf = evaluator.add_leaf(
        id="Span_Limitation",
        desc="If using 3/4 inch plywood shelves, maximum unsupported span must not exceed 32 inches",
        parent=struct_node,
        critical=True
    )
    span_text = plan.design.structural.max_unsupported_span_in or "unspecified"
    ply_text = plan.design.structural.plywood_thickness_in or "unspecified"
    await evaluator.verify(
        claim=f"If using 3/4 inch plywood shelves, the plan limits unsupported spans to 32 inches or less (answer: plywood {ply_text}, span {span_text}).",
        node=span_leaf,
        sources=None,
        additional_instruction="If the plan uses 3/4\" plywood, spans must be ≤ 32\". If not using 3/4\", consider the claim satisfied if no violation is implied."
    )

    # Structural reference URLs
    struct_urls = plan.design.structural.structural_urls
    if struct_urls and len(struct_urls) > 0:
        struct_ref_leaf = evaluator.add_leaf(
            id="Structural_Reference",
            desc="Provide URL reference(s) supporting structural span/load guidance or calculations used",
            parent=struct_node,
            critical=True
        )
        await evaluator.verify(
            claim="The provided references support shelf load capacity expectations and span guidance for 3/4\" plywood shelves.",
            node=struct_ref_leaf,
            sources=struct_urls,
            additional_instruction="Mark incorrect if URLs are irrelevant or do not substantiate load/span guidance."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Structural_Reference",
            desc="Provide URL reference(s) supporting structural guidance — none provided",
            parent=struct_node,
            critical=True
        )

    # Joinery method suitable for beginners
    joinery_leaf = evaluator.add_leaf(
        id="Joinery_Method",
        desc="Specify an appropriate joinery method suitable for beginners (e.g., pocket holes, dowels, or reinforced butt joints)",
        parent=design_node,
        critical=True
    )
    jm = plan.design.joinery_method or "unspecified"
    await evaluator.verify(
        claim=f"The joinery method '{jm}' specified is appropriate for beginners (e.g., pocket holes, dowels, reinforced butt joints).",
        node=joinery_leaf,
        sources=None,
        additional_instruction="Accept beginner-friendly methods like pocket holes (Kreg), dowels, biscuits, or reinforced butt joints."
    )


async def verify_material_and_finishing_plan(evaluator: Evaluator, parent_node, plan: ProjectPlanExtraction) -> None:
    mf_node = evaluator.add_parallel(
        id="Material_and_Finishing_Plan",
        desc="Material selection and finishing plan meeting budget and process requirements",
        parent=parent_node,
        critical=True
    )

    # Material selection
    mat_sel_node = evaluator.add_parallel(
        id="Material_Selection",
        desc="Materials and budget plan meeting availability and cost constraints",
        parent=mf_node,
        critical=True
    )

    # Budget compliance
    budget_leaf = evaluator.add_leaf(
        id="Budget_Compliance",
        desc="Total material cost must not exceed $100",
        parent=mat_sel_node,
        critical=True
    )
    total_cost_text = plan.materials.total_material_cost or "unspecified"
    await evaluator.verify(
        claim=f"The total material cost is at or below $100 (answer indicates: {total_cost_text}).",
        node=budget_leaf,
        sources=None,
        additional_instruction="If total cost clearly exceeds $100 or is missing, mark incorrect."
    )

    # Lumber type
    lumber_leaf = evaluator.add_leaf(
        id="Lumber_Type",
        desc="Must use commonly available lumber (pine dimensional lumber or plywood)",
        parent=mat_sel_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan uses commonly available lumber (pine dimensional lumber and/or plywood).",
        node=lumber_leaf,
        sources=None,
        additional_instruction="Look for 'pine', 'dimensional lumber', or 'plywood' in the materials."
    )

    # Materials list completeness (existence check as custom leaf)
    mat_list_result = _has_materials_with_dims_and_cost(plan.materials.materials)
    evaluator.add_custom_node(
        result=mat_list_result,
        id="Material_List",
        desc="Provide a complete materials list with dimensions and estimated costs",
        parent=mat_sel_node,
        critical=True
    )

    # Finishing process (sequential)
    fin_seq_node = evaluator.add_sequential(
        id="Finishing_Process",
        desc="Finishing plan following sanding, finishing, and inter-coat procedures",
        parent=mf_node,
        critical=True
    )

    # Surface preparation
    surf_prep_node = evaluator.add_parallel(
        id="Surface_Preparation",
        desc="Sanding plan following proper grit progression",
        parent=fin_seq_node,
        critical=True
    )

    # Grit progression
    grit_leaf = evaluator.add_leaf(
        id="Grit_Progression",
        desc="Use minimum 3 grits starting around 120 and ending at 180-220",
        parent=surf_prep_node,
        critical=True
    )
    grits_text = _list_to_readable(plan.finishing.sanding_grits)
    await evaluator.verify(
        claim=f"The sanding plan uses at least three grits, starting around 120 and ending between 180–220 (answer shows: {grits_text}).",
        node=grit_leaf,
        sources=None,
        additional_instruction="Accept progressions like 120→150→220, 120→180→220, etc. Mark incorrect if fewer than 3 grits or ending grit <180."
    )

    # Sanding reference URLs
    sanding_urls = plan.finishing.sanding_urls
    if sanding_urls and len(sanding_urls) > 0:
        sand_ref_leaf = evaluator.add_leaf(
            id="Sanding_Reference",
            desc="Provide URL reference(s) supporting sanding progression/technique",
            parent=surf_prep_node,
            critical=True
        )
        await evaluator.verify(
            claim="These references support proper sanding progression for furniture finishing (starting ~120, ending ~180–220).",
            node=sand_ref_leaf,
            sources=sanding_urls,
            additional_instruction="Mark incorrect if URLs do not substantiate sanding progression."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Sanding_Reference",
            desc="Provide URL reference(s) supporting sanding progression — none provided",
            parent=surf_prep_node,
            critical=True
        )

    # Protective finish
    prot_fin_node = evaluator.add_parallel(
        id="Protective_Finish",
        desc="Protective finish application plan suitable for furniture",
        parent=fin_seq_node,
        critical=True
    )

    # Finish type
    finish_leaf = evaluator.add_leaf(
        id="Finish_Type",
        desc="Apply a protective finish suitable for furniture",
        parent=prot_fin_node,
        critical=True
    )
    ftype = plan.finishing.finish_type or "unspecified"
    await evaluator.verify(
        claim=f"The plan applies a protective finish suitable for furniture (finish type: {ftype}).",
        node=finish_leaf,
        sources=None,
        additional_instruction="Accept polyurethane (water/oil-based), varnish, lacquer, shellac, or other appropriate furniture finishes."
    )

    # Coat requirements (conditional)
    if _is_water_based_poly(plan.finishing.finish_type):
        coat_leaf = evaluator.add_leaf(
            id="Coat_Requirements",
            desc="If using water-based polyurethane: specify at least 3 coats",
            parent=prot_fin_node,
            critical=True
        )
        coats_text = plan.finishing.coats_count or "unspecified"
        await evaluator.verify(
            claim=f"The plan specifies at least 3 coats of water-based polyurethane (answer indicates: {coats_text}).",
            node=coat_leaf,
            sources=None,
            additional_instruction="If coats < 3 or missing, mark incorrect."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="Coat_Requirements",
            desc="Coat requirement not applicable (finish type is not water-based polyurethane)",
            parent=prot_fin_node,
            critical=True
        )

    # Drying time (conditional)
    if _is_water_based_poly(plan.finishing.finish_type):
        dry_leaf = evaluator.add_leaf(
            id="Drying_Time",
            desc="If using water-based polyurethane: specify drying time of 2-4 hours between coats",
            parent=prot_fin_node,
            critical=True
        )
        dry_text = plan.finishing.drying_time_hours or "unspecified"
        await evaluator.verify(
            claim=f"The plan specifies 2–4 hours drying time between coats for water-based polyurethane (answer indicates: {dry_text}).",
            node=dry_leaf,
            sources=None,
            additional_instruction="If drying time outside 2–4 hours or missing, mark incorrect."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="Drying_Time",
            desc="Drying time requirement not applicable (finish type is not water-based polyurethane)",
            parent=prot_fin_node,
            critical=True
        )

    # Inter-coat sanding
    inter_sand_leaf = evaluator.add_leaf(
        id="Inter_Coat_Sanding",
        desc="Include light sanding between finish coats",
        parent=prot_fin_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes light sanding between finish coats.",
        node=inter_sand_leaf,
        sources=None,
        additional_instruction="Look for 'light sanding', 'scuff sanding', or similar between coats."
    )

    # Finishing reference URLs
    fin_urls = plan.finishing.finishing_urls
    if fin_urls and len(fin_urls) > 0:
        fin_ref_leaf = evaluator.add_leaf(
            id="Finishing_Reference",
            desc="Provide URL reference(s) supporting finishing technique/coat schedule",
            parent=prot_fin_node,
            critical=True
        )
        await evaluator.verify(
            claim="These references support the specified finishing technique and coat schedule.",
            node=fin_ref_leaf,
            sources=fin_urls,
            additional_instruction="Mark incorrect if URLs are irrelevant or do not support the stated finishing procedure."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Finishing_Reference",
            desc="Provide URL reference(s) supporting finishing technique — none provided",
            parent=prot_fin_node,
            critical=True
        )


# ---------------------------
# Main Evaluation Entry Point
# ---------------------------

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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract the whole project plan
    plan = await evaluator.extract(
        prompt=prompt_extract_project_plan(),
        template_class=ProjectPlanExtraction,
        extraction_name="project_plan"
    )

    # Build the top-level critical sequential node as per rubric
    project_node = evaluator.add_sequential(
        id="Complete_Beginner_Bookshelf_Project",
        desc="Evaluation of a complete beginner woodworking project plan including makerspace selection, safety compliance, design specifications, and material/finishing plans",
        parent=root,
        critical=True
    )

    # 1. Makerspace Selection
    await verify_makerspace_selection(evaluator, project_node, plan)

    # 2. Safety Compliance
    await verify_safety_compliance(evaluator, project_node, plan)

    # 3. Design Specifications
    await verify_design_specifications(evaluator, project_node, plan)

    # 4. Material and Finishing Plan
    await verify_material_and_finishing_plan(evaluator, project_node, plan)

    return evaluator.get_summary()