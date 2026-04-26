import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "beginner_dining_table_plan"
TASK_DESCRIPTION = """
You are planning to build your first dining table as a beginner woodworking project in your home workshop. Create a comprehensive project plan that addresses all essential requirements for safely completing this project.

Your plan must include:

1. Safety Equipment: Identify all required personal protective equipment necessary for woodworking operations.

2. Essential Tools: Specify the essential woodworking tools needed for this project, including:
   - Measuring and marking tools
   - A primary cutting tool (if you specify a table saw, include all OSHA-required safety features)
   - Power drill
   - Sander
   - Clamps (specify type and minimum quantity needed for panel glue-ups)
   - Work surface

3. Lumber Selection: Specify appropriate lumber for the table, including:
   - A hardwood species suitable for furniture making
   - Furniture-grade quality specifications
   - Required moisture content range (verify using moisture meter)
   - Specify that lumber should be kiln-dried
   - Calculate lumber quantity using the board feet formula and include waste allowance

4. Food-Safe Finishing: Select an appropriate finish for the dining table that is food-safe when fully cured and provides adequate durability for dining table use. Specify the finish type and application method.

5. Workshop Requirements: Confirm that your workshop meets minimum space requirements (minimum square footage) and specify an appropriate dust collection system that meets CFM requirements for your tools (including specific CFM needs for table saws if applicable).

6. Documentation: Provide references to:
   - OSHA woodworking safety standards
   - Sources for furniture-grade hardwood lumber (optional)
   - Beginner woodworking instruction resources (optional)
   - Food-safe finishing product specifications (optional)

Ensure all specifications are practical, verifiable, and suitable for a beginner woodworker building their first dining table.
"""

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class PPEExtraction(BaseModel):
    eye_protection: Optional[bool] = None
    hearing_protection: Optional[bool] = None
    respiratory_protection: Optional[bool] = None
    first_aid_kit: Optional[bool] = None


class ToolsExtraction(BaseModel):
    tape_measure: Optional[bool] = None
    combination_square_or_equivalent: Optional[bool] = None
    primary_cutting_tool: Optional[str] = None  # e.g., "table saw", "circular saw", "miter saw"
    uses_table_saw: Optional[bool] = None
    table_saw_blade_guard_included: Optional[bool] = None
    table_saw_riving_knife_or_splitter_included: Optional[bool] = None
    power_drill: Optional[bool] = None
    sander_specified: Optional[bool] = None
    clamps_type: Optional[str] = None  # e.g., "parallel clamps", "bar clamps", "pipe clamps"
    clamps_min_quantity_text: Optional[str] = None  # e.g., "at least 4", "4+"
    work_surface_specified: Optional[bool] = None


class LumberExtraction(BaseModel):
    hardwood_species: Optional[str] = None  # e.g., "oak", "maple", "walnut", "cherry"
    furniture_grade_spec: Optional[str] = None  # e.g., "Select & Better", "FAS", "S4S", "Quarter-sawn"
    kiln_dried_required: Optional[bool] = None
    moisture_content_range_text: Optional[str] = None  # e.g., "6–11% (ideally 6–8%)"
    moisture_meter_use: Optional[bool] = None
    board_feet_formula_used: Optional[bool] = None
    waste_allowance_text: Optional[str] = None  # e.g., "include 20–30% waste"


class FinishExtraction(BaseModel):
    finish_type: Optional[str] = None  # e.g., "oil-based polyurethane", "wipe-on poly", "hardwax oil"
    food_safe_when_fully_cured: Optional[bool] = None
    cure_before_food_contact: Optional[bool] = None
    durability_for_table_use: Optional[bool] = None
    application_method: Optional[str] = None  # e.g., "wipe-on", "brush", "spray"


class WorkshopExtraction(BaseModel):
    min_square_footage_stated: Optional[bool] = None
    min_square_footage_value_text: Optional[str] = None  # e.g., "≥100 sq ft", "at least 100 sq ft"
    ideal_square_footage_recommendation: Optional[bool] = None  # e.g., "~200+ sq ft"
    dust_collection_system_spec_text: Optional[str] = None
    general_cfm_range_mentioned: Optional[bool] = None  # 250–1000 CFM tool-dependent
    table_saw_cfm_mentioned: Optional[bool] = None  # ~500–600 CFM if table saw
    table_saw_cfm_value_text: Optional[str] = None
    air_velocity_4000_fpm_mentioned: Optional[bool] = None


class DocumentationExtraction(BaseModel):
    osha_urls: List[str] = Field(default_factory=list)
    lumber_source_urls: List[str] = Field(default_factory=list)
    beginner_resources_urls: List[str] = Field(default_factory=list)
    finish_product_spec_urls: List[str] = Field(default_factory=list)


class PlanExtraction(BaseModel):
    ppe: Optional[PPEExtraction] = None
    tools: Optional[ToolsExtraction] = None
    lumber: Optional[LumberExtraction] = None
    finish: Optional[FinishExtraction] = None
    workshop: Optional[WorkshopExtraction] = None
    docs: Optional[DocumentationExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
    Extract structured information from the beginner dining table project plan as follows.
    Return true/false booleans when the plan explicitly includes the item; otherwise return false or null if unknown.
    Return exact text for fields that require textual specification. For URL lists, extract explicit full URLs only.

    1) ppe:
       - eye_protection: true if safety glasses or goggles are included.
       - hearing_protection: true if earmuffs or earplugs are included.
       - respiratory_protection: true if a dust mask or respirator is included.
       - first_aid_kit: true if availability of a first aid kit is included/confirmed.

    2) tools:
       - tape_measure: true if a tape measure is specified.
       - combination_square_or_equivalent: true if a combination square (or equivalent square tool like speed square/try square) is specified.
       - primary_cutting_tool: the main saw/cutting tool named (e.g., "table saw", "circular saw", "miter saw", "track saw", "jigsaw"); return null if not specified.
       - uses_table_saw: true if the plan explicitly specifies a table saw as a primary cutting tool; else false.
       - table_saw_blade_guard_included: true if the plan includes a blade guard for the table saw; false otherwise or if no table saw.
       - table_saw_riving_knife_or_splitter_included: true if the plan includes a riving knife or splitter for the table saw; false otherwise or if no table saw.
       - power_drill: true if a power drill/driver is specified.
       - sander_specified: true if a random orbital sander (or equivalent) is specified.
       - clamps_type: the clamp type named for panel glue-ups (e.g., "parallel clamps", "bar clamps", "pipe clamps"); return null if not specified.
       - clamps_min_quantity_text: how many clamps are specified (e.g., "at least 4", "4", "4+"); return null if not specified.
       - work_surface_specified: true if a workbench or stable work surface is specified.

    3) lumber:
       - hardwood_species: the furniture-suitable hardwood species specified (e.g., oak, maple, walnut, cherry); return null if not specified.
       - furniture_grade_spec: any furniture-grade quality specification (e.g., "FAS", "Select & Better", "S4S", "quarter-sawn"); return null if not specified.
       - kiln_dried_required: true if the plan specifies lumber must be kiln-dried.
       - moisture_content_range_text: a text range (e.g., "acceptable 6–11%, ideally 6–8%") if specified.
       - moisture_meter_use: true if using a moisture meter to verify moisture content is specified.
       - board_feet_formula_used: true if the board-feet formula "Length(ft) × Width(in) × Thickness(in) ÷ 12" is explicitly stated or used.
       - waste_allowance_text: include the waste allowance statement if present (e.g., "include 20–30% waste").

    4) finish:
       - finish_type: the finish type (e.g., "oil-based polyurethane", "wipe-on poly", "hardwax oil"); return null if not specified.
       - food_safe_when_fully_cured: true if explicitly stated the finish is food-safe when fully cured or the plan references a product spec that supports this.
       - cure_before_food_contact: true if the plan states the finish must be fully cured before food contact.
       - durability_for_table_use: true if the plan addresses durability/protection suitable for dining table use.
       - application_method: the application method (e.g., "wipe-on", "brush", "spray"); return null if not specified.

    5) workshop:
       - min_square_footage_stated: true if the workshop minimum square footage is stated to meet at least 100 sq ft.
       - min_square_footage_value_text: the stated minimum square footage text (e.g., "at least 100 sq ft").
       - ideal_square_footage_recommendation: true if ~200+ sq ft is mentioned as ideal.
       - dust_collection_system_spec_text: any dust collection system description (e.g., "1.5 HP 700 CFM dust collector", "shop vac").
       - general_cfm_range_mentioned: true if the plan addresses the 250–1000 CFM tool-dependent requirement.
       - table_saw_cfm_mentioned: true if ~500–600 CFM at the table saw cabinet is addressed (only if a table saw is used or mentioned).
       - table_saw_cfm_value_text: the stated CFM value/range text for the table saw if mentioned (e.g., "500–600 CFM").
       - air_velocity_4000_fpm_mentioned: true if the 4,000 FPM air-velocity recommendation is mentioned.

    6) docs:
       - osha_urls: list all URLs that reference OSHA woodworking safety standards (prefer osha.gov pages; include any 1910.213 references if present).
       - lumber_source_urls: list URLs that are sources for furniture-grade hardwood lumber (optional).
       - beginner_resources_urls: list URLs for beginner woodworking instruction resources (optional).
       - finish_product_spec_urls: list URLs for food-safe finishing product specifications (optional).
    """


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_safety_nodes(evaluator: Evaluator, parent, plan: PlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Safety_Equipment_Compliance",
        desc="Includes all required PPE/safety items from constraints.",
        parent=parent,
        critical=False
    )

    # Eye Protection
    leaf_eye = evaluator.add_leaf(
        id="Eye_Protection",
        desc="Safety glasses or goggles are included.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes eye protection such as safety glasses or goggles.",
        node=leaf_eye,
        additional_instruction="Allow synonyms like 'ANSI-rated safety glasses' or 'protective goggles'."
    )

    # Hearing Protection
    leaf_hear = evaluator.add_leaf(
        id="Hearing_Protection",
        desc="Hearing protection (earmuffs or earplugs) is included.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes hearing protection such as earmuffs or earplugs.",
        node=leaf_hear,
        additional_instruction="Accept 'hearing protection', 'ear defenders', 'ear plugs', 'earmuffs'."
    )

    # Respiratory Protection
    leaf_resp = evaluator.add_leaf(
        id="Respiratory_Protection",
        desc="Dust mask or respirator is included.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes respiratory protection such as a dust mask or respirator.",
        node=leaf_resp,
        additional_instruction="Accept 'N95 mask', 'P100 respirator', 'dust mask', 'respirator'."
    )

    # First Aid Kit
    leaf_first_aid = evaluator.add_leaf(
        id="First_Aid_Kit",
        desc="First aid kit availability is included/confirmed.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan confirms availability of a first aid kit in the workshop.",
        node=leaf_first_aid,
        additional_instruction="Look for explicit mention of 'first aid kit' or equivalent readiness."
    )


async def build_tools_nodes(evaluator: Evaluator, parent, plan: PlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Essential_Tool_Requirements",
        desc="Specifies all essential tool categories required by the question/constraints.",
        parent=parent,
        critical=False
    )

    # Measuring and Marking Tools
    measure_node = evaluator.add_parallel(
        id="Measuring_And_Marking_Tools",
        desc="Includes required measuring/marking tools (tape measure and combination square or equivalent).",
        parent=node,
        critical=False
    )

    leaf_tape = evaluator.add_leaf(
        id="Tape_Measure",
        desc="Tape measure is specified.",
        parent=measure_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies a tape measure for measuring.",
        node=leaf_tape,
        additional_instruction="Accept common measuring tools; focus on explicit 'tape measure'."
    )

    leaf_square = evaluator.add_leaf(
        id="Combination_Square_Or_Equivalent",
        desc="Combination square (or equivalent square tool) is specified.",
        parent=measure_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies a square tool such as a combination square, speed square, or try square for marking and squaring.",
        node=leaf_square,
        additional_instruction="Accept 'combination square', 'speed square', 'try square', 'engineer's square'."
    )

    # Primary Cutting Tool
    cutting_node = evaluator.add_parallel(
        id="Primary_Cutting_Tool",
        desc="Specifies a primary cutting tool/saw; if a table saw is specified, it includes OSHA-required safety features.",
        parent=node,
        critical=False
    )

    leaf_saw_spec = evaluator.add_leaf(
        id="Saw_Specified",
        desc="A primary saw/cutting tool is specified for cutting lumber.",
        parent=cutting_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies a primary saw/cutting tool for cutting lumber, such as a table saw, circular saw, miter saw, track saw, or jigsaw.",
        node=leaf_saw_spec,
        additional_instruction="Accept any reasonable primary cutting tool suitable for this project."
    )

    uses_table_saw = bool(plan.tools and plan.tools.uses_table_saw)
    if uses_table_saw:
        # OSHA-required safety features for table saw
        ts_safety_node = evaluator.add_parallel(
            id="Table_Saw_OSHA_Safety_Features_If_Applicable",
            desc="If a table saw is specified, it includes blade guard and a riving knife or splitter (per OSHA 1910.213 constraint).",
            parent=cutting_node,
            critical=False
        )

        leaf_guard = evaluator.add_leaf(
            id="Blade_Guard",
            desc="Blade guard is included if a table saw is chosen.",
            parent=ts_safety_node,
            critical=True
        )
        await evaluator.verify(
            claim="The plan includes a blade guard for the table saw.",
            node=leaf_guard,
            additional_instruction="Check for explicit 'blade guard' on table saw; synonyms like 'guard' attached to blade are acceptable."
        )

        leaf_riving = evaluator.add_leaf(
            id="Riving_Knife_Or_Splitter",
            desc="Riving knife or splitter is included if a table saw is chosen.",
            parent=ts_safety_node,
            critical=True
        )
        await evaluator.verify(
            claim="The plan includes a riving knife or splitter for the table saw.",
            node=leaf_riving,
            additional_instruction="Accept 'riving knife' or 'splitter' as anti-kickback device in line with OSHA 1910.213."
        )

    # Power Drill
    leaf_drill = evaluator.add_leaf(
        id="Power_Drill",
        desc="Power drill/driver is specified.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies a power drill or drill/driver.",
        node=leaf_drill,
        additional_instruction="Accept cordless or corded drill/driver."
    )

    # Sander
    leaf_sander = evaluator.add_leaf(
        id="Sander",
        desc="Random orbital sander (or equivalent) is specified.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies a random orbital sander or an equivalent sander for surface preparation.",
        node=leaf_sander,
        additional_instruction="Accept 'random orbital sander', 'ROS', or equivalent sanding tool."
    )

    # Clamps for Panel Glue-ups
    clamp_node = evaluator.add_parallel(
        id="Clamps_For_Panel_Glueups",
        desc="Specifies clamp type and minimum quantity required for panel glue-ups.",
        parent=node,
        critical=False
    )

    leaf_clamp_qty = evaluator.add_leaf(
        id="Clamp_Quantity_Minimum",
        desc="Specifies at least 4 clamps for panel glue-ups.",
        parent=clamp_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies at least four clamps for panel glue-ups.",
        node=leaf_clamp_qty,
        additional_instruction="Accept phrasing like 'at least 4 clamps', '≥4 clamps', 'minimum four clamps'."
    )

    leaf_clamp_type = evaluator.add_leaf(
        id="Clamp_Type",
        desc="Specifies an appropriate clamp type for panel glue-ups (e.g., bar clamps or parallel clamps).",
        parent=clamp_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies an appropriate clamp type for panel glue-ups, such as bar clamps, parallel clamps, or pipe clamps.",
        node=leaf_clamp_type,
        additional_instruction="Accept 'bar clamps', 'parallel clamps', 'pipe clamps'; reject spring clamps for panel glue-ups."
    )

    # Work Surface
    leaf_work_surface = evaluator.add_leaf(
        id="Work_Surface",
        desc="Workbench or stable work surface is specified.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies a workbench or stable work surface.",
        node=leaf_work_surface,
        additional_instruction="Accept any sturdy flat work surface suitable for clamping and assembly."
    )


async def build_lumber_nodes(evaluator: Evaluator, parent, plan: PlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Lumber_Selection_And_Quantity",
        desc="Specifies appropriate lumber selection and calculates quantity per constraints.",
        parent=parent,
        critical=False
    )

    leaf_species = evaluator.add_leaf(
        id="Hardwood_Species_Selected",
        desc="Specifies a hardwood species suitable for furniture making.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies a hardwood species suitable for furniture making (e.g., oak, maple, walnut, or cherry).",
        node=leaf_species,
        additional_instruction="Accept any common furniture hardwood; reject softwoods unless clearly specified as hardwood."
    )

    leaf_quality = evaluator.add_leaf(
        id="Furniture_Grade_Quality_Specified",
        desc="Specifies furniture-grade quality requirements/specs (property-based quality specification).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies furniture-grade quality requirements (e.g., FAS, Select & Better, S4S, quarter-sawn).",
        node=leaf_quality,
        additional_instruction="Look for explicit grading/spec quality terms suitable for furniture."
    )

    leaf_kiln = evaluator.add_leaf(
        id="Kiln_Dried_Required",
        desc="Specifies the lumber must be kiln-dried (not air-dried/green).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan requires kiln-dried lumber rather than green or solely air-dried stock.",
        node=leaf_kiln,
        additional_instruction="Accept 'kiln-dried' explicitly; mentioning green lumber avoidance is supportive."
    )

    leaf_mc_range = evaluator.add_leaf(
        id="Moisture_Content_Range",
        desc="Specifies the moisture content range per constraints (acceptable 6–11%, ideally 6–8%).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies an acceptable moisture content range around 6–11%, ideally 6–8%.",
        node=leaf_mc_range,
        additional_instruction="Allow minor wording variations (e.g., 'about 6 to 11 percent')."
    )

    leaf_mc_meter = evaluator.add_leaf(
        id="Moisture_Meter_Use",
        desc="Specifies using a moisture meter to verify moisture content.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies using a moisture meter to verify lumber moisture content.",
        node=leaf_mc_meter,
        additional_instruction="Accept any explicit instruction to measure moisture with a meter."
    )

    leaf_bf_formula = evaluator.add_leaf(
        id="Board_Feet_Formula_Used",
        desc="Uses the board-feet formula: Length(ft) × Width(in) × Thickness(in) ÷ 12.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan uses or states the board-feet formula: Length(ft) × Width(in) × Thickness(in) ÷ 12.",
        node=leaf_bf_formula,
        additional_instruction="Minor formatting variations are acceptable; formula must be clearly present."
    )

    leaf_waste = evaluator.add_leaf(
        id="Waste_Allowance_Included",
        desc="Includes 20–30% waste allowance in lumber quantity planning.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan includes a waste allowance of approximately 20–30% in lumber quantity planning.",
        node=leaf_waste,
        additional_instruction="Accept 'around 20–30%', 'add 25% waste', or similar phrasing."
    )


async def build_finish_nodes(evaluator: Evaluator, parent, plan: PlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Food_Safe_Finishing",
        desc="Selects a finish that is food-safe when fully cured, durable enough for table use, and includes application method.",
        parent=parent,
        critical=False
    )

    leaf_type = evaluator.add_leaf(
        id="Finish_Type_Specified",
        desc="Specifies a finish type (e.g., oil, film finish, etc.).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies a finish type (e.g., oil finish, polyurethane film finish, or hardwax oil).",
        node=leaf_type,
        additional_instruction="Accept common finish types suitable for dining tables."
    )

    leaf_foodsafe = evaluator.add_leaf(
        id="Food_Safe_When_Fully_Cured",
        desc="Explicitly states the finish is food-safe when fully cured (and/or provides a product spec reference supporting this).",
        parent=node,
        critical=True
    )
    finish_urls = plan.docs.finish_product_spec_urls if plan.docs else []
    await evaluator.verify(
        claim="The chosen finish is food-safe when fully cured.",
        node=leaf_foodsafe,
        sources=finish_urls if finish_urls else None,
        additional_instruction="If product spec URLs are provided, confirm the food-safe claim from the product page; otherwise verify the statement within the plan."
    )

    leaf_cure = evaluator.add_leaf(
        id="Cure_Before_Food_Contact",
        desc="Specifies that the finish must be fully cured before food contact (especially for film finishes such as polyurethane).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies that the finish must be fully cured before food contact.",
        node=leaf_cure,
        additional_instruction="Accept explicit cure-time instruction before food contact; applicable particularly to film finishes."
    )

    leaf_durability = evaluator.add_leaf(
        id="Durability_For_Dining_Table_Use",
        desc="Addresses that the finish provides adequate durability/protection for dining table use.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan addresses that the finish provides adequate durability/protection for dining table use.",
        node=leaf_durability,
        additional_instruction="Look for 'durable', 'protective film', 'resists spills/scratches', etc."
    )

    leaf_application = evaluator.add_leaf(
        id="Application_Method",
        desc="Specifies an application method for the chosen finish.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies an application method for the finish (e.g., wipe-on, brush, or spray).",
        node=leaf_application,
        additional_instruction="Accept any reasonable application method explicitly stated."
    )


async def build_workshop_nodes(evaluator: Evaluator, parent, plan: PlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Workshop_Requirements",
        desc="Addresses minimum workshop space and dust collection CFM requirements per constraints.",
        parent=parent,
        critical=False
    )

    leaf_min_sq = evaluator.add_leaf(
        id="Minimum_Square_Footage",
        desc="States the workshop meets a minimum of 100 square feet.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan states that the workshop meets a minimum of 100 square feet.",
        node=leaf_min_sq,
        additional_instruction="Accept phrasing like '≥100 sq ft', 'at least 100 square feet'."
    )

    leaf_ideal_sq = evaluator.add_leaf(
        id="Ideal_Square_Footage_Recommendation",
        desc="Mentions that ~200+ square feet is ideal (recommendation).",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="The plan mentions that approximately 200+ square feet is ideal for a home workshop.",
        node=leaf_ideal_sq,
        additional_instruction="This is a recommendation; allow '~200 sq ft', '200 or more square feet'."
    )

    dust_node = evaluator.add_parallel(
        id="Dust_Collection_CFM",
        desc="Specifies a dust collection system and its required CFM for the tools being used, including table saw CFM if applicable.",
        parent=node,
        critical=False
    )

    leaf_cfm_general = evaluator.add_leaf(
        id="General_CFM_Range",
        desc="Addresses the 250–1000 CFM tool-dependent requirement.",
        parent=dust_node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan addresses a dust collection CFM requirement in the approximate 250–1000 CFM range depending on the tool.",
        node=leaf_cfm_general,
        additional_instruction="Accept any reasonable statement indicating tool-dependent CFM needs within roughly 250–1000 CFM."
    )

    uses_table_saw = bool(plan.tools and plan.tools.uses_table_saw)
    if uses_table_saw:
        leaf_cfm_tsaw = evaluator.add_leaf(
            id="Table_Saw_CFM_If_Applicable",
            desc="If a table saw is used, addresses ~500–600 CFM at the cabinet.",
            parent=dust_node,
            critical=True
        )
        await evaluator.verify(
            claim="For the table saw, the plan addresses approximately 500–600 CFM at the cabinet.",
            node=leaf_cfm_tsaw,
            additional_instruction="Accept phrasing like '500 CFM', 'around 600 CFM', '500–600 CFM at the cabinet'."
        )

    leaf_fpm = evaluator.add_leaf(
        id="Air_Velocity_Recommendation",
        desc="Mentions the 4,000 FPM air-velocity recommendation for dust transport.",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="The plan mentions the 4,000 FPM air-velocity recommendation for effective dust transport.",
        node=leaf_fpm,
        additional_instruction="Accept phrasing like 'around 4000 feet per minute', '4,000 FPM'."
    )


async def build_docs_nodes(evaluator: Evaluator, parent, plan: PlanExtraction) -> None:
    node = evaluator.add_parallel(
        id="Documentation_And_References",
        desc="Provides required and optional references as specified.",
        parent=parent,
        critical=False
    )

    # Required OSHA reference: existence check (critical)
    osha_urls = plan.docs.osha_urls if plan.docs else []
    exists_osha = bool(osha_urls and len(osha_urls) > 0)
    evaluator.add_custom_node(
        result=exists_osha,
        id="OSHA_Reference_Exists",
        desc="At least one OSHA woodworking safety standards reference URL is provided.",
        parent=node,
        critical=True
    )

    # Verify the OSHA page(s) content (critical)
    leaf_osha = evaluator.add_leaf(
        id="OSHA_Woodworking_Safety_Reference",
        desc="Provides a reference to OSHA woodworking safety standards.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is an OSHA page or OSHA standard relevant to woodworking machinery/safety (e.g., OSHA 1910.213) and supports woodworking safety requirements.",
        node=leaf_osha,
        sources=osha_urls if osha_urls else None,
        additional_instruction="Prefer osha.gov domain and sections such as 29 CFR 1910.213 for woodworking machinery; confirm relevance to woodworking safety."
    )

    # Optional references (add only if present; non-critical)
    lumber_urls = plan.docs.lumber_source_urls if plan.docs else []
    if lumber_urls:
        leaf_lumber_ref = evaluator.add_leaf(
            id="Lumber_Source_Reference_Optional",
            desc="Provides sources for furniture-grade hardwood lumber (optional).",
            parent=node,
            critical=False
        )
        await evaluator.verify(
            claim="This webpage is a source for furniture-grade hardwood lumber.",
            node=leaf_lumber_ref,
            sources=lumber_urls,
            additional_instruction="Accept reputable lumber suppliers or sources specifically offering furniture-grade hardwood."
        )

    beginner_urls = plan.docs.beginner_resources_urls if plan.docs else []
    if beginner_urls:
        leaf_beginner_ref = evaluator.add_leaf(
            id="Beginner_Instruction_Reference_Optional",
            desc="Provides beginner woodworking instruction resources (optional).",
            parent=node,
            critical=False
        )
        await evaluator.verify(
            claim="This webpage provides beginner woodworking instruction resources/tutorials.",
            node=leaf_beginner_ref,
            sources=beginner_urls,
            additional_instruction="Accept basic woodworking tutorials from reputable sources suitable for beginners."
        )

    finish_urls = plan.docs.finish_product_spec_urls if plan.docs else []
    if finish_urls:
        leaf_finish_spec_ref = evaluator.add_leaf(
            id="Finish_Product_Spec_Reference_Optional",
            desc="Provides food-safe finishing product specifications (optional).",
            parent=node,
            critical=False
        )
        await evaluator.verify(
            claim="This webpage provides a product specification stating the finish is food-safe when fully cured and suitable for table use.",
            node=leaf_finish_spec_ref,
            sources=finish_urls,
            additional_instruction="Confirm food-safe when cured and suitability/durability for table tops in the product documentation."
        )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate the beginner dining table project plan answer against the rubric.
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

    # Extract structured plan info
    plan = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction",
    )

    # Add a top-level plan node (non-critical to allow partial scoring with optional items)
    plan_root = evaluator.add_parallel(
        id="Complete_Beginner_Dining_Table_Project_Planning",
        desc="Comprehensive project plan covering safety equipment, essential tools, lumber selection and quantity, food-safe finishing, workshop requirements, and documentation references.",
        parent=root,
        critical=False
    )

    # Build subtrees
    await build_safety_nodes(evaluator, plan_root, plan)
    await build_tools_nodes(evaluator, plan_root, plan)
    await build_lumber_nodes(evaluator, plan_root, plan)
    await build_finish_nodes(evaluator, plan_root, plan)
    await build_workshop_nodes(evaluator, plan_root, plan)
    await build_docs_nodes(evaluator, plan_root, plan)

    # Record helpful custom info
    evaluator.add_custom_info(
        info={
            "uses_table_saw": bool(plan.tools and plan.tools.uses_table_saw),
            "primary_cutting_tool": plan.tools.primary_cutting_tool if plan.tools and plan.tools.primary_cutting_tool else None,
            "finish_type": plan.finish.finish_type if plan.finish and plan.finish.finish_type else None,
            "osha_urls_count": len(plan.docs.osha_urls) if plan.docs and plan.docs.osha_urls else 0,
        },
        info_type="plan_summary",
        info_name="extraction_summary"
    )

    return evaluator.get_summary()