import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "diy_table_project_plan"
TASK_DESCRIPTION = "Comprehensive beginner DIY dining table project plan covering required safety items, essential tools, material specifications, and finishing process requirements per the question/constraints."


# =========================
# Step 2: Extraction Models
# =========================

class SafetyExtraction(BaseModel):
    eye_meets_ansi_z87: Optional[bool] = None
    hearing_protection_for_85dB_plus: Optional[bool] = None
    respirator_is_n95_or_better: Optional[bool] = None
    fire_extinguisher_abc_included: Optional[bool] = None
    first_aid_kit_included: Optional[bool] = None


class ToolsExtraction(BaseModel):
    measuring_tools: List[str] = Field(default_factory=list)
    cutting_tools: List[str] = Field(default_factory=list)
    fastening_tools: List[str] = Field(default_factory=list)
    finishing_tools: List[str] = Field(default_factory=list)
    has_tape_measure: Optional[bool] = None
    has_combination_square: Optional[bool] = None
    has_marking_knife_or_pencil: Optional[bool] = None
    has_circular_or_table_saw: Optional[bool] = None
    has_jigsaw: Optional[bool] = None
    has_power_drill_driver: Optional[bool] = None
    has_clamps: Optional[bool] = None
    has_random_orbital_sander: Optional[bool] = None


class MaterialsExtraction(BaseModel):
    selected_wood_species: List[str] = Field(default_factory=list)
    wood_species_within_allowed_set: Optional[bool] = None
    moisture_content_mentions_6_to_12_percent: Optional[bool] = None
    acknowledges_nominal_vs_actual: Optional[bool] = None
    apron_thickness_at_least_0_75_in: Optional[bool] = None
    apron_width_at_least_4_in: Optional[bool] = None


class FinishingExtraction(BaseModel):
    sanding_progression_list: List[str] = Field(default_factory=list)
    progression_never_skips_more_than_one: Optional[bool] = None
    final_grit_180_for_clear: Optional[bool] = None
    final_grit_220_for_stain: Optional[bool] = None
    pre_stain_conditioner_for_softwoods: Optional[bool] = None
    protective_finish_min_3_coats: Optional[bool] = None
    water_based_poly_dry_time_included_2_to_4h: Optional[bool] = None
    oil_based_poly_dry_time_included_8_to_24h: Optional[bool] = None


class GlueExtraction(BaseModel):
    pva_min_clamp_time_20_to_30_min: Optional[bool] = None


# ==========================
# Step 3: Extraction Prompts
# ==========================

def prompt_extract_safety() -> str:
    return (
        "From the answer, extract whether the plan explicitly includes each of the following workshop safety items:\n"
        "- Eye protection that meets ANSI Z87.1 (return true if explicitly mentions ANSI Z87.1 or Z87+).\n"
        "- Hearing protection for power-tool noise above 85 dB (earmuffs or earplugs).\n"
        "- Respiratory protection rated at least N95 (≥95% filtration); return true also if P100/half-mask with P100 is specified.\n"
        "- ABC-type fire extinguisher (Class ABC).\n"
        "- A first aid kit available in the workshop.\n"
        "Return booleans for: eye_meets_ansi_z87, hearing_protection_for_85dB_plus, respirator_is_n95_or_better, "
        "fire_extinguisher_abc_included, first_aid_kit_included."
    )


def prompt_extract_tools() -> str:
    return (
        "From the answer, extract the listed tools grouped into categories (measuring_tools, cutting_tools, fastening_tools, finishing_tools). "
        "Also compute boolean flags indicating if the plan includes:\n"
        "- has_tape_measure\n"
        "- has_combination_square\n"
        "- has_marking_knife_or_pencil (true if either a marking knife or a pencil is listed)\n"
        "- has_circular_or_table_saw (true if either circular saw or table saw is listed)\n"
        "- has_jigsaw\n"
        "- has_power_drill_driver\n"
        "- has_clamps\n"
        "- has_random_orbital_sander\n"
        "Return the lists and these boolean flags."
    )


def prompt_extract_materials() -> str:
    return (
        "From the answer, extract material-related statements for an indoor wooden dining table. "
        "Return:\n"
        "- selected_wood_species: the wood species explicitly recommended/selected.\n"
        "- wood_species_within_allowed_set: true if all selected species are within {Oak, Maple, Walnut, Cherry, Pine, Poplar, Soft Maple}; false otherwise.\n"
        "- moisture_content_mentions_6_to_12_percent: true if the plan specifies wood MC (moisture content) between 6% and 12%.\n"
        "- acknowledges_nominal_vs_actual: true if the plan acknowledges nominal vs actual lumber dimensions.\n"
        "- apron_thickness_at_least_0_75_in: true if apron thickness is stated as ≥ 3/4 inch.\n"
        "- apron_width_at_least_4_in: true if apron width is stated as ≥ 4 inches."
    )


def prompt_extract_finishing() -> str:
    return (
        "From the answer, extract finishing process details. Return:\n"
        "- sanding_progression_list: the sequence of sanding grits (e.g., ['80','120','150','180']).\n"
        "- progression_never_skips_more_than_one: true if the plan never skips more than one grit level between steps.\n"
        "- final_grit_180_for_clear: true if final sanding grit is 180 when applying a clear finish.\n"
        "- final_grit_220_for_stain: true if final sanding grit is 220 when staining.\n"
        "- pre_stain_conditioner_for_softwoods: true if it states a pre-stain conditioner is required for softwoods to prevent blotching.\n"
        "- protective_finish_min_3_coats: true if the plan specifies at least 3 coats of a protective finish.\n"
        "- water_based_poly_dry_time_included_2_to_4h: true if the plan includes dry/recoat times 2–4 hours for water-based polyurethane.\n"
        "- oil_based_poly_dry_time_included_8_to_24h: true if the plan includes dry/recoat times 8–24 hours for oil-based polyurethane."
    )


def prompt_extract_glue() -> str:
    return (
        "From the answer, extract whether the plan specifies a PVA wood glue minimum clamp time of 20–30 minutes. "
        "Return boolean pva_min_clamp_time_20_to_30_min."
    )


# =================================
# Step 4: Implement Verification Logic
# =================================

async def add_safety_checks(evaluator: Evaluator, parent_node) -> None:
    safety_node = evaluator.add_parallel(
        id="Safety_Equipment_Requirements",
        desc="All required personal protective equipment and safety items for the workshop.",
        parent=parent_node,
        critical=True
    )

    tasks: List[tuple] = []

    eye_node = evaluator.add_leaf(
        id="Eye_Protection",
        desc="Eye protection specified that meets ANSI Z87.1 certification standards.",
        parent=safety_node,
        critical=True
    )
    tasks.append((
        "The plan specifies eye protection that meets ANSI Z87.1 (or Z87+) certification standards.",
        None,
        eye_node,
        "Accept 'ANSI Z87.1', 'ANSI Z87', or 'Z87+' as compliant safety glasses/goggles. Minor wording variations are acceptable."
    ))

    hearing_node = evaluator.add_leaf(
        id="Hearing_Protection",
        desc="Hearing protection specified for power-tool noise levels above 85 dB.",
        parent=safety_node,
        critical=True
    )
    tasks.append((
        "The plan specifies hearing protection suitable for power tool noise above 85 dB (earmuffs or earplugs).",
        None,
        hearing_node,
        "Look for earmuffs, earplugs, or any explicit mention of protection for >85 dB tools."
    ))

    respiratory_node = evaluator.add_leaf(
        id="Respiratory_Protection",
        desc="Respiratory protection specified that is rated at least N95 (≥95% filtration efficiency) for wood dust.",
        parent=safety_node,
        critical=True
    )
    tasks.append((
        "The plan specifies respiratory protection rated at least N95 (≥95% filtration) for wood dust.",
        None,
        respiratory_node,
        "Accept 'N95', 'P100', or half-mask respirator with P100 filters as meeting or exceeding the requirement."
    ))

    fire_node = evaluator.add_leaf(
        id="Fire_Safety_Equipment",
        desc="ABC-type fire extinguisher included (suitable for wood, flammable liquids, and electrical fires).",
        parent=safety_node,
        critical=True
    )
    tasks.append((
        "The plan includes an ABC-type fire extinguisher.",
        None,
        fire_node,
        "Accept 'Class ABC', 'ABC-rated', or 'multi-purpose dry chemical' extinguisher as valid."
    ))

    first_aid_node = evaluator.add_leaf(
        id="First_Aid_Kit",
        desc="A first aid kit is specified as readily available in the workshop.",
        parent=safety_node,
        critical=True
    )
    tasks.append((
        "The plan specifies that a first aid kit is readily available in the workshop.",
        None,
        first_aid_node,
        "Any explicit mention of a first aid kit counts."
    ))

    await evaluator.batch_verify(tasks)


async def add_tools_checks(evaluator: Evaluator, parent_node) -> None:
    tools_node = evaluator.add_parallel(
        id="Essential_Tools",
        desc="Minimum essential tools are specified across the four required categories.",
        parent=parent_node,
        critical=True
    )

    tasks: List[tuple] = []

    meas_node = evaluator.add_leaf(
        id="Measuring_and_Marking_Tools",
        desc="Includes tape measure, combination square, and marking knife or pencil.",
        parent=tools_node,
        critical=True
    )
    tasks.append((
        "The plan includes a tape measure, a combination square, and a marking knife or pencil for measuring and marking.",
        None,
        meas_node,
        "Accept pencil or marking knife as fulfilling the marking requirement."
    ))

    cutting_node = evaluator.add_leaf(
        id="Cutting_Tools",
        desc="Includes circular saw or table saw, and jigsaw.",
        parent=tools_node,
        critical=True
    )
    tasks.append((
        "The plan includes at least one of a circular saw or a table saw, and also includes a jigsaw.",
        None,
        cutting_node,
        "Both parts are required: (1) circular or table saw; (2) jigsaw."
    ))

    fastening_node = evaluator.add_leaf(
        id="Fastening_and_Assembly_Tools",
        desc="Includes power drill/driver and clamps.",
        parent=tools_node,
        critical=True
    )
    tasks.append((
        "The plan includes a power drill/driver and clamps for fastening and assembly.",
        None,
        fastening_node,
        "Clamps can be any suitable woodworking clamps; drill/driver can be corded or cordless."
    ))

    finishing_node = evaluator.add_leaf(
        id="Finishing_Tools",
        desc="Includes a random orbital sander.",
        parent=tools_node,
        critical=True
    )
    tasks.append((
        "The plan includes a random orbital sander for finishing.",
        None,
        finishing_node,
        "Accept 'random orbital sander' or 'ROS'."
    ))

    await evaluator.batch_verify(tasks)


async def add_material_checks(evaluator: Evaluator, parent_node) -> None:
    materials_node = evaluator.add_parallel(
        id="Material_Specifications",
        desc="Required material/wood specifications for an indoor dining table are provided.",
        parent=parent_node,
        critical=True
    )

    tasks: List[tuple] = []

    wood_type_node = evaluator.add_leaf(
        id="Wood_Type_Selection",
        desc="Wood species selection is from the allowed set (Oak, Maple, Walnut, Cherry, Pine, Poplar, Soft Maple).",
        parent=materials_node,
        critical=True
    )
    tasks.append((
        "The plan selects wood species only from the allowed set: Oak, Maple, Walnut, Cherry, Pine, Poplar, or Soft Maple.",
        None,
        wood_type_node,
        "At least one allowed species must be explicitly recommended/selected. Do not accept species outside the list as the primary choice."
    ))

    mc_node = evaluator.add_leaf(
        id="Moisture_Content",
        desc="Wood moisture content is verified/specified as suitable for indoor furniture: 6–12%.",
        parent=materials_node,
        critical=True
    )
    tasks.append((
        "The plan specifies checking or verifying wood moisture content to be between 6% and 12% before construction.",
        None,
        mc_node,
        "Exact numeric range must be within 6–12% (inclusive)."
    ))

    dimensions_node = evaluator.add_leaf(
        id="Lumber_Dimensions",
        desc="Acknowledges nominal vs. actual lumber dimensions (i.e., nominal sizes differ from actual measurements).",
        parent=materials_node,
        critical=True
    )
    tasks.append((
        "The plan explicitly acknowledges that nominal lumber sizes differ from actual measurements.",
        None,
        dimensions_node,
        "Any clear statement about nominal vs. actual sizes is acceptable."
    ))

    apron_node = evaluator.add_leaf(
        id="Table_Apron_Specifications",
        desc="Table apron is specified as at least 3/4 inch thick and 4 inches wide.",
        parent=materials_node,
        critical=True
    )
    tasks.append((
        "The plan specifies a table apron at least 3/4 inch thick and at least 4 inches wide.",
        None,
        apron_node,
        "Both constraints (thickness ≥ 3/4 in and width ≥ 4 in) must be present."
    ))

    await evaluator.batch_verify(tasks)


async def add_finishing_checks(evaluator: Evaluator, parent_node) -> None:
    finishing_node = evaluator.add_parallel(
        id="Finishing_Process",
        desc="Complete finishing process plan including sanding, any required pre-finish preparation, and protective finish application.",
        parent=parent_node,
        critical=True
    )

    tasks: List[tuple] = []

    sanding_node = evaluator.add_leaf(
        id="Sanding_Progression_and_Final_Grit",
        desc="Specifies sanding grit progression that never skips more than one grit level, and specifies final sanding grit as 180 for clear finish or 220 for stain.",
        parent=finishing_node,
        critical=True
    )
    tasks.append((
        "The plan specifies a sanding grit progression that does not skip more than one grit level and specifies final sanding grit as 180 for clear finishes and 220 for stained finishes.",
        None,
        sanding_node,
        "Accept reasonable sequences like 80→120→150→180 or 120→150→180. For staining, final 220 must be stated."
    ))

    prefinish_node = evaluator.add_leaf(
        id="Pre_Finish_Preparation_for_Softwoods",
        desc="States that pre-stain conditioner is required for softwoods to prevent blotching when staining.",
        parent=finishing_node,
        critical=True
    )
    tasks.append((
        "The plan states that a pre-stain conditioner is required for softwoods to prevent blotching when staining.",
        None,
        prefinish_node,
        "Softwoods include pine, fir, spruce; accept equivalent phrasing like 'use wood conditioner on softwoods before stain'."
    ))

    coats_dry_node = evaluator.add_leaf(
        id="Protective_Finish_Coats_and_Dry_Times",
        desc="Specifies applying a protective finish with at least 3 coats and includes appropriate drying times between coats (water-based polyurethane 2–4 hours; oil-based polyurethane 8–24 hours).",
        parent=finishing_node,
        critical=True
    )
    tasks.append((
        "The plan specifies applying a protective finish with at least 3 coats and includes drying times between coats: water-based polyurethane 2–4 hours; oil-based polyurethane 8–24 hours.",
        None,
        coats_dry_node,
        "All elements must be present: (1) ≥3 coats; (2) water-based dry time 2–4h; (3) oil-based dry time 8–24h."
    ))

    await evaluator.batch_verify(tasks)


async def add_glue_checks(evaluator: Evaluator, parent_node) -> None:
    glue_node = evaluator.add_parallel(
        id="Glue_Up_Clamp_Time",
        desc="Includes required glue-up constraint for assembly if using PVA wood glue.",
        parent=parent_node,
        critical=True
    )

    pva_node = evaluator.add_leaf(
        id="PVA_Glue_Min_Clamp_Time",
        desc="Specifies PVA wood glue minimum clamp time: 20–30 minutes minimum.",
        parent=glue_node,
        critical=True
    )

    await evaluator.verify(
        claim="The plan specifies that PVA wood glue requires a minimum clamp time of 20–30 minutes.",
        node=pva_node,
        sources=None,
        additional_instruction="Accept '20 to 30 minutes' or equivalent phrasing like 'at least 20 minutes, up to 30 minutes'."
    )


# ======================================
# Step 5: Main Evaluation Entry Function
# ======================================

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
        strategy=AggregationStrategy.PARALLEL,
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

    # Run extractions in parallel (for record/analysis; verification relies on LLM checks directly)
    safety_task = evaluator.extract(
        prompt=prompt_extract_safety(),
        template_class=SafetyExtraction,
        extraction_name="safety_extraction"
    )
    tools_task = evaluator.extract(
        prompt=prompt_extract_tools(),
        template_class=ToolsExtraction,
        extraction_name="tools_extraction"
    )
    materials_task = evaluator.extract(
        prompt=prompt_extract_materials(),
        template_class=MaterialsExtraction,
        extraction_name="materials_extraction"
    )
    finishing_task = evaluator.extract(
        prompt=prompt_extract_finishing(),
        template_class=FinishingExtraction,
        extraction_name="finishing_extraction"
    )
    glue_task = evaluator.extract(
        prompt=prompt_extract_glue(),
        template_class=GlueExtraction,
        extraction_name="glue_extraction"
    )

    await asyncio.gather(safety_task, tools_task, materials_task, finishing_task, glue_task)

    # Build critical plan root (since the rubric's top-level node is critical)
    plan_root = evaluator.add_parallel(
        id="Complete_DIY_Table_Project_Plan",
        desc="Comprehensive beginner DIY dining table project plan covering required safety items, essential tools, material specifications, and finishing process requirements per the question/constraints.",
        parent=root,
        critical=True
    )

    # Add verification subtrees
    await add_safety_checks(evaluator, plan_root)
    await add_tools_checks(evaluator, plan_root)
    await add_material_checks(evaluator, plan_root)
    await add_finishing_checks(evaluator, plan_root)
    await add_glue_checks(evaluator, plan_root)

    # Add reference expectations as custom info
    evaluator.add_custom_info(
        info={
            "allowed_wood_species": ["Oak", "Maple", "Walnut", "Cherry", "Pine", "Poplar", "Soft Maple"],
            "moisture_content_range_percent": "6–12",
            "apron_min_thickness_in": "0.75",
            "apron_min_width_in": "4",
            "sanding_final_grit_clear": "180",
            "sanding_final_grit_stain": "220",
            "water_based_poly_dry_time_hours": "2–4",
            "oil_based_poly_dry_time_hours": "8–24",
            "pva_glue_min_clamp_time_minutes": "20–30"
        },
        info_type="requirements_reference"
    )

    return evaluator.get_summary()