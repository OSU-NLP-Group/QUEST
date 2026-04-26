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
TASK_ID = "woodworking_nhla_safety"
TASK_DESCRIPTION = (
    "You are planning to build indoor fine furniture and need to source appropriate hardwood lumber. "
    "Research the NHLA (National Hardwood Lumber Association) grading standards for hardwood lumber and woodworking safety requirements. "
    "Provide the following information: "
    "(1) What is the name of the NHLA lumber grade that represents the best/highest quality hardwood lumber? "
    "(2) What are the minimum board dimensions (width in inches × length in feet) required for this grade? "
    "(3) What is the minimum percentage of clear wood yield required for this grade? "
    "(4) What are the two acceptable minimum clear cutting dimension options (width × length) for this grade? "
    "(5) What is the acceptable moisture content percentage range for fine furniture making and indoor woodworking? "
    "(6) What are the three required types of personal protective equipment (PPE) for woodworking operations with power tools?"
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class GradeSpecs(BaseModel):
    top_grade_name: Optional[str] = None
    min_board_dimensions: Optional[str] = None
    min_clear_yield_percent: Optional[str] = None
    clear_cutting_options: List[str] = Field(default_factory=list)


class GeneralRequirements(BaseModel):
    moisture_content_range: Optional[str] = None
    ppe_eye: Optional[str] = None
    ppe_respiratory: Optional[str] = None
    ppe_hearing: Optional[str] = None


class WoodworkingExtraction(BaseModel):
    grade_specs: Optional[GradeSpecs] = None
    general: Optional[GeneralRequirements] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_woodworking_info() -> str:
    return (
        "Extract the requested information from the answer text exactly as stated, without adding anything:\n"
        "Section A: NHLA Top Grade and Technical Specs:\n"
        "- top_grade_name: The NHLA lumber grade name identified as the best/highest quality (e.g., 'FAS', 'FAS 1-Face', 'Firsts and Seconds').\n"
        "- min_board_dimensions: The minimum board dimensions required for this grade, formatted as width in inches × length in feet "
        "(preserve the answer's formatting, e.g., '6 in × 8 ft', '6\" x 8'').\n"
        "- min_clear_yield_percent: The minimum clear wood yield percentage (preserve formatting, e.g., '83-1/3%', '83.3%', '83%').\n"
        "- clear_cutting_options: An array of up to two strings for the acceptable minimum clear cutting dimension options (width × length), "
        "in the exact wording from the answer. If more than two are listed, include only the first two. If none are listed, return an empty array.\n"
        "\n"
        "Section B: General Woodworking Requirements:\n"
        "- moisture_content_range: The acceptable moisture content percentage range for fine furniture/indoor woodworking (preserve formatting, e.g., '6–8%').\n"
        "- ppe_eye: The text describing the required eye protection (e.g., 'safety glasses', 'goggles').\n"
        "- ppe_respiratory: The text describing the required respiratory protection (e.g., 'dust mask', 'respirator').\n"
        "- ppe_hearing: The text describing the required hearing protection (e.g., 'earmuffs', 'earplugs').\n"
        "\n"
        "Return a JSON object with keys 'grade_specs' and 'general' following the provided schema. "
        "If any item is not present in the answer, return null for that field or an empty array as appropriate."
    )


# --------------------------------------------------------------------------- #
# Verification tree construction functions                                    #
# --------------------------------------------------------------------------- #
async def build_lumber_grade_spec(
    evaluator: Evaluator,
    parent_node,
    extraction: WoodworkingExtraction,
) -> None:
    # Create the critical sequential node for Lumber Grade Specifications
    lgs_node = evaluator.add_sequential(
        id="LumberGradeSpecifications",
        desc="Identify the best NHLA hardwood lumber grade and provide its required technical specifications",
        parent=parent_node,
        critical=True,
    )

    grade_name = extraction.grade_specs.top_grade_name if extraction.grade_specs else None
    min_dims = extraction.grade_specs.min_board_dimensions if extraction.grade_specs else None
    min_yield = extraction.grade_specs.min_clear_yield_percent if extraction.grade_specs else None
    cutting_opts = extraction.grade_specs.clear_cutting_options if (extraction.grade_specs and extraction.grade_specs.clear_cutting_options) else []

    # Top Grade Identification (critical leaf)
    top_grade_leaf = evaluator.add_leaf(
        id="TopGradeIdentification",
        desc="Identify the NHLA lumber grade that represents the best/highest quality hardwood lumber",
        parent=lgs_node,
        critical=True,
    )
    claim_top_grade = (
        f"According to the answer, the NHLA lumber grade representing the best/highest quality hardwood lumber is '{grade_name}'."
    )
    await evaluator.verify(
        claim=claim_top_grade,
        node=top_grade_leaf,
        additional_instruction=(
            "Only judge whether the answer explicitly names a grade as the best/highest quality. "
            "Accept typical NHLA top-grade names like 'FAS', 'Firsts and Seconds', 'FAS 1-Face'. "
            "This check is based solely on the answer text, not external correctness."
        ),
    )

    # Technical Specifications (parallel critical container)
    specs_node = evaluator.add_parallel(
        id="TopGradeTechnicalSpecifications",
        desc="Provide the required technical specifications for the identified top grade",
        parent=lgs_node,
        critical=True,
    )

    # Minimum Board Dimensions (critical leaf)
    min_dims_leaf = evaluator.add_leaf(
        id="MinimumBoardDimensions",
        desc="Provide the minimum board dimensions (width in inches × length in feet) required for this grade",
        parent=specs_node,
        critical=True,
    )
    claim_min_dims = (
        f"The answer provides the minimum board dimensions for the grade '{grade_name}' as '{min_dims}'."
    )
    await evaluator.verify(
        claim=claim_min_dims,
        node=min_dims_leaf,
        additional_instruction=(
            "Judge whether the answer provides minimum board dimensions for the specified grade. "
            "Allow formatting variants such as 'x' vs '×', quotes, or words ('in', 'inches', 'ft', 'feet'). "
            "This check is limited to the answer text."
        ),
    )

    # Minimum Clear Yield (critical leaf)
    min_yield_leaf = evaluator.add_leaf(
        id="MinimumClearYield",
        desc="Provide the minimum percentage of clear wood yield required for this grade",
        parent=specs_node,
        critical=True,
    )
    claim_min_yield = (
        f"The answer states that the minimum clear wood yield for the grade '{grade_name}' is '{min_yield}'."
    )
    await evaluator.verify(
        claim=claim_min_yield,
        node=min_yield_leaf,
        additional_instruction=(
            "Accept equivalent numeric representations (e.g., 83-1/3%, 83.3%, 83%). "
            "This check verifies that the answer reports a minimum clear wood yield value."
        ),
    )

    # Minimum Cutting Options (convert to parallel critical container with two leaves)
    cutting_node = evaluator.add_parallel(
        id="MinimumCuttingOptions",
        desc="Provide the two acceptable minimum clear cutting dimension options (width × length) for this grade",
        parent=specs_node,
        critical=True,
    )

    # Option A
    opt_a = cutting_opts[0] if len(cutting_opts) >= 1 else None
    opt_a_leaf = evaluator.add_leaf(
        id="MinimumCuttingOption_A",
        desc="Provide the first acceptable minimum clear cutting dimension option (width × length) for this grade",
        parent=cutting_node,
        critical=True,
    )
    claim_opt_a = (
        f"The answer includes an acceptable minimum clear cutting dimension option for the grade '{grade_name}': '{opt_a}'."
    )
    await evaluator.verify(
        claim=claim_opt_a,
        node=opt_a_leaf,
        additional_instruction=(
            "Verify that the answer lists at least one acceptable minimum clear cutting dimension option. "
            "Allow formatting variants ('x' vs '×', quotes, words)."
        ),
    )

    # Option B
    opt_b = cutting_opts[1] if len(cutting_opts) >= 2 else None
    opt_b_leaf = evaluator.add_leaf(
        id="MinimumCuttingOption_B",
        desc="Provide the second acceptable minimum clear cutting dimension option (width × length) for this grade",
        parent=cutting_node,
        critical=True,
    )
    claim_opt_b = (
        f"The answer includes a second acceptable minimum clear cutting dimension option for the grade '{grade_name}': '{opt_b}'."
    )
    await evaluator.verify(
        claim=claim_opt_b,
        node=opt_b_leaf,
        additional_instruction=(
            "Verify that the answer lists a second acceptable minimum clear cutting dimension option. "
            "Allow formatting variants ('x' vs '×', quotes, words)."
        ),
    )


async def build_general_requirements(
    evaluator: Evaluator,
    parent_node,
    extraction: WoodworkingExtraction,
) -> None:
    general_node = evaluator.add_parallel(
        id="GeneralWoodworkingRequirements",
        desc="Provide moisture content specifications and required PPE for indoor fine furniture woodworking",
        parent=parent_node,
        critical=True,
    )

    moisture_range = extraction.general.moisture_content_range if extraction.general else None
    ppe_eye = extraction.general.ppe_eye if extraction.general else None
    ppe_resp = extraction.general.ppe_respiratory if extraction.general else None
    ppe_hear = extraction.general.ppe_hearing if extraction.general else None

    # Acceptable Moisture Content (critical leaf)
    moisture_leaf = evaluator.add_leaf(
        id="AcceptableMoistureContent",
        desc="Provide the acceptable moisture content percentage range for fine furniture making and indoor woodworking",
        parent=general_node,
        critical=True,
    )
    claim_moisture = (
        f"The answer states the acceptable moisture content percentage range for fine furniture/indoor woodworking as '{moisture_range}'."
    )
    await evaluator.verify(
        claim=claim_moisture,
        node=moisture_leaf,
        additional_instruction=(
            "Judge whether the answer provides an indoor furniture moisture content range (commonly expressed in percent, e.g., 6–8%). "
            "Allow hyphens, en-dashes, 'to', and approximate wording."
        ),
    )

    # Required Safety Equipment (critical parallel container)
    ppe_node = evaluator.add_parallel(
        id="RequiredSafetyEquipment",
        desc="Provide the three required PPE types for woodworking operations with power tools",
        parent=general_node,
        critical=True,
    )

    # Eye Protection (critical leaf)
    eye_leaf = evaluator.add_leaf(
        id="EyeProtectionPPE",
        desc="Include required eye protection PPE (safety glasses or goggles)",
        parent=ppe_node,
        critical=True,
    )
    claim_eye = (
        f"The answer includes required eye protection for power tool woodworking, such as safety glasses or goggles (listed as '{ppe_eye}')."
    )
    await evaluator.verify(
        claim=claim_eye,
        node=eye_leaf,
        additional_instruction=(
            "Accept common eye protection terms: safety glasses, goggles, face shield. "
            "Focus on whether the answer includes an eye protection item."
        ),
    )

    # Respiratory Protection (critical leaf)
    resp_leaf = evaluator.add_leaf(
        id="RespiratoryProtectionPPE",
        desc="Include required respiratory protection PPE (dust mask or respirator)",
        parent=ppe_node,
        critical=True,
    )
    claim_resp = (
        f"The answer includes required respiratory protection for woodworking, such as a dust mask or respirator (listed as '{ppe_resp}')."
    )
    await evaluator.verify(
        claim=claim_resp,
        node=resp_leaf,
        additional_instruction=(
            "Accept common respiratory protection terms: dust mask, N95, respirator, cartridge respirator. "
            "Focus on whether the answer includes a respiratory protection item."
        ),
    )

    # Hearing Protection (critical leaf)
    hear_leaf = evaluator.add_leaf(
        id="HearingProtectionPPE",
        desc="Include required hearing protection PPE (hearing protection when operating power tools)",
        parent=ppe_node,
        critical=True,
    )
    claim_hear = (
        f"The answer includes required hearing protection for woodworking with power tools (listed as '{ppe_hear}')."
    )
    await evaluator.verify(
        claim=claim_hear,
        node=hear_leaf,
        additional_instruction=(
            "Accept common hearing protection terms: earmuffs, earplugs, hearing protectors. "
            "Focus on whether the answer includes a hearing protection item."
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

    # Extract all required information
    extraction = await evaluator.extract(
        prompt=prompt_extract_woodworking_info(),
        template_class=WoodworkingExtraction,
        extraction_name="woodworking_extraction",
    )

    # Build top-level critical node mirroring rubric root
    project_node = evaluator.add_parallel(
        id="WoodworkingProjectResearch",
        desc="Research and document lumber specifications and safety requirements for fine furniture making",
        parent=root,
        critical=True,
    )

    # Build subtrees
    await build_lumber_grade_spec(evaluator, project_node, extraction)
    await build_general_requirements(evaluator, project_node, extraction)

    # Return the evaluator summary
    return evaluator.get_summary()