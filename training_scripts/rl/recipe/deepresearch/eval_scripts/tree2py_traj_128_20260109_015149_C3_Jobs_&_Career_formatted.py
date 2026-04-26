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
TASK_ID = "interventional_cardiology_training_pathway"
TASK_DESCRIPTION = (
    "What are the complete sequential training requirements and durations for a physician to become a board-certified "
    "interventional cardiologist in the United States? Provide the following information for each training stage: "
    "(1) Internal Medicine Stage: The required duration of ACGME-accredited internal medicine residency training "
    "(including minimum months of training in general internal medicine and training level structure), and the board "
    "certification requirement. "
    "(2) Cardiovascular Disease Stage: The required duration of ACGME-accredited cardiovascular disease fellowship "
    "training (including minimum months of intensive clinical training), the prerequisite board certification, and the "
    "board certification requirement for this stage. "
    "(3) Interventional Cardiology Stage: The required duration of ACGME-accredited interventional cardiology "
    "fellowship training, the prerequisite fellowship training required before entry, the minimum number of coronary "
    "interventions that must be performed as primary operator, the minimum annual procedural volume requirement for the "
    "primary catheterization laboratory at the training facility, the specific procedural competencies that must be "
    "demonstrated, and the final board certification requirement. "
    "For each stage, include the relevant accrediting and certifying bodies (ACGME, ABIM), and specify all critical "
    "requirements including medical licensure and examination passage."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class InternalMedicineStageInfo(BaseModel):
    duration: Optional[str] = None  # e.g., "36 months", "3 years"
    min_general_im_months: Optional[str] = None  # e.g., "30 months"
    training_levels_structure: Optional[str] = None  # e.g., "PGY-1 (R1), PGY-2 (R2), PGY-3 (R3), 12-month intervals"
    acgme_accreditation_mentioned: Optional[bool] = None
    abim_internal_med_exam_required: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class CardiovascularDiseaseStageInfo(BaseModel):
    duration: Optional[str] = None  # e.g., "36 months", "3 years"
    min_intensive_clinical_months: Optional[str] = None  # e.g., "24 months"
    acgme_accreditation_mentioned: Optional[bool] = None
    prerequisite_im_board_cert: Optional[bool] = None
    abim_cvd_exam_required: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class InterventionalCardiologyStageInfo(BaseModel):
    duration: Optional[str] = None  # e.g., "12 months", "1 year"
    prerequisite_cvd_fellowship: Optional[bool] = None  # completion of 3-year CVD fellowship before entry
    min_primary_operator_coronary_interventions: Optional[str] = None  # e.g., "250"
    cath_lab_annual_volume_requirement: Optional[str] = None  # e.g., "400 per year"
    procedural_competencies: List[str] = Field(default_factory=list)  # list of procedures/skills
    acgme_accreditation_mentioned: Optional[bool] = None
    abim_ic_exam_required: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class TrainingPathExtraction(BaseModel):
    licensure_statement: Optional[str] = None
    licensure_sources: List[str] = Field(default_factory=list)
    internal_medicine: Optional[InternalMedicineStageInfo] = None
    cardiovascular_disease: Optional[CardiovascularDiseaseStageInfo] = None
    interventional_cardiology: Optional[InterventionalCardiologyStageInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_training_path() -> str:
    return """
    Extract the complete, sequential training pathway information for becoming a board-certified interventional cardiologist in the U.S., strictly from the provided answer text. Return a JSON object matching this schema:

    {
      "licensure_statement": string | null,   // Exact phrase/sentence in the answer about medical licensure requirement (valid/unrestricted license)
      "licensure_sources": string[],          // URLs explicitly cited in the answer that support the licensure requirement

      "internal_medicine": {
        "duration": string | null,                         // e.g., "36 months", "3 years"
        "min_general_im_months": string | null,            // e.g., "30 months"
        "training_levels_structure": string | null,        // e.g., "PGY-1/PGY-2/PGY-3 (12-month intervals)"
        "acgme_accreditation_mentioned": boolean | null,   // true if answer explicitly mentions ACGME-accredited
        "abim_internal_med_exam_required": boolean | null, // true if answer explicitly mentions needing to pass ABIM IM exam
        "sources": string[]                                // URLs explicitly cited in the answer specific to this stage
      },

      "cardiovascular_disease": {
        "duration": string | null,                         // e.g., "36 months", "3 years"
        "min_intensive_clinical_months": string | null,    // e.g., "24 months"
        "acgme_accreditation_mentioned": boolean | null,
        "prerequisite_im_board_cert": boolean | null,      // true if IM board certification prerequisite is stated
        "abim_cvd_exam_required": boolean | null,          // true if ABIM CVD exam requirement is stated
        "sources": string[]
      },

      "interventional_cardiology": {
        "duration": string | null,                                     // e.g., "12 months"
        "prerequisite_cvd_fellowship": boolean | null,                 // true if completion of 3y CVD fellowship before entry is stated
        "min_primary_operator_coronary_interventions": string | null,  // e.g., "250"
        "cath_lab_annual_volume_requirement": string | null,           // e.g., "400"
        "procedural_competencies": string[],                           // list of competencies explicitly listed in the answer
        "acgme_accreditation_mentioned": boolean | null,
        "abim_ic_exam_required": boolean | null,                       // true if ABIM IC exam requirement is stated
        "sources": string[]
      }
    }

    Rules:
    - Extract only what is explicitly present in the answer; do not infer or invent.
    - For any field not present, use null (or empty array for lists).
    - For boolean fields, set true only if the answer explicitly states it.
    - For URLs, extract actual links mentioned (including markdown links). If no URL is given, leave the list empty.
    - Prefer strings for durations and counts (e.g., "36 months", "3 years", "250").
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_medical_licensure(
    evaluator: Evaluator,
    parent_node,
    extraction: TrainingPathExtraction,
) -> None:
    # Existence of licensure sources (critical precondition)
    evaluator.add_custom_node(
        result=_has_urls(extraction.licensure_sources),
        id="Medical_Licensure_Sources_Provided",
        desc="Medical licensure sources are provided (URLs)",
        parent=parent_node,
        critical=True,
    )

    # Licensure requirement verification leaf
    licensure_leaf = evaluator.add_leaf(
        id="Medical_Licensure",
        desc="Hold a valid, unrestricted, and unchallenged medical license to practice medicine",
        parent=parent_node,
        critical=True,
    )

    claim = (
        "A valid, unrestricted medical license is required for ABIM board certification in Internal Medicine and "
        "subspecialties such as Interventional Cardiology."
    )
    await evaluator.verify(
        claim=claim,
        node=licensure_leaf,
        sources=extraction.licensure_sources,
        additional_instruction=(
            "Verify that ABIM states physicians must hold a valid/unrestricted license. Accept equivalent phrasing such "
            "as 'active and unrestricted license'."
        ),
    )


async def verify_internal_medicine_stage(
    evaluator: Evaluator,
    parent_node,
    im: Optional[InternalMedicineStageInfo],
) -> None:
    # Create the IM stage (sequential)
    im_stage = evaluator.add_sequential(
        id="Internal_Medicine_Stage",
        desc="Complete internal medicine residency training and meet ABIM internal medicine certification requirement",
        parent=parent_node,
        critical=True,
    )

    # Residency verification group (parallel within stage)
    im_residency = evaluator.add_parallel(
        id="Internal_Medicine_Residency",
        desc="Complete a 36-month ACGME-accredited internal medicine residency, including ≥30 months of general internal medicine, completed across R1/R2/R3",
        parent=im_stage,
        critical=True,
    )

    # Existence checks (critical preconditions)
    im_sources_ok = _has_urls(im.sources if im else None)
    evaluator.add_custom_node(
        result=im_sources_ok,
        id="IM_Sources_Provided",
        desc="Internal Medicine stage sources are provided (URLs)",
        parent=im_residency,
        critical=True,
    )

    im_fields_present = (
        im is not None
        and isinstance(im.duration, str) and im.duration.strip() != ""
        and isinstance(im.min_general_im_months, str) and im.min_general_im_months.strip() != ""
        and isinstance(im.training_levels_structure, str) and im.training_levels_structure.strip() != ""
    )
    evaluator.add_custom_node(
        result=im_fields_present,
        id="IM_Residency_Core_Fields_Present",
        desc="IM residency fields provided: duration, minimum general IM months, training-level structure",
        parent=im_residency,
        critical=True,
    )

    # Duration check
    im_duration_leaf = evaluator.add_leaf(
        id="IM_Duration",
        desc="Internal Medicine residency duration matches requirement",
        parent=im_residency,
        critical=True,
    )
    duration_txt = (im.duration if im and im.duration else "")
    await evaluator.verify(
        claim=f"The required duration of internal medicine residency is {duration_txt}.",
        node=im_duration_leaf,
        sources=(im.sources if im else []),
        additional_instruction=(
            "Confirm official requirement. Treat '3 years' as equivalent to '36 months'. Rely on ABIM/ACGME references."
        ),
    )

    # Minimum months in general internal medicine
    im_min_gim_leaf = evaluator.add_leaf(
        id="IM_Min_GIM_Months",
        desc="Minimum months of general internal medicine matches requirement",
        parent=im_residency,
        critical=True,
    )
    min_gim_txt = (im.min_general_im_months if im and im.min_general_im_months else "")
    await evaluator.verify(
        claim=f"The internal medicine residency requires at least {min_gim_txt} of general internal medicine patient care.",
        node=im_min_gim_leaf,
        sources=(im.sources if im else []),
        additional_instruction="Confirm the minimum general internal medicine time requirement on ABIM/ACGME official pages.",
    )

    # Training level structure (R-1/R-2/R-3 or PGY-1/2/3)
    im_levels_leaf = evaluator.add_leaf(
        id="IM_Training_Levels",
        desc="Training level structure across R1/R2/R3 (or PGY1/2/3) in 12-month intervals is supported",
        parent=im_residency,
        critical=True,
    )
    levels_txt = (im.training_levels_structure if im and im.training_levels_structure else "")
    await evaluator.verify(
        claim=(
            f"The internal medicine residency training levels are described as '{levels_txt}', indicating progression "
            f"across three residency years (e.g., PGY-1/R1, PGY-2/R2, PGY-3/R3) typically in 12-month intervals."
        ),
        node=im_levels_leaf,
        sources=(im.sources if im else []),
        additional_instruction=(
            "Allow equivalent terminology (PGY vs R-levels). Verify that the program spans three progressive years."
        ),
    )

    # ACGME Accreditation requirement
    im_acgme_leaf = evaluator.add_leaf(
        id="IM_ACGME_Accreditation",
        desc="Internal Medicine residency must be ACGME-accredited",
        parent=im_residency,
        critical=True,
    )
    await evaluator.verify(
        claim="The internal medicine residency must be ACGME-accredited.",
        node=im_acgme_leaf,
        sources=(im.sources if im else []),
        additional_instruction="Confirm that ABIM/ACGME require training in an ACGME-accredited residency.",
    )

    # ABIM exam mention (existence gate) then verification
    evaluator.add_custom_node(
        result=bool(im and im.abim_internal_med_exam_required),
        id="IM_ABIM_Exam_Mentioned",
        desc="Answer explicitly mentions passing the ABIM Internal Medicine Certification Examination",
        parent=im_stage,
        critical=True,
    )

    im_exam_leaf = evaluator.add_leaf(
        id="ABIM_Internal_Medicine_Exam",
        desc="Pass the ABIM Internal Medicine Certification Examination (board certification requirement for internal medicine)",
        parent=im_stage,
        critical=True,
    )
    await evaluator.verify(
        claim="Passing the ABIM Internal Medicine Certification Examination is required to become board certified in Internal Medicine.",
        node=im_exam_leaf,
        sources=(im.sources if im else []),
        additional_instruction="Verify via ABIM official certification pages.",
    )


async def verify_cvd_stage(
    evaluator: Evaluator,
    parent_node,
    cvd: Optional[CardiovascularDiseaseStageInfo],
) -> None:
    # Create the CVD stage (sequential)
    cvd_stage = evaluator.add_sequential(
        id="Cardiovascular_Disease_Stage",
        desc="Complete cardiovascular disease fellowship training and meet ABIM cardiovascular disease certification requirements",
        parent=parent_node,
        critical=True,
    )

    # Fellowship verification group
    cvd_fellowship = evaluator.add_parallel(
        id="Cardiovascular_Disease_Fellowship",
        desc="Complete a 3-year ACGME-accredited CVD fellowship with at least 24 months of intensive clinical training",
        parent=cvd_stage,
        critical=True,
    )

    # Existence checks (critical preconditions)
    cvd_sources_ok = _has_urls(cvd.sources if cvd else None)
    evaluator.add_custom_node(
        result=cvd_sources_ok,
        id="CVD_Sources_Provided",
        desc="Cardiovascular Disease stage sources are provided (URLs)",
        parent=cvd_fellowship,
        critical=True,
    )

    cvd_fields_present = (
        cvd is not None
        and isinstance(cvd.duration, str) and cvd.duration.strip() != ""
        and isinstance(cvd.min_intensive_clinical_months, str) and cvd.min_intensive_clinical_months.strip() != ""
    )
    evaluator.add_custom_node(
        result=cvd_fields_present,
        id="CVD_Fellowship_Core_Fields_Present",
        desc="CVD fellowship fields provided: duration and minimum intensive clinical months",
        parent=cvd_fellowship,
        critical=True,
    )

    # Duration check
    cvd_duration_leaf = evaluator.add_leaf(
        id="CVD_Duration",
        desc="Cardiovascular Disease fellowship duration matches requirement",
        parent=cvd_fellowship,
        critical=True,
    )
    duration_txt = (cvd.duration if cvd and cvd.duration else "")
    await evaluator.verify(
        claim=f"The cardiovascular disease fellowship duration is {duration_txt} in an ACGME-accredited program.",
        node=cvd_duration_leaf,
        sources=(cvd.sources if cvd else []),
        additional_instruction="Treat '3 years' as equivalent to '36 months'. Verify on ABIM/ACGME official references.",
    )

    # Minimum intensive clinical months
    cvd_min_clin_leaf = evaluator.add_leaf(
        id="CVD_Min_Intensive_Clinical_Months",
        desc="Minimum months of intensive clinical training matches requirement",
        parent=cvd_fellowship,
        critical=True,
    )
    min_clin_txt = (cvd.min_intensive_clinical_months if cvd and cvd.min_intensive_clinical_months else "")
    await evaluator.verify(
        claim=f"The fellowship includes at least {min_clin_txt} of intensive clinical training.",
        node=cvd_min_clin_leaf,
        sources=(cvd.sources if cvd else []),
        additional_instruction="Verify that ABIM/ACGME specify this minimum intensive clinical duration.",
    )

    # ACGME Accreditation requirement
    cvd_acgme_leaf = evaluator.add_leaf(
        id="CVD_ACGME_Accreditation",
        desc="Cardiovascular Disease fellowship must be ACGME-accredited",
        parent=cvd_fellowship,
        critical=True,
    )
    await evaluator.verify(
        claim="The cardiovascular disease fellowship must be ACGME-accredited.",
        node=cvd_acgme_leaf,
        sources=(cvd.sources if cvd else []),
        additional_instruction="Confirm on ABIM/ACGME documentation.",
    )

    # Prerequisite: IM board certification (mention gate + verification)
    evaluator.add_custom_node(
        result=bool(cvd and cvd.prerequisite_im_board_cert),
        id="CVD_Prerequisite_IM_Board_Mentioned",
        desc="Answer states ABIM Internal Medicine certification prerequisite for CVD certification",
        parent=cvd_stage,
        critical=True,
    )
    cvd_prereq_leaf = evaluator.add_leaf(
        id="Prerequisite_IM_Board_Certification_For_CVD",
        desc="Hold ABIM certification in internal medicine before pursuing cardiovascular disease board certification",
        parent=cvd_stage,
        critical=True,
    )
    await evaluator.verify(
        claim="ABIM certification in Internal Medicine is a prerequisite to ABIM Cardiovascular Disease certification.",
        node=cvd_prereq_leaf,
        sources=(cvd.sources if cvd else []),
        additional_instruction="Verify this prerequisite on ABIM certification policy pages.",
    )

    # ABIM CVD exam (mention gate + verification)
    evaluator.add_custom_node(
        result=bool(cvd and cvd.abim_cvd_exam_required),
        id="CVD_ABIM_Exam_Mentioned",
        desc="Answer explicitly mentions passing the ABIM Cardiovascular Disease Certification Examination",
        parent=cvd_stage,
        critical=True,
    )
    cvd_exam_leaf = evaluator.add_leaf(
        id="ABIM_Cardiovascular_Disease_Exam",
        desc="Pass the ABIM Cardiovascular Disease Certification Examination (board certification requirement for cardiovascular disease)",
        parent=cvd_stage,
        critical=True,
    )
    await evaluator.verify(
        claim="Passing the ABIM Cardiovascular Disease Certification Examination is required to be board certified in Cardiovascular Disease.",
        node=cvd_exam_leaf,
        sources=(cvd.sources if cvd else []),
        additional_instruction="Verify via ABIM official certification pages.",
    )


async def verify_interventional_cardiology_stage(
    evaluator: Evaluator,
    parent_node,
    ic: Optional[InterventionalCardiologyStageInfo],
) -> None:
    # Create the Interventional Cardiology stage (sequential)
    ic_stage = evaluator.add_sequential(
        id="Interventional_Cardiology_Stage",
        desc="Complete interventional cardiology fellowship training (including volumes/competencies) and meet ABIM interventional cardiology certification requirement",
        parent=parent_node,
        critical=True,
    )

    # Fellowship requirements group (parallel)
    ic_req = evaluator.add_parallel(
        id="Interventional_Cardiology_Fellowship_Requirements",
        desc="Meet all ACGME-accredited interventional cardiology fellowship entry and training requirements",
        parent=ic_stage,
        critical=True,
    )

    # Existence checks (critical preconditions)
    ic_sources_ok = _has_urls(ic.sources if ic else None)
    evaluator.add_custom_node(
        result=ic_sources_ok,
        id="IC_Sources_Provided",
        desc="Interventional Cardiology stage sources are provided (URLs)",
        parent=ic_req,
        critical=True,
    )

    ic_core_fields_present = (
        ic is not None
        and isinstance(ic.duration, str) and ic.duration.strip() != ""
        and isinstance(ic.min_primary_operator_coronary_interventions, str) and ic.min_primary_operator_coronary_interventions.strip() != ""
        and isinstance(ic.cath_lab_annual_volume_requirement, str) and ic.cath_lab_annual_volume_requirement.strip() != ""
        and isinstance(ic.procedural_competencies, list) and len(ic.procedural_competencies) > 0
    )
    evaluator.add_custom_node(
        result=ic_core_fields_present,
        id="IC_Core_Fields_Present",
        desc="IC fellowship fields provided: duration, primary-operator volume, cath-lab annual volume, competencies",
        parent=ic_req,
        critical=True,
    )

    # Duration (12 months) and ACGME accreditation
    ic_duration_leaf = evaluator.add_leaf(
        id="IC_Fellowship_Duration",
        desc="Interventional Cardiology fellowship duration matches requirement",
        parent=ic_req,
        critical=True,
    )
    duration_txt = (ic.duration if ic and ic.duration else "")
    await evaluator.verify(
        claim=f"The interventional cardiology fellowship duration is {duration_txt}.",
        node=ic_duration_leaf,
        sources=(ic.sources if ic else []),
        additional_instruction="Treat '1 year' as equivalent to '12 months'. Verify on ABIM/ACGME references.",
    )

    ic_acgme_leaf = evaluator.add_leaf(
        id="IC_ACGME_Accreditation",
        desc="Interventional Cardiology fellowship must be ACGME-accredited",
        parent=ic_req,
        critical=True,
    )
    await evaluator.verify(
        claim="The interventional cardiology fellowship must be ACGME-accredited.",
        node=ic_acgme_leaf,
        sources=(ic.sources if ic else []),
        additional_instruction="Verify via ABIM/ACGME program requirements.",
    )

    # Prerequisite: completion of 3y CVD fellowship before entry
    ic_prereq_cvd_leaf = evaluator.add_leaf(
        id="IC_Prerequisite_CVD_Fellowship_Completion",
        desc="Complete the required 3 years of ACGME-accredited cardiovascular disease fellowship training before entering interventional cardiology fellowship",
        parent=ic_req,
        critical=True,
    )
    await evaluator.verify(
        claim="Completion of a full 3-year ACGME-accredited Cardiovascular Disease fellowship is required before entering an Interventional Cardiology fellowship.",
        node=ic_prereq_cvd_leaf,
        sources=(ic.sources if ic else []),
        additional_instruction="Confirm using ABIM/ACGME documents that IC requires prior CVD fellowship completion.",
    )

    # Minimum 250 primary-operator coronary interventions
    ic_primary_ops_leaf = evaluator.add_leaf(
        id="IC_Primary_Operator_Coronary_Interventions",
        desc="Perform a minimum of 250 coronary interventions as the primary operator during interventional cardiology fellowship training",
        parent=ic_req,
        critical=True,
    )
    min_ops_txt = (ic.min_primary_operator_coronary_interventions if ic and ic.min_primary_operator_coronary_interventions else "")
    await evaluator.verify(
        claim=f"The fellow must perform at least {min_ops_txt} coronary interventions as the primary operator during training.",
        node=ic_primary_ops_leaf,
        sources=(ic.sources if ic else []),
        additional_instruction="Verify official minimum case volume as primary operator (exclude assistant-only cases).",
    )

    # Training facility primary cath lab minimum annual volume
    ic_lab_volume_leaf = evaluator.add_leaf(
        id="IC_Training_Facility_Annual_Volume",
        desc="Primary catheterization laboratory has a minimum annual interventional volume requirement",
        parent=ic_req,
        critical=True,
    )
    vol_txt = (ic.cath_lab_annual_volume_requirement if ic and ic.cath_lab_annual_volume_requirement else "")
    await evaluator.verify(
        claim=f"The primary cardiac catheterization laboratory must perform at least {vol_txt} interventional procedures per year.",
        node=ic_lab_volume_leaf,
        sources=(ic.sources if ic else []),
        additional_instruction="Confirm the minimum annual volume requirement for the training facility's primary cath lab.",
    )

    # Procedural competencies
    ic_competencies_leaf = evaluator.add_leaf(
        id="IC_Procedural_Competencies",
        desc="Specific procedural competencies are required during interventional cardiology fellowship",
        parent=ic_req,
        critical=True,
    )
    comp_txt = ", ".join(ic.procedural_competencies) if ic and ic.procedural_competencies else ""
    await evaluator.verify(
        claim=f"The fellow must demonstrate competence in the following procedures: {comp_txt}.",
        node=ic_competencies_leaf,
        sources=(ic.sources if ic else []),
        additional_instruction=(
            "Typical competencies include coronary angiography, coronary interventions (balloon angioplasty/stents), "
            "intravascular imaging (e.g., IVUS), hemodynamic measurements, and ventriculography/aortography. Verify that "
            "the listed competencies are supported by official requirements."
        ),
    )

    # ABIM IC exam (mention gate + verification)
    evaluator.add_custom_node(
        result=bool(ic and ic.abim_ic_exam_required),
        id="IC_ABIM_Exam_Mentioned",
        desc="Answer explicitly mentions passing the ABIM Interventional Cardiology Certification Examination",
        parent=ic_stage,
        critical=True,
    )
    ic_exam_leaf = evaluator.add_leaf(
        id="ABIM_Interventional_Cardiology_Exam",
        desc="Pass the ABIM Interventional Cardiology Certification Examination (final board certification requirement)",
        parent=ic_stage,
        critical=True,
    )
    await evaluator.verify(
        claim="Passing the ABIM Interventional Cardiology Certification Examination is required for board certification in Interventional Cardiology.",
        node=ic_exam_leaf,
        sources=(ic.sources if ic else []),
        additional_instruction="Verify via ABIM official certification pages.",
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
    # Initialize evaluator with a parallel root, then create a critical top-level node mirroring rubric root
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

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_training_path(),
        template_class=TrainingPathExtraction,
        extraction_name="training_path_extraction",
    )

    # Build verification tree according to rubric
    complete_node = evaluator.add_parallel(
        id="Complete_Interventional_Cardiology_Training_Pathway",
        desc="All requirements to become a board-certified interventional cardiologist in the United States",
        parent=root,
        critical=True,
    )

    # 1) Medical licensure (critical)
    await verify_medical_licensure(evaluator, complete_node, extraction)

    # 2) Sequential training stages (critical)
    stages_node = evaluator.add_sequential(
        id="Sequential_Training_Stages",
        desc="Complete the sequential training stages and associated ABIM certifications/exams",
        parent=complete_node,
        critical=True,
    )

    # Internal Medicine Stage
    await verify_internal_medicine_stage(evaluator, stages_node, extraction.internal_medicine)

    # Cardiovascular Disease Stage
    await verify_cvd_stage(evaluator, stages_node, extraction.cardiovascular_disease)

    # Interventional Cardiology Stage
    await verify_interventional_cardiology_stage(evaluator, stages_node, extraction.interventional_cardiology)

    # Return standard summary
    return evaluator.get_summary()