import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "morocco_construction_insurance_2024"
TASK_DESCRIPTION = (
    "Morocco introduced mandatory construction insurance requirements that became effective in December 2024. Identify "
    "the two complementary insurance types that are now required for real estate construction projects, specify which "
    "types and sizes of buildings are subject to these requirements, describe what each insurance type covers and for "
    "how long, and explain the mandatory technical inspection requirements that must be fulfilled for the insurance "
    "coverage to be activated."
)


# -------------------------------
# Extraction Models
# -------------------------------
class FrameworkExtraction(BaseModel):
    insurance_types: List[str] = Field(default_factory=list)
    effective_date_text: Optional[str] = None
    sources_framework: List[str] = Field(default_factory=list)


class ApplicabilityExtraction(BaseModel):
    residential_threshold_statement: Optional[str] = None
    non_residential_threshold_statement: Optional[str] = None
    public_exclusion_statement: Optional[str] = None
    sources_applicability: List[str] = Field(default_factory=list)


class TRCCoverageExtraction(BaseModel):
    trc_phase_statement: Optional[str] = None
    trc_accidental_damage_statement: Optional[str] = None
    trc_integrated_materials_equipment_statement: Optional[str] = None
    trc_third_party_liability_statement: Optional[str] = None
    sources_trc: List[str] = Field(default_factory=list)


class RCDCoverageExtraction(BaseModel):
    rcd_start_time_statement: Optional[str] = None
    rcd_duration_statement: Optional[str] = None
    rcd_structural_integrity_statement: Optional[str] = None
    rcd_soil_or_design_flaws_statement: Optional[str] = None
    sources_rcd: List[str] = Field(default_factory=list)


class InspectionExtraction(BaseModel):
    two_inspections_statement: Optional[str] = None
    pre_construction_timing_statement: Optional[str] = None
    pre_construction_validates_plans_statement: Optional[str] = None
    pre_construction_validates_soil_statement: Optional[str] = None
    pre_construction_validates_tech_specs_statement: Optional[str] = None
    delivery_timing_statement: Optional[str] = None
    delivery_ensures_compliance_statement: Optional[str] = None
    delivery_finalizes_contract_statement: Optional[str] = None
    activation_condition_statement: Optional[str] = None
    sources_inspections: List[str] = Field(default_factory=list)


# -------------------------------
# Extraction Prompts
# -------------------------------
def prompt_extract_framework() -> str:
    return """
    Extract the identification of Morocco's mandatory construction insurance framework (effective December 2024) from the answer.

    Required fields:
    - insurance_types: List of the two complementary mandatory types named or acronymed in the answer, including any synonyms or language variants (e.g., "TRC", "Tous Risques Chantier", "All Risks Construction"; "RCD", "Responsabilité Civile Décennale", "Ten-Year Civil Liability"). Only include items the answer explicitly claims.
    - effective_date_text: The exact phrasing of when the requirements became mandatory, e.g., "December 2024", "1 December 2024", etc., as stated in the answer.
    - sources_framework: All URLs explicitly cited in the answer that support the framework identification and timing. Only include valid URLs present in the answer text (markdown links allowed).

    If any field is missing in the answer, return null (for single values) or [] for lists.
    """


def prompt_extract_applicability() -> str:
    return """
    Extract the applicability criteria for which buildings are subject to Morocco's mandatory construction insurance (TRC/RCD) and any exclusions, as stated in the answer.

    Required fields:
    - residential_threshold_statement: The answer's statement about residential buildings being subject (e.g., "over 3 floors OR over 800 m²").
    - non_residential_threshold_statement: The answer's statement about mixed-use, industrial, hotel, educational, or office buildings (e.g., "covered surface area exceeds 400 m²").
    - public_exclusion_statement: The answer's statement that public projects (e.g., roads, ports, dams) are excluded.
    - sources_applicability: All URLs explicitly cited in the answer that support these applicability thresholds and exclusions.

    Return null for missing statements; return [] if no sources are provided.
    """


def prompt_extract_trc() -> str:
    return """
    Extract TRC (All Risks Construction / Tous Risques Chantier) coverage details from the answer.

    Required fields:
    - trc_phase_statement: The statement that TRC applies during the construction phase.
    - trc_accidental_damage_statement: The statement that TRC covers accidental damage occurring on the construction site.
    - trc_integrated_materials_equipment_statement: The statement that TRC covers integrated materials and equipment.
    - trc_third_party_liability_statement: The statement that TRC covers civil liability for damages to third parties.
    - sources_trc: All URLs explicitly cited in the answer that support the TRC coverage details.

    Return null for missing statements; return [] if no sources are provided.
    """


def prompt_extract_rcd() -> str:
    return """
    Extract RCD (Ten-Year Civil Liability / Responsabilité Civile Décennale) coverage details and duration from the answer.

    Required fields:
    - rcd_start_time_statement: The statement that RCD takes effect after delivery/completion of the project.
    - rcd_duration_statement: The statement that RCD coverage lasts for 10 years after delivery.
    - rcd_structural_integrity_statement: The statement that RCD covers construction defects affecting structural integrity.
    - rcd_soil_or_design_flaws_statement: The statement that RCD covers soil defects or design flaws making the building unsuitable for its intended purpose.
    - sources_rcd: All URLs explicitly cited in the answer that support the RCD coverage details and duration.

    Return null for missing statements; return [] if no sources are provided.
    """


def prompt_extract_inspections() -> str:
    return """
    Extract the mandatory technical inspection requirements and activation conditions for insurance coverage from the answer.

    Required fields:
    - two_inspections_statement: The statement that there are two mandatory technical inspections performed by certified control offices.
    - pre_construction_timing_statement: The statement that the first inspection occurs before construction begins.
    - pre_construction_validates_plans_statement: The statement that the first inspection validates plans.
    - pre_construction_validates_soil_statement: The statement that the first inspection validates soil studies.
    - pre_construction_validates_tech_specs_statement: The statement that the first inspection validates technical specifications.
    - delivery_timing_statement: The statement that the second inspection occurs at delivery.
    - delivery_ensures_compliance_statement: The statement that the second inspection ensures compliance.
    - delivery_finalizes_contract_statement: The statement that the second inspection finalizes the insurance contract.
    - activation_condition_statement: The statement that without the two inspection reports, insurance coverage cannot be activated.
    - sources_inspections: All URLs explicitly cited in the answer that support these inspection requirements and activation conditions.

    Return null for missing statements; return [] if no sources are provided.
    """


# -------------------------------
# Helper: Additional instruction builder
# -------------------------------
def build_url_support_instruction(section_name: str, urls: List[str]) -> str:
    if urls and len(urls) > 0:
        return (
            f"For the '{section_name}' verification, rely strictly on the provided webpage URLs. Consider minor wording or "
            "language variants (English/French/Arabic) acceptable if the substance matches. If any URL is irrelevant, does not load, "
            "or contradicts the claim, judge as not supported (Incorrect). Prefer official bulletins/regulatory texts or authoritative "
            "industry communications when available."
        )
    else:
        return (
            f"For the '{section_name}' verification, no URLs were provided in the answer to support the claim. You must judge this "
            "as not supported (Incorrect). Do not rely on general knowledge—evidence from cited webpages is required."
        )


# -------------------------------
# Verification Subtrees
# -------------------------------
async def verify_regulation_framework(
    evaluator: Evaluator,
    parent_node,
    framework: FrameworkExtraction,
):
    reg_node = evaluator.add_parallel(
        id="RegulationFrameworkIdentification",
        desc="Identify the two mandatory complementary insurance types and when they became mandatory.",
        parent=parent_node,
        critical=True,
    )

    # Insurance types identification
    types_leaf = evaluator.add_leaf(
        id="InsuranceTypesIdentification",
        desc="Identifies both required insurance types: TRC (All Risks Construction Insurance) and RCD (Ten-Year Civil Liability Insurance).",
        parent=reg_node,
        critical=True,
    )
    claim_types = (
        "Morocco's mandatory construction insurance for real estate projects consists of two complementary types: "
        "TRC (Tous Risques Chantier / All Risks Construction) and RCD (Responsabilité Civile Décennale / Ten-Year Civil Liability)."
    )
    await evaluator.verify(
        claim=claim_types,
        node=types_leaf,
        sources=framework.sources_framework,
        additional_instruction=build_url_support_instruction("InsuranceTypesIdentification", framework.sources_framework),
    )

    # Temporal implementation (mandatory effective December 2024)
    temporal_leaf = evaluator.add_leaf(
        id="TemporalImplementation",
        desc="Specifies the requirements became mandatory effective December 2024.",
        parent=reg_node,
        critical=True,
    )
    claim_time = "These construction insurance requirements became mandatory in December 2024."
    await evaluator.verify(
        claim=claim_time,
        node=temporal_leaf,
        sources=framework.sources_framework,
        additional_instruction=build_url_support_instruction("TemporalImplementation", framework.sources_framework)
        + " Allow minor date formatting variants such as '1 December 2024' or 'Dec 2024'.",
    )


async def verify_applicability_criteria(
    evaluator: Evaluator,
    parent_node,
    applicability: ApplicabilityExtraction,
):
    app_node = evaluator.add_parallel(
        id="ApplicabilityCriteria",
        desc="State which building types/sizes are subject to the mandatory insurance requirements and what is excluded.",
        parent=parent_node,
        critical=True,
    )

    # Residential threshold
    residential_leaf = evaluator.add_leaf(
        id="ResidentialBuildingThreshold",
        desc="States residential buildings are subject if over 3 floors OR over 800 m².",
        parent=app_node,
        critical=True,
    )
    claim_res = "Residential buildings are subject to the mandatory insurance if they are over 3 floors or exceed 800 m² in area."
    await evaluator.verify(
        claim=claim_res,
        node=residential_leaf,
        sources=applicability.sources_applicability,
        additional_instruction=build_url_support_instruction("ResidentialBuildingThreshold", applicability.sources_applicability),
    )

    # Non-residential threshold
    nonres_leaf = evaluator.add_leaf(
        id="NonResidentialBuildingThreshold",
        desc="States mixed-use, industrial, hotel, educational, or office buildings are subject if covered surface area exceeds 400 m².",
        parent=app_node,
        critical=True,
    )
    claim_nonres = (
        "Mixed-use, industrial, hotel, educational, or office buildings are subject to the mandatory insurance if the covered surface "
        "area exceeds 400 m²."
    )
    await evaluator.verify(
        claim=claim_nonres,
        node=nonres_leaf,
        sources=applicability.sources_applicability,
        additional_instruction=build_url_support_instruction("NonResidentialBuildingThreshold", applicability.sources_applicability),
    )

    # Public projects exclusion
    public_excl_leaf = evaluator.add_leaf(
        id="PublicProjectsExclusion",
        desc="States public projects (e.g., roads, ports, dams) are excluded from these insurance requirements.",
        parent=app_node,
        critical=True,
    )
    claim_public = "Public projects such as roads, ports, and dams are excluded from these mandatory insurance requirements."
    await evaluator.verify(
        claim=claim_public,
        node=public_excl_leaf,
        sources=applicability.sources_applicability,
        additional_instruction=build_url_support_instruction("PublicProjectsExclusion", applicability.sources_applicability),
    )


async def verify_trc_coverage(
    evaluator: Evaluator,
    parent_node,
    trc: TRCCoverageExtraction,
):
    trc_node = evaluator.add_parallel(
        id="TRCCoverageDetails",
        desc="Describe TRC coverage scope.",
        parent=parent_node,
        critical=True,
    )

    # TRC applies during construction phase
    trc_phase_leaf = evaluator.add_leaf(
        id="TRCPhase",
        desc="States TRC coverage applies during the construction phase.",
        parent=trc_node,
        critical=True,
    )
    claim_trc_phase = "TRC coverage applies during the construction phase."
    await evaluator.verify(
        claim=claim_trc_phase,
        node=trc_phase_leaf,
        sources=trc.sources_trc,
        additional_instruction=build_url_support_instruction("TRCPhase", trc.sources_trc),
    )

    # TRC covers accidental damage on site
    trc_acc_leaf = evaluator.add_leaf(
        id="TRCAccidentalDamageOnSite",
        desc="States TRC covers accidental damage occurring on the construction site.",
        parent=trc_node,
        critical=True,
    )
    claim_trc_acc = "TRC covers accidental damage occurring on the construction site."
    await evaluator.verify(
        claim=claim_trc_acc,
        node=trc_acc_leaf,
        sources=trc.sources_trc,
        additional_instruction=build_url_support_instruction("TRCAccidentalDamageOnSite", trc.sources_trc),
    )

    # TRC covers integrated materials and equipment
    trc_mat_leaf = evaluator.add_leaf(
        id="TRCIntegratedMaterialsEquipment",
        desc="States TRC covers integrated materials and equipment.",
        parent=trc_node,
        critical=True,
    )
    claim_trc_mat = "TRC covers integrated materials and equipment."
    await evaluator.verify(
        claim=claim_trc_mat,
        node=trc_mat_leaf,
        sources=trc.sources_trc,
        additional_instruction=build_url_support_instruction("TRCIntegratedMaterialsEquipment", trc.sources_trc),
    )

    # TRC covers third-party civil liability
    trc_tpl_leaf = evaluator.add_leaf(
        id="TRCThirdPartyLiability",
        desc="States TRC covers civil liability for damages to third parties.",
        parent=trc_node,
        critical=True,
    )
    claim_trc_tpl = "TRC covers civil liability for damages to third parties."
    await evaluator.verify(
        claim=claim_trc_tpl,
        node=trc_tpl_leaf,
        sources=trc.sources_trc,
        additional_instruction=build_url_support_instruction("TRCThirdPartyLiability", trc.sources_trc),
    )


async def verify_rcd_coverage(
    evaluator: Evaluator,
    parent_node,
    rcd: RCDCoverageExtraction,
):
    rcd_node = evaluator.add_parallel(
        id="RCDCoverageDetails",
        desc="Describe RCD coverage scope and duration.",
        parent=parent_node,
        critical=True,
    )

    # RCD start time after delivery
    rcd_start_leaf = evaluator.add_leaf(
        id="RCDStartTime",
        desc="States RCD takes effect after delivery/completion of the project.",
        parent=rcd_node,
        critical=True,
    )
    claim_rcd_start = "RCD takes effect after the delivery/completion of the project."
    await evaluator.verify(
        claim=claim_rcd_start,
        node=rcd_start_leaf,
        sources=rcd.sources_rcd,
        additional_instruction=build_url_support_instruction("RCDStartTime", rcd.sources_rcd),
    )

    # RCD duration 10 years after delivery
    rcd_duration_leaf = evaluator.add_leaf(
        id="RCDDuration",
        desc="States RCD coverage lasts for 10 years after delivery.",
        parent=rcd_node,
        critical=True,
    )
    claim_rcd_duration = "RCD coverage lasts for ten years after delivery."
    await evaluator.verify(
        claim=claim_rcd_duration,
        node=rcd_duration_leaf,
        sources=rcd.sources_rcd,
        additional_instruction=build_url_support_instruction("RCDDuration", rcd.sources_rcd),
    )

    # RCD structural integrity defects
    rcd_struct_leaf = evaluator.add_leaf(
        id="RCDStructuralIntegrityDefects",
        desc="States RCD covers construction defects affecting structural integrity.",
        parent=rcd_node,
        critical=True,
    )
    claim_rcd_struct = "RCD covers construction defects affecting the structural integrity of the building."
    await evaluator.verify(
        claim=claim_rcd_struct,
        node=rcd_struct_leaf,
        sources=rcd.sources_rcd,
        additional_instruction=build_url_support_instruction("RCDStructuralIntegrityDefects", rcd.sources_rcd),
    )

    # RCD soil or design flaws making building unsuitable
    rcd_soil_design_leaf = evaluator.add_leaf(
        id="RCDSoilOrDesignFlaws",
        desc="States RCD covers soil defects or design flaws that make the building unsuitable for its intended purpose.",
        parent=rcd_node,
        critical=True,
    )
    claim_rcd_soil_design = "RCD covers soil defects or design flaws that render the building unsuitable for its intended purpose."
    await evaluator.verify(
        claim=claim_rcd_soil_design,
        node=rcd_soil_design_leaf,
        sources=rcd.sources_rcd,
        additional_instruction=build_url_support_instruction("RCDSoilOrDesignFlaws", rcd.sources_rcd),
    )


async def verify_inspection_requirements(
    evaluator: Evaluator,
    parent_node,
    insp: InspectionExtraction,
):
    insp_node = evaluator.add_parallel(
        id="MandatoryTechnicalInspectionRequirements",
        desc="Explain the mandatory technical inspections required and the activation condition for insurance coverage.",
        parent=parent_node,
        critical=True,
    )

    # Two inspections by certified control offices
    two_insp_leaf = evaluator.add_leaf(
        id="TwoInspectionsByCertifiedOffices",
        desc="States there are two mandatory technical inspections performed by certified control offices.",
        parent=insp_node,
        critical=True,
    )
    claim_two_insp = "There are two mandatory technical inspections performed by certified control offices."
    await evaluator.verify(
        claim=claim_two_insp,
        node=two_insp_leaf,
        sources=insp.sources_inspections,
        additional_instruction=build_url_support_instruction("TwoInspectionsByCertifiedOffices", insp.sources_inspections),
    )

    # Pre-construction inspection block
    pre_node = evaluator.add_parallel(
        id="PreConstructionInspection",
        desc="Describe the first (pre-construction) inspection.",
        parent=insp_node,
        critical=True,
    )

    pre_time_leaf = evaluator.add_leaf(
        id="PreConstructionTiming",
        desc="States the first inspection occurs before construction begins.",
        parent=pre_node,
        critical=True,
    )
    claim_pre_time = "The first inspection occurs before construction begins."
    await evaluator.verify(
        claim=claim_pre_time,
        node=pre_time_leaf,
        sources=insp.sources_inspections,
        additional_instruction=build_url_support_instruction("PreConstructionTiming", insp.sources_inspections),
    )

    pre_plans_leaf = evaluator.add_leaf(
        id="PreConstructionValidatesPlans",
        desc="States the first inspection validates plans.",
        parent=pre_node,
        critical=True,
    )
    claim_pre_plans = "The first inspection validates plans."
    await evaluator.verify(
        claim=claim_pre_plans,
        node=pre_plans_leaf,
        sources=insp.sources_inspections,
        additional_instruction=build_url_support_instruction("PreConstructionValidatesPlans", insp.sources_inspections),
    )

    pre_soil_leaf = evaluator.add_leaf(
        id="PreConstructionValidatesSoilStudies",
        desc="States the first inspection validates soil studies.",
        parent=pre_node,
        critical=True,
    )
    claim_pre_soil = "The first inspection validates soil studies."
    await evaluator.verify(
        claim=claim_pre_soil,
        node=pre_soil_leaf,
        sources=insp.sources_inspections,
        additional_instruction=build_url_support_instruction("PreConstructionValidatesSoilStudies", insp.sources_inspections),
    )

    pre_specs_leaf = evaluator.add_leaf(
        id="PreConstructionValidatesTechnicalSpecifications",
        desc="States the first inspection validates technical specifications.",
        parent=pre_node,
        critical=True,
    )
    claim_pre_specs = "The first inspection validates technical specifications."
    await evaluator.verify(
        claim=claim_pre_specs,
        node=pre_specs_leaf,
        sources=insp.sources_inspections,
        additional_instruction=build_url_support_instruction("PreConstructionValidatesTechnicalSpecifications", insp.sources_inspections),
    )

    # Delivery inspection block
    deliv_node = evaluator.add_parallel(
        id="DeliveryInspection",
        desc="Describe the second (delivery) inspection.",
        parent=insp_node,
        critical=True,
    )

    deliv_time_leaf = evaluator.add_leaf(
        id="DeliveryTiming",
        desc="States the second inspection occurs at delivery.",
        parent=deliv_node,
        critical=True,
    )
    claim_deliv_time = "The second inspection occurs at delivery."
    await evaluator.verify(
        claim=claim_deliv_time,
        node=deliv_time_leaf,
        sources=insp.sources_inspections,
        additional_instruction=build_url_support_instruction("DeliveryTiming", insp.sources_inspections),
    )

    deliv_compliance_leaf = evaluator.add_leaf(
        id="DeliveryEnsuresCompliance",
        desc="States the second inspection ensures compliance.",
        parent=deliv_node,
        critical=True,
    )
    claim_deliv_compliance = "The second inspection ensures compliance."
    await evaluator.verify(
        claim=claim_deliv_compliance,
        node=deliv_compliance_leaf,
        sources=insp.sources_inspections,
        additional_instruction=build_url_support_instruction("DeliveryEnsuresCompliance", insp.sources_inspections),
    )

    deliv_finalize_leaf = evaluator.add_leaf(
        id="DeliveryFinalizesInsuranceContract",
        desc="States the second inspection finalizes the insurance contract.",
        parent=deliv_node,
        critical=True,
    )
    claim_deliv_finalize = "The second inspection finalizes the insurance contract."
    await evaluator.verify(
        claim=claim_deliv_finalize,
        node=deliv_finalize_leaf,
        sources=insp.sources_inspections,
        additional_instruction=build_url_support_instruction("DeliveryFinalizesInsuranceContract", insp.sources_inspections),
    )

    # Activation condition
    activation_leaf = evaluator.add_leaf(
        id="ActivationCondition",
        desc="States that without the two inspection reports, insurance coverage cannot be activated.",
        parent=insp_node,
        critical=True,
    )
    claim_activation = "Without the two inspection reports, insurance coverage cannot be activated."
    await evaluator.verify(
        claim=claim_activation,
        node=activation_leaf,
        sources=insp.sources_inspections,
        additional_instruction=build_url_support_instruction("ActivationCondition", insp.sources_inspections),
    )


# -------------------------------
# Main Evaluation Function
# -------------------------------
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

    # Create a critical task node under root (root itself is non-critical by design)
    task_node = evaluator.add_parallel(
        id="MoroccoConstructionInsuranceTask",
        desc=(
            "Evaluate identification of Morocco's mandatory construction insurance requirements effective December 2024, "
            "including required insurance types, applicability thresholds/exclusions, coverage and duration, and technical "
            "inspections required for activation."
        ),
        parent=root,
        critical=True,
    )

    # Run extractions
    framework, applicability, trc, rcd, inspections = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_framework(),
            template_class=FrameworkExtraction,
            extraction_name="framework_identification",
        ),
        evaluator.extract(
            prompt=prompt_extract_applicability(),
            template_class=ApplicabilityExtraction,
            extraction_name="applicability_criteria",
        ),
        evaluator.extract(
            prompt=prompt_extract_trc(),
            template_class=TRCCoverageExtraction,
            extraction_name="trc_coverage_details",
        ),
        evaluator.extract(
            prompt=prompt_extract_rcd(),
            template_class=RCDCoverageExtraction,
            extraction_name="rcd_coverage_details",
        ),
        evaluator.extract(
            prompt=prompt_extract_inspections(),
            template_class=InspectionExtraction,
            extraction_name="technical_inspections",
        ),
    )

    # Optional: add a ground truth summary for transparency (not used for scoring directly)
    evaluator.add_ground_truth({
        "expected_insurance_types": ["TRC (Tous Risques Chantier / All Risks Construction)", "RCD (Responsabilité Civile Décennale / Ten-Year Civil Liability)"],
        "effective_month_year": "December 2024",
        "applicability": {
            "residential": "Subject if over 3 floors OR over 800 m²",
            "non_residential": "Mixed-use/industrial/hotel/educational/office subject if covered surface area exceeds 400 m²",
            "exclusions": "Public projects (e.g., roads, ports, dams) excluded"
        },
        "coverage": {
            "TRC": [
                "Applies during construction phase",
                "Covers accidental damage on site",
                "Covers integrated materials and equipment",
                "Covers civil liability to third parties"
            ],
            "RCD": [
                "Takes effect after delivery/completion",
                "Lasts 10 years after delivery",
                "Covers defects affecting structural integrity",
                "Covers soil defects/design flaws rendering building unsuitable"
            ]
        },
        "inspections": {
            "count": 2,
            "pre_construction": ["Occurs before construction begins", "Validates plans", "Validates soil studies", "Validates technical specifications"],
            "delivery": ["Occurs at delivery", "Ensures compliance", "Finalizes insurance contract"],
            "activation": "Coverage cannot be activated without the two inspection reports"
        }
    }, gt_type="expected_requirements")

    # Build and verify subtrees according to rubric
    await verify_regulation_framework(evaluator, task_node, framework)
    await verify_applicability_criteria(evaluator, task_node, applicability)
    await verify_trc_coverage(evaluator, task_node, trc)
    await verify_rcd_coverage(evaluator, task_node, rcd)
    await verify_inspection_requirements(evaluator, task_node, inspections)

    return evaluator.get_summary()