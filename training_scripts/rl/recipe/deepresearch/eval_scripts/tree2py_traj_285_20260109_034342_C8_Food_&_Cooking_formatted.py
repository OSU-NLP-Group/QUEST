import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sacramento_restaurant_compliance"
TASK_DESCRIPTION = (
    "You are preparing to open a new full-service restaurant with 100 seats in Sacramento, California. "
    "Before you can legally open for business, you must comply with numerous state and local regulatory requirements. "
    "Provide a comprehensive compliance report that identifies and documents the following mandatory requirements:\n\n"
    "1. What business licenses and permits must be obtained?\n"
    "2. What food safety certifications must employees have?\n"
    "3. What are the required refrigerator and freezer temperature ranges for food storage?\n"
    "4. What equipment certification standards must commercial kitchen equipment meet?\n"
    "5. What is the required maintenance interval for fire suppression systems?\n"
    "6. What are the handwashing station requirements regarding water temperature and employee ratio?\n"
    "7. What percentage of seating must be ADA-accessible, and what is the minimum doorway width?\n"
    "8. What grease management systems are required?\n"
    "9. What insurance is legally required?\n"
    "10. What is the minimum passing score for health department inspections?\n"
    "11. What are the safety clearance requirements for commercial kitchen ventilation hoods?\n\n"
    "For each requirement, provide the specific regulatory standard or threshold that must be met, along with supporting reference sources."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SectionLicensing(BaseModel):
    business_license_statement: Optional[str] = None
    food_service_permit_statement: Optional[str] = None
    sellers_permit_statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SectionCertifications(BaseModel):
    food_handler_card_requirement_statement: Optional[str] = None  # e.g., "within 30 days of hire, pass exam ≥70%"
    food_protection_manager_requirement_statement: Optional[str] = None  # e.g., "ANAB-accredited certification required for at least one person"
    sources: List[str] = Field(default_factory=list)


class SectionColdStorage(BaseModel):
    refrigerator_temperature_statement: Optional[str] = None  # e.g., "40°F or below"
    freezer_temperature_statement: Optional[str] = None  # e.g., "0°F"
    perishable_food_time_limit_statement: Optional[str] = None  # e.g., "refrigerate within 2 hours (1 hour if ambient >90°F)"
    sources: List[str] = Field(default_factory=list)


class SectionEquipment(BaseModel):
    nsf_ansi_standards_statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SectionFireSuppression(BaseModel):
    maintenance_interval_statement: Optional[str] = None  # e.g., "at least every 6 months"
    sources: List[str] = Field(default_factory=list)


class SectionHandwashing(BaseModel):
    warm_water_duration_statement: Optional[str] = None  # e.g., "warm water available for at least 15 seconds"
    station_to_employee_ratio_statement: Optional[str] = None  # e.g., "≥1 station per 20 employees"
    sources: List[str] = Field(default_factory=list)


class SectionADA(BaseModel):
    accessible_seating_percentage_statement: Optional[str] = None  # e.g., "≥5% seating ADA-accessible"
    minimum_doorway_width_statement: Optional[str] = None  # e.g., "≥32 inches"
    sources: List[str] = Field(default_factory=list)


class SectionGreaseManagement(BaseModel):
    grease_traps_or_interceptors_statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SectionInsurance(BaseModel):
    workers_comp_requirement_statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SectionHealthInspection(BaseModel):
    grade_a_passing_score_statement: Optional[str] = None  # e.g., "Grade A (90-100) required to pass"
    sources: List[str] = Field(default_factory=list)


class SectionVentilation(BaseModel):
    hood_clearance_from_combustibles_statement: Optional[str] = None  # e.g., "≥18 inches"
    sources: List[str] = Field(default_factory=list)


class ComplianceExtraction(BaseModel):
    licensing: SectionLicensing = Field(default_factory=SectionLicensing)
    certifications: SectionCertifications = Field(default_factory=SectionCertifications)
    cold_storage: SectionColdStorage = Field(default_factory=SectionColdStorage)
    equipment: SectionEquipment = Field(default_factory=SectionEquipment)
    fire_suppression: SectionFireSuppression = Field(default_factory=SectionFireSuppression)
    handwashing: SectionHandwashing = Field(default_factory=SectionHandwashing)
    ada_access: SectionADA = Field(default_factory=SectionADA)
    grease_management: SectionGreaseManagement = Field(default_factory=SectionGreaseManagement)
    insurance: SectionInsurance = Field(default_factory=SectionInsurance)
    health_inspection: SectionHealthInspection = Field(default_factory=SectionHealthInspection)
    ventilation: SectionVentilation = Field(default_factory=SectionVentilation)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_compliance() -> str:
    return (
        "Extract the compliance information explicitly stated in the answer for the Sacramento, CA restaurant task. "
        "For each section, capture the specific statement text (as quoted or summarized directly from the answer) and all supporting reference URLs. "
        "Return JSON with the following structure:\n\n"
        "{\n"
        '  "licensing": {\n'
        '    "business_license_statement": string|null,\n'
        '    "food_service_permit_statement": string|null,\n'
        '    "sellers_permit_statement": string|null,\n'
        '    "sources": [url, ...]\n'
        "  },\n"
        '  "certifications": {\n'
        '    "food_handler_card_requirement_statement": string|null,\n'
        '    "food_protection_manager_requirement_statement": string|null,\n'
        '    "sources": [url, ...]\n'
        "  },\n"
        '  "cold_storage": {\n'
        '    "refrigerator_temperature_statement": string|null,\n'
        '    "freezer_temperature_statement": string|null,\n'
        '    "perishable_food_time_limit_statement": string|null,\n'
        '    "sources": [url, ...]\n'
        "  },\n"
        '  "equipment": {\n'
        '    "nsf_ansi_standards_statement": string|null,\n'
        '    "sources": [url, ...]\n'
        "  },\n"
        '  "fire_suppression": {\n'
        '    "maintenance_interval_statement": string|null,\n'
        '    "sources": [url, ...]\n'
        "  },\n"
        '  "handwashing": {\n'
        '    "warm_water_duration_statement": string|null,\n'
        '    "station_to_employee_ratio_statement": string|null,\n'
        '    "sources": [url, ...]\n'
        "  },\n"
        '  "ada_access": {\n'
        '    "accessible_seating_percentage_statement": string|null,\n'
        '    "minimum_doorway_width_statement": string|null,\n'
        '    "sources": [url, ...]\n'
        "  },\n"
        '  "grease_management": {\n'
        '    "grease_traps_or_interceptors_statement": string|null,\n'
        '    "sources": [url, ...]\n'
        "  },\n"
        '  "insurance": {\n'
        '    "workers_comp_requirement_statement": string|null,\n'
        '    "sources": [url, ...]\n'
        "  },\n"
        '  "health_inspection": {\n'
        '    "grade_a_passing_score_statement": string|null,\n'
        '    "sources": [url, ...]\n'
        "  },\n"
        '  "ventilation": {\n'
        '    "hood_clearance_from_combustibles_statement": string|null,\n'
        '    "sources": [url, ...]\n'
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- Extract only what is explicitly present in the answer.\n"
        "- For URLs, extract actual URLs shown (including markdown links). Do not invent URLs.\n"
        "- If a statement is missing, set it to null. If no sources are cited for a section, return an empty array.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


# --------------------------------------------------------------------------- #
# Section verification builders                                                #
# --------------------------------------------------------------------------- #
async def build_business_licenses_and_permits(
    evaluator: Evaluator,
    parent_node,
    data: SectionLicensing,
) -> None:
    section_node = evaluator.add_parallel(
        id="Business_Licenses_and_Permits",
        desc="Required business licenses and permits are identified and sourced.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Business license requirement stated
    leaf_bl = evaluator.add_leaf(
        id="Business_License_Requirement",
        desc="States that a business license must be obtained from the local city or county jurisdiction.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that a business license must be obtained from the local city or county jurisdiction "
            "(e.g., City of Sacramento business license or county license/operations tax certificate)."
        ),
        node=leaf_bl,
        additional_instruction="Judge based on the answer text; allow equivalent phrasing such as 'city business license' or 'local business license'."
    )

    # Leaf: Food service establishment permit requirement stated
    leaf_fsep = evaluator.add_leaf(
        id="Food_Service_Establishment_Permit_Requirement",
        desc="States that a Food Service Establishment Permit must be obtained from the local health department.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that a Food Service Establishment Permit (or Health Permit) must be obtained from the local health department "
            "(e.g., Sacramento County Environmental Management Department)."
        ),
        node=leaf_fsep,
        additional_instruction="Allow synonyms such as 'health permit' or 'food facility permit'."
    )

    # Leaf: Seller's permit requirement stated
    leaf_sp = evaluator.add_leaf(
        id="Sellers_Permit_Requirement",
        desc="States that a Seller's Permit must be obtained from the California Department of Tax and Fee Administration.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that a seller's permit must be obtained from the California Department of Tax and Fee Administration (CDTFA)."
        ),
        node=leaf_sp,
        additional_instruction="Allow 'CDTFA seller’s permit' phrasing and reasonable variants."
    )

    # Leaf: Licensing references provided (existence check)
    evaluator.add_custom_node(
        result=_has_sources(data.sources),
        id="Licensing_References",
        desc="Provides at least one supporting reference URL for the licensing/permit requirements stated.",
        parent=section_node,
        critical=True,
    )


async def build_food_safety_certifications(
    evaluator: Evaluator,
    parent_node,
    data: SectionCertifications,
) -> None:
    section_node = evaluator.add_parallel(
        id="Food_Safety_Certifications",
        desc="Required food safety certifications are identified and sourced.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Food Handler Card requirement
    leaf_fhc = evaluator.add_leaf(
        id="Food_Handler_Card_Requirement",
        desc="States that all food-handling employees must obtain a Food Handler Card within 30 days of hire by passing an exam with at least a 70% score.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that all food-handling employees must obtain a Food Handler Card within 30 days of hire by passing an exam with at least a 70% score."
        ),
        node=leaf_fhc,
        additional_instruction="Focus on presence of timing (30 days) and minimum passing score (≥70%)."
    )

    # Leaf: Food Protection Manager Certification requirement
    leaf_fpm = evaluator.add_leaf(
        id="Food_Protection_Manager_Requirement",
        desc="States that at least one person must hold a Food Protection Manager Certification from an ANAB-accredited program.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that at least one person must hold a Food Protection Manager Certification from an ANAB-accredited program."
        ),
        node=leaf_fpm,
        additional_instruction="Allow variants mentioning ANSI/ANAB and equivalent accreditation phrasing."
    )

    # Leaf: Certification references provided (existence check)
    evaluator.add_custom_node(
        result=_has_sources(data.sources),
        id="Certification_References",
        desc="Provides at least one supporting reference URL for the certification requirements stated.",
        parent=section_node,
        critical=True,
    )


async def build_cold_storage_requirements(
    evaluator: Evaluator,
    parent_node,
    data: SectionColdStorage,
) -> None:
    section_node = evaluator.add_parallel(
        id="Food_Storage_Cold_Holding_Requirements",
        desc="Cold storage temperature/time thresholds are stated and sourced.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Refrigerator temperature threshold
    leaf_fridge = evaluator.add_leaf(
        id="Refrigerator_Temperature",
        desc="States that refrigerator temperature must be maintained at 40°F or below.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that refrigerator temperature must be maintained at 40°F or below.",
        node=leaf_fridge,
        additional_instruction="Allow equivalent phrasing such as '≤40°F' or 'at/below 40°F'."
    )

    # Leaf: Freezer temperature threshold
    leaf_freezer = evaluator.add_leaf(
        id="Freezer_Temperature",
        desc="States that freezer temperature must be maintained at 0°F.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that freezer temperature must be maintained at 0°F.",
        node=leaf_freezer,
        additional_instruction="Allow equivalent phrasing such as 'at 0°F' or '0 degrees Fahrenheit'."
    )

    # Leaf: Perishable food time limit
    leaf_time = evaluator.add_leaf(
        id="Perishable_Food_Time_Limit",
        desc="States that perishable foods must be refrigerated within 2 hours (1 hour if ambient temperature exceeds 90°F).",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that perishable foods must be refrigerated within 2 hours (or 1 hour if ambient temperature exceeds 90°F).",
        node=leaf_time,
        additional_instruction="Verify presence of both the 2-hour rule and the 1-hour rule above 90°F."
    )

    # Leaf: Cold storage references provided (existence check)
    evaluator.add_custom_node(
        result=_has_sources(data.sources),
        id="Cold_Storage_References",
        desc="Provides at least one supporting reference URL for the cold storage temperature/time thresholds stated.",
        parent=section_node,
        critical=True,
    )


async def build_equipment_standards(
    evaluator: Evaluator,
    parent_node,
    data: SectionEquipment,
) -> None:
    section_node = evaluator.add_parallel(
        id="Commercial_Equipment_Standards",
        desc="Commercial kitchen equipment certification standards are stated and sourced.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: NSF/ANSI standards stated
    leaf_nsf = evaluator.add_leaf(
        id="NSF_ANSI_Standards",
        desc="States that commercial food equipment should meet applicable NSF/ANSI standards.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that commercial food equipment should meet applicable NSF/ANSI standards.",
        node=leaf_nsf,
        additional_instruction="Allow phrasing mentioning NSF or ANSI standards for food equipment."
    )

    # Leaf: Equipment references provided (existence check)
    evaluator.add_custom_node(
        result=_has_sources(data.sources),
        id="Equipment_Standards_References",
        desc="Provides at least one supporting reference URL for the NSF/ANSI equipment standards statement.",
        parent=section_node,
        critical=True,
    )


async def build_fire_suppression_maintenance(
    evaluator: Evaluator,
    parent_node,
    data: SectionFireSuppression,
) -> None:
    section_node = evaluator.add_parallel(
        id="Fire_Suppression_Maintenance",
        desc="Fire suppression system maintenance interval is stated and sourced.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Maintenance interval
    leaf_interval = evaluator.add_leaf(
        id="Maintenance_Interval",
        desc="States that fire suppression systems must be maintained at least every 6 months.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that fire suppression systems must be maintained at least every 6 months.",
        node=leaf_interval,
        additional_instruction="Allow variants such as 'semiannual inspection/maintenance' or 'twice per year'."
    )

    # Leaf: Fire suppression references provided (existence check)
    evaluator.add_custom_node(
        result=_has_sources(data.sources),
        id="Fire_Suppression_References",
        desc="Provides at least one supporting reference URL for the fire suppression maintenance interval stated.",
        parent=section_node,
        critical=True,
    )


async def build_handwashing_requirements(
    evaluator: Evaluator,
    parent_node,
    data: SectionHandwashing,
) -> None:
    section_node = evaluator.add_parallel(
        id="Handwashing_Station_Requirements",
        desc="Handwashing station thresholds are stated and sourced.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Warm water duration
    leaf_warm = evaluator.add_leaf(
        id="Warm_Water_Duration",
        desc="States that handwashing stations must provide warm water for at least 15 seconds.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that handwashing stations must provide warm water for at least 15 seconds.",
        node=leaf_warm,
        additional_instruction="Allow equivalent phrasing indicating continuous warm water availability for ≥15 seconds."
    )

    # Leaf: Station-to-employee ratio
    leaf_ratio = evaluator.add_leaf(
        id="Station_to_Employee_Ratio",
        desc="States that at least one handwashing station is required per 20 employees.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that at least one handwashing station is required per 20 employees.",
        node=leaf_ratio,
        additional_instruction="Focus on the ratio requirement (≥1 per 20 employees) in the answer."
    )

    # Leaf: Handwashing references provided (existence check)
    evaluator.add_custom_node(
        result=_has_sources(data.sources),
        id="Handwashing_References",
        desc="Provides at least one supporting reference URL for the handwashing requirements stated.",
        parent=section_node,
        critical=True,
    )


async def build_ada_requirements(
    evaluator: Evaluator,
    parent_node,
    data: SectionADA,
) -> None:
    section_node = evaluator.add_parallel(
        id="ADA_Accessibility_Requirements",
        desc="ADA accessibility thresholds are stated and sourced.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Accessible seating percentage
    leaf_pct = evaluator.add_leaf(
        id="Accessible_Seating_Percentage",
        desc="States that at least 5% of restaurant seating must be ADA-accessible.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that at least 5% of restaurant seating must be ADA-accessible.",
        node=leaf_pct,
        additional_instruction="Allow phrasing such as '≥5% seating accessible' or 'five percent of seating'."
    )

    # Leaf: Minimum doorway width
    leaf_width = evaluator.add_leaf(
        id="Minimum_Doorway_Width",
        desc="States that entrance doorways must be at least 32 inches wide for ADA compliance.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that entrance doorways must be at least 32 inches wide for ADA compliance.",
        node=leaf_width,
        additional_instruction="Allow 'minimum clear opening of 32 inches' phrasing."
    )

    # Leaf: ADA references provided (existence check)
    evaluator.add_custom_node(
        result=_has_sources(data.sources),
        id="ADA_References",
        desc="Provides at least one supporting reference URL for the ADA accessibility thresholds stated.",
        parent=section_node,
        critical=True,
    )


async def build_grease_management(
    evaluator: Evaluator,
    parent_node,
    data: SectionGreaseManagement,
) -> None:
    section_node = evaluator.add_parallel(
        id="Grease_Management_Requirements",
        desc="Grease management system requirement is stated and sourced.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Grease traps/interceptors required
    leaf_grease = evaluator.add_leaf(
        id="Grease_Traps_Or_Interceptors",
        desc="States that grease traps or interceptors are required for food service facilities.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that grease traps or interceptors are required for food service facilities.",
        node=leaf_grease,
        additional_instruction="Allow 'FOG (fats, oils, grease) interceptor required' phrasing."
    )

    # Leaf: Grease management references provided (existence check)
    evaluator.add_custom_node(
        result=_has_sources(data.sources),
        id="Grease_Management_References",
        desc="Provides at least one supporting reference URL for the grease management requirement stated.",
        parent=section_node,
        critical=True,
    )


async def build_insurance_requirements(
    evaluator: Evaluator,
    parent_node,
    data: SectionInsurance,
) -> None:
    section_node = evaluator.add_parallel(
        id="Legally_Required_Insurance",
        desc="Legally required insurance is stated and sourced.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Workers' comp required
    leaf_wc = evaluator.add_leaf(
        id="Workers_Comp_Requirement",
        desc="States that workers' compensation insurance is required if the restaurant has at least one employee.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that workers' compensation insurance is required if the restaurant has at least one employee.",
        node=leaf_wc,
        additional_instruction="Allow equivalent phrasing such as 'must carry workers’ comp when employing ≥1 person'."
    )

    # Leaf: Insurance references provided (existence check)
    evaluator.add_custom_node(
        result=_has_sources(data.sources),
        id="Insurance_References",
        desc="Provides at least one supporting reference URL for the legally required insurance stated.",
        parent=section_node,
        critical=True,
    )


async def build_health_inspection_standard(
    evaluator: Evaluator,
    parent_node,
    data: SectionHealthInspection,
) -> None:
    section_node = evaluator.add_parallel(
        id="Health_Inspection_Passing_Standard",
        desc="Minimum passing standard for health inspection is stated and sourced.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Grade A passing score
    leaf_grade = evaluator.add_leaf(
        id="Grade_A_Passing_Score",
        desc="States that the restaurant must achieve a Grade A score (90-100 points) to pass health department inspection.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the restaurant must achieve a Grade A score (90-100 points) to pass health department inspection.",
        node=leaf_grade,
        additional_instruction="Focus on Grade A range 90–100 as the passing threshold."
    )

    # Leaf: Health inspection references provided (existence check)
    evaluator.add_custom_node(
        result=_has_sources(data.sources),
        id="Health_Inspection_References",
        desc="Provides at least one supporting reference URL for the health inspection passing standard stated.",
        parent=section_node,
        critical=True,
    )


async def build_kitchen_ventilation_clearance(
    evaluator: Evaluator,
    parent_node,
    data: SectionVentilation,
) -> None:
    section_node = evaluator.add_parallel(
        id="Kitchen_Ventilation_Safety_Clearance",
        desc="Ventilation hood clearance requirement is stated and sourced.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Hood clearance from combustibles
    leaf_clearance = evaluator.add_leaf(
        id="Hood_Clearance_From_Combustibles",
        desc="States that commercial kitchen ventilation hoods must be installed at least 18 inches away from combustible materials.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that commercial kitchen ventilation hoods must be installed at least 18 inches away from combustible materials.",
        node=leaf_clearance,
        additional_instruction="Allow 'minimum clearance of 18 inches from combustibles' phrasing."
    )

    # Leaf: Ventilation clearance references provided (existence check)
    evaluator.add_custom_node(
        result=_has_sources(data.sources),
        id="Ventilation_Clearance_References",
        desc="Provides at least one supporting reference URL for the ventilation hood clearance requirement stated.",
        parent=section_node,
        critical=True,
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
) -> Dict:
    """
    Evaluate the compliance report answer for opening a full-service restaurant in Sacramento, CA.
    Builds a verification tree with critical checks per the rubric and returns a structured summary.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    _ = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root default; we will add a critical section node under it
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

    # Add context info
    evaluator.add_custom_info(
        info={"city": "Sacramento, CA", "restaurant_type": "Full-service", "seat_count": 100},
        info_type="context",
        info_name="compliance_context",
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_compliance(),
        template_class=ComplianceExtraction,
        extraction_name="compliance_extraction",
    )

    # Build the top-level critical compliance node
    compliance_node = evaluator.add_parallel(
        id="Restaurant_Opening_Compliance",
        desc="All mandatory regulatory requirements for opening a full-service restaurant in Sacramento, California are identified, include required thresholds, and include supporting sources.",
        parent=evaluator.root,
        critical=True,
    )

    # Build all sections under the critical compliance node
    await build_business_licenses_and_permits(evaluator, compliance_node, extracted.licensing)
    await build_food_safety_certifications(evaluator, compliance_node, extracted.certifications)
    await build_cold_storage_requirements(evaluator, compliance_node, extracted.cold_storage)
    await build_equipment_standards(evaluator, compliance_node, extracted.equipment)
    await build_fire_suppression_maintenance(evaluator, compliance_node, extracted.fire_suppression)
    await build_handwashing_requirements(evaluator, compliance_node, extracted.handwashing)
    await build_ada_requirements(evaluator, compliance_node, extracted.ada_access)
    await build_grease_management(evaluator, compliance_node, extracted.grease_management)
    await build_insurance_requirements(evaluator, compliance_node, extracted.insurance)
    await build_health_inspection_standard(evaluator, compliance_node, extracted.health_inspection)
    await build_kitchen_ventilation_clearance(evaluator, compliance_node, extracted.ventilation)

    # Return evaluation summary
    return evaluator.get_summary()