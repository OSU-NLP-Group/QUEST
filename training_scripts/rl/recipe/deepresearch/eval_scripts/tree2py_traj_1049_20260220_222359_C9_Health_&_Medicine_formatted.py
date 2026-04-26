import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pa_academic_medical_center"
TASK_DESCRIPTION = (
    "Identify an academic medical center in the United States that meets ALL of the following 15 requirements:\n\n"
    "1. The medical center must be located in Pennsylvania\n"
    "2. The medical center must have a licensed bed capacity of at least 400 beds\n"
    "3. The medical center must hold Joint Commission Comprehensive Stroke Center certification (not Primary Stroke Center)\n"
    "4. The medical center must have board-certified neurosurgeons available 24 hours a day, 7 days per week\n"
    "5. The medical center must hold Level I Trauma Center verification from the American College of Surgeons\n"
    "6. The medical center must be affiliated with a medical school\n"
    "7. The affiliated medical school must hold LCME accreditation\n"
    "8. The affiliated medical school must offer an MD degree program\n"
    "9. The medical center must operate an ACGME-accredited emergency medicine residency program\n"
    "10. The emergency medicine residency program must be either 36 months or 48 months in length per ACGME requirements\n"
    "11. The emergency department must have attending physicians who are board-certified in emergency medicine by the American Board of Emergency Medicine (ABEM)\n"
    "12. The medical center must provide endovascular thrombectomy services for acute ischemic stroke treatment\n"
    "13. The medical center must have on-site advanced imaging capabilities including MRI/MRA, CT angiography (CTA), and digital subtraction angiography (DSA)\n"
    "14. The medical center must accept Medicare Part A for inpatient hospital services\n"
    "15. The medical center must have at least one active clinical trial registered with the FDA (e.g., ClinicalTrials.gov) for pancreatic cancer treatment that is currently in Phase 2 or Phase 3\n\n"
    "For the identified medical center, provide its official name, city location, and reference URLs verifying each of the 15 requirements."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MedicalCenterInfo(BaseModel):
    official_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    bed_capacity_text: Optional[str] = None


class MedicalSchoolInfo(BaseModel):
    name: Optional[str] = None
    lcme_accreditation_text: Optional[str] = None
    md_program_text: Optional[str] = None


class EmergencyResidencyInfo(BaseModel):
    acgme_accredited_text: Optional[str] = None
    length_text: Optional[str] = None
    abem_certified_text: Optional[str] = None


class StrokeCareInfo(BaseModel):
    stroke_certification_text: Optional[str] = None
    neurosurgeons_247_text: Optional[str] = None
    thrombectomy_text: Optional[str] = None
    imaging_text: Optional[str] = None


class AdministrativeInfo(BaseModel):
    trauma_level_text: Optional[str] = None
    medicare_text: Optional[str] = None
    pancreatic_trial_text: Optional[str] = None
    trial_phase_text: Optional[str] = None


class RequirementSources(BaseModel):
    location_urls: List[str] = Field(default_factory=list)
    bed_capacity_urls: List[str] = Field(default_factory=list)
    stroke_certification_urls: List[str] = Field(default_factory=list)
    neurosurgery_urls: List[str] = Field(default_factory=list)
    thrombectomy_urls: List[str] = Field(default_factory=list)
    imaging_urls: List[str] = Field(default_factory=list)
    trauma_center_urls: List[str] = Field(default_factory=list)
    affiliation_urls: List[str] = Field(default_factory=list)
    lcme_urls: List[str] = Field(default_factory=list)
    md_program_urls: List[str] = Field(default_factory=list)
    em_residency_urls: List[str] = Field(default_factory=list)
    program_length_urls: List[str] = Field(default_factory=list)
    abem_certification_urls: List[str] = Field(default_factory=list)
    medicare_urls: List[str] = Field(default_factory=list)
    clinical_trial_urls: List[str] = Field(default_factory=list)


class PACenterExtraction(BaseModel):
    center: Optional[MedicalCenterInfo] = None
    medical_school: Optional[MedicalSchoolInfo] = None
    emergency_residency: Optional[EmergencyResidencyInfo] = None
    stroke_care: Optional[StrokeCareInfo] = None
    admin: Optional[AdministrativeInfo] = None
    sources: Optional[RequirementSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_center_info() -> str:
    return (
        "Extract structured information for a single academic medical center identified in the answer that purportedly meets all the requirements.\n"
        "Return a JSON object matching this schema:\n"
        "center: { official_name, city, state, bed_capacity_text }\n"
        "medical_school: { name, lcme_accreditation_text, md_program_text }\n"
        "emergency_residency: { acgme_accredited_text, length_text, abem_certified_text }\n"
        "stroke_care: { stroke_certification_text, neurosurgeons_247_text, thrombectomy_text, imaging_text }\n"
        "admin: { trauma_level_text, medicare_text, pancreatic_trial_text, trial_phase_text }\n"
        "sources: {\n"
        "  location_urls, bed_capacity_urls, stroke_certification_urls, neurosurgery_urls, thrombectomy_urls,\n"
        "  imaging_urls, trauma_center_urls, affiliation_urls, lcme_urls, md_program_urls, em_residency_urls,\n"
        "  program_length_urls, abem_certification_urls, medicare_urls, clinical_trial_urls\n"
        "}\n\n"
        "Extraction rules:\n"
        "- official_name: exact hospital/medical center name used in the answer.\n"
        "- city/state: city and state mentioned for the medical center in the answer; state must be extracted as written (e.g., 'PA' or 'Pennsylvania').\n"
        "- bed_capacity_text: the specific bed count phrase from the answer (e.g., '847 licensed beds').\n"
        "- medical_school.name: the affiliated medical school's name.\n"
        "- *_text fields: short phrases from the answer that correspond to each requirement (if present). If not present, null.\n"
        "- For each *_urls list: extract ONLY the actual URLs explicitly cited in the answer for that requirement. Include all URLs mentioned; if none are cited, return an empty list.\n"
        "- Do not invent information; if any field is missing in the answer, return null or an empty list accordingly.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_get_sources(ext: PACenterExtraction) -> RequirementSources:
    return ext.sources or RequirementSources()

def _text_has_digits(text: Optional[str]) -> bool:
    return bool(text and re.search(r"\d", text))

def _is_pa_state(state: Optional[str]) -> bool:
    if not state:
        return False
    s = state.strip().lower()
    return s in {"pa", "pennsylvania"}

def _center_name(ext: PACenterExtraction) -> str:
    return (ext.center.official_name if ext.center and ext.center.official_name else "the medical center")

def _city_state_str(ext: PACenterExtraction) -> str:
    city = ext.center.city if ext.center else None
    state = ext.center.state if ext.center else None
    if city and state:
        return f"{city}, {state}"
    if city:
        return city
    return "Pennsylvania"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_location_checks(evaluator: Evaluator, parent_node, ext: PACenterExtraction) -> None:
    node = evaluator.add_sequential(
        id="requirement_01_pennsylvania_location",
        desc="The medical center is located in Pennsylvania with city and state location provided",
        parent=parent_node,
        critical=True,
    )

    # Existence: city/state provided and state is Pennsylvania
    state_ok = _is_pa_state(ext.center.state if ext.center else None)
    city_ok = bool(ext.center and ext.center.city and ext.center.city.strip())
    evaluator.add_custom_node(
        result=(state_ok and city_ok),
        id="location_details_provided",
        desc="City and Pennsylvania state are provided in the answer",
        parent=node,
        critical=True,
    )

    # Existence: reference URL(s)
    src = _safe_get_sources(ext).location_urls
    evaluator.add_custom_node(
        result=(bool(src)),
        id="location_url_exists",
        desc="Reference URL documenting Pennsylvania location is provided",
        parent=node,
        critical=True,
    )

    # Leaf: verify location with sources
    leaf = evaluator.add_leaf(
        id="location_supported",
        desc="Pennsylvania location is supported by cited sources",
        parent=node,
        critical=True,
    )
    claim = f"{_center_name(ext)} is located in {_city_state_str(ext)} and in the state of Pennsylvania."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=src,
        additional_instruction=(
            "Confirm the hospital/medical center's location is in Pennsylvania. "
            "Accept city forms like 'Philadelphia, PA' or 'Pittsburgh, Pennsylvania'. "
            "Minor naming variations are acceptable."
        ),
    )


async def build_bed_capacity_checks(evaluator: Evaluator, parent_node, ext: PACenterExtraction) -> None:
    node = evaluator.add_sequential(
        id="requirement_02_bed_capacity",
        desc="The medical center has at least 400 licensed beds with specific bed count provided",
        parent=parent_node,
        critical=True,
    )

    # Existence: bed count text has digits (specific count provided)
    bed_text = ext.center.bed_capacity_text if ext.center else None
    evaluator.add_custom_node(
        result=_text_has_digits(bed_text),
        id="bed_count_provided",
        desc="Specific bed count is provided in the answer (contains a number)",
        parent=node,
        critical=True,
    )

    # Existence: reference URL(s)
    src = _safe_get_sources(ext).bed_capacity_urls
    evaluator.add_custom_node(
        result=(bool(src)),
        id="bed_capacity_url_exists",
        desc="Reference URL documenting bed capacity is provided",
        parent=node,
        critical=True,
    )

    # Leaf: verify threshold >= 400
    leaf = evaluator.add_leaf(
        id="bed_capacity_supported",
        desc="At least 400 licensed beds are supported by cited sources",
        parent=node,
        critical=True,
    )
    claim = f"{_center_name(ext)} has a licensed bed capacity of at least 400 beds."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=src,
        additional_instruction=(
            "Check official or reliable sources (e.g., hospital facts page, annual report, or state data) "
            "that indicate the hospital has 400 or more licensed beds."
        ),
    )


async def build_stroke_care_checks(evaluator: Evaluator, parent_node, ext: PACenterExtraction) -> None:
    group = evaluator.add_parallel(
        id="stroke_care_requirements",
        desc="Stroke care certification and capabilities requirements",
        parent=parent_node,
        critical=True,
    )

    # 3. Comprehensive Stroke Center certification (The Joint Commission)
    req3 = evaluator.add_sequential(
        id="requirement_03_comprehensive_stroke_center",
        desc="The medical center holds Joint Commission Comprehensive Stroke Center certification (not Primary Stroke Center)",
        parent=group,
        critical=True,
    )
    src3 = _safe_get_sources(ext).stroke_certification_urls
    evaluator.add_custom_node(
        result=(bool(src3)),
        id="stroke_certification_url_exists",
        desc="Reference URL from Joint Commission or official source documenting Comprehensive Stroke Center certification",
        parent=req3,
        critical=True,
    )
    leaf3 = evaluator.add_leaf(
        id="stroke_certification_supported",
        desc="Comprehensive Stroke Center certification is supported by cited sources",
        parent=req3,
        critical=True,
    )
    claim3 = f"{_center_name(ext)} is certified by The Joint Commission as a Comprehensive Stroke Center."
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=src3,
        additional_instruction=(
            "Confirm the certification is specifically 'Comprehensive Stroke Center' by The Joint Commission. "
            "Do not accept 'Primary Stroke Center' or other levels."
        ),
    )

    # 4. Board-certified neurosurgeons available 24/7
    req4 = evaluator.add_sequential(
        id="requirement_04_neurosurgical_services_24_7",
        desc="The medical center has board-certified neurosurgeons available 24 hours a day, 7 days per week",
        parent=group,
        critical=True,
    )
    src4 = _safe_get_sources(ext).neurosurgery_urls
    evaluator.add_custom_node(
        result=(bool(src4)),
        id="neurosurgery_url_exists",
        desc="Reference URL documenting 24/7 neurosurgical services",
        parent=req4,
        critical=True,
    )
    leaf4 = evaluator.add_leaf(
        id="neurosurgery_247_supported",
        desc="24/7 board-certified neurosurgeons are supported by cited sources",
        parent=req4,
        critical=True,
    )
    claim4 = f"{_center_name(ext)} has board-certified neurosurgeons available 24/7."
    await evaluator.verify(
        claim=claim4,
        node=leaf4,
        sources=src4,
        additional_instruction=(
            "Look for explicit statements that neurosurgeons are available 24 hours a day, 7 days per week, "
            "and that they are board-certified."
        ),
    )

    # 12. Endovascular thrombectomy services
    req12 = evaluator.add_sequential(
        id="requirement_12_endovascular_thrombectomy",
        desc="The medical center provides endovascular thrombectomy services for acute ischemic stroke",
        parent=group,
        critical=True,
    )
    src12 = _safe_get_sources(ext).thrombectomy_urls
    evaluator.add_custom_node(
        result=(bool(src12)),
        id="thrombectomy_url_exists",
        desc="Reference URL documenting thrombectomy services",
        parent=req12,
        critical=True,
    )
    leaf12 = evaluator.add_leaf(
        id="thrombectomy_supported",
        desc="Endovascular thrombectomy services are supported by cited sources",
        parent=req12,
        critical=True,
    )
    claim12 = f"{_center_name(ext)} provides endovascular thrombectomy for acute ischemic stroke."
    await evaluator.verify(
        claim=claim12,
        node=leaf12,
        sources=src12,
        additional_instruction="Confirm that mechanical/endovascular thrombectomy for ischemic stroke is available.",
    )

    # 13. Advanced imaging capabilities (MRI/MRA, CTA, DSA)
    req13 = evaluator.add_parallel(
        id="requirement_13_advanced_imaging",
        desc="The medical center has on-site MRI/MRA, CT angiography, and digital subtraction angiography",
        parent=group,
        critical=True,
    )
    src13 = _safe_get_sources(ext).imaging_urls
    evaluator.add_custom_node(
        result=(bool(src13)),
        id="imaging_url_exists",
        desc="Reference URL documenting advanced imaging capabilities is provided",
        parent=req13,
        critical=True,
    )

    # Separate leaves for each modality for clearer debugging
    leaf13_mri = evaluator.add_leaf(
        id="advanced_imaging_mri_mra_supported",
        desc="On-site MRI/MRA capability is supported by cited sources",
        parent=req13,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_center_name(ext)} has on-site MRI/MRA capability.",
        node=leaf13_mri,
        sources=src13,
        additional_instruction="Verify that MRI and/or MRA services are available on-site.",
    )

    leaf13_cta = evaluator.add_leaf(
        id="advanced_imaging_cta_supported",
        desc="On-site CT angiography (CTA) capability is supported by cited sources",
        parent=req13,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_center_name(ext)} provides on-site CT angiography (CTA).",
        node=leaf13_cta,
        sources=src13,
        additional_instruction="Verify that CT angiography is available on-site.",
    )

    leaf13_dsa = evaluator.add_leaf(
        id="advanced_imaging_dsa_supported",
        desc="On-site digital subtraction angiography (DSA) capability is supported by cited sources",
        parent=req13,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{_center_name(ext)} provides on-site digital subtraction angiography (DSA).",
        node=leaf13_dsa,
        sources=src13,
        additional_instruction="Verify that DSA (digital subtraction angiography) is available on-site.",
    )


async def build_trauma_checks(evaluator: Evaluator, parent_node, ext: PACenterExtraction) -> None:
    node = evaluator.add_sequential(
        id="requirement_05_level_one_trauma_center",
        desc="The medical center holds Level I Trauma Center verification from the American College of Surgeons",
        parent=parent_node,
        critical=True,
    )
    src = _safe_get_sources(ext).trauma_center_urls
    evaluator.add_custom_node(
        result=(bool(src)),
        id="trauma_center_url_exists",
        desc="Reference URL from ACS or state authority documenting Level I Trauma Center designation",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="trauma_center_supported",
        desc="Level I Trauma Center verification is supported by cited sources",
        parent=node,
        critical=True,
    )
    claim = f"{_center_name(ext)} is verified by the American College of Surgeons as a Level I Trauma Center."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=src,
        additional_instruction=(
            "Prefer ACS verification listings. State authoritative listings are acceptable if they reflect ACS verification. "
            "Confirm Level I status specifically."
        ),
    )


async def build_med_school_checks(evaluator: Evaluator, parent_node, ext: PACenterExtraction) -> None:
    group = evaluator.add_parallel(
        id="medical_school_requirements",
        desc="Medical school affiliation and accreditation requirements",
        parent=parent_node,
        critical=True,
    )

    # 6. Affiliation with a medical school (school identified)
    req6 = evaluator.add_sequential(
        id="requirement_06_medical_school_affiliation",
        desc="The medical center is affiliated with a medical school (school name identified)",
        parent=group,
        critical=True,
    )
    school_name_ok = bool(ext.medical_school and ext.medical_school.name and ext.medical_school.name.strip())
    src6 = _safe_get_sources(ext).affiliation_urls
    evaluator.add_custom_node(
        result=school_name_ok,
        id="affiliation_school_name_provided",
        desc="Affiliated medical school name is provided in the answer",
        parent=req6,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(bool(src6)),
        id="affiliation_url_exists",
        desc="Reference URL documenting medical school affiliation is provided",
        parent=req6,
        critical=True,
    )
    leaf6 = evaluator.add_leaf(
        id="affiliation_supported",
        desc="Medical school affiliation is supported by cited sources",
        parent=req6,
        critical=True,
    )
    school_name = ext.medical_school.name if ext.medical_school else "a medical school"
    claim6 = f"{_center_name(ext)} is affiliated with {school_name}."
    await evaluator.verify(
        claim=claim6,
        node=leaf6,
        sources=src6,
        additional_instruction="Confirm an affiliation/teaching relationship between the hospital and the named medical school.",
    )

    # 7. LCME accreditation
    req7 = evaluator.add_sequential(
        id="requirement_07_lcme_accreditation",
        desc="The affiliated medical school holds LCME accreditation",
        parent=group,
        critical=True,
    )
    src7 = _safe_get_sources(ext).lcme_urls
    evaluator.add_custom_node(
        result=(bool(src7)),
        id="lcme_url_exists",
        desc="Reference URL documenting LCME accreditation is provided",
        parent=req7,
        critical=True,
    )
    leaf7 = evaluator.add_leaf(
        id="lcme_accreditation_supported",
        desc="LCME accreditation is supported by cited sources",
        parent=req7,
        critical=True,
    )
    claim7 = f"{school_name} is accredited by the LCME."
    await evaluator.verify(
        claim=claim7,
        node=leaf7,
        sources=src7,
        additional_instruction="Prefer LCME official directory listings or authoritative accreditation statements.",
    )

    # 8. MD degree program offered
    req8 = evaluator.add_sequential(
        id="requirement_08_md_degree_program",
        desc="The affiliated medical school offers an MD degree program",
        parent=group,
        critical=True,
    )
    src8 = _safe_get_sources(ext).md_program_urls
    evaluator.add_custom_node(
        result=(bool(src8)),
        id="md_program_url_exists",
        desc="Reference URL documenting MD degree program is provided",
        parent=req8,
        critical=True,
    )
    leaf8 = evaluator.add_leaf(
        id="md_program_supported",
        desc="MD degree program is supported by cited sources",
        parent=req8,
        critical=True,
    )
    claim8 = f"{school_name} offers an MD degree program."
    await evaluator.verify(
        claim=claim8,
        node=leaf8,
        sources=src8,
        additional_instruction="Confirm that the school offers a Doctor of Medicine (MD) degree program.",
    )


async def build_emergency_medicine_checks(evaluator: Evaluator, parent_node, ext: PACenterExtraction) -> None:
    group = evaluator.add_parallel(
        id="emergency_medicine_requirements",
        desc="Emergency medicine residency program and staffing requirements",
        parent=parent_node,
        critical=True,
    )

    # 9. ACGME-accredited EM residency program
    req9 = evaluator.add_sequential(
        id="requirement_09_acgme_em_residency",
        desc="The medical center operates an ACGME-accredited emergency medicine residency program",
        parent=group,
        critical=True,
    )
    src9 = _safe_get_sources(ext).em_residency_urls
    evaluator.add_custom_node(
        result=(bool(src9)),
        id="em_residency_url_exists",
        desc="Reference URL from ACGME or institution documenting EM residency accreditation is provided",
        parent=req9,
        critical=True,
    )
    leaf9 = evaluator.add_leaf(
        id="em_residency_supported",
        desc="ACGME-accredited EM residency is supported by cited sources",
        parent=req9,
        critical=True,
    )
    claim9 = f"{_center_name(ext)} operates an ACGME-accredited emergency medicine residency program."
    await evaluator.verify(
        claim=claim9,
        node=leaf9,
        sources=src9,
        additional_instruction="Prefer ACGME listings or official program pages that explicitly state ACGME accreditation.",
    )

    # 10. EM residency length 36 or 48 months
    req10 = evaluator.add_sequential(
        id="requirement_10_em_residency_length",
        desc="The emergency medicine residency program is 36 months or 48 months in length per ACGME requirements",
        parent=group,
        critical=True,
    )
    src10 = _safe_get_sources(ext).program_length_urls
    evaluator.add_custom_node(
        result=(bool(src10)),
        id="program_length_url_exists",
        desc="Reference URL documenting EM residency program length is provided",
        parent=req10,
        critical=True,
    )
    leaf10 = evaluator.add_leaf(
        id="em_residency_length_supported",
        desc="EM residency length (36 or 48 months) is supported by cited sources",
        parent=req10,
        critical=True,
    )
    claim10 = (
        f"The emergency medicine residency program at {_center_name(ext)} is either 36 months or 48 months in length."
    )
    await evaluator.verify(
        claim=claim10,
        node=leaf10,
        sources=src10,
        additional_instruction="Confirm program duration is 36 or 48 months according to official program or ACGME description.",
    )

    # 11. ABEM-certified ED attending physicians
    req11 = evaluator.add_sequential(
        id="requirement_11_abem_certified_physicians",
        desc="The emergency department has attending physicians who are board-certified in emergency medicine by ABEM",
        parent=group,
        critical=True,
    )
    src11 = _safe_get_sources(ext).abem_certification_urls
    evaluator.add_custom_node(
        result=(bool(src11)),
        id="abem_certification_url_exists",
        desc="Reference URL documenting ABEM-certified emergency physicians is provided",
        parent=req11,
        critical=True,
    )
    leaf11 = evaluator.add_leaf(
        id="abem_certified_supported",
        desc="ABEM board-certified ED attendings are supported by cited sources",
        parent=req11,
        critical=True,
    )
    claim11 = f"The emergency department at {_center_name(ext)} has attending physicians who are ABEM board-certified."
    await evaluator.verify(
        claim=claim11,
        node=leaf11,
        sources=src11,
        additional_instruction="Look for explicit mention of ABEM board certification among ED attendings.",
    )


async def build_medicare_checks(evaluator: Evaluator, parent_node, ext: PACenterExtraction) -> None:
    node = evaluator.add_sequential(
        id="requirement_14_medicare_acceptance",
        desc="The medical center accepts Medicare Part A for inpatient services",
        parent=parent_node,
        critical=True,
    )
    src = _safe_get_sources(ext).medicare_urls
    evaluator.add_custom_node(
        result=(bool(src)),
        id="medicare_url_exists",
        desc="Reference URL documenting Medicare participation is provided",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="medicare_part_a_supported",
        desc="Medicare Part A acceptance for inpatient services is supported by cited sources",
        parent=node,
        critical=True,
    )
    claim = f"{_center_name(ext)} accepts Medicare Part A for inpatient hospital services."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=src,
        additional_instruction="Confirm participation in Medicare Part A for inpatient services on official or authoritative pages.",
    )


async def build_clinical_trial_checks(evaluator: Evaluator, parent_node, ext: PACenterExtraction) -> None:
    node = evaluator.add_sequential(
        id="requirement_15_pancreatic_cancer_trial",
        desc="The medical center has at least one active FDA-registered clinical trial for pancreatic cancer in Phase 2 or Phase 3",
        parent=parent_node,
        critical=True,
    )
    src = _safe_get_sources(ext).clinical_trial_urls
    evaluator.add_custom_node(
        result=(bool(src)),
        id="clinical_trial_url_exists",
        desc="Reference URL from ClinicalTrials.gov or FDA documenting pancreatic cancer trial is provided",
        parent=node,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="pancreatic_trial_supported",
        desc="Active Phase 2/3 pancreatic cancer trial is supported by cited sources",
        parent=node,
        critical=True,
    )
    claim = (
        f"{_center_name(ext)} has at least one active clinical trial registered with the FDA/ClinicalTrials.gov for pancreatic cancer "
        "treatment that is currently in Phase 2 or Phase 3."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=src,
        additional_instruction=(
            "Confirm the trial is for pancreatic cancer, is currently active (e.g., recruiting/active status), "
            "and the phase is 2 or 3. ClinicalTrials.gov or FDA listings preferred."
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
    Evaluate an answer for the Pennsylvania academic medical center requirements.
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

    # Extract structured information from the answer
    ext: PACenterExtraction = await evaluator.extract(
        prompt=prompt_extract_center_info(),
        template_class=PACenterExtraction,
        extraction_name="pa_center_extraction",
    )

    # Add a critical wrapper node to represent "all requirements must be met"
    all_req_node = evaluator.add_parallel(
        id="all_requirements",
        desc="All 15 requirements must be met for the selected academic medical center",
        parent=root,
        critical=True,
    )

    # Build verification subtrees
    await build_location_checks(evaluator, all_req_node, ext)
    await build_bed_capacity_checks(evaluator, all_req_node, ext)
    await build_stroke_care_checks(evaluator, all_req_node, ext)
    await build_trauma_checks(evaluator, all_req_node, ext)
    await build_med_school_checks(evaluator, all_req_node, ext)
    await build_emergency_medicine_checks(evaluator, all_req_node, ext)
    await build_medicare_checks(evaluator, all_req_node, ext)
    await build_clinical_trial_checks(evaluator, all_req_node, ext)

    # Return structured summary
    return evaluator.get_summary()