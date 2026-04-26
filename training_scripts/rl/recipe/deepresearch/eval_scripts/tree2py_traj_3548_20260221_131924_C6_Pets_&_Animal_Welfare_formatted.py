import asyncio
import logging
from typing import Any, List, Optional, Tuple, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "vethosp_aaha_fl_24hr"
TASK_DESCRIPTION = (
    "Identify two AAHA-accredited veterinary hospitals located in Florida that provide 24-hour emergency services. "
    "For each hospital, you must verify that it offers at least three of the following four specialized services: "
    "(1) Advanced surgery capabilities, (2) Dentistry with full oral radiography, (3) Comprehensive pain management protocols, "
    "and (4) Diagnostic imaging including ultrasound or CT/MRI. For each hospital, provide: the hospital name, complete street "
    "address (including city, state, and ZIP code), contact phone number, verification that the hospital is currently AAHA-accredited, "
    "verification that it provides 24-hour emergency care, detailed verification of at least three specialized services with specific "
    "evidence that each service meets the stated criteria, and reference URLs supporting each piece of information. All information must "
    "be verifiable through publicly accessible sources."
)

ALLOWED_SERVICE_CATEGORIES = [
    "advanced surgery capabilities",
    "dentistry with full oral radiography",
    "comprehensive pain management protocols",
    "diagnostic imaging including ultrasound or CT/MRI",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AddressInfo(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    full_address: Optional[str] = None


class SpecializedService(BaseModel):
    service_type: Optional[str] = None
    evidence_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HospitalRecord(BaseModel):
    name: Optional[str] = None
    address: Optional[AddressInfo] = None
    phone: Optional[str] = None

    basic_info_urls: List[str] = Field(default_factory=list)

    aaha_urls: List[str] = Field(default_factory=list)

    emergency_urls: List[str] = Field(default_factory=list)

    specialized_services: List[SpecializedService] = Field(default_factory=list)


class HospitalsExtraction(BaseModel):
    hospitals: List[HospitalRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hospitals() -> str:
    return (
        "Extract from the answer up to two veterinary hospitals located in Florida that are claimed to be AAHA-accredited "
        "and provide 24-hour emergency services. For each hospital, return a JSON object with the following fields:\n"
        "- name: The hospital name as stated.\n"
        "- address: An object with fields street, city, state, zip_code, and full_address. If parts are missing, set them to null.\n"
        "- phone: The contact phone number as stated in the answer.\n"
        "- basic_info_urls: An array of URLs cited for the hospital’s basic information/address.\n"
        "- aaha_urls: An array of URLs specifically cited to verify AAHA accreditation (prefer AAHA.org locator pages, if present).\n"
        "- emergency_urls: An array of URLs cited to verify 24-hour emergency services.\n"
        "- specialized_services: An array (up to 4 items) of objects, each with:\n"
        "  • service_type: A short phrase indicating which of the four categories it belongs to. Map to one of:\n"
        "    'advanced surgery capabilities', 'dentistry with full oral radiography', "
        "'comprehensive pain management protocols', 'diagnostic imaging including ultrasound or CT/MRI'. "
        "Use the closest category based on the answer.\n"
        "  • evidence_text: A short snippet or summary of the evidence mentioned (e.g., board-certified surgeons, full-mouth dental radiographs, multimodal analgesia, ultrasound/CT/MRI availability).\n"
        "  • urls: An array of URLs cited for this specific service.\n"
        "GENERAL RULES:\n"
        "1. Extract ONLY what is explicitly present in the answer. Do not invent or infer missing information.\n"
        "2. If a field is missing, set it to null (or empty array for URLs).\n"
        "3. Include URLs exactly as provided (plain or markdown links). Extract the actual URL string.\n"
        "4. For addresses, attempt to fill street, city, state, and ZIP code. If the answer provides a full address line, copy it to full_address and parse components when possible.\n"
        "Return a JSON object with a single key 'hospitals' containing an array of such hospital objects (up to 2)."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _desc_hosp(index: int) -> str:
    return "First AAHA-accredited veterinary hospital meeting all requirements" if index == 0 else "Second AAHA-accredited veterinary hospital meeting all requirements"


def _service_desc(index: int, s_idx: int) -> str:
    base = ["First", "Second", "Third", "Fourth"]
    return f"{base[s_idx]} specialized service offered by Hospital {index + 1}"


def _build_service_claim(hospital_name: str, service_type: Optional[str]) -> str:
    t = (service_type or "").strip().lower()
    if "dent" in t and ("radio" in t or "x-ray" in t):
        return f"The hospital {hospital_name} offers dentistry with full oral radiography (digital dental X-rays/full-mouth radiographs)."
    if "pain" in t or "analges" in t:
        return f"The hospital {hospital_name} provides comprehensive pain management protocols (e.g., multimodal analgesia plans or dedicated pain management services)."
    if "diagnostic" in t or "ultrasound" in t or "ct" in t or "mri" in t or "imaging" in t:
        return f"The hospital {hospital_name} offers diagnostic imaging that includes ultrasound or CT/MRI."
    # Default to advanced surgery if ambiguous mentions of surgery/surgeons/suite appear
    return f"The hospital {hospital_name} offers advanced surgery capabilities (e.g., board-certified surgeons or advanced surgical suite for complex procedures)."


def _type_check_instruction() -> str:
    return (
        "Decide whether the provided service_type text clearly maps to one of the four allowed categories:\n"
        "1) advanced surgery capabilities\n"
        "2) dentistry with full oral radiography\n"
        "3) comprehensive pain management protocols\n"
        "4) diagnostic imaging including ultrasound or CT/MRI\n"
        "Return Correct if it matches or is an obvious synonym; otherwise Incorrect."
    )


def _evidence_instruction(service_type: Optional[str]) -> str:
    st = (service_type or "").strip().lower()
    if "dent" in st and ("radio" in st or "x-ray" in st):
        return "Verify that the page explicitly supports dentistry with full oral radiography, e.g., mentions full-mouth dental radiographs, dental X-rays, or digital dental radiography."
    if "pain" in st or "analges" in st:
        return "Verify explicit support for comprehensive pain management protocols, such as multimodal analgesia plans, dedicated pain services, or stated protocols/policies."
    if "diagnostic" in st or "ultrasound" in st or "ct" in st or "mri" in st or "imaging" in st:
        return "Verify explicit support for diagnostic imaging including ultrasound or CT/MRI; the page should mention at least one of ultrasound, CT, or MRI."
    return "Verify explicit support for advanced surgery capabilities; evidence examples include a surgical suite, board-certified surgeons, or advanced/complex surgical procedures being offered."


def _aaha_instruction() -> str:
    return (
        "Confirm AAHA accreditation from the provided URLs. Strong evidence includes the AAHA.org locator page showing the hospital as 'AAHA Accredited'. "
        "If only the hospital's own site is provided, accept if it clearly states AAHA accreditation."
    )


def _emergency_instruction() -> str:
    return (
        "Confirm that the hospital provides 24-hour emergency care (open 24/7 or 24/7/365). Accept reasonable variants such as 'open 24 hours', 'around-the-clock emergency', or '24/7 emergency'."
    )


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def _verify_basic_info(
    evaluator: Evaluator,
    parent: VerificationNode,
    hospital: HospitalRecord,
    index: int
) -> VerificationNode:
    node = evaluator.add_parallel(
        id=f"H{index + 1}_Basic_Information",
        desc="Basic identification information for Hospital 1, including verification of Florida location through address" if index == 0
             else "Basic identification information for Hospital 2, including verification of Florida location through address",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(hospital.name and hospital.name.strip()),
        id=f"H{index + 1}_Name",
        desc="Hospital name is provided",
        parent=node,
        critical=True
    )

    addr = hospital.address or AddressInfo()
    has_components = bool(addr.street and addr.city and addr.state and addr.zip_code)
    is_fl = (addr.state or "").strip().lower() in ("fl", "florida")

    evaluator.add_custom_node(
        result=has_components and is_fl,
        id=f"H{index + 1}_Address",
        desc="Complete street address is provided including street, city, state (must be Florida/FL), and ZIP code",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(hospital.phone and hospital.phone.strip()),
        id=f"H{index + 1}_Phone",
        desc="Contact phone number is provided",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(hospital.basic_info_urls),
        id=f"H{index + 1}_Basic_Info_URL",
        desc="Reference URL for basic information and location verification",
        parent=node,
        critical=True
    )
    return node


async def _verify_aaha(
    evaluator: Evaluator,
    parent: VerificationNode,
    hospital: HospitalRecord,
    index: int
) -> VerificationNode:
    node = evaluator.add_parallel(
        id=f"H{index + 1}_AAHA_Accreditation",
        desc="Verification that Hospital 1 is AAHA-accredited" if index == 0 else "Verification that Hospital 2 is AAHA-accredited",
        parent=parent,
        critical=True
    )

    status_leaf = evaluator.add_leaf(
        id=f"H{index + 1}_AAHA_Status",
        desc="Hospital is currently AAHA-accredited (AAHA is the only organization that accredits companion animal hospitals in the U.S. and Canada)",
        parent=node,
        critical=True
    )

    name = hospital.name or "the hospital"
    claim = f"{name} is AAHA-accredited."

    await evaluator.verify(
        claim=claim,
        node=status_leaf,
        sources=hospital.aaha_urls if hospital.aaha_urls else None,
        additional_instruction=_aaha_instruction()
    )

    evaluator.add_custom_node(
        result=bool(hospital.aaha_urls),
        id=f"H{index + 1}_AAHA_URL",
        desc="Reference URL verifying AAHA accreditation status",
        parent=node,
        critical=True
    )

    return node


async def _verify_emergency(
    evaluator: Evaluator,
    parent: VerificationNode,
    hospital: HospitalRecord,
    index: int
) -> VerificationNode:
    node = evaluator.add_parallel(
        id=f"H{index + 1}_Emergency_Services",
        desc="Verification that Hospital 1 provides 24-hour emergency services" if index == 0 else "Verification that Hospital 2 provides 24-hour emergency services",
        parent=parent,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id=f"H{index + 1}_24Hour_Availability",
        desc="Hospital provides 24-hour emergency care (open 24/7/365)",
        parent=node,
        critical=True
    )

    name = hospital.name or "the hospital"
    claim = f"{name} provides 24-hour emergency care (open 24/7)."

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=hospital.emergency_urls if hospital.emergency_urls else None,
        additional_instruction=_emergency_instruction()
    )

    evaluator.add_custom_node(
        result=bool(hospital.emergency_urls),
        id=f"H{index + 1}_Emergency_URL",
        desc="Reference URL verifying 24-hour emergency services",
        parent=node,
        critical=True
    )

    return node


async def _verify_one_service(
    evaluator: Evaluator,
    parent: VerificationNode,
    hospital: HospitalRecord,
    index: int,
    s_idx: int,
    service: SpecializedService
) -> Tuple[VerificationNode, bool]:
    svc_node = evaluator.add_parallel(
        id=f"H{index + 1}_Service_{s_idx + 1}",
        desc=_service_desc(index, s_idx),
        parent=parent,
        critical=False
    )

    # Type check: simple verification that the label maps to one of the allowed categories.
    type_leaf = evaluator.add_leaf(
        id=f"H{index + 1}_S{s_idx + 1}_Type",
        desc="Type of specialized service identified (advanced surgery/dentistry with oral radiography/pain management/diagnostic imaging)",
        parent=svc_node,
        critical=True if s_idx < 3 else False  # prioritize first three services
    )
    type_text = (service.service_type or "").strip()
    type_claim = (
        f"The service_type '{type_text}' clearly maps to one of the allowed categories: "
        f"{', '.join(ALLOWED_SERVICE_CATEGORIES)}."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=None,
        additional_instruction=_type_check_instruction()
    )

    # Evidence verification using URLs
    evidence_leaf = evaluator.add_leaf(
        id=f"H{index + 1}_S{s_idx + 1}_Evidence",
        desc="Specific evidence that this service meets the requirement criteria (e.g., board-certified surgeons, advanced surgical suite, dental radiography, pain management protocols, CT/MRI/ultrasound)",
        parent=svc_node,
        critical=True if s_idx < 3 else False
    )
    name = hospital.name or "the hospital"
    evidence_claim = _build_service_claim(name, service.service_type)

    await evaluator.verify(
        claim=evidence_claim,
        node=evidence_leaf,
        sources=service.urls if service.urls else None,
        additional_instruction=_evidence_instruction(service.service_type)
    )

    url_presence = evaluator.add_custom_node(
        result=bool(service.urls),
        id=f"H{index + 1}_S{s_idx + 1}_URL",
        desc="Reference URL verifying this specialized service",
        parent=svc_node,
        critical=True if s_idx < 3 else False
    )

    svc_pass = (type_leaf.score == 1.0) and (evidence_leaf.score == 1.0) and (url_presence.score == 1.0)
    return svc_node, svc_pass


async def _verify_services_and_threshold(
    evaluator: Evaluator,
    parent: VerificationNode,
    hospital: HospitalRecord,
    index: int
) -> Tuple[VerificationNode, int]:
    # Specialized services group (non-critical to allow flexible composition)
    svc_group = evaluator.add_parallel(
        id=f"H{index + 1}_Specialized_Services",
        desc="Verification that Hospital 1 offers at least three of the four specified specialized services (threshold: minimum 3 of 4 service nodes must pass)"
             if index == 0 else
             "Verification that Hospital 2 offers at least three of the four specified specialized services (threshold: minimum 3 of 4 service nodes must pass)",
        parent=parent,
        critical=False
    )

    # Ensure up to 4 services, pad with empty entries if fewer provided
    services = list(hospital.specialized_services[:4])
    while len(services) < 4:
        services.append(SpecializedService())

    passed_count = 0
    for s_idx, svc in enumerate(services):
        _, passed = await _verify_one_service(
            evaluator=evaluator,
            parent=svc_group,
            hospital=hospital,
            index=index,
            s_idx=s_idx,
            service=svc
        )
        if passed:
            passed_count += 1

    # Add a critical threshold gate under the hospital (not inside the group) to enforce >=3 services
    evaluator.add_custom_node(
        result=(passed_count >= 3),
        id=f"H{index + 1}_Min3Of4",
        desc=f"Hospital {index + 1} has at least three specialized services verified and meeting the stated criteria",
        parent=parent,
        critical=True
    )

    return svc_group, passed_count


# --------------------------------------------------------------------------- #
# Main verification per hospital                                              #
# --------------------------------------------------------------------------- #
async def verify_hospital(
    evaluator: Evaluator,
    root: VerificationNode,
    hospital: HospitalRecord,
    index: int
) -> None:
    hosp_node = evaluator.add_parallel(
        id=f"Hospital_{index + 1}",
        desc=_desc_hosp(index),
        parent=root,
        critical=False
    )

    # Basic Info
    await _verify_basic_info(evaluator, hosp_node, hospital, index)

    # AAHA Accreditation
    await _verify_aaha(evaluator, hosp_node, hospital, index)

    # 24-hour Emergency Services
    await _verify_emergency(evaluator, hosp_node, hospital, index)

    # Specialized Services + Threshold
    await _verify_services_and_threshold(evaluator, hosp_node, hospital, index)


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Hospitals evaluated independently
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

    # Extract hospitals data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_hospitals(),
        template_class=HospitalsExtraction,
        extraction_name="hospitals_extraction"
    )

    # Ensure exactly 2 hospitals for evaluation (pad if fewer)
    hospitals = list(extraction.hospitals[:2])
    while len(hospitals) < 2:
        hospitals.append(HospitalRecord())

    # Verify each hospital
    for idx in range(2):
        await verify_hospital(evaluator, root, hospitals[idx], idx)

    # Provide custom info for allowed categories
    evaluator.add_custom_info(
        info={"allowed_specialized_service_categories": ALLOWED_SERVICE_CATEGORIES},
        info_type="allowed_service_categories"
    )

    return evaluator.get_summary()