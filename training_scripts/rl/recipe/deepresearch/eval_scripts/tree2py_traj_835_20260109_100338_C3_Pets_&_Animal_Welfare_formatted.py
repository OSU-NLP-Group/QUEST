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
TASK_ID = "va_raptor_rehab_facility"
TASK_DESCRIPTION = """
Identify a wildlife rehabilitation facility located in Virginia that specializes in or is authorized to rehabilitate raptors (birds of prey such as hawks, eagles, owls, and falcons). The facility must meet all of the following requirements: (1) The facility must be physically located in the state of Virginia and must be explicitly authorized to care for and rehabilitate raptors. (2) The facility must hold a valid Federal Migratory Bird Rehabilitation Permit from the U.S. Fish & Wildlife Service that authorizes the rehabilitation of raptors. (3) The facility must have at least one staff member who holds professional wildlife rehabilitation credentials, specifically a Certified Wildlife Rehabilitator (CWR™) certification from the International Wildlife Rehabilitation Council (IWRC) or equivalent professional wildlife rehabilitation certification recognized in the field. (4) The facility must maintain enclosures and housing that meet minimum standards for raptor rehabilitation as specified in wildlife rehabilitation guidelines, including appropriate cage sizes, perching structures, and protection from environmental elements suitable for birds of prey. (5) The facility must have a documented partnership with a licensed veterinarian who provides veterinary oversight for the animals in their care. Provide the name of the facility and supporting reference URLs that verify each of these five requirements.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacilityExtraction(BaseModel):
    facility_name: Optional[str] = None

    # Evidence URL buckets for each requirement (extract strictly from the answer)
    location_authorization_urls: List[str] = Field(default_factory=list)
    federal_permit_urls: List[str] = Field(default_factory=list)
    staff_credentials_urls: List[str] = Field(default_factory=list)
    housing_standards_urls: List[str] = Field(default_factory=list)
    veterinary_partnership_urls: List[str] = Field(default_factory=list)

    # Optional helpful context from the answer (names/certs mentioned)
    staff_member_names: List[str] = Field(default_factory=list)
    staff_certifications: List[str] = Field(default_factory=list)
    veterinarian_names: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facility_info() -> str:
    return """
    From the answer, extract information for a single wildlife rehabilitation facility (choose the first one mentioned if multiple):
    - facility_name: The name of the facility (string).
    - location_authorization_urls: Array of URLs that support BOTH (a) that the facility is physically located in Virginia, AND (b) that it is explicitly authorized to rehabilitate raptors (birds of prey such as hawks, eagles, owls, falcons).
    - federal_permit_urls: Array of URLs that support the claim that the facility holds a valid Federal Migratory Bird Rehabilitation Permit from the U.S. Fish & Wildlife Service authorizing raptor rehabilitation. Prefer official/credible sources (e.g., USFWS pages or facility pages detailing permit).
    - staff_credentials_urls: Array of URLs that support that at least one staff member holds IWRC Certified Wildlife Rehabilitator (CWR™) or an equivalent professional wildlife rehabilitation certification recognized in the field. If the answer names staff/certs, still only extract URLs explicitly present in the answer.
    - housing_standards_urls: Array of URLs evidencing that the facility maintains raptor enclosures/housing meeting minimum standards per wildlife rehabilitation guidelines (e.g., appropriate cage sizes, suitable perching, protection from elements).
    - veterinary_partnership_urls: Array of URLs evidencing a documented partnership with a licensed veterinarian who provides veterinary oversight for animals in care.

    Additionally, extract helper context if present in the answer:
    - staff_member_names: Array of names of staff members mentioned with credentials (if any).
    - staff_certifications: Array of certification names mentioned (e.g., "IWRC CWR").
    - veterinarian_names: Array of names of veterinarians or veterinary partners (if any).

    IMPORTANT:
    - Extract ONLY URLs explicitly present in the answer (plain URLs, markdown links, etc.). Do not invent URLs.
    - If a field cannot be found, set it to null (for facility_name) or return an empty array (for URL arrays and helper lists).
    - If multiple facilities are mentioned, extract the first one only.
    """


# --------------------------------------------------------------------------- #
# Helper for safe joining lists                                               #
# --------------------------------------------------------------------------- #
def _safe_join(items: List[str]) -> str:
    return ", ".join([s for s in items if isinstance(s, str) and s.strip()])


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root_node, extracted: FacilityExtraction) -> None:
    # Create the critical parent node for the whole facility identification (parallel aggregation per rubric)
    facility_node = evaluator.add_parallel(
        id="FacilityIdentification",
        desc="Identify one wildlife rehabilitation facility in Virginia that is authorized to rehabilitate raptors and provide supporting URLs verifying each required constraint.",
        parent=root_node,
        critical=True,
    )

    # 1) Facility name must be provided (critical leaf)
    name_present = bool(extracted.facility_name and extracted.facility_name.strip())
    facility_name_leaf = evaluator.add_custom_node(
        result=name_present,
        id="FacilityNameProvided",
        desc="Provide the name of the facility.",
        parent=facility_node,
        critical=True
    )

    # Convenience variables
    facility_name = extracted.facility_name or "the facility"
    staff_names_str = _safe_join(extracted.staff_member_names)
    staff_certs_str = _safe_join(extracted.staff_certifications)
    vet_names_str = _safe_join(extracted.veterinarian_names)

    # 2) Location AND Raptor Authorization (critical parallel group)
    loc_auth_node = evaluator.add_parallel(
        id="LocationAndRaptorAuthorization",
        desc="Verify the facility is located in Virginia and is explicitly authorized to rehabilitate raptors (birds of prey).",
        parent=facility_node,
        critical=True
    )

    # 2.a Meets requirement - verify against the answer text (simple check)
    loc_auth_meets_leaf = evaluator.add_leaf(
        id="MeetsLocationAndAuthorization",
        desc="Facility is physically located in Virginia AND explicitly authorized to care for/rehabilitate raptors (e.g., hawks, eagles, owls, falcons).",
        parent=loc_auth_node,
        critical=True
    )
    loc_auth_meets_claim = (
        f"The facility named '{facility_name}' is physically located in Virginia and is explicitly authorized to rehabilitate raptors "
        f"(birds of prey such as hawks, eagles, owls, and falcons)."
    )
    await evaluator.verify(
        claim=loc_auth_meets_claim,
        node=loc_auth_meets_leaf,
        additional_instruction=(
            "Judge based on the provided answer text. Consider synonyms for raptors (birds of prey). "
            "Both conditions (Virginia location AND explicit raptor authorization) must be present in the answer to pass."
        ),
        extra_prerequisites=[facility_name_leaf]
    )

    # 2.b Evidence URLs must support the same claim (critical leaf, URL-based)
    loc_auth_evidence_leaf = evaluator.add_leaf(
        id="EvidenceURLs_LocationAndAuthorization",
        desc="Provide at least one reference URL that supports the Virginia location and raptor-authorization claim.",
        parent=loc_auth_node,
        critical=True
    )
    await evaluator.verify(
        claim=loc_auth_meets_claim,
        node=loc_auth_evidence_leaf,
        sources=extracted.location_authorization_urls,
        additional_instruction=(
            "Use the provided URLs to verify BOTH the Virginia location and explicit raptor rehabilitation authorization for the facility. "
            "Accept reasonable phrasing (e.g., 'birds of prey', listing of specific raptors like hawks/owls/eagles/falcons). "
            "If no URLs were provided, mark as not supported."
        ),
        extra_prerequisites=[facility_name_leaf]
    )

    # 3) Federal Permit Compliance (critical parallel group)
    fed_node = evaluator.add_parallel(
        id="FederalPermitCompliance",
        desc="Verify the facility holds the required federal permit authorizing raptor rehabilitation.",
        parent=facility_node,
        critical=True
    )

    # 3.a Meets federal permit requirement (simple check)
    fed_meets_leaf = evaluator.add_leaf(
        id="MeetsFederalPermitRequirement",
        desc="Facility holds a valid Federal Migratory Bird Rehabilitation Permit from the U.S. Fish & Wildlife Service that authorizes rehabilitation of raptors.",
        parent=fed_node,
        critical=True
    )
    fed_meets_claim = (
        f"The facility '{facility_name}' holds a valid Federal Migratory Bird Rehabilitation Permit from the U.S. Fish & Wildlife Service "
        f"that authorizes rehabilitation of raptors."
    )
    await evaluator.verify(
        claim=fed_meets_claim,
        node=fed_meets_leaf,
        additional_instruction=(
            "Judge from the answer text only. Accept common synonyms such as 'USFWS Migratory Bird Rehabilitation Permit' or "
            "'Special Purpose—Rehabilitation (Migratory Bird) permit' that includes raptors."
        ),
        extra_prerequisites=[facility_name_leaf]
    )

    # 3.b Evidence URLs for federal permit (URL-based)
    fed_evidence_leaf = evaluator.add_leaf(
        id="EvidenceURLs_FederalPermit",
        desc="Provide at least one reference URL that supports the federal permit claim (including raptor authorization).",
        parent=fed_node,
        critical=True
    )
    await evaluator.verify(
        claim=fed_meets_claim,
        node=fed_evidence_leaf,
        sources=extracted.federal_permit_urls,
        additional_instruction=(
            "Use the provided URLs to confirm that the facility holds a valid USFWS Migratory Bird Rehabilitation permit and that it authorizes raptor rehabilitation. "
            "Official sources (e.g., USFWS) or credible facility documentation are acceptable. If no URLs were provided, mark as not supported."
        ),
        extra_prerequisites=[facility_name_leaf]
    )

    # 4) Professional Staff Credentials (critical parallel group)
    staff_node = evaluator.add_parallel(
        id="ProfessionalStaffCredentials",
        desc="Verify the facility has at least one staff member with required professional wildlife rehabilitation credentials.",
        parent=facility_node,
        critical=True
    )

    # 4.a Meets staff credential requirement (simple check)
    staff_meets_leaf = evaluator.add_leaf(
        id="MeetsStaffCredentialRequirement",
        desc="Facility has at least one staff member who holds IWRC Certified Wildlife Rehabilitator (CWR™) certification OR an equivalent professional wildlife rehabilitation certification recognized in the field.",
        parent=staff_node,
        critical=True
    )
    staff_meets_claim = (
        f"At least one staff member at '{facility_name}' holds IWRC Certified Wildlife Rehabilitator (CWR™) certification "
        f"or an equivalent professional wildlife rehabilitation certification recognized in the field."
    )
    await evaluator.verify(
        claim=staff_meets_claim,
        node=staff_meets_leaf,
        additional_instruction=(
            f"The answer may mention staff or certifications (e.g., {staff_names_str} / {staff_certs_str}). "
            "Judge based on the answer text; passing requires explicit mention of CWR (IWRC) or an equivalent professional wildlife rehabilitation certification."
        ),
        extra_prerequisites=[facility_name_leaf]
    )

    # 4.b Evidence URLs for staff credentials (URL-based)
    staff_evidence_leaf = evaluator.add_leaf(
        id="EvidenceURLs_StaffCredentials",
        desc="Provide at least one reference URL that supports the staff-credential claim.",
        parent=staff_node,
        critical=True
    )
    await evaluator.verify(
        claim=staff_meets_claim,
        node=staff_evidence_leaf,
        sources=extracted.staff_credentials_urls,
        additional_instruction=(
            f"Use the provided URLs to verify that at least one staff member at '{facility_name}' holds IWRC CWR certification "
            f"or an equivalent professional wildlife rehabilitation certification. Names mentioned in the answer: {staff_names_str}. "
            f"Certifications mentioned: {staff_certs_str}. If no URLs were provided, mark as not supported."
        ),
        extra_prerequisites=[facility_name_leaf]
    )

    # 5) Facility Housing Standards (critical parallel group)
    housing_node = evaluator.add_parallel(
        id="FacilityHousingStandards",
        desc="Verify the facility maintains enclosures/housing meeting minimum standards for raptor rehabilitation per wildlife rehabilitation guidelines.",
        parent=facility_node,
        critical=True
    )

    # 5.a Meets housing standards requirement (simple check)
    housing_meets_leaf = evaluator.add_leaf(
        id="MeetsHousingStandardsRequirement",
        desc="Facility maintains raptor enclosures/housing that meet minimum standards described in wildlife rehabilitation guidelines (e.g., suitable cage sizes, perching structures, protection from environmental elements).",
        parent=housing_node,
        critical=True
    )
    housing_meets_claim = (
        f"The facility '{facility_name}' maintains raptor enclosures/housing that meet minimum standards per wildlife rehabilitation guidelines, "
        f"including appropriate cage sizes, suitable perching structures, and protection from environmental elements."
    )
    await evaluator.verify(
        claim=housing_meets_claim,
        node=housing_meets_leaf,
        additional_instruction=(
            "Judge from the answer text. Look for explicit mention of enclosures/housing standards suitable for raptors—cage sizes, perches, shelter, etc."
        ),
        extra_prerequisites=[facility_name_leaf]
    )

    # 5.b Evidence URLs for housing standards (URL-based)
    housing_evidence_leaf = evaluator.add_leaf(
        id="EvidenceURLs_HousingStandards",
        desc="Provide at least one reference URL that supports the housing/enclosure standards claim.",
        parent=housing_node,
        critical=True
    )
    await evaluator.verify(
        claim=housing_meets_claim,
        node=housing_evidence_leaf,
        sources=extracted.housing_standards_urls,
        additional_instruction=(
            "Use the provided URLs to confirm that the facility maintains raptor enclosures/housing meeting minimum standards (e.g., proper sizes, perches, weather protection). "
            "If no URLs were provided, mark as not supported."
        ),
        extra_prerequisites=[facility_name_leaf]
    )

    # 6) Veterinary Partnership (critical parallel group)
    vet_node = evaluator.add_parallel(
        id="VeterinaryPartnership",
        desc="Verify the facility has documented veterinary oversight via partnership with a licensed veterinarian.",
        parent=facility_node,
        critical=True
    )

    # 6.a Meets veterinary partnership requirement (simple check)
    vet_meets_leaf = evaluator.add_leaf(
        id="MeetsVeterinaryPartnershipRequirement",
        desc="Facility has a documented partnership with a licensed veterinarian who provides veterinary oversight for animals in care.",
        parent=vet_node,
        critical=True
    )
    vet_meets_claim = (
        f"The facility '{facility_name}' has a documented partnership with a licensed veterinarian who provides veterinary oversight "
        f"for animals in their care."
    )
    await evaluator.verify(
        claim=vet_meets_claim,
        node=vet_meets_leaf,
        additional_instruction=(
            f"Judge from the answer text. Look for explicit mention of veterinary partnership or oversight (e.g., DVM names such as {vet_names_str}, "
            "terms like 'veterinary partner', 'supervising veterinarian')."
        ),
        extra_prerequisites=[facility_name_leaf]
    )

    # 6.b Evidence URLs for veterinary partnership (URL-based)
    vet_evidence_leaf = evaluator.add_leaf(
        id="EvidenceURLs_VeterinaryPartnership",
        desc="Provide at least one reference URL that supports the veterinary partnership/oversight claim.",
        parent=vet_node,
        critical=True
    )
    await evaluator.verify(
        claim=vet_meets_claim,
        node=vet_evidence_leaf,
        sources=extracted.veterinary_partnership_urls,
        additional_instruction=(
            "Use the provided URLs to confirm a documented veterinary partnership or oversight (licensed veterinarian, DVM). "
            "If no URLs were provided, mark as not supported."
        ),
        extra_prerequisites=[facility_name_leaf]
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
    Evaluate an answer for the Virginia raptor rehabilitation facility task.
    """
    # Initialize evaluator with a parallel root (only one main group, but parallel is fine)
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

    # Extract structured facility information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_facility_info(),
        template_class=FacilityExtraction,
        extraction_name="facility_extraction",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extracted)

    # Return standardized evaluation summary
    return evaluator.get_summary()