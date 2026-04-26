import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "philly_therapy_dog"
TASK_DESCRIPTION = (
    "You are planning to adopt a therapy dog in Philadelphia, Pennsylvania and volunteer with the dog at local hospitals. "
    "What are all the legal requirements, certification requirements, and practical considerations you must fulfill for compliant dog ownership and hospital therapy work in Philadelphia? "
    "Your answer should include specific details about licensing, vaccination requirements, therapy dog certification, hospital-specific requirements, the legal distinction between service dogs and therapy dogs, "
    "public space regulations, and emergency veterinary resources."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LicenseDetails(BaseModel):
    age_threshold: Optional[str] = None
    obtain_timeframe: Optional[str] = None
    fee: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RabiesLicensing(BaseModel):
    statement: Optional[str] = None  # e.g., "Rabies vaccination is required to license a dog in Philadelphia"
    sources: List[str] = Field(default_factory=list)


class StateLicenseDetails(BaseModel):
    age_threshold: Optional[str] = None
    annual_deadline_or_renewal: Optional[str] = None
    penalties: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TherapyCertification(BaseModel):
    certification_required: Optional[str] = None  # e.g., "Therapy dogs must be certified through recognized orgs"
    prerequisite_testing: Optional[str] = None  # e.g., "AKC Canine Good Citizen (CGC)"
    sources: List[str] = Field(default_factory=list)


class HospitalVisitRequirements(BaseModel):
    vet_health_clearance: Optional[str] = None  # e.g., "documented veterinary health clearance needed"
    bath_within_24h: Optional[str] = None  # e.g., "dog must be bathed within 24 hours prior to visits"
    sources: List[str] = Field(default_factory=list)


class LegalDistinction(BaseModel):
    service_dogs_cert_required: Optional[str] = None  # e.g., "service dogs are not required to be certified/registered"
    service_dogs_public_access: Optional[str] = None  # e.g., "service dogs have public access rights under ADA"
    therapy_dogs_public_access: Optional[str] = None  # e.g., "therapy dogs do not have public access rights"
    sources: List[str] = Field(default_factory=list)


class LeashRule(BaseModel):
    leash_length_requirement: Optional[str] = None  # e.g., "leashes must be no longer than 6 feet"
    sources: List[str] = Field(default_factory=list)


class EmergencyVetResource(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    location: Optional[str] = None  # e.g., "Philadelphia, PA" (optional)


class EmergencyResources(BaseModel):
    resources: List[EmergencyVetResource] = Field(default_factory=list)


class PhiladelphiaTherapyDogExtraction(BaseModel):
    philadelphia_dog_licensing: Optional[LicenseDetails] = None
    rabies_vaccination_for_licensing: Optional[RabiesLicensing] = None
    pennsylvania_state_dog_licensing: Optional[StateLicenseDetails] = None
    therapy_dog_certification: Optional[TherapyCertification] = None
    hospital_therapy_visit_program_requirements: Optional[HospitalVisitRequirements] = None
    service_vs_therapy_dog_legal_distinction: Optional[LegalDistinction] = None
    philadelphia_public_space_leash_rule: Optional[LeashRule] = None
    emergency_veterinary_resources: Optional[EmergencyResources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return (
        "Extract structured information from the answer about compliant dog ownership and hospital therapy work in Philadelphia. "
        "Capture the exact text provided in the answer for each field. Also extract the URLs cited in the answer that support each section.\n\n"
        "Return a JSON object with the following top-level fields:\n"
        "1) philadelphia_dog_licensing: {\n"
        "   age_threshold: text or null,\n"
        "   obtain_timeframe: text or null,\n"
        "   fee: text or null,\n"
        "   sources: [list of URLs supporting Philadelphia licensing] (empty list if none)\n"
        "}\n"
        "2) rabies_vaccination_for_licensing: {\n"
        "   statement: text or null (e.g., 'Rabies vaccination is required to license a dog in Philadelphia'),\n"
        "   sources: [list of URLs] (empty list if none)\n"
        "}\n"
        "3) pennsylvania_state_dog_licensing: {\n"
        "   age_threshold: text or null,\n"
        "   annual_deadline_or_renewal: text or null,\n"
        "   penalties: text or null,\n"
        "   sources: [list of URLs] (empty list if none)\n"
        "}\n"
        "4) therapy_dog_certification: {\n"
        "   certification_required: text or null (e.g., 'Therapy dogs must be certified through recognized therapy dog orgs'),\n"
        "   prerequisite_testing: text or null (e.g., 'AKC Canine Good Citizen (CGC)'),\n"
        "   sources: [list of URLs] (empty list if none)\n"
        "}\n"
        "5) hospital_therapy_visit_program_requirements: {\n"
        "   vet_health_clearance: text or null (e.g., 'documented veterinary health clearance required'),\n"
        "   bath_within_24h: text or null (e.g., 'dog bathed within 24 hours before visits'),\n"
        "   sources: [list of URLs] (empty list if none)\n"
        "}\n"
        "6) service_vs_therapy_dog_legal_distinction: {\n"
        "   service_dogs_cert_required: text or null (e.g., 'service dogs are not required to be certified/registered'),\n"
        "   service_dogs_public_access: text or null (e.g., 'service dogs have public access rights under ADA'),\n"
        "   therapy_dogs_public_access: text or null (e.g., 'therapy dogs do not have public access rights; only by permission/policy'),\n"
        "   sources: [list of URLs] (empty list if none)\n"
        "}\n"
        "7) philadelphia_public_space_leash_rule: {\n"
        "   leash_length_requirement: text or null,\n"
        "   sources: [list of URLs] (empty list if none)\n"
        "}\n"
        "8) emergency_veterinary_resources: {\n"
        "   resources: [\n"
        "       { name: text or null, url: URL or null, location: text or null }\n"
        "   ]\n"
        "}\n\n"
        "Rules:\n"
        "- Extract only information explicitly present in the answer; do not invent details.\n"
        "- For sources, include only actual URLs present in the answer (plain URLs or markdown links). If a source is referenced without a URL, do not add it.\n"
        "- If a field is missing, set it to null; if no URLs are present for a section, return an empty list.\n"
        "- Preserve the answer's wording for each field; do not rewrite or normalize numeric values.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_nonempty_text(*vals: Optional[str]) -> bool:
    return all(v is not None and str(v).strip() != "" for v in vals)


def _has_sources(sources: Optional[List[str]]) -> bool:
    return bool(sources) and len(sources) > 0


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_philadelphia_dog_licensing(
    evaluator: Evaluator,
    parent_node,
    info: Optional[LicenseDetails],
) -> None:
    node = evaluator.add_parallel(
        id="Philadelphia_Dog_Licensing",
        desc="States Philadelphia dog licensing requirement with required details (age threshold, required timeframe to obtain license, and license fee)",
        parent=parent_node,
        critical=True,
    )

    # Existence gate: all three details present and sources provided
    existence = info is not None and _has_nonempty_text(info.age_threshold, info.obtain_timeframe, info.fee) and _has_sources(info.sources)
    evaluator.add_custom_node(
        result=existence,
        id="phila_licensing_details_provided",
        desc="Philadelphia licensing details (age threshold, timeframe, fee) are present with cited sources",
        parent=node,
        critical=True,
    )

    # Age threshold
    leaf_age = evaluator.add_leaf(
        id="phila_licensing_age_supported",
        desc="Philadelphia dog licensing age threshold is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Philadelphia's dog licensing age threshold is {info.age_threshold}.",
        node=leaf_age,
        sources=info.sources,
        additional_instruction="Verify that the cited sources explicitly state the minimum age threshold at which a dog must be licensed in Philadelphia. Allow minor phrasing variations.",
    )

    # Timeframe to obtain license
    leaf_timeframe = evaluator.add_leaf(
        id="phila_licensing_timeframe_supported",
        desc="Philadelphia dog licensing timeframe to obtain the license is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"In Philadelphia, a dog license must be obtained within {info.obtain_timeframe}.",
        node=leaf_timeframe,
        sources=info.sources,
        additional_instruction="Verify the timeframe requirement (e.g., within X days of acquisition or moving). Allow minor wording variations.",
    )

    # License fee
    leaf_fee = evaluator.add_leaf(
        id="phila_licensing_fee_supported",
        desc="Philadelphia dog license fee is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The Philadelphia dog license fee is {info.fee}.",
        node=leaf_fee,
        sources=info.sources,
        additional_instruction="If fee tiers exist (e.g., altered vs. unaltered), the claim is supported if the cited sources include the provided fee among the options.",
    )


async def verify_rabies_vaccination_for_licensing(
    evaluator: Evaluator,
    parent_node,
    info: Optional[RabiesLicensing],
) -> None:
    node = evaluator.add_parallel(
        id="Rabies_Vaccination_For_Licensing",
        desc="States that rabies vaccination is required in order to license a dog in Philadelphia",
        parent=parent_node,
        critical=True,
    )

    existence = info is not None and _has_nonempty_text(info.statement) and _has_sources(info.sources)
    evaluator.add_custom_node(
        result=existence,
        id="rabies_for_license_statement_provided",
        desc="Rabies vaccination requirement statement present with cited sources",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="rabies_for_license_supported",
        desc="Rabies vaccination requirement for licensing in Philadelphia is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Philadelphia requires proof of rabies vaccination in order to license a dog.",
        node=leaf,
        sources=info.sources,
        additional_instruction="Confirm the rabies vaccination requirement for dog licensing in Philadelphia.",
    )


async def verify_pennsylvania_state_dog_licensing(
    evaluator: Evaluator,
    parent_node,
    info: Optional[StateLicenseDetails],
) -> None:
    node = evaluator.add_parallel(
        id="Pennsylvania_State_Dog_Licensing",
        desc="States Pennsylvania dog licensing requirement with required details (age threshold, annual licensing deadline/renewal expectation, and penalties for non-compliance)",
        parent=parent_node,
        critical=True,
    )

    existence = info is not None and _has_nonempty_text(info.age_threshold, info.annual_deadline_or_renewal, info.penalties) and _has_sources(info.sources)
    evaluator.add_custom_node(
        result=existence,
        id="pa_state_licensing_details_provided",
        desc="Pennsylvania licensing details (age threshold, annual deadline/renewal, penalties) present with cited sources",
        parent=node,
        critical=True,
    )

    # Age threshold
    leaf_age = evaluator.add_leaf(
        id="pa_licensing_age_supported",
        desc="Pennsylvania dog licensing age threshold is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Pennsylvania's dog licensing age threshold is {info.age_threshold}.",
        node=leaf_age,
        sources=info.sources,
        additional_instruction="Verify the statewide minimum age at which dogs must be licensed in Pennsylvania.",
    )

    # Annual deadline/renewal expectation
    leaf_deadline = evaluator.add_leaf(
        id="pa_licensing_deadline_supported",
        desc="Pennsylvania annual licensing deadline/renewal expectation is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"In Pennsylvania, the annual licensing deadline or renewal expectation is {info.annual_deadline_or_renewal}.",
        node=leaf_deadline,
        sources=info.sources,
        additional_instruction="Confirm the annual licensing deadline or renewal expectation for Pennsylvania dog licenses.",
    )

    # Penalties for non-compliance
    leaf_penalties = evaluator.add_leaf(
        id="pa_licensing_penalties_supported",
        desc="Pennsylvania penalties for non-compliance are supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Pennsylvania imposes penalties for non-compliance: {info.penalties}.",
        node=leaf_penalties,
        sources=info.sources,
        additional_instruction="Verify penalties/fines for failing to license a dog under Pennsylvania law.",
    )


async def verify_therapy_dog_certification(
    evaluator: Evaluator,
    parent_node,
    info: Optional[TherapyCertification],
) -> None:
    node = evaluator.add_parallel(
        id="Therapy_Dog_Certification",
        desc="States that therapy dogs must be certified through a recognized therapy dog organization and notes typical prerequisite testing (e.g., AKC CGC) as part of eligibility",
        parent=parent_node,
        critical=True,
    )

    existence = info is not None and _has_nonempty_text(info.certification_required, info.prerequisite_testing) and _has_sources(info.sources)
    evaluator.add_custom_node(
        result=existence,
        id="therapy_cert_details_provided",
        desc="Therapy dog certification requirement and prerequisite testing provided with cited sources",
        parent=node,
        critical=True,
    )

    leaf_cert = evaluator.add_leaf(
        id="therapy_cert_required_supported",
        desc="Therapy dogs must be certified through recognized organizations (supported by sources)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Therapy dogs must be certified through a recognized therapy dog organization.",
        node=leaf_cert,
        sources=info.sources,
        additional_instruction="Confirm certification requirement via recognized therapy dog organizations.",
    )

    leaf_prereq = evaluator.add_leaf(
        id="therapy_prereq_testing_supported",
        desc="Typical prerequisite testing (e.g., AKC CGC) for therapy dog eligibility is supported by sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Typical therapy dog eligibility includes prerequisite testing such as {info.prerequisite_testing}.",
        node=leaf_prereq,
        sources=info.sources,
        additional_instruction="Confirm that common therapy dog orgs require baseline tests (e.g., AKC Canine Good Citizen).",
    )


async def verify_hospital_therapy_visit_program_requirements(
    evaluator: Evaluator,
    parent_node,
    info: Optional[HospitalVisitRequirements],
) -> None:
    node = evaluator.add_parallel(
        id="Hospital_Therapy_Visit_Program_Requirements",
        desc="States hospital therapy-visit requirements beyond general therapy-dog certification: documented veterinary health clearance and the dog being bathed within 24 hours prior to visits",
        parent=parent_node,
        critical=True,
    )

    existence = info is not None and _has_nonempty_text(info.vet_health_clearance, info.bath_within_24h) and _has_sources(info.sources)
    evaluator.add_custom_node(
        result=existence,
        id="hospital_visit_requirements_provided",
        desc="Hospital therapy-visit requirements (vet health clearance and bath within 24h) provided with cited sources",
        parent=node,
        critical=True,
    )

    leaf_vet = evaluator.add_leaf(
        id="hospital_vet_clearance_supported",
        desc="Hospitals require documented veterinary health clearance for therapy dogs (supported by sources)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Hospitals require therapy dogs to have documented veterinary health clearance for visit programs.",
        node=leaf_vet,
        sources=info.sources,
        additional_instruction="Confirm hospital therapy dog program requirements include veterinary health clearances.",
    )

    leaf_bath = evaluator.add_leaf(
        id="hospital_bath_24h_supported",
        desc="Hospitals require therapy dogs to be bathed within 24 hours prior to visits (supported by sources)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Hospitals require therapy dogs to be bathed within 24 hours prior to visits.",
        node=leaf_bath,
        sources=info.sources,
        additional_instruction="Confirm hospital visit hygiene requirements include bathing within 24 hours.",
    )


async def verify_service_vs_therapy_dog_legal_distinction(
    evaluator: Evaluator,
    parent_node,
    info: Optional[LegalDistinction],
) -> None:
    node = evaluator.add_parallel(
        id="Service_vs_Therapy_Dog_Legal_Distinction",
        desc="Explains ADA-related distinction: service dogs are not required to be certified/registered and have public access rights; therapy dogs do not have public access rights (may only access facilities by permission/invitation/policy)",
        parent=parent_node,
        critical=True,
    )

    existence = info is not None and _has_nonempty_text(
        info.service_dogs_cert_required, info.service_dogs_public_access, info.therapy_dogs_public_access
    ) and _has_sources(info.sources)
    evaluator.add_custom_node(
        result=existence,
        id="ada_distinction_details_provided",
        desc="ADA-related distinction details present with cited sources",
        parent=node,
        critical=True,
    )

    leaf_service_cert = evaluator.add_leaf(
        id="ada_service_cert_not_required_supported",
        desc="Under ADA, service dogs are not required to be certified/registered (supported by sources)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Under the ADA, service dogs are not required to be certified or registered.",
        node=leaf_service_cert,
        sources=info.sources,
        additional_instruction="Confirm ADA guidance states no certification/registration requirement for service dogs.",
    )

    leaf_service_access = evaluator.add_leaf(
        id="ada_service_public_access_supported",
        desc="Under ADA, service dogs have public access rights (supported by sources)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Under the ADA, service dogs have public access rights to accompany their handlers in public places.",
        node=leaf_service_access,
        sources=info.sources,
        additional_instruction="Confirm ADA public access rights for service dogs.",
    )

    leaf_therapy_access = evaluator.add_leaf(
        id="therapy_dogs_no_public_access_supported",
        desc="Therapy dogs do not have public access rights; access only by facility permission/policy (supported by sources)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Therapy dogs do not have public access rights; they may enter facilities only by permission, invitation, or policy.",
        node=leaf_therapy_access,
        sources=info.sources,
        additional_instruction="Confirm that therapy dogs lack ADA public access and rely on facility policies/permissions.",
    )


async def verify_philadelphia_public_space_leash_rule(
    evaluator: Evaluator,
    parent_node,
    info: Optional[LeashRule],
) -> None:
    node = evaluator.add_parallel(
        id="Philadelphia_Public_Space_Leash_Rule",
        desc="States Philadelphia public space/park leash regulation (leash length requirement)",
        parent=parent_node,
        critical=True,
    )

    existence = info is not None and _has_nonempty_text(info.leash_length_requirement) and _has_sources(info.sources)
    evaluator.add_custom_node(
        result=existence,
        id="phila_leash_rule_details_provided",
        desc="Philadelphia leash length requirement present with cited sources",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="phila_leash_length_supported",
        desc="Philadelphia leash length requirement is supported by cited sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Philadelphia public space/park leash regulation sets a leash length requirement of {info.leash_length_requirement}.",
        node=leaf,
        sources=info.sources,
        additional_instruction="Verify the leash length requirement in Philadelphia code or official policy. Allow minor phrasing variations.",
    )


async def verify_emergency_veterinary_resources(
    evaluator: Evaluator,
    parent_node,
    info: Optional[EmergencyResources],
) -> None:
    node = evaluator.add_parallel(
        id="Emergency_Veterinary_Resources",
        desc="Provides at least one 24-hour emergency veterinary resource in the Philadelphia area",
        parent=parent_node,
        critical=True,
    )

    # Existence: at least one resource with a URL
    has_any_url = False
    urls: List[str] = []
    if info and info.resources:
        for r in info.resources:
            if r and r.url and str(r.url).strip():
                has_any_url = True
                urls.append(r.url)
    evaluator.add_custom_node(
        result=has_any_url,
        id="emergency_vet_resource_provided",
        desc="At least one emergency veterinary resource with a URL is provided",
        parent=node,
        critical=True,
    )

    # Verification: at least one of the provided URLs corresponds to a 24-hour emergency vet in the Philadelphia area
    leaf = evaluator.add_leaf(
        id="emergency_vet_24hr_phila_supported",
        desc="At least one provided URL is a 24-hour emergency veterinary resource in the Philadelphia area",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="At least one of the provided veterinary resources offers 24-hour emergency services and is located in the Philadelphia, PA area.",
        node=leaf,
        sources=urls,
        additional_instruction="Confirm that at least one URL indicates 24/7 emergency services and a Philadelphia-area location/address.",
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
    Evaluate an answer for Philadelphia therapy dog compliance requirements.
    """
    # Initialize evaluator (root is a non-critical node managed by the framework)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation at overall level
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

    # Create the critical top-level node per rubric
    top = evaluator.add_parallel(
        id="Complete_Requirements_Identification",
        desc="Identifies legal requirements, certification requirements, and practical considerations for compliant dog ownership and hospital therapy work in Philadelphia",
        parent=root,
        critical=True,
    )

    # Extract structured information from the answer
    extracted: PhiladelphiaTherapyDogExtraction = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=PhiladelphiaTherapyDogExtraction,
        extraction_name="requirements_extraction",
    )

    # Build verification subtrees per rubric
    await verify_philadelphia_dog_licensing(
        evaluator, top, extracted.philadelphia_dog_licensing
    )
    await verify_rabies_vaccination_for_licensing(
        evaluator, top, extracted.rabies_vaccination_for_licensing
    )
    await verify_pennsylvania_state_dog_licensing(
        evaluator, top, extracted.pennsylvania_state_dog_licensing
    )
    await verify_therapy_dog_certification(
        evaluator, top, extracted.therapy_dog_certification
    )
    await verify_hospital_therapy_visit_program_requirements(
        evaluator, top, extracted.hospital_therapy_visit_program_requirements
    )
    await verify_service_vs_therapy_dog_legal_distinction(
        evaluator, top, extracted.service_vs_therapy_dog_legal_distinction
    )
    await verify_philadelphia_public_space_leash_rule(
        evaluator, top, extracted.philadelphia_public_space_leash_rule
    )
    await verify_emergency_veterinary_resources(
        evaluator, top, extracted.emergency_veterinary_resources
    )

    # Return the evaluation summary
    return evaluator.get_summary()