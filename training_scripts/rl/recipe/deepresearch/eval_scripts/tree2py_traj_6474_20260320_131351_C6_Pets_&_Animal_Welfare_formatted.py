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
TASK_ID = "therapy_animal_program_requirements"
TASK_DESCRIPTION = """
A healthcare facility wants to establish a new therapy animal program that partners with local animal welfare organizations for educational outreach. The program coordinator needs to ensure compliance with all relevant standards from nationally recognized certification organizations, industry training requirements, and healthcare facility protocols.

Identify and document the following:
1. A nationally recognized therapy animal certification organization (such as Pet Partners, Alliance of Therapy Dogs, or Therapy Dogs International) and verify what services they provide to registered teams
2. The minimum age requirement for handlers according to your selected certification organization
3. The training duration requirement according to your selected organization and verify if it meets or exceeds IAADP minimum standards (120 hours over 6 months)
4. The minimum age requirement for therapy dogs according to your selected organization
5. What health documentation is required for therapy animals
6. What written policies healthcare facilities must have in place for therapy animal programs

For each requirement, provide the specific information and a reference URL from the relevant organization's official documentation.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class OrganizationInfo(BaseModel):
    name: Optional[str] = None
    official_org_urls: List[str] = Field(default_factory=list)
    national_scope_claim: Optional[str] = None
    national_scope_urls: List[str] = Field(default_factory=list)

    services_description: Optional[str] = None
    services_urls: List[str] = Field(default_factory=list)

    liability_insurance_statement: Optional[str] = None
    liability_insurance_urls: List[str] = Field(default_factory=list)

    formal_evaluation_statement: Optional[str] = None
    formal_evaluation_urls: List[str] = Field(default_factory=list)


class HandlerRequirements(BaseModel):
    min_age: Optional[str] = None
    min_age_urls: List[str] = Field(default_factory=list)

    course_requirement_statement: Optional[str] = None
    course_urls: List[str] = Field(default_factory=list)

    knowledge_assessment_statement: Optional[str] = None
    knowledge_assessment_urls: List[str] = Field(default_factory=list)


class AnimalRequirements(BaseModel):
    min_age_dog: Optional[str] = None
    min_age_urls: List[str] = Field(default_factory=list)

    veterinary_health_doc_statement: Optional[str] = None
    veterinary_health_doc_urls: List[str] = Field(default_factory=list)

    vaccination_statement: Optional[str] = None
    vaccination_urls: List[str] = Field(default_factory=list)

    fitness_to_participate_statement: Optional[str] = None
    fitness_to_participate_urls: List[str] = Field(default_factory=list)


class TrainingStandards(BaseModel):
    org_training_duration_statement: Optional[str] = None
    org_training_duration_urls: List[str] = Field(default_factory=list)

    iaadp_minimum_standard_statement: Optional[str] = None
    iaadp_urls: List[str] = Field(default_factory=list)

    meets_or_exceeds_statement: Optional[str] = None


class HealthcarePolicies(BaseModel):
    written_policy_statement: Optional[str] = None
    written_policy_urls: List[str] = Field(default_factory=list)

    hand_hygiene_statement: Optional[str] = None
    hand_hygiene_urls: List[str] = Field(default_factory=list)

    animal_health_screening_statement: Optional[str] = None
    animal_health_screening_urls: List[str] = Field(default_factory=list)


class ProgramExtraction(BaseModel):
    selected_organization: OrganizationInfo = Field(default_factory=OrganizationInfo)
    handler: HandlerRequirements = Field(default_factory=HandlerRequirements)
    animal: AnimalRequirements = Field(default_factory=AnimalRequirements)
    training: TrainingStandards = Field(default_factory=TrainingStandards)
    facility_policies: HealthcarePolicies = Field(default_factory=HealthcarePolicies)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program() -> str:
    return """
    Extract, exactly as stated in the answer, the structured information needed to evaluate a healthcare therapy animal program.
    IMPORTANT:
    - Return only URLs that the answer explicitly includes. Do not infer or add any URLs.
    - For organization-specific items, extract URLs only from the selected organization's official site (e.g., petpartners.org, therapydogs.com, tdi-dog.org) when the answer provides them.
    - For IAADP, the URL(s) must be from the official IAADP domain (iaadp.org) if present in the answer.
    - For healthcare facility policy items, extract URLs from recognized health authorities/professional bodies (e.g., CDC, AVMA, APIC, Joint Commission) if present in the answer.

    Return the following JSON fields:

    selected_organization:
      - name: The single certification organization selected in the answer.
      - official_org_urls: All official organization homepage/about/contact URLs included in the answer (array).
      - national_scope_claim: The phrase/summary in the answer describing the organization as national/nationwide/serving teams across the U.S. (null if not present).
      - national_scope_urls: All official org URLs cited for national/nationwide scope claims (array).
      - services_description: The answer's description of services provided to registered teams (null if not present).
      - services_urls: All official org URLs that support services for registered teams (array).
      - liability_insurance_statement: The answer's statement about org-provided liability insurance (null if not present).
      - liability_insurance_urls: All official org URLs supporting liability insurance coverage (array).
      - formal_evaluation_statement: The answer's statement that a formal team evaluation is required before certification/registration (null if not present).
      - formal_evaluation_urls: All official org URLs supporting the formal evaluation requirement (array).

    handler:
      - min_age: The minimum handler age stated (string; keep exactly as the answer phrases it).
      - min_age_urls: All official org URLs supporting the minimum handler age (array).
      - course_requirement_statement: Statement that a handler course/education is required (null if not present).
      - course_urls: All official org URLs supporting the handler course requirement (array).
      - knowledge_assessment_statement: Statement that a knowledge test/quiz is required (null if not present).
      - knowledge_assessment_urls: All official org URLs supporting the knowledge assessment requirement (array).

    animal:
      - min_age_dog: The minimum dog age requirement (string; keep the answer's wording).
      - min_age_urls: All official org URLs supporting the minimum dog age (array).
      - veterinary_health_doc_statement: Statement describing veterinary health screening documentation required (null if not present).
      - veterinary_health_doc_urls: All official org URLs supporting veterinary health documentation requirements (array).
      - vaccination_statement: Statement describing vaccination documentation/requirements (null if not present).
      - vaccination_urls: All official org URLs supporting vaccination requirements (array).
      - fitness_to_participate_statement: Statement that animal must be free of illness/injury/parasites at time of participation (null if not present).
      - fitness_to_participate_urls: All official org URLs supporting that condition (array).

    training:
      - org_training_duration_statement: The selected organization's training duration requirement as stated (or that the org does not specify a duration).
      - org_training_duration_urls: All official org URLs supporting that training duration/non-specification (array).
      - iaadp_minimum_standard_statement: The answer's statement of IAADP minimum standard (should indicate 120 hours over at least 6 months).
      - iaadp_urls: All IAADP official URLs included (must be iaadp.org) (array).
      - meets_or_exceeds_statement: The explicit comparison the answer provides indicating whether the org's requirement meets/exceeds IAADP minimum OR that the org does not specify and thus cannot be verified (null if not provided).

    facility_policies:
      - written_policy_statement: Statement that healthcare facilities must have a written policy governing therapy animal access/programs (null if not present).
      - written_policy_urls: All URLs from recognized bodies (CDC, AVMA, APIC, Joint Commission, etc.) supporting written-policy requirement (array).
      - hand_hygiene_statement: Statement that infection prevention includes hand hygiene protocols for interactions with animals (null if not present).
      - hand_hygiene_urls: All URLs from recognized bodies supporting hand hygiene in this context (array).
      - animal_health_screening_statement: Statement that infection prevention includes animal health screening procedures (null if not present).
      - animal_health_screening_urls: All URLs from recognized bodies supporting animal health screening in this context (array).

    If any field is not present in the answer, return null (for strings) or [] (for URL arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls or [])


def org_is_pet_partners(name: Optional[str]) -> bool:
    if not name:
        return False
    return name.strip().lower() in {
        "pet partners", "petpartners", "pet partners®", "pet partners (delta society)"
    }


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_certification_org_checks(evaluator: Evaluator, parent, data: ProgramExtraction):
    node = evaluator.add_parallel(
        id="Certification_Organization",
        desc="Selected therapy animal certification organization meets required standards and documentation is provided.",
        parent=parent,
        critical=True
    )

    org = data.selected_organization
    org_name = org.name or ""

    # Existence gates
    evaluator.add_custom_node(
        result=bool(org.name and org.name.strip()),
        id="org_name_provided",
        desc="Selected organization name is provided",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(org.national_scope_urls),
        id="org_national_scope_url_present",
        desc="Official URL(s) provided for national scope evidence",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(org.services_urls),
        id="org_services_url_present",
        desc="Official URL(s) provided for services to registered teams",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(org.liability_insurance_urls),
        id="org_insurance_url_present",
        desc="Official URL(s) provided for liability insurance provision",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(org.formal_evaluation_urls),
        id="org_eval_url_present",
        desc="Official URL(s) provided for formal team evaluation requirement",
        parent=node,
        critical=True
    )

    # 1) Selected organization national scope with official URL
    leaf1 = evaluator.add_leaf(
        id="Selected_Organization_And_National_Scope_Evidence_With_Official_URL",
        desc="Names a single selected certification organization AND provides an official organization URL where the organization explicitly describes itself as national/nationwide/serving teams across the United States.",
        parent=node,
        critical=True
    )
    claim1 = (
        f"The cited official page(s) state that {org_name} is national/nationwide or serves teams across the United States."
    )
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=org.national_scope_urls,
        additional_instruction="Accept clear phrases such as 'national', 'nationwide', 'across the U.S.', 'throughout the United States', or equivalent wording on the official organization website."
    )

    # 2) Services provided to registered teams with official URL
    leaf2 = evaluator.add_leaf(
        id="Services_Provided_to_Registered_Teams_With_Official_URL",
        desc="Describes services the organization provides to registered teams AND provides an official organization URL supporting those services.",
        parent=node,
        critical=True
    )
    services_text = org.services_description or "services for registered teams"
    claim2 = (
        f"The cited official page(s) describe services provided by {org_name} to registered therapy animal teams, consistent with: {services_text}"
    )
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=org.services_urls,
        additional_instruction="Focus on benefits/services afforded to registered/active teams (e.g., credentialing, placement support, resources, continuing education)."
    )

    # 3) Liability insurance provision with official URL
    leaf3 = evaluator.add_leaf(
        id="Liability_Insurance_Provision_With_Official_URL",
        desc="States that the organization provides liability insurance coverage for registered therapy animal teams AND provides an official organization URL supporting the insurance coverage.",
        parent=node,
        critical=True
    )
    claim3 = (
        f"The cited official page(s) indicate that {org_name} provides liability insurance coverage for registered therapy animal teams."
    )
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=org.liability_insurance_urls,
        additional_instruction="Look for explicit mention of liability insurance coverage, e.g., general liability policy for volunteer teams."
    )

    # 4) Formal evaluation requirement with official URL
    leaf4 = evaluator.add_leaf(
        id="Formal_Evaluation_Requirement_With_Official_URL",
        desc="States that the organization requires a formal team evaluation before certification/registration AND provides an official organization URL supporting the evaluation requirement.",
        parent=node,
        critical=True
    )
    claim4 = (
        f"The cited official page(s) state that {org_name} requires a formal team evaluation before certification/registration."
    )
    await evaluator.verify(
        claim=claim4,
        node=leaf4,
        sources=org.formal_evaluation_urls,
        additional_instruction="Accept terms such as 'team evaluation', 'evaluation', 'assessment' prior to registration/certification."
    )


async def build_handler_qualification_checks(evaluator: Evaluator, parent, data: ProgramExtraction):
    node = evaluator.add_parallel(
        id="Handler_Qualifications",
        desc="Handler requirements are documented per the selected organization and constraints.",
        parent=parent,
        critical=True
    )

    org_name = data.selected_organization.name or ""
    handler = data.handler

    # Existence gates
    evaluator.add_custom_node(
        result=bool(handler.min_age and handler.min_age.strip()) and has_urls(handler.min_age_urls),
        id="handler_min_age_url_present",
        desc="Official URL(s) provided for minimum handler age requirement",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(handler.course_urls),
        id="handler_course_url_present",
        desc="Official URL(s) provided for handler course/education requirement",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(handler.knowledge_assessment_urls),
        id="handler_knowledge_url_present",
        desc="Official URL(s) provided for handler knowledge assessment requirement",
        parent=node,
        critical=True
    )

    # 1) Handler minimum age requirement with official URL
    leaf1 = evaluator.add_leaf(
        id="Handler_Minimum_Age_Requirement_With_Official_URL",
        desc="States the minimum handler age required by the selected organization AND provides an official organization URL supporting that minimum age.",
        parent=node,
        critical=True
    )
    claim1 = f"The minimum handler age per {org_name} is {handler.min_age}."
    add_ins1 = "If the organization is Pet Partners, the minimum handler age must be at least 10 years." if org_is_pet_partners(org_name) else "Verify the stated minimum handler age exactly as shown on the official page."
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=handler.min_age_urls,
        additional_instruction=add_ins1
    )

    # 2) Handler course completion requirement with official URL
    leaf2 = evaluator.add_leaf(
        id="Handler_Course_Completion_Requirement_With_Official_URL",
        desc="States that the selected organization requires completion of a handler course (or equivalent required handler education) AND provides an official organization URL supporting this requirement.",
        parent=node,
        critical=True
    )
    claim2 = f"The cited official page(s) indicate that {org_name} requires completion of a handler course or equivalent required education."
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=handler.course_urls,
        additional_instruction="Accept 'handler course', 'training course', 'education module', or similar mandatory handler education prior to evaluation/registration."
    )

    # 3) Handler knowledge assessment requirement with official URL
    leaf3 = evaluator.add_leaf(
        id="Handler_Knowledge_Assessment_Requirement_With_Official_URL",
        desc="States that the selected organization requires passing a knowledge assessment (or equivalent required test/quiz) AND provides an official organization URL supporting this requirement.",
        parent=node,
        critical=True
    )
    claim3 = f"The cited official page(s) indicate that {org_name} requires passing a knowledge assessment/test/quiz."
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=handler.knowledge_assessment_urls,
        additional_instruction="Look for 'knowledge assessment', 'exam', 'test', or 'quiz' that handlers must pass."
    )


async def build_animal_qualification_checks(evaluator: Evaluator, parent, data: ProgramExtraction):
    node = evaluator.add_parallel(
        id="Animal_Qualifications",
        desc="Therapy animal requirements are documented per the selected organization and constraints.",
        parent=parent,
        critical=True
    )

    org_name = data.selected_organization.name or ""
    animal = data.animal

    # Existence gates
    evaluator.add_custom_node(
        result=bool(animal.min_age_dog and animal.min_age_dog.strip()) and has_urls(animal.min_age_urls),
        id="animal_min_age_url_present",
        desc="Official URL(s) provided for minimum therapy dog age requirement",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(animal.veterinary_health_doc_urls),
        id="animal_vet_doc_url_present",
        desc="Official URL(s) provided for veterinary health documentation",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(animal.vaccination_urls),
        id="animal_vaccination_url_present",
        desc="Official URL(s) provided for vaccination documentation/requirements",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(animal.fitness_to_participate_urls),
        id="animal_fitness_url_present",
        desc="Official URL(s) provided for fitness to participate (free of illness/injury/parasites) requirement",
        parent=node,
        critical=True
    )

    # 1) Minimum therapy dog age requirement with official URL
    leaf1 = evaluator.add_leaf(
        id="Animal_Minimum_Age_Requirement_With_Official_URL",
        desc="States the minimum therapy animal age required by the selected organization AND provides an official organization URL supporting that minimum.",
        parent=node,
        critical=True
    )
    claim1 = f"The minimum age for therapy dogs per {org_name} is {animal.min_age_dog}."
    add_ins1 = (
        "If the organization is Pet Partners, confirm dogs must be at least 1 year old at evaluation (and note that other species like rabbits/guinea pigs/rats are ≥6 months, if mentioned). "
        "Otherwise, verify the minimum age for dogs exactly as stated on the official page."
        if org_is_pet_partners(org_name)
        else "Verify the minimum dog age exactly as stated on the official page."
    )
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=animal.min_age_urls,
        additional_instruction=add_ins1
    )

    # 2) Veterinary health screening documentation with official URL
    leaf2 = evaluator.add_leaf(
        id="Veterinary_Health_Screening_Documentation_With_Official_URL",
        desc="States what current veterinary health screening documentation is required (from a licensed veterinarian) AND provides an official organization URL supporting this requirement.",
        parent=node,
        critical=True
    )
    claim2 = "The cited official page(s) state that current veterinary health screening documentation from a licensed veterinarian is required for participating therapy animals."
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=animal.veterinary_health_doc_urls,
        additional_instruction="Look for health screening/health form/medical clearance from a licensed veterinarian."
    )

    # 3) Vaccination documentation with official URL
    leaf3 = evaluator.add_leaf(
        id="Vaccination_Documentation_With_Official_URL",
        desc="States what current vaccination documentation/requirements apply per the selected organization AND provides an official organization URL supporting these vaccination requirements.",
        parent=node,
        critical=True
    )
    claim3 = "The cited official page(s) specify required current vaccinations and/or vaccination documentation for therapy animals."
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=animal.vaccination_urls,
        additional_instruction="Common examples include rabies, core vaccinations, titers if allowed, and proof-of-vaccination documentation."
    )

    # 4) Fitness to participate with official URL
    leaf4 = evaluator.add_leaf(
        id="Fitness_to_Participate_With_Official_URL",
        desc="States that the animal must be free from illness, injury, and parasites at the time of participation AND provides an official organization URL supporting this condition.",
        parent=node,
        critical=True
    )
    claim4 = "The cited official page(s) state that therapy animals must be free from illness, injury, and external/internal parasites at time of participation/visits."
    await evaluator.verify(
        claim=claim4,
        node=leaf4,
        sources=animal.fitness_to_participate_urls,
        additional_instruction="Wording such as 'in good health', 'no signs of illness', 'free of parasites' is acceptable if clearly required for participation."
    )


async def build_training_standards_checks(evaluator: Evaluator, parent, data: ProgramExtraction):
    node = evaluator.add_parallel(
        id="Training_Standards_Compliance",
        desc="Training duration requirement is documented for the selected organization and checked against IAADP minimum standard.",
        parent=parent,
        critical=True
    )

    org_name = data.selected_organization.name or ""
    tr = data.training

    # Existence gates
    evaluator.add_custom_node(
        result=bool(tr.org_training_duration_statement and tr.org_training_duration_statement.strip()) and has_urls(tr.org_training_duration_urls),
        id="org_training_duration_url_present",
        desc="Official URL(s) provided for the organization's training duration requirement (or explicit non-specification)",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(tr.iaadp_minimum_standard_statement and tr.iaadp_minimum_standard_statement.strip()) and has_urls(tr.iaadp_urls),
        id="iaadp_url_present",
        desc="Official IAADP URL(s) provided for the IAADP minimum standard",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(tr.meets_or_exceeds_statement and tr.meets_or_exceeds_statement.strip()),
        id="comparison_statement_present",
        desc="Answer includes an explicit comparison to IAADP minimum or explicitly states non-specification",
        parent=node,
        critical=True
    )

    # 1) Selected organization's training duration requirement with official URL
    leaf1 = evaluator.add_leaf(
        id="Selected_Organization_Training_Duration_Requirement_With_Official_URL",
        desc="States the selected organization’s training duration requirement (hours and/or timeframe, if specified) AND provides an official organization URL supporting the stated training duration requirement.",
        parent=node,
        critical=True
    )
    claim1 = (
        f"The cited official page(s) from {org_name} support this statement about training duration or that no minimum duration is specified: {tr.org_training_duration_statement}"
    )
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=tr.org_training_duration_urls,
        additional_instruction="If the page indicates no specified minimum duration, that is acceptable; otherwise, confirm hours/timeframe as stated."
    )

    # 2) IAADP minimum standard with official URL
    leaf2 = evaluator.add_leaf(
        id="IAADP_Minimum_Standard_With_Official_URL",
        desc="States the IAADP minimum standard (120 hours over at least 6 months) AND provides an official IAADP URL supporting this standard.",
        parent=node,
        critical=True
    )
    claim2 = "IAADP’s minimum training standard for public access is at least 120 hours over a period of at least 6 months."
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=tr.iaadp_urls,
        additional_instruction="Ensure the page is from the official IAADP website (iaadp.org) and clearly states both 120 hours and a timeframe of at least 6 months."
    )

    # 3) Meets or exceeds IAADP minimum verification (explicit comparison present in the answer)
    leaf3 = evaluator.add_leaf(
        id="Meets_or_Exceeds_IAADP_Minimum_Verification",
        desc="Provides an explicit comparison showing whether the selected organization’s training duration requirement meets or exceeds the IAADP minimum standard (120 hours and ≥6 months), or states that the organization does not specify a duration (in which case it cannot be verified as meeting/exceeding).",
        parent=node,
        critical=True
    )
    claim3 = (
        "The answer explicitly provides a comparison stating whether the selected organization's training requirement meets or exceeds the IAADP minimum "
        "(120 hours over at least 6 months), OR explicitly states that the organization does not specify a duration and thus this cannot be verified."
    )
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=None,  # This check is about presence of explicit comparison in the answer itself.
        additional_instruction=f"Check the answer text to confirm an explicit comparison statement exists. Extracted statement: {tr.meets_or_exceeds_statement or 'None provided'}"
    )


async def build_healthcare_policies_checks(evaluator: Evaluator, parent, data: ProgramExtraction):
    node = evaluator.add_parallel(
        id="Healthcare_Facility_Requirements",
        desc="Healthcare facility written policies and infection prevention measures are documented per constraints.",
        parent=parent,
        critical=True
    )

    pol = data.facility_policies

    # Existence gates
    evaluator.add_custom_node(
        result=has_urls(pol.written_policy_urls),
        id="written_policy_url_present",
        desc="Recognized authority URL(s) provided for written policy requirement",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(pol.hand_hygiene_urls),
        id="hand_hygiene_url_present",
        desc="Recognized authority URL(s) provided for hand hygiene protocols",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_urls(pol.animal_health_screening_urls),
        id="animal_screening_url_present",
        desc="Recognized authority URL(s) provided for animal health screening procedures",
        parent=node,
        critical=True
    )

    # 1) Written policy requirement with official URL
    leaf1 = evaluator.add_leaf(
        id="Written_Policy_Requirement_With_Official_URL",
        desc="States that healthcare facilities must have a written policy governing therapy animal access and requirements AND provides an official URL from a recognized health authority or professional body supporting the need for such written policies/guidelines.",
        parent=node,
        critical=True
    )
    claim1 = "The cited page states that healthcare facilities should/must have a written policy governing animal visitation or therapy animal programs."
    await evaluator.verify(
        claim=claim1,
        node=leaf1,
        sources=pol.written_policy_urls,
        additional_instruction="The URL should be from a recognized authority (e.g., CDC, AVMA, APIC, Joint Commission). Confirm explicit mention of a written policy/guideline for animal programs in healthcare settings."
    )

    # 2) Infection prevention hand hygiene with official URL
    leaf2 = evaluator.add_leaf(
        id="Infection_Prevention_Hand_Hygiene_With_Official_URL",
        desc="States that the program includes infection prevention measures including hand hygiene protocols AND provides an official URL from a recognized health authority or professional body supporting this measure for animals in healthcare settings.",
        parent=node,
        critical=True
    )
    claim2 = "The cited page supports hand hygiene protocols in the context of animal-assisted activities/therapy or animal visitation in healthcare."
    await evaluator.verify(
        claim=claim2,
        node=leaf2,
        sources=pol.hand_hygiene_urls,
        additional_instruction="Confirm the page explicitly addresses hand hygiene related to interactions with animals in healthcare settings."
    )

    # 3) Infection prevention animal health screening procedures with official URL
    leaf3 = evaluator.add_leaf(
        id="Infection_Prevention_Animal_Health_Screening_Procedures_With_Official_URL",
        desc="States that the program includes infection prevention measures including animal health screening procedures AND provides an official URL from a recognized health authority or professional body supporting this measure for animals in healthcare settings.",
        parent=node,
        critical=True
    )
    claim3 = "The cited page supports the need for animal health screening procedures for animals participating in healthcare-based animal visitation/therapy programs."
    await evaluator.verify(
        claim=claim3,
        node=leaf3,
        sources=pol.animal_health_screening_urls,
        additional_instruction="Confirm the page discusses health screening/health checks/eligibility of animals for visits in healthcare environments."
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
    # Initialize evaluator (root is non-critical by framework design)
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

    # Create a critical root-like container to mirror rubric semantics
    rubric_root = evaluator.add_parallel(
        id="Therapy_Animal_Program_Requirements",
        desc="Evaluate all required documentation for establishing and operating a therapy animal program at a healthcare facility per the question and constraints.",
        parent=root,
        critical=True
    )

    # Extraction
    extracted: ProgramExtraction = await evaluator.extract(
        prompt=prompt_extract_program(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction"
    )

    # Optional info for summary/debugging
    evaluator.add_custom_info(
        info={
            "selected_org": extracted.selected_organization.name,
            "org_urls": extracted.selected_organization.official_org_urls,
        },
        info_type="selection_overview",
        info_name="selection_overview"
    )

    # Build and run verification subtrees
    await build_certification_org_checks(evaluator, rubric_root, extracted)
    await build_handler_qualification_checks(evaluator, rubric_root, extracted)
    await build_animal_qualification_checks(evaluator, rubric_root, extracted)
    await build_training_standards_checks(evaluator, rubric_root, extracted)
    await build_healthcare_policies_checks(evaluator, rubric_root, extracted)

    # Return structured summary
    return evaluator.get_summary()