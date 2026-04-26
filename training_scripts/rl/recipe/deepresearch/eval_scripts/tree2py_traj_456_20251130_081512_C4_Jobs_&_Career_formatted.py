import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "ihsa_nonfaculty_head_fb_coach_requirements"
TASK_DESCRIPTION = """
What are the mandatory requirements that a non-faculty candidate must meet to become certified as a high school head football coach in Illinois according to Illinois High School Association (IHSA) regulations? Your answer must include: (1) the minimum age requirement, (2) the complete coaching education certification requirements including all required course components, (3) the approved certification provider requirements, (4) any mandatory health and safety certifications, and (5) background screening requirements. Additionally, provide information about (6) any exemptions available for candidates with teaching credentials, and (7) continuing education requirements for maintaining certification.
"""


class IHSARequirementsExtraction(BaseModel):
    """Structured extraction of IHSA non-faculty head coach certification requirements, as stated in the answer."""
    # Core requirement fields
    min_age: Optional[str] = None

    coaching_components: List[str] = Field(default_factory=list)
    providers: List[str] = Field(default_factory=list)

    health_safety_certs: List[str] = Field(default_factory=list)
    background_screening: Optional[str] = None

    # Additional information fields
    teaching_credential_exemption: Optional[str] = None

    continuing_ed_hours_total: Optional[str] = None
    continuing_ed_hours_sport_specific: Optional[str] = None
    continuing_ed_cycle: Optional[str] = None

    # Source URLs explicitly cited in the answer for each requirement category
    sources_min_age: List[str] = Field(default_factory=list)
    sources_coaching: List[str] = Field(default_factory=list)
    sources_providers: List[str] = Field(default_factory=list)
    sources_health_safety: List[str] = Field(default_factory=list)
    sources_background: List[str] = Field(default_factory=list)
    sources_teaching_exemption: List[str] = Field(default_factory=list)
    sources_cont_ed: List[str] = Field(default_factory=list)


def prompt_extract_requirements() -> str:
    return """
    Extract, exactly as stated in the answer, the IHSA requirements for a non-faculty high school head football coach certification.
    Return a JSON object with the following fields:

    Core requirement fields (strings or arrays of strings):
    - min_age: the stated minimum age requirement (e.g., "19")
    - coaching_components: array listing each required course component explicitly named in the answer (e.g., ["general coaching principles", "sports first aid", "IHSA state component/by-law exam"])
    - providers: array listing any approved certification providers as named in the answer (e.g., ["ASEP", "NFHS"])
    - health_safety_certs: array of mandatory health and safety certifications named (e.g., ["concussion certification"])
    - background_screening: a short phrase summarizing the background check requirement (e.g., "pass background screening")

    Additional information fields:
    - teaching_credential_exemption: description if the answer states any exemption for candidates with an Illinois Professional Educator License (PEL) or teaching credential
    - continuing_ed_hours_total: total hours required per cycle (e.g., "5")
    - continuing_ed_hours_sport_specific: required sports-specific hours per cycle (e.g., "2")
    - continuing_ed_cycle: the cycle duration (e.g., "every two years")

    For each category, also extract arrays of explicit URL sources mentioned in the answer:
    - sources_min_age
    - sources_coaching
    - sources_providers
    - sources_health_safety
    - sources_background
    - sources_teaching_exemption
    - sources_cont_ed

    IMPORTANT:
    - Only include information explicitly present in the answer.
    - For URLs, extract the actual link targets. Accept plain URLs or markdown links; return the URL.
    - If a field is not mentioned, set it to null (for strings) or [] (for arrays).
    - Do not invent or infer any information; do not add any URLs that are not explicitly present in the answer.
    """


async def verify_min_age(evaluator: Evaluator, parent_node, ex: IHSARequirementsExtraction) -> None:
    group = evaluator.add_parallel(
        id="min_age_group",
        desc="Minimum age requirement verification",
        parent=parent_node,
        critical=True,
    )

    # Critical existence of sources for evidence-based verification
    age_sources_exist = bool(ex.sources_min_age)
    evaluator.add_custom_node(
        result=age_sources_exist,
        id="min_age_sources_provided",
        desc="Sources for minimum age requirement are provided",
        parent=group,
        critical=True
    )

    # Verify that IHSA minimum age is 19 for non-faculty head coaches
    leaf = evaluator.add_leaf(
        id="min_age_is_19_supported",
        desc="IHSA requires non-faculty head coaches to be at least 19 years old",
        parent=group,
        critical=True
    )
    claim = "IHSA regulations set the minimum age for non-faculty high school head coaches at 19 years old."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ex.sources_min_age,
        additional_instruction="Look for IHSA By-Laws or Board Policy language that explicitly states the minimum age for non-faculty head coaches is 19."
    )


async def verify_coaching_certification(evaluator: Evaluator, parent_node, ex: IHSARequirementsExtraction) -> None:
    group = evaluator.add_parallel(
        id="coaching_cert_group",
        desc="Coaching education certification requirements verification",
        parent=parent_node,
        critical=True
    )

    sources_exist = bool(ex.sources_coaching)
    evaluator.add_custom_node(
        result=sources_exist,
        id="coaching_sources_provided",
        desc="Sources for coaching education certification components are provided",
        parent=group,
        critical=True
    )

    # Component: general coaching principles
    comp_general = evaluator.add_leaf(
        id="coaching_component_general_principles_supported",
        desc="IHSA requires a general coaching principles course as part of the approved coaching education",
        parent=group,
        critical=True
    )
    claim_general = "IHSA requires completion of a general coaching principles course within the approved coaching education bundle."
    await evaluator.verify(
        claim=claim_general,
        node=comp_general,
        sources=ex.sources_coaching,
        additional_instruction="Confirm the IHSA policy or approved provider bundle lists a 'Coaching Principles' or equivalent general coaching course as required."
    )

    # Component: sports first aid
    comp_first_aid = evaluator.add_leaf(
        id="coaching_component_sports_first_aid_supported",
        desc="IHSA requires a sports first aid course as part of the approved coaching education",
        parent=group,
        critical=True
    )
    claim_first_aid = "IHSA requires completion of a sports first aid course within the approved coaching education bundle."
    await evaluator.verify(
        claim=claim_first_aid,
        node=comp_first_aid,
        sources=ex.sources_coaching,
        additional_instruction="Confirm the IHSA policy or approved provider bundle lists 'Sports First Aid' or an equivalent first aid course as required."
    )

    # Component: IHSA state component/by-law examination
    comp_state = evaluator.add_leaf(
        id="coaching_component_state_bylaw_exam_supported",
        desc="IHSA requires completion of an IHSA state component/by-law examination",
        parent=group,
        critical=True
    )
    claim_state = "IHSA requires a state component/by-law examination specific to IHSA as part of the approved coaching education."
    await evaluator.verify(
        claim=claim_state,
        node=comp_state,
        sources=ex.sources_coaching,
        additional_instruction="Confirm there is an IHSA-specific state component or by-laws exam required to complete certification."
    )


async def verify_approved_provider(evaluator: Evaluator, parent_node, ex: IHSARequirementsExtraction) -> None:
    group = evaluator.add_parallel(
        id="approved_provider_group",
        desc="Approved certification provider requirement verification",
        parent=parent_node,
        critical=True
    )

    sources_exist = bool(ex.sources_providers)
    evaluator.add_custom_node(
        result=sources_exist,
        id="providers_sources_provided",
        desc="Sources for approved certification providers are provided",
        parent=group,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="provider_approved_by_ihsa_supported",
        desc="Coaching certification must be obtained from IHSA Board-approved providers (e.g., ASEP, NFHS)",
        parent=group,
        critical=True
    )
    claim = "IHSA requires that coaching certification be obtained from IHSA Board-approved providers, such as ASEP or NFHS."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ex.sources_providers,
        additional_instruction="Verify the IHSA policy listing board-approved certification providers and confirm examples include ASEP or NFHS."
    )


async def verify_concussion_cert(evaluator: Evaluator, parent_node, ex: IHSARequirementsExtraction) -> None:
    group = evaluator.add_parallel(
        id="concussion_cert_group",
        desc="Concussion certification requirement verification",
        parent=parent_node,
        critical=True
    )

    sources_exist = bool(ex.sources_health_safety)
    evaluator.add_custom_node(
        result=sources_exist,
        id="health_safety_sources_provided",
        desc="Sources for mandatory health and safety certifications are provided",
        parent=group,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="concussion_cert_required_supported",
        desc="IHSA mandates concussion certification for coaches",
        parent=group,
        critical=True
    )
    claim = "IHSA mandates that coaches complete concussion certification."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ex.sources_health_safety,
        additional_instruction="Confirm IHSA requires completion of a concussion training/certification (e.g., NFHS Concussion in Sports) for coaches."
    )


async def verify_background_check(evaluator: Evaluator, parent_node, ex: IHSARequirementsExtraction) -> None:
    group = evaluator.add_parallel(
        id="background_check_group",
        desc="Background screening requirement verification",
        parent=parent_node,
        critical=True
    )

    sources_exist = bool(ex.sources_background)
    evaluator.add_custom_node(
        result=sources_exist,
        id="background_sources_provided",
        desc="Sources for background screening requirement are provided",
        parent=group,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="background_check_required_supported",
        desc="IHSA requires candidates to pass background screening",
        parent=group,
        critical=True
    )
    claim = "IHSA requires candidates to pass a background screening process."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ex.sources_background,
        additional_instruction="Verify that IHSA policy requires background checks/screening for non-faculty head coaches."
    )


async def verify_teaching_exemption(evaluator: Evaluator, parent_node, ex: IHSARequirementsExtraction) -> None:
    # Optional information, verify in sequential order: sources exist -> claim check
    group = evaluator.add_sequential(
        id="teaching_exemption_group",
        desc="Teaching credential exemption verification",
        parent=parent_node,
        critical=False
    )

    sources_exist = bool(ex.sources_teaching_exemption)
    evaluator.add_custom_node(
        result=sources_exist,
        id="teaching_exemption_sources_provided",
        desc="Sources for teaching credential exemption are provided",
        parent=group,
        critical=False
    )

    leaf = evaluator.add_leaf(
        id="teaching_credential_exemption_supported",
        desc="Candidates with a valid Illinois PEL allowing unsupervised classroom teaching are exempt from the three-part coaching education requirement",
        parent=group,
        critical=False
    )
    claim = "Candidates with a valid Illinois Professional Educator License (PEL) permitting unsupervised classroom teaching are exempt from the three-part coaching education bundle requirement."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ex.sources_teaching_exemption,
        additional_instruction="Confirm IHSA policy states that holders of a valid PEL (unsupervised classroom teaching privilege) are exempt from the standard three-course coaching education bundle."
    )


async def verify_continuing_education(evaluator: Evaluator, parent_node, ex: IHSARequirementsExtraction) -> None:
    # Optional information, verify in sequential order: sources exist -> multiple specific checks
    group = evaluator.add_sequential(
        id="continuing_education_group",
        desc="Continuing education requirements verification",
        parent=parent_node,
        critical=False
    )

    sources_exist = bool(ex.sources_cont_ed)
    evaluator.add_custom_node(
        result=sources_exist,
        id="cont_ed_sources_provided",
        desc="Sources for continuing education requirements are provided",
        parent=group,
        critical=False
    )

    # Check total hours and cycle
    leaf_total_cycle = evaluator.add_leaf(
        id="cont_ed_total_hours_cycle_supported",
        desc="Certified coaches must complete at least 5 hours of continuing education every two years",
        parent=group,
        critical=False
    )
    claim_total_cycle = "Certified coaches must complete at least 5 hours of continuing education every two years."
    await evaluator.verify(
        claim=claim_total_cycle,
        node=leaf_total_cycle,
        sources=ex.sources_cont_ed,
        additional_instruction="Verify IHSA or IHSA-approved provider policy specifying at least 5 hours every two years."
    )

    # Check sports-specific hours
    leaf_sport_specific = evaluator.add_leaf(
        id="cont_ed_sport_specific_hours_supported",
        desc="At least 2 hours of the continuing education every two years must be sports-specific",
        parent=group,
        critical=False
    )
    claim_sport_specific = "At least 2 hours of the required continuing education every two years must be sports-specific."
    await evaluator.verify(
        claim=claim_sport_specific,
        node=leaf_sport_specific,
        sources=ex.sources_cont_ed,
        additional_instruction="Verify IHSA or IHSA-approved provider policy specifying at least 2 sports-specific hours within the biennial requirement."
    )


async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the IHSA non-faculty high school head football coach certification requirements.
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
        default_model=model
    )

    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=IHSARequirementsExtraction,
        extraction_name="ihsa_requirements_extraction"
    )

    # Build verification tree with mandatory and optional sections to respect critical constraints
    mandatory = evaluator.add_parallel(
        id="mandatory_requirements",
        desc="Mandatory IHSA requirements for non-faculty high school head football coach certification",
        parent=root,
        critical=True
    )

    optional = evaluator.add_parallel(
        id="additional_information",
        desc="Additional IHSA information (exemptions and continuing education for certification maintenance)",
        parent=root,
        critical=False
    )

    # Mandatory checks
    await verify_min_age(evaluator, mandatory, extracted)
    await verify_coaching_certification(evaluator, mandatory, extracted)
    await verify_approved_provider(evaluator, mandatory, extracted)
    await verify_concussion_cert(evaluator, mandatory, extracted)
    await verify_background_check(evaluator, mandatory, extracted)

    # Optional checks
    await verify_teaching_exemption(evaluator, optional, extracted)
    await verify_continuing_education(evaluator, optional, extracted)

    return evaluator.get_summary()