import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ga_cobb_ap_pathway"
TASK_DESCRIPTION = (
    "You currently hold a bachelor's degree and are interested in becoming an assistant principal at Cobb County "
    "Schools in Georgia. What are the complete certification requirements you must fulfill at each stage of this career "
    "pathway, including: (1) What type of initial teaching certificate will you receive in Georgia, and what is its "
    "validity period? (2) What are the requirements to convert your initial teaching certificate to a Professional "
    "certificate, including the amount of teaching experience required in Georgia? (3) What educational degree and "
    "certification level are required for administrative positions in Georgia, specifically for becoming an assistant "
    "principal? (4) What are Cobb County Schools' specific requirements for assistant principal positions regarding "
    "educational credentials, certification level, and teaching experience? (5) What is the minimum number of years this "
    "complete pathway will take from your current position as a bachelor's degree holder to becoming eligible for an "
    "assistant principal position at Cobb County Schools, assuming you complete your master's degree and administrative "
    "certification requirements during your teaching years? Provide specific answers with supporting evidence and URL "
    "references for each requirement."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class TeachingPrereq(BaseModel):
    bachelor_required_text: Optional[str] = None
    ga_program_required_text: Optional[str] = None
    prereq_urls: List[str] = Field(default_factory=list)


class InductionInfo(BaseModel):
    certificate_name: Optional[str] = None
    validity_period: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ProfessionalInfo(BaseModel):
    experience_required_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class AdminEducationInfo(BaseModel):
    masters_required_text: Optional[str] = None
    leadership_program_required_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CertificateLevelInfo(BaseModel):
    required_level_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class AdminExperienceInfo(BaseModel):
    experience_required_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CobbRequirementsInfo(BaseModel):
    masters_required_text: Optional[str] = None
    leadership_level_required_text: Optional[str] = None
    teaching_experience_required_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class CobbScreeningInfo(BaseModel):
    background_drug_screen_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class TimelineInfo(BaseModel):
    required_teaching_years_text: Optional[str] = None
    concurrency_possible_text: Optional[str] = None
    min_timeline_years_text: Optional[str] = None
    teaching_duration_urls: List[str] = Field(default_factory=list)
    masters_timeline_urls: List[str] = Field(default_factory=list)
    total_timeline_urls: List[str] = Field(default_factory=list)


class PathwayExtraction(BaseModel):
    teaching_prereq: Optional[TeachingPrereq] = None
    induction: Optional[InductionInfo] = None
    professional: Optional[ProfessionalInfo] = None
    admin_education: Optional[AdminEducationInfo] = None
    certificate_level: Optional[CertificateLevelInfo] = None
    admin_experience: Optional[AdminExperienceInfo] = None
    cobb_requirements: Optional[CobbRequirementsInfo] = None
    cobb_screening: Optional[CobbScreeningInfo] = None
    timeline: Optional[TimelineInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pathway() -> str:
    return """
Extract the user's provided requirements and cited URLs about the Georgia educator certification pathway and Cobb County Schools assistant principal requirements. Return a JSON object that matches exactly the following schema. Use null for any missing field and [] for any missing URL list. Do not infer new content not present in the answer; only extract what is explicitly stated.

Schema:
{
  "teaching_prereq": {
    "bachelor_required_text": string or null,
    "ga_program_required_text": string or null,
    "prereq_urls": string[]  // URLs cited that support initial teaching prerequisites
  },
  "induction": {
    "certificate_name": string or null,               // what the answer calls the initial certificate (e.g., "Induction")
    "validity_period": string or null,                // validity period text (e.g., "5 years")
    "urls": string[]                                  // URLs cited for induction info
  },
  "professional": {
    "experience_required_text": string or null,       // e.g., "3 years of successful teaching in Georgia"
    "urls": string[]                                  // URLs cited for professional conversion
  },
  "admin_education": {
    "masters_required_text": string or null,          // e.g., "Master's degree is required"
    "leadership_program_required_text": string or null, // e.g., "Complete a state-approved Educational Leadership program"
    "urls": string[]                                  // URLs cited for admin educational requirements
  },
  "certificate_level": {
    "required_level_text": string or null,            // e.g., "Level 5 or higher Educational Leadership certificate"
    "urls": string[]                                  // URLs cited for certificate level requirement
  },
  "admin_experience": {
    "experience_required_text": string or null,       // e.g., "3 years of successful teaching or leadership experience"
    "urls": string[]                                  // URLs cited for admin experience requirement
  },
  "cobb_requirements": {
    "masters_required_text": string or null,          // e.g., "Master's degree required by Cobb County"
    "leadership_level_required_text": string or null, // e.g., "Valid Georgia Leadership certificate L5 or higher"
    "teaching_experience_required_text": string or null, // e.g., "3 years successful teaching"
    "urls": string[]                                  // URLs cited for Cobb-specific requirements
  },
  "cobb_screening": {
    "background_drug_screen_text": string or null,    // e.g., "Background check and drug screen are required"
    "urls": string[]                                  // URLs cited for screening requirements
  },
  "timeline": {
    "required_teaching_years_text": string or null,   // e.g., "At least 3 years of teaching"
    "concurrency_possible_text": string or null,      // e.g., "Master's/leadership program can be completed while teaching"
    "min_timeline_years_text": string or null,        // the answer's claimed minimum total years to eligibility
    "teaching_duration_urls": string[],               // URLs cited to support required teaching years
    "masters_timeline_urls": string[],                // URLs cited to support concurrency timeline
    "total_timeline_urls": string[]                   // URLs cited to support minimum total years calculation
  }
}

Guidance:
- Extract only URLs explicitly present in the answer (plain or markdown links).
- Do not guess missing items; use null or [] if not provided.
- Keep the original phrasing of each extracted *_text field where possible.
"""


# --------------------------------------------------------------------------- #
# Helper utilities for building nodes & verification                          #
# --------------------------------------------------------------------------- #
def _nonempty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


async def _verify_with_urls(
    evaluator: Evaluator,
    node_id: str,
    node_desc: str,
    parent,
    claim: str,
    urls: List[str],
    additional_instruction: str,
    *,
    critical: bool = True,
    extra_prereq_node=None,
):
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=critical
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=additional_instruction,
        extra_prerequisites=[extra_prereq_node] if extra_prereq_node else None
    )
    return leaf


def _add_urls_existence_node(
    evaluator: Evaluator,
    node_id: str,
    node_desc: str,
    parent,
    urls: Optional[List[str]],
    *,
    critical: bool = True
):
    has_any = len(_nonempty_urls(urls)) > 0
    return evaluator.add_custom_node(
        result=has_any,
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=critical
    )


# --------------------------------------------------------------------------- #
# Verification stages                                                         #
# --------------------------------------------------------------------------- #
async def build_teaching_certification_stage(evaluator: Evaluator, parent, ex: PathwayExtraction):
    # Teaching Certification Stage (sequential, critical)
    stage = evaluator.add_sequential(
        id="Teaching_Certification_Stage",
        desc="Verification of requirements for obtaining and advancing through teaching certification in Georgia",
        parent=parent,
        critical=True
    )

    # 1) Initial Teaching Prerequisites (parallel, critical)
    initial_prereq = evaluator.add_parallel(
        id="Initial_Teaching_Prerequisites",
        desc="Verification of prerequisites for initial teaching certification",
        parent=stage,
        critical=True
    )
    t_pr = ex.teaching_prereq or TeachingPrereq()

    prereq_urls_node = _add_urls_existence_node(
        evaluator,
        "Teaching_Prerequisites_URLs",
        "URL references for initial teaching prerequisites",
        parent=initial_prereq,
        urls=t_pr.prereq_urls,
        critical=True  # Adjusted to satisfy framework constraint
    )

    await _verify_with_urls(
        evaluator,
        "Bachelor_Degree_Required",
        "Bachelor's degree from accredited institution is required for teaching certification",
        parent=initial_prereq,
        claim="Georgia requires a bachelor's degree from an accredited institution to qualify for an initial teaching certificate.",
        urls=_nonempty_urls(t_pr.prereq_urls),
        additional_instruction=(
            "Confirm on official Georgia (GaPSC) or other authoritative state sources that a bachelor's degree "
            "is a prerequisite for initial educator certification."
        ),
        critical=True,
        extra_prereq_node=prereq_urls_node
    )

    await _verify_with_urls(
        evaluator,
        "Georgia_Approved_Educator_Program",
        "Completion of Georgia-approved educator preparation program is required",
        parent=initial_prereq,
        claim="Georgia requires completion of a state-approved educator preparation program for initial certification eligibility.",
        urls=_nonempty_urls(t_pr.prereq_urls),
        additional_instruction=(
            "Look for explicit mention that a Georgia-approved (state-approved) educator preparation program is "
            "required for initial certification (e.g., GaPSC program approval)."
        ),
        critical=True,
        extra_prereq_node=prereq_urls_node
    )

    # 2) Induction Certificate Requirements (parallel, critical)
    induction = evaluator.add_parallel(
        id="Induction_Certificate_Requirements",
        desc="Verification of initial certificate type and validity",
        parent=stage,
        critical=True
    )
    ind = ex.induction or InductionInfo()

    induction_urls_node = _add_urls_existence_node(
        evaluator,
        "Induction_URLs",
        "URL references for Induction certificate information",
        parent=induction,
        urls=ind.urls,
        critical=True  # Adjusted to satisfy framework constraint
    )

    await _verify_with_urls(
        evaluator,
        "Induction_Certificate_Type",
        "Initial certificate is called Induction certificate and is valid for 5 years",
        parent=induction,
        claim="In Georgia, the initial educator certificate is called the Induction certificate and it is valid for 5 years.",
        urls=_nonempty_urls(ind.urls),
        additional_instruction=(
            "Verify on GaPSC or equivalent authoritative Georgia sources that the initial certificate is named "
            "'Induction' and that the Induction certificate has a 5-year validity period."
        ),
        critical=True,
        extra_prereq_node=induction_urls_node
    )

    # 3) Professional Certificate Requirements (parallel, critical)
    professional = evaluator.add_parallel(
        id="Professional_Certificate_Requirements",
        desc="Verification of requirements to convert to Professional certificate",
        parent=stage,
        critical=True
    )
    prof = ex.professional or ProfessionalInfo()

    prof_urls_node = _add_urls_existence_node(
        evaluator,
        "Professional_Certificate_URLs",
        "URL references for Professional certificate conversion",
        parent=professional,
        urls=prof.urls,
        critical=True  # Adjusted to satisfy framework constraint
    )

    await _verify_with_urls(
        evaluator,
        "Three_Years_Successful_Teaching",
        "3 years of successful teaching experience in Georgia is required for Professional certificate conversion",
        parent=professional,
        claim="Georgia requires three years of successful teaching experience to convert the Induction certificate to a Professional certificate.",
        urls=_nonempty_urls(prof.urls),
        additional_instruction=(
            "Confirm that the GaPSC (or equivalent official Georgia documents) state a 3-year successful teaching "
            "experience requirement for conversion from Induction to Professional."
        ),
        critical=True,
        extra_prereq_node=prof_urls_node
    )


async def build_administrative_certification_stage(evaluator: Evaluator, parent, ex: PathwayExtraction):
    # Administrative Certification Stage (parallel, critical)
    stage = evaluator.add_parallel(
        id="Administrative_Certification_Stage",
        desc="Verification of requirements for Educational Leadership certification in Georgia",
        parent=parent,
        critical=True
    )

    # A) Educational prerequisites (parallel, critical)
    admin_ed = evaluator.add_parallel(
        id="Administrative_Educational_Requirements",
        desc="Verification of educational prerequisites for administrative certification",
        parent=stage,
        critical=True
    )
    ae = ex.admin_education or AdminEducationInfo()

    admin_ed_urls_node = _add_urls_existence_node(
        evaluator,
        "Admin_Education_URLs",
        "URL references for administrative educational requirements",
        parent=admin_ed,
        urls=ae.urls,
        critical=True  # Adjusted to satisfy framework constraint
    )

    await _verify_with_urls(
        evaluator,
        "Master_Degree_Required",
        "Master's degree is required for Educational Leadership certification",
        parent=admin_ed,
        claim="A master's degree is required for Educational Leadership (administrative) certification in Georgia.",
        urls=_nonempty_urls(ae.urls),
        additional_instruction=(
            "Confirm on GaPSC or official program guidelines that Educational Leadership certification requires "
            "a master's degree (or higher)."
        ),
        critical=True,
        extra_prereq_node=admin_ed_urls_node
    )

    await _verify_with_urls(
        evaluator,
        "Leadership_Program_Required",
        "State-approved Educational Leadership program at master's level or higher must be completed",
        parent=admin_ed,
        claim="Completion of a state-approved Educational Leadership program at the master's level or higher is required for certification in Georgia.",
        urls=_nonempty_urls(ae.urls),
        additional_instruction=(
            "Look for explicit mention of state-approved (GaPSC-approved) Educational Leadership programs that "
            "lead to certification (Tier I/Tier II, as applicable)."
        ),
        critical=True,
        extra_prereq_node=admin_ed_urls_node
    )

    # B) Leadership certificate level (parallel, critical)
    cert_lvl = evaluator.add_parallel(
        id="Leadership_Certificate_Level_Required",
        desc="Verification of required certification level",
        parent=stage,
        critical=True
    )
    cl = ex.certificate_level or CertificateLevelInfo()

    cert_lvl_urls_node = _add_urls_existence_node(
        evaluator,
        "Certificate_Level_URLs",
        "URL references for certificate level requirements",
        parent=cert_lvl,
        urls=cl.urls,
        critical=True  # Adjusted to satisfy framework constraint
    )

    await _verify_with_urls(
        evaluator,
        "Level_Five_Or_Higher",
        "Georgia Educational Leadership Certificate at Level 5 or higher is required for administrative positions",
        parent=cert_lvl,
        claim="Administrative positions such as assistant principal in Georgia require a valid Educational Leadership certificate at Level 5 or higher.",
        urls=_nonempty_urls(cl.urls),
        additional_instruction=(
            "Confirm that Georgia administrative roles (e.g., assistant principal) require Level 5 (master's level) "
            "or higher Educational Leadership certification."
        ),
        critical=True,
        extra_prereq_node=cert_lvl_urls_node
    )

    # C) Administrative experience requirement (parallel, critical)
    admin_exp = evaluator.add_parallel(
        id="Administrative_Experience_Required",
        desc="Verification of experience requirement for administrative certification",
        parent=stage,
        critical=True
    )
    ax = ex.admin_experience or AdminExperienceInfo()

    admin_exp_urls_node = _add_urls_existence_node(
        evaluator,
        "Admin_Experience_URLs",
        "URL references for administrative experience requirements",
        parent=admin_exp,
        urls=ax.urls,
        critical=True  # Adjusted to satisfy framework constraint
    )

    await _verify_with_urls(
        evaluator,
        "Three_Years_Teaching_Experience",
        "3 years of successful teaching or leadership experience is required for Educational Leadership certification",
        parent=admin_exp,
        claim="Georgia requires three years of successful teaching or leadership experience for Educational Leadership certification.",
        urls=_nonempty_urls(ax.urls),
        additional_instruction=(
            "Verify on GaPSC or equivalent that 3 years of successful experience (teaching/leadership) is required "
            "for Educational Leadership certification (e.g., Tier-based criteria)."
        ),
        critical=True,
        extra_prereq_node=admin_exp_urls_node
    )


async def build_cobb_county_requirements(evaluator: Evaluator, parent, ex: PathwayExtraction):
    # Cobb County Requirements (parallel, critical)
    cobb = evaluator.add_parallel(
        id="Cobb_County_Requirements",
        desc="Verification of Cobb County Schools specific requirements for assistant principal positions",
        parent=parent,
        critical=True
    )

    # 1) Credentials & Experience (parallel, critical)
    creds = evaluator.add_parallel(
        id="Cobb_Credentials_And_Experience",
        desc="Verification of Cobb County's educational, certification, and experience requirements",
        parent=cobb,
        critical=True
    )
    cr = ex.cobb_requirements or CobbRequirementsInfo()

    cobb_urls_node = _add_urls_existence_node(
        evaluator,
        "Cobb_Requirements_URLs",
        "URL references for Cobb County requirements",
        parent=creds,
        urls=cr.urls,
        critical=True  # Adjusted to satisfy framework constraint
    )

    await _verify_with_urls(
        evaluator,
        "Cobb_Master_Degree",
        "Cobb County requires master's degree for administrative positions",
        parent=creds,
        claim="Cobb County Schools requires a master's degree for assistant principal (administrative) positions.",
        urls=_nonempty_urls(cr.urls),
        additional_instruction=(
            "Use official Cobb County School District job postings, HR pages, or position descriptions to confirm "
            "that a master's degree is required for an assistant principal."
        ),
        critical=True,
        extra_prereq_node=cobb_urls_node
    )

    await _verify_with_urls(
        evaluator,
        "Cobb_Level_Five_Leadership",
        "Cobb County requires Valid Georgia Educational Leadership Certificate at Level 5 or higher",
        parent=creds,
        claim="Cobb County Schools requires a valid Georgia Educational Leadership certificate at Level 5 or higher for assistant principal roles.",
        urls=_nonempty_urls(cr.urls),
        additional_instruction=(
            "Confirm on Cobb County job postings or HR requirements that L5 (or higher) Educational Leadership "
            "certification is required for assistant principal."
        ),
        critical=True,
        extra_prereq_node=cobb_urls_node
    )

    await _verify_with_urls(
        evaluator,
        "Cobb_Three_Years_Teaching",
        "Cobb County requires 3 years of successful teaching experience for administrative positions",
        parent=creds,
        claim="Cobb County Schools requires at least three years of successful teaching experience for assistant principal positions.",
        urls=_nonempty_urls(cr.urls),
        additional_instruction=(
            "Verify via Cobb County postings/HR documents that three years of successful teaching is required for "
            "assistant principal."
        ),
        critical=True,
        extra_prereq_node=cobb_urls_node
    )

    # 2) Pre-Employment Screening (parallel, adjusted to critical to satisfy framework constraint)
    screening = evaluator.add_parallel(
        id="Cobb_PreEmployment_Screening",
        desc="Verification of pre-employment screening requirements",
        parent=cobb,
        critical=True  # Adjusted: parent is critical; all children must be critical
    )
    cs = ex.cobb_screening or CobbScreeningInfo()

    screening_urls_node = _add_urls_existence_node(
        evaluator,
        "Screening_URLs",
        "URL references for screening requirements",
        parent=screening,
        urls=cs.urls,
        critical=True
    )

    await _verify_with_urls(
        evaluator,
        "Background_Drug_Screen",
        "Criminal background check and drug screen are required prior to employment at Cobb County",
        parent=screening,
        claim="Cobb County School District requires a criminal background check and drug screen prior to employment.",
        urls=_nonempty_urls(cs.urls),
        additional_instruction=(
            "Confirm on Cobb County HR/policy pages or job postings that both background checks and drug screening "
            "are required before employment."
        ),
        critical=True,
        extra_prereq_node=screening_urls_node
    )


async def build_minimum_timeline(evaluator: Evaluator, parent, ex: PathwayExtraction):
    # Minimum Timeline Calculation (sequential, critical)
    timeline = evaluator.add_sequential(
        id="Minimum_Timeline_Calculation",
        desc="Calculation of the minimum total timeline from bachelor's degree holder to assistant principal eligibility",
        parent=parent,
        critical=True
    )
    tl = ex.timeline or TimelineInfo()

    # A) Teaching Experience Duration (parallel, critical)
    teach_dur = evaluator.add_parallel(
        id="Teaching_Experience_Duration",
        desc="Verification of mandatory teaching experience duration",
        parent=timeline,
        critical=True
    )

    teach_dur_urls_node = _add_urls_existence_node(
        evaluator,
        "Teaching_Duration_URLs",
        "URL references for teaching experience duration requirement",
        parent=teach_dur,
        urls=tl.teaching_duration_urls,
        critical=True
    )

    await _verify_with_urls(
        evaluator,
        "Three_Year_Teaching_Period",
        "3 years of teaching experience is mandatory before administrative certification eligibility",
        parent=teach_dur,
        claim="A minimum of three years of successful teaching experience is required before becoming eligible for Educational Leadership certification in Georgia.",
        urls=_nonempty_urls(tl.teaching_duration_urls),
        additional_instruction=(
            "Use GaPSC or authoritative Georgia sources showing a 3-year experience requirement that gates "
            "administrative certification eligibility."
        ),
        critical=True,
        extra_prereq_node=teach_dur_urls_node
    )

    # B) Master's Degree Timeline (parallel, adjusted to critical to satisfy framework constraint)
    masters_tl = evaluator.add_parallel(
        id="Master_Degree_Timeline",
        desc="Consideration of master's degree completion timeline",
        parent=timeline,
        critical=True  # Adjusted: parent critical requires child critical
    )

    masters_tl_urls_node = _add_urls_existence_node(
        evaluator,
        "Master_Timeline_URLs",
        "URL references for master's degree timeline information",
        parent=masters_tl,
        urls=tl.masters_timeline_urls,
        critical=True
    )

    await _verify_with_urls(
        evaluator,
        "Concurrent_Completion_Possible",
        "Master's degree and Educational Leadership program can typically be completed during teaching experience",
        parent=masters_tl,
        claim="It is typically possible to complete a master's degree and an Educational Leadership program while employed as a teacher.",
        urls=_nonempty_urls(tl.masters_timeline_urls),
        additional_instruction=(
            "Look for program/HR language indicating that candidates may pursue master's/leadership programs "
            "concurrently with teaching employment (e.g., part-time, online, cohort formats)."
        ),
        critical=True,
        extra_prereq_node=masters_tl_urls_node
    )

    # C) Total Minimum Years (parallel, critical)
    total = evaluator.add_parallel(
        id="Total_Minimum_Years",
        desc="Calculation of minimum years from bachelor's degree to assistant principal eligibility",
        parent=timeline,
        critical=True
        )
    total_urls_node = _add_urls_existence_node(
        evaluator,
        "Timeline_Calculation_URLs",
        "URL references supporting timeline calculation",
        parent=total,
        urls=tl.total_timeline_urls,
        critical=True
    )

    await _verify_with_urls(
        evaluator,
        "Minimum_Timeline_Value",
        "The minimum timeline is at least 3 years (for required teaching experience), assuming master's degree and leadership program are completed concurrently",
        parent=total,
        claim="Assuming the master's degree and leadership program are completed during the required teaching years, the minimum timeline to eligibility for an assistant principal role is at least 3 years.",
        urls=_nonempty_urls(tl.total_timeline_urls) or _nonempty_urls(tl.teaching_duration_urls),
        additional_instruction=(
            "Accept reasoning that the minimum timeline equals the required teaching-experience duration if the "
            "master's/leadership program can be completed concurrently. Verify that the sources corroborate the "
            "3-year experience requirement and that the leadership certificate is required for assistant principal roles."
        ),
        critical=True,
        extra_prereq_node=total_urls_node
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Georgia/Cobb assistant principal pathway task.
    """

    # Initialize evaluator with a sequential root strategy
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured pathway information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_pathway(),
        template_class=PathwayExtraction,
        extraction_name="pathway_extraction"
    )

    # Build the rubric tree
    # Root critical node for the complete pathway requirements
    # Note: In the framework, a critical parent cannot have non-critical children.
    # Some rubric items originally marked non-critical were adjusted to critical
    # to satisfy this constraint while still reflecting the intended checks.
    complete_node = evaluator.add_sequential(
        id="Complete_Pathway_Requirements",
        desc="Verification of the complete certification pathway, requirements, and minimum timeline for becoming an assistant principal at Cobb County Schools in Georgia, starting from holding only a bachelor's degree",
        parent=root,
        critical=True
    )

    # Teaching certification stage
    await build_teaching_certification_stage(evaluator, complete_node, extracted)

    # Administrative certification stage
    await build_administrative_certification_stage(evaluator, complete_node, extracted)

    # Cobb County specific requirements
    await build_cobb_county_requirements(evaluator, complete_node, extracted)

    # Minimum total timeline
    await build_minimum_timeline(evaluator, complete_node, extracted)

    # Return the structured summary
    return evaluator.get_summary()