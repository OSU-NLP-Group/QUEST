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
TASK_ID = "nys_professional_teacher_certificate_eligibility"
TASK_DESCRIPTION = (
    "A teacher in New York State is currently considering applying for a Professional Teacher Certificate. "
    "To determine their eligibility, what are the specific sequential requirements they must satisfy, "
    "and what documentation or verification is needed at each stage? Provide a comprehensive breakdown of: "
    "(1) the Initial Certificate prerequisites including the required passing score for the Educating All Students exam, "
    "(2) the graduate degree requirement and institutional accreditation standard, "
    "(3) the minimum teaching experience duration measured in days, "
    "(4) the verification process for teaching experience, and "
    "(5) the mentored experience requirement including the specific types of educational institutions where this experience must be completed."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class StageInitial(BaseModel):
    # Requirement text signals
    mentions_initial_certificate: Optional[bool] = None
    mentions_unexpired: Optional[bool] = None
    mentions_5_year_validity: Optional[bool] = None
    validity_years_text: Optional[str] = None
    # Documentation description (as stated in the answer)
    documentation_description: Optional[str] = None
    # URLs explicitly cited in the answer for this stage
    sources: List[str] = Field(default_factory=list)


class StageEAS(BaseModel):
    mentions_eas_requirement: Optional[bool] = None
    exam_code: Optional[str] = None
    passing_score: Optional[str] = None  # Keep as string to tolerate variations (e.g., "520/600")
    mentions_scale_600: Optional[bool] = None
    documentation_description: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StageMasters(BaseModel):
    mentions_masters_degree_required: Optional[bool] = None
    documentation_description: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StageAccreditation(BaseModel):
    mentions_accredited_institution_required: Optional[bool] = None
    documentation_description: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StageExperienceDuration(BaseModel):
    mentions_three_school_years: Optional[bool] = None
    mentions_540_days_paid: Optional[bool] = None
    stated_days_equivalence_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StageExperienceVerification(BaseModel):
    mentions_employing_district_verifies: Optional[bool] = None
    mentions_experience_verification_form: Optional[bool] = None
    form_name_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StageMentored(BaseModel):
    mentions_one_year_mentored: Optional[bool] = None
    location_types: List[str] = Field(default_factory=list)  # e.g., ["NY public school", "BOCES", "special act school district"]
    documentation_description: Optional[str] = None
    notes_typically_first_year: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class ProCertExtraction(BaseModel):
    initial: Optional[StageInitial] = None
    eas: Optional[StageEAS] = None
    masters: Optional[StageMasters] = None
    accreditation: Optional[StageAccreditation] = None
    exp_duration: Optional[StageExperienceDuration] = None
    exp_verification: Optional[StageExperienceVerification] = None
    mentored: Optional[StageMentored] = None
    global_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pro_cert() -> str:
    return """
Extract from the answer a structured breakdown of the sequential eligibility requirements for the New York State Professional Teacher Certificate and the documentation/verification described at each step. Return JSON matching this schema:

{
  "initial": {
    "mentions_initial_certificate": boolean or null,
    "mentions_unexpired": boolean or null,
    "mentions_5_year_validity": boolean or null,
    "validity_years_text": string or null,
    "documentation_description": string or null,
    "sources": string[]   // all URLs explicitly cited for this Initial Certificate requirement
  },
  "eas": {
    "mentions_eas_requirement": boolean or null,
    "exam_code": string or null,               // e.g., "201"
    "passing_score": string or null,           // e.g., "520" or "520/600"
    "mentions_scale_600": boolean or null,     // true if "out of 600" is stated
    "documentation_description": string or null,
    "sources": string[]
  },
  "masters": {
    "mentions_masters_degree_required": boolean or null,
    "documentation_description": string or null,
    "sources": string[]
  },
  "accreditation": {
    "mentions_accredited_institution_required": boolean or null,
    "documentation_description": string or null,
    "sources": string[]
  },
  "exp_duration": {
    "mentions_three_school_years": boolean or null,
    "mentions_540_days_paid": boolean or null,
    "stated_days_equivalence_text": string or null, // e.g., "three school years equals 540 days of paid teaching"
    "sources": string[]
  },
  "exp_verification": {
    "mentions_employing_district_verifies": boolean or null,
    "mentions_experience_verification_form": boolean or null,
    "form_name_text": string or null, // e.g., "Experience Verification Form"
    "sources": string[]
  },
  "mentored": {
    "mentions_one_year_mentored": boolean or null,
    "location_types": string[] , // capture phrases like "NY public school", "BOCES", "special act school district"
    "documentation_description": string or null,
    "notes_typically_first_year": boolean or null,
    "sources": string[]
  },
  "global_sources": string[] // ALL URLs cited anywhere in the answer, deduplicated
}

Rules:
- Extract ONLY what is explicitly present in the answer.
- Do not infer or invent values.
- For URLs, extract the actual links the answer provides (including markdown links). If no URLs are present for a field, return an empty array for that field.
- Use booleans to indicate whether the answer explicitly states a condition (true), explicitly contradicts (false), or is silent/unclear (null).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if not isinstance(x, str):
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _choose_sources(primary: Optional[List[str]], fallback: Optional[List[str]]) -> Optional[List[str]]:
    cand: List[str] = []
    if primary:
        cand.extend(primary)
    if not cand and fallback:
        cand.extend(fallback)
    cand = [u for u in cand if isinstance(u, str) and len(u.strip()) > 0]
    cand = _unique_preserve_order(cand)
    return cand if cand else None


# --------------------------------------------------------------------------- #
# Verification functions per stage                                            #
# --------------------------------------------------------------------------- #
async def verify_stage_1(evaluator: Evaluator, parent_node, ex: ProCertExtraction) -> None:
    stage_node = evaluator.add_parallel(
        id="Stage_1_Initial_Certificate_Prerequisites",
        desc="Covers the Initial Certificate prerequisite details, including the EAS exam requirement and documentation/verification for each prerequisite.",
        parent=parent_node,
        critical=True
    )

    # Prepare sources
    init_sources = _choose_sources(ex.initial.sources if ex.initial else None, ex.global_sources)
    eas_sources = _choose_sources(ex.eas.sources if ex.eas else None, ex.global_sources)

    # 1.A Initial Certificate requirement (valid/unexpired, 5-year validity)
    leaf_ic = evaluator.add_leaf(
        id="Initial_Certificate_Requirement",
        desc="States the candidate must hold a valid (unexpired) New York State Initial Teaching Certificate (5-year validity from issuance).",
        parent=stage_node,
        critical=True
    )
    claim_ic = (
        "For New York State Professional (Classroom Teaching) certification eligibility, "
        "the applicant must hold a valid (unexpired) NYS Initial Teaching Certificate, "
        "and the Initial Certificate is valid for five years from the date of issuance."
    )
    await evaluator.verify(
        claim=claim_ic,
        node=leaf_ic,
        sources=init_sources,
        additional_instruction="Verify this requirement specifically for New York State teacher certification. Prefer official NYSED/TEACH sources. Allow synonymous phrasing."
    )

    # 1.B EAS requirement: pass EAS (code 201) with minimum passing score 520 out of 600
    leaf_eas = evaluator.add_leaf(
        id="EAS_Requirement",
        desc="States the candidate must have passed the Educating All Students (EAS) test (exam code 201) with a minimum passing score of 520 out of 600.",
        parent=stage_node,
        critical=True
    )
    claim_eas = (
        "The Educating All Students (EAS) test (exam code 201) requires a minimum passing score of 520 out of 600 "
        "for New York State teacher certification eligibility."
    )
    await evaluator.verify(
        claim=claim_eas,
        node=leaf_eas,
        sources=eas_sources,
        additional_instruction="Confirm both the exam code (201) and the passing score threshold (520 out of 600) from official or authoritative NYS certification resources."
    )

    # 1.C Documentation described for Initial Certificate status/validity (answer-level presence check)
    leaf_ic_doc = evaluator.add_leaf(
        id="Initial_Certificate_Documentation_Described",
        desc="Describes documentation/verification needed to demonstrate Initial Certificate status/validity (without introducing specific sources/bodies not stated in constraints).",
        parent=stage_node,
        critical=True
    )
    claim_ic_doc = (
        "The answer includes a description of what documentation or verification is needed to demonstrate Initial "
        "Certificate status/validity (e.g., TEACH account records, certificate lookup, or similar proofs)."
    )
    await evaluator.verify(
        claim=claim_ic_doc,
        node=leaf_ic_doc,
        sources=None,
        additional_instruction="Judge this by checking the ANSWER text only. Pass if the answer clearly describes how to document/verify Initial Certificate status or validity."
    )

    # 1.D Documentation described for EAS passing result/score (answer-level presence check)
    leaf_eas_doc = evaluator.add_leaf(
        id="EAS_Documentation_Described",
        desc="Describes documentation/verification needed to demonstrate the EAS passing result/score (without introducing specific sources/bodies not stated in constraints).",
        parent=stage_node,
        critical=True
    )
    claim_eas_doc = (
        "The answer includes a description of what documentation or verification is needed to demonstrate a passing "
        "Educating All Students (EAS) test result/score (e.g., official score report)."
    )
    await evaluator.verify(
        claim=claim_eas_doc,
        node=leaf_eas_doc,
        sources=None,
        additional_instruction="Judge this by checking the ANSWER text only. Pass if the answer clearly describes how to document/verify an EAS passing result/score."
    )


async def verify_stage_2(evaluator: Evaluator, parent_node, ex: ProCertExtraction) -> None:
    stage_node = evaluator.add_parallel(
        id="Stage_2_Graduate_Degree_And_Accreditation",
        desc="Covers the master's degree requirement and the institutional accreditation standard, plus documentation/verification for each.",
        parent=parent_node,
        critical=True
    )

    masters_sources = _choose_sources(ex.masters.sources if ex.masters else None, ex.global_sources)
    accred_sources = _choose_sources(ex.accreditation.sources if ex.accreditation else None, ex.global_sources)

    # 2.A Master's degree requirement
    leaf_md = evaluator.add_leaf(
        id="Masters_Degree_Requirement",
        desc="States the candidate must complete an appropriate master's degree.",
        parent=stage_node,
        critical=True
    )
    claim_md = "For the NYS Professional (Classroom Teaching) Certificate, completion of an appropriate master's degree is required."
    await evaluator.verify(
        claim=claim_md,
        node=leaf_md,
        sources=masters_sources,
        additional_instruction="Verify specifically for New York State Professional (Classroom Teaching) certification. Prefer official NYSED sources."
    )

    # 2.B Accredited institution requirement
    leaf_accred = evaluator.add_leaf(
        id="Accredited_Institution_Requirement",
        desc="States the master's degree must be from an accredited institution.",
        parent=stage_node,
        critical=True
    )
    claim_accred = "The master's degree must be earned from an accredited institution to qualify for the NYS Professional (Classroom Teaching) Certificate."
    await evaluator.verify(
        claim=claim_accred,
        node=leaf_accred,
        sources=accred_sources,
        additional_instruction="Confirm that NYS requires the master's degree to come from an accredited institution. Prefer official NYSED references."
    )

    # 2.C Documentation described for master's degree completion (answer-level presence check)
    leaf_md_doc = evaluator.add_leaf(
        id="Masters_Degree_Documentation_Described",
        desc="Describes documentation/verification used to demonstrate master's degree completion (without adding specific accreditors not stated in constraints).",
        parent=stage_node,
        critical=True
    )
    claim_md_doc = "The answer describes what documentation or verification is used to demonstrate completion of the master's degree (e.g., official transcripts)."
    await evaluator.verify(
        claim=claim_md_doc,
        node=leaf_md_doc,
        sources=None,
        additional_instruction="Judge based on the ANSWER text only. Pass if the answer clearly describes documentation/verification for master's completion."
    )

    # 2.D Documentation described for accreditation (answer-level presence check)
    leaf_accred_doc = evaluator.add_leaf(
        id="Accreditation_Documentation_Described",
        desc="Describes documentation/verification used to demonstrate the institution meets the accreditation standard (without adding specific accreditors not stated in constraints).",
        parent=stage_node,
        critical=True
    )
    claim_accred_doc = "The answer describes how accreditation of the granting institution would be documented or verified at a high level."
    await evaluator.verify(
        claim=claim_accred_doc,
        node=leaf_accred_doc,
        sources=None,
        additional_instruction="Judge based on the ANSWER text only. Pass if the answer includes a reasonable description of accreditation-related documentation/verification."
    )


async def verify_stage_3(evaluator: Evaluator, parent_node, ex: ProCertExtraction) -> None:
    stage_node = evaluator.add_parallel(
        id="Stage_3_Teaching_Experience_Duration",
        desc="Covers the minimum paid teaching experience requirement measured in days.",
        parent=parent_node,
        critical=True
    )

    exp_sources = _choose_sources(ex.exp_duration.sources if ex.exp_duration else None, ex.global_sources)

    leaf_exp_days = evaluator.add_leaf(
        id="Teaching_Experience_Duration_Requirement",
        desc="States the candidate must complete three school years of teaching experience, equivalent to 540 days of paid teaching.",
        parent=stage_node,
        critical=True
    )
    claim_exp_days = (
        "Eligibility for the NYS Professional (Classroom Teaching) Certificate requires three school years of teaching experience, "
        "equivalent to 540 days of paid teaching."
    )
    await evaluator.verify(
        claim=claim_exp_days,
        node=leaf_exp_days,
        sources=exp_sources,
        additional_instruction="Verify this specific quantity (three school years = 540 paid days) for NYS Professional certification."
    )


async def verify_stage_4(evaluator: Evaluator, parent_node, ex: ProCertExtraction) -> None:
    stage_node = evaluator.add_parallel(
        id="Stage_4_Teaching_Experience_Verification_Process",
        desc="Covers the formal verification process for teaching experience, including the verifier and required form.",
        parent=parent_node,
        critical=True
    )

    ver_sources = _choose_sources(ex.exp_verification.sources if ex.exp_verification else None, ex.global_sources)

    leaf_ver = evaluator.add_leaf(
        id="Teaching_Experience_Verification_Requirement",
        desc="States teaching experience must be verified by the employing school district or institution through an Experience Verification Form.",
        parent=stage_node,
        critical=True
    )
    claim_ver = (
        "Teaching experience for the NYS Professional (Classroom Teaching) Certificate must be verified by the employing school "
        "district or institution using an Experience Verification Form."
    )
    await evaluator.verify(
        claim=claim_ver,
        node=leaf_ver,
        sources=ver_sources,
        additional_instruction="Confirm the verifier (employing district/institution) and the use of an Experience Verification Form per NYS requirements."
    )


async def verify_stage_5(evaluator: Evaluator, parent_node, ex: ProCertExtraction) -> None:
    # NOTE: To allow a non-critical child within this stage, we set this stage node as non-critical.
    stage_node = evaluator.add_parallel(
        id="Stage_5_Mentored_Experience_Requirement",
        desc="Covers the mentored teaching experience requirement and the allowed institution types, plus documentation/verification described.",
        parent=parent_node,
        critical=False
    )

    ment_sources = _choose_sources(ex.mentored.sources if ex.mentored else None, ex.global_sources)

    # 5.A One year of mentored teaching experience (critical)
    leaf_mentored_year = evaluator.add_leaf(
        id="Mentored_Experience_Duration",
        desc="States the candidate must complete one year of mentored teaching experience.",
        parent=stage_node,
        critical=True
    )
    claim_mentored_year = "The NYS Professional (Classroom Teaching) Certificate requires one year of mentored teaching experience."
    await evaluator.verify(
        claim=claim_mentored_year,
        node=leaf_mentored_year,
        sources=ment_sources,
        additional_instruction="Verify the one-year mentored experience requirement for NYS certification from authoritative sources."
    )

    # 5.B Allowed institution types (critical)
    leaf_mentored_locations = evaluator.add_leaf(
        id="Mentored_Experience_Location_Types",
        desc="States the mentored experience must take place in a New York public school, BOCES, or special act school district.",
        parent=stage_node,
        critical=True
    )
    claim_mentored_locations = (
        "The required mentored experience must take place in a New York public school, a BOCES, or a special act school district."
    )
    await evaluator.verify(
        claim=claim_mentored_locations,
        node=leaf_mentored_locations,
        sources=ment_sources,
        additional_instruction="Confirm these specific allowed setting types for mentored experience per NYSED requirements."
    )

    # 5.C Documentation described for mentored experience (critical, answer-level presence)
    leaf_mentored_doc = evaluator.add_leaf(
        id="Mentored_Experience_Documentation_Described",
        desc="Describes documentation/verification used to demonstrate completion of the mentored experience (without adding specific officials/sources not stated in constraints).",
        parent=stage_node,
        critical=True
    )
    claim_mentored_doc = "The answer describes how completion of the mentored experience would be documented or verified."
    await evaluator.verify(
        claim=claim_mentored_doc,
        node=leaf_mentored_doc,
        sources=None,
        additional_instruction="Judge by ANSWER text only. Pass if the answer explains how mentored experience is documented/verified."
    )

    # 5.D Notes typically first year (non-critical)
    leaf_mentored_first_year = evaluator.add_leaf(
        id="Mentored_Experience_Typically_First_Year_Noted",
        desc="Notes that mentored experience is typically during the first year of teaching, without treating this as a strict mandatory condition.",
        parent=stage_node,
        critical=False
    )
    claim_mentored_first_year = (
        "The answer notes that mentored experience is typically completed during the first year of teaching, and does not present it as strictly mandatory."
    )
    await evaluator.verify(
        claim=claim_mentored_first_year,
        node=leaf_mentored_first_year,
        sources=None,
        additional_instruction="Judge by ANSWER text only. This is informational, not a hard requirement; pass if the answer mentions it appropriately."
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
    model: str = "o4-mini"
) -> Dict:
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Sequential stage-by-stage evaluation
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

    # IMPORTANT: To allow a non-critical leaf within Stage 5, keep root as NON-CRITICAL.
    # This avoids the framework's constraint that a critical parent cannot have non-critical children.
    root.critical = False

    # Extract structured info from the answer
    extracted: ProCertExtraction = await evaluator.extract(
        prompt=prompt_extract_pro_cert(),
        template_class=ProCertExtraction,
        extraction_name="pro_cert_extraction"
    )

    # Add an explicit ground-truth reference outline (for transparency in results)
    evaluator.add_ground_truth({
        "expected_requirements_outline": {
            "stage_1_initial_certificate": {
                "must_hold_valid_initial": True,
                "initial_validity_years": 5,
                "eas_exam_code": "201",
                "eas_passing_score": "520/600"
            },
            "stage_2_graduate_and_accreditation": {
                "masters_degree_required": True,
                "from_accredited_institution": True
            },
            "stage_3_experience_duration": {
                "three_school_years": True,
                "equivalent_days_paid": 540
            },
            "stage_4_experience_verification": {
                "verified_by_employing_district_or_institution": True,
                "experience_verification_form": True
            },
            "stage_5_mentored_experience": {
                "one_year_required": True,
                "location_types": ["NY public school", "BOCES", "special act school district"],
                "typically_first_year": "informational, non-mandatory"
            }
        }
    })

    # Build the rubric tree (root-level container for the professional eligibility evaluation)
    # The rubric JSON names the top as "Professional_Certificate_Eligibility"
    top = evaluator.add_sequential(
        id="Professional_Certificate_Eligibility",
        desc="Evaluate whether the answer provides the complete, stage-by-stage (sequential) eligibility requirements and associated documentation/verification for the NYS Professional Teacher Certificate, per the provided constraints.",
        parent=root,
        critical=False  # Keep non-critical to permit mixed criticality in descendants
    )

    # Verify each stage
    await verify_stage_1(evaluator, top, extracted)
    await verify_stage_2(evaluator, top, extracted)
    await verify_stage_3(evaluator, top, extracted)
    await verify_stage_4(evaluator, top, extracted)
    await verify_stage_5(evaluator, top, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()