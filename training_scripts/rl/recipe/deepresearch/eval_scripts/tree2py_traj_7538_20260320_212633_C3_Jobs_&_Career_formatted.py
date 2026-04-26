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
TASK_ID = "nyc_ap_pathway"
TASK_DESCRIPTION = """
Starting with someone who holds a bachelor's degree and wants to become an Assistant Principal in New York City public schools, identify the complete step-by-step certification pathway they must follow from Initial teaching certificate through School Building Leader certificate. For each certification stage in the pathway, provide: (1) The name and validity period of each certificate, (2) All educational requirements (degrees, GPA minimums, coursework), (4) All experience requirements (years, mentorship), (3) All examination requirements, and (5) Any program completion requirements. Your answer should trace the sequential progression through each required certification level, documenting every requirement at each stage with supporting reference URLs from official New York State Education Department or NYC Department of Education sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StageInitialExtraction(BaseModel):
    name: Optional[str] = None
    validity_period: Optional[str] = None
    degree_requirement: Optional[str] = None
    gpa_minimum: Optional[str] = None  # e.g., "2.5"
    exam_requirements: List[str] = Field(default_factory=list)  # e.g., ["EAS", "edTPA", "CST"]
    observation_hours: Optional[str] = None  # e.g., "100 hours"
    student_teaching_days: Optional[str] = None  # e.g., "40 days"
    citations: List[str] = Field(default_factory=list)


class StageProfessionalExtraction(BaseModel):
    name: Optional[str] = None
    validity_policy: Optional[str] = None  # description of validity/maintenance
    degree_requirement: Optional[str] = None
    content_core_grad_hours: Optional[str] = None  # e.g., "12"
    experience_years: Optional[str] = None  # e.g., "3"
    mentorship_policy: Optional[str] = None  # text describing mentorship/exception
    ctle_hours_per_5_years: Optional[str] = None  # e.g., "100"
    citations: List[str] = Field(default_factory=list)


class StageSBLExtraction(BaseModel):
    name: Optional[str] = None
    validity_policy: Optional[str] = None
    degree_requirement: Optional[str] = None
    gpa_minimum: Optional[str] = None  # e.g., "3.0"
    experience_requirement: Optional[str] = None  # text describing 3 years teaching/PPS in PK-12
    program_credits: Optional[str] = None  # e.g., "30"
    exams_required: List[str] = Field(default_factory=list)  # e.g., ["School Building Leader Assessment"]
    citations: List[str] = Field(default_factory=list)


class PathwayExtraction(BaseModel):
    context_citations: List[str] = Field(default_factory=list)  # to support the NYC requires NYSED certification statement
    initial: Optional[StageInitialExtraction] = None
    professional: Optional[StageProfessionalExtraction] = None
    sbl: Optional[StageSBLExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pathway() -> str:
    return """
    Extract structured information from the answer about the certification pathway to become an Assistant Principal in NYC public schools, starting from an Initial teaching certificate through the School Building Leader (SBL) certificate.

    IMPORTANT: Extract ONLY information explicitly stated in the answer. Do not infer or add anything not present. For URLs, extract ONLY actual URLs that appear in the answer. Prefer official sources (nysed.gov or schools.nyc.gov) if present.

    Provide the following fields:

    1) context_citations: An array of official URLs (if any) the answer cites to support the statement that NYC DOE teachers must hold New York State teacher certification from NYSED.

    2) initial: Object for the Initial teaching certificate stage with fields:
       - name: The certificate name as given (e.g., "Initial Certificate").
       - validity_period: The validity period text as given (e.g., "5 years", "valid for five years").
       - degree_requirement: Text for degree requirement (e.g., "bachelor's degree", and any GPA constraint).
       - gpa_minimum: The numeric GPA minimum mentioned for Initial, if any (e.g., "2.5"). Use only the numeric fragment as a string if possible; if not provided, set null.
       - exam_requirements: Array of exam names/acronyms listed (e.g., "EAS", "edTPA", "CST").
       - observation_hours: The observation hours requirement as text if present (e.g., "100 hours").
       - student_teaching_days: The supervised student teaching days as text if present (e.g., "40 days").
       - citations: Array of URLs cited for this stage; include only those associated with Initial stage.

    3) professional: Object for the Professional certificate stage:
       - name
       - validity_policy: Text describing validity/expiration/maintenance (e.g., if it doesn't expire but requires maintenance).
       - degree_requirement
       - content_core_grad_hours: Numeric hours mentioned for "content core" (e.g., "12"), as a string if possible.
       - experience_years: Numeric years of classroom teaching required (e.g., "3") as a string if possible.
       - mentorship_policy: Text describing first-year mentorship and any exceptions (e.g., unless 2 years prior experience).
       - ctle_hours_per_5_years: Numeric CTLE hours over 5 years (e.g., "100") as a string if possible.
       - citations: Array of URLs cited for this stage; include only those associated with Professional stage.

    4) sbl: Object for the School Building Leader certificate stage:
       - name
       - validity_policy: Text describing validity/maintenance policy for SBL, if provided.
       - degree_requirement
       - gpa_minimum: The numeric GPA minimum for SBL (e.g., "3.0") as a string if provided.
       - experience_requirement: Text describing minimum years and type of experience (e.g., 3 years teaching or pupil personnel services in PK-12).
       - program_credits: Numeric credits for program completion (e.g., "30") as a string if provided.
       - exams_required: Array of exam names/acronyms listed (e.g., "School Building Leader Assessment").
       - citations: Array of URLs cited for this stage; include only those associated with SBL stage.

    Rules:
    - If a field is not present in the answer, set it to null (for strings) or [] (for arrays).
    - For URLs, include full links and only those explicitly present in the answer text (or in markdown).
    - Do not invent GPA values or numeric counts. If not clearly stated, set null.
    - Prefer exact exam acronyms when present.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def urls_are_official(urls: List[str]) -> bool:
    if not urls:
        return False
    allowed = ("nysed.gov", "schools.nyc.gov")
    ok = True
    for u in urls:
        if not isinstance(u, str) or not u.strip():
            ok = False
            break
        if not any(dom in u.lower() for dom in allowed):
            ok = False
            break
    return ok


def text_contains_number(txt: Optional[str], target: str) -> bool:
    if not txt or not target:
        return False
    # Normalize digits by stripping non-digit-dot characters and search for the exact token
    # Also allow textual "five" for 5 in a lenient manner if target is "5" or "5.0"
    norm = txt.lower()
    if target == "5" or target == "5.0":
        if "5" in norm or "five" in norm:
            return True
    if target == "100":
        if "100" in norm or "one hundred" in norm:
            return True
    if target == "40":
        if "40" in norm or "forty" in norm:
            return True
    # generic fallback exact token search
    return target in norm


def list_contains_keyword(items: List[str], keywords: List[str]) -> bool:
    if not items or not keywords:
        return False
    low_items = [s.lower() for s in items if isinstance(s, str)]
    for kw in keywords:
        kw_l = kw.lower()
        if any(kw_l in it for it in low_items):
            return True
    return False


def exams_present_all(initial_exams: List[str], required: List[List[str]]) -> bool:
    """
    required: list of alternative keyword groups. Each group requires at least one match in items.
    Example for Initial: [["EAS", "Educating All Students"], ["edTPA"], ["CST", "Content Specialty Test"]]
    """
    if not initial_exams:
        return False
    low_items = [s.lower() for s in initial_exams]
    for group in required:
        found = any(any(alt.lower() in it for it in low_items) for alt in group)
        if not found:
            return False
    return True


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_context(evaluator: Evaluator, parent_node, extraction: PathwayExtraction) -> None:
    ctx_node = evaluator.add_parallel(
        id="Context_NYC_Requires_NYSED_Certification",
        desc="States that NYC DOE teachers must hold New York State teacher certification granted by NYSED.",
        parent=parent_node,
        critical=True,
    )

    ctx_citations = extraction.context_citations or []

    # Existence of citations
    ctx_cit_exists = evaluator.add_custom_node(
        result=bool(ctx_citations),
        id="context_citations_provided",
        desc="Context: Official citations are provided in the answer to support the NYC requires NYSED certification statement",
        parent=ctx_node,
        critical=True
    )

    # Domain is official
    evaluator.add_custom_node(
        result=urls_are_official(ctx_citations),
        id="context_citations_official",
        desc="Context: Citations are official NYSED or NYC DOE sources",
        parent=ctx_node,
        critical=True
    )

    # Verify the policy statement is supported by the cited sources
    ctx_claim_node = evaluator.add_leaf(
        id="context_claim_supported",
        desc="NYC DOE teachers must hold NYSED teacher certification is supported by cited sources",
        parent=ctx_node,
        critical=True
    )
    await evaluator.verify(
        claim="Teachers employed by the NYC Department of Education are required to hold New York State teacher certification (issued by NYSED).",
        node=ctx_claim_node,
        sources=ctx_citations,
        additional_instruction="Verify that the official NYC DOE or NYSED policy requires NYC public school teachers to hold NYSED certification.",
        extra_prerequisites=[ctx_cit_exists]
    )


async def verify_stage_initial(evaluator: Evaluator, parent_node, initial: Optional[StageInitialExtraction]) -> None:
    stage_node = evaluator.add_parallel(
        id="Stage_1_Initial_Certificate",
        desc="Initial teaching certificate stage: include required name/validity and all constraint-listed requirements, with official citations.",
        parent=parent_node,
        critical=True
    )

    # Guard existence of the stage object
    stage_exists = evaluator.add_custom_node(
        result=initial is not None,
        id="initial_stage_present",
        desc="Initial stage information is present in the answer",
        parent=stage_node,
        critical=True
    )

    initial = initial or StageInitialExtraction()

    # ---- Citations group (will be used as prerequisite for other verifications) ----
    cit_group = evaluator.add_parallel(
        id="Initial_Official_Citations",
        desc="Provides supporting reference URL(s) for the Initial Certificate stage from official NYSED (nysed.gov) or NYC DOE (schools.nyc.gov) sources.",
        parent=stage_node,
        critical=True
    )
    init_cit_exists = evaluator.add_custom_node(
        result=bool(initial.citations),
        id="initial_citations_provided",
        desc="Initial: At least one official citation URL is provided",
        parent=cit_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=urls_are_official(initial.citations),
        id="initial_citations_official",
        desc="Initial: All citations are official NYSED or NYC DOE sources",
        parent=cit_group,
        critical=True
    )

    # ---- Name & Validity ----
    name_validity = evaluator.add_parallel(
        id="Initial_Name_And_Validity_Period",
        desc="Provides the certificate name (Initial Certificate) and its validity period (5 years).",
        parent=stage_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(initial.name and initial.validity_period),
        id="initial_name_validity_present",
        desc="Initial: The answer provides both the certificate name and a validity period",
        parent=name_validity,
        critical=True
    )
    name_validity_supported = evaluator.add_leaf(
        id="initial_name_validity_supported",
        desc="Initial: The stated certificate name and validity period are supported by cited sources",
        parent=name_validity,
        critical=True
    )
    await evaluator.verify(
        claim=f"The certificate '{initial.name or 'Initial Certificate'}' has a validity period of {initial.validity_period or '[missing]'} for New York State.",
        node=name_validity_supported,
        sources=initial.citations,
        additional_instruction="Treat '5 years', 'five years', and 'valid for five years' as equivalent wordings.",
        extra_prerequisites=[stage_exists, init_cit_exists]
    )

    # ---- Education requirement (bachelor's + GPA minimum) ----
    edu_node = evaluator.add_parallel(
        id="Initial_Education_Requirement",
        desc="Includes the educational requirement: bachelor's degree with minimum 2.5 GPA.",
        parent=stage_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(initial.degree_requirement) and bool(initial.gpa_minimum),
        id="initial_education_present",
        desc="Initial: The answer provides a bachelor's degree requirement and a GPA minimum",
        parent=edu_node,
        critical=True
    )
    edu_supported = evaluator.add_leaf(
        id="initial_education_supported",
        desc="Initial: Bachelor's degree with the stated GPA minimum is supported by cited sources",
        parent=edu_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For the Initial teaching certificate, the educational requirement includes a bachelor's degree and a minimum GPA of {initial.gpa_minimum or '[missing]'}.",
        node=edu_supported,
        sources=initial.citations,
        additional_instruction="Allow reasonable paraphrases; confirm that a bachelor's degree and the stated GPA threshold are required for Initial certification.",
        extra_prerequisites=[stage_exists, init_cit_exists]
    )

    # ---- Exam requirements: EAS, edTPA, CST ----
    exams_node = evaluator.add_parallel(
        id="Initial_Exam_Requirements",
        desc="Includes examination requirements: passing NYSTCE exams (EAS, edTPA, CST).",
        parent=stage_node,
        critical=True
    )
    required_exam_groups = [["EAS", "Educating All Students"], ["edTPA"], ["CST", "Content Specialty Test"]]
    evaluator.add_custom_node(
        result=exams_present_all(initial.exam_requirements, required_exam_groups),
        id="initial_exams_present_in_answer",
        desc="Initial: The answer lists EAS, edTPA, and CST among exam requirements",
        parent=exams_node,
        critical=True
    )
    # EAS support
    initial_eas_supported = evaluator.add_leaf(
        id="initial_exam_eas_supported",
        desc="Initial: EAS requirement is supported by cited sources",
        parent=exams_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Initial teaching certificate requires passing the Educating All Students (EAS) exam.",
        node=initial_eas_supported,
        sources=initial.citations,
        additional_instruction="Confirm NYSTCE EAS is part of Initial certification requirements.",
        extra_prerequisites=[stage_exists, init_cit_exists]
    )
    # edTPA support
    initial_edtpa_supported = evaluator.add_leaf(
        id="initial_exam_edtpa_supported",
        desc="Initial: edTPA requirement is supported by cited sources",
        parent=exams_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Initial teaching certificate requires passing the edTPA teacher performance assessment.",
        node=initial_edtpa_supported,
        sources=initial.citations,
        additional_instruction="Confirm edTPA is required as part of Initial certification exam requirements (if the cited official source specifies this).",
        extra_prerequisites=[stage_exists, init_cit_exists]
    )
    # CST support
    initial_cst_supported = evaluator.add_leaf(
        id="initial_exam_cst_supported",
        desc="Initial: CST requirement is supported by cited sources",
        parent=exams_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Initial teaching certificate requires passing the appropriate Content Specialty Test (CST).",
        node=initial_cst_supported,
        sources=initial.citations,
        additional_instruction="Confirm that a Content Specialty Test corresponding to the certificate area is required.",
        extra_prerequisites=[stage_exists, init_cit_exists]
    )

    # ---- Student teaching requirements ----
    st_node = evaluator.add_parallel(
        id="Initial_Student_Teaching_Requirements",
        desc="Includes student teaching requirements: minimum 100 hours field observation AND at least 40 days supervised student teaching.",
        parent=stage_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=text_contains_number(initial.observation_hours, "100"),
        id="initial_observation_100_hours_present",
        desc="Initial: The answer includes at least 100 hours of field observation",
        parent=st_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=text_contains_number(initial.student_teaching_days, "40"),
        id="initial_supervised_40_days_present",
        desc="Initial: The answer includes at least 40 days of supervised student teaching",
        parent=st_node,
        critical=True
    )
    obs_supported = evaluator.add_leaf(
        id="initial_observation_supported",
        desc="Initial: 100 hours field observation is supported by cited sources",
        parent=st_node,
        critical=True
    )
    await evaluator.verify(
        claim="For the Initial teaching certificate, candidates must complete at least 100 hours of field observation.",
        node=obs_supported,
        sources=initial.citations,
        additional_instruction="Confirm the minimum 100 hours of field observation requirement.",
        extra_prerequisites=[stage_exists, init_cit_exists]
    )
    st_days_supported = evaluator.add_leaf(
        id="initial_student_teaching_days_supported",
        desc="Initial: 40 days supervised student teaching is supported by cited sources",
        parent=st_node,
        critical=True
    )
    await evaluator.verify(
        claim="For the Initial teaching certificate, candidates must complete at least 40 days of supervised student teaching.",
        node=st_days_supported,
        sources=initial.citations,
        additional_instruction="Confirm the minimum 40 days of supervised student teaching requirement.",
        extra_prerequisites=[stage_exists, init_cit_exists]
    )


async def verify_stage_professional(evaluator: Evaluator, parent_node, professional: Optional[StageProfessionalExtraction]) -> None:
    stage_node = evaluator.add_parallel(
        id="Stage_2_Professional_Certificate",
        desc="Professional Certificate stage: include required name/validity/maintenance policy and all constraint-listed requirements, with official citations.",
        parent=parent_node,
        critical=True
    )

    stage_exists = evaluator.add_custom_node(
        result=professional is not None,
        id="professional_stage_present",
        desc="Professional stage information is present in the answer",
        parent=stage_node,
        critical=True
    )

    professional = professional or StageProfessionalExtraction()

    # Citations group
    cit_group = evaluator.add_parallel(
        id="Professional_Official_Citations",
        desc="Provides supporting reference URL(s) for the Professional Certificate stage from official NYSED (nysed.gov) or NYC DOE (schools.nyc.gov) sources.",
        parent=stage_node,
        critical=True
    )
    prof_cit_exists = evaluator.add_custom_node(
        result=bool(professional.citations),
        id="professional_citations_provided",
        desc="Professional: At least one official citation URL is provided",
        parent=cit_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=urls_are_official(professional.citations),
        id="professional_citations_official",
        desc="Professional: All citations are official NYSED or NYC DOE sources",
        parent=cit_group,
        critical=True
    )

    # Name & validity/maintenance policy
    name_val = evaluator.add_parallel(
        id="Professional_Name_And_Validity_Policy",
        desc="Provides the certificate name (Professional Certificate) and states its validity/expiration/maintenance policy.",
        parent=stage_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(professional.name and professional.validity_policy),
        id="professional_name_validity_present",
        desc="Professional: The answer provides both the certificate name and validity/maintenance policy",
        parent=name_val,
        critical=True
    )
    val_supported = evaluator.add_leaf(
        id="professional_name_validity_supported",
        desc="Professional: The stated validity/maintenance policy is supported by cited sources",
        parent=name_val,
        critical=True
    )
    await evaluator.verify(
        claim=f"The certificate '{professional.name or 'Professional Certificate'}' has the following validity/maintenance policy: {professional.validity_policy or '[missing]'}",
        node=val_supported,
        sources=professional.citations,
        additional_instruction="Confirm the described Professional certificate validity or maintenance policy (e.g., no expiration but CTLE maintenance).",
        extra_prerequisites=[stage_exists, prof_cit_exists]
    )

    # Education requirement: master's + 12 grad hours in content core
    edu = evaluator.add_parallel(
        id="Professional_Education_Requirement",
        desc="Includes the educational requirement: master's degree including 12 graduate semester hours in the content core.",
        parent=stage_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(professional.degree_requirement) and text_contains_number(professional.content_core_grad_hours, "12"),
        id="professional_education_present",
        desc="Professional: The answer includes master's degree and specifies 12 graduate hours in the content core",
        parent=edu,
        critical=True
    )
    edu_supported = evaluator.add_leaf(
        id="professional_education_supported",
        desc="Professional: Master's + 12 content-core graduate hours are supported by cited sources",
        parent=edu,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Professional certificate requires a master's degree that includes at least {professional.content_core_grad_hours or '[missing]'} graduate semester hours in the content core.",
        node=edu_supported,
        sources=professional.citations,
        additional_instruction="Confirm the minimum graduate semester hours in the content core for Professional certification.",
        extra_prerequisites=[stage_exists, prof_cit_exists]
    )

    # Experience requirement: 3 years classroom teaching
    exp = evaluator.add_parallel(
        id="Professional_Experience_Requirement",
        desc="Includes the experience requirement: 3 years of classroom teaching experience.",
        parent=stage_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=text_contains_number(professional.experience_years, "3"),
        id="professional_experience_present",
        desc="Professional: The answer includes a requirement of 3 years of classroom teaching experience",
        parent=exp,
        critical=True
    )
    exp_supported = evaluator.add_leaf(
        id="professional_experience_supported",
        desc="Professional: 3 years classroom teaching experience requirement is supported by cited sources",
        parent=exp,
        critical=True
    )
    await evaluator.verify(
        claim="The Professional certificate requires at least 3 years of classroom teaching experience.",
        node=exp_supported,
        sources=professional.citations,
        additional_instruction="Confirm the minimum of three years of classroom teaching experience for Professional certification.",
        extra_prerequisites=[stage_exists, prof_cit_exists]
    )

    # Mentorship requirement: first-year mentorship unless 2 years prior experience
    mentor = evaluator.add_parallel(
        id="Professional_Mentorship_Requirement",
        desc="Includes the mentorship requirement: first-year mentorship unless 2 years prior experience.",
        parent=stage_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(professional.mentorship_policy),
        id="professional_mentorship_present",
        desc="Professional: The answer includes a mentorship requirement/policy description",
        parent=mentor,
        critical=True
    )
    mentor_supported = evaluator.add_leaf(
        id="professional_mentorship_supported",
        desc="Professional: Mentorship policy (first-year mentoring unless 2 years prior) is supported by cited sources",
        parent=mentor,
        critical=True
    )
    await evaluator.verify(
        claim="For Professional certification, teachers are subject to a first-year mentoring/induction requirement unless they have at least two years of prior relevant experience.",
        node=mentor_supported,
        sources=professional.citations,
        additional_instruction="Confirm that a first-year mentorship (or mentoring/induction) is required unless the educator has 2+ years of prior experience or an equivalent exemption.",
        extra_prerequisites=[stage_exists, prof_cit_exists]
    )

    # Maintenance CTLE: 100 hours/5 years
    ctle = evaluator.add_parallel(
        id="Professional_Maintenance_CTLE",
        desc="Includes the maintenance requirement: 100 CTLE hours every 5 years to maintain certification.",
        parent=stage_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=text_contains_number(professional.ctle_hours_per_5_years, "100"),
        id="professional_ctle_present",
        desc="Professional: The answer specifies 100 CTLE hours every 5 years",
        parent=ctle,
        critical=True
    )
    ctle_supported = evaluator.add_leaf(
        id="professional_ctle_supported",
        desc="Professional: 100 CTLE hours/5 years is supported by cited sources",
        parent=ctle,
        critical=True
    )
    await evaluator.verify(
        claim=f"To maintain the Professional certificate, educators must complete {professional.ctle_hours_per_5_years or '[missing]'} Continuing Teacher and Leader Education (CTLE) hours every five years.",
        node=ctle_supported,
        sources=professional.citations,
        additional_instruction="Confirm the CTLE requirement threshold and 5-year window for Professional certification maintenance.",
        extra_prerequisites=[stage_exists, prof_cit_exists]
    )


async def verify_stage_sbl(evaluator: Evaluator, parent_node, sbl: Optional[StageSBLExtraction]) -> None:
    stage_node = evaluator.add_parallel(
        id="Stage_3_School_Building_Leader_Certificate",
        desc="SBL stage: include required name/validity/maintenance policy and all constraint-listed requirements, with official citations.",
        parent=parent_node,
        critical=True
    )

    stage_exists = evaluator.add_custom_node(
        result=sbl is not None,
        id="sbl_stage_present",
        desc="SBL stage information is present in the answer",
        parent=stage_node,
        critical=True
    )

    sbl = sbl or StageSBLExtraction()

    # Citations group
    cit_group = evaluator.add_parallel(
        id="SBL_Official_Citations",
        desc="Provides supporting reference URL(s) for the SBL stage from official NYSED (nysed.gov) or NYC DOE (schools.nyc.gov) sources.",
        parent=stage_node,
        critical=True
    )
    sbl_cit_exists = evaluator.add_custom_node(
        result=bool(sbl.citations),
        id="sbl_citations_provided",
        desc="SBL: At least one official citation URL is provided",
        parent=cit_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=urls_are_official(sbl.citations),
        id="sbl_citations_official",
        desc="SBL: All citations are official NYSED or NYC DOE sources",
        parent=cit_group,
        critical=True
    )

    # Name & validity policy
    name_val = evaluator.add_parallel(
        id="SBL_Name_And_Validity_Policy",
        desc="Provides the certificate name (School Building Leader / SBL) and states its validity/expiration/maintenance policy.",
        parent=stage_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(sbl.name and sbl.validity_policy),
        id="sbl_name_validity_present",
        desc="SBL: The answer provides both the certificate name and validity/maintenance policy",
        parent=name_val,
        critical=True
    )
    name_val_supported = evaluator.add_leaf(
        id="sbl_name_validity_supported",
        desc="SBL: The stated validity/maintenance policy is supported by cited sources",
        parent=name_val,
        critical=True
    )
    await evaluator.verify(
        claim=f"The certificate '{sbl.name or 'School Building Leader (SBL)'}' has the following validity/maintenance policy: {sbl.validity_policy or '[missing]'}",
        node=name_val_supported,
        sources=sbl.citations,
        additional_instruction="Confirm the validity/maintenance policy for the SBL certificate.",
        extra_prerequisites=[stage_exists, sbl_cit_exists]
    )

    # Education requirement with GPA 3.0 minimum
    edu = evaluator.add_parallel(
        id="SBL_Education_Requirement",
        desc="Includes the educational requirement: master's degree with minimum 3.0 GPA.",
        parent=stage_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(sbl.degree_requirement) and text_contains_number(sbl.gpa_minimum, "3.0"),
        id="sbl_education_present",
        desc="SBL: The answer includes a master's degree and specifies a 3.0 minimum GPA",
        parent=edu,
        critical=True
    )
    edu_supported = evaluator.add_leaf(
        id="sbl_education_supported",
        desc="SBL: Master's degree with the stated GPA minimum is supported by cited sources",
        parent=edu,
        critical=True
    )
    await evaluator.verify(
        claim=f"The SBL certificate requires a master's degree with a minimum GPA of {sbl.gpa_minimum or '[missing]'}.",
        node=edu_supported,
        sources=sbl.citations,
        additional_instruction="Confirm the minimum GPA and master's degree requirement for SBL.",
        extra_prerequisites=[stage_exists, sbl_cit_exists]
    )

    # Experience requirement: minimum 3 years teaching or PPS in PK-12
    exp = evaluator.add_parallel(
        id="SBL_Experience_Requirement",
        desc="Includes the experience requirement: minimum 3 years teaching or pupil personnel services experience in an accredited PK-12 setting.",
        parent=stage_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(sbl.experience_requirement),
        id="sbl_experience_present",
        desc="SBL: The answer includes an experience requirement description",
        parent=exp,
        critical=True
    )
    exp_supported = evaluator.add_leaf(
        id="sbl_experience_supported",
        desc="SBL: 3 years teaching or PPS in accredited PK-12 is supported by cited sources",
        parent=exp,
        critical=True
    )
    await evaluator.verify(
        claim="For the SBL certificate, candidates must have a minimum of 3 years of teaching or pupil personnel services experience in an accredited P-12 school setting.",
        node=exp_supported,
        sources=sbl.citations,
        additional_instruction="Confirm the minimum three years of appropriate experience in an accredited P-12 context.",
        extra_prerequisites=[stage_exists, sbl_cit_exists]
    )

    # Program completion: 30 credits
    prog = evaluator.add_parallel(
        id="SBL_Program_Completion_Requirement",
        desc="Includes the program completion requirement: completion of a 30-credit program.",
        parent=stage_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=text_contains_number(sbl.program_credits, "30"),
        id="sbl_program_credits_present",
        desc="SBL: The answer specifies a 30-credit program",
        parent=prog,
        critical=True
    )
    prog_supported = evaluator.add_leaf(
        id="sbl_program_supported",
        desc="SBL: 30-credit program completion requirement is supported by cited sources",
        parent=prog,
        critical=True
    )
    await evaluator.verify(
        claim="For the SBL certificate, candidates must complete a program of at least 30 graduate credits.",
        node=prog_supported,
        sources=sbl.citations,
        additional_instruction="Confirm that the SBL program is at least 30 credits.",
        extra_prerequisites=[stage_exists, sbl_cit_exists]
    )

    # Examination: NYS SBL Assessment
    exam = evaluator.add_parallel(
        id="SBL_Exam_Requirement",
        desc="Includes the examination requirement: passing the NYS School Building Leader Assessment.",
        parent=stage_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=list_contains_keyword(sbl.exams_required, ["School Building Leader", "SBL"]),
        id="sbl_exam_present",
        desc="SBL: The answer lists the SBL assessment among the exam requirements",
        parent=exam,
        critical=True
    )
    exam_supported = evaluator.add_leaf(
        id="sbl_exam_supported",
        desc="SBL: Passing the NYS School Building Leader Assessment is supported by cited sources",
        parent=exam,
        critical=True
    )
    await evaluator.verify(
        claim="The SBL certificate requires passing the New York State School Building Leader (SBL) assessment.",
        node=exam_supported,
        sources=sbl.citations,
        additional_instruction="Confirm that the SBL assessment is required for certification.",
        extra_prerequisites=[stage_exists, sbl_cit_exists]
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # We'll create our own sequential main node under root
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

    # Create the main sequential node as specified by the rubric
    main_seq = evaluator.add_sequential(
        id="Complete_Certification_Pathway",
        desc="Identify the sequential certification pathway for becoming an Assistant Principal in NYC public schools from Initial teaching certificate through School Building Leader (SBL), and provide required elements per stage with official NYSED/NYC DOE citations.",
        parent=root,
        critical=True
    )

    # Extraction
    extraction: PathwayExtraction = await evaluator.extract(
        prompt=prompt_extract_pathway(),
        template_class=PathwayExtraction,
        extraction_name="pathway_extraction"
    )

    # Verification in the specified order (sequential)
    await verify_context(evaluator, main_seq, extraction)

    await verify_stage_initial(evaluator, main_seq, extraction.initial)

    await verify_stage_professional(evaluator, main_seq, extraction.professional)

    await verify_stage_sbl(evaluator, main_seq, extraction.sbl)

    # Return summary
    return evaluator.get_summary()