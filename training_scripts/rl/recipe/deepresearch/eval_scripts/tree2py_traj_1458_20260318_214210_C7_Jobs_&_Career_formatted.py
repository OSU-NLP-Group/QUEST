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
TASK_ID = "oh_it_cert_compliance"
TASK_DESCRIPTION = """
A high school student in Ohio is planning their career pathway into Information Technology and wants to earn a professional IT certification that will help them meet the state's graduation requirements while also preparing them for entry-level IT positions. They need to identify an appropriate IT certification that satisfies ALL of the following requirements:

1. The certification must be an Ohio Department of Education-approved industry-recognized credential in the Information Technology career field that contributes toward the 12-point graduation requirement
2. The certification must have no formal prerequisites, prior certifications, or licenses required
3. The typical total preparation time should not exceed 300 hours
4. All required exams must be computer-based
5. The passing score for each exam must not exceed 85% of the maximum possible score
6. Each exam must not exceed 120 minutes in duration
7. The certification must be valid for at least 3 years without requiring renewal
8. All exams must be available in English
9. Earning the certification must not require mandatory membership fees to any professional organization
10. The certification must be issued by a recognized industry organization or vendor
11. The certification must have a clearly defined exam structure with published passing criteria
12. All required exams must be completable within a 12-month period

Identify one IT certification that meets all of these requirements. For your answer, provide:
- The official name of the certification
- The organization/vendor that issues it
- The current exam code(s) required
- Verification details for each of the 12 requirements listed above, including specific numbers, durations, scores, and reference sources
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RequirementEvidence(BaseModel):
    """Evidence per requirement, extracted from the answer."""
    claim_text: Optional[str] = None  # The answer's own phrasing of the claim for this requirement
    numbers: Optional[str] = None     # Any numeric details (e.g., 70%, 90 minutes, 3 years, ~120 hours)
    urls: List[str] = Field(default_factory=list)  # Supporting URLs explicitly listed in the answer


class CertificationExtraction(BaseModel):
    """Structured extraction of the certification and all requirement evidences."""
    certification_name: Optional[str] = None
    issuer: Optional[str] = None
    exam_codes: List[str] = Field(default_factory=list)

    # Per-requirement evidence objects
    ohio_approved: Optional[RequirementEvidence] = None
    no_prereqs: Optional[RequirementEvidence] = None
    study_time: Optional[RequirementEvidence] = None
    cbt: Optional[RequirementEvidence] = None
    passing_score: Optional[RequirementEvidence] = None
    duration: Optional[RequirementEvidence] = None
    validity: Optional[RequirementEvidence] = None
    english: Optional[RequirementEvidence] = None
    membership_fees: Optional[RequirementEvidence] = None
    recognized_org: Optional[RequirementEvidence] = None
    exam_structure: Optional[RequirementEvidence] = None
    completable_12mo: Optional[RequirementEvidence] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_certification() -> str:
    return """
    Extract the single IT certification proposed in the answer and the detailed evidence provided for each of the 12 listed requirements. Extract ONLY what is explicitly present in the answer; do not infer or invent information. For each requirement, also extract all source URLs explicitly cited in the answer that support that requirement.

    Output JSON fields:
    - certification_name: string | null
    - issuer: string | null
    - exam_codes: string[] (list of exam codes; if none are provided, return an empty list)

    For each requirement below, extract an object with:
      - claim_text: string | null   (the answer’s own statement for this requirement, if present)
      - numbers: string | null      (key numeric details like scores, durations, hours, years mentioned)
      - urls: string[]              (all explicit URLs cited in the answer that support this requirement)

    Requirement objects (all optional; return null if not present in the answer):
    - ohio_approved
    - no_prereqs
    - study_time
    - cbt
    - passing_score
    - duration
    - validity
    - english
    - membership_fees
    - recognized_org
    - exam_structure
    - completable_12mo

    Notes:
    - The 'urls' lists must only contain URLs that are explicitly present in the answer text (including markdown links).
    - If multiple requirements share the same URL(s), repeat them under each relevant requirement object.
    - Prefer strings for numbers (e.g., "70%", "90 minutes", "~120 hours") rather than numeric types.

    Return a single JSON object strictly following the defined schema.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(vals: Optional[List[str]]) -> List[str]:
    if not vals:
        return []
    return [v for v in vals if isinstance(v, str) and v.strip()]


def _codes_str(codes: List[str]) -> str:
    return ", ".join(codes) if codes else "N/A"


def _default_claim(cert_name: Optional[str], issuer: Optional[str], key: str) -> str:
    cn = cert_name or "the certification"
    if key == "ohio_approved":
        return f"The certification '{cn}' is listed by the Ohio education authority as an approved industry-recognized credential in the Information Technology career field that contributes toward the 12-point graduation requirement."
    if key == "no_prereqs":
        return f"The certification '{cn}' has no formal prerequisites, prior certifications, or licenses required to take its required exam(s)."
    if key == "study_time":
        return f"The typical total preparation time to earn '{cn}' does not exceed 300 hours."
    if key == "cbt":
        return f"All required exams for '{cn}' are delivered via computer-based testing."
    if key == "passing_score":
        return f"For each required exam of '{cn}', the passing score does not exceed 85% of the maximum possible score."
    if key == "duration":
        return f"Each required exam for '{cn}' is no longer than 120 minutes."
    if key == "validity":
        return f"The certification '{cn}' is valid for at least 3 years without requiring renewal within that period."
    if key == "english":
        return f"All required exams for '{cn}' are available in English."
    if key == "membership_fees":
        return f"Earning '{cn}' does not require paying mandatory membership fees to any professional organization."
    if key == "recognized_org":
        issuer_str = issuer or "a recognized industry organization or vendor"
        return f"The certification '{cn}' is issued by {issuer_str}."
    if key == "exam_structure":
        return f"The certification '{cn}' has a clearly defined exam structure with published passing criteria."
    if key == "completable_12mo":
        return f"All required exams for '{cn}' can be scheduled and completed within 12 months."
    return f"The requirement '{key}' is satisfied for '{cn}'."


def _additional_instruction_for(key: str, cert_name: Optional[str], issuer: Optional[str], exam_codes: List[str], numbers: Optional[str]) -> str:
    base = []
    cn = cert_name or "the certification"
    ic = issuer or "the issuing organization/vendor"
    codes = _codes_str(exam_codes)
    nums = numbers or "N/A"

    if key == "ohio_approved":
        base.append(
            f"Confirm that official Ohio education sources (Department of Education or Department of Education and Workforce) list '{cn}' in the Information Technology career field and that it contributes points toward the Industry-Recognized Credential graduation requirement (12-point system)."
        )
        base.append("Reject sources that are unrelated to Ohio's official credential lists.")
    elif key == "no_prereqs":
        base.append(f"Look for an explicit prerequisites section for '{cn}' and verify it states no formal prerequisites are required to take the exam(s).")
        base.append("If any prerequisite (e.g., prior certification, mandatory training, licensure) is required, treat this as NOT supported.")
    elif key == "study_time":
        base.append(f"Verify typical total preparation time for all required '{cn}' exam(s) does not exceed 300 hours. Use vendor guidance or official training hour guidance where available.")
        base.append(f"Reported numbers in the answer: {nums}. If sources do not clearly indicate ≤300 hours, treat as NOT supported.")
    elif key == "cbt":
        base.append(f"Confirm that all required exams for '{cn}' are administered as computer-based tests (CBT).")
    elif key == "passing_score":
        base.append(f"Verify that passing score for every required exam (codes: {codes}) is ≤ 85% of the maximum score.")
        base.append("If the source provides a scaled score (e.g., 700/1000), interpret it as 70% and check against the threshold.")
        base.append(f"Reported numbers in the answer: {nums}. If any exam exceeds 85%, treat as NOT supported.")
    elif key == "duration":
        base.append(f"Verify that the duration of each required exam (codes: {codes}) is ≤ 120 minutes.")
        base.append(f"Reported numbers in the answer: {nums}. If any exam exceeds 120 minutes, treat as NOT supported.")
    elif key == "validity":
        base.append(f"Confirm '{cn}' remains valid for at least 3 years without requiring renewal within 3 years.")
        base.append("If the credential is 'no expiration' or 'valid for 3+ years', this qualifies.")
    elif key == "english":
        base.append(f"Confirm that all required exams for '{cn}' are available in English.")
    elif key == "membership_fees":
        base.append(f"Verify that earning '{cn}' does not require paying mandatory membership fees to any professional association as a prerequisite.")
        base.append("If membership is optional or not mentioned as mandatory, treat as supported; if mandatory, NOT supported.")
    elif key == "recognized_org":
        base.append(f"Verify that '{cn}' is issued by a recognized industry organization/vendor: {ic}.")
        base.append("Use official issuer pages or reputable documentation indicating issuer status.")
    elif key == "exam_structure":
        base.append(f"Confirm that '{cn}' has a defined exam structure (e.g., number of questions, format) and published passing criteria (score threshold or scaled score).")
        base.append(f"Use official exam guide or blueprint references when possible. Numbers mentioned in the answer: {nums}.")
    elif key == "completable_12mo":
        base.append(f"Verify that all required exam(s) for '{cn}' can be scheduled and completed within a 12-month period.")
        base.append("Use scheduling frequency, availability, or program policy pages; if nothing contradicts completing within 12 months, and normal scheduling permits, consider it supported.")
    else:
        base.append("Check the claim exactly as written against the provided sources.")

    # Provide general context and reminders to the verifier:
    base.append(f"Certification: {cn}. Issuer: {ic}. Exam codes (if any): {codes}.")
    base.append("Focus ONLY on the provided webpages. Do not rely on external knowledge.")
    return " ".join(base)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_requirement(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    key: str,
    extracted: CertificationExtraction,
    evidence: Optional[RequirementEvidence],
) -> None:
    """
    Build a sequential sub-tree for one requirement:
      1) Critical sources-present check (custom node)
      2) Critical evidence-supported verification leaf
    """
    cert_name = extracted.certification_name
    issuer = extracted.issuer
    exam_codes = extracted.exam_codes or []
    urls = _safe_list(evidence.urls if evidence else None)

    # Create the requirement parent node (critical)
    req_node = evaluator.add_sequential(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=True
    )

    # 1) Sources-present gate (critical)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"{node_id}_sources_present",
        desc=f"Sources are explicitly provided in the answer for: {node_desc}",
        parent=req_node,
        critical=True
    )

    # 2) Evidence-supported verification (critical)
    leaf_node = evaluator.add_leaf(
        id=f"{node_id}_supported_by_sources",
        desc=node_desc,
        parent=req_node,
        critical=True
    )

    # Build claim and additional instruction
    claim_text = (evidence.claim_text if (evidence and evidence.claim_text) else _default_claim(cert_name, issuer, key))
    add_ins = _additional_instruction_for(key, cert_name, issuer, exam_codes, (evidence.numbers if evidence else None))

    await evaluator.verify(
        claim=claim_text,
        node=leaf_node,
        sources=urls,
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    """
    Evaluate an answer against the 12 Ohio IT certification requirements.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # root wrapper node (always non-critical by framework)
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

    # Extract structured certification + evidence info from the answer
    extracted: CertificationExtraction = await evaluator.extract(
        prompt=prompt_extract_certification(),
        template_class=CertificationExtraction,
        extraction_name="certification_extraction",
    )

    # Record some custom info for visibility
    evaluator.add_custom_info(
        {
            "certification_name": extracted.certification_name,
            "issuer": extracted.issuer,
            "exam_codes": extracted.exam_codes,
        },
        info_type="extracted_overview",
        info_name="extracted_overview"
    )

    # Build the compliance node as the true root for rubric aggregation (critical parallel)
    compliance_node = evaluator.add_parallel(
        id="IT_Certification_Requirements_Compliance",
        desc="Evaluates whether the identified IT certification meets all specified requirements for an Ohio high school student pursuing the Industry-Recognized Credential Seal while preparing for an IT career",
        parent=root,
        critical=True
    )

    # 1) Ohio Approved Credential
    await verify_requirement(
        evaluator,
        compliance_node,
        "Ohio_Approved_Credential",
        "The certification is listed as an Ohio Department of Education-approved industry-recognized credential in the Information Technology career field that contributes toward the 12-point graduation requirement",
        "ohio_approved",
        extracted,
        extracted.ohio_approved
    )

    # 2) No Formal Prerequisites
    await verify_requirement(
        evaluator,
        compliance_node,
        "No_Formal_Prerequisites",
        "The certification has no formal prerequisites, prior certifications, or licenses required as a condition for taking the exam(s)",
        "no_prereqs",
        extracted,
        extracted.no_prereqs
    )

    # 3) Study Time Within 300 Hours
    await verify_requirement(
        evaluator,
        compliance_node,
        "Study_Time_Within_300_Hours",
        "The typical total preparation time for all required exams does not exceed 300 hours",
        "study_time",
        extracted,
        extracted.study_time
    )

    # 4) Computer-Based Testing Format
    await verify_requirement(
        evaluator,
        compliance_node,
        "Computer_Based_Testing_Format",
        "All required exams for the certification are administered via computer-based testing format",
        "cbt",
        extracted,
        extracted.cbt
    )

    # 5) Passing Score ≤ 85%
    await verify_requirement(
        evaluator,
        compliance_node,
        "Passing_Score_Not_Exceeding_85_Percent",
        "The passing score for each required exam does not exceed 85% of the maximum possible score",
        "passing_score",
        extracted,
        extracted.passing_score
    )

    # 6) Exam Duration ≤ 120 Minutes
    await verify_requirement(
        evaluator,
        compliance_node,
        "Exam_Duration_Not_Exceeding_120_Minutes",
        "Each individual required exam has a duration that does not exceed 120 minutes",
        "duration",
        extracted,
        extracted.duration
    )

    # 7) Minimum 3-Year Validity
    await verify_requirement(
        evaluator,
        compliance_node,
        "Minimum_Three_Year_Validity",
        "The certification remains valid for at least 3 years from the date of award without requiring renewal",
        "validity",
        extracted,
        extracted.validity
    )

    # 8) English Language Availability
    await verify_requirement(
        evaluator,
        compliance_node,
        "English_Language_Availability",
        "All required exams are available to be taken in English",
        "english",
        extracted,
        extracted.english
    )

    # 9) No Mandatory Membership Fees
    await verify_requirement(
        evaluator,
        compliance_node,
        "No_Mandatory_Membership_Fees",
        "Earning the certification does not require payment of mandatory membership fees to any professional organization as a prerequisite",
        "membership_fees",
        extracted,
        extracted.membership_fees
    )

    # 10) Industry-Recognized Credential (Issuer)
    await verify_requirement(
        evaluator,
        compliance_node,
        "Industry_Recognized_Credential",
        "The certification is issued by a recognized industry organization, vendor, or professional body",
        "recognized_org",
        extracted,
        extracted.recognized_org
    )

    # 11) Clearly Defined Exam Structure
    await verify_requirement(
        evaluator,
        compliance_node,
        "Clearly_Defined_Exam_Structure",
        "The certification has a clearly defined exam format with specific published passing criteria",
        "exam_structure",
        extracted,
        extracted.exam_structure
    )

    # 12) Completable Within One Year
    await verify_requirement(
        evaluator,
        compliance_node,
        "Completable_Within_One_Year",
        "All required examinations can be scheduled and completed within a 12-month period",
        "completable_12mo",
        extracted,
        extracted.completable_12mo
    )

    # Return the evaluation summary
    return evaluator.get_summary()