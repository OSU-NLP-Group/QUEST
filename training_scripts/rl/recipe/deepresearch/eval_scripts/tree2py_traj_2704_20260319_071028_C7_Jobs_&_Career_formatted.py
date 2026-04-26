import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "career_service_certifications_2025_usa"
TASK_DESCRIPTION = """
Identify 4 distinct career service professional certifications that are currently available in the United States (as of 2025-2026) and are specifically designed for individuals working in career development, career counseling, career coaching, resume writing, or workforce development roles. For each certification, provide the following information: (1) The full name of the certification and its acronym, (2) The name of the issuing organization, (3) The type of assessment required (e.g., exam format, evaluation type), (4) Any prerequisite requirements (training, education, or experience), (5) The application or enrollment process, (6) The professional credential designation conferred, (7) The application or examination fee, (8) Maintenance requirements (if any), (9) Time limits for completing the certification process (if any), (10) Passing criteria or evaluation standards for the assessment, (11) A reference URL from the official issuing organization. Ensure that all 4 certifications are from recognized professional associations or credentialing organizations in the career services field, and that each confers a distinct, industry-recognized credential.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CertificationItem(BaseModel):
    full_name: Optional[str] = None
    acronym: Optional[str] = None
    issuing_org: Optional[str] = None
    assessment: Optional[str] = None
    prerequisites: Optional[str] = None
    application_process: Optional[str] = None
    credential: Optional[str] = None  # Professional designation letters granted
    fee: Optional[str] = None  # Application or exam fee; allow text/range/notes
    maintenance: Optional[str] = None  # CE/renewal/membership requirements
    time_limits: Optional[str] = None  # Any timeframe constraints
    passing_criteria: Optional[str] = None  # Passing/evaluation standard
    reference_urls: List[str] = Field(default_factory=list)  # Official URLs only


class CertificationsExtraction(BaseModel):
    certifications: List[CertificationItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_certifications() -> str:
    return """
    Extract up to 6 career service professional certifications listed in the answer (we will evaluate only the first 4 unique ones).
    For each certification, extract the following fields exactly as stated in the answer text:
    - full_name: Full official name of the certification (e.g., Certified Career Counselor)
    - acronym: Acronym or short designation (e.g., CCC, CPRW) if provided
    - issuing_org: Name of the issuing/credentialing organization (professional association/board)
    - assessment: The type of assessment or evaluation (e.g., proctored exam, portfolio review, performance exam, written test)
    - prerequisites: Any prerequisite training, education, experience, or membership
    - application_process: Steps or process to apply/enroll (e.g., application, verification, scheduling, portfolio submission)
    - credential: The professional designation or post-nominal letters conferred (e.g., CCC, CPRW, NCRW)
    - fee: The application/exam fee; include currency and any ranges if present (text allowed)
    - maintenance: Maintenance/renewal/CE/membership requirements (if stated)
    - time_limits: Any time limits or completion windows (if stated)
    - passing_criteria: Passing score or evaluation standard (if stated)
    - reference_urls: Array of official URLs from the issuing organization related to the certification (home page, handbook, or official PDF). Only include URLs explicitly present in the answer; ignore non-official sources.
    Return a JSON object with one key "certifications": an array of objects with exactly the fields above. 
    Do not invent missing values. If a field is not present in the answer, set it to null (or empty array for reference_urls).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty(s: Optional[str]) -> bool:
    return s is not None and str(s).strip() != ""


def _normalize_name(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", s.lower().strip())


def _first_k_distinct_certifications(certs: List[CertificationItem], k: int = 4) -> List[CertificationItem]:
    seen = set()
    result: List[CertificationItem] = []
    for c in certs:
        if not _is_nonempty(c.full_name):
            continue
        key = _normalize_name(c.full_name or "")
        if key in seen:
            continue
        seen.add(key)
        result.append(c)
        if len(result) >= k:
            break
    return result


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def _add_count_node(evaluator: Evaluator, parent, items: List[CertificationItem]) -> None:
    # Exactly 4 distinct certifications are provided (considering the filtered 4)
    distinct_names = {_normalize_name(c.full_name or "") for c in items if _is_nonempty(c.full_name)}
    exactly_4 = (len(items) == 4) and (len(distinct_names) == 4)
    evaluator.add_custom_node(
        result=exactly_4,
        id="Certification_Count",
        desc="Exactly 4 distinct certifications are provided",
        parent=parent,
        critical=True
    )


def _add_category_nodes(evaluator: Evaluator, root):
    nodes = {}
    # Root categories (non-critical to allow partial credit as per rubric)
    nodes["Certification_Names"] = evaluator.add_parallel(
        id="Certification_Names",
        desc="The full name and acronym for each of the 4 certifications are provided",
        parent=root,
        critical=False
    )
    nodes["Geographic_Availability"] = evaluator.add_parallel(
        id="Geographic_Availability",
        desc="All 4 certifications are confirmed to be available in the United States and actively offered in 2025-2026",
        parent=root,
        critical=False
    )
    nodes["Target_Profession_Appropriateness"] = evaluator.add_parallel(
        id="Target_Profession_Appropriateness",
        desc="All 4 certifications are specifically designed for career services professionals (career counselors, coaches, resume writers, advisors, or workforce practitioners)",
        parent=root,
        critical=False
    )
    nodes["Issuing_Organizations"] = evaluator.add_parallel(
        id="Issuing_Organizations",
        desc="The issuing organization for each of the 4 certifications is identified and is a recognized professional association or credentialing body",
        parent=root,
        critical=False
    )
    nodes["Assessment_Requirements"] = evaluator.add_parallel(
        id="Assessment_Requirements",
        desc="The assessment component (exam, evaluation, or similar) for each of the 4 certifications is documented",
        parent=root,
        critical=False
    )
    nodes["Application_Processes"] = evaluator.add_parallel(
        id="Application_Processes",
        desc="The application or enrollment process for each of the 4 certifications is described",
        parent=root,
        critical=False
    )
    nodes["Professional_Recognition"] = evaluator.add_parallel(
        id="Professional_Recognition",
        desc="Each of the 4 certifications confers a recognized professional credential designation in the career services field",
        parent=root,
        critical=False
    )
    nodes["Prerequisite_Specifications"] = evaluator.add_parallel(
        id="Prerequisite_Specifications",
        desc="Prerequisites (training, education, or experience) for each of the 4 certifications are clearly specified",
        parent=root,
        critical=False
    )
    nodes["Cost_Information"] = evaluator.add_parallel(
        id="Cost_Information",
        desc="Application or examination fees for each of the 4 certifications are provided",
        parent=root,
        critical=False
    )
    nodes["Maintenance_Requirements"] = evaluator.add_parallel(
        id="Maintenance_Requirements",
        desc="Maintenance requirements (membership, continuing education, renewal) for each of the 4 certifications are documented",
        parent=root,
        critical=False
    )
    nodes["Time_Frames"] = evaluator.add_parallel(
        id="Time_Frames",
        desc="Any time limits for completing the certification process for each of the 4 certifications are stated",
        parent=root,
        critical=False
    )
    nodes["Evaluation_Criteria"] = evaluator.add_parallel(
        id="Evaluation_Criteria",
        desc="For certifications with exams or assessments, the passing criteria or evaluation standards for each of the 4 certifications are documented",
        parent=root,
        critical=False
    )
    nodes["Reference_URLs"] = evaluator.add_parallel(
        id="Reference_URLs",
        desc="Valid reference URLs from official sources are provided for each of the 4 certifications",
        parent=root,
        critical=False
    )
    return nodes


async def _add_per_cert_verifications(
    evaluator: Evaluator,
    nodes: Dict[str, Any],
    cert: CertificationItem,
    idx: int
) -> None:
    # Convenience
    name = cert.full_name or ""
    acronym = cert.acronym or ""
    org = cert.issuing_org or ""
    urls = cert.reference_urls or []

    # 1) Names provided (presence checks)
    names_provided_leaf = evaluator.add_custom_node(
        result=_is_nonempty(cert.full_name) and _is_nonempty(cert.acronym),
        id=f"cert_{idx}_name_acronym_provided",
        desc=f"Certification #{idx + 1}: Full name and acronym are provided",
        parent=nodes["Certification_Names"],
        critical=False
    )

    # 2) Reference URLs presence and officiality
    urls_provided_leaf = evaluator.add_custom_node(
        result=bool(urls),
        id=f"cert_{idx}_reference_url_provided",
        desc=f"Certification #{idx + 1}: At least one reference URL is provided",
        parent=nodes["Reference_URLs"],
        critical=False
    )
    url_official_leaf = evaluator.add_leaf(
        id=f"cert_{idx}_reference_url_official",
        desc=f"Certification #{idx + 1}: Reference URL(s) are official pages by the issuing organization",
        parent=nodes["Reference_URLs"],
        critical=False
    )
    claim_official = (
        f"At least one of these URLs is an official page by the issuing organization '{org}' that "
        f"describes the '{name}' certification (acronym '{acronym}')."
    )
    await evaluator.verify(
        claim=claim_official,
        node=url_official_leaf,
        sources=urls,
        additional_instruction="Prefer URLs on the issuing organization's official domain(s). The page should clearly describe the certification.",
    )

    # 3) Geographic availability and currency (US, 2025-2026)
    geo_leaf = evaluator.add_leaf(
        id=f"cert_{idx}_geo_us_active",
        desc=f"Certification #{idx + 1}: Confirmed available in the U.S. and actively offered in 2025-2026",
        parent=nodes["Geographic_Availability"],
        critical=False
    )
    geo_claim = (
        f"The '{name}' certification is available in the United States and is currently offered/active for 2025-2026."
    )
    await evaluator.verify(
        claim=geo_claim,
        node=geo_leaf,
        sources=urls,
        additional_instruction="Check that the certification is not retired/sunset and is open to US candidates; active pages, current handbooks, or U.S.-specific notes support this."
    )

    # 4) Target profession appropriateness
    target_leaf = evaluator.add_leaf(
        id=f"cert_{idx}_target_profession",
        desc=f"Certification #{idx + 1}: Designed for career services professionals",
        parent=nodes["Target_Profession_Appropriateness"],
        critical=False
    )
    target_claim = (
        f"The '{name}' certification is specifically designed for career services professionals, such as career counselors, "
        f"career coaches, resume writers, career advisors, or workforce development practitioners."
    )
    await evaluator.verify(
        claim=target_claim,
        node=target_leaf,
        sources=urls,
        additional_instruction="Look for language indicating the intended audience is career development/counseling/coaching/resume writing/workforce services professionals."
    )

    # 5) Issuing organization
    org_provided_leaf = evaluator.add_custom_node(
        result=_is_nonempty(org),
        id=f"cert_{idx}_issuing_org_provided",
        desc=f"Certification #{idx + 1}: Issuing organization is provided",
        parent=nodes["Issuing_Organizations"],
        critical=False
    )
    org_match_leaf = evaluator.add_leaf(
        id=f"cert_{idx}_issuing_org_matches",
        desc=f"Certification #{idx + 1}: Issuing organization matches official source",
        parent=nodes["Issuing_Organizations"],
        critical=False
    )
    org_claim = f"The issuing organization for the '{name}' certification is '{org}'."
    await evaluator.verify(
        claim=org_claim,
        node=org_match_leaf,
        sources=urls,
        additional_instruction="Verify the organization named is the official credentialing/issuing body on the referenced page(s)."
    )

    org_recognized_leaf = evaluator.add_leaf(
        id=f"cert_{idx}_issuing_org_recognized",
        desc=f"Certification #{idx + 1}: Issuing organization is a recognized professional association or credentialing body",
        parent=nodes["Issuing_Organizations"],
        critical=False
    )
    org_recog_claim = f"'{org}' is a recognized professional association or credentialing organization in the career services field."
    await evaluator.verify(
        claim=org_recog_claim,
        node=org_recognized_leaf,
        sources=urls,
        additional_instruction="Use the official page(s) to judge whether the body is a legitimate, established professional association/board for career services."
    )

    # 6) Assessment requirements
    assess_provided_leaf = evaluator.add_custom_node(
        result=_is_nonempty(cert.assessment),
        id=f"cert_{idx}_assessment_provided",
        desc=f"Certification #{idx + 1}: Assessment component is provided",
        parent=nodes["Assessment_Requirements"],
        critical=False
    )
    assess_leaf = evaluator.add_leaf(
        id=f"cert_{idx}_assessment_supported",
        desc=f"Certification #{idx + 1}: Assessment component matches official source",
        parent=nodes["Assessment_Requirements"],
        critical=False
    )
    assess_claim = f"The assessment/evaluation for '{name}' uses the following format(s): {cert.assessment or ''}"
    await evaluator.verify(
        claim=assess_claim,
        node=assess_leaf,
        sources=urls,
        additional_instruction="Check exam type (e.g., proctored/written), portfolio/performance review, interview, or other evaluation described officially."
    )

    # 7) Application / enrollment process
    app_provided_leaf = evaluator.add_custom_node(
        result=_is_nonempty(cert.application_process),
        id=f"cert_{idx}_application_provided",
        desc=f"Certification #{idx + 1}: Application/enrollment process is provided",
        parent=nodes["Application_Processes"],
        critical=False
    )
    app_leaf = evaluator.add_leaf(
        id=f"cert_{idx}_application_supported",
        desc=f"Certification #{idx + 1}: Application/enrollment process matches official source",
        parent=nodes["Application_Processes"],
        critical=False
    )
    app_claim = f"The application/enrollment process for '{name}' is: {cert.application_process or ''}"
    await evaluator.verify(
        claim=app_claim,
        node=app_leaf,
        sources=urls,
        additional_instruction="Look for application steps, required documentation, scheduling/registration procedures, or submission portals described officially."
    )

    # 8) Professional recognition / credential designation
    cred_provided_leaf = evaluator.add_custom_node(
        result=_is_nonempty(cert.credential),
        id=f"cert_{idx}_credential_provided",
        desc=f"Certification #{idx + 1}: Credential designation is provided",
        parent=nodes["Professional_Recognition"],
        critical=False
    )
    cred_leaf = evaluator.add_leaf(
        id=f"cert_{idx}_credential_supported",
        desc=f"Certification #{idx + 1}: Credential designation matches official source",
        parent=nodes["Professional_Recognition"],
        critical=False
    )
    cred_claim = (
        f"Holders of the '{name}' certification are granted the professional designation '{cert.credential or ''}'"
        + (f" (acronym '{acronym}')." if _is_nonempty(acronym) else ".")
    )
    await evaluator.verify(
        claim=cred_claim,
        node=cred_leaf,
        sources=urls,
        additional_instruction="Check the official page/handbook for the granted post-nominal letters or credential title."
    )

    # 9) Prerequisites
    preq_provided_leaf = evaluator.add_custom_node(
        result=_is_nonempty(cert.prerequisites),
        id=f"cert_{idx}_prereq_provided",
        desc=f"Certification #{idx + 1}: Prerequisites are provided",
        parent=nodes["Prerequisite_Specifications"],
        critical=False
    )
    preq_leaf = evaluator.add_leaf(
        id=f"cert_{idx}_prereq_supported",
        desc=f"Certification #{idx + 1}: Prerequisites match official source",
        parent=nodes["Prerequisite_Specifications"],
        critical=False
    )
    preq_claim = f"The prerequisites for '{name}' include: {cert.prerequisites or ''}"
    await evaluator.verify(
        claim=preq_claim,
        node=preq_leaf,
        sources=urls,
        additional_instruction="Confirm experience, education, training, or membership prerequisites as stated on the official page."
    )

    # 10) Fees
    fee_provided_leaf = evaluator.add_custom_node(
        result=_is_nonempty(cert.fee),
        id=f"cert_{idx}_fee_provided",
        desc=f"Certification #{idx + 1}: Application/exam fee is provided",
        parent=nodes["Cost_Information"],
        critical=False
    )
    fee_leaf = evaluator.add_leaf(
        id=f"cert_{idx}_fee_supported",
        desc=f"Certification #{idx + 1}: Fee matches official source",
        parent=nodes["Cost_Information"],
        critical=False
    )
    fee_claim = f"The application/exam fee for '{name}' is: {cert.fee or ''}"
    await evaluator.verify(
        claim=fee_claim,
        node=fee_leaf,
        sources=urls,
        additional_instruction="Allow ranges/member vs non-member pricing; verify fee amount/type aligns with official information."
    )

    # 11) Maintenance requirements
    maint_provided_leaf = evaluator.add_custom_node(
        result=_is_nonempty(cert.maintenance),
        id=f"cert_{idx}_maintenance_provided",
        desc=f"Certification #{idx + 1}: Maintenance/renewal requirements are provided",
        parent=nodes["Maintenance_Requirements"],
        critical=False
    )
    maint_leaf = evaluator.add_leaf(
        id=f"cert_{idx}_maintenance_supported",
        desc=f"Certification #{idx + 1}: Maintenance/renewal matches official source",
        parent=nodes["Maintenance_Requirements"],
        critical=False
    )
    maint_claim = f"The maintenance/renewal requirements for '{name}' are: {cert.maintenance or ''}"
    await evaluator.verify(
        claim=maint_claim,
        node=maint_leaf,
        sources=urls,
        additional_instruction="Check renewal cycles, CE/CEU hours, membership requirements, fees, and documentation on official sources."
    )

    # 12) Time limits
    time_provided_leaf = evaluator.add_custom_node(
        result=_is_nonempty(cert.time_limits),
        id=f"cert_{idx}_timeframe_provided",
        desc=f"Certification #{idx + 1}: Time limits for completion are provided",
        parent=nodes["Time_Frames"],
        critical=False
    )
    time_leaf = evaluator.add_leaf(
        id=f"cert_{idx}_timeframe_supported",
        desc=f"Certification #{idx + 1}: Time limits match official source",
        parent=nodes["Time_Frames"],
        critical=False
    )
    time_claim = f"The time limits/completion window for '{name}' are: {cert.time_limits or ''}"
    await evaluator.verify(
        claim=time_claim,
        node=time_leaf,
        sources=urls,
        additional_instruction="Look for eligibility windows, deadline windows (e.g., 1 year to complete after application) stated on the official page."
    )

    # 13) Passing criteria / evaluation standards
    pass_provided_leaf = evaluator.add_custom_node(
        result=_is_nonempty(cert.passing_criteria),
        id=f"cert_{idx}_passing_provided",
        desc=f"Certification #{idx + 1}: Passing criteria/evaluation standards are provided",
        parent=nodes["Evaluation_Criteria"],
        critical=False
    )
    pass_leaf = evaluator.add_leaf(
        id=f"cert_{idx}_passing_supported",
        desc=f"Certification #{idx + 1}: Passing criteria/evaluation standards match official source",
        parent=nodes["Evaluation_Criteria"],
        critical=False
    )
    pass_claim = f"The passing criteria/evaluation standard for '{name}' is: {cert.passing_criteria or ''}"
    await evaluator.verify(
        claim=pass_claim,
        node=pass_leaf,
        sources=urls,
        additional_instruction="Accept explicit scores (e.g., 70%), rubric/pass-fail standards, portfolio scoring thresholds, or equivalent evaluation language."
    )

    # Note: For all verify() calls above, URLs are provided; presence nodes created earlier help diagnose missing info.
    # We rely on the general non-critical aggregation to allow partial credit across items/categories.


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
) -> Dict:
    # Initialize evaluator (root is non-critical parallel to allow partial scoring)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_certifications(),
        template_class=CertificationsExtraction,
        extraction_name="certifications_extraction"
    )

    # Use only the first 4 distinct certifications by name
    selected = _first_k_distinct_certifications(extracted.certifications if extracted and extracted.certifications else [], 4)

    evaluator.add_custom_info(
        info={
            "extracted_total": len(extracted.certifications if extracted and extracted.certifications else []),
            "selected_count": len(selected),
            "selected_names": [c.full_name for c in selected]
        },
        info_type="selection_info",
        info_name="selection_info"
    )

    # Build category nodes
    category_nodes = _add_category_nodes(evaluator, root)

    # Global count node (critical)
    await _add_count_node(evaluator, root, selected)

    # Per-certification verification leaves
    for i in range(4):
        cert = selected[i] if i < len(selected) else CertificationItem()
        await _add_per_cert_verifications(evaluator, category_nodes, cert, i)

    # Return summarized evaluation
    return evaluator.get_summary()