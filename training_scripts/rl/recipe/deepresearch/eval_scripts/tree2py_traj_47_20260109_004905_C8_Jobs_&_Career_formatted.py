import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "certs_domains_comparison"
TASK_DESCRIPTION = (
    "You are advising a career development center that is creating a comprehensive guide comparing professional "
    "certifications across different business domains. Identify four prominent professional certifications, one from "
    "each of the following career domains: Human Resources, Project Management, Financial Planning/Analysis, and "
    "Information Security.\n\n"
    "For the Human Resources certification, provide:\n"
    "- The full official certification name\n"
    "- The minimum education requirement for eligibility\n"
    "- The minimum work experience requirement (in years or months)\n"
    "- The total number of exam questions\n"
    "- The exam duration (in hours and minutes)\n"
    "- The continuing education credits required for renewal and the renewal period\n\n"
    "For the Project Management certification, provide:\n"
    "- The full official certification name\n"
    "- The education requirement options\n"
    "- The project management experience requirement in months for each education level option\n"
    "- The required pre-exam training or contact hours\n"
    "- Whether the certification includes a random audit process for verifying experience\n\n"
    "For the Financial certification, provide:\n"
    "- The full official certification name\n"
    "- The number of exam levels required to complete the certification\n"
    "- The total number of exam questions for the first exam level\n"
    "- The minimum work experience required (in both hours and months) for the full designation\n"
    "- The recommended or required annual professional learning or continuing education credits\n\n"
    "For the Information Security certification, provide:\n"
    "- The full official certification name\n"
    "- The minimum years of cumulative work experience required\n"
    "- The minimum number of knowledge domains in which experience must be demonstrated\n"
    "- The total number of knowledge domains covered by the certification\n"
    "- The number of CPE credits required for renewal and the renewal cycle period (in years)\n"
    "- The annual maintenance fee in US dollars\n\n"
    "For each certification, include a supporting reference URL from an official or authoritative source."
)

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class HRCertification(BaseModel):
    cert_name: Optional[str] = None
    min_education_requirement: Optional[str] = None
    min_work_experience_requirement: Optional[str] = None
    exam_total_questions: Optional[str] = None
    exam_duration: Optional[str] = None
    renewal_credits_and_period: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class PMEducationOption(BaseModel):
    education_level: Optional[str] = None
    experience_months_required: Optional[str] = None


class PMCertification(BaseModel):
    cert_name: Optional[str] = None
    education_options: List[PMEducationOption] = Field(default_factory=list)
    training_or_contact_hours: Optional[str] = None
    has_random_audit: Optional[str] = None  # Expected values like "yes", "no", "true", "false", etc.
    reference_urls: List[str] = Field(default_factory=list)


class FinanceCertification(BaseModel):
    cert_name: Optional[str] = None
    exam_levels_count: Optional[str] = None
    first_level_total_questions: Optional[str] = None
    min_work_experience_hours: Optional[str] = None
    min_work_experience_months: Optional[str] = None
    annual_professional_learning_credits: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class SecurityCertification(BaseModel):
    cert_name: Optional[str] = None
    min_years_cumulative_experience: Optional[str] = None
    min_domains_experience_required: Optional[str] = None
    total_knowledge_domains: Optional[str] = None
    cpe_credits_and_cycle_years: Optional[str] = None
    annual_maintenance_fee_usd: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class CertsExtraction(BaseModel):
    hr: Optional[HRCertification] = None
    pm: Optional[PMCertification] = None
    finance: Optional[FinanceCertification] = None
    security: Optional[SecurityCertification] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_certs() -> str:
    return """
    Extract exactly one prominent certification for each of the four domains listed below, from the provided answer text.
    Return null for any field that is missing in the answer. Do not infer or invent details.

    For each certification, also extract reference_urls as an array of actual URLs explicitly present in the answer text.
    If the answer lists multiple URLs, include all of them. If no URL is given, return an empty array.

    Domains and fields:

    1) Human Resources (HRCertification object):
       - cert_name
       - min_education_requirement
       - min_work_experience_requirement
       - exam_total_questions
       - exam_duration
       - renewal_credits_and_period
       - reference_urls (array of URLs)
    
    2) Project Management (PMCertification object):
       - cert_name
       - education_options (array of PMEducationOption objects; for each option include:
           • education_level
           • experience_months_required
         )
       - training_or_contact_hours
       - has_random_audit (answer text's yes/no or equivalent; return the phrase as-is if present)
       - reference_urls (array of URLs)
    
    3) Financial Planning/Analysis (FinanceCertification object):
       - cert_name
       - exam_levels_count
       - first_level_total_questions
       - min_work_experience_hours
       - min_work_experience_months
       - annual_professional_learning_credits
       - reference_urls (array of URLs)
    
    4) Information Security (SecurityCertification object):
       - cert_name
       - min_years_cumulative_experience
       - min_domains_experience_required
       - total_knowledge_domains
       - cpe_credits_and_cycle_years
       - annual_maintenance_fee_usd
       - reference_urls (array of URLs)

    Notes:
    - Prefer strings for all values (e.g., "60 credits every 3 years", "180 questions", "3 years / 36 months") to capture the answer's exact phrasing.
    - Only include URLs explicitly present in the answer (plain URL or markdown link). If no URL is specified, use an empty array.
    - If the answer provides multiple certifications per domain, extract the first one mentioned.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_yes_no(value: Optional[str]) -> str:
    if not value:
        return "unknown"
    v = value.strip().lower()
    if v in {"yes", "true", "y", "t"}:
        return "yes"
    if v in {"no", "false", "n", "f"}:
        return "no"
    return "unknown"


def pm_options_names(pm: Optional[PMCertification]) -> str:
    if not pm or not pm.education_options:
        return ""
    names = [opt.education_level for opt in pm.education_options if opt.education_level]
    return "; ".join(names)


def pm_experience_mapping(pm: Optional[PMCertification]) -> str:
    if not pm or not pm.education_options:
        return ""
    parts = []
    for opt in pm.education_options:
        lvl = opt.education_level or ""
        months = opt.experience_months_required or ""
        if lvl or months:
            parts.append(f"{lvl}: {months}")
    return "; ".join(parts)


def sources_or_none(urls: Optional[List[str]]) -> Optional[List[str]]:
    if not urls:
        return None
    return urls


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_hr_certification(evaluator: Evaluator, parent_node, hr: Optional[HRCertification]) -> None:
    hr_node = evaluator.add_parallel(
        id="hr_certification",
        desc="Human Resources certification and required details",
        parent=parent_node,
        critical=False
    )

    # Reference existence (critical gate)
    ref_exists = bool(hr and hr.reference_urls and len(hr.reference_urls) > 0)
    evaluator.add_custom_node(
        result=ref_exists,
        id="hr_reference",
        desc="Provide a supporting reference URL from an official or authoritative source for the HR certification information",
        parent=hr_node,
        critical=True
    )

    # Prepare leaf nodes
    domain_leaf = evaluator.add_leaf(
        id="hr_domain_match",
        desc="The selected certification is in the Human Resources domain",
        parent=hr_node,
        critical=True
    )
    name_leaf = evaluator.add_leaf(
        id="hr_cert_name",
        desc="Provide the full official certification name",
        parent=hr_node,
        critical=True
    )
    edu_leaf = evaluator.add_leaf(
        id="hr_education_req",
        desc="Specify the minimum education requirement for eligibility",
        parent=hr_node,
        critical=True
    )
    exp_leaf = evaluator.add_leaf(
        id="hr_experience_req",
        desc="Specify the minimum work experience requirement (in years or months)",
        parent=hr_node,
        critical=True
    )
    questions_leaf = evaluator.add_leaf(
        id="hr_exam_questions",
        desc="Provide the total number of exam questions",
        parent=hr_node,
        critical=True
    )
    duration_leaf = evaluator.add_leaf(
        id="hr_exam_duration",
        desc="Provide the exam duration (in hours and minutes)",
        parent=hr_node,
        critical=True
    )
    renewal_leaf = evaluator.add_leaf(
        id="hr_renewal_credits",
        desc="Provide continuing education credits required for renewal AND the renewal period",
        parent=hr_node,
        critical=True
    )

    # Build claims
    cert_name = hr.cert_name if hr else ""
    domain_claim = f"The certification '{cert_name}' belongs to the Human Resources domain."
    name_claim = f"The official certification name is '{cert_name}'."
    edu_claim = f"The minimum education requirement for eligibility is: {hr.min_education_requirement if hr else ''}."
    exp_claim = f"The minimum work experience required is: {hr.min_work_experience_requirement if hr else ''}."
    questions_claim = f"The exam has a total of {hr.exam_total_questions if hr else ''} questions."
    duration_claim = f"The exam duration is {hr.exam_duration if hr else ''}."
    renewal_claim = f"The renewal requirements are: {hr.renewal_credits_and_period if hr else ''} (include credits and renewal period)."

    # Verify concurrently (excluding reference existence which is already added)
    await evaluator.batch_verify([
        (domain_claim, sources_or_none(hr.reference_urls if hr else None), domain_leaf,
         "Confirm the certification is an HR certification; rely on the referenced page's description or categorization."),
        (name_claim, sources_or_none(hr.reference_urls if hr else None), name_leaf,
         "Verify the official certification name exactly or with minor acceptable variants (e.g., abbreviation)."),
        (edu_claim, sources_or_none(hr.reference_urls if hr else None), edu_leaf,
         "Verify the minimum education requirement; accept equivalent phrasings."),
        (exp_claim, sources_or_none(hr.reference_urls if hr else None), exp_leaf,
         "Verify the minimum required work experience; accept equivalent units and minor rounding."),
        (questions_claim, sources_or_none(hr.reference_urls if hr else None), questions_leaf,
         "Verify the total number of exam questions; allow minor variations if versions differ but keep within reason."),
        (duration_claim, sources_or_none(hr.reference_urls if hr else None), duration_leaf,
         "Verify the exam duration; accept equivalent formatting of hours/minutes."),
        (renewal_claim, sources_or_none(hr.reference_urls if hr else None), renewal_leaf,
         "Verify both the continuing education credits required and the renewal period; accept official equivalents such as 'PDCs', 'CEUs', or 'recertification credits'."),
    ])


async def verify_pm_certification(evaluator: Evaluator, parent_node, pm: Optional[PMCertification]) -> None:
    pm_node = evaluator.add_parallel(
        id="pm_certification",
        desc="Project Management certification and required details",
        parent=parent_node,
        critical=False
    )

    # Reference existence (critical gate)
    ref_exists = bool(pm and pm.reference_urls and len(pm.reference_urls) > 0)
    evaluator.add_custom_node(
        result=ref_exists,
        id="pm_reference",
        desc="Provide a supporting reference URL from an official or authoritative source for the PM certification information",
        parent=pm_node,
        critical=True
    )

    # Leaf nodes
    domain_leaf = evaluator.add_leaf(
        id="pm_domain_match",
        desc="The selected certification is in the Project Management domain",
        parent=pm_node,
        critical=True
    )
    name_leaf = evaluator.add_leaf(
        id="pm_cert_name",
        desc="Provide the full official certification name",
        parent=pm_node,
        critical=True
    )
    edu_leaf = evaluator.add_leaf(
        id="pm_education_req",
        desc="Specify the education requirement options",
        parent=pm_node,
        critical=True
    )
    exp_leaf = evaluator.add_leaf(
        id="pm_experience_req",
        desc="Specify the project management experience requirement (in months) for each education level option",
        parent=pm_node,
        critical=True
    )
    training_leaf = evaluator.add_leaf(
        id="pm_training_hours",
        desc="Specify the required pre-exam training or contact hours",
        parent=pm_node,
        critical=True
    )
    audit_leaf = evaluator.add_leaf(
        id="pm_audit_process",
        desc="Indicate whether there is a random audit process for verifying experience",
        parent=pm_node,
        critical=True
    )

    # Build claims
    cert_name = pm.cert_name if pm else ""
    domain_claim = f"The certification '{cert_name}' belongs to the Project Management domain."
    name_claim = f"The official certification name is '{cert_name}'."
    edu_options_text = pm_options_names(pm)
    edu_claim = f"Education requirement options include: {edu_options_text}."
    exp_mapping_text = pm_experience_mapping(pm)
    exp_claim = f"The required project management experience in months per option is: {exp_mapping_text}."
    training_claim = f"The required pre-exam training/contact hours are: {pm.training_or_contact_hours if pm else ''}."

    audit_norm = normalize_yes_no(pm.has_random_audit if pm else None)
    if audit_norm == "yes":
        audit_claim = "The certification includes a random audit process for verifying experience."
    elif audit_norm == "no":
        audit_claim = "The certification does not include a random audit process for verifying experience."
    else:
        audit_claim = "The certification's policy mentions whether there is a random audit process for verifying experience."

    await evaluator.batch_verify([
        (domain_claim, sources_or_none(pm.reference_urls if pm else None), domain_leaf,
         "Confirm the certification is a Project Management certification via the referenced page."),
        (name_claim, sources_or_none(pm.reference_urls if pm else None), name_leaf,
         "Verify the official certification name exactly or accept widely used abbreviations as equivalent."),
        (edu_claim, sources_or_none(pm.reference_urls if pm else None), edu_leaf,
         "Verify the listed education requirement options; small phrasing differences are acceptable."),
        (exp_claim, sources_or_none(pm.reference_urls if pm else None), exp_leaf,
         "Verify the experience (in months) required for each education level option; accept reasonable formatting."),
        (training_claim, sources_or_none(pm.reference_urls if pm else None), training_leaf,
         "Verify the required pre-exam training/contact hours; accept minor variants in wording."),
        (audit_claim, sources_or_none(pm.reference_urls if pm else None), audit_leaf,
         "Verify whether a random audit process exists; for a 'no audit' claim, expect the source to explicitly indicate lack of random audits."),
    ])


async def verify_finance_certification(evaluator: Evaluator, parent_node, fin: Optional[FinanceCertification]) -> None:
    fin_node = evaluator.add_parallel(
        id="finance_certification",
        desc="Financial Planning/Analysis certification and required details",
        parent=parent_node,
        critical=False
    )

    # Reference existence (critical gate)
    ref_exists = bool(fin and fin.reference_urls and len(fin.reference_urls) > 0)
    evaluator.add_custom_node(
        result=ref_exists,
        id="fin_reference",
        desc="Provide a supporting reference URL from an official or authoritative source for the financial certification information",
        parent=fin_node,
        critical=True
    )

    # Leaf nodes
    domain_leaf = evaluator.add_leaf(
        id="fin_domain_match",
        desc="The selected certification is in the Financial Planning/Analysis domain",
        parent=fin_node,
        critical=True
    )
    name_leaf = evaluator.add_leaf(
        id="fin_cert_name",
        desc="Provide the full official certification name",
        parent=fin_node,
        critical=True
    )
    levels_leaf = evaluator.add_leaf(
        id="fin_exam_levels",
        desc="Specify the number of exam levels required to complete the certification",
        parent=fin_node,
        critical=True
    )
    questions_leaf = evaluator.add_leaf(
        id="fin_exam_questions",
        desc="Provide the total number of exam questions for the first exam level",
        parent=fin_node,
        critical=True
    )
    experience_leaf = evaluator.add_leaf(
        id="fin_work_experience",
        desc="Specify the minimum work experience required for the full designation in BOTH hours and months",
        parent=fin_node,
        critical=True
    )
    cpe_leaf = evaluator.add_leaf(
        id="fin_annual_cpe",
        desc="Specify the recommended or required annual professional learning / continuing education credits",
        parent=fin_node,
        critical=True
    )

    # Claims
    cert_name = fin.cert_name if fin else ""
    domain_claim = f"The certification '{cert_name}' belongs to the Financial Planning/Analysis domain."
    name_claim = f"The official certification name is '{cert_name}'."
    levels_claim = f"The certification requires {fin.exam_levels_count if fin else ''} exam level(s) to complete."
    questions_claim = f"The first exam level has a total of {fin.first_level_total_questions if fin else ''} questions."
    experience_claim = (
        f"The minimum work experience required for the full designation is "
        f"{fin.min_work_experience_hours if fin else ''} hours and {fin.min_work_experience_months if fin else ''} months."
    )
    cpe_claim = (
        f"The recommended or required annual professional learning/continuing education credits are: "
        f"{fin.annual_professional_learning_credits if fin else ''}."
    )

    await evaluator.batch_verify([
        (domain_claim, sources_or_none(fin.reference_urls if fin else None), domain_leaf,
         "Confirm the certification aligns with Financial Planning/Analysis (FP&A)."),
        (name_claim, sources_or_none(fin.reference_urls if fin else None), name_leaf,
         "Verify the official certification name; accept abbreviations if page shows equivalence."),
        (levels_claim, sources_or_none(fin.reference_urls if fin else None), levels_leaf,
         "Verify the number of exam levels."),
        (questions_claim, sources_or_none(fin.reference_urls if fin else None), questions_leaf,
         "Verify the total number of questions for the first exam level."),
        (experience_claim, sources_or_none(fin.reference_urls if fin else None), experience_leaf,
         "Verify work experience requirements including both hours and months; accept equivalent statements or ranges."),
        (cpe_claim, sources_or_none(fin.reference_urls if fin else None), cpe_leaf,
         "Verify annual professional learning or continuing education credit requirements or recommendations."),
    ])


async def verify_security_certification(evaluator: Evaluator, parent_node, sec: Optional[SecurityCertification]) -> None:
    sec_node = evaluator.add_parallel(
        id="security_certification",
        desc="Information Security certification and required details",
        parent=parent_node,
        critical=False
    )

    # Reference existence (critical gate)
    ref_exists = bool(sec and sec.reference_urls and len(sec.reference_urls) > 0)
    evaluator.add_custom_node(
        result=ref_exists,
        id="sec_reference",
        desc="Provide a supporting reference URL from an official or authoritative source for the security certification information",
        parent=sec_node,
        critical=True
    )

    # Leaf nodes
    domain_leaf = evaluator.add_leaf(
        id="sec_domain_match",
        desc="The selected certification is in the Information Security domain",
        parent=sec_node,
        critical=True
    )
    name_leaf = evaluator.add_leaf(
        id="sec_cert_name",
        desc="Provide the full official certification name",
        parent=sec_node,
        critical=True
    )
    years_leaf = evaluator.add_leaf(
        id="sec_experience_years",
        desc="Specify the minimum years of cumulative work experience required",
        parent=sec_node,
        critical=True
    )
    min_domains_leaf = evaluator.add_leaf(
        id="sec_domain_requirement",
        desc="Specify the minimum number of knowledge domains in which experience must be demonstrated",
        parent=sec_node,
        critical=True
    )
    total_domains_leaf = evaluator.add_leaf(
        id="sec_total_domains",
        desc="Specify the total number of knowledge domains covered by the certification",
        parent=sec_node,
        critical=True
    )
    cpe_leaf = evaluator.add_leaf(
        id="sec_cpe_credits",
        desc="Specify the CPE credits required for renewal AND the renewal cycle period (in years)",
        parent=sec_node,
        critical=True
    )
    fee_leaf = evaluator.add_leaf(
        id="sec_annual_fee",
        desc="Specify the annual maintenance fee in US dollars",
        parent=sec_node,
        critical=True
    )

    # Claims
    cert_name = sec.cert_name if sec else ""
    domain_claim = f"The certification '{cert_name}' belongs to the Information Security domain."
    name_claim = f"The official certification name is '{cert_name}'."
    years_claim = f"The minimum cumulative work experience required is {sec.min_years_cumulative_experience if sec else ''} years."
    min_domains_claim = (
        f"Experience must be demonstrated in at least {sec.min_domains_experience_required if sec else ''} knowledge domains."
    )
    total_domains_claim = f"The certification covers a total of {sec.total_knowledge_domains if sec else ''} knowledge domains."
    cpe_claim = (
        f"The renewal requires: {sec.cpe_credits_and_cycle_years if sec else ''} (CPE credits and cycle period)."
    )
    fee_claim = f"The annual maintenance fee is {sec.annual_maintenance_fee_usd if sec else ''} USD."

    await evaluator.batch_verify([
        (domain_claim, sources_or_none(sec.reference_urls if sec else None), domain_leaf,
         "Confirm the certification is an Information Security certification via the referenced page."),
        (name_claim, sources_or_none(sec.reference_urls if sec else None), name_leaf,
         "Verify the official certification name; abbreviations are acceptable if shown as equivalents."),
        (years_claim, sources_or_none(sec.reference_urls if sec else None), years_leaf,
         "Verify minimum years of cumulative experience; accept minor rounding."),
        (min_domains_claim, sources_or_none(sec.reference_urls if sec else None), min_domains_leaf,
         "Verify the minimum number of domains in which experience must be demonstrated."),
        (total_domains_claim, sources_or_none(sec.reference_urls if sec else None), total_domains_leaf,
         "Verify the total number of knowledge domains specified by the certification."),
        (cpe_claim, sources_or_none(sec.reference_urls if sec else None), cpe_leaf,
         "Verify CPE credits AND renewal cycle period in years."),
        (fee_claim, sources_or_none(sec.reference_urls if sec else None), fee_leaf,
         "Verify the annual maintenance fee in USD; accept minor currency formatting differences."),
    ])


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
    Evaluate the agent's answer for the certification comparison task.
    """
    # Initialize evaluator (root is non-critical by design in framework; use parallel aggregation)
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

    # Extract certifications structured data
    extraction = await evaluator.extract(
        prompt=prompt_extract_certs(),
        template_class=CertsExtraction,
        extraction_name="certs_extraction",
    )

    # Build and verify domain subtrees concurrently
    tasks: List[asyncio.Task] = []
    tasks.append(asyncio.create_task(verify_hr_certification(evaluator, root, extraction.hr)))
    tasks.append(asyncio.create_task(verify_pm_certification(evaluator, root, extraction.pm)))
    tasks.append(asyncio.create_task(verify_finance_certification(evaluator, root, extraction.finance)))
    tasks.append(asyncio.create_task(verify_security_certification(evaluator, root, extraction.security)))
    await asyncio.gather(*tasks, return_exceptions=True)

    # Return summarized evaluation result
    return evaluator.get_summary()