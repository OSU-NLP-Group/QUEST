import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task metadata
# -----------------------------------------------------------------------------
TASK_ID = "cert_eligibility_eval"
TASK_DESCRIPTION = (
    "A software engineer holds a bachelor's degree in computer science and has exactly 4 years of full-time professional experience in their career so far. "
    "Their experience breaks down as follows: 2 years working in application security (performing security assessments, implementing security controls, and managing incident response), "
    "1.5 years serving as a project lead managing software development projects, and 6 months working with AWS cloud infrastructure and services. "
    "Additionally, they have completed a 40-hour project management training course. "
    "This professional wants to pursue certification to advance their career. Evaluate their current eligibility for the following four certifications: "
    "Project Management Professional (PMP), Certified Information Systems Security Professional (CISSP), AWS Certified Solutions Architect - Associate, and CompTIA Security+. "
    "For each certification, determine whether they currently meet all eligibility requirements to take the certification exam, and provide specific reasoning based on the official requirements."
)


# -----------------------------------------------------------------------------
# Extraction Models
# -----------------------------------------------------------------------------
class PMPExtract(BaseModel):
    eligibility: Optional[str] = None  # "eligible" | "not eligible" | "unclear"
    requirement_urls: List[str] = Field(default_factory=list)
    degree_path_acknowledged: Optional[bool] = None  # recognizes bachelor's → 36-month pathway
    pm_experience_months: Optional[str] = None       # what the answer claims for PM months
    pm_experience_hours: Optional[str] = None        # what the answer claims for PM hours (if any)
    pm_training_hours: Optional[str] = None          # what the answer claims for PM education hours


class CISSPExtract(BaseModel):
    eligibility: Optional[str] = None
    requirement_urls: List[str] = Field(default_factory=list)
    degree_waiver_acknowledged: Optional[bool] = None  # recognizes 1-year waiver for a 4-year degree
    total_experience_years: Optional[str] = None
    domains_covered: List[str] = Field(default_factory=list)  # domains the answer claims are covered


class AWSAssociateExtract(BaseModel):
    eligibility: Optional[str] = None
    requirement_urls: List[str] = Field(default_factory=list)
    one_year_recommended_acknowledged: Optional[bool] = None  # recognizes recommendation (not a strict prerequisite)
    aws_experience_months: Optional[str] = None               # what the answer claims for AWS months


class SecurityPlusExtract(BaseModel):
    eligibility: Optional[str] = None
    requirement_urls: List[str] = Field(default_factory=list)
    no_formal_prereq_claimed: Optional[bool] = None           # recognizes no formal prerequisites


class EligibilityExtraction(BaseModel):
    pmp: Optional[PMPExtract] = None
    cissp: Optional[CISSPExtract] = None
    aws_sa_associate: Optional[AWSAssociateExtract] = None
    security_plus: Optional[SecurityPlusExtract] = None


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_eligibility() -> str:
    return """
    You will extract the candidate's stated eligibility assessment (from the answer text) for four certifications and the sources they cited.
    For each certification, extract the following fields strictly from the answer text:

    PMP (Project Management Professional):
    - eligibility: one of ["eligible", "not eligible", "unclear"] based on the answer's explicit conclusion
    - requirement_urls: list of URLs that the answer cites as official/authoritative requirement pages for PMP
    - degree_path_acknowledged: true/false if the answer explicitly recognizes that a four-year bachelor's degree places the candidate on the 36-month experience pathway (as opposed to the 60-month pathway for diploma holders)
    - pm_experience_months: the project's management experience in months according to the answer (string as stated; do not compute)
    - pm_experience_hours: any project management experience hours stated (string as stated)
    - pm_training_hours: the PM education/training hours stated in the answer (string as stated)
    
    CISSP:
    - eligibility: one of ["eligible", "not eligible", "unclear"]
    - requirement_urls: list of URLs the answer cites for CISSP requirements
    - degree_waiver_acknowledged: true/false if the answer explicitly recognizes the 1-year experience waiver for a 4-year degree
    - total_experience_years: the total years of professional experience as the answer uses in the CISSP reasoning (string as stated)
    - domains_covered: list of CISSP domains the answer explicitly claims the candidate's experience covers

    AWS Certified Solutions Architect – Associate:
    - eligibility: one of ["eligible", "not eligible", "unclear"]
    - requirement_urls: list of URLs the answer cites for AWS SA-Associate exam details/recommendations
    - one_year_recommended_acknowledged: true/false if the answer explicitly acknowledges the recommendation of ~1 year of hands-on AWS experience (not a hard prerequisite)
    - aws_experience_months: the months of AWS hands-on experience used in the answer's reasoning (string as stated)

    CompTIA Security+:
    - eligibility: one of ["eligible", "not eligible", "unclear"]
    - requirement_urls: list of URLs the answer cites for Security+ requirements
    - no_formal_prereq_claimed: true/false if the answer explicitly states there are no formal prerequisites

    Rules:
    - Extract only what is explicitly mentioned in the answer. Do not infer or compute values.
    - If an item is not mentioned, set it to null (for strings/booleans) or [] for lists.
    - For URLs, return only actual URLs present in the answer text (plain links or markdown links).
    """


# -----------------------------------------------------------------------------
# Helper to build safe source lists
# -----------------------------------------------------------------------------
def safe_urls(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


# -----------------------------------------------------------------------------
# Verification Subtrees
# -----------------------------------------------------------------------------
async def verify_pmp(evaluator: Evaluator, parent, pmp: Optional[PMPExtract]) -> None:
    # Parent node for PMP
    pmp_node = evaluator.add_parallel(
        id="pmp_certification_assessment",
        desc="Evaluates eligibility for PMP (Project Management Professional) certification",
        parent=parent,
        critical=False,
    )

    # Requirements source existence (critical)
    pmp_source_exists = evaluator.add_custom_node(
        result=bool(pmp and pmp.requirement_urls and len(pmp.requirement_urls) > 0),
        id="pmp_requirements_source",
        desc="Provides authoritative source URL for PMP certification requirements",
        parent=pmp_node,
        critical=True
    )

    # Eligibility requirement checks (critical group)
    pmp_reqs = evaluator.add_parallel(
        id="pmp_eligibility_requirements",
        desc="Verifies that all PMP eligibility requirements are correctly evaluated against the candidate's profile",
        parent=pmp_node,
        critical=True
    )

    # Bachelor's Degree Verification (by URL)
    deg_node = evaluator.add_leaf(
        id="pmp_bachelors_degree_verification",
        desc="Confirms that the candidate's bachelor's degree qualifies them for the 36-month experience pathway (as opposed to the 60-month pathway for high school diploma holders)",
        parent=pmp_reqs,
        critical=True
    )
    await evaluator.verify(
        claim="For the PMP, candidates with a four-year bachelor's degree must have 36 months (3 years) of leading projects (within the last 8 years).",
        node=deg_node,
        sources=safe_urls(pmp.requirement_urls if pmp else []),
        additional_instruction="Verify on the official PMI PMP eligibility page. Focus only on the policy text regarding a 4-year degree and months of experience."
    )

    # 36-Month PM Experience Verification (logical check using profile)
    pm36_node = evaluator.add_leaf(
        id="pmp_36_month_pm_experience_verification",
        desc="Verifies whether the candidate's project management experience meets the minimum 36-month duration requirement for PMP eligibility with a bachelor's degree",
        parent=pmp_reqs,
        critical=True
    )
    await evaluator.verify(
        claim="Based on the provided profile, the candidate does NOT meet the PMP requirement of at least 36 months of leading projects (they have only 1.5 years as a project lead, i.e., ~18 months).",
        node=pm36_node,
        additional_instruction="Use only the task description's profile to compute months of project leadership. Do not assume other periods included project leadership unless explicitly stated."
    )

    # 4500-Hour PM Experience Verification (logical check using profile; treat as an additional check)
    hours_node = evaluator.add_leaf(
        id="pmp_4500_hour_pm_experience_verification",
        desc="Verifies whether the candidate has accumulated at least 4,500 hours of project management experience as required for PMP eligibility with a bachelor's degree",
        parent=pmp_reqs,
        critical=True
    )
    await evaluator.verify(
        claim="Given only 1.5 years (approximately 18 months) in a project lead capacity, the candidate has NOT accumulated as many as 4,500 hours of project management leadership experience.",
        node=hours_node,
        additional_instruction="Assume ~2,000 working hours per year for rough estimation. This is a logical arithmetic check using the profile, not a requirements policy verification."
    )

    # 35-Hour PM Training Verification (logical check using profile)
    training_node = evaluator.add_leaf(
        id="pmp_35_hour_pm_training_verification",
        desc="Verifies whether the candidate has completed at least 35 hours of project management education or training as required for PMP eligibility",
        parent=pmp_reqs,
        critical=True
    )
    await evaluator.verify(
        claim="The candidate completed a 40-hour project management training course, which satisfies the PMP requirement of at least 35 hours of PM education.",
        node=training_node,
        additional_instruction="Use the task description's profile for the training hours; this is a simple threshold check."
    )


async def verify_cissp(evaluator: Evaluator, parent, cissp: Optional[CISSPExtract]) -> None:
    # Parent node for CISSP
    cissp_node = evaluator.add_parallel(
        id="cissp_certification_assessment",
        desc="Evaluates eligibility for CISSP (Certified Information Systems Security Professional) certification",
        parent=parent,
        critical=False
    )

    # Requirements source existence (critical)
    cissp_source_exists = evaluator.add_custom_node(
        result=bool(cissp and cissp.requirement_urls and len(cissp.requirement_urls) > 0),
        id="cissp_requirements_source",
        desc="Provides authoritative source URL for CISSP certification requirements",
        parent=cissp_node,
        critical=True
    )

    # Eligibility requirement checks (critical group)
    cissp_reqs = evaluator.add_parallel(
        id="cissp_eligibility_requirements",
        desc="Verifies that all CISSP eligibility requirements are correctly evaluated against the candidate's profile",
        parent=cissp_node,
        critical=True
    )

    # Degree Waiver Recognition (by URL)
    waiver_node = evaluator.add_leaf(
        id="cissp_degree_waiver_recognition",
        desc="Recognizes that the candidate's four-year bachelor's degree qualifies for a one-year experience waiver, reducing the requirement from 5 years to 4 years",
        parent=cissp_reqs,
        critical=True
    )
    await evaluator.verify(
        claim="For CISSP, a four-year college degree qualifies for a one-year experience waiver, reducing the required paid work experience from 5 years to 4 years.",
        node=waiver_node,
        sources=safe_urls(cissp.requirement_urls if cissp else []),
        additional_instruction="Verify this policy on the official (ISC)²/ISC2 CISSP eligibility page."
    )

    # 4-Year Experience Verification (logical check using profile)
    exp4_node = evaluator.add_leaf(
        id="cissp_4_year_experience_verification",
        desc="Verifies whether the candidate's total professional experience meets the 4-year requirement (after applying the degree waiver)",
        parent=cissp_reqs,
        critical=True
    )
    await evaluator.verify(
        claim="Based on the profile, the candidate has a total of 4 years of full-time professional experience, meeting the 4-year requirement after the one-year degree waiver.",
        node=exp4_node,
        additional_instruction="Use only the task description's total experience (exactly 4 years)."
    )

    # Multi-Domain Coverage Verification (logical check)
    domains_node = evaluator.add_leaf(
        id="cissp_multi_domain_coverage_verification",
        desc="Verifies whether the candidate's experience covers two or more of the eight CISSP domains (Security and Risk Management, Asset Security, Security Architecture and Engineering, Communication and Network Security, Identity and Access Management, Security Assessment and Testing, Security Operations, Software Development Security)",
        parent=cissp_reqs,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "Based on the profile, the candidate's experience covers at least two CISSP domains. "
            "Examples: application security work involves Security Assessment and Testing and Security Operations; "
            "AWS work relates to Security Architecture and Engineering and potentially IAM; "
            "project leadership touches Security and Risk Management for governance/planning."
        ),
        node=domains_node,
        additional_instruction="Map the described duties to the CISSP domains and determine if at least two domains are reasonably covered."
    )


async def verify_aws_sa_associate(evaluator: Evaluator, parent, aws: Optional[AWSAssociateExtract]) -> None:
    # Parent node for AWS SA-Associate
    aws_node = evaluator.add_parallel(
        id="aws_sa_associate_assessment",
        desc="Evaluates eligibility for AWS Certified Solutions Architect - Associate certification",
        parent=parent,
        critical=False
    )

    # Requirements source existence (critical)
    aws_source_exists = evaluator.add_custom_node(
        result=bool(aws and aws.requirement_urls and len(aws.requirement_urls) > 0),
        id="aws_requirements_source",
        desc="Provides authoritative source URL for AWS Certified Solutions Architect - Associate certification requirements",
        parent=aws_node,
        critical=True
    )

    # Experience recommendation verification (treat as logical check against profile)
    aws_exp_node = evaluator.add_leaf(
        id="aws_experience_verification",
        desc="Verifies whether the candidate meets the recommended 1 year of hands-on experience designing cloud solutions using AWS services",
        parent=aws_node,
        critical=True
    )
    await evaluator.verify(
        claim="The candidate does NOT meet the recommended ~1 year of hands-on AWS experience because they have only ~6 months.",
        node=aws_exp_node,
        additional_instruction="This is a simple comparison against the profile's AWS experience duration. Note: the recommendation is not a strict prerequisite."
    )


async def verify_security_plus(evaluator: Evaluator, parent, secp: Optional[SecurityPlusExtract]) -> None:
    # Parent node for Security+
    secp_node = evaluator.add_parallel(
        id="security_plus_assessment",
        desc="Evaluates eligibility for CompTIA Security+ certification",
        parent=parent,
        critical=False
    )

    # Requirements source existence (critical)
    secp_source_exists = evaluator.add_custom_node(
        result=bool(secp and secp.requirement_urls and len(secp.requirement_urls) > 0),
        id="security_plus_requirements_source",
        desc="Provides authoritative source URL for CompTIA Security+ certification requirements",
        parent=secp_node,
        critical=True
    )

    # Prerequisites Analysis (by URL)
    secp_prereq_node = evaluator.add_leaf(
        id="security_plus_prerequisites_analysis",
        desc="Correctly identifies that CompTIA Security+ has no formal prerequisites, meaning any candidate automatically qualifies to take the exam regardless of their experience level",
        parent=secp_node,
        critical=True
    )
    await evaluator.verify(
        claim="CompTIA Security+ has no formal prerequisites to sit for the exam; recommendations (e.g., foundational knowledge, years of experience) are not required.",
        node=secp_prereq_node,
        sources=safe_urls(secp.requirement_urls if secp else []),
        additional_instruction="Verify on the official CompTIA Security+ exam page that there are no formal prerequisites."
    )


# -----------------------------------------------------------------------------
# Main evaluation entrypoint
# -----------------------------------------------------------------------------
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
    Evaluate the given answer for certification eligibility checks using a rubric-based verification tree.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation at top-level across certifications
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Evaluates whether a candidate with a specified professional profile meets the eligibility requirements for PMP, CISSP, AWS Certified Solutions Architect - Associate, and CompTIA Security+ certifications",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )
    # NOTE: Root is intentionally non-critical to satisfy the framework constraint
    # that a critical parent cannot have non-critical children.

    # 1) Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_eligibility(),
        template_class=EligibilityExtraction,
        extraction_name="eligibility_extraction"
    )

    # 2) Build subtrees for each certification (parallel under root)
    await verify_pmp(evaluator, root, extraction.pmp if extraction else None)
    await verify_cissp(evaluator, root, extraction.cissp if extraction else None)
    await verify_aws_sa_associate(evaluator, root, extraction.aws_sa_associate if extraction else None)
    await verify_security_plus(evaluator, root, extraction.security_plus if extraction else None)

    # 3) Optional: Add a compact computed overview for context (non-scoring)
    #    This is purely informational and does not affect grading.
    evaluator.add_custom_info({
        "profile_summary": {
            "education": "Bachelor's degree in computer science",
            "experience_years_total": 4.0,
            "experience_breakdown": {
                "application_security_years": 2.0,
                "project_lead_years": 1.5,
                "aws_cloud_years": 0.5
            },
            "project_management_training_hours": 40
        },
        "notes": "PMP requires 36 months leading projects and 35 hours of PM education (training met, months not met). "
                 "CISSP with 4-year degree recognizes a 1-year waiver (4 years required; candidate has 4 years), plus coverage of 2+ domains. "
                 "AWS SA-Associate has no strict prerequisite; ~1 year hands-on recommended (candidate has ~6 months). "
                 "CompTIA Security+ has no formal prerequisites."
    }, info_type="context", info_name="computed_context")

    # 4) Return the structured evaluation summary
    return evaluator.get_summary()