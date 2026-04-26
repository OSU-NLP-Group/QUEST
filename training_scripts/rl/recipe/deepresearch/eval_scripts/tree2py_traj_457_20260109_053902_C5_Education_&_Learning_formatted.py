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
TASK_ID = "coursera_ace_certificates"
TASK_DESCRIPTION = """
I'm planning to accelerate my bachelor's degree by completing professional certificates that offer ACE (American Council on Education) credit recommendations. I want to identify four different professional certificates offered on Coursera that have active ACE credit recommendations and can be transferred to U.S. universities.

For each of the four certificates, please provide the following information:
1. The complete certificate name and the provider organization
2. The number of ACE credit hours recommended for the certificate
3. The validity period of the ACE recommendation (with start and end dates)
4. At least two U.S. universities that have documented policies accepting ACE credits from professional certificates for transfer
5. The required documentation method for transferring these credits (such as Credly transcript or ACE transcript service)
6. A reference URL to the ACE National Guide page or official source confirming the ACE credit recommendation

The certificates should be from major providers such as Google, IBM, Meta, or Microsoft, and should currently have active ACE recommendations valid as of January 2026.
"""

ALLOWED_MAJOR_PROVIDERS = {"google", "ibm", "meta", "microsoft"}
ACTIVE_CHECK_MONTH = "January 2026"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityPolicy(BaseModel):
    name: Optional[str] = None
    policy_url: Optional[str] = None
    # Optional field in answer describing method (e.g., Credly, ACE transcript)
    doc_method: Optional[str] = None


class CertificateInfo(BaseModel):
    certificate_name: Optional[str] = None
    provider: Optional[str] = None
    coursera_url: Optional[str] = None
    ace_credit_hours: Optional[str] = None
    validity_start_date: Optional[str] = None
    validity_end_date: Optional[str] = None
    ace_reference_url: Optional[str] = None
    universities: List[UniversityPolicy] = Field(default_factory=list)
    documentation_methods: List[str] = Field(default_factory=list)
    documentation_source_urls: List[str] = Field(default_factory=list)


class CertificatesExtraction(BaseModel):
    certificates: List[CertificateInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_certificates() -> str:
    return """
    Extract up to four distinct Coursera Professional Certificates that are mentioned in the answer and have ACE credit recommendations. If the answer includes more than four, extract the first four by their order of appearance. For each certificate, return an object with the following fields:

    - certificate_name: Full certificate title as stated in the answer.
    - provider: Provider organization (e.g., Google, IBM, Meta, Microsoft).
    - coursera_url: The Coursera certificate page URL (full URL; use http/https).
    - ace_credit_hours: The ACE-recommended credit hours (as text; do not convert to a number; keep any ranges or units).
    - validity_start_date: The start date of the ACE recommendation validity period (as text).
    - validity_end_date: The end date of the ACE recommendation validity period (as text).
    - ace_reference_url: A URL to the ACE National Guide page or other official source that explicitly confirms the ACE recommendation for this certificate (full URL).
    - universities: An array of at least two items (if available), each with:
        - name: University name.
        - policy_url: URL to the university policy page (or registrar/credit transfer page) that documents accepting ACE-recommended credits from professional certificates.
        - doc_method: If mentioned, the required/accepted documentation method (e.g., "Credly transcript", "ACE transcript service"). If not mentioned, set null.
    - documentation_methods: A list of documentation methods mentioned anywhere in the answer for transferring these credits (e.g., "Credly transcript", "ACE transcript service"). If none, return an empty list.
    - documentation_source_urls: A list of URLs that describe or confirm the documentation method accepted (can reuse university policy URLs if applicable). If none, return an empty list.

    RULES:
    - Only extract URLs explicitly present in the answer. If a URL is missing but referenced (e.g., "ACE National Guide"), return null for that URL.
    - Preserve text exactly as presented, including date formats (e.g., "Jan 1, 2024" or "2024-01-01").
    - If any field is missing, set it to null (for singular fields) or an empty list (for array fields).
    - Do not invent information. Extract only from the provided answer content.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_provider(provider: Optional[str]) -> str:
    if not provider:
        return ""
    return provider.strip().lower()


def is_major_provider(provider: Optional[str]) -> bool:
    return normalize_provider(provider) in ALLOWED_MAJOR_PROVIDERS


def pick_first_non_empty(items: List[str]) -> Optional[str]:
    for s in items:
        if s and s.strip():
            return s.strip()
    return None


def get_two_universities(unis: List[UniversityPolicy]) -> List[UniversityPolicy]:
    result = []
    for u in unis:
        if u and u.name and u.name.strip():
            result.append(u)
        if len(result) >= 2:
            break
    return result


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_certificate(
        evaluator: Evaluator,
        parent_node,
        cert: CertificateInfo,
        idx: int,
) -> None:
    """
    Build verification sub-tree and run checks for a single certificate.
    All children under a certificate node are critical per rubric.
    """
    cert_node = evaluator.add_parallel(
        id=f"certificate_{idx + 1}",
        desc=f"Certificate {idx + 1:02d}: required fields provided and meet constraints",
        parent=parent_node,
        critical=False  # Certificate groups contribute soft credit to root
    )

    # 1) Name & Provider existence (critical)
    name_provider_exists = bool(cert.certificate_name and cert.certificate_name.strip()) and bool(cert.provider and cert.provider.strip())
    evaluator.add_custom_node(
        result=name_provider_exists,
        id=f"cert{idx + 1}_name_provider",
        desc="Provides complete certificate name and provider organization",
        parent=cert_node,
        critical=True
    )

    # 2) Major provider check (critical)
    evaluator.add_custom_node(
        result=is_major_provider(cert.provider),
        id=f"cert{idx + 1}_major_provider",
        desc="Provider is one of: Google, IBM, Meta, Microsoft",
        parent=cert_node,
        critical=True
    )

    # 3) Coursera platform check (critical)
    coursera_leaf = evaluator.add_leaf(
        id=f"cert{idx + 1}_coursera_platform",
        desc="Certificate is offered on Coursera",
        parent=cert_node,
        critical=True
    )
    coursera_claim = f"This page is a Coursera Professional Certificate page for '{cert.certificate_name}' provided by {cert.provider}."
    await evaluator.verify(
        claim=coursera_claim,
        node=coursera_leaf,
        sources=cert.coursera_url,
        additional_instruction="Verify the page is on coursera.org and corresponds to a Professional Certificate. Allow minor naming variations."
    )

    # 4) ACE credits hours (critical)
    ace_credits_leaf = evaluator.add_leaf(
        id=f"cert{idx + 1}_ace_credits",
        desc="States the ACE-recommended credit hours for the certificate",
        parent=cert_node,
        critical=True
    )
    ace_credits_claim = f"The ACE National Guide (or official source) recommends '{cert.ace_credit_hours}' credit hours for the certificate '{cert.certificate_name}'."
    await evaluator.verify(
        claim=ace_credits_claim,
        node=ace_credits_leaf,
        sources=cert.ace_reference_url,
        additional_instruction="Check that the page explicitly states an ACE credit recommendation for the certificate. Accept equivalent wording (e.g., 'semester hours')."
    )

    # 5) Validity period + active in January 2026 (critical, split into two checks under a critical parallel node)
    validity_node = evaluator.add_parallel(
        id=f"cert{idx + 1}_validity_period",
        desc=f"ACE validity period provided and active as of {ACTIVE_CHECK_MONTH}",
        parent=cert_node,
        critical=True
    )

    validity_dates_leaf = evaluator.add_leaf(
        id=f"cert{idx + 1}_validity_dates",
        desc="ACE recommendation validity period (start and end dates) is stated and matches the source",
        parent=validity_node,
        critical=True
    )
    validity_dates_claim = (
        f"The ACE page lists a validity period for this certificate with start date '{cert.validity_start_date}' and end date '{cert.validity_end_date}', "
        f"or an equivalent date representation."
    )
    await evaluator.verify(
        claim=validity_dates_claim,
        node=validity_dates_leaf,
        sources=cert.ace_reference_url,
        additional_instruction="Confirm the page shows a validity period. Allow minor date format differences; focus on whether the start/end dates match in substance."
    )

    valid_in_jan2026_leaf = evaluator.add_leaf(
        id=f"cert{idx + 1}_valid_in_jan2026",
        desc=f"ACE recommendation is active/valid as of {ACTIVE_CHECK_MONTH}",
        parent=validity_node,
        critical=True
    )
    valid_in_jan2026_claim = f"The ACE recommendation for this certificate is active and valid in {ACTIVE_CHECK_MONTH}."
    await evaluator.verify(
        claim=valid_in_jan2026_claim,
        node=valid_in_jan2026_leaf,
        sources=cert.ace_reference_url,
        additional_instruction="Use the validity period on the page to determine if the recommendation remains active in January 2026 (e.g., end date after 2026-01-31 or 'current')."
    )

    # 6) Universities policies + RA (critical, broken into granular checks under a critical parallel node)
    universities_node = evaluator.add_parallel(
        id=f"cert{idx + 1}_universities_policy_ra",
        desc="At least two RA U.S. universities with documented policies accepting ACE-recommended credits",
        parent=cert_node,
        critical=True
    )

    two_unis = get_two_universities(cert.universities)
    min_two_with_urls = len(two_unis) >= 2 and all(u.policy_url and u.policy_url.strip() for u in two_unis)
    evaluator.add_custom_node(
        result=min_two_with_urls,
        id=f"cert{idx + 1}_universities_min_two",
        desc="At least two universities provided with policy URLs",
        parent=universities_node,
        critical=True
    )

    # Policy acceptance checks for first two provided universities
    for j, uni in enumerate(two_unis[:2]):
        uni_leaf = evaluator.add_leaf(
            id=f"cert{idx + 1}_uni_policy_{j + 1}",
            desc=f"Policy page supports acceptance of ACE credits for transfer: {uni.name}",
            parent=universities_node,
            critical=True
        )
        uni_claim = (
            f"According to this policy page, {uni.name} accepts ACE-recommended credits for transfer, "
            f"including credits originating from professional certificates."
        )
        await evaluator.verify(
            claim=uni_claim,
            node=uni_leaf,
            sources=uni.policy_url,
            additional_instruction="Confirm acceptance of ACE credits. Allow equivalent phrasing such as 'ACE credit recommendations' or 'ACE evaluated learning'."
        )

    # Regional Accreditation check (using simple verification; explicitly allow using general knowledge)
    if len(two_unis) >= 2:
        ra_leaf = evaluator.add_leaf(
            id=f"cert{idx + 1}_universities_ra_status",
            desc="The listed universities are regionally accredited in the U.S.",
            parent=universities_node,
            critical=True
        )
        names_str = ", ".join([u.name for u in two_unis[:2]])
        ra_claim = f"The universities {names_str} are regionally accredited in the United States."
        await evaluator.verify(
            claim=ra_claim,
            node=ra_leaf,
            sources=None,
            additional_instruction="You may use your own knowledge to judge regional accreditation for well-known U.S. universities."
        )

    # 7) Documentation method (critical): require/accept Credly or ACE transcript service; verify via provided sources
    documentation_parent = evaluator.add_parallel(
        id=f"cert{idx + 1}_documentation_main",
        desc="Documentation method for transferring ACE credits is specified and supported",
        parent=cert_node,
        critical=True
    )

    doc_method = pick_first_non_empty(cert.documentation_methods) or pick_first_non_empty([u.doc_method or "" for u in cert.universities])
    doc_exists = bool(doc_method)
    evaluator.add_custom_node(
        result=doc_exists,
        id=f"cert{idx + 1}_documentation_provided",
        desc="Documentation method is provided (e.g., Credly transcript or ACE transcript service)",
        parent=documentation_parent,
        critical=True
    )

    docs_leaf = evaluator.add_leaf(
        id=f"cert{idx + 1}_documentation_supported",
        desc="Required/accepted documentation method is supported by cited sources",
        parent=documentation_parent,
        critical=True
    )
    sources_for_docs = cert.documentation_source_urls[:] if cert.documentation_source_urls else [u.policy_url for u in two_unis if u.policy_url]
    docs_claim = f"The accepted documentation method for transferring ACE credits includes '{doc_method}'."
    await evaluator.verify(
        claim=docs_claim,
        node=docs_leaf,
        sources=sources_for_docs,
        additional_instruction="Confirm that the sources mention or accept the stated documentation method (e.g., Credly transcript, ACE transcript service). Allow minor wording differences."
    )

    # 8) Reference URL (critical): ACE National Guide or official source confirms recommendation
    ref_leaf = evaluator.add_leaf(
        id=f"cert{idx + 1}_reference",
        desc="Provides a reference URL to the ACE National Guide page or other official source confirming the ACE recommendation",
        parent=cert_node,
        critical=True
    )
    ref_claim = f"This reference page confirms the ACE credit recommendation for the certificate '{cert.certificate_name}'."
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=cert.ace_reference_url,
        additional_instruction="Verify that this is an authoritative page (ACE National Guide or equivalent) explicitly stating an ACE recommendation."
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
    """
    Evaluate an answer for the Coursera ACE Professional Certificates task.
    """
    # Initialize evaluator
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

    # Record allowed providers info (custom info)
    evaluator.add_custom_info(
        {"allowed_major_providers": sorted(list(ALLOWED_MAJOR_PROVIDERS)), "active_check_month": ACTIVE_CHECK_MONTH},
        info_type="config",
        info_name="provider_constraints"
    )

    # Extract certificates from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_certificates(),
        template_class=CertificatesExtraction,
        extraction_name="coursera_ace_certificates",
    )

    # Ensure we have exactly 4 items; if fewer, pad; if more, slice
    certs = (extracted.certificates or [])[:4]
    while len(certs) < 4:
        certs.append(CertificateInfo())

    # Critical check: All four certificates are distinct (no duplicates)
    names = [c.certificate_name.strip() if c.certificate_name else "" for c in certs]
    non_empty_names = [n for n in names if n]
    unique_names_lower = set(n.lower() for n in non_empty_names)
    distinct_all_four = (len(non_empty_names) == 4) and (len(unique_names_lower) == 4)

    evaluator.add_custom_node(
        result=distinct_all_four,
        id="certificate_distinctness",
        desc="All four certificates are distinct (no duplicates)",
        parent=root,
        critical=True
    )

    # Build verification subtrees for each certificate
    for i, cert in enumerate(certs):
        await verify_certificate(evaluator, root, cert, i)

    # Return structured result using the evaluator's summary
    return evaluator.get_summary()