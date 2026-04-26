import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
import re

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pmi_hrci_shrm_certifications"
TASK_DESCRIPTION = (
    "A professional with a bachelor's degree and 4 years of relevant work experience is considering pursuing a "
    "professional certification to advance their career in either project management or human resources. Identify "
    "professional certifications from PMI (Project Management Institute) and/or HRCI/SHRM (human resources certifications) "
    "that this person would be eligible to pursue. For each certification you identify, provide: (1) The specific eligibility "
    "requirements (education, experience, and training hours), (2) The current exam cost (including both member and non-member "
    "pricing where applicable), (3) The renewal/recertification requirements (if applicable), and (4) A reference URL from the official "
    "certifying organization. Additionally, for any PMI certifications that require formal project management education hours, identify how "
    "the candidate can fulfill this requirement through PMI Authorized Training Partners, and provide the official PMI resource for finding such "
    "training providers."
)

ALLOWED_ISSUERS = {
    "pmi": ["pmi", "project management institute"],
    "hrci": ["hrci", "hr certification institute"],
    "shrm": ["shrm", "society for human resource management"],
}
OFFICIAL_DOMAINS = {
    "PMI": ["pmi.org"],
    "HRCI": ["hrci.org"],
    "SHRM": ["shrm.org", "shrmcertification.org"],
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EligibilityInfo(BaseModel):
    education: Optional[str] = None
    experience: Optional[str] = None
    training_hours_required: Optional[bool] = None
    training_hours_detail: Optional[str] = None


class CostInfo(BaseModel):
    member_price: Optional[str] = None
    non_member_price: Optional[str] = None
    other_fee_notes: Optional[str] = None


class RenewalInfo(BaseModel):
    renewal_requirements: Optional[str] = None


class CertificationExtractedItem(BaseModel):
    name: Optional[str] = None
    issuer: Optional[str] = None
    eligibility: Optional[EligibilityInfo] = None
    candidate_eligibility_justification: Optional[str] = None
    exam_cost: Optional[CostInfo] = None
    renewal: Optional[RenewalInfo] = None
    official_urls: List[str] = Field(default_factory=list)


class CertificationsExtraction(BaseModel):
    certifications: List[CertificationExtractedItem] = Field(default_factory=list)
    pmi_atp_guidance_text: Optional[str] = None
    pmi_atp_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_certifications() -> str:
    return """
    Extract up to 5 certifications (in the order they appear) that the answer proposes for the candidate. Only include certifications from PMI, HRCI, or SHRM.
    For each certification, extract the following fields:
    - name: The certification name as stated (e.g., "PMP", "CAPM", "SHRM-CP", "SHRM-SCP", "PHR", "SPHR").
    - issuer: The certifying organization named in the answer (e.g., PMI, HRCI, SHRM; accept full names such as "Project Management Institute" or "Society for Human Resource Management").
    - official_urls: A list of official reference URLs from the certifying organization (must be explicit URLs in the answer). Only include official organization domains:
        * PMI: pmi.org
        * HRCI: hrci.org
        * SHRM: shrm.org or shrmcertification.org
      If the answer does not provide any official URL for the certification, return an empty list.
    - eligibility: The certification’s eligibility requirements as stated in the answer:
        * education: education requirement text (copy exact phrasing from the answer; if not provided, null)
        * experience: experience requirement text (copy exact phrasing; if not provided, null)
        * training_hours_required: true/false if the answer explicitly indicates training/contact/education hours are required (if not clear, return null)
        * training_hours_detail: the stated number or description of required formal education/training/contact hours (or explicitly "none" if the answer states none; if absent, null)
    - candidate_eligibility_justification: the explanation (from the answer) of why the candidate with a bachelor's degree and ~4 years relevant experience qualifies for this certification. If no explicit justification is provided, null.
    - exam_cost: the exam cost information as stated in the answer:
        * member_price: member pricing (if applicable and provided; else null)
        * non_member_price: non-member pricing (if applicable and provided; else null)
        * other_fee_notes: any additional fee notes or price-structure description; if not provided, null
    - renewal:
        * renewal_requirements: the renewal/recertification requirements as stated in the answer (or a clear statement that renewal is not applicable); if the answer does not provide any renewal information, null

    Also extract PMI ATP guidance (if present anywhere in the answer):
    - pmi_atp_guidance_text: the sentence(s) that explain the PMI Authorized Training Partner (ATP) path to fulfill required PM education/contact hours. If not present, null.
    - pmi_atp_urls: official PMI URL(s) for locating PMI Authorized Training Partners or the official ATP directory (must be on pmi.org). If the answer does not include such a URL, return an empty list.

    IMPORTANT RULES:
    - Do not invent any information; only extract what appears in the answer.
    - For URL extraction, only include explicit URLs present in the answer.
    - Keep all fields as strings when possible; do not normalize numbers or currencies; keep them as written.
    - If a field is missing, set it to null (or [] for lists).

    Return a JSON object matching the specified schema.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_issuer(issuer_text: Optional[str]) -> Optional[str]:
    if not issuer_text:
        return None
    low = issuer_text.strip().lower()
    for norm, variants in ALLOWED_ISSUERS.items():
        for v in variants:
            if v in low:
                return norm.upper()
    return None


def official_domains_for(norm_issuer: Optional[str]) -> List[str]:
    if not norm_issuer:
        return []
    return OFFICIAL_DOMAINS.get(norm_issuer, [])


def url_is_official_for_issuer(url: str, norm_issuer: Optional[str]) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    if not norm_issuer:
        return False
    for d in official_domains_for(norm_issuer):
        if host == d or host.endswith(f".{d}"):
            return True
    return False


def any_official_url(issuer_norm: Optional[str], urls: List[str]) -> bool:
    if not urls:
        return False
    for u in urls:
        if url_is_official_for_issuer(u, issuer_norm):
            return True
    return False


def hours_required_from_detail(detail: Optional[str], explicit_flag: Optional[bool]) -> Optional[bool]:
    """
    Determine if hours are required based on detail text and explicit boolean.
    """
    if explicit_flag is not None:
        return explicit_flag
    if not detail:
        return None
    low = detail.lower()
    if "none" in low or "not required" in low or "no " in low and "hour" in low:
        return False
    # Look for numbers + hour
    if re.search(r"\b\d+\s*(?:hour|hr|hrs|contact hour)s?\b", low):
        return True
    if "hour" in low or "contact hour" in low:
        return True
    return None


def compose_eligibility_claim(item: CertificationExtractedItem, norm_issuer: Optional[str]) -> str:
    edu = item.eligibility.education if item.eligibility else None
    exp = item.eligibility.experience if item.eligibility else None
    th_req = item.eligibility.training_hours_required if item.eligibility else None
    th_det = item.eligibility.training_hours_detail if item.eligibility else None
    th_phrase = "none required" if (th_req is False or (th_det and th_det.strip().lower() == "none")) else (th_det or "unspecified")
    issuer_name = norm_issuer or (item.issuer or "the certifying organization")
    return (
        f"According to the official {issuer_name} page(s), the eligibility for the {item.name} certification includes: "
        f"education requirement: {edu if edu else 'unspecified'}; "
        f"experience requirement: {exp if exp else 'unspecified'}; "
        f"training/contact education hours: {th_phrase}."
    )


def compose_cost_claim(item: CertificationExtractedItem) -> str:
    if not item.exam_cost:
        return f"The current exam cost for {item.name} is unspecified."
    mem = item.exam_cost.member_price
    non = item.exam_cost.non_member_price
    notes = item.exam_cost.other_fee_notes
    pieces = []
    if mem:
        pieces.append(f"member pricing: {mem}")
    if non:
        pieces.append(f"non-member pricing: {non}")
    if notes:
        pieces.append(f"additional notes: {notes}")
    if not pieces:
        return f"The current exam cost for {item.name} is unspecified."
    return f"The current exam cost for {item.name} is: " + "; ".join(pieces) + "."


def compose_renewal_claim(item: CertificationExtractedItem) -> str:
    ren = item.renewal.renewal_requirements if item.renewal else None
    if ren:
        return f"The renewal/recertification requirement for {item.name} is: {ren}"
    return f"The renewal/recertification requirement for {item.name} is unspecified."


def atp_needed_any(cert_items: List[CertificationExtractedItem]) -> bool:
    for it in cert_items:
        norm = normalize_issuer(it.issuer)
        if norm == "PMI":
            th_req = hours_required_from_detail(
                it.eligibility.training_hours_detail if it.eligibility else None,
                it.eligibility.training_hours_required if it.eligibility else None
            )
            if th_req is True:
                return True
    return False


# --------------------------------------------------------------------------- #
# Verification logic for a single certification item                          #
# --------------------------------------------------------------------------- #
async def verify_cert_item(
    evaluator: Evaluator,
    parent_node,
    item: CertificationExtractedItem,
    idx: int,
) -> None:
    # Create a container node for this certification item
    item_node = evaluator.add_parallel(
        id=f"CertificationItem{idx+1}",
        desc=f"Certification #{idx+1}: {item.name or 'Unnamed'} (if provided).",
        parent=parent_node,
        critical=False,  # Each item provides partial credit independently
    )

    # Normalize issuer
    issuer_norm = normalize_issuer(item.issuer)

    # 1) CertNameAndIssuer (Critical)
    has_name = bool(item.name and item.name.strip())
    issuer_ok = issuer_norm in ("PMI", "HRCI", "SHRM")
    evaluator.add_custom_node(
        result=(has_name and issuer_ok),
        id=f"CertificationItem{idx+1}_CertNameAndIssuer",
        desc="States the certification name and clearly identifies the certifying organization; organization must be PMI, HRCI, or SHRM.",
        parent=item_node,
        critical=True,
    )

    # 6) OfficialReferenceURL presence/officialness (Critical)
    has_official_ref = any_official_url(issuer_norm, item.official_urls)
    evaluator.add_custom_node(
        result=has_official_ref,
        id=f"CertificationItem{idx+1}_OfficialReferenceURL",
        desc="Provides an official reference URL from the certifying organization (PMI/HRCI/SHRM) supporting the stated information.",
        parent=item_node,
        critical=True,
    )

    # 2) EligibilityRequirements (Critical) -> verify against official URL(s)
    elig_leaf = evaluator.add_leaf(
        id=f"CertificationItem{idx+1}_EligibilityRequirements",
        desc="Provides the certification's eligibility requirements, including education, experience, and any required training/contact/education hours (or explicitly states if none are required).",
        parent=item_node,
        critical=True,
    )
    elig_claim = compose_eligibility_claim(item, issuer_norm)
    await evaluator.verify(
        claim=elig_claim,
        node=elig_leaf,
        sources=item.official_urls if item.official_urls else None,
        additional_instruction=(
            "Verify that the official page(s) explicitly support the stated eligibility details: education, experience, "
            "and training/contact hours (or explicitly 'none required'). Allow equivalent wording."
        ),
    )

    # 3) CandidateEligibilityJustification (Critical) -> verify from answer itself
    just_leaf = evaluator.add_leaf(
        id=f"CertificationItem{idx+1}_CandidateEligibilityJustification",
        desc="Explains why the given candidate (bachelor’s degree + ~4 years relevant experience) is eligible.",
        parent=item_node,
        critical=True,
    )
    justification_text = item.candidate_eligibility_justification or ""
    just_claim = (
        f"The answer includes a clear explanation of why a candidate with a bachelor's degree and about 4 years of "
        f"relevant experience meets the eligibility requirements for {item.name}. "
        f"Explanation excerpt from the answer: {justification_text}"
    )
    await evaluator.verify(
        claim=just_claim,
        node=just_leaf,
        additional_instruction=(
            "Judge solely based on the provided answer text whether it explains eligibility with respect to degree and ~4 years experience. "
            "Minor paraphrasing is acceptable; the reasoning must be explicit and relevant to the stated eligibility."
        ),
    )

    # 4) ExamCost (Critical) -> verify against official URL(s)
    cost_leaf = evaluator.add_leaf(
        id=f"CertificationItem{idx+1}_ExamCost",
        desc="Provides current exam cost; includes both member and non-member pricing where applicable (or states when only one pricing applies).",
        parent=item_node,
        critical=True,
    )
    cost_claim = compose_cost_claim(item)
    await evaluator.verify(
        claim=cost_claim,
        node=cost_leaf,
        sources=item.official_urls if item.official_urls else None,
        additional_instruction=(
            "Check the official page(s) to verify the stated exam fee(s). If both member and non-member fees exist, ensure both are accurate; "
            "if only a single price structure applies on the official page, it's acceptable if the claim reflects a single price."
        ),
    )

    # 5) RenewalOrRecertification (Critical) -> verify against official URL(s)
    renewal_leaf = evaluator.add_leaf(
        id=f"CertificationItem{idx+1}_RenewalOrRecertification",
        desc="Provides renewal/recertification requirements if applicable, or explicitly states that renewal/recertification is not applicable/required.",
        parent=item_node,
        critical=True,
    )
    renewal_claim = compose_renewal_claim(item)
    await evaluator.verify(
        claim=renewal_claim,
        node=renewal_leaf,
        sources=item.official_urls if item.official_urls else None,
        additional_instruction=(
            "Verify renewal/recertification requirements on the official page(s). Allow equivalent phrasing and reasonable summarization."
        ),
    )


# --------------------------------------------------------------------------- #
# Verification logic for PMI ATP requirement (conditional)                    #
# --------------------------------------------------------------------------- #
async def verify_pmi_atp_requirement(
    evaluator: Evaluator,
    parent_node,
    extracted: CertificationsExtraction,
) -> None:
    """
    If any PMI certification requires formal project management education/contact hours,
    the answer must (a) explain ATP as a path to fulfill this requirement, and
    (b) provide an official PMI URL for finding/locating Authorized Training Partners.
    """
    needs_atp = atp_needed_any(extracted.certifications)

    # If ATP not applicable, mark as passed via custom node and return
    if not needs_atp:
        evaluator.add_custom_node(
            result=True,
            id="PMI_ATP_Requirement",
            desc="No PMI certification with formal PM education/contact hours was included; ATP guidance not applicable.",
            parent=parent_node,
            critical=True,
        )
        return

    # Otherwise, create a critical parallel container and verify presence + official link
    atp_node = evaluator.add_parallel(
        id="PMI_ATP_Requirement",
        desc="PMI ATP guidance and official PMI resource provided when PMI cert requires formal PM education hours.",
        parent=parent_node,
        critical=True,
    )

    # Existence of guidance text and at least one PMI official URL
    has_text = bool(extracted.pmi_atp_guidance_text and extracted.pmi_atp_guidance_text.strip())
    has_official_link = any_official_url("PMI", extracted.pmi_atp_urls)
    evaluator.add_custom_node(
        result=(has_text and has_official_link),
        id="PMI_ATP_Requirement_Provided",
        desc="Explains fulfilling hours via PMI Authorized Training Partners and provides at least one official PMI ATP URL.",
        parent=atp_node,
        critical=True,
    )

    # Verify that the provided URL(s) are official PMI resource(s) for finding/locating ATPs
    atp_url_leaf = evaluator.add_leaf(
        id="PMI_ATP_Requirement_OfficialURL",
        desc="Provided URL is an official PMI page for Authorized Training Partners (ATP) or directory.",
        parent=atp_node,
        critical=True,
    )
    atp_claim = (
        "This page is an official PMI resource for Authorized Training Partners (ATP) and helps users learn about or find PMI ATP providers."
    )
    await evaluator.verify(
        claim=atp_claim,
        node=atp_url_leaf,
        sources=extracted.pmi_atp_urls if extracted.pmi_atp_urls else None,
        additional_instruction=(
            "Accept if the URL is on pmi.org and the page is clearly about PMI Authorized Training Partners (ATP), "
            "including pages that let users find, browse, or learn about authorized training providers."
        ),
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the PMI/HRCI/SHRM certification eligibility & details task.
    """
    evaluator = Evaluator()
    # IMPORTANT: The root in rubric was marked critical; however, the framework requires all children
    # of a critical parent to also be critical. Because we have non-critical children (optional items),
    # we set the root as non-critical here to respect framework constraints while still enforcing
    # critical checks on the relevant sub-nodes (e.g., PMI_ATP requirement).
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

    # 1) Extract structured information from the answer
    extracted: CertificationsExtraction = await evaluator.extract(
        prompt=prompt_extract_certifications(),
        template_class=CertificationsExtraction,
        extraction_name="certifications_extraction",
    )

    # 2) Build "CertificationAnalysis" container (parallel)
    analysis_node = evaluator.add_parallel(
        id="CertificationAnalysis",
        desc="Evaluate identified eligible PMI and/or HRCI/SHRM certifications and required details.",
        parent=root,
        critical=False,  # Root of analysis allows partial credit across items
    )

    # 3) Verify each identified certification (up to first 5)
    max_items = 5
    items = extracted.certifications[:max_items] if extracted and extracted.certifications else []
    for i, item in enumerate(items):
        await verify_cert_item(evaluator, analysis_node, item, i)

    # 4) PMI ATP requirement (critical when applicable)
    await verify_pmi_atp_requirement(evaluator, analysis_node, extracted)

    # 5) Return structured summary
    return evaluator.get_summary()