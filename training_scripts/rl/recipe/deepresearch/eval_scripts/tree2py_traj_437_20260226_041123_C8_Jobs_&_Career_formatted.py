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
TASK_ID = "experience_ce_certifications_4_fields"
TASK_DESCRIPTION = (
    "I am researching professional certifications to advance my career and want to understand which established "
    "certifications require significant professional experience and ongoing commitment to continuing education. "
    "Please identify four professional certifications from four different career fields (e.g., project management, "
    "cybersecurity, education, finance, healthcare, human resources) that each meet ALL of the following criteria:\n\n"
    "1. The certification requires a minimum of 3 years of professional work experience in the relevant field "
    "(or the equivalent in hours, typically 5,000-6,000 hours for full-time work)\n"
    "2. The certification requires mandatory continuing education or recertification, with a specified number of credits, "
    "hours, or Professional Development Units (PDUs) that must be completed within a defined time period to maintain the certification\n"
    "3. The experience requirements and renewal/recertification requirements are clearly documented on the official certification body's "
    "website or authoritative professional organization\n\n"
    "For each certification, please provide:\n"
    "- The full name and common abbreviation of the certification\n"
    "- The specific experience requirement (years or hours)\n"
    "- The specific renewal/recertification requirement (number of credits/hours and time period)\n"
    "- A reference URL to the official source documenting these requirements"
)

MAX_CERTS = 4


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Certification(BaseModel):
    full_name: Optional[str] = None
    abbreviation: Optional[str] = None
    career_field: Optional[str] = None
    experience_requirement: Optional[str] = None  # free text from answer, e.g., "3 years of experience"
    renewal_requirement: Optional[str] = None     # free text from answer, e.g., "60 PDUs every 3 years"
    documentation_url: Optional[str] = None       # official/reference URL


class CertificationsExtraction(BaseModel):
    certifications: List[Certification] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_certifications() -> str:
    return (
        "Extract up to four professional certifications listed in the answer that the author claims meet the criteria. "
        "For each certification, extract:\n"
        "1) full_name: The full official certification name as stated in the answer\n"
        "2) abbreviation: The common abbreviation (e.g., PMP, CISSP); if none is explicitly stated, return null\n"
        "3) career_field: The clear career field for this certification (e.g., project management, cybersecurity, education, "
        "finance, healthcare, human resources). Use the wording from the answer if present; otherwise, infer the most likely concise field label from the answer's context.\n"
        "4) experience_requirement: The specific experience requirement in quantifiable terms if provided (e.g., '3 years', '36 months', '6000 hours'). "
        "If not provided in the answer, return null.\n"
        "5) renewal_requirement: The specific continuing education or recertification requirement (include both a quantity and a time period if provided, e.g., "
        "'60 PDUs every 3 years', '120 CPE hours over 3 years'). If not provided in the answer, return null.\n"
        "6) documentation_url: A single reference URL (preferably the official certification body or a recognized professional organization) that the answer cites "
        "for the requirements. If multiple URLs are given, pick the one that appears most official/authoritative. If no URL is present, return null.\n\n"
        "Return a JSON object with a key 'certifications' that is an array of up to 4 certification objects with the above fields.\n"
        "Do not invent details not present in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_number(text: Optional[str]) -> bool:
    if not text:
        return False
    return re.search(r"\d", text) is not None


def _contains_any(text: Optional[str], keywords: List[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def has_quantified_experience(spec: Optional[str]) -> bool:
    # Quantifiable if it contains a number and mentions years/months/hours
    if not spec:
        return False
    if not _has_number(spec):
        return False
    units = ["year", "years", "yr", "yrs", "month", "months", "hour", "hours", "hr", "hrs"]
    return _contains_any(spec, units)


def has_quantified_renewal(spec: Optional[str]) -> bool:
    # Quantifiable if contains a number AND both a unit (credits/hours/PDUs/CEUs/CPE) and a time period (years/months)
    if not spec:
        return False
    if not _has_number(spec):
        return False
    units = ["pdu", "pdus", "ceu", "ceus", "cpe", "cpes", "credit", "credits", "hour", "hours", "unit", "units", "point", "points"]
    periods = ["year", "years", "month", "months"]
    return _contains_any(spec, units) and _contains_any(spec, periods)


def field_distinct_instruction(prev_fields: List[str]) -> str:
    if not prev_fields:
        return "There are no prior fields to compare for distinctness."
    # Provide guidance for synonyms
    synonyms_hint = (
        "Treat obvious synonyms as the same field (e.g., 'IT security' ~ 'cybersecurity'; "
        "'finance' ~ 'accounting' when context is licensing like CPA; 'HR' ~ 'human resources'). "
        "Do not be overly strict on wording; focus on whether the professional domain is materially the same."
    )
    return (
        f"The previously used fields are: {prev_fields}. "
        f"Confirm this certification's field is clearly identifiable as stated and is distinct from all previous fields. "
        f"{synonyms_hint}"
    )


# --------------------------------------------------------------------------- #
# Verification builder for one certification                                  #
# --------------------------------------------------------------------------- #
async def verify_one_certification(
    evaluator: Evaluator,
    parent_node,
    cert: Certification,
    idx: int,
    prev_fields: List[str],
) -> None:
    """
    Build verification nodes for a single certification as per rubric.
    Each leaf node is a single binary check.
    """
    # Create a container node for this certification (parallel; non-critical)
    cert_node = evaluator.add_parallel(
        id=f"certification_{idx+1}",
        desc=(
            "Professional certification meeting all criteria "
            f"(item #{idx+1})"
        ),
        parent=parent_node,
        critical=False,
    )

    # Leaf 1: Identification provided (full name and abbreviation present)
    ident_ok = bool(cert.full_name and cert.full_name.strip()) and bool(cert.abbreviation and cert.abbreviation.strip())
    evaluator.add_custom_node(
        result=ident_ok,
        id=f"cert{idx+1}_identification",
        desc="The full name and common abbreviation of the certification are provided",
        parent=cert_node,
        critical=True,
    )

    # Leaf 2: Career field claimed and distinct vs prior (verified against official page)
    career_field_leaf = evaluator.add_leaf(
        id=f"cert{idx+1}_career_field",
        desc=(
            "The certification belongs to a clearly identifiable career field"
            + ("" if idx == 0 else " that is different from earlier certifications")
        ),
        parent=cert_node,
        critical=True,
    )
    cf = cert.career_field or ""
    name_for_claim = cert.full_name or (cert.abbreviation or "this certification")
    cf_claim = (
        f"The webpage for {name_for_claim} indicates that it is a certification in the '{cf}' career field. "
        + ("" if idx == 0 else "Also confirm that this field is distinct from the previously used fields.")
    )
    await evaluator.verify(
        claim=cf_claim,
        node=career_field_leaf,
        sources=cert.documentation_url,
        additional_instruction=(
            "Use the page content to infer the professional field (e.g., project management, cybersecurity, education, finance, "
            "healthcare, human resources). If synonyms indicate the same field, treat them as the same field. "
            + field_distinct_instruction(prev_fields)
        ),
    )

    # Leaf 3: Experience requirement meets minimum 3 years (or ~5000-6000 hours)
    exp_req_leaf = evaluator.add_leaf(
        id=f"cert{idx+1}_experience_requirement",
        desc=(
            "The certification requires a minimum of 3 years of professional work experience "
            "(or equivalent hours roughly 5,000–6,000 hours)"
        ),
        parent=cert_node,
        critical=True,
    )
    exp_claim = (
        f"The official page for {name_for_claim} states that the certification requires at least 3 years of professional experience "
        "OR an equivalent number of hours at approximately 5,000 to 6,000 hours (e.g., ~2,000 hours/year). "
        "If the page states a higher minimum, it still satisfies this requirement."
    )
    await evaluator.verify(
        claim=exp_claim,
        node=exp_req_leaf,
        sources=cert.documentation_url,
        additional_instruction=(
            "Accept equivalent phrasing such as '36 months', 'X years', 'Y hours', or similar eligibility/experience statements. "
            "If experience depends on education level, consider whether at least one allowed path requires ≥3 years (or ≥~5000 hours)."
        ),
    )

    # Leaf 4: Experience specifics provided (quantifiable terms in the answer)
    evaluator.add_custom_node(
        result=has_quantified_experience(cert.experience_requirement),
        id=f"cert{idx+1}_experience_specifics",
        desc="The specific experience requirement is provided in quantifiable terms (either years or hours)",
        parent=cert_node,
        critical=True,
    )

    # Leaf 5: Renewal/CE requirement exists (must specify quantity & time period on page)
    renewal_req_leaf = evaluator.add_leaf(
        id=f"cert{idx+1}_renewal_requirement",
        desc=(
            "The certification requires mandatory continuing education or recertification with a specified number of credits/hours "
            "within a defined time period"
        ),
        parent=cert_node,
        critical=True,
    )
    renewal_claim = (
        f"The official page for {name_for_claim} explicitly states that maintaining the certification requires mandatory continuing "
        "education or recertification, and specifies both a quantity (e.g., hours/credits/PDUs) and a time period (e.g., every N years)."
    )
    await evaluator.verify(
        claim=renewal_claim,
        node=renewal_req_leaf,
        sources=cert.documentation_url,
        additional_instruction=(
            "Look for terms like PDUs, CEUs, CPEs, credits, or hours, and a defined cycle such as 'every 3 years' or 'annually'. "
            "The requirement must be mandatory for maintenance (not just recommended)."
        ),
    )

    # Leaf 6: Renewal specifics provided (quantifiable terms in the answer)
    evaluator.add_custom_node(
        result=has_quantified_renewal(cert.renewal_requirement),
        id=f"cert{idx+1}_renewal_specifics",
        desc="The specific renewal requirement is provided including both the number of credits/hours/PDUs and the time period",
        parent=cert_node,
        critical=True,
    )

    # Leaf 7: Documentation URL provided (existence check per rubric wording)
    evaluator.add_custom_node(
        result=bool(cert.documentation_url and cert.documentation_url.strip()),
        id=f"cert{idx+1}_documentation_url",
        desc="A reference URL to an official source (certification body website or authoritative professional organization) documenting the requirements is provided",
        parent=cert_node,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict:
    """
    Entry point to evaluate the agent's answer for the professional certifications task.
    """
    # Initialize evaluator with a parallel root to aggregate four certifications
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # multiple certifications evaluated independently
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

    # NOTE: To satisfy framework constraints, we keep root as non-critical here.
    # The rubric's criticality is enforced at the leaf level for strict gating.

    # Extract certifications
    extracted = await evaluator.extract(
        prompt=prompt_extract_certifications(),
        template_class=CertificationsExtraction,
        extraction_name="certifications_extraction",
    )

    # Keep only the first 4 certifications, pad with empty ones if fewer
    certs: List[Certification] = list(extracted.certifications[:MAX_CERTS])
    while len(certs) < MAX_CERTS:
        certs.append(Certification())

    # Build verification tree nodes for each certification
    observed_fields: List[str] = []
    for i, cert in enumerate(certs):
        # Maintain a list of previously used fields for distinctness checks
        prev_fields = observed_fields.copy()
        await verify_one_certification(evaluator, root, cert, i, prev_fields)

        # Update observed fields list with the current certification's field (if present)
        if cert.career_field and cert.career_field.strip():
            observed_fields.append(cert.career_field.strip())

    # Return final structured evaluation summary
    return evaluator.get_summary()