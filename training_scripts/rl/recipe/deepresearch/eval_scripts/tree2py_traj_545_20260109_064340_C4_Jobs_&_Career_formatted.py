import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "single_it_cyber_cert_under_500"
TASK_DESCRIPTION = (
    "I am seeking to start a career in IT or cybersecurity but currently do not hold a bachelor's degree. "
    "I have a limited budget of $500 for a certification exam. Identify ONE professional certification that meets ALL of the following criteria:\n\n"
    "1. The certification is specifically focused on IT, cybersecurity, or information security\n"
    "2. No bachelor's degree is required to sit for the certification exam\n"
    "3. No prerequisite professional certifications are required before taking this exam\n"
    "4. The exam cost is $500 USD or less when purchased directly from the certification provider\n"
    "5. The certification has a formal renewal or continuing education requirement to maintain active status\n"
    "6. The certification is issued by a recognized professional organization or vendor that operates in the United States\n"
    "7. The certification has an official webpage from the issuing organization that clearly states the exam requirements and cost\n\n"
    "For the identified certification, provide:\n"
    "- The full official name of the certification (including the acronym in parentheses)\n"
    "- The name of the issuing organization\n"
    "- The exact current exam cost in USD\n"
    "- The specific renewal or continuing education requirement (including the time period)\n"
    "- The direct URL to the official certification page from the issuing organization"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CertificationItem(BaseModel):
    full_official_name: Optional[str] = None
    acronym: Optional[str] = None
    issuing_organization: Optional[str] = None
    exam_cost_usd: Optional[str] = None
    renewal_requirement: Optional[str] = None
    renewal_period: Optional[str] = None  # e.g., "every 3 years", "3-year cycle"
    official_url: Optional[str] = None


class CertificationsExtraction(BaseModel):
    certifications: List[CertificationItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_certifications() -> str:
    return (
        "Extract every professional certification mentioned in the answer that could be interpreted as a proposed option. "
        "Return them as a list named 'certifications'. For each certification, extract these fields strictly from the answer text:\n"
        "1) full_official_name: The full official certification name exactly as written, including the acronym in parentheses if present (e.g., 'Security+ (CompTIA Security+)').\n"
        "2) acronym: The acronym only (e.g., 'Security+', 'A+', 'CC'), if present; otherwise null.\n"
        "3) issuing_organization: The name of the issuing organization or vendor (e.g., 'CompTIA', 'EC-Council').\n"
        "4) exam_cost_usd: The exact current exam cost in USD as a text string (preserve symbols/format as provided, e.g., '$392', 'USD 399', '399 USD').\n"
        "5) renewal_requirement: The specific renewal or continuing education requirement text (e.g., 'earn 50 CEU every 3 years', 'renew every 3 years').\n"
        "6) renewal_period: The time period for renewal (e.g., 'every 3 years', '3-year cycle'). If not explicitly stated, set to null.\n"
        "7) official_url: The direct URL to the official certification page from the issuing organization (a single URL). If multiple URLs are shown, choose the most direct page about the certification exam.\n\n"
        "Rules:\n"
        "- Extract only what is explicitly present in the answer. Do not invent values.\n"
        "- If the answer mentions multiple certifications, include them all in the array. If none are mentioned, return an empty array.\n"
        "- If a field is missing for a certification, set it to null.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def select_primary_cert(extracted: CertificationsExtraction) -> CertificationItem:
    return extracted.certifications[0] if extracted.certifications else CertificationItem()


def has_acronym_in_name(name: Optional[str], explicit_acronym: Optional[str]) -> bool:
    if explicit_acronym and explicit_acronym.strip():
        # If acronym field is provided, accept
        return True
    if not name:
        return False
    # Check if name includes something in parentheses as an acronym-like token
    if "(" in name and ")" in name:
        inside = name.split("(", 1)[-1].split(")", 1)[0].strip()
        # Accept if there is at least one non-space char inside
        return len(inside) >= 1
    return False


def looks_like_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return bool(re.match(r"^https?://", url.strip()))


def nonempty_text(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def contains_usd_amount(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept if contains a digit; USD signifiers make it better
    return bool(re.search(r"\d", text))


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: CertificationsExtraction) -> None:
    root = evaluator.root

    # Single certification identified
    single_cert_node = evaluator.add_custom_node(
        result=(len(extracted.certifications) == 1),
        id="single_certification_identified",
        desc="Exactly ONE certification is identified (not multiple options)",
        parent=root,
        critical=True
    )

    # Prepare selected certification (first one if provided; still used even if multiple, but single_cert may fail)
    cert = select_primary_cert(extracted)

    # Required information provided (critical group)
    req_info_node = evaluator.add_parallel(
        id="required_information_provided",
        desc="All required output fields are provided for the identified certification",
        parent=root,
        critical=True
    )

    name_acronym_ok = evaluator.add_custom_node(
        result=(nonempty_text(cert.full_official_name) and has_acronym_in_name(cert.full_official_name, cert.acronym)),
        id="provide_full_official_name_and_acronym",
        desc="Provides the full official certification name including the acronym in parentheses",
        parent=req_info_node,
        critical=True
    )

    issuing_org_ok = evaluator.add_custom_node(
        result=nonempty_text(cert.issuing_organization),
        id="provide_issuing_organization_name",
        desc="Provides the name of the issuing organization",
        parent=req_info_node,
        critical=True
    )

    cost_text_ok = evaluator.add_custom_node(
        result=contains_usd_amount(cert.exam_cost_usd),
        id="provide_exact_exam_cost_usd",
        desc="Provides the exact current exam cost in USD",
        parent=req_info_node,
        critical=True
    )

    renewal_text_ok = evaluator.add_custom_node(
        result=(nonempty_text(cert.renewal_requirement) and nonempty_text(cert.renewal_period)),
        id="provide_specific_renewal_requirement_and_period",
        desc="Provides the specific renewal/continuing-education requirement including the time period",
        parent=req_info_node,
        critical=True
    )

    url_ok = evaluator.add_custom_node(
        result=looks_like_url(cert.official_url),
        id="provide_direct_official_url",
        desc="Provides the direct URL to the official certification page from the issuing organization",
        parent=req_info_node,
        critical=True
    )

    # Constraints group (critical)
    constraints_node = evaluator.add_parallel(
        id="certification_constraints_met",
        desc="The identified certification satisfies all stated eligibility/source constraints",
        parent=root,
        critical=True
    )

    # 1) IT/Cybersecurity focus
    focus_leaf = evaluator.add_leaf(
        id="it_cybersecurity_focus",
        desc="Certification is focused on IT, cybersecurity, or information security",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The certification '{cert.full_official_name or 'the certification'}' is focused on IT, cybersecurity, or information security.",
        node=focus_leaf,
        sources=cert.official_url,
        additional_instruction=(
            "Confirm from the official page that the certification pertains to information technology, cybersecurity, or information security. "
            "Accept if the certification clearly targets IT/cyber, security fundamentals, or entry-level security/IT roles."
        )
    )

    # 2) No bachelor's degree is required
    no_degree_leaf = evaluator.add_leaf(
        id="no_degree_requirement",
        desc="No bachelor's degree is required to sit for the exam",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"No bachelor's degree is required to sit for the '{cert.full_official_name or 'the certification'}' exam.",
        node=no_degree_leaf,
        sources=cert.official_url,
        additional_instruction=(
            "Check the eligibility/requirements section on this official page. "
            "If the page explicitly states 'no prerequisites' or does not impose a degree requirement for exam eligibility, "
            "consider this claim supported. If a bachelor's degree is explicitly required, the claim is false."
        )
    )

    # 3) No prerequisite certifications are required
    no_prereq_certs_leaf = evaluator.add_leaf(
        id="no_prerequisite_certifications",
        desc="No prerequisite professional certifications are required before taking the exam",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"No prerequisite professional certifications are required before taking the '{cert.full_official_name or 'the certification'}' exam.",
        node=no_prereq_certs_leaf,
        sources=cert.official_url,
        additional_instruction=(
            "Examine the eligibility or prerequisites section on this official page. "
            "If it says 'no prerequisites' or lists only recommended experience/training without requiring other certifications, "
            "consider the claim supported. If specific prior certifications are required, the claim is false."
        )
    )

    # 4) Exam cost is <= $500 when purchased directly from provider
    cost_within_budget_leaf = evaluator.add_leaf(
        id="cost_within_budget",
        desc="Exam cost is $500 USD or less when purchased directly from the certification provider",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="The official exam fee shown on this page is less than or equal to $500 (USD).",
        node=cost_within_budget_leaf,
        sources=cert.official_url,
        additional_instruction=(
            "Use only the official provider's exam price on this page (not training, bundles, or third-party vouchers). "
            "If the listed exam fee is $500 or less, mark supported; otherwise, not supported. "
            "If no exam fee is present on this page, mark not supported."
        )
    )

    # 5) Formal renewal/continuing education requirement exists
    renewal_exists_leaf = evaluator.add_leaf(
        id="renewal_requirement_exists",
        desc="Certification has a formal renewal/continuing-education requirement to maintain active status",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="This certification has a formal renewal or continuing education requirement with a defined time period (e.g., a multi-year cycle).",
        node=renewal_exists_leaf,
        sources=cert.official_url,
        additional_instruction=(
            "Confirm that the official page mentions renewal, continuing education (CE/CEU/CPE), or maintenance requirements, "
            "and indicates a time period (e.g., valid for 3 years, 3-year cycle). "
            "If none of these are present on this page, mark not supported."
        )
    )

    # 6) Issuer recognized and operates in the United States
    us_issuer_leaf = evaluator.add_leaf(
        id="us_recognized_issuer",
        desc="Issued by a recognized professional organization or vendor that operates in the United States",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The issuing organization '{cert.issuing_organization or 'the issuer'}' is a recognized professional organization or vendor that operates in the United States.",
        node=us_issuer_leaf,
        sources=cert.official_url,
        additional_instruction=(
            "Rely on indications from this official site such as a US address/contact, US presence, or other cues on the page "
            "(e.g., 'United States', 'US operations', or headquarters in the US). "
            "If the page provides no evidence of US operations, mark not supported."
        )
    )

    # 7) Official page states requirements AND cost
    official_page_both_leaf = evaluator.add_leaf(
        id="official_webpage_states_requirements_and_cost",
        desc="There is an official issuer webpage that clearly states exam requirements and exam cost",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="This single official certification page clearly states both the exam eligibility/requirements and the exam cost.",
        node=official_page_both_leaf,
        sources=cert.official_url,
        additional_instruction=(
            "Both pieces of information (exam eligibility/requirements and exam cost) must appear on this same URL. "
            "If one or both are missing here (or are only available on other pages), mark not supported."
        )
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
    evaluator.initialize(
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

    # Extract certification(s) mentioned in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_certifications(),
        template_class=CertificationsExtraction,
        extraction_name="certifications_extraction"
    )

    # Optionally record additional info for debugging/analysis
    evaluator.add_custom_info(
        {"num_certifications_in_answer": len(extracted.certifications)},
        info_type="extraction_meta",
        info_name="cert_item_count"
    )

    # Build tree and run verifications
    await build_and_verify_tree(evaluator, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()