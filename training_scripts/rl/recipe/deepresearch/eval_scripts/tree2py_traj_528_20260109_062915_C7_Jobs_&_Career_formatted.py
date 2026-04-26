import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "entry_level_it_cert_2026"
TASK_DESCRIPTION = (
    "Find one entry-level IT certification that meets ALL of the following criteria:\n\n"
    "1. Offered by a major certification provider (CompTIA, Cisco, Microsoft, AWS, Google, GIAC, or PMI)\n"
    "2. Explicitly described as entry-level, foundational, or suitable for beginners\n"
    "3. Has an exam cost under $300 USD per exam\n"
    "4. Has no formal prerequisites for taking the exam\n"
    "5. Requires only ONE exam to obtain the certification\n"
    "6. Has a specific, stated exam cost (not just 'varies' or 'contact for pricing')\n"
    "7. Specifies recommended prior experience or preparation time\n"
    "8. Lists potential job roles or positions associated with the certification\n"
    "9. Describes the next-level certifications or career progression path\n"
    "10. Clearly indicates whether it is vendor-neutral or vendor-specific\n"
    "11. Is currently offered and relevant in 2026\n"
    "12. Clearly states its primary technology focus or domain\n"
    "13. Can be verified from an official certification provider website or reputable educational platform\n\n"
    "Provide the following information:\n"
    "- Certification name\n"
    "- Provider\n"
    "- Exam cost\n"
    "- A reference URL where this information can be verified\n"
    "- Brief confirmation of how each criterion is met"
)

MAJOR_PROVIDERS = {
    "comptia",
    "cisco",
    "microsoft",
    "aws",
    "amazon web services",
    "google",
    "giac",
    "pmi",
    "project management institute",
}


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class CertificationExtraction(BaseModel):
    cert_name: Optional[str] = None
    provider: Optional[str] = None
    exam_cost_text: Optional[str] = None
    exam_cost_currency: Optional[str] = None
    exam_cost_amount_usd: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)

    entry_level_phrase: Optional[str] = None
    prerequisites_text: Optional[str] = None
    exam_count_text: Optional[str] = None
    vendor_status_text: Optional[str] = None

    recommended_experience: Optional[str] = None
    prep_time_text: Optional[str] = None
    job_roles: List[str] = Field(default_factory=list)
    career_path_text: Optional[str] = None
    technology_focus: Optional[str] = None
    current_status_text: Optional[str] = None

    criteria_confirmation_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_certification() -> str:
    return (
        "Extract exactly the following fields from the answer.\n"
        "Only use information explicitly present in the answer text.\n"
        "Return null for any missing field.\n\n"
        "Fields to extract:\n"
        "- cert_name: The certification name as stated\n"
        "- provider: The certification provider (e.g., CompTIA, Cisco, Microsoft, AWS, Google, GIAC, PMI)\n"
        "- exam_cost_text: The stated exam cost text (include currency and number if present)\n"
        "- exam_cost_currency: Currency code or name, if stated (e.g., USD)\n"
        "- exam_cost_amount_usd: The numeric amount in USD if the answer states it; otherwise null (keep as string)\n"
        "- reference_urls: All URLs provided in the answer as verification references (array)\n"
        "- entry_level_phrase: The phrase indicating entry-level/foundational/beginner suitability\n"
        "- prerequisites_text: Text indicating prerequisites (if states 'no prerequisites', include that)\n"
        "- exam_count_text: Text describing how many exams are required\n"
        "- vendor_status_text: Text indicating vendor-neutral or vendor-specific, if explicitly mentioned\n"
        "- recommended_experience: Any stated recommended prior experience (text)\n"
        "- prep_time_text: Any stated recommended preparation time (text, e.g., '3–6 months')\n"
        "- job_roles: List of job roles/positions mentioned (array)\n"
        "- career_path_text: Text describing next-level certifications or career progression\n"
        "- technology_focus: Main technology domain/focus (e.g., cybersecurity, networking, cloud)\n"
        "- current_status_text: Text indicating currently offered/relevant (e.g., not retired)\n"
        "- criteria_confirmation_text: Brief confirmations for how each criterion is met if the answer provided a section summarizing these\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_provider(p: Optional[str]) -> str:
    return (p or "").strip().lower()


def _is_major_provider(p: Optional[str]) -> bool:
    prov = _normalize_provider(p)
    return any(
        prov == mp or prov in mp for mp in MAJOR_PROVIDERS
    )


def _has_specific_amount(text: Optional[str]) -> bool:
    if not text:
        return False
    # Look for explicit dollar amounts or USD numbers
    patterns = [
        r"\$\s*\d+(?:\.\d{2})?",             # $123 or $123.45
        r"\bUSD\s*\d+(?:\.\d{2})?\b",        # USD 123
        r"\bUS\s*dollars?\s*\d+(?:\.\d{2})?",# US dollars 123
        r"\b\d+\s*USD\b",                    # 123 USD
    ]
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False


def _infer_vendor_status(provider: Optional[str], vendor_status_text: Optional[str]) -> Tuple[str, str]:
    """
    Returns (status_label, claim_template).
    status_label is either 'vendor-neutral' or 'vendor-specific'.
    claim_template is the human-readable claim we will verify.
    """
    vst = (vendor_status_text or "").lower()
    prov = _normalize_provider(provider)

    if "vendor-neutral" in vst or "vendor neutral" in vst:
        return "vendor-neutral", "The certification is vendor-neutral (explicitly stated on the page)."
    if "vendor-specific" in vst or "vendor specific" in vst:
        return "vendor-specific", "The certification is vendor-specific to the provider."

    # If not explicitly stated in the answer, infer based on provider brand
    # Commonly vendor-neutral: CompTIA, PMI, GIAC (industry-neutral)
    neutral_providers = {"comptia", "pmi", "project management institute", "giac"}
    if prov in neutral_providers:
        return "vendor-neutral", "The certification is vendor-neutral (industry-neutral certification by a neutral body)."
    else:
        # For AWS, Microsoft, Google, Cisco – treat as vendor-specific
        return "vendor-specific", "The certification is vendor-specific to the provider (brand-owned technology)."


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_response_nodes(
    evaluator: Evaluator,
    parent_node,
    ext: CertificationExtraction
) -> Dict[str, Any]:
    """
    Build 'response_includes_required_information' subtree and return useful handles.
    """
    response_node = evaluator.add_parallel(
        id="response_includes_required_information",
        desc="Response includes all requested fields and confirms criteria satisfaction",
        parent=parent_node,
        critical=True
    )

    # Provides the certification name
    name_ok = bool(ext.cert_name and ext.cert_name.strip())
    name_node = evaluator.add_custom_node(
        result=name_ok,
        id="provides_certification_name",
        desc="Provides the certification name",
        parent=response_node,
        critical=True
    )

    # Provides the provider
    provider_ok = bool(ext.provider and ext.provider.strip())
    provider_node = evaluator.add_custom_node(
        result=provider_ok,
        id="provides_provider",
        desc="Provides the certification provider",
        parent=response_node,
        critical=True
    )

    # Provides exam cost as a specific stated amount
    cost_ok = bool(ext.exam_cost_text and ext.exam_cost_text.strip() and _has_specific_amount(ext.exam_cost_text))
    cost_node = evaluator.add_custom_node(
        result=cost_ok,
        id="provides_exam_cost",
        desc="Provides the exam cost as a specific stated amount",
        parent=response_node,
        critical=True
    )

    # Provides at least one reference URL
    urls_ok = bool(ext.reference_urls and len(ext.reference_urls) > 0)
    url_node = evaluator.add_custom_node(
        result=urls_ok,
        id="provides_reference_url",
        desc="Provides at least one reference URL where the information can be verified",
        parent=response_node,
        critical=True
    )

    # Brief confirmation of how each listed criterion is met (judge based on the answer content)
    brief_confirm_leaf = evaluator.add_leaf(
        id="brief_confirmation_each_criterion",
        desc="Provides brief confirmation of how each listed criterion is met",
        parent=response_node,
        critical=True
    )
    # Simple verification based on the answer content itself
    await evaluator.verify(
        claim=(
            "The answer includes brief confirmations or statements showing how each of the listed criteria "
            "(entry-level, cost under $300, no prerequisites, single exam, specific cost, recommended experience/prep time, "
            "job roles, career progression path, vendor status, current in 2026, technology focus, and verifiable source) is met."
        ),
        node=brief_confirm_leaf,
        additional_instruction=(
            "Check whether the answer text itself presents short confirmations (they can be inline or as bullet points). "
            "It does not need to be verbose; concise confirmations covering each criterion are sufficient."
        )
    )

    return {
        "response_node": response_node,
        "name_node": name_node,
        "provider_node": provider_node,
        "cost_node": cost_node,
        "url_node": url_node,
        "brief_confirm_leaf": brief_confirm_leaf,
    }


async def build_criteria_nodes(
    evaluator: Evaluator,
    parent_node,
    ext: CertificationExtraction,
    url_prereq_node
) -> None:
    """
    Build 'certification_meets_all_criteria' subtree and verify each criterion against provided sources.
    """
    criteria_node = evaluator.add_parallel(
        id="certification_meets_all_criteria",
        desc="Chosen certification satisfies every stated constraint",
        parent=parent_node,
        critical=True
    )

    # Provider major check (custom boolean based on extracted provider)
    provider_major_node = evaluator.add_custom_node(
        result=_is_major_provider(ext.provider),
        id="provider_major",
        desc="Certification is offered by CompTIA, Cisco, Microsoft, AWS, Google, GIAC, or PMI",
        parent=criteria_node,
        critical=True
    )

    # Prepare reusable bits
    cert_display = ext.cert_name or "the certification"
    provider_display = ext.provider or "the provider"
    urls = ext.reference_urls

    # Entry-level explicit
    entry_level_leaf = evaluator.add_leaf(
        id="entry_level",
        desc="Certification is explicitly described as entry-level, foundational, or suitable for beginners",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{cert_display} is explicitly described as entry-level, foundational, or suitable for beginners.",
        node=entry_level_leaf,
        sources=urls,
        extra_prerequisites=[url_prereq_node],
        additional_instruction="Accept synonyms like 'beginner', 'introductory', or 'foundational'. It must be explicit on the page."
    )

    # Cost under $300 USD per exam
    cost_under_300_leaf = evaluator.add_leaf(
        id="cost_under_300",
        desc="Exam cost is under $300 USD per exam",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The exam cost for {cert_display} is under $300 USD per exam (United States voucher pricing).",
        node=cost_under_300_leaf,
        sources=urls,
        extra_prerequisites=[url_prereq_node],
        additional_instruction=(
            "Check for US exam voucher price or equivalent. If multiple regions are shown, judge based on US price if available. "
            "If the page shows a specific amount such as $134, $200, etc., confirm it is under $300."
        )
    )

    # No formal prerequisites
    no_prereq_leaf = evaluator.add_leaf(
        id="no_formal_prerequisites",
        desc="Certification has no formal prerequisites for taking the exam",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"There are no formal prerequisites required to take the {cert_display} exam.",
        node=no_prereq_leaf,
        sources=urls,
        extra_prerequisites=[url_prereq_node],
        additional_instruction=(
            "Look for explicit 'no prerequisite' statements. Preparation recommendations do not count as prerequisites."
        )
    )

    # Single exam required
    single_exam_leaf = evaluator.add_leaf(
        id="single_exam",
        desc="Only one exam is required to obtain the certification",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{cert_display} requires only one exam to obtain the certification.",
        node=single_exam_leaf,
        sources=urls,
        extra_prerequisites=[url_prereq_node],
        additional_instruction="Confirm that the certification is earned by passing a single exam. Multiple-core exams would fail."
    )

    # Specific cost stated
    specific_cost_leaf = evaluator.add_leaf(
        id="specific_cost_stated",
        desc="Exam cost is stated as a specific dollar amount (not only 'varies'/'contact for pricing')",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim="The page explicitly states a specific exam cost amount (e.g., $134), not merely 'varies' or 'contact for pricing'.",
        node=specific_cost_leaf,
        sources=urls,
        extra_prerequisites=[url_prereq_node],
        additional_instruction="Look for a numeric price with currency (e.g., USD)."
    )

    # Recommended prior experience or prep time
    rec_exp_leaf = evaluator.add_leaf(
        id="recommended_experience_or_prep_time",
        desc="Materials specify recommended prior experience or preparation time",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim="The page specifies recommended prior experience or an estimated preparation time for this certification/exam.",
        node=rec_exp_leaf,
        sources=urls,
        extra_prerequisites=[url_prereq_node],
        additional_instruction="Accept statements like 'no prior experience needed but recommended familiarity', or 'typical prep time is X weeks/months'."
    )

    # Job roles listed
    job_roles_leaf = evaluator.add_leaf(
        id="job_roles_listed",
        desc="Potential job roles/positions associated with the certification are listed",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page lists potential job roles or positions associated with {cert_display}.",
        node=job_roles_leaf,
        sources=urls,
        extra_prerequisites=[url_prereq_node],
        additional_instruction="Accept a section that mentions roles such as help desk, junior analyst, IT support, etc."
    )

    # Career progression path described
    career_path_leaf = evaluator.add_leaf(
        id="career_progression_path",
        desc="Next-level certifications or career progression path is described",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page describes next-level certifications or a career progression path after {cert_display}.",
        node=career_path_leaf,
        sources=urls,
        extra_prerequisites=[url_prereq_node],
        additional_instruction="Accept 'what's next', 'advance to', or 'recommended next certifications' sections."
    )

    # Vendor status clear
    status_label, vendor_claim_template = _infer_vendor_status(ext.provider, ext.vendor_status_text)
    vendor_status_leaf = evaluator.add_leaf(
        id="vendor_status_clear",
        desc="Clearly indicates whether the certification is vendor-neutral or vendor-specific",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=vendor_claim_template,
        node=vendor_status_leaf,
        sources=urls,
        extra_prerequisites=[url_prereq_node],
        additional_instruction=(
            "If the page explicitly says 'vendor-neutral', accept that. "
            "Otherwise, if the certification is branded to a specific provider's technology (e.g., AWS, Microsoft, Cisco, Google), "
            "treat it as vendor-specific even if the term 'vendor-specific' is not explicitly stated."
        )
    )

    # Currently offered and relevant in 2026
    current_2026_leaf = evaluator.add_leaf(
        id="current_in_2026",
        desc="Certification is currently offered and relevant in 2026 (not retired/deprecated)",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"As of 2026, {cert_display} is currently offered and not retired or deprecated.",
        node=current_2026_leaf,
        sources=urls,
        extra_prerequisites=[url_prereq_node],
        additional_instruction=(
            "Check for signs of active offering: availability, registration/open exam, no 'retired/discontinued' notices. "
            "If the page indicates retirement, mark as not current."
        )
    )

    # Technology focus stated
    tech_focus_leaf = evaluator.add_leaf(
        id="technology_focus_stated",
        desc="Primary technology focus/domain is clearly stated",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page clearly states the primary technology focus/domain for {cert_display}.",
        node=tech_focus_leaf,
        sources=urls,
        extra_prerequisites=[url_prereq_node],
        additional_instruction="Accept domains like 'cloud', 'networking', 'cybersecurity', 'project management', etc., clearly indicated."
    )

    # Verifiable source type (official provider or reputable educational platform)
    verifiable_source_leaf = evaluator.add_leaf(
        id="verifiable_source_type",
        desc="Claims are verifiable from an official certification provider website or a reputable educational platform",
        parent=criteria_node,
        critical=True
    )
    await evaluator.verify(
        claim="This URL is either an official certification provider website or a reputable educational platform.",
        node=verifiable_source_leaf,
        sources=urls,
        extra_prerequisites=[url_prereq_node],
        additional_instruction=(
            "Accept official domains like comptia.org, cisco.com, microsoft.com (including learn/certifications), "
            "aws.amazon.com/training/certification, cloud.google.com/certification, giac.org, pmi.org; "
            "or reputable platforms like Coursera, edX, LinkedIn Learning, Pluralsight, Udacity. "
            "If the domain is a random blog or non-authoritative site, do not accept."
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
    """
    Evaluate an answer for the entry-level IT certification task and return a structured result dictionary.
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

    # Top-level critical task node under root (since initialize creates non-critical root)
    top_node = evaluator.add_parallel(
        id="entry_level_it_certification_task",
        desc="Answer identifies one entry-level IT certification meeting all criteria and provides the requested fields with verification",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    ext = await evaluator.extract(
        prompt=prompt_extract_certification(),
        template_class=CertificationExtraction,
        extraction_name="certification_extraction"
    )

    # Add helpful info for debugging
    evaluator.add_custom_info(
        info={"major_providers": sorted(list(MAJOR_PROVIDERS))},
        info_type="meta",
        info_name="major_providers_list"
    )

    # Build response nodes first to create the reference URL prerequisite leaf
    response_handles = await build_response_nodes(evaluator, top_node, ext)
    url_prereq_node = response_handles["url_node"]

    # Build and verify all criteria using provided sources
    await build_criteria_nodes(evaluator, top_node, ext, url_prereq_node)

    # Return summary
    return evaluator.get_summary()