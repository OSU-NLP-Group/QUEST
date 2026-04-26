import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "hr_cert_requirements"
TASK_DESCRIPTION = (
    "A human resources professional considering career advancement is evaluating three professional certifications: "
    "SHRM Certified Professional (SHRM-CP), SHRM Senior Certified Professional (SHRM-SCP), and Professional in Human Resources (PHR). "
    "For each of these three certifications, provide: (1) the complete work experience eligibility requirements, including any variations "
    "based on education level or other qualifying factors, (2) the recertification/renewal requirements, including the specific number of "
    "continuing education hours or credits needed and the renewal timeframe, and (3) an official source URL from the certifying organization "
    "that documents these requirements."
)


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------
class CertificationSection(BaseModel):
    # Eligibility info as explicitly stated in the answer (verbatim or close paraphrase)
    eligibility_summary: Optional[str] = None
    # Renewal info as explicitly stated in the answer (verbatim or close paraphrase)
    renewal_summary: Optional[str] = None
    # Extracted numbers/timeframe as free-form strings (keep loose to be robust)
    renewal_credits: Optional[str] = None
    renewal_timeframe: Optional[str] = None
    # All URLs cited for this certification in the answer (as-is)
    source_urls: List[str] = Field(default_factory=list)


class HRRequirementsExtraction(BaseModel):
    # One section per certification
    shrm_cp: Optional[CertificationSection] = None
    shrm_scp: Optional[CertificationSection] = None
    phr: Optional[CertificationSection] = None


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_hr_requirements() -> str:
    return """
    Extract, for each of the following certifications, exactly what the answer states about eligibility (especially work-experience
    requirements) and recertification/renewal requirements, and list all source URLs provided in the answer for that certification.

    Certifications to extract:
    - SHRM Certified Professional (SHRM-CP)
    - SHRM Senior Certified Professional (SHRM-SCP)
    - Professional in Human Resources (PHR)

    For each certification, extract these fields:
    - eligibility_summary: A concise summary (from the answer text) of the work-experience eligibility requirements. If the answer states there are no mandatory degree, job title, or prior HR-experience requirements (e.g., for SHRM-CP), capture that clearly. If there are different paths based on education or other qualifiers (e.g., for PHR), include those variations and years of required experience exactly as the answer gives them. Include mentions like “professional-level HR positions” or “strategic HR” if present in the answer.
    - renewal_summary: A concise summary (from the answer) of how renewal/recertification works (e.g., number of credits/hours and timeframe).
    - renewal_credits: The specific number of credits/hours mentioned for renewal (as a string, e.g., "60", "60 PDCs", etc.). If not specified, return null.
    - renewal_timeframe: The renewal timeframe as stated (as a string, e.g., "every 3 years", "within 3 years"). If not specified, return null.
    - source_urls: A list of all URLs that the answer explicitly cites for this certification’s requirements. Include only URLs actually present in the answer (plain or markdown). Prefer official certification body URLs if they are listed, but still include any other URLs that are present.

    Notes:
    - Do not invent or normalize numbers; extract exactly what's stated. Keep everything as strings where applicable.
    - If the answer does not provide some field for a certification, set it to null (or an empty list for URLs).
    - Ensure URLs are captured in full (prepend http:// if protocol missing).
    """


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def safe_urls(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


def nonempty(text: Optional[str]) -> str:
    return text or ""


# -----------------------------------------------------------------------------
# Verification builders
# -----------------------------------------------------------------------------
async def verify_global_professional_level_constraint(evaluator: Evaluator, parent) -> None:
    """
    Verify the global constraint:
    For every certification that has a work-experience requirement, the answer specifies that the experience must be
    in professional-level HR positions (or equivalent wording).
    """
    node = evaluator.add_leaf(
        id="experience_professional_level_constraint",
        desc="For every certification that has a work-experience requirement, the answer specifies that the experience must be in professional-level HR positions",
        parent=parent,
        critical=True
    )

    claim = (
        "In the provided answer, for every certification that requires work experience (i.e., SHRM-SCP and PHR), "
        "the answer explicitly states that the required experience must be in professional-level HR positions. "
        "Accept equivalent phrasings such as 'professional-level HR', 'HR professional role', 'professional HR experience', "
        "'strategic-level HR', or 'strategic HR role' as satisfying this requirement. "
        "The answer does not need to assert this for SHRM-CP (which has no mandatory prior HR experience requirement)."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=None,
        additional_instruction=(
            "Carefully read the answer text to confirm explicit mention of professional-level (or equivalent) for the "
            "certifications that require experience. If either SHRM-SCP or PHR is missing this explicit level qualifier, mark Incorrect."
        )
    )


async def verify_shrm_cp(evaluator: Evaluator, parent, section: Optional[CertificationSection]) -> None:
    cert_node = evaluator.add_parallel(
        id="SHRM_CP_certification",
        desc="Complete requirements documentation for SHRM Certified Professional (SHRM-CP)",
        parent=parent,
        critical=False
    )

    srcs = safe_urls(section.source_urls if section else None)
    elig_text = nonempty(section.eligibility_summary if section else None)
    ren_text = nonempty(section.renewal_summary if section else None)

    # Eligibility leaf
    elig_node = evaluator.add_leaf(
        id="SHRM_CP_eligibility",
        desc="Correctly states that SHRM-CP has no mandatory degree, HR title, or prior HR experience requirements for eligibility",
        parent=cert_node,
        critical=True
    )
    elig_claim = (
        "The answer explicitly states that SHRM-CP has no mandatory degree requirement, no mandatory HR job title requirement, "
        "and no prior HR experience requirement to be eligible to sit for the exam. "
        f"Answer's SHRM-CP eligibility summary: '{elig_text}'. "
        "This statement is supported by the provided official SHRM source page(s)."
    )
    await evaluator.verify(
        claim=elig_claim,
        node=elig_node,
        sources=srcs,
        additional_instruction=(
            "Confirm both that: (1) the answer contains the 'no mandatory degree/HR title/prior HR experience' assertion for SHRM-CP, "
            "and (2) at least one of the provided SHRM pages supports this assertion."
        ),
    )

    # Renewal leaf
    renewal_node = evaluator.add_leaf(
        id="SHRM_CP_renewal",
        desc="Correctly states that SHRM-CP renewal requires 60 professional development credits (PDCs) every 3 years",
        parent=cert_node,
        critical=True
    )
    renewal_claim = (
        "The answer states that SHRM-CP renewal requires 60 professional development credits (PDCs) every 3 years. "
        f"Answer's SHRM-CP renewal summary: '{ren_text}'. "
        "This renewal requirement (60 PDCs in a 3-year cycle) is supported by the provided official SHRM page(s)."
    )
    await evaluator.verify(
        claim=renewal_claim,
        node=renewal_node,
        sources=srcs,
        additional_instruction=(
            "Check that the answer explicitly mentions '60 PDCs' and 'every 3 years' (accept equivalent phrasing like 'within a 3-year cycle'), "
            "and confirm that an official SHRM source page corroborates this."
        ),
    )

    # Source leaf
    source_node = evaluator.add_leaf(
        id="SHRM_CP_source",
        desc="Provides an official SHRM website URL documenting SHRM-CP requirements",
        parent=cert_node,
        critical=True
    )
    source_claim = (
        "At least one of the provided URLs is an official SHRM website page (on a shrm.org domain) that documents SHRM-CP eligibility "
        "and/or recertification/renewal requirements."
    )
    await evaluator.verify(
        claim=source_claim,
        node=source_node,
        sources=srcs,
        additional_instruction=(
            "Verify that at least one URL is on shrm.org and that the page content discusses SHRM-CP eligibility or renewal/PDC requirements."
        ),
    )


async def verify_shrm_scp(evaluator: Evaluator, parent, section: Optional[CertificationSection]) -> None:
    cert_node = evaluator.add_parallel(
        id="SHRM_SCP_certification",
        desc="Complete requirements documentation for SHRM Senior Certified Professional (SHRM-SCP)",
        parent=parent,
        critical=False
    )

    srcs = safe_urls(section.source_urls if section else None)
    elig_text = nonempty(section.eligibility_summary if section else None)
    ren_text = nonempty(section.renewal_summary if section else None)

    # Eligibility leaf
    elig_node = evaluator.add_leaf(
        id="SHRM_SCP_eligibility",
        desc="Correctly states SHRM-SCP eligibility requirements including the 3-year strategic HR work requirement with a minimum of 1,000 hours per calendar year, and the alternative pathway for SHRM-CP holders with 3+ years of certification who are in or transitioning to strategic roles",
        parent=cert_node,
        critical=True
    )
    elig_claim = (
        "The answer states that SHRM-SCP eligibility includes: (a) a 3-year strategic HR work requirement with a minimum of 1,000 hours per calendar year, "
        "and (b) an alternative pathway for SHRM-CP holders with 3+ years of certification who are in or transitioning to strategic HR roles. "
        f"Answer's SHRM-SCP eligibility summary: '{elig_text}'. "
        "These details are supported by the provided official SHRM source page(s)."
    )
    await evaluator.verify(
        claim=elig_claim,
        node=elig_node,
        sources=srcs,
        additional_instruction=(
            "Confirm the answer explicitly includes BOTH the 3-years + 1,000-hours-per-year strategic HR requirement and the SHRM-CP alternative pathway, "
            "and verify that at least one official SHRM page corroborates these specific details."
        ),
    )

    # Renewal leaf
    renewal_node = evaluator.add_leaf(
        id="SHRM_SCP_renewal",
        desc="Correctly states that SHRM-SCP renewal requires 60 professional development credits (PDCs) every 3 years",
        parent=cert_node,
        critical=True
    )
    renewal_claim = (
        "The answer states that SHRM-SCP renewal requires 60 professional development credits (PDCs) every 3 years. "
        f"Answer's SHRM-SCP renewal summary: '{ren_text}'. "
        "This renewal requirement is supported by the provided official SHRM source page(s)."
    )
    await evaluator.verify(
        claim=renewal_claim,
        node=renewal_node,
        sources=srcs,
        additional_instruction=(
            "Check that the answer explicitly mentions '60 PDCs' and 'every 3 years' (accept equivalent phrasing like 'within a 3-year cycle'), "
            "and confirm that an official SHRM source corroborates this."
        ),
    )

    # Source leaf
    source_node = evaluator.add_leaf(
        id="SHRM_SCP_source",
        desc="Provides an official SHRM website URL documenting SHRM-SCP requirements",
        parent=cert_node,
        critical=True
    )
    source_claim = (
        "At least one of the provided URLs is an official SHRM website page (on a shrm.org domain) that documents SHRM-SCP eligibility "
        "and/or recertification/renewal requirements."
    )
    await evaluator.verify(
        claim=source_claim,
        node=source_node,
        sources=srcs,
        additional_instruction=(
            "Verify that at least one URL is on shrm.org and that the page content discusses SHRM-SCP eligibility or renewal/PDC requirements."
        ),
    )


async def verify_phr(evaluator: Evaluator, parent, section: Optional[CertificationSection]) -> None:
    cert_node = evaluator.add_parallel(
        id="PHR_certification",
        desc="Complete requirements documentation for Professional in Human Resources (PHR)",
        parent=parent,
        critical=False
    )

    srcs = safe_urls(section.source_urls if section else None)
    elig_text = nonempty(section.eligibility_summary if section else None)
    ren_text = nonempty(section.renewal_summary if section else None)

    # Eligibility leaf
    elig_node = evaluator.add_leaf(
        id="PHR_eligibility",
        desc="Correctly states all three PHR eligibility pathways based on education level: (1) 1 year experience with Master's degree or higher, (2) 2 years experience with Bachelor's degree, or (3) 4 years experience with no degree requirement",
        parent=cert_node,
        critical=True
    )
    elig_claim = (
        "The answer explicitly states all three PHR eligibility pathways based on education level: "
        "(1) at least 1 year of professional-level HR experience with a Master's degree or higher; "
        "(2) at least 2 years of professional-level HR experience with a Bachelor's degree; "
        "(3) at least 4 years of professional-level HR experience with no degree requirement. "
        f"Answer's PHR eligibility summary: '{elig_text}'. "
        "These pathways are supported by the provided official HRCI source page(s)."
    )
    await evaluator.verify(
        claim=elig_claim,
        node=elig_node,
        sources=srcs,
        additional_instruction=(
            "Confirm the answer includes all three pathways with the correct years. Accept equivalent phrasing, but years must match 1/2/4. "
            "Verify that an HRCI (hrci.org) official page corroborates these eligibility pathways and that they are described as professional-level HR experience."
        ),
    )

    # Renewal leaf
    renewal_node = evaluator.add_leaf(
        id="PHR_renewal",
        desc="Correctly states that PHR renewal requires 60 HR recertification credits every 3 years",
        parent=cert_node,
        critical=True
    )
    renewal_claim = (
        "The answer states that PHR renewal requires 60 HR recertification credits every 3 years. "
        f"Answer's PHR renewal summary: '{ren_text}'. "
        "This renewal requirement is supported by the provided official HRCI source page(s)."
    )
    await evaluator.verify(
        claim=renewal_claim,
        node=renewal_node,
        sources=srcs,
        additional_instruction=(
            "Check that the answer explicitly mentions '60 credits' and 'every 3 years' (accept equivalent phrasing like 'within a 3-year cycle'), "
            "and confirm that an official HRCI page corroborates this."
        ),
    )

    # Source leaf
    source_node = evaluator.add_leaf(
        id="PHR_source",
        desc="Provides an official HRCI website URL documenting PHR requirements",
        parent=cert_node,
        critical=True
    )
    source_claim = (
        "At least one of the provided URLs is an official HRCI website page (on an hrci.org domain) that documents PHR eligibility "
        "and/or recertification/renewal requirements."
    )
    await evaluator.verify(
        claim=source_claim,
        node=source_node,
        sources=srcs,
        additional_instruction=(
            "Verify that at least one URL is on hrci.org and that the page content discusses PHR eligibility or renewal/recertification requirements."
        ),
    )


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
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
    Evaluate an answer for HR certification eligibility/renewal requirements with official sources.
    """
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_hr_requirements(),
        template_class=HRRequirementsExtraction,
        extraction_name="hr_cert_requirements_extraction",
    )

    # Global constraint check (critical leaf)
    await verify_global_professional_level_constraint(evaluator, root)

    # SHRM-CP subtree
    await verify_shrm_cp(evaluator, root, extracted.shrm_cp if extracted else None)

    # SHRM-SCP subtree
    await verify_shrm_scp(evaluator, root, extracted.shrm_scp if extracted else None)

    # PHR subtree
    await verify_phr(evaluator, root, extracted.phr if extracted else None)

    return evaluator.get_summary()