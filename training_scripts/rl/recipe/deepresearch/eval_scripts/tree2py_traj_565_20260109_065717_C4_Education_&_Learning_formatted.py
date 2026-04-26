import asyncio
import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "one_professional_certificate"
TASK_DESCRIPTION = (
    "Find ONE online professional certificate program that meets ALL of the following requirements:\n"
    "- Completion Time: within 6 months or less when studying part-time\n"
    "- Weekly Commitment: 10 hours per week or less\n"
    "- Cost: Monthly cost $75 USD or less\n"
    "- Platform: Offered by Coursera, edX, Google Career Certificates, or eCornell\n"
    "- Credential: Awards a professional certificate or career certificate (not just course completion)\n"
    "- Practical Learning: Includes hands-on projects or practical assignments\n"
    "- Flexibility: 100% online and self-paced (no fixed class meeting times)\n"
    "- Field: Focuses on a professional career field (e.g., Data Analytics, Project Management, Digital Marketing, "
    "Business Analytics, Cybersecurity, or similar)\n"
    "Also provide: program name, platform, duration (months), weekly hours, monthly cost, and a direct official URL."
)

ALLOWED_PLATFORMS = [
    "Coursera",
    "edX",
    "Google Career Certificates",
    "eCornell",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProgramInfo(BaseModel):
    """Structured extraction of the single selected program from the answer."""
    program_name: Optional[str] = None
    platform: Optional[str] = None
    url: Optional[str] = None
    duration_months: Optional[str] = None  # keep as string for flexible formats (e.g., "3–6 months")
    weekly_hours: Optional[str] = None     # keep as string (e.g., "5–10 hours/week")
    monthly_cost_usd: Optional[str] = None # keep as string (e.g., "$59/month")
    certificate_type: Optional[str] = None # e.g., "Professional Certificate", "Career Certificate"
    includes_projects: Optional[str] = None  # e.g., "Yes, includes hands-on projects"
    online_self_paced: Optional[str] = None  # e.g., "Self-paced, 100% online"
    field: Optional[str] = None              # e.g., "Data Analytics", "Project Management"


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_program_info() -> str:
    return (
        "Extract exactly ONE program (the primary or first-mentioned program) from the answer, returning:\n"
        "1. program_name: The exact program name string\n"
        "2. platform: The platform offering it (e.g., Coursera, edX, Google Career Certificates, eCornell)\n"
        "3. url: A direct official URL to the program page (full URL, include http/https). If multiple URLs are given, "
        "   choose the official program page for the selected program.\n"
        "4. duration_months: The completion time expressed in months as stated in the answer (e.g., '3 months', '3–6 months')\n"
        "5. weekly_hours: The weekly time commitment as stated (e.g., '5–10 hours/week', '≤10 hours/week')\n"
        "6. monthly_cost_usd: The monthly cost or subscription cost as stated (e.g., '$59/month'). If only total cost is "
        "   given, still return it verbatim as provided.\n"
        "7. certificate_type: The credential awarded (e.g., 'Professional Certificate', 'Career Certificate').\n"
        "8. includes_projects: Whether the program includes hands-on projects/practical assignments (verbatim phrase).\n"
        "9. online_self_paced: Whether the program is 100% online and self-paced (verbatim phrase).\n"
        "10. field: The professional field focus (e.g., 'Data Analytics', 'Project Management', 'Cybersecurity').\n\n"
        "Rules:\n"
        "- Extract only what appears in the answer; do not invent.\n"
        "- If any field is missing in the answer, set it to null.\n"
        "- For URL extraction, return the actual full link explicitly present in the answer "
        "(plain URL or markdown link); do not infer.\n"
    )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_program_requirements(
    evaluator: Evaluator,
    parent_node,
    info: ProgramInfo,
) -> None:
    """
    Build and verify all requirements for the single selected program.
    All children are critical because the parent is critical and all requirements must be met.
    """
    # A single critical parent to gate all requirements
    req_node = evaluator.add_parallel(
        id="requirements",
        desc="All requirements must be met for the selected program (critical)",
        parent=parent_node,
        critical=True,
    )

    # Basic existence check (critical) to gate subsequent verifications
    required_info_ok = (
        (info is not None)
        and (info.program_name is not None and info.program_name.strip() != "")
        and (info.platform is not None and info.platform.strip() != "")
        and (info.url is not None and info.url.strip() != "")
    )
    evaluator.add_custom_node(
        result=required_info_ok,
        id="required_info_provided",
        desc="Program name, platform, and official program URL are provided in the answer",
        parent=req_node,
        critical=True,
    )

    # Reference URL is the official program page and contains the key verifiable information
    ref_url_leaf = evaluator.add_leaf(
        id="reference_url",
        desc="Provided URL is the official program page containing duration, weekly hours, cost, certificate info",
        parent=req_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "This URL is the program's official page where the duration, weekly hours, monthly cost, "
            "certificate/credential type, platform/provider, and delivery format (online, self-paced) are presented."
        ),
        node=ref_url_leaf,
        sources=info.url,
        additional_instruction=(
            "Confirm the page is the official program page (not generic marketing or third-party listing). "
            "It should show core facts: program details including duration, weekly commitment, cost/subscription price, "
            "credential/award type, platform/provider, and online/self-paced delivery. If the page lacks these details "
            "or is not official, mark as not supported."
        ),
    )

    # Platform requirement
    platform_leaf = evaluator.add_leaf(
        id="platform_requirement",
        desc="The program is offered by Coursera, edX, Google Career Certificates, or eCornell",
        parent=req_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "This program is offered by one of the following platforms: Coursera, edX, Google Career Certificates, or eCornell."
        ),
        node=platform_leaf,
        sources=info.url,
        additional_instruction=(
            "Accept reasonable branding variants. For Google Career Certificates, the program may be hosted on Coursera "
            "but still counts as Google Career Certificates. For eCornell, branding may show Cornell University/eCornell. "
            "Confirm the offering platform shown on the official page is in the allowed set."
        ),
    )

    # Duration requirement (<= 6 months, part-time)
    duration_leaf = evaluator.add_leaf(
        id="duration_requirement",
        desc="Completable within 6 months or less when studying part-time",
        parent=req_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "This program can be completed within 6 months or less when studying part-time."
        ),
        node=duration_leaf,
        sources=info.url,
        additional_instruction=(
            "Interpret phrases like '3 months', '4–6 months', 'approximately 6 months', or 'under 6 months' as compliant. "
            "If the page indicates longer than 6 months, mark as not supported. If completion time is unclear or missing, fail."
        ),
    )

    # Weekly hours requirement (<= 10 hours/week)
    weekly_leaf = evaluator.add_leaf(
        id="weekly_hours_requirement",
        desc="Requires 10 hours per week or less of study time",
        parent=req_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The program requires no more than 10 hours per week of study time.",
        node=weekly_leaf,
        sources=info.url,
        additional_instruction=(
            "Look for weekly time commitment statements such as 'X hours/week'. "
            "Ranges like '5–10 hours/week' are acceptable. If it exceeds 10, or if not stated, mark as not supported."
        ),
    )

    # Cost requirement ($75/month or less)
    cost_leaf = evaluator.add_leaf(
        id="cost_requirement",
        desc="Monthly cost is $75 USD or less",
        parent=req_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The monthly cost (subscription or program monthly price) is $75 USD or less.",
        node=cost_leaf,
        sources=info.url,
        additional_instruction=(
            "Prefer the regular advertised monthly subscription or monthly program price on the official page. "
            "Discounts or temporary promos should not be relied upon unless clearly the normal monthly price. "
            "If the page only provides total cost without monthly breakdown, and cannot confirm <= $75/month, mark as not supported."
        ),
    )

    # Certificate type requirement (Professional or Career Certificate)
    credential_leaf = evaluator.add_leaf(
        id="certificate_type_requirement",
        desc="Awards a professional certificate or career certificate (not just course completion)",
        parent=req_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "Upon completion, this program awards a professional certificate or career certificate (not merely a course completion certificate)."
        ),
        node=credential_leaf,
        sources=info.url,
        additional_instruction=(
            "Confirm wording such as 'Professional Certificate', 'Career Certificate', or equivalent credential designation. "
            "Simple statements like 'course completion certificate' alone do NOT satisfy this requirement."
        ),
    )

    # Practical learning requirement
    practical_leaf = evaluator.add_leaf(
        id="practical_application_requirement",
        desc="Includes hands-on projects or practical assignments",
        parent=req_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This program includes hands-on projects or practical assignments as part of the curriculum.",
        node=practical_leaf,
        sources=info.url,
        additional_instruction=(
            "Look for terms like 'hands-on projects', 'practical assignments', 'applied projects', 'capstone project'. "
            "If the curriculum lacks such practical components, mark as not supported."
        ),
    )

    # Online, self-paced requirement
    delivery_leaf = evaluator.add_leaf(
        id="online_delivery_requirement",
        desc="100% online and self-paced (no fixed class meeting times)",
        parent=req_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The program is 100% online and self-paced with no required fixed class meeting times.",
        node=delivery_leaf,
        sources=info.url,
        additional_instruction=(
            "Accept phrases like 'self-paced', 'on-demand', 'learn at your own schedule'. "
            "If required fixed live sessions or in-person meetings exist, mark as not supported."
        ),
    )

    # Professional field requirement
    field_leaf = evaluator.add_leaf(
        id="professional_field_requirement",
        desc="Focuses on a professional career field (e.g., Data Analytics, Project Management, Digital Marketing, Business Analytics, Cybersecurity, or similar)",
        parent=req_node,
        critical=True,
    )
    # Build a helpful claim referencing the extracted field if present
    extracted_field = info.field or "a professional, career-focused field"
    await evaluator.verify(
        claim=f"This program focuses on {extracted_field}, which is a professional, career-focused area.",
        node=field_leaf,
        sources=info.url,
        additional_instruction=(
            "Confirm that the program targets a professional career field such as Data Analytics, Project Management, "
            "Digital Marketing, Business Analytics, Cybersecurity, or a closely related career-focused area. "
            "General non-career-focused topics do not satisfy this requirement."
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
) -> Dict:
    """
    Evaluate the agent's answer for the professional certificate task using the Mind2Web2 evaluation framework.
    """
    # Initialize evaluator (root is non-critical by framework; we'll add a critical child node)
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

    # Record allowed platforms for transparency
    evaluator.add_custom_info(
        info={"allowed_platforms": ALLOWED_PLATFORMS},
        info_type="constraints",
        info_name="platform_constraints",
    )

    # Extract the single selected program info from the answer
    program_info = await evaluator.extract(
        prompt=prompt_extract_program_info(),
        template_class=ProgramInfo,
        extraction_name="selected_program_info",
    )

    # Build the critical "Root" node mirroring rubric Root
    critical_root = evaluator.add_parallel(
        id="Root",
        desc="Find one online professional certificate program that meets all specified requirements",
        parent=root,
        critical=True,
    )

    # Verify all requirements under the critical root
    await verify_program_requirements(evaluator, critical_root, program_info)

    # Return the structured summary with verification tree
    return evaluator.get_summary()