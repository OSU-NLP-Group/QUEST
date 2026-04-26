import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "al_ad_opening_2025_2026"
TASK_DESCRIPTION = (
    "Find an Alabama high school that currently has an open athletic director position (posted in 2025 or 2026) and verify "
    "that the position meets all of the following requirements: "
    "(1) The school is a member of the Alabama High School Athletic Association (AHSAA), "
    "(2) The position is for a high school level (not middle school or elementary school), "
    "(3) The position is full-time (not part-time or assistant athletic director), "
    "(4) The position requires a bachelor's degree from an accredited institution, "
    "(5) The position requires or prefers a valid Alabama teaching certificate, "
    "(6) The position requires a minimum of 5 years of coaching or athletic administration experience, "
    "(7) The position description mentions knowledge of or compliance with AHSAA rules and regulations, "
    "(8) The job posting appears on a recognized official channel (AHSAA classifieds, school district website, or major job board like Indeed or ZipRecruiter). "
    "Additionally, identify whether the position mentions any of these preferred qualifications: salary range (and whether it falls within $60,000-$110,000 annually), "
    "preference for an administration certificate, or NIAAA certification as desired or beneficial. Provide the school name, location, and a reference URL for the job posting."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class JobPostingInfo(BaseModel):
    # Identification
    school_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    position_title: Optional[str] = None

    # URLs
    posting_url: Optional[str] = None   # primary job posting link
    membership_urls: List[str] = Field(default_factory=list)  # URLs that may demonstrate AHSAA membership
    school_website_url: Optional[str] = None
    district_website_url: Optional[str] = None
    ahsaa_classifieds_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)

    # Dates and extra info
    posting_date_text: Optional[str] = None
    salary_text: Optional[str] = None
    salary_min: Optional[str] = None
    salary_max: Optional[str] = None

    # Optional preferred mentions (as parsed from answer text)
    mentions_admin_certificate: Optional[bool] = None
    mentions_niaaa_certification: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_job_posting_info() -> str:
    return (
        "From the answer, extract details for exactly one Alabama high school athletic director job posting that is described. "
        "If multiple postings are mentioned, pick the first one that appears to meet the criteria and is posted in 2025 or 2026. "
        "Return the following fields:\n"
        "1) school_name: The school's name.\n"
        "2) city: The city of the school, if provided.\n"
        "3) state: The state (e.g., 'Alabama' or 'AL').\n"
        "4) position_title: The role title (e.g., 'Athletic Director').\n"
        "5) posting_url: The primary URL of the job posting page (prefer the specific job listing; if multiple, choose the most official or most specific one). Extract only if explicitly present in the answer.\n"
        "6) membership_urls: An array of URLs, if provided, that can help verify the school’s AHSAA membership (e.g., AHSAA directory page for the school). Extract only if explicitly present in the answer.\n"
        "7) school_website_url: The official school website URL if present in the answer; otherwise null.\n"
        "8) district_website_url: The official district website URL if present in the answer; otherwise null.\n"
        "9) ahsaa_classifieds_url: If the AHSAA classifieds link is mentioned in the answer, include it; otherwise null.\n"
        "10) additional_urls: Any other URLs in the answer that relate to this posting or the school.\n"
        "11) posting_date_text: The posting date text as written in the answer (e.g., 'Posted January 15, 2026').\n"
        "12) salary_text: The salary text if the answer mentions an amount or range; otherwise null.\n"
        "13) salary_min: The lower bound of the annual salary if a range or number is given (numbers only, no symbols); otherwise null.\n"
        "14) salary_max: The upper bound of the annual salary if a range is given (numbers only, no symbols); otherwise null.\n"
        "15) mentions_admin_certificate: true if the answer explicitly says the posting mentions preference for an administration certificate; false if the answer explicitly says it does not; null if not mentioned.\n"
        "16) mentions_niaaa_certification: true if the answer explicitly says the posting mentions NIAAA certification (desired/preferred/beneficial); false if the answer explicitly says it does not; null if not mentioned.\n"
        "Only extract values that actually appear in the answer text. Do not invent any URLs or values."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_urls(*urls_or_lists: Any) -> List[str]:
    urls: List[str] = []
    for item in urls_or_lists:
        if not item:
            continue
        if isinstance(item, str):
            s = item.strip()
            if s:
                urls.append(s)
        elif isinstance(item, list):
            for u in item:
                if isinstance(u, str) and u.strip():
                    urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def _salary_within_range_text(min_text: Optional[str], max_text: Optional[str]) -> Optional[str]:
    """
    Build a minimal textual summary for later logging (not used for verification claim).
    """
    if not min_text and not max_text:
        return None
    if min_text and max_text:
        return f"{min_text} - {max_text}"
    if min_text and not max_text:
        return f"{min_text}+"
    if not min_text and max_text:
        return f"up to {max_text}"
    return None


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def verify_job_posting(evaluator: Evaluator, parent_node, info: JobPostingInfo) -> None:
    """
    Build verification leaves according to the rubric and trigger verifications.
    """
    # Consolidated source lists
    posting_sources = _non_empty_urls(
        info.posting_url,
        info.school_website_url,
        info.district_website_url,
        info.ahsaa_classifieds_url,
        info.additional_urls
    )
    membership_sources = _non_empty_urls(info.membership_urls, posting_sources)

    school_name = (info.school_name or "").strip()
    city = (info.city or "").strip()
    state = (info.state or "").strip()

    # 1) School identification (existence)
    evaluator.add_custom_node(
        result=bool(school_name) and bool(city or state),
        id="school_identification",
        desc="The identified school name and location are provided",
        parent=parent_node,
        critical=True
    )

    # 2) Alabama location (source-grounded)
    node_al_location = evaluator.add_leaf(
        id="alabama_location",
        desc="The school is located in Alabama",
        parent=parent_node,
        critical=True
    )
    claim_al = f"The school named '{school_name}' is located in the U.S. state of Alabama."
    await evaluator.verify(
        claim=claim_al,
        node=node_al_location,
        sources=posting_sources,
        additional_instruction=(
            "Use the provided page(s) to confirm the school is in Alabama. Accept 'AL' as equivalent to 'Alabama'. "
            "If the page shows a city in Alabama or uses a .k12.al.us domain or clearly indicates Alabama, consider supported."
        )
    )

    # 3) AHSAA membership (source-grounded)
    node_ahsaa_member = evaluator.add_leaf(
        id="ahsaa_membership",
        desc="The school is a verified member of the Alabama High School Athletic Association (AHSAA)",
        parent=parent_node,
        critical=True
    )
    claim_member = f"The school '{school_name}' is a member school of the Alabama High School Athletic Association (AHSAA)."
    await evaluator.verify(
        claim=claim_member,
        node=node_ahsaa_member,
        sources=membership_sources,
        additional_instruction=(
            "Look for explicit membership in AHSAA on an AHSAA page (e.g., ahsaa.com), a school athletics page that states AHSAA membership, "
            "or an official listing in an AHSAA directory/classification/schedules page. If no provided page supports AHSAA membership, mark as not supported."
        )
    )

    # 4) High school level (source-grounded)
    node_hs_level = evaluator.add_leaf(
        id="high_school_level",
        desc="The position is specifically for a high school (not middle school or elementary school)",
        parent=parent_node,
        critical=True
    )
    claim_hs = (
        "The job is for an Athletic Director role at the high school level (not a middle school or elementary school position)."
    )
    await evaluator.verify(
        claim=claim_hs,
        node=node_hs_level,
        sources=posting_sources,
        additional_instruction=(
            "Confirm that the posting refers to a high school position (e.g., 'High School Athletic Director', grades 9–12, or clearly "
            "linked to the high school campus). If the posting is for middle/elementary level only, or not clearly high school, mark as not supported."
        )
    )

    # 5) Active posting and correct year (source-grounded)
    node_active = evaluator.add_leaf(
        id="active_posting",
        desc="The job posting is active and current (posted in 2025 or 2026)",
        parent=parent_node,
        critical=True
    )
    claim_active = (
        "The job posting for the school's Athletic Director was posted in 2025 or 2026 and appears to be currently open (e.g., accepting applications)."
    )
    await evaluator.verify(
        claim=claim_active,
        node=node_active,
        sources=posting_sources,
        additional_instruction=(
            "Look for a posting date or context indicating the listing is from 2025 or 2026. "
            "Also look for signs the posting is active (e.g., 'Apply', 'Accepting applications', no indication of 'closed' or 'filled'). "
            "If the date is not in 2025/2026 or it appears closed/expired, mark as not supported."
        )
    )

    # 6) Full-time and not assistant (source-grounded)
    node_full_time = evaluator.add_leaf(
        id="full_time_position",
        desc="The position is for a full-time athletic director (not part-time or assistant athletic director)",
        parent=parent_node,
        critical=True
    )
    claim_full_time = (
        "The job is a full-time Athletic Director position and not an Assistant Athletic Director position."
    )
    await evaluator.verify(
        claim=claim_full_time,
        node=node_full_time,
        sources=posting_sources,
        additional_instruction=(
            "Confirm that the posting specifies full-time (e.g., 'Full-Time', 'FT', 1.0 FTE) and that the role is Athletic Director, not Assistant AD. "
            "If ambiguous or shows part-time or 'Assistant Athletic Director', mark as not supported."
        )
    )

    # 7) Bachelor's degree required (source-grounded)
    node_bachelor = evaluator.add_leaf(
        id="bachelors_degree",
        desc="The position requires a bachelor's degree from an accredited institution",
        parent=parent_node,
        critical=True
    )
    claim_bachelor = (
        "The posting states that a bachelor's degree from an accredited college or university is required."
    )
    await evaluator.verify(
        claim=claim_bachelor,
        node=node_bachelor,
        sources=posting_sources,
        additional_instruction=(
            "Look for text like 'Bachelor's degree required' and preferably 'from an accredited institution/college/university'. "
            "If only 'preferred' (not required), mark as not supported."
        )
    )

    # 8) Alabama teaching certificate required or preferred (source-grounded)
    node_al_cert = evaluator.add_leaf(
        id="alabama_certificate",
        desc="The position requires or prefers a valid Alabama teaching certificate",
        parent=parent_node,
        critical=True
    )
    claim_cert = (
        "The posting states that a valid Alabama teaching certificate is required or preferred."
    )
    await evaluator.verify(
        claim=claim_cert,
        node=node_al_cert,
        sources=posting_sources,
        additional_instruction=(
            "Accept formulations like 'valid Alabama teaching certificate required' or 'preferred', "
            "or 'ALSDE certification' (Alabama State Dept. of Education). If certification is non-Alabama or not mentioned, mark as not supported."
        )
    )

    # 9) Experience requirement >= 5 years (source-grounded)
    node_experience = evaluator.add_leaf(
        id="experience_requirement",
        desc="The position requires a minimum of 5 years of coaching or athletic administration experience",
        parent=parent_node,
        critical=True
    )
    claim_exp = (
        "The posting requires at least 5 years of experience in coaching or athletic administration (or a closely related combination)."
    )
    await evaluator.verify(
        claim=claim_exp,
        node=node_experience,
        sources=posting_sources,
        additional_instruction=(
            "Look for phrases like 'minimum five (5) years', 'at least 5 years', etc., in coaching or athletic administration contexts. "
            "If fewer than 5 years or not required, mark as not supported."
        )
    )

    # 10) AHSAA knowledge/compliance mention (source-grounded)
    node_ahsaa_knowledge = evaluator.add_leaf(
        id="ahsaa_knowledge",
        desc="The position description mentions knowledge of or compliance with AHSAA rules and regulations",
        parent=parent_node,
        critical=True
    )
    claim_ahsaa_knowledge = (
        "The posting mentions knowledge of, adherence to, or compliance with AHSAA rules, regulations, or bylaws."
    )
    await evaluator.verify(
        claim=claim_ahsaa_knowledge,
        node=node_ahsaa_knowledge,
        sources=posting_sources,
        additional_instruction=(
            "Accept explicit references to 'AHSAA rules', 'AHSAA bylaws', or 'AHSAA regulations'. "
            "General 'state rules' without AHSAA mention should not count."
        )
    )

    # 11) Posting on recognized official channel (source-grounded)
    node_posting_source = evaluator.add_leaf(
        id="posting_source",
        desc="The job posting appears on a recognized official channel (AHSAA classifieds, school district website, major job board)",
        parent=parent_node,
        critical=True
    )
    claim_channel = (
        "This job posting appears on a recognized official channel: the AHSAA website/classifieds, "
        "an official school or Alabama school district website (often .k12.al.us), "
        "or a major job board like Indeed, ZipRecruiter, or LinkedIn Jobs."
    )
    await evaluator.verify(
        claim=claim_channel,
        node=node_posting_source,
        sources=info.posting_url or posting_sources,
        additional_instruction=(
            "Consider the channel recognized if the domain is clearly AHSAA (e.g., ahsaa.com), a school/district domain "
            "(e.g., *.k12.al.us or the district's official site), or a major job board (indeed.com, ziprecruiter.com, linkedin.com/jobs). "
            "If it's a dubious aggregator or unrelated site, mark as not supported."
        )
    )

    # 12) Salary information within range if provided (non-critical)
    salary_present = bool((info.salary_text or "").strip())
    if salary_present:
        node_salary = evaluator.add_leaf(
            id="salary_information",
            desc="If salary is provided, it falls within the typical Alabama range of $60,000-$110,000 annually",
            parent=parent_node,
            critical=False
        )
        claim_salary = (
            "The posted salary or salary range for this Athletic Director role is within $60,000 to $110,000 per year. "
            "If a single annual number is posted, it should fall within this range."
        )
        await evaluator.verify(
            claim=claim_salary,
            node=node_salary,
            sources=posting_sources,
            additional_instruction=(
                "Use the job page to identify any salary text. If an annual salary number or range is present, check whether it fits entirely within "
                "$60,000 to $110,000. If only hourly/daily rates are provided without annual equivalence, treat as not supported."
            )
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="salary_information",
            desc="If salary is provided, it falls within the typical Alabama range of $60,000-$110,000 annually (no salary provided in answer, not required)",
            parent=parent_node,
            critical=False
        )

    # 13) Administration certificate preference (non-critical, source-grounded)
    node_admin_cert = evaluator.add_leaf(
        id="administration_certificate",
        desc="The position mentions preference for an administration certificate or administrative certification",
        parent=parent_node,
        critical=False
    )
    claim_admin_cert = (
        "The posting mentions a preference for an administration certificate or administrative certification."
    )
    await evaluator.verify(
        claim=claim_admin_cert,
        node=node_admin_cert,
        sources=posting_sources,
        additional_instruction=(
            "Look for terms like 'administrative certification preferred', 'administration certificate preferred', "
            "or equivalent. If not mentioned, mark as not supported."
        )
    )

    # 14) NIAAA certification mention (non-critical, source-grounded)
    node_niaaa = evaluator.add_leaf(
        id="niaaa_certification",
        desc="The posting mentions NIAAA certification as desired, preferred, or beneficial",
        parent=parent_node,
        critical=False
    )
    claim_niaaa = (
        "The posting mentions NIAAA certification as desired, preferred, or beneficial."
    )
    await evaluator.verify(
        claim=claim_niaaa,
        node=node_niaaa,
        sources=posting_sources,
        additional_instruction=(
            "Look for explicit references to 'NIAAA' or 'NIAAA certification'. If absent, mark as not supported."
        )
    )

    # Record some custom info for transparency
    evaluator.add_custom_info(
        info={
            "school_name": school_name,
            "city": city,
            "state": state,
            "posting_url": info.posting_url,
            "membership_urls": info.membership_urls,
            "salary_present": salary_present,
            "salary_summary": _salary_within_range_text(info.salary_min, info.salary_max),
        },
        info_type="extracted_overview"
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
    Entry point to evaluate an answer for the Alabama High School Athletic Director posting task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # root aggregates all checks in parallel
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
    posting_info: JobPostingInfo = await evaluator.extract(
        prompt=prompt_extract_job_posting_info(),
        template_class=JobPostingInfo,
        extraction_name="posting_info"
    )

    # Attach ground truth-like task context for reference
    evaluator.add_ground_truth({
        "requirements": [
            "AHSAA membership",
            "High school level",
            "Full-time (not assistant)",
            "Bachelor's degree required from accredited institution",
            "Alabama teaching certificate required or preferred",
            "Minimum 5 years coaching/athletic administration experience required",
            "AHSAA rules/regulations knowledge/compliance mentioned",
            "Recognized official posting channel (AHSAA/site/major job board)",
            "Optional: salary within $60k-$110k if provided",
            "Optional: admin certificate preference",
            "Optional: NIAAA certification mention"
        ],
        "year_constraint": "Posted in 2025 or 2026",
        "location_constraint": "School must be in Alabama"
    }, gt_type="task_requirements")

    # Build verification leaves according to rubric
    await verify_job_posting(evaluator, root, posting_info)

    return evaluator.get_summary()