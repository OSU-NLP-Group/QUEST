import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "senior_academic_leadership_positions_eval"
TASK_DESCRIPTION = """Identify three senior academic leadership positions at three different accredited four-year universities in the United States that meet all of the following criteria:

1. The position must be for one of the following senior academic leadership roles: Dean, Provost, Chief Academic Officer, or Vice President for Academic Affairs.

2. The job posting must explicitly require a terminal degree (PhD, EdD, or equivalent doctoral degree) as a minimum qualification.

3. The job posting must explicitly require a minimum of 3 years of prior administrative or leadership experience in higher education.

4. The advertised salary range or minimum salary must be at least $150,000 annually.

5. The position must be currently open or have been posted within the past 6 months (posted after August 2025).

6. The job posting must be accessible on an official university employment website or a recognized higher education job board (such as HigherEdJobs, Chronicle of Higher Education, or Inside Higher Ed).

7. The job posting must include a specific application deadline or closing date.

8. The position must be for a full-time, permanent appointment (not interim, acting, or temporary).

For each position, provide:
- The name of the university
- The specific position title
- A link to the official job posting
- Confirmation of how each criterion is met, with supporting details from the job posting
"""

CUTOFF_DATE_ISO = "2025-08-01"  # Posted after this date
RECOGNIZED_JOB_BOARDS = [
    "higheredjobs.com",
    "jobs.chronicle.com",
    "careers.insidehighered.com",
]
ALLOWED_SENIOR_ROLES = [
    "Dean",
    "Provost",
    "Chief Academic Officer",
    "Vice President for Academic Affairs",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PositionItem(BaseModel):
    university: Optional[str] = None
    position_title: Optional[str] = None
    job_posting_url: Optional[str] = None
    support_urls: List[str] = Field(default_factory=list)


class PositionsExtraction(BaseModel):
    positions: List[PositionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    Extract up to three senior academic leadership positions mentioned in the answer. For each position, extract:
    - university: The name of the university or institution.
    - position_title: The specific job title as written in the answer.
    - job_posting_url: The URL pointing to the official job posting for this position.
    - support_urls: A list of any additional URLs included in the answer that support or reference this position (e.g., accreditation pages, university pages, or job board listings). Only include URLs explicitly present in the answer text.

    Special rules for URLs:
    - Include only URLs that appear explicitly in the answer. Do not invent URLs.
    - If a URL is missing a protocol, prepend with http:// as per instructions.

    Return a JSON with a top-level field 'positions' which is an array of objects, each object containing the above fields.
    If fewer than three positions are present, include only those found.
    If any field for a position is missing from the answer, set it to null (or an empty list for support_urls).
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def unique_urls(primary: Optional[str], others: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    if primary and primary.strip():
        u = primary.strip()
        if u not in seen:
            ordered.append(u)
            seen.add(u)
    for u in others:
        if not u:
            continue
        uu = u.strip()
        if uu and uu not in seen:
            ordered.append(uu)
            seen.add(uu)
    return ordered


def ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][n] if 0 <= n < 5 else f"#{n+1}"


# --------------------------------------------------------------------------- #
# Verification for a single position                                          #
# --------------------------------------------------------------------------- #
async def verify_position(
    evaluator: Evaluator,
    parent_node,
    position: PositionItem,
    idx: int,
    prior_universities: List[str]
) -> None:
    # Build common URL list for evidence
    all_urls = unique_urls(position.job_posting_url, position.support_urls)

    # Create the sequential node for this position
    pos_node = evaluator.add_sequential(
        id=f"position_{idx+1}",
        desc=f"{ordinal(idx)} senior academic leadership position meeting all requirements" if idx < 3 else f"Position #{idx+1} verification",
        parent=parent_node,
        critical=False
    )

    # 1) Identification (critical, parallel)
    ident_node = evaluator.add_parallel(
        id=f"position_{idx+1}_identification",
        desc="Position is identified with institution name, position title, and official job posting URL",
        parent=pos_node,
        critical=True
    )

    # 1.1 University provided (critical existence)
    evaluator.add_custom_node(
        result=bool(position.university and position.university.strip()),
        id=f"position_{idx+1}_institution_provided",
        desc="The name of the university is provided",
        parent=ident_node,
        critical=True
    )

    # 1.2 University type verified (critical) - accreditation and 4-year US institution
    institution_type_leaf = evaluator.add_leaf(
        id=f"position_{idx+1}_institution_type_verified",
        desc="The university is verified to be an accredited four-year institution in the United States",
        parent=ident_node,
        critical=True
    )
    uni_name_for_claim = position.university or "Unknown university"
    await evaluator.verify(
        claim=f"'{uni_name_for_claim}' is a U.S.-based accredited four-year university (awards bachelor's degrees).",
        node=institution_type_leaf,
        sources=all_urls,
        additional_instruction=(
            "Verify using the provided URLs only. Acceptable evidence includes: statements that the institution "
            "is accredited by a U.S.-recognized accreditor (regional or national), mentions of awarding bachelor's degrees, "
            "or clearly being a U.S. university. Use any provided URL (job posting, university pages, accreditation pages). "
            "If none of the provided pages supports that it is a U.S. accredited four-year institution, mark as not supported."
        )
    )

    # 1.3 Position title provided (critical existence)
    evaluator.add_custom_node(
        result=bool(position.position_title and position.position_title.strip()),
        id=f"position_{idx+1}_position_title_provided",
        desc="The specific title of the senior academic leadership position is provided",
        parent=ident_node,
        critical=True
    )

    # 1.4 Job posting URL provided (critical existence)
    evaluator.add_custom_node(
        result=bool(position.job_posting_url and position.job_posting_url.strip()),
        id=f"position_{idx+1}_job_posting_url_provided",
        desc="A URL link to the official job posting is provided",
        parent=ident_node,
        critical=True
    )

    # 1.5 Uniqueness constraints for positions 2 and 3
    if idx == 1:
        # Different from position 1
        diff_leaf = evaluator.add_leaf(
            id=f"position_{idx+1}_different_from_position_1",
            desc="The university is different from the university in Position 1",
            parent=ident_node,
            critical=True
        )
        prev1 = prior_universities[0] if len(prior_universities) > 0 else ""
        claim_diff = f"The university '{position.university or ''}' is different from '{prev1 or ''}'."
        await evaluator.verify(
            claim=claim_diff,
            node=diff_leaf,
            additional_instruction="Treat this as a logical comparison of the two institution names. Minor punctuation or casing differences do not count as the same if the institutions are actually different; conversely, name variants referring to the same university should be considered the same."
        )
    if idx == 2:
        # Different from positions 1 and 2
        diff_leaf = evaluator.add_leaf(
            id=f"position_{idx+1}_different_from_positions_1_and_2",
            desc="The university is different from the universities in Position 1 and Position 2",
            parent=ident_node,
            critical=True
        )
        prev1 = prior_universities[0] if len(prior_universities) > 0 else ""
        prev2 = prior_universities[1] if len(prior_universities) > 1 else ""
        claim_diff = f"The university '{position.university or ''}' is different from both '{prev1 or ''}' and '{prev2 or ''}'."
        await evaluator.verify(
            claim=claim_diff,
            node=diff_leaf,
            additional_instruction="Treat this as a logical comparison of the institution names with reasonable normalization (case-insensitive, ignore minor punctuation). Name variants that clearly refer to the same institution should not be considered different."
        )

    # 2) Position verification (critical, parallel)
    verify_node = evaluator.add_parallel(
        id=f"position_{idx+1}_verification",
        desc="All required criteria are verified from the job posting",
        parent=pos_node,
        critical=True
    )

    # 2.a Leadership level (critical) -> leaf inside a container (as per rubric)
    leadership_container = evaluator.add_parallel(
        id=f"position_{idx+1}_leadership_level",
        desc="Position is verified to be a senior academic leadership role",
        parent=verify_node,
        critical=True
    )
    leadership_leaf = evaluator.add_leaf(
        id=f"position_{idx+1}_leadership_role_confirmed",
        desc="The position title explicitly indicates one of the qualifying senior academic leadership roles: Dean, Provost, Chief Academic Officer, or Vice President for Academic Affairs",
        parent=leadership_container,
        critical=True
    )
    title_for_claim = position.position_title or ""
    await evaluator.verify(
        claim=(
            f"The job posting shows the position title '{title_for_claim}' and it corresponds to one of the allowed senior roles: "
            f"{', '.join(ALLOWED_SENIOR_ROLES)}."
        ),
        node=leadership_leaf,
        sources=all_urls,
        additional_instruction=(
            "Confirm from the provided job posting page. Only accept 'Dean', 'Provost', 'Chief Academic Officer', or "
            "'Vice President for Academic Affairs' (including reasonable variants like 'VP for Academic Affairs'). "
            "Do NOT accept other titles like 'Vice Provost', 'Associate/Assistant Dean', or 'Interim' roles."
        )
    )

    # 2.b Terminal degree requirement (critical)
    degree_container = evaluator.add_parallel(
        id=f"position_{idx+1}_terminal_degree_requirement",
        desc="Job posting explicitly requires a terminal degree",
        parent=verify_node,
        critical=True
    )
    degree_leaf = evaluator.add_leaf(
        id=f"position_{idx+1}_terminal_degree_stated",
        desc="The job posting explicitly states that a terminal degree (PhD, EdD, or equivalent doctoral degree) is required",
        parent=degree_container,
        critical=True
    )
    await evaluator.verify(
        claim="The job posting explicitly requires a terminal doctoral degree (e.g., PhD, EdD, or equivalent) as a minimum qualification.",
        node=degree_leaf,
        sources=all_urls,
        additional_instruction=(
            "Look for phrases like 'terminal degree required', 'PhD required', 'EdD required', or 'doctoral degree required'. "
            "The requirement must be mandatory (minimum qualification), not merely preferred."
        )
    )

    # 2.c Administrative experience requirement (critical)
    exp_container = evaluator.add_parallel(
        id=f"position_{idx+1}_administrative_experience_requirement",
        desc="Job posting explicitly requires administrative experience",
        parent=verify_node,
        critical=True
    )
    exp_leaf = evaluator.add_leaf(
        id=f"position_{idx+1}_experience_years_stated",
        desc="The job posting explicitly states a requirement for at least 3 years of administrative or leadership experience in higher education",
        parent=exp_container,
        critical=True
    )
    await evaluator.verify(
        claim="The job posting requires at least 3 years of administrative or leadership experience in higher education.",
        node=exp_leaf,
        sources=all_urls,
        additional_instruction=(
            "Accept if the requirement is for 3 or more years (e.g., '5 years') as satisfying 'at least 3 years'. "
            "The experience must be administrative/leadership and specifically in higher education. "
            "The requirement must be mandatory, not just preferred."
        )
    )

    # 2.d Salary requirement (critical)
    salary_container = evaluator.add_parallel(
        id=f"position_{idx+1}_salary_requirement",
        desc="Salary meets minimum threshold",
        parent=verify_node,
        critical=True
    )
    salary_leaf = evaluator.add_leaf(
        id=f"position_{idx+1}_salary_meets_threshold",
        desc="The job posting indicates a salary range with a minimum of at least $150,000 annually, or states a salary of at least $150,000",
        parent=salary_container,
        critical=True
    )
    await evaluator.verify(
        claim="The job posting shows a salary with a minimum value of at least $150,000 per year, or a stated salary of at least $150,000 annually.",
        node=salary_leaf,
        sources=all_urls,
        additional_instruction=(
            "Look for explicit salary information on the page. Accept ranges where the lower bound is >= 150,000, or explicit minimums "
            "or 'starting at' values >= 150,000. If only total compensation or benefits are mentioned without a clear salary minimum, do not accept."
        )
    )

    # 2.e Posting status (critical) -> recency, official channel, deadline
    posting_container = evaluator.add_parallel(
        id=f"position_{idx+1}_posting_status",
        desc="Position posting meets recency and accessibility requirements",
        parent=verify_node,
        critical=True
    )
    # Recency
    recency_leaf = evaluator.add_leaf(
        id=f"position_{idx+1}_posting_recency",
        desc=f"The job posting was posted within the past 6 months from February 2026 (posted after August 2025)",
        parent=posting_container,
        critical=True
    )
    await evaluator.verify(
        claim=f"The job posting was posted after {CUTOFF_DATE_ISO}.",
        node=recency_leaf,
        sources=all_urls,
        additional_instruction=(
            f"Find an explicit 'Posted', 'Posting date', or similar indicator on the page. Accept if the posting date is strictly after {CUTOFF_DATE_ISO}. "
            "If only an 'Updated' or 'Reposted' date is shown, use the latest clearly indicated posting-related date. "
            "If no date is visible, mark as not supported."
        )
    )
    # Official channel accessibility
    official_leaf = evaluator.add_leaf(
        id=f"position_{idx+1}_official_channel_accessibility",
        desc="The job posting is accessible on an official university employment website or a recognized higher education job board (e.g., HigherEdJobs, Chronicle of Higher Education, Inside Higher Ed)",
        parent=posting_container,
        critical=True
    )
    recognized_domains_str = ", ".join(RECOGNIZED_JOB_BOARDS)
    await evaluator.verify(
        claim=(
            "The job posting URL is either hosted on an official university employment/careers website (often a .edu domain) "
            f"or on a recognized higher education job board among: {recognized_domains_str}."
        ),
        node=official_leaf,
        sources=all_urls,
        additional_instruction=(
            "You will be shown the URL. If it ends with .edu and is a careers/employment portal for the university, accept. "
            f"Also accept if the domain matches one of: {recognized_domains_str}. "
            "Do not accept general job aggregators or social networks (e.g., LinkedIn, Indeed) as recognized boards for this task."
        )
    )
    # Application deadline
    deadline_leaf = evaluator.add_leaf(
        id=f"position_{idx+1}_application_deadline_stated",
        desc="The job posting includes a specific application deadline or closing date",
        parent=posting_container,
        critical=True
    )
    await evaluator.verify(
        claim="The job posting includes a specific application deadline or closing date (a concrete calendar date).",
        node=deadline_leaf,
        sources=all_urls,
        additional_instruction=(
            "Look for a specific date (e.g., 'Apply by January 15, 2026'). Phrases like 'open until filled' without a date do not satisfy this criterion."
        )
    )

    # 2.f Appointment type (critical)
    appoint_container = evaluator.add_parallel(
        id=f"position_{idx+1}_appointment_type",
        desc="Position is for permanent, full-time appointment",
        parent=verify_node,
        critical=True
    )
    appoint_leaf = evaluator.add_leaf(
        id=f"position_{idx+1}_permanent_full_time_confirmed",
        desc="The job posting indicates the position is for a full-time, permanent appointment and does not use terms like 'interim,' 'acting,' or 'temporary'",
        parent=appoint_container,
        critical=True
    )
    await evaluator.verify(
        claim="The job posting indicates the position is a full-time, permanent appointment and not interim, acting, or temporary.",
        node=appoint_leaf,
        sources=all_urls,
        additional_instruction=(
            "Look for explicit 'Full-time' and terms implying permanence (e.g., permanent, ongoing). "
            "If the page uses 'interim', 'acting', or 'temporary', or lacks clear permanence, do not accept."
        )
    )


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
        default_model=model
    )

    # Extraction: positions
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction"
    )

    # Normalize to exactly 3 positions (pad if needed)
    positions: List[PositionItem] = list(extracted.positions[:3])
    while len(positions) < 3:
        positions.append(PositionItem())

    # Add custom info for recognized job boards and cutoff for transparency
    evaluator.add_custom_info(
        {
            "cutoff_date_iso": CUTOFF_DATE_ISO,
            "recognized_job_boards": RECOGNIZED_JOB_BOARDS,
            "allowed_senior_roles": ALLOWED_SENIOR_ROLES
        },
        info_type="policy",
        info_name="verification_policy"
    )

    # Build tree for three positions
    prior_unis: List[str] = []
    for i in range(3):
        # Track previously seen universities for uniqueness checks
        current_uni = positions[i].university or ""
        await verify_position(
            evaluator=evaluator,
            parent_node=root,
            position=positions[i],
            idx=i,
            prior_universities=prior_unis
        )
        if current_uni:
            prior_unis.append(current_uni)

    return evaluator.get_summary()