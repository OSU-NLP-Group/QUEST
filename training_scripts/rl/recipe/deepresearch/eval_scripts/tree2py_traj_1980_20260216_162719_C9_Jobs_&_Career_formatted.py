import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "edu_superintendent_positions_2026"
TASK_DESCRIPTION = """A senior educational administrator with a Doctorate in Educational Leadership and 12 years of experience in K-12 administration (including 7 years as a school principal) is seeking a Superintendent or Assistant Superintendent position for the 2026-2027 school year. They are willing to relocate anywhere in the United States but have specific requirements for their next role.

Identify at least three distinct school district Superintendent or Assistant Superintendent positions that meet ALL of the following criteria:

1. The position must have an application deadline between February 12, 2026, and March 27, 2026 (inclusive).
2. The expected start date must be July 1, 2026.
3. The minimum annual salary must be at least $185,000.
4. The position must require a Master's degree as the minimum educational qualification.
5. The position must require at least 5 years of educational leadership experience.
6. The position must require a valid state administrative or superintendent certification (or demonstrate a pathway to obtain one).
7. Applications must be submitted through a publicly accessible online portal.

For each position, provide:
- The school district name and position title
- The state location
- The application deadline
- The start date
- The salary range
- A summary of the minimum qualification requirements (education, experience, and certification)
- The online application portal URL
- The school district enrollment (if available)
- A reference URL to the official job posting or announcement
"""

# Policy thresholds and constants
DEADLINE_START_TEXT = "February 12, 2026"
DEADLINE_END_TEXT = "March 27, 2026"
EXPECTED_START_DATE_TEXT = "July 1, 2026"
MIN_SALARY_THRESHOLD = 185000

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PositionItem(BaseModel):
    district_name: Optional[str] = None
    position_title: Optional[str] = None
    state: Optional[str] = None
    application_deadline: Optional[str] = None
    start_date: Optional[str] = None
    salary_range: Optional[str] = None
    education_requirement: Optional[str] = None
    experience_requirement: Optional[str] = None
    certification_requirement: Optional[str] = None
    application_portal_url: Optional[str] = None
    job_posting_url: Optional[str] = None
    enrollment: Optional[str] = None
    enrollment_source_url: Optional[str] = None
    extra_reference_urls: List[str] = Field(default_factory=list)


class PositionsExtraction(BaseModel):
    positions: List[PositionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
Extract up to five (5) distinct school district leadership positions from the answer that are Superintendent or Assistant Superintendent roles.

For each position, extract the following fields exactly as presented in the answer:
- district_name: The school district name
- position_title: The exact position title
- state: The U.S. state in which the district is located
- application_deadline: The application deadline date (verbatim as written)
- start_date: The expected start date (verbatim as written)
- salary_range: The salary range or minimum salary figure (verbatim as written; do not convert)
- education_requirement: The minimum education requirement summary (verbatim as written)
- experience_requirement: The experience requirement summary (verbatim as written)
- certification_requirement: The certification requirement summary (verbatim as written)
- application_portal_url: The URL to the online application portal for this position
- job_posting_url: The URL to the official job posting or announcement
- enrollment: The school district student enrollment if provided (verbatim as written)
- enrollment_source_url: A source URL that supports the enrollment figure if provided in the answer
- extra_reference_urls: Any additional reference URLs cited for this specific position in the answer (exclude duplicates of application_portal_url or job_posting_url)

Special rules for URLs:
- Extract only URLs that are explicitly present in the answer (plain text or markdown links).
- Ensure URLs are complete (include http/https). If a URL is missing a protocol, prepend http://.
- If a URL for a required field is not present in the answer, set it to null.

If any field is missing for a position, set it to null. Do not infer information.
Return a JSON object with a top-level array field 'positions' containing up to 5 objects.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][n] if 0 <= n < 5 else f"#{n+1}"


def _fail_leaf_due_to_missing(node) -> None:
    node.score = 0.0
    node.status = "failed"


def _has_text(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _source_list(*urls: Optional[str]) -> List[str]:
    return [u for u in urls if _has_text(u)]


# --------------------------------------------------------------------------- #
# Verification for a single position                                          #
# --------------------------------------------------------------------------- #
async def verify_position(
    evaluator: Evaluator,
    parent_node,
    pos: PositionItem,
    idx: int,
) -> None:
    """
    Build and verify a single position subtree according to the rubric.
    """
    pos_node = evaluator.add_parallel(
        id=f"position_{idx+1}",
        desc=f"{ordinal(idx)} qualifying position identified with complete details",
        parent=parent_node,
        critical=False
    )

    # Sources
    posting_url = pos.job_posting_url if _has_text(pos.job_posting_url) else None
    portal_url = pos.application_portal_url if _has_text(pos.application_portal_url) else None

    # 1) Identification: district + title supported by job posting
    ident_node = evaluator.add_leaf(
        id=f"position_{idx+1}_identification",
        desc="Correctly identifies the school district name and specific position title",
        parent=pos_node,
        critical=True
    )
    if posting_url and _has_text(pos.district_name) and _has_text(pos.position_title):
        claim = (
            f"This official job posting is for the position '{pos.position_title}' at the "
            f"'{pos.district_name}' school district (or equivalent, e.g., 'Public Schools', 'School System')."
        )
        await evaluator.verify(
            claim=claim,
            node=ident_node,
            sources=posting_url,
            additional_instruction="Allow reasonable title variants (e.g., 'Superintendent of Schools', 'Assistant Superintendent'). "
                                   "Focus on whether the posting clearly shows the named district and the stated title."
        )
    else:
        _fail_leaf_due_to_missing(ident_node)

    # 2) Location: state on job posting
    location_node = evaluator.add_leaf(
        id=f"position_{idx+1}_location",
        desc="Provides the correct state location of the school district",
        parent=pos_node,
        critical=True
    )
    if posting_url and _has_text(pos.state):
        claim = f"The job posting indicates the district is located in the U.S. state of {pos.state}."
        await evaluator.verify(
            claim=claim,
            node=location_node,
            sources=posting_url,
            additional_instruction="The state may be shown in an address block, header/footer, or page content. "
                                   "Use page text and screenshot only; do not rely on external knowledge."
        )
    else:
        _fail_leaf_due_to_missing(location_node)

    # 3) Deadline within range
    deadline_node = evaluator.add_leaf(
        id=f"position_{idx+1}_deadline",
        desc=f"States the application deadline and verifies it falls between {DEADLINE_START_TEXT} and {DEADLINE_END_TEXT}",
        parent=pos_node,
        critical=True
    )
    if posting_url and _has_text(pos.application_deadline):
        claim = (
            f"The application deadline for this position is {pos.application_deadline}, and this date falls between "
            f"{DEADLINE_START_TEXT} and {DEADLINE_END_TEXT}, inclusive."
        )
        await evaluator.verify(
            claim=claim,
            node=deadline_node,
            sources=posting_url,
            additional_instruction="Accept equivalent phrases like 'Apply by' or 'Closes on'. If the posting says "
                                   "'Open until filled' without a clear date, this fails. Interpret dates in common U.S. formats."
        )
    else:
        _fail_leaf_due_to_missing(deadline_node)

    # 4) Start date = July 1, 2026
    start_node = evaluator.add_leaf(
        id=f"position_{idx+1}_start_date",
        desc=f"Confirms the position has an expected start date of {EXPECTED_START_DATE_TEXT}",
        parent=pos_node,
        critical=True
    )
    if posting_url and _has_text(pos.start_date):
        claim = f"The expected start date for this position is {EXPECTED_START_DATE_TEXT} (or equivalent wording indicating {EXPECTED_START_DATE_TEXT})."
        await evaluator.verify(
            claim=claim,
            node=start_node,
            sources=posting_url,
            additional_instruction="Accept equivalent phrasing like 'on or about July 1, 2026' or 'anticipated start July 1, 2026'."
        )
    else:
        _fail_leaf_due_to_missing(start_node)

    # 5) Salary minimum >= $185,000
    salary_node = evaluator.add_leaf(
        id=f"position_{idx+1}_salary",
        desc=f"Reports the salary range and verifies the minimum salary is at least ${MIN_SALARY_THRESHOLD:,}",
        parent=pos_node,
        critical=True
    )
    if posting_url:
        claim = f"The minimum annual salary for the position is at least ${MIN_SALARY_THRESHOLD:,}."
        await evaluator.verify(
            claim=claim,
            node=salary_node,
            sources=posting_url,
            additional_instruction="Accept if the posting shows a salary range whose lower bound is >= 185,000, or explicitly states "
                                   "a minimum at or above $185,000. If only 'commensurate' or no minimum is given, this fails."
        )
    else:
        _fail_leaf_due_to_missing(salary_node)

    # 6) Education: Master's degree required
    education_node = evaluator.add_leaf(
        id=f"position_{idx+1}_education",
        desc="Confirms the position requires a Master's degree as minimum qualification",
        parent=pos_node,
        critical=True
    )
    if posting_url:
        claim = "The position requires at minimum a Master's degree."
        await evaluator.verify(
            claim=claim,
            node=education_node,
            sources=posting_url,
            additional_instruction="The posting must indicate a Master's degree is required (not just preferred). "
                                   "Phrasing such as 'Master's degree or higher required' qualifies."
        )
    else:
        _fail_leaf_due_to_missing(education_node)

    # 7) Experience: >= 5 years educational leadership
    experience_node = evaluator.add_leaf(
        id=f"position_{idx+1}_experience",
        desc="Verifies the position requires at least 5 years of educational leadership experience",
        parent=pos_node,
        critical=True
    )
    if posting_url:
        claim = "The position requires at least five years of educational leadership experience."
        await evaluator.verify(
            claim=claim,
            node=experience_node,
            sources=posting_url,
            additional_instruction="Accept relevant phrases like 'minimum five (5) years' in roles such as principal, central office, "
                                   "or district-level leadership. If only fewer than 5 years are required, fail."
        )
    else:
        _fail_leaf_due_to_missing(experience_node)

    # 8) Certification: state admin/superintendent certification or pathway
    cert_node = evaluator.add_leaf(
        id=f"position_{idx+1}_certification",
        desc="Confirms the position requires valid state administrative or superintendent certification",
        parent=pos_node,
        critical=True
    )
    if posting_url:
        claim = ("The position requires a valid state administrative or superintendent certification, "
                 "or explicitly allows eligibility/a pathway to obtain such certification by the start date.")
        await evaluator.verify(
            claim=claim,
            node=cert_node,
            sources=posting_url,
            additional_instruction="Accept language like 'must hold' or 'must be eligible for' the relevant administrator/superintendent "
                                   "license/certification in the state."
        )
    else:
        _fail_leaf_due_to_missing(cert_node)

    # 9) Application portal: publicly accessible online portal
    application_node = evaluator.add_leaf(
        id=f"position_{idx+1}_application",
        desc="Provides the online application portal URL and confirms it is publicly accessible",
        parent=pos_node,
        critical=True
    )
    if portal_url:
        claim = "This URL is a publicly accessible online application portal for this job (or this district's application system for this posting)."
        await evaluator.verify(
            claim=claim,
            node=application_node,
            sources=portal_url,
            additional_instruction="The page should be viewable without special credentials and clearly provide a way to apply."
        )
    else:
        _fail_leaf_due_to_missing(application_node)

    # 10) Enrollment (optional, non-critical) — only verify if provided in the answer
    if _has_text(pos.enrollment):
        enrollment_node = evaluator.add_leaf(
            id=f"position_{idx+1}_enrollment",
            desc="Reports the school district enrollment or confirms district size qualification",
            parent=pos_node,
            critical=False
        )
        enrollment_sources = _source_list(pos.enrollment_source_url, posting_url)
        if len(enrollment_sources) > 0:
            claim = f"The school district's student enrollment is {pos.enrollment} (approximate phrasing acceptable if indicated)."
            await evaluator.verify(
                claim=claim,
                node=enrollment_node,
                sources=enrollment_sources,
                additional_instruction="Accept approximate statements like 'serves approximately X students'. Minor rounding differences are acceptable."
            )
        else:
            # If no usable source is available, fail this soft leaf
            _fail_leaf_due_to_missing(enrollment_node)
    # If enrollment not provided, we intentionally do not create the soft leaf to avoid penalizing for 'if available'.

    # 11) URL reference: official posting/announcement
    ref_node = evaluator.add_leaf(
        id=f"position_{idx+1}_url_reference",
        desc="Provides URL reference to official job posting or announcement",
        parent=pos_node,
        critical=True
    )
    if posting_url:
        claim = "This URL is the official job posting or official announcement for the stated position at the named district."
        await evaluator.verify(
            claim=claim,
            node=ref_node,
            sources=posting_url,
            additional_instruction="The page should be an official district/district HR partner posting or equivalent formal announcement."
        )
    else:
        _fail_leaf_due_to_missing(ref_node)


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for Superintendent/Assistant Superintendent roles meeting strict criteria.
    """
    evaluator = Evaluator()
    # Important: root must be non-critical to avoid the framework's constraint that critical parents require all critical children.
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

    # Record policy thresholds for transparency
    evaluator.add_custom_info(
        info={
            "deadline_window_inclusive": [DEADLINE_START_TEXT, DEADLINE_END_TEXT],
            "expected_start_date": EXPECTED_START_DATE_TEXT,
            "min_salary_threshold": MIN_SALARY_THRESHOLD
        },
        info_type="policy_thresholds"
    )

    # Extract up to 5 positions, but evaluate only the first 3
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction"
    )

    # Normalize and take the first three positions; pad with empty items if fewer found
    positions: List[PositionItem] = list(extracted.positions)[:3]
    while len(positions) < 3:
        positions.append(PositionItem())

    # Build and verify each position subtree
    for i in range(3):
        await verify_position(evaluator, root, positions[i], i)

    # Return evaluation summary
    return evaluator.get_summary()