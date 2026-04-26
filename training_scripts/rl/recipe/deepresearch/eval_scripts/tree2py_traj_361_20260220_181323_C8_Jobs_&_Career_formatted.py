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
TASK_ID = "teacher_positions_4_states_reciprocity_salary_shortage"
TASK_DESCRIPTION = (
    "I am a certified teacher with a valid teaching license from my home state, and I am interested in relocating to pursue new teaching opportunities. "
    "I want to identify four teaching positions in four different U.S. states where I could potentially teach for the 2025-2026 or 2026-2027 school year.\n\n"
    "For each of the four positions, please identify:\n\n"
    "1. State and Certification Reciprocity: The position must be in a U.S. state that either (a) offers full teacher license reciprocity for all eligible, fully licensed teachers, "
    "OR (b) participates in the NASDTEC Interstate Agreement for teacher license reciprocity. Provide a direct URL to the state's official education department page that documents "
    "their certification requirements or reciprocity policy.\n\n"
    "2. Minimum Starting Salary: The starting teacher salary for the position or district must be at least $45,000 per year. "
    "Provide a URL to the district's official salary schedule or state salary data that confirms this information.\n\n"
    "3. Teacher Shortage Area: The position's subject area must be designated as a teacher shortage area by either the federal government or the state for the 2024-2025, 2025-2026, "
    "or 2026-2027 school year. Provide a URL to an official state or federal source that documents this teacher shortage designation.\n\n"
    "4. Position Details: For each position, provide:\n"
    "   - The specific school district name\n"
    "   - The specific subject area or teaching position type (e.g., Special Education, Mathematics, ESL, Bilingual Education)\n"
    "   - The grade level or range (e.g., Elementary K-5, Secondary 6-12, EC-12)\n"
    "   - A direct URL to the job posting or the district's employment/careers page\n\n"
    "All four positions must be in four different states. Each position must meet all four criteria above with appropriate documentation and reference URLs."
)

ALLOWED_SHORTAGE_YEARS = ["2024-2025", "2025-2026", "2026-2027"]

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PositionInfo(BaseModel):
    """Information for a single teaching position as extracted from the answer."""
    state: Optional[str] = None
    # Certification reciprocity / DOE references
    state_cert_urls: List[str] = Field(default_factory=list)
    reciprocity_claim: Optional[str] = None  # e.g., "Participates in NASDTEC" or "allows full reciprocity"
    # Salary
    salary_url: Optional[str] = None
    salary_claim: Optional[str] = None  # e.g., "$47,000 starting salary" or ">= 45k"
    # Shortage area
    shortage_urls: List[str] = Field(default_factory=list)  # official state or federal shortage source(s)
    shortage_claim: Optional[str] = None  # e.g., "Special Education is a shortage area in 2025-2026"
    # Position details
    district_name: Optional[str] = None
    subject_area: Optional[str] = None
    grade_level: Optional[str] = None
    job_url: Optional[str] = None


class PositionsExtraction(BaseModel):
    """Top-level extraction model capturing up to four positions."""
    positions: List[PositionInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    Extract up to four distinct teaching positions mentioned in the answer. For each position, extract the following fields exactly as stated in the answer text:

    Required fields per position (use null for any missing field):
    - state: The U.S. state where the position/district is located (e.g., "Texas").
    - state_cert_urls: An array of URL(s) to the official state education department page(s) documenting certification requirements or reciprocity policy (e.g., DOE certification page, reciprocity page). Only include actual URLs explicitly mentioned in the answer.
    - reciprocity_claim: The text (from the answer) describing how the state meets reciprocity (e.g., "participates in NASDTEC Interstate Agreement" or "offers full teacher license reciprocity").
    - salary_url: The URL to the district's official salary schedule or a state-level official teacher salary data page.
    - salary_claim: The text (from the answer) describing the starting teacher salary for the position or district (e.g., "$48,000 starting salary", "starting salary ≥ $45,000").
    - shortage_urls: An array of URL(s) to official state or federal sources documenting the teacher shortage designation (e.g., state DOE shortage list, U.S. Department of Education shortage tables).
    - shortage_claim: The text (from the answer) describing the shortage area (must be for one of these years: 2024-2025, 2025-2026, or 2026-2027). Include the subject area mentioned (e.g., "Special Education shortage 2025-2026").
    - district_name: The specific school district name.
    - subject_area: The subject area or teaching role (e.g., "Mathematics", "ESL", "Special Education", "Bilingual Education").
    - grade_level: The grade level or range (e.g., "Elementary K-5", "Secondary 6-12", "EC-12").
    - job_url: A direct URL to the job posting or the district's employment/careers page.

    Notes:
    - Extract only the URLs explicitly present in the answer (plain or Markdown). Do not invent URLs.
    - If more than four positions are mentioned, extract the first four in order of appearance.
    - States must be distinct across the four positions (we will verify uniqueness later).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_str(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _has_urls(urls: Optional[List[str]] | None) -> bool:
    return bool(urls and len(urls) > 0 and any(u.strip() for u in urls))


def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if _nonempty_str(u)]


# --------------------------------------------------------------------------- #
# Verification for one position                                               #
# --------------------------------------------------------------------------- #
async def verify_one_position(
    evaluator: Evaluator,
    parent_node,
    position: PositionInfo,
    idx: int,
    previous_states: List[str]
) -> None:
    """
    Build the verification subtree for a single position and perform verifications.
    All leaves are binary checks; existence checks are implemented as custom nodes.
    """
    pos_num = idx + 1
    pos_node = evaluator.add_parallel(
        id=f"position_{pos_num}",
        desc=f"Position #{pos_num}: Meets all criteria in a unique state",
        parent=parent_node,
        critical=True  # Each position is essential: failure of any required part fails the whole task
    )

    # ------------------------ 1) State certification & reciprocity ------------------------ #
    cert_node = evaluator.add_parallel(
        id=f"position_{pos_num}_state_certification",
        desc="State certification requirements and reciprocity verification",
        parent=pos_node,
        critical=True
    )

    # 1.1 State identified (existence)
    evaluator.add_custom_node(
        result=_nonempty_str(position.state),
        id=f"position_{pos_num}_state_identified",
        desc="The U.S. state where the position is located is clearly identified",
        parent=cert_node,
        critical=True
    )

    # 1.2 State uniqueness compared to previous positions (strict check)
    if pos_num >= 2:
        is_unique_so_far = _nonempty_str(position.state) and (position.state.strip().lower() not in [s.strip().lower() for s in previous_states if _nonempty_str(s)])
        evaluator.add_custom_node(
            result=is_unique_so_far,
            id=f"position_{pos_num}_state_unique_so_far",
            desc=f"The position's state is different from earlier position(s)",
            parent=cert_node,
            critical=True
        )

    # 1.3 Certification reference provided (existence)
    state_cert_urls = _safe_urls(position.state_cert_urls)
    evaluator.add_custom_node(
        result=_has_urls(state_cert_urls),
        id=f"position_{pos_num}_cert_reference",
        desc="A direct URL to the state's official education department page documenting certification requirements or reciprocity information is provided",
        parent=cert_node,
        critical=True
    )

    # 1.4 Reciprocity verification (by URLs)
    reciprocity_leaf = evaluator.add_leaf(
        id=f"position_{pos_num}_reciprocity",
        desc="The state either offers full teacher license reciprocity for all eligible, fully licensed teachers, OR has documented reciprocity through the NASDTEC Interstate Agreement",
        parent=cert_node,
        critical=True
    )
    reciprocity_claim = (
        f"The official state education department page(s) indicate that {position.state or 'the state'} either offers full teacher license reciprocity "
        f"for fully licensed teachers or participates in the NASDTEC Interstate Agreement."
    )
    await evaluator.verify(
        claim=reciprocity_claim,
        node=reciprocity_leaf,
        sources=state_cert_urls,
        additional_instruction=(
            "Verify directly from the provided official state education department page(s). "
            "Treat the requirement as satisfied if the page explicitly documents either (a) full reciprocity for eligible, fully licensed teachers, "
            "or (b) participation in the NASDTEC Interstate Agreement. If the URL is not an official DOE/SEA page or does not mention reciprocity/NASDTEC, mark as not supported."
        )
    )

    # ------------------------ 2) Salary ------------------------ #
    salary_node = evaluator.add_parallel(
        id=f"position_{pos_num}_salary",
        desc="Salary information verification",
        parent=pos_node,
        critical=True
    )

    # 2.1 Salary reference provided (existence)
    evaluator.add_custom_node(
        result=_nonempty_str(position.salary_url),
        id=f"position_{pos_num}_salary_reference",
        desc="A URL to the district's official salary schedule or state salary data is provided",
        parent=salary_node,
        critical=True
    )

    # 2.2 Starting salary >= $45,000 verification (by URL)
    salary_leaf = evaluator.add_leaf(
        id=f"position_{pos_num}_starting_salary",
        desc="The starting teacher salary for this position or district is at least $45,000 per year",
        parent=salary_node,
        critical=True
    )
    salary_claim = (
        "The starting teacher salary shown on this page is at least $45,000 per year. "
        "If the page shows multi-lane schedules (e.g., BA/Step 1), check the lowest lane's starting/base step value. "
        "If amounts are monthly or daily, convert appropriately ($45,000 ≈ $3,750/month over 12 months)."
    )
    await evaluator.verify(
        claim=salary_claim,
        node=salary_leaf,
        sources=position.salary_url,
        additional_instruction=(
            "Confirm this page is an official salary schedule or official state/district salary data and that the starting/base salary (e.g., BA Step 0/1) is ≥ $45,000. "
            "If multiple schedules or years are displayed, use the relevant current or upcoming schedule. "
            "If the page does not show salary values clearly, mark as not supported."
        )
    )

    # ------------------------ 3) Shortage area ------------------------ #
    shortage_node = evaluator.add_parallel(
        id=f"position_{pos_num}_shortage_area",
        desc="Teacher shortage area verification",
        parent=pos_node,
        critical=True
    )

    # 3.1 Shortage reference provided (existence)
    shortage_urls = _safe_urls(position.shortage_urls)
    evaluator.add_custom_node(
        result=_has_urls(shortage_urls),
        id=f"position_{pos_num}_shortage_reference",
        desc="A URL to an official state or federal source documenting the teacher shortage designation is provided",
        parent=shortage_node,
        critical=True
    )

    # 3.2 Shortage designation verification (by URLs)
    shortage_leaf = evaluator.add_leaf(
        id=f"position_{pos_num}_shortage_designated",
        desc="The position's subject area is listed as a teacher shortage area by either the federal government or the state for the current or upcoming school year",
        parent=shortage_node,
        critical=True
    )
    shortage_claim = (
        f"The subject area {position.subject_area or 'the subject'} is designated as a teacher shortage area "
        f"by an official state or federal source for one of these school years: {', '.join(ALLOWED_SHORTAGE_YEARS)}."
    )
    await evaluator.verify(
        claim=shortage_claim,
        node=shortage_leaf,
        sources=shortage_urls,
        additional_instruction=(
            "Verify that the page is an official state education agency (SEA/DOE) source or an official federal source (e.g., U.S. Department of Education). "
            "Confirm that the specific subject area (or a clearly equivalent category) appears in the shortage list for any of the allowed years. "
            "If only older years or non-official lists are shown, mark as not supported."
        )
    )

    # ------------------------ 4) Position details ------------------------ #
    details_node = evaluator.add_parallel(
        id=f"position_{pos_num}_details",
        desc="Position-specific details and documentation",
        parent=pos_node,
        critical=True
    )

    # 4.1 Job URL provided (existence)
    evaluator.add_custom_node(
        result=_nonempty_str(position.job_url),
        id=f"position_{pos_num}_url",
        desc="A direct URL to the job posting or district's employment page is provided",
        parent=details_node,
        critical=True
    )

    # 4.2 District name appears on job page
    district_leaf = evaluator.add_leaf(
        id=f"position_{pos_num}_district",
        desc="The specific school district name is provided",
        parent=details_node,
        critical=True
    )
    district_claim = (
        f"The job posting or employment/careers page clearly indicates the district name '{position.district_name}'. "
        "This can appear in the page header, footer, employer field, or job details."
    )
    await evaluator.verify(
        claim=district_claim,
        node=district_leaf,
        sources=position.job_url,
        additional_instruction=(
            "Confirm the page corresponds to the stated district and that the district name appears clearly. "
            "Minor variations in naming (e.g., 'ISD', 'School District', abbreviations) are acceptable if they clearly refer to the same district."
        )
    )

    # 4.3 Subject area appears on job page
    subject_leaf = evaluator.add_leaf(
        id=f"position_{pos_num}_subject",
        desc="The specific subject area or teaching position type is clearly stated",
        parent=details_node,
        critical=True
    )
    subject_claim = (
        f"The job posting or employment page clearly indicates the subject/role '{position.subject_area}'. "
        "It should appear in the job title or description (e.g., Special Education, Mathematics, ESL, Bilingual)."
    )
    await evaluator.verify(
        claim=subject_claim,
        node=subject_leaf,
        sources=position.job_url,
        additional_instruction=(
            "Allow reasonable synonyms (e.g., 'SPED' for Special Education, 'English Learner' for ESL). "
            "If the page does not clearly indicate the subject/role, mark as not supported."
        )
    )

    # 4.4 Grade level appears on job page
    grade_leaf = evaluator.add_leaf(
        id=f"position_{pos_num}_grade_level",
        desc="The grade level or range is specified",
        parent=details_node,
        critical=True
    )
    grade_claim = (
        f"The job posting or employment page clearly indicates the grade level/range '{position.grade_level}'. "
        "Examples: Elementary K-5, Secondary 6-12, EC-12."
    )
    await evaluator.verify(
        claim=grade_claim,
        node=grade_leaf,
        sources=position.job_url,
        additional_instruction=(
            "Minor formatting differences are acceptable (e.g., 'Grades K–5', 'K-05'). "
            "If grade level/range is not clearly indicated, mark as not supported."
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
    Evaluate an answer for the 'four teaching positions in four different states' task.
    """
    # Initialize evaluator (root is parallel; set non-critical to allow critical children)
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

    # Record accepted shortage years
    evaluator.add_ground_truth({
        "allowed_shortage_years": ALLOWED_SHORTAGE_YEARS,
        "requirement_summary": "Each of four positions must meet reciprocity (or NASDTEC membership), starting salary >= $45,000, subject shortage designation, and provide full position details with URLs. All in different states."
    })

    # Extract positions from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction"
    )

    # Normalize count to exactly four positions (pad with empty objects if fewer, take first four if more)
    positions: List[PositionInfo] = list(extracted.positions[:4])
    while len(positions) < 4:
        positions.append(PositionInfo())

    # Build position subtrees
    previous_states: List[str] = []
    for idx, pos in enumerate(positions):
        await verify_one_position(evaluator, root, pos, idx, previous_states)
        # Update state list for uniqueness checks between positions
        if _nonempty_str(pos.state):
            previous_states.append(pos.state)

    # Global constraint: all four states are distinct (strict check).
    global_node = evaluator.add_parallel(
        id="global_constraints",
        desc="Global constraints across positions",
        parent=root,
        critical=True
    )
    states_all = [p.state for p in positions]
    distinct_states = [s.strip().lower() for s in states_all if _nonempty_str(s)]
    all_four_distinct = (len(distinct_states) == 4) and (len(set(distinct_states)) == 4)

    evaluator.add_custom_node(
        result=all_four_distinct,
        id="distinct_states",
        desc="All four positions are in four different U.S. states",
        parent=global_node,
        critical=True
    )

    # Return structured summary
    return evaluator.get_summary()