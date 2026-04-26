import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "psych_tenuretrack_2026"
TASK_DESCRIPTION = (
    "A PhD candidate in Clinical Psychology will complete their doctorate by August 15, 2026 and is seeking tenure-track assistant professor positions. "
    "They require positions with a teaching load of no more than 3 courses per semester, a minimum annual salary of $70,000, and application deadlines that have not yet passed as of March 20, 2026. "
    "Identify 4 different tenure-track assistant professor positions in Clinical or Counseling Psychology at universities in the United States that begin in August or Fall 2026 and meet all these requirements. "
    "For each position, provide the university name, confirm all criteria are satisfied, and include the reference URL for the job posting."
)

CONSTRAINT_DATE_ISO = "2026-03-20"
FALL_START_YEAR = 2026
MIN_SALARY = 70000
MAX_LOAD_PER_SEM = 3
MAX_LOAD_PER_YEAR = 6


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PositionItem(BaseModel):
    university_name: Optional[str] = None
    position_title: Optional[str] = None
    field: Optional[str] = None
    degree_requirements: Optional[str] = None
    teaching_load: Optional[str] = None
    salary: Optional[str] = None
    application_deadline: Optional[str] = None
    start_date: Optional[str] = None
    posting_url: Optional[str] = None


class PositionsExtraction(BaseModel):
    positions: List[PositionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return f"""
Extract up to the first 4 distinct tenure-track Assistant Professor positions listed in the answer (ignore extras beyond the fourth).
For each position, return an object with the following fields (strings; use exactly null if missing):
- university_name: the university's name
- position_title: the full advertised title
- field: the advertised field or area (e.g., "Clinical Psychology", "Counseling Psychology")
- degree_requirements: verbatim requirement text about PhD/ABD timing or equivalent
- teaching_load: verbatim text describing the standard teaching load (e.g., "2-2", "3/3", "2-3 per semester", "9 credit hours/semester")
- salary: verbatim text describing annual salary or salary range
- application_deadline: verbatim text for the application deadline (e.g., "Open until filled", "Priority date Jan 15, 2026", "April 1, 2026")
- start_date: verbatim text describing the anticipated start (e.g., "August 2026", "Fall 2026", "AY 2026-27")
- posting_url: the direct URL to the job posting (must be an actual URL present in the answer)

Rules:
1) Extract exactly what the answer provides; do not fabricate.
2) For URLs, return only actual URLs present in the answer (plain or markdown). If missing, set posting_url to null.
3) If multiple URLs are given for a position, choose the primary job-posting URL (e.g., the official university HR page or Interfolio page).
4) Keep all text as found, including date formats, currency symbols, and load formats.
5) Only include U.S. university positions if the answer clearly indicates that (otherwise still extract, but fields may be null).

Return JSON with a single key "positions" whose value is an array of up to 4 objects in the original order of appearance.
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_position(
    evaluator: Evaluator,
    parent_node,
    pos: PositionItem,
    idx_one_based: int,
) -> None:
    """
    Build verification subtree and run checks for a single position.
    All concrete judging logic is implemented as leaf nodes (binary).
    """

    # Parent node for this position (non-critical to allow partial across 4 positions)
    pos_node = evaluator.add_parallel(
        id=f"position_{idx_one_based}",
        desc=f"Position #{idx_one_based} identified meets all requirements",
        parent=parent_node,
        critical=False,
    )

    # Critical existence checks used to gate source-grounded verifications
    uni_exists_node = evaluator.add_custom_node(
        result=bool(pos.university_name and pos.university_name.strip()),
        id=f"p{idx_one_based}_university_name",
        desc="University name is provided for the position",
        parent=pos_node,
        critical=True,
    )

    url_exists_node = evaluator.add_custom_node(
        result=bool(pos.posting_url and pos.posting_url.strip()),
        id=f"p{idx_one_based}_reference_url",
        desc="Reference URL for the job posting is provided",
        parent=pos_node,
        critical=True,
    )

    # 1) Tenure-track Assistant Professor
    n_type = evaluator.add_leaf(
        id=f"p{idx_one_based}_position_type",
        desc="Position is tenure-track at assistant professor level",
        parent=pos_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This job posting is for a tenure-track (or tenure-eligible/tenure line/tenure-system) Assistant Professor position. Titles like 'Assistant/Associate' still include Assistant. Non-tenure-track (e.g., lecturer/visiting/clinical non-tenure) does not satisfy.",
        node=n_type,
        sources=pos.posting_url,
        additional_instruction="Check both text and screenshot on the page. Look for explicit tenure language and the Assistant Professor rank.",
    )

    # 2) Field is Clinical/Counseling Psychology (or closely related subfield)
    n_field = evaluator.add_leaf(
        id=f"p{idx_one_based}_field",
        desc="Field is Clinical or Counseling Psychology or closely related subfield",
        parent=pos_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This job is in Clinical Psychology or Counseling Psychology, or a closely related subfield within Clinical/Counseling (e.g., Clinical Health Psychology, Clinical Neuropsychology, Clinical-Community). Purely unrelated areas (e.g., Industrial-Organizational, Experimental without clinical/counseling focus) do not satisfy.",
        node=n_field,
        sources=pos.posting_url,
        additional_instruction="Accept reasonable variants like 'Clinical/Counseling', 'Clinical-Community', or 'Health Psychology in a Clinical program'.",
    )

    # 3) Degree timing compatible with Aug/Fall 2026 start (PhD by start or ABD w/ completion by start)
    n_degree = evaluator.add_leaf(
        id=f"p{idx_one_based}_degree",
        desc="PhD or ABD in Psychology accepted with completion timeline compatible with August or Fall 2026 start",
        parent=pos_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The posting states that candidates must have a PhD (or equivalent such as a PsyD, if clearly within Clinical/Counseling Psychology) by the start date, or that ABD candidates are considered if the doctorate is completed by the start date (August or Fall {FALL_START_YEAR}).",
        node=n_degree,
        sources=pos.posting_url,
        additional_instruction="Treat phrases like 'by start of appointment' or 'by August 2026' as compatible. If the page explicitly requires a PhD completed earlier (e.g., by 2025) with no ABD allowance by start, mark incorrect.",
    )

    # 4) Teaching load <= 3/semester or <= 6/year
    n_load = evaluator.add_leaf(
        id=f"p{idx_one_based}_teaching_load",
        desc="Teaching load is 3 courses per semester or less (or 6 courses per academic year or less)",
        parent=pos_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The standard teaching load is at most 3 courses per semester OR at most 6 courses per academic year.",
        node=n_load,
        sources=pos.posting_url,
        additional_instruction=(
            "Accept formats like 2/2, 3/2, 2-2, 3-2, 'two-two', '2–3 per semester' (max 3), or yearly totals ≤ 6. "
            "If only credit hours are given, assume 3 credit hours ≈ 1 course (e.g., 9 credit hours/semester ≈ 3 courses). "
            "If no teaching load info is present, this requirement is not satisfied."
        ),
    )

    # 5) Salary minimum >= $70,000 annually
    n_salary = evaluator.add_leaf(
        id=f"p{idx_one_based}_salary",
        desc="Minimum salary is $70,000 or higher annually",
        parent=pos_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The minimum annual salary (or lower bound of the advertised salary range) is at least ${MIN_SALARY:,}.",
        node=n_salary,
        sources=pos.posting_url,
        additional_instruction=(
            "Look for explicit dollar amounts or ranges (e.g., '$70,000', '$70k', '70,000 USD'). "
            "If only 'commensurate with experience' is given with no numeric minimum, mark incorrect."
        ),
    )

    # 6) Deadline not passed as of 2026-03-20
    n_deadline = evaluator.add_leaf(
        id=f"p{idx_one_based}_deadline",
        desc="Application deadline has not passed as of March 20, 2026",
        parent=pos_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The application deadline has not passed as of March 20, 2026. "
            "This is satisfied if (a) the hard deadline date is on/after 2026-03-20, "
            "(b) the posting states 'Open until filled' or equivalent, or "
            "(c) rolling review continuing until filled with no closed status."
        ),
        node=n_deadline,
        sources=pos.posting_url,
        additional_instruction=(
            "If the only stated deadline is before 2026-03-20 (e.g., 2026-03-01) and there's no 'open until filled' language, mark as passed deadline (incorrect). "
            "If the page shows the search is closed or no longer accepting applications, mark incorrect."
        ),
    )

    # 7) Start date August/Fall 2026
    n_start = evaluator.add_leaf(
        id=f"p{idx_one_based}_start_date",
        desc="Position starts in August or Fall 2026",
        parent=pos_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The position starts in August {FALL_START_YEAR} or Fall {FALL_START_YEAR} (including AY {FALL_START_YEAR}-{FALL_START_YEAR+1}).",
        node=n_start,
        sources=pos.posting_url,
        additional_instruction=(
            "Accept phrases like 'beginning August 2026', 'Fall 2026', or 'Academic Year 2026–27'. "
            "Do not accept Spring 2026, Spring 2027, or Fall 2025."
        ),
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
    Evaluate an answer for the 'psych_tenuretrack_2026' task and return a structured summary.
    """
    # Initialize evaluator (root as parallel aggregation)
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

    # Extract structured positions
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction",
    )

    # Keep only first 4 positions; pad with empty objects if fewer
    positions: List[PositionItem] = list(extracted.positions[:4])
    while len(positions) < 4:
        positions.append(PositionItem())

    # Record constraints as auxiliary info
    evaluator.add_custom_info(
        {
            "min_salary_usd": MIN_SALARY,
            "max_load_per_semester": MAX_LOAD_PER_SEM,
            "max_load_per_academic_year": MAX_LOAD_PER_YEAR,
            "deadline_not_passed_as_of": CONSTRAINT_DATE_ISO,
            "target_start": f"August/Fall {FALL_START_YEAR}",
            "required_positions": 4,
        },
        info_type="constraints",
        info_name="constraint_settings",
    )

    # Build four parallel position nodes
    for idx, pos in enumerate(positions, start=1):
        await verify_position(evaluator, root, pos, idx)

    # Return the evaluation summary
    return evaluator.get_summary()