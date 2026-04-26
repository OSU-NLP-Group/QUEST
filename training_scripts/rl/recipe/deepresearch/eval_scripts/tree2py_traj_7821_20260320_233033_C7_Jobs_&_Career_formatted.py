import asyncio
import logging
from typing import Optional, List, Dict, Any, Set

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "faculty_positions_oh_pa_nj"
TASK_DESCRIPTION = """
I am searching for faculty positions in the academic job market and would like to identify three (3) opportunities that meet my specific criteria. Please find three distinct faculty positions that satisfy ALL of the following requirements:

1. The position must be at a university located in Ohio, Pennsylvania, or New Jersey.
2. The position must be in one of these departments: Computer Science, Education (including specializations such as Teacher Education, Early Childhood Education, etc.), or Social Work.
3. The position must be at the Assistant Professor or Lecturer rank.
4. The job posting must include an explicitly stated application deadline date (positions listed as "open until filled" or with no deadline do not qualify).
5. The job posting must explicitly list the required application materials (such as CV, cover letter, teaching statement, research statement, letters of reference, etc.).
6. The position must have a start date in 2026 or later.

For each of the three positions, please provide:
- The university name
- The department/unit name
- The position title
- The application deadline
- The list of required application materials as stated in the posting
- The start date (or anticipated start semester/year)
- A direct URL link to the official job posting
"""

ALLOWED_STATES = ["Ohio", "Pennsylvania", "New Jersey"]
ALLOWED_STATE_ABBR = ["OH", "PA", "NJ"]
ALLOWED_DEPARTMENT_FAMILIES = [
    "Computer Science",
    "Education",
    "Social Work"
]
ALLOWED_RANKS = [
    "Assistant Professor",
    "Lecturer"
]
START_YEAR_THRESHOLD = 2026

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PositionItem(BaseModel):
    university: Optional[str] = None
    department: Optional[str] = None
    position_title: Optional[str] = None
    application_deadline: Optional[str] = None  # Keep as string to maximize compatibility (e.g., "March 15, 2026")
    required_materials: List[str] = Field(default_factory=list)
    start_date: Optional[str] = None  # e.g., "Fall 2026", "August 2026", "2027"
    job_url: Optional[str] = None


class PositionsExtraction(BaseModel):
    positions: List[PositionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    Extract up to five distinct faculty job positions as presented in the answer. For each position, extract the following fields:

    - university: The full university name.
    - department: The department or unit name (e.g., Computer Science, School of Education, Department of Social Work).
    - position_title: The exact position title as stated.
    - application_deadline: A concrete calendar date string for the application deadline (e.g., "March 15, 2026" or "03/15/2026").
        • If the answer says only "open until filled" or similar without a concrete date, set this field to null.
    - required_materials: An array of the explicitly listed application materials (e.g., "CV", "cover letter", "teaching statement", "research statement", "letters of recommendation", "contact information for 3 references"). Do not invent items. If the answer does not enumerate materials, return an empty array.
    - start_date: The start date or anticipated start term/year exactly as stated in the answer (e.g., "Fall 2026", "August 2026", "2027"). If missing, set to null.
    - job_url: A direct URL to the official job posting (university or official jobs site). If missing, set to null.

    Rules:
    1) Extract only what is explicitly present in the answer. Do not infer or add missing information.
    2) Normalize URLs to include protocol (http/https). Ignore obviously invalid URLs.
    3) For 'required_materials', provide a clean list of items; split combined text into distinct items when clearly enumerated.
    4) If any field is missing for a position, set it to null (or [] for 'required_materials').

    Return a JSON object with a single key:
    {
      "positions": [ ... up to 5 PositionItem objects ... ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third"}
    return mapping.get(n, f"Position {n}")


def _materials_to_str(materials: List[str]) -> str:
    if not materials:
        return ""
    return "; ".join([m.strip() for m in materials if m and m.strip()])


def _has_complete_info(p: PositionItem) -> bool:
    return (
        (p.university is not None and p.university.strip() != "") and
        (p.department is not None and p.department.strip() != "") and
        (p.position_title is not None and p.position_title.strip() != "") and
        (p.application_deadline is not None and p.application_deadline.strip() != "") and
        (p.required_materials is not None and len([m for m in p.required_materials if m and m.strip()]) > 0) and
        (p.start_date is not None and p.start_date.strip() != "") and
        (p.job_url is not None and p.job_url.strip() != "")
    )


def _dedup_positions_by_url(items: List[PositionItem], k: int = 3) -> List[PositionItem]:
    seen: Set[str] = set()
    result: List[PositionItem] = []
    for p in items:
        url = (p.job_url or "").strip().lower()
        key = url if url != "" else f"__missing_url__#{len(result)}"
        if key in seen:
            continue
        seen.add(key)
        result.append(p)
        if len(result) >= k:
            break
    return result


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_position(
    evaluator: Evaluator,
    parent_node,
    pos: PositionItem,
    index: int
) -> None:
    """
    Build verification subtree and run checks for a single position.
    The subtree structure mirrors the rubric: a parallel node containing seven critical leaves.
    The 'Complete Information' check is added first and marked as critical so subsequent leaves
    can be auto-skipped if it fails.
    """
    nth = ordinal(index + 1)
    pos_node = evaluator.add_parallel(
        id=f"Position_{index + 1}",
        desc=f"{nth} faculty position meets all requirements and includes complete information",
        parent=parent_node,
        critical=False
    )

    # Critical existence/coverage check first (to gate subsequent leaves if missing info)
    complete_info_result = _has_complete_info(pos)
    evaluator.add_custom_node(
        result=complete_info_result,
        id=f"Position_{index + 1}_Complete_Information",
        desc=f"{nth} position answer includes all required information: university name, department/unit name, position title, application deadline, required materials list, start date, and job posting URL",
        parent=pos_node,
        critical=True
    )

    # Prepare common fields
    job_url = (pos.job_url or "").strip()
    dept = (pos.department or "").strip()
    uni = (pos.university or "").strip()
    title = (pos.position_title or "").strip()
    deadline = (pos.application_deadline or "").strip()
    start_date = (pos.start_date or "").strip()
    materials_str = _materials_to_str(pos.required_materials)

    # 1) University Location (OH/PA/NJ)
    loc_leaf = evaluator.add_leaf(
        id=f"Position_{index + 1}_University_Location",
        desc=f"{nth} position is at a university in Ohio, Pennsylvania, or New Jersey",
        parent=pos_node,
        critical=True
    )
    loc_claim = (
        "This job posting page shows that the position's institution/campus is located in one of these U.S. states: "
        "Ohio (OH), Pennsylvania (PA), or New Jersey (NJ). The webpage itself should display a location within OH/PA/NJ "
        "(e.g., a city and state like 'Columbus, OH' or 'New Brunswick, NJ', or a clear state name/abbreviation)."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=job_url,
        additional_instruction=(
            "Rely only on information present on the job posting page (text or screenshot). "
            "Accept the claim if the page explicitly mentions a location in OH, PA, or NJ (city+state, or state name/abbrev). "
            "If the page lacks any clear OH/PA/NJ location cue, mark as not supported."
        ),
    )

    # 2) Department family check (CS / Education / Social Work)
    dept_leaf = evaluator.add_leaf(
        id=f"Position_{index + 1}_Department",
        desc=f"{nth} position is in Computer Science, Education, or Social Work department",
        parent=pos_node,
        critical=True
    )
    dept_claim = (
        f"The job posting indicates the department/unit for this role is '{dept}' (or an equivalent name), and this "
        "falls within one of these allowed families: "
        "• Computer Science / Computing / Computer and Information Science / Computer Science & Engineering / School of Computing, "
        "• Education (including Teacher Education, Early Childhood Education, Curriculum & Instruction, Educational Studies, Special Education, etc.), "
        "• Social Work (School/College/Department of Social Work)."
    )
    await evaluator.verify(
        claim=dept_claim,
        node=dept_leaf,
        sources=job_url,
        additional_instruction=(
            "Verify from the posting that the role belongs to one of the three allowed department families. "
            "Allow reasonable synonyms listed in the claim. The department/unit may be stated near the title, "
            "in headers, or within the description."
        ),
    )

    # 3) Rank check (Assistant Professor or Lecturer)
    rank_leaf = evaluator.add_leaf(
        id=f"Position_{index + 1}_Rank",
        desc=f"{nth} position is an Assistant Professor or Lecturer position",
        parent=pos_node,
        critical=True
    )
    rank_claim = (
        f"The job posting indicates that the rank/title corresponds to Assistant Professor or Lecturer. "
        f"The title shown in the answer is '{title}'. Acceptable variants include 'Assistant Professor' "
        "(any track, such as tenure-track/tenure-eligible, clinical, research, or visiting) and 'Lecturer' "
        "(including 'Senior Lecturer' or similar lecturer levels). Titles like 'Assistant Teaching Professor' "
        "do not count as Assistant Professor."
    )
    await evaluator.verify(
        claim=rank_claim,
        node=rank_leaf,
        sources=job_url,
        additional_instruction=(
            "Confirm the page clearly uses 'Assistant Professor' (any track) or 'Lecturer' (any level). "
            "If the title is 'Assistant Teaching Professor' or 'Teaching Professor' without 'Lecturer', do NOT accept."
        ),
    )

    # 4) Application deadline: explicit concrete date (not 'open until filled')
    deadline_leaf = evaluator.add_leaf(
        id=f"Position_{index + 1}_Application_Deadline",
        desc=f"{nth} position has an explicitly stated application deadline date (not 'open until filled')",
        parent=pos_node,
        critical=True
    )
    deadline_claim = (
        f"The job posting explicitly states a concrete application deadline date, which is '{deadline}'. "
        "Dates like 'March 15, 2026', '03/15/2026', or '15 March 2026' qualify. Phrases like 'open until filled' "
        "or 'applications reviewed on a rolling basis' without a concrete date do NOT qualify."
    )
    await evaluator.verify(
        claim=deadline_claim,
        node=deadline_leaf,
        sources=job_url,
        additional_instruction=(
            "Accept if a specific calendar date for the deadline appears anywhere on the page (including phrases like "
            "'priority deadline' or 'full consideration by' followed by a concrete date). Reject if only 'open until filled' or no date is present."
        ),
    )

    # 5) Required application materials explicitly listed
    materials_leaf = evaluator.add_leaf(
        id=f"Position_{index + 1}_Required_Materials",
        desc=f"{nth} position listing explicitly specifies required application materials",
        parent=pos_node,
        critical=True
    )
    materials_claim = (
        "The job posting explicitly lists the required application materials for applying. "
        f"The answer enumerates the materials as: [{materials_str}]. "
        "Accept if the posting presents an explicit list/bullets/paragraph indicating required materials (e.g., "
        "CV/resume, cover letter/letter of interest, teaching statement/statement of teaching philosophy, research statement, "
        "diversity statement, references/contact information for referees, letters of recommendation, transcripts, etc.). "
        "Synonyms or minor phrasing differences are acceptable as long as the items are explicitly required."
    )
    await evaluator.verify(
        claim=materials_claim,
        node=materials_leaf,
        sources=job_url,
        additional_instruction=(
            "Look for a clearly enumerated set of required application materials. "
            "Allow synonyms (e.g., 'CV' vs 'curriculum vitae'; 'cover letter' vs 'letter of interest'). "
            "If the posting does not explicitly list materials, do NOT accept."
        ),
    )

    # 6) Start date in 2026 or later
    start_leaf = evaluator.add_leaf(
        id=f"Position_{index + 1}_Start_Date",
        desc=f"{nth} position has a start date in 2026 or later",
        parent=pos_node,
        critical=True
    )
    start_claim = (
        f"The job posting indicates a start date of '{start_date}' and that the start is in {START_YEAR_THRESHOLD} or later. "
        "Accept if the page clearly states a year of 2026 or beyond, or a term/semester that unmistakably corresponds to 2026 or later "
        "(e.g., 'Fall 2026', 'August 2026', 'Spring 2027'). Reject if the start date is earlier than 2026 or not stated."
    )
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=job_url,
        additional_instruction=(
            f"Focus on evidence on the page. Accept explicit years >= {START_YEAR_THRESHOLD} (e.g., '2026', '2027') or "
            "terms that map to those years. If absent or earlier than 2026, mark as not supported."
        ),
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
) -> Dict:
    """
    Evaluate an answer for the faculty position search task (OH/PA/NJ; CS/Education/Social Work; Assistant Professor or Lecturer; explicit deadline; required materials; start date >= 2026).
    """
    # Initialize evaluator with a parallel root (three positions evaluated independently)
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

    # Record high-level ground-truth constraints for transparency
    evaluator.add_ground_truth({
        "allowed_states": ALLOWED_STATES,
        "allowed_state_abbreviations": ALLOWED_STATE_ABBR,
        "allowed_department_families": ALLOWED_DEPARTMENT_FAMILIES,
        "allowed_ranks": ALLOWED_RANKS,
        "min_start_year": START_YEAR_THRESHOLD
    }, gt_type="constraints")

    # Extract structured positions from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction"
    )

    # Prepare up to 3 positions (deduplicate by URL, keep first k)
    raw_positions = extracted.positions or []
    filtered_positions = _dedup_positions_by_url(raw_positions, k=3)

    # Pad to exactly 3
    while len(filtered_positions) < 3:
        filtered_positions.append(PositionItem())

    # Build verification subtrees for three positions
    for i in range(3):
        await verify_position(
            evaluator=evaluator,
            parent_node=root,
            pos=filtered_positions[i],
            index=i
        )

    # Return summary
    return evaluator.get_summary()