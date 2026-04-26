import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "edu_jobs_ca_or_2026_deadlines"
TASK_DESCRIPTION = """
Identify two distinct job positions in the education, academic instruction, or school/activities administration sector that meet ALL of the following criteria:

1. The position must be located in or associated with an organization based in either California or Oregon.
2. The application deadline must fall between January 1, 2026 and April 30, 2026 (inclusive).
3. The position must require an advanced degree (Master's or Doctorate) as a minimum qualification, OR a Bachelor's degree combined with at least 5 years of relevant professional experience.
4. The position must explicitly require at least one year of relevant professional experience.
5. The job posting must provide a specific email address for application submission.
6. The position must be in the education, academic instruction, or school/activities administration sector.

For each of the two positions, provide:
- The organization or institution name
- The exact position title
- The specific application deadline
- The email address for application submission
- A reference URL to the official job posting
"""

DATE_RANGE_START = datetime(2026, 1, 1)
DATE_RANGE_END = datetime(2026, 4, 30)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class JobPosition(BaseModel):
    organization_name: Optional[str] = None
    position_title: Optional[str] = None
    application_deadline: Optional[str] = None  # Keep as string for flexibility
    application_email: Optional[str] = None
    reference_url: Optional[str] = None

    # Helpful context snippets (from the answer) to assist verification
    location_text: Optional[str] = None
    sector_text: Optional[str] = None
    degree_requirement_text: Optional[str] = None
    experience_requirement_text: Optional[str] = None


class PositionsExtraction(BaseModel):
    positions: List[JobPosition] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    From the provided answer, extract up to TWO job positions (in order of appearance) that the answer proposes.
    For each job, return the following fields exactly as stated in the answer:
    - organization_name: The organization or institution name.
    - position_title: The exact position title.
    - application_deadline: The specific deadline date as written (e.g., 'April 1, 2026', '2026-04-01', '04/01/2026').
    - application_email: The specific email address for application submission (not just a generic HR email; it must be the one to send applications to, if provided).
    - reference_url: The URL to the official job posting.
    - location_text: Any location information mentioned (city/state or statement that the org is in CA/OR).
    - sector_text: Any text in the answer indicating this is in the education/academic instruction/school or activities administration sector.
    - degree_requirement_text: Any text in the answer describing the degree qualification requirement.
    - experience_requirement_text: Any text in the answer describing required years of professional experience.

    Important:
    - Only extract jobs explicitly mentioned in the answer.
    - If any field is missing for a job, set it to null.
    - Ensure URLs are captured in full (include http/https).
    - If there are more than two jobs, include only the first two.
    """


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _strip_ordinals(s: str) -> str:
    return re.sub(r'(\d+)(st|nd|rd|th)', r'\1', s, flags=re.IGNORECASE)


def _try_strptime(date_str: str, fmts: List[str]) -> Optional[datetime]:
    for f in fmts:
        try:
            return datetime.strptime(date_str, f)
        except Exception:
            continue
    return None


def parse_date_string(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    s = date_str.strip()
    if not s:
        return None
    s = _strip_ordinals(s)

    # Common formats
    fmts = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%m-%d-%Y",
        "%Y/%m/%d",
        "%b %d, %Y",
        "%B %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d %Y",
        "%b %d %Y",
    ]
    dt = _try_strptime(s, fmts)
    if dt:
        return dt

    # Heuristic: month name + day + year (order-insensitive)
    # Examples: "April 30, 2026", "30 April 2026"
    tokens = re.split(r'[\s,]+', s.lower())
    year = None
    month = None
    day = None
    for t in tokens:
        if t.isdigit():
            num = int(t)
            if 1900 <= num <= 2100:
                year = num
            elif 1 <= num <= 31 and day is None:
                day = num
        else:
            if t in _MONTHS and month is None:
                month = _MONTHS[t]
    if year and month and day:
        try:
            return datetime(year, month, day)
        except Exception:
            return None

    # Fallback: detect yyyy-mm or mm/yyyy when day is missing (assume last day of month if April, else first)
    m1 = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', s)
    if m1:
        y, m, d = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
        try:
            return datetime(y, m, d)
        except Exception:
            return None
    m2 = re.search(r'(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})', s)
    if m2:
        m, d, y = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        if y < 100:
            y += 2000
        try:
            return datetime(y, m, d)
        except Exception:
            return None

    return None


def is_in_range_inclusive(dt: Optional[datetime], start: datetime, end: datetime) -> bool:
    if dt is None:
        return False
    return start <= dt <= end


def position_label(idx1_based: int) -> str:
    return "First" if idx1_based == 1 else "Second"


# --------------------------------------------------------------------------- #
# Verification for one position                                               #
# --------------------------------------------------------------------------- #
async def verify_position(
    evaluator: Evaluator,
    parent_node,
    pos: JobPosition,
    idx1_based: int,
) -> None:
    # Position node (non-critical to allow partial credit across positions)
    pos_node = evaluator.add_parallel(
        id=f"position_{idx1_based}",
        desc=f"{position_label(idx1_based)} qualifying job position identified with all required information",
        parent=parent_node,
        critical=False,
    )

    # 1) Position Identification (critical, parallel)
    ident_node = evaluator.add_parallel(
        id=f"position_{idx1_based}_identification",
        desc="Basic identifying information for the position is provided",
        parent=pos_node,
        critical=True,
    )

    # 1.a) Organization name provided (existence check)
    evaluator.add_custom_node(
        result=bool(pos.organization_name and pos.organization_name.strip()),
        id=f"position_{idx1_based}_organization_name",
        desc="The organization or institution name is provided",
        parent=ident_node,
        critical=True,
    )

    # 1.b) Position title provided (existence check)
    evaluator.add_custom_node(
        result=bool(pos.position_title and pos.position_title.strip()),
        id=f"position_{idx1_based}_position_title",
        desc="The exact position title is provided",
        parent=ident_node,
        critical=True,
    )

    # 1.c) Reference URL validity (verify page is an official job posting)
    ref_leaf = evaluator.add_leaf(
        id=f"position_{idx1_based}_reference_url",
        desc="Valid URL reference to the official job posting",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This webpage is an official job posting or position announcement (not a general careers homepage). It contains position details such as title, qualifications, responsibilities, and application instructions.",
        node=ref_leaf,
        sources=pos.reference_url,
        additional_instruction="Accept HR/ATS postings, official institution job pages, or official PDFs that clearly present a single position with qualifications and how to apply.",
    )

    # 2) Qualification Requirements (critical, parallel)
    qual_node = evaluator.add_parallel(
        id=f"position_{idx1_based}_qualification_requirements",
        desc="Position meets the degree and experience requirements",
        parent=pos_node,
        critical=True,
    )

    # 2.a) Degree requirement (advanced degree OR bachelor's + 5 years)
    degree_leaf = evaluator.add_leaf(
        id=f"position_{idx1_based}_degree_requirement",
        desc="Position requires an advanced degree (Master's or Doctorate) as minimum qualification, OR Bachelor's degree with at least 5 years of relevant experience",
        parent=qual_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The posting explicitly requires either (a) an advanced degree (Master's or Doctorate/PhD) OR (b) a Bachelor's degree together with at least 5 years of relevant professional experience.",
        node=degree_leaf,
        sources=pos.reference_url,
        additional_instruction="Check the Minimum Qualifications/Education section. Accept synonyms (e.g., MS/MA/M.Ed./EdD/PhD; '5+ years', 'five years'). If the posting lists multiple pathways, it's sufficient that at least one explicitly matches the described rule.",
    )

    # 2.b) Experience requirement (at least 1 year explicitly required)
    exp_leaf = evaluator.add_leaf(
        id=f"position_{idx1_based}_experience_requirement",
        desc="Position requires at least one year of relevant professional experience as stated in the job posting",
        parent=qual_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The posting explicitly requires at least one year of relevant professional experience.",
        node=exp_leaf,
        sources=pos.reference_url,
        additional_instruction="Look for phrases like 'at least 1 year', 'minimum of one year', '1+ years', or '12 months'. If the minimum is 2+ years, it still satisfies 'at least one year'.",
    )

    # 3) Context Requirements (critical, parallel)
    ctx_node = evaluator.add_parallel(
        id=f"position_{idx1_based}_context_requirements",
        desc="Position meets location and sector requirements",
        parent=pos_node,
        critical=True,
    )

    # 3.a) Location verification (CA or OR)
    loc_leaf = evaluator.add_leaf(
        id=f"position_{idx1_based}_location_verification",
        desc="Position is located in or associated with an organization based in California or Oregon",
        parent=ctx_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This job posting is for a position located in California or Oregon, or for an organization that is based in California or Oregon.",
        node=loc_leaf,
        sources=pos.reference_url,
        additional_instruction="Accept explicit city/state mentions (e.g., 'CA', 'California', 'OR', 'Oregon'), university/school campuses in CA/OR, or statements that the organization is based in CA/OR. Remote roles still qualify if the institution is based in CA/OR.",
    )

    # 3.b) Sector classification (education/school/activities admin)
    sector_leaf = evaluator.add_leaf(
        id=f"position_{idx1_based}_sector_classification",
        desc="Position is in the education, academic instruction, or school/activities administration sector",
        parent=ctx_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This job is in the education sector (academic instruction or school/activities administration).",
        node=sector_leaf,
        sources=pos.reference_url,
        additional_instruction="Accept roles like teacher, professor, instructor, lecturer, dean, principal, registrar, admissions, academic advisor, coach, school/district/college administrative positions, or similar educational services roles.",
    )

    # 4) Application details (critical, parallel)
    app_node = evaluator.add_parallel(
        id=f"position_{idx1_based}_application_details",
        desc="Application deadline and contact information meet requirements",
        parent=pos_node,
        critical=True,
    )

    # 4.a) Deadline verification (Jan 1, 2026 to Apr 30, 2026 inclusive) — range check via custom node
    parsed_deadline = parse_date_string(pos.application_deadline)
    in_range = is_in_range_inclusive(parsed_deadline, DATE_RANGE_START, DATE_RANGE_END)
    evaluator.add_custom_node(
        result=in_range,
        id=f"position_{idx1_based}_deadline_verification",
        desc="Application deadline falls between January 1, 2026 and April 30, 2026 (inclusive)",
        parent=app_node,
        critical=True,
    )

    # 4.b) Contact information: specific application submission email present and matches page
    email_leaf = evaluator.add_leaf(
        id=f"position_{idx1_based}_contact_information",
        desc="Job posting provides a specific email address for application submission",
        parent=app_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The job posting instructs applicants to submit applications to the email address '{pos.application_email}'.",
        node=email_leaf,
        sources=pos.reference_url,
        additional_instruction="Confirm that the email shown is specifically for submitting applications/materials (e.g., 'Apply by emailing ...', 'Send application to ...'). Reject if only a generic HR contact with no instruction to submit applications.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    # Initialize evaluator (root: parallel aggregation across the two positions)
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

    # Extract up to two positions from the answer
    extracted: PositionsExtraction = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction",
    )

    # Normalize to exactly two entries (pad with empty if needed)
    positions: List[JobPosition] = (extracted.positions or [])[:2]
    while len(positions) < 2:
        positions.append(JobPosition())

    # Record custom info about date range to aid debugging
    evaluator.add_custom_info(
        info={
            "deadline_range_start": DATE_RANGE_START.strftime("%Y-%m-%d"),
            "deadline_range_end": DATE_RANGE_END.strftime("%Y-%m-%d"),
        },
        info_type="date_range",
        info_name="deadline_requirements",
    )

    # Build verification subtrees for two positions (in parallel at root)
    await verify_position(evaluator, root, positions[0], 1)
    await verify_position(evaluator, root, positions[1], 2)

    # Return structured summary
    return evaluator.get_summary()