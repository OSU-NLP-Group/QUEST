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
TASK_ID = "stem_tt_positions_fall_2026"
TASK_DESCRIPTION = (
    "I am completing my PhD in a STEM field and planning to apply for tenure-track faculty positions starting in Fall 2026. "
    "I want to identify opportunities at different universities across the United States. Find four tenure-track assistant professor positions in STEM fields (Science, Technology, Engineering, or Mathematics), each at a different U.S. university, that are currently accepting applications for Fall 2026 start dates. For each position, provide:\n\n"
    "1. The university name and department\n"
    "2. The specific field or area of specialization\n"
    "3. Confirmation that it is a tenure-track position at the assistant professor rank\n"
    "4. The PhD/doctorate requirement and expected completion date\n"
    "5. A list of the required application materials (such as CV, cover letter, research statement, teaching statement, reference letters)\n"
    "6. The URL to the official job posting"
)
CURRENT_DATE_STR = "2026-02-26"  # Used by the judge model for temporal checks


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Position(BaseModel):
    university: Optional[str] = None
    department: Optional[str] = None
    field_specialization: Optional[str] = None
    rank_text: Optional[str] = None  # e.g., "Assistant Professor", or "Open rank (Assistant/Associate)"
    tenure_track_phrase: Optional[str] = None  # text indicating tenure-track
    degree_requirement: Optional[str] = None  # e.g., "PhD required by start date"
    expected_completion_date: Optional[str] = None  # e.g., "by August 2026"
    start_date: Optional[str] = None  # e.g., "Fall 2026", or "August 2026"
    application_materials: List[str] = Field(default_factory=list)  # Canonical names if possible
    job_posting_url: Optional[str] = None
    location_country: Optional[str] = None  # e.g., "United States"
    location_state_or_city: Optional[str] = None  # e.g., "CA", "San Diego, CA"
    accepting_applications_phrase: Optional[str] = None  # e.g., "Open until filled", "Applications accepted until ..."


class PositionsExtraction(BaseModel):
    positions: List[Position] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    Extract up to 8 tenure-track faculty positions described in the answer, focusing on STEM fields and Fall 2026 start. 
    For each position, return an object with the following fields (use null when missing):

    - university: University name (e.g., "University of X")
    - department: Department or school/college (e.g., "Department of Computer Science")
    - field_specialization: Specific field or area (e.g., "Machine Learning", "Electrical Engineering", "Statistics")
    - rank_text: The rank text (e.g., "Assistant Professor", "Open-rank (Assistant/Associate/Full)")
    - tenure_track_phrase: The exact phrase indicating tenure-track (e.g., "tenure-track", "tenure eligible")
    - degree_requirement: Text about the PhD/doctorate requirement (e.g., "PhD required by start date")
    - expected_completion_date: Any stated expected completion timing (e.g., "by August 2026")
    - start_date: Stated start date (e.g., "Fall 2026", "August 2026", "September 2026")
    - application_materials: List of required materials; normalize common variants to canonical names:
        * "CV" (accept "curriculum vitae", "resume")
        * "Cover Letter" (accept "letter of interest/intent")
        * "Research Statement" (accept "research plan/proposal")
        * "Teaching Statement" (accept "teaching philosophy")
        * "References" (accept "reference letters", "names and contact info of referees")
      Include only materials explicitly mentioned in the answer.
    - job_posting_url: The URL of the official posting (university HR page, Interfolio, AcademicJobsOnline, Workday, iCIMS, PeopleSoft, or a university career site). If multiple, prefer the primary application page.
    - location_country: Country (e.g., "United States", "USA")
    - location_state_or_city: If available, the state or city (e.g., "CA", "Pittsburgh, PA")
    - accepting_applications_phrase: Any indication the search is open (e.g., "Open until filled", "review begins ...", "apply link")

    IMPORTANT:
    - Extract only what appears in the answer text; do not invent missing fields.
    - Preserve textual fields exactly; use null for missing.
    - application_materials should be an array with the canonical names above when possible.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _first_n_positions(extraction: PositionsExtraction, n: int = 4) -> List[Position]:
    items = extraction.positions[:n]
    while len(items) < n:
        items.append(Position())
    return items


def _positions_count_with_urls(positions: List[Position]) -> int:
    return sum(1 for p in positions if p.job_posting_url and p.job_posting_url.strip())


def _all_universities_unique(positions: List[Position]) -> bool:
    names = [p.university.strip() for p in positions if p.university and p.university.strip()]
    return len(names) == 4 and len(set(n.lower() for n in names)) == 4


# --------------------------------------------------------------------------- #
# Verification functions per-position                                         #
# --------------------------------------------------------------------------- #
async def verify_position(
    evaluator: Evaluator,
    parent_node,
    pos: Position,
    idx: int,
) -> None:
    """
    Build verification subtree for a single position.
    We verify the URL first (critical sibling), so subsequent verifications can auto-skip if URL is invalid.
    """
    # Create a parallel node for this position (non-critical so parent can average across positions)
    position_node = evaluator.add_parallel(
        id=f"position_{idx+1}",
        desc=(
            f"{['First','Second','Third','Fourth'][idx]} tenure-track assistant professor position "
            f"in a STEM field at a U.S. university"
        ),
        parent=parent_node,
        critical=False
    )

    # 0) University + Department existence check (critical custom)
    uni_dept_ok = bool(pos.university and pos.university.strip()) and bool(pos.department and pos.department.strip())
    evaluator.add_custom_node(
        result=uni_dept_ok,
        id=f"university_department_{idx+1}",
        desc=f"University name and department are provided (position #{idx+1})",
        parent=position_node,
        critical=True
    )

    # 1) URL leaf (critical) – verify it is a valid posting page
    url_leaf = evaluator.add_leaf(
        id=f"reference_url_{idx+1}",
        desc=f"Valid URL provided to official university job posting (position #{idx+1})",
        parent=position_node,
        critical=True
    )
    url_claim = (
        f"This URL is an official job posting page for a faculty position at {pos.university or 'the stated university'} "
        f"in {pos.department or 'the stated department'}."
    )
    await evaluator.verify(
        claim=url_claim,
        node=url_leaf,
        sources=pos.job_posting_url,
        additional_instruction=(
            "Treat official HR/careers systems as valid (e.g., Interfolio, AcademicJobsOnline, Workday, iCIMS, PeopleSoft, "
            "university careers portals). The page should clearly be a job posting, not a generic homepage."
        ),
    )

    # 2) STEM field check (critical)
    field_leaf = evaluator.add_leaf(
        id=f"field_match_{idx+1}",
        desc=f"Position is in a STEM field (Science/Technology/Engineering/Mathematics) (position #{idx+1})",
        parent=position_node,
        critical=True
    )
    field_claim = (
        "This job posting is for a STEM field. STEM includes disciplines such as Computer Science, Data Science, "
        "Statistics, Mathematics, Applied Mathematics, Physics, Chemistry, Biology, Earth/Environmental Sciences, "
        "Astronomy, Engineering disciplines (Electrical, Mechanical, Civil, Chemical, Aerospace, Materials, Biomedical), "
        "and clearly STEM-anchored areas like AI, Robotics, Machine Learning."
    )
    await evaluator.verify(
        claim=field_claim,
        node=field_leaf,
        sources=pos.job_posting_url,
        additional_instruction=(
            "Use the department and field descriptions on the page to decide if it is STEM. "
            "Reject primarily non-STEM fields (e.g., Business, Law, History, Philosophy, most Social Sciences), "
            "unless the posting explicitly anchors the field in STEM (e.g., Computational Biology under a STEM department)."
        ),
    )

    # 3) Position type: tenure-track + assistant rank (critical)
    type_leaf = evaluator.add_leaf(
        id=f"position_type_{idx+1}",
        desc=f"Position is explicitly tenure-track at assistant professor rank (or open-rank including assistant) (position #{idx+1})",
        parent=position_node,
        critical=True
    )
    type_claim = (
        "This posting explicitly indicates a tenure-track position at the Assistant Professor rank, "
        "or it is an open-rank search that includes the Assistant Professor level."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=pos.job_posting_url,
        additional_instruction=(
            "Accept phrases like 'tenure-track', 'tenure eligible'. "
            "Open-rank postings are acceptable if they include Assistant among the listed ranks."
        ),
    )

    # 4) Degree requirement: PhD/doctorate by start date (critical)
    degree_leaf = evaluator.add_leaf(
        id=f"degree_requirement_{idx+1}",
        desc=f"Position requires PhD/doctorate completed by position start date (position #{idx+1})",
        parent=position_node,
        critical=True
    )
    degree_claim = (
        "This posting requires a PhD or equivalent doctorate degree to be completed by the position start date "
        "or time of appointment."
    )
    await evaluator.verify(
        claim=degree_claim,
        node=degree_leaf,
        sources=pos.job_posting_url,
        additional_instruction=(
            "Look for language like 'PhD required by time of appointment', 'doctorate required before start', "
            "or equivalent wording."
        ),
    )

    # 5) Start date: Fall 2026 (critical)
    start_leaf = evaluator.add_leaf(
        id=f"start_date_{idx+1}",
        desc=f"Position has a Fall 2026 (August/September 2026) start date (position #{idx+1})",
        parent=position_node,
        critical=True
    )
    start_claim = (
        "The posting specifies a Fall 2026 start date, or explicitly states August or September 2026 as the start."
    )
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=pos.job_posting_url,
        additional_instruction=(
            "Accept 'Fall 2026', 'August 2026', or 'September 2026'. "
            "If the page states academic year 2026–27 with a fall start, it is acceptable."
        ),
    )

    # 6) Currently accepting applications (critical)
    accepting_leaf = evaluator.add_leaf(
        id=f"accepting_{idx+1}",
        desc=f"Posting indicates it is currently accepting applications (position #{idx+1})",
        parent=position_node,
        critical=True
    )
    accepting_claim = (
        f"As of {CURRENT_DATE_STR}, the posting indicates applications are being accepted "
        "(e.g., 'Open until filled', 'review begins' not yet past, an active 'Apply' button/link, "
        "or an application deadline later than the current date)."
    )
    await evaluator.verify(
        claim=accepting_claim,
        node=accepting_leaf,
        sources=pos.job_posting_url,
        additional_instruction=(
            f"Use dates on the page. If a deadline is in the future relative to {CURRENT_DATE_STR}, or if "
            "it says 'Open until filled' or provides an active apply link, consider it currently accepting."
        ),
    )

    # 7) U.S. location (critical)
    us_leaf = evaluator.add_leaf(
        id=f"us_university_{idx+1}",
        desc=f"Position is at a U.S. university (position #{idx+1})",
        parent=position_node,
        critical=True
    )
    us_claim = "This job posting is for a position located in the United States (USA)."
    await evaluator.verify(
        claim=us_claim,
        node=us_leaf,
        sources=pos.job_posting_url,
        additional_instruction=(
            "Look for city/state (e.g., 'CA', 'NY', 'TX', etc.), or 'United States/USA' indicators on the page "
            "or the institution address."
        ),
    )

    # 8) Application materials group (critical parent with 5 critical leaves)
    materials_node = evaluator.add_parallel(
        id=f"application_materials_main_{idx+1}",
        desc=f"Job posting specifies required application materials (position #{idx+1})",
        parent=position_node,
        critical=True
    )

    # Define the five canonical materials leaves
    mat_items = [
        ("CV", "cv", "A curriculum vitae (CV) or resume is required."),
        ("Cover Letter", "cover_letter", "A cover letter is required."),
        ("Research Statement", "research_statement", "A research statement/plan/proposal is required."),
        ("Teaching Statement", "teaching_statement", "A teaching statement/philosophy is required."),
        ("References", "references", "Reference letters or names/contact information for referees are required."),
    ]

    # For each material, verify presence on the page
    for canonical, short_id, claim_text in mat_items:
        leaf = evaluator.add_leaf(
            id=f"application_materials_{short_id}_{idx+1}",
            desc=f"Posting requires {canonical} (position #{idx+1})",
            parent=materials_node,
            critical=True
        )
        await evaluator.verify(
            claim=claim_text,
            node=leaf,
            sources=pos.job_posting_url,
            additional_instruction=(
                "Accept synonyms: CV/resume for CV; letter of interest/intent for Cover Letter; "
                "research plan/proposal for Research Statement; teaching philosophy for Teaching Statement; "
                "either reference letters or names/contact info for referees for References."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the STEM tenure-track positions (Fall 2026) task.
    """
    # Initialize evaluator (root is non-critical to allow partial scoring across positions)
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

    # Extract positions
    extracted_positions = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction",
    )

    # Select up to 4 positions and pad if fewer were provided
    positions = _first_n_positions(extracted_positions, 4)

    # Add custom info (current date, counts)
    evaluator.add_custom_info(
        {"current_date": CURRENT_DATE_STR, "positions_found_in_answer": len(extracted_positions.positions)},
        info_type="context",
        info_name="evaluation_context"
    )

    # Verify each of the 4 positions
    for idx, pos in enumerate(positions):
        await verify_position(evaluator, root, pos, idx)

    # Critical check: we truly have 4 usable positions (URLs provided)
    evaluator.add_custom_node(
        result=_positions_count_with_urls(positions) == 4,
        id="four_positions_with_urls",
        desc="All four positions have a valid job posting URL provided",
        parent=root,
        critical=True
    )

    # Critical check: universities are all distinct
    evaluator.add_custom_node(
        result=_all_universities_unique(positions),
        id="distinct_universities",
        desc="Each position is at a different U.S. university (no duplicates among the four)",
        parent=root,
        critical=True
    )

    # Return structured summary
    return evaluator.get_summary()