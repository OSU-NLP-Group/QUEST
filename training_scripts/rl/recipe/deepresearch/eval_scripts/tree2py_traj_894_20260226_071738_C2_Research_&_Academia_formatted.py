import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ai_lab_director_2025"
TASK_DESCRIPTION = (
    "In 2025, a prominent university appointed a new director for its artificial intelligence lab. "
    "Identify the name of the university, the name of the newly appointed director, and the exact date (Month Day, Year) "
    "when this appointment was officially announced."
)

# Ground-truth expectations encoded by the rubric
UNIVERSITY_EXPECTED = "Stanford University"
DIRECTOR_EXPECTED = "Carlos Guestrin"
ANNOUNCEMENT_MONTH_EXPECTED = "February"
ANNOUNCEMENT_DAY_EXPECTED = "18"
ANNOUNCEMENT_YEAR_EXPECTED = "2025"
FULL_DATE_EXPECTED = f"{ANNOUNCEMENT_MONTH_EXPECTED} {ANNOUNCEMENT_DAY_EXPECTED}, {ANNOUNCEMENT_YEAR_EXPECTED}"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AppointmentExtraction(BaseModel):
    """
    Structured info extracted from the agent's answer about the 2025 AI lab director appointment.
    All fields must be extracted exactly as they appear in the answer; do not invent any values.
    """
    university: Optional[str] = None
    director: Optional[str] = None
    # The full announcement date string exactly as written in the answer (prefer 'Month Day, Year' if present)
    announcement_date: Optional[str] = None
    # Split parts if clearly present in the answer (do not infer if not explicitly present)
    announcement_month: Optional[str] = None
    announcement_day: Optional[str] = None
    announcement_year: Optional[str] = None
    # All URLs the answer cites that directly support this appointment announcement
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_appointment_info() -> str:
    return """
    Extract the following information about the 2025 AI lab director appointment from the provided answer text.

    Required fields:
    1) university: The name of the university where the AI lab director appointment occurred. Return exactly as it appears in the answer.
    2) director: The full name of the newly appointed AI lab director, exactly as presented in the answer.
    3) announcement_date: The exact date of the official announcement, formatted exactly as written in the answer. Prefer the "Month Day, Year" format if the answer provides it (e.g., "February 18, 2025"). If the answer uses an alternative but equivalent textual date (e.g., "Feb 18, 2025"), return it exactly as written.
    4) announcement_month: The month portion of the announcement date, as it appears in the answer (e.g., "February" or "Feb"). Only provide if the month is explicitly present; otherwise null.
    5) announcement_day: The day-of-month as digits (e.g., "18"). Only provide if the day is explicitly present; otherwise null.
    6) announcement_year: The year (e.g., "2025"). Only provide if the year is explicitly present; otherwise null.
    7) source_urls: All URLs cited in the answer that directly support this appointment announcement (official lab pages, university news releases, press announcements, etc.). Extract actual URLs only (including those in markdown links), and return them as a list. If none are provided, return an empty list.

    Important rules:
    - Do not add, infer, or invent any information. Return exactly what the answer provides.
    - If any requested field is missing in the answer, set it to null (or empty list for source_urls).
    - Preserve the exact wording and formatting found in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper for additional instruction for evidence-based checks                 #
# --------------------------------------------------------------------------- #
def evidence_required_instruction(extra: Optional[str] = None) -> str:
    base = (
        "You must base your judgment solely on the content of the provided webpage(s). "
        "Look for official announcement pages from the university or the AI lab (press release, news post, or authoritative lab page). "
        "Allow reasonable naming variants (e.g., 'Stanford' vs 'Stanford University', 'Carlos E. Guestrin' vs 'Carlos Guestrin'). "
        "If no URLs are provided, treat the claim as not supported and mark it Incorrect."
    )
    if extra:
        return f"{base}\nAdditional focus: {extra}"
    return base


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root_node,
    extracted: AppointmentExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """
    # Top-level critical parallel node (as the rubric root)
    top_node = evaluator.add_parallel(
        id="AI_Lab_Director_Appointment_Information",
        desc="Complete and accurate identification of the university, director name, and announcement date for a 2025 AI lab director appointment",
        parent=root_node,
        critical=True,
    )

    # Normalize sources list from extraction
    sources: List[str] = extracted.source_urls if extracted and extracted.source_urls else []

    # 1) University identification (leaf, critical)
    uni_node = evaluator.add_leaf(
        id="University_Identification",
        desc="The university where the AI lab director appointment occurred is correctly identified as Stanford University",
        parent=top_node,
        critical=True,
    )
    uni_claim = (
        "The university where the AI lab director appointment occurred is Stanford University (also referred to as 'Stanford')."
    )
    await evaluator.verify(
        claim=uni_claim,
        node=uni_node,
        sources=sources,
        additional_instruction=evidence_required_instruction(
            "Confirm that the page explicitly ties the appointment to Stanford University or SAIL (Stanford AI Lab)."
        ),
    )

    # 2) Director identification (leaf, critical)
    dir_node = evaluator.add_leaf(
        id="Director_Name_Identification",
        desc="The name of the newly appointed AI lab director is correctly identified as Carlos Guestrin",
        parent=top_node,
        critical=True,
    )
    dir_claim = "The newly appointed AI lab director is Carlos Guestrin."
    await evaluator.verify(
        claim=dir_claim,
        node=dir_node,
        sources=sources,
        additional_instruction=evidence_required_instruction(
            "Look for language such as 'appointed as director', 'named director', or equivalent phrasing referring to Carlos Guestrin."
        ),
    )

    # 3) Announcement date (critical parallel with sub-checks for month/day/year)
    date_node = evaluator.add_parallel(
        id="Announcement_Date",
        desc="The exact date when the appointment was officially announced is provided in the correct format (Month Day, Year)",
        parent=top_node,
        critical=True,
    )

    # Month check (critical leaf)
    month_node = evaluator.add_leaf(
        id="Month_Correct",
        desc="The month of the announcement is correctly identified as February",
        parent=date_node,
        critical=True,
    )
    month_claim = "The official appointment announcement occurred in the month of February."
    await evaluator.verify(
        claim=month_claim,
        node=month_node,
        sources=sources,
        additional_instruction=evidence_required_instruction(
            "Match the announcement date text; accept 'February' or 'Feb' as equivalent."
        ),
    )

    # Day check (critical leaf)
    day_node = evaluator.add_leaf(
        id="Day_Correct",
        desc="The day of the announcement is correctly identified as 18",
        parent=date_node,
        critical=True,
    )
    day_claim = "The official appointment announcement occurred on the 18th day of the month."
    await evaluator.verify(
        claim=day_claim,
        node=day_node,
        sources=sources,
        additional_instruction=evidence_required_instruction(
            "Confirm that the announcement date text shows day '18' (accept '18' or '18th')."
        ),
    )

    # Year check (critical leaf)
    year_node = evaluator.add_leaf(
        id="Year_Correct",
        desc="The year of the announcement is correctly identified as 2025",
        parent=date_node,
        critical=True,
    )
    year_claim = "The official appointment announcement occurred in the year 2025."
    await evaluator.verify(
        claim=year_claim,
        node=year_node,
        sources=sources,
        additional_instruction=evidence_required_instruction(
            "Confirm the year '2025' in the announcement date."
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
    Evaluate an answer for the 2025 AI lab director appointment identification task.
    """
    # Initialize evaluator (root is non-critical holder; we add a critical top node under it)
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_appointment_info(),
        template_class=AppointmentExtraction,
        extraction_name="appointment_extraction",
    )

    # Add ground truth info (for transparency in the summary)
    evaluator.add_ground_truth(
        {
            "expected_university": UNIVERSITY_EXPECTED,
            "expected_director": DIRECTOR_EXPECTED,
            "expected_month": ANNOUNCEMENT_MONTH_EXPECTED,
            "expected_day": ANNOUNCEMENT_DAY_EXPECTED,
            "expected_year": ANNOUNCEMENT_YEAR_EXPECTED,
            "expected_full_date": FULL_DATE_EXPECTED,
        },
        gt_type="expected_values",
    )

    # Optionally record the raw extracted fields for debugging
    evaluator.add_custom_info(
        {
            "university_extracted": extracted.university,
            "director_extracted": extracted.director,
            "announcement_date_extracted": extracted.announcement_date,
            "announcement_month_extracted": extracted.announcement_month,
            "announcement_day_extracted": extracted.announcement_day,
            "announcement_year_extracted": extracted.announcement_year,
            "source_urls_extracted_count": len(extracted.source_urls) if extracted and extracted.source_urls else 0,
        },
        info_type="extraction_debug",
        info_name="extraction_debug_info",
    )

    # Build the verification tree and run checks
    await build_and_verify_tree(evaluator, root, extracted)

    # Return the structured evaluation summary
    return evaluator.get_summary()