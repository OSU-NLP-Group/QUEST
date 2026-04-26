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
TASK_ID = "ccps_closure_2026_01_26"
TASK_DESCRIPTION = (
    "On January 26, 2026, were Carroll County Public Schools in Maryland closed for students? "
    "Verify this information using an official source from the school district and provide the URL of the official source used for verification."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CCPSClosureExtraction(BaseModel):
    district_text: Optional[str] = None
    date_text: Optional[str] = None
    student_closure_status: Optional[str] = None  # e.g., "closed", "open", "2-hour delay", "virtual", etc.
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ccps_closure() -> str:
    return """
    From the provided answer text, extract the following fields related to the Carroll County Public Schools (CCPS) student closure inquiry:
    - district_text: The district name as explicitly written in the answer (e.g., "Carroll County Public Schools", "CCPS (Maryland)"), or null if not specified.
    - date_text: The specific date that the answer addresses for the school status (e.g., "January 26, 2026", "Jan 26, 2026", "1/26/2026"), or null if not specified.
    - student_closure_status: The explicit student closure status stated in the answer for that date (e.g., "closed for students", "not closed", "open", "2-hour delay", "virtual", "schools and offices closed"). Use the exact phrasing from the answer if possible, or a concise paraphrase if needed. If not stated, set to null.
    - source_urls: A list of all URLs that the answer cites as sources for verification. Include only URLs explicitly present in the answer. Do not invent URLs.

    Return a JSON object with these fields. If any field is missing, set it to null (or an empty array for source_urls).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_student_closure_status(status_text: Optional[str]) -> Optional[bool]:
    """
    Normalize the student's closure status to a boolean if possible.
    Returns:
      True  -> closed for students
      False -> not closed for students (e.g., open, delay, virtual)
      None -> cannot determine from text
    """
    if not status_text:
        return None

    s = status_text.strip().lower()
    # Positive closure cues
    closed_cues = [
        "closed for students", "schools closed", "school is closed", "no school for students",
        "schools and offices closed", "schools will be closed", "closed on", "closed monday"
    ]
    # Non-closure cues (open, delay, remote)
    not_closed_cues = [
        "open", "two hour delay", "2-hour delay", "2 hour delay", "delayed", "virtual", "remote learning",
        "open on", "open monday"
    ]

    if any(c in s for c in closed_cues):
        return True
    if any(c in s for c in not_closed_cues):
        return False

    # If explicitly says "not closed" etc.
    if "not closed" in s:
        return False

    return None


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extracted: CCPSClosureExtraction,
    parent_node_desc: str = "CCPS (MD) closure verification for students on Jan 26, 2026"
) -> None:
    """
    Build the verification tree according to the rubric and run all checks.
    We slightly reorganize nodes into two groups to avoid over-gating:
    - Answer content checks
    - Source verification checks (sequential, gated by 'URL provided')
    """

    # Root-level grouping (non-critical to allow partial credit as the original task only mandates core verification)
    # We keep the evaluator root as the overall root; here we create two subgroups.
    answer_group = evaluator.add_parallel(
        id="answer_content_checks",
        desc="Answer content checks (district/date/explicit status)",
        critical=False
    )

    source_group = evaluator.add_sequential(
        id="source_verification_checks",
        desc="Source verification checks (must provide URL, then validate source type and claims)",
        critical=False
    )

    # ------------------------- Answer content checks ------------------------- #
    # 1) District identification
    district_leaf = evaluator.add_leaf(
        id="district_is_ccps_md",
        desc="Answer identifies the district as Carroll County Public Schools in Maryland.",
        parent=answer_group,
        critical=True,
    )
    claim_district = (
        "In the answer text, the school district is identified as Carroll County Public Schools in Maryland "
        "(CCPS, MD). Accept reasonable abbreviations like 'CCPS' only if the context clearly refers to the "
        "Maryland district, not another state."
    )
    # Simple verification against the answer text only
    # (no external source needed; this is a meta-check on the answer content).
    # Note: We do not rely on extracted.district_text to avoid missing-NULL issues.
    await evaluator.verify(
        claim=claim_district,
        node=district_leaf,
        additional_instruction="Focus strictly on the answer text and the task description context."
    )

    # 2) Date addressed
    date_leaf = evaluator.add_leaf(
        id="date_is_jan_26_2026",
        desc="Answer addresses the specific date January 26, 2026.",
        parent=answer_group,
        critical=True,
    )
    claim_date = (
        "The answer explicitly addresses the specific date January 26, 2026 (acceptable formats include "
        "'January 26, 2026', 'Jan 26, 2026', or '1/26/2026')."
    )
    await evaluator.verify(
        claim=claim_date,
        node=date_leaf,
        additional_instruction="Focus strictly on the answer text. Equivalent date formats are acceptable."
    )

    # 3) Explicitly states closure status for students
    status_leaf = evaluator.add_leaf(
        id="states_student_closure_status",
        desc="Answer explicitly states whether CCPS schools were closed for students on January 26, 2026 (yes/no).",
        parent=answer_group,
        critical=True,
    )
    claim_status = (
        "The answer explicitly states whether CCPS schools were closed for students on January 26, 2026 (a clear yes/no). "
        "Accept synonymous phrasings such as 'schools closed for students', 'no school for students', or clearly 'open', "
        "'delay', or 'virtual' for not closed."
    )
    await evaluator.verify(
        claim=claim_status,
        node=status_leaf,
        additional_instruction="Judge only the explicitness within the answer text."
    )

    # ------------------------- Source verification checks -------------------- #
    # 4) Official source URL provided (gate for all subsequent source checks)
    urls = extracted.source_urls or []
    url_provided = evaluator.add_custom_node(
        result=(len(urls) > 0),
        id="official_source_url_provided",
        desc="Answer provides a URL to the source used for verification.",
        parent=source_group,
        critical=True
    )

    # 5) Source is an allowed official type
    allowed_type_leaf = evaluator.add_leaf(
        id="source_is_allowed_official_type",
        desc=(
            "The provided URL is an allowed official source type per constraints: "
            "CCPS official website (carrollk12.org), an official district social media account/page/post, "
            "OR a reputable news source directly reporting the district's official announcement."
        ),
        parent=source_group,
        critical=True
    )
    claim_allowed_type = (
        "This page qualifies as an allowed official source type: either (1) a page on the CCPS official domain "
        "carrollk12.org; (2) an official CCPS district social media account/page/post (e.g., Facebook, X/Twitter, "
        "Instagram, YouTube) that clearly belongs to the district; or (3) a reputable news outlet article that "
        "directly reports the district's official announcement."
    )
    await evaluator.verify(
        claim=claim_allowed_type,
        node=allowed_type_leaf,
        sources=urls,
        additional_instruction=(
            "Use the page content and the provided URL domain to judge. Accept if the page clearly belongs to CCPS "
            "or is an obviously official CCPS social account/post; for news, accept only if the article explicitly "
            "reports the district's official announcement."
        )
    )

    # 6) Source supports the stated student closure status on Jan 26, 2026
    supports_leaf = evaluator.add_leaf(
        id="source_supports_student_closure_on_2026_01_26",
        desc="The cited source content supports the stated student closure status for January 26, 2026.",
        parent=source_group,
        critical=True
    )
    normalized_closed = normalize_student_closure_status(extracted.student_closure_status)
    if normalized_closed is True:
        claim_supports = (
            "The page states that Carroll County Public Schools (Maryland) were closed for students on Monday, "
            "January 26, 2026."
        )
    elif normalized_closed is False:
        claim_supports = (
            "The page indicates that Carroll County Public Schools (Maryland) were not closed for students on "
            "Monday, January 26, 2026 (e.g., they were open, delayed, or virtual rather than closed)."
        )
    else:
        # If the answer's status cannot be normalized, verify generically that the page supports whatever the answer claimed.
        claim_supports = (
            "The page supports the specific student closure status for Carroll County Public Schools (Maryland) "
            "on Monday, January 26, 2026, as claimed in the answer (closed vs not closed)."
        )
    await evaluator.verify(
        claim=claim_supports,
        node=supports_leaf,
        sources=urls,
        additional_instruction=(
            "Treat 'no school for students', 'schools closed for students', or 'schools and offices closed' as closed. "
            "Treat 'open', 'delay', or 'virtual instruction' as not closed. The date must match January 26, 2026."
        )
    )

    # 7) Source indicates announcement made on Jan 25, 2026 (non-critical detail)
    announce_leaf = evaluator.add_leaf(
        id="source_indicates_announcement_made_jan_25_2026",
        desc="The cited source indicates the closure announcement was made on January 25, 2026.",
        parent=source_group,
        critical=False
    )
    claim_announce = (
        "The page shows that the closure announcement was posted or dated on Sunday, January 25, 2026 (U.S. Eastern Time). "
        "Evidence may include the post timestamp, page update date, or explicit mention."
    )
    await evaluator.verify(
        claim=claim_announce,
        node=announce_leaf,
        sources=urls,
        additional_instruction="Prefer explicit timestamps. If a timezone is shown, interpret in U.S. Eastern Time."
    )

    # 8) Source indicates inclement weather reason (non-critical detail)
    weather_leaf = evaluator.add_leaf(
        id="source_indicates_inclement_weather_reason",
        desc="The cited source indicates the closure was due to inclement weather.",
        parent=source_group,
        critical=False
    )
    claim_weather = (
        "The page indicates that the closure decision for January 26, 2026 was due to inclement weather "
        "(e.g., snow, ice, winter weather)."
    )
    await evaluator.verify(
        claim=claim_weather,
        node=weather_leaf,
        sources=urls,
        additional_instruction="Look for wording like inclement weather, snow, ice, winter storm, or hazardous conditions."
    )

    # 9) Source indicates employees also closed (non-critical detail)
    employees_leaf = evaluator.add_leaf(
        id="source_indicates_employees_also_closed",
        desc="The cited source indicates schools were closed for employees as well on January 26, 2026.",
        parent=source_group,
        critical=False
    )
    claim_employees = (
        "The page explicitly states that CCPS schools or offices were also closed for employees on Monday, January 26, 2026 "
        "(not only for students)."
    )
    await evaluator.verify(
        claim=claim_employees,
        node=employees_leaf,
        sources=urls,
        additional_instruction=(
            "Accept language like 'schools and offices closed', 'employees do not report', 'Code Red' where it clearly "
            "means employees are not to report. If the page only mentions student closure without employee status, do not accept."
        )
    )

    # Optional: record custom info for transparency
    evaluator.add_custom_info(
        {
            "extracted_district_text": extracted.district_text,
            "extracted_date_text": extracted.date_text,
            "extracted_student_closure_status": extracted.student_closure_status,
            "extracted_source_urls": urls,
            "normalized_closed_interpretation": (
                "closed_for_students" if normalized_closed is True
                else ("not_closed_for_students" if normalized_closed is False else "undetermined")
            ),
        },
        info_type="extraction_summary",
        info_name="extraction_summary"
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the CCPS (Maryland) student closure on January 26, 2026 task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation at the top level
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_ccps_closure(),
        template_class=CCPSClosureExtraction,
        extraction_name="ccps_closure_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()