import asyncio
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ray_jhu_tenure"
TASK_DESCRIPTION = (
    "How long will Ray Jayawardhana have served as provost of Johns Hopkins University from when he started the "
    "position until he assumes his new role as president of Caltech?"
)

# Ground-truth focal dates (as per rubric)
GROUND_TRUTH_START_ISO = "2023-10-15"
GROUND_TRUTH_END_ISO = "2026-07-01"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DateInfo(BaseModel):
    date_text: Optional[str] = None
    iso_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TenureExtraction(BaseModel):
    start: Optional[DateInfo] = None
    end: Optional[DateInfo] = None
    tenure_duration_text: Optional[str] = None  # The duration as stated in the answer (e.g., "about 2 years, 8 months")


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tenure() -> str:
    return """
    Extract the following information from the answer about Ray Jayawardhana's tenure as provost at Johns Hopkins University and his transition to Caltech:

    1) start:
       - date_text: The start date text phrase for when he began as provost at Johns Hopkins (e.g., "October 15, 2023").
       - iso_date: Convert that start date to ISO format YYYY-MM-DD if a precise calendar date is given; otherwise return null.
       - sources: All URLs explicitly cited in the answer that support his start date (e.g., an official JHU announcement page, news article, etc.). If none are given, return an empty array.

    2) end:
       - date_text: The date text phrase for when he will assume the Caltech presidency (e.g., "July 1, 2026").
       - iso_date: Convert that end date to ISO format YYYY-MM-DD if a precise calendar date is given; otherwise return null.
       - sources: All URLs explicitly cited in the answer that support the Caltech presidency start date (e.g., Caltech announcement page, news article, etc.). If none are given, return an empty array.

    3) tenure_duration_text:
       - The tenure duration described or calculated in the answer for the period from the start date at JHU to the Caltech presidency start date (e.g., "2 years, 8 months and 16 days", "about 2 years and 9 months", etc.). If the answer does not provide a duration, return null.

    Rules:
    - Do not invent information not present in the answer.
    - For URLs, only include explicit URLs present in the answer (markdown links are okay, extract the actual URL).
    """


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def _try_parse_date(s: Optional[str]) -> Optional[date]:
    """Attempt to parse a date string across common formats."""
    if not s:
        return None
    s = s.strip()
    fmts = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
        "%m/%d/%Y",
        "%d %B %Y",
        "%d %b %Y",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    return None


def _parse_iso_or_text(iso_text: Optional[str], fallback_text: Optional[str]) -> Optional[date]:
    """Parse ISO first; fallback to textual parse if needed."""
    d = _try_parse_date(iso_text)
    if d:
        return d
    return _try_parse_date(fallback_text)


def _format_date_long(d: date) -> str:
    """Format a date like 'October 15, 2023'."""
    return d.strftime("%B %-d, %Y") if hasattr(d, "strftime") else str(d)


def _days_in_month(year: int, month: int) -> int:
    """Return number of days in a given month/year."""
    # Simple approach without external libs
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    this_month = date(year, month, 1)
    return (next_month - this_month).days


def _diff_ymd(start: date, end: date) -> Dict[str, int]:
    """
    Compute the calendar difference in years, months, days between two dates (start <= end).
    Algorithm: compute initial y,m,d deltas then normalize negatives by borrowing from months and years.
    """
    y = end.year - start.year
    m = end.month - start.month
    d = end.day - start.day

    if d < 0:
        # Borrow one month
        m -= 1
        # Add days from the previous month of 'end'
        prev_month = end.month - 1 or 12
        prev_year = end.year - 1 if end.month == 1 else end.year
        d += _days_in_month(prev_year, prev_month)

    if m < 0:
        y -= 1
        m += 12

    return {"years": y, "months": m, "days": d}


def _build_human_duration(y: int, m: int, d: int) -> str:
    """Build a friendly string like '2 years, 8 months, and 16 days' (skipping zero parts)."""
    parts = []
    if y:
        parts.append(f"{y} year" + ("s" if y != 1 else ""))
    if m:
        parts.append(f"{m} month" + ("s" if m != 1 else ""))
    if d:
        parts.append(f"{d} day" + ("s" if d != 1 else ""))

    if not parts:
        return "0 days"

    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


# --------------------------------------------------------------------------- #
# Verification subroutine                                                     #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, root_node, extraction: TenureExtraction) -> None:
    """
    Build the verification tree and run all checks, following the rubric:
    1) Start_Date (critical; sequential step 1)
    2) End_Date (critical; sequential step 2)
    3) Duration_Calculation (critical; sequential step 3)
    """

    # Create a critical sequential main node to reflect the rubric
    main_node = evaluator.add_sequential(
        id="Ray_Jayawardhana_JHU_Tenure",
        desc="Calculate the duration of Ray Jayawardhana's tenure as provost at Johns Hopkins University",
        parent=root_node,
        critical=True
    )

    # Prepare ground-truth dates from rubric
    start_gt = _try_parse_date(GROUND_TRUTH_START_ISO)
    end_gt = _try_parse_date(GROUND_TRUTH_END_ISO)

    # Extracted sources (may be empty)
    start_sources = extraction.start.sources if (extraction and extraction.start) else []
    end_sources = extraction.end.sources if (extraction and extraction.end) else []

    # 1) Start_Date leaf (critical)
    start_leaf = evaluator.add_leaf(
        id="Start_Date",
        desc="Identify that Ray Jayawardhana started as Johns Hopkins University's provost on October 15, 2023",
        parent=main_node,
        critical=True
    )
    start_claim_date_long = _format_date_long(start_gt) if start_gt else "October 15, 2023"
    start_claim = f"Ray Jayawardhana started as provost of Johns Hopkins University on {start_claim_date_long}."

    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=start_sources,  # Prefer verifying with URLs; falls back to simple verify if none
        additional_instruction=(
            "Verify that the cited page(s) explicitly state that Ray Jayawardhana began his role as "
            "Johns Hopkins University's provost on October 15, 2023. Accept phrasing like 'effective' or 'began on'. "
            "Minor wording variations are fine, but the date must match."
        )
    )

    # 2) End_Date leaf (critical)
    end_leaf = evaluator.add_leaf(
        id="End_Date",
        desc="Identify that Ray Jayawardhana will conclude his provost role when he assumes the Caltech presidency on July 1, 2026",
        parent=main_node,
        critical=True
    )
    end_claim_date_long = _format_date_long(end_gt) if end_gt else "July 1, 2026"
    # Focus on the presidency effective date, which implicitly marks the transition
    end_claim = f"Ray Jayawardhana will assume the presidency of Caltech on {end_claim_date_long}."

    await evaluator.verify(
        claim=end_claim,
        node=end_leaf,
        sources=end_sources,  # Prefer verifying with URLs; falls back to simple verify if none
        additional_instruction=(
            "Verify that the cited page(s) explicitly state that Ray Jayawardhana becomes (assumes) the presidency of "
            "Caltech on July 1, 2026. Accept phrasing like 'effective July 1, 2026'."
        )
    )

    # 3) Duration_Calculation leaf (critical)
    duration_leaf = evaluator.add_leaf(
        id="Duration_Calculation",
        desc="Calculate and provide the tenure duration from October 15, 2023 to July 1, 2026, expressed in years, months, and/or days",
        parent=main_node,
        critical=True
    )

    # Compute expected duration (based on rubric dates)
    # If for some reason parsing failed (extremely unlikely given fixed GT), skip computation gracefully
    computed_duration_str = ""
    if start_gt and end_gt and end_gt >= start_gt:
        diff = _diff_ymd(start_gt, end_gt)
        computed_duration_str = _build_human_duration(diff["years"], diff["months"], diff["days"])
    else:
        computed_duration_str = "2 years, 8 months, and 16 days"  # Fallback to the expected human calculation

    # Extract the answer's stated duration (if any)
    answer_duration_text = extraction.tenure_duration_text if extraction else None
    answer_duration_text = answer_duration_text or ""

    duration_claim = (
        f"The correct tenure duration between {start_claim_date_long} and {end_claim_date_long} is "
        f"{computed_duration_str}. The answer's stated tenure duration is '{answer_duration_text}'. "
        f"These are consistent (allowing small rounding or equivalent expressions)."
    )

    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        additional_instruction=(
            "Judge whether the answer's stated tenure duration matches the correct difference between "
            "October 15, 2023 and July 1, 2026. Accept equivalent expressions such as: "
            "'about 2 years and 9 months', 'roughly 33 months', or omission of days if the year+month "
            "components are correct. Small rounding differences (within about a week) are acceptable. "
            "If the answer provides no duration, mark this as incorrect."
        )
    )

    # Record helpful custom info for debugging/traceability
    evaluator.add_custom_info(
        info={
            "ground_truth": {
                "start_iso": GROUND_TRUTH_START_ISO,
                "end_iso": GROUND_TRUTH_END_ISO
            },
            "computed_duration": computed_duration_str,
            "extracted": extraction.dict() if extraction else {}
        },
        info_type="debug",
        info_name="computed_and_extracted_info"
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
    Evaluate an answer for Ray Jayawardhana's JHU provost tenure duration task.
    """
    # Initialize evaluator with a sequential root to reflect the staged nature
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract relevant information from the answer
    extraction: TenureExtraction = await evaluator.extract(
        prompt=prompt_extract_tenure(),
        template_class=TenureExtraction,
        extraction_name="tenure_extraction"
    )

    # Add ground-truth dates to the record for transparency
    evaluator.add_ground_truth(
        {
            "start_date_iso": GROUND_TRUTH_START_ISO,
            "end_date_iso": GROUND_TRUTH_END_ISO,
            "task": "Compute tenure duration from JHU provost start to Caltech presidency start"
        },
        gt_type="ground_truth_dates"
    )

    # Build tree and run verification
    await build_and_verify_tree(evaluator, root, extraction)

    # Return summary
    return evaluator.get_summary()