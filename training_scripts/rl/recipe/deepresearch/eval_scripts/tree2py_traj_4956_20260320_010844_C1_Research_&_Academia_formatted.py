import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# ------------------------------------------------------------------------------
# Task constants
# ------------------------------------------------------------------------------
TASK_ID = "lunar_eclipse_2026_totality"
TASK_DESCRIPTION = """
For the total lunar eclipse occurring on March 3, 2026, provide the following information about the totality phase:
(1) the exact UTC time when totality begins,
(2) the exact UTC time when totality ends, and
(3) the total duration of the totality phase in minutes.
Include a reference URL from an authoritative astronomy source that confirms this information.
"""

# Ground truth reference values per rubric (used for exact-match checks)
EXPECTED_START_UTC = "11:04:34"
EXPECTED_END_UTC = "12:02:49"
EXPECTED_DURATION_MINUTES = "58"


# ------------------------------------------------------------------------------
# Extraction models
# ------------------------------------------------------------------------------
class EclipseInfo(BaseModel):
    totality_start_utc: Optional[str] = None
    totality_end_utc: Optional[str] = None
    totality_duration_minutes: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# ------------------------------------------------------------------------------
# Extraction prompt
# ------------------------------------------------------------------------------
def prompt_extract_eclipse_info() -> str:
    return """
    Extract the information about the total eclipse (totality) phase for the March 3, 2026 lunar eclipse exactly as stated in the answer.

    Required fields:
    - totality_start_utc: The UTC time the totality phase begins (as stated in the answer). Keep the exact formatting used by the answer (e.g., '11:04:34 UTC', '11:04 UTC', '11:04:34Z'). If missing, return null.
    - totality_end_utc: The UTC time the totality phase ends (as stated in the answer). Keep the exact formatting used by the answer. If missing, return null.
    - totality_duration_minutes: The duration of the totality phase in minutes (as a string, e.g., '58'). If missing, return null.
    - reference_urls: A list of all URLs explicitly cited in the answer to support these eclipse timings (extract actual URLs only; include full protocols; if none, return an empty list).

    Do not infer or transform times (e.g., do not convert time zones). Only extract what is explicitly stated in the answer.
    """


# ------------------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------------------
def _is_nonempty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _domain_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def _has_authoritative_url(urls: List[str]) -> bool:
    # Consider widely accepted authoritative sources for eclipse timings
    # Expandable list; keep conservative but practical.
    allowed_suffixes = [
        "nasa.gov",          # includes eclipse.gsfc.nasa.gov
        "timeanddate.com",
    ]
    for u in urls:
        d = _domain_of(u)
        if any(d.endswith(suf) for suf in allowed_suffixes):
            return True
    return False


# ------------------------------------------------------------------------------
# Verification subroutines
# ------------------------------------------------------------------------------
async def verify_time_group(
    evaluator: Evaluator,
    parent,
    group_id: str,
    extracted_time: Optional[str],
    expected_time_hms: str,
    role_desc: str,  # e.g., "Totality start time (UTC)"
    urls: List[str],
) -> None:
    """
    Build a critical parallel group to verify a time item:
    - Existence provided by the answer
    - Exact value matches the expected hh:mm:ss (UTC) per rubric (simple logical check)
    - Cited sources support the provided time
    """
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=f"The answer correctly states that the {role_desc} equals {expected_time_hms} UTC on March 3, 2026, and is supported by sources",
        parent=parent,
        critical=True,
    )

    # 1) Existence
    exists_node = evaluator.add_custom_node(
        result=_is_nonempty_str(extracted_time),
        id=f"{group_id}_provided",
        desc=f"{role_desc} is provided in the answer",
        parent=group_node,
        critical=True,
    )

    # 2) Exact value equals expected
    eq_node = evaluator.add_leaf(
        id=f"{group_id}_exact_match",
        desc=f"{role_desc} exactly matches the required value {expected_time_hms} UTC on March 3, 2026",
        parent=group_node,
        critical=True,
    )

    provided_str = extracted_time or ""
    eq_claim = (
        f"The provided {role_desc.lower()} '{provided_str}' is exactly equivalent to '03 March 2026 "
        f"{expected_time_hms} UTC'. Consider formatting variants like 'UT', 'UTC', 'Z' acceptable, "
        f"but the second component must be {expected_time_hms.split(':')[-1]} and the minute and hour must be identical."
    )
    await evaluator.verify(
        claim=eq_claim,
        node=eq_node,
        additional_instruction=(
            "Judge purely by logical/format equivalence, not by external knowledge. "
            "Do not accept minute-only times (e.g., '11:04 UTC') as exact unless they explicitly indicate the same seconds."
        ),
    )

    # 3) Source support for the provided time
    supported_node = evaluator.add_leaf(
        id=f"{group_id}_source_supported",
        desc=f"The provided {role_desc} is supported by the cited authoritative source(s)",
        parent=group_node,
        critical=True,
    )

    support_claim = (
        f"For the March 3, 2026 total lunar eclipse, the totality {role_desc.lower()} is {provided_str} (in UTC)."
    )
    await evaluator.verify(
        claim=support_claim,
        node=supported_node,
        sources=urls if urls else None,
        additional_instruction=(
            "Verify the time for the totality phase on the page. "
            "For the start time, look for 'start of totality' or 'U2'; for the end time, look for 'end of totality' or 'U3'. "
            "Confirm that the page states the same UTC time as in the claim (allowing trivial formatting variants of UTC like UT/Z)."
        ),
    )


async def verify_duration_group(
    evaluator: Evaluator,
    parent,
    group_id: str,
    extracted_duration_min: Optional[str],
    expected_minutes: str,
    urls: List[str],
) -> None:
    """
    Build a critical parallel group to verify duration:
    - Existence provided by the answer
    - Exact value matches expected minutes (simple logical check)
    - Cited sources support the provided duration
    """
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=f"The answer correctly states that the duration of totality is {expected_minutes} minutes and is supported by sources",
        parent=parent,
        critical=True,
    )

    # 1) Existence
    exists_node = evaluator.add_custom_node(
        result=_is_nonempty_str(extracted_duration_min),
        id=f"{group_id}_provided",
        desc="Duration of totality (in minutes) is provided in the answer",
        parent=group_node,
        critical=True,
    )

    # 2) Exact value equals expected minutes
    eq_node = evaluator.add_leaf(
        id=f"{group_id}_exact_match",
        desc=f"Duration of totality exactly matches {expected_minutes} minutes",
        parent=group_node,
        critical=True,
    )
    provided_str = (extracted_duration_min or "").strip()
    eq_claim = (
        f"The provided duration value '{provided_str}' minutes is exactly equal to {expected_minutes} minutes."
    )
    await evaluator.verify(
        claim=eq_claim,
        node=eq_node,
        additional_instruction="Judge this purely as a numeric/value equivalence check, allowing minor formatting like surrounding text but requiring the number to be identical.",
    )

    # 3) Source support
    supported_node = evaluator.add_leaf(
        id=f"{group_id}_source_supported",
        desc="The provided duration of totality is supported by the cited source(s)",
        parent=group_node,
        critical=True,
    )
    support_claim = (
        f"For the March 3, 2026 total lunar eclipse, the totality duration is {provided_str} minutes."
    )
    await evaluator.verify(
        claim=support_claim,
        node=supported_node,
        sources=urls if urls else None,
        additional_instruction="Verify that the referenced page explicitly states the duration of the totality phase in minutes, matching the claim.",
    )


# ------------------------------------------------------------------------------
# Main evaluation entry point
# ------------------------------------------------------------------------------
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root-level aggregation (non-critical root)
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

    # Extract structured information
    extracted: EclipseInfo = await evaluator.extract(
        prompt=prompt_extract_eclipse_info(),
        template_class=EclipseInfo,
        extraction_name="eclipse_totality_info",
    )

    # Record ground truth targets (for transparency)
    evaluator.add_ground_truth({
        "expected_start_utc": EXPECTED_START_UTC,
        "expected_end_utc": EXPECTED_END_UTC,
        "expected_duration_minutes": EXPECTED_DURATION_MINUTES,
        "event_date": "2026-03-03",
        "phase": "Totality (total eclipse phase)",
    })

    # Build top-level critical node per rubric
    main_node = evaluator.add_parallel(
        id="Total_Eclipse_Phase_Information",
        desc="Verify that the answer provides accurate information about the total eclipse phase (totality) of the March 3, 2026 lunar eclipse",
        parent=root,
        critical=True,
    )

    # Reference URL requirement (single-leaf check, critical)
    ref_leaf = evaluator.add_custom_node(
        result=(len(extracted.reference_urls) > 0 and _has_authoritative_url(extracted.reference_urls)),
        id="Reference_URL_Provided",
        desc="At least one authoritative reference URL (e.g., NASA or timeanddate.com) is provided to confirm the information",
        parent=main_node,
        critical=True,
    )

    # Start time verification group
    await verify_time_group(
        evaluator=evaluator,
        parent=main_node,
        group_id="Totality_Start_Time_UTC",
        extracted_time=extracted.totality_start_utc,
        expected_time_hms=EXPECTED_START_UTC,
        role_desc="Totality start time (UTC)",
        urls=extracted.reference_urls,
    )

    # End time verification group
    await verify_time_group(
        evaluator=evaluator,
        parent=main_node,
        group_id="Totality_End_Time_UTC",
        extracted_time=extracted.totality_end_utc,
        expected_time_hms=EXPECTED_END_UTC,
        role_desc="Totality end time (UTC)",
        urls=extracted.reference_urls,
    )

    # Duration verification group
    await verify_duration_group(
        evaluator=evaluator,
        parent=main_node,
        group_id="Duration_of_Totality",
        extracted_duration_min=extracted.totality_duration_minutes,
        expected_minutes=EXPECTED_DURATION_MINUTES,
        urls=extracted.reference_urls,
    )

    # Add custom info for debugging (optional)
    evaluator.add_custom_info(
        {
            "extracted_start_utc": extracted.totality_start_utc,
            "extracted_end_utc": extracted.totality_end_utc,
            "extracted_duration_minutes": extracted.totality_duration_minutes,
            "reference_urls": extracted.reference_urls,
            "authoritative_url_present": _has_authoritative_url(extracted.reference_urls),
            "parsed_reference_domains": [_domain_of(u) for u in extracted.reference_urls],
        },
        info_type="extraction_debug",
        info_name="extraction_debug_info",
    )

    return evaluator.get_summary()