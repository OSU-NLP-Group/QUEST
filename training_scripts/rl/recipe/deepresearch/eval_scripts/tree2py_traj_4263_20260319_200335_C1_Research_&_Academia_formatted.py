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
TASK_ID = "worm_moon_2026_peak_time"
TASK_DESCRIPTION = """
On March 3, 2026, the full moon known as the Worm Moon reached its peak illumination and coincided with a total lunar eclipse visible from North America. What was the exact peak time of the Worm Moon on this date? Please provide the time with its corresponding time zone and include a reference URL that verifies this information.
"""

# Ground truth reference for equivalence (for judge context/logging only)
GROUND_TRUTH = {
    "expected_peak_time_ET": "6:38 AM Eastern Time (ET) on March 3, 2026",
    "equivalents": {
        "UTC": "11:38 UTC on March 3, 2026",
        "CST": "5:38 AM CST on March 3, 2026",
        "MST": "4:38 AM MST on March 3, 2026",
        "PST": "3:38 AM PST on March 3, 2026"
    },
    "notes": "On March 3, 2026, U.S. remains on Standard Time (no DST yet), so ET=EST (UTC-5), PT=PST (UTC-8), etc."
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PeakTimeExtraction(BaseModel):
    """
    Information the answer claims about the Worm Moon peak time and its sources.
    """
    claimed_peak_time: Optional[str] = None  # e.g., "6:38 AM", "11:38", "06:38", etc.
    claimed_timezone: Optional[str] = None   # e.g., "ET", "EST", "UTC", "PST", "Pacific Time", etc.
    claimed_date: Optional[str] = None       # e.g., "March 3, 2026"
    source_urls: List[str] = Field(default_factory=list)  # Any reference URLs explicitly provided in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_peak_time() -> str:
    return """
    Your task is to extract from the answer:
    1) claimed_peak_time: The exact time string the answer claims for the full (peak illumination) time of the March 3, 2026 Worm Moon. Keep it exactly as written in the answer (e.g., '6:38 AM', '6:38 A.M.', '06:38', '11:38').
    2) claimed_timezone: The time zone string that accompanies that time in the answer (e.g., 'ET', 'EST', 'Eastern Time', 'UTC', 'GMT', 'PT', 'PST', 'Pacific Time', etc.). If the answer gives multiple equivalent times across zones, choose the one that is presented as the main answer; prefer Eastern Time/ET if present. If none is provided, return null.
    3) claimed_date: The date phrase tied to that time if it is explicitly stated (e.g., 'March 3, 2026'); otherwise null.
    4) source_urls: All reference URLs explicitly included in the answer text (including inline Markdown links). Return only actual URLs.

    IMPORTANT:
    - Do NOT infer or invent values. Only extract what is present in the answer.
    - If multiple times are present, pick the one asserted as the full moon's peak/maximum illumination time on March 3, 2026 (not moonrise, not eclipse maximum).
    - If the time string contains the time zone inline (e.g., '6:38 AM ET'), split them: 'claimed_peak_time' should be '6:38 AM', and 'claimed_timezone' should be 'ET'.
    """


# --------------------------------------------------------------------------- #
# Main evaluation logic                                                       #
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
    Evaluate an answer for the March 3, 2026 Worm Moon peak time task.
    """
    # 1) Initialize evaluator
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

    # 2) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_peak_time(),
        template_class=PeakTimeExtraction,
        extraction_name="peak_time_extraction"
    )

    # 3) Add ground truth info (for transparency in summary)
    evaluator.add_ground_truth(GROUND_TRUTH, gt_type="ground_truth_peak_time")

    # 4) Build rubric root (critical) and leaves
    rubric_root = evaluator.add_parallel(
        id="March_3_2026_Worm_Moon_Peak_Time",
        desc="Verify the response satisfies the question and all provided constraints for the March 3, 2026 Worm Moon peak time and related context.",
        parent=root,
        critical=True
    )

    # 4.1 Peak time exact and correct
    peak_time_node = evaluator.add_leaf(
        id="Peak_Time_Is_Exact_And_Correct",
        desc="Response states an exact peak time for the Worm Moon on March 3, 2026, and it matches 6:38 A.M. Eastern Time (ET) or a correct time-zone conversion equivalent (not a range; not a different date/event).",
        parent=rubric_root,
        critical=True
    )
    # Use simple verification focused on answer content and equivalence
    peak_time_claim = (
        "The answer explicitly provides a single exact peak time (not a range) for the full (peak illumination) time of "
        "the March 3, 2026 Worm Moon and includes its time zone. That stated time equals 6:38 AM Eastern Time (EST) on "
        "that date or a precisely equivalent conversion (examples: 11:38 UTC, 5:38 AM CST, 4:38 AM MST, 3:38 AM PST). "
        "Ignore times for moonrise/set or eclipse maximum; only the full moon peak/maximum illumination counts."
    )
    await evaluator.verify(
        claim=peak_time_claim,
        node=peak_time_node,
        additional_instruction=(
            "Consider ET as EST on 2026-03-03 (no DST yet). Allow reasonable formatting variants such as 'A.M.' vs 'AM', "
            "24-hour times (e.g., '06:38'), and case/spacing differences for zone names (ET/EST/Eastern Time). "
            "If the answer lacks a timezone, if it gives a range/approximation, or if the time does not match 6:38 AM ET "
            "or an exact equivalent, mark it incorrect."
        )
    )

    # 4.2 Reference URL verifies the peak time
    ref_node = evaluator.add_leaf(
        id="Reference_URL_Verifies_Peak_Time",
        desc="Response includes at least one reference URL, and the referenced source explicitly supports the stated peak time (including validating any time-zone conversion claimed).",
        parent=rubric_root,
        critical=True
    )
    # Build a claim grounded to the stated time when available; otherwise fall back to ET ground truth
    stated_time_str = (extracted.claimed_peak_time or "").strip()
    stated_tz_str = (extracted.claimed_timezone or "").strip()
    combined_stated = (f"{stated_time_str} {stated_tz_str}".strip()) if (stated_time_str or stated_tz_str) else None

    if combined_stated:
        ref_claim = (
            f"The referenced page explicitly confirms that the full moon (Worm Moon) on March 3, 2026 reached full/peak "
            f"illumination at {combined_stated}, or clearly provides an equivalent time in another zone corresponding to "
            f"the same instant. The page must refer to the full moon time (not moonrise/set or eclipse maximum)."
        )
    else:
        ref_claim = (
            "The referenced page explicitly confirms that the full moon (Worm Moon) on March 3, 2026 reached full/peak "
            "illumination at 6:38 AM Eastern Time (EST), or provides an equivalent time (e.g., 11:38 UTC) that clearly "
            "refers to the same instant. The page must refer to the full moon time (not moonrise/set or eclipse maximum)."
        )

    await evaluator.verify(
        claim=ref_claim,
        node=ref_node,
        sources=extracted.source_urls,
        additional_instruction=(
            "Accept pages that provide the exact full moon moment as a time and zone (e.g., 'Full Moon at 11:38 UTC'). "
            "If the page shows a different timezone, judge equivalence precisely (e.g., 6:38 AM ET == 11:38 UTC on 2026-03-03). "
            "Reject if the URLs are missing/invalid or if the page only mentions related but different times (e.g., moonrise, "
            "maximum eclipse, or a different date)."
        )
    )

    # 4.3 Mentions Worm Moon name
    worm_moon_node = evaluator.add_leaf(
        id="Mentions_Worm_Moon_Name",
        desc="Response states that the March full moon is traditionally called the 'Worm Moon'.",
        parent=rubric_root,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the March full moon is called the 'Worm Moon'.",
        node=worm_moon_node,
        additional_instruction="Allow minor formatting/case variants; look for an explicit mention of 'Worm Moon'."
    )

    # 4.4 Mentions total lunar eclipse visible from North America
    na_eclipse_node = evaluator.add_leaf(
        id="Mentions_Total_Lunar_Eclipse_Visible_From_North_America",
        desc="Response states that the total lunar eclipse was visible from North America.",
        parent=rubric_root,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the total lunar eclipse on March 3, 2026 was visible from North America.",
        node=na_eclipse_node,
        additional_instruction="Accept phrasing like 'visible across/throughout North America' or equivalent."
    )

    # 4.5 Mentions West Coast viewing especially good
    west_coast_node = evaluator.add_leaf(
        id="Mentions_West_Coast_Viewing_Is_Especially_Good",
        desc="Response states that viewing was especially good from the West Coast.",
        parent=rubric_root,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that viewing was especially good from the West Coast.",
        node=west_coast_node,
        additional_instruction="Accept phrases like 'best visibility on the West Coast' or 'particularly favorable from the West Coast (U.S./North America)'."
    )

    # 4.6 Mentions eclipse occurred right before sunrise
    before_sunrise_node = evaluator.add_leaf(
        id="Mentions_Eclipse_Occurred_Right_Before_Sunrise",
        desc="Response states that the eclipse occurred right before sunrise on March 3, 2026.",
        parent=rubric_root,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the eclipse occurred right before sunrise on March 3, 2026.",
        node=before_sunrise_node,
        additional_instruction="Accept equivalents like 'just before dawn', 'pre-dawn', or 'before sunrise that morning'."
    )

    # 4.7 Mentions Blood Moon term
    blood_moon_node = evaluator.add_leaf(
        id="Mentions_Blood_Moon_Term",
        desc="Response notes the event is also referred to as a 'Blood Moon' due to the lunar eclipse.",
        parent=rubric_root,
        critical=True
    )
    await evaluator.verify(
        claim="The answer mentions that the event is also referred to as a 'Blood Moon'.",
        node=blood_moon_node,
        additional_instruction="Allow variants like 'blood-red Moon' as long as the 'Blood Moon' notion of total lunar eclipse is clearly conveyed."
    )

    # 5) Return summary
    return evaluator.get_summary()