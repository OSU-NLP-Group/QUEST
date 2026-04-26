import asyncio
import logging
from datetime import date, timedelta
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "yosemite_wilderness_lottery_timeline_2026_week1"
TASK_DESCRIPTION = (
    "You are planning a backpacking trip to Yosemite National Park starting from the Taft Point/Sentinel Dome trailhead "
    "on Glacier Point Road. Your intended hiking start date is during the first week of July 2026 (specifically July 5–11, 2026).\n\n"
    "Since a wilderness permit is required for overnight stays in Yosemite Wilderness, you need to apply through the Recreation.gov "
    "lottery system, which allocates 60% of wilderness permits 24 weeks in advance of the hiking start date.\n\n"
    "Provide the complete lottery application timeline for this trip, including:\n\n"
    "1. Lottery Application Period: The specific dates and times (in Pacific Time) when the lottery application period opens and closes\n"
    "2. Results Notification: The date and time (in Pacific Time) by which lottery results will be announced\n"
    "3. Acceptance Deadline: The date and time (in Pacific Time) by which lottery winners must accept their permits\n"
    "4. Leftover Permits Release: The date and time (in Pacific Time) when any remaining permits from the lottery become available "
    "on Recreation.gov on a first-come, first-served basis\n\n"
    "For each timeline component, provide the exact date, time, and day of the week. Include a reference URL to the official Yosemite "
    "National Park wilderness permit reservation window page that shows the lottery schedule."
)

# --------------------------------------------------------------------------- #
# Helpers to compute expected dates                                           #
# --------------------------------------------------------------------------- #
def sunday_of_week(d: date) -> date:
    # Return the Sunday of the week containing d (Sunday=0)
    # Python weekday(): Monday=0...Sunday=6, so compute offset to Sunday
    offset = (d.weekday() + 1) % 7  # Monday=0 -> 1; Sunday=6 -> 0
    return d - timedelta(days=offset)


def format_date_long(d: date) -> str:
    # Example: "January 18, 2026"
    return d.strftime("%B %-d, %Y") if hasattr(d, "strftime") else str(d)


def format_date_long_portable(d: date) -> str:
    # On some systems, %-d not supported; provide fallback
    try:
        return d.strftime("%B %-d, %Y")
    except Exception:
        return d.strftime("%B %d, %Y").replace(" 0", " ")


def weekday_name(d: date) -> str:
    return d.strftime("%A")


def compute_expected_schedule_for_week(start_week_sunday: date) -> Dict[str, Dict[str, Any]]:
    """
    Given the hiking start week Sunday, compute the expected lottery schedule that opens 24 weeks prior.
    Returns a dict with expected dates and canonical time strings for each milestone.
    """
    app_week_sunday = start_week_sunday - timedelta(weeks=24)
    app_week_saturday = app_week_sunday + timedelta(days=6)
    results_monday = app_week_saturday + timedelta(days=2)   # following Monday
    acceptance_thursday = results_monday + timedelta(days=3) # Thursday
    leftover_friday = results_monday + timedelta(days=4)     # Friday

    return {
        "application_open": {
            "date": app_week_sunday,
            "weekday": weekday_name(app_week_sunday),
            "time": "12:01 am",
            "timezone": "PT",
        },
        "application_close": {
            "date": app_week_saturday,
            "weekday": weekday_name(app_week_saturday),
            "time": "11:59 pm",
            "timezone": "PT",
        },
        "results_notification": {
            "date": results_monday,
            "weekday": weekday_name(results_monday),
            "time": "5:00 pm",
            "timezone": "PT",
        },
        "acceptance_deadline": {
            "date": acceptance_thursday,
            "weekday": weekday_name(acceptance_thursday),
            "time": "11:59 pm",
            "timezone": "PT",
        },
        "leftover_permits_release": {
            "date": leftover_friday,
            "weekday": weekday_name(leftover_friday),
            "time": "9:00 am",
            "timezone": "PT",
        },
    }


# For this task: hiking start week is July 5–11, 2026; its Sunday is July 5, 2026.
HIKE_WEEK_SUNDAY = date(2026, 7, 5)
EXPECTED = compute_expected_schedule_for_week(HIKE_WEEK_SUNDAY)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Milestone(BaseModel):
    date: Optional[str] = None           # e.g., "January 18, 2026" or "Sun, Jan 18, 2026"
    day_of_week: Optional[str] = None    # e.g., "Sunday"
    time: Optional[str] = None           # e.g., "12:01 am"
    timezone: Optional[str] = None       # e.g., "PT", "PST", "PDT"
    note: Optional[str] = None           # any extra details stated


class TimelineExtraction(BaseModel):
    reference_url: Optional[str] = None

    application_open: Optional[Milestone] = None
    application_close: Optional[Milestone] = None
    results_notification: Optional[Milestone] = None
    acceptance_deadline: Optional[Milestone] = None
    leftover_permits_release: Optional[Milestone] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_timeline() -> str:
    return """
    Extract the Yosemite wilderness permit weekly lottery timeline information as presented in the answer.

    You must extract:
    - reference_url: A single URL to the official Yosemite National Park (NPS) webpage that shows the reservation window / lottery schedule.
      Return a single string URL (not markdown). If multiple are given, return the most directly relevant NPS page.

    For each milestone below, extract the exact fields as they appear in the answer:
    - application_open: date, day_of_week, time, timezone, note
    - application_close: date, day_of_week, time, timezone, note
    - results_notification: date, day_of_week, time, timezone, note
    - acceptance_deadline: date, day_of_week, time, timezone, note
    - leftover_permits_release: date, day_of_week, time, timezone, note

    Rules:
    1) date should be the exact calendar date string the answer states (e.g., "January 18, 2026" or "Sun, Jan 18, 2026").
    2) day_of_week should be the explicit day name if provided (e.g., "Sunday", "Mon", etc.). If not provided, return null.
    3) time should capture the specific time given (e.g., "12:01 am", "11:59 pm", "5 pm", "9:00 am"). Preserve the answer’s formatting if possible.
    4) timezone should capture "PT"/"PST"/"PDT" or the phrase "Pacific Time" if the answer includes it. If not provided, return null.
    5) note can capture any extra essential phrase like "by 5 pm", "first-come, first-served on Recreation.gov", etc. If none, return null.

    If any field is missing in the answer, return null for that field. Do not invent information.
    """


# --------------------------------------------------------------------------- #
# Verification utilities                                                      #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _pt_mentioned(tz: Optional[str]) -> bool:
    if not tz:
        return False
    tz_low = tz.lower()
    return ("pt" in tz_low) or ("pacific" in tz_low) or ("pst" in tz_low) or ("pdt" in tz_low)


def _expected_date_str(key: str) -> str:
    return format_date_long_portable(EXPECTED[key]["date"])


def _expected_weekday_str(key: str) -> str:
    return EXPECTED[key]["weekday"]


def _expected_time_str(key: str) -> str:
    return EXPECTED[key]["time"]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_official_reference_verification(evaluator: Evaluator, parent_node, extracted: TimelineExtraction):
    """
    Create verification nodes for the official reference URL.
    """
    node = evaluator.add_parallel(
        id="official_reference_url",
        desc="Provide a reference URL to the official Yosemite National Park wilderness permit reservation window / lottery schedule page that supports the schedule used.",
        parent=parent_node,
        critical=True
    )

    # Leaf 1: URL provided
    evaluator.add_custom_node(
        result=_nonempty(extracted.reference_url),
        id="official_reference_url_provided",
        desc="Official reference URL is provided in the answer",
        parent=node,
        critical=True
    )

    # Leaf 2: URL is an NPS Yosemite page (domain heuristic)
    evaluator.add_custom_node(
        result=bool(extracted.reference_url and "nps.gov" in extracted.reference_url.lower()),
        id="official_reference_url_is_nps",
        desc="Official URL is an NPS domain page",
        parent=node,
        critical=True
    )

    # Leaf 3: Page supports the weekly lottery schedule and 24 weeks rule
    support_leaf = evaluator.add_leaf(
        id="official_reference_url_supports_schedule",
        desc="Official page shows the weekly lottery schedule and the '24 weeks in advance' rule (with Sunday open 12:01 am PT; Saturday close 11:59 pm PT; results Monday by 5 pm PT; acceptance Thursday 11:59 pm PT; leftover Friday 9 am PT).",
        parent=node,
        critical=True
    )
    if _nonempty(extracted.reference_url):
        claim = (
            "This page is an official Yosemite National Park (NPS) page that explains the wilderness permit weekly lottery schedule, "
            "including that 60% of permits are released 24 weeks in advance and the weekly timing details: "
            "applications open Sunday at 12:01 am PT, close Saturday at 11:59 pm PT, results posted Monday by 5:00 pm PT, "
            "winners must accept by Thursday 11:59 pm PT, and leftover permits are released Friday at 9:00 am PT."
        )
        await evaluator.verify(
            claim=claim,
            node=support_leaf,
            sources=extracted.reference_url,
            additional_instruction="Allow minor wording differences for the times (e.g., 'by 5 pm'). Confirm both the 24-week rule and the weekly schedule are shown."
        )
    else:
        # Without URL, verification cannot proceed
        await evaluator.verify(
            claim="No URL provided.",
            node=support_leaf,
            sources=None,
            additional_instruction="This should fail because no official URL was supplied."
        )


async def build_milestone_verification(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    milestone: Optional[Milestone],
    expected_key: str,
    require_fcfs_recreation_gov: bool = False
):
    """
    Build verification sub-tree for one milestone.
    """
    node = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=True
    )

    # 1) Existence of components: date, day_of_week, time, timezone must all be present per rubric
    exists_leaf = evaluator.add_custom_node(
        result=all([_nonempty(milestone.date if milestone else None),
                    _nonempty(milestone.day_of_week if milestone else None),
                    _nonempty(milestone.time if milestone else None),
                    _nonempty(milestone.timezone if milestone else None)]),
        id=f"{node_id}_fields_present",
        desc=f"{node_id} has explicit date, day-of-week, time, and timezone (PT) in the answer.",
        parent=node,
        critical=True
    )

    # 2) Time zone is PT/PST/PDT
    tz_leaf = evaluator.add_custom_node(
        result=_pt_mentioned(milestone.timezone if milestone else None),
        id=f"{node_id}_mentions_pt",
        desc=f"{node_id} explicitly indicates Pacific Time (PT).",
        parent=node,
        critical=True
    )

    # 3) Day-of-week and time correctness (verified against the rubric's expected weekly schedule)
    time_day_leaf = evaluator.add_leaf(
        id=f"{node_id}_time_day_correct",
        desc=f"{node_id} day-of-week and time match Yosemite weekly lottery schedule.",
        parent=node,
        critical=True
    )
    exp_weekday = _expected_weekday_str(expected_key)
    exp_time = _expected_time_str(expected_key)
    await evaluator.verify(
        claim=f"For this milestone, the answer states it occurs on {exp_weekday} at {exp_time} PT (allowing variants like 'by {exp_time}' when applicable).",
        node=time_day_leaf,
        sources=None,
        additional_instruction="Read the answer and verify it states the expected day-of-week and time in Pacific Time. Allow minor formatting or wording (e.g., 'by 5 pm')."
    )

    # 4) Date correctness (matches the expected calendar date for this schedule)
    date_leaf = evaluator.add_leaf(
        id=f"{node_id}_date_correct",
        desc=f"{node_id} date matches the expected date from the 24-weeks-in-advance schedule for the July 5–11, 2026 hiking week.",
        parent=node,
        critical=True
    )
    exp_date_str = _expected_date_str(expected_key)
    await evaluator.verify(
        claim=f"The date stated in the answer for this milestone is {exp_date_str}.",
        node=date_leaf,
        sources=None,
        additional_instruction="Accept reasonable date formatting variants (e.g., 'Jan 18, 2026' vs 'January 18, 2026')."
    )

    # 5) Additional requirement for leftover permits: must say Recreation.gov first-come/first-served
    if require_fcfs_recreation_gov:
        fcfs_leaf = evaluator.add_leaf(
            id=f"{node_id}_fcfs_recreationgov_mentioned",
            desc="Leftover permits are described as being released on Recreation.gov on a first-come, first-served basis.",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim="The answer explicitly states that leftover permits are released on Recreation.gov on a first-come, first-served basis.",
            node=fcfs_leaf,
            sources=None,
            additional_instruction="Check exact phrasing allowing small variations (e.g., 'first come first served')."
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
    Evaluate an answer for the Yosemite wilderness permit lottery schedule for hiking dates July 5–11, 2026.
    """
    # Initialize evaluator (root is a non-critical container)
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

    # Add the top-level critical node that matches the rubric's root
    main = evaluator.add_parallel(
        id="wilderness_permit_lottery_timeline",
        desc="Complete Yosemite wilderness permit lottery timeline for hiking start dates July 5–11, 2026, with all required milestones (date, day-of-week, time in PT) and an official NPS reference URL.",
        parent=root,
        critical=True
    )

    # Extract structured info from the answer
    extraction: TimelineExtraction = await evaluator.extract(
        prompt=prompt_extract_timeline(),
        template_class=TimelineExtraction,
        extraction_name="timeline_extraction"
    )

    # Store ground-truth computed schedule as custom info
    evaluator.add_custom_info(
        info={
            "start_week_sunday": format_date_long_portable(HIKE_WEEK_SUNDAY),
            "expected": {
                k: {
                    "date": format_date_long_portable(v["date"]),
                    "weekday": v["weekday"],
                    "time": v["time"],
                    "timezone": v["timezone"],
                } for k, v in EXPECTED.items()
            }
        },
        info_type="expected_schedule",
        info_name="expected_schedule_for_july_5_11_2026"
    )

    # 1) Official reference URL verification
    await build_official_reference_verification(evaluator, main, extraction)

    # 2) Application period open
    await build_milestone_verification(
        evaluator=evaluator,
        parent_node=main,
        node_id="application_period_open",
        node_desc="Provide the lottery application opening milestone with exact date, day-of-week, and time in PT; Sunday 12:01 am PT; corresponds to 24 weeks before the hiking start week.",
        milestone=extraction.application_open,
        expected_key="application_open",
    )

    # 3) Application period close
    await build_milestone_verification(
        evaluator=evaluator,
        parent_node=main,
        node_id="application_period_close",
        node_desc="Provide the lottery application closing milestone with exact date, day-of-week, and time in PT; Saturday 11:59 pm PT of the same application week.",
        milestone=extraction.application_close,
        expected_key="application_close",
    )

    # 4) Results notification
    await build_milestone_verification(
        evaluator=evaluator,
        parent_node=main,
        node_id="results_notification",
        node_desc="Provide the results notification milestone with exact date, day-of-week, and time in PT; the following Monday by 5:00 pm PT after the application period closes.",
        milestone=extraction.results_notification,
        expected_key="results_notification",
    )

    # 5) Acceptance deadline
    await build_milestone_verification(
        evaluator=evaluator,
        parent_node=main,
        node_id="acceptance_deadline",
        node_desc="Provide the acceptance deadline milestone with exact date, day-of-week, and time in PT; Thursday 11:59 pm PT following results notification.",
        milestone=extraction.acceptance_deadline,
        expected_key="acceptance_deadline",
    )

    # 6) Leftover permits release
    await build_milestone_verification(
        evaluator=evaluator,
        parent_node=main,
        node_id="leftover_permits_release",
        node_desc="Provide the leftover permits release milestone with exact date, day-of-week, and time in PT; Friday 9:00 am PT following acceptance deadline, and state it is on Recreation.gov first-come/first-served.",
        milestone=extraction.leftover_permits_release,
        expected_key="leftover_permits_release",
        require_fcfs_recreation_gov=True
    )

    # Return evaluation summary
    return evaluator.get_summary()