import asyncio
import logging
import calendar
from datetime import date, timedelta
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "inyo_wilderness_planning_2026_07_15"
TASK_DESCRIPTION = (
    "Provide comprehensive planning information for an 8-person John Muir Wilderness overnight backpacking trip "
    "via Inyo National Forest with entry date July 15, 2026. Topics include official reservation platform and URL, "
    "quota release system (percentages and time of day), booking windows for the specified date, complete permit cost "
    "breakdown and total cost, and additional mandatory requirements (with deadlines). All items should be supported "
    "by official reference URLs."
)


# --------------------------------------------------------------------------- #
# Data Models for Extracted Information                                       #
# --------------------------------------------------------------------------- #
class PlatformInfo(BaseModel):
    name: Optional[str] = None
    permits_url: Optional[str] = None
    ref_urls: List[str] = Field(default_factory=list)


class QuotaInfo(BaseModel):
    release_60_percent: Optional[str] = None
    release_40_percent: Optional[str] = None
    opening_time: Optional[str] = None
    ref_urls: List[str] = Field(default_factory=list)


class BookingWindows(BaseModel):
    first_window: Optional[str] = None
    second_window: Optional[str] = None
    ref_urls: List[str] = Field(default_factory=list)


class CostInfo(BaseModel):
    reservation_fee: Optional[str] = None
    per_person_fee: Optional[str] = None
    total_cost: Optional[str] = None
    ref_urls: List[str] = Field(default_factory=list)


class AdditionalReqInfo(BaseModel):
    campfire_permit: Optional[str] = None
    permit_print_deadline: Optional[str] = None
    group_size_compliance: Optional[str] = None
    ref_urls: List[str] = Field(default_factory=list)


class PlanningExtraction(BaseModel):
    platform: Optional[PlatformInfo] = None
    quota: Optional[QuotaInfo] = None
    booking: Optional[BookingWindows] = None
    cost: Optional[CostInfo] = None
    additional: Optional[AdditionalReqInfo] = None


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_planning() -> str:
    return """
Extract the following structured information as presented in the answer. Extract only what is explicitly mentioned.

1) platform:
   - name: The official online platform for reserving Inyo National Forest wilderness permits (e.g., "Recreation.gov").
   - permits_url: The specific permits page URL for Inyo National Forest wilderness permits (e.g., https://www.recreation.gov/permits/233262).
   - ref_urls: A list of supporting reference URLs cited for the platform information (include the permits_url if it is used as a reference).

2) quota:
   - release_60_percent: The statement describing that 60% of quota is released 6 months in advance (if mentioned).
   - release_40_percent: The statement describing that 40% of quota is released 2 weeks in advance on the same day of the week (if mentioned).
   - opening_time: The time reservations open (e.g., "7:00 AM PT/PST").
   - ref_urls: A list of supporting reference URLs cited for the quota release system information.

3) booking:
   - first_window: The stated date/time when the first booking window opens for a July 15, 2026 entry (60% release).
   - second_window: The stated date/time when the second booking window opens for a July 15, 2026 entry (40% release).
   - ref_urls: A list of reference URLs cited to support the booking window calculations/policy.

4) cost:
   - reservation_fee: The stated non-refundable reservation fee per permit (e.g., "$6 per permit").
   - per_person_fee: The stated per-person recreation fee for non-Whitney Zone permits (e.g., "$5 per person").
   - total_cost: The stated total for the 8-person group including all fees (e.g., "$46").
   - ref_urls: A list of supporting reference URLs cited for the cost information.

5) additional:
   - campfire_permit: The statement that a California Campfire Permit is required for campfires or operating stoves (if mentioned).
   - permit_print_deadline: The statement about the deadline to print the wilderness permit (e.g., "must be printed before 10:00 AM on the entry date or canceled").
   - group_size_compliance: The statement verifying the 8-person group is within limits and any rules about combining groups (e.g., "max 10 people for camping; groups cannot combine to exceed 15 people").
   - ref_urls: A list of supporting reference URLs cited for the additional requirements.

Rules for URL extraction:
- Extract only actual URLs explicitly present in the answer.
- If the answer uses markdown links, extract the underlying URLs.
- Deduplicate URLs within each list.
"""


# --------------------------------------------------------------------------- #
# Utility Functions                                                            #
# --------------------------------------------------------------------------- #
def last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def subtract_months(d: date, months: int) -> date:
    # Compute year and month
    y = d.year + ((d.month - 1 - months) // 12)
    m = ((d.month - 1 - months) % 12) + 1
    day = min(d.day, last_day_of_month(y, m))
    return date(y, m, day)


def format_release_dt(d: date) -> str:
    # Format consistently as "Month D, YYYY at 7:00 AM PT"
    return f"{d.strftime('%B')} {d.day}, {d.year} at 7:00 AM PT"


def merge_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for urls in url_lists:
        for u in urls or []:
            if u and u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification Subtrees                                                        #
# --------------------------------------------------------------------------- #
async def verify_question1_platform(evaluator: Evaluator, parent, data: Optional[PlatformInfo]) -> None:
    node = evaluator.add_parallel(
        id="q1_platform",
        desc="Identify the official online platform and provide the specific URL for Inyo National Forest wilderness permits",
        parent=parent,
        critical=True
    )

    platform_name_leaf = evaluator.add_leaf(
        id="q1_platform_name",
        desc="Identify Recreation.gov as the official online platform for Inyo wilderness permits",
        parent=node,
        critical=True
    )
    platform_sources = merge_urls([data.permits_url] if data and data.permits_url else [], data.ref_urls if data else [])
    await evaluator.verify(
        claim="The official online platform for reserving Inyo National Forest wilderness permits is Recreation.gov.",
        node=platform_name_leaf,
        sources=platform_sources,
        additional_instruction="Allow phrasing variants that clearly indicate Recreation.gov is the official reservation platform."
    )

    platform_url_leaf = evaluator.add_leaf(
        id="q1_platform_url",
        desc="Provide the specific URL recreation.gov/permits/233262 for Inyo wilderness permits",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is the official Inyo National Forest Wilderness Permits page on Recreation.gov.",
        node=platform_url_leaf,
        sources=(data.permits_url if data and data.permits_url else None),
        additional_instruction="Accept if the page clearly indicates 'Inyo National Forest Wilderness Permits' and is hosted on recreation.gov."
    )

    ref_presence = evaluator.add_custom_node(
        result=(bool(data and data.ref_urls and len(data.ref_urls) > 0)),
        id="q1_platform_ref_urls_present",
        desc="Provide a reference URL supporting the platform information",
        parent=node,
        critical=True
    )


async def verify_question2_quota(evaluator: Evaluator, parent, data: Optional[QuotaInfo], platform: Optional[PlatformInfo]) -> None:
    node = evaluator.add_parallel(
        id="q2_quota",
        desc="Explain how the quota release system works for Inyo wilderness permits",
        parent=parent,
        critical=True
    )

    sources = merge_urls(data.ref_urls if data else [], [platform.permits_url] if platform and platform.permits_url else [])

    sixty_leaf = evaluator.add_leaf(
        id="q2_sixty_percent",
        desc="State that 60% of quota is released 6 months in advance",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="For Inyo National Forest wilderness permits, 60% of the daily entry quota is released 6 months in advance.",
        node=sixty_leaf,
        sources=sources,
        additional_instruction="Look for explicit mention of '60%' and '6 months' on official pages (e.g., Recreation.gov Inyo permits page)."
    )

    forty_leaf = evaluator.add_leaf(
        id="q2_forty_percent",
        desc="State that 40% of quota is released 2 weeks in advance on the same day of the week",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="For Inyo National Forest wilderness permits, the remaining 40% of the quota is released 2 weeks before the entry date on the same day of the week.",
        node=forty_leaf,
        sources=sources,
        additional_instruction="Look for wording that the 'second release is 2 weeks in advance at 7:00 AM PT and follows the same weekday as the entry date'. Minor phrasing differences are acceptable."
    )

    time_leaf = evaluator.add_leaf(
        id="q2_opening_time",
        desc="State that reservations open at 7:00 AM Pacific Standard Time",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Inyo wilderness permit reservations open at 7:00 AM Pacific Time (PT).",
        node=time_leaf,
        sources=sources,
        additional_instruction="Treat 'Pacific Time', 'PT', 'PST', or 'PDT' as equivalent for the opening time."
    )

    ref_presence = evaluator.add_custom_node(
        result=(bool(data and data.ref_urls and len(data.ref_urls) > 0)),
        id="q2_quota_ref_urls_present",
        desc="Provide a reference URL supporting the quota release system information",
        parent=node,
        critical=True
    )


async def verify_question3_booking_windows(evaluator: Evaluator, parent, booking: Optional[BookingWindows]) -> None:
    node = evaluator.add_parallel(
        id="q3_booking_windows",
        desc="Calculate the specific dates and times when permits for July 15, 2026 entry become available",
        parent=parent,
        critical=True
    )

    entry_date = date(2026, 7, 15)
    expected_first_date = subtract_months(entry_date, 6)  # 6 months prior
    expected_second_date = entry_date - timedelta(days=14)  # 2 weeks prior
    expected_first_str = format_release_dt(expected_first_date)
    expected_second_str = format_release_dt(expected_second_date)

    # First window (60% release)
    first_leaf = evaluator.add_leaf(
        id="q3_first_window",
        desc="Determine that the first window (60% quota) opens on January 15, 2026 at 7 AM PST",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For a July 15, 2026 entry date, the first booking window (60% release, 6 months prior at 7:00 AM PT) opens on {expected_first_str}.",
        node=first_leaf,
        sources=None,
        additional_instruction=(
            "Use basic date arithmetic: 6 months before July 15, 2026 is January 15, 2026. "
            "Treat 'PT/PST/PDT' as equivalent for this purpose."
        )
    )

    # Second window (40% release)
    second_leaf = evaluator.add_leaf(
        id="q3_second_window",
        desc="Determine that the second window (40% quota) opens on July 1, 2026 at 7 AM PST",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For a July 15, 2026 entry date, the second booking window (40% release, 2 weeks prior at 7:00 AM PT, same weekday) opens on {expected_second_str}.",
        node=second_leaf,
        sources=None,
        additional_instruction=(
            "Use basic date arithmetic: 2 weeks before July 15, 2026 is July 1, 2026. "
            "Treat 'PT/PST/PDT' as equivalent for this purpose."
        )
    )

    # Reference URL presence for booking windows (policy support)
    ref_presence = evaluator.add_custom_node(
        result=(bool(booking and booking.ref_urls and len(booking.ref_urls) > 0)),
        id="q3_booking_refs_present",
        desc="Provide a reference URL supporting the booking window calculations",
        parent=node,
        critical=True
    )


async def verify_question4_cost(evaluator: Evaluator, parent, cost: Optional[CostInfo], platform: Optional[PlatformInfo], group_size: int) -> None:
    node = evaluator.add_parallel(
        id="q4_total_cost",
        desc="Provide complete cost breakdown and total cost for the 8-person group permit",
        parent=parent,
        critical=False
    )

    sources = merge_urls(cost.ref_urls if cost else [], [platform.permits_url] if platform and platform.permits_url else [])

    # Reservation fee ($6 per permit)
    res_fee_leaf = evaluator.add_leaf(
        id="q4_reservation_fee",
        desc="State the $6 non-refundable reservation fee per permit",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="There is a non-refundable reservation fee of $6 per wilderness permit for Inyo National Forest on Recreation.gov.",
        node=res_fee_leaf,
        sources=sources,
        additional_instruction="Accept equivalent phrasing that clearly indicates a $6 reservation/transaction fee per permit."
    )

    # Per-person recreation fee ($5 per person, non-Whitney Zone)
    per_person_leaf = evaluator.add_leaf(
        id="q4_per_person_fee",
        desc="State the $5 per-person recreation fee for non-Whitney Zone areas",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="For non-Whitney Zone entries, the recreation fee is $5 per person.",
        node=per_person_leaf,
        sources=sources,
        additional_instruction="Ensure the source indicates $5/person applies to non-Whitney Zone wilderness permits."
    )

    # Total cost calculation check
    total_leaf = evaluator.add_leaf(
        id="q4_total_cost_calc",
        desc="Calculate the total cost as $6 + ($5 × 8) = $46",
        parent=node,
        critical=False
    )
    computed_total = 6 + 5 * group_size
    await evaluator.verify(
        claim=f"Given a $6 reservation fee and $5 per person for {group_size} people, the total cost is ${computed_total}.",
        node=total_leaf,
        sources=None,
        additional_instruction="This is a straightforward arithmetic check."
    )

    # Reference URL presence for cost info
    ref_presence = evaluator.add_custom_node(
        result=(bool(cost and cost.ref_urls and len(cost.ref_urls) > 0)),
        id="q4_cost_refs_present",
        desc="Provide a reference URL supporting the cost information",
        parent=node,
        critical=False
    )


async def verify_question5_additional(evaluator: Evaluator, parent, additional: Optional[AdditionalReqInfo], platform: Optional[PlatformInfo], group_size: int) -> None:
    node = evaluator.add_parallel(
        id="q5_additional_requirements",
        desc="List mandatory permits, documents, deadlines, and verify group size compliance",
        parent=parent,
        critical=False
    )

    sources = merge_urls(additional.ref_urls if additional else [], [platform.permits_url] if platform and platform.permits_url else [])

    # California Campfire Permit requirement
    campfire_leaf = evaluator.add_leaf(
        id="q5_campfire_permit",
        desc="State that a California Campfire Permit is required for campfires or operating stoves",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="A California Campfire Permit is required to build campfires or operate portable stoves on public lands, including Inyo National Forest wilderness.",
        node=campfire_leaf,
        sources=sources,
        additional_instruction="Accept official sources (USFS, CAL FIRE, or Recreation.gov) that clearly state the campfire permit requirement for stoves/campfires."
    )

    # Permit print deadline
    print_deadline_leaf = evaluator.add_leaf(
        id="q5_permit_print_deadline",
        desc="State that the wilderness permit must be printed before 10:00 AM on the entry date (July 15, 2026) or will be canceled",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim="Inyo National Forest wilderness permits must be printed before 10:00 AM on the entry date; otherwise the reservation will be canceled.",
        node=print_deadline_leaf,
        sources=sources,
        additional_instruction="This rule is commonly stated on the Recreation.gov Inyo permits page; allow slight wording variations (e.g., 'no later than 10 AM')."
    )

    # Group size compliance (single combined check as specified)
    group_size_leaf = evaluator.add_leaf(
        id="q5_group_size_compliance",
        desc="Verify that the 8-person group complies with the 10-person maximum limit for wilderness camping and note that groups cannot combine to exceed 15 people",
        parent=node,
        critical=False
    )
    await evaluator.verify(
        claim=(
            "The maximum wilderness camping group size limit is 10 people, and groups may not combine to exceed 15 people; "
            "therefore, an 8-person group complies with these limits."
        ),
        node=group_size_leaf,
        sources=sources,
        additional_instruction="Verify the specific group size limits and the 'no combining groups' rule from official sources for the John Muir Wilderness/Inyo National Forest."
    )

    # Reference URL presence for additional requirements
    ref_presence = evaluator.add_custom_node(
        result=(bool(additional and additional.ref_urls and len(additional.ref_urls) > 0)),
        id="q5_additional_refs_present",
        desc="Provide reference URLs supporting the additional requirements information",
        parent=node,
        critical=False
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry Point                                                  #
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

    # Extract structured planning info
    extracted: PlanningExtraction = await evaluator.extract(
        prompt=prompt_extract_planning(),
        template_class=PlanningExtraction,
        extraction_name="planning_extraction"
    )

    # Add derived ground truth (expected booking windows for reference in summary)
    entry_date = date(2026, 7, 15)
    expected_first_date = subtract_months(entry_date, 6)
    expected_second_date = entry_date - timedelta(days=14)
    evaluator.add_ground_truth({
        "entry_date": "July 15, 2026",
        "group_size": 8,
        "expected_first_window": format_release_dt(expected_first_date),
        "expected_second_window": format_release_dt(expected_second_date)
    }, gt_type="expected_booking_windows")

    # Top-level planning node (non-critical to allow partial credit across sections)
    planning_node = evaluator.add_parallel(
        id="planning_main",
        desc="Provide comprehensive planning information for an 8-person John Muir Wilderness overnight backpacking trip via Inyo National Forest with entry date July 15, 2026",
        parent=root,
        critical=False
    )

    # Build subtrees
    await verify_question1_platform(evaluator, planning_node, extracted.platform)
    await verify_question2_quota(evaluator, planning_node, extracted.quota, extracted.platform if extracted else None)
    await verify_question3_booking_windows(evaluator, planning_node, extracted.booking)
    await verify_question4_cost(evaluator, planning_node, extracted.cost, extracted.platform if extracted else None, group_size=8)
    await verify_question5_additional(evaluator, planning_node, extracted.additional, extracted.platform if extracted else None, group_size=8)

    return evaluator.get_summary()