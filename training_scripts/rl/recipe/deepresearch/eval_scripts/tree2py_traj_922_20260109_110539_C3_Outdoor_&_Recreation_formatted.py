import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "gtet_permit_reservation_window"
TASK_DESCRIPTION = "For a group of 4 hikers planning a 3-day, 2-night backpacking trip on the Paintbrush-Cascade Canyon Loop in Grand Teton National Park with a start date of July 15, 2026, when does the advance reservation window open on Recreation.gov (specify both date and time), and what is the deadline for picking up the reserved permit in person?"


class ReservationWindow(BaseModel):
    opening_date: Optional[str] = None
    opening_time: Optional[str] = None
    opening_timezone: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PickupDeadline(BaseModel):
    deadline_time: Optional[str] = None
    date_relation: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PermitAnswerExtraction(BaseModel):
    reservation_window: Optional[ReservationWindow] = None
    pickup_deadline: Optional[PickupDeadline] = None


def prompt_extract_permit_info() -> str:
    return """
    Extract exactly what the answer claims about Grand Teton backcountry permit reservations and pickup deadlines.

    Return a JSON object with two sections:

    reservation_window:
      - opening_date: The date the answer says Recreation.gov advance reservations open for peak season permits (e.g., "January 7", "Jan 7", "January 7th"). Use the exact phrasing from the answer.
      - opening_time: The time the answer says the reservations open (e.g., "8:00 a.m.", "8 AM", "8am").
      - opening_timezone: The timezone string as stated (e.g., "MST", "Mountain Standard Time", "Mountain Time", "MT").
      - sources: All URLs cited in the answer that are intended to support the reservation opening details. Include only valid URLs.

    pickup_deadline:
      - deadline_time: The time by which reserved permits must be picked up in person (e.g., "10:00 a.m.", "10 AM").
      - date_relation: How the answer relates the pickup deadline to the permit date (e.g., "on the permit start date", "by 10 a.m. on the start date", "morning of start date").
      - sources: All URLs cited in the answer that support the pickup deadline rules. Include only valid URLs.

    If a field is not stated in the answer, set it to null and return an empty list for sources when none are cited.
    """


async def build_verification_tree(evaluator: Evaluator, extracted: PermitAnswerExtraction) -> None:
    root = evaluator.find_node("root")

    permit_node = evaluator.add_parallel(
        id="Permit_Acquisition_Process_Verification",
        desc="Verify the answer states when the advance reservation window opens (date and time) and the in-person pickup deadline.",
        parent=root,
        critical=True
    )

    advance_node = evaluator.add_parallel(
        id="Advance_Reservation_Window_Open",
        desc="Specify when Recreation.gov advance reservations open for peak season permits.",
        parent=permit_node,
        critical=True
    )

    date_leaf = evaluator.add_leaf(
        id="Reservation_Opening_Date",
        desc="States the opening date is January 7 (for peak season advance reservations).",
        parent=advance_node,
        critical=True
    )
    date_claim = (
        "The answer explicitly states that Recreation.gov advance reservations for Grand Teton peak season permits "
        "open on January 7."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        additional_instruction=(
            "Check only the answer text for whether it states January 7 as the opening date. "
            "Accept minor variants like 'Jan 7' or 'January 7th'. Do not require a year."
        )
    )

    time_leaf = evaluator.add_leaf(
        id="Reservation_Opening_Time_and_Timezone",
        desc="States the opening time is 8:00 a.m. Mountain Standard Time (MST).",
        parent=advance_node,
        critical=True
    )
    time_claim = (
        "The answer explicitly states that the advance reservation opening time is 8:00 a.m. Mountain Standard Time (MST)."
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_leaf,
        additional_instruction=(
            "Check only the answer text for whether it includes both the time '8:00 a.m.' (allow '8 AM'/'8am') "
            "and the timezone specifically as 'MST' or 'Mountain Standard Time'. "
            "If it says only 'Mountain Time (MT)' without 'MST', consider that insufficient for this check."
        )
    )

    pickup_node = evaluator.add_parallel(
        id="In_Person_Permit_Pickup_Deadline",
        desc="Specify the deadline for picking up the reserved permit in person.",
        parent=permit_node,
        critical=True
    )

    pickup_time_leaf = evaluator.add_leaf(
        id="Pickup_Deadline_Time",
        desc="States permits must be picked up by 10:00 a.m.",
        parent=pickup_node,
        critical=True
    )
    pickup_time_claim = "The answer explicitly states that reserved permits must be picked up by 10:00 a.m."
    await evaluator.verify(
        claim=pickup_time_claim,
        node=pickup_time_leaf,
        additional_instruction="Accept '10:00 a.m.', '10 AM', or '10am' as equivalent phrasing."
    )

    pickup_relation_leaf = evaluator.add_leaf(
        id="Pickup_Deadline_Date_Relation",
        desc="States the 10:00 a.m. pickup deadline is on the permit start date.",
        parent=pickup_node,
        critical=True
    )
    pickup_relation_claim = (
        "The answer explicitly states that the 10:00 a.m. pickup deadline is on the permit start date."
    )
    await evaluator.verify(
        claim=pickup_relation_claim,
        node=pickup_relation_leaf,
        additional_instruction=(
            "Accept phrasing such as 'by 10 a.m. on the start date', 'morning of the permit start date by 10 a.m.', "
            "or equivalent language clearly indicating the deadline applies on the start date (e.g., July 15, 2026 for this trip)."
        )
    )

    # Record extracted info and any sources for transparency
    evaluator.add_custom_info(
        info={
            "reservation_window_extracted": extracted.reservation_window.dict() if extracted.reservation_window else {},
            "pickup_deadline_extracted": extracted.pickup_deadline.dict() if extracted.pickup_deadline else {}
        },
        info_type="extraction_summary",
        info_name="extracted_permit_details"
    )


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
    evaluator.initialize(
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_permit_info(),
        template_class=PermitAnswerExtraction,
        extraction_name="permit_answer_extraction"
    )

    evaluator.add_ground_truth({
        "expected_opening_date": "January 7",
        "expected_opening_time_timezone": "8:00 a.m. MST",
        "expected_pickup_deadline": "By 10:00 a.m. on the permit start date"
    }, gt_type="expected_values")

    await build_verification_tree(evaluator, extracted)

    return evaluator.get_summary()