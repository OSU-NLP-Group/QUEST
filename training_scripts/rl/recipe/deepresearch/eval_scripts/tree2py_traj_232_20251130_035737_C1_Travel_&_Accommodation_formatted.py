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
TASK_ID = "royal_caribbean_latest_arrival_mia"
TASK_DESCRIPTION = """
A traveler is boarding a Royal Caribbean cruise that departs from PortMiami at 4:00 PM. They are flying into Miami International Airport (MIA) on the same day of embarkation with checked luggage. Based on Royal Caribbean's boarding requirements and typical travel conditions between Miami International Airport and PortMiami, what is the latest time their flight should arrive at MIA to ensure they can realistically board the cruise?
"""

# Ground truth context to record (not used as hard constraints, but for transparency)
GROUND_TRUTH_CONTEXT = {
    "sailing_time": "4:00 PM",
    "boarding_cutoff_rule": "Guests must be checked in/onboard no later than 90 minutes before departure",
    "implied_cutoff_time": "2:30 PM",
    "typical_post_landing_processing_range_minutes": "30–45",
    "typical_mia_to_portmiami_travel_range_minutes": "15–20",
}


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class DurInfo(BaseModel):
    """A duration mention the answer uses for some allowance."""
    described: Optional[str] = None  # raw text like "about 40 minutes", "30–45 minutes"
    chosen_minutes: Optional[int] = None  # a single number if the answer picked one
    min_minutes: Optional[int] = None  # lower bound if a range was provided
    max_minutes: Optional[int] = None  # upper bound if a range was provided
    included_in_calculation: Optional[bool] = None  # whether the answer actually used this in its backward calc


class TimeWindow(BaseModel):
    """A time window if the answer gave a window instead of a single time."""
    start_time: Optional[str] = None  # e.g., "1:15 PM"
    end_time: Optional[str] = None    # e.g., "1:30 PM"


class ArrivalCalcExtraction(BaseModel):
    """Structured info extracted from the answer for this calculation task."""
    # Boarding cutoff application
    boarding_cutoff_time_str: Optional[str] = None  # e.g., "2:30 PM"
    boarding_cutoff_minutes_before_departure: Optional[int] = None  # e.g., 90

    # Time allowances used
    post_flight_processing: Optional[DurInfo] = None  # deplaning + bag claim + exit
    airport_to_port_travel: Optional[DurInfo] = None  # MIA -> PortMiami
    additional_buffer: Optional[DurInfo] = None       # any extra padding the answer chose

    # Stated latest arrival time
    latest_arrival_time_str: Optional[str] = None         # e.g., "1:30 PM"
    latest_arrival_window: Optional[TimeWindow] = None    # e.g., "1:15–1:30 PM"

    # Reasoning text if present
    backward_calc_text: Optional[str] = None              # explanation of the backward-time calculation


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_arrival_calc() -> str:
    return """
    Extract the specific timing and allowances the answer used to determine the latest feasible flight arrival time at MIA for a 4:00 PM Royal Caribbean departure from PortMiami.

    Return a JSON object with the following fields:

    1) boarding_cutoff_time_str: The exact boarding cutoff time used in the answer (e.g., "2:30 PM"), if stated. Otherwise null.
    2) boarding_cutoff_minutes_before_departure: The exact minutes-before-departure cutoff the answer used (e.g., 90), if explicitly stated. Otherwise null.

    3) post_flight_processing: An object describing the allowance for deplaning + baggage claim + exiting the airport.
       {
         "described": string or null (e.g., "about 40 minutes", "30–45 minutes"),
         "chosen_minutes": integer or null (single number used if the answer picked one, e.g., 40),
         "min_minutes": integer or null (if a range was given),
         "max_minutes": integer or null (if a range was given),
         "included_in_calculation": boolean or null (true if the answer explicitly used this allowance when working backward)
       }

    4) airport_to_port_travel: An object describing the time allowance for ground travel from MIA to PortMiami with the same fields as above.

    5) additional_buffer: An object describing any extra buffer the answer included beyond the two allowances above. Use the same fields; if no buffer was used, return all fields as null.

    6) latest_arrival_time_str: The explicit latest arrival time at MIA the answer states (e.g., "1:30 PM"). If not provided as a single time, return null.

    7) latest_arrival_window: If the answer presents a small window instead of a single time (e.g., "1:15–1:30 PM"), return:
       {
         "start_time": string or null,
         "end_time": string or null
       }
       If no window is provided, return null.

    8) backward_calc_text: If the answer explains its backward-time calculation in words or numbers, include that explanation text. Otherwise null.

    IMPORTANT:
    - Do not invent numbers. Only extract what the answer explicitly mentions.
    - Keep times exactly as written (e.g., "1:30 PM", "1 pm").
    - For durations, try to fill chosen_minutes if the answer clearly picks a single number. Otherwise, use min_minutes and max_minutes if a range is given. Always preserve 'described' with the exact text from the answer.
    """


# --------------------------------------------------------------------------- #
# Helper formatting functions                                                 #
# --------------------------------------------------------------------------- #
def format_duration_for_claim(d: Optional[DurInfo], fallback_label: str) -> str:
    """
    Produce a human-friendly phrase for the duration, using extracted info only.
    """
    if not d:
        return f"an allowance (unspecified) for {fallback_label}"
    if d.chosen_minutes is not None:
        return f"{d.chosen_minutes} minutes"
    if d.min_minutes is not None and d.max_minutes is not None:
        return f"{d.min_minutes}–{d.max_minutes} minutes"
    if d.described:
        return d.described
    return f"an allowance (unspecified) for {fallback_label}"


def format_latest_arrival_for_claim(extracted: ArrivalCalcExtraction) -> str:
    """
    Build a readable latest-arrival phrase from extraction.
    """
    if extracted.latest_arrival_time_str:
        return extracted.latest_arrival_time_str
    if extracted.latest_arrival_window and (extracted.latest_arrival_window.start_time or extracted.latest_arrival_window.end_time):
        s = extracted.latest_arrival_window.start_time or ""
        e = extracted.latest_arrival_window.end_time or ""
        if s and e:
            return f"{s}–{e}"
        return s or e
    return "an explicit latest arrival time"


def buffer_phrase(extracted: ArrivalCalcExtraction) -> str:
    """
    Describe buffer if clearly included in calculation.
    """
    buf = extracted.additional_buffer
    if not buf:
        return ""
    # Consider it included if either explicitly marked or if a duration is present
    included = buf.included_in_calculation or False
    if not included and (buf.chosen_minutes or buf.min_minutes or buf.max_minutes or (buf.described and "buffer" in buf.described.lower())):
        included = True
    if included:
        return f" and any additional buffer of {format_duration_for_claim(buf, 'buffer')}"
    return ""


# --------------------------------------------------------------------------- #
# Verification logic (build tree + LLM checks)                                #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: ArrivalCalcExtraction) -> None:
    """
    Construct the verification tree exactly according to the rubric and run verifications.
    """

    # Top-level (critical sequential) node as described by the rubric
    calc_root = evaluator.add_sequential(
        id="LatestArrivalTimeCalculation",
        desc="Determines the latest realistic flight arrival time at MIA that still allows boarding a 4:00 PM PortMiami Royal Caribbean departure with checked luggage, consistent with the given constraints.",
        parent=evaluator.root,
        critical=True
    )

    # 1) BoardingDeadlineApplied (critical leaf)
    leaf_boarding_cutoff = evaluator.add_leaf(
        id="BoardingDeadlineApplied",
        desc="Computes and uses the boarding cutoff implied by the constraints: guests must be checked in/onboard no later than 90 minutes before the 4:00 PM sailing time (i.e., establishes a 2:30 PM cutoff).",
        parent=calc_root,
        critical=True
    )

    cutoff_str = extracted.boarding_cutoff_time_str or "2:30 PM"
    cutoff_min_str = (
        f"{extracted.boarding_cutoff_minutes_before_departure} minutes"
        if extracted.boarding_cutoff_minutes_before_departure is not None else "90 minutes"
    )
    claim_cutoff = (
        f"The answer applies the correct boarding cutoff for a 4:00 PM sailing by using a 2:30 PM cutoff "
        f"(i.e., {cutoff_min_str} before departure), or it explicitly states '{cutoff_min_str} before 4:00 PM'."
    )
    await evaluator.verify(
        claim=claim_cutoff,
        node=leaf_boarding_cutoff,
        additional_instruction="Credit if the answer either states '2:30 PM' as the cutoff or explicitly mentions '90 minutes before a 4:00 PM sailing' (or equivalent phrasing)."
    )

    # 2) TypicalTimeAllowancesUsed (critical parallel) with two critical leaves
    node_typical = evaluator.add_parallel(
        id="TypicalTimeAllowancesUsed",
        desc="Uses time allowances consistent with the provided typical ranges for (1) post-landing airport processing with checked luggage and (2) ground transport from MIA to PortMiami.",
        parent=calc_root,
        critical=True
    )

    # 2a) PostFlightProcessingTimeWithinRange
    leaf_postflight = evaluator.add_leaf(
        id="PostFlightProcessingTimeWithinRange",
        desc="Includes a post-landing processing allowance (deplaning + baggage claim + exiting airport) that is within the stated typical 30–45 minutes.",
        parent=node_typical,
        critical=True
    )
    post_desc = format_duration_for_claim(extracted.post_flight_processing, "post-landing processing")
    claim_post = (
        f"The answer includes a post-landing processing allowance of {post_desc}, and this allowance is within a typical 30–45 minute range."
    )
    await evaluator.verify(
        claim=claim_post,
        node=leaf_postflight,
        additional_instruction="Look for a deplaning/baggage-claim/exiting-airport allowance. Accept any value or narrow range that falls within 30–45 minutes; wording like 'about 40 minutes' or '30–45 min' is acceptable."
    )

    # 2b) AirportToPortTravelTimeWithinRange
    leaf_travel = evaluator.add_leaf(
        id="AirportToPortTravelTimeWithinRange",
        desc="Includes an MIA-to-PortMiami travel-time allowance that is within the stated typical 15–20 minutes.",
        parent=node_typical,
        critical=True
    )
    travel_desc = format_duration_for_claim(extracted.airport_to_port_travel, "MIA-to-PortMiami travel")
    claim_travel = (
        f"The answer includes a ground-travel allowance from MIA to PortMiami of {travel_desc}, which is within a typical 15–20 minute range."
    )
    await evaluator.verify(
        claim=claim_travel,
        node=leaf_travel,
        additional_instruction="This refers to typical non-rush travel time from MIA to PortMiami. Accept phrasing like 'about 15–20 minutes' or a specific value within that range."
    )

    # 3) LatestArrivalTimeProvidedAndCalculated (critical parallel)
    node_latest = evaluator.add_parallel(
        id="LatestArrivalTimeProvidedAndCalculated",
        desc="Provides a latest arrival time at MIA and supports it with a correct backward-time calculation from the 2:30 PM cutoff using the stated allowances (and any optional additional buffer, if included).",
        parent=calc_root,
        critical=True
    )

    # 3a) LatestArrivalTimeStated (critical leaf)
    leaf_latest_stated = evaluator.add_leaf(
        id="LatestArrivalTimeStated",
        desc="States an explicit latest arrival time at MIA (a concrete clock time or a narrow time window).",
        parent=node_latest,
        critical=True
    )
    latest_text = format_latest_arrival_for_claim(extracted)
    claim_latest_stated = (
        "The answer states an explicit latest MIA flight arrival time as either a specific clock time "
        "or a narrow time window (e.g., 'by 1:30 PM' or '1:15–1:30 PM')."
    )
    await evaluator.verify(
        claim=claim_latest_stated,
        node=leaf_latest_stated,
        additional_instruction="General advice like 'arrive earlier' does not count. There must be a concrete time or a narrow time window explicitly stated."
    )

    # 3b) BackwardCalculationConsistent (critical leaf)
    leaf_consistent = evaluator.add_leaf(
        id="BackwardCalculationConsistent",
        desc="The stated latest arrival time is arithmetically consistent with working backward from the 2:30 PM cutoff by subtracting the stated airport-processing time and travel time (and subtracting any additional buffer only if the answer explicitly includes one).",
        parent=node_latest,
        critical=True
    )

    # Build a claim that references only what the answer states, letting the judge check arithmetic
    post_phrase = format_duration_for_claim(extracted.post_flight_processing, "post-landing processing")
    travel_phrase = format_duration_for_claim(extracted.airport_to_port_travel, "MIA-to-PortMiami travel")
    buf_phrase = buffer_phrase(extracted)
    arrival_phrase = format_latest_arrival_for_claim(extracted)

    claim_consistent = (
        f"Working backward from the 2:30 PM boarding cutoff by subtracting {post_phrase} for airport processing and "
        f"{travel_phrase} for MIA-to-PortMiami travel{buf_phrase}, the result is consistent with the answer’s stated latest MIA arrival time of {arrival_phrase}."
    )
    await evaluator.verify(
        claim=claim_consistent,
        node=leaf_consistent,
        additional_instruction=(
            "Use the numbers the answer itself provides. Allow reasonable rounding (±5 minutes). "
            "If a small window is given, the latest bound (the maximum time in the window) should still leave enough time to meet the 2:30 PM cutoff."
        )
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
    Evaluate an answer for the latest-arrival-time MIA -> PortMiami boarding feasibility task.
    Returns a standardized summary dictionary produced by the evaluator.
    """
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
        default_model=model,
    )

    # Record expected context as "ground truth" info (for transparency in the report)
    evaluator.add_ground_truth(GROUND_TRUTH_CONTEXT, gt_type="reference_context")

    # Extract structured fields from the provided answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_arrival_calc(),
        template_class=ArrivalCalcExtraction,
        extraction_name="arrival_calculation_extraction"
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extracted)

    # Return the framework's standardized summary
    return evaluator.get_summary()