import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "total_lunar_eclipse_2026"
TASK_DESCRIPTION = """
In 2026, there will be only one total lunar eclipse visible from anywhere in the world. For an astronomy education program planning observation activities, what is the exact date of this eclipse, what time (in UTC) will it reach its peak (greatest eclipse), and approximately how long will the totality phase last?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EclipseExtraction(BaseModel):
    """
    Structured extraction of eclipse details from the agent's answer.
    All fields should be extracted exactly as presented in the answer.
    """
    event_type: Optional[str] = None  # e.g., "total lunar eclipse"
    date: Optional[str] = None        # e.g., "March 3, 2026", "3 March 2026"
    greatest_eclipse_utc: Optional[str] = None  # e.g., "11:33 UTC", "11:33:20 UT"
    totality_duration: Optional[str] = None     # e.g., "about 58 minutes", "≈58 min", "roughly one hour"
    visibility_statement: Optional[str] = None  # sentence/phrase about visibility regions
    uniqueness_statement: Optional[str] = None  # sentence/phrase asserting "only total lunar eclipse in 2026"
    next_tle_year: Optional[str] = None         # e.g., "2028", or phrase mentioning 2028
    source_urls: List[str] = Field(default_factory=list)  # all URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_eclipse_info() -> str:
    return """
    Extract the following items exactly as they appear in the answer text (do not invent):
    - event_type: The described event type (e.g., "total lunar eclipse", "partial lunar eclipse"). Prefer the most specific phrase used.
    - date: The calendar date the eclipse is said to occur on, as written (e.g., "March 3, 2026" or "3 March 2026").
    - greatest_eclipse_utc: The time of greatest eclipse as a UTC time string exactly as written (e.g., "11:33 UTC", "11:33:20 UT", "11:33 am UTC").
    - totality_duration: The duration of the totality phase as written (e.g., "about 58 minutes", "≈58 min", "roughly one hour").
    - visibility_statement: The sentence or short phrase describing where the eclipse is visible (e.g., "visible from North America..."). If multiple, pick the most relevant one that mentions broad regions.
    - uniqueness_statement: The sentence or short phrase indicating that this is the only total lunar eclipse in 2026 (if stated).
    - next_tle_year: If the answer states the next total lunar eclipse year after this event, extract that year or the phrase containing that year (e.g., "2028", "late 2028").
    - source_urls: Extract all URLs explicitly present in the answer (including markdown links). Return only valid URLs with protocol.

    If any item is not present, set it to null (or an empty list for source_urls).
    """


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: EclipseExtraction) -> None:
    """
    Build the verification tree according to the rubric.
    All checks are critical under a single critical parallel node.
    """
    root = evaluator.root
    assert root is not None, "Evaluator root must be initialized."

    # Top-level critical node (must keep children critical due to framework constraint)
    top_node = evaluator.add_parallel(
        id="Total_Lunar_Eclipse_2026_Response",
        desc="Evaluate whether the answer satisfies all stated constraints and provides the requested eclipse planning details.",
        parent=root,
        critical=True
    )

    # Helper: Safe string for display
    def ss(val: Optional[str]) -> str:
        return val if (val is not None and str(val).strip() != "") else "None"

    # 1) Event_Is_Total_Lunar_Eclipse
    node_event = evaluator.add_leaf(
        id="Event_Is_Total_Lunar_Eclipse",
        desc="Answer identifies the event as a total lunar eclipse (not partial/penumbral/solar).",
        parent=top_node,
        critical=True
    )
    claim_event = "The event identified in the answer is a total lunar eclipse (not partial, penumbral, or solar)."
    await evaluator.verify(
        claim=claim_event,
        node=node_event,
        additional_instruction=(
            f"Use the extracted event_type='{ss(extracted.event_type)}'. "
            "Judge Correct only if the phrasing clearly indicates a total lunar eclipse "
            "(e.g., 'total lunar eclipse', 'total eclipse of the Moon'). "
            "Treat letter case and minor wording variants leniently. If missing or indicates a different type, judge Incorrect."
        )
    )

    # 2) Uniqueness_In_2026
    node_unique = evaluator.add_leaf(
        id="Uniqueness_In_2026",
        desc="Answer indicates this is the only total lunar eclipse visible anywhere in the world in 2026.",
        parent=top_node,
        critical=True
    )
    claim_unique = "The answer explicitly states that 2026 has exactly one total lunar eclipse visible anywhere in the world."
    await evaluator.verify(
        claim=claim_unique,
        node=node_unique,
        additional_instruction=(
            f"Use the extracted uniqueness_statement='{ss(extracted.uniqueness_statement)}'. "
            "Accept phrasing like 'the only total lunar eclipse of 2026' or 'the year's lone total lunar eclipse'. "
            "If missing, ambiguous, or contradicts uniqueness, judge Incorrect."
        )
    )

    # 3) Eclipse_Date_March_3_2026
    node_date = evaluator.add_leaf(
        id="Eclipse_Date_March_3_2026",
        desc="Answer gives the eclipse date as March 3, 2026.",
        parent=top_node,
        critical=True
    )
    claim_date = "The answer gives the eclipse date as March 3, 2026."
    await evaluator.verify(
        claim=claim_date,
        node=node_date,
        additional_instruction=(
            f"Compare the extracted date='{ss(extracted.date)}' to March 3, 2026. "
            "Accept common formatting variants like '3 March 2026' or 'Mar 3, 2026'. "
            "If the extracted date is absent or clearly different, judge Incorrect."
        )
    )

    # 4) Visibility_From_North_America
    node_visibility = evaluator.add_leaf(
        id="Visibility_From_North_America",
        desc="Answer states the eclipse is visible from North America.",
        parent=top_node,
        critical=True
    )
    claim_visibility = "The answer states that the eclipse is visible from North America."
    await evaluator.verify(
        claim=claim_visibility,
        node=node_visibility,
        additional_instruction=(
            f"Use the extracted visibility_statement='{ss(extracted.visibility_statement)}'. "
            "Accept phrasing such as 'visible from North America', 'seen across much of North America', "
            "'visible in the United States/Canada/Mexico', or 'visible across the Americas' (if North America is clearly included). "
            "If North America is not mentioned or the statement excludes it, judge Incorrect."
        )
    )

    # 5) Greatest_Eclipse_Time_UTC_11_33
    node_peak = evaluator.add_leaf(
        id="Greatest_Eclipse_Time_UTC_11_33",
        desc="Answer gives greatest eclipse (peak) time as 11:33 UTC (allowing seconds, e.g., 11:33:xx UTC).",
        parent=top_node,
        critical=True
    )
    claim_peak = "The answer gives the time of greatest eclipse as 11:33 UTC."
    await evaluator.verify(
        claim=claim_peak,
        node=node_peak,
        additional_instruction=(
            f"Use the extracted greatest_eclipse_utc='{ss(extracted.greatest_eclipse_utc)}'. "
            "Accept exact '11:33 UTC', '11:33 UT', or '11:33:SS UTC/UT' (any seconds). "
            "Do NOT accept 11:32 or 11:34. The time must be explicitly in UTC/UT (minor formatting variants acceptable)."
        )
    )

    # 6) Totality_Duration_Approx_58_Min
    node_duration = evaluator.add_leaf(
        id="Totality_Duration_Approx_58_Min",
        desc="Answer gives totality duration as approximately 58 minutes (in minutes or an equivalent time unit).",
        parent=top_node,
        critical=True
    )
    claim_duration = "The answer states the totality lasts approximately 58 minutes."
    await evaluator.verify(
        claim=claim_duration,
        node=node_duration,
        additional_instruction=(
            f"Use the extracted totality_duration='{ss(extracted.totality_duration)}'. "
            "Consider 'approximately' as within ±5 minutes of 58 (i.e., 53–63 minutes). "
            "Accept reasonable equivalents such as 'about 58 minutes', '≈58 min', or 'about one hour' (clearly implying ~58–60 min). "
            "If the duration is missing or clearly outside ~58 minutes, judge Incorrect."
        )
    )

    # 7) Next_Total_Lunar_Eclipse_Not_Until_2028
    node_next = evaluator.add_leaf(
        id="Next_Total_Lunar_Eclipse_Not_Until_2028",
        desc="Answer states the next total lunar eclipse after this event does not occur until 2028 (i.e., no total lunar eclipse in 2027).",
        parent=top_node,
        critical=True
    )
    claim_next = "The answer states that the next total lunar eclipse after this event occurs in 2028 (i.e., not in 2027)."
    await evaluator.verify(
        claim=claim_next,
        node=node_next,
        additional_instruction=(
            f"Use the extracted next_tle_year='{ss(extracted.next_tle_year)}'. "
            "Accept statements that clearly indicate '2028' as the next total lunar eclipse year and/or that there is none in 2027. "
            "If missing or a different year is asserted, judge Incorrect."
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
    Evaluate an answer for the 2026 Total Lunar Eclipse planning task.
    Returns a structured summary with the verification tree and score.
    """
    # Initialize evaluator
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
        default_model=model,
    )

    # Extract structured eclipse info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_eclipse_info(),
        template_class=EclipseExtraction,
        extraction_name="eclipse_extraction"
    )

    # Add expected target info (for reference only; not used to gate)
    evaluator.add_ground_truth({
        "expected": {
            "event_type": "total lunar eclipse",
            "date": "March 3, 2026",
            "greatest_eclipse_time_utc": "11:33 UTC (seconds allowed)",
            "totality_duration_approx_minutes": 58,
            "visibility_includes": "North America",
            "uniqueness_in_2026": "only total lunar eclipse in 2026",
            "next_total_lunar_eclipse_year": "2028"
        }
    }, gt_type="ground_truth")

    # Build and run the verification tree
    await build_verification_tree(evaluator, extracted)

    # Return summary with results
    return evaluator.get_summary()