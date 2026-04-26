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
TASK_ID = "gaming_event_aug2024_cologne"
TASK_DESCRIPTION = (
    "What is the name of the gaming event that took place in August 2024 in Cologne, Germany, "
    "which attracted at least 300,000 visitors, was held at the Koelnmesse exhibition center, "
    "and spanned 5 consecutive days?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventInfo(BaseModel):
    """Structured information about the gaming event extracted from the answer."""
    event_name: Optional[str] = None
    attendance: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    venue: Optional[str] = None
    month_year: Optional[str] = None  # e.g., "August 2024"
    duration_days: Optional[str] = None  # e.g., "5 days", "Aug 21–25 (5 days)"
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_event_info() -> str:
    return """
    Extract the specific event details mentioned in the answer that correspond to a gaming event. 
    Return a JSON object with these fields, strictly reflecting what is stated in the answer:

    - event_name: The explicit name of the gaming event (e.g., "gamescom 2024", "Gamescom"), not just a description.
    - attendance: The reported visitor count or phrase (e.g., "320,000 visitors", "over 300,000").
    - city: The city where the event occurred (e.g., "Cologne", "Köln").
    - country: The country (e.g., "Germany", "Deutschland").
    - venue: The venue/exhibition center name (e.g., "Koelnmesse", "Cologne Exhibition Centre").
    - month_year: The month and year when the event took place (e.g., "August 2024").
    - duration_days: The duration phrasing (e.g., "5 days", "five consecutive days", or a date range implying 5 days).
    - sources: All URLs cited in the answer that are relevant to these facts (official website, news, Wikipedia, etc.).

    Rules:
    - Only extract information explicitly present in the answer text; do not infer or invent.
    - Keep the event_name concise and use the official branding if clearly stated (e.g., prefer "gamescom 2024" over a paraphrase).
    - For attendance, include qualifiers present (e.g., "over", "~", "approximately") if used.
    - For city, allow "Cologne" or "Köln" as written in the answer; for country, "Germany" or "Deutschland".
    - For venue, keep the string exactly as presented (e.g., "Koelnmesse").
    - For month_year, provide the explicit month and year stated; do not infer.
    - For duration_days, use the phrasing from the answer (e.g., "5 consecutive days", or a date range that implies 5 days).
    - For sources, include only actual URLs present in the answer (plain or markdown links). If absent, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def safe_event_name(info: EventInfo) -> str:
    """Fallback to a generic phrase if event name is missing."""
    return info.event_name.strip() if info.event_name else "the event"


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_event_tree(evaluator: Evaluator, extracted: EventInfo) -> None:
    """
    Build the verification tree according to the rubric and perform verifications.
    The parent node 'Gaming_Event_Identification' is critical with parallel aggregation.
    All child checks are critical leaf nodes.
    """
    # Create main critical parallel node
    main_node = evaluator.add_parallel(
        id="Gaming_Event_Identification",
        desc="Identifies the gaming event name that matches all specified criteria",
        parent=evaluator.root,
        critical=True
    )

    # 1) Event name provided (existence check as a custom leaf)
    name_provided = evaluator.add_custom_node(
        result=bool(extracted.event_name and extracted.event_name.strip()),
        id="Event_Name_Provided",
        desc="Response provides the name of the gaming event (not just a description)",
        parent=main_node,
        critical=True
    )

    # Prepare sources for subsequent verifications (can be empty)
    sources_list: List[str] = extracted.sources if extracted.sources else []

    # 2) Attendance: at least 300,000 visitors
    attendance_node = evaluator.add_leaf(
        id="Attendance_Scale",
        desc="Event attracted at least 300,000 visitors",
        parent=main_node,
        critical=True
    )

    attendance_claim = f"{safe_event_name(extracted)} attracted at least 300,000 visitors."
    await evaluator.verify(
        claim=attendance_claim,
        node=attendance_node,
        sources=sources_list,
        additional_instruction=(
            "Determine if the provided page(s) explicitly support that the event's attendance meets or exceeds 300,000. "
            "Phrases like 'over 300,000', '~320,000', 'approximately 320,000' should be accepted as >= 300,000. "
            "Focus on the event in question."
        ),
    )

    # 3) Location: Cologne, Germany
    location_node = evaluator.add_leaf(
        id="Location_City_Country",
        desc="Event was held in Cologne, Germany",
        parent=main_node,
        critical=True
    )

    location_claim = f"{safe_event_name(extracted)} was held in Cologne, Germany."
    await evaluator.verify(
        claim=location_claim,
        node=location_node,
        sources=sources_list,
        additional_instruction=(
            "Accept 'Cologne' or 'Köln' for the city and 'Germany' or 'Deutschland' for the country. "
            "Verify that the event location matches Cologne, Germany."
        ),
    )

    # 4) Venue: Koelnmesse exhibition center
    venue_node = evaluator.add_leaf(
        id="Venue_Koelnmesse",
        desc="Event was held at the Koelnmesse exhibition center",
        parent=main_node,
        critical=True
    )

    venue_claim = f"{safe_event_name(extracted)} was held at the Koelnmesse exhibition center."
    await evaluator.verify(
        claim=venue_claim,
        node=venue_node,
        sources=sources_list,
        additional_instruction=(
            "Allow reasonable variants such as 'Koelnmesse', 'Kölnmesse', or 'Cologne Exhibition Centre' "
            "if they clearly refer to the Koelnmesse venue complex."
        ),
    )

    # 5) Timing: August 2024
    timing_node = evaluator.add_leaf(
        id="Timing_August_2024",
        desc="Event took place in August 2024",
        parent=main_node,
        critical=True
    )

    timing_claim = f"{safe_event_name(extracted)} took place in August 2024."
    await evaluator.verify(
        claim=timing_claim,
        node=timing_node,
        sources=sources_list,
        additional_instruction=(
            "Confirm that the event dates fall within August 2024 (e.g., a date range entirely in August 2024). "
            "If the source shows specific dates (e.g., 21–25 August 2024), that qualifies."
        ),
    )

    # 6) Duration: 5 consecutive days
    duration_node = evaluator.add_leaf(
        id="Duration_5_Consecutive_Days",
        desc="Event spanned 5 consecutive days",
        parent=main_node,
        critical=True
    )

    duration_claim = f"{safe_event_name(extracted)} spanned five consecutive days."
    await evaluator.verify(
        claim=duration_claim,
        node=duration_node,
        sources=sources_list,
        additional_instruction=(
            "Confirm that the event's public/show days cover five consecutive days "
            "(e.g., a date range like 21–25 August implies 5 consecutive days). "
            "Minor phrasing variants like 'five-day event' should be accepted if clearly consecutive."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the August 2024 Cologne gaming event identification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Single main verification node with parallel children
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

    # Extract event details from the answer
    extracted_event = await evaluator.extract(
        prompt=prompt_extract_event_info(),
        template_class=EventInfo,
        extraction_name="event_info",
    )

    # Optionally record ground truth expectations (non-binding; for reference)
    evaluator.add_ground_truth({
        "expected_constraints": {
            "attendance_at_least": 300000,
            "city": "Cologne (Köln)",
            "country": "Germany (Deutschland)",
            "venue": "Koelnmesse",
            "month_year": "August 2024",
            "duration_days": "5 consecutive days"
        },
        "note": "Typical matching event is gamescom 2024, but evaluation must rely on the answer's cited sources."
    }, gt_type="expected_conditions")

    # Build tree and run verifications
    await build_and_verify_event_tree(evaluator, extracted_event)

    # Return structured result summary
    return evaluator.get_summary()