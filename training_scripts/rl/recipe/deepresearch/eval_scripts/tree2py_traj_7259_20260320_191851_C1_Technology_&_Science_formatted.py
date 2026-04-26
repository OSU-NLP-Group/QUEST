import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "verizon_outage_jan2026"
TASK_DESCRIPTION = (
    "In January 2026, Verizon experienced a major nationwide cellular service outage that affected hundreds of thousands of customers across the United States. "
    "Please provide the following information about this outage: "
    "(1) the specific date when the outage occurred, "
    "(2) the approximate duration of the outage in hours, and "
    "(3) the confirmed cause of the outage, including whether or not it was a cyberattack."
)

# Ground truth references for verification
EXPECTED_OUTAGE_DATE = "January 14, 2026"
EXPECTED_DURATION_HOURS_APPROX = "10"
EXPECTED_TIME_WINDOW = "around noon ET to approximately 10:20 PM ET"
EXPECTED_CAUSE_DESC = "a software issue or server failure in New Jersey (not a cyberattack)"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OutageExtraction(BaseModel):
    outage_date: Optional[str] = None
    outage_duration_hours: Optional[str] = None
    outage_start_time_et: Optional[str] = None
    outage_end_time_et: Optional[str] = None
    cause: Optional[str] = None
    cyberattack_statement: Optional[str] = None

    date_sources: List[str] = Field(default_factory=list)
    duration_sources: List[str] = Field(default_factory=list)
    cause_sources: List[str] = Field(default_factory=list)
    general_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outage_details() -> str:
    return (
        "Extract the key details about the January 2026 Verizon outage as stated in the provided answer. "
        "Return a JSON object with the following fields:\n"
        "1) outage_date: The specific date of the outage exactly as stated in the answer (e.g., 'January 14, 2026', 'Jan 14, 2026', or '1/14/26'). If not provided, set to null.\n"
        "2) outage_duration_hours: The approximate duration of the outage in hours as described in the answer (e.g., '10 hours', 'about 10', 'nearly 10 hours'). If not provided, set to null.\n"
        "3) outage_start_time_et: If the answer provides an approximate start time in Eastern Time (ET), extract it as a string (e.g., 'noon ET', '12:00 PM ET'); otherwise null.\n"
        "4) outage_end_time_et: If the answer provides an approximate end time in Eastern Time (ET), extract it as a string (e.g., '10:20 PM ET'); otherwise null.\n"
        "5) cause: The stated cause summary as written in the answer (e.g., 'software issue in New Jersey', 'server failure in New Jersey'). If not provided, set to null.\n"
        "6) cyberattack_statement: The exact phrase in the answer that confirms whether it was or was not a cyberattack (e.g., 'not a cyberattack'); if not present, set to null.\n"
        "7) date_sources: All URLs cited in the answer that support the outage date; if the answer only provides a general sources list, duplicate those here. If none, return an empty list.\n"
        "8) duration_sources: All URLs cited in the answer that support the duration/time window; if the answer only provides a general sources list, duplicate those here. If none, return an empty list.\n"
        "9) cause_sources: All URLs cited in the answer that support the cause and cyberattack status; if the answer only provides a general sources list, duplicate those here. If none, return an empty list.\n"
        "10) general_sources: If the answer includes a combined or general list of sources that apply to all facts, list those URLs here as well. Otherwise, return an empty list.\n"
        "Rules:\n"
        "- Extract only what is explicitly written in the answer; do not invent text.\n"
        "- URLs must be actual links present in the answer (plain URLs or markdown links). If no URL is provided, leave the list empty.\n"
        "- If only a single combined sources list is given in the answer, copy those URLs into each of date_sources, duration_sources, and cause_sources in addition to general_sources.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _pick_sources(primary: List[str], fallback: List[str]) -> List[str]:
    if primary and len(primary) > 0:
        return primary
    if fallback and len(fallback) > 0:
        return fallback
    return []


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_outage_date(evaluator: Evaluator, parent_node, data: OutageExtraction) -> None:
    node = evaluator.add_sequential(
        id="Outage_Date",
        desc="The answer correctly identifies that the outage occurred on January 14, 2026",
        parent=parent_node,
        critical=True
    )

    # 1) Presence of a stated date in the answer
    present = isinstance(data.outage_date, str) and data.outage_date.strip() != ""
    evaluator.add_custom_node(
        result=present,
        id="outage_date_present",
        desc="The answer provides a specific outage date",
        parent=node,
        critical=True
    )

    # 2) Match extracted date to expected truth (simple logical check)
    match_leaf = evaluator.add_leaf(
        id="outage_date_matches_expected",
        desc=f"The provided outage date matches {EXPECTED_OUTAGE_DATE}",
        parent=node,
        critical=True
    )
    extracted_date = data.outage_date or ""
    await evaluator.verify(
        claim=(
            f"The stated outage date '{extracted_date}' is equivalent to {EXPECTED_OUTAGE_DATE}. "
            "Treat formats like 'Jan 14, 2026', 'January 14, 2026', and '1/14/26' as equivalent."
        ),
        node=match_leaf,
        additional_instruction=(
            "Be lenient to common date formats and minor variations (e.g., ordinal suffixes like '14th'). "
            "Only judge whether the provided date effectively denotes January 14, 2026."
        )
    )

    # 3) Sources provided for the date
    date_sources = _pick_sources(data.date_sources, data.general_sources)
    evaluator.add_custom_node(
        result=len(date_sources) > 0,
        id="outage_date_sources_provided",
        desc="At least one source URL is provided to support the outage date",
        parent=node,
        critical=True
    )

    # 4) Sources support the date claim
    support_leaf = evaluator.add_leaf(
        id="outage_date_supported_by_sources",
        desc="Cited source(s) support that the Verizon outage occurred on January 14, 2026",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Verizon experienced a major nationwide cellular service outage on January 14, 2026.",
        node=support_leaf,
        sources=date_sources,
        additional_instruction=(
            "Focus on whether the page explicitly mentions the Verizon outage date as January 14, 2026. "
            "Minor wording differences are fine as long as the date is clearly supported."
        )
    )


async def verify_outage_duration(evaluator: Evaluator, parent_node, data: OutageExtraction) -> None:
    node = evaluator.add_sequential(
        id="Outage_Duration",
        desc="The answer correctly states that the outage lasted approximately 10 hours (from around noon ET to approximately 10:20 PM ET)",
        parent=parent_node,
        critical=True
    )

    # 1) Presence of a stated duration or time window
    duration_present = (
        (isinstance(data.outage_duration_hours, str) and data.outage_duration_hours.strip() != "")
        or (
            (isinstance(data.outage_start_time_et, str) and data.outage_start_time_et.strip() != "")
            and (isinstance(data.outage_end_time_et, str) and data.outage_end_time_et.strip() != "")
        )
    )
    evaluator.add_custom_node(
        result=bool(duration_present),
        id="outage_duration_present",
        desc="The answer provides an approximate duration in hours or a start/end time window",
        parent=node,
        critical=True
    )

    # 2) Match to ~10 hours (simple reasoning check)
    match_leaf = evaluator.add_leaf(
        id="outage_duration_matches_expected",
        desc=f"The stated outage duration corresponds to approximately {EXPECTED_DURATION_HOURS_APPROX} hours",
        parent=node,
        critical=True
    )

    if isinstance(data.outage_duration_hours, str) and data.outage_duration_hours.strip():
        match_claim = (
            f"The stated outage duration '{data.outage_duration_hours}' is approximately equal to "
            f"{EXPECTED_DURATION_HOURS_APPROX} hours."
        )
    else:
        start_str = data.outage_start_time_et or ""
        end_str = data.outage_end_time_et or ""
        match_claim = (
            f"The time span from '{start_str}' ET to '{end_str}' ET is approximately "
            f"{EXPECTED_DURATION_HOURS_APPROX} hours."
        )

    await evaluator.verify(
        claim=match_claim,
        node=match_leaf,
        additional_instruction=(
            "Allow for approximate phrasing like 'about', 'around', or 'nearly'. "
            "Treat small differences (e.g., 9.5–10.5 hours) as approximately 10 hours."
        )
    )

    # 3) Sources provided for duration/time window
    duration_sources = _pick_sources(data.duration_sources, data.general_sources)
    evaluator.add_custom_node(
        result=len(duration_sources) > 0,
        id="outage_duration_sources_provided",
        desc="At least one source URL is provided to support the outage duration/time window",
        parent=node,
        critical=True
    )

    # 4) Sources support ~10 hours and the window (noon ET -> ~10:20 PM ET)
    support_leaf = evaluator.add_leaf(
        id="outage_duration_supported_by_sources",
        desc="Cited source(s) support that the outage lasted approximately 10 hours (around noon ET to ~10:20 PM ET)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The January 2026 Verizon outage lasted approximately 10 hours, roughly from around noon ET "
            "to about 10:20 PM ET."
        ),
        node=support_leaf,
        sources=duration_sources,
        additional_instruction=(
            "Accept minor differences in reported times (e.g., 'shortly after noon', 'about 10 PM'). "
            "The key is that duration is about 10 hours, and coverage approximates a midday start and a late-evening end."
        )
    )


async def verify_outage_cause(evaluator: Evaluator, parent_node, data: OutageExtraction) -> None:
    node = evaluator.add_sequential(
        id="Outage_Cause",
        desc="The answer correctly identifies that the cause was a software issue or server failure in New Jersey, and confirms it was not a cyberattack",
        parent=parent_node,
        critical=True
    )

    # 1) Presence of a stated cause or explicit cyberattack statement
    cause_present = (
        (isinstance(data.cause, str) and data.cause.strip() != "")
        or (isinstance(data.cyberattack_statement, str) and data.cyberattack_statement.strip() != "")
    )
    evaluator.add_custom_node(
        result=bool(cause_present),
        id="outage_cause_present",
        desc="The answer provides a cause and/or explicitly states whether it was a cyberattack",
        parent=node,
        critical=True
    )

    # 2) Match to expected cause semantics (simple reasoning check)
    match_leaf = evaluator.add_leaf(
        id="outage_cause_matches_expected",
        desc="The stated cause aligns with a software issue/server failure in New Jersey and explicitly not a cyberattack",
        parent=node,
        critical=True
    )
    stated_cause = data.cause or ""
    cyber_stmt = data.cyberattack_statement or ""
    await evaluator.verify(
        claim=(
            f"The stated cause ('{stated_cause}' and '{cyber_stmt}') indicates a software issue or server failure "
            "in New Jersey and explicitly confirms it was not a cyberattack."
        ),
        node=match_leaf,
        additional_instruction=(
            "Allow equivalent phrasings such as 'software bug', 'software update problem', 'server crash', or "
            "'data center issue' as software/server failure. It must be associated with New Jersey and clearly state "
            "that it was not a cyberattack."
        )
    )

    # 3) Sources provided for cause/cyberattack status
    cause_sources = _pick_sources(data.cause_sources, data.general_sources)
    evaluator.add_custom_node(
        result=len(cause_sources) > 0,
        id="outage_cause_sources_provided",
        desc="At least one source URL is provided to support the cause and the non-cyberattack confirmation",
        parent=node,
        critical=True
    )

    # 4) Sources support the cause and non-cyberattack confirmation
    support_leaf = evaluator.add_leaf(
        id="outage_cause_supported_by_sources",
        desc="Cited source(s) support that the cause was a software/server issue in New Jersey and not a cyberattack",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The January 2026 Verizon outage was caused by a software issue or server failure in New Jersey and was not a cyberattack.",
        node=support_leaf,
        sources=cause_sources,
        additional_instruction=(
            "Look for explicit mention that the outage stemmed from a software/server issue in New Jersey and that "
            "it was not the result of a cyberattack. Allow wording variations that clearly convey the same facts."
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

    # Extract structured outage details from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_outage_details(),
        template_class=OutageExtraction,
        extraction_name="outage_extraction"
    )

    # Add ground truth info for transparency
    evaluator.add_ground_truth({
        "expected_date": EXPECTED_OUTAGE_DATE,
        "expected_duration_hours_approx": EXPECTED_DURATION_HOURS_APPROX,
        "expected_time_window": EXPECTED_TIME_WINDOW,
        "expected_cause_summary": EXPECTED_CAUSE_DESC
    }, gt_type="ground_truth")

    # Build top-level critical node for the Verizon outage info
    top = evaluator.add_parallel(
        id="Verizon_Outage_Information",
        desc="Verify the key details about the January 2026 Verizon outage",
        parent=root,
        critical=True
    )

    # Verify each component
    await verify_outage_date(evaluator, top, extracted)
    await verify_outage_duration(evaluator, top, extracted)
    await verify_outage_cause(evaluator, top, extracted)

    return evaluator.get_summary()