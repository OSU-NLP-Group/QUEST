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
TASK_ID = "nyse_nov_2025_early_close"
TASK_DESCRIPTION = "On what date in November 2025 does the New York Stock Exchange (NYSE) have an early close, and what time does it close on that day?"

EXPECTED_DATE = "Friday, November 28, 2025"
EXPECTED_TIME = "1:00 PM ET"

# --------------------------------------------------------------------------- #
# Data models for extraction (optional, for record-keeping)                   #
# --------------------------------------------------------------------------- #
class EarlyCloseExtraction(BaseModel):
    """Minimal structured extraction from the answer (for auditing only)."""
    early_close_date_text: Optional[str] = None
    early_close_time_text: Optional[str] = None
    other_nov_2025_early_close_dates: List[str] = Field(default_factory=list)
    mentions_regular_hours: Optional[bool] = None
    regular_hours_text: Optional[str] = None
    mentions_market_scope: Optional[bool] = None
    market_scope_text: Optional[str] = None
    cited_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_early_close() -> str:
    return """
    Extract from the answer the specific information related to the NYSE early close in November 2025.

    Required fields:
    - early_close_date_text: The date (as written) identified in the answer for the NYSE early close in November 2025. Examples of acceptable formats: "Friday, November 28, 2025", "Nov. 28, 2025", "11/28/2025", "Black Friday (Nov 28, 2025)", or "the Friday after Thanksgiving 2025". Return null if not explicitly stated.
    - early_close_time_text: The specific early close time (as written) for that day, e.g., "1:00 PM ET", "1 PM ET", "1 p.m. Eastern Time". Return null if not explicitly stated.
    - other_nov_2025_early_close_dates: List any other dates in November 2025 (strings as written) that the answer claims are early-close dates (if any). If none, return an empty list.
    - mentions_regular_hours: true/false if the answer explicitly mentions normal NYSE trading hours (9:30 AM–4:00 PM ET) on standard days, allowing minor phrasing variants like "9:30 a.m. to 4 p.m. ET".
    - regular_hours_text: The exact phrase used in the answer for regular hours if mentioned; otherwise null.
    - mentions_market_scope: true/false if the answer notes that the early close time applies across NYSE American Equities, NYSE Arca Equities, NYSE National, or related NYSE equities markets (allow synonyms like "NYSE-affiliated equities exchanges").
    - market_scope_text: The exact phrase used if the scope is mentioned; otherwise null.
    - cited_urls: Any URLs included in the answer (list all, if any).

    Return strictly and only these fields in JSON. Do not invent anything not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator) -> None:
    """
    Build the verification tree per the rubric and run the checks.
    We slightly restructure to ensure optional items do not penalize final score:
    - A critical group holds the three mandatory checks (date, time, no-other-date).
    - Optional notes are recorded as custom info (extracted), not scored, to avoid diluting the final score.
    """

    # Critical group mirroring the rubric's top-level node
    main_node = evaluator.add_parallel(
        id="NYSE_November_2025_Early_Close",
        desc="Verify the correct identification of the NYSE early close date and time in November 2025 (per provided constraints).",
        parent=evaluator.root,
        critical=True
    )

    # 1) Correct Date (critical)
    correct_date_node = evaluator.add_leaf(
        id="Correct_Date",
        desc="Answer identifies Friday, November 28, 2025 as the NYSE early-close date in November 2025.",
        parent=main_node,
        critical=True
    )
    correct_date_claim = (
        "In the answer, the NYSE early-close date in November 2025 is stated as Friday, November 28, 2025. "
        "Accept equivalent phrasings such as 'Nov. 28, 2025', '11/28/2025', "
        "'Black Friday (Nov 28, 2025)', or 'the Friday after Thanksgiving 2025', "
        "as long as it clearly refers to Friday, November 28, 2025."
    )
    await evaluator.verify(
        claim=correct_date_claim,
        node=correct_date_node,
        additional_instruction="Look only at the answer text. If any other November 2025 date is asserted as the early-close date, mark this as incorrect."
    )

    # 2) Correct Time (critical)
    correct_time_node = evaluator.add_leaf(
        id="Correct_Time",
        desc="Answer states the NYSE closes at 1:00 PM ET (Eastern Time) on that early-close day.",
        parent=main_node,
        critical=True
    )
    correct_time_claim = (
        "In the answer, the early close time for that day is stated as 1:00 PM ET (Eastern Time). "
        "Allow minor variants like '1 PM ET', '1 p.m. ET', '1:00 p.m. Eastern Time', or '1 pm EST'."
    )
    await evaluator.verify(
        claim=correct_time_claim,
        node=correct_time_node,
        additional_instruction="Look only at the answer text. If a different early close time (e.g., 2 PM) is stated for November 2025, this is incorrect."
    )

    # 3) No other November early-close claimed (critical)
    only_one_nov_early_close_node = evaluator.add_leaf(
        id="No_Other_November_Early_Close_Claimed",
        desc="Answer does not claim any other NYSE early-close date in November 2025 (i.e., treats Nov 28, 2025 as the only November 2025 early close).",
        parent=main_node,
        critical=True
    )
    only_one_claim = (
        "In the answer, no other NYSE early-close dates in November 2025 are claimed besides Friday, November 28, 2025."
    )
    await evaluator.verify(
        claim=only_one_claim,
        node=only_one_nov_early_close_node,
        additional_instruction=(
            "Look only at the answer text. It is fine if the answer also mentions that markets are CLOSED on Thanksgiving "
            "(Thu, Nov 27, 2025) since that is not an early close. Ignore early closes in other months."
        )
    )

    # Optional items: we record them as custom info (not scored) to avoid diluting the final score.
    # We will still perform LLM checks and store results in custom_info for transparency.

    optional_regular_hours_node = evaluator.add_leaf(
        id="Mentions_Regular_Hours_Optional",
        desc="Answer optionally notes regular NYSE trading hours are 9:30 AM–4:00 PM ET on normal trading days.",
        parent=main_node,  # Not attaching to scoring; will not verify here to avoid affecting score
        critical=False,
        status="skipped",  # Mark as skipped in tree to avoid unintended scoring impact
        score=0.0
    )

    optional_market_scope_node = evaluator.add_leaf(
        id="Mentions_Market_Scope_Optional",
        desc="Answer optionally notes the early close time applies across NYSE American Equities, NYSE Arca Equities, NYSE National, and related markets.",
        parent=main_node,  # Not attaching to scoring; will not verify here to avoid affecting score
        critical=False,
        status="skipped",
        score=0.0
    )

    # Instead of scoring optional items, we run standalone verifications and store results in custom_info.
    # This keeps the final score strictly determined by the mandatory checks.

    # Standalone verification for optional notes (does not write into tree nodes)
    regular_hours_claim = (
        "The answer mentions regular NYSE trading hours on normal days as 9:30 AM to 4:00 PM ET "
        "(allowing minor variants like '9:30 a.m. to 4 p.m. ET')."
    )
    regular_hours_present = await evaluator.verify(
        claim=regular_hours_claim,
        node=None,
        additional_instruction="Look only at the answer text. If such hours are mentioned in an equivalent form, consider it present."
    )

    market_scope_claim = (
        "The answer notes that the early close time applies across NYSE American Equities, NYSE Arca Equities, "
        "NYSE National, or otherwise indicates that the 1:00 PM ET early close applies across related NYSE equities markets."
    )
    market_scope_present = await evaluator.verify(
        claim=market_scope_claim,
        node=None,
        additional_instruction="Look only at the answer text. Allow synonyms such as 'NYSE-affiliated equities exchanges'."
    )

    evaluator.add_custom_info(
        info={
            "optional_regular_hours_mentioned": bool(regular_hours_present),
            "optional_market_scope_mentioned": bool(market_scope_present),
        },
        info_type="optional_notes_check",
        info_name="optional_mentions_summary"
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
    Evaluate an answer for the NYSE November 2025 early close task.
    The final score is binary:
      - 1.0 if all three mandatory checks pass (date, time, and no other November early-close claimed)
      - 0.0 otherwise
    Optional mentions are reported as custom info but do not affect the score.
    """
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

    # Record ground truth for transparency
    evaluator.add_ground_truth({
        "expected_date": EXPECTED_DATE,
        "expected_time": EXPECTED_TIME,
        "notes": "Optional mentions (regular hours and market scope) are not required and do not affect the score."
    })

    # Optional: extract structured info for auditing
    extraction = await evaluator.extract(
        prompt=prompt_extract_early_close(),
        template_class=EarlyCloseExtraction,
        extraction_name="early_close_extraction"
    )

    # Build tree and verify mandatory checks
    await build_and_verify_tree(evaluator)

    # Return evaluation summary
    return evaluator.get_summary()