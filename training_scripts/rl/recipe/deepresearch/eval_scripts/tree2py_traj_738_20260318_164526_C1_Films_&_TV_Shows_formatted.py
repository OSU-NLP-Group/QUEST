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
TASK_ID = "the_pitt_s1_eval"
TASK_DESCRIPTION = (
    "The Pitt is a medical drama series that premiered on Max. When did Season 1 premiere, "
    "how many episodes were in the first season, and what type of release schedule did it follow (weekly or all-at-once)?"
)

EXPECTED_PREMIERE_DATE = "January 9, 2025"
EXPECTED_EPISODE_COUNT = "15"
EXPECTED_WEEKLY_DAY = "Thursday"
EXPECTED_RELEASE_TIME_ET = "9 PM ET"
EXPECTED_RELEASE_TIME_PT = "6 PM PT"
EXPECTED_FINALE_DATE = "April 10, 2025"


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class AnswerSignals(BaseModel):
    series_name: Optional[str] = None
    genre_or_descriptor: Optional[str] = None
    platform: Optional[str] = None

    season1_premiere_date_text: Optional[str] = None
    season1_episode_count_text: Optional[str] = None

    release_schedule_type_text: Optional[str] = None  # e.g., "weekly", "all at once", "weekly release"
    weekly_day_text: Optional[str] = None             # e.g., "Thursday"
    release_time_text: Optional[str] = None           # e.g., "9 PM ET / 6 PM PT", allow variants
    premiere_drop_two_text: Optional[str] = None      # e.g., "first two episodes together", "two-episode premiere"
    season_finale_date_text: Optional[str] = None     # e.g., "April 10, 2025"

    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_answer_signals() -> str:
    return """
    Extract the exact details that the answer text provides regarding the series and Season 1 specifics for "The Pitt".
    Do not invent information; only extract what the answer explicitly states.

    Return a JSON object with the following fields:
    - series_name: The series title as written (e.g., "The Pitt")
    - genre_or_descriptor: The descriptor/genre text used for the series (e.g., "medical drama")
    - platform: The streaming platform named in the answer (e.g., "Max", "HBO Max")

    - season1_premiere_date_text: The Season 1 premiere date string as written (e.g., "January 9, 2025" or "Jan 9, 2025")
    - season1_episode_count_text: The episode count text as written for Season 1 (e.g., "15", "15 episodes")

    - release_schedule_type_text: The release model as stated (e.g., "weekly", "released weekly", "all at once", "binge drop")
    - weekly_day_text: The weekday for new episodes if stated (e.g., "Thursday", "Thursdays")
    - release_time_text: The release time if stated, ideally including time zones (e.g., "9 PM ET / 6 PM PT")
    - premiere_drop_two_text: Text that indicates whether the premiere dropped two episodes together (e.g., "two-episode premiere", "first two episodes together")
    - season_finale_date_text: The season finale date as written (e.g., "April 10, 2025")

    - sources: An array of all explicit URLs present anywhere in the answer. Include full URLs; deduplicate.

    If any field is not present in the answer, set it to null (or an empty array for 'sources').
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def add_series_identification_check(
    evaluator: Evaluator,
    parent,
) -> None:
    node = evaluator.add_leaf(
        id="Series_Identification",
        desc="Answer identifies the series as The Pitt and indicates it is a medical drama that premiered on the Max streaming platform.",
        parent=parent,
        critical=True,
    )
    claim = (
        "The answer explicitly identifies the show as 'The Pitt', describes it as a medical drama, "
        "and states that it premiered on the Max streaming platform (Max, formerly HBO Max)."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction=(
            "Judge only based on the provided answer text. Accept reasonable phrasing equivalents such as "
            "'debuted' for 'premiered', and 'Max' possibly referenced as 'HBO Max/Max'. "
            "All three elements must appear in the answer: the title 'The Pitt', 'medical drama', and 'Max'. "
            "If any one is missing or a different platform is claimed, mark incorrect."
        ),
    )


async def add_premiere_date_check(
    evaluator: Evaluator,
    parent,
) -> None:
    node = evaluator.add_leaf(
        id="Season1_Premiere_Date",
        desc=f"Answer states that Season 1 premiered on {EXPECTED_PREMIERE_DATE}.",
        parent=parent,
        critical=True,
    )
    claim = f"The answer explicitly states that Season 1 premiered on {EXPECTED_PREMIERE_DATE}."
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction=(
            "Judge only from the answer text. Allow minor formatting variants like 'Jan 9, 2025'. "
            "If the answer lists a different date or is ambiguous, mark incorrect."
        ),
    )


async def add_episode_count_check(
    evaluator: Evaluator,
    parent,
) -> None:
    node = evaluator.add_leaf(
        id="Season1_Episode_Count",
        desc=f"Answer states that Season 1 has exactly {EXPECTED_EPISODE_COUNT} episodes.",
        parent=parent,
        critical=True,
    )
    claim = f"The answer explicitly states that Season 1 has {EXPECTED_EPISODE_COUNT} episodes."
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction=(
            "Judge only from the answer text. Accept equivalent phrasing like '15 episodes'. "
            "If the answer mentions a different number or is unclear, mark incorrect."
        ),
    )


async def add_release_constraints_checks(
    evaluator: Evaluator,
    parent,
) -> None:
    group = evaluator.add_parallel(
        id="Release_Schedule_Constraints",
        desc="Answer correctly describes the release schedule constraints for Season 1.",
        parent=parent,
        critical=True,
    )

    # Release schedule type: weekly (not all-at-once)
    type_node = evaluator.add_leaf(
        id="Release_Schedule_Type",
        desc="Answer characterizes the release model as weekly (not all-at-once/binge).",
        parent=group,
        critical=True,
    )
    claim_type = (
        "The answer explicitly characterizes the Season 1 release model as weekly (e.g., 'new episodes every week'), "
        "and does not say it was released all at once."
    )
    await evaluator.verify(
        claim=claim_type,
        node=type_node,
        additional_instruction=(
            "Judge only from the answer text. Accept phrasing like 'weekly releases', 'airs weekly', "
            "'new episodes each week'. If the answer suggests an all-at-once drop or binge model, mark incorrect."
        ),
    )

    # Weekly day: Thursdays
    day_node = evaluator.add_leaf(
        id="Weekly_Day",
        desc="Answer states that new episodes released weekly on Thursdays.",
        parent=group,
        critical=True,
    )
    claim_day = "The answer states that new Season 1 episodes released on Thursdays."
    await evaluator.verify(
        claim=claim_day,
        node=day_node,
        additional_instruction=(
            "Judge only from the answer text. Accept variants like 'every Thursday', 'on Thursday evenings'. "
            "If a different weekday is given or not stated, mark incorrect."
        ),
    )

    # Release time: 9 PM ET / 6 PM PT
    time_node = evaluator.add_leaf(
        id="Release_Time",
        desc="Answer states that new episodes premiered at 9 PM ET / 6 PM PT.",
        parent=group,
        critical=True,
    )
    claim_time = "The answer states that new episodes premiered at 9 PM ET (which corresponds to 6 PM PT)."
    await evaluator.verify(
        claim=claim_time,
        node=time_node,
        additional_instruction=(
            "Judge only from the answer text. Accept reasonable formatting variants such as '9 p.m. ET (6 p.m. PT)', "
            "'9:00 ET / 6:00 PT', etc. Both ET and PT times must be present to pass."
        ),
    )

    # Premiere dropped two episodes together on Jan 9, 2025
    two_ep_node = evaluator.add_leaf(
        id="Premiere_Drop_Two_Episodes",
        desc="Answer states that the series premiere released the first two episodes together on January 9, 2025.",
        parent=group,
        critical=True,
    )
    claim_two = (
        f"The answer states that the premiere released the first two episodes together on {EXPECTED_PREMIERE_DATE}."
    )
    await evaluator.verify(
        claim=claim_two,
        node=two_ep_node,
        additional_instruction=(
            "Judge only from the answer text. Accept phrasing like 'two-episode premiere', "
            "'first two episodes dropped together'. If the answer does not clearly indicate two episodes at premiere, mark incorrect."
        ),
    )

    # Season finale date: April 10, 2025
    finale_node = evaluator.add_leaf(
        id="Season_Finale_Date",
        desc=f"Answer states that the season finale aired on {EXPECTED_FINALE_DATE}.",
        parent=group,
        critical=True,
    )
    claim_finale = f"The answer explicitly states that the Season 1 finale aired on {EXPECTED_FINALE_DATE}."
    await evaluator.verify(
        claim=claim_finale,
        node=finale_node,
        additional_instruction=(
            "Judge only from the answer text. Allow minor formatting variants like 'Apr 10, 2025'. "
            "If the answer lists a different date or is ambiguous, mark incorrect."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
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
    # Initialize evaluator with a root node (always non-critical by framework design)
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

    # Optional: extract signals (for logging/analysis)
    extracted = await evaluator.extract(
        prompt=prompt_extract_answer_signals(),
        template_class=AnswerSignals,
        extraction_name="answer_signals",
    )

    # Build a critical overall node to mirror the rubric root (all must pass)
    overall = evaluator.add_parallel(
        id="Root",
        desc="Evaluate whether the answer satisfies all stated constraints and provides the requested Season 1 premiere date, episode count, and release schedule type for The Pitt on Max.",
        parent=root,
        critical=True,
    )

    # Add checks according to rubric
    await add_series_identification_check(evaluator, overall)
    await add_premiere_date_check(evaluator, overall)
    await add_episode_count_check(evaluator, overall)
    await add_release_constraints_checks(evaluator, overall)

    # Return evaluation summary
    return evaluator.get_summary()