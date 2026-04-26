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
TASK_ID = "thanksgiving_2025_tv_events"
TASK_DESCRIPTION = (
    "For the Thanksgiving Day 2025 live television events in the United States: "
    "(1) Identify the U.S. state where the handler of the National Dog Show Best in Show winner is based, "
    "along with the dog's breed and the prize amount awarded to the winner; "
    "(2) Provide the broadcast start time (in Eastern Time) and the television network for the National Dog Show; "
    "and (3) Identify how many Broadway shows performed at the Macy's Thanksgiving Day Parade and what time (in Eastern Time) the parade broadcast began. "
    "For each piece of information, provide supporting reference URLs."
)
TARGET_YEAR = 2025


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DogShowWinnerDetails(BaseModel):
    handler_base_state: Optional[str] = None
    handler_state_urls: List[str] = Field(default_factory=list)

    winner_breed: Optional[str] = None
    breed_urls: List[str] = Field(default_factory=list)

    prize_amount: Optional[str] = None
    prize_urls: List[str] = Field(default_factory=list)


class DogShowBroadcastInfo(BaseModel):
    start_time_et: Optional[str] = None
    start_time_urls: List[str] = Field(default_factory=list)

    network: Optional[str] = None
    network_urls: List[str] = Field(default_factory=list)


class MacysParadeInfo(BaseModel):
    broadway_show_count: Optional[str] = None
    broadway_urls: List[str] = Field(default_factory=list)

    broadcast_start_time_et: Optional[str] = None
    parade_start_urls: List[str] = Field(default_factory=list)


class ThanksgivingTV2025Extraction(BaseModel):
    dog_show_winner: DogShowWinnerDetails = Field(default_factory=DogShowWinnerDetails)
    dog_show_broadcast: DogShowBroadcastInfo = Field(default_factory=DogShowBroadcastInfo)
    macys_parade: MacysParadeInfo = Field(default_factory=MacysParadeInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_thanksgiving_2025() -> str:
    return f"""
Extract the specific facts requested for Thanksgiving Day {TARGET_YEAR} live television events in the United States, strictly from the provided answer text. For each required fact, also extract the reference URL(s) explicitly cited in the answer to support that fact. Do not fabricate or infer any URLs; only extract URLs that actually appear in the answer.

Return a single JSON object with this structure:

- dog_show_winner:
  - handler_base_state: string | null
  - handler_state_urls: string[] (URLs cited that support the handler's base state)
  - winner_breed: string | null
  - breed_urls: string[] (URLs cited that support the breed)
  - prize_amount: string | null  (e.g., "$20,000" or "no cash prize")
  - prize_urls: string[] (URLs cited that support the prize amount)

- dog_show_broadcast:
  - start_time_et: string | null  (as written in the answer, e.g., "12:00 PM ET", "Noon ET", "12 p.m. ET")
  - start_time_urls: string[] (URLs cited that support the National Dog Show broadcast start time)
  - network: string | null (e.g., "NBC")
  - network_urls: string[] (URLs cited that support the National Dog Show network)

- macys_parade:
  - broadway_show_count: string | null (allow numbers as numerals or words, e.g., "9" or "nine")
  - broadway_urls: string[] (URLs cited that support how many Broadway shows performed)
  - broadcast_start_time_et: string | null (e.g., "8:30 AM ET")
  - parade_start_urls: string[] (URLs cited that support the parade broadcast start time)

Rules:
- Only extract information explicitly present in the answer. If a value is missing, put null.
- For URL arrays, include every unique URL that the answer provides for that specific fact. Accept plain URLs or markdown links; extract the actual URL targets.
- Do not include URLs that are unrelated to the specific fact they purportedly support.
- Do not deduplicate across different facts; each fact should have its own URL list.
- Do not try to fix broken or partial URLs; if not valid, omit them.
- Keep all values as strings exactly as written in the answer (do not normalize currencies or times).
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u.strip() for u in (urls or []) if isinstance(u, str) and u.strip()]


def _has_value_and_citation(value: Optional[str], urls: Optional[List[str]]) -> bool:
    return bool(value and value.strip()) and len(_non_empty_urls(urls)) > 0


def _common_year_instruction() -> str:
    return (
        f"Focus strictly on Thanksgiving Day {TARGET_YEAR} in the United States. "
        f"The cited page must be about the {TARGET_YEAR} event or an official page explicitly covering the {TARGET_YEAR} details. "
        "If a page obviously refers to a different year or is generic without confirming the specific requested fact for 2025, "
        "treat it as not supported."
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _add_citation_and_support(
    evaluator: Evaluator,
    *,
    parent_node,
    group_id: str,
    group_desc: str,
    value: Optional[str],
    urls: Optional[List[str]],
    support_leaf_desc: str,
    claim_text: str,
    add_ins: str,
) -> None:
    """
    Create a critical group node with:
    - a critical custom node checking that value and at least one URL are provided
    - a critical verification leaf that checks the claim is supported by provided URLs
    """
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent_node,
        critical=True,
    )

    # Existence of both the value and its citations
    has_citation = _has_value_and_citation(value, urls)
    evaluator.add_custom_node(
        result=has_citation,
        id=f"{group_id}_citation_present",
        desc="The answer provides the required value and at least one supporting reference URL",
        parent=group_node,
        critical=True,
    )

    # Support verification (will be effectively gated by the sibling citation-present node if it already failed)
    support_leaf = evaluator.add_leaf(
        id=f"{group_id}_supported",
        desc=support_leaf_desc,
        parent=group_node,
        critical=True,
    )
    await evaluator.verify(
        claim=claim_text,
        node=support_leaf,
        sources=_non_empty_urls(urls),
        additional_instruction=add_ins,
    )


async def verify_dog_show_winner_details(
    evaluator: Evaluator,
    parent_node,
    winner: DogShowWinnerDetails,
) -> None:
    section_node = evaluator.add_parallel(
        id="national_dog_show_winner_details",
        desc="National Dog Show Best in Show winner-related details requested (with citations)",
        parent=parent_node,
        critical=True,
    )

    # Handler base state
    handler_claim = (
        f"The handler of the National Dog Show {TARGET_YEAR} Best in Show winner is based in {winner.handler_base_state}."
        if (winner.handler_base_state and winner.handler_base_state.strip())
        else "The handler of the National Dog Show 2025 Best in Show winner is based in <unspecified>."
    )
    handler_ins = (
        _common_year_instruction()
        + " Verify the handler's base location/state (resides in, based in, from) for the Best in Show winner. "
          "Ensure the cited page identifies the handler for the Best in Show winner and their state."
    )
    await _add_citation_and_support(
        evaluator,
        parent_node=section_node,
        group_id="handler_base_state_with_citation",
        group_desc="Identifies the U.S. state where the handler of the Best in Show winner is based, with supporting reference URL(s)",
        value=winner.handler_base_state,
        urls=winner.handler_state_urls,
        support_leaf_desc="Handler base state is supported by the cited source(s)",
        claim_text=handler_claim,
        add_ins=handler_ins,
    )

    # Winner breed
    breed_claim = (
        f"The Best in Show winner of the National Dog Show {TARGET_YEAR} is a {winner.winner_breed}."
        if (winner.winner_breed and winner.winner_breed.strip())
        else "The Best in Show winner of the National Dog Show 2025 is a <unspecified breed>."
    )
    breed_ins = (
        _common_year_instruction()
        + " Verify the dog's breed specifically for the National Dog Show Best in Show winner."
    )
    await _add_citation_and_support(
        evaluator,
        parent_node=section_node,
        group_id="winner_breed_with_citation",
        group_desc="Identifies the Best in Show winner's breed, with supporting reference URL(s)",
        value=winner.winner_breed,
        urls=winner.breed_urls,
        support_leaf_desc="Winner's breed is supported by the cited source(s)",
        claim_text=breed_claim,
        add_ins=breed_ins,
    )

    # Prize amount
    prize_claim = (
        f"The prize amount awarded to the National Dog Show {TARGET_YEAR} Best in Show winner was {winner.prize_amount}."
        if (winner.prize_amount and winner.prize_amount.strip())
        else "The prize amount awarded to the National Dog Show 2025 Best in Show winner was <unspecified>."
    )
    prize_ins = (
        _common_year_instruction()
        + " Verify the monetary award (if any) granted to the Best in Show winner. If a page states that there is no cash prize, "
          "that should only be accepted if explicitly stated for the 2025 event."
    )
    await _add_citation_and_support(
        evaluator,
        parent_node=section_node,
        group_id="best_in_show_prize_amount_with_citation",
        group_desc="Identifies the prize amount awarded to the Best in Show winner, with supporting reference URL(s)",
        value=winner.prize_amount,
        urls=winner.prize_urls,
        support_leaf_desc="Best in Show prize amount is supported by the cited source(s)",
        claim_text=prize_claim,
        add_ins=prize_ins,
    )


async def verify_dog_show_broadcast_info(
    evaluator: Evaluator,
    parent_node,
    info: DogShowBroadcastInfo,
) -> None:
    section_node = evaluator.add_parallel(
        id="national_dog_show_broadcast_info",
        desc="National Dog Show broadcast details requested (with citations)",
        parent=parent_node,
        critical=True,
    )

    # Broadcast start time (ET)
    start_time_claim = (
        f"The National Dog Show {TARGET_YEAR} broadcast started at {info.start_time_et} Eastern Time."
        if (info.start_time_et and info.start_time_et.strip())
        else "The National Dog Show 2025 broadcast started at <unspecified time> Eastern Time."
    )
    start_time_ins = (
        _common_year_instruction()
        + " Verify the broadcast start time specifically in Eastern Time (accept ET, EST, or EDT as equivalent). "
          "If the source lists multiple time zones, confirm the Eastern Time value."
    )
    await _add_citation_and_support(
        evaluator,
        parent_node=section_node,
        group_id="broadcast_start_time_et_with_citation",
        group_desc="Provides the broadcast start time in Eastern Time for the National Dog Show, with supporting reference URL(s)",
        value=info.start_time_et,
        urls=info.start_time_urls,
        support_leaf_desc="National Dog Show broadcast start time (ET) is supported by the cited source(s)",
        claim_text=start_time_claim,
        add_ins=start_time_ins,
    )

    # Broadcast network
    network_claim = (
        f"The National Dog Show {TARGET_YEAR} was broadcast on {info.network}."
        if (info.network and info.network.strip())
        else "The National Dog Show 2025 was broadcast on <unspecified network>."
    )
    network_ins = (
        _common_year_instruction()
        + " Verify the television network that carried the National Dog Show in the U.S. "
          "If multiple networks or platforms are mentioned, confirm the primary U.S. linear TV broadcaster."
    )
    await _add_citation_and_support(
        evaluator,
        parent_node=section_node,
        group_id="broadcast_network_with_citation",
        group_desc="Provides the television network for the National Dog Show, with supporting reference URL(s)",
        value=info.network,
        urls=info.network_urls,
        support_leaf_desc="National Dog Show broadcast network is supported by the cited source(s)",
        claim_text=network_claim,
        add_ins=network_ins,
    )


async def verify_macys_parade_info(
    evaluator: Evaluator,
    parent_node,
    parade: MacysParadeInfo,
) -> None:
    section_node = evaluator.add_parallel(
        id="macys_parade_info",
        desc="Macy's Thanksgiving Day Parade Broadway-show count and broadcast start time requested (with citations)",
        parent=parent_node,
        critical=True,
    )

    # Broadway show count
    count_claim = (
        f"There were {parade.broadway_show_count} Broadway shows performing at the Macy's Thanksgiving Day Parade {TARGET_YEAR}."
        if (parade.broadway_show_count and parade.broadway_show_count.strip())
        else "There were <unspecified> Broadway shows performing at the Macy's Thanksgiving Day Parade 2025."
    )
    count_ins = (
        _common_year_instruction()
        + " Verify how many Broadway shows performed in the parade. "
          "Accept reasonable equivalence between numerals and spelled-out numbers (e.g., 9 vs. nine). "
          "If a page lists the shows individually, you may infer the count by enumeration only if the page clearly indicates these are the {TARGET_YEAR} performers."
    )
    await _add_citation_and_support(
        evaluator,
        parent_node=section_node,
        group_id="broadway_show_count_with_citation",
        group_desc="Identifies how many Broadway shows performed at the Macy's Thanksgiving Day Parade, with supporting reference URL(s)",
        value=parade.broadway_show_count,
        urls=parade.broadway_urls,
        support_leaf_desc="Broadway show count is supported by the cited source(s)",
        claim_text=count_claim,
        add_ins=count_ins,
    )

    # Parade broadcast start time (ET)
    parade_time_claim = (
        f"The broadcast of the Macy's Thanksgiving Day Parade {TARGET_YEAR} began at {parade.broadcast_start_time_et} Eastern Time."
        if (parade.broadcast_start_time_et and parade.broadcast_start_time_et.strip())
        else "The broadcast of the Macy's Thanksgiving Day Parade 2025 began at <unspecified time> Eastern Time."
    )
    parade_time_ins = (
        _common_year_instruction()
        + " Verify the broadcast start time in Eastern Time (ET/EST/EDT are acceptable indications of Eastern Time). "
          "If multiple time zones are present, extract the Eastern Time."
    )
    await _add_citation_and_support(
        evaluator,
        parent_node=section_node,
        group_id="parade_broadcast_start_time_et_with_citation",
        group_desc="Provides what time (Eastern Time) the parade broadcast began, with supporting reference URL(s)",
        value=parade.broadcast_start_time_et,
        urls=parade.parade_start_urls,
        support_leaf_desc="Parade broadcast start time (ET) is supported by the cited source(s)",
        claim_text=parade_time_claim,
        add_ins=parade_time_ins,
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for Thanksgiving Day 2025 live TV event facts with citation-backed verification.
    """
    # Initialize evaluator
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_thanksgiving_2025(),
        template_class=ThanksgivingTV2025Extraction,
        extraction_name="thanksgiving_2025_extraction",
    )

    # Add a critical task root node (since Evaluator's root is always non-critical)
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="Provide the requested Thanksgiving Day 2025 U.S. live TV event facts with supporting reference URLs for each required fact",
        parent=root,
        critical=True,
    )

    # Build and verify sub-sections (all critical under task_root)
    await verify_dog_show_winner_details(evaluator, task_root, extraction.dog_show_winner)
    await verify_dog_show_broadcast_info(evaluator, task_root, extraction.dog_show_broadcast)
    await verify_macys_parade_info(evaluator, task_root, extraction.macys_parade)

    # Optional: record custom info
    evaluator.add_custom_info({"target_year": TARGET_YEAR}, info_type="context", info_name="evaluation_context")

    return evaluator.get_summary()