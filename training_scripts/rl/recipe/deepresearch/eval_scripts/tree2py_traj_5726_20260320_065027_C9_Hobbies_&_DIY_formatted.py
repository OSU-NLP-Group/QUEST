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
TASK_ID = "diy_craft_holiday_2025"
TASK_DESCRIPTION = (
    "Among the four major DIY and craft retail chains in the United States—Home Depot, Lowe's, Michaels, and Hobby Lobby—"
    "analyze their 2025 holiday shopping accessibility and market presence for a holiday craft project planning guide.\n\n"
    "1) Identify which of the four chains has the most U.S. store locations, with exact number and a supporting URL.\n"
    "2) For Black Friday 2025 (Nov 28, 2025), determine each chain’s opening time and which open earliest, with supporting URLs.\n"
    "3) For Christmas Eve 2025 (Dec 24, 2025), determine each chain’s closing time and which close latest, with supporting URLs.\n"
    "4) Confirm whether all four chains are closed on Thanksgiving Day 2025 (Nov 27, 2025) and Christmas Day 2025 (Dec 25, 2025), with supporting URLs.\n"
    "All times should be local time typical for most store locations, and all claims must be supported by official or reliable sources."
)

CHAIN_NAMES = ["Home Depot", "Lowe's", "Michaels", "Hobby Lobby"]


# --------------------------------------------------------------------------- #
# Extraction Models                                                           #
# --------------------------------------------------------------------------- #
class LargestChainExtraction(BaseModel):
    chain_name: Optional[str] = None
    store_count: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class BlackFridayExtraction(BaseModel):
    home_depot_time: Optional[str] = None
    home_depot_urls: List[str] = Field(default_factory=list)

    lowes_time: Optional[str] = None
    lowes_urls: List[str] = Field(default_factory=list)

    michaels_time: Optional[str] = None
    michaels_urls: List[str] = Field(default_factory=list)

    hobby_lobby_time: Optional[str] = None
    hobby_lobby_urls: List[str] = Field(default_factory=list)

    earliest_chains: List[str] = Field(default_factory=list)


class ChristmasEveExtraction(BaseModel):
    home_depot_time: Optional[str] = None
    home_depot_urls: List[str] = Field(default_factory=list)

    lowes_time: Optional[str] = None
    lowes_urls: List[str] = Field(default_factory=list)

    michaels_time: Optional[str] = None
    michaels_urls: List[str] = Field(default_factory=list)

    hobby_lobby_time: Optional[str] = None
    hobby_lobby_urls: List[str] = Field(default_factory=list)

    latest_chains: List[str] = Field(default_factory=list)


class HolidayChainURLs(BaseModel):
    home_depot: List[str] = Field(default_factory=list)
    lowes: List[str] = Field(default_factory=list)
    michaels: List[str] = Field(default_factory=list)
    hobby_lobby: List[str] = Field(default_factory=list)


class HolidayClosuresExtraction(BaseModel):
    thanksgiving: HolidayChainURLs = Field(default_factory=HolidayChainURLs)
    christmas: HolidayChainURLs = Field(default_factory=HolidayChainURLs)


# --------------------------------------------------------------------------- #
# Extraction Prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_largest_chain() -> str:
    return """
    Extract the identification of which chain has the most U.S. store locations among these four: "Home Depot", "Lowe's", "Michaels", "Hobby Lobby".
    Return:
    - chain_name: the name of the chain (one of the four exactly as listed above).
    - store_count: the exact number of U.S. store locations as stated in the answer (leave as a string as written, e.g., "2,333").
    - reference_urls: a list of URL(s) explicitly cited in the answer that support the store count and/or explicitly state it is the largest by U.S. stores.
    Only extract values explicitly present in the answer text. If something is missing, set it to null or an empty array accordingly.
    """


def prompt_extract_black_friday() -> str:
    return """
    For Black Friday 2025 (November 28, 2025), extract opening times and sources for the four chains.
    Return:
    - home_depot_time: opening time for Home Depot on Black Friday 2025 (local time string, e.g., "5:00 AM").
    - home_depot_urls: list of reference URLs explicitly cited in the answer supporting Home Depot's Black Friday 2025 opening time.
    - lowes_time: opening time for Lowe's on Black Friday 2025.
    - lowes_urls: list of reference URLs for Lowe's Black Friday 2025 opening time.
    - michaels_time: opening time for Michaels on Black Friday 2025.
    - michaels_urls: list of reference URLs for Michaels' Black Friday 2025 opening time.
    - hobby_lobby_time: opening time for Hobby Lobby on Black Friday 2025.
    - hobby_lobby_urls: list of reference URLs for Hobby Lobby's Black Friday 2025 opening time.
    - earliest_chains: array of chain name(s) (from the set: "Home Depot", "Lowe's", "Michaels", "Hobby Lobby") that open earliest based on the provided times; include ties if equal.
    Use times exactly as they appear in the answer. Extract only URLs explicitly present in the answer.
    """


def prompt_extract_christmas_eve() -> str:
    return """
    For Christmas Eve 2025 (December 24, 2025), extract closing times and sources for the four chains.
    Return:
    - home_depot_time: closing time for Home Depot on Christmas Eve 2025 (local time string, e.g., "6:00 PM").
    - home_depot_urls: list of reference URLs explicitly cited in the answer supporting Home Depot's Christmas Eve 2025 closing time.
    - lowes_time: closing time for Lowe's on Christmas Eve 2025.
    - lowes_urls: list of reference URLs for Lowe's Christmas Eve 2025 closing time.
    - michaels_time: closing time for Michaels on Christmas Eve 2025.
    - michaels_urls: list of reference URLs for Michaels' Christmas Eve 2025 closing time.
    - hobby_lobby_time: closing time for Hobby Lobby on Christmas Eve 2025.
    - hobby_lobby_urls: list of reference URLs for Hobby Lobby's Christmas Eve 2025 closing time.
    - latest_chains: array of chain name(s) (from: "Home Depot", "Lowe's", "Michaels", "Hobby Lobby") that close latest based on the provided times; include ties if equal.
    Use times exactly as they appear in the answer. Extract only URLs explicitly present in the answer.
    """


def prompt_extract_holiday_closures() -> str:
    return """
    Extract closure reference URLs (official or reliable news) for Thanksgiving Day 2025 (Nov 27, 2025) and Christmas Day 2025 (Dec 25, 2025) for each chain.
    Return an object with two keys: 'thanksgiving' and 'christmas'. Each should contain:
    - home_depot: array of URLs supporting closure claim for Home Depot on that day.
    - lowes: array of URLs supporting closure claim for Lowe's on that day.
    - michaels: array of URLs supporting closure claim for Michaels on that day.
    - hobby_lobby: array of URLs supporting closure claim for Hobby Lobby on that day.
    Only include URLs that are explicitly present in the answer. If none are provided for a chain/day, leave the array empty.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_urls(*url_lists: Optional[List[str]]) -> List[str]:
    combined: List[str] = []
    for lst in url_lists:
        if lst:
            combined.extend([u for u in lst if isinstance(u, str) and u.strip()])
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for u in combined:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _fmt_chain_list(names: List[str]) -> str:
    return ", ".join(names) if names else "None"


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_largest_chain(
    evaluator: Evaluator,
    parent_node,
    data: LargestChainExtraction,
) -> None:
    # Parent node for "Largest_Chain_by_Store_Count"
    node = evaluator.add_parallel(
        id="Largest_Chain_by_Store_Count",
        desc="Identify the DIY/craft chain with the most U.S. store locations among Home Depot, Lowe's, Michaels, and Hobby Lobby",
        parent=parent_node,
        critical=True,  # Parent is critical; all children must be critical
    )

    # Leaf: Chain_Name (verify with provided reference URLs if any)
    chain_leaf = evaluator.add_leaf(
        id="Chain_Name",
        desc="Provide the name of the chain with the most stores among the four specified chains",
        parent=node,
        critical=True,
    )
    chain_claim = (
        f"Among Home Depot, Lowe's, Michaels, and Hobby Lobby, the chain with the most U.S. store locations is '{data.chain_name}'."
    )
    await evaluator.verify(
        claim=chain_claim,
        node=chain_leaf,
        sources=data.reference_urls,
        additional_instruction="Validate that the provided source(s) explicitly support that this chain is the largest by number of U.S. stores among the four specified chains."
    )

    # Leaf: Store_Count (value verification with URL)
    count_leaf = evaluator.add_leaf(
        id="Store_Count",
        desc="Provide the exact number of U.S. store locations for the identified chain",
        parent=node,
        critical=True,
    )
    count_claim = f"As of 2025 or the most recent figure cited, {data.chain_name} has {data.store_count} U.S. stores."
    await evaluator.verify(
        claim=count_claim,
        node=count_leaf,
        sources=data.reference_urls,
        additional_instruction="Check the referenced page(s) for an explicit U.S. store count for the named chain. Allow minor formatting differences like commas."
    )

    # Leaf: Store_Count_Reference (existence of at least one URL)
    ref_exists = bool(data.reference_urls)
    evaluator.add_custom_node(
        result=ref_exists,
        id="Store_Count_Reference",
        desc="Provide a verifiable reference URL for the store count",
        parent=node,
        critical=True
    )


async def _verify_chain_time_group(
    evaluator: Evaluator,
    parent_node,
    *,
    group_id: str,
    group_desc: str,
    time_leaf_id: str,
    ref_leaf_id: str,
    chain_display_name: str,
    time_str: Optional[str],
    urls: List[str],
    event_desc: str,
    event_date_str: str,
    open_or_close: str,  # "opening" or "closing"
) -> None:
    """
    Build a chain-specific sub-node with (1) time verification via URLs and (2) reference existence check.
    """
    grp = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent_node,
        critical=True,  # Under a critical parent, must be critical
    )

    # Time value verification leaf
    time_leaf = evaluator.add_leaf(
        id=time_leaf_id,
        desc=f"State {chain_display_name}'s {open_or_close} time on {event_desc}",
        parent=grp,
        critical=True,
    )
    time_claim = (
        f"On {event_desc} ({event_date_str}), {chain_display_name} stores {('open' if open_or_close=='opening' else 'close')} at {time_str} local time."
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_leaf,
        sources=urls,
        additional_instruction=(
            f"Confirm the claimed {open_or_close} time for {chain_display_name} on {event_desc} ({event_date_str}) "
            f"from the provided URL(s). Use U.S. store policies; allow phrasing like 'Most stores open at ...'."
        ),
    )

    # Reference existence check
    evaluator.add_custom_node(
        result=bool(urls),
        id=ref_leaf_id,
        desc=f"Provide reference URL for {chain_display_name}'s {event_desc} hours",
        parent=grp,
        critical=True
    )


async def verify_black_friday(
    evaluator: Evaluator,
    parent_node,
    bf: BlackFridayExtraction,
) -> None:
    # Parent node for BF analysis
    node = evaluator.add_parallel(
        id="Black_Friday_Opening_Analysis",
        desc="Determine opening times for all four chains on Black Friday 2025 and identify which open earliest",
        parent=parent_node,
        critical=True,
    )

    # Four chain groups
    await _verify_chain_time_group(
        evaluator, node,
        group_id="Home_Depot_Black_Friday",
        group_desc="Provide Home Depot's Black Friday 2025 opening time with reference",
        time_leaf_id="Home_Depot_Time_Value",
        ref_leaf_id="Home_Depot_BF_Reference",
        chain_display_name="Home Depot",
        time_str=bf.home_depot_time,
        urls=bf.home_depot_urls,
        event_desc="Black Friday 2025",
        event_date_str="November 28, 2025",
        open_or_close="opening",
    )

    await _verify_chain_time_group(
        evaluator, node,
        group_id="Lowes_Black_Friday",
        group_desc="Provide Lowe's Black Friday 2025 opening time with reference",
        time_leaf_id="Lowes_Time_Value",
        ref_leaf_id="Lowes_BF_Reference",
        chain_display_name="Lowe's",
        time_str=bf.lowes_time,
        urls=bf.lowes_urls,
        event_desc="Black Friday 2025",
        event_date_str="November 28, 2025",
        open_or_close="opening",
    )

    await _verify_chain_time_group(
        evaluator, node,
        group_id="Michaels_Black_Friday",
        group_desc="Provide Michaels' Black Friday 2025 opening time with reference",
        time_leaf_id="Michaels_Time_Value",
        ref_leaf_id="Michaels_BF_Reference",
        chain_display_name="Michaels",
        time_str=bf.michaels_time,
        urls=bf.michaels_urls,
        event_desc="Black Friday 2025",
        event_date_str="November 28, 2025",
        open_or_close="opening",
    )

    await _verify_chain_time_group(
        evaluator, node,
        group_id="Hobby_Lobby_Black_Friday",
        group_desc="Provide Hobby Lobby's Black Friday 2025 opening time with reference",
        time_leaf_id="Hobby_Lobby_Time_Value",
        ref_leaf_id="Hobby_Lobby_BF_Reference",
        chain_display_name="Hobby Lobby",
        time_str=bf.hobby_lobby_time,
        urls=bf.hobby_lobby_urls,
        event_desc="Black Friday 2025",
        event_date_str="November 28, 2025",
        open_or_close="opening",
    )

    # Earliest chain identification (logical check based on extracted times)
    earliest_leaf = evaluator.add_leaf(
        id="Earliest_Chain_Identification",
        desc="Identify which chain(s) open earliest on Black Friday based on the provided times",
        parent=node,
        critical=True,
    )
    times_summary = (
        f"Home Depot: {bf.home_depot_time}; Lowe's: {bf.lowes_time}; "
        f"Michaels: {bf.michaels_time}; Hobby Lobby: {bf.hobby_lobby_time}."
    )
    earliest_claim = (
        f"Given the opening times for Black Friday 2025 are: {times_summary} "
        f"the earliest opening chain(s) are: {_fmt_chain_list(bf.earliest_chains)}."
    )
    await evaluator.verify(
        claim=earliest_claim,
        node=earliest_leaf,
        additional_instruction=(
            "Determine the earliest opening time among the four given local times (treat equivalent times as a tie). "
            "If any time is missing or 'varies', this item should be judged based on the provided times only."
        ),
    )


async def verify_christmas_eve(
    evaluator: Evaluator,
    parent_node,
    ce: ChristmasEveExtraction,
) -> None:
    # Parent node for Christmas Eve analysis
    node = evaluator.add_parallel(
        id="Christmas_Eve_Closing_Analysis",
        desc="Determine closing times for all four chains on Christmas Eve 2025 and identify which close latest",
        parent=parent_node,
        critical=True,
    )

    # Four chain groups
    await _verify_chain_time_group(
        evaluator, node,
        group_id="Home_Depot_Christmas_Eve",
        group_desc="Provide Home Depot's Christmas Eve 2025 closing time with reference",
        time_leaf_id="Home_Depot_CE_Time_Value",
        ref_leaf_id="Home_Depot_CE_Reference",
        chain_display_name="Home Depot",
        time_str=ce.home_depot_time,
        urls=ce.home_depot_urls,
        event_desc="Christmas Eve 2025",
        event_date_str="December 24, 2025",
        open_or_close="closing",
    )

    await _verify_chain_time_group(
        evaluator, node,
        group_id="Lowes_Christmas_Eve",
        group_desc="Provide Lowe's Christmas Eve 2025 closing time with reference",
        time_leaf_id="Lowes_CE_Time_Value",
        ref_leaf_id="Lowes_CE_Reference",
        chain_display_name="Lowe's",
        time_str=ce.lowes_time,
        urls=ce.lowes_urls,
        event_desc="Christmas Eve 2025",
        event_date_str="December 24, 2025",
        open_or_close="closing",
    )

    await _verify_chain_time_group(
        evaluator, node,
        group_id="Michaels_Christmas_Eve",
        group_desc="Provide Michaels' Christmas Eve 2025 closing time with reference",
        time_leaf_id="Michaels_CE_Time_Value",
        ref_leaf_id="Michaels_CE_Reference",
        chain_display_name="Michaels",
        time_str=ce.michaels_time,
        urls=ce.michaels_urls,
        event_desc="Christmas Eve 2025",
        event_date_str="December 24, 2025",
        open_or_close="closing",
    )

    await _verify_chain_time_group(
        evaluator, node,
        group_id="Hobby_Lobby_Christmas_Eve",
        group_desc="Provide Hobby Lobby's Christmas Eve 2025 closing time with reference",
        time_leaf_id="Hobby_Lobby_CE_Time_Value",
        ref_leaf_id="Hobby_Lobby_CE_Reference",
        chain_display_name="Hobby Lobby",
        time_str=ce.hobby_lobby_time,
        urls=ce.hobby_lobby_urls,
        event_desc="Christmas Eve 2025",
        event_date_str="December 24, 2025",
        open_or_close="closing",
    )

    # Latest chain identification (logical check based on extracted times)
    latest_leaf = evaluator.add_leaf(
        id="Latest_Chain_Identification",
        desc="Identify which chain(s) close latest on Christmas Eve based on the provided times",
        parent=node,
        critical=True,
    )
    times_summary = (
        f"Home Depot: {ce.home_depot_time}; Lowe's: {ce.lowes_time}; "
        f"Michaels: {ce.michaels_time}; Hobby Lobby: {ce.hobby_lobby_time}."
    )
    latest_claim = (
        f"Given the closing times for Christmas Eve 2025 are: {times_summary} "
        f"the latest closing chain(s) are: {_fmt_chain_list(ce.latest_chains)}."
    )
    await evaluator.verify(
        claim=latest_claim,
        node=latest_leaf,
        additional_instruction=(
            "Determine the latest closing time among the four given local times (treat equivalent times as a tie). "
            "If any time is missing or 'varies', this item should be judged based on the provided times only."
        ),
    )


async def verify_additional_holidays(
    evaluator: Evaluator,
    parent_node,
    hol: HolidayClosuresExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Additional_Holiday_Verification",
        desc="Verify Thanksgiving and Christmas Day closure status for all chains",
        parent=parent_node,
        critical=True,
    )

    # Thanksgiving closure
    tg = evaluator.add_parallel(
        id="Thanksgiving_Closure",
        desc="Verify all four chains are closed on Thanksgiving Day 2025",
        parent=node,
        critical=True,
    )
    tg_urls = _combine_urls(
        hol.thanksgiving.home_depot,
        hol.thanksgiving.lowes,
        hol.thanksgiving.michaels,
        hol.thanksgiving.hobby_lobby,
    )
    tg_closed_leaf = evaluator.add_leaf(
        id="All_Chains_Closed_Thanksgiving",
        desc="Confirm Home Depot, Lowe's, Michaels, and Hobby Lobby are all closed on Thanksgiving",
        parent=tg,
        critical=True,
    )
    tg_claim = (
        "On Thanksgiving Day 2025 (November 27, 2025), Home Depot, Lowe's, Michaels, and Hobby Lobby U.S. stores are closed."
    )
    await evaluator.verify(
        claim=tg_claim,
        node=tg_closed_leaf,
        sources=tg_urls,
        additional_instruction="Verify closure status for all four chains on the specified date using the provided URLs. Prefer official or reliable news sources."
    )

    tg_refs_leaf = evaluator.add_custom_node(
        result=all([
            bool(hol.thanksgiving.home_depot),
            bool(hol.thanksgiving.lowes),
            bool(hol.thanksgiving.michaels),
            bool(hol.thanksgiving.hobby_lobby),
        ]),
        id="Thanksgiving_References",
        desc="Provide reference URLs supporting Thanksgiving closure claims",
        parent=tg,
        critical=True
    )

    # Christmas Day closure
    xmas = evaluator.add_parallel(
        id="Christmas_Day_Closure",
        desc="Verify all four chains are closed on Christmas Day 2025",
        parent=node,
        critical=True,
    )
    x_urls = _combine_urls(
        hol.christmas.home_depot,
        hol.christmas.lowes,
        hol.christmas.michaels,
        hol.christmas.hobby_lobby,
    )
    x_closed_leaf = evaluator.add_leaf(
        id="All_Chains_Closed_Christmas",
        desc="Confirm Home Depot, Lowe's, Michaels, and Hobby Lobby are all closed on Christmas Day",
        parent=xmas,
        critical=True,
    )
    x_claim = (
        "On Christmas Day 2025 (December 25, 2025), Home Depot, Lowe's, Michaels, and Hobby Lobby U.S. stores are closed."
    )
    await evaluator.verify(
        claim=x_claim,
        node=x_closed_leaf,
        sources=x_urls,
        additional_instruction="Verify closure status for all four chains on Christmas Day using the provided URLs. Prefer official or reliable news sources."
    )

    evaluator.add_custom_node(
        result=all([
            bool(hol.christmas.home_depot),
            bool(hol.christmas.lowes),
            bool(hol.christmas.michaels),
            bool(hol.christmas.hobby_lobby),
        ]),
        id="Christmas_References",
        desc="Provide reference URLs supporting Christmas Day closure claims",
        parent=xmas,
        critical=True
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
    """
    Entry point for evaluating an answer to the DIY/Craft Holiday 2025 analysis task.
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
        default_model=model,
    )

    # Extraction (can be parallelized)
    largest_chain_task = evaluator.extract(
        prompt=prompt_extract_largest_chain(),
        template_class=LargestChainExtraction,
        extraction_name="largest_chain_extraction",
    )
    bf_task = evaluator.extract(
        prompt=prompt_extract_black_friday(),
        template_class=BlackFridayExtraction,
        extraction_name="black_friday_extraction",
    )
    ce_task = evaluator.extract(
        prompt=prompt_extract_christmas_eve(),
        template_class=ChristmasEveExtraction,
        extraction_name="christmas_eve_extraction",
    )
    hol_task = evaluator.extract(
        prompt=prompt_extract_holiday_closures(),
        template_class=HolidayClosuresExtraction,
        extraction_name="holiday_closures_extraction",
    )

    largest_chain, bf, ce, hol = await asyncio.gather(
        largest_chain_task, bf_task, ce_task, hol_task
    )

    # Build the top-level "Task_Completion" node (critical) and attach all subtrees
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Complete identification and verification of DIY and craft retail chains' holiday hours and rankings for 2025",
        parent=root,
        critical=True,
    )

    # Subtrees
    await verify_largest_chain(evaluator, task_node, largest_chain)
    await verify_black_friday(evaluator, task_node, bf)
    await verify_christmas_eve(evaluator, task_node, ce)
    await verify_additional_holidays(evaluator, task_node, hol)

    # Return evaluator summary
    return evaluator.get_summary()