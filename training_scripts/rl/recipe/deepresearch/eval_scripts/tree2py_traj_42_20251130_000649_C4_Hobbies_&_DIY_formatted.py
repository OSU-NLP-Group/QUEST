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
TASK_ID = "thanksgiving_crafts_plan_2025"
TASK_DESCRIPTION = (
    "I'm planning to create a DIY Thanksgiving centerpiece and want to watch the Macy's Thanksgiving Day Parade in 2025 "
    "before shopping for craft supplies on Black Friday morning. Please provide the following information: "
    "(1) What date is Thanksgiving Day in 2025? "
    "(2) What date is Black Friday in 2025? "
    "(3) What time does the Macy's Thanksgiving Day Parade 2025 broadcast start? "
    "(4) What time does the parade broadcast end? "
    "(5) What TV channel broadcasts the parade? "
    "(6) How long is the parade route in miles? "
    "(7) Among Home Depot, Michaels, and Hobby Lobby, which store opens earliest on Black Friday 2025? "
    "(8) What time does that earliest store open?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StoreHours(BaseModel):
    store: Optional[str] = None
    opening_time_bf2025: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ThanksgivingPlanExtraction(BaseModel):
    thanksgiving_date: Optional[str] = None
    thanksgiving_sources: List[str] = Field(default_factory=list)

    black_friday_date: Optional[str] = None
    black_friday_sources: List[str] = Field(default_factory=list)

    parade_start_time: Optional[str] = None
    parade_start_sources: List[str] = Field(default_factory=list)

    parade_end_time: Optional[str] = None
    parade_end_sources: List[str] = Field(default_factory=list)

    parade_channel: Optional[str] = None
    parade_channel_sources: List[str] = Field(default_factory=list)

    parade_route_length_miles: Optional[str] = None
    parade_route_sources: List[str] = Field(default_factory=list)

    earliest_store: Optional[str] = None
    earliest_store_sources: List[str] = Field(default_factory=list)

    earliest_open_time: Optional[str] = None
    earliest_open_time_sources: List[str] = Field(default_factory=list)

    stores: List[StoreHours] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_thanksgiving_plan_info() -> str:
    return """
    Extract from the answer the specific information requested for planning the 2025 Thanksgiving parade viewing and Black Friday craft shopping. 
    Return a single JSON object with the following fields exactly (use null for missing values; use empty arrays for missing URL lists):

    1) thanksgiving_date: string or null
    2) thanksgiving_sources: array of URL strings (the URLs explicitly mentioned for the Thanksgiving date)
    3) black_friday_date: string or null
    4) black_friday_sources: array of URL strings (the URLs explicitly mentioned for the Black Friday date)
    5) parade_start_time: string or null  (e.g., "8:30 AM ET")
    6) parade_start_sources: array of URL strings (the URLs explicitly mentioned for the start time)
    7) parade_end_time: string or null    (e.g., "12:00 PM ET" or "noon ET")
    8) parade_end_sources: array of URL strings (the URLs explicitly mentioned for the end time)
    9) parade_channel: string or null     (e.g., "NBC", "NBC and Peacock")
    10) parade_channel_sources: array of URL strings (the URLs explicitly mentioned for the broadcast channel)
    11) parade_route_length_miles: string or null  (keep any units if provided by the answer, e.g., "2.5 miles")
    12) parade_route_sources: array of URL strings (the URLs explicitly mentioned for the route length)
    13) earliest_store: string or null (one of "Home Depot", "Michaels", "Hobby Lobby" if the answer claims an earliest opening)
    14) earliest_store_sources: array of URL strings (the URLs explicitly mentioned for identifying the earliest store)
    15) earliest_open_time: string or null (e.g., "5:00 AM", "6 AM", include timezone text if present in answer)
    16) earliest_open_time_sources: array of URL strings (the URLs explicitly mentioned for that earliest opening time)
    17) stores: array of objects, each with:
        - store: string or null (e.g., "Home Depot", "Michaels", "Hobby Lobby")
        - opening_time_bf2025: string or null (e.g., "5:00 AM", "6 AM", include timezone text if present)
        - sources: array of URL strings explicitly tied to that store's Black Friday 2025 opening time

    RULES:
    - Extract ONLY what is explicitly present in the answer text. Do not invent or infer values.
    - For URL fields, extract the actual URL strings (from raw links or markdown links).
    - If the answer provides general statements without URLs, keep the relevant URL array empty.
    - Preserve the original phrasing/format of dates and times as they appear in the answer.
    - If multiple URLs are cited for a single item, include them all.
    - If the answer lists store hours for multiple locations or shows a range, extract the opening time text as-is for each store.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _collect_all_store_sources(stores: List[StoreHours]) -> List[str]:
    out: List[str] = []
    for sh in stores:
        if sh and sh.sources:
            out.extend([u for u in sh.sources if isinstance(u, str) and u.strip()])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in out:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _find_store_sources(stores: List[StoreHours], store_name: str) -> List[str]:
    target = (_norm(store_name)).lower()
    out: List[str] = []
    for sh in stores:
        if not sh or not sh.store:
            continue
        if _norm(sh.store).lower() == target:
            out.extend([u for u in sh.sources if isinstance(u, str) and u.strip()])
    # Deduplicate
    seen = set()
    deduped = []
    for u in out:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Thanksgiving 2025 planning task.
    Builds a parallel verification tree with eight leaf checks as specified in the rubric.
    """
    # 1) Initialize evaluator and root (root is non-critical to allow partial credit)
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

    # 2) Extract structured info from the answer
    extraction: ThanksgivingPlanExtraction = await evaluator.extract(
        prompt=prompt_extract_thanksgiving_plan_info(),
        template_class=ThanksgivingPlanExtraction,
        extraction_name="thanksgiving_plan_extraction",
    )

    # 3) Build leaf nodes per rubric
    leaf_nodes = {}

    # Thanksgiving date
    leaf_nodes["Thanksgiving_Date"] = evaluator.add_leaf(
        id="Thanksgiving_Date",
        desc="Correctly identifies the date of Thanksgiving Day 2025",
        parent=root,
        critical=False,
    )
    # Black Friday date
    leaf_nodes["Black_Friday_Date"] = evaluator.add_leaf(
        id="Black_Friday_Date",
        desc="Correctly identifies the date of Black Friday 2025",
        parent=root,
        critical=False,
    )
    # Parade start time
    leaf_nodes["Parade_Start_Time"] = evaluator.add_leaf(
        id="Parade_Start_Time",
        desc="Correctly identifies the start time of the Macy's Thanksgiving Day Parade 2025 broadcast",
        parent=root,
        critical=False,
    )
    # Parade end time
    leaf_nodes["Parade_End_Time"] = evaluator.add_leaf(
        id="Parade_End_Time",
        desc="Correctly identifies the end time of the Macy's Thanksgiving Day Parade 2025 broadcast",
        parent=root,
        critical=False,
    )
    # Parade channel
    leaf_nodes["Parade_Channel"] = evaluator.add_leaf(
        id="Parade_Channel",
        desc="Correctly identifies the TV channel broadcasting the Macy's Thanksgiving Day Parade 2025",
        parent=root,
        critical=False,
    )
    # Parade route length
    leaf_nodes["Parade_Route_Length"] = evaluator.add_leaf(
        id="Parade_Route_Length",
        desc="Correctly identifies the length of the Macy's Thanksgiving Day Parade 2025 route",
        parent=root,
        critical=False,
    )
    # Earliest craft store
    leaf_nodes["Earliest_Craft_Store"] = evaluator.add_leaf(
        id="Earliest_Craft_Store",
        desc="Correctly identifies which craft store among Home Depot, Michaels, and Hobby Lobby opens earliest on Black Friday 2025",
        parent=root,
        critical=False,
    )
    # Earliest opening time
    leaf_nodes["Earliest_Opening_Time"] = evaluator.add_leaf(
        id="Earliest_Opening_Time",
        desc="Correctly identifies the opening time of the earliest-opening craft store on Black Friday 2025",
        parent=root,
        critical=False,
    )

    # 4) Prepare claims and sources
    thanksgiving_date = _norm(extraction.thanksgiving_date)
    black_friday_date = _norm(extraction.black_friday_date)
    parade_start = _norm(extraction.parade_start_time)
    parade_end = _norm(extraction.parade_end_time)
    parade_channel = _norm(extraction.parade_channel)
    parade_route_len = _norm(extraction.parade_route_length_miles)
    earliest_store = _norm(extraction.earliest_store)
    earliest_time = _norm(extraction.earliest_open_time)

    # Sources
    thanksgiving_sources = extraction.thanksgiving_sources or []
    black_friday_sources = extraction.black_friday_sources or []
    start_sources = extraction.parade_start_sources or []
    end_sources = extraction.parade_end_sources or []
    channel_sources = extraction.parade_channel_sources or []
    route_sources = extraction.parade_route_sources or []
    earliest_store_sources = extraction.earliest_store_sources or []
    earliest_time_sources = extraction.earliest_open_time_sources or []

    # Store-wise sources and aggregation
    all_store_sources = _collect_all_store_sources(extraction.stores or [])
    specific_store_sources = _find_store_sources(extraction.stores or [], earliest_store) if earliest_store else []

    # 5) Batch verify (parallel)
    claims_and_sources = [
        (
            f"Thanksgiving Day in 2025 falls on {thanksgiving_date}.",
            thanksgiving_sources,
            leaf_nodes["Thanksgiving_Date"],
            "Verify that the cited webpage explicitly states the date for U.S. Thanksgiving Day 2025. "
            "Accept reasonable date formats (e.g., 'Thursday, November 27, 2025', 'Nov 27, 2025')."
        ),
        (
            f"Black Friday in 2025 falls on {black_friday_date}.",
            black_friday_sources,
            leaf_nodes["Black_Friday_Date"],
            "Verify that the cited webpage explicitly states the date for Black Friday 2025 (the day after U.S. Thanksgiving). "
            "Accept reasonable date formats."
        ),
        (
            f"The 2025 Macy's Thanksgiving Day Parade broadcast starts at {parade_start} (Eastern Time).",
            start_sources,
            leaf_nodes["Parade_Start_Time"],
            "Check the broadcast start time on the cited source(s). Treat times as U.S. Eastern Time (ET/EST/EDT). "
            "Allow minor formatting differences (e.g., '8:30 AM ET' vs '8:30 a.m. ET')."
        ),
        (
            f"The 2025 Macy's Thanksgiving Day Parade broadcast ends at {parade_end} (Eastern Time).",
            end_sources,
            leaf_nodes["Parade_End_Time"],
            "Check the broadcast end time on the cited source(s). Treat times as U.S. Eastern Time (ET). "
            "Allow minor formatting differences and synonyms like 'noon' for 12:00 PM."
        ),
        (
            f"The 2025 Macy's Thanksgiving Day Parade is broadcast on {parade_channel}.",
            channel_sources,
            leaf_nodes["Parade_Channel"],
            "Verify that the cited source(s) name the TV broadcaster. Accept 'NBC' as the primary TV channel. "
            "If the source lists both NBC and Peacock (streaming), consider 'NBC' correct for TV channel; "
            "phrases like 'NBC and Peacock' are acceptable if present in the answer."
        ),
        (
            f"The length of the Macy's Thanksgiving Day Parade route is {parade_route_len}.",
            route_sources,
            leaf_nodes["Parade_Route_Length"],
            "Verify the parade route length (in miles) on the cited source(s). "
            "Allow minor rounding differences (e.g., 2.5 miles vs 2.6 miles) if clearly the same fact."
        ),
        (
            f"Among Home Depot, Michaels, and Hobby Lobby, the store that opens earliest on Black Friday 2025 is {earliest_store}.",
            (earliest_store_sources or []) + all_store_sources,
            leaf_nodes["Earliest_Craft_Store"],
            "Use the cited source(s) to compare Black Friday 2025 opening times for Home Depot, Michaels, and Hobby Lobby. "
            "If two or more share the same earliest time, the claim should reflect a tie; otherwise, confirm the single earliest. "
            "Prefer national announcements or official hours when available; acknowledge that local store hours can vary."
        ),
        (
            f"On Black Friday 2025, {earliest_store} opens at {earliest_time}.",
            (specific_store_sources or []) + earliest_time_sources + earliest_store_sources,
            leaf_nodes["Earliest_Opening_Time"],
            "Verify the claimed opening time for the identified earliest-opening store on Black Friday 2025 using the cited source(s). "
            "Allow minor formatting differences (e.g., '5 AM' vs '5:00 a.m.'). Prefer official or national communications where available."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)

    # 6) Return structured summary
    return evaluator.get_summary()