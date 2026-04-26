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
TASK_ID = "summerfest_2026_amfam_amphitheater"
TASK_DESCRIPTION = """
Identify three American Family Insurance Amphitheater headliner performances at Summerfest 2026 in Milwaukee, Wisconsin, that meet the following criteria:

Performance 1: A country music artist performing during the first festival weekend (June 18-20, 2026)

Performance 2: A pop or rock music artist performing during the second festival weekend (June 25-27, 2026)

Performance 3: Any genre artist performing during the third festival weekend (July 2-4, 2026)

For each performance, provide:
- The artist's name and music genre
- The exact performance date and showtime
- The venue name and location (city, state)
- The supporting act or special guest (if one is listed)
- A reference URL from the official Summerfest website (www.summerfest.com) confirming the performance details

Note: The American Family Insurance Amphitheater is Summerfest's main stage venue and requires a separate amphitheater ticket. Summerfest 2026 takes place over three consecutive weekends at Henry Maier Festival Park in Milwaukee, Wisconsin.
"""

W1_DATES = ["June 18, 2026", "June 19, 2026", "June 20, 2026"]
W2_DATES = ["June 25, 2026", "June 26, 2026", "June 27, 2026"]
W3_DATES = ["July 2, 2026", "July 3, 2026", "July 4, 2026"]

W1_RANGE_STR = "June 18–20, 2026"
W2_RANGE_STR = "June 25–27, 2026"
W3_RANGE_STR = "July 2–4, 2026"


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class PerformanceItem(BaseModel):
    artist_name: Optional[str] = None
    genre: Optional[str] = None
    date: Optional[str] = None                 # Example: "June 19, 2026"
    time: Optional[str] = None                 # Example: "7:30 PM"
    venue_name: Optional[str] = None           # Expect: "American Family Insurance Amphitheater"
    location_city: Optional[str] = None        # Expect: "Milwaukee"
    location_state: Optional[str] = None       # Expect: "Wisconsin" or "WI"
    supporting_act: Optional[str] = None       # If multiple listed, include primary or first listed
    reference_url: Optional[str] = None        # Must be an official Summerfest URL (summerfest.com)


class AmphitheaterExtraction(BaseModel):
    performances: List[PerformanceItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_performances() -> str:
    return """
    Extract American Family Insurance Amphitheater headliner performances mentioned in the answer.
    Rules:
    - Only extract events that the answer explicitly claims are at the "American Family Insurance Amphitheater" during Summerfest 2026.
    - Each item should correspond to one headliner performance at that amphitheater.
    - For the reference_url, only include an official Summerfest website URL (must contain 'summerfest.com'). If the answer provides multiple URLs, prefer the one that most directly confirms the date/time/venue for that show. If no valid Summerfest URL is present, set reference_url to null.

    For each performance, extract these fields:
    - artist_name: the headlining artist name
    - genre: the artist's genre exactly as stated in the answer (e.g., "country", "pop", "rock", etc.). If not explicitly stated, return null.
    - date: the exact performance date string as presented (e.g., "June 19, 2026"). Do not normalize beyond what the answer states.
    - time: the showtime string as presented (e.g., "7:30 PM"). If not explicitly stated, return null.
    - venue_name: the venue name as stated (should be "American Family Insurance Amphitheater")
    - location_city: the city (should be "Milwaukee")
    - location_state: the state (e.g., "Wisconsin" or "WI")
    - supporting_act: the supporting act or special guest if explicitly mentioned; otherwise null
    - reference_url: a single Summerfest official URL (must contain 'summerfest.com') that the answer cites for this performance; otherwise null

    Return a JSON object with:
    {
      "performances": [ { ... up to 6 items if the answer lists more than 3 ... } ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper Utilities                                                            #
# --------------------------------------------------------------------------- #
def is_nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def is_summerfest_url(url: Optional[str]) -> bool:
    if not is_nonempty(url):
        return False
    return "summerfest.com" in url.lower()


def date_mentions_any(date_str: Optional[str], allowed_dates: List[str]) -> bool:
    if not is_nonempty(date_str):
        return False
    low = date_str.lower()
    return any(d.lower() in low for d in allowed_dates)


def select_index_by_constraints(
    items: List[PerformanceItem],
    used: set,
    allowed_dates: Optional[List[str]] = None,
    genre_contains_any: Optional[List[str]] = None,
) -> int:
    # Try strict match: date within weekend AND genre condition (if provided)
    for i, it in enumerate(items):
        if i in used:
            continue
        ok_date = True if allowed_dates is None else date_mentions_any(it.date, allowed_dates)
        ok_genre = True
        if genre_contains_any is not None:
            g = (it.genre or "").lower()
            ok_genre = any(k.lower() in g for k in genre_contains_any)
        if ok_date and ok_genre:
            return i
    # Try relaxed: only date
    if allowed_dates is not None:
        for i, it in enumerate(items):
            if i in used:
                continue
            if date_mentions_any(it.date, allowed_dates):
                return i
    # Fallback: first unused
    for i, _ in enumerate(items):
        if i not in used:
            return i
    # If no items, return -1 (caller should handle padding)
    return -1


# --------------------------------------------------------------------------- #
# Verification Builders                                                       #
# --------------------------------------------------------------------------- #
async def build_performance_verification(
    evaluator: Evaluator,
    parent_node,
    perf: PerformanceItem,
    perf_label: str,                       # "P1", "P2", "P3"
    performance_node_desc: str,           # description for the performance node
    weekend_range_str: str,               # e.g., "June 18–20, 2026"
    allowed_dates: List[str],             # list of allowed exact strings
    genre_requirement: str,               # "country", "pop_or_rock", "any"
) -> None:
    # Performance-level node (parallel, non-critical)
    perf_node = evaluator.add_parallel(
        id=f"{perf_label}_Node",
        desc=performance_node_desc,
        parent=parent_node,
        critical=False
    )

    # Artist Identification (critical)
    artist_ident_node = evaluator.add_parallel(
        id=f"{perf_label}_Artist_Identification",
        desc="The artist is identified and confirmed",
        parent=perf_node,
        critical=True
    )

    # P*_Artist_Name (critical) - existence check
    evaluator.add_custom_node(
        result=is_nonempty(perf.artist_name),
        id=f"{perf_label}_Artist_Name",
        desc="The artist's name is provided",
        parent=artist_ident_node,
        critical=True
    )

    # P*_Artist_Genre (critical) - verify genre requirement
    genre_leaf = evaluator.add_leaf(
        id=f"{perf_label}_Artist_Genre",
        desc=(
            "The artist is confirmed to perform country music" if genre_requirement == "country"
            else "The artist is confirmed to perform pop or rock music" if genre_requirement == "pop_or_rock"
            else "The artist's genre is identified"
        ),
        parent=artist_ident_node,
        critical=True
    )
    if genre_requirement == "country":
        genre_claim = f"The performing artist {perf.artist_name or 'the artist'} is a country music artist."
    elif genre_requirement == "pop_or_rock":
        genre_claim = f"The performing artist {perf.artist_name or 'the artist'} is a pop or rock music artist (either pop or rock satisfies this requirement)."
    else:
        # any genre: just ensure the genre is identifiable (use extracted genre string if present)
        if is_nonempty(perf.genre):
            genre_claim = f"The artist's genre can be identified as '{perf.genre}'."
        else:
            genre_claim = "The official page identifies the artist's music genre (any genre is acceptable)."

    await evaluator.verify(
        claim=genre_claim,
        node=genre_leaf,
        sources=perf.reference_url,
        additional_instruction=(
            "Only accept if the provided URL is an official Summerfest page (URL contains 'summerfest.com') "
            "and the page explicitly states or clearly implies the claimed genre. "
            "If the URL is missing or not a Summerfest domain, mark as not supported."
        )
    )

    # P*_Artist_Reference_URL (critical) - verify artist and headliner/Amphitheater context
    artist_ref_leaf = evaluator.add_leaf(
        id=f"{perf_label}_Artist_Reference_URL",
        desc="A valid reference URL from official Summerfest website confirming the artist",
        parent=artist_ident_node,
        critical=True
    )
    artist_ref_claim = (
        f"This official Summerfest webpage confirms that {perf.artist_name or 'the artist'} will perform at the "
        f"American Family Insurance Amphitheater during Summerfest 2026."
    )
    await evaluator.verify(
        claim=artist_ref_claim,
        node=artist_ref_leaf,
        sources=perf.reference_url,
        additional_instruction=(
            "Confirm the URL is on the official Summerfest website (must contain 'summerfest.com'). "
            "The page should clearly indicate the artist is performing at the 'American Family Insurance Amphitheater' "
            "as part of Summerfest 2026. If the URL is missing or not a Summerfest domain, mark as not supported."
        )
    )

    # Performance Date & Time (critical)
    dt_node = evaluator.add_parallel(
        id=f"{perf_label}_Performance_Date_Time",
        desc="The performance date and time are correctly identified",
        parent=perf_node,
        critical=True
    )

    # P*_Date_Verification (critical) - within the given weekend
    date_leaf = evaluator.add_leaf(
        id=f"{perf_label}_Date_Verification",
        desc=f"The performance date falls within {weekend_range_str}",
        parent=dt_node,
        critical=True
    )
    exact_date = perf.date or "(date not provided)"
    date_claim = (
        f"The performance date is {exact_date}. It falls within {weekend_range_str} "
        f"({', '.join(allowed_dates)})."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=perf.reference_url,
        additional_instruction=(
            "Check the date shown on the official Summerfest page. Judge True only if the date on the page "
            f"is one of: {', '.join(allowed_dates)}. If the URL is missing or not on 'summerfest.com', mark as not supported."
        )
    )

    # P*_Time_Verification (critical) - showtime
    time_leaf = evaluator.add_leaf(
        id=f"{perf_label}_Time_Verification",
        desc="The specific showtime is correctly provided",
        parent=dt_node,
        critical=True
    )
    exact_time = perf.time or "(time not provided)"
    time_claim = f"The scheduled showtime for this performance is {exact_time} (local time)."
    await evaluator.verify(
        claim=time_claim,
        node=time_leaf,
        sources=perf.reference_url,
        additional_instruction=(
            "Verify the showtime on the official Summerfest event page. Allow minor formatting variations "
            "(e.g., '7:30 PM' vs '7:30 p.m.'). If the URL is missing or not on 'summerfest.com', mark as not supported."
        )
    )

    # P*_DateTime_Reference_URL (critical) - page shows both date & time
    dt_ref_leaf = evaluator.add_leaf(
        id=f"{perf_label}_DateTime_Reference_URL",
        desc="A valid reference URL confirming the date and time",
        parent=dt_node,
        critical=True
    )
    dt_ref_claim = (
        f"The official Summerfest page explicitly lists both the date '{exact_date}' and time '{exact_time}' "
        f"for this Amphitheater show."
    )
    await evaluator.verify(
        claim=dt_ref_claim,
        node=dt_ref_leaf,
        sources=perf.reference_url,
        additional_instruction=(
            "Confirm that both date and time are explicitly present on the official Summerfest page. "
            "If either is missing on the page, or the URL is not from 'summerfest.com', mark as not supported."
        )
    )

    # Venue Details (critical)
    venue_node = evaluator.add_parallel(
        id=f"{perf_label}_Venue_Details",
        desc="Venue information is correctly provided",
        parent=perf_node,
        critical=True
    )

    # P*_Venue_Name (critical)
    venue_name_leaf = evaluator.add_leaf(
        id=f"{perf_label}_Venue_Name",
        desc="Confirmed as American Family Insurance Amphitheater",
        parent=venue_node,
        critical=True
    )
    venue_name_claim = "The venue for this show is the American Family Insurance Amphitheater."
    await evaluator.verify(
        claim=venue_name_claim,
        node=venue_name_leaf,
        sources=perf.reference_url,
        additional_instruction=(
            "Verify that the official Summerfest page explicitly shows the venue as 'American Family Insurance Amphitheater'. "
            "If the URL is missing or not from 'summerfest.com', mark as not supported."
        )
    )

    # P*_Venue_Location (critical)
    venue_loc_leaf = evaluator.add_leaf(
        id=f"{perf_label}_Venue_Location",
        desc="Location specified as Milwaukee, Wisconsin",
        parent=venue_node,
        critical=True
    )
    city = perf.location_city or "Milwaukee"
    state = perf.location_state or "Wisconsin"
    venue_loc_claim = f"The show location is {city}, {state}."
    await evaluator.verify(
        claim=venue_loc_claim,
        node=venue_loc_leaf,
        sources=perf.reference_url,
        additional_instruction=(
            "Verify that the official Summerfest page indicates the location in Milwaukee, Wisconsin (or WI). "
            "If the URL is missing or not from 'summerfest.com', mark as not supported."
        )
    )

    # Supporting Act (non-critical)
    support_node = evaluator.add_parallel(
        id=f"{perf_label}_Supporting_Act",
        desc="If the performance has a supporting act or special guest, it is identified",
        parent=perf_node,
        critical=False
    )

    # P*_Support_Artist_Name (non-critical)
    support_name_leaf = evaluator.add_leaf(
        id=f"{perf_label}_Support_Artist_Name",
        desc="The supporting artist's name is provided if one exists",
        parent=support_node,
        critical=False
    )
    if is_nonempty(perf.supporting_act):
        support_name_claim = (
            f"The official page lists '{perf.supporting_act}' as a supporting act or special guest for this Amphitheater show."
        )
    else:
        support_name_claim = (
            "The official page does not list any supporting act or special guest for this Amphitheater show."
        )
    await evaluator.verify(
        claim=support_name_claim,
        node=support_name_leaf,
        sources=perf.reference_url,
        additional_instruction=(
            "Check the official Summerfest page. If a support/special guest is shown, the provided name must match. "
            "If none is shown, an empty or missing supporting name is acceptable. "
            "If the URL is missing or not on 'summerfest.com', treat as not supported."
        )
    )

    # P*_Support_Reference_URL (non-critical)
    support_ref_leaf = evaluator.add_leaf(
        id=f"{perf_label}_Support_Reference_URL",
        desc="A reference URL confirming the supporting act",
        parent=support_node,
        critical=False
    )
    if is_nonempty(perf.supporting_act):
        support_ref_claim = (
            f"This official Summerfest page confirms '{perf.supporting_act}' as a supporting act or special guest."
        )
    else:
        support_ref_claim = "This official Summerfest page confirms there is no listed supporting act or special guest."
    await evaluator.verify(
        claim=support_ref_claim,
        node=support_ref_leaf,
        sources=perf.reference_url,
        additional_instruction=(
            "Confirm on the official Summerfest page. If a support/special guest is listed, it should match the extracted name. "
            "If none is listed, this claim should still be considered supported. "
            "If the URL is missing or not on 'summerfest.com', treat as not supported."
        )
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry                                                       #
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
    # Initialize evaluator (root must be non-critical to allow mixed-critical children)
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

    # Extract performances
    extraction = await evaluator.extract(
        prompt=prompt_extract_performances(),
        template_class=AmphitheaterExtraction,
        extraction_name="amfam_amphitheater_performances"
    )

    # Add high-level root node to mirror rubric (set non-critical to allow partial)
    summerfest_root = evaluator.add_parallel(
        id="Summerfest_2026_Amphitheater_Performances",
        desc="Identify three American Family Insurance Amphitheater headliner performances at Summerfest 2026 in Milwaukee, Wisconsin, that meet specific criteria",
        parent=root,
        critical=False
    )

    # Prepare exactly 3 performance slots
    items = list(extraction.performances or [])
    while len(items) < 3:
        items.append(PerformanceItem())

    # Selection heuristic: try to map answer items to required weekends/genres
    used_indices: set = set()
    idx_p1 = select_index_by_constraints(items, used_indices, allowed_dates=W1_DATES, genre_contains_any=["country"])
    if idx_p1 == -1:
        idx_p1 = 0
    used_indices.add(idx_p1)

    idx_p2 = select_index_by_constraints(items, used_indices, allowed_dates=W2_DATES, genre_contains_any=["pop", "rock"])
    if idx_p2 == -1:
        idx_p2 = 1 if 1 < len(items) else 0
    used_indices.add(idx_p2)

    idx_p3 = select_index_by_constraints(items, used_indices, allowed_dates=W3_DATES, genre_contains_any=None)
    if idx_p3 == -1:
        # Pick any remaining or fallback
        for i in range(len(items)):
            if i not in used_indices:
                idx_p3 = i
                break
        if idx_p3 == -1:
            idx_p3 = 2 if 2 < len(items) else 0
    used_indices.add(idx_p3)

    perf1 = items[idx_p1] if idx_p1 < len(items) else PerformanceItem()
    perf2 = items[idx_p2] if idx_p2 < len(items) else PerformanceItem()
    perf3 = items[idx_p3] if idx_p3 < len(items) else PerformanceItem()

    # Add custom info for transparency
    evaluator.add_custom_info(
        {
            "selected_indices": {"P1": idx_p1, "P2": idx_p2, "P3": idx_p3},
            "weekend_windows": {
                "P1": W1_DATES,
                "P2": W2_DATES,
                "P3": W3_DATES
            }
        },
        info_type="selection_summary"
    )

    # Build subtrees per performance as per rubric
    # Performance 1: Country artist, first weekend
    p1_parent = evaluator.add_parallel(
        id="Performance_1_Country_Artist_First_Weekend",
        desc="A country music artist headlining performance during the first festival weekend (June 18-20, 2026)",
        parent=summerfest_root,
        critical=False
    )
    await build_performance_verification(
        evaluator=evaluator,
        parent_node=p1_parent,
        perf=perf1,
        perf_label="P1",
        performance_node_desc="Performance 1 verification",
        weekend_range_str=W1_RANGE_STR,
        allowed_dates=W1_DATES,
        genre_requirement="country",
    )

    # Performance 2: Pop or Rock artist, second weekend
    p2_parent = evaluator.add_parallel(
        id="Performance_2_Pop_Rock_Artist_Second_Weekend",
        desc="A pop or rock artist headlining performance during the second festival weekend (June 25-27, 2026)",
        parent=summerfest_root,
        critical=False
    )
    await build_performance_verification(
        evaluator=evaluator,
        parent_node=p2_parent,
        perf=perf2,
        perf_label="P2",
        performance_node_desc="Performance 2 verification",
        weekend_range_str=W2_RANGE_STR,
        allowed_dates=W2_DATES,
        genre_requirement="pop_or_rock",
    )

    # Performance 3: Any genre, third weekend
    p3_parent = evaluator.add_parallel(
        id="Performance_3_Third_Weekend_Artist",
        desc="Any genre artist headlining performance during the third festival weekend (July 2-4, 2026)",
        parent=summerfest_root,
        critical=False
    )
    await build_performance_verification(
        evaluator=evaluator,
        parent_node=p3_parent,
        perf=perf3,
        perf_label="P3",
        performance_node_desc="Performance 3 verification",
        weekend_range_str=W3_RANGE_STR,
        allowed_dates=W3_DATES,
        genre_requirement="any",
    )

    # Return evaluation summary
    return evaluator.get_summary()