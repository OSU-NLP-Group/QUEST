import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "summerfest_2026_schedule"
TASK_DESCRIPTION = (
    "I am planning to attend Summerfest 2026 in Milwaukee during the second weekend (June 25-27, 2026) and would like to see 4 different headliner performances. "
    "Help me identify 4 distinct headliners that meet all of the following requirements: "
    "(1) All 4 performers must headline at either the American Family Insurance Amphitheater or BMO Pavilion, "
    "(2) All 4 performances must take place during the second weekend (June 25, June 26, or June 27, 2026), "
    "(3) All 4 performances must have start times between 7:00 PM and 8:00 PM (inclusive), "
    "(4) At least one of the 4 performers must be from the country music genre, "
    "(5) At least one of the 4 performers must perform at BMO Pavilion, "
    "(6) The 4 performances must collectively span at least 2 different days within the second weekend, and "
    "(7) All 4 performers must be distinct individuals (no duplicate artists). "
    "For each of the 4 performers, provide their name, performance venue, date, start time, and a URL reference to verify their Summerfest 2026 performance details."
)


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class PerformanceItem(BaseModel):
    name: Optional[str] = None
    venue: Optional[str] = None
    date: Optional[str] = None  # Keep as free-form string (e.g., "June 25, 2026", "2026-06-25")
    start_time: Optional[str] = None  # Free-form (e.g., "7:30 PM")
    performance_url: Optional[str] = None  # Primary verification URL for Summerfest performance details
    additional_urls: List[str] = Field(default_factory=list)  # Any other URLs mentioned for this performer
    genres: List[str] = Field(default_factory=list)  # Any genres mentioned in the answer text for this performer


class ScheduleExtraction(BaseModel):
    items: List[PerformanceItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_schedule() -> str:
    return """
Extract up to 4 headliner performances mentioned in the answer for Summerfest 2026 that the user proposes. Preserve the original order from the answer and return at most 4 items. If more than 4 are mentioned, include only the first 4. If fewer than 4 are mentioned, include as many as available.

For each item, extract the following fields exactly as presented in the answer:
- name: The performer/artist name.
- venue: The stage/venue name for the performance (e.g., "American Family Insurance Amphitheater" or "BMO Pavilion"; do not normalize, extract verbatim from the answer).
- date: The performance date string as written (e.g., "June 25, 2026", "Thu, Jun 25", "2026-06-25", or similar). Do not reformat.
- start_time: The start time string as written (e.g., "7:30 PM", "7 PM", "8:00 pm").
- performance_url: A single URL explicitly provided in the answer that directly corresponds to the Summerfest 2026 performance details for this performer (e.g., Summerfest website page, official event listing, or ticketing page cited in the answer). If multiple such URLs are present, pick the most direct Summerfest/Event page. If no such URL is provided, set to null.
- additional_urls: Any other URLs explicitly cited in the answer for this performer (artist site, Wikipedia, news articles, etc.), excluding the chosen performance_url. If none, return an empty list.
- genres: A list of any genre terms for this performer that the answer explicitly mentions (e.g., ["country", "country pop"]). If the answer provides none, return an empty list.

Return a JSON object with a single field:
{
  "items": [ PerformanceItem, ... up to 4 ]
}

Special rules for URL extraction:
- Only extract URLs explicitly present in the answer. Do not invent or infer URLs.
- Normalize obviously incomplete URLs by prepending "http://" if missing a protocol.

If any specific field for an item is missing, set it to null (for strings) or an empty list (for arrays).
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return bool(re.match(r"^https?://", url.strip(), flags=re.IGNORECASE))


def _compile_sources(perf: PerformanceItem) -> List[str]:
    urls: List[str] = []
    if _valid_url(perf.performance_url):
        urls.append(perf.performance_url.strip())
    for u in perf.additional_urls:
        if _valid_url(u):
            if u.strip() not in urls:
                urls.append(u.strip())
    return urls


def _is_bmo_venue(venue: Optional[str]) -> bool:
    if not venue:
        return False
    v = venue.lower()
    # Allow common variants
    return "bmo" in v and "pavilion" in v


def _parse_day_from_date(date_str: Optional[str]) -> Optional[int]:
    """
    Attempt to parse the day of month for June 25, 26, or 27, 2026 from a free-form date string.
    Accepts formats like:
      - 2026-06-25, 2026/06/26, 2026.06.27
      - June 25, 2026 / Jun 26, 2026 / Thursday, June 27
      - 06/25/2026, 6/26/26, 06-27-2026, etc.
    Returns 25, 26, 27 if detected; otherwise None.
    """
    if not date_str:
        return None
    s = date_str.strip().lower()

    # ISO-like or numeric with separators and explicit month as 06
    m = re.search(r"2026[-/\.]0?6[-/\.](25|26|27)\b", s)
    if m:
        return int(m.group(1))

    # Month name based
    m = re.search(r"\b(june|jun)\s*(25|26|27)(?:\s*,?\s*2026)?\b", s)
    if m:
        return int(m.group(2))

    # Numeric month/day[/year]
    m = re.search(r"\b0?6[/-](25|26|27)(?:[/-](?:20)?26)?\b", s)
    if m:
        return int(m.group(1))

    # Day first variants like 25-06-2026 (less likely in US context, but handle)
    m = re.search(r"\b(25|26|27)[-/\.]0?6([-/\.](?:20)?26)?\b", s)
    if m:
        return int(m.group(1))

    return None


def _normalize_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    # Basic normalization for uniqueness checks
    return re.sub(r"\s+", " ", name).strip().lower()


# -----------------------------------------------------------------------------
# Verification subroutines
# -----------------------------------------------------------------------------
async def verify_performer(
    evaluator: Evaluator,
    parent_node,
    perf: PerformanceItem,
    idx1: int,
) -> Dict[str, Any]:
    """
    Build the per-performer verification subtree and run checks.
    Returns a dict with helpful flags for cross-constraints (e.g., is_bmo, parsed_day).
    """
    # Parent node for this performer (critical: all must pass under schedule_main)
    performer_node = evaluator.add_parallel(
        id=f"Performer_{idx1}",
        desc=f"Performer #{idx1} identified headliner meets all required per-item criteria",
        parent=parent_node,
        critical=True,
    )

    # Prepare sources (performance_url first, then additional_urls)
    sources = _compile_sources(perf)

    # 1) Reference existence (critical)
    ref_ok = _valid_url(perf.performance_url)
    evaluator.add_custom_node(
        result=ref_ok,
        id=f"Performer_{idx1}_Reference",
        desc=f"Valid URL reference provided for Performer {idx1}'s Summerfest 2026 performance details",
        parent=performer_node,
        critical=True,
    )

    # 2) Venue check (critical)
    venue_leaf = evaluator.add_leaf(
        id=f"Performer_{idx1}_Venue",
        desc=f"Performer {idx1} performs at either American Family Insurance Amphitheater or BMO Pavilion",
        parent=performer_node,
        critical=True,
    )
    # Build a tolerant claim; allow synonyms/abbreviations for venue names
    performer_name = perf.name or "(unknown performer)"
    claim_venue = (
        f"The provided webpage shows that {performer_name} will perform at Summerfest 2026 "
        f"at either the American Family Insurance Amphitheater or the BMO Pavilion."
    )
    await evaluator.verify(
        claim=claim_venue,
        node=venue_leaf,
        sources=sources if sources else None,
        additional_instruction=(
            "Verify on the page the stage/venue for this Summerfest 2026 performance. "
            "Accept reasonable naming variants/synonyms, e.g., 'American Family Insurance Amphitheatre', "
            "'American Family Insurance Amphitheater', 'AmFam Amphitheater', 'BMO Harris Pavilion', or 'BMO Pavilion'. "
            "If the page indicates a different stage/venue, mark as Incorrect."
        ),
    )

    # 3) Weekend (date) check (critical)
    weekend_leaf = evaluator.add_leaf(
        id=f"Performer_{idx1}_Weekend",
        desc=f"Performer {idx1} performs during the second weekend (June 25, 26, or 27, 2026)",
        parent=performer_node,
        critical=True,
    )
    claim_weekend = (
        "This Summerfest 2026 performance date is on June 25, 2026 or June 26, 2026 or June 27, 2026."
    )
    await evaluator.verify(
        claim=claim_weekend,
        node=weekend_leaf,
        sources=sources if sources else None,
        additional_instruction=(
            "Check the performance date on the page. Only accept if the date explicitly falls on "
            "June 25, 2026; June 26, 2026; or June 27, 2026."
        ),
    )

    # 4) Start time between 7:00 PM and 8:00 PM inclusive (critical)
    time_leaf = evaluator.add_leaf(
        id=f"Performer_{idx1}_Time",
        desc=f"Performer {idx1} has a start time between 7:00 PM and 8:00 PM inclusive",
        parent=performer_node,
        critical=True,
    )
    claim_time = (
        "The performance start time is between 7:00 PM and 8:00 PM inclusive."
    )
    await evaluator.verify(
        claim=claim_time,
        node=time_leaf,
        sources=sources if sources else None,
        additional_instruction=(
            "Verify the scheduled start time on the page (local time in Milwaukee, Central Time). "
            "Accept exactly 7:00 PM, any time after 7:00 PM up to and including 8:00 PM (e.g., 7:15 PM, 7:30 PM, 7:45 PM, 8:00 PM). "
            "If the listed time is earlier than 7:00 PM or later than 8:00 PM, mark as Incorrect."
        ),
    )

    return {
        "is_bmo": _is_bmo_venue(perf.venue),
        "day": _parse_day_from_date(perf.date),
        "name_norm": _normalize_name(perf.name),
        "sources": sources,
        "name": perf.name or "",
    }


async def verify_country_evidence_for_performer(
    evaluator: Evaluator,
    parent_node,
    perf: PerformanceItem,
    idx1: int,
):
    """
    Add a leaf node to verify whether this performer is a country artist, using any provided URLs.
    Returns the created leaf node (so caller can aggregate results).
    """
    node = evaluator.add_leaf(
        id=f"Performer_{idx1}_Country_Evidence",
        desc=f"Country genre evidence for Performer #{idx1}",
        parent=parent_node,
        critical=False,
    )
    performer_name = perf.name or "(unknown performer)"
    sources = _compile_sources(perf)
    claim_country = f"{performer_name} is a country music artist (including subgenres like country pop/country rock)."
    await evaluator.verify(
        claim=claim_country,
        node=node,
        sources=sources if sources else None,
        additional_instruction=(
            "Look for explicit genre information on the provided pages. "
            "Accept if the artist is described as 'country' or a clear country subgenre (e.g., 'country pop', 'country rock', 'alt-country'). "
            "If the pages do not indicate the artist as country-related, return Incorrect."
        ),
    )
    return node


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
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
    # Initialize evaluator/root
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

    # Extract up to 4 schedule items
    extracted = await evaluator.extract(
        prompt=prompt_extract_schedule(),
        template_class=ScheduleExtraction,
        extraction_name="summerfest_schedule",
    )
    items: List[PerformanceItem] = list(extracted.items[:4]) if extracted and extracted.items else []
    while len(items) < 4:
        items.append(PerformanceItem())

    # Create a critical aggregator node to gate the full personalized schedule validity (all must pass)
    schedule_main = evaluator.add_parallel(
        id="Summerfest_2026_Personalized_Schedule",
        desc="Identify 4 different Summerfest 2026 headliners for a personalized schedule during the second weekend (June 25-27), including required constraints",
        parent=root,
        critical=True,
    )

    # Build per-performer verification subtrees (critical under schedule_main)
    per_flags: List[Dict[str, Any]] = []
    for i in range(4):
        flags = await verify_performer(evaluator, schedule_main, items[i], i + 1)
        per_flags.append(flags)

    # Add a separate (non-critical) evidence group to show country checks for transparency
    country_evidence_group = evaluator.add_parallel(
        id="Country_Artist_Evidence",
        desc="Per-performer genre checks (country) for transparency",
        parent=root,
        critical=False,
    )
    country_leaves = []
    for i in range(4):
        node = await verify_country_evidence_for_performer(evaluator, country_evidence_group, items[i], i + 1)
        country_leaves.append(node)

    # Compute cross-item constraints and add as critical custom nodes under schedule_main

    # 1) Country artist requirement: at least one of the 4 is a country artist
    at_least_one_country = any(n.status == "passed" for n in country_leaves)
    evaluator.add_custom_node(
        result=at_least_one_country,
        id="Country_Artist_Requirement",
        desc="At least one of the 4 performers is from the country music genre",
        parent=schedule_main,
        critical=True,
    )

    # 2) At least one BMO Pavilion performer
    at_least_one_bmo = any(f["is_bmo"] for f in per_flags)
    evaluator.add_custom_node(
        result=at_least_one_bmo,
        id="BMO_Pavilion_Requirement",
        desc="At least one of the 4 performers performs at BMO Pavilion",
        parent=schedule_main,
        critical=True,
    )

    # 3) Multi-day requirement: collectively span at least 2 different days (25/26/27)
    days = [f["day"] for f in per_flags if f["day"] in (25, 26, 27)]
    unique_days = sorted(set(days))
    multi_day_ok = len(unique_days) >= 2
    evaluator.add_custom_node(
        result=multi_day_ok,
        id="Multi_Day_Requirement",
        desc="The 4 performers collectively span at least 2 different days within the second weekend",
        parent=schedule_main,
        critical=True,
    )

    # 4) No duplicates: all 4 performers are distinct individuals
    names_norm = [f["name_norm"] for f in per_flags if f["name_norm"]]
    no_duplicates = (len(names_norm) == 4) and (len(set(names_norm)) == 4)
    evaluator.add_custom_node(
        result=no_duplicates,
        id="No_Duplicates",
        desc="All 4 identified performers are distinct individuals (no duplicate artists)",
        parent=schedule_main,
        critical=True,
    )

    # Provide some helpful custom info for debugging
    evaluator.add_custom_info(
        info={
            "parsed_days": unique_days,
            "at_least_one_bmo": at_least_one_bmo,
            "at_least_one_country": at_least_one_country,
            "distinct_names_count": len(set(names_norm)),
            "performer_names": [f["name"] for f in per_flags],
        },
        info_type="diagnostics",
        info_name="cross_constraints_debug",
    )

    # Return evaluation summary
    return evaluator.get_summary()