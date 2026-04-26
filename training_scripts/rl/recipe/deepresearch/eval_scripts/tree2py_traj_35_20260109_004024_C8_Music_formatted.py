import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_music_festivals_apr2025"
TASK_DESCRIPTION = (
    "Identify four multi-day music festivals taking place in the United States during April 2025. "
    "For each festival, provide the following information: (1) The festival name, "
    "(2) The specific venue or park where it takes place, (3) The U.S. state location, "
    "(4) The exact dates of the festival, (5) At least two headlining artists, and "
    "(6) The starting price for general admission tickets. Each festival must be at least 2 days long "
    "and must take place entirely or partially within April 2025."
)


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class FestivalItem(BaseModel):
    """One festival entry extracted from the answer."""
    name: Optional[str] = None
    venue_or_park: Optional[str] = None
    state: Optional[str] = None
    exact_dates: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    headliners: List[str] = Field(default_factory=list)
    ga_starting_price: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class FestivalsExtraction(BaseModel):
    """All festivals extracted from the answer."""
    festivals: List[FestivalItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_festivals() -> str:
    return """
    Extract up to ALL distinct music festival entries that the answer presents (do not invent new ones).
    For each festival, extract the following fields exactly as stated:
    - name: The festival name as written in the answer.
    - venue_or_park: The specific venue or park name (not just the city). If only a city is mentioned without a venue/park, still return what is stated (or null if missing).
    - state: The U.S. state where the festival occurs. Use the exact text from the answer (e.g., 'California' or 'CA').
    - exact_dates: The exact date range string as it appears (e.g., 'April 12–14, 2025', 'Mar 29–Apr 1, 2025', or 'two weekends in April 2025'). If multiple weekends/dates are given, include the text that best summarises the range(s).
    - start_date: If the answer gives an interpretable earliest calendar date for the 2025 edition, extract it as a text string (e.g., 'April 12, 2025'). Otherwise null.
    - end_date: If the answer gives an interpretable last calendar date for the 2025 edition, extract it as a text string (e.g., 'April 14, 2025'). Otherwise null.
    - headliners: A list of at least two headlining artists (as many as are listed in the answer). If fewer than two are in the answer, include what is provided (possibly fewer), or return an empty list if none.
    - ga_starting_price: The starting price text for general admission tickets (e.g., '$299', 'from $249 + fees'), as written. If missing, null.
    - source_urls: All URLs the answer cites that specifically support the festival’s details (official site, lineup page, ticket page, news articles, etc.). Extract only URLs explicitly present in the answer. If no URLs are given, return an empty list.

    Notes:
    - Do not infer or fabricate fields; if a field is missing in the answer, return null or an empty list as appropriate.
    - Keep all text strings exactly as they appear (do not normalize numbers or names).
    - Prefer URLs that directly support key facts (dates, location, headliners, prices).
    - The answer may contain more than four entries; still extract them all. The evaluator will later select the first four for scoring.
    """


# --------------------------------------------------------------------------- #
# Helper Functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_non_empty(text: Optional[str]) -> bool:
    return bool(text and str(text).strip())


def _has_two_headliners(headliners: List[str]) -> bool:
    clean = [h.strip() for h in headliners if _is_non_empty(h)]
    return len(clean) >= 2


def _collect_valid_names(items: List[FestivalItem]) -> List[str]:
    return [it.name for it in items if _is_non_empty(it.name)]


def _first_k(items: List[FestivalItem], k: int) -> List[FestivalItem]:
    result = items[:k]
    # Pad with empty entries if fewer than k
    while len(result) < k:
        result.append(FestivalItem())
    return result


# --------------------------------------------------------------------------- #
# Verification per Festival                                                   #
# --------------------------------------------------------------------------- #
async def verify_one_festival(
    evaluator: Evaluator,
    parent_node,
    fest: FestivalItem,
    idx_one_based: int,
) -> None:
    """
    Build verification subtree for a single festival.
    The structure follows the rubric: required fields (existence) and eligibility checks (verified with sources).
    """
    # Create festival parent node (non-critical to allow partial credit per festival)
    fest_node = evaluator.add_parallel(
        id=f"festival_{idx_one_based}",
        desc=f"Festival entry {idx_one_based} meets all criteria and includes all required fields",
        parent=parent_node,
        critical=False,
    )

    # ---------------------- Required Fields (existence) ---------------------- #
    req_node = evaluator.add_parallel(
        id=f"festival_{idx_one_based}_required_fields",
        desc=f"Festival {idx_one_based} includes all required output fields",
        parent=fest_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_non_empty(fest.name),
        id=f"festival_{idx_one_based}_name",
        desc=f"Festival {idx_one_based} name is provided",
        parent=req_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_non_empty(fest.venue_or_park),
        id=f"festival_{idx_one_based}_venue_or_park",
        desc=f"Festival {idx_one_based} specific venue or park name is provided (not just the city)",
        parent=req_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_non_empty(fest.state),
        id=f"festival_{idx_one_based}_state",
        desc=f"Festival {idx_one_based} U.S. state is specified",
        parent=req_node,
        critical=True,
    )

    # exact_dates presence: either exact_dates string or both start/end present
    has_dates_text = _is_non_empty(fest.exact_dates)
    has_start_end = _is_non_empty(fest.start_date) and _is_non_empty(fest.end_date)
    evaluator.add_custom_node(
        result=bool(has_dates_text or has_start_end),
        id=f"festival_{idx_one_based}_exact_dates",
        desc=f"Festival {idx_one_based} exact dates are provided",
        parent=req_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_two_headliners(fest.headliners),
        id=f"festival_{idx_one_based}_headliners",
        desc=f"At least two headlining artists for Festival {idx_one_based} are identified",
        parent=req_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_non_empty(fest.ga_starting_price),
        id=f"festival_{idx_one_based}_ga_starting_price",
        desc=f"Starting price for general admission tickets for Festival {idx_one_based} is provided",
        parent=req_node,
        critical=True,
    )

    # ---------------------- Eligibility (verified with sources) -------------- #
    elig_node = evaluator.add_parallel(
        id=f"festival_{idx_one_based}_eligibility",
        desc=f"Festival {idx_one_based} satisfies eligibility constraints",
        parent=fest_node,
        critical=True,
    )

    # Dates overlap with April 2025
    dates_overlap_leaf = evaluator.add_leaf(
        id=f"festival_{idx_one_based}_dates_overlap_april_2025",
        desc=f"Festival {idx_one_based} occurs entirely or partially within April 2025",
        parent=elig_node,
        critical=True,
    )
    overlap_claim = (
        f"The 2025 edition of the festival '{fest.name or 'UNKNOWN'}' takes place at least partially "
        f"in April 2025 (i.e., at least one festival date falls between April 1 and April 30, 2025)."
    )
    await evaluator.verify(
        claim=overlap_claim,
        node=dates_overlap_leaf,
        sources=fest.source_urls,
        additional_instruction=(
            "Use the provided URLs to check the official 2025 dates. Consider festivals spanning multiple "
            "weekends or crossing months; if any festival day is in April 2025, this condition is satisfied. "
            "Be careful about the year—confirm it's the 2025 edition."
        ),
    )

    # Multi-day (>= 2 days)
    multi_day_leaf = evaluator.add_leaf(
        id=f"festival_{idx_one_based}_multi_day",
        desc=f"Festival {idx_one_based} duration is at least 2 days",
        parent=elig_node,
        critical=True,
    )
    multi_day_claim = (
        f"The 2025 edition of the festival '{fest.name or 'UNKNOWN'}' lasts at least two calendar days "
        f"(e.g., two or more dates, whether consecutive or split across multiple weekends)."
    )
    await evaluator.verify(
        claim=multi_day_claim,
        node=multi_day_leaf,
        sources=fest.source_urls,
        additional_instruction=(
            "Verify from the festival schedule or announcement that there are at least two distinct calendar dates. "
            "If the event runs over two weekends (with multiple dates), it still counts as multi-day."
        ),
    )

    # US location (state is a US state)
    us_loc_leaf = evaluator.add_leaf(
        id=f"festival_{idx_one_based}_us_location",
        desc=f"Festival {idx_one_based} is located in the United States (e.g., state listed is a U.S. state)",
        parent=elig_node,
        critical=True,
    )
    state_text = fest.state or "UNKNOWN"
    us_loc_claim = (
        f"The festival '{fest.name or 'UNKNOWN'}' takes place in the United States; the state listed is '{state_text}'."
    )
    await evaluator.verify(
        claim=us_loc_claim,
        node=us_loc_leaf,
        sources=fest.source_urls,
        additional_instruction=(
            "Check that the location on the page is within the United States (e.g., a valid US state or "
            "standard US postal abbreviation like CA, NY, TX). If the URL content indicates a non-US country, "
            "the claim is not supported."
        ),
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry                                                       #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 'US multi-day music festivals in April 2025' task.
    Builds a verification tree according to the rubric and returns the evaluation summary.
    """
    # Initialize evaluator (root is non-critical to allow partial credit; set-level checks will be critical)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level: independent checks across festivals
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_festivals(),
        template_class=FestivalsExtraction,
        extraction_name="festivals_extraction",
    )

    # Select first 4 (pad if fewer)
    selected = _first_k(extracted.festivals, 4)

    # Record some helpful custom info
    evaluator.add_custom_info(
        info={
            "total_festivals_found_in_answer": len(extracted.festivals),
            "selected_for_evaluation": 4,
        },
        info_type="stats",
        info_name="extraction_summary",
    )

    # ---------------------- Set-level requirements (critical) ---------------- #
    set_level = evaluator.add_parallel(
        id="set_level_requirements",
        desc="Response includes the required number of distinct qualifying festivals",
        parent=root,
        critical=True,  # Essential gating: if not >=4 distinct entries, overall fails
    )

    # At least four festival entries (with names) are provided
    provided_names = [nm for nm in _collect_valid_names(selected) if _is_non_empty(nm)]
    evaluator.add_custom_node(
        result=len(provided_names) >= 4,
        id="at_least_four_festivals",
        desc="At least four festival entries are provided",
        parent=set_level,
        critical=True,
    )

    # Distinct festivals (by normalized names) among the first four
    normalized = [_normalize_name(nm) for nm in provided_names]
    unique_count = len(set(n for n in normalized if n))
    evaluator.add_custom_node(
        result=(len(provided_names) >= 4 and unique_count >= 4),
        id="festivals_are_distinct",
        desc="The festivals identified are distinct (not duplicates of the same festival)",
        parent=set_level,
        critical=True,
    )

    # ---------------------- Per-festival verification ------------------------ #
    # Build nodes for festival 1..4
    for i, fest in enumerate(selected, start=1):
        await verify_one_festival(evaluator, root, fest, i)

    # Return evaluation summary
    return evaluator.get_summary()