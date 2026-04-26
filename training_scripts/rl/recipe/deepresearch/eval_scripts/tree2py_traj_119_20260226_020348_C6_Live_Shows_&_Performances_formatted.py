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
TASK_ID = "broadway_musical_constraints_2026"
TASK_DESCRIPTION = (
    "Identify a Broadway musical that meets all of the following criteria: "
    "(1) The show must have achieved more than 11,000 performances on Broadway as of February 2026; "
    "(2) The show must be currently running on Broadway (not closed); "
    "(3) The show must rank among the top 3 longest-running currently active Broadway productions; "
    "(4) The production must be a musical (not a play); "
    "(5) The show must be performed in a Broadway theater with a seating capacity of at least 1,700 seats; "
    "(6) The show's original Broadway opening must have occurred between 1995 and 2000 (inclusive); "
    "(7) The show must have received Tony Award recognition (nomination or win) in major categories during its original Broadway run. "
    "For your answer, provide: (1) The name of the show; (2) The specific Broadway theater where it currently performs and that theater's seating capacity; "
    "(3) The total number of performances as documented in official Broadway records; (4) The original Broadway opening date; "
    "(5) The Tony Awards won by the show; (6) Reference URLs supporting each piece of information."
)

AS_OF_DATE = "February 2026"
MIN_THEATER_CAPACITY = 1700
OPENING_YEAR_START = 1995
OPENING_YEAR_END = 2000

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ShowExtraction(BaseModel):
    # Core fields extracted from the answer (prefer strings for flexibility)
    show_name: Optional[str] = None

    # Current theatre info and capacity
    current_theater_name: Optional[str] = None
    theater_capacity: Optional[str] = None
    theater_capacity_urls: List[str] = Field(default_factory=list)

    # Performance count and sources
    performance_count: Optional[str] = None
    performance_count_as_of: Optional[str] = None
    performance_count_urls: List[str] = Field(default_factory=list)

    # Currently running status
    currently_running_urls: List[str] = Field(default_factory=list)

    # Top-3 currently active ranking
    top3_active_urls: List[str] = Field(default_factory=list)

    # Type confirmation (musical vs play)
    is_musical_urls: List[str] = Field(default_factory=list)

    # Opening date between 1995 and 2000 inclusive
    opening_date: Optional[str] = None
    opening_date_urls: List[str] = Field(default_factory=list)

    # Tony recognition (major categories) and awards won
    tony_recognition_urls: List[str] = Field(default_factory=list)
    tony_awards_won: List[str] = Field(default_factory=list)
    tony_awards_won_urls: List[str] = Field(default_factory=list)

    # All references mentioned in the answer (if provided)
    all_reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_show_info() -> str:
    return """
    Extract the structured information for the single Broadway show proposed in the answer. Return JSON with these fields:

    Required core field:
    - show_name: the exact name of the show specified in the answer.

    Current theatre and capacity:
    - current_theater_name: the specific Broadway theatre where the show currently performs, exactly as written in the answer.
    - theater_capacity: the theatre's seating capacity as stated in the answer (keep any separators or qualifiers as-is).
    - theater_capacity_urls: array of URLs cited in the answer that support the theatre and/or its capacity.

    Performance count:
    - performance_count: the total number of Broadway performances as stated in the answer (string; keep any separators or qualifiers as-is).
    - performance_count_as_of: the "as of" timing or date if included in the answer (e.g., "as of February 2026"); otherwise null.
    - performance_count_urls: array of URLs cited that support the performance count (prefer official records such as IBDB/Broadway League; include what the answer cited).

    Currently running status:
    - currently_running_urls: array of URLs cited in the answer that support that the show is currently running on Broadway.

    Top-3 currently active ranking:
    - top3_active_urls: array of URLs cited in the answer that support the claim that the show is among the top 3 longest-running currently active Broadway productions.

    Show type:
    - is_musical_urls: array of URLs cited that confirm the production is a musical (not a play).

    Opening date:
    - opening_date: the original Broadway opening date string as written in the answer.
    - opening_date_urls: array of URLs cited that support the original Broadway opening date.

    Tony Awards:
    - tony_recognition_urls: array of URLs cited that support Tony Award recognition (nominations or wins) in major categories during the original Broadway run.
    - tony_awards_won: list of Tony Awards that the answer claims the show won (each item as a free-form string as written).
    - tony_awards_won_urls: array of URLs cited that support the list of Tony Awards won.

    General:
    - all_reference_urls: array of ALL URLs mentioned anywhere in the answer (including all of the above and any other references), in full URL form.

    Rules:
    - Only extract data explicitly present in the answer text.
    - For any missing field, set it to null (for strings) or [] (for lists).
    - For URLs, extract actual URLs even if embedded in markdown links. If protocol missing, prepend http:// as needed.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_merge_url_lists(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst:
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def add_show_name_check(evaluator: Evaluator, root_node, data: ShowExtraction) -> None:
    evaluator.add_custom_node(
        result=bool(data.show_name and data.show_name.strip()),
        id="show_name_provided",
        desc="Answer provides a specific show name (not a vague description).",
        parent=root_node,
        critical=True
    )


async def add_currently_running_checks(evaluator: Evaluator, parent_node, data: ShowExtraction) -> None:
    wrapper = evaluator.add_sequential(
        id="currently_running_with_citation",
        desc="States the show is currently running on Broadway (not closed) and provides a supporting URL.",
        parent=parent_node,
        critical=True
    )

    # Source presence
    evaluator.add_custom_node(
        result=len(data.currently_running_urls) > 0,
        id="currently_running_sources_present",
        desc="At least one supporting URL is provided for 'currently running' status.",
        parent=wrapper,
        critical=True
    )

    # Verification by URL(s)
    leaf = evaluator.add_leaf(
        id="currently_running_supported",
        desc="Show is currently running on Broadway (not closed), supported by the cited URL(s).",
        parent=wrapper,
        critical=True
    )

    show_name = data.show_name or "the show"
    claim = f"The show '{show_name}' is currently running on Broadway (not closed) as of {AS_OF_DATE}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.currently_running_urls,
        additional_instruction=(
            "Confirm that the show is active on Broadway now (e.g., 'Open Run', active listings, "
            "upcoming performances, or official show site indicating ongoing performances). "
            "If the page explicitly indicates the show has closed, this claim is false."
        )
    )


async def add_performance_count_checks(evaluator: Evaluator, parent_node, data: ShowExtraction) -> None:
    wrapper = evaluator.add_sequential(
        id="performance_count_threshold_with_citation",
        desc="Provides the total Broadway performance count; the count is > 11,000 as of February 2026; and provides a supporting URL documenting the count and timeframe (official Broadway records or equivalent authoritative source).",
        parent=parent_node,
        critical=True
    )

    # Presence checks
    evaluator.add_custom_node(
        result=len(data.performance_count_urls) > 0,
        id="performance_count_sources_present",
        desc="At least one supporting URL is provided for the performance count.",
        parent=wrapper,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(data.performance_count and data.performance_count.strip()),
        id="performance_count_value_provided",
        desc="The answer provides a total Broadway performance count value.",
        parent=wrapper,
        critical=True
    )

    # Verification by URL(s)
    leaf = evaluator.add_leaf(
        id="performance_count_threshold_supported",
        desc="The performance count is > 11,000 as of February 2026, supported by the cited URL(s).",
        parent=wrapper,
        critical=True
    )
    show_name = data.show_name or "the show"
    count_str = data.performance_count or "(not specified)"
    as_of_str = data.performance_count_as_of or AS_OF_DATE
    claim = (
        f"The total Broadway performance count for '{show_name}' is '{count_str}', "
        f"and this implies the show has exceeded 11,000 total performances as of {as_of_str}."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.performance_count_urls,
        additional_instruction=(
            "Use the cited page(s) to determine whether the total number of Broadway performances exceeds 11,000 "
            f"by {AS_OF_DATE}. Prefer official records (e.g., The Broadway League/IBDB). "
            "If the page shows a figure below the threshold or clearly outdated without justification, fail."
        )
    )


async def add_top3_checks(evaluator: Evaluator, parent_node, data: ShowExtraction) -> None:
    wrapper = evaluator.add_sequential(
        id="top3_currently_active_longevity_with_citation",
        desc="Establishes the show is among the top 3 longest-running currently active Broadway productions and provides a supporting URL for this ranking.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(data.top3_active_urls) > 0,
        id="top3_sources_present",
        desc="At least one supporting URL is provided for the 'top 3 currently active' ranking.",
        parent=wrapper,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="top3_currently_active_supported",
        desc="Show is among the top 3 longest-running currently active Broadway productions, supported by the cited URL(s).",
        parent=wrapper,
        critical=True
    )
    show_name = data.show_name or "the show"
    claim = f"The show '{show_name}' is among the top 3 longest-running currently active Broadway productions."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.top3_active_urls,
        additional_instruction=(
            "Verify from the cited ranking/list page(s) that the show is top 3 among Broadway productions that are currently running, "
            "not including closed shows. If the page ranks all-time but includes closed shows, ensure the filtered 'currently running' "
            "subset still places the show in the top three."
        )
    )


async def add_is_musical_checks(evaluator: Evaluator, parent_node, data: ShowExtraction) -> None:
    wrapper = evaluator.add_sequential(
        id="is_musical_not_play_with_citation",
        desc="Confirms the production is a musical (not a play) and provides a supporting URL.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(data.is_musical_urls) > 0,
        id="is_musical_sources_present",
        desc="At least one supporting URL is provided to confirm the production type.",
        parent=wrapper,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="is_musical_supported",
        desc="The production is a musical (not a play), supported by the cited URL(s).",
        parent=wrapper,
        critical=True
    )
    show_name = data.show_name or "the show"
    claim = f"The production '{show_name}' is a musical (not a straight play)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.is_musical_urls,
        additional_instruction=(
            "Use the cited page(s) to confirm the work is categorized as a musical. "
            "If the page explicitly categorizes it as a play (non-musical), fail."
        )
    )


async def add_theater_capacity_checks(evaluator: Evaluator, parent_node, data: ShowExtraction) -> None:
    wrapper = evaluator.add_sequential(
        id="current_theater_and_capacity_with_citation",
        desc="Provides the specific current Broadway theater name and seating capacity; capacity is ≥ 1,700; and provides a supporting URL for the theater and capacity.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(bool(data.current_theater_name and data.current_theater_name.strip())
                and len(data.theater_capacity_urls) > 0
                and bool(data.theater_capacity and data.theater_capacity.strip())),
        id="theater_and_capacity_provided_with_sources",
        desc="The answer provides current theater name, a capacity value, and supporting URL(s).",
        parent=wrapper,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="theater_and_capacity_threshold_supported",
        desc="The show currently performs at the named theater and that theater has capacity ≥ 1,700, supported by the cited URL(s).",
        parent=wrapper,
        critical=True
    )
    show_name = data.show_name or "the show"
    theater_name = data.current_theater_name or "(unspecified theater)"
    capacity_text = data.theater_capacity or "(unspecified capacity)"
    claim = (
        f"The show '{show_name}' currently performs at the '{theater_name}' Broadway theatre, and that theatre has "
        f"a seating capacity of at least {MIN_THEATER_CAPACITY} (capacity cited as '{capacity_text}')."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.theater_capacity_urls,
        additional_instruction=(
            "Verify BOTH: (1) the show is currently housed at the named Broadway theater, and (2) that theater's seating capacity "
            f"is at least {MIN_THEATER_CAPACITY}. Use authoritative theatre pages (official theatre, Playbill venue profile) or other credible sources. "
            "If capacity listed is below threshold, fail."
        )
    )


async def add_opening_date_checks(evaluator: Evaluator, parent_node, data: ShowExtraction) -> None:
    wrapper = evaluator.add_sequential(
        id="opening_date_range_with_citation",
        desc="Provides the original Broadway opening date; date is between 1995 and 2000 inclusive; and provides a supporting URL.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(len(data.opening_date_urls) > 0 and bool(data.opening_date and data.opening_date.strip())),
        id="opening_date_value_and_sources_present",
        desc="The answer provides an original Broadway opening date and supporting URL(s).",
        parent=wrapper,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="opening_date_range_supported",
        desc="The original Broadway opening date is between 1995 and 2000 inclusive, supported by the cited URL(s).",
        parent=wrapper,
        critical=True
    )
    show_name = data.show_name or "the show"
    opening_text = data.opening_date or "(unspecified opening date)"
    claim = (
        f"The original Broadway opening date for '{show_name}' is '{opening_text}', and this date falls between "
        f"{OPENING_YEAR_START}-01-01 and {OPENING_YEAR_END}-12-31 inclusive."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.opening_date_urls,
        additional_instruction=(
            "Verify the ORIGINAL Broadway opening date from authoritative sources (e.g., IBDB/The Broadway League, "
            "Playbill show page, Wikipedia with citations). Ensure the interpreted date lies within 1995–2000 inclusive."
        )
    )


async def add_tony_recognition_checks(evaluator: Evaluator, parent_node, data: ShowExtraction) -> None:
    wrapper = evaluator.add_sequential(
        id="tony_major_category_recognition_with_citation",
        desc="Confirms the show received Tony Award recognition (nomination or win) in major categories during its original Broadway run and provides a supporting URL.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(data.tony_recognition_urls) > 0,
        id="tony_recognition_sources_present",
        desc="At least one supporting URL is provided for Tony Award recognition in major categories.",
        parent=wrapper,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="tony_major_category_recognition_supported",
        desc="Tony Award recognition (nomination or win) in major categories during original Broadway run is supported by cited URL(s).",
        parent=wrapper,
        critical=True
    )
    show_name = data.show_name or "the show"
    claim = (
        f"The show '{show_name}' received Tony Award recognition (nominations or wins) in major categories during its original Broadway run."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.tony_recognition_urls,
        additional_instruction=(
            "Major categories include Best Musical/Best Revival of a Musical, Best Leading/Featured Actor/Actress, "
            "Best Direction, Best Book, Best Score, Best Choreography, etc. Recognition must be during the original Broadway run. "
            "Use Tony Awards official site, Playbill, IBDB, or equivalently authoritative sources."
        )
    )


async def add_tony_wins_checks(evaluator: Evaluator, parent_node, data: ShowExtraction) -> None:
    wrapper = evaluator.add_sequential(
        id="tony_awards_won_listed_with_citation",
        desc="Provides the Tony Awards won by the show (as requested) and provides supporting URL(s).",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(len(data.tony_awards_won) > 0 and len(data.tony_awards_won_urls) > 0),
        id="tony_awards_won_values_and_sources_present",
        desc="The answer lists Tony Awards won and provides supporting URL(s).",
        parent=wrapper,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="tony_awards_won_supported",
        desc="The Tony Awards claimed as won are supported by the cited URL(s).",
        parent=wrapper,
        critical=True
    )
    show_name = data.show_name or "the show"
    awards_list = data.tony_awards_won if data.tony_awards_won else []
    claim = f"The show '{show_name}' won the following Tony Awards: {awards_list}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.tony_awards_won_urls,
        additional_instruction=(
            "Confirm that each listed award is a Tony Award that the show actually won (not just a nomination). "
            "Minor phrasing differences are acceptable (e.g., 'Best Musical' vs. 'Tony Award for Best Musical')."
        )
    )


async def add_authoritative_sources_check(evaluator: Evaluator, parent_node, data: ShowExtraction) -> None:
    # Combine URLs that support required facts
    combined_urls = _unique_merge_url_lists(
        data.currently_running_urls,
        data.performance_count_urls,
        data.top3_active_urls,
        data.is_musical_urls,
        data.theater_capacity_urls,
        data.opening_date_urls,
        data.tony_recognition_urls,
        data.tony_awards_won_urls,
    )
    # If the answer also provided a separate all_reference_urls list, include it for completeness
    if data.all_reference_urls:
        combined_urls = _unique_merge_url_lists(combined_urls, data.all_reference_urls)

    # Create the critical leaf as specified
    node = evaluator.add_leaf(
        id="all_citations_are_authoritative_sources",
        desc="All URLs used to support required facts are from authoritative sources (e.g., Playbill, The Broadway League/IBDB, official show/theater websites, or Wikipedia), consistent with the stated source constraint.",
        parent=parent_node,
        critical=True
    )

    # Build a reasoning-focused simple verification (no direct page-content grounding needed)
    urls_preview = "; ".join(combined_urls[:12])  # preview limited to keep prompt concise
    claim = (
        "All of the cited URLs used to support the required facts come from authoritative sources "
        "(e.g., Playbill, IBDB/The Broadway League, Tony Awards official site, official show/theatre websites, or Wikipedia). "
        f"Here is a representative list of the cited URLs: {urls_preview}"
    )

    await evaluator.verify(
        claim=claim,
        node=node,
        sources=None,
        additional_instruction=(
            "Use your general knowledge and domain heuristics to judge whether the cited domains are authoritative for Broadway information. "
            "Authoritative examples: playbill.com, ibdb.com (The Broadway League), tonyawards.com, official show sites, official theatre sites, and Wikipedia.org. "
            "If most or all of the supporting URLs come from these or equivalently authoritative domains, pass; otherwise fail."
        )
    )

    # Record combined URLs to the summary for transparency
    evaluator.add_custom_info(
        info={"combined_fact_support_urls": combined_urls},
        info_type="urls",
        info_name="combined_fact_support_urls"
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Broadway musical constraints task.
    """
    # Initialize evaluator with sequential aggregation at root (per rubric)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # 1) Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_show_info(),
        template_class=ShowExtraction,
        extraction_name="show_extraction"
    )

    # 2) Tree: Step 1 — Show name must be provided (critical)
    await add_show_name_check(evaluator, root, extracted)

    # 3) Tree: Step 2 — All constraints and required outputs (critical, parallel group)
    constraints_node = evaluator.add_parallel(
        id="all_constraints_and_required_outputs_met",
        desc="The identified show meets every constraint and the answer includes the requested information with supporting URLs.",
        parent=root,
        critical=True
    )

    # Add each constraint subtree (each internally handles its own source existence + verification)
    await add_currently_running_checks(evaluator, constraints_node, extracted)
    await add_performance_count_checks(evaluator, constraints_node, extracted)
    await add_top3_checks(evaluator, constraints_node, extracted)
    await add_is_musical_checks(evaluator, constraints_node, extracted)
    await add_theater_capacity_checks(evaluator, constraints_node, extracted)
    await add_opening_date_checks(evaluator, constraints_node, extracted)
    await add_tony_recognition_checks(evaluator, constraints_node, extracted)
    await add_tony_wins_checks(evaluator, constraints_node, extracted)
    await add_authoritative_sources_check(evaluator, constraints_node, extracted)

    # 4) Return summary
    return evaluator.get_summary()