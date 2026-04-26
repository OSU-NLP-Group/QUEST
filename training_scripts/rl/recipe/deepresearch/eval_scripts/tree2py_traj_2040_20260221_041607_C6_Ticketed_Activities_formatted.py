import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tpr_acdc_power_up_2026_july_states"
TASK_DESCRIPTION = (
    "The Pretty Reckless is supporting AC/DC on the Power Up Tour 2026 across U.S. stadiums. "
    "Identify three specific performances scheduled in July 2026 based on these criteria:\n\n"
    "1. The performance in Wisconsin\n"
    "2. The performance in Ohio\n"
    "3. The performance in North Carolina\n\n"
    "For each performance, provide:\n"
    "- The venue name\n"
    "- The exact date in July 2026\n"
    "- The official seating capacity\n\n"
    "Then, rank these three venues from smallest to largest by official seating capacity."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PerformanceItem(BaseModel):
    venue: Optional[str] = None
    date: Optional[str] = None  # Keep as string to allow flexible formats, e.g., "July 12, 2026"
    capacity: Optional[str] = None  # String to allow ranges/approximate formats
    venue_urls: List[str] = Field(default_factory=list)
    date_urls: List[str] = Field(default_factory=list)
    capacity_urls: List[str] = Field(default_factory=list)


class ShowsExtraction(BaseModel):
    wisconsin: Optional[PerformanceItem] = None
    ohio: Optional[PerformanceItem] = None
    north_carolina: Optional[PerformanceItem] = None
    ranking_order: List[str] = Field(default_factory=list)  # Ascending by capacity as stated in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_performances() -> str:
    return """
Extract the following structured information from the answer text exactly as presented.

You must extract three July 2026 performances for AC/DC's "Power Up Tour 2026" where The Pretty Reckless is supporting:
- One in the U.S. state of Wisconsin
- One in the U.S. state of Ohio
- One in the U.S. state of North Carolina

For each state, extract:
1) venue: The venue name (e.g., "Lambeau Field")
2) date: The exact date string in July 2026 as written in the answer (e.g., "July 12, 2026")
3) capacity: The official seating capacity for that venue as written in the answer (keep formatting as-is; do not convert)
4) venue_urls: All URL(s) explicitly cited in the answer that support the venue identification for that state
5) date_urls: All URL(s) explicitly cited in the answer that support the performance date for that state
6) capacity_urls: All URL(s) explicitly cited in the answer that support the venue’s official seating capacity for that state

Additionally extract:
- ranking_order: A list of exactly three strings representing the venues ranked from smallest to largest by official seating capacity, in the exact order presented in the answer. If the answer does not provide a ranking, return an empty list.

Rules:
- Only extract URLs that are explicitly present in the answer (including markdown links).
- If any field is missing, set it to null (for strings) or an empty list (for url lists).
- If multiple performances are mentioned for a state, choose the one explicitly scheduled in July 2026 as per the answer.

Return JSON with keys: "wisconsin", "ohio", "north_carolina", and "ranking_order".
Each state key should be an object with: venue, date, capacity, venue_urls, date_urls, capacity_urls.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_any_http_url(urls: List[str]) -> bool:
    return any(isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")) for u in urls)


def combine_capacity_urls(items: List[Optional[PerformanceItem]]) -> List[str]:
    urls: List[str] = []
    for item in items:
        if item and item.capacity_urls:
            urls.extend(item.capacity_urls)
    # Deduplicate while preserving order
    seen = set()
    uniq_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq_urls.append(u)
    return uniq_urls


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_performance_verification(
    evaluator: Evaluator,
    parent_node,
    state_key: str,
    state_name: str,
    item: Optional[PerformanceItem],
) -> None:
    """
    Build verification subtree for a single state's performance.

    Structure (all critical under this performance node):
      - {state}_venue (parallel)
          • {state}_venue_exists (custom)
          • {state}_venue_supported (leaf, verify by URLs)
          • {state}_venue_url (custom: at least one URL)
      - {state}_date (parallel)
          • {state}_date_exists (custom)
          • {state}_date_supported (leaf, verify by URLs)
          • {state}_date_url (custom)
      - {state}_capacity (parallel)
          • {state}_capacity_exists (custom)
          • {state}_capacity_supported (leaf, verify by URLs)
          • {state}_capacity_url (custom)
    """
    perf_node = evaluator.add_parallel(
        id=f"{state_key}_performance",
        desc=f"Identify the performance in {state_name} during July 2026",
        parent=parent_node,
        critical=True,  # Adjusted to satisfy critical-parent constraint
    )

    venue_val = item.venue if item else None
    date_val = item.date if item else None
    capacity_val = item.capacity if item else None

    venue_urls = item.venue_urls if item else []
    date_urls = item.date_urls if item else []
    capacity_urls = item.capacity_urls if item else []

    # 1) Venue block
    venue_block = evaluator.add_parallel(
        id=f"{state_key}_venue",
        desc=f"Provide venue name verifiable as hosting The Pretty Reckless in {state_name} in July 2026",
        parent=perf_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(venue_val and venue_val.strip()),
        id=f"{state_key}_venue_exists",
        desc=f"{state_name} venue is provided",
        parent=venue_block,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_any_http_url(venue_urls),
        id=f"{state_key}_venue_url",
        desc=f"URL reference supporting {state_name} venue exists",
        parent=venue_block,
        critical=True,
    )
    venue_supported_leaf = evaluator.add_leaf(
        id=f"{state_key}_venue_supported",
        desc=f"{state_name} venue is correctly identified and supported by cited sources",
        parent=venue_block,
        critical=True,
    )
    venue_claim = (
        f"According to the cited source(s), during July 2026 in {state_name}, "
        f"The Pretty Reckless is scheduled to support AC/DC on the Power Up Tour 2026 at the venue '{venue_val}'."
    )
    await evaluator.verify(
        claim=venue_claim,
        node=venue_supported_leaf,
        sources=venue_urls,
        additional_instruction=(
            "Confirm that the page(s) indicate AC/DC's 2026 Power Up Tour show in the specified state and that "
            "The Pretty Reckless is the supporting act at the named venue. Minor naming variations are acceptable."
        ),
    )

    # 2) Date block
    date_block = evaluator.add_parallel(
        id=f"{state_key}_date",
        desc=f"Provide date in July 2026 verifiable as performance date at the {state_name} venue",
        parent=perf_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(date_val and date_val.strip()),
        id=f"{state_key}_date_exists",
        desc=f"{state_name} date is provided",
        parent=date_block,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_any_http_url(date_urls),
        id=f"{state_key}_date_url",
        desc=f"URL reference supporting {state_name} date exists",
        parent=date_block,
        critical=True,
    )
    date_supported_leaf = evaluator.add_leaf(
        id=f"{state_key}_date_supported",
        desc=f"{state_name} performance date is correctly identified and supported by cited sources",
        parent=date_block,
        critical=True,
    )
    date_claim = (
        f"The performance in {state_name} at '{venue_val}' is scheduled on '{date_val}', and this date is in July 2026 "
        f"for AC/DC's Power Up Tour 2026 with The Pretty Reckless supporting."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_supported_leaf,
        sources=date_urls,
        additional_instruction=(
            "Verify that the cited source(s) explicitly list the performance date. Ensure the date falls in July 2026. "
            "Accept common date formats and minor formatting differences. The show context should match AC/DC's Power Up Tour 2026 "
            "with The Pretty Reckless as supporting act."
        ),
    )

    # 3) Capacity block
    capacity_block = evaluator.add_parallel(
        id=f"{state_key}_capacity",
        desc=f"Provide capacity matching official {state_name} venue capacity",
        parent=perf_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(capacity_val and capacity_val.strip()),
        id=f"{state_key}_capacity_exists",
        desc=f"{state_name} venue capacity is provided",
        parent=capacity_block,
        critical=True,
    )
    evaluator.add_custom_node(
        result=has_any_http_url(capacity_urls),
        id=f"{state_key}_capacity_url",
        desc=f"URL reference supporting {state_name} capacity exists",
        parent=capacity_block,
        critical=True,
    )
    capacity_supported_leaf = evaluator.add_leaf(
        id=f"{state_key}_capacity_supported",
        desc=f"{state_name} venue capacity is correctly identified and supported by cited sources",
        parent=capacity_block,
        critical=True,
    )
    capacity_claim = f"The official seating capacity of '{venue_val}' is '{capacity_val}'."
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_supported_leaf,
        sources=capacity_urls,
        additional_instruction=(
            "Check that the cited source(s) clearly state the venue's official seating capacity. "
            "Allow reasonable synonyms (e.g., 'seating capacity', 'capacity') and typical rounding/approximate wording. "
            "If multiple capacities are listed (e.g., for different configurations), prefer the general/official figure."
        ),
    )


async def build_capacity_ranking_verification(
    evaluator: Evaluator,
    parent_node,
    extracted: ShowsExtraction,
) -> None:
    """
    Build verification for capacity ranking (smallest -> largest).
    Creates:
      - ranking_provided (custom)
      - ranking_url (custom) – checks capacity URLs exist for all three
      - ranking_supported (leaf) – verify the stated order with capacity sources
    """
    ranking_node = evaluator.add_parallel(
        id="capacity_ranking",
        desc="Rank the three venues from smallest to largest by capacity",
        parent=parent_node,
        critical=True,  # ranking is critical per rubric
    )

    # Check that a ranking of 3 items is provided
    evaluator.add_custom_node(
        result=(len(extracted.ranking_order) == 3 and all(isinstance(x, str) and x.strip() for x in extracted.ranking_order)),
        id="ranking_provided",
        desc="Ranking of three venues is provided in the answer",
        parent=ranking_node,
        critical=True,
    )

    # Verify capacity URLs for all three venues exist
    wi = extracted.wisconsin
    oh = extracted.ohio
    nc = extracted.north_carolina
    all_three_have_capacity_urls = (
        has_any_http_url(wi.capacity_urls if wi else []) and
        has_any_http_url(oh.capacity_urls if oh else []) and
        has_any_http_url(nc.capacity_urls if nc else [])
    )
    evaluator.add_custom_node(
        result=all_three_have_capacity_urls,
        id="ranking_url",
        desc="URL references supporting the capacity values used for ranking exist for all three venues",
        parent=ranking_node,
        critical=True,
    )

    # Verify the stated ranking order using capacity sources
    ranking_supported_leaf = evaluator.add_leaf(
        id="ranking_supported",
        desc="The stated smallest-to-largest capacity order is supported by cited capacity sources",
        parent=ranking_node,
        critical=True,
    )

    if extracted.ranking_order:
        order = " < ".join([f"'{v}'" for v in extracted.ranking_order])
    else:
        order = "'', '', ''"  # placeholder if missing

    ranking_claim = f"Based on the official capacity sources, the smallest-to-largest venue order is: {order}."
    combined_capacity_urls = combine_capacity_urls([wi, oh, nc])

    await evaluator.verify(
        claim=ranking_claim,
        node=ranking_supported_leaf,
        sources=combined_capacity_urls,
        additional_instruction=(
            "Use the provided capacity source URLs to verify each venue's official seating capacity, then confirm the ascending order. "
            "Allow typical rounding differences. If two capacities are effectively equal or the pages cannot support the given numbers, "
            "the order should be considered not supported."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
) -> Dict[str, Any]:
    """
    Evaluation entry point for:
    'The Pretty Reckless supporting AC/DC Power Up Tour 2026 – July 2026 shows in WI, OH, NC + capacity ranking'
    """
    # Initialize evaluator (root sequential: identify performances first, then ranking)
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

    # Note: The rubric marks 'venue_identification' as critical with children originally non-critical.
    # To satisfy framework constraint that critical parent cannot have non-critical children,
    # we elevate each performance node to critical.
    evaluator.add_custom_info(
        info={
            "adjustments": [
                "Set each performance node (WI, OH, NC) to critical to satisfy critical-parent constraint.",
                "Added explicit existence checks for fields and URLs to enforce source-grounding prior to verification."
            ]
        },
        info_type="design_notes",
        info_name="criticality_adjustments"
    )

    # Extract structured information from the answer
    extracted: ShowsExtraction = await evaluator.extract(
        prompt=prompt_extract_performances(),
        template_class=ShowsExtraction,
        extraction_name="extracted_performances",
    )

    # Build 'venue_identification' stage (parallel, critical)
    identification_node = evaluator.add_parallel(
        id="venue_identification",
        desc="Identify all three required performances with complete information",
        parent=root,
        critical=True,
    )

    # Wisconsin
    await build_performance_verification(
        evaluator=evaluator,
        parent_node=identification_node,
        state_key="wi",
        state_name="Wisconsin",
        item=extracted.wisconsin,
    )

    # Ohio
    await build_performance_verification(
        evaluator=evaluator,
        parent_node=identification_node,
        state_key="oh",
        state_name="Ohio",
        item=extracted.ohio,
    )

    # North Carolina
    await build_performance_verification(
        evaluator=evaluator,
        parent_node=identification_node,
        state_key="nc",
        state_name="North Carolina",
        item=extracted.north_carolina,
    )

    # Capacity ranking stage (critical)
    await build_capacity_ranking_verification(
        evaluator=evaluator,
        parent_node=root,
        extracted=extracted,
    )

    # Return evaluation summary
    return evaluator.get_summary()