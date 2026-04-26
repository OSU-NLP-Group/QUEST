import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "houston_rodeo_2026_lineup"
TASK_DESCRIPTION = """
I'm planning to attend the Houston Rodeo in March 2026 and need detailed information about upcoming performances. Please find two different performances from the 2026 Houston Rodeo lineup and provide the following information for each:

1. Performer Name and Date: The name of the performer and their specific performance date in March 2026
2. Venue Details: Confirm that both performances take place at NRG Stadium in Houston, Texas, and provide the complete address of the venue
3. Venue Capacity: What is the seating capacity of NRG Stadium for events?
4. Ticket Information: For each performance, provide the starting ticket price and include a link to an official website where tickets can be purchased

Please ensure both performances are officially scheduled for the 2026 Houston Rodeo in March and provide accurate, verifiable information with reference URLs.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Performance(BaseModel):
    performer: Optional[str] = None
    date: Optional[str] = None
    event_urls: List[str] = Field(default_factory=list)       # URLs citing lineup/schedule/details
    purchase_urls: List[str] = Field(default_factory=list)    # URLs for purchasing tickets (e.g., rodeohouston.com or axs.com)


class RodeoExtraction(BaseModel):
    performances: List[Performance] = Field(default_factory=list)

    # Venue information (as provided in the answer)
    venue_address: Optional[str] = None
    venue_address_sources: List[str] = Field(default_factory=list)

    venue_capacity: Optional[str] = None
    venue_capacity_sources: List[str] = Field(default_factory=list)

    # Ticket price information (as provided in the answer)
    ticket_start_price: Optional[str] = None
    ticket_price_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_rodeo_info() -> str:
    return """
    Extract structured information about the Houston Rodeo 2026 from the provided answer.

    You must extract at most TWO performances (use the first two mentioned if more than two are present). For each performance, extract:
    - performer: The performer/artist name exactly as written in the answer.
    - date: The performance date exactly as written in the answer.
    - event_urls: A list of URLs cited in the answer that support the lineup/schedule details for this performance (e.g., RodeoHouston lineup page, announcement, news page). Include only URLs explicitly present in the answer.
    - purchase_urls: A list of URLs cited in the answer where tickets for this performance can be purchased (ideally rodeohouston.com or axs.com). Include only URLs explicitly present in the answer.

    Also extract venue and ticketing details mentioned in the answer (global information):
    - venue_address: The complete address string of NRG Stadium as stated in the answer (if provided). Example: "1 NRG Parkway, Houston, TX 77054".
    - venue_address_sources: All URLs provided in the answer that support the NRG Stadium address.
    - venue_capacity: The seating capacity of NRG Stadium for events, as stated in the answer (e.g., "72,000", "approximately 72,000").
    - venue_capacity_sources: All URLs provided in the answer that support the capacity value.
    - ticket_start_price: The starting ticket price for Houston Rodeo 2026 performances as stated in the answer (e.g., "$35 including a $5 convenience fee").
    - ticket_price_sources: All URLs provided in the answer that support the starting ticket price.

    Return a JSON object with these fields:
    {
      "performances": [
        {
          "performer": str | null,
          "date": str | null,
          "event_urls": str[],
          "purchase_urls": str[]
        },
        {
          "performer": str | null,
          "date": str | null,
          "event_urls": str[],
          "purchase_urls": str[]
        }
      ],
      "venue_address": str | null,
      "venue_address_sources": str[],
      "venue_capacity": str | null,
      "venue_capacity_sources": str[],
      "ticket_start_price": str | null,
      "ticket_price_sources": str[]
    }

    IMPORTANT:
    - Only include URLs that are explicitly present in the answer (including markdown links).
    - Do not fabricate or infer any URLs.
    - Preserve the exact strings (names, dates, address, price) as they appear in the answer.
    - If a field is not present in the answer, set it to null (for strings) or an empty list (for URL arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s) and bool(s.strip())


def _gather_sources(*url_lists: List[str]) -> List[str]:
    urls: List[str] = []
    for lst in url_lists:
        urls.extend(lst or [])
    # Remove duplicates while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_performance(
    evaluator: Evaluator,
    parent_node,
    perf: Performance,
    perf_index: int,
    prev_performer_name: Optional[str] = None
) -> None:
    """
    Build verification subtree for a single performance.

    perf_index: 0 for first performance, 1 for second performance
    """
    agg = evaluator.add_parallel(
        id=f"performance_{perf_index + 1}_complete",
        desc=(
            "First performance: Provide a valid performer officially scheduled for Houston Rodeo 2026 in March with the "
            "correct date, and confirm it takes place at NRG Stadium, Houston, Texas"
            if perf_index == 0 else
            "Second performance: Provide a different valid performer officially scheduled for Houston Rodeo 2026 in March "
            "with the correct date, and confirm it takes place at NRG Stadium, Houston, Texas (must be different from the first performance)"
        ),
        parent=parent_node,
        critical=False
    )

    # Existence and basic fields (critical gate for sub-checks in spirit)
    exists_node = evaluator.add_custom_node(
        result=(_non_empty_str(perf.performer) and _non_empty_str(perf.date) and
                (len(perf.event_urls) > 0 or len(perf.purchase_urls) > 0)),
        id=f"performance_{perf_index + 1}_has_required_fields",
        desc=f"Performance #{perf_index + 1} has performer, date, and at least one source URL",
        parent=agg,
        critical=True
    )

    # Date in March 2026 (logic check; no source needed)
    date_node = evaluator.add_leaf(
        id=f"performance_{perf_index + 1}_date_march_2026",
        desc=f"Performance #{perf_index + 1}: the provided date is in March 2026",
        parent=agg,
        critical=True
    )
    await evaluator.verify(
        claim=f"The date '{perf.date or ''}' falls in March 2026 (i.e., the month is March and the year is 2026).",
        node=date_node,
        additional_instruction="Accept common date formats (e.g., 'March 5, 2026', 'Mar 5, 2026', '03/05/2026')."
    )

    # Officially scheduled on the specified date as part of RodeoHouston 2026
    schedule_node = evaluator.add_leaf(
        id=f"performance_{perf_index + 1}_scheduled_rodeohouston",
        desc=f"Performance #{perf_index + 1}: performer is officially scheduled for RodeoHouston 2026 on the given date",
        parent=agg,
        critical=True
    )
    schedule_claim = (
        f"{perf.performer or 'The performer'} is officially scheduled to perform at the Houston Livestock Show and Rodeo "
        f"(RodeoHouston) in March 2026 on {perf.date or 'the specified date'}."
    )
    await evaluator.verify(
        claim=schedule_claim,
        node=schedule_node,
        sources=_gather_sources(perf.event_urls, perf.purchase_urls),
        additional_instruction=(
            "Verify on the provided page(s) that the performer is listed for the 2026 RodeoHouston lineup with the "
            "specified March 2026 date. Look for schedule/lineup pages or official announcements."
        )
    )

    # Venue must be NRG Stadium, Houston, Texas
    venue_node = evaluator.add_leaf(
        id=f"performance_{perf_index + 1}_venue_is_nrg",
        desc=f"Performance #{perf_index + 1}: takes place at NRG Stadium, Houston, Texas",
        parent=agg,
        critical=True
    )
    await evaluator.verify(
        claim="This performance takes place at NRG Stadium in Houston, Texas.",
        node=venue_node,
        sources=_gather_sources(perf.event_urls, perf.purchase_urls),
        additional_instruction=(
            "Confirm that the event page explicitly lists the venue as NRG Stadium (Houston, Texas). "
            "Allow minor formatting variants (e.g., 'NRG Stadium, Houston TX')."
        )
    )

    # For performance #2, ensure different performer than #1
    if perf_index == 1 and _non_empty_str(perf.performer) and _non_empty_str(prev_performer_name):
        diff_node = evaluator.add_leaf(
            id="performance_2_different_from_first",
            desc="Second performance: performer is different from the first performance",
            parent=agg,
            critical=True
        )
        await evaluator.verify(
            claim=f"The performer '{perf.performer}' is different from '{prev_performer_name}'.",
            node=diff_node,
            additional_instruction="Consider names equal if they clearly refer to the same artist; otherwise, they are different."
        )


async def verify_global_venue_info(
    evaluator: Evaluator,
    parent_node,
    extraction: RodeoExtraction
) -> None:
    # Venue Address (critical)
    addr_leaf = evaluator.add_leaf(
        id="venue_address",
        desc="Provide the complete address of NRG Stadium (1 NRG Parkway, Houston, TX 77054)",
        parent=parent_node,
        critical=True
    )
    venue_address_claim = (
        "The complete address of NRG Stadium is 1 NRG Parkway, Houston, TX 77054."
    )
    await evaluator.verify(
        claim=venue_address_claim,
        node=addr_leaf,
        sources=_gather_sources(extraction.venue_address_sources),
        additional_instruction="Verify on the cited page(s) that NRG Stadium's address matches exactly or reasonably (allow punctuation/formatting variations)."
    )

    # Venue Capacity (critical)
    cap_leaf = evaluator.add_leaf(
        id="venue_capacity",
        desc="Provide the seating capacity of NRG Stadium for events (approximately 72,000)",
        parent=parent_node,
        critical=True
    )
    capacity_claim = (
        "The seating capacity of NRG Stadium for events is approximately 72,000."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=cap_leaf,
        sources=_gather_sources(extraction.venue_capacity_sources),
        additional_instruction="Accept approximate phrasing around 72,000 (e.g., 72,220, ~72k)."
    )


async def verify_ticketing_info(
    evaluator: Evaluator,
    parent_node,
    extraction: RodeoExtraction
) -> None:
    # Ticket starting price (critical)
    price_leaf = evaluator.add_leaf(
        id="ticket_price_info",
        desc="Provide the starting ticket price for Houston Rodeo 2026 performances ($35 with $5 convenience fee included)",
        parent=parent_node,
        critical=True
    )
    price_claim = (
        "The starting ticket price for Houston Rodeo 2026 performances is $35, and this includes a $5 convenience fee."
    )
    await evaluator.verify(
        claim=price_claim,
        node=price_leaf,
        sources=_gather_sources(extraction.ticket_price_sources),
        additional_instruction=(
            "Confirm that the cited page states a starting price of $35 and indicates it includes a $5 convenience fee. "
            "If fees are listed separately and total differs, this claim is not supported."
        )
    )

    # Ticket purchase platform (critical) — verify provided purchase links are official purchase pages
    purchase_leaf = evaluator.add_leaf(
        id="ticket_purchase_platform",
        desc="Provide a link to an official website where tickets can be purchased (rodeohouston.com or AXS.com)",
        parent=parent_node,
        critical=True
    )

    # Collect all purchase URLs from both performances
    p0 = extraction.performances[0] if len(extraction.performances) > 0 else Performance()
    p1 = extraction.performances[1] if len(extraction.performances) > 1 else Performance()
    all_purchase_urls = _gather_sources(p0.purchase_urls, p1.purchase_urls)

    await evaluator.verify(
        claim="This page is an official ticket purchase page for a Houston Rodeo 2026 performance, hosted on rodeohouston.com or axs.com.",
        node=purchase_leaf,
        sources=all_purchase_urls,
        additional_instruction=(
            "The page should clearly facilitate purchasing tickets (e.g., 'Buy Tickets', seat selection) "
            "and be on rodeohouston.com or axs.com domains."
        )
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
    Evaluate an answer for the Houston Rodeo 2026 lineup task.
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

    # 1) Extraction
    extraction: RodeoExtraction = await evaluator.extract(
        prompt=prompt_extract_rodeo_info(),
        template_class=RodeoExtraction,
        extraction_name="rodeo_2026_info"
    )

    # Normalize to exactly 2 performances (pad with empty if fewer; trim if more)
    perfs: List[Performance] = list(extraction.performances[:2])
    while len(perfs) < 2:
        perfs.append(Performance())
    extraction.performances = perfs  # keep in recorded extractions too

    # 2) Build verification tree
    # Two performance blocks
    prev_name = perfs[0].performer if _non_empty_str(perfs[0].performer) else None
    await verify_performance(evaluator, root, perfs[0], perf_index=0, prev_performer_name=None)
    await verify_performance(evaluator, root, perfs[1], perf_index=1, prev_performer_name=prev_name)

    # Global venue info (address & capacity)
    await verify_global_venue_info(evaluator, root, extraction)

    # Ticketing information (starting price claim and official purchase platform)
    await verify_ticketing_info(evaluator, root, extraction)

    # 3) Return standardized summary
    return evaluator.get_summary()