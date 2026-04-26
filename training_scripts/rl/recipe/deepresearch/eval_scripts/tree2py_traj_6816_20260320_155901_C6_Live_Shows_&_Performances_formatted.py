import asyncio
import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "lincoln_fin_2026_concerts"
TASK_DESCRIPTION = (
    "Find four stadium concerts scheduled at Lincoln Financial Field in Philadelphia, PA between July 1, 2026 and "
    "September 30, 2026. For each concert, provide: (1) The headlining artist name, (2) The official tour name, "
    "(3) The exact date of the concert (in YYYY-MM-DD format), (4) A complete list of all support acts/opening performers, "
    "(5) At least one official ticket sales platform where tickets can be purchased, and "
    "(6) A direct URL to the ticket purchasing page for that specific concert. "
    "Also provide reference URLs that confirm the basic concert information and support acts for each show."
)

DATE_RANGE_START = date(2026, 7, 1)
DATE_RANGE_END = date(2026, 9, 30)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class ConcertItem(BaseModel):
    venue: Optional[str] = None
    headliner: Optional[str] = None
    tour_name: Optional[str] = None
    # Expected format: YYYY-MM-DD (string). If not in that format in the answer, still extract as-is.
    date: Optional[str] = None

    # URLs that confirm the basic concert information (venue/headliner/tour/date)
    basic_info_urls: List[str] = Field(default_factory=list)

    # Support/opening acts and URLs that confirm them
    support_acts: List[str] = Field(default_factory=list)
    support_urls: List[str] = Field(default_factory=list)

    # Ticketing information: one or more official platforms and direct purchase URLs for this specific concert
    ticket_platforms: List[str] = Field(default_factory=list)
    ticket_urls: List[str] = Field(default_factory=list)


class ConcertExtraction(BaseModel):
    concerts: List[ConcertItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_concerts() -> str:
    return """
Extract up to four concerts listed in the answer that are described as taking place at Lincoln Financial Field in Philadelphia, PA between July 1, 2026 and September 30, 2026. Extract them in the order they appear in the answer and include at most the first four.

For each concert, extract the following fields:
- venue: The venue name as written (e.g., "Lincoln Financial Field").
- headliner: The headlining artist's name.
- tour_name: The official tour name as written on the cited page(s).
- date: The exact concert date string as provided in the answer (ideally 'YYYY-MM-DD'; do not reformat).
- basic_info_urls: All URLs cited that confirm the event’s basic information (venue, headliner, tour name, exact date). Prefer official sources (venue website, the artist’s official site, or official ticketing platforms).
- support_acts: A complete list of all support acts/opening performers as provided in the answer for that specific show/date; return an empty array if none are listed.
- support_urls: All URL(s) cited that confirm the support/opening acts for that specific show; return an empty array if not provided.
- ticket_platforms: At least one official ticket sales platform name, such as Ticketmaster, AXS, SeatGeek (primary), Live Nation, or the venue’s official ticketing, where tickets can be purchased. If multiple are listed, include all. If none, return an empty array.
- ticket_urls: One or more direct URL(s) to the ticket purchase page(s) for that specific concert/date (not a generic homepage). If none, return an empty array.

General rules:
- Extract only what is explicitly present in the answer; do not invent or infer missing data.
- Return null for missing scalar fields (e.g., venue/headliner/tour_name/date if absent).
- Return an empty array for missing lists (e.g., support_acts, basic_info_urls, support_urls, ticket_platforms, ticket_urls).
- Apply the URL extraction rules: only extract valid URLs explicitly present; handle markdown links; if protocol missing, prepend http://.
- If the answer includes more than four relevant concerts, only extract the first four.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _parse_date_str_to_date(d: Optional[str]) -> Optional[date]:
    if not d or not isinstance(d, str):
        return None
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%B %d, %Y",   # e.g., September 8, 2026
        "%b %d, %Y",   # e.g., Sep 8, 2026
        "%m/%d/%Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(d.strip(), fmt)
            return dt.date()
        except Exception:
            continue
    return None


def _date_in_window(d: Optional[str]) -> bool:
    parsed = _parse_date_str_to_date(d)
    if not parsed:
        return False
    return DATE_RANGE_START <= parsed <= DATE_RANGE_END


def _first_or_empty(items: List[str]) -> str:
    return items[0] if items else ""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_basic_info(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    concert: ConcertItem,
) -> None:
    # Create a critical parallel group for basic info
    basic_node = evaluator.add_parallel(
        id=f"concert_{idx}_basic_info",
        desc=f"Basic concert information including venue, headliner, tour name, and exact date for concert #{idx}",
        parent=parent_node,
        critical=True,
    )

    # Existence of at least one basic info URL (critical; gates the rest)
    url_exists_node = evaluator.add_custom_node(
        result=bool(concert.basic_info_urls),
        id=f"concert_{idx}_basic_info_url",
        desc="Provide a reference URL from an official venue, ticketing platform, or artist website that confirms the basic concert information",
        parent=basic_node,
        critical=True,
    )

    # Venue verification
    venue_leaf = evaluator.add_leaf(
        id=f"concert_{idx}_venue",
        desc="Verify the concert is at Lincoln Financial Field in Philadelphia, PA",
        parent=basic_node,
        critical=True,
    )
    headliner_for_context = concert.headliner or "the headlining artist"
    date_for_context = concert.date or "the specified date"
    venue_claim = (
        f"The concert featuring {headliner_for_context} on {date_for_context} is scheduled at Lincoln Financial Field in Philadelphia, PA."
    )
    await evaluator.verify(
        claim=venue_claim,
        node=venue_leaf,
        sources=concert.basic_info_urls,
        additional_instruction="Confirm that the event page explicitly lists the venue as 'Lincoln Financial Field' in Philadelphia, Pennsylvania (e.g., 'Philadelphia, PA' or 'Philadelphia, Pennsylvania'). Allow common formatting variants but require clear venue match.",
    )

    # Headliner verification
    headliner_leaf = evaluator.add_leaf(
        id=f"concert_{idx}_headliner",
        desc="Identify the headlining artist for this concert",
        parent=basic_node,
        critical=True,
    )
    headliner_claim = (
        f"The headlining artist for the show at Lincoln Financial Field on {date_for_context} is '{concert.headliner}'."
        if concert.headliner else
        "The headlining artist for the show at Lincoln Financial Field is explicitly identified."
    )
    await evaluator.verify(
        claim=headliner_claim,
        node=headliner_leaf,
        sources=concert.basic_info_urls,
        additional_instruction="Verify that the page clearly identifies the main/headlining artist for the specified event/date. Allow minor name variants or casing differences.",
    )

    # Tour name verification
    tour_leaf = evaluator.add_leaf(
        id=f"concert_{idx}_tour_name",
        desc="Provide the official tour name",
        parent=basic_node,
        critical=True,
    )
    tour_claim = (
        f"The event is part of the official '{concert.tour_name}' tour."
        if concert.tour_name else
        "The event page explicitly names the official tour for this concert."
    )
    await evaluator.verify(
        claim=tour_claim,
        node=tour_leaf,
        sources=concert.basic_info_urls,
        additional_instruction="Confirm the official tour name as shown on the event/artist/ticketing page. Allow slight punctuation/casing differences but require semantic equivalence.",
    )

    # Date supported by sources
    date_leaf = evaluator.add_leaf(
        id=f"concert_{idx}_date",
        desc="Provide the exact date in the format YYYY-MM-DD, which must fall between July 1, 2026 and September 30, 2026",
        parent=basic_node,
        critical=True,
    )
    date_claim = (
        f"The concert date is {concert.date} for the Lincoln Financial Field event."
        if concert.date else
        "The event page explicitly lists the concert date for the Lincoln Financial Field show."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=concert.basic_info_urls,
        additional_instruction="Verify the exact date for the Lincoln Financial Field event. If multiple dates are shown for different cities/venues, ensure the date corresponds to Lincoln Financial Field.",
    )

    # Programmatic check: date string parses to a date in the required window (critical)
    date_range_node = evaluator.add_custom_node(
        result=_date_in_window(concert.date),
        id=f"concert_{idx}_date_in_range",
        desc=f"Date '{concert.date}' is within 2026-07-01 to 2026-09-30 (inclusive) and is a valid date",
        parent=basic_node,
        critical=True,
    )


async def verify_support_acts(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    concert: ConcertItem,
) -> None:
    support_node = evaluator.add_parallel(
        id=f"concert_{idx}_support_acts",
        desc="Support acts and opening performers for this concert",
        parent=parent_node,
        critical=True,
    )

    # Presence of a support acts list (even if empty, it is a provided list)
    support_list_node = evaluator.add_custom_node(
        result=concert.support_acts is not None,
        id=f"concert_{idx}_support_list",
        desc="List all support acts scheduled to perform at this concert",
        parent=support_node,
        critical=True,
    )

    # Existence of at least one URL for support acts confirmation (critical gate)
    support_url_exists = evaluator.add_custom_node(
        result=bool(concert.support_urls) or bool(concert.basic_info_urls),
        id=f"concert_{idx}_support_url_exists",
        desc="At least one reference URL is provided to confirm support acts for this concert",
        parent=support_node,
        critical=True,
    )

    # Verification: the provided support acts are confirmed by the reference URLs
    support_verify_leaf = evaluator.add_leaf(
        id=f"concert_{idx}_support_url",
        desc="Provide a reference URL that confirms the support acts for this concert",
        parent=support_node,
        critical=True,
    )
    # Prefer dedicated support URLs; if none, fall back to basic info URLs (some official event pages list support acts)
    support_sources = concert.support_urls if concert.support_urls else concert.basic_info_urls
    if concert.support_acts:
        support_claim = (
            f"The following support/opening performers are listed for the Lincoln Financial Field show on {concert.date}: "
            f"{', '.join(concert.support_acts)}."
        )
    else:
        support_claim = (
            f"The reference page(s) indicate there are no listed support/opening acts for the Lincoln Financial Field show on {concert.date}."
        )
    await evaluator.verify(
        claim=support_claim,
        node=support_verify_leaf,
        sources=support_sources,
        additional_instruction="Confirm that the page(s) explicitly list the same set of support/opening performers for this exact show/date at Lincoln Financial Field. Ignore order and allow minor name variants.",
    )


async def verify_ticketing(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    concert: ConcertItem,
) -> None:
    ticket_node = evaluator.add_parallel(
        id=f"concert_{idx}_ticketing",
        desc="Ticket purchase information for this concert",
        parent=parent_node,
        critical=True,
    )

    # At least one official ticket platform named (critical)
    platform_exists = evaluator.add_custom_node(
        result=bool(concert.ticket_platforms),
        id=f"concert_{idx}_ticket_platform",
        desc="Identify at least one official ticket sales platform where tickets can be purchased",
        parent=ticket_node,
        critical=True,
    )

    # Existence of at least one direct ticket URL (critical gate)
    ticket_url_present = evaluator.add_custom_node(
        result=bool(concert.ticket_urls),
        id=f"concert_{idx}_ticket_url_present",
        desc="At least one direct URL to the ticket purchasing page is provided for this specific concert",
        parent=ticket_node,
        critical=True,
    )

    # Verify that at least one provided URL is a direct purchase page for this specific concert
    ticket_url_leaf = evaluator.add_leaf(
        id=f"concert_{idx}_ticket_url",
        desc="Provide a direct URL to the ticket purchasing page for this specific concert",
        parent=ticket_node,
        critical=True,
    )
    platform_for_context = _first_or_empty(concert.ticket_platforms) or "the official ticketing platform"
    ticket_claim = (
        f"The provided URL is a direct ticket purchasing page on {platform_for_context} for the concert featuring "
        f"{concert.headliner or 'the headlining artist'} at Lincoln Financial Field in Philadelphia, PA on {concert.date}."
    )
    await evaluator.verify(
        claim=ticket_claim,
        node=ticket_url_leaf,
        sources=concert.ticket_urls,
        additional_instruction=(
            "Verify that the page is a direct ticket purchase page (e.g., shows 'Buy Tickets', seat map, or checkout) "
            "specifically for the Lincoln Financial Field event on the stated date. "
            "Treat primary/official platforms as Ticketmaster, AXS, SeatGeek primary, Live Nation, or the venue's official ticketing. "
            "Do not count third-party resale-only marketplaces as official unless the venue or artist lists them as primary."
        ),
    )


async def verify_concert(
    evaluator: Evaluator,
    root_node,
    idx: int,
    concert: ConcertItem,
) -> None:
    """
    Build and run the verification subtree for a single concert.
    """
    concert_node = evaluator.add_parallel(
        id=f"concert_{idx}",
        desc=f"Concert #{idx} verification",
        parent=root_node,
        critical=False,  # Each concert contributes independently; allow partial completion across concerts
    )

    # Basic info checks (critical)
    await verify_basic_info(evaluator, concert_node, idx, concert)

    # Support acts checks (critical)
    await verify_support_acts(evaluator, concert_node, idx, concert)

    # Ticketing checks (critical)
    await verify_ticketing(evaluator, concert_node, idx, concert)


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
) -> Dict:
    """
    Evaluate an answer for the Lincoln Financial Field 2026 summer concerts task.
    """
    # Initialize evaluator (root is non-critical by design to allow partial credit across concerts)
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

    # Record the required date window as ground truth context
    evaluator.add_ground_truth(
        {
            "venue": "Lincoln Financial Field, Philadelphia, PA",
            "date_window_start": DATE_RANGE_START.isoformat(),
            "date_window_end": DATE_RANGE_END.isoformat(),
            "required_count": 4,
        },
        gt_type="constraints",
    )

    # Extract concerts
    extracted = await evaluator.extract(
        prompt=prompt_extract_concerts(),
        template_class=ConcertExtraction,
        extraction_name="concerts_extraction",
    )

    # Ensure exactly 4 slots by padding or truncating
    concerts: List[ConcertItem] = list(extracted.concerts[:4])
    while len(concerts) < 4:
        concerts.append(ConcertItem())

    # Build verification subtrees for each concert
    for i in range(1, 5):
        try:
            await verify_concert(evaluator, root, i, concerts[i - 1])
        except Exception as e:
            # In case of unexpected errors per concert, add a failed leaf to capture it
            err_node = evaluator.add_leaf(
                id=f"concert_{i}_internal_error",
                desc=f"Internal error while verifying concert #{i}: {e}",
                parent=root,
                critical=False,
                score=0.0,
                status="failed",
            )
            # No further action; continue with remaining concerts

    # Return final structured summary
    return evaluator.get_summary()