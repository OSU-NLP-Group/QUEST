import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dwts_ca_apr_2026"
TASK_DESCRIPTION = (
    "Identify four upcoming Dancing with the Stars: Live! tour performances taking place in California during April 2026. "
    "For each performance, provide the exact date, venue name, venue city, guest performers scheduled for that specific show, "
    "and a direct ticket purchase link. Also include the official tour schedule URL and a reference URL for each venue."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PerformanceItem(BaseModel):
    """One performance entry extracted from the answer."""
    date: Optional[str] = None
    venue_name: Optional[str] = None
    city: Optional[str] = None
    guest_performers: List[str] = Field(default_factory=list)
    ticket_url: Optional[str] = None
    venue_ref_url: Optional[str] = None


class DWTSExtraction(BaseModel):
    """Structured extraction of the DWTS California April 2026 performances."""
    schedule_url: Optional[str] = None
    performances: List[PerformanceItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_dwts_performances() -> str:
    return (
        "Extract information about Dancing with the Stars: Live! tour performances in California during April 2026 from the answer.\n\n"
        "You must return a JSON object with the following structure:\n"
        "{\n"
        '  "schedule_url": string | null,\n'
        '  "performances": [\n'
        "    {\n"
        '      "date": string | null,                // exact performance date as written (e.g., "April 12, 2026" or "2026-04-12")\n'
        '      "venue_name": string | null,          // venue name\n'
        '      "city": string | null,                // city name (e.g., "Los Angeles, CA" or "Los Angeles")\n'
        '      "guest_performers": string[] ,        // list of guest performer names for that specific show; empty list if not provided\n'
        '      "ticket_url": string | null,          // direct link to purchase tickets for that show\n'
        '      "venue_ref_url": string | null        // a reference URL for the venue’s information (official site or reputable page)\n'
        "    },\n"
        "    ... up to 4 entries ...\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Extract only URLs explicitly present in the answer (plain or markdown links). Do not invent URLs.\n"
        "- If more than four valid performances are present, include only the first four in the order they appear.\n"
        "- If fewer than four are present, include those available.\n"
        "- Do not add information not explicitly given in the answer.\n"
        "- For any missing field, return null (or empty array for guest_performers).\n"
        "- Leave dates as strings exactly as written in the answer (do not normalize).\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def clean_urls(urls: List[Optional[str]]) -> List[str]:
    """Filter out None/empty and deduplicate preserving order."""
    seen = set()
    result: List[str] = []
    for u in urls:
        if not u:
            continue
        u_str = u.strip()
        if not u_str:
            continue
        if u_str not in seen:
            seen.add(u_str)
            result.append(u_str)
    return result


def join_guest_names(names: List[str]) -> str:
    if not names:
        return ""
    return ", ".join([n.strip() for n in names if n and n.strip()])


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_schedule_reference(evaluator: Evaluator, root, schedule_url: Optional[str]) -> None:
    """
    Add nodes to verify the official tour schedule URL.
    """
    # Existence check (critical)
    evaluator.add_custom_node(
        result=bool(schedule_url and schedule_url.strip()),
        id="schedule_url_present",
        desc="Official tour schedule URL is provided",
        parent=root,
        critical=True
    )

    # Verify schedule URL describes official tour schedule (critical)
    schedule_leaf = evaluator.add_leaf(
        id="schedule_url_is_official",
        desc="The provided URL is the official Dancing with the Stars: Live! tour schedule page",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is the official schedule page for the Dancing with the Stars: Live! tour.",
        node=schedule_leaf,
        sources=schedule_url,
        additional_instruction=(
            "Confirm the page clearly lists tour dates/locations for Dancing with the Stars: Live!. "
            "It can be an official production page or the official tour site."
        ),
    )


async def verify_four_distinct(evaluator: Evaluator, root, performances: List[PerformanceItem]) -> None:
    """
    Global constraint: exactly four performances, and they are distinct (no duplicates).
    """
    four_node = evaluator.add_parallel(
        id="four_distinct_performances",
        desc="Provide four performances and ensure they are distinct (not duplicates).",
        parent=root,
        critical=True
    )

    # Exactly four provided
    evaluator.add_custom_node(
        result=(len(performances) == 4),
        id="exactly_four_provided",
        desc="Exactly four performances are provided",
        parent=four_node,
        critical=True
    )

    # Distinctness check (deduplicate by (date, venue_name, city))
    keys = []
    for p in performances[:4]:
        keys.append((p.date or "", p.venue_name or "", p.city or ""))

    no_dups = len(keys) == len(set(keys))

    evaluator.add_custom_node(
        result=no_dups,
        id="no_duplicate_performances",
        desc="All four performances are distinct (no duplicates by date+venue+city)",
        parent=four_node,
        critical=True
    )


async def verify_performance(
    evaluator: Evaluator,
    parent_node,
    perf: PerformanceItem,
    idx: int,
    schedule_url: Optional[str]
) -> None:
    """
    Verify the individual performance details according to rubric.
    Each performance subtree is critical: all required fields must be correct and grounded.
    """
    perf_node = evaluator.add_parallel(
        id=f"Performance_{idx+1}",
        desc=f"California performance #{idx+1} in April 2026 with all required fields.",
        parent=parent_node,
        critical=True
    )

    # Field existence gating (critical per field)
    date_present = evaluator.add_custom_node(
        result=bool(perf.date and perf.date.strip()),
        id=f"performance_{idx+1}_date_present",
        desc="Exact performance date is provided",
        parent=perf_node,
        critical=True
    )
    venue_present = evaluator.add_custom_node(
        result=bool(perf.venue_name and perf.venue_name.strip()),
        id=f"performance_{idx+1}_venue_present",
        desc="Venue name is provided",
        parent=perf_node,
        critical=True
    )
    city_present = evaluator.add_custom_node(
        result=bool(perf.city and perf.city.strip()),
        id=f"performance_{idx+1}_city_present",
        desc="Venue city is provided",
        parent=perf_node,
        critical=True
    )
    ticket_present = evaluator.add_custom_node(
        result=bool(perf.ticket_url and perf.ticket_url.strip()),
        id=f"performance_{idx+1}_ticket_present",
        desc="Direct ticket purchase link is provided",
        parent=perf_node,
        critical=True
    )
    venue_ref_present = evaluator.add_custom_node(
        result=bool(perf.venue_ref_url and perf.venue_ref_url.strip()),
        id=f"performance_{idx+1}_venue_ref_present",
        desc="Venue reference URL is provided",
        parent=perf_node,
        critical=True
    )
    guests_present = evaluator.add_custom_node(
        result=bool(perf.guest_performers and len([g for g in perf.guest_performers if g and g.strip()]) > 0),
        id=f"performance_{idx+1}_guests_present",
        desc="Guest performers are provided for this show",
        parent=perf_node,
        critical=True
    )

    # Build sources for verification
    common_sources = clean_urls([schedule_url, perf.ticket_url, perf.venue_ref_url])
    schedule_and_ticket = clean_urls([schedule_url, perf.ticket_url])
    venue_sources = clean_urls([perf.venue_ref_url, schedule_url, perf.ticket_url])

    # Leaf: performance date grounded by sources
    date_leaf = evaluator.add_leaf(
        id=f"performance_{idx+1}_date",
        desc="Provide the exact performance date.",
        parent=perf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The performance date for this show is '{perf.date or ''}'.",
        node=date_leaf,
        sources=schedule_and_ticket,
        additional_instruction=(
            "Check the event/ticket page and/or the official schedule to confirm the specific date string matches."
        )
    )

    # Leaf: date is in April 2026 (logical check)
    date_april_leaf = evaluator.add_leaf(
        id=f"performance_{idx+1}_date_in_april_2026",
        desc="The performance date occurs in April 2026.",
        parent=perf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The performance date '{perf.date or ''}' occurs in April 2026.",
        node=date_april_leaf,
        sources=None,
        additional_instruction=(
            "Judge purely by the provided date string: is it in April 2026? Allow common date formats."
        )
    )

    # Leaf: venue name grounded by sources
    venue_leaf = evaluator.add_leaf(
        id=f"performance_{idx+1}_venue_name",
        desc="Provide the venue name for this performance.",
        parent=perf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue for this performance is '{perf.venue_name or ''}'.",
        node=venue_leaf,
        sources=venue_sources,
        additional_instruction=(
            "Confirm the venue name as shown on the venue's page and/or the event ticket page or official schedule."
        )
    )

    # Leaf: venue city grounded by sources
    city_leaf = evaluator.add_leaf(
        id=f"performance_{idx+1}_venue_city",
        desc="Provide the city where the venue is located.",
        parent=perf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue is located in '{perf.city or ''}'.",
        node=city_leaf,
        sources=venue_sources,
        additional_instruction=(
            "Confirm the city/location on the venue page and/or event page."
        )
    )

    # Leaf: located in California grounded by sources
    ca_leaf = evaluator.add_leaf(
        id=f"performance_{idx+1}_located_in_california",
        desc="This performance is located in California.",
        parent=perf_node,
        critical=True
    )
    await evaluator.verify(
        claim="This performance (its venue) is located in California.",
        node=ca_leaf,
        sources=venue_sources,
        additional_instruction=(
            "Confirm the venue's state/region is California (CA). City pages often show the state. "
            "If city includes ', CA', that indicates California."
        )
    )

    # Leaf: guest performers grounded by sources
    guests_leaf = evaluator.add_leaf(
        id=f"performance_{idx+1}_guest_performers",
        desc="List the guest performer(s) scheduled to appear at this specific show.",
        parent=perf_node,
        critical=True
    )
    guests_text = join_guest_names(perf.guest_performers)
    await evaluator.verify(
        claim=f"The guest performers scheduled to appear at this show include: {guests_text}.",
        node=guests_leaf,
        sources=common_sources,
        additional_instruction=(
            "Verify the named guests are explicitly listed for this specific date/location on either the official tour schedule or the ticket/event page. "
            "Minor name formatting variations are acceptable."
        )
    )

    # Leaf: ticket link is a direct purchase page
    ticket_leaf = evaluator.add_leaf(
        id=f"performance_{idx+1}_ticket_link",
        desc="Provide a direct link to purchase tickets for this specific performance.",
        parent=perf_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"This URL is a direct ticket purchase page for the {perf.date or ''} show at {perf.venue_name or ''}."
        ),
        node=ticket_leaf,
        sources=perf.ticket_url,
        additional_instruction=(
            "Confirm the page is specifically for buying tickets (e.g., Ticketmaster/Eventbrite/venue ticketing) for the exact date/location."
        )
    )

    # Leaf: venue reference URL is a valid venue info page
    venue_ref_leaf = evaluator.add_leaf(
        id=f"performance_{idx+1}_venue_reference_url",
        desc="Provide a reference URL for this venue's information.",
        parent=perf_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This webpage provides official or reputable information about the venue '{perf.venue_name or ''}' in '{perf.city or ''}'.",
        node=venue_ref_leaf,
        sources=perf.venue_ref_url,
        additional_instruction=(
            "Accept the official venue site or a reputable venue profile page that lists address/city and basic info."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the DWTS California April 2026 performances task.
    Returns a structured summary including the verification tree and final score.
    """
    # Initialize evaluator with a parallel root (non-critical).
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

    # Extract structured data from the answer.
    extracted: DWTSExtraction = await evaluator.extract(
        prompt=prompt_extract_dwts_performances(),
        template_class=DWTSExtraction,
        extraction_name="dwts_ca_apr_2026"
    )

    # Record minimal ground truth requirements (contextual info only).
    evaluator.add_ground_truth({
        "requirements": [
            "Exactly four performances must be provided.",
            "All performances must be in California.",
            "All performances must occur in April 2026.",
            "Each performance must include: date, venue name, city, guest performers, ticket link, venue reference URL.",
            "Include the official tour schedule URL."
        ],
        "timeframe": "April 2026",
        "region": "California"
    })

    # Global: schedule reference verification
    await verify_schedule_reference(evaluator, root, extracted.schedule_url)

    # Global: ensure exactly four and distinct
    # Keep only the first four items (pad placeholders if fewer provided)
    performances = (extracted.performances or [])[:4]
    while len(performances) < 4:
        performances.append(PerformanceItem())

    await verify_four_distinct(evaluator, root, performances)

    # Verify each performance (each subtree is critical)
    for idx in range(4):
        await verify_performance(
            evaluator=evaluator,
            parent_node=root,
            perf=performances[idx],
            idx=idx,
            schedule_url=extracted.schedule_url
        )

    # Return the evaluation summary
    return evaluator.get_summary()