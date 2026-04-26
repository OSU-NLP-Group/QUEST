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
TASK_ID = "spring_2026_events"
TASK_DESCRIPTION = (
    "Find four different ticketed entertainment events scheduled in major US cities during Spring 2026 (March 1 - May 31, 2026). "
    "You must select exactly one event from each of the following four categories: (1) a live concert, (2) a Broadway or theater show, "
    "(3) a professional sporting event, and (4) a comedy show. For each event, provide the following information: event name and specific date, "
    "venue name and location (city and state), a valid URL link where tickets can be purchased, and a reference URL that verifies the event information. "
    "All events must have publicly available ticket purchasing information and must take place at established entertainment venues in major US cities."
)

DATE_RANGE_STR = "March 1, 2026 to May 31, 2026 (inclusive)"
MAJOR_US_CITIES_HINT = (
    "When judging whether the location is a major US city, consider widely recognized large metropolitan cities such as (non-exhaustive): "
    "New York, Los Angeles, Chicago, Houston, Phoenix, Philadelphia, San Antonio, San Diego, Dallas, San Jose, Austin, Jacksonville, Fort Worth, "
    "Columbus, Charlotte, San Francisco, Indianapolis, Seattle, Denver, Boston, Detroit, Nashville, Baltimore, Washington (DC), Portland, Las Vegas, "
    "Miami, Atlanta, Minneapolis, Sacramento, Tampa, Orlando, Cleveland, Cincinnati, Kansas City, St. Louis, New Orleans, Pittsburgh, Raleigh, "
    "Salt Lake City, Milwaukee, San Juan (PR), etc. Use common sense; well-known metro areas count."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    name: Optional[str] = None
    date: Optional[str] = None  # Keep as free-form string to be robust
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    ticket_url: Optional[str] = None
    reference_url: Optional[str] = None


class EventsExtraction(BaseModel):
    concert: Optional[EventItem] = None
    theater: Optional[EventItem] = None
    sporting: Optional[EventItem] = None
    comedy: Optional[EventItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
Extract exactly four events from the answer, one for each category:
- concert: a live concert event
- theater: a Broadway or theater show
- sporting: a professional sporting event (e.g., NBA/MLB/NHL/MLS/NFL preseason or other pro leagues, or major pro competitions)
- comedy: a stand-up comedy show or comedy event

For each category, extract the following fields as they appear in the answer:
- name: the event name (e.g., artist or show title; include teams for sporting)
- date: the specific event date
- venue_name: the name of the venue
- city: the city where the venue is located
- state: the US state (use standard 2-letter code or full name if provided)
- ticket_url: a valid URL where tickets for this specific event can be purchased (box office or legitimate ticketing platform)
- reference_url: a URL that verifies the event details (official site, venue calendar, reputable listing, or ticketing page)

Return a JSON object with four top-level fields: concert, theater, sporting, comedy.
If any field for a category is missing in the answer, set that field to null.
If an entire category is missing, set that category to null.
Do not invent information not present in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper verification builder                                                 #
# --------------------------------------------------------------------------- #
async def verify_event_category(
    evaluator: Evaluator,
    parent_node,
    event: Optional[EventItem],
    id_prefix: str,
    category_short_desc: str,
    name_date_desc: str,
    venue_desc: str,
    ticket_desc: str,
    reference_desc: str,
) -> None:
    """
    Build and verify the four critical leaves for a category:
    - Name & Date within Spring 2026 (verified by reference_url)
    - Venue & major US city (verified by reference_url)
    - Ticket link validity (verified by ticket_url)
    - Reference URL verifies event info (verified by reference_url)
    """
    # ---- Name & Date ----
    name_date_node = evaluator.add_leaf(
        id=f"{id_prefix}_Name_Date",
        desc=name_date_desc,
        parent=parent_node,
        critical=True,
    )
    if not event or not event.name or not event.date or not event.reference_url:
        name_date_node.score = 0.0
        name_date_node.status = "failed"
    else:
        claim_nd = (
            f"The webpage explicitly lists a {category_short_desc} named '{event.name}' scheduled on {event.date}. "
            f"The event date lies within {DATE_RANGE_STR}."
        )
        await evaluator.verify(
            claim=claim_nd,
            node=name_date_node,
            sources=event.reference_url,
            additional_instruction=(
                "Verify that the page clearly shows the event's name and an exact performance date, "
                f"and that the date falls between March 1, 2026 and May 31, 2026 (inclusive). "
                "Allow minor formatting differences in the event name (e.g., casing, punctuation) but it should be the same event."
            ),
        )

    # ---- Venue & Location ----
    venue_node = evaluator.add_leaf(
        id=f"{id_prefix}_Venue",
        desc=venue_desc,
        parent=parent_node,
        critical=True,
    )
    if not event or not event.venue_name or not event.city or not event.state or not event.reference_url:
        venue_node.score = 0.0
        venue_node.status = "failed"
    else:
        claim_venue = (
            f"The webpage shows that '{event.name}' takes place at '{event.venue_name}' in {event.city}, {event.state}, "
            "which is a major US city, and the venue is an established entertainment venue."
        )
        await evaluator.verify(
            claim=claim_venue,
            node=venue_node,
            sources=event.reference_url,
            additional_instruction=(
                "Confirm the venue name and its city/state from the page. "
                f"{MAJOR_US_CITIES_HINT} "
                "If the venue is a well-known theater, arena, stadium, or comedy club in a major metro area, consider it established. "
                "Allow if the metro area is clearly a major city even if the suburb is listed."
            ),
        )

    # ---- Ticket Link ----
    ticket_node = evaluator.add_leaf(
        id=f"{id_prefix}_Ticket_Link",
        desc=ticket_desc,
        parent=parent_node,
        critical=True,
    )
    if not event or not event.ticket_url or not event.name or not event.date:
        ticket_node.score = 0.0
        ticket_node.status = "failed"
    else:
        claim_ticket = (
            f"This URL is a legitimate page to purchase tickets for '{event.name}' on {event.date}"
            + (f" at '{event.venue_name}', {event.city}, {event.state}." if event.venue_name and event.city and event.state else ".")
        )
        await evaluator.verify(
            claim=claim_ticket,
            node=ticket_node,
            sources=event.ticket_url,
            additional_instruction=(
                "Check for clear purchase affordances such as 'Buy Tickets', 'Find Tickets', seat maps, cart/checkout, or pricing sections. "
                "Accept official venue box office pages and reputable platforms like Ticketmaster, AXS, SeatGeek, Eventbrite, etc. "
                "If the page is only an article or announcement without a purchase flow, do not consider it valid."
            ),
        )

    # ---- Reference URL ----
    ref_node = evaluator.add_leaf(
        id=f"{id_prefix}_Reference_URL",
        desc=reference_desc,
        parent=parent_node,
        critical=True,
    )
    if not event or not event.reference_url or not event.name or not event.date or not event.venue_name or not event.city or not event.state:
        ref_node.score = 0.0
        ref_node.status = "failed"
    else:
        claim_ref = (
            f"This webpage verifies the event details: '{event.name}' on {event.date} at '{event.venue_name}', "
            f"{event.city}, {event.state}."
        )
        await evaluator.verify(
            claim=claim_ref,
            node=ref_node,
            sources=event.reference_url,
            additional_instruction=(
                "Confirm that the page explicitly lists the event's name, date (within the Spring 2026 window), and venue/location. "
                "Minor name formatting differences are acceptable if they clearly refer to the same event."
            ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Spring 2026 events task.
    """
    # Initialize evaluator
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

    # Record auxiliary info
    evaluator.add_custom_info(
        {"date_window": DATE_RANGE_STR, "categories": ["concert", "theater", "sporting", "comedy"]},
        info_type="metadata",
        info_name="evaluation_parameters",
    )

    # Extract structured events
    events = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="extracted_events",
    )

    # Build category nodes under root (parallel, non-critical to allow partial credit)
    concert_node = evaluator.add_parallel(
        id="Concert_Event",
        desc="A live concert event scheduled in Spring 2026 at a major venue in a US city",
        parent=root,
        critical=False,
    )
    theater_node = evaluator.add_parallel(
        id="Theater_Event",
        desc="A Broadway or theater show event scheduled in Spring 2026",
        parent=root,
        critical=False,
    )
    sporting_node = evaluator.add_parallel(
        id="Sporting_Event",
        desc="A professional sporting event scheduled in Spring 2026",
        parent=root,
        critical=False,
    )
    comedy_node = evaluator.add_parallel(
        id="Comedy_Event",
        desc="A comedy show event scheduled in Spring 2026",
        parent=root,
        critical=False,
    )

    # Verify each category
    await verify_event_category(
        evaluator=evaluator,
        parent_node=concert_node,
        event=events.concert,
        id_prefix="Concert",
        category_short_desc="live concert",
        name_date_desc="The concert must have a specific event name and a confirmed date between March 1, 2026 and May 31, 2026",
        venue_desc="The concert must take place at a named venue with a verifiable location/address in a major US city",
        ticket_desc="A valid URL link to purchase tickets for the concert must be provided",
        reference_desc="A reference URL that verifies the concert information",
    )

    await verify_event_category(
        evaluator=evaluator,
        parent_node=theater_node,
        event=events.theater,
        id_prefix="Theater",
        category_short_desc="Broadway/theater show",
        name_date_desc="The show must have a specific name and confirmed performance dates within March 1 - May 31, 2026",
        venue_desc="The show must be at a named theater with a verifiable location in a major US city",
        ticket_desc="A valid URL link to purchase tickets for the theater show must be provided",
        reference_desc="A reference URL that verifies the theater show information",
    )

    await verify_event_category(
        evaluator=evaluator,
        parent_node=sporting_node,
        event=events.sporting,
        id_prefix="Sporting",
        category_short_desc="professional sporting event",
        name_date_desc="The sporting event must have a specific name (including teams or competition name) and a confirmed date between March 1 - May 31, 2026",
        venue_desc="The event must take place at a named sports venue (arena or stadium) with a verifiable location in a major US city",
        ticket_desc="A valid URL link to purchase tickets for the sporting event must be provided",
        reference_desc="A reference URL that verifies the sporting event information",
    )

    await verify_event_category(
        evaluator=evaluator,
        parent_node=comedy_node,
        event=events.comedy,
        id_prefix="Comedy",
        category_short_desc="comedy show",
        name_date_desc="The comedy show must have a specific name (including comedian name) and a confirmed date between March 1 - May 31, 2026",
        venue_desc="The show must take place at a named comedy club or entertainment venue with a verifiable location in a major US city",
        ticket_desc="A valid URL link to purchase tickets for the comedy show must be provided",
        reference_desc="A reference URL that verifies the comedy show information",
    )

    # Return the summary with the verification tree and computed scores
    return evaluator.get_summary()