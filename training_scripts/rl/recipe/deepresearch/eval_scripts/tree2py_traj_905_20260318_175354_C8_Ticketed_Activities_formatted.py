import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_events_2026_unique_categories"
TASK_DESCRIPTION = (
    "Identify four distinct ticketed entertainment events scheduled in the United States during 2026, "
    "where each event must be from a different category and held in a different U.S. state. The four categories are: "
    "(1) a multi-day music festival held at an outdoor festival ground or beach park that accommodates at least 20,000 "
    "daily attendees and features multiple stages, occurring in Spring (March-May) or Summer (June-August); "
    "(2) a concert residency with at least 2 scheduled shows at a single venue; "
    "(3) an arena event (sporting or concert) at a venue with capacity between 14,000-20,000; and "
    "(4) a Broadway or theater show at a venue with at least 500 seats. "
    "For each event, provide the venue name, complete address, event dates, and a supporting reference URL."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Address(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    full: Optional[str] = None  # The complete address string as written in the answer (optional)


class MusicFestival(BaseModel):
    festival_name: Optional[str] = None
    venue_name: Optional[str] = None
    venue_type: Optional[str] = None  # e.g., "outdoor festival grounds", "beach park"
    multi_stage: Optional[bool] = None
    capacity_text: Optional[str] = None  # any phrasing about capacity; can include numbers
    start_date: Optional[str] = None     # try to use YYYY-MM-DD if possible
    end_date: Optional[str] = None       # try to use YYYY-MM-DD if possible
    dates_text: Optional[str] = None     # any additional date phrasing, e.g., "June 6–8, 2026"
    season_label: Optional[str] = None   # optional label like "Summer 2026"
    address: Optional[Address] = None
    ticket_price: Optional[str] = None   # GA multi-day pass price range
    reference_urls: List[str] = Field(default_factory=list)


class ConcertResidency(BaseModel):
    artist: Optional[str] = None
    venue_name: Optional[str] = None
    scheduled_dates: List[str] = Field(default_factory=list)  # aim for YYYY-MM-DD, else any date text
    address: Optional[Address] = None
    ticket_info: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class ArenaEvent(BaseModel):
    event_name: Optional[str] = None  # description like "X vs Y" or artist name for arena concert
    venue_name: Optional[str] = None
    capacity_text: Optional[str] = None  # any phrasing that can help verify 14k–20k
    event_date: Optional[str] = None  # try YYYY-MM-DD
    address: Optional[Address] = None
    ticket_info: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class TheaterShow(BaseModel):
    show_name: Optional[str] = None
    theater_name: Optional[str] = None
    capacity_text: Optional[str] = None  # any phrasing that can help verify ≥500 seats
    scheduled_dates: List[str] = Field(default_factory=list)  # one or multiple dates/range entries
    address: Optional[Address] = None
    ticket_info: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class EventsExtraction(BaseModel):
    music_festival: Optional[MusicFestival] = None
    residency: Optional[ConcertResidency] = None
    arena_event: Optional[ArenaEvent] = None
    theater_show: Optional[TheaterShow] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
Extract the four required 2026 U.S. events from the answer, one per category. Return a JSON object matching the schema below. Rules:
- Extract ONLY what appears in the answer text. Do not invent details.
- Where dates are provided, prefer normalized YYYY-MM-DD if possible; otherwise keep the original text.
- For addresses, extract individual fields if present; also include a `full` address string if the answer wrote one.
- For each event, collect one or more reference URLs mentioned in the answer; keep them in `reference_urls` (array). If none are provided, keep it empty.
- If a field is missing in the answer, set it to null (or [] for arrays).

Schema:
{
  "music_festival": {
    "festival_name": str|null,
    "venue_name": str|null,
    "venue_type": str|null,
    "multi_stage": bool|null,
    "capacity_text": str|null,
    "start_date": str|null,   // prefer YYYY-MM-DD if available
    "end_date": str|null,     // prefer YYYY-MM-DD if available
    "dates_text": str|null,   // free-form date text if used in the answer
    "season_label": str|null, // e.g., "Summer 2026"
    "address": {
      "street": str|null, "city": str|null, "state": str|null, "zip": str|null, "full": str|null
    }|null,
    "ticket_price": str|null, // GA multi-day pass price/range if present
    "reference_urls": [str, ...] // URLs explicitly present in the answer
  },
  "residency": {
    "artist": str|null,
    "venue_name": str|null,
    "scheduled_dates": [str, ...], // each an explicit date string mentioned
    "address": {
      "street": str|null, "city": str|null, "state": str|null, "zip": str|null, "full": str|null
    }|null,
    "ticket_info": str|null,
    "reference_urls": [str, ...]
  },
  "arena_event": {
    "event_name": str|null,
    "venue_name": str|null,
    "capacity_text": str|null,
    "event_date": str|null, // prefer YYYY-MM-DD if available
    "address": {
      "street": str|null, "city": str|null, "state": str|null, "zip": str|null, "full": str|null
    }|null,
    "ticket_info": str|null,
    "reference_urls": [str, ...]
  },
  "theater_show": {
    "show_name": str|null,
    "theater_name": str|null,
    "capacity_text": str|null,
    "scheduled_dates": [str, ...],
    "address": {
      "street": str|null, "city": str|null, "state": str|null, "zip": str|null, "full": str|null
    }|null,
    "ticket_info": str|null,
    "reference_urls": [str, ...]
  }
}
    """.strip()


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def address_is_complete(addr: Optional[Address]) -> bool:
    return bool(addr and addr.street and addr.city and addr.state and addr.zip)


def dates_str_list(dates: Optional[List[str]]) -> str:
    if not dates:
        return "[]"
    return "; ".join(dates)


def nonempty_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def safe(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_music_festival(evaluator: Evaluator, parent) -> None:
    mf: MusicFestival = getattr(evaluator, "_mf_data", None)  # set by main
    mf_node = evaluator.add_parallel(
        id="event_1_music_festival",
        desc="Identify a multi-day music festival in 2026",
        parent=parent,
        critical=False
    )

    refs = nonempty_urls(mf.reference_urls if mf else [])

    # festival_venue_type (critical, URL-grounded)
    node = evaluator.add_leaf(
        id="festival_venue_type",
        desc="The festival must be held at an outdoor festival ground or beach park venue",
        parent=mf_node,
        critical=True
    )
    claim = (
        f"The music festival at venue '{safe(mf.venue_name)}' is held at an outdoor festival ground or a beach park."
        " Accept equivalent phrasing implying open-air festival grounds or a beach park."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=refs,
        additional_instruction="Verify the venue classification using the provided page(s). Allow reasonable synonyms like 'outdoor festival grounds', 'beach park', or 'open-air festival park'."
    )

    # festival_multi_stage (critical, URL-grounded)
    node = evaluator.add_leaf(
        id="festival_multi_stage",
        desc="The festival must feature multiple stages for performances",
        parent=mf_node,
        critical=True
    )
    claim = "The festival features multiple stages (more than one performance stage)."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=refs,
        additional_instruction="Look for explicit mentions such as 'multiple stages', 'Stage A and Stage B', or stage listings."
    )

    # festival_capacity (critical, URL-grounded)
    node = evaluator.add_leaf(
        id="festival_capacity",
        desc="The festival venue must accommodate at least 20,000 daily attendees",
        parent=mf_node,
        critical=True
    )
    claim = "The festival/venue accommodates at least 20,000 daily attendees (20,000 or more)."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=refs,
        additional_instruction="Allow phrasing like '20,000+', 'over 20,000', or any clearly ≥ 20,000 daily capacity statement."
    )

    # festival_duration (critical, logic check using extracted dates)
    node = evaluator.add_leaf(
        id="festival_duration",
        desc="The festival must be a multi-day event spanning at least 2 consecutive days",
        parent=mf_node,
        critical=True
    )
    claim = (
        f"Based on the provided dates for the festival, start_date='{safe(mf.start_date)}' and end_date='{safe(mf.end_date)}', "
        f"the event spans at least two consecutive days."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Treat any date range covering two or more consecutive days as satisfying the condition. If only free-text dates were provided, reason from that text; if dates are missing, mark incorrect."
    )

    # festival_season_and_year (critical, logic check)
    node = evaluator.add_leaf(
        id="festival_season_and_year",
        desc="The festival must take place during Spring (March-May) or Summer (June-August) 2026",
        parent=mf_node,
        critical=True
    )
    season_str = mf.season_label or ""
    claim = (
        f"The festival takes place during Spring (March–May) or Summer (June–August) in calendar year 2026, "
        f"based on start_date='{safe(mf.start_date)}', end_date='{safe(mf.end_date)}', dates_text='{safe(mf.dates_text)}', season_label='{season_str}'."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Check months are between March and August inclusive, and the year is 2026. If only a range like 'June 6–8, 2026' is present, infer the months and year."
    )

    # festival_specific_dates (critical, URL-grounded)
    node = evaluator.add_leaf(
        id="festival_specific_dates",
        desc="Provide the specific dates when the festival takes place",
        parent=mf_node,
        critical=True
    )
    dates_repr = safe(mf.dates_text) if mf and mf.dates_text else f"{safe(mf.start_date)} to {safe(mf.end_date)}"
    claim = f"The 2026 festival is scheduled on these dates: {dates_repr}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=refs,
        additional_instruction="Minor variations in date formatting are acceptable. The page should clearly indicate the same dates/range for the 2026 edition."
    )

    # festival_venue_name (critical, presence check in the answer)
    result = bool(mf and mf.venue_name and mf.venue_name.strip())
    evaluator.add_custom_node(
        result=result,
        id="festival_venue_name",
        desc="Provide the specific festival grounds or park name",
        parent=mf_node,
        critical=True
    )

    # festival_complete_address (critical, presence/completeness in the answer)
    result = address_is_complete(mf.address if mf else None)
    evaluator.add_custom_node(
        result=result,
        id="festival_complete_address",
        desc="Provide the complete venue address including street, city, state, and ZIP code",
        parent=mf_node,
        critical=True
    )

    # festival_ticket_price (non-critical; URL-grounded if present, else fail softly)
    node = evaluator.add_leaf(
        id="festival_ticket_price",
        desc="Provide the general admission multi-day pass price range",
        parent=mf_node,
        critical=False
    )
    if mf and mf.ticket_price:
        claim = f"The general admission multi-day pass price/range for the festival is: {mf.ticket_price}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=refs,
            additional_instruction="Match the gist of pricing/range; minor formatting differences are acceptable."
        )
    else:
        # No price provided in the answer -> fail non-critical leaf
        await evaluator.verify(
            claim="A general admission multi-day pass price/range is provided in the answer.",
            node=node,
            additional_instruction="Mark incorrect if the answer did not provide any ticket price/range."
        )

    # festival_reference (critical; presence of at least one URL in the answer)
    evaluator.add_custom_node(
        result=len(refs) > 0,
        id="festival_reference",
        desc="Provide a URL reference supporting all festival information",
        parent=mf_node,
        critical=True
    )


async def verify_residency(evaluator: Evaluator, parent) -> None:
    mf: MusicFestival = getattr(evaluator, "_mf_data", None)
    res: ConcertResidency = getattr(evaluator, "_res_data", None)
    res_node = evaluator.add_parallel(
        id="event_2_concert_residency",
        desc="Identify a concert residency or recurring performance series in 2026",
        parent=parent,
        critical=False
    )

    refs = nonempty_urls(res.reference_urls if res else [])

    # residency_venue_name (critical; presence)
    evaluator.add_custom_node(
        result=bool(res and res.venue_name and res.venue_name.strip()),
        id="residency_venue_name",
        desc="Provide the specific venue name where the residency takes place",
        parent=res_node,
        critical=True
    )

    # residency_artist (critical; presence)
    evaluator.add_custom_node(
        result=bool(res and res.artist and res.artist.strip()),
        id="residency_artist",
        desc="Identify the performing artist or group for the residency",
        parent=res_node,
        critical=True
    )

    # residency_show_count (critical; at least 2 scheduled dates)
    evaluator.add_custom_node(
        result=bool(res and len(res.scheduled_dates) >= 2),
        id="residency_show_count",
        desc="The residency must include at least 2 scheduled performance dates in 2026",
        parent=res_node,
        critical=True
    )

    # residency_specific_dates (critical; URL-grounded)
    node = evaluator.add_leaf(
        id="residency_specific_dates",
        desc="Provide the specific dates of the residency shows",
        parent=res_node,
        critical=True
    )
    claim = (
        f"The residency by '{safe(res.artist)}' at '{safe(res.venue_name)}' in 2026 is scheduled on the following dates: "
        f"{dates_str_list(res.scheduled_dates if res else [])}."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=refs,
        additional_instruction="It is acceptable if the page lists the same dates with slightly different formatting; at least two of the listed dates must appear."
    )

    # residency_year (critical; logic check that all dates are in 2026)
    node = evaluator.add_leaf(
        id="residency_year",
        desc="The residency shows must take place during calendar year 2026",
        parent=res_node,
        critical=True
    )
    claim = (
        f"All the following listed residency dates occur in calendar year 2026: {dates_str_list(res.scheduled_dates if res else [])}."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="If any date is outside 2026 or dates are missing, mark incorrect."
    )

    # residency_complete_address (critical; presence/completeness)
    evaluator.add_custom_node(
        result=address_is_complete(res.address if res else None),
        id="residency_complete_address",
        desc="Provide the complete venue address including street, city, state, and ZIP code",
        parent=res_node,
        critical=True
    )

    # residency_state_uniqueness (critical; different from music festival state)
    node = evaluator.add_leaf(
        id="residency_state_uniqueness",
        desc="The residency must be in a different U.S. state than the music festival",
        parent=res_node,
        critical=True
    )
    claim = (
        f"The residency state '{safe(res.address.state if res and res.address else None)}' is different from "
        f"the music festival state '{safe(mf.address.state if mf and mf.address else None)}'."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Treat standard postal abbreviations and full state names as equivalent when comparing."
    )

    # residency_ticket_info (non-critical; URL-grounded if present)
    node = evaluator.add_leaf(
        id="residency_ticket_info",
        desc="Provide information about ticket availability or pricing",
        parent=res_node,
        critical=False
    )
    if res and res.ticket_info:
        claim = f"Ticket availability/pricing information for this residency is: {res.ticket_info}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=refs,
            additional_instruction="Confirm the gist of availability/pricing on the page(s)."
        )
    else:
        await evaluator.verify(
            claim="The answer provides ticket availability or pricing for this residency.",
            node=node,
            additional_instruction="Mark incorrect if the answer did not provide any ticket info."
        )

    # residency_reference (critical; presence)
    evaluator.add_custom_node(
        result=len(refs) > 0,
        id="residency_reference",
        desc="Provide a URL reference supporting all residency information",
        parent=res_node,
        critical=True
    )


async def verify_arena(evaluator: Evaluator, parent) -> None:
    mf: MusicFestival = getattr(evaluator, "_mf_data", None)
    res: ConcertResidency = getattr(evaluator, "_res_data", None)
    ar: ArenaEvent = getattr(evaluator, "_arena_data", None)

    ar_node = evaluator.add_parallel(
        id="event_3_arena_event",
        desc="Identify an arena sporting event or major arena concert in 2026",
        parent=parent,
        critical=False
    )

    refs = nonempty_urls(ar.reference_urls if ar else [])

    # arena_venue_name (critical; presence)
    evaluator.add_custom_node(
        result=bool(ar and ar.venue_name and ar.venue_name.strip()),
        id="arena_venue_name",
        desc="Provide the specific arena venue name",
        parent=ar_node,
        critical=True
    )

    # arena_capacity_range (critical; URL-grounded)
    node = evaluator.add_leaf(
        id="arena_capacity_range",
        desc="The arena must have a seating capacity between 14,000 and 20,000",
        parent=ar_node,
        critical=True
    )
    claim = (
        f"The seating capacity of the arena '{safe(ar.venue_name)}' is between 14,000 and 20,000 inclusive."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=refs,
        additional_instruction="The supporting page(s) may be a venue page, Wikipedia, or official site showing capacity. Accept basketball/hockey capacity within 14k–20k."
    )

    # arena_event_description (critical; presence)
    evaluator.add_custom_node(
        result=bool(ar and ar.event_date and ar.event_name and ar.event_name.strip()),
        id="arena_event_description",
        desc="Provide a description of the specific event (sporting match, concert, etc.)",
        parent=ar_node,
        critical=True
    )

    # arena_specific_date (critical; URL-grounded)
    node = evaluator.add_leaf(
        id="arena_specific_date",
        desc="Provide the specific date when the arena event takes place",
        parent=ar_node,
        critical=True
    )
    claim = f"The arena event is scheduled on {safe(ar.event_date)}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=refs,
        additional_instruction="Accept small formatting differences (e.g., 'Jan 5, 2026' vs '2026-01-05') if the same date."
    )

    # arena_year_2026 (critical; logic)
    node = evaluator.add_leaf(
        id="arena_year_2026",
        desc="The event must take place during calendar year 2026",
        parent=ar_node,
        critical=True
    )
    claim = f"The arena event date '{safe(ar.event_date)}' is in calendar year 2026."
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="If the date does not clearly indicate 2026, mark incorrect."
    )

    # arena_complete_address (critical; presence/completeness)
    evaluator.add_custom_node(
        result=address_is_complete(ar.address if ar else None),
        id="arena_complete_address",
        desc="Provide the complete arena address including street (if applicable), city, state, and ZIP code",
        parent=ar_node,
        critical=True
    )

    # arena_state_uniqueness (critical; different from festival and residency)
    node = evaluator.add_leaf(
        id="arena_state_uniqueness",
        desc="The arena must be in a different U.S. state than both the music festival and the concert residency",
        parent=ar_node,
        critical=True
    )
    claim = (
        f"The arena event state '{safe(ar.address.state if ar and ar.address else None)}' is different from the "
        f"music festival state '{safe(mf.address.state if mf and mf.address else None)}' and the residency state "
        f"'{safe(res.address.state if res and res.address else None)}'."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Treat state abbreviations and full names as equivalent. All three must be pairwise different."
    )

    # arena_ticket_info (non-critical; URL-grounded if present)
    node = evaluator.add_leaf(
        id="arena_ticket_info",
        desc="Provide information about ticket sales or availability",
        parent=ar_node,
        critical=False
    )
    if ar and ar.ticket_info:
        claim = f"Ticket sales/availability for this arena event: {ar.ticket_info}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=refs,
            additional_instruction="Confirm that the page mentions ticket availability or pricing consistent with the answer."
        )
    else:
        await evaluator.verify(
            claim="The answer provides ticket sales/availability for this arena event.",
            node=node,
            additional_instruction="Mark incorrect if the answer did not provide any ticket info."
        )

    # arena_reference (critical; presence)
    evaluator.add_custom_node(
        result=len(refs) > 0,
        id="arena_reference",
        desc="Provide a URL reference supporting all arena event information",
        parent=ar_node,
        critical=True
    )


async def verify_theater(evaluator: Evaluator, parent) -> None:
    mf: MusicFestival = getattr(evaluator, "_mf_data", None)
    res: ConcertResidency = getattr(evaluator, "_res_data", None)
    ar: ArenaEvent = getattr(evaluator, "_arena_data", None)
    th: TheaterShow = getattr(evaluator, "_theater_data", None)

    th_node = evaluator.add_parallel(
        id="event_4_theater_show",
        desc="Identify a Broadway theater show or major theater production in 2026",
        parent=parent,
        critical=False
    )

    refs = nonempty_urls(th.reference_urls if th else [])

    # theater_venue_name (critical; presence)
    evaluator.add_custom_node(
        result=bool(th and th.theater_name and th.theater_name.strip()),
        id="theater_venue_name",
        desc="Provide the specific theater venue name",
        parent=th_node,
        critical=True
    )

    # theater_capacity (critical; URL-grounded ≥500 seats)
    node = evaluator.add_leaf(
        id="theater_capacity",
        desc="The theater must have a seating capacity of at least 500 seats to qualify as a Broadway or major theater venue",
        parent=th_node,
        critical=True
    )
    claim = f"The theater '{safe(th.theater_name)}' has a seating capacity of at least 500 seats."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=refs,
        additional_instruction="Use authoritative sources like the theater's site, Broadway League, or reliable listings. Accept any explicit statement ≥ 500."
    )

    # theater_show_name (critical; presence)
    evaluator.add_custom_node(
        result=bool(th and th.show_name and th.show_name.strip()),
        id="theater_show_name",
        desc="Provide the name of the show or production",
        parent=th_node,
        critical=True
    )

    # theater_specific_dates (critical; URL-grounded)
    node = evaluator.add_leaf(
        id="theater_specific_dates",
        desc="Provide information about when the show is scheduled (specific dates or date range)",
        parent=th_node,
        critical=True
    )
    claim = f"The show '{safe(th.show_name)}' is scheduled on: {dates_str_list(th.scheduled_dates if th else [])}."
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=refs,
        additional_instruction="Minor formatting differences are acceptable; the page should clearly show the listed performance dates/range."
    )

    # theater_year_2026 (critical; logic)
    node = evaluator.add_leaf(
        id="theater_year_2026",
        desc="The show must be scheduled for performances during 2026",
        parent=th_node,
        critical=True
    )
    claim = f"All the listed performance dates for the show occur in calendar year 2026: {dates_str_list(th.scheduled_dates if th else [])}."
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="If any date is outside 2026 or missing, mark incorrect."
    )

    # theater_complete_address (critical; presence/completeness)
    evaluator.add_custom_node(
        result=address_is_complete(th.address if th else None),
        id="theater_complete_address",
        desc="Provide the complete theater address including street, city, state, and ZIP code",
        parent=th_node,
        critical=True
    )

    # theater_state_uniqueness (critical; different from the other three)
    node = evaluator.add_leaf(
        id="theater_state_uniqueness",
        desc="The theater must be in a different U.S. state than the music festival, concert residency, and arena event",
        parent=th_node,
        critical=True
    )
    claim = (
        f"The theater show state '{safe(th.address.state if th and th.address else None)}' is different from the "
        f"music festival state '{safe(mf.address.state if mf and mf.address else None)}', the residency state "
        f"'{safe(res.address.state if res and res.address else None)}', and the arena event state "
        f"'{safe(ar.address.state if ar and ar.address else None)}'."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Treat state abbreviations and full names as equivalent. All four must be pairwise different."
    )

    # theater_ticket_info (non-critical; URL-grounded if present)
    node = evaluator.add_leaf(
        id="theater_ticket_info",
        desc="Provide information about ticket pricing or availability",
        parent=th_node,
        critical=False
    )
    if th and th.ticket_info:
        claim = f"Ticket pricing/availability for this theater show: {th.ticket_info}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=refs,
            additional_instruction="Confirm the gist on the page(s); minor formatting differences acceptable."
        )
    else:
        await evaluator.verify(
            claim="The answer provides ticket pricing or availability for this theater show.",
            node=node,
            additional_instruction="Mark incorrect if the answer did not provide any ticket info."
        )

    # theater_reference (critical; presence)
    evaluator.add_custom_node(
        result=len(refs) > 0,
        id="theater_reference",
        desc="Provide a URL reference supporting all theater show information",
        parent=th_node,
        critical=True
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
    Evaluate the agent's answer for the 2026 U.S. events task with four distinct categories and different states.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates four parallel event checks
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

    # 1) Extract structured data from the answer
    extracted: EventsExtraction = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction",
    )

    # Store per-event data on evaluator for easy access in subroutines
    mf = extracted.music_festival or MusicFestival()
    res = extracted.residency or ConcertResidency()
    ar = extracted.arena_event or ArenaEvent()
    th = extracted.theater_show or TheaterShow()
    setattr(evaluator, "_mf_data", mf)
    setattr(evaluator, "_res_data", res)
    setattr(evaluator, "_arena_data", ar)
    setattr(evaluator, "_theater_data", th)

    # Optionally record a quick custom info summary for states (helps debugging uniqueness checks)
    evaluator.add_custom_info(
        info={
            "festival_state": mf.address.state if mf.address else None,
            "residency_state": res.address.state if res.address else None,
            "arena_state": ar.address.state if ar.address else None,
            "theater_state": th.address.state if th.address else None,
        },
        info_type="state_summary",
        info_name="extracted_event_states"
    )

    # 2) Build verification tree following the rubric

    # Event 1: Music Festival
    await verify_music_festival(evaluator, root)

    # Event 2: Concert Residency
    await verify_residency(evaluator, root)

    # Event 3: Arena Event
    await verify_arena(evaluator, root)

    # Event 4: Theater Show
    await verify_theater(evaluator, root)

    # 3) Return final structured summary
    return evaluator.get_summary()