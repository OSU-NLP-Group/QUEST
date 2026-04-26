import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "music_events_2026_four_specific_performances"
TASK_DESCRIPTION = """
A music event documentation project requires detailed information about four specific ticketed live performances scheduled in 2026:

1. Jason Momoa performing with his band Oof Tatata in Dubai during January 2026
2. The electronic music duo ODESZA performing at a stadium venue in Los Angeles during June 2026
3. Till Lindemann's festival event at an outdoor venue in Leipzig during July 2026
4. The electronic music duo ODESZA performing at an amphitheater venue in California during June 2026

For each performance, provide: the specific event name, the complete venue name and location (city, region/state, country), the exact performance date(s), the venue's stated capacity for the event type, and any minimum age restrictions that apply to attendees.
"""


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class PerformanceItem(BaseModel):
    # Core fields extracted from the answer
    event_name: Optional[str] = None
    event_name_sources: List[str] = Field(default_factory=list)

    venue_name: Optional[str] = None
    venue_name_sources: List[str] = Field(default_factory=list)

    venue_location: Optional[str] = None  # free-form, e.g., "Dubai Media City, Dubai, United Arab Emirates"
    venue_location_sources: List[str] = Field(default_factory=list)

    performance_dates: Optional[str] = None  # free-form, e.g., "January 28, 2026" or "June 7–8, 2026"
    performance_date_sources: List[str] = Field(default_factory=list)

    venue_capacity: Optional[str] = None  # free-form, e.g., "5,000 (standing)", "22,000–24,000"
    capacity_sources: List[str] = Field(default_factory=list)

    age_restriction: Optional[str] = None  # e.g., "21+", "All ages", "No minimum age specified"
    age_restriction_sources: List[str] = Field(default_factory=list)

    # General sources that may support multiple fields for the performance
    general_sources: List[str] = Field(default_factory=list)

    # Optional helper/context fields
    description: Optional[str] = None  # brief description of the performance


class PerformancesExtraction(BaseModel):
    p1: Optional[PerformanceItem] = None  # Jason Momoa + Oof Tatata in Dubai (Jan 2026)
    p2: Optional[PerformanceItem] = None  # ODESZA stadium in Los Angeles (June 2026)
    p3: Optional[PerformanceItem] = None  # Till Lindemann festival outdoor in Leipzig (July 2026)
    p4: Optional[PerformanceItem] = None  # ODESZA amphitheater in California (June 2026)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_performances() -> str:
    return """
Extract structured details for FOUR specific 2026 performances mentioned in the answer. Return a JSON object with keys p1, p2, p3, p4, each being an object with the fields below. All URLs must be explicitly present in the answer text (plain URL or Markdown link). If a field is missing, set it to null (for strings) or [] (for arrays).

For each performance (p1–p4), extract:
- event_name: the specific event or tour name as stated
- event_name_sources: array of URLs supporting the event name for the performance
- venue_name: complete venue name
- venue_name_sources: array of URLs supporting the venue name
- venue_location: full location string (city, region/state if applicable, country)
- venue_location_sources: array of URLs supporting the location (venue or event page)
- performance_dates: exact date(s), e.g., "January 28, 2026" or "June 7–8, 2026"
- performance_date_sources: array of URLs supporting the date(s)
- venue_capacity: the venue’s stated capacity relevant to the event type (e.g., "5,000 (standing)", "22,000–24,000"), if explicitly available from event/venue sources
- capacity_sources: array of URLs that explicitly state that capacity (prefer venue’s official site or authoritative sources)
- age_restriction: minimum age restriction if stated (e.g., "21+", "18+", "All ages", or "No minimum age specified")
- age_restriction_sources: array of URLs supporting the age policy (event page, ticketing, or venue policy)
- general_sources: array of URLs that broadly support multiple details (event announcements, ticketing pages, venue event pages, etc.)
- description: brief free-text description of the performance to aid identification

Map the four requested performances exactly as follows:
- p1: “Jason Momoa performing with Oof Tatata in Dubai during January 2026”
- p2: “ODESZA at a stadium venue in Los Angeles during June 2026”
- p3: “Till Lindemann's festival at an outdoor venue in Leipzig during July 2026”
- p4: “ODESZA at an amphitheater venue in California during June 2026”

Rules:
- Do NOT invent URLs. Only use URLs explicitly provided in the answer.
- If a field is mentioned but without any supporting URL in the answer, leave the corresponding *_sources array empty.
- Prefer authoritative sources (official event/venue/ticketing/news) when they are present in the answer.
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls or []:
        if not u or not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _combine_sources(*lists: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in lists:
        if lst:
            combined.extend(lst)
    return _dedup_urls(combined)


def _missing_sources_instruction(sources: List[str]) -> str:
    if not sources:
        return "IMPORTANT: No URLs were provided for this check; you must mark the claim as NOT SUPPORTED/INCORRECT regardless of the answer text."
    return ""


def _build_context_description(perf_key: str) -> str:
    if perf_key == "p1":
        return "Jason Momoa performing with his band Oof Tatata in Dubai in January 2026."
    if perf_key == "p2":
        return "ODESZA performing at a stadium venue in Los Angeles in June 2026."
    if perf_key == "p3":
        return "Till Lindemann's festival event at an outdoor venue in Leipzig in July 2026."
    if perf_key == "p4":
        return "ODESZA performing at an amphitheater venue in California in June 2026."
    return ""


# -----------------------------------------------------------------------------
# Verification per performance
# -----------------------------------------------------------------------------
async def verify_performance_1(evaluator: Evaluator, parent_node, item: Optional[PerformanceItem]) -> None:
    perf_node = evaluator.add_parallel(
        id="performance_1",
        desc="Jason Momoa performing with Oof Tatata in Dubai during January 2026.",
        parent=parent_node,
        critical=False
    )

    # Safe defaults
    item = item or PerformanceItem()

    # Sources
    ev_sources = _combine_sources(item.event_name_sources, item.general_sources)
    vn_sources = _combine_sources(item.venue_name_sources, item.general_sources)
    loc_sources = _combine_sources(item.venue_location_sources, item.venue_name_sources, item.general_sources)
    dt_sources = _combine_sources(item.performance_date_sources, item.event_name_sources, item.general_sources)
    cap_sources = _combine_sources(item.capacity_sources, item.venue_name_sources)
    age_sources = _combine_sources(item.age_restriction_sources, item.event_name_sources, item.venue_name_sources, item.general_sources)

    # 1) Event name exact: "Meili Society Nights"
    leaf = evaluator.add_leaf(
        id="p1_event_name",
        desc="Event name is stated as \"Meili Society Nights\".",
        parent=perf_node,
        critical=True
    )
    claim = "The event name for the Jason Momoa + Oof Tatata Dubai performance is 'Meili Society Nights'."
    add_ins = (
        "Verify on the cited event/ticket/venue pages that the named performance is explicitly branded as 'Meili Society Nights'. "
        "Allow minor punctuation/casing differences. "
        + _missing_sources_instruction(ev_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=ev_sources, additional_instruction=add_ins)

    # 2) Venue name exact: "The Agenda Dubai"
    leaf = evaluator.add_leaf(
        id="p1_venue_name",
        desc="Complete venue name is stated as \"The Agenda Dubai\".",
        parent=perf_node,
        critical=True
    )
    claim = "The venue hosting this performance is 'The Agenda Dubai'."
    add_ins = (
        "Confirm the exact venue branding on the cited page(s). Accept 'THE AGENDA' if the page indicates it is the Dubai venue 'The Agenda Dubai'. "
        + _missing_sources_instruction(vn_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=vn_sources, additional_instruction=add_ins)

    # 3) Location includes: Dubai Media City, Dubai, United Arab Emirates
    leaf = evaluator.add_leaf(
        id="p1_venue_location",
        desc="Venue location includes Dubai Media City, Dubai, United Arab Emirates.",
        parent=perf_node,
        critical=True
    )
    claim = "The venue location includes 'Dubai Media City, Dubai, United Arab Emirates' (UAE)."
    add_ins = (
        "Accept 'UAE' as equivalent to 'United Arab Emirates'. "
        "The page should clearly place the venue within Dubai Media City in Dubai, UAE. "
        + _missing_sources_instruction(loc_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=loc_sources, additional_instruction=add_ins)

    # 4) Date exact: January 28, 2026
    leaf = evaluator.add_leaf(
        id="p1_performance_date",
        desc="Exact performance date is January 28, 2026.",
        parent=perf_node,
        critical=True
    )
    claim = "The exact performance date is January 28, 2026."
    add_ins = (
        "Verify the date text on the cited page(s). Allow localized month/day formatting variants (e.g., 28 January 2026). "
        + _missing_sources_instruction(dt_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=dt_sources, additional_instruction=add_ins)

    # 5) Venue capacity: 5,000 (standing capacity)
    leaf = evaluator.add_leaf(
        id="p1_venue_capacity",
        desc="Venue stated capacity for the relevant event type is given as 5,000 (standing capacity).",
        parent=perf_node,
        critical=True
    )
    claim = "The venue's stated standing capacity (relevant for concerts) is approximately 5,000."
    add_ins = (
        "Look for explicit capacity statements on venue or authoritative sources. "
        "Accept reasonable numeric variants (e.g., 'up to 5,000' or '5k standing'). "
        + _missing_sources_instruction(cap_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=cap_sources, additional_instruction=add_ins)

    # 6) Age restriction: 21+
    leaf = evaluator.add_leaf(
        id="p1_age_restrictions",
        desc="Minimum age restriction is stated as 21+.",
        parent=perf_node,
        critical=True
    )
    claim = "The minimum age restriction for attendees is 21+."
    add_ins = (
        "Confirm the age policy on the event/ticketing/venue page(s). Look for '21+' or equivalent adult-only admission language. "
        + _missing_sources_instruction(age_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=age_sources, additional_instruction=add_ins)


async def verify_performance_2(evaluator: Evaluator, parent_node, item: Optional[PerformanceItem]) -> None:
    perf_node = evaluator.add_parallel(
        id="performance_2",
        desc="ODESZA performing at a stadium venue in Los Angeles during June 2026.",
        parent=parent_node,
        critical=False
    )

    item = item or PerformanceItem()

    ev_sources = _combine_sources(item.event_name_sources, item.general_sources)
    vn_sources = _combine_sources(item.venue_name_sources, item.general_sources)
    loc_sources = _combine_sources(item.venue_location_sources, item.venue_name_sources, item.general_sources)
    dt_sources = _combine_sources(item.performance_date_sources, item.event_name_sources, item.general_sources)
    cap_sources = _combine_sources(item.capacity_sources, item.venue_name_sources)
    age_sources = _combine_sources(item.age_restriction_sources, item.event_name_sources, item.venue_name_sources, item.general_sources)

    # 1) Event/tour name provided and supported
    leaf = evaluator.add_leaf(
        id="p2_event_name",
        desc="Provides an event name or tour name for the ODESZA Los Angeles performance.",
        parent=perf_node,
        critical=True
    )
    claim = (
        "An event or tour name is clearly stated on the cited page(s) for the ODESZA stadium performance in Los Angeles in June 2026, "
        "and the answer provides such a name."
    )
    add_ins = (
        "Check the page(s) for a named tour or event label (e.g., a tour title). "
        "If the answer does not present a distinct event/tour name, mark as incorrect. "
        + _missing_sources_instruction(ev_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=ev_sources, additional_instruction=add_ins)

    # 2) Venue name exact: BMO Stadium
    leaf = evaluator.add_leaf(
        id="p2_venue_name",
        desc="Complete venue name is stated as \"BMO Stadium\".",
        parent=perf_node,
        critical=True
    )
    claim = "The venue for this ODESZA Los Angeles show is BMO Stadium."
    add_ins = (
        "Accept mention of 'BMO Stadium (formerly Banc of California Stadium)' as equivalent. "
        + _missing_sources_instruction(vn_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=vn_sources, additional_instruction=add_ins)

    # 3) Location includes Los Angeles, California, United States
    leaf = evaluator.add_leaf(
        id="p2_venue_location",
        desc="Venue location includes Los Angeles, California, United States.",
        parent=perf_node,
        critical=True
    )
    claim = "The venue location includes 'Los Angeles, California, United States'."
    add_ins = (
        "Allow 'USA' as equivalent to 'United States'. "
        + _missing_sources_instruction(loc_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=loc_sources, additional_instruction=add_ins)

    # 4) Dates exact: June 7–8, 2026
    leaf = evaluator.add_leaf(
        id="p2_performance_dates",
        desc="Exact performance date(s) are June 7–8, 2026.",
        parent=perf_node,
        critical=True
    )
    claim = "The exact performance dates are June 7–8, 2026."
    add_ins = (
        "Confirm that two dates, June 7 and June 8, 2026, are shown. Allow minor formatting variants. "
        + _missing_sources_instruction(dt_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=dt_sources, additional_instruction=add_ins)

    # 5) Concert capacity range: 22,000–24,000
    leaf = evaluator.add_leaf(
        id="p2_venue_capacity",
        desc="Venue stated concert capacity is given as 22,000–24,000.",
        parent=perf_node,
        critical=True
    )
    claim = "BMO Stadium's concert capacity is in the range of approximately 22,000 to 24,000."
    add_ins = (
        "Look for a capacity statement (concert configuration). Accept approximate range statements (e.g., ~22k–24k). "
        + _missing_sources_instruction(cap_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=cap_sources, additional_instruction=add_ins)

    # 6) Age restrictions: provided or none specified (verify based on pages)
    leaf = evaluator.add_leaf(
        id="p2_age_restrictions",
        desc="Provides any minimum age restriction stated for attendees (or indicates that no minimum age restriction is specified by the event/venue).",
        parent=perf_node,
        critical=True
    )
    if item.age_restriction and item.age_restriction.strip() and item.age_restriction.strip().lower() not in {
        "no minimum age specified", "no minimum age restriction specified", "not specified", "none"
    }:
        claim = f"The minimum age restriction stated for attendees is '{item.age_restriction}'."
        add_ins = (
            "Verify the exact or equivalent age policy (e.g., 'All ages', '18+', '21+'). "
            + _missing_sources_instruction(age_sources)
        )
    else:
        claim = "No minimum age restriction is explicitly specified on the cited page(s) for this event."
        add_ins = (
            "Pass this only if the page(s) lack any explicit minimum age policy. If an age policy is present, mark as incorrect. "
            + _missing_sources_instruction(age_sources)
        )
    await evaluator.verify(claim=claim, node=leaf, sources=age_sources, additional_instruction=add_ins)


async def verify_performance_3(evaluator: Evaluator, parent_node, item: Optional[PerformanceItem]) -> None:
    perf_node = evaluator.add_parallel(
        id="performance_3",
        desc="Till Lindemann festival event at an outdoor venue in Leipzig during July 2026.",
        parent=parent_node,
        critical=False
    )

    item = item or PerformanceItem()

    ev_sources = _combine_sources(item.event_name_sources, item.general_sources)
    vn_sources = _combine_sources(item.venue_name_sources, item.general_sources)
    loc_sources = _combine_sources(item.venue_location_sources, item.venue_name_sources, item.general_sources)
    dt_sources = _combine_sources(item.performance_date_sources, item.event_name_sources, item.general_sources)
    cap_sources = _combine_sources(item.capacity_sources, item.venue_name_sources)
    age_sources = _combine_sources(item.age_restriction_sources, item.event_name_sources, item.venue_name_sources, item.general_sources)

    # 1) Event name exact: "Till Fest 2026"
    leaf = evaluator.add_leaf(
        id="p3_event_name",
        desc="Festival event name is stated as \"Till Fest 2026\".",
        parent=perf_node,
        critical=True
    )
    claim = "The festival/event name is 'Till Fest 2026'."
    add_ins = (
        "Verify that the cited pages for the Leipzig July 2026 Till Lindemann event explicitly brand it as 'Till Fest 2026'. "
        + _missing_sources_instruction(ev_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=ev_sources, additional_instruction=add_ins)

    # 2) Venue name exact: "Völkerschlachtdenkmal"
    leaf = evaluator.add_leaf(
        id="p3_venue_name",
        desc="Complete venue name is stated as \"Völkerschlachtdenkmal\".",
        parent=perf_node,
        critical=True
    )
    claim = "The event is held at 'Völkerschlachtdenkmal' (Monument to the Battle of the Nations)."
    add_ins = (
        "Accept the German spelling 'Völkerschlachtdenkmal'. English alias 'Monument to the Battle of the Nations' may appear; this should still confirm the venue identity. "
        + _missing_sources_instruction(vn_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=vn_sources, additional_instruction=add_ins)

    # 3) Location includes Leipzig, Saxony, Germany
    leaf = evaluator.add_leaf(
        id="p3_venue_location",
        desc="Venue location includes Leipzig, Saxony, Germany.",
        parent=perf_node,
        critical=True
    )
    claim = "The venue location includes 'Leipzig, Saxony, Germany'."
    add_ins = (
        "Allow 'Sachsen' as equivalent to 'Saxony'. "
        + _missing_sources_instruction(loc_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=loc_sources, additional_instruction=add_ins)

    # 4) Dates exact: July 3–4, 2026
    leaf = evaluator.add_leaf(
        id="p3_performance_dates",
        desc="Exact festival date(s) are July 3–4, 2026.",
        parent=perf_node,
        critical=True
    )
    claim = "The festival dates are July 3–4, 2026."
    add_ins = (
        "Confirm both dates are listed for 2026 (3 and 4 July 2026). "
        + _missing_sources_instruction(dt_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=dt_sources, additional_instruction=add_ins)

    # 5) Venue stated capacity relevant to event type (or indicates not available)
    leaf = evaluator.add_leaf(
        id="p3_venue_capacity",
        desc="Provides the venue’s stated capacity relevant to the event type (or indicates that a stated capacity is not available from the event/venue).",
        parent=perf_node,
        critical=True
    )
    if item.venue_capacity and item.venue_capacity.strip().lower() not in {"not available", "n/a", "unknown"}:
        claim = f"The venue’s stated capacity relevant to this event is '{item.venue_capacity}'."
        add_ins = (
            "Verify the presence of a numeric capacity value attributable to the venue/event context. "
            + _missing_sources_instruction(cap_sources)
        )
    else:
        claim = "No explicit venue capacity relevant to this event type is stated on the cited event/venue sources."
        add_ins = (
            "Pass only if the page(s) do not present a clear, stated capacity. If a capacity is shown, mark as incorrect. "
            + _missing_sources_instruction(cap_sources)
        )
    await evaluator.verify(claim=claim, node=leaf, sources=cap_sources, additional_instruction=add_ins)

    # 6) Age restriction: provided or none specified
    leaf = evaluator.add_leaf(
        id="p3_age_restrictions",
        desc="Provides any minimum age restriction stated for attendees (or indicates that no minimum age restriction is specified by the event/venue).",
        parent=perf_node,
        critical=True
    )
    if item.age_restriction and item.age_restriction.strip().lower() not in {
        "no minimum age specified", "no minimum age restriction specified", "not specified", "none"
    }:
        claim = f"The minimum age restriction stated for attendees is '{item.age_restriction}'."
        add_ins = (
            "Verify the exact or equivalent age policy. "
            + _missing_sources_instruction(age_sources)
        )
    else:
        claim = "No minimum age restriction is explicitly specified on the cited page(s) for this event."
        add_ins = (
            "Pass this only if no explicit minimum age is found on the cited sources. "
            + _missing_sources_instruction(age_sources)
        )
    await evaluator.verify(claim=claim, node=leaf, sources=age_sources, additional_instruction=add_ins)


async def verify_performance_4(evaluator: Evaluator, parent_node, item: Optional[PerformanceItem]) -> None:
    perf_node = evaluator.add_parallel(
        id="performance_4",
        desc="ODESZA performing at an amphitheater venue in California during June 2026.",
        parent=parent_node,
        critical=False
    )

    item = item or PerformanceItem()

    ev_sources = _combine_sources(item.event_name_sources, item.general_sources)
    vn_sources = _combine_sources(item.venue_name_sources, item.general_sources)
    loc_sources = _combine_sources(item.venue_location_sources, item.venue_name_sources, item.general_sources)
    dt_sources = _combine_sources(item.performance_date_sources, item.event_name_sources, item.general_sources)
    cap_sources = _combine_sources(item.capacity_sources, item.venue_name_sources)
    age_sources = _combine_sources(item.age_restriction_sources, item.event_name_sources, item.venue_name_sources, item.general_sources)

    # 1) Event/tour name provided and supported
    leaf = evaluator.add_leaf(
        id="p4_event_name",
        desc="Provides an event name or tour name for the ODESZA performance.",
        parent=perf_node,
        critical=True
    )
    claim = (
        "An event or tour name is clearly stated on the cited page(s) for the ODESZA amphitheater show in California in June 2026, "
        "and the answer provides such a name."
    )
    add_ins = (
        "Look for a tour title or event branding string on the cited pages. "
        "If the answer did not provide a distinct name, mark incorrect. "
        + _missing_sources_instruction(ev_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=ev_sources, additional_instruction=add_ins)

    # 2) Venue name: Greek Theatre Berkeley / Greek Theatre (William Randolph Hearst Greek Theatre)
    leaf = evaluator.add_leaf(
        id="p4_venue_name",
        desc="Complete venue name is stated as Greek Theatre Berkeley / Greek Theatre (William Randolph Hearst Greek Theatre).",
        parent=perf_node,
        critical=True
    )
    claim = "The venue is the Greek Theatre in Berkeley (aka William Randolph Hearst Greek Theatre)."
    add_ins = (
        "Accept 'Greek Theatre', 'Greek Theatre Berkeley', or 'William Randolph Hearst Greek Theatre' if the page context shows it's the Berkeley venue. "
        + _missing_sources_instruction(vn_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=vn_sources, additional_instruction=add_ins)

    # 3) Location includes Berkeley, California, United States
    leaf = evaluator.add_leaf(
        id="p4_venue_location",
        desc="Venue location includes Berkeley, California, United States.",
        parent=perf_node,
        critical=True
    )
    claim = "The venue location includes 'Berkeley, California, United States'."
    add_ins = (
        "Allow 'USA' as equivalent to 'United States'. "
        + _missing_sources_instruction(loc_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=loc_sources, additional_instruction=add_ins)

    # 4) Date exact: June 13, 2026
    leaf = evaluator.add_leaf(
        id="p4_performance_date",
        desc="Exact performance date is June 13, 2026.",
        parent=perf_node,
        critical=True
    )
    claim = "The exact performance date is June 13, 2026."
    add_ins = (
        "Confirm the date text on the cited pages; allow localized formatting ('13 June 2026'). "
        + _missing_sources_instruction(dt_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=dt_sources, additional_instruction=add_ins)

    # 5) Venue capacity: 8,500
    leaf = evaluator.add_leaf(
        id="p4_venue_capacity",
        desc="Venue stated capacity is given as 8,500.",
        parent=perf_node,
        critical=True
    )
    claim = "The Greek Theatre in Berkeley has a stated capacity of approximately 8,500."
    add_ins = (
        "Verify a stated capacity near 8,500 on authoritative pages (venue official preferred). "
        "Accept minor numeric rounding. "
        + _missing_sources_instruction(cap_sources)
    )
    await evaluator.verify(claim=claim, node=leaf, sources=cap_sources, additional_instruction=add_ins)

    # 6) Age restrictions: provided or none specified
    leaf = evaluator.add_leaf(
        id="p4_age_restrictions",
        desc="Provides any minimum age restriction stated for attendees (or indicates that no minimum age restriction is specified by the event/venue).",
        parent=perf_node,
        critical=True
    )
    if item.age_restriction and item.age_restriction.strip().lower() not in {
        "no minimum age specified", "no minimum age restriction specified", "not specified", "none"
    }:
        claim = f"The minimum age restriction stated for attendees is '{item.age_restriction}'."
        add_ins = (
            "Verify exact or equivalent age policy. "
            + _missing_sources_instruction(age_sources)
        )
    else:
        claim = "No minimum age restriction is explicitly specified on the cited page(s) for this event."
        add_ins = (
            "Pass only if the cited pages do not include an explicit minimum age. "
            + _missing_sources_instruction(age_sources)
        )
    await evaluator.verify(claim=claim, node=leaf, sources=age_sources, additional_instruction=add_ins)


# -----------------------------------------------------------------------------
# Main evaluation entry point
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
    """
    Evaluate an answer for the four specified 2026 performances task.
    """
    evaluator = Evaluator()
    # Note: Make root non-critical to comply with framework rule that critical parent
    # cannot have non-critical children. We still evaluate all criteria strictly at leaf level.
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=(
            "Provide the required details (event name, complete venue name and location, exact date(s), "
            "venue stated capacity for the relevant event type, and minimum age restriction info) "
            "for the four specified 2026 performances."
        ),
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extraction
    extracted: PerformancesExtraction = await evaluator.extract(
        prompt=prompt_extract_performances(),
        template_class=PerformancesExtraction,
        extraction_name="performances_extraction",
    )

    # Ground truth/context expectations for verifications that require exact values
    evaluator.add_ground_truth({
        "p1_expected": {
            "event_name": "Meili Society Nights",
            "venue_name": "The Agenda Dubai",
            "location_includes": "Dubai Media City, Dubai, United Arab Emirates",
            "date": "January 28, 2026",
            "capacity": "5,000 (standing)",
            "age_restriction": "21+"
        },
        "p2_expected": {
            "venue_name": "BMO Stadium",
            "location_includes": "Los Angeles, California, United States",
            "dates": "June 7–8, 2026",
            "concert_capacity_range": "22,000–24,000"
        },
        "p3_expected": {
            "event_name": "Till Fest 2026",
            "venue_name": "Völkerschlachtdenkmal",
            "location_includes": "Leipzig, Saxony, Germany",
            "dates": "July 3–4, 2026"
        },
        "p4_expected": {
            "venue_name": "Greek Theatre (Berkeley) aka William Randolph Hearst Greek Theatre",
            "location_includes": "Berkeley, California, United States",
            "date": "June 13, 2026",
            "capacity": "8,500"
        }
    }, gt_type="expected_values")

    # Build performance groups and verify leaves
    await verify_performance_1(evaluator, root, getattr(extracted, "p1", None))
    await verify_performance_2(evaluator, root, getattr(extracted, "p2", None))
    await verify_performance_3(evaluator, root, getattr(extracted, "p3", None))
    await verify_performance_4(evaluator, root, getattr(extracted, "p4", None))

    return evaluator.get_summary()