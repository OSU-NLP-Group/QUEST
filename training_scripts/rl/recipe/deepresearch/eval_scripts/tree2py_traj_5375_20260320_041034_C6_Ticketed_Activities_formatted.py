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
TASK_ID = "march_2026_performances"
TASK_DESCRIPTION = """
Identify four distinct ticketed performances taking place during March 2026 in the United States. Each performance must meet the following specific criteria:

Performance 1: An Off-Broadway theatrical production
- Must be performed at a venue classified as Off-Broadway (seating capacity between 100-499 seats)
- Must have at least one performance date in March 2026

Performance 2: A classical music concert
- Must take place at a concert hall with a minimum seating capacity of 1,500
- Must feature classical music repertoire
- Must have a performance date in March 2026

Performance 3: A stage theater production
- Must be a play, musical, or theatrical stage adaptation
- Must be a live performance (not a film screening)
- Must have at least one performance date in March 2026

Performance 4: Any additional ticketed performance
- Must be a ticketed performing arts or entertainment event
- Must have at least one performance date in March 2026

For each performance, provide:
- The name of the performance/production
- The venue name and location (city, state)
- The venue's seating capacity
- Confirmation that performance dates include March 2026
- At least one reference URL from an official production or venue source verifying the information
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class PerfBase(BaseModel):
    name: Optional[str] = None
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    seating_capacity: Optional[str] = None  # keep as free text; verification will use sources
    date_text: Optional[str] = None         # free text dates/range as given in the answer
    reference_urls: List[str] = Field(default_factory=list)


class Performance1(PerfBase):
    # Off-Broadway production (no extra fields; classification verified via capacity and nature)
    pass


class Performance2(PerfBase):
    # Classical music concert
    repertoire: Optional[str] = None
    composers: List[str] = Field(default_factory=list)
    performer_type: Optional[str] = None  # e.g., orchestra, chamber ensemble, soloist


class Performance3(PerfBase):
    # Stage theater production
    production_type: Optional[str] = None  # play, musical, stage adaptation
    live_indicator: Optional[str] = None   # any phrase in the answer indicating live performance


class Performance4(PerfBase):
    # Any additional ticketed performance
    pass


class ExtractedPerformances(BaseModel):
    performance1: Optional[Performance1] = None
    performance2: Optional[Performance2] = None
    performance3: Optional[Performance3] = None
    performance4: Optional[Performance4] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_performances() -> str:
    return """
    Extract structured information for FOUR distinct performances as presented in the answer. Do not invent anything.
    For each performance, extract exactly the following fields:

    Common fields for all four:
    - name: The performance/production/event name.
    - venue_name: The venue name.
    - city: City where the venue is located.
    - state: State where the venue is located (two-letter abbreviation or full name).
    - seating_capacity: The venue's seating capacity as stated in the answer (keep as text if not numeric).
    - date_text: The performance date(s) or run range as stated in the answer.
    - reference_urls: A list of URLs explicitly included in the answer that support this performance. Only include valid URLs (plain or markdown). If none, return an empty list.

    Additional fields for performance #2 (classical music):
    - repertoire: The described program or repertoire text (if any).
    - composers: A list of composer names mentioned (if any).
    - performer_type: The type of performer (e.g., orchestra, chamber ensemble, string quartet, pianist, violinist).

    Additional fields for performance #3 (stage theater):
    - production_type: One of: "play", "musical", "stage adaptation", or a close variant (if identifiable).
    - live_indicator: Any explicit indicator in the answer that it is a live performance (e.g., "live on stage", "stage production", "theatrical performance"). If none, return null.

    Map the extracted information into this JSON structure:
    {
      "performance1": { ... common fields ... },
      "performance2": { ... common fields + classical fields ... },
      "performance3": { ... common fields + theater fields ... },
      "performance4": { ... common fields ... }
    }

    Rules:
    - If a field is missing in the answer, set it to null (or [] for lists).
    - Only include URLs that are explicitly present in the answer text.
    - Do not deduplicate across performances; keep them separate.
    - Do not infer data beyond what is written in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _clean_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out = []
    for u in urls:
        if isinstance(u, str):
            s = u.strip()
            if s:
                # basic normalization: prepend http if missing scheme
                if not (s.startswith("http://") or s.startswith("https://")) and "://" not in s:
                    s = "http://" + s
                out.append(s)
    # de-duplicate maintaining order
    deduped = []
    seen = set()
    for u in out:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def _city_state_present(city: Optional[str], state: Optional[str]) -> bool:
    return bool(city and city.strip()) and bool(state and state.strip())


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _add_reference_group(
    evaluator: Evaluator,
    parent,
    perf_prefix: str,
    urls: List[str],
) -> Dict[str, Any]:
    """
    Build the 'reference' group:
      - url_provided (custom, critical)
      - url_official (leaf, critical) verified by the provided URLs
    Return dict with nodes for later use as prerequisites.
    """
    ref_group = evaluator.add_parallel(
        id=f"{perf_prefix}_reference",
        desc="At least one reference URL from official source provided",
        parent=parent,
        critical=True
    )

    url_provided_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"{perf_prefix}_url_provided",
        desc="Reference URL is included",
        parent=ref_group,
        critical=True
    )

    url_official_node = evaluator.add_leaf(
        id=f"{perf_prefix}_url_official",
        desc="URL is from official production or venue source",
        parent=ref_group,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of the provided URLs is an official page for the production or the venue (e.g., venue's own website, producing organization's website, or the venue's official ticketing page). Third-party aggregators or resellers should not be considered official.",
        node=url_official_node,
        sources=urls,
        additional_instruction="Consider the domain and on-page branding to determine official status. Accept if the URL clearly represents the official venue/organization/production site or its native ticketing system.",
        extra_prerequisites=[url_provided_node]
    )

    return {
        "group": ref_group,
        "url_provided": url_provided_node,
        "url_official": url_official_node
    }


async def verify_performance_1(
    evaluator: Evaluator,
    root,
    perf: Optional[Performance1]
) -> None:
    """
    Performance 1: Off-Broadway theatrical production (venue capacity 100-499) + March 2026 date.
    """
    node = evaluator.add_parallel(
        id="performance_1",
        desc="First performance: Off-Broadway production (venue with 100-499 seat capacity)",
        parent=root,
        critical=False
    )
    # Extract fields safely
    name = (perf.name or "").strip() if perf else ""
    venue = (perf.venue_name or "").strip() if perf else ""
    city = (perf.city or "").strip() if perf else ""
    state = (perf.state or "").strip() if perf else ""
    urls = _clean_urls(perf.reference_urls if perf else [])

    # Build reference group first (used as prerequisite for most verifications)
    refs = await _add_reference_group(evaluator, node, "p1", urls)
    url_provided = refs["url_provided"]

    # Venue classification group (critical)
    vclass = evaluator.add_parallel(
        id="p1_venue_classification",
        desc="Venue is classified as Off-Broadway with seating capacity between 100-499 seats",
        parent=node,
        critical=True
    )

    # Capacity verification (critical)
    cap_group = evaluator.add_parallel(
        id="p1_capacity_verification",
        desc="Venue seating capacity is verifiable and within Off-Broadway range",
        parent=vclass,
        critical=True
    )

    cap_range = evaluator.add_leaf(
        id="p1_capacity_range",
        desc="Stated capacity falls within 100-499 seats",
        parent=cap_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{venue}' has a seating capacity between 100 and 499 seats.",
        node=cap_range,
        sources=urls,
        additional_instruction="Verify from the provided page(s) that the venue's capacity is within [100, 499]. Accept if the page explicitly states a capacity in that range or if an official spec implies it.",
        extra_prerequisites=[url_provided]
    )

    cap_documented = evaluator.add_leaf(
        id="p1_capacity_documented",
        desc="Capacity figure is documented in provided reference",
        parent=cap_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided page(s) explicitly state a numeric seating capacity for the venue '{venue}'.",
        node=cap_documented,
        sources=urls,
        additional_instruction="Look for explicit numeric seat counts on an official venue or producing organization page.",
        extra_prerequisites=[url_provided]
    )

    # US location (critical)
    loc_group = evaluator.add_parallel(
        id="p1_us_location",
        desc="Venue is located in the United States",
        parent=vclass,
        critical=True
    )

    city_state_node = evaluator.add_custom_node(
        result=_city_state_present(city, state),
        id="p1_city_state",
        desc="City and state are provided",
        parent=loc_group,
        critical=True
    )

    loc_verified = evaluator.add_leaf(
        id="p1_location_verified",
        desc="Location is confirmed in reference source",
        parent=loc_group,
        critical=True
    )
    loc_claim = f"The venue '{venue}' is located in {city}, {state}, United States."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_verified,
        sources=urls,
        additional_instruction="Confirm city and state on the official venue or production page.",
        extra_prerequisites=[url_provided, city_state_node]
    )

    # Timing (critical)
    timing_group = evaluator.add_parallel(
        id="p1_march_timing",
        desc="Performance run dates include or overlap with March 2026",
        parent=node,
        critical=True
    )

    p1_dates_provided = evaluator.add_parallel(
        id="p1_dates_provided",
        desc="Performance dates or run period is specified",
        parent=timing_group,
        critical=True
    )

    p1_date_format = evaluator.add_leaf(
        id="p1_date_format",
        desc="Dates are clearly stated (specific dates or date range)",
        parent=p1_dates_provided,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official page(s) for '{name}' at '{venue}' clearly state performance date(s) or a date range.",
        node=p1_date_format,
        sources=urls,
        additional_instruction="Check if dates are explicitly listed in a readable format (e.g., Mar 5–29, 2026; or specific March 2026 dates).",
        extra_prerequisites=[url_provided]
    )

    p1_date_source = evaluator.add_leaf(
        id="p1_date_source",
        desc="Dates are verifiable from provided reference",
        parent=p1_dates_provided,
        critical=True
    )
    await evaluator.verify(
        claim="The performance dates are presented on the provided page(s).",
        node=p1_date_source,
        sources=urls,
        additional_instruction="There should be an explicit schedule or calendar on the page.",
        extra_prerequisites=[url_provided]
    )

    p1_march_overlap = evaluator.add_leaf(
        id="p1_march_overlap",
        desc="At least one performance date falls within March 1-31, 2026",
        parent=timing_group,
        critical=True
    )
    await evaluator.verify(
        claim="This event has at least one performance date between March 1 and March 31, 2026 (inclusive).",
        node=p1_march_overlap,
        sources=urls,
        additional_instruction="Confirm at least one listed performance date is in March 2026.",
        extra_prerequisites=[url_provided, p1_date_source]
    )

    # Performance attributes (critical)
    attrs_group = evaluator.add_parallel(
        id="p1_performance_attributes",
        desc="Performance is a ticketed Off-Broadway theatrical production",
        parent=node,
        critical=True
    )

    p1_ticketed = evaluator.add_leaf(
        id="p1_ticketed",
        desc="Evidence of ticketed event with publicly available sales",
        parent=attrs_group,
        critical=True
    )
    await evaluator.verify(
        claim="Tickets are available for purchase (or ticketing is clearly indicated) for this event on the provided page(s).",
        node=p1_ticketed,
        sources=urls,
        additional_instruction="Look for 'Tickets', 'Buy Tickets', or equivalent call-to-action for this specific performance.",
        extra_prerequisites=[url_provided]
    )

    p1_theatrical = evaluator.add_leaf(
        id="p1_theatrical_nature",
        desc="Performance is identified as a theatrical or performing arts production",
        parent=attrs_group,
        critical=True
    )
    await evaluator.verify(
        claim="This event is a theatrical stage production (e.g., play or musical), not a concert or talk.",
        node=p1_theatrical,
        sources=urls,
        additional_instruction="Accept if the page describes it as a play, musical, or theater production.",
        extra_prerequisites=[url_provided]
    )


async def verify_performance_2(
    evaluator: Evaluator,
    root,
    perf: Optional[Performance2]
) -> None:
    """
    Performance 2: Classical music concert at a concert hall with 1,500+ capacity + March 2026 date.
    """
    node = evaluator.add_parallel(
        id="performance_2",
        desc="Second performance: Classical music concert at a concert hall with 1,500+ seat capacity",
        parent=root,
        critical=False
    )
    name = (perf.name or "").strip() if perf else ""
    venue = (perf.venue_name or "").strip() if perf else ""
    city = (perf.city or "").strip() if perf else ""
    state = (perf.state or "").strip() if perf else ""
    urls = _clean_urls(perf.reference_urls if perf else [])

    refs = await _add_reference_group(evaluator, node, "p2", urls)
    url_provided = refs["url_provided"]

    # Venue specifications (critical)
    ven_spec = evaluator.add_parallel(
        id="p2_venue_specifications",
        desc="Venue is a concert hall with minimum 1,500 seat capacity suitable for classical music",
        parent=node,
        critical=True
    )

    cap_group = evaluator.add_parallel(
        id="p2_capacity_verification",
        desc="Venue seating capacity is verifiable and meets minimum requirement",
        parent=ven_spec,
        critical=True
    )

    cap_min = evaluator.add_leaf(
        id="p2_capacity_minimum",
        desc="Stated capacity is 1,500 or greater",
        parent=cap_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{venue}' has a seating capacity of at least 1,500 seats.",
        node=cap_min,
        sources=urls,
        additional_instruction="Verify that the official page indicates capacity ≥ 1500.",
        extra_prerequisites=[url_provided]
    )

    cap_doc = evaluator.add_leaf(
        id="p2_capacity_documented",
        desc="Capacity figure is documented in provided reference",
        parent=cap_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided page(s) explicitly state a numeric seating capacity for the venue '{venue}'.",
        node=cap_doc,
        sources=urls,
        additional_instruction="Look for an explicit seat count or spec on an official page.",
        extra_prerequisites=[url_provided]
    )

    vtype_loc = evaluator.add_parallel(
        id="p2_venue_type_location",
        desc="Venue type and location are appropriate",
        parent=ven_spec,
        critical=True
    )

    concert_hall = evaluator.add_leaf(
        id="p2_concert_hall",
        desc="Venue is designated or functions as a concert hall or music center",
        parent=vtype_loc,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{venue}' is a concert hall (e.g., symphony hall, concert hall, music center).",
        node=concert_hall,
        sources=urls,
        additional_instruction="Confirm the venue is purpose-built or commonly designated for classical music performances.",
        extra_prerequisites=[url_provided]
    )

    us_loc = evaluator.add_leaf(
        id="p2_us_location",
        desc="Venue is located in the United States with city and state provided",
        parent=vtype_loc,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{venue}' is located in {city}, {state}, United States.",
        node=us_loc,
        sources=urls,
        additional_instruction="Confirm city and state on the official page.",
        extra_prerequisites=[url_provided]
    )

    # Classical performance checks (critical)
    classical = evaluator.add_parallel(
        id="p2_classical_performance",
        desc="Performance is a classical music concert or recital",
        parent=node,
        critical=True
    )

    rep_group = evaluator.add_parallel(
        id="p2_repertoire",
        desc="Performance features classical music repertoire",
        parent=classical,
        critical=True
    )

    genre = evaluator.add_leaf(
        id="p2_music_genre",
        desc="Musical works are identified as classical",
        parent=rep_group,
        critical=True
    )
    await evaluator.verify(
        claim="The program is classical music (e.g., symphonic, chamber, solo classical repertoire).",
        node=genre,
        sources=urls,
        additional_instruction="Accept if the page uses classical terminology or references standard classical repertoire.",
        extra_prerequisites=[url_provided]
    )

    composers = evaluator.add_leaf(
        id="p2_composer_works",
        desc="Specific composers or works are mentioned or verifiable",
        parent=rep_group,
        critical=True
    )
    await evaluator.verify(
        claim="The provided page(s) mention specific classical composers or works for this performance.",
        node=composers,
        sources=urls,
        additional_instruction="Look for names like Beethoven, Mozart, Brahms, etc., or specific classical piece titles.",
        extra_prerequisites=[url_provided]
    )

    performer_type = evaluator.add_leaf(
        id="p2_performer_type",
        desc="Performance involves classical musicians, orchestra, or solo classical artist",
        parent=classical,
        critical=True
    )
    await evaluator.verify(
        claim="The performers are classical musicians (e.g., orchestra, symphony, chamber ensemble, or classical soloist).",
        node=performer_type,
        sources=urls,
        additional_instruction="Confirm performer identity/type (orchestra, quartet, pianist, etc.).",
        extra_prerequisites=[url_provided]
    )

    # Timing (critical)
    p2_timing = evaluator.add_parallel(
        id="p2_march_timing",
        desc="Performance occurs in March 2026",
        parent=node,
        critical=True
    )

    p2_date_provided = evaluator.add_parallel(
        id="p2_date_provided",
        desc="Performance date is specified",
        parent=p2_timing,
        critical=True
    )

    p2_date_format = evaluator.add_leaf(
        id="p2_date_format",
        desc="Date is clearly stated",
        parent=p2_date_provided,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official page(s) for '{name}' at '{venue}' clearly state the performance date(s).",
        node=p2_date_format,
        sources=urls,
        additional_instruction="Look for explicit calendar/date entries.",
        extra_prerequisites=[url_provided]
    )

    p2_date_source = evaluator.add_leaf(
        id="p2_date_source",
        desc="Date is verifiable from provided reference",
        parent=p2_date_provided,
        critical=True
    )
    await evaluator.verify(
        claim="The performance date(s) are presented on the provided page(s).",
        node=p2_date_source,
        sources=urls,
        additional_instruction="There should be an explicit date shown for the concert.",
        extra_prerequisites=[url_provided]
    )

    p2_march = evaluator.add_leaf(
        id="p2_march_date",
        desc="Performance date falls within March 1-31, 2026",
        parent=p2_timing,
        critical=True
    )
    await evaluator.verify(
        claim="This concert has at least one performance date between March 1 and March 31, 2026 (inclusive).",
        node=p2_march,
        sources=urls,
        additional_instruction="Confirm the listed date is in March 2026.",
        extra_prerequisites=[url_provided, p2_date_source]
    )

    # Ticketed (critical)
    p2_ticketed = evaluator.add_leaf(
        id="p2_ticketed_event",
        desc="Event is a ticketed performance with publicly available ticket sales",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Tickets are available to purchase (or clearly indicated) for this concert on the provided page(s).",
        node=p2_ticketed,
        sources=urls,
        additional_instruction="Look for 'Tickets', 'Buy Tickets', or similar.",
        extra_prerequisites=[url_provided]
    )


async def verify_performance_3(
    evaluator: Evaluator,
    root,
    perf: Optional[Performance3]
) -> None:
    """
    Performance 3: Stage theater production (play/musical/adaptation), live, with March 2026 date.
    """
    node = evaluator.add_parallel(
        id="performance_3",
        desc="Third performance: Stage theater production at any qualifying venue",
        parent=root,
        critical=False
    )
    name = (perf.name or "").strip() if perf else ""
    venue = (perf.venue_name or "").strip() if perf else ""
    city = (perf.city or "").strip() if perf else ""
    state = (perf.state or "").strip() if perf else ""
    ptype = (perf.production_type or "").strip() if perf else ""
    urls = _clean_urls(perf.reference_urls if perf else [])

    refs = await _add_reference_group(evaluator, node, "p3", urls)
    url_provided = refs["url_provided"]

    # Theater production classification (critical)
    theater = evaluator.add_parallel(
        id="p3_theater_production",
        desc="Performance is a stage theater production (play, musical, or theatrical adaptation)",
        parent=node,
        critical=True
    )

    prod_class = evaluator.add_parallel(
        id="p3_production_classification",
        desc="Production is identified as play, musical, or stage adaptation",
        parent=theater,
        critical=True
    )

    prod_type = evaluator.add_leaf(
        id="p3_production_type",
        desc="Specific type (play/musical/adaptation) is clear",
        parent=prod_class,
        critical=True
    )
    prod_type_text = ptype if ptype else "a play or musical or stage adaptation"
    await evaluator.verify(
        claim=f"This production is {prod_type_text}.",
        node=prod_type,
        sources=urls,
        additional_instruction="Confirm the page classifies the production (e.g., 'a new play', 'a musical', 'stage adaptation').",
        extra_prerequisites=[url_provided]
    )

    prod_verified = evaluator.add_leaf(
        id="p3_production_verified",
        desc="Production type is confirmed in reference source",
        parent=prod_class,
        critical=True
    )
    await evaluator.verify(
        claim="The production's classification as a play, musical, or stage adaptation is explicitly confirmed on the provided page(s).",
        node=prod_verified,
        sources=urls,
        additional_instruction="Look for explicit wording confirming the type.",
        extra_prerequisites=[url_provided]
    )

    live_leaf = evaluator.add_leaf(
        id="p3_live_performance",
        desc="Performance is a live theatrical production (not film screening)",
        parent=theater,
        critical=True
    )
    await evaluator.verify(
        claim="This is a live theatrical performance, not a film screening.",
        node=live_leaf,
        sources=urls,
        additional_instruction="Confirm that the page describes a live, on-stage production.",
        extra_prerequisites=[url_provided]
    )

    # Venue details (critical)
    venue_details = evaluator.add_parallel(
        id="p3_venue_details",
        desc="Venue information including location and seating capacity is provided",
        parent=node,
        critical=True
    )

    loc_cap = evaluator.add_parallel(
        id="p3_location_capacity",
        desc="Venue location and capacity are specified",
        parent=venue_details,
        critical=True
    )

    us_loc = evaluator.add_leaf(
        id="p3_us_location",
        desc="Venue is in the United States with city and state provided",
        parent=loc_cap,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{venue}' is located in {city}, {state}, United States.",
        node=us_loc,
        sources=urls,
        additional_instruction="Confirm venue city and state on the official page.",
        extra_prerequisites=[url_provided]
    )

    cap_provided = evaluator.add_leaf(
        id="p3_capacity_provided",
        desc="Venue seating capacity number is specified",
        parent=loc_cap,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided page(s) explicitly state a numeric seating capacity for the venue '{venue}'.",
        node=cap_provided,
        sources=urls,
        additional_instruction="Look for a seat count or capacity number on an official venue or production page.",
        extra_prerequisites=[url_provided]
    )

    venue_verified = evaluator.add_leaf(
        id="p3_venue_verified",
        desc="Venue details are confirmed in reference source",
        parent=venue_details,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided page(s) confirm the venue name '{venue}' and its location (city and state).",
        node=venue_verified,
        sources=urls,
        additional_instruction="Look for explicit venue naming and address/location information.",
        extra_prerequisites=[url_provided]
    )

    # Timing (critical)
    p3_timing = evaluator.add_parallel(
        id="p3_march_timing",
        desc="Performance run dates include or overlap with March 2026",
        parent=node,
        critical=True
    )

    p3_dates_provided = evaluator.add_parallel(
        id="p3_dates_provided",
        desc="Performance dates or run period is specified",
        parent=p3_timing,
        critical=True
    )

    p3_date_format = evaluator.add_leaf(
        id="p3_date_format",
        desc="Dates are clearly stated (specific dates or date range)",
        parent=p3_dates_provided,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official page(s) for '{name}' at '{venue}' clearly state the performance date(s) or run range.",
        node=p3_date_format,
        sources=urls,
        additional_instruction="Look for a run range or individual performance dates.",
        extra_prerequisites=[url_provided]
    )

    p3_date_source = evaluator.add_leaf(
        id="p3_date_source",
        desc="Dates are verifiable from provided reference",
        parent=p3_dates_provided,
        critical=True
    )
    await evaluator.verify(
        claim="The performance dates are presented on the provided page(s).",
        node=p3_date_source,
        sources=urls,
        additional_instruction="There should be an explicit schedule presented.",
        extra_prerequisites=[url_provided]
    )

    p3_march_overlap = evaluator.add_leaf(
        id="p3_march_overlap",
        desc="At least one performance date falls within March 1-31, 2026",
        parent=p3_timing,
        critical=True
    )
    await evaluator.verify(
        claim="This event has at least one performance date between March 1 and March 31, 2026 (inclusive).",
        node=p3_march_overlap,
        sources=urls,
        additional_instruction="Confirm at least one listed performance date is in March 2026.",
        extra_prerequisites=[url_provided, p3_date_source]
    )

    # Ticketed (critical)
    p3_ticketed = evaluator.add_leaf(
        id="p3_ticketed_event",
        desc="Event is a ticketed performance with publicly available ticket sales",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Tickets are available to purchase (or clearly indicated) for this stage production on the provided page(s).",
        node=p3_ticketed,
        sources=urls,
        additional_instruction="Look for 'Tickets', 'Buy Tickets', or similar.",
        extra_prerequisites=[url_provided]
    )


async def verify_performance_4(
    evaluator: Evaluator,
    root,
    perf: Optional[Performance4]
) -> None:
    """
    Performance 4: Any additional ticketed performance meeting basic criteria + March 2026 date.
    """
    node = evaluator.add_parallel(
        id="performance_4",
        desc="Fourth performance: Any additional ticketed performance meeting basic criteria",
        parent=root,
        critical=False
    )
    name = (perf.name or "").strip() if perf else ""
    venue = (perf.venue_name or "").strip() if perf else ""
    city = (perf.city or "").strip() if perf else ""
    state = (perf.state or "").strip() if perf else ""
    urls = _clean_urls(perf.reference_urls if perf else [])

    refs = await _add_reference_group(evaluator, node, "p4", urls)
    url_provided = refs["url_provided"]

    # Venue details (critical)
    venue_details = evaluator.add_parallel(
        id="p4_venue_details",
        desc="Venue information including location and seating capacity is provided",
        parent=node,
        critical=True
    )

    loc_cap = evaluator.add_parallel(
        id="p4_location_capacity",
        desc="Venue location and capacity are specified",
        parent=venue_details,
        critical=True
    )

    us_loc = evaluator.add_leaf(
        id="p4_us_location",
        desc="Venue is in the United States with city and state provided",
        parent=loc_cap,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{venue}' is located in {city}, {state}, United States.",
        node=us_loc,
        sources=urls,
        additional_instruction="Confirm city and state on the official page(s).",
        extra_prerequisites=[url_provided]
    )

    cap_provided = evaluator.add_leaf(
        id="p4_capacity_provided",
        desc="Venue seating capacity number is specified",
        parent=loc_cap,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided page(s) explicitly state a numeric seating capacity for the venue '{venue}'.",
        node=cap_provided,
        sources=urls,
        additional_instruction="Look for a seat count or capacity number on an official page.",
        extra_prerequisites=[url_provided]
    )

    venue_verified = evaluator.add_leaf(
        id="p4_venue_verified",
        desc="Venue details are confirmed in reference source",
        parent=venue_details,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided page(s) confirm the venue name '{venue}' and its location (city and state).",
        node=venue_verified,
        sources=urls,
        additional_instruction="Look for explicit venue naming and address/location.",
        extra_prerequisites=[url_provided]
    )

    # Timing (critical)
    p4_timing = evaluator.add_parallel(
        id="p4_march_timing",
        desc="Performance occurs or has run dates overlapping with March 2026",
        parent=node,
        critical=True
    )

    p4_dates_provided = evaluator.add_parallel(
        id="p4_dates_provided",
        desc="Performance dates or run period is specified",
        parent=p4_timing,
        critical=True
    )

    p4_date_format = evaluator.add_leaf(
        id="p4_date_format",
        desc="Dates are clearly stated",
        parent=p4_dates_provided,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official page(s) for '{name}' at '{venue}' clearly state the performance date(s) or run period.",
        node=p4_date_format,
        sources=urls,
        additional_instruction="Look for explicit calendar or date range.",
        extra_prerequisites=[url_provided]
    )

    p4_date_source = evaluator.add_leaf(
        id="p4_date_source",
        desc="Dates are verifiable from provided reference",
        parent=p4_dates_provided,
        critical=True
    )
    await evaluator.verify(
        claim="The performance dates are presented on the provided page(s).",
        node=p4_date_source,
        sources=urls,
        additional_instruction="There should be explicit dates shown.",
        extra_prerequisites=[url_provided]
    )

    p4_march_overlap = evaluator.add_leaf(
        id="p4_march_overlap",
        desc="At least one performance date falls within March 1-31, 2026",
        parent=p4_timing,
        critical=True
    )
    await evaluator.verify(
        claim="This event has at least one performance date between March 1 and March 31, 2026 (inclusive).",
        node=p4_march_overlap,
        sources=urls,
        additional_instruction="Confirm at least one listed performance date is in March 2026.",
        extra_prerequisites=[url_provided, p4_date_source]
    )

    # Performance attributes (critical)
    attrs_group = evaluator.add_parallel(
        id="p4_performance_attributes",
        desc="Performance is a ticketed performing arts or entertainment event",
        parent=node,
        critical=True
    )

    p4_ticketed = evaluator.add_leaf(
        id="p4_ticketed",
        desc="Evidence of ticketed event with ticket information available",
        parent=attrs_group,
        critical=True
    )
    await evaluator.verify(
        claim="Tickets are available to purchase (or clearly indicated) for this event on the provided page(s).",
        node=p4_ticketed,
        sources=urls,
        additional_instruction="Look for 'Tickets', 'Buy Tickets', or similar.",
        extra_prerequisites=[url_provided]
    )

    p4_nature = evaluator.add_leaf(
        id="p4_performance_nature",
        desc="Event is identified as performing arts or entertainment production",
        parent=attrs_group,
        critical=True
    )
    await evaluator.verify(
        claim="This event is a performing arts or entertainment production (e.g., theater, concert, comedy, dance).",
        node=p4_nature,
        sources=urls,
        additional_instruction="Confirm on the page that it is a public performance event.",
        extra_prerequisites=[url_provided]
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
    Evaluate an answer for the 'four distinct ticketed performances in March 2026' task.
    """
    evaluator = Evaluator()
    # IMPORTANT: set root as NON-CRITICAL to allow non-critical children per framework constraints
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # independent verification branches per performance
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

    # 1) Extract structured info
    extracted: ExtractedPerformances = await evaluator.extract(
        prompt=prompt_extract_performances(),
        template_class=ExtractedPerformances,
        extraction_name="performances_extraction"
    )

    # 2) Build verification tree for each performance
    await verify_performance_1(evaluator, root, extracted.performance1 if extracted else None)
    await verify_performance_2(evaluator, root, extracted.performance2 if extracted else None)
    await verify_performance_3(evaluator, root, extracted.performance3 if extracted else None)
    await verify_performance_4(evaluator, root, extracted.performance4 if extracted else None)

    # 3) Return summary with the full verification tree
    return evaluator.get_summary()