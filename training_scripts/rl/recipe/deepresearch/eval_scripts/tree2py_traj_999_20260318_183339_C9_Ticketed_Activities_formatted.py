import asyncio
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nyc_concerts_capacity_15k_20k"
TASK_DESCRIPTION = (
    "Find three upcoming live music concerts scheduled in New York City within the next two months (by May 18, 2026), "
    "where each concert must be held at a venue with a seating capacity between 15,000 and 20,000 people. For each concert, provide the following information: "
    "(1) Event Details: Artist/band name, concert date and time, and an official URL confirming the event. "
    "(2) Venue Information: Venue name, exact seating capacity (which must be between 15,000 and 20,000), full address in New York City, and an official URL with venue details. "
    "(3) Ticket Pricing: Three different pricing tiers (lowest, mid-range, and highest price options) with the tier name/designation and price in USD for each tier, along with URL references for pricing information. "
    "(4) Accessibility: Confirmation that wheelchair accessible seating is available at the venue, information about whether accessible seating is offered across multiple price tiers, and a URL reference for accessibility information. "
    "(5) Public Transportation: The nearest subway or public transit station to the venue, approximate walking distance from the station to the venue, and a URL reference for transportation information. "
    "All information must be verifiable through official sources such as venue websites, ticketing platforms, or event promoter pages."
)

# Evaluation-time window (frozen for deterministic judging)
EVAL_TODAY = date(2026, 3, 22)
DEADLINE_DATE = date(2026, 5, 18)
CAPACITY_MIN = 15000
CAPACITY_MAX = 20000

NYC_BOROUGHS = ["New York", "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
NYC_ABBREV = ["NY", "NYC"]


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class PriceTier(BaseModel):
    tier_name: Optional[str] = None
    price_usd: Optional[str] = None  # Keep as string to tolerate formats like "$150", "USD 150", "150.00"
    notes: Optional[str] = None


class TicketPricing(BaseModel):
    tiers: List[PriceTier] = Field(default_factory=list)  # Expect 3, but tolerate variable count
    urls: List[str] = Field(default_factory=list)  # Official/authoritative pricing references


class AccessibilityInfo(BaseModel):
    wheelchair_accessible: Optional[str] = None  # e.g., "yes", "available", "wheelchair seating available"
    accessible_across_multiple_price_tiers: Optional[str] = None  # textual confirmation
    urls: List[str] = Field(default_factory=list)


class TransportationInfo(BaseModel):
    nearest_station: Optional[str] = None
    walking_distance: Optional[str] = None  # e.g., "5-minute walk", "0.3 miles (6 min)"
    urls: List[str] = Field(default_factory=list)


class EventDetails(BaseModel):
    artist_or_band: Optional[str] = None
    date: Optional[str] = None               # Prefer explicit date string
    time: Optional[str] = None               # Prefer explicit time string
    datetime_raw: Optional[str] = None       # If the answer combined date & time, capture raw text
    event_url: Optional[str] = None          # Official/authoritative URL confirming the event
    is_live_music_concert_flag: Optional[str] = None  # If answer explicitly states it's a concert, capture text


class VenueInfo(BaseModel):
    venue_name: Optional[str] = None
    address_full: Optional[str] = None
    seating_capacity_exact: Optional[str] = None  # keep as string; we'll parse digits
    urls: List[str] = Field(default_factory=list)  # Official/authoritative venue pages (about, plan your visit, etc.)


class ConcertItem(BaseModel):
    event: Optional[EventDetails] = None
    venue: Optional[VenueInfo] = None
    pricing: Optional[TicketPricing] = None
    accessibility: Optional[AccessibilityInfo] = None
    transportation: Optional[TransportationInfo] = None


class ConcertsExtraction(BaseModel):
    concerts: List[ConcertItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_concerts() -> str:
    return """
Extract all concert items presented in the answer in the original order. For each item, fill the following fields as completely as possible, using EXACT strings as written in the answer. Do NOT infer or invent information. If something is missing, set it to null or an empty list.

Return JSON with a top-level field "concerts": an array of objects, each with the structure below.

For each concert object:
- event:
  - artist_or_band: the performing artist or band name (string)
  - date: the specific concert date string as written (string; e.g., "April 28, 2026")
  - time: the specific start time string as written (string; e.g., "7:30 PM"), if provided
  - datetime_raw: if date/time appear together in one phrase, copy that phrase here exactly (string)
  - event_url: an official/authoritative URL confirming the event (string URL)
  - is_live_music_concert_flag: if the answer text explicitly says "concert" or equivalent confirmation, capture that text; else null
- venue:
  - venue_name: official venue name (string)
  - address_full: full venue address as written (string)
  - seating_capacity_exact: exact seating capacity (string as written, e.g., "19,000")
  - urls: list of official/authoritative venue URLs supporting details (array of strings)
- pricing:
  - tiers: array of up to 3 items (lowest, mid-range, highest) with fields:
      - tier_name: designation of the tier (string; e.g., "GA", "Lower Bowl", "VIP")
      - price_usd: price as written in USD (string; e.g., "$85", "USD 225")
      - notes: any clarifying note in the answer for this tier (string or null)
  - urls: list of official/authoritative URL(s) supporting pricing (array of strings)
- accessibility:
  - wheelchair_accessible: confirmation text for wheelchair accessible seating (string or null)
  - accessible_across_multiple_price_tiers: confirmation text about accessible seating offered across multiple price tiers (string or null)
  - urls: list of official/authoritative URL(s) for accessibility information (array of strings)
- transportation:
  - nearest_station: nearest subway/public transit station name as written (string)
  - walking_distance: approximate walking distance as written (string)
  - urls: list of official/authoritative URL(s) for transportation info (array of strings)

IMPORTANT:
- Only extract URLs explicitly present in the answer. If a URL is missing a protocol, prepend "http://".
- Do not normalize names or rewrite prices; copy what the answer states.
- If the answer lists more than three concerts, extract all; if fewer than three, extract what is present.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _digits_to_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    import re
    digits = re.findall(r"\d+", s.replace(",", ""))
    if not digits:
        return None
    try:
        # Sometimes capacity includes ranges; take first number
        return int(digits[0])
    except Exception:
        return None


def _parse_date_str(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    s_clean = s.strip()
    fmts = [
        "%B %d, %Y",   # April 15, 2026
        "%b %d, %Y",   # Apr 15, 2026
        "%Y-%m-%d",    # 2026-04-15
        "%m/%d/%Y",    # 04/15/2026
        "%m/%d/%y",    # 04/15/26
        "%d %B %Y",    # 15 April 2026
        "%d %b %Y",    # 15 Apr 2026
    ]
    for f in fmts:
        try:
            return datetime.strptime(s_clean, f).date()
        except Exception:
            continue
    # Try to extract month-name day, year loosely
    try:
        import re
        pattern = r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})"
        m = re.search(pattern, s_clean)
        if m:
            guess = f"{m.group(1)} {m.group(2)}, {m.group(3)}"
            return datetime.strptime(guess, "%B %d, %Y").date()
    except Exception:
        pass
    return None


def _is_date_within_window(d: Optional[date]) -> bool:
    if not d:
        return False
    return EVAL_TODAY <= d <= DEADLINE_DATE


def _is_nyc_address(addr: Optional[str]) -> bool:
    if not addr:
        return False
    a = addr.lower()
    borough_hit = any(b.lower() in a for b in NYC_BOROUGHS)
    abbrev_hit = any(f", {abbrev.lower()}" in a for abbrev in NYC_ABBREV) or " new york, " in a
    # zip code starting with 10xxx (Manhattan) or 11xxx (Queens), 112xx (Brooklyn), 104xx (Bronx), 103xx (Staten Island)
    import re
    zip_hit = bool(re.search(r"\b10\d{3}\b|\b11\d{3}\b|\b112\d{2}\b|\b104\d{2}\b|\b103\d{2}\b", a))
    return borough_hit or abbrev_hit or zip_hit


def _non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _gather_urls(*url_lists: List[Optional[str] | List[str] | None]) -> List[str]:
    urls: List[str] = []
    for block in url_lists:
        if not block:
            continue
        if isinstance(block, list):
            for u in block:
                if isinstance(u, str) and _non_empty(u):
                    urls.append(u.strip())
        elif isinstance(block, str) and _non_empty(block):
            urls.append(block.strip())
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _tier_or_placeholder(tiers: List[PriceTier], index: int) -> PriceTier:
    if 0 <= index < len(tiers):
        return tiers[index]
    return PriceTier()  # empty placeholder


# --------------------------------------------------------------------------- #
# Verification builder for a single concert                                   #
# --------------------------------------------------------------------------- #
async def verify_concert(
    evaluator: Evaluator,
    parent_node,
    concert: ConcertItem,
    idx: int,
) -> None:
    """
    Build verification subtree for one concert item.
    idx is 1-based index (1, 2, 3).
    """
    cnode = evaluator.add_parallel(
        id=f"concert_{idx}",
        desc=f"Concert {idx} (one qualifying concert item)",
        parent=parent_node,
        critical=False  # A single concert failing shouldn't fail the whole set
    )

    event = concert.event or EventDetails()
    venue = concert.venue or VenueInfo()
    pricing = concert.pricing or TicketPricing()
    accessibility = concert.accessibility or AccessibilityInfo()
    transport = concert.transportation or TransportationInfo()

    event_urls = _gather_urls(event.event_url)
    venue_urls = _gather_urls(venue.urls)
    pricing_urls = _gather_urls(pricing.urls)
    accessibility_urls = _gather_urls(accessibility.urls)
    transport_urls = _gather_urls(transport.urls)

    # -------------------- Event details -------------------- #
    event_node = evaluator.add_parallel(
        id=f"event_details_{idx}",
        desc=f"Event details for Concert {idx}",
        parent=cnode,
        critical=True
    )

    # 1) Event official URL (gating other event checks)
    ev_official_leaf = evaluator.add_leaf(
        id=f"event_official_url_{idx}",
        desc=f"Provides an official/authoritative URL confirming the event details",
        parent=event_node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is an official or authoritative source (venue website, primary ticketing platform, or event promoter) that confirms the event details (artist and date/time).",
        node=ev_official_leaf,
        sources=event_urls,
        additional_instruction="Treat domains like the venue's official site, Ticketmaster, AXS, Live Nation, or the promoter's official site as authoritative. The page must reference the specific event."
    )

    # 2) Live music concert confirmation
    live_leaf = evaluator.add_leaf(
        id=f"is_live_music_concert_{idx}",
        desc=f"The event is a live music concert (not a non-music event)",
        parent=event_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page indicates the event is a live music concert (a live performance by an artist/band).",
        node=live_leaf,
        sources=event_urls,
        additional_instruction="Look for wording like 'concert', 'tour', 'live performance', or otherwise clear indicators that the event is a music concert."
    )

    # 3) Artist/band name verification
    artist_text = event.artist_or_band or ""
    artist_leaf = evaluator.add_leaf(
        id=f"artist_or_band_{idx}",
        desc=f"Provides the performing artist/band name",
        parent=event_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The performing artist or band for this event is '{artist_text}'.",
        node=artist_leaf,
        sources=event_urls,
        additional_instruction="Accept reasonable variants or capitalization. The page should clearly list the main headliner/performer."
    )

    # 4) Date/time supported by URL
    # Build a human-readable date/time string for the claim
    dt_components = []
    if _non_empty(event.date):
        dt_components.append(event.date.strip())
    if _non_empty(event.time):
        dt_components.append(event.time.strip())
    dt_text = " at ".join(dt_components) if len(dt_components) == 2 else (dt_components[0] if dt_components else (event.datetime_raw or ""))

    dt_supported_leaf = evaluator.add_leaf(
        id=f"date_time_within_window_{idx}",
        desc=f"Provides a specific concert date and start time, supported by an official page",
        parent=event_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event page states that the concert occurs on '{dt_text}'.",
        node=dt_supported_leaf,
        sources=event_urls,
        additional_instruction="Focus on confirming both a specific calendar date and a start time for the concert."
    )

    # 5) Date is within allowed window (logic check)
    # Parse date if possible and check bounds
    parsed_date = _parse_date_str(event.date or event.datetime_raw or "")
    date_in_window = _is_date_within_window(parsed_date)
    evaluator.add_custom_node(
        result=date_in_window,
        id=f"date_within_window_logic_{idx}",
        desc=f"The provided event date is within the required window ({EVAL_TODAY.isoformat()} to {DEADLINE_DATE.isoformat()})",
        parent=event_node,
        critical=True
    )

    # -------------------- Venue information -------------------- #
    venue_node = evaluator.add_parallel(
        id=f"venue_info_{idx}",
        desc=f"Venue information for Concert {idx}",
        parent=cnode,
        critical=True
    )

    # Gating: venue official URL(s)
    venue_official_leaf = evaluator.add_leaf(
        id=f"venue_official_url_{idx}",
        desc=f"Provides an official/authoritative URL supporting venue details (e.g., venue site/about page)",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page is an official or authoritative venue source that provides venue details such as name, address, or capacity.",
        node=venue_official_leaf,
        sources=venue_urls,
        additional_instruction="Venue-owned domains are preferred. If not present, an authoritative promoter/arena operator or primary ticketing page with venue details can be acceptable."
    )

    # Venue name verification
    vname = venue.venue_name or ""
    venue_name_leaf = evaluator.add_leaf(
        id=f"venue_name_{idx}",
        desc=f"Provides the official venue name",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue name is '{vname}'.",
        node=venue_name_leaf,
        sources=_gather_urls(venue_urls, event_urls),
        additional_instruction="Confirm the exact official naming of the venue. Allow minor capitalization or punctuation differences."
    )

    # Venue address in NYC
    vaddr = venue.address_full or ""
    venue_addr_leaf = evaluator.add_leaf(
        id=f"venue_address_in_nyc_{idx}",
        desc=f"Provides the full venue address and confirms it is in New York City",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue's full address is '{vaddr}', and it is located in New York City.",
        node=venue_addr_leaf,
        sources=_gather_urls(venue_urls, event_urls),
        additional_instruction="The page should clearly show the address within New York City (any of its five boroughs)."
    )

    # Supplemental logic check for NYC location based on the extracted string
    evaluator.add_custom_node(
        result=_is_nyc_address(vaddr),
        id=f"venue_address_nyc_logic_{idx}",
        desc="The extracted address text indicates a New York City location (string-level heuristic).",
        parent=venue_node,
        critical=True
    )

    # Capacity exact supported by URL
    cap_text = venue.seating_capacity_exact or ""
    capacity_supported_leaf = evaluator.add_leaf(
        id=f"capacity_exact_and_in_range_{idx}",
        desc=f"Provides the exact seating capacity supported by an official venue URL",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The seating capacity of '{vname}' is '{cap_text}'.",
        node=capacity_supported_leaf,
        sources=venue_urls,
        additional_instruction="The venue page (or another authoritative venue source) should specify the venue's capacity for concerts or general seating capacity."
    )

    # Capacity numeric-range logic check
    cap_num = _digits_to_int(cap_text)
    cap_in_range = (cap_num is not None) and (CAPACITY_MIN <= cap_num <= CAPACITY_MAX)
    evaluator.add_custom_node(
        result=cap_in_range,
        id=f"capacity_in_range_logic_{idx}",
        desc=f"The venue capacity is an integer between {CAPACITY_MIN} and {CAPACITY_MAX} inclusive",
        parent=venue_node,
        critical=True
    )

    # -------------------- Ticket pricing (three tiers) -------------------- #
    pricing_node = evaluator.add_parallel(
        id=f"ticket_pricing_{idx}",
        desc=f"Ticket pricing for Concert {idx} (three tiers)",
        parent=cnode,
        critical=True
    )

    # Gating: pricing URL(s)
    pricing_url_leaf = evaluator.add_leaf(
        id=f"pricing_url_{idx}",
        desc=f"Provides official/authoritative URL reference(s) supporting the three pricing tiers",
        parent=pricing_node,
        critical=True
    )
    await evaluator.verify(
        claim="These pages are official or authoritative and provide ticket pricing information for the event.",
        node=pricing_url_leaf,
        sources=pricing_urls,
        additional_instruction="Primary ticketing providers (e.g., Ticketmaster, AXS) or venue-operated ticketing pages are preferred. Pages must contain price information."
    )

    # Retrieve up to 3 tiers (lowest, mid, highest) as provided
    lowest = _tier_or_placeholder(pricing.tiers, 0)
    mid = _tier_or_placeholder(pricing.tiers, 1)
    high = _tier_or_placeholder(pricing.tiers, 2)

    # Lowest tier
    low_leaf = evaluator.add_leaf(
        id=f"lowest_tier_{idx}",
        desc="Lowest price tier: tier designation + price in USD",
        parent=pricing_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"There is a ticket tier '{lowest.tier_name}' priced at '{lowest.price_usd}' (USD) for this event.",
        node=low_leaf,
        sources=_gather_urls(pricing_urls, event_urls),
        additional_instruction="Confirm that this specific tier name (or equivalent) and price appear on the referenced page(s)."
    )

    # Mid tier
    mid_leaf = evaluator.add_leaf(
        id=f"mid_tier_{idx}",
        desc="Mid-range price tier: tier designation + price in USD",
        parent=pricing_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"There is a ticket tier '{mid.tier_name}' priced at '{mid.price_usd}' (USD) for this event.",
        node=mid_leaf,
        sources=_gather_urls(pricing_urls, event_urls),
        additional_instruction="Confirm the mid-range tier and its price appear on authoritative ticketing/venue pages."
    )

    # Highest tier
    high_leaf = evaluator.add_leaf(
        id=f"highest_tier_{idx}",
        desc="Highest price tier: tier designation + price in USD",
        parent=pricing_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"There is a ticket tier '{high.tier_name}' priced at '{high.price_usd}' (USD) for this event.",
        node=high_leaf,
        sources=_gather_urls(pricing_urls, event_urls),
        additional_instruction="Confirm the top-tier (e.g., VIP/premium) and price on official/authoritative pages."
    )

    # -------------------- Accessibility -------------------- #
    access_node = evaluator.add_parallel(
        id=f"accessibility_{idx}",
        desc=f"Accessibility requirements for Concert {idx} venue",
        parent=cnode,
        critical=True
    )

    access_url_leaf = evaluator.add_leaf(
        id=f"accessibility_url_{idx}",
        desc="Provides an official/authoritative URL reference for accessibility information",
        parent=access_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page is an official or authoritative source that provides accessibility information for the venue/event.",
        node=access_url_leaf,
        sources=_gather_urls(accessibility_urls, venue_urls),
        additional_instruction="Venue 'Accessibility' or 'ADA' pages are ideal. Promoter/ticketing pages with clear accessibility policies are acceptable."
    )

    wheelchair_leaf = evaluator.add_leaf(
        id=f"wheelchair_accessible_seating_{idx}",
        desc="Confirms wheelchair accessible seating is available",
        parent=access_node,
        critical=True
    )
    await evaluator.verify(
        claim="Wheelchair accessible seating is available at the venue for this event.",
        node=wheelchair_leaf,
        sources=_gather_urls(accessibility_urls, venue_urls, event_urls),
        additional_instruction="Look for explicit statements about accessible/wheelchair seating policies."
    )

    multi_tiers_leaf = evaluator.add_leaf(
        id=f"accessible_seating_multiple_price_tiers_{idx}",
        desc="Confirms accessible seating is offered across multiple price tiers",
        parent=access_node,
        critical=True
    )
    await evaluator.verify(
        claim="Accessible seating is offered across multiple ticket price tiers (not limited to a single price tier).",
        node=multi_tiers_leaf,
        sources=_gather_urls(accessibility_urls, pricing_urls, venue_urls),
        additional_instruction="Prefer explicit statements. If multiple sections/tiers list accessible seating availability, that suffices."
    )

    # -------------------- Public transportation -------------------- #
    transit_node = evaluator.add_parallel(
        id=f"public_transportation_{idx}",
        desc=f"Public transportation information for Concert {idx} venue",
        parent=cnode,
        critical=True
    )

    transit_url_leaf = evaluator.add_leaf(
        id=f"transportation_url_{idx}",
        desc="Provides an official/authoritative URL reference for transportation information",
        parent=transit_node,
        critical=True
    )
    await evaluator.verify(
        claim="This page is an official or authoritative source that provides public transportation guidance for reaching the venue.",
        node=transit_url_leaf,
        sources=_gather_urls(transport_urls, venue_urls),
        additional_instruction="Venue 'Getting Here' or 'Plan Your Visit' pages are ideal. Official transit authority pages are also acceptable."
    )

    nearest_station_leaf = evaluator.add_leaf(
        id=f"nearest_station_{idx}",
        desc="Provides the nearest subway/public transit station name",
        parent=transit_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The nearest public transit station to the venue is '{(transport.nearest_station or '').strip()}'.",
        node=nearest_station_leaf,
        sources=_gather_urls(transport_urls, venue_urls),
        additional_instruction="Confirm the specific station name on the referenced page(s)."
    )

    walking_distance_leaf = evaluator.add_leaf(
        id=f"walking_distance_{idx}",
        desc="Provides approximate walking distance from the station to the venue",
        parent=transit_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The approximate walking distance from the nearest station to the venue is '{(transport.walking_distance or '').strip()}'.",
        node=walking_distance_leaf,
        sources=_gather_urls(transport_urls, venue_urls),
        additional_instruction="Confirm the distance or a reasonable walk-time estimate as listed on the page(s)."
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
    Evaluate an answer for the NYC concerts (15k–20k capacity) task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root: allow independent credit across items and set-level checks
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

    # Extract structured concert list from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_concerts(),
        template_class=ConcertsExtraction,
        extraction_name="concerts_extraction"
    )

    concerts: List[ConcertItem] = extracted.concerts if extracted and extracted.concerts else []

    # Add custom information about evaluation parameters for transparency
    evaluator.add_custom_info(
        {
            "eval_today": EVAL_TODAY.isoformat(),
            "deadline_date": DEADLINE_DATE.isoformat(),
            "capacity_range_inclusive": [CAPACITY_MIN, CAPACITY_MAX]
        },
        info_type="evaluation_parameters",
        info_name="evaluation_parameters"
    )

    # -------------------- Set-level requirements (critical) -------------------- #
    set_node = evaluator.add_parallel(
        id="set_requirements",
        desc="Set-level requirements for the full answer",
        parent=root,
        critical=True
    )

    # Exactly three items
    exactly_three = evaluator.add_custom_node(
        result=(len(concerts) == 3),
        id="exactly_three_concert_items",
        desc="Provides exactly three concert items (not fewer or more)",
        parent=set_node,
        critical=True
    )

    # Distinct events (use event_url if available; else artist+date+venue tuple)
    def _concert_key(c: ConcertItem) -> Tuple[str, str, str]:
        e = c.event or EventDetails()
        v = c.venue or VenueInfo()
        key_url = (e.event_url or "").strip().lower()
        if key_url:
            return (key_url, "", "")
        a = (e.artist_or_band or "").strip().lower()
        d = (e.date or e.datetime_raw or "").strip().lower()
        vn = (v.venue_name or "").strip().lower()
        return (a, d, vn)

    keys = [_concert_key(c) for c in concerts]
    distinct_count = len(set(keys))
    concerts_are_distinct = evaluator.add_custom_node(
        result=(distinct_count == len(concerts) and len(concerts) > 0),
        id="concerts_are_distinct",
        desc="The three concert items are distinct events (not duplicates of the same event)",
        parent=set_node,
        critical=True
    )

    # -------------------- Per-concert verification -------------------- #
    # Per "Final Reminder": evaluate only the first 3 items if more are present; if fewer, handle gracefully.
    concerts_to_check = concerts[:3] if len(concerts) >= 3 else concerts + [ConcertItem()] * (3 - len(concerts))

    # Build subtrees for up to three concerts
    for i, c in enumerate(concerts_to_check, start=1):
        await verify_concert(evaluator, root, c, i)

    # Return standardized summary
    return evaluator.get_summary()