import asyncio
import logging
import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "chicago_live_events_2026_spring"
TASK_DESCRIPTION = (
    "Find four distinct live entertainment events in Chicago, Illinois, that are scheduled between "
    "March 20, 2026, and May 20, 2026. The four events must collectively span at least three different "
    "entertainment categories from among: concerts, theater/Broadway performances, comedy shows, and sports events. "
    "Additionally, at least one of the four events must be held in a venue with a seating capacity of 1,500 or more.\n\n"
    "For each event, provide the following information:\n"
    "1. The specific performer, artist name, show title, or teams/participants\n"
    "2. The date and start time of the event\n"
    "3. The complete venue name and full street address\n"
    "4. The seating capacity of the venue\n"
    "5. The event category (concert, theater/Broadway, comedy, or sports)\n"
    "6. A direct URL link to an official ticket purchasing page for the event\n"
    "7. A reference URL that supports the event information"
)

DATE_RANGE_START = date(2026, 3, 20)
DATE_RANGE_END = date(2026, 5, 20)
MIN_CAPACITY = 1500
REQUIRED_CATEGORIES = {"concert", "theater/Broadway", "comedy", "sports"}


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------
class EventItem(BaseModel):
    title_or_performer: Optional[str] = None
    date: Optional[str] = None
    start_time: Optional[str] = None
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    venue_capacity: Optional[str] = None
    category: Optional[str] = None  # one of the 4 categories
    ticket_url: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    extra_urls: List[str] = Field(default_factory=list)  # any other URLs included in the answer for this event


class EventsExtraction(BaseModel):
    events: List[EventItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_events() -> str:
    return """
    Extract up to 6 live entertainment events provided in the answer. For each event, return the following fields:

    - title_or_performer: The specific performer, artist name, show title, or teams/participants as stated in the answer.
    - date: The event's calendar date as presented in the answer (e.g., "April 12, 2026", "Apr 12, 2026", "2026-04-12"). If a date range is given, extract the specific date associated with the event listing in the answer; do not return a range.
    - start_time: The start time of the event as presented in the answer (e.g., "7:00 PM", "19:30", include AM/PM if present). If multiple times are listed, choose the one associated with the specific performance mentioned.
    - venue_name: The full name of the venue as written in the answer.
    - venue_address: The full street address as written in the answer, including city and state (e.g., "123 Main St, Chicago, IL 60601").
    - venue_capacity: The seating capacity of the venue as written in the answer. Preserve the format (e.g., "20,000", "approximately 3,500", "1.5k", or ranges like "18,000–20,000").
    - category: Choose exactly one from: "concert", "theater/Broadway", "comedy", or "sports". If the answer uses synonyms, map them accordingly:
        • musicals/plays -> "theater/Broadway"
        • stand-up/comedy show/improv -> "comedy"
        • live music/gig -> "concert"
        • any sports games/matches -> "sports"
      Return the mapped standard label.
    - ticket_url: A direct official ticket purchasing URL for the event (venue box office, team/artist official, Ticketmaster, AXS, SeatGeek, MLB/Tickets.com, etc.). If multiple are given, pick the most direct official one. If not provided, set to null.
    - reference_urls: An array of all other URLs in the answer that support the event information (official listings, venue/artist pages, team schedule pages, etc.). Return an empty array if none are given.
    - extra_urls: Any additional URLs mentioned in the answer for this event (e.g., venue page, Wikipedia page with capacity), excluding the ticket_url itself. If none are provided, return an empty array.

    IMPORTANT:
    - Extract only what is explicitly present in the answer text; do not invent details.
    - Include full URLs with protocol (http/https).
    - Return a JSON object with an "events" array of EventItem objects with these fields.
    """


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
WEEKDAYS = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun"
]


def _strip_ordinals(s: str) -> str:
    if not s:
        return s
    return re.sub(r'(\d+)(st|nd|rd|th)', r'\1', s, flags=re.IGNORECASE)


def _remove_weekday_tokens(s: str) -> str:
    if not s:
        return s
    tokens = re.split(r'[,\s]+', s)
    filtered = [t for t in tokens if t.lower().strip(",") not in WEEKDAYS]
    return " ".join(filtered)


def parse_date_string(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    raw = s.strip()
    raw = _strip_ordinals(raw)
    raw = _remove_weekday_tokens(raw)

    # Try to isolate a likely date token if string contains extra content
    # e.g., "April 5, 2026 at 7pm" -> we keep the part before 'at'
    m = re.split(r'\bat\b', raw, flags=re.IGNORECASE)
    if m:
        raw = m[0].strip()

    # Quick patterns to try
    candidates = [raw]

    # Extract first YYYY-MM-DD if present
    iso_match = re.search(r'\b(20\d{2})-(\d{1,2})-(\d{1,2})\b', raw)
    if iso_match:
        candidates.insert(0, iso_match.group(0))

    # Extract Month D, YYYY patterns from text
    # Example: April 5, 2026 or Apr 5, 2026
    month_name = r'(January|February|March|April|May|June|July|August|September|October|November|December|' \
                 r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)'
    month_pat = re.search(rf'\b{month_name}\s+\d{{1,2}},\s*20\d{{2}}\b', raw, flags=re.IGNORECASE)
    if month_pat:
        candidates.insert(0, month_pat.group(0))

    # Try multiple known formats
    fmts = [
        "%Y-%m-%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%d %B %Y",
        "%d %b %Y",
    ]
    for cand in candidates:
        c = cand.strip().strip(",")
        for fmt in fmts:
            try:
                dt = datetime.strptime(c, fmt).date()
                return dt
            except Exception:
                continue

    # Fallback: try to extract numeric patterns like M/D/YYYY
    mdyyyy = re.search(r'\b(\d{1,2})/(\d{1,2})/(20\d{2})\b', raw)
    if mdyyyy:
        try:
            mm, dd, yyyy = mdyyyy.groups()
            return date(int(yyyy), int(mm), int(dd))
        except Exception:
            pass

    return None


def is_date_in_range(d: Optional[date], start: date, end: date) -> bool:
    if d is None:
        return False
    return start <= d <= end


def parse_capacity_to_int(cap_str: Optional[str]) -> Optional[int]:
    if not cap_str:
        return None
    s = cap_str.strip().lower()

    # Handle k notation like "1.5k"
    k_match = re.search(r'(\d+(?:\.\d+)?)\s*k\b', s)
    if k_match:
        val = float(k_match.group(1)) * 1000
        return int(round(val))

    # Extract all numeric groups (with commas or dots)
    nums = re.findall(r'\d[\d,\.]*', s)
    if not nums:
        return None

    # Convert to ints taking commas out; for decimals, round
    parsed = []
    for n in nums:
        n2 = n.replace(",", "")
        try:
            if "." in n2:
                parsed.append(int(round(float(n2))))
            else:
                parsed.append(int(n2))
        except Exception:
            continue

    if not parsed:
        return None

    # If a range was given like "18,000–20,000", choose the max value
    return max(parsed)


def normalize_category(cat: Optional[str]) -> Optional[str]:
    if not cat:
        return None
    c = cat.strip().lower()
    if c in {"concert", "concerts", "live music", "gig"}:
        return "concert"
    if c in {"theater", "theatre", "theater/broadway", "theatre/broadway", "broadway", "musical", "musicals", "play", "plays"}:
        return "theater/Broadway"
    if c in {"comedy", "stand-up", "stand up", "improv", "comedy show"}:
        return "comedy"
    if c in {"sports", "sport", "game", "match"}:
        return "sports"
    # If it already matches one of the required forms (case-insensitive)
    if c == "theater/broadway":
        return "theater/Broadway"
    if c in {rc.lower() for rc in REQUIRED_CATEGORIES}:
        # Return the correctly cased label if possible
        for rc in REQUIRED_CATEGORIES:
            if rc.lower() == c:
                return rc
    return None


def is_chicago_address(addr: Optional[str]) -> bool:
    if not addr:
        return False
    s = addr.lower()
    return ("chicago" in s) and (", il" in s or " illinois" in s)


def gather_sources(ev: EventItem) -> List[str]:
    urls: List[str] = []
    if ev.ticket_url and ev.ticket_url.strip().lower().startswith(("http://", "https://")):
        urls.append(ev.ticket_url.strip())
    for u in (ev.reference_urls or []):
        if isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")):
            urls.append(u.strip())
    for u in (ev.extra_urls or []):
        if isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")):
            urls.append(u.strip())
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def event_required_fields_present(ev: EventItem) -> bool:
    return all([
        bool(ev.title_or_performer and ev.title_or_performer.strip()),
        bool(ev.date and ev.date.strip()),
        bool(ev.start_time and ev.start_time.strip()),
        bool(ev.venue_name and ev.venue_name.strip()),
        bool(ev.venue_address and ev.venue_address.strip()),
        bool(ev.venue_capacity and ev.venue_capacity.strip()),
        bool(ev.category and normalize_category(ev.category) in REQUIRED_CATEGORIES),
        bool(ev.ticket_url and ev.ticket_url.strip()),
        bool(ev.reference_urls and len(ev.reference_urls) > 0),
    ])


# -----------------------------------------------------------------------------
# Verification for a single event
# -----------------------------------------------------------------------------
async def verify_single_event(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    ev: EventItem,
    index_zero_based: int
) -> Dict[str, Any]:
    idx = index_zero_based + 1
    ev_node = evaluator.add_parallel(
        id=f"Event_{idx}",
        desc=f"Event #{idx} verification (complete and correct)",
        parent=parent_node,
        critical=True  # All four events must pass to satisfy the task
    )

    # Gate: required fields present
    presence = event_required_fields_present(ev)
    presence_node = evaluator.add_custom_node(
        result=presence,
        id=f"Event_{idx}_Required_Fields",
        desc=f"Event #{idx} - All required fields are provided (performer/title, date, start time, venue name & address, capacity, category, ticket URL, and at least one reference URL)",
        parent=ev_node,
        critical=True
    )

    # Prepare sources
    sources_all = gather_sources(ev)
    ref_urls = ev.reference_urls or []
    ticket_url = ev.ticket_url or None

    # 1) Performer/Show
    perf_node = evaluator.add_leaf(
        id=f"Event_{idx}_Performer_Show",
        desc=f"Identifies the specific performer/artist/show/teams for Event #{idx}",
        parent=ev_node,
        critical=True
    )
    perf_claim = f"The event is for '{ev.title_or_performer}'. Treat this as the headliner performer, show title, or the teams/participants for this specific event."
    perf_add_ins = (
        "Verify that at least one provided page explicitly shows the same performer, artist, show title, "
        "or teams/participants. Allow minor formatting or naming variations (e.g., punctuation, case, or middle names)."
    )

    # 2) Date & Time
    dt_node = evaluator.add_leaf(
        id=f"Event_{idx}_Date_Time",
        desc=f"Provides the specific date and start time for Event #{idx}",
        parent=ev_node,
        critical=True
    )
    dt_claim = f"The event is scheduled on {ev.date} at {ev.start_time} local time (CT)."
    dt_add_ins = (
        "Verify that the page shows the same date and a matching start time (allowing minor variations like 7 vs 7:00 PM). "
        "Focus on the specific performance corresponding to the provided date."
    )

    # 3) Venue name & address
    venue_node = evaluator.add_leaf(
        id=f"Event_{idx}_Venue_Name_Address",
        desc=f"Provides the complete venue name and full street address for Event #{idx}",
        parent=ev_node,
        critical=True
    )
    venue_claim = f"The event takes place at '{ev.venue_name}', located at '{ev.venue_address}'."
    venue_add_ins = (
        "Verify both the venue name and full street address (including city and state). "
        "Minor formatting differences (e.g., abbreviations, punctuation, or presence/absence of ZIP code) are acceptable if clearly equivalent."
    )

    # 4) Venue capacity
    cap_node = evaluator.add_leaf(
        id=f"Event_{idx}_Venue_Capacity",
        desc=f"Provides the seating capacity of the venue for Event #{idx}",
        parent=ev_node,
        critical=True
    )
    cap_claim = f"The seating capacity of {ev.venue_name} is {ev.venue_capacity}."
    cap_add_ins = (
        "Check whether the cited capacity is correct for the venue. Accept reasonable approximations and typical configurations. "
        "For multipurpose venues (sports vs. concerts), allow ±10% tolerance. "
        "Use credible sources such as the official venue website or Wikipedia."
    )

    # 5) Category
    cat_norm = normalize_category(ev.category) or (ev.category or "")
    cat_node = evaluator.add_leaf(
        id=f"Event_{idx}_Category",
        desc=f"Correctly categorizes Event #{idx} (concert, theater/Broadway, comedy, or sports)",
        parent=ev_node,
        critical=True
    )
    cat_claim = f"This event qualifies as a '{cat_norm}' event."
    cat_add_ins = (
        "Use the page content to judge the category: "
        "concert = live music performance; theater/Broadway = musicals/plays; comedy = stand-up/improv; sports = competitive sports game/match. "
        "Allow synonyms; map appropriately."
    )

    # 6) Ticket purchasing page (official/direct)
    ticket_node = evaluator.add_leaf(
        id=f"Event_{idx}_Ticket_Link",
        desc=f"Provides a direct URL link to an official ticket purchasing page for Event #{idx}",
        parent=ev_node,
        critical=True
    )
    ticket_claim = (
        "This URL is a direct official ticket purchasing page for this specific event (not a resale aggregator). "
        "It should correspond to an authorized vendor such as the venue box office, Ticketmaster, AXS, SeatGeek, a team/artist official site, or MLB/Tickets.com, and include a clear path to purchase seats."
    )
    ticket_add_ins = (
        "Confirm that the page shows the specific event with purchase flow (e.g., Buy Tickets, seat map, or checkout). "
        "Reject generic listings without purchase functionality or unofficial resale marketplaces."
    )

    # 7) Reference URL(s) support
    ref_node = evaluator.add_leaf(
        id=f"Event_{idx}_Reference_URL",
        desc=f"Provides URL reference(s) supporting Event #{idx} information",
        parent=ev_node,
        critical=True
    )
    ref_claim = (
        "These page(s) provide official or credible information confirming the event details "
        f"(performer/show, date & time, and venue in Chicago) for '{ev.title_or_performer}'."
    )
    ref_add_ins = (
        "At least one URL must clearly confirm the key event details. Accept official sources (venue, artist, team) "
        "or reputable listings with full details. Reject pages that do not substantively confirm the event."
    )

    # Build batch verifications under the same parent; automatic preconditions will pick up presence_node (critical sibling)
    claims_and_sources: List[Tuple[str, Any, VerificationNode, Optional[str]]] = [
        (perf_claim, sources_all, perf_node, perf_add_ins),
        (dt_claim, sources_all, dt_node, dt_add_ins),
        (venue_claim, sources_all, venue_node, venue_add_ins),
        (cap_claim, sources_all, cap_node, cap_add_ins),
        (cat_claim, sources_all, cat_node, cat_add_ins),
        (ticket_claim, ticket_url, ticket_node, ticket_add_ins),
        (ref_claim, ref_urls, ref_node, ref_add_ins),
    ]
    await evaluator.batch_verify(claims_and_sources)

    return {
        "event_node": ev_node,
        "presence_node": presence_node,
        "all_sources": sources_all,
    }


# -----------------------------------------------------------------------------
# Aggregated constraint checks (custom nodes)
# -----------------------------------------------------------------------------
def compute_event_diversity_ok(first_four: List[EventItem]) -> bool:
    cats = set()
    for ev in first_four:
        cn = normalize_category(ev.category)
        if cn in REQUIRED_CATEGORIES:
            cats.add(cn)
    return len(cats) >= 3


def compute_date_range_ok(first_four: List[EventItem]) -> bool:
    # All four events must have dates within the specified range
    for ev in first_four:
        d = parse_date_string(ev.date)
        if not is_date_in_range(d, DATE_RANGE_START, DATE_RANGE_END):
            return False
    return True


def compute_capacity_requirement_ok(first_four: List[EventItem]) -> bool:
    # At least one event capacity >= MIN_CAPACITY
    for ev in first_four:
        cap = parse_capacity_to_int(ev.venue_capacity)
        if cap is not None and cap >= MIN_CAPACITY:
            return True
    return False


def compute_geographic_ok(first_four: List[EventItem]) -> bool:
    # All four events should be in Chicago, IL
    return all(is_chicago_address(ev.venue_address) for ev in first_four)


# -----------------------------------------------------------------------------
# Main evaluation entry
# -----------------------------------------------------------------------------
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
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

    # Extract events
    extracted: EventsExtraction = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction"
    )

    # Keep exactly first four; pad with empty if fewer
    events = list(extracted.events[:4])
    while len(events) < 4:
        events.append(EventItem())

    # Record ground-truth constraints to the summary
    evaluator.add_ground_truth({
        "date_range_start": str(DATE_RANGE_START),
        "date_range_end": str(DATE_RANGE_END),
        "required_categories_min_distinct": 3,
        "capacity_min_requirement": MIN_CAPACITY,
        "required_categories": sorted(list(REQUIRED_CATEGORIES))
    })

    # Group node to require all 4 events to pass
    events_group = evaluator.add_parallel(
        id="Events_Group",
        desc="All four events must be complete and correct",
        parent=root,
        critical=True
    )

    # Verify each event
    presence_nodes: List[VerificationNode] = []
    for i in range(4):
        result_info = await verify_single_event(evaluator, events_group, events[i], i)
        presence_nodes.append(result_info["presence_node"])

    # Aggregated constraints (critical)
    # 1) Event diversity across categories
    evaluator.add_custom_node(
        result=compute_event_diversity_ok(events),
        id="Event_Diversity",
        desc="The four events collectively span at least three different categories (concert, theater/Broadway, comedy, sports)",
        parent=root,
        critical=True
    )

    # 2) Date range compliance
    evaluator.add_custom_node(
        result=compute_date_range_ok(events),
        id="Date_Range_Compliance",
        desc=f"All four events occur within the specified date range ({DATE_RANGE_START} to {DATE_RANGE_END})",
        parent=root,
        critical=True
    )

    # 3) Capacity requirement
    evaluator.add_custom_node(
        result=compute_capacity_requirement_ok(events),
        id="Capacity_Requirement",
        desc=f"At least one event is in a venue with seating capacity of {MIN_CAPACITY} or more",
        parent=root,
        critical=True
    )

    # 4) Geographic compliance (Chicago, IL)
    evaluator.add_custom_node(
        result=compute_geographic_ok(events),
        id="Geographic_Compliance",
        desc="All four events are located in Chicago, Illinois",
        parent=root,
        critical=True
    )

    return evaluator.get_summary()