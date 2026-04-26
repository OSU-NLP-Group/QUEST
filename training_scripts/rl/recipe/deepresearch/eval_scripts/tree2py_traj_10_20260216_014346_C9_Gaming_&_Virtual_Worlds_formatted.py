import asyncio
import logging
import re
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gaming_conventions_2026"
TASK_DESCRIPTION = """
A small independent video game development studio is planning to showcase their new PC game at major North American gaming conventions in 2026. They need to identify exactly 4 conventions or gaming events that meet ALL of the following criteria:

1. The event must take place entirely between March 1, 2026 and August 31, 2026 (inclusive)
2. The event must be located in the continental United States
3. The event must have an expected attendance of at least 30,000 people
4. The event must focus on video gaming, PC gaming, gaming industry networking, or prominently feature indie/video games (events that focus exclusively on tabletop games, card games, or fighting game tournaments do NOT qualify)

For each of the 4 qualifying events, provide:
- The exact start date and end date (including month, day, and year)
- The host city and state
- The official venue name
- The expected or confirmed attendance number
- A description of the event's primary focus or type
- Reference URLs confirming each piece of information

All information must be verifiable through official event websites, convention schedules, or reliable news sources.
"""

DATE_RANGE_START = date(2026, 3, 1)
DATE_RANGE_END = date(2026, 8, 31)

# Contiguous US states + DC
CONTIGUOUS_STATE_ABBR = {
    'AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','IA','ID','IL','IN','KS','KY','LA','MA','MD','ME','MI','MN',
    'MO','MS','MT','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VA',
    'VT','WA','WI','WV','WY'
}
STATE_NAME_TO_ABBR = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO","Connecticut":"CT",
    "Delaware":"DE","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA",
    "Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI",
    "Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH",
    "New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH",
    "Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD",
    "Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA","West Virginia":"WV",
    "Wisconsin":"WI","Wyoming":"WY","District of Columbia":"DC","Washington, D.C.":"DC","Washington DC":"DC"
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    """Information for a single event as extracted from the answer."""
    name: Optional[str] = None

    start_date: Optional[str] = None
    start_date_sources: List[str] = Field(default_factory=list)

    end_date: Optional[str] = None
    end_date_sources: List[str] = Field(default_factory=list)

    city: Optional[str] = None
    state: Optional[str] = None
    location_sources: List[str] = Field(default_factory=list)

    venue: Optional[str] = None
    venue_sources: List[str] = Field(default_factory=list)

    attendance: Optional[str] = None
    attendance_sources: List[str] = Field(default_factory=list)

    focus_description: Optional[str] = None
    focus_sources: List[str] = Field(default_factory=list)

    primary_source_url: Optional[str] = None


class EventsExtraction(BaseModel):
    """List of up to 4 events extracted from the answer."""
    events: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract up to FOUR qualifying gaming events mentioned in the answer. Each event should include the following fields exactly as presented in the answer (do not invent or normalize beyond what the answer states):
    - name: The official event name (string)
    - start_date: The exact start date of the 2026 edition (e.g., "June 3, 2026"). Do not provide a range here; provide a single date for the start.
    - start_date_sources: List of URLs that specifically support the start date
    - end_date: The exact end date of the 2026 edition (e.g., "June 6, 2026"). Do not provide a range here; provide a single date for the end.
    - end_date_sources: List of URLs that specifically support the end date
    - city: The host city (string)
    - state: The host state (full name or two-letter abbreviation)
    - location_sources: List of URLs that support city and state
    - venue: The official venue name
    - venue_sources: List of URLs that support the venue
    - attendance: The expected or confirmed attendance number as written in the answer (e.g., "40,000", "35k", "30,000+")
    - attendance_sources: List of URLs that support the attendance figure
    - focus_description: The primary focus or type of the event as described (e.g., "video gaming expo", "PC gaming", "gaming industry networking", "indie games showcase")
    - focus_sources: List of URLs that support the event's focus/type
    - primary_source_url: A main official or reliable page for the 2026 event
    
    SPECIAL RULES:
    - Only extract URLs that are explicitly present in the answer text. Do not infer or create URLs.
    - Always include full URLs with protocol (http:// or https://). If the answer omitted protocol, prepend http:// as needed.
    - If the answer provides more than 4 events, include only the first 4 as they appear.
    - If a field is missing, set it to null (or an empty list for source lists).
    """


# --------------------------------------------------------------------------- #
# Helper functions for value checks                                           #
# --------------------------------------------------------------------------- #
def try_parse_date_str(date_str: Optional[str]) -> Optional[date]:
    """Try to parse a date string into a date object using several common formats. Also handle range-like strings by splitting."""
    if not date_str:
        return None
    s = date_str.strip()
    # If contains an en dash/em dash/hyphen indicating a range, try to take the first number for start, last for end
    # For robustness, just remove everything after the first dash for start or before last dash for end is handled outside.
    # Here we try multiple patterns.
    candidates = [s]

    # If pattern includes a dash-like, collect left and right segments as possible candidates
    if any(ch in s for ch in ["–", "—", "-"]):
        parts = re.split(r"[–—-]", s)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            # Try to reconstruct with the year if only month/day provided
            # Use first part and last part as potential single-date strings
            candidates.insert(0, parts[0])
            candidates.append(parts[-1])
            # Append year to first part if missing and we see a 2026 in the string
            if "2026" in s and "2026" not in parts[0]:
                candidates.insert(0, f"{parts[0]}, 2026")

    fmts = [
        "%B %d, %Y",  # June 3, 2026
        "%b %d, %Y",  # Jun 3, 2026
        "%m/%d/%Y",   # 06/03/2026
        "%Y-%m-%d",   # 2026-06-03
        "%B %d %Y",   # June 3 2026
        "%b %d %Y",   # Jun 3 2026
    ]
    for cand in candidates:
        for fmt in fmts:
            try:
                return datetime.strptime(cand, fmt).date()
            except Exception:
                continue
    # Try to extract month/day/year with regex
    m = re.search(r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),?\s+(?P<year>2026)", s)
    if m:
        try:
            return datetime.strptime(f"{m.group('month')} {m.group('day')}, {m.group('year')}", "%B %d, %Y").date()
        except Exception:
            try:
                return datetime.strptime(f"{m.group('month')} {m.group('day')}, {m.group('year')}", "%b %d, %Y").date()
            except Exception:
                pass
    m2 = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m2:
        try:
            return datetime.strptime(m2.group(0), "%Y-%m-%d").date()
        except Exception:
            pass
    return None


def attendance_to_int(att: Optional[str]) -> Optional[int]:
    """Parse attendance string into an integer if possible. Supports '40,000', '35k', '30k+', '50 thousand'."""
    if not att:
        return None
    s = att.strip().lower()
    # Remove qualifiers like '+' or '~' or 'approx.'
    s = s.replace("+", "").replace("~", "").replace("approximately", "").replace("approx.", "").strip()

    # e.g., 40,000
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)", s)
    if m:
        try:
            n = int(m.group(1).replace(",", ""))
            return n
        except Exception:
            pass

    # e.g., 35k or 35 k
    m = re.search(r"(\d+(?:\.\d+)?)\s*k\b", s)
    if m:
        try:
            val = float(m.group(1))
            return int(round(val * 1000))
        except Exception:
            pass

    # e.g., '50 thousand'
    m = re.search(r"(\d+(?:\.\d+)?)\s*thousand", s)
    if m:
        try:
            val = float(m.group(1))
            return int(round(val * 1000))
        except Exception:
            pass

    return None


def normalize_state_str(s: Optional[str]) -> Optional[str]:
    """Return a normalized two-letter state abbreviation for contiguous US + DC, if possible."""
    if not s:
        return None
    raw = s.strip()
    # Direct abbr
    if len(raw) == 2:
        abbr = raw.upper()
        # Some users might include punctuation like 'D.C.'
        abbr = abbr.replace(".", "")
        return abbr
    # Try title case mapping
    name = raw.strip()
    # Normalize various DC spellings
    if name.lower() in {"washington, dc", "washington dc", "dc", "district of columbia", "washington d.c."}:
        return "DC"
    # Title-case for mapping
    title_name = " ".join(w.capitalize() for w in name.split())
    if title_name in STATE_NAME_TO_ABBR:
        return STATE_NAME_TO_ABBR[title_name]
    return None


def is_continental_us_state(state_str: Optional[str]) -> bool:
    """Return True if the state is in the contiguous US + DC."""
    abbr = normalize_state_str(state_str)
    if not abbr:
        return False
    return abbr in CONTIGUOUS_STATE_ABBR


def within_required_date_range(d: Optional[date], check_start: bool) -> bool:
    """Check if date is within required range depending on start vs end check."""
    if not d:
        return False
    # Start date must be >= March 1, 2026
    if check_start:
        return d >= DATE_RANGE_START
    # End date must be <= August 31, 2026
    else:
        return d <= DATE_RANGE_END


def coalesce_sources(primary: Optional[str], others: List[str]) -> List[str]:
    """Combine primary URL and other sources, de-duplicate, keep non-empty."""
    urls = []
    if primary and primary.strip():
        urls.append(primary.strip())
    for u in others:
        if u and u.strip():
            urls.append(u.strip())
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification logic for a single event                                       #
# --------------------------------------------------------------------------- #
async def verify_event(evaluator: Evaluator, parent_node, event: EventItem, idx: int) -> None:
    """
    Build the verification subtree for one event and run leaf verifications.
    Note: To comply with the framework's constraint that critical parents cannot have non-critical children,
    we make section nodes non-critical and mark essential 'value' leaves as critical, while 'reference' leaves are non-critical.
    """
    event_no = idx + 1
    ev_node = evaluator.add_parallel(
        id=f"event_{event_no}",
        desc=f"{['First','Second','Third','Fourth'][idx]} qualifying gaming event identification and verification",
        parent=parent_node,
        critical=False
    )

    # ---------------- Date range compliance ----------------
    date_range_node = evaluator.add_parallel(
        id=f"event_{event_no}_date_range_compliance",
        desc=f"Verification that event #{event_no} takes place entirely between March 1 and August 31, 2026",
        parent=ev_node,
        critical=False
    )

    # Start date group
    start_group = evaluator.add_parallel(
        id=f"event_{event_no}_start_date",
        desc="Start date verification",
        parent=date_range_node,
        critical=False
    )
    # Value check
    parsed_start = try_parse_date_str(event.start_date)
    start_ok = within_required_date_range(parsed_start, check_start=True) and (parsed_start.year == 2026 if parsed_start else False)
    evaluator.add_custom_node(
        result=start_ok,
        id=f"event_{event_no}_start_date_value",
        desc="Exact start date is provided and on or after March 1, 2026",
        parent=start_group,
        critical=True
    )
    # Reference check
    start_ref_leaf = evaluator.add_leaf(
        id=f"event_{event_no}_start_date_reference",
        desc="Reference URL confirming the start date",
        parent=start_group,
        critical=False
    )
    start_sources = coalesce_sources(event.primary_source_url, event.start_date_sources)
    start_claim = f"The event{f' {event.name}' if event.name else ''} starts on {event.start_date}."
    await evaluator.verify(
        claim=start_claim,
        node=start_ref_leaf,
        sources=start_sources if start_sources else None,
        additional_instruction="Verify against the provided webpage(s). If no source URLs are provided, return Incorrect."
    )

    # End date group
    end_group = evaluator.add_parallel(
        id=f"event_{event_no}_end_date",
        desc="End date verification",
        parent=date_range_node,
        critical=False
    )
    parsed_end = try_parse_date_str(event.end_date)
    end_ok = within_required_date_range(parsed_end, check_start=False) and (parsed_end.year == 2026 if parsed_end else False)
    evaluator.add_custom_node(
        result=end_ok,
        id=f"event_{event_no}_end_date_value",
        desc="Exact end date is provided and on or before August 31, 2026",
        parent=end_group,
        critical=True
    )
    end_ref_leaf = evaluator.add_leaf(
        id=f"event_{event_no}_end_date_reference",
        desc="Reference URL confirming the end date",
        parent=end_group,
        critical=False
    )
    end_sources = coalesce_sources(event.primary_source_url, event.end_date_sources)
    end_claim = f"The event{f' {event.name}' if event.name else ''} ends on {event.end_date}."
    await evaluator.verify(
        claim=end_claim,
        node=end_ref_leaf,
        sources=end_sources if end_sources else None,
        additional_instruction="Verify against the provided webpage(s). If no source URLs are provided, return Incorrect."
    )

    # ---------------- Location compliance ----------------
    loc_node = evaluator.add_parallel(
        id=f"event_{event_no}_location_compliance",
        desc=f"Verification that event #{event_no} is located in the continental United States",
        parent=ev_node,
        critical=False
    )
    loc_details = evaluator.add_parallel(
        id=f"event_{event_no}_location_details",
        desc="Host city and state identification",
        parent=loc_node,
        critical=False
    )
    city_ok = bool(event.city and event.city.strip())
    state_ok = is_continental_us_state(event.state)
    evaluator.add_custom_node(
        result=city_ok,
        id=f"event_{event_no}_city_name",
        desc="Name of the host city in the continental US is provided",
        parent=loc_details,
        critical=True
    )
    evaluator.add_custom_node(
        result=state_ok,
        id=f"event_{event_no}_state_name",
        desc="Name of the host state in the continental US is valid",
        parent=loc_details,
        critical=True
    )
    loc_ref_leaf = evaluator.add_leaf(
        id=f"event_{event_no}_location_reference",
        desc="Reference URL confirming the location",
        parent=loc_details,
        critical=False
    )
    loc_sources = coalesce_sources(event.primary_source_url, event.location_sources)
    loc_claim = f"The event{f' {event.name}' if event.name else ''} is located in {event.city}, {event.state}."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_ref_leaf,
        sources=loc_sources if loc_sources else None,
        additional_instruction="Verify the city and state on the provided webpage(s). If no source URLs are provided, return Incorrect."
    )

    # ---------------- Venue identification ----------------
    venue_node = evaluator.add_parallel(
        id=f"event_{event_no}_venue_identification",
        desc=f"Identification of the official venue name for event #{event_no}",
        parent=ev_node,
        critical=False
    )
    venue_ok = bool(event.venue and event.venue.strip())
    evaluator.add_custom_node(
        result=venue_ok,
        id=f"event_{event_no}_venue_name",
        desc="Official venue name is provided",
        parent=venue_node,
        critical=True
    )
    venue_ref_leaf = evaluator.add_leaf(
        id=f"event_{event_no}_venue_reference",
        desc="Reference URL confirming the venue",
        parent=venue_node,
        critical=False
    )
    venue_sources = coalesce_sources(event.primary_source_url, event.venue_sources)
    venue_claim = f"The official venue for the event{f' {event.name}' if event.name else ''} is '{event.venue}'."
    await evaluator.verify(
        claim=venue_claim,
        node=venue_ref_leaf,
        sources=venue_sources if venue_sources else None,
        additional_instruction="Verify the venue on the provided webpage(s). If no source URLs are provided, return Incorrect."
    )

    # ---------------- Attendance compliance ----------------
    att_node = evaluator.add_parallel(
        id=f"event_{event_no}_attendance_compliance",
        desc=f"Verification that event #{event_no} has expected attendance of at least 30,000",
        parent=ev_node,
        critical=False
    )
    att_value = attendance_to_int(event.attendance)
    att_ok = (att_value is not None) and (att_value >= 30000)
    evaluator.add_custom_node(
        result=att_ok,
        id=f"event_{event_no}_attendance_threshold",
        desc="Expected or confirmed attendance number is provided and is at least 30,000",
        parent=att_node,
        critical=True
    )
    att_ref_leaf = evaluator.add_leaf(
        id=f"event_{event_no}_attendance_reference",
        desc="Reference URL confirming the attendance figure",
        parent=att_node,
        critical=False
    )
    att_sources = coalesce_sources(event.primary_source_url, event.attendance_sources)
    att_claim = f"The event{f' {event.name}' if event.name else ''} expects or confirms attendance of {event.attendance}."
    await evaluator.verify(
        claim=att_claim,
        node=att_ref_leaf,
        sources=att_sources if att_sources else None,
        additional_instruction="Verify the attendance figure (or equivalent statement of expected attendance) on the provided webpage(s). If no source URLs are provided, return Incorrect."
    )

    # ---------------- Type/focus compliance ----------------
    focus_node = evaluator.add_parallel(
        id=f"event_{event_no}_type_compliance",
        desc=f"Verification that event #{event_no} focuses on video/PC gaming or industry networking (not exclusively tabletop/card/fighting tourneys)",
        parent=ev_node,
        critical=False
    )
    # Critical verification via source support
    focus_verify_leaf = evaluator.add_leaf(
        id=f"event_{event_no}_focus_verification",
        desc="Event focus/type meets the gaming requirement",
        parent=focus_node,
        critical=True
    )
    focus_sources = coalesce_sources(event.primary_source_url, event.focus_sources)
    focus_claim = (
        "This event primarily focuses on video gaming, PC gaming, gaming industry networking, or prominently features indie/video games; "
        "it is not exclusively about tabletop games, card games, or fighting game tournaments."
    )
    await evaluator.verify(
        claim=focus_claim,
        node=focus_verify_leaf,
        sources=focus_sources if focus_sources else None,
        additional_instruction="Judge based on the provided webpage(s). If the pages show video/PC gaming or industry networking focus (or strong indie/video games presence), mark Correct; if exclusively tabletop/card/fighting game tournament, mark Incorrect. If no source URLs are provided, return Incorrect."
    )
    # Non-critical reference leaf to confirm the described focus text
    focus_ref_leaf = evaluator.add_leaf(
        id=f"event_{event_no}_focus_reference",
        desc="Reference URL confirming the event focus",
        parent=focus_node,
        critical=False
    )
    focus_desc_claim = f"The event's primary focus/type is described as: {event.focus_description}."
    await evaluator.verify(
        claim=focus_desc_claim,
        node=focus_ref_leaf,
        sources=focus_sources if focus_sources else None,
        additional_instruction="Confirm that the provided webpage(s) include a description of the event's focus consistent with the claim. If no source URLs are provided, return Incorrect."
    )

    # ---------------- Primary source documentation ----------------
    source_leaf = evaluator.add_leaf(
        id=f"event_{event_no}_source_documentation",
        desc=f"Primary source URL for event #{event_no}'s official information",
        parent=ev_node,
        critical=False
    )
    primary_sources = [event.primary_source_url] if (event.primary_source_url and event.primary_source_url.strip()) else []
    source_claim = f"This URL is an official or reliable source page for the 2026 edition of {event.name}." if event.name else "This URL is an official or reliable source page for the event's 2026 edition."
    await evaluator.verify(
        claim=source_claim,
        node=source_leaf,
        sources=primary_sources if primary_sources else None,
        additional_instruction="Accept official event websites, pages on the event's domain, or reliable news/press releases as valid. If no source URL is provided, return Incorrect."
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
    Evaluate an answer for identifying 4 gaming events in 2026 that meet specified criteria.
    """
    # Initialize evaluator (root as non-critical to allow partial credit across events)
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

    # Extract events from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction"
    )

    # Prepare exactly 4 events (pad with empty if fewer; trim if more)
    events: List[EventItem] = list(extracted.events[:4])
    while len(events) < 4:
        events.append(EventItem())

    # Build verification subtrees for each event
    for i in range(4):
        await verify_event(evaluator, root, events[i], i)

    # Add custom info for transparency
    evaluator.add_custom_info(
        info={
            "date_range_window": {"start": str(DATE_RANGE_START), "end": str(DATE_RANGE_END)},
            "contiguous_states_abbr": sorted(list(CONTIGUOUS_STATE_ABBR))
        },
        info_type="config",
        info_name="evaluation_constraints"
    )

    return evaluator.get_summary()