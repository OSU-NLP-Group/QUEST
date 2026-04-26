import asyncio
import logging
import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants and category definitions                                     #
# --------------------------------------------------------------------------- #

TASK_ID = "four_us_tv_programs_2025_2026_window"
TASK_DESCRIPTION = (
    "Identify four distinct annual U.S. television programs or sporting events within "
    "Nov 27, 2025 – Feb 22, 2026, one per category with all required fields and constraints."
)

GLOBAL_WINDOW_START = date(2025, 11, 27)
GLOBAL_WINDOW_END = date(2026, 2, 22)

CATEGORY_LABELS = {
    "Thanksgiving Afternoon Special": "Category 1 - Thanksgiving Afternoon Special",
    "Christmas Morning Special": "Category 2 - Christmas Morning Special",
    "Winter Sporting Event": "Category 3 - Winter Sporting Event",
    "Weekday Morning Show": "Category 4 - Weekday Morning Show",
}

ALLOWED_CATEGORIES = set(CATEGORY_LABELS.keys())


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #

class ProgramItem(BaseModel):
    """
    One selected item in the user's answer, assigned to exactly one category.
    Please keep strings as they appear in the answer; do not normalize or invent.
    """
    category: Optional[str] = None  # One of: Thanksgiving Afternoon Special, Christmas Morning Special, Winter Sporting Event, Weekday Morning Show
    name: Optional[str] = None

    # Dates / times
    date_or_range: Optional[str] = None  # e.g., "November 27, 2025" or "February 6–22, 2026"
    time_or_window: Optional[str] = None  # e.g., "12:00 PM–2:00 PM ET" or "7:00 AM ET"

    # Networks and coverage
    network: Optional[str] = None  # Primary broadcast network (e.g., NBC, ABC)
    us_broadcast_networks: List[str] = Field(default_factory=list)  # For Category 3

    # Location (for events)
    location: Optional[str] = None  # e.g., "Milan–Cortina d'Ampezzo, Italy"

    # People
    hosts_or_anchors: List[str] = Field(default_factory=list)  # hosts / presenters / anchors

    # Streaming
    streaming_platforms: List[str] = Field(default_factory=list)  # e.g., Peacock, Hulu, Disney+
    streaming_urls: List[str] = Field(default_factory=list)  # URLs to streaming destinations if provided

    # References / citations
    reference_urls: List[str] = Field(default_factory=list)  # Official or reliable source URLs


class ItemsExtraction(BaseModel):
    items: List[ProgramItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #

def prompt_extract_items() -> str:
    return """
Extract exactly four distinct items from the provided answer. Each item must be assigned to exactly one of the following categories:

- Thanksgiving Afternoon Special
- Christmas Morning Special
- Winter Sporting Event
- Weekday Morning Show

For each of the four items, extract the following fields as they appear in the answer (do not invent):
- category: One of the exact strings above (choose exactly one per item, and cover all four categories exactly once in total)
- name: Official program/event name
- date_or_range: The specific broadcast date (for specials/shows) or the date range (for multi-day events) cited for the 2025/2026 occurrence (e.g., "November 27, 2025", "December 25, 2025", or "February 6–22, 2026")
- time_or_window: The broadcast time for daily/special programs (e.g., "12:00 PM–2:00 PM ET" or "7:00 AM ET"). For multi-day events, you may put null.
- network: The broadcasting network (e.g., NBC, ABC). For Category 3, this can be null if networks are instead listed in us_broadcast_networks.
- us_broadcast_networks: (Only if applicable, e.g., Category 3) A list of U.S. TV networks covering the event
- location: (Only if applicable, e.g., Category 3) The host location (city/region/country)
- hosts_or_anchors: Names of hosts/anchors/presenters where applicable
- streaming_platforms: At least one streaming service/platform if mentioned (e.g., Peacock, Hulu, Disney+). Names only.
- streaming_urls: Any URLs to streaming pages if the answer provides them
- reference_urls: At least one official/reliable reference URL for verification (network/event/streaming official page, or reputable entertainment/news)

Rules:
- Do not invent URLs or names. Extract only what appears in the answer.
- If a field is not provided in the answer for an item, set it to null (or empty list for list fields).
- Ensure there are exactly four items total and each is assigned to a distinct category from the list above (no duplicates).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #

_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

def _month_to_num(mstr: str) -> Optional[int]:
    return _MONTHS.get(mstr.strip().lower())

def _safe_date(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except Exception:
        return None

def parse_date_or_range(text: Optional[str]) -> Optional[Tuple[date, date]]:
    """
    Attempt to parse a date or a date range from text such as:
      - "November 27, 2025"
      - "Feb 6–22, 2026" or "February 6-22, 2026"
      - "February 2026" (interpreted as full month)
    Returns (start_date, end_date) if parsed, else None.
    """
    if not text:
        return None

    s = text.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip().replace(".", "")

    # Pattern 1: "Month D, YYYY" or "Mon D, YYYY"
    m1 = re.search(r"\b([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})\b", s)
    if m1:
        mon = _month_to_num(m1.group(1))
        day = int(m1.group(2))
        year = int(m1.group(3))
        if mon:
            d = _safe_date(year, mon, day)
            if d:
                return (d, d)

    # Pattern 2: "Month D-D, YYYY" or "Mon D-D, YYYY"
    m2 = re.search(r"\b([A-Za-z]+)\s+(\d{1,2})\s*-\s*(\d{1,2}),\s*(\d{4})\b", s)
    if m2:
        mon = _month_to_num(m2.group(1))
        d1 = int(m2.group(2))
        d2 = int(m2.group(3))
        year = int(m2.group(4))
        if mon:
            sd = _safe_date(year, mon, d1)
            ed = _safe_date(year, mon, d2)
            if sd and ed and ed >= sd:
                return (sd, ed)

    # Pattern 3: "Month YYYY" (interpret as entire month)
    m3 = re.search(r"\b([A-Za-z]+)\s+(\d{4})\b", s)
    if m3:
        mon = _month_to_num(m3.group(1))
        year = int(m3.group(2))
        if mon:
            # Month length approximation (no leap year for Feb 2026)
            month_lengths = {1:31, 2:29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28, 3:31, 4:30,
                             5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
            sd = _safe_date(year, mon, 1)
            ed = _safe_date(year, mon, month_lengths.get(mon, 28))
            if sd and ed:
                return (sd, ed)

    # Pattern 4: Loose "D Month YYYY" (less common in answers)
    m4 = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b", s)
    if m4:
        day = int(m4.group(1))
        mon = _month_to_num(m4.group(2))
        year = int(m4.group(3))
        if mon:
            d = _safe_date(year, mon, day)
            if d:
                return (d, d)

    # Pattern 5: Numeric MM/DD/YYYY (rare in this context but safe)
    m5 = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", s)
    if m5:
        mon = int(m5.group(1))
        day = int(m5.group(2))
        year = int(m5.group(3))
        d = _safe_date(year, mon, day)
        if d:
            return (d, d)

    return None

def overlaps_window(date_range: Optional[Tuple[date, date]],
                    window_start: date = GLOBAL_WINDOW_START,
                    window_end: date = GLOBAL_WINDOW_END) -> bool:
    if not date_range:
        return False
    start, end = date_range
    return not (end < window_start or start > window_end)

def first_nonempty_url(urls: List[str]) -> Optional[str]:
    for u in urls:
        if isinstance(u, str) and u.strip().startswith("http"):
            return u.strip()
    return None

def merge_urls(item: ProgramItem) -> List[str]:
    # Deduplicate while preserving order
    seen = set()
    result: List[str] = []
    for u in (item.reference_urls or []) + (item.streaming_urls or []):
        if isinstance(u, str):
            u = u.strip()
            if u and u not in seen:
                seen.add(u)
                result.append(u)
    return result

def norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def distinct_names(items: List[ProgramItem]) -> bool:
    names = [norm(it.name) for it in items if it.name and it.name.strip()]
    return len(names) == 4 and len(set(names)) == 4

def categories_covered_once(items: List[ProgramItem]) -> bool:
    cats = [it.category for it in items if it.category]
    return len(cats) == 4 and set(cats) == ALLOWED_CATEGORIES

def count_nonempty(lst: Optional[List[str]]) -> int:
    return sum(1 for x in (lst or []) if isinstance(x, str) and x.strip())


# --------------------------------------------------------------------------- #
# Verification builders per category                                          #
# --------------------------------------------------------------------------- #

async def verify_category_1_thanksgiving(evaluator: Evaluator, parent: VerificationNode, item: ProgramItem) -> None:
    """
    Category 1 - Thanksgiving Afternoon Special
    Requirements summary:
    - Airs annually on Thanksgiving Day
    - Broadcast during afternoon (11:00 AM – 3:00 PM in any TZ)
    - Airs on NBC
    - Animal-related content
    - Consistent annual hosts/presenters; host names provided
    - 2025 occurrence date provided and within window
    - Streaming platform provided
    - At least one acceptable reference URL
    """
    node = evaluator.add_parallel(
        id="item_1_thanksgiving_afternoon_special",
        desc="Category 1 item: Thanksgiving Afternoon Special meets all Category 1 constraints and required fields.",
        parent=parent,
        critical=False
    )

    urls = merge_urls(item)
    one_ref = first_nonempty_url(item.reference_urls or [])

    # 1) Official name provided (critical, existence)
    evaluator.add_custom_node(
        result=bool(item.name and item.name.strip()),
        id="item1_official_name_provided",
        desc="Official name of the program is provided.",
        parent=node,
        critical=True
    )

    # 2) 2025 occurrence date provided (critical, existence)
    # Require a specific 2025 date for this category
    dtr = item.date_or_range or ""
    parsed = parse_date_or_range(dtr)
    year_ok = False
    if parsed:
        year_ok = (parsed[0].year == 2025 or parsed[1].year == 2025)
    elif "2025" in dtr:
        year_ok = True
    evaluator.add_custom_node(
        result=bool(dtr.strip()) and year_ok,
        id="item1_2025_occurrence_date_provided",
        desc="Specific broadcast date for the 2025/2026 occurrence is provided.",
        parent=node,
        critical=True
    )

    # 3) 2025 occurrence within global window (critical, logical)
    in_window = overlaps_window(parse_date_or_range(dtr))
    evaluator.add_custom_node(
        result=in_window,
        id="item1_2025_occurrence_within_global_window",
        desc="The provided 2025 occurrence date falls within Nov 27, 2025–Feb 22, 2026.",
        parent=node,
        critical=True
    )

    # 4) Broadcast time provided (critical, existence)
    evaluator.add_custom_node(
        result=bool(item.time_or_window and item.time_or_window.strip()),
        id="item1_broadcast_time_provided",
        desc="Broadcast time for the 2025 occurrence is provided (with a stated time zone or clear time-zone context).",
        parent=node,
        critical=True
    )

    # 5) Afternoon time window met (critical, LLM-simple check on provided time string)
    leaf_time = evaluator.add_leaf(
        id="item1_afternoon_time_window_met",
        desc="The stated broadcast time is between 11:00 AM and 3:00 PM in the stated time zone context.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The stated broadcast time '{item.time_or_window or ''}' falls between 11:00 AM and 3:00 PM in its stated time zone.",
        node=leaf_time,
        additional_instruction="Interpret the provided time(s) leniently (e.g., ranges). If any portion falls within 11:00–15:00 local, consider it meeting the requirement."
    )

    # 6) Network is NBC (critical, evidence-backed)
    leaf_nbc = evaluator.add_leaf(
        id="item1_network_is_nbc",
        desc="Broadcasting network is NBC.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The program '{item.name or ''}' airs on the NBC television network.",
        node=leaf_nbc,
        sources=urls,
        additional_instruction="Look for 'NBC' branding or explicit statements that NBC broadcasts this special."
    )

    # 7) Animal-related content (critical, evidence-backed)
    leaf_animal = evaluator.add_leaf(
        id="item1_animal_related_content",
        desc="Program features animal-related content.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The program '{item.name or ''}' features animal-related content (e.g., animals, pets, dog show, etc.).",
        node=leaf_animal,
        sources=urls,
        additional_instruction="Confirm that the central theme or content involves animals."
    )

    # 8) Consistent annual hosts (critical, evidence-backed)
    leaf_hosts_consistent = evaluator.add_leaf(
        id="item1_consistent_annual_hosts",
        desc="Program has consistent annual hosts/presenters (per reliable sources).",
        parent=node,
        critical=True
    )
    hosts_list = ", ".join(item.hosts_or_anchors) if item.hosts_or_anchors else ""
    await evaluator.verify(
        claim=(
            f"The program '{item.name or ''}' has consistent annual hosts/presenters "
            f"(for example, recurring hosts such as {hosts_list} year after year)."
        ),
        node=leaf_hosts_consistent,
        sources=urls,
        additional_instruction="Accept minor year-to-year variations but there should be clear evidence of a stable, recurring host lineup."
    )

    # 9) Hosts/presenters names provided (critical, existence)
    evaluator.add_custom_node(
        result=count_nonempty(item.hosts_or_anchors) > 0,
        id="item1_hosts_names_provided",
        desc="Names of hosts/presenters/key presenters are provided.",
        parent=node,
        critical=True
    )

    # 10) Streaming platform provided (critical, existence)
    evaluator.add_custom_node(
        result=count_nonempty(item.streaming_platforms) > 0,
        id="item1_streaming_platform_provided",
        desc="At least one streaming platform where the program is available is provided.",
        parent=node,
        critical=True
    )

    # 11) Reference URL provided and acceptable (critical, evidence-backed)
    if one_ref:
        leaf_ref = evaluator.add_leaf(
            id="item1_reference_url_provided_and_acceptable",
            desc="At least one reference URL is provided, and it is an official network/streaming page or a reliable entertainment/news source confirming the program details.",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=(
                f"This page is an official network/streaming page or a reputable entertainment/news source and confirms key details "
                f"for '{item.name or ''}' such as date/time or network."
            ),
            node=leaf_ref,
            sources=one_ref,
            additional_instruction="Assess domain and on-page context; accept official network/streaming domains (e.g., nbc.com/nbcsports/peacock) or reputable news/entertainment outlets."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="item1_reference_url_provided_and_acceptable",
            desc="At least one reference URL is provided, and it is an official/reliable source.",
            parent=node,
            critical=True
        )


async def verify_category_2_christmas(evaluator: Evaluator, parent: VerificationNode, item: ProgramItem) -> None:
    """
    Category 2 - Christmas Morning Special
    Requirements summary:
    - Airs annually on December 25 (Christmas Day)
    - Morning broadcast (5:00 AM – 12:00 PM)
    - Airs on ABC
    - Disney theme park content
    - Includes musical performances
    - Presenter names provided
    - 2025 date provided and within window
    - Streaming platform provided
    - Acceptable reference URL
    """
    node = evaluator.add_parallel(
        id="item_2_christmas_morning_special",
        desc="Category 2 item: Christmas Morning Special meets all Category 2 constraints and required fields.",
        parent=parent,
        critical=False
    )

    urls = merge_urls(item)
    one_ref = first_nonempty_url(item.reference_urls or [])
    dtr = item.date_or_range or ""
    parsed = parse_date_or_range(dtr)
    year_ok = False
    if parsed:
        year_ok = (parsed[0].year == 2025 or parsed[1].year == 2025)
    elif "2025" in dtr:
        year_ok = True

    # 1) Official name provided
    evaluator.add_custom_node(
        result=bool(item.name and item.name.strip()),
        id="item2_official_name_provided",
        desc="Official name of the program is provided.",
        parent=node,
        critical=True
    )

    # 2) Airs annually on Dec 25 (evidence)
    leaf_annual = evaluator.add_leaf(
        id="item2_airs_annually_on_dec_25",
        desc="Program airs annually on Christmas Day (December 25).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The program '{item.name or ''}' airs annually on Christmas Day (December 25).",
        node=leaf_annual,
        sources=urls,
        additional_instruction="Look for language indicating a yearly tradition on Dec 25."
    )

    # 3) 2025 occurrence date provided (must indicate Dec 25, 2025)
    # We'll require 2025 + a December indicator in text or a parsed Dec 25 date
    dec25_2025 = False
    if parsed and parsed[0] == parsed[1]:
        dec25_2025 = (parsed[0].year == 2025 and parsed[0].month == 12 and parsed[0].day == 25)
    if not dec25_2025 and "2025" in dtr and ("Dec" in dtr or "December" in dtr) and ("25" in dtr):
        dec25_2025 = True
    evaluator.add_custom_node(
        result=bool(dtr.strip()) and dec25_2025,
        id="item2_2025_occurrence_date_provided",
        desc="Specific broadcast date for the 2025/2026 occurrence is provided and is December 25.",
        parent=node,
        critical=True
    )

    # 4) 2025 occurrence within global window
    evaluator.add_custom_node(
        result=overlaps_window(parse_date_or_range(dtr)),
        id="item2_2025_occurrence_within_global_window",
        desc="The provided 2025 occurrence date falls within Nov 27, 2025–Feb 22, 2026.",
        parent=node,
        critical=True
    )

    # 5) Broadcast time provided
    evaluator.add_custom_node(
        result=bool(item.time_or_window and item.time_or_window.strip()),
        id="item2_broadcast_time_provided",
        desc="Broadcast time for the 2025 occurrence is provided (with a stated time zone or clear time-zone context).",
        parent=node,
        critical=True
    )

    # 6) Morning time window met (LLM simple)
    leaf_morning = evaluator.add_leaf(
        id="item2_morning_time_window_met",
        desc="The stated broadcast time is between 5:00 AM and 12:00 PM in the stated time zone context.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The stated broadcast time '{item.time_or_window or ''}' falls between 5:00 AM and 12:00 PM in its stated time zone.",
        node=leaf_morning,
        additional_instruction="Interpret ranges leniently; if start time is within the morning window, consider it meeting the requirement."
    )

    # 7) Network is ABC (evidence)
    leaf_abc = evaluator.add_leaf(
        id="item2_network_is_abc",
        desc="Broadcasting network is ABC.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The program '{item.name or ''}' airs on the ABC television network.",
        node=leaf_abc,
        sources=urls,
        additional_instruction="Look for ABC branding or explicit network attribution."
    )

    # 8) Disney theme park content (evidence)
    leaf_disney = evaluator.add_leaf(
        id="item2_disney_theme_park_content",
        desc="Program features Disney theme park content.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The program '{item.name or ''}' features Disney theme park content.",
        node=leaf_disney,
        sources=urls,
        additional_instruction="Confirm the program showcases Disney Parks experiences, parade, or segments set in Disney parks."
    )

    # 9) Includes musical performances (evidence)
    leaf_music = evaluator.add_leaf(
        id="item2_includes_musical_performances",
        desc="Program includes musical performances.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The program '{item.name or ''}' includes musical performances.",
        node=leaf_music,
        sources=urls,
        additional_instruction="Look for mentions of performers, songs, or musical segments."
    )

    # 10) Presenter names provided (existence)
    evaluator.add_custom_node(
        result=count_nonempty(item.hosts_or_anchors) > 0,
        id="item2_presenter_names_provided",
        desc="Names of hosts/anchors/key presenters are provided (where applicable).",
        parent=node,
        critical=True
    )

    # 11) Streaming platform provided (existence)
    evaluator.add_custom_node(
        result=count_nonempty(item.streaming_platforms) > 0,
        id="item2_streaming_platform_provided",
        desc="At least one streaming platform where the program is available is provided.",
        parent=node,
        critical=True
    )

    # 12) Reference URL provided and acceptable (evidence)
    if one_ref:
        leaf_ref = evaluator.add_leaf(
            id="item2_reference_url_provided_and_acceptable",
            desc="At least one reference URL is provided, and it is an official network/streaming page or a reliable entertainment/news source confirming the program details.",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=(
                f"This page is an official network/streaming page or a reputable entertainment/news source and confirms key details "
                f"for '{item.name or ''}' such as date/time or network."
            ),
            node=leaf_ref,
            sources=one_ref,
            additional_instruction="Assess domain and page content; confirm relevance to the program and inclusion of concrete details."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="item2_reference_url_provided_and_acceptable",
            desc="At least one reference URL is provided, and it is an official/reliable source.",
            parent=node,
            critical=True
        )


async def verify_category_3_winter_event(evaluator: Evaluator, parent: VerificationNode, item: ProgramItem) -> None:
    """
    Category 3 - Winter Sporting Event
    Requirements summary:
    - February 2026, in Europe
    - Multi-day (several days)
    - Winter sports competitions
    - Recurring event
    - U.S. TV coverage confirmed; U.S. networks provided
    - Date range provided; falls in Feb 2026; within global window
    - Streaming platform provided
    - Acceptable reference URL
    """
    node = evaluator.add_parallel(
        id="item_3_winter_sporting_event",
        desc="Category 3 item: Winter Sporting Event meets all Category 3 constraints and required fields.",
        parent=parent,
        critical=False
    )

    urls = merge_urls(item)
    one_ref = first_nonempty_url(item.reference_urls or [])
    dtr = item.date_or_range or ""
    parsed = parse_date_or_range(dtr)

    # 1) Official name provided
    evaluator.add_custom_node(
        result=bool(item.name and item.name.strip()),
        id="item3_official_name_provided",
        desc="Official name of the event is provided.",
        parent=node,
        critical=True
    )

    # 2) Event is recurring (evidence)
    leaf_recurring = evaluator.add_leaf(
        id="item3_event_is_recurring",
        desc="Event is recurring (not a one-time event).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event '{item.name or ''}' is recurring (held regularly, not a one-time event).",
        node=leaf_recurring,
        sources=urls,
        additional_instruction="Look for language like 'every X years', 'annual', 'biennial', 'quadrennial', etc."
    )

    # 3) Date range provided (existence)
    evaluator.add_custom_node(
        result=bool(dtr.strip()),
        id="item3_date_range_provided",
        desc="Event date range is provided.",
        parent=node,
        critical=True
    )

    # 4) Date range in February 2026 (evidence)
    leaf_feb26 = evaluator.add_leaf(
        id="item3_date_range_in_feb_2026",
        desc="Event occurs during February 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event '{item.name or ''}' takes place during February 2026.",
        node=leaf_feb26,
        sources=urls,
        additional_instruction="Even if it starts earlier in February or ends later in February, the central competition window should be in February 2026."
    )

    # 5) Date range within global window (logical)
    evaluator.add_custom_node(
        result=overlaps_window(parsed),
        id="item3_date_range_within_global_window",
        desc="Event date range lies within Nov 27, 2025–Feb 22, 2026.",
        parent=node,
        critical=True
    )

    # 6) Location provided (existence)
    evaluator.add_custom_node(
        result=bool(item.location and item.location.strip()),
        id="item3_location_provided",
        desc="Event location is provided.",
        parent=node,
        critical=True
    )

    # 7) Location is in Europe (evidence)
    leaf_europe = evaluator.add_leaf(
        id="item3_location_in_europe",
        desc="Event occurs in Europe.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event '{item.name or ''}' takes place in Europe (host location: {item.location or ''}).",
        node=leaf_europe,
        sources=urls,
        additional_instruction="Confirm that the host country/city is in Europe."
    )

    # 8) Multi-day event (evidence)
    leaf_multiday = evaluator.add_leaf(
        id="item3_multi_day_event",
        desc="Event spans multiple days (at least several days).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event '{item.name or ''}' spans multiple days.",
        node=leaf_multiday,
        sources=urls,
        additional_instruction="Look for the official schedule spanning several days."
    )

    # 9) Features winter sports competitions (evidence)
    leaf_winter_sports = evaluator.add_leaf(
        id="item3_features_winter_sports",
        desc="Event features winter sports competitions.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event '{item.name or ''}' features winter sports competitions.",
        node=leaf_winter_sports,
        sources=urls,
        additional_instruction="Look for disciplines such as skiing, skating, hockey, biathlon, etc."
    )

    # 10) U.S. TV broadcast coverage confirmed (evidence)
    leaf_us_tv = evaluator.add_leaf(
        id="item3_us_tv_broadcast_coverage_confirmed",
        desc="U.S. television broadcast coverage is confirmed.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"There is U.S. television broadcast coverage for '{item.name or ''}'.",
        node=leaf_us_tv,
        sources=urls,
        additional_instruction="Confirm that a U.S. TV network will carry coverage and not just international feeds."
    )

    # 11) U.S. broadcast networks provided (existence)
    evaluator.add_custom_node(
        result=count_nonempty(item.us_broadcast_networks) > 0,
        id="item3_us_broadcast_networks_provided",
        desc="U.S. broadcasting network(s) covering the event are provided.",
        parent=node,
        critical=True
    )

    # 12) Streaming platform provided (existence)
    evaluator.add_custom_node(
        result=count_nonempty(item.streaming_platforms) > 0,
        id="item3_streaming_platform_provided",
        desc="At least one streaming platform where the event can be watched is provided.",
        parent=node,
        critical=True
    )

    # 13) Reference URL provided and acceptable (evidence)
    if one_ref:
        leaf_ref = evaluator.add_leaf(
            id="item3_reference_url_provided_and_acceptable",
            desc="At least one reference URL is provided, and it is an official event/broadcaster page or a reliable news source confirming timing/location and U.S. broadcast details.",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=(
                f"This page is an official event/broadcaster page or a reputable news source and confirms timing/location and U.S. broadcast/streaming details for '{item.name or ''}'."
            ),
            node=leaf_ref,
            sources=one_ref,
            additional_instruction="Check that the page clearly ties to the event with dates, location, and indicates U.S. coverage/broadcast info."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="item3_reference_url_provided_and_acceptable",
            desc="At least one reference URL is provided, and it is an official/reliable source.",
            parent=node,
            critical=True
        )


async def verify_category_4_weekday_show(evaluator: Evaluator, parent: VerificationNode, item: ProgramItem) -> None:
    """
    Category 4 - Weekday Morning Show
    Requirements summary:
    - Airs Monday–Friday
    - Begins at 7:00 AM Eastern Time
    - Airs on NBC
    - Exactly two main co-anchors hosting the 7:00–8:00 AM hour; names provided
    - Provide a specific date/range within global window (existence+window)
    - Streaming platform provided
    - Acceptable reference URL
    """
    node = evaluator.add_parallel(
        id="item_4_weekday_morning_show",
        desc="Category 4 item: Weekday Morning Show meets all Category 4 constraints and required fields.",
        parent=parent,
        critical=False
    )

    urls = merge_urls(item)
    one_ref = first_nonempty_url(item.reference_urls or [])
    dtr = item.date_or_range or ""
    parsed = parse_date_or_range(dtr)

    # 1) Official name provided
    evaluator.add_custom_node(
        result=bool(item.name and item.name.strip()),
        id="item4_official_name_provided",
        desc="Official name of the show is provided.",
        parent=node,
        critical=True
    )

    # 2) Airs Monday through Friday (evidence)
    leaf_mf = evaluator.add_leaf(
        id="item4_airs_monday_through_friday",
        desc="Show airs Monday through Friday.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The show '{item.name or ''}' airs Monday through Friday.",
        node=leaf_mf,
        sources=urls,
        additional_instruction="Look for the daily weekday schedule information on network/official pages."
    )

    # 3) Begins at 7:00 AM ET (evidence)
    leaf_7am = evaluator.add_leaf(
        id="item4_begins_at_7am_et",
        desc="Show begins at 7:00 AM Eastern Time.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The show '{item.name or ''}' begins at 7:00 AM ET (Eastern Time).",
        node=leaf_7am,
        sources=urls,
        additional_instruction="Confirm the stated start time for the 7–8 AM hour."
    )

    # 4) Network is NBC (evidence)
    leaf_nbc = evaluator.add_leaf(
        id="item4_network_is_nbc",
        desc="Broadcasting network is NBC.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The show '{item.name or ''}' airs on NBC.",
        node=leaf_nbc,
        sources=urls,
        additional_instruction="Check official network references."
    )

    # 5) Exactly two main co-anchors for 7–8 (existence/count)
    evaluator.add_custom_node(
        result=(len([h for h in (item.hosts_or_anchors or []) if isinstance(h, str) and h.strip()]) == 2),
        id="item4_exactly_two_main_coanchors_for_7_to_8",
        desc="There are exactly two main co-anchors hosting the 7:00–8:00 AM hour.",
        parent=node,
        critical=True
    )

    # 6) Co-anchor names provided (existence)
    evaluator.add_custom_node(
        result=count_nonempty(item.hosts_or_anchors) >= 2,
        id="item4_coanchor_names_provided",
        desc="Names of the two main co-anchors for the relevant 2025/2026 period are provided.",
        parent=node,
        critical=True
    )

    # 7) Broadcast date/range provided within window (existence + logic)
    evaluator.add_custom_node(
        result=bool(dtr.strip()) and overlaps_window(parsed),
        id="item4_broadcast_date_or_range_provided",
        desc="A specific broadcast date (or a clear date range) for the 2025/2026 occurrence within Nov 27, 2025–Feb 22, 2026 is provided.",
        parent=node,
        critical=True
    )

    # 8) Streaming platform provided (existence)
    evaluator.add_custom_node(
        result=count_nonempty(item.streaming_platforms) > 0,
        id="item4_streaming_platform_provided",
        desc="At least one streaming platform where the show can be watched is provided.",
        parent=node,
        critical=True
    )

    # 9) Reference URL provided and acceptable (evidence)
    if one_ref:
        leaf_ref = evaluator.add_leaf(
            id="item4_reference_url_provided_and_acceptable",
            desc="At least one reference URL is provided, and it is an official network/streaming page or a reliable news source confirming schedule/time/network and anchor lineup.",
            parent=node,
            critical=True
        )
        anchors_joined = ", ".join(item.hosts_or_anchors) if item.hosts_or_anchors else ""
        await evaluator.verify(
            claim=(
                f"This page is an official network/streaming page or a reputable news source confirming the schedule/time/network and, when applicable, "
                f"the 7–8 AM co-anchors ({anchors_joined}) for '{item.name or ''}'."
            ),
            node=leaf_ref,
            sources=one_ref,
            additional_instruction="Use the page to corroborate start time (7 AM ET), NBC network, weekday airing, and co-anchor lineup."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="item4_reference_url_provided_and_acceptable",
            desc="At least one reference URL is provided, and it is an official/reliable source.",
            parent=node,
            critical=True
        )


# --------------------------------------------------------------------------- #
# Tree-wide/global checks                                                     #
# --------------------------------------------------------------------------- #

def pick_items_by_category(extracted: ItemsExtraction) -> Dict[str, ProgramItem]:
    """
    Build a mapping from category name to the first corresponding ProgramItem.
    If a category is missing, map to a placeholder empty ProgramItem.
    """
    mapping: Dict[str, ProgramItem] = {}
    for it in extracted.items:
        if it.category in ALLOWED_CATEGORIES and it.category not in mapping:
            mapping[it.category] = it

    # Ensure all four categories exist
    for cat in ALLOWED_CATEGORIES:
        if cat not in mapping:
            mapping[cat] = ProgramItem(category=cat)

    return mapping


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    Evaluate an answer for the four-programs/events task using the Mind2Web2 framework.
    """
    # Initialize evaluator (root as NON-CRITICAL parallel to avoid critical-child constraint)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=(
            "Identify four distinct recurring U.S. TV programs/events with occurrences within Nov 27, 2025–Feb 22, 2026: "
            "one per category (Thanksgiving afternoon special, Christmas morning special, February 2026 winter sporting event in Europe, weekday morning show). "
            "Each item must satisfy its category constraints and provide required fields (name, date(s), network, time/date-range, hosts/anchors/presenters where applicable, "
            "streaming platform, reference URL)."
        ),
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # --------------------- Extraction --------------------- #
    extracted: ItemsExtraction = await evaluator.extract(
        prompt=prompt_extract_items(),
        template_class=ItemsExtraction,
        extraction_name="items_extraction",
    )

    # Normalize to exactly one per category
    by_cat = pick_items_by_category(extracted)
    ordered_items = [
        by_cat["Thanksgiving Afternoon Special"],
        by_cat["Christmas Morning Special"],
        by_cat["Winter Sporting Event"],
        by_cat["Weekday Morning Show"],
    ]

    # --------------------- Global checks ------------------ #
    # 1) Exactly four items, one per category
    evaluator.add_custom_node(
        result=(len(extracted.items) >= 4 and categories_covered_once(ordered_items)),
        id="four_items_present_one_per_category",
        desc="Response includes exactly four items, matching Categories 1–4 (one item per category).",
        parent=root,
        critical=True
    )

    # 2) Distinct items (by official name)
    evaluator.add_custom_node(
        result=distinct_names(ordered_items),
        id="distinct_items",
        desc="All four selected programs/events are distinct (no duplicates).",
        parent=root,
        critical=True
    )

    # ------------------ Per-category verification --------- #
    # Category 1
    await verify_category_1_thanksgiving(evaluator, root, ordered_items[0])

    # Category 2
    await verify_category_2_christmas(evaluator, root, ordered_items[1])

    # Category 3
    await verify_category_3_winter_event(evaluator, root, ordered_items[2])

    # Category 4
    await verify_category_4_weekday_show(evaluator, root, ordered_items[3])

    # ------------------ Return structured result ---------- #
    return evaluator.get_summary()