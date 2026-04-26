import asyncio
import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "broadway_week_20260316_4shows"
TASK_DESCRIPTION = (
    "Identify four currently running Broadway shows that have scheduled performances during the week of "
    "March 16-22, 2026. For each show, provide: (1) Show Name, (2) Theater Name and Address, "
    "(3) Geographic Verification (between 41st-54th St and 6th-9th Ave), (4) Official Broadway Status, "
    "(5) Performance Schedule for Mar 16–22, 2026 (>= 3 performance days), (6) Ticket Availability via official channels, "
    "(7) At least one official reference URL (Broadway.org, Playbill, Broadway.com, the theater’s official site)."
)

WEEK_START = date(2026, 3, 16)
WEEK_END = date(2026, 3, 22)

# Allowed/official domains (reference and ticket channels)
ALLOWED_REFERENCE_DOMAINS = {
    "broadway.org",
    "playbill.com",
    "broadway.com",
    "ticketmaster.com",
}
# Heuristic substrings for theatre official sites (accepting well-known operators, and obvious theatre domains)
OFFICIAL_THEATRE_DOMAIN_SUBSTR = [
    "theatre", "theater", "shubert", "jujamcyn", "nederlander", "roundabout", "nycitycenter", "lincolncenter",
    "hudsonbroadway", "circleon", "newamsterdam", "alhirschfeld", "broadhursttheatre", "imperialtheatre",
    "majestic", "minskofftheatre", "stjames", "wintergarden", "palacetheatre", "ambassadortheatre",
]

# Ticket channels (superset; includes official league listing sites with ticket CTAs and primary ticketing)
ALLOWED_TICKET_DOMAINS = {
    "ticketmaster.com",
    "telecharge.com",
    "seatgeek.com",
    "broadway.com",
    "broadway.org",
}


# --------------------------------------------------------------------------- #
# Pydantic data models for extraction                                         #
# --------------------------------------------------------------------------- #
class PerformanceSlot(BaseModel):
    date: Optional[str] = None  # Prefer ISO: YYYY-MM-DD within 2026-03-16..2026-03-22
    day: Optional[str] = None   # e.g., Monday, Tue, Wed
    time: Optional[str] = None  # e.g., 7:00 PM, 2 PM


class ShowItem(BaseModel):
    show_name: Optional[str] = None
    theater_name: Optional[str] = None
    theater_address: Optional[str] = None  # Full Manhattan street address
    reference_urls: List[str] = Field(default_factory=list)  # schedule/theatre confirmation
    ticket_urls: List[str] = Field(default_factory=list)     # purchase channels
    schedule: List[PerformanceSlot] = Field(default_factory=list)  # only entries within Mar 16–22, 2026


class ShowsExtraction(BaseModel):
    shows: List[ShowItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_shows() -> str:
    return f"""
Extract up to four Broadway shows exactly as presented in the answer that claim to have scheduled performances during the week of March 16–22, 2026 (inclusive). Return a JSON object with a 'shows' array; each element must include:

- show_name: The name of the Broadway show as written in the answer.
- theater_name: The name of the theater where the show is performed.
- theater_address: The complete Manhattan street address for the theater (e.g., "242 West 45th Street, New York, NY 10036"). Do not invent—extract exactly from the answer.
- reference_urls: A list of official reference URLs cited in the answer that confirm the show’s schedule and theater info. Allowed: broadway.org, playbill.com, broadway.com, ticketmaster.com, OR the theater’s official website. Extract only explicit URLs present in the answer.
- ticket_urls: A list of URLs in the answer where tickets can be purchased via official channels (e.g., Ticketmaster, Telecharge, SeatGeek, Broadway.com, Broadway.org, or the theater’s official website). Extract only explicit URLs present in the answer.
- schedule: A list of performance slots during the specific week (Mar 16–22, 2026). For each slot, extract:
  - date: Use ISO format YYYY-MM-DD for dates within 2026-03-16..2026-03-22 only.
  - day: Optional day-of-week string (e.g., Monday, Tue).
  - time: The performance time string (e.g., "7:00 PM", "2 PM").

Rules:
- Only include up to the first four shows from the answer. If fewer than four are present, include as many as available.
- For schedule, include ONLY entries that are explicitly for Mar 16–22, 2026. Omit entries outside this week.
- If a field is missing, set it to null (for strings) or [] (for lists).
- Extract only actual URLs as they appear. Do not infer or fabricate URLs. If none are present for a field, return an empty list.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _domain_from_url(u: str) -> Optional[str]:
    try:
        from urllib.parse import urlparse
        if not u:
            return None
        parsed = urlparse(u if "://" in u else f"http://{u}")
        host = parsed.netloc.lower()
        return host.split(":")[0]
    except Exception:
        return None


def _is_official_reference_url(u: str) -> bool:
    d = _domain_from_url(u) or ""
    if not d:
        return False
    # Direct allow-list
    for allowed in ALLOWED_REFERENCE_DOMAINS:
        if d.endswith(allowed):
            return True
    # Theatre/operator heuristics
    for sub in OFFICIAL_THEATRE_DOMAIN_SUBSTR:
        if sub in d:
            return True
    return False


def _is_official_ticket_url(u: str) -> bool:
    d = _domain_from_url(u) or ""
    if not d:
        return False
    for allowed in ALLOWED_TICKET_DOMAINS:
        if d.endswith(allowed):
            return True
    # Theatre/operator heuristics count as official purchase channels, too
    for sub in OFFICIAL_THEATRE_DOMAIN_SUBSTR:
        if sub in d:
            return True
    return False


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        key = u.strip()
        if key and key not in seen:
            out.append(key)
            seen.add(key)
    return out


def _parse_iso_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        # Pydantic extraction instructs ISO; support flexible but prefer YYYY-MM-DD
        return datetime.fromisoformat(s.strip()).date()
    except Exception:
        try:
            return datetime.strptime(s.strip(), "%Y-%m-%d").date()
        except Exception:
            return None


def _date_in_week(d: Optional[date]) -> bool:
    if not d:
        return False
    return WEEK_START <= d <= WEEK_END


def _count_unique_performance_days(slots: List[PerformanceSlot]) -> int:
    days: set[str] = set()
    for s in slots:
        di = _parse_iso_date(s.date)
        if di and _date_in_week(di):
            days.add(di.isoformat())
        elif not di and s.day:
            # Fallback: use provided day-of-week label as proxy if no date
            days.add(s.day.strip().lower())
    return len(days)


def _format_schedule_for_claim(slots: List[PerformanceSlot]) -> str:
    items = []
    for s in slots:
        part_date = s.date if s.date else (s.day or "")
        time = s.time or ""
        piece = part_date.strip()
        if s.day and s.date:
            piece = f"{s.day} {s.date}"
        if time:
            piece = f"{piece} {time}"
        if piece:
            items.append(piece)
    return "; ".join(items)


def _ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][n - 1] if 1 <= n <= 5 else f"#{n}"


# --------------------------------------------------------------------------- #
# Verification for one show                                                   #
# --------------------------------------------------------------------------- #
async def verify_one_show(
    evaluator: Evaluator,
    parent_node,
    show: ShowItem,
    idx: int,
) -> None:
    show_node = evaluator.add_parallel(
        id=f"show_{idx}",
        desc=f"{_ordinal(idx)} qualifying Broadway show meeting all criteria",
        parent=parent_node,
        critical=False,  # each show contributes to partial credit at root
    )

    # Normalize URL lists and filter by official rules
    ref_urls_all = _dedup_urls(show.reference_urls or [])
    official_refs = [u for u in ref_urls_all if _is_official_reference_url(u)]
    tix_urls_all = _dedup_urls(show.ticket_urls or [])
    official_tix = [u for u in tix_urls_all if _is_official_ticket_url(u)]
    # Fallback: sometimes purchase CTAs are on reference pages (e.g., Playbill/Broadway.com)
    tix_sources_to_use = official_tix if official_tix else [u for u in official_refs if _is_official_ticket_url(u)]

    # 1) Show name provided (existence)
    evaluator.add_custom_node(
        result=bool(show.show_name and show.show_name.strip()),
        id=f"show_{idx}_show_name",
        desc="The name of the Broadway show is provided",
        parent=show_node,
        critical=True,
    )

    # 2) Theater name provided (existence)
    evaluator.add_custom_node(
        result=bool(show.theater_name and show.theater_name.strip()),
        id=f"show_{idx}_theater_name",
        desc="The name of the theater where the show is performed is provided",
        parent=show_node,
        critical=True,
    )

    # 3) At least one official reference URL present (existence + domain check)
    evaluator.add_custom_node(
        result=len(official_refs) > 0,
        id=f"show_{idx}_reference_url",
        desc="A reference URL from an official source (Broadway.org, Playbill, Broadway.com, or theater's official website) is provided",
        parent=show_node,
        critical=True,
    )

    # 4) Theater address verified via references
    addr_node = evaluator.add_leaf(
        id=f"show_{idx}_theater_address",
        desc="A complete street address in Manhattan is provided for the theater",
        parent=show_node,
        critical=True,
    )
    addr_claim = f"The official address of the theater '{show.theater_name or ''}' is '{show.theater_address or ''}' in Manhattan, New York."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_node,
        sources=official_refs,
        additional_instruction="Verify that the cited page explicitly lists the same full street address for the theater in Manhattan (NYC). Minor formatting/punctuation differences are acceptable.",
    )

    # 5) Geographic bounding box verification
    geo_node = evaluator.add_leaf(
        id=f"show_{idx}_theater_location",
        desc="The theater is located between 41st and 54th Streets, and between Sixth and Ninth Avenues in Manhattan",
        parent=show_node,
        critical=True,
    )
    geo_claim = (
        f"The theater located at '{show.theater_address or ''}' lies within Manhattan's Broadway theatre district bounds: "
        f"between 41st and 54th Streets, and between Sixth Avenue (Avenue of the Americas) and Ninth Avenue (inclusive)."
    )
    await evaluator.verify(
        claim=geo_claim,
        node=geo_node,
        sources=official_refs,
        additional_instruction=(
            "Use the address and any nearby cross-street/avenue cues on the page. "
            "Interpret street numbers like 'West 44th Street' as within 41st–54th. "
            "Treat Broadway or Seventh Avenue frontage as within the 6th–9th Ave band. "
            "If the page clearly shows a map/address confirming this area, consider it supported."
        ),
    )

    # 6) Official Broadway designation verification
    broadway_node = evaluator.add_leaf(
        id=f"show_{idx}_broadway_designation",
        desc="The theater is one of the 41 officially recognized Broadway theaters",
        parent=show_node,
        critical=True,
    )
    broadway_claim = (
        f"'{show.theater_name or ''}' is one of the 41 officially recognized Broadway theaters (a Broadway house)."
    )
    await evaluator.verify(
        claim=broadway_claim,
        node=broadway_node,
        sources=official_refs,
        additional_instruction=(
            "Confirm that the cited page indicates the venue is a Broadway theatre/house (e.g., on Broadway.org theatre listings, "
            "or explicitly labeled 'Broadway theatre' on an official/theatre site)."
        ),
    )

    # 7) Has >= 3 performance days within week (existence/count check from extracted schedule)
    unique_days_count = _count_unique_performance_days(show.schedule or [])
    evaluator.add_custom_node(
        result=unique_days_count >= 3,
        id=f"show_{idx}_performance_week",
        desc="The show has scheduled performances during at least three days in the week of March 16-22, 2026",
        parent=show_node,
        critical=True,
    )

    # 8) Specific performance days and times verified via references
    perf_times_node = evaluator.add_leaf(
        id=f"show_{idx}_performance_times",
        desc="Specific performance days and times are listed for the scheduled performances during March 16-22, 2026",
        parent=show_node,
        critical=True,
    )
    schedule_str = _format_schedule_for_claim(show.schedule or [])
    perf_claim = (
        f"During the week of 2026-03-16 to 2026-03-22 inclusive, the show '{show.show_name or ''}' has scheduled performances as follows: "
        f"{schedule_str}"
    )
    await evaluator.verify(
        claim=perf_claim,
        node=perf_times_node,
        sources=official_refs,
        additional_instruction=(
            "Check that the cited page explicitly lists these performance dates/times for the specific week of Mar 16–22, 2026. "
            "Minor formatting variations (e.g., '7 PM' vs. '7:00 PM') are acceptable."
        ),
    )

    # 9) Ticket availability via official channels verified
    ticket_node = evaluator.add_leaf(
        id=f"show_{idx}_ticket_availability",
        desc="Tickets are confirmed to be available for purchase through official channels during the specified week",
        parent=show_node,
        critical=True,
    )
    # If no dedicated ticket URLs, try references that act as official purchase funnels (e.g., Playbill/Broadway.com)
    ticket_sources = tix_sources_to_use if tix_sources_to_use else official_refs
    tix_claim = (
        "Tickets are available for purchase for this show through at least one official channel among the provided pages "
        "(e.g., a 'Buy Tickets' button or primary ticketing checkout)."
    )
    await evaluator.verify(
        claim=tix_claim,
        node=ticket_node,
        sources=ticket_sources,
        additional_instruction=(
            "Confirm that the page provides an official purchase pathway (Ticketmaster/Telecharge/SeatGeek/Broadway.com/"
            "Broadway.org/theatre site box office). General availability on or around the specified week is acceptable."
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

    # Record task-week info
    evaluator.add_custom_info(
        info={
            "week_range": {"start": WEEK_START.isoformat(), "end": WEEK_END.isoformat()},
            "allowed_reference_domains": sorted(list(ALLOWED_REFERENCE_DOMAINS)),
            "allowed_ticket_domains": sorted(list(ALLOWED_TICKET_DOMAINS)),
            "theatre_domain_heuristics": OFFICIAL_THEATRE_DOMAIN_SUBSTR,
        },
        info_type="task_requirements",
        info_name="broadway_task_requirements",
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_shows(),
        template_class=ShowsExtraction,
        extraction_name="shows_extraction",
    )

    # Build tree: four parallel show groups
    shows: List[ShowItem] = list(extracted.shows or [])
    # Ensure exactly 4 slots (pad with empty if fewer)
    if len(shows) < 4:
        shows = shows + [ShowItem() for _ in range(4 - len(shows))]
    else:
        shows = shows[:4]

    # Add a parent node is already root (parallel). Each show subtree is parallel as per rubric.
    # Verify each show
    for idx in range(1, 5):
        await verify_one_show(evaluator, root, shows[idx - 1], idx)

    return evaluator.get_summary()