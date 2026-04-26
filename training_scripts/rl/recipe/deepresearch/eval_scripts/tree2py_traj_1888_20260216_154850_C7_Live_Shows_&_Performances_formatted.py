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
TASK_ID = "nyc_arena_2026"
TASK_DESCRIPTION = (
    "I'm planning a trip to New York City in spring 2026 and want to visit a major sporting event. "
    "Find the major indoor arena located in Manhattan that serves as the home venue for both an NBA team and an NHL team. "
    "Provide the following information about this arena: (1) The name of the arena, (2) The complete street address, "
    "(3) The maximum seating capacity, (4) The seating capacity for concerts, (5) The seating capacity for basketball games, "
    "(6) The seating capacity for hockey games, (7) The name of the NBA team that plays there, (8) The name of the NHL team that plays there, "
    "(9) The year the arena opened or was last renovated, (10) The arena's official website URL, (11) Information about where to purchase tickets "
    "(website or box office details), and (12) The name and date of at least one upcoming event scheduled at the arena between March 1 and May 31, 2026."
)

DATE_RANGE_START = date(2026, 3, 1)
DATE_RANGE_END = date(2026, 5, 31)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventItem(BaseModel):
    name: Optional[str] = None
    date: Optional[str] = None  # Keep as string; we'll parse
    source_urls: List[str] = Field(default_factory=list)


class ArenaExtraction(BaseModel):
    arena_name: Optional[str] = None

    # Official website
    official_website_url: Optional[str] = None

    # Indoor arena verification sources
    indoor_arena_sources: List[str] = Field(default_factory=list)

    # Address and sources
    street_address: Optional[str] = None
    street_address_sources: List[str] = Field(default_factory=list)

    # Teams and home-venue sources
    nba_team_name: Optional[str] = None
    nba_home_source_urls: List[str] = Field(default_factory=list)
    nhl_team_name: Optional[str] = None
    nhl_home_source_urls: List[str] = Field(default_factory=list)

    # Capacities + sources
    capacity_max: Optional[str] = None
    capacity_max_sources: List[str] = Field(default_factory=list)
    capacity_concert: Optional[str] = None
    capacity_concert_sources: List[str] = Field(default_factory=list)
    capacity_basketball: Optional[str] = None
    capacity_basketball_sources: List[str] = Field(default_factory=list)
    capacity_hockey: Optional[str] = None
    capacity_hockey_sources: List[str] = Field(default_factory=list)

    # Opening or last renovation year + sources
    opening_or_last_renovation_year: Optional[str] = None
    opening_or_last_renovation_year_sources: List[str] = Field(default_factory=list)

    # Ticket info (text blurb + urls)
    ticket_info_text: Optional[str] = None
    ticket_info_urls: List[str] = Field(default_factory=list)

    # Upcoming events
    upcoming_events: List[EventItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_arena_info() -> str:
    return (
        "Extract the arena details from the answer. The expected arena is a major indoor arena located in Manhattan, NYC, "
        "that serves as the home venue for an NBA team and an NHL team.\n\n"
        "Return a single JSON object with the following fields:\n"
        "1) arena_name: string (the name of the arena)\n"
        "2) official_website_url: string (the arena's official website URL; must be a URL shown in the answer)\n"
        "3) indoor_arena_sources: array of strings (URLs cited in the answer that support the arena being an indoor arena)\n"
        "4) street_address: string (the complete street address as given in the answer)\n"
        "5) street_address_sources: array of strings (URLs cited in the answer that support the address)\n"
        "6) nba_team_name: string (NBA team name as given in the answer)\n"
        "7) nba_home_source_urls: array of strings (URLs cited in the answer that support that this arena is the NBA team's home venue)\n"
        "8) nhl_team_name: string (NHL team name as given in the answer)\n"
        "9) nhl_home_source_urls: array of strings (URLs cited in the answer that support that this arena is the NHL team's home venue)\n"
        "10) capacity_max: string (the maximum seating capacity as presented in the answer; keep formatting as-is)\n"
        "11) capacity_max_sources: array of strings (URLs cited in the answer that support the maximum capacity)\n"
        "12) capacity_concert: string (concert seating capacity as presented in the answer; keep formatting as-is)\n"
        "13) capacity_concert_sources: array of strings (URLs cited in the answer that support the concert capacity)\n"
        "14) capacity_basketball: string (basketball seating capacity as presented in the answer; keep formatting as-is)\n"
        "15) capacity_basketball_sources: array of strings (URLs cited in the answer that support the basketball capacity)\n"
        "16) capacity_hockey: string (hockey seating capacity as presented in the answer; keep formatting as-is)\n"
        "17) capacity_hockey_sources: array of strings (URLs cited in the answer that support the hockey capacity)\n"
        "18) opening_or_last_renovation_year: string (the opening year or the year of the last renovation as presented)\n"
        "19) opening_or_last_renovation_year_sources: array of strings (URLs cited in the answer that support that year)\n"
        "20) ticket_info_text: string (the provided ticket purchase/box office info blurb)\n"
        "21) ticket_info_urls: array of strings (URLs cited in the answer for ticketing or box office info)\n"
        "22) upcoming_events: array of objects, each with:\n"
        "    - name: string (event name exactly as shown in the answer)\n"
        "    - date: string (event date exactly as shown; can be 'YYYY-MM-DD' or 'Month DD, YYYY' etc.)\n"
        "    - source_urls: array of strings (URLs cited in the answer for that event)\n\n"
        "Rules:\n"
        "- Only extract URLs that are explicitly present in the answer text; do not infer URLs.\n"
        "- If any field is missing in the answer, return null for the string field or an empty array for URL lists.\n"
        "- Preserve numbers and formats as written in the answer; do not normalize.\n"
        "- For upcoming_events, include all events mentioned in the answer; do not fabricate dates.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def combine_sources(*lists: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in lists:
        if not lst:
            continue
        combined.extend(lst)
    return _dedup_urls(combined)


def pick_sources(preferred: List[str], *fallback_lists: List[str]) -> List[str]:
    if preferred and len([u for u in preferred if u]) > 0:
        return _dedup_urls(preferred)
    for fb in fallback_lists:
        if fb and len([u for u in fb if u]) > 0:
            return _dedup_urls(fb)
    return []


def as_list(url_or_none: Optional[str]) -> List[str]:
    return [url_or_none] if url_or_none else []


def _strip_ordinals(s: str) -> str:
    return re.sub(r'(\d{1,2})(st|nd|rd|th)\b', r'\1', s)


def parse_date_str(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    txt = _strip_ordinals(s.strip())
    # Try extracting clear ISO pattern first
    m = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', txt)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            pass
    # Remove time parts if present
    txt = re.split(r'\bat\b|\b@|\b\d{1,2}:\d{2}\b', txt)[0].strip()
    # Common patterns
    patterns = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%d %B %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%Y/%m/%d",
    ]
    for p in patterns:
        try:
            dt = datetime.strptime(txt, p)
            return dt.date()
        except Exception:
            continue
    # Try to find a month name and a year
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }
    r = re.search(r'(?P<mon>January|February|March|April|May|June|July|August|September|October|November|December)'
                  r'\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})', txt, flags=re.IGNORECASE)
    if r:
        try:
            mon = months[r.group('mon').lower()]
            day = int(r.group('day'))
            yr = int(r.group('year'))
            return date(yr, mon, day)
        except Exception:
            pass
    return None


def select_event_in_range(events: List[EventItem]) -> Optional[EventItem]:
    for ev in events:
        d = parse_date_str(ev.date)
        if d and DATE_RANGE_START <= d <= DATE_RANGE_END and ev.name and ev.name.strip():
            return ev
    return None


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def verify_arena_identity_and_location(evaluator: Evaluator, parent_node, data: ArenaExtraction) -> None:
    node = evaluator.add_parallel(
        id="arena_identity_and_location",
        desc="Core identifying information, indoor-arena status, and Manhattan location.",
        parent=parent_node,
        critical=True
    )

    # 1) Arena Name
    name_leaf = evaluator.add_leaf(
        id="arena_name",
        desc="Provide the name of the arena.",
        parent=node,
        critical=True
    )
    arena_name = data.arena_name or ""
    name_sources = pick_sources(
        as_list(data.official_website_url),
        data.nba_home_source_urls,
        data.nhl_home_source_urls,
        data.street_address_sources,
        data.indoor_arena_sources,
        data.capacity_max_sources + data.capacity_concert_sources + data.capacity_basketball_sources + data.capacity_hockey_sources,
    )
    await evaluator.verify(
        claim=f"The arena's official name is '{arena_name}'.",
        node=name_leaf,
        sources=name_sources,
        additional_instruction="Verify that the page clearly identifies the arena name (minor variations like inclusion/exclusion of 'The' or 'Arena' are acceptable)."
    )

    # 2) Indoor Arena Verifiable
    indoor_leaf = evaluator.add_leaf(
        id="indoor_arena_verifiable",
        desc="Verify the venue is an indoor arena (e.g., described as an indoor arena) and provide a verifiable source URL.",
        parent=node,
        critical=True
    )
    indoor_sources = pick_sources(
        data.indoor_arena_sources,
        as_list(data.official_website_url),
        data.nba_home_source_urls,
        data.nhl_home_source_urls
    )
    await evaluator.verify(
        claim="This venue is an indoor arena (or indoor multi‑purpose arena).",
        node=indoor_leaf,
        sources=indoor_sources,
        additional_instruction="Confirm that the page explicitly or implicitly indicates an indoor arena (e.g., 'indoor arena', 'arena hosting NBA/NHL games'). Do not accept outdoor stadiums."
    )

    # 3) Street Address in Manhattan
    address_leaf = evaluator.add_leaf(
        id="street_address_manhattan",
        desc="Provide a complete, specific street address in Manhattan, NYC, and include a verifiable citation/source URL for the address.",
        parent=node,
        critical=True
    )
    address = data.street_address or ""
    address_sources = pick_sources(
        data.street_address_sources,
        as_list(data.official_website_url)
    )
    await evaluator.verify(
        claim=f"The arena's complete street address is '{address}', and it is located in Manhattan, New York City.",
        node=address_leaf,
        sources=address_sources,
        additional_instruction="Verify that the page lists the full street address (number + street + city/state + ZIP). Ensure the address is in Manhattan (borough of NYC). Reject addresses located in other boroughs like Brooklyn or in other cities."
    )


async def verify_home_venue_requirements(evaluator: Evaluator, parent_node, data: ArenaExtraction) -> None:
    node = evaluator.add_parallel(
        id="home_venue_requirements",
        desc="Arena must serve as the home venue for both an NBA team and an NHL team; team names must be accurate/current.",
        parent=parent_node,
        critical=True
    )

    # NBA
    nba_leaf = evaluator.add_leaf(
        id="nba_home_team_accurate",
        desc="Provide the NBA team name and indicate the arena is that team's home venue (must be accurate/current).",
        parent=node,
        critical=True
    )
    nba_team = data.nba_team_name or ""
    nba_sources = pick_sources(
        data.nba_home_source_urls,
        as_list(data.official_website_url)
    )
    await evaluator.verify(
        claim=f"The NBA team '{nba_team}' uses this arena as its home venue.",
        node=nba_leaf,
        sources=nba_sources,
        additional_instruction="Confirm that the page states the NBA team uses this arena as its home court/home arena (not just occasional games). Prefer official team or arena pages."
    )

    # NHL
    nhl_leaf = evaluator.add_leaf(
        id="nhl_home_team_accurate",
        desc="Provide the NHL team name and indicate the arena is that team's home venue (must be accurate/current).",
        parent=node,
        critical=True
    )
    nhl_team = data.nhl_team_name or ""
    nhl_sources = pick_sources(
        data.nhl_home_source_urls,
        as_list(data.official_website_url)
    )
    await evaluator.verify(
        claim=f"The NHL team '{nhl_team}' uses this arena as its home venue.",
        node=nhl_leaf,
        sources=nhl_sources,
        additional_instruction="Confirm that the page states the NHL team uses this arena as its home ice/home arena (not just occasional games). Prefer official team or arena pages."
    )


async def verify_seating_capacities(evaluator: Evaluator, parent_node, data: ArenaExtraction) -> None:
    node = evaluator.add_parallel(
        id="seating_capacities_official",
        desc="Provide all required seating capacities, each documented via an official source.",
        parent=parent_node,
        critical=True
    )

    # Maximum
    max_leaf = evaluator.add_leaf(
        id="maximum_capacity_documented",
        desc="Provide the maximum seating capacity as a number AND cite an official source URL documenting it.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The arena's maximum seating capacity is {data.capacity_max}.",
        node=max_leaf,
        sources=data.capacity_max_sources,
        additional_instruction="Accept only if the provided URL is an official source (arena/operator/team). Do NOT accept Wikipedia or third-party fan sites as official documentation. Minor rounding differences are acceptable."
    )

    # Concerts
    concert_leaf = evaluator.add_leaf(
        id="concert_capacity_documented",
        desc="Provide the concert seating capacity as a number AND cite an official source URL documenting it.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The seating capacity for concerts at this arena is {data.capacity_concert}.",
        node=concert_leaf,
        sources=data.capacity_concert_sources,
        additional_instruction="Accept only if the provided URL is an official source (arena/operator/team). Do NOT accept Wikipedia or third-party fan sites as official documentation. Minor rounding differences are acceptable."
    )

    # Basketball
    basketball_leaf = evaluator.add_leaf(
        id="basketball_capacity_documented",
        desc="Provide the basketball seating capacity as a number AND cite an official source URL documenting it.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The seating capacity for basketball games at this arena is {data.capacity_basketball}.",
        node=basketball_leaf,
        sources=data.capacity_basketball_sources,
        additional_instruction="Accept only if the provided URL is an official source (arena/operator/team). Do NOT accept Wikipedia or third-party fan sites as official documentation. Minor rounding differences are acceptable."
    )

    # Hockey
    hockey_leaf = evaluator.add_leaf(
        id="hockey_capacity_documented",
        desc="Provide the hockey seating capacity as a number AND cite an official source URL documenting it.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The seating capacity for hockey games at this arena is {data.capacity_hockey}.",
        node=hockey_leaf,
        sources=data.capacity_hockey_sources,
        additional_instruction="Accept only if the provided URL is an official source (arena/operator/team). Do NOT accept Wikipedia or third-party fan sites as official documentation. Minor rounding differences are acceptable."
    )


async def verify_opening_or_renovation_year(evaluator: Evaluator, parent_node, data: ArenaExtraction) -> None:
    node = evaluator.add_parallel(
        id="opening_or_renovation_year",
        desc="Provide the year the arena opened OR was last renovated, with documentation.",
        parent=parent_node,
        critical=True
    )

    year_leaf = evaluator.add_leaf(
        id="opening_or_last_renovation_year_with_source",
        desc="Provide the opening year or last renovation year AND cite a source URL documenting it.",
        parent=node,
        critical=True
    )
    year_txt = data.opening_or_last_renovation_year or ""
    await evaluator.verify(
        claim=f"The arena opened or was last renovated in {year_txt}.",
        node=year_leaf,
        sources=data.opening_or_last_renovation_year_sources,
        additional_instruction="Verify the year explicitly appears on the provided source. Prefer official arena/operator sources; do not accept Wikipedia unless it is the official site."
    )


async def verify_official_website_and_ticketing(evaluator: Evaluator, parent_node, data: ArenaExtraction) -> None:
    node = evaluator.add_parallel(
        id="official_website_and_ticketing",
        desc="Official website and ticket-purchase details must be provided; official site must have event/ticketing information.",
        parent=parent_node,
        critical=True
    )

    # Official website
    website_leaf = evaluator.add_leaf(
        id="official_website_url_with_events_ticketing",
        desc="Provide the arena's official website URL, and it must be an official venue site that contains event listings and ticketing information.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This URL is the arena's official website and it contains event listings and ticketing information.",
        node=website_leaf,
        sources=as_list(data.official_website_url),
        additional_instruction="Verify that the site is the official arena site (e.g., venue/operator domain) and that it visibly includes Events and/or Tickets information (e.g., navigation items or listings)."
    )

    # Ticket purchase information
    ticket_leaf = evaluator.add_leaf(
        id="ticket_purchase_information_available",
        desc="Provide where/how to purchase tickets (ticketing website and/or box office details).",
        parent=node,
        critical=True
    )
    ticket_sources = pick_sources(
        data.ticket_info_urls,
        as_list(data.official_website_url)
    )
    ticket_info_text = data.ticket_info_text or ""
    await evaluator.verify(
        claim="The provided ticketing URL(s) offer a way to purchase tickets or include box office details for the arena.",
        node=ticket_leaf,
        sources=ticket_sources,
        additional_instruction="Look for 'Tickets', 'Buy Tickets', 'Box Office', or a ticketing platform link (e.g., Ticketmaster) clearly associated with the arena. The page should support the provided ticket information text."
    )


async def verify_upcoming_event(evaluator: Evaluator, parent_node, data: ArenaExtraction) -> None:
    node = evaluator.add_parallel(
        id="upcoming_event_in_range",
        desc="At least one upcoming arena event between March 1 and May 31, 2026 (inclusive), with required details.",
        parent=parent_node,
        critical=True
    )

    event_leaf = evaluator.add_leaf(
        id="event_name_and_specific_date_in_range_with_source",
        desc="Provide at least one event name AND a specific event date between 2026-03-01 and 2026-05-31 (inclusive), AND cite a source URL (e.g., official event listing) verifying the event and date.",
        parent=node,
        critical=True
    )

    # Select an event in the required range (if any)
    selected = select_event_in_range(data.upcoming_events)
    if selected:
        claim = f"An event named '{selected.name}' is scheduled at the arena on {selected.date}."
        sources = selected.source_urls
        # Record a bit of info for debugging
        evaluator.add_custom_info(
            info={"selected_event": {"name": selected.name, "date": selected.date, "sources": selected.source_urls}},
            info_type="debug",
            info_name="selected_event_in_range"
        )
    else:
        claim = "No qualifying event with a specific date in the required range was provided."
        sources = []  # This will likely fail the verification as required

    await evaluator.verify(
        claim=claim,
        node=event_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the event page(s) clearly show the event name and a specific date. "
            f"The date must fall between {DATE_RANGE_START.isoformat()} and {DATE_RANGE_END.isoformat()} inclusive. "
            "If the date is outside this range or not shown, consider the claim unsupported."
        )
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
    Evaluate an answer for the Manhattan indoor arena (NBA+NHL) task.
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
        default_model=model
    )

    # Extract all relevant information from the answer
    data: ArenaExtraction = await evaluator.extract(
        prompt=prompt_extract_arena_info(),
        template_class=ArenaExtraction,
        extraction_name="arena_extraction"
    )

    # Create top-level critical node (mirrors rubric's Arena_Information as a critical root child)
    arena_info_node = evaluator.add_parallel(
        id="arena_information",
        desc="Identify the Manhattan indoor arena that is home to both an NBA team and an NHL team, and provide the required venue details with documentation where required.",
        parent=root,
        critical=True
    )

    # Build and verify each section
    await verify_arena_identity_and_location(evaluator, arena_info_node, data)
    await verify_home_venue_requirements(evaluator, arena_info_node, data)
    await verify_seating_capacities(evaluator, arena_info_node, data)
    await verify_opening_or_renovation_year(evaluator, arena_info_node, data)
    await verify_official_website_and_ticketing(evaluator, arena_info_node, data)
    await verify_upcoming_event(evaluator, arena_info_node, data)

    # Return evaluation summary
    return evaluator.get_summary()