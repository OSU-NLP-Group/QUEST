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
TASK_ID = "gaming_events_2026_schedule"
TASK_DESCRIPTION = """
You are planning a comprehensive travel schedule for a gaming industry professional who wants to attend major gaming-related events across different categories and regions in 2026. Identify five specific conventions or tournaments that meet the following criteria:

1. North American Tabletop Gaming Convention: One major tabletop gaming convention in North America (USA or Canada) taking place during summer 2026 (June 1 - August 31) with an expected or historical attendance of at least 50,000 people. Provide the exact dates, specific city and state/province, and a supporting reference URL.

2. European PC Gaming Convention: One major PC or video gaming convention in Europe taking place during August 2026 that spans at least 3 days (comparing the start and end dates). Provide the exact dates, specific city and country, and a supporting reference URL.

3. European Streaming/Content Creator Convention: One convention focused on streaming, content creation, or live broadcasting (such as Twitch-related events) in Europe taking place during the first half of 2026 (January 1 - June 30). Provide the exact dates, specific city and country, and a supporting reference URL.

4. Major Esports Tournament: One esports tournament or championship in 2026 with a total prize pool of at least $50,000,000 USD. Provide the start and end dates, specific city and country, total prize pool amount, and a supporting reference URL.

5. US West Coast Gaming Convention: One major gaming convention on the US West Coast (Washington, Oregon, or California) taking place during September 2026. Provide the exact dates, specific city, specific venue or convention center name, and a supporting reference URL.

For each event, provide:
- The event name
- Exact dates (start and end dates)
- Specific location (city, state/province/country)
- Additional required details (attendance figures, duration verification, prize pool, venue name, or focus area as specified above)
- A reference URL from an official or authoritative source
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventBase(BaseModel):
    event_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    city: Optional[str] = None
    state_or_province: Optional[str] = None
    country: Optional[str] = None
    reference_url: Optional[str] = None


class NATabletopEvent(EventBase):
    attendance: Optional[str] = None  # e.g., "50,000+", "70,000 attendees"


class EuropePCEvent(EventBase):
    pass  # dates + location verified separately


class EuropeStreamingEvent(EventBase):
    focus_area: Optional[str] = None  # e.g., "streaming", "content creation", "live broadcasting"


class EsportsTournamentEvent(EventBase):
    prize_pool_usd: Optional[str] = None  # e.g., "$50,000,000", "USD 60M"


class USWestCoastEvent(EventBase):
    venue_name: Optional[str] = None


class GamingEventsExtraction(BaseModel):
    north_america_tabletop: Optional[NATabletopEvent] = None
    europe_pc_convention: Optional[EuropePCEvent] = None
    europe_streaming_convention: Optional[EuropeStreamingEvent] = None
    major_esports_tournament: Optional[EsportsTournamentEvent] = None
    us_west_coast_convention: Optional[USWestCoastEvent] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract one event for each of the five categories described below from the answer. If multiple events are mentioned for a category, select the first clearly identified one. Do not invent any information; if a field is not explicitly present, return null.

    For each category, extract the following fields exactly as presented in the answer (use strings for all values):
    1) north_america_tabletop:
       - event_name
       - start_date
       - end_date
       - city
       - state_or_province
       - country
       - attendance  (expected or historical attendance, e.g., "70,000")
       - reference_url (URL string)

    2) europe_pc_convention:
       - event_name
       - start_date
       - end_date
       - city
       - country
       - reference_url

    3) europe_streaming_convention:
       - event_name
       - start_date
       - end_date
       - city
       - country
       - focus_area (e.g., "streaming", "content creation", "live broadcasting")
       - reference_url

    4) major_esports_tournament:
       - event_name
       - start_date
       - end_date
       - city
       - country
       - prize_pool_usd (e.g., "$50,000,000", "USD 60M")
       - reference_url

    5) us_west_coast_convention:
       - event_name
       - start_date
       - end_date
       - city
       - state_or_province
       - country
       - venue_name (e.g., "Seattle Convention Center")
       - reference_url

    IMPORTANT:
    - Only extract URLs explicitly present in the answer. If a URL is missing, set reference_url to null.
    - Keep dates and numbers as strings exactly as stated.
    - If a location field (state/province/country) is not specified, set it to null.
    - Return a single JSON object matching the GamingEventsExtraction schema.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    s = url.strip().lower()
    return s.startswith("http://") or s.startswith("https://")


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_na_tabletop(
    evaluator: Evaluator,
    parent_node,
    info: Optional[NATabletopEvent],
) -> None:
    node = evaluator.add_parallel(
        id="North_America_Tabletop_Convention",
        desc="Identify one major tabletop gaming convention in North America during summer 2026 (June-August) with expected attendance of at least 50,000 people",
        parent=parent_node,
        critical=False
    )

    name = info.event_name if info else None
    start = info.start_date if info else None
    end = info.end_date if info else None
    city = info.city if info else None
    state = info.state_or_province if info else None
    country = info.country if info else None
    attendance = info.attendance if info else None
    url = info.reference_url if info else None

    # Reference (existence/validity)
    ref_node = evaluator.add_custom_node(
        result=_valid_url(url),
        id="NA_Convention_Reference",
        desc="Provide a valid URL reference supporting the convention information",
        parent=node,
        critical=True
    )

    # Dates within summer 2026 (June 1 - Aug 31)
    dates_leaf = evaluator.add_leaf(
        id="NA_Convention_Dates",
        desc="Provide the exact dates of the convention and verify it takes place during summer 2026 (June 1 - August 31)",
        parent=node,
        critical=True
    )
    claim_dates = (
        f"The event '{name or 'the event'}' takes place from {start or 'UNKNOWN'} to {end or 'UNKNOWN'}, "
        f"and those dates fall between June 1, 2026 and August 31, 2026 (inclusive)."
    )
    await evaluator.verify(
        claim=claim_dates,
        node=dates_leaf,
        sources=url,
        extra_prerequisites=[ref_node],
        additional_instruction="Use the webpage to confirm the event's official 2026 dates and ensure they occur within summer 2026."
    )

    # Location city + state/province (North America)
    location_leaf = evaluator.add_leaf(
        id="NA_Convention_Location",
        desc="Provide the specific city and state/province where the convention takes place",
        parent=node,
        critical=True
    )
    location_str = f"{city or 'UNKNOWN'}, {state or 'UNKNOWN'}"
    claim_loc = (
        f"The event '{name or 'the event'}' takes place in {location_str} "
        f"{'(in ' + country + ')' if country else ''}."
    )
    await evaluator.verify(
        claim=claim_loc,
        node=location_leaf,
        sources=url,
        extra_prerequisites=[ref_node],
        additional_instruction="Confirm from the page the specific city and the state or province in North America (USA or Canada)."
    )

    # Attendance >= 50,000
    attendance_leaf = evaluator.add_leaf(
        id="NA_Convention_Attendance",
        desc="Verify the convention has an expected or historical attendance of at least 50,000 people",
        parent=node,
        critical=True
    )
    claim_att = (
        f"The event '{name or 'the event'}' has expected or historical attendance of at least 50,000 attendees "
        f"(the answer cites {attendance or 'UNKNOWN'})."
    )
    await evaluator.verify(
        claim=claim_att,
        node=attendance_leaf,
        sources=url,
        extra_prerequisites=[ref_node],
        additional_instruction="Check the page for prior years' attendance or official projections; pass only if ≥ 50,000 is supported."
    )


async def verify_europe_pc(
    evaluator: Evaluator,
    parent_node,
    info: Optional[EuropePCEvent],
) -> None:
    node = evaluator.add_parallel(
        id="Europe_PC_Gaming_Convention",
        desc="Identify one major PC/video gaming convention in Europe during August 2026 with duration of at least 3 days",
        parent=parent_node,
        critical=False
    )

    name = info.event_name if info else None
    start = info.start_date if info else None
    end = info.end_date if info else None
    city = info.city if info else None
    country = info.country if info else None
    url = info.reference_url if info else None

    ref_node = evaluator.add_custom_node(
        result=_valid_url(url),
        id="EU_Convention_Reference",
        desc="Provide a valid URL reference supporting the convention information",
        parent=node,
        critical=True
    )

    # Dates in August 2026
    dates_leaf = evaluator.add_leaf(
        id="EU_Convention_Dates",
        desc="Provide the exact dates of the convention and verify it takes place during August 2026",
        parent=node,
        critical=True
    )
    claim_dates = (
        f"The event '{name or 'the event'}' takes place from {start or 'UNKNOWN'} to {end or 'UNKNOWN'} "
        f"and these dates fall in August 2026."
    )
    await evaluator.verify(
        claim=claim_dates,
        node=dates_leaf,
        sources=url,
        extra_prerequisites=[ref_node],
        additional_instruction="Confirm from the page the event occurs in August 2026 (any dates within August)."
    )

    # Location city + country in Europe
    loc_leaf = evaluator.add_leaf(
        id="EU_Convention_Location",
        desc="Provide the specific city and country in Europe where the convention takes place",
        parent=node,
        critical=True
    )
    claim_loc = (
        f"The event '{name or 'the event'}' takes place in {city or 'UNKNOWN'}, {country or 'UNKNOWN'} in Europe."
    )
    await evaluator.verify(
        claim=claim_loc,
        node=loc_leaf,
        sources=url,
        extra_prerequisites=[ref_node],
        additional_instruction="Verify the city and country are in Europe and match what's shown on the page."
    )

    # Duration ≥ 3 days
    duration_leaf = evaluator.add_leaf(
        id="EU_Convention_Duration",
        desc="Verify the convention spans at least 3 days (comparing start and end dates)",
        parent=node,
        critical=True
    )
    claim_dur = (
        f"The event '{name or 'the event'}' spans at least 3 days based on its official start and end dates."
    )
    await evaluator.verify(
        claim=claim_dur,
        node=duration_leaf,
        sources=url,
        extra_prerequisites=[ref_node],
        additional_instruction="Use the start and end dates on the page; compute inclusive difference to verify ≥ 3 days."
    )


async def verify_europe_streaming(
    evaluator: Evaluator,
    parent_node,
    info: Optional[EuropeStreamingEvent],
) -> None:
    node = evaluator.add_parallel(
        id="Europe_Streaming_Convention",
        desc="Identify one streaming or content creator focused gaming convention in Europe during the first half of 2026 (January-June)",
        parent=parent_node,
        critical=False
    )

    name = info.event_name if info else None
    start = info.start_date if info else None
    end = info.end_date if info else None
    city = info.city if info else None
    country = info.country if info else None
    focus = info.focus_area if info else None
    url = info.reference_url if info else None

    ref_node = evaluator.add_custom_node(
        result=_valid_url(url),
        id="Stream_Convention_Reference",
        desc="Provide a valid URL reference supporting the convention information",
        parent=node,
        critical=True
    )

    # Dates in first half 2026
    dates_leaf = evaluator.add_leaf(
        id="Stream_Convention_Dates",
        desc="Provide the exact dates of the convention and verify it takes place during the first half of 2026 (January 1 - June 30)",
        parent=node,
        critical=True
    )
    claim_dates = (
        f"The event '{name or 'the event'}' takes place from {start or 'UNKNOWN'} to {end or 'UNKNOWN'}, "
        f"and these dates fall between January 1 and June 30, 2026 (inclusive)."
    )
    await evaluator.verify(
        claim=claim_dates,
        node=dates_leaf,
        sources=url,
        extra_prerequisites=[ref_node],
        additional_instruction="Confirm using the page that the event occurs in the first half of 2026."
    )

    # Location city + country in Europe
    loc_leaf = evaluator.add_leaf(
        id="Stream_Convention_Location",
        desc="Provide the specific city and country in Europe where the convention takes place",
        parent=node,
        critical=True
    )
    claim_loc = (
        f"The event '{name or 'the event'}' takes place in {city or 'UNKNOWN'}, {country or 'UNKNOWN'} in Europe."
    )
    await evaluator.verify(
        claim=claim_loc,
        node=loc_leaf,
        sources=url,
        extra_prerequisites=[ref_node],
        additional_instruction="Verify city and country on the page; ensure the location is in Europe."
    )

    # Focus on streaming/content creation/live broadcasting
    focus_leaf = evaluator.add_leaf(
        id="Stream_Convention_Focus",
        desc="Verify the convention is focused on streaming, content creation, or live broadcasting (e.g., Twitch-related)",
        parent=node,
        critical=True
    )
    claim_focus = (
        f"The event '{name or 'the event'}' is focused on streaming, content creation, or live broadcasting "
        f"(the answer cites focus area: {focus or 'UNKNOWN'})."
    )
    await evaluator.verify(
        claim=claim_focus,
        node=focus_leaf,
        sources=url,
        extra_prerequisites=[ref_node],
        additional_instruction="Look for page descriptors like 'streaming', 'content creator', 'Twitch', or 'live broadcasting.'"
    )


async def verify_esports_tournament(
    evaluator: Evaluator,
    parent_node,
    info: Optional[EsportsTournamentEvent],
) -> None:
    node = evaluator.add_parallel(
        id="Major_Esports_Tournament",
        desc="Identify one major esports tournament or championship in 2026 with total prize pool of at least $50,000,000 USD",
        parent=parent_node,
        critical=False
    )

    name = info.event_name if info else None
    start = info.start_date if info else None
    end = info.end_date if info else None
    city = info.city if info else None
    country = info.country if info else None
    prize = info.prize_pool_usd if info else None
    url = info.reference_url if info else None

    ref_node = evaluator.add_custom_node(
        result=_valid_url(url),
        id="Esports_Tournament_Reference",
        desc="Provide a valid URL reference supporting the tournament information",
        parent=node,
        critical=True
    )

    # Dates occur in 2026
    dates_leaf = evaluator.add_leaf(
        id="Esports_Tournament_Dates",
        desc="Provide the start and end dates of the tournament and verify it takes place in 2026",
        parent=node,
        critical=True
    )
    claim_dates = (
        f"The esports tournament '{name or 'the tournament'}' runs from {start or 'UNKNOWN'} to {end or 'UNKNOWN'} "
        f"and those dates fall in calendar year 2026."
    )
    await evaluator.verify(
        claim=claim_dates,
        node=dates_leaf,
        sources=url,
        extra_prerequisites=[ref_node],
        additional_instruction="Confirm the 2026 schedule on the page. If multi-stage, the main finals must occur in 2026."
    )

    # Location city + country
    loc_leaf = evaluator.add_leaf(
        id="Esports_Tournament_Location",
        desc="Provide the specific city and country where the tournament takes place",
        parent=node,
        critical=True
    )
    claim_loc = (
        f"The esports tournament '{name or 'the tournament'}' takes place in {city or 'UNKNOWN'}, {country or 'UNKNOWN'}."
    )
    await evaluator.verify(
        claim=claim_loc,
        node=loc_leaf,
        sources=url,
        extra_prerequisites=[ref_node],
        additional_instruction="Verify city and country from the official page or authoritative source."
    )

    # Prize pool ≥ $50,000,000 USD
    prize_leaf = evaluator.add_leaf(
        id="Esports_Tournament_Prize_Pool",
        desc="Verify the tournament has a total prize pool of at least $50,000,000 USD",
        parent=node,
        critical=True
    )
    claim_prize = (
        f"The total prize pool for '{name or 'the tournament'}' is at least $50,000,000 USD "
        f"(the answer cites {prize or 'UNKNOWN'})."
    )
    await evaluator.verify(
        claim=claim_prize,
        node=prize_leaf,
        sources=url,
        extra_prerequisites=[ref_node],
        additional_instruction="Confirm the page explicitly states a prize pool ≥ $50,000,000 USD (or clearly equivalent)."
    )


async def verify_us_west_coast(
    evaluator: Evaluator,
    parent_node,
    info: Optional[USWestCoastEvent],
) -> None:
    node = evaluator.add_parallel(
        id="US_West_Coast_Convention",
        desc="Identify one major gaming convention on the US West Coast (Washington, Oregon, or California) during September 2026",
        parent=parent_node,
        critical=False
    )

    name = info.event_name if info else None
    start = info.start_date if info else None
    end = info.end_date if info else None
    city = info.city if info else None
    state = info.state_or_province if info else None
    country = info.country if info else None
    venue = info.venue_name if info else None
    url = info.reference_url if info else None

    ref_node = evaluator.add_custom_node(
        result=_valid_url(url),
        id="West_Convention_Reference",
        desc="Provide a valid URL reference supporting the convention information",
        parent=node,
        critical=True
    )

    # Dates in September 2026
    dates_leaf = evaluator.add_leaf(
        id="West_Convention_Dates",
        desc="Provide the exact dates of the convention and verify it takes place during September 2026",
        parent=node,
        critical=True
    )
    claim_dates = (
        f"The event '{name or 'the event'}' takes place from {start or 'UNKNOWN'} to {end or 'UNKNOWN'} "
        f"and those dates fall in September 2026."
    )
    await evaluator.verify(
        claim=claim_dates,
        node=dates_leaf,
        sources=url,
        extra_prerequisites=[ref_node],
        additional_instruction="Confirm via the page that the dates are in September 2026."
    )

    # Location city in WA/OR/CA
    loc_leaf = evaluator.add_leaf(
        id="West_Convention_Location",
        desc="Provide the specific city and verify it is located in Washington, Oregon, or California",
        parent=node,
        critical=True
    )
    claim_loc = (
        f"The event '{name or 'the event'}' takes place in {city or 'UNKNOWN'}, {state or 'UNKNOWN'} "
        f"in the United States (must be Washington, Oregon, or California)."
    )
    await evaluator.verify(
        claim=claim_loc,
        node=loc_leaf,
        sources=url,
        extra_prerequisites=[ref_node],
        additional_instruction="Verify the state is WA, OR, or CA based on the event's official page or venue information."
    )

    # Venue name
    venue_leaf = evaluator.add_leaf(
        id="West_Convention_Venue",
        desc="Provide the specific venue or convention center name where the event takes place",
        parent=node,
        critical=True
    )
    claim_venue = (
        f"The event '{name or 'the event'}' is hosted at {venue or 'UNKNOWN'}."
    )
    await evaluator.verify(
        claim=claim_venue,
        node=venue_leaf,
        sources=url,
        extra_prerequisites=[ref_node],
        additional_instruction="Confirm the venue/convention center name from the page."
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
    Evaluate an answer for the 2026 gaming events travel schedule task.
    """
    evaluator = Evaluator()
    # IMPORTANT: Root must be non-critical in this framework to allow non-critical children.
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

    # Extract structured events data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=GamingEventsExtraction,
        extraction_name="events_extraction"
    )

    # Build verification subtrees for each category
    await verify_na_tabletop(evaluator, root, extracted.north_america_tabletop)
    await verify_europe_pc(evaluator, root, extracted.europe_pc_convention)
    await verify_europe_streaming(evaluator, root, extracted.europe_streaming_convention)
    await verify_esports_tournament(evaluator, root, extracted.major_esports_tournament)
    await verify_us_west_coast(evaluator, root, extracted.us_west_coast_convention)

    # Return structured summary
    return evaluator.get_summary()