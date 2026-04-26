import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "ariana_summer_arenas_2026"
TASK_DESCRIPTION = (
    "Ariana Grande is embarking on The Eternal Sunshine Tour in 2026. Identify 4 distinct major indoor arena venues "
    "in the United States where she is scheduled to perform during the months of June, July, or August 2026. For each "
    "venue, provide: (1) The venue name, (2) The city where the venue is located, (3) The state where the venue is "
    "located, (4) At least one specific concert date (in YYYY-MM-DD format) when she will perform at that venue, "
    "and (5) A reference URL that confirms this venue and date information. Ensure that all 4 venues are different "
    "locations (not multiple dates at the same venue), and that each is an indoor arena suitable for major touring "
    "artists."
)


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
US_STATE_NAMES = {
    # Full names
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware",
    "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky",
    "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey", "new mexico",
    "new york", "north carolina", "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
    "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming", "district of columbia",
    # Common territories sometimes used in event contexts (rare)
    "puerto rico"
}
US_STATE_ABBR = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME",
    "MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA",
    "RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC","PR"
}


def is_valid_us_state(state: Optional[str]) -> bool:
    if not state or not isinstance(state, str):
        return False
    s = state.strip()
    if not s:
        return False
    upper = s.upper()
    lower = s.lower()
    return (upper in US_STATE_ABBR) or (lower in US_STATE_NAMES)


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def is_yyyy_mm_dd(date_str: Optional[str]) -> bool:
    if not date_str or not isinstance(date_str, str):
        return False
    if not DATE_RE.match(date_str.strip()):
        return False
    try:
        datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return True
    except Exception:
        return False


def is_summer_2026(date_str: Optional[str]) -> bool:
    if not is_yyyy_mm_dd(date_str):
        return False
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return dt.year == 2026 and dt.month in (6, 7, 8)
    except Exception:
        return False


def normalize_key(*parts: Optional[str]) -> str:
    def norm(s: Optional[str]) -> str:
        if not s:
            return ""
        t = re.sub(r"[^a-z0-9]+", "", s.lower())
        return t
    return "|".join(norm(p) for p in parts)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    dates: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)
    supporting_urls: List[str] = Field(default_factory=list)  # venue info links if provided
    artist_name: Optional[str] = None
    tour_name: Optional[str] = None


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract all venue entries for Ariana Grande's 2026 tour that are mentioned in the answer. Focus on indoor arena
    venues in the United States that are scheduled during June, July, or August 2026. For each venue mentioned,
    extract the following fields exactly as presented in the answer:

    - venue_name: The name of the venue.
    - city: The city where the venue is located.
    - state: The US state (either full name or 2-letter abbreviation).
    - dates: A list of concert dates at this venue, as given in the answer. Preserve the format if already in YYYY-MM-DD.
             If multiple dates are present for the same venue, include them all. If none are present, use an empty list.
    - reference_urls: All URLs cited that directly confirm this venue's show(s) and date(s) (e.g., venue event pages,
                      artist website, reputable listings). Extract only URLs explicitly present in the answer.
    - supporting_urls: Any additional URLs provided in the answer that describe the venue itself (e.g., the venue’s
                       official site, Wikipedia) which might include venue type or capacity details.
    - artist_name: The artist name as written in the answer for this venue’s entry (if provided).
    - tour_name: The tour name as written in the answer for this venue’s entry (if provided).

    Return a JSON object with:
    {
      "venues": [ ... up to all venues found in the answer ... ]
    }

    Rules:
    - Do not fabricate any URLs or dates; only extract those that appear explicitly in the answer.
    - If a field is missing, set it to null (for strings) or [] (for lists).
    - Keep venues in the order they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def pick_first_valid_date(dates: List[str]) -> Optional[str]:
    for d in dates:
        if is_yyyy_mm_dd(d):
            return d
    return None


def combined_urls(item: VenueItem) -> List[str]:
    # Combine and de-duplicate while preserving order
    seen = set()
    urls: List[str] = []
    for u in (item.reference_urls or []) + (item.supporting_urls or []):
        if isinstance(u, str):
            uu = u.strip()
            if uu and uu not in seen:
                seen.add(uu)
                urls.append(uu)
    return urls


# --------------------------------------------------------------------------- #
# Venue verification logic                                                    #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    item: VenueItem,
    index: int,
) -> None:
    """
    Build verification sub-tree for a single venue.
    """
    v_id = index + 1
    venue_node = evaluator.add_parallel(
        id=f"venue_{v_id}",
        desc=f"Venue #{v_id} meeting all requirements",
        parent=parent_node,
        critical=False  # venue groups are non-critical under root
    )

    # Provided/existence checks (custom boolean leaves)
    name_ok = bool(item.venue_name and item.venue_name.strip())
    city_ok = bool(item.city and item.city.strip())
    state_ok = is_valid_us_state(item.state)
    urls_ok = len(item.reference_urls) > 0

    # Dates: at least one date provided in correct YYYY-MM-DD format
    any_valid_date = any(is_yyyy_mm_dd(d) for d in (item.dates or []))

    # Timeframe: at least one date in Jun/Jul/Aug 2026
    any_summer_2026 = any(is_summer_2026(d) for d in (item.dates or []))

    evaluator.add_custom_node(
        result=name_ok,
        id=f"venue_{v_id}_name",
        desc="Venue name is provided",
        parent=venue_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=city_ok,
        id=f"venue_{v_id}_city",
        desc="City where the venue is located is provided",
        parent=venue_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=state_ok,
        id=f"venue_{v_id}_state",
        desc="State where the venue is located is provided (confirming US location)",
        parent=venue_node,
        critical=True
    )
    # Reference URL existence (critical; also used as precondition for URL-based verifications)
    ref_node = evaluator.add_custom_node(
        result=urls_ok,
        id=f"venue_{v_id}_reference",
        desc="Reference URL confirming this venue, artist, and date(s) is provided",
        parent=venue_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=any_valid_date,
        id=f"venue_{v_id}_dates",
        desc="At least one concert date is provided in YYYY-MM-DD format",
        parent=venue_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=any_summer_2026,
        id=f"venue_{v_id}_timeframe",
        desc="The provided date(s) fall within June, July, or August 2026",
        parent=venue_node,
        critical=True
    )

    # URL-based factual verifications
    urls = combined_urls(item)

    # 1) Artist check (Ariana Grande)
    artist_leaf = evaluator.add_leaf(
        id=f"venue_{v_id}_artist",
        desc="The performance is by Ariana Grande",
        parent=venue_node,
        critical=True
    )

    artist_claim_parts = []
    artist_claim_parts.append("The referenced page confirms an event for Ariana Grande")
    if item.venue_name:
        artist_claim_parts.append(f"at the venue '{item.venue_name}'")
    if item.city and item.state:
        artist_claim_parts.append(f"in {item.city}, {item.state}")
    artist_claim = " ".join(artist_claim_parts) + "."

    await evaluator.verify(
        claim=artist_claim,
        node=artist_leaf,
        sources=urls,
        additional_instruction=(
            "Verify that the page clearly indicates Ariana Grande as the performing artist/headliner. "
            "Minor name variations (e.g., capitalization or middle/last name additions) are acceptable."
        ),
        extra_prerequisites=[ref_node]
    )

    # 2) Tour check (Eternal Sunshine Tour)
    tour_leaf = evaluator.add_leaf(
        id=f"venue_{v_id}_tour",
        desc="The performance is part of The Eternal Sunshine Tour",
        parent=venue_node,
        critical=True
    )
    tour_claim = (
        "The referenced page indicates that the Ariana Grande performance is part of "
        "the 'Eternal Sunshine Tour' (case-insensitive; synonyms like 'Eternal Sunshine World Tour' are acceptable)."
    )
    await evaluator.verify(
        claim=tour_claim,
        node=tour_leaf,
        sources=urls,
        additional_instruction=(
            "Accept reasonable variants such as 'Eternal Sunshine Tour', 'the eternal sunshine tour', "
            "'Eternal Sunshine World Tour', or abbreviations if clearly referring to the same tour."
        ),
        extra_prerequisites=[ref_node]
    )

    # 3) Venue type: indoor arena
    type_leaf = evaluator.add_leaf(
        id=f"venue_{v_id}_type",
        desc="The venue is identified as an indoor arena",
        parent=venue_node,
        critical=True
    )
    type_claim = (
        f"The venue{f' {item.venue_name}' if item.venue_name else ''} is an indoor arena (i.e., an enclosed arena). "
        "Phrases like 'indoor arena', 'multi-purpose indoor arena', or 'enclosed arena' qualify."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=urls,
        additional_instruction=(
            "Look for explicit mentions such as 'indoor arena', 'indoor', 'enclosed arena', or similar. "
            "If the page states it is an 'arena' and makes clear it's indoors (e.g., typical NBA/NHL arena), accept it. "
            "Do not accept 'stadium' or open-air amphitheater as indoor arenas."
        ),
        extra_prerequisites=[ref_node]
    )

    # 4) Capacity: suitable for major touring artists (15,000+)
    capacity_leaf = evaluator.add_leaf(
        id=f"venue_{v_id}_capacity",
        desc="The venue is suitable for major touring artists (typically 15,000+ capacity)",
        parent=venue_node,
        critical=False
    )
    capacity_claim = (
        f"The venue{f' {item.venue_name}' if item.venue_name else ''} has a listed capacity of at least 15,000 "
        "for concerts or typical arena events (basketball/hockey capacities are acceptable proxies if concert capacity "
        "is not specified)."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=urls,
        additional_instruction=(
            "Use any explicit capacity listed on the venue or Wikipedia-type page. "
            "If multiple capacities are given (e.g., for basketball/hockey), accept if any are ≥ 15,000. "
            "Approximate or range values are acceptable if the lower bound is ≥ 15,000."
        ),
        extra_prerequisites=[ref_node]
    )


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
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point for evaluating an answer for the Ariana Grande summer 2026 indoor arenas task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Root combines sub-criteria independently
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

    # IMPORTANT: Make root non-critical to allow non-critical venue children (framework requires this).
    root.critical = False

    # Extract venues from the answer
    extracted: VenuesExtraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Ensure exactly 4 venues for evaluation (pad with empty ones if needed)
    venues: List[VenueItem] = list(extracted.venues or [])
    if len(venues) < 4:
        for _ in range(4 - len(venues)):
            venues.append(VenueItem())
    if len(venues) > 4:
        venues = venues[:4]

    # Distinctness check across the 4 venues (critical at root)
    def distinct_ok(vs: List[VenueItem]) -> bool:
        keys = []
        for it in vs:
            if not (it.venue_name and it.city and it.state):
                return False
            k = normalize_key(it.venue_name, it.city, it.state)
            keys.append(k)
        return len(set(keys)) == len(keys) == 4

    evaluator.add_custom_node(
        result=distinct_ok(venues),
        id="venue_distinctness",
        desc="All 4 venues are different geographic locations (not the same venue listed multiple times)",
        parent=root,
        critical=True
    )

    # Build verification subtrees for each venue
    tasks = []
    for i, item in enumerate(venues[:4]):
        tasks.append(verify_single_venue(evaluator, root, item, i))
    await asyncio.gather(*tasks)

    return evaluator.get_summary()