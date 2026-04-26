import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.evaluator import Evaluator

# ------------------------------------------------------------------------------------
# Task-specific constants
# ------------------------------------------------------------------------------------
TASK_ID = "coldplay_mots_2025_us_stadiums"
TASK_DESCRIPTION = (
    "Identify 5 stadium venues where Coldplay is scheduled to perform during their Music of the Spheres World Tour "
    "in the United States between May 1 and August 31, 2025, that meet the following criteria: each stadium must have "
    "a minimum seating capacity of 50,000, the venues must represent at least 4 different U.S. states, and no single "
    "state can have more than 2 of these venues. For each venue, provide the venue name, city, state, concert date, "
    "and a reference URL that confirms both the venue details and the scheduled concert date."
)

DATE_WINDOW_START = "2025-05-01"
DATE_WINDOW_END = "2025-08-31"

# ------------------------------------------------------------------------------------
# Utility: US state normalization (full name -> USPS 2-letter)
# ------------------------------------------------------------------------------------
_US_STATE_ABBR = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
    "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT",
    "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI", "WYOMING": "WY", "DISTRICT OF COLUMBIA": "DC", "WASHINGTON, DC": "DC",
    "D.C.": "DC", "DC": "DC"
}
_US_ABBR_SET = set(_US_STATE_ABBR.values())

def normalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip().upper()
    if not s:
        return None
    if s in _US_ABBR_SET:
        return s
    return _US_STATE_ABBR.get(s, s)


def dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

# ------------------------------------------------------------------------------------
# Extraction models
# ------------------------------------------------------------------------------------
class VenueItem(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    concert_date: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    capacity_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# ------------------------------------------------------------------------------------
# Extraction prompt
# ------------------------------------------------------------------------------------
def prompt_extract_venues() -> str:
    return (
        "Extract up to 5 stadium venues from the answer that match the user's request. For each venue, extract the "
        "following fields exactly as presented in the answer:\n"
        "- venue_name: The name of the stadium.\n"
        "- city: The city where the stadium is located.\n"
        "- state: The U.S. state where the stadium is located (extract as shown; do not invent). If both a state name "
        "  and an abbreviation appear, prefer the abbreviation.\n"
        "- concert_date: The scheduled concert date for Coldplay at that venue (extract as-is from the answer; do not reformat).\n"
        "- reference_urls: A list (array) of one or more URLs explicitly provided in the answer that confirm both the "
        "  venue details and the scheduled concert date. Include all relevant URLs the answer cites for this venue. "
        "  If the answer only mentions a website without a concrete URL, do not include it.\n"
        "- capacity_urls: A list (array) of any URLs in the answer that specifically support the stadium's seating capacity. "
        "  If none are provided separately, leave this as an empty list.\n\n"
        "Return a JSON object with a single field 'venues' that is a list of such venue objects (maximum 5). "
        "If fewer than 5 venues are present in the answer, return only those found."
    )

# ------------------------------------------------------------------------------------
# Venue verification
# ------------------------------------------------------------------------------------
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    index: int
) -> None:
    # Parent node for this venue (parallel aggregation, non-critical to allow partial credit per venue)
    venue_node = evaluator.add_parallel(
        id=f"venue_{index+1}",
        desc=f"{['First','Second','Third','Fourth','Fifth'][index]} qualifying stadium venue",
        parent=parent_node,
        critical=False
    )

    name_ok = bool(venue.venue_name and venue.venue_name.strip())
    location_ok = bool(venue.city and venue.city.strip()) and bool(venue.state and venue.state.strip())
    has_reference = bool(venue.reference_urls and len(venue.reference_urls) > 0)

    evaluator.add_custom_node(
        result=name_ok,
        id=f"v{index+1}_name",
        desc="Venue name is provided",
        parent=venue_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=location_ok,
        id=f"v{index+1}_location",
        desc="Venue location (city and state) is provided",
        parent=venue_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_reference,
        id=f"v{index+1}_reference",
        desc="Reference URL provided supporting the venue and date information",
        parent=venue_node,
        critical=True
    )

    # Prepare sources
    base_sources = dedup_preserve_order(venue.reference_urls or [])
    capacity_sources = dedup_preserve_order((venue.capacity_urls or []) + base_sources)

    # Capacity check
    capacity_node = evaluator.add_leaf(
        id=f"v{index+1}_capacity",
        desc="Venue has minimum seating capacity of 50,000 for concerts",
        parent=venue_node,
        critical=True
    )
    cap_claim = (
        f"The stadium '{venue.venue_name or ''}' has a seating capacity of at least 50,000 "
        f"for concerts or common large-event configurations (e.g., football)."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=capacity_node,
        sources=capacity_sources,
        additional_instruction=(
            "Verify from the provided page(s) whether the listed stadium capacity is >= 50,000. "
            "Accept if any reasonable official or reputable configuration (football/soccer/concert) "
            "shows capacity at or above 50,000. If multiple capacities are listed, passing any value >= 50,000 is acceptable. "
            "Do not treat historical attendance as capacity unless explicitly labeled as capacity."
        )
    )

    # Date check
    date_node = evaluator.add_leaf(
        id=f"v{index+1}_date",
        desc="Concert date falls between May 1-August 31, 2025",
        parent=venue_node,
        critical=True
    )
    date_claim = (
        f"The provided source confirms that Coldplay is scheduled to perform at "
        f"{venue.venue_name or ''} in {venue.city or ''}, {venue.state or ''} on {venue.concert_date or ''}, "
        f"and that date falls between May 1 and August 31, 2025 inclusive."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=base_sources,
        additional_instruction=(
            "Confirm the page lists a Coldplay concert at the specified venue on the specified date, "
            "and check that the date is within 2025-05-01 to 2025-08-31 inclusive. "
            "Accept reasonable date format variations and multiple-date listings as long as the specified date is included. "
            "If the page shows a different year or a date outside the window, mark incorrect."
        )
    )

    # Tour check
    tour_node = evaluator.add_leaf(
        id=f"v{index+1}_tour",
        desc="Concert is part of Coldplay's Music of the Spheres World Tour",
        parent=venue_node,
        critical=True
    )
    tour_claim = (
        "The source explicitly indicates that this Coldplay concert is part of the 'Music of the Spheres' World Tour."
    )
    await evaluator.verify(
        claim=tour_claim,
        node=tour_node,
        sources=base_sources,
        additional_instruction=(
            "Look for phrasing such as 'Music of the Spheres', 'Music Of The Spheres Tour', or 'MOTS World Tour'. "
            "Minor capitalization or wording differences are acceptable. "
            "If the page only mentions Coldplay without any tour name, mark as not supported."
        )
    )


# ------------------------------------------------------------------------------------
# Main evaluation entry point
# ------------------------------------------------------------------------------------
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
        strategy=AggregationStrategy.PARALLEL,  # Root parallel aggregation
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

    # 1) Extract up to 5 venues from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    venues: List[VenueItem] = list(extraction.venues or [])
    if len(venues) > 5:
        venues = venues[:5]
    while len(venues) < 5:
        venues.append(VenueItem())

    # 2) Build verification subtrees for each of the 5 venues
    for i in range(5):
        await verify_single_venue(evaluator, root, venues[i], i)

    # 3) Group-level constraints (critical)
    # Normalize states for diversity checks
    norm_states = []
    for v in venues:
        ns = normalize_state(v.state)
        if ns:
            # Convert arbitrary tokens to uppercase two-letter if possible (already normalized)
            norm_states.append(ns)

    # at least 4 different states among the 5 venues
    state_diverse = len(set(norm_states)) >= 4
    evaluator.add_custom_node(
        result=state_diverse,
        id="state_diversity",
        desc="The 5 venues represent at least 4 different U.S. states",
        parent=root,
        critical=True
    )

    # no single state has more than 2 of the 5 venues
    from collections import Counter
    counts = Counter(norm_states)
    no_state_over_2 = True
    if counts:
        no_state_over_2 = max(counts.values()) <= 2

    evaluator.add_custom_node(
        result=no_state_over_2,
        id="no_state_exceeds_limit",
        desc="No single state has more than 2 of the 5 venues",
        parent=root,
        critical=True
    )

    # 4) Return structured result
    return evaluator.get_summary()