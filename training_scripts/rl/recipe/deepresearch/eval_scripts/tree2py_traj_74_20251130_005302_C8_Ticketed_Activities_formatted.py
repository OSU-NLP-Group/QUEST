import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "us_indoor_arenas_4"
TASK_DESCRIPTION = """I'm researching major indoor concert venues across the United States for a potential multi-city tour planning project. I need to identify four major indoor concert arenas, each located in a different U.S. state, with each venue having a concert seating capacity of at least 17,000 people.

For each of the four venues, please provide:
1. The venue name
2. The exact concert seating capacity
3. The city and state where the venue is located
4. The year the venue opened (or the year it reopened if there was a major renovation)
5. Information about on-site or adjacent parking facilities
6. Information about nearby public transportation options
7. At least one professional sports team that uses this venue as their home arena
8. Confirmation that accessible/wheelchair seating is available
9. The official website URL or a reliable reference URL (such as Wikipedia or a major ticketing platform page)

Please ensure that all four venues are indoor arenas (not outdoor amphitheaters or stadiums) and that each venue is located in a different U.S. state."""


# ----------------------------- Data Models ---------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string to allow flexible formats
    city: Optional[str] = None
    state: Optional[str] = None
    opening_year: Optional[str] = None
    parking_info: Optional[str] = None
    public_transit: Optional[str] = None
    home_teams: List[str] = Field(default_factory=list)  # At least one professional team
    accessible_seating: Optional[str] = None  # Confirmation or note
    reference_urls: List[str] = Field(default_factory=list)  # Official/Wiki/Ticketing URLs


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------- Extraction Prompt ------------------------------ #
def prompt_extract_venues() -> str:
    return (
        "Extract up to four venues (indoor concert arenas) mentioned in the answer. For each venue, return:\n"
        "1. name: The venue name (string)\n"
        "2. capacity: The exact concert seating capacity value as written in the answer (string; do not convert to number)\n"
        "3. city: The city where the venue is located (string)\n"
        "4. state: The U.S. state where the venue is located (string; can be full name or abbreviation)\n"
        "5. opening_year: The year it opened or reopened after a major renovation (string)\n"
        "6. parking_info: Information about on-site or adjacent parking facilities (string)\n"
        "7. public_transit: Information about nearby public transportation options (string)\n"
        "8. home_teams: An array of at least one professional sports team that uses the venue as home arena (array of strings)\n"
        "9. accessible_seating: Confirmation that accessible/wheelchair seating is available (string)\n"
        "10. reference_urls: Array of reference URLs (official venue website, Wikipedia, or major ticketing platform) (array of strings)\n\n"
        "GENERAL RULES:\n"
        "- If the answer mentions more than four venues, include only the first four in the final extraction.\n"
        "- If a field is missing for a venue, return null or an empty array as appropriate.\n"
        "- Extract only what is explicitly present in the answer; do not invent values.\n"
        "- Keep capacity as the exact string provided, even if it contains commas or units.\n"
    )


# ----------------------------- Helper Utils --------------------------------- #
def parse_capacity_to_int(cap_str: Optional[str]) -> Optional[int]:
    if not cap_str:
        return None
    # Accept formats like "17,500", "17500", "17k", "17,000-19,000" (take first number), "~18,000", "about 18,200"
    # Extract first integer-like number
    match = re.search(r"(\d[\d,\.]*)", cap_str)
    if not match:
        return None
    num_str = match.group(1)
    num_str = num_str.replace(",", "")
    try:
        # If something like "17.5k" appears, strip non-digits then int
        if re.search(r"[kK]", cap_str):
            # Approximate: 17.5k -> 17500
            try:
                val_float = float(re.sub(r"[^\d\.]", "", num_str))
                return int(round(val_float * 1000))
            except Exception:
                pass
        return int(float(num_str))
    except Exception:
        return None


def has_valid_urls(urls: List[str]) -> bool:
    return any(isinstance(u, str) and (u.strip().startswith("http://") or u.strip().startswith("https://")) for u in urls)


def normalize_str(s: Optional[str]) -> str:
    return (s or "").strip().lower()


# --------------------------- Venue Verification ----------------------------- #
async def verify_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    index: int,
) -> None:
    venue_node = evaluator.add_parallel(
        id=f"venue_{index + 1}",
        desc=f"Checks for venue #{index + 1}.",
        parent=parent_node,
        critical=False
    )

    # Critical existence checks
    name_exists = evaluator.add_custom_node(
        result=bool(venue.name and venue.name.strip()),
        id=f"venue_{index + 1}_name",
        desc="Venue name is provided.",
        parent=venue_node,
        critical=True
    )

    location_exists = evaluator.add_custom_node(
        result=bool(venue.city and venue.city.strip() and venue.state and venue.state.strip()),
        id=f"venue_{index + 1}_location",
        desc="City and U.S. state are provided.",
        parent=venue_node,
        critical=True
    )

    ref_urls_exist = evaluator.add_custom_node(
        result=has_valid_urls(venue.reference_urls),
        id=f"venue_{index + 1}_reference_url",
        desc="A reference URL is provided (official venue site, Wikipedia, or a major ticketing platform page).",
        parent=venue_node,
        critical=True
    )

    # Capacity threshold (>= 17,000)
    cap_int = parse_capacity_to_int(venue.capacity)
    capacity_threshold = evaluator.add_custom_node(
        result=(cap_int is not None and cap_int >= 17000),
        id=f"venue_{index + 1}_capacity_threshold",
        desc="Capacity is at least 17,000.",
        parent=venue_node,
        critical=True
    )

    # Prepare leaf nodes for evidence-based checks
    claims_and_nodes: List[tuple[str, List[str], Any, str]] = []

    # Indoor arena primary use
    indoor_leaf = evaluator.add_leaf(
        id=f"venue_{index + 1}_indoor_arena_primary_use",
        desc="Venue is an indoor arena (not an outdoor amphitheater/stadium) and is used for concerts/sports/entertainment events.",
        parent=venue_node,
        critical=True
    )
    indoor_claim = (
        f"The venue '{venue.name or ''}' is an indoor arena used for concerts, sports, or entertainment events."
    )
    indoor_ins = (
        "Confirm the page indicates it is an indoor arena (multipurpose or basketball/hockey arena). "
        "Do not accept outdoor amphitheaters or stadiums. Minor wording variations are acceptable."
    )
    claims_and_nodes.append((indoor_claim, venue.reference_urls, indoor_leaf, indoor_ins))

    # Capacity value supported
    capacity_leaf = evaluator.add_leaf(
        id=f"venue_{index + 1}_capacity_supported",
        desc="The exact concert seating capacity value is supported by the reference URLs.",
        parent=venue_node,
        critical=True
    )
    cap_claim = (
        f"The concert seating capacity of '{venue.name or ''}' is {venue.capacity or ''}."
    )
    cap_ins = (
        "Verify the capacity figure on the referenced pages. If multiple capacities are listed "
        "(basketball/hockey/concert), prefer concert capacity. Allow minor rounding differences."
    )
    claims_and_nodes.append((cap_claim, venue.reference_urls, capacity_leaf, cap_ins))

    # Opening year supported
    opening_leaf = evaluator.add_leaf(
        id=f"venue_{index + 1}_opening_year",
        desc="Year opened (or year reopened after major renovation) is provided and supported.",
        parent=venue_node,
        critical=True
    )
    opening_claim = (
        f"The venue '{venue.name or ''}' opened (or reopened after major renovation) in {venue.opening_year or ''}."
    )
    opening_ins = (
        "Confirm the opening year (or a clearly indicated reopening year after major renovation). "
        "Accept synonyms like 'first opened' or 'reopened'."
    )
    claims_and_nodes.append((opening_claim, venue.reference_urls, opening_leaf, opening_ins))

    # Parking info supported
    parking_leaf = evaluator.add_leaf(
        id=f"venue_{index + 1}_parking",
        desc="Information about on-site or adjacent parking facilities is provided and supported.",
        parent=venue_node,
        critical=True
    )
    parking_claim = (
        f"The venue '{venue.name or ''}' has on-site or adjacent parking facilities (garages or lots)."
    )
    parking_ins = (
        "Look for terms like 'parking', 'garage', 'lot', or 'on-site/adjacent'. "
        "General guidance pages or visitor info pages are acceptable."
    )
    claims_and_nodes.append((parking_claim, venue.reference_urls, parking_leaf, parking_ins))

    # Public transit info supported
    transit_leaf = evaluator.add_leaf(
        id=f"venue_{index + 1}_public_transit",
        desc="Information about nearby public transportation options is provided and supported.",
        parent=venue_node,
        critical=True
    )
    transit_claim = (
        f"The venue '{venue.name or ''}' is served by nearby public transportation (e.g., bus, rail, subway/light rail)."
    )
    transit_ins = (
        "Look for mentions of bus routes, rail/subway stations, or official transit guidance. "
        "Nearby stations within walking distance count."
    )
    claims_and_nodes.append((transit_claim, venue.reference_urls, transit_leaf, transit_ins))

    # Home team existence (critical) and supported
    home_team_exists = evaluator.add_custom_node(
        result=bool(venue.home_teams),
        id=f"venue_{index + 1}_home_team_exists",
        desc="At least one professional sports team that uses the venue as its home arena is identified.",
        parent=venue_node,
        critical=True
    )
    first_team = venue.home_teams[0] if venue.home_teams else ""
    home_team_leaf = evaluator.add_leaf(
        id=f"venue_{index + 1}_home_team_supported",
        desc="Home team usage is supported by the reference URLs.",
        parent=venue_node,
        critical=True
    )
    home_claim = (
        f"The professional sports team '{first_team}' uses '{venue.name or ''}' as its home arena."
    )
    home_ins = (
        "Accept phrasing like 'home arena', 'plays home games at', or 'hosts the team'. "
        "Ensure the team is professional (NBA, NHL, WNBA, etc.)."
    )
    claims_and_nodes.append((home_claim, venue.reference_urls, home_team_leaf, home_ins))

    # Accessible seating supported
    accessible_leaf = evaluator.add_leaf(
        id=f"venue_{index + 1}_accessible_seating",
        desc="Accessible/wheelchair seating availability is confirmed and supported.",
        parent=venue_node,
        critical=True
    )
    accessible_claim = (
        f"Accessible or wheelchair seating is available at '{venue.name or ''}'."
    )
    accessible_ins = (
        "Look for ADA, accessibility, wheelchair seating, or accessible services pages. "
        "General accessibility statements on official or ticketing pages are acceptable."
    )
    claims_and_nodes.append((accessible_claim, venue.reference_urls, accessible_leaf, accessible_ins))

    # Batch verify all evidence-based leaves (auto preconditions will gate on critical siblings like name/location/reference_url)
    await evaluator.batch_verify(claims_and_nodes)


# ----------------------------- Main Evaluation ------------------------------ #
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Select exactly the first 4 venues for evaluation (padding with empty entries if fewer)
    selected: List[VenueItem] = list(extracted.venues[:4])
    while len(selected) < 4:
        selected.append(VenueItem())

    # Top-level critical checks (placed directly under root)
    venue_count_node = evaluator.add_custom_node(
        result=(len(selected) == 4),
        id="venue_count",
        desc="Exactly four venues are provided.",
        parent=root,
        critical=True
    )

    # Different states check (critical)
    states = [normalize_str(v.state) for v in selected]
    valid_states = all(bool(s) for s in states)
    different_states = len(set(states)) == 4 if valid_states else False
    different_states_node = evaluator.add_custom_node(
        result=different_states,
        id="different_states",
        desc="All four venues are located in different U.S. states.",
        parent=root,
        critical=True
    )

    # Build venue subtrees (non-critical parents, critical leaves inside)
    for i, venue in enumerate(selected):
        venue_parent = evaluator.add_parallel(
            id=f"venue_{i + 1}_main",
            desc=f"Checks for venue #{i + 1}.",
            parent=root,
            critical=False
        )
        await verify_venue(evaluator, venue_parent, venue, i)

    return evaluator.get_summary()