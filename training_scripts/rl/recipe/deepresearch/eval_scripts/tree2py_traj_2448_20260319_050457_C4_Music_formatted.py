import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "outdoor_amphitheaters_7k_10k_4_venues_2026"
TASK_DESCRIPTION = (
    "I am helping plan a mid-sized summer concert tour for 2026 and need to identify suitable outdoor amphitheater "
    "venues across the United States. Find four outdoor amphitheaters that have a seating capacity between 7,000 "
    "and 10,000 (inclusive). The four venues must be located in at least three different U.S. states to ensure good "
    "geographic coverage for the tour. For each venue, provide the official venue name, the city and state where it "
    "is located, the exact seating capacity, and a direct link to either the venue's official website or its entry "
    "on Wikipedia's 'List of outdoor music venues in the United States' page."
)


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None
    url: Optional[str] = None


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to four venues as presented in the answer. For each venue, return:
    - name: the official venue name (string)
    - city: the city where it is located (string)
    - state: the U.S. state where it is located (string; can be full name or 2-letter postal abbreviation)
    - capacity: the exact seating capacity as written in the answer (string; keep digits/commas as presented)
    - url: a single direct link (URL) provided for that venue in the answer (string)
    
    Rules:
    - Do NOT invent or infer any missing data. If a field is not present in the answer, set it to null.
    - Only extract URLs explicitly present in the answer. If multiple URLs are provided for a venue, pick the first one that looks like an official venue website or a Wikipedia link.
    - Preserve the original formatting of names and capacities (e.g., keep commas in "7,500").
    - If the answer includes more than four venues, list the first four in order of appearance.
    
    Return a JSON object with a 'venues' array of objects having fields: name, city, state, capacity, url.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
STATE_ABBR_TO_NAME = {
    "AL": "ALABAMA", "AK": "ALASKA", "AZ": "ARIZONA", "AR": "ARKANSAS", "CA": "CALIFORNIA",
    "CO": "COLORADO", "CT": "CONNECTICUT", "DE": "DELAWARE", "FL": "FLORIDA", "GA": "GEORGIA",
    "HI": "HAWAII", "ID": "IDAHO", "IL": "ILLINOIS", "IN": "INDIANA", "IA": "IOWA",
    "KS": "KANSAS", "KY": "KENTUCKY", "LA": "LOUISIANA", "ME": "MAINE", "MD": "MARYLAND",
    "MA": "MASSACHUSETTS", "MI": "MICHIGAN", "MN": "MINNESOTA", "MS": "MISSISSIPPI", "MO": "MISSOURI",
    "MT": "MONTANA", "NE": "NEBRASKA", "NV": "NEVADA", "NH": "NEW HAMPSHIRE", "NJ": "NEW JERSEY",
    "NM": "NEW MEXICO", "NY": "NEW YORK", "NC": "NORTH CAROLINA", "ND": "NORTH DAKOTA", "OH": "OHIO",
    "OK": "OKLAHOMA", "OR": "OREGON", "PA": "PENNSYLVANIA", "RI": "RHODE ISLAND", "SC": "SOUTH CAROLINA",
    "SD": "SOUTH DAKOTA", "TN": "TENNESSEE", "TX": "TEXAS", "UT": "UTAH", "VT": "VERMONT",
    "VA": "VIRGINIA", "WA": "WASHINGTON", "WV": "WEST VIRGINIA", "WI": "WISCONSIN", "WY": "WYOMING",
    "DC": "DISTRICT OF COLUMBIA",
}
STATE_FULL_NAMES = set(STATE_ABBR_TO_NAME.values())
WIKIPEDIA_LIST_PAGE_PATH = "/wiki/List_of_outdoor_music_venues_in_the_United_States"
WIKIPEDIA_DOMAIN = "wikipedia.org"


def normalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip().upper().replace(".", "")
    # Normalize common DC variants
    if s in {"DC", "D C", "WASHINGTON DC", "WASHINGTON, DC", "WASHINGTON D C", "WASHINGTON, D C"}:
        return "DISTRICT OF COLUMBIA"
    if s in STATE_ABBR_TO_NAME:
        return STATE_ABBR_TO_NAME[s]
    if s in STATE_FULL_NAMES:
        return s
    # Try removing words like "STATE"
    if s.endswith(" STATE"):
        s2 = s[:-6].strip()
        if s2 in STATE_FULL_NAMES:
            return s2
    return s


def parse_capacity_to_int(cap_str: Optional[str]) -> Optional[int]:
    if not cap_str:
        return None
    digits = "".join(ch for ch in cap_str if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def is_valid_url_format(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    if not (url.lower().startswith("http://") or url.lower().startswith("https://")):
        return False
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme) and bool(parsed.netloc) and "." in parsed.netloc
    except Exception:
        return False


def is_wikipedia_list_page(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if WIKIPEDIA_DOMAIN in parsed.netloc:
            return parsed.path == WIKIPEDIA_LIST_PAGE_PATH
        return False
    except Exception:
        return False


def is_wikipedia_domain(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return WIKIPEDIA_DOMAIN in parsed.netloc
    except Exception:
        return False


def is_allowed_url_type(url: Optional[str]) -> bool:
    """
    Allowed if:
      - It's a non-Wikipedia domain (assumed to be the venue's official website or similar), OR
      - It's the specific Wikipedia 'List of outdoor music venues in the United States' page (any anchor allowed).
    Disallowed if it's a Wikipedia URL but not that list page (e.g., a venue's standalone Wikipedia article).
    """
    if not url:
        return False
    if is_wikipedia_list_page(url):
        return True
    if is_wikipedia_domain(url):
        return False
    return True


def ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(n, f"#{n}")


# --------------------------------------------------------------------------- #
# Verification logic per venue                                                #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    index: int,
) -> None:
    """
    Build verification subtree and run checks for a single venue.
    """
    ord_label = ordinal(index + 1)

    # Create main sequential node for this venue
    venue_node = evaluator.add_sequential(
        id=f"venue_{index+1}",
        desc=f"{ord_label} venue verification",
        parent=parent_node,
        critical=False
    )

    # 1) Required information presence (critical, gating)
    required_present = (
        (venue.name is not None and venue.name.strip() != "") and
        (venue.city is not None and venue.city.strip() != "") and
        (venue.state is not None and venue.state.strip() != "") and
        (venue.capacity is not None and venue.capacity.strip() != "") and
        is_valid_url_format(venue.url)
    )
    evaluator.add_custom_node(
        result=required_present,
        id=f"venue_{index+1}_required_info",
        desc=f"{ord_label} venue has required information (official name, city, state, exact capacity, and a valid URL)",
        parent=venue_node,
        critical=True
    )

    # 2) Constraint checks and source-grounded verifications (parallel under venue node)
    constraints_node = evaluator.add_parallel(
        id=f"venue_{index+1}_constraints",
        desc=f"{ord_label} venue meets constraints and facts are source-supported",
        parent=venue_node,
        critical=False
    )

    # 2.1) URL type allowed (critical)
    evaluator.add_custom_node(
        result=is_allowed_url_type(venue.url),
        id=f"venue_{index+1}_url_type_allowed",
        desc=f"{ord_label} venue URL is either the official site (non-Wikipedia) or the specific Wikipedia list page",
        parent=constraints_node,
        critical=True
    )

    # 2.2) Capacity within [7000, 10000] (critical)
    cap_int = parse_capacity_to_int(venue.capacity)
    capacity_in_range = (cap_int is not None) and (7000 <= cap_int <= 10000)
    evaluator.add_custom_node(
        result=capacity_in_range,
        id=f"venue_{index+1}_capacity_range",
        desc=f"{ord_label} venue capacity is between 7,000 and 10,000 (inclusive)",
        parent=constraints_node,
        critical=True
    )

    # 2.3) Name matches page (critical, source-grounded)
    name_leaf = evaluator.add_leaf(
        id=f"venue_{index+1}_name_supported",
        desc=f"{ord_label} venue name is supported by the provided URL",
        parent=constraints_node,
        critical=True
    )
    name_claim = f"The webpage clearly corresponds to a venue named '{venue.name}'. Minor variants (e.g., amphitheatre vs amphitheater, sponsor prefixes/suffixes) are acceptable if they refer to the same venue."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=venue.url,
        additional_instruction="Confirm the page is specifically about this venue and the name shown on the page reasonably matches the provided name."
    )

    # 2.4) Venue is an outdoor amphitheater (critical, source-grounded)
    amph_leaf = evaluator.add_leaf(
        id=f"venue_{index+1}_outdoor_amphitheater",
        desc=f"{ord_label} venue is an outdoor amphitheater",
        parent=constraints_node,
        critical=True
    )
    amph_claim = (
        "This venue is an outdoor amphitheater (open-air amphitheater or amphitheater-style outdoor music venue). "
        "Descriptions such as 'open-air', 'outdoor amphitheatre/amphitheater', 'outdoor pavilion with amphitheater-style seating' should count as outdoor amphitheater."
    )
    await evaluator.verify(
        claim=amph_claim,
        node=amph_leaf,
        sources=venue.url,
        additional_instruction="Focus on whether the venue is explicitly outdoor and amphitheater-like. If it's clearly an indoor arena or generic indoor theater, this should fail."
    )

    # 2.5) Location (city, state) supported (critical, source-grounded)
    loc_leaf = evaluator.add_leaf(
        id=f"venue_{index+1}_location_supported",
        desc=f"{ord_label} venue location (city, state) is supported by the provided URL",
        parent=constraints_node,
        critical=True
    )
    city = (venue.city or "").strip()
    state_text = (venue.state or "").strip()
    loc_claim = f"The venue '{venue.name}' is located in {city}, {state_text}."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=venue.url,
        additional_instruction="Allow common variants like 'St.' vs 'Saint' and state abbreviations vs full names as equivalent."
    )

    # 2.6) Capacity supported (critical, source-grounded)
    cap_leaf = evaluator.add_leaf(
        id=f"venue_{index+1}_capacity_supported",
        desc=f"{ord_label} venue seating capacity is supported by the provided URL",
        parent=constraints_node,
        critical=True
    )
    if cap_int is None:
        cap_claim = f"The seating capacity of the venue '{venue.name}' is {venue.capacity}."
    else:
        cap_claim = f"The seating capacity of the venue '{venue.name}' is {cap_int}."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=venue.url,
        additional_instruction="Treat it as supported if the page states the same integer value (ignoring commas). Minor formatting differences are OK; ranges or 'up to N' should not count as exact unless the exact number is clearly listed."
    )


# --------------------------------------------------------------------------- #
# Geographic diversity verification (critical to root)                        #
# --------------------------------------------------------------------------- #
def compute_geographic_diversity(venues: List[VenueItem]) -> Tuple[bool, List[str]]:
    normalized_states = []
    for v in venues:
        ns = normalize_state(v.state)
        if ns:
            normalized_states.append(ns)
    distinct = sorted(set(normalized_states))
    return (len(distinct) >= 3), distinct


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

    # Extract venues from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Keep exactly first 4 venues (pad with empty if fewer)
    venues = list(extracted.venues[:4])
    while len(venues) < 4:
        venues.append(VenueItem())

    # Record ground-truth constraints
    evaluator.add_ground_truth({
        "required_count": 4,
        "capacity_range_inclusive": [7000, 10000],
        "geographic_diversity_min_distinct_states": 3,
        "allowed_link_types": "Official venue website (non-Wikipedia) OR entry on Wikipedia's 'List of outdoor music venues in the United States' page"
    })

    # Per-venue verification
    for idx, v in enumerate(venues):
        await verify_single_venue(evaluator, root, v, idx)

    # Geographic diversity (critical to root)
    diversity_ok, distinct_states = compute_geographic_diversity(venues)
    evaluator.add_custom_node(
        result=diversity_ok,
        id="geographic_diversity",
        desc="The four identified venues are located in at least three different U.S. states",
        parent=root,
        critical=True
    )
    evaluator.add_custom_info(
        info={"distinct_states_normalized": distinct_states, "count": len(distinct_states)},
        info_type="diversity_summary",
        info_name="geographic_diversity_info"
    )

    # Return structured evaluation summary
    return evaluator.get_summary()