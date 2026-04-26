import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "se_outdoor_amphitheaters"
TASK_DESCRIPTION = """
Find three outdoor amphitheater concert venues in the southeastern United States (specifically in Georgia, Florida, Tennessee, Alabama, or South Carolina) that each have a total seating capacity between 4,000 and 15,000 people and are currently operational for concerts.

For each venue, provide the following information:
1. Venue name, city, and state
2. Complete physical street address
3. Total seating capacity
4. Description of seating configuration types (such as reserved seating, lawn seating, etc.)
5. Information about ADA-compliant wheelchair-accessible seating options
6. At least one type of premium seating offering (such as VIP boxes, club seats, or luxury suites)
7. Information about on-site or designated parking facilities
8. Information about food and beverage concessions or services available at the venue
9. Official venue website URL or a verified source URL where this information can be confirmed

Each of the three venues must be distinct locations with different names and addresses.
"""

TARGET_YEARS = [2025, 2026]

ALLOWED_STATES = {
    "GA": "Georgia",
    "FL": "Florida",
    "TN": "Tennessee",
    "AL": "Alabama",
    "SC": "South Carolina",
}
ALLOWED_STATE_NAMES_UPPER = {v.upper(): k for k, v in ALLOWED_STATES.items()}


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Accept either full name (e.g., Georgia) or code (GA)
    address: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string to allow ranges/approx text
    seating_types: Optional[str] = None
    accessibility_info: Optional[str] = None
    premium_seating: Optional[str] = None
    parking_info: Optional[str] = None
    concessions_info: Optional[str] = None
    venue_type_desc: Optional[str] = None  # e.g., outdoor amphitheater, open-air, etc.
    operational_info: Optional[str] = None  # e.g., "2025 concert calendar" mentioned
    official_url: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_venues() -> str:
    return """
    Extract up to the first 5 venues (if more are present) from the answer that claim to satisfy the task. Return them in an array called 'venues'. For each venue, extract the following fields exactly as they appear in the answer:

    - name: venue name
    - city: city name
    - state: U.S. state (can be full name or 2-letter code)
    - address: complete street address (street number/name, city, state, ZIP if provided)
    - capacity: total seating capacity (as stated; keep as text if it's a range or approximate)
    - seating_types: description of seating configuration types (e.g., reserved seats, lawn seating, general admission lawn, etc.)
    - accessibility_info: any mention of ADA-compliant or wheelchair-accessible seating options
    - premium_seating: at least one type of premium seating offering (e.g., VIP boxes, club seats, suites) mentioned in the answer
    - parking_info: any mention of on-site or designated parking facilities
    - concessions_info: any mention of food and beverage concessions or services
    - venue_type_desc: any description indicating this is an outdoor amphitheater or open-air concert venue
    - operational_info: any text indicating the venue is currently operational and has hosted or scheduled concerts in 2025 or 2026
    - official_url: the official venue website URL if explicitly provided
    - source_urls: an array of any other URLs explicitly provided in the answer that contain information about the venue

    SPECIAL RULES:
    - Only include URLs explicitly present in the answer text. If missing protocol, prepend http://
    - Do not invent data. If any field is not present in the answer, set it to null (or an empty array for source_urls).
    """


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _is_nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _normalize_state_code_or_name(s: Optional[str]) -> Optional[str]:
    if not _is_nonempty(s):
        return None
    s2 = s.strip().replace(".", "")
    # Try code
    code = s2.upper()
    if code in ALLOWED_STATES:
        return code
    # Try full name
    upper_name = s2.upper()
    if upper_name in ALLOWED_STATE_NAMES_UPPER:
        return ALLOWED_STATE_NAMES_UPPER[upper_name]
    # Try title-case compare (robust to casing)
    title_name_upper = s2.title().upper()
    if title_name_upper in ALLOWED_STATE_NAMES_UPPER:
        return ALLOWED_STATE_NAMES_UPPER[title_name_upper]
    # Not in allowed
    return None


def _full_state_name_from_any(s: Optional[str]) -> Optional[str]:
    code = _normalize_state_code_or_name(s)
    if not code:
        return None
    return ALLOWED_STATES[code]


def _combine_urls(v: VenueItem) -> List[str]:
    urls = []
    if _is_nonempty(v.official_url):
        urls.append(v.official_url.strip())
    if v.source_urls:
        urls.extend([u.strip() for u in v.source_urls if _is_nonempty(u)])
    # Deduplicate preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _ordinal(idx0: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][idx0] if idx0 < 5 else f"#{idx0+1}"


# -----------------------------------------------------------------------------
# Verification for a single venue
# -----------------------------------------------------------------------------
async def verify_single_venue(evaluator: Evaluator, parent_node, venue: VenueItem, idx0: int) -> None:
    ord_word = _ordinal(idx0)
    node = evaluator.add_parallel(
        id=f"venue_{idx0+1}",
        desc=f"{ord_word} qualifying outdoor amphitheater venue with all required information",
        parent=parent_node,
        critical=False,  # allow partial credit per venue
    )

    # Prepare shared values
    name = venue.name or "the venue"
    city = venue.city or ""
    state_full = _full_state_name_from_any(venue.state) or (venue.state or "")
    address = venue.address or ""
    urls = _combine_urls(venue)

    # 1) Basic info existence (name, city, state)
    evaluator.add_custom_node(
        result=(_is_nonempty(venue.name) and _is_nonempty(venue.city) and _is_nonempty(venue.state)),
        id=f"venue_{idx0+1}_basic_info",
        desc="Venue name, city, and state are provided",
        parent=node,
        critical=True,
    )

    # 2) Source/reference presence (must have at least one URL)
    evaluator.add_custom_node(
        result=(len(urls) > 0),
        id=f"venue_{idx0+1}_reference",
        desc="Official venue website or verified source URL is provided",
        parent=node,
        critical=True,
    )

    # 3) Location constraint (verify by URL that location matches city/state)
    loc_node = evaluator.add_leaf(
        id=f"venue_{idx0+1}_location_constraint",
        desc="Venue is located in one of the specified southeastern states: Georgia, Florida, Tennessee, Alabama, or South Carolina",
        parent=node,
        critical=True,
    )
    loc_claim = f"The venue named '{name}' is located in {city}, {state_full}."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=urls,
        additional_instruction="Verify that the page clearly indicates the venue's city and state. Minor formatting or abbreviation differences (e.g., 'St.' vs 'Street') are acceptable.",
    )

    # 3.1) Allowed-state check (custom logical check using extracted state)
    evaluator.add_custom_node(
        result=(_normalize_state_code_or_name(venue.state) in ALLOWED_STATES),
        id=f"venue_{idx0+1}_state_allowed",
        desc="State is one of GA, FL, TN, AL, or SC (allowed states)",
        parent=node,
        critical=True,
    )

    # 4) Venue type: outdoor amphitheater primarily for live music
    type_node = evaluator.add_leaf(
        id=f"venue_{idx0+1}_type_constraint",
        desc="Venue is an outdoor amphitheater designed primarily for live music concerts",
        parent=node,
        critical=True,
    )
    type_claim = f"'{name}' is an outdoor amphitheater (open-air venue) primarily used for live music concerts."
    await evaluator.verify(
        claim=type_claim,
        node=type_node,
        sources=urls,
        additional_instruction="Treat 'amphitheater' and 'amphitheatre' as equivalent. Look for language indicating open-air/outdoor and that it hosts concerts.",
    )

    # 5) Operational status in 2025 or 2026
    op_node = evaluator.add_leaf(
        id=f"venue_{idx0+1}_operational",
        desc="Venue is currently operational and hosting concerts in 2025 or 2026",
        parent=node,
        critical=True,
    )
    op_claim = f"'{name}' is currently operational and has hosted or has scheduled concerts in {TARGET_YEARS[0]} or {TARGET_YEARS[1]}."
    await evaluator.verify(
        claim=op_claim,
        node=op_node,
        sources=urls,
        additional_instruction=f"Check the events, schedule, or calendar pages for {TARGET_YEARS[0]} or {TARGET_YEARS[1]}. Listings from ticketing partners are acceptable.",
    )

    # 6) Address: provided + verified
    evaluator.add_custom_node(
        result=_is_nonempty(venue.address),
        id=f"venue_{idx0+1}_address_provided",
        desc="Complete physical address string is present in the answer",
        parent=node,
        critical=True,
    )
    addr_node = evaluator.add_leaf(
        id=f"venue_{idx0+1}_address",
        desc="Complete physical address is provided",
        parent=node,
        critical=True,
    )
    addr_claim = f"The official or authoritative page lists the venue's address as '{address}'."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_node,
        sources=urls,
        additional_instruction="Allow minor formatting or abbreviation differences (e.g., Ave vs Avenue, Rd vs Road, punctuation). City, state, and street should match in meaning.",
    )

    # 7) Capacity: provided + within 4,000–15,000 supported by sources
    evaluator.add_custom_node(
        result=_is_nonempty(venue.capacity),
        id=f"venue_{idx0+1}_capacity_provided",
        desc="A total seating capacity value/text is provided in the answer",
        parent=node,
        critical=True,
    )
    cap_node = evaluator.add_leaf(
        id=f"venue_{idx0+1}_capacity",
        desc="Total seating capacity is provided and falls within the range of 4,000 to 15,000",
        parent=node,
        critical=True,
    )
    cap_claim = f"The total seating capacity of '{name}' is between 4,000 and 15,000 people."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_node,
        sources=urls,
        additional_instruction="The page should state or imply a total capacity within 4,000–15,000. Accept approximate or 'up to' phrasing. Reject if it clearly falls outside this range.",
    )

    # 8) Seating configuration types
    seat_types_text = venue.seating_types or "reserved seating and/or lawn/general admission seating"
    st_node = evaluator.add_leaf(
        id=f"venue_{idx0+1}_seating_types",
        desc="Description of seating configuration types (e.g., reserved seats, lawn seating, etc.) is provided",
        parent=node,
        critical=True,
    )
    st_claim = f"The venue offers seating configuration types including: {seat_types_text}."
    await evaluator.verify(
        claim=st_claim,
        node=st_node,
        sources=urls,
        additional_instruction="Look for mentions of 'reserved seats', 'lawn seating', 'general admission lawn', 'pit', or similar configuration terminology.",
    )

    # 9) Accessibility (ADA/wheelchair-accessible seating)
    acc_node = evaluator.add_leaf(
        id=f"venue_{idx0+1}_accessibility",
        desc="Information about ADA-compliant wheelchair-accessible seating is provided",
        parent=node,
        critical=True,
    )
    acc_claim = f"'{name}' provides ADA-compliant wheelchair-accessible seating options."
    await evaluator.verify(
        claim=acc_claim,
        node=acc_node,
        sources=urls,
        additional_instruction="Look for ADA, accessibility, wheelchair, companion seating, and similar terms on the page.",
    )

    # 10) Premium seating
    prem_text = venue.premium_seating or "VIP boxes, club seats, or luxury suites"
    prem_node = evaluator.add_leaf(
        id=f"venue_{idx0+1}_premium_seating",
        desc="At least one type of premium seating option (VIP boxes, club seating, or luxury suites) is identified",
        parent=node,
        critical=True,
    )
    prem_claim = f"The venue offers premium seating options such as {prem_text}."
    await evaluator.verify(
        claim=prem_claim,
        node=prem_node,
        sources=urls,
        additional_instruction="Accept mentions like VIP boxes, VIP club, club seats, suites, premium lawn, or season boxes as premium offerings.",
    )

    # 11) Parking facilities
    park_node = evaluator.add_leaf(
        id=f"venue_{idx0+1}_parking",
        desc="Information about on-site or designated parking facilities is provided",
        parent=node,
        critical=True,
    )
    park_claim = f"'{name}' has on-site or designated parking facilities for event attendees."
    await evaluator.verify(
        claim=park_claim,
        node=park_node,
        sources=urls,
        additional_instruction="Look for 'parking', 'on-site parking', 'designated parking', 'lots', 'garages', or official parking guidance.",
    )

    # 12) Food & beverage concessions
    amen_node = evaluator.add_leaf(
        id=f"venue_{idx0+1}_amenities",
        desc="Information about food and beverage concessions or services is provided",
        parent=node,
        critical=True,
    )
    amen_claim = f"Food and beverage concessions or services are available at '{name}'."
    await evaluator.verify(
        claim=amen_claim,
        node=amen_node,
        sources=urls,
        additional_instruction="Look for 'concessions', 'food & beverage', 'vendors', 'snacks', 'drinks', or 'beer/wine/cocktails' on official or authoritative pages.",
    )


# -----------------------------------------------------------------------------
# Distinctness check across venues
# -----------------------------------------------------------------------------
def venues_are_distinct(venues: List[VenueItem]) -> bool:
    def norm(x: Optional[str]) -> Optional[str]:
        if not _is_nonempty(x):
            return None
        return " ".join(x.strip().lower().split())

    names = [norm(v.name) for v in venues]
    addrs = [norm(v.address) for v in venues]

    # Require all three names and addresses to be present and pairwise distinct
    if any(n is None for n in names) or any(a is None for a in addrs):
        return False
    return len(set(names)) == len(names) and len(set(addrs)) == len(addrs)


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    # Initialize evaluator (root as non-critical to allow partial credit)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Venues evaluated independently
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

    # Extract venues
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Prepare exactly 3 venues (pad with empty if fewer)
    venues: List[VenueItem] = list(extracted.venues[:3])
    while len(venues) < 3:
        venues.append(VenueItem())

    # Verify each venue
    for i in range(3):
        await verify_single_venue(evaluator, root, venues[i], i)

    # Distinctness across the three venues (critical to overall task intent)
    evaluator.add_custom_node(
        result=venues_are_distinct(venues),
        id="distinctness",
        desc="All three venues are distinct locations with different names and addresses",
        parent=root,
        critical=True,
    )

    # Return structured summary
    return evaluator.get_summary()