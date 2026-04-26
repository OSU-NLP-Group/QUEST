import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "venues_us_multi_city_tour"
TASK_DESCRIPTION = (
    "I am organizing a multi-city entertainment tour and need to identify four suitable indoor venues across the United States. "
    "For each venue, provide the following information:\n\n"
    "1. Venue identification: Official venue name, complete physical address (including street, city, state, and ZIP code), "
    "and a reference URL from the venue's official website or a reliable source.\n"
    "2. Capacity: The venue's seating capacity, which must be either between 15,000-25,000 seats (for arena-type venues) OR between "
    "1,500-6,000 seats (for theater/performing arts venues). Include a reference URL verifying the capacity.\n"
    "3. Accessibility: Confirmation that the venue provides wheelchair-accessible seating, with a reference URL verifying this accessibility feature.\n"
    "4. Parking facilities: Information about on-site or adjacent parking facilities, including the number of parking spaces available. "
    "Include a reference URL verifying the parking information.\n"
    "5. Public transportation: Confirmation that the venue is accessible via public transportation (bus, rail, subway, or light rail), "
    "with a reference URL verifying this access.\n"
    "6. Contact information: Box office phone number or email contact, with a reference URL verifying this contact information.\n\n"
    "Additional requirement: The four venues must be located in at least three different U.S. states to ensure proper geographic distribution for the tour."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class USAddress(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None


class VenueItem(BaseModel):
    name: Optional[str] = None
    address: Optional[USAddress] = None

    identification_url: Optional[str] = None

    capacity: Optional[str] = None
    capacity_url: Optional[str] = None

    accessibility_url: Optional[str] = None

    parking_availability: Optional[str] = None
    parking_spaces: Optional[str] = None
    parking_url: Optional[str] = None

    transit_url: Optional[str] = None

    contact_info: Optional[str] = None
    contact_url: Optional[str] = None


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract up to six indoor entertainment venues mentioned in the answer. For each venue, extract the following fields exactly as stated:

- name: The official venue name.
- address:
  - street: Street address.
  - city: City name.
  - state: State (e.g., CA or California).
  - zip: ZIP code.
- identification_url: A URL (preferably from the official venue website or a reliable source) that supports the venue identification/address.
- capacity: The seating capacity as described (allow ranges or approximate values as written).
- capacity_url: A URL that verifies the seating capacity.
- accessibility_url: A URL that verifies wheelchair-accessible seating.
- parking_availability: A brief description of on-site or adjacent parking availability (e.g., “on-site parking garage”).
- parking_spaces: The number of parking spaces, as mentioned (can be approximate or a range; extract the stated number/range).
- parking_url: A URL that verifies parking information (availability and/or capacity).
- transit_url: A URL that verifies public transport access (bus, rail, subway, or light rail).
- contact_info: Box office phone number or email contact (extract exactly as written).
- contact_url: A URL that verifies the box office contact information.

Rules:
- Only extract URLs explicitly present in the answer text; do not invent URLs.
- If any required field is missing for a venue, set it to null (for strings) or an empty object for address.
- Return a JSON object with a 'venues' array of these objects in the order they appear in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    return url.strip().lower().startswith(("http://", "https://"))


def _has_full_address(addr: Optional[USAddress]) -> bool:
    if not addr:
        return False
    return bool(addr.street and addr.city and addr.state and addr.zip)


def _normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return re.sub(r"[\s\-\.,&]+", " ", name.strip().lower())


def _parse_ints(text: Optional[str]) -> List[int]:
    if not text:
        return []
    # Extract numbers like 1,500, 1500, 15,000–20,000, etc.
    nums = re.findall(r"\d[\d,]*", text)
    parsed = []
    for n in nums:
        try:
            parsed.append(int(n.replace(",", "")))
        except Exception:
            pass
    return parsed


def capacity_in_allowed_range(capacity_text: Optional[str]) -> bool:
    """
    Allowed ranges:
      - Arena-type: 15,000–25,000
      - Theater/performing arts: 1,500–6,000
    We pass if ANY number in the extracted capacity string falls within either range.
    """
    values = _parse_ints(capacity_text)
    if not values:
        return False
    for v in values:
        if 15000 <= v <= 25000:
            return True
        if 1500 <= v <= 6000:
            return True
    return False


def parking_spaces_provided(spaces_text: Optional[str]) -> bool:
    """Return True if at least one integer is present in the parking spaces field."""
    return len(_parse_ints(spaces_text)) > 0


def get_state_norm(state_text: Optional[str]) -> Optional[str]:
    if not state_text:
        return None
    s = state_text.strip().upper()
    # If it's a two-letter code, keep it; else return the uppercase full string
    if re.fullmatch(r"[A-Z]{2}", s):
        return s
    # Normalize common full names to codes could be added; for now return uppercase full name
    return s


# --------------------------------------------------------------------------- #
# Verification builder for a single venue                                     #
# --------------------------------------------------------------------------- #
async def verify_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    venue_index: int,
) -> None:
    """
    Build verification sub-tree for one venue and execute required verifications.
    """
    vid = f"Venue_{venue_index}"

    venue_node = evaluator.add_parallel(
        id=vid,
        desc=f"{['First','Second','Third','Fourth','Fifth','Sixth'][venue_index-1] if venue_index<=6 else f'Venue #{venue_index}'} venue and its required attributes.",
        parent=parent_node,
        critical=False  # allow partial credit per venue
    )

    # Identification group
    ident_group = evaluator.add_parallel(
        id=f"{vid}_Identification",
        desc=f"{vid.replace('_', ' ')} identification details and a supporting reference URL.",
        parent=venue_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(venue.name and venue.name.strip()),
        id=f"{vid}_Name",
        desc="Official venue name is provided.",
        parent=ident_group,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_full_address(venue.address),
        id=f"{vid}_Full_Address",
        desc="Complete physical address is provided (street, city, state, ZIP code).",
        parent=ident_group,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_valid_url(venue.identification_url),
        id=f"{vid}_Identification_URL",
        desc="A reference URL is provided (official venue website or reliable source) supporting the venue identification/address.",
        parent=ident_group,
        critical=True
    )

    # Eligibility group
    elig_group = evaluator.add_parallel(
        id=f"{vid}_Eligibility",
        desc="Venue meets the basic eligibility constraints (indoor, entertainment venue type, US location).",
        parent=venue_node,
        critical=True
    )

    indoor_leaf = evaluator.add_leaf(
        id=f"{vid}_Is_Indoor_Entertainment_Venue",
        desc="Venue is an indoor entertainment venue (arena, theater, or performing arts center).",
        parent=elig_group,
        critical=True
    )
    name_for_claim = venue.name or f"Venue #{venue_index}"
    await evaluator.verify(
        claim=f"The venue named '{name_for_claim}' is an indoor entertainment venue (arena, theater, or performing arts center).",
        node=indoor_leaf,
        sources=venue.identification_url,
        additional_instruction=(
            "Use the provided page to determine if events are held indoors. "
            "Accept synonyms like 'concert hall', 'performing arts center', 'arena', 'coliseum'. "
            "Evidence such as indoor seating charts, enclosed auditorium references, or descriptions implying indoor events suffices."
        ),
    )

    us_leaf = evaluator.add_leaf(
        id=f"{vid}_Is_In_US",
        desc="Venue is located in the United States.",
        parent=elig_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{name_for_claim}' is located in the United States.",
        node=us_leaf,
        sources=venue.identification_url,
        additional_instruction=(
            "Confirm via the page that the location is within the U.S. "
            "Indications like a U.S. city and state (e.g., 'Houston, TX') or 'USA' should be considered sufficient."
        ),
    )

    # Capacity group
    cap_group = evaluator.add_parallel(
        id=f"{vid}_Capacity",
        desc="Capacity satisfies the allowed range and is verifiable.",
        parent=venue_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=capacity_in_allowed_range(venue.capacity),
        id=f"{vid}_Capacity_Range",
        desc="Seating capacity falls within either 15,000–25,000 (arena-type) OR 1,500–6,000 (theater/performing arts type).",
        parent=cap_group,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_valid_url(venue.capacity_url),
        id=f"{vid}_Capacity_URL",
        desc="A reference URL verifying the seating capacity is provided.",
        parent=cap_group,
        critical=True
    )

    # Accessibility group
    access_group = evaluator.add_parallel(
        id=f"{vid}_Accessibility",
        desc="Wheelchair-accessible seating is confirmed and verifiable.",
        parent=venue_node,
        critical=True
    )

    wchair_leaf = evaluator.add_leaf(
        id=f"{vid}_Wheelchair_Seating",
        desc="Venue provides wheelchair-accessible seating.",
        parent=access_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{name_for_claim}' provides wheelchair-accessible seating.",
        node=wchair_leaf,
        sources=venue.accessibility_url,
        additional_instruction=(
            "Look for terms like 'ADA', 'accessible seating', 'wheelchair seating', or similar on the provided page."
        ),
    )

    evaluator.add_custom_node(
        result=_is_valid_url(venue.accessibility_url),
        id=f"{vid}_Accessibility_URL",
        desc="A reference URL verifying wheelchair-accessible seating is provided.",
        parent=access_group,
        critical=True
    )

    # Parking group
    parking_group = evaluator.add_parallel(
        id=f"{vid}_Parking",
        desc="Parking is available, includes number of spaces, and is verifiable.",
        parent=venue_node,
        critical=True
    )

    park_avail_leaf = evaluator.add_leaf(
        id=f"{vid}_Parking_Availability",
        desc="Venue has on-site or adjacent parking facilities.",
        parent=parking_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{name_for_claim}' has on-site or adjacent parking facilities.",
        node=park_avail_leaf,
        sources=venue.parking_url,
        additional_instruction=(
            "Accept evidence for on-site parking garages/lots or adjacent facilities (official or venue-managed). "
            "Mentions of 'parking', 'garage', or 'lot' serving the venue are sufficient."
        ),
    )

    evaluator.add_custom_node(
        result=parking_spaces_provided(venue.parking_spaces),
        id=f"{vid}_Parking_Capacity",
        desc="Number of parking spaces is provided.",
        parent=parking_group,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_valid_url(venue.parking_url),
        id=f"{vid}_Parking_URL",
        desc="A reference URL verifying parking availability and/or capacity is provided.",
        parent=parking_group,
        critical=True
    )

    # Public Transportation group
    transit_group = evaluator.add_parallel(
        id=f"{vid}_Public_Transportation",
        desc="Public transportation access is confirmed and verifiable.",
        parent=venue_node,
        critical=True
    )

    transit_access_leaf = evaluator.add_leaf(
        id=f"{vid}_Transit_Access",
        desc="Venue is accessible via public transportation (bus, rail, subway, or light rail).",
        parent=transit_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{name_for_claim}' is accessible via public transportation (bus, rail, subway, or light rail).",
        node=transit_access_leaf,
        sources=venue.transit_url,
        additional_instruction=(
            "Look for references to nearby transit lines, stations, bus routes, or rail services enabling access to the venue."
        ),
    )

    evaluator.add_custom_node(
        result=_is_valid_url(venue.transit_url),
        id=f"{vid}_Transit_URL",
        desc="A reference URL verifying public transportation access is provided.",
        parent=transit_group,
        critical=True
    )

    # Contact group
    contact_group = evaluator.add_parallel(
        id=f"{vid}_Contact",
        desc="Box office contact is provided and verifiable.",
        parent=venue_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(venue.contact_info and venue.contact_info.strip()),
        id=f"{vid}_Contact_Info",
        desc="Box office phone number or email contact is provided.",
        parent=contact_group,
        critical=True
    )

    evaluator.add_custom_node(
        result=_is_valid_url(venue.contact_url),
        id=f"{vid}_Contact_URL",
        desc="A reference URL verifying the box office contact information is provided.",
        parent=contact_group,
        critical=True
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
    Evaluate an answer for the multi-city indoor venues task.
    """
    # Initialize evaluator (root should be non-critical to allow partial credit across venues)
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

    # Extract venue information
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Select first 4 venues; pad if fewer
    venues = (extracted.venues or [])[:4]
    while len(venues) < 4:
        venues.append(VenueItem())

    # Venue set completeness checks (critical gate)
    set_node = evaluator.add_parallel(
        id="Venue_Set_Completeness",
        desc="Submission includes the required number of venues and they are distinct venues.",
        parent=root,
        critical=True
    )

    # Four venues provided (names present)
    names_present = [v for v in venues if v.name and v.name.strip()]
    evaluator.add_custom_node(
        result=(len(names_present) >= 4),
        id="Four_Venues_Provided",
        desc="Exactly four venues are provided.",
        parent=set_node,
        critical=True
    )

    # Venues are distinct (by normalized names)
    normalized_names = [_normalize_name(v.name) for v in venues if v.name]
    distinct = len(set(n for n in normalized_names if n)) == len([n for n in normalized_names if n])
    evaluator.add_custom_node(
        result=distinct and len(names_present) >= 4,
        id="Venues_Are_Distinct",
        desc="The four venues are not duplicates of the same venue (distinct venue identities).",
        parent=set_node,
        critical=True
    )

    # Build per-venue verification trees
    for idx, venue in enumerate(venues, start=1):
        await verify_venue(evaluator, root, venue, idx)

    # Geographic distribution: at least three different states among the four venues
    states = []
    for v in venues:
        s = get_state_norm(v.address.state if v.address else None)
        if s:
            states.append(s)
    unique_states = set(states)
    evaluator.add_custom_node(
        result=(len(unique_states) >= 3),
        id="Geographic_Distribution",
        desc="The four venues are located in at least three different U.S. states.",
        parent=root,
        critical=True
    )

    # Add custom info for debugging purposes
    evaluator.add_custom_info(
        info={
            "selected_venue_names": [v.name for v in venues],
            "states_extracted": list(unique_states),
            "capacities_text": [v.capacity for v in venues],
            "parking_spaces_text": [v.parking_spaces for v in venues],
        },
        info_type="debug",
        info_name="extraction_summary"
    )

    return evaluator.get_summary()