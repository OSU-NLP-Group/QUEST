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
TASK_ID = "venues_4000_10000_ada"
TASK_DESCRIPTION = """
Identify three live music performance venues in the United States that host ticketed events and meet all of the following requirements:

1. Each venue must have a seating capacity between 4,000 and 10,000 people (inclusive).

2. Each venue must be classified as either an indoor arena or an outdoor amphitheater.

3. Each venue must provide wheelchair-accessible seating that complies with ADA requirements (minimum 1% of total capacity for venues over 1,000 seats).

4. At least one of the three venues must be an outdoor amphitheater, and at least one must be an indoor arena.

5. All three venues must be located in different U.S. states.

For each venue, provide the following information:
   - The venue's official name
   - The complete physical address (street address, city, state, and ZIP code)
   - The exact seating capacity
   - The venue type classification (indoor arena or outdoor amphitheater)
   - Information about wheelchair-accessible seating availability
   - A direct URL to the venue's official website or the official operator's website page for that venue

All information must be verifiable through official sources such as the venue's website, the operator's website, or official city/municipal sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Address(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    refs: List[str] = Field(default_factory=list)  # Official page URLs supporting the address


class CapacityInfo(BaseModel):
    capacity_raw: Optional[str] = None
    capacity_number: Optional[int] = None
    refs: List[str] = Field(default_factory=list)  # Official page URLs supporting capacity


class TypeInfo(BaseModel):
    classification: Optional[str] = None  # e.g., "indoor arena" or "outdoor amphitheater" (or synonyms)
    refs: List[str] = Field(default_factory=list)  # Official page URLs supporting type classification


class AccessibilityInfo(BaseModel):
    info_text: Optional[str] = None  # Description text about accessible seating from the answer
    accessible_seats_number: Optional[int] = None  # If the answer mentions a specific number
    refs: List[str] = Field(default_factory=list)  # Official accessibility/ADA URLs for the venue


class VenueItem(BaseModel):
    name: Optional[str] = None
    address: Optional[Address] = None
    capacity: Optional[CapacityInfo] = None
    venue_type: Optional[TypeInfo] = None
    accessibility: Optional[AccessibilityInfo] = None
    official_url: Optional[str] = None  # The official venue or operator URL


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to the first FIVE venue entries mentioned in the answer that are intended to satisfy the task.
    For each venue, extract the following fields as a JSON array `venues`:

    - name: The official venue name as written in the answer.
    - address: An object with
        - street: Street address (include suite/building if present).
        - city: City.
        - state: U.S. state (full name or 2-letter code).
        - zip: ZIP code.
        - refs: A list of official URL(s) provided in the answer that specifically support the address (prefer venue's own website, operator site, or official government/city site).
    - capacity: An object with
        - capacity_raw: Capacity as shown in the answer (string, may include commas/ranges).
        - capacity_number: A single integer for the main/standard seating capacity if explicitly provided or clearly implied (strip commas). If a range is given, choose the primary/typical capacity number if available; otherwise, leave null.
        - refs: A list of official URL(s) supporting the capacity.
    - venue_type: An object with
        - classification: The venue type string as stated in the answer (e.g., "indoor arena", "outdoor amphitheater"; allow synonyms like "arena", "amphitheatre", "shed", etc.).
        - refs: Official URL(s) supporting the type classification (can reuse the official venue page).
    - accessibility: An object with
        - info_text: The answer’s statement about wheelchair-accessible seating (e.g., "ADA-compliant accessible seating is available").
        - accessible_seats_number: If the answer states a specific count of wheelchair-accessible seats, extract that integer; otherwise null.
        - refs: Official URL(s) supporting accessibility/ADA info (venue policy page, seating map, city/operator page).
    - official_url: A single URL that is the venue's official website or the official operator's page for that venue (if multiple provided, choose the most authoritative—venue domain or operator domain like Live Nation/AEG/municipal).

    Rules:
    - Extract ONLY what appears in the provided answer. Do not invent.
    - For any missing field, set to null (or [] for lists).
    - Always include full URLs with protocol.
    - Prefer official sources (venue domain, operator domain, .gov city sites). Avoid third-party media or wikis for refs if more official URLs are present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
STATE_ABBR = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "FLORIDA": "FL", "GEORGIA": "GA",
    "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA",
    "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS", "MISSOURI": "MO",
    "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM", "NEW YORK": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH",
    "OKLAHOMA": "OK", "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT",
    "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
    "DISTRICT OF COLUMBIA": "DC", "WASHINGTON, DC": "DC", "DC": "DC",
}

ABBR_SET = set(STATE_ABBR.values())


def normalize_state(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = s.strip().upper()
    # Normalize common punctuation
    t = t.replace(".", "").replace(",", "")
    if t in ABBR_SET:
        return t
    # Try mapping full name
    if t in STATE_ABBR:
        return STATE_ABBR[t]
    # Handle "State of X"
    if t.startswith("STATE OF "):
        key = t.replace("STATE OF ", "")
        if key in STATE_ABBR:
            return STATE_ABBR[key]
    # If 2 letters but not recognized, return as-is
    if len(t) == 2:
        return t
    return t  # fallback, still used for uniqueness test


def normalize_venue_type(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    s = label.strip().lower()
    # Heuristics to map common synonyms
    if any(k in s for k in ["amphitheater", "amphitheatre", "shed", "outdoor amph"]):
        return "outdoor_amphitheater"
    if any(k in s for k in ["arena", "indoor arena", "coliseum", "indoor stadium"]):
        # Avoid classifying "amphitheater" as arena when both present
        if "amph" in s:
            return "outdoor_amphitheater"
        return "indoor_arena"
    # If simply "outdoor" + live venue
    if "outdoor" in s and ("theater" in s or "theatre" in s):
        return "outdoor_amphitheater"
    return None


def safe_first(lst: Optional[List[str]]) -> Optional[str]:
    if not lst:
        return None
    return lst[0]


def pick_sources(primary: Optional[List[str]], fallback: Optional[str]) -> List[str]:
    urls: List[str] = []
    if primary:
        urls.extend([u for u in primary if isinstance(u, str) and u.strip()])
    if fallback and (not urls):
        urls.append(fallback)
    return urls


# --------------------------------------------------------------------------- #
# Verification builder per-venue                                              #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    idx: int
) -> Dict[str, Any]:
    """
    Build verification sub-tree for a single venue and execute leaf verifications.
    Returns a dict with normalized state and type for cross-venue checks.
    """
    vn = evaluator.add_parallel(
        id=f"venue_{idx+1}",
        desc=f"{['First','Second','Third'][idx]} venue meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # -------- Name (critical existence) ---------------------------------- #
    name_ok = venue is not None and venue.name is not None and venue.name.strip() != ""
    evaluator.add_custom_node(
        result=name_ok,
        id=f"v{idx+1}_name",
        desc="Official venue name is provided",
        parent=vn,
        critical=True
    )
    venue_name = venue.name.strip() if (venue and venue.name) else "the venue"

    # -------- Address (parallel group) ----------------------------------- #
    addr_group = evaluator.add_parallel(
        id=f"v{idx+1}_address",
        desc="Complete physical address is provided",
        parent=vn,
        critical=True
    )
    street_ok = bool(venue and venue.address and venue.address.street and venue.address.street.strip())
    city_ok = bool(venue and venue.address and venue.address.city and venue.address.city.strip())
    state_ok = bool(venue and venue.address and venue.address.state and venue.address.state.strip())
    zip_ok = bool(venue and venue.address and venue.address.zip and venue.address.zip.strip())

    evaluator.add_custom_node(
        result=street_ok,
        id=f"v{idx+1}_street",
        desc="Street address is provided",
        parent=addr_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=city_ok,
        id=f"v{idx+1}_city",
        desc="City is provided",
        parent=addr_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=state_ok,
        id=f"v{idx+1}_state",
        desc="State is provided",
        parent=addr_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=zip_ok,
        id=f"v{idx+1}_zip",
        desc="ZIP code is provided",
        parent=addr_group,
        critical=True
    )

    addr_ref_node = evaluator.add_leaf(
        id=f"v{idx+1}_address_reference",
        desc="URL reference supporting the address information",
        parent=addr_group,
        critical=True
    )
    street = venue.address.street if (venue and venue.address) else None
    city = venue.address.city if (venue and venue.address) else None
    state = venue.address.state if (venue and venue.address) else None
    zipc = venue.address.zip if (venue and venue.address) else None

    addr_claim = f"The official address of {venue_name} is '{street}, {city}, {state} {zipc}'."
    addr_sources = pick_sources(venue.address.refs if (venue and venue.address) else [], venue.official_url if venue else None)
    await evaluator.verify(
        claim=addr_claim,
        node=addr_ref_node,
        sources=addr_sources,
        additional_instruction="Verify the full street, city, state, and ZIP on the official venue/operator page. Allow minor formatting (e.g., St. vs Street)."
    )

    # -------- Capacity (sequential group) -------------------------------- #
    cap_group = evaluator.add_sequential(
        id=f"v{idx+1}_capacity",
        desc="Seating capacity is within the required range of 4,000-10,000",
        parent=vn,
        critical=True
    )
    cap_raw = venue.capacity.capacity_raw if (venue and venue.capacity) else None
    cap_num = venue.capacity.capacity_number if (venue and venue.capacity) else None

    cap_stated = bool((cap_raw and str(cap_raw).strip()) or isinstance(cap_num, int))
    evaluator.add_custom_node(
        result=cap_stated,
        id=f"v{idx+1}_capacity_stated",
        desc="Seating capacity number is provided",
        parent=cap_group,
        critical=True
    )

    in_range = bool(isinstance(cap_num, int) and 4000 <= cap_num <= 10000)
    evaluator.add_custom_node(
        result=in_range,
        id=f"v{idx+1}_capacity_range",
        desc="Capacity falls between 4,000 and 10,000 inclusive",
        parent=cap_group,
        critical=True
    )

    cap_ref_node = evaluator.add_leaf(
        id=f"v{idx+1}_capacity_reference",
        desc="URL reference supporting the capacity information",
        parent=cap_group,
        critical=True
    )
    if isinstance(cap_num, int):
        cap_claim = f"The seating capacity of {venue_name} is {cap_num}."
    else:
        cap_claim = f"The venue's seating capacity is stated as '{cap_raw}'."
    cap_sources = pick_sources(venue.capacity.refs if (venue and venue.capacity) else [], venue.official_url if venue else None)
    await evaluator.verify(
        claim=cap_claim,
        node=cap_ref_node,
        sources=cap_sources,
        additional_instruction="Check the official venue/operator page to confirm the stated seating capacity (allow minor wording differences like 'approx.' or 'about')."
    )

    # -------- Type (sequential group) ------------------------------------ #
    type_group = evaluator.add_sequential(
        id=f"v{idx+1}_type",
        desc="Venue type classification is provided and valid",
        parent=vn,
        critical=True
    )
    vtype_str = venue.venue_type.classification if (venue and venue.venue_type) else None

    evaluator.add_custom_node(
        result=bool(vtype_str and vtype_str.strip()),
        id=f"v{idx+1}_type_stated",
        desc="Venue type (indoor arena or outdoor amphitheater) is specified",
        parent=type_group,
        critical=True
    )

    type_valid_node = evaluator.add_leaf(
        id=f"v{idx+1}_type_valid",
        desc="Venue type is either indoor arena or outdoor amphitheater",
        parent=type_group,
        critical=True
    )
    type_valid_claim = f"The label '{vtype_str}' corresponds to one of the two categories: indoor arena or outdoor amphitheater."
    await evaluator.verify(
        claim=type_valid_claim,
        node=type_valid_node,
        additional_instruction="Allow reasonable synonyms (e.g., 'arena' implies indoor arena; 'amphitheatre/shed' implies outdoor amphitheater)."
    )

    type_ref_node = evaluator.add_leaf(
        id=f"v{idx+1}_type_reference",
        desc="URL reference supporting the venue type classification",
        parent=type_group,
        critical=True
    )
    norm_type = normalize_venue_type(vtype_str)
    if norm_type == "indoor_arena":
        type_ref_claim = f"{venue_name} is an indoor arena."
    elif norm_type == "outdoor_amphitheater":
        type_ref_claim = f"{venue_name} is an outdoor amphitheater."
    else:
        # Fall back to claimed raw string
        type_ref_claim = f"{venue_name} matches the claimed type classification: '{vtype_str}'."
    type_sources = pick_sources(venue.venue_type.refs if (venue and venue.venue_type) else [], venue.official_url if venue else None)
    await evaluator.verify(
        claim=type_ref_claim,
        node=type_ref_node,
        sources=type_sources,
        additional_instruction="Check the official page to confirm the venue is indeed categorized as claimed. Consider cues like 'indoor', 'arena', 'amphitheater', seating bowl, roof, etc."
    )

    # -------- Accessibility (sequential group) --------------------------- #
    acc_group = evaluator.add_sequential(
        id=f"v{idx+1}_accessibility",
        desc="ADA wheelchair-accessible seating compliance is verified",
        parent=vn,
        critical=True
    )
    acc_text = venue.accessibility.info_text if (venue and venue.accessibility) else None
    acc_num = venue.accessibility.accessible_seats_number if (venue and venue.accessibility) else None

    evaluator.add_custom_node(
        result=bool(acc_text and acc_text.strip()) or bool(venue and venue.accessibility and venue.accessibility.refs),
        id=f"v{idx+1}_accessibility_info",
        desc="Information about wheelchair-accessible seating is provided",
        parent=acc_group,
        critical=True
    )

    acc_comp_node = evaluator.add_leaf(
        id=f"v{idx+1}_accessibility_compliance",
        desc="Wheelchair-accessible seating meets ADA minimum requirement (at least 1% of capacity for venues over 1,000 seats)",
        parent=acc_group,
        critical=True
    )
    # Build compliance claim: If we know numbers, state explicit; otherwise, verify ADA-compliant availability meets/exceeds minimum.
    if isinstance(cap_num, int) and cap_num > 1000:
        min_req = max(1, round(cap_num * 0.01))
        if isinstance(acc_num, int):
            comp_claim = f"{venue_name} provides at least {min_req} wheelchair-accessible seats, meeting/exceeding 1% of total capacity ({cap_num}). The venue lists {acc_num} accessible seats."
        else:
            comp_claim = f"{venue_name} provides wheelchair-accessible seating in compliance with ADA (at least 1% of {cap_num} total seats, i.e., ≥{min_req})."
    else:
        # If capacity not clearly numeric, verify general ADA compliance statement
        comp_claim = f"{venue_name} provides wheelchair-accessible seating that complies with ADA requirements."
    acc_sources = pick_sources(venue.accessibility.refs if (venue and venue.accessibility) else [], venue.official_url if venue else None)
    await evaluator.verify(
        claim=comp_claim,
        node=acc_comp_node,
        sources=acc_sources,
        additional_instruction="Accept if the official page explicitly states ADA-compliant accessible seating, seating maps with ADA sections, or policy pages indicating compliance. If explicit accessible seat counts are provided and meet ≥1% threshold, that also satisfies compliance."
    )

    acc_ref_node = evaluator.add_leaf(
        id=f"v{idx+1}_accessibility_reference",
        desc="URL reference supporting the accessibility information",
        parent=acc_group,
        critical=True
    )
    info_claim = f"Wheelchair-accessible seating is available at {venue_name}."
    await evaluator.verify(
        claim=info_claim,
        node=acc_ref_node,
        sources=acc_sources,
        additional_instruction="Verify accessible seating availability on an official venue/operator/government page."
    )

    # -------- Official URL (sequential group) ---------------------------- #
    url_group = evaluator.add_sequential(
        id=f"v{idx+1}_url",
        desc="Official website URL is provided",
        parent=vn,
        critical=True
    )
    url_provided = bool(venue and venue.official_url and venue.official_url.strip())
    evaluator.add_custom_node(
        result=url_provided,
        id=f"v{idx+1}_url_provided",
        desc="A URL to the venue's official website or operator's page is provided",
        parent=url_group,
        critical=True
    )

    url_valid_node = evaluator.add_leaf(
        id=f"v{idx+1}_url_valid",
        desc="The URL is a valid official source (venue website or official operator website)",
        parent=url_group,
        critical=True
    )
    url_claim = "This URL is the official website of the venue or the official operator's dedicated venue page."
    await evaluator.verify(
        claim=url_claim,
        node=url_valid_node,
        sources=venue.official_url if venue else None,
        additional_instruction="Confirm the page clearly represents the venue officially (venue's own domain, recognized operator like Live Nation/AEG, or city/municipal .gov)."
    )

    # Normalized state/type for cross-venue checks
    norm_state_val = normalize_state(state if state_ok else None)
    return {
        "state": norm_state_val,
        "type": norm_type
    }


# --------------------------------------------------------------------------- #
# Cross-venue checks                                                          #
# --------------------------------------------------------------------------- #
def add_cross_venue_checks(
    evaluator: Evaluator,
    parent_node,
    normalized_states: List[Optional[str]],
    normalized_types: List[Optional[str]]
) -> None:
    cross = evaluator.add_parallel(
        id="cross_venue_diversity",
        desc="Cross-venue requirements for diversity are met",
        parent=parent_node,
        critical=True
    )

    # Different states (critical)
    states_clean = [s for s in normalized_states if s]
    different_states = len(states_clean) == 3 and len(set(states_clean)) == 3
    evaluator.add_custom_node(
        result=different_states,
        id="different_states",
        desc="All three venues are located in different U.S. states",
        parent=cross,
        critical=True
    )

    # Type diversity (critical group with two critical children)
    type_div = evaluator.add_parallel(
        id="type_diversity",
        desc="At least one venue is an outdoor amphitheater and at least one is an indoor arena",
        parent=cross,
        critical=True
    )
    has_outdoor = any(t == "outdoor_amphitheater" for t in normalized_types)
    has_indoor = any(t == "indoor_arena" for t in normalized_types)

    evaluator.add_custom_node(
        result=has_outdoor,
        id="has_outdoor",
        desc="At least one venue is classified as an outdoor amphitheater",
        parent=type_div,
        critical=True
    )
    evaluator.add_custom_node(
        result=has_indoor,
        id="has_indoor",
        desc="At least one venue is classified as an indoor arena",
        parent=type_div,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the venues capacity/type/ADA task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall venues are independent items
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

    # Extract structured venues info
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Keep exactly first 3 venues (pad with empty if fewer)
    venues: List[VenueItem] = list(extraction.venues[:3])
    while len(venues) < 3:
        venues.append(VenueItem())

    # Build per-venue subtrees
    norm_states: List[Optional[str]] = []
    norm_types: List[Optional[str]] = []

    # Parent nodes for three venues
    # We keep Root non-critical (framework constraint); each venue subtree allows partial credit
    for i in range(3):
        result = await verify_single_venue(evaluator, root, venues[i], i)
        norm_states.append(result.get("state"))
        norm_types.append(result.get("type"))

    # Add cross-venue diversity checks (critical)
    add_cross_venue_checks(evaluator, root, norm_states, norm_types)

    # Add custom info for debugging
    evaluator.add_custom_info(
        info={
            "normalized_states": norm_states,
            "normalized_types": norm_types
        },
        info_type="derived_fields",
        info_name="normalized_fields_summary"
    )

    # Return evaluation summary
    return evaluator.get_summary()