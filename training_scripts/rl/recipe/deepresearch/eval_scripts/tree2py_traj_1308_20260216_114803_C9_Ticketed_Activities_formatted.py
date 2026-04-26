import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_mid_sized_concert_venues_4_regions"
TASK_DESCRIPTION = """
Identify four mid-sized concert venues (capacity between 1,000 and 6,500) in the United States, with one venue located in each of the following four regions: Northeast, Southeast, Midwest, and West. For each venue, provide the following information:

1. Venue Name: The official name of the concert venue
2. Location: Complete physical address including street address, city, state, and zip code
3. Capacity: The exact seating capacity of the venue
4. Confirmed Event: At least one specific concert or musical performance scheduled at this venue during 2025 or 2026, including the event name and date
5. Ticket Information: Evidence that tickets are or were available for purchase for the identified event
6. Accessibility: Confirmation that the venue offers ADA-compliant accessible seating
7. Official Website: A direct link to the venue's official website or primary ticketing page

Each venue must be located in a different state, and all information must be verifiable through the provided URLs.
"""

# --------------------------------------------------------------------------- #
# Region/state utilities                                                      #
# --------------------------------------------------------------------------- #

NORTHEAST_STATES = {
    "CONNECTICUT", "MAINE", "MASSACHUSETTS", "NEW HAMPSHIRE", "RHODE ISLAND", "VERMONT",
    "NEW JERSEY", "NEW YORK", "PENNSYLVANIA"
}
SOUTHEAST_STATES = {
    "ALABAMA", "FLORIDA", "GEORGIA", "KENTUCKY", "MISSISSIPPI", "NORTH CAROLINA", "SOUTH CAROLINA",
    "TENNESSEE", "VIRGINIA", "WEST VIRGINIA", "ARKANSAS", "LOUISIANA"
}
MIDWEST_STATES = {
    "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "MICHIGAN", "MINNESOTA", "MISSOURI",
    "NEBRASKA", "NORTH DAKOTA", "OHIO", "SOUTH DAKOTA", "WISCONSIN"
}
WEST_STATES = {
    "ALASKA", "ARIZONA", "CALIFORNIA", "COLORADO", "HAWAII", "IDAHO", "MONTANA", "NEVADA",
    "NEW MEXICO", "OREGON", "UTAH", "WASHINGTON", "WYOMING"
}

STATE_ABBREVIATIONS = {
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
    "DC": "DISTRICT OF COLUMBIA"
}

def normalize_state_name(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip().upper()
    if s in STATE_ABBREVIATIONS:
        return STATE_ABBREVIATIONS[s]
    # Handle cases like "N.Y." or extra punctuation
    s_clean = re.sub(r"[^\w\s]", "", s).strip()
    if s_clean in STATE_ABBREVIATIONS:
        return STATE_ABBREVIATIONS[s_clean]
    return s

def state_to_region(state: Optional[str]) -> Optional[str]:
    s = normalize_state_name(state)
    if not s:
        return None
    if s in NORTHEAST_STATES:
        return "Northeast"
    if s in SOUTHEAST_STATES:
        return "Southeast"
    if s in MIDWEST_STATES:
        return "Midwest"
    if s in WEST_STATES:
        return "West"
    return None

# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #

def dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def parse_capacity_number(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # Remove non-digit except commas and spaces
    m = re.findall(r"\d[\d,\.]*", text)
    if not m:
        return None
    # Use the first numeric token as capacity
    num_str = m[0]
    num_str = num_str.replace(",", "")
    try:
        # If decimal, cast to int
        return int(float(num_str))
    except Exception:
        return None

def any_year_in_range(date_text: Optional[str], years: Tuple[int, int] = (2025, 2026)) -> bool:
    if not date_text:
        return False
    yrs = re.findall(r"\b(20\d{2})\b", date_text)
    for y in yrs:
        try:
            yi = int(y)
            if years[0] <= yi <= years[1]:
                return True
        except Exception:
            continue
    return False

def make_sources(*url_lists: List[str], single_urls: Optional[List[Optional[str]]] = None) -> List[str]:
    urls: List[str] = []
    for lst in url_lists:
        urls.extend(lst or [])
    if single_urls:
        for u in single_urls:
            if u:
                urls.append(u)
    return dedup_urls(urls)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #

class VenueItem(BaseModel):
    region: Optional[str] = None  # If the answer explicitly labels the region; optional
    name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None

    capacity: Optional[str] = None
    capacity_source_urls: List[str] = Field(default_factory=list)

    # Location verification sources (e.g., official site "Contact"/"Visit us" page)
    location_source_urls: List[str] = Field(default_factory=list)
    # Region reference URLs (e.g., a page listing which states belong to a region)
    regional_reference_urls: List[str] = Field(default_factory=list)

    official_website_url: Optional[str] = None

    event_name: Optional[str] = None
    event_date: Optional[str] = None
    event_source_urls: List[str] = Field(default_factory=list)

    ticket_urls: List[str] = Field(default_factory=list)

    accessibility_urls: List[str] = Field(default_factory=list)

class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #

def prompt_extract_venues() -> str:
    return """
Extract up to four mid-sized concert venues mentioned in the answer, aiming to include one from each U.S. region: Northeast, Southeast, Midwest, and West (if available). For each venue, extract ONLY what is explicitly present in the answer. Do not invent or infer URLs or details.

For each venue, return these fields:
- region: The region label used in the answer (e.g., "Northeast", "Southeast", "Midwest", "West") if explicitly mentioned; otherwise null.
- name: Official venue name
- street_address: Street address (e.g., "123 Main St")
- city: City
- state: State (full name or abbreviation exactly as in the answer)
- zip_code: ZIP code (5-digit or ZIP+4 if present)

- capacity: The exact seating capacity number or text as stated in the answer
- capacity_source_urls: Array of URL(s) cited that support the capacity

- location_source_urls: Array of URL(s) cited that show the venue’s address/location (often the venue’s official website contact/visit page or profile page)
- regional_reference_urls: Array of URL(s) cited that show the state belongs to the specified region (for example, a reputable page listing the states in "Northeast" or "Midwest"); if none are cited, return an empty array

- official_website_url: URL to the official venue website or the primary ticketing page as cited

- event_name: The name of at least one concert or musical performance at the venue
- event_date: The event’s date as presented (any format)
- event_source_urls: Array of URL(s) cited that show the event at this venue (e.g., event listing page, calendar)

- ticket_urls: Array of URL(s) cited that show tickets were or are available to purchase for the event (can be sold out/archived)

- accessibility_urls: Array of URL(s) cited that mention ADA-compliant accessible seating or accessibility information for the venue

Return a JSON object with a field "venues" that is an array of up to 4 such venue objects. If the answer provides more than four venues, select at most the first one per region in the answer’s order; if a field is missing, set it to null (or an empty array for URL lists). Follow URL extraction rules strictly and only return URLs explicitly present in the answer.
"""

# --------------------------------------------------------------------------- #
# Verification helpers per-venue                                              #
# --------------------------------------------------------------------------- #

async def verify_geographic(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    venue: VenueItem,
    target_region: str
) -> None:
    geo_node = evaluator.add_parallel(
        id=f"Geographic_Compliance_V{idx}",
        desc=f"Venue is located in the {target_region} region of the United States",
        parent=parent_node,
        critical=True
    )

    # Existence checks
    evaluator.add_custom_node(
        result=bool(venue.state and venue.state.strip()),
        id=f"State_Identification_V{idx}",
        desc="Specific state within {} region is identified".format(target_region),
        parent=geo_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(venue.city and venue.city.strip()),
        id=f"City_Identification_V{idx}",
        desc="City within the identified state is provided",
        parent=geo_node,
        critical=True
    )

    # Regional verification using URLs
    regional_leaf = evaluator.add_leaf(
        id=f"Regional_Verification_URL_V{idx}",
        desc=f"URL reference confirms venue location in {target_region} region",
        parent=geo_node,
        critical=True
    )
    region_claim = (
        f"The venue '{venue.name or 'the venue'}' is located in {venue.city or ''}, "
        f"{venue.state or ''}, and {normalize_state_name(venue.state) or venue.state or ''} "
        f"is part of the {target_region} region of the United States."
    )
    sources = make_sources(
        venue.location_source_urls,
        venue.regional_reference_urls,
        single_urls=[venue.official_website_url]
    )
    await evaluator.verify(
        claim=region_claim,
        node=regional_leaf,
        sources=sources,
        additional_instruction=(
            "Verify in two parts using the provided URLs: (1) the venue's city/state is as claimed; "
            "(2) the state is recognized as part of the specified region. "
            "It's acceptable if the venue page confirms the city/state while a separate reputable page "
            "confirms the state-region mapping. Allow minor formatting variations."
        )
    )

async def verify_capacity(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    venue: VenueItem
) -> None:
    cap_node = evaluator.add_parallel(
        id=f"Capacity_Requirements_V{idx}",
        desc="Venue capacity falls within mid-sized range (1,000-6,500)",
        parent=parent_node,
        critical=True
    )

    # Capacity provided
    evaluator.add_custom_node(
        result=bool(venue.capacity and venue.capacity.strip()),
        id=f"Capacity_Number_V{idx}",
        desc="Specific capacity number is provided",
        parent=cap_node,
        critical=True
    )

    # Capacity in range
    cap_int = parse_capacity_number(venue.capacity)
    in_range = cap_int is not None and 1000 <= cap_int <= 6500
    evaluator.add_custom_node(
        result=in_range,
        id=f"Capacity_Range_Check_V{idx}",
        desc="Capacity is between 1,000 and 6,500 inclusive",
        parent=cap_node,
        critical=True
    )

    # Capacity source verification
    cap_leaf = evaluator.add_leaf(
        id=f"Capacity_Source_URL_V{idx}",
        desc="URL reference confirms the stated capacity",
        parent=cap_node,
        critical=True
    )
    cap_claim = (
        f"The seating capacity of the venue '{venue.name or 'the venue'}' is {cap_int if cap_int is not None else (venue.capacity or '').strip()}."
    )
    sources = make_sources(
        venue.capacity_source_urls,
        single_urls=[venue.official_website_url]
    )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the provided page(s) explicitly state the venue's seating capacity matching the claim. "
            "Allow common synonyms like 'capacity'/'seating capacity'. Prefer a single exact number; "
            "if multiple capacities are shown (e.g., configurations), accept the number the answer claims."
        )
    )

async def verify_event(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    venue: VenueItem
) -> None:
    evt_node = evaluator.add_parallel(
        id=f"Event_Confirmation_V{idx}",
        desc="At least one confirmed concert event scheduled for 2025-2026",
        parent=parent_node,
        critical=True
    )

    # Event name provided
    evaluator.add_custom_node(
        result=bool(venue.event_name and venue.event_name.strip()),
        id=f"Event_Name_V{idx}",
        desc="Specific concert or musical performance name is provided",
        parent=evt_node,
        critical=True
    )

    # Event date in range 2025-2026 (logic check)
    evaluator.add_custom_node(
        result=any_year_in_range(venue.event_date, (2025, 2026)),
        id=f"Event_Date_V{idx}",
        desc="Event date falls within 2025-2026 timeframe",
        parent=evt_node,
        critical=True
    )

    # Ticket availability evidence (URL verify)
    tix_leaf = evaluator.add_leaf(
        id=f"Ticket_Availability_V{idx}",
        desc="Evidence that tickets are or were available for purchase",
        parent=evt_node,
        critical=True
    )
    ticket_claim = (
        f"Tickets are or were available for purchase for the event '{venue.event_name or ''}' "
        f"at '{venue.name or 'the venue'}' on '{venue.event_date or ''}'."
    )
    await evaluator.verify(
        claim=ticket_claim,
        node=tix_leaf,
        sources=dedup_urls(venue.ticket_urls),
        additional_instruction=(
            "Confirm that the linked page is a ticket purchase or listing page (e.g., Ticketmaster, AXS, Live Nation, venue's box office). "
            "Sold out or past event pages still count as evidence if they originally offered tickets. "
            "Look for cues like 'Tickets', 'Buy', 'See Tickets', pricing, or seat maps."
        )
    )

    # Event source verification (URL verify)
    evt_leaf = evaluator.add_leaf(
        id=f"Event_Source_URL_V{idx}",
        desc="URL reference confirms the event at this specific venue",
        parent=evt_node,
        critical=True
    )
    evt_claim = (
        f"The event '{venue.event_name or ''}' is scheduled at the venue '{venue.name or 'the venue'}' "
        f"in {venue.city or ''}, {venue.state or ''} on {venue.event_date or 'the stated date'}."
    )
    evt_sources = make_sources(
        venue.event_source_urls,
        single_urls=[venue.official_website_url]
    )
    await evaluator.verify(
        claim=evt_claim,
        node=evt_leaf,
        sources=evt_sources,
        additional_instruction=(
            "Verify that the event page explicitly associates the event with the specified venue and date. "
            "Minor formatting differences in the event name or date are acceptable (case, punctuation, month abbreviations). "
            "If multiple dates are shown, ensure at least one matches the stated 2025/2026 date."
        )
    )

async def verify_accessibility(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    venue: VenueItem
) -> None:
    acc_node = evaluator.add_parallel(
        id=f"Accessibility_Features_V{idx}",
        desc="ADA-compliant accessible seating information is provided",
        parent=parent_node,
        critical=True
    )

    # Verify accessible seating via URL(s)
    acc_leaf = evaluator.add_leaf(
        id=f"Accessible_Seating_Availability_V{idx}",
        desc="Venue offers designated accessible seating",
        parent=acc_node,
        critical=True
    )
    acc_claim = (
        f"The venue '{venue.name or 'the venue'}' provides ADA-compliant accessible seating (e.g., wheelchair-accessible seating)."
    )
    acc_sources = make_sources(
        venue.accessibility_urls,
        single_urls=[venue.official_website_url]
    )
    await evaluator.verify(
        claim=acc_claim,
        node=acc_leaf,
        sources=acc_sources,
        additional_instruction=(
            "Confirm language indicating ADA, accessibility, wheelchair-accessible seating, companion seating, or similar. "
            "Accept official venue pages or authoritative ticketing provider pages detailing accessibility."
        )
    )

    # Existence of documentation URL(s)
    evaluator.add_custom_node(
        result=bool(venue.accessibility_urls and len(venue.accessibility_urls) > 0),
        id=f"Accessibility_Documentation_URL_V{idx}",
        desc="URL reference confirms accessibility features",
        parent=acc_node,
        critical=True
    )

async def verify_venue_docs(
    evaluator: Evaluator,
    parent_node,
    idx: int,
    venue: VenueItem
) -> None:
    docs_node = evaluator.add_parallel(
        id=f"Venue_Documentation_V{idx}",
        desc="Complete venue identification and contact information",
        parent=parent_node,
        critical=True
    )

    # Venue name provided
    evaluator.add_custom_node(
        result=bool(venue.name and venue.name.strip()),
        id=f"Venue_Name_V{idx}",
        desc="Official venue name is provided",
        parent=docs_node,
        critical=True
    )

    # Full address presence check
    has_full_address = all([
        bool(venue.street_address and venue.street_address.strip()),
        bool(venue.city and venue.city.strip()),
        bool(venue.state and venue.state.strip()),
        bool(venue.zip_code and venue.zip_code.strip())
    ])
    evaluator.add_custom_node(
        result=has_full_address,
        id=f"Physical_Address_V{idx}",
        desc="Complete street address including street, city, state, and zip code",
        parent=docs_node,
        critical=True
    )

    # Official website / primary ticketing page verification
    site_leaf = evaluator.add_leaf(
        id=f"Official_Website_URL_V{idx}",
        desc="Link to venue's official website or primary ticketing page",
        parent=docs_node,
        critical=True
    )
    site_claim = (
        f"This page is the official website or the primary ticketing page for the venue '{venue.name or 'the venue'}' "
        f"in {venue.city or ''}, {venue.state or ''}."
    )
    await evaluator.verify(
        claim=site_claim,
        node=site_leaf,
        sources=venue.official_website_url or None,
        additional_instruction=(
            "Verify that the page appears to be the venue's official website (branding, contact info) "
            "or its primary ticketing provider page (e.g., Ticketmaster/AXS/Live Nation) for the venue."
        )
    )

# --------------------------------------------------------------------------- #
# Venue-level orchestrator                                                    #
# --------------------------------------------------------------------------- #

async def verify_one_venue(
    evaluator: Evaluator,
    root,
    idx: int,
    target_region: str,
    venue: VenueItem
) -> None:
    """
    Build and verify the subtree for a single venue.
    idx: 1..4
    target_region: one of ["Northeast", "Southeast", "Midwest", "West"]
    """
    venue_node = evaluator.add_parallel(
        id=f"Venue_{idx}",
        desc=f"{['First','Second','Third','Fourth'][idx-1]} venue meeting all specified criteria",
        parent=root,
        critical=False
    )

    # Geographic checks (region-specific)
    await verify_geographic(evaluator, venue_node, idx, venue, target_region)

    # Capacity checks
    await verify_capacity(evaluator, venue_node, idx, venue)

    # Event checks
    await verify_event(evaluator, venue_node, idx, venue)

    # Accessibility checks
    await verify_accessibility(evaluator, venue_node, idx, venue)

    # Venue documentation checks
    await verify_venue_docs(evaluator, venue_node, idx, venue)

# --------------------------------------------------------------------------- #
# Assignment of extracted venues to target regions                            #
# --------------------------------------------------------------------------- #

def assign_venues_to_regions(extracted: VenuesExtraction) -> Dict[str, VenueItem]:
    """
    Assign up to one venue per target region using either the explicit region label or
    by mapping the state to a region. Returns mapping for the four target regions.
    """
    targets = ["Northeast", "Southeast", "Midwest", "West"]
    assignment: Dict[str, VenueItem] = {}

    # First pass: if answer explicitly labeled region
    for v in extracted.venues:
        if not v:
            continue
        if v.region and v.region.strip():
            label = v.region.strip().title()
            if label in targets and label not in assignment:
                assignment[label] = v
        if len(assignment) == 4:
            break

    # Second pass: infer by state if not yet assigned
    for v in extracted.venues:
        if len(assignment) == 4:
            break
        if not v:
            continue
        if any(v is vv for vv in assignment.values()):
            continue
        inferred = state_to_region(v.state)
        if inferred and inferred in targets and inferred not in assignment:
            assignment[inferred] = v

    # If still missing, fill with empty placeholders
    for t in targets:
        if t not in assignment:
            assignment[t] = VenueItem()

    return assignment

# --------------------------------------------------------------------------- #
# Root-level uniqueness check                                                 #
# --------------------------------------------------------------------------- #

def compute_state_uniqueness(assigned: Dict[str, VenueItem]) -> bool:
    states = []
    for region in ["Northeast", "Southeast", "Midwest", "West"]:
        st = assigned[region].state if assigned.get(region) else None
        st_norm = normalize_state_name(st) if st else None
        if st_norm:
            states.append(st_norm)
    if len(states) < 4:
        return False
    return len(set(states)) == 4

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
    Evaluate an answer for the task: four mid-sized U.S. concert venues across regions with complete info.
    """
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
        default_model=model
    )

    # Extract structured venue information
    extracted: VenuesExtraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Assign venues to the target four regions
    assigned = assign_venues_to_regions(extracted)

    # Record assignment info
    evaluator.add_custom_info(
        info={
            "assigned_regions": {
                region: {
                    "name": assigned[region].name,
                    "city": assigned[region].city,
                    "state": assigned[region].state,
                    "capacity": assigned[region].capacity,
                    "official_website_url": assigned[region].official_website_url
                } for region in ["Northeast", "Southeast", "Midwest", "West"]
            }
        },
        info_type="assignment",
        info_name="region_assignment"
    )

    # Root-level critical uniqueness check: all four in different states
    evaluator.add_custom_node(
        result=compute_state_uniqueness(assigned),
        id="State_Uniqueness_Check",
        desc="All four venues are located in different US states",
        parent=root,
        critical=True
    )

    # Build and verify each venue subtree
    region_order = ["Northeast", "Southeast", "Midwest", "West"]
    for i, region in enumerate(region_order, start=1):
        await verify_one_venue(
            evaluator=evaluator,
            root=root,
            idx=i,
            target_region=region,
            venue=assigned[region]
        )

    return evaluator.get_summary()