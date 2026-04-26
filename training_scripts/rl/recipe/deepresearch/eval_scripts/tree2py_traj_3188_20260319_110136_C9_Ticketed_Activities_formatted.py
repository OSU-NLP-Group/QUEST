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
TASK_ID = "summer_2026_tour_venues"
TASK_DESCRIPTION = """
A concert promotion company is planning a summer 2026 arena tour for a major touring artist and needs to identify suitable venues across four different regions of the United States. For each of the four US regions listed below, identify one indoor arena venue that meets all of the following requirements:

Regional Requirements:
- One venue in the Northeastern United States (ME, NH, VT, MA, RI, CT, NY, NJ, PA)
- One venue in the Southeastern United States (FL, GA, SC, NC, VA, WV, KY, TN, AL, MS, AR, LA)
- One venue in the Midwestern United States (OH, IN, IL, MI, WI, MN, IA, MO, ND, SD, NE, KS)
- One venue in the Western United States (CA, OR, WA, NV, AZ, UT, CO, NM, WY, MT, ID)

Venue Requirements for Each Location:
1. The venue must be an indoor arena (not a stadium, outdoor amphitheater, or theater)
2. The venue must have a seating capacity between 5,000 and 25,000 people
3. The venue must comply with ADA accessibility standards, including accessible parking and accessible routes with ramps or elevators to seating areas

For each venue, provide:
- The venue name
- The specific city and state
- The seating capacity
- Confirmation that it is an indoor arena
- Confirmation of ADA accessibility features
- Reference URLs supporting each piece of information
"""


# --------------------------------------------------------------------------- #
# Region/state utilities                                                      #
# --------------------------------------------------------------------------- #
US_STATE_FULL_TO_ABBR = {
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
    "WISCONSIN": "WI", "WYOMING": "WY",
    "DISTRICT OF COLUMBIA": "DC", "WASHINGTON DC": "DC", "WASHINGTON D C": "DC",
}

US_STATE_ABBRS = set(US_STATE_FULL_TO_ABBR.values())

NORTHEAST_STATES = {"ME", "NH", "VT", "MA", "RI", "CT", "NY", "NJ", "PA"}
SOUTHEAST_STATES = {"FL", "GA", "SC", "NC", "VA", "WV", "KY", "TN", "AL", "MS", "AR", "LA"}
MIDWEST_STATES = {"OH", "IN", "IL", "MI", "WI", "MN", "IA", "MO", "ND", "SD", "NE", "KS"}
WEST_STATES = {"CA", "OR", "WA", "NV", "AZ", "UT", "CO", "NM", "WY", "MT", "ID"}


def normalize_state_abbrev(state_str: Optional[str]) -> Optional[str]:
    if not state_str:
        return None
    s = state_str.strip().upper()
    s = re.sub(r'[^A-Z\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()

    # Direct abbr like "NY"
    nospace = s.replace(" ", "")
    if len(nospace) == 2 and nospace in US_STATE_ABBRS:
        return nospace

    # Map full to abbr (handles e.g., "NEW YORK")
    if s in US_STATE_FULL_TO_ABBR:
        return US_STATE_FULL_TO_ABBR[s]

    # Special-case DC variations
    if "WASHINGTON" in s and ("DC" in nospace or "COLUMBIA" in s):
        return "DC"

    # Try without generic suffixes
    s2 = s.replace("STATE OF ", "").replace("COMMONWEALTH OF ", "").strip()
    if s2 in US_STATE_FULL_TO_ABBR:
        return US_STATE_FULL_TO_ABBR[s2]

    # Try removing spaces (NEWYORK etc.)
    compact = s.replace(" ", "")
    for full_name, abbr in US_STATE_FULL_TO_ABBR.items():
        if full_name.replace(" ", "") == compact:
            return abbr

    return None


def region_for_state_abbr(abbr: Optional[str]) -> Optional[str]:
    if not abbr:
        return None
    if abbr in NORTHEAST_STATES:
        return "NE"
    if abbr in SOUTHEAST_STATES:
        return "SE"
    if abbr in MIDWEST_STATES:
        return "MW"
    if abbr in WEST_STATES:
        return "W"
    return None


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # As written in the answer; can be full name or abbreviation
    seating_capacity: Optional[str] = None  # Keep as free text from the answer

    # The following textual confirmations are extracted as-is from the answer (if present)
    is_indoor_arena: Optional[str] = None
    ada_parking_confirmed: Optional[str] = None
    ada_routes_confirmed: Optional[str] = None

    # Categorized URLs (extracted exactly as present in the answer)
    location_urls: List[str] = Field(default_factory=list)
    capacity_urls: List[str] = Field(default_factory=list)
    type_urls: List[str] = Field(default_factory=list)
    accessibility_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract all distinct venues mentioned in the answer that are proposed for the 2026 summer arena tour.

    For each venue, extract the following fields exactly as written in the answer:
    - name: the venue name
    - city: the specific city
    - state: the specific state (can be full name or two-letter abbreviation, as presented in the answer)
    - seating_capacity: the seating capacity text (do not convert to a number; keep formatting as provided)
    - is_indoor_arena: the answer's explicit confirmation text that this is an indoor arena (if provided), otherwise null
    - ada_parking_confirmed: the answer's explicit confirmation text for accessible parking (if provided), otherwise null
    - ada_routes_confirmed: the answer's explicit confirmation text for accessible routes with ramps/elevators to seating (if provided), otherwise null

    Categorize the reference URLs as they are explicitly used in the answer:
    - location_urls: URLs that the answer associates with confirming the venue’s location (city/state or address)
    - capacity_urls: URLs that the answer associates with confirming the seating capacity
    - type_urls: URLs that the answer associates with confirming that it is an indoor arena (and not a stadium/amphitheater/theater)
    - accessibility_urls: URLs that the answer associates with confirming ADA features (accessible parking and/or accessible routes)
    - other_urls: any additional URLs cited for the venue that are not clearly tied to a specific category above

    IMPORTANT:
    - Extract only URLs explicitly present in the answer text (including markdown links). Do not invent any URLs.
    - If a URL category is not explicitly supported by the answer, leave that category empty and, if applicable, place the URLs into other_urls.
    - Deduplicate URLs within each category; preserve the original order of appearance.

    Return a JSON object with a single field:
    {
      "venues": [ ... array of venue objects in order of appearance ... ]
    }
    """


# --------------------------------------------------------------------------- #
# Helpers for sources and verification                                        #
# --------------------------------------------------------------------------- #
def unique_in_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items or []:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        if not (s.startswith("http://") or s.startswith("https://")):
            # Be lenient: allow the extractor's rule to add http:// if missing, else add ourselves
            if re.match(r"^[A-Za-z0-9\.\-]+(\.[A-Za-z]{2,})+.*$", s):
                s = "http://" + s
        out.append(s)
    return unique_in_order(out)


def gather_all_sources(v: VenueItem) -> List[str]:
    return unique_in_order(
        valid_urls(v.location_urls)
        + valid_urls(v.type_urls)
        + valid_urls(v.capacity_urls)
        + valid_urls(v.accessibility_urls)
        + valid_urls(v.other_urls)
    )


def select_sources(primary: List[str], fallback: List[str]) -> List[str]:
    if primary:
        return primary
    return fallback


async def verify_with_sources_or_fail(
    evaluator: Evaluator,
    node,
    claim: str,
    sources: Optional[List[str]],
    additional_instruction: str,
) -> bool:
    srcs = valid_urls(sources or [])
    if not srcs:
        # No evidence -> mark as failed directly (treat missing sources as a quality issue)
        node.score = 0.0
        node.status = "failed"
        return False
    return await evaluator.verify(
        claim=claim,
        node=node,
        sources=srcs,
        additional_instruction=additional_instruction
    )


def assign_by_region(extracted: VenuesExtraction) -> Dict[str, Optional[VenueItem]]:
    region_slot: Dict[str, Optional[VenueItem]] = {"NE": None, "SE": None, "MW": None, "W": None}
    for v in extracted.venues:
        abbr = normalize_state_abbrev(v.state)
        r = region_for_state_abbr(abbr)
        if r and region_slot[r] is None:
            region_slot[r] = v
        # Stop early if all filled
        if all(region_slot[k] is not None for k in region_slot):
            break
    return region_slot


# --------------------------------------------------------------------------- #
# Verification subroutine for a single venue                                  #
# --------------------------------------------------------------------------- #
async def verify_one_venue(
    evaluator: Evaluator,
    parent_node,
    venue: Optional[VenueItem],
    *,
    venue_node_id: str,
    venue_node_desc: str,
    prefix: str,  # e.g., "V1"
    region_desc_for_geo: str,  # The long region description from rubric
) -> None:
    # Create the top-level node for this venue (non-critical to allow partial credit across venues)
    venue_node = evaluator.add_parallel(
        id=venue_node_id,
        desc=venue_node_desc,
        parent=parent_node,
        critical=False
    )

    # Prepare convenience variables
    v = venue or VenueItem()
    name = v.name or "the venue"
    city = (v.city or "").strip()
    state_raw = (v.state or "").strip()
    state_abbr = normalize_state_abbrev(state_raw)
    all_urls = gather_all_sources(v)

    # ---------------- Geographic Location (critical) ---------------- #
    geo_node = evaluator.add_parallel(
        id=f"{prefix}_Geographic_Location",
        desc=f"Venue is located in the {region_desc_for_geo}",
        parent=venue_node,
        critical=True
    )

    # State leaf - verify from sources that the venue is located in the stated state
    state_leaf = evaluator.add_leaf(
        id=f"{prefix}_Location_State",
        desc=f"Specific state within the {region_desc_for_geo.split('(')[0].strip()} is identified",
        parent=geo_node,
        critical=True
    )
    # Prefer location URLs, fallback to any
    state_sources = select_sources(valid_urls(v.location_urls), all_urls)
    if not state_raw or not state_sources:
        state_leaf.score = 0.0
        state_leaf.status = "failed"
    else:
        claim_state = f"The venue '{name}' is located in the U.S. state of {state_raw}."
        await evaluator.verify(
            claim=claim_state,
            node=state_leaf,
            sources=state_sources,
            additional_instruction="Use the provided source(s) to confirm the venue's state. Accept either state full name or standard two-letter abbreviation."
        )

    # City leaf - verify from sources that the venue is located in the stated city (and state, if available)
    city_leaf = evaluator.add_leaf(
        id=f"{prefix}_Location_City",
        desc="Specific city within the identified state is provided",
        parent=geo_node,
        critical=True
    )
    if not city or not state_sources:
        city_leaf.score = 0.0
        city_leaf.status = "failed"
    else:
        if state_raw:
            city_claim = f"The venue '{name}' is located in {city}, {state_raw}."
        else:
            city_claim = f"The venue '{name}' is located in {city}."
        await evaluator.verify(
            claim=city_claim,
            node=city_leaf,
            sources=state_sources,
            additional_instruction="Verify that the page shows the venue is in the specified city (and state if provided), e.g., by address or location description."
        )

    # Location reference - ensure a dedicated location URL is provided by the answer
    location_ref_node = evaluator.add_custom_node(
        result=bool(valid_urls(v.location_urls)),
        id=f"{prefix}_Location_Reference",
        desc="Source URL confirming the venue's geographic location is provided",
        parent=geo_node,
        critical=True
    )

    # ---------------- Capacity Requirements (critical) ---------------- #
    cap_node = evaluator.add_parallel(
        id=f"{prefix}_Capacity_Requirements",
        desc="Venue meets capacity specifications for indoor arena concerts",
        parent=venue_node,
        critical=True
    )

    cap_sources_pref = valid_urls(v.capacity_urls)
    cap_sources = select_sources(cap_sources_pref, all_urls)

    # Minimum capacity >= 5,000
    min_cap_leaf = evaluator.add_leaf(
        id=f"{prefix}_Minimum_Capacity",
        desc="Venue has a seating capacity of at least 5,000 people",
        parent=cap_node,
        critical=True
    )
    if not cap_sources:
        min_cap_leaf.score = 0.0
        min_cap_leaf.status = "failed"
    else:
        min_claim = f"The seating capacity of '{name}' is at least 5,000."
        await evaluator.verify(
            claim=min_claim,
            node=min_cap_leaf,
            sources=cap_sources,
            additional_instruction="Check the listed seating capacity on the source page(s). If multiple configurations exist, consider the maximum typical event seating. Threshold is inclusive (≥ 5,000)."
        )

    # Maximum capacity <= 25,000
    max_cap_leaf = evaluator.add_leaf(
        id=f"{prefix}_Maximum_Capacity",
        desc="Venue has a seating capacity not exceeding 25,000 people",
        parent=cap_node,
        critical=True
    )
    if not cap_sources:
        max_cap_leaf.score = 0.0
        max_cap_leaf.status = "failed"
    else:
        max_claim = f"The seating capacity of '{name}' does not exceed 25,000."
        await evaluator.verify(
            claim=max_claim,
            node=max_cap_leaf,
            sources=cap_sources,
            additional_instruction="Check the listed seating capacity on the source page(s). If multiple capacities are shown, use the largest typical event capacity. Threshold is inclusive (≤ 25,000)."
        )

    # Capacity reference existence
    cap_ref_node = evaluator.add_custom_node(
        result=bool(valid_urls(v.capacity_urls)),
        id=f"{prefix}_Capacity_Reference",
        desc="Source URL confirming the venue's seating capacity is provided",
        parent=cap_node,
        critical=True
    )

    # ---------------- Venue Type (critical) ---------------- #
    type_node = evaluator.add_parallel(
        id=f"{prefix}_Venue_Type",
        desc="Venue is classified as an indoor arena suitable for concert production",
        parent=venue_node,
        critical=True
    )

    type_sources_pref = valid_urls(v.type_urls)
    type_sources = select_sources(type_sources_pref, all_urls)

    indoor_leaf = evaluator.add_leaf(
        id=f"{prefix}_Indoor_Classification",
        desc="Venue is confirmed as an enclosed indoor facility",
        parent=type_node,
        critical=True
    )
    if not type_sources:
        indoor_leaf.score = 0.0
        indoor_leaf.status = "failed"
    else:
        indoor_claim = f"The venue '{name}' is an indoor (enclosed) facility."
        await evaluator.verify(
            claim=indoor_claim,
            node=indoor_leaf,
            sources=type_sources,
            additional_instruction="Confirm that the venue is indoors (not open-air). Pages that describe the venue as an 'indoor arena' or similar suffice."
        )

    arena_leaf = evaluator.add_leaf(
        id=f"{prefix}_Arena_Designation",
        desc="Venue is identified as an arena (not a stadium, amphitheater, or theater)",
        parent=type_node,
        critical=True
    )
    if not type_sources:
        arena_leaf.score = 0.0
        arena_leaf.status = "failed"
    else:
        arena_claim = f"The venue '{name}' is an arena (not a stadium, amphitheater, or theater)."
        await evaluator.verify(
            claim=arena_claim,
            node=arena_leaf,
            sources=type_sources,
            additional_instruction="Confirm the venue is described as an 'arena' (e.g., arena, coliseum). If the page calls it a stadium, amphitheater, or theater, this should fail."
        )

    type_ref_node = evaluator.add_custom_node(
        result=bool(valid_urls(v.type_urls)),
        id=f"{prefix}_Type_Reference",
        desc="Source URL confirming the venue type classification is provided",
        parent=type_node,
        critical=True
    )

    # ---------------- Accessibility Compliance (critical) ---------------- #
    acc_node = evaluator.add_parallel(
        id=f"{prefix}_Accessibility_Compliance",
        desc="Venue meets ADA accessibility standards for public venues",
        parent=venue_node,
        critical=True
    )

    acc_sources_pref = valid_urls(v.accessibility_urls)
    acc_sources = select_sources(acc_sources_pref, all_urls)

    parking_leaf = evaluator.add_leaf(
        id=f"{prefix}_Accessible_Parking",
        desc="Venue provides accessible parking spaces as required by ADA",
        parent=acc_node,
        critical=True
    )
    if not acc_sources:
        parking_leaf.score = 0.0
        parking_leaf.status = "failed"
    else:
        park_claim = f"The venue '{name}' provides ADA-compliant accessible parking spaces."
        await evaluator.verify(
            claim=park_claim,
            node=parking_leaf,
            sources=acc_sources,
            additional_instruction="Look for official accessibility info, parking maps, or policies that explicitly mention accessible/ADA parking."
        )

    routes_leaf = evaluator.add_leaf(
        id=f"{prefix}_Accessible_Routes",
        desc="Venue has accessible routes including ramps or elevators to all seating areas",
        parent=acc_node,
        critical=True
    )
    if not acc_sources:
        routes_leaf.score = 0.0
        routes_leaf.status = "failed"
    else:
        routes_claim = f"The venue '{name}' has accessible routes to seating areas, including ramps or elevators."
        await evaluator.verify(
            claim=routes_claim,
            node=routes_leaf,
            sources=acc_sources,
            additional_instruction="Verify that accessible routes from parking/entrances to seating exist, including ramps or elevators per ADA guidance."
        )

    acc_ref_node = evaluator.add_custom_node(
        result=bool(valid_urls(v.accessibility_urls)),
        id=f"{prefix}_Accessibility_Reference",
        desc="Source URL confirming venue accessibility features is provided",
        parent=acc_node,
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for identifying four suitable indoor arena venues across US regions
    with capacity and accessibility requirements.
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
        default_model=model,
    )

    # Extract structured venue info from the answer
    extracted: VenuesExtraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extracted"
    )

    # Assign one venue per required region based on the state
    assigned = assign_by_region(extracted)
    # Record assignment decision for transparency
    evaluator.add_custom_info(
        info={
            "assigned_regions": {
                "NE": (assigned["NE"].dict() if assigned["NE"] else None),
                "SE": (assigned["SE"].dict() if assigned["SE"] else None),
                "MW": (assigned["MW"].dict() if assigned["MW"] else None),
                "W": (assigned["W"].dict() if assigned["W"] else None),
            }
        },
        info_type="region_assignment",
        info_name="region_assignment"
    )

    # Build the overall node (make it non-critical to allow partial credit across regions)
    overall = evaluator.add_parallel(
        id="Summer_2026_Tour_Venues",
        desc="Identification of four suitable concert venues across different US regions for a summer 2026 arena tour, each meeting specific capacity, accessibility, technical, and regulatory requirements",
        parent=root,
        critical=False
    )

    # Per-region verification using the prescribed Venue_1..Venue_4 nodes
    # Venue_1 -> Northeast
    await verify_one_venue(
        evaluator=evaluator,
        parent_node=overall,
        venue=assigned.get("NE"),
        venue_node_id="Venue_1",
        venue_node_desc="First venue meeting all specified requirements",
        prefix="V1",
        region_desc_for_geo="Northeastern United States (states including: ME, NH, VT, MA, RI, CT, NY, NJ, PA)",
    )

    # Venue_2 -> Southeast
    await verify_one_venue(
        evaluator=evaluator,
        parent_node=overall,
        venue=assigned.get("SE"),
        venue_node_id="Venue_2",
        venue_node_desc="Second venue meeting all specified requirements",
        prefix="V2",
        region_desc_for_geo="Southeastern United States (states including: FL, GA, SC, NC, VA, WV, KY, TN, AL, MS, AR, LA)",
    )

    # Venue_3 -> Midwest
    await verify_one_venue(
        evaluator=evaluator,
        parent_node=overall,
        venue=assigned.get("MW"),
        venue_node_id="Venue_3",
        venue_node_desc="Third venue meeting all specified requirements",
        prefix="V3",
        region_desc_for_geo="Midwestern United States (states including: OH, IN, IL, MI, WI, MN, IA, MO, ND, SD, NE, KS)",
    )

    # Venue_4 -> West
    await verify_one_venue(
        evaluator=evaluator,
        parent_node=overall,
        venue=assigned.get("W"),
        venue_node_id="Venue_4",
        venue_node_desc="Fourth venue meeting all specified requirements",
        prefix="V4",
        region_desc_for_geo="Western United States (states including: CA, OR, WA, NV, AZ, UT, CO, NM, WY, MT, ID)",
    )

    # Return the structured summary
    return evaluator.get_summary()