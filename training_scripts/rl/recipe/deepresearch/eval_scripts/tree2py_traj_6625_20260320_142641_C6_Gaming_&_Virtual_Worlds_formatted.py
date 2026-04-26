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
TASK_ID = "esports_venues_us"
TASK_DESCRIPTION = """I am organizing a regional esports tournament and need to identify suitable venues for hosting the event. Please find three dedicated esports facilities in the United States that meet all of the following requirements:

1. The venue must be a dedicated esports facility or gaming venue specifically equipped for esports events (not a general convention center)
2. The venue must be located in the United States
3. The venue must have a seating capacity of at least 500 spectators
4. The facility must have a total size of at least 10,000 square feet
5. The venue must have high-speed internet connectivity suitable for gaming and streaming
6. The venue must have on-site or nearby parking facilities available

For each of the three venues, provide:
- The official name of the venue
- The complete street address including city and state
- The seating capacity (number of spectators)
- The total facility size in square feet
- Confirmation of high-speed internet availability
- Confirmation of parking availability
- Reference URLs from official venue websites or authoritative sources that verify each piece of information
"""


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool((s or "").strip())


def _first_non_empty_list(*lists: List[str]) -> List[str]:
    for lst in lists:
        if lst and len(lst) > 0:
            return lst
    return []


def _parse_number_from_text(text: Optional[str]) -> Optional[int]:
    """
    Extract a representative integer from a messy numeric string.
    Handles:
      - comma/period separated numbers like '12,500' or '12.5'
      - 'k' suffix like '12k' -> 12000
      - multiple numbers -> choose the maximum as the representative capacity/size
    """
    if not text:
        return None

    t = text.lower().strip()

    # Handle k-suffix numbers (e.g., '10k', '12.5k')
    k_matches = re.findall(r'(\d+(?:\.\d+)?)\s*[kＫ]', t)
    k_values = [float(m) * 1000 for m in k_matches]

    # Plain numbers with commas/decimals
    num_matches = re.findall(r'\d[\d,\.]*', t)
    values: List[float] = []
    for m in num_matches:
        try:
            # Remove commas
            v = float(m.replace(",", ""))
            values.append(v)
        except Exception:
            continue

    all_vals = k_values + values
    if not all_vals:
        return None
    return int(max(all_vals))


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    # Basic
    name: Optional[str] = None
    venue_type: Optional[str] = None  # e.g., "esports arena", "gaming venue"
    # Address
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    # Specs
    seating_capacity: Optional[str] = None  # keep as string; we'll parse numbers
    facility_size_sqft: Optional[str] = None  # keep as string; we'll parse numbers
    # Amenities
    high_speed_internet: Optional[str] = None  # textual confirmation (e.g., "10Gbps fiber")
    parking_available: Optional[str] = None    # textual confirmation (e.g., "on-site parking garage")
    # References (per-attribute)
    basic_urls: List[str] = Field(default_factory=list)     # name/type/overview
    address_urls: List[str] = Field(default_factory=list)   # address verification
    capacity_urls: List[str] = Field(default_factory=list)  # capacity verification
    size_urls: List[str] = Field(default_factory=list)      # facility size verification
    technical_urls: List[str] = Field(default_factory=list) # internet / tech verification
    parking_urls: List[str] = Field(default_factory=list)   # parking verification


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract up to three esports venues mentioned in the answer. For each venue, return the following fields:

- name: Official venue name (string)
- venue_type: How the venue is described (e.g., "esports arena", "gaming venue") (string; optional)
- street_address: Street address (string; do not include city/state here)
- city: City (string)
- state: State (string)
- seating_capacity: Seating capacity as written in the answer (string; do NOT convert to a number)
- facility_size_sqft: Total facility size as written in the answer (string; do NOT convert to a number)
- high_speed_internet: Confirmation text about high-speed internet availability (string; e.g., "10 Gbps fiber", "dedicated gigabit", "yes")
- parking_available: Confirmation text about parking availability (string; e.g., "on-site garage", "nearby lot", "validated parking")
- basic_urls: URLs (list) that substantiate the venue's identity/type (prefer official venue pages or authoritative sources)
- address_urls: URLs (list) that substantiate the full address
- capacity_urls: URLs (list) that substantiate the seating capacity
- size_urls: URLs (list) that substantiate the total facility size in square feet
- technical_urls: URLs (list) that substantiate high-speed internet/network capabilities (gaming/streaming ready)
- parking_urls: URLs (list) that substantiate on-site or nearby parking availability

Rules:
- Extract ONLY what is explicitly present in the provided answer.
- For any field that is missing in the answer, set it to null (for strings) or [] (for URL lists).
- For URL lists, include only valid, explicit URLs that appear in the answer (plain or markdown). Do not invent URLs.
- Do not merge city/state into street_address; keep them separate.
- Do not normalize or compute numbers; keep what appears in the answer text.

Return as:
{
  "venues": [ { ... up to 3 venues ... } ]
}
"""


# --------------------------------------------------------------------------- #
# Verification builder for one venue                                          #
# --------------------------------------------------------------------------- #
async def verify_one_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    index_1based: int,
) -> None:
    """
    Build verification sub-tree and run verifications for a single venue.
    Follows the rubric structure exactly with criticality and node IDs.
    """
    vn_id = f"venue_{index_1based}"
    venue_node = evaluator.add_parallel(
        id=vn_id,
        desc=f"{['First','Second','Third'][index_1based-1]} qualifying esports venue meeting all requirements",
        parent=parent_node,
        critical=False  # Non-critical to allow partial credit across venues
    )

    # -------------------- Basic info (critical group) -------------------- #
    basic_node = evaluator.add_parallel(
        id=f"{vn_id}_basic_info",
        desc="Basic venue identification and type verification",
        parent=venue_node,
        critical=True
    )

    # Name provided (existence)
    evaluator.add_custom_node(
        result=_non_empty(venue.name),
        id=f"{vn_id}_name",
        desc="Venue name is provided",
        parent=basic_node,
        critical=True
    )

    # Basic reference present (existence)
    evaluator.add_custom_node(
        result=len(venue.basic_urls) > 0,
        id=f"{vn_id}_basic_reference",
        desc="Reference URL for venue's basic information is provided from official or authoritative source",
        parent=basic_node,
        critical=True
    )

    # Venue type: dedicated esports facility (verify with URLs)
    type_leaf = evaluator.add_leaf(
        id=f"{vn_id}_type",
        desc="Venue is identified as a dedicated esports facility or gaming venue (not a general convention center)",
        parent=basic_node,
        critical=True
    )

    type_claim = (
        f"The venue '{venue.name or 'the venue'}' is a dedicated esports facility or gaming venue specifically "
        f"equipped for esports events, not a general-purpose convention center."
    )
    type_sources = _first_non_empty_list(venue.basic_urls, venue.capacity_urls, venue.size_urls)

    # US Location (verify with URLs)
    us_loc_leaf = evaluator.add_leaf(
        id=f"{vn_id}_us_location",
        desc="Venue is confirmed to be located in the United States",
        parent=basic_node,
        critical=True
    )
    us_loc_claim = (
        f"The venue '{venue.name or 'the venue'}' is located in the United States."
    )
    us_sources = _first_non_empty_list(venue.address_urls, venue.basic_urls)

    # -------------------- Address (critical group) ----------------------- #
    address_node = evaluator.add_parallel(
        id=f"{vn_id}_address",
        desc="Complete physical address information",
        parent=venue_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(venue.street_address),
        id=f"{vn_id}_street_address",
        desc="Street address is provided",
        parent=address_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(venue.city) and _non_empty(venue.state),
        id=f"{vn_id}_city_state",
        desc="City and state are provided",
        parent=address_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(venue.address_urls) > 0,
        id=f"{vn_id}_address_reference",
        desc="Reference URL for address information is provided",
        parent=address_node,
        critical=True
    )

    # -------------------- Capacity (critical group) ---------------------- #
    capacity_node = evaluator.add_parallel(
        id=f"{vn_id}_capacity",
        desc="Seating capacity specifications",
        parent=venue_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(venue.seating_capacity),
        id=f"{vn_id}_capacity_specified",
        desc="Seating capacity number is provided",
        parent=capacity_node,
        critical=True
    )

    cap_val = _parse_number_from_text(venue.seating_capacity)
    evaluator.add_custom_node(
        result=(cap_val is not None and cap_val >= 500),
        id=f"{vn_id}_capacity_minimum",
        desc="Seating capacity meets or exceeds 500 spectators",
        parent=capacity_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(venue.capacity_urls) > 0,
        id=f"{vn_id}_capacity_reference",
        desc="Reference URL for capacity information is provided",
        parent=capacity_node,
        critical=True
    )

    # -------------------- Facility size (critical group) ----------------- #
    size_node = evaluator.add_parallel(
        id=f"{vn_id}_facility_size",
        desc="Total facility size specifications",
        parent=venue_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty(venue.facility_size_sqft),
        id=f"{vn_id}_size_specified",
        desc="Total square footage is provided",
        parent=size_node,
        critical=True
    )

    size_val = _parse_number_from_text(venue.facility_size_sqft)
    evaluator.add_custom_node(
        result=(size_val is not None and size_val >= 10000),
        id=f"{vn_id}_size_minimum",
        desc="Facility size meets or exceeds 10,000 square feet",
        parent=size_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(venue.size_urls) > 0,
        id=f"{vn_id}_size_reference",
        desc="Reference URL for facility size information is provided",
        parent=size_node,
        critical=True
    )

    # -------------------- Technical (critical group) --------------------- #
    technical_node = evaluator.add_parallel(
        id=f"{vn_id}_technical",
        desc="Technical infrastructure capabilities",
        parent=venue_node,
        critical=True
    )

    internet_leaf = evaluator.add_leaf(
        id=f"{vn_id}_internet",
        desc="High-speed internet connectivity for gaming and streaming is confirmed available",
        parent=technical_node,
        critical=True
    )
    internet_claim = (
        f"The venue '{venue.name or 'the venue'}' offers high-speed internet connectivity suitable for competitive "
        f"gaming and live streaming (e.g., gigabit fiber, enterprise-grade networking)."
    )
    internet_sources = _first_non_empty_list(venue.technical_urls, venue.basic_urls)

    evaluator.add_custom_node(
        result=len(venue.technical_urls) > 0,
        id=f"{vn_id}_technical_reference",
        desc="Reference URL for technical specifications is provided",
        parent=technical_node,
        critical=True
    )

    # -------------------- Parking (critical group) ----------------------- #
    parking_node = evaluator.add_parallel(
        id=f"{vn_id}_parking",
        desc="Parking facility availability",
        parent=venue_node,
        critical=True
    )

    parking_leaf = evaluator.add_leaf(
        id=f"{vn_id}_parking_available",
        desc="On-site or nearby parking is confirmed available",
        parent=parking_node,
        critical=True
    )
    parking_claim = (
        f"On-site or nearby parking is available for '{venue.name or 'the venue'}' (e.g., on-site lot/garage, "
        f"adjacent public parking, or validated nearby parking)."
    )
    parking_sources = _first_non_empty_list(venue.parking_urls, venue.basic_urls)

    evaluator.add_custom_node(
        result=len(venue.parking_urls) > 0,
        id=f"{vn_id}_parking_reference",
        desc="Reference URL for parking information is provided",
        parent=parking_node,
        critical=True
    )

    # -------------------- Execute URL-based verifications ---------------- #
    verify_jobs: List[Tuple[str, List[str], Any, Optional[str]]] = []

    verify_jobs.append((
        type_claim,
        type_sources,
        type_leaf,
        "Confirm the venue is a specialized esports/gaming facility (e.g., 'esports arena', 'gaming arena', 'LAN center'). "
        "Do NOT accept general-purpose convention centers or venues without explicit esports/gaming positioning."
    ))

    verify_jobs.append((
        us_loc_claim,
        us_sources,
        us_loc_leaf,
        "The page should indicate a U.S. location explicitly or implicitly (e.g., city+state like 'Dallas, TX', a US ZIP code, "
        "or 'United States' wording). If a U.S. state abbreviation or state name appears with the city, treat it as U.S."
    ))

    verify_jobs.append((
        internet_claim,
        internet_sources,
        internet_leaf,
        "Look for evidence like 'fiber', 'Gbps', 'dedicated bandwidth', 'low latency LAN', or references to live streaming, "
        "broadcast/production networking. The page should clearly imply suitability for competitive gaming/streaming."
    ))

    verify_jobs.append((
        parking_claim,
        parking_sources,
        parking_leaf,
        "Evidence can include 'on-site parking', 'parking garage', 'parking lot', 'nearby parking', 'validated parking', "
        "or a parking section/policy indicating availability."
    ))

    # Run verifications in parallel (where applicable)
    await evaluator.batch_verify(verify_jobs)


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
    Evaluate an answer for identifying three qualifying esports venues in the U.S.
    """
    # Initialize evaluator (root node is always non-critical per framework)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify three dedicated esports facilities in the United States that meet all specified venue requirements",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured venues from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Add custom info about the evaluation requirements
    evaluator.add_custom_info(
        info={
            "requirements": {
                "dedicated_esports_facility": True,
                "us_location": True,
                "min_seating_capacity": 500,
                "min_facility_size_sqft": 10000,
                "high_speed_internet": True,
                "parking_available": True
            },
            "max_venues_evaluated": 3,
            "extracted_venues_count": len(extracted.venues)
        },
        info_type="task_requirements",
        info_name="requirements_overview"
    )

    # Select up to three venues (pad with empty records if fewer)
    venues: List[VenueItem] = list(extracted.venues[:3])
    while len(venues) < 3:
        venues.append(VenueItem())

    # Build verification subtrees for each venue
    for idx, venue in enumerate(venues, start=1):
        await verify_one_venue(evaluator, root, venue, idx)

    # Return summary
    return evaluator.get_summary()