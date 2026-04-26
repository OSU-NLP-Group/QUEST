import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_indoor_arenas_2026"
TASK_DESCRIPTION = """A major recording artist is planning a 2026 concert tour and needs to identify suitable indoor arena venues in California. Identify two (2) indoor arenas in California that meet ALL of the following requirements:

1. Located in California, USA
2. Indoor arena venue (not an outdoor venue)
3. Concert seating capacity between 15,000 and 20,000
4. Operational and available for bookings in 2026
5. Provides ADA-compliant wheelchair-accessible seating
6. Capable of supporting large-scale concert production with stage dimensions of at least 60 feet wide
7. Features loading dock access for 53-foot semi-truck trailers

For each venue, provide:
- The venue's name
- Its location (city, California)
- Its concert seating capacity
- A reference URL that supports the venue's specifications
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Expect "California" or "CA"
    capacity: Optional[str] = None  # Keep as string to be flexible (e.g., "18,000 for concerts")
    stage_width_feet: Optional[str] = None  # If the answer explicitly mentions stage width
    reference_urls: List[str] = Field(default_factory=list)  # URLs cited for this venue


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to the first 5 candidate venues mentioned in the answer that are intended for the 2026 concert tour.
    For each venue, extract the following fields from the answer exactly as written:
    - name: The venue's name
    - city: The city name (do NOT include the state here)
    - state: The state (e.g., "California" or "CA") if provided
    - capacity: The concert seating capacity as a text snippet (e.g., "18,200 for concerts")
    - stage_width_feet: The stage width mentioned (e.g., "60 ft", "18m") if explicitly stated; otherwise null
    - reference_urls: The URL(s) cited for this venue in the answer. Only include actual URLs (plain or markdown).
    
    Rules:
    - Extract only what is explicitly present in the answer text. Do not invent values.
    - If any field is missing for a venue, set it to null (or [] for reference_urls).
    - For state, if it is omitted in the answer, set it to null. Do not infer "California" unless the answer explicitly contains it.
    - For reference_urls, include all URLs cited for the venue (production specs, ADA info, booking page, official site, etc.).
    - Return a JSON with a top-level "venues" array of venue objects in the order they appear.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _norm_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return " ".join(name.strip().lower().split())


def select_first_n_unique(venues: List[VenueItem], n: int) -> List[VenueItem]:
    seen = set()
    selected: List[VenueItem] = []
    for v in venues:
        key = _norm_name(v.name)
        if not key:
            continue
        if key not in seen:
            seen.add(key)
            selected.append(v)
        if len(selected) >= n:
            break
    return selected


def ensure_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Basic cleanup: strip whitespace, drop empties
    clean = []
    for u in urls:
        if isinstance(u, str):
            uu = u.strip()
            if uu:
                clean.append(uu)
    return clean


# --------------------------------------------------------------------------- #
# Venue verification                                                          #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    venue_index: int,
) -> None:
    """
    Build verification nodes for a single venue (Venue1 or Venue2).
    All children under this node are critical. If any requirement fails, the venue fails.
    """
    # Create parent group node for this venue (parallel, critical)
    venue_node = evaluator.add_parallel(
        id=f"Venue{venue_index+1}",
        desc=f"{'First' if venue_index == 0 else 'Second'} venue meets all requirements and includes required fields.",
        parent=parent_node,
        critical=True,
    )

    # Basic existence / provided checks (custom immediate results)
    name_provided = evaluator.add_custom_node(
        result=bool(venue and venue.name and venue.name.strip()),
        id=f"Venue{venue_index+1}_NameProvided",
        desc="Venue name is provided.",
        parent=venue_node,
        critical=True,
    )

    # Location provided as (city, California). Require city and state present and state indicates California/CA.
    state_val = (venue.state or "").strip()
    city_val = (venue.city or "").strip()
    is_ca_state = state_val.lower() in {"california", "ca"}
    location_provided_ok = bool(city_val and is_ca_state)
    evaluator.add_custom_node(
        result=location_provided_ok,
        id=f"Venue{venue_index+1}_LocationProvided",
        desc="Venue location is provided as (city, California).",
        parent=venue_node,
        critical=True,
    )

    # Capacity provided
    evaluator.add_custom_node(
        result=bool(venue and venue.capacity and venue.capacity.strip()),
        id=f"Venue{venue_index+1}_CapacityProvided",
        desc="Concert seating capacity value is provided.",
        parent=venue_node,
        critical=True,
    )

    # At least one reference URL provided
    src_urls = ensure_urls(venue.reference_urls)
    evaluator.add_custom_node(
        result=len(src_urls) > 0,
        id=f"Venue{venue_index+1}_ReferenceURLProvided",
        desc="At least one reference URL is provided.",
        parent=venue_node,
        critical=True,
    )

    # Now evidence-backed checks (verified via URLs)
    # 1) Located in California, USA
    loc_leaf = evaluator.add_leaf(
        id=f"Venue{venue_index+1}_LocatedInCaliforniaUSA",
        desc="Venue is located in California, USA.",
        parent=venue_node,
        critical=True,
    )
    if city_val:
        loc_claim = f"The venue '{venue.name}' is located in {city_val}, California, USA."
    else:
        loc_claim = f"The venue '{venue.name}' is located in California, USA."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=src_urls,
        additional_instruction="Accept 'CA' as an abbreviation for California. Verify the location from the provided sources.",
    )

    # 2) Indoor arena (not outdoor)
    indoor_leaf = evaluator.add_leaf(
        id=f"Venue{venue_index+1}_IndoorArena",
        desc="Venue is an indoor arena (not an outdoor amphitheater or stadium).",
        parent=venue_node,
        critical=True,
    )
    indoor_claim = f"The venue '{venue.name}' is an indoor arena (enclosed, roofed), not an outdoor amphitheater or open-air stadium."
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_leaf,
        sources=src_urls,
        additional_instruction="Look for terms such as 'indoor arena', 'arena', or enclosed venue. If sources clearly indicate outdoor-only spaces, fail.",
    )

    # 3) Capacity is between 15,000 and 20,000 (inclusive)
    cap_leaf = evaluator.add_leaf(
        id=f"Venue{venue_index+1}_CapacityInRange",
        desc="Concert seating capacity is between 15,000 and 20,000.",
        parent=venue_node,
        critical=True,
    )
    cap_str = (venue.capacity or "").strip()
    cap_claim = (
        f"The venue '{venue.name}' has a concert seating capacity between 15,000 and 20,000 (inclusive). "
        f"{'The answer states: ' + cap_str if cap_str else ''}"
    )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=src_urls,
        additional_instruction=(
            "Use the highest or typical 'concert' capacity if multiple configurations exist; "
            "basketball/hockey capacities can be used as a proxy if concert capacity is unspecified. "
            "If the maximum listed capacity exceeds 20,000 or is below 15,000, this fails. Minor rounding is acceptable."
        ),
    )

    # 4) Operational and available for bookings in 2026
    op_leaf = evaluator.add_leaf(
        id=f"Venue{venue_index+1}_Operational2026",
        desc="Venue is operational and available for concert bookings in 2026.",
        parent=venue_node,
        critical=True,
    )
    op_claim = (
        f"As of 2026, the venue '{venue.name}' is operational and available for concert bookings "
        f"(e.g., has an event calendar with 2026 events or active booking/contact info)."
    )
    await evaluator.verify(
        claim=op_claim,
        node=op_leaf,
        sources=src_urls,
        additional_instruction=(
            "Consider official pages showing current operations, event calendars listing 2026 shows, "
            "or active 'Book an Event' pages as evidence. If only historical/archived info exists, do not pass."
        ),
    )

    # 5) ADA-compliant wheelchair-accessible seating
    ada_leaf = evaluator.add_leaf(
        id=f"Venue{venue_index+1}_ADA",
        desc="Venue provides ADA-compliant wheelchair-accessible seating.",
        parent=venue_node,
        critical=True,
    )
    ada_claim = f"The venue '{venue.name}' provides ADA-compliant wheelchair-accessible seating."
    await evaluator.verify(
        claim=ada_claim,
        node=ada_leaf,
        sources=src_urls,
        additional_instruction="Look for 'ADA', 'accessibility', 'wheelchair accessible seating', or similar policy pages. Official venue pages are preferred.",
    )

    # 6) Stage width at least 60 feet
    stage_leaf = evaluator.add_leaf(
        id=f"Venue{venue_index+1}_StageSpecs",
        desc="Venue can support stage dimensions of at least 60 feet wide.",
        parent=venue_node,
        critical=True,
    )
    stage_claim = (
        f"The venue '{venue.name}' can support a stage width of at least 60 feet "
        f"(e.g., technical specifications, production guide, or event planner docs)."
    )
    await evaluator.verify(
        claim=stage_claim,
        node=stage_leaf,
        sources=src_urls,
        additional_instruction=(
            "Look for stage width, proscenium width, or clear width values. If dimensions are in meters, convert approximately "
            "(e.g., 18 m ≈ 59 ft; 18.3 m ≈ 60 ft). Pass if width ≥ 60 ft or ≈ 18.3 m."
        ),
    )

    # 7) Loading docks for 53-foot semi-truck trailers
    dock_leaf = evaluator.add_leaf(
        id=f"Venue{venue_index+1}_LoadingDock",
        desc="Venue features loading dock access for 53-foot semi-truck trailers.",
        parent=venue_node,
        critical=True,
    )
    dock_claim = (
        f"The venue '{venue.name}' has loading dock access suitable for 53-foot (53') semi-truck trailers."
    )
    await evaluator.verify(
        claim=dock_claim,
        node=dock_leaf,
        sources=src_urls,
        additional_instruction="Search for loading dock specs mentioning '53 ft', '53’ trucks', 'semi-trailer', or truck bay details.",
    )

    # 8) References support the specs used
    ref_support_leaf = evaluator.add_leaf(
        id=f"Venue{venue_index+1}_ReferenceSupportsSpecs",
        desc="Provided reference URL(s) support the venue specifications used to justify compliance with the constraints.",
        parent=venue_node,
        critical=True,
    )
    ref_support_claim = (
        f"The provided references for '{venue.name}' include at least one credible/official page (e.g., production guide, "
        f"technical specs, ADA/accessibility, event planner, or booking info) that supports one or more of the required venue specs."
    )
    await evaluator.verify(
        claim=ref_support_claim,
        node=ref_support_leaf,
        sources=src_urls,
        additional_instruction="Support is satisfied if any provided URL contains explicit specs or policies (capacity, ADA, stage/production, loading docks, booking).",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 'CA indoor arenas for 2026 tour' task.
    """
    # Initialize evaluator (use sequential root to align with gating flow)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract venue candidates from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Add helpful custom info for transparency
    total_venues = len(extraction.venues) if extraction and extraction.venues else 0
    unique_names = []
    seen = set()
    for v in extraction.venues:
        key = _norm_name(v.name)
        if key and key not in seen:
            seen.add(key)
            unique_names.append(v.name)
    evaluator.add_custom_info(
        info={
            "total_venues_extracted": total_venues,
            "unique_venue_names": unique_names,
            "note": "If more than two are provided, the first two distinct venues are evaluated."
        },
        info_type="extraction_summary",
    )

    # Build top-level task node (critical sequential)
    task_node = evaluator.add_sequential(
        id="VenueSelectionTask",
        desc="Evaluate whether the submission identifies exactly two distinct indoor arena venues in California that meet all specified requirements for a 2026 concert tour and provides required fields with supporting references.",
        parent=root,
        critical=True,
    )

    # Venue count check (critical)
    # We pass this check if there are at least two distinct venues; we will evaluate only the first two.
    unique_count = len(unique_names)
    evaluator.add_custom_node(
        result=unique_count >= 2,
        id="VenueCountCheck",
        desc="The submission provides exactly two distinct venues (no duplicates).",
        parent=task_node,
        critical=True,
    )

    # Parallel evaluation of the two venues (critical)
    venues_eval_node = evaluator.add_parallel(
        id="VenuesEvaluation",
        desc="Each of the two venues meets all constraints and includes the required reported fields and reference URL(s).",
        parent=task_node,
        critical=True,
    )

    # Select the first two distinct venues
    first_two = select_first_n_unique(extraction.venues if extraction and extraction.venues else [], 2)

    # Pad with empty items if fewer than 2 to ensure tree structure is consistent
    while len(first_two) < 2:
        first_two.append(VenueItem())

    # Verify each venue
    for idx in range(2):
        await verify_single_venue(evaluator, venues_eval_node, first_two[idx], idx)

    # Return structured summary
    return evaluator.get_summary()