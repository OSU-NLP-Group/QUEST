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
TASK_ID = "us_esports_venues_4"
TASK_DESCRIPTION = """
Identify four dedicated esports venues in the United States that could host a regional gaming tournament series. Each venue must meet the following requirements: (1) Be classified as an 'Esports' type venue (not a multipurpose arena), (2) Have a minimum seating capacity of 500 people, (3) Be located in a different U.S. state (no two venues from the same state), (4) Be currently operational with a confirmed opening year. For each venue, provide: the complete official venue name, city, state, seating capacity, and a reference URL that verifies this information.
"""


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
US_STATE_MAP = {
    # States
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
    # Districts/Territories (if present in answers, we'll canonicalize DC only)
    "DISTRICT OF COLUMBIA": "DC", "WASHINGTON DC": "DC", "WASHINGTON, DC": "DC", "DC": "DC"
}
US_STATE_CODES = set(US_STATE_MAP.values())


def canonicalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip().upper().replace(".", "").replace(",", "")
    if len(s) == 2 and s in US_STATE_CODES:
        return s
    return US_STATE_MAP.get(s, None)


def is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip()
    return (u.startswith("http://") or u.startswith("https://")) and "." in u and " " not in u


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None
    opening_year: Optional[str] = None
    type_label: Optional[str] = None
    operational_status: Optional[str] = None
    reference_url: Optional[str] = None


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to 10 esports venues in the United States mentioned in the answer. For each venue, return the following fields exactly as presented in the answer:
    - name: Complete official venue name (string)
    - city: City where the venue is located (string)
    - state: U.S. state (two-letter abbreviation preferred, or full name if provided)
    - capacity: Seating capacity as stated (string; do not convert to number; include qualifiers like "seats", "up to", etc.)
    - opening_year: The year the venue opened (string; if a range is present, extract the year stated for opening)
    - type_label: How the answer classifies the venue (e.g., "Esports arena", "Esports facility", "Gaming arena", etc.)
    - operational_status: Operational status mentioned (e.g., "operational", "open", "closed", "suspended", etc.)
    - reference_url: A single URL provided in the answer that best verifies the venue's details; if multiple URLs are provided, pick the most authoritative or most directly relevant one
    
    Only extract venues that are in the United States as per the answer text. If a field is missing, set it to null. Do not invent or infer any information.
    Return a JSON object with a top-level key 'venues' that is an array of venue objects.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_geographic_diversity(evaluator: Evaluator, parent_node, venues: List[VenueItem]) -> None:
    # Only consider the first 4 venues (pad if fewer)
    selected = (venues + [VenueItem(), VenueItem(), VenueItem(), VenueItem()])[:4]
    canon_states = [canonicalize_state(v.state) for v in selected]
    # All 4 states must be recognized US states and all distinct
    result = all(s is not None for s in canon_states) and len(set(canon_states)) == 4

    states_display = [s if s is not None else "(missing/invalid)" for s in canon_states]

    evaluator.add_custom_node(
        result=result,
        id="geographic_diversity",
        desc="All four identified venues must be located in different U.S. states (no two venues from the same state)",
        parent=parent_node,
        critical=True
    )

    # Record info for debugging
    evaluator.add_custom_info(
        info={"selected_states_canonical": states_display},
        info_type="diagnostics",
        info_name="geographic_diversity_states"
    )


async def verify_single_venue(evaluator: Evaluator, parent_node, venue: VenueItem, idx: int) -> None:
    """
    Build and verify checks for a single venue according to the rubric.
    The venue node is parallel and non-critical (partial credit allowed),
    while individual leaves are critical within this venue.
    """
    venue_num = idx + 1
    venue_node = evaluator.add_parallel(
        id=f"venue_{venue_num}",
        desc=f"Venue #{venue_num} verification",
        parent=parent_node,
        critical=False
    )

    # Critical: Reference URL presence/validity (precondition for other checks)
    url_ok = is_valid_url(venue.reference_url)
    evaluator.add_custom_node(
        result=url_ok,
        id=f"venue_{venue_num}_reference_url",
        desc=f"Provides a verifiable reference URL documenting the venue information",
        parent=venue_node,
        critical=True
    )

    # Prepare common elements for claims
    name = venue.name or ""
    city = venue.city or ""
    state = venue.state or ""
    opening_year = venue.opening_year or ""
    url = venue.reference_url if url_ok else None

    # Leaf: Name and Location supported by the URL
    name_loc_leaf = evaluator.add_leaf(
        id=f"venue_{venue_num}_name_and_location",
        desc="Provides the complete official name of the venue along with its city and state location",
        parent=venue_node,
        critical=True
    )
    name_loc_claim = (
        f"The referenced page corresponds to the venue named '{name}' located in {city}, {state}, United States."
    )
    await evaluator.verify(
        claim=name_loc_claim,
        node=name_loc_leaf,
        sources=url,
        additional_instruction=(
            "Verify that the page clearly identifies the venue by the given name and lists its city and state location "
            "in the United States. Allow minor formatting or punctuation differences in the name. "
            "Accept 'USA' or 'U.S.' as equivalent to 'United States'."
        )
    )

    # Leaf: Arena Type is Esports (not multipurpose)
    type_leaf = evaluator.add_leaf(
        id=f"venue_{venue_num}_arena_type",
        desc="The venue is classified as an 'Esports' type arena (not a Multipurpose venue)",
        parent=venue_node,
        critical=True
    )
    type_claim = (
        f"The referenced page describes '{name}' as an esports venue (e.g., esports arena/facility/stadium) "
        f"dedicated to esports, rather than a general multipurpose arena."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=url,
        additional_instruction=(
            "Pass if the page explicitly calls it an esports venue (e.g., 'esports arena', 'esports facility', "
            "'gaming arena') or describes esports as a primary, dedicated purpose. "
            "Fail if the page clearly states it is a general 'multipurpose' arena without being a dedicated esports venue."
        )
    )

    # Leaf: Capacity >= 500
    cap_leaf = evaluator.add_leaf(
        id=f"venue_{venue_num}_capacity",
        desc="The venue has a minimum seating capacity of 500 people",
        parent=venue_node,
        critical=True
    )
    cap_claim = (
        f"The seating capacity of '{name}' is at least 500 spectators."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=url,
        additional_instruction=(
            "Check for wording like 'capacity', 'seating capacity', 'seats', 'up to X seats', etc. "
            "If multiple capacities are listed, use the main spectator seating number. "
            "Pass if any supported capacity is 500 or more."
        )
    )

    # Leaf: Operational status and confirmed opening year
    op_leaf = evaluator.add_leaf(
        id=f"venue_{venue_num}_operational_status",
        desc="The venue is currently operational with a confirmed opening year (not TBD)",
        parent=venue_node,
        critical=True
    )
    op_claim = (
        f"'{name}' opened in {opening_year} and is currently operational."
    )
    await evaluator.verify(
        claim=op_claim,
        node=op_leaf,
        sources=url,
        additional_instruction=(
            "Verify that the page provides a concrete opening year (not 'TBD' or missing) "
            "and indicates the venue is currently open or operational (e.g., hosting events, active facility). "
            "If the page explicitly states it is closed or permanently shut down, fail."
        )
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
    Evaluate an answer for the U.S. dedicated esports venues task.
    """
    # Initialize evaluator with parallel root
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

    # Create the main aggregation node for this task (non-critical to allow partial credit)
    main_node = evaluator.add_parallel(
        id="esports_venue_identification",
        desc="Identify four dedicated esports venues in the United States that meet the specified criteria for hosting regional tournaments",
        parent=root,
        critical=False
    )

    # Extract venues from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Select first 4 venues (pad if fewer)
    selected_venues: List[VenueItem] = (extracted.venues + [VenueItem(), VenueItem(), VenueItem(), VenueItem()])[:4]

    # Record summary info for debugging
    evaluator.add_custom_info(
        info={
            "num_venues_provided": len(extracted.venues),
            "selected_venues": [v.dict() for v in selected_venues]
        },
        info_type="diagnostics",
        info_name="venues_overview"
    )

    # Geographic diversity check (critical leaf under main)
    await verify_geographic_diversity(evaluator, main_node, selected_venues)

    # Build verification for each venue
    for i in range(4):
        await verify_single_venue(evaluator, main_node, selected_venues[i], i)

    # Return evaluation summary
    return evaluator.get_summary()