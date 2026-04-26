import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "three_us_indoor_concert_venues"
TASK_DESCRIPTION = """
Identify three major indoor concert venues located in the United States, with each venue in a different state. Each venue must meet the following requirements:

1. The venue must be an indoor arena or similar enclosed multi-purpose entertainment facility
2. The venue must have a concert seating capacity between 15,000 and 25,000
3. The venue must provide ADA-compliant wheelchair accessible seating
4. The venue must offer multi-tier ticket pricing for concerts, including both standard seating options and premium seating or VIP packages

For each venue, provide:
- Venue name
- City and state location
- Concert seating capacity
- Confirmation of wheelchair accessible seating availability
- Description of ticket tier structure
- Reference URLs supporting each piece of information
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    """One venue with attributes and per-attribute citations."""
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    concert_capacity: Optional[str] = None  # keep as string to allow ranges/approx.
    accessible_seating_statement: Optional[str] = None
    ticket_tier_description: Optional[str] = None

    sources_location: List[str] = Field(default_factory=list)
    sources_facility_type: List[str] = Field(default_factory=list)
    sources_capacity: List[str] = Field(default_factory=list)
    sources_accessibility: List[str] = Field(default_factory=list)
    sources_ticket_tiers: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    """All venues extracted from the agent's answer."""
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract all venue entries mentioned in the answer that appear to be indoor concert venues in the United States.
    For each venue entry, return an object with the following fields:

    - name: The venue name, exactly as stated in the answer (string; null if missing)
    - city: City location (string; null if missing)
    - state: State location (string; null if missing; should be a U.S. state or common abbreviation if provided)
    - concert_capacity: The concert seating capacity value or range mentioned (string; null if missing)
    - accessible_seating_statement: The answer's confirmation or statement about ADA/accessible seating (string; null if missing)
    - ticket_tier_description: The answer's description of the concert ticket tier structure (string; null if missing)

    For each of the following aspects, extract the URLs explicitly cited in the answer as supporting references (include only valid, full URLs; ignore non-URL mentions):
    - sources_location: URLs that support the stated city/state location
    - sources_facility_type: URLs that support that the venue is indoor/enclosed arena or similar
    - sources_capacity: URLs that support the stated concert seating capacity
    - sources_accessibility: URLs that support ADA-compliant wheelchair accessible seating availability
    - sources_ticket_tiers: URLs that support multi-tier ticket pricing including standard seating and premium/VIP options

    IMPORTANT:
    - Extract URLs exactly as shown in the answer (including inside markdown links).
    - Do not invent or infer URLs; if none are provided for a particular aspect, return an empty list for that aspect.
    - If the answer mentions more than 3 venues, extract them all; the evaluator will consider only the first 3 later.
    - If a field is not present in the answer for a venue, set it to null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_valid_url(url: str) -> bool:
    """Basic URL validity check."""
    if not isinstance(url, str):
        return False
    u = url.strip().lower()
    return u.startswith("http://") or u.startswith("https://")


def _urls_nonempty_and_valid(urls: List[str]) -> bool:
    """Check that URLs list is non-empty and each is plausibly valid."""
    return bool(urls) and all(_is_valid_url(u) for u in urls)


def _normalize_string(s: Optional[str]) -> str:
    return (s or "").strip()


def _distinct_nonempty_names(venues: List[VenueItem]) -> bool:
    names = [(_normalize_string(v.name).casefold()) for v in venues[:3] if _normalize_string(v.name)]
    return len(names) == 3 and len(set(names)) == 3


def _states_all_different(venues: List[VenueItem]) -> bool:
    states = [(_normalize_string(v.state).casefold()) for v in venues[:3] if _normalize_string(v.state)]
    return len(states) == 3 and len(set(states)) == 3


# --------------------------------------------------------------------------- #
# Verification subroutine for a single venue                                  #
# --------------------------------------------------------------------------- #
async def verify_one_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    index: int
) -> None:
    """
    Build verification nodes for a single venue and perform checks.
    Follows the rubric structure: parallel group under Venue_i with critical leaves.
    """
    venue_label = f"Venue_{index + 1}"

    # Parent node for this venue (parallel, non-critical to allow partial credit)
    venue_node = evaluator.add_parallel(
        id=venue_label,
        desc=("First venue and its required attributes and citations."
              if index == 0 else
              "Second venue and its required attributes and citations."
              if index == 1 else
              "Third venue and its required attributes and citations."),
        parent=parent_node,
        critical=False
    )

    # 1) Venue_Name_Provided (critical, existence check)
    evaluator.add_custom_node(
        result=bool(_normalize_string(venue.name)),
        id=f"{venue_label}_Venue_Name_Provided",
        desc="Venue name is provided.",
        parent=venue_node,
        critical=True
    )

    # 2) City_And_State_Provided (critical, existence check)
    evaluator.add_custom_node(
        result=bool(_normalize_string(venue.city)) and bool(_normalize_string(venue.state)),
        id=f"{venue_label}_City_And_State_Provided",
        desc="Venue city and state are provided.",
        parent=venue_node,
        critical=True
    )

    # 3) State_Is_US_State (critical, simple factual check; non-web)
    state_check_node = evaluator.add_leaf(
        id=f"{venue_label}_State_Is_US_State",
        desc="The provided state is a U.S. state (supports that the venue is located in the United States).",
        parent=venue_node,
        critical=True
    )
    claim_state = f"The state '{_normalize_string(venue.state)}' is a U.S. state."
    await evaluator.verify(
        claim=claim_state,
        node=state_check_node,
        additional_instruction="Consider standard U.S. state names or common abbreviations (e.g., 'CA' for California) as valid."
    )

    # 4) Citations existence checks (critical). Create first so subsequent verifications can depend on these.
    # Location citations
    evaluator.add_custom_node(
        result=_urls_nonempty_and_valid(venue.sources_location),
        id=f"{venue_label}_Citations_For_Location",
        desc="Reference URL(s) are provided supporting the venue's city/state location.",
        parent=venue_node,
        critical=True
    )
    # Facility type citations
    evaluator.add_custom_node(
        result=_urls_nonempty_and_valid(venue.sources_facility_type),
        id=f"{venue_label}_Citations_For_Facility_Type",
        desc="Reference URL(s) are provided supporting that the venue is an indoor/enclosed arena or similar facility.",
        parent=venue_node,
        critical=True
    )
    # Capacity citations
    evaluator.add_custom_node(
        result=_urls_nonempty_and_valid(venue.sources_capacity),
        id=f"{venue_label}_Citations_For_Capacity",
        desc="Reference URL(s) are provided supporting the stated concert seating capacity.",
        parent=venue_node,
        critical=True
    )
    # Accessibility citations
    evaluator.add_custom_node(
        result=_urls_nonempty_and_valid(venue.sources_accessibility),
        id=f"{venue_label}_Citations_For_Accessible_Seating",
        desc="Reference URL(s) are provided supporting wheelchair accessible/ADA seating availability.",
        parent=venue_node,
        critical=True
    )
    # Ticket tiers citations
    evaluator.add_custom_node(
        result=_urls_nonempty_and_valid(venue.sources_ticket_tiers),
        id=f"{venue_label}_Citations_For_Ticket_Tiers",
        desc="Reference URL(s) are provided supporting the described multi-tier ticket pricing (standard and premium/VIP options).",
        parent=venue_node,
        critical=True
    )

    # 5) Indoor_Enclosed_Facility (critical, verify with sources)
    facility_node = evaluator.add_leaf(
        id=f"{venue_label}_Indoor_Enclosed_Facility",
        desc="Venue is an indoor arena or similar enclosed multi-purpose entertainment facility.",
        parent=venue_node,
        critical=True
    )
    facility_claim = f"The venue '{_normalize_string(venue.name)}' is an indoor arena or similar enclosed multi-purpose entertainment facility."
    await evaluator.verify(
        claim=facility_claim,
        node=facility_node,
        sources=venue.sources_facility_type,
        additional_instruction="Verify the venue is indoor/enclosed (not open-air). Accept equivalently 'indoor arena', 'enclosed arena', or 'indoor multi-purpose facility' stated on official or authoritative sources."
    )

    # 6) Concert_Capacity_Provided_And_In_Range (critical, verify with sources)
    capacity_node = evaluator.add_leaf(
        id=f"{venue_label}_Concert_Capacity_Provided_And_In_Range",
        desc="A concert seating capacity is provided, and it is between 15,000 and 25,000.",
        parent=venue_node,
        critical=True
    )
    capacity_text = _normalize_string(venue.concert_capacity)
    capacity_claim = (
        f"The venue '{_normalize_string(venue.name)}' has a concert seating capacity between 15,000 and 25,000."
        if not capacity_text
        else f"The concert seating capacity of '{_normalize_string(venue.name)}' is '{capacity_text}', and this value indicates the capacity is between 15,000 and 25,000."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_node,
        sources=venue.sources_capacity,
        additional_instruction=(
            "Check the stated capacity (or seating chart/official specs). Focus on concert seating capacity; "
            "if only general or sports seating capacity is stated but clearly equivalent, that is acceptable. "
            "Confirm the value falls between 15,000 and 25,000."
        )
    )

    # 7) ADA_Wheelchair_Accessible_Seating_Confirmed (critical, verify with sources)
    accessibility_node = evaluator.add_leaf(
        id=f"{venue_label}_ADA_Wheelchair_Accessible_Seating_Confirmed",
        desc="Answer confirms the venue provides ADA-compliant wheelchair accessible seating.",
        parent=venue_node,
        critical=True
    )
    accessibility_claim = (
        f"The venue '{_normalize_string(venue.name)}' provides ADA-compliant wheelchair accessible seating for concerts."
    )
    await evaluator.verify(
        claim=accessibility_claim,
        node=accessibility_node,
        sources=venue.sources_accessibility,
        additional_instruction="Confirm availability of ADA-compliant wheelchair accessible seating. Accept official ADA/accessibility policy pages, seat maps, or ticketing pages mentioning accessible seating."
    )

    # 8) Ticket_Tiers_Described_With_Standard_And_Premium_VIP (critical, verify with sources)
    ticket_node = evaluator.add_leaf(
        id=f"{venue_label}_Ticket_Tiers_Described_With_Standard_And_Premium_VIP",
        desc="Answer describes multi-tier concert ticket pricing that includes both standard seating options and premium seating and/or VIP packages.",
        parent=venue_node,
        critical=True
    )
    ticket_claim = (
        f"The venue '{_normalize_string(venue.name)}' offers multi-tier ticket pricing for concerts including both standard seating options and premium seating or VIP packages."
    )
    await evaluator.verify(
        claim=ticket_claim,
        node=ticket_node,
        sources=venue.sources_ticket_tiers,
        additional_instruction=(
            "Look for references to standard/general seating and premium/VIP options (e.g., VIP packages, suites, club seats, premium seating). "
            "Ticketing pages, seat maps, or official venue ticket information pages are acceptable."
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
    Evaluate an agent's answer for the three US indoor concert venues task.
    Returns the evaluation summary dict produced by the Mind2Web2 evaluator.
    """
    # Initialize evaluator (root node is non-critical to allow partial credit across venue groups)
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Keep only the first 3 venues for evaluation; pad if fewer
    venues = extraction.venues[:3]
    while len(venues) < 3:
        venues.append(VenueItem())

    # Record a structured overview of extracted venues as custom info
    evaluator.add_custom_info(
        info={
            "venues_count_total": len(extraction.venues),
            "venues_used_for_eval": [
                {
                    "name": v.name,
                    "city": v.city,
                    "state": v.state,
                    "concert_capacity": v.concert_capacity,
                    "accessible_seating_statement": v.accessible_seating_statement,
                    "ticket_tier_description": v.ticket_tier_description,
                    "sources_location": v.sources_location,
                    "sources_facility_type": v.sources_facility_type,
                    "sources_capacity": v.sources_capacity,
                    "sources_accessibility": v.sources_accessibility,
                    "sources_ticket_tiers": v.sources_ticket_tiers,
                }
                for v in venues
            ],
        },
        info_type="extraction_summary"
    )

    # Global critical checks under root

    # Venue_Count_Is_Three: ensure the answer provides at least 3 distinct venue entries
    evaluator.add_custom_node(
        result=(len(extraction.venues) >= 3) and _distinct_nonempty_names(venues),
        id="Venue_Count_Is_Three",
        desc="The answer provides three distinct venue entries (3 venues).",
        parent=root,
        critical=True
    )

    # Build venue verification subtrees (Venue_1, Venue_2, Venue_3)
    for i, venue in enumerate(venues):
        await verify_one_venue(evaluator, root, venue, i)

    # All_Venues_In_Different_States: ensure the three venues are in different states
    evaluator.add_custom_node(
        result=_states_all_different(venues),
        id="All_Venues_In_Different_States",
        desc="The three venues are each located in a different U.S. state (no two share the same state).",
        parent=root,
        critical=True
    )

    # Return summary
    return evaluator.get_summary()