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
TASK_ID = "ca_performing_arts_venues"
TASK_DESCRIPTION = """Identify two performing arts venues located in California, United States, that meet all of the following criteria as of December 2023:

1. The venue must have a seating capacity between 2,500 and 3,500 seats (inclusive).

2. The venue must be classified as a performing arts theater or concert hall designed for live theatrical or musical performances, not a sports arena, movie theater, or comedy club.

3. The venue must be currently operational and actively hosting live performances.

4. The venue must provide wheelchair-accessible seating on multiple levels or sections of the theater.

5. The venue must have accessible restrooms on all seating levels or floors where audience seating is located.

6. The venue must provide companion seating adjacent to or near wheelchair-accessible spaces.

7. The venue must offer transfer seats with swing-out or removable arms to assist patrons with mobility limitations.

8. The venue must regularly host Broadway touring productions, opera performances, or major concert events (not exclusively used for movie screenings or stand-up comedy shows).

9. The venue must be owned or operated by a recognized performing arts organization, professional venue management company, or municipal arts commission.

10. The venue must be purpose-built or extensively renovated specifically for live performance acoustics (not a converted sports facility or multipurpose arena without proper acoustic treatment).

11. The venue must have four or more exits as required by building codes for assembly occupancies with capacity exceeding 1,000 seats.

For each venue, provide: (a) the venue name, (b) the city location within California, (c) the seating capacity, and (d) at least one reference URL that verifies the venue meets these criteria.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    capacity: Optional[str] = None
    classification: Optional[str] = None  # e.g., "performing arts theater", "concert hall"
    operational_status: Optional[str] = None  # e.g., "operational", "currently hosting live performances"
    wheelchair_seating_multi_level: Optional[str] = None
    accessible_restrooms_all_levels: Optional[str] = None
    companion_seating: Optional[str] = None
    transfer_seating: Optional[str] = None
    event_types: Optional[str] = None  # e.g., "Broadway touring, opera, major concerts"
    professional_management: Optional[str] = None  # e.g., "operated by XYZ Performing Arts"
    acoustic_design: Optional[str] = None  # e.g., "purpose-built acoustics"
    safety_exits: Optional[str] = None  # e.g., "4+ exits"
    reference_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to five venues mentioned in the answer that are claimed to meet the California performing-arts venue criteria.

    For each venue, extract the following fields exactly as stated in the answer (use strings, not numbers):
    - name: The venue name.
    - city: The city location (must be within California).
    - capacity: The stated seating capacity (string).
    - classification: The venue type (e.g., "performing arts theater", "concert hall"). Do not include sports arenas, movie theaters, or comedy clubs.
    - operational_status: A phrase indicating the venue is currently operational and hosting live performances (as of December 2023).
    - wheelchair_seating_multi_level: Statement indicating wheelchair-accessible seating on multiple levels or sections.
    - accessible_restrooms_all_levels: Statement indicating accessible restrooms on all seating levels or floors where audience seating is located.
    - companion_seating: Statement indicating companion seating adjacent to/near wheelchair spaces.
    - transfer_seating: Statement indicating transfer seats with swing-out or removable arms.
    - event_types: Statement indicating the venue regularly hosts Broadway touring productions, opera performances, or major concert events (not exclusively movies or stand-up comedy).
    - professional_management: Statement indicating the venue is owned/operated by a recognized performing arts org, professional venue management company, or municipal arts commission.
    - acoustic_design: Statement indicating the venue is purpose-built or extensively renovated specifically for live performance acoustics (not a converted sports/multipurpose arena without proper acoustic treatment).
    - safety_exits: Statement indicating the venue has four or more exits per building code for large assembly occupancies.
    - reference_urls: An array of all URLs cited in the answer specifically for this venue. Extract only valid URLs explicitly present in the answer (plain URLs or markdown links). If none, return an empty list.

    Return:
    { "venues": [ VenueItem, ... ] }

    If any field is missing for a venue, set it to null (or empty list for reference_urls).
    Extract strictly what appears in the answer without inventing details.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _ensure_two(venues: List[VenueItem]) -> List[VenueItem]:
    # Keep only the first two venues; pad with empty placeholders if fewer
    v = venues[:2]
    while len(v) < 2:
        v.append(VenueItem())
    return v


def _safe_city_claim(city: Optional[str]) -> str:
    if _non_empty_str(city):
        return f"{city}, California, United States"
    return "California, United States"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _build_output_format_nodes(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    idx: int
) -> Dict[str, Any]:
    """
    Create Output Format subtree and return important leaf nodes for prerequisites.
    """
    out_node = evaluator.add_parallel(
        id=f"Venue_{idx}_Output_Format",
        desc=f"Verification that all required output elements are provided for the {'first' if idx == 1 else 'second'} venue",
        parent=parent_node,
        critical=True
    )

    # Name provided
    evaluator.add_custom_node(
        result=_non_empty_str(venue.name),
        id=f"Venue_{idx}_Name_Provided",
        desc=f"The answer provides the name of the {'first' if idx == 1 else 'second'} venue",
        parent=out_node,
        critical=True
    )

    # City provided
    evaluator.add_custom_node(
        result=_non_empty_str(venue.city),
        id=f"Venue_{idx}_City_Provided",
        desc=f"The answer provides the city location within California for the {'first' if idx == 1 else 'second'} venue",
        parent=out_node,
        critical=True
    )

    # Capacity stated
    evaluator.add_custom_node(
        result=_non_empty_str(venue.capacity),
        id=f"Venue_{idx}_Capacity_Stated",
        desc=f"The answer states the seating capacity for the {'first' if idx == 1 else 'second'} venue",
        parent=out_node,
        critical=True
    )

    # Reference URL provided (we will use this as a prerequisite for factual checks)
    ref_url_node = evaluator.add_custom_node(
        result=bool(venue.reference_urls and len(venue.reference_urls) > 0),
        id=f"Venue_{idx}_Reference_URL",
        desc=f"At least one credible reference URL is provided that supports the venue's qualification under the stated criteria",
        parent=out_node,
        critical=True
    )

    return {"out_node": out_node, "ref_url_node": ref_url_node}


async def _build_basic_criteria_nodes(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    idx: int,
    prerequisite_node
) -> None:
    """
    Create Basic Criteria subtree and run verification leaves.
    """
    basic_node = evaluator.add_parallel(
        id=f"Venue_{idx}_Basic_Criteria",
        desc=f"Verification of fundamental qualifying criteria for the {'first' if idx == 1 else 'second'} venue",
        parent=parent_node,
        critical=True
    )

    # California Location
    loc_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_California_Location",
        desc="The venue must be physically located within the state of California, United States",
        parent=basic_node,
        critical=True
    )
    loc_claim = f"The venue '{venue.name or ''}' is located in {_safe_city_claim(venue.city)}."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=venue.reference_urls,
        additional_instruction="Verify that the venue is in California, USA. If a city is provided, confirm that city is in California.",
        extra_prerequisites=[prerequisite_node]
    )

    # Capacity Range 2500–3500 inclusive
    cap_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Capacity_Range",
        desc="The venue's seating capacity must be between 2,500 and 3,500 seats (inclusive)",
        parent=basic_node,
        critical=True
    )
    cap_claim = (
        f"The seating capacity of '{venue.name or ''}' is '{venue.capacity or ''}', and it lies between 2,500 and 3,500 seats inclusive."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=venue.reference_urls,
        additional_instruction=(
            "Check the stated capacity on the referenced page(s). "
            "If the page shows a specific capacity, judge whether it falls within [2500, 3500]. "
            "Minor rounding is acceptable."
        ),
        extra_prerequisites=[prerequisite_node]
    )

    # Performing arts type (not sports arena/movie theater/comedy club)
    type_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Performing_Arts_Type",
        desc="The venue must be classified as a performing arts theater or concert hall, not a sports arena, movie theater, or comedy club",
        parent=basic_node,
        critical=True
    )
    type_claim = (
        f"'{venue.name or ''}' is a performing arts theater or concert hall designed for live theatrical or musical performances, "
        f"not a sports arena, movie theater, or comedy club."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=venue.reference_urls,
        additional_instruction="Verify the venue classification on the official or reputable page: it should be a theater or concert hall for live performances.",
        extra_prerequisites=[prerequisite_node]
    )

    # Operational status (as of Dec 2023)
    op_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Operational_Status",
        desc="The venue must be currently operational and actively hosting live performances as of December 2023",
        parent=basic_node,
        critical=True
    )
    op_claim = (
        f"As of December 2023, '{venue.name or ''}' is currently operational and actively hosting live performances."
    )
    await evaluator.verify(
        claim=op_claim,
        node=op_leaf,
        sources=venue.reference_urls,
        additional_instruction="Check event calendars, schedule pages, or announcements around late 2023 to confirm active operations.",
        extra_prerequisites=[prerequisite_node]
    )


async def _build_detailed_requirements_nodes(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    idx: int,
    prerequisite_node
) -> None:
    """
    Create Detailed Requirements subtree and run verification leaves.
    """
    det_node = evaluator.add_parallel(
        id=f"Venue_{idx}_Detailed_Requirements",
        desc=f"Verification of accessibility, operational, and safety requirements for the {'first' if idx == 1 else 'second'} venue",
        parent=parent_node,
        critical=True
    )

    # Wheelchair-accessible seating on multiple levels/sections
    wheel_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Wheelchair_Seating",
        desc="The venue provides wheelchair-accessible seating locations on multiple levels or sections of the theater",
        parent=det_node,
        critical=True
    )
    wheel_claim = (
        f"'{venue.name or ''}' provides wheelchair-accessible seating on multiple levels or in multiple sections of the theater."
    )
    await evaluator.verify(
        claim=wheel_claim,
        node=wheel_leaf,
        sources=venue.reference_urls,
        additional_instruction="Look for ADA/accessibility pages or seating charts indicating accessible seating is available on multiple levels/sections.",
        extra_prerequisites=[prerequisite_node]
    )

    # Accessible restrooms on all seating levels/floors
    rest_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Accessible_Restrooms",
        desc="The venue has accessible restrooms available on all seating levels or floors where audience seating is located",
        parent=det_node,
        critical=True
    )
    rest_claim = (
        f"'{venue.name or ''}' has accessible restrooms on all seating levels or floors where audience seating is located."
    )
    await evaluator.verify(
        claim=rest_claim,
        node=rest_leaf,
        sources=venue.reference_urls,
        additional_instruction="Check accessibility information stating accessible restrooms are available on all audience seating levels/floors.",
        extra_prerequisites=[prerequisite_node]
    )

    # Companion seating adjacent to wheelchair-accessible spaces
    comp_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Companion_Seating",
        desc="The venue provides companion seating adjacent to or near wheelchair-accessible spaces",
        parent=det_node,
        critical=True
    )
    comp_claim = (
        f"'{venue.name or ''}' provides companion seating adjacent to or near wheelchair-accessible spaces."
    )
    await evaluator.verify(
        claim=comp_claim,
        node=comp_leaf,
        sources=venue.reference_urls,
        additional_instruction="Verify the accessibility page mentions companion seating adjacent to wheelchair spaces.",
        extra_prerequisites=[prerequisite_node]
    )

    # Transfer seats with swing-out/removable arms
    trans_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Transfer_Seating",
        desc="The venue offers transfer seats with swing-out or removable arms to assist patrons with mobility limitations",
        parent=det_node,
        critical=True
    )
    trans_claim = (
        f"'{venue.name or ''}' offers transfer seats with swing-out or removable arms to assist patrons with mobility limitations."
    )
    await evaluator.verify(
        claim=trans_claim,
        node=trans_leaf,
        sources=venue.reference_urls,
        additional_instruction="Look for mention of transfer seats with swing-out/removable arms on the accessibility page or seating information.",
        extra_prerequisites=[prerequisite_node]
    )

    # Event types: Broadway touring / opera / major concerts (not exclusively movies/comedy)
    evt_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Event_Types",
        desc="The venue regularly hosts Broadway touring productions, opera performances, or major concert events (not exclusively movies or stand-up comedy)",
        parent=det_node,
        critical=True
    )
    evt_claim = (
        f"'{venue.name or ''}' regularly hosts Broadway touring productions, opera performances, or major concert events and is not exclusively used for movies or stand-up comedy."
    )
    await evaluator.verify(
        claim=evt_claim,
        node=evt_leaf,
        sources=venue.reference_urls,
        additional_instruction="Check event calendars, past events, or programming history for Broadway tours, opera, or major concerts.",
        extra_prerequisites=[prerequisite_node]
    )

    # Professional management: owned/operated by recognized org/company/municipal commission
    mgmt_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Professional_Management",
        desc="The venue is owned or operated by a recognized performing arts organization, professional venue management company, or municipal arts commission",
        parent=det_node,
        critical=True
    )
    mgmt_claim = (
        f"'{venue.name or ''}' is owned or operated by a recognized performing arts organization, professional venue management company, or a municipal arts commission."
    )
    await evaluator.verify(
        claim=mgmt_claim,
        node=mgmt_leaf,
        sources=venue.reference_urls,
        additional_instruction="Verify ownership/operations details on the venue's about page or credible sources.",
        extra_prerequisites=[prerequisite_node]
    )

    # Acoustic design: purpose-built or extensively renovated for live performance acoustics
    acoust_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Acoustic_Design",
        desc="The venue is purpose-built or extensively renovated specifically for live performance acoustics",
        parent=det_node,
        critical=True
    )
    acoust_claim = (
        f"'{venue.name or ''}' is purpose-built or extensively renovated specifically for live performance acoustics (not a converted sports facility or untreated multipurpose arena)."
    )
    await evaluator.verify(
        claim=acoust_claim,
        node=acoust_leaf,
        sources=venue.reference_urls,
        additional_instruction="Look for design/renovation notes about acoustics optimization for live performance.",
        extra_prerequisites=[prerequisite_node]
    )

    # Safety exits: four or more exits
    exits_leaf = evaluator.add_leaf(
        id=f"Venue_{idx}_Safety_Exits",
        desc="The venue has four or more exits as required by building codes for assembly occupancies with over 1,000 seat capacity",
        parent=det_node,
        critical=True
    )
    exits_claim = (
        f"'{venue.name or ''}' has four or more exits as required by building codes for large assembly occupancies."
    )
    await evaluator.verify(
        claim=exits_claim,
        node=exits_leaf,
        sources=venue.reference_urls,
        additional_instruction="Check building/safety information, evacuation plans, or credible documentation indicating exit counts (>=4). If not explicitly stated, treat as not supported.",
        extra_prerequisites=[prerequisite_node]
    )


async def verify_venue(
    evaluator: Evaluator,
    root_node,
    venue: VenueItem,
    idx: int
) -> None:
    """
    Build the full verification subtree for one venue (idx = 1 or 2).
    Employ sequential gating; ensure URL existence gates factual checks.
    """
    # Parent sequential node for this venue
    venue_seq = evaluator.add_sequential(
        id=f"Venue_{idx}_Identification",
        desc=f"{'First' if idx == 1 else 'Second'} qualifying venue identification and verification",
        parent=root_node,
        critical=False
    )

    # We create Output Format first to gate subsequent leaves on reference URL existence
    output_nodes = await _build_output_format_nodes(evaluator, venue_seq, venue, idx)
    ref_url_leaf = output_nodes["ref_url_node"]

    # Basic criteria subtree with verifications
    await _build_basic_criteria_nodes(evaluator, venue_seq, venue, idx, ref_url_leaf)

    # Detailed requirements subtree with verifications
    await _build_detailed_requirements_nodes(evaluator, venue_seq, venue, idx, ref_url_leaf)


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
    Evaluate an answer for the California performing arts venues task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel across the two venues
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
        extraction_name="venues_extraction",
    )

    # Ensure we have exactly two venues to verify
    venues_to_check = _ensure_two(extracted.venues)

    # Build verification subtrees for Venue 1 and Venue 2
    # Note: Each venue subtree is sequential and will apply critical gating internally.
    for i, venue in enumerate(venues_to_check, start=1):
        await verify_venue(evaluator, root, venue, i)

    # Return structured summary
    return evaluator.get_summary()