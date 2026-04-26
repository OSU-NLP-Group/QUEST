import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "la_indoor_venues_ada_festival"
TASK_DESCRIPTION = """
Identify four indoor performing arts venues in Los Angeles, California that would be suitable for hosting a multi-day theater festival. Each venue must meet all of the following requirements: (1) Have a seating capacity between 1,500 and 3,000 seats; (2) Be located within Los Angeles, California (provide the complete street address); (3) Provide ADA-compliant wheelchair accessible seating that meets federal requirements, including minimum wheelchair accessible seating (at least 1% of total capacity or required ADA ratio), wheelchair spaces at least 36 inches wide for single spaces, adjacent companion seating for wheelchair spaces, and accessible seating priced equivalently to comparable seats in the same section; (4) Have accessible entrance(s) meeting ADA requirements; (5) Have accessible restroom facilities; (6) Operate a box office or ticket sales operation; (7) Have a stated ticketing policy regarding age requirements and admission; (8) Host ticketed live entertainment or performing arts events. For each venue, provide the venue name, complete address, seating capacity, and a reference URL supporting the information.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ADASeatingDetails(BaseModel):
    wheelchair_minimum: Optional[str] = None  # e.g., "At least 1% of seats reserved for wheelchair users"
    wheelchair_dimensions: Optional[str] = None  # e.g., "36 inches wide single wheelchair spaces"
    companion_seating: Optional[str] = None  # e.g., "Adjacent companion seats available"
    accessible_pricing: Optional[str] = None  # e.g., "Accessible seats priced same as comparable seats"


class VenueItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None  # Prefer full address string if provided
    capacity: Optional[str] = None  # Prefer string to allow ranges or approximations
    sources: List[str] = Field(default_factory=list)  # All URLs cited for the venue
    ada: Optional[ADASeatingDetails] = None
    indoor_statement: Optional[str] = None
    entrance_statement: Optional[str] = None
    restrooms_statement: Optional[str] = None
    box_office_statement: Optional[str] = None
    ticketing_policy_statement: Optional[str] = None
    live_events_statement: Optional[str] = None


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to four indoor performing arts venues in Los Angeles, California from the answer. Return a JSON object with a 'venues' array (max length 4). For each venue, extract:
    - name: The venue name (string as written in the answer)
    - address: The complete street address as presented (include street number/name, city, state, and ZIP if available)
    - capacity: The seating capacity text exactly as stated (e.g., "2,300", "approx. 2,000", "2,100–2,200")
    - sources: An array of all URLs cited for this venue (official venue site, ticketing pages, policy pages, ADA/accessibility pages, reputable third-party pages, etc.)
    - ada.wheelchair_minimum: Text in the answer related to minimum ADA wheelchair accessible seating amounts/ratios (e.g., "at least 1%")
    - ada.wheelchair_dimensions: Text related to wheelchair space dimensions (e.g., "36 inches wide")
    - ada.companion_seating: Text confirming companion seats adjacent to wheelchair spaces
    - ada.accessible_pricing: Text confirming accessible seating is priced equivalently to comparable seats in the same section
    - indoor_statement: Any text indicating it is an indoor theater/performing arts space
    - entrance_statement: Any text indicating accessible entrance(s) meeting ADA requirements
    - restrooms_statement: Any text indicating accessible restroom facilities
    - box_office_statement: Any text indicating the venue operates a box office or ticket sales operation
    - ticketing_policy_statement: Any text of a ticketing/admission/age policy statement
    - live_events_statement: Any text indicating the venue hosts ticketed live entertainment or performing arts events

    Rules:
    - Do not invent or infer information absent from the answer; extract literally from the answer text.
    - Only include URLs that are explicitly present in the answer (plain URLs or markdown links).
    - If a field is not present for a venue, return null for that field. If no URLs are present, return an empty array for 'sources'.
    - Maintain the exact wording/casing from the answer for textual fields.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][n] if 0 <= n < 5 else f"#{n+1}"


def _clean_sources(urls: List[str]) -> List[str]:
    # Keep only non-empty strings; obj_task_eval will further normalize
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


def _venue_desc(idx: int) -> str:
    return f"{_ordinal(idx)} venue meets all specified requirements"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _verify_with_sources_or_fail(
    evaluator: Evaluator,
    node,
    claim: str,
    sources: List[str],
    additional_instruction: str
) -> None:
    if sources:
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=sources,
            additional_instruction=additional_instruction
        )
    else:
        # Enforce source-grounding policy: no sources -> immediate fail
        node.score = 0.0
        node.status = "failed"


async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    idx: int
) -> None:
    """
    Build the verification sub-tree for a single venue and run all verifications.
    """
    # Parent node for this venue
    venue_node = evaluator.add_parallel(
        id=f"venue_{idx+1}",
        desc=_venue_desc(idx),
        parent=parent_node,
        critical=False
    )

    name = venue.name or "the venue"
    address = venue.address or ""
    capacity = venue.capacity or ""
    sources = _clean_sources(venue.sources)

    # 1) Capacity between 1,500 and 3,000 seats (critical)
    node_capacity = evaluator.add_leaf(
        id=f"venue_{idx+1}_capacity",
        desc="Venue has seating capacity between 1,500 and 3,000 seats",
        parent=venue_node,
        critical=True
    )
    capacity_claim = (
        f"The seating capacity of '{name}' is {capacity} seats, and this capacity lies between 1,500 and 3,000 inclusive."
        if capacity else
        f"The webpage for '{name}' indicates a seating capacity that lies between 1,500 and 3,000 inclusive."
    )
    await _verify_with_sources_or_fail(
        evaluator, node_capacity, capacity_claim, sources,
        "Use the page(s) to identify the stated seating capacity, allowing minor rounding or formatting (e.g., 2,000+, approx. 2,250). Confirm the number falls within 1,500–3,000 inclusive."
    )

    # 2) Location is in Los Angeles, California (critical)
    node_location = evaluator.add_leaf(
        id=f"venue_{idx+1}_location",
        desc="Venue is located in Los Angeles, California",
        parent=venue_node,
        critical=True
    )
    location_claim = f"The venue '{name}' is located in Los Angeles, California."
    await _verify_with_sources_or_fail(
        evaluator, node_location, location_claim, sources,
        "Confirm that the address or location on the page indicates 'Los Angeles, CA' (City of Los Angeles). Accept LA neighborhood names if the page makes clear they are in Los Angeles, CA."
    )

    # 3) Complete street address is provided and verifiable (critical)
    node_address = evaluator.add_leaf(
        id=f"venue_{idx+1}_address",
        desc="Complete street address is provided and verifiable",
        parent=venue_node,
        critical=True
    )
    address_claim = (
        f"The page provides the complete street address for '{name}': '{address}', including street number and name, city, state, and ZIP (or equivalent completeness)."
        if address else
        f"The page provides the complete street address for '{name}', including street number and name, city, state, and ZIP (or equivalent completeness)."
    )
    await _verify_with_sources_or_fail(
        evaluator, node_address, address_claim, sources,
        "Confirm the page lists a full postal address (street number/name, Los Angeles, CA, and ideally ZIP). If only partial info is shown, do not pass."
    )

    # 4) Indoor theater/performing arts space (critical)
    node_indoor = evaluator.add_leaf(
        id=f"venue_{idx+1}_indoor",
        desc="Venue is an indoor theater or performing arts space",
        parent=venue_node,
        critical=True
    )
    indoor_claim = f"The venue '{name}' is an indoor theater or indoor performing arts venue (not an outdoor amphitheater)."
    await _verify_with_sources_or_fail(
        evaluator, node_indoor, indoor_claim, sources,
        "Look for indications such as 'theatre', 'concert hall', 'indoor', or photos/description implying a closed indoor auditorium."
    )

    # 5) Accessibility (ADA seating) - parent node critical with 4 critical children
    access_parent = evaluator.add_parallel(
        id=f"venue_{idx+1}_accessibility",
        desc="Venue provides ADA-compliant wheelchair accessible seating",
        parent=venue_node,
        critical=True
    )
    # 5a) Minimum wheelchair seating >= 1% (critical)
    node_wc_min = evaluator.add_leaf(
        id=f"venue_{idx+1}_wheelchair_percentage",
        desc="Wheelchair accessible seating meets minimum ADA requirements (at least 1% of capacity or required ratio)",
        parent=access_parent,
        critical=True
    )
    wc_min_claim = f"The venue '{name}' provides ADA-compliant wheelchair accessible seating that meets minimum quantity requirements (around at least 1% of total capacity or ADA-required ratio)."
    await _verify_with_sources_or_fail(
        evaluator, node_wc_min, wc_min_claim, sources,
        "Pass only if the page explicitly states ADA wheelchair seating quantity compliance (e.g., states ADA compliance for required minimums) or otherwise clearly implies compliance with ADA-required minimum amounts."
    )

    # 5b) Wheelchair space dimensions >= 36 inches wide (critical)
    node_wc_dim = evaluator.add_leaf(
        id=f"venue_{idx+1}_wheelchair_dimensions",
        desc="Wheelchair spaces meet minimum dimension requirements (36 inches wide for single spaces)",
        parent=access_parent,
        critical=True
    )
    wc_dim_claim = f"Wheelchair spaces at '{name}' meet ADA minimum dimensions, i.e., at least 36 inches wide for single spaces."
    await _verify_with_sources_or_fail(
        evaluator, node_wc_dim, wc_dim_claim, sources,
        "Look for explicit wheelchair space dimensions or an ADA policy page that states compliance with ADA seat dimensions. If such info is absent, do not pass."
    )

    # 5c) Companion seating adjacent to wheelchair spaces (critical)
    node_companion = evaluator.add_leaf(
        id=f"venue_{idx+1}_companion_seating",
        desc="Adjacent companion seats are provided for wheelchair spaces",
        parent=access_parent,
        critical=True
    )
    companion_claim = f"Companion seating adjacent to wheelchair spaces is provided at '{name}'."
    await _verify_with_sources_or_fail(
        evaluator, node_companion, companion_claim, sources,
        "Verify the page mentions companion seating adjacent to wheelchair spaces or an ADA policy that ensures this."
    )

    # 5d) Accessible seating priced equivalently (critical)
    node_pricing = evaluator.add_leaf(
        id=f"venue_{idx+1}_accessible_pricing",
        desc="Accessible seating is priced the same as comparable seats in the same section",
        parent=access_parent,
        critical=True
    )
    pricing_claim = f"Accessible seating at '{name}' is priced equivalently to comparable seats in the same section."
    await _verify_with_sources_or_fail(
        evaluator, node_pricing, pricing_claim, sources,
        "Pass only if the page indicates price parity/equivalence for accessible seating compared to comparable nearby seats."
    )

    # 6) Accessible entrance(s) (critical)
    node_entrance = evaluator.add_leaf(
        id=f"venue_{idx+1}_entrance",
        desc="Venue has accessible entrance(s) meeting ADA requirements",
        parent=venue_node,
        critical=True
    )
    entrance_claim = f"The venue '{name}' has accessible entrance(s) that meet ADA requirements."
    await _verify_with_sources_or_fail(
        evaluator, node_entrance, entrance_claim, sources,
        "Look for 'accessible entrance', 'ADA entrance', step-free access, ramps, automatic doors, etc."
    )

    # 7) Box office / ticket sales operation (critical)
    node_box = evaluator.add_leaf(
        id=f"venue_{idx+1}_box_office",
        desc="Venue operates a box office or ticket sales operation",
        parent=venue_node,
        critical=True
    )
    box_claim = f"The venue '{name}' operates a box office or ticket sales operation (on-site or official ticketing)."
    await _verify_with_sources_or_fail(
        evaluator, node_box, box_claim, sources,
        "Accept if the page mentions 'Box Office', 'Ticket Office', 'Tickets', or an official ticketing partner link for the venue."
    )

    # 8) Ticketing policy (age/admission) (critical)
    node_policy = evaluator.add_leaf(
        id=f"venue_{idx+1}_ticketing_policy",
        desc="Venue has a stated ticketing policy regarding age requirements and admission",
        parent=venue_node,
        critical=True
    )
    policy_claim = f"The venue '{name}' has a stated ticketing policy regarding age requirements and/or admission (e.g., children policy, minimum age, ID requirements)."
    await _verify_with_sources_or_fail(
        evaluator, node_policy, policy_claim, sources,
        "Look for explicit age restrictions, 'children under ...', 'everyone requires a ticket', or admission policy text."
    )

    # 9) Hosts ticketed live entertainment or performing arts events (critical)
    node_events = evaluator.add_leaf(
        id=f"venue_{idx+1}_live_events",
        desc="Venue hosts ticketed live entertainment or performing arts events",
        parent=venue_node,
        critical=True
    )
    events_claim = f"The venue '{name}' hosts ticketed live entertainment and/or performing arts events."
    await _verify_with_sources_or_fail(
        evaluator, node_events, events_claim, sources,
        "Check event listings, past shows, calendars, or descriptive text indicating ticketed performing arts or live entertainment."
    )

    # 10) Accessible restroom facilities (critical)
    node_restrooms = evaluator.add_leaf(
        id=f"venue_{idx+1}_restrooms",
        desc="Venue has accessible restroom facilities",
        parent=venue_node,
        critical=True
    )
    restrooms_claim = f"The venue '{name}' provides accessible restroom facilities."
    await _verify_with_sources_or_fail(
        evaluator, node_restrooms, restrooms_claim, sources,
        "Look for 'accessible restrooms', 'ADA restrooms', 'wheelchair accessible restrooms', or similar."
    )

    # 11) Reference URL support (critical)
    node_source = evaluator.add_leaf(
        id=f"venue_{idx+1}_source",
        desc="Information is supported by a verifiable reference URL",
        parent=venue_node,
        critical=True
    )
    source_claim = f"The provided reference page(s) are about the venue '{name}' and support key details (e.g., address and/or capacity and policy/accessibility information)."
    await _verify_with_sources_or_fail(
        evaluator, node_source, source_claim, sources,
        "Pass if at least one provided URL is an official or authoritative page that clearly supports the venue's identity and key facts (address/capacity/policies)."
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
    Evaluate an answer for the LA indoor performing arts venues ADA task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # venues evaluated independently
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

    # Extract venues
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    venues: List[VenueItem] = list(extraction.venues or [])

    # Normalize to exactly 4 venues: take first four or pad with blanks
    if len(venues) > 4:
        venues = venues[:4]
    while len(venues) < 4:
        venues.append(VenueItem())

    # Add small custom info about URL counts
    evaluator.add_custom_info(
        {
            f"venue_{i+1}_url_count": len(_clean_sources(v.sources))
            for i, v in enumerate(venues)
        },
        info_type="url_stats",
        info_name="per_venue_url_counts"
    )

    # Build verification tree for each of the 4 venues (parallel under root)
    verify_tasks = []
    for i, venue in enumerate(venues):
        verify_tasks.append(verify_single_venue(evaluator, root, venue, i))

    await asyncio.gather(*verify_tasks)

    # Return final structured summary
    return evaluator.get_summary()