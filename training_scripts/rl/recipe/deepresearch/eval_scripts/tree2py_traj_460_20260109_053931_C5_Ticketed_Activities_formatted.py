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
TASK_ID = "chi_indoor_concert_venues_2"
TASK_DESCRIPTION = (
    "Identify two indoor concert venues in Chicago, Illinois that meet the following requirements:\n"
    "1) Seating capacity between 3,000 and 8,000 seats; "
    "2) ADA accessibility including at least 1% wheelchair-accessible seating, companion seats adjacent to wheelchair seating, and dispersed wheelchair seating; "
    "3) Currently operational and available for hosting live music; "
    "4) Located within Chicago city limits. "
    "For each venue, provide the name, specific seating capacity, confirmation of ADA features, and a reference URL."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    capacity: Optional[str] = None
    is_indoor: Optional[str] = None
    operational_status: Optional[str] = None
    ada_wheelchair_minimum_percent: Optional[str] = None
    ada_companion_seats: Optional[str] = None
    ada_dispersed_wheelchair_locations: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    city_or_address: Optional[str] = None


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return (
        "From the provided answer, extract up to the first two indoor concert venues in Chicago, Illinois that the answer proposes. "
        "For each venue, extract the following fields exactly as stated in the answer:\n"
        "- name: The venue name.\n"
        "- capacity: The specific seating capacity value mentioned (keep formatting as in the answer; if a range is provided, include the range as-is).\n"
        "- is_indoor: A brief phrase indicating that it is an indoor concert/live music venue if stated; otherwise null.\n"
        "- operational_status: A brief phrase indicating the venue is operational/hosting live music (e.g., upcoming events/ticketing) if stated; otherwise null.\n"
        "- ada_wheelchair_minimum_percent: Any text indicating the venue provides at least 1% wheelchair-accessible seating or complies with ADA seating requirements; otherwise null.\n"
        "- ada_companion_seats: Any text indicating companion seats adjacent to wheelchair seating; otherwise null.\n"
        "- ada_dispersed_wheelchair_locations: Any text indicating wheelchair seating is dispersed in multiple locations/sections; otherwise null.\n"
        "- city_or_address: The city/address if the answer mentions it; otherwise null.\n"
        "- reference_urls: A list of all URLs cited in the answer for this venue (include venue site pages, ADA/accessibility pages, seating charts, Wikipedia/ticketing pages, etc.).\n"
        "Only extract venues explicitly mentioned in the answer; do not invent. "
        "If the answer mentions more than two venues, only include the first two. "
        "If the answer mentions fewer than two venues, fill the remainder with empty/null values."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    s = url.strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def _any_valid_url(urls: List[str]) -> bool:
    return any(_valid_url(u) for u in urls)


def _first_two_venues(extracted: VenuesExtraction) -> List[VenueItem]:
    items = extracted.venues[:2]
    while len(items) < 2:
        items.append(VenueItem())
    return items


# --------------------------------------------------------------------------- #
# Venue verification                                                          #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    index: int,
) -> None:
    """
    Build verification sub-tree and run checks for a single venue.
    """
    # Parent node for this venue (parallel aggregation, non-critical wrt root)
    vnode = evaluator.add_parallel(
        id=f"venue_{index + 1}",
        desc=f"Venue #{index + 1} evaluated against all requirements",
        parent=parent_node,
        critical=False,
    )

    # Extract URL list (normalize)
    urls = list(dict.fromkeys([u for u in (venue.reference_urls or []) if isinstance(u, str) and u.strip() != ""]))

    # 1) Venue name provided (critical, existence)
    evaluator.add_custom_node(
        result=bool(venue.name and venue.name.strip()),
        id=f"venue_{index + 1}_venue_name_provided",
        desc="Venue name is provided",
        parent=vnode,
        critical=True,
    )

    # 2) Reference URL(s) exist (critical)
    # The rubric requires a valid reference URL documenting info; as a gating step, require at least one valid-looking URL.
    evaluator.add_custom_node(
        result=_any_valid_url(urls),
        id=f"venue_{index + 1}_reference_url_supports_claims",
        desc="At least one valid reference URL is provided that can document the venue’s capacity and ADA accessibility features",
        parent=vnode,
        critical=True,
    )

    # 3) Capacity value provided (critical, existence of specific value in the answer text)
    evaluator.add_custom_node(
        result=bool(venue.capacity and venue.capacity.strip()),
        id=f"venue_{index + 1}_capacity_value_provided",
        desc="A specific seating capacity value is provided for the venue",
        parent=vnode,
        critical=True,
    )

    # The remaining checks are verified against sources. They should be gated by the critical siblings implicitly.
    # We verify them after creating the critical existence checks above to leverage auto preconditions.

    # 4) Within Chicago city limits (critical)
    within_chi_node = evaluator.add_leaf(
        id=f"venue_{index + 1}_within_chicago_city_limits",
        desc="Venue is located within Chicago city limits (not suburbs)",
        parent=vnode,
        critical=True,
    )
    within_chi_claim = (
        f"The venue '{venue.name or 'the venue'}' is located within the City of Chicago, Illinois (not outside city limits or in a suburb)."
    )
    await evaluator.verify(
        claim=within_chi_claim,
        node=within_chi_node,
        sources=urls if urls else None,
        additional_instruction=(
            "Check the venue's official address on the provided page(s). It must clearly indicate 'Chicago, IL'. "
            "If it shows another municipality such as Rosemont, Tinley Park, Evanston, or similar, it is NOT within Chicago city limits."
        ),
    )

    # 5) Indoor concert venue (critical)
    indoor_node = evaluator.add_leaf(
        id=f"venue_{index + 1}_indoor_concert_venue",
        desc="Venue is an indoor concert/live music venue",
        parent=vnode,
        critical=True,
    )
    indoor_claim = (
        f"The venue '{venue.name or 'the venue'}' is an indoor concert or live music venue (not an outdoor amphitheatre/pavilion)."
    )
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_node,
        sources=urls if urls else None,
        additional_instruction=(
            "Look for cues like 'theatre', 'arena', 'auditorium', 'indoor', or seating maps that are clearly indoors. "
            "Venues described as amphitheatre, pavilion, or outdoor lawn are NOT indoor."
        ),
    )

    # 6) Capacity in required range (critical)
    cap_range_node = evaluator.add_leaf(
        id=f"venue_{index + 1}_capacity_in_range",
        desc="Venue seating capacity is between 3,000 and 8,000 seats",
        parent=vnode,
        critical=True,
    )
    cap_range_claim = (
        f"The seating capacity of '{venue.name or 'the venue'}' is between 3,000 and 8,000 seats."
    )
    await evaluator.verify(
        claim=cap_range_claim,
        node=cap_range_node,
        sources=urls if urls else None,
        additional_instruction=(
            "Use the seating capacity figure shown on the provided page(s). "
            "If multiple capacities are listed (e.g., different configurations), use the typical seated capacity for concerts. "
            "Confirm that the capacity lies inclusively between 3,000 and 8,000."
        ),
    )

    # 7) ADA: at least 1% wheelchair-accessible seating (critical)
    ada_min_node = evaluator.add_leaf(
        id=f"venue_{index + 1}_ada_wheelchair_minimum_percent",
        desc="Venue provides at least 1% of total seating as wheelchair-accessible seating locations",
        parent=vnode,
        critical=True,
    )
    ada_min_claim = (
        "The venue provides at least 1% of total seating as wheelchair-accessible seating locations in compliance with ADA requirements."
    )
    await evaluator.verify(
        claim=ada_min_claim,
        node=ada_min_node,
        sources=urls if urls else None,
        additional_instruction=(
            "Check the venue's ADA/accessibility policy or seating information. "
            "If the page explicitly states ADA compliance for wheelchair seating, you may accept it as satisfying the 1% minimum. "
            "If there is no indication of wheelchair seating or ADA compliance, do not support the claim."
        ),
    )

    # 8) ADA: companion seats adjacent (critical)
    ada_comp_node = evaluator.add_leaf(
        id=f"venue_{index + 1}_ada_companion_seats",
        desc="Venue provides companion seats adjacent to wheelchair seating locations",
        parent=vnode,
        critical=True,
    )
    ada_comp_claim = (
        "The venue provides companion seats adjacent to wheelchair seating locations."
    )
    await evaluator.verify(
        claim=ada_comp_claim,
        node=ada_comp_node,
        sources=urls if urls else None,
        additional_instruction=(
            "Look for explicit mentions of companion seating next to wheelchair spaces (e.g., 'each wheelchair space includes adjacent companion seat(s)')."
        ),
    )

    # 9) ADA: dispersed wheelchair seating (critical)
    ada_disp_node = evaluator.add_leaf(
        id=f"venue_{index + 1}_ada_dispersed_wheelchair_locations",
        desc="Venue implements dispersed wheelchair seating across multiple locations in the venue",
        parent=vnode,
        critical=True,
    )
    ada_disp_claim = (
        "The venue implements dispersed wheelchair seating across multiple locations in the venue (e.g., multiple sections/levels/price points)."
    )
    await evaluator.verify(
        claim=ada_disp_claim,
        node=ada_disp_node,
        sources=urls if urls else None,
        additional_instruction=(
            "Accept if the page indicates wheelchair seating is available in multiple sections or levels, "
            "or that wheelchair locations are distributed around the venue."
        ),
    )

    # 10) Operational and hosting live music (critical)
    operational_node = evaluator.add_leaf(
        id=f"venue_{index + 1}_operational_and_hosting_live_music",
        desc="Venue is currently operational and available for hosting live music performances",
        parent=vnode,
        critical=True,
    )
    operational_claim = (
        f"The venue '{venue.name or 'the venue'}' is currently operational and available for hosting live music performances."
    )
    await evaluator.verify(
        claim=operational_claim,
        node=operational_node,
        sources=urls if urls else None,
        additional_instruction=(
            "Evidence includes current event calendars, ticketing/booking pages, or recent announcements indicating active operations. "
            "If the venue appears closed, under indefinite renovation, or without any current events, do not support the claim."
        ),
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
    Evaluate a single answer for the Chicago indoor concert venues task and return a structured result dictionary.
    """
    # Initialize evaluator (root = parallel per rubric)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Keep exactly two venues (pad if needed)
    venues = _first_two_venues(extracted)

    # Build subtrees for two venues
    for i in range(2):
        await verify_single_venue(
            evaluator=evaluator,
            parent_node=root,
            venue=venues[i],
            index=i,
        )

    # Return evaluator summary
    return evaluator.get_summary()