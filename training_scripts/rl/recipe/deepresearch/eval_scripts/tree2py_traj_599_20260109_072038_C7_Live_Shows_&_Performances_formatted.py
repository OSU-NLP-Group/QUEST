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
TASK_ID = "concert_venue_requirements"
TASK_DESCRIPTION = """
Identify a major indoor concert venue in the United States that meets ALL of the following requirements for hosting large-scale touring concerts:

1. Has a concert/basketball seating capacity between 17,000 and 21,000
2. Is located in a major U.S. city that currently hosts NBA or NHL professional sports teams
3. Offers luxury suites or VIP boxes for premium ticket holders
4. Has parking facilities with capacity for at least 3,000 vehicles
5. Meets ADA accessibility requirements with wheelchair accessible seating
6. Has at least 2 loading dock bays for equipment and production load-in
7. Provides at least 3 artist dressing rooms in the backstage area
8. Has multiple concession stands distributed throughout the facility
9. Has at least one dedicated VIP entrance for premium ticket holders
10. Has multiple emergency exits positioned throughout the facility for safe evacuation
11. Offers club-level seating with premium amenities
12. Has an on-site box office for ticket purchases and will-call services
13. Was either built or underwent major renovation after 1990

Provide the name of the venue, its location (city and state), and its official website URL.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """Information about the venue extracted from the agent's answer."""
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    website_url: Optional[str] = None
    other_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return """
    From the answer, extract the following fields about the proposed concert venue:

    1. venue_name: The full name of the venue.
    2. city: The city where the venue is located.
    3. state: The U.S. state where the venue is located (as a two-letter abbreviation or full name).
    4. website_url: The official website URL of the venue as provided in the answer. If multiple URLs are present, pick the one that appears to be the venue's official site (e.g., the venue's own domain). If no official website is explicitly provided, return null.
    5. other_urls: Extract all other URLs mentioned in the answer that are relevant to the venue (e.g., informational pages, planning guides, premium seating pages). Do not include the official website URL in this list. If none are present, return an empty list.

    Rules:
    - Extract only information explicitly present in the answer.
    - Return null for any missing fields.
    - For URLs, extract valid URLs and include the protocol. If missing, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(extracted: VenueExtraction) -> List[str]:
    sources: List[str] = []
    if extracted.website_url:
        sources.append(extracted.website_url)
    if extracted.other_urls:
        sources.extend(extracted.other_urls)
    return sources


def _venue_display_name(extracted: VenueExtraction) -> str:
    return extracted.venue_name or "the venue"


def _venue_location_str(extracted: VenueExtraction) -> str:
    if extracted.city and extracted.state:
        return f"{extracted.city}, {extracted.state}"
    if extracted.city:
        return extracted.city
    if extracted.state:
        return extracted.state
    return "its location"


async def _add_and_verify_leaf(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    sources: List[str],
    additional_instruction: str,
    critical: bool = True,
) -> None:
    node = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources if sources else None,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_venue_verification_tree(evaluator: Evaluator, extracted: VenueExtraction, parent_node) -> None:
    """
    Build the verification subtree for the concert venue requirements and run verifications.
    All checks under this subtree are critical, as the venue must meet ALL requirements.
    """
    # Create the critical parent node
    venue_node = evaluator.add_parallel(
        id="concert_venue_identification",
        desc="Identify a major concert venue in the United States that meets all specified requirements for hosting large-scale touring concerts",
        parent=parent_node,
        critical=True,
    )

    # Critical existence checks (name, location, website)
    name_ok = bool(extracted.venue_name and extracted.venue_name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id="venue_name_provided",
        desc="The answer provides the name of the venue",
        parent=venue_node,
        critical=True,
    )

    location_ok = bool(extracted.city and extracted.city.strip()) and bool(extracted.state and extracted.state.strip())
    evaluator.add_custom_node(
        result=location_ok,
        id="location_provided",
        desc="The answer provides the venue's location including both city and state",
        parent=venue_node,
        critical=True,
    )

    website_ok = bool(extracted.website_url and extracted.website_url.strip())
    evaluator.add_custom_node(
        result=website_ok,
        id="website_url_provided",
        desc="The answer provides the official website URL for the venue",
        parent=venue_node,
        critical=True,
    )

    # Prepare sources
    sources = _safe_sources(extracted)
    venue_name = _venue_display_name(extracted)
    venue_loc = _venue_location_str(extracted)

    # Seating capacity between 17,000 and 21,000
    await _add_and_verify_leaf(
        evaluator,
        venue_node,
        "seating_capacity",
        "The venue has a concert/basketball seating capacity between 17,000 and 21,000",
        sources,
        additional_instruction=(
            f"Verify using the provided sources whether {venue_name} lists its seating capacity "
            "for basketball or concert configurations within 17,000–21,000. Allow reasonable phrasing "
            "like 'approximately 19,000' or different configurations; confirm an explicit number in that range."
        ),
    )

    # Major city hosting NBA or NHL teams
    await _add_and_verify_leaf(
        evaluator,
        venue_node,
        "major_city_location",
        f"The venue is located in {venue_loc}, a major U.S. city that currently hosts NBA or NHL professional sports teams",
        sources,
        additional_instruction=(
            f"Confirm two things from the provided sources: (1) {venue_name} is in {venue_loc}, and "
            "(2) that city hosts at least one NBA or NHL team. If the sports team information is not present "
            "in any provided source, treat the claim as not supported."
        ),
    )

    # Luxury suites or VIP boxes
    await _add_and_verify_leaf(
        evaluator,
        venue_node,
        "luxury_suites_available",
        f"{venue_name} offers luxury suites or VIP boxes for premium ticket holders",
        sources,
        additional_instruction=(
            "Look for terms such as 'luxury suites', 'VIP boxes', 'executive suites', 'premium suites', "
            "or similar offerings indicating private premium seating areas."
        ),
    )

    # Parking capacity at least 3,000 vehicles
    await _add_and_verify_leaf(
        evaluator,
        venue_node,
        "parking_capacity_sufficient",
        f"{venue_name} has parking facilities with capacity for at least 3,000 vehicles",
        sources,
        additional_instruction=(
            "Check parking pages, event guides, or venue info. The evidence must indicate a numeric capacity "
            "meeting or exceeding 3,000 vehicles (including on-site garages or official lots). "
            "Vague statements like 'ample parking' are insufficient."
        ),
    )

    # ADA compliance
    await _add_and_verify_leaf(
        evaluator,
        venue_node,
        "ada_compliance",
        f"{venue_name} meets ADA accessibility requirements with wheelchair accessible seating",
        sources,
        additional_instruction=(
            "Look for ADA/accessibility policy pages or seating charts explicitly referencing wheelchair accessible seating "
            "and compliance with ADA requirements."
        ),
    )

    # Loading docks at least 2 bays
    await _add_and_verify_leaf(
        evaluator,
        venue_node,
        "loading_dock_facilities",
        f"{venue_name} has at least 2 loading dock bays for equipment and production load-in",
        sources,
        additional_instruction=(
            "Check technical/production specifications or event planner resources for 'loading docks', 'truck bays', "
            "or similar terminology indicating at least two bays."
        ),
    )

    # At least 3 artist dressing rooms
    await _add_and_verify_leaf(
        evaluator,
        venue_node,
        "backstage_dressing_rooms",
        f"{venue_name} provides at least 3 artist dressing rooms in the backstage area",
        sources,
        additional_instruction=(
            "Search production/venue specs for 'dressing rooms', 'green rooms', or 'talent rooms' and confirm the count "
            "is three or more."
        ),
    )

    # Multiple concession stands
    await _add_and_verify_leaf(
        evaluator,
        venue_node,
        "multiple_concession_locations",
        f"{venue_name} has multiple concession stands distributed throughout the facility",
        sources,
        additional_instruction=(
            "Look for 'concessions', 'food & beverage', or 'restaurants' information indicating several stands "
            "across different levels/sections."
        ),
    )

    # Dedicated VIP entrance
    await _add_and_verify_leaf(
        evaluator,
        venue_node,
        "vip_entrance_access",
        f"{venue_name} has at least one dedicated VIP entrance for premium ticket holders",
        sources,
        additional_instruction=(
            "Verify language like 'VIP entrance', 'premium entrance', or dedicated access points for suite/club/VIP guests."
        ),
    )

    # Multiple emergency exits
    await _add_and_verify_leaf(
        evaluator,
        venue_node,
        "emergency_exit_safety",
        f"{venue_name} has multiple emergency exits positioned throughout the facility for safe evacuation",
        sources,
        additional_instruction=(
            "Check building safety, evacuation maps, or policy pages for references to multiple emergency exits. "
            "General stadium/arena architecture typically includes many exits—evidence must explicitly reference exits."
        ),
    )

    # Club-level seating with premium amenities
    await _add_and_verify_leaf(
        evaluator,
        venue_node,
        "club_level_seating",
        f"{venue_name} offers club-level seating with premium amenities",
        sources,
        additional_instruction=(
            "Look for 'club level', 'club seats', 'loge club', or similar premium seating tiers with amenities "
            "such as private lounges, upgraded food/beverage, or dedicated services."
        ),
    )

    # On-site box office
    await _add_and_verify_leaf(
        evaluator,
        venue_node,
        "on_site_box_office",
        f"{venue_name} has an on-site box office for ticket purchases and will-call services",
        sources,
        additional_instruction=(
            "Find references to 'box office', 'ticket office', 'will-call', or similar services located at the venue."
        ),
    )

    # Built or major renovation after 1990
    await _add_and_verify_leaf(
        evaluator,
        venue_node,
        "modern_facility_standard",
        f"{venue_name} was built or underwent major renovation after 1990",
        sources,
        additional_instruction=(
            "Confirm the venue's opening year or a documented major renovation date post-1990. "
            "Accept phrasing like 'opened in 1994' or 'major renovation completed in 2008'."
        ),
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
    Evaluate an answer for the concert venue requirements task.
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

    # Extract core venue info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Build verification tree and run checks
    await build_venue_verification_tree(evaluator, extracted, root)

    # Return structured summary
    return evaluator.get_summary()