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
TASK_ID = "ct_indoor_arenas"
TASK_DESCRIPTION = """
Identify three indoor arena venues located in Connecticut that meet all of the following requirements for hosting a touring concert production:

1. The venue must be an indoor arena facility suitable for live concert performances
2. The venue's seating capacity must be between 7,000 and 12,000 seats (inclusive)
3. The venue must have loading dock facilities for equipment load-in
4. The venue's stage must have a depth of at least 16 feet to accommodate performance equipment
5. The venue must provide ADA-compliant wheelchair accessible seating
6. The venue must have box office facilities for ticket sales operations
7. The venue must have parking facilities available for event attendees

For each venue, provide the venue name, a brief description confirming it meets all requirements, and an official reference URL (website or documentation) that supports your answer.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract the venues listed in the answer. For each distinct venue, return:
    - name: the venue name exactly as written in the answer
    - reference_urls: all URLs explicitly cited in the answer that support information about this venue (e.g., official website pages, venue technical specs PDFs, event/guest services pages). Only include URLs that appear in the answer text. Do not invent any URL.

    Important:
    - If the answer lists more than three venues, extract all; evaluation will use the first three.
    - If a URL is shown as a markdown link, extract the underlying URL.
    - If no URL is provided for a venue, set reference_urls to an empty list.

    Return a JSON object { "venues": [ { "name": ..., "reference_urls": [...] }, ... ] }.
    """.strip()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _clean_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    deduped = []
    seen = set()
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        # basic protocol fix handled by Extractor rules if missing; keep as-is here
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def _ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][n] if 0 <= n < 5 else f"#{n+1}"


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    index: int,
) -> None:
    """
    Build verification subtree and run checks for a single venue.
    """
    display_name = venue.name or f"Venue #{index + 1}"
    urls = _clean_urls(venue.reference_urls)

    # Venue container (non-critical; allows partial credit across venues)
    venue_node = evaluator.add_parallel(
        id=f"venue_{index}",
        desc=f"{_ordinal(index)} qualifying arena venue in Connecticut",
        parent=parent_node,
        critical=False,
    )

    # Critical reference URL presence gate (used as precondition for all other checks)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"venue_{index}_reference_url",
        desc="Provide official website or documentation URL for the venue",
        parent=venue_node,
        critical=True,
    )

    # Location and Type (critical group)
    loc_type_node = evaluator.add_parallel(
        id=f"venue_{index}_location_and_type",
        desc="Verify venue location and type specifications",
        parent=venue_node,
        critical=True,
    )

    ct_loc_leaf = evaluator.add_leaf(
        id=f"venue_{index}_connecticut_location",
        desc="Venue is located in Connecticut",
        parent=loc_type_node,
        critical=True,
    )
    indoor_arena_leaf = evaluator.add_leaf(
        id=f"venue_{index}_indoor_arena",
        desc="Venue is an indoor arena facility suitable for concerts",
        parent=loc_type_node,
        critical=True,
    )

    # Capacity and Technical (critical group)
    cap_tech_node = evaluator.add_parallel(
        id=f"venue_{index}_capacity_and_technical",
        desc="Verify seating capacity and technical specifications",
        parent=venue_node,
        critical=True,
    )

    capacity_leaf = evaluator.add_leaf(
        id=f"venue_{index}_seating_capacity_range",
        desc="Venue seating capacity is between 7,000 and 12,000 seats inclusive",
        parent=cap_tech_node,
        critical=True,
    )
    loading_dock_leaf = evaluator.add_leaf(
        id=f"venue_{index}_loading_dock",
        desc="Venue has loading dock facilities for equipment load-in",
        parent=cap_tech_node,
        critical=True,
    )
    stage_depth_leaf = evaluator.add_leaf(
        id=f"venue_{index}_stage_depth",
        desc="Venue stage depth is at least 16 feet",
        parent=cap_tech_node,
        critical=True,
    )

    # Operations and Accessibility (critical group)
    ops_acc_node = evaluator.add_parallel(
        id=f"venue_{index}_operations_and_accessibility",
        desc="Verify operational features and accessibility compliance",
        parent=venue_node,
        critical=True,
    )

    ada_leaf = evaluator.add_leaf(
        id=f"venue_{index}_ada_wheelchair_seating",
        desc="Venue provides ADA-compliant wheelchair accessible seating",
        parent=ops_acc_node,
        critical=True,
    )
    box_office_leaf = evaluator.add_leaf(
        id=f"venue_{index}_box_office",
        desc="Venue has box office facilities for ticket sales",
        parent=ops_acc_node,
        critical=True,
    )
    parking_leaf = evaluator.add_leaf(
        id=f"venue_{index}_parking_facilities",
        desc="Venue has parking facilities available for attendees",
        parent=ops_acc_node,
        critical=True,
    )

    # Assemble claims for batch verification
    claims_and_sources = [
        (
            f"The venue named '{display_name}' is located in the U.S. state of Connecticut.",
            urls,
            ct_loc_leaf,
            "Allow CT abbreviations (e.g., 'CT', 'Conn.'). Use the venue's About/Contact footer or address on the page to confirm the state is Connecticut.",
        ),
        (
            f"'{display_name}' is an indoor arena (an enclosed facility) suitable for hosting concerts or live music performances.",
            urls,
            indoor_arena_leaf,
            "Confirm it is an indoor arena or multipurpose indoor venue. Evidence can include phrases like 'indoor arena', 'multi-purpose arena', 'hosts concerts/shows', or a calendar listing concerts.",
        ),
        (
            f"'{display_name}' has a total seating capacity between 7,000 and 12,000 seats inclusive.",
            urls,
            capacity_leaf,
            "Use published capacities (overall, basketball/hockey, end-stage, center-stage). If multiple capacities are given, prefer the maximum seats for concerts. Accept approximate wordings (e.g., 'about 10,000').",
        ),
        (
            f"'{display_name}' provides loading dock facilities for production equipment load-in.",
            urls,
            loading_dock_leaf,
            "Look for 'loading dock', 'truck bays', 'production/load-in', 'freight access', or similar terms in technical specs, event services, or venue guides.",
        ),
        (
            f"The stage depth at '{display_name}' is at least 16 feet (>= 4.9 meters).",
            urls,
            stage_depth_leaf,
            "Look for stage dimensions (e.g., 'stage 60' W x 40' D'). Consider 'depth', 'upstage-downstage', or metric equivalents (>= 4.9 m).",
        ),
        (
            f"'{display_name}' provides ADA-compliant wheelchair accessible seating for patrons.",
            urls,
            ada_leaf,
            "Search for 'ADA', 'accessible seating', 'wheelchair', or accessibility policy pages. Venue policies, seating charts, or FAQs are acceptable evidence.",
        ),
        (
            f"'{display_name}' has box office facilities for ticket sales operations.",
            urls,
            box_office_leaf,
            "Accept 'box office', 'ticket office', 'ticket windows', or 'box office hours'. Ticketing information pages on the official site count.",
        ),
        (
            f"'{display_name}' has parking facilities available for event attendees.",
            urls,
            parking_leaf,
            "Look for 'parking', 'garage', 'surface lot', 'on-site parking', or official partner parking details on the venue or campus site.",
        ),
    ]

    # Run verifications in parallel (auto-preconditions will skip when Reference_URL fails)
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the Connecticut indoor arenas task.
    """
    # Initialize evaluator (root is parallel per rubric)
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

    # Extract venue list from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Take first three venues; pad with empty items if fewer than three
    venues = (extracted.venues or [])[:3]
    while len(venues) < 3:
        venues.append(VenueItem())

    # Build verification subtrees for each venue
    for idx, venue in enumerate(venues[:3]):
        await verify_venue(evaluator, root, venue, idx)

    # Include a summary of constraints as custom info
    evaluator.add_custom_info(
        info={
            "constraints": {
                "state": "Connecticut",
                "venue_type": "Indoor arena suitable for concerts",
                "seating_capacity_range": "[7000, 12000]",
                "loading_dock": True,
                "stage_depth_ft_min": 16,
                "ada_wheelchair_seating": True,
                "box_office": True,
                "parking": True,
            }
        },
        info_type="requirements",
        info_name="venue_requirements",
    )

    return evaluator.get_summary()