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
TASK_ID = "hamilton_mo_venue_feb2026"
TASK_DESCRIPTION = (
    "As of February 26, 2026, identify the name of the venue in Missouri that is hosting the Hamilton national "
    "touring production during a performance period that begins in February 2026 and has a seating capacity of less than 2,500 seats."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    # Core identification
    venue_name: Optional[str] = None
    show_name: Optional[str] = None

    # Performance timing at this venue
    performance_start_date: Optional[str] = None
    date_urls: List[str] = Field(default_factory=list)

    # Tour / show schedule sources (official tour site or official event pages)
    tour_urls: List[str] = Field(default_factory=list)

    # Venue-related sources
    venue_urls: List[str] = Field(default_factory=list)

    # Location verification
    location_state: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)

    # Seating capacity verification
    seating_capacity_text: Optional[str] = None
    capacity_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    Extract the venue identification details exactly as stated in the answer.

    You must extract the following fields:
    1. venue_name: The name of the venue identified as hosting Hamilton.
    2. show_name: The show name as stated in the answer (e.g., "Hamilton", "Hamilton – National Tour").
    3. performance_start_date: The start date of the performance period at that venue, as given in the answer
       (e.g., "February 10, 2026" or "Feb 10, 2026"). If the answer only mentions a range (e.g., "Feb 10–22, 2026"),
       extract the earliest (starting) date verbatim as written.
    4. tour_urls: All URLs cited in the answer that point to the official Hamilton tour schedule or official event
       listing pages that confirm the Hamilton tour stop at the venue (e.g., hamiltonmusical.com/tour, the official venue
       event page, Ticketmaster event page). List all such URLs mentioned.
    5. date_urls: All URLs cited that specifically support the performance dates at the venue. Often these may overlap
       with the tour_urls; include them here too if they support the dates.
    6. venue_urls: All URLs cited that point to the venue’s official site or an authoritative venue page that is cited in the answer.
    7. location_state: The state of the venue as directly stated in the answer (e.g., "Missouri", "MO"), if present.
    8. location_urls: All URLs cited that support the venue’s location (could be the venue's official site, Wikipedia, etc.).
    9. seating_capacity_text: The seating capacity statement (e.g., "2,300 seats") as written in the answer for the relevant hall/space.
    10. capacity_urls: All URLs cited that support the seating capacity information.

    IMPORTANT:
    - Only extract URLs explicitly present in the answer. Do not invent any.
    - Include all relevant URLs mentioned in the answer for each category (tour_urls, date_urls, venue_urls, location_urls, capacity_urls).
    - If a field is not mentioned in the answer, return null for strings and an empty array for lists.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def dedup_urls(urls: List[str]) -> List[str]:
    """Deduplicate URLs while preserving order and filtering out empty entries."""
    seen = set()
    result: List[str] = []
    for u in urls or []:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, parent_node, info: VenueExtraction) -> None:
    """
    Build verification nodes based on the rubric:
    venue_identification (critical, parallel)
      ├─ show_verification (critical leaf)
      ├─ temporal_constraint (critical leaf)
      └─ venue_specifications (critical, parallel)
           ├─ geographic_requirement (critical leaf)
           └─ capacity_requirement (critical leaf)
    """
    # Parent critical node: venue_identification
    venue_ident_node = evaluator.add_parallel(
        id="venue_identification",
        desc="The correct venue hosting Hamilton in the specified time period and meeting all constraints is identified",
        parent=parent_node,
        critical=True
    )

    # Prepare common sources
    show_sources = dedup_urls((info.tour_urls or []) + (info.venue_urls or []) + (info.date_urls or []))
    date_sources = dedup_urls((info.date_urls or []) + (info.tour_urls or []) + (info.venue_urls or []))
    location_sources = dedup_urls((info.location_urls or []) + (info.venue_urls or []))
    capacity_sources = dedup_urls((info.capacity_urls or []) + (info.venue_urls or []))

    # Leaf: show_verification
    show_leaf = evaluator.add_leaf(
        id="show_verification",
        desc="The performance is confirmed as the Hamilton national touring production in 2026 with reference URL from official tour schedule",
        parent=venue_ident_node,
        critical=True
    )
    show_claim = (
        f"The provided webpage confirms that the Hamilton (Broadway musical) national touring production "
        f"is scheduled to perform at {info.venue_name or 'the identified venue'} in 2026."
    )
    await evaluator.verify(
        claim=show_claim,
        node=show_leaf,
        sources=show_sources,
        additional_instruction=(
            "Confirm that the page clearly indicates a touring stop of Hamilton (not the resident Broadway run). "
            "Accept authoritative sources such as the official Hamilton tour website, the venue's official event listing, "
            "or Ticketmaster/Eventim pages. Variants like 'Hamilton – National Tour' or 'Hamilton (Tour)' should be treated as the same show."
        )
    )

    # Leaf: temporal_constraint
    temporal_leaf = evaluator.add_leaf(
        id="temporal_constraint",
        desc="The performance period at the identified venue begins in February 2026",
        parent=venue_ident_node,
        critical=True
    )
    temporal_claim = (
        f"The performance period at {info.venue_name or 'the identified venue'} begins in February 2026."
    )
    await evaluator.verify(
        claim=temporal_claim,
        node=temporal_leaf,
        sources=date_sources,
        additional_instruction=(
            "Focus on the first/earliest performance date at the specified venue. "
            "If the schedule shows a date range (e.g., Feb 10–22, 2026), the 'begins' date is the earliest date in that range. "
            "Consider common date formats and accept reasonable month abbreviations. "
            "Pass only if the earliest date is in February 2026."
        )
    )

    # Critical parallel sub-node: venue_specifications
    specs_node = evaluator.add_parallel(
        id="venue_specifications",
        desc="The venue meets both geographic and capacity requirements",
        parent=venue_ident_node,
        critical=True
    )

    # Leaf: geographic_requirement
    geo_leaf = evaluator.add_leaf(
        id="geographic_requirement",
        desc="The venue is located in Missouri",
        parent=specs_node,
        critical=True
    )
    geo_claim = f"The venue {info.venue_name or 'the identified venue'} is located in the U.S. state of Missouri (MO)."
    await evaluator.verify(
        claim=geo_claim,
        node=geo_leaf,
        sources=location_sources,
        additional_instruction=(
            "Verify the venue’s location is in Missouri. Accept forms like 'St. Louis, MO' or 'Kansas City, Missouri'. "
            "If the venue has multiple locations, ensure the specific venue used for the Hamilton stop is in Missouri."
        )
    )

    # Leaf: capacity_requirement
    capacity_leaf = evaluator.add_leaf(
        id="capacity_requirement",
        desc="The venue has a seating capacity of less than 2,500 seats, verified with reference URL",
        parent=specs_node,
        critical=True
    )
    capacity_claim = (
        f"The seating capacity of {info.venue_name or 'the identified venue'} is less than 2,500 seats."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=capacity_sources,
        additional_instruction=(
            "Check the stated seating capacity for the specific hall or theater space where Hamilton is performed. "
            "If multiple spaces exist, ensure the capacity cited corresponds to the main auditorium used for this production. "
            "Accept approximate figures that are clearly under 2,500 (e.g., 'approximately 2,300')."
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
    Evaluate an answer for the Hamilton Missouri venue identification task.
    """
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

    # Extract structured venue info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extracted_info)

    # Return summary with verification tree and scores
    return evaluator.get_summary()