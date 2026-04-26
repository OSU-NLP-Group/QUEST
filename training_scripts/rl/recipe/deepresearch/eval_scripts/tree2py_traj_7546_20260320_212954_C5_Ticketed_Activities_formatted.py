import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "multi_venue_tour_2026"
TASK_DESCRIPTION = """
A touring theatrical production company is planning a 2026 North American tour and needs to book two specific types of venues for major performances.

Identify two venues that meet ALL of the following requirements:

Venue 1 - Chicago Large Indoor Arena:
- Must be a large indoor arena located in Chicago, Illinois
- Must have a concert seating capacity of at least 15,000
- Must provide the actual concert seating capacity number
- Must meet ADA accessibility requirements: provide wheelchair accessible seating for at least 1% of total capacity, with adjacent companion seats for each wheelchair space
- Include a reference URL supporting the venue information

Venue 2 - New York Broadway Theater:
- Must be a Broadway theater (minimum 500 seats) located in Manhattan's Theater District (between 41st Street and 54th Street, and between 6th Avenue and 9th Avenue)
- Must provide the actual seating capacity number
- Must meet ADA accessibility requirements: provide wheelchair accessible seating for at least 1% of total capacity, with adjacent companion seats for each wheelchair space
- Include a reference URL supporting the venue information

For each venue, provide: the venue name, actual seating capacity, location confirmation, verification of ADA accessibility compliance, and a supporting reference URL.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    """Structured info for a single venue as stated in the answer."""
    name: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string to handle ranges/formatting; we'll verify via URLs
    location: Optional[str] = None  # City/State or narrative location from the answer
    address: Optional[str] = None   # Street address if provided
    ada_wheelchair: Optional[str] = None  # Any phrase about wheelchair seating from the answer
    ada_companion: Optional[str] = None   # Any phrase about companion seating from the answer
    sources: List[str] = Field(default_factory=list)  # URLs explicitly cited in the answer


class TourVenuesExtraction(BaseModel):
    """Top-level extraction for both required venues."""
    chicago_arena: Optional[VenueItem] = None
    ny_broadway: Optional[VenueItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract exactly two venues as presented in the answer text, one for each category below. Do NOT invent or infer information not explicitly present. If an item is missing in the answer, set it to null (or empty list for sources).

1) chicago_arena:
   - name: The exact venue name for the large indoor arena in Chicago, IL.
   - capacity: The specific seating capacity number for CONCERTS as stated in the answer. If multiple capacities are listed, prefer the concert configuration; if only a single total capacity is given, extract that.
   - location: The location string as written in the answer (e.g., "Chicago, Illinois", "Chicago, IL").
   - address: The street address if stated (e.g., "1901 W Madison St, Chicago, IL 60612").
   - ada_wheelchair: Any ADA or accessibility text from the answer that mentions wheelchair seating or quantities/percentages.
   - ada_companion: Any ADA text from the answer that mentions companion seats being adjacent to wheelchair spaces.
   - sources: All URLs explicitly cited in the answer that support this venue. Include only valid URLs; if none are present, return an empty list.

2) ny_broadway:
   - name: The exact venue name for the Broadway theater in Manhattan.
   - capacity: The specific seating capacity number as stated in the answer (Broadway theaters have >=500 seats).
   - location: The location string as written in the answer (e.g., "New York, NY", "Manhattan").
   - address: The street address if stated (ideally shows a street between 41st–54th St, and avenues 6th–9th).
   - ada_wheelchair: Any ADA or accessibility text from the answer that mentions wheelchair seating or quantities/percentages.
   - ada_companion: Any ADA text from the answer that mentions companion seats adjacent to wheelchair spaces.
   - sources: All URLs explicitly cited in the answer that support this venue. Include only valid URLs; if none are present, return an empty list.

Special rules for URLs:
- Extract only URLs that appear in the answer (including markdown links).
- Do not infer or create URLs.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def contains_number(s: Optional[str]) -> bool:
    if not s:
        return False
    return bool(re.search(r"\d", s))


def first_nonempty_str(*vals: Optional[str]) -> str:
    for v in vals:
        if v and str(v).strip():
            return str(v).strip()
    return ""


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_chicago_arena(evaluator: Evaluator, parent_node, venue: Optional[VenueItem]) -> None:
    """
    Build and verify the Chicago Large Indoor Arena subtree.

    Structure mirrors rubric:
    - Arena Identification (critical)
    - Capacity Verification (critical, parallel)
        - Meets 15,000 Minimum Capacity (critical)
        - Actual Capacity Stated (critical)
    - Chicago Location Confirmed (critical)
    - ADA Compliance (critical, parallel)
        - Minimum 1% Wheelchair Seating (critical)
        - Companion Seats Adjacent (critical)
    - Reference URL (critical)
    """
    node = evaluator.add_parallel(
        id="chicago_arena_venue",
        desc="Evaluation of large indoor arena venue in Chicago meeting capacity and accessibility requirements",
        parent=parent_node,
        critical=False
    )

    name = venue.name.strip() if venue and venue.name else ""
    capacity_text = venue.capacity if venue else None
    sources = venue.sources if venue else []

    # 1) Arena Identification (critical)
    evaluator.add_custom_node(
        result=bool(name),
        id="arena_identification",
        desc="Specific arena venue in Chicago is identified by name",
        parent=node,
        critical=True
    )

    # 2) Capacity Verification (critical, parallel)
    cap_node = evaluator.add_parallel(
        id="arena_capacity_verification",
        desc="Arena capacity meets minimum requirements and actual capacity is documented",
        parent=node,
        critical=True
    )

    # 2.a) Actual Capacity Stated (critical) - existence check in the answer
    evaluator.add_custom_node(
        result=contains_number(capacity_text),
        id="arena_actual_capacity_stated",
        desc="Specific concert seating capacity number is provided",
        parent=cap_node,
        critical=True
    )

    # 2.b) Meets 15,000 Minimum Capacity (critical) - verify via sources
    meets_min_leaf = evaluator.add_leaf(
        id="arena_meets_15000_min",
        desc="Arena has seating capacity of at least 15,000 for concerts",
        parent=cap_node,
        critical=True
    )
    capacity_phrase = capacity_text if capacity_text else "the stated concert seating capacity"
    claim_min_capacity = (
        f"The concert seating capacity of the venue '{name}' is {capacity_phrase}, which is at least 15,000."
        if capacity_text else
        f"The concert seating capacity of the venue '{name}' is at least 15,000."
    )
    await evaluator.verify(
        claim=claim_min_capacity,
        node=meets_min_leaf,
        sources=sources,
        additional_instruction=(
            "Verify from the provided page(s) that the venue's concert (or maximum event) seating capacity is at least 15,000. "
            "If multiple configurations are listed (e.g., basketball, hockey, concerts), prefer the concert or maximum configuration. "
            "If the page does not explicitly support ≥ 15,000, mark as not supported."
        )
    )

    # 3) Chicago Location Confirmed (critical)
    chicago_loc_leaf = evaluator.add_leaf(
        id="arena_chicago_location",
        desc="Venue is confirmed to be located in Chicago, Illinois",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{name}' is located in Chicago, Illinois.",
        node=chicago_loc_leaf,
        sources=sources,
        additional_instruction=(
            "Use the address or location information on the page to confirm the venue is in Chicago, IL. "
            "Minor formatting differences (e.g., 'Chicago, IL') are acceptable."
        )
    )

    # 4) ADA Compliance (critical, parallel)
    ada_node = evaluator.add_parallel(
        id="arena_ada_compliance",
        desc="Venue meets ADA accessibility requirements for wheelchair and companion seating",
        parent=node,
        critical=True
    )

    # 4.a) Minimum 1% Wheelchair Seating (critical)
    ada_wheel_leaf = evaluator.add_leaf(
        id="arena_minimum_1pct_wheelchair",
        desc="Venue provides wheelchair accessible seating for at least 1% of total capacity",
        parent=ada_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{name}' provides wheelchair accessible seating for at least 1% of its total seating capacity.",
        node=ada_wheel_leaf,
        sources=sources,
        additional_instruction=(
            "Look for explicit statements or data indicating that wheelchair seating is provided at ≥1% of total capacity; "
            "this may be a ratio, percentage, or a minimum number of wheelchair locations relative to capacity stated on the page. "
            "If the page specifies both total capacity and number of wheelchair spaces, evaluate whether spaces ≥ 1% of capacity. "
            "Generic statements like 'ADA compliant' without numeric support should be treated as not supported."
        )
    )

    # 4.b) Companion Seats Adjacent (critical)
    ada_companion_leaf = evaluator.add_leaf(
        id="arena_companion_adjacent",
        desc="Each wheelchair space has an adjacent companion seat",
        parent=ada_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Each wheelchair space at the venue '{name}' has an adjacent companion seat.",
        node=ada_companion_leaf,
        sources=sources,
        additional_instruction=(
            "Verify from the page that companion seats are provided adjacent to (i.e., next to) each wheelchair space. "
            "If adjacency is not explicitly stated, do not assume; treat as not supported."
        )
    )

    # 5) Reference URL (critical)
    ref_url_leaf = evaluator.add_leaf(
        id="arena_reference_url",
        desc="Valid reference URL provided supporting venue information",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided page is an authoritative or relevant page about the venue '{name}' in Chicago (e.g., official site, venue page, reputable reference).",
        node=ref_url_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that at least one provided URL is a valid page focused on the specified venue "
            "(official venue website, operator's page, a reputable database, or Wikipedia). "
            "If no URLs are provided or the URL is unrelated, mark as not supported."
        )
    )


async def verify_ny_broadway(evaluator: Evaluator, parent_node, venue: Optional[VenueItem]) -> None:
    """
    Build and verify the New York Broadway Theater subtree.

    Structure mirrors rubric:
    - Theater Identification (critical)
    - Capacity Verification (critical, parallel)
        - Meets 500 Seat Minimum (critical)
        - Actual Capacity Stated (critical)
    - Theater District Location (critical)
    - ADA Compliance (critical, parallel)
        - Minimum 1% Wheelchair Seating (critical)
        - Companion Seats Adjacent (critical)
    - Reference URL (critical)
    """
    node = evaluator.add_parallel(
        id="ny_broadway_venue",
        desc="Evaluation of Broadway theater venue in New York meeting capacity and accessibility requirements",
        parent=parent_node,
        critical=False
    )

    name = venue.name.strip() if venue and venue.name else ""
    capacity_text = venue.capacity if venue else None
    sources = venue.sources if venue else []

    # 1) Theater Identification (critical)
    evaluator.add_custom_node(
        result=bool(name),
        id="theater_identification",
        desc="Specific Broadway theater in New York is identified by name",
        parent=node,
        critical=True
    )

    # 2) Capacity Verification (critical, parallel)
    cap_node = evaluator.add_parallel(
        id="theater_capacity_verification",
        desc="Theater capacity meets minimum requirements and actual capacity is documented",
        parent=node,
        critical=True
    )

    # 2.a) Actual Capacity Stated (critical)
    evaluator.add_custom_node(
        result=contains_number(capacity_text),
        id="theater_actual_capacity_stated",
        desc="Specific seating capacity number is provided",
        parent=cap_node,
        critical=True
    )

    # 2.b) Meets 500 Seat Minimum (critical) - verify via sources
    meets_min_leaf = evaluator.add_leaf(
        id="theater_meets_500_min",
        desc="Theater has seating capacity of at least 500 seats (Broadway theater requirement)",
        parent=cap_node,
        critical=True
    )
    capacity_phrase = capacity_text if capacity_text else "the stated seating capacity"
    claim_min_capacity = (
        f"The seating capacity of the theater '{name}' is {capacity_phrase}, which is at least 500."
        if capacity_text else
        f"The seating capacity of the theater '{name}' is at least 500."
    )
    await evaluator.verify(
        claim=claim_min_capacity,
        node=meets_min_leaf,
        sources=sources,
        additional_instruction=(
            "Verify from the provided page(s) that this theater has ≥500 seats (Broadway threshold). "
            "If multiple capacities are listed (or different configurations), use the standard seating capacity. "
            "If the page does not support ≥500, mark as not supported."
        )
    )

    # 3) Theater District Location (critical)
    district_leaf = evaluator.add_leaf(
        id="theater_district_location",
        desc="Theater is located within the Theater District boundaries (41st-54th Street, 6th-9th Avenue, Manhattan)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The theater '{name}' is located within Manhattan's Theater District boundaries "
            f"(between 41st and 54th Street, and between 6th and 9th Avenue)."
        ),
        node=district_leaf,
        sources=sources,
        additional_instruction=(
            "Use the address on the page to determine whether it falls within W 41st–W 54th Streets and 6th–9th Avenues in Manhattan. "
            "If the page explicitly states 'Theater District' or clearly shows an address (e.g., W 45th St between 7th & 8th Ave), "
            "treat this as within the boundary. If insufficient information is present, mark as not supported."
        )
    )

    # 4) ADA Compliance (critical, parallel)
    ada_node = evaluator.add_parallel(
        id="theater_ada_compliance",
        desc="Theater meets ADA accessibility requirements for wheelchair and companion seating",
        parent=node,
        critical=True
    )

    # 4.a) Minimum 1% Wheelchair Seating (critical)
    ada_wheel_leaf = evaluator.add_leaf(
        id="theater_minimum_1pct_wheelchair",
        desc="Theater provides wheelchair accessible seating for at least 1% of total capacity",
        parent=ada_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater '{name}' provides wheelchair accessible seating for at least 1% of its total seating capacity.",
        node=ada_wheel_leaf,
        sources=sources,
        additional_instruction=(
            "Look for explicit statements or figures indicating wheelchair seating ≥1% of total capacity. "
            "If both capacity and wheelchair seat counts are provided, check the ratio. "
            "Generic statements like 'ADA compliant' without quantitative support should be treated as not supported."
        )
    )

    # 4.b) Companion Seats Adjacent (critical)
    ada_companion_leaf = evaluator.add_leaf(
        id="theater_companion_adjacent",
        desc="Each wheelchair space has an adjacent companion seat",
        parent=ada_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"Each wheelchair space at the theater '{name}' has an adjacent companion seat.",
        node=ada_companion_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the page states companion seats are provided adjacent to each wheelchair space. "
            "If proximity/adjacency isn't explicitly stated, do not assume."
        )
    )

    # 5) Reference URL (critical)
    ref_url_leaf = evaluator.add_leaf(
        id="theater_reference_url",
        desc="Valid reference URL provided supporting venue information",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided page is an authoritative or relevant page about the theater '{name}' in Manhattan (e.g., official site, reputable source).",
        node=ref_url_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that at least one provided URL is a valid page focused on the specified theater "
            "(official website, operator's page, reputable database, or Wikipedia). "
            "If no URLs are provided or the URL is unrelated, mark as not supported."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the multi-venue tour requirements task and return a structured summary.
    """
    # Initialize evaluator with a parallel aggregation at the root
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

    # Extract structured info for both venues
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=TourVenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Build two parallel subtrees: Chicago Arena + NY Broadway Theater
    # Chicago Arena
    await verify_chicago_arena(
        evaluator=evaluator,
        parent_node=root,
        venue=extraction.chicago_arena if extraction else None
    )

    # New York Broadway Theater
    await verify_ny_broadway(
        evaluator=evaluator,
        parent_node=root,
        venue=extraction.ny_broadway if extraction else None
    )

    # Return the evaluation summary
    return evaluator.get_summary()