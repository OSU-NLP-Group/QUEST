import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bocelli_2026_venues"
TASK_DESCRIPTION = """Andrea Bocelli is planning his 2026 North American concert tour and needs to book major arena venues in four specific US cities. For each of the following cities, identify the concert venue that meets ALL of these requirements:

1. Boston, Massachusetts
2. Chicago, Illinois
3. New York City, New York
4. Seattle, Washington

Each venue must satisfy the following criteria:
- Has a minimum seating capacity of 17,000 for concerts
- Provides wheelchair accessible seating on multiple levels (not just ground level)
- Has Andrea Bocelli scheduled to perform there during his 2026 tour

For each venue, provide:
- The complete venue name
- The confirmed seating capacity
- Confirmation of multi-level accessibility features
- The scheduled Andrea Bocelli performance date in 2026
- A reference URL verifying this information
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueDetails(BaseModel):
    """Information about one city's venue."""
    venue_name: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string to allow ranges or formatted numbers
    accessibility: Optional[str] = None  # Free-text confirmation/details from the answer
    tour_date_2026: Optional[str] = None  # The scheduled performance date string (should be year 2026)
    reference_urls: List[str] = Field(default_factory=list)  # URLs cited in the answer to verify info


class VenuesExtraction(BaseModel):
    """Extraction for the four specified cities."""
    boston: Optional[VenueDetails] = None
    chicago: Optional[VenueDetails] = None
    new_york_city: Optional[VenueDetails] = None
    seattle: Optional[VenueDetails] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract the venue information the answer provides for the following cities:
    - Boston, Massachusetts (key: "boston")
    - Chicago, Illinois (key: "chicago")
    - New York City, New York (key: "new_york_city")
    - Seattle, Washington (key: "seattle")

    For each city, extract these fields from the answer EXACTLY as presented:
    1) venue_name: The full official venue name for Andrea Bocelli's 2026 tour stop in that city.
    2) capacity: The confirmed seating capacity for concerts (string, can include commas or ranges; extract verbatim).
    3) accessibility: A textual confirmation that wheelchair accessible seating is available on multiple levels (not just ground level). Extract the precise phrasing or summary provided.
    4) tour_date_2026: The scheduled Andrea Bocelli performance date in 2026 for that venue (string as shown).
    5) reference_urls: An array of one or more URLs explicitly mentioned in the answer that substantiate the above information. Only include valid URLs that appear in the answer.

    If a field is missing for a city, set it to null (for strings) or an empty array for reference_urls.
    Return a JSON object with keys: boston, chicago, new_york_city, seattle, each mapping to an object of the above fields.
    Do NOT invent any values. Only extract what is actually in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe(val: Optional[str]) -> str:
    return val or ""


def _city_display_name(city_key: str) -> str:
    mapping = {
        "boston": "Boston",
        "chicago": "Chicago",
        "new_york_city": "New York City",
        "seattle": "Seattle",
    }
    return mapping.get(city_key, city_key)


def _get_city_node_id(city_key: str) -> str:
    mapping = {
        "boston": "boston_venue",
        "chicago": "chicago_venue",
        "new_york_city": "new_york_venue",
        "seattle": "seattle_venue",
    }
    return mapping[city_key]


def _get_leaf_ids(city_key: str) -> Dict[str, str]:
    mapping = {
        "boston": {
            "capacity": "boston_capacity",
            "accessibility": "boston_accessibility",
            "tour_date": "boston_tour_date",
            "reference": "boston_reference",
        },
        "chicago": {
            "capacity": "chicago_capacity",
            "accessibility": "chicago_accessibility",
            "tour_date": "chicago_tour_date",
            "reference": "chicago_reference",
        },
        "new_york_city": {
            "capacity": "nyc_capacity",
            "accessibility": "nyc_accessibility",
            "tour_date": "nyc_tour_date",
            "reference": "nyc_reference",
        },
        "seattle": {
            "capacity": "seattle_capacity",
            "accessibility": "seattle_accessibility",
            "tour_date": "seattle_tour_date",
            "reference": "seattle_reference",
        },
    }
    return mapping[city_key]


# --------------------------------------------------------------------------- #
# Verification per city                                                       #
# --------------------------------------------------------------------------- #
async def verify_city_venue(
    evaluator: Evaluator,
    parent_node,
    city_key: str,
    city_data: Optional[VenueDetails],
) -> None:
    """
    Build and verify the subtree for a single city's venue.

    Structure per city (parallel aggregation, non-critical parent, critical leaves):
      - capacity (critical): verify concert capacity >= 17,000 via URLs
      - accessibility (critical): verify wheelchair accessible seating on multiple levels via URLs
      - tour_date (critical): verify Bocelli scheduled in 2026 and date via URLs
      - reference (critical): existence of at least one reference URL (custom node)
    """
    city_display = _city_display_name(city_key)
    node_id = _get_city_node_id(city_key)
    leaf_ids = _get_leaf_ids(city_key)

    # Create city parent node
    city_node = evaluator.add_parallel(
        id=node_id,
        desc=f"Identify the venue in {city_display} that meets all requirements",
        parent=parent_node,
        critical=False,  # Allow partial scoring across cities
    )

    # Normalize data to avoid None
    data = city_data or VenueDetails()
    urls = data.reference_urls or []

    # Reference existence (critical) - added first to act as a gating prerequisite
    evaluator.add_custom_node(
        result=bool(urls),
        id=leaf_ids["reference"],
        desc=f"Provide URL reference confirming the {city_display} venue information",
        parent=city_node,
        critical=True,
    )

    # Capacity verification leaf (critical)
    capacity_leaf = evaluator.add_leaf(
        id=leaf_ids["capacity"],
        desc=f"The {city_display} venue has a minimum capacity of 17,000 seats",
        parent=city_node,
        critical=True,
    )
    capacity_claim = (
        f"The concert seating capacity of {_safe(data.venue_name)} is at least 17,000 seats."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=urls,
        additional_instruction=(
            "Use the provided URL(s) to confirm the venue's concert seating capacity. "
            "If the page lists a numeric capacity (e.g., 18,000 or 19,500), that supports the claim of 'at least 17,000'. "
            "Prefer official venue pages or authoritative sources. Capacity may differ by event configuration; "
            "accept the standard concert seating capacity when clearly stated."
        ),
    )

    # Accessibility verification leaf (critical)
    accessibility_leaf = evaluator.add_leaf(
        id=leaf_ids["accessibility"],
        desc=f"The {city_display} venue provides wheelchair accessible seating on multiple levels",
        parent=city_node,
        critical=True,
    )
    accessibility_claim = (
        f"The venue {_safe(data.venue_name)} provides wheelchair accessible seating on multiple levels (not just ground level)."
    )
    await evaluator.verify(
        claim=accessibility_claim,
        node=accessibility_leaf,
        sources=urls,
        additional_instruction=(
            "Check the venue's accessibility/ADA page or official seating policies. "
            "Support the claim specifically if the page indicates accessible seating or companion seating is available "
            "in multiple sections/levels (e.g., lower bowl and upper levels), not restricted to just one level."
        ),
    )

    # Tour date verification leaf (critical)
    tour_leaf = evaluator.add_leaf(
        id=leaf_ids["tour_date"],
        desc=f"The {city_display} venue has Andrea Bocelli scheduled to perform during his 2026 tour",
        parent=city_node,
        critical=True,
    )
    tour_date_text = _safe(data.tour_date_2026)
    tour_claim = (
        f"Andrea Bocelli is scheduled to perform at {_safe(data.venue_name)} in 2026 on {tour_date_text}."
        if tour_date_text
        else f"Andrea Bocelli is scheduled to perform at {_safe(data.venue_name)} in 2026."
    )
    await evaluator.verify(
        claim=tour_claim,
        node=tour_leaf,
        sources=urls,
        additional_instruction=(
            "Verify on the artist's official tour page, venue event listing, or trusted ticketing site that "
            "Andrea Bocelli has a scheduled performance at this venue in calendar year 2026. "
            "Minor variations in date formatting are acceptable, but the year must be 2026."
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
    Evaluate an answer for Andrea Bocelli's 2026 venues in Boston, Chicago, NYC, and Seattle.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root combines cities independently
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify suitable concert venues in specified US cities that meet all requirements for hosting Andrea Bocelli's 2026 tour",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Note: Root is initialized as non-critical by design in Evaluator.initialize(),
    # which avoids the critical-parent constraint and allows partial credit across cities.

    # Extract structured venue info from the answer
    venues_info = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Optional: record target cities into custom info for summary
    evaluator.add_custom_info(
        info={"target_cities": ["Boston, MA", "Chicago, IL", "New York City, NY", "Seattle, WA"]},
        info_type="task_metadata",
        info_name="target_cities",
    )

    # Build verification subtrees for each city
    await verify_city_venue(evaluator, root, "boston", venues_info.boston)
    await verify_city_venue(evaluator, root, "chicago", venues_info.chicago)
    await verify_city_venue(evaluator, root, "new_york_city", venues_info.new_york_city)
    await verify_city_venue(evaluator, root, "seattle", venues_info.seattle)

    # Return structured evaluation summary
    return evaluator.get_summary()