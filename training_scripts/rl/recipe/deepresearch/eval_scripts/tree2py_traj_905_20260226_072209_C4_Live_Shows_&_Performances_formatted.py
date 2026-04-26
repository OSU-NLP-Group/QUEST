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
TASK_ID = "concert_venues_us_2025"
TASK_DESCRIPTION = (
    "For a hypothetical mid-size concert tour planned for 2025, identify three major indoor concert arenas in the "
    "United States that would be suitable venues. You must select one arena from each of the following cities: "
    "New York City, Los Angeles, and Chicago. For each of the three arenas, provide: (1) The official name of the venue, "
    "(2) The concert seating capacity (must be between 15,000 and 25,000 seats), and (3) Confirmation of its location "
    "within the specified city or metropolitan area. Each arena must be an established indoor venue known for hosting "
    "major concert events."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    """Information for a single city’s selected arena, extracted from the answer."""
    name: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string to handle ranges/notes (e.g., "approx. 20,000 for concerts")
    source_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    """Extraction container for three city arenas."""
    nyc: Optional[VenueItem] = None
    la: Optional[VenueItem] = None
    chicago: Optional[VenueItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract exactly one arena per each of the following cities from the answer: New York City (NYC), Los Angeles (LA), and Chicago.
    For each city, extract:
    - name: The official name of the arena as written in the answer.
    - capacity: The concert seating capacity as written in the answer (keep text exactly; do not convert to a pure number).
    - source_urls: All URLs the answer cites for that arena (official site, Wikipedia, venue pages, etc.). Extract only actual URLs mentioned in the answer text. If none are cited, return an empty array.

    Return a JSON object with keys: nyc, la, chicago. Each key maps to an object with fields: name, capacity, source_urls.
    If the answer mentions multiple arenas for a city, choose the first one mentioned.
    If a city’s arena is not provided, set that city’s object to null.
    IMPORTANT for URLs:
    - Only extract URLs explicitly present in the answer (plain links or markdown links).
    - Do not create or infer URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _city_node_id_prefix(city_key: str) -> str:
    if city_key == "nyc":
        return "NYC_Arena"
    if city_key == "la":
        return "LA_Arena"
    if city_key == "chicago":
        return "Chicago_Arena"
    return f"{city_key}_Arena"


def _city_display_name(city_key: str) -> str:
    return {
        "nyc": "New York City",
        "la": "Los Angeles",
        "chicago": "Chicago",
    }.get(city_key, city_key)


def _location_additional_instruction(city_key: str) -> str:
    if city_key == "nyc":
        return (
            "Confirm that the venue is located within New York City. NYC includes its five boroughs: "
            "Manhattan, Brooklyn, Queens, The Bronx, and Staten Island. Accept indications such as "
            "“New York, NY”, “Manhattan, NY”, “Brooklyn, NY”, “Queens, NY”, “The Bronx, NY”, or “Staten Island, NY”."
        )
    if city_key == "la":
        return (
            "Confirm that the venue is located in the Los Angeles metropolitan area (Greater Los Angeles). "
            "Clear evidence that it is in Los Angeles, CA (e.g., Downtown LA) suffices. If the page explicitly states a "
            "city within the LA metro (e.g., Inglewood, Anaheim, or Glendale), that also counts as within the LA metro."
        )
    if city_key == "chicago":
        return (
            "Confirm that the venue is located in Chicago, Illinois (the City of Chicago proper). "
            "The page should explicitly indicate 'Chicago, IL'. Locations like 'Rosemont, IL' are outside the city and "
            "should not be counted as Chicago proper."
        )
    return "Confirm the venue’s location per the specified city/region."


# --------------------------------------------------------------------------- #
# Verification for each city                                                  #
# --------------------------------------------------------------------------- #
async def verify_city_venue(
    evaluator: Evaluator,
    parent_node,
    city_key: str,
    venue: Optional[VenueItem],
) -> None:
    """
    Build and verify the sub-tree for a single city's arena according to the rubric leaves.

    Leaves to implement (all critical within the city node):
      - {CITY}_Arena_Name: official name is correctly given and supported by cited sources
      - {CITY}_Arena_Capacity: concert capacity between 15k and 25k, supported by sources
      - {CITY}_Arena_Indoor: verify it is an indoor venue (not outdoor)
      - {CITY}_Arena_Location: verify it is located in the specified city/metropolitan area
    """
    node_prefix = _city_node_id_prefix(city_key)
    display_city = _city_display_name(city_key)

    # City node (non-critical; parallel aggregation across cities)
    city_node = evaluator.add_parallel(
        id=node_prefix,
        desc=f"Provide complete information for a major indoor concert arena in {display_city}",
        parent=parent_node,
        critical=False
    )

    # Normalize extracted fields
    name = (venue.name or "").strip() if venue else ""
    capacity_text = (venue.capacity or "").strip() if venue else ""
    sources = venue.source_urls if (venue and venue.source_urls) else []

    # 1) Official Name (Critical)
    name_node = evaluator.add_leaf(
        id=f"{node_prefix}_Name",
        desc="Provide the official name of the arena",
        parent=city_node,
        critical=True
    )
    name_claim = (
        f"The venue's official name is '{name}'. Confirm that at least one of the cited webpages clearly shows this "
        f"official name."
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=sources,
        additional_instruction=(
            "Check the page title, infobox, or prominent headings. Allow minor formatting or punctuation differences, "
            "and accept former/alternate names if the page clearly indicates the current official name is the one given."
        )
    )

    # 2) Capacity between 15,000 and 25,000 (Critical)
    capacity_node = evaluator.add_leaf(
        id=f"{node_prefix}_Capacity",
        desc="Report the concert seating capacity, which must be between 15,000 and 25,000 seats",
        parent=city_node,
        critical=True
    )
    capacity_claim = (
        "The arena's concert seating capacity is between 15,000 and 25,000 seats."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_node,
        sources=sources,
        additional_instruction=(
            f"The answer's stated capacity is: '{capacity_text}'. Verify using the cited page(s) that the concert "
            "configuration capacity (or maximum seating capacity for concerts) is within the 15,000–25,000 range. "
            "If multiple capacities (e.g., basketball/hockey vs concerts) are listed, focus on concert or maximum "
            "event capacity. Allow reasonable approximations (e.g., ~20,000). If evidence does not support being in "
            "this range, mark as not supported."
        )
    )

    # 3) Indoor venue (Critical)
    indoor_node = evaluator.add_leaf(
        id=f"{node_prefix}_Indoor",
        desc="Confirm the arena is an indoor venue (not an outdoor stadium or amphitheater)",
        parent=city_node,
        critical=True
    )
    indoor_claim = "This is an established indoor arena (enclosed venue), not an outdoor stadium or amphitheater."
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_node,
        sources=sources,
        additional_instruction=(
            "Look for indications such as 'indoor arena', 'multipurpose indoor arena', or enclosed roof structure. "
            "If the page indicates it is an open-air stadium or amphitheater, do not support the claim."
        )
    )

    # 4) Location within specified city/metropolitan area (Critical)
    location_node = evaluator.add_leaf(
        id=f"{node_prefix}_Location",
        desc=f"Confirm the arena is located in {display_city if city_key != 'la' else 'the Los Angeles metropolitan area'}",
        parent=city_node,
        critical=True
    )

    if city_key == "la":
        location_claim = "This arena is located within the Los Angeles metropolitan area."
    elif city_key == "nyc":
        location_claim = "This arena is located within New York City (one of its five boroughs)."
    else:  # chicago
        location_claim = "This arena is located in Chicago, Illinois (City of Chicago)."

    await evaluator.verify(
        claim=location_claim,
        node=location_node,
        sources=sources,
        additional_instruction=_location_additional_instruction(city_key)
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
    Evaluate an answer for the concert venue selection task using the Mind2Web2 evaluation framework.
    """
    # Initialize evaluator (root node uses parallel aggregation across cities)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify three major indoor concert arenas in NYC, LA, and Chicago with specific capacity requirements",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract venues info from the answer
    extraction: VenuesExtraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Build rubric tree according to the provided JSON structure
    # Top-level node mirroring "Concert_Venue_Identification" (parallel, non-critical)
    concert_node = evaluator.add_parallel(
        id="Concert_Venue_Identification",
        desc="Identify three major indoor concert arenas in NYC, LA, and Chicago with specific capacity requirements",
        parent=root,
        critical=False
    )

    # Verify each city block
    await verify_city_venue(evaluator, concert_node, "nyc", extraction.nyc)
    await verify_city_venue(evaluator, concert_node, "la", extraction.la)
    await verify_city_venue(evaluator, concert_node, "chicago", extraction.chicago)

    # Optional: record a small summary of which cities had sources
    evaluator.add_custom_info(
        {
            "nyc_urls_count": len(extraction.nyc.source_urls) if extraction.nyc else 0,
            "la_urls_count": len(extraction.la.source_urls) if extraction.la else 0,
            "chicago_urls_count": len(extraction.chicago.source_urls) if extraction.chicago else 0,
        },
        info_type="url_counts",
        info_name="per_city_url_counts"
    )

    # Return structured evaluation summary
    return evaluator.get_summary()