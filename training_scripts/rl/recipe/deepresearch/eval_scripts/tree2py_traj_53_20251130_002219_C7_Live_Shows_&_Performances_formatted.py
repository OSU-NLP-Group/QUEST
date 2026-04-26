import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_concert_venues_4_cities"
TASK_DESCRIPTION = (
    "You are planning a major concert tour across the United States and need to identify suitable indoor arena venues in four major cities. "
    "For each of the following cities, identify one major indoor arena that hosts concerts and has a documented seating capacity of over 15,000:\n\n"
    "1. Chicago, Illinois\n2. Atlanta, Georgia\n3. Las Vegas, Nevada\n4. San Francisco, California\n\n"
    "For each venue, provide:\n- The official venue name\n- The city location\n- The state location\n- The documented seating capacity (specify the number and configuration type, such as concert capacity or basketball capacity)\n\n"
    "Include a reference source (URL) for each venue's capacity information."
)

CITY_STATE_EXPECTATIONS = [
    ("chicago", "Chicago", "Illinois"),
    ("atlanta", "Atlanta", "Georgia"),
    ("las_vegas", "Las Vegas", "Nevada"),
    ("san_francisco", "San Francisco", "California"),
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity_number: Optional[str] = None
    capacity_configuration: Optional[str] = None  # e.g., "concert", "basketball", "end-stage", "center-stage"
    capacity_source_url: Optional[str] = None     # URL that supports the capacity figure/config
    additional_info_urls: List[str] = Field(default_factory=list)  # other URLs (official site, Wikipedia, events page, etc.)


class VenuesExtraction(BaseModel):
    chicago: Optional[VenueInfo] = None
    atlanta: Optional[VenueInfo] = None
    las_vegas: Optional[VenueInfo] = None
    san_francisco: Optional[VenueInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract venue information for the four specified cities from the provided answer. For each city, extract ONE venue (choose the first one mentioned if multiple are provided) that is a major indoor arena hosting concerts, with a documented seating capacity over 15,000.

    Return a JSON object with these keys: "chicago", "atlanta", "las_vegas", "san_francisco".
    Each key should map to an object containing:
      - name: the official venue name
      - city: the city stated in the answer for the venue
      - state: the state stated in the answer for the venue
      - capacity_number: the documented seating capacity number as written in the answer (include commas/plus signs as written; do NOT normalize)
      - capacity_configuration: the configuration corresponding to the capacity (e.g., "concert", "basketball", "end-stage", "center-stage"); if multiple are mentioned, choose the one tied to the provided number
      - capacity_source_url: the URL cited in the answer that supports the capacity number/configuration
      - additional_info_urls: an array of any other URLs mentioned in the answer for that venue (official site, Wikipedia, events page, etc.). Exclude the capacity_source_url from this list.

    Rules:
    - Only extract URLs that are explicitly present in the answer. Include full URLs (prepend http:// if missing).
    - If any field is not present in the answer for a city, set it to null (or empty array for additional_info_urls).
    - If the answer mentions more than one venue for a city, choose the first and ignore the rest.
    - If the answer fails to provide a qualifying venue for a city, set that city's value to null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedupe_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def build_all_sources(venue: Optional[VenueInfo]) -> List[str]:
    if not venue:
        return []
    urls: List[str] = []
    if venue.capacity_source_url:
        urls.append(venue.capacity_source_url)
    urls.extend(venue.additional_info_urls or [])
    return _dedupe_preserve_order(urls)


def venue_display_name(venue: Optional[VenueInfo], city_label: str) -> str:
    if venue and venue.name and venue.name.strip():
        return venue.name.strip()
    return f"the selected {city_label} venue"


# --------------------------------------------------------------------------- #
# Verification for a single city                                              #
# --------------------------------------------------------------------------- #
async def verify_city_venue(
    evaluator: Evaluator,
    parent_node,
    city_key: str,
    city_label: str,
    state_label: str,
    venue: Optional[VenueInfo],
) -> None:
    """
    Build and verify the tree for a single city's venue according to rubric.
    """
    city_node = evaluator.add_parallel(
        id=f"{city_key}_venue",
        desc=f"One qualifying major indoor arena venue in {city_label}, {state_label}, with all required details and capacity sourcing.",
        parent=parent_node,
        critical=False,
    )

    # Critical: Official venue name provided (existence check)
    name_provided = bool(venue and venue.name and venue.name.strip())
    evaluator.add_custom_node(
        result=name_provided,
        id=f"{city_key}_venue_name",
        desc="Provides the official venue name.",
        parent=city_node,
        critical=True,
    )

    # Critical: City matches expected (verify with sources if available)
    city_leaf = evaluator.add_leaf(
        id=f"{city_key}_venue_city",
        desc=f"City is {city_label}.",
        parent=city_node,
        critical=True,
    )
    vname = venue_display_name(venue, city_label)
    await evaluator.verify(
        claim=f"The venue '{vname}' is located in {city_label}.",
        node=city_leaf,
        sources=build_all_sources(venue),
        additional_instruction=(
            "Verify the venue's city location from the provided webpages. "
            "Allow reasonable variants like 'Chicago, IL' for Chicago; 'Atlanta, GA' for Atlanta; 'San Francisco, CA' for San Francisco. "
            "For Las Vegas, pages sometimes list 'Paradise, NV' because the Las Vegas Strip sits in Paradise—consider this acceptable as Las Vegas area."
        ),
    )

    # Critical: State matches expected (verify with sources if available)
    state_leaf = evaluator.add_leaf(
        id=f"{city_key}_venue_state",
        desc=f"State is {state_label}.",
        parent=city_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The venue '{vname}' is located in {state_label}.",
        node=state_leaf,
        sources=build_all_sources(venue),
        additional_instruction=(
            "Verify the venue's state location from the provided webpages. "
            "Allow abbreviations (e.g., 'IL' for Illinois, 'GA' for Georgia, 'NV' for Nevada, 'CA' for California)."
        ),
    )

    # Critical: Venue is a major indoor arena and hosts concerts
    arena_concerts_leaf = evaluator.add_leaf(
        id=f"{city_key}_venue_is_major_indoor_arena_hosts_concerts",
        desc="Venue is a major indoor arena and hosts concerts/live performances.",
        parent=city_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The venue '{vname}' is an indoor arena and hosts concerts or live music performances. "
            "It is a major arena suitable for large-scale events."
        ),
        node=arena_concerts_leaf,
        sources=build_all_sources(venue),
        additional_instruction=(
            "Confirm that the venue is an indoor arena and hosts concerts/live performances. "
            "Evidence may include descriptors such as 'arena', event listings, or references to concerts/music events."
        ),
    )

    # Critical: Venue is currently operational in 2025
    operational_leaf = evaluator.add_leaf(
        id=f"{city_key}_venue_operational_2025",
        desc="Venue is currently operational in 2025.",
        parent=city_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The venue '{vname}' was operational during 2025 (open and hosting events).",
        node=operational_leaf,
        sources=build_all_sources(venue),
        additional_instruction=(
            "Look for indications of ongoing operations, such as event calendars (including 2025 events), recent schedules, or statements indicating the venue is active. "
            "If the provided sources explicitly show events or operational status spanning 2025, consider the claim supported."
        ),
    )

    # Capacity details presence gating (custom critical)
    capacity_details_provided = bool(
        venue and venue.capacity_number and venue.capacity_number.strip()
        and venue.capacity_configuration and venue.capacity_configuration.strip()
    )
    evaluator.add_custom_node(
        result=capacity_details_provided,
        id=f"{city_key}_venue_capacity_details_provided",
        desc="Capacity number and configuration type are provided.",
        parent=city_node,
        critical=True,
    )

    # Capacity source URL presence gating (custom critical)
    capacity_source_provided = bool(venue and venue.capacity_source_url and venue.capacity_source_url.strip())
    evaluator.add_custom_node(
        result=capacity_source_provided,
        id=f"{city_key}_venue_capacity_source_url_provided",
        desc="Capacity source URL is provided.",
        parent=city_node,
        critical=True,
    )

    # Critical: Capacity over 15,000 with specific number & configuration, supported by capacity source
    capacity_leaf = evaluator.add_leaf(
        id=f"{city_key}_venue_capacity_over_15000_with_details",
        desc="Provides a documented seating capacity > 15,000, including a specific number and the configuration type (e.g., concert vs. basketball).",
        parent=city_node,
        critical=True,
    )
    cap_num = (venue.capacity_number or "").strip() if venue else ""
    cap_cfg = (venue.capacity_configuration or "").strip() if venue else ""
    await evaluator.verify(
        claim=(
            f"The documented seating capacity of '{vname}' is {cap_num} for {cap_cfg}, and this capacity exceeds 15,000."
        ),
        node=capacity_leaf,
        sources=(venue.capacity_source_url if venue else None),
        additional_instruction=(
            "Verify both the specific capacity number and its configuration (e.g., concert or basketball) on the provided capacity source page. "
            "Minor numeric rounding or formatting differences are acceptable if they clearly refer to the same capacity figure."
        ),
    )

    # Critical: Capacity authoritative source URL
    capacity_source_leaf = evaluator.add_leaf(
        id=f"{city_key}_venue_capacity_authoritative_source_url",
        desc="Provides a URL from an official or otherwise authoritative source supporting the stated capacity figure/configuration.",
        parent=city_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The provided page is an official or otherwise authoritative source for '{vname}' capacity "
            f"for {cap_cfg}, and it explicitly states a capacity of {cap_num} (or an equivalent clearly matching figure)."
        ),
        node=capacity_source_leaf,
        sources=(venue.capacity_source_url if venue else None),
        additional_instruction=(
            "Assess whether the page is official or authoritative (e.g., venue's official site, operator/management site, a governing body, or a reputable reference like Wikipedia with citations). "
            "The page must explicitly state the capacity and configuration or clearly support them."
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
    Evaluate an answer for identifying one major indoor arena per specified city with capacity > 15,000 and required details.
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
        default_model=model,
    )

    # Extract venues from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Add ground truth expectations for transparency (not used for strict matching)
    evaluator.add_ground_truth({
        "expected_cities": [
            {"key": key, "city": city, "state": state}
            for key, city, state in CITY_STATE_EXPECTATIONS
        ],
        "requirement": "Each venue must be a major indoor arena hosting concerts with documented capacity > 15,000, including number & configuration, supported by an authoritative URL."
    })

    # Build verification subtrees for each city
    await verify_city_venue(evaluator, root, "chicago", "Chicago", "Illinois", extraction.chicago)
    await verify_city_venue(evaluator, root, "atlanta", "Atlanta", "Georgia", extraction.atlanta)
    await verify_city_venue(evaluator, root, "las_vegas", "Las Vegas", "Nevada", extraction.las_vegas)
    await verify_city_venue(evaluator, root, "san_francisco", "San Francisco", "California", extraction.san_francisco)

    return evaluator.get_summary()