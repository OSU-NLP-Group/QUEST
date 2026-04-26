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
TASK_ID = "outdoor_amphitheater_venue"
TASK_DESCRIPTION = """
Identify an outdoor amphitheater venue in the United States that meets all of the following requirements for hosting a major concert event:

1. The venue must be an outdoor amphitheater (not an indoor arena, indoor theater, or stadium)
2. Total seating capacity must be at least 15,000 people
3. The venue must have both a reserved seating section and a lawn seating section
4. The venue must meet ADA accessibility requirements with wheelchair accessible seating
5. The venue must be located in the United States
6. The venue must have on-site parking facilities available for attendees
7. The reserved seating section must be covered or under a roof structure
8. The venue must have an operational box office for ticket sales

Provide the name and location of one amphitheater venue that satisfies all these criteria, along with supporting documentation for each requirement.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """Structured information we expect the agent to provide in the answer."""
    venue_name: Optional[str] = None
    venue_location: Optional[str] = None  # Prefer a "City, State" string or full address
    official_url: Optional[str] = None  # The main official venue website (if provided)
    general_sources: List[str] = Field(default_factory=list)  # Any general source URLs

    # Optional helpful details from the answer
    capacity_reported: Optional[str] = None  # Capacity as reported in the answer (string)

    # Requirement-specific source URLs (explicitly mentioned in the answer)
    sources_venue_type: List[str] = Field(default_factory=list)
    sources_capacity: List[str] = Field(default_factory=list)
    sources_seating_configuration: List[str] = Field(default_factory=list)
    sources_ada_accessibility: List[str] = Field(default_factory=list)
    sources_us_location: List[str] = Field(default_factory=list)
    sources_parking_facilities: List[str] = Field(default_factory=list)
    sources_covered_seating: List[str] = Field(default_factory=list)
    sources_box_office: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    Extract the amphitheater venue details and all source URLs explicitly mentioned in the answer. Return the following fields:

    1) venue_name: The exact venue name provided in the answer (e.g., "Xfinity Center", "PNC Music Pavilion"). If not provided, return null.
    2) venue_location: The location string provided in the answer (e.g., "Mansfield, MA" or full address). If not provided, return null.
    3) official_url: The main official venue website URL if the answer provides one; otherwise null.
    4) general_sources: A list of any URLs mentioned that generally support the venue information but are not tied to a particular requirement.
    5) capacity_reported: If the answer states a capacity value (e.g., "19,900"), extract it verbatim as a string; otherwise null.

    For each specific requirement, extract the list of source URLs explicitly provided in the answer that support that requirement:
    - sources_venue_type: URLs supporting that the venue is an outdoor amphitheater (not an indoor arena/theater/stadium)
    - sources_capacity: URLs supporting that the total capacity is at least 15,000
    - sources_seating_configuration: URLs supporting that the venue has both reserved seating and a lawn section
    - sources_ada_accessibility: URLs supporting ADA compliance with wheelchair accessible seating
    - sources_us_location: URLs supporting that the venue is located in the United States
    - sources_parking_facilities: URLs supporting that the venue has on-site parking available
    - sources_covered_seating: URLs supporting that the reserved seating section is covered or under a roof
    - sources_box_office: URLs supporting that the venue has an operational box office for ticket sales

    IMPORTANT URL RULES:
    - Only include actual URLs that are explicitly present in the answer (plain URLs or markdown links). Do not invent or infer URLs.
    - Include full URLs, including protocol. If a URL is missing protocol, prepend http://.
    - If the answer describes a source (e.g., "according to Wikipedia") but does not provide a URL, then do not include it and simply return an empty list for that requirement.
    - If multiple URLs are provided, include them all.

    If any of these fields are not present in the answer, set them to null (for single strings) or [] (for lists).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _unique_nonempty_urls(urls: List[str]) -> List[str]:
    """Remove duplicates, trim whitespace, and filter out empty strings."""
    seen = set()
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            cleaned.append(s)
    return cleaned


def gather_sources(extracted: VenueExtraction, specific: List[str]) -> List[str]:
    """
    Build a sources list for verification of one requirement:
    Prefer specific sources; if empty, fall back to official_url + general_sources.
    """
    specific_clean = _unique_nonempty_urls(specific)
    fallback = []

    if extracted.official_url:
        fallback.append(extracted.official_url)
    fallback.extend(extracted.general_sources)

    combined = specific_clean if specific_clean else fallback
    return _unique_nonempty_urls(combined)[:8]  # Keep at most 8 sources to avoid excessive verification


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_venue_verifications(evaluator: Evaluator, parent_node, info: VenueExtraction) -> None:
    """
    Build verification leaf nodes for each requirement under the critical parent node.
    Includes an existence check node for venue identification (name + location).
    """
    # Existence check: Name and location must be provided in the answer
    venue_identified_node = evaluator.add_custom_node(
        result=(bool(info.venue_name) and bool(info.venue_location)),
        id="venue_identified",
        desc="A specific amphitheater venue name and location are provided in the answer",
        parent=parent_node,
        critical=True
    )

    # Common details for claims
    venue_name = info.venue_name or "the venue"
    location_str = info.venue_location or "the provided location"

    # 1) Venue type: outdoor amphitheater
    node_type = evaluator.add_leaf(
        id="venue_type",
        desc="The venue must be an outdoor amphitheater (not an indoor arena, indoor theater, or stadium)",
        parent=parent_node,
        critical=True,
    )
    claim_type = f"The venue named '{venue_name}' is an outdoor amphitheater, not an indoor arena, indoor theater, or stadium."
    await evaluator.verify(
        claim=claim_type,
        node=node_type,
        sources=gather_sources(info, info.sources_venue_type),
        additional_instruction=(
            "Use only the provided webpage(s). If no source URLs are provided, mark this as not supported. "
            "Look for explicit evidence (e.g., 'amphitheater', 'outdoor venue', pavilion/lawn) indicating it is outdoors "
            "and not an indoor arena/theater/stadium."
        ),
        extra_prerequisites=[venue_identified_node],
    )

    # 2) Capacity >= 15,000
    node_capacity = evaluator.add_leaf(
        id="total_capacity",
        desc="The venue's total seating capacity must be at least 15,000 people",
        parent=parent_node,
        critical=True,
    )
    capacity_note = f" It is reported as {info.capacity_reported}." if info.capacity_reported else ""
    claim_capacity = f"The total capacity of '{venue_name}' is at least 15,000 attendees.{capacity_note}"
    await evaluator.verify(
        claim=claim_capacity,
        node=node_capacity,
        sources=gather_sources(info, info.sources_capacity),
        additional_instruction=(
            "Use the provided source page(s) to confirm the venue capacity. If multiple numbers appear, "
            "use the total capacity (seated + lawn if applicable). If capacity is 15,000 or higher, pass; "
            "otherwise fail. If no source URLs are provided, mark this as not supported."
        ),
        extra_prerequisites=[venue_identified_node],
    )

    # 3) Seating configuration: reserved seating + lawn
    node_seating_cfg = evaluator.add_leaf(
        id="seating_configuration",
        desc="The venue must have both a reserved seating section and a lawn seating section",
        parent=parent_node,
        critical=True,
    )
    claim_seating_cfg = (
        f"'{venue_name}' has both reserved seating and a lawn seating section (e.g., a pavilion/pit with seats and a GA lawn)."
    )
    await evaluator.verify(
        claim=claim_seating_cfg,
        node=node_seating_cfg,
        sources=gather_sources(info, info.sources_seating_configuration),
        additional_instruction=(
            "Check seating maps or venue description to confirm that reserved seats exist and there is a distinct lawn section. "
            "If no source URLs are provided, mark this as not supported."
        ),
        extra_prerequisites=[venue_identified_node],
    )

    # 4) ADA accessibility with wheelchair accessible seating
    node_ada = evaluator.add_leaf(
        id="ada_accessibility",
        desc="The venue must meet ADA accessibility requirements with wheelchair accessible seating",
        parent=parent_node,
        critical=True,
    )
    claim_ada = f"'{venue_name}' provides wheelchair accessible seating and meets ADA accessibility requirements."
    await evaluator.verify(
        claim=claim_ada,
        node=node_ada,
        sources=gather_sources(info, info.sources_ada_accessibility),
        additional_instruction=(
            "Look for an Accessibility/ADA page or explicit statements about ADA compliance and wheelchair accessible seating. "
            "If no source URLs are provided, mark this as not supported."
        ),
        extra_prerequisites=[venue_identified_node],
    )

    # 5) Located in the United States
    node_us = evaluator.add_leaf(
        id="us_location",
        desc="The venue must be located in the United States",
        parent=parent_node,
        critical=True,
    )
    claim_us = f"'{venue_name}' is located in the United States. Its cited location is '{location_str}'."
    await evaluator.verify(
        claim=claim_us,
        node=node_us,
        sources=gather_sources(info, info.sources_us_location),
        additional_instruction=(
            "Confirm that the venue is in the U.S. (city/state or U.S. address). "
            "If no source URLs are provided, mark this as not supported."
        ),
        extra_prerequisites=[venue_identified_node],
    )

    # 6) On-site parking facilities
    node_parking = evaluator.add_leaf(
        id="parking_facilities",
        desc="The venue must have on-site parking facilities available for attendees",
        parent=parent_node,
        critical=True,
    )
    claim_parking = f"'{venue_name}' offers on-site parking facilities for attendees."
    await evaluator.verify(
        claim=claim_parking,
        node=node_parking,
        sources=gather_sources(info, info.sources_parking_facilities),
        additional_instruction=(
            "Check parking information pages for 'on-site parking', venue parking lots, or similar. "
            "If only off-site remote parking is available with shuttles and no on-site lots, fail. "
            "If no source URLs are provided, mark this as not supported."
        ),
        extra_prerequisites=[venue_identified_node],
    )

    # 7) Covered reserved seating (under roof)
    node_covered = evaluator.add_leaf(
        id="covered_seating",
        desc="The reserved seating section must be covered or under a roof structure",
        parent=parent_node,
        critical=True,
    )
    claim_covered = f"The reserved seating section at '{venue_name}' is covered by a roof/canopy/pavilion structure."
    await evaluator.verify(
        claim=claim_covered,
        node=node_covered,
        sources=gather_sources(info, info.sources_covered_seating),
        additional_instruction=(
            "Look for references to a 'covered pavilion', 'roof over reserved seats', 'canopy', or similar. "
            "The lawn may be uncovered; the reserved/pavilion seats must be covered. "
            "If no source URLs are provided, mark this as not supported."
        ),
        extra_prerequisites=[venue_identified_node],
    )

    # 8) Operational box office
    node_box_office = evaluator.add_leaf(
        id="box_office",
        desc="The venue must have an operational box office for ticket sales",
        parent=parent_node,
        critical=True,
    )
    claim_box = f"'{venue_name}' has an operational box office where tickets can be purchased."
    await evaluator.verify(
        claim=claim_box,
        node=node_box_office,
        sources=gather_sources(info, info.sources_box_office),
        additional_instruction=(
            "Look for 'box office' information, hours, or on-site ticket purchase details. "
            "If no source URLs are provided, mark this as not supported."
        ),
        extra_prerequisites=[venue_identified_node],
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
    Evaluate an answer for the outdoor amphitheater venue task.
    """
    # Initialize evaluator (root node is non-critical by design; we'll add a critical child node)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Requirements are independent checks
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Add a critical parent node representing the overall venue verification
    venue_root = evaluator.add_parallel(
        id="outdoor_amphitheater_venue",
        desc="Identify an outdoor amphitheater venue in the United States that meets all specified requirements for hosting a concert event",
        parent=root,
        critical=True  # If any child requirement fails, this node (and thus overall) fails
    )

    # Build all requirement verifications
    await build_venue_verifications(evaluator, venue_root, extracted)

    # Return structured result using the evaluator's summary
    return evaluator.get_summary()