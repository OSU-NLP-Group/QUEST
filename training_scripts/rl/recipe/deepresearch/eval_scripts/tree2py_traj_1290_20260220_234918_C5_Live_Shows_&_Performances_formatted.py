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
TASK_ID = "chicago_historic_theaters"
TASK_DESCRIPTION = """
I am a concert promoter planning a multi-night acoustic concert series in Chicago for a mid-sized touring artist. I need to identify three historic theater venues in Chicago that meet the following requirements:

1. The venue must be located within Chicago city limits (not suburbs)
2. The venue must have a seating capacity between 3,000 and 5,000 people for concert configuration
3. The venue must provide wheelchair-accessible seating and accessible parking facilities in compliance with ADA requirements
4. The venue must have loading dock access for equipment and professional sound system capabilities suitable for live concerts
5. The venue must be a historic theater with documented historic preservation status

For each of the three venues you identify, please provide:
- The venue name
- The exact concert seating capacity
- Confirmation of its historic theater status
- A reference URL to the official venue website or credible source that documents the venue's specifications and features
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Venue(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string to allow ranges/text like "approx. 3,600 seated"
    reference_url: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[Venue] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to three Chicago theater venues described in the answer. These should be historic theaters suitable for hosting concerts.
    For each venue, extract the following fields from the answer exactly as presented:
    - name: The venue name
    - city: The city of the venue (e.g., "Chicago")
    - state: The state abbreviation (e.g., "IL")
    - capacity: The exact concert seating capacity mentioned in the answer (as text). If multiple capacities are mentioned, choose the one specific to concert/theater seated configuration.
    - reference_url: A single best URL to the official venue website or a credible source page that documents venue specifications/features. If multiple URLs are given, choose the most official or comprehensive one.
    - source_urls: An array of all URLs mentioned in the answer that are relevant to this venue (including the reference_url, if it appears in the answer).

    Rules:
    - Only extract what is explicitly present in the answer text. Do not invent information.
    - If more than three venues are present, include only the first three in the order they appear.
    - If any field is missing for a venue, set it to null (for strings) or an empty array (for source_urls).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def collect_all_urls(venue: Venue) -> List[str]:
    """Collect all available URLs for verification for a venue, deduplicated and non-empty."""
    urls = []
    if venue.reference_url and venue.reference_url.strip():
        urls.append(venue.reference_url.strip())
    for u in venue.source_urls:
        if u and isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            unique.append(u)
            seen.add(u)
    return unique


# --------------------------------------------------------------------------- #
# Verification logic per venue                                                #
# --------------------------------------------------------------------------- #
async def verify_venue(evaluator: Evaluator, parent_node, venue: Venue, index: int) -> None:
    """
    Build the verification subtree for a single venue and run checks.
    """
    # Parent node for this venue
    venue_node = evaluator.add_parallel(
        id=f"venue_{index + 1}",
        desc=[
            "First suitable concert venue meeting all requirements",
            "Second suitable concert venue meeting all requirements",
            "Third suitable concert venue meeting all requirements",
        ][index],
        parent=parent_node,
        critical=False  # Each venue contributes partial credit; failures here don't fail entire root
    )

    # Gather URLs for evidence
    all_urls = collect_all_urls(venue)

    # 1) Location within Chicago city limits
    location_node = evaluator.add_leaf(
        id=f"venue_{index + 1}_location",
        desc="Venue is located within Chicago city limits, Illinois",
        parent=venue_node,
        critical=True
    )
    location_claim = "This venue is located within Chicago city limits (Chicago, IL), not in a separate municipality or suburb."
    await evaluator.verify(
        claim=location_claim,
        node=location_node,
        sources=all_urls,
        additional_instruction=(
            "Confirm that the address indicates 'Chicago, IL' or a Chicago neighborhood. "
            "Do not accept suburbs or separate municipalities (e.g., Rosemont, Evanston, Oak Park, Cicero, Skokie, etc.)."
        )
    )

    # 2) Capacity between 3,000 and 5,000 for concert configuration
    capacity_node = evaluator.add_leaf(
        id=f"venue_{index + 1}_capacity",
        desc="Venue has seating capacity between 3,000 and 5,000 people for concert configuration",
        parent=venue_node,
        critical=True
    )
    capacity_claim = "The venue's concert/theater seated capacity is between 3,000 and 5,000 people."
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_node,
        sources=all_urls,
        additional_instruction=(
            "Check for capacity values specific to seated concert/theater configuration. "
            "If multiple capacities are listed, prefer seated/concert configuration. "
            "Accept reasonable variants like 'approximately 3,500' or ranges within 3,000–5,000."
        )
    )

    # 3) Accessibility compliance (wheelchair seating + accessible parking)
    accessibility_main = evaluator.add_parallel(
        id=f"venue_{index + 1}_accessibility",
        desc="Venue accessibility compliance",
        parent=venue_node,
        critical=True
    )

    wheelchair_node = evaluator.add_leaf(
        id=f"venue_{index + 1}_wheelchair_seating",
        desc="Venue provides wheelchair-accessible seating",
        parent=accessibility_main,
        critical=True
    )
    wheelchair_claim = "The venue provides wheelchair-accessible seating options in compliance with ADA."
    await evaluator.verify(
        claim=wheelchair_claim,
        node=wheelchair_node,
        sources=all_urls,
        additional_instruction=(
            "Look for accessibility pages or seating maps indicating ADA or wheelchair-accessible seating."
        )
    )

    parking_node = evaluator.add_leaf(
        id=f"venue_{index + 1}_accessible_parking",
        desc="Venue has accessible parking facilities",
        parent=accessibility_main,
        critical=True
    )
    parking_claim = "The venue offers ADA-compliant accessible parking facilities (on-site or in an affiliated/nearby garage)."
    await evaluator.verify(
        claim=parking_claim,
        node=parking_node,
        sources=all_urls,
        additional_instruction=(
            "Accept mentions of accessible parking in venue parking guides or official partner garages with ADA accommodations."
        )
    )

    # 4) Technical specifications (loading dock + professional sound system)
    tech_main = evaluator.add_parallel(
        id=f"venue_{index + 1}_technical_specs",
        desc="Venue technical specifications for concert production",
        parent=venue_node,
        critical=True
    )

    loading_node = evaluator.add_leaf(
        id=f"venue_{index + 1}_loading_dock",
        desc="Venue has loading dock access for equipment",
        parent=tech_main,
        critical=True
    )
    loading_claim = "The venue has a loading dock or designated load-in access for equipment."
    await evaluator.verify(
        claim=loading_claim,
        node=loading_node,
        sources=all_urls,
        additional_instruction=(
            "Look for production, technical specifications, or venue information pages mentioning 'loading dock', 'load-in', or 'stage door' access."
        )
    )

    sound_node = evaluator.add_leaf(
        id=f"venue_{index + 1}_sound_system",
        desc="Venue has professional sound system capabilities suitable for concerts",
        parent=tech_main,
        critical=True
    )
    sound_claim = "The venue provides professional sound system/PA capabilities suitable for live concert performances."
    await evaluator.verify(
        claim=sound_claim,
        node=sound_node,
        sources=all_urls,
        additional_instruction=(
            "Evidence can include mentions of house PA, professional audio, technical specs, or touring-grade sound support."
        )
    )

    # 5) Historic theater status with documented preservation status
    historic_node = evaluator.add_leaf(
        id=f"venue_{index + 1}_historic_status",
        desc="Venue is a historic theater with documented historic preservation status",
        parent=venue_node,
        critical=True
    )
    historic_claim = (
        "The venue is a historic theater with documented preservation status, such as Chicago Landmark designation or listing on the National Register of Historic Places."
    )
    await evaluator.verify(
        claim=historic_claim,
        node=historic_node,
        sources=all_urls,
        additional_instruction=(
            "Look for explicit designations: 'Chicago Landmark', 'National Register of Historic Places', 'historic theater', "
            "or recognition by official preservation bodies. Accept credible third-party sources if the official site lacks details."
        )
    )

    # 6) Reference URL: official venue website or credible source documenting specs/features
    reference_node = evaluator.add_leaf(
        id=f"venue_{index + 1}_reference_url",
        desc="Provide official venue website or credible source URL documenting the venue's specifications",
        parent=venue_node,
        critical=True
    )
    # Prefer checking the specific reference URL; if missing, fall back to any available source URLs.
    reference_sources = venue.reference_url if (venue.reference_url and venue.reference_url.strip()) else all_urls
    reference_claim = (
        "This page is either the official venue website or a credible source page, and it documents key venue specifications/features "
        "(e.g., capacity, accessibility, production/technical info)."
    )
    await evaluator.verify(
        claim=reference_claim,
        node=reference_node,
        sources=reference_sources,
        additional_instruction=(
            "Assess whether the page is official (venue-owned domain) or a credible source (e.g., government preservation site, major publication). "
            "Also confirm the page contains venue specifications/features (capacity, accessibility, production or technical information)."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the answer for identifying three historic Chicago theater venues meeting specified requirements.
    """
    # Initialize evaluator (root is parallel since venues are independent)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Find three concert venues in Chicago that meet all specified requirements for hosting a multi-night concert series",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Record the task requirements as ground truth info for context
    evaluator.add_ground_truth({
        "requirements": {
            "location": "Within Chicago city limits (not suburbs)",
            "capacity_range": "3,000 to 5,000 seated for concert configuration",
            "accessibility": ["Wheelchair-accessible seating", "Accessible parking"],
            "technical": ["Loading dock access", "Professional sound system suitable for concerts"],
            "historic": "Historic theater with documented preservation status",
            "reference": "Provide official venue website or credible source documenting specifications and features"
        }
    })

    # Extract up to three venues from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Limit to first three venues; pad with empty Venue objects if fewer
    venues_list: List[Venue] = list(extracted.venues[:3])
    while len(venues_list) < 3:
        venues_list.append(Venue())

    # Build verification subtree for each venue
    for i in range(3):
        await verify_venue(evaluator, root, venues_list[i], i)

    # Return structured summary (includes verification tree and scores)
    return evaluator.get_summary()