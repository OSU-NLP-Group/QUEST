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
TASK_ID = "atl_indoor_concert_venues_3k_5k_ada_2024_2026"
TASK_DESCRIPTION = (
    "Identify 2 indoor concert venues located in Atlanta, Georgia (including the metro Atlanta area) "
    "that meet all of the following requirements:\n\n"
    "1. The venue must be an indoor facility (not an outdoor amphitheater)\n"
    "2. The venue's seating capacity must be between 3,000 and 5,000 seats (inclusive)\n"
    "3. The venue must meet ADA accessibility requirements, including at least 1% of seats designated as wheelchair accessible\n"
    "4. The venue must have at least 2 emergency exits as required by safety regulations\n"
    "5. The venue must regularly host live music concerts or performances\n"
    "6. The venue must be operational and actively hosting events as of 2024-2026\n\n"
    "For each venue, provide:\n"
    "- The official venue name\n"
    "- The exact seating capacity\n"
    "- The physical address\n"
    "- A reference URL from an official or reputable source verifying the information"
)

METRO_ATLANTA_HINT = (
    "Metro Atlanta includes the City of Atlanta and surrounding cities and counties such as Fulton, DeKalb, Cobb, "
    "Gwinnett, Clayton, Henry, Cherokee, Forsyth, Rockdale, and cities like Sandy Springs, Alpharetta, Decatur, "
    "Duluth, Marietta, Roswell, etc."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Venue(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    capacity_exact: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    notes_indoor: Optional[str] = None
    notes_location: Optional[str] = None
    notes_ada: Optional[str] = None
    notes_emergency_exits: Optional[str] = None
    notes_concerts: Optional[str] = None
    notes_operational: Optional[str] = None


class VenuesExtraction(BaseModel):
    venues: List[Venue] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return (
        "From the provided answer, extract at most 2 indoor concert venues in Atlanta, Georgia (including the metro "
        "Atlanta area) that the answer claims meet all requirements. For each venue, extract the following fields:\n"
        "1. name: The official venue name as stated in the answer.\n"
        "2. address: The physical street address (include city/state).\n"
        "3. capacity_exact: The exact seating capacity number as quoted by the answer (keep formatting as-is; do not convert).\n"
        "4. reference_urls: A list of one or more URLs (official or reputable sources) cited in the answer for this venue.\n"
        "5. notes_indoor: Any text in the answer indicating the venue is indoors (e.g., 'indoor venue', 'indoor arena').\n"
        "6. notes_location: Any text mentioning the venue is in Atlanta or a metro Atlanta city/county.\n"
        "7. notes_ada: Any text in the answer about ADA accessibility (e.g., 'wheelchair accessible seating', 'ADA compliant').\n"
        "8. notes_emergency_exits: Any text in the answer about emergency exits count or compliance.\n"
        "9. notes_concerts: Any text indicating the venue regularly hosts concerts or live performances.\n"
        "10. notes_operational: Any text indicating the venue is operational with events during 2024-2026.\n\n"
        "Return a JSON object with a 'venues' array of up to 2 venue objects containing these fields. "
        "If a field is missing for a venue, set it to null. Only extract venues explicitly mentioned in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def has_valid_urls(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    valid = 0
    for u in urls:
        if isinstance(u, str) and u.strip() and (u.strip().startswith("http://") or u.strip().startswith("https://")):
            valid += 1
    return valid > 0


# --------------------------------------------------------------------------- #
# Verification for a single venue                                             #
# --------------------------------------------------------------------------- #
async def verify_venue(
    evaluator: Evaluator,
    parent_node,
    venue: Venue,
    venue_index: int,
) -> None:
    """
    Build verification nodes for one venue and run checks.
    All children under this venue node are marked critical to reflect the rubric's mandatory requirements.
    """
    vid = venue_index + 1
    venue_node = evaluator.add_parallel(
        id=f"Venue_{vid}",
        desc=f"{'First' if vid == 1 else 'Second'} qualifying venue meeting all requirements",
        parent=parent_node,
        critical=True  # Task requires all criteria per venue
    )

    # Critical existence checks (custom nodes)
    evaluator.add_custom_node(
        result=bool(venue.name and venue.name.strip()),
        id=f"V{vid}_Name",
        desc="Venue name is provided",
        parent=venue_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(venue.capacity_exact and venue.capacity_exact.strip()),
        id=f"V{vid}_Exact_Capacity",
        desc="Exact seating capacity number is specified",
        parent=venue_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(venue.address and venue.address.strip()),
        id=f"V{vid}_Address",
        desc="Physical address of the venue is provided",
        parent=venue_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_valid_urls(venue.reference_urls),
        id=f"V{vid}_Reference_URL",
        desc="Verifiable reference URL from official or reputable source is provided",
        parent=venue_node,
        critical=True
    )

    # Evidence-backed verifications (leaf nodes)
    # Location in Atlanta (including metro area)
    loc_node = evaluator.add_leaf(
        id=f"V{vid}_Location_Atlanta",
        desc="Venue is located in Atlanta, Georgia (including metro Atlanta area)",
        parent=venue_node,
        critical=True
    )
    loc_claim = (
        f"The venue '{venue.name or ''}' is located within the City of Atlanta, Georgia or the greater metro Atlanta area. "
        f"Address: {venue.address or 'unknown'}."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=venue.reference_urls,
        additional_instruction=(
            f"Verify the address belongs to Atlanta, GA or the broader Metro Atlanta region. {METRO_ATLANTA_HINT} "
            "Accept clear evidence from the source's address or 'About/Contact' page. If the source is unrelated or "
            "doesn't provide location/address, mark as not supported."
        )
    )

    # Indoor facility
    indoor_node = evaluator.add_leaf(
        id=f"V{vid}_Indoor_Facility",
        desc="Venue is an indoor facility, not an outdoor amphitheater",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{venue.name or ''}' is an indoor facility (enclosed building, not an outdoor amphitheater).",
        node=indoor_node,
        sources=venue.reference_urls,
        additional_instruction=(
            "Look for descriptions such as 'indoor', 'arena', 'theater', 'enclosed space', 'roofed facility', or "
            "photos/layout indicating indoor seating. If pages indicate an outdoor amphitheater or open-air venue, fail."
        )
    )

    # Capacity range check (3,000–5,000 inclusive)
    cap_range_node = evaluator.add_leaf(
        id=f"V{vid}_Capacity_Range",
        desc="Venue capacity is between 3,000 and 5,000 seats (inclusive)",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{venue.name or ''}' has a seating capacity between 3,000 and 5,000 seats (inclusive).",
        node=cap_range_node,
        sources=venue.reference_urls,
        additional_instruction=(
            "Use the cited source(s) to confirm capacity. If the source lists a specific capacity, check whether it "
            "falls within [3000, 5000]. Allow minor wording variations. If no capacity is stated on the sources, fail."
        )
    )

    # ADA compliance: at least 1% wheelchair-accessible seating
    ada_node = evaluator.add_leaf(
        id=f"V{vid}_ADA_Compliance",
        desc="Venue meets ADA requirement of minimum 1% wheelchair accessible seating",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The venue '{venue.name or ''}' meets ADA accessibility requirements, including at least 1% of seats "
            "designated as wheelchair accessible."
        ),
        node=ada_node,
        sources=venue.reference_urls,
        additional_instruction=(
            "Look for explicit ADA/accessibility pages, seating charts indicating wheelchair-accessible sections, or "
            "official statements that the venue provides compliant wheelchair seating at or above 1% of total seats. "
            "If the sources do not substantiate the 1% threshold or accessible seating presence, mark as not supported."
        )
    )

    # Emergency exits: at least 2
    exits_node = evaluator.add_leaf(
        id=f"V{vid}_Emergency_Exits",
        desc="Venue has at least 2 emergency exits as required by safety regulations",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{venue.name or ''}' has at least 2 emergency exits.",
        node=exits_node,
        sources=venue.reference_urls,
        additional_instruction=(
            "Accept evidence from official floor plans, safety policy pages, or credible descriptions explicitly "
            "mentioning emergency exits count. If sources do not provide exit count or credible confirmation, fail."
        )
    )

    # Regular live music concerts/performances
    concerts_node = evaluator.add_leaf(
        id=f"V{vid}_Concert_Venue",
        desc="Venue hosts live music concerts or performances",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{venue.name or ''}' regularly hosts live music concerts or performances.",
        node=concerts_node,
        sources=venue.reference_urls,
        additional_instruction=(
            "Check event calendars, 'Events' pages, or press listings on the official/reputable sources to confirm "
            "live music or performance events occur regularly (e.g., multiple per month/season). If not evident, fail."
        )
    )

    # Operational as of 2024–2026
    operational_node = evaluator.add_leaf(
        id=f"V{vid}_Operational_2024",
        desc="Venue is operational and hosting events as of 2024-2026",
        parent=venue_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{venue.name or ''}' is operational and actively hosting events in 2024, 2025, or 2026.",
        node=operational_node,
        sources=venue.reference_urls,
        additional_instruction=(
            "Verify recent events, schedules, or announcements dated within 2024–2026. Upcoming events listings, "
            "ticket pages, or recent press releases qualify. If sources are outdated or no events in 2024–2026, fail."
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
    Entry point to evaluate the agent's answer for the Atlanta concert venues task.
    Builds a verification tree aligned to the rubric and returns a structured summary.
    """
    # Initialize evaluator with a parallel root
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

    # Extract venues from the answer (limit to first 2 later)
    venues_extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Create top-level critical task node (reflecting rubric root requirement)
    task_node = evaluator.add_parallel(
        id="Atlanta_Concert_Venues_Task",
        desc="Find 2 indoor concert venues in Atlanta, Georgia with capacity between 3,000 and 5,000 seats that meet all specified requirements",
        parent=root,
        critical=True
    )

    # Prepare up to 2 venues (pad if needed)
    venues_list: List[Venue] = list(venues_extraction.venues[:2])
    while len(venues_list) < 2:
        venues_list.append(Venue())

    # Verify each of the 2 venues
    for idx in range(2):
        await verify_venue(evaluator, task_node, venues_list[idx], idx)

    # Return summary with verification tree and score
    return evaluator.get_summary()