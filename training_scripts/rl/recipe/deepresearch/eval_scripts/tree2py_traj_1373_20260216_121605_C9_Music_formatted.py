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
TASK_ID = "indoor_concert_venues_mid_to_large_us_metros"
TASK_DESCRIPTION = """
Identify four mid-to-large indoor concert venues located in major United States metropolitan areas (population over 500,000) that meet ALL of the following requirements:

1. Capacity: The venue must have a stated seating capacity between 2,500 and 10,000 seats (inclusive).
2. Venue Type: The venue must be classified as an indoor concert hall, indoor arena, or indoor theater. Outdoor amphitheaters and outdoor stadiums do not qualify.
3. Location: The venue must be located in a major U.S. metropolitan area with a population exceeding 500,000.
4. Operational Status: The venue must be currently operational and actively booking or hosting events during the 2025-2026 season.
5. Performance History: The venue must have documented evidence of hosting at least one performance by an artist who has won a Grammy Award OR been inducted into the Rock and Roll Hall of Fame.
6. Seating Configuration: The venue must offer reserved seating with a documented seating chart that shows sections and seat assignments.
7. Ticket Pricing: The venue should have publicly available information showing multiple ticket price tiers or pricing sections (preferred but not absolutely required).

For each of the four venues, provide:
- The official venue name
- The city and state location
- The seating capacity (with source URL)
- Evidence of the venue type and operational status (with source URL)
- Documentation of at least one qualifying artist performance (with source URL)
- Evidence of reserved seating availability (with source URL)
- If available, evidence of multiple pricing tiers (with source URL)

All information must be verifiable through official venue websites, ticketing platforms, tour archives, or other reliable online sources. Include specific URLs for each piece of supporting evidence.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    # Identification
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    # Metro population support (e.g., census or metro wiki page)
    metro_population_source_urls: List[str] = Field(default_factory=list)
    # Venue classification (indoor)
    venue_type_claim: Optional[str] = None  # e.g., "indoor theater", "indoor arena"
    type_source_urls: List[str] = Field(default_factory=list)
    # Operational status (active bookings 2025-2026)
    operational_status_source_urls: List[str] = Field(default_factory=list)
    # Capacity
    capacity_text: Optional[str] = None  # e.g., "4,500", "approx. 7,800", "up to 9,000"
    capacity_source_urls: List[str] = Field(default_factory=list)
    # Performance credentials
    qualifying_artist_name: Optional[str] = None
    qualification_type: Optional[str] = None  # "Grammy" or "Rock Hall" or similar
    artist_qualification_source_urls: List[str] = Field(default_factory=list)  # e.g., grammy.com or rockhall.com
    performance_evidence_urls: List[str] = Field(default_factory=list)  # event page, tour archive, setlist.fm, news
    # Seating system
    seating_chart_urls: List[str] = Field(default_factory=list)  # venue or ticketing seating map pages
    # Pricing tiers (non-critical, preferred)
    pricing_evidence_urls: List[str] = Field(default_factory=list)  # ticketing pages showing multiple prices


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract up to FOUR venues from the answer that the user provided. For each venue, extract the following fields exactly as presented in the answer and collect the explicit URLs cited for each verification:

For each venue object, include:
- name: Official venue name (string)
- city: City (string)
- state: State (string or abbreviation)
- metro_population_source_urls: Array of URLs that support that the city/metropolitan area has population > 500,000 (e.g., US Census, Wikipedia MSA lists). If not provided, return [].
- venue_type_claim: The venue type phrase stated in the answer (e.g., "indoor theater", "indoor arena", "concert hall"). If not stated, set to null.
- type_source_urls: Array of URLs that support that the venue is an indoor concert hall/arena/theater (not amphitheater or stadium). If not provided, return [].
- operational_status_source_urls: Array of URLs (e.g., venue calendar, events page, ticketing listings) showing active events in 2025 or 2026. If none provided, return [].
- capacity_text: The seating capacity as written (string). If a range or approximate is provided, extract the exact wording (e.g., "approx. 4,500"). If missing, set to null.
- capacity_source_urls: Array of URLs that explicitly show or state the seating capacity for the venue. If none, return [].
- qualifying_artist_name: The name of at least one artist who performed at the venue and meets the qualification (Grammy winner or Rock Hall inductee). If missing, set to null.
- qualification_type: "Grammy", "Rock Hall", or another phrase if stated. If not specified, set to null.
- artist_qualification_source_urls: Array of URLs that support the artist's Grammy win(s) or Rock Hall induction (e.g., grammy.com/awards, rockhall.com/inductees). If none, return [].
- performance_evidence_urls: Array of URLs showing that the qualifying artist performed at the venue (e.g., venue archive, setlist.fm, tour page, news). If none, return [].
- seating_chart_urls: Array of URLs with a seating chart showing sections and seat numbers (venue site or ticketing). If none, return [].
- pricing_evidence_urls: Array of URLs with evidence of multiple ticket price tiers or sections. If none, return [].

Rules:
- Only extract items that are explicitly present in the answer text. Do not invent fields or URLs.
- Return at most four venue objects in the 'venues' array. If more are present in the answer, keep only the first four.
- For any missing field, set null (for strings) or [] (for URL arrays).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(n, f"{n}th")


def safe_urls(urls: Optional[List[str]]) -> List[str]:
    return urls if isinstance(urls, list) else []


# --------------------------------------------------------------------------- #
# Verification for a single venue                                             #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    index: int,
) -> None:
    """
    Build the verification sub-tree for a single venue and execute all verifications.
    The tree structure mirrors the rubric. Some parent criticalities are adjusted to satisfy
    the framework constraint that a critical parent cannot have non-critical children.
    """

    # Top-level node for this venue (non-critical per rubric)
    venue_node = evaluator.add_parallel(
        id=f"venue_{index}",
        desc=f"{ordinal(index)} qualifying venue meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # ---------------- Identification & basic classification ---------------- #
    ident_node = evaluator.add_parallel(
        id=f"v{index}_identification",
        desc="Venue identification and basic classification",
        parent=venue_node,
        critical=True
    )

    # Name + City/State + Metro verification
    name_city_node = evaluator.add_parallel(
        id=f"v{index}_name_city",
        desc="Specific venue name and city location identified",
        parent=ident_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(venue.name and venue.name.strip()),
        id=f"v{index}_name_provided",
        desc="Official venue name stated",
        parent=name_city_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(venue.city and venue.city.strip() and venue.state and venue.state.strip()),
        id=f"v{index}_city_provided",
        desc="City and state location stated",
        parent=name_city_node,
        critical=True
    )

    metro_leaf = evaluator.add_leaf(
        id=f"v{index}_metro_verification",
        desc="Located in metropolitan area with population over 500,000 (verifiable from census data)",
        parent=name_city_node,
        critical=True
    )
    metro_claim_city = venue.city or ""
    metro_claim_state = venue.state or ""
    metro_claim = (
        f"The metropolitan area that includes {metro_claim_city}, {metro_claim_state} has a population over 500,000."
    )
    await evaluator.verify(
        claim=metro_claim,
        node=metro_leaf,
        sources=safe_urls(venue.metro_population_source_urls),
        additional_instruction="Accept support from credible sources (e.g., US Census, Wikipedia pages for MSAs/CSAs) that clearly indicate the metro area exceeds 500,000 population."
    )

    # Venue classification (indoor)
    type_node = evaluator.add_parallel(
        id=f"v{index}_venue_classification",
        desc="Venue type meets indoor concert facility requirement",
        parent=ident_node,
        critical=True
    )

    # Source presence (existence) for type evidence
    evaluator.add_custom_node(
        result=len(safe_urls(venue.type_source_urls)) > 0,
        id=f"v{index}_type_source",
        desc="Venue type verifiable from official source with URL provided",
        parent=type_node,
        critical=True
    )

    indoor_leaf = evaluator.add_leaf(
        id=f"v{index}_indoor_type",
        desc="Classified as indoor concert hall, arena, or theater (not outdoor amphitheater/stadium)",
        parent=type_node,
        critical=True
    )
    type_phrase = venue.venue_type_claim or "an indoor concert hall, indoor arena, or indoor theater"
    indoor_claim = (
        f"{venue.name or 'This venue'} is {type_phrase} and is not an outdoor amphitheater or outdoor stadium."
    )
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_leaf,
        sources=safe_urls(venue.type_source_urls),
        additional_instruction="Look for explicit or strongly implied indications that the venue is indoors and is a theater/arena/concert hall. Pages that clearly label it 'indoor', 'theater', 'arena', or 'concert hall' qualify. Exclude amphitheaters and stadiums."
    )

    # Operational status for 2025-2026 season
    op_node = evaluator.add_parallel(
        id=f"v{index}_operational_status",
        desc="Venue currently operational for 2025-2026 season",
        parent=ident_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(safe_urls(venue.operational_status_source_urls)) > 0,
        id=f"v{index}_status_source",
        desc="Operational status verifiable from venue calendar/website with URL provided",
        parent=op_node,
        critical=True
    )

    active_leaf = evaluator.add_leaf(
        id=f"v{index}_active_booking",
        desc="Evidence of active event bookings or schedule for 2025-2026",
        parent=op_node,
        critical=True
    )
    active_claim = (
        f"{venue.name or 'This venue'} has upcoming events or a published schedule in 2025 or 2026."
    )
    await evaluator.verify(
        claim=active_claim,
        node=active_leaf,
        sources=safe_urls(venue.operational_status_source_urls),
        additional_instruction="Verify the page shows event listings, a calendar, or announcements indicating shows in 2025 or 2026 (e.g., specific dates, '2025-2026 season', or similar)."
    )

    # ---------------- Capacity requirements -------------------------------- #
    cap_node = evaluator.add_parallel(
        id=f"v{index}_capacity_requirements",
        desc="Venue capacity specifications and verification",
        parent=venue_node,
        critical=True
    )

    cap_min_node = evaluator.add_parallel(
        id=f"v{index}_capacity_minimum",
        desc="Stated capacity meets 2,500 seat minimum threshold",
        parent=cap_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(safe_urls(venue.capacity_source_urls)) > 0,
        id=f"v{index}_capacity_documentation",
        desc="Capacity verifiable from official venue website or reliable source with URL provided",
        parent=cap_min_node,
        critical=True
    )

    cap_value_leaf = evaluator.add_leaf(
        id=f"v{index}_capacity_value",
        desc="Specific capacity number provided and is at least 2,500",
        parent=cap_min_node,
        critical=True
    )
    cap_text = venue.capacity_text or ""
    cap_value_claim = (
        f"The seating capacity of {venue.name or 'the venue'} is {cap_text}, and this is at least 2,500 seats."
    )
    await evaluator.verify(
        claim=cap_value_claim,
        node=cap_value_leaf,
        sources=safe_urls(venue.capacity_source_urls),
        additional_instruction="Check the page for an explicit or implied capacity. If it states a figure >= 2,500 (including approximations like 'about 3,000'), consider it meeting the minimum."
    )

    cap_range_node = evaluator.add_parallel(
        id=f"v{index}_capacity_range",
        desc="Capacity falls within mid-to-large venue classification",
        parent=cap_node,
        critical=True
    )

    cap_upper_leaf = evaluator.add_leaf(
        id=f"v{index}_upper_limit",
        desc="Capacity does not exceed 10,000 seats (upper bound for specified range)",
        parent=cap_range_node,
        critical=True
    )
    cap_upper_claim = (
        f"The seating capacity of {venue.name or 'the venue'} does not exceed 10,000 seats."
    )
    await evaluator.verify(
        claim=cap_upper_claim,
        node=cap_upper_leaf,
        sources=safe_urls(venue.capacity_source_urls),
        additional_instruction="If the page states a capacity ≤ 10,000 (e.g., 7,500; 9,000), then this claim is supported."
    )

    # Confirm range uses same source as value (existence of capacity source suffices here)
    evaluator.add_custom_node(
        result=len(safe_urls(venue.capacity_source_urls)) > 0,
        id=f"v{index}_range_verification",
        desc="Capacity range confirmed from same source as capacity value",
        parent=cap_range_node,
        critical=True
    )

    # ---------------- Performance credentials ------------------------------ #
    perf_node = evaluator.add_parallel(
        id=f"v{index}_performance_credentials",
        desc="Historical performance record by recognized artists",
        parent=venue_node,
        critical=True
    )

    notable_node = evaluator.add_parallel(
        id=f"v{index}_notable_performances",
        desc="Documented performances by qualifying artists",
        parent=perf_node,
        critical=True
    )

    artist_qual_leaf = evaluator.add_leaf(
        id=f"v{index}_artist_qualification",
        desc="At least one documented performance by Grammy-winning or Rock Hall of Fame inducted artist identified",
        parent=notable_node,
        critical=True
    )
    artist_name = venue.qualifying_artist_name or "the artist"
    qual_phrase = venue.qualification_type or "a Grammy Award winner or Rock and Roll Hall of Fame inductee"
    artist_qual_claim = f"{artist_name} has achieved qualification as {qual_phrase}."
    await evaluator.verify(
        claim=artist_qual_claim,
        node=artist_qual_leaf,
        sources=safe_urls(venue.artist_qualification_source_urls),
        additional_instruction="Prefer official sources (grammy.com, rockhall.com). Reliable secondary sources are acceptable if clearly stating the award/induction."
    )

    perf_evidence_leaf = evaluator.add_leaf(
        id=f"v{index}_performance_evidence",
        desc="Performance history verifiable through venue archives, tour databases, or news sources with URL provided",
        parent=notable_node,
        critical=True
    )
    perf_claim = f"{artist_name} has performed at {venue.name or 'the venue'}."
    await evaluator.verify(
        claim=perf_claim,
        node=perf_evidence_leaf,
        sources=safe_urls(venue.performance_evidence_urls),
        additional_instruction="Accept evidence such as venue event archives, setlist.fm pages, tour date listings, or credible news reports indicating the artist performed at the venue."
    )

    # ---------------- Facility features (seating & pricing) ---------------- #
    # NOTE: Adjusted to non-critical to allow a non-critical pricing subtree under it without violating
    # the framework's constraint that a critical parent cannot have non-critical children.
    facility_node = evaluator.add_parallel(
        id=f"v{index}_facility_features",
        desc="Seating configuration and pricing structure",
        parent=venue_node,
        critical=False
    )

    seating_node = evaluator.add_parallel(
        id=f"v{index}_seating_system",
        desc="Reserved seating with documented configuration",
        parent=facility_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(safe_urls(venue.seating_chart_urls)) > 0,
        id=f"v{index}_seating_source",
        desc="Seating chart verifiable from venue website or ticketing platform with URL provided",
        parent=seating_node,
        critical=True
    )

    seating_chart_leaf = evaluator.add_leaf(
        id=f"v{index}_seating_chart",
        desc="Seating chart showing sections and seat assignments available",
        parent=seating_node,
        critical=True
    )
    seating_claim = f"{venue.name or 'The venue'} provides a seating chart that shows sections and seat assignments (reserved seating)."
    await evaluator.verify(
        claim=seating_claim,
        node=seating_chart_leaf,
        sources=safe_urls(venue.seating_chart_urls),
        additional_instruction="The page should display a seating map with sections and seat numbers or assigned seating, either on the venue site or a ticketing platform."
    )

    pricing_node = evaluator.add_parallel(
        id=f"v{index}_pricing_tiers",
        desc="Multiple ticket price levels documented",
        parent=facility_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=len(safe_urls(venue.pricing_evidence_urls)) > 0,
        id=f"v{index}_pricing_source",
        desc="Pricing information verifiable from past events or current listings with URL provided",
        parent=pricing_node,
        critical=False
    )

    tier_leaf = evaluator.add_leaf(
        id=f"v{index}_tier_evidence",
        desc="Evidence of at least two different price tiers or sections",
        parent=pricing_node,
        critical=False
    )
    tier_claim = f"{venue.name or 'The venue'} has multiple ticket price tiers or sections for events."
    await evaluator.verify(
        claim=tier_claim,
        node=tier_leaf,
        sources=safe_urls(venue.pricing_evidence_urls),
        additional_instruction="Look for pages showing at least two distinct price points or price ranges for seats/sections. Accept historical or current listings."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Entry point to evaluate an answer for the indoor concert venue task.
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

    # Extract up to 4 venues from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Select only the first 4 venues, pad if fewer
    venues: List[VenueItem] = list(extraction.venues[:4])
    while len(venues) < 4:
        venues.append(VenueItem())

    # Build sub-trees for each of the 4 venues
    for i in range(1, 5):
        await verify_single_venue(
            evaluator=evaluator,
            parent_node=root,
            venue=venues[i - 1],
            index=i
        )

    # Return evaluation summary
    return evaluator.get_summary()