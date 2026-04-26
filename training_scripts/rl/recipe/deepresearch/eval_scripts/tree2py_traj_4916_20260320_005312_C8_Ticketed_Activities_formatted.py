import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_indoor_arena_by_region_2026"
TASK_DESCRIPTION = """
Identify four indoor arena concert venues in the United States, with one venue from each of the following regions: Northeast, Southeast, Midwest, and West. Each venue must meet all of the following criteria:

1. Indoor climate-controlled arena with a permanent roof structure (not outdoor or open-air venues)
2. Concert seating capacity between 15,000 and 25,000
3. Currently operational and hosting events as of March 2026
4. Regularly hosts concerts and entertainment events (not exclusively sports venues)

For each venue, provide:
- The official venue name
- Complete street address (including city and state)
- Concert seating capacity
- A reference URL to the venue's official website or official venue information page
- Brief description of ADA-compliant wheelchair accessible seating features

Regional definitions:
- Northeast: Maine, New Hampshire, Vermont, Massachusetts, Rhode Island, Connecticut, New York, New Jersey, Pennsylvania
- Southeast: Maryland, Delaware, West Virginia, Virginia, Kentucky, Tennessee, North Carolina, South Carolina, Georgia, Florida, Alabama, Mississippi, Arkansas, Louisiana
- Midwest: Ohio, Indiana, Illinois, Michigan, Wisconsin, Minnesota, Iowa, Missouri, North Dakota, South Dakota, Nebraska, Kansas
- West: Montana, Idaho, Wyoming, Colorado, New Mexico, Arizona, Utah, Nevada, California, Oregon, Washington, Alaska, Hawaii
"""

REGION_DEFS = {
    "northeast": {
        "label": "Northeast",
        "prefix": "NE",
        "states_codes": ["ME", "NH", "VT", "MA", "RI", "CT", "NY", "NJ", "PA"],
    },
    "southeast": {
        "label": "Southeast",
        "prefix": "SE",
        "states_codes": ["MD", "DE", "WV", "VA", "KY", "TN", "NC", "SC", "GA", "FL", "AL", "MS", "AR", "LA"],
    },
    "midwest": {
        "label": "Midwest",
        "prefix": "MW",
        "states_codes": ["OH", "IN", "IL", "MI", "WI", "MN", "IA", "MO", "ND", "SD", "NE", "KS"],
    },
    "west": {
        "label": "West",
        "prefix": "W",
        "states_codes": ["MT", "ID", "WY", "CO", "NM", "AZ", "UT", "NV", "CA", "OR", "WA", "AK", "HI"],
    },
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueAccessibility(BaseModel):
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class VenueInfo(BaseModel):
    official_name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None            # Full state name if provided
    state_code: Optional[str] = None       # 2-letter postal abbreviation if provided
    capacity: Optional[str] = None         # Keep as string to be tolerant (e.g., "18,200 for concerts")
    reference_url: Optional[str] = None    # Official website or official info page
    extra_urls: List[str] = Field(default_factory=list)  # Events, seating, facts pages, etc.
    accessibility: Optional[VenueAccessibility] = None


class VenuesExtraction(BaseModel):
    northeast: Optional[VenueInfo] = None
    southeast: Optional[VenueInfo] = None
    midwest: Optional[VenueInfo] = None
    west: Optional[VenueInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract exactly one venue for each U.S. region: Northeast, Southeast, Midwest, and West, as presented in the answer.
    For each region (northeast, southeast, midwest, west), return a VenueInfo object with:
    - official_name: Official venue name exactly as written in the answer
    - street_address: Full street address string as provided (should include city and state)
    - city: City name if given; otherwise null
    - state: State name if given (e.g., "Massachusetts"); otherwise null
    - state_code: 2-letter state abbreviation if present or inferable from the answer text; otherwise null
    - capacity: Concert seating capacity as presented (keep as string; do not coerce to number). If a range or multiple capacities are given (e.g., for basketball/hockey/concert), extract the most relevant concert/general seating capacity string.
    - reference_url: A single URL that is the venue's official website or an official venue information page (team page at the arena, university/municipal official page, or the arena's official domain).
    - extra_urls: Any additional URLs mentioned for that venue in the answer (e.g., events calendar, seating chart, accessibility page). Return an empty list if none.
    - accessibility:
        - description: The brief ADA/accessible seating features text included in the answer for this venue (if any); otherwise null
        - urls: Any accessibility-specific URLs cited (e.g., ADA policy page). Return empty list if none.

    IMPORTANT:
    - If the answer mentions multiple candidate venues for a region, extract the first clearly identified one.
    - If any field is missing in the answer, return null (or [] for url lists).
    - Only extract URLs that are explicitly present in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(text: Optional[str]) -> bool:
    return bool(text and str(text).strip())


def _gather_sources(venue: VenueInfo) -> List[str]:
    urls: List[str] = []
    if venue is None:
        return urls
    if _non_empty(venue.reference_url):
        urls.append(venue.reference_url.strip())
    for u in venue.extra_urls or []:
        if _non_empty(u):
            urls.append(u.strip())
    if venue.accessibility and venue.accessibility.urls:
        for u in venue.accessibility.urls:
            if _non_empty(u):
                urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _format_state_display(venue: VenueInfo) -> str:
    parts = []
    if _non_empty(venue.city):
        parts.append(venue.city.strip())
    if _non_empty(venue.state):
        parts.append(venue.state.strip())
    elif _non_empty(venue.state_code):
        parts.append(venue.state_code.strip())
    return ", ".join(parts) if parts else "the stated location"


# --------------------------------------------------------------------------- #
# Verification logic per region                                               #
# --------------------------------------------------------------------------- #
async def verify_region_block(
    evaluator: Evaluator,
    parent_node,
    region_key: str,
    extracted: VenuesExtraction,
) -> None:
    cfg = REGION_DEFS[region_key]
    region_label = cfg["label"]
    prefix = cfg["prefix"]
    state_codes = cfg["states_codes"]
    state_codes_str = ", ".join(state_codes)

    # Get venue info for region
    venue: Optional[VenueInfo] = getattr(extracted, region_key)

    # Create region-level container (non-critical to root to allow partial credit across regions)
    region_node = evaluator.add_parallel(
        id=f"{prefix}_Venue",
        desc=f"Identify a qualifying venue in the {region_label} U.S. region",
        parent=parent_node,
        critical=False,
    )

    # Identification group (critical within the region)
    ident_node = evaluator.add_parallel(
        id=f"{prefix}_Identification",
        desc=f"Provide complete identification information for the {region_label} venue",
        parent=region_node,
        critical=True,
    )

    # Official name existence
    evaluator.add_custom_node(
        result=bool(venue and _non_empty(venue.official_name)),
        id=f"{prefix}_Official_Name",
        desc="Provide the official name of the venue",
        parent=ident_node,
        critical=True,
    )

    # Street address existence (require some non-empty string; city/state may be verified later)
    evaluator.add_custom_node(
        result=bool(venue and _non_empty(venue.street_address)),
        id=f"{prefix}_Street_Address",
        desc="Provide the complete street address including city and state",
        parent=ident_node,
        critical=True,
    )

    # Reference URL existence (used as a prerequisite for URL-based checks)
    refurl_leaf = evaluator.add_custom_node(
        result=bool(venue and _non_empty(venue.reference_url)),
        id=f"{prefix}_Reference_URL",
        desc="Provide a reference URL to the venue's official website or official information page",
        parent=ident_node,
        critical=True,
    )

    # Prepare common data for verifications
    sources = _gather_sources(venue) if venue else []
    venue_name = venue.official_name if (venue and _non_empty(venue.official_name)) else "the venue"
    loc_display = _format_state_display(venue or VenueInfo())

    # Region verification (critical)
    region_leaf = evaluator.add_leaf(
        id=f"{prefix}_Region",
        desc=f"Verify the venue is located in a {region_label} U.S. state ({state_codes_str})",
        parent=region_node,
        critical=True,
    )
    region_claim = (
        f"The venue {venue_name} is located in {loc_display}. "
        f"Confirm the city and state from the provided official webpage(s). "
        f"The state must be one of: {state_codes_str} (2-letter codes for the {region_label})."
    )
    await evaluator.verify(
        claim=region_claim,
        node=region_leaf,
        sources=sources,
        additional_instruction=(
            "Use the official page(s) to confirm the venue's address (city and state). "
            f"Then check that the state's 2-letter postal code is one of: {state_codes_str}. "
            "Treat minor formatting differences as acceptable. If the page shows only the full state name, "
            "consider its standard postal code equivalent."
        ),
        extra_prerequisites=[refurl_leaf],
    )

    # Venue type verification (critical)
    vtype_leaf = evaluator.add_leaf(
        id=f"{prefix}_Venue_Type",
        desc="Verify the venue is an indoor arena with climate control and permanent roof structure",
        parent=region_node,
        critical=True,
    )
    vtype_claim = (
        f"{venue_name} is an indoor, climate-controlled arena with a permanent roof (not an outdoor or open-air venue)."
    )
    await evaluator.verify(
        claim=vtype_claim,
        node=vtype_leaf,
        sources=sources,
        additional_instruction=(
            "Look for descriptors such as 'indoor arena', 'multi-purpose arena', 'indoor venue', 'climate-controlled', "
            "or context implying a permanent roof (e.g., hosts basketball/hockey). If the page clearly indicates an "
            "outdoor amphitheater or open-air stadium, this should fail. When explicit 'roof' or 'climate control' "
            "phrasing is absent, reasonable inference from 'indoor arena' usage is acceptable."
        ),
        extra_prerequisites=[refurl_leaf],
    )

    # Concert events verification (critical)
    concerts_leaf = evaluator.add_leaf(
        id=f"{prefix}_Concert_Events",
        desc="Verify the venue regularly hosts concerts and entertainment events (not exclusively sports venues)",
        parent=region_node,
        critical=True,
    )
    concerts_claim = (
        f"{venue_name} regularly hosts concerts and entertainment events, not exclusively sports."
    )
    await evaluator.verify(
        claim=concerts_claim,
        node=concerts_leaf,
        sources=sources,
        additional_instruction=(
            "Check the events calendar, announcements, or 'events' pages for multiple concert or live entertainment "
            "listings across different dates. One-off references are weaker; look for evidence of recurring or regular "
            "concert programming."
        ),
        extra_prerequisites=[refurl_leaf],
    )

    # Capacity verification (critical)
    capacity_leaf = evaluator.add_leaf(
        id=f"{prefix}_Capacity",
        desc="Verify the venue's concert seating capacity is between 15,000 and 25,000",
        parent=region_node,
        critical=True,
    )
    capacity_text = venue.capacity if (venue and _non_empty(venue.capacity)) else "the stated capacity"
    capacity_claim = (
        f"The concert/general seating capacity for {venue_name} is {capacity_text} and is between 15,000 and 25,000 (inclusive)."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=sources,
        additional_instruction=(
            "Use the official venue/facts/seating pages. If multiple capacities are provided (basketball/hockey/concert), "
            "prefer 'concert' or 'maximum seating'. If only basketball/hockey capacity is stated and lies within the range, "
            "consider it acceptable for this check. Minor rounding differences are acceptable."
        ),
        extra_prerequisites=[refurl_leaf],
    )

    # Operational status as of March 2026 (critical)
    op_leaf = evaluator.add_leaf(
        id=f"{prefix}_Operational_Status",
        desc="Verify the venue is currently operational and hosting events as of March 2026",
        parent=region_node,
        critical=True,
    )
    op_claim = (
        f"As of March 2026, {venue_name} is operational and hosting events (e.g., shows or games scheduled in March 2026 or later)."
    )
    await evaluator.verify(
        claim=op_claim,
        node=op_leaf,
        sources=sources,
        additional_instruction=(
            "Check the official events/calendar pages for listings in March 2026 or later. "
            "If the site shows current season events, ticketing, or a clear statement of ongoing operations overlapping March 2026, "
            "that supports the claim."
        ),
        extra_prerequisites=[refurl_leaf],
    )

    # Accessibility verification (critical)
    access_leaf = evaluator.add_leaf(
        id=f"{prefix}_Accessibility",
        desc="Provide information about ADA-compliant wheelchair accessible seating features",
        parent=region_node,
        critical=True,
    )
    access_claim = (
        f"{venue_name} provides ADA-compliant wheelchair accessible seating features."
    )
    await evaluator.verify(
        claim=access_claim,
        node=access_leaf,
        sources=sources,
        additional_instruction=(
            "Look for mentions of 'accessible seating', 'wheelchair accessible seating', 'ADA seating', companion seating, "
            "or accessibility services/policies on the official site."
        ),
        extra_prerequisites=[refurl_leaf],
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
    # Initialize evaluator (root is non-critical by framework design; we enforce criticality at child levels)
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

    # Record region definitions as GT-style info (for transparency)
    evaluator.add_ground_truth({
        "region_state_codes": {
            k: v["states_codes"] for k, v in REGION_DEFS.items()
        },
        "requirement_summary": {
            "type": "indoor arena with permanent roof",
            "capacity_range": "15,000-25,000",
            "operational_as_of": "March 2026",
            "programming": "regularly hosts concerts/entertainment (not exclusively sports)",
        }
    })

    # Extract structured venue info from the answer
    extracted: VenuesExtraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_by_region",
    )

    # Build verification blocks per region
    await verify_region_block(evaluator, root, "northeast", extracted)
    await verify_region_block(evaluator, root, "southeast", extracted)
    await verify_region_block(evaluator, root, "midwest", extracted)
    await verify_region_block(evaluator, root, "west", extracted)

    # Return final structured summary
    return evaluator.get_summary()