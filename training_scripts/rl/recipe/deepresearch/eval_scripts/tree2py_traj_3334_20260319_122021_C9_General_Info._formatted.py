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
TASK_ID = "corp_training_venues_spring2026"
TASK_DESCRIPTION = (
    "I'm planning a corporate training tour across the United States in spring 2026 (April-May timeframe) and need to "
    "identify 4 suitable venue options in 4 different major US cities. For each venue, I need the following information: "
    "1) The venue name and its city location (must be in New York, Los Angeles, Chicago, Houston, Phoenix, Philadelphia, "
    "San Antonio, San Diego, Dallas, or San Jose); 2) Confirmation that the main conference/meeting space can accommodate "
    "at least 500 people in theater-style seating; 3) Verification that the venue supports multiple seating configurations "
    "(at least theater-style and classroom-style); 4) Confirmation of ADA compliance and wheelchair accessibility for "
    "entrances, restrooms, and meeting spaces; 5) Availability of built-in audiovisual equipment (projection systems or "
    "large displays and sound systems); 6) High-speed internet connectivity (minimum 100 Mbps or business-grade service); "
    "7) On-site catering services or exclusive catering partnerships; 8) At least one hotel with 100+ rooms within a "
    "10-minute walk OR shuttle service to nearby accommodations; 9) Either on-site parking for at least 200 vehicles OR "
    "location within a 10-minute walk of public transportation; 10) A reference URL (website or listing page) that "
    "confirms the venue's specifications. Each of the 4 venues must be in a different city, and all venues should meet "
    "all the requirements listed above."
)

ALLOWED_CITIES = [
    "New York",
    "Los Angeles",
    "Chicago",
    "Houston",
    "Phoenix",
    "Philadelphia",
    "San Antonio",
    "San Diego",
    "Dallas",
    "San Jose",
]

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to 6 proposed venues for a corporate training tour from the provided answer.

    For each venue, extract:
    - name: The specific venue name (e.g., "Javits Center", "Phoenix Convention Center")
    - city: The city where the venue is located (as written in the answer; do not infer)
    - sources: A list of ALL URLs cited for this venue in the answer. Include the venue’s official site and any supporting pages that the answer references for capacity/layouts, floor plans, ADA/accessibility details, audiovisual/internet specs, catering, parking/transit, or nearby hotels (100+ rooms) and walking/shuttle details. Include mapping links only if explicitly provided. Do not invent URLs.

    Rules:
    - Only extract venues explicitly listed in the answer.
    - Return a JSON object with a 'venues' array. Each element must contain the three fields above.
    - If a field is not mentioned for a venue, set it to null (for strings) or [] (for arrays).
    - Do not add venues beyond those explicitly listed in the answer.
    - Keep URLs exactly as written in the answer; convert markdown links to direct URLs.

    We will later evaluate only the first 4 venues. Still, extract up to 6 if present.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(n, f"#{n}")


def dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def canonical_city(city: Optional[str]) -> Optional[str]:
    if not city:
        return None
    s = city.strip().lower()
    # Common synonyms/variants to improve fairness in uniqueness check
    mapping = {
        "nyc": "new york",
        "new york city": "new york",
        "manhattan": "new york",
        "los angeles": "los angeles",
        "la": "los angeles",
        "l.a.": "los angeles",
        "chicago": "chicago",
        "houston": "houston",
        "phoenix": "phoenix",
        "phx": "phoenix",
        "philadelphia": "philadelphia",
        "philly": "philadelphia",
        "san antonio": "san antonio",
        "san diego": "san diego",
        "dallas": "dallas",
        "san jose": "san jose",
        "san josé": "san jose",
    }
    return mapping.get(s, s)


def city_in_allowed(city: Optional[str]) -> bool:
    if not city:
        return False
    can = canonical_city(city)
    return any(can == c.lower() for c in ALLOWED_CITIES)


# --------------------------------------------------------------------------- #
# Verification logic per venue                                                #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent,
    item: VenueItem,
    idx: int,
) -> None:
    vidx = idx + 1
    vtitle = f"{ordinal(vidx)} venue meeting all specified requirements"

    venue_node = evaluator.add_parallel(
        id=f"venue_{vidx}",
        desc=vtitle,
        parent=parent,
        critical=False,
    )

    # Consolidate sources for this venue
    sources = dedupe_urls(item.sources)

    # 1) Basic identification (critical)
    basic_node = evaluator.add_parallel(
        id=f"basic_identification_v{vidx}",
        desc="Venue identification and location information",
        parent=venue_node,
        critical=True,
    )

    # 1a) City location (critical): verify the venue is stated to be in the provided city AND that city is one of the allowed cities
    city_node = evaluator.add_leaf(
        id=f"city_location_v{vidx}",
        desc="Venue is located in one of the specified major US cities (New York, Los Angeles, Chicago, Houston, Phoenix, Philadelphia, San Antonio, San Diego, Dallas, or San Jose)",
        parent=basic_node,
        critical=True,
    )
    venue_name_for_claim = item.name or "the venue"
    city_for_claim = item.city or "UNKNOWN CITY"
    await evaluator.verify(
        claim=(
            f"According to the provided webpage(s), the venue named '{venue_name_for_claim}' is located in '{city_for_claim}', "
            f"and this city is one of the allowed major US cities for this task "
            f"(New York, Los Angeles, Chicago, Houston, Phoenix, Philadelphia, San Antonio, San Diego, Dallas, or San Jose)."
        ),
        node=city_node,
        sources=sources,
        additional_instruction=(
            "Use the webpage(s) to confirm the venue's city. Accept reasonable variants and neighborhood names that clearly "
            "correspond to the city (e.g., 'Manhattan' for New York City, 'LA' for Los Angeles). The membership-in-allowed-list "
            "check can be validated by comparing the extracted city text to the allowed list."
        ),
    )

    # 1b) Venue name provided (critical) - existence check
    evaluator.add_custom_node(
        result=bool(item.name and item.name.strip()),
        id=f"venue_name_v{vidx}",
        desc="Specific venue name is provided",
        parent=basic_node,
        critical=True,
    )

    # 1c) Availability timeframe (critical)
    avail_node = evaluator.add_leaf(
        id=f"availability_timeframe_v{vidx}",
        desc="Venue is available for booking between April 1, 2026, and May 31, 2026",
        parent=basic_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The venue accepts event bookings and can host corporate events during the April–May 2026 timeframe "
            "(between April 1, 2026 and May 31, 2026)."
        ),
        node=avail_node,
        sources=sources,
        additional_instruction=(
            "Treat explicit calendars, seasonal schedules, or general year-round booking statements as sufficient. "
            "If a page indicates they are available for corporate events with no blackout/closure during April–May 2026, "
            "consider the condition met. If there is clear evidence of closure or no bookings in that period, it should fail."
        ),
    )

    # 2) Capacity and space (critical)
    cap_node = evaluator.add_parallel(
        id=f"capacity_and_space_v{vidx}",
        desc="Venue capacity and configuration requirements",
        parent=venue_node,
        critical=True,
    )

    # 2a) Minimum capacity (critical)
    capacity_leaf = evaluator.add_leaf(
        id=f"minimum_capacity_v{vidx}",
        desc="Main conference/meeting space has minimum capacity of 500 people in theater-style seating",
        parent=cap_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The venue has at least one primary conference/meeting space that can accommodate 500 or more people in "
            "theater-style (theatre/auditorium-style) seating."
        ),
        node=capacity_leaf,
        sources=sources,
        additional_instruction=(
            "Look for capacity charts, floor plans, or specs referencing 'theater' or 'auditorium' setups. "
            "If multiple rooms are listed, it is sufficient that at least one main room meets the 500 theater-style threshold."
        ),
    )

    # 2b) Configuration flexibility (critical)
    config_leaf = evaluator.add_leaf(
        id=f"configuration_flexibility_v{vidx}",
        desc="Venue supports multiple seating configurations including at least theater-style and classroom-style setups",
        parent=cap_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The venue supports multiple seating configurations and specifically offers both theater-style and classroom-style "
            "setups for its meeting spaces."
        ),
        node=config_leaf,
        sources=sources,
        additional_instruction=(
            "Accept if the venue lists available setups including both 'theater' (or 'theatre'/'auditorium') and 'classroom'. "
            "Mentions may appear on setup charts, floor plan PDFs, or event specs pages."
        ),
    )

    # 3) Accessibility features (critical)
    access_node = evaluator.add_parallel(
        id=f"accessibility_features_v{vidx}",
        desc="ADA compliance and accessibility requirements",
        parent=venue_node,
        critical=True,
    )

    ada_leaf = evaluator.add_leaf(
        id=f"ada_compliance_v{vidx}",
        desc="Venue is ADA-compliant",
        parent=access_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The venue is ADA-compliant (complies with the Americans with Disabilities Act).",
        node=ada_leaf,
        sources=sources,
        additional_instruction=(
            "Accept explicit ADA statements or equivalent accessibility policies. "
            "Mentions like 'ADA compliant', 'fully accessible', or compliance statements are sufficient."
        ),
    )

    wheelchair_leaf = evaluator.add_leaf(
        id=f"wheelchair_access_v{vidx}",
        desc="Wheelchair-accessible entrances, restrooms, and meeting spaces are available",
        parent=access_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The venue provides wheelchair accessibility for entrances, restrooms, and meeting spaces (e.g., ramps/elevators, "
            "accessible restrooms, accessible meeting rooms)."
        ),
        node=wheelchair_leaf,
        sources=sources,
        additional_instruction=(
            "Look for specific mentions of accessible entrances/paths, accessible restrooms, elevators, ramps, or wheelchair "
            "accessibility statements for meeting spaces."
        ),
    )

    # 4) Technical and services (critical)
    tech_node = evaluator.add_parallel(
        id=f"technical_and_services_v{vidx}",
        desc="Technical infrastructure and service requirements",
        parent=venue_node,
        critical=True,
    )

    av_leaf = evaluator.add_leaf(
        id=f"audiovisual_equipment_v{vidx}",
        desc="Built-in audiovisual equipment including projection systems or large displays and sound systems",
        parent=tech_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The venue provides built-in audiovisual equipment suitable for conferences, including projectors or large displays "
            "and an in-room sound/PA system."
        ),
        node=av_leaf,
        sources=sources,
        additional_instruction=(
            "Accept mentions of built-in projectors, LED/LCD walls, drop-down screens, or large displays, and a sound system "
            "(PA/speakers/mixers) provided by the venue."
        ),
    )

    internet_leaf = evaluator.add_leaf(
        id=f"internet_connectivity_v{vidx}",
        desc="High-speed internet connectivity available (minimum 100 Mbps or equivalent business-grade service)",
        parent=tech_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The venue offers high-speed internet of at least 100 Mbps or equivalent business-grade service (e.g., gigabit fiber, "
            "dedicated bandwidth) for events."
        ),
        node=internet_leaf,
        sources=sources,
        additional_instruction=(
            "If an exact Mbps figure is not given, accept clear indications of business-grade/dedicated lines, gigabit fiber, "
            "1 Gbps, or similar enterprise connectivity appropriate for large conferences."
        ),
    )

    catering_leaf = evaluator.add_leaf(
        id=f"catering_services_v{vidx}",
        desc="On-site catering services or exclusive catering partnerships available",
        parent=tech_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The venue provides on-site catering or has exclusive/preferred catering partnerships for events.",
        node=catering_leaf,
        sources=sources,
        additional_instruction="Accept in-house kitchens, banquet/catering teams, or exclusive/preferred vendor lists.",
    )

    # 5) Location and access (critical)
    loc_node = evaluator.add_parallel(
        id=f"location_and_access_v{vidx}",
        desc="Proximity to accommodations and transportation access",
        parent=venue_node,
        critical=True,
    )

    hotels_leaf = evaluator.add_leaf(
        id=f"nearby_accommodations_v{vidx}",
        desc="At least one hotel with 100+ rooms within 10-minute walk or shuttle service to nearby accommodations",
        parent=loc_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "There is at least one hotel with 100 or more rooms within a 10-minute walk of the venue, "
            "OR the venue provides shuttle service to nearby accommodations."
        ),
        node=hotels_leaf,
        sources=sources,
        additional_instruction=(
            "Evidence can come from the venue's site, hotel pages listing room counts, or third-party listings mentioned in the answer. "
            "Treat roughly 0.5 miles (0.8 km) as ~10 minutes walking. If a shuttle to nearby hotels is stated, that also satisfies this."
        ),
    )

    parking_leaf = evaluator.add_leaf(
        id=f"parking_or_transit_v{vidx}",
        desc="Either on-site parking for at least 200 vehicles OR location within 10-minute walk of public transportation",
        parent=loc_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "Either the venue provides on-site parking for at least 200 vehicles, "
            "or the venue is within a 10-minute walk of public transportation (e.g., subway, light rail, or major bus lines)."
        ),
        node=parking_leaf,
        sources=sources,
        additional_instruction=(
            "Accept explicit parking capacity statements, garage/lot capacity of 200+, or clear proximity to transit stops/stations. "
            "Treat roughly 0.5 miles (0.8 km) as ~10 minutes walking."
        ),
    )

    # 6) Verification reference (critical)
    ref_leaf = evaluator.add_leaf(
        id=f"verification_reference_v{vidx}",
        desc="Publicly accessible reference URL confirming venue specifications",
        parent=venue_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "At least one of the provided URLs is a publicly accessible webpage about the venue that confirms its event specifications "
            "(such as capacity or seating configurations, AV/internet, accessibility, catering, parking/transit, or nearby hotels)."
        ),
        node=ref_leaf,
        sources=sources if sources else None,
        additional_instruction=(
            "The page should be reachable and relevant to the venue. Accept official venue pages or authoritative listings that "
            "explicitly mention event-related specifications."
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

    # Extract structured venues info
    venues_data = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Keep first 4 venues; pad if fewer
    venues = list(venues_data.venues[:4])
    while len(venues) < 4:
        venues.append(VenueItem())

    # Add GT/constraints info
    evaluator.add_ground_truth(
        {
            "allowed_cities": ALLOWED_CITIES,
            "requirements": [
                ">=500 theater-style capacity in main meeting space",
                "Theater and classroom seating supported",
                "ADA compliant + wheelchair-accessible entrances/restrooms/meeting spaces",
                "Built-in AV (projection/large display + sound system)",
                "High-speed internet >=100 Mbps or business-grade service",
                "On-site catering or exclusive/ preferred partnerships",
                "Hotel with 100+ rooms within 10-minute walk OR shuttle to hotels",
                "On-site parking >=200 vehicles OR within 10-minute walk to public transit",
                "Reference URL(s) confirming specs",
                "4 different cities"
            ],
            "timeframe": "April 1, 2026 – May 31, 2026",
        },
        gt_type="constraints",
    )

    # Build venue verifications
    for i in range(4):
        await verify_single_venue(evaluator, root, venues[i], i)

    # Unique cities verification (critical)
    extracted_cities = []
    for v in venues:
        if v.city and v.city.strip():
            extracted_cities.append(canonical_city(v.city.strip()))
        else:
            extracted_cities.append(None)

    # Must have 4 non-empty and all different
    non_empty = [c for c in extracted_cities if c]
    unique_ok = (len(non_empty) == 4) and (len(set(non_empty)) == 4)

    evaluator.add_custom_node(
        result=unique_ok,
        id="unique_cities_verification",
        desc="All 4 venues are located in different cities from each other",
        parent=root,
        critical=True,
    )

    return evaluator.get_summary()