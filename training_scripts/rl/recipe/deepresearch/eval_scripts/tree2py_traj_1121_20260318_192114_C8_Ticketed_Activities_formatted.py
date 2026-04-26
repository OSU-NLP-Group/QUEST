import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "concert_tour_march_2026_venues"
TASK_DESCRIPTION = (
    "A major production company is planning a 3-city concert tour across the United States in March 2026. "
    "They need to identify one suitable large-scale concert venue in each of the following cities: New York City, Los Angeles, and Seattle. "
    "For each venue, provide: the venue name and its location; verification that it has a seating capacity of at least 17,000 for concert events; "
    "confirmation that the venue has concert events scheduled during March 2026; evidence that the venue meets ADA accessibility requirements "
    "(providing accessible seating for individuals with disabilities, representing approximately 1% of total capacity); "
    "information about luxury amenities available (such as luxury suites, VIP seating, or club seats); and a reference URL supporting your answer. "
    "Identify the three venues (one in each city) that meet all these requirements for the March 2026 tour."
)

REQUIRED_CAPACITY_THRESHOLD = 17000
TARGET_MONTH = 3
TARGET_YEAR = 2026


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueEvidence(BaseModel):
    """URLs that support different aspects of the venue."""
    general_urls: List[str] = Field(default_factory=list)
    capacity_urls: List[str] = Field(default_factory=list)
    march_2026_urls: List[str] = Field(default_factory=list)
    accessibility_urls: List[str] = Field(default_factory=list)
    luxury_urls: List[str] = Field(default_factory=list)


class CityVenue(BaseModel):
    """Structured info for one city's chosen venue."""
    name: Optional[str] = None
    location: Optional[str] = None
    capacity_text: Optional[str] = None  # e.g., "20,000 for concerts" or "approx. 18,500"
    evidence: VenueEvidence = Field(default_factory=VenueEvidence)


class VenuesExtraction(BaseModel):
    """Top-level extraction containing venues for three required cities."""
    new_york_city: Optional[CityVenue] = None
    los_angeles: Optional[CityVenue] = None
    seattle: Optional[CityVenue] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return f"""
Extract exactly one venue per city for a March {TARGET_YEAR} U.S. concert tour from the provided answer text.
We need three cities: New York City, Los Angeles, and Seattle. For each city, extract:

For each city's object (new_york_city, los_angeles, seattle), include:
- name: The venue name as written in the answer.
- location: The location string for the venue as written in the answer (e.g., "New York, NY", "Brooklyn, NY", "Los Angeles, CA", "Inglewood, CA", "Seattle, WA").
- capacity_text: The seating capacity figure or description as written in the answer (if any, otherwise null).
- evidence: An object of URL arrays for supporting evidence. Only include URLs explicitly present in the answer (do NOT invent).
  • general_urls: URLs that generally describe the venue.
  • capacity_urls: URLs that support the stated or implied seating capacity for concerts.
  • march_2026_urls: URLs that show at least one concert event scheduled in March {TARGET_YEAR} for the venue (e.g., calendar, event listing, ticketing, or announcements).
  • accessibility_urls: URLs that support ADA-compliant accessible seating (e.g., accessibility policy pages, ADA statements).
  • luxury_urls: URLs that support availability of luxury amenities (suites, VIP seating, club seats, etc.).

Rules for URL extraction:
- Extract only valid, fully qualified URLs explicitly present in the answer.
- If the answer provides category-specific URLs (e.g., a specific link for ADA or suites), place them into the corresponding array.
- If the answer provides only general references (e.g., the main venue page) and no category-specific links, include those in general_urls.
- Deduplicate URLs if repeated.

If any field is not available for a city, set it to null (for strings) or an empty list (for URL arrays). Always include all three city objects, even if some fields are null.

Return a single JSON object with keys: new_york_city, los_angeles, seattle.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls or []:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if not uu:
            continue
        if uu not in seen:
            seen.add(uu)
            out.append(uu)
    return out


def all_urls(ev: VenueEvidence) -> List[str]:
    return _dedup(
        (ev.capacity_urls or [])
        + (ev.march_2026_urls or [])
        + (ev.accessibility_urls or [])
        + (ev.luxury_urls or [])
        + (ev.general_urls or [])
    )


def pick_sources(ev: VenueEvidence, kind: str) -> List[str]:
    by_kind = {
        "capacity": ev.capacity_urls or [],
        "march": ev.march_2026_urls or [],
        "access": ev.accessibility_urls or [],
        "luxury": ev.luxury_urls or [],
        "general": ev.general_urls or [],
    }
    primary = by_kind.get(kind, [])
    if primary:
        return _dedup(primary)
    if ev.general_urls:
        return _dedup(ev.general_urls)
    return _dedup(all_urls(ev))


def location_matches_city(location: Optional[str], city: str) -> bool:
    if not location:
        return False
    loc = location.lower()

    if city == "New York City":
        nyc_terms = [
            "new york", "new york city", "nyc", "manhattan", "brooklyn", "queens", "bronx", "staten island", "new york, ny"
        ]
        return any(term in loc for term in nyc_terms)

    if city == "Los Angeles":
        la_terms = [
            "los angeles", "los angeles, ca", "la, ca", "l.a.", "downtown la", "inglewood", "inglewood, ca", "hollywood", "exposition park"
        ]
        return any(term in loc for term in la_terms)

    if city == "Seattle":
        sea_terms = ["seattle", "seattle, wa"]
        return any(term in loc for term in sea_terms)

    return False


# --------------------------------------------------------------------------- #
# Verification per city                                                       #
# --------------------------------------------------------------------------- #
async def verify_city(
    evaluator: Evaluator,
    parent_node,
    city_key: str,
    city_display: str,
    venue: Optional[CityVenue],
) -> None:
    """
    Build verification subtree for one city.
    """
    city_node = evaluator.add_parallel(
        id=f"{city_key}_Venue",
        desc=f"Venue identified in {city_display} meeting all specified requirements",
        parent=parent_node,
        critical=False  # allow partial credit across cities
    )

    # Identification check: must have name, location, and at least one URL.
    has_name = bool(venue and venue.name and venue.name.strip())
    has_location = bool(venue and venue.location and venue.location.strip())
    location_ok = has_location and location_matches_city(venue.location, city_display)
    urls_present = bool(venue and all_urls(venue.evidence))

    evaluator.add_custom_node(
        result=(has_name and has_location and location_ok and urls_present),
        id=f"{city_key}_Venue_Identification",
        desc=f"Venue name and location in {city_display} provided with valid reference URL",
        parent=city_node,
        critical=True
    )

    # Capacity >= threshold
    capacity_leaf = evaluator.add_leaf(
        id=f"{city_key}_Venue_Capacity",
        desc=f"Venue has seating capacity of at least {REQUIRED_CAPACITY_THRESHOLD} for concerts",
        parent=city_node,
        critical=True
    )
    cap_sources = pick_sources(venue.evidence, "capacity") if venue else []
    if not cap_sources:
        capacity_leaf.score = 0.0
        capacity_leaf.status = "failed"
    else:
        cap_claim = (
            f"The seating capacity for concert events at '{venue.name}' is at least {REQUIRED_CAPACITY_THRESHOLD}."
            if venue and venue.name else
            f"The venue's concert seating capacity is at least {REQUIRED_CAPACITY_THRESHOLD}."
        )
        await evaluator.verify(
            claim=cap_claim,
            node=capacity_leaf,
            sources=cap_sources,
            additional_instruction=(
                "Look for explicit capacity figures related to concerts, end‑stage, or maximum event configuration. "
                f"Treat values >= {REQUIRED_CAPACITY_THRESHOLD} as satisfying the requirement. "
                "If multiple capacities are listed (e.g., basketball vs. concerts), prefer the concert or maximum configuration. "
                "Allow approximate wordings (e.g., ~18,000) if clearly >= threshold."
            ),
        )

    # March 2026 availability (concert event scheduled)
    march_leaf = evaluator.add_leaf(
        id=f"{city_key}_March_{TARGET_YEAR}_Availability",
        desc=f"Venue has confirmed concert events scheduled in March {TARGET_YEAR}",
        parent=city_node,
        critical=True
    )
    march_sources = pick_sources(venue.evidence, "march") if venue else []
    if not march_sources:
        march_leaf.score = 0.0
        march_leaf.status = "failed"
    else:
        march_claim = (
            f"At least one concert event is scheduled at '{venue.name}' in March {TARGET_YEAR}."
            if venue and venue.name else
            f"At least one concert event is scheduled in March {TARGET_YEAR} at this venue."
        )
        await evaluator.verify(
            claim=march_claim,
            node=march_leaf,
            sources=march_sources,
            additional_instruction=(
                f"Verify from the provided page(s) that there is at least one concert event dated between {TARGET_YEAR}-03-01 and {TARGET_YEAR}-03-31 inclusive. "
                "Accept official calendars, event pages, or reputable ticketing listings that clearly show the event date and month/year."
            ),
        )

    # Accessibility features (ADA-compliant accessible seating approx. 1% of capacity)
    access_leaf = evaluator.add_leaf(
        id=f"{city_key}_Accessibility_Features",
        desc="Venue provides ADA-compliant accessible seating (approximately 1% of capacity reserved for individuals with disabilities)",
        parent=city_node,
        critical=True
    )
    access_sources = pick_sources(venue.evidence, "access") if venue else []
    if not access_sources:
        access_leaf.score = 0.0
        access_leaf.status = "failed"
    else:
        access_claim = (
            f"'{venue.name}' provides ADA-compliant accessible seating, including wheelchair-accessible seating and companion seating, meeting ADA requirements (about 1% of seating)."
            if venue and venue.name else
            "The venue provides ADA-compliant accessible seating, including wheelchair-accessible and companion seating, meeting ADA requirements (about 1% of seating)."
        )
        await evaluator.verify(
            claim=access_claim,
            node=access_leaf,
            sources=access_sources,
            additional_instruction=(
                "Confirm that the venue offers ADA/accessible seating and complies with ADA requirements. "
                "An explicit percentage around 1% is ideal, but if the policy explicitly references ADA-compliant accessible seating availability consistent with ADA requirements, count that as sufficient evidence."
            ),
        )

    # Luxury amenities (suites, VIP, club seats)
    luxury_leaf = evaluator.add_leaf(
        id=f"{city_key}_Luxury_Amenities",
        desc="Venue offers luxury amenities such as suites, VIP seating, or club seats",
        parent=city_node,
        critical=True
    )
    luxury_sources = pick_sources(venue.evidence, "luxury") if venue else []
    if not luxury_sources:
        luxury_leaf.score = 0.0
        luxury_leaf.status = "failed"
    else:
        luxury_claim = (
            f"'{venue.name}' offers premium/luxury amenities such as luxury suites, VIP seating, or club seats."
            if venue and venue.name else
            "The venue offers premium/luxury amenities such as suites, VIP seating, or club seats."
        )
        await evaluator.verify(
            claim=luxury_claim,
            node=luxury_leaf,
            sources=luxury_sources,
            additional_instruction=(
                "Look for terms like suites, luxury suites, VIP, premium seating, club seats, clubs, hospitality, or similar premium offerings."
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
    Evaluate an answer for the March 2026 3-city concert tour venue planning task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # evaluate three cities independently
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

    # Extract venues and their evidences from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Add a small custom info block for the threshold/date used
    evaluator.add_custom_info(
        info={
            "capacity_threshold": REQUIRED_CAPACITY_THRESHOLD,
            "target_month": TARGET_MONTH,
            "target_year": TARGET_YEAR
        },
        info_type="evaluation_parameters",
        info_name="eval_params"
    )

    # Build verification subtrees for each city
    await verify_city(
        evaluator,
        root,
        city_key="NYC",
        city_display="New York City",
        venue=extracted.new_york_city,
    )
    await verify_city(
        evaluator,
        root,
        city_key="LA",
        city_display="Los Angeles",
        venue=extracted.los_angeles,
    )
    await verify_city(
        evaluator,
        root,
        city_key="Seattle",
        city_display="Seattle",
        venue=extracted.seattle,
    )

    # Return structured summary
    return evaluator.get_summary()