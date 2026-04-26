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
TASK_ID = "national_concert_venues_4_cities"
TASK_DESCRIPTION = """
You are planning a national concert tour for a major music artist and need to identify suitable venue options. Find 4 indoor concert arenas located in 4 different major U.S. cities (each with a population exceeding 500,000) that meet the following requirements:

1. Each venue must have a seating capacity between 15,000 and 25,000 for concert configurations
2. Each venue must have a stage width of at least 40 feet to accommodate the full production setup
3. Each venue must be classified as an arena or indoor concert venue (not an outdoor amphitheater or stadium)
4. All 4 venues must be in different cities

For each venue, provide:
- The venue name
- The city and state where it is located
- The concert seating capacity
- The stage width (or confirm it meets the 40-foot minimum requirement)
- A reference URL that verifies these details
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    venue_type: Optional[str] = None
    concert_capacity: Optional[str] = None
    stage_width: Optional[str] = None
    stage_width_meets_minimum: Optional[bool] = None
    url: Optional[str] = None
    extra_urls: List[str] = Field(default_factory=list)


class VenueList(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract up to 6 venue entries mentioned in the answer for the national tour venue shortlist. For each venue, extract the following fields exactly as presented in the answer text:

- name: Venue name
- city: City name
- state: State abbreviation or full state name
- country: Country (if provided; otherwise leave null)
- venue_type: Stated classification (e.g., "indoor arena", "multi-purpose arena", "stadium", "amphitheater", etc.)
- concert_capacity: The concert seating capacity as presented (e.g., "18,000", "approx. 20,000", "18,000–20,000 for concerts")
- stage_width: The stage width figure if provided (include units as written, e.g., "50 ft", "15 m")
- stage_width_meets_minimum: A boolean true/false only if the answer explicitly states the stage width meets or exceeds the 40 ft minimum without giving a numeric; otherwise null
- url: A primary reference URL for the venue details (e.g., official site, operator page, or a reliable reference cited in the answer)
- extra_urls: Any additional URLs in the answer tied to this venue (e.g., city or census links for population, venue spec pages). Include only valid URLs and avoid duplicates. Do not include `url` again here.

Rules:
1) Only extract information explicitly present in the answer.
2) If a field is missing, set it to null (or an empty list for extra_urls).
3) For URLs, extract the actual URL strings (including from markdown links).
4) Do not invent numeric values or units. Keep them as strings if present.
5) Preserve the order venues appear in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def first_n_venues(venues: List[VenueItem], n: int = 4) -> List[VenueItem]:
    out = list(venues[:n])
    while len(out) < n:
        out.append(VenueItem())
    return out


def dedup_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    results: List[str] = []
    for u in urls:
        if not u:
            continue
        s = u.strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            results.append(s)
    return results


def ordinal(idx_zero_based: int) -> str:
    mapping = {0: "First", 1: "Second", 2: "Third", 3: "Fourth", 4: "Fifth", 5: "Sixth"}
    return mapping.get(idx_zero_based, f"#{idx_zero_based+1}")


def city_state_str(venue: VenueItem) -> str:
    c = (venue.city or "").strip()
    s = (venue.state or "").strip()
    if c and s:
        return f"{c}, {s}"
    return c or s or "unknown location"


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    venue_index: int,
    prior_cities: List[str],
) -> None:
    """
    Build verification subtree and run checks for a single venue.
    All critical checks are independent (parallel) but we also add a critical 'reference' group
    so other checks have a meaningful source. Even if checks run, the final aggregation will
    enforce critical criteria.
    """

    venue_desc = f"{ordinal(venue_index)} venue meeting all specified requirements"
    venue_node = evaluator.add_parallel(
        id=f"venue_{venue_index+1}",
        desc=venue_desc,
        parent=parent_node,
        critical=False
    )

    # Collect sources for this venue
    all_sources = dedup_urls(([venue.url] if venue.url else []) + (venue.extra_urls or []))

    # Reference group: critical – URL must be present and page must be about the venue
    ref_group = evaluator.add_parallel(
        id=f"venue_{venue_index+1}_reference_group",
        desc="Reference presence and relevance verification",
        parent=venue_node,
        critical=True
    )

    # 1) URL presence (custom binary)
    evaluator.add_custom_node(
        result=bool(venue.url and venue.url.strip()),
        id=f"venue_{venue_index+1}_url_present",
        desc="A reference URL is provided for the venue",
        parent=ref_group,
        critical=True
    )

    # 2) URL relevance: page is about the venue (name + location if available)
    about_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index+1}_reference",
        desc="Valid URL reference provided that confirms the venue details",
        parent=ref_group,
        critical=True
    )

    about_claim_parts = []
    if venue.name:
        about_claim_parts.append(f"this page is about the venue '{venue.name}'")
    loc_str = city_state_str(venue)
    if loc_str and loc_str != "unknown location":
        about_claim_parts.append(f"located in {loc_str}, United States")
    about_claim_text = " and ".join(about_claim_parts) if about_claim_parts else "this page is about the specified venue"
    about_claim = f"Confirm that {about_claim_text}."
    await evaluator.verify(
        claim=about_claim,
        node=about_leaf,
        sources=venue.url if venue.url else None,
        additional_instruction="Treat the claim as supported if the page is clearly the official or authoritative page for the venue and identifies the venue and its location."
    )

    # Capacity: critical
    capacity_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index+1}_capacity",
        desc="Venue has a seating capacity between 15,000 and 25,000 for concerts",
        parent=venue_node,
        critical=True
    )
    if venue.concert_capacity and venue.concert_capacity.strip():
        cap_claim = f"The concert seating capacity for {venue.name or 'the venue'} is {venue.concert_capacity.strip()}, and it is between 15,000 and 25,000."
    else:
        cap_claim = f"The concert seating capacity for {venue.name or 'the venue'} is between 15,000 and 25,000."
    await evaluator.verify(
        claim=cap_claim,
        node=capacity_leaf,
        sources=all_sources if all_sources else None,
        additional_instruction=(
            "Prioritize 'concert' capacity if explicitly stated. If only a general capacity is given, "
            "accept if it lies within 15,000–25,000. If multiple figures are shown, use the figure applicable to concerts. "
            "The claim is supported if the webpage explicitly lists a concert capacity in range or a general capacity in range."
        )
    )

    # Stage width: critical
    stage_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index+1}_stage",
        desc="Venue has stage dimensions suitable for a full production (minimum 40 feet wide)",
        parent=venue_node,
        critical=True
    )
    if venue.stage_width and venue.stage_width.strip():
        stage_claim = f"The stage width at {venue.name or 'the venue'} is {venue.stage_width.strip()}, which is at least 40 feet."
    elif venue.stage_width_meets_minimum is True:
        stage_claim = f"The stage width at {venue.name or 'the venue'} is at least 40 feet."
    else:
        stage_claim = f"The stage width at {venue.name or 'the venue'} is at least 40 feet."
    await evaluator.verify(
        claim=stage_claim,
        node=stage_leaf,
        sources=all_sources if all_sources else None,
        additional_instruction=(
            "Verify that the stage width meets or exceeds 40 feet (≈ 12.2 meters). "
            "If only metric units are provided, convert accordingly; 12.2 m or larger qualifies. "
            "Do not accept 12.0 m (≈ 39.37 ft). If the page explicitly states a minimum stage width that meets/exceeds 40 ft, accept."
        )
    )

    # Venue type: critical
    type_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index+1}_type",
        desc="Venue is classified as an arena or indoor concert venue",
        parent=venue_node,
        critical=True
    )
    type_claim = (
        f"{venue.name or 'The venue'} is an indoor arena or indoor concert venue, not an outdoor amphitheater or outdoor stadium."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=all_sources if all_sources else None,
        additional_instruction=(
            "Accept if the page clearly identifies the venue as an indoor arena (multi-purpose arena, indoor arena, etc.). "
            "Reject if it is characterized as an outdoor amphitheater, stadium, ballpark, or similar outdoor-only facility."
        )
    )

    # Location group: critical – major US city and uniqueness vs prior
    loc_group = evaluator.add_parallel(
        id=f"venue_{venue_index+1}_location_group",
        desc="Location verification",
        parent=venue_node,
        critical=True
    )

    # Major US city (>500k population)
    loc_major_leaf = evaluator.add_leaf(
        id=f"venue_{venue_index+1}_location_major_city",
        desc="Venue is located in a major U.S. city (population > 500,000)",
        parent=loc_group,
        critical=True
    )
    city = (venue.city or "").strip()
    state = (venue.state or "").strip()
    loc_claim = (
        f"The venue is located in {city}, {state}, United States, and the city's population exceeds 500,000."
        if city and state else
        f"The venue is in a major U.S. city with population exceeding 500,000."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_major_leaf,
        sources=all_sources if all_sources else None,
        additional_instruction=(
            "Verify both location (city/state in the U.S.) and the city's population threshold (>500,000). "
            "If population is not stated on the venue page, use any additional provided URL(s) (e.g., city government, census, Wikipedia) in extra URLs. "
            "If no provided URL supports the population threshold, mark as not supported."
        )
    )

    # Uniqueness vs prior cities (for second to fourth venues)
    if prior_cities:
        unique_leaf = evaluator.add_leaf(
            id=f"venue_{venue_index+1}_location_unique",
            desc="Venue city is different from previously selected cities",
            parent=loc_group,
            critical=True
        )
        prior_list_str = "; ".join(prior_cities)
        uniq_claim = (
            f"The city for this venue ({city_state_str(venue)}) is different from each of the previously selected cities: {prior_list_str}."
        )
        await evaluator.verify(
            claim=uniq_claim,
            node=unique_leaf,
            sources=None,
            additional_instruction=(
                "Perform a logical comparison of city names. Treat common variants as equivalent (e.g., "
                "'NYC' ≈ 'New York', 'New York City' ≈ 'New York', 'Washington' ≈ 'Washington, DC'). "
                "Ignore case and punctuation. Return Correct only if this city's identity is distinct from all prior cities."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 4 indoor concert arenas task.
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

    # Extract venues from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenueList,
        extraction_name="venues_extraction"
    )

    venues = first_n_venues(extracted.venues, 4)

    # Track previously used city identities for uniqueness checks
    prior_city_labels: List[str] = []

    # Build verification tree and run checks for each of the 4 venues
    for idx, v in enumerate(venues):
        await verify_venue(
            evaluator=evaluator,
            parent_node=root,
            venue=v,
            venue_index=idx,
            prior_cities=prior_city_labels.copy()
        )
        # Update prior cities list with normalized label
        label = city_state_str(v)
        if label and label != "unknown location":
            prior_city_labels.append(label)

    # Add custom info to help debugging
    evaluator.add_custom_info(
        {
            "extracted_cities": [city_state_str(v) for v in venues],
            "extracted_venue_names": [v.name for v in venues],
            "note": "Cities must be distinct and each must exceed 500,000 population, supported by provided URLs."
        },
        info_type="extraction_summary",
        info_name="extraction_summary"
    )

    return evaluator.get_summary()