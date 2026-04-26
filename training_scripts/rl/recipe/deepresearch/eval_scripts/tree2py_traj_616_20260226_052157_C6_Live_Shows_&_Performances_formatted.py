import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_performing_arts_venues"
TASK_DESCRIPTION = (
    "Identify four specific performing arts venues across the United States, each meeting distinct historical and "
    "capacity criteria. For each venue, provide: name, city and state, founding/opening year, relevant capacity "
    "information, and a reference URL that supports the identification."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Venue(BaseModel):
    """Unified model representing a venue entry extracted from the answer."""
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    year: Optional[str] = None  # founding/opening year as free-form text
    capacity_info: Optional[str] = None  # free-form capacity text
    total_capacity: Optional[str] = None  # e.g., "9,500", "over 9,000"
    seat_count: Optional[str] = None  # e.g., "1,100 seats"
    number_of_theaters: Optional[str] = None  # e.g., "3 theaters", "multiple venues"
    continuous_operation: Optional[str] = None  # e.g., "yes", "continuously operating since 1805"
    broadway_seat_requirement_met: Optional[str] = None  # e.g., "yes", ">=500 seats"
    architecture_style: Optional[str] = None  # e.g., "movie palace", "Art Deco movie palace"
    still_operating_or_restored: Optional[str] = None  # e.g., "restored and operating"
    outside_nyc: Optional[str] = None  # e.g., "yes"
    western_state: Optional[str] = None  # e.g., "yes"
    source_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    """Container for the four venues."""
    venue1: Optional[Venue] = None  # Oldest continuously operating theater in America; founded 1800–1815
    venue2: Optional[Venue] = None  # Oldest continuously operating legitimate Broadway theater; opened 1900–1910; >=500 seats
    venue3: Optional[Venue] = None  # Largest performing arts center outside NYC; total capacity > 9,000 across multiple venues
    venue4: Optional[Venue] = None  # Historic 1920s movie palace; Western U.S. state; opened 1920–1929


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract four specific venue entries from the answer, corresponding to these categories:
    1) venue1: America's oldest continuously operating theater (founded between 1800–1815)
    2) venue2: Broadway's oldest continuously operating legitimate theater (opened in 1900–1910; Broadway classification requires at least 500 seats)
    3) venue3: The largest performing arts center in the U.S. outside New York City (total seating capacity > 9,000 across multiple theaters/venues)
    4) venue4: A historic movie palace theater from the 1920s era (opened 1920–1929) located in a Western U.S. state (west of the Mississippi River)

    For each venue, extract the following fields (return null if the answer does not provide the field):
    - name: The venue's official name
    - city: City location
    - state: State location
    - year: Founding/opening year (as presented in the answer, free-form)
    - capacity_info: Any capacity-related information presented (free-form)
    - total_capacity: Total seating capacity figure (if applicable; free-form, e.g., "9,500")
    - seat_count: Single-venue seat count figure (if applicable; free-form, e.g., "1,100 seats")
    - number_of_theaters: Mention of multiple venues/theaters (free-form)
    - continuous_operation: Whether the answer claims continuous operation (free-form, e.g., "yes")
    - broadway_seat_requirement_met: Whether the answer claims Broadway classification (>=500 seats) is met (free-form)
    - architecture_style: Notable style (e.g., "movie palace")
    - still_operating_or_restored: Whether it is still operating or has been restored (free-form)
    - outside_nyc: Whether the venue is outside NYC (free-form)
    - western_state: Whether the state is west of the Mississippi River (free-form)
    - source_urls: An array of URLs cited to support this venue's identification and details

    Return a JSON with keys venue1, venue2, venue3, venue4. If any venue is missing, return null for that venue.
    IMPORTANT: Only extract URLs explicitly present in the answer. Do not invent any URLs.
    """


# --------------------------------------------------------------------------- #
# Helper parsing functions                                                    #
# --------------------------------------------------------------------------- #
def parse_first_year(text: Optional[str]) -> Optional[int]:
    """Extract the first 4-digit year present in the text."""
    if not text:
        return None
    m = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def parse_max_int(text: Optional[str]) -> Optional[int]:
    """
    Extract the maximum integer found in the text (e.g., "9,500+", "1,100 seats" -> returns 9500 or 1100).
    Useful for capacity checks.
    """
    if not text:
        return None
    nums = re.findall(r"\b\d{1,3}(?:,\d{3})+|\b\d+\b", text)
    if not nums:
        return None
    values = []
    for n in nums:
        try:
            values.append(int(n.replace(",", "")))
        except Exception:
            continue
    return max(values) if values else None


def has_valid_sources(urls: Optional[List[str]]) -> bool:
    """Check presence of at least one plausible URL."""
    return bool(urls) and any(isinstance(u, str) and len(u.strip()) >= 8 for u in urls or [])


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_venue_1(evaluator: Evaluator, root_node, v: Optional[Venue]) -> None:
    """
    Venue 1: America's oldest continuously operating theater; founded between 1800 and 1815.
    """
    node = evaluator.add_sequential(
        id="venue_1_oldest_us_theater",
        desc="Identify America's oldest continuously operating theater",
        parent=root_node,
        critical=False
    )

    # Required info gate
    req_node = evaluator.add_custom_node(
        result=bool(v and v.name and has_valid_sources(v.source_urls)),
        id="venue_1_required_info",
        desc="Venue 1 has required information (name and source URL)",
        parent=node,
        critical=True
    )

    # Identification
    ident = evaluator.add_parallel(
        id="venue_1_identification",
        desc="Correctly identify the theater name and city",
        parent=node,
        critical=True
    )

    name_leaf = evaluator.add_leaf(
        id="venue_1_name",
        desc="Provide the correct theater name",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater's official name is '{v.name}'.",
        node=name_leaf,
        sources=v.source_urls,
        additional_instruction="Verify the venue's name on the cited source."
    )

    city_leaf = evaluator.add_leaf(
        id="venue_1_city",
        desc="Provide the correct city location",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater is located in the city of {v.city}.",
        node=city_leaf,
        sources=v.source_urls,
        additional_instruction="Verify city location on the cited source."
    )

    state_leaf = evaluator.add_leaf(
        id="venue_1_state",
        desc="Provide the correct state location",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater is located in the state of {v.state}.",
        node=state_leaf,
        sources=v.source_urls,
        additional_instruction="Verify state location on the cited source."
    )

    # Verification
    ver = evaluator.add_parallel(
        id="venue_1_verification",
        desc="Verify the theater meets the oldest continuously operating criterion",
        parent=node,
        critical=True
    )

    # Founding year supported
    founding_leaf = evaluator.add_leaf(
        id="venue_1_founding_year",
        desc="Provide the founding year between 1800 and 1815",
        parent=ver,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater was founded (or opened) in {v.year}.",
        node=founding_leaf,
        sources=v.source_urls,
        additional_instruction="Verify the founding/opening year on the cited source."
    )

    # Founding year in required range (custom check)
    year_int = parse_first_year(v.year)
    range_node = evaluator.add_custom_node(
        result=bool(year_int is not None and 1800 <= year_int <= 1815),
        id="venue_1_founding_year_in_range",
        desc=f"Founding year {year_int if year_int else 'unknown'} is within 1800–1815",
        parent=ver,
        critical=True
    )

    cont_leaf = evaluator.add_leaf(
        id="venue_1_continuous_operation",
        desc="Confirm continuous operation status",
        parent=ver,
        critical=True
    )
    await evaluator.verify(
        claim="This theater is described as America's oldest continuously operating theater (i.e., continuous operation).",
        node=cont_leaf,
        sources=v.source_urls,
        additional_instruction="Verify explicit phrasing indicating continuous operation and 'oldest continuously operating theater'."
    )

    source_leaf = evaluator.add_leaf(
        id="venue_1_source_url",
        desc="Provide a reference URL supporting the oldest theater claim",
        parent=ver,
        critical=True
    )
    await evaluator.verify(
        claim="The cited source supports the claim that this is America's oldest continuously operating theater.",
        node=source_leaf,
        sources=v.source_urls,
        additional_instruction="Confirm that the cited page explicitly supports 'oldest continuously operating theater' for the named venue."
    )


async def verify_venue_2(evaluator: Evaluator, root_node, v: Optional[Venue]) -> None:
    """
    Venue 2: Broadway's oldest continuously operating legitimate theater.
    Requirements: NYC location; opening year 1900–1910; Broadway classification (>=500 seats); continuous operation claim.
    """
    node = evaluator.add_sequential(
        id="venue_2_oldest_broadway",
        desc="Identify Broadway's oldest continuously operating legitimate theater",
        parent=root_node,
        critical=False
    )

    req_node = evaluator.add_custom_node(
        result=bool(v and v.name and has_valid_sources(v.source_urls)),
        id="venue_2_required_info",
        desc="Venue 2 has required information (name and source URL)",
        parent=node,
        critical=True
    )

    ident = evaluator.add_parallel(
        id="venue_2_identification",
        desc="Correctly identify the theater name",
        parent=node,
        critical=True
    )

    name_leaf = evaluator.add_leaf(
        id="venue_2_name",
        desc="Provide the correct Broadway theater name",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater's official name is '{v.name}'.",
        node=name_leaf,
        sources=v.source_urls,
        additional_instruction="Verify the venue's name on the cited source."
    )

    nyc_leaf = evaluator.add_leaf(
        id="venue_2_location_nyc",
        desc="Confirm the theater is located in New York City",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim="The theater is located in New York City (NYC).",
        node=nyc_leaf,
        sources=v.source_urls,
        additional_instruction="Verify that the location indicates New York City."
    )

    ver = evaluator.add_parallel(
        id="venue_2_verification",
        desc="Verify the theater meets Broadway's oldest operating criterion",
        parent=node,
        critical=True
    )

    opening_leaf = evaluator.add_leaf(
        id="venue_2_opening_year",
        desc="Provide the opening year in the early 1900s (1900-1910)",
        parent=ver,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater opened in {v.year}.",
        node=opening_leaf,
        sources=v.source_urls,
        additional_instruction="Verify opening year on the cited source."
    )

    year_int = parse_first_year(v.year)
    range_node = evaluator.add_custom_node(
        result=bool(year_int is not None and 1900 <= year_int <= 1910),
        id="venue_2_opening_year_in_range",
        desc=f"Opening year {year_int if year_int else 'unknown'} is within 1900–1910",
        parent=ver,
        critical=True
    )

    broadway_leaf = evaluator.add_leaf(
        id="venue_2_broadway_classification",
        desc="Confirm the theater meets Broadway classification (at least 500 seats)",
        parent=ver,
        critical=True
    )
    await evaluator.verify(
        claim="The theater meets Broadway classification by having a seating capacity of at least 500 seats.",
        node=broadway_leaf,
        sources=v.source_urls,
        additional_instruction="Verify that the seat count is ≥ 500 on the cited source (or that it is explicitly classified as a Broadway theater per seat count standards)."
    )

    cont_leaf = evaluator.add_leaf(
        id="venue_2_continuous_operation",
        desc="Confirm continuous operation as a legitimate theater",
        parent=ver,
        critical=True
    )
    await evaluator.verify(
        claim="This theater has been continuously operating as a legitimate Broadway theater.",
        node=cont_leaf,
        sources=v.source_urls,
        additional_instruction="Look for explicit phrasing indicating continuous operation as a legitimate theater."
    )

    source_leaf = evaluator.add_leaf(
        id="venue_2_source_url",
        desc="Provide a reference URL supporting the oldest Broadway theater claim",
        parent=ver,
        critical=True
    )
    await evaluator.verify(
        claim="The cited source supports the claim that this is Broadway's oldest continuously operating legitimate theater.",
        node=source_leaf,
        sources=v.source_urls,
        additional_instruction="Confirm that the source explicitly supports 'oldest continuously operating legitimate Broadway theater' for the named venue."
    )


async def verify_venue_3(evaluator: Evaluator, root_node, v: Optional[Venue]) -> None:
    """
    Venue 3: Largest performing arts center in the U.S. outside NYC.
    Requirements: total seating capacity > 9,000; multiple theater venues; outside NYC; supported by sources.
    """
    node = evaluator.add_sequential(
        id="venue_3_largest_pac_outside_nyc",
        desc="Identify the largest performing arts center in the United States outside of New York City",
        parent=root_node,
        critical=False
    )

    req_node = evaluator.add_custom_node(
        result=bool(v and v.name and has_valid_sources(v.source_urls)),
        id="venue_3_required_info",
        desc="Venue 3 has required information (name and source URL)",
        parent=node,
        critical=True
    )

    ident = evaluator.add_parallel(
        id="venue_3_identification",
        desc="Correctly identify the performing arts center name and location",
        parent=node,
        critical=True
    )

    name_leaf = evaluator.add_leaf(
        id="venue_3_name",
        desc="Provide the correct performing arts center name",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"The performing arts center's name is '{v.name}'.",
        node=name_leaf,
        sources=v.source_urls,
        additional_instruction="Verify the PAC's name on the cited source."
    )

    city_leaf = evaluator.add_leaf(
        id="venue_3_city",
        desc="Provide the correct city location",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"The performing arts center is located in {v.city}.",
        node=city_leaf,
        sources=v.source_urls,
        additional_instruction="Verify the city on the cited source."
    )

    state_leaf = evaluator.add_leaf(
        id="venue_3_state",
        desc="Provide the correct state location",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"The performing arts center is located in {v.state}.",
        node=state_leaf,
        sources=v.source_urls,
        additional_instruction="Verify the state on the cited source."
    )

    ver = evaluator.add_parallel(
        id="venue_3_verification",
        desc="Verify the center meets the largest performing arts center criterion",
        parent=node,
        critical=True
    )

    capacity_leaf = evaluator.add_leaf(
        id="venue_3_total_capacity",
        desc="Provide total seating capacity exceeding 9,000 seats",
        parent=ver,
        critical=True
    )
    await evaluator.verify(
        claim=f"The total seating capacity across the center's venues is '{v.total_capacity or v.capacity_info}'.",
        node=capacity_leaf,
        sources=v.source_urls,
        additional_instruction="Verify the total capacity figure as presented on the cited source."
    )

    # Numeric check > 9000
    cap_val = parse_max_int(v.total_capacity or v.capacity_info)
    cap_check = evaluator.add_custom_node(
        result=bool(cap_val is not None and cap_val > 9000),
        id="venue_3_capacity_over_9000",
        desc=f"Total seating capacity {cap_val if cap_val is not None else 'unknown'} exceeds 9,000",
        parent=ver,
        critical=True
    )

    multi_leaf = evaluator.add_leaf(
        id="venue_3_multiple_theaters",
        desc="Confirm the center consists of multiple theater venues",
        parent=ver,
        critical=True
    )
    await evaluator.verify(
        claim="This performing arts center comprises multiple venues/theaters.",
        node=multi_leaf,
        sources=v.source_urls,
        additional_instruction="Verify wording indicating multiple venues/theaters within the center."
    )

    outside_leaf = evaluator.add_leaf(
        id="venue_3_outside_nyc",
        desc="Confirm the center is located outside of New York City",
        parent=ver,
        critical=True
    )
    await evaluator.verify(
        claim="The performing arts center is outside New York City.",
        node=outside_leaf,
        sources=v.source_urls,
        additional_instruction="Verify the city/location and confirm it is not NYC."
    )

    source_leaf = evaluator.add_leaf(
        id="venue_3_source_url",
        desc="Provide a reference URL supporting the largest PAC outside NYC claim",
        parent=ver,
        critical=True
    )
    await evaluator.verify(
        claim="The cited source supports the claim that this is the largest performing arts center in the United States outside New York City.",
        node=source_leaf,
        sources=v.source_urls,
        additional_instruction="Confirm the cited page supports 'largest performing arts center outside NYC' (by capacity) for the named venue."
    )


async def verify_venue_4(evaluator: Evaluator, root_node, v: Optional[Venue]) -> None:
    """
    Venue 4: Historic 1920s movie palace theater in a Western U.S. state.
    Requirements: opening year 1920–1929; movie palace architectural style; Western U.S. state (west of Mississippi); source URL.
    Non-critical extras: original capacity, still operating/restored.
    """
    node = evaluator.add_sequential(
        id="venue_4_1920s_western_theater",
        desc="Identify a historic movie palace theater from the 1920s era located in a Western U.S. state",
        parent=root_node,
        critical=False
    )

    req_node = evaluator.add_custom_node(
        result=bool(v and v.name and has_valid_sources(v.source_urls)),
        id="venue_4_required_info",
        desc="Venue 4 has required information (name and source URL)",
        parent=node,
        critical=True
    )

    ident = evaluator.add_parallel(
        id="venue_4_identification",
        desc="Correctly identify the theater name and location",
        parent=node,
        critical=True
    )

    name_leaf = evaluator.add_leaf(
        id="venue_4_name",
        desc="Provide the correct theater name",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater's official name is '{v.name}'.",
        node=name_leaf,
        sources=v.source_urls,
        additional_instruction="Verify the theater's name on the cited source."
    )

    city_leaf = evaluator.add_leaf(
        id="venue_4_city",
        desc="Provide the correct city location",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater is located in {v.city}.",
        node=city_leaf,
        sources=v.source_urls,
        additional_instruction="Verify the city on the cited source."
    )

    state_leaf = evaluator.add_leaf(
        id="venue_4_state",
        desc="Provide the correct Western state location (west of the Mississippi River)",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater is located in the state of {v.state}.",
        node=state_leaf,
        sources=v.source_urls,
        additional_instruction="Verify the state on the cited source."
    )

    # Supplemental check: western (west of Mississippi) – general non-web factual check
    western_leaf = evaluator.add_leaf(
        id="venue_4_west_of_mississippi",
        desc="Confirm the state is west of the Mississippi River",
        parent=ident,
        critical=True
    )
    await evaluator.verify(
        claim=f"The state {v.state} is west of the Mississippi River.",
        node=western_leaf,
        additional_instruction="Use general U.S. geography knowledge to confirm whether the state lies west of the Mississippi River."
    )

    # Verification group – mixed criticality; set parent non-critical to allow partial credit for non-essential details
    ver = evaluator.add_parallel(
        id="venue_4_verification",
        desc="Verify the theater meets the 1920s movie palace criterion",
        parent=node,
        critical=False
    )

    opening_leaf = evaluator.add_leaf(
        id="venue_4_opening_year",
        desc="Provide opening year in the 1920s (1920-1929)",
        parent=ver,
        critical=True
    )
    await evaluator.verify(
        claim=f"The theater opened in {v.year}.",
        node=opening_leaf,
        sources=v.source_urls,
        additional_instruction="Verify opening year on the cited source."
    )

    year_int = parse_first_year(v.year)
    range_node = evaluator.add_custom_node(
        result=bool(year_int is not None and 1920 <= year_int <= 1929),
        id="venue_4_opening_year_in_range",
        desc=f"Opening year {year_int if year_int else 'unknown'} is within 1920–1929",
        parent=ver,
        critical=True
    )

    style_leaf = evaluator.add_leaf(
        id="venue_4_movie_palace_architecture",
        desc="Confirm the theater features movie palace architectural style",
        parent=ver,
        critical=True
    )
    await evaluator.verify(
        claim="This theater features a 'movie palace' architectural style (or is explicitly described as a movie palace).",
        node=style_leaf,
        sources=v.source_urls,
        additional_instruction="Verify explicit description of 'movie palace' style on the cited source."
    )

    orig_cap_leaf = evaluator.add_leaf(
        id="venue_4_original_capacity",
        desc="Provide original seating capacity from opening era",
        parent=ver,
        critical=False
    )
    await evaluator.verify(
        claim=f"The theater's original (opening-era) seating capacity was '{v.capacity_info or v.seat_count}'.",
        node=orig_cap_leaf,
        sources=v.source_urls,
        additional_instruction="Verify any original/opening-era capacity figure shown on the source."
    )

    operating_leaf = evaluator.add_leaf(
        id="venue_4_still_operating",
        desc="Confirm the theater is still operating or has been restored",
        parent=ver,
        critical=False
    )
    await evaluator.verify(
        claim="The theater is still operating today or has been restored for continued use.",
        node=operating_leaf,
        sources=v.source_urls,
        additional_instruction="Look for wording indicating ongoing operation or restoration."
    )

    source_leaf = evaluator.add_leaf(
        id="venue_4_source_url",
        desc="Provide a reference URL supporting the 1920s theater details",
        parent=ver,
        critical=True
    )
    await evaluator.verify(
        claim="The cited source supports key 1920s-era theater details including opening year and movie palace characteristics.",
        node=source_leaf,
        sources=v.source_urls,
        additional_instruction="Confirm that the source supports 1920s-era details and the 'movie palace' characterization for the named venue."
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
    Evaluate an answer for the performing arts venues task using the Mind2Web2 framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates four venues independently
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

    # IMPORTANT: Root must be non-critical to allow mixed criticality in children
    root.critical = False

    # Extract structured venue info from the answer
    extracted: VenuesExtraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Build and verify subtrees for each venue
    # Venue 1
    await verify_venue_1(evaluator, root, extracted.venue1 or Venue())

    # Venue 2
    await verify_venue_2(evaluator, root, extracted.venue2 or Venue())

    # Venue 3
    await verify_venue_3(evaluator, root, extracted.venue3 or Venue())

    # Venue 4
    await verify_venue_4(evaluator, root, extracted.venue4 or Venue())

    # Return structured evaluation summary
    return evaluator.get_summary()