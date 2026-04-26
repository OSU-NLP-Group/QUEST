import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ohio_broadway_venues_2026"
TASK_DESCRIPTION = (
    "A touring Broadway production is planning its 2026 route through Ohio and needs to identify suitable performance "
    "venues in the state. Find four different performing arts venues or theaters, each located in a different Ohio city, "
    "that could host this touring Broadway show. Each venue must meet professional touring requirements including: "
    "(1) seating capacity between 1,000 and 2,000 seats (the typical scale for Broadway touring productions), "
    "(2) being a suitable venue type (performing arts center, theater, or concert hall designed for theatrical productions), "
    "(3) meeting ADA accessibility standards with wheelchair-accessible seating for at least 1% of capacity, "
    "(4) having adequate stage dimensions capable of hosting touring theatrical productions, and "
    "(5) being located in Ohio. For each venue, provide its name, city, capacity, and a reference URL that confirms these specifications."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    capacity: Optional[str] = None
    venue_type: Optional[str] = None
    ada_info: Optional[str] = None
    stage_info: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to four Ohio performing arts venues or theaters from the answer. The goal is to identify four distinct venues,
    each located in a different Ohio city, that are suitable for hosting touring Broadway productions.

    For each venue, extract the following fields exactly as present in the answer:
    - name: The venue's name (e.g., "Ohio Theatre")
    - city: The Ohio city where the venue is located (e.g., "Columbus"). If the answer provides "Columbus, OH" or "Columbus, Ohio", extract just the city name if possible.
    - capacity: The stated seating capacity (as text; keep formatting such as commas or ranges if present)
    - venue_type: The venue type as described (e.g., "performing arts center", "theater", "concert hall")
    - ada_info: Any text in the answer describing ADA accessibility or wheelchair seating availability
    - stage_info: Any text in the answer describing stage dimensions, stage specs, or technical specifications
    - reference_urls: A list of URL(s) explicitly mentioned in the answer that relate to this venue. Extract actual URLs only. If none are given, return an empty list.

    Rules:
    - Return a JSON object with a single field "venues", which is an array of venue objects with the above fields.
    - Include at most the first four venues that appear in the answer.
    - If a field is missing for a venue, set it to null (or empty list for reference_urls).
    - Only extract URLs that are explicitly present in the answer text (plain URL or markdown link); do not invent URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_city(city: Optional[str]) -> str:
    if not city:
        return ""
    s = city.strip().lower()
    # Remove trailing state markers like ", oh", ", ohio", "oh", "ohio"
    s = re.sub(r",?\s*oh(io)?\b", "", s)
    # Remove country if present
    s = re.sub(r",?\s*usa|united states", "", s)
    # Remove "city of " prefix
    s = re.sub(r"^city of\s+", "", s)
    # Collapse spaces and punctuation at ends
    s = re.sub(r"\s+", " ", s).strip(" ,.")
    return s


def _cities_different(curr_city: Optional[str], prev_cities: List[Optional[str]]) -> bool:
    a = _normalize_city(curr_city)
    if not a:
        return False
    for pc in prev_cities:
        if a == _normalize_city(pc):
            return False
    return True


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for u in urls:
        if not u:
            continue
        u2 = u.strip()
        if not u2:
            continue
        if u2 not in seen:
            seen.add(u2)
            deduped.append(u2)
    return deduped


# --------------------------------------------------------------------------- #
# Verification for a single venue                                             #
# --------------------------------------------------------------------------- #
async def verify_one_venue(
    evaluator: Evaluator,
    venue_parent_node,
    venue: VenueItem,
    index: int,
    previous_cities: List[Optional[str]],
) -> None:
    """
    Build and verify all leaf checks for a single venue according to the rubric.
    """
    # Ensure URLs are clean/deduplicated
    urls = _dedup_urls(venue.reference_urls or [])

    # 1) Reference URL leaf - verify first to gate other checks
    ref_node = evaluator.add_leaf(
        id=f"Venue_{index}_Reference",
        desc="Valid reference URL provided for the venue",
        parent=venue_parent_node,
        critical=True,
    )
    ref_claim = (
        f"At least one of the provided URLs is an official or authoritative webpage about the venue "
        f"'{venue.name or ''}' in '{venue.city or ''}', Ohio, and provides venue information such as seating capacity, "
        f"ADA accessibility, or stage/technical specifications."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_node,
        sources=urls,  # If empty, verification falls back to simple and likely fails (no URLs)
        additional_instruction=(
            "Consider 'official or authoritative' to include: the venue's own website, the operating performing arts center, "
            "city/government operator pages, or reputable venue directories/technical specification documents. "
            "The page should be clearly about the same venue and include venue details."
        ),
    )

    # 2) Capacity between 1,000 and 2,000 seats
    cap_node = evaluator.add_leaf(
        id=f"Venue_{index}_Capacity",
        desc="Venue has capacity between 1,000 and 2,000 seats as required for Broadway touring productions",
        parent=venue_parent_node,
        critical=True,
    )
    cap_claim = (
        f"The seating capacity of the venue '{venue.name or ''}' is between 1,000 and 2,000 seats (inclusive). "
        f"If a single specific capacity is provided on the page, check that it falls within 1,000–2,000. "
        f"If a capacity range is provided, ensure that the typical seated capacity for theatrical use is within this band."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_node,
        sources=urls,
        additional_instruction=(
            "Look for terms like 'seating capacity', 'seats', or 'capacity'. Consider minor formatting differences (e.g., 1,700 vs 1700). "
            "If multiple capacities exist for different configurations, ensure a standard theatrical configuration is within 1,000–2,000."
        ),
        extra_prerequisites=[ref_node],
    )

    # 3) Venue type suitability
    type_node = evaluator.add_leaf(
        id=f"Venue_{index}_Type",
        desc="Venue is a performing arts center, theater, or concert hall suitable for theatrical productions",
        parent=venue_parent_node,
        critical=True,
    )
    type_claim = (
        f"The venue '{venue.name or ''}' is a performing arts center, theater, or concert hall designed and suitable for theatrical productions."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_node,
        sources=urls,
        additional_instruction=(
            "Accept venue types such as 'theater', 'theatre', 'performing arts center', or 'concert hall' that explicitly host theatrical productions. "
            "The webpage should indicate suitability for stage/theatrical events, not just sports or other unrelated uses."
        ),
        extra_prerequisites=[ref_node],
    )

    # 4) ADA compliance (wheelchair-accessible seating >= 1% capacity, or at least 10 accessible seats for capacities >= 1000)
    ada_node = evaluator.add_leaf(
        id=f"Venue_{index}_ADA_Compliance",
        desc="Venue provides wheelchair-accessible seating meeting ADA standards (minimum 1% of capacity or at least 10 accessible seats for 1,000+ capacity venues)",
        parent=venue_parent_node,
        critical=True,
    )
    ada_claim = (
        f"The venue '{venue.name or ''}' provides wheelchair-accessible seating that meets ADA standards: "
        f"for a 1,000+ capacity venue, at least 10 accessible seats or at least 1% of total capacity."
    )
    await evaluator.verify(
        claim=ada_claim,
        node=ada_node,
        sources=urls,
        additional_instruction=(
            "Look for an explicit accessibility or ADA policy page indicating wheelchair-accessible seating and its quantity. "
            "Accept if the page clearly indicates compliance with ADA and specifies accessible seating capacity that meets or exceeds the minimums "
            "(≥10 accessible seats for 1,000+ capacity or ≥1% of total). If no quantity is stated, and only a generic 'ADA accessible' note exists without counts, "
            "treat as not sufficiently supported for this quantitative requirement."
        ),
        extra_prerequisites=[ref_node],
    )

    # 5) Stage dimensions adequate
    stage_node = evaluator.add_leaf(
        id=f"Venue_{index}_Stage_Dimensions",
        desc="Venue has adequate stage dimensions capable of hosting touring productions (minimum stage dimensions suitable for professional theater productions)",
        parent=venue_parent_node,
        critical=True,
    )
    stage_claim = (
        f"The venue '{venue.name or ''}' has stage dimensions (width/depth/proscenium height and related specs) that are adequate "
        f"to host professional touring theatrical productions."
    )
    await evaluator.verify(
        claim=stage_claim,
        node=stage_node,
        sources=urls,
        additional_instruction=(
            "Look for technical specifications or stage dimensions. Adequate typically means a mainstage roughly on the order of ~40' width and ~30' depth or larger, "
            "with sufficient wing/fly space; also accept explicit statements such as 'suitable for national touring productions' if accompanied by credible technical specs."
        ),
        extra_prerequisites=[ref_node],
    )

    # 6) Location in Ohio
    loc_node = evaluator.add_leaf(
        id=f"Venue_{index}_Location",
        desc="Venue is located in an Ohio city",
        parent=venue_parent_node,
        critical=True,
    )
    city_txt = venue.city or ""
    loc_claim = (
        f"The venue '{venue.name or ''}' is located in {city_txt}, Ohio."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=urls,
        additional_instruction=(
            "Confirm the venue's address lists a city in Ohio (e.g., shows 'OH' or 'Ohio'). A clear address line containing the city and state is sufficient."
        ),
        extra_prerequisites=[ref_node],
    )

    # 7) Different city check (for venues 2-4)
    if index >= 2:
        diff_id = f"Venue_{index}_Different_City"
        if index == 2:
            desc = "Venue is located in a different Ohio city than Venue 1"
        elif index == 3:
            desc = "Venue is located in a different Ohio city than Venues 1 and 2"
        else:
            desc = "Venue is located in a different Ohio city than Venues 1, 2, and 3"

        # Custom binary node based on extracted city names (no URL needed)
        evaluator.add_custom_node(
            result=_cities_different(venue.city, previous_cities),
            id=diff_id,
            desc=desc,
            parent=venue_parent_node,
            critical=True,
        )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for identifying four suitable Ohio venues for a touring Broadway production.
    """
    # Initialize Evaluator with a parallel root as per rubric
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

    # Extract structured venues info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Limit to first 4 venues (pad with empty if fewer)
    venues: List[VenueItem] = list(extracted.venues[:4])
    while len(venues) < 4:
        venues.append(VenueItem())

    # Add root node per rubric
    root_node = evaluator.add_parallel(
        id="Root_Ohio_Broadway_Venues",
        desc="Evaluation of whether four suitable Broadway touring venues in different Ohio cities have been identified with proper specifications",
        parent=root,
        critical=False,
    )

    # Record some custom info for debugging
    evaluator.add_custom_info(
        {
            "num_extracted": len(extracted.venues),
            "used_venues_count": 4,
            "extracted_cities": [v.city for v in venues],
            "extracted_names": [v.name for v in venues],
        },
        info_type="extraction_summary",
    )

    # Build venue nodes and verify
    previous_cities: List[Optional[str]] = []

    # Venue 1
    venue1_node = evaluator.add_parallel(
        id="Venue_1",
        desc="First Ohio venue meets all Broadway touring requirements",
        parent=root_node,
        critical=False,
    )
    await verify_one_venue(evaluator, venue1_node, venues[0], 1, previous_cities)
    previous_cities.append(venues[0].city)

    # Venue 2
    venue2_node = evaluator.add_parallel(
        id="Venue_2",
        desc="Second Ohio venue in a different city meets all Broadway touring requirements",
        parent=root_node,
        critical=False,
    )
    await verify_one_venue(evaluator, venue2_node, venues[1], 2, previous_cities)
    previous_cities.append(venues[1].city)

    # Venue 3
    venue3_node = evaluator.add_parallel(
        id="Venue_3",
        desc="Third Ohio venue in a different city meets all Broadway touring requirements",
        parent=root_node,
        critical=False,
    )
    await verify_one_venue(evaluator, venue3_node, venues[2], 3, previous_cities)
    previous_cities.append(venues[2].city)

    # Venue 4
    venue4_node = evaluator.add_parallel(
        id="Venue_4",
        desc="Fourth Ohio venue in a different city meets all Broadway touring requirements",
        parent=root_node,
        critical=False,
    )
    await verify_one_venue(evaluator, venue4_node, venues[3], 4, previous_cities)
    previous_cities.append(venues[3].city)

    # Return evaluation summary
    return evaluator.get_summary()