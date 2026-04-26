import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_arena_tour_venues"
TASK_DESCRIPTION = (
    "A concert tour manager is planning a multi-city arena tour across the United States and needs to identify "
    "suitable indoor arena venues in four major cities: Philadelphia (Pennsylvania), New York City (New York), "
    "Nashville (Tennessee), and Los Angeles (California). For each of these four cities, identify the major indoor "
    "arena venue that meets ALL of the following requirements: (1) The venue must be located in the specified city "
    "and state, (2) The venue must have a minimum seating capacity of 18,000 for concerts, (3) The venue must be an "
    "indoor arena (not an outdoor stadium), (4) The venue must support end-stage concert configuration, (5) The venue "
    "must have loading dock access for equipment transport via trucks, (6) The venue must provide dressing room "
    "facilities for performers, (7) The venue must comply with ADA requirements for wheelchair accessible seating. "
    "For each venue, provide: the official venue name, the city and state location, the concert seating capacity, and "
    "reference URLs that verify the venue information and capacity."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    official_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    concert_capacity: Optional[str] = None  # keep as string to be robust to formats like "20,000+" or ranges
    indoor_arena: Optional[str] = None      # should reflect whether the answer claims indoor arena
    end_stage_supported: Optional[str] = None
    loading_dock_access: Optional[str] = None
    dressing_rooms: Optional[str] = None
    ada_wheelchair: Optional[str] = None
    sources: List[str] = Field(default_factory=list)  # URLs explicitly mentioned in the answer


class VenuesExtraction(BaseModel):
    philadelphia_pa: Optional[VenueInfo] = None
    new_york_city_manhattan_ny: Optional[VenueInfo] = None
    nashville_tn: Optional[VenueInfo] = None
    los_angeles_ca: Optional[VenueInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract structured information for four venues as presented in the answer text. You must extract exactly and only what the answer states; do not invent data.

    For each of the following cities, extract an object with these fields:
      - official_name: The official venue name as written in the answer.
      - city: The city name as written in the answer (e.g., "Philadelphia", "New York City", "Nashville", "Los Angeles"; include "Manhattan" if explicitly noted for NYC).
      - state: The state abbreviation or full name (e.g., "PA", "Pennsylvania", "NY", "New York", etc.) exactly as written in the answer.
      - concert_capacity: The concert seating capacity value as written in the answer (e.g., "20,000", "20,000+", "approx. 19,500"). If not provided, return null.
      - indoor_arena: A short phrase or "yes"/"no" as written in the answer indicating indoor arena (not outdoor stadium). If not provided, return null.
      - end_stage_supported: A short phrase or "yes"/"no" as written in the answer indicating end-stage concert configuration support. If not provided, return null.
      - loading_dock_access: A short phrase or "yes"/"no" as written in the answer indicating loading dock access for trucks/equipment. If not provided, return null.
      - dressing_rooms: A short phrase or "yes"/"no" as written in the answer indicating dressing room facilities. If not provided, return null.
      - ada_wheelchair: A short phrase or "yes"/"no" as written in the answer indicating ADA-compliant wheelchair accessible seating. If not provided, return null.
      - sources: A list of the reference URLs explicitly mentioned in the answer for that venue. Only include actual URLs; return an empty list if none.

    Return a JSON with four top-level keys and VenueInfo objects as values:
      - philadelphia_pa
      - new_york_city_manhattan_ny
      - nashville_tn
      - los_angeles_ca

    If a venue is missing in the answer, set the corresponding object to null.
    """


# --------------------------------------------------------------------------- #
# Helper to sanitize and build claims                                         #
# --------------------------------------------------------------------------- #
def _safe(val: Optional[str]) -> str:
    return val.strip() if isinstance(val, str) else ""


def _city_state_claim(city: str, state: str, is_nyc_manhattan: bool = False) -> str:
    if is_nyc_manhattan:
        return f"The venue is located in Manhattan in New York City, {state}."
    return f"The venue is located in {city}, {state}."


# --------------------------------------------------------------------------- #
# Verification logic per city                                                 #
# --------------------------------------------------------------------------- #
async def verify_city(
    evaluator: Evaluator,
    parent_node,
    info: Optional[VenueInfo],
    city_code: str,
    city_node_desc: str,
    is_nyc_manhattan: bool = False,
) -> None:
    """
    Build verification nodes for one city and run checks using the provided sources.
    Each check corresponds to a rubric leaf and is a binary verification with sources.
    """
    # Create city-level node (parallel aggregation, non-critical to allow partial credit)
    city_node = evaluator.add_parallel(
        id=f"{city_code}_Venue",
        desc=city_node_desc,
        parent=parent_node,
        critical=False
    )

    # Prepare extracted fields
    name = _safe(info.official_name if info else None)
    city = _safe(info.city if info else None)
    state = _safe(info.state if info else None)
    cap = _safe(info.concert_capacity if info else None)
    indoor = _safe(info.indoor_arena if info else None)
    end_stage = _safe(info.end_stage_supported if info else None)
    dock = _safe(info.loading_dock_access if info else None)
    dressing = _safe(info.dressing_rooms if info else None)
    ada = _safe(info.ada_wheelchair if info else None)
    sources = info.sources if info and info.sources else []

    # 1) Official name correctly identified
    leaf_name = evaluator.add_leaf(
        id=f"{city_code}_Official_Name_Correct",
        desc="Provides the official venue name (correctly identified)",
        parent=city_node,
        critical=True
    )
    claim_name = f"The official name of this venue is '{name}'."
    await evaluator.verify(
        claim=claim_name,
        node=leaf_name,
        sources=sources,
        additional_instruction="Verify the venue's official name on the provided URLs. Allow minor variations like inclusion of city or sponsor naming, but ensure it refers to the same venue."
    )

    # 2) Location correct (city/state requirement)
    leaf_loc = evaluator.add_leaf(
        id=f"{city_code}_Location_Correct",
        desc="Venue is located in the specified city and state (city/state match requirement)",
        parent=city_node,
        critical=True
    )
    claim_loc = _city_state_claim(city if city else "the specified city", state if state else "the specified state", is_nyc_manhattan=is_nyc_manhattan)
    await evaluator.verify(
        claim=claim_loc,
        node=leaf_loc,
        sources=sources,
        additional_instruction=(
            "Confirm the venue's location on the provided URLs. "
            "For New York City, accept references to 'New York, NY', 'Manhattan, NY', or 'New York City (Manhattan)'. "
            "Minor formatting variations are acceptable."
        )
    )

    # 3) Concert capacity >= 18,000 (and a value is provided)
    leaf_cap = evaluator.add_leaf(
        id=f"{city_code}_Concert_Capacity_GTE_18000",
        desc="Provides a concert seating capacity value and it is >= 18,000",
        parent=city_node,
        critical=True
    )
    # Include the extracted value in the claim if present; otherwise the statement will likely fail
    if cap:
        claim_cap = f"The venue has a concert seating capacity of {cap}, and this capacity is at least 18,000."
    else:
        claim_cap = "The venue has a concert seating capacity that is at least 18,000."
    await evaluator.verify(
        claim=claim_cap,
        node=leaf_cap,
        sources=sources,
        additional_instruction=(
            "Check the specific concert capacity or maximum seating for concerts. "
            "Numbers presented as approximations or ranges are acceptable if clearly ≥ 18,000. "
            "If the answer did not provide any capacity value, treat this as incorrect."
        )
    )

    # 4) Indoor arena (not outdoor stadium)
    leaf_indoor = evaluator.add_leaf(
        id=f"{city_code}_Indoor_Arena",
        desc="Venue is an indoor arena (not an outdoor stadium)",
        parent=city_node,
        critical=True
    )
    claim_indoor = "The venue is an indoor arena (an enclosed, indoor venue), not an outdoor stadium."
    await evaluator.verify(
        claim=claim_indoor,
        node=leaf_indoor,
        sources=sources,
        additional_instruction="Verify that the venue is described as an indoor/multipurpose arena (enclosed building), not an outdoor stadium."
    )

    # 5) End-stage concert configuration supported
    leaf_end = evaluator.add_leaf(
        id=f"{city_code}_End_Stage_Supported",
        desc="Venue supports end-stage concert configuration",
        parent=city_node,
        critical=True
    )
    claim_end = "The venue supports end-stage concert configuration (stage placed at one end of the bowl)."
    await evaluator.verify(
        claim=claim_end,
        node=leaf_end,
        sources=sources,
        additional_instruction=(
            "Look for references to end-stage setups, end-stage seating charts, or production specs indicating end-stage configurations. "
            "Terms like 'end stage', 'stage at one end', or seating charts that show end-stage are acceptable."
        )
    )

    # 6) Loading dock access suitable for trucks/equipment
    leaf_dock = evaluator.add_leaf(
        id=f"{city_code}_Loading_Dock_Access",
        desc="Venue has loading dock access suitable for trucks/equipment transport",
        parent=city_node,
        critical=True
    )
    claim_dock = "The venue has loading dock access suitable for trucks and equipment transport."
    await evaluator.verify(
        claim=claim_dock,
        node=leaf_dock,
        sources=sources,
        additional_instruction=(
            "Check production guides, facility specs, or venue operations pages that mention loading docks, truck access, or freight entrances."
        )
    )

    # 7) Dressing room facilities for performers
    leaf_dressing = evaluator.add_leaf(
        id=f"{city_code}_Dressing_Rooms",
        desc="Venue provides dressing room facilities for performers",
        parent=city_node,
        critical=True
    )
    claim_dress = "The venue provides dressing room facilities for performers."
    await evaluator.verify(
        claim=claim_dress,
        node=leaf_dressing,
        sources=sources,
        additional_instruction="Look for venue specs, backstage amenities, event production information, or facility descriptions that mention dressing rooms."
    )

    # 8) ADA wheelchair-accessible seating compliance
    leaf_ada = evaluator.add_leaf(
        id=f"{city_code}_ADA_Wheelchair_Seating",
        desc="Venue complies with ADA requirements for wheelchair-accessible seating",
        parent=city_node,
        critical=True
    )
    claim_ada = "The venue complies with ADA requirements for wheelchair-accessible seating."
    await evaluator.verify(
        claim=claim_ada,
        node=leaf_ada,
        sources=sources,
        additional_instruction="Check the venue's accessibility or ADA policy pages for references to wheelchair-accessible seating and ADA compliance."
    )

    # 9) Reference URLs verify claims (capacity verification focus)
    leaf_refs = evaluator.add_leaf(
        id=f"{city_code}_Reference_URLs_Verify_Claims",
        desc="Provides verifiable reference URL(s) that support the venue information and concert capacity (one or more URLs allowed)",
        parent=city_node,
        critical=True
    )
    if cap:
        claim_refs = f"At least one of the provided URLs explicitly supports that the concert seating capacity of {name} is {cap} (or clearly ≥ 18,000)."
    else:
        claim_refs = f"At least one of the provided URLs explicitly supports that the concert seating capacity of {name} is at least 18,000."
    await evaluator.verify(
        claim=claim_refs,
        node=leaf_refs,
        sources=sources,
        additional_instruction=(
            "Confirm that at least one provided URL explicitly states the venue's concert seating capacity or maximum concert seating, "
            "and that it clearly meets or exceeds 18,000."
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
    Evaluate the agent's answer for identifying qualifying indoor arena venues in Philadelphia, NYC (Manhattan), Nashville, and Los Angeles.
    """
    # Initialize evaluator (root is parallel aggregation, non-critical to allow partial credit across cities)
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

    # Extract venue information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Build and run verification for each city
    await verify_city(
        evaluator=evaluator,
        parent_node=root,
        info=extracted.philadelphia_pa,
        city_code="PHL",
        city_node_desc="Philadelphia, Pennsylvania venue meets all requirements and required fields are provided",
        is_nyc_manhattan=False
    )

    await verify_city(
        evaluator=evaluator,
        parent_node=root,
        info=extracted.new_york_city_manhattan_ny,
        city_code="NYC",
        city_node_desc="New York City (Manhattan), New York venue meets all requirements and required fields are provided",
        is_nyc_manhattan=True
    )

    await verify_city(
        evaluator=evaluator,
        parent_node=root,
        info=extracted.nashville_tn,
        city_code="NSH",
        city_node_desc="Nashville, Tennessee venue meets all requirements and required fields are provided",
        is_nyc_manhattan=False
    )

    await verify_city(
        evaluator=evaluator,
        parent_node=root,
        info=extracted.los_angeles_ca,
        city_code="LA",
        city_node_desc="Los Angeles, California venue meets all requirements and required fields are provided",
        is_nyc_manhattan=False
    )

    # Optional: record custom statistics
    total_urls = sum(len(v.sources) for v in [
        extracted.philadelphia_pa or VenueInfo(),
        extracted.new_york_city_manhattan_ny or VenueInfo(),
        extracted.nashville_tn or VenueInfo(),
        extracted.los_angeles_ca or VenueInfo(),
    ])
    evaluator.add_custom_info({"total_reference_urls_provided": total_urls}, info_type="stats", info_name="reference_url_stats")

    # Return the evaluation summary
    return evaluator.get_summary()