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
TASK_ID = "pa_broadway_venue_specs"
TASK_DESCRIPTION = (
    "Identify a theater venue in Pennsylvania that regularly hosts touring Broadway productions. "
    "For this venue, provide the following verified technical specifications: "
    "(1) the total seating capacity, which must meet the minimum 500-seat requirement for Broadway theaters; "
    "(2) the loading dock height in inches; and "
    "(3) the stage depth measurement from the plaster line to the upstage wall in feet. "
    "Include reference URLs that verify each specification."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    # URLs relevant to venue location/identity (e.g., official website, about/contact page)
    location_urls: List[str] = Field(default_factory=list)
    # URLs that specifically support the claim that the venue hosts touring Broadway productions
    broadway_urls: List[str] = Field(default_factory=list)


class SeatingSpec(BaseModel):
    # Capacity text exactly as presented in the answer (e.g., "1,728 seats", "approx. 1800")
    capacity_text: Optional[str] = None
    # URLs that verify seating capacity
    capacity_urls: List[str] = Field(default_factory=list)


class LoadingDockSpec(BaseModel):
    # Loading dock height exactly as presented (e.g., "48 inches", "4'-0\"", "44 in")
    height_in_text: Optional[str] = None
    # URLs that verify loading dock height
    height_urls: List[str] = Field(default_factory=list)


class StageDepthSpec(BaseModel):
    # Stage depth exactly as presented (e.g., "30 feet", "29 ft from plaster line to upstage wall")
    depth_ft_text: Optional[str] = None
    # URLs that verify stage depth
    depth_urls: List[str] = Field(default_factory=list)


class VenueSpecsExtraction(BaseModel):
    venue: Optional[VenueInfo] = None
    seating: Optional[SeatingSpec] = None
    loading_dock: Optional[LoadingDockSpec] = None
    stage_depth: Optional[StageDepthSpec] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_specs() -> str:
    return """
    You must extract a single theater venue in Pennsylvania that the answer claims regularly hosts touring Broadway productions,
    and extract three technical specifications for this specific venue with their verification URLs.

    Extraction requirements:
    1) venue:
       - name: Exact name of the theater venue mentioned.
       - city: The city mentioned alongside the venue (if present in the answer; else null).
       - state: The state string (e.g., "Pennsylvania" or "PA") if present; else null.
       - location_urls: URLs explicitly mentioned that can help verify basic venue details (e.g., official website, about/contact page).
       - broadway_urls: URLs explicitly mentioned that support that the venue regularly hosts touring Broadway productions (e.g., season calendar, "Broadway" series pages, Broadway touring listings).
    2) seating:
       - capacity_text: The seating capacity value exactly as stated in the answer (include units/words like "seats" if present). Do NOT invent or normalize; copy verbatim.
       - capacity_urls: URLs explicitly mentioned that support the seating capacity figure.
    3) loading_dock:
       - height_in_text: Loading dock height exactly as stated in the answer (e.g., "48 inches", "4'-0\"", "44 in"). Copy verbatim; do not convert.
       - height_urls: URLs explicitly mentioned that support the loading dock height.
    4) stage_depth:
       - depth_ft_text: The stage depth from the plaster line to the upstage wall in feet exactly as stated in the answer (e.g., "30 ft", "25 feet from PL to upstage wall"). Copy verbatim.
       - depth_urls: URLs explicitly mentioned that support this stage depth.

    Special rules:
    - Extract only URLs explicitly present in the answer text. If a field has no URLs explicitly cited, return an empty list.
    - If multiple venues are mentioned, select the FIRST venue and extract specs for that venue only.
    - If any required value (like capacity_text, height_in_text, depth_ft_text) is missing in the answer, return null for that field and an empty URL list for its verification URLs.
    - If the state is presented as "PA", treat that as Pennsylvania, but still return the exact text found for 'state'.

    Return a JSON object with the schema provided; do not add extra fields.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def parse_first_number(text: Optional[str]) -> Optional[float]:
    """
    Parse the first numeric value from a textual field. Supports commas and decimals.
    Examples:
        "1,728 seats" -> 1728.0
        "4'-0\"" -> 4.0 (feet; caller must decide unit handling)
        "48 inches" -> 48.0
        "29-30 ft" -> 29.0
    """
    if not text:
        return None
    # First try to capture standard numbers with optional commas/decimals
    m = re.search(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+)", text)
    if not m:
        return None
    num_str = m.group(1).replace(",", "")
    try:
        return float(num_str)
    except Exception:
        return None


def state_is_pennsylvania(state_text: Optional[str]) -> bool:
    if not state_text:
        return False
    s = state_text.strip().lower()
    return s in {"pa", "pennsylvania"}


def merge_sources(*lists: List[str]) -> List[str]:
    """Merge and deduplicate URL lists, keeping non-empty strings only."""
    merged = []
    seen = set()
    for lst in lists:
        for u in lst or []:
            u = (u or "").strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_venue_selection(
    evaluator: Evaluator,
    parent_node,
    ex: VenueSpecsExtraction
) -> None:
    """
    Build and verify the VenueSelection subtree:
      - VenueNamed (existence)
      - VenueInPennsylvania (location claim)
      - VenueHostsTouringBroadway (programming claim)
    All are critical under a critical parent.
    """
    venue_node = evaluator.add_parallel(
        id="VenueSelection",
        desc="Select a qualifying venue.",
        parent=parent_node,
        critical=True
    )

    venue_name = ex.venue.name if ex and ex.venue else None
    venue_city = ex.venue.city if ex and ex.venue else None
    venue_state = ex.venue.state if ex and ex.venue else None
    loc_urls = ex.venue.location_urls if ex and ex.venue else []
    broadway_urls = ex.venue.broadway_urls if ex and ex.venue else []

    # VenueNamed - existence check
    evaluator.add_custom_node(
        result=bool(venue_name and venue_name.strip()),
        id="VenueNamed",
        desc="Provide the name of a specific theater venue (single venue).",
        parent=venue_node,
        critical=True
    )

    # VenueInPennsylvania - verify with available URLs (prefer location URLs, but allow all known URLs as backup)
    venue_in_pa_leaf = evaluator.add_leaf(
        id="VenueInPennsylvania",
        desc="The identified venue is located in Pennsylvania.",
        parent=venue_node,
        critical=True
    )
    pa_sources = merge_sources(loc_urls, broadway_urls)
    pa_claim_city = f"{venue_city}, " if venue_city else ""
    pa_claim_state = "Pennsylvania" if state_is_pennsylvania(venue_state) else "Pennsylvania"
    pa_claim = f"The venue '{venue_name}' is located in {pa_claim_city}{pa_claim_state}."
    await evaluator.verify(
        claim=pa_claim,
        node=venue_in_pa_leaf,
        sources=pa_sources if pa_sources else None,
        additional_instruction="Treat 'PA' as Pennsylvania. If a page shows the venue address with 'PA' or the city in Pennsylvania, count it as Pennsylvania."
    )

    # VenueHostsTouringBroadway - verify programming
    broadway_leaf = evaluator.add_leaf(
        id="VenueHostsTouringBroadway",
        desc="The identified venue regularly hosts touring Broadway productions.",
        parent=venue_node,
        critical=True
    )
    bw_sources = merge_sources(broadway_urls, loc_urls)
    bw_claim = f"The venue '{venue_name}' regularly hosts touring Broadway productions."
    await evaluator.verify(
        claim=bw_claim,
        node=broadway_leaf,
        sources=bw_sources if bw_sources else None,
        additional_instruction=(
            "Look for evidence such as a 'Broadway' or 'National Tour' series in the venue's season, "
            "membership with a local 'Broadway in [City]' program, or listings of touring Broadway shows across seasons. "
            "One-off isolated events do not count as 'regularly'."
        )
    )


async def build_technical_specifications(
    evaluator: Evaluator,
    parent_node,
    ex: VenueSpecsExtraction
) -> None:
    """
    Build and verify the TechnicalSpecifications subtree with three parallel critical categories:
      - SeatingCapacitySpecification
      - LoadingDockSpecification
      - StageDepthSpecification
    Each category includes a value/minimum check (custom node) and a URL-supported verification leaf.
    """
    tech_node = evaluator.add_parallel(
        id="TechnicalSpecifications",
        desc="Provide all required technical specifications for the identified venue, each with verification URLs.",
        parent=parent_node,
        critical=True
    )

    venue_name = ex.venue.name if ex and ex.venue else "the venue"

    # ------------------------- Seating Capacity ------------------------- #
    seating_node = evaluator.add_parallel(
        id="SeatingCapacitySpecification",
        desc="Provide and verify seating capacity.",
        parent=tech_node,
        critical=True
    )
    cap_text = ex.seating.capacity_text if ex and ex.seating else None
    cap_urls = ex.seating.capacity_urls if ex and ex.seating else []

    # combined value/minimum check (existence + >= 500 seats)
    cap_num = parse_first_number(cap_text)
    evaluator.add_custom_node(
        result=bool(cap_num is not None and cap_num >= 500.0),
        id="SeatingCapacityValueAndMinimum",
        desc="Provide total seating capacity (numeric) and it is at least 500 seats.",
        parent=seating_node,
        critical=True
    )

    # URL verification leaf
    seating_url_leaf = evaluator.add_leaf(
        id="SeatingCapacityURL",
        desc="Provide reference URL(s) that verify the seating capacity.",
        parent=seating_node,
        critical=True
    )
    cap_claim = f"The total seating capacity of {venue_name} is {cap_text}."
    await evaluator.verify(
        claim=cap_claim,
        node=seating_url_leaf,
        sources=cap_urls if cap_urls else None,
        additional_instruction=(
            "Verify that the cited page(s) explicitly state the venue's total seating capacity. "
            "Minor rounding or formatting differences (e.g., commas) are acceptable."
        )
    )

    # ------------------------- Loading Dock Height ---------------------- #
    load_node = evaluator.add_parallel(
        id="LoadingDockSpecification",
        desc="Provide and verify loading dock height.",
        parent=tech_node,
        critical=True
    )
    height_text = ex.loading_dock.height_in_text if ex and ex.loading_dock else None
    height_urls = ex.loading_dock.height_urls if ex and ex.loading_dock else []
    height_num = parse_first_number(height_text)

    # combined value + OSHA typical alignment (44–48 inches)
    evaluator.add_custom_node(
        result=bool(height_num is not None and 44.0 <= height_num <= 48.0),
        id="LoadingDockValueAndOSHAAlignment",
        desc="Provide loading dock height in inches and it falls within 44 to 48 inches (inclusive).",
        parent=load_node,
        critical=True
    )

    load_url_leaf = evaluator.add_leaf(
        id="LoadingDockURL",
        desc="Provide reference URL(s) that verify the loading dock height.",
        parent=load_node,
        critical=True
    )
    height_claim = f"The loading dock height of {venue_name} is {height_text}."
    await evaluator.verify(
        claim=height_claim,
        node=load_url_leaf,
        sources=height_urls if height_urls else None,
        additional_instruction=(
            "Confirm the loading dock height from the cited page(s). "
            "If the page uses feet/inches notation like 4'-0\", recognize that as 48 inches. "
            "Allow exact-equivalent unit conversions."
        )
    )

    # ------------------------- Stage Depth (PL to Upstage Wall) --------- #
    stage_node = evaluator.add_parallel(
        id="StageDepthSpecification",
        desc="Provide and verify stage depth from plaster line to upstage wall.",
        parent=tech_node,
        critical=True
    )
    depth_text = ex.stage_depth.depth_ft_text if ex and ex.stage_depth else None
    depth_urls = ex.stage_depth.depth_urls if ex and ex.stage_depth else []
    depth_num = parse_first_number(depth_text)

    evaluator.add_custom_node(
        result=bool(depth_num is not None and depth_num >= 25.0),
        id="StageDepthValueAndMinimum",
        desc="Provide stage depth from plaster line to upstage wall in feet and it is at least 25 feet.",
        parent=stage_node,
        critical=True
    )

    stage_url_leaf = evaluator.add_leaf(
        id="StageDepthURL",
        desc="Provide reference URL(s) that verify the stage depth measurement.",
        parent=stage_node,
        critical=True
    )
    depth_claim = f"The stage depth from the plaster line to the upstage wall at {venue_name} is {depth_text}."
    await evaluator.verify(
        claim=depth_claim,
        node=stage_url_leaf,
        sources=depth_urls if depth_urls else None,
        additional_instruction=(
            "Verify that the cited page(s) explicitly indicate the measurement from the 'plaster line' (or proscenium line) "
            "to the upstage (rear) wall. Accept common synonyms and equivalent wording."
        )
    )

    # Record computed numeric info for debugging
    evaluator.add_custom_info(
        info={
            "venue_name": venue_name,
            "capacity_text": cap_text,
            "capacity_numeric": cap_num,
            "loading_dock_text": height_text,
            "loading_dock_numeric": height_num,
            "stage_depth_text": depth_text,
            "stage_depth_numeric": depth_num,
        },
        info_type="parsed_numbers",
        info_name="numeric_parsing_summary"
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
    Evaluate the answer for the Pennsylvania Broadway venue specs task.
    Builds a critical sequential root task node with two critical phases:
      1) VenueSelection (parallel critical)
      2) TechnicalSpecifications (parallel critical)
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # We'll add a critical sequential task node under this root
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

    # Extract structured info from the answer
    ex = await evaluator.extract(
        prompt=prompt_extract_venue_specs(),
        template_class=VenueSpecsExtraction,
        extraction_name="venue_specs_extraction"
    )

    # Build critical sequential task node (to respect JSON's critical Root + sequential)
    task_node = evaluator.add_sequential(
        id="Root",
        desc="Identify one Pennsylvania theater that regularly hosts touring Broadway productions and provide verified technical specifications with citations.",
        parent=root,
        critical=True
    )

    # Phase 1: Venue selection and qualification
    await build_venue_selection(evaluator, task_node, ex)

    # Phase 2: Technical specifications with verification URLs
    await build_technical_specifications(evaluator, task_node, ex)

    # Return structured evaluation summary
    return evaluator.get_summary()