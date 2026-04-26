import asyncio
import logging
import math
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "chi_concert_venues"
TASK_DESCRIPTION = (
    "A touring music artist is planning concert stops in Chicago and needs to identify two suitable venues "
    "that can accommodate their full band production. Your task is to identify two distinct concert venues in Chicago "
    "that meet ALL of the following requirements: Venue Requirements: (1) First Venue: Must have a seating capacity "
    "between 3,000 and 4,000 seats. (2) Second Venue: Must have a seating capacity between 4,500 and 5,500 seats. "
    "(3) Both venues must be located in Chicago, Illinois. For each venue, provide: A. Venue Identification: Official "
    "venue name and full address in Chicago. B. Capacity Information: Exact seating capacity for concerts and URL "
    "reference documenting this capacity. C. ADA Accessibility: Calculate the minimum required wheelchair-accessible "
    "seats (approximately 1% of the venue's total capacity) and confirm that the venue provides accessible seating or "
    "meets ADA requirements. D. Stage Specifications: Confirm the venue can accommodate a stage of at least 24 feet "
    "wide by 16 feet deep (minimum requirement for a full band) and provide URL reference for stage specifications. "
    "E. Operating Information: Venue's insurance requirements or policy for events, and venue's curfew or time restriction "
    "policies for evening concerts. Provide all information with supporting URL references from official venue websites, "
    "venue directories, or reputable sources."
)

SMALL_RANGE = (3000, 4000)
LARGE_RANGE = (4500, 5500)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    capacity_text: Optional[str] = None
    id_urls: List[str] = Field(default_factory=list)
    capacity_urls: List[str] = Field(default_factory=list)
    accessibility_urls: List[str] = Field(default_factory=list)
    stage_urls: List[str] = Field(default_factory=list)
    insurance_urls: List[str] = Field(default_factory=list)
    curfew_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venue_3000_4000: Optional[VenueInfo] = None
    venue_4500_5500: Optional[VenueInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract exactly two distinct Chicago concert venues from the answer that meet the capacity range requirements:

    1) venue_3000_4000: Must have seating capacity between 3,000 and 4,000 seats.
    2) venue_4500_5500: Must have seating capacity between 4,500 and 5,500 seats.

    For each venue, extract the following fields strictly from the answer:
    - name: Official venue name.
    - address: Full address in Chicago, Illinois (as presented).
    - capacity_text: The seating capacity value/description (e.g., "3,700 seats", "about 4,000").
    - id_urls: URL(s) that show the official venue name and/or address (prefer official venue websites or reputable directories).
    - capacity_urls: URL(s) that explicitly document the seating capacity.
    - accessibility_urls: URL(s) confirming accessible seating or ADA compliance.
    - stage_urls: URL(s) with stage specifications or tech specs that can demonstrate accommodating at least 24' x 16' stage.
    - insurance_urls: URL(s) showing venue event insurance requirements or policy.
    - curfew_urls: URL(s) showing curfew policy or time restrictions for events/evening concerts.

    Rules:
    - Return full URLs with protocol (http:// or https://). If not present in the answer, set the URL list to an empty array.
    - Do not invent or infer any data not explicitly present in the answer. If any field is missing, return null (for text fields) or [] (for URL lists).
    - If the same URL supports multiple fields, include it in all relevant URL lists.
    - Ensure the two venues are distinct.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_capacity_number(capacity_text: Optional[str]) -> Optional[int]:
    """Extract an integer capacity from text like '3,700', 'about 4500', '4,000 seats'."""
    if not capacity_text:
        return None
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d{3,5})", capacity_text)
    if not m:
        return None
    raw = m.group(1)
    try:
        return int(raw.replace(",", ""))
    except Exception:
        return None


def compute_required_accessible_seats(capacity: Optional[int]) -> Optional[int]:
    """Approximate minimum required wheelchair-accessible seats (~1% of capacity, round up)."""
    if capacity is None or capacity <= 0:
        return None
    return math.ceil(capacity * 0.01)


def combine_urls(*url_lists: List[str]) -> List[str]:
    """Combine multiple URL lists and deduplicate."""
    seen = set()
    result = []
    for lst in url_lists:
        for u in lst or []:
            u_norm = (u or "").strip()
            if not u_norm:
                continue
            if u_norm not in seen:
                seen.add(u_norm)
                result.append(u_norm)
    return result


def pick_identification_sources(venue: VenueInfo) -> List[str]:
    """Prefer identification URLs, fall back to other venue URLs if necessary."""
    urls = combine_urls(venue.id_urls)
    if urls:
        return urls
    fallback = combine_urls(venue.capacity_urls, venue.stage_urls, venue.accessibility_urls, venue.insurance_urls, venue.curfew_urls)
    return fallback


def is_chicago_address(address: Optional[str]) -> bool:
    """Simple string check for 'Chicago' and 'IL'/'Illinois' in the address."""
    if not address:
        return False
    a = address.lower()
    return ("chicago" in a) and (" il" in a or "illinois" in a)


async def verify_with_urls_or_fail(
    evaluator: Evaluator,
    claim: str,
    node_id: str,
    node_desc: str,
    parent_node,
    urls: List[str],
    critical: bool = True,
    additional_instruction: str = "None"
) -> None:
    """Create a leaf node and verify against URLs, or fail explicitly if URLs are missing."""
    if urls:
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=node_desc,
            parent=parent_node,
            critical=critical,
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=additional_instruction,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=node_desc + " (Missing source URL)",
            parent=parent_node,
            critical=critical,
        )


# --------------------------------------------------------------------------- #
# Verification for a single venue                                             #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    root_parent,
    venue: VenueInfo,
    prefix: str,
    cap_low: int,
    cap_high: int
) -> None:
    """
    Build and verify the sub-tree for one venue with a given capacity range.
    prefix examples: 'venue_3000_4000' or 'venue_4500_5500'
    """
    # Identification (critical, parallel)
    ident_node = evaluator.add_parallel(
        id=f"{prefix}_identification",
        desc="Venue identification information",
        parent=root_parent,
        critical=True
    )

    # Name (critical leaf)
    ident_sources = pick_identification_sources(venue)
    await verify_with_urls_or_fail(
        evaluator,
        claim=f"The official venue name is '{venue.name or ''}'.",
        node_id=f"{prefix}_name",
        node_desc="Correct official venue name",
        parent_node=ident_node,
        urls=ident_sources,
        critical=True,
        additional_instruction="Confirm that the page explicitly indicates the venue's official name (allow minor variations or punctuation)."
    )

    # Address (critical leaf)
    await verify_with_urls_or_fail(
        evaluator,
        claim=f"The venue's address is '{venue.address or ''}' and it is located in Chicago, Illinois.",
        node_id=f"{prefix}_address",
        node_desc="Full address in Chicago, Illinois",
        parent_node=ident_node,
        urls=ident_sources,
        critical=True,
        additional_instruction="Verify the address, and confirm it is in Chicago, IL (accept 'Chicago, Illinois'). Minor formatting differences are fine."
    )

    # Capacity (critical, sequential)
    cap_node = evaluator.add_sequential(
        id=f"{prefix}_capacity",
        desc="Venue capacity verification",
        parent=root_parent,
        critical=True
    )

    capacity_n = parse_capacity_number(venue.capacity_text)
    in_range = (capacity_n is not None) and (cap_low <= capacity_n <= cap_high)

    evaluator.add_custom_node(
        result=in_range,
        id=f"{prefix}_capacity_value",
        desc=f"Stated capacity falls within {cap_low}-{cap_high} seat range",
        parent=cap_node,
        critical=True
    )

    cap_doc_node = evaluator.add_sequential(
        id=f"{prefix}_capacity_documentation",
        desc="Capacity documentation",
        parent=cap_node,
        critical=True
    )

    # Capacity source (critical leaf)
    cap_claim = (
        f"The venue's seating capacity is approximately {capacity_n} seats."
        if capacity_n is not None
        else "This webpage documents the venue's seating capacity for concerts."
    )
    await verify_with_urls_or_fail(
        evaluator,
        claim=cap_claim,
        node_id=f"{prefix}_capacity_source",
        node_desc="URL reference for capacity information from official or reputable source",
        parent_node=cap_doc_node,
        urls=venue.capacity_urls,
        critical=True,
        additional_instruction="Confirm the page states the seating capacity (approximate values or ranges acceptable). Prefer official or reputable sources."
    )

    # Accessibility (critical, sequential)
    acc_node = evaluator.add_sequential(
        id=f"{prefix}_accessibility",
        desc="ADA wheelchair seating requirements",
        parent=root_parent,
        critical=True
    )

    acc_calc_node = evaluator.add_parallel(
        id=f"{prefix}_wheelchair_calculation",
        desc="Wheelchair-accessible seating calculation",
        parent=acc_node,
        critical=True
    )

    required_seats = compute_required_accessible_seats(capacity_n)
    calc_ok = required_seats is not None and required_seats >= math.ceil(cap_low * 0.01)  # conservative check

    evaluator.add_custom_node(
        result=calc_ok,
        id=f"{prefix}_wheelchair_seats",
        desc=(
            f"Calculate required wheelchair-accessible seats (≈1% of capacity). "
            f"Capacity={capacity_n if capacity_n is not None else 'unknown'}, "
            f"Required≈{required_seats if required_seats is not None else 'unknown'}"
        ),
        parent=acc_calc_node,
        critical=True
    )

    acc_conf_node = evaluator.add_sequential(
        id=f"{prefix}_accessibility_confirmation",
        desc="ADA compliance confirmation",
        parent=acc_node,
        critical=True
    )

    await verify_with_urls_or_fail(
        evaluator,
        claim=f"The venue provides accessible seating or meets ADA requirements.",
        node_id=f"{prefix}_accessibility_statement",
        node_desc="Confirmation that venue provides accessible seating or meets ADA requirements",
        parent_node=acc_conf_node,
        urls=venue.accessibility_urls,
        critical=True,
        additional_instruction="Look for statements about ADA compliance, accessible seating, wheelchair seating, companion seating, ramps, elevators, or similar."
    )

    # Stage (critical, sequential)
    stage_node = evaluator.add_sequential(
        id=f"{prefix}_stage",
        desc="Stage specifications for full band performance",
        parent=root_parent,
        critical=True
    )

    await verify_with_urls_or_fail(
        evaluator,
        claim="The venue can accommodate a stage of at least 24 feet wide by 16 feet deep for a full band production.",
        node_id=f"{prefix}_stage_size",
        node_desc="Venue has or can accommodate a stage of at least 24 feet wide by 16 feet deep",
        parent_node=stage_node,
        urls=venue.stage_urls,
        critical=True,
        additional_instruction="Confirm stage dimensions or technical specs indicate ≥24' width and ≥16' depth (or equivalent area). Accept equivalent phrasing or metric conversions."
    )

    stage_doc_node = evaluator.add_sequential(
        id=f"{prefix}_stage_documentation",
        desc="Stage documentation",
        parent=stage_node,
        critical=True
    )

    await verify_with_urls_or_fail(
        evaluator,
        claim="This page provides stage specifications or technical specs for the venue.",
        node_id=f"{prefix}_stage_source",
        node_desc="URL reference for stage specifications from official or reputable source",
        parent_node=stage_doc_node,
        urls=venue.stage_urls,
        critical=True,
        additional_instruction="Verify the page is a technical specs/stage specs document or official venue tech sheet."
    )

    # Operations (non-critical, parallel)
    ops_node = evaluator.add_parallel(
        id=f"{prefix}_operations",
        desc="Venue operating information",
        parent=root_parent,
        critical=False
    )

    await verify_with_urls_or_fail(
        evaluator,
        claim="The venue requires or accepts standard event insurance (typically $1–2 million liability coverage, COI).",
        node_id=f"{prefix}_insurance",
        node_desc="Venue requires or accepts standard event insurance (typically $1-2 million liability coverage)",
        parent_node=ops_node,
        urls=venue.insurance_urls,
        critical=False,
        additional_instruction="Look for insurance requirements, COI, liability coverage amounts, or risk/insurance policy details on the venue page."
    )

    await verify_with_urls_or_fail(
        evaluator,
        claim="The venue has a curfew policy or time restriction for evening events/concerts.",
        node_id=f"{prefix}_curfew",
        node_desc="Venue has a curfew policy or time restriction for events",
        parent_node=ops_node,
        urls=venue.curfew_urls,
        critical=False,
        additional_instruction="Look for curfew, event end times, noise ordinances, or policy statements on operating hours/restrictions."
    )


# --------------------------------------------------------------------------- #
# Root-level pair requirements (critical gate)                                #
# --------------------------------------------------------------------------- #
def add_pair_requirements_gate(
    evaluator: Evaluator,
    root_node,
    v_small: Optional[VenueInfo],
    v_large: Optional[VenueInfo]
) -> None:
    """Add a critical gate under root to enforce two distinct venues presence."""
    gate = evaluator.add_sequential(
        id="pair_requirements",
        desc="Two distinct venues provided and basic constraints",
        parent=root_node,
        critical=True
    )

    both_present = (
        v_small is not None and (v_small.name or "").strip() and
        v_large is not None and (v_large.name or "").strip()
    )
    evaluator.add_custom_node(
        result=bool(both_present),
        id="both_venues_present",
        desc="Both venues are present with names provided",
        parent=gate,
        critical=True
    )

    names_distinct = False
    if v_small and v_large and v_small.name and v_large.name:
        names_distinct = v_small.name.strip().casefold() != v_large.name.strip().casefold()

    evaluator.add_custom_node(
        result=names_distinct,
        id="venues_distinct_names",
        desc="Two venues have distinct names",
        parent=gate,
        critical=True
    )

    # Optional basic address checks for Chicago to strengthen gate (non-string parsing).
    small_addr_ok = is_chicago_address(v_small.address if v_small else None)
    large_addr_ok = is_chicago_address(v_large.address if v_large else None)

    evaluator.add_custom_node(
        result=small_addr_ok,
        id="small_venue_chicago_address_basic",
        desc="Small-range venue address indicates Chicago, IL (basic string check)",
        parent=gate,
        critical=True
    )
    evaluator.add_custom_node(
        result=large_addr_ok,
        id="large_venue_chicago_address_basic",
        desc="Large-range venue address indicates Chicago, IL (basic string check)",
        parent=gate,
        critical=True
    )

    # Optional basic capacity range checks for gate
    small_cap_n = parse_capacity_number(v_small.capacity_text if v_small else None)
    large_cap_n = parse_capacity_number(v_large.capacity_text if v_large else None)

    evaluator.add_custom_node(
        result=(small_cap_n is not None and SMALL_RANGE[0] <= small_cap_n <= SMALL_RANGE[1]),
        id="small_venue_capacity_range_basic",
        desc=f"Small-range venue capacity appears within {SMALL_RANGE[0]}-{SMALL_RANGE[1]} (basic numeric parse)",
        parent=gate,
        critical=True
    )
    evaluator.add_custom_node(
        result=(large_cap_n is not None and LARGE_RANGE[0] <= large_cap_n <= LARGE_RANGE[1]),
        id="large_venue_capacity_range_basic",
        desc=f"Large-range venue capacity appears within {LARGE_RANGE[0]}-{LARGE_RANGE[1]} (basic numeric parse)",
        parent=gate,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate a single answer for the Chicago concert venues task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Use parallel at root; add a critical gate to enforce both venues
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

    # Extract the two venues
    extracted: VenuesExtraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    v_small = extracted.venue_3000_4000 or VenueInfo()
    v_large = extracted.venue_4500_5500 or VenueInfo()

    # Record computed ADA seat requirements and parsed capacities into custom info for transparency
    small_cap_n = parse_capacity_number(v_small.capacity_text)
    large_cap_n = parse_capacity_number(v_large.capacity_text)
    small_req_seats = compute_required_accessible_seats(small_cap_n)
    large_req_seats = compute_required_accessible_seats(large_cap_n)

    evaluator.add_custom_info(
        info={
            "venue_3000_4000": {
                "name": v_small.name,
                "address": v_small.address,
                "capacity_text": v_small.capacity_text,
                "parsed_capacity": small_cap_n,
                "required_accessible_seats_approx": small_req_seats,
                "id_urls": v_small.id_urls,
                "capacity_urls": v_small.capacity_urls,
                "accessibility_urls": v_small.accessibility_urls,
                "stage_urls": v_small.stage_urls,
                "insurance_urls": v_small.insurance_urls,
                "curfew_urls": v_small.curfew_urls,
            },
            "venue_4500_5500": {
                "name": v_large.name,
                "address": v_large.address,
                "capacity_text": v_large.capacity_text,
                "parsed_capacity": large_cap_n,
                "required_accessible_seats_approx": large_req_seats,
                "id_urls": v_large.id_urls,
                "capacity_urls": v_large.capacity_urls,
                "accessibility_urls": v_large.accessibility_urls,
                "stage_urls": v_large.stage_urls,
                "insurance_urls": v_large.insurance_urls,
                "curfew_urls": v_large.curfew_urls,
            }
        },
        info_type="extraction_summary",
        info_name="extracted_venues_summary"
    )

    # Add a critical gate under root to enforce two distinct venues and basic constraints
    add_pair_requirements_gate(evaluator, root, v_small, v_large)

    # Build and verify sub-tree for each venue (non-critical nodes at venue level to allow partial credit per venue)
    venue_small_node = evaluator.add_parallel(
        id="venue_3000_4000",
        desc="Venue with capacity between 3,000-4,000 seats",
        parent=root,
        critical=False
    )
    venue_large_node = evaluator.add_parallel(
        id="venue_4500_5500",
        desc="Venue with capacity between 4,500-5,500 seats",
        parent=root,
        critical=False
    )

    # Verify both venues
    await verify_single_venue(evaluator, venue_small_node, v_small, "venue_3000_4000", SMALL_RANGE[0], SMALL_RANGE[1])
    await verify_single_venue(evaluator, venue_large_node, v_large, "venue_4500_5500", LARGE_RANGE[0], LARGE_RANGE[1])

    # Return structured result
    return evaluator.get_summary()