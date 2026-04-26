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
TASK_ID = "pa_concert_arena_2026"
TASK_DESCRIPTION = (
    "A major touring artist is planning a concert in Pennsylvania during summer 2026. "
    "Identify a suitable indoor arena venue in Pennsylvania that meets the following requirements: "
    "(1) The venue must be an indoor arena (not an outdoor amphitheater or stadium); "
    "(2) The venue must have a concert seating capacity of at least 18,000; "
    "(3) The venue must provide ADA-compliant wheelchair-accessible seating; "
    "(4) The venue must have adequate emergency exit capacity for safe evacuation; "
    "(5) Preferably, the venue should have truck loading dock access for equipment and backstage dressing room facilities. "
    "Provide the venue name, its location (city, Pennsylvania), its concert capacity, and reference URLs that verify this information."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """Structured info for a single proposed Pennsylvania concert venue."""
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Expect "PA" or "Pennsylvania"
    venue_type: Optional[str] = None  # e.g., "indoor arena", "arena", etc.
    capacity: Optional[str] = None  # Prefer strings to allow ranges or textual qualifiers

    # URL sources explicitly cited in the answer
    venue_urls: List[str] = Field(default_factory=list)          # Official website or authoritative venue page(s)
    capacity_urls: List[str] = Field(default_factory=list)       # Pages that state capacity
    accessibility_urls: List[str] = Field(default_factory=list)  # ADA / accessibility pages
    emergency_urls: List[str] = Field(default_factory=list)      # Safety / egress / evacuation info pages
    loading_dock_urls: List[str] = Field(default_factory=list)   # Technical specs or docks info
    backstage_urls: List[str] = Field(default_factory=list)      # Dressing rooms / backstage facilities
    other_urls: List[str] = Field(default_factory=list)          # Any other relevant references


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return """
    From the provided answer, extract details for exactly one proposed Pennsylvania concert venue.

    Required fields:
    - name: The venue name as stated in the answer.
    - city: The city of the venue, if stated.
    - state: The state associated with the venue (e.g., "PA" or "Pennsylvania") as stated in the answer.
    - venue_type: The described type of the venue (e.g., "indoor arena", "arena", "stadium", "amphitheater").
    - capacity: The stated or implied concert seating capacity for the venue (string as given, do not convert to number).

    Source URL fields (extract only URLs explicitly present in the answer; do not invent):
    - venue_urls: URLs that are official venue websites or highly authoritative venue profile pages (e.g., venue’s official site, operator page).
    - capacity_urls: URLs that document the venue seating capacity.
    - accessibility_urls: URLs that document ADA-compliant or wheelchair-accessible seating for the venue.
    - emergency_urls: URLs that document emergency exit/egress/evacuation capacity or life-safety compliance for the venue.
    - loading_dock_urls: URLs that show truck loading dock access or production load-in/out details.
    - backstage_urls: URLs that show backstage dressing room facilities.
    - other_urls: Any other URLs the answer cites for this venue.

    Rules:
    - Extract only what is explicitly present in the answer. If an item is not present, set it to null (for strings) or an empty list (for URLs).
    - For URLs, accept plain URLs or URLs inside markdown links; extract the actual URL string.
    - Do not extract multiple venues. If multiple are present, select the first one mentioned.

    Return a single JSON object matching the VenueExtraction schema.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _clean_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    seen = set()
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        # accept http/https; if missing protocol, Extractor may have prepended - otherwise keep as is
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _merge_sources(*lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for l in lists:
        for u in _clean_urls(l):
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


async def _verify_with_sources_or_fail(
    evaluator: Evaluator,
    *,
    node_id: str,
    desc: str,
    parent,
    critical: bool,
    claim: str,
    sources: List[str],
    additional_instruction: str
):
    """
    Create a leaf for a URL-grounded claim. If sources are empty, mark the leaf as failed (no verification without evidence).
    Otherwise, run URL-based verification.
    """
    cleaned_sources = _clean_urls(sources)
    if not cleaned_sources:
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=parent,
            critical=critical,
            score=0.0,
            status="failed"
        )
        evaluator.add_custom_info(
            info={"reason": "missing_sources", "node": node_id, "description": desc},
            info_type="missing_source_info",
            info_name=f"missing_sources_{node_id}"
        )
        return leaf

    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=cleaned_sources,
        additional_instruction=additional_instruction
    )
    return leaf


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: VenueExtraction):
    root = evaluator.root

    # Group 1: Venue Basic Qualifications (Critical)
    basic_node = evaluator.add_parallel(
        id="venue_basic_qualifications",
        desc="Venue must be a suitable indoor arena in Pennsylvania with adequate capacity",
        parent=root,
        critical=True
    )

    # Pennsylvania Location (Critical leaf)
    loc_sources = _merge_sources(extracted.venue_urls, extracted.other_urls, extracted.capacity_urls)
    if extracted.name and extracted.city:
        loc_claim = f"The venue named '{extracted.name}' is located in {extracted.city}, Pennsylvania."
    elif extracted.city:
        loc_claim = f"The venue is located in {extracted.city}, Pennsylvania."
    else:
        loc_claim = "This venue is located in the state of Pennsylvania."

    await _verify_with_sources_or_fail(
        evaluator,
        node_id="pennsylvania_location",
        desc="Venue is located within the state of Pennsylvania",
        parent=basic_node,
        critical=True,
        claim=loc_claim,
        sources=loc_sources,
        additional_instruction=(
            "Verify that the webpage clearly identifies the venue's address or city and state as being in Pennsylvania (PA). "
            "If the city is mentioned, it must be a Pennsylvania city."
        )
    )

    # Indoor Arena Type (Critical leaf)
    arena_sources = _merge_sources(extracted.venue_urls, extracted.other_urls)
    arena_claim = (
        "The venue is an indoor arena (fully enclosed, climate-controlled) that is suitable for hosting concerts, "
        "and it is not an outdoor amphitheater or open-air stadium."
    )
    await _verify_with_sources_or_fail(
        evaluator,
        node_id="indoor_arena_type",
        desc="Venue is an indoor arena suitable for concerts",
        parent=basic_node,
        critical=True,
        claim=arena_claim,
        sources=arena_sources,
        additional_instruction=(
            "Look for phrases like 'indoor arena', 'arena', 'enclosed', or indications of a roof/climate control. "
            "If the page emphasizes 'amphitheater' or 'outdoor', it should not pass."
        )
    )

    # Minimum Capacity (Critical leaf)
    capacity_sources = _merge_sources(extracted.capacity_urls, extracted.venue_urls)
    min_capacity_claim = "The venue's concert seating capacity is at least 18,000."
    await _verify_with_sources_or_fail(
        evaluator,
        node_id="minimum_capacity",
        desc="Venue has a concert seating capacity of at least 18,000",
        parent=basic_node,
        critical=True,
        claim=min_capacity_claim,
        sources=capacity_sources,
        additional_instruction=(
            "Check for any stated capacity values (overall, max concert, end-stage, or basketball/hockey capacities). "
            "If a listed capacity is 18,000 or more, treat as satisfying the requirement. Minor rounding differences are acceptable."
        )
    )

    # Group 2: Accessibility and Safety Standards (Critical)
    access_safety_node = evaluator.add_parallel(
        id="accessibility_and_safety",
        desc="Venue must meet ADA accessibility and emergency safety requirements",
        parent=root,
        critical=True
    )

    # ADA Accessible Seating (Critical leaf)
    ada_sources = _merge_sources(extracted.accessibility_urls, extracted.venue_urls)
    ada_claim = "The venue provides ADA-compliant wheelchair-accessible seating options."
    await _verify_with_sources_or_fail(
        evaluator,
        node_id="ada_accessible_seating",
        desc="Venue provides wheelchair-accessible seating options as required by ADA",
        parent=access_safety_node,
        critical=True,
        claim=ada_claim,
        sources=ada_sources,
        additional_instruction=(
            "Look for explicit references to 'ADA', 'accessibility', 'wheelchair accessible seating', "
            "or similar language on the venue's official site or an authoritative source."
        )
    )

    # Emergency Exits (Critical leaf)
    emergency_sources = _merge_sources(extracted.emergency_urls, extracted.venue_urls)
    emergency_claim = (
        "The venue has adequate emergency exit/egress capacity for safe evacuation of at least 18,000 attendees, "
        "consistent with applicable life-safety codes or published venue specifications."
    )
    await _verify_with_sources_or_fail(
        evaluator,
        node_id="emergency_exits",
        desc="Venue has adequate emergency exit capacity for the stated attendance",
        parent=access_safety_node,
        critical=True,
        claim=emergency_claim,
        sources=emergency_sources,
        additional_instruction=(
            "Accept explicit references to emergency exits, egress routes, evacuation plans, or life-safety/egress capacities. "
            "If the page clearly states compliance or adequate egress for full-capacity events, that is sufficient."
        )
    )

    # Group 3: Operational Facilities (Non-critical)
    ops_node = evaluator.add_parallel(
        id="operational_facilities",
        desc="Venue has facilities to support professional touring productions",
        parent=root,
        critical=False
    )

    # Loading Dock (Non-critical leaf)
    loading_sources = _merge_sources(extracted.loading_dock_urls, extracted.venue_urls, extracted.other_urls)
    loading_claim = "The venue has truck loading dock access for production equipment load-in and load-out."
    await _verify_with_sources_or_fail(
        evaluator,
        node_id="loading_dock",
        desc="Venue has truck loading dock access for equipment",
        parent=ops_node,
        critical=False,
        claim=loading_claim,
        sources=loading_sources,
        additional_instruction=(
            "Look for technical specifications, production guides, or venue operations pages that mention loading docks, "
            "truck access, or load-in procedures."
        )
    )

    # Backstage Areas (Non-critical leaf)
    backstage_sources = _merge_sources(extracted.backstage_urls, extracted.venue_urls, extracted.other_urls)
    backstage_claim = "The venue has backstage dressing room facilities available for performers."
    await _verify_with_sources_or_fail(
        evaluator,
        node_id="backstage_areas",
        desc="Venue has dressing room facilities",
        parent=ops_node,
        critical=False,
        claim=backstage_claim,
        sources=backstage_sources,
        additional_instruction=(
            "Look for 'dressing rooms', 'green rooms', 'backstage facilities', 'star dressing room', or similar terminology "
            "in production specs or venue information."
        )
    )

    # Group 4: Documentation and Verification (Critical)
    docs_node = evaluator.add_parallel(
        id="documentation_and_verification",
        desc="All venue claims must be supported by reference URLs",
        parent=root,
        critical=True
    )

    # Venue Website Reference (Critical leaf)
    website_ref_sources = _merge_sources(extracted.venue_urls)
    if extracted.name and extracted.city:
        website_ref_claim = (
            f"This webpage is the official website of '{extracted.name}' or an authoritative venue profile page, "
            f"and it clearly identifies the venue in {extracted.city}, Pennsylvania."
        )
    elif extracted.name:
        website_ref_claim = (
            f"This webpage is the official website of '{extracted.name}' or an authoritative venue profile page in Pennsylvania."
        )
    else:
        website_ref_claim = (
            "This webpage is the official website of the venue or an authoritative venue profile page in Pennsylvania."
        )

    await _verify_with_sources_or_fail(
        evaluator,
        node_id="venue_website_reference",
        desc="Official venue website or authoritative source URL is provided for venue identification",
        parent=docs_node,
        critical=True,
        claim=website_ref_claim,
        sources=website_ref_sources,
        additional_instruction=(
            "Prefer the official venue domain (e.g., venue-owned site). If unavailable, an authoritative operator/management page "
            "or major venue directory page that clearly identifies the venue and its location is acceptable."
        )
    )

    # Capacity Verification Reference (Critical leaf)
    cap_ref_sources = _merge_sources(extracted.capacity_urls, extracted.venue_urls)
    cap_ref_claim = (
        "This webpage explicitly states the venue's seating capacity as at least 18,000, "
        "or lists a numeric seating capacity that is 18,000 or higher."
    )
    await _verify_with_sources_or_fail(
        evaluator,
        node_id="capacity_verification_reference",
        desc="Concert capacity is documented with a verifiable reference URL",
        parent=docs_node,
        critical=True,
        claim=cap_ref_claim,
        sources=cap_ref_sources,
        additional_instruction=(
            "Verify the presence of a numeric capacity value on the page. If multiple capacities are listed (e.g., basketball vs. concert), "
            "use any configuration that meets or exceeds 18,000."
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
    Evaluate an answer for the Pennsylvania indoor concert arena task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Evaluate whether the identified venue in Pennsylvania meets all requirements for hosting a major touring concert with 18,000+ attendees",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured venue info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction"
    )

    # Add some helpful context as custom info (optional)
    evaluator.add_custom_info(
        info={
            "expected_state": "Pennsylvania (PA)",
            "capacity_threshold": ">= 18,000",
            "preference": ["loading dock", "backstage dressing rooms"]
        },
        info_type="evaluation_parameters",
        info_name="pa_venue_requirements"
    )

    # Build and run verification tree
    await build_verification_tree(evaluator, extracted)

    # Return the final structured summary
    return evaluator.get_summary()