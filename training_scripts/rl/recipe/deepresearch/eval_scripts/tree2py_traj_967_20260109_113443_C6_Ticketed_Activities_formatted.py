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
TASK_ID = "nyc_residency_arena_2026"
TASK_DESCRIPTION = (
    "A major touring artist is planning a multi-night concert residency in the New York metropolitan area in 2026 and "
    "requires a modern arena venue. Identify a suitable venue that meets ALL of the following requirements:\n\n"
    "1. Located in New York City or immediately adjacent areas (including the outer boroughs or Nassau County)\n"
    "2. Concert seating capacity between 17,000-20,000 seats\n"
    "3. Offers both luxury suites and loge boxes or club seating for premium ticket buyers\n"
    "4. Has technical capabilities to support arena-scale productions, including adequate backstage facilities and "
    "loading dock access for 6-8 hour load-in requirements\n"
    "5. Can accommodate multi-night residency scheduling with same-show turnaround capabilities\n"
    "6. Requires or accepts standard artist general liability insurance of at least $1 million per occurrence\n"
    "7. Complies with fire safety, emergency exit, and occupancy regulations appropriate for its capacity\n"
    "8. Meets ADA accessibility requirements with wheelchair-accessible seating and facilities\n\n"
    "For the identified venue, provide:\n"
    "- The venue name and specific location\n"
    "- Concert seating capacity with source documentation\n"
    "- Description of available premium seating options (suites and loge/club seating) with source\n"
    "- Technical capabilities including backstage and load-in facilities with source\n"
    "- Confirmation of insurance and safety compliance standards with source\n"
    "- Accessibility features and ADA compliance with source"
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    # Identity and location
    name: Optional[str] = None
    location: Optional[str] = None
    overview_urls: List[str] = Field(default_factory=list)

    # Arena and capacity
    capacity_text: Optional[str] = None
    capacity_sources: List[str] = Field(default_factory=list)

    # Premium seating
    premium_suites_desc: Optional[str] = None
    premium_loge_or_club_desc: Optional[str] = None
    premium_sources: List[str] = Field(default_factory=list)

    # Technical capabilities and load-in
    backstage_desc: Optional[str] = None
    loading_desc: Optional[str] = None
    loadin_time_desc: Optional[str] = None
    technical_sources: List[str] = Field(default_factory=list)

    # Multi-night residency capability
    residency_multi_night_desc: Optional[str] = None
    residency_turnaround_desc: Optional[str] = None
    residency_sources: List[str] = Field(default_factory=list)

    # Insurance and safety compliance
    insurance_liability_desc: Optional[str] = None
    insurance_amount_text: Optional[str] = None
    safety_fire_occupancy_desc: Optional[str] = None
    emergency_exit_desc: Optional[str] = None
    insurance_safety_sources: List[str] = Field(default_factory=list)

    # ADA accessibility
    accessibility_wheelchair_desc: Optional[str] = None
    accessibility_restrooms_paths_desc: Optional[str] = None
    accessibility_ticketing_desc: Optional[str] = None
    accessibility_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return """
    Extract exactly one recommended venue from the answer that is intended for the artist's multi-night concert residency.
    Return a single JSON object filling the fields below, extracting only what is explicitly present in the answer text:
    - name: Venue name
    - location: Specific location details (e.g., borough/city and state, and/or area/address text that identifies geography)
    - overview_urls: All general venue/homepage/about/contact or specification URLs present in the answer (array of URLs)

    - capacity_text: The concert seating capacity as described in the answer (keep as text, may include a number or range)
    - capacity_sources: All URLs cited in the answer that support the concert capacity figure(s) (array of URLs)

    - premium_suites_desc: Text describing luxury suites offering (if present)
    - premium_loge_or_club_desc: Text describing loge boxes or club seating (if present)
    - premium_sources: All URLs cited that support these premium seating offerings (array of URLs)

    - backstage_desc: Text describing backstage facilities (dressing rooms, production offices, etc.) (if present)
    - loading_desc: Text describing loading dock or equipment access (if present)
    - loadin_time_desc: Text describing ability to meet 6–8 hour load-in (if present)
    - technical_sources: All URLs cited that support technical/backstage/loading capabilities (array of URLs)

    - residency_multi_night_desc: Text indicating support for multi-night residency scheduling (if present)
    - residency_turnaround_desc: Text indicating ability to do same-show turnaround between nights (if present)
    - residency_sources: All URLs cited that support residency-related capabilities (array of URLs)

    - insurance_liability_desc: Text describing general liability insurance requirement or acceptance (if present)
    - insurance_amount_text: The amount stated for general liability per occurrence (as text, e.g., "$1 million") (if present)
    - safety_fire_occupancy_desc: Text indicating compliance with fire safety/occupancy regulations (if present)
    - emergency_exit_desc: Text describing emergency exit capacity or compliance (if present)
    - insurance_safety_sources: All URLs cited that support insurance/safety/occupancy/egress compliance (array of URLs)

    - accessibility_wheelchair_desc: Text describing wheelchair-accessible seating (if present)
    - accessibility_restrooms_paths_desc: Text describing accessible restrooms and accessible paths/routes (if present)
    - accessibility_ticketing_desc: Text describing equivalence of accessible ticketing (same conditions/prices) (if present)
    - accessibility_sources: All URLs cited that support ADA/accessibility features (array of URLs)

    Rules:
    - Extract URLs exactly as in the answer; include only valid URLs.
    - If a field is not explicitly present in the answer, set it to null (for strings) or [] (for arrays).
    - If multiple venues are mentioned, choose the one that is the main recommended venue for the residency.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for u in urls:
        if not u:
            continue
        if isinstance(u, str):
            uu = u.strip()
            if uu and uu not in seen:
                seen.add(uu)
                ordered.append(uu)
    return ordered


def _pick_sources(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        merged.extend(lst or [])
    return _dedup_urls(merged)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_venue_identity_and_location(
    evaluator: Evaluator,
    parent,
    data: VenueExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="venue_identity_and_location",
        desc="Answer provides the venue name and specific location, and the venue is within the allowed geography (NYC or immediately adjacent areas including outer boroughs or Nassau County)",
        parent=parent,
        critical=True,
    )

    # Existence checks first (critical gating)
    evaluator.add_custom_node(
        result=bool(data.name and data.name.strip()),
        id="venue_name_provided",
        desc="Venue name is explicitly stated",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.location and data.location.strip()),
        id="specific_location_provided",
        desc="Specific location is provided (e.g., city/borough and state and/or address area) sufficient to verify geography",
        parent=node,
        critical=True,
    )

    # Geography constraint verification
    geo_leaf = evaluator.add_leaf(
        id="geography_constraint_satisfied",
        desc="Venue location is in New York City or immediately adjacent areas (outer boroughs or Nassau County)",
        parent=node,
        critical=True,
    )

    claim = (
        f"The venue '{data.name or ''}' located at '{data.location or ''}' is in New York City "
        f"(Manhattan, Brooklyn, Queens, Bronx, Staten Island) or in Nassau County, New York."
    )
    geo_sources = _pick_sources(data.overview_urls)
    await evaluator.verify(
        claim=claim,
        node=geo_leaf,
        sources=geo_sources,
        additional_instruction="Accept if the location is within any of NYC's five boroughs or Nassau County. Minor formatting differences are fine.",
    )


async def build_arena_and_capacity(
    evaluator: Evaluator,
    parent,
    data: VenueExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="arena_and_capacity",
        desc="Venue is an arena suitable for major touring concerts and meets the required concert capacity range, with source documentation for capacity",
        parent=parent,
        critical=True,
    )

    # Source presence for capacity (critical gating)
    evaluator.add_custom_node(
        result=len(data.capacity_sources) > 0,
        id="capacity_source_provided",
        desc="Source documentation/citation is provided for the stated concert seating capacity",
        parent=node,
        critical=True,
    )

    # Arena style verification
    arena_leaf = evaluator.add_leaf(
        id="arena_style_venue",
        desc="Venue is an arena-style venue suitable for arena-scale concert productions",
        parent=node,
        critical=True,
    )
    arena_sources = _pick_sources(data.overview_urls, data.technical_sources)
    claim_arena = (
        f"'{data.name or 'This venue'}' is an arena-style venue suitable for arena-scale concert productions."
    )
    await evaluator.verify(
        claim=claim_arena,
        node=arena_leaf,
        sources=arena_sources,
        additional_instruction="Look for the venue being described as an arena, with seating bowl and event configuration suitable for major concerts.",
    )

    # Capacity in range verification (use capacity sources)
    capacity_leaf = evaluator.add_leaf(
        id="capacity_in_range",
        desc="Concert seating capacity is between 17,000 and 20,000 seats",
        parent=node,
        critical=True,
    )
    claim_capacity = "The concert seating capacity is between 17,000 and 20,000 seats."
    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_leaf,
        sources=_pick_sources(data.capacity_sources),
        additional_instruction=(
            "Confirm using the provided source(s). Use the concert configuration capacity (not necessarily basketball/hockey). "
            "Allow reasonable variants or ranges as long as they fall within 17,000–20,000."
        ),
    )


async def build_premium_seating(
    evaluator: Evaluator,
    parent,
    data: VenueExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="premium_seating",
        desc="Venue offers required premium seating (luxury suites and loge boxes or club seating), with source documentation",
        parent=parent,
        critical=True,
    )

    # Source presence first (critical)
    evaluator.add_custom_node(
        result=len(data.premium_sources) > 0,
        id="premium_seating_source_provided",
        desc="Source documentation/citation is provided supporting the premium seating offerings",
        parent=node,
        critical=True,
    )

    suites_leaf = evaluator.add_leaf(
        id="luxury_suites_available",
        desc="Venue offers luxury suites",
        parent=node,
        critical=True,
    )
    claim_suites = "The venue offers luxury suites for premium ticket buyers."
    await evaluator.verify(
        claim=claim_suites,
        node=suites_leaf,
        sources=_pick_sources(data.premium_sources, data.overview_urls),
        additional_instruction="Verify from premium seating or hospitality pages that luxury suites are offered.",
    )

    loge_leaf = evaluator.add_leaf(
        id="loge_or_club_seating_available",
        desc="Venue offers loge boxes or club seating options",
        parent=node,
        critical=True,
    )
    claim_loge = "The venue offers loge boxes or club seating options."
    await evaluator.verify(
        claim=claim_loge,
        node=loge_leaf,
        sources=_pick_sources(data.premium_sources, data.overview_urls),
        additional_instruction="Verify presence of loge boxes and/or club seating (e.g., club seats, lounge seating, loge).",
    )


async def build_technical_capabilities(
    evaluator: Evaluator,
    parent,
    data: VenueExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="technical_capabilities_and_loadin",
        desc="Venue has technical/backstage/loading capabilities for arena-scale productions including 6–8 hour load-in, with source documentation",
        parent=parent,
        critical=True,
    )

    # Source presence first (critical)
    evaluator.add_custom_node(
        result=len(data.technical_sources) > 0,
        id="technical_capabilities_source_provided",
        desc="Source documentation/citation is provided for technical/backstage/loading capabilities",
        parent=node,
        critical=True,
    )

    backstage_leaf = evaluator.add_leaf(
        id="backstage_facilities",
        desc="Venue has adequate backstage facilities including dressing rooms and production offices",
        parent=node,
        critical=True,
    )
    claim_backstage = (
        "The venue has adequate backstage facilities including dressing rooms and production offices suitable for major touring productions."
    )
    await evaluator.verify(
        claim=claim_backstage,
        node=backstage_leaf,
        sources=_pick_sources(data.technical_sources),
        additional_instruction="Confirm mentions of dressing rooms, production offices, green rooms, or similar backstage support spaces.",
    )

    loading_leaf = evaluator.add_leaf(
        id="loading_dock_or_equipment_access",
        desc="Venue has a loading dock or suitable equipment access for large-scale productions",
        parent=node,
        critical=True,
    )
    claim_loading = "The venue provides loading docks or suitable equipment access for large-scale touring productions."
    await evaluator.verify(
        claim=claim_loading,
        node=loading_leaf,
        sources=_pick_sources(data.technical_sources),
        additional_instruction="Look for mentions of loading docks, truck access, freight elevators, or ramp access for production gear.",
    )

    loadin_time_leaf = evaluator.add_leaf(
        id="supports_6_to_8_hour_loadin",
        desc="Venue can accommodate 6–8 hour load-in requirements for arena-scale productions",
        parent=node,
        critical=True,
    )
    claim_loadin = "The venue can accommodate 6–8 hour load-in requirements typical for arena-scale concert productions."
    await evaluator.verify(
        claim=claim_loadin,
        node=loadin_time_leaf,
        sources=_pick_sources(data.technical_sources),
        additional_instruction="Accept if documentation implies standard same-day arena load-in windows of roughly 6–8 hours or equivalent readiness.",
    )


async def build_residency_capability(
    evaluator: Evaluator,
    parent,
    data: VenueExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="multi_night_residency_capability",
        desc="Venue can accommodate multi-night residency scheduling with same-show turnaround capabilities",
        parent=parent,
        critical=True,
    )

    multi_night_leaf = evaluator.add_leaf(
        id="supports_multi_night_scheduling",
        desc="Venue can accommodate multi-night residency scheduling",
        parent=node,
        critical=True,
    )
    claim_multi = "The venue can accommodate multi-night residency scheduling for the same artist."
    await evaluator.verify(
        claim=claim_multi,
        node=multi_night_leaf,
        sources=_pick_sources(data.residency_sources, data.overview_urls),
        additional_instruction="Evidence can include references to residencies or multiple consecutive nights by a single artist.",
    )

    turnaround_leaf = evaluator.add_leaf(
        id="supports_same_show_turnaround",
        desc="Venue can support same-show turnaround capabilities between nights",
        parent=node,
        critical=True,
    )
    claim_turnaround = "The venue can support same-show turnaround capabilities between consecutive nights."
    await evaluator.verify(
        claim=claim_turnaround,
        node=turnaround_leaf,
        sources=_pick_sources(data.residency_sources, data.technical_sources),
        additional_instruction="Look for references to changeover/turnaround capabilities or operational readiness for back-to-back nights.",
    )


async def build_insurance_and_safety(
    evaluator: Evaluator,
    parent,
    data: VenueExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="insurance_and_safety_compliance",
        desc="Venue meets insurance and safety-related constraints, with source documentation",
        parent=parent,
        critical=True,
    )

    # Source presence first (critical)
    evaluator.add_custom_node(
        result=len(data.insurance_safety_sources) > 0,
        id="insurance_and_safety_source_provided",
        desc="Source documentation/citation is provided for insurance requirements/acceptance and safety compliance standards",
        parent=node,
        critical=True,
    )

    liability_leaf = evaluator.add_leaf(
        id="liability_insurance_minimum",
        desc="Venue requires or accepts standard artist general liability insurance of at least $1 million per occurrence",
        parent=node,
        critical=True,
    )
    claim_liability = (
        "The venue requires or accepts standard artist general liability insurance of at least $1,000,000 per occurrence."
    )
    await evaluator.verify(
        claim=claim_liability,
        node=liability_leaf,
        sources=_pick_sources(data.insurance_safety_sources),
        additional_instruction="Confirm that the minimum per-occurrence general liability amount is at least $1,000,000.",
    )

    fire_occ_leaf = evaluator.add_leaf(
        id="fire_safety_and_occupancy_compliance",
        desc="Venue complies with fire safety and occupancy regulations appropriate for its capacity",
        parent=node,
        critical=True,
    )
    claim_fire_occ = "The venue complies with applicable fire safety and occupancy regulations appropriate for its capacity."
    await evaluator.verify(
        claim=claim_fire_occ,
        node=fire_occ_leaf,
        sources=_pick_sources(data.insurance_safety_sources),
        additional_instruction="Look for official policy, building code compliance statements, or authoritative documentation.",
    )

    egress_leaf = evaluator.add_leaf(
        id="emergency_exit_capacity",
        desc="Venue has adequate emergency exit capacity for its stated occupancy",
        parent=node,
        critical=True,
    )
    claim_egress = "The venue has adequate emergency exit capacity for its stated occupancy."
    await evaluator.verify(
        claim=claim_egress,
        node=egress_leaf,
        sources=_pick_sources(data.insurance_safety_sources),
        additional_instruction="Accept documentation or policy references indicating adequate egress for the venue's occupancy.",
    )


async def build_accessibility(
    evaluator: Evaluator,
    parent,
    data: VenueExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="ada_accessibility",
        desc="Venue meets ADA accessibility constraints, with source documentation",
        parent=parent,
        critical=True,
    )

    # Source presence first (critical)
    evaluator.add_custom_node(
        result=len(data.accessibility_sources) > 0,
        id="accessibility_source_provided",
        desc="Source documentation/citation is provided for accessibility/ADA features",
        parent=node,
        critical=True,
    )

    wheelchair_leaf = evaluator.add_leaf(
        id="wheelchair_accessible_seating",
        desc="Venue provides wheelchair-accessible seating locations meeting ADA requirements",
        parent=node,
        critical=True,
    )
    claim_wheelchair = "The venue provides wheelchair-accessible seating meeting ADA requirements."
    await evaluator.verify(
        claim=claim_wheelchair,
        node=wheelchair_leaf,
        sources=_pick_sources(data.accessibility_sources),
        additional_instruction="Look for explicit mentions of wheelchair-accessible seating areas that comply with ADA.",
    )

    rest_path_leaf = evaluator.add_leaf(
        id="accessible_restrooms_and_paths",
        desc="Venue has accessible restrooms and accessible pathways",
        parent=node,
        critical=True,
    )
    claim_rest_path = "The venue has accessible restrooms and accessible routes or pathways."
    await evaluator.verify(
        claim=claim_rest_path,
        node=rest_path_leaf,
        sources=_pick_sources(data.accessibility_sources),
        additional_instruction="Verify availability of accessible restrooms and accessible routes/paths to seating and amenities.",
    )

    ticket_eq_leaf = evaluator.add_leaf(
        id="accessible_ticketing_equivalence",
        desc="Accessible seating tickets are available under the same conditions as other seats",
        parent=node,
        critical=True,
    )
    claim_ticket_eq = "Accessible seating tickets are available under the same conditions as other seats."
    await evaluator.verify(
        claim=claim_ticket_eq,
        node=ticket_eq_leaf,
        sources=_pick_sources(data.accessibility_sources),
        additional_instruction="Look for policy statements indicating equivalent pricing/terms for accessible seats in line with ADA guidance.",
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
        task_description="Identify one suitable modern arena venue for a multi-night concert residency in the NYC metro area that satisfies all stated constraints and required deliverables",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured info from the answer
    data: VenueExtraction = await evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Build rubric tree (root is critical; all children must be critical)
    # Section 1: Identity and Location
    await build_venue_identity_and_location(evaluator, root, data)

    # Section 2: Arena + Capacity
    await build_arena_and_capacity(evaluator, root, data)

    # Section 3: Premium Seating
    await build_premium_seating(evaluator, root, data)

    # Section 4: Technical Capabilities and Load-in
    await build_technical_capabilities(evaluator, root, data)

    # Section 5: Residency capability
    await build_residency_capability(evaluator, root, data)

    # Section 6: Insurance and Safety
    await build_insurance_and_safety(evaluator, root, data)

    # Section 7: ADA Accessibility
    await build_accessibility(evaluator, root, data)

    # Return evaluation summary
    return evaluator.get_summary()