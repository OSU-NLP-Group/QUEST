import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "texas_outdoor_amphitheater"
TASK_DESCRIPTION = (
    "Identify an outdoor amphitheater venue located in Texas that meets all of the following operational and facility "
    "requirements:\n"
    "1. Total venue capacity must be between 12,000 and 20,000 attendees\n"
    "2. The venue must have both fixed reserved seating and a general admission lawn area\n"
    "3. Fixed reserved seating capacity must be at least 6,000 seats\n"
    "4. A lawn or general admission grass seating area must be present\n"
    "5. The venue must have a permanent stage structure\n"
    "6. On-site parking facilities must be available\n"
    "7. Permanent restroom facilities must be present\n"
    "8. A truck loading dock or dedicated equipment load-in area must exist\n"
    "9. Dressing rooms or green rooms for performers must be available\n"
    "10. Concession stands or food/beverage areas must be present\n"
    "11. An on-site box office or ticket sales location must exist\n"
    "12. Designated wheelchair accessible seating sections must be available\n"
    "13. The venue must be currently active and hosting live music events\n"
    "14. The venue must be operated by a recognized professional venue management company\n\n"
    "Provide the official name of the venue and reference URLs documenting each of these requirements."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    # Identity
    venue_name: Optional[str] = None
    venue_name_sources: List[str] = Field(default_factory=list)

    # Location and type
    outdoor_texas_sources: List[str] = Field(default_factory=list)

    # Capacity and seating
    total_capacity_sources: List[str] = Field(default_factory=list)
    reserved_seating_min_6000_sources: List[str] = Field(default_factory=list)
    lawn_sources: List[str] = Field(default_factory=list)

    # Facilities
    permanent_stage_sources: List[str] = Field(default_factory=list)
    parking_sources: List[str] = Field(default_factory=list)
    restrooms_sources: List[str] = Field(default_factory=list)
    loading_dock_sources: List[str] = Field(default_factory=list)
    dressing_rooms_sources: List[str] = Field(default_factory=list)
    concessions_sources: List[str] = Field(default_factory=list)
    box_office_sources: List[str] = Field(default_factory=list)
    wheelchair_sources: List[str] = Field(default_factory=list)

    # Activity
    active_events_sources: List[str] = Field(default_factory=list)

    # Operator
    operator_name: Optional[str] = None
    operator_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return (
        "From the answer, extract the structured information for a single Texas outdoor amphitheater venue and the "
        "URLs cited to support each required property. Only extract URLs explicitly present in the answer. If a field "
        "is not present in the answer, set it to null (for strings) or an empty list (for URL lists). Do not invent "
        "any URLs.\n\n"
        "Return a JSON object with the following fields:\n"
        "1) venue_name: string | null (the official venue name as stated in the answer)\n"
        "2) venue_name_sources: list of URL strings (URLs that support the official name)\n"
        "3) outdoor_texas_sources: list of URL strings (URLs supporting that it is an outdoor amphitheater in Texas)\n"
        "4) total_capacity_sources: list of URL strings (URLs supporting that total capacity is between 12,000 and 20,000)\n"
        "5) reserved_seating_min_6000_sources: list of URL strings (URLs supporting fixed reserved seating of at least 6,000)\n"
        "6) lawn_sources: list of URL strings (URLs supporting a general admission lawn/grass seating area exists)\n"
        "7) permanent_stage_sources: list of URL strings (URLs supporting a permanent stage structure exists)\n"
        "8) parking_sources: list of URL strings (URLs supporting on-site parking is available)\n"
        "9) restrooms_sources: list of URL strings (URLs supporting permanent restroom facilities are present)\n"
        "10) loading_dock_sources: list of URL strings (URLs supporting a truck loading dock or load-in area exists)\n"
        "11) dressing_rooms_sources: list of URL strings (URLs supporting dressing rooms or green rooms exist)\n"
        "12) concessions_sources: list of URL strings (URLs supporting food/beverage or concession stands are present)\n"
        "13) box_office_sources: list of URL strings (URLs supporting an on-site box office or ticket sales location exists)\n"
        "14) wheelchair_sources: list of URL strings (URLs supporting designated wheelchair-accessible seating exists)\n"
        "15) active_events_sources: list of URL strings (URLs supporting the venue is currently active and hosting live music events)\n"
        "16) operator_name: string | null (the operator/management company name as stated in the answer)\n"
        "17) operator_sources: list of URL strings (URLs stating the venue is operated/managed by that professional company)\n\n"
        "If URLs are missing a protocol, prepend http://. Remove duplicates and keep only valid-looking URLs."
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _normalize_urls(urls: Optional[List[str]]) -> List[str]:
    """Normalize a list of URL strings: strip, ensure protocol, deduplicate."""
    if not urls:
        return []
    normalized: List[str] = []
    for u in urls:
        if not u:
            continue
        s = u.strip()
        if not s:
            continue
        if not (s.startswith("http://") or s.startswith("https://")):
            s = "http://" + s
        if s not in normalized:
            normalized.append(s)
    return normalized


def _safe_name(name: Optional[str]) -> str:
    return name.strip() if name else ""


# --------------------------------------------------------------------------- #
# Verification node builders                                                  #
# --------------------------------------------------------------------------- #
async def _build_official_name_requirement(
    evaluator: Evaluator,
    parent,
    venue_name: Optional[str],
    name_sources: List[str],
) -> None:
    node = evaluator.add_parallel(
        id="Official_Venue_Name_With_Citation",
        desc="Provides the venue’s official name AND includes at least one URL that supports the stated official name.",
        parent=parent,
        critical=True,
    )

    # Existence: name present
    evaluator.add_custom_node(
        result=bool(venue_name and venue_name.strip()),
        id="official_name_present",
        desc="Official venue name is provided in the answer.",
        parent=node,
        critical=True,
    )

    # Existence: sources present
    name_srcs = _normalize_urls(name_sources)
    evaluator.add_custom_node(
        result=len(name_srcs) > 0,
        id="official_name_sources_present",
        desc="At least one source URL is provided for the official name.",
        parent=node,
        critical=True,
    )

    # Verification: name supported by sources
    verify_leaf = evaluator.add_leaf(
        id="official_name_supported_by_sources",
        desc="The official venue name is supported by the cited sources.",
        parent=node,
        critical=True,
    )
    claim = f"The venue's official name is '{_safe_name(venue_name)}'."
    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=name_srcs,
        additional_instruction=(
            "Verify that the provided page(s) clearly use or confirm the exact same official venue name. "
            "Allow minor variations (e.g., presence/absence of 'The', middle names, or abbreviations) if they clearly "
            "refer to the same venue."
        ),
    )


async def _build_simple_requirement(
    evaluator: Evaluator,
    parent,
    req_id: str,
    req_desc: str,
    claim: str,
    sources: List[str],
    add_ins: str,
) -> None:
    node = evaluator.add_parallel(
        id=req_id,
        desc=req_desc,
        parent=parent,
        critical=True,
    )

    # Existence: at least one URL
    srcs = _normalize_urls(sources)
    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id=f"{req_id}_sources_present",
        desc="At least one supporting source URL is provided for this requirement.",
        parent=node,
        critical=True,
    )

    # Verification against sources
    verify_leaf = evaluator.add_leaf(
        id=f"{req_id}_supported_by_sources",
        desc=f"{req_desc} — Verified by cited sources.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=srcs,
        additional_instruction=add_ins,
    )


async def _build_operator_requirement(
    evaluator: Evaluator,
    parent,
    venue_name: Optional[str],
    operator_name: Optional[str],
    operator_sources: List[str],
) -> None:
    node = evaluator.add_parallel(
        id="Operated_By_Professional_Venue_Management_Company_With_Citation",
        desc="Identifies the operator/management company and provides at least one URL indicating the venue is operated/managed by a professional venue management company (as stated in the source).",
        parent=parent,
        critical=True,
    )

    # Existence: operator name present
    evaluator.add_custom_node(
        result=bool(operator_name and operator_name.strip()),
        id="operator_name_present",
        desc="Operator/management company name is provided in the answer.",
        parent=node,
        critical=True,
    )

    # Existence: operator sources present
    srcs = _normalize_urls(operator_sources)
    evaluator.add_custom_node(
        result=len(srcs) > 0,
        id="operator_sources_present",
        desc="At least one source URL is provided for the operator/management company.",
        parent=node,
        critical=True,
    )

    # Verification: operator supported by sources
    verify_leaf = evaluator.add_leaf(
        id="operator_supported_by_sources",
        desc="The operator/management company is supported by the cited sources.",
        parent=node,
        critical=True,
    )

    op_name = _safe_name(operator_name)
    vname = _safe_name(venue_name)
    claim = f"The venue '{vname}' is operated or managed by the professional venue management company '{op_name}'."
    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=srcs,
        additional_instruction=(
            "Confirm that the page explicitly states operation/management by the named company (e.g., Live Nation, "
            "ASM Global, OVG, or another professional venue operator). The reference should clearly indicate a formal "
            "operational/management relationship, not just sponsorship."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    # Initialize evaluator
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

    # Extract structured info from the answer
    extraction: VenueExtraction = await evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Main critical node (as per rubric)
    main_node = evaluator.add_parallel(
        id="Texas_Outdoor_Amphitheater_Identification",
        desc="Identify one outdoor amphitheater venue in Texas that satisfies all listed operational/facility constraints, and provide at least one reference URL supporting each required property.",
        parent=root,
        critical=True,
    )

    # 1) Official Venue Name with citation
    await _build_official_name_requirement(
        evaluator=evaluator,
        parent=main_node,
        venue_name=extraction.venue_name,
        name_sources=extraction.venue_name_sources,
    )

    # 2) Outdoor amphitheater in Texas
    await _build_simple_requirement(
        evaluator=evaluator,
        parent=main_node,
        req_id="Outdoor_Amphitheater_In_Texas_With_Citation",
        req_desc="Shows the venue is an outdoor amphitheater located in Texas AND includes at least one URL supporting both the venue type (outdoor amphitheater) and the Texas location.",
        claim=f"The venue '{_safe_name(extraction.venue_name)}' is an outdoor (open-air) amphitheater located in the U.S. state of Texas.",
        sources=extraction.outdoor_texas_sources,
        add_ins=(
            "The evidence should explicitly indicate both that the venue is an outdoor/open-air amphitheater (allow synonyms "
            "like 'outdoor pavilion' or 'open-air amphitheater') and that it is in Texas. A single page can satisfy both "
            "if it clearly states both facts."
        ),
    )

    # 3) Total capacity between 12,000 and 20,000
    await _build_simple_requirement(
        evaluator=evaluator,
        parent=main_node,
        req_id="Total_Capacity_12000_to_20000_With_Citation",
        req_desc="Shows total venue capacity is between 12,000 and 20,000 attendees AND includes at least one URL supporting the total capacity figure.",
        claim=f"The total capacity of the venue '{_safe_name(extraction.venue_name)}' is between 12,000 and 20,000 attendees.",
        sources=extraction.total_capacity_sources,
        add_ins=(
            "Confirm that the total capacity (including fixed seats plus lawn/GA when applicable) falls within 12,000–20,000. "
            "If the source lists separate numbers (e.g., fixed seating plus lawn), it should imply a total within this range."
        ),
    )

    # 4) Fixed reserved seating capacity at least 6,000 seats
    await _build_simple_requirement(
        evaluator=evaluator,
        parent=main_node,
        req_id="Fixed_Reserved_Seating_Min_6000_With_Citation",
        req_desc="Shows the venue has fixed reserved seating with capacity of at least 6,000 seats AND includes at least one URL supporting this reserved-seating capacity.",
        claim=f"The venue '{_safe_name(extraction.venue_name)}' has at least 6,000 fixed reserved seats (not including lawn/GA).",
        sources=extraction.reserved_seating_min_6000_sources,
        add_ins=(
            "Look for language like 'reserved seating', 'fixed seats', or 'covered seating' that clearly excludes the lawn/GA. "
            "The number should be at least 6,000."
        ),
    )

    # 5) General admission lawn present
    await _build_simple_requirement(
        evaluator=evaluator,
        parent=main_node,
        req_id="General_Admission_Lawn_Present_With_Citation",
        req_desc="Shows a general admission lawn/grass seating area is present AND includes at least one URL supporting the presence of the lawn/grass GA area.",
        claim=f"The venue '{_safe_name(extraction.venue_name)}' has a general admission lawn or grass seating area.",
        sources=extraction.lawn_sources,
        add_ins="Confirm the presence of a lawn/grass GA area used for audience seating.",
    )

    # 6) Permanent stage structure
    await _build_simple_requirement(
        evaluator=evaluator,
        parent=main_node,
        req_id="Permanent_Stage_Structure_With_Citation",
        req_desc="Shows the venue has a permanent stage structure AND includes at least one URL supporting this.",
        claim=f"The venue '{_safe_name(extraction.venue_name)}' has a permanent stage structure.",
        sources=extraction.permanent_stage_sources,
        add_ins="Look for language such as 'permanent stage', 'fixed stage', or production specs indicating a permanent structure.",
    )

    # 7) On-site parking available
    await _build_simple_requirement(
        evaluator=evaluator,
        parent=main_node,
        req_id="Onsite_Parking_Available_With_Citation",
        req_desc="Shows on-site parking facilities are available AND includes at least one URL supporting this.",
        claim=f"On-site parking is available at the venue '{_safe_name(extraction.venue_name)}'.",
        sources=extraction.parking_sources,
        add_ins="The evidence should describe venue parking (on-site lots/garages/areas).",
    )

    # 8) Permanent restrooms present
    await _build_simple_requirement(
        evaluator=evaluator,
        parent=main_node,
        req_id="Permanent_Restrooms_Present_With_Citation",
        req_desc="Shows permanent restroom facilities are present AND includes at least one URL supporting this.",
        claim=f"The venue '{_safe_name(extraction.venue_name)}' has permanent restroom facilities.",
        sources=extraction.restrooms_sources,
        add_ins="Confirm that permanent/installed restrooms are available; not solely temporary portable toilets.",
    )

    # 9) Truck loading dock or dedicated load-in area
    await _build_simple_requirement(
        evaluator=evaluator,
        parent=main_node,
        req_id="Truck_Loading_or_Dedicated_LoadIn_With_Citation",
        req_desc="Shows a truck loading dock or dedicated equipment load-in area exists AND includes at least one URL supporting this.",
        claim=f"The venue '{_safe_name(extraction.venue_name)}' has a truck loading dock or a dedicated equipment load-in area.",
        sources=extraction.loading_dock_sources,
        add_ins="Production/technical specs pages or venue maps should indicate truck dock/load-in locations.",
    )

    # 10) Dressing rooms or green rooms available
    await _build_simple_requirement(
        evaluator=evaluator,
        parent=main_node,
        req_id="Dressing_or_Green_Rooms_With_Citation",
        req_desc="Shows dressing rooms and/or green rooms for performers are available AND includes at least one URL supporting this.",
        claim=f"The venue '{_safe_name(extraction.venue_name)}' provides dressing rooms and/or green rooms for performers.",
        sources=extraction.dressing_rooms_sources,
        add_ins="Artist info or production specs should indicate backstage dressing/green rooms are available.",
    )

    # 11) Concessions / food & beverage present
    await _build_simple_requirement(
        evaluator=evaluator,
        parent=main_node,
        req_id="Concessions_Food_Beverage_With_Citation",
        req_desc="Shows concession stands or food/beverage areas are present AND includes at least one URL supporting this.",
        claim=f"Concession stands or food/beverage areas are available at the venue '{_safe_name(extraction.venue_name)}'.",
        sources=extraction.concessions_sources,
        add_ins="Visitor info pages typically list food, beverage, or concessions availability.",
    )

    # 12) On-site box office / ticket sales location
    await _build_simple_requirement(
        evaluator=evaluator,
        parent=main_node,
        req_id="Onsite_Box_Office_Ticket_Sales_With_Citation",
        req_desc="Shows an on-site box office or ticket sales location exists AND includes at least one URL supporting this.",
        claim=f"There is an on-site box office or ticket sales location at the venue '{_safe_name(extraction.venue_name)}'.",
        sources=extraction.box_office_sources,
        add_ins="Look for 'box office', 'ticket office', or on-site ticketing details.",
    )

    # 13) Wheelchair-accessible seating sections
    await _build_simple_requirement(
        evaluator=evaluator,
        parent=main_node,
        req_id="Wheelchair_Accessible_Seating_With_Citation",
        req_desc="Shows designated wheelchair-accessible seating sections are available AND includes at least one URL supporting this.",
        claim=f"Designated wheelchair-accessible seating sections are available at the venue '{_safe_name(extraction.venue_name)}'.",
        sources=extraction.wheelchair_sources,
        add_ins="Accessibility/ADA pages often detail wheelchair seating sections and policies.",
    )

    # 14) Currently active and hosting live music events
    await _build_simple_requirement(
        evaluator=evaluator,
        parent=main_node,
        req_id="Currently_Active_Hosting_Live_Music_With_Citation",
        req_desc="Shows the venue is currently active and hosting live music events AND includes at least one URL supporting current/ongoing events activity (e.g., events calendar/upcoming shows page).",
        claim=f"The venue '{_safe_name(extraction.venue_name)}' is currently active and hosts live music events.",
        sources=extraction.active_events_sources,
        add_ins=(
            "Prefer an official events calendar or an authoritative listing that shows upcoming or very recent live music events. "
            "Presence of an events schedule with current/future dates is strong evidence."
        ),
    )

    # 15) Operated by a professional venue management company
    await _build_operator_requirement(
        evaluator=evaluator,
        parent=main_node,
        venue_name=extraction.venue_name,
        operator_name=extraction.operator_name,
        operator_sources=extraction.operator_sources,
    )

    # Return evaluation summary
    return evaluator.get_summary()