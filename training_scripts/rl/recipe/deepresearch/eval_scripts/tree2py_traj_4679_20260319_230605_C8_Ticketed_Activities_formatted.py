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
TASK_ID = "tour_venue_selection_us_four_indoor_15k_25k"
TASK_DESCRIPTION = """
You are planning a multi-city concert tour across the United States and need to identify four suitable indoor concert venues. Each venue must be located in a different US state and must meet all of the following requirements:

Geographic Requirements:
- One venue must be located in California
- One venue must be located in New York
- One venue must be located in Texas
- One venue must be located in either Washington, North Carolina, or Illinois

Venue Specifications:
- Each venue must be an indoor arena or facility (not an outdoor amphitheater or open-air stadium)
- Each venue must have an indoor concert seating capacity between 15,000 and 25,000 people

Accessibility Requirements:
- Each venue must provide wheelchair-accessible seating in compliance with ADA (Americans with Disabilities Act) requirements

For each of the four venues, provide:
1. The venue name
2. The city and state location
3. The indoor concert seating capacity
4. Confirmation that it is an indoor facility
5. Confirmation of wheelchair-accessible seating availability
6. A reference URL from an official source (venue website, venue management company, or established venue directory) that verifies the capacity and facility type
"""

ADDITIONAL_ALLOWED_STATES = ["Washington", "North Carolina", "Illinois"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueEntry(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None  # keep as string to maximize compatibility
    is_indoor: Optional[str] = None  # "yes"/"no"/"unknown" as free text
    wheelchair_accessible: Optional[str] = None  # "yes"/"no"/"unknown" as free text

    # URLs
    reference_urls: List[str] = Field(default_factory=list)  # intended to verify capacity + indoor facility type
    ada_urls: List[str] = Field(default_factory=list)  # intended to verify ADA/accessible seating
    logistics_urls: List[str] = Field(default_factory=list)  # intended for loading dock verification
    green_room_urls: List[str] = Field(default_factory=list)  # intended for green room/dressing rooms verification


class VenuesExtraction(BaseModel):
    california: Optional[VenueEntry] = None
    new_york: Optional[VenueEntry] = None
    texas: Optional[VenueEntry] = None
    additional: Optional[VenueEntry] = None  # must be in WA/NC/IL


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract structured information for exactly four venues from the answer, mapping them into these four buckets by state:
    - california: a venue in California
    - new_york: a venue in New York
    - texas: a venue in Texas
    - additional: a venue in either Washington, North Carolina, or Illinois (choose the first one mentioned among these three)

    For each venue, extract the following fields:
    1) name: the venue name as written in the answer
    2) city: the city of the venue as written in the answer
    3) state: the U.S. state as written in the answer (use full state names, e.g., "California", "New York", "Texas", "Washington", "North Carolina", "Illinois")
    4) capacity: the stated indoor concert seating capacity (as text exactly as in the answer; do not convert to a number)
    5) is_indoor: confirm "indoor"/"outdoor"/or "unknown" exactly based on the answer claim
    6) wheelchair_accessible: "yes"/"no"/"unknown" based on the answer claim
    7) reference_urls: a list of URL(s) explicitly cited in the answer that can verify BOTH capacity and the facility type (indoor vs. outdoor). Prefer official venue sites or management companies; established venue directories are acceptable.
    8) ada_urls: a list of URL(s) explicitly cited in the answer that verify wheelchair-accessible seating (ADA) for the specific venue. If none are given, return an empty list.
    9) logistics_urls: a list of URL(s) explicitly cited in the answer that verify loading docks or load-in/out facilities for the venue. If none are given, return an empty list.
    10) green_room_urls: a list of URL(s) explicitly cited in the answer that verify green room or performers' dressing rooms. If none are given, return an empty list.

    Notes:
    - Only extract URLs that are explicitly present in the answer. If the answer provides URLs in markdown links, extract the raw URLs.
    - If the answer mentions more than one venue in a required state bucket, choose the first one that appears in the answer text for that state.
    - For the "additional" bucket, pick the first venue that is located in Washington (the state), North Carolina, or Illinois. Do NOT use Washington, D.C.
    - If any field is missing in the answer for a venue, return null (or an empty list for URL arrays) for that field.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # remove empties and deduplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _collect_all_sources(v: Optional[VenueEntry]) -> List[str]:
    if not v:
        return []
    merged = []
    for block in [v.reference_urls, v.ada_urls, v.logistics_urls, v.green_room_urls]:
        merged.extend(block or [])
    return _safe_urls(merged)


def _format_state_list(states: List[str]) -> str:
    if not states:
        return ""
    if len(states) == 1:
        return states[0]
    return ", ".join(states[:-1]) + ", or " + states[-1]


# --------------------------------------------------------------------------- #
# Verification logic per venue                                                #
# --------------------------------------------------------------------------- #
async def verify_one_venue(
    evaluator: Evaluator,
    parent_node,
    venue: Optional[VenueEntry],
    group_id: str,
    group_desc: str,
    location_leaf_id: str,
    capacity_leaf_id: str,
    indoor_leaf_id: str,
    ada_leaf_id: str,
    refurl_leaf_id: str,
    loading_leaf_id: str,
    green_room_leaf_id: str,
    allowed_states: List[str],
) -> None:
    """
    Build the sub-tree and run verifications for one venue group.
    We introduce two internal wrappers to achieve fair scoring:
    - Required_Checks: a non-critical node containing ONLY the critical leaves. It will yield 1.0 only if all required checks pass.
    - Optional_Features: a non-critical node containing non-critical leaves for extra credit.
    """
    # Group (parallel aggregation, non-critical across different venues)
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent_node,
        critical=False
    )

    # Wrappers to avoid optional leaves overshadowing mandatory checks in aggregation
    required_node = evaluator.add_parallel(
        id=f"{group_id}_Required_Checks",
        desc=f"{group_desc} - Required checks (must all pass)",
        parent=group_node,
        critical=False
    )
    optional_node = evaluator.add_parallel(
        id=f"{group_id}_Optional_Features",
        desc=f"{group_desc} - Optional features (extra credit)",
        parent=group_node,
        critical=False
    )

    # Prepare data
    name = (venue.name if venue else "") or ""
    city = (venue.city if venue else "") or ""
    extracted_state = (venue.state if venue else "") or ""
    ref_urls = _safe_urls(venue.reference_urls if venue else [])
    ada_urls = _safe_urls(venue.ada_urls if venue else [])
    logistics_urls = _safe_urls(venue.logistics_urls if venue else [])
    green_urls = _safe_urls(venue.green_room_urls if venue else [])
    all_sources = _collect_all_sources(venue)

    # 1) Location (Critical)
    loc_leaf = evaluator.add_leaf(
        id=location_leaf_id,
        desc="The venue is located in the specified required state for this group",
        parent=required_node,
        critical=True
    )

    if not all_sources:
        # No sources at all -> fail this critical location check
        loc_leaf.score = 0.0
        loc_leaf.status = "failed"
    else:
        if len(allowed_states) == 1:
            # Enforce that the page shows the venue (by name if available) is in that single state
            allowed_str = allowed_states[0]
            claim_loc = (
                f"The cited page(s) are about the venue '{name}' and indicate that it is located in the U.S. state of {allowed_str}."
            )
            add_ins_loc = (
                "Treat this as Correct if the page clearly shows the venue (by name if available) is in the specified state. "
                "Minor formatting or casing differences in the venue name are fine. "
                "If the page is unrelated, irrelevant, or does not mention location at all, return Incorrect."
            )
        else:
            allowed_str = _format_state_list(allowed_states)
            claim_loc = (
                f"The cited page(s) are about the venue '{name}' and indicate that it is located in one of these U.S. states: {allowed_str}."
            )
            add_ins_loc = (
                "Return Correct if the page clearly indicates the venue is in any one of these states. "
                "Do not accept Washington, D.C.; 'Washington' must mean the U.S. state."
            )

        await evaluator.verify(
            claim=claim_loc,
            node=loc_leaf,
            sources=all_sources,
            additional_instruction=add_ins_loc
        )

    # 2) Capacity range 15,000–25,000 (Critical)
    cap_leaf = evaluator.add_leaf(
        id=capacity_leaf_id,
        desc="The venue has an indoor concert seating capacity between 15,000 and 25,000",
        parent=required_node,
        critical=True
    )
    if not ref_urls:
        cap_leaf.score = 0.0
        cap_leaf.status = "failed"
    else:
        claim_cap = (
            f"On the cited official/reference page(s) for '{name}', the seating capacity for concerts or indoor events "
            f"is between 15,000 and 25,000 people, inclusive."
        )
        add_ins_cap = (
            "Prefer a specifically stated 'concert capacity' if available. If multiple capacities are listed (e.g., basketball, hockey, concert), "
            "use the 'concert' capacity. If only a general or maximum seated capacity is given and it clearly applies to the indoor arena configuration, "
            "it is acceptable. If the page lists a capacity clearly outside the 15,000–25,000 range or provides no capacity, return Incorrect."
        )
        await evaluator.verify(
            claim=claim_cap,
            node=cap_leaf,
            sources=ref_urls,
            additional_instruction=add_ins_cap
        )

    # 3) Indoor facility (Critical)
    indoor_leaf = evaluator.add_leaf(
        id=indoor_leaf_id,
        desc="The venue is an indoor arena or indoor facility (not an outdoor amphitheater or open-air stadium)",
        parent=required_node,
        critical=True
    )
    if not ref_urls:
        indoor_leaf.score = 0.0
        indoor_leaf.status = "failed"
    else:
        claim_indoor = (
            f"The cited page(s) for '{name}' indicate that the venue is an indoor facility (e.g., indoor arena, enclosed or domed), "
            f"and it is not an outdoor amphitheater or open-air stadium."
        )
        add_ins_indoor = (
            "Return Correct if the page clearly describes the venue as an indoor arena/facility, enclosed, or domed. "
            "If the venue is described as an outdoor amphitheater, open-air stadium, or similar, return Incorrect."
        )
        await evaluator.verify(
            claim=claim_indoor,
            node=indoor_leaf,
            sources=ref_urls,
            additional_instruction=add_ins_indoor
        )

    # 4) Wheelchair-accessible seating (Critical)
    ada_leaf = evaluator.add_leaf(
        id=ada_leaf_id,
        desc="The venue provides wheelchair-accessible seating in compliance with ADA requirements",
        parent=required_node,
        critical=True
    )
    ada_sources = ada_urls if ada_urls else all_sources
    if not ada_sources:
        ada_leaf.score = 0.0
        ada_leaf.status = "failed"
    else:
        claim_ada = (
            f"The cited page(s) for '{name}' indicate the venue provides wheelchair-accessible seating (ADA compliant or equivalent accessible seating policy)."
        )
        add_ins_ada = (
            "Accept terms like 'ADA seating', 'wheelchair accessible seating', 'accessibility information', or 'accessible services'. "
            "The evidence should clearly indicate that wheelchair-accessible seating is available to patrons."
        )
        await evaluator.verify(
            claim=claim_ada,
            node=ada_leaf,
            sources=ada_sources,
            additional_instruction=add_ins_ada
        )

    # 5) Reference URL confirms both capacity and facility type (Critical)
    ref_leaf = evaluator.add_leaf(
        id=refurl_leaf_id,
        desc="A reference URL from an official source verifies both capacity and indoor facility type",
        parent=required_node,
        critical=True
    )
    if not ref_urls:
        ref_leaf.score = 0.0
        ref_leaf.status = "failed"
    else:
        claim_ref = (
            f"At least one of the provided reference URLs is an official venue source (venue website or venue management company) "
            f"or an established venue directory, and explicitly provides BOTH the seating capacity and that the venue is an indoor arena/facility."
        )
        add_ins_ref = (
            "Evaluate each provided reference URL. Consider official venue sites, facility managers (e.g., ASM Global/AEG), or well-established venue directories. "
            "The page must include BOTH: (1) a seating capacity statement (preferably for concerts/indoor events) and (2) confirmation that the facility is indoor. "
            "If none of the URLs meet both criteria, return Incorrect."
        )
        await evaluator.verify(
            claim=claim_ref,
            node=ref_leaf,
            sources=ref_urls,
            additional_instruction=add_ins_ref
        )

    # 6) Optional: Loading dock (Non-critical)
    load_leaf = evaluator.add_leaf(
        id=loading_leaf_id,
        desc="The venue has loading dock facilities for equipment delivery",
        parent=optional_node,
        critical=False
    )
    load_sources = logistics_urls if logistics_urls else all_sources
    if not load_sources:
        # If no sources at all for this optional detail, mark as skipped to avoid misleading 'failed'
        load_leaf.score = 0.0
        load_leaf.status = "skipped"
    else:
        claim_load = (
            f"The cited page(s) for '{name}' indicate that the venue has a loading dock/loading bay or similar truck load-in/out facilities."
        )
        add_ins_load = (
            "Look for terms like 'loading dock', 'loading bay', 'truck dock', 'freight dock', or similar. "
            "Event production guides or technical specs pages often include this information."
        )
        await evaluator.verify(
            claim=claim_load,
            node=load_leaf,
            sources=load_sources,
            additional_instruction=add_ins_load
        )

    # 7) Optional: Green room / dressing rooms (Non-critical)
    green_leaf = evaluator.add_leaf(
        id=green_room_leaf_id,
        desc="The venue has green room or dressing room facilities for performers",
        parent=optional_node,
        critical=False
    )
    green_sources = green_urls if green_urls else all_sources
    if not green_sources:
        green_leaf.score = 0.0
        green_leaf.status = "skipped"
    else:
        claim_green = (
            f"The cited page(s) for '{name}' indicate that the venue provides green room(s) and/or performers' dressing rooms."
        )
        add_ins_green = (
            "Accept mentions of 'green room', 'star dressing room', 'dressing rooms', 'backstage rooms', or similar performer support spaces."
        )
        await evaluator.verify(
            claim=claim_green,
            node=green_leaf,
            sources=green_sources,
            additional_instruction=add_ins_green
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
    Evaluate an answer for selecting four indoor concert venues across specified U.S. states
    with capacities between 15,000 and 25,000 and ADA seating, with official reference URLs.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # venues are independent groups
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Add GT/requirements summary for transparency
    evaluator.add_ground_truth({
        "required_venues": [
            "California",
            "New York",
            "Texas",
            f"One of: {ADDITIONAL_ALLOWED_STATES}"
        ],
        "capacity_range_required": "15,000–25,000 (inclusive), indoor concert seating",
        "accessibility_required": "Wheelchair-accessible seating (ADA)"
    }, gt_type="requirements")

    # Build and verify each venue group
    await verify_one_venue(
        evaluator=evaluator,
        parent_node=root,
        venue=extracted.california,
        group_id="Venue_1_California",
        group_desc="A concert venue located in California meeting all specified requirements",
        location_leaf_id="California_Location",
        capacity_leaf_id="California_Capacity_Range",
        indoor_leaf_id="California_Indoor_Facility",
        ada_leaf_id="California_Wheelchair_Accessibility",
        refurl_leaf_id="California_Reference_URL",
        loading_leaf_id="California_Loading_Dock",
        green_room_leaf_id="California_Green_Room",
        allowed_states=["California"],
    )

    await verify_one_venue(
        evaluator=evaluator,
        parent_node=root,
        venue=extracted.new_york,
        group_id="Venue_2_New_York",
        group_desc="A concert venue located in New York meeting all specified requirements",
        location_leaf_id="New_York_Location",
        capacity_leaf_id="New_York_Capacity_Range",
        indoor_leaf_id="New_York_Indoor_Facility",
        ada_leaf_id="New_York_Wheelchair_Accessibility",
        refurl_leaf_id="New_York_Reference_URL",
        loading_leaf_id="New_York_Loading_Dock",
        green_room_leaf_id="New_York_Green_Room",
        allowed_states=["New York"],
    )

    await verify_one_venue(
        evaluator=evaluator,
        parent_node=root,
        venue=extracted.texas,
        group_id="Venue_3_Texas",
        group_desc="A concert venue located in Texas meeting all specified requirements",
        location_leaf_id="Texas_Location",
        capacity_leaf_id="Texas_Capacity_Range",
        indoor_leaf_id="Texas_Indoor_Facility",
        ada_leaf_id="Texas_Wheelchair_Accessibility",
        refurl_leaf_id="Texas_Reference_URL",
        loading_leaf_id="Texas_Loading_Dock",
        green_room_leaf_id="Texas_Green_Room",
        allowed_states=["Texas"],
    )

    await verify_one_venue(
        evaluator=evaluator,
        parent_node=root,
        venue=extracted.additional,
        group_id="Venue_4_Additional_State",
        group_desc="A concert venue located in Washington, North Carolina, or Illinois meeting all specified requirements",
        location_leaf_id="Additional_State_Location",
        capacity_leaf_id="Additional_State_Capacity_Range",
        indoor_leaf_id="Additional_State_Indoor_Facility",
        ada_leaf_id="Additional_State_Wheelchair_Accessibility",
        refurl_leaf_id="Additional_State_Reference_URL",
        loading_leaf_id="Additional_State_Loading_Dock",
        green_room_leaf_id="Additional_State_Green_Room",
        allowed_states=ADDITIONAL_ALLOWED_STATES,
    )

    return evaluator.get_summary()