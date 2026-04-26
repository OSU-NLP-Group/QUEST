import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "touring_venues_four_states"
TASK_DESCRIPTION = (
    "Identify four professional performing arts venues suitable for a touring theatrical production with a full orchestra, "
    "with exactly one venue from each of four different states from the following list: Ohio, Indiana, Michigan, Illinois, or Pennsylvania. "
    "Each venue must meet all of the following requirements:\n\n"
    "Capacity and Stage Configuration:\n"
    "- Seating capacity between 1,200 and 2,100 seats\n"
    "- Proscenium stage configuration\n"
    "- Proscenium opening width of at least 50 feet\n"
    "- Stage depth of at least 35 feet from the proscenium line to the back wall\n\n"
    "Orchestra Pit:\n"
    "- Must have an orchestra pit that can be lowered below stage level\n\n"
    "Accessibility:\n"
    "- Must meet ADA requirements for wheelchair accessible seating\n\n"
    "Technical Infrastructure:\n"
    "- Must have a theatrical fly system or counterweight rigging system for scenery\n\n"
    "Loading and Backstage:\n"
    "- Must have a loading dock with truck/vehicle access\n"
    "- Loading dock door height of at least 10 feet OR loading dock door width of at least 10 feet\n"
    "- Must have at least 4 separate dressing rooms for cast members\n\n"
    "For each venue, provide:\n"
    "1. The official name of the venue\n"
    "2. The complete physical address (street address, city, state, ZIP code)\n"
    "3. The state in which it is located\n"
    "4. The exact seating capacity\n"
    "5. The proscenium opening width (in feet)\n"
    "6. The stage depth from proscenium line to back wall (in feet)\n"
    "7. Confirmation that an orchestra pit is present\n"
    "8. The total number of dressing rooms\n"
    "9. The loading dock door dimensions (height x width in feet)\n"
    "10. A direct link to the venue's official website or technical specifications page that confirms these specifications"
)

ALLOWED_STATES_FULL = {"Ohio", "Indiana", "Michigan", "Illinois", "Pennsylvania"}
STATE_ABBR_TO_FULL = {
    "OH": "Ohio",
    "IN": "Indiana",
    "MI": "Michigan",
    "IL": "Illinois",
    "PA": "Pennsylvania",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueRecord(BaseModel):
    name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # May be full name or abbreviation (OH, IN, MI, IL, PA)
    zip_code: Optional[str] = None
    seating_capacity: Optional[str] = None  # Prefer string to handle ranges or text formats
    proscenium_stage: Optional[str] = None  # e.g., "proscenium"
    proscenium_width_ft: Optional[str] = None
    stage_depth_ft: Optional[str] = None
    orchestra_pit_present: Optional[str] = None  # e.g., "yes"/"no" or textual confirmation
    orchestra_pit_lowerable: Optional[str] = None  # e.g., "yes"/"no" or textual confirmation
    ada_wheelchair_access: Optional[str] = None  # e.g., "ADA compliant"
    fly_system_present: Optional[str] = None  # e.g., "fly system" or "counterweight rigging"
    loading_dock_truck_access: Optional[str] = None  # textual confirmation
    loading_dock_door_dimensions: Optional[str] = None  # "height x width in feet" textual representation
    dressing_room_count: Optional[str] = None
    doc_urls: List[str] = Field(default_factory=list)  # Official website or technical specs page URLs


class VenuesExtraction(BaseModel):
    venues: List[VenueRecord] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return (
        "Extract up to four professional performing arts venues from the answer. For each venue, return a JSON object with the following fields:\n"
        "- name: Official name of the venue\n"
        "- street_address: The street address\n"
        "- city: City name\n"
        "- state: State (can be full name or standard abbreviation: OH, IN, MI, IL, PA)\n"
        "- zip_code: ZIP code\n"
        "- seating_capacity: Exact seating capacity value if given (string)\n"
        "- proscenium_stage: Confirmation string indicating proscenium stage configuration if mentioned\n"
        "- proscenium_width_ft: Proscenium opening width in feet (string)\n"
        "- stage_depth_ft: Stage depth (proscenium line to back wall) in feet (string)\n"
        "- orchestra_pit_present: Confirmation string indicating an orchestra pit is present (string)\n"
        "- orchestra_pit_lowerable: Confirmation string indicating the orchestra pit can be lowered below stage level (string)\n"
        "- ada_wheelchair_access: Confirmation string indicating ADA wheelchair accessible seating (string)\n"
        "- fly_system_present: Confirmation string indicating a theatrical fly system or counterweight rigging system (string)\n"
        "- loading_dock_truck_access: Confirmation string indicating loading dock with truck/vehicle access (string)\n"
        "- loading_dock_door_dimensions: Loading dock door dimensions formatted as 'height x width' in feet (string) if provided\n"
        "- dressing_room_count: Total number of dressing rooms (string)\n"
        "- doc_urls: An array of the venue's official website or technical specifications page URLs that confirm these specs. Include actual URLs mentioned in the answer. If multiple are given, include them all.\n\n"
        "Important:\n"
        "• Extract only what is explicitly provided in the answer; do not invent.\n"
        "• If any field is missing for a venue, set it to null (or use empty array for doc_urls).\n"
        "• Return a JSON object: { \"venues\": [ ... up to 4 venue objects ... ] }."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())

def normalize_state(s: Optional[str]) -> Optional[str]:
    if not is_non_empty(s):
        return None
    raw = s.strip()
    up = raw.upper()
    # If already full allowed name
    full = raw.title()
    if full in ALLOWED_STATES_FULL:
        return full
    # Try abbreviation
    if up in STATE_ABBR_TO_FULL:
        return STATE_ABBR_TO_FULL[up]
    # Attempt common punctuation, e.g., "Pa." or "Ill."
    cleaned = re.sub(r"[^\w]", "", up)
    if cleaned in STATE_ABBR_TO_FULL:
        return STATE_ABBR_TO_FULL[cleaned]
    return full  # return title-cased as best effort

def parse_first_number(text: Optional[str]) -> Optional[float]:
    if not is_non_empty(text):
        return None
    # Find first number like 1,200 or 1500.5 or 50'
    m = re.search(r"(?<!\d)(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)", text)
    if not m:
        return None
    num_str = m.group(1).replace(",", "")
    try:
        return float(num_str)
    except Exception:
        return None

def collect_doc_urls(venue: VenueRecord) -> List[str]:
    # Filter to reasonable http/https URLs
    urls = []
    for u in venue.doc_urls or []:
        if is_non_empty(u):
            u2 = u.strip()
            if not u2.lower().startswith(("http://", "https://")):
                # The extractor will prepend http:// per framework rules only if asked,
                # but we keep the URL as-is here.
                pass
            urls.append(u2)
    return urls


# --------------------------------------------------------------------------- #
# Verification for one venue                                                  #
# --------------------------------------------------------------------------- #
async def verify_venue(evaluator: Evaluator, parent_node, venue: VenueRecord, idx: int) -> None:
    """
    Build verification sub-tree and run checks for one venue.
    """
    venue_node = evaluator.add_parallel(
        id=f"venue_{idx+1}",
        desc=f"Venue {idx+1} (must meet all venue requirements; evaluated independently for partial credit)",
        parent=parent_node,
        critical=False
    )

    # -------------------- Required Output Fields (Critical) --------------------
    req_fields = evaluator.add_parallel(
        id=f"venue_{idx+1}_required_fields",
        desc="All required fields are provided for this venue",
        parent=venue_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(venue.name),
        id=f"venue_{idx+1}_name_provided",
        desc="Official name of the venue is provided",
        parent=req_fields,
        critical=True
    )

    complete_address = all([
        is_non_empty(venue.street_address),
        is_non_empty(venue.city),
        is_non_empty(venue.state),
        is_non_empty(venue.zip_code)
    ])
    evaluator.add_custom_node(
        result=complete_address,
        id=f"venue_{idx+1}_address_provided",
        desc="Complete physical address is provided (street address, city, state, ZIP code)",
        parent=req_fields,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(venue.state),
        id=f"venue_{idx+1}_state_provided",
        desc="State in which the venue is located is explicitly provided",
        parent=req_fields,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(venue.seating_capacity),
        id=f"venue_{idx+1}_capacity_value_provided",
        desc="Exact seating capacity value is provided",
        parent=req_fields,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(venue.proscenium_width_ft),
        id=f"venue_{idx+1}_proscenium_width_value_provided",
        desc="Proscenium opening width (in feet) is provided",
        parent=req_fields,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(venue.stage_depth_ft),
        id=f"venue_{idx+1}_stage_depth_value_provided",
        desc="Stage depth from proscenium line to back wall (in feet) is provided",
        parent=req_fields,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(venue.orchestra_pit_present),
        id=f"venue_{idx+1}_orchestra_pit_confirmation_provided",
        desc="Confirmation that an orchestra pit is present is provided",
        parent=req_fields,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(venue.dressing_room_count),
        id=f"venue_{idx+1}_dressing_room_count_provided",
        desc="Total number of dressing rooms is provided",
        parent=req_fields,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_non_empty(venue.loading_dock_door_dimensions),
        id=f"venue_{idx+1}_loading_door_dimensions_provided",
        desc="Loading dock door dimensions are provided (height x width in feet)",
        parent=req_fields,
        critical=True
    )

    doc_urls = collect_doc_urls(venue)
    evaluator.add_custom_node(
        result=(len(doc_urls) > 0),
        id=f"venue_{idx+1}_documentation_link_provided",
        desc="A direct link to the venue's official website or technical specifications page is provided that confirms the required specifications",
        parent=req_fields,
        critical=True
    )

    # -------------------- Venue Constraints (Critical) ------------------------
    constraints = evaluator.add_parallel(
        id=f"venue_{idx+1}_constraints",
        desc="This venue satisfies all stated constraints",
        parent=venue_node,
        critical=True
    )

    # State in allowed list (custom check)
    normalized_state = normalize_state(venue.state)
    evaluator.add_custom_node(
        result=(normalized_state in ALLOWED_STATES_FULL) if normalized_state else False,
        id=f"venue_{idx+1}_state_in_allowed_list",
        desc="Venue is located in one of: Ohio, Indiana, Michigan, Illinois, Pennsylvania",
        parent=constraints,
        critical=True
    )

    # Capacity in range (verify by URL)
    cap_leaf = evaluator.add_leaf(
        id=f"venue_{idx+1}_capacity_in_range",
        desc="Seating capacity is between 1,200 and 2,100 seats (inclusive)",
        parent=constraints,
        critical=True
    )
    cap_val = venue.seating_capacity or "unknown"
    await evaluator.verify(
        claim=f"The venue's seating capacity is between 1,200 and 2,100 seats. Reported capacity: {cap_val}.",
        node=cap_leaf,
        sources=doc_urls,
        additional_instruction="Check the official or technical page(s) for listed capacity and confirm it is within 1,200–2,100 seats. Allow reasonable synonyms like 'seats'."
    )

    # Proscenium stage configuration (verify by URL)
    prosc_leaf = evaluator.add_leaf(
        id=f"venue_{idx+1}_proscenium_stage_configuration",
        desc="Venue has a proscenium stage configuration",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue has a proscenium stage configuration.",
        node=prosc_leaf,
        sources=doc_urls,
        additional_instruction="Look for terms like 'proscenium', 'proscenium arch', or explicit stage configuration notes on the official/specs page."
    )

    # Proscenium width >= 50 ft (verify by URL)
    width_leaf = evaluator.add_leaf(
        id=f"venue_{idx+1}_proscenium_width_min_50ft",
        desc="Proscenium opening width is at least 50 feet",
        parent=constraints,
        critical=True
    )
    width_val = venue.proscenium_width_ft or "unknown"
    await evaluator.verify(
        claim=f"The proscenium opening width is at least 50 feet. Reported width: {width_val} feet.",
        node=width_leaf,
        sources=doc_urls,
        additional_instruction="Confirm that the page lists a proscenium opening width ≥ 50 ft (allow unit variants like feet, ft, ')."
    )

    # Stage depth >= 35 ft (verify by URL)
    depth_leaf = evaluator.add_leaf(
        id=f"venue_{idx+1}_stage_depth_min_35ft",
        desc="Stage depth from proscenium line to back wall is at least 35 feet",
        parent=constraints,
        critical=True
    )
    depth_val = venue.stage_depth_ft or "unknown"
    await evaluator.verify(
        claim=f"The stage depth from proscenium line to back wall is at least 35 feet. Reported depth: {depth_val} feet.",
        node=depth_leaf,
        sources=doc_urls,
        additional_instruction="Confirm that the page lists stage depth ≥ 35 ft (allow unit variants)."
    )

    # Lowerable orchestra pit (verify by URL)
    pit_leaf = evaluator.add_leaf(
        id=f"venue_{idx+1}_lowerable_orchestra_pit",
        desc="Venue has an orchestra pit that can be lowered below stage level",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim="The venue has an orchestra pit that can be lowered below stage level (e.g., pit lift, hydraulic pit).",
        node=pit_leaf,
        sources=doc_urls,
        additional_instruction="Look for phrases such as 'orchestra pit lift', 'lowerable pit', 'pit elevator', or explicit mention that the pit can be lowered below stage level."
    )

    # ADA wheelchair accessible seating (verify by URL)
    ada_leaf = evaluator.add_leaf(
        id=f"venue_{idx+1}_ada_accessible_seating",
        desc="Venue meets ADA requirements for wheelchair accessible seating",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim="The venue meets ADA requirements for wheelchair accessible seating.",
        node=ada_leaf,
        sources=doc_urls,
        additional_instruction="Confirm ADA compliance or explicit mention of wheelchair accessible seating per ADA on official/site pages; allow synonyms like 'accessible seating'."
    )

    # Fly system or counterweight rigging (verify by URL)
    fly_leaf = evaluator.add_leaf(
        id=f"venue_{idx+1}_fly_or_counterweight_rigging",
        desc="Venue has a theatrical fly system or counterweight rigging system",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim="The venue has a theatrical fly system or a counterweight rigging system for scenery.",
        node=fly_leaf,
        sources=doc_urls,
        additional_instruction="Look for 'fly system', 'counterweight rigging', 'linesets', etc., on the technical specs or stage pages."
    )

    # Loading dock with truck/vehicle access (verify by URL)
    load_leaf = evaluator.add_leaf(
        id=f"venue_{idx+1}_loading_dock_truck_access",
        desc="Venue has a loading dock with truck/vehicle access",
        parent=constraints,
        critical=True
    )
    await evaluator.verify(
        claim="The venue has a loading dock with truck or vehicle access.",
        node=load_leaf,
        sources=doc_urls,
        additional_instruction="Confirm mention of a loading dock suitable for truck/vehicle access on the page(s)."
    )

    # Loading door size requirement (verify by URL)
    door_leaf = evaluator.add_leaf(
        id=f"venue_{idx+1}_loading_door_size_requirement",
        desc="Loading dock door has height ≥ 10 feet OR width ≥ 10 feet",
        parent=constraints,
        critical=True
    )
    dim_val = venue.loading_dock_door_dimensions or "unknown"
    await evaluator.verify(
        claim=f"The loading dock door satisfies at least one dimension ≥ 10 feet (height or width). Reported dimensions: {dim_val}.",
        node=door_leaf,
        sources=doc_urls,
        additional_instruction="Confirm at least one of the loading door dimensions (height or width) is ≥ 10 ft; look for numeric specs on the page."
    )

    # Dressing rooms >= 4 (verify by URL)
    dress_leaf = evaluator.add_leaf(
        id=f"venue_{idx+1}_dressing_rooms_min_4",
        desc="Venue has at least 4 separate dressing rooms for cast members",
        parent=constraints,
        critical=True
    )
    dr_count = venue.dressing_room_count or "unknown"
    await evaluator.verify(
        claim=f"The venue has at least 4 separate dressing rooms for cast members. Reported number: {dr_count}.",
        node=dress_leaf,
        sources=doc_urls,
        additional_instruction="Confirm count of dressing rooms on specs/backstage pages; allow mention like '4+ dressing rooms'."
    )


# --------------------------------------------------------------------------- #
# Distinct states check across venues                                         #
# --------------------------------------------------------------------------- #
def four_distinct_allowed_states(venues: List[VenueRecord]) -> Tuple[bool, List[str]]:
    """
    Check that the first four venues are each in distinct states from the allowed list.
    Returns (result, normalized_states_list_used).
    """
    if len(venues) < 4:
        return False, []

    norms = []
    for v in venues[:4]:
        ns = normalize_state(v.state)
        norms.append(ns if ns else "")

    # Must be all non-empty and all in allowed
    if any(not s for s in norms):
        return False, norms
    if not all(s in ALLOWED_STATES_FULL for s in norms):
        return False, norms
    # Must be distinct
    if len(set(norms)) != 4:
        return False, norms
    return True, norms


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
    Evaluate an answer for the touring theatrical venues task.
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
        default_model=model,
    )

    # Extract venues
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Keep only the first four venues; pad with empty if fewer
    venues: List[VenueRecord] = list(extracted.venues[:4])
    while len(venues) < 4:
        venues.append(VenueRecord())

    # Build per-venue subtrees
    for i, venue in enumerate(venues):
        await verify_venue(evaluator, root, venue, i)

    # Distinct states across four venues (Critical at root level)
    distinct_ok, normalized_states = four_distinct_allowed_states(venues)
    evaluator.add_custom_node(
        result=distinct_ok,
        id="distinct_states_across_venues",
        desc="All four venues are located in four different states (no two venues share the same state)",
        parent=root,
        critical=True
    )

    # Add custom info to summary (optional diagnostics)
    evaluator.add_custom_info(
        info={"normalized_states": normalized_states, "allowed_states": sorted(list(ALLOWED_STATES_FULL))},
        info_type="diagnostics",
        info_name="state_normalization"
    )

    return evaluator.get_summary()