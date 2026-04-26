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
TASK_ID = "concert_venues_2026"
TASK_DESCRIPTION = """
Identify four distinct major concert venues in the United States that are operational and hosting concerts in 2026, each representing a different venue capacity category: (1) a mid-sized outdoor amphitheater with capacity between 9,000-10,000 people, (2) a large indoor arena with capacity between 20,000-25,000 people, (3) a stadium venue with capacity of 65,000 or more people, and (4) an outdoor amphitheater with capacity between 15,000-20,000 people. Each venue must be in a different U.S. state. For each venue, provide its name, city and state location, exact capacity, venue type (amphitheater/arena/stadium), and a reference URL confirming these details.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    venue_type: Optional[str] = None  # e.g., "outdoor amphitheater", "indoor arena", "stadium"
    capacity: Optional[str] = None    # keep as string to handle ranges or formatted numbers
    city: Optional[str] = None
    state: Optional[str] = None       # full name or 2-letter code
    reference_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    mid_sized_amphitheater: Optional[VenueItem] = None           # 9,000–10,000 outdoor amphitheater
    large_arena: Optional[VenueItem] = None                      # 20,000–25,000 indoor arena
    stadium_venue: Optional[VenueItem] = None                    # 65,000+ stadium
    outdoor_amphitheater_large: Optional[VenueItem] = None       # 15,000–20,000 outdoor amphitheater


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract exactly four venues from the answer, mapping each to the required capacity/type category.
    For each category, extract the venue that the answer associates with that category (or the best match based on the answer content). 
    If the answer lists more than four venues, choose the best-matching one for each category in the order they appear. If fewer than four are available, leave the missing category as null.

    Categories (keys) and requirements:
    - mid_sized_amphitheater: outdoor amphitheater, capacity between 9,000 and 10,000 (inclusive)
    - large_arena: indoor arena, capacity between 20,000 and 25,000 (inclusive)
    - stadium_venue: stadium, capacity at least 65,000
    - outdoor_amphitheater_large: outdoor amphitheater, capacity between 15,000 and 20,000 (inclusive)

    For each category object, extract:
    - name: the venue name exactly as written in the answer
    - venue_type: the venue type exactly as written (e.g., "outdoor amphitheater", "indoor arena", "stadium")
    - capacity: the capacity value exactly as written in the answer (string; do not parse to a number)
    - city: the city name
    - state: the U.S. state (full name or two-letter postal abbreviation) 
    - reference_urls: an array of all URLs included in the answer that are cited for this venue (include official site, Wikipedia, ticketing, or news/event pages; omit clearly unrelated links)

    Rules:
    - Do not invent data. Use only what the answer provides. If any field is missing, set it to null (or empty array for reference_urls).
    - Only include URLs explicitly present in the answer text. Extract them into the 'reference_urls' array.
    - Ensure each category maps to at most one venue object.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _ensure_list(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


def _has_required_info(item: Optional[VenueItem]) -> bool:
    if not item:
        return False
    return bool(item.name and item.city and item.state and item.capacity and _ensure_list(item.reference_urls))


# --------------------------------------------------------------------------- #
# Verification for a single venue category                                    #
# --------------------------------------------------------------------------- #
async def verify_venue_category(
    evaluator: Evaluator,
    parent_node,
    *,
    node_id: str,
    node_desc: str,
    item: Optional[VenueItem],
    required_kind: str,           # "amphitheater" | "arena" | "stadium"
    require_outdoor: Optional[bool],  # True/False/None (None = don't check outdoor/indoor)
    capacity_min: Optional[int],  # inclusive lower bound or None
    capacity_max: Optional[int],  # inclusive upper bound or None
) -> None:
    """
    Build verification nodes for one venue category and run checks.
    All factual checks use URL grounding where appropriate; logical constraint checks use simple verification.
    """
    node = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=False
    )

    # Critical: Required information present (name, city, state, capacity, at least one URL)
    evaluator.add_custom_node(
        result=_has_required_info(item),
        id=f"{node_id}_required_info",
        desc="Required fields present: name, city, state, capacity, and at least one reference URL",
        parent=node,
        critical=True
    )

    # Prepare safe fields and URLs for downstream checks
    name = item.name if item and item.name else ""
    city = item.city if item and item.city else ""
    state = item.state if item and item.state else ""
    vtype = item.venue_type if item and item.venue_type else ""
    capacity_text = item.capacity if item and item.capacity else ""
    urls = _ensure_list(item.reference_urls)

    # Critical: The provided URLs correspond to the venue (basic venue-page match)
    title_url_node = evaluator.add_leaf(
        id=f"{node_id}_title_url_match",
        desc="Reference URL(s) correspond to the named venue",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least one of the provided webpages clearly corresponds to the concert venue named '{name}'.",
        node=title_url_node,
        sources=urls,
        additional_instruction="Accept the venue's official website, Wikipedia page, official ticketing pages, or reputable event listings as valid. The page should clearly name the venue."
    )

    # Critical: Location supported by URL(s)
    location_node = evaluator.add_leaf(
        id=f"{node_id}_location_supported",
        desc="Location (city and state) is supported by the reference URL(s)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{name}' is located in {city}, {state}, United States.",
        node=location_node,
        sources=urls,
        additional_instruction="Look for the venue address or location on the page. It should explicitly indicate the city and state. Accept common abbreviations (e.g., 'CA' for California)."
    )

    # Critical: State is a U.S. state (simple logical check)
    state_us_node = evaluator.add_leaf(
        id=f"{node_id}_state_is_us",
        desc="State value represents a valid U.S. state",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The token '{state}' represents a valid U.S. state (either full name or USPS two-letter code).",
        node=state_us_node,
        additional_instruction="This is a simple fact-check. USPS state codes like 'CA' or full names like 'California' should be treated as valid."
    )

    # Critical: Venue type supported by URL(s)
    kind_phrase = required_kind
    type_desc_parts = []
    if require_outdoor is True:
        type_desc_parts.append("outdoor")
    if require_outdoor is False:
        type_desc_parts.append("indoor")
    type_desc_parts.append(required_kind)
    type_desc = " ".join(type_desc_parts)

    type_node = evaluator.add_leaf(
        id=f"{node_id}_type_supported",
        desc="Venue type matches the required category and is supported by URL(s)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{name}' is a {type_desc}.",
        node=type_node,
        sources=urls,
        additional_instruction=(
            "Verify the venue classification on the page. "
            "Allow common synonyms and spelling variants (e.g., amphitheater/amphitheatre). "
            "For arenas, it is acceptable if the page just says 'arena' without saying 'indoor', since arenas are typically indoor. "
            "For amphitheaters, they are generally outdoor by definition; accept 'outdoor amphitheater' or equivalent phrasing."
        )
    )

    # Critical: Capacity constraints supported by URL(s)
    # Build capacity claim depending on bounds
    if capacity_min is not None and capacity_max is not None:
        cap_claim = (
            f"The venue '{name}' has a typical/seated concert capacity between {capacity_min:,} and {capacity_max:,} inclusive."
        )
        cap_instruction = (
            "Use the seating capacity suitable for concerts or general events. "
            "If multiple capacities are listed (e.g., expandable), confirm that at least one commonly cited capacity falls within the specified range. "
            "Allow minor rounding or approximation (e.g., 'about 10,000')."
        )
    elif capacity_min is not None:
        cap_claim = (
            f"The venue '{name}' has a typical/seated concert capacity of at least {capacity_min:,}."
        )
        cap_instruction = (
            "Use the seating capacity suitable for concerts or general events. "
            "If multiple capacities are listed for different configurations, accept if one common capacity is at least the threshold. "
            "Allow minor rounding or approximation."
        )
    else:
        # Fallback if configuration missing (shouldn't happen with our setup)
        cap_claim = f"The venue '{name}' has a clear capacity disclosed on the provided page(s)."
        cap_instruction = "Confirm that the venue's capacity is stated."

    capacity_node = evaluator.add_leaf(
        id=f"{node_id}_capacity_supported",
        desc="Capacity meets the category constraint and is supported by URL(s)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=cap_claim,
        node=capacity_node,
        sources=urls,
        additional_instruction=cap_instruction
    )

    # Critical: Operational and hosting concerts in 2026 supported by URL(s)
    op2026_node = evaluator.add_leaf(
        id=f"{node_id}_operational_2026",
        desc="Venue is operational and hosting concerts in 2026 (supported by URL evidence)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue '{name}' hosted or is scheduled to host concerts in 2026.",
        node=op2026_node,
        sources=urls,
        additional_instruction=(
            "Look for 2026 on event calendars, schedules, or announcements indicating concerts in the year 2026. "
            "Accept official venue calendars, reputable ticketing sites (e.g., Ticketmaster), or news posts that clearly show concerts in 2026. "
            "If no 2026 concert is shown or inferred from the provided page(s), the claim is not supported."
        )
    )

    # Non-critical: The provided capacity field exists (agent provided an exact capacity string)
    cap_field_node = evaluator.add_custom_node(
        result=bool(capacity_text.strip()) if capacity_text else False,
        id=f"{node_id}_capacity_field_present",
        desc="Answer provides an explicit capacity value (as text)",
        parent=node,
        critical=False
    )

    # Non-critical: The provided venue_type field exists (agent provided a venue type string)
    type_field_node = evaluator.add_custom_node(
        result=bool(vtype.strip()) if vtype else False,
        id=f"{node_id}_type_field_present",
        desc="Answer provides a venue type value (as text)",
        parent=node,
        critical=False
    )


# --------------------------------------------------------------------------- #
# State diversity check                                                       #
# --------------------------------------------------------------------------- #
def add_state_diversity_check(
    evaluator: Evaluator,
    parent_node,
    items: List[Optional[VenueItem]]
) -> None:
    states = []
    for it in items:
        if it and it.state:
            s = it.state.strip()
            if s:
                states.append(s.upper())

    unique_states = set(states)
    result = (len(states) == 4) and (len(unique_states) == 4)

    evaluator.add_custom_node(
        result=result,
        id="State_Diversity",
        desc="All four identified venues are located in different U.S. states, with no state repeated",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_info(
        info={"states_extracted": states, "unique_states": list(unique_states)},
        info_type="state_diversity_details",
        info_name="state_diversity_details"
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Concert Venue Research task.
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
        default_model=model
    )

    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Build verification subtrees for each category
    await verify_venue_category(
        evaluator,
        root,
        node_id="Mid_Sized_Amphitheater",
        node_desc="Mid-sized outdoor amphitheater (9,000–10,000 capacity), operational with concerts in 2026",
        item=extracted.mid_sized_amphitheater,
        required_kind="amphitheater",
        require_outdoor=True,
        capacity_min=9000,
        capacity_max=10000
    )

    await verify_venue_category(
        evaluator,
        root,
        node_id="Large_Arena",
        node_desc="Large indoor arena (20,000–25,000 capacity), operational with concerts in 2026",
        item=extracted.large_arena,
        required_kind="arena",
        require_outdoor=False,  # indoor
        capacity_min=20000,
        capacity_max=25000
    )

    await verify_venue_category(
        evaluator,
        root,
        node_id="Stadium_Venue",
        node_desc="Stadium venue (65,000+ capacity), operational with concerts in 2026",
        item=extracted.stadium_venue,
        required_kind="stadium",
        require_outdoor=None,  # not required to specify
        capacity_min=65000,
        capacity_max=None
    )

    await verify_venue_category(
        evaluator,
        root,
        node_id="Outdoor_Amphitheater_Large",
        node_desc="Outdoor amphitheater (15,000–20,000 capacity), operational with concerts in 2026",
        item=extracted.outdoor_amphitheater_large,
        required_kind="amphitheater",
        require_outdoor=True,
        capacity_min=15000,
        capacity_max=20000
    )

    # Add critical state diversity check
    add_state_diversity_check(
        evaluator,
        root,
        [
            extracted.mid_sized_amphitheater,
            extracted.large_arena,
            extracted.stadium_venue,
            extracted.outdoor_amphitheater_large
        ]
    )

    return evaluator.get_summary()