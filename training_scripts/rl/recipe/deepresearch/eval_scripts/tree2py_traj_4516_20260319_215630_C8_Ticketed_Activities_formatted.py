import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_concert_venues_2026"
TASK_DESCRIPTION = """
A touring music artist is planning a multi-city summer 2026 concert tour across the United States and needs to identify suitable indoor concert venues in four different states: California, Texas, Florida, and New York. For each state, identify one venue that meets all of the following requirements:

1. Seating Capacity: The venue must have a seating capacity between 15,000 and 25,000 people for concert events
2. ADA Accessibility Compliance: The venue must provide ADA-compliant accessible seating, accessible parking spaces, and accessible entrance doors
3. Technical Infrastructure: The venue must have professional-grade sound and lighting systems suitable for major concert productions
4. Loading Dock Facilities: The venue must have loading dock access for equipment load-in and load-out
5. Parking Capacity: The venue must provide adequate on-site or adjacent parking facilities
6. Backstage Facilities: The venue must provide dressing rooms and green room facilities for performers
7. Insurance Requirements: The venue must require proof of general liability insurance from event organizers
8. Geographic Location: The venue must be located in or near a major metropolitan area within the specified state

For each of the four venues identified, provide:
- The venue name
- The city and state
- The seating capacity for concerts
- A reference URL confirming the venue meets the requirements
"""

ROOT_DESC = "Identify exactly one qualifying indoor concert venue in each of CA, TX, FL, and NY, and provide required fields and supporting reference URL(s)."

STATE_MAP: Dict[str, str] = {
    "CA": "California",
    "TX": "Texas",
    "FL": "Florida",
    "NY": "New York",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    """One venue as presented in the answer."""
    state_code: Optional[str] = None          # e.g., "CA"
    state_name: Optional[str] = None          # e.g., "California"
    venue_name: Optional[str] = None
    city: Optional[str] = None
    capacity_concert: Optional[str] = None    # Keep as string to be lenient (could be '18,000-20,000', etc.)
    reference_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    """All venues parsed from the answer."""
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract every indoor concert venue mentioned in the answer that is proposed for California (CA), Texas (TX), Florida (FL), or New York (NY).
    For each venue, return an object with:
    - state_code: two-letter code (CA, TX, FL, NY) if provided or obviously implied; else null
    - state_name: full state name if provided or obviously implied; else null
    - venue_name: the venue's name as written in the answer
    - city: the city where the venue is located, as written in the answer
    - capacity_concert: the concert seating capacity value as written; if only a range or approximate number is given, include that exact text; else null
    - reference_urls: an array of all explicit URLs in the answer that are intended to substantiate this venue's suitability (e.g., venue's official page, production guide, specifications, rental/booking page, or other credible references)

    Rules:
    - Only include venues for CA, TX, FL, or NY.
    - If multiple venues are listed for a state, extract all of them separately (we will select the first one later).
    - Extract URLs exactly as they appear (plain URLs or markdown links). Ignore any non-URL citations.
    - Do not invent any information not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_state_values(code: Optional[str], name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    code_norm = (code or "").strip().upper() or None
    name_norm = (name or "").strip()
    if code_norm in STATE_MAP:
        return code_norm, STATE_MAP[code_norm]
    # Try to map by name if only name provided
    for abbr, full in STATE_MAP.items():
        if name_norm.lower() == full.lower():
            return abbr, full
    return code_norm, name_norm or None


def is_state_match(item: VenueItem, target_code: str, target_full: str) -> bool:
    code, name = normalize_state_values(item.state_code, item.state_name)
    return (code == target_code) or ((name or "").lower() == target_full.lower())


def pick_first_venue_for_state(extracted: VenuesExtraction, target_code: str, target_full: str) -> Tuple[Optional[VenueItem], int]:
    """Return (first_venue_item, count_for_state)."""
    matched = [v for v in extracted.venues if is_state_match(v, target_code, target_full)]
    return (matched[0] if matched else None), len(matched)


def nonempty_str(s: Optional[str]) -> bool:
    return bool(s is not None and str(s).strip() != "")


def valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Basic cleanup and filtering for plausible URLs
    out = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u2 = u.strip()
        if u2.startswith("http://") or u2.startswith("https://"):
            if u2 not in out:
                out.append(u2)
        else:
            # If missing protocol, prepend http:// as per framework special rule
            if "." in u2 and " " not in u2:
                fixed = "http://" + u2
                if fixed not in out:
                    out.append(fixed)
    return out


# --------------------------------------------------------------------------- #
# Verification builder per state                                              #
# --------------------------------------------------------------------------- #
async def verify_state_venue(
    evaluator: Evaluator,
    parent_node,
    state_code: str,
    state_full: str,
    extracted: VenuesExtraction,
) -> None:
    """
    Build the verification subtree for one state (e.g., CA).
    """
    prefix = state_code.lower()
    state_node = evaluator.add_parallel(
        id=f"{state_full.lower().replace(' ', '_')}_venue",
        desc=f"{state_full} venue requirements",
        parent=parent_node,
        critical=False,  # The root aggregates states non-critically (partial credit allowed)
    )

    # Select item and count
    selected_item, count_for_state = pick_first_venue_for_state(extracted, state_code, state_full)

    # Critical: exactly one venue identified
    evaluator.add_custom_node(
        result=(count_for_state == 1),
        id=f"{prefix}_exactly_one_venue",
        desc=f"Identifies exactly one venue in {state_full}",
        parent=state_node,
        critical=True,
    )

    # Critical: required fields present
    req_fields = evaluator.add_parallel(
        id=f"{prefix}_required_fields",
        desc=f"Provides required output fields for the {state_full} venue",
        parent=state_node,
        critical=True,
    )

    name_ok = nonempty_str(selected_item.venue_name) if selected_item else False
    city_ok = nonempty_str(selected_item.city) if selected_item else False
    state_present = False
    if selected_item:
        sc, sn = normalize_state_values(selected_item.state_code, selected_item.state_name)
        state_present = (sc == state_code) or ((sn or "").lower() == state_full.lower())

    capacity_ok = nonempty_str(selected_item.capacity_concert) if selected_item else False
    urls_ok = len(valid_urls(selected_item.reference_urls if selected_item else [])) > 0

    evaluator.add_custom_node(
        result=name_ok,
        id=f"{prefix}_venue_name",
        desc="Venue name is provided",
        parent=req_fields,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(city_ok and state_present),
        id=f"{prefix}_city_state",
        desc="City and state are provided",
        parent=req_fields,
        critical=True,
    )
    evaluator.add_custom_node(
        result=capacity_ok,
        id=f"{prefix}_concert_capacity_value",
        desc="Concert seating capacity value is provided",
        parent=req_fields,
        critical=True,
    )
    evaluator.add_custom_node(
        result=urls_ok,
        id=f"{prefix}_reference_url",
        desc="At least one reference URL is provided that substantiates the venue meets the stated requirements",
        parent=req_fields,
        critical=True,
    )

    # Critical: all constraints must be satisfied
    constraints = evaluator.add_parallel(
        id=f"{prefix}_constraints",
        desc=f"{state_full} venue satisfies all stated constraints",
        parent=state_node,
        critical=True,
    )

    # Prepare for verification
    venue_name = selected_item.venue_name if selected_item else ""
    city = selected_item.city if selected_item else ""
    sources = valid_urls(selected_item.reference_urls if selected_item else [])

    # Build all leaves first
    leaf_indoor = evaluator.add_leaf(
        id=f"{prefix}_indoor",
        desc="Venue is an indoor concert/arena venue",
        parent=constraints,
        critical=True,
    )
    leaf_capacity = evaluator.add_leaf(
        id=f"{prefix}_capacity_range",
        desc="Concert seating capacity is between 15,000 and 25,000",
        parent=constraints,
        critical=True,
    )
    leaf_metro = evaluator.add_leaf(
        id=f"{prefix}_major_metro",
        desc=f"Venue is located in or near a major metropolitan area within {state_full}",
        parent=constraints,
        critical=True,
    )
    leaf_acc_seating = evaluator.add_leaf(
        id=f"{prefix}_accessible_seating",
        desc="ADA-compliant accessible seating is provided and is at least 5% of total capacity",
        parent=constraints,
        critical=True,
    )
    leaf_acc_parking = evaluator.add_leaf(
        id=f"{prefix}_accessible_parking",
        desc="Accessible parking spaces are provided near entrances",
        parent=constraints,
        critical=True,
    )
    leaf_acc_doors = evaluator.add_leaf(
        id=f"{prefix}_accessible_doors",
        desc="Accessible entrance doors are provided and meet the minimum 32-inch width requirement",
        parent=constraints,
        critical=True,
    )
    leaf_sound = evaluator.add_leaf(
        id=f"{prefix}_sound",
        desc="Professional-grade sound system suitable for major concert productions is available",
        parent=constraints,
        critical=True,
    )
    leaf_lighting = evaluator.add_leaf(
        id=f"{prefix}_lighting",
        desc="Professional-grade lighting system suitable for major concert productions is available",
        parent=constraints,
        critical=True,
    )
    leaf_loading = evaluator.add_leaf(
        id=f"{prefix}_loading_dock",
        desc="Loading dock access for equipment load-in/load-out is available",
        parent=constraints,
        critical=True,
    )
    leaf_parking = evaluator.add_leaf(
        id=f"{prefix}_parking_adequate",
        desc="Adequate parking facilities (on-site or adjacent) are available",
        parent=constraints,
        critical=True,
    )
    leaf_dressing = evaluator.add_leaf(
        id=f"{prefix}_dressing_rooms",
        desc="Backstage dressing rooms for performers are available",
        parent=constraints,
        critical=True,
    )
    leaf_green = evaluator.add_leaf(
        id=f"{prefix}_green_room",
        desc="Green room facilities for performers are available",
        parent=constraints,
        critical=True,
    )
    leaf_insurance = evaluator.add_leaf(
        id=f"{prefix}_insurance",
        desc="Venue requires proof of general liability insurance from event organizers (typically $1 million minimum coverage)",
        parent=constraints,
        critical=True,
    )
    leaf_exits = evaluator.add_leaf(
        id=f"{prefix}_emergency_exits",
        desc="Venue has at least two emergency exit routes as required by fire safety codes",
        parent=constraints,
        critical=True,
    )

    # Prepare claims
    # Note: All constraint leaves will automatically consider critical sibling preconditions
    # (e.g., exactly_one_venue and required fields) and will be skipped if they fail.
    claims_and_sources = [
        (
            f"'{venue_name}' in {city}, {state_full} is an indoor arena or indoor concert venue (not an open-air amphitheater or outdoor stadium).",
            sources,
            leaf_indoor,
            "Confirm the venue is indoors (roofed/enclosed). Accept phrases like 'indoor arena', 'indoor venue', 'enclosed', or 'domed'. Reject amphitheaters or outdoor stadiums."
        ),
        (
            f"The concert seating capacity for '{venue_name}' is between 15,000 and 25,000 inclusive.",
            sources,
            leaf_capacity,
            "Use any official specs, production guides, or credible sources on the provided URLs. Prefer 'concert capacity' or 'end-stage capacity' if multiple capacities are listed."
        ),
        (
            f"'{venue_name}' is located in or near a major metropolitan area within {state_full}. The venue is in {city}.",
            sources,
            leaf_metro,
            "Use the provided URLs to confirm the city/metro context. Treat well-known large cities or clearly metro-adjacent venues as major metros. If the provided pages do not allow concluding 'major metro', consider the claim unsupported."
        ),
        (
            f"'{venue_name}' provides ADA-compliant accessible seating, and the amount of accessible seating is at least 5% of the total capacity.",
            sources,
            leaf_acc_seating,
            "Look specifically for explicit mention (numbers, ratios, or policy) that support at least 5% accessible seating; generic 'ADA seating available' without quantities is insufficient."
        ),
        (
            f"'{venue_name}' provides ADA-accessible parking spaces located near accessible entrances.",
            sources,
            leaf_acc_parking,
            "Look for phrases like 'accessible parking', 'ADA parking', 'van-accessible spaces', or parking guidance indicating proximity to entrances."
        ),
        (
            f"'{venue_name}' has accessible entrance doors meeting a minimum clear width of at least 32 inches.",
            sources,
            leaf_acc_doors,
            "Seek explicit doorway width specifications or ADA entrance door compliance statements indicating 32-inch minimum clear width."
        ),
        (
            f"'{venue_name}' offers a professional-grade sound system suitable for major concert productions.",
            sources,
            leaf_sound,
            "Look for production/technical specs indicating 'state-of-the-art' or professional concert sound reinforcement systems."
        ),
        (
            f"'{venue_name}' offers a professional-grade lighting system suitable for major concert productions.",
            sources,
            leaf_lighting,
            "Look for production/technical specs indicating concert-ready lighting systems (moving heads, rigging capacity, professional consoles)."
        ),
        (
            f"'{venue_name}' has loading dock access for equipment load-in and load-out.",
            sources,
            leaf_loading,
            "Evidence may include 'loading docks', 'truck bays', 'freight elevators', or a production/rental guide that describes loading access."
        ),
        (
            f"'{venue_name}' provides adequate parking facilities on-site or adjacent to the venue.",
            sources,
            leaf_parking,
            "Look for 'parking garage', 'on-site parking', 'adjacent parking', or clear parking guidance indicating capacity suitable for large events."
        ),
        (
            f"'{venue_name}' provides backstage dressing rooms for performers.",
            sources,
            leaf_dressing,
            "Confirm explicit 'dressing rooms', 'star rooms', or equivalent performer support spaces in production guides or venue specs."
        ),
        (
            f"'{venue_name}' provides a green room facility for performers.",
            sources,
            leaf_green,
            "Look for 'green room', 'artist lounge', or equivalent performer lounge space."
        ),
        (
            f"'{venue_name}' requires event organizers to provide proof of general liability insurance, typically with a minimum of $1,000,000 coverage.",
            sources,
            leaf_insurance,
            "Seek booking/rental policies or event guidelines mentioning COI (certificate of insurance) and minimum coverage amounts."
        ),
        (
            f"'{venue_name}' has at least two emergency exit routes as required by fire safety codes.",
            sources,
            leaf_exits,
            "Look for egress/evacuation plans, 'multiple exits', or statements satisfying code requirements that indicate at least two exit routes."
        ),
    ]

    # Execute verifications (parallelized)
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the multi-state concert venues task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Four states evaluated independently
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=ROOT_DESC,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured venues from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Add some custom info for debugging
    evaluator.add_custom_info(
        {
            "total_extracted": len(extracted.venues),
            "states_expected": list(STATE_MAP.keys()),
        },
        info_type="extraction_stats"
    )

    # Build verification for each state
    for abbr, full in STATE_MAP.items():
        await verify_state_venue(evaluator, root, abbr, full, extracted)

    # Return structured summary
    return evaluator.get_summary()