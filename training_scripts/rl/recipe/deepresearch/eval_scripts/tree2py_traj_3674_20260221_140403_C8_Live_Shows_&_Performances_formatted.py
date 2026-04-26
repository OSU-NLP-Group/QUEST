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
TASK_ID = "multi_state_venues_2026"
TASK_DESCRIPTION = (
    "A national touring band is planning a 2026 concert tour and needs to identify suitable concert venues in four different US states: "
    "California, Texas, Illinois, and Florida. For each state, identify one concert venue that meets ALL of the following requirements:\n\n"
    "1. Location: The venue must be physically located within the specified state.\n"
    "2. Capacity: The venue must have a minimum seating capacity of at least 2,000 persons for concert configurations.\n"
    "3. Stage Specifications: The venue must have a stage with minimum dimensions of at least 16 feet deep and 20 feet wide (320 square feet minimum) "
    "to accommodate a 5-8 piece band with full equipment.\n"
    "4. Loading Access: The venue must have a loading dock with a doorway clearance of at least 8 feet tall and 10 feet wide for equipment delivery.\n"
    "5. ADA Accessibility Compliance:\n"
    "   - Wheelchair-accessible seating must equal at least 1% of the venue's total seating capacity\n"
    "   - Companion seats must be provided adjacent to each wheelchair space\n"
    "   - The venue must have ADA-compliant accessible restrooms\n"
    "6. Technical Infrastructure:\n"
    "   - Professional sound system adequate for the venue size\n"
    "   - At least one dedicated backstage dressing room for performers\n"
    "   - Functional HVAC system for climate control during performances\n\n"
    "For each of the four venues (one per state), provide the venue name, the city and state location, confirmation that it meets each of the above requirements, "
    "and a reference URL that verifies the venue's specifications."
)

STATE_CONFIGS = {
    "CA": {
        "state_name": "California",
        "node_id": "California_Venue",
        "node_desc": "A suitable concert venue identified in California",
        "prefix": "CA",
    },
    "TX": {
        "state_name": "Texas",
        "node_id": "Texas_Venue",
        "node_desc": "A suitable concert venue identified in Texas",
        "prefix": "TX",
    },
    "IL": {
        "state_name": "Illinois",
        "node_id": "Illinois_Venue",
        "node_desc": "A suitable concert venue identified in Illinois",
        "prefix": "IL",
    },
    "FL": {
        "state_name": "Florida",
        "node_id": "Florida_Venue",
        "node_desc": "A suitable concert venue identified in Florida",
        "prefix": "FL",
    },
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Venue(BaseModel):
    """Minimal venue info required to run verification."""
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class VenueSelection(BaseModel):
    """One venue per target state."""
    california: Optional[Venue] = None
    texas: Optional[Venue] = None
    illinois: Optional[Venue] = None
    florida: Optional[Venue] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract exactly one concert venue for each of the following US states from the provided answer text: California, Texas, Illinois, and Florida.
    For each state, extract the following fields:

    - name: The venue's name as stated in the answer.
    - city: The venue's city as stated in the answer (if provided).
    - state: The venue's state as stated in the answer (use the full state name if possible).
    - reference_urls: A list of one or more explicit URLs included in the answer that verify the venue's specifications or technical details.
                      Only include URLs that are explicitly present in the answer text; do not invent URLs. If multiple URLs are given for the venue, include them all.
                      If the answer provides no URL for a venue, return an empty list for that venue.

    Important:
    - If the answer provides multiple venues per state, extract only the first clearly identified venue for that state.
    - If a field is missing in the answer, set it to null (for name/city/state) or empty list (for reference_urls).
    - Ensure URLs are valid and complete (including protocol). If a URL is missing protocol, prepend http://.
    - Do not infer or add any information not present in the answer.

    Return a JSON object with fields: california, texas, illinois, florida. Each field should be an object with keys: name, city, state, reference_urls.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(venue: Optional[Venue]) -> List[str]:
    return venue.reference_urls if (venue and venue.reference_urls) else []


def _location_claim(venue: Optional[Venue], target_state: str) -> str:
    if venue and venue.city and venue.state:
        vn = venue.name or "the venue"
        return f"The venue named '{vn}' is located in {venue.city}, {venue.state}."
    return f"The referenced venue is physically located within {target_state}."


# --------------------------------------------------------------------------- #
# Verification functions for each state                                       #
# --------------------------------------------------------------------------- #
async def verify_state_venue(
    evaluator: Evaluator,
    parent_node,
    state_key: str,
    venue: Optional[Venue],
) -> None:
    cfg = STATE_CONFIGS[state_key]
    state_name = cfg["state_name"]
    prefix = cfg["prefix"]

    # Create the state node (non-critical; partial credit allowed per state)
    state_node = evaluator.add_parallel(
        id=cfg["node_id"],
        desc=cfg["node_desc"],
        parent=parent_node,
        critical=False
    )

    sources = _safe_sources(venue)

    # Optional gating: ensure at least one reference URL is provided
    evaluator.add_custom_node(
        result=(len(sources) > 0),
        id=f"{prefix}_Reference_URL_Provided",
        desc="At least one reference URL is provided for this venue in the answer",
        parent=state_node,
        critical=True
    )

    # 1) Location verification (critical)
    loc_node = evaluator.add_leaf(
        id=f"{prefix}_Location_Verification",
        desc=f"The venue is physically located in {state_name}",
        parent=state_node,
        critical=True
    )
    await evaluator.verify(
        claim=_location_claim(venue, state_name),
        node=loc_node,
        sources=sources,
        additional_instruction=(
            f"Verify from the referenced page(s) that the venue is in {state_name}. "
            "Use the address section or 'About/Contact/Visit' page content. Accept reasonable variants (e.g., state abbreviations like CA/TX/IL/FL)."
        )
    )

    # 2) Capacity and space specifications group (critical, parallel)
    cap_node = evaluator.add_parallel(
        id=f"{prefix}_Capacity_Requirements",
        desc="Venue capacity and space specifications",
        parent=state_node,
        critical=True
    )

    # 2.1) Minimum capacity >= 2,000 (critical)
    cap_leaf = evaluator.add_leaf(
        id=f"{prefix}_Minimum_Capacity",
        desc="Venue has minimum seating capacity of at least 2,000 persons",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue's seating capacity for concerts is at least 2,000 persons.",
        node=cap_leaf,
        sources=sources,
        additional_instruction=(
            "Check the page(s) for stated seating capacity, maximum occupancy, or concert capacity. "
            "If multiple configurations are listed, use the one relevant to seated concerts. "
            "If the capacity is not explicitly given, conclude 'not supported'."
        )
    )

    # 2.2) Stage dimensions >= 16 ft deep and 20 ft wide (critical)
    stage_leaf = evaluator.add_leaf(
        id=f"{prefix}_Stage_Dimensions",
        desc="Stage dimensions are at least 16 feet deep and 20 feet wide (320 sq ft minimum) suitable for a 5-8 piece band",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue has a stage at least 16 ft deep and 20 ft wide (≥320 sq ft).",
        node=stage_leaf,
        sources=sources,
        additional_instruction=(
            "Look for explicit stage dimensions. If metric units are used, convert approximately (1 ft ≈ 0.3048 m). "
            "If only stage area is given, verify that it is ≥ 320 sq ft. "
            "If dimensions/area are not explicitly provided, conclude 'not supported'."
        )
    )

    # 2.3) Loading dock doorway clearance ≥ 8 ft tall and ≥ 10 ft wide (critical)
    load_leaf = evaluator.add_leaf(
        id=f"{prefix}_Loading_Access",
        desc="Venue has loading dock with doorway clearance of at least 8 feet tall and 10 feet wide",
        parent=cap_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue has a loading dock with doorway clearance at least 8 ft in height and 10 ft in width.",
        node=load_leaf,
        sources=sources,
        additional_instruction=(
            "Look for 'loading dock', 'load-in', or 'delivery' information with doorway dimensions. "
            "If doorway clearances are not specified, conclude 'not supported'."
        )
    )

    # 3) ADA accessibility compliance group (critical, parallel)
    ada_node = evaluator.add_parallel(
        id=f"{prefix}_Accessibility_Compliance",
        desc="ADA accessibility requirements",
        parent=state_node,
        critical=True
    )

    # 3.1) Wheelchair seating ≥ 1% of capacity (critical)
    wc_leaf = evaluator.add_leaf(
        id=f"{prefix}_Wheelchair_Seating",
        desc="Wheelchair-accessible seating equals at least 1% of total venue capacity",
        parent=ada_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue provides wheelchair-accessible seating equal to at least 1% of the total capacity.",
        node=wc_leaf,
        sources=sources,
        additional_instruction=(
            "Seek explicit counts/percentages of wheelchair-accessible seating. "
            "General statements like 'ADA seating available' without quantity do NOT meet the 1% threshold requirement."
        )
    )

    # 3.2) Companion seats adjacent (critical)
    comp_leaf = evaluator.add_leaf(
        id=f"{prefix}_Companion_Seats",
        desc="Companion seats are provided adjacent to wheelchair spaces",
        parent=ada_node,
        critical=True
    )
    await evaluator.verify(
        claim="Companion seats are provided adjacent to each wheelchair space.",
        node=comp_leaf,
        sources=sources,
        additional_instruction=(
            "Look for ADA seating policies indicating companion seats adjacent to wheelchair spaces. "
            "If not explicitly stated, conclude 'not supported'."
        )
    )

    # 3.3) ADA-compliant accessible restrooms (critical)
    rr_leaf = evaluator.add_leaf(
        id=f"{prefix}_Accessible_Restrooms",
        desc="Venue has ADA-compliant accessible restrooms",
        parent=ada_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue has ADA-compliant accessible restrooms.",
        node=rr_leaf,
        sources=sources,
        additional_instruction=(
            "Look for accessibility statements about restrooms compliant with ADA. "
            "If not explicitly stated, conclude 'not supported'."
        )
    )

    # 4) Technical infrastructure group (critical, parallel)
    tech_node = evaluator.add_parallel(
        id=f"{prefix}_Technical_Infrastructure",
        desc="Technical and operational capabilities",
        parent=state_node,
        critical=True
    )

    # 4.1) Professional sound system adequate for venue size (critical)
    sound_leaf = evaluator.add_leaf(
        id=f"{prefix}_Sound_System",
        desc="Venue has professional sound system adequate for venue size",
        parent=tech_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue has a professional sound system adequate for its size.",
        node=sound_leaf,
        sources=sources,
        additional_instruction=(
            "Look for mentions of installed PA systems, mixing consoles, line arrays, or 'state-of-the-art sound'. "
            "Generic marketing without mention of professional sound may be insufficient."
        )
    )

    # 4.2) At least one dedicated backstage dressing room (critical)
    dress_leaf = evaluator.add_leaf(
        id=f"{prefix}_Dressing_Rooms",
        desc="Backstage includes at least one dedicated dressing room for performers",
        parent=tech_node,
        critical=True
    )
    await evaluator.verify(
        claim="The backstage includes at least one dedicated dressing room for performers.",
        node=dress_leaf,
        sources=sources,
        additional_instruction=(
            "Look for 'dressing room(s)', 'green room(s)', or dedicated performer spaces in backstage amenities."
        )
    )

    # 4.3) Functional HVAC system for climate control (critical)
    hvac_leaf = evaluator.add_leaf(
        id=f"{prefix}_Climate_Control",
        desc="Venue has functional HVAC system for climate control during performances",
        parent=tech_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue has a functional HVAC system for climate control during performances.",
        node=hvac_leaf,
        sources=sources,
        additional_instruction=(
            "Look for mentions of air conditioning, heating, climate control, or HVAC systems."
        )
    )

    # 5) Reference documentation page relevance (critical)
    ref_leaf = evaluator.add_leaf(
        id=f"{prefix}_Reference_Documentation",
        desc="Valid reference URL provided that confirms the venue's existence and verifiable specifications",
        parent=state_node,
        critical=True
    )
    venue_name = (venue.name if venue and venue.name else "the venue")
    await evaluator.verify(
        claim=f"The provided reference page(s) are specifically about {venue_name} and include verifiable venue specifications (e.g., capacity, stage, loading, ADA, or technical details).",
        node=ref_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the page is about the venue and contains specification-type information (not just generic marketing). "
            "If the URL is unrelated, invalid, or lacks specifications, conclude 'not supported'."
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
    Evaluate an answer for the multi-state venue selection task.
    """
    # Initialize evaluator with a parallel root (states evaluated independently)
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

    # Extract venues from the answer
    venues: VenueSelection = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenueSelection,
        extraction_name="venue_selection",
    )

    # Build the verification tree following the rubric
    # Create a top-level node to represent the rubric root (optional, for clarity)
    rubric_root = evaluator.add_parallel(
        id="Multi_State_Venue_Selection",
        desc="Evaluation of suitable concert venues identified across four different US states, each meeting specific technical, accessibility, safety, and operational requirements",
        parent=root,
        critical=False
    )

    # Verify each state (parallel)
    await verify_state_venue(evaluator, rubric_root, "CA", venues.california)
    await verify_state_venue(evaluator, rubric_root, "TX", venues.texas)
    await verify_state_venue(evaluator, rubric_root, "IL", venues.illinois)
    await verify_state_venue(evaluator, rubric_root, "FL", venues.florida)

    # Return structured evaluation summary
    return evaluator.get_summary()