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
TASK_ID = "us_concert_venues_4states"
TASK_DESCRIPTION = """Identify four major music concert venues in the United States, with exactly one venue from each of the following four states: California, Texas, Colorado, and New York. Each venue must meet all of the following specific requirements:

California Venue Requirements:
- Located in Los Angeles, California
- Outdoor amphitheater
- Seating capacity of approximately 17,000-18,000 people
- Major concert venue that hosts nationally touring artists

Texas Venue Requirements:
- Located in Austin, Texas
- Indoor arena
- Seating capacity of approximately 15,000 people
- Major concert venue that hosts nationally touring artists

Colorado Venue Requirements:
- Located in Morrison (Denver area), Colorado
- Outdoor amphitheater featuring distinctive natural rock formations
- Seating capacity of approximately 9,000-10,000 people
- Major concert venue that hosts nationally touring artists

New York Venue Requirements:
- Located in Manhattan, New York City, New York
- Indoor arena
- Concert seating capacity of approximately 19,000-20,000 people
- Major concert venue that hosts nationally touring artists

For each venue, provide: (1) the venue name, (2) specific location (city and state), (3) seating capacity, (4) venue type (indoor/outdoor, and additional characteristics), and (5) at least one reference URL that verifies this information.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string to allow ranges/approx text
    venue_type: Optional[str] = None  # e.g., "outdoor amphitheater", "indoor arena", etc.
    reference_urls: List[str] = Field(default_factory=list)  # URLs explicitly provided in the answer


class VenuesExtraction(BaseModel):
    california: Optional[VenueItem] = None
    texas: Optional[VenueItem] = None
    colorado: Optional[VenueItem] = None
    new_york: Optional[VenueItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
Extract from the answer exactly one venue for each of the following states: California, Texas, Colorado, and New York.
For each state, if a venue is presented in the answer, extract the following fields:
- name: venue name (string; return null if not provided)
- city: city as stated in the answer (string; return null if missing)
- state: state as stated in the answer (string; return null if missing)
- capacity: seating capacity as stated in the answer; keep the exact phrasing (e.g., "17,500", "approximately 17,000–18,000") (string; return null if missing)
- venue_type: the venue type/descriptor as stated (e.g., "outdoor amphitheater", "indoor arena", possibly with extra characteristics) (string; return null if missing)
- reference_urls: an array of one or more URLs explicitly provided in the answer that verify the venue info; if none are provided, return an empty array.

Return a JSON object with keys:
- california: VenueItem for the California venue (or null if none mentioned)
- texas: VenueItem for the Texas venue (or null if none mentioned)
- colorado: VenueItem for the Colorado venue (or null if none mentioned)
- new_york: VenueItem for the New York venue (or null if none mentioned)

Important:
- Only extract information explicitly present in the answer.
- For URLs: extract the actual URLs from the answer. If no URLs are provided for a venue, use an empty list for reference_urls.
"""


# --------------------------------------------------------------------------- #
# Constraints (used to form verification claims)                              #
# --------------------------------------------------------------------------- #
CONSTRAINTS = {
    "california": {
        "node_id": "venue_1_california",
        "node_desc": "California venue (Los Angeles) requirements satisfied and required fields provided.",
        "required_city": "Los Angeles",
        "required_state": "California",
        "capacity_claim_text": "between 17,000 and 18,000",
        "type_claim_text": "an outdoor amphitheater",
        "location_claim_text": "Los Angeles, California",
        "reference_claim_text": "name, location (Los Angeles, California), seating capacity approximately between 17,000 and 18,000, and that it is an outdoor amphitheater",
        "capacity_instruction": "Treat any capacity clearly within the 17,000–18,000 range as acceptable. Minor rounding differences are fine."
    },
    "texas": {
        "node_id": "venue_2_texas",
        "node_desc": "Texas venue (Austin) requirements satisfied and required fields provided.",
        "required_city": "Austin",
        "required_state": "Texas",
        "capacity_claim_text": "approximately 15,000",
        "type_claim_text": "an indoor arena",
        "location_claim_text": "Austin, Texas",
        "reference_claim_text": "name, location (Austin, Texas), seating capacity approximately 15,000, and that it is an indoor arena",
        "capacity_instruction": "Treat values around 15,000 as acceptable for 'approximately 15,000' (e.g., 14,500–15,500)."
    },
    "colorado": {
        "node_id": "venue_3_colorado",
        "node_desc": "Colorado venue (Morrison/Denver area) requirements satisfied and required fields provided.",
        "required_city": "Morrison",
        "required_state": "Colorado",
        "capacity_claim_text": "between 9,000 and 10,000",
        "type_claim_text": "an outdoor amphitheater featuring distinctive natural rock formations",
        "location_claim_text": "Morrison, Colorado (in the Denver metropolitan area)",
        "reference_claim_text": "name, location (Morrison, Colorado / Denver area), seating capacity approximately between 9,000 and 10,000, and that it is an outdoor amphitheater featuring distinctive natural rock formations",
        "capacity_instruction": "Treat any capacity clearly within the 9,000–10,000 range as acceptable. Minor rounding differences are fine."
    },
    "new_york": {
        "node_id": "venue_4_new_york",
        "node_desc": "New York venue (Manhattan, NYC) requirements satisfied and required fields provided.",
        "required_city": "Manhattan, New York City",
        "required_state": "New York",
        "capacity_claim_text": "between 19,000 and 20,000 for concerts",
        "type_claim_text": "an indoor arena",
        "location_claim_text": "Manhattan, New York City, New York",
        "reference_claim_text": "name, location (Manhattan, New York City, New York), concert seating capacity approximately between 19,000 and 20,000, and that it is an indoor arena",
        "capacity_instruction": "Treat concert seating figures in the 19,000–20,000 range as acceptable. Minor rounding differences are fine."
    }
}


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_state_venue(
    evaluator: Evaluator,
    parent_node,
    state_key: str,
    venue: Optional[VenueItem],
) -> None:
    """
    Build the verification subtree for a single state's venue and run verifications.
    """
    cfg = CONSTRAINTS[state_key]
    state_node = evaluator.add_parallel(
        id=cfg["node_id"],
        desc=cfg["node_desc"],
        parent=parent_node,
        critical=False
    )

    # Extracted fields (safe fallbacks)
    name = (venue.name or "").strip() if venue else ""
    city = (venue.city or "").strip() if venue else ""
    state = (venue.state or "").strip() if venue else ""
    capacity_text = (venue.capacity or "").strip() if venue else ""
    type_text = (venue.venue_type or "").strip() if venue else ""
    urls = venue.reference_urls if (venue and venue.reference_urls) else []

    # 1) Name provided (existence) - critical leaf as per rubric
    evaluator.add_custom_node(
        result=bool(name),
        id=f"{cfg['node_id'].replace('venue_', 'venue_')}_name",  # keep consistency
        desc="Venue name is provided.",
        parent=state_node,
        critical=True
    )

    # 2) Location verification
    loc_leaf = evaluator.add_leaf(
        id=f"{cfg['node_id']}_location",
        desc=f"Venue is located in {cfg['location_claim_text']}.",
        parent=state_node,
        critical=True
    )
    loc_claim = f"The venue '{name}' is located in {cfg['location_claim_text']}."
    loc_instruction = (
        "Verify the venue's location using the provided source(s). "
        "Allow reasonable synonyms/variants (e.g., 'LA' for Los Angeles; "
        "'Manhattan' within New York City; 'Denver area' to indicate Morrison proximity). "
        "If the page indicates the same location, consider it supported."
    )

    # 3) Capacity verification
    cap_leaf = evaluator.add_leaf(
        id=f"{cfg['node_id']}_capacity",
        desc=f"Venue seating capacity is {cfg['capacity_claim_text']}.",
        parent=state_node,
        critical=True
    )
    cap_claim = (
        f"The seating capacity of '{name}' is {cfg['capacity_claim_text']}."
    )
    if capacity_text:
        cap_claim = (
            f"The seating capacity of '{name}' (often stated as '{capacity_text}') "
            f"is {cfg['capacity_claim_text']}."
        )
    cap_instruction = (
        "Use the source(s) to confirm the capacity. Accept approximate or configuration-dependent numbers "
        "as long as they fall in the described range or around the described value. "
        + cfg["capacity_instruction"]
    )

    # 4) Type verification
    type_leaf = evaluator.add_leaf(
        id=f"{cfg['node_id']}_type",
        desc=f"Venue is {cfg['type_claim_text']}.",
        parent=state_node,
        critical=True
    )
    type_claim = f"The venue '{name}' is {cfg['type_claim_text']}."
    if type_text:
        type_claim = (
            f"The venue '{name}' is {cfg['type_claim_text']} (the answer describes it as '{type_text}')."
        )
    type_instruction = (
        "Confirm indoor/outdoor classification and any distinctive features using the sources. "
        "Allow phrasing variants (e.g., 'open-air' ~ 'outdoor'; 'arena' synonyms are acceptable). "
        "For Colorado, confirm the distinctive natural rock formations."
    )

    # 5) Major status verification
    major_leaf = evaluator.add_leaf(
        id=f"{cfg['node_id']}_major_status",
        desc="Venue is a major concert venue that hosts nationally touring artists.",
        parent=state_node,
        critical=True
    )
    major_claim = (
        f"The venue '{name}' is a major concert venue that hosts nationally touring artists."
    )
    major_instruction = (
        "Check if the venue commonly hosts nationally touring artists (e.g., listings of major acts, "
        "tour schedules, references to 'national tours'). Evidence can be from official venue pages, "
        "reputable media, or encyclopedic sources."
    )

    # 6) References verification (single leaf per rubric)
    ref_leaf = evaluator.add_leaf(
        id=f"{cfg['node_id']}_reference",
        desc=(
            "At least one official/reliable reference URL is provided that supports the stated name, "
            "location, capacity, and venue type."
        ),
        parent=state_node,
        critical=True
    )
    ref_claim = (
        f"At least one of these URLs is a reliable or official source and supports the venue's "
        f"{cfg['reference_claim_text']} for '{name}'."
    )
    ref_instruction = (
        "Reliability preference: official venue/site, government/tourism pages, or high-quality encyclopedic "
        "entries (e.g., Wikipedia). The URL must substantively support the claim details."
    )

    # Batch verify the 4 evidence-backed leaves (location, capacity, type, major status) plus references
    claims_and_sources = [
        (loc_claim, urls, loc_leaf, loc_instruction),
        (cap_claim, urls, cap_leaf, cap_instruction),
        (type_claim, urls, type_leaf, type_instruction),
        (major_claim, urls, major_leaf, major_instruction),
        (ref_claim, urls, ref_leaf, ref_instruction),
    ]
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
    Evaluate an answer for the four-state US concert venues task.
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

    # Extract structured venue info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Record constraints in the summary for transparency
    evaluator.add_ground_truth({
        "required_states": ["California", "Texas", "Colorado", "New York"],
        "constraints": {
            "California": {
                "location": "Los Angeles, California",
                "venue_type": "outdoor amphitheater",
                "capacity": "approximately 17,000–18,000",
                "major_status": "nationally touring artists"
            },
            "Texas": {
                "location": "Austin, Texas",
                "venue_type": "indoor arena",
                "capacity": "approximately 15,000",
                "major_status": "nationally touring artists"
            },
            "Colorado": {
                "location": "Morrison (Denver area), Colorado",
                "venue_type": "outdoor amphitheater with distinctive natural rock formations",
                "capacity": "approximately 9,000–10,000",
                "major_status": "nationally touring artists"
            },
            "New York": {
                "location": "Manhattan, New York City, New York",
                "venue_type": "indoor arena",
                "capacity": "approximately 19,000–20,000 (concert seating)",
                "major_status": "nationally touring artists"
            }
        }
    }, gt_type="requirements")

    # Build and run verifications per state
    await verify_state_venue(evaluator, root, "california", extracted.california)
    await verify_state_venue(evaluator, root, "texas", extracted.texas)
    await verify_state_venue(evaluator, root, "colorado", extracted.colorado)
    await verify_state_venue(evaluator, root, "new_york", extracted.new_york)

    return evaluator.get_summary()