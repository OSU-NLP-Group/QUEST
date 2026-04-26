import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "touring_venues_2026"
TASK_DESCRIPTION = """A major touring music production is planning its 2026 North American arena tour and needs to identify suitable venue options. The production company requires three specific venues across three different U.S. states that meet their technical, capacity, and accessibility requirements.

Identify three concert/performance venues (one venue per state, in three different U.S. states) that satisfy ALL of the following criteria:

1. Each venue must be located in a different U.S. state
2. Each venue must have a concert seating capacity between 15,000 and 23,500 people
3. Each venue must be capable of accommodating a touring stage with dimensions of at least 60 feet wide by 40 feet deep
4. Each venue must comply with ADA accessibility requirements by providing wheelchair-accessible seating for at least 1% of its total concert capacity
5. At least one of the three venues must be located in a state where Ariana Grande's 'The Eternal Sunshine Tour' has confirmed 2026 tour dates
6. At least one of the three venues must be located in a state where Bruce Springsteen's 2026 'Land of Hope and Dreams American Tour' has confirmed tour dates

For each venue, provide:
- Official venue name
- City and state location
- Concert seating capacity
- Confirmation that stage dimensions of 60ft x 40ft can be accommodated
- Confirmation of ADA-compliant wheelchair seating (at least 1% of capacity)
- A reference URL to the venue's official website or a reliable source confirming the specifications
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string to maximize compatibility (e.g., "18,000", "18k", "15,000-18,000")
    stage_dimensions_supported: Optional[bool] = None  # True if answer claims 60ft x 40ft can be accommodated
    ada_wheelchair_seating_confirmed: Optional[bool] = None  # True if answer claims >=1%
    reference_urls: List[str] = Field(default_factory=list)  # Official or reliable URLs supporting specs


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)
    ariana_urls: List[str] = Field(default_factory=list)  # URLs confirming Ariana Grande 2026 tour schedule
    bruce_urls: List[str] = Field(default_factory=list)   # URLs confirming Bruce Springsteen 2026 tour schedule


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    From the provided answer text, extract structured information for up to three distinct concert/performance venues and the referenced tour schedule URLs.

    For each venue mentioned (limit to the first three venues if more are listed), extract the following fields exactly as stated in the answer:
    - name: Official venue name (string)
    - city: City where the venue is located (string)
    - state: U.S. state where the venue is located (string, e.g., "CA" or "California"; prefer full state name if provided)
    - capacity: The concert seating capacity mentioned in the answer (string, keep any formatting, e.g., "18,000", "18k", "15,000–18,000")
    - stage_dimensions_supported: Return true if the answer explicitly confirms that the venue can accommodate a touring stage of at least 60 feet wide by 40 feet deep; return false if it explicitly cannot; return null if not mentioned
    - ada_wheelchair_seating_confirmed: Return true if the answer explicitly confirms the venue provides wheelchair-accessible seating for at least 1% of total concert capacity; return false if it explicitly cannot; return null if not mentioned
    - reference_urls: An array of URLs (official venue site or reliable sources) cited in the answer to support any of the venue specifications (capacity, stage, ADA). Extract the exact URLs; include all relevant ones. If no URLs are provided for a venue, return an empty array.

    Also extract:
    - ariana_urls: An array of URLs cited in the answer that confirm Ariana Grande's "The Eternal Sunshine Tour" 2026 dates/schedule (official site, ticketing, or reliable press)
    - bruce_urls: An array of URLs cited in the answer that confirm Bruce Springsteen's 2026 "Land of Hope and Dreams American Tour" dates/schedule (official site, ticketing, or reliable press)

    Important:
    - Do not invent any information. If a field is not mentioned, set it to null (for booleans) or empty string for text fields or empty array for URLs.
    - For URLs, extract the actual URLs. Accept plain URLs or markdown links; always return full URLs with http/https.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return url.startswith("http://") or url.startswith("https://")


def parse_capacity_to_int(capacity_str: Optional[str]) -> Optional[int]:
    """
    Attempt to parse a single numeric capacity value from a capacity string.
    Supports formats like "18,000", "18000", "18k", "18.5k", and ranges like "15,000–18,000" (uses the most relevant single number).
    Returns None if parsing fails.
    """
    if not capacity_str:
        return None

    s = capacity_str.lower().strip()

    # Handle "18k" or "18.5k"
    k_match = re.findall(r'(\d+(?:\.\d+)?)\s*k\b', s)
    if k_match:
        try:
            # Use the first occurrence interpreted in thousands
            val = float(k_match[0]) * 1000
            return int(round(val))
        except Exception:
            pass

    # Extract all comma or plain numbers (e.g., "18,000", "15000")
    nums = re.findall(r'\d{1,3}(?:,\d{3})+|\d+', s)
    if not nums:
        return None

    # Convert all numbers to ints (remove commas)
    candidates = []
    for n in nums:
        try:
            candidates.append(int(n.replace(",", "")))
        except Exception:
            continue

    if not candidates:
        return None

    # Heuristic:
    # - If a range appears, often the first number is the concert capacity (but pages vary).
    # - Prefer a value between realistic arena bounds (10k–30k) if available; else take the first.
    for c in candidates:
        if 10000 <= c <= 30000:
            return c

    return candidates[0]


def all_states_distinct(states: List[Optional[str]]) -> bool:
    cleaned = [s.strip() for s in states if s and s.strip()]
    if len(cleaned) < 3:
        return False
    return len(set(cleaned)) == 3


def venue_state_unique(state_i: Optional[str], other_states: List[Optional[str]]) -> bool:
    if not state_i or not state_i.strip():
        return False
    s = state_i.strip()
    others_clean = [x.strip() for x in other_states if x and x.strip()]
    return s not in others_clean


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    index: int,
    all_states: List[Optional[str]],
) -> None:
    """
    Build verification sub-tree and run checks for a single venue.
    """
    # Top-level node for this venue (non-critical, allows partial credit per venue)
    v_node = evaluator.add_parallel(
        id=f"venue_{index+1}",
        desc=f"{['First','Second','Third'][index]} venue identification and verification",
        parent=parent_node,
        critical=False,
    )

    # Reference URL existence (critical sibling to gate spec verifications)
    has_valid_url = any(is_valid_url(u) for u in (venue.reference_urls or []))
    evaluator.add_custom_node(
        result=has_valid_url,
        id=f"venue_{index+1}_url_reference",
        desc="Valid reference URL is provided that confirms the venue specifications (capacity, stage capability, and ADA compliance)",
        parent=v_node,
        critical=True,
    )

    # Basic info group (critical)
    basic_node = evaluator.add_parallel(
        id=f"venue_{index+1}_basic_info",
        desc=f"Provide official venue name, city, and state location for the {['first','second','third'][index]} venue",
        parent=v_node,
        critical=True,
    )

    # Name present (critical leaf)
    evaluator.add_custom_node(
        result=bool(venue.name and venue.name.strip()),
        id=f"venue_{index+1}_name",
        desc="Official venue name is provided",
        parent=basic_node,
        critical=True,
    )

    # Location present (critical leaf)
    evaluator.add_custom_node(
        result=bool(venue.city and venue.city.strip()) and bool(venue.state and venue.state.strip()),
        id=f"venue_{index+1}_location",
        desc="City and state location are provided",
        parent=basic_node,
        critical=True,
    )

    # Unique state check (critical leaf)
    other_states = [all_states[j] for j in range(3) if j != index]
    evaluator.add_custom_node(
        result=venue_state_unique(venue.state, other_states),
        id=f"venue_{index+1}_unique_state",
        desc="Venue is located in a state different from the other two venues",
        parent=basic_node,
        critical=True,
    )

    # Capacity group (critical)
    cap_node = evaluator.add_parallel(
        id=f"venue_{index+1}_capacity",
        desc="Concert seating capacity is stated and falls within the required 15,000 to 23,500 range",
        parent=v_node,
        critical=True,
    )

    # Capacity stated (critical leaf, verify by URLs)
    cap_stated_leaf = evaluator.add_leaf(
        id=f"venue_{index+1}_capacity_stated",
        desc="Concert capacity number is explicitly stated",
        parent=cap_node,
        critical=True,
    )
    capacity_text = venue.capacity or ""
    await evaluator.verify(
        claim=f"The concert seating capacity for the venue is stated as '{capacity_text}'.",
        node=cap_stated_leaf,
        sources=venue.reference_urls,
        additional_instruction=(
            "Check the venue's official or reliable source page for 'concert capacity' explicitly. "
            "If multiple capacities are listed (e.g., basketball vs. concert), focus on the concert seating capacity. "
            "Allow minor rounding differences (e.g., 18000 vs 18,000)."
        ),
    )

    # Capacity range check (critical leaf, custom numeric)
    parsed_cap = parse_capacity_to_int(venue.capacity)
    in_range = (parsed_cap is not None) and (15000 <= parsed_cap <= 23500)
    evaluator.add_custom_node(
        result=in_range,
        id=f"venue_{index+1}_capacity_range",
        desc="Stated capacity falls within the 15,000 to 23,500 range",
        parent=cap_node,
        critical=True,
    )

    # Stage capability (critical)
    stage_node = evaluator.add_parallel(
        id=f"venue_{index+1}_stage",
        desc="Venue capability to accommodate 60ft x 40ft touring stage is confirmed",
        parent=v_node,
        critical=True,
    )
    stage_leaf = evaluator.add_leaf(
        id=f"venue_{index+1}_stage_capability",
        desc="Venue is confirmed to accommodate standard touring stage dimensions of at least 60ft x 40ft",
        parent=stage_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The venue can accommodate a touring stage of at least 60 feet wide by 40 feet deep.",
        node=stage_leaf,
        sources=venue.reference_urls,
        additional_instruction=(
            "Look for production specs, stage diagrams, rigging guides, or tech specs confirming stage dimensions >= 60' width and >= 40' depth. "
            "Equivalent statements (e.g., 'minimum stage width 60ft', 'stage depth 42ft') should be considered sufficient."
        ),
    )

    # ADA compliance (critical)
    ada_node = evaluator.add_parallel(
        id=f"venue_{index+1}_ada",
        desc="ADA compliance with 1% wheelchair seating requirement is confirmed",
        parent=v_node,
        critical=True,
    )
    ada_leaf = evaluator.add_leaf(
        id=f"venue_{index+1}_ada_compliance",
        desc="Venue is confirmed to provide wheelchair-accessible seating for at least 1% of concert capacity",
        parent=ada_node,
        critical=True,
    )
    # Build claim considering capacity if available
    if parsed_cap is not None:
        one_percent = max(1, int(round(parsed_cap * 0.01)))
        ada_claim = (
            f"The venue provides wheelchair-accessible seating for at least {one_percent} seats "
            f"(>= 1% of concert capacity {parsed_cap})."
        )
    else:
        ada_claim = (
            "The venue provides wheelchair-accessible seating for at least 1% of its total concert capacity."
        )
    await evaluator.verify(
        claim=ada_claim,
        node=ada_leaf,
        sources=venue.reference_urls,
        additional_instruction=(
            "Verify ADA compliance or accessibility policy pages for explicit counts or statements indicating "
            "wheelchair-accessible seating meeting or exceeding 1% of total concert capacity. "
            "If exact counts are shown, compare to capacity. "
            "Official ADA/accessibility statements meeting this threshold are acceptable."
        ),
    )


async def verify_tour_requirements(
    evaluator: Evaluator,
    parent_node,
    venue_states: List[Optional[str]],
    ariana_urls: List[str],
    bruce_urls: List[str],
) -> None:
    """
    Verify cross-venue tour requirements for Ariana Grande and Bruce Springsteen.
    """
    # Critical parent: failing tour requirements fails the whole evaluation
    tour_node = evaluator.add_parallel(
        id="tour_requirements",
        desc="Verify at least one venue is in each specified touring artist's 2026 tour state",
        parent=parent_node,
        critical=True,
    )

    # Prepare state list for claims
    states_clean = [s.strip() for s in venue_states if s and s.strip()]
    states_str = ", ".join(states_clean) if states_clean else "N/A"

    # Ariana Grande block (critical)
    ariana_node = evaluator.add_parallel(
        id="ariana_grande_tour_state",
        desc="At least one venue is located in a state where Ariana Grande's 'The Eternal Sunshine Tour' has confirmed 2026 tour dates",
        parent=tour_node,
        critical=True,
    )

    ariana_url_leaf = evaluator.add_leaf(
        id="ariana_tour_url",
        desc="Reference URL provided confirms the 2026 Ariana Grande tour schedule includes dates in the identified venue's state",
        parent=ariana_node,
        critical=True,
    )
    await evaluator.verify(
        claim="These URLs are reliable sources displaying Ariana Grande's 'The Eternal Sunshine Tour' 2026 U.S. dates/schedule.",
        node=ariana_url_leaf,
        sources=ariana_urls,
        additional_instruction=(
            "Prefer official site, artist socials, major ticketing, or reputable press pages that explicitly list 2026 dates."
        ),
    )

    ariana_state_match_leaf = evaluator.add_leaf(
        id="ariana_state_match",
        desc="One of the three venues is confirmed to be in a state with Ariana Grande 2026 tour dates, verified by reference",
        parent=ariana_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Ariana Grande's 2026 tour schedule includes dates in at least one of the following states: {states_str}.",
        node=ariana_state_match_leaf,
        sources=ariana_urls,
        additional_instruction=(
            "Scan the schedule and confirm that at least one state among the three venue states appears in the 2026 tour dates."
        ),
    )

    # Bruce Springsteen block (critical)
    bruce_node = evaluator.add_parallel(
        id="bruce_springsteen_tour_state",
        desc="At least one venue is located in a state where Bruce Springsteen's 2026 'Land of Hope and Dreams American Tour' has confirmed tour dates",
        parent=tour_node,
        critical=True,
    )

    bruce_url_leaf = evaluator.add_leaf(
        id="bruce_tour_url",
        desc="Reference URL provided confirms the 2026 Bruce Springsteen tour schedule includes dates in the identified venue's state",
        parent=bruce_node,
        critical=True,
    )
    await evaluator.verify(
        claim="These URLs are reliable sources displaying Bruce Springsteen's 'Land of Hope and Dreams American Tour' 2026 U.S. dates/schedule.",
        node=bruce_url_leaf,
        sources=bruce_urls,
        additional_instruction=(
            "Prefer official site, artist socials, major ticketing, or reputable press pages that explicitly list 2026 dates."
        ),
    )

    bruce_state_match_leaf = evaluator.add_leaf(
        id="bruce_state_match",
        desc="One of the three venues is confirmed to be in a state with Bruce Springsteen 2026 tour dates, verified by reference",
        parent=bruce_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Bruce Springsteen's 2026 tour schedule includes dates in at least one of the following states: {states_str}.",
        node=bruce_state_match_leaf,
        sources=bruce_urls,
        additional_instruction=(
            "Scan the schedule and confirm that at least one state among the three venue states appears in the 2026 tour dates."
        ),
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
    Evaluate an answer for the 2026 touring venues selection task.
    """
    # Initialize evaluator (root is non-critical per framework; we add critical children to gate)
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

    # Extract venues and tour URLs from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_and_tour_urls",
    )

    # Normalize to exactly 3 venues
    venues: List[VenueItem] = list(extraction.venues or [])
    if len(venues) > 3:
        venues = venues[:3]
    while len(venues) < 3:
        venues.append(VenueItem())

    # Gather states for cross-venue checks
    venue_states = [v.state for v in venues]

    # Record custom info for debugging
    evaluator.add_custom_info(
        info={
            "extracted_states": venue_states,
            "venue_url_counts": [len(v.reference_urls or []) for v in venues],
            "ariana_urls_count": len(extraction.ariana_urls or []),
            "bruce_urls_count": len(extraction.bruce_urls or []),
        },
        info_type="extraction_stats",
        info_name="extraction_statistics",
    )

    # Build venue subtrees
    for i, venue in enumerate(venues):
        await verify_single_venue(evaluator, root, venue, i, venue_states)

    # Verify tour requirements (critical)
    await verify_tour_requirements(
        evaluator,
        root,
        venue_states,
        extraction.ariana_urls or [],
        extraction.bruce_urls or [],
    )

    # Return structured summary
    return evaluator.get_summary()