import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "accessible_arenas_2026"
TASK_DESCRIPTION = (
    "I am researching accessible entertainment venues across the United States for a report on ADA compliance in large-scale concert and sports arenas. "
    "Please identify four major indoor arenas, each located in a different U.S. state, that meet the following criteria:\n\n"
    "1. The venue must have a seating capacity between 18,000 and 21,000 for concerts or basketball events\n"
    "2. The venue must currently be operational and hosting events in 2026\n"
    "3. Each of the four venues must be in a different state\n\n"
    "For each venue, provide:\n"
    "- The official venue name\n"
    "- The city and state where it is located\n"
    "- The seating capacity for concerts or basketball\n"
    "- A link to the venue's official website\n\n"
    "Additionally, verify and document that each venue meets all of the following ADA accessibility requirements:\n"
    "- Wheelchair-accessible seating areas are available\n"
    "- Companion seats are provided adjacent to wheelchair spaces\n"
    "- Accessible parking with designated spaces is offered\n"
    "- At least one accessible entrance with step-free or ramped access exists\n"
    "- ADA-compliant restroom facilities are available\n"
    "- Provide a link to the venue's official accessibility information page"
)

TARGET_YEAR = 2026


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string for robustness (ranges, notes)
    official_website: Optional[str] = None
    accessibility_page_url: Optional[str] = None


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_venues() -> str:
    return (
        "Extract up to four indoor venues listed in the answer. For each venue, return these fields exactly as stated:\n"
        "1) name: official venue name\n"
        "2) city: city where the venue is located\n"
        "3) state: state where the venue is located\n"
        "4) capacity: the seating capacity number or text specifically for concerts or basketball (do not convert; copy exactly as written)\n"
        "5) official_website: the venue's official website URL\n"
        "6) accessibility_page_url: a URL to the venue's official accessibility information page if provided\n\n"
        "Output as a JSON object with a 'venues' array. Each element in 'venues' must contain the 6 fields above. "
        "If any field is missing in the answer for a venue, set it to null. "
        "Include the venues in the same order as they appear in the answer."
    )


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _normalize_state(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    txt = s.strip().upper().replace(".", "")
    txt = txt.replace("STATE OF ", "").strip()
    return txt


def _is_url(s: Optional[str]) -> bool:
    if not s:
        return False
    s = s.strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def _build_sources_list(*urls: Optional[str]) -> List[str]:
    return [u for u in urls if isinstance(u, str) and u.strip() != ""]


# -----------------------------------------------------------------------------
# Verification logic for one venue
# -----------------------------------------------------------------------------
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    venue_index: int,
    state_is_unique: bool,
) -> None:
    n = venue_index + 1
    v_node = evaluator.add_parallel(
        id=f"venue_{n}",
        desc=f"Venue #{n} meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # ----------------------- Basic Information & Ops Status (critical) -----------------------
    info_node = evaluator.add_parallel(
        id=f"venue_{n}_information",
        desc="Basic venue information and operational status",
        parent=v_node,
        critical=True
    )

    # a) Basic info existence (name, city, state)
    basic_info_ok = bool(venue and venue.name and venue.city and venue.state)
    evaluator.add_custom_node(
        result=basic_info_ok,
        id=f"venue_{n}_basic_info",
        desc="Venue name, city, and state are provided",
        parent=info_node,
        critical=True
    )

    # b) Official website URL presence
    official_present = _is_url(venue.official_website)
    evaluator.add_custom_node(
        result=official_present,
        id=f"venue_{n}_reference",
        desc="Official venue website URL is provided",
        parent=info_node,
        critical=True
    )

    # c) Capacity within 18,000–21,000; verify with official website if available
    cap_leaf = evaluator.add_leaf(
        id=f"venue_{n}_capacity",
        desc="Venue seating capacity for concerts or basketball is between 18,000 and 21,000",
        parent=info_node,
        critical=True
    )
    cap_claim = (
        f"The seating capacity for concerts or basketball at {venue.name or 'the venue'} is "
        f"{venue.capacity} and this value lies between 18,000 and 21,000 inclusive."
        if venue.capacity
        else f"The seating capacity for concerts or basketball at {venue.name or 'the venue'} lies between 18,000 and 21,000 inclusive."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=_build_sources_list(venue.official_website),
        additional_instruction=(
            "Confirm the commonly stated capacity for either concerts or basketball. "
            "If multiple capacities are listed (e.g., basketball vs. concerts), it is sufficient if at least one of those "
            "lies within 18,000–21,000. Allow minor formatting or rounding differences. "
            "If a range is provided, consider it valid if the typical or stated capacity used for concerts or basketball is within the range."
        )
    )

    # d) State uniqueness among the four venues
    evaluator.add_custom_node(
        result=bool(state_is_unique),
        id=f"venue_{n}_state_unique",
        desc="Venue is located in a state different from the other three venues",
        parent=info_node,
        critical=True
    )

    # e) Operational and hosting events in 2026 (verify using official site)
    ops_leaf = evaluator.add_leaf(
        id=f"venue_{n}_operational_2026",
        desc="Venue is currently operational and hosting events in 2026",
        parent=info_node,
        critical=True
    )
    ops_claim = (
        f"{venue.name or 'The venue'} is operational and hosting events in {TARGET_YEAR}."
    )
    await evaluator.verify(
        claim=ops_claim,
        node=ops_leaf,
        sources=_build_sources_list(venue.official_website),
        additional_instruction=(
            f"Look for evidence on the official site (e.g., events, tickets, schedule, calendar) indicating events in {TARGET_YEAR} "
            f"or the {TARGET_YEAR} season. If the events calendar or schedule pages clearly show any {TARGET_YEAR} dates, consider it supported."
        )
    )

    # ----------------------- Accessibility (critical) -----------------------
    acc_node = evaluator.add_parallel(
        id=f"venue_{n}_accessibility",
        desc="Venue meets all required ADA accessibility compliance features",
        parent=v_node,
        critical=True
    )

    # Accessibility reference presence
    acc_ref_present = _is_url(venue.accessibility_page_url)
    evaluator.add_custom_node(
        result=acc_ref_present,
        id=f"venue_{n}_accessibility_reference",
        desc="URL reference to venue's official accessibility information page",
        parent=acc_node,
        critical=True
    )

    # Build leaves for the five ADA requirements
    urls_for_access = _build_sources_list(venue.accessibility_page_url, venue.official_website)

    wc_leaf = evaluator.add_leaf(
        id=f"venue_{n}_wheelchair_seating",
        desc="Venue provides wheelchair-accessible seating areas",
        parent=acc_node,
        critical=True
    )
    comp_leaf = evaluator.add_leaf(
        id=f"venue_{n}_companion_seating",
        desc="Venue provides companion seats adjacent to wheelchair spaces",
        parent=acc_node,
        critical=True
    )
    park_leaf = evaluator.add_leaf(
        id=f"venue_{n}_accessible_parking",
        desc="Venue offers accessible parking with designated spaces",
        parent=acc_node,
        critical=True
    )
    ent_leaf = evaluator.add_leaf(
        id=f"venue_{n}_accessible_entrance",
        desc="Venue has at least one accessible entrance with step-free or ramped access",
        parent=acc_node,
        critical=True
    )
    rest_leaf = evaluator.add_leaf(
        id=f"venue_{n}_accessible_restrooms",
        desc="Venue provides ADA-compliant restroom facilities",
        parent=acc_node,
        critical=True
    )

    claims = [
        (
            f"{venue.name or 'The venue'} provides wheelchair-accessible seating areas.",
            urls_for_access,
            wc_leaf,
            "Look for terms like 'wheelchair accessible seating', 'ADA seating', 'wheelchair positions', or similar language on the official accessibility page."
        ),
        (
            f"{venue.name or 'The venue'} provides companion seats adjacent to wheelchair spaces.",
            urls_for_access,
            comp_leaf,
            "Look for 'companion seating', 'companion seats next to wheelchair spaces', or similar phrasing indicating adjacent companion seating."
        ),
        (
            f"{venue.name or 'The venue'} offers accessible parking with designated spaces.",
            urls_for_access,
            park_leaf,
            "Look for 'accessible parking', 'ADA parking', 'designated accessible spaces', or similar terminology."
        ),
        (
            f"{venue.name or 'The venue'} has at least one accessible entrance with step-free or ramped access.",
            urls_for_access,
            ent_leaf,
            "Look for 'accessible entrance', 'step-free entry', 'ramp', 'elevator access', 'no stairs', or similar indicators."
        ),
        (
            f"{venue.name or 'The venue'} provides ADA-compliant restroom facilities.",
            urls_for_access,
            rest_leaf,
            "Look for 'accessible restrooms', 'ADA-compliant restrooms', 'family/unisex accessible restrooms', or similar language."
        ),
    ]

    # Batch verify the five ADA requirements in parallel
    await evaluator.batch_verify(claims)


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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

    # NOTE: Although the provided JSON marks root as critical, doing so would force all children to be critical
    # due to framework constraints. We keep root non-critical to allow partial credit across venues.

    # Extract venues from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Normalize and select exactly four venues (truncate or pad)
    venues: List[VenueItem] = list(extracted.venues[:4])
    while len(venues) < 4:
        venues.append(VenueItem())

    # Compute uniqueness for states across the four
    norm_states = [_normalize_state(v.state) for v in venues]
    state_counts = {}
    for s in norm_states:
        if s:
            state_counts[s] = state_counts.get(s, 0) + 1

    unique_flags = []
    for s in norm_states:
        if not s:
            unique_flags.append(False)
        else:
            unique_flags.append(state_counts.get(s, 0) == 1)

    # Build verification tree per venue
    tasks = []
    for i in range(4):
        tasks.append(
            verify_single_venue(
                evaluator=evaluator,
                parent_node=root,
                venue=venues[i],
                venue_index=i,
                state_is_unique=unique_flags[i]
            )
        )
    await asyncio.gather(*tasks)

    return evaluator.get_summary()