import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "conan_gray_wishbone_2026_na_venues"
TASK_DESCRIPTION = (
    "For Conan Gray's Wishbone World Tour 2026, identify four concert venues from the North American leg "
    "that meet all of the following criteria: (1) The venue must be a confirmed stop on the Wishbone World "
    "Tour 2026; (2) The performance date must fall between February 15, 2026 and March 15, 2026 (inclusive); "
    "(3) The venue must be located in a U.S. state that is east of the Mississippi River; (4) The venue's "
    "concert seating capacity must be at least 19,000. For each of the four venues, provide: venue name, "
    "city, state, exact performance date, concert seating capacity, and source URL(s) confirming the venue "
    "information, tour date, and capacity."
)

DATE_RANGE_START = "2026-02-15"
DATE_RANGE_END = "2026-03-15"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Accept full state name or 2-letter code
    performance_date: Optional[str] = None  # Prefer YYYY-MM-DD if available; otherwise as in answer
    concert_capacity: Optional[str] = None  # Keep as string to accommodate ranges/text
    source_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract every venue entry that the answer provides for Conan Gray's Wishbone World Tour 2026.
    For each venue, extract:
    - venue_name: The name of the venue (e.g., "Madison Square Garden").
    - city: The city of the venue (e.g., "New York").
    - state: The U.S. state for the venue (either full name like "New York" or 2-letter code like "NY").
    - performance_date: The exact date for the concert at that venue. Prefer ISO format "YYYY-MM-DD" if present; otherwise keep the exact date string given.
    - concert_capacity: The stated concert seating capacity (if a range or multiple values are given, extract the text as-is).
    - source_urls: A list of URL(s) provided in the answer that are intended to verify the venue, the tour stop/date, and/or the capacity.
    
    RULES:
    - Extract only what is explicitly present in the answer; do not invent or normalize beyond converting date to ISO format if the answer provides it unambiguously.
    - If any field is not present for a given venue, set it to null (or an empty list for source_urls).
    - Include all venues mentioned in the answer (the evaluation script will consider only the first four if more are provided).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_nonempty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def first_k(items: List[Any], k: int) -> List[Any]:
    return items[:k] if items else []


def make_venue_key(v: VenueItem) -> str:
    """Create a canonical key for distinctness checking: venue name + city + state (case-insensitive)."""
    name = (v.venue_name or "").strip().lower()
    city = (v.city or "").strip().lower()
    state = (v.state or "").strip().lower()
    return f"{name}||{city}||{state}"


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_venue(
    evaluator: Evaluator,
    root_parent,
    venue: VenueItem,
    idx_one_based: int
) -> None:
    """
    Build the verification subtree for a single venue (1-based index).
    """
    # Add the venue group node
    venue_node = evaluator.add_parallel(
        id=f"venue_{idx_one_based}",
        desc=f"{idx_one_based}st venue satisfies all constraints and reporting requirements."
             if idx_one_based == 1 else (
                 f"{idx_one_based}nd venue satisfies all constraints and reporting requirements."
                 if idx_one_based == 2 else (
                     f"{idx_one_based}rd venue satisfies all constraints and reporting requirements."
                     if idx_one_based == 3 else
                     f"{idx_one_based}th venue satisfies all constraints and reporting requirements."
                 )
             ),
        parent=root_parent,
        critical=False
    )

    # 1) Required fields present
    required_fields_ok = (
        is_nonempty_str(venue.venue_name) and
        is_nonempty_str(venue.city) and
        is_nonempty_str(venue.state) and
        is_nonempty_str(venue.performance_date) and
        is_nonempty_str(venue.concert_capacity)
    )
    evaluator.add_custom_node(
        result=required_fields_ok,
        id=f"venue_{idx_one_based}_required_fields",
        desc="Includes venue name, city, state, exact performance date, and concert seating capacity.",
        parent=venue_node,
        critical=True
    )

    # 2) Confirmed NA leg stop (verify via provided sources)
    stop_node = evaluator.add_leaf(
        id=f"venue_{idx_one_based}_confirmed_na_leg_stop",
        desc="Is a confirmed stop on the Wishbone World Tour 2026 and is part of the North American leg.",
        parent=venue_node,
        critical=True
    )
    claim_stop = (
        f"{venue.venue_name} in {venue.city}, {venue.state} is a confirmed stop on Conan Gray's Wishbone World Tour 2026 "
        f"(North American leg), with a scheduled performance on {venue.performance_date}."
    )
    await evaluator.verify(
        claim=claim_stop,
        node=stop_node,
        sources=venue.source_urls,
        additional_instruction=(
            "Verify that the page confirms both (1) the show belongs to Conan Gray's Wishbone World Tour 2026 and "
            "(2) the North American leg (U.S./Canada). The page should explicitly mention Conan Gray and the tour/year, "
            "and it should match the exact venue and city/state (and the date if provided)."
        )
    )

    # 3) Performance date in the required range (simple logical check)
    date_range_node = evaluator.add_leaf(
        id=f"venue_{idx_one_based}_date_in_range",
        desc="Performance date is between 2026-02-15 and 2026-03-15 inclusive.",
        parent=venue_node,
        critical=True
    )
    claim_date_range = (
        f"The performance date '{venue.performance_date}' falls between {DATE_RANGE_START} and {DATE_RANGE_END} inclusive."
    )
    await evaluator.verify(
        claim=claim_date_range,
        node=date_range_node,
        additional_instruction=(
            "If the date is not in ISO format, interpret it reasonably (e.g., 'Feb 28, 2026' or '2/28/26' "
            "should be understood as 2026-02-28)."
        )
    )

    # 4) State east of the Mississippi River
    east_state_node = evaluator.add_leaf(
        id=f"venue_{idx_one_based}_east_of_mississippi_us_state",
        desc="Located in a U.S. state east of the Mississippi River.",
        parent=venue_node,
        critical=True
    )
    claim_east = (
        f"The U.S. state '{venue.state}' is east of the Mississippi River."
    )
    await evaluator.verify(
        claim=claim_east,
        node=east_state_node,
        additional_instruction=(
            "Base this on general geographic knowledge of U.S. states relative to the Mississippi River. "
            "States like NY, NJ, PA, FL, GA, MA, etc. are east; states like CA, TX, CO are not. "
            "Border states (e.g., IL, KY, TN, MS, LA, MN, WI, MO, IA, AR) should be considered carefully—"
            "for the purpose of this task, if the state is generally categorized as east of the Mississippi, accept it."
        )
    )

    # 5) Capacity minimum (19,000+) — verify via sources
    capacity_min_node = evaluator.add_leaf(
        id=f"venue_{idx_one_based}_capacity_minimum",
        desc="Concert seating capacity is stated and is at least 19,000.",
        parent=venue_node,
        critical=True
    )
    claim_capacity = (
        f"The concert seating capacity for {venue.venue_name} is at least 19,000."
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_min_node,
        sources=venue.source_urls,
        additional_instruction=(
            "Prefer capacity information that explicitly references the concert seating configuration "
            "(not just sports configurations). If multiple capacities are provided (e.g., basketball vs. concert), "
            "use the concert capacity. If ranges or estimates are given, accept if they clearly meet or exceed 19,000."
        )
    )

    # 6) Sources verification as a sub-tree (parallel critical children)
    sources_parent = evaluator.add_parallel(
        id=f"venue_{idx_one_based}_sources",
        desc="Provides source URL(s) that verify (a) venue info, (b) tour stop and exact date, and (c) concert seating capacity.",
        parent=venue_node,
        critical=True
    )

    # 6.1) Sources present (critical)
    sources_present = evaluator.add_custom_node(
        result=(len(venue.source_urls) > 0),
        id=f"venue_{idx_one_based}_sources_present",
        desc="At least one source URL is provided for this venue.",
        parent=sources_parent,
        critical=True
    )

    # 6.2) Venue info supported (venue name + location)
    src_venue_info = evaluator.add_leaf(
        id=f"venue_{idx_one_based}_sources_venue_info",
        desc="Sources support the venue's name and location (city/state).",
        parent=sources_parent,
        critical=True
    )
    claim_src_venue = (
        f"At least one of these sources shows that the venue '{venue.venue_name}' is located in {venue.city}, {venue.state}."
    )
    await evaluator.verify(
        claim=claim_src_venue,
        node=src_venue_info,
        sources=venue.source_urls,
        additional_instruction=(
            "Verify that a single page clearly states the venue's proper name and its city/state. "
            "Allow minor naming variations (e.g., abbreviations or official suffixes)."
        )
    )

    # 6.3) Tour stop and exact date supported
    src_tour_date = evaluator.add_leaf(
        id=f"venue_{idx_one_based}_sources_tour_date",
        desc="Sources support that this venue is a stop on Conan Gray's Wishbone World Tour 2026 with the exact listed date.",
        parent=sources_parent,
        critical=True
    )
    claim_src_tour = (
        f"At least one of these sources confirms Conan Gray's Wishbone World Tour 2026 includes a show at {venue.venue_name} "
        f"in {venue.city}, {venue.state} on {venue.performance_date}."
    )
    await evaluator.verify(
        claim=claim_src_tour,
        node=src_tour_date,
        sources=venue.source_urls,
        additional_instruction=(
            "Confirm both the artist (Conan Gray), the tour (Wishbone World Tour 2026), the venue, "
            "and the exact date as listed. Single-page verification is required (i.e., one page containing all elements)."
        )
    )

    # 6.4) Capacity supported
    src_capacity = evaluator.add_leaf(
        id=f"venue_{idx_one_based}_sources_capacity",
        desc="Sources support the venue's concert seating capacity (or that it is >= 19,000).",
        parent=sources_parent,
        critical=True
    )
    claim_src_capacity = (
        f"At least one of these sources provides the concert seating capacity for {venue.venue_name} "
        f"(or states it is at least 19,000)."
    )
    await evaluator.verify(
        claim=claim_src_capacity,
        node=src_capacity,
        sources=venue.source_urls,
        additional_instruction=(
            "The page should explicitly state the venue capacity. If it lists multiple capacities, "
            "it must include a concert configuration value, or state a number clearly >= 19,000."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Conan Gray Wishbone World Tour 2026 venues (North America) task.
    """
    # Initialize evaluator (root is non-critical by design to avoid forcing all-children critical)
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

    # Extract the venues from the answer
    extracted: VenuesExtraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Filter to first 4 venues (padding later if needed)
    venues = first_k(extracted.venues, 4)

    # Add a check for four distinct venues among the first four provided
    # We follow the general evaluation reminder to accept answers with >4 venues by using the first four.
    distinct_ok = False
    if len(venues) >= 4:
        keys = [make_venue_key(v) for v in venues[:4]]
        distinct_ok = len(set(keys)) == 4

    evaluator.add_custom_node(
        result=(len(venues) >= 4 and distinct_ok),
        id="four_distinct_venues",
        desc="Provides four venues (first four considered) and all are distinct (no duplicates by venue name + city + state).",
        parent=root,
        critical=True
    )

    # Pad to exactly 4 for downstream checks
    while len(venues) < 4:
        venues.append(VenueItem())

    # Build each venue subtree
    for idx in range(4):
        await verify_venue(
            evaluator=evaluator,
            root_parent=root,
            venue=venues[idx],
            idx_one_based=idx + 1
        )

    # Return the evaluation summary
    return evaluator.get_summary()