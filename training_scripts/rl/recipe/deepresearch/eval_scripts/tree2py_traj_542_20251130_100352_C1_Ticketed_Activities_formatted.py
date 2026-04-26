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
TASK_ID = "zach_bryan_venue_20260314"
TASK_DESCRIPTION = (
    "I'm planning to attend Zach Bryan's concert on Saturday, March 14, 2026. "
    "Please provide the following information about the venue: (1) the name of the stadium or venue, "
    "(2) the standard seating capacity for concert events, and (3) the city and state where the venue is located."
)

TARGET_EVENT_DATE_TEXT = "Saturday, March 14, 2026"
TARGET_ARTIST = "Zach Bryan"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """
    Structured information extracted from the agent's answer for the requested concert.
    """
    # Core facts
    event_date: Optional[str] = None
    venue_name: Optional[str] = None
    capacity_concert_standard: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

    # URLs explicitly mentioned in the answer
    # Prefer official sources for verification (venue official site and artist/tour official site).
    event_specific_urls: List[str] = Field(default_factory=list)          # URLs that specifically show the event listing/details for this date
    official_venue_urls: List[str] = Field(default_factory=list)          # Official venue site URLs (home, event page, seating/capacity page, info page, etc.)
    official_tour_urls: List[str] = Field(default_factory=list)           # Official artist/tour URLs (e.g., artist site tour page)
    other_urls: List[str] = Field(default_factory=list)                   # Any other supporting URLs explicitly cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_facts() -> str:
    return """
Extract the venue facts for the Zach Bryan concert on Saturday, March 14, 2026, as presented in the answer text. Return a single JSON object with these fields:
- event_date: The date string the answer claims for the concert (e.g., "Saturday, March 14, 2026"). If multiple appear, pick the one tied to the Zach Bryan concert instance.
- venue_name: The stadium/arena/venue name the answer claims is hosting the concert.
- capacity_concert_standard: The venue's standard concert/event seating capacity that the answer states (not a sports-only configuration and not only a special/expandable maximum figure). Return the text exactly as stated (e.g., "approximately 65,000", "20,000 for concerts").
- city: The city of the venue.
- state: The state or state/region abbreviation of the venue (e.g., "TX" or "Texas", "DC", "CA", etc.).

Also extract the URLs explicitly cited in the answer, categorized as:
- event_specific_urls: URLs that specifically list or detail the Zach Bryan concert for this exact date (e.g., an official venue event page or the artist’s official tour date page for March 14, 2026).
- official_venue_urls: URLs on the official venue website where the above facts (name, location, capacity) could be verified (e.g., venue homepage, info page, seating/capacity page).
- official_tour_urls: URLs on the artist's official website or official tour page where the event (date & venue) is listed.
- other_urls: Any other URLs cited by the answer that are not clearly official venue or official tour pages (e.g., news articles, ticket marketplaces, blogs).

Rules:
- Extract ONLY what is explicitly provided in the answer. Do not invent URLs or facts.
- If a field is missing, set it to null. If a URL list is missing, return an empty list for that field.
- Accept URLs presented as plain links or in markdown [text](url) format — but extract the actual URL.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def unique_urls(*lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for url in lst or []:
            if isinstance(url, str):
                u = url.strip()
                if u and u not in seen:
                    seen.add(u)
                    out.append(u)
    return out


def format_city_state(city: Optional[str], state: Optional[str]) -> Optional[str]:
    if city and state:
        return f"{city.strip()}, {state.strip()}"
    return None


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_venue_tree(
    evaluator: Evaluator,
    venue_node_desc: str,
    extracted: VenueExtraction,
) -> None:
    """
    Build the verification tree based on the rubric and run verifications for each leaf.
    All nodes here are critical because the rubric marks them essential.
    """
    # Parent node representing the rubric root (critical parallel aggregation)
    venue_node = evaluator.add_parallel(
        id="Venue_Information",
        desc=venue_node_desc,
        parent=evaluator.root,
        critical=True,  # Critical parent means all children must also be critical
    )

    # Compose URL bundles for different checks
    all_urls = unique_urls(
        extracted.official_venue_urls,
        extracted.official_tour_urls,
        extracted.event_specific_urls,
        extracted.other_urls,
    )
    official_urls = unique_urls(
        extracted.official_venue_urls,
        extracted.official_tour_urls,
    )
    event_urls = unique_urls(extracted.event_specific_urls)

    # Pre-format strings
    venue_name = extracted.venue_name or ""
    capacity_text = extracted.capacity_concert_standard or ""
    loc_text = format_city_state(extracted.city, extracted.state) or ""
    event_date_text = extracted.event_date or TARGET_EVENT_DATE_TEXT  # Prefer agent's date mention; fall back to target

    # 1) Correct_Event_Match (Leaf, critical)
    node_event_match = evaluator.add_leaf(
        id="Correct_Event_Match",
        desc="The identified venue corresponds specifically to the Zach Bryan tour date scheduled for Saturday, March 14, 2026.",
        parent=venue_node,
        critical=True,
    )
    # If there are no event-specific URLs, we cannot verify — fail this critical check directly.
    if not event_urls:
        node_event_match.score = 0.0
        node_event_match.status = "failed"
    else:
        event_claim = (
            f"On this page, there is a listing for a {TARGET_ARTIST} concert scheduled on {TARGET_EVENT_DATE_TEXT}. "
            f"It should indicate that the event is at '{venue_name}'"
            + (f" in {loc_text}." if loc_text else ".")
        )
        await evaluator.verify(
            claim=event_claim,
            node=node_event_match,
            sources=event_urls,
            additional_instruction=(
                "Verify that the page explicitly lists an event for Zach Bryan on the specified date. "
                "Allow date format variations (e.g., '03/14/2026' or 'March 14, 2026'). "
                "The page should clearly identify the venue name and, ideally, city/state; "
                "minor formatting or abbreviation differences (e.g., 'St.' vs 'Saint') are acceptable."
            ),
        )

    # 2) Venue_Name (Leaf, critical)
    node_venue_name = evaluator.add_leaf(
        id="Venue_Name",
        desc="Provides the name of the stadium/venue hosting the concert.",
        parent=venue_node,
        critical=True,
    )
    venue_claim = f"The venue hosting the {TARGET_ARTIST} concert is named '{venue_name}'."
    await evaluator.verify(
        claim=venue_claim,
        node=node_venue_name,
        sources=all_urls if all_urls else None,  # fall back to simple verify if no URLs; other nodes will enforce sourcing
        additional_instruction=(
            "Check the page to confirm the venue name is as stated. "
            "Allow minor punctuation/casing variations and official naming variants."
        ),
    )

    # 3) Seating_Capacity_Concert_Standard (Leaf, critical)
    node_capacity = evaluator.add_leaf(
        id="Seating_Capacity_Concert_Standard",
        desc="Provides the venue's standard seating capacity for concerts/events (not sports-only or expandable maximum).",
        parent=venue_node,
        critical=True,
    )
    if capacity_text.strip():
        cap_claim = (
            f"The standard seating capacity for concerts/events at '{venue_name}' is '{capacity_text}'."
        )
        preferred_sources_for_capacity = official_urls if official_urls else all_urls
        await evaluator.verify(
            claim=cap_claim,
            node=node_capacity,
            sources=preferred_sources_for_capacity if preferred_sources_for_capacity else None,
            additional_instruction=(
                "Confirm the capacity is for concerts/events (not a sports-only configuration). "
                "If multiple capacities exist, ensure the claim matches the standard concert capacity or a clearly stated typical configuration. "
                "Allow reasonable numeric formatting differences (e.g., commas, approximations)."
            ),
        )
    else:
        # Missing capacity in the answer; fail this critical leaf
        node_capacity.score = 0.0
        node_capacity.status = "failed"

    # 4) Venue_Location_City_State (Leaf, critical)
    node_location = evaluator.add_leaf(
        id="Venue_Location_City_State",
        desc="Provides the venue location including both the city and state.",
        parent=venue_node,
        critical=True,
    )
    if loc_text:
        loc_claim = f"The venue is located in {loc_text}."
        await evaluator.verify(
            claim=loc_claim,
            node=node_location,
            sources=all_urls if all_urls else None,
            additional_instruction=(
                "Verify that the page shows the venue's city and state. "
                "Accept state abbreviations or full names (e.g., 'TX' vs 'Texas') and minor formatting variants."
            ),
        )
    else:
        node_location.score = 0.0
        node_location.status = "failed"

    # 5) Official_Sourcing_For_Venue_Facts (Leaf, critical)
    node_official_sources = evaluator.add_leaf(
        id="Official_Sourcing_For_Venue_Facts",
        desc="Venue facts (name, capacity, and location) are supported with citations/URLs from official venue and/or official tour websites so the information is verifiable.",
        parent=venue_node,
        critical=True,
    )
    if not official_urls:
        # No official venue or official tour URLs present — fail this critical sourcing requirement
        node_official_sources.score = 0.0
        node_official_sources.status = "failed"
    else:
        official_claim = (
            "This page is an official source (either the official venue website or the artist's official tour website) "
            "and it supports at least one of the required venue facts (venue name, concert capacity, city/state)."
        )
        await evaluator.verify(
            claim=official_claim,
            node=node_official_sources,
            sources=official_urls,
            additional_instruction=(
                "Determine whether the page is clearly official (e.g., venue's own domain or artist's official site) "
                "and whether it explicitly supports any of the venue facts (name, location, or concert capacity). "
                "If the page is not official or does not support any fact, mark as not supported."
            ),
        )

    # 6) Current_Information (Leaf, critical)
    node_current_info = evaluator.add_leaf(
        id="Current_Information",
        desc="Sources used are current (active official pages reflecting present information, not clearly outdated/archived pages).",
        parent=venue_node,
        critical=True,
    )
    # Consider event and official URLs as primary for "current" checks
    urls_for_currency = unique_urls(event_urls, official_urls)
    if not urls_for_currency:
        node_current_info.score = 0.0
        node_current_info.status = "failed"
    else:
        current_claim = (
            "This page is a current, active official page (not an archived snapshot) and reflects present venue/tour information."
        )
        await evaluator.verify(
            claim=current_claim,
            node=node_current_info,
            sources=urls_for_currency,
            additional_instruction=(
                "Treat pages from 'web.archive.org' or clearly archived snapshots as NOT current. "
                "Official venue or artist pages that appear active and up-to-date should be considered current."
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
    Evaluate an answer for the Zach Bryan 2026-03-14 venue facts task using the Mind2Web2 framework.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Single top-level rubric node under this root
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

    # Extract structured venue facts from the agent's answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_facts(),
        template_class=VenueExtraction,
        extraction_name="venue_facts_extraction",
    )

    # Optionally add ground-truth target info (date and artist as context)
    evaluator.add_ground_truth(
        {"target_artist": TARGET_ARTIST, "target_event_date": TARGET_EVENT_DATE_TEXT},
        gt_type="target_context",
    )

    # Build and verify rubric tree
    await build_and_verify_venue_tree(
        evaluator=evaluator,
        venue_node_desc=(
            "Provide venue name, standard concert/event seating capacity, and city/state for the Zach Bryan concert "
            "on Saturday, March 14, 2026, using official, current, verifiable sources."
        ),
        extracted=extracted,
    )

    # Return summary
    return evaluator.get_summary()