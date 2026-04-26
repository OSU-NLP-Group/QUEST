import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "phantom_midwest_capacity_venue_2025_2026"
TASK_DESCRIPTION = (
    "Among the venues hosting The Phantom of the Opera during its 2025-2026 North American tour, identify the venue "
    "with the largest seating capacity that meets the following criteria: (1) the show must be performed at this venue "
    "between November 2025 and February 2026 (inclusive), and (2) the venue must be located in a Midwest United States "
    "state (Illinois, Indiana, Iowa, Kansas, Michigan, Minnesota, Missouri, Nebraska, North Dakota, Ohio, South Dakota, "
    "or Wisconsin). Provide the complete venue name, the city and state where it is located, and its exact seating capacity."
)

MIDWEST_STATES = [
    "Illinois", "Indiana", "Iowa", "Kansas", "Michigan", "Minnesota",
    "Missouri", "Nebraska", "North Dakota", "Ohio", "South Dakota", "Wisconsin"
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SelectedVenueExtraction(BaseModel):
    """
    Structured extraction of the selected venue from the agent's answer.
    """
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    seating_capacity: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_selected_venue() -> str:
    return (
        "From the provided answer, extract the single selected venue that the answer claims is the largest seating capacity "
        "among eligible Midwest venues for The Phantom of the Opera 2025–2026 North American tour.\n"
        "Return a JSON object with the following fields:\n"
        "1) venue_name: The complete venue name as stated.\n"
        "2) city: The city where the venue is located.\n"
        "3) state: The full state name (e.g., 'Illinois'). If an abbreviation is used in the answer (e.g., 'IL'), "
        "   normalize to the full state name when possible; otherwise return the abbreviation string.\n"
        "4) seating_capacity: The exact seating capacity as stated (string). Do not convert to a number; keep any commas or units.\n"
        "5) source_urls: An array of all URLs explicitly mentioned in the answer that support the venue's tour schedule, "
        "   dates, location, or seating capacity. Include any venue pages, ticketing pages, official tour pages, press releases, or news pages. "
        "   If no URLs are provided, return an empty array.\n"
        "If any field is missing in the answer, return null for that field (or an empty array for source_urls). "
        "Do not invent or infer any information beyond what the answer explicitly provides."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def sources_or_none(urls: List[str]) -> Optional[List[str]]:
    """Return None if list is empty to route verification to simple mode; otherwise the list."""
    return urls if urls else None


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    Evaluate an answer for selecting the largest-capacity eligible Midwest venue on
    The Phantom of the Opera 2025–2026 North American tour.
    """
    # 1) Initialize evaluator (root node is non-critical by framework design)
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

    # 2) Extract selected venue details and sources from the answer
    selection: SelectedVenueExtraction = await evaluator.extract(
        prompt=prompt_extract_selected_venue(),
        template_class=SelectedVenueExtraction,
        extraction_name="selected_venue",
    )

    # Add custom info for transparency
    evaluator.add_custom_info(
        info={
            "allowed_midwest_states": MIDWEST_STATES,
        },
        info_type="constraints",
        info_name="midwest_states_constraint",
    )

    # 3) Build the rubric verification tree
    # Create top-level critical node (parallel aggregation)
    correct_node = evaluator.add_parallel(
        id="CorrectVenueIdentification",
        desc="Answer identifies the correct venue meeting all constraints and provides all required venue details.",
        parent=root,
        critical=True,
    )

    # 3a) Required information provided (critical, parallel)
    required_info_node = evaluator.add_parallel(
        id="RequiredInformationProvided",
        desc="The answer provides all required output fields for the selected venue.",
        parent=correct_node,
        critical=True,
    )

    # Required fields existence checks (critical leaf/custom nodes)
    evaluator.add_custom_node(
        result=bool(selection.venue_name and selection.venue_name.strip()),
        id="VenueNameProvided",
        desc="The complete venue name is provided.",
        parent=required_info_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(selection.city and selection.city.strip()),
        id="CityProvided",
        desc="The city where the venue is located is provided.",
        parent=required_info_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(selection.state and selection.state.strip()),
        id="StateProvided",
        desc="The state where the venue is located is provided.",
        parent=required_info_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(selection.seating_capacity and selection.seating_capacity.strip()),
        id="ExactSeatingCapacityProvided",
        desc="The exact seating capacity of the venue is stated.",
        parent=required_info_node,
        critical=True,
    )

    # 3b) Venue selection criteria met (critical, sequential)
    criteria_node = evaluator.add_sequential(
        id="VenueSelectionCriteriaMet",
        desc="The selected venue meets all selection constraints (tour schedule, date window, Midwest location, and largest capacity among eligible venues).",
        parent=correct_node,
        critical=True,
    )

    # Eligibility criteria (critical, parallel)
    eligibility_node = evaluator.add_parallel(
        id="EligibilityCriteriaMet",
        desc="The selected venue meets the base eligibility filters (tour schedule + timeframe + Midwest state).",
        parent=criteria_node,
        critical=True,
    )

    # Leaf nodes for eligibility checks
    on_tour_node = evaluator.add_leaf(
        id="OnTourSchedule",
        desc="The venue is part of The Phantom of the Opera 2025-2026 North American tour schedule.",
        parent=eligibility_node,
        critical=True,
    )
    within_window_node = evaluator.add_leaf(
        id="WithinDateRange",
        desc="The venue hosts performances between November 2025 and February 2026 (inclusive).",
        parent=eligibility_node,
        critical=True,
    )
    midwest_loc_node = evaluator.add_leaf(
        id="LocatedInMidwestState",
        desc="The venue is located in one of the specified Midwest states (Illinois, Indiana, Iowa, Kansas, Michigan, Minnesota, Missouri, Nebraska, North Dakota, Ohio, South Dakota, Wisconsin).",
        parent=eligibility_node,
        critical=True,
    )

    # Prepare claims and sources
    venue_name = selection.venue_name or ""
    city = selection.city or ""
    state = selection.state or ""
    capacity = selection.seating_capacity or ""
    urls = selection.source_urls or []

    # Batch verify the three eligibility leaf nodes
    eligibility_claims: List[
        tuple[str, Optional[List[str]], Any, Optional[str]]
    ] = []

    # Claim: OnTourSchedule
    eligibility_claims.append((
        f"The Phantom of the Opera 2025–2026 North American tour schedule includes performances at {venue_name}.",
        sources_or_none(urls),
        on_tour_node,
        (
            "Use the provided sources (official tour schedule pages, venue announcements, ticketing pages, or trustworthy press releases). "
            "Confirm that the lineup explicitly includes this venue for the 2025–2026 tour. "
            "If the page references a different season/year or a different show, treat as not supported."
        ),
    ))

    # Claim: WithinDateRange (Nov 2025 – Feb 2026 inclusive)
    eligibility_claims.append((
        f"There is at least one performance of The Phantom of the Opera at {venue_name} scheduled between November 1, 2025 and February 28, 2026 (inclusive).",
        sources_or_none(urls),
        within_window_node,
        (
            "From the schedule/date evidence in the provided sources, verify that at least one listed performance at this venue falls between "
            "November 1, 2025 and February 28, 2026 inclusive. Consider both textual dates and any calendar images; "
            "if dates are ambiguous or outside this window, judge as not supported."
        ),
    ))

    # Claim: LocatedInMidwestState
    eligibility_claims.append((
        f"{venue_name} is located in {city}, {state}. The state must be one of: {', '.join(MIDWEST_STATES)}.",
        sources_or_none(urls),
        midwest_loc_node,
        (
            "Confirm the venue's city/state from the provided sources (venue website, event page, or ticketing page). "
            "Treat common state abbreviations (e.g., IL for Illinois, OH for Ohio) as equivalent to full names. "
            "Also check that the state is in the specified Midwest list."
        ),
    ))

    await evaluator.batch_verify(eligibility_claims)

    # Largest capacity among eligible venues (critical leaf)
    largest_node = evaluator.add_leaf(
        id="LargestCapacityAmongEligible",
        desc="Among all venues that satisfy the eligibility criteria, the selected venue has the largest seating capacity.",
        parent=criteria_node,
        critical=True,
    )

    largest_claim = (
        f"Among all eligible Midwest venues on The Phantom of the Opera 2025–2026 tour with performances between Nov 2025 and Feb 2026 inclusive, "
        f"{venue_name} has the largest seating capacity at {capacity}."
    )
    await evaluator.verify(
        claim=largest_claim,
        node=largest_node,
        sources=sources_or_none(urls),
        additional_instruction=(
            "Use the combined evidence from the provided URLs to determine capacities and eligibility (Midwest + date window) of all relevant venues. "
            "To mark as supported, you must find explicit confirmation or a defensible comparison that no other eligible venue in the specified states "
            "exceeds this venue's stated capacity. If you cannot establish the comparison across eligible venues using the provided sources, judge as not supported."
        ),
    )

    # 4) Return final summary
    return evaluator.get_summary()