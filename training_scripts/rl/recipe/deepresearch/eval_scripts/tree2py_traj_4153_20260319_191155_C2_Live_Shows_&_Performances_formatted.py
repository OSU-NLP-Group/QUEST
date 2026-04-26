import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "shaun_cassidy_ne_venue_sept_2025"
TASK_DESCRIPTION = """
What is the name and location (city and state) of the first New England venue where Shaun Cassidy performed during his 'The Road to Us' tour in September 2025, given that the venue must have a seating capacity between 800 and 900 seats?
"""


# ----------------------------- Data Models --------------------------------- #
class SelectedVenue(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Prefer 2-letter code if present (e.g., MA)
    performance_date: Optional[str] = None  # As stated in the answer (e.g., "September 10, 2025")
    tour_name: Optional[str] = None  # As stated (e.g., "The Road to Us")
    capacity_text: Optional[str] = None  # e.g., "850 seats"
    capacity_number: Optional[str] = None  # just digits if present, e.g., "850"

    # URLs cited in the answer
    tour_schedule_urls: List[str] = Field(default_factory=list)   # Official tour schedule, artist site, etc.
    event_urls: List[str] = Field(default_factory=list)           # Specific venue show pages, Ticketmaster, etc.
    capacity_urls: List[str] = Field(default_factory=list)        # Venue pages/Wikipedia that state seating capacity
    other_urls: List[str] = Field(default_factory=list)           # Any other URLs mentioned


class AnswerExtraction(BaseModel):
    selected_venue: Optional[SelectedVenue] = None


# --------------------------- Extraction Prompt ------------------------------ #
def prompt_extract_selected_venue() -> str:
    return """
    Extract the specific venue the answer claims satisfies the task. Return a single 'selected_venue' object with:
    - name: The venue's name as written in the answer (e.g., "The Cabot", "Jane Pickens Theater", etc.)
    - city: The city of the venue
    - state: The state (use 2-letter code if provided; otherwise as written)
    - performance_date: The date the show took place at this venue, as stated in the answer (e.g., "September 10, 2025")
    - tour_name: The tour name as written (e.g., "The Road to Us")
    - capacity_text: The seating capacity description as written (e.g., "850 seats")
    - capacity_number: If the answer states a clear numeric seat count, extract just the digits (e.g., "850"); otherwise null

    Also extract all URLs from the answer split into categories:
    - tour_schedule_urls: Any URL(s) to official tour schedule or pages that comprehensively list the tour dates
    - event_urls: URL(s) specific to this venue's show (venue events page, ticketing page, etc.)
    - capacity_urls: URL(s) that state or substantiate the venue's seating capacity
    - other_urls: Any other URLs in the answer that are relevant

    If any fields are missing in the answer, set them to null or empty arrays accordingly.
    """


# --------------------------- Helper Utilities ------------------------------- #
NEW_ENGLAND_STATES = {"CT", "MA", "ME", "NH", "RI", "VT"}


def _unique_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in url_lists:
        for u in lst:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


# --------------------------- Verification Logic ----------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: AnswerExtraction) -> None:
    # Root child node: overall identification (critical)
    venue_ident_node = evaluator.add_parallel(
        id="venue_identification",
        desc="Correctly identify the specific New England venue meeting all criteria",
        parent=evaluator.root,
        critical=True,
    )

    sv = extracted.selected_venue or SelectedVenue()

    # Convenience URL bundles
    all_urls = _unique_urls(sv.tour_schedule_urls, sv.event_urls, sv.capacity_urls, sv.other_urls)
    schedule_pref_urls = sv.tour_schedule_urls if sv.tour_schedule_urls else all_urls

    # Optional but useful gate: ensure the answer provides a concrete venue and at least one source URL
    selection_provided = evaluator.add_custom_node(
        result=bool(sv.name and sv.city and sv.state),
        id="selection_provided",
        desc="Answer provides a concrete venue name and its city/state",
        parent=venue_ident_node,
        critical=True,
    )
    sources_present = evaluator.add_custom_node(
        result=len(all_urls) > 0,
        id="sources_present",
        desc="At least one supporting URL is cited for the selected venue/tour",
        parent=venue_ident_node,
        critical=True,
    )

    # 1) Tour verification leaf
    tour_leaf = evaluator.add_leaf(
        id="tour_verification",
        desc="Verify the identified venue is part of Shaun Cassidy's 'The Road to Us' 2025-2026 tour",
        parent=venue_ident_node,
        critical=True,
    )
    tour_claim_parts = []
    if sv.name and sv.city and sv.state:
        tour_claim_parts.append(
            f"The venue {sv.name} in {sv.city}, {sv.state} hosted a performance by Shaun Cassidy "
            f"as part of his 'The Road to Us' 2025–2026 tour."
        )
    else:
        tour_claim_parts.append(
            "The identified venue is part of Shaun Cassidy's 'The Road to Us' 2025–2026 tour."
        )
    if sv.performance_date:
        tour_claim_parts.append(f"The performance occurred in September 2025 (claimed date: {sv.performance_date}).")
    else:
        tour_claim_parts.append("The performance occurred in September 2025.")

    tour_claim = " ".join(tour_claim_parts)

    await evaluator.verify(
        claim=tour_claim,
        node=tour_leaf,
        sources=schedule_pref_urls,
        additional_instruction=(
            "Focus on whether the provided URL(s) explicitly indicate the stop is part of Shaun Cassidy's "
            "'The Road to Us' tour and that the show date falls within September 2025. "
            "Allow minor punctuation/hyphen variations in the tour title. If the URL(s) do not clearly "
            "show the tour name or the date month/year, treat the claim as not supported."
        ),
    )

    # 2) Venue criteria aggregate (critical parallel)
    criteria_node = evaluator.add_parallel(
        id="venue_criteria",
        desc="Verify the venue satisfies all specified constraints",
        parent=venue_ident_node,
        critical=True,
    )

    # 2.a) Geographic + Capacity (leaf)
    geo_cap_leaf = evaluator.add_leaf(
        id="geographic_and_capacity",
        desc="The venue is located in a New England state and has a seating capacity between 800-900 seats (inclusive)",
        parent=criteria_node,
        critical=True,
    )

    if sv.name and sv.city and sv.state:
        geo_cap_claim = (
            f"The venue {sv.name} is located in {sv.city}, {sv.state}, where the state is one of the "
            f"New England states (CT, MA, ME, NH, RI, VT), and the venue's seating capacity is between "
            f"800 and 900 seats inclusive."
        )
    else:
        geo_cap_claim = (
            "The identified venue is located in a New England state (CT, MA, ME, NH, RI, VT) and has a "
            "seating capacity between 800 and 900 seats inclusive."
        )

    await evaluator.verify(
        claim=geo_cap_claim,
        node=geo_cap_leaf,
        sources=_unique_urls(sv.capacity_urls, sv.event_urls, sv.other_urls),
        additional_instruction=(
            "Verify two things: (1) the venue's state is one of CT, MA, ME, NH, RI, VT; "
            "(2) the venue's seated capacity (main auditorium/theater) is within 800–900 inclusive. "
            "If multiple capacity figures are shown, prefer the primary 'seated capacity' for the main room. "
            "Treat approximations like 'about 850' as valid if they reasonably indicate a value within the range. "
            "If either the New England location or the 800–900 capacity cannot be confirmed from the provided URL(s), "
            "mark as not supported."
        ),
    )

    # 2.b) Temporal Priority (leaf)
    temporal_leaf = evaluator.add_leaf(
        id="temporal_priority",
        desc="The performance occurs in September 2025 and is the earliest date among all venues meeting the geographic and capacity criteria",
        parent=criteria_node,
        critical=True,
    )

    if sv.name and sv.performance_date:
        temporal_claim = (
            f"Within September 2025, among all Shaun Cassidy 'The Road to Us' tour stops located in New England "
            f"with venue seating capacity between 800 and 900 seats inclusive, the performance at {sv.name} "
            f"on {sv.performance_date} is the earliest such date."
        )
    else:
        temporal_claim = (
            "Within September 2025, among all Shaun Cassidy 'The Road to Us' tour stops located in New England "
            "with venue seating capacity between 800 and 900 seats inclusive, the selected venue's performance "
            "is the earliest such date."
        )

    await evaluator.verify(
        claim=temporal_claim,
        node=temporal_leaf,
        sources=_unique_urls(sv.tour_schedule_urls, sv.event_urls),
        additional_instruction=(
            "Use the provided tour schedule URL(s) (and event page if needed) to examine all New England stops "
            "in September 2025. Confirm that the chosen venue's show date is the earliest among those venues that "
            "also meet the 800–900 seat capacity constraint. If the available pages do not allow you to confidently "
            "determine that no earlier qualifying New England date exists in September 2025, mark as not supported."
        ),
    )


# ----------------------------- Main Entrypoint ------------------------------ #
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
    evaluator = Evaluator()
    evaluator.initialize(
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_selected_venue(),
        template_class=AnswerExtraction,
        extraction_name="selected_venue_extraction",
    )

    evaluator.add_custom_info(
        info={"new_england_states": sorted(list(NEW_ENGLAND_STATES))},
        info_type="region_reference",
        info_name="new_england_definition",
    )

    await build_verification_tree(evaluator, extracted)

    return evaluator.get_summary()