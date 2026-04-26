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
TASK_ID = "till_lindemann_smallest_venue_2025_eu_arena_leg"
TASK_DESCRIPTION = """
What is the name and concert seating capacity of the smallest indoor arena venue on Till Lindemann's Meine Welt Tour 2025 European arena leg (October 29 - December 18, 2025)?
The answer must provide a specific venue name and its concert seating capacity, grounded by sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueSelection(BaseModel):
    """Selected venue information as stated in the answer."""
    name: Optional[str] = None
    stated_concert_capacity: Optional[str] = None
    capacity_source_urls: List[str] = Field(default_factory=list)
    venue_info_urls: List[str] = Field(default_factory=list)


class TourEvidence(BaseModel):
    """Tour and leg evidence as cited in the answer."""
    tour_name: Optional[str] = None  # e.g., "Meine Welt Tour 2025"
    leg_label: Optional[str] = None  # e.g., "European arena leg"
    start_date: Optional[str] = None  # e.g., "October 29, 2025"
    end_date: Optional[str] = None    # e.g., "December 18, 2025"
    schedule_urls: List[str] = Field(default_factory=list)  # URLs that document the leg schedule


class AnswerExtraction(BaseModel):
    """Top-level extracted structure from the answer."""
    venue: Optional[VenueSelection] = None
    tour: Optional[TourEvidence] = None
    comparison_urls: List[str] = Field(default_factory=list)  # Any additional URLs used to justify "smallest"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_answer_core() -> str:
    return """
Extract the following structured information from the answer text about the smallest indoor arena on Till Lindemann's Meine Welt Tour 2025 European arena leg (October 29 - December 18, 2025).

Return a JSON with:
- venue:
  - name: The specific venue name the answer identifies as the smallest.
  - stated_concert_capacity: The concert seating capacity number/value as stated in the answer (keep as a string; do not normalize; include qualifiers like "~", "approx.", or ranges exactly as written).
  - capacity_source_urls: All URLs provided that directly document the venue’s concert seating capacity.
  - venue_info_urls: URLs provided that describe or document the venue’s type/classification (e.g., indoor arena), location, or official info. If none, reuse capacity_source_urls as appropriate. Only include URLs explicitly present in the answer.
- tour:
  - tour_name: The tour name as mentioned in the answer (e.g., "Meine Welt Tour 2025").
  - leg_label: The leg name/label as mentioned in the answer (e.g., "European arena leg").
  - start_date: The leg start date as mentioned in the answer (e.g., "October 29, 2025"). If not mentioned, return null.
  - end_date: The leg end date as mentioned in the answer (e.g., "December 18, 2025"). If not mentioned, return null.
  - schedule_urls: All URLs that document the leg’s schedule (dates and venues) for the relevant period in the answer. Only include URLs explicitly present in the answer.
- comparison_urls: Any additional URLs included in the answer that support comparison of venue capacities (e.g., capacity pages for other venues used to argue “smallest”). If none, return an empty array.

Rules:
- Do not fabricate URLs or values. Only extract exactly what appears in the answer.
- If any requested field is missing in the answer, return null or an empty list as appropriate.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(x: Optional[List[str]]) -> List[str]:
    return x if isinstance(x, list) else []


def _unique_merge(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst:
            if url and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: AnswerExtraction) -> None:
    """
    Build the verification tree according to the rubric and run all verifications.
    This function assumes evaluator.initialize() has already been called.
    """
    # Create the top-level critical sequential node "Complete_Answer"
    complete_node = evaluator.add_sequential(
        id="Complete_Answer",
        desc=("The answer must provide both the name of a specific venue and its concert seating capacity for a venue "
              "on Till Lindemann's Meine Welt Tour 2025 European arena leg (October 29 - December 18, 2025)"),
        parent=evaluator.root,
        critical=True
    )

    # Extracted fields (safe access)
    venue = extracted.venue or VenueSelection()
    tour = extracted.tour or TourEvidence()

    schedule_urls = _safe_list(tour.schedule_urls)
    capacity_urls = _safe_list(venue.capacity_source_urls)
    venue_info_urls = _safe_list(venue.venue_info_urls)
    comparison_urls = _safe_list(extracted.comparison_urls)

    # ------------------------ Existence check: venue & capacity ------------------------
    evaluator.add_custom_node(
        result=bool(venue.name and venue.name.strip()) and bool(venue.stated_concert_capacity and venue.stated_concert_capacity.strip()),
        id="Answer_Provides_Venue_And_Capacity",
        desc="Answer provides both a venue name and a concert seating capacity value.",
        parent=complete_node,
        critical=True
    )

    # ------------------------ Tour Context Verification (critical) ---------------------
    tour_ctx_node = evaluator.add_sequential(
        id="Tour_Context_Verification",
        desc=("The answer must correctly identify the tour as Till Lindemann's 'Meine Welt Tour 2025' and correctly "
              "specify that the venue is from the European arena leg during the time period October 29 - December 18, 2025"),
        parent=complete_node,
        critical=True
    )

    # Tour documentation URL subtree (critical)
    tour_doc_node = evaluator.add_sequential(
        id="Tour_Documentation_URL",
        desc=("A valid reference URL must be provided that documents the Meine Welt Tour 2025 European arena leg "
              "schedule, including dates and venue list for the specified time period"),
        parent=tour_ctx_node,
        critical=True
    )

    # Leaf: schedule URLs provided
    evaluator.add_custom_node(
        result=len(schedule_urls) > 0,
        id="Tour_URLs_Provided",
        desc="At least one schedule URL is provided that documents the European arena leg schedule.",
        parent=tour_doc_node,
        critical=True
    )

    # Leaf: tour URL supports leg/dates/list
    tour_url_supports_node = evaluator.add_leaf(
        id="Tour_URL_Supports_European_Arena_Leg_Period",
        desc=("Provided URL(s) document the Meine Welt Tour 2025 European arena leg schedule, including dates between "
              "October 29 and December 18, 2025, and listing the venues for that period."),
        parent=tour_doc_node,
        critical=True
    )
    await evaluator.verify(
        claim=("This page documents the schedule for Till Lindemann's 'Meine Welt Tour 2025' European arena leg, "
               "covering dates between October 29, 2025 and December 18, 2025 (inclusive), and lists the venues for that leg."),
        node=tour_url_supports_node,
        sources=schedule_urls,
        additional_instruction=("Confirm that the page references 'Meine Welt Tour 2025' by Till Lindemann and specifically "
                                "shows the European arena leg during the stated timeframe with dates and venue list. "
                                "Minor wording variations are acceptable (e.g., 'European arena tour').")
    )

    # ---------------- Smallest Venue Identification subtree (critical) -----------------
    smallest_node = evaluator.add_sequential(
        id="Smallest_Venue_Identification",
        desc=("The identified venue must be verified as the smallest indoor arena by concert seating capacity among all "
              "venues on the European arena leg (October 29 - December 18, 2025), and must be classified as an indoor arena "
              "(not an outdoor venue, not a club or concert hall)"),
        parent=tour_ctx_node,
        critical=True
    )

    # Leaf: venue appears on that leg (via schedule URLs)
    venue_in_leg_node = evaluator.add_leaf(
        id="Venue_Listed_On_Leg",
        desc="The identified venue appears on the Meine Welt Tour 2025 European arena leg schedule within the specified dates.",
        parent=smallest_node,
        critical=True
    )
    await evaluator.verify(
        claim=(f"The venue '{venue.name or ''}' is listed as one of the venues on the Meine Welt Tour 2025 European arena leg "
               "between October 29 and December 18, 2025."),
        node=venue_in_leg_node,
        sources=schedule_urls,
        additional_instruction=("Check if the venue name appears among the dates in the specified time window on the schedule page(s). "
                                "Allow minor spelling variations or local-language equivalents.")
    )

    # Leaf: venue is an indoor arena (classification)
    venue_indoor_node = evaluator.add_leaf(
        id="Venue_Is_Indoor_Arena",
        desc="The identified venue is an indoor arena (not outdoor, not club or concert hall).",
        parent=smallest_node,
        critical=True
    )
    indoor_sources = _unique_merge(venue_info_urls, capacity_urls)
    await evaluator.verify(
        claim=(f"The venue '{venue.name or ''}' is an indoor arena (i.e., enclosed arena), not an outdoor venue, club, or concert hall."),
        node=venue_indoor_node,
        sources=indoor_sources,
        additional_instruction=("Confirm the venue is an indoor arena. Use official venue pages, reputable databases, or Wikipedia. "
                                "If a page ambiguously classifies the venue, do not count it as supported.")
    )

    # Venue documentation subtree (critical)
    venue_doc_node = evaluator.add_sequential(
        id="Venue_Documentation_URL",
        desc="A valid reference URL must be provided that documents the concert seating capacity of the identified venue",
        parent=smallest_node,
        critical=True
    )

    # Leaf: capacity URL provided
    evaluator.add_custom_node(
        result=len(capacity_urls) > 0,
        id="Capacity_URL_Provided",
        desc="At least one URL is provided that documents the venue's concert seating capacity.",
        parent=venue_doc_node,
        critical=True
    )

    # Capacity accuracy leaf (critical)
    capacity_accuracy_node = evaluator.add_leaf(
        id="Capacity_Accuracy",
        desc=("The concert seating capacity stated in the answer matches the documented concert capacity (not sports capacity) "
              "of the identified venue according to reliable sources."),
        parent=venue_doc_node,
        critical=True
    )
    await evaluator.verify(
        claim=(f"The concert seating capacity of '{venue.name or ''}' is {venue.stated_concert_capacity or ''}."),
        node=capacity_accuracy_node,
        sources=capacity_urls,
        additional_instruction=("Verify the capacity refers to concerts (or general maximum seated capacity for concerts), not a sports configuration. "
                                "If multiple capacities exist for different configurations, ensure the cited figure corresponds to concert or event seating. "
                                "Allow minor rounding differences.")
    )

    # Leaf: smallest across the leg (critical)
    smallest_across_leg_node = evaluator.add_leaf(
        id="Smallest_Across_Leg",
        desc=("The identified venue has the smallest concert seating capacity among all venues on the European arena leg "
              "between October 29 and December 18, 2025."),
        parent=smallest_node,
        critical=True
    )
    # Combine available sources to attempt to support the 'smallest' assertion
    smallest_sources = _unique_merge(schedule_urls, capacity_urls, comparison_urls)
    await evaluator.verify(
        claim=(f"Among all venues on Till Lindemann's 'Meine Welt Tour 2025' European arena leg "
               f"(Oct 29–Dec 18, 2025), the venue '{venue.name or ''}' has the smallest concert seating capacity."),
        node=smallest_across_leg_node,
        sources=smallest_sources,
        additional_instruction=("Look for an explicit statement that this venue has the smallest capacity among that leg or "
                                "sufficient comparative evidence listing capacities for multiple venues on the leg that allows concluding it is the smallest. "
                                "If the provided sources are insufficient to confirm 'smallest', mark as not supported.")
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
    Evaluate an answer for the smallest indoor arena venue on Till Lindemann's
    Meine Welt Tour 2025 European arena leg (Oct 29 - Dec 18, 2025).
    """
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_answer_core(),
        template_class=AnswerExtraction,
        extraction_name="answer_extraction"
    )

    # Build and verify the rubric-based tree
    await build_and_verify_tree(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()