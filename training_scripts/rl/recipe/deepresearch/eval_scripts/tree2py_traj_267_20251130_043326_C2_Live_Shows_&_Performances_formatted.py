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
TASK_ID = "tate_mcrae_2025_miss_possessive_us_venue_largest_capacity_multishow"
TASK_DESCRIPTION = (
    "Identify the United States venue from Tate McRae's 2025 Miss Possessive Tour that hosted multiple performances "
    "(at least 2 shows) and had the largest concert seating capacity among all such US venues with multiple performances. "
    "Provide the venue name, the city where it is located, the concert seating capacity, and at least two specific "
    "performance dates from the tour at this venue."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SelectedVenueExtraction(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state_or_region: Optional[str] = None
    country: Optional[str] = None
    tour_name: Optional[str] = None
    tour_year: Optional[str] = None
    concert_capacity: Optional[str] = None
    performance_dates: List[str] = Field(default_factory=list)
    # URLs explicitly cited in the answer supporting venue facts (any mixture)
    source_urls: List[str] = Field(default_factory=list)
    # Optional more granular source buckets if present in the answer
    capacity_source_urls: List[str] = Field(default_factory=list)
    date_source_urls: List[str] = Field(default_factory=list)
    venue_profile_urls: List[str] = Field(default_factory=list)


class OtherVenueItem(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state_or_region: Optional[str] = None
    country: Optional[str] = None
    concert_capacity: Optional[str] = None
    num_performances: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class OtherVenuesExtraction(BaseModel):
    other_us_multi_show_venues: List[OtherVenueItem] = Field(default_factory=list)
    comparison_claim_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_selected_venue() -> str:
    return """
    From the provided answer, identify the single US venue that the answer claims meets ALL of the following:
    - It is part of Tate McRae’s Miss Possessive Tour in 2025.
    - It hosted multiple performances (at least 2 shows) on that tour.
    - Among the US venues that hosted ≥2 performances on this tour, it has the largest concert seating capacity.
    
    Extract the following fields exactly as stated in the answer (do not infer or invent):
    - venue_name: the venue's name
    - city: the city of the venue
    - state_or_region: the US state or region if stated
    - country: the country of the venue (should be United States or equivalent)
    - tour_name: the tour name associated with these performances (e.g., "Miss Possessive Tour")
    - tour_year: the tour year (e.g., "2025")
    - concert_capacity: the concert seating capacity at this venue (string; keep formatting as in the answer)
    - performance_dates: an array of specific performance dates at this venue, from the 2025 tour (include at least two if present)
    - source_urls: an array of all URLs in the answer that support these facts
    - capacity_source_urls: an array of URLs (if any) that specifically mention or support the capacity figure
    - date_source_urls: an array of URLs (if any) that specifically mention or support the dates at this venue
    - venue_profile_urls: an array of URLs (if any) to the venue's official page or profile pages cited in the answer
    
    If any field is not explicitly provided in the answer text, set it to null (for single values) or [] (for arrays).
    Only include URLs that are explicitly present in the answer text.
    """


def prompt_extract_other_venues() -> str:
    return """
    From the provided answer, extract information about OTHER US venues (excluding the selected one) that are mentioned as having hosted multiple performances (≥2 shows) on Tate McRae’s Miss Possessive Tour (2025).
    
    For each such venue, extract:
    - venue_name
    - city
    - state_or_region
    - country
    - concert_capacity (string; keep formatting as in the answer)
    - num_performances (string or number as written; keep formatting)
    - source_urls: an array of URLs in the answer that support the facts for that venue
    
    Also, extract:
    - comparison_claim_urls: an array of any URLs cited that compare capacities or explicitly claim which venue has the largest capacity among US multi‑show venues on the tour.
    
    If the answer does not mention any other US multi‑show venues, return an empty array for other_us_multi_show_venues.
    Only include URLs that are explicitly present in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_unique_urls(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for lst in url_lists:
        for u in lst or []:
            if not isinstance(u, str):
                continue
            u2 = u.strip()
            if not u2:
                continue
            if u2 not in seen:
                seen.add(u2)
                merged.append(u2)
    return merged


def _first_n_dates(dates: List[str], n: int = 2) -> str:
    if not dates:
        return ""
    subset = [d for d in dates if isinstance(d, str) and d.strip()]
    if not subset:
        return ""
    return ", ".join(subset[:n])


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    selected: SelectedVenueExtraction,
    others: OtherVenuesExtraction,
) -> None:
    """
    Build the verification tree based on the rubric and run all verifications.
    """
    # Top-level node from rubric (critical + sequential)
    us_node = evaluator.add_sequential(
        id="US_Venue_Identification",
        desc="Identify the US venue from Tate McRae's 2025 Miss Possessive Tour that had multiple performances and the largest concert seating capacity among such US venues; provide required venue details and dates.",
        parent=evaluator.root,
        critical=True
    )

    # Prepare sources
    selected_sources = _merge_unique_urls(
        selected.source_urls,
        selected.capacity_source_urls,
        selected.date_source_urls,
        selected.venue_profile_urls,
    )
    others_sources_flat: List[str] = []
    for item in others.other_us_multi_show_venues:
        others_sources_flat.extend(item.source_urls or [])
    comparison_sources = _merge_unique_urls(others_sources_flat, others.comparison_claim_urls)
    all_selection_sources = _merge_unique_urls(selected_sources, comparison_sources)

    # ------------------------------------------------------------------- #
    # Venue_Selection_Criterion (critical + parallel)                     #
    # ------------------------------------------------------------------- #
    selection_node = evaluator.add_parallel(
        id="Venue_Selection_Criterion",
        desc="The identified venue satisfies all selection constraints.",
        parent=us_node,
        critical=True
    )

    # 1) Correct_Tour_And_Year
    correct_tour_node = evaluator.add_leaf(
        id="Correct_Tour_And_Year",
        desc="The venue’s listed performances are part of Tate McRae's Miss Possessive Tour (2025).",
        parent=selection_node,
        critical=True
    )
    venue_name = selected.venue_name or "the identified venue"
    city_piece = f" in {selected.city}" if selected.city else ""
    first_date = _first_n_dates(selected.performance_dates, 1)
    if first_date:
        claim_cty = f"At {venue_name}{city_piece}, the Tate McRae concert on {first_date} is part of the Miss Possessive Tour in 2025."
    else:
        claim_cty = f"The Tate McRae concert(s) at {venue_name}{city_piece} are part of the Miss Possessive Tour in 2025."
    await evaluator.verify(
        claim=claim_cty,
        node=correct_tour_node,
        sources=selected_sources,
        additional_instruction="Support is sufficient if the page(s) clearly indicate the show(s) at this venue belong to Tate McRae's 'Miss Possessive Tour' and are dated in 2025. Allow minor name variants (e.g., '2025 Miss Possessive Tour')."
    )

    # 2) Located_In_United_States
    located_us_node = evaluator.add_leaf(
        id="Located_In_United_States",
        desc="The identified venue is located in the United States.",
        parent=selection_node,
        critical=True
    )
    claim_loc = f"{venue_name} is located in the United States."
    await evaluator.verify(
        claim=claim_loc,
        node=located_us_node,
        sources=selected_sources,
        additional_instruction="Evidence can be the venue's city/state within the USA or an explicit mention of the country."
    )

    # 3) Hosted_At_Least_Two_Shows
    multishow_node = evaluator.add_leaf(
        id="Hosted_At_Least_Two_Shows",
        desc="The identified venue hosted at least two performances (≥2 shows) on the 2025 Miss Possessive Tour.",
        parent=selection_node,
        critical=True
    )
    two_dates_str = _first_n_dates(selected.performance_dates, 2)
    if two_dates_str:
        claim_ms = f"{venue_name} hosted at least two Miss Possessive Tour shows in 2025, such as on {two_dates_str}."
    else:
        claim_ms = f"{venue_name} hosted at least two Miss Possessive Tour shows in 2025."
    await evaluator.verify(
        claim=claim_ms,
        node=multishow_node,
        sources=selected_sources,
        additional_instruction="Support is sufficient if the source(s) list at least two distinct 2025 tour dates for this venue."
    )

    # 4) Largest_Concert_Capacity_Among_Eligible_US_Venues
    largest_cap_node = evaluator.add_leaf(
        id="Largest_Concert_Capacity_Among_Eligible_US_Venues",
        desc="Among all US venues that hosted ≥2 performances on the 2025 Miss Possessive Tour, the identified venue has the largest concert seating capacity.",
        parent=selection_node,
        critical=True
    )
    capacity_text = selected.concert_capacity or "the largest capacity"
    claim_largest = (
        f"Among US venues that hosted at least two Miss Possessive Tour shows in 2025, {venue_name} has the largest concert seating capacity "
        f"(reported as {capacity_text})."
    )
    await evaluator.verify(
        claim=claim_largest,
        node=largest_cap_node,
        sources=all_selection_sources,
        additional_instruction=(
            "This claim must be explicitly supported by the page content. A page may support it by either: "
            "(a) directly stating this venue has the largest capacity among US multi‑show venues for the tour, or "
            "(b) providing comparative capacity listings that clearly imply it. "
            "If the page is unrelated or does not provide sufficient evidence, mark as not supported."
        )
    )

    # ------------------------------------------------------------------- #
    # Venue_Details (critical + parallel)                                 #
    # ------------------------------------------------------------------- #
    details_node = evaluator.add_parallel(
        id="Venue_Details",
        desc="All required details about the identified venue are provided.",
        parent=us_node,
        critical=True
    )

    # Venue_Name provided
    evaluator.add_custom_node(
        result=bool(selected.venue_name and selected.venue_name.strip()),
        id="Venue_Name",
        desc="The venue name is provided.",
        parent=details_node,
        critical=True
    )

    # City_Name provided
    evaluator.add_custom_node(
        result=bool(selected.city and selected.city.strip()),
        id="City_Name",
        desc="The city where the venue is located is provided.",
        parent=details_node,
        critical=True
    )

    # Concert_Capacity provided
    evaluator.add_custom_node(
        result=bool(selected.concert_capacity and selected.concert_capacity.strip()),
        id="Concert_Capacity",
        desc="The venue's concert seating capacity is provided.",
        parent=details_node,
        critical=True
    )

    # Performance_Dates provided (at least two)
    evaluator.add_custom_node(
        result=(len([d for d in (selected.performance_dates or []) if isinstance(d, str) and d.strip()]) >= 2),
        id="Performance_Dates",
        desc="At least two specific performance dates from the 2025 Miss Possessive Tour at this venue are provided.",
        parent=details_node,
        critical=True
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
    Evaluate an answer for the Tate McRae 2025 Miss Possessive Tour US venue (largest capacity among multi‑show venues) task.
    """
    # Initialize evaluator/root
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root container; actual rubric root is added as a child node
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

    # Perform extractions in parallel
    selected_venue_task = evaluator.extract(
        prompt=prompt_extract_selected_venue(),
        template_class=SelectedVenueExtraction,
        extraction_name="selected_venue_extraction"
    )
    other_venues_task = evaluator.extract(
        prompt=prompt_extract_other_venues(),
        template_class=OtherVenuesExtraction,
        extraction_name="other_venues_extraction"
    )
    selected_extraction, others_extraction = await asyncio.gather(selected_venue_task, other_venues_task)

    # Optional: record a quick custom info summary
    evaluator.add_custom_info(
        info={
            "selected_venue_name": selected_extraction.venue_name,
            "selected_city": selected_extraction.city,
            "selected_capacity": selected_extraction.concert_capacity,
            "num_selected_dates": len(selected_extraction.performance_dates or []),
            "num_selected_sources": len(selected_extraction.source_urls or []),
            "num_other_us_multi_show_venues": len(others_extraction.other_us_multi_show_venues or []),
            "num_comparison_urls": len(others_extraction.comparison_claim_urls or [])
        },
        info_type="extraction_summary"
    )

    # Build and verify rubric tree
    await build_and_verify(evaluator, selected_extraction, others_extraction)

    # Return standard summary
    return evaluator.get_summary()