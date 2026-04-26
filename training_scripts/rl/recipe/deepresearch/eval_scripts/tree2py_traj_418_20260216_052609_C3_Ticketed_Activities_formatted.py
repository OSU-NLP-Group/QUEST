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
TASK_ID = "westgate_concert_march2026"
TASK_DESCRIPTION = """
Identify a concert performer who announced new tour dates in January 2026 for performances in March 2026, and who has a scheduled performance at the International Theater at Westgate Las Vegas Resort & Casino on March 27, 2026 at 8:00 PM. Verify that the venue has a seating capacity of approximately 1,600 seats with all seats located within 87 feet from the stage. Provide the performer's name, the venue details, and URL references confirming: (1) the January 2026 tour announcement, (2) the venue's seating capacity, (3) the stage distance specification, (4) the March 27, 2026 performance date, and (5) the 8:00 PM show time.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class PerformerExtraction(BaseModel):
    performer_name: Optional[str] = None
    tour_announcement_urls: List[str] = Field(default_factory=list)
    award_urls: List[str] = Field(default_factory=list)


class VenueExtraction(BaseModel):
    venue_name: Optional[str] = None
    venue_urls: List[str] = Field(default_factory=list)
    type_urls: List[str] = Field(default_factory=list)
    location_urls: List[str] = Field(default_factory=list)
    capacity_urls: List[str] = Field(default_factory=list)
    stage_distance_urls: List[str] = Field(default_factory=list)


class PerformanceExtraction(BaseModel):
    performance_date: Optional[str] = None  # e.g., "March 27, 2026"
    performance_time: Optional[str] = None  # e.g., "8:00 PM"
    performance_urls: List[str] = Field(default_factory=list)


class ConcertExtraction(BaseModel):
    performer: Optional[PerformerExtraction] = None
    venue: Optional[VenueExtraction] = None
    performance: Optional[PerformanceExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_concert_info() -> str:
    return """
    Extract structured information from the answer about the performer, venue, and performance. Return a JSON object with three top-level sections: performer, venue, and performance.

    For performer:
    - performer_name: The name of the performer mentioned in the answer.
    - tour_announcement_urls: Array of URLs in the answer that confirm the performer announced new tour dates in January 2026 for performances in March 2026.
    - award_urls: Array of URLs in the answer that confirm the performer has WON (not just nominated) a Grammy Award, a Tony Award, and an Emmy Award.

    For venue:
    - venue_name: The venue name mentioned for the performance (expected to be "International Theater at Westgate Las Vegas Resort & Casino" if stated).
    - venue_urls: Array of URLs that confirm the venue identity (pages that explicitly state or imply the performance venue is the International Theater at Westgate Las Vegas Resort & Casino).
    - type_urls: Array of URLs that confirm the venue is a theater (not an arena).
    - location_urls: Array of URLs that confirm the venue is located in Las Vegas, Nevada.
    - capacity_urls: Array of URLs that confirm the venue has approximately 1,600 seats.
    - stage_distance_urls: Array of URLs that confirm all seats are within 87 feet of the stage (or an equivalent close figure).

    For performance:
    - performance_date: The performance date string as stated in the answer (e.g., "March 27, 2026"). If not found, return null.
    - performance_time: The performance time string as stated in the answer (e.g., "8:00 PM"). If not found, return null.
    - performance_urls: Array of URLs that confirm the scheduled performance date and time (e.g., venue schedule page, official ticketing page, press release).

    Rules:
    - Only extract URLs explicitly present in the answer text. Do not invent URLs.
    - Include full URLs with protocol. If a URL is missing the protocol, prepend http://.
    - If a field is not explicitly stated in the answer, return null (for strings) or an empty array (for lists).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*lists: List[str]) -> List[str]:
    """Merge multiple lists of URLs, preserving order and uniqueness."""
    seen = set()
    merged = []
    for lst in lists:
        for url in lst:
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_performer_identification(evaluator: Evaluator, parent_node, data: ConcertExtraction) -> None:
    """
    Build and verify the Performer_Identification subtree:
    - Tour announcement in January 2026 for March 2026 performances
    - Performer award verification (Grammy, Tony, Emmy winners)
    """
    performer = data.performer or PerformerExtraction()

    perf_node = evaluator.add_parallel(
        id="Performer_Identification",
        desc="Identify the performer and verify they meet the specified criteria",
        parent=parent_node,
        critical=True
    )

    # Optional: performer name existence (helps downstream clarity)
    evaluator.add_custom_node(
        result=bool(performer.performer_name and performer.performer_name.strip()),
        id="Performer_Name_Provided",
        desc="Performer name is provided",
        parent=perf_node,
        critical=True
    )

    # Tour announcement verification subtree
    tour_node = evaluator.add_parallel(
        id="Tour_Announcement_Verification",
        desc="The performer must have announced new tour dates in January 2026 for March 2026 performances",
        parent=perf_node,
        critical=True
    )

    # Existence of tour announcement sources
    evaluator.add_custom_node(
        result=bool(performer.tour_announcement_urls),
        id="Tour_Announcement_Sources_Exist",
        desc="Tour announcement URLs are provided",
        parent=tour_node,
        critical=True
    )

    # Actual verification leaf: URL-supported claim
    tour_verify_leaf = evaluator.add_leaf(
        id="Tour_Announcement_URL_Reference",
        desc="Provide URL reference confirming the January 2026 tour announcement for March 2026 performances",
        parent=tour_node,
        critical=True
    )
    performer_name = performer.performer_name or "the performer"
    tour_claim = (
        f"{performer_name} announced new tour dates in January 2026 that include performances scheduled in March 2026."
    )
    await evaluator.verify(
        claim=tour_claim,
        node=tour_verify_leaf,
        sources=performer.tour_announcement_urls,
        additional_instruction=(
            "Confirm the announcement date falls within January 2026 and explicitly mentions new dates in March 2026. "
            "Press releases, official social posts, or reputable news articles are acceptable. "
            "Allow minor phrasing variations but the core facts (January 2026 announcement and March 2026 dates) must be present."
        )
    )

    # Award verification subtree
    award_node = evaluator.add_parallel(
        id="Award_Verification",
        desc="The performer must be a Grammy, Tony, and Emmy award winner",
        parent=perf_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(performer.award_urls),
        id="Award_Sources_Exist",
        desc="Award confirmation URLs are provided",
        parent=award_node,
        critical=True
    )

    award_verify_leaf = evaluator.add_leaf(
        id="Award_URL_Reference",
        desc="Provide URL reference confirming the performer's Grammy, Tony, and Emmy awards",
        parent=award_node,
        critical=True
    )
    award_claim = (
        f"{performer_name} has WON a Grammy Award, a Tony Award, and an Emmy Award (wins, not just nominations)."
    )
    await evaluator.verify(
        claim=award_claim,
        node=award_verify_leaf,
        sources=performer.award_urls,
        additional_instruction=(
            "Verify that the performer has actual award wins (not merely nominations) for each of Grammy, Tony, and Emmy. "
            "Official award sites, credible biographies, or reputable news sources are acceptable."
        )
    )


async def verify_venue_and_performance_details(evaluator: Evaluator, parent_node, data: ConcertExtraction) -> None:
    """
    Build and verify the Venue_and_Performance_Details subtree:
    Sequential ordering:
      1) Venue identification (must confirm venue)
      2) Venue specifications (location/type, capacity/stage distance)
      3) Performance details (date/time)
    """
    venue = data.venue or VenueExtraction()
    performance = data.performance or PerformanceExtraction()

    vp_node = evaluator.add_sequential(
        id="Venue_and_Performance_Details",
        desc="Verify the venue identification and performance details",
        parent=parent_node,
        critical=True
    )

    # 1) Venue Identification
    venue_id_node = evaluator.add_parallel(
        id="Venue_Identification",
        desc="The venue must be the International Theater at Westgate Las Vegas Resort & Casino",
        parent=vp_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(venue.venue_name and venue.venue_name.strip()),
        id="Venue_Name_Provided",
        desc="The venue name is provided in the answer",
        parent=venue_id_node,
        critical=True
    )

    venue_sources_union = _merge_sources(venue.venue_urls, performance.performance_urls)
    evaluator.add_custom_node(
        result=bool(venue_sources_union),
        id="Venue_Sources_Exist",
        desc="Venue identification URLs are provided",
        parent=venue_id_node,
        critical=True
    )

    venue_verify_leaf = evaluator.add_leaf(
        id="Venue_URL_Reference",
        desc="Provide URL reference confirming the venue is the International Theater at Westgate Las Vegas Resort & Casino",
        parent=venue_id_node,
        critical=True
    )

    venue_claim = (
        "The performance venue is the International Theater at Westgate Las Vegas Resort & Casino."
    )
    await evaluator.verify(
        claim=venue_claim,
        node=venue_verify_leaf,
        sources=venue_sources_union,
        additional_instruction=(
            "Confirm that the venue for the performance is explicitly stated as the International Theater at Westgate Las Vegas Resort & Casino. "
            "Official venue pages, ticketing listings, or schedule pages are acceptable evidence."
        )
    )

    # 2) Venue Specifications
    venue_specs_node = evaluator.add_parallel(
        id="Venue_Specifications",
        desc="Verify the venue meets all specified requirements",
        parent=vp_node,
        critical=True
    )

    # 2.a) Venue Location and Type
    loc_type_node = evaluator.add_parallel(
        id="Venue_Location_and_Type",
        desc="Verify the venue is a theater in Las Vegas, Nevada",
        parent=venue_specs_node,
        critical=True
    )

    # Theater type
    theater_type_node = evaluator.add_parallel(
        id="Theater_Type",
        desc="The venue must be a theater setting (not an arena)",
        parent=loc_type_node,
        critical=True
    )

    theater_sources_union = _merge_sources(venue.type_urls, venue.venue_urls)
    evaluator.add_custom_node(
        result=bool(theater_sources_union),
        id="Theater_Type_Sources_Exist",
        desc="Theater type confirmation URLs are provided",
        parent=theater_type_node,
        critical=True
    )

    theater_verify_leaf = evaluator.add_leaf(
        id="Theater_Type_URL_Reference",
        desc="Provide URL reference confirming the venue is a theater",
        parent=theater_type_node,
        critical=True
    )
    theater_claim = (
        "The International Theater at Westgate Las Vegas Resort & Casino is a theater (not an arena)."
    )
    await evaluator.verify(
        claim=theater_claim,
        node=theater_verify_leaf,
        sources=theater_sources_union,
        additional_instruction=(
            "Verify the venue is described and operated as a theater. "
            "Evidence such as venue descriptions or official naming conventions is acceptable."
        )
    )

    # Location verification
    location_node = evaluator.add_parallel(
        id="Location_Verification",
        desc="The venue must be located in Las Vegas, Nevada",
        parent=loc_type_node,
        critical=True
    )

    location_sources_union = _merge_sources(venue.location_urls, venue.venue_urls)
    evaluator.add_custom_node(
        result=bool(location_sources_union),
        id="Location_Sources_Exist",
        desc="Location confirmation URLs are provided",
        parent=location_node,
        critical=True
    )

    location_verify_leaf = evaluator.add_leaf(
        id="Location_URL_Reference",
        desc="Provide URL reference confirming the Las Vegas, Nevada location",
        parent=location_node,
        critical=True
    )
    location_claim = (
        "The International Theater at Westgate Las Vegas Resort & Casino is located in Las Vegas, Nevada."
    )
    await evaluator.verify(
        claim=location_claim,
        node=location_verify_leaf,
        sources=location_sources_union,
        additional_instruction=(
            "Confirm the venue's location is in Las Vegas, Nevada."
        )
    )

    # 2.b) Capacity and Layout
    capacity_layout_node = evaluator.add_parallel(
        id="Capacity_and_Layout",
        desc="Verify the venue's seating capacity and stage distance specifications",
        parent=venue_specs_node,
        critical=True
    )

    # Seating capacity
    capacity_node = evaluator.add_parallel(
        id="Seating_Capacity",
        desc="The venue must have a seating capacity of approximately 1,600 seats",
        parent=capacity_layout_node,
        critical=True
    )

    capacity_sources_union = _merge_sources(venue.capacity_urls, venue.venue_urls)
    evaluator.add_custom_node(
        result=bool(capacity_sources_union),
        id="Capacity_Sources_Exist",
        desc="Capacity confirmation URLs are provided",
        parent=capacity_node,
        critical=True
    )

    capacity_verify_leaf = evaluator.add_leaf(
        id="Capacity_URL_Reference",
        desc="Provide URL reference confirming the venue seating capacity of approximately 1,600 seats",
        parent=capacity_node,
        critical=True
    )
    capacity_claim = (
        "The International Theater at Westgate Las Vegas Resort & Casino has a seating capacity of approximately 1,600 seats."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_verify_leaf,
        sources=capacity_sources_union,
        additional_instruction=(
            "Allow reasonable approximations and formatting variations (e.g., 1600 vs. 1,600). "
            "The evidence should clearly indicate a capacity near 1,600 seats."
        )
    )

    # Stage distance
    stage_dist_node = evaluator.add_parallel(
        id="Stage_Distance",
        desc="All seats must be within 87 feet from the stage",
        parent=capacity_layout_node,
        critical=True
    )

    stage_sources_union = _merge_sources(venue.stage_distance_urls, venue.venue_urls)
    evaluator.add_custom_node(
        result=bool(stage_sources_union),
        id="Stage_Distance_Sources_Exist",
        desc="Stage distance confirmation URLs are provided",
        parent=stage_dist_node,
        critical=True
    )

    stage_verify_leaf = evaluator.add_leaf(
        id="Stage_Distance_URL_Reference",
        desc="Provide URL reference confirming all seats are within 87 feet from the stage",
        parent=stage_dist_node,
        critical=True
    )
    stage_claim = (
        "At the International Theater at Westgate Las Vegas Resort & Casino, all seats are within 87 feet of the stage."
    )
    await evaluator.verify(
        claim=stage_claim,
        node=stage_verify_leaf,
        sources=stage_sources_union,
        additional_instruction=(
            "Confirm that the venue materials state that every seat is within about 87 feet of the stage. "
            "Minor rounding differences (e.g., 88 feet) are acceptable if the intent clearly matches the claim."
        )
    )

    # 3) Performance Details
    perf_details_node = evaluator.add_parallel(
        id="Performance_Details",
        desc="Verify the specific performance date and time",
        parent=vp_node,
        critical=True
    )

    # Shared existence of performance URLs for both date and time checks
    evaluator.add_custom_node(
        result=bool(performance.performance_urls),
        id="Performance_Sources_Exist",
        desc="Performance schedule URLs are provided",
        parent=perf_details_node,
        critical=True
    )

    # Performance date
    perf_date_node = evaluator.add_parallel(
        id="Performance_Date",
        desc="The performance must be scheduled for March 27, 2026",
        parent=perf_details_node,
        critical=True
    )
    date_verify_leaf = evaluator.add_leaf(
        id="Date_URL_Reference",
        desc="Provide URL reference confirming the March 27, 2026 performance date",
        parent=perf_date_node,
        critical=True
    )
    perf_name_for_claim = (data.performer.performer_name if data.performer and data.performer.performer_name else "the performer")
    date_claim = (
        f"{perf_name_for_claim} has a scheduled performance on March 27, 2026 at the International Theater at Westgate Las Vegas Resort & Casino."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_verify_leaf,
        sources=performance.performance_urls,
        additional_instruction=(
            "Confirm that the schedule or ticketing page clearly lists a performance on March 27, 2026 for this artist at the specified venue."
        )
    )

    # Performance time
    perf_time_node = evaluator.add_parallel(
        id="Performance_Time",
        desc="The show time must be 8:00 PM",
        parent=perf_details_node,
        critical=True
    )
    time_verify_leaf = evaluator.add_leaf(
        id="Time_URL_Reference",
        desc="Provide URL reference confirming the 8:00 PM show time",
        parent=perf_time_node,
        critical=True
    )
    time_claim = (
        "The show time for the March 27, 2026 performance is 8:00 PM."
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_verify_leaf,
        sources=performance.performance_urls,
        additional_instruction=(
            "Verify the listed show time is 8:00 PM for the March 27, 2026 performance. "
            "Minor formatting variations (e.g., 8 PM) are acceptable."
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
    Evaluate the answer for the Westgate International Theater March 2026 concert task.
    Returns a structured summary containing the verification tree and final score.
    """
    # Initialize evaluator with a parallel root (as per rubric)
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

    # Root node mirrors rubric top-level
    top_node = evaluator.add_parallel(
        id="Concert_Performance_Identification",
        desc="Identify a concert performance in Las Vegas in March 2026 that meets all specified criteria",
        parent=root,
        critical=False
    )

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_concert_info(),
        template_class=ConcertExtraction,
        extraction_name="concert_extraction"
    )

    # Build and verify Performer subtree
    await verify_performer_identification(evaluator, top_node, extraction)

    # Build and verify Venue & Performance Details subtree
    await verify_venue_and_performance_details(evaluator, top_node, extraction)

    # Return summary
    return evaluator.get_summary()