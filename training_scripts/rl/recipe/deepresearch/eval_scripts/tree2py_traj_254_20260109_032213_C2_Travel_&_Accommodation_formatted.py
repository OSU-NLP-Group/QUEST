import asyncio
import logging
from typing import Any, List, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "jfk_airport_hotel_2025_skytrax"
TASK_DESCRIPTION = """
A company is planning a large international business conference in New York and requires an airport hotel with extensive meeting facilities. The hotel must be located at John F. Kennedy International Airport (JFK) and ranked as the #1 Best Airport Hotel in North America for 2025 by Skytrax World Airport Awards. The venue must provide at least 50,000 square feet of event space and have at least 45 separate event rooms available (not including hospitality suites). Which hotel meets all these requirements?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AirportHotelExtraction(BaseModel):
    """
    Structured extraction of the hotel claimed to meet all requirements and supporting sources.
    All URL fields must be explicitly present in the answer text per the extraction rules.
    """
    hotel_name: Optional[str] = None
    hotel_official_url: Optional[str] = None

    # Sources explicitly cited in the answer that support each requirement
    location_sources: List[str] = Field(default_factory=list)
    ranking_sources: List[str] = Field(default_factory=list)
    event_space_sqft: Optional[str] = None
    event_space_sources: List[str] = Field(default_factory=list)
    event_rooms_count: Optional[str] = None
    event_rooms_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_airport_hotel() -> str:
    return """
    Extract the single hotel the answer claims meets all specified criteria, along with any explicitly cited supporting URLs.
    Return a JSON object with the following fields:

    1) hotel_name: The specific hotel name provided in the answer (string).
    2) hotel_official_url: The official hotel/property page URL if explicitly present in the answer (string or null).

    3) location_sources: An array of URLs explicitly cited in the answer that support the hotel's location being at John F. Kennedy International Airport (JFK) in New York. Include official hotel pages if they state on-airport/JFK location and are explicitly cited.

    4) ranking_sources: An array of URLs explicitly cited in the answer that support the statement that the hotel is ranked #1 Best Airport Hotel in North America for 2025 by the Skytrax World Airport Awards. Prefer the Skytrax awards page for 2025 if present; otherwise include any cited credible source that explicitly states this ranking.

    5) event_space_sqft: The event/meeting space size as stated in the answer (string, e.g., "50,000+", "over 50,000 square feet", "50,000 sq ft"). Extract exactly as written.

    6) event_space_sources: An array of URLs explicitly cited in the answer that support the event/meeting space size claim.

    7) event_rooms_count: The number of event/meeting rooms as stated in the answer (string, e.g., "45", "45+", "over 45 rooms"). Extract exactly as written. Exclude hospitality suites in the meaning of this count if the answer distinguishes that.

    8) event_rooms_sources: An array of URLs explicitly cited in the answer that support the event/meeting room count.

    GENERAL RULES:
    - Extract only information explicitly present in the answer.
    - If a field is missing in the answer, set it to null or an empty array as appropriate.
    - For URLs, extract the actual URLs (plain, markdown, etc.). Do not invent or infer URLs.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _combine_sources(primary_list: List[str], maybe_extra: Optional[str]) -> List[str]:
    """
    Combine a list of sources with an optional extra URL (e.g., an official hotel URL)
    while ensuring uniqueness and preserving order.
    Only include the extra URL if it was explicitly extracted (i.e., present in the answer).
    """
    combined = list(primary_list) if primary_list else []
    if maybe_extra and maybe_extra.strip():
        if maybe_extra not in combined:
            combined.append(maybe_extra)
    return combined


async def build_verification_tree(
    evaluator: Evaluator,
    root_node,
    extracted: AirportHotelExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and perform verifications.
    """

    # Top-level critical node: "Correct_Hotel_Identification"
    correct_hotel_node = evaluator.add_parallel(
        id="Correct_Hotel_Identification",
        desc="The answer correctly identifies the hotel that meets all specified criteria",
        parent=root_node,
        critical=True  # All children must be critical too (framework constraint)
    )

    # 1) Hotel Name Provided (Critical, existence check)
    name_provided = evaluator.add_custom_node(
        result=bool(extracted.hotel_name and extracted.hotel_name.strip()),
        id="Hotel_Name_Provided",
        desc="The answer provides a specific hotel name",
        parent=correct_hotel_node,
        critical=True
    )

    # 2) Location Requirement (Critical, verify by cited sources)
    location_node = evaluator.add_leaf(
        id="Location_Requirement",
        desc="The identified hotel is located at John F. Kennedy International Airport (JFK) in New York",
        parent=correct_hotel_node,
        critical=True
    )
    location_claim = f"{extracted.hotel_name or 'The hotel'} is located at John F. Kennedy International Airport (JFK) in New York."
    location_sources = _combine_sources(extracted.location_sources, extracted.hotel_official_url)
    location_instruction = (
        "Confirm the hotel is on JFK airport property (on-airport) in New York (Queens). "
        "Accept explicit phrases such as 'at JFK', 'on-airport hotel at JFK', 'inside Terminal X at JFK', etc. "
        "If the page only says 'near JFK', 'minutes from JFK', or is not clearly on-airport, consider the claim not supported."
    )

    # 3) Ranking Requirement (Critical, verify by cited sources)
    ranking_node = evaluator.add_leaf(
        id="Ranking_Requirement",
        desc="The identified hotel is ranked as the #1 Best Airport Hotel in North America for 2025 by Skytrax World Airport Awards",
        parent=correct_hotel_node,
        critical=True
    )
    ranking_claim = (
        f"{extracted.hotel_name or 'The hotel'} is ranked as the #1 Best Airport Hotel in North America for 2025 by Skytrax World Airport Awards."
    )
    ranking_sources = list(extracted.ranking_sources)  # Do not auto-add official URL unless it was cited
    ranking_instruction = (
        "Verify the specific Skytrax award category and year: '#1 Best Airport Hotel in North America' for 2025. "
        "The source should explicitly indicate the hotel is ranked #1 for North America in 2025 by Skytrax World Airport Awards. "
        "If the source indicates a different year or region, or a non-#1 ranking, the claim is not supported."
    )

    # 4) Meeting Space Requirements (Critical, parallel child requirements)
    meeting_node = evaluator.add_parallel(
        id="Meeting_Space_Requirements",
        desc="The identified hotel meets both meeting space requirements",
        parent=correct_hotel_node,
        critical=True
    )

    # 4.1) Event Space Size ≥ 50,000 sq ft (Critical)
    event_space_node = evaluator.add_leaf(
        id="Event_Space_Size",
        desc="The hotel has at least 50,000 square feet of event space",
        parent=meeting_node,
        critical=True
    )
    event_space_claim = (
        f"{extracted.hotel_name or 'The hotel'} has at least 50,000 square feet of event or meeting space."
    )
    event_space_sources = _combine_sources(extracted.event_space_sources, extracted.hotel_official_url)
    event_space_instruction = (
        "Check that the cited page explicitly states the hotel has ≥ 50,000 sq ft of event/meeting/conference/function space. "
        "Accept equivalent phrasing like '50,000+', 'over 50,000', 'approximately 50,000', or '50,000 square feet'."
    )

    # 4.2) Event Rooms Count ≥ 45 (Critical, excluding hospitality suites)
    event_rooms_node = evaluator.add_leaf(
        id="Event_Rooms_Count",
        desc="The hotel has at least 45 separate event rooms (excluding hospitality suites)",
        parent=meeting_node,
        critical=True
    )
    event_rooms_claim = (
        f"{extracted.hotel_name or 'The hotel'} has at least 45 separate event/meeting rooms (excluding hospitality suites)."
    )
    event_rooms_sources = _combine_sources(extracted.event_rooms_sources, extracted.hotel_official_url)
    event_rooms_instruction = (
        "Confirm the count of dedicated event/meeting rooms is ≥ 45. "
        "Treat 'meeting rooms', 'function rooms', or 'event rooms' as equivalent. "
        "Do not count hospitality suites; the page should refer to dedicated meeting/event spaces."
    )

    # Batch verify independent leaves (location, ranking, event space size, event rooms count)
    claims_and_sources = [
        (location_claim, location_sources if location_sources else None, location_node, location_instruction),
        (ranking_claim, ranking_sources if ranking_sources else None, ranking_node, ranking_instruction),
        (event_space_claim, event_space_sources if event_space_sources else None, event_space_node, event_space_instruction),
        (event_rooms_claim, event_rooms_sources if event_rooms_sources else None, event_rooms_node, event_rooms_instruction),
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
    Evaluate an agent's answer for the JFK airport hotel task using the Mind2Web2 framework.
    """
    # Initialize evaluator with a parallel root (non-critical)
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_airport_hotel(),
        template_class=AirportHotelExtraction,
        extraction_name="airport_hotel_extraction",
    )

    # Optionally record extracted highlights
    evaluator.add_custom_info(
        info={
            "hotel_name": extracted.hotel_name,
            "hotel_official_url": extracted.hotel_official_url,
            "event_space_sqft": extracted.event_space_sqft,
            "event_rooms_count": extracted.event_rooms_count,
            "location_sources_count": len(extracted.location_sources),
            "ranking_sources_count": len(extracted.ranking_sources),
            "event_space_sources_count": len(extracted.event_space_sources),
            "event_rooms_sources_count": len(extracted.event_rooms_sources),
        },
        info_type="extraction_summary",
    )

    # Build the verification tree and run checks
    await build_verification_tree(evaluator, root, extracted)

    # Return unified summary
    return evaluator.get_summary()