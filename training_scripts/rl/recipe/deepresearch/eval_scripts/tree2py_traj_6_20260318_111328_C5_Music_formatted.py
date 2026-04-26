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
TASK_ID = "summerfest_2026_details"
TASK_DESCRIPTION = """
Summerfest, one of the largest music festivals in the United States, is scheduled to take place in 2026. Provide the following information about Summerfest 2026: (1) The exact start date and end date of the festival, (2) The city and state where the festival is held, and (3) The names of three artists confirmed as American Family Insurance Amphitheater headliners. For each piece of information, provide supporting reference URLs from official sources or reliable music industry publications.
"""

# Expected canonical values based on the rubric
EXPECTED_START_DATE = "June 16, 2026"
EXPECTED_END_DATE = "June 20, 2026"
EXPECTED_CITY = "Milwaukee"
EXPECTED_STATE = "Wisconsin"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HeadlinerInfo(BaseModel):
    artist: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Summerfest2026Extraction(BaseModel):
    # Dates
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    date_sources: List[str] = Field(default_factory=list)

    # Location
    city: Optional[str] = None
    state: Optional[str] = None
    location_sources: List[str] = Field(default_factory=list)

    # Headliners (American Family Insurance Amphitheater)
    headliners: List[HeadlinerInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_summerfest_2026() -> str:
    return """
    Extract structured information about Summerfest 2026 from the provided answer.

    Return a JSON object with the following fields:
    - start_date: The first day of the festival as explicitly stated in the answer. If dates are given as a range (e.g., "June 16–20, 2026"), extract the start date as a full date (e.g., "June 16, 2026"). Accept variants like "2026-06-16" or "6/16/2026" as they appear in the answer.
    - end_date: The last day of the festival as explicitly stated in the answer. If given as a range, extract the end date as a full date (e.g., "June 20, 2026"). Accept variants like "2026-06-20" or "6/20/2026" as they appear in the answer.
    - date_sources: An array of all URLs cited in the answer that directly support the 2026 date range for Summerfest. Prefer official Summerfest websites, official amphitheater pages, or reputable publications (Billboard, Pollstar, Rolling Stone, etc.). If no URLs are provided for dates, return an empty list.

    - city: The city where Summerfest is held as explicitly stated in the answer (e.g., "Milwaukee").
    - state: The state where Summerfest is held as explicitly stated in the answer (e.g., "Wisconsin" or "WI"; preserve the exact text from the answer).
    - location_sources: An array of URLs cited in the answer that support the location of Summerfest (city and state). Prefer official Summerfest sources or reputable publications. If none are provided, return an empty list.

    - headliners: An array of up to 5 objects. Each object must contain:
        - artist: The artist name explicitly listed as an "American Family Insurance Amphitheater" headliner for Summerfest 2026 in the answer. Only include artists that the answer claims are amphitheater headliners (do not include ground-stage headliners or general performers).
        - sources: An array of URLs the answer cites that support this headliner claim. Prefer official announcements, official event pages, Live Nation listings, or reputable media outlets. If none are provided, return an empty list.

    Notes:
    - Do not invent information. If any field is not mentioned in the answer, return null (for single values) or [] (for arrays).
    - If the answer provides more than three amphitheater headliners, still extract them all; the evaluator will use only the first three.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_festival_dates(
    evaluator: Evaluator,
    parent_node,
    extracted: Summerfest2026Extraction
) -> None:
    # Parent node for festival dates (critical)
    dates_node = evaluator.add_parallel(
        id="festival_dates",
        desc="The date range when Summerfest 2026 takes place",
        parent=parent_node,
        critical=True
    )

    # Start date: verify it matches the expected canonical date
    start_leaf = evaluator.add_leaf(
        id="start_date",
        desc="The first day of the festival (June 16, 2026)",
        parent=dates_node,
        critical=True
    )
    provided_start = extracted.start_date or ""
    claim_start = (
        f"The start date of Summerfest 2026 provided in the answer ('{provided_start}') "
        f"corresponds to the calendar day {EXPECTED_START_DATE}."
    )
    await evaluator.verify(
        claim=claim_start,
        node=start_leaf,
        additional_instruction="Treat minor formatting variants as equivalent (e.g., 'June 16th, 2026', '2026-06-16', '6/16/2026', or including a weekday). If the provided value is missing or clearly not the same day, mark as incorrect."
    )

    # End date: verify it matches the expected canonical date
    end_leaf = evaluator.add_leaf(
        id="end_date",
        desc="The last day of the festival (June 20, 2026)",
        parent=dates_node,
        critical=True
    )
    provided_end = extracted.end_date or ""
    claim_end = (
        f"The end date of Summerfest 2026 provided in the answer ('{provided_end}') "
        f"corresponds to the calendar day {EXPECTED_END_DATE}."
    )
    await evaluator.verify(
        claim=claim_end,
        node=end_leaf,
        additional_instruction="Treat minor formatting variants as equivalent (e.g., 'June 20th, 2026', '2026-06-20', '6/20/2026', or including a weekday). If the provided value is missing or clearly not the same day, mark as incorrect."
    )

    # Reference URLs confirm the dates
    ref_dates_leaf = evaluator.add_leaf(
        id="reference_url_dates",
        desc="URL source confirming the festival dates",
        parent=dates_node,
        critical=True
    )
    dates_sources = extracted.date_sources or []
    claim_dates_support = (
        f"Summerfest 2026 runs from {EXPECTED_START_DATE} to {EXPECTED_END_DATE}."
    )
    await evaluator.verify(
        claim=claim_dates_support,
        node=ref_dates_leaf,
        sources=dates_sources,
        additional_instruction=(
            "Verify that at least one provided URL explicitly supports the 2026 date range "
            f"({EXPECTED_START_DATE} to {EXPECTED_END_DATE}). Prefer official Summerfest or American Family Insurance Amphitheater pages, "
            "or reputable music industry publications (e.g., Billboard, Pollstar, Rolling Stone). "
            "Reject pages that refer to other years (e.g., 2024/2025) or do not clearly indicate 2026."
        )
    )


async def build_and_verify_location(
    evaluator: Evaluator,
    parent_node,
    extracted: Summerfest2026Extraction
) -> None:
    # Parent node for location (critical)
    loc_node = evaluator.add_parallel(
        id="festival_location",
        desc="The geographic location where Summerfest 2026 is held",
        parent=parent_node,
        critical=True
    )

    # City check (simple factual match)
    city_leaf = evaluator.add_leaf(
        id="city",
        desc="The city where the festival takes place (Milwaukee)",
        parent=loc_node,
        critical=True
    )
    provided_city = extracted.city or ""
    claim_city = (
        f"The city of Summerfest 2026 provided in the answer ('{provided_city}') "
        f"matches '{EXPECTED_CITY}'."
    )
    await evaluator.verify(
        claim=claim_city,
        node=city_leaf,
        additional_instruction="Allow case-insensitive matches and common variants like 'Milwaukee, WI'. If city is missing or not Milwaukee, mark incorrect."
    )

    # State check (simple factual match)
    state_leaf = evaluator.add_leaf(
        id="state",
        desc="The US state where the festival is located (Wisconsin)",
        parent=loc_node,
        critical=True
    )
    provided_state = extracted.state or ""
    claim_state = (
        f"The state of Summerfest 2026 provided in the answer ('{provided_state}') "
        f"matches '{EXPECTED_STATE}' (or its common abbreviation 'WI')."
    )
    await evaluator.verify(
        claim=claim_state,
        node=state_leaf,
        additional_instruction="Allow 'Wisconsin' and 'WI' as equivalent. If missing or a different state, mark incorrect."
    )

    # Reference URLs confirm the location
    ref_loc_leaf = evaluator.add_leaf(
        id="reference_url_location",
        desc="URL source confirming the festival location",
        parent=loc_node,
        critical=True
    )
    location_sources = extracted.location_sources or []
    claim_loc_support = "Summerfest takes place in Milwaukee, Wisconsin."
    await evaluator.verify(
        claim=claim_loc_support,
        node=ref_loc_leaf,
        sources=location_sources,
        additional_instruction=(
            "Verify that at least one provided URL explicitly states that Summerfest is held in Milwaukee, Wisconsin. "
            "Prefer official Summerfest or venue pages, or reputable publications. Reject irrelevant or generic pages."
        )
    )


async def build_and_verify_headliner(
    evaluator: Evaluator,
    parent_node,
    headliner: HeadlinerInfo,
    index_zero_based: int
) -> None:
    label_num = index_zero_based + 1
    # Parent node for a single headliner (non-critical group)
    headline_node = evaluator.add_parallel(
        id=f"headliner_{label_num}",
        desc=(
            "First American Family Insurance Amphitheater headliner" if label_num == 1 else
            "Second American Family Insurance Amphitheater headliner" if label_num == 2 else
            "Third American Family Insurance Amphitheater headliner"
        ),
        parent=parent_node,
        critical=False
    )

    # Artist name presence (critical in this group)
    artist_present_leaf = evaluator.add_custom_node(
        result=bool(headliner and headliner.artist and headliner.artist.strip()),
        id=f"artist_name_{label_num}",
        desc=(
            "Name of the first headlining artist" if label_num == 1 else
            "Name of the second headlining artist" if label_num == 2 else
            "Name of the third headlining artist"
        ),
        parent=headline_node,
        critical=True
    )

    # Reference URL checks that this artist is indeed an Amphitheater headliner
    ref_headliner_leaf = evaluator.add_leaf(
        id=f"reference_url_{label_num}",
        desc="URL source confirming this artist as a Summerfest 2026 headliner",
        parent=headline_node,
        critical=True
    )

    artist_name = (headliner.artist or "").strip()
    headliner_sources = headliner.sources or []

    claim_headliner_support = (
        f"{artist_name} is confirmed as an American Family Insurance Amphitheater headliner at Summerfest 2026."
    )
    await evaluator.verify(
        claim=claim_headliner_support,
        node=ref_headliner_leaf,
        sources=headliner_sources,
        additional_instruction=(
            "The page should explicitly confirm this artist is a Summerfest 2026 headliner for the American Family Insurance Amphitheater "
            "(main stage). Accept official Summerfest announcements, the official amphitheater or ticketing listing (e.g., Live Nation), "
            "or reputable music industry publications. Reject sources that refer to a different year, a different event, or a different stage."
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root rubric: parallel aggregation
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

    # Add ground truth context (from rubric expectations)
    evaluator.add_ground_truth({
        "expected_dates": {
            "start_date": EXPECTED_START_DATE,
            "end_date": EXPECTED_END_DATE
        },
        "expected_location": {
            "city": EXPECTED_CITY,
            "state": EXPECTED_STATE
        },
        "notes": "Dates and location are evaluated against these canonical values, with source-grounding checks."
    })

    # Extract information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_summerfest_2026(),
        template_class=Summerfest2026Extraction,
        extraction_name="summerfest_2026_extraction",
    )

    # Build subtrees according to the rubric
    # 1) Festival Dates (critical)
    await build_and_verify_festival_dates(evaluator, root, extracted)

    # 2) Festival Location (critical)
    await build_and_verify_location(evaluator, root, extracted)

    # 3) Three Amphitheater Headliners (non-critical groups)
    # Use only the first three headliners if more are provided; pad with empty placeholders if fewer.
    headliners = extracted.headliners[:3] if extracted.headliners else []
    while len(headliners) < 3:
        headliners.append(HeadlinerInfo())

    await build_and_verify_headliner(evaluator, root, headliners[0], 0)
    await build_and_verify_headliner(evaluator, root, headliners[1], 1)
    await build_and_verify_headliner(evaluator, root, headliners[2], 2)

    # Return structured evaluation summary
    return evaluator.get_summary()