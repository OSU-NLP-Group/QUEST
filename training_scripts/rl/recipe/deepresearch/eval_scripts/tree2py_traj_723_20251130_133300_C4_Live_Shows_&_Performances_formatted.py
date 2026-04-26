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
TASK_ID = "anuel_rhlm2_us_sept2025"
TASK_DESCRIPTION = (
    "Find information about Anuel AA's Real Hasta La Muerte 2 Tour stops in the United States during September 2025. "
    "Specifically, for his three concerts in Charlotte, North Carolina; Atlanta, Georgia; and Miami, Florida, provide "
    "the following for each city: the exact concert date (month, day, and year), the name of the concert venue, and "
    "the seating capacity of the venue. Organize your response by city and include reference URLs supporting your information."
)

EXPECTED_INFO = {
    "charlotte": {
        "city_label": "Charlotte, North Carolina",
        "expected_date": "September 14, 2025",
        "expected_venue": "Bojangles Coliseum",
        "capacity_min": 10000,
        "capacity_max": 11000,
    },
    "atlanta": {
        "city_label": "Atlanta, Georgia",
        "expected_date": "September 17, 2025",
        "expected_venue": "State Farm Arena",
        "capacity_min": 16500,
        "capacity_max": 17500,
    },
    "miami": {
        "city_label": "Miami, Florida",
        "expected_date": "September 19, 2025",
        "expected_venue": "Kaseya Center",
        "capacity_min": 19000,
        "capacity_max": 20000,
    },
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TourCityInfo(BaseModel):
    date: Optional[str] = None
    venue: Optional[str] = None
    capacity: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class TourExtraction(BaseModel):
    charlotte: Optional[TourCityInfo] = None
    atlanta: Optional[TourCityInfo] = None
    miami: Optional[TourCityInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tour_info() -> str:
    return """
    Extract the information provided in the answer for Anuel AA's Real Hasta La Muerte 2 Tour September 2025 U.S. stops, organized by city.

    For each of the following cities — Charlotte, Atlanta, and Miami — extract:
    - date: The exact concert date as stated in the answer (string; allow any human-readable format such as "September 14, 2025", "Sept 14, 2025", or "2025-09-14").
    - venue: The venue name as stated in the answer (string).
    - capacity: The seating capacity value as stated in the answer (string; keep as written, e.g., "10,200", "about 10k", "10,000–11,000").
    - references: A list of URL(s) explicitly included in the answer that are intended to support the information for this city. Include only valid URLs. If a URL is missing a protocol, prepend http://. Deduplicate URLs.

    Return a JSON object with keys "charlotte", "atlanta", and "miami", each being an object with the fields above.
    If any field for a city is not present in the answer, set it to null (or [] for references).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_city(info: Optional[TourCityInfo]) -> TourCityInfo:
    return info or TourCityInfo()


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_city(
    evaluator: Evaluator,
    parent_node,
    city_key: str,
    city_info: TourCityInfo,
    expected: Dict[str, Any],
) -> None:
    """
    Build verification nodes and checks for a single city.
    """
    city_label = expected["city_label"]
    expected_date = expected["expected_date"]
    expected_venue = expected["expected_venue"]
    cap_min = expected["capacity_min"]
    cap_max = expected["capacity_max"]

    # City-level container node (parallel aggregation; non-critical to allow partial credit city-by-city)
    city_node = evaluator.add_parallel(
        id=f"{city_key}_concert",
        desc=f"Provide the required information for the {city_label.split(',')[0]}, {city_label.split(',')[1].strip()} concert.",
        parent=parent_node,
        critical=False,
    )

    # 1) Date exact (must be stated exactly as expected in the answer)
    date_leaf = evaluator.add_leaf(
        id=f"{city_key}_date_exact",
        desc=f"Concert date is exactly {expected_date}.",
        parent=city_node,
        critical=True,
    )
    date_claim = (
        f"In the answer, for {city_label}, the Anuel AA Real Hasta La Muerte 2 Tour concert date is stated as "
        f"'{expected_date}'. Allow minor formatting variants (e.g., 'Sept' vs 'September', optional commas, '14th')."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        additional_instruction="Judge using only the answer text for whether this exact date is stated for the specified city.",
    )

    # 2) Venue exact (must be stated exactly as expected in the answer)
    venue_leaf = evaluator.add_leaf(
        id=f"{city_key}_venue_exact",
        desc=f"Concert venue is {expected_venue}.",
        parent=city_node,
        critical=True,
    )
    venue_claim = (
        f"In the answer, for {city_label}, the concert venue is stated as '{expected_venue}'. "
        f"Allow minor naming variants (e.g., leading 'The', city suffixes in parentheses)."
    )
    await evaluator.verify(
        claim=venue_claim,
        node=venue_leaf,
        additional_instruction="Judge using only the answer text for whether this exact venue name is stated for the specified city.",
    )

    # 3) Capacity presence (ensure capacity is actually stated in the answer)
    capacity_present = evaluator.add_custom_node(
        result=bool(city_info.capacity and str(city_info.capacity).strip()),
        id=f"{city_key}_capacity_provided",
        desc=f"Seating capacity is stated in the answer for {expected_venue}.",
        parent=city_node,
        critical=True,
    )

    # 4) Capacity range supported by sources (approximate within expected bounds)
    capacity_range_leaf = evaluator.add_leaf(
        id=f"{city_key}_capacity_range",
        desc=f"{expected_venue} seating capacity is approximately within {cap_min}–{cap_max} seats.",
        parent=city_node,
        critical=True,
    )
    cap_claim = (
        f"The typical/maximum concert seating capacity for {expected_venue} is approximately between {cap_min} and {cap_max} seats. "
        f"Values like {cap_min}, {cap_max}, or numbers in between (e.g., {cap_min + (cap_max - cap_min)//2}) should be considered within range. "
        f"If multiple configurations are listed (e.g., basketball/hockey vs concerts), consider the concert or maximum configuration capacity."
    )
    await evaluator.verify(
        claim=cap_claim,
        node=capacity_range_leaf,
        sources=city_info.references if city_info.references else None,
        additional_instruction=(
            "Verify the capacity range using the provided URLs for this city. "
            "If no relevant capacity number can be found on any URL, this should fail."
        ),
        extra_prerequisites=[capacity_present],  # Ensure capacity was actually stated in the answer
    )

    # 5) References present (at least one URL included in the answer for this city)
    refs_present = evaluator.add_custom_node(
        result=bool(city_info.references and len(city_info.references) > 0),
        id=f"{city_key}_references",
        desc=f"Includes at least one reference URL supporting the {city_label.split(',')[0]} date/venue/capacity information.",
        parent=city_node,
        critical=True,
    )
    # Note: The specific support by URL is assessed in (4) for capacity; date/venue correctness is checked in-answer.
    # The rubric requires at least one URL; this node enforces presence of at least one reference per city.


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Anuel AA RLH 2 Tour September 2025 US stops (Charlotte, Atlanta, Miami).
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel across top-level criteria
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
    # IMPORTANT: Set root as non-critical to comply with framework rule that a critical parent cannot have non-critical children
    root.critical = False

    # Extract structured information from the answer
    extracted: TourExtraction = await evaluator.extract(
        prompt=prompt_extract_tour_info(),
        template_class=TourExtraction,
        extraction_name="tour_extraction_by_city",
    )

    # Organization-by-city check (critical leaf)
    org_leaf = evaluator.add_leaf(
        id="response_organization_by_city",
        desc="Response is organized by city (Charlotte, Atlanta, Miami) with clearly separated sections/headings.",
        parent=root,
        critical=True,
    )
    org_claim = (
        "The answer is organized by city with clearly separated sections or headings for 'Charlotte', 'Atlanta', and 'Miami'. "
        "The order may vary, but each city should be visibly separated (e.g., headings, bold labels, or clear subsections)."
    )
    await evaluator.verify(
        claim=org_claim,
        node=org_leaf,
        additional_instruction="Judge this using only the answer text. Do not require a specific formatting style; any clear separation by city is acceptable.",
    )

    # City verifications
    charlotte_info = _safe_city(extracted.charlotte)
    atlanta_info = _safe_city(extracted.atlanta)
    miami_info = _safe_city(extracted.miami)

    # Charlotte
    await verify_city(
        evaluator=evaluator,
        parent_node=root,
        city_key="charlotte",
        city_info=charlotte_info,
        expected=EXPECTED_INFO["charlotte"],
    )

    # Atlanta
    await verify_city(
        evaluator=evaluator,
        parent_node=root,
        city_key="atlanta",
        city_info=atlanta_info,
        expected=EXPECTED_INFO["atlanta"],
    )

    # Miami
    await verify_city(
        evaluator=evaluator,
        parent_node=root,
        city_key="miami",
        city_info=miami_info,
        expected=EXPECTED_INFO["miami"],
    )

    # Ground truth info for transparency
    evaluator.add_ground_truth(
        {
            "expected": EXPECTED_INFO,
            "notes": "Expected dates, venues, and approximate concert seating capacity ranges per city."
        },
        gt_type="expected_info"
    )

    return evaluator.get_summary()