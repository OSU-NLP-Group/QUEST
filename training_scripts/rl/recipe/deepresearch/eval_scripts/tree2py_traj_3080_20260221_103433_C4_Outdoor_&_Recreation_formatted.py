import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_theme_breeze_np"
TASK_DESCRIPTION = (
    "A family is planning a California vacation and wants to visit a theme park with world-record roller coasters while also accessing a national park for day hiking. "
    "They are flying on Breeze Airways to keep costs down. Identify the California theme park that holds the world record for most roller coasters (20 total) in a single US park "
    "and is home to the world's tallest single-rail roller coaster. The park must be within 50 miles of a city served by Breeze Airways. Additionally, from that same Breeze Airways "
    "destination city, there must be a national park accessible within 2.5 hours of driving where day hiking is allowed without requiring advance permits. Provide: "
    "(1) the name of the theme park, (2) the specifications of its tallest single-rail roller coaster including height in feet, length in feet, and top speed in mph, "
    "(3) the nearby Breeze Airways destination city, and (4) the accessible national park."
)

# Thresholds for specs (as per rubric)
MIN_HEIGHT_FT = 131
MIN_LENGTH_FT = 3300
MIN_SPEED_MPH = 58


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class SolutionExtraction(BaseModel):
    # Theme park and general sources
    theme_park_name: Optional[str] = None
    theme_park_sources: List[str] = Field(default_factory=list)

    # Single-rail roller coaster specs
    single_rail_name: Optional[str] = None
    single_rail_height_ft: Optional[str] = None
    single_rail_length_ft: Optional[str] = None
    single_rail_speed_mph: Optional[str] = None
    single_rail_sources: List[str] = Field(default_factory=list)

    # Breeze city
    breeze_city: Optional[str] = None
    breeze_airport: Optional[str] = None
    breeze_city_sources: List[str] = Field(default_factory=list)

    # Proximity evidence for park <-> Breeze city
    proximity_sources: List[str] = Field(default_factory=list)

    # National park and evidence
    national_park_name: Optional[str] = None
    national_park_sources: List[str] = Field(default_factory=list)

    # Drive time evidence and permit policy evidence
    national_park_drive_sources: List[str] = Field(default_factory=list)
    permit_policy_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_solution() -> str:
    return """
    Extract the structured information provided by the answer for this California trip planning task. Return a JSON object with the following fields:

    1) theme_park_name: The exact name of the theme park identified.
    2) theme_park_sources: A list of all URLs cited in the answer that support information about the theme park (e.g., park location, coaster count, record claims).

    3) single_rail_name: The exact name of the world's tallest single-rail roller coaster at the park (if provided).
    4) single_rail_height_ft: The height in feet as stated (extract as a plain string, e.g., "131").
    5) single_rail_length_ft: The length in feet as stated (plain string).
    6) single_rail_speed_mph: The top speed in mph as stated (plain string).
    7) single_rail_sources: A list of URLs cited in the answer that support the single-rail coaster specs or 'world's tallest' claim.

    8) breeze_city: The name of the Breeze Airways destination city used for proximity and national park access.
    9) breeze_airport: The airport name/code if provided (optional).
    10) breeze_city_sources: A list of URLs cited that indicate Breeze Airways serves that city/airport.

    11) proximity_sources: A list of URLs cited that support the proximity between the theme park and the Breeze Airways destination city (e.g., mapping pages, official info with distances, travel guides).

    12) national_park_name: The name of the national park identified for day hiking.
    13) national_park_sources: A list of URLs cited that describe the park and day hiking context.
    14) national_park_drive_sources: A list of URLs cited that support the claim that the park is accessible within 2.5 hours of driving from the Breeze city.
    15) permit_policy_sources: A list of URLs cited that support the claim that day hiking is allowed without requiring advance permits.

    IMPORTANT:
    - Extract URLs ONLY if they are explicitly present in the answer (plain URLs or markdown links). Do not invent URLs.
    - If any field is not mentioned, set it to null (for single strings) or [] (for lists).
    - Keep values as strings to maximize compatibility (do not coerce to numbers).
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _safe_str(v: Optional[str]) -> str:
    return v if (v is not None and str(v).strip() != "") else ""


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
    Build the verification tree for the California theme park / Breeze city / national park task,
    extract structured data from the answer, and run evidence-based checks.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root is parallel per rubric
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

    # Extract structured info from the answer
    extracted: SolutionExtraction = await evaluator.extract(
        prompt=prompt_extract_solution(),
        template_class=SolutionExtraction,
        extraction_name="solution_extraction",
    )

    # Record useful configuration/thresholds for transparency
    evaluator.add_custom_info(
        info={
            "min_height_ft_required": MIN_HEIGHT_FT,
            "min_length_ft_required": MIN_LENGTH_FT,
            "min_speed_mph_required": MIN_SPEED_MPH,
        },
        info_type="thresholds",
    )

    # ------------------------------ #
    # Create existence (critical)    #
    # ------------------------------ #
    # 1. Theme park name provided (CRITICAL)
    evaluator.add_custom_node(
        result=bool(_safe_str(extracted.theme_park_name)),
        id="theme_park_name_provided",
        desc="The solution provides the name of a specific California theme park",
        parent=root,
        critical=True,
    )

    # 2. Breeze city provided (CRITICAL)
    evaluator.add_custom_node(
        result=bool(_safe_str(extracted.breeze_city)),
        id="breeze_city_provided",
        desc="The solution identifies a specific city served by Breeze Airways",
        parent=root,
        critical=True,
    )

    # 3. National park provided (CRITICAL)
    evaluator.add_custom_node(
        result=bool(_safe_str(extracted.national_park_name)),
        id="national_park_provided",
        desc="The solution identifies a specific national park",
        parent=root,
        critical=True,
    )

    # Prepare values
    park_name = _safe_str(extracted.theme_park_name)
    breeze_city = _safe_str(extracted.breeze_city)
    np_name = _safe_str(extracted.national_park_name)
    single_rail_name = _safe_str(extracted.single_rail_name)
    height_ft = _safe_str(extracted.single_rail_height_ft)
    length_ft = _safe_str(extracted.single_rail_length_ft)
    speed_mph = _safe_str(extracted.single_rail_speed_mph)

    # ------------------------------ #
    # Build leaf nodes for verifications
    # ------------------------------ #
    # California location (CRITICAL)
    node_california = evaluator.add_leaf(
        id="california_location",
        desc="The identified theme park is located in California",
        parent=root,
        critical=True,
    )
    claim_california = f"The theme park {park_name} is located in California."
    # Verify by URLs using theme_park_sources
    await evaluator.verify(
        claim=claim_california,
        node=node_california,
        sources=extracted.theme_park_sources,
        additional_instruction="Confirm the park's location is in California using the cited sources (official park page, Wikipedia, etc.).",
    )

    # Roller coaster count: 20 (NON-CRITICAL)
    node_count20 = evaluator.add_leaf(
        id="roller_coaster_count_20",
        desc="The theme park has 20 roller coasters",
        parent=root,
        critical=False,
    )
    claim_count20 = f"The theme park {park_name} has 20 roller coasters."
    await evaluator.verify(
        claim=claim_count20,
        node=node_count20,
        sources=extracted.theme_park_sources,
        additional_instruction="Check that the cited sources explicitly state the total number of roller coasters is 20.",
    )

    # US record: most coasters in a single US theme park (NON-CRITICAL)
    node_us_record = evaluator.add_leaf(
        id="us_record_most_coasters",
        desc="The theme park holds the world record for most roller coasters in one US theme park",
        parent=root,
        critical=False,
    )
    claim_us_record = f"{park_name} holds the record for most roller coasters in a single U.S. theme park."
    await evaluator.verify(
        claim=claim_us_record,
        node=node_us_record,
        sources=extracted.theme_park_sources,
        additional_instruction="The sources should explicitly indicate the record status for the most roller coasters in a single U.S. park.",
    )

    # World's tallest single-rail coaster present (NON-CRITICAL)
    node_tallest_single_rail = evaluator.add_leaf(
        id="has_tallest_single_rail",
        desc="The theme park has the world's tallest single-rail roller coaster",
        parent=root,
        critical=False,
    )
    claim_tallest_single_rail = (
        f"The theme park {park_name} has the world's tallest single-rail roller coaster named {single_rail_name}."
    )
    await evaluator.verify(
        claim=claim_tallest_single_rail,
        node=node_tallest_single_rail,
        sources=extracted.single_rail_sources,
        additional_instruction="Verify that the coaster is explicitly described as the world's tallest single-rail roller coaster.",
    )

    # Single-rail height spec ≥ 131 ft (NON-CRITICAL)
    node_height_spec = evaluator.add_leaf(
        id="single_rail_height_spec",
        desc="The solution provides the single-rail coaster's height specification, which is at least 131 feet",
        parent=root,
        critical=False,
    )
    claim_height_spec = (
        f"The single-rail coaster {single_rail_name} has a height of {height_ft} feet, which is at least {MIN_HEIGHT_FT} feet."
    )
    await evaluator.verify(
        claim=claim_height_spec,
        node=node_height_spec,
        sources=extracted.single_rail_sources,
        additional_instruction="Confirm the stated height from the source. Minor rounding or formatting differences are acceptable. Ensure the value meets or exceeds 131 ft.",
    )

    # Single-rail length spec ≥ 3,300 ft (NON-CRITICAL)
    node_length_spec = evaluator.add_leaf(
        id="single_rail_length_spec",
        desc="The solution provides the single-rail coaster's length specification, which is at least 3,300 feet",
        parent=root,
        critical=False,
    )
    claim_length_spec = (
        f"The single-rail coaster {single_rail_name} has a length of {length_ft} feet, which is at least {MIN_LENGTH_FT} feet."
    )
    await evaluator.verify(
        claim=claim_length_spec,
        node=node_length_spec,
        sources=extracted.single_rail_sources,
        additional_instruction="Confirm the stated length from the source and that it is ≥ 3,300 ft. Minor rounding differences are acceptable.",
    )

    # Single-rail speed spec ≥ 58 mph (NON-CRITICAL)
    node_speed_spec = evaluator.add_leaf(
        id="single_rail_speed_spec",
        desc="The solution provides the single-rail coaster's speed specification, which is at least 58 mph",
        parent=root,
        critical=False,
    )
    claim_speed_spec = (
        f"The single-rail coaster {single_rail_name} has a top speed of {speed_mph} mph, which is at least {MIN_SPEED_MPH} mph."
    )
    await evaluator.verify(
        claim=claim_speed_spec,
        node=node_speed_spec,
        sources=extracted.single_rail_sources,
        additional_instruction="Confirm the stated top speed from the sources and that it is ≥ 58 mph. Minor rounding differences are acceptable.",
    )

    # Breeze city proximity (NON-CRITICAL)
    node_breeze_proximity = evaluator.add_leaf(
        id="breeze_city_proximity",
        desc="The theme park is within 50 miles of the identified Breeze Airways destination city",
        parent=root,
        critical=False,
    )
    claim_breeze_proximity = (
        f"The theme park {park_name} is within 50 miles of the Breeze Airways destination city {breeze_city}."
    )
    await evaluator.verify(
        claim=claim_breeze_proximity,
        node=node_breeze_proximity,
        sources=extracted.proximity_sources,
        additional_instruction=(
            "Use the cited distance or mapping sources to confirm that the driving distance between the city and the park is ≤ 50 miles. "
            "Allow reasonable approximate phrasing (e.g., 'about 35 miles')."
        ),
    )

    # National park drive time ≤ 2.5 hours (NON-CRITICAL)
    node_np_drive = evaluator.add_leaf(
        id="national_park_drive_time",
        desc="The identified national park is accessible within 2.5 hours drive from the Breeze Airways destination city",
        parent=root,
        critical=False,
    )
    claim_np_drive = (
        f"The national park {np_name} is accessible within 2.5 hours (150 minutes) of driving from {breeze_city}."
    )
    await evaluator.verify(
        claim=claim_np_drive,
        node=node_np_drive,
        sources=extracted.national_park_drive_sources,
        additional_instruction=(
            "Confirm via the cited route/distance/time sources that typical driving time from the Breeze city to the park is ≤ 2.5 hours. "
            "Consider reasonable conditions; ignore non-driving segments (e.g., ferries) unless explicitly part of the drive time source."
        ),
    )

    # National park day hiking without advance permits (NON-CRITICAL)
    node_np_permit = evaluator.add_leaf(
        id="national_park_no_permit",
        desc="The identified national park allows day hiking without requiring advance permits",
        parent=root,
        critical=False,
    )
    claim_np_permit = f"Day hiking at {np_name} does not require advance permits."
    await evaluator.verify(
        claim=claim_np_permit,
        node=node_np_permit,
        sources=extracted.permit_policy_sources or extracted.national_park_sources,
        additional_instruction=(
            "Focus on general day hiking access (not overnight/backcountry/special use). The sources should indicate that typical day hikes do not require advance permits."
        ),
    )

    # Return structured summary with verification tree
    return evaluator.get_summary()