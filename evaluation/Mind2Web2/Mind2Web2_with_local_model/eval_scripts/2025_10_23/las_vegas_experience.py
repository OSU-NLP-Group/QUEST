import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient

TASK_ID = "las_vegas_experience"
TASK_DESCRIPTION = """
I'm planning my three-day itinerary in Las Vegas using Airbnb Experiences. For each day, recommend at least one experience available on Airbnb. Across the three-day trip, my itinerary must include at least one Las Vegas Strip tour and at least two nature & outdoor experiences, with each day’s combined experience durations totaling at least 4 hours.

For each day, please recommend which Airbnb Experiences I should book, including the name, price per person, duration, and a direct link on Airbnb for each experience.
"""

EVAL_NOTES = ""
GROUND_TRUTH = {}



class TripOverview(BaseModel):
    """Model for checking if trip contains required experience types"""
    has_strip_tour: bool = Field(default=False,
                                 description="Whether the trip includes at least one Las Vegas Strip tour")
    has_two_nature_outdoor: bool = Field(default=False,
                                         description="Whether the trip includes at least two nature/outdoor experiences")


class DayExperienceNames(BaseModel):
    """Model for extracting experience names for a specific day"""
    day_number: int = Field(description="Day number (1, 2, or 3)")
    experience_names: List[str] = Field(default_factory=list, description="Names of experiences for this day")


class ExperienceDetails(BaseModel):
    """Model for detailed information about a single experience"""
    name: Optional[str] = Field(default=None, description="Name of the experience")
    price: Optional[str] = Field(default=None, description="Price per person")
    duration: Optional[str] = Field(default=None, description="Duration of the experience")
    url: Optional[str] = Field(default=None, description="Direct Airbnb link")


def prompt_extract_trip_overview() -> str:
    """Check if trip contains required experience types"""
    return """
    Review the entire 3-day Las Vegas itinerary and determine:

    1. has_strip_tour: Does the trip include at least one Las Vegas Strip tour?
       - Look for experiences with titles mentioning "Las Vegas Strip", "Strip Tour", "Vegas Strip Walking Tour", "Strip Food Tour", etc.
       - These are specific tours that focus on the Las Vegas Strip area

    2. has_two_nature_outdoor: Does the trip include at least two nature/outdoor experiences?
       - Look for activities mentioning: hiking, park, canyon, outdoor, nature, desert, mountain, trail, etc.
       - Count how many such experiences are included across all 3 days

    Set each boolean to true only if the requirement is clearly met.
    """


def prompt_extract_day_experience_names(day_number: int) -> str:
    """Extract experience names for a specific day"""
    return f"""
    For Day {day_number} of the Las Vegas trip, extract ONLY the names of all Airbnb experiences scheduled for this day.

    Look for sections labeled "Day {day_number}", "Day{day_number}", "{day_number}st day", "{day_number}nd day", "{day_number}rd day", or similar.

    Extract the exact experience names as listed.
    If no experiences are found for Day {day_number}, return an empty list.
    """


def prompt_extract_experience_details(experience_name: str, day_number: int) -> str:
    """Extract details for a specific experience"""
    return f"""
    Extract the complete details for the Airbnb experience named "{experience_name}" scheduled for Day {day_number}.

    Extract:
    - name: The exact name as it appears (should match "{experience_name}")
    - price: The price per person (keep exactly as written, including currency)
    - duration: The duration (keep exactly as written, e.g., "2 hours", "90 minutes")
    - url: The direct Airbnb experience link

    Look specifically in the Day {day_number} section.
    If any information is missing, set it to null.
    """


async def verify_trip_requirements(
        evaluator: Evaluator,
        parent_node: VerificationNode,
) -> None:
    """Verify trip-wide requirements as a critical node"""

    # First extract trip overview
    trip_overview = await evaluator.extract(
        prompt=prompt_extract_trip_overview(),
        template_class=TripOverview,
        extraction_name="trip_overview",
    )

    # Create a critical node for trip requirements
    trip_req_node = evaluator.add_leaf(
        id="trip_requirements",
        desc="Trip includes at least 1 Strip tour and 2 nature/outdoor experiences",
        parent=parent_node,
        critical=True
    )

    # Verify using simple_verify
    claim = f"""The trip itinerary includes:
    1. At least one Las Vegas Strip tour (a tour specifically focused on the Las Vegas Strip area)
    2. At least two nature/outdoor experiences (such as hiking, parks, canyons, or other outdoor activities)

    Based on extraction: Strip tour found: {trip_overview.has_strip_tour}, Two+ nature/outdoor found: {trip_overview.has_two_nature_outdoor}"""

    await evaluator.verify(
        claim=claim,
        node=trip_req_node,
        sources=None,  # Simple verify
        additional_instruction="Verify that BOTH requirements are met: at least 1 Strip tour AND at least 2 nature/outdoor experiences"
    )


async def verify_day_experiences(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        day_number: int,
) -> None:
    """Verify experiences for a specific day"""

    # Create day node (non-critical for partial credit)
    day_node = evaluator.add_parallel(
        id=f"day_{day_number}",
        desc=f"Day {day_number} experiences and requirements",
        parent=parent_node,
        critical=False
    )

    # Extract experience names for this day
    day_experiences = await evaluator.extract(
        prompt=prompt_extract_day_experience_names(day_number),
        template_class=DayExperienceNames,
        extraction_name=f"day_{day_number}_experience_names",
    )

    # Check if any experiences exist
    if not day_experiences.experience_names:
        # Add a failed node for no experiences
        no_exp_node = evaluator.add_custom_node(
            result=False,
            id=f"day_{day_number}_has_experiences",
            desc=f"Day {day_number} has no experiences scheduled",
            parent=day_node,
            critical=True
        )
        return

    # Extract details for each experience
    experience_details_list = []
    durations_list = []

    for exp_name in day_experiences.experience_names[:5]:  # Limit to first 5 to avoid excessive calls
        exp_details = await evaluator.extract(
            prompt=prompt_extract_experience_details(exp_name, day_number),
            template_class=ExperienceDetails,
            extraction_name=f"day_{day_number}_{exp_name[:30]}_details",
        )
        experience_details_list.append(exp_details)
        if exp_details.duration:
            durations_list.append(exp_details.duration)

    # Verify total duration (critical)
    duration_node = evaluator.add_leaf(
        id=f"day_{day_number}_duration_check",
        desc=f"Day {day_number} total duration is between 4-16 hours",
        parent=day_node,
        critical=True
    )

    durations_str = ", ".join(durations_list) if durations_list else "No durations found"
    claim = f"The combined duration of these experiences: [{durations_str}] totals at least 4 hours but no more than 16 hours (reasonable daily limit)"

    await evaluator.verify(
        claim=claim,
        node=duration_node,
        sources=None,
        additional_instruction="Add up all the durations and verify the total is >= 4 hours and <= 16 hours. Convert all durations to hours before adding."
    )

    # Verify each experience
    for idx, (exp_name, exp_details) in enumerate(zip(day_experiences.experience_names[:5], experience_details_list)):
        exp_node = evaluator.add_parallel(
            id=f"day_{day_number}_exp_{idx + 1}",
            desc=f"Experience: {exp_name[:50]}",
            parent=day_node,
            critical=True  # Each experience is critical
        )

        # Existence check - all info must exist
        info_exists = all([
            exp_details.name and exp_details.name.strip(),
            exp_details.price and exp_details.price.strip(),
            exp_details.duration and exp_details.duration.strip(),
            exp_details.url and exp_details.url.strip()
        ])

        existence_node = evaluator.add_custom_node(
            result=info_exists,
            id=f"day_{day_number}_exp_{idx + 1}_info_exists",
            desc="All required information exists (name, price, duration, URL)",
            parent=exp_node,
            critical=True
        )

        # URL verification
        url_verification_node = evaluator.add_leaf(
            id=f"day_{day_number}_exp_{idx + 1}_url_verify",
            desc="Airbnb experience page verification",
            parent=exp_node,
            critical=True
        )

        # Construct verification claim
        claim = f"""This is an Airbnb experience page that shows:
        1. The experience name: {exp_details.name}
        2. The price: {exp_details.price}
        3. The duration: {exp_details.duration}"""

        await evaluator.verify(
            claim=claim,
            node=url_verification_node,
            sources=exp_details.url,
            additional_instruction=f"For duration verification, the claim is valid if it matches either the duration stated in the description or the listed time slot."
        )


async def evaluate_answer(
        client: LLMClient,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Main evaluation function for Las Vegas experience task
    """

    # Initialize evaluator
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

    # First verify trip-wide requirements (critical)
    await verify_trip_requirements(evaluator, root)

    # Then verify each day (non-critical)
    for day_number in [1, 2, 3]:
        await verify_day_experiences(evaluator, root, day_number)

    return evaluator.get_summary()