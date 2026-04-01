import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "solar_eclipses"
TASK_DESCRIPTION = """
Your task is to gather detailed information about two total solar eclipses:

- Identify the **first total solar eclipse after January 1, 1900**.
- Identify the **first total solar eclipse after January 1, 2050**.

### For each of these two eclipses, provide the following details:

1. **Exact Date** of the eclipse (month, day, year).
2. **Greatest Observation Point**:
   - Include exact geographical coordinates (latitude and longitude).
3. **Longest Duration of Totality** (expressed in minutes and seconds).
4. **Saros series number** to which each eclipse belongs.

### Background Information

- A **total solar eclipse** occurs when the Moon completely blocks the Sun, briefly turning day into night within a narrow region on Earth's surface.
- The **Greatest Observation Point** (also called Greatest Eclipse Point) is the geographic location where the eclipse achieves the longest totality duration.
- **Duration of Totality** refers to how long the Sun is completely covered, typically measured in minutes and seconds.
- A **Saros series** is an approximately 18-year, 11-day, 8-hour cycle used to predict eclipses. Eclipses within the same Saros series share similar characteristics and geographic patterns but shift slightly in position with each recurrence. Each Saros series is identified by a unique number.
"""

# Ground truth answers for date verification
GROUND_TRUTH_DATES = {
    "1900": "1900 May 28",
    "2050": "2050 May 20"
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EclipseInfo(BaseModel):
    """Model for a single eclipse's information"""
    date: Optional[str] = None
    observation_point: Optional[str] = None
    coordinates: Optional[str] = None
    duration: Optional[str] = None
    saros_number: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)  # Added source URLs field


class SourceURLs(BaseModel):
    """Model for extracting source URLs"""
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_eclipse_1900() -> str:
    return """
    Extract information about the first total solar eclipse after January 1, 1900 mentioned in the answer.

    Extract:
    - The exact date (month, day, year)
    - The greatest observation point description and geographical coordinates (latitude and longitude)
    - The duration of totality (in minutes and seconds)
    - The Saros series number
    - Source URLs that specifically support information about this 1900 eclipse

    If any information is missing, set the value to null.
    Structure the output as follows:
    {
        "date": "Month Day, Year or YYYY Month DD format",
        "observation_point": "Description of greatest observation point",
        "coordinates": "Latitude, Longitude",
        "duration": "X minutes Y seconds",
        "saros_number": "Number",
        "source_urls": ["url1", "url2", ...] // URLs that contain information about this specific eclipse
    }
    """


def prompt_extract_eclipse_2050() -> str:
    return """
    Extract information about the first total solar eclipse after January 1, 2050 mentioned in the answer.

    Extract:
    - The exact date (month, day, year)
    - The greatest observation point description and geographical coordinates (latitude and longitude)
    - The duration of totality (in minutes and seconds)
    - The Saros series number
    - Source URLs that specifically support information about this 2050 eclipse

    If any information is missing, set the value to null.
    Structure the output as follows:
    {
        "date": "Month Day, Year or YYYY Month DD format",
        "observation_point": "Description of greatest observation point",
        "coordinates": "Latitude, Longitude",
        "duration": "X minutes Y seconds",
        "saros_number": "Number",
        "source_urls": ["url1", "url2", ...] // URLs that contain information about this specific eclipse
    }
    """


# --------------------------------------------------------------------------- #
# Individual verification functions                                           #
# --------------------------------------------------------------------------- #
async def verify_date_correctness(
        evaluator: Evaluator,
        parent_node,
        eclipse_info: EclipseInfo,
        eclipse_identifier: str
):
    """
    Verify the correctness of the eclipse date against ground truth.
    This is a critical node - the date must be exactly correct.
    """
    date_node = evaluator.add_leaf(
        id=f"{eclipse_identifier}_date_correctness",
        desc=f"Verify if the extracted date matches the ground truth for the first total solar eclipse after January 1, {eclipse_identifier}",
        parent=parent_node,
        critical=True,
    )

    if eclipse_info and eclipse_info.date:
        ground_truth = GROUND_TRUTH_DATES[eclipse_identifier]
        claim = f"The date '{eclipse_info.date}' is the same as the date '{ground_truth}' ."

        additional_instruction = f"""
        The ground truth date for the first total solar eclipse after January 1, {eclipse_identifier} is {ground_truth}. 
        So, you just need to verify if the extracted date here corresponds to exactly this date.
        Consider different date formats (e.g., "May 28, 1900" vs "1900 May 28") as equivalent.
        The extracted date must refer to the same calendar date as the ground truth.
        """

        await evaluator.verify(
            claim=claim,
            node=date_node,
            additional_instruction=additional_instruction
        )


async def verify_observation_point(
        evaluator: Evaluator,
        eclipse_node,
        eclipse_info: EclipseInfo,
        all_urls: List[str],
        eclipse_identifier: str
):
    """
    Verify observation point information with existence check and URL substantiation.
    """
    observation_node = evaluator.add_parallel(
        id=f"{eclipse_identifier}_observation_point",
        desc=f"Verification of greatest observation point information for the {eclipse_identifier} eclipse",
        parent=eclipse_node,
        critical=False,
    )

    # Add existence check (critical)
    evaluator.add_custom_node(
        result=bool(eclipse_info and eclipse_info.coordinates),
        id=f"{eclipse_identifier}_observation_exists",
        desc=f"Check if observation point information is provided for the {eclipse_identifier} eclipse",
        parent=observation_node,
        critical=True
    )

    # Add URL substantiation verification
    substantiated_node = evaluator.add_leaf(
        id=f"{eclipse_identifier}_observation_substantiated",
        desc=f"Verify if the greatest observation point information for the {eclipse_identifier} eclipse is substantiated by source URLs",
        parent=observation_node,
        critical=True,
    )

    ground_truth_date = GROUND_TRUTH_DATES[eclipse_identifier]
    claim_parts = []
    
    if eclipse_info.observation_point:
        claim_parts.append(
            f"The greatest observation point for the total solar eclipse on {ground_truth_date} is {eclipse_info.observation_point}")
    
    if eclipse_info.coordinates:
        claim_parts.append(
            f"The coordinates of the greatest eclipse point for the eclipse on {ground_truth_date} are {eclipse_info.coordinates}")
    
    claim = ". ".join(claim_parts)
    
    await evaluator.verify(
        claim=claim,
        node=substantiated_node,
        sources=all_urls
    )


async def verify_duration(
        evaluator: Evaluator,
        eclipse_node,
        eclipse_info: EclipseInfo,
        all_urls: List[str],
        eclipse_identifier: str
):
    """
    Verify duration information with existence check and URL substantiation.
    """
    duration_node = evaluator.add_parallel(
        id=f"{eclipse_identifier}_duration",
        desc=f"Verification of duration of totality for the {eclipse_identifier} eclipse",
        parent=eclipse_node,
        critical=False,
    )

    # Add existence check (critical)
    evaluator.add_custom_node(
        result=bool(eclipse_info and eclipse_info.duration),
        id=f"{eclipse_identifier}_duration_exists",
        desc=f"Check if duration of totality is provided for the {eclipse_identifier} eclipse",
        parent=duration_node,
        critical=True
    )

    # Add URL substantiation verification
    substantiated_node = evaluator.add_leaf(
        id=f"{eclipse_identifier}_duration_substantiated",
        desc=f"Verify if the duration of totality for the {eclipse_identifier} eclipse is substantiated by source URLs",
        parent=duration_node,
        critical=True,
    )

    ground_truth_date = GROUND_TRUTH_DATES[eclipse_identifier]
    claim = f"The duration of totality for the total solar eclipse on {ground_truth_date} is {eclipse_info.duration}"
    
    await evaluator.verify(
        claim=claim,
        node=substantiated_node,
        sources=all_urls,
        additional_instruction="allow different formats for duration (e.g., '2 minutes 30 seconds' vs '2:30') or any other reasonable representation and variations."
    )


async def verify_saros(
        evaluator: Evaluator,
        eclipse_node,
        eclipse_info: EclipseInfo,
        all_urls: List[str],
        eclipse_identifier: str
):
    """
    Verify Saros series information with existence check and URL substantiation.
    """
    saros_node = evaluator.add_parallel(
        id=f"{eclipse_identifier}_saros",
        desc=f"Verification of Saros series number for the {eclipse_identifier} eclipse",
        parent=eclipse_node,
        critical=False,
    )

    # Add existence check (critical)
    evaluator.add_custom_node(
        result=bool(eclipse_info and eclipse_info.saros_number),
        id=f"{eclipse_identifier}_saros_exists",
        desc=f"Check if Saros series number is provided for the {eclipse_identifier} eclipse",
        parent=saros_node,
        critical=True
    )

    # Add URL substantiation verification
    substantiated_node = evaluator.add_leaf(
        id=f"{eclipse_identifier}_saros_substantiated",
        desc=f"Verify if the Saros series number for the {eclipse_identifier} eclipse is substantiated by source URLs",
        parent=saros_node,
        critical=True,
    )

    ground_truth_date = GROUND_TRUTH_DATES[eclipse_identifier]
    claim = f"The Saros series number for the total solar eclipse on {ground_truth_date} is {eclipse_info.saros_number}"
    
    await evaluator.verify(
        claim=claim,
        node=substantiated_node,
        sources=all_urls
    )


# --------------------------------------------------------------------------- #
# Complete eclipse verification                                               #
# --------------------------------------------------------------------------- #
async def verify_eclipse(
        evaluator: Evaluator,
        parent_node,
        eclipse_info: EclipseInfo,
        eclipse_identifier: str
):
    """
    Complete verification for a single eclipse following the specified structure.
    Now uses eclipse-specific URLs from the extracted info.
    """
    eclipse_node = evaluator.add_parallel(
        id=f"eclipse_{eclipse_identifier}",
        desc=f"Verification of the {eclipse_identifier} eclipse information",
        parent=parent_node,
        critical=False,
    )

    # Get eclipse-specific URLs
    eclipse_urls = eclipse_info.source_urls if eclipse_info else []

    # 1) Date correctness verification (critical)
    await verify_date_correctness(
        evaluator=evaluator,
        parent_node=eclipse_node,
        eclipse_info=eclipse_info,
        eclipse_identifier=eclipse_identifier
    )

    # 2) Observation point verification
    await verify_observation_point(
        evaluator=evaluator,
        eclipse_node=eclipse_node,
        eclipse_info=eclipse_info,
        all_urls=eclipse_urls,  # Use eclipse-specific URLs
        eclipse_identifier=eclipse_identifier
    )

    # 3) Duration verification
    await verify_duration(
        evaluator=evaluator,
        eclipse_node=eclipse_node,
        eclipse_info=eclipse_info,
        all_urls=eclipse_urls,  # Use eclipse-specific URLs
        eclipse_identifier=eclipse_identifier
    )

    # 4) Saros verification
    await verify_saros(
        evaluator=evaluator,
        eclipse_node=eclipse_node,
        eclipse_info=eclipse_info,
        all_urls=eclipse_urls,  # Use eclipse-specific URLs
        eclipse_identifier=eclipse_identifier
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: LLMClient,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer to the solar eclipses task.
    Now extracts and uses eclipse-specific URLs for verification.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        agent_name=agent_name,
        answer_name=answer_name,
        task_description=TASK_DESCRIPTION,
        client=client,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract information about each eclipse (including their specific URLs)
    eclipse_1900_info = await evaluator.extract(
        prompt=prompt_extract_eclipse_1900(),
        template_class=EclipseInfo,
        extraction_name="eclipse_1900_info"
    )

    eclipse_2050_info = await evaluator.extract(
        prompt=prompt_extract_eclipse_2050(),
        template_class=EclipseInfo,
        extraction_name="eclipse_2050_info"
    )

    # Add ground truth information
    evaluator.add_ground_truth({
        "ground_truth_dates": GROUND_TRUTH_DATES
    })

    # Add custom info about extracted URLs for debugging/transparency
    evaluator.add_custom_info({
        "1900_urls": eclipse_1900_info.source_urls if eclipse_1900_info else [],
        "2050_urls": eclipse_2050_info.source_urls if eclipse_2050_info else [],
    }, "extracted_eclipse_urls")

    # Verify the 1900 eclipse with its specific URLs
    await verify_eclipse(
        evaluator=evaluator,
        parent_node=evaluator.root,
        eclipse_info=eclipse_1900_info,
        eclipse_identifier="1900"
    )

    # Verify the 2050 eclipse with its specific URLs
    await verify_eclipse(
        evaluator=evaluator,
        parent_node=evaluator.root,
        eclipse_info=eclipse_2050_info,
        eclipse_identifier="2050"
    )

    # Return structured result using evaluator's summary
    return evaluator.get_summary()