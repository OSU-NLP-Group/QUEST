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
TASK_ID = "europa_mission_verification_2024"
TASK_DESCRIPTION = """
NASA launched a spacecraft mission to Jupiter's moon Europa from Kennedy Space Center in October 2024. What was the exact launch date, what launch vehicle was used, and what is the spacecraft's solar array span when fully deployed?
"""

# Expected ground-truth claims (used for verification wording)
EXPECTED_LAUNCH_DATE = "October 14, 2024"
EXPECTED_LAUNCH_SITE = "Kennedy Space Center in Florida"
EXPECTED_LAUNCH_VEHICLE = "SpaceX Falcon Heavy rocket"
EXPECTED_SOLAR_ARRAY_SPAN = "more than 100 feet (approximately 30 meters)"
EXPECTED_JUPITER_ARRIVAL = "April 2030"
EXPECTED_MISSION_TARGET = "Jupiter's moon Europa"
EXPECTED_LARGEST_PLANETARY = "NASA's largest planetary mission spacecraft"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EuropaMissionExtraction(BaseModel):
    """
    Extracted mission details from the agent's answer.
    Prefer strings for flexibility; URLs should be explicit.
    """
    mission_name: Optional[str] = None
    mission_target: Optional[str] = None
    launch_date: Optional[str] = None
    launch_site: Optional[str] = None
    launch_vehicle: Optional[str] = None
    solar_array_span: Optional[str] = None
    jupiter_arrival_timing: Optional[str] = None
    largest_planetary_mission_note: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_europa_mission() -> str:
    return """
    Extract the mission details stated in the answer related to NASA's mission to Jupiter's moon Europa.
    Return a JSON object with the following fields (use null if a field is not present in the answer):

    - mission_name: The name of the spacecraft/mission (e.g., "Europa Clipper"), if mentioned.
    - mission_target: The celestial target (e.g., "Jupiter's moon Europa"), if mentioned.
    - launch_date: The exact launch date (e.g., "October 14, 2024"), if provided.
    - launch_site: The launch site/location (e.g., "Kennedy Space Center in Florida"), if provided.
    - launch_vehicle: The name of the launch vehicle (e.g., "SpaceX Falcon Heavy rocket"), if provided.
    - solar_array_span: The described span of the spacecraft's solar arrays when fully deployed (e.g., "more than 100 feet (approximately 30 meters)"), if provided.
    - jupiter_arrival_timing: The scheduled arrival timing at Jupiter (e.g., "April 2030"), if provided.
    - largest_planetary_mission_note: Any statement indicating the spacecraft is NASA's largest planetary mission spacecraft (copy the phrase or a concise paraphrase), if present.
    - sources: An array of all URLs explicitly cited in the answer that support the above details.
      • Extract actual URLs only (plain URLs or markdown links). If no URLs are present, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    parent_node,
    extracted: EuropaMissionExtraction,
) -> None:
    """
    Build the verification tree under the main Europa_Mission_Verification node
    and run all leaf verifications in parallel.
    """

    # Create the main Europa mission verification node (critical, parallel)
    europa_node = evaluator.add_parallel(
        id="Europa_Mission_Verification",
        desc="Verify the mission and required attributes per the question and the provided constraints.",
        parent=parent_node,
        critical=True,  # All child checks are essential
    )

    # Helper: Additional instruction common policy
    # If sources are missing, we instruct the judge to mark claims as not supported.
    base_additional_instruction = (
        "Use the provided source URL(s) to verify the claim explicitly. "
        "Allow minor wording variations and abbreviations (e.g., 'KSC' for Kennedy Space Center). "
        "IMPORTANT: If there are no valid source URLs provided in the answer, treat the claim as NOT SUPPORTED and return Incorrect."
    )

    # Prepare leaf nodes
    mission_leaf = evaluator.add_leaf(
        id="Mission_Studies_Europa",
        desc="The mission is specifically designed to study Jupiter's moon Europa.",
        parent=europa_node,
        critical=True,
    )
    launch_date_leaf = evaluator.add_leaf(
        id="Launch_Date",
        desc=f"The mission launch date is {EXPECTED_LAUNCH_DATE}.",
        parent=europa_node,
        critical=True,
    )
    launch_site_leaf = evaluator.add_leaf(
        id="Launch_Site",
        desc=f"The launch occurred from {EXPECTED_LAUNCH_SITE}.",
        parent=europa_node,
        critical=True,
    )
    launch_vehicle_leaf = evaluator.add_leaf(
        id="Launch_Vehicle",
        desc=f"The launch vehicle used was a {EXPECTED_LAUNCH_VEHICLE}.",
        parent=europa_node,
        critical=True,
    )
    solar_array_leaf = evaluator.add_leaf(
        id="Solar_Array_Span",
        desc=f"When fully deployed, the spacecraft's solar arrays span {EXPECTED_SOLAR_ARRAY_SPAN}.",
        parent=europa_node,
        critical=True,
    )
    jupiter_arrival_leaf = evaluator.add_leaf(
        id="Jupiter_Arrival_Timing",
        desc=f"The mission is scheduled to arrive at Jupiter in {EXPECTED_JUPITER_ARRIVAL}.",
        parent=europa_node,
        critical=True,
    )
    largest_planetary_leaf = evaluator.add_leaf(
        id="Largest_Planetary_Mission_Spacecraft",
        desc="The spacecraft is NASA's largest planetary mission spacecraft.",
        parent=europa_node,
        critical=True,
    )

    # Build claims and run them in parallel
    claims_and_sources = [
        (
            f"The mission is specifically designed to study {EXPECTED_MISSION_TARGET}.",
            extracted.sources,
            mission_leaf,
            base_additional_instruction,
        ),
        (
            f"The mission launched on {EXPECTED_LAUNCH_DATE}.",
            extracted.sources,
            launch_date_leaf,
            base_additional_instruction
            + " Accept minor date formatting variations (e.g., '14 October 2024').",
        ),
        (
            f"The mission launched from {EXPECTED_LAUNCH_SITE}.",
            extracted.sources,
            launch_site_leaf,
            base_additional_instruction
            + " 'Kennedy Space Center in Florida' may also appear as 'KSC, Florida'.",
        ),
        (
            f"The launch vehicle was a {EXPECTED_LAUNCH_VEHICLE}.",
            extracted.sources,
            launch_vehicle_leaf,
            base_additional_instruction
            + " Accept phrasing like 'SpaceX Falcon Heavy' or 'Falcon Heavy rocket'.",
        ),
        (
            f"When fully deployed, the spacecraft's solar arrays span {EXPECTED_SOLAR_ARRAY_SPAN}.",
            extracted.sources,
            solar_array_leaf,
            base_additional_instruction
            + " Accept equivalent phrasing like 'over 100 feet' or '~30 meters'.",
        ),
        (
            f"The mission is scheduled to arrive at Jupiter in {EXPECTED_JUPITER_ARRIVAL}.",
            extracted.sources,
            jupiter_arrival_leaf,
            base_additional_instruction
            + " Accept minor phrasing variations indicating arrival in April 2030.",
        ),
        (
            "The spacecraft is NASA's largest planetary mission spacecraft.",
            extracted.sources,
            largest_planetary_leaf,
            base_additional_instruction
            + " Accept equivalent phrasing such as 'largest spacecraft NASA has flown for a planetary mission'.",
        ),
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
    Evaluate the agent's answer for the Europa mission verification task.
    Build the verification tree and return a structured summary.
    """
    # 1) Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator parallel
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

    # 2) Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_europa_mission(),
        template_class=EuropaMissionExtraction,
        extraction_name="europa_mission_extraction",
    )

    # 3) Build verification tree and run checks
    await build_verification_tree(evaluator, root, extracted)

    # 4) Add ground truth info for transparency
    evaluator.add_ground_truth({
        "expected_values": {
            "mission_target": EXPECTED_MISSION_TARGET,
            "launch_date": EXPECTED_LAUNCH_DATE,
            "launch_site": EXPECTED_LAUNCH_SITE,
            "launch_vehicle": EXPECTED_LAUNCH_VEHICLE,
            "solar_array_span": EXPECTED_SOLAR_ARRAY_SPAN,
            "jupiter_arrival": EXPECTED_JUPITER_ARRIVAL,
            "largest_planetary_mission_spacecraft": EXPECTED_LARGEST_PLANETARY,
        }
    }, gt_type="ground_truth")

    # 5) Add custom info (e.g., number of source URLs extracted)
    evaluator.add_custom_info(
        info={
            "source_url_count": len(extracted.sources),
            "source_urls": extracted.sources,
        },
        info_type="source_summary",
    )

    # 6) Return result summary
    return evaluator.get_summary()