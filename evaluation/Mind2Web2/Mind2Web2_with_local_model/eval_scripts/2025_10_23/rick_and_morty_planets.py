import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.llm_client.base_client import LLMClient
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import Evaluator
from mind2web2.verification_tree import VerificationNode, AggregationStrategy

TASK_ID = "rick_and_morty_planets"
TASK_DESCRIPTION = """
Identify every real-world celestial body featured in Rick & Morty (Seasons 1–7). For each one, find the first episode in which it appeared or was mentioned, providing its season and episode number (e.g., S02E11) and the episode's title.
"""

EVAL_NOTES = ""
GROUND_TRUTH = {
    "celestial_bodies": [
        {"name": "Earth", "episode": "S01E01", "title": "Pilot"},
        {"name": "Jupiter", "episode": "S01E09", "title": "Something Ricked This Way Comes"},
        {"name": "Mars", "episode": "S04E03", "title": "One Crew Over the Crewcoo's Morty"},
        {"name": "Mercury", "episode": "S06E09", "title": "A Rick in King Mortur's Mort"},
        {"name": "Neptune", "episode": "S05E10", "title": "Rickmurai Jack"},
        {"name": "Saturn", "episode": "S06E09", "title": "A Rick in King Mortur's Mort"},
        {"name": "Uranus", "episode": "S05E10", "title": "Rickmurai Jack"},
        {"name": "Venus", "episode": "S06E09", "title": "A Rick in King Mortur's Mort"},
        {"name": "Pluto", "episode": "S01E09", "title": "Something Ricked This Way Comes"}
    ]
}

# Create ground truth string for verification
GROUND_TRUTH_STRING = "\n".join([
    f"{body['name']}: {body['episode']}, {body['title']}"
    for body in GROUND_TRUTH["celestial_bodies"]
])


class CelestialBodyList(BaseModel):
    """List of celestial body names mentioned in the answer."""
    celestial_bodies: List[str] = Field(default_factory=list,
                                        description="Names of real-world celestial bodies mentioned as final answers")


class CelestialBodyDetails(BaseModel):
    """Details for a specific celestial body."""
    name: Optional[str] = Field(default=None, description="Name of the celestial body")
    episode: Optional[str] = Field(default=None, description="Season and episode number (e.g., S01E01)")
    title: Optional[str] = Field(default=None, description="Episode title")
    sources: List[str] = Field(default_factory=list, description="URLs or sources mentioned for this celestial body")


def prompt_extract_celestial_body_names() -> str:
    """
    Extraction prompt for getting the list of celestial body names from the answer.
    """
    return """
    Extract all real-world celestial body names that are presented as final answers in the response.

    Look for celestial bodies that are:
    - Explicitly mentioned as real-world planets, dwarf planets, moons, or other celestial objects
    - Presented as part of the final answer or conclusion
    - NOT fictional planets or locations from the show

    Return only the names of the celestial bodies, exactly as they appear in the text.
    Do not include any episode information or other details - just the names.

    Examples of what to extract: "Earth", "Mars", "Jupiter", "Pluto"
    Examples of what NOT to extract: "Gazorpazorp", "Cronenberg World", "Bird Person's planet"
    """


def prompt_extract_celestial_body_details(celestial_body_name: str) -> str:
    """
    Extraction prompt for getting details about a specific celestial body.
    """
    return f"""
    Extract all information about the celestial body "{celestial_body_name}" from the answer.

    Look for:
    - name: The name of this celestial body (should be "{celestial_body_name}")
    - episode: The season and episode number where it first appeared (format like S01E01, S02E11, etc.)
    - title: The title of that episode
    - sources: Any URLs or web sources mentioned in relation to this celestial body

    Extract information exactly as it appears in the text.
    If any field is not mentioned for this celestial body, set it to null or empty list.
    Focus only on information specifically related to "{celestial_body_name}".
    """


async def verify_celestial_body_details(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        body_details: CelestialBodyDetails,
        body_index: int,
) -> None:
    """
    Verify the details of a single celestial body.

    Args:
        evaluator: The evaluator instance
        parent_node: Parent node for this celestial body verification
        body_details: Extracted celestial body details
        body_index: Index of this body in the list
    """

    body_name = body_details.name or f"body_{body_index + 1}"

    # Create a container node for this celestial body
    body_node = evaluator.add_parallel(
        id=f"body_{body_index + 1}",
        desc=f"Celestial body {body_index + 1}: {body_name}",
        parent=parent_node,
        critical=False  # Non-critical for partial scoring
    )

    # Check if celestial body details exist
    has_details = (body_details.name is not None and
                   body_details.episode is not None and
                   body_details.title is not None and body_details.sources)

    existence_node = evaluator.add_custom_node(
        result=has_details,
        id=f"body_{body_index + 1}_has_details",
        desc=f"Celestial body {body_index + 1} has complete episode information",
        parent=body_node,
        critical=True  # Critical - need complete info for meaningful verification
    )

    # if has_details:
        # Verify against ground truth using the ground truth string
    gt_verification_node = evaluator.add_leaf(
        id=f"body_{body_index + 1}_matches_gt",
        desc=f"Celestial body {body_index + 1} information matches ground truth",
        parent=body_node,
        critical=True  # Critical - must match ground truth
    )

    body_info_string = f"{body_details.name}: {body_details.episode}, {body_details.title}"
    gt_verification_claim = f"""
    The celestial body information "{body_info_string}" matches one of the entries in the ground truth list:

    {GROUND_TRUTH_STRING}

    Check if the celestial body name, episode number, and episode title correspond to any entry in the ground truth list.
    """

    await evaluator.verify(
        claim=gt_verification_claim,
        node=gt_verification_node,
        sources=None,
        additional_instruction="Allow minor spelling variations and formatting differences. Focus on whether the celestial body name, episode, and title match any entry in the ground truth list."
    )

    source_verification_node = evaluator.add_leaf(
        id=f"body_{body_index + 1}_source_verified",
        desc=f"Celestial body {body_index + 1} episode title verified by web sources",
        parent=body_node,
        critical=False  # Non-critical - additional verification
    )

    episode_title_claim = f"The episode title '{body_details.title}' is mentioned or confirmed in the provided web sources"

    await evaluator.verify(
        claim=episode_title_claim,
        node=source_verification_node,
        sources=body_details.sources,
        additional_instruction="Check if the webpage mentions the episode title. Any mention or reference to the episode title counts as verification."
    )

async def create_placeholder_celestial_body(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        body_index: int,
) -> None:
    """
    Create placeholder nodes for missing celestial bodies.
    """

    # Create a container node for missing celestial body
    body_node = evaluator.add_parallel(
        id=f"body_{body_index + 1}_missing",
        desc=f"Celestial body {body_index + 1}: [Missing]",
        parent=parent_node,
        critical=False  # Non-critical for partial scoring
    )

    # Create placeholder nodes
    missing_details_node = evaluator.add_leaf(
        id=f"body_{body_index + 1}_missing_existence",
        desc=f"Celestial body {body_index + 1} - not provided in answer",
        parent=body_node,
        critical=True,
        score=0.0,
        status="failed"
    )

    missing_gt_node = evaluator.add_leaf(
        id=f"body_{body_index + 1}_missing_gt_match",
        desc=f"Celestial body {body_index + 1} - ground truth matching not possible",
        parent=body_node,
        critical=True,
        score=0.0,
        status="failed"
    )

    missing_source_node = evaluator.add_leaf(
        id=f"body_{body_index + 1}_missing_source_verify",
        desc=f"Celestial body {body_index + 1} - source verification not possible",
        parent=body_node,
        critical=False,
        score=0.0,
        status="skipped"
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                               #
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
) -> Dict[str, Any]:
    """
    Main evaluation function for Rick & Morty celestial bodies identification task.

    This function:
    1. Extracts celestial body names from the answer
    2. Selects the first len(ground_truth) celestial bodies
    3. For each body, extracts detailed information and verifies against ground truth
    4. Verifies episode information through web sources if available
    5. Returns evaluation summary with partial scoring
    """

    # -------- 1. Initialize evaluator ----------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel for partial scoring
        agent_name=agent_name,
        answer_name=answer_name,
        # Evaluator creation parameters
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Add ground truth information
    evaluator.add_ground_truth(GROUND_TRUTH, "expected_celestial_bodies")

    # -------- 2. Extract celestial body names ------------------- #
    celestial_body_names = await evaluator.extract(
        prompt=prompt_extract_celestial_body_names(),
        template_class=CelestialBodyList,
        extraction_name="celestial_body_names",
        source=None,  # Extract from answer text
    )

    # -------- 3. Process the celestial bodies list -------------- #
    expected_count = len(GROUND_TRUTH["celestial_bodies"])
    extracted_names = celestial_body_names.celestial_bodies[:expected_count]  # Take first len(gt) items

    # Pad with empty strings if we have fewer than expected
    while len(extracted_names) < expected_count:
        extracted_names.append("")

    # Add information about the extraction
    evaluator.add_custom_info({
        "total_extracted": len(celestial_body_names.celestial_bodies),
        "used_for_evaluation": len(extracted_names),
        "expected_count": expected_count,
        "extracted_names": extracted_names
    }, "extraction_summary")

    # -------- 4. Build verification tree -------------------------- #

    # Create main verification section
    bodies_verification_node = evaluator.add_parallel(
        id="celestial_bodies_verification",
        desc="Individual celestial bodies verification",
        parent=root,
        critical=False  # Non-critical for partial scoring
    )

    # Process each celestial body position
    for i in range(expected_count):
        if i < len(extracted_names) and extracted_names[i].strip():
            # Extract detailed information for this celestial body
            body_name = extracted_names[i].strip()

            body_details = await evaluator.extract(
                prompt=prompt_extract_celestial_body_details(body_name),
                template_class=CelestialBodyDetails,
                extraction_name=f"body_{i + 1}_details",
                source=None,
            )

            # Verify this celestial body
            await verify_celestial_body_details(
                evaluator=evaluator,
                parent_node=bodies_verification_node,
                body_details=body_details,
                body_index=i,
            )
        else:
            # Create placeholder for missing celestial body
            await create_placeholder_celestial_body(
                evaluator=evaluator,
                parent_node=bodies_verification_node,
                body_index=i,
            )

    # -------- 5. Return evaluation results ------------------------ #
    return evaluator.get_summary()