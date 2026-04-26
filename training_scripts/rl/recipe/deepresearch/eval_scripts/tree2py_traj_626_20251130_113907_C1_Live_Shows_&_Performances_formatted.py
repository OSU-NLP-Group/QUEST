import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "detroit_lions_thanksgiving_2024_halftime_show"
TASK_DESCRIPTION = "Who were the two main performers at the Detroit Lions Thanksgiving Day halftime show on November 27, 2024, at Ford Field? Identify both the headlining artist and the surprise guest who joined for the performance."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HalftimeShowExtraction(BaseModel):
    """
    Extracted information from the agent's answer regarding the Detroit Lions
    Thanksgiving Day 2024 halftime show at Ford Field.
    """
    headliner: Optional[str] = None
    surprise_guest: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_halftime_show() -> str:
    return """
    Extract the names of the two main performers described in the answer for the Detroit Lions Thanksgiving Day halftime show on November 27, 2024, at Ford Field.

    Required fields:
    1) headliner: The headlining artist/main performer explicitly identified in the answer (e.g., "Jack White"). Use common synonyms like "headliner", "main performer", "headline act", "main artist", or similar phrasing to determine this.
    2) surprise_guest: The surprise guest performer who joined on stage/for the performance (e.g., "Eminem"). Use synonyms like "surprise guest", "special guest", "guest appearance", "joined by", or similar phrasing.
    3) sources: A list of any URLs explicitly cited in the answer that support these identifications. If none are cited, return an empty list.

    Rules:
    - Extract exactly what is stated in the answer. Do not infer beyond the text.
    - If a required field is not present, set it to null.
    - For URLs, return only valid and complete URLs explicitly present in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extracted: HalftimeShowExtraction,
) -> None:
    """
    Build the verification tree per the rubric and run the leaf verifications.
    """
    # Top-level critical node mirroring the rubric root
    top_node = evaluator.add_parallel(
        id="Detroit_Lions_Thanksgiving_2024_Halftime_Show",
        desc="Verification of the performers at the Detroit Lions Thanksgiving 2024 halftime show at Ford Field",
        parent=evaluator.root,
        critical=True
    )

    # Leaf: Headliner identification (critical)
    headliner_leaf = evaluator.add_leaf(
        id="Headliner_Identification",
        desc="Correctly identifies Jack White as the headlining performer",
        parent=top_node,
        critical=True
    )

    # Leaf: Guest performer identification (critical)
    guest_leaf = evaluator.add_leaf(
        id="Guest_Performer_Identification",
        desc="Correctly identifies Eminem as the surprise guest performer",
        parent=top_node,
        critical=True
    )

    # Prepare claims
    headliner_claim = (
        "According to the answer, the headlining performer at the Detroit Lions Thanksgiving Day halftime show "
        "on November 27, 2024, at Ford Field is Jack White."
    )
    guest_claim = (
        "According to the answer, the surprise guest performer who joined the halftime show performance is Eminem."
    )

    # Additional instructions to guide simple verification
    headliner_instruction = (
        "Verify strictly based on the provided answer text whether it explicitly identifies Jack White as the headliner "
        "or main performer. Accept reasonable synonyms such as 'headliner', 'main performer', 'headline act'. "
        "Do not rely on external knowledge; focus only on whether the answer states Jack White as the headliner."
    )
    guest_instruction = (
        "Verify strictly based on the provided answer text whether it explicitly identifies Eminem as the surprise guest "
        "or special guest who joined. Accept reasonable synonyms such as 'surprise guest', 'special guest', 'guest appearance', "
        "'joined by'. Do not rely on external knowledge; focus only on whether the answer states Eminem as the surprise guest."
    )

    # Run both verifications concurrently to avoid sibling prerequisite gating
    await evaluator.batch_verify([
        (headliner_claim, None, headliner_leaf, headliner_instruction),
        (guest_claim, None, guest_leaf, guest_instruction),
    ])


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
    Evaluate an answer for the Detroit Lions Thanksgiving 2024 halftime show performers task.

    Returns a structured summary containing the verification tree and final score.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation strategy (wrapper root)
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
        prompt=prompt_extract_halftime_show(),
        template_class=HalftimeShowExtraction,
        extraction_name="halftime_show_extraction"
    )

    # Record ground truth for clarity
    evaluator.add_ground_truth({
        "event": "Detroit Lions Thanksgiving Day halftime show",
        "date": "2024-11-27",
        "location": "Ford Field",
        "expected_headliner": "Jack White",
        "expected_surprise_guest": "Eminem"
    })

    # Build tree and run verifications
    await build_and_verify_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()