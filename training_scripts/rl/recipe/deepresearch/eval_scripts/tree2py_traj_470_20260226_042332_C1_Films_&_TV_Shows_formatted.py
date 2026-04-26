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
TASK_ID = "fallout_premiere_2024"
TASK_DESCRIPTION = "What streaming platform premiered the TV series Fallout in 2024, and on what date did it premiere?"

EXPECTED_PLATFORM = "Amazon Prime Video"  # Also commonly referred to as "Prime Video"
EXPECTED_DATE = "April 10, 2024"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FalloutPremiereInfo(BaseModel):
    """
    Structured extraction for the Fallout premiere answer.
    """
    streaming_platform: Optional[str] = None
    premiere_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_fallout_premiere_info() -> str:
    return """
    Extract the streaming platform and the premiere date for the TV series "Fallout" as stated in the provided answer.
    Also extract all URL sources mentioned in the answer that are related to the Fallout series or its release.

    Return a JSON object with the following fields:
    - streaming_platform: the name of the streaming service explicitly mentioned in the answer (e.g., "Amazon Prime Video" or "Prime Video"). If not provided, return null.
    - premiere_date: the premiere date explicitly mentioned in the answer (e.g., "April 10, 2024"). If not provided, return null.
    - sources: an array of all URLs present in the answer that serve as sources or references for the information about Fallout or its premiere. Include full URLs (with protocol). If none are present, return an empty array.

    Important:
    - Do not infer or invent information; only extract what is explicitly written in the answer.
    - For URLs, include only valid URLs present in the answer text (including markdown links).
    """


# --------------------------------------------------------------------------- #
# Verification sub-tree construction                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_requirements(
    evaluator: Evaluator,
    root_node,
    extracted: FalloutPremiereInfo,
) -> None:
    """
    Create the 'answer_requirements' node and its leaf verifications for:
    - Streaming platform (critical)
    - Premiere date (critical)
    Additionally add a non-critical check for whether sources were provided (quality signal).
    """
    # Parent node for answer requirements
    req_node = evaluator.add_parallel(
        id="answer_requirements",
        desc="The answer must correctly identify both the streaming platform and the premiere date for Fallout",
        parent=root_node,
        critical=False
    )

    # Optional non-critical quality check: sources provided
    sources_present = bool(extracted.sources) and len(extracted.sources) > 0
    evaluator.add_custom_node(
        result=sources_present,
        id="sources_provided",
        desc="At least one source URL is provided in the answer (quality signal)",
        parent=req_node,
        critical=False
    )

    # Leaf: Streaming platform correctness (critical)
    platform_leaf = evaluator.add_leaf(
        id="streaming_platform",
        desc="The streaming platform is correctly identified as Amazon Prime Video",
        parent=req_node,
        critical=True
    )
    platform_text = extracted.streaming_platform or ""
    platform_claim = (
        f"The streaming platform mentioned in the answer ('{platform_text}') refers to Amazon Prime Video, "
        f"also known as 'Prime Video'."
    )
    await evaluator.verify(
        claim=platform_claim,
        node=platform_leaf,
        sources=extracted.sources if extracted.sources else None,
        additional_instruction=(
            "Confirm that the referenced webpage(s) support that Fallout premiered on Amazon Prime Video "
            "(often branded simply as 'Prime Video'). Allow minor naming variations and branding usage."
        ),
    )

    # Leaf: Premiere date correctness (critical)
    date_leaf = evaluator.add_leaf(
        id="premiere_date",
        desc="The premiere date is correctly identified as April 10, 2024",
        parent=req_node,
        critical=True
    )
    date_text = extracted.premiere_date or ""
    date_claim = (
        f"The premiere date mentioned in the answer ('{date_text}') is {EXPECTED_DATE}."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=extracted.sources if extracted.sources else None,
        additional_instruction=(
            "Verify from the referenced webpage(s) that the TV series Fallout premiered on April 10, 2024. "
            "Accept reasonable date formatting variations (e.g., 'Apr 10, 2024')."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer to the Fallout premiere question.
    """
    # Initialize evaluator (root is non-critical parallel aggregator by default)
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

    # Extract structured info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_fallout_premiere_info(),
        template_class=FalloutPremiereInfo,
        extraction_name="fallout_premiere_info",
    )

    # Add ground truth for reference (non-evaluative, metadata)
    evaluator.add_ground_truth({
        "expected_platform": EXPECTED_PLATFORM,
        "expected_premiere_date": EXPECTED_DATE,
    }, gt_type="ground_truth_fallout")

    # Build and run verification according to rubric
    await build_and_verify_requirements(evaluator, root, extracted_info)

    # Return structured summary
    return evaluator.get_summary()