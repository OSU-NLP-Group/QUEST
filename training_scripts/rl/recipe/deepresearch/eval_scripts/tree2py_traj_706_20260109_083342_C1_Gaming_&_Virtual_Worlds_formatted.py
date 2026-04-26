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
TASK_ID = "meta_quest_resolution_model"
TASK_DESCRIPTION = "What is the model name of the Meta Quest VR headset that features a per-eye resolution of 2,064 x 2,208 pixels?"

TARGET_RESOLUTION = "2,064 x 2,208"
TARGET_RESOLUTION_ALT = "2064 x 2208"  # Alternate formatting often seen


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class VRHeadsetInfo(BaseModel):
    """Structured info extracted from the agent's answer."""
    model_name: Optional[str] = None
    resolution_per_eye_text: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_headset_info() -> str:
    return """
    Extract the headset model name, the per-eye resolution (as stated in the answer), and all cited source URLs from the provided answer.

    Return a JSON object with the following fields:
    - model_name: The headset model name explicitly claimed in the answer (e.g., "Meta Quest 3", "Quest 3"). If multiple models are mentioned, pick the one directly associated with the specified resolution. If not explicitly stated, return null.
    - resolution_per_eye_text: The per-eye resolution string exactly as written in the answer (e.g., "2,064 x 2,208", "2064×2208"). If not stated, return null.
    - source_urls: An array of all URLs (including markdown links) that the answer cites as sources. Only include valid URLs; if no URLs are provided, return an empty array.

    Notes:
    - Do not infer or add information not present in the answer.
    - Prefer extracting strings rather than numbers; keep the exact resolution format used by the answer.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_headset(
    evaluator: Evaluator,
    parent_node,
    info: VRHeadsetInfo,
) -> None:
    """
    Build the verification tree and run checks according to the rubric.
    """
    # Critical aggregator node (all children must be critical per framework rules)
    correct_node = evaluator.add_parallel(
        id="Correct_VR_Headset_Identification",
        desc="Identifies the correct Meta Quest VR headset model that matches the specified resolution",
        parent=parent_node,
        critical=True
    )

    model = info.model_name or ""
    urls = info.source_urls

    # 1) Meta Quest product line check (critical leaf)
    meta_quest_node = evaluator.add_leaf(
        id="Meta_Quest_Product_Line",
        desc="The identified headset is from the Meta Quest product line",
        parent=correct_node,
        critical=True
    )
    meta_quest_claim = f"The headset model '{model}' is a Meta Quest headset (part of Meta's Quest product line)."
    await evaluator.verify(
        claim=meta_quest_claim,
        node=meta_quest_node,
        sources=urls,  # If provided, verify using the pages; otherwise simple verify
        additional_instruction=(
            "Use the provided URLs if available to confirm the product line. "
            "Accept legacy branding 'Oculus Quest' as equivalent to 'Meta Quest' due to Meta rebranding. "
            "If no URLs are provided and the answer does not clearly indicate Meta Quest, mark Incorrect."
        ),
    )

    # 2) Resolution match check (critical leaf)
    resolution_node = evaluator.add_leaf(
        id="Resolution_Match",
        desc=f"The headset has a per-eye resolution of {TARGET_RESOLUTION} pixels as specified in the question",
        parent=correct_node,
        critical=True
    )
    resolution_claim = (
        f"The headset '{model}' has a per-eye resolution of {TARGET_RESOLUTION} pixels."
    )
    await evaluator.verify(
        claim=resolution_claim,
        node=resolution_node,
        sources=urls,
        additional_instruction=(
            f"Verify the per-eye resolution using the cited sources if present. "
            f"Allow minor formatting variants such as '{TARGET_RESOLUTION_ALT}', '2064×2208', '2,064 × 2,208', "
            f"or 'per-eye 2064 by 2208'. If pages show an equivalent per-eye resolution that matches these numbers, "
            f"consider it supported."
        ),
    )

    # 3) Official verification presence and support (critical leaf)
    official_node = evaluator.add_leaf(
        id="Official_Verification",
        desc="An official Meta or reliable technical source URL is provided to verify the specifications",
        parent=correct_node,
        critical=True
    )
    official_claim = (
        f"At least one of the cited sources is either an official Meta website (e.g., meta.com, store.meta.com) "
        f"or a reputable technical publication (e.g., The Verge, Road to VR, UploadVR, Tom's Hardware, CNET, PCMag), "
        f"and it explicitly confirms that '{model}' has a per-eye resolution of {TARGET_RESOLUTION} pixels."
    )
    await evaluator.verify(
        claim=official_claim,
        node=official_node,
        sources=urls,
        additional_instruction=(
            "Use the URLs to check both the credibility (official Meta or reputable tech site) and the resolution support. "
            "If no URLs are provided, mark as Incorrect. If URLs are provided but none are official/reputable or they do not "
            "explicitly confirm the resolution, mark as Incorrect."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate the agent's answer for the Meta Quest headset resolution identification task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Single rubric cluster; parallel aggregation is appropriate
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
        prompt=prompt_extract_headset_info(),
        template_class=VRHeadsetInfo,
        extraction_name="vr_headset_info"
    )

    # Provide ground truth for context (not used to score directly)
    evaluator.add_ground_truth({
        "expected_model_name_example": "Meta Quest 3",
        "target_resolution_per_eye": TARGET_RESOLUTION,
        "notes": "Quest 3 is widely documented to have a per-eye resolution around 2064 x 2208."
    }, gt_type="reference_info")

    # Build verification tree and run checks
    await verify_headset(evaluator, root, extracted_info)

    # Return structured summary
    return evaluator.get_summary()