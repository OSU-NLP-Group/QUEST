import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "allegiant_personal_item_dimensions"
TASK_DESCRIPTION = (
    "What are the three maximum dimensions (height x width x length, in inches) allowed for Allegiant Air's "
    "free personal item that passengers can bring on board at no charge?"
)

# Ground truth dimensions for Allegiant Air free personal item
GROUND_TRUTH = {
    "height": "7 inches",
    "width": "15 inches",
    "length": "16 inches",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DimensionExtraction(BaseModel):
    """
    Extracted dimensions for Allegiant Air free personal item from the agent's answer.
    Values should be strings to maximize compatibility with formats such as '7 in', '7"', or '7 inches'.
    """
    height: Optional[str] = None
    width: Optional[str] = None
    length: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_dimensions() -> str:
    return """
    Extract the Allegiant Air free personal item maximum dimensions that the answer provides.
    Return a JSON object with the following fields:
    - height: The maximum height value for the free personal item, as a string (e.g., "7 inches", "7\"", "7 in"). If not explicitly labeled, interpret triples in the order Height × Width × Length (H × W × L) when the answer uses that order, or use any explicit labeling present (e.g., "L × W × H"). If only a triple is given without labels, default to Height × Width × Length order.
    - width: The maximum width value, as a string.
    - length: The maximum length value, as a string. Treat "depth" as "length" if the answer uses that synonym.
    - sources: All URLs explicitly cited in the answer that relate to baggage policy or item dimensions (e.g., Allegiant website pages). Extract actual URLs; if none are present, return an empty list.

    General rules:
    1. Extract exactly what the answer states; do not invent or infer values not present.
    2. Preserve units and formatting as given (e.g., "inches", "in", or quote notation).
    3. If a field is missing, set it to null. For sources, use an empty list if none are provided.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def additional_instruction_for_dimension_check(dimension_name: str, expected_value: str) -> str:
    return (
        f"Verify that, in the agent's answer, the maximum {dimension_name} for Allegiant Air's free personal item "
        f"is stated as {expected_value}.\n"
        f"Important clarifications:\n"
        f"- Focus on the 'free personal item' that fits under the seat (no charge), NOT the paid carry-on bag.\n"
        f"- Accept minor formatting variants like '{expected_value.replace(' inches', ' in')}', numeric plus double quote "
        f'(e.g., {expected_value.split()[0]}" ), or phrasing such as "{expected_value.split()[0]}-inch".\n'
        f"- If the answer presents a triple (e.g., '7 × 15 × 16'), interpret it as Height × Width × Length unless the answer "
        f"explicitly labels a different order (e.g., L × W × H). In that labeled case, map appropriately (height = H, width = W, length = L).\n"
        f"- Treat 'depth' as synonymous with 'length' if used.\n"
        f"- Your judgment should be based solely on the provided answer text (not external facts)."
    )


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    root_node,
    extracted: DimensionExtraction,
) -> None:
    """
    Build the verification tree and run checks.
    The rubric has a single critical parallel parent with three critical leaves.
    """
    # Parent node representing the overall requirement from the rubric (critical, parallel)
    parent = evaluator.add_parallel(
        id="All_three_dimensions_correctly_provided",
        desc="The answer provides all three maximum dimensions for Allegiant Air's free personal item allowance",
        parent=root_node,
        critical=True,
    )

    # Height leaf
    height_leaf = evaluator.add_leaf(
        id="Height_dimension",
        desc="The height dimension is correctly stated as 7 inches",
        parent=parent,
        critical=True,
        status="initialized",
        score=0.0,
    )
    height_claim = "In the agent's answer, the maximum height for the free personal item is 7 inches."
    await evaluator.verify(
        claim=height_claim,
        node=height_leaf,
        sources=None,  # We are verifying the answer content itself; not verifying against external webpages
        additional_instruction=additional_instruction_for_dimension_check("height", GROUND_TRUTH["height"]),
    )

    # Width leaf
    width_leaf = evaluator.add_leaf(
        id="Width_dimension",
        desc="The width dimension is correctly stated as 15 inches",
        parent=parent,
        critical=True,
        status="initialized",
        score=0.0,
    )
    width_claim = "In the agent's answer, the maximum width for the free personal item is 15 inches."
    await evaluator.verify(
        claim=width_claim,
        node=width_leaf,
        sources=None,
        additional_instruction=additional_instruction_for_dimension_check("width", GROUND_TRUTH["width"]),
    )

    # Length leaf
    length_leaf = evaluator.add_leaf(
        id="Length_dimension",
        desc="The length dimension is correctly stated as 16 inches",
        parent=parent,
        critical=True,
        status="initialized",
        score=0.0,
    )
    length_claim = "In the agent's answer, the maximum length for the free personal item is 16 inches."
    await evaluator.verify(
        claim=length_claim,
        node=length_leaf,
        sources=None,
        additional_instruction=additional_instruction_for_dimension_check("length", GROUND_TRUTH["length"]),
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
    Evaluate an answer for Allegiant Air free personal item dimensions.
    Returns a summary dict containing the verification tree and scores.
    """
    # Initialize evaluator with root node
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

    # Extract dimensions from the answer
    extracted_dims = await evaluator.extract(
        prompt=prompt_extract_dimensions(),
        template_class=DimensionExtraction,
        extraction_name="dimensions_extraction",
    )

    # Add ground truth info for transparency
    evaluator.add_ground_truth(
        {
            "expected_dimensions": GROUND_TRUTH,
            "note": "These are the expected maximum dimensions for Allegiant Air's free personal item as per the rubric."
        },
        gt_type="ground_truth_dimensions",
    )

    # Build and verify according to rubric
    await build_and_verify_tree(evaluator, root, extracted_dims)

    # Return structured evaluation summary
    return evaluator.get_summary()