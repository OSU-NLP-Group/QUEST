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
TASK_ID = "publisher_james_percival_everett"
TASK_DESCRIPTION = """
What is the name of the publisher of the book 'James' by Percival Everett, which won the 2024 National Book Award for Fiction?
"""

EXPECTED_PUBLISHER = "Doubleday"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PublisherExtraction(BaseModel):
    """
    Extracted publisher information from the agent's answer.
    """
    publisher: Optional[str] = None
    # Capture any URLs (if the answer included them) just for logging/debugging purposes.
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_publisher() -> str:
    return """
    From the provided answer, extract the name of the publisher explicitly stated for the book "James" by Percival Everett.
    Rules:
    - Extract exactly the publisher name as written in the answer text.
    - If the answer provides multiple organizations (e.g., parent company, imprint group, or distributor), extract the one that is explicitly identified as the publisher of the book "James".
    - If the answer gives a variant such as "Doubleday Books" or "Doubleday Publishing", extract it as-is.
    - If no publisher is explicitly stated, return null for the publisher.
    - If any URLs are present that the answer uses as sources, include them in 'urls'; otherwise, return an empty array.
    JSON fields to return:
    - publisher: string | null
    - urls: string[] (may be empty)
    """


# --------------------------------------------------------------------------- #
# Verification logic builder                                                  #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extracted: PublisherExtraction,
) -> None:
    """
    Construct verification nodes based on the rubric and run the checks.
    Rubric summary:
    - Parent (critical, parallel): "Publisher_Identification"
      - Leaf (critical): "Publisher_Is_Doubleday"
    """
    # Add the rubric's critical, parallel parent node
    publisher_node = evaluator.add_parallel(
        id="Publisher_Identification",
        desc="Determine the publisher of the specified book in the question ('James' by Percival Everett, described as the 2024 National Book Award for Fiction winner) and verify it matches the stated constraints.",
        parent=evaluator.root,
        critical=True,
    )

    # Single critical leaf: Verify that the answer identifies the publisher as Doubleday
    is_doubleday_leaf = evaluator.add_leaf(
        id="Publisher_Is_Doubleday",
        desc="The answer identifies the publisher of the specified book as Doubleday (i.e., the publisher named is 'Doubleday').",
        parent=publisher_node,
        critical=True,
    )

    # Build the claim to judge strictly against the answer text:
    # We phrase it as: "According to the answer, the publisher is Doubleday."
    # The judge will evaluate if the answer explicitly states Doubleday (allowing minor, reasonable variants).
    claim = (
        "According to the answer text, the publisher of the book 'James' by Percival Everett is Doubleday. "
        "Treat minor variants like 'Doubleday Books' or 'Doubleday Publishing' as equivalent to 'Doubleday'. "
        "Do not consider parent companies or imprints that are not literally 'Doubleday' as equivalent."
    )

    await evaluator.verify(
        claim=claim,
        node=is_doubleday_leaf,
        sources=None,  # No external verification required; we judge based on the answer content
        additional_instruction="Focus only on the answer content. If the answer does not explicitly say Doubleday (or a minor variant), mark this as incorrect."
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
    Evaluate an answer for the publisher identification task.
    """
    # Initialize Evaluator with a parallel root (overall aggregation)
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract the publisher mentioned in the answer (for logging/debugging and transparency)
    extracted_pub = await evaluator.extract(
        prompt=prompt_extract_publisher(),
        template_class=PublisherExtraction,
        extraction_name="publisher_extraction",
    )

    # Add ground truth information
    evaluator.add_ground_truth(
        {
            "expected_publisher": EXPECTED_PUBLISHER,
            "book": "James",
            "author": "Percival Everett",
            "award_context": "2024 National Book Award for Fiction (context only)",
        },
        gt_type="ground_truth_publisher",
    )

    # Build and verify the tree per the rubric
    await build_verification_tree(evaluator, extracted_pub)

    # Return structured summary
    return evaluator.get_summary()