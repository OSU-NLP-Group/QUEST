import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "james_publisher"
TASK_DESCRIPTION = (
    'Who is the publisher of the novel "James" by Percival Everett? Please provide a reference URL to support your answer.'
)

EXPECTED_PUBLISHER = "Doubleday"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PublisherExtraction(BaseModel):
    """
    Extracted information from the answer:
    - publisher: the publisher name stated for "James" by Percival Everett
    - urls: all URLs provided in the answer (as references/support)
    """
    publisher: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_publisher() -> str:
    return """
    From the answer text, extract the following fields specifically for the novel "James" by Percival Everett:

    - publisher: The publisher name explicitly stated in the answer for this book. If multiple publisher-like names are mentioned, choose the one identified as the publisher of "James". If the publisher is not provided, return null.
    - urls: A list of all reference URLs present anywhere in the answer (these might be in plain form or markdown links). Include every valid URL you can find in the answer.

    Rules:
    1. Only extract information explicitly present in the answer text. Do not infer or add information.
    2. For urls, capture actual URLs (from plain text or markdown). If a URL is missing a protocol, prepend http://.
    3. If no URLs are present, return an empty list for urls.
    """


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer to identify the publisher of "James" by Percival Everett and verify source support.
    """
    # Initialize evaluator with a parallel root (independent checks under the task node)
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_publisher(),
        template_class=PublisherExtraction,
        extraction_name="publisher_extraction",
    )

    # Record ground truth for transparency
    evaluator.add_ground_truth(
        {
            "expected_publisher": EXPECTED_PUBLISHER,
            "book_title": "James",
            "author": "Percival Everett",
        },
        gt_type="ground_truth_publisher"
    )

    # Build the rubric tree according to the provided structure
    task_node = evaluator.add_parallel(
        id="James_Publisher_Information",
        desc="Verify the publisher of the novel 'James' by Percival Everett",
        parent=root,
        critical=True  # Critical parent: all children must be critical
    )

    # Leaf 1: Publisher_Name
    publisher_leaf = evaluator.add_leaf(
        id="Publisher_Name",
        desc="The answer identifies Doubleday as the publisher of 'James'",
        parent=task_node,
        critical=True
    )

    # We check directly against the answer text: require that the answer explicitly names "Doubleday"
    publisher_claim = (
        "Within the answer text, the publisher of the novel 'James' by Percival Everett is explicitly stated as "
        "'Doubleday' (allow minor variants such as 'Doubleday Books')."
    )
    await evaluator.verify(
        claim=publisher_claim,
        node=publisher_leaf,
        additional_instruction=(
            "Judge solely based on the answer text. Consider this correct only if the answer mentions 'Doubleday' "
            "as the publisher (case-insensitive). Variants like 'Doubleday Books' are acceptable. "
            "Mentions of broader corporate groups without the explicit word 'Doubleday' should not be considered sufficient."
        )
    )

    # Leaf 2: Reference_URL
    reference_leaf = evaluator.add_leaf(
        id="Reference_URL",
        desc="The answer provides a reference URL supporting the publisher information",
        parent=task_node,
        critical=True
    )

    # Verify that at least one provided URL supports the publisher claim (Doubleday for "James")
    urls = extraction.urls or []
    support_claim = (
        "This webpage explicitly states that the publisher of the novel 'James' by Percival Everett is Doubleday "
        "(e.g., phrases like 'Publisher: Doubleday' or 'Published by Doubleday')."
    )
    await evaluator.verify(
        claim=support_claim,
        node=reference_leaf,
        sources=urls,
        additional_instruction=(
            "STRICT REQUIREMENT: If the answer includes no URLs, return Incorrect. "
            "When URLs are provided, judge Correct only if at least one of the webpages clearly indicates "
            "that Doubleday is the publisher of 'James' by Percival Everett. "
            "Accept authoritative sources such as publisher pages, major booksellers, or reputable media coverage "
            "that explicitly list Doubleday as the publisher."
        )
    )

    return evaluator.get_summary()