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
TASK_ID = "detroit_lions_thanksgiving_stadium"
TASK_DESCRIPTION = """
What is the name and seating capacity of the home stadium where the Detroit Lions traditionally host their annual Thanksgiving Day NFL game?
"""

# Ground truth references for clarity (recorded in summary)
GROUND_TRUTH = {
    "expected_stadium_name": "Ford Field",
    "expected_official_capacity": "65,000"
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StadiumExtraction(BaseModel):
    """
    Extracted stadium information from the agent's answer.
    All fields are optional to maximize compatibility with varied phrasing.
    """
    stadium_name: Optional[str] = None
    seating_capacity: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stadium_info() -> str:
    return """
    Extract the stadium information the answer provides for where the Detroit Lions traditionally host their annual Thanksgiving Day NFL game.
    Return the following fields:
    - stadium_name: The stadium's name as written in the answer (e.g., "Ford Field"). If not stated, return null.
    - seating_capacity: The seating capacity number or phrase as written (e.g., "65,000", "65k", "around 65,000"). If not stated, return null.
    - source_urls: A list of URLs explicitly mentioned in the answer that relate to the stadium or its capacity (official site, Wikipedia, NFL, etc.). If none, return an empty list.
    Do not infer or invent information; extract exactly what the answer states.
    """


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root_node,
) -> None:
    """
    Build the verification tree per the rubric:
    Critical parallel parent with two critical leaf checks:
      - Stadium name is Ford Field.
      - Official seating capacity is 65,000.
    """

    # Critical parent node (as specified in rubric)
    rubric_root = evaluator.add_parallel(
        id="DetroitLionsThanksgivingStadiumInformation",
        desc="Verify the answer gives the correct name and official seating capacity of the Detroit Lions' home stadium where they traditionally host their annual Thanksgiving Day NFL game.",
        parent=root_node,
        critical=True
    )

    # Leaf 1: Stadium name is Ford Field
    stadium_name_leaf = evaluator.add_leaf(
        id="StadiumNameIsFordField",
        desc="Answer states the stadium name is Ford Field.",
        parent=rubric_root,
        critical=True
    )
    name_claim = "The answer states that the stadium name is 'Ford Field'."
    name_instruction = (
        "Check the provided answer text and determine if it explicitly names the stadium as 'Ford Field'. "
        "Allow minor formatting variations (case-insensitive, quotes, possessives), but the stadium must be Ford Field."
    )

    # Leaf 2: Official seating capacity is 65,000
    capacity_leaf = evaluator.add_leaf(
        id="StadiumCapacityIs65000",
        desc="Answer states the official seating capacity is 65,000.",
        parent=rubric_root,
        critical=True
    )
    capacity_claim = "The answer states that the stadium's official seating capacity is 65,000."
    capacity_instruction = (
        "Check the answer text for a capacity of 65,000. Accept common variants like '65k', "
        "'65,000 spectators', or phrasing such as 'around 65,000' or 'official capacity 65,000'. "
        "Mentioning 'expandable to ~70,000' along with 65,000 is acceptable. "
        "If the answer only states 70,000 without mentioning 65,000, consider it incorrect."
    )

    # Run both verifications in parallel
    await evaluator.batch_verify([
        (name_claim, None, stadium_name_leaf, name_instruction),
        (capacity_claim, None, capacity_leaf, capacity_instruction),
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
    Evaluate the agent's answer for the Detroit Lions Thanksgiving stadium task.
    Returns a structured summary including the verification tree and final score.
    """
    # Initialize evaluator (root is non-critical by framework design)
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
        default_model=model
    )

    # Record ground truth expectations
    evaluator.add_ground_truth(
        {
            "expected_stadium_name": GROUND_TRUTH["expected_stadium_name"],
            "expected_official_capacity": GROUND_TRUTH["expected_official_capacity"]
        },
        gt_type="ground_truth"
    )

    # Extract info from the answer (for transparency in the summary; not strictly required for verification)
    extracted = await evaluator.extract(
        prompt=prompt_extract_stadium_info(),
        template_class=StadiumExtraction,
        extraction_name="stadium_info"
    )

    # Build and run the rubric-based verification tree
    await build_verification_tree(evaluator, root)

    # Return structured result
    return evaluator.get_summary()