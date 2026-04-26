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
TASK_ID = "nds_2025_handler_state"
TASK_DESCRIPTION = """
What U.S. state is the handler of the 2025 National Dog Show Best in Show winner from?
"""

GROUND_TRUTH = {
    "expected_state_full": "North Carolina",
    "accepted_abbreviations": ["NC", "N.C."]
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HandlerStateExtraction(BaseModel):
    """
    Extract the handler's U.S. state as stated in the answer.
    """
    # Full state name if explicitly mentioned (e.g., "North Carolina")
    state_full_name: Optional[str] = None
    # State abbreviation if explicitly mentioned (e.g., "NC" or "N.C.")
    state_abbreviation: Optional[str] = None
    # Optional short supporting snippets directly quoted from the answer
    evidence_snippets: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_handler_state() -> str:
    return """
    Extract the U.S. state that the answer explicitly says the handler of the 2025 National Dog Show Best in Show winner is from.
    Rules:
    - state_full_name: The full U.S. state name if provided (e.g., "North Carolina").
    - state_abbreviation: The state abbreviation if provided (e.g., "NC" or "N.C."). If both full name and abbreviation are present, fill both fields.
    - evidence_snippets: Up to 2 short quotes (verbatim substrings) from the answer that directly support the extracted state. If none, return an empty list.
    - If the answer does not state the handler's state, return null for both fields and an empty evidence_snippets list.
    - Do not guess or infer beyond what is written in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_handler_state(
    evaluator: Evaluator,
    parent_node,
    extracted: HandlerStateExtraction
) -> None:
    """
    Build the verification subtree and run the check that the answer identifies the handler's
    state as North Carolina (accepting abbreviation 'NC' or 'N.C.' as equivalent).
    """
    # Aggregation node (critical as per rubric)
    handler_state_node = evaluator.add_parallel(
        id="handler_state_answer",
        desc="Answer states the U.S. state that the handler of the 2025 National Dog Show Best in Show winner is from.",
        parent=parent_node,
        critical=True
    )

    # Leaf check: The answer identifies the handler's state as North Carolina (or NC)
    leaf_nc = evaluator.add_leaf(
        id="state_is_north_carolina",
        desc="Answer identifies the handler's state as North Carolina (accept full name or standard abbreviation such as \"NC\").",
        parent=handler_state_node,
        critical=True
    )

    # Construct claim independent of extraction, verified against the provided answer text
    claim = (
        "In the provided answer, the handler's U.S. state is identified as North Carolina "
        "(the answer may use the full name 'North Carolina' or the standard abbreviation 'NC' or 'N.C.')."
    )

    # Additional instruction to guide the verifier
    add_ins = (
        "Judge solely based on the provided answer text and the task description. "
        "Treat 'NC' and 'N.C.' as equivalent to 'North Carolina'. "
        "If multiple locations are mentioned (e.g., the show venue), focus specifically on the handler's home state."
    )

    await evaluator.verify(
        claim=claim,
        node=leaf_nc,
        additional_instruction=add_ins
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
    Evaluate an answer for the 2025 National Dog Show Best in Show handler's state question.
    """
    # Initialize evaluator (root is non-critical; we add a critical sub-node per rubric)
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

    # Extraction (recorded for transparency, not strictly required for verification)
    extracted = await evaluator.extract(
        prompt=prompt_extract_handler_state(),
        template_class=HandlerStateExtraction,
        extraction_name="handler_state_extraction"
    )

    # Add GT metadata to summary
    evaluator.add_ground_truth({
        "expected_state_full": GROUND_TRUTH["expected_state_full"],
        "accepted_abbreviations": GROUND_TRUTH["accepted_abbreviations"],
        "focus": "handler's home state for the 2025 National Dog Show Best in Show winner"
    })

    # Build verification subtree and run checks
    await verify_handler_state(evaluator, root, extracted)

    # Return the evaluation summary
    return evaluator.get_summary()