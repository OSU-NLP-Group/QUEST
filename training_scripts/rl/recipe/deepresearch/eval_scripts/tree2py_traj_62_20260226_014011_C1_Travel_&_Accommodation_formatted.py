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
TASK_ID = "turkish_airlines_international_arrival_time"
TASK_DESCRIPTION = """
A passenger is traveling on Turkish Airlines international flight departing from Washington Dulles International Airport at 2:00 PM. Based on Turkish Airlines' official recommendations for international flights, what time should the passenger arrive at the airport?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ArrivalExtraction(BaseModel):
    """
    Extract exactly what the answer stated as the recommended airport arrival time.
    Do not calculate or infer; only extract explicit content from the answer.
    """
    arrival_time_text: Optional[str] = None  # e.g., "11:00 AM", "11 AM", "11am", "by 11:00 AM"
    timing_rule_text: Optional[str] = None   # e.g., "3 hours before departure"
    cited_sources: List[str] = Field(default_factory=list)  # any URLs cited in the answer (if any)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_arrival_info() -> str:
    return """
    Extract the explicit airport arrival time recommended in the answer for the Turkish Airlines international flight.
    Return:
    - arrival_time_text: the concrete clock time explicitly given (e.g., "11:00 AM", "11 AM", "11am", "by 11:00 AM"). If the answer does not present a specific time, set this to null.
    - timing_rule_text: any phrasing describing the timing rule (e.g., "3 hours before departure"). If absent, set to null.
    - cited_sources: all URLs present in the answer (if any are provided).
    Do not compute or infer a time; only extract what is explicitly stated in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: ArrivalExtraction) -> None:
    """
    Build the verification tree from the rubric and run the single required verification.
    """

    # Top-level rubric node under root (critical, parallel as specified)
    arrival_node = evaluator.add_parallel(
        id="Arrival_Time_Determination",
        desc="Determines the correct airport arrival time for a Turkish Airlines international flight departing at 2:00 PM using the stated 3-hour recommendation.",
        parent=evaluator.root,
        critical=True
    )

    # Leaf: provides explicit arrival time exactly 3 hours before 2:00 PM (i.e., 11:00 AM)
    leaf_node = evaluator.add_leaf(
        id="Provides_Arrival_Time_Three_Hours_Before_Departure",
        desc="Answer provides an airport arrival time that is exactly 3 hours before the stated 2:00 PM departure time, consistent with the international-flight recommendation.",
        parent=arrival_node,
        critical=True
    )

    # We perform a simple verification against the answer content:
    # The answer must explicitly provide a time equivalent to "11:00 AM" (three hours before 2:00 PM).
    # Accept reasonable variants like "11 AM", "11am", "by 11:00 AM", "no later than 11 AM".
    claim = (
        "In the provided answer, the recommended airport arrival time is explicitly given as 11:00 AM "
        "(or an equivalent format such as '11 AM', '11am', or 'by 11:00 AM'), which is exactly three hours "
        "before a 2:00 PM departure."
    )

    add_ins = (
        "Judge only based on the answer text. The answer must clearly include a concrete time expression equivalent "
        "to 11:00 AM (e.g., '11 AM', '11am', 'by 11:00 AM', 'no later than 11 AM'). If the answer merely says "
        "'3 hours before departure' without stating an explicit clock time, mark it incorrect. "
        "Do not penalize minor formatting differences."
    )

    await evaluator.verify(
        claim=claim,
        node=leaf_node,
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
    Evaluate an answer for the Turkish Airlines international arrival time task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation strategy
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

    # Extract any explicit arrival time the answer states (for record-keeping)
    extracted = await evaluator.extract(
        prompt=prompt_extract_arrival_info(),
        template_class=ArrivalExtraction,
        extraction_name="arrival_recommendation"
    )

    # Add expected ground truth info to summary
    evaluator.add_ground_truth({
        "expected_departure_time": "2:00 PM",
        "expected_arrival_time": "11:00 AM",
        "reasoning": "Turkish Airlines recommends arriving 3 hours before international flights."
    })

    # Build verification tree and run verification
    await build_and_verify_tree(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()