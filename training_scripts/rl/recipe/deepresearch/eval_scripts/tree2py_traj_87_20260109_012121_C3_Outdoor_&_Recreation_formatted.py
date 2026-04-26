import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "whitney_2026_claim_deadline_fee"
TASK_DESCRIPTION = (
    "For the highest peak in the contiguous United States that requires an annual permit lottery system for hikers, "
    "what is the deadline date by which lottery winners must claim their awarded reservation for the 2026 hiking season, "
    "and what is the mandatory per-person fee that must be paid by this deadline to secure the permit?"
)

# Ground truth expectations for reference (informational; verification checks below enforce these)
GROUND_TRUTH = {
    "peak": "Mount Whitney",
    "deadline_date_2026": "April 21, 2026",
    "fee": "$15.00 per person recreation fee"
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WhitneyAnswerExtraction(BaseModel):
    """
    Extracted key facts from the agent's answer. Keep fields as strings and lists for robustness.
    """
    peak_name: Optional[str] = None
    mentions_annual_lottery: Optional[bool] = None
    deadline_date_text: Optional[str] = None
    per_person_fee_text: Optional[str] = None

    # URLs mentioned in the answer (if any). These are not required by the rubric but are recorded.
    sources_all: List[str] = Field(default_factory=list)
    sources_deadline: List[str] = Field(default_factory=list)
    sources_fee: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_whitney_answer() -> str:
    return """
    From the provided answer, extract the following fields exactly as they appear:

    1) peak_name: The name of the peak the answer identifies in response to the question. If the answer refers to “Mount Whitney” or “Mt. Whitney,” return the exact text used (e.g., "Mount Whitney" or "Mt. Whitney"). If not stated, return null.

    2) mentions_annual_lottery: true if the answer explicitly states that the peak uses an annual permit lottery system for hikers (look for words like "annual permit lottery", "lottery", "permit lottery", or similar), otherwise false or null if unclear.

    3) deadline_date_text: The deadline date the answer provides for when 2026 lottery winners must claim their awarded reservation (e.g., "April 21, 2026", "Apr 21, 2026", or including a time like "April 21, 2026 at 11:59 PM PT"). If not stated, return null.

    4) per_person_fee_text: The per-person fee amount the answer states must be paid by that deadline to secure the permit (e.g., "$15 per person", "$15.00 per person recreation fee"). If not stated, return null.

    5) sources_all: Array of all URLs explicitly mentioned in the answer (include markdown link targets as plain URLs).

    6) sources_deadline: Array of URLs (subset of sources_all) that the answer appears to use as evidence for the deadline date. If unclear, return an empty array.

    7) sources_fee: Array of URLs (subset of sources_all) that the answer appears to use as evidence for the fee. If unclear, return an empty array.

    Important:
    - Do not invent any information; extract only what is explicitly present in the answer.
    - Preserve the exact formatting for dates and dollar amounts as shown in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    extraction: WhitneyAnswerExtraction,
) -> None:
    """
    Build the rubric tree and execute the three verifications as leaf nodes.
    Tree:
      root (parallel, non-critical)
        └── Complete_Question_Response (parallel, critical)
              ├── Peak_With_Annual_Lottery_Identified (leaf, critical)
              ├── Claim_Deadline_Date_2026 (leaf, critical)
              └── Mandatory_Per_Person_Fee (leaf, critical)
    """

    # Create the top-level rubric node (critical, parallel)
    cq_node = evaluator.add_parallel(
        id="Complete_Question_Response",
        desc="Answer identifies the correct peak and provides both the 2026 claim deadline date and the mandatory per-person fee required by that deadline.",
        parent=evaluator.root,
        critical=True
    )

    # Leaf 1: Peak identification and annual lottery mention
    peak_node = evaluator.add_leaf(
        id="Peak_With_Annual_Lottery_Identified",
        desc="States that Mount Whitney is the highest peak in the contiguous United States and that it uses an annual permit lottery system for hikers (i.e., identifies the correct peak described in the question).",
        parent=cq_node,
        critical=True
    )
    peak_claim = (
        "In the answer, the highest peak in the contiguous United States is identified as Mount Whitney (allow 'Mt. Whitney'), "
        "and the answer indicates that this peak uses an annual permit lottery system for hikers."
    )
    await evaluator.verify(
        claim=peak_claim,
        node=peak_node,
        additional_instruction=(
            "Judge based solely on the answer text. The answer must explicitly indicate the peak is Mount Whitney "
            "(accept 'Mt. Whitney') and explicitly mention an annual permit lottery (permit lottery) system for hikers. "
            "Do not require checking external webpages for this node."
        )
    )

    # Leaf 2: 2026 claim deadline date
    deadline_node = evaluator.add_leaf(
        id="Claim_Deadline_Date_2026",
        desc="Provides the deadline date by which 2026 lottery winners must claim their awarded reservation: April 21, 2026. (Time-of-day may be included but is not required if the date is correct.)",
        parent=cq_node,
        critical=True
    )
    deadline_claim = (
        "The answer states that the deadline date by which 2026 lottery winners must claim their awarded reservation is April 21, 2026. "
        "Minor formatting variations are acceptable (e.g., 'Apr 21, 2026', inclusion of the time-of-day)."
    )
    await evaluator.verify(
        claim=deadline_claim,
        node=deadline_node,
        additional_instruction=(
            "Check only the answer text for whether it clearly gives April 21, 2026 as the claim deadline date. "
            "Allow small format differences (e.g., missing comma, abbreviated month, explicit time-of-day)."
        )
    )

    # Leaf 3: Mandatory per-person fee
    fee_node = evaluator.add_leaf(
        id="Mandatory_Per_Person_Fee",
        desc="Provides the mandatory per-person fee that must be paid by the claim deadline to secure the permit: $15.00 per person recreation fee.",
        parent=cq_node,
        critical=True
    )
    fee_claim = (
        "The answer states that the mandatory per-person fee due by the claim deadline to secure the permit is $15 per person "
        "(i.e., the $15.00 per person recreation fee). Minor formatting differences like '$15' vs '$15.00' are acceptable."
    )
    await evaluator.verify(
        claim=fee_claim,
        node=fee_node,
        additional_instruction=(
            "Check only the answer text and confirm it explicitly states a per-person fee of $15 (the recreation fee) due by the deadline. "
            "Accept '$15' or '$15.00' and accept the phrase 'recreation fee' if provided."
        )
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
    Evaluate an answer for the Mount Whitney 2026 lottery claim deadline and fee task.
    Returns the standard evaluation summary dict from the evaluator.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root can be parallel; the critical gating is on the child node.
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

    # Extract structured facts from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_whitney_answer(),
        template_class=WhitneyAnswerExtraction,
        extraction_name="whitney_answer_extraction"
    )

    # Record expected ground truth info for reference in the summary
    evaluator.add_ground_truth(
        {
            "expected_peak": GROUND_TRUTH["peak"],
            "expected_2026_claim_deadline_date": GROUND_TRUTH["deadline_date_2026"],
            "expected_per_person_fee": GROUND_TRUTH["fee"]
        },
        gt_type="expected_answer_requirements"
    )

    # Also include extracted URLs and texts for debugging/traceability
    evaluator.add_custom_info(
        info={
            "extracted_peak_name": extraction.peak_name,
            "mentions_annual_lottery": extraction.mentions_annual_lottery,
            "extracted_deadline_date_text": extraction.deadline_date_text,
            "extracted_per_person_fee_text": extraction.per_person_fee_text,
            "sources_all": extraction.sources_all,
            "sources_deadline": extraction.sources_deadline,
            "sources_fee": extraction.sources_fee
        },
        info_type="extraction_debug",
        info_name="extraction_debug_info"
    )

    # Build and run verification tree
    await build_and_verify(evaluator, extraction)

    # Return the evaluation summary
    return evaluator.get_summary()