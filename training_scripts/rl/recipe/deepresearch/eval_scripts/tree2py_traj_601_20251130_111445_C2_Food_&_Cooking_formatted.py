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
TASK_ID = "thanksgiving_latest_closing_2024"
TASK_DESCRIPTION = "On Thanksgiving Day 2024 (November 28, 2024), among the major U.S. grocery store chains that remained open for at least part of the day, which chain stayed open the latest, and what time did it close?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ThanksgivingLatestClosingExtraction(BaseModel):
    """
    Structured extraction from the agent's answer:
    - chain_name: The chain claimed to have stayed open the latest on Thanksgiving Day 2024.
    - closing_time: The time claimed as the latest closing time.
    - sources: URLs provided in the answer that support the chain/time claim.
    """
    chain_name: Optional[str] = None
    closing_time: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_thanksgiving_latest() -> str:
    return """
    Extract the specific answer elements for the question:
    "On Thanksgiving Day 2024 (November 28, 2024), among the major U.S. grocery store chains that remained open for at least part of the day, which chain stayed open the latest, and what time did it close?"

    Return the following fields:
    1) chain_name: The name of the grocery chain the answer claims stayed open the latest.
       Accept reasonable variants such as "Kroger", "The Kroger Co.", "Kroger Family of Companies".
       If multiple chains are mentioned, choose the one explicitly indicated as the latest-open chain.
    2) closing_time: The latest closing time reported in the answer for Thanksgiving Day 2024.
       Accept formats like "5:00 PM", "5 PM", "5pm", "5 p.m.", etc. Return exactly the string form found in the answer.
    3) sources: All URLs cited in the answer that are intended to support the claim.
       Include any valid http(s) URLs mentioned. If none are provided, return an empty list.

    If any field is not stated, return null for that field (or an empty list for sources).
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extracted: ThanksgivingLatestClosingExtraction,
    root_node_desc: str
) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    """
    # Create the critical parent node mirroring the rubric root
    main_node = evaluator.add_parallel(
        id="Thanksgiving_Latest_Closing_Store",
        desc=root_node_desc,
        parent=evaluator.root,
        critical=True
    )

    # Child leaf 1: Chain identified as Kroger
    chain_leaf = evaluator.add_leaf(
        id="Latest_Open_Chain",
        desc="Answer identifies Kroger as the major grocery chain with the latest closing time among those open on Thanksgiving Day 2024.",
        parent=main_node,
        critical=True
    )

    # Construct claim using extracted chain_name but judging against Kroger
    extracted_chain = extracted.chain_name or ""
    chain_claim = (
        f"The chain identified in the answer as staying open the latest on Thanksgiving Day 2024 is Kroger. "
        f"(Extracted chain: '{extracted_chain}')"
    )
    await evaluator.verify(
        claim=chain_claim,
        node=chain_leaf,
        additional_instruction=(
            "Judge by the answer content whether the latest-open chain is Kroger. "
            "Treat 'Kroger', 'The Kroger Co.', and 'Kroger Family of Companies' as equivalent. "
            "Focus on whether the answer explicitly points to Kroger as the latest-open chain."
        )
    )

    # Child leaf 2: Latest closing time stated as 5:00 PM (equivalent formats accepted)
    time_leaf = evaluator.add_leaf(
        id="Latest_Closing_Time",
        desc="Answer states the latest closing time as 5:00 PM (or equivalent format) for Thanksgiving Day 2024.",
        parent=main_node,
        critical=True
    )

    extracted_time = extracted.closing_time or ""
    time_claim = (
        f"The latest closing time reported in the answer for Thanksgiving Day 2024 is 5:00 PM. "
        f"(Extracted time: '{extracted_time}')"
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_leaf,
        additional_instruction=(
            "Accept equivalent formats such as '5 PM', '5:00 pm', '5pm', '5 p.m.' as matching 5:00 PM. "
            "Focus on whether the answer indeed conveys 5 PM as the latest closing time."
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
    Entry point to evaluate the agent's answer for the Thanksgiving latest closing task.
    """
    # Initialize evaluator
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_thanksgiving_latest(),
        template_class=ThanksgivingLatestClosingExtraction,
        extraction_name="thanksgiving_latest_closing_extraction"
    )

    # Add ground truth expectation for clarity
    evaluator.add_ground_truth({
        "expected_chain": "Kroger",
        "expected_latest_closing_time": "5:00 PM (equivalents accepted)"
    }, gt_type="thanksgiving_2024_expectation")

    # Build verification tree and run checks
    await build_and_verify_tree(
        evaluator=evaluator,
        extracted=extracted,
        root_node_desc="Identify the major U.S. grocery chain that stayed open the latest on Thanksgiving Day 2024 and state its closing time."
    )

    # Return the evaluation summary
    return evaluator.get_summary()