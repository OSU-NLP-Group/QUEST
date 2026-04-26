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
TASK_ID = "doj_boasberg_complaint_date_2025"
TASK_DESCRIPTION = "What date did the U.S. Department of Justice file a misconduct complaint against Chief Judge James Boasberg in 2025?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ComplaintExtraction(BaseModel):
    """
    Structured extraction of the key facts as stated in the answer.
    """
    date: Optional[str] = None  # The specific calendar date stated in the answer for the filing
    event_description: Optional[str] = None  # The event the answer claims occurred on that date (as stated)
    target: Optional[str] = None  # The person/office the complaint was against (as stated)
    sources: List[str] = Field(default_factory=list)  # All URLs cited in the answer supporting this claim


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_complaint_info() -> str:
    return """
    Extract the specific information the answer claims about the DOJ misconduct complaint involving Chief Judge James Boasberg.

    Required fields:
    1) date: The exact calendar date that the answer claims is when the U.S. Department of Justice filed the misconduct complaint.
       - Return the date exactly as written in the answer (e.g., "January 5, 2025", "Jan. 5, 2025", "2025-01-05", or "1/5/2025").
       - If multiple dates are mentioned, choose the date explicitly tied to the filing action by the DOJ.
       - If the answer does not provide a clear specific date, return null.

    2) event_description: A short phrase or sentence directly quoting or closely paraphrasing what the answer says happened on that date (e.g., "DOJ filed a misconduct complaint", "news article published", etc.). If unclear or missing, return null.

    3) target: The person or role the complaint was against as stated in the answer (e.g., "Chief Judge James Boasberg"). If not clearly stated, return null.

    4) sources: A list of all URLs cited in the answer that are presented as evidence for this claim.
       - Include URLs presented in plain text or markdown (extract the actual link).
       - Only include valid, explicit URLs found in the answer.
       - If no sources are provided, return an empty array.

    Only extract what is present in the answer; do not infer or fabricate any values.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree_and_verify(evaluator: Evaluator, extracted: ComplaintExtraction) -> None:
    """
    Build the rubric tree and perform verifications according to the provided rubric.
    """

    # Top-level critical node: all children must be critical (framework enforces this)
    main_node = evaluator.add_parallel(
        id="Complete_and_Accurate_Answer",
        desc="Answer identifies the specific 2025 date when the U.S. Department of Justice filed a misconduct complaint against Chief Judge James Boasberg.",
        parent=evaluator.root,
        critical=True
    )

    # Leaf 1: Provides_Specific_Date_in_2025
    date_leaf = evaluator.add_leaf(
        id="Provides_Specific_Date_in_2025",
        desc="Answer states a specific calendar date (month/day/year) that falls in 2025.",
        parent=main_node,
        critical=True
    )

    if extracted.date:
        claim_date = f"The answer states the filing date as '{extracted.date}', it is a specific calendar date (month/day/year format or equivalent), and the year is 2025."
    else:
        # Fall back to a general presence check in the answer text
        claim_date = "The answer explicitly states a specific calendar date for the filing (with month, day, and year), and that year is 2025."

    await evaluator.verify(
        claim=claim_date,
        node=date_leaf,
        additional_instruction=(
            "Judge only based on the provided answer text. Accept common date formats like 'January 5, 2025', "
            "'Jan. 5, 2025', '2025-01-05', or '1/5/2025'. The date must be a specific calendar date and the year must be 2025. "
            "If the answer provides no clear date, or uses a non-specific reference (e.g., 'early 2025'), mark incorrect."
        )
    )

    # Leaf 2: Date_Is_When_DOJ_Filed_Misconduct_Complaint
    event_leaf = evaluator.add_leaf(
        id="Date_Is_When_DOJ_Filed_Misconduct_Complaint",
        desc="Answer makes clear the stated date corresponds to the DOJ filing a misconduct complaint (not another related event such as reporting, allegations, hearings, or publication).",
        parent=main_node,
        critical=True
    )

    if extracted.date:
        claim_event = (
            f"On {extracted.date}, the U.S. Department of Justice filed a misconduct complaint. "
            f"This date refers to the filing action itself (the act of filing the complaint), not merely a report, article publication, hearing, or other follow-up event."
        )
    else:
        claim_event = (
            "The date stated in the answer corresponds to the filing by the U.S. Department of Justice of a misconduct complaint. "
            "It is the filing event date, not just a report, article publication, hearing, or other related milestone."
        )

    await evaluator.verify(
        claim=claim_event,
        node=event_leaf,
        sources=extracted.sources if extracted.sources else None,
        additional_instruction=(
            "Use the cited webpages if provided. Confirm that the event on that date was explicitly the DOJ filing a misconduct complaint. "
            "Accept references to 'Department of Justice', 'U.S. Department of Justice', or 'DOJ' as equivalent. "
            "Do not accept dates referring only to news publication, blog posts, hearings, or announcements unless they explicitly state the filing occurred on that same date."
        )
    )

    # Leaf 3: Complaint_Target_Is_Chief_Judge_James_Boasberg
    target_leaf = evaluator.add_leaf(
        id="Complaint_Target_Is_Chief_Judge_James_Boasberg",
        desc="Answer clearly indicates the misconduct complaint was against Chief Judge James Boasberg.",
        parent=main_node,
        critical=True
    )

    # Claim focusing solely on target identity (avoid mixing with other checks)
    claim_target = (
        "The misconduct complaint referenced in the answer was against Chief Judge James Boasberg."
    )

    await evaluator.verify(
        claim=claim_target,
        node=target_leaf,
        sources=extracted.sources if extracted.sources else None,
        additional_instruction=(
            "Use the cited webpages if available. Accept reasonable name variants like 'James E. Boasberg' and references to his role as Chief Judge. "
            "Confirm that he is the target of the misconduct complaint, not merely mentioned."
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
    Evaluate an answer for the DOJ-Boasberg complaint date task and return a structured result dictionary.
    """
    # Initialize evaluator with a parallel root (default)
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
        default_model=model
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_complaint_info(),
        template_class=ComplaintExtraction,
        extraction_name="complaint_extraction"
    )

    # Build tree and run verifications
    await build_verification_tree_and_verify(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()