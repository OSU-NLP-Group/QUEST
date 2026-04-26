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
TASK_ID = "venezuela_travel_advisory"
TASK_DESCRIPTION = "What is the current U.S. State Department travel advisory level for Venezuela, when was the advisory most recently reissued, and provide at least one specific reason cited in the advisory for the travel restriction?"

# Official State Department advisory URL(s) for Venezuela (used as fallback if the answer lacks sources)
DEFAULT_STATE_DEPT_VEN_URLS = [
    "https://travel.state.gov/content/travel/en/traveladvisories/traveladvisories/venezuela-travel-advisory.html"
]

EXPECTED_ADVISORY_LEVEL = "Level 4: Do Not Travel"
EXPECTED_REISSUE_DATE = "December 3, 2025"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AdvisoryExtraction(BaseModel):
    advisory_level: Optional[str] = None
    reissue_date: Optional[str] = None
    reasons: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_advisory_info() -> str:
    return """
    Extract the Venezuela travel advisory information stated in the answer. Return the following fields:
    - advisory_level: The advisory level text for Venezuela as stated in the answer (e.g., "Level 4: Do Not Travel"). Extract exactly as written in the answer.
    - reissue_date: The most recent reissue date stated in the answer (e.g., "December 3, 2025"). Extract exactly as written, allowing common date formats.
    - reasons: A list of specific reasons cited in the answer that the advisory mentions (e.g., "wrongful detention", "torture", "terrorism", "kidnapping", "arbitrary enforcement", "crime", "civil unrest", "poor health infrastructure"). Each reason should be a short phrase as it appears in the answer.
    - source_urls: All URLs provided in the answer that are cited as sources for the advisory (prefer official U.S. State Department advisory pages).
    If any field is missing in the answer, set it to null (for strings) or an empty list (for arrays).
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_venezuela_advisory(
    evaluator: Evaluator,
    parent_node,
    extracted: AdvisoryExtraction,
) -> None:
    """
    Build and execute verification checks for Venezuela travel advisory.
    Implements three critical leaf checks under a critical parallel node.
    """

    # Create the main critical node (as per rubric)
    main_node = evaluator.add_parallel(
        id="Venezuela_Travel_Advisory",
        desc="Verify the current U.S. State Department travel advisory information for Venezuela",
        parent=parent_node,
        critical=True
    )

    # Prepare sources: use those extracted from the answer; if none, fall back to official State Dept URL(s)
    sources_to_use: List[str] = extracted.source_urls if extracted.source_urls else DEFAULT_STATE_DEPT_VEN_URLS

    # 1) Advisory Level verification (critical leaf)
    level_node = evaluator.add_leaf(
        id="Advisory_Level",
        desc="The travel advisory level is correctly identified as Level 4: Do Not Travel",
        parent=main_node,
        critical=True
    )
    level_value = extracted.advisory_level or ""
    level_claim = (
        f"According to the official U.S. State Department advisory page(s), the current travel advisory level for "
        f"Venezuela is '{level_value}'."
    )
    await evaluator.verify(
        claim=level_claim,
        node=level_node,
        sources=sources_to_use,
        additional_instruction=(
            "Focus on the official U.S. State Department Travel Advisory page for Venezuela. "
            "Determine whether the page clearly states that the advisory level is Level 4: Do Not Travel. "
            "Allow minor phrasing variations (e.g., 'Level Four') but it must clearly correspond to Level 4. "
            "If the answer's stated level differs from the page, mark as incorrect."
        ),
    )

    # 2) Reissue Date verification (critical leaf)
    date_node = evaluator.add_leaf(
        id="Reissue_Date",
        desc="The reissue date of the advisory is correctly identified as December 3, 2025",
        parent=main_node,
        critical=True
    )
    reissue_value = extracted.reissue_date or ""
    reissue_claim = (
        f"The official advisory for Venezuela indicates it was reissued on '{reissue_value}'."
    )
    await evaluator.verify(
        claim=reissue_claim,
        node=date_node,
        sources=sources_to_use,
        additional_instruction=(
            "Check the advisory page(s) for 'Reissued on' or similar language indicating the most recent reissue date. "
            f"The expected correct date is '{EXPECTED_REISSUE_DATE}'. Allow minor formatting variations "
            "(e.g., 'Dec 3, 2025', 'December 03, 2025'), but the underlying date must match. "
            "If the answer's date does not match the page, mark as incorrect."
        ),
    )

    # 3) Cited Reason verification (critical leaf)
    reason_node = evaluator.add_leaf(
        id="Cited_Reason",
        desc="At least one specific reason cited in the advisory is provided (e.g., wrongful detention, torture, terrorism, kidnapping, arbitrary enforcement, crime, civil unrest, or poor health infrastructure)",
        parent=main_node,
        critical=True
    )
    first_reason = (extracted.reasons[0] if extracted.reasons else "").strip()
    reason_claim = (
        f"The advisory page for Venezuela cites '{first_reason}' as a specific reason for the travel restriction."
    )
    await evaluator.verify(
        claim=reason_claim,
        node=reason_node,
        sources=sources_to_use,
        additional_instruction=(
            "Verify that the advisory page explicitly mentions the cited reason as part of its justification. "
            "Accept closely-related phrasing or synonyms. Examples include: wrongful detention, torture, terrorism, "
            "kidnapping, arbitrary enforcement, crime, civil unrest, poor health infrastructure, or similar. "
            "If the answer does not provide any reason, or if the cited reason is not supported by the page, mark as incorrect."
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
    Evaluate an answer for the Venezuela travel advisory task.
    """
    # Initialize evaluator with parallel root strategy
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

    # Extract advisory information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_advisory_info(),
        template_class=AdvisoryExtraction,
        extraction_name="venezuela_travel_advisory_extraction",
    )

    # Record expected ground truth targets for transparency
    evaluator.add_ground_truth({
        "expected_advisory_level": EXPECTED_ADVISORY_LEVEL,
        "expected_reissue_date": EXPECTED_REISSUE_DATE,
        "verification_focus": "U.S. State Department Travel Advisory page(s) for Venezuela"
    })

    # Perform verification according to rubric
    await verify_venezuela_advisory(evaluator, root, extracted_info)

    # Return structured summary
    return evaluator.get_summary()