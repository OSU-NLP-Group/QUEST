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
TASK_ID = "pulitzer_bnp_2025"
TASK_DESCRIPTION = "Who won the 2025 Pulitzer Prize for Breaking News Photography, and which news organization were they affiliated with?"

# Ground truth context for summary and guidance
GROUND_TRUTH = {
    "category": "Breaking News Photography",
    "year": 2025,
    "winner_name": "Doug Mills",
    "news_organization": "The New York Times",
    "award_date": "May 5, 2025",
    "winning_work_description": "A sequence of photos of the attempted assassination of then-presidential candidate Donald Trump",
    "event_date": "July 13, 2024",
    "bullet_mid_air_image_included": True,
}


# --------------------------------------------------------------------------- #
# Data models for extracting information from the agent's answer              #
# --------------------------------------------------------------------------- #
class PulitzerExtraction(BaseModel):
    photographer_name: Optional[str] = None
    news_organization: Optional[str] = None
    award_date: Optional[str] = None
    winning_work_description: Optional[str] = None
    event_date: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pulitzer_info() -> str:
    return """
    You must extract exactly and only what the answer explicitly states regarding the 2025 Pulitzer Prize for Breaking News Photography.

    Extract the following fields:
    1) photographer_name: The person the answer states won the 2025 Pulitzer Prize for Breaking News Photography. If not stated, return null.
    2) news_organization: The news organization the answer states the winner is affiliated with. Prefer the exact phrase used (e.g., 'The New York Times' or 'NYT'). If not stated, return null.
    3) award_date: The date the answer states the award was announced or given (e.g., 'May 5, 2025'). If not stated, return null.
    4) winning_work_description: The description the answer gives for the winning work (e.g., 'a sequence of photos of the attempted assassination of then-presidential candidate Donald Trump'). If not stated, return null.
    5) event_date: The date the answer states the photographed event occurred (e.g., 'July 13' or 'July 13, 2024'). If not stated, return null.
    6) urls: An array of all explicit URLs (including markdown links) the answer cites as sources for these claims. If none are provided, return an empty array.

    Important:
    - Do not infer or invent information not present in the answer.
    - For URLs, include only actual URLs appearing in the answer text (plain or in markdown).
    - Use strings for all date fields as they may include various formats.
    """


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def verify_pulitzer_bnp_2025(
    evaluator: Evaluator,
    root: Any,
    extracted: PulitzerExtraction,
) -> None:
    """
    Build and verify all rubric leaves under a critical parallel node for the
    2025 Pulitzer Prize (Breaking News Photography) constraints.
    """
    # Create the critical parent node (parallel aggregation)
    parent_node = evaluator.add_parallel(
        id="2025_Pulitzer_Breaking_News_Photography",
        desc="Verify all required constraints for the 2025 Pulitzer Prize for Breaking News Photography winner and affiliation",
        parent=root,
        critical=True,
    )

    # Use URLs provided in the agent's answer; if none, we'll mark leaves failed to penalize lack of evidence
    sources = extracted.urls if extracted and extracted.urls else []

    # Helper to add a leaf and either verify via sources or fail if sources are missing
    async def add_and_verify_leaf(
        node_id: str,
        desc: str,
        claim: str,
        add_ins: str,
    ) -> None:
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=parent_node,
            critical=True,
        )
        if not sources:
            # No sources provided -> fail this critical check to penalize unsupported claims
            leaf.score = 0.0
            leaf.status = "failed"
            return
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=sources,
            additional_instruction=add_ins,
        )

    # Photographer_Name: Winner is Doug Mills
    await add_and_verify_leaf(
        node_id="Photographer_Name",
        desc="Winner is Doug Mills",
        claim="The winner of the 2025 Pulitzer Prize for Breaking News Photography is Doug Mills.",
        add_ins=(
            "Verify specifically the 2025 'Breaking News Photography' category (not other Pulitzer categories). "
            "Prefer official Pulitzer website or highly credible sources. "
            "Allow minor variations in name formatting (e.g., 'Douglas' vs 'Doug')."
        ),
    )

    # News_Organization: The New York Times
    await add_and_verify_leaf(
        node_id="News_Organization",
        desc="Winner's affiliated news organization is The New York Times",
        claim="Doug Mills is affiliated with The New York Times.",
        add_ins=(
            "Confirm the winner's affiliation at the time of the winning work. "
            "Treat 'NYT' and 'The New York Times' as equivalent. "
            "Use credible sources; the Pulitzer citation often lists affiliation."
        ),
    )

    # Award_Date: May 5, 2025
    await add_and_verify_leaf(
        node_id="Award_Date",
        desc="Award date is May 5, 2025",
        claim="The 2025 Pulitzer Prizes (including Breaking News Photography) were announced on May 5, 2025.",
        add_ins=(
            "Focus on the official Pulitzer site or reliable news coverage of the 2025 announcement. "
            "Accept equivalent wording such as 'announced May 5, 2025'."
        ),
    )

    # Winning_Work_Description
    await add_and_verify_leaf(
        node_id="Winning_Work_Description",
        desc="Winning work is a sequence of photos of the attempted assassination of then-presidential candidate Donald Trump",
        claim="Doug Mills won for a sequence of photos of the attempted assassination of then-presidential candidate Donald Trump.",
        add_ins=(
            "Accept synonyms such as 'assassination attempt' or 'attempted assassination'. "
            "References to the rally in Butler, Pennsylvania are consistent context."
        ),
    )

    # Event_Date: July 13 (the event occurred on July 13, 2024)
    await add_and_verify_leaf(
        node_id="Event_Date",
        desc="The photographed event occurred on July 13",
        claim="The attempted assassination of Donald Trump occurred on July 13, 2024.",
        add_ins=(
            "Only the month and day 'July 13' are essential for this check; "
            "year may be '2024'. If the source states 'July 13' without a year, consider it acceptable."
        ),
    )

    # Bullet_Mid_Air_Image
    await add_and_verify_leaf(
        node_id="Bullet_Mid_Air_Image",
        desc="The sequence includes an image capturing a bullet in mid-air",
        claim="The winning sequence includes an image capturing a bullet in mid-air.",
        add_ins=(
            "Confirm that one of the images in the winning sequence visibly shows a bullet mid-flight. "
            "Accept equivalent phrasing like 'bullet visible in the air' or 'bullet in flight'."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the agent's answer for the 2025 Pulitzer Prize (Breaking News Photography) question.
    """
    # Initialize evaluator with a parallel root (we'll attach a critical parallel subtree under it)
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_pulitzer_info(),
        template_class=PulitzerExtraction,
        extraction_name="pulitzer_extraction",
    )

    # Record ground truth/reference info
    evaluator.add_ground_truth(
        {
            "category": GROUND_TRUTH["category"],
            "year": GROUND_TRUTH["year"],
            "winner_name": GROUND_TRUTH["winner_name"],
            "news_organization": GROUND_TRUTH["news_organization"],
            "award_date": GROUND_TRUTH["award_date"],
            "winning_work_description": GROUND_TRUTH["winning_work_description"],
            "event_date": GROUND_TRUTH["event_date"],
            "bullet_mid_air_image_included": GROUND_TRUTH["bullet_mid_air_image_included"],
        },
        gt_type="ground_truth_pulitzer_bnp_2025",
    )

    # Add custom info for diagnostics (what the agent stated and number of URLs)
    evaluator.add_custom_info(
        {
            "agent_claims": {
                "photographer_name": extracted.photographer_name,
                "news_organization": extracted.news_organization,
                "award_date": extracted.award_date,
                "winning_work_description": extracted.winning_work_description,
                "event_date": extracted.event_date,
            },
            "source_urls_count": len(extracted.urls),
            "source_urls": extracted.urls,
        },
        info_type="diagnostics",
        info_name="agent_answer_diagnostics",
    )

    # Build verification subtree and run checks
    await verify_pulitzer_bnp_2025(evaluator, root, extracted)

    # Return summary
    return evaluator.get_summary()