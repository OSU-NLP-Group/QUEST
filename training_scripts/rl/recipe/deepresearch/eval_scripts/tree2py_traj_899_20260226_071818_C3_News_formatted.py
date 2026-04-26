import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "maduro_reward_progression"
TASK_DESCRIPTION = (
    "Document the complete progression of U.S. State Department reward amounts offered for information "
    "leading to the arrest and/or conviction of Nicolás Maduro from March 2020 through August 2025. For each reward milestone "
    "(initial offer and subsequent increases), provide: (1) the exact dollar amount offered, (2) the precise announcement date, "
    "and (3) reference URLs from credible sources that confirm each fact."
)

# Optional: reference info for logging/ground truth context (not used for judging logic)
EXPECTED_MILESTONES = [
    {
        "label": "Initial Offer",
        "expected_amount": "up to $15 million",
        "expected_announcement_date": "March 26, 2020",
        "note": "Narcotics Rewards Program; arrest and/or conviction purpose; connected to DOJ charges including narco-terrorism conspiracy, cocaine importation conspiracy, possession of machine guns and destructive devices.",
    },
    {
        "label": "First Increase",
        "expected_amount": "up to $25 million",
        "expected_announcement_date": "January 10, 2025",
        "note": "Narcotics Rewards Program; arrest and/or conviction purpose; connected to DOJ charges.",
    },
    {
        "label": "Second Increase",
        "expected_amount": "up to $50 million",
        "expected_announcement_date": "August 7, 2025",
        "note": "Narcotics Rewards Program; arrest and/or conviction purpose; connected to DOJ charges; claim of highest reward ever offered for a foreign head of state.",
    },
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Milestone(BaseModel):
    amount: Optional[str] = None
    announcement_date: Optional[str] = None
    program: Optional[str] = None
    purpose: Optional[str] = None
    charges_context: Optional[str] = None
    highest_reward_claim: Optional[str] = None  # Only relevant for milestone 3; leave null otherwise
    citations: List[str] = Field(default_factory=list)


class RewardProgressionExtraction(BaseModel):
    milestone_1: Optional[Milestone] = None   # Initial offer (March 2020)
    milestone_2: Optional[Milestone] = None   # First increase (January 2025)
    milestone_3: Optional[Milestone] = None   # Second increase (August 2025)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_progression() -> str:
    return """
Extract the progression of U.S. State Department reward amounts for information leading to the arrest and/or conviction of Nicolás Maduro from the provided answer. 
You must organize the information into three discrete milestones, exactly as follows:

- milestone_1 (Initial offer – March 2020)
- milestone_2 (First increase – January 2025)
- milestone_3 (Second increase – August 2025)

For each milestone, extract these fields from the answer text exactly as presented:
1) amount: The exact phrasing of the dollar amount offered (e.g., "up to $15 million", "$25 million", "USD 50,000,000", etc.).
2) announcement_date: The precise announcement date as written in the answer (e.g., "March 26, 2020", "Jan 10, 2025").
3) program: The program name or description as stated (e.g., "U.S. State Department's Narcotics Rewards Program").
4) purpose: The stated purpose (e.g., "for information leading to the arrest and/or conviction of Nicolás Maduro").
5) charges_context: The charges context as described in the answer (e.g., mentions of "narco-terrorism conspiracy", "cocaine importation conspiracy", "possession of machine guns and destructive devices"). Extract the wording as given.
6) highest_reward_claim: ONLY for milestone_3, if the answer claims something like "the highest reward ever offered for a foreign head of state", extract that sentence or phrase. Otherwise, return null.
7) citations: An array of all URL(s) explicitly cited in the answer that support this milestone's facts. 
   - Only include valid URLs that appear in the answer.
   - If no URLs are cited for the milestone, return an empty array.

If any field is missing in the answer for a milestone, set it to null (or an empty array for citations).
Return a single JSON object conforming to the RewardProgressionExtraction schema with keys milestone_1, milestone_2, milestone_3.
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_milestone(
    evaluator: Evaluator,
    parent_node,
    milestone_id: str,
    milestone_desc: str,
    milestone: Optional[Milestone],
    include_highest_claim: bool = False
) -> None:
    """
    Build the milestone subtree and perform verifications.

    - Parent (critical): parallel aggregation for the milestone.
    - First leaf (critical): citations existence.
    - Other leaves (critical): verify each required fact with the provided URLs.
    """
    # Create milestone parallel node (critical because its parent progression node is critical)
    ms_node = evaluator.add_parallel(
        id=milestone_id,
        desc=milestone_desc,
        parent=parent_node,
        critical=True
    )

    # Normalize milestone object
    ms = milestone or Milestone()

    # 1) Citations existence check (critical). Evaluate first to gate others.
    citations_exist = bool(ms.citations)
    citations_desc_extra = " (amount, date, program, purpose, charges context"
    if include_highest_claim:
        citations_desc_extra += ", and highest-reward claim"
    citations_desc_extra += ")."

    evaluator.add_custom_node(
        result=citations_exist,
        id=f"{milestone_id}_Citations",
        desc=f"Provides credible reference URL(s) that substantiate the required facts for this milestone{citations_desc_extra}",
        parent=ms_node,
        critical=True
    )

    # Prepare common additional instruction to emphasize scope and flexibility
    common_instruction = (
        "Only judge whether the provided webpage(s) explicitly support the claim for Nicolás Maduro's reward at this milestone. "
        "Allow minor formatting variations (e.g., '$15,000,000' vs '$15 million', 'arrest and/or conviction' vs 'arrest or conviction'). "
        "If a URL is irrelevant to this specific claim, treat it as not supported."
    )

    # 2) Amount
    amount_node = evaluator.add_leaf(
        id=f"{milestone_id}_Amount",
        desc="Reward amount is correctly stated for this milestone.",
        parent=ms_node,
        critical=True
    )
    amount_claim = (
        f"The announced reward amount for Nicolás Maduro at this milestone is stated as '{ms.amount}'."
        if ms.amount else
        "The announced reward amount for Nicolás Maduro at this milestone is correctly stated."
    )
    await evaluator.verify(
        claim=amount_claim,
        node=amount_node,
        sources=ms.citations,
        additional_instruction=(
            common_instruction + " Focus on whether the page confirms the amount for this milestone. "
            "Accept equivalent phrasing such as 'up to $X' vs '$X', and '$X million' vs '$X,000,000'."
        ),
    )

    # 3) Announcement date
    date_node = evaluator.add_leaf(
        id=f"{milestone_id}_Announcement_Date",
        desc="Reward announcement date is correctly stated for this milestone.",
        parent=ms_node,
        critical=True
    )
    date_claim = (
        f"The announcement date of this reward milestone is '{ms.announcement_date}'."
        if ms.announcement_date else
        "The announcement date of this reward milestone is correctly stated."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=ms.citations,
        additional_instruction=(
            common_instruction + " Verify the specific announcement date. "
            "Allow minor formatting variations (e.g., 'March 26, 2020' vs 'Mar 26, 2020')."
        ),
    )

    # 4) Program
    program_node = evaluator.add_leaf(
        id=f"{milestone_id}_Program",
        desc="This reward is offered through the U.S. State Department's Narcotics Rewards Program.",
        parent=ms_node,
        critical=True
    )
    program_claim = (
        f"The reward for Nicolás Maduro at this milestone is offered through '{ms.program}'."
        if ms.program else
        "The reward for Nicolás Maduro at this milestone is offered through the U.S. State Department's Narcotics Rewards Program."
    )
    await evaluator.verify(
        claim=program_claim,
        node=program_node,
        sources=ms.citations,
        additional_instruction=(
            common_instruction + " Confirm that the program is specifically the U.S. State Department's Narcotics Rewards Program (NRP), "
            "even if the exact phrasing varies."
        ),
    )

    # 5) Purpose
    purpose_node = evaluator.add_leaf(
        id=f"{milestone_id}_Purpose",
        desc="This reward is for information leading to the arrest and/or conviction of Nicolás Maduro.",
        parent=ms_node,
        critical=True
    )
    purpose_claim = (
        f"The stated purpose of the reward for this milestone is: '{ms.purpose}'."
        if ms.purpose else
        "The reward is for information leading to the arrest and/or conviction of Nicolás Maduro."
    )
    await evaluator.verify(
        claim=purpose_claim,
        node=purpose_node,
        sources=ms.citations,
        additional_instruction=(
            common_instruction + " Confirm the purpose refers to information leading to the arrest and/or conviction of Nicolás Maduro."
        ),
    )

    # 6) Charges Context
    charges_node = evaluator.add_leaf(
        id=f"{milestone_id}_Charges_Context",
        desc="Reward is connected to charges including narco-terrorism conspiracy, cocaine importation conspiracy, and possession of machine guns and destructive devices.",
        parent=ms_node,
        critical=True
    )
    charges_claim = (
        f"The reward is connected to charges including: {ms.charges_context}."
        if ms.charges_context else
        "The reward is connected to charges including narco-terrorism conspiracy, cocaine importation conspiracy, and possession of machine guns and destructive devices."
    )
    await evaluator.verify(
        claim=charges_claim,
        node=charges_node,
        sources=ms.citations,
        additional_instruction=(
            common_instruction + " Confirm the page ties the reward to DOJ charges against Nicolás Maduro, including narco-terrorism conspiracy, "
            "cocaine importation conspiracy, and possession of machine guns and destructive devices (allow minor phrasing differences)."
        ),
    )

    # 7) Highest-reward claim (milestone 3 only)
    if include_highest_claim:
        highest_node = evaluator.add_leaf(
            id=f"{milestone_id}_Highest_Reward_Claim",
            desc="States that the $50 million reward is the highest reward ever offered for a foreign head of state.",
            parent=ms_node,
            critical=True
        )
        highest_claim = (
            f"{ms.highest_reward_claim}"
            if ms.highest_reward_claim else
            "The $50 million reward is the highest reward ever offered for a foreign head of state."
        )
        await evaluator.verify(
            claim=highest_claim,
            node=highest_node,
            sources=ms.citations,
            additional_instruction=(
                common_instruction + " Verify that the page explicitly claims this is the highest reward ever offered for a foreign head of state "
                "by the U.S. Government (allow equivalent wording)."
            ),
        )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Nicolás Maduro reward progression task.
    """
    # Initialize evaluator (root is non-critical by design)
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

    # Record ground truth context for debugging (not used in judging)
    evaluator.add_ground_truth({"expected_milestones": EXPECTED_MILESTONES}, gt_type="ground_truth")

    # Extract structured progression information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_progression(),
        template_class=RewardProgressionExtraction,
        extraction_name="reward_progression"
    )

    # Build a critical sequential node for overall progression to reflect dependency order
    progression_node = evaluator.add_sequential(
        id="Maduro_Reward_Progression",
        desc="Verify the progression of U.S. State Department reward amounts for Nicolás Maduro (March 2020 through August 2025), with required attributes and supporting URLs.",
        parent=root,
        critical=True
    )

    # Milestone 1: Initial Offer (March 2020)
    await verify_milestone(
        evaluator=evaluator,
        parent_node=progression_node,
        milestone_id="Milestone_1_Initial_Offer",
        milestone_desc="Initial reward offer milestone (March 2020).",
        milestone=extraction.milestone_1,
        include_highest_claim=False
    )

    # Milestone 2: First Increase (January 2025)
    await verify_milestone(
        evaluator=evaluator,
        parent_node=progression_node,
        milestone_id="Milestone_2_First_Increase",
        milestone_desc="First reward increase milestone (January 2025).",
        milestone=extraction.milestone_2,
        include_highest_claim=False
    )

    # Milestone 3: Second Increase (August 2025)
    await verify_milestone(
        evaluator=evaluator,
        parent_node=progression_node,
        milestone_id="Milestone_3_Second_Increase",
        milestone_desc="Second reward increase milestone (August 2025).",
        milestone=extraction.milestone_3,
        include_highest_claim=True
    )

    # Return the evaluation summary
    return evaluator.get_summary()