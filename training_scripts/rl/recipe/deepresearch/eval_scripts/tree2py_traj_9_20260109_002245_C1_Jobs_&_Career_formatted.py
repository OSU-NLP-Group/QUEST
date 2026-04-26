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
TASK_ID = "amazon_rto_2024"
TASK_DESCRIPTION = """
Which major tech company announced in September 2024 that employees must return to the office five days per week starting in January 2025, and who made this announcement?
"""

# Ground truth facts expected in a correct answer
GROUND_TRUTH = {
    "company": "Amazon",
    "policy": "five days per week",
    "announcement_date": "September 16, 2024",
    "announcer": "Andy Jassy",
    "announcer_title": "CEO",
    "effective_date": "January 2, 2025",
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RTOAnnouncement(BaseModel):
    """Structured extraction of key facts from the answer."""
    company: Optional[str] = None
    policy_days_per_week: Optional[str] = None
    announcement_date: Optional[str] = None
    announcer_name: Optional[str] = None
    announcer_title: Optional[str] = None
    effective_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_rto_info() -> str:
    return """
    Extract the following fields from the answer text, capturing them exactly as stated in the answer:
    1. company: The name of the company that made the return-to-office announcement.
    2. policy_days_per_week: The stated requirement for office attendance (e.g., "five days per week", "5 days/week").
    3. announcement_date: The date the announcement was made (e.g., "September 16, 2024"). Preserve the formatting as written.
    4. announcer_name: The person who made the announcement (e.g., "Andy Jassy").
    5. announcer_title: That person's title/role (e.g., "CEO").
    6. effective_date: The date the policy becomes effective (e.g., "January 2, 2025"). Preserve formatting as written.
    7. sources: A list of all URLs explicitly provided in the answer as sources (include plain URLs and URLs in markdown links). Do not invent URLs.

    Rules:
    - If any field is not present in the answer, set it to null (or empty list for sources).
    - Do not normalize or rephrase values; extract them verbatim from the answer.
    - For URLs, include only valid-looking links; if a protocol is missing, prepend "http://" to the URL.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verifications(evaluator: Evaluator, extracted: RTOAnnouncement) -> None:
    """
    Build the verification tree based on the rubric and run verifications.
    """
    # Create the top-level critical parallel node reflecting the rubric's "Answer_Evaluation"
    answer_eval_node = evaluator.add_parallel(
        id="Answer_Evaluation",
        desc="Evaluates whether the answer satisfies all stated constraints for the return-to-office announcement.",
        parent=evaluator.root,
        critical=True
    )

    # 1) Company Identification: Amazon
    company_node = evaluator.add_leaf(
        id="Company_Identification",
        desc="Answer identifies Amazon as the company.",
        parent=answer_eval_node,
        critical=True
    )
    company_claim = (
        f"The answer identifies the company as Amazon. "
        f"Extracted company: '{extracted.company}'. Consider it correct if the answer clearly indicates Amazon."
    )
    await evaluator.verify(
        claim=company_claim,
        node=company_node,
        additional_instruction="Focus only on the answer text. Accept minor variants like 'Amazon.com' or 'Amazon (AMZN)'."
    )

    # 2) Policy: Five days per week
    policy_node = evaluator.add_leaf(
        id="Policy_Five_Days_Per_Week",
        desc="Answer states the policy requires working in the office five days per week.",
        parent=answer_eval_node,
        critical=True
    )
    policy_claim = (
        f"The answer states that employees must work from the office five days per week. "
        f"Extracted policy text: '{extracted.policy_days_per_week}'."
    )
    await evaluator.verify(
        claim=policy_claim,
        node=policy_node,
        additional_instruction=(
            "Accept equivalent phrasing such as 'five days a week', '5 days/week', or 'Monday through Friday'. "
            "The requirement must clearly be five office days per week."
        )
    )

    # 3) Announcement date: September 16, 2024
    ann_date_node = evaluator.add_leaf(
        id="Announcement_Date",
        desc="Answer states the announcement date is September 16, 2024 (September 2024 specifically on the 16th).",
        parent=answer_eval_node,
        critical=True
    )
    ann_date_claim = (
        f"The answer states the announcement was made on September 16, 2024. "
        f"Extracted announcement date: '{extracted.announcement_date}'."
    )
    await evaluator.verify(
        claim=ann_date_claim,
        node=ann_date_node,
        additional_instruction="Allow minor formatting variants like 'Sept. 16, 2024' or '2024-09-16', but the calendar date must be the 16th of September, 2024."
    )

    # 4) Announcer identity: Andy Jassy
    announcer_node = evaluator.add_leaf(
        id="Announcer_Identity",
        desc="Answer identifies Andy Jassy as the person who made the announcement.",
        parent=answer_eval_node,
        critical=True
    )
    announcer_claim = (
        f"The answer identifies Andy Jassy as the person who made the announcement. "
        f"Extracted announcer: '{extracted.announcer_name}'."
    )
    await evaluator.verify(
        claim=announcer_claim,
        node=announcer_node,
        additional_instruction="Accept minor name variants (e.g., with or without middle initial). The person must be Andy Jassy."
    )

    # 5) Announcer title: CEO
    title_node = evaluator.add_leaf(
        id="Announcer_Title",
        desc="Answer states that Andy Jassy's role/title is CEO.",
        parent=answer_eval_node,
        critical=True
    )
    title_claim = (
        f"The answer states Andy Jassy's title/role is CEO. "
        f"Extracted title: '{extracted.announcer_title}'."
    )
    await evaluator.verify(
        claim=title_claim,
        node=title_node,
        additional_instruction="Accept equivalent phrasing such as 'Chief Executive Officer' or 'Amazon CEO'."
    )

    # 6) Effective date: January 2, 2025
    effective_node = evaluator.add_leaf(
        id="Effective_Date",
        desc="Answer states the policy becomes effective on January 2, 2025.",
        parent=answer_eval_node,
        critical=True
    )
    effective_claim = (
        f"The answer states the policy becomes effective on January 2, 2025. "
        f"Extracted effective date: '{extracted.effective_date}'."
    )
    await evaluator.verify(
        claim=effective_claim,
        node=effective_node,
        additional_instruction="Allow minor formatting variants such as 'Jan 2, 2025' or '2025-01-02'."
    )

    # 7) Sourcing
    # Create a critical sequential group to gate the credibility check by existence of sources
    sourcing_group = evaluator.add_sequential(
        id="Sourcing_Group",
        desc="Sourcing verification sequence (existence and credibility).",
        parent=answer_eval_node,
        critical=True
    )

    sources_exist = bool(extracted.sources)
    sources_exist_node = evaluator.add_custom_node(
        result=sources_exist,
        id="Sources_Provided",
        desc="Answer provides at least one source URL.",
        parent=sourcing_group,
        critical=True
    )

    sourcing_leaf = evaluator.add_leaf(
        id="Sourcing_Requirement",
        desc="Answer provides sourcing from official Amazon communications or other credible news sources.",
        parent=sourcing_group,
        critical=True
    )
    sourcing_claim = (
        "This webpage is either an official Amazon communication (e.g., About Amazon blog or press release on amazon.com) "
        "or a credible news article reporting Amazon's return-to-office policy requiring five days per week, announced in September 2024 and effective January 2025."
    )
    await evaluator.verify(
        claim=sourcing_claim,
        node=sourcing_leaf,
        sources=extracted.sources,
        additional_instruction=(
            "Treat official Amazon sources as pages under aboutamazon.com or amazon.com corporate communications. "
            "Credible news sources include major outlets like Reuters, Bloomberg, WSJ, CNBC, AP, The Verge, etc. "
            "Pass if at least one provided URL clearly supports the described policy and context."
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
    Evaluate an agent's answer for the Amazon return-to-office announcement task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root can be parallel; rubric's main node added beneath
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
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_rto_info(),
        template_class=RTOAnnouncement,
        extraction_name="rto_announcement_info"
    )

    # Record ground truth for reference
    evaluator.add_ground_truth({
        "expected_company": GROUND_TRUTH["company"],
        "expected_policy": GROUND_TRUTH["policy"],
        "expected_announcement_date": GROUND_TRUTH["announcement_date"],
        "expected_announcer": GROUND_TRUTH["announcer"],
        "expected_announcer_title": GROUND_TRUTH["announcer_title"],
        "expected_effective_date": GROUND_TRUTH["effective_date"],
    })

    # Build and run verifications
    await build_verifications(evaluator, extracted_info)

    # Return structured summary
    return evaluator.get_summary()