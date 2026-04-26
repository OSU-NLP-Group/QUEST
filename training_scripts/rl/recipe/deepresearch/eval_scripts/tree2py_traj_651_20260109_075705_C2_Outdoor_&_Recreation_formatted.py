import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


TASK_ID = "glacier_np_wilderness_permit_lottery_2025"
TASK_DESCRIPTION = (
    "For individuals planning a backpacking trip to Glacier National Park in summer 2025 with a group of 4–6 people, "
    "what is the application window (including the specific date and time range in Mountain Time) for the appropriate "
    "early access wilderness camping permit lottery, when will lottery winners be notified, and what is the non-refundable application fee?"
)


class GlacierLotteryExtraction(BaseModel):
    """Structured info extracted from the agent's answer about Glacier NP early access wilderness permit lottery."""
    lottery_name: Optional[str] = None
    application_window_text: Optional[str] = None
    winners_notification_date: Optional[str] = None
    application_fee_text: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


def prompt_extract_glacier_lottery_details() -> str:
    return (
        "Extract the details the answer provides about Glacier National Park's early access wilderness camping permit lottery "
        "that would be appropriate for a group of 4–6 backpackers in summer 2025. Return a JSON object with fields:\n"
        "1) lottery_name: The name of the lottery the answer identifies (e.g., 'Standard Group Lottery', "
        "'Standard Group Early Access Lottery').\n"
        "2) application_window_text: The application window as stated in the answer, including date and specific Mountain Time range "
        "(e.g., 'March 15, 2025 from 12:00 am (MT) to 11:59 pm (MT)').\n"
        "3) winners_notification_date: The date the answer states winners are notified (e.g., 'March 17, 2025').\n"
        "4) application_fee_text: The non-refundable application fee as stated (e.g., '$10').\n"
        "5) source_urls: An array of all URLs explicitly mentioned in the answer that pertain to this lottery "
        "(NPS or Recreation.gov pages). Extract actual URLs present in the answer text or markdown links.\n"
        "If any item is missing in the answer, return null for that field (or an empty array for source_urls). "
        "Do not invent or infer values."
    )


async def verify_correct_lottery_identification(
    evaluator: Evaluator,
    parent_node,
    details: GlacierLotteryExtraction,
) -> None:
    """
    Builds a critical sub-tree to check:
      - The answer identifies the appropriate lottery as 'Standard Group Lottery'.
      - The 'Standard Group Lottery' is designed for groups of 1–8 campers (source-supported if URLs are provided).
    """
    node = evaluator.add_parallel(
        id="Correct_Lottery_Identification",
        desc="Answer identifies the Standard Group Lottery, designed for groups of 1–8 campers.",
        parent=parent_node,
        critical=True,
    )

    # 1) Check that the answer explicitly identifies the appropriate lottery as 'Standard Group Lottery'
    name_leaf = evaluator.add_leaf(
        id="Lottery_Name_Identified_As_Standard_Group",
        desc="Answer identifies the appropriate lottery as 'Standard Group Lottery' (or an equivalent naming).",
        parent=node,
        critical=True,
    )
    claim_name = (
        "Within the answer, the identified lottery for Glacier NP early access wilderness camping permits "
        "appropriate for a 4–6 person group in 2025 is the 'Standard Group Lottery' (allowing minor naming variants "
        "like 'Standard Group Early Access Lottery')."
    )
    await evaluator.verify(
        claim=claim_name,
        node=name_leaf,
        additional_instruction=(
            "Judge only whether the answer identifies the appropriate lottery as 'Standard Group Lottery' or an obviously equivalent naming. "
            "Allow minor variants such as including 'Early Access' or capitalization differences."
        ),
    )

    # 2) Verify that the Standard Group Lottery is designed for groups of 1–8 campers (prefer source-supported)
    group_leaf = evaluator.add_leaf(
        id="Lottery_Group_Size_Designed_1_to_8",
        desc="Standard Group Lottery is designed for groups of 1–8 campers.",
        parent=node,
        critical=True,
    )
    claim_group = "The Standard Group Lottery is designed for groups of 1–8 campers."
    await evaluator.verify(
        claim=claim_group,
        node=group_leaf,
        sources=details.source_urls,
        additional_instruction=(
            "Use the cited webpages to confirm the group size range for the Standard Group Lottery is 1–8. "
            "Prefer Recreation.gov or official NPS sources about Glacier NP wilderness camping early access lotteries."
        ),
    )


async def verify_application_window(
    evaluator: Evaluator,
    parent_node,
    details: GlacierLotteryExtraction,
) -> None:
    """
    Builds a critical sub-tree to check the application window is correctly stated in the answer and supported by sources:
      - March 15, 2025 from 12:00 am (MT) to 11:59 pm (MT).
    """
    node = evaluator.add_parallel(
        id="Application_Window",
        desc="Answer provides the correct lottery application window: March 15, 2025 from 12:00 am (MT) to 11:59 pm (MT).",
        parent=parent_node,
        critical=True,
    )

    # 1) Verify the answer itself states the correct window (Mountain Time)
    window_answer_leaf = evaluator.add_leaf(
        id="Application_Window_Answer_Match",
        desc="The answer states the application window exactly as March 15, 2025 from 12:00 am (MT) to 11:59 pm (MT).",
        parent=node,
        critical=True,
    )
    claim_window_answer = (
        "The answer states the lottery application window is March 15, 2025 from 12:00 am (Mountain Time) to 11:59 pm (Mountain Time)."
    )
    await evaluator.verify(
        claim=claim_window_answer,
        node=window_answer_leaf,
        additional_instruction=(
            "Allow minor wording variations like 'midnight' for 12:00 am, but it must be Mountain Time and the date must be March 15, 2025 "
            "with the end time 11:59 pm."
        ),
    )

    # 2) Verify the window is supported by cited sources
    window_source_leaf = evaluator.add_leaf(
        id="Application_Window_Source_Supported",
        desc="The application window (March 15, 2025 from 12:00 am MT to 11:59 pm MT) is supported by the cited sources.",
        parent=node,
        critical=True,
    )
    claim_window_source = (
        "The lottery application window is March 15, 2025 from 12:00 am (MT) to 11:59 pm (MT)."
    )
    await evaluator.verify(
        claim=claim_window_source,
        node=window_source_leaf,
        sources=details.source_urls,
        additional_instruction=(
            "Confirm the exact date and Mountain Time range on official pages for Glacier NP wilderness camping early access lotteries."
        ),
    )


async def verify_winner_notification(
    evaluator: Evaluator,
    parent_node,
    details: GlacierLotteryExtraction,
) -> None:
    """
    Builds a critical sub-tree to check winners notification is March 17, 2025, both in-answer and source-supported.
    """
    node = evaluator.add_parallel(
        id="Winner_Notification_Date",
        desc="Answer states winners are notified on March 17, 2025.",
        parent=parent_node,
        critical=True,
    )

    # 1) Verify the answer states the correct notification date
    notify_answer_leaf = evaluator.add_leaf(
        id="Winner_Notification_Date_Answer_Match",
        desc="The answer states winners are notified on March 17, 2025.",
        parent=node,
        critical=True,
    )
    claim_notify_answer = "The answer states lottery winners are notified on March 17, 2025."
    await evaluator.verify(
        claim=claim_notify_answer,
        node=notify_answer_leaf,
        additional_instruction="Minor phrasing variants are okay, but the date must be March 17, 2025.",
    )

    # 2) Verify the notification date is supported by cited sources
    notify_source_leaf = evaluator.add_leaf(
        id="Winner_Notification_Date_Source_Supported",
        desc="Winners notification date (March 17, 2025) is supported by the cited sources.",
        parent=node,
        critical=True,
    )
    claim_notify_source = "Lottery winners are notified on March 17, 2025."
    await evaluator.verify(
        claim=claim_notify_source,
        node=notify_source_leaf,
        sources=details.source_urls,
        additional_instruction="Confirm the winners notification date on official pages for Glacier NP early access lotteries.",
    )


async def verify_application_fee(
    evaluator: Evaluator,
    parent_node,
    details: GlacierLotteryExtraction,
) -> None:
    """
    Builds a critical sub-tree to check the non-refundable application fee is $10, both in-answer and source-supported.
    """
    node = evaluator.add_parallel(
        id="Nonrefundable_Application_Fee",
        desc="Answer states the non-refundable application fee is $10.",
        parent=parent_node,
        critical=True,
    )

    # 1) Verify the answer states the correct fee
    fee_answer_leaf = evaluator.add_leaf(
        id="Application_Fee_Answer_Match",
        desc="The answer states the non-refundable application fee is $10.",
        parent=node,
        critical=True,
    )
    claim_fee_answer = "The answer states the non-refundable application fee is $10."
    await evaluator.verify(
        claim=claim_fee_answer,
        node=fee_answer_leaf,
        additional_instruction="Minor currency formatting variants are okay, but the value must be $10 and non-refundable.",
    )

    # 2) Verify the fee is supported by cited sources
    fee_source_leaf = evaluator.add_leaf(
        id="Application_Fee_Source_Supported",
        desc="The non-refundable application fee ($10) is supported by the cited sources.",
        parent=node,
        critical=True,
    )
    claim_fee_source = "The non-refundable application fee for the lottery is $10."
    await evaluator.verify(
        claim=claim_fee_source,
        node=fee_source_leaf,
        sources=details.source_urls,
        additional_instruction="Confirm the application fee amount and non-refundable nature on official pages.",
    )


async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point to evaluate an agent's answer for Glacier NP wilderness camping permit lottery details (summer 2025).
    Builds the verification tree per rubric and returns a structured summary.
    """
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

    extracted_details = await evaluator.extract(
        prompt=prompt_extract_glacier_lottery_details(),
        template_class=GlacierLotteryExtraction,
        extraction_name="glacier_lottery_extraction",
    )

    main_node = evaluator.add_parallel(
        id="Glacier_National_Park_Lottery_Information",
        desc=(
            "Evaluate whether the answer correctly identifies and provides complete information about the appropriate "
            "Glacier National Park wilderness camping permit lottery for a group of 4–6 people in summer 2025, per the stated constraints."
        ),
        parent=root,
        critical=True,
    )

    evaluator.add_ground_truth({
        "expected": {
            "lottery_name": "Standard Group Lottery (Early Access)",
            "group_size_range": "1–8 campers",
            "application_window_mt": "March 15, 2025 from 12:00 am (MT) to 11:59 pm (MT)",
            "winners_notification_date": "March 17, 2025",
            "application_fee": "$10 (non-refundable)",
        }
    }, gt_type="ground_truth")

    await verify_correct_lottery_identification(evaluator, main_node, extracted_details)
    await verify_application_window(evaluator, main_node, extracted_details)
    await verify_winner_notification(evaluator, main_node, extracted_details)
    await verify_application_fee(evaluator, main_node, extracted_details)

    return evaluator.get_summary()