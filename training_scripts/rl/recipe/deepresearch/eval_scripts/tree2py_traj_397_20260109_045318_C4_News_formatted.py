import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


TASK_ID = "marco_rubio_confirmation_2025"
TASK_DESCRIPTION = (
    "Research Marco Rubio's confirmation as U.S. Secretary of State in January 2025. Provide the following information: "
    "(1) the Cabinet position title, (2) the exact date of his Senate confirmation, (3) the Senate vote count including both the number of yes votes and the number of no votes, "
    "(4) the date he was sworn into office, (5) his sequential position number (e.g., '72nd Secretary of State'), (6) his previous government position held immediately before this appointment, "
    "and (7) reference URLs from official government sources or credible news outlets to support your information."
)


class ConfirmationExtraction(BaseModel):
    position_title: Optional[str] = None
    confirmation_date: Optional[str] = None
    vote_yes_count: Optional[str] = None
    vote_no_count: Optional[str] = None
    sworn_in_date: Optional[str] = None
    sequential_number: Optional[str] = None
    previous_position: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


def prompt_extract_confirmation_info() -> str:
    return (
        "Extract the structured information about Marco Rubio's confirmation as U.S. Secretary of State from the provided answer.\n"
        "Return a JSON object with the following fields:\n"
        "1. position_title: The Cabinet position title exactly as stated in the answer (e.g., 'U.S. Secretary of State')\n"
        "2. confirmation_date: The exact date of Senate confirmation exactly as written in the answer (e.g., 'January 20, 2025')\n"
        "3. vote_yes_count: The number of 'yes/yea/aye' votes for the confirmation (as a string extracted exactly from the answer). If multiple numbers are shown, extract the one explicitly linked to 'yes/yea/aye' votes.\n"
        "4. vote_no_count: The number of 'no/nay' votes for the confirmation (as a string extracted exactly from the answer). If multiple numbers are shown, extract the one explicitly linked to 'no/nay' votes.\n"
        "5. sworn_in_date: The date he was sworn into office exactly as written in the answer (e.g., 'January 21, 2025')\n"
        "6. sequential_number: The sequential position number with ordinal wording exactly as shown (e.g., '72nd Secretary of State')\n"
        "7. previous_position: The previous government position held immediately before this appointment (e.g., 'U.S. Senator from Florida')\n"
        "8. reference_urls: An array of all reference URLs explicitly provided in the answer that support these facts. Extract only valid URLs mentioned in the answer (including markdown links). Do not invent URLs.\n"
        "For any missing field, return null. For URLs, include complete URLs with protocol. If no URLs are given, return an empty array."
    )


async def verify_position_title(evaluator: Evaluator, parent_node, info: ConfirmationExtraction) -> None:
    node = evaluator.add_sequential(
        id="Position_Title",
        desc="Identifies the Cabinet position title as U.S. Secretary of State.",
        parent=parent_node,
        critical=True,
    )

    exists = bool(info.position_title and info.position_title.strip())
    evaluator.add_custom_node(
        result=exists,
        id="position_title_present",
        desc="Position title is provided in the answer",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="position_title_correct",
        desc="Position title matches 'U.S. Secretary of State' supported by sources",
        parent=node,
        critical=True,
    )
    title_value = info.position_title or ""
    claim = f"The Cabinet position title for Marco Rubio is '{title_value}', i.e., U.S. Secretary of State."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=info.reference_urls,
        additional_instruction=(
            "Confirm that Marco Rubio's position is U.S. Secretary of State. Allow minor wording variants such as 'Secretary of State' or 'United States Secretary of State'."
        ),
    )


async def verify_confirmation_date(evaluator: Evaluator, parent_node, info: ConfirmationExtraction) -> None:
    node = evaluator.add_sequential(
        id="Confirmation_Date",
        desc="States the exact date of Senate confirmation as January 20, 2025.",
        parent=parent_node,
        critical=True,
    )

    exists = bool(info.confirmation_date and info.confirmation_date.strip())
    evaluator.add_custom_node(
        result=exists,
        id="confirmation_date_present",
        desc="Confirmation date is provided in the answer",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="confirmation_date_supported",
        desc="The stated Senate confirmation date is supported by sources",
        parent=node,
        critical=True,
    )
    date_value = info.confirmation_date or ""
    claim = f"Marco Rubio was confirmed by the U.S. Senate on {date_value}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=info.reference_urls,
        additional_instruction=(
            "Verify the exact confirmation date from the cited sources. Accept common date formatting variants (e.g., 'Jan. 20, 2025')."
        ),
    )


async def verify_vote_counts(evaluator: Evaluator, parent_node, info: ConfirmationExtraction) -> None:
    node = evaluator.add_sequential(
        id="Vote_Counts",
        desc="Provides the Senate vote count for the confirmation including both the number of yes votes and the number of no votes (as numeric counts).",
        parent=parent_node,
        critical=True,
    )

    yes_present = bool(info.vote_yes_count and str(info.vote_yes_count).strip())
    no_present = bool(info.vote_no_count and str(info.vote_no_count).strip())
    evaluator.add_custom_node(
        result=yes_present and no_present,
        id="vote_counts_present",
        desc="Both yes and no vote counts are provided in the answer",
        parent=node,
        critical=True,
    )

    checks = evaluator.add_parallel(
        id="vote_counts_checks",
        desc="Vote counts are accurately supported by sources",
        parent=node,
        critical=True,
    )

    yes_leaf = evaluator.add_leaf(
        id="vote_yes_supported",
        desc="Yes vote count is supported by sources",
        parent=checks,
        critical=True,
    )
    yes_val = info.vote_yes_count or ""
    yes_claim = (
        f"The Senate recorded {yes_val} 'yes/yea/aye' votes for Marco Rubio's confirmation as U.S. Secretary of State."
    )
    await evaluator.verify(
        claim=yes_claim,
        node=yes_leaf,
        sources=info.reference_urls,
        additional_instruction=(
            "Verify the number of affirmative votes (yes/yea/aye). Allow vocabulary variants like 'yeas', 'ayes', or 'in favor'."
        ),
    )

    no_leaf = evaluator.add_leaf(
        id="vote_no_supported",
        desc="No vote count is supported by sources",
        parent=checks,
        critical=True,
    )
    no_val = info.vote_no_count or ""
    no_claim = (
        f"The Senate recorded {no_val} 'no/nay' votes for Marco Rubio's confirmation as U.S. Secretary of State."
    )
    await evaluator.verify(
        claim=no_claim,
        node=no_leaf,
        sources=info.reference_urls,
        additional_instruction=(
            "Verify the number of negative votes (no/nay). Allow vocabulary variants like 'nays' or 'against'."
        ),
    )


async def verify_sworn_in_date(evaluator: Evaluator, parent_node, info: ConfirmationExtraction) -> None:
    node = evaluator.add_sequential(
        id="Swearing_In_Date",
        desc="States the date he was sworn into office as January 21, 2025.",
        parent=parent_node,
        critical=True,
    )

    exists = bool(info.sworn_in_date and info.sworn_in_date.strip())
    evaluator.add_custom_node(
        result=exists,
        id="sworn_in_date_present",
        desc="Swearing-in date is provided in the answer",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="sworn_in_date_supported",
        desc="The stated swearing-in date is supported by sources",
        parent=node,
        critical=True,
    )
    date_val = info.sworn_in_date or ""
    claim = f"Marco Rubio was sworn into office on {date_val}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=info.reference_urls,
        additional_instruction=(
            "Verify the exact swearing-in date from the cited sources. Accept common date formatting variants."
        ),
    )


async def verify_sequential_number(evaluator: Evaluator, parent_node, info: ConfirmationExtraction) -> None:
    node = evaluator.add_sequential(
        id="Sequential_Number",
        desc="Identifies his sequential position number as the 72nd Secretary of State.",
        parent=parent_node,
        critical=True,
    )

    exists = bool(info.sequential_number and info.sequential_number.strip())
    evaluator.add_custom_node(
        result=exists,
        id="sequential_number_present",
        desc="Sequential position number is provided in the answer",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="sequential_number_supported",
        desc="The stated sequential position number is supported by sources",
        parent=node,
        critical=True,
    )
    seq_val = info.sequential_number or ""
    claim = f"Marco Rubio is the {seq_val}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=info.reference_urls,
        additional_instruction=(
            "Verify that the ordinal (e.g., '72nd Secretary of State') matches the sources. Allow minor variants like 'the 72nd U.S. Secretary of State'."
        ),
    )


async def verify_previous_position(evaluator: Evaluator, parent_node, info: ConfirmationExtraction) -> None:
    node = evaluator.add_sequential(
        id="Previous_Position",
        desc="Identifies his previous government position held immediately before appointment as U.S. Senator from Florida.",
        parent=parent_node,
        critical=True,
    )

    exists = bool(info.previous_position and info.previous_position.strip())
    evaluator.add_custom_node(
        result=exists,
        id="previous_position_present",
        desc="Previous government position is provided in the answer",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="previous_position_supported",
        desc="The stated previous position is supported by sources",
        parent=node,
        critical=True,
    )
    prev_val = info.previous_position or ""
    claim = f"Immediately prior to this appointment, Marco Rubio served as {prev_val}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=info.reference_urls,
        additional_instruction=(
            "Confirm that his most recent prior role immediately before becoming Secretary of State was U.S. Senator from Florida. Allow minor wording variants."
        ),
    )


async def verify_reference_urls(evaluator: Evaluator, parent_node, info: ConfirmationExtraction) -> None:
    node = evaluator.add_parallel(
        id="Reference_URLs",
        desc="Provides reference URL(s) from official government sources or credible news outlets that support the provided information.",
        parent=parent_node,
        critical=True,
    )

    provided = bool(info.reference_urls and len(info.reference_urls) > 0)
    evaluator.add_custom_node(
        result=provided,
        id="reference_urls_provided",
        desc="At least one reference URL is provided in the answer",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="reference_urls_credible",
        desc="Reference URLs are from official government sources or credible mainstream news outlets",
        parent=node,
        critical=True,
    )
    urls_str = ", ".join(info.reference_urls) if info.reference_urls else "none"
    claim = (
        f"The provided reference URLs are official government websites (.gov, state.gov, senate.gov, whitehouse.gov) "
        f"or credible mainstream news outlets (e.g., AP, Reuters, WSJ, NYT, Washington Post, BBC, NPR, Politico, etc.): {urls_str}"
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction=(
            "Judge credibility based on the domain only. .gov domains are official. Recognize well-known major outlets "
            "like AP, Reuters, WSJ, NYT, Washington Post, BBC, NPR, Politico, ABC, CBS, NBC, CNN, FT, Bloomberg."
        ),
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_confirmation_info(),
        template_class=ConfirmationExtraction,
        extraction_name="confirmation_extraction",
    )

    top = evaluator.add_parallel(
        id="Marco_Rubio_Confirmation_Research",
        desc="Provides all required information about Marco Rubio's confirmation as U.S. Secretary of State in January 2025, supported by credible references.",
        parent=root,
        critical=True,
    )

    await verify_position_title(evaluator, top, extracted)
    await verify_confirmation_date(evaluator, top, extracted)
    await verify_vote_counts(evaluator, top, extracted)
    await verify_sworn_in_date(evaluator, top, extracted)
    await verify_sequential_number(evaluator, top, extracted)
    await verify_previous_position(evaluator, top, extracted)
    await verify_reference_urls(evaluator, top, extracted)

    return evaluator.get_summary()