import asyncio
import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

TASK_ID = "nbc_award_2019"
TASK_DESCRIPTION = "On February 24, 2025, a veteran broadcast journalist announced he was stepping down from his role as weekday anchor of NBC Nightly News after serving in that position for 10 years. What is the full name of the journalism award he received in 2019?"

EXPECTED_AWARD_FULL_NAME = "Walter Cronkite Award for Excellence in Journalism"


class AwardExtraction(BaseModel):
    award_2019_name: Optional[str] = None


def prompt_extract_award_2019() -> str:
    return (
        "From the answer, extract the full formal name of the journalism award that the journalist received in 2019."
        " Return this exactly as written in the answer (do not normalize or abbreviate)."
        " Use the JSON field: award_2019_name."
        " If the answer does not explicitly mention an award tied to 2019, return null."
        " If multiple awards are mentioned, choose the one associated with 2019."
    )


async def build_verification_tree(evaluator: Evaluator, root_node, extracted: AwardExtraction) -> None:
    answer_verif = evaluator.add_parallel(
        id="Answer_Verification",
        desc="Verifies that the answer correctly identifies the full name of the journalism award received in 2019.",
        parent=root_node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Award_Full_Name",
        desc="The answer identifies the award received in 2019 as the Walter Cronkite Award for Excellence in Journalism (full name).",
        parent=answer_verif,
        critical=True,
    )

    claim = (
        f"The answer identifies the award received in 2019 as the {EXPECTED_AWARD_FULL_NAME}."
    )

    add_ins = (
        "Check the answer text only. The answer must clearly state the full formal award name "
        f"'{EXPECTED_AWARD_FULL_NAME}' for the 2019 award. Shorthand like 'Cronkite Award' or partial names "
        "without 'for Excellence in Journalism' should be considered incorrect. "
        f"For reference, the extracted award_2019_name from the answer is: {extracted.award_2019_name or 'null'}."
    )

    await evaluator.verify(
        claim=claim,
        node=leaf,
        additional_instruction=add_ins,
    )


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
        prompt=prompt_extract_award_2019(),
        template_class=AwardExtraction,
        extraction_name="award_2019_extraction",
    )

    evaluator.add_ground_truth(
        {
            "expected_award_2019_full_name": EXPECTED_AWARD_FULL_NAME,
            "task": "Identify the full name of the journalism award received in 2019",
        }
    )

    await build_verification_tree(evaluator, root, extracted)

    return evaluator.get_summary()