import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "ulta_beauty_roster_youngest_age_feb2026"
TASK_DESCRIPTION = "What is the age of the youngest athlete in the Ulta Beauty Roster announced in February 2026?"


class YoungestExtraction(BaseModel):
    youngest_age: Optional[str] = None
    youngest_athlete_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


def prompt_extract_youngest_info() -> str:
    return """
    Extract the requested information strictly from the provided answer text.

    Fields to extract:
    - youngest_age: The age (in years) that the answer claims for the youngest athlete in the Ulta Beauty Roster announced in February 2026. Prefer digits only if present (e.g., "19" from "19 years old"). If the answer gives a range or a non-numeric form (e.g., "about 19" or "19+"), return it exactly as written.
    - youngest_athlete_name: The name of the athlete that the answer claims is the youngest, if such a name is provided. If no name is provided, return null.
    - sources: All URLs explicitly provided in the answer that are used to support the claim (could be press releases, official announcements, news articles, or roster pages). Extract actual URLs only (including from markdown links).

    Rules:
    - Do not infer or invent. Only extract exactly what the answer states.
    - If youngest_age is present multiple times, take the one associated with the "youngest athlete" statement.
    - If the answer provides no URLs, return an empty list for sources.
    """


async def build_verification_tree(evaluator: Evaluator, root, extracted: YoungestExtraction) -> None:
    ac_node = evaluator.add_parallel(
        id="answer_correctness",
        desc="Answer addresses the Ulta Beauty Roster announced in February 2026 and provides the age of the youngest athlete in that roster.",
        parent=root,
        critical=False  # Keep parent non-critical to allow non-critical child; critical checks live on leaves
    )

    sources_list = extracted.sources or []

    # 1) Roster Context Correct (Critical)
    ctx_node = evaluator.add_leaf(
        id="roster_context_correct",
        desc="Answer is explicitly about the 'Ulta Beauty Roster' announcement in February 2026 (not a different Ulta partnership/roster/timeframe).",
        parent=ac_node,
        critical=True
    )
    if sources_list:
        await evaluator.verify(
            claim="The cited source(s) explicitly cover Ulta Beauty's 'Ulta Beauty Roster' announcement that took place in February 2026 (not a different roster, campaign, or timeframe).",
            node=ctx_node,
            sources=sources_list,
            additional_instruction="Verify the page(s) clearly reference an Ulta Beauty roster announcement and that it is anchored in February 2026. If the source is about a different year/month or a different initiative, mark as not supported."
        )
    else:
        # No sources to ground the context; mark as failed
        ctx_node.score = 0.0
        ctx_node.status = "failed"

    # 2) Youngest Age Provided (Critical)
    age_provided = bool(extracted.youngest_age and extracted.youngest_age.strip())
    evaluator.add_custom_node(
        result=age_provided,
        id="youngest_age_provided",
        desc="Answer provides an age for the youngest athlete in the roster.",
        parent=ac_node,
        critical=True
    )

    # 3) Age Is Numeric (Critical)
    numeric_node = evaluator.add_leaf(
        id="age_is_numeric",
        desc="The age is expressed as a number (e.g., an integer number of years).",
        parent=ac_node,
        critical=True
    )
    # Even if age is missing, the auto-precondition (critical sibling 'youngest_age_provided') will skip this
    await evaluator.verify(
        claim=f"The value '{extracted.youngest_age or ''}' represents an integer number of years.",
        node=numeric_node,
        sources=None,
        additional_instruction="Accept reasonable variants like '19', '19 years', or '19 yrs' as integer ages. Reject ranges (e.g., '18-19'), approximations (e.g., 'about 19'), or non-numeric forms (e.g., 'teenager')."
    )

    # 4) Youngest Age Correct (Critical)
    age_correct_node = evaluator.add_leaf(
        id="youngest_age_correct",
        desc="The provided age is correct for the youngest athlete in the Ulta Beauty Roster announced in February 2026 (i.e., matches the minimum age among roster athletes as of the announcement).",
        parent=ac_node,
        critical=True
    )
    if sources_list:
        await evaluator.verify(
            claim=f"As of the Ulta Beauty Roster announcement in February 2026, the youngest athlete's age is {extracted.youngest_age or ''} years (i.e., this equals the minimum age across the roster at that time).",
            node=age_correct_node,
            sources=sources_list,
            additional_instruction="Use the cited pages to identify the roster athletes and their ages (or birthdates) as of February 2026, and confirm that the claimed number is indeed the minimum. Allow minor rounding if the precise birthday falls very near the announcement date; otherwise, require explicit support."
        )
    else:
        age_correct_node.score = 0.0
        age_correct_node.status = "failed"

    # 5) Youngest Athlete Named (Non-Critical)
    name_present = bool(extracted.youngest_athlete_name and extracted.youngest_athlete_name.strip())
    evaluator.add_custom_node(
        result=name_present,
        id="youngest_athlete_named",
        desc="Answer also names which roster athlete is the youngest.",
        parent=ac_node,
        critical=False
    )


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

    extracted = await evaluator.extract(
        prompt=prompt_extract_youngest_info(),
        template_class=YoungestExtraction,
        extraction_name="youngest_info_extraction"
    )

    await build_verification_tree(evaluator, root, extracted)

    return evaluator.get_summary()