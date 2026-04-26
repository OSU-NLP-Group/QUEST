import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "yaghi_affiliation_2025"
TASK_DESCRIPTION = "What university was Omar M. Yaghi affiliated with when he was awarded the 2025 Nobel Prize in Chemistry?"


class AffiliationExtraction(BaseModel):
    laureate_name: Optional[str] = None
    prize_field: Optional[str] = None
    prize_year: Optional[str] = None
    affiliation_university: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


def prompt_extract_affiliation() -> str:
    return (
        "From the answer, extract the following fields exactly as presented:\n"
        "1. laureate_name: The full name of the laureate the answer addresses.\n"
        "2. prize_field: The Nobel Prize field (e.g., Chemistry).\n"
        "3. prize_year: The year of the Nobel Prize (e.g., 2025).\n"
        "4. affiliation_university: The name of the university or institution stated as Omar M. Yaghi's affiliation at the time of the 2025 award.\n"
        "5. source_urls: All URLs cited in the answer that support the affiliation or the award context. Extract actual URLs if present (including markdown links).\n"
        "If any field is missing in the answer, set it to null (for strings) or empty list (for source_urls)."
    )


async def build_university_affiliation_verification(
    evaluator: Evaluator,
    parent_node,
    data: AffiliationExtraction,
) -> None:
    ua_node = evaluator.add_parallel(
        id="University_Affiliation_Identification",
        desc="Determine the university Omar M. Yaghi was affiliated with at the time he was awarded the 2025 Nobel Prize in Chemistry.",
        parent=parent_node,
        critical=True,
    )

    correct_laureate_node = evaluator.add_leaf(
        id="Correct_Laureate",
        desc="The answer addresses Omar M. Yaghi (not a different person) as the laureate in question.",
        parent=ua_node,
        critical=True,
    )
    extracted_name = (data.laureate_name or "").strip()
    if extracted_name:
        laureate_claim = f"The person named '{extracted_name}' in the answer refers to Omar M. Yaghi."
    else:
        laureate_claim = "The laureate discussed in the answer is Omar M. Yaghi."
    await evaluator.verify(
        claim=laureate_claim,
        node=correct_laureate_node,
        additional_instruction="Verify that the answer is about Omar M. Yaghi. Allow minor name variations, middle initials, or casing differences.",
    )

    correct_award_context_node = evaluator.add_leaf(
        id="Correct_Award_Context",
        desc="The answer is explicitly tied to the 2025 Nobel Prize in Chemistry (not a different prize, year, or field).",
        parent=ua_node,
        critical=True,
    )
    year_str = (data.prize_year or "").strip()
    field_str = (data.prize_field or "").strip()
    if year_str and field_str:
        context_claim = f"The award context described in the answer is the {year_str} Nobel Prize in {field_str}."
    else:
        context_claim = "The award context in the answer is the 2025 Nobel Prize in Chemistry."
    await evaluator.verify(
        claim=context_claim,
        node=correct_award_context_node,
        additional_instruction="Confirm the answer references the 2025 Nobel Prize in Chemistry, not a different year or field. Synonyms like '2025 Chemistry Nobel' are acceptable.",
    )

    affiliation_stated_node = evaluator.add_custom_node(
        result=bool(data.affiliation_university and data.affiliation_university.strip()),
        id="University_Affiliation_Stated",
        desc="The answer provides the name of a university/institution as Omar M. Yaghi's affiliation at the time of the award.",
        parent=ua_node,
        critical=True,
    )

    affiliation_correct_node = evaluator.add_leaf(
        id="Affiliation_Is_Correct_For_Award_Time",
        desc="The stated university/institution matches Omar M. Yaghi's actual affiliation at the time of the 2025 Nobel Prize in Chemistry award.",
        parent=ua_node,
        critical=True,
    )
    affil_name = (data.affiliation_university or "").strip()
    affil_claim = (
        f"At the time of the 2025 Nobel Prize in Chemistry award, Omar M. Yaghi was affiliated with {affil_name}."
        if affil_name
        else "At the time of the 2025 Nobel Prize in Chemistry award, Omar M. Yaghi's stated affiliation in the answer is correct."
    )
    await evaluator.verify(
        claim=affil_claim,
        node=affiliation_correct_node,
        sources=data.source_urls if data.source_urls else None,
        additional_instruction=(
            "Verify the affiliation at the time of the 2025 Chemistry Nobel announcement using the cited URLs. "
            "Prefer official or reputable sources (e.g., Nobel Prize press release, university news). "
            "Allow reasonable naming variants such as 'University of California, Berkeley' vs 'UC Berkeley'."
        ),
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
        prompt=prompt_extract_affiliation(),
        template_class=AffiliationExtraction,
        extraction_name="affiliation_extraction",
    )

    await build_university_affiliation_verification(evaluator, root, extracted)

    return evaluator.get_summary()