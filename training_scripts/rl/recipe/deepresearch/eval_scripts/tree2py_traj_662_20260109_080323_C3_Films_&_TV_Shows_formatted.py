import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "coen_mccarthy_2007_film"
TASK_DESCRIPTION = """
In 2007, the Coen Brothers released a film adapted from a Cormac McCarthy novel, with cinematography by Roger Deakins.
What was the publication date of this source novel, and which publishing house released it?
"""


class NovelPublicationExtraction(BaseModel):
    film_title: Optional[str] = None
    novel_title: Optional[str] = None
    publication_date: Optional[str] = None
    publishing_house: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


def prompt_extract_novel_publication() -> str:
    return """
    Identify the film described in the question (a 2007 Coen Brothers film with cinematography by Roger Deakins) and the source novel by Cormac McCarthy mentioned in the answer, then extract the following exactly as presented in the answer text:

    - film_title: The title of the film, if mentioned in the answer.
    - novel_title: The title of the Cormac McCarthy novel that the film is adapted from, if mentioned.
    - publication_date: The publication date (or year) of the source novel as stated in the answer. If multiple dates are mentioned, prefer the one clearly identified as the original publication date.
    - publishing_house: The publishing house (or imprint) that released the source novel as stated in the answer.
    - source_urls: All URLs cited in the answer that are intended to support these details (the film-to-novel adaptation, the novel’s publication date, and/or the publisher). Extract actual URLs only; include both direct and indirect supporting URLs if the answer presents them.

    Rules:
    - Extract exactly what appears in the answer; do not infer missing items from your own knowledge.
    - If any field is not present in the answer, return null for that field (or an empty list for source_urls).
    - For URLs, accept plain or markdown-formatted links and normalize them to full URLs.
    """


async def verify_task_completion(
    evaluator: Evaluator,
    parent_node,
    extraction: NovelPublicationExtraction,
) -> None:
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Provide the publication date and publishing house of the source novel for the film described in the question.",
        parent=parent_node,
        critical=True,
    )

    # Publication Date leaf
    pub_date_leaf = evaluator.add_leaf(
        id="Publication_Date",
        desc="Answer includes the correct publication date of the source novel corresponding to the film described.",
        parent=task_node,
        critical=True,
    )

    if extraction.publication_date is None or not str(extraction.publication_date).strip():
        pub_date_leaf.score = 0.0
        pub_date_leaf.status = "failed"
    else:
        novel_ref = (
            f"the Cormac McCarthy novel '{extraction.novel_title}'"
            if extraction.novel_title
            else "the source novel by Cormac McCarthy"
        )
        claim_date = f"{novel_ref} was published on {extraction.publication_date}."
        await evaluator.verify(
            claim=claim_date,
            node=pub_date_leaf,
            sources=extraction.source_urls,
            additional_instruction=(
                "Verify that the provided URLs explicitly support the novel's original publication date/year. "
                "Allow minor formatting differences (e.g., 'July 11, 2005' vs '2005') as equivalent if clearly referring to the same publication. "
                "Ensure the page is about the source novel that the 2007 Coen Brothers film (cinematography by Roger Deakins) was adapted from."
            ),
        )

    # Publishing House leaf
    publisher_leaf = evaluator.add_leaf(
        id="Publishing_House",
        desc="Answer includes the correct publishing house that released the source novel corresponding to the film described.",
        parent=task_node,
        critical=True,
    )

    if extraction.publishing_house is None or not str(extraction.publishing_house).strip():
        publisher_leaf.score = 0.0
        publisher_leaf.status = "failed"
    else:
        novel_ref = (
            f"'{extraction.novel_title}'"
            if extraction.novel_title
            else "the source novel by Cormac McCarthy"
        )
        claim_publisher = f"The publishing house that released {novel_ref} is '{extraction.publishing_house}'."
        await evaluator.verify(
            claim=claim_publisher,
            node=publisher_leaf,
            sources=extraction.source_urls,
            additional_instruction=(
                "Verify that the provided URLs explicitly support the publisher/imprint of the source novel. "
                "Accept reasonable imprint vs parent-house equivalences (e.g., Knopf vs Alfred A. Knopf) when clearly indicated by the source. "
                "Ensure the page is about the same book that the 2007 Coen Brothers film (cinematography by Roger Deakins) adapted."
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

    extraction = await evaluator.extract(
        prompt=prompt_extract_novel_publication(),
        template_class=NovelPublicationExtraction,
        extraction_name="novel_publication_info",
    )

    await verify_task_completion(evaluator, root, extraction)

    return evaluator.get_summary()