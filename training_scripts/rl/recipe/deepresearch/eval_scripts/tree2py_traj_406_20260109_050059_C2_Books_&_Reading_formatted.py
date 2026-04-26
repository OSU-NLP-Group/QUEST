import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "booker_2024_winner_details"
TASK_DESCRIPTION = "What is the title of the book that won the 2024 Booker Prize, and what are its publication details including the publisher and original publication date?"


class BookerWinnerExtraction(BaseModel):
    winner_title: Optional[str] = None
    winner_sources: List[str] = Field(default_factory=list)
    publisher: Optional[str] = None
    publisher_sources: List[str] = Field(default_factory=list)
    original_publication_date: Optional[str] = None
    date_sources: List[str] = Field(default_factory=list)


def prompt_extract_booker_winner() -> str:
    return """
    Extract the following information from the answer regarding the 2024 Booker Prize winner.
    Fields to extract:
    1. winner_title: The exact title of the book that won the 2024 Booker Prize.
    2. winner_sources: A list of all URLs explicitly cited in the answer that support or announce the 2024 Booker Prize winner and/or the title of the winning book. Include any official Booker Prize webpages or credible news sites if provided. If the answer has a general sources section, include all relevant URLs here.
    3. publisher: The name of the publisher of the winning book (prefer the original edition's publisher; if multiple imprints or regional publishers are mentioned, extract the one explicitly stated in the answer).
    4. publisher_sources: A list of all URLs explicitly cited in the answer that support the publisher claim. If the answer only provides a general sources section without field-specific mapping, copy those URLs here as well.
    5. original_publication_date: The original publication date of the winning book (keep the string exactly as written in the answer; accept formats like YYYY-MM-DD, Month YYYY, or Year).
    6. date_sources: A list of all URLs explicitly cited in the answer that support the original publication date. If the answer only provides a general sources section, copy those URLs here as well.

    GENERAL RULES:
    - Extract only information explicitly present in the answer text.
    - For any missing field, set it to null (for strings) or an empty list (for URLs).
    - For URLs, extract the actual links (including markdown links). If missing protocol, prepend http://.
    - Do not invent or infer any information not present in the answer.
    """


def _combine_sources(primary: List[str], fallback: List[str]) -> List[str]:
    seen = set()
    combined = []
    for url in (primary or []) + (fallback or []):
        if url and url.strip() and url not in seen:
            seen.add(url)
            combined.append(url)
    return combined


async def build_verification_tree(
    evaluator: Evaluator,
    extracted: BookerWinnerExtraction,
) -> None:
    booker_node = evaluator.add_parallel(
        id="Booker_Prize_2024_Winner_Details",
        desc="Answer identifies the title of the 2024 Booker Prize winning book and provides its publication details (publisher and original publication date).",
        parent=evaluator.root,
        critical=True,
    )

    title_exists = bool(extracted.winner_title and extracted.winner_title.strip())
    title_sources = extracted.winner_sources or []
    title_has_sources = len(title_sources) > 0

    evaluator.add_custom_node(
        result=title_exists and title_has_sources,
        id="Winner_Title_Provided",
        desc="Winner title is provided and has at least one source URL",
        parent=booker_node,
        critical=True,
    )

    winner_title_leaf = evaluator.add_leaf(
        id="Winner_Title",
        desc="States the correct title of the book that won the 2024 Booker Prize (verifiable against authoritative sources).",
        parent=booker_node,
        critical=True,
    )

    winner_claim = f"The book that won the 2024 Booker Prize is titled '{extracted.winner_title or ''}'."
    await evaluator.verify(
        claim=winner_claim,
        node=winner_title_leaf,
        sources=title_sources,
        additional_instruction="Verify the page(s) explicitly announce the 2024 Booker Prize winner and that the title string matches, allowing minor formatting or punctuation variations.",
    )

    pub_details_node = evaluator.add_parallel(
        id="Publication_Details",
        desc="Provides publication details for the winning book: publisher and original publication date.",
        parent=booker_node,
        critical=True,
    )

    combined_publisher_sources = _combine_sources(extracted.publisher_sources, extracted.winner_sources)
    publisher_exists = bool(extracted.publisher and extracted.publisher.strip())
    publisher_has_sources = len(combined_publisher_sources) > 0

    evaluator.add_custom_node(
        result=publisher_exists and publisher_has_sources,
        id="Publisher_Provided",
        desc="Publisher is provided and has at least one supporting source URL",
        parent=pub_details_node,
        critical=True,
    )

    publisher_leaf = evaluator.add_leaf(
        id="Publisher",
        desc="Gives the publisher of the winning book (verifiable against authoritative sources).",
        parent=pub_details_node,
        critical=True,
    )

    publisher_claim = f"The publisher of '{extracted.winner_title or ''}' is '{extracted.publisher or ''}'."
    await evaluator.verify(
        claim=publisher_claim,
        node=publisher_leaf,
        sources=combined_publisher_sources,
        additional_instruction="Confirm the source page states the publisher of the original edition or explicitly identifies the publisher for the book; allow imprint-level naming if clearly tied to the publisher.",
    )

    combined_date_sources = _combine_sources(extracted.date_sources, extracted.winner_sources)
    date_exists = bool(extracted.original_publication_date and extracted.original_publication_date.strip())
    date_has_sources = len(combined_date_sources) > 0

    evaluator.add_custom_node(
        result=date_exists and date_has_sources,
        id="Original_Publication_Date_Provided",
        desc="Original publication date is provided and has at least one supporting source URL",
        parent=pub_details_node,
        critical=True,
    )

    original_pub_date_leaf = evaluator.add_leaf(
        id="Original_Publication_Date",
        desc="Gives the original publication date of the winning book (verifiable against authoritative sources).",
        parent=pub_details_node,
        critical=True,
    )

    date_claim = f"The original publication date of '{extracted.winner_title or ''}' is '{extracted.original_publication_date or ''}'."
    await evaluator.verify(
        claim=date_claim,
        node=original_pub_date_leaf,
        sources=combined_date_sources,
        additional_instruction="Allow reasonable date format variations (e.g., 'Month YYYY' vs 'YYYY-MM-DD'). Confirm this is the original publication date, not a reissue or later edition.",
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
    evaluator.initialize(
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
        prompt=prompt_extract_booker_winner(),
        template_class=BookerWinnerExtraction,
        extraction_name="booker_2024_winner_extraction",
    )

    await build_verification_tree(evaluator, extracted)

    return evaluator.get_summary()