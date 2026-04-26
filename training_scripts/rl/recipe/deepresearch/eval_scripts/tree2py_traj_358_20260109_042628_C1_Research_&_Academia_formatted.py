import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "natureindex_gs_2025_realtime_detection"
TASK_DESCRIPTION = "In the 2025 Google Scholar most influential papers ranking reported by Nature Index, a paper about real-time object detection was authored by researchers from Tsinghua University in Beijing. What is the paper's title and how many citations did it receive according to this ranking?"


class QualifyingPaper(BaseModel):
    title: Optional[str] = None
    citation_count: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


def prompt_extract_paper_info() -> str:
    return (
        "Identify the specific paper referenced in the answer that matches the following criteria: "
        "it appears in the Nature Index report covering the 2025 Google Scholar most influential papers ranking, "
        "it is about real-time object detection, and it is authored by researchers from Tsinghua University in Beijing. "
        "Extract the following fields:\n"
        "1) title: The title of the qualifying paper as stated in the answer.\n"
        "2) citation_count: The number of citations the paper received according to that Nature Index ranking/report, exactly as stated in the answer (keep formatting as-is; return as a string).\n"
        "3) sources: All URLs explicitly cited in the answer that point to the Nature Index report or relevant pages for the 2025 Google Scholar most influential papers ranking. "
        "Extract only valid URLs that appear in the answer (including markdown links). If no URLs are present, return an empty list."
    )


async def verify_paper_information(
    evaluator: Evaluator,
    parent_node,
    extracted: QualifyingPaper,
) -> None:
    paper_node = evaluator.add_sequential(
        id="Paper_Information",
        desc="Answer identifies the qualifying paper in the Nature Index report of the 2025 Google Scholar most influential papers ranking and reports its title and citation count from that ranking.",
        parent=parent_node,
        critical=True,
    )

    title = extracted.title or "Unknown title (not provided)"
    citations = extracted.citation_count or "Unknown citations (not provided)"
    sources = extracted.sources if extracted.sources else []

    identify_node = evaluator.add_leaf(
        id="Identify_Qualifying_Paper",
        desc="The referenced paper is one that appears in the Nature Index report of the 2025 Google Scholar most influential papers ranking, is about real-time object detection, and is authored by researchers from Tsinghua University in Beijing.",
        parent=paper_node,
        critical=True,
    )
    identify_claim = (
        f"In the Nature Index report covering the 2025 Google Scholar most influential papers ranking, "
        f"the paper titled '{title}' is included; the paper is about real-time object detection and has authors affiliated with Tsinghua University in Beijing."
    )
    await evaluator.verify(
        claim=identify_claim,
        node=identify_node,
        sources=sources,
        additional_instruction=(
            "Verify directly on the provided Nature Index page(s) whether a paper about real-time object detection authored by researchers from Tsinghua University in Beijing "
            "is listed for the 2025 Google Scholar most influential papers ranking. Allow minor wording variations (e.g., 'real time' vs 'real-time'; affiliation phrasing), "
            "but the page must clearly support inclusion, topic, and Tsinghua affiliation."
        ),
    )

    title_node = evaluator.add_leaf(
        id="Provide_Paper_Title",
        desc="Provides the paper's title corresponding to the qualifying paper identified from the ranking/report.",
        parent=paper_node,
        critical=True,
    )
    title_claim = (
        f"The Nature Index report/page lists the qualifying paper with the title '{title}', or an equivalent title with minor formatting variations."
    )
    await evaluator.verify(
        claim=title_claim,
        node=title_node,
        sources=sources,
        additional_instruction=(
            "Check the Nature Index page(s) to confirm the paper’s title. Allow reasonable variants or formatting differences, "
            "such as punctuation, capitalization, or minor spacing differences, as long as it clearly refers to the same paper."
        ),
    )

    citations_node = evaluator.add_leaf(
        id="Provide_Citation_Count",
        desc="Provides the paper's citation count as reported in the same ranking/report for the qualifying paper.",
        parent=paper_node,
        critical=True,
    )
    citations_claim = (
        f"According to the Nature Index report covering the 2025 Google Scholar most influential papers ranking, "
        f"the paper '{title}' has '{citations}' citations (match the numeric value or clearly equivalent representation)."
    )
    await evaluator.verify(
        claim=citations_claim,
        node=citations_node,
        sources=sources,
        additional_instruction=(
            "Locate the citation count reported on the Nature Index page for this ranking. Match the value provided in the answer, allowing reasonable numeric formatting differences "
            "(e.g., thousand separators, abbreviations like 1k = 1000, or minor rounding). The value must be clearly supported by the page."
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
        strategy=AggregationStrategy.SEQUENTIAL,
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
        prompt=prompt_extract_paper_info(),
        template_class=QualifyingPaper,
        extraction_name="paper_info",
    )

    evaluator.add_custom_info(
        info={"extracted_title": extracted.title, "extracted_citations": extracted.citation_count, "extracted_sources": extracted.sources},
        info_type="extraction_overview",
        info_name="paper_extraction_overview",
    )

    await verify_paper_information(evaluator, root, extracted)

    return evaluator.get_summary()