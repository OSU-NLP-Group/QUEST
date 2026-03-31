import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "openreview_papers"
TASK_DESCRIPTION = """
Find 3 papers on OpenReview that were accepted to ICLR main conference hosted in the last year, where the 'Related Work' sections do not appear anywhere in the main body of the paper (the main body is defined as all sections appearing before the references and appendix). Provide direct OpenReview PDF links and forum links for each paper.
"""

JUDGE_MODEL = "o4-mini"
ICLR_YEAR = datetime.utcnow().year - 1

# --------------------------------------------------------------------------- #
# Data models for extracting information                                      #
# --------------------------------------------------------------------------- #
class PaperInfo(BaseModel):
    title: Optional[str] = None
    pdf_link: Optional[str] = None
    forum_link: Optional[str] = None


class ExtractedPapers(BaseModel):
    papers: List[PaperInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_paper_links() -> str:
    return """
    Extract a list of all papers mentioned in the answer. For each paper, extract:
    1. The title of the paper (if mentioned)
    2. The PDF link to the paper (usually, it's the URL containing "pdf" or ending with ".pdf"; or explicitly mentioned as the PDF link in the answer)
    3. The forum link to the paper (usually, it's the OpenReview URL that does not contain "pdf"; or explicitly mentioned as the forum link in the answer)

    If a link is not provided or cannot be clearly identified, return null for that field.
    If no papers are mentioned, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Verification functions for individual papers                                #
# --------------------------------------------------------------------------- #
async def verify_paper(
        evaluator: Evaluator,
        parent_node,
        paper_info: PaperInfo,
        paper_idx: int
) -> None:
    """
    Verify a single paper meets all requirements using parallel verification.
    """
    paper_node = evaluator.add_parallel(
        id=f"paper_{paper_idx}",
        desc=f"Paper {paper_idx + 1}: '{paper_info.title or 'Untitled'}' meets all requirements",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )

    # Create parent node for verifications
    verification_parent = evaluator.add_parallel(
        id=f"paper_{paper_idx}_verifications",
        desc=f"All verifications for paper {paper_idx + 1}",
        parent=paper_node,
        critical=False,
    )

    # Combined existence check for all required data
    data_exists = evaluator.add_custom_node(
        result=(
            paper_info.title is not None and 
            paper_info.title.strip() != "" and
            paper_info.pdf_link is not None and
            paper_info.pdf_link.strip() != "" and
            paper_info.forum_link is not None and
            paper_info.forum_link.strip() != ""
        ),
        id=f"paper_{paper_idx}_data_exists",
        desc=f"Check if paper {paper_idx + 1} has title, PDF link, and forum link",
        parent=verification_parent,
        critical=True
    )

    # 1. Verify PDF link correspondence
    pdf_correspondence_node = evaluator.add_leaf(
        id=f"paper_{paper_idx}_pdf_correspondence",
        desc=f"Paper {paper_idx + 1}: PDF link corresponds to the paper title '{paper_info.title or 'Untitled'}'",
        parent=verification_parent,
        critical=True,
    )

    await evaluator.verify(
        claim=f"This is a PDF for the paper titled '{paper_info.title}'",
        node=pdf_correspondence_node,
        sources=paper_info.pdf_link,
        additional_instruction="If you cannot tell if this page is a pdf since it's parsed into text and screenshots, just check if all the information given looks like content extracted from a PDF for this paper.",
    )

    # 2. Verify Forum link correspondence
    forum_correspondence_node = evaluator.add_leaf(
        id=f"paper_{paper_idx}_forum_correspondence",
        desc=f"Paper {paper_idx + 1}: Forum link corresponds to the paper title '{paper_info.title or 'Untitled'}'",
        parent=verification_parent,
        critical=True,
    )

    await evaluator.verify(
        claim=f"This is an official ICLR {ICLR_YEAR} OpenReview forum link for the paper titled '{paper_info.title}'",
        node=forum_correspondence_node,
        sources=paper_info.forum_link,
        additional_instruction="Check if this is a OpenReview forum page and it corresponds to the specified paper title. Look at the paper title displayed on the forum page.",
    )

    # 3. Verify ICLR last-year status
    iclr_status_node = evaluator.add_leaf(
        id=f"paper_{paper_idx}_iclr_status",
        desc=f"Paper {paper_idx + 1}: Paper is an ICLR {ICLR_YEAR} accepted conference paper",
        parent=verification_parent,
        critical=True,
    )

    # Collect valid links for ICLR status verification
    valid_links = []
    if paper_info.pdf_link:
        valid_links.append(paper_info.pdf_link)
    if paper_info.forum_link:
        valid_links.append(paper_info.forum_link)

    await evaluator.verify(
        claim=f"This page shows that the paper '{paper_info.title}' is an ICLR {ICLR_YEAR} accepted conference paper",
        node=iclr_status_node,
        sources=valid_links,
        additional_instruction=f"Check if this paper is an ICLR {ICLR_YEAR} accepted conference paper. Look for explicit evidence such as 'ICLR {ICLR_YEAR}' in headers/footers (but it shouldn't be under review) if this is the pdf page, conference information, or decision status showing 'Accept' on the OpenReview forum page.",
    )

    # 4. Verify Related Work requirement
    related_work_node = evaluator.add_leaf(
        id=f"paper_{paper_idx}_related_work",
        desc=f"Paper {paper_idx + 1}: 'Related Work' section does not appear in the main body",
        parent=verification_parent,
        critical=True,
    )

    await evaluator.verify(
        claim=f"This paper does not have a 'Related Work' section in the main body (for example, it is in the appendix instead, or the paper does not have a 'Related Work' section at all in the page.)",
        node=related_work_node,
        sources=paper_info.pdf_link,
        additional_instruction="Check if the main body of the paper (all sections before references and appendix) does NOT contain a 'Related Work' section or similar sections like 'Background and Related Work', 'Prior Work', 'Literature Review', etc. The main body is defined as all sections appearing before the references and appendix. If the 'Related Work' section is present in the main body, it should be marked as failed.",
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                   #
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
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()
    
    # Initialize evaluator
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
        default_model=model if model else JUDGE_MODEL
    )

    # -------- 2. Extract structured info from the answer ---------------- #
    extracted_papers = await evaluator.extract(
        prompt=prompt_extract_paper_links(),
        template_class=ExtractedPapers,
        extraction_name="extracted_papers"
    )

    # -------- 3. Build verification tree -------------------------------- #
    # We need exactly 3 papers
    required_papers = 3
    provided_papers = extracted_papers.papers[:required_papers]  # Take only first 3

    # Pad with empty papers if needed
    while len(provided_papers) < required_papers:
        provided_papers.append(PaperInfo())

    # Verify each paper (including empty ones)
    for i, paper in enumerate(provided_papers):
        await verify_paper(evaluator, root, paper, i)

    # -------- 4. Return structured result ------------------------------- #
    evaluator.add_custom_info({
        "num_required": required_papers,
        "num_provided": len(extracted_papers.papers),
        "num_evaluated": required_papers,
    }, "evaluation_stats")

    return evaluator.get_summary()
