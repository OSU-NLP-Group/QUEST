import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "search_summarize_arxiv"
TASK_DESCRIPTION = """
Find 10 papers related to web agents published on arXiv between January and September of last year (based on the submission time of the first version). For each paper, provide the paper name, a link to the pdf, the name and the affiliation of the first author and submission time of the first version on arXiv.
"""

# Time period constraints (papers submitted between January and September of last year)
CURRENT_DATE = datetime.now()
LAST_YEAR = CURRENT_DATE.year - 1
START_DATE = datetime(LAST_YEAR, 1, 1)
END_DATE = datetime(LAST_YEAR, 9, 30, 23, 59, 59)
START_DATE_STR = START_DATE.strftime("%B %d, %Y")
END_DATE_STR = END_DATE.strftime("%B %d, %Y")

JUDGE_MODEL = "o4-mini"


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class PaperTitle(BaseModel):
    title: Optional[str] = None


class PaperTitles(BaseModel):
    papers: List[PaperTitle] = Field(default_factory=list)


class ArxivPaper(BaseModel):
    title: Optional[str] = None
    pdf_link: Optional[str] = None
    first_author: Optional[str] = None
    affiliation: Optional[str] = None
    submission_date: Optional[str] = None


class PaperLinks(BaseModel):
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_paper_titles() -> str:
    return """
    Extract only the titles of all papers mentioned in the answer. Return a list of paper titles in the order they appear in the answer.
    If there are no papers mentioned, return an empty list.
    """


def prompt_extract_paper_info(paper_title: str) -> str:
    return f"""
    Extract the information for the paper titled "{paper_title}" from the answer. Extract the following information:
    - title: The full title of the paper
    - pdf_link: The URL link to the PDF of the paper
    - first_author: The name of the first author (if there are multiple co-first authors in the answer, only consider the exact first among them)
    - affiliation: The affiliation of the first author  (if there are multiple co-first authors in the answer, only consider the affiliation of exact first among them)
    - submission_date: The submission date of the first version on arXiv

    Return null for any field that is not provided.
    """


def prompt_extract_paper_urls(paper_title: str) -> str:
    return f"""
    Extract all URLs that are associated with the paper titled "{paper_title}" from the answer (e.g., pdf link, arXiv pages, etc.):

    Return an empty list if no URLs are found.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def is_arxiv_url(url: str) -> bool:
    """Check if a URL is likely an arXiv URL."""
    return "arxiv.org" in url.lower()


def is_pdf_url(url: str) -> bool:
    """Check if a URL is likely a PDF link."""
    return url.lower().endswith(".pdf") or "/pdf/" in url.lower()


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_paper(
        evaluator: Evaluator,
        parent_node,
        paper_title: Optional[str],
        paper_index: int,
) -> None:
    """
    Verify a single paper with consistent node structure even for empty papers.
    """
    # Create the paper node with sequential strategy since all criteria must be met
    paper_node = evaluator.add_sequential(
        id=f"paper_{paper_index}",
        desc=f"Paper {paper_index + 1}: '{paper_title or 'Empty paper slot'}' meets all requirements",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )

    # Always extract information (even for empty papers)
    paper_info = await evaluator.extract(
        prompt=prompt_extract_paper_info(paper_title or ""),
        template_class=ArxivPaper,
        extraction_name=f"paper_{paper_index}_info"
    )

    # Extract all URLs associated with this paper
    paper_urls_info = await evaluator.extract(
        prompt=prompt_extract_paper_urls(paper_title or ""),
        template_class=PaperLinks,
        extraction_name=f"paper_{paper_index}_urls"
    )
    urls = paper_urls_info.urls

    # Collect URLs that are likely to be PDF links
    pdf_candidate_urls = [url for url in urls if is_arxiv_url(url) or is_pdf_url(url)]
    if paper_info.pdf_link and (is_arxiv_url(paper_info.pdf_link) or is_pdf_url(paper_info.pdf_link)):
        if paper_info.pdf_link not in pdf_candidate_urls:
            pdf_candidate_urls.append(paper_info.pdf_link)

    # Add comprehensive existence check - this gates all subsequent verifications
    evaluator.add_custom_node(
        result=(
            # Paper title exists
            paper_title is not None and paper_title.strip() != "" and
            # All required fields exist
            paper_info.submission_date is not None and paper_info.submission_date.strip() != "" and
            paper_info.title is not None and paper_info.title.strip() != "" and
            paper_info.first_author is not None and paper_info.first_author.strip() != "" and
            paper_info.affiliation is not None and paper_info.affiliation.strip() != "" and
            # URLs exist for verification
            bool(urls) and
            # At least one PDF candidate URL exists
            bool(pdf_candidate_urls)
        ),
        id=f"paper_{paper_index}_existence",
        desc=f"Paper {paper_index + 1} has all required information for verification",
        parent=paper_node,
        critical=True  # Critical to gate all subsequent verifications
    )

    # Now add all verification nodes directly to paper_node
    # 1. Date range verification
    date_range_node = evaluator.add_leaf(
        id=f"paper_{paper_index}_date_range",
        desc=f"Paper {paper_index + 1} submission date is between {START_DATE_STR} and {END_DATE_STR}",
        parent=paper_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The submission date '{paper_info.submission_date}' for paper '{paper_info.title}' is between {START_DATE_STR} and {END_DATE_STR} (inclusive).",
        node=date_range_node,
        additional_instruction=f"A paper with a submission date like '{START_DATE.strftime('%b %Y')}' or '{END_DATE.strftime('%b %Y')}' would be in range. The date must fall within {START_DATE_STR} to {END_DATE_STR}, inclusive."
    )

    # 2. Date provenance verification
    date_provenance_node = evaluator.add_leaf(
        id=f"paper_{paper_index}_date_provenance",
        desc=f"Paper {paper_index + 1} submission date is verifiable from provided sources",
        parent=paper_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"The first version (v1) of this paper was submitted to arXiv on or in '{paper_info.submission_date}'.",
        node=date_provenance_node,
        sources=pdf_candidate_urls,
        additional_instruction="Look for the submission date of the first version (v1) of the paper. Check if it matches the claimed date. Minor differences in date format are acceptable."
    )

    # 3. PDF link verification
    pdf_node = evaluator.add_leaf(
        id=f"paper_{paper_index}_pdf_link",
        desc=f"Paper {paper_index + 1} has a valid PDF link",
        parent=paper_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"This is a valid PDF page for the paper titled '{paper_info.title}'.",
        node=pdf_node,
        sources=pdf_candidate_urls,
        additional_instruction="Check if the URL is a valid PDF link that allows accessing the paper. If you don't understand how to check it's a pdf, as long as it shows the full paper just like the paper PDF and in a readable format, it can be considered a valid PDF link."
    )

    # 4. Basic information verification
    info_node = evaluator.add_leaf(
        id=f"paper_{paper_index}_basic_info",
        desc=f"Paper {paper_index + 1} has verifiable basic information (title, first author, affiliation)",
        parent=paper_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"This paper has the title '{paper_info.title}', first author '{paper_info.first_author}', and the first author's affiliation is '{paper_info.affiliation}'.",
        node=info_node,
        sources=urls,
        additional_instruction="Check if the paper's title, first author name, and first author's affiliation match the information provided. Minor differences in capitalization or formatting are acceptable."
    )

    # 5. Web agent relevance verification
    relevance_node = evaluator.add_leaf(
        id=f"paper_{paper_index}_web_agent_relevance",
        desc=f"Paper {paper_index + 1} is related to web agents",
        parent=paper_node,
        critical=True,
    )

    # Check if the title obviously indicates relevance to web agents
    title_obviously_relevant = (
        paper_info.title and 
        any(term.lower() in paper_info.title.lower() for term in ["web agent", "web-agent", "webagent"])
    )

    if title_obviously_relevant:
        # Automatically pass without LLM verification (matches original behavior)
        relevance_node.score = 1.0
        relevance_node.status = "passed"
    else:
        # Otherwise verify relevance from URL content
        await evaluator.verify(
            claim=f"This paper is related to web agents (AI systems that interact with web interfaces, browser automation, or automated web navigation and task completion).",
            node=relevance_node,
            sources=urls,
            additional_instruction="Check if the paper discusses topics related to web agents, such as AI systems interacting with web interfaces, browser automation, or automated web navigation and task completion. Papers related to GUI Agents or Computer-Use agents are also okay."
        )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: openai.AsyncAzureOpenAI,
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
    
    # Initialize evaluator with parallel strategy for root
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

    # -------- 2. Extract paper titles from the answer -------------------- #
    paper_titles = await evaluator.extract(
        prompt=prompt_extract_paper_titles(),
        template_class=PaperTitles,
        extraction_name="paper_titles"
    )

    # -------- 3. Build verification tree --------------------------------- #
    # Ensure we have exactly 10 paper slots (add empty slots if needed)
    titles = paper_titles.papers[:10]  # Take at most 10 papers
    while len(titles) < 10:
        titles.append(PaperTitle())  # Add empty slots

    # Verify each paper
    for i, paper_title in enumerate(titles):
        await verify_paper(evaluator, root, paper_title.title, i)

    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()
