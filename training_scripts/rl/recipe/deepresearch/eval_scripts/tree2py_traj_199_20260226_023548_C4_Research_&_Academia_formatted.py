import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# -----------------------------------------------------------------------------
# Task constants (from rubric JSON)
# -----------------------------------------------------------------------------
TASK_ID = "notebooklm_free_tier_suitability"
TASK_DESCRIPTION = (
    "A graduate student is planning to use NotebookLM's free tier for their Master's thesis literature review project. "
    "The project specifications are as follows: The review will include 45 peer-reviewed research papers; each paper is "
    "approximately 8,000 words in length; all papers are available as PDF files, with each file being under 15MB in size; "
    "the student plans to interact with NotebookLM through approximately 30 chat queries per day to generate summaries and "
    "identify themes; the student will maintain only this one literature review notebook for the semester; one supplementary "
    "source will be a conference presentation in Google Slides format containing 85 slides; the student has no other concurrent "
    "projects requiring separate notebooks. Based on the documented limits and capabilities of NotebookLM's free tier as of "
    "February 2026, does this tool meet all of the student's requirements for completing their literature review project?"
)

# -----------------------------------------------------------------------------
# Project parameters (from task description)
# -----------------------------------------------------------------------------
NUM_PAPERS = 45
WORDS_PER_PAPER = 8000
PDF_FILE_SIZE_MB = 15  # each file is "under 15MB"
NUM_CHAT_QUERIES_PER_DAY = 30
NUM_NOTEBOOKS = 1
GSLIDES_SLIDE_COUNT = 85
TOTAL_SOURCES = NUM_PAPERS + 1  # 45 PDFs + 1 Google Slides deck

# Expected free-tier limits (to be verified against documentation pages)
LIMIT_SOURCES_PER_NOTEBOOK = 50
LIMIT_WORDS_PER_SOURCE = 500_000
LIMIT_CHAT_QUERIES_PER_DAY = 50
LIMIT_NOTEBOOKS_PER_USER = 100
LIMIT_GSLIDES_MAX_SLIDES = 100
LIMIT_FILE_SIZE_MB = 200

# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class DocURLExtraction(BaseModel):
    urls: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompts
# -----------------------------------------------------------------------------
def prompt_extract_doc_urls() -> str:
    return """
    Extract all URLs that the answer cites or references as documentation or official sources about NotebookLM, especially any pages related to plans, limits, capabilities, or supported formats. 
    Return them in a field named 'urls' as a list of strings. 
    Include only actual URLs present in the answer text (plain URLs or markdown links).
    """


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def filter_relevant_urls(urls: List[str]) -> List[str]:
    """Keep unique URLs likely relevant to official NotebookLM documentation."""
    allow_domains = [
        "support.google.com",
        "notebooklm.google.com",
        "blog.google",
        "ai.google",
        "workspace.google.com",
        "help.google.com",
    ]
    seen = set()
    filtered: List[str] = []
    for u in urls:
        lu = u.strip()
        if not lu:
            continue
        if any(d in lu for d in allow_domains):
            if lu not in seen:
                seen.add(lu)
                filtered.append(lu)
    return filtered


async def add_doc_and_fit_check(
    evaluator: Evaluator,
    parent_node,
    base_id: str,
    doc_claim: str,
    fit_claim: str,
    doc_sources: Optional[List[str]],
    doc_instruction: Optional[str] = None,
    fit_instruction: Optional[str] = None,
) -> None:
    """
    For a single requirement, add two critical leaves:
    1) Documentation-backed limit check (URL verification preferred)
    2) Project-requirement fit check (simple logical verification), gated by the doc check.
    """
    # 1) Documentation-backed limit leaf
    doc_leaf = evaluator.add_leaf(
        id=f"{base_id}_limit_doc",
        desc=f"[Doc] {doc_claim}",
        parent=parent_node,
        critical=True,
    )
    await evaluator.verify(
        claim=doc_claim,
        node=doc_leaf,
        sources=doc_sources if doc_sources else None,
        additional_instruction=doc_instruction or "Prefer explicit statements of limits on official NotebookLM or Google Help pages.",
    )

    # 2) Project fit leaf, gated by doc leaf
    fit_leaf = evaluator.add_leaf(
        id=f"{base_id}_within_limit",
        desc=f"[Fit] {fit_claim}",
        parent=parent_node,
        critical=True,
    )
    await evaluator.verify(
        claim=fit_claim,
        node=fit_leaf,
        sources=None,  # logical check using task description context
        additional_instruction=fit_instruction or "Use the task description numbers to check if the requirement is within the documented limit.",
        extra_prerequisites=[doc_leaf],
    )


# -----------------------------------------------------------------------------
# Verification tree construction
# -----------------------------------------------------------------------------
async def build_suitability_tree(
    evaluator: Evaluator,
    doc_urls: List[str],
) -> None:
    """
    Build the rubric tree per JSON and run verifications.
    Root: critical parallel node "NotebookLM_Free_Tier_Suitability"
    Children: critical checks (implemented as sub-nodes with doc + fit leaves; 
              for file format we split PDF and Slides support into separate doc leaves + one fit leaf).
    """
    # Top-level critical parallel node
    top = evaluator.add_parallel(
        id="NotebookLM_Free_Tier_Suitability",
        desc="Evaluates whether NotebookLM free tier meets all specified project requirements",
        parent=evaluator.root,
        critical=True,
    )

    # 1) Source_Count_Requirement
    sc_node = evaluator.add_parallel(
        id="Source_Count_Requirement",
        desc="The total number of research papers in the project does not exceed the free tier limit of 50 sources per notebook",
        parent=top,
        critical=True,
    )
    await add_doc_and_fit_check(
        evaluator=evaluator,
        parent_node=sc_node,
        base_id="source_count",
        doc_claim=f"NotebookLM free tier supports up to {LIMIT_SOURCES_PER_NOTEBOOK} sources per notebook.",
        fit_claim=(
            f"The project uses {TOTAL_SOURCES} sources in a single notebook (45 PDFs + 1 Google Slides), "
            f"which is within the {LIMIT_SOURCES_PER_NOTEBOOK}-sources-per-notebook limit."
        ),
        doc_sources=doc_urls,
        doc_instruction="Pass only if the page clearly states the per-notebook sources limit.",
        fit_instruction="Compute 45+1 and compare to the documented per-notebook sources limit; accept if <=.",
    )

    # 2) Per_Source_Word_Limit
    psw_node = evaluator.add_parallel(
        id="Per_Source_Word_Limit",
        desc="Each individual source document does not exceed the 500,000 word limit per source",
        parent=top,
        critical=True,
    )
    await add_doc_and_fit_check(
        evaluator=evaluator,
        parent_node=psw_node,
        base_id="per_source_words",
        doc_claim=f"NotebookLM free tier allows up to {LIMIT_WORDS_PER_SOURCE:,} words per source.",
        fit_claim=(
            f"Each paper is approximately {WORDS_PER_PAPER:,} words, and the Google Slides deck has {GSLIDES_SLIDE_COUNT} slides, "
            f"so all sources are far below the {LIMIT_WORDS_PER_SOURCE:,}-words-per-source limit."
        ),
        doc_sources=doc_urls,
        doc_instruction="Confirm the documented per-source word limit (explicit numeric limit).",
        fit_instruction="Use commonsense: 8,000 words and an 85-slide deck are below 500,000 words.",
    )

    # 3) File_Format_Compatibility (split into PDF support, Slides support, plus a fit leaf)
    ffc_node = evaluator.add_parallel(
        id="File_Format_Compatibility",
        desc="All source materials are in formats supported by NotebookLM (PDF and Google Slides)",
        parent=top,
        critical=True,
    )

    # 3.a) PDF supported (doc)
    pdf_doc = evaluator.add_leaf(
        id="file_format_pdf_supported_doc",
        desc="[Doc] NotebookLM supports PDF files as source documents.",
        parent=ffc_node,
        critical=True,
    )
    await evaluator.verify(
        claim="NotebookLM supports PDF files as source documents.",
        node=pdf_doc,
        sources=doc_urls if doc_urls else None,
        additional_instruction="Verify on official NotebookLM/Google help pages that PDF is a supported input format.",
    )

    # 3.b) Google Slides supported (doc)
    slides_doc = evaluator.add_leaf(
        id="file_format_slides_supported_doc",
        desc="[Doc] NotebookLM supports Google Slides as source documents.",
        parent=ffc_node,
        critical=True,
    )
    await evaluator.verify(
        claim="NotebookLM supports Google Slides as source documents.",
        node=slides_doc,
        sources=doc_urls if doc_urls else None,
        additional_instruction="Verify on official pages that Google Slides is a supported input format.",
    )

    # 3.c) Fit leaf: Only PDFs and one Slides deck used
    formats_fit = evaluator.add_leaf(
        id="file_format_supported_fit",
        desc="[Fit] All sources are either PDFs (45) or a Google Slides deck (1), which are supported formats.",
        parent=ffc_node,
        critical=True,
    )
    await evaluator.verify(
        claim="All planned sources are PDF files (45) or a Google Slides deck (1), so all are supported formats.",
        node=formats_fit,
        sources=None,
        additional_instruction="Use the task description; pass if only PDF and Google Slides are used.",
        extra_prerequisites=[pdf_doc, slides_doc],
    )

    # 4) Daily_Chat_Query_Requirement
    dcq_node = evaluator.add_parallel(
        id="Daily_Chat_Query_Requirement",
        desc="The daily chat interaction needs do not exceed the free tier limit of 50 chat queries per day",
        parent=top,
        critical=True,
    )
    await add_doc_and_fit_check(
        evaluator=evaluator,
        parent_node=dcq_node,
        base_id="daily_chat",
        doc_claim=f"NotebookLM free tier allows up to {LIMIT_CHAT_QUERIES_PER_DAY} chat queries per day.",
        fit_claim=f"The student plans {NUM_CHAT_QUERIES_PER_DAY} chat queries per day, which is within the {LIMIT_CHAT_QUERIES_PER_DAY}-per-day limit.",
        doc_sources=doc_urls,
        doc_instruction="Confirm the documented daily chat query limit for the free tier.",
        fit_instruction="Compare 30 to the documented daily chat query limit; pass if <=.",
    )

    # 5) Notebook_Count_Requirement
    ncr_node = evaluator.add_parallel(
        id="Notebook_Count_Requirement",
        desc="The number of concurrent projects does not exceed the free tier limit of 100 notebooks per user",
        parent=top,
        critical=True,
    )
    await add_doc_and_fit_check(
        evaluator=evaluator,
        parent_node=ncr_node,
        base_id="notebook_count",
        doc_claim=f"NotebookLM free tier allows up to {LIMIT_NOTEBOOKS_PER_USER} notebooks per user.",
        fit_claim=f"The student will maintain {NUM_NOTEBOOKS} notebook for the semester, which is within the {LIMIT_NOTEBOOKS_PER_USER}-notebook limit.",
        doc_sources=doc_urls,
        doc_instruction="Confirm the documented maximum notebooks per user.",
        fit_instruction="Compare 1 to the documented notebooks-per-user limit; pass if <=.",
    )

    # 6) Google_Slides_Constraint
    gsc_node = evaluator.add_parallel(
        id="Google_Slides_Constraint",
        desc="If any source is a Google Slides presentation, it does not exceed the 100 slides maximum limit",
        parent=top,
        critical=True,
    )
    await add_doc_and_fit_check(
        evaluator=evaluator,
        parent_node=gsc_node,
        base_id="slides_max",
        doc_claim=f"NotebookLM can ingest Google Slides presentations up to {LIMIT_GSLIDES_MAX_SLIDES} slides per deck.",
        fit_claim=f"The Google Slides deck has {GSLIDES_SLIDE_COUNT} slides, which is within the {LIMIT_GSLIDES_MAX_SLIDES}-slide limit.",
        doc_sources=doc_urls,
        doc_instruction="Verify that there is an explicit slides-per-deck limit and its value.",
        fit_instruction="Compare 85 to the documented slides-per-deck limit; pass if <=.",
    )

    # 7) File_Size_Constraint
    fsz_node = evaluator.add_parallel(
        id="File_Size_Constraint",
        desc="All uploaded files do not exceed the 200MB per source file size limit",
        parent=top,
        critical=True,
    )
    await add_doc_and_fit_check(
        evaluator=evaluator,
        parent_node=fsz_node,
        base_id="file_size",
        doc_claim=f"NotebookLM free tier supports source file sizes up to {LIMIT_FILE_SIZE_MB} MB per file.",
        fit_claim=f"Each PDF file is under {PDF_FILE_SIZE_MB} MB, so all sources are within the {LIMIT_FILE_SIZE_MB} MB per-file limit.",
        doc_sources=doc_urls,
        doc_instruction="Confirm the documented per-file size limit for uploaded sources.",
        fit_instruction="Compare per-file size (under 15 MB) to the documented per-file limit; pass if <=.",
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    """
    Build the NotebookLM Free Tier Suitability verification tree and run checks.
    Returns obj_task_eval-standard summary.
    """
    # Initialize evaluator
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

    # 1) Extract any documentation URLs from the agent's answer
    doc_extraction = await evaluator.extract(
        prompt=prompt_extract_doc_urls(),
        template_class=DocURLExtraction,
        extraction_name="doc_urls_from_answer",
    )
    extracted_urls = doc_extraction.urls or []
    filtered_urls = filter_relevant_urls(extracted_urls)

    # 2) Record GT info and custom info
    evaluator.add_ground_truth(
        {
            "project_spec": {
                "num_papers": NUM_PAPERS,
                "words_per_paper": WORDS_PER_PAPER,
                "pdf_file_size_mb": PDF_FILE_SIZE_MB,
                "daily_chat_queries": NUM_CHAT_QUERIES_PER_DAY,
                "num_notebooks": NUM_NOTEBOOKS,
                "gslides_slides": GSLIDES_SLIDE_COUNT,
                "total_sources": TOTAL_SOURCES,
            },
            "expected_limits_to_verify": {
                "sources_per_notebook": LIMIT_SOURCES_PER_NOTEBOOK,
                "words_per_source": LIMIT_WORDS_PER_SOURCE,
                "chat_queries_per_day": LIMIT_CHAT_QUERIES_PER_DAY,
                "notebooks_per_user": LIMIT_NOTEBOOKS_PER_USER,
                "gslides_max_slides": LIMIT_GSLIDES_MAX_SLIDES,
                "file_size_mb": LIMIT_FILE_SIZE_MB,
            },
        },
        gt_type="expected_requirements_and_limits",
    )
    evaluator.add_custom_info(
        info={
            "extracted_doc_urls_count": len(extracted_urls),
            "filtered_doc_urls_count": len(filtered_urls),
            "filtered_doc_urls": filtered_urls,
            "note": "If no valid documentation URLs were provided in the answer, some documentation-backed checks may fall back to simple verification (less preferred).",
        },
        info_type="doc_source_statistics",
    )

    # 3) Build verification tree and run checks
    await build_suitability_tree(evaluator, filtered_urls)

    # 4) Return summary
    return evaluator.get_summary()