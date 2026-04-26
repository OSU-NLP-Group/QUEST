import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
# Note: AggregationStrategy is also available from obj_task_eval.verification_tree
# but importing from evaluator for convenience.

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "arxiv_3i_atlas_july2025"
TASK_DESCRIPTION = (
    "I am a researcher planning to use Google NotebookLM to analyze early research papers about the interstellar "
    "comet 3I/ATLAS, which was discovered in July 2025. I need to identify papers from the initial discovery period "
    "that are compatible with NotebookLM's free version constraints.\n\n"
    "Find one research paper about 3I/ATLAS that was published on arXiv in July 2025. For this paper, provide:\n\n"
    "1. The arXiv identifier (in the format arXiv:XXXX.XXXXX)\n"
    "2. The paper's title\n"
    "3. Confirmation that the paper is within NotebookLM's free version word limit of 500,000 words per source\n"
    "4. The direct arXiv URL where the paper can be accessed\n\n"
    "The paper must specifically focus on the interstellar comet 3I/ATLAS and must have been published on arXiv during July 2025."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ArxivPaper(BaseModel):
    """
    Represents a single arXiv paper as extracted from the answer.
    All fields are optional to maximize robustness to varied answer formats.
    """
    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    arxiv_url: Optional[str] = None
    # Optional date string as mentioned in the answer (not required for verification, but useful to record)
    date_mentioned_in_answer: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_arxiv_paper() -> str:
    return """
    From the provided answer, extract details for ONE arXiv paper about the interstellar comet 3I/ATLAS
    that was published in July 2025. If multiple papers are mentioned, select the first one that matches.
    Return these fields:
    - arxiv_id: the arXiv identifier in the format "arXiv:YYMM.NNNNN" (e.g., "arXiv:2507.12345"). If not present, return null.
    - title: the paper's title as written in the answer. If not present, return null.
    - arxiv_url: the direct arXiv URL (prefer the abstract page on arxiv.org; PDF is acceptable if that's all provided). If not present, return null.
    - date_mentioned_in_answer: any date string associated with the arXiv posting/publication mentioned in the answer (e.g., "July 2025", "2025-07-15"). If not present, return null.

    IMPORTANT:
    - Do NOT invent information. Only extract what is explicitly present in the answer.
    - If the answer mentions multiple items, pick the first that clearly refers to an arXiv paper about 3I/ATLAS and indicates July 2025 (even loosely, like "July 2025").
    - If a field is missing in the answer, return null for that field.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _url_list(url: Optional[str]) -> List[str]:
    """
    Helper to pass sources properly to verify(); if url is None or empty, return empty list (=> treated as None).
    """
    if url and url.strip():
        return [url.strip()]
    return []


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_arxiv_paper(
    evaluator: Evaluator,
    root_node,
    paper: ArxivPaper,
) -> None:
    """
    Build the verification tree and run checks based on the rubric:
    Root (sequential)
      ├─ Paper_Identification (parallel, critical)
      │   ├─ Topic_3I_ATLAS (leaf, critical)
      │   ├─ ArXiv_July_2025 (leaf, critical)
      │   ├─ ArXiv_Identifier_Provided (leaf, critical)
      │   └─ Paper_Title_Provided (leaf/custom, critical)
      └─ NotebookLM_Compatibility_Verification (parallel, critical)
          ├─ Word_Count_Verification (leaf, critical)
          └─ Reference_URL_Provided (custom, critical)
    """
    # Child 1: Paper Identification
    paper_ident_node = evaluator.add_parallel(
        id="Paper_Identification",
        desc="Correctly identify a research paper meeting all specified criteria",
        parent=root_node,
        critical=True,
    )

    # Topic: paper focuses on interstellar comet 3I/ATLAS
    topic_node = evaluator.add_leaf(
        id="Topic_3I_ATLAS",
        desc="The paper is about the interstellar comet 3I/ATLAS",
        parent=paper_ident_node,
        critical=True,
    )
    topic_claim = (
        "This arXiv paper specifically focuses on the interstellar comet 3I/ATLAS "
        "(also acceptable: '3I ATLAS', 'C/2025 P1 (ATLAS)', 'third interstellar object 3I', "
        "or equivalent naming)."
    )
    await evaluator.verify(
        claim=topic_claim,
        node=topic_node,
        sources=_url_list(paper.arxiv_url),
        additional_instruction=(
            "Verify that the arXiv page text (title, abstract, or metadata) clearly indicates "
            "the paper is about 3I/ATLAS. Allow reasonable naming variants (e.g., 'C/2025 P1 (ATLAS)', "
            "'3I ATLAS', or 'the third interstellar object 3I')."
        ),
    )

    # Publication timing: on arXiv in July 2025
    july2025_node = evaluator.add_leaf(
        id="ArXiv_July_2025",
        desc="The paper was published on arXiv in July 2025",
        parent=paper_ident_node,
        critical=True,
    )
    july_claim = (
        "This arXiv record shows the paper was posted on arXiv in July 2025 "
        "(typically corresponding to the v1 'Submitted on' date falling within July 2025)."
    )
    await evaluator.verify(
        claim=july_claim,
        node=july2025_node,
        sources=_url_list(paper.arxiv_url),
        additional_instruction=(
            "Use the arXiv page's submission history. Treat the initial v1 'Submitted on' date as the publication-on-arXiv "
            "date. The date must be in July 2025. If the page clearly shows any initial posting in July 2025, consider it correct."
        ),
    )

    # Identifier format: arXiv:XXXX.XXXXX (modern style: arXiv:YYMM.NNNNN)
    id_node = evaluator.add_leaf(
        id="ArXiv_Identifier_Provided",
        desc="The arXiv identifier is provided in the correct format (arXiv:XXXX.XXXXX)",
        parent=paper_ident_node,
        critical=True,
    )
    identifier_str = paper.arxiv_id or ""
    id_claim = (
        f"The provided arXiv identifier '{identifier_str}' is in the correct modern format "
        "like 'arXiv:YYMM.NNNNN' (e.g., 'arXiv:2507.12345'). Optional version suffix (e.g., 'v1') is allowed."
    )
    await evaluator.verify(
        claim=id_claim,
        node=id_node,
        additional_instruction=(
            "Judge the identifier string format itself; do not require web evidence here. "
            "Accept the modern style 'arXiv:YYMM.NNNNN' and optionally 'vX' at the end. "
            "If it's missing or malformed, mark incorrect."
        ),
    )

    # Title provided: check existence (since rubric asks 'provided')
    title_exists = bool(paper.title and paper.title.strip())
    evaluator.add_custom_node(
        result=title_exists,
        id="Paper_Title_Provided",
        desc="The paper's title is provided",
        parent=paper_ident_node,
        critical=True,
    )

    # Child 2: NotebookLM compatibility verification
    notebooklm_node = evaluator.add_parallel(
        id="NotebookLM_Compatibility_Verification",
        desc="Verify and document that the paper meets NotebookLM's constraints",
        parent=root_node,
        critical=True,
    )

    # Word count verification: under 500,000 words
    wc_node = evaluator.add_leaf(
        id="Word_Count_Verification",
        desc="Confirm the paper is within the 500,000-word limit per source",
        parent=notebooklm_node,
        critical=True,
    )
    wc_claim = (
        "The full text of this arXiv paper contains fewer than 500,000 words, which satisfies "
        "NotebookLM's free version limit per source."
    )
    await evaluator.verify(
        claim=wc_claim,
        node=wc_node,
        sources=_url_list(paper.arxiv_url),
        additional_instruction=(
            "Use reasonable evidence available on the arXiv page or PDF (e.g., typical research article lengths, "
            "page counts, or visible indicators). A normal-length research paper (tens of pages, not thousands) "
            "is clearly far below 500,000 words. If the paper appears to be standard length, mark as within limit."
        ),
    )

    # Reference URL provided: direct arXiv URL present
    url_exists = bool(paper.arxiv_url and paper.arxiv_url.strip())
    evaluator.add_custom_node(
        result=url_exists,
        id="Reference_URL_Provided",
        desc="Provide the direct arXiv URL for accessing the paper",
        parent=notebooklm_node,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
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
    Evaluate the agent's answer for the task of finding a July 2025 arXiv paper about 3I/ATLAS and
    checking NotebookLM compatibility.
    """
    # Initialize evaluator with a sequential root to reflect step-wise nature
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

    # Record NotebookLM constraint in summary for transparency
    evaluator.add_custom_info(
        info={"NotebookLM_free_version_word_limit_per_source": 500_000},
        info_type="constraint",
        info_name="notebooklm_constraints",
    )

    # Extract the single target arXiv paper info from the answer
    paper_info = await evaluator.extract(
        prompt=prompt_extract_arxiv_paper(),
        template_class=ArxivPaper,
        extraction_name="extracted_arxiv_paper",
    )

    # Build verification tree and execute checks
    await verify_arxiv_paper(evaluator, root, paper_info)

    # Return summary
    return evaluator.get_summary()