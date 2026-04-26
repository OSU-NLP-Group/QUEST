import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ml_ai_papers_2024_2025"
TASK_DESCRIPTION = (
    "Find two research papers on machine learning or artificial intelligence that were published or made available "
    "as preprints between January 2024 and November 2025, where at least one author of each paper is affiliated with "
    "a university or research institution in California, USA. For each paper, provide the following information: "
    "(1) Full title of the paper, (2) Complete list of authors with their institutional affiliations, "
    "(3) Publication venue (journal, conference, or preprint server name), (4) Publication or submission date, "
    "(5) Persistent identifier (DOI, arXiv ID, or similar), (6) Direct URL to access the paper or its official metadata page."
)

DATE_RANGE_START = "2024-01-01"
DATE_RANGE_END = "2025-11-30"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AuthorAffiliation(BaseModel):
    name: Optional[str] = None
    affiliation: Optional[str] = None


class PaperItem(BaseModel):
    title: Optional[str] = None
    authors: List[AuthorAffiliation] = Field(default_factory=list)
    venue: Optional[str] = None
    date: Optional[str] = None  # publication/submission date as string
    identifier: Optional[str] = None  # DOI, arXiv ID, or similar
    url: Optional[str] = None  # direct URL to paper or official metadata page


class PapersExtraction(BaseModel):
    papers: List[PaperItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_papers() -> str:
    return """
    Extract up to two (2) distinct ML/AI research papers mentioned in the answer. Each paper must include:
    1) title: Full paper title
    2) authors: An array of author objects; each object must include:
       - name: the author's full name (exactly as in the answer)
       - affiliation: the author's institutional affiliation (exactly as in the answer; city/state/country if provided)
    3) venue: Publication venue or preprint server name (e.g., journal, conference, arXiv)
    4) date: Publication or submission date string exactly as stated in the answer
    5) identifier: Persistent identifier such as a DOI (e.g., "10.1145/xxxx") or an arXiv ID (e.g., "arXiv:2401.xxxx")
    6) url: A direct, publicly accessible URL to the full paper or an official metadata page
    
    Rules:
    - Extract only what is explicitly present in the answer. If any field is missing, return null for that field.
    - If more than two papers are mentioned, include only the first two.
    - If fewer than two are mentioned, include the provided ones; do NOT invent missing papers.
    - Preserve the original formatting of names, affiliations, dates, and identifiers as they appear.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_str(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return s.strip().lower()


def paper_signature(p: PaperItem) -> Optional[str]:
    """
    Build a signature to detect duplication based on identifier or title or URL.
    """
    for field in [p.identifier, p.title, p.url]:
        ns = normalize_str(field)
        if ns:
            return ns
    return None


def ensure_two_papers(extracted: PapersExtraction) -> List[PaperItem]:
    """
    Ensure we have exactly two PaperItem entries by truncating or padding with empty items.
    """
    papers = extracted.papers[:2]
    while len(papers) < 2:
        papers.append(PaperItem())
    return papers


def authors_affiliations_provided(paper: PaperItem) -> bool:
    """
    Check that there is at least one author and each has both name and affiliation provided (non-empty).
    """
    if not paper.authors:
        return False
    for a in paper.authors:
        if not a.name or not a.name.strip():
            return False
        if not a.affiliation or not a.affiliation.strip():
            return False
    return True


def valid_public_url(url: Optional[str]) -> bool:
    """
    Basic URL validity check for existence node.
    """
    if not url or not url.strip():
        return False
    u = url.strip().lower()
    return u.startswith("http://") or u.startswith("https://")


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_paper(
    evaluator: Evaluator,
    parent_node,
    paper: PaperItem,
    index: int,
) -> None:
    """
    Build and verify the subtree for a single paper.
    """
    # Paper aggregator (non-critical to allow partial credit across papers)
    paper_node = evaluator.add_parallel(
        id=f"paper_{index}",
        desc=f"Paper #{index + 1} requirements and required output fields.",
        parent=parent_node,
        critical=False,
    )

    # 1) Required Output Fields (critical)
    req_fields_node = evaluator.add_parallel(
        id=f"paper_{index}_required_fields",
        desc=f"All required bibliographic fields are provided for paper #{index + 1}.",
        parent=paper_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(paper.title is not None and paper.title.strip() != ""),
        id=f"paper_{index}_title_provided",
        desc="Full title is provided.",
        parent=req_fields_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=authors_affiliations_provided(paper),
        id=f"paper_{index}_authors_affiliations_provided",
        desc="Complete author list is provided, and each author’s institutional affiliation is clearly stated.",
        parent=req_fields_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(paper.venue is not None and paper.venue.strip() != ""),
        id=f"paper_{index}_venue_name_provided",
        desc="Publication venue (journal/conference) or preprint server name is provided.",
        parent=req_fields_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(paper.date is not None and paper.date.strip() != ""),
        id=f"paper_{index}_date_provided",
        desc="A publication date or submission date is explicitly provided.",
        parent=req_fields_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(paper.identifier is not None and paper.identifier.strip() != ""),
        id=f"paper_{index}_persistent_identifier_provided",
        desc="A persistent identifier (DOI, arXiv ID, or similar) is provided.",
        parent=req_fields_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=valid_public_url(paper.url),
        id=f"paper_{index}_public_url_provided",
        desc="A direct, publicly accessible URL to the full paper or an official metadata page is provided.",
        parent=req_fields_node,
        critical=True,
    )

    # 2) Eligibility Constraints (critical)
    elig_node = evaluator.add_parallel(
        id=f"paper_{index}_eligibility_constraints",
        desc=f"Paper #{index + 1} meets the stated eligibility constraints (time, topic, CA affiliation, venue type).",
        parent=paper_node,
        critical=True,
    )

    # Date in range
    date_range_leaf = evaluator.add_leaf(
        id=f"paper_{index}_date_in_range",
        desc="The provided publication/submission date falls between January 2024 and November 2025 (inclusive).",
        parent=elig_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The publication/submission date shown on the official page for this paper is between {DATE_RANGE_START} "
            f"and {DATE_RANGE_END} (inclusive). If multiple dates appear, use the main published/submitted date."
        ),
        node=date_range_leaf,
        sources=paper.url,
        additional_instruction=(
            "Verify the date displayed on the official source page. Accept reasonable date formats (e.g., "
            "YYYY-MM-DD, Month YYYY). If the date clearly falls in the window Jan 2024–Nov 2025, mark correct."
        ),
    )

    # Topic is ML or AI
    topic_leaf = evaluator.add_leaf(
        id=f"paper_{index}_topic_is_ml_ai",
        desc="Paper is about machine learning or artificial intelligence research.",
        parent=elig_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "This paper is about machine learning or artificial intelligence research (methods, models, systems, "
            "applications, or theory in ML/AI)."
        ),
        node=topic_leaf,
        sources=paper.url,
        additional_instruction=(
            "Check the title/abstract/keywords or visible text on the official page. Accept if ML/AI is clearly the primary focus."
        ),
    )

    # Has California-affiliated author
    ca_affil_leaf = evaluator.add_leaf(
        id=f"paper_{index}_has_ca_affiliated_author",
        desc="At least one author is affiliated with a university or research institution located in California, USA.",
        parent=elig_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "At least one author of this paper has an affiliation located in California, USA (e.g., Stanford University, "
            "UC Berkeley, UCLA, UC San Diego, Caltech, USC, UC Davis, UC Irvine, UC Santa Barbara, UC Riverside, "
            "UC Santa Cruz, UC Merced, Santa Clara University, San Diego State University, Google Research (Mountain View), "
            "Apple (Cupertino), NVIDIA (Santa Clara), etc.)."
        ),
        node=ca_affil_leaf,
        sources=paper.url,
        additional_instruction=(
            "Look for author affiliations/addresses on the official page. If any affiliation references a California "
            "institution or location within CA, mark correct. Allow reasonable variants in names."
        ),
    )

    # Venue type qualifies
    venue_type_leaf = evaluator.add_leaf(
        id=f"paper_{index}_venue_type_qualifies",
        desc="Venue is either a peer-reviewed journal/conference proceedings OR a recognized preprint server (e.g., arXiv, bioRxiv).",
        parent=elig_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The venue '{paper.venue or ''}' is either a peer-reviewed journal/conference proceedings or a recognized "
            "preprint server (e.g., arXiv, bioRxiv, medRxiv, SSRN, HAL, institutional repository)."
        ),
        node=venue_type_leaf,
        sources=paper.url,
        additional_instruction=(
            "Use the venue label on the official page. If it is a well-known journal/conference or recognized preprint server, accept."
        ),
    )

    # 3) Verifiability (critical)
    verif_node = evaluator.add_parallel(
        id=f"paper_{index}_verifiability",
        desc=f"Provided information for paper #{index + 1} is verifiable via official academic sources.",
        parent=paper_node,
        critical=True,
    )

    # URL resolves to the correct paper / official metadata page
    resolves_leaf = evaluator.add_leaf(
        id=f"paper_{index}_url_resolves_correct_paper",
        desc="The provided URL resolves to the referenced paper or its official metadata page.",
        parent=verif_node,
        critical=True,
    )
    title_for_claim = paper.title or ""
    identifier_for_claim = paper.identifier or ""
    await evaluator.verify(
        claim=(
            f"The provided URL points to the official page/metadata page of the paper titled '{title_for_claim}' "
            f"and/or shows the matching persistent identifier '{identifier_for_claim}'. Allow minor title formatting variants."
        ),
        node=resolves_leaf,
        sources=paper.url,
        additional_instruction=(
            "Verify that the page title or metadata clearly corresponds to the given paper. Minor variations in casing or punctuation are acceptable."
        ),
    )

    # Official source for verification
    official_source_leaf = evaluator.add_leaf(
        id=f"paper_{index}_official_source",
        desc="The verification source is an official academic source (e.g., journal site, preprint server, conference proceedings site, university repository).",
        parent=verif_node,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "This webpage is an official academic source, such as a publisher journal site (e.g., IEEE, ACM, Springer, Elsevier), "
            "a recognized preprint server (arXiv, bioRxiv, medRxiv, SSRN), a conference proceedings site, or an institutional repository."
        ),
        node=official_source_leaf,
        sources=paper.url,
        additional_instruction=(
            "Judge by domain, branding, and page content. Personal blogs, news articles, or random aggregators do not qualify."
        ),
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
    Evaluate an answer for the ML/AI papers (2024–2025) task.

    Returns a structured summary with verification tree and final score.
    """
    # Initialize evaluator (root is non-critical by design)
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

    # Top-level aggregator for the task (keep non-critical to allow partial credit across papers)
    task_main = evaluator.add_parallel(
        id="Machine_Learning_Papers_Research",
        desc="Evaluation of exactly two distinct ML/AI research papers (Jan 2024–Nov 2025) with California-affiliated author(s), including required bibliographic fields and verifiable official sources.",
        parent=root,
        critical=False,
    )

    # Extract papers from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_papers(),
        template_class=PapersExtraction,
        extraction_name="papers_extraction",
    )

    # Ensure exactly two items for evaluation
    papers = ensure_two_papers(extracted)

    # Exactly two distinct papers (critical leaf)
    # Compute distinctness based on signatures
    sigs = [paper_signature(p) for p in papers]
    two_present = all(sigs)  # both signatures exist
    distinct = (sigs[0] != sigs[1]) if two_present else False

    evaluator.add_custom_node(
        result=(two_present and distinct),
        id="exactly_two_distinct_papers",
        desc="Output identifies exactly two (2) distinct research papers (not the same paper repeated).",
        parent=task_main,
        critical=True,
    )

    # Paper #1 subtree (non-critical at this layer for partial credit)
    await verify_paper(evaluator, task_main, papers[0], 0)

    # Paper #2 subtree (non-critical at this layer for partial credit)
    await verify_paper(evaluator, task_main, papers[1], 1)

    # Return structured summary
    return evaluator.get_summary()