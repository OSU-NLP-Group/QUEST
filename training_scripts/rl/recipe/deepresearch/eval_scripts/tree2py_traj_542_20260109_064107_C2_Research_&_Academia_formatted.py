import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cs_award_paper_2024"
TASK_DESCRIPTION = """
Identify a paper that received a Best Paper Award or Outstanding Paper Award at a major computer science conference in 2024. The paper must have at least 3 authors, and at least 3 of the authors must be affiliated with different universities or research institutions. Provide the complete list of all author names in the order they appear on the paper, along with each author's institutional affiliation(s) as listed in the paper. Also provide a URL to the paper's official conference proceedings page or other authoritative source where this information can be verified.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AuthorInfo(BaseModel):
    """Model representing a single author and their affiliations."""
    name: Optional[str] = None
    affiliations: List[str] = Field(default_factory=list)


class PaperExtraction(BaseModel):
    """Model representing the selected paper and its key details."""
    paper_title: Optional[str] = None
    conference_name: Optional[str] = None
    conference_year: Optional[str] = None  # keep as string for robustness (answers may say "2024", "2024 (Dec)")
    award_type: Optional[str] = None  # e.g., "Best Paper", "Outstanding Paper", "Distinguished Paper"
    authors_ordered: List[str] = Field(default_factory=list)  # full ordered author list (names only)
    author_infos: List[AuthorInfo] = Field(default_factory=list)  # ordered authors with affiliations
    verification_urls: List[str] = Field(default_factory=list)  # authoritative page(s) or official PDF URLs


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_paper() -> str:
    return """
    Extract exactly one qualifying paper from the answer. If multiple papers are mentioned, choose the first one that meets the constraints.
    Return the following fields:
    1) paper_title: The full paper title.
    2) conference_name: The name of the major computer science conference (e.g., NeurIPS, ICML, CVPR, ACL, SIGGRAPH, WWW, ICLR, KDD, SIGCOMM, OSDI, SOSP, CHI, ICSE, PLDI, POPL, EMNLP, ECCV, AAAI).
    3) conference_year: The year of the conference meeting (must be 2024 if present in the answer).
    4) award_type: The award name given to the paper, e.g., "Best Paper", "Outstanding Paper", "Distinguished Paper" (top-tier paper award).
    5) authors_ordered: The complete list of author names IN ORDER as shown on the paper/official source.
    6) author_infos: For each author IN ORDER, include:
       - name: The author's full name as listed,
       - affiliations: An array of the institutional affiliation(s) exactly as listed (include multiple affiliations when present).
    7) verification_urls: A list of authoritative URLs (official conference proceedings page, official conference/publisher website, official paper PDF). Extract only URLs explicitly present in the answer; include full URLs. If none are present, return an empty list.

    RULES:
    - Do NOT invent any fields. If some fields are missing in the answer, set them to null or empty arrays as appropriate.
    - For URLs, extract only valid, complete URLs present in the answer text (plain or markdown links). If a URL is missing a protocol, prepend http://.
    - Preserve author order exactly as stated in the answer and ensure author_infos follows the same order.
    - If the answer lists more than one affiliation per author, include all.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_affiliation(aff: str) -> str:
    """Light normalization for affiliation strings to aid distinct counting."""
    import re
    s = aff.lower()
    s = re.sub(r"[\,;]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _count_distinct_affiliations(author_infos: List[AuthorInfo]) -> int:
    """Count distinct institutions using the first listed affiliation per author."""
    seen = set()
    for info in author_infos:
        if info and info.affiliations:
            first = info.affiliations[0].strip()
            if first:
                seen.add(_normalize_affiliation(first))
    return len(seen)


def _authors_from_infos(author_infos: List[AuthorInfo]) -> List[str]:
    """Derive ordered author names from author_infos."""
    return [a.name.strip() for a in author_infos if a.name and a.name.strip()]


def _format_authors_claim(authors: List[str]) -> str:
    """Build a claim string listing authors in order."""
    return "Ordered author list: " + " | ".join(authors)


def _format_affiliations_claim(author_infos: List[AuthorInfo]) -> str:
    """Build a claim string mapping authors to affiliations exactly."""
    parts = []
    for info in author_infos:
        name = info.name or ""
        affs = "; ".join([a.strip() for a in info.affiliations if a and a.strip()])
        parts.append(f"{name}: {affs}")
    return "Author affiliations mapping -> " + " || ".join(parts)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_paper_identification(
    evaluator: Evaluator,
    root_node,
    data: PaperExtraction,
) -> None:
    """
    Build the Paper_Identification sub-tree:
      - Award_And_Conference_2024 (verify by URL)
      - Minimum_Author_Count (custom check)
      - Minimum_Distinct_Affiliations (custom check)
    All children must be critical since root is critical.
    """
    pid_node = evaluator.add_parallel(
        id="Paper_Identification",
        desc="Select a paper meeting the award/conference and author/affiliation constraints",
        parent=root_node,
        critical=True
    )

    # Award & Conference 2024 verification (by authoritative URLs)
    award_node = evaluator.add_leaf(
        id="Award_And_Conference_2024",
        desc="The paper received a Best Paper / Outstanding Paper (or equivalent top-tier) award at a major peer-reviewed computer science conference whose meeting occurred in 2024",
        parent=pid_node,
        critical=True
    )

    title = data.paper_title or "Unknown title"
    conf = data.conference_name or "Unknown conference"
    year = data.conference_year or "Unknown year"
    award = data.award_type or "Unknown award"

    award_claim = (
        f"According to the provided source(s), the paper titled '{title}' received a top-tier paper award "
        f"('{award}' or equivalent) at '{conf}' and the conference meeting is in 2024."
    )
    await evaluator.verify(
        claim=award_claim,
        node=award_node,
        sources=data.verification_urls,
        additional_instruction=(
            "Verify the page(s) explicitly state a top-tier paper award: Best Paper, Outstanding Paper, "
            "Distinguished Paper, or equivalent at the main conference (not a workshop). "
            "Confirm the conference year is 2024. Consider typical major CS conferences (e.g., NeurIPS, ICML, CVPR, ICCV, ACL, EMNLP, ICLR, AAAI, KDD, SIGCOMM, OSDI, SOSP, CHI, SIGGRAPH, WWW, ICSE, PLDI, POPL, ECCV). "
            "If the source is irrelevant/invalid or year ≠ 2024, mark as not supported."
        )
    )

    # Minimum author count >= 3
    authors_count = len(data.authors_ordered) if data.authors_ordered else len(_authors_from_infos(data.author_infos))
    evaluator.add_custom_node(
        result=(authors_count >= 3),
        id="Minimum_Author_Count",
        desc="The paper lists at least 3 authors",
        parent=pid_node,
        critical=True
    )

    # Minimum distinct affiliations among authors >= 3
    distinct_aff_count = _count_distinct_affiliations(data.author_infos)
    evaluator.add_custom_node(
        result=(distinct_aff_count >= 3 and authors_count >= 3),
        id="Minimum_Distinct_Affiliations",
        desc="At least 3 authors are affiliated with different universities or research institutions as indicated by the paper’s official author affiliations",
        parent=pid_node,
        critical=True
    )


async def build_author_info_extraction(
    evaluator: Evaluator,
    root_node,
    data: PaperExtraction,
) -> None:
    """
    Build the Author_Information_Extraction sub-tree:
      - Author_Names_In_Order (verify by URL)
      - Affiliations_Per_Author (verify by URL)
      - Verification_URL (verify that provided URLs are authoritative and include author list & affiliations)
    All children must be critical since root is critical.
    """
    aie_node = evaluator.add_parallel(
        id="Author_Information_Extraction",
        desc="Report the full author list and affiliations exactly as shown in an authoritative source",
        parent=root_node,
        critical=True
    )

    # Author names in exact order
    authors_claim_list = data.authors_ordered if data.authors_ordered else _authors_from_infos(data.author_infos)
    names_leaf = evaluator.add_leaf(
        id="Author_Names_In_Order",
        desc="Provide the complete list of all author names in the order they appear on the paper",
        parent=aie_node,
        critical=True
    )
    await evaluator.verify(
        claim=_format_authors_claim(authors_claim_list),
        node=names_leaf,
        sources=data.verification_urls,
        additional_instruction=(
            "Verify the ordered author names against the provided authoritative source(s). "
            "Order must match exactly as listed on the paper. Allow minor variations in formatting (middle initials, accents), "
            "but the sequence must be correct."
        )
    )

    # Affiliations per author
    aff_leaf = evaluator.add_leaf(
        id="Affiliations_Per_Author",
        desc="For each author, provide the institutional affiliation(s) exactly as listed in the paper",
        parent=aie_node,
        critical=True
    )
    await evaluator.verify(
        claim=_format_affiliations_claim(data.author_infos),
        node=aff_leaf,
        sources=data.verification_urls,
        additional_instruction=(
            "Verify that each author's affiliation(s) match exactly what is listed on the authoritative page/PDF. "
            "Include all affiliations for multi-affiliated authors. If affiliations are not present on the provided source(s), "
            "or the mapping differs, mark as not supported."
        )
    )

    # Verification URL(s) validity and authority
    url_leaf = evaluator.add_leaf(
        id="Verification_URL",
        desc="Provide a valid, publicly accessible URL to an authoritative source (e.g., official conference proceedings page, official conference website, or the official paper PDF) where the author list and affiliations can be verified",
        parent=aie_node,
        critical=True
    )

    # Build a claim summarizing the expected authority and content
    url_claim = (
        "The provided URL(s) is an authoritative source (official conference proceedings page, official conference/publisher website, "
        "or the official paper PDF) and contains the author list and affiliations for this paper."
    )
    await evaluator.verify(
        claim=url_claim,
        node=url_leaf,
        sources=data.verification_urls,
        additional_instruction=(
            "Check that each URL is publicly accessible and authoritative (e.g., acm.org, ieee.org, usenix.org, neurips.cc, icml.cc, cvpr.thecvf.com, aclanthology.org, openreview.net (when it hosts the official proceedings), conference official sites, or official PDFs). "
            "Blog posts, news articles, or third-party summaries are not authoritative. "
            "Confirm the page actually includes the ordered author list and affiliations."
        )
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 2024 major CS conference top-award paper task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # sequential: identification first, then author info verification
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

    # Extract structured paper information
    extracted_paper = await evaluator.extract(
        prompt=prompt_extract_paper(),
        template_class=PaperExtraction,
        extraction_name="paper_extraction",
    )

    # Record constraints as ground truth info (for transparency)
    evaluator.add_ground_truth({
        "constraints": {
            "award": "Best/Outstanding/Distinguished Paper (top-tier) at a major CS conference",
            "conference_year": 2024,
            "min_authors": 3,
            "min_distinct_affiliations": 3,
            "author_order_and_affiliations": "Must match authoritative source",
            "verification_url": "Official conference/publisher site or official paper PDF"
        }
    })

    # Build verification tree
    await build_paper_identification(evaluator, root, extracted_paper)
    await build_author_info_extraction(evaluator, root, extracted_paper)

    # Return structured result
    return evaluator.get_summary()