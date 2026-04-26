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
TASK_ID = "cf_first_novel_prize_2024"
TASK_DESCRIPTION = (
    "Who won the 2024 Center for Fiction First Novel Prize? Provide comprehensive information including: "
    "the author's name, the book title, the publisher, the publication date, the page count of the hardcover edition, "
    "the ISBN-13 of the hardcover edition, the monetary prize amount, and verification that this is the author's debut novel."
)

PRIZE_NAME = "2024 Center for Fiction First Novel Prize"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WinnerExtraction(BaseModel):
    # Core required fields
    author_name: Optional[str] = None
    book_title: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[str] = None  # Accept free text: month year, or YYYY-MM-DD, etc.
    hardcover_page_count: Optional[str] = None
    hardcover_isbn13: Optional[str] = None
    prize_amount: Optional[str] = None  # Free text acceptable (e.g., "$15,000", "15000 USD")

    # Source URLs (explicitly mentioned in the answer)
    urls_winner: List[str] = Field(default_factory=list)         # Official announcement/news confirming winner identity
    urls_book_details: List[str] = Field(default_factory=list)   # Publisher, retailer, or authoritative book metadata
    urls_prize: List[str] = Field(default_factory=list)          # Prize page with winner and amount
    urls_debut: List[str] = Field(default_factory=list)          # Confirmation it's a debut novel
    urls_all: List[str] = Field(default_factory=list)            # Any and all other URLs present in the answer


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_winner_info() -> str:
    return """
    Extract from the provided answer the complete information for the winner of the 2024 Center for Fiction First Novel Prize.

    Required fields (return null if not present):
    - author_name: the winning author's full name
    - book_title: the exact winning book title (include subtitle if provided)
    - publisher: the publisher/imprint of the winning book (prefer hardcover publisher/imprint)
    - publication_date: publication date of the hardcover edition (month and year at minimum if available)
    - hardcover_page_count: page count for the hardcover edition
    - hardcover_isbn13: ISBN-13 for the hardcover edition (return as written, with or without hyphens)
    - prize_amount: monetary prize amount (e.g., "$15,000", "15000 USD", etc.)

    Also extract any URLs explicitly present in the answer:
    - urls_winner: URLs that directly announce or confirm the 2024 Center for Fiction First Novel Prize winner
    - urls_book_details: URLs that provide authoritative book metadata (publisher site, bookseller like Amazon, Google Books, distributor, etc.)
    - urls_prize: URLs related to the prize announcement or details for the 2024 Center for Fiction First Novel Prize
    - urls_debut: URLs that explicitly state this is the author's "debut novel" or "first novel"
    - urls_all: All URLs present anywhere in the answer (include every URL you see, deduplicate if needed)

    General rules:
    - Do not invent values. Only extract what is explicitly present in the answer.
    - For URLs, include only valid URLs explicitly present in the answer (plain links or markdown links).
    - If a field is not available, return null. For URL lists with no entries, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s if s is not None else ""

def _merge_sources(*url_lists: Optional[List[str]]) -> Optional[List[str]]:
    """Merge multiple URL lists, preserve order, de-duplicate, return None if empty."""
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if not isinstance(u, str):
                continue
            uu = u.strip()
            if not uu:
                continue
            if uu not in seen:
                seen.add(uu)
                merged.append(uu)
    return merged if merged else None


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_nodes(evaluator: Evaluator, parent_node, extracted: WinnerExtraction) -> None:
    """
    Build the verification tree under the given parent node and run verifications for each rubric item.
    """
    # Prepare claims and corresponding sources + additional instructions
    author = _safe(extracted.author_name)
    title = _safe(extracted.book_title)
    publisher = _safe(extracted.publisher)
    pub_date = _safe(extracted.publication_date)
    pages = _safe(extracted.harcover_page_count if hasattr(extracted, 'harcover_page_count') else extracted.hardcover_page_count)
    isbn13 = _safe(extracted.hardcover_isbn13)
    prize_amount = _safe(extracted.prize_amount)

    # Source pools
    winner_sources = _merge_sources(extracted.urls_winner, extracted.urls_prize, extracted.urls_all)
    book_meta_sources = _merge_sources(extracted.urls_book_details, extracted.urls_all)
    prize_sources = _merge_sources(extracted.urls_prize, extracted.urls_winner, extracted.urls_all)
    debut_sources = _merge_sources(extracted.urls_debut, extracted.urls_prize, extracted.urls_all)

    # Create leaf nodes
    node_author = evaluator.add_leaf(
        id="author_name",
        desc="The correct author name is provided",
        parent=parent_node,
        critical=True
    )
    node_title = evaluator.add_leaf(
        id="book_title",
        desc="The correct book title is provided",
        parent=parent_node,
        critical=True
    )
    node_publisher = evaluator.add_leaf(
        id="publisher",
        desc="The correct publisher name is provided",
        parent=parent_node,
        critical=True
    )
    node_pub_date = evaluator.add_leaf(
        id="publication_date",
        desc="The correct publication date (at least month and year) is provided",
        parent=parent_node,
        critical=True
    )
    node_pages = evaluator.add_leaf(
        id="page_count",
        desc="The correct page count for the hardcover edition is provided",
        parent=parent_node,
        critical=True
    )
    node_isbn = evaluator.add_leaf(
        id="isbn",
        desc="The correct ISBN-13 for the hardcover edition is provided",
        parent=parent_node,
        critical=True
    )
    node_prize_amount = evaluator.add_leaf(
        id="prize_amount",
        desc="The correct monetary prize amount is stated",
        parent=parent_node,
        critical=True
    )
    node_debut = evaluator.add_leaf(
        id="debut_novel_verification",
        desc="Confirmation that this is the author's debut novel (first novel, though not necessarily first book)",
        parent=parent_node,
        critical=True
    )

    # Prepare batch verifications
    claims_and_sources = [
        (
            f"The winner of the {PRIZE_NAME} is {author}.",
            winner_sources,
            node_author,
            "Verify on the provided page(s) that the person named is explicitly the winner of the 2024 Center for Fiction First Novel Prize. "
            "Allow minor name variants (accents, middle initials). Prefer official or reputable announcements."
        ),
        (
            f"The winning book for the {PRIZE_NAME} is titled \"{title}\".",
            winner_sources,
            node_title,
            "Verify that the page(s) explicitly state the winning title. Allow minor punctuation/case variations and subtitle presence/absence. "
            "If a subtitle is present or omitted, consider it a match if the core title matches."
        ),
        (
            f"The publisher of the hardcover edition of \"{title}\" is {publisher}.",
            book_meta_sources,
            node_publisher,
            "Confirm that the publisher/imprint listed corresponds to the hardcover edition of the book. "
            "Allow reasonable imprint variations (e.g., 'Knopf' vs. 'Alfred A. Knopf')."
        ),
        (
            f"The publication date of the hardcover edition of \"{title}\" is {pub_date}.",
            book_meta_sources,
            node_pub_date,
            "Confirm the hardcover publication date. Month and year should match; minor day differences are acceptable. "
            "If multiple regional dates exist, any page showing the extracted date is acceptable."
        ),
        (
            f"The hardcover edition of \"{title}\" has {pages} pages.",
            book_meta_sources,
            node_pages,
            "Confirm that the page count corresponds to the hardcover edition. If format is not explicitly stated but context indicates hardcover, accept it."
        ),
        (
            f"The ISBN-13 of the hardcover edition of \"{title}\" is {isbn13}.",
            book_meta_sources,
            node_isbn,
            "Match the 13-digit ISBN ignoring hyphens and spaces. Accept common hyphenation or spacing variants. "
            "Some sites label it as EAN/ISBN-13; treat as equivalent."
        ),
        (
            f"The prize amount awarded for the {PRIZE_NAME} is {prize_amount}.",
            prize_sources,
            node_prize_amount,
            "Verify the monetary amount for the prize. Accept formatting variations like '$15,000', '15,000 USD', or '15000 U.S. dollars'."
        ),
        (
            f"The book \"{title}\" is the author's debut novel (first novel).",
            debut_sources,
            node_debut,
            "Verify that the page(s) explicitly state 'debut novel' or equivalent (e.g., 'first novel'). "
            "If the author has prior non-fiction or short stories, it can still be a debut novel."
        ),
    ]

    # Run verifications in parallel
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the 2024 Center for Fiction First Novel Prize comprehensive information task.
    """
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_winner_info(),
        template_class=WinnerExtraction,
        extraction_name="winner_info"
    )

    # Build critical parallel node per rubric and verify each child leaf
    main_node = evaluator.add_parallel(
        id="2024_first_novel_prize_winner_information",
        desc="Comprehensive information about the 2024 Center for Fiction First Novel Prize winner",
        parent=root,
        critical=True
    )

    await build_and_verify_nodes(evaluator, main_node, extracted)

    return evaluator.get_summary()