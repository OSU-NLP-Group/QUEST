import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "lit_fiction_bigfive_awards_2023_2025"
TASK_DESCRIPTION = """
Identify 4 distinct literary fiction books that meet ALL of the following criteria: 
(1) Published in the United States between January 1, 2023, and December 31, 2025; 
(2) Published by one of the Big Five publishers (Penguin Random House, HarperCollins, Hachette Book Group, Macmillan Publishers, or Simon & Schuster) or one of their imprints; 
(3) The print edition has between 280 and 400 pages; 
(4) The book won or was a finalist for either the Pulitzer Prize for Fiction or the National Book Award for Fiction in 2024 or 2025; 
(5) The author is a US citizen or maintains their primary, long-term residence in the United States. 
For each book, provide: the title, author, publisher/imprint, page count, publication year, and award status (winner or finalist, which award, which year).
"""

BIG_FIVE_LIST = [
    "Penguin Random House",
    "HarperCollins",
    "Hachette Book Group",
    "Macmillan Publishers",
    "Simon & Schuster",
]
VALID_AWARDS = [
    "Pulitzer Prize for Fiction",
    "National Book Award for Fiction",
]
VALID_AWARD_YEARS = {"2024", "2025"}
VALID_AWARD_STATUSES = {"winner", "finalist"}

# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
class BookAward(BaseModel):
    award_name: Optional[str] = None  # e.g., "Pulitzer Prize for Fiction" or "National Book Award for Fiction"
    award_year: Optional[str] = None  # e.g., "2024" or "2025" as a string
    award_status: Optional[str] = None  # "winner" or "finalist"


class BookInfo(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher_or_imprint: Optional[str] = None
    page_count: Optional[str] = None  # Keep string to handle formats like "320 pages"
    publication_year: Optional[str] = None  # Keep as string for robustness
    genre: Optional[str] = None  # Expected to contain "literary fiction" or similar
    author_us_status: Optional[str] = None  # Free-text justification, e.g., "U.S. citizen" or "lives in New York"
    award: Optional[BookAward] = None
    source_urls: List[str] = Field(default_factory=list)


class BooksExtraction(BaseModel):
    books: List[BookInfo] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_books() -> str:
    return """
    Extract up to four (4) books exactly as presented in the answer. Return them in order of appearance and DO NOT include more than 4.
    For each book, extract the following fields:
    - title: The book title as given in the answer text.
    - author: The primary author name as given.
    - publisher_or_imprint: The publisher or imprint as given (e.g., "Alfred A. Knopf", "Scribner", "Riverhead Books").
    - page_count: The print page count string if stated (e.g., "320", "320 pages"). If multiple editions are shown, prefer the US hardcover or paperback page count.
    - publication_year: The publication year (four digits) for the US edition if available.
    - genre: Any genre label given (we're looking for "literary fiction" specifically).
    - author_us_status: Any explicit info about the author’s US citizenship or long-term residence (free text copied from the answer).
    - award: An object with:
        * award_name: The award name (exactly as given, e.g., "Pulitzer Prize for Fiction" or "National Book Award for Fiction").
        * award_year: The year (e.g., "2024" or "2025").
        * award_status: One of "winner" or "finalist" (lowercase).
    - source_urls: All URLs cited in the answer that are explicitly associated with this book (publisher page, retailer/Books page, authoritative award pages, author/publisher bios, etc.). 
      Extract only actual URLs present in the answer text. If none are given, return an empty array.

    Rules:
    - Do not invent or infer data not present in the answer. If missing, set the field to null (or an empty array for URLs).
    - Keep "publication_year" and "page_count" as they appear (strings). Do not coerce to integers.
    - Include at most the first four books.
    """


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _clean(s: Optional[str]) -> str:
    return (s or "").strip()


def _first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\d{1,5}", text.replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _year_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\b(20[0-5]\d|19\d{2})\b", text)  # Keep general but we will bound later
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _normalize_title_author_pair(title: Optional[str], author: Optional[str]) -> Optional[Tuple[str, str]]:
    t = _clean(title).lower()
    a = _clean(author).lower()
    if not t or not a:
        return None
    return (t, a)


# -----------------------------------------------------------------------------
# Verification logic for a single book
# -----------------------------------------------------------------------------
async def verify_book(
    evaluator: Evaluator,
    root_parent,
    book: BookInfo,
    idx_one_based: int,
) -> None:
    """
    Build the verification subtree for one book and perform URL-grounded checks.
    According to rubric, each sub-check is critical within the book, while the book itself (as a subtask) is non-critical at the root.
    """

    # Parent node for this book (parallel aggregation; all children evaluated independently)
    book_node = evaluator.add_parallel(
        id=f"book_{idx_one_based}",
        desc=f"The {['first','second','third','fourth'][idx_one_based-1]} book meets all requirements: Big Five publisher, 280-400 pages, published 2023-2025 in US, literary fiction, author is US citizen/resident, won or finalist for Pulitzer/NBA Fiction 2024-2025, and title/author provided.",
        parent=root_parent,
        critical=False,
    )

    title = _clean(book.title)
    author = _clean(book.author)
    publisher = _clean(book.publisher_or_imprint)
    page_count_str = _clean(book.page_count)
    publication_year_str = _clean(book.publication_year)
    genre = _clean(book.genre)
    author_us_status = _clean(book.author_us_status)
    award_name = _clean(book.award.award_name) if book.award else ""
    award_year = _clean(book.award.award_year) if book.award else ""
    award_status = _clean(book.award.award_status).lower() if book.award and book.award.award_status else ""
    sources = book.source_urls or []

    # 1) Title + Author provided (critical existence check)
    evaluator.add_custom_node(
        result=bool(title) and bool(author),
        id=f"book_{idx_one_based}_title_author",
        desc=f"The {['first','second','third','fourth'][idx_one_based-1]}'s book title and author are provided.",
        parent=book_node,
        critical=True,
    )

    # 2) Publisher is Big Five or its imprint (critical)
    pub_leaf = evaluator.add_leaf(
        id=f"book_{idx_one_based}_publisher",
        desc="The book is published by one of the Big Five publishers or one of their imprints.",
        parent=book_node,
        critical=True,
    )
    pub_claim = (
        f"The publisher or imprint for the book '{title}' is '{publisher}', "
        f"and it is either one of the Big Five publishers or an imprint owned by one of them "
        f"(Penguin Random House, HarperCollins, Hachette Book Group, Macmillan Publishers, or Simon & Schuster)."
    )
    pub_instruction = (
        "Verify using the provided URLs whether the named publisher/imprint belongs to one of the Big Five groups. "
        "If an imprint, it should be explicitly stated (e.g., 'an imprint of Penguin Random House'). "
        "Accept if the evidence shows the publisher itself is one of the Big Five or the imprint is owned by one of them."
    )
    await evaluator.verify(claim=pub_claim, node=pub_leaf, sources=sources, additional_instruction=pub_instruction)

    # 3) Page count 280-400 inclusive for print (critical)
    pages_int = _first_int(page_count_str)
    page_leaf = evaluator.add_leaf(
        id=f"book_{idx_one_based}_pagecount",
        desc="The book's print edition has between 280 and 400 pages (inclusive).",
        parent=book_node,
        critical=True,
    )
    if pages_int is not None:
        page_claim = (
            f"The primary US print edition (hardcover or paperback) of '{title}' has {pages_int} pages, "
            f"which is between 280 and 400 inclusive."
        )
    else:
        page_claim = (
            f"The primary US print edition (hardcover or paperback) of '{title}' has a page count between 280 and 400 inclusive."
        )
    page_instruction = (
        "Use the provided URLs to confirm print (hardcover or paperback) page count. "
        "Ignore ebook or audiobook lengths. If multiple editions are listed, prefer the US print edition."
    )
    await evaluator.verify(claim=page_claim, node=page_leaf, sources=sources, additional_instruction=page_instruction)

    # 4) Published in the United States between 2023 and 2025 inclusive (critical)
    year_int = _year_int(publication_year_str)
    pubyear_leaf = evaluator.add_leaf(
        id=f"book_{idx_one_based}_publication_year",
        desc="The book was published in the United States between January 1, 2023 and December 31, 2025 (inclusive).",
        parent=book_node,
        critical=True,
    )
    if year_int is not None:
        year_claim = (
            f"'{title}' was published in the United States in {year_int}, "
            f"which falls between 2023 and 2025 inclusive."
        )
    else:
        year_claim = (
            f"'{title}' was published in the United States in the period from 2023 through 2025 inclusive."
        )
    year_instruction = (
        "Confirm the US publication year/version from the provided URLs. "
        "If multiple regions/editions are shown, prefer the US edition/publication info. "
        "Accept only if the US publication year is 2023, 2024, or 2025."
    )
    await evaluator.verify(claim=year_claim, node=pubyear_leaf, sources=sources, additional_instruction=year_instruction)

    # 5) Genre is literary fiction (critical)
    genre_leaf = evaluator.add_leaf(
        id=f"book_{idx_one_based}_genre",
        desc="The book is classified as literary fiction.",
        parent=book_node,
        critical=True,
    )
    genre_claim = f"The book '{title}' is a literary fiction novel."
    genre_instruction = (
        "Use the provided URLs (publisher, retailer, or credible review sources) to confirm that the book is "
        "categorized as 'literary fiction' or is described in a way consistent with literary fiction. "
        "Do not accept only 'thriller', 'mystery', 'science fiction', or generic 'fiction' unless it is explicitly literary."
    )
    await evaluator.verify(claim=genre_claim, node=genre_leaf, sources=sources, additional_instruction=genre_instruction)

    # 6) Author eligibility: US citizen or primary long-term US residence (critical)
    auth_leaf = evaluator.add_leaf(
        id=f"book_{idx_one_based}_author_eligibility",
        desc="The author is a US citizen or maintains their primary, long-term residence in the United States.",
        parent=book_node,
        critical=True,
    )
    auth_claim = (
        f"The author {author} is a U.S. citizen or maintains their primary, long-term residence in the United States."
    )
    auth_instruction = (
        "Rely on credible sources (official bios, interviews, publishers, major news) among the provided URLs. "
        "If dual citizenship includes the U.S., accept. If residency is clearly long-term in the U.S., accept. "
        "Do not accept temporary/short-term residence or unverified claims."
    )
    await evaluator.verify(claim=auth_claim, node=auth_leaf, sources=sources, additional_instruction=auth_instruction)

    # 7) Award: winner or finalist for Pulitzer Prize for Fiction or National Book Award for Fiction (2024 or 2025) (critical)
    award_leaf = evaluator.add_leaf(
        id=f"book_{idx_one_based}_award",
        desc="The book won or was a finalist for Pulitzer/National Book Award for Fiction in 2024 or 2025.",
        parent=book_node,
        critical=True,
    )

    if award_name and award_status and award_year:
        aw_claim = (
            f"The book '{title}' was a {award_status.lower()} for the {award_name} in {award_year} (Fiction category)."
        )
    else:
        aw_claim = (
            f"The book '{title}' was either a winner or a finalist for the Pulitzer Prize for Fiction or the National Book Award for Fiction in 2024 or 2025."
        )
    aw_instruction = (
        "Verify the award and year strictly. Only accept if the book is a 'winner' or 'finalist' for the "
        "Pulitzer Prize for Fiction or the National Book Award for Fiction and the year is 2024 or 2025. "
        "Prefer official award websites (pulitzer.org, nationalbook.org) or credible major media references."
    )
    await evaluator.verify(claim=aw_claim, node=award_leaf, sources=sources, additional_instruction=aw_instruction)


# -----------------------------------------------------------------------------
# Distinctness check
# -----------------------------------------------------------------------------
def compute_distinctness(books: List[BookInfo]) -> bool:
    """
    Critical distinctness: All 4 books must be distinct.
    We enforce:
    - All 4 title+author pairs are provided (non-empty).
    - All 4 title+author pairs are unique (case-insensitive).
    - As an extra safeguard, titles alone should also be unique (case-insensitive).
    """
    if len(books) < 4:
        return False

    pairs: List[Optional[Tuple[str, str]]] = [_normalize_title_author_pair(b.title, b.author) for b in books[:4]]
    if any(p is None for p in pairs):
        return False

    pair_set = set(pairs)  # type: ignore
    if len(pair_set) != 4:
        return False

    titles = [(_clean(b.title)).lower() for b in books[:4]]
    if len(set(titles)) != 4:
        return False

    return True


# -----------------------------------------------------------------------------
# Main evaluation
# -----------------------------------------------------------------------------
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point for evaluating the literary fiction book selection task.
    """
    # Initialize evaluator and root node
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

    # Extract structured book info
    extracted = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction",
    )

    # Keep only first 4; pad if fewer
    books = list(extracted.books[:4])
    while len(books) < 4:
        books.append(BookInfo())

    # Create book verification subtrees
    for i, book in enumerate(books, start=1):
        await verify_book(evaluator, root, book, i)

    # Distinctness (critical leaf at root)
    distinct_ok = compute_distinctness(books)
    evaluator.add_custom_node(
        result=distinct_ok,
        id="distinctness",
        desc="All 4 books are distinct (no duplicate titles or author-title combinations).",
        parent=root,
        critical=True,
    )

    # Optional: record some custom info for transparency
    evaluator.add_custom_info(
        info={
            "big_five_list": BIG_FIVE_LIST,
            "allowed_awards": VALID_AWARDS,
            "allowed_award_years": sorted(list(VALID_AWARD_YEARS)),
            "allowed_award_statuses": sorted(list(VALID_AWARD_STATUSES)),
        },
        info_type="constraints",
        info_name="evaluation_constraints",
    )

    return evaluator.get_summary()