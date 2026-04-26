import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "books_2025_entertainment"
TASK_DESCRIPTION = """
Identify four books that were published in the year 2025 (between January 1, 2025 and December 31, 2025) and are authored by individuals with documented careers in the entertainment industry, such as actors, musicians, models, or television personalities. For each, provide the following information: (1) the book's title, (2) the author's name, (3) the exact publication date, (4) the physical format (either paperback or hardcover), and (5) the page count. Ensure that each book represents a different author.
""".strip()


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class BookItem(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[str] = None  # Keep as string for robustness (e.g., "March 3, 2025")
    physical_format: Optional[str] = None   # e.g., "paperback" or "hardcover" as stated in the answer
    page_count: Optional[str] = None        # Keep as string (e.g., "320", "320 pages")
    isbn: Optional[str] = None

    # Evidence links (must be explicitly present in the answer)
    book_urls: List[str] = Field(default_factory=list)            # URLs directly about the book/edition
    author_profile_urls: List[str] = Field(default_factory=list)  # URLs establishing entertainment-industry career


class BooksExtraction(BaseModel):
    books: List[BookItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
    From the answer, extract up to four distinct books that the answer claims meet the task requirements.
    For each book, return the following fields exactly as they appear in the answer:
    - title: the book's title string
    - author: the principal credited author string (only one person; if multiple are listed, choose the first/main author mentioned)
    - publication_date: the exact publication/release date string as written in the answer (e.g., "March 3, 2025")
    - physical_format: the physical format string as written in the answer (e.g., "paperback" or "hardcover"; if the answer uses a synonym like "hardback", keep it as written)
    - page_count: the page count string as written in the answer (e.g., "320" or "320 pages")
    - isbn: the ISBN string as written in the answer
    - book_urls: an array of URLs explicitly cited in the answer that are directly about this book/edition (publisher, retailer, library, or catalog pages)
    - author_profile_urls: an array of URLs explicitly cited in the answer that document the author's entertainment-industry career (e.g., Wikipedia, IMDb, AllMusic, official bio)

    Rules:
    - Only extract information explicitly present in the answer.
    - For URLs, only include those explicitly present in the answer text (including in markdown links). Do not invent or infer URLs.
    - If a field is missing for a book, set it to null (or an empty array for URL lists).
    - Preserve strings exactly as written in the answer (do not normalize casing or wording).

    Return a JSON object with a single field:
    {
      "books": [
        {
          "title": ...,
          "author": ...,
          "publication_date": ...,
          "physical_format": ...,
          "page_count": ...,
          "isbn": ...,
          "book_urls": [...],
          "author_profile_urls": [...]
        },
        ...
      ]
    }

    If the answer provides more than four books, keep only the first four in the answer's order.
    If fewer than four are provided, return only those present.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _digits_only_first_number(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    m = re.search(r"\d{1,5}", s.replace(",", ""))
    return m.group(0) if m else None


def _canonical_author(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    # Keep only the main credited person by splitting on common separators.
    lowered = name.strip().lower()
    # Remove content after " with ", " and ", "&", "," to get principal
    for sep in [" with ", " and ", "&", ",", ";"]:
        if sep in lowered:
            lowered = lowered.split(sep)[0]
    # Remove extra parentheses info
    lowered = re.sub(r"\(.*?\)", "", lowered).strip()
    return lowered if lowered else None


def _is_valid_isbn10(isbn: str) -> bool:
    # Remove hyphens/spaces
    s = re.sub(r"[\s-]", "", isbn)
    if len(s) != 10:
        return False
    total = 0
    for i, ch in enumerate(s):
        if ch == 'X' and i == 9:
            val = 10
        elif ch.isdigit():
            val = int(ch)
        else:
            return False
        weight = 10 - i  # 10..1
        total += weight * val
    return total % 11 == 0


def _is_valid_isbn13(isbn: str) -> bool:
    s = re.sub(r"[\s-]", "", isbn)
    if len(s) != 13 or not s.isdigit():
        return False
    total = 0
    for i, ch in enumerate(s):
        val = int(ch)
        weight = 1 if i % 2 == 0 else 3
        total += weight * val
    return total % 10 == 0


def _is_valid_isbn(isbn: Optional[str]) -> bool:
    if not isbn:
        return False
    s = isbn.strip().upper()
    s_compact = re.sub(r"[\s-]", "", s)
    if len(s_compact) == 10:
        return _is_valid_isbn10(s)
    if len(s_compact) == 13:
        return _is_valid_isbn13(s)
    return False


def _ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth"][n]


# --------------------------------------------------------------------------- #
# Verification for a single book                                              #
# --------------------------------------------------------------------------- #
async def verify_single_book(
    evaluator: Evaluator,
    parent_node,
    book: BookItem,
    index: int,
) -> None:
    """
    Build verification leaves for a single book under the provided parent node.
    All leaves are critical for the book subtask.
    """
    ord_word = _ordinal(index)

    # Title
    title_node = evaluator.add_leaf(
        id=f"Book_{index+1}_Title",
        desc=f"{ord_word} book's title is correctly identified",
        parent=parent_node,
        critical=True,
    )
    title_str = book.title or ""
    await evaluator.verify(
        claim=f"On at least one of the cited book pages, the book title is '{title_str}'.",
        node=title_node,
        sources=book.book_urls,
        additional_instruction=(
            "Verify directly from the provided webpage(s) that the title matches. "
            "Allow minor punctuation, casing, or subtitle variations (e.g., presence/absence of a colon or series info)."
        ),
    )

    # Author + Entertainment career
    author_node = evaluator.add_leaf(
        id=f"Book_{index+1}_Author",
        desc=f"{ord_word} book's author is correctly identified and has documented entertainment industry career",
        parent=parent_node,
        critical=True,
    )
    author_str = book.author or ""
    combined_sources = list(dict.fromkeys((book.book_urls or []) + (book.author_profile_urls or [])))
    await evaluator.verify(
        claim=(
            f"The author of the book is '{author_str}'. In addition, credible sources show that '{author_str}' "
            f"has a documented career in the entertainment industry (e.g., actor/actress, musician/singer/rapper/DJ, "
            f"model, or television personality)."
        ),
        node=author_node,
        sources=combined_sources,
        additional_instruction=(
            "Use the provided book page(s) and/or author profile pages (e.g., Wikipedia, IMDb, AllMusic, official bios, "
            "reputable news coverage) to verify that the named person is indeed the book's author and is an entertainment "
            "industry figure. Accept synonyms like 'singer', 'rapper', 'performer', 'comedian', 'hard rock musician', "
            "'reality TV star', 'TV host'. Minor name variants (stage names, initials) are acceptable if clearly the same person."
        ),
    )

    # Publication date (must be in 2025)
    pub_node = evaluator.add_leaf(
        id=f"Book_{index+1}_Publication_Date",
        desc=f"{ord_word} book was published between January 1, 2025 and December 31, 2025",
        parent=parent_node,
        critical=True,
    )
    pub_date_str = book.publication_date or ""
    await evaluator.verify(
        claim=(
            f"On the cited book page(s), the publication (or release) date is '{pub_date_str}', and this date falls within "
            f"the calendar year 2025 (between January 1, 2025 and December 31, 2025, inclusive)."
        ),
        node=pub_node,
        sources=book.book_urls,
        additional_instruction=(
            "Check fields like 'Publication date', 'Release date', or 'Pub date'. Accept common date formats "
            "(e.g., 'March 3, 2025', '2025-03-03', '3 March 2025'). If the page lists multiple editions, ensure the date "
            "corresponds to the described edition. The date must clearly lie in 2025."
        ),
    )

    # Physical format (paperback or hardcover)
    format_node = evaluator.add_leaf(
        id=f"Book_{index+1}_Format",
        desc=f"{ord_word} book's physical format (paperback or hardcover) is correctly specified",
        parent=parent_node,
        critical=True,
    )
    fmt_str = (book.physical_format or "").strip()
    await evaluator.verify(
        claim=(
            f"On the cited book page(s), the physical format for the described edition is '{fmt_str}', "
            f"and it is a paperback or hardcover edition."
        ),
        node=format_node,
        sources=book.book_urls,
        additional_instruction=(
            "Verify that the format is a physical print format: paperback or hardcover. Treat 'hardback' as 'hardcover'. "
            "If the page shows multiple formats (e.g., hardcover, paperback, ebook), ensure the claimed format is indeed one "
            "of the available physical formats for that edition."
        ),
    )

    # Page count
    page_node = evaluator.add_leaf(
        id=f"Book_{index+1}_Page_Count",
        desc=f"{ord_word} book's page count is provided and verifiable",
        parent=parent_node,
        critical=True,
    )
    pages_num = _digits_only_first_number(book.page_count) or (book.page_count or "")
    # Build a robust claim using numeric pages if available
    if pages_num and pages_num.isdigit():
        pages_claim = f"On the cited book page(s), the book has {pages_num} pages."
    else:
        # Fall back to raw string if no clear numeric substring
        pages_claim = f"On the cited book page(s), the book has {book.page_count or ''} pages."
    await evaluator.verify(
        claim=pages_claim,
        node=page_node,
        sources=book.book_urls,
        additional_instruction=(
            "Check synonyms like 'pages', 'pp', 'page count'. Minor formatting differences are acceptable as long as the "
            "numeric page count matches for the claimed edition."
        ),
    )

    # ISBN validity (logical check)
    isbn_valid = _is_valid_isbn(book.isbn)
    evaluator.add_custom_node(
        result=isbn_valid,
        id=f"Book_{index+1}_ISBN",
        desc=f"{ord_word} book has a valid ISBN number provided",
        parent=parent_node,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the 'books_2025_entertainment' task.
    """
    # Initialize evaluator
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

    # Add a main collection node to mirror the rubric tree
    collection_node = evaluator.add_parallel(
        id="Books_Published_2025_Entertainment",
        desc="Collection of four books published in 2025 by entertainment industry figures, with each book representing a different author",
        parent=root,
        critical=False,
    )

    # Extract structured information
    extracted: BooksExtraction = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction",
    )

    # Keep exactly 4 books (pad with empty placeholders if needed)
    books = list(extracted.books[:4])
    while len(books) < 4:
        books.append(BookItem())

    # Build per-book parallel nodes and verify
    for i in range(4):
        book_node = evaluator.add_parallel(
            id=f"Book_{i+1}",
            desc=f"{_ordinal(i)} book meeting all requirements",
            parent=collection_node,
            critical=False,  # Each book contributes partial credit under the collection
        )
        await verify_single_book(evaluator, book_node, books[i], i)

    # Different authors check (critical for the collection)
    author_names = [b.author for b in books]
    normalized_authors = [_canonical_author(a) for a in author_names]
    all_present = all(a is not None and a.strip() != "" for a in author_names)
    unique_ok = len({a for a in normalized_authors if a}) == 4 if all_present else False

    evaluator.add_custom_node(
        result=unique_ok,
        id="Different_Authors",
        desc="All four books represent different authors (no author appears more than once)",
        parent=collection_node,
        critical=True,
    )

    # Return structured summary
    return evaluator.get_summary()