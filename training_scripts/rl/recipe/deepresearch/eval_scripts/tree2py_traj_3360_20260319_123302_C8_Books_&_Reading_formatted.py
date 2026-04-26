import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "award_fiction_2020_2024"
TASK_DESCRIPTION = """
Identify three distinct fiction books that meet all of the following criteria:

1. Each book was originally published between January 2020 and December 2024 (inclusive)
2. Each book won at least one of these major literary awards: Pulitzer Prize for Fiction, Booker Prize, or National Book Award for Fiction (the award could have been won in 2020-2025)
3. Each book was published by one of the "Big Five" publishers (Penguin Random House, HarperCollins, Simon & Schuster, Hachette Book Group, or Macmillan) or one of their imprints
4. Each book was released in both hardcover and paperback editions, with the paperback edition released after the hardcover edition

For each book, provide:
- The complete title
- The author's name
- The specific award won and the year it was won
- The publisher (including imprint if applicable)
- The original publication year
- The hardcover publication date
- The paperback publication date
- The hardcover edition page count (if available)
- Reference URLs confirming the award win, publisher, publication dates, and editions
"""

ALLOWED_AWARDS = [
    "Pulitzer Prize for Fiction",
    "Booker Prize",
    "National Book Award for Fiction",
]

BIG_FIVE = [
    "Penguin Random House",
    "HarperCollins",
    "Simon & Schuster",
    "Hachette Book Group",
    "Macmillan",
]

# Helpful common imprints (non-exhaustive; judge should primarily rely on sources)
KNOWN_BIG_FIVE_IMPRINTS = [
    # PRH
    "Knopf", "Alfred A. Knopf", "Doubleday", "Riverhead Books", "Viking", "Penguin Press", "Random House", "Ballantine",
    # HarperCollins
    "Harper", "Ecco", "William Morrow",
    # Simon & Schuster
    "Scribner", "Atria", "Gallery Books",
    # Hachette
    "Little, Brown and Company", "Little, Brown", "Grand Central Publishing", "Mulholland Books",
    # Macmillan
    "Farrar, Straus and Giroux", "FSG", "St. Martin's Press", "Henry Holt", "Tor", "Flatiron Books", "Picador"
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BookItem(BaseModel):
    # Basic identification
    title: Optional[str] = None
    author: Optional[str] = None
    genre: Optional[str] = None  # e.g., "Fiction", "Novel", etc.

    # Award details
    award_name: Optional[str] = None
    award_year: Optional[str] = None
    award_urls: List[str] = Field(default_factory=list)

    # Publisher details
    publisher: Optional[str] = None  # Parent publisher or imprint
    imprint: Optional[str] = None    # Optional imprint (if separate)
    publisher_urls: List[str] = Field(default_factory=list)

    # Publication dates
    original_publication_year: Optional[str] = None  # e.g., "2021"
    original_publication_date: Optional[str] = None  # if present, keep as string
    publication_urls: List[str] = Field(default_factory=list)

    # Editions
    hardcover_pub_date: Optional[str] = None
    paperback_pub_date: Optional[str] = None
    editions_urls: List[str] = Field(default_factory=list)

    # Page count (hardcover)
    hardcover_pages: Optional[str] = None
    pagecount_urls: List[str] = Field(default_factory=list)

    # Other refs
    other_urls: List[str] = Field(default_factory=list)


class BooksExtraction(BaseModel):
    books: List[BookItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
    Extract up to the first three distinct fiction books described in the answer that claim to meet the task requirements.
    For each book, extract the following fields exactly as mentioned in the answer (do not invent):

    Basic identification:
    - title: full title string
    - author: author name(s)
    - genre: the genre label mentioned (e.g., "Fiction", "Novel", "Literary fiction"); if not stated, set to null

    Award:
    - award_name: the major award the book won (if provided)
    - award_year: the year that award was won (if provided)
    - award_urls: array of explicit URLs cited that confirm the award win

    Publisher:
    - publisher: the publisher name or the imprint name that published the book (as stated)
    - imprint: if an imprint is stated separately from the publisher, put it here; otherwise null
    - publisher_urls: array of explicit URLs that confirm the publisher/imprint for this book

    Publication timing:
    - original_publication_year: the original publication year of the book (string)
    - original_publication_date: if a specific original publication date is provided, include it; otherwise null
    - publication_urls: array of explicit URLs that confirm the original publication year/date

    Editions:
    - hardcover_pub_date: hardcover publication date string as presented
    - paperback_pub_date: paperback publication date string as presented
    - editions_urls: array of explicit URLs that confirm the existence of both hardcover and paperback editions and provide their dates (publisher, bookseller, or official pages)

    Page count:
    - hardcover_pages: the hardcover edition page count string if provided, else null
    - pagecount_urls: array of explicit URLs that confirm the page count

    Other:
    - other_urls: array of any other URLs provided for this specific book (if any)

    Return a JSON object:
    {
      "books": [
        {
          "title": ...,
          "author": ...,
          "genre": ...,
          "award_name": ...,
          "award_year": ...,
          "award_urls": [...],
          "publisher": ...,
          "imprint": ...,
          "publisher_urls": [...],
          "original_publication_year": ...,
          "original_publication_date": ...,
          "publication_urls": [...],
          "hardcover_pub_date": ...,
          "paperback_pub_date": ...,
          "editions_urls": [...],
          "hardcover_pages": ...,
          "pagecount_urls": [...],
          "other_urls": [...]
        }
      ]
    }

    Rules:
    - Extract only URLs explicitly present in the answer; keep them as full URLs.
    - If a requested field is missing, set it to null (or empty list for arrays).
    - Maintain strings for dates and years; do not normalize formats.
    - Include at most the first three books mentioned in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _dedup(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not _nonempty(u):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _collect_urls(book: BookItem, fields: List[str]) -> List[str]:
    acc: List[str] = []
    for f in fields:
        v = getattr(book, f, [])
        if isinstance(v, list):
            acc.extend(v)
        elif isinstance(v, str) and _nonempty(v):
            acc.append(v)
    return _dedup(acc)


def _all_relevant_urls(book: BookItem) -> List[str]:
    return _dedup(
        _collect_urls(book, [
            "award_urls",
            "publisher_urls",
            "publication_urls",
            "editions_urls",
            "pagecount_urls",
            "other_urls",
        ])
    )


# --------------------------------------------------------------------------- #
# Verification for a single book                                              #
# --------------------------------------------------------------------------- #
async def verify_one_book(
    evaluator: Evaluator,
    parent_node,
    book: BookItem,
    index_1based: int,
) -> None:
    book_id = f"book_{index_1based}"
    book_desc = f"{['First', 'Second', 'Third'][index_1based - 1]} qualifying book with all required attributes"

    # Book group node (parallel; allow partial across books as per rubric)
    book_node = evaluator.add_parallel(
        id=book_id,
        desc=book_desc,
        parent=parent_node,
        critical=False
    )

    title = book.title or ""
    author = book.author or ""
    imprint = book.imprint or ""
    publisher = book.publisher or ""

    # 1) Basic info
    basic_node = evaluator.add_parallel(
        id=f"{book_id}_basic_info",
        desc=f"Basic identification information for the {['first', 'second', 'third'][index_1based - 1]} book",
        parent=book_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(book.title),
        id=f"{book_id}_title",
        desc=f"The complete title of the {['first', 'second', 'third'][index_1based - 1]} book is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(book.author),
        id=f"{book_id}_author",
        desc=f"The author's name of the {['first', 'second', 'third'][index_1based - 1]} book is provided",
        parent=basic_node,
        critical=True
    )

    genre_leaf = evaluator.add_leaf(
        id=f"{book_id}_genre",
        desc="The book is identified as fiction",
        parent=basic_node,
        critical=True
    )
    genre_claim = f"The book '{title}' by {author} is a work of fiction (e.g., a novel or fiction collection)."
    await evaluator.verify(
        claim=genre_claim,
        node=genre_leaf,
        sources=_all_relevant_urls(book),
        additional_instruction="Use the provided sources to confirm the work is fiction. Accept synonyms like 'novel', 'literary fiction', 'short story collection', etc."
    )

    # 2) Award information
    award_node = evaluator.add_parallel(
        id=f"{book_id}_award",
        desc=f"Award information for the {['first', 'second', 'third'][index_1based - 1]} book",
        parent=book_node,
        critical=True
    )

    award_name = book.award_name or ""
    award_year = book.award_year or ""

    award_name_leaf = evaluator.add_leaf(
        id=f"{book_id}_award_name",
        desc="The book won at least one of: Pulitzer Prize for Fiction, Booker Prize, or National Book Award for Fiction",
        parent=award_node,
        critical=True
    )
    award_name_claim = (
        f"According to the provided source(s), the book '{title}' by {author} won the '{award_name}'. "
        f"This award must be one of: {', '.join(ALLOWED_AWARDS)}."
    )
    await evaluator.verify(
        claim=award_name_claim,
        node=award_name_leaf,
        sources=book.award_urls,
        additional_instruction=(
            "Confirm both that the book won the stated award and that this award is one of the allowed set. "
            "Treat reasonable name variants as equivalent, e.g.: "
            "- 'The Booker Prize', 'Booker Prize for Fiction' => 'Booker Prize'\n"
            "- 'Pulitzer Prize (Fiction)' => 'Pulitzer Prize for Fiction'\n"
            "- 'National Book Award: Fiction' => 'National Book Award for Fiction'\n"
            "Reject if the sources do not clearly confirm a win for one of the allowed awards."
        )
    )

    award_year_leaf = evaluator.add_leaf(
        id=f"{book_id}_award_year",
        desc="The year the award was won is between 2020-2025 (inclusive)",
        parent=award_node,
        critical=True
    )
    award_year_claim = (
        f"The book '{title}' by {author} won the '{award_name}' in {award_year}, "
        "and that year is between 2020 and 2025 inclusive."
    )
    await evaluator.verify(
        claim=award_year_claim,
        node=award_year_leaf,
        sources=book.award_urls,
        additional_instruction="Verify both the award year and that it lies within [2020, 2025]."
    )

    evaluator.add_custom_node(
        result=len(book.award_urls) > 0,
        id=f"{book_id}_award_reference",
        desc="A reference URL confirming the award win is provided",
        parent=award_node,
        critical=True
    )

    # 3) Publisher information
    publisher_node = evaluator.add_parallel(
        id=f"{book_id}_publisher",
        desc=f"Publisher information for the {['first', 'second', 'third'][index_1based - 1]} book",
        parent=book_node,
        critical=True
    )

    publisher_name_leaf = evaluator.add_leaf(
        id=f"{book_id}_publisher_name",
        desc="The book was published by one of the Big Five publishers or their imprints",
        parent=publisher_node,
        critical=True
    )

    publisher_label = imprint if _nonempty(imprint) else publisher
    publisher_claim = (
        f"The book '{title}' by {author} was published by '{publisher_label}'. "
        "This publisher (or its imprint) is part of one of the 'Big Five' publishers: "
        f"{', '.join(BIG_FIVE)}."
    )
    await evaluator.verify(
        claim=publisher_claim,
        node=publisher_name_leaf,
        sources=book.publisher_urls,
        additional_instruction=(
            "Use the provided sources to confirm the book's publisher or imprint, and that it belongs to a Big Five group "
            "(Penguin Random House, HarperCollins, Simon & Schuster, Hachette Book Group, or Macmillan). "
            "It's acceptable if the source shows an imprint that is part of a Big Five publisher. "
            f"Examples of well-known imprints include: {', '.join(KNOWN_BIG_FIVE_IMPRINTS)}. "
            "Prefer explicit statements on the page; do not rely on memory."
        )
    )

    evaluator.add_custom_node(
        result=len(book.publisher_urls) > 0,
        id=f"{book_id}_publisher_reference",
        desc="A reference URL confirming the publisher is provided",
        parent=publisher_node,
        critical=True
    )

    # 4) Publication timing
    pub_node = evaluator.add_parallel(
        id=f"{book_id}_publication_date",
        desc=f"Publication timing information for the {['first', 'second', 'third'][index_1based - 1]} book",
        parent=book_node,
        critical=True
    )

    orig_year = book.original_publication_year or ""

    orig_year_leaf = evaluator.add_leaf(
        id=f"{book_id}_original_publication_year",
        desc="The book's original publication year is between 2020-2024 (inclusive)",
        parent=pub_node,
        critical=True
    )
    orig_year_claim = (
        f"The original publication year of '{title}' by {author} is {orig_year}, "
        "and that year falls between 2020 and 2024 inclusive."
    )
    await evaluator.verify(
        claim=orig_year_claim,
        node=orig_year_leaf,
        sources=_collect_urls(book, ["publication_urls", "publisher_urls", "editions_urls", "award_urls"]),
        additional_instruction="Verify both (a) that the page states the original publication year and (b) that the year lies within [2020, 2024]."
    )

    evaluator.add_custom_node(
        result=(len(book.publication_urls) > 0) or (len(book.publisher_urls) > 0) or (len(book.editions_urls) > 0),
        id=f"{book_id}_publication_reference",
        desc="A reference URL confirming the original publication date is provided",
        parent=pub_node,
        critical=True
    )

    # 5) Editions
    editions_node = evaluator.add_parallel(
        id=f"{book_id}_editions",
        desc=f"Edition format information for the {['first', 'second', 'third'][index_1based - 1]} book",
        parent=book_node,
        critical=True
    )

    # Hardcover edition leaf
    hc_leaf = evaluator.add_leaf(
        id=f"{book_id}_hardcover_edition",
        desc="The book was released in hardcover edition with publication date provided",
        parent=editions_node,
        critical=True
    )
    hc_claim = (
        f"The book '{title}' by {author} has a hardcover edition with publication date {book.hardcover_pub_date}."
    )
    await evaluator.verify(
        claim=hc_claim,
        node=hc_leaf,
        sources=_collect_urls(book, ["editions_urls", "publisher_urls", "publication_urls"]),
        additional_instruction="Confirm that a hardcover edition exists and that the stated hardcover publication date is supported."
    )

    # Paperback edition leaf
    pb_leaf = evaluator.add_leaf(
        id=f"{book_id}_paperback_edition",
        desc="The book was released in paperback edition with publication date provided",
        parent=editions_node,
        critical=True
    )
    pb_claim = (
        f"The book '{title}' by {author} has a paperback edition with publication date {book.paperback_pub_date}."
    )
    await evaluator.verify(
        claim=pb_claim,
        node=pb_leaf,
        sources=_collect_urls(book, ["editions_urls", "publisher_urls", "publication_urls"]),
        additional_instruction="Confirm that a paperback edition exists and that the stated paperback publication date is supported."
    )

    # Paperback after Hardcover
    order_leaf = evaluator.add_leaf(
        id=f"{book_id}_paperback_after_hardcover",
        desc="The paperback edition was released after the hardcover edition",
        parent=editions_node,
        critical=True
    )
    order_claim = (
        f"For '{title}' by {author}, the paperback publication date ({book.paperback_pub_date}) is later than "
        f"the hardcover publication date ({book.hardcover_pub_date})."
    )
    await evaluator.verify(
        claim=order_claim,
        node=order_leaf,
        sources=_collect_urls(book, ["editions_urls", "publisher_urls", "publication_urls"]),
        additional_instruction="Use the dates on the provided pages to determine order; the paperback must be strictly later than the hardcover."
    )

    evaluator.add_custom_node(
        result=(len(book.editions_urls) > 0) or (len(book.publisher_urls) > 0),
        id=f"{book_id}_editions_reference",
        desc="A reference URL confirming both editions and their publication dates is provided",
        parent=editions_node,
        critical=True
    )

    # 6) Page count (non-critical)
    pages_node = evaluator.add_parallel(
        id=f"{book_id}_page_count",
        desc=f"Page count information for the {['first', 'second', 'third'][index_1based - 1]} book",
        parent=book_node,
        critical=False
    )

    pages_leaf = evaluator.add_leaf(
        id=f"{book_id}_hardcover_pages",
        desc="The page count for the hardcover edition is provided",
        parent=pages_node,
        critical=False
    )
    pages_claim = (
        f"The hardcover edition of '{title}' by {author} has {book.hardcover_pages} pages."
    )
    await evaluator.verify(
        claim=pages_claim,
        node=pages_leaf,
        sources=_collect_urls(book, ["pagecount_urls", "publisher_urls", "editions_urls"]),
        additional_instruction="Verify that the page count refers specifically to the hardcover edition if the page distinguishes formats."
    )

    evaluator.add_custom_node(
        result=(len(book.pagecount_urls) > 0) or (len(book.publisher_urls) > 0) or (len(book.editions_urls) > 0),
        id=f"{book_id}_page_reference",
        desc="A reference URL confirming the page count is provided",
        parent=pages_node,
        critical=False
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
    Evaluate an answer for the three-awarded-fiction-books task.
    """
    evaluator = Evaluator()
    # Note: We set root as non-critical to allow mixed critical/non-critical children (e.g., page count).
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
        default_model=model
    )

    # Extract up to three books
    extracted = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction"
    )

    # Prepare exactly three items (pad with empty if needed)
    books: List[BookItem] = list(extracted.books[:3])
    while len(books) < 3:
        books.append(BookItem())

    # Build verification subtrees for each book
    for idx, book in enumerate(books, start=1):
        await verify_one_book(evaluator, root, book, idx)

    # Return evaluation summary
    return evaluator.get_summary()