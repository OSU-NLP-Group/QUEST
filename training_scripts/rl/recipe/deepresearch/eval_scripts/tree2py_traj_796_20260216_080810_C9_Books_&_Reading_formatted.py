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
TASK_ID = "rachel_reid_game_changers_4books_2019_2025"
TASK_DESCRIPTION = """Identify exactly 4 books from Rachel Reid's Game Changers series that were published between 2019 and 2025 (inclusive). For each of the 4 books you identify, provide the following comprehensive information:

1. Title Information:
   - The complete official title of the book
   - Confirmation that it is part of the Game Changers series

2. Publication Details:
   - The exact publication date (month, day, and year)
   - The publisher name
   - A valid ISBN-13 number for any edition of the book
   - A reference URL that supports these publication details

3. Content Details:
   - The page count for a specific edition of the book
   - The names of both main characters featured in the book
   - A reference URL that supports these content details

4. Audiobook Details:
   - The name of the audiobook narrator
   - A reference URL that supports the audiobook narrator information

5. Series Position:
   - The book's position number in the Game Changers series (e.g., Book 1, Book 2, etc.)

All information must be accurate and verifiable through the provided reference URLs. Each book must be part of the Game Changers series and published in 2019–2025 (inclusive)."""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BookEntry(BaseModel):
    # Title information
    title: Optional[str] = None
    series_name: Optional[str] = None  # e.g., "Game Changers" or "A Game Changers Novel"

    # Publication details
    publication_date: Optional[str] = None  # keep as string for flexibility
    publisher: Optional[str] = None
    isbn13: Optional[str] = None
    publication_url: Optional[str] = None

    # Content details
    page_count: Optional[str] = None
    main_characters: List[str] = Field(default_factory=list)
    content_url: Optional[str] = None

    # Audiobook details
    narrator: Optional[str] = None
    audiobook_url: Optional[str] = None

    # Series position
    series_position: Optional[str] = None  # e.g., "Book 3", "#3", "3"


class BooksExtraction(BaseModel):
    books: List[BookEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
    Extract up to the first 4 books from Rachel Reid's Game Changers series as they are presented in the answer. 
    For each identified book, return the following fields exactly as stated in the answer (do not infer or invent):
    - title: Complete official title string.
    - series_name: The series name string if stated (e.g., "Game Changers", "A Game Changers Novel"); otherwise null.
    - publication_date: The exact publication date string (include month, day, year if provided; leave as-is).
    - publisher: The publisher string as presented.
    - isbn13: A 13-digit ISBN string if present (allow hyphens/spaces); otherwise null.
    - publication_url: A single reference URL cited for publication details if present; otherwise null.
    - page_count: The page count string for a specific edition if present (e.g., "352 pages", "352"); otherwise null.
    - main_characters: An array of the two main character names if provided; if only one is given, include it; otherwise return an empty array.
    - content_url: A single reference URL cited for content details if present; otherwise null.
    - narrator: The audiobook narrator name if provided; otherwise null.
    - audiobook_url: A single reference URL cited for audiobook narrator information if present; otherwise null.
    - series_position: The book’s position in the Game Changers series as provided in the answer (e.g., "Book 1", "#3", "3"); otherwise null.

    Rules:
    - Only extract information explicitly present in the answer text.
    - If the answer mentions more than 4 books, only return the first 4 that belong to the Game Changers series.
    - If fewer than 4 are provided, still return those, with missing fields as null/empty.
    - For any missing field, set it to null (or empty array for main_characters).
    - Do not convert or standardize values—preserve the formatting and wording from the answer.

    Output JSON schema:
    {
      "books": [
        {
          "title": str|null,
          "series_name": str|null,
          "publication_date": str|null,
          "publisher": str|null,
          "isbn13": str|null,
          "publication_url": str|null,
          "page_count": str|null,
          "main_characters": [str, ...],
          "content_url": str|null,
          "narrator": str|null,
          "audiobook_url": str|null,
          "series_position": str|null
        }, ...
      ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _dedup_urls(urls: List[Optional[str]]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for u in urls:
        if _non_empty(u):
            us = u.strip()
            if us not in seen:
                seen.add(us)
                ordered.append(us)
    return ordered


def _within_year_range_instruction() -> str:
    return "Additionally ensure that the publication year lies between 2019 and 2025 inclusive."


# --------------------------------------------------------------------------- #
# Verification for a single book                                              #
# --------------------------------------------------------------------------- #
async def verify_single_book(
    evaluator: Evaluator,
    parent_node,
    book: BookEntry,
    index: int,
) -> None:
    # Create Book node (parallel, non-critical)
    book_node = evaluator.add_parallel(
        id=f"book_{index+1}",
        desc=f"{['First','Second','Third','Fourth'][index]} identified Game Changers series book (published 2019-2025)",
        parent=parent_node,
        critical=False
    )

    # Aggregate available sources
    all_sources = _dedup_urls([book.publication_url, book.content_url, book.audiobook_url])

    # ---------------------------- Title Info -------------------------------- #
    title_info_node = evaluator.add_parallel(
        id=f"book_{index+1}_title_info",
        desc=f"Complete and accurate title information for Book {index+1}",
        parent=book_node,
        critical=True
    )

    # Full Title leaf
    title_leaf = evaluator.add_leaf(
        id=f"book_{index+1}_full_title",
        desc="The complete official title of the book is provided",
        parent=title_info_node,
        critical=True
    )
    # If no title or no sources, fail this leaf (we require URL grounding)
    if not _non_empty(book.title) or len(all_sources) == 0:
        title_leaf.score = 0.0
        title_leaf.status = "failed"
    else:
        title_claim = f"The official title of the book is '{book.title}'. Allow minor punctuation/casing variations."
        await evaluator.verify(
            claim=title_claim,
            node=title_leaf,
            sources=all_sources,
            additional_instruction="Verify that the page shows this book title (allow minor punctuation, capitalization, or subtitle formatting differences)."
        )

    # Series Designation leaf
    series_leaf = evaluator.add_leaf(
        id=f"book_{index+1}_series_designation",
        desc="The book is correctly identified as part of the Game Changers series",
        parent=title_info_node,
        critical=True
    )
    if len(all_sources) == 0:
        series_leaf.score = 0.0
        series_leaf.status = "failed"
    else:
        series_claim = "This book is part of Rachel Reid's 'Game Changers' series."
        await evaluator.verify(
            claim=series_claim,
            node=series_leaf,
            sources=all_sources,
            additional_instruction="Look for signals like 'Game Changers', 'Game Changers series', 'A Game Changers Novel', or 'Game Changers #n'. Allow minor format variants."
        )

    # ------------------------- Publication Details -------------------------- #
    pub_node = evaluator.add_parallel(
        id=f"book_{index+1}_publication_details",
        desc=f"Accurate publication information for Book {index+1}",
        parent=book_node,
        critical=True
    )

    # Publication URL provided (existence check)
    evaluator.add_custom_node(
        result=_non_empty(book.publication_url),
        id=f"book_{index+1}_publication_url",
        desc="A reference URL supporting the publication details is provided",
        parent=pub_node,
        critical=True
    )

    # Publication Date leaf
    pub_date_leaf = evaluator.add_leaf(
        id=f"book_{index+1}_publication_date",
        desc="The exact publication date (month, day, year) is provided and correct",
        parent=pub_node,
        critical=True
    )
    # Always call verify; preconditions will skip if publication_url missing
    pub_sources = _dedup_urls([book.publication_url])
    pub_date_str = book.publication_date or ""
    pub_date_claim = f"The publication date for this book is '{pub_date_str}'. {_within_year_range_instruction()}"
    await evaluator.verify(
        claim=pub_date_claim,
        node=pub_date_leaf,
        sources=pub_sources if len(pub_sources) > 0 else None,
        additional_instruction="Match the full date if possible. Accept equivalent formats. Also confirm the year is within 2019–2025."
    )

    # Publisher leaf
    publisher_leaf = evaluator.add_leaf(
        id=f"book_{index+1}_publisher",
        desc="The publisher (Carina Press) is correctly identified",
        parent=pub_node,
        critical=True
    )
    publisher_val = book.publisher or ""
    publisher_claim = f"The publisher of this book is '{publisher_val}'."
    await evaluator.verify(
        claim=publisher_claim,
        node=publisher_leaf,
        sources=pub_sources if len(pub_sources) > 0 else None,
        additional_instruction="Verify the listed publisher matches the page. If 'Carina Adores' is shown, treat it as an imprint of Carina Press (acceptable variant)."
    )

    # ISBN-13 leaf
    isbn_leaf = evaluator.add_leaf(
        id=f"book_{index+1}_isbn13",
        desc="A valid ISBN-13 for the book is provided",
        parent=pub_node,
        critical=True
    )
    isbn_val = book.isbn13 or ""
    isbn_claim = f"An ISBN-13 for this book is '{isbn_val}'."
    await evaluator.verify(
        claim=isbn_claim,
        node=isbn_leaf,
        sources=pub_sources if len(pub_sources) > 0 else None,
        additional_instruction="Confirm the page lists this ISBN-13 (ignore hyphens/spaces). Any edition’s ISBN-13 is acceptable."
    )

    # --------------------------- Content Details ---------------------------- #
    content_node = evaluator.add_parallel(
        id=f"book_{index+1}_content_details",
        desc=f"Accurate content information for Book {index+1}",
        parent=book_node,
        critical=True
    )

    # Content URL provided (existence)
    evaluator.add_custom_node(
        result=_non_empty(book.content_url),
        id=f"book_{index+1}_content_url",
        desc="A reference URL supporting the content details is provided",
        parent=content_node,
        critical=True
    )

    # Page Count leaf
    page_leaf = evaluator.add_leaf(
        id=f"book_{index+1}_page_count",
        desc="The page count for a specific edition is provided",
        parent=content_node,
        critical=True
    )
    content_sources = _dedup_urls([book.content_url])
    page_val = book.page_count or ""
    page_claim = f"The page count for this book (for a specific edition) is '{page_val}'."
    await evaluator.verify(
        claim=page_claim,
        node=page_leaf,
        sources=content_sources if len(content_sources) > 0 else None,
        additional_instruction="Check for 'pages' or 'print length' on the page. Edition-specific counts are acceptable; match the extracted count."
    )

    # Main Characters leaf
    chars_leaf = evaluator.add_leaf(
        id=f"book_{index+1}_main_characters",
        desc="Both main character names are correctly identified",
        parent=content_node,
        critical=True
    )
    # Build character claim
    if len(book.main_characters) >= 2:
        char_a, char_b = book.main_characters[0], book.main_characters[1]
        chars_claim = f"The two main characters of this book are '{char_a}' and '{char_b}'."
    elif len(book.main_characters) == 1:
        chars_claim = f"One of the two main characters of this book is '{book.main_characters[0]}'."
    else:
        chars_claim = "The two main characters of this book are correctly identified in the answer."
    await evaluator.verify(
        claim=chars_claim,
        node=chars_leaf,
        sources=content_sources if len(content_sources) > 0 else None,
        additional_instruction="Verify that the page mentions these as the central/main leads. Allow minor name variants (nicknames, shortened forms)."
    )

    # -------------------------- Audiobook Details --------------------------- #
    audio_node = evaluator.add_parallel(
        id=f"book_{index+1}_audiobook_details",
        desc=f"Accurate audiobook information for Book {index+1}",
        parent=book_node,
        critical=True
    )

    # Audiobook URL provided (existence)
    evaluator.add_custom_node(
        result=_non_empty(book.audiobook_url),
        id=f"book_{index+1}_audiobook_url",
        desc="A reference URL supporting the audiobook narrator information is provided",
        parent=audio_node,
        critical=True
    )

    # Narrator leaf
    narrator_leaf = evaluator.add_leaf(
        id=f"book_{index+1}_narrator",
        desc="The audiobook narrator name is correctly identified",
        parent=audio_node,
        critical=True
    )
    audio_sources = _dedup_urls([book.audiobook_url])
    narrator_val = book.narrator or ""
    narrator_claim = f"The audiobook narrator for this book is '{narrator_val}'."
    await evaluator.verify(
        claim=narrator_claim,
        node=narrator_leaf,
        sources=audio_sources if len(audio_sources) > 0 else None,
        additional_instruction="Look for phrases like 'Narrated by' or 'Narrator'. Confirm the extracted narrator appears on the page."
    )

    # ---------------------------- Series Position --------------------------- #
    series_pos_leaf = evaluator.add_leaf(
        id=f"book_{index+1}_series_position",
        desc="The book's position number in the Game Changers series is correctly identified",
        parent=book_node,
        critical=True
    )
    pos_val = book.series_position or ""
    pos_claim = f"This book's position in the 'Game Changers' series is '{pos_val}'."
    # Prefer any available URL(s)
    pos_sources = all_sources
    if len(pos_sources) == 0:
        # No grounding sources; mark as failed
        series_pos_leaf.score = 0.0
        series_pos_leaf.status = "failed"
    else:
        await evaluator.verify(
            claim=pos_claim,
            node=series_pos_leaf,
            sources=pos_sources,
            additional_instruction="Accept formats like 'Game Changers #n', 'Book n', or 'Book n of Game Changers'."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Rachel Reid Game Changers 4-books task.
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
        default_model=model
    )

    # Extract up to 4 books
    extracted = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction"
    )

    # Keep exactly 4 slots: truncate or pad with empty entries
    books: List[BookEntry] = (extracted.books or [])[:4]
    while len(books) < 4:
        books.append(BookEntry())

    # Build four book subtrees in parallel under root
    # Book 1
    await verify_single_book(evaluator, root, books[0], 0)
    # Book 2
    await verify_single_book(evaluator, root, books[1], 1)
    # Book 3
    await verify_single_book(evaluator, root, books[2], 2)
    # Book 4
    await verify_single_book(evaluator, root, books[3], 3)

    return evaluator.get_summary()