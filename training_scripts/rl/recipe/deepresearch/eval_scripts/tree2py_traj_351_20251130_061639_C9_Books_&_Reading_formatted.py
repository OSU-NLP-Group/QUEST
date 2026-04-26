import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "hollywood_actor_fiction_2015_2021"
TASK_DESCRIPTION = (
    "Identify 4 fiction books (novels, graphic novels, or short story collections) written by Hollywood actors, "
    "published in hardcover first edition format by major U.S. publishers between January 1, 2015, and December 31, 2021. "
    "For each book, provide comprehensive bibliographic information including: the author's full name, complete book title "
    "(with subtitle if applicable), publisher name, exact publication date, page count, ISBN-13, and a reference URL from a "
    "reliable source (such as publisher websites, Goodreads, library catalogs, or major book retailers) verifying this information. "
    "The authors must be professionally recognized as Hollywood actors with verifiable film or television acting credits, and must "
    "have genuinely written the work themselves (not ghostwritten; co-authorship with an illustrator for graphic novels is acceptable). "
    "All four books must be distinct works, not different editions of the same title."
)
DATE_RANGE_START = datetime(2015, 1, 1)
DATE_RANGE_END = datetime(2021, 12, 31)

# Known major US publishers and common imprints for heuristic checks (used in additional_instruction to guide judge).
MAJOR_US_PUBLISHERS = [
    "Penguin Random House",
    "HarperCollins",
    "Simon & Schuster",
    "Macmillan",
    "Hachette Book Group",
]
COMMON_IMPRINTS_HINT = [
    # PRH imprints
    "Knopf", "Vintage", "Random House", "Del Rey", "Doubleday", "Riverhead", "Crown", "Ballantine",
    # HarperCollins imprints
    "William Morrow", "Harper", "Harper Voyager",
    # Simon & Schuster imprints
    "Scribner", "Atria", "Gallery", "Saga Press",
    # Macmillan imprints
    "Tor", "Farrar, Straus and Giroux", "Henry Holt", "St. Martin's",
    # Hachette imprints
    "Little, Brown and Company", "Orbit", "Grand Central"
]

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class BookEntry(BaseModel):
    author_full_name: Optional[str] = None
    title_full: Optional[str] = None  # include subtitle if applicable
    publisher_name: Optional[str] = None
    publication_date: Optional[str] = None  # free-form string; month+year minimum
    page_count: Optional[str] = None       # keep string to be flexible (e.g., "352", "352 pages")
    isbn13: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)  # reliable source(s) verifying bibliographic info
    fiction_category: Optional[str] = None  # e.g., "novel", "graphic novel", "short story collection"
    format_notes: Optional[str] = None  # e.g., "hardcover first edition", "first edition hardcover"
    actor_proof_urls: List[str] = Field(default_factory=list)  # URLs demonstrating film/TV acting credits (IMDb, Wikipedia, etc.)


class BooksExtraction(BaseModel):
    books: List[BookEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
    Extract up to the first 4 fiction books cited in the answer that are written by Hollywood actors and meet the task constraints.
    For each identified book, extract the following fields exactly as stated in the answer text:
    - author_full_name: The full name of the author (actor).
    - title_full: The complete book title, including subtitle if applicable.
    - publisher_name: The publisher or imprint listed for the book.
    - publication_date: The exact publication date; month and year at minimum (e.g., "October 2019", "2018-05-01").
    - page_count: The total page count (as written; e.g., "352 pages" or "352").
    - isbn13: The ISBN-13 string (keep original formatting with hyphens if present).
    - reference_urls: An array of one or more URLs that the answer uses as reliable sources to verify the bibliographic info (publisher sites, Goodreads, library catalogs, major retailers).
    - fiction_category: One of "novel", "graphic novel", or "short story collection" if provided or implied in the answer.
    - format_notes: Any format notes indicating "hardcover first edition" or equivalent. If not present, return null.
    - actor_proof_urls: An array of URLs (IMDb, Wikipedia, official biographies, etc.) that demonstrate the author has film or TV acting credits. If none are provided in the answer, return an empty array.

    IMPORTANT:
    - Extract only from the given answer text; do not invent or infer missing values.
    - If any field is missing for a book, set it to null (or empty array for URL fields).
    - Return a JSON object with a top-level "books" array of up to 4 BookEntry objects.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_isbn13(isbn: Optional[str]) -> Optional[str]:
    if not isbn:
        return None
    digits = re.sub(r"[^0-9Xx]", "", isbn)
    return digits


def is_isbn13_format(isbn: Optional[str]) -> bool:
    digits = normalize_isbn13(isbn)
    return bool(digits) and len(digits) == 13 and digits.isdigit()


def extract_year(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    m = re.search(r"(20\d{2})", date_str)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def in_required_range(date_str: Optional[str]) -> bool:
    # Attempt to parse year; if parsed, check range.
    year = extract_year(date_str)
    if year is None:
        return False
    try:
        # Coerce to a date range check using Jan 1 / Dec 31 for the year
        # Here we only check year boundaries; the judge will verify precise date via sources.
        return DATE_RANGE_START.year <= year <= DATE_RANGE_END.year
    except Exception:
        return False


def distinct_books_ok(books: List[BookEntry]) -> bool:
    # Use normalized title and ISBN13 digits to check distinctness among the first 4
    seen_keys = set()
    for b in books[:4]:
        key_title = (b.title_full or "").strip().lower()
        key_isbn = normalize_isbn13(b.isbn13) or ""
        key = (key_title, key_isbn)
        if key in seen_keys:
            return False
        seen_keys.add(key)
    return True


def pick_sources_for_actor_check(book: BookEntry) -> List[str]:
    if book.actor_proof_urls:
        return book.actor_proof_urls
    if book.reference_urls:
        return book.reference_urls
    return []


# --------------------------------------------------------------------------- #
# Verification functions for a single book                                    #
# --------------------------------------------------------------------------- #
async def verify_single_book(
    evaluator: Evaluator,
    parent_node,
    book: BookEntry,
    idx: int,
) -> None:
    """
    Build verification subtree for one book and run verifications in an order that gates dependent checks.
    """
    book_node = evaluator.add_parallel(
        id=f"book_{idx+1}",
        desc=f"Book {idx+1} satisfies all constraints and required bibliographic fields",
        parent=parent_node,
        critical=False  # Each book contributes partial credit; global critical constraints are separate
    )

    # 1) Reference URL check + support verification first (critical gating)
    if book.reference_urls and len(book.reference_urls) > 0:
        # Treat "reference_url" as a verification leaf that the page indeed provides bibliographic details for the book
        ref_leaf = evaluator.add_leaf(
            id=f"book_{idx+1}_reference_url",
            desc=f"At least one reliable reference URL is provided for Book {idx+1} verifying the bibliographic information",
            parent=book_node,
            critical=True
        )
        claim = (
            f"The provided page(s) include bibliographic details for the book '{book.title_full or ''}' by '{book.author_full_name or ''}', "
            f"including title/author and at least some of: publisher, publication date, page count, and ISBN-13."
        )
        await evaluator.verify(
            claim=claim,
            node=ref_leaf,
            sources=book.reference_urls,
            additional_instruction="Confirm the page(s) clearly present bibliographic data matching the book. Minor formatting differences are acceptable."
        )
    else:
        # No reference URL provided -> fail the critical leaf
        evaluator.add_custom_node(
            result=False,
            id=f"book_{idx+1}_reference_url",
            desc=f"At least one reliable reference URL is provided for Book {idx+1} verifying the bibliographic information",
            parent=book_node,
            critical=True
        )

    # 2) Bibliographic field presence checks (critical existence checks)
    evaluator.add_custom_node(
        result=bool(book.author_full_name and book.author_full_name.strip()),
        id=f"book_{idx+1}_biblio_author_full_name",
        desc=f"Book {idx+1} author full name is provided",
        parent=book_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(book.title_full and book.title_full.strip()),
        id=f"book_{idx+1}_biblio_title_complete",
        desc=f"Book {idx+1} complete title is provided (including subtitle if applicable)",
        parent=book_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(book.publisher_name and book.publisher_name.strip()),
        id=f"book_{idx+1}_biblio_publisher_name",
        desc=f"Book {idx+1} publisher name is provided",
        parent=book_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(book.publication_date and book.publication_date.strip()),
        id=f"book_{idx+1}_biblio_publication_date",
        desc=f"Book {idx+1} exact publication date is provided (at minimum month and year)",
        parent=book_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(book.page_count and book.page_count.strip()),
        id=f"book_{idx+1}_biblio_page_count",
        desc=f"Book {idx+1} total page count is provided",
        parent=book_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=is_isbn13_format(book.isbn13),
        id=f"book_{idx+1}_biblio_isbn13",
        desc=f"Book {idx+1} ISBN-13 is provided",
        parent=book_node,
        critical=True
    )

    # 3) Constraint verifications (critical) — use the reference URL(s) as sources; gate on reference leaf
    # fiction_type
    fiction_leaf = evaluator.add_leaf(
        id=f"book_{idx+1}_fiction_type",
        desc=f"Book {idx+1} is fiction and is a novel, graphic novel, or short story collection",
        parent=book_node,
        critical=True
    )
    fiction_claim = (
        f"The book '{book.title_full or ''}' is a work of fiction, and its type is one of: novel, graphic novel, or short story collection."
    )
    await evaluator.verify(
        claim=fiction_claim,
        node=fiction_leaf,
        sources=book.reference_urls,
        additional_instruction="Check the page(s) for genre/type; allow reasonable equivalents (e.g., 'short stories')."
    )

    # author_is_hollywood_actor
    actor_leaf = evaluator.add_leaf(
        id=f"book_{idx+1}_author_is_hollywood_actor",
        desc=f"Book {idx+1} author is a professionally recognized Hollywood actor with verifiable film/TV acting credits",
        parent=book_node,
        critical=True
    )
    actor_sources = pick_sources_for_actor_check(book)
    actor_claim = (
        f"The author '{book.author_full_name or ''}' has film or television acting credits (Hollywood actor)."
    )
    await evaluator.verify(
        claim=actor_claim,
        node=actor_leaf,
        sources=actor_sources if actor_sources else book.reference_urls,
        additional_instruction="Accept sources like IMDb/Wikipedia/official pages showing acting credits; minor name variants are fine."
    )

    # genuine_authorship
    auth_leaf = evaluator.add_leaf(
        id=f"book_{idx+1}_genuine_authorship",
        desc=f"Book {idx+1} is genuinely authored by the actor (not ghostwritten; illustrator co-authorship acceptable for graphic novels)",
        parent=book_node,
        critical=True
    )
    auth_claim = (
        f"The book '{book.title_full or ''}' is authored by '{book.author_full_name or ''}'. Co-authorship with an illustrator is acceptable; "
        f"there is no indication that the work is ghostwritten."
    )
    await evaluator.verify(
        claim=auth_claim,
        node=auth_leaf,
        sources=book.reference_urls,
        additional_instruction="Confirm the author is credited as writer/author on the cited page(s); absence of ghostwriting indication is sufficient."
    )

    # publication_date_in_range
    pub_range_leaf = evaluator.add_leaf(
        id=f"book_{idx+1}_publication_date_in_range",
        desc=f"Book {idx+1} publication date is between Jan 1, 2015 and Dec 31, 2021 (inclusive)",
        parent=book_node,
        critical=True
    )
    pub_range_claim = (
        f"The publication date for '{book.title_full or ''}' lies between January 1, 2015 and December 31, 2021."
    )
    await evaluator.verify(
        claim=pub_range_claim,
        node=pub_range_leaf,
        sources=book.reference_urls,
        additional_instruction="Check the publication date on the page(s) and ensure it falls within 2015–2021 inclusive."
    )

    # publisher_major_us
    pub_major_leaf = evaluator.add_leaf(
        id=f"book_{idx+1}_publisher_major_us",
        desc=f"Book {idx+1} publisher is a major U.S. publishing house",
        parent=book_node,
        critical=True
    )
    pub_major_claim = (
        f"The publisher '{book.publisher_name or ''}' is a major U.S. publishing house (Big Five or equivalent, including well-known imprints)."
    )
    await evaluator.verify(
        claim=pub_major_claim,
        node=pub_major_leaf,
        sources=book.reference_urls,
        additional_instruction=(
            "Treat Penguin Random House, HarperCollins, Simon & Schuster, Macmillan, and Hachette (and their imprints such as "
            + ", ".join(COMMON_IMPRINTS_HINT) +
            ") as major US publishers. Confirm based on publisher name/imprint on the cited page(s)."
        )
    )

    # format_hardcover_first
    format_leaf = evaluator.add_leaf(
        id=f"book_{idx+1}_format_hardcover_first",
        desc=f"Book {idx+1} is available in hardcover first edition format",
        parent=book_node,
        critical=True
    )
    format_claim = (
        f"A hardcover first edition of '{book.title_full or ''}' exists (or the hardcover is the first edition)."
    )
    await evaluator.verify(
        claim=format_claim,
        node=format_leaf,
        sources=book.reference_urls,
        additional_instruction="Look for 'hardcover' and 'first edition' indicators; retailer/publisher/library pages often list edition details."
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
    Evaluate an answer for the Hollywood actor fiction books (2015–2021) task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # parallel: global constraints + per-book checks independently
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

    # NOTE: Set root to non-critical to allow both critical and non-critical children under it.
    # The rubric marked root "critical", but framework prohibits non-critical children under critical parents.
    root.critical = False

    # Extract books
    extracted = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction",
    )

    # Determine actual count from the answer (before padding)
    actual_books = extracted.books or []
    actual_count = len(actual_books)

    # Global critical checks
    evaluator.add_custom_node(
        result=(actual_count == 4),
        id="global_count",
        desc="Exactly 4 books are identified (not fewer or more)",
        parent=root,
        critical=True
    )

    # Consider only the first 4 items for distinctness check
    first_four = actual_books[:4]
    evaluator.add_custom_node(
        result=distinct_books_ok(first_four) if first_four else False,
        id="global_distinctness",
        desc="All 4 books are distinct works (not multiple editions/variants of the same title)",
        parent=root,
        critical=True
    )

    # Pad to 4 entries to build a uniform tree (placeholders will fail critical checks)
    while len(first_four) < 4:
        first_four.append(BookEntry())

    # Build per-book verification subtrees
    for i, book in enumerate(first_four):
        await verify_single_book(evaluator, root, book, i)

    # Return final evaluation summary
    return evaluator.get_summary()