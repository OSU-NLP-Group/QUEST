import asyncio
import logging
from typing import Optional, List, Dict, Any
import re

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "literary_awards_2024_books"
TASK_DESCRIPTION = (
    "Identify four books that won major literary awards announced in 2024 (including awards for books published in 2023-2024). "
    "Each book must have won a different award from the following list: Booker Prize, National Book Award for Fiction, "
    "Women's Prize for Fiction, PEN/Faulkner Award for Fiction, Kirkus Prize, International Booker Prize, Pulitzer Prize for Fiction, "
    "or Giller Prize. For each book, provide: (1) The specific award won, (2) Complete book title and author name, "
    "(3) Publisher name and type (major publisher or independent press), (4) Available formats (hardcover, paperback, e-book, audiobook), "
    "(5) Primary publication format, (6) Page count, (7) Genre/category, (8) Publication year, and (9) Reference URLs verifying all information. "
    "All four books must be from different awards, and all information must be verifiable through authoritative sources."
)

PERMITTED_AWARDS = [
    "Booker Prize",
    "National Book Award for Fiction",
    "Women's Prize for Fiction",
    "PEN/Faulkner Award for Fiction",
    "Kirkus Prize",
    "International Booker Prize",
    "Pulitzer Prize for Fiction",
    "Giller Prize",
]

# Some common synonyms or variants -> canonical mapping
AWARD_SYNONYM_MAP = {
    "the booker prize": "Booker Prize",
    "man booker prize": "Booker Prize",
    "booker prize": "Booker Prize",
    "international booker prize": "International Booker Prize",
    "women's prize": "Women's Prize for Fiction",
    "women’s prize": "Women's Prize for Fiction",
    "women's prize for fiction": "Women's Prize for Fiction",
    "pen/faulkner award for fiction": "PEN/Faulkner Award for Fiction",
    "pen-faulkner award for fiction": "PEN/Faulkner Award for Fiction",
    "kirkus prize": "Kirkus Prize",
    "pulitzer prize for fiction": "Pulitzer Prize for Fiction",
    "national book award for fiction": "National Book Award for Fiction",
    "national book award – fiction": "National Book Award for Fiction",
    "national book award: fiction": "National Book Award for Fiction",
    "giller prize": "Giller Prize",
    "scotiabank giller prize": "Giller Prize",
}

ALLOWED_FORMATS = {"hardcover", "paperback", "e-book", "audiobook"}

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class BookEntry(BaseModel):
    """One book with all required fields."""
    award: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    publisher_name: Optional[str] = None
    publisher_type: Optional[str] = None  # Expected values: "major publisher" or "independent press"
    available_formats: List[str] = Field(default_factory=list)  # Allowed categories
    primary_format: Optional[str] = None  # One of allowed categories
    page_count: Optional[str] = None  # Prefer string for robustness
    genre_category: Optional[str] = None
    publication_year: Optional[str] = None  # Prefer string for robustness
    reference_urls: List[str] = Field(default_factory=list)  # Authoritative URLs


class BooksExtraction(BaseModel):
    """List of up to 4 books extracted from answer."""
    books: List[BookEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    permitted = ", ".join(PERMITTED_AWARDS)
    return f"""
    Extract up to four books (first four only) presented in the answer that won awards announced in 2024.
    For each book, extract the following fields exactly as stated in the answer:
    - award: The exact award name (e.g., one of [{permitted}]). Keep the name as provided in the answer.
    - title: The complete book title
    - author: The author name (primary author; include co-authors if the answer states them)
    - publisher_name: The publisher/imprint name
    - publisher_type: Either "major publisher" or "independent press" (use exactly these labels if the answer provides them; if not provided, return null)
    - available_formats: A list of format labels from these categories only: "hardcover", "paperback", "e-book", "audiobook".
      Normalize each format to one of these four labels if the answer clearly indicates it (e.g., "ebook" -> "e-book", "audio book" -> "audiobook").
      If unclear or unspecified, do not invent; omit that format.
    - primary_format: The primary publication format, one of the allowed categories above. If not explicitly stated, return null.
    - page_count: The page count string exactly as stated; do not convert to a number if not clearly stated.
    - genre_category: The genre/category string (e.g., "literary fiction", "historical fiction"). If not stated, return null.
    - publication_year: The publication year as a string (e.g., "2023" or "2024"). If not clearly stated, return null.
    - reference_urls: A list of authoritative URLs the answer cites for this book. Include official award pages, publisher pages, major media, library catalogs,
      or Google Books/ISBN catalog pages if present. If the answer does not provide URLs for this book, return an empty list.

    Return a JSON object with a top-level "books" array of objects using the fields above.
    Do not add or infer information; extract only what the answer explicitly provides.
    """


# --------------------------------------------------------------------------- #
# Helper normalization & checks                                               #
# --------------------------------------------------------------------------- #
def canonical_award(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    key = name.strip().lower()
    # Normalize quotes and dashes
    key = key.replace("’", "'").replace("–", "-").strip()
    if key in AWARD_SYNONYM_MAP:
        return AWARD_SYNONYM_MAP[key]
    # If it's already exactly permitted (case-insensitive), match to canonical case
    for p in PERMITTED_AWARDS:
        if key == p.lower():
            return p
    return None  # Not recognized


def normalize_format_tag(fmt: str) -> Optional[str]:
    s = fmt.strip().lower()
    s = s.replace(" ", "").replace("-", "")
    if s in {"hardcover", "hardback", "hardcoveredition"}:
        return "hardcover"
    if s in {"paperback", "softcover", "paperbackedition"}:
        return "paperback"
    if s in {"ebook", "digital", "kindle", "digitaledition", "e-book"}:
        return "e-book"
    if s in {"audiobook", "audiobookedition", "audiobookaudio", "audiobookaudioedition", "audiobookaudioformat", "audiobookaudiofile", "audiobookaudiofiles"}:
        return "audiobook"
    # If answer used "audio book", handle that
    if s in {"audiobook", "audiobookaudio", "audiobookaudioedition"}:
        return "audiobook"
    return None


def normalize_formats_list(formats: List[str]) -> List[str]:
    out = []
    for f in formats:
        nf = normalize_format_tag(f)
        if nf:
            out.append(nf)
    # Deduplicate while preserving order
    seen = set()
    normalized_unique = []
    for f in out:
        if f not in seen:
            normalized_unique.append(f)
            seen.add(f)
    return normalized_unique


def normalize_publisher_type(ptype: Optional[str]) -> Optional[str]:
    if not ptype:
        return None
    s = ptype.strip().lower()
    s = s.replace("–", "-")
    if "major" in s:
        return "major publisher"
    if "independent" in s or "small press" in s or "indie" in s:
        return "independent press"
    # If exactly matches expected labels
    if s == "major publisher":
        return "major publisher"
    if s == "independent press":
        return "independent press"
    return None


def parse_page_count(count_str: Optional[str]) -> Optional[int]:
    if not count_str:
        return None
    # Extract the first reasonable integer from the string
    nums = re.findall(r"\d{2,4}", count_str)
    if not nums:
        return None
    try:
        val = int(nums[0])
        return val
    except Exception:
        return None


def parse_year(year_str: Optional[str]) -> Optional[int]:
    if not year_str:
        return None
    # Extract first four-digit year
    nums = re.findall(r"\b(20\d{2})\b", year_str)
    if not nums:
        return None
    try:
        return int(nums[0])
    except Exception:
        return None


def check_award_uniqueness(books: List[BookEntry]) -> bool:
    # Must have 4 recognized canonical awards and all distinct
    if len(books) < 4:
        return False
    canon = []
    for b in books[:4]:
        canon.append(canonical_award(b.award))
    if any(c is None for c in canon):
        return False
    return len(set(canon)) == 4


# --------------------------------------------------------------------------- #
# Verification functions per book                                             #
# --------------------------------------------------------------------------- #
async def verify_book(
    evaluator: Evaluator,
    parent_node,
    book: BookEntry,
    book_index: int,
) -> None:
    """
    Build verification sub-tree for a single book and run verifications.
    book_index is 1-based.
    """
    # Create the Book node
    book_node = evaluator.add_parallel(
        id=f"Book_{book_index}",
        desc=f"Book {book_index} entry meets the required constraints and fields",
        parent=parent_node,
        critical=False  # Allow partial credit per book
    )

    # ---------------- Award Info (parallel) ----------------
    award_info_node = evaluator.add_parallel(
        id=f"Book_{book_index}_Award_Info",
        desc=f"Award info for Book {book_index} satisfies the task constraints",
        parent=book_node,
        critical=False
    )

    # Book_i_Award_From_Permitted_List (leaf)
    award_name = book.award or ""
    award_from_list_leaf = evaluator.add_leaf(
        id=f"Book_{book_index}_Award_From_Permitted_List",
        desc="Award name is one of the permitted awards listed in the prompt",
        parent=award_info_node,
        critical=True,
    )
    claim_award_in_list = (
        f"The award '{award_name}' is one of the permitted awards "
        f"(allowing canonical naming): {', '.join(PERMITTED_AWARDS)}."
    )
    await evaluator.verify(
        claim=claim_award_in_list,
        node=award_from_list_leaf,
        additional_instruction=(
            "Treat commonly used canonical or synonymous names as valid (e.g., 'Scotiabank Giller Prize' → 'Giller Prize', "
            "'The Booker Prize' → 'Booker Prize')."
        ),
    )

    # Book_i_Award_Announced_2024 (leaf) - VERIFIED VIA URLs
    award_2024_leaf = evaluator.add_leaf(
        id=f"Book_{book_index}_Award_Announced_2024",
        desc="The award win is for an award announcement made in 2024",
        parent=award_info_node,
        critical=True,
    )
    claim_award_2024 = (
        f"The provided references show that the book '{book.title or ''}' by {book.author or ''} "
        f"won the '{award_name}', and the official winner announcement occurred in 2024."
    )
    await evaluator.verify(
        claim=claim_award_2024,
        node=award_2024_leaf,
        sources=book.reference_urls,
        additional_instruction=(
            "Confirm the winner announcement year is 2024 from official award pages or authoritative media. "
            "Even if the book was published in 2023, the win must be announced in 2024."
        ),
    )

    # Book_i_Title_Author provided (custom)
    title_author_exists = bool((book.title or "").strip()) and bool((book.author or "").strip())
    evaluator.add_custom_node(
        result=title_author_exists,
        id=f"Book_{book_index}_Title_Author",
        desc="Complete book title and author name are provided",
        parent=award_info_node,
        critical=True
    )

    # ---------------- Publisher Info (parallel) ----------------
    publisher_info_node = evaluator.add_parallel(
        id=f"Book_{book_index}_Publisher_Info",
        desc=f"Publisher details for Book {book_index}",
        parent=book_node,
        critical=False
    )

    # Publisher name provided
    publisher_name_exists = bool((book.publisher_name or "").strip())
    evaluator.add_custom_node(
        result=publisher_name_exists,
        id=f"Book_{book_index}_Publisher_Name",
        desc="Publisher name is provided",
        parent=publisher_info_node,
        critical=True
    )

    # Publisher type categorized
    normalized_ptype = normalize_publisher_type(book.publisher_type)
    evaluator.add_custom_node(
        result=normalized_ptype in {"major publisher", "independent press"},
        id=f"Book_{book_index}_Publisher_Type",
        desc="Publisher type is categorized as major publisher or independent press",
        parent=publisher_info_node,
        critical=True
    )

    # ---------------- Format Info (parallel) ----------------
    format_info_node = evaluator.add_parallel(
        id=f"Book_{book_index}_Format_Info",
        desc=f"Formats and primary format for Book {book_index}",
        parent=book_node,
        critical=False
    )

    normalized_formats = normalize_formats_list(book.available_formats or [])
    formats_valid = bool(normalized_formats) and all(f in ALLOWED_FORMATS for f in normalized_formats)
    evaluator.add_custom_node(
        result=formats_valid,
        id=f"Book_{book_index}_Available_Formats",
        desc="Available formats are listed using the required format categories (hardcover, paperback, e-book, audiobook) as applicable",
        parent=format_info_node,
        critical=True
    )

    primary_fmt_norm = normalize_format_tag(book.primary_format or "") or ""
    primary_format_valid = primary_fmt_norm in ALLOWED_FORMATS
    evaluator.add_custom_node(
        result=primary_format_valid,
        id=f"Book_{book_index}_Primary_Format",
        desc="Primary publication format is identified",
        parent=format_info_node,
        critical=True
    )

    # ---------------- Bibliographic Data (parallel) ----------------
    biblio_node = evaluator.add_parallel(
        id=f"Book_{book_index}_Bibliographic_Data",
        desc=f"Bibliographic requirements for Book {book_index}",
        parent=book_node,
        critical=False
    )

    # Page count provided & in reasonable range
    pages_val = parse_page_count(book.page_count)
    pages_reasonable = pages_val is not None and 30 <= pages_val <= 2000
    evaluator.add_custom_node(
        result=pages_reasonable,
        id=f"Book_{book_index}_Page_Count",
        desc="Page count is provided and is within a reasonable range for the stated genre/category",
        parent=biblio_node,
        critical=True
    )

    # Genre category provided
    genre_exists = bool((book.genre_category or "").strip())
    evaluator.add_custom_node(
        result=genre_exists,
        id=f"Book_{book_index}_Genre_Category",
        desc="Genre/category is provided",
        parent=biblio_node,
        critical=True
    )

    # Publication year is 2023 or 2024
    pub_year_val = parse_year(book.publication_year)
    pub_year_ok = pub_year_val in (2023, 2024)
    evaluator.add_custom_node(
        result=pub_year_ok,
        id=f"Book_{book_index}_Publication_Year",
        desc="Publication year is 2023 or 2024",
        parent=biblio_node,
        critical=True
    )

    # ---------------- References (parallel, critical) ----------------
    # Convert the single "References" leaf in the rubric into granular verification leaves
    # to ensure evidence-backed checks for each required attribute.
    references_node = evaluator.add_parallel(
        id=f"Book_{book_index}_References",
        desc=("Provides authoritative reference URLs that collectively verify the award win (including 2024 announcement), "
              "title/author, publisher name/type, formats/primary format, page count, genre/category, and publication year"),
        parent=book_node,
        critical=True
    )

    # Prepare leaf nodes and claims for batch verification
    ref_tasks: List[tuple[str, List[str], Any, Optional[str]]] = []

    # 1) Award win with 2024 announcement (redundant but demanded evidence-backed)
    ref_award_leaf = evaluator.add_leaf(
        id=f"Book_{book_index}_Ref_Award_2024",
        desc="References verify the award win and that the announcement occurred in 2024",
        parent=references_node,
        critical=True
    )
    ref_award_claim = (
        f"The sources show that '{book.title or ''}' by {book.author or ''} won the '{award_name}', "
        f"and the winner announcement occurred in 2024."
    )
    ref_tasks.append((
        ref_award_claim,
        book.reference_urls,
        ref_award_leaf,
        "Use official award sites or credible media coverage to confirm both the win and that the announcement took place in 2024."
    ))

    # 2) Title & author
    ref_title_author_leaf = evaluator.add_leaf(
        id=f"Book_{book_index}_Ref_Title_Author",
        desc="References verify the complete book title and author",
        parent=references_node,
        critical=True
    )
    ref_title_author_claim = f"The sources show the book titled '{book.title or ''}' authored by {book.author or ''}."
    ref_tasks.append((
        ref_title_author_claim,
        book.reference_urls,
        ref_title_author_leaf,
        "Allow minor formatting variations; accept publisher pages, library catalogs, or reputable listings."
    ))

    # 3) Publisher name
    ref_publisher_leaf = evaluator.add_leaf(
        id=f"Book_{book_index}_Ref_Publisher",
        desc="References verify the publisher name",
        parent=references_node,
        critical=True
    )
    ref_publisher_claim = f"The sources explicitly indicate the publisher as '{book.publisher_name or ''}'."
    ref_tasks.append((
        ref_publisher_claim,
        book.reference_urls,
        ref_publisher_leaf,
        "Publisher or imprint pages are authoritative; library catalogs or reputable retail listings are acceptable."
    ))

    # 4) Primary format
    ref_primary_format_leaf = evaluator.add_leaf(
        id=f"Book_{book_index}_Ref_Primary_Format",
        desc="References verify the primary publication format",
        parent=references_node,
        critical=True
    )
    ref_primary_format_claim = f"The sources indicate the primary publication format is '{primary_fmt_norm or (book.primary_format or '')}'."
    ref_tasks.append((
        ref_primary_format_claim,
        book.reference_urls,
        ref_primary_format_leaf,
        "Confirm the primary format from publisher pages or catalog listings; allow reasonable equivalents."
    ))

    # 5) At least one listed available format
    # Verify existence of at least one of the available formats in references
    ref_any_format_leaf = evaluator.add_leaf(
        id=f"Book_{book_index}_Ref_Any_Format",
        desc="References verify at least one of the listed available formats",
        parent=references_node,
        critical=True
    )
    if normalized_formats:
        any_fmt = normalized_formats[0]
    else:
        any_fmt = ""
    ref_any_format_claim = f"The sources indicate the book is available in {any_fmt} format."
    ref_tasks.append((
        ref_any_format_claim,
        book.reference_urls,
        ref_any_format_leaf,
        "Check if any source mentions the availability of the indicated format; allow obvious equivalents (e.g., 'ebook'/'e-book')."
    ))

    # 6) Page count
    ref_pages_leaf = evaluator.add_leaf(
        id=f"Book_{book_index}_Ref_Page_Count",
        desc="References verify the page count",
        parent=references_node,
        critical=True
    )
    ref_pages_claim = f"The sources show the page count is '{book.page_count or ''}'."
    ref_tasks.append((
        ref_pages_claim,
        book.reference_urls,
        ref_pages_leaf,
        "Page counts on publisher pages or library catalogs are authoritative; accept minor numeric rounding differences."
    ))

    # 7) Genre/category
    ref_genre_leaf = evaluator.add_leaf(
        id=f"Book_{book_index}_Ref_Genre",
        desc="References verify the genre/category",
        parent=references_node,
        critical=True
    )
    ref_genre_claim = f"The sources indicate the genre/category is '{book.genre_category or ''}'."
    ref_tasks.append((
        ref_genre_claim,
        book.reference_urls,
        ref_genre_leaf,
        "Genre labels can vary slightly by source; accept clearly equivalent categories."
    ))

    # 8) Publication year
    ref_pubyear_leaf = evaluator.add_leaf(
        id=f"Book_{book_index}_Ref_Publication_Year",
        desc="References verify the publication year",
        parent=references_node,
        critical=True
    )
    ref_pubyear_claim = f"The sources indicate the publication year as '{book.publication_year or ''}'."
    ref_tasks.append((
        ref_pubyear_claim,
        book.reference_urls,
        ref_pubyear_leaf,
        "Confirm the publication year from publisher pages, ISBN catalogs, or reputable listings."
    ))

    # Run batch verification for reference-backed leaves
    await evaluator.batch_verify(ref_tasks)


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
    Evaluate an answer for the 'literary_awards_2024_books' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root: parallel aggregation
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

    # IMPORTANT: Set root as non-critical to allow partial credit across books (and to satisfy critical consistency constraint)
    root.critical = False

    # Record permitted awards list for transparency
    evaluator.add_custom_info(
        info={"permitted_awards": PERMITTED_AWARDS},
        info_type="config",
        info_name="permitted_awards"
    )

    # Extract books from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction",
    )

    # Keep exactly 4 books (pad with empty entries if fewer)
    books = list(extraction.books[:4])
    while len(books) < 4:
        books.append(BookEntry())

    # Award uniqueness check (critical)
    evaluator.add_custom_node(
        result=check_award_uniqueness(books),
        id="Award_Uniqueness",
        desc="All 4 books correspond to 4 different awards (no duplicate awards across the four selections)",
        parent=root,
        critical=True
    )

    # Build and verify each book subtree
    for i in range(4):
        await verify_book(evaluator, root, books[i], i + 1)

    # Return structured summary
    return evaluator.get_summary()