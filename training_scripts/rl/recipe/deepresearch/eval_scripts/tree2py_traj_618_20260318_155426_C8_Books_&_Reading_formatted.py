import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "nyt_best_books_2025_awards"
TASK_DESCRIPTION = """
Find three books from The New York Times' '10 Best Books of 2025' list that also won or were finalists/shortlisted for at least one major literary award (Pulitzer Prize, Booker Prize, National Book Award, or Kirkus Prize). For each of the three books, provide the following information: (1) The book's full title and author name, (2) The 13-digit ISBN number for the hardcover edition, (3) The publication date (month and year), (4) The publisher name, (5) The hardcover price, (6) The name of the major literary award(s) the book won or was a finalist/shortlist for and whether it won or was a finalist, (7) A reference URL confirming the book appears on the NYT 10 Best Books of 2025 list, and (8) A reference URL confirming the book's award status. All information must be verifiable through the provided reference URLs.
"""

QUALIFYING_AWARDS_NOTE = "Qualifying awards include Pulitzer Prize (any category), Booker Prize, National Book Award (any category), and the Kirkus Prize."


# ----------------------------- Data Models --------------------------------- #
class BookItem(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    isbn13_hardcover: Optional[str] = None
    publication_month_year: Optional[str] = None  # e.g., "March 2025"
    publisher: Optional[str] = None
    hardcover_price: Optional[str] = None  # e.g., "$28.99"
    nyt_list_url: Optional[str] = None
    award_name: Optional[str] = None
    award_status: Optional[str] = None  # e.g., "winner", "finalist", "shortlist", "longlist"
    award_url: Optional[str] = None
    metadata_urls: List[str] = Field(default_factory=list)  # optional extra URLs that can verify ISBN/publisher/price/date


class BooksExtraction(BaseModel):
    books: List[BookItem] = Field(default_factory=list)


# --------------------------- Extraction Prompt ----------------------------- #
def prompt_extract_books() -> str:
    return """
    Extract every book mentioned in the answer. For each, return the following fields:

    - title: Full official book title exactly as stated in the answer (string).
    - author: Full author name as stated (string).
    - isbn13_hardcover: The 13-digit ISBN for the hardcover edition as stated (keep as string; allow hyphens if present).
    - publication_month_year: Publication month and year for the hardcover edition as a string like "March 2025". If only a full date is given, convert to "Month YYYY". If unknown, return null.
    - publisher: The publisher name (string).
    - hardcover_price: Hardcover price string as stated (e.g., "$28.99" or "US$28.99"). If unknown, return null.
    - nyt_list_url: A URL provided in the answer that specifically confirms the book appears on The New York Times “The 10 Best Books of 2025” list. Must be a valid full URL; if not provided, return null.
    - award_name: Name of at least one major award (if any) as stated (e.g., "Pulitzer Prize for Fiction", "Booker Prize", "National Book Award", "Kirkus Prize"). If not stated, return null.
    - award_status: The status as stated (e.g., "winner", "finalist", "shortlist", or "longlist"). If not stated, return null.
    - award_url: A URL that confirms the award and status for this book. Must be a valid full URL; if not provided, return null.
    - metadata_urls: An array of additional URLs (publisher pages, major retailer pages, WorldCat/LoC records, etc.) that could verify ISBN-13, publisher, publication date, and hardcover price. Only include URLs explicitly present in the answer. May be empty.

    IMPORTANT:
    - Only extract values explicitly present in the answer; do not invent.
    - For all URL fields, return full valid URLs; if missing, return null (for single URL) or [] (for metadata_urls).
    - Do not deduplicate or filter; include all books as they appear in the answer, in order.
    """


# ------------------------------ Helpers ------------------------------------ #
def _norm_key(title: Optional[str], author: Optional[str]) -> str:
    t = (title or "").strip().lower()
    a = (author or "").strip().lower()
    return f"{t}||{a}"


def select_first_three_unique(all_books: List[BookItem]) -> List[BookItem]:
    seen = set()
    selected: List[BookItem] = []
    for b in all_books:
        key = _norm_key(b.title, b.author)
        if key and key not in seen:
            seen.add(key)
            selected.append(b)
        if len(selected) >= 3:
            break
    return selected


def collect_verification_urls(book: BookItem) -> List[str]:
    urls: List[str] = []
    # Prefer metadata URLs for metadata checks
    for u in (book.metadata_urls or []):
        if isinstance(u, str) and u and u not in urls:
            urls.append(u)
    # Add NYT and award as fallback sources
    for u in [book.nyt_list_url, book.award_url]:
        if isinstance(u, str) and u and u not in urls:
            urls.append(u)
    return urls


# ----------------------------- Verification -------------------------------- #
async def verify_book(evaluator: Evaluator, parent_node, book: BookItem, index: int) -> None:
    """
    Build verification leaves for a single book under the given parent node.
    Leaves align with rubric tree leaf nodes and are all critical within the book group.
    """
    group = evaluator.add_parallel(
        id=f"book_{index+1}",
        desc=f"Book {index+1} satisfies NYT-list + award constraints and includes all required fields with verifiable support in the provided URLs",
        parent=parent_node,
        critical=False
    )

    all_urls = collect_verification_urls(book)

    # 1) Title + Author
    leaf = evaluator.add_leaf(
        id=f"book_{index+1}_title_author_exact_and_verifiable",
        desc="Provides the book’s full official title and author name, and the provided URL(s) allow verifying the title/author",
        parent=group,
        critical=True
    )
    claim = (
        f"The provided webpages explicitly confirm a book titled '{book.title or ''}' authored by '{book.author or ''}'. "
        f"Treat as incorrect if either title or author is missing, empty, or not clearly supported by the content."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=all_urls if all_urls else None,
        additional_instruction=(
            "Judge true only if at least one of the provided URLs clearly shows both the book title and the author. "
            "Allow minor casing/diacritics/punctuation differences and subtitle variations. "
            "If no URLs are provided or the pages do not show both title and author for this specific book, judge false."
        )
    )

    # 2) ISBN-13 for Hardcover
    leaf = evaluator.add_leaf(
        id=f"book_{index+1}_isbn_13_hardcover_and_verifiable",
        desc="Provides the 13-digit ISBN for the hardcover edition, and the provided URL(s) allow verifying it is the hardcover ISBN-13",
        parent=group,
        critical=True
    )
    isbn_claim = (
        f"The hardcover 13-digit ISBN for this book is '{book.isbn13_hardcover or ''}', as confirmed by the provided webpages."
    )
    await evaluator.verify(
        claim=isbn_claim,
        node=leaf,
        sources=all_urls if all_urls else None,
        additional_instruction=(
            "Verify that the pages list an ISBN-13 specifically for the hardcover (a.k.a. hardback) edition matching the stated value. "
            "Pages might show multiple ISBNs; the one labeled 'Hardcover' or equivalent must match. "
            "If ISBN is missing/empty, if edition is not hardcover, or if no URLs support this, judge false."
        )
    )

    # 3) Publication date (Month Year) in 2025
    leaf = evaluator.add_leaf(
        id=f"book_{index+1}_publication_date_month_year_2025_and_verifiable",
        desc="Provides publication date as month and year, the date is in 2025, and the provided URL(s) allow verifying the publication month/year",
        parent=group,
        critical=True
    )
    pub_claim = (
        f"The hardcover publication date for this book is '{book.publication_month_year or ''}', and that date is in 2025, "
        f"as confirmed by the provided webpages."
    )
    await evaluator.verify(
        claim=pub_claim,
        node=leaf,
        sources=all_urls if all_urls else None,
        additional_instruction=(
            "Judge true only if at least one provided page explicitly shows the publication month and year for the hardcover edition "
            "and the year is 2025. If only a day-specific date is shown, it must align with a 2025 month/year. "
            "If value is missing/empty, the year is not 2025, or no URL supports it, judge false."
        )
    )

    # 4) Publisher
    leaf = evaluator.add_leaf(
        id=f"book_{index+1}_publisher_name_and_verifiable",
        desc="Provides the publisher name, and the provided URL(s) allow verifying the publisher",
        parent=group,
        critical=True
    )
    publisher_claim = (
        f"The publisher of this book is '{book.publisher or ''}', as confirmed by the provided webpages."
    )
    await evaluator.verify(
        claim=publisher_claim,
        node=leaf,
        sources=all_urls if all_urls else None,
        additional_instruction=(
            "Judge true only if at least one provided page clearly lists the publisher matching the stated value "
            "(allowing minor imprint/parent-company variations). Otherwise judge false."
        )
    )

    # 5) Hardcover price
    leaf = evaluator.add_leaf(
        id=f"book_{index+1}_hardcover_price_and_verifiable",
        desc="Provides the hardcover price, and the provided URL(s) allow verifying the stated price (e.g., retailer or publisher listing)",
        parent=group,
        critical=True
    )
    price_claim = (
        f"The hardcover price (list price/MSRP) is stated as '{book.hardcover_price or ''}', and this is supported by the provided webpages."
    )
    await evaluator.verify(
        claim=price_claim,
        node=leaf,
        sources=all_urls if all_urls else None,
        additional_instruction=(
            "Prefer publisher or major retailer listings. If multiple prices appear, the 'list price' for hardcover must match the stated value. "
            "Allow minor formatting/currency symbol variations. If missing/empty or no URL supports it, judge false."
        )
    )

    # 6) NYT list inclusion
    leaf = evaluator.add_leaf(
        id=f"book_{index+1}_nyt_reference_url_confirms_inclusion",
        desc="Provides a reference URL that confirms this specific book appears on the NYT '10 Best Books of 2025' list",
        parent=group,
        critical=True
    )
    nyt_claim = (
        f"The provided NYT URL confirms that the book '{book.title or ''}' by '{book.author or ''}' appears on The New York Times "
        f"'10 Best Books of 2025' list."
    )
    await evaluator.verify(
        claim=nyt_claim,
        node=leaf,
        sources=book.nyt_list_url if book.nyt_list_url else None,
        additional_instruction=(
            "Judge true only if the page is a New York Times page for 'The 10 Best Books of 2025' (or equivalent phrasing) "
            "and it explicitly lists the specific book. If the URL is missing, not NYT, not that list, or does not show this book, judge false."
        )
    )

    # 7) Award name + status + qualifying + URL confirms
    leaf = evaluator.add_leaf(
        id=f"book_{index+1}_award_name_status_qualifying_and_url_confirms",
        desc="Identifies at least one qualifying award and the status (winner/finalist/shortlist/longlist), and provides a reference URL that confirms the award name and status",
        parent=group,
        critical=True
    )
    award_claim = (
        f"The provided award page confirms that the book '{book.title or ''}' by '{book.author or ''}' is a "
        f"'{book.award_status or ''}' for the '{book.award_name or ''}', and that this award is one of the following major awards: "
        f"Pulitzer Prize, Booker Prize, National Book Award, or Kirkus Prize."
    )
    await evaluator.verify(
        claim=award_claim,
        node=leaf,
        sources=book.award_url if book.award_url else None,
        additional_instruction=(
            f"{QUALIFYING_AWARDS_NOTE} Category variants (e.g., fiction, nonfiction) are acceptable. "
            "Statuses may include winner, finalist, shortlist, or longlist (treat 'shortlist' and 'finalist' as equivalents across prizes if synonyms are used). "
            "Judge true only if the page explicitly ties this exact book (match title/author) to the stated award and status. "
            "If URL is missing or does not confirm both the award and status for this book, judge false."
        )
    )


# ------------------------------ Main Eval ---------------------------------- #
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

    # Extract all books as presented in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction"
    )

    all_books = extracted.books or []
    selected = select_first_three_unique(all_books)

    # Record some helpful info
    evaluator.add_custom_info(
        info={
            "total_books_extracted": len(all_books),
            "selected_books_count": len(selected),
            "selected_titles": [f"{b.title or ''} — {b.author or ''}" for b in selected]
        },
        info_type="selection_info",
        info_name="selection_info"
    )
    evaluator.add_custom_info(
        info={"qualifying_awards_policy": QUALIFYING_AWARDS_NOTE},
        info_type="policy",
        info_name="qualifying_awards_policy"
    )

    # Global item count and uniqueness (critical)
    # We follow the common evaluation policy to select the first 3 unique items for detailed verification.
    # This node passes only if we indeed have 3 unique selections.
    global_ok = (len(selected) == 3)
    evaluator.add_custom_node(
        result=global_ok,
        id="global_item_count_and_uniqueness",
        desc="Response provides three distinct books for evaluation (first three unique items selected).",
        parent=root,
        critical=True
    )

    # Ensure we always evaluate 3 book slots (pad with empty if needed so the tree structure is stable)
    while len(selected) < 3:
        selected.append(BookItem())

    # Build per-book verification groups
    for i in range(3):
        await verify_book(evaluator, root, selected[i], i)

    return evaluator.get_summary()