import asyncio
import logging
import re
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "books_h1_2026"
TASK_DESCRIPTION = (
    "I am planning my reading list for the first half of 2026 and want to focus on substantial works from established "
    "authors published by major traditional publishers. Please identify two books that meet all of the following requirements:\n\n"
    "1. The book must have a release date between January 1, 2026 and June 30, 2026\n"
    "2. The book must be published by Ballantine Books, Grand Central Publishing, Knopf, or another major imprint of Penguin Random House\n"
    "3. The book must be available in hardcover format with at least 400 pages\n"
    "4. The book must have a valid ISBN-13 number for the hardcover edition\n"
    "5. The author must have published at least one previous book (not a debut author's first work)\n\n"
    "For each book, please provide:\n"
    "- Book title and author name\n"
    "- Publisher name\n"
    "- Exact release date\n"
    "- Page count of the hardcover edition\n"
    "- ISBN-13 of the hardcover edition\n"
    "- A reference URL to the publisher's official page or a major book retailer (Amazon or Barnes & Noble) confirming the publication details\n"
    "- Evidence that the author has previously published at least one other book, with a supporting reference URL"
)

# Allowed reference URL domains (publisher official pages or major retailers)
ALLOWED_REF_DOMAINS = [
    # Major retailers
    "amazon.com",
    "barnesandnoble.com",
    "bn.com",
    # PRH and common imprints/domains
    "penguinrandomhouse.com",
    "prh.com",
    "randomhousebooks.com",
    "knopf.com",
    "doubleday.com",
    "vintagebooks.com",
    "crownpublishing.com",
    "vikingbooks.com",
    "riverheadbooks.com",
    "putnam.com",
    "duttonbooks.com",
    "delreybooks.com",
    "penguin.com",
    "us.penguingroup.com",
    "ballantinebooks.com",
    # Grand Central (Hachette) family
    "grandcentralpublishing.com",
    "hachettebookgroup.com",
    "littlebrown.com",
]

# Allowed publisher or imprint names (string match, case-insensitive; minor variants acceptable)
ALLOWED_PUBLISHER_NAMES = [
    # Explicitly listed
    "Ballantine Books",
    "Grand Central Publishing",
    "Knopf",
    "Alfred A. Knopf",
    # Common major PRH imprints
    "Random House",
    "Doubleday",
    "Crown",
    "Viking",
    "Riverhead Books",
    "G.P. Putnam's Sons",
    "Putnam",
    "Dutton",
    "Del Rey",
    "Penguin Press",
    "Penguin Books",
    "Vintage",
    "Everyman's Library",
    "Harmony",
    "Ten Speed Press",
    "Portfolio",
    "Spiegel & Grau",
]


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class BookItem(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    release_date: Optional[str] = None  # Keep as string; verification will check the page
    hardcover_pages: Optional[str] = None  # Keep as string (e.g., "432 pages")
    isbn13_hardcover: Optional[str] = None
    reference_url: Optional[str] = None  # Publisher official page or Amazon/B&N page
    author_prev_urls: List[str] = Field(default_factory=list)  # Evidence URLs author previously published


class BooksExtraction(BaseModel):
    books: List[BookItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return (
        "Extract up to the first TWO books that the answer claims satisfy the task requirements. For each book, extract:\n"
        "1) title\n"
        "2) author\n"
        "3) publisher\n"
        "4) release_date (the exact on-sale/publication/release date as stated)\n"
        "5) hardcover_pages (the page count for the hardcover edition, as written; do not convert to a number)\n"
        "6) isbn13_hardcover (ISBN-13 for the hardcover edition)\n"
        "7) reference_url (a URL to the publisher's official page or a major retailer page like Amazon or Barnes & Noble that confirms publication details)\n"
        "8) author_prev_urls (an array of 1–3 URLs that show the author has published at least one other book previously; if none provided, return an empty array)\n\n"
        "Return a JSON object with a top-level 'books' array of objects with exactly these fields. If any field is missing for a book, set it to null (or an empty array for 'author_prev_urls'). "
        "Only extract information explicitly present in the answer; do not invent."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    u = url.strip()
    if not u:
        return None
    if not re.match(r"^https?://", u, flags=re.IGNORECASE):
        u = "http://" + u
    return u


def _domain_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        # Strip common www prefix
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return None


def _is_allowed_reference_url(url: Optional[str]) -> bool:
    norm = _normalize_url(url)
    if not norm:
        return False
    domain = _domain_from_url(norm)
    if not domain:
        return False
    return any(domain.endswith(allowed) for allowed in ALLOWED_REF_DOMAINS)


def _first_two_books(extracted: BooksExtraction) -> List[BookItem]:
    items = list(extracted.books or [])
    if len(items) >= 2:
        return items[:2]
    # pad to exactly 2
    while len(items) < 2:
        items.append(BookItem())
    return items


def _allowed_publishers_display() -> str:
    return ", ".join(ALLOWED_PUBLISHER_NAMES)


# --------------------------------------------------------------------------- #
# Verification logic per-book                                                 #
# --------------------------------------------------------------------------- #
async def verify_single_book(
    evaluator: Evaluator,
    parent_node,
    book: BookItem,
    index: int,
) -> None:
    """
    Build verification subtree for a single book.
    This follows the rubric tree:
      - Book_i (parallel, non-critical)
        - Publication_Details (parallel, critical)
            - Release_Date_Verification (leaf, critical)
            - Major_Publisher_Verification (leaf, critical)
            - Publisher_URL_Reference (leaf via custom node, critical)
        - Format_Requirements (parallel, critical)
            - Hardcover_Availability (leaf, critical)
            - Page_Count_Requirement (leaf, critical)
            - ISBN_Documentation (leaf, critical)
        - Author_Status (parallel, critical)
            - Previous_Publication (leaf, critical)
            - Author_Reference_URL (leaf via custom node, critical)
    """
    book_label = f"Book_{index + 1}"
    book_node = evaluator.add_parallel(
        id=book_label,
        desc="First qualifying book meeting all requirements" if index == 0 else "Second qualifying book meeting all requirements",
        parent=parent_node,
        critical=False,
    )

    # -------------------- Publication_Details (critical) -------------------- #
    pub_node = evaluator.add_parallel(
        id=f"{book_label}_Publication_Details",
        desc="Book meets publication timing and publisher requirements",
        parent=book_node,
        critical=True,
    )

    # Publisher_URL_Reference (existence + domain gating)
    ref_ok = _is_allowed_reference_url(book.reference_url)
    ref_node = evaluator.add_custom_node(
        result=ref_ok,
        id=f"{book_label}_Publisher_URL_Reference",
        desc="Publisher's official page or major retailer page (Amazon, Barnes & Noble) confirming publication details is provided",
        parent=pub_node,
        critical=True,
    )

    # Release_Date_Verification
    release_node = evaluator.add_leaf(
        id=f"{book_label}_Release_Date_Verification",
        desc="Book has a 2026 release date between January 1 and June 30, 2026",
        parent=pub_node,
        critical=True,
    )
    release_claim = (
        f"According to the provided page, the release/publication/on-sale date of the hardcover edition is between "
        f"January 1, 2026 and June 30, 2026 inclusive. The answer lists the date as '{book.release_date}'."
    )
    await evaluator.verify(
        claim=release_claim,
        node=release_node,
        sources=_normalize_url(book.reference_url),
        additional_instruction=(
            "Check the page for fields like Publication Date, On Sale Date, or Release Date. "
            "Accept synonyms and minor wording variations. The date must be within 2026-01-01 to 2026-06-30 inclusive. "
            "If multiple formats are listed, consider the hardcover-specific date when available."
        ),
        extra_prerequisites=[ref_node],
    )

    # Major_Publisher_Verification
    publisher_node = evaluator.add_leaf(
        id=f"{book_label}_Major_Publisher_Verification",
        desc="Book is published by Ballantine Books, Grand Central Publishing, Knopf, or another major Penguin Random House imprint",
        parent=pub_node,
        critical=True,
    )
    allowed_display = _allowed_publishers_display()
    publisher_claim = (
        f"On the provided page, the publisher listed for this book is '{book.publisher}'. "
        f"Treat the requirement as satisfied if the page shows the publisher/imprint is one of: {allowed_display}. "
        f"Allow minor case and punctuation variations (e.g., Alfred A. Knopf vs Knopf, Putnam vs G.P. Putnam's Sons)."
    )
    await evaluator.verify(
        claim=publisher_claim,
        node=publisher_node,
        sources=_normalize_url(book.reference_url),
        additional_instruction=(
            "Look specifically at the 'Publisher' or imprint field on the page. "
            "Do not rely on other sources beyond the provided URL. "
            "If the publisher matches any name in the allowed list, count as satisfied."
        ),
        extra_prerequisites=[ref_node],
    )

    # -------------------- Format_Requirements (critical) -------------------- #
    fmt_node = evaluator.add_parallel(
        id=f"{book_label}_Format_Requirements",
        desc="Book meets hardcover format and length specifications",
        parent=book_node,
        critical=True,
    )

    # Hardcover_Availability
    hardcover_node = evaluator.add_leaf(
        id=f"{book_label}_Hardcover_Availability",
        desc="Book is available in hardcover format",
        parent=fmt_node,
        critical=True,
    )
    hardcover_claim = "The provided page explicitly indicates a hardcover edition (e.g., format/binding shows 'Hardcover')."
    await evaluator.verify(
        claim=hardcover_claim,
        node=hardcover_node,
        sources=_normalize_url(book.reference_url),
        additional_instruction=(
            "Check format/binding options or metadata for 'Hardcover'. "
            "On Amazon, look at the format selector; on Barnes & Noble, check 'Format'; on publisher pages, look for 'Hardcover'."
        ),
        extra_prerequisites=[ref_node],
    )

    # Page_Count_Requirement
    pages_node = evaluator.add_leaf(
        id=f"{book_label}_Page_Count_Requirement",
        desc="Hardcover edition has at least 400 pages",
        parent=fmt_node,
        critical=True,
    )
    pages_claim = (
        f"The hardcover edition has at least 400 pages. The answer states the hardcover page count is '{book.hardcover_pages}'."
    )
    await evaluator.verify(
        claim=pages_claim,
        node=pages_node,
        sources=_normalize_url(book.reference_url),
        additional_instruction=(
            "Verify the page count for the hardcover edition specifically. "
            "Accept if the page shows a number >= 400 for the hardcover. "
            "If multiple formats are listed, ensure the count corresponds to hardcover."
        ),
        extra_prerequisites=[ref_node],
    )

    # ISBN_Documentation
    isbn_node = evaluator.add_leaf(
        id=f"{book_label}_ISBN_Documentation",
        desc="Valid ISBN-13 number for the hardcover edition is provided",
        parent=fmt_node,
        critical=True,
    )
    isbn_claim = (
        f"The page lists the hardcover edition's ISBN-13 as '{book.isbn13_hardcover}'. "
        f"Treat hyphenation variations as acceptable, but the digits must match and be 13 digits overall."
    )
    await evaluator.verify(
        claim=isbn_claim,
        node=isbn_node,
        sources=_normalize_url(book.reference_url),
        additional_instruction=(
            "Confirm the 13-digit ISBN for the hardcover edition appears on the page. "
            "Ignore hyphens/spaces; digits should match. If the page lists multiple ISBNs for different formats, "
            "use the one explicitly labeled for hardcover."
        ),
        extra_prerequisites=[ref_node],
    )

    # -------------------- Author_Status (critical) ------------------------- #
    auth_node = evaluator.add_parallel(
        id=f"{book_label}_Author_Status",
        desc="Author has published at least one previous book",
        parent=book_node,
        critical=True,
    )

    # Author_Reference_URL (existence)
    has_author_ref = bool(book.author_prev_urls and len(book.author_prev_urls) > 0)
    author_ref_node = evaluator.add_custom_node(
        result=has_author_ref,
        id=f"{book_label}_Author_Reference_URL",
        desc="Reference URL confirming author's previous publication(s) is provided",
        parent=auth_node,
        critical=True,
    )

    # Previous_Publication
    prev_pub_node = evaluator.add_leaf(
        id=f"{book_label}_Previous_Publication",
        desc="Evidence shows the author has published at least one other book before this 2026 release",
        parent=auth_node,
        critical=True,
    )
    prev_claim = (
        f"The author '{book.author}' has published at least one other book prior to 2026 (i.e., not a debut). "
        f"The provided reference page(s) should show any earlier book by the same author."
    )
    await evaluator.verify(
        claim=prev_claim,
        node=prev_pub_node,
        sources=[_normalize_url(u) for u in (book.author_prev_urls or []) if _normalize_url(u)],
        additional_instruction=(
            "Accept any credible page (publisher, retailer, or author bibliography) that shows a previously published "
            "book by the same author with a publication year earlier than 2026. Minor name variants are acceptable."
        ),
        extra_prerequisites=[author_ref_node],
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 'two qualifying books in H1 2026' task using the Mind2Web2 evaluation framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallels two books
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify two books published in 2026 that meet all specified criteria",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured book info from the answer
    extracted_books = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction",
    )

    # Record task requirement info for transparency
    evaluator.add_custom_info(
        info={
            "date_window_inclusive": ["2026-01-01", "2026-06-30"],
            "allowed_reference_domains": ALLOWED_REF_DOMAINS,
            "allowed_publishers_or_imprints": ALLOWED_PUBLISHER_NAMES,
        },
        info_type="constraints",
        info_name="task_constraints",
    )

    # Take exactly two books (pad if fewer present)
    books = _first_two_books(extracted_books)

    # Build verification subtrees for each book
    for idx, book in enumerate(books):
        await verify_single_book(evaluator, root, book, idx)

    # Return evaluation summary
    return evaluator.get_summary()