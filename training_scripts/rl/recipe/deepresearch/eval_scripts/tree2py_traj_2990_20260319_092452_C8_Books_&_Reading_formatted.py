import asyncio
import logging
import re
from urllib.parse import urlparse
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "books_2025_awards_purchase_info"
TASK_DESCRIPTION = """Identify four distinct fiction books from the 2025 major literary awards season and provide comprehensive purchasing and availability information for each:

1. The book that won the 2025 National Book Award for Fiction
2. The book that won the 2025 Booker Prize
3. The book that won the 2025 Pulitzer Prize for Fiction
4. One book that was a finalist (but did not win) for the 2025 National Book Award for Fiction

For each of the four books, provide the following information:
- Book title
- Author's full name
- Publisher name
- ISBN-13 number
- A direct product page URL from Barnes & Noble (barnesandnoble.com) where the book can be purchased
- Page count
- At least one independent bookstore (name and evidence such as a URL or mention) where the book is available or featured
- Evidence (such as a catalog link or library website reference) that the book is available in at least one of the top three largest US public library systems: New York Public Library, Public Library of Cincinnati and Hamilton County, or Boston Public Library
- Current retail price for at least one format (hardcover, paperback, or ebook)

Additionally, for each book, provide a URL from an authoritative source (such as the official award organization website) that confirms the book's award winner or finalist status.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BookDetail(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    isbn13: Optional[str] = None
    barnes_noble_url: Optional[str] = None
    page_count: Optional[str] = None
    indie_store_name: Optional[str] = None
    indie_store_urls: List[str] = Field(default_factory=list)
    library_evidence_urls: List[str] = Field(default_factory=list)
    price_text: Optional[str] = None
    price_source_urls: List[str] = Field(default_factory=list)
    award_source_url: Optional[str] = None


class BooksExtraction(BaseModel):
    nba_winner: Optional[BookDetail] = None
    booker_winner: Optional[BookDetail] = None
    pulitzer_winner: Optional[BookDetail] = None
    nba_finalist: Optional[BookDetail] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
Extract the four specific 2025 award-season fiction books mentioned in the answer, with full purchasing and availability details.

Return a JSON object with the following fields. For any field that is not explicitly present in the answer, set it to null (for strings) or [] (for lists). Do not fabricate information.

Fields to extract:
- nba_winner: object with fields:
  - title, author, publisher, isbn13, barnes_noble_url, page_count,
    indie_store_name, indie_store_urls (array of URLs),
    library_evidence_urls (array of URLs),
    price_text (e.g., "$28.00 hardcover", "$14.99 ebook", etc.),
    price_source_urls (array of URLs),
    award_source_url (URL to authoritative confirmation, preferably nationalbook.org)
- booker_winner: same fields; authoritative source preferably thebookerprizes.com
- pulitzer_winner: same fields; authoritative source preferably pulitzer.org
- nba_finalist: same fields; authoritative source preferably nationalbook.org, and must be a finalist, not the winner

Special rules:
- Only include URLs that are explicitly shown in the answer. If a source is referenced without a URL, leave the corresponding URL field null or [].
- For barnes_noble_url, extract a direct product page on barnesandnoble.com (not a search page).
- For indie_store_urls, extract independent bookstore URLs (e.g., bookshop of a local/indie store).
- For library_evidence_urls, extract URLs pointing to catalog/record pages (or dedicated library site pages) from one of: New York Public Library (nypl.org or relevant subdomains like nypl.overdrive.com, nypl.bibliocommons.com), Public Library of Cincinnati and Hamilton County (cincinnatilibrary.org), or Boston Public Library (bpl.org or relevant subdomains like bpl.bibliocommons.com).
- For price_text, extract the specific price string as written in the answer (e.g., "$28.00", "$17.99 paperback").
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _is_isbn13_like(s: Optional[str]) -> bool:
    if not s:
        return False
    digits = re.sub(r"[^0-9Xx]", "", s)
    if len(digits) != 13 or not digits.isdigit():
        return False
    # Optional: checksum validation (EAN-13)
    try:
        total = 0
        for i, ch in enumerate(digits[:12]):
            d = int(ch)
            total += d if i % 2 == 0 else 3 * d
        check = (10 - (total % 10)) % 10
        return check == int(digits[-1])
    except Exception:
        return True  # Fall back to len/digit-only check if needed
        

def _domain_contains(url: Optional[str], allowed_fragments: List[str]) -> bool:
    if not url:
        return False
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return any(frag.lower() in host for frag in allowed_fragments)


def _has_any_allowed_library_domain(urls: List[str]) -> bool:
    allowed_frags = [
        "nypl.org", "nypl.overdrive.com", "nypl.bibliocommons.com",
        "cincinnatilibrary.org",
        "bpl.org", "bpl.overdrive.com", "bpl.bibliocommons.com"
    ]
    return any(_domain_contains(u, allowed_frags) for u in urls)


def _dedup_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _sources_for(*url_lists: List[str | None]) -> List[str]:
    flat: List[Optional[str]] = []
    for lst in url_lists:
        if isinstance(lst, list):
            flat.extend(lst)
        else:
            flat.append(lst)  # type: ignore
    return _dedup_urls(flat)


# --------------------------------------------------------------------------- #
# Verification logic per book                                                 #
# --------------------------------------------------------------------------- #
async def verify_award_book(
    evaluator: Evaluator,
    parent_node,
    book: BookDetail,
    category_id: str,
    award_name: str,
    expect_status: str,  # "winner" or "finalist"
    authority_domains: List[str],
) -> None:
    """
    Build verification subtree for one award book.
    """
    # Create the book-level node (sequential: identify first, then details)
    book_node = evaluator.add_sequential(
        id=f"{category_id}",
        desc=f"{award_name} - {expect_status.capitalize()} verification and details",
        parent=parent_node,
        critical=False,
    )

    # ---------------- Identification (Critical group) ---------------- #
    ident_node = evaluator.add_parallel(
        id=f"{category_id}_identification",
        desc=f"Correct identification for {award_name} {expect_status}",
        parent=book_node,
        critical=True
    )

    # 1) Award source URL provided and on correct domain (Critical)
    src_domain_ok = evaluator.add_custom_node(
        result=bool(book.award_source_url) and _domain_contains(book.award_source_url, authority_domains),
        id=f"{category_id}_award_source_url",
        desc=f"Provide authoritative URL confirming the 2025 {award_name} {expect_status}",
        parent=ident_node,
        critical=True
    )

    # 2) Award status supported by that page (Critical)
    award_check_node = evaluator.add_leaf(
        id=f"{category_id}_award_status_supported",
        desc=f"The 2025 {award_name} {expect_status} is correctly identified (title/author) per the authoritative page",
        parent=ident_node,
        critical=True
    )
    ident_title = _norm(book.title)
    ident_author = _norm(book.author)

    if expect_status == "winner":
        claim_ident = (
            f"According to the page, the 2025 {award_name} winner for Fiction is '{ident_title}' by {ident_author}."
        )
        add_ins = (
            "Confirm the page explicitly indicates the 2025 Fiction winner. "
            "Allow minor variations in punctuation or capitalization of the title and author. "
            "Prefer exact match for the 2025 cycle."
        )
    else:
        claim_ident = (
            f"According to the page, '{ident_title}' by {ident_author} is a 2025 {award_name} Fiction finalist, "
            f"and it is not the winner."
        )
        add_ins = (
            "Confirm the page explicitly lists this book as a 2025 Fiction finalist (shortlisted) and it is not labeled as the winner. "
            "If the page indicates 'Winner' for this title, this claim is not supported."
        )

    await evaluator.verify(
        claim=claim_ident,
        node=award_check_node,
        sources=book.award_source_url,
        additional_instruction=add_ins
    )

    # ---------------- Book details (Parallel, partial credit allowed) ---------------- #
    details_node = evaluator.add_parallel(
        id=f"{category_id}_details",
        desc=f"Provide complete and accurate details for the {award_name} {expect_status}",
        parent=book_node,
        critical=False
    )

    # Title (Critical)
    title_node = evaluator.add_leaf(
        id=f"{category_id}_book_title",
        desc="Provide the title of the book (supported by sources)",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The book title is '{_norm(book.title)}'.",
        node=title_node,
        sources=_sources_for([_norm(book.barnes_noble_url)], [book.award_source_url]),
        additional_instruction="Verify the exact or near-exact title appears on the provided page(s). Minor punctuation/case variations acceptable."
    )

    # Author (Critical)
    author_node = evaluator.add_leaf(
        id=f"{category_id}_author_name",
        desc="Provide the full name of the book's author (supported by sources)",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The author of '{_norm(book.title)}' is '{_norm(book.author)}'.",
        node=author_node,
        sources=_sources_for([_norm(book.barnes_noble_url)], [book.award_source_url]),
        additional_instruction="Confirm the author's name on the page. Allow minor name variants (e.g., middle initials)."
    )

    # Publisher (Critical)
    publisher_node = evaluator.add_leaf(
        id=f"{category_id}_publisher",
        desc="Provide the name of the book's publisher",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publisher of '{_norm(book.title)}' is '{_norm(book.publisher)}'.",
        node=publisher_node,
        sources=_sources_for([_norm(book.barnes_noble_url)]),
        additional_instruction="Verify the listed publisher matches the product detail on the page."
    )

    # ISBN-13 (Critical): format provided + supported by BN page
    isbn_format_node = evaluator.add_custom_node(
        result=_is_isbn13_like(book.isbn13),
        id=f"{category_id}_isbn13_format_valid",
        desc="Provide a valid ISBN-13 format (13 digits; hyphens allowed)",
        parent=details_node,
        critical=True
    )
    isbn_supported_node = evaluator.add_leaf(
        id=f"{category_id}_isbn13_supported",
        desc="ISBN-13 is supported by a product or authoritative page",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ISBN-13 of '{_norm(book.title)}' is '{_norm(book.isbn13)}'.",
        node=isbn_supported_node,
        sources=_sources_for([_norm(book.barnes_noble_url)]),
        additional_instruction="Confirm the exact ISBN-13 string appears on the page. Ignore hyphen differences."
    )

    # Barnes & Noble product page URL (Critical): provided + matches title
    bn_url_valid_node = evaluator.add_custom_node(
        result=bool(_norm(book.barnes_noble_url)) and _domain_contains(book.barnes_noble_url, ["barnesandnoble.com"]),
        id=f"{category_id}_bn_url_valid",
        desc="A direct Barnes & Noble product page URL is provided (barnesandnoble.com)",
        parent=details_node,
        critical=True
    )
    bn_title_match_node = evaluator.add_leaf(
        id=f"{category_id}_bn_title_match",
        desc="BN product page corresponds to the book with the claimed title/author",
        parent=details_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This Barnes & Noble page is a product page for '{_norm(book.title)}' by '{_norm(book.author)}'.",
        node=bn_title_match_node,
        sources=_norm(book.barnes_noble_url),
        additional_instruction="Check the page title/details match the book; confirm it is a product page (e.g., purchase options visible)."
    )

    # Page count (Non-critical)
    if _norm(book.page_count):
        page_count_node = evaluator.add_leaf(
            id=f"{category_id}_page_count",
            desc="Provide the page count of the book",
            parent=details_node,
            critical=False
        )
        await evaluator.verify(
            claim=f"The page count of '{_norm(book.title)}' is approximately '{_norm(book.page_count)}' pages.",
            node=page_count_node,
            sources=_sources_for([_norm(book.barnes_noble_url)]),
            additional_instruction="Verify that at least one print format page count matches (allow minor discrepancies across editions)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"{category_id}_page_count_missing",
            desc="Page count is missing",
            parent=details_node,
            critical=False
        )

    # Independent bookstore availability (Non-critical group with gating)
    indie_group = evaluator.add_parallel(
        id=f"{category_id}_indie_store",
        desc="Independent bookstore availability evidence",
        parent=details_node,
        critical=False
    )
    indie_provided = evaluator.add_custom_node(
        result=bool(_norm(book.indie_store_name)) and bool(book.indie_store_urls),
        id=f"{category_id}_indie_provided",
        desc="At least one indie bookstore name and URL provided",
        parent=indie_group,
        critical=True
    )
    indie_supported = evaluator.add_leaf(
        id=f"{category_id}_indie_supported",
        desc="The indie bookstore page lists or features the book",
        parent=indie_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page(s) from '{_norm(book.indie_store_name)}' list or feature the book '{_norm(book.title)}' by '{_norm(book.author)}'.",
        node=indie_supported,
        sources=book.indie_store_urls,
        additional_instruction="Confirm the book appears on the indie bookstore page (listing, product page, event, or feature)."
    )

    # Library availability (Non-critical group with gating)
    library_group = evaluator.add_parallel(
        id=f"{category_id}_library",
        desc="Library availability evidence in specified large systems",
        parent=details_node,
        critical=False
    )
    lib_provided = evaluator.add_custom_node(
        result=bool(book.library_evidence_urls) and _has_any_allowed_library_domain(book.library_evidence_urls),
        id=f"{category_id}_library_urls_provided",
        desc="At least one library evidence URL provided from NYPL, Cincinnati, or BPL",
        parent=library_group,
        critical=True
    )
    lib_supported = evaluator.add_leaf(
        id=f"{category_id}_library_supported",
        desc="Library page shows that the book is available/listed",
        parent=library_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided library catalog or website page shows availability/listing for '{_norm(book.title)}' by '{_norm(book.author)}'.",
        node=lib_supported,
        sources=book.library_evidence_urls,
        additional_instruction="Confirm the title/author appear on the catalog/record or relevant library page."
    )

    # Price (Non-critical group with gating)
    price_group = evaluator.add_parallel(
        id=f"{category_id}_price",
        desc="Current retail price evidence for at least one format",
        parent=details_node,
        critical=False
    )
    price_provided = evaluator.add_custom_node(
        result=bool(_norm(book.price_text)),
        id=f"{category_id}_price_provided",
        desc="A current retail price text is provided",
        parent=price_group,
        critical=True
    )
    price_supported = evaluator.add_leaf(
        id=f"{category_id}_price_supported",
        desc="Provided price is supported by a product page (e.g., BN) or other verifiable source",
        parent=price_group,
        critical=True
    )
    price_sources = _sources_for([_norm(book.barnes_noble_url)], book.price_source_urls)
    await evaluator.verify(
        claim=f"The page shows a current price of '{_norm(book.price_text)}' for at least one format of '{_norm(book.title)}'.",
        node=price_supported,
        sources=price_sources,
        additional_instruction=(
            "Confirm the price string (sale or list) appears on the page for any format (hardcover, paperback, ebook). "
            "Allow minor formatting/currency symbol/spacing variations."
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
    Evaluate an answer for the 2025 awards fiction books purchasing/availability task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent subtrees per book
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

    # Extract structured info
    extracted: BooksExtraction = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction"
    )

    # Root node: Task_Completion (parallel aggregation)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Successfully identify and provide comprehensive verifiable information for four distinct 2025 fiction books from specified categories",
        parent=root,
        critical=False
    )

    # Books distinctness (Critical)
    titles = [
        _norm(extracted.nba_winner.title if extracted.nba_winner else None),
        _norm(extracted.booker_winner.title if extracted.booker_winner else None),
        _norm(extracted.pulitzer_winner.title if extracted.pulitzer_winner else None),
        _norm(extracted.nba_finalist.title if extracted.nba_finalist else None),
    ]
    # Require all four titles present and all distinct
    non_empty = [t for t in titles if t]
    distinct = len(non_empty) == 4 and len(set(map(str.lower, non_empty))) == 4
    evaluator.add_custom_node(
        result=distinct,
        id="Books_Distinctness",
        desc="Verify that all four books are distinct with no duplicates",
        parent=task_node,
        critical=True
    )

    # Verify each award category
    # 1) National Book Award - Winner
    await verify_award_book(
        evaluator=evaluator,
        parent_node=task_node,
        book=extracted.nba_winner or BookDetail(),
        category_id="Book_1_National_Book_Award_Winner",
        award_name="National Book Award for Fiction",
        expect_status="winner",
        authority_domains=["nationalbook.org"]
    )

    # 2) Booker Prize - Winner
    await verify_award_book(
        evaluator=evaluator,
        parent_node=task_node,
        book=extracted.booker_winner or BookDetail(),
        category_id="Book_2_Booker_Prize_Winner",
        award_name="Booker Prize",
        expect_status="winner",
        authority_domains=["thebookerprizes.com", "bookerprizes.com"]
    )

    # 3) Pulitzer Prize - Winner
    await verify_award_book(
        evaluator=evaluator,
        parent_node=task_node,
        book=extracted.pulitzer_winner or BookDetail(),
        category_id="Book_3_Pulitzer_Prize_Winner",
        award_name="Pulitzer Prize for Fiction",
        expect_status="winner",
        authority_domains=["pulitzer.org"]
    )

    # 4) National Book Award - Finalist (not winner)
    await verify_award_book(
        evaluator=evaluator,
        parent_node=task_node,
        book=extracted.nba_finalist or BookDetail(),
        category_id="Book_4_NBA_Finalist",
        award_name="National Book Award for Fiction",
        expect_status="finalist",
        authority_domains=["nationalbook.org"]
    )

    # Return structured evaluation summary
    return evaluator.get_summary()