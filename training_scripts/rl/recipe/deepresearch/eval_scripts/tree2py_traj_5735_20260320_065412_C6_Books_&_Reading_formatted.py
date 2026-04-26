import asyncio
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from urllib.parse import urlparse
from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "book_club_q1_2026_variety"
TASK_DESCRIPTION = (
    "I'm organizing a book club and want to select 3 recently published books (January-March 2026) "
    "that meet different literary criteria to ensure variety and quality. Please help me find:\n\n"
    "1. One book published between January 1, 2026 and March 31, 2026, written by an author who won either "
    "the 2025 National Book Award for Fiction or the 2025 Pulitzer Prize for Fiction.\n\n"
    "2. One book that was selected as the March 2026 pick for either Oprah's Book Club or Reese's Book Club. "
    "This book must also have been published between January 1, 2026 and March 31, 2026.\n\n"
    "3. One book published between January 1, 2026 and March 31, 2026, whose author is scheduled to appear at "
    "either the Virginia Festival of the Book (March 20-22, 2026 in Charlottesville, Virginia) or the New Orleans "
    "Book Festival (March 12-15, 2026), AND who also has a scheduled author event at an independent bookstore in "
    "New York City during March 2026.\n\n"
    "For each of the 3 books, please provide:\n"
    "- Title\n- Author's full name\n- Exact publication date\n- Publisher name\n- A brief description (2-3 sentences) of the book\n"
    "- Links to purchase the book from at least 2 different online retailers (choose from: Bookshop.org, Amazon, Barnes & Noble, "
    "Better World Books, or ThriftBooks)\n"
    "- Reference URLs that verify the book meets the stated criteria (e.g., award announcement, book club selection, festival schedule, "
    "bookstore event page)"
)

JAN_1_2026 = date(2026, 1, 1)
MAR_31_2026 = date(2026, 3, 31)

APPROVED_RETAILERS = [
    "bookshop.org",
    "amazon.com",
    "barnesandnoble.com",
    "betterworldbooks.com",
    "thriftbooks.com",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BookEntry(BaseModel):
    # Core bibliographic fields
    title: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[str] = None  # Keep string to be robust to various formats
    publisher: Optional[str] = None
    description: Optional[str] = None

    # Commerce
    retailer_links: List[str] = Field(default_factory=list)

    # General references
    reference_urls: List[str] = Field(default_factory=list)

    # Book 1 specific
    award_verification_urls: List[str] = Field(default_factory=list)

    # Book 2 specific
    club_verification_urls: List[str] = Field(default_factory=list)

    # Book 3 specific
    festival_name: Optional[str] = None  # "Virginia Festival of the Book" or "New Orleans Book Festival"
    festival_verification_urls: List[str] = Field(default_factory=list)
    nyc_bookstore_name: Optional[str] = None
    nyc_event_urls: List[str] = Field(default_factory=list)


class BooksExtraction(BaseModel):
    # Expect three books mapped to their required criteria
    book1: Optional[BookEntry] = None  # Award-winner author book
    book2: Optional[BookEntry] = None  # March 2026 Oprah/Reese book club pick
    book3: Optional[BookEntry] = None  # Festival appearance + NYC indie bookstore event


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
Extract exactly three books (book1, book2, book3) from the answer, each satisfying the specified criterion below. 
For each book, extract all requested bibliographic data, purchase links, and verification references.

General fields for every book (book1, book2, book3):
- title: The exact book title as stated in the answer.
- author: The author's full name as stated in the answer.
- publication_date: The exact publication date string as provided (include month, day, and year if present; do not normalize).
- publisher: The publisher name.
- description: A brief 2–3 sentence description from the answer if provided (do not invent).
- retailer_links: A list of all purchase links mentioned for approved retailers only. Allowed domains:
  bookshop.org, amazon.com, barnesandnoble.com, betterworldbooks.com, thriftbooks.com. 
  Include full URLs. Exclude any other domains.
- reference_urls: A list of additional references that support factual details about the book (e.g., publisher page, press coverage, catalog entries).
  Include only URLs explicitly present in the answer.

Book-specific fields:
- For book1 (award-winner author book):
  - award_verification_urls: URLs that directly confirm the author won either the 2025 National Book Award for Fiction 
    or the 2025 Pulitzer Prize for Fiction. Include only URLs explicitly present in the answer. If none are present, return an empty list.

- For book2 (celebrity book club pick):
  - club_verification_urls: URLs that directly confirm the book was the March 2026 pick for either Oprah's Book Club or Reese's Book Club.
    Include only URLs explicitly present in the answer. If none are present, return an empty list.

- For book3 (festival + NYC indie bookstore event):
  - festival_name: One of: "Virginia Festival of the Book" or "New Orleans Book Festival" (as mentioned in the answer). If neither is stated, set to null.
  - festival_verification_urls: URLs that directly show the author’s scheduled appearance at that festival in 2026. If none, return an empty list.
  - nyc_bookstore_name: Name of the NYC independent bookstore hosting the author event (as stated). If not clearly provided, set to null.
  - nyc_event_urls: URLs that show the NYC bookstore event scheduled in March 2026. If none, return an empty list.

Rules:
- Do not invent data. If a field is not provided in the answer, set it to null (for scalars) or [] (for lists).
- For URL fields, return only valid and complete URLs explicitly present in the answer text (recognize plain links or markdown links).
- Deduplicate URLs while preserving order of first appearance.
- Do not include shortened URLs unless explicitly provided.

Return a JSON object strictly matching this schema:
{
  "book1": { ... BookEntry fields ... },
  "book2": { ... BookEntry fields ... },
  "book3": { ... BookEntry fields ... }
}
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_list(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items or []:
        if it and it not in seen:
            out.append(it)
            seen.add(it)
    return out


def combined_sources(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        merged.extend(lst or [])
    return _dedup_list(merged)


def parse_date_str(dt_str: Optional[str]) -> Optional[date]:
    if not dt_str:
        return None
    candidates = [
        "%B %d, %Y",     # January 15, 2026
        "%b %d, %Y",     # Jan 15, 2026
        "%Y-%m-%d",      # 2026-01-15
        "%m/%d/%Y",      # 01/15/2026
        "%d %B %Y",      # 15 January 2026
        "%d %b %Y",      # 15 Jan 2026
        "%B %d %Y",      # January 15 2026
        "%b %d %Y",      # Jan 15 2026
    ]
    s = dt_str.strip()
    for fmt in candidates:
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    # Quick heuristic: if it looks like ISO without zero-padded month/day or other common variants
    # Fallback: try to extract a numeric date "YYYY-MM-DD" loosely
    try:
        if "," in s:
            s2 = s.replace(",", "")
            # e.g., March 1 2026
            for fmt in ["%B %d %Y", "%b %d %Y"]:
                try:
                    return datetime.strptime(s2, fmt).date()
                except Exception:
                    pass
    except Exception:
        pass
    return None


def has_day_component(dt_str: Optional[str]) -> bool:
    d = parse_date_str(dt_str)
    return d is not None


def domain_of(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return None


def retailer_is_approved(url: Optional[str]) -> bool:
    d = domain_of(url)
    if not d:
        return False
    return any(d == dom or d.endswith("." + dom) for dom in APPROVED_RETAILERS)


def purchase_claim(title: Optional[str], author: Optional[str]) -> str:
    t = title or "the specified book"
    a = author or "the specified author"
    return (
        f"This webpage is a valid purchase or pre-order page for the book titled '{t}' by {a}. "
        f"The retailer must be one of: Bookshop.org, Amazon, Barnes & Noble, Better World Books, or ThriftBooks."
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_essential_info(evaluator: Evaluator, parent, book: BookEntry, book_idx: int, label: str):
    """
    Build the 'Essential Information' block:
    - All children are marked critical to comply with framework constraint for a critical parent
    """
    node = evaluator.add_parallel(
        id=f"book_{book_idx}_essential_information",
        desc=f"{label}: Complete bibliographic information for the book",
        parent=parent,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(book.title and book.title.strip()),
        id=f"book_{book_idx}_title_provided",
        desc="Book title is provided",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(book.author and book.author.strip()),
        id=f"book_{book_idx}_author_name_provided",
        desc="Author's full name is provided",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=has_day_component(book.publication_date),
        id=f"book_{book_idx}_exact_publication_date",
        desc="Exact publication date (not just month/year) is provided",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(book.publisher and book.publisher.strip()),
        id=f"book_{book_idx}_publisher_provided",
        desc="Publisher name is provided",
        parent=node,
        critical=True
    )

    # Even though JSON marks description as non-critical, framework requires all children to be critical
    # when parent is critical. We promote this to critical to satisfy constraints.
    evaluator.add_custom_node(
        result=bool(book.description and len([s for s in (book.description or "").replace('!', '.').replace('?', '.').split('.') if s.strip()]) >= 2),
        id=f"book_{book_idx}_description_provided",
        desc="Brief description (2-3 sentences) is provided",
        parent=node,
        critical=True
    )


async def verify_publication_in_range(evaluator: Evaluator, parent, book: BookEntry, book_idx: int, label: str):
    node = evaluator.add_leaf(
        id=f"book_{book_idx}_pubdate_range",
        desc=f"{label}: Book published between January 1, 2026 and March 31, 2026",
        parent=parent,
        critical=True
    )
    claim = (
        f"The book '{book.title or ''}' by {book.author or ''} was published on {book.publication_date or 'an unknown date'}, "
        f"and this date falls between January 1, 2026 and March 31, 2026 (inclusive)."
    )
    urls = combined_sources(book.reference_urls, book.retailer_links)
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls,
        additional_instruction=(
            "Verify the exact publication date shown on any provided page and confirm it is within Jan 1–Mar 31, 2026. "
            "If multiple formats appear (hardcover, paperback), any qualifying format in this window counts."
        ),
    )


async def verify_purchase_links(evaluator: Evaluator, parent, book: BookEntry, book_idx: int, label: str):
    """
    Sequential block: first retailer then second retailer (must be a different approved retailer).
    """
    node = evaluator.add_sequential(
        id=f"book_{book_idx}_purchasing_options",
        desc=f"{label}: Valid purchasing links from at least 2 different retailers",
        parent=parent,
        critical=True
    )

    links = [u for u in (book.retailer_links or []) if u]
    first_url = links[0] if len(links) >= 1 else None
    second_url = links[1] if len(links) >= 2 else None

    # First retailer link
    first_leaf = evaluator.add_leaf(
        id=f"book_{book_idx}_first_retailer",
        desc="Working purchase link from an approved retailer (Bookshop.org, Amazon, Barnes & Noble, Better World Books, or ThriftBooks)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=purchase_claim(book.title, book.author),
        node=first_leaf,
        sources=first_url,
        additional_instruction=(
            "Pass only if: (a) the URL belongs to one of the approved domains "
            f"{APPROVED_RETAILERS}, allowing any subdomain; and (b) the page clearly sells or allows pre-order for the specified book "
            "(accept hardcover/paperback/ebook/audiobook variants). Allow minor title/author formatting differences."
        ),
    )

    # Second retailer link
    second_leaf = evaluator.add_leaf(
        id=f"book_{book_idx}_second_retailer",
        desc="Working purchase link from a different approved retailer than the first",
        parent=node,
        critical=True
    )
    first_dom = domain_of(first_url) or "unknown"
    await evaluator.verify(
        claim=purchase_claim(book.title, book.author),
        node=second_leaf,
        sources=second_url,
        additional_instruction=(
            "Pass only if: (a) the URL belongs to one of the approved domains "
            f"{APPROVED_RETAILERS}, allowing any subdomain; (b) the page clearly sells or allows pre-order for the specified book; and "
            f"(c) this retailer's domain must be different from the first retailer's domain ({first_dom})."
        ),
    )


# -------------------------- Book 1 (Award Winner) -------------------------- #
async def verify_book1_award_winner(evaluator: Evaluator, root, book: Optional[BookEntry]):
    label = "First book: published January-March 2026 by a 2025 major fiction award winner"
    parent = evaluator.add_parallel(
        id="book_1_award_winner",
        desc=label,
        parent=root,
        critical=False
    )
    book = book or BookEntry()

    # Award qualification (critical)
    award_leaf = evaluator.add_leaf(
        id="award_winner_qualification",
        desc="Author won 2025 National Book Award for Fiction or 2025 Pulitzer Prize for Fiction",
        parent=parent,
        critical=True
    )
    award_claim = (
        f"The author {book.author or ''} won either the 2025 National Book Award for Fiction "
        f"or the 2025 Pulitzer Prize for Fiction."
    )
    award_sources = book.award_verification_urls or book.reference_urls
    await evaluator.verify(
        claim=award_claim,
        node=award_leaf,
        sources=award_sources,
        additional_instruction=(
            "Confirm the author is the WINNER (not longlist, finalist, nominee) of either the 2025 National Book Award for Fiction "
            "or the 2025 Pulitzer Prize for Fiction. The verification page should clearly indicate a WIN for the 2025 cycle and the Fiction category."
        ),
    )

    # Publication date in range (critical)
    await verify_publication_in_range(evaluator, parent, book, 1, label="Book published between January 1, 2026 and March 31, 2026")

    # Essential info (critical block)
    await verify_essential_info(evaluator, parent, book, 1, label="Book 1 Essential Information")

    # Purchasing options (critical sequential)
    await verify_purchase_links(evaluator, parent, book, 1, label="Book 1 Purchasing Options")

    # Reference URLs group (non-critical parent to allow mixed strictness)
    ref_parent = evaluator.add_parallel(
        id="book_1_reference_urls",
        desc="Book 1: Reference URLs verifying stated criteria",
        parent=parent,
        critical=False
    )

    # Award verification URL (critical leaf within the block)
    award_url_leaf = evaluator.add_leaf(
        id="book_1_award_verification_url",
        desc="Reference URL provided verifying the author's 2025 award win",
        parent=ref_parent,
        critical=True
    )
    await evaluator.verify(
        claim="This URL confirms the author is the 2025 winner of either the National Book Award for Fiction or the Pulitzer Prize for Fiction.",
        node=award_url_leaf,
        sources=(book.award_verification_urls[0] if book.award_verification_urls else None),
        additional_instruction="The page must explicitly state the author is the 2025 Fiction winner. Other years or categories do not count.",
    )

    # Additional verification URLs (non-critical)
    addl_refs_leaf = evaluator.add_leaf(
        id="book_1_additional_verification_urls",
        desc="Additional reference URLs as needed to verify book details and criteria",
        parent=ref_parent,
        critical=False
    )
    await evaluator.verify(
        claim="At least one of these URLs is a credible source about the book (publisher, retailer, catalog, or trusted media) that supports details like title, author, publisher, or publication date.",
        node=addl_refs_leaf,
        sources=book.reference_urls,
        additional_instruction="If the list is empty or irrelevant, mark as Incorrect.",
    )


# ----------------------- Book 2 (Celebrity Book Club) ---------------------- #
async def verify_book2_celebrity_club(evaluator: Evaluator, root, book: Optional[BookEntry]):
    label = "Second book: March 2026 Oprah's or Reese's Book Club selection (published Jan–Mar 2026)"
    parent = evaluator.add_parallel(
        id="book_2_celebrity_book_club",
        desc=label,
        parent=root,
        critical=False
    )
    book = book or BookEntry()

    # Book club selection status (critical)
    club_leaf = evaluator.add_leaf(
        id="book_2_book_club_status",
        desc="Book was selected as March 2026 pick for Oprah's Book Club or Reese's Book Club",
        parent=parent,
        critical=True
    )
    club_claim = (
        "This book was the March 2026 pick for either Oprah's Book Club or Reese's Book Club."
    )
    club_sources = book.club_verification_urls or book.reference_urls
    await evaluator.verify(
        claim=club_claim,
        node=club_leaf,
        sources=club_sources,
        additional_instruction="The page should clearly indicate the book is the 'March 2026' selection for either Oprah's Book Club or Reese's Book Club.",
    )

    # Publication date in range (critical)
    await verify_publication_in_range(evaluator, parent, book, 2, label="Book published between January 1, 2026 and March 31, 2026")

    # Essential info (critical block)
    await verify_essential_info(evaluator, parent, book, 2, label="Book 2 Essential Information")

    # Purchasing options (critical sequential)
    await verify_purchase_links(evaluator, parent, book, 2, label="Book 2 Purchasing Options")

    # Reference URLs group (non-critical parent)
    ref_parent = evaluator.add_parallel(
        id="book_2_reference_urls",
        desc="Book 2: Reference URLs verifying stated criteria",
        parent=parent,
        critical=False
    )

    club_url_leaf = evaluator.add_leaf(
        id="book_2_book_club_verification_url",
        desc="Reference URL provided verifying the March 2026 book club selection",
        parent=ref_parent,
        critical=True
    )
    await evaluator.verify(
        claim="This URL confirms the book is the March 2026 pick for either Oprah's Book Club or Reese's Book Club.",
        node=club_url_leaf,
        sources=(book.club_verification_urls[0] if book.club_verification_urls else None),
        additional_instruction="Page must mention 'March 2026' and identify the book as the monthly pick.",
    )

    addl_refs_leaf = evaluator.add_leaf(
        id="book_2_additional_verification_urls",
        desc="Additional reference URLs as needed to verify book details and criteria",
        parent=ref_parent,
        critical=False
    )
    await evaluator.verify(
        claim="At least one of these URLs supports bibliographic details (title/author/publisher/publication date) for the book.",
        node=addl_refs_leaf,
        sources=book.reference_urls,
        additional_instruction="If the list is empty or clearly irrelevant, mark as Incorrect.",
    )


# ---------------- Book 3 (Festival + NYC Indie Bookstore Event) ------------ #
async def verify_book3_festival_nyc(evaluator: Evaluator, root, book: Optional[BookEntry]):
    label = "Third book: published Jan–Mar 2026; author at major literary festival AND NYC indie bookstore event in March 2026"
    parent = evaluator.add_parallel(
        id="book_3_festival_nyc_event",
        desc=label,
        parent=root,
        critical=False
    )
    book = book or BookEntry()

    # Festival appearance (critical)
    fest_leaf = evaluator.add_leaf(
        id="book_3_festival_appearance",
        desc="Author scheduled to appear at Virginia Festival of the Book (Mar 20–22, 2026) or New Orleans Book Festival (Mar 12–15, 2026)",
        parent=parent,
        critical=True
    )
    fest_claim = (
        f"The author {book.author or ''} is scheduled to appear at the "
        f"{book.festival_name or 'Virginia Festival of the Book or New Orleans Book Festival'} in 2026."
    )
    await evaluator.verify(
        claim=fest_claim,
        node=fest_leaf,
        sources=book.festival_verification_urls,
        additional_instruction=(
            "Accept only if the festival page or schedule explicitly lists the author as a 2026 participant. "
            "Target festivals: Virginia Festival of the Book (Mar 20–22, 2026) OR New Orleans Book Festival (Mar 12–15, 2026)."
        ),
    )

    # NYC bookstore event (critical)
    nyc_leaf = evaluator.add_leaf(
        id="book_3_nyc_bookstore_event",
        desc="Author has scheduled event at an independent bookstore in New York City during March 2026",
        parent=parent,
        critical=True
    )
    nyc_claim = (
        f"The author {book.author or ''} has a scheduled in-person event in March 2026 at an independent bookstore "
        f"in New York City, such as Greenlight Bookstore, McNally Jackson, Books Are Magic, The Strand, Community Bookstore, "
        f"WORD, Housing Works Bookstore, Three Lives & Company, The Lit. Bar, Astoria Bookshop, Kew & Willow, or Bluestockings."
    )
    await evaluator.verify(
        claim=nyc_claim,
        node=nyc_leaf,
        sources=book.nyc_event_urls,
        additional_instruction=(
            "Confirm the event occurs in March 2026 and is hosted by an independent bookstore within NYC's five boroughs. "
            "Chain stores like Barnes & Noble do NOT count as independent."
        ),
    )

    # Publication date in range (critical)
    await verify_publication_in_range(evaluator, parent, book, 3, label="Book published between January 1, 2026 and March 31, 2026")

    # Essential info (critical block)
    await verify_essential_info(evaluator, parent, book, 3, label="Book 3 Essential Information")

    # Purchasing options (critical sequential)
    await verify_purchase_links(evaluator, parent, book, 3, label="Book 3 Purchasing Options")

    # Reference URLs (non-critical parent)
    ref_parent = evaluator.add_parallel(
        id="book_3_reference_urls",
        desc="Book 3: Reference URLs verifying stated criteria",
        parent=parent,
        critical=False
    )

    # Festival verification URL (critical inside this block)
    fest_url_leaf = evaluator.add_leaf(
        id="book_3_festival_verification_url",
        desc="Reference URL provided verifying the festival appearance",
        parent=ref_parent,
        critical=True
    )
    await evaluator.verify(
        claim="This URL confirms the author is a 2026 participant at the specified festival.",
        node=fest_url_leaf,
        sources=(book.festival_verification_urls[0] if book.festival_verification_urls else None),
        additional_instruction="The page should list the author on the official festival site or a credible festival schedule page.",
    )

    # NYC event verification URL (critical inside this block)
    nyc_url_leaf = evaluator.add_leaf(
        id="book_3_nyc_event_verification_url",
        desc="Reference URL provided verifying the NYC bookstore event",
        parent=ref_parent,
        critical=True
    )
    await evaluator.verify(
        claim="This URL confirms a March 2026 author event at an independent bookstore located in New York City.",
        node=nyc_url_leaf,
        sources=(book.nyc_event_urls[0] if book.nyc_event_urls else None),
        additional_instruction="Event page should show date in March 2026 and indicate the venue is an independent bookstore in NYC.",
    )

    # Additional references (non-critical)
    addl_refs_leaf = evaluator.add_leaf(
        id="book_3_additional_verification_urls",
        desc="Additional reference URLs as needed to verify book details and criteria",
        parent=ref_parent,
        critical=False
    )
    await evaluator.verify(
        claim="At least one of these URLs supports core bibliographic or availability details for the book.",
        node=addl_refs_leaf,
        sources=book.reference_urls,
        additional_instruction="If empty or irrelevant, mark as Incorrect.",
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
    Evaluate an answer against the 'book_club_q1_2026_variety' rubric using obj_task_eval.
    """
    # Initialize evaluator and root node
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Three independent book tracks
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

    # Extract structured info
    extracted: BooksExtraction = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction"
    )

    # Add helpful custom info
    evaluator.add_custom_info(
        info={"approved_retailers": APPROVED_RETAILERS},
        info_type="policy",
        info_name="approved_retailers_policy"
    )

    # Build verification trees for each book
    await verify_book1_award_winner(evaluator, root, extracted.book1)
    await verify_book2_celebrity_club(evaluator, root, extracted.book2)
    await verify_book3_festival_nyc(evaluator, root, extracted.book3)

    # Return evaluation summary
    return evaluator.get_summary()