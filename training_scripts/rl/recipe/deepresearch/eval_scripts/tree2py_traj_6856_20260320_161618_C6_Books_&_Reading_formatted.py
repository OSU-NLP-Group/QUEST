import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "curated_recent_fiction_2024_awards"
TASK_DESCRIPTION = """
I am creating a curated reading list for my book club focused on critically acclaimed recent fiction. Find four books that meet ALL of the following criteria:

1. The book must have won or been a finalist for at least one of these major literary awards in 2024: Pulitzer Prize for Fiction, National Book Award for Fiction, or Goodreads Choice Awards (Fiction or Historical Fiction category)

2. The book must have been published between January 2023 and December 2024

3. The book must have a Goodreads rating of 4.0 or higher

4. The book must have at least 40,000 ratings on Goodreads (showing significant reader engagement)

5. The book must be available in hardcover format

For each book, provide:
- Book title and author name
- The specific award it won or was a finalist for in 2024, with the book's status (winner or finalist)
- Publisher name
- Publication year
- Page count (for hardcover edition)
- ISBN-13
- Current Goodreads rating
- Number of Goodreads ratings
- URL references confirming: (a) award status from official award website, (b) publication details from publisher or major retailer, (c) Goodreads page, (d) hardcover availability
"""

ALLOWED_2024_AWARDS_NOTE = (
    "Allowed 2024 awards: (1) Pulitzer Prize for Fiction (2024); "
    "(2) National Book Award for Fiction (2024); "
    "(3) Goodreads Choice Awards 2024: Fiction or Historical Fiction categories."
)

GOODREADS_MIN_RATING = 4.0
GOODREADS_MIN_RATING_COUNT = 40000
VALID_PUB_YEARS = {2023, 2024}

# --------------------------------------------------------------------------- #
# Pydantic extraction models                                                  #
# --------------------------------------------------------------------------- #
class BookEntry(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None

    award_name: Optional[str] = None
    award_status: Optional[str] = None  # expected "winner" or "finalist"
    award_urls: List[str] = Field(default_factory=list)  # official award site(s)

    publisher: Optional[str] = None
    publication_year: Optional[str] = None  # keep as string; we'll parse to int when validating
    publication_urls: List[str] = Field(default_factory=list)  # publisher or major retailer pages confirming pub details

    page_count: Optional[str] = None  # string to be robust; we'll just verify against source
    page_count_urls: List[str] = Field(default_factory=list)  # publisher/retailer pages with page count detail

    isbn13: Optional[str] = None

    goodreads_rating: Optional[str] = None  # string in case formatting like "4.15"
    goodreads_ratings_count: Optional[str] = None  # string like "45,381"
    goodreads_url: Optional[str] = None  # Goodreads page URL

    hardcover_urls: List[str] = Field(default_factory=list)  # publisher or major retailer page(s) showing hardcover availability


class BooksExtraction(BaseModel):
    books: List[BookEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return f"""
Extract up to four books listed in the answer. For each book, strictly extract the following fields exactly as they appear in the answer (do not invent):

- title: The book title (string)
- author: The author name (string)
- award_name: The specific award name claimed for 2024 (string; e.g., "Pulitzer Prize for Fiction", "National Book Award for Fiction", "Goodreads Choice Awards: Fiction" or "Goodreads Choice Awards: Historical Fiction")
- award_status: The status (string; one of "winner" or "finalist" or "shortlist" if stated; keep lowercase as given)
- award_urls: An array of URL(s) that the answer cites as the official award page(s) confirming the stated status for 2024 (e.g., pulitzer.org, nationalbook.org, or goodreads.com/choiceawards). Include only URLs explicitly present in the answer text.
- publisher: Publisher name (string)
- publication_year: The publication year as mentioned (string, keep as-is; do not normalize)
- publication_urls: An array of URL(s) to a publisher or major retailer page cited in the answer that confirm publication details (publisher and/or year). Include only URLs explicitly present in the answer text.
- page_count: The hardcover page count as stated (string, keep as-is)
- page_count_urls: An array of URL(s) to a publisher or major retailer page cited in the answer that shows the hardcover page count. Include only URLs explicitly present in the answer text.
- isbn13: The ISBN-13 as written, including hyphens if present (string)
- goodreads_rating: The Goodreads average rating value as stated (string, e.g., "4.22")
- goodreads_ratings_count: The Goodreads ratings count as stated (string, e.g., "42,381")
- goodreads_url: The Goodreads book page URL cited in the answer (single URL string)
- hardcover_urls: An array of URL(s) cited in the answer that clearly show hardcover availability from the publisher or a major retailer (amazon.com, barnesandnoble.com, bookshop.org, publisher sites, etc.)

GENERAL RULES:
- Extract only what is explicitly present in the answer; if a field is not provided, set it to null (or empty array for URL lists).
- Keep numbers as strings; do not convert formats.
- For URL fields, include only valid URLs that appear in the answer text. If a URL is missing protocol, prepend http:// as per the framework’s URL extraction rules.
- If the answer provides more than 4 books, extract all then the evaluator will only use the first 4.

Allowed award families for 2024 (for your understanding only; still extract the raw strings from the answer):
{ALLOWED_2024_AWARDS_NOTE}
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(val: Optional[str]) -> str:
    return val or ""

def parse_first_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(20\d{2})", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def strip_non_digits(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"[^\d]", "", s)

def is_valid_isbn13_format(s: Optional[str]) -> bool:
    if not s:
        return False
    digits = strip_non_digits(s)
    if len(digits) != 13:
        return False
    if not digits.isdigit():
        return False
    # Basic prefix check (not strict): ISBN-13 usually starts with 978 or 979
    return digits.startswith("978") or digits.startswith("979")

def rating_count_meets_threshold(raw: Optional[str], threshold: int) -> bool:
    digits = strip_non_digits(raw)
    if not digits:
        return False
    try:
        return int(digits) >= threshold
    except Exception:
        return False

def rating_value_meets_threshold(raw: Optional[str], threshold: float) -> bool:
    if not raw:
        return False
    # Extract a float-like number from the string
    m = re.search(r"(\d+(\.\d+)?)", raw)
    if not m:
        return False
    try:
        return float(m.group(1)) >= threshold
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Verification per-book                                                       #
# --------------------------------------------------------------------------- #
async def verify_one_book(
    evaluator: Evaluator,
    parent_node,
    book: BookEntry,
    book_idx: int,
) -> None:
    bnum = book_idx + 1
    title = _safe_str(book.title)
    author = _safe_str(book.author)
    award_name = _safe_str(book.award_name)
    award_status = _safe_str(book.award_status).lower().strip()
    publisher = _safe_str(book.publisher)
    pub_year_raw = _safe_str(book.publication_year)
    page_count_raw = _safe_str(book.page_count)
    isbn13_raw = _safe_str(book.isbn13)
    gr_url = book.goodreads_url or None

    # Resolve URL groups with fallbacks where sensible
    award_urls = book.award_urls or []
    publication_urls = book.publication_urls or []
    page_count_urls = book.page_count_urls or publication_urls  # fallback to publication URLs if specific page-count URLs missing
    hardcover_urls = book.hardcover_urls or publication_urls  # fallback to publication URLs if specific hardcover URLs missing

    # Container node for this book (non-critical at the book level to allow partial credit across books)
    book_node = evaluator.add_parallel(
        id=f"book_{bnum}",
        desc=f"Book #{bnum} verification - meets award, publication, rating, and format criteria",
        parent=parent_node,
        critical=False
    )

    # 1) Identification (critical)
    ident_node = evaluator.add_parallel(
        id=f"book_{bnum}_identification",
        desc="Verify book title and author name are provided",
        parent=book_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(title.strip()),
        id=f"book_{bnum}_title_provided",
        desc="Book title is provided",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(author.strip()),
        id=f"book_{bnum}_author_provided",
        desc="Author name is provided",
        parent=ident_node,
        critical=True
    )

    # 2) Award verification (critical)
    award_node = evaluator.add_parallel(
        id=f"book_{bnum}_award_verification",
        desc="Verify the book won or was a finalist for a major 2024 literary award",
        parent=book_node,
        critical=True
    )
    # 2.1 Award name validity (simple check against allowed families)
    award_name_leaf = evaluator.add_leaf(
        id=f"book_{bnum}_award_name_valid",
        desc="Award name matches one of the allowed 2024 awards (Pulitzer Fiction, National Book Award Fiction, or Goodreads Choice Awards Fiction/Historical Fiction)",
        parent=award_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The provided award name '{award_name}' corresponds to one of the following 2024 awards: "
            f"Pulitzer Prize for Fiction (2024); National Book Award for Fiction (2024); "
            f"Goodreads Choice Awards 2024 (Fiction or Historical Fiction). "
            f"Allow reasonable phrasing variations."
        ),
        node=award_name_leaf,
        additional_instruction="This is a simple logical/membership check; do not require external sources. Focus on whether the phrasing clearly refers to one of the allowed 2024 awards."
    )
    # 2.2 Award status presence and validity (winner/finalist/shortlist treated as finalist)
    award_status_is_valid = award_status in {"winner", "finalist", "shortlist"}
    evaluator.add_custom_node(
        result=award_status_is_valid,
        id=f"book_{bnum}_award_status_valid",
        desc="Book's award status is clearly stated and is winner/finalist (treat 'shortlist' as finalist equivalent)",
        parent=award_node,
        critical=True
    )
    # 2.3 Award URL presence (official site)
    evaluator.add_custom_node(
        result=len(award_urls) > 0,
        id=f"book_{bnum}_award_url_present",
        desc="At least one official award URL is provided",
        parent=award_node,
        critical=True
    )
    # 2.4 Award status supported by official page(s)
    award_support_leaf = evaluator.add_leaf(
        id=f"book_{bnum}_award_source_support",
        desc="Official award page confirms the book's 2024 award status",
        parent=award_node,
        critical=True
    )
    award_claim = (
        f"The book '{title}' by {author} is a {('finalist' if award_status == 'shortlist' else award_status)} "
        f"for the {award_name} in 2024."
    )
    await evaluator.verify(
        claim=award_claim,
        node=award_support_leaf,
        sources=award_urls,
        additional_instruction=(
            "Verify this claim using only official award sources: pulitzer.org (Pulitzer), nationalbook.org (National Book Award), "
            "or goodreads.com (Goodreads Choice Awards). The page should clearly indicate 2024 and the book's status "
            "(winner or finalist/shortlist). If the URL is not an official site or doesn't confirm the status for 2024, mark as not supported."
        ),
    )

    # 3) Publication details (critical)
    pub_node = evaluator.add_parallel(
        id=f"book_{bnum}_publication_details",
        desc="Verify publication information is complete and accurate",
        parent=book_node,
        critical=True
    )
    # 3.1 Publisher provided
    evaluator.add_custom_node(
        result=bool(publisher.strip()),
        id=f"book_{bnum}_publisher_provided",
        desc="Publisher name is provided",
        parent=pub_node,
        critical=True
    )
    # 3.2 Publisher URL presence
    evaluator.add_custom_node(
        result=len(publication_urls) > 0,
        id=f"book_{bnum}_publisher_url_present",
        desc="URL reference to publisher or major retailer page is provided",
        parent=pub_node,
        critical=True
    )
    # 3.3 Publisher supported by source(s)
    pub_support_leaf = evaluator.add_leaf(
        id=f"book_{bnum}_publisher_source_support",
        desc="Publisher name is confirmed by the publisher or a major retailer page",
        parent=pub_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publisher of the book '{title}' by {author} is '{publisher}'.",
        node=pub_support_leaf,
        sources=publication_urls,
        additional_instruction="Confirm the publisher field for this specific book on the provided page(s). Prefer publisher sites; major retailers (Amazon, Barnes & Noble, Bookshop, etc.) are acceptable if they clearly state the publisher."
    )
    # 3.4 Publication year within 2023–2024 (logical check)
    pub_year = parse_first_year(pub_year_raw)
    evaluator.add_custom_node(
        result=(pub_year in VALID_PUB_YEARS),
        id=f"book_{bnum}_publication_year_in_range",
        desc="Publication year is between 2023–2024 (inclusive)",
        parent=pub_node,
        critical=True
    )
    # 3.5 Page count provided (for hardcover)
    evaluator.add_custom_node(
        result=bool(page_count_raw.strip()),
        id=f"book_{bnum}_page_count_provided",
        desc="Hardcover page count is provided",
        parent=pub_node,
        critical=True
    )
    # 3.6 Page count URL presence
    evaluator.add_custom_node(
        result=len(page_count_urls) > 0,
        id=f"book_{bnum}_page_count_url_present",
        desc="URL reference confirming the hardcover page count is provided",
        parent=pub_node,
        critical=True
    )
    # 3.7 Page count supported by source(s)
    page_support_leaf = evaluator.add_leaf(
        id=f"book_{bnum}_page_count_source_support",
        desc="Hardcover page count is confirmed by the publisher or a major retailer page",
        parent=pub_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The hardcover edition of '{title}' by {author} has {page_count_raw} pages.",
        node=page_support_leaf,
        sources=page_count_urls,
        additional_instruction="Verify the hardcover page count on the provided publisher/retailer page(s). Look for an explicit 'Hardcover' format and its page count."
    )
    # 3.8 ISBN-13 format validity (logical check)
    evaluator.add_custom_node(
        result=is_valid_isbn13_format(isbn13_raw),
        id=f"book_{bnum}_isbn13_format_valid",
        desc="ISBN-13 is provided in a valid 13-digit format (hyphens allowed)",
        parent=pub_node,
        critical=True
    )

    # 4) Rating metrics (critical)
    rating_node = evaluator.add_parallel(
        id=f"book_{bnum}_rating_metrics",
        desc="Verify Goodreads rating meets minimum thresholds",
        parent=book_node,
        critical=True
    )
    # 4.1 Goodreads URL presence
    evaluator.add_custom_node(
        result=bool(gr_url),
        id=f"book_{bnum}_rating_url_present",
        desc="Goodreads URL is provided",
        parent=rating_node,
        critical=True
    )
    # 4.2 Goodreads page validity (optional strengthening of grounding)
    rating_url_leaf = evaluator.add_leaf(
        id=f"book_{bnum}_rating_url_valid",
        desc="Provided URL is a Goodreads page for this specific book",
        parent=rating_node,
        critical=True
    )
    if gr_url:
        await evaluator.verify(
            claim=f"This page is the Goodreads page for the book '{title}' by {author}.",
            node=rating_url_leaf,
            sources=gr_url,
            additional_instruction="Check that the page is on goodreads.com and corresponds to the stated book (title/author). Minor title formatting variations are acceptable."
        )
    else:
        # Still add a verify call; it will be routed without sources and likely fail, but node already exists.
        await evaluator.verify(
            claim=f"This page is the Goodreads page for the book '{title}' by {author}.",
            node=rating_url_leaf,
            sources=None,
            additional_instruction="No URL provided; this should fail."
        )
    # 4.3 Rating score threshold (>= 4.0), grounded on Goodreads page
    rating_score_leaf = evaluator.add_leaf(
        id=f"book_{bnum}_rating_score_threshold",
        desc="Goodreads rating is 4.0 or higher",
        parent=rating_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Goodreads page shows an average rating of at least 4.0 out of 5 for this book.",
        node=rating_score_leaf,
        sources=gr_url if gr_url else None,
        additional_instruction="Use the Goodreads displayed average rating. If it's 3.99 or lower, mark as not supported. Minor rounding differences are allowed."
    )
    # 4.4 Rating count threshold (>= 40,000), grounded on Goodreads page
    rating_count_leaf = evaluator.add_leaf(
        id=f"book_{bnum}_rating_count_threshold",
        desc="Book has at least 40,000 ratings on Goodreads",
        parent=rating_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Goodreads page shows at least 40,000 total ratings for this book.",
        node=rating_count_leaf,
        sources=gr_url if gr_url else None,
        additional_instruction="Look for the 'ratings' count on the Goodreads page and confirm it's >= 40,000. Allow minor formatting differences (commas, abbreviations)."
    )

    # 5) Format availability (hardcover) (critical)
    fmt_node = evaluator.add_parallel(
        id=f"book_{bnum}_format_availability",
        desc="Verify book is available in hardcover format",
        parent=book_node,
        critical=True
    )
    # 5.1 Hardcover URL presence
    evaluator.add_custom_node(
        result=len(hardcover_urls) > 0,
        id=f"book_{bnum}_hardcover_url_present",
        desc="URL reference showing hardcover availability is provided",
        parent=fmt_node,
        critical=True
    )
    # 5.2 Hardcover availability supported by source(s)
    hardcover_leaf = evaluator.add_leaf(
        id=f"book_{bnum}_hardcover_available_supported",
        desc="Provided page confirms hardcover availability",
        parent=fmt_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided page shows a Hardcover format available for '{title}' by {author}.",
        node=hardcover_leaf,
        sources=hardcover_urls if hardcover_urls else None,
        additional_instruction="Confirm that the page indicates 'Hardcover' format (e.g., a format selector, explicitly labeled format, or product detail). Prefer publisher or major retailer pages."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    # Initialize evaluator with a non-critical root to allow partial credit across 4 books
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

    # Record allowed awards info as "ground truth context" (not strict ground truth but useful reference)
    evaluator.add_ground_truth({
        "allowed_awards_2024": [
            "Pulitzer Prize for Fiction (2024)",
            "National Book Award for Fiction (2024)",
            "Goodreads Choice Awards 2024: Fiction",
            "Goodreads Choice Awards 2024: Historical Fiction",
        ],
        "rating_min": GOODREADS_MIN_RATING,
        "rating_count_min": GOODREADS_MIN_RATING_COUNT,
        "valid_publication_years": sorted(list(VALID_PUB_YEARS)),
    })

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction"
    )

    # Take first 4 books or pad with empty entries
    books: List[BookEntry] = (extracted.books or [])[:4]
    while len(books) < 4:
        books.append(BookEntry())

    # Build verification subtrees per book
    tasks = []
    for idx, book in enumerate(books):
        tasks.append(verify_one_book(evaluator, root, book, idx))
    await asyncio.gather(*tasks)

    return evaluator.get_summary()