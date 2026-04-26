import asyncio
import logging
import re
from datetime import datetime
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "celebrity_cookbook_clarkson_potter_2020_2024"
TASK_DESCRIPTION = (
    "Find a celebrity-authored cookbook or cocktail/drinks recipe book that meets all of the following requirements: "
    "Published by Clarkson Potter (located in New York City), published between January 1, 2020 and December 31, 2024, "
    "became a New York Times bestseller, available in hardcover format, and authored by a celebrity (such as a TV personality or reality TV star). "
    "Provide comprehensive details about this book, including: the complete book title, author's full name, publisher name and location, "
    "exact publication date (month, day, and year), ISBN-13 number, page count, original retail price in USD, physical dimensions, "
    "and confirmation that the book is indeed a cookbook or cocktail book and that all stated requirements are met."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BookExtraction(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    publisher_location: Optional[str] = None
    publication_date: Optional[str] = None  # e.g., "October 12, 2021"
    isbn13: Optional[str] = None
    page_count: Optional[str] = None
    price_usd: Optional[str] = None  # e.g., "$32.00 USD"
    dimensions: Optional[str] = None  # e.g., "8 x 10 inches"
    format: Optional[str] = None  # e.g., "Hardcover"
    nyt_bestseller: Optional[str] = None  # free text the answer used
    genre: Optional[str] = None  # e.g., "Cookbook", "Cocktail", etc.
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_book_info() -> str:
    return """
    Extract the single book identified in the answer that the user intends to present as the solution. Return the following fields exactly as presented in the answer (do not infer or add your own information):
    - title: complete book title
    - author: author's full name
    - publisher: publisher name (e.g., "Clarkson Potter")
    - publisher_location: the location/city of the publisher as stated in the answer (if given)
    - publication_date: the exact publication date (month day, year) if provided in the answer (e.g., "October 12, 2021")
    - isbn13: the ISBN-13 value as shown in the answer (with or without hyphens)
    - page_count: the page count (as shown in the answer)
    - price_usd: the original retail price in USD (e.g., "$32.00", "USD 32.00")
    - dimensions: the physical dimensions of the book (as shown in the answer)
    - format: the format or formats mentioned for the book (e.g., "Hardcover", "Hardcover and Kindle")
    - nyt_bestseller: the text from the answer that claims it is a New York Times bestseller (if present)
    - genre: the category/genre description from the answer (e.g., "cookbook", "cocktail", "drinks recipes")
    - source_urls: ALL URLs explicitly cited in the answer that are relevant to this book (publisher pages, retailer pages like Amazon, Google Books, Penguin Random House, Clarkson Potter, New York Times links, author biography pages, etc.). Extract only actual URLs present in the answer (including markdown links). Do not invent URLs.

    Rules:
    - If any field is missing in the answer, return null for that field (or [] for source_urls).
    - For source_urls, include every unique URL mentioned for this book. Preserve the order they appear in the answer. Do not include duplicates.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _dedupe_urls(urls: List[str], max_urls: int = 12) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u or not isinstance(u, str):
            continue
        u = u.strip()
        if len(u) < 4:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            # Skip obviously invalid or non-http urls (mail, etc.)
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
        if len(out) >= max_urls:
            break
    return out


def _non_empty(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _contains_digits(s: Optional[str]) -> bool:
    if not _non_empty(s):
        return False
    return bool(re.search(r"\d", s or ""))


def _price_in_usd(s: Optional[str]) -> bool:
    if not _non_empty(s):
        return False
    t = (s or "").upper()
    return ("$" in s) or ("USD" in t)


def _normalize_isbn13(s: Optional[str]) -> Optional[str]:
    if not _non_empty(s):
        return None
    digits = re.sub(r"[^0-9]", "", s)
    return digits if digits else None


def _is_valid_isbn13(s: Optional[str]) -> bool:
    digits = _normalize_isbn13(s)
    if not digits or len(digits) != 13 or not digits.isdigit():
        return False
    total = 0
    for i in range(12):
        n = int(digits[i])
        total += n if i % 2 == 0 else 3 * n
    check = (10 - (total % 10)) % 10
    return check == int(digits[12])


def _try_parse_full_date(s: Optional[str]) -> Optional[datetime]:
    """
    Try parsing a full date that includes month, day, and year.
    We intentionally avoid month-year-only formats, since the rubric requires exact date.
    """
    if not _non_empty(s):
        return None
    txt = s.strip()
    fmts = [
        "%B %d, %Y",   # October 12, 2021
        "%b %d, %Y",   # Oct 12, 2021
        "%Y-%m-%d",    # 2021-10-12
        "%m/%d/%Y",    # 10/12/2021
        "%m-%d-%Y",    # 10-12-2021
        "%d %B %Y",    # 12 October 2021
        "%d %b %Y",    # 12 Oct 2021
    ]
    for f in fmts:
        try:
            return datetime.strptime(txt, f)
        except Exception:
            continue
    return None


def _in_required_range(d: Optional[datetime]) -> bool:
    if not d:
        return False
    lo = datetime(2020, 1, 1)
    hi = datetime(2024, 12, 31, 23, 59, 59)
    return lo <= d <= hi


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_celeb_book(
    evaluator: Evaluator,
    parent_node,
    book: BookExtraction,
) -> None:
    """
    Build the verification tree for the celebrity-authored cookbook/cocktail book and execute checks.
    """
    # Prepare commonly used values
    title_for_claim = book.title if _non_empty(book.title) else "the identified book"
    author_for_claim = book.author if _non_empty(book.author) else "the identified author"
    sources_list = _dedupe_urls(book.source_urls)

    # Main critical verification node
    celeb_node = evaluator.add_parallel(
        id="CelebrityBookVerification",
        desc="Verify that the identified book and all required information meet the specified criteria",
        parent=parent_node,
        critical=True,
    )

    # 1) Existence checks (critical)
    evaluator.add_custom_node(
        result=_non_empty(book.title),
        id="BookTitleProvided",
        desc="The complete book title is provided",
        parent=celeb_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_non_empty(book.author),
        id="AuthorNameProvided",
        desc="The author's full name is provided",
        parent=celeb_node,
        critical=True,
    )

    # 2) Publisher and imprint verification (critical, split into 2 concrete leaves)
    pub_imprint_node = evaluator.add_parallel(
        id="PublisherAndImprintVerification",
        desc="The publisher is confirmed to be Clarkson Potter, and Clarkson Potter is confirmed to be an imprint of Crown Publishing Group (part of Penguin Random House)",
        parent=celeb_node,
        critical=True,
    )

    # 2.1) Publisher is Clarkson Potter
    n_publisher = evaluator.add_leaf(
        id="PublisherIsClarksonPotter",
        desc="The book's publisher is Clarkson Potter",
        parent=pub_imprint_node,
        critical=True,
    )
    claim_publisher = f"The book titled '{title_for_claim}' by {author_for_claim} is published by Clarkson Potter."
    add_ins_publisher = (
        "Verify on the provided source pages (e.g., publisher/retailer/PRH pages) that the listed publisher for the "
        f"book '{title_for_claim}' is explicitly Clarkson Potter. Allow minor formatting differences."
    )

    # 2.2) Clarkson Potter imprint relationship
    n_imprint = evaluator.add_leaf(
        id="ImprintRelationshipConfirmed",
        desc="Clarkson Potter is an imprint of Crown Publishing Group (part of Penguin Random House)",
        parent=pub_imprint_node,
        critical=True,
    )
    claim_imprint = (
        "Clarkson Potter is an imprint of Crown Publishing Group, which is part of Penguin Random House."
    )
    add_ins_imprint = (
        "Look for explicit wording like 'imprint of Crown Publishing Group' or 'an imprint of Crown at Penguin Random House'. "
        "Accept equivalent phrasing."
    )

    # 3) Publisher location is NYC (critical)
    n_location = evaluator.add_leaf(
        id="PublisherLocationVerification",
        desc="The publisher's location is confirmed to be New York City",
        parent=celeb_node,
        critical=True,
    )
    claim_location = (
        "Clarkson Potter is located in New York City, New York (NYC). 'New York, NY' should be interpreted as New York City."
    )
    add_ins_location = (
        "Accept 'New York, NY' as New York City. Official PRH/Clarkson Potter pages or other credible sources are valid."
    )

    # 4) Publication date checks (critical)
    parsed_pub_date = _try_parse_full_date(book.publication_date)
    evaluator.add_custom_node(
        result=parsed_pub_date is not None,
        id="ExactPublicationDate",
        desc="The exact publication date (month, day, and year) is provided",
        parent=celeb_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_in_required_range(parsed_pub_date),
        id="PublicationDateRange",
        desc="The publication date falls between January 1, 2020 and December 31, 2024",
        parent=celeb_node,
        critical=True,
    )

    # 5) NYT bestseller status (critical)
    n_nyt = evaluator.add_leaf(
        id="NYTBestsellerStatus",
        desc="The book is confirmed to have achieved New York Times bestseller status",
        parent=celeb_node,
        critical=True,
    )
    claim_nyt = (
        f"The book titled '{title_for_claim}' by {author_for_claim} achieved New York Times bestseller status (at any time)."
    )
    add_ins_nyt = (
        "Look for explicit text such as 'New York Times Bestseller' on the book's page, publisher page, or credible media. "
        "Do not confuse 'New York Times bestselling author' with the book itself; the claim must be about the book."
    )

    # 6) Hardcover format availability (critical)
    n_hardcover = evaluator.add_leaf(
        id="HardcoverFormat",
        desc="The book is confirmed to be available in hardcover format",
        parent=celeb_node,
        critical=True,
    )
    claim_hardcover = (
        f"The book titled '{title_for_claim}' by {author_for_claim} is available in hardcover format."
    )
    add_ins_hardcover = (
        "Verify that the book has a hardcover edition. Accept 'Hardcover' indicated on retailer or publisher pages."
    )

    # 7) Celebrity author status (critical)
    n_celebrity = evaluator.add_leaf(
        id="CelebrityAuthorStatus",
        desc="The author is confirmed to be a celebrity (TV personality, reality TV star, or similar public figure)",
        parent=celeb_node,
        critical=True,
    )
    claim_celebrity = (
        f"The author {author_for_claim} is a celebrity public figure (e.g., a TV personality or reality TV star)."
    )
    add_ins_celebrity = (
        "Check credible sources indicating the author is widely recognized as a public figure (e.g., TV personality, "
        "reality TV star, actor, musician). Wikipedia or major outlet bios are acceptable if provided."
    )

    # 8) Book genre is cookbook or cocktail/drinks (critical)
    n_genre = evaluator.add_leaf(
        id="BookGenreVerification",
        desc="The book is confirmed to be either a cookbook or cocktail/drinks recipe book",
        parent=celeb_node,
        critical=True,
    )
    claim_genre = (
        f"The book titled '{title_for_claim}' is a cookbook or a cocktail/drinks recipe book."
    )
    add_ins_genre = (
        "Accept phrases like 'cookbook', 'recipes', 'cooking', 'cocktails', 'drinks', 'beverages'. "
        "The primary focus must be recipes for food or drinks."
    )

    # 9) Other required info provided (critical existence/format checks)
    evaluator.add_custom_node(
        result=_is_valid_isbn13(book.isbn13),
        id="ISBN13Provided",
        desc="A valid ISBN-13 number is provided",
        parent=celeb_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_contains_digits(book.page_count),
        id="PageCountProvided",
        desc="The total page count is provided",
        parent=celeb_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_price_in_usd(book.price_usd),
        id="RetailPriceProvided",
        desc="The original retail price in USD is provided",
        parent=celeb_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_non_empty(book.dimensions),
        id="BookDimensionsProvided",
        desc="The physical dimensions of the book are provided",
        parent=celeb_node,
        critical=True,
    )

    # Batch verify all leaf nodes that require URL/content verification
    claims_and_sources = [
        (claim_publisher, sources_list, n_publisher, add_ins_publisher),
        (claim_imprint, sources_list, n_imprint, add_ins_imprint),
        (claim_location, sources_list, n_location, add_ins_location),
        (claim_nyt, sources_list, n_nyt, add_ins_nyt),
        (claim_hardcover, sources_list, n_hardcover, add_ins_hardcover),
        (claim_celebrity, sources_list, n_celebrity, add_ins_celebrity),
        (claim_genre, sources_list, n_genre, add_ins_genre),
    ]

    await evaluator.batch_verify(claims_and_sources)

    # Record some helpful custom info for debugging
    evaluator.add_custom_info(
        info={
            "normalized_isbn13": _normalize_isbn13(book.isbn13),
            "parsed_publication_date_iso": parsed_pub_date.isoformat() if parsed_pub_date else None,
            "sources_used_count": len(sources_list),
            "sources_used": sources_list,
        },
        info_type="debug",
        info_name="parsed_and_sources_info"
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
    Entry point to evaluate an answer for the celebrity cookbook/cocktail book task.
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
        default_model=model,
    )

    # 1) Extract structured book info from the answer
    book_info = await evaluator.extract(
        prompt=prompt_extract_book_info(),
        template_class=BookExtraction,
        extraction_name="book_info",
    )

    # 2) Build the verification tree and run checks
    await build_and_verify_celeb_book(evaluator, root, book_info)

    # 3) Return standard summary
    return evaluator.get_summary()