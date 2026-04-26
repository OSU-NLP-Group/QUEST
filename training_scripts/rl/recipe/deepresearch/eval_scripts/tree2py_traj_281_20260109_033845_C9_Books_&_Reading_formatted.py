import asyncio
import logging
import re
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "award_bigfive_institution_books_4"
TASK_DESCRIPTION = (
    "Identify 4 books that won major US or UK literary fiction awards (Pulitzer Prize for Fiction, "
    "National Book Award for Fiction, or Booker Prize) with award announcements between November 1, 2023, "
    "and May 31, 2025. Each book must be published by an imprint that is part of one of the Big Five publishers "
    "(Penguin Random House, HarperCollins, Simon & Schuster, Hachette Book Group, or Macmillan). Additionally, "
    "each book must have a verifiable connection to a specific literary institution or event—either an independent "
    "bookstore, a public library system, or a literary festival. For each book, provide: (1) the book title and author, "
    "(2) the specific award won and the award announcement date, (3) the publisher imprint name and its Big Five parent company, "
    "(4) the type, name, and a key identifying detail of the connected literary institution or event, and (5) reference URLs "
    "confirming all claims."
)

ALLOWED_AWARDS = [
    "Pulitzer Prize for Fiction",
    "National Book Award for Fiction",
    "Booker Prize",
    "The Booker Prize"
]
ALLOWED_BIG_FIVE = [
    "Penguin Random House",
    "HarperCollins",
    "Simon & Schuster",
    "Hachette Book Group",
    "Macmillan"
]
ALLOWED_CONNECTION_TYPES = [
    "independent bookstore",
    "public library system",
    "literary festival"
]

ANNOUNCEMENT_START = date(2023, 11, 1)
ANNOUNCEMENT_END = date(2025, 5, 31)

MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class BookAwardInfo(BaseModel):
    award_name: Optional[str] = None
    announcement_date: Optional[str] = None
    award_urls: List[str] = Field(default_factory=list)


class PublisherInfo(BaseModel):
    imprint: Optional[str] = None
    big_five_parent: Optional[str] = None
    publisher_urls: List[str] = Field(default_factory=list)


class ConnectionInfo(BaseModel):
    type: Optional[str] = None  # one of independent bookstore / public library system / literary festival
    name: Optional[str] = None
    detail_label: Optional[str] = None  # e.g., street address, founder name, architectural style, main building street address, month/year
    detail_value: Optional[str] = None
    festival_month: Optional[str] = None  # only for festival: month text, e.g., "October"
    festival_year: Optional[str] = None   # only for festival: 4-digit year, e.g., "2024"
    urls: List[str] = Field(default_factory=list)


class BookInfo(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    award: BookAwardInfo = Field(default_factory=BookAwardInfo)
    publisher: PublisherInfo = Field(default_factory=PublisherInfo)
    connection: ConnectionInfo = Field(default_factory=ConnectionInfo)


class BooksExtraction(BaseModel):
    books: List[BookInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
    Extract up to the FIRST FOUR books presented in the answer that are intended to meet the task requirements.
    For each book, extract the following fields EXACTLY as presented in the answer (do not invent or infer):

    1) Basic:
       - title: the book title
       - author: the book author(s)

    2) Award:
       - award.award_name: the name of the award the book won (e.g., "Pulitzer Prize for Fiction", "National Book Award for Fiction", "Booker Prize")
       - award.announcement_date: the announcement date text as provided (any reasonable date format is acceptable if present)
       - award.award_urls: a list of URL(s) explicitly cited that support the award win and (ideally) the announcement date
         IMPORTANT: Extract only URLs that appear in the answer. If no URLs were provided, return an empty list.

    3) Publisher:
       - publisher.imprint: the imprint name
       - publisher.big_five_parent: the Big Five parent company name as provided (e.g., "Penguin Random House", "HarperCollins", "Simon & Schuster", "Hachette Book Group", or "Macmillan")
       - publisher.publisher_urls: a list of URL(s) explicitly cited that support the imprint and its Big Five parent relationship
         IMPORTANT: Extract only URLs appearing in the answer. If none, return an empty list.

    4) Institution/Event Connection:
       - connection.type: one of "independent bookstore", "public library system", or "literary festival" (normalize phrasing; e.g., "book festival" -> "literary festival")
       - connection.name: the specific institution or event name
       - connection.detail_label: a concise label for the identifying detail provided (e.g., "street address", "founder name", "architectural style", "main building street address", "month/year")
       - connection.detail_value: the identifying detail text as provided (e.g., "123 Main St", "Founded by Jane Doe", "Brutalist", "10 Downing St", "October 2024")
       - connection.festival_month: if the type is "literary festival" and month/year are provided separately, record the month here (e.g., "October")
       - connection.festival_year: if the type is "literary festival" and month/year are provided separately, record the year here (e.g., "2024")
       - connection.urls: a list of URL(s) explicitly cited that support the claimed institution/event connection AND identifying detail.
         IMPORTANT: Extract only URLs appearing in the answer. If none, return an empty list.

    NOTE:
    - If any field is missing in the answer, set it to null (for strings) or an empty list (for URLs).
    - Do NOT fabricate values. Only extract what is present.
    - Return a JSON object with a 'books' array of up to 4 BookInfo objects.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def canonical_big_five_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = _normalize_text(name)
    mapping = {
        "prh": "penguin random house",
        "penguin random house": "penguin random house",
        "penguin-random house": "penguin random house",
        "penguin randomhouse": "penguin random house",
        "harpercollins": "harpercollins",
        "harper collins": "harpercollins",
        "simon & schuster": "simon & schuster",
        "simon and schuster": "simon & schuster",
        "hachette book group": "hachette book group",
        "hachette": "hachette book group",
        "macmillan": "macmillan",
        "macmillan publishers": "macmillan",
    }
    canon = mapping.get(n, n)
    # ensure it matches one of the allowed canonical forms exactly
    for allowed in ALLOWED_BIG_FIVE:
        if _normalize_text(allowed) == canon:
            return allowed
    # If not exact, return title-case of canon for readability
    return canon.title()


def is_valid_big_five_parent(name: Optional[str]) -> bool:
    canon = canonical_big_five_name(name)
    return canon is not None and any(_normalize_text(canon) == _normalize_text(x) for x in ALLOWED_BIG_FIVE)


def parse_date_flexible(text: Optional[str]) -> Optional[date]:
    if not text:
        return None
    t = text.strip()
    fmts = [
        "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y",
        "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
        "%B %Y", "%b %Y",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(t, fmt)
            # For formats without day, assume day 1
            if fmt in ("%B %Y", "%b %Y"):
                return date(dt.year, dt.month, 1)
            return dt.date()
        except Exception:
            continue
    # Try extracting patterns like "MonthName DD, YYYY" loosely
    m = re.search(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", t)
    if m:
        try:
            dt = datetime.strptime(m.group(0), "%B %d, %Y")
            return dt.date()
        except Exception:
            try:
                dt = datetime.strptime(m.group(0), "%b %d, %Y")
                return dt.date()
            except Exception:
                pass
    # Try "MonthName YYYY"
    m2 = re.search(r"([A-Za-z]+)\s+(\d{4})", t)
    if m2:
        mon = m2.group(1).lower()
        yr = int(m2.group(2))
        if mon in MONTH_NAMES:
            month_idx = MONTH_NAMES.index(mon) + 1
            return date(yr, month_idx, 1)
    # Try just year
    m3 = re.search(r"\b(20\d{2})\b", t)
    if m3:
        yr = int(m3.group(1))
        return date(yr, 1, 1)
    return None


def date_in_announcement_window(text: Optional[str]) -> bool:
    d = parse_date_flexible(text)
    if not d:
        return False
    return ANNOUNCEMENT_START <= d <= ANNOUNCEMENT_END


def normalize_connection_type(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    n = _normalize_text(t)
    if "festival" in n:
        return "literary festival"
    if "library" in n:
        return "public library system"
    if "bookstore" in n or "book shop" in n or "book shoppe" in n or "indie" in n:
        return "independent bookstore"
    return t.strip()


def is_valid_connection_type(t: Optional[str]) -> bool:
    nt = normalize_connection_type(t)
    return nt is not None and any(_normalize_text(nt) == _normalize_text(x) for x in ALLOWED_CONNECTION_TYPES)


def month_year_in_text(text: Optional[str]) -> bool:
    if not text:
        return False
    s = _normalize_text(text)
    has_month = any(m in s for m in MONTH_NAMES)
    has_year = bool(re.search(r"\b(19|20)\d{2}\b", s))
    return has_month and has_year


def validate_connection_detail(conn: ConnectionInfo) -> bool:
    ctype = normalize_connection_type(conn.type)
    label = _normalize_text(conn.detail_label)
    value = (conn.detail_value or "").strip()
    if not ctype:
        return False
    if ctype == "independent bookstore":
        # Require street address OR founder name (or owner)
        addr_ok = label and ("address" in label or "street address" in label)
        founder_ok = label and ("founder" in label or "owner" in label)
        return (addr_ok or founder_ok) and bool(value)
    if ctype == "public library system":
        # Require architectural style OR main building street address
        style_ok = label and ("architectural" in label or "style" in label)
        main_addr_ok = label and (("main building" in label and "address" in label) or "street address" in label)
        return (style_ok or main_addr_ok) and bool(value)
    if ctype == "literary festival":
        # Require month AND year
        month_text = _normalize_text(conn.festival_month) if conn.festival_month else ""
        year_text = _normalize_text(conn.festival_year) if conn.festival_year else ""
        month_ok = month_text in MONTH_NAMES or month_year_in_text(conn.detail_value)
        year_ok = bool(re.fullmatch(r"(19|20)\d{2}", year_text)) or month_year_in_text(conn.detail_value)
        return month_ok and year_ok
    return False


def require_sources_instruction(base_text: str, urls: List[str]) -> str:
    if urls and len(urls) > 0:
        return base_text
    return base_text + " IMPORTANT: No source URLs were provided in the answer for this verification. You must mark this verification as not supported (Incorrect)."


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_book(
    evaluator: Evaluator,
    parent_node,
    book: BookInfo,
    idx: int,
) -> None:
    # Book-level node (non-critical under root to allow partial credit across books)
    book_node = evaluator.add_parallel(
        id=f"book_{idx + 1}",
        desc=f"Book {idx + 1} (one of the four qualifying books)",
        parent=parent_node,
        critical=False
    )

    # -------------------- Basic Info --------------------
    basic_node = evaluator.add_parallel(
        id=f"book_{idx + 1}_basic_info",
        desc=f"Provide title and author for Book {idx + 1}",
        parent=book_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool((book.title or "").strip()),
        id=f"book_{idx + 1}_title",
        desc=f"Book {idx + 1} title is provided",
        parent=basic_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool((book.author or "").strip()),
        id=f"book_{idx + 1}_author",
        desc=f"Book {idx + 1} author is provided",
        parent=basic_node,
        critical=True
    )

    # -------------------- Award --------------------
    award_node = evaluator.add_parallel(
        id=f"book_{idx + 1}_award",
        desc=f"Award eligibility and announcement date for Book {idx + 1}",
        parent=book_node,
        critical=True
    )

    # Award name validity (eligible award)
    award_name_leaf = evaluator.add_leaf(
        id=f"book_{idx + 1}_award_name_valid",
        desc=f"Book {idx + 1} won one of: Pulitzer Prize for Fiction, National Book Award for Fiction, or Booker Prize",
        parent=award_node,
        critical=True
    )
    award_name_claim = (
        f"The award '{book.award.award_name or ''}' is one of the eligible awards: "
        f"Pulitzer Prize for Fiction, National Book Award for Fiction, or Booker Prize."
    )
    await evaluator.verify(
        claim=award_name_claim,
        node=award_name_leaf,
        additional_instruction=(
            "Judge strictly but allow minor naming variants (e.g., 'The Booker Prize' vs 'Booker Prize'). "
            "For National Book Award, it must be the Fiction category."
        )
    )

    # Announcement date in range (Nov 1, 2023 – May 31, 2025) and provided
    evaluator.add_custom_node(
        result=date_in_announcement_window(book.award.announcement_date),
        id=f"book_{idx + 1}_announcement_date_in_range",
        desc=f"Book {idx + 1} award announcement date is provided and falls between Nov 1, 2023 and May 31, 2025",
        parent=award_node,
        critical=True
    )

    # Award supported by URLs (win + announcement date)
    award_url_leaf = evaluator.add_leaf(
        id=f"book_{idx + 1}_award_url",
        desc=f"At least one credible reference URL supports Book {idx + 1} award win and announcement date",
        parent=award_node,
        critical=True
    )
    award_urls = book.award.award_urls or []
    award_support_claim = (
        f"Book '{book.title or ''}' by {book.author or ''} won the {book.award.award_name or ''}, "
        f"announced on {book.award.announcement_date or ''}."
    )
    await evaluator.verify(
        claim=award_support_claim,
        node=award_url_leaf,
        sources=award_urls,
        additional_instruction=require_sources_instruction(
            "Use only the provided URLs to confirm BOTH the award win and the announcement date. "
            "The page(s) must explicitly support these claims.",
            award_urls
        )
    )

    # -------------------- Publisher --------------------
    publisher_node = evaluator.add_parallel(
        id=f"book_{idx + 1}_publisher",
        desc=f"Publisher imprint and Big Five parent verification for Book {idx + 1}",
        parent=book_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool((book.publisher.imprint or "").strip()),
        id=f"book_{idx + 1}_imprint_name",
        desc=f"Book {idx + 1} publisher imprint name is provided",
        parent=publisher_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_valid_big_five_parent(book.publisher.big_five_parent),
        id=f"book_{idx + 1}_parent_big_five_valid",
        desc=f"Book {idx + 1} imprint is part of one of the Big Five (Penguin Random House, HarperCollins, Simon & Schuster, Hachette Book Group, or Macmillan)",
        parent=publisher_node,
        critical=True
    )

    publisher_url_leaf = evaluator.add_leaf(
        id=f"book_{idx + 1}_publisher_url",
        desc=f"At least one credible reference URL supports the imprint and its Big Five parent relationship",
        parent=publisher_node,
        critical=True
    )
    publisher_urls = book.publisher.publisher_urls or []
    publisher_claim = (
        f"The imprint '{book.publisher.imprint or ''}' belongs to the Big Five parent company '{book.publisher.big_five_parent or ''}'."
    )
    await evaluator.verify(
        claim=publisher_claim,
        node=publisher_url_leaf,
        sources=publisher_urls,
        additional_instruction=require_sources_instruction(
            "Verify the imprint-to-parent relationship using the provided URLs (prefer official publisher pages or reliable sources).",
            publisher_urls
        )
    )

    # -------------------- Institution/Event Connection --------------------
    inst_node = evaluator.add_parallel(
        id=f"book_{idx + 1}_institution_or_event",
        desc=f"Verified connection to an independent bookstore, public library system, or literary festival for Book {idx + 1}",
        parent=book_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=is_valid_connection_type(book.connection.type),
        id=f"book_{idx + 1}_type_valid",
        desc=f"Connection type is specified as independent bookstore, public library system, or literary festival",
        parent=inst_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool((book.connection.name or "").strip()),
        id=f"book_{idx + 1}_name_provided",
        desc=f"Specific institution/event name is provided",
        parent=inst_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=validate_connection_detail(book.connection),
        id=f"book_{idx + 1}_required_identifying_detail",
        desc=(
            "Key identifying detail is provided: bookstore (street address OR founder name); "
            "library system (architectural style OR main building street address); festival (month AND year)"
        ),
        parent=inst_node,
        critical=True
    )

    inst_url_leaf = evaluator.add_leaf(
        id=f"book_{idx + 1}_institution_url",
        desc=f"At least one credible reference URL supports the claimed connection and identifying detail",
        parent=inst_node,
        critical=True
    )
    inst_urls = book.connection.urls or []
    conn_type = normalize_connection_type(book.connection.type) or (book.connection.type or "")
    detail_label = book.connection.detail_label or ""
    detail_value = book.connection.detail_value or ""
    month_part = book.connection.festival_month or ""
    year_part = book.connection.festival_year or ""
    inst_claim = (
        f"The book '{book.title or ''}' by {book.author or ''} has a verifiable connection to the {conn_type} "
        f"'{book.connection.name or ''}', and the identifying detail is '{detail_label}: {detail_value}'. "
        f"For festivals, month/year provided: '{month_part} {year_part}'."
    )
    await evaluator.verify(
        claim=inst_claim,
        node=inst_url_leaf,
        sources=inst_urls,
        additional_instruction=require_sources_instruction(
            "The URL(s) must corroborate both the existence/identity of the institution/event and the identifying detail provided; "
            "they should also reasonably indicate a connection to the book (e.g., event listing, store page, reading, or mention).",
            inst_urls
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 4-books award/publisher/institution task.
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

    # Extract books data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction"
    )

    # Prepare exactly 4 books (pad with empty entries if fewer extracted)
    books: List[BookInfo] = list(extracted.books[:4])
    while len(books) < 4:
        books.append(BookInfo())

    # Add ground truth info for transparency
    evaluator.add_ground_truth({
        "allowed_awards": ALLOWED_AWARDS,
        "allowed_big_five": ALLOWED_BIG_FIVE,
        "allowed_connection_types": ALLOWED_CONNECTION_TYPES,
        "announcement_date_window": {
            "start": ANNOUNCEMENT_START.isoformat(),
            "end": ANNOUNCEMENT_END.isoformat()
        }
    }, gt_type="constraints")

    # Build verification for each of the four books
    for i in range(4):
        await verify_book(evaluator, root, books[i], i)

    # Return evaluation summary
    return evaluator.get_summary()