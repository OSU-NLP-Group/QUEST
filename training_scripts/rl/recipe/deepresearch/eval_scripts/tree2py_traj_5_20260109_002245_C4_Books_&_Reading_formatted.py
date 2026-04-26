import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nba_fiction_2024_winner_biblio"
TASK_DESCRIPTION = (
    "Identify the fiction book that won the 2024 National Book Award for Fiction (the winner, not a finalist). "
    "Provide the following complete bibliographic information for this book: the author's full name as it appears on the book, "
    "the complete book title, the publisher's name, a valid ISBN-13 (in 13-digit format), the publication year, "
    "a reference URL from an authoritative source (such as the National Book Foundation or major literary publications) "
    "confirming the book's award win, and the total page count of the book."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BookInfo(BaseModel):
    """Structured extraction of winner's bibliographic information from the agent's answer."""
    author_full_name: Optional[str] = None
    book_title: Optional[str] = None
    publisher_name: Optional[str] = None
    isbn_13: Optional[str] = None
    publication_year: Optional[str] = None
    award_reference_url: Optional[str] = None
    page_count: Optional[str] = None

    # All URLs mentioned in the answer (including but not limited to the award URL)
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_book_info() -> str:
    return """
    Extract the bibliographic information for the book identified in the answer as the winner of the 2024 National Book Award for Fiction (winner, not finalist).

    Return a JSON object with the following fields:
    - author_full_name: The author's full name as it appears on the published book (do not abbreviate; include middle names/initials if present).
    - book_title: The complete title of the book exactly as it appears on the published work, including any subtitles or punctuation.
    - publisher_name: The name of the publisher as shown in the book's publication information or reputable bibliographic listings.
    - isbn_13: The ISBN-13 for the book. If hyphens are present, preserve them; if absent, provide just the digits. This must correspond to the book.
    - publication_year: The publication year (e.g., "2024"). Prefer the year of the edition referenced by the answer.
    - award_reference_url: A single authoritative URL explicitly confirming that this book won the 2024 National Book Award for Fiction. Prefer nationalbook.org (National Book Foundation) or reputable major literary publications (e.g., The New York Times, The Guardian). If multiple such URLs are present, select the most authoritative one (nationalbook.org first, then others).
    - page_count: The total number of pages in the book for the referenced edition.
    - sources: An array of all URLs mentioned in the answer (including the award_reference_url and any other URLs such as publisher page, bookstore page, bibliographic record, etc.).

    Rules:
    1. Extract values exactly as they appear in the answer. Do not invent or infer values not present.
    2. If any field is missing from the answer, return null for that field (or an empty array for sources).
    3. Include all URLs you can find in the answer in the 'sources' array. Valid formats include raw URLs, markdown links, or embedded references. Extract the actual URL strings.
    4. If a URL lacks a protocol, prepend http:// to make it a valid URL.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _collect_all_sources(info: BookInfo) -> List[str]:
    """Combine award_reference_url and sources, deduplicate and filter empties."""
    urls = []
    if info.award_reference_url and info.award_reference_url.strip():
        urls.append(info.award_reference_url.strip())
    urls.extend([u.strip() for u in info.sources if isinstance(u, str) and u.strip()])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _sanitize_isbn13(isbn: Optional[str]) -> Optional[str]:
    if not isbn:
        return None
    return re.sub(r"[^0-9Xx]", "", isbn)


def _is_valid_isbn13(isbn: Optional[str]) -> bool:
    """
    Validate ISBN-13 checksum (digits only, 13 length).
    Hyphens/spaces removed. 'X' is not used in ISBN-13 (only ISBN-10), so reject if present.
    """
    if isbn is None:
        return False
    digits = re.sub(r"[^0-9]", "", isbn)
    if len(digits) != 13:
        return False
    # checksum: sum of digits with alternated weights (1,3) must be divisible by 10
    total = 0
    for i, ch in enumerate(digits):
        d = int(ch)
        total += d if i % 2 == 0 else 3 * d
    return total % 10 == 0


def _extract_year_number(year_str: Optional[str]) -> Optional[int]:
    """Extract first plausible 4-digit year from the string."""
    if not year_str:
        return None
    m = re.search(r"\b(19|20)\d{2}\b", year_str)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _is_valid_award_year(year_int: Optional[int]) -> bool:
    """For 2024 National Book Award eligibility, we accept publication year 2023 or 2024."""
    if year_int is None:
        return False
    return year_int in (2023, 2024)


def _parse_page_count(page_str: Optional[str]) -> Optional[int]:
    """Extract numeric page count (positive integer) from the string."""
    if not page_str:
        return None
    m = re.search(r"\b(\d{1,5})\b", page_str)
    if not m:
        return None
    try:
        val = int(m.group(1))
        return val if val > 0 else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_author_full_name(evaluator: Evaluator, parent_node, info: BookInfo) -> None:
    group = evaluator.add_parallel(
        id="Author_Full_Name",
        desc="Provides the complete and correct name of the author as it appears in official award announcements and on the published book.",
        parent=parent_node,
        critical=True
    )

    # Existence
    author_provided = bool(info.author_full_name and info.author_full_name.strip())
    evaluator.add_custom_node(
        result=author_provided,
        id="author_provided",
        desc="Author full name is provided in the answer",
        parent=group,
        critical=True
    )

    # Verification against sources (prefer award + any credible sources)
    author_leaf = evaluator.add_leaf(
        id="author_verified",
        desc="Author full name matches authoritative sources for the winner book",
        parent=group,
        critical=True
    )

    title = info.book_title or ""
    author = info.author_full_name or ""
    claim = f"The book '{title}' is authored by '{author}'."

    await evaluator.verify(
        claim=claim,
        node=author_leaf,
        sources=_collect_all_sources(info),
        additional_instruction="Verify via authoritative page(s) that the stated author corresponds to the winner book. "
                               "Allow minor formatting variations (middle initials, accents, casing), but confirm identity."
    )


async def verify_book_title(evaluator: Evaluator, parent_node, info: BookInfo) -> None:
    group = evaluator.add_parallel(
        id="Book_Title",
        desc="Provides the complete and correct title of the book exactly as it appears on the published work.",
        parent=parent_node,
        critical=True
    )

    # Existence
    title_provided = bool(info.book_title and info.book_title.strip())
    evaluator.add_custom_node(
        result=title_provided,
        id="title_provided",
        desc="Book title is provided in the answer",
        parent=group,
        critical=True
    )

    # Verification
    title_leaf = evaluator.add_leaf(
        id="title_verified",
        desc="Book title matches authoritative sources for the winner book",
        parent=group,
        critical=True
    )

    title = info.book_title or ""
    claim = f"The complete title of the winner book is '{title}'."

    await evaluator.verify(
        claim=claim,
        node=title_leaf,
        sources=_collect_all_sources(info),
        additional_instruction="Confirm the exact book title (including subtitle and punctuation) from authoritative sources "
                               "(e.g., the National Book Foundation page or publisher page). Allow minor punctuation variations if clearly the same work."
    )


async def verify_publisher_name(evaluator: Evaluator, parent_node, info: BookInfo) -> None:
    group = evaluator.add_parallel(
        id="Publisher_Name",
        desc="Provides the correct name of the publisher as it appears in the book's publication information.",
        parent=parent_node,
        critical=True
    )

    # Existence
    pub_provided = bool(info.publisher_name and info.publisher_name.strip())
    evaluator.add_custom_node(
        result=pub_provided,
        id="publisher_provided",
        desc="Publisher name is provided in the answer",
        parent=group,
        critical=True
    )

    # Verification
    pub_leaf = evaluator.add_leaf(
        id="publisher_verified",
        desc="Publisher matches authoritative sources for the winner book",
        parent=group,
        critical=True
    )

    title = info.book_title or ""
    publisher = info.publisher_name or ""
    claim = f"The publisher of '{title}' is '{publisher}'."

    await evaluator.verify(
        claim=claim,
        node=pub_leaf,
        sources=_collect_all_sources(info),
        additional_instruction="Verify the publisher from credible bibliographic or publisher pages (e.g., publisher site, bibliographic catalog)."
    )


async def verify_isbn_13(evaluator: Evaluator, parent_node, info: BookInfo) -> None:
    group = evaluator.add_parallel(
        id="ISBN_13",
        desc="Provides a valid ISBN-13 in the standard 13-digit format for the book.",
        parent=parent_node,
        critical=True
    )

    # Existence
    isbn_provided = bool(info.isbn_13 and info.isbn_13.strip())
    evaluator.add_custom_node(
        result=isbn_provided,
        id="isbn_provided",
        desc="ISBN-13 is provided in the answer",
        parent=group,
        critical=True
    )

    # Format validity (13-digit checksum)
    is_valid_format = _is_valid_isbn13(info.isbn_13)
    evaluator.add_custom_node(
        result=is_valid_format,
        id="isbn_format_valid",
        desc="ISBN-13 is a valid 13-digit number with correct checksum (hyphens/spaces ignored)",
        parent=group,
        critical=True
    )

    # Verification against sources
    isbn_leaf = evaluator.add_leaf(
        id="isbn_verified",
        desc="ISBN-13 matches authoritative sources for the winner book",
        parent=group,
        critical=True
    )

    title = info.book_title or ""
    isbn = info.isbn_13 or ""
    claim = f"The ISBN-13 for '{title}' is '{isbn}'."

    await evaluator.verify(
        claim=claim,
        node=isbn_leaf,
        sources=_collect_all_sources(info),
        additional_instruction="Confirm the ISBN-13 (ignoring hyphenation differences) from authoritative bibliographic records or publisher listings."
    )


async def verify_publication_year(evaluator: Evaluator, parent_node, info: BookInfo) -> None:
    group = evaluator.add_parallel(
        id="Publication_Year",
        desc="Provides the correct publication year of the book, which should be 2023 or 2024 to be eligible for the 2024 award.",
        parent=parent_node,
        critical=True
    )

    # Existence
    year_provided = bool(info.publication_year and info.publication_year.strip())
    evaluator.add_custom_node(
        result=year_provided,
        id="pub_year_provided",
        desc="Publication year is provided in the answer",
        parent=group,
        critical=True
    )

    # Range validity
    year_int = _extract_year_number(info.publication_year)
    evaluator.add_custom_node(
        result=_is_valid_award_year(year_int),
        id="pub_year_in_valid_range",
        desc=f"Publication year is valid for 2024 award eligibility (extracted year: {year_int})",
        parent=group,
        critical=True
    )

    # Verification
    year_leaf = evaluator.add_leaf(
        id="pub_year_verified",
        desc="Publication year matches authoritative sources for the winner book",
        parent=group,
        critical=True
    )

    title = info.book_title or ""
    year_str = str(year_int) if year_int is not None else (info.publication_year or "")
    claim = f"The publication year of '{title}' is {year_str}."

    await evaluator.verify(
        claim=claim,
        node=year_leaf,
        sources=_collect_all_sources(info),
        additional_instruction="Verify the publication year from authoritative bibliographic listings or the publisher page. "
                               "The year must be 2023 or 2024 for 2024 award eligibility."
    )


async def verify_award_confirmation_reference(evaluator: Evaluator, parent_node, info: BookInfo) -> None:
    group = evaluator.add_parallel(
        id="Award_Confirmation_Reference",
        desc="Provides a reference URL from an authoritative source confirming this book won the 2024 National Book Award for Fiction.",
        parent=parent_node,
        critical=True
    )

    # Existence & basic validity
    url_ok = bool(info.award_reference_url and info.award_reference_url.strip() and re.match(r"^https?://", info.award_reference_url.strip()))
    evaluator.add_custom_node(
        result=url_ok,
        id="award_url_provided",
        desc="Authoritative award confirmation URL is provided (valid http/https)",
        parent=group,
        critical=True
    )

    # Verification: Winner confirmation (not finalist)
    award_leaf = evaluator.add_leaf(
        id="award_win_confirmed",
        desc="The provided authoritative URL confirms the book won (not just finalist) the 2024 National Book Award for Fiction",
        parent=group,
        critical=True
    )

    title = info.book_title or ""
    author = info.author_full_name or ""
    claim = f"The book '{title}' by '{author}' won the 2024 National Book Award for Fiction (winner, not finalist)."

    # Prefer the single authoritative award URL for this check
    sources = info.award_reference_url if (info.award_reference_url and info.award_reference_url.strip()) else None

    await evaluator.verify(
        claim=claim,
        node=award_leaf,
        sources=sources,
        additional_instruction="Confirm the page explicitly states the book is the 'Winner' of the 2024 National Book Award for Fiction. "
                               "Do not accept pages that only indicate 'Finalist'."
    )


async def verify_page_count(evaluator: Evaluator, parent_node, info: BookInfo) -> None:
    group = evaluator.add_parallel(
        id="Page_Count",
        desc="Provides the total number of pages in the book.",
        parent=parent_node,
        critical=True
    )

    # Existence
    page_provided = bool(info.page_count and info.page_count.strip())
    evaluator.add_custom_node(
        result=page_provided,
        id="page_count_provided",
        desc="Page count is provided in the answer",
        parent=group,
        critical=True
    )

    # Numeric validity
    pages_int = _parse_page_count(info.page_count)
    evaluator.add_custom_node(
        result=(pages_int is not None and pages_int > 0),
        id="page_count_numeric_valid",
        desc=f"Page count is a positive integer (extracted pages: {pages_int})",
        parent=group,
        critical=True
    )

    # Verification
    page_leaf = evaluator.add_leaf(
        id="page_count_verified",
        desc="Page count matches authoritative sources for the winner book",
        parent=group,
        critical=True
    )

    title = info.book_title or ""
    page_str = str(pages_int) if pages_int is not None else (info.page_count or "")
    claim = f"The total number of pages of '{title}' is {page_str}."

    await evaluator.verify(
        claim=claim,
        node=page_leaf,
        sources=_collect_all_sources(info),
        additional_instruction="Verify the page count from authoritative bibliographic listings or the publisher page. "
                               "Focus on the edition referenced by the answer; minor edition-to-edition differences should be noted but verify the stated count."
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
    Evaluate an answer for the 2024 National Book Award for Fiction winner bibliographic task.
    """
    # Initialize evaluator (root is parallel to independently assess each bibliographic criterion)
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

    # Extract structured bibliographic information from the agent's answer
    book_info = await evaluator.extract(
        prompt=prompt_extract_book_info(),
        template_class=BookInfo,
        extraction_name="winner_book_info"
    )

    # Build verification tree according to rubric
    # Root child: Overall information group (parallel, non-critical per rubric root)
    nb_group = evaluator.add_parallel(
        id="2024_National_Book_Award_Fiction_Winner_Information",
        desc="Correctly identifies the 2024 National Book Award for Fiction winner and provides complete bibliographic information.",
        parent=root,
        critical=False
    )

    # Verify each critical component (as critical groups with critical children)
    await verify_award_confirmation_reference(evaluator, nb_group, book_info)
    await verify_author_full_name(evaluator, nb_group, book_info)
    await verify_book_title(evaluator, nb_group, book_info)
    await verify_publisher_name(evaluator, nb_group, book_info)
    await verify_isbn_13(evaluator, nb_group, book_info)
    await verify_publication_year(evaluator, nb_group, book_info)
    await verify_page_count(evaluator, nb_group, book_info)

    # Return structured summary
    return evaluator.get_summary()