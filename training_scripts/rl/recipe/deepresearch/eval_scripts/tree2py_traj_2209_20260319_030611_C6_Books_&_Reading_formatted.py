import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_books_criteria_eval"
TASK_DESCRIPTION = """
Please identify three books published in the United States that meet the following specific criteria:

Book 1:
- Published between January 1, 2024 and December 31, 2025 (inclusive)
- Written by an author who is primarily known as a film director or producer (not primarily known as an author)
- The book must be a memoir focused on the author's career
- Page count must be greater than 500 pages
- Published by a major US publishing house

Book 2:
- Published between January 1, 2023 and December 31, 2024 (inclusive)
- Must have won either the Booker Prize, National Book Award, or Pulitzer Prize
- The winning announcement must have occurred in 2024 or 2025
- Page count must be less than 200 pages
- Must be a work of literary fiction

Book 3:
- Published in 2016
- Written by a musician who is primarily known for rock music performance
- Must be an autobiography covering the author's life and career
- Page count must be between 400 and 600 pages (inclusive)
- Published by a major US publishing house

For each book, provide:
1. The exact title
2. The author's full name
3. The publisher name
4. The publication date (month, day, and year)
5. The exact page count
6. A reference URL from a reliable source (publisher's website, major retailer, or reputable book database) that confirms these details
"""


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class BookItem(BaseModel):
    title: Optional[str] = None
    author_full_name: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[str] = None  # Prefer Month Day, Year as given by the answer
    page_count: Optional[str] = None        # Keep as string to maximize compatibility
    urls: List[str] = Field(default_factory=list)  # All reference URLs for this book

    # Optional supporting fields (if the answer provides them; not strictly required)
    author_primary_profession: Optional[str] = None  # e.g., "film director", "producer", "rock musician"
    content_type: Optional[str] = None               # e.g., "memoir", "autobiography"
    genre: Optional[str] = None                      # e.g., "literary fiction"
    award_name: Optional[str] = None                 # For Book 2 (e.g., "Pulitzer Prize")
    award_announcement_date: Optional[str] = None    # For Book 2 (e.g., "April 2024")


class BooksExtraction(BaseModel):
    book1: Optional[BookItem] = None
    book2: Optional[BookItem] = None
    book3: Optional[BookItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
Extract structured information for exactly three books (Book 1, Book 2, Book 3) as they appear in the answer.

For each book (book1, book2, book3), extract the following fields:
- title: exact title string as written in the answer (null if missing)
- author_full_name: full author name as written in the answer (null if missing)
- publisher: publisher/imprint name as written in the answer (null if missing)
- publication_date: publication date string as written in the answer (aim for Month Day, Year; null if missing)
- page_count: exact page count string as written in the answer (e.g., "544 pages" or "544"; null if missing)
- urls: array of all reliable reference URLs explicitly provided in the answer for that specific book (publishers, major retailers, reputable databases; can include multiple; empty if none)
- author_primary_profession: author's primary profession as described in the answer (e.g., "film director", "producer", "musician", "rock musician"; null if not mentioned)
- content_type: how the book is characterized, e.g., "memoir", "autobiography", "novel" (null if not mentioned)
- genre: genre if provided (e.g., "literary fiction"; null if not mentioned)
- award_name: the major award mentioned for Book 2 (e.g., "Pulitzer Prize", "Booker Prize", "National Book Award"; null if not mentioned)
- award_announcement_date: the announcement date/year as written for Book 2 (null if not mentioned)

Important:
- Only extract information explicitly present in the answer text.
- For urls, include every valid HTTP/HTTPS URL that the answer cites for that book. Do not invent URLs.
- If any field is not present, set it to null (or empty array for urls).
- Do not merge or swap books; map the first presented candidate to book1, the second to book2, the third to book3.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
_MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]

# A pragmatic list to guide the verifier about "major US publishing house" (non-exhaustive, includes Big Five + well-known imprints)
MAJOR_US_PUBLISHERS_HINT = [
    # Parent groups
    "Penguin Random House", "HarperCollins", "Simon & Schuster", "Hachette Book Group", "Macmillan",
    "W. W. Norton",  # independent but major US publisher
    # Common imprints (PRH)
    "Knopf", "Alfred A. Knopf", "Doubleday", "Crown", "Random House", "Viking", "Riverhead", "Scribner",
    "Little, Brown and Company", "Grand Central Publishing", "Farrar, Straus and Giroux", "Henry Holt",
    "St. Martin's Press", "Flatiron Books"
]


def _contains_month_day_year(date_str: Optional[str]) -> bool:
    """Heuristic check that a date string contains a month name, a day (1–31), and a 4-digit year."""
    if not date_str:
        return False
    s = date_str.strip()
    if not s:
        return False
    lower = s.lower()

    has_month = any(m in lower for m in _MONTHS)
    has_year = re.search(r"\b(19|20)\d{2}\b", s) is not None

    # Day-of-month: look for 1–31 appearing as a standalone number or with ordinal suffix
    # Use a regex that ensures digit boundaries are not part of a longer number (e.g., "2024" shouldn't count as day "20" or "24").
    has_day = re.search(r"(?<!\d)([1-9]|[12]\d|3[01])(st|nd|rd|th)?(?!\d)", s) is not None

    return has_month and has_day and has_year


def _contains_digits(text: Optional[str]) -> bool:
    return bool(text and re.search(r"\d", text))


def _safe_list(urls: Optional[List[str]]) -> List[str]:
    return urls if isinstance(urls, list) and len(urls) > 0 else []


def _title_author_snippet(title: Optional[str], author: Optional[str]) -> str:
    t = title or "the book"
    a = author or "the stated author"
    return f"'{t}' by {a}"


# --------------------------------------------------------------------------- #
# Verification builders per book                                              #
# --------------------------------------------------------------------------- #
async def build_book_1_verification(evaluator: Evaluator, root, b: BookItem) -> None:
    """
    Book 1:
    - Published 2024-01-01 to 2025-12-31 (inclusive)
    - Author primarily known as film director or producer
    - Memoir focused on author's career
    - Page count > 500
    - Major US publishing house
    """
    book_node = evaluator.add_parallel(
        id="book_1",
        desc="Book published 2024-2025 by entertainment professional, memoir, over 500 pages",
        parent=root,
        critical=False
    )

    # 0) Required fields (critical)
    req = evaluator.add_parallel(
        id="book_1_required_fields",
        desc="Required information fields for Book 1",
        parent=book_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(b.title and b.title.strip()),
        id="book_1_title",
        desc="Exact title is provided",
        parent=req,
        critical=True
    )
    evaluator.add_custom_node(
        result=_contains_month_day_year(b.publication_date),
        id="book_1_publication_date_provided",
        desc="Publication date with month, day, and year is provided",
        parent=req,
        critical=True
    )
    evaluator.add_custom_node(
        result=_contains_digits(b.page_count),
        id="book_1_page_count_provided",
        desc="Exact page count is provided",
        parent=req,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_list(b.urls)) > 0,
        id="book_1_reference_url",
        desc="At least one reference URL from a reliable source is provided that confirms the details",
        parent=req,
        critical=True
    )

    # 1) Publication info (critical)
    pub = evaluator.add_parallel(
        id="book_1_publication_info",
        desc="Publication details verification for Book 1",
        parent=book_node,
        critical=True
    )
    # 1.a) Publication date in range
    n_pubdate = evaluator.add_leaf(
        id="book_1_publication_date",
        desc="Publication date is between January 1, 2024 and December 31, 2025 (inclusive)",
        parent=pub,
        critical=True
    )
    claim_pubdate = (
        f"The publication date for {_title_author_snippet(b.title, b.author_full_name)} is between "
        "January 1, 2024 and December 31, 2025 (inclusive)."
    )
    await evaluator.verify(
        claim=claim_pubdate,
        node=n_pubdate,
        sources=_safe_list(b.urls),
        additional_instruction=(
            "Verify the publication or on-sale date shown on the page. "
            "If multiple editions/formats are listed, use the US edition's publication date when available. "
            "Accept synonyms like 'Publication date', 'On-sale date', or 'Release date'."
        ),
    )

    # 1.b) Publisher is a major US publishing house
    n_publisher = evaluator.add_leaf(
        id="book_1_publisher",
        desc="Published by a major US publishing house",
        parent=pub,
        critical=True
    )
    claim_publisher = (
        f"The publisher listed on the reference page for {_title_author_snippet(b.title, b.author_full_name)} "
        f"is '{b.publisher}'. This is a major US publishing house (Big Five or an imprint thereof, "
        f"or another widely recognized major US house)."
    )
    await evaluator.verify(
        claim=claim_publisher,
        node=n_publisher,
        sources=_safe_list(b.urls),
        additional_instruction=(
            "Determine the publisher/imprint from the page. Consider as 'major US publishing house' if it is one of "
            "the Big Five (Penguin Random House, HarperCollins, Simon & Schuster, Hachette, Macmillan) or a well-known "
            "US imprint under them (e.g., Knopf, Doubleday, Crown, Viking, Riverhead, Scribner, Little, Brown, Grand Central, "
            "FSG, Henry Holt, St. Martin's, Flatiron). "
            "W. W. Norton is also acceptable as a major US house. "
            "If the page clearly shows such a publisher/imprint, consider this supported."
        ),
    )

    # 2) Author info (critical)
    auth = evaluator.add_parallel(
        id="book_1_author_info",
        desc="Author background verification for Book 1",
        parent=book_node,
        critical=True
    )
    # 2.a) Profession: film director/producer
    n_prof = evaluator.add_leaf(
        id="book_1_author_profession",
        desc="Author is primarily known as a film director or producer (not primarily as an author)",
        parent=auth,
        critical=True
    )
    claim_prof = (
        f"The author {b.author_full_name or 'the stated author'} is primarily known as a film director or producer "
        "(e.g., 'film director', 'filmmaker', 'movie director', 'film producer') rather than primarily as an author."
    )
    await evaluator.verify(
        claim=claim_prof,
        node=n_prof,
        sources=_safe_list(b.urls),
        additional_instruction=(
            "Use reliable pages among the provided URLs (e.g., Wikipedia/official bio/publisher author page) to confirm "
            "the person is widely recognized primarily as a film director or producer. "
            "Mentions like 'filmmaker' also count."
        ),
    )
    # 2.b) Author name provided (presence)
    evaluator.add_custom_node(
        result=bool(b.author_full_name and b.author_full_name.strip()),
        id="book_1_author_name",
        desc="Author's full name is provided",
        parent=auth,
        critical=True
    )

    # 3) Content type (critical)
    n_content = evaluator.add_leaf(
        id="book_1_content_type",
        desc="Book is a memoir focused on the author's career",
        parent=book_node,
        critical=True
    )
    claim_content = (
        f"The book {_title_author_snippet(b.title, b.author_full_name)} is a memoir focused on the author's career "
        "(autobiographical memoir about their work/career)."
    )
    await evaluator.verify(
        claim=claim_content,
        node=n_content,
        sources=_safe_list(b.urls),
        additional_instruction=(
            "Look for explicit descriptors such as 'memoir', or descriptions indicating it is the author's own "
            "account of their career. Accept 'autobiographical memoir' or clear statements implying memoir of the author's career. "
            "Do not accept third-person biographies."
        ),
    )

    # 4) Physical specs (critical)
    phys = evaluator.add_parallel(
        id="book_1_physical_specs",
        desc="Physical specifications verification for Book 1",
        parent=book_node,
        critical=True
    )
    n_pages = evaluator.add_leaf(
        id="book_1_page_count",
        desc="Page count is greater than 500 pages",
        parent=phys,
        critical=True
    )
    claim_pages = (
        f"The exact print page count for {_title_author_snippet(b.title, b.author_full_name)} is strictly greater than 500 pages."
    )
    await evaluator.verify(
        claim=claim_pages,
        node=n_pages,
        sources=_safe_list(b.urls),
        additional_instruction=(
            "Use the page that lists 'pages', 'page count', or 'print length'. Prefer hardcover or print page count. "
            "Ignore audiobook hours or ebook 'location' counts."
        ),
    )


async def build_book_2_verification(evaluator: Evaluator, root, b: BookItem) -> None:
    """
    Book 2:
    - Published 2023-01-01 to 2024-12-31 (inclusive)
    - Won Booker, National Book Award, or Pulitzer Prize
    - Winning announcement in 2024 or 2025
    - Page count < 200
    - Literary fiction
    """
    book_node = evaluator.add_parallel(
        id="book_2",
        desc="Book published 2023-2024, won major award, literary fiction, under 200 pages",
        parent=root,
        critical=False
    )

    # 0) Required fields (critical)
    req = evaluator.add_parallel(
        id="book_2_required_fields",
        desc="Required information fields for Book 2",
        parent=book_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(b.title and b.title.strip()),
        id="book_2_title",
        desc="Exact title is provided",
        parent=req,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(b.author_full_name and b.author_full_name.strip()),
        id="book_2_author_name",
        desc="Author's full name is provided",
        parent=req,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(b.publisher and b.publisher.strip()),
        id="book_2_publisher",
        desc="Publisher name is provided",
        parent=req,
        critical=True
    )
    evaluator.add_custom_node(
        result=_contains_month_day_year(b.publication_date),
        id="book_2_publication_date_provided",
        desc="Publication date with month, day, and year is provided",
        parent=req,
        critical=True
    )
    evaluator.add_custom_node(
        result=_contains_digits(b.page_count),
        id="book_2_page_count_provided",
        desc="Exact page count is provided",
        parent=req,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_list(b.urls)) > 0,
        id="book_2_reference_url",
        desc="At least one reference URL from a reliable source is provided that confirms the details",
        parent=req,
        critical=True
    )

    # 1) Publication info (critical)
    pub = evaluator.add_parallel(
        id="book_2_publication_info",
        desc="Publication details verification for Book 2",
        parent=book_node,
        critical=True
    )
    n_pubdate = evaluator.add_leaf(
        id="book_2_publication_date",
        desc="Publication date is between January 1, 2023 and December 31, 2024 (inclusive)",
        parent=pub,
        critical=True
    )
    claim_pubdate = (
        f"The publication date for {_title_author_snippet(b.title, b.author_full_name)} is between "
        "January 1, 2023 and December 31, 2024 (inclusive)."
    )
    await evaluator.verify(
        claim=claim_pubdate,
        node=n_pubdate,
        sources=_safe_list(b.urls),
        additional_instruction=(
            "Verify publication or on-sale date on the page; accept synonyms. "
            "If multiple editions, use US edition date when available."
        ),
    )

    # 2) Award info (critical)
    awd = evaluator.add_parallel(
        id="book_2_award_info",
        desc="Award recognition verification for Book 2",
        parent=book_node,
        critical=True
    )
    # 2.a) Won a qualifying award
    n_award_won = evaluator.add_leaf(
        id="book_2_award_won",
        desc="Book won either the Booker Prize, National Book Award, or Pulitzer Prize",
        parent=awd,
        critical=True
    )
    award_name_phrase = (
        b.award_name if b.award_name else "one of the Booker Prize, National Book Award, or Pulitzer Prize"
    )
    claim_award_won = (
        f"{_title_author_snippet(b.title, b.author_full_name)} won {award_name_phrase} "
        "(must be the winner, not just shortlisted or finalist)."
    )
    await evaluator.verify(
        claim=claim_award_won,
        node=n_award_won,
        sources=_safe_list(b.urls),
        additional_instruction=(
            "Confirm the book is the winner of the specified major award (Booker Prize, National Book Award, or Pulitzer Prize). "
            "Shortlist/finalist/longlist is NOT sufficient."
        ),
    )
    # 2.b) Award timing 2024 or 2025
    n_award_time = evaluator.add_leaf(
        id="book_2_award_timing",
        desc="The award announcement occurred in 2024 or 2025",
        parent=awd,
        critical=True
    )
    claim_award_time = (
        f"The winning announcement for {award_name_phrase} related to {_title_author_snippet(b.title, b.author_full_name)} "
        "occurred in 2024 or 2025."
    )
    await evaluator.verify(
        claim=claim_award_time,
        node=n_award_time,
        sources=_safe_list(b.urls),
        additional_instruction=(
            "Use pages from the awarding organization or reputable news outlets to confirm the YEAR of the award WIN "
            "(announcement/publication of winners) is 2024 or 2025. Do not use shortlist/longlist dates."
        ),
    )

    # 3) Genre: literary fiction (critical)
    n_genre = evaluator.add_leaf(
        id="book_2_genre",
        desc="Book is a work of literary fiction",
        parent=book_node,
        critical=True
    )
    claim_genre = (
        f"The book {_title_author_snippet(b.title, b.author_full_name)} is a work of literary fiction."
    )
    await evaluator.verify(
        claim=claim_genre,
        node=n_genre,
        sources=_safe_list(b.urls),
        additional_instruction=(
            "Accept descriptors such as 'Literary Fiction', 'Fiction / Literary', or clear phrasing indicating it is a literary novel."
        ),
    )

    # 4) Physical specs: < 200 pages (critical)
    phys = evaluator.add_parallel(
        id="book_2_physical_specs",
        desc="Physical specifications verification for Book 2",
        parent=book_node,
        critical=True
    )
    n_pages = evaluator.add_leaf(
        id="book_2_page_count",
        desc="Page count is less than 200 pages",
        parent=phys,
        critical=True
    )
    claim_pages = (
        f"The exact print page count for {_title_author_snippet(b.title, b.author_full_name)} is strictly less than 200 pages."
    )
    await evaluator.verify(
        claim=claim_pages,
        node=n_pages,
        sources=_safe_list(b.urls),
        additional_instruction=(
            "Use the page that lists 'pages', 'page count', or 'print length'. Prefer hardcover or print page count."
        ),
    )


async def build_book_3_verification(evaluator: Evaluator, root, b: BookItem) -> None:
    """
    Book 3:
    - Published in 2016
    - Author is a rock musician (primarily known for rock performance)
    - Autobiography covering life and career
    - Page count between 400 and 600 inclusive
    - Major US publishing house
    """
    book_node = evaluator.add_parallel(
        id="book_3",
        desc="Book published 2016, autobiography by rock musician, 400-600 pages",
        parent=root,
        critical=False
    )

    # 0) Required fields (critical)
    req = evaluator.add_parallel(
        id="book_3_required_fields",
        desc="Required information fields for Book 3",
        parent=book_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(b.title and b.title.strip()),
        id="book_3_title",
        desc="Exact title is provided",
        parent=req,
        critical=True
    )
    evaluator.add_custom_node(
        result=_contains_month_day_year(b.publication_date),
        id="book_3_publication_date_provided",
        desc="Publication date with month, day, and year is provided",
        parent=req,
        critical=True
    )
    evaluator.add_custom_node(
        result=_contains_digits(b.page_count),
        id="book_3_page_count_provided",
        desc="Exact page count is provided",
        parent=req,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(_safe_list(b.urls)) > 0,
        id="book_3_reference_url",
        desc="At least one reference URL from a reliable source is provided that confirms the details",
        parent=req,
        critical=True
    )

    # 1) Publication info (critical)
    pub = evaluator.add_parallel(
        id="book_3_publication_info",
        desc="Publication details verification for Book 3",
        parent=book_node,
        critical=True
    )
    # 1.a) Publication year is 2016
    n_year = evaluator.add_leaf(
        id="book_3_publication_year",
        desc="Publication year is 2016",
        parent=pub,
        critical=True
    )
    claim_year = (
        f"The publication year for {_title_author_snippet(b.title, b.author_full_name)} is 2016."
    )
    await evaluator.verify(
        claim=claim_year,
        node=n_year,
        sources=_safe_list(b.urls),
        additional_instruction=(
            "Look for publication date or 'first published' year indicating 2016. "
            "If multiple editions, prefer the US edition date/year."
        ),
    )
    # 1.b) Publisher is a major US publishing house
    n_publisher = evaluator.add_leaf(
        id="book_3_publisher",
        desc="Published by a major US publishing house",
        parent=pub,
        critical=True
    )
    claim_publisher = (
        f"The publisher listed on the reference page for {_title_author_snippet(b.title, b.author_full_name)} "
        f"is '{b.publisher}'. This is a major US publishing house (Big Five or an imprint thereof, "
        f"or another widely recognized major US house)."
    )
    await evaluator.verify(
        claim=claim_publisher,
        node=n_publisher,
        sources=_safe_list(b.urls),
        additional_instruction=(
            "Determine the publisher/imprint from the page. Consider as 'major US publishing house' if it is one of "
            "the Big Five (Penguin Random House, HarperCollins, Simon & Schuster, Hachette, Macmillan) or a well-known "
            "US imprint under them (e.g., Knopf, Doubleday, Crown, Viking, Riverhead, Scribner, Little, Brown, Grand Central, "
            "FSG, Henry Holt, St. Martin's, Flatiron). "
            "W. W. Norton is also acceptable as a major US house."
        ),
    )

    # 2) Author info (critical)
    auth = evaluator.add_parallel(
        id="book_3_author_info",
        desc="Author background verification for Book 3",
        parent=book_node,
        critical=True
    )
    # 2.a) Profession: rock musician
    n_prof = evaluator.add_leaf(
        id="book_3_author_profession",
        desc="Author is a musician primarily known for rock music performance",
        parent=auth,
        critical=True
    )
    claim_prof = (
        f"The author {b.author_full_name or 'the stated author'} is a musician primarily known for rock music performance "
        "(e.g., rock singer, guitarist, frontperson, or member of a rock band)."
    )
    await evaluator.verify(
        claim=claim_prof,
        node=n_prof,
        sources=_safe_list(b.urls),
        additional_instruction=(
            "Use reliable pages among the provided URLs (e.g., Wikipedia/official site/publisher author page/news) to confirm "
            "the person is widely recognized as a rock musician."
        ),
    )
    # 2.b) Author name provided (presence)
    evaluator.add_custom_node(
        result=bool(b.author_full_name and b.author_full_name.strip()),
        id="book_3_author_name",
        desc="Author's full name is provided",
        parent=auth,
        critical=True
    )

    # 3) Content type (critical)
    n_content = evaluator.add_leaf(
        id="book_3_content_type",
        desc="Book is an autobiography covering the author's life and career",
        parent=book_node,
        critical=True
    )
    claim_content = (
        f"The book {_title_author_snippet(b.title, b.author_full_name)} is an autobiography covering the author's life and career."
    )
    await evaluator.verify(
        claim=claim_content,
        node=n_content,
        sources=_safe_list(b.urls),
        additional_instruction=(
            "Look for descriptors 'autobiography', 'memoir' (authored by the subject), or clear statements indicating it is the author's own life story."
        ),
    )

    # 4) Physical specs (critical)
    phys = evaluator.add_parallel(
        id="book_3_physical_specs",
        desc="Physical specifications verification for Book 3",
        parent=book_node,
        critical=True
    )
    n_pages = evaluator.add_leaf(
        id="book_3_page_count",
        desc="Page count is between 400 and 600 pages (inclusive)",
        parent=phys,
        critical=True
    )
    claim_pages = (
        f"The exact print page count for {_title_author_snippet(b.title, b.author_full_name)} is between 400 and 600 pages (inclusive)."
    )
    await evaluator.verify(
        claim=claim_pages,
        node=n_pages,
        sources=_safe_list(b.urls),
        additional_instruction=(
            "Use the page that lists 'pages', 'page count', or 'print length'. Prefer hardcover or print page count."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 'three books with complex criteria' task using the obj_task_eval framework.
    """
    # Initialize evaluator and root
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

    # Add helpful custom info (hint list for transparency)
    evaluator.add_custom_info(
        {"examples": MAJOR_US_PUBLISHERS_HINT},
        info_type="guidance",
        info_name="major_us_publishers_hint",
    )

    # Extract structured info
    books = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction",
    )

    # Build verification subtrees for each book
    # Ensure each tree is created even if missing fields; presence nodes will fail accordingly.
    await build_book_1_verification(evaluator, root, books.book1 or BookItem())
    await build_book_2_verification(evaluator, root, books.book2 or BookItem())
    await build_book_3_verification(evaluator, root, books.book3 or BookItem())

    # Return evaluation summary
    return evaluator.get_summary()