import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "award_fiction_2024_2025"
TASK_DESCRIPTION = (
    "Identify four fiction books that won major literary awards (Pulitzer Prize for Fiction, "
    "National Book Award for Fiction, Booker Prize, Kirkus Prize, Women's Prize for Fiction, "
    "or Andrew Carnegie Medal for Excellence in Fiction) in either 2024 or 2025. At least one of the four "
    "books must have won multiple awards from the list above. For each book, provide: (1) the book title, "
    "(2) the author's full name, (3) the publisher, (4) the publication date (month and year), "
    "(5) the specific award(s) won and the year the award was won, and (6) a reference URL from an official award page, "
    "major book retailer, or publisher website. The four books must be published by four different publishers."
)

ALLOWED_AWARDS = [
    "Pulitzer Prize for Fiction",
    "National Book Award for Fiction",
    "Booker Prize",
    "Kirkus Prize",
    "Women's Prize for Fiction",
    "Andrew Carnegie Medal for Excellence in Fiction",
]

ALLOWED_AWARD_HINTS = [
    # Award site domains (examples)
    "pulitzer.org",
    "nationalbook.org",
    "thebookerprizes.com",
    "kirkusreviews.com",
    "kirkus.com",
    "womensprizeforfiction.co.uk",
    "womensprize.com",
    "ala.org",
    # Major retailers (examples)
    "amazon.com",
    "barnesandnoble.com",
    "bookshop.org",
    "waterstones.com",
    # Publishers/imprints (examples)
    "penguinrandomhouse.com",
    "harpercollins.com",
    "simonandschuster.com",
    "us.macmillan.com",
    "macmillan.com",
    "hachettebookgroup.com",
    "bloomsbury.com",
    "fsgbooks.com",
    "knopfdoubleday.com",
    "randomhousebooks.com",
    "riverheadbooks.com",
    "wwnorton.com",
    "liveright.com",
    "graywolfpress.org",
    "littlebrown.com",
    "orionbooks.co.uk",
    "canongate.co.uk",
]  # Non-exhaustive examples; the judge LLM will still read the page contents.


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AwardItem(BaseModel):
    award_name: Optional[str] = None
    award_year: Optional[str] = None  # keep as string to be robust (e.g., "2024", "2025")


class BookItem(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[str] = None  # Month and Year as a single string (e.g., "May 2024")
    category_or_genre: Optional[str] = None  # e.g., "novel", "fiction", "short stories"
    awards: List[AwardItem] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


class BooksExtraction(BaseModel):
    books: List[BookItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
    Extract all books explicitly presented by the answer as part of the requested list. For each such book, extract:
    - title: Exact book title as given
    - author: Full author name as provided
    - publisher: Publisher or imprint as presented in the answer
    - publication_date: Publication month and year as a single string (e.g., "May 2024"); if the answer gives a full date, reduce to "Month YYYY"
    - category_or_genre: The category/genre string as presented (e.g., "novel", "fiction", "short stories"). If not stated, return null.
    - awards: An array of objects; each object should contain:
        • award_name: The name of each specific award the book is claimed to have won
        • award_year: The year (e.g., "2024" or "2025") the award was won
    - reference_urls: All URLs provided in the answer that are specifically intended as references for that book.
      Include only URLs that appear in the answer text (plain or inside markdown).
      Do not invent URLs.

    Important:
    - Only extract the books the answer itself lists as the four items for this task. Preserve the order in which they are listed.
    - If any field is missing in the answer for a given book, set it to null (or for arrays, use empty array).
    - For publication_date, if only a year is present, use the year; but prefer "Month YYYY" if month is available.
    - Do not deduplicate or merge books; keep them as listed even if similar.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _award_canonical_name(name: str) -> Optional[str]:
    n = _normalize_text(name)
    if not n:
        return None
    if ("pulitzer" in n) and ("fiction" in n):
        return "Pulitzer Prize for Fiction"
    if ("national book award" in n) and ("fiction" in n):
        return "National Book Award for Fiction"
    if "booker" in n:
        return "Booker Prize"
    if "kirkus" in n:
        # Kirkus Prize has categories; we rely on fiction check elsewhere
        return "Kirkus Prize"
    if ("women" in n) and ("fiction" in n):
        return "Women's Prize for Fiction"
    if ("carnegie" in n) and ("fiction" in n):
        return "Andrew Carnegie Medal for Excellence in Fiction"
    return None


def _year_is_2024_or_2025(year_str: Optional[str]) -> bool:
    y = _normalize_text(year_str)
    return "2024" in y or "2025" in y


def _select_primary_allowed_award(book: BookItem) -> Optional[Tuple[str, str]]:
    """
    Pick one allowed award-year pair (2024/2025) from the extracted awards for this book.
    Returns (canonical_award_name, award_year_str) or None.
    """
    for a in book.awards:
        can = _award_canonical_name(a.award_name or "")
        if can and _year_is_2024_or_2025(a.award_year):
            return can, (a.award_year or "").strip()
    return None


def _has_multiple_allowed_awards(book: BookItem) -> bool:
    count = 0
    for a in book.awards:
        if _award_canonical_name(a.award_name or "") and _year_is_2024_or_2025(a.award_year):
            count += 1
        if count >= 2:
            return True
    return False


def _unique_nonempty_publishers(books: List[BookItem]) -> int:
    pubs = []
    for b in books[:4]:  # only first 4 matter
        p = (b.publisher or "").strip()
        if p:
            pubs.append(p.lower())
    return len(set(pubs))


def _urls_list_or_empty(book: BookItem) -> List[str]:
    return [u for u in (book.reference_urls or []) if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Book verification                                                           #
# --------------------------------------------------------------------------- #
async def verify_one_book(evaluator: Evaluator, parent_node, book: BookItem, idx: int) -> None:
    """
    Build and run verification checks for a single book.
    All per-book checks are critical under the book_i node (which is non-critical),
    enabling pass/fail per-book while allowing partial credit across books.
    """
    book_node = evaluator.add_parallel(
        id=f"book_{idx+1}",
        desc=f"{['First','Second','Third','Fourth'][idx]} award-winning fiction book meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    urls = _urls_list_or_empty(book)
    title = book.title or ""
    author = book.author or ""
    publisher = book.publisher or ""
    pub_date = book.publication_date or ""
    category = book.category_or_genre or ""

    # 1) Reference URL presence and category (simple verify against the answer text)
    ref_leaf = evaluator.add_leaf(
        id=f"book_{idx+1}_reference_url",
        desc=f"Book {idx+1} has at least one reference URL from official award page, major retailer, or publisher website",
        parent=book_node,
        critical=True
    )
    claim_ref = (
        f"The answer provides at least one reference URL for this book (title: '{title}', author: '{author}'). "
        f"Here are the extracted URLs: {urls}. At least one of these URLs is from an official award site, "
        f"a major retailer, or a publisher/imprint website."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=None,
        additional_instruction=(
            "Judge only by the provided answer text (not external knowledge). "
            "A URL qualifies if its domain is an official award site (e.g., pulitzer.org, nationalbook.org, "
            "thebookerprizes.com, kirkusreviews.com/kirkus.com, womensprizeforfiction.co.uk/womensprize.com, ala.org), "
            "a major retailer (e.g., amazon.com, barnesandnoble.com, bookshop.org, waterstones.com), "
            "or a publisher/imprint site (e.g., penguinrandomhouse.com, harpercollins.com, simonandschuster.com, "
            "macmillan.com/us.macmillan.com, hachettebookgroup.com, bloomsbury.com, fsgbooks.com, "
            "knopfdoubleday.com, randomhousebooks.com, riverheadbooks.com, wwnorton.com, liveright.com, graywolfpress.org). "
            "Pass if at least one listed URL is valid and in a qualifying domain category."
        ),
    )

    # 2) Fiction category verification
    fiction_leaf = evaluator.add_leaf(
        id=f"book_{idx+1}_fiction_category",
        desc=f"Book {idx+1} is in the fiction category",
        parent=book_node,
        critical=True
    )
    claim_fiction = (
        f"The book titled '{title}' by {author} is a work of fiction (e.g., a novel or a short story collection)."
    )
    await evaluator.verify(
        claim=claim_fiction,
        node=fiction_leaf,
        sources=urls,
        additional_instruction=(
            "Check the provided pages for explicit or clearly implied fiction categorization "
            "(e.g., 'novel', 'fiction', 'short stories', 'literary fiction'). "
            "Allow reasonable inferences if the page calls it a novel or a short story collection."
        )
    )

    # 3) Title exactness (as appears on official sources)
    title_leaf = evaluator.add_leaf(
        id=f"book_{idx+1}_title",
        desc=f"Book {idx+1} title is provided exactly as it appears in official sources",
        parent=book_node,
        critical=True
    )
    claim_title = (
        f"According to at least one of the provided reference URLs, the exact book title is '{title}'."
    )
    await evaluator.verify(
        claim=claim_title,
        node=title_leaf,
        sources=urls,
        additional_instruction=(
            "Compare the provided title to the title shown on the page. "
            "Treat case-insensitive matches and minor punctuation differences as acceptable. "
            "If a subtitle like 'A Novel' appears on the source but is missing or vice versa, "
            "consider it acceptable if the core title matches exactly (ignore the 'A Novel' subtitle term)."
        )
    )

    # 4) Author full name
    author_leaf = evaluator.add_leaf(
        id=f"book_{idx+1}_author",
        desc=f"Book {idx+1} author's full name is provided",
        parent=book_node,
        critical=True
    )
    claim_author = (
        f"According to at least one of the provided reference URLs, the author of '{title}' is '{author}'."
    )
    await evaluator.verify(
        claim=claim_author,
        node=author_leaf,
        sources=urls,
        additional_instruction=(
            "Allow minor variations such as middle initials or accents, but ensure it refers to the same person."
        )
    )

    # 5) Publisher correctness
    publisher_leaf = evaluator.add_leaf(
        id=f"book_{idx+1}_publisher",
        desc=f"Book {idx+1} publisher is correctly identified and verifiable",
        parent=book_node,
        critical=True
    )
    claim_publisher = (
        f"According to at least one of the provided reference URLs, the publisher (or imprint) of '{title}' is '{publisher}'."
    )
    await evaluator.verify(
        claim=claim_publisher,
        node=publisher_leaf,
        sources=urls,
        additional_instruction=(
            "Accept either the imprint name or the larger publishing house when the imprint belongs to that house. "
            "For example, 'Riverhead Books' under Penguin Random House. "
            "Pass if the page clearly supports the provided publisher/imprint."
        )
    )

    # 6) Publication date (month and year)
    pubdate_leaf = evaluator.add_leaf(
        id=f"book_{idx+1}_publication_date",
        desc=f"Book {idx+1} publication date (at minimum month and year) is provided",
        parent=book_node,
        critical=True
    )
    claim_pubdate = (
        f"According to at least one of the provided reference URLs, the publication month and year of '{title}' is '{pub_date}'."
    )
    await evaluator.verify(
        claim=claim_pubdate,
        node=pubdate_leaf,
        sources=urls,
        additional_instruction=(
            "Match on month and year. If the page shows a full date like 'May 7, 2024', "
            "accept 'May 2024' as correct. Allow minor formatting variations of month name."
        )
    )

    # 7) At least one allowed major award in 2024/2025
    award_leaf = evaluator.add_leaf(
        id=f"book_{idx+1}_award",
        desc=f"Book {idx+1} won at least one of the specified major literary awards (2024 or 2025)",
        parent=book_node,
        critical=True
    )
    primary_award = _select_primary_allowed_award(book)
    if primary_award:
        can_name, a_year = primary_award
        claim_award = (
            f"According to at least one of the provided reference URLs, '{title}' by {author} won the {can_name} in {a_year}."
        )
    else:
        # Fall back to a generic statement if extraction didn't capture a concrete allowed award-year pair
        claim_award = (
            f"According to at least one of the provided reference URLs, '{title}' by {author} won at least one of the following "
            f"awards in 2024 or 2025: {ALLOWED_AWARDS}."
        )
    await evaluator.verify(
        claim=claim_award,
        node=award_leaf,
        sources=urls,
        additional_instruction=(
            "Verify the award and year. Allow common name variants (e.g., 'The Booker Prize', 'Booker Prize'). "
            "For Pulitzer and National Book Award, ensure the Fiction category. "
            "For Carnegie, ensure it's the Medal for Excellence in Fiction. "
            "Pass if the page clearly supports that the book won an allowed award in 2024 or 2025."
        )
    )

    # 8) Award details (specific name(s) and year(s) stated)
    award_details_leaf = evaluator.add_leaf(
        id=f"book_{idx+1}_award_details",
        desc=f"Book {idx+1} specific award(s) won and year(s) are stated",
        parent=book_node,
        critical=True
    )
    if primary_award:
        can_name, a_year = primary_award
        claim_award_details = (
            f"At least one provided reference URL explicitly states that '{title}' by {author} won the {can_name} in {a_year}."
        )
    else:
        # If we don't have a clear pair, we still try with a generic claim asking the judge to find at least one name+year pair.
        pairs_str = "; ".join(
            [f"{(ai.award_name or '').strip()} ({(ai.award_year or '').strip()})" for ai in (book.awards or []) if (ai.award_name or ai.award_year)]
        )
        claim_award_details = (
            f"At least one provided reference URL states a concrete award+year for '{title}' by {author}. "
            f"Claimed pairs include: {pairs_str}. Pass if at least one name+year pair is supported."
        )
    await evaluator.verify(
        claim=claim_award_details,
        node=award_details_leaf,
        sources=urls,
        additional_instruction=(
            "This check focuses on explicit award name and year pairing as stated on the page. "
            "Pass if at least one award name together with its year is clearly supported on a provided page."
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
) -> Dict:
    """
    Evaluate an answer for the 2024/2025 award-winning fiction books task.
    """
    # Initialize evaluator (root is non-critical to allow non-critical children while still honoring critical checks)
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

    # Add helpful info for debugging/summary
    evaluator.add_custom_info(
        info={
            "allowed_awards": ALLOWED_AWARDS,
            "allowed_award_domains_examples": ALLOWED_AWARD_HINTS[:10],
            "timeframe_years": ["2024", "2025"]
        },
        info_type="config",
        info_name="award_task_config"
    )

    # Extract structured books
    extracted = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction"
    )

    # Ensure we have at least a list to operate on
    books = extracted.books if extracted and extracted.books else []
    # We'll evaluate only the first 4 for per-book checks (as per typical evaluation practice),
    # but we also check the count requirement exactly equals 4 below.
    first_four: List[BookItem] = (books + [BookItem()] * 4)[:4]

    # Per-book verifications (four books)
    for i in range(4):
        await verify_one_book(evaluator, root, first_four[i], i)

    # Global critical requirements (direct children of root so they can gate the overall result)

    # 1) Multiple awards requirement: at least one of the four books won multiple allowed awards (in 2024/2025)
    multiple_awards = any(_has_multiple_allowed_awards(b) for b in first_four)
    evaluator.add_custom_node(
        result=multiple_awards,
        id="multiple_awards_requirement",
        desc="At least one of the four books won multiple awards from the specified list (in 2024/2025)",
        parent=root,
        critical=True
    )

    # 2) Different publishers requirement: four different publishers across the four books
    unique_pub_count = _unique_nonempty_publishers(first_four)
    different_publishers = (unique_pub_count == 4)
    evaluator.add_custom_node(
        result=different_publishers,
        id="different_publishers_requirement",
        desc="The four books are published by four different publishers",
        parent=root,
        critical=True
    )

    # 3) Exactly four books provided in the answer
    exactly_four = (len(books) == 4)
    evaluator.add_custom_node(
        result=exactly_four,
        id="book_count_requirement",
        desc="Exactly 4 books are provided, no more and no fewer",
        parent=root,
        critical=True
    )

    # Add some diagnostics
    evaluator.add_custom_info(
        info={
            "extracted_book_count": len(books),
            "unique_publishers_in_first_four": unique_pub_count,
            "has_book_with_multiple_awards": multiple_awards
        },
        info_type="diagnostics",
        info_name="global_checks"
    )

    # Return structured evaluation summary
    return evaluator.get_summary()