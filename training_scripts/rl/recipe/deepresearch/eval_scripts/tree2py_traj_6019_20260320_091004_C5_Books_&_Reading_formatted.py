import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "books_awards_2025"
TASK_DESCRIPTION = """
Identify three fiction books that each won a different major literary award in 2025. The awards to consider are:
- the Pulitzer Prize for Fiction,
- the National Book Award for Fiction,
- the Booker Prize,
- the Goodreads Choice Award for Fiction.

For each book, provide:
1) Book Title and Author (full as appears on the book)
2) Award Won (one of the above, year 2025)
3) Publisher
4) Author's Goodreads Profile (URL)
5) Author's Nationality (if publicly available)
6) Publication Year (must be 2024 or 2025)
7) Goodreads Rating (must be 4.0 or higher)
8) Reading Group Guide (URL from publisher/major retailer/reputable site)
9) Supporting URL(s) (at least one URL verifying the award win from official award org or major news)

Each entry must be a work of fiction, and each must have a different award from the above list.
"""

ALLOWED_AWARDS = [
    "Pulitzer Prize for Fiction",
    "National Book Award for Fiction",
    "Booker Prize",
    "Goodreads Choice Award for Fiction",
]

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BookItem(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    award: Optional[str] = None
    publisher: Optional[str] = None

    # URLs and optional details
    goodreads_author_profile_url: Optional[str] = None
    author_nationality: Optional[str] = None

    publication_year: Optional[str] = None  # keep as string for robustness
    goodreads_rating: Optional[str] = None

    reading_group_guide_url: Optional[str] = None
    award_supporting_urls: List[str] = Field(default_factory=list)

    # Helpful extra sources for verification (optional but useful)
    book_goodreads_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


class BooksExtraction(BaseModel):
    books: List[BookItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
    Extract up to 5 candidate book entries from the provided answer text (do NOT invent any).
    For each book, return the following fields (use null if missing; use empty list [] for arrays when missing):
    - title: Full title of the book as stated in the answer.
    - author: Full author name as stated in the answer.
    - award: The specific award string as stated (e.g., "2025 Booker Prize", "Pulitzer Prize for Fiction 2025", etc.).
    - publisher: The publisher name as stated in the answer.
    - goodreads_author_profile_url: Direct URL to the author's Goodreads profile page (if provided).
    - author_nationality: Nationality or country of residence as stated (if provided).
    - publication_year: The original publication year (as shown/claimed in the answer).
    - goodreads_rating: The book's current average rating on Goodreads as provided in the answer (string as-is).
    - reading_group_guide_url: URL to a reading group guide/book club kit/discussion guide page (publisher/major retailer/reputable site).
    - award_supporting_urls: Array of URLs that directly verify the award win (prefer official award website or major news).
    - book_goodreads_url: The Goodreads URL for the book (if any).
    - additional_urls: Any other URLs cited in the answer that support the information (e.g., publisher book page, retailer page, etc.).

    Important:
    - Only extract URLs explicitly present in the answer (plain link or markdown). Do not infer or fabricate URLs.
    - Preserve the strings exactly as they appear for names and awards.
    - If multiple items are present in the answer, include them in order of appearance.
    - Do not deduplicate or normalize; that will be handled later.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _valid_url(u: Optional[str]) -> bool:
    if not u:
        return False
    u = u.strip()
    return u.startswith("http://") or u.startswith("https://")


def _collect_book_sources(book: BookItem) -> List[str]:
    urls: List[str] = []
    if _valid_url(book.book_goodreads_url):
        urls.append(book.book_goodreads_url)  # Goodreads book page (best for rating/genre/year)
    if _valid_url(book.reading_group_guide_url):
        urls.append(book.reading_group_guide_url)  # reading guide / book club kit
    if _valid_url(book.goodreads_author_profile_url):
        urls.append(book.goodreads_author_profile_url)  # author profile page
    if book.award_supporting_urls:
        urls.extend([u for u in book.award_supporting_urls if _valid_url(u)])
    if book.additional_urls:
        urls.extend([u for u in book.additional_urls if _valid_url(u)])
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _canonicalize_award(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.lower()

    if ("pulitzer" in s) and ("fiction" in s):
        return "Pulitzer Prize for Fiction"
    if ("national book award" in s) and ("fiction" in s):
        return "National Book Award for Fiction"
    if "booker" in s:
        # Accept "The Booker Prize", "Booker Prize 2025", etc.
        return "Booker Prize"
    if ("goodreads" in s) and ("fiction" in s):
        # Accept "Goodreads Choice Awards (Best Fiction)" variants
        return "Goodreads Choice Award for Fiction"
    return None


def _parse_year_safe(year_str: Optional[str]) -> Optional[int]:
    if not year_str:
        return None
    s = year_str.strip()
    # Try to find a 4-digit year in the string
    import re
    m = re.search(r"(20\d{2})", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification for a single book                                              #
# --------------------------------------------------------------------------- #
async def verify_single_book(
    evaluator: Evaluator,
    parent_node,
    book: BookItem,
    book_index: int,
) -> None:
    """
    Build the verification sub-tree for one book and run verifications.
    """
    book_node = evaluator.add_parallel(
        id=f"book_{book_index+1}",
        desc=f"{['First','Second','Third','Fourth','Fifth'][book_index] if book_index<5 else f'#{book_index+1}'} award-winning fiction book with all required information",
        parent=parent_node,
        critical=False,
    )

    # Pre-check node for URL coverage to gate later checks
    has_award_src = bool([u for u in (book.award_supporting_urls or []) if _valid_url(u)])
    has_reading_guide = _valid_url(book.reading_group_guide_url)
    has_author_profile = _valid_url(book.goodreads_author_profile_url)
    has_book_gr = _valid_url(book.book_goodreads_url)

    url_refs_node = evaluator.add_custom_node(
        result=has_award_src and has_reading_guide and has_author_profile and has_book_gr,
        id=f"book_{book_index+1}_url_references",
        desc="All information is supported by publicly accessible URLs from reputable sources",
        parent=book_node,
        critical=True
    )

    # 1) Title existence
    title_exists = bool(book.title and book.title.strip())
    evaluator.add_custom_node(
        result=title_exists,
        id=f"book_{book_index+1}_book_title",
        desc="Full title of the book is provided",
        parent=book_node,
        critical=True
    )

    # Author information group (set as non-critical to allow mixed critical children)
    author_info_node = evaluator.add_parallel(
        id=f"book_{book_index+1}_author_information",
        desc="Complete author information is provided",
        parent=book_node,
        critical=False
    )

    # 2) Author name existence
    author_exists = bool(book.author and book.author.strip())
    evaluator.add_custom_node(
        result=author_exists,
        id=f"book_{book_index+1}_author_name",
        desc="Author's full name as it appears on the book is provided",
        parent=author_info_node,
        critical=True
    )

    # 3) Goodreads author profile URL correctness
    goodreads_profile_leaf = evaluator.add_leaf(
        id=f"book_{book_index+1}_goodreads_profile",
        desc="Link to author's Goodreads profile page is provided",
        parent=author_info_node,
        critical=True
    )
    profile_claim_author = book.author or "the author"
    await evaluator.verify(
        claim=f"This page is the Goodreads author profile page of {profile_claim_author}.",
        node=goodreads_profile_leaf,
        sources=book.goodreads_author_profile_url if _valid_url(book.goodreads_author_profile_url) else None,
        additional_instruction="Verify that the URL is an author profile page on goodreads.com (commonly '/author/show/...' pattern) and that the displayed author name matches (allow minor variations)."
    )

    # 4) Author nationality (optional / non-critical)
    author_nat_leaf = evaluator.add_leaf(
        id=f"book_{book_index+1}_author_nationality",
        desc="Author's nationality or country of residence is provided if publicly available",
        parent=author_info_node,
        critical=False
    )
    if book.author_nationality and book.author_nationality.strip():
        sources_for_nat = []
        if _valid_url(book.goodreads_author_profile_url):
            sources_for_nat.append(book.goodreads_author_profile_url)
        sources_for_nat.extend([u for u in (book.additional_urls or []) if _valid_url(u)])
        await evaluator.verify(
            claim=f"{profile_claim_author}'s nationality or country of residence is '{book.author_nationality}'.",
            node=author_nat_leaf,
            sources=sources_for_nat if sources_for_nat else None,
            additional_instruction="Confirm the nationality/residence from the author profile page or another reputable source (e.g., publisher bio, Wikipedia). Allow reasonable wording variants (e.g., 'British' vs 'United Kingdom')."
        )
    else:
        # No claim possible -> mark as failed due to missing info
        # Use simple_verify to explicitly set to failed
        await evaluator.verify(
            claim="The author's nationality is provided.",
            node=author_nat_leaf,
            sources=None,
            additional_instruction="Fail this check if the nationality/country is not provided in the answer."
        )

    # Assemble common sources for book-level checks
    common_sources = _collect_book_sources(book)

    # 5) Genre verification (must be fiction)
    genre_leaf = evaluator.add_leaf(
        id=f"book_{book_index+1}_genre_verification",
        desc="Book is fiction (not non-fiction, poetry, or young adult)",
        parent=book_node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The book "{book.title or "this book"}" is a work of fiction (e.g., a novel or short-story collection), and it is not non-fiction, poetry, or young adult.',
        node=genre_leaf,
        sources=common_sources if common_sources else None,
        additional_instruction="Look for genre/category labels like 'Fiction', 'Novel', 'Literary Fiction', etc., on the Goodreads book page, publisher page, or reading guide. Reject if it is clearly labeled 'Nonfiction', 'Poetry', or 'Young Adult'."
    )

    # 6) Award verification (won one of the specified awards in 2025)
    award_leaf = evaluator.add_leaf(
        id=f"book_{book_index+1}_award_verification",
        desc="Book won one of the four specified major literary awards in 2025 (Pulitzer Prize for Fiction, National Book Award for Fiction, Booker Prize, or Goodreads Choice Award for Fiction)",
        parent=book_node,
        critical=True
    )
    award_name = book.award or ""
    await evaluator.verify(
        claim=f'The book "{book.title or "this book"}" by {book.author or "the author"} won the {award_name} in 2025, and this is one of the allowed awards (Pulitzer Prize for Fiction, National Book Award for Fiction, Booker Prize, or Goodreads Choice Award for Fiction).',
        node=award_leaf,
        sources=[u for u in (book.award_supporting_urls or []) if _valid_url(u)],
        additional_instruction="Confirm the award and year explicitly from the provided URL(s). Prefer the official award site or a major news outlet. Ensure the category matches Fiction when applicable (e.g., National Book Award for Fiction; Goodreads 'Best Fiction')."
    )

    # 7) Publisher verification
    publisher_leaf = evaluator.add_leaf(
        id=f"book_{book_index+1}_publisher_info",
        desc="Publisher name is correctly identified and verifiable",
        parent=book_node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The publisher of "{book.title or "this book"}" is "{book.publisher or "(publisher missing)"}".',
        node=publisher_leaf,
        sources=common_sources if common_sources else None,
        additional_instruction="Prefer the publisher site or reading group guide. Goodreads book page is acceptable if publisher is shown. Allow imprints to count as the publisher."
    )

    # 8) Publication year is 2024 or 2025 (and matches stated)
    pub_year_leaf = evaluator.add_leaf(
        id=f"book_{book_index+1}_publication_date",
        desc="Original publication year is 2024 or 2025",
        parent=book_node,
        critical=True
    )
    yr = _parse_year_safe(book.publication_year)
    year_clause = f"{yr}" if yr is not None else "an unknown year"
    await evaluator.verify(
        claim=f'The original publication year of "{book.title or "this book"}" is {year_clause}, and it is 2024 or 2025.',
        node=pub_year_leaf,
        sources=common_sources if common_sources else None,
        additional_instruction="Use the 'first published' or original publication year if multiple dates are shown. Consider regional editions but focus on original publication. Accept only if the original year is 2024 or 2025."
    )

    # 9) Goodreads rating >= 4.0
    rating_leaf = evaluator.add_leaf(
        id=f"book_{book_index+1}_goodreads_rating",
        desc="Book has a Goodreads average rating of 4.0 or higher",
        parent=book_node,
        critical=True
    )
    await evaluator.verify(
        claim=f'The current Goodreads average rating for "{book.title or "this book"}" is at least 4.0 (4.0 or higher).',
        node=rating_leaf,
        sources=book.book_goodreads_url if _valid_url(book.book_goodreads_url) else None,
        additional_instruction="Use the Goodreads book page to check the displayed average rating. Allow for rounding to one decimal place. If exactly 3.95–3.99 is displayed, this should not pass."
    )

    # 10) Reading group guide URL correctness
    guide_leaf = evaluator.add_leaf(
        id=f"book_{book_index+1}_reading_guide",
        desc="Reading group guide or book club discussion guide is available and link is provided",
        parent=book_node,
        critical=True
    )
    await evaluator.verify(
        claim=f'This page is a reading group guide/book club kit/discussion guide for "{book.title or "this book"}".',
        node=guide_leaf,
        sources=book.reading_group_guide_url if _valid_url(book.reading_group_guide_url) else None,
        additional_instruction="Look for explicit cues like 'Reading Group Guide', 'Book Club Kit', 'Discussion Guide'. Accept pages from the publisher, major book retailers, or reputable book sites."
    )


# --------------------------------------------------------------------------- #
# Root-level unique awards verification                                       #
# --------------------------------------------------------------------------- #
def _unique_awards_three(books: List[BookItem]) -> bool:
    # Use first three books only
    cand = books[:3]
    mapped = []
    for b in cand:
        mapped.append(_canonicalize_award(b.award))
    # All must be recognized and distinct
    if any(m is None for m in mapped):
        return False
    return len(set(mapped)) == 3


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
    Evaluate an answer for the 2025 award-winning fiction books task.
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

    # Extract structured book info
    extracted = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction"
    )

    # Keep first 3 items; pad to exactly 3
    books: List[BookItem] = list(extracted.books[:3])
    while len(books) < 3:
        books.append(BookItem())

    # Add allowed awards into the summary as custom info
    evaluator.add_custom_info({"allowed_awards": ALLOWED_AWARDS}, info_type="config", info_name="allowed_awards")

    # Root-level critical unique awards check
    evaluator.add_custom_node(
        result=_unique_awards_three(books),
        id="unique_awards",
        desc="All three books won different awards from the specified list",
        parent=root,
        critical=True
    )

    # Build and verify each book subtree
    verify_tasks = []
    for i in range(3):
        verify_tasks.append(verify_single_book(evaluator, root, books[i], i))

    await asyncio.gather(*verify_tasks)

    return evaluator.get_summary()