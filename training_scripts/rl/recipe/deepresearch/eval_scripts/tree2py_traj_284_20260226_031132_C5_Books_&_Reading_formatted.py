import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "entertainment_memoirs_2014_2022"
TASK_DESCRIPTION = (
    "Identify four memoir or autobiography books written by entertainment personalities (actors, comedians, or TV hosts) "
    "that were published between 2014 and 2022 (inclusive). The four books must meet the following criteria: "
    "(1) One book must have been published by a publisher that was founded in the 19th century (1800-1899); "
    "(2) One book must have been published by a publisher that was founded in the 1950s (1950-1959); "
    "(3) One book must have been published by a publisher that was founded in the 21st century (2000 or later); "
    "(4) One book must include a foreword written by the author's spouse, where the spouse is also in the entertainment industry. "
    "For each book, provide: the complete book title, the author's name, the publication year, the publisher's name, the page count, "
    "and a reference URL supporting the information."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BookItem(BaseModel):
    """Structured info for a single book."""
    title: Optional[str] = None
    author: Optional[str] = None
    publication_year: Optional[str] = None  # Keep as string to handle formats like "2015 (hardcover)"
    publisher: Optional[str] = None
    page_count: Optional[str] = None  # String to allow ranges or formats like "320 pages"

    # Genre and author profession checks
    is_memoir_or_autobiography: Optional[bool] = None
    author_is_entertainment_personality: Optional[bool] = None
    author_personality_type: Optional[str] = None  # e.g., "actor", "comedian", "tv host"

    # URLs for evidence
    book_urls: List[str] = Field(default_factory=list)  # book pages: Amazon, publisher, Goodreads, etc.
    author_urls: List[str] = Field(default_factory=list)  # author bio pages: Wikipedia, official site
    publisher_founded_urls: List[str] = Field(default_factory=list)  # evidence for publisher founding year
    spouse_urls: List[str] = Field(default_factory=list)  # spouse bio pages for entertainment check
    foreword_urls: List[str] = Field(default_factory=list)  # evidence that spouse wrote the foreword

    # Publisher founding info (optional exact year; verification will focus on range)
    publisher_founded_year: Optional[str] = None

    # Foreword / spouse info
    foreword_by_spouse: Optional[bool] = None
    spouse_name: Optional[str] = None
    spouse_profession: Optional[str] = None


class BooksExtraction(BaseModel):
    """All four required category books."""
    book_19th: Optional[BookItem] = None  # Publisher founded in 19th century (1800-1899)
    book_1950s: Optional[BookItem] = None  # Publisher founded in 1950s (1950-1959)
    book_21st: Optional[BookItem] = None  # Publisher founded in 21st century (2000+)
    book_spouse_foreword: Optional[BookItem] = None  # Foreword by spouse who is also in entertainment


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
    From the provided answer, identify exactly four distinct memoir or autobiography books by entertainment personalities
    (actors, comedians, or TV hosts) that the answer claims were published between 2014 and 2022 (inclusive). Assign each
    book to one of the following four categories based on what the answer states:
      1) book_19th: Publisher was founded in the 19th century (1800-1899).
      2) book_1950s: Publisher was founded in the 1950s (1950-1959).
      3) book_21st: Publisher was founded in the 21st century (year 2000 or later).
      4) book_spouse_foreword: Book includes a foreword written by the author's spouse, and the spouse is also in the entertainment industry.

    IMPORTANT:
    - Use only information explicitly present in the answer.
    - If multiple books could fit a category, pick the first one mentioned that fits.
    - If the answer does not provide a book for a category, return null for that category.
    - Extract all URLs explicitly listed in the answer for the book, author, publisher founding info, and spouse/foreword info.

    For each selected book, extract the following fields (use null if missing):
      - title: complete book title exactly as stated
      - author: full author name
      - publication_year: the stated publication year (prefer a 4-digit year; if not given plainly, use the format in the answer)
      - publisher: publisher name
      - page_count: page count (string; examples: "320", "320 pages", "320 (hardcover)")
      - is_memoir_or_autobiography: true/false if the answer explicitly indicates memoir/autobiography/personal stories
      - author_is_entertainment_personality: true/false based on the answer (actor, comedian, TV host)
      - author_personality_type: one of "actor", "comedian", "tv host", or null if unspecified
      - book_urls: list of all URLs for the book's page(s) (Amazon, publisher page, Goodreads, etc.)
      - author_urls: list of author bio/reference URLs (Wikipedia, official site) if provided
      - publisher_founded_year: the founding year stated for the publisher (string or null)
      - publisher_founded_urls: list of URLs that support the publisher founding year (if any)
      - foreword_by_spouse: true/false based on the answer
      - spouse_name: name of the spouse who wrote the foreword (if stated)
      - spouse_profession: spouse profession (if stated), e.g., actor/comedian/tv host
      - spouse_urls: URLs about the spouse (if provided)
      - foreword_urls: URLs that support the foreword being written by the spouse (if provided)

    Return a JSON object with fields: book_19th, book_1950s, book_21st, book_spouse_foreword.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(*lists: Optional[List[str]]) -> List[str]:
    """Combine and deduplicate sources, keeping only non-empty strings."""
    seen = set()
    result: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if isinstance(url, str):
                u = url.strip()
                if u and u not in seen:
                    seen.add(u)
                    result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_book_with_requirements(
    evaluator: Evaluator,
    parent_node,
    book: Optional[BookItem],
    prefix: str,
    *,
    founding_requirement: Optional[str] = None,  # "19c", "1950s", "21c", or None
    require_spouse_foreword: bool = False
) -> None:
    """
    Build verification nodes and run checks for one book category.
    """
    # Create category node (non-critical to allow partial credit across categories)
    category_node = evaluator.add_parallel(
        id=f"{prefix}_main",
        desc=f"Verification for category '{prefix}'",
        parent=parent_node,
        critical=False
    )

    # Prepare sources
    book_urls = book.book_urls if book else []
    author_urls = book.author_urls if book else []
    publisher_urls = book.publisher_founded_urls if book else []
    foreword_urls = book.foreword_urls if book else []
    spouse_urls = book.spouse_urls if book else []

    all_sources = combine_sources(book_urls, author_urls, publisher_urls, foreword_urls, spouse_urls)

    # ---------------- Existence checks (critical/non-critical) ----------------
    evaluator.add_custom_node(
        result=bool(book and book.title and book.title.strip()),
        id=f"{prefix}_title_provided",
        desc="Complete book title is provided",
        parent=category_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(book and book.author and book.author.strip()),
        id=f"{prefix}_author_name_provided",
        desc="Author's name is provided",
        parent=category_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(book and book.publication_year and book.publication_year.strip()),
        id=f"{prefix}_publication_year_provided",
        desc="Publication year is provided",
        parent=category_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(book and book.publisher and book.publisher.strip()),
        id=f"{prefix}_publisher_name_provided",
        desc="Publisher's name is provided",
        parent=category_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(book and book.page_count and book.page_count.strip()),
        id=f"{prefix}_page_count_provided",
        desc="Page count is provided and verifiable",
        parent=category_node,
        critical=False  # Non-critical per rubric
    )

    evaluator.add_custom_node(
        result=bool(all_sources),
        id=f"{prefix}_reference_url_provided",
        desc="Supporting reference URL is provided",
        parent=category_node,
        critical=True
    )

    # ---------------- Author is entertainment personality (critical) ----------------
    author_ent_leaf = evaluator.add_leaf(
        id=f"{prefix}_author_entertainment_personality",
        desc="Author is an entertainment personality (actor, comedian, or TV host)",
        parent=category_node,
        critical=True
    )
    author_name = book.author if book and book.author else ""
    await evaluator.verify(
        claim=f"The author {author_name} is an entertainment personality (actor, comedian, or TV host).",
        node=author_ent_leaf,
        sources=combine_sources(author_urls, book_urls),
        additional_instruction="Check the provided URLs to confirm the author works as an actor, comedian, or TV host (or equivalent). Allow reasonable synonyms."
    )

    # ---------------- Publication Year range (critical) ----------------
    pub_range_leaf = evaluator.add_leaf(
        id=f"{prefix}_publication_year_range",
        desc="Book was published between 2014 and 2022 (inclusive)",
        parent=category_node,
        critical=True
    )
    title_for_claim = book.title if book and book.title else "the book"
    await evaluator.verify(
        claim=f"The book titled '{title_for_claim}' was published between 2014 and 2022 inclusive.",
        node=pub_range_leaf,
        sources=book_urls,
        additional_instruction=(
            "Verify the publication/release date shown on the book page. If multiple editions are listed, "
            "accept any edition published within 2014–2022 inclusive."
        )
    )

    # ---------------- Genre is memoir/autobiography/personal stories (critical) ----------------
    genre_leaf = evaluator.add_leaf(
        id=f"{prefix}_genre_memoir",
        desc="Book is a memoir, autobiography, or personal story collection",
        parent=category_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The book '{title_for_claim}' is a memoir or autobiography or a collection of personal stories.",
        node=genre_leaf,
        sources=book_urls,
        additional_instruction=(
            "Check classification, description, or genre tags. Accept synonyms like 'memoir', 'autobiography', "
            "'personal essays', 'personal stories'."
        )
    )

    # ---------------- Publisher founding constraints (critical for 3 categories) ----------------
    if founding_requirement in ("19c", "1950s", "21c"):
        req_map = {
            "19c": ("in the 19th century (between 1800 and 1899)", "Publisher was founded in the 19th century (1800-1899)"),
            "1950s": ("in the 1950s (1950–1959)", "Publisher was founded in the 1950s (1950-1959)"),
            "21c": ("in the 21st century (year 2000 or later)", "Publisher was founded in the 21st century (2000 or later)")
        }
        phrase, desc_text = req_map[founding_requirement]
        pub_range_leaf = evaluator.add_leaf(
            id=f"{prefix}_publisher_found_range",
            desc=desc_text,
            parent=category_node,
            critical=True
        )
        publisher_name = book.publisher if book and book.publisher else ""
        await evaluator.verify(
            claim=f"The publisher '{publisher_name}' was founded {phrase}.",
            node=pub_range_leaf,
            sources=publisher_urls if publisher_urls else all_sources,
            additional_instruction=(
                "Use the publisher profile/reference page(s). Confirm the founding year falls within the specified range. "
                "If multiple historical dates are mentioned, use the original founding date."
            )
        )

    # ---------------- Foreword by spouse (critical only for spouse category) ----------------
    if require_spouse_foreword:
        foreword_leaf = evaluator.add_leaf(
            id=f"{prefix}_foreword_by_spouse",
            desc="Book includes a foreword written by the author's spouse",
            parent=category_node,
            critical=True
        )
        spouse_name = book.spouse_name if book and book.spouse_name else "the author's spouse"
        await evaluator.verify(
            claim=f"The book '{title_for_claim}' includes a foreword written by {spouse_name}, who is the author's spouse.",
            node=foreword_leaf,
            sources=foreword_urls if foreword_urls else book_urls,
            additional_instruction=(
                "Check the front-matter information such as 'Foreword' credits. "
                "Allow reasonable synonyms (e.g., 'foreword', 'preface' labeled as foreword). "
                "It must be explicitly authored by the spouse."
            )
        )

        spouse_ent_leaf = evaluator.add_leaf(
            id=f"{prefix}_spouse_entertainment_industry",
            desc="The spouse who wrote the foreword is also in the entertainment industry",
            parent=category_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"{spouse_name} is an entertainment industry personality (e.g., actor, comedian, TV host or similar).",
            node=spouse_ent_leaf,
            sources=combine_sources(spouse_urls, foreword_urls, book_urls),
            additional_instruction="Confirm the spouse's entertainment profession using the provided URLs. Allow reasonable synonyms and roles."
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
    Evaluate an answer for the entertainment memoir/autobiography task.
    """
    # Initialize evaluator (root represents Task Completion)
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

    # Extract structured information
    extracted: BooksExtraction = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction"
    )

    # Build category nodes directly under root (as per rubric)
    # 1) 19th century publisher
    await verify_book_with_requirements(
        evaluator=evaluator,
        parent_node=root,
        book=extracted.book_19th,
        prefix="Book_19th_Century_Publisher",
        founding_requirement="19c",
        require_spouse_foreword=False
    )

    # 2) 1950s publisher
    await verify_book_with_requirements(
        evaluator=evaluator,
        parent_node=root,
        book=extracted.book_1950s,
        prefix="Book_1950s_Publisher",
        founding_requirement="1950s",
        require_spouse_foreword=False
    )

    # 3) 21st century publisher
    await verify_book_with_requirements(
        evaluator=evaluator,
        parent_node=root,
        book=extracted.book_21st,
        prefix="Book_21st_Century_Publisher",
        founding_requirement="21c",
        require_spouse_foreword=False
    )

    # 4) Spouse foreword category
    await verify_book_with_requirements(
        evaluator=evaluator,
        parent_node=root,
        book=extracted.book_spouse_foreword,
        prefix="Book_With_Spouse_Foreword",
        founding_requirement=None,
        require_spouse_foreword=True
    )

    # Optional: Add custom info summarizing constraints
    evaluator.add_custom_info(
        info={
            "required_categories": [
                "Publisher founded in 19th century (1800-1899)",
                "Publisher founded in 1950s (1950-1959)",
                "Publisher founded in 21st century (2000 or later)",
                "Foreword written by author's spouse (spouse in entertainment)"
            ],
            "publication_year_range": "2014–2022 (inclusive)",
            "author_requirement": "Actor, comedian, or TV host",
            "genre_requirement": "Memoir/autobiography/personal stories"
        },
        info_type="task_requirements",
        info_name="memoir_autobiography_requirements"
    )

    # Return evaluation summary
    return evaluator.get_summary()