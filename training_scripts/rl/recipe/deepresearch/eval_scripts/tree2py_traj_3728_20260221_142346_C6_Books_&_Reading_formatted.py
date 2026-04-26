import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "three_2024_award_winning_books"
TASK_DESCRIPTION = (
    "Identify three books that won major literary awards in 2024. For each book, provide the following information: "
    "(1) Book Identification: The complete title and the full name of the author; "
    "(2) Award Information: The name of the major literary award won, the specific category, and confirmation that the award was won in 2024; "
    "(3) Publisher Information: The name of the publisher who released the book; "
    "(4) Publication Details: The publication date (month and year) and the page count of the hardcover edition. "
    "For each piece of information, include a reference URL that confirms these details."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BookInfo(BaseModel):
    # Core values (all as strings for robustness)
    title: Optional[str] = None
    author: Optional[str] = None
    award_name: Optional[str] = None
    award_category: Optional[str] = None
    award_year: Optional[str] = None
    publisher_name: Optional[str] = None
    publication_date: Optional[str] = None  # Month and year preferred
    page_count: Optional[str] = None  # Hardcover page count

    # Field-specific confirming URLs (lists to allow multiple sources)
    title_urls: List[str] = Field(default_factory=list)
    author_urls: List[str] = Field(default_factory=list)
    award_name_urls: List[str] = Field(default_factory=list)
    award_category_urls: List[str] = Field(default_factory=list)
    award_year_urls: List[str] = Field(default_factory=list)
    publisher_name_urls: List[str] = Field(default_factory=list)
    publication_date_urls: List[str] = Field(default_factory=list)
    page_count_urls: List[str] = Field(default_factory=list)


class BooksExtraction(BaseModel):
    books: List[BookInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
    Extract up to three distinct books mentioned in the provided answer that the answer claims won major literary awards in 2024.
    For each book, extract the following fields:

    - title: The complete title of the book exactly as presented in the answer text.
    - author: The full name of the author as presented in the answer.
    - award_name: The name of the literary award the book purportedly won (e.g., "Pulitzer Prize", "National Book Award", "Booker Prize").
    - award_category: The specific category within that award (e.g., "Fiction", "Nonfiction", "Poetry", etc.).
    - award_year: The year stated in the answer for when the award was won (should be "2024" if provided).
    - publisher_name: The name of the publisher for this book (imprint names are acceptable as written).
    - publication_date: The publication date for the hardcover edition, preferably as "Month YYYY" (keep the exact wording from the answer; if a day is included, keep it as-is).
    - page_count: The hardcover edition page count (extract as a string; e.g., "320").

    For each of the above fields, also extract field-specific confirming URLs that the answer cites. Only include URLs that are explicitly present in the answer. If the same URL supports multiple fields, include it in each relevant URL list. Use the following URL fields:

    - title_urls: URLs confirming the title.
    - author_urls: URLs confirming the author.
    - award_name_urls: URLs confirming the award name (ideally a winners page or official announcement).
    - award_category_urls: URLs confirming the specific award category.
    - award_year_urls: URLs confirming the year the award was won (should indicate 2024 winners for this book).
    - publisher_name_urls: URLs confirming the publisher name.
    - publication_date_urls: URLs confirming the publication date of the hardcover edition.
    - page_count_urls: URLs confirming the hardcover page count.

    IMPORTANT RULES:
    1) Do NOT invent any URLs. Only extract URLs that are explicitly present in the answer.
    2) If a field value is not provided in the answer, set it to null.
    3) If no confirming URLs are provided for a field, return an empty list for that field's URLs.
    4) If more than three books are mentioned, only extract the first three in the order they appear.
    5) If fewer than three books are mentioned, return only what is available (the evaluator will handle missing slots).

    Return a JSON object with a top-level key "books" that is an array of at most three objects. Each object must contain all the fields described above, with either string values or null, and lists for URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _ordinal(n: int) -> str:
    return ["first", "second", "third", "fourth", "fifth"][n - 1] if 1 <= n <= 5 else f"#{n}"


async def _verify_with_urls_or_fail(
    evaluator: Evaluator,
    node_desc: str,
    leaf_id: str,
    parent,
    claim: Optional[str],
    urls: Optional[List[str]],
    additional_instruction: str,
    critical: bool = True
) -> None:
    leaf = evaluator.add_leaf(
        id=leaf_id,
        desc=node_desc,
        parent=parent,
        critical=critical
    )

    # Enforce source-grounding: if missing claim or missing URLs, fail immediately.
    if not claim or not claim.strip() or not urls or len(urls) == 0:
        leaf.score = 0.0
        leaf.status = "failed"
        return

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Verification logic per book                                                 #
# --------------------------------------------------------------------------- #
async def verify_book(
    evaluator: Evaluator,
    root_node,
    book: BookInfo,
    book_index: int
) -> None:
    idx = book_index + 1
    book_node = evaluator.add_parallel(
        id=f"Book_{idx}",
        desc=f"{_ordinal(idx).capitalize()} book meeting all award and publication criteria",
        parent=root_node,
        critical=False
    )

    # 1) Identification (Title + Author)
    ident_node = evaluator.add_parallel(
        id=f"Book_{idx}_Identification",
        desc=f"Basic identification information for the {_ordinal(idx)} book",
        parent=book_node,
        critical=True
    )

    # Title group
    title_group = evaluator.add_parallel(
        id=f"Book_{idx}_Title",
        desc="The exact title of the book",
        parent=ident_node,
        critical=True
    )
    await _verify_with_urls_or_fail(
        evaluator=evaluator,
        node_desc="URL reference confirming the book's title",
        leaf_id=f"Book_{idx}_Title_URL",
        parent=title_group,
        claim=f"The book's title is '{book.title}'." if (book.title and book.title.strip()) else None,
        urls=book.title_urls,
        additional_instruction="Verify that at least one provided URL explicitly shows the book's title. Allow minor punctuation or capitalization differences and the presence of subtitles, but it must clearly refer to the same book."
    )

    # Author group
    author_group = evaluator.add_parallel(
        id=f"Book_{idx}_Author",
        desc="The full name of the author",
        parent=ident_node,
        critical=True
    )
    await _verify_with_urls_or_fail(
        evaluator=evaluator,
        node_desc="URL reference confirming the author's name",
        leaf_id=f"Book_{idx}_Author_URL",
        parent=author_group,
        claim=f"The author of the book '{book.title}' is '{book.author}'." if (book.author and book.author.strip() and book.title) else None,
        urls=book.author_urls,
        additional_instruction="Verify that the provided URL shows the author for this exact book. Allow minor name variants (e.g., middle initials, accents)."
    )

    # 2) Award Information (Name, Category, Year)
    award_node = evaluator.add_parallel(
        id=f"Book_{idx}_Award_Information",
        desc=f"Information about the literary award won by the {_ordinal(idx)} book",
        parent=book_node,
        critical=True
    )

    # Award name
    award_name_group = evaluator.add_parallel(
        id=f"Book_{idx}_Award_Name",
        desc="The name of a major literary award won in 2024",
        parent=award_node,
        critical=True
    )
    await _verify_with_urls_or_fail(
        evaluator=evaluator,
        node_desc="URL reference confirming the award name",
        leaf_id=f"Book_{idx}_Award_Name_URL",
        parent=award_name_group,
        claim=(
            f"The book '{book.title}' by {book.author} won the '{book.award_name}' award."
            if (book.title and book.author and book.award_name) else None
        ),
        urls=book.award_name_urls,
        additional_instruction="Verify that the page explicitly states this book is a winner (not just nominated/shortlisted/finalist) of the specified award. The page should clearly link the title and the award name."
    )

    # Award category
    award_cat_group = evaluator.add_parallel(
        id=f"Book_{idx}_Award_Category",
        desc="The specific category of the award (e.g., Fiction, Nonfiction, etc.)",
        parent=award_node,
        critical=True
    )
    await _verify_with_urls_or_fail(
        evaluator=evaluator,
        node_desc="URL reference confirming the award category",
        leaf_id=f"Book_{idx}_Award_Category_URL",
        parent=award_cat_group,
        claim=(
            f"The book '{book.title}' won the '{book.award_name}' in the '{book.award_category}' category."
            if (book.title and book.award_name and book.award_category) else None
        ),
        urls=book.award_category_urls,
        additional_instruction="Verify the exact award category for which this book is listed as the winner. Accept close variants (e.g., 'Fiction' vs 'Best Fiction') if clearly the same category."
    )

    # Award year (must be 2024)
    award_year_group = evaluator.add_parallel(
        id=f"Book_{idx}_Award_Year",
        desc="The year the award was won must be 2024",
        parent=award_node,
        critical=True
    )
    # We verify explicitly for 2024 irrespective of extracted award_year field, but require URLs.
    year_claim = (
        f"The book '{book.title}' won the '{book.award_name}' in 2024."
        if (book.title and book.award_name) else None
    )
    await _verify_with_urls_or_fail(
        evaluator=evaluator,
        node_desc="URL reference confirming the award year",
        leaf_id=f"Book_{idx}_Award_Year_URL",
        parent=award_year_group,
        claim=year_claim,
        urls=book.award_year_urls,
        additional_instruction="Confirm that the page shows this book as a winner for the year 2024 (not a different year). If the page lists multiple years, it must clearly indicate 2024 for this book."
    )

    # 3) Publisher Information
    publisher_node = evaluator.add_parallel(
        id=f"Book_{idx}_Publisher_Information",
        desc=f"Publisher details for the {_ordinal(idx)} book",
        parent=book_node,
        critical=True
    )
    publisher_name_group = evaluator.add_parallel(
        id=f"Book_{idx}_Publisher_Name",
        desc="The name of the publisher",
        parent=publisher_node,
        critical=True
    )
    await _verify_with_urls_or_fail(
        evaluator=evaluator,
        node_desc="URL reference confirming the publisher name",
        leaf_id=f"Book_{idx}_Publisher_Name_URL",
        parent=publisher_name_group,
        claim=(
            f"The publisher of '{book.title}' is '{book.publisher_name}'."
            if (book.title and book.publisher_name) else None
        ),
        urls=book.publisher_name_urls,
        additional_instruction="Verify the publisher for this exact book/edition. If an imprint is shown (e.g., Knopf vs Alfred A. Knopf), consider it valid as long as it clearly refers to the same publishing entity/imprint."
    )

    # 4) Publication Details (Date and Hardcover Page Count)
    pub_details_node = evaluator.add_parallel(
        id=f"Book_{idx}_Publication_Details",
        desc=f"Detailed publication information for the {_ordinal(idx)} book",
        parent=book_node,
        critical=True
    )

    # Publication date
    pub_date_group = evaluator.add_parallel(
        id=f"Book_{idx}_Publication_Date",
        desc="The publication date (month and year)",
        parent=pub_details_node,
        critical=True
    )
    await _verify_with_urls_or_fail(
        evaluator=evaluator,
        node_desc="URL reference confirming the publication date",
        leaf_id=f"Book_{idx}_Publication_Date_URL",
        parent=pub_date_group,
        claim=(
            f"The publication date of the hardcover edition of '{book.title}' is {book.publication_date}."
            if (book.title and book.publication_date) else None
        ),
        urls=book.publication_date_urls,
        additional_instruction="Verify the publication date for the hardcover edition (preferably 'Month YYYY'). If a full date is shown, it must include the same month and year. If multiple formats are listed, prefer hardcover."
    )

    # Page count
    page_count_group = evaluator.add_parallel(
        id=f"Book_{idx}_Page_Count",
        desc="The total number of pages in the hardcover edition",
        parent=pub_details_node,
        critical=True
    )
    await _verify_with_urls_or_fail(
        evaluator=evaluator,
        node_desc="URL reference confirming the page count",
        leaf_id=f"Book_{idx}_Page_Count_URL",
        parent=page_count_group,
        claim=(
            f"The hardcover edition of '{book.title}' has {book.page_count} pages."
            if (book.title and book.page_count) else None
        ),
        urls=book.page_count_urls,
        additional_instruction="Verify the page count for the hardcover edition specifically. If multiple formats are shown, ensure the count corresponds to hardcover."
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
    # Initialize evaluator with parallel aggregation at the root
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

    # Extract structured information for books
    extracted = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction"
    )

    # Normalize to exactly three items (pad with empty entries if fewer)
    books: List[BookInfo] = list(extracted.books)[:3]
    while len(books) < 3:
        books.append(BookInfo())

    # Build verification subtrees for each book
    tasks = []
    for i in range(3):
        tasks.append(verify_book(evaluator, root, books[i], i))
    # Run verifications sequentially or concurrently; sequential is fine, but we can await all
    for t in tasks:
        await t

    # Return the complete evaluation summary
    return evaluator.get_summary()