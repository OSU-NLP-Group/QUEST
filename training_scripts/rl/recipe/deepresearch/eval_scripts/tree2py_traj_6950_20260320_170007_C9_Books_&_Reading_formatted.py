import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "lit_awards_2024_big5_books"
TASK_DESCRIPTION = """
Find three literary fiction books that were finalists for major US literary awards in 2024. Specifically, each book must have been a finalist (or winner) for the National Book Award for Fiction, the Pulitzer Prize for Fiction, or the Booker Prize in 2024. Additionally, each book must have been published by one of the Big Five publishing houses (Penguin Random House, HarperCollins, Simon & Schuster, Hachette, or Macmillan) or their imprints during 2024. For each book, provide: (1) The complete book title and author name, (2) The specific award for which it was a finalist or winner in 2024, (3) A direct link to the official award website page confirming its finalist or winner status, (4) The publisher or imprint name, (5) Confirmation that this publisher/imprint is part of one of the Big Five houses, (6) A link confirming the publisher information, (7) The publication date (month and year) in 2024, (8) The ISBN number, (9) The page count, (10) Confirmation that the book was released in hardcover format, (11) Whether the author is a debut novelist or an established author with previous books, and (12) A link with information about the author.
"""

BIG_FIVE_LIST = [
    "Penguin Random House",
    "HarperCollins",
    "Simon & Schuster",
    "Hachette",
    "Macmillan",
]

ALLOWED_AWARDS_2024 = [
    "National Book Award for Fiction",
    "Pulitzer Prize for Fiction",
    "Booker Prize",
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BookItem(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    award_name: Optional[str] = None
    award_result: Optional[str] = None  # "Finalist" or "Winner"
    award_url: Optional[str] = None

    publisher: Optional[str] = None           # Publisher or imprint
    big_five_parent: Optional[str] = None     # One of BIG_FIVE_LIST, if provided
    publisher_url: Optional[str] = None

    publication_date: Optional[str] = None    # Prefer "Month YYYY"
    isbn: Optional[str] = None
    page_count: Optional[str] = None          # Keep as string to maximize robustness
    hardcover: Optional[str] = None           # "yes"/"no" or similar

    author_status: Optional[str] = None       # "debut" or "established"
    author_url: Optional[str] = None

    extra_urls: List[str] = Field(default_factory=list)  # any other supporting links mentioned


class BooksExtraction(BaseModel):
    books: List[BookItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
    You will extract up to three books from the answer that meet ALL of the following:
    - Literary fiction books that were finalists or winners in 2024 for one of:
      • National Book Award for Fiction
      • Pulitzer Prize for Fiction
      • Booker Prize
    - Published by a Big Five publisher or one of their imprints during 2024
      • Big Five list: Penguin Random House, HarperCollins, Simon & Schuster, Hachette, Macmillan

    Return a JSON object with the field:
    - books: an array containing at most the first three qualifying books mentioned in the answer.
      For each book include EXACTLY the following fields (use null when missing in the answer):

      1) title: Full book title as shown in the answer
      2) author: Author's full name as shown
      3) award_name: The specific award name as claimed in the answer; prefer exactly one of:
         "National Book Award for Fiction", "Pulitzer Prize for Fiction", or "Booker Prize"
         (If the answer uses a close variant like "National Book Awards (Fiction)", normalize to the closest above; if unclear, keep the original text.)
      4) award_result: "Finalist" or "Winner" (if the answer states it; otherwise null)
      5) award_url: A direct URL cited in the answer to the official award webpage that confirms finalist/winner status.
         If multiple are provided, choose the most direct official page for that specific book entry.
      6) publisher: Publisher or imprint name as stated
      7) big_five_parent: If the answer explicitly states the Big Five parent (e.g., "an imprint of Penguin Random House"),
         extract that exact parent name; otherwise null. Prefer one of:
         ["Penguin Random House", "HarperCollins", "Simon & Schuster", "Hachette", "Macmillan"].
      8) publisher_url: A URL in the answer that confirms the publisher/imprint info for this book;
         ideally the official publisher or imprint page for the book.
      9) publication_date: Month and year in 2024 as stated in the answer (e.g., "April 2024").
         If the answer gives a full date, keep it as "Month YYYY" while preserving the month and 2024.
      10) isbn: The ISBN mentioned in the answer (ISBN-13 preferred, but accept ISBN-10 if that's all the answer has).
      11) page_count: The page count mentioned (digits only if possible; otherwise keep as-is).
      12) hardcover: "yes" if the answer explicitly confirms a hardcover edition exists (or was released), "no" if it denies it, else null.
      13) author_status: "debut" if the answer says it's their first novel; otherwise "established" if prior books are mentioned; else null if unknown.
      14) author_url: A URL cited in the answer that provides information about the author (publisher bio, author site, Wikipedia, etc.).
      15) extra_urls: Any other URLs cited in the answer related to this book (array; can be empty).

    General rules:
    - Extract only from the provided answer. Do not invent or search for new URLs or facts.
    - Keep strings exactly as they appear (except for normalizing award_name to the allowed set if the intent is obvious).
    - If multiple books are included in the answer, return the first three that appear to meet the task intent.
    - If fewer than three books are present, return as many as you can find (1 or 2), and omit the rest.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _filter_valid_urls(urls: List[Optional[str]]) -> List[str]:
    return [u for u in urls if isinstance(u, str) and u.strip()]


def _has_value(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


# --------------------------------------------------------------------------- #
# Verification for a single book                                              #
# --------------------------------------------------------------------------- #
async def verify_book(evaluator: Evaluator, parent_node, book: BookItem, index_1based: int) -> None:
    """
    Build the verification subtree and run checks for one book.
    Tree structure strictly follows the rubric provided.
    """
    # Book-level container (non-critical to allow partial credit across books)
    book_node = evaluator.add_parallel(
        id=f"book_{index_1based}",
        desc=[
            "First qualifying book identified and verified",
            "Second qualifying book identified and verified",
            "Third qualifying book identified and verified",
        ][index_1based - 1],
        parent=parent_node,
        critical=False,
    )

    # ------------------------------ Award status ------------------------------ #
    award_node = evaluator.add_parallel(
        id=f"award_status_{index_1based}",
        desc="Book was a finalist for National Book Award, Pulitzer Prize for Fiction, or Booker Prize in 2024",
        parent=book_node,
        critical=True,
    )

    # Leaf: Official award website URL provided (existence check)
    evaluator.add_custom_node(
        result=_has_value(book.award_url),
        id=f"award_reference_url_{index_1based}",
        desc="Official award website URL provided confirming finalist/winner status",
        parent=award_node,
        critical=True,
    )

    # Leaf: Finalist/winner status confirmed through official award source
    award_verify_leaf = evaluator.add_leaf(
        id=f"award_finalist_verification_{index_1based}",
        desc="Finalist or winner status confirmed through official award source",
        parent=award_node,
        critical=True,
    )
    if _has_value(book.title) and _has_value(book.author) and _has_value(book.award_name) and _has_value(book.award_url):
        award_claim = (
            f"'{book.title}' by {book.author} was a finalist or winner for the {book.award_name} in 2024."
        )
        award_add_ins = (
            "Verify on the provided official award webpage that this exact book and author were either a Finalist or "
            "the Winner in 2024. Do NOT accept 'longlisted' or 'nominated' if not listed as 'finalist' or 'winner'. "
            "Treat reasonable naming variants as equivalent (e.g., 'shortlisted' for Booker equals 'finalist')."
        )
        await evaluator.verify(
            claim=award_claim,
            node=award_verify_leaf,
            sources=book.award_url,
            additional_instruction=award_add_ins,
        )
    else:
        award_verify_leaf.score = 0.0
        award_verify_leaf.status = "failed"

    # --------------------------- Publication details -------------------------- #
    pub_details_node = evaluator.add_parallel(
        id=f"publication_details_{index_1based}",
        desc="Publication metadata correctly identified",
        parent=book_node,
        critical=True,
    )

    # Leaf: Title and author
    title_author_leaf = evaluator.add_leaf(
        id=f"title_and_author_{index_1based}",
        desc="Correct book title and author name provided",
        parent=pub_details_node,
        critical=True,
    )
    ta_sources = _filter_valid_urls([book.award_url, book.publisher_url])
    if _has_value(book.title) and _has_value(book.author) and len(ta_sources) > 0:
        ta_claim = f"The book title is '{book.title}' and the author is '{book.author}'."
        ta_add_ins = (
            "Check the provided pages for the exact or equivalent title and author. "
            "Allow minor punctuation, subtitle, capitalization, or diacritics differences."
        )
        await evaluator.verify(
            claim=ta_claim,
            node=title_author_leaf,
            sources=ta_sources if len(ta_sources) > 1 else ta_sources[0],
            additional_instruction=ta_add_ins,
        )
    else:
        title_author_leaf.score = 0.0
        title_author_leaf.status = "failed"

    # Leaf: Publication date in 2024
    pubdate_leaf = evaluator.add_leaf(
        id=f"publication_date_{index_1based}",
        desc="Publication date in 2024 confirmed",
        parent=pub_details_node,
        critical=True,
    )
    pd_sources = _filter_valid_urls([book.publisher_url, book.award_url])
    if _has_value(book.publication_date) and len(pd_sources) > 0:
        pubdate_claim = f"The publication date of '{book.title}' is {book.publication_date}, and it is in 2024."
        pubdate_add_ins = (
            "Prefer the publisher page. If the page shows a full date (e.g., 'April 2, 2024'), this still satisfies "
            "the 'in 2024' requirement. If multiple editions exist, prefer the U.S. hardcover date. "
            "Do not accept 2023 or earlier/2025 or later dates."
        )
        await evaluator.verify(
            claim=pubdate_claim,
            node=pubdate_leaf,
            sources=pd_sources if len(pd_sources) > 1 else pd_sources[0],
            additional_instruction=pubdate_add_ins,
        )
    else:
        pubdate_leaf.score = 0.0
        pubdate_leaf.status = "failed"

    # Leaf: ISBN
    isbn_leaf = evaluator.add_leaf(
        id=f"isbn_{index_1based}",
        desc="Valid ISBN provided",
        parent=pub_details_node,
        critical=True,
    )
    isbn_sources = _filter_valid_urls([book.publisher_url, book.award_url])
    if _has_value(book.isbn) and len(isbn_sources) > 0:
        isbn_claim = f"The ISBN of '{book.title}' is {book.isbn}."
        isbn_add_ins = (
            "Match ISBN-13 or ISBN-10 as listed; ignore hyphens/spaces when comparing. "
            "If multiple ISBNs for different formats are present, pass if any ISBN exactly matches the claimed one "
            "(ignoring hyphens/spaces). Prefer hardcover ISBN if specified."
        )
        await evaluator.verify(
            claim=isbn_claim,
            node=isbn_leaf,
            sources=isbn_sources if len(isbn_sources) > 1 else isbn_sources[0],
            additional_instruction=isbn_add_ins,
        )
    else:
        isbn_leaf.score = 0.0
        isbn_leaf.status = "failed"

    # Leaf: Page count ≥ 200
    pagecount_leaf = evaluator.add_leaf(
        id=f"page_count_{index_1based}",
        desc="Page count of at least 200 pages confirmed",
        parent=pub_details_node,
        critical=True,
    )
    pc_sources = _filter_valid_urls([book.publisher_url, book.award_url])
    if _has_value(book.page_count) and len(pc_sources) > 0:
        pagecount_claim = (
            f"'{book.title}' has {book.page_count} pages (for the primary/print or hardcover edition), "
            f"which is at least 200 pages."
        )
        pagecount_add_ins = (
            "Prefer the publisher page. If multiple editions show slightly different counts, use the hardcover or "
            "main print edition. Pass if the listed page count is ≥ 200. Fail if the page shows < 200."
        )
        await evaluator.verify(
            claim=pagecount_claim,
            node=pagecount_leaf,
            sources=pc_sources if len(pc_sources) > 1 else pc_sources[0],
            additional_instruction=pagecount_add_ins,
        )
    else:
        pagecount_leaf.score = 0.0
        pagecount_leaf.status = "failed"

    # ------------------------ Publisher verification ------------------------- #
    publisher_node = evaluator.add_parallel(
        id=f"publisher_verification_{index_1based}",
        desc="Publisher is confirmed as Big Five imprint",
        parent=book_node,
        critical=True,
    )

    # Leaf: Publisher or imprint name correctly identified
    publisher_name_leaf = evaluator.add_leaf(
        id=f"publisher_name_{index_1based}",
        desc="Publisher or imprint name correctly identified",
        parent=publisher_node,
        critical=True,
    )
    pubname_sources = _filter_valid_urls([book.publisher_url, book.award_url])
    if _has_value(book.publisher) and len(pubname_sources) > 0:
        pubname_claim = f"The publisher/imprint of '{book.title}' is '{book.publisher}'."
        pubname_add_ins = (
            "Verify on the publisher (or imprint) page for the specific book. "
            "If multiple imprints/lines are mentioned, pass if the claimed name appears as the publisher/imprint."
        )
        await evaluator.verify(
            claim=pubname_claim,
            node=publisher_name_leaf,
            sources=pubname_sources if len(pubname_sources) > 1 else pubname_sources[0],
            additional_instruction=pubname_add_ins,
        )
    else:
        publisher_name_leaf.score = 0.0
        publisher_name_leaf.status = "failed"

    # Leaf: Big Five affiliation confirmed
    bigfive_leaf = evaluator.add_leaf(
        id=f"big_five_affiliation_{index_1based}",
        desc="Publisher confirmed as part of Penguin Random House, HarperCollins, Simon & Schuster, Hachette, or Macmillan",
        parent=publisher_node,
        critical=True,
    )
    bigfive_sources = _filter_valid_urls([book.publisher_url])
    if len(bigfive_sources) > 0 and _has_value(book.publisher):
        parent_hint = book.big_five_parent if _has_value(book.big_five_parent) else "one of the Big Five"
        bigfive_claim = (
            f"The publisher/imprint '{book.publisher}' is part of {parent_hint} "
            f"(Penguin Random House, HarperCollins, Simon & Schuster, Hachette, or Macmillan)."
        )
        bigfive_add_ins = (
            "Look for language like 'an imprint of' or corporate ownership indicating membership in one of the Big Five. "
            "Mentions of PRH, HarperCollins, Simon & Schuster, Hachette (or Hachette Book Group), or Macmillan count. "
            "Pass if the page clearly indicates affiliation with any one of these five groups."
        )
        await evaluator.verify(
            claim=bigfive_claim,
            node=bigfive_leaf,
            sources=bigfive_sources[0] if len(bigfive_sources) == 1 else bigfive_sources,
            additional_instruction=bigfive_add_ins,
        )
    else:
        bigfive_leaf.score = 0.0
        bigfive_leaf.status = "failed"

    # Leaf: URL confirming publisher info provided (existence check)
    evaluator.add_custom_node(
        result=_has_value(book.publisher_url),
        id=f"publisher_reference_url_{index_1based}",
        desc="URL confirming publisher information provided",
        parent=publisher_node,
        critical=True,
    )

    # --------------------------- Genre and format ---------------------------- #
    genre_node = evaluator.add_parallel(
        id=f"genre_and_format_{index_1based}",
        desc="Genre and format requirements met",
        parent=book_node,
        critical=True,
    )

    # Leaf: Literary fiction classification
    litfic_leaf = evaluator.add_leaf(
        id=f"literary_fiction_classification_{index_1based}",
        desc="Book classified as literary fiction (not genre fiction)",
        parent=genre_node,
        critical=True,
    )
    lit_sources = _filter_valid_urls([book.award_url, book.publisher_url])
    if len(lit_sources) > 0:
        lit_claim = (
            f"'{book.title}' is a work of literary fiction (not primarily genre fiction like SFF/romance/crime)."
        )
        lit_add_ins = (
            "Use the award page and/or publisher page to judge. Being a 2024 finalist or winner for any of the three "
            "specified major awards in 'Fiction' is a strong indicator that it is literary fiction. "
            "Pass unless the provided pages clearly brand the work as primarily a genre category."
        )
        await evaluator.verify(
            claim=lit_claim,
            node=litfic_leaf,
            sources=lit_sources if len(lit_sources) > 1 else lit_sources[0],
            additional_instruction=lit_add_ins,
        )
    else:
        litfic_leaf.score = 0.0
        litfic_leaf.status = "failed"

    # Leaf: Hardcover release
    hardcover_leaf = evaluator.add_leaf(
        id=f"hardcover_release_{index_1based}",
        desc="Book released in hardcover format",
        parent=genre_node,
        critical=True,
    )
    hc_sources = _filter_valid_urls([book.publisher_url])
    if len(hc_sources) > 0:
        hc_claim = f"'{book.title}' was released in a hardcover format."
        hc_add_ins = (
            "Check the publisher page for format listings such as 'Hardcover', 'Hard back', or 'HB/HC'. "
            "Passing requires explicit evidence of a hardcover edition."
        )
        await evaluator.verify(
            claim=hc_claim,
            node=hardcover_leaf,
            sources=hc_sources[0],
            additional_instruction=hc_add_ins,
        )
    else:
        hardcover_leaf.score = 0.0
        hardcover_leaf.status = "failed"

    # ------------------------- Author information ---------------------------- #
    author_node = evaluator.add_parallel(
        id=f"author_information_{index_1based}",
        desc="Author publication history identified",
        parent=book_node,
        critical=True,
    )

    # Leaf: Author status (debut vs. established)
    author_status_leaf = evaluator.add_leaf(
        id=f"author_status_{index_1based}",
        desc="Author identified as debut or established (with previous books)",
        parent=author_node,
        critical=True,
    )
    as_sources = _filter_valid_urls([book.author_url])
    if _has_value(book.author) and _has_value(book.author_status) and len(as_sources) > 0:
        if "debut" in (book.author_status or "").lower():
            as_claim = f"The author {book.author} is a debut novelist (this is their first novel)."
        else:
            as_claim = f"The author {book.author} is an established author with previous books."
        as_add_ins = (
            "Pass if the page states 'debut novel', 'first novel', or similar for debut; "
            "or lists prior books/novels for established. Short story collections do not invalidate 'debut novelist' "
            "if this is their first novel."
        )
        await evaluator.verify(
            claim=as_claim,
            node=author_status_leaf,
            sources=as_sources[0],
            additional_instruction=as_add_ins,
        )
    else:
        author_status_leaf.score = 0.0
        author_status_leaf.status = "failed"

    # Leaf: Author info URL provided (existence check)
    evaluator.add_custom_node(
        result=_has_value(book.author_url),
        id=f"author_reference_url_{index_1based}",
        desc="URL with author information provided",
        parent=author_node,
        critical=True,
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the literary awards 2024 Big Five books task.
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

    # 1) Extract structured book entries from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction",
    )

    # Normalize to exactly 3 entries (pad with empty items if fewer)
    books: List[BookItem] = list(extracted.books[:3])
    while len(books) < 3:
        books.append(BookItem())

    # 2) Build verification subtrees for each of the three books
    # Parallelize per-book verification for speed
    await asyncio.gather(
        verify_book(evaluator, root, books[0], 1),
        verify_book(evaluator, root, books[1], 2),
        verify_book(evaluator, root, books[2], 3),
    )

    # 3) Return evaluation summary (includes tree and extraction info)
    return evaluator.get_summary()