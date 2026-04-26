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
TASK_ID = "four_books_2024_2025"
TASK_DESCRIPTION = (
    "Identify 4 books from the 2024-2025 period, each satisfying detailed award, club, and publisher criteria. "
    "For each, verify requested details and provide publisher verification URL(s)."
)

BIG_FIVE_KEYWORDS = [
    "penguin random house",
    "harpercollins",
    "macmillan",
    "hachette",
    "simon & schuster",
    "simon and schuster"
]

GROVE_IMPRINTS = [
    "grove atlantic",
    "grove press",
    "atlantic books"
]


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class BookInfo(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None

    # Publishing-related fields
    publisher: Optional[str] = None               # e.g., Big Five name or Grove/Grove Press/Atlantic Books or Doubleday
    imprint: Optional[str] = None                 # book 3 specific imprint; optional for others
    publication_date: Optional[str] = None        # free-form date string
    publication_year: Optional[str] = None        # for book 2, e.g., "2024" or "2025"
    page_count: Optional[str] = None              # free-form to accommodate variations
    primary_genre: Optional[str] = None           # for book 3

    # Award/club meta
    award_name: Optional[str] = None              # e.g., "National Book Award", "Booker Prize"
    award_year: Optional[str] = None              # e.g., "2024"
    award_category: Optional[str] = None          # e.g., "Fiction"
    book_club_name: Optional[str] = None          # e.g., "Reese's Book Club", "Oprah's Book Club", "Read with Jenna"
    book_club_month_year: Optional[str] = None    # e.g., "May 2025", "November 2025"

    # URL sources
    publisher_verification_urls: List[str] = Field(default_factory=list)           # required for each book
    award_verification_urls: List[str] = Field(default_factory=list)               # if present in answer
    book_club_verification_urls: List[str] = Field(default_factory=list)           # if present in answer
    publisher_parent_verification_urls: List[str] = Field(default_factory=list)    # for book 4 (Doubleday -> PRH)


class BooksExtraction(BaseModel):
    book1: Optional[BookInfo] = None
    book2: Optional[BookInfo] = None
    book3: Optional[BookInfo] = None
    book4: Optional[BookInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_books() -> str:
    return """
Extract structured information for exactly four books from the answer according to these per‑book requirements. 
Return a JSON object with fields: book1, book2, book3, book4; each an object with the following superset of fields:
- title
- author
- publisher (the specific publisher or imprint name as claimed)
- imprint (if an imprint is claimed separately, otherwise null)
- publication_date
- publication_year
- page_count
- primary_genre
- award_name
- award_year
- award_category
- book_club_name
- book_club_month_year
- publisher_verification_urls (array of URLs explicitly provided in the answer to verify the publisher)
- award_verification_urls (array of URLs provided in the answer that verify the award claim, if any)
- book_club_verification_urls (array of URLs provided in the answer that verify the book club claim, if any)
- publisher_parent_verification_urls (array of URLs that verify an imprint‑to‑parent relationship, e.g., Doubleday → Penguin Random House)

Specific guidance for each book:

Book 1 (2024 National Book Award winner in Fiction + celebrity book club selection in 2025 + Big Five publisher):
- title: complete title
- author: full name
- publisher: one of the Big Five: Penguin Random House, HarperCollins, Macmillan, Hachette, or Simon & Schuster (use the exact publisher name given in the answer)
- publication_date: must be in 2024
- award_name: should indicate "National Book Award"
- award_year: "2024"
- award_category: "Fiction"
- book_club_name: one of Reese's Book Club, Oprah's Book Club, or Read with Jenna (Jenna Bush Hager)
- book_club_month_year: any month in 2025 if present, otherwise include the year 2025
- publisher_verification_urls: extract at least one URL provided in the answer to verify the publisher
- award_verification_urls / book_club_verification_urls: extract if present

Book 2 (2024 Booker Prize winner + published by Grove Atlantic or its imprints Grove Press or Atlantic Books):
- title: complete title
- author: full name
- publisher: should be Grove Atlantic, Grove Press, or Atlantic Books (as given in the answer)
- publication_year: year as provided (preferably within 2024–2025)
- page_count: if provided
- award_name: "Booker Prize"
- award_year: "2024"
- publisher_verification_urls: at least one URL provided in the answer to verify the publisher
- award_verification_urls: extract if present

Book 3 (Reese Witherspoon's book club pick for May 2025 + published by PRH or its imprints):
- title: complete title
- author: full name
- imprint: the specific Penguin Random House imprint (e.g., "Viking", "Dutton", "Ballantine", etc.)
- publisher: may be "Penguin Random House" or the imprint; if both appear, keep imprint in 'imprint' and the broader house in 'publisher'
- publication_date: must be April 22, 2025
- primary_genre: the main genre claimed
- book_club_name: "Reese's Book Club"
- book_club_month_year: "May 2025"
- publisher_verification_urls: at least one URL provided in the answer to verify the publisher/imprint
- book_club_verification_urls: extract if present

Book 4 (Read with Jenna November 2025 pick + author Oyinkan Braithwaite + published by Doubleday which is part of PRH):
- title: complete title
- author: should be "Oyinkan Braithwaite" (as provided in the answer)
- publisher: "Doubleday"
- publication_date: must be November 4, 2025
- page_count: if provided
- book_club_name: "Read with Jenna"
- book_club_month_year: "November 2025"
- publisher_verification_urls: at least one URL provided in the answer to verify that the book is published by Doubleday
- publisher_parent_verification_urls: URL(s) provided in the answer that verify Doubleday is an imprint under Penguin Random House
- book_club_verification_urls: extract if present

Rules:
- Extract EXACTLY what appears in the answer.
- If any field is not mentioned, set it to null (or empty array for URLs).
- For URLs, extract actual links; include full protocols. 
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _present(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _contains_any(text: Optional[str], keywords: List[str]) -> bool:
    if not _present(text):
        return False
    t = text.lower()
    return any(k in t for k in keywords)


def _normalize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Basic cleanup: strip and keep unique while preserving order
    seen = set()
    cleaned: List[str] = []
    for u in urls:
        if not _present(u):
            continue
        us = u.strip()
        if us not in seen:
            seen.add(us)
            cleaned.append(us)
    return cleaned


def _union_urls(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for u in lst or []:
            if _present(u) and u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _year_in_date_str(date_str: Optional[str], year: str) -> bool:
    if not _present(date_str):
        return False
    return year in date_str


def _equals_any_ci(text: Optional[str], targets: List[str]) -> bool:
    if not _present(text):
        return False
    t = text.strip().lower()
    return any(t == x.strip().lower() for x in targets)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_book_1(evaluator: Evaluator, parent_node, b1: Optional[BookInfo]) -> None:
    node = evaluator.add_parallel(
        id="book_1",
        desc="Book 1: A 2024 National Book Award winner selected by a celebrity book club in 2025",
        parent=parent_node,
        critical=False
    )

    # Safe guards
    title = b1.title if b1 and _present(b1.title) else ""
    author = b1.author if b1 and _present(b1.author) else ""
    publisher = b1.publisher if b1 and _present(b1.publisher) else ""
    pub_date = b1.publication_date if b1 and _present(b1.publication_date) else ""
    club = b1.book_club_name if b1 and _present(b1.book_club_name) else ""
    award_urls = _normalize_urls(b1.award_verification_urls if b1 else [])
    club_urls = _normalize_urls(b1.book_club_verification_urls if b1 else [])
    publisher_urls = _normalize_urls(b1.publisher_verification_urls if b1 else [])

    # title_1 (critical) – presence
    evaluator.add_custom_node(
        result=_present(title),
        id="title_1",
        desc="Provide the complete book title",
        parent=node,
        critical=True
    )

    # author_info_1 (critical) – presence
    evaluator.add_custom_node(
        result=_present(author),
        id="author_info_1",
        desc="Provide the author's full name",
        parent=node,
        critical=True
    )

    # publication_date_1 (critical) – must be in 2024 (presence + year check)
    evaluator.add_custom_node(
        result=_present(pub_date) and _year_in_date_str(pub_date, "2024"),
        id="publication_date_1",
        desc="Provide the publication date in 2024",
        parent=node,
        critical=True
    )

    # publisher_name_1 (critical) – Big Five membership check
    evaluator.add_custom_node(
        result=_present(publisher) and _contains_any(publisher, BIG_FIVE_KEYWORDS),
        id="publisher_name_1",
        desc="Published by one of the Big Five publishers - identify the specific publisher",
        parent=node,
        critical=True
    )

    # publisher_verification_1 (critical) – verify URL supports publisher claim
    leaf_pub = evaluator.add_leaf(
        id="publisher_verification_1",
        desc="Provide reference URL verifying the publisher",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The book '{title}' is published by {publisher}.",
        node=leaf_pub,
        sources=publisher_urls,
        additional_instruction="Verify that the provided webpage(s) explicitly indicate the book's publisher as stated. "
                               "If the page is unrelated or does not clearly show the publisher, mark as not supported."
    )

    # award_category_1 (critical) – verify NBA 2024 Fiction win via URL(s)
    leaf_award = evaluator.add_leaf(
        id="award_category_1",
        desc="Won the 2024 National Book Award in the Fiction category",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The book '{title}' won the 2024 National Book Award in the Fiction category.",
        node=leaf_award,
        sources=award_urls,
        additional_instruction="Check that the page clearly states this title won (not shortlisted or nominated) "
                               "the National Book Award for Fiction in 2024."
    )

    # book_club_selection_1 (critical) – verify Reese/Oprah/Jenna selection in 2025
    leaf_club = evaluator.add_leaf(
        id="book_club_selection_1",
        desc="Was selected by at least one major celebrity book club (Reese Witherspoon, Oprah Winfrey, or Jenna Bush Hager) in 2025",
        parent=node,
        critical=True
    )
    claimed_club_text = club if _present(club) else "one of Reese's/Oprah's/Jenna Bush Hager's book clubs"
    await evaluator.verify(
        claim=f"The book '{title}' was selected by {claimed_club_text} in 2025.",
        node=leaf_club,
        sources=club_urls,
        additional_instruction="Look for an official announcement or book club page indicating the selection took place in 2025."
    )


async def verify_book_2(evaluator: Evaluator, parent_node, b2: Optional[BookInfo]) -> None:
    node = evaluator.add_parallel(
        id="book_2",
        desc="Book 2: The 2024 Booker Prize winner with specific publisher",
        parent=parent_node,
        critical=False
    )

    title = b2.title if b2 and _present(b2.title) else ""
    author = b2.author if b2 and _present(b2.author) else ""
    publisher = b2.publisher if b2 and _present(b2.publisher) else ""
    pub_year = b2.publication_year if b2 and _present(b2.publication_year) else ""
    page_count = b2.page_count if b2 and _present(b2.page_count) else ""
    award_urls = _normalize_urls(b2.award_verification_urls if b2 else [])
    publisher_urls = _normalize_urls(b2.publisher_verification_urls if b2 else [])

    # title_2 (critical) – presence
    evaluator.add_custom_node(
        result=_present(title),
        id="title_2",
        desc="Provide the complete book title",
        parent=node,
        critical=True
    )

    # author_info_2 (critical) – presence
    evaluator.add_custom_node(
        result=_present(author),
        id="author_info_2",
        desc="Provide the author's full name",
        parent=node,
        critical=True
    )

    # publication_year_2 (critical) – presence and (preferably) 2024–2025
    evaluator.add_custom_node(
        result=_present(pub_year) and any(y in pub_year for y in ["2024", "2025"]),
        id="publication_year_2",
        desc="Provide the publication year",
        parent=node,
        critical=True
    )

    # page_count_2 (non-critical) – presence
    evaluator.add_custom_node(
        result=_present(page_count),
        id="page_count_2",
        desc="Provide the page count",
        parent=node,
        critical=False
    )

    # publisher_confirmation_2 (critical) – Grove Atlantic or its imprints
    evaluator.add_custom_node(
        result=_present(publisher) and _contains_any(publisher, GROVE_IMPRINTS),
        id="publisher_confirmation_2",
        desc="Published by Grove Atlantic (or its imprints Grove Press or Atlantic Books)",
        parent=node,
        critical=True
    )

    # publisher_reference_2 (critical) – verify via URL
    leaf_pub = evaluator.add_leaf(
        id="publisher_reference_2",
        desc="Provide reference URL for publisher verification",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The book '{title}' is published by {publisher}.",
        node=leaf_pub,
        sources=publisher_urls,
        additional_instruction="Verify that the page clearly shows the publisher (Grove Atlantic, Grove Press, or Atlantic Books) "
                               "for this exact title."
    )

    # award_verification_2 (critical) – won 2024 Booker Prize
    leaf_award = evaluator.add_leaf(
        id="award_verification_2",
        desc="Won the 2024 Booker Prize",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The book '{title}' won the 2024 Booker Prize.",
        node=leaf_award,
        sources=award_urls,
        additional_instruction="Ensure the page explicitly states that this book is the 2024 Booker Prize winner (not longlisted/shortlisted only)."
    )


async def verify_book_3(evaluator: Evaluator, parent_node, b3: Optional[BookInfo]) -> None:
    node = evaluator.add_parallel(
        id="book_3",
        desc="Book 3: Reese Witherspoon's May 2025 book club pick",
        parent=parent_node,
        critical=False
    )

    title = b3.title if b3 and _present(b3.title) else ""
    author = b3.author if b3 and _present(b3.author) else ""
    imprint = b3.imprint if b3 and _present(b3.imprint) else ""
    genre = b3.primary_genre if b3 and _present(b3.primary_genre) else ""
    pub_date = b3.publication_date if b3 and _present(b3.publication_date) else ""
    club_urls = _normalize_urls(b3.book_club_verification_urls if b3 else [])
    publisher_urls = _normalize_urls(b3.publisher_verification_urls if b3 else [])

    # title_3 (critical) – presence
    evaluator.add_custom_node(
        result=_present(title),
        id="title_3",
        desc="Provide the complete book title",
        parent=node,
        critical=True
    )

    # author_info_3 (critical) – presence
    evaluator.add_custom_node(
        result=_present(author),
        id="author_info_3",
        desc="Provide the author's full name",
        parent=node,
        critical=True
    )

    # publisher_imprint_3 (critical) – imprint presence (PRH imprint expected)
    evaluator.add_custom_node(
        result=_present(imprint),
        id="publisher_imprint_3",
        desc="Published by Penguin Random House - identify the specific imprint",
        parent=node,
        critical=True
    )

    # publisher_reference_3 (critical) – verify via URL that imprint/PRH publishes the book
    leaf_pub = evaluator.add_leaf(
        id="publisher_reference_3",
        desc="Provide reference URL for publisher verification",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The book '{title}' is published by the imprint '{imprint}' of Penguin Random House.",
        node=leaf_pub,
        sources=publisher_urls,
        additional_instruction="The page should indicate the imprint name and that it is part of Penguin Random House, or be an official PRH/imprint product page."
    )

    # publication_details_3 (critical) – verify exact date April 22, 2025 via URL
    leaf_date = evaluator.add_leaf(
        id="publication_details_3",
        desc="Published on April 22, 2025",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publication date of '{title}' is April 22, 2025.",
        node=leaf_date,
        sources=publisher_urls,
        additional_instruction="Confirm the specific date (April 22, 2025) on the product/publisher page. Allow minor formatting variations (e.g., Apr. 22, 2025)."
    )

    # genre_3 (non-critical) – presence
    evaluator.add_custom_node(
        result=_present(genre),
        id="genre_3",
        desc="Identify the primary genre",
        parent=node,
        critical=False
    )

    # book_club_verification_3 (critical) – verify Reese's Book Club May 2025
    leaf_club = evaluator.add_leaf(
        id="book_club_verification_3",
        desc="Selected as Reese Witherspoon's book club pick for May 2025",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The book '{title}' was Reese's Book Club pick for May 2025.",
        node=leaf_club,
        sources=club_urls,
        additional_instruction="Look for an official Reese's Book Club announcement/page indicating the May 2025 pick."
    )


async def verify_book_4(evaluator: Evaluator, parent_node, b4: Optional[BookInfo]) -> None:
    node = evaluator.add_parallel(
        id="book_4",
        desc="Book 4: Read with Jenna November 2025 pick by a specific author",
        parent=parent_node,
        critical=False
    )

    title = b4.title if b4 and _present(b4.title) else ""
    author = b4.author if b4 and _present(b4.author) else ""
    publisher = b4.publisher if b4 and _present(b4.publisher) else ""
    pub_date = b4.publication_date if b4 and _present(b4.publication_date) else ""
    page_count = b4.page_count if b4 and _present(b4.page_count) else ""
    club_urls = _normalize_urls(b4.book_club_verification_urls if b4 else [])
    publisher_urls = _normalize_urls(b4.publisher_verification_urls if b4 else [])
    parent_urls = _normalize_urls(b4.publisher_parent_verification_urls if b4 else [])

    # title_4 (critical) – presence
    evaluator.add_custom_node(
        result=_present(title),
        id="title_4",
        desc="Provide the complete book title",
        parent=node,
        critical=True
    )

    # page_count_4 (non-critical) – presence
    evaluator.add_custom_node(
        result=_present(page_count),
        id="page_count_4",
        desc="Provide the page count",
        parent=node,
        critical=False
    )

    # author_verification_4 (critical) – verify author is Oyinkan Braithwaite (use simple verify; allow minor variants)
    leaf_author = evaluator.add_leaf(
        id="author_verification_4",
        desc="Written by Oyinkan Braithwaite",
        parent=node,
        critical=True
    )
    # Use available URLs as support if provided
    author_sources = _union_urls(publisher_urls, club_urls)
    await evaluator.verify(
        claim=f"The author of '{title}' is Oyinkan Braithwaite (the provided author is '{author}').",
        node=leaf_author,
        sources=author_sources if author_sources else None,
        additional_instruction="Treat minor name variations as acceptable if they clearly refer to Oyinkan Braithwaite (e.g., middle initials). Prefer evidence on provided pages if available."
    )

    # publisher_identification_4 (critical) – publisher string indicates Doubleday
    evaluator.add_custom_node(
        result=_present(publisher) and "doubleday" in publisher.strip().lower(),
        id="publisher_identification_4",
        desc="Published by Doubleday",
        parent=node,
        critical=True
    )

    # publisher_reference_4 (critical) – verify via URL that book is published by Doubleday
    leaf_pub = evaluator.add_leaf(
        id="publisher_reference_4",
        desc="Provide reference URL for publisher verification",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The book '{title}' is published by Doubleday.",
        node=leaf_pub,
        sources=publisher_urls,
        additional_instruction="Confirm on the product/publisher page that the publisher is Doubleday (an imprint)."
    )

    # publisher_parent_company_4 (critical) – verify Doubleday is part of PRH via URL(s)
    leaf_parent = evaluator.add_leaf(
        id="publisher_parent_company_4",
        desc="Verify Doubleday is part of Penguin Random House",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Doubleday is an imprint of Penguin Random House.",
        node=leaf_parent,
        sources=parent_urls if parent_urls else publisher_urls,
        additional_instruction="Accept official PRH pages or credible sources explicitly stating this imprint relationship."
    )

    # publication_date_4 (critical) – verify exact date Nov 4, 2025 via URL(s)
    leaf_date = evaluator.add_leaf(
        id="publication_date_4",
        desc="Published on November 4, 2025",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publication date of '{title}' is November 4, 2025.",
        node=leaf_date,
        sources=publisher_urls,
        additional_instruction="Confirm the specific date on the product/publisher page. Allow minor formatting variations (e.g., Nov. 4, 2025)."
    )

    # book_club_verification_4 (critical) – verify Read with Jenna November 2025 pick
    leaf_club = evaluator.add_leaf(
        id="book_club_verification_4",
        desc="Selected as Jenna Bush Hager's Read with Jenna book club pick for November 2025",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The book '{title}' was Read with Jenna's pick for November 2025.",
        node=leaf_club,
        sources=club_urls,
        additional_instruction="Look for an official Read with Jenna (Today Show) announcement/page indicating the November 2025 pick."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer for the four-books task and return a structured result dictionary.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Books evaluated independently; partial credit across books
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

    # Add reference info for Big Five and Grove imprints (for transparency)
    evaluator.add_custom_info(
        info={
            "big_five": [
                "Penguin Random House", "HarperCollins", "Macmillan",
                "Hachette", "Simon & Schuster"
            ],
            "grove_imprints_allowed": ["Grove Atlantic", "Grove Press", "Atlantic Books"]
        },
        info_type="reference_lists",
        info_name="publisher_reference_lists"
    )

    # Extract information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_books(),
        template_class=BooksExtraction,
        extraction_name="books_extraction"
    )

    # Build verification subtrees for each book
    await verify_book_1(evaluator, root, extracted.book1)
    await verify_book_2(evaluator, root, extracted.book2)
    await verify_book_3(evaluator, root, extracted.book3)
    await verify_book_4(evaluator, root, extracted.book4)

    return evaluator.get_summary()