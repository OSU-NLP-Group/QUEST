import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "award_book_event_2025"
TASK_DESCRIPTION = (
    "I'm organizing a special book club event celebrating recent literary achievements and want to feature a book that received exceptional recognition across multiple prestigious awards. "
    "Please identify the book that won both the 2024 National Book Award for Fiction and the 2025 Pulitzer Prize for Fiction. "
    "For this book, provide the following information: the book's title, the author's name, and the publisher. "
    "Additionally, I need details about the 76th National Book Awards Ceremony (where one of these awards was presented): the ceremony date, the venue location in New York City, and the livestream broadcast time. "
    "Finally, I'm considering coordinating our book club event with Independent Bookstore Day 2025. Please provide the date of Independent Bookstore Day 2025 and confirmation that it was held on the last Saturday in April. "
    "Please include reference URLs for all information provided."
)


# ----------------------------- Data Models ---------------------------------- #

class LiteraryAwardsExtraction(BaseModel):
    # Book core info
    book_title: Optional[str] = None
    book_author: Optional[str] = None
    book_publisher: Optional[str] = None

    # Book-related sources
    book_metadata_urls: List[str] = Field(default_factory=list, description="General book metadata URLs (publisher/book page, booksellers, catalogs, etc.)")
    publisher_urls: List[str] = Field(default_factory=list, description="Publisher or book page URLs on the publisher website")
    nba_award_urls: List[str] = Field(default_factory=list, description="Sources specifically supporting the 2024 National Book Award for Fiction win")
    pulitzer_award_urls: List[str] = Field(default_factory=list, description="Sources specifically supporting the 2025 Pulitzer Prize for Fiction win")

    # National Book Awards ceremony info (76th, 2025)
    ceremony_date: Optional[str] = None
    ceremony_venue: Optional[str] = None
    livestream_time: Optional[str] = None
    ceremony_urls: List[str] = Field(default_factory=list, description="Sources for the 76th National Book Awards ceremony details")

    # Independent Bookstore Day 2025
    bookstore_event_date: Optional[str] = None
    bookstore_urls: List[str] = Field(default_factory=list, description="Sources for Independent Bookstore Day 2025")


# -------------------------- Extraction Prompt -------------------------------- #

def prompt_extract_all() -> str:
    return """
Extract the following information from the answer text, exactly as provided, and collect all cited URLs. Do NOT invent or infer anything not present in the answer.

BOOK THAT WON BOTH AWARDS
- book_title: The book's title
- book_author: The author's name
- book_publisher: The book's publisher
- book_metadata_urls: List all URLs in the answer that present general book metadata (publisher page, official book page, bookseller/catalog pages, library entries, etc.)
- publisher_urls: List all URLs in the answer that specifically point to the publisher's page for the book (if any)
- nba_award_urls: List all URLs cited that specifically support the claim that the book won the 2024 National Book Award for Fiction (e.g., National Book Foundation pages, credible news announcements)
- pulitzer_award_urls: List all URLs cited that specifically support the claim that the book won the 2025 Pulitzer Prize for Fiction (e.g., official Pulitzer site pages, credible news announcements)

76TH NATIONAL BOOK AWARDS CEREMONY (2025)
- ceremony_date: The ceremony date as given in the answer (expected format like "November 19, 2025" if present)
- ceremony_venue: The venue location in New York City as given (e.g., "Cipriani Wall Street, New York City")
- livestream_time: The livestream broadcast time as given (e.g., "8:00pm EST"/"8:00 PM ET")
- ceremony_urls: List all URLs cited that specifically support the ceremony details (official event pages, press releases, credible announcements)

INDEPENDENT BOOKSTORE DAY 2025
- bookstore_event_date: The date as given (e.g., "April 26, 2025")
- bookstore_urls: List all URLs cited that specifically support Independent Bookstore Day 2025 information (official pages, credible announcements)

RULES:
- Only extract URLs explicitly present in the answer (plain links or markdown). Return them in the appropriate lists.
- If any field is missing from the answer, set it to null (for strings) or an empty list (for arrays).
- Do not deduplicate automatically; include all URLs mentioned. The evaluator will handle duplicates later.
"""


# --------------------------- Helper Utilities -------------------------------- #

def merge_sources(*lists: List[str]) -> List[str]:
    """Merge lists preserving order and deduplicate."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst or []:
            if url and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


def safe_str(val: Optional[str]) -> str:
    return val or ""


# ------------------------- Verification Builders ----------------------------- #

async def verify_award_winning_book(
    evaluator: Evaluator,
    parent_node,
    ex: LiteraryAwardsExtraction
) -> None:
    """
    Build and verify the 'Award_Winning_Book_Information' subtree.
    """
    book_group = evaluator.add_parallel(
        id="Award_Winning_Book_Information",
        desc="Verify information about the book that won both the 2024 National Book Award for Fiction and the 2025 Pulitzer Prize for Fiction",
        parent=parent_node,
        critical=True
    )

    # Reference URLs existence (critical)
    all_book_urls = merge_sources(ex.book_metadata_urls, ex.publisher_urls, ex.nba_award_urls, ex.pulitzer_award_urls)
    evaluator.add_custom_node(
        result=len(all_book_urls) > 0,
        id="Book_Reference_URLs",
        desc="Valid reference URLs are provided for the book information",
        parent=book_group,
        critical=True
    )

    # Won 2024 National Book Award for Fiction (critical)
    nba_leaf = evaluator.add_leaf(
        id="Won_2024_National_Book_Award",
        desc="The book is correctly identified as winning the 2024 National Book Award for Fiction",
        parent=book_group,
        critical=True
    )
    nba_claim = (
        f"The book titled '{safe_str(ex.book_title)}' by {safe_str(ex.book_author)} won the 2024 National Book Award for Fiction."
    ).strip()
    await evaluator.verify(
        claim=nba_claim,
        node=nba_leaf,
        sources=ex.nba_award_urls if ex.nba_award_urls else all_book_urls,
        additional_instruction=(
            "Verify that the cited page(s) explicitly list this book as the WINNER of the 2024 National Book Award for Fiction "
            "(not just a finalist). Allow minor variations in name casing or punctuation."
        )
    )

    # Won 2025 Pulitzer Prize for Fiction (critical)
    pulitzer_leaf = evaluator.add_leaf(
        id="Won_2025_Pulitzer_Prize",
        desc="The book is correctly identified as winning the 2025 Pulitzer Prize for Fiction",
        parent=book_group,
        critical=True
    )
    pulitzer_claim = (
        f"The book titled '{safe_str(ex.book_title)}' by {safe_str(ex.book_author)} won the 2025 Pulitzer Prize for Fiction."
    ).strip()
    await evaluator.verify(
        claim=pulitzer_claim,
        node=pulitzer_leaf,
        sources=ex.pulitzer_award_urls if ex.pulitzer_award_urls else all_book_urls,
        additional_instruction=(
            "Verify that the cited page(s) explicitly list this book as the WINNER of the 2025 Pulitzer Prize for Fiction. "
            "Allow minor variations in name casing or punctuation."
        )
    )

    # Book Title (critical)
    title_leaf = evaluator.add_leaf(
        id="Book_Title",
        desc="The correct title of the book is provided",
        parent=book_group,
        critical=True
    )
    title_claim = f"The title of the book is '{safe_str(ex.book_title)}'."
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=all_book_urls,
        additional_instruction=(
            "Check that at least one cited page clearly shows the book title exactly or in a very close form. "
            "Minor formatting variations (e.g., capitalization, subtitle punctuation) are acceptable."
        )
    )

    # Author Name (critical)
    author_leaf = evaluator.add_leaf(
        id="Author_Name",
        desc="The correct author name is provided",
        parent=book_group,
        critical=True
    )
    author_claim = f"The author of the book is {safe_str(ex.book_author)}."
    await evaluator.verify(
        claim=author_claim,
        node=author_leaf,
        sources=all_book_urls,
        additional_instruction=(
            "Verify that the cited page(s) identify the same author for this book. "
            "Allow minor variants, such as presence/absence of middle initials."
        )
    )

    # Publisher Name (critical)
    publisher_leaf = evaluator.add_leaf(
        id="Publisher_Name",
        desc="The publisher name is provided",
        parent=book_group,
        critical=True
    )
    publisher_claim = f"The publisher of the book is {safe_str(ex.book_publisher)}."
    publisher_sources = merge_sources(ex.publisher_urls, ex.book_metadata_urls, ex.nba_award_urls, ex.pulitzer_award_urls)
    await evaluator.verify(
        claim=publisher_claim,
        node=publisher_leaf,
        sources=publisher_sources,
        additional_instruction=(
            "Check that at least one cited page (preferably the publisher's book page or an authoritative catalog entry) "
            "explicitly lists the publisher for this book."
        )
    )


async def verify_ceremony_info(
    evaluator: Evaluator,
    parent_node,
    ex: LiteraryAwardsExtraction
) -> None:
    """
    Build and verify the 'National_Book_Awards_Ceremony_Information' subtree.
    """
    ceremony_group = evaluator.add_parallel(
        id="National_Book_Awards_Ceremony_Information",
        desc="Verify information about the 76th National Book Awards Ceremony",
        parent=parent_node,
        critical=True
    )

    # Reference URLs existence (critical)
    evaluator.add_custom_node(
        result=len(ex.ceremony_urls) > 0,
        id="Ceremony_Reference_URLs",
        desc="Valid reference URLs are provided for the ceremony information",
        parent=ceremony_group,
        critical=True
    )

    # Ceremony Date (critical)
    ceremony_date_leaf = evaluator.add_leaf(
        id="Ceremony_Date",
        desc="The ceremony date (November 19, 2025) is correctly provided",
        parent=ceremony_group,
        critical=True
    )
    ceremony_date_claim = "The 76th National Book Awards Ceremony took place on November 19, 2025."
    await evaluator.verify(
        claim=ceremony_date_claim,
        node=ceremony_date_leaf,
        sources=ex.ceremony_urls,
        additional_instruction=(
            "Confirm that the cited page(s) state the date for the 76th National Book Awards (2025) as November 19, 2025. "
            "Accept reasonable date formatting variations such as 'Nov. 19, 2025'."
        )
    )

    # Ceremony Venue (critical)
    ceremony_venue_leaf = evaluator.add_leaf(
        id="Ceremony_Venue",
        desc="The ceremony venue (Cipriani Wall Street, New York City) is correctly provided",
        parent=ceremony_group,
        critical=True
    )
    ceremony_venue_claim = "The 76th National Book Awards Ceremony was held at Cipriani Wall Street in New York City."
    await evaluator.verify(
        claim=ceremony_venue_claim,
        node=ceremony_venue_leaf,
        sources=ex.ceremony_urls,
        additional_instruction=(
            "Verify that the venue is described as Cipriani Wall Street (sometimes phrased as 'Cipriani on Wall Street') "
            "in New York City (NYC). Minor phrasing differences are acceptable."
        )
    )

    # Livestream Time (critical)
    livestream_leaf = evaluator.add_leaf(
        id="Livestream_Time",
        desc="The livestream time (8:00pm EST) is correctly provided",
        parent=ceremony_group,
        critical=True
    )
    livestream_claim = "The livestream broadcast for the 76th National Book Awards began at 8:00 PM Eastern time."
    await evaluator.verify(
        claim=livestream_claim,
        node=livestream_leaf,
        sources=ex.ceremony_urls,
        additional_instruction=(
            "Confirm that the livestream is listed as starting at 8:00 PM in the Eastern time zone. "
            "Treat 'ET', 'EST', or 'Eastern Time' as acceptable equivalents, and allow minor formatting differences like '8 PM ET'."
        )
    )


async def verify_bookstore_day_info(
    evaluator: Evaluator,
    parent_node,
    ex: LiteraryAwardsExtraction
) -> None:
    """
    Build and verify the 'Independent_Bookstore_Day_Information' subtree.
    """
    bookstore_group = evaluator.add_parallel(
        id="Independent_Bookstore_Day_Information",
        desc="Verify information about Independent Bookstore Day 2025",
        parent=parent_node,
        critical=True
    )

    # Reference URLs existence (critical)
    evaluator.add_custom_node(
        result=len(ex.bookstore_urls) > 0,
        id="Bookstore_Day_Reference_URLs",
        desc="Valid reference URLs are provided for Independent Bookstore Day information",
        parent=bookstore_group,
        critical=True
    )

    # Event Date (critical)
    event_date_leaf = evaluator.add_leaf(
        id="Event_Date",
        desc="The event date (April 26, 2025) is correctly provided",
        parent=bookstore_group,
        critical=True
    )
    event_date_claim = "Independent Bookstore Day 2025 took place on April 26, 2025."
    await evaluator.verify(
        claim=event_date_claim,
        node=event_date_leaf,
        sources=ex.bookstore_urls,
        additional_instruction=(
            "Verify that the cited page(s) explicitly indicate the 2025 Independent Bookstore Day date as April 26, 2025."
        )
    )

    # Day of Week Verification (critical)
    dow_leaf = evaluator.add_leaf(
        id="Day_of_Week_Verification",
        desc="Confirmation is provided that the event was held on the last Saturday in April",
        parent=bookstore_group,
        critical=True
    )
    dow_claim = "Independent Bookstore Day 2025 was held on the last Saturday in April."
    await evaluator.verify(
        claim=dow_claim,
        node=dow_leaf,
        sources=ex.bookstore_urls,
        additional_instruction=(
            "Verify that the cited page(s) state that Independent Bookstore Day is held on the last Saturday in April, "
            "and that in 2025 this corresponded to April 26."
        )
    )


# ----------------------------- Main Evaluation ------------------------------ #

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
    Evaluate an answer for the award-winning book and literary events task.
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

    # Extract all structured info in one pass
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=LiteraryAwardsExtraction,
        extraction_name="extracted_literary_awards_info"
    )

    # Build and verify subtrees
    await verify_award_winning_book(evaluator, root, extraction)
    await verify_ceremony_info(evaluator, root, extraction)
    await verify_bookstore_day_info(evaluator, root, extraction)

    return evaluator.get_summary()