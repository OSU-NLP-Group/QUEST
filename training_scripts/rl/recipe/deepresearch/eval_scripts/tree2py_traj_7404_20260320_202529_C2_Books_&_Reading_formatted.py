import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient  # for typing
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "maclachlan_memoir_pagecounts"
TASK_DESCRIPTION = (
    "Kyle MacLachlan's memoir \"Fictional Selves\" is being published simultaneously in the United States and the "
    "United Kingdom on October 6, 2026. What is the page count of the US edition (published by Crown), and what is the "
    "page count of the UK edition (published by Century)? Provide reference URLs confirming each page count."
)

# Expected ground-truth targets for verification
EXPECTED_TITLE = "Fictional Selves: A Memoir"
EXPECTED_AUTHOR = "Kyle MacLachlan"
EXPECTED_PUB_DATE = "October 6, 2026"

US_EXPECTED_PUBLISHER = "Crown"
US_EXPECTED_PAGES = "288"
US_EXPECTED_ISBN = "9798217086320"
US_EXPECTED_PRICE = "$32.00"

UK_EXPECTED_PUBLISHER = "Century"
UK_EXPECTED_PAGES = "336"
UK_EXPECTED_ISBN = "9781529955057"
UK_EXPECTED_PRICE = "£25.00"

AUDIOBOOK_EXPECTED_LENGTH = "8 hours"


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class EditionInfo(BaseModel):
    publisher: Optional[str] = None
    page_count: Optional[str] = None
    page_count_urls: List[str] = Field(
        default_factory=list,
        description="URLs explicitly cited to support the page count for this edition."
    )
    supporting_urls: List[str] = Field(
        default_factory=list,
        description="Any other URLs specifically about this edition (publisher page, retailer, press page, etc.)."
    )
    isbn: Optional[str] = None
    hardcover_price: Optional[str] = None


class BookExtraction(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[str] = None
    us: Optional[EditionInfo] = None
    uk: Optional[EditionInfo] = None
    audiobook_length: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_book_info() -> str:
    return """
Extract the following fields from the provided answer text about Kyle MacLachlan's memoir and its editions. Return a single JSON object with this schema:

- title: the book title as written in the answer (string, or null)
- author: the author as written in the answer (string, or null)
- publication_date: the publication date as written in the answer (string, or null)
- us: object for the US (Crown) edition, or null if not found
  - publisher: publisher name (string or null)
  - page_count: page count as written (string, e.g., "288" or "288 pages"; do not normalize beyond trimming) 
  - page_count_urls: array of URL(s) explicitly provided in the answer that confirm the US page count 
                     (extract actual URLs even if presented as markdown links; if none, return [])
  - supporting_urls: array of any other URL(s) in the answer that are specifically about the US edition 
                     (do not duplicate page_count_urls; if none, return [])
  - isbn: ISBN for the US edition if given (string or null; keep digits and hyphens as in the answer)
  - hardcover_price: hardcover price for the US edition if given (string or null, e.g., "$32.00")
- uk: object for the UK (Century) edition, or null if not found
  - publisher: publisher name (string or null)
  - page_count: page count as written (string, e.g., "336" or "336 pages"; do not normalize beyond trimming)
  - page_count_urls: array of URL(s) explicitly provided in the answer that confirm the UK page count 
                     (extract actual URLs even if presented as markdown links; if none, return [])
  - supporting_urls: array of any other URL(s) in the answer that are specifically about the UK edition 
                     (do not duplicate page_count_urls; if none, return [])
  - isbn: ISBN for the UK edition if given (string or null; keep digits and hyphens as in the answer)
  - hardcover_price: hardcover price for the UK edition if given (string or null, e.g., "£25.00")
- audiobook_length: audiobook length if mentioned (string or null, e.g., "8 hours")

Important:
- Return only information explicitly present in the answer. Do not invent or infer values.
- For URLs, extract the actual URL strings (convert markdown links to plain URLs).
- If fields are missing in the answer, return null for that field (or [] for arrays).
    """.strip()


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def combine_urls(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                result.append(url)
    return result


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_core_metadata(evaluator: Evaluator, parent_node, data: BookExtraction) -> None:
    core_node = evaluator.add_parallel(
        id="Book_Core_Metadata",
        desc="Core book metadata matches the specified memoir.",
        parent=parent_node,
        critical=False  # Non-critical for this task focused on page counts
    )

    # Title check
    title_leaf = evaluator.add_leaf(
        id="Title_Check",
        desc="Book title is exactly 'Fictional Selves: A Memoir'.",
        parent=core_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The book title is exactly '{EXPECTED_TITLE}'.",
        node=title_leaf,
        additional_instruction="Judge solely based on whether the answer states this exact title. Allow minor case variations."
    )

    # Author check
    author_leaf = evaluator.add_leaf(
        id="Author_Check",
        desc="Author is Kyle MacLachlan.",
        parent=core_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The author is {EXPECTED_AUTHOR}.",
        node=author_leaf,
        additional_instruction="Judge based on the answer text. Allow minor case variations and ignore middle initials."
    )

    # Publication date check
    pubdate_leaf = evaluator.add_leaf(
        id="Publication_Date_Check",
        desc="Publication date is October 6, 2026.",
        parent=core_node,
        critical=False
    )
    await evaluator.verify(
        claim="The publication date is October 6, 2026.",
        node=pubdate_leaf,
        additional_instruction="Accept common date format variants like 'Oct 6, 2026' or '2026-10-06'. Judge from the answer text."
    )


async def build_us_constraints(evaluator: Evaluator, parent_node, us: Optional[EditionInfo]) -> None:
    us_node = evaluator.add_parallel(
        id="US_Edition_Constraints",
        desc="US edition (Crown) constraints are met, including page count and supporting reference URL.",
        parent=parent_node,
        critical=False
    )

    # Publisher check (from answer content)
    us_pub_leaf = evaluator.add_leaf(
        id="US_Publisher_Check",
        desc="US publisher is Crown.",
        parent=us_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The US publisher is {US_EXPECTED_PUBLISHER}.",
        node=us_pub_leaf,
        additional_instruction="Judge from the answer text; this refers specifically to the US edition publisher."
    )

    # Essentials for page count: existence of at least one reference URL and the page count verification
    us_essentials = evaluator.add_parallel(
        id="US_PageCount_Essentials",
        desc="US page count essentials (value + at least one confirming URL).",
        parent=us_node,
        critical=True
    )

    # Existence of at least one URL explicitly supporting page count
    has_us_url = bool(us and us.page_count_urls and len(us.page_count_urls) > 0)
    evaluator.add_custom_node(
        result=has_us_url,
        id="US_Page_Count_Reference_URL_Check",
        desc="Provides at least one reference URL that explicitly supports the US edition page count (288) for the Crown US edition.",
        parent=us_essentials,
        critical=True
    )

    # Page count verification by URLs
    us_page_leaf = evaluator.add_leaf(
        id="US_Page_Count_Check",
        desc="US edition page count is 288 pages.",
        parent=us_essentials,
        critical=True
    )
    us_sources = combine_urls(us.page_count_urls if us else None, us.supporting_urls if us else None)
    await evaluator.verify(
        claim=f"The US edition published by Crown has {US_EXPECTED_PAGES} pages.",
        node=us_page_leaf,
        sources=us_sources,
        additional_instruction=(
            "Verify that the provided webpage(s) explicitly indicate the US (Crown) edition's page count is 288. "
            "Allow minor phrasing like '288 pp' or '288 pages'. The page should clearly pertain to the US/Crown edition."
        )
    )

    # Optional/non-critical additional checks using any available URLs
    all_us_urls = us_sources

    # ISBN check
    us_isbn_leaf = evaluator.add_leaf(
        id="US_ISBN_Check",
        desc="US ISBN is 9798217086320.",
        parent=us_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The US hardcover ISBN is {US_EXPECTED_ISBN}.",
        node=us_isbn_leaf,
        sources=all_us_urls,
        additional_instruction="Verify on the provided page(s). Accept ISBN with or without hyphens, but digit sequence must match."
    )

    # Hardcover price check
    us_price_leaf = evaluator.add_leaf(
        id="US_Hardcover_Price_Check",
        desc="US hardcover price is $32.00.",
        parent=us_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The US hardcover price is {US_EXPECTED_PRICE}.",
        node=us_price_leaf,
        sources=all_us_urls,
        additional_instruction="Verify on the provided page(s). Confirm it's USD price for the US hardcover edition."
    )


async def build_uk_constraints(evaluator: Evaluator, parent_node, uk: Optional[EditionInfo]) -> None:
    uk_node = evaluator.add_parallel(
        id="UK_Edition_Constraints",
        desc="UK edition (Century) constraints are met, including page count and supporting reference URL.",
        parent=parent_node,
        critical=False
    )

    # Publisher check (from answer content)
    uk_pub_leaf = evaluator.add_leaf(
        id="UK_Publisher_Check",
        desc="UK publisher is Century.",
        parent=uk_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The UK publisher is {UK_EXPECTED_PUBLISHER}.",
        node=uk_pub_leaf,
        additional_instruction="Judge from the answer text; this refers specifically to the UK edition publisher."
    )

    # Essentials for page count: existence of at least one reference URL and the page count verification
    uk_essentials = evaluator.add_parallel(
        id="UK_PageCount_Essentials",
        desc="UK page count essentials (value + at least one confirming URL).",
        parent=uk_node,
        critical=True
    )

    # Existence of at least one URL explicitly supporting page count
    has_uk_url = bool(uk and uk.page_count_urls and len(uk.page_count_urls) > 0)
    evaluator.add_custom_node(
        result=has_uk_url,
        id="UK_Page_Count_Reference_URL_Check",
        desc="Provides at least one reference URL that explicitly supports the UK edition page count (336) for the Century UK edition.",
        parent=uk_essentials,
        critical=True
    )

    # Page count verification by URLs
    uk_page_leaf = evaluator.add_leaf(
        id="UK_Page_Count_Check",
        desc="UK edition page count is 336 pages.",
        parent=uk_essentials,
        critical=True
    )
    uk_sources = combine_urls(uk.page_count_urls if uk else None, uk.supporting_urls if uk else None)
    await evaluator.verify(
        claim=f"The UK edition published by Century has {UK_EXPECTED_PAGES} pages.",
        node=uk_page_leaf,
        sources=uk_sources,
        additional_instruction=(
            "Verify that the provided webpage(s) explicitly indicate the UK (Century) edition's page count is 336. "
            "Allow minor phrasing like '336 pp' or '336 pages'. The page should clearly pertain to the UK/Century edition."
        )
    )

    # Optional/non-critical additional checks using any available URLs
    all_uk_urls = uk_sources

    # ISBN check
    uk_isbn_leaf = evaluator.add_leaf(
        id="UK_ISBN_Check",
        desc="UK ISBN is 9781529955057.",
        parent=uk_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The UK hardcover ISBN is {UK_EXPECTED_ISBN}.",
        node=uk_isbn_leaf,
        sources=all_uk_urls,
        additional_instruction="Verify on the provided page(s). Accept ISBN with or without hyphens, but digit sequence must match."
    )

    # Hardcover price check
    uk_price_leaf = evaluator.add_leaf(
        id="UK_Hardcover_Price_Check",
        desc="UK hardcover price is £25.00.",
        parent=uk_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The UK hardcover price is {UK_EXPECTED_PRICE}.",
        node=uk_price_leaf,
        sources=all_uk_urls,
        additional_instruction="Verify on the provided page(s). Confirm it's GBP price for the UK hardcover edition."
    )


async def build_audiobook_constraint(evaluator: Evaluator, parent_node, audiobook_len: Optional[str]) -> None:
    ab_node = evaluator.add_parallel(
        id="Audiobook_Constraint",
        desc="Audiobook edition length constraint is met.",
        parent=parent_node,
        critical=False  # Non-critical; not required by the user task
    )

    ab_leaf = evaluator.add_leaf(
        id="Audiobook_Length_Check",
        desc="Audiobook edition is 8 hours long.",
        parent=ab_node,
        critical=False
    )
    await evaluator.verify(
        claim=f"The audiobook edition is {AUDIOBOOK_EXPECTED_LENGTH} long.",
        node=ab_leaf,
        additional_instruction="Judge from the answer text or any cited audiobook page if present. Allow minor format like '8 hrs' or '8h'."
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

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_book_info(),
        template_class=BookExtraction,
        extraction_name="book_extraction"
    )

    # Add GT info for transparency
    evaluator.add_ground_truth({
        "expected": {
            "title": EXPECTED_TITLE,
            "author": EXPECTED_AUTHOR,
            "publication_date": EXPECTED_PUB_DATE,
            "us": {
                "publisher": US_EXPECTED_PUBLISHER,
                "page_count": US_EXPECTED_PAGES,
                "isbn": US_EXPECTED_ISBN,
                "hardcover_price": US_EXPECTED_PRICE,
            },
            "uk": {
                "publisher": UK_EXPECTED_PUBLISHER,
                "page_count": UK_EXPECTED_PAGES,
                "isbn": UK_EXPECTED_ISBN,
                "hardcover_price": UK_EXPECTED_PRICE,
            },
            "audiobook_length": AUDIOBOOK_EXPECTED_LENGTH
        }
    })

    # Task compliance top-level node (non-critical to allow partial credit for ancillary fields)
    task_node = evaluator.add_parallel(
        id="Task_Compliance",
        desc="Answer satisfies all stated constraints and provides the US and UK edition page counts with reference URLs confirming each page count.",
        parent=root,
        critical=False
    )

    # Build subtrees
    await build_core_metadata(evaluator, task_node, extracted)
    await build_us_constraints(evaluator, task_node, extracted.us)
    await build_uk_constraints(evaluator, task_node, extracted.uk)
    await build_audiobook_constraint(evaluator, task_node, extracted.audiobook_length)

    # Return evaluation summary
    return evaluator.get_summary()