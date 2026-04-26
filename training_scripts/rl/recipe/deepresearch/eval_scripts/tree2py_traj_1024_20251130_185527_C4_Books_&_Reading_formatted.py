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
TASK_ID = "celebrity_memoir_2024"
TASK_DESCRIPTION = (
    "Identify a celebrity memoir published in 2024 that would be suitable for a book club selection. "
    "The memoir must meet the following criteria: published in 2024, written by a celebrity or well-known figure from the entertainment industry, "
    "under 400 pages in length (to accommodate the preference of most book clubs), available in hardcover format, and published by an established publishing house. "
    "Provide the following information: (1) book title and author name, (2) exact number of pages, (3) publisher name, (4) specific publication date (month and year), "
    "and (5) confirmation that it is available in hardcover format."
)


# --------------------------------------------------------------------------- #
# Data model for extraction                                                   #
# --------------------------------------------------------------------------- #
class BookSelection(BaseModel):
    """Single selected memoir with bibliographic fields and all URLs mentioned in the answer."""
    title: Optional[str] = None
    author: Optional[str] = None
    page_count: Optional[str] = None  # keep as string to maximize robustness; we'll parse number later
    publisher: Optional[str] = None
    publication_month: Optional[str] = None  # e.g., "January", "Feb", "June"
    publication_year: Optional[str] = None   # e.g., "2024"
    hardcover_mentioned: Optional[bool] = None  # optional; verification will rely on URLs, not this flag

    # URL buckets extracted exactly as presented in the answer text
    book_urls: List[str] = Field(default_factory=list)       # links that list the book (publisher page, Goodreads, Amazon, B&N, etc.)
    publisher_urls: List[str] = Field(default_factory=list)  # official publisher pages relevant to the book or publisher
    author_urls: List[str] = Field(default_factory=list)     # author bio/official page/Wikipedia/IMDb, etc.
    retailer_urls: List[str] = Field(default_factory=list)   # Amazon, Barnes & Noble, Bookshop, IndieBound, etc.
    other_urls: List[str] = Field(default_factory=list)      # any additional URLs mentioned for evidence


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_book_selection() -> str:
    return """
    From the provided answer, extract details for exactly one (1) selected celebrity memoir that the answer proposes as the book club choice.
    If the answer lists multiple books, select the first one that is explicitly recommended or described as meeting the criteria.
    Do NOT invent or infer any information that is not explicitly present in the answer text.

    Return a JSON object with the following fields:
    - title: string or null
    - author: string or null
    - page_count: string or null (extract exactly as written, e.g., "384 pages", "352", "352 pp.")
    - publisher: string or null
    - publication_month: string or null (e.g., "January", "Jan", "June"; extract exactly as stated)
    - publication_year: string or null (e.g., "2024"; extract exactly as stated)
    - hardcover_mentioned: boolean or null (true if the answer explicitly states a hardcover/hardback edition is available)
    - book_urls: array of strings; include URLs that directly list the book (publisher page, Amazon, Barnes & Noble, Goodreads, Bookshop, etc.) as explicitly written in the answer
    - publisher_urls: array of strings; include official publisher webpages (home page, imprint information, book page) explicitly listed in the answer
    - author_urls: array of strings; include author bio/official page, Wikipedia, IMDb, etc., explicitly listed in the answer
    - retailer_urls: array of strings; include retailer pages (Amazon, B&N, etc.) if explicitly listed in the answer
    - other_urls: array of strings; any other URLs explicitly present in the answer not covered above

    SPECIAL RULES:
    - Only include URLs explicitly present in the answer. Do not create or infer any URL.
    - For markdown links, extract the actual URL target.
    - If a field is not present, return null (for scalars) or [] (for arrays).

    Your output must be valid JSON matching the specified schema exactly.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_first_int(text: Optional[str]) -> Optional[int]:
    """Extract the first integer found in a string, e.g., '384 pages' -> 384."""
    if not text:
        return None
    m = re.search(r"\d{1,5}", text.replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def dedupe_urls(urls: List[str]) -> List[str]:
    """Deduplicate URLs while preserving order."""
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def combine_book_sources(sel: BookSelection) -> List[str]:
    """Combine all book-relevant URLs for verifying book facts."""
    return dedupe_urls((sel.book_urls or []) + (sel.publisher_urls or []) + (sel.retailer_urls or []) + (sel.other_urls or []))


def combine_author_sources(sel: BookSelection) -> List[str]:
    """Combine author-related URLs for verifying celebrity/entertainment status."""
    return dedupe_urls((sel.author_urls or []) + (sel.other_urls or []))


def combine_publisher_sources(sel: BookSelection) -> List[str]:
    """Combine publisher-related URLs for verifying publisher status."""
    return dedupe_urls((sel.publisher_urls or []) + (sel.other_urls or []) + (sel.book_urls or []))


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def verify_book_selection(evaluator: Evaluator, parent_node, sel: BookSelection) -> None:
    """
    Build verification nodes under the given parent and run checks for the selected book.
    All children are critical, matching the rubric.
    """
    # Prepare sources
    book_sources = combine_book_sources(sel)
    author_sources = combine_author_sources(sel)
    publisher_sources = combine_publisher_sources(sel)

    # 1) book_identification: title and author must be provided (existence check)
    has_title_author = (sel.title is not None and str(sel.title).strip() != "") and (sel.author is not None and str(sel.author).strip() != "")
    evaluator.add_custom_node(
        result=has_title_author,
        id="book_identification",
        desc="A specific memoir book title and author name must be identified and provided.",
        parent=parent_node,
        critical=True
    )

    # 2) publication_year_2024: verify via sources that the book was published in 2024
    node_pub_year = evaluator.add_leaf(
        id="publication_year_2024",
        desc="The book must be published in 2024.",
        parent=parent_node,
        critical=True
    )
    claim_pub_year = "This book was published in 2024."
    await evaluator.verify(
        claim=claim_pub_year,
        node=node_pub_year,
        sources=book_sources if book_sources else None,
        additional_instruction="Check the publication/issue date or metadata on the provided pages. If multiple dates appear, use the official publication date for the listed edition. The year must be 2024."
    )

    # 3) memoir_genre: verify it's a memoir/autobiography
    node_genre = evaluator.add_leaf(
        id="memoir_genre",
        desc="The book must be a memoir or autobiography (personal narrative written by the subject about their own life).",
        parent=parent_node,
        critical=True
    )
    claim_genre = "This book is a memoir or an autobiography (a personal narrative by the author about their own life)."
    await evaluator.verify(
        claim=claim_genre,
        node=node_genre,
        sources=book_sources if book_sources else None,
        additional_instruction="Look for genre labels, descriptions, or metadata indicating 'memoir', 'autobiography', or 'memoir-in-essays'."
    )

    # 4) celebrity_author: verify the author is a celebrity or well-known figure from entertainment industry
    node_celebrity = evaluator.add_leaf(
        id="celebrity_author",
        desc="The author must be a celebrity or well-known figure from the entertainment industry (e.g., actor, musician, TV personality).",
        parent=parent_node,
        critical=True
    )
    author_name = sel.author or "the author"
    claim_celebrity = f"The author {author_name} is a celebrity or well-known figure in the entertainment industry (e.g., actor, musician, TV personality)."
    await evaluator.verify(
        claim=claim_celebrity,
        node=node_celebrity,
        sources=(author_sources if author_sources else book_sources if book_sources else None),
        additional_instruction="Use the provided links (e.g., Wikipedia, official sites, IMDb, press pages) to confirm that the author is from the entertainment industry."
    )

    # 5) page_length_under_400: parse the page count and ensure it's < 400
    pages_num = parse_first_int(sel.page_count)
    under_400 = pages_num is not None and 1 <= pages_num < 400
    evaluator.add_custom_node(
        result=under_400,
        id="page_length_under_400",
        desc="The book must be under 400 pages in length.",
        parent=parent_node,
        critical=True
    )

    # 6) exact_page_count: the exact page count must be specified in the answer (existence of a concrete number)
    exact_page_present = pages_num is not None
    evaluator.add_custom_node(
        result=exact_page_present,
        id="exact_page_count",
        desc="The exact number of pages must be specified.",
        parent=parent_node,
        critical=True
    )

    # 7) publisher_name_provided: existence check
    publisher_present = sel.publisher is not None and str(sel.publisher).strip() != ""
    evaluator.add_custom_node(
        result=publisher_present,
        id="publisher_name_provided",
        desc="The publisher name must be provided.",
        parent=parent_node,
        critical=True
    )

    # 8) publisher_established_major: verify via sources that publisher is established/major (or an imprint of one)
    node_publisher_major = evaluator.add_leaf(
        id="publisher_established_major",
        desc="The publisher must be a major/established publishing house.",
        parent=parent_node,
        critical=True
    )
    publisher_name = sel.publisher or "the publisher"
    claim_publisher_major = (
        f"{publisher_name} is an established publishing house, or an imprint/division owned by a major publisher."
    )
    await evaluator.verify(
        claim=claim_publisher_major,
        node=node_publisher_major,
        sources=(publisher_sources if publisher_sources else book_sources if book_sources else None),
        additional_instruction=(
            "Accept Big Five publishers and their imprints (Penguin Random House, HarperCollins, Simon & Schuster, Hachette, Macmillan) "
            "and other widely recognized established houses. Evidence can include the publisher's official site or reputable references (e.g., Wikipedia) indicating "
            "it is a major house or an imprint/division thereof."
        )
    )

    # 9) publication_date_details: month and year must be provided (existence check at month-year granularity)
    month_year_present = (sel.publication_month is not None and str(sel.publication_month).strip() != "") and \
                         (sel.publication_year is not None and str(sel.publication_year).strip() != "")
    evaluator.add_custom_node(
        result=month_year_present,
        id="publication_date_details",
        desc="The publication date must be provided at least to the month and year level.",
        parent=parent_node,
        critical=True
    )

    # 10) hardcover_format: verify via sources that a hardcover/hardback edition exists
    node_hardcover = evaluator.add_leaf(
        id="hardcover_format",
        desc="The book must be available in hardcover format.",
        parent=parent_node,
        critical=True
    )
    claim_hardcover = "A hardcover (hardback) edition of this book is available."
    await evaluator.verify(
        claim=claim_hardcover,
        node=node_hardcover,
        sources=book_sources if book_sources else None,
        additional_instruction="Look for format options showing 'Hardcover' or 'Hardback' on retailer or publisher pages."
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
    Evaluate an answer for the 2024 celebrity memoir selection task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # top-level aggregation
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

    # Add a critical "main" node under the (non-critical) framework root to mirror rubric's critical root
    main = evaluator.add_parallel(
        id="main_verification",
        desc="Identify one celebrity memoir published in 2024 that meets all stated constraints and provide the required bibliographic/format details.",
        parent=root,
        critical=True
    )

    # Extract the selected book details from the answer
    selection: BookSelection = await evaluator.extract(
        prompt=prompt_extract_book_selection(),
        template_class=BookSelection,
        extraction_name="book_selection"
    )

    # Optionally record the constraints as custom info
    evaluator.add_custom_info(
        {
            "required_year": 2024,
            "max_pages": 399,
            "required_fields": [
                "title", "author", "page_count", "publisher", "publication_month", "publication_year", "hardcover availability"
            ],
            "publisher_status": "major/established or imprint of a major house"
        },
        info_type="constraints",
        info_name="task_constraints"
    )

    # Build verification nodes and run checks
    await verify_book_selection(evaluator, main, selection)

    # Return structured summary
    return evaluator.get_summary()