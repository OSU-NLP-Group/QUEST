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
TASK_ID = "book_identification_2024_award"
TASK_DESCRIPTION = """Identify a book that meets all of the following criteria:

1. The book won at least one major literary award in 2024, specifically the National Book Award, the Booker Prize, or the Pulitzer Prize for Fiction
2. The book was first published in the United States in 2024
3. The book is classified as literary fiction
4. The book is published by one of the Big Five publishing houses (Penguin Random House, HarperCollins, Simon & Schuster, Hachette Book Group, or Macmillan Publishers)
5. The standard hardcover edition contains between 200 and 500 pages
6. The author has published at least one previous book before this award-winning work
7. The award was announced between October 2024 and December 2024
8. The book is available in both hardcover and ebook formats

Please provide the book title, author name, the specific award it won, the publisher, and reference URLs that verify each of the criteria.
"""

ALLOWED_AWARDS = [
    "National Book Award",
    "Booker Prize",
    "Pulitzer Prize for Fiction",
]
BIG_FIVE_PUBLISHERS = [
    "Penguin Random House",
    "HarperCollins",
    "Simon & Schuster",
    "Hachette Book Group",
    "Macmillan Publishers",
]

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BookItem(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    award: Optional[str] = None  # e.g., "National Book Award" or "Booker Prize" or "Pulitzer Prize for Fiction"
    publisher: Optional[str] = None
    publication_year_us: Optional[str] = None  # e.g., "2024"
    genre: Optional[str] = None  # e.g., "Literary Fiction"
    hardcover_page_count: Optional[str] = None  # keep as string (e.g., "352 pages")
    formats: List[str] = Field(default_factory=list)  # e.g., ["hardcover", "ebook", "paperback"]
    award_announcement_date: Optional[str] = None  # e.g., "November 20, 2024"
    author_prev_works_mentioned: Optional[str] = None  # textual mention, if any

class BookSources(BaseModel):
    award_urls: List[str] = Field(default_factory=list)               # verifies award won in 2024
    award_timeline_urls: List[str] = Field(default_factory=list)      # verifies announcement date Oct–Dec 2024
    publication_year_urls: List[str] = Field(default_factory=list)    # verifies first US publication year is 2024
    genre_urls: List[str] = Field(default_factory=list)               # verifies literary fiction classification
    publisher_urls: List[str] = Field(default_factory=list)           # verifies publisher for the book
    page_count_urls: List[str] = Field(default_factory=list)          # verifies hardcover page count
    author_prev_urls: List[str] = Field(default_factory=list)         # verifies author had previous book(s)
    format_urls: List[str] = Field(default_factory=list)              # verifies hardcover + ebook formats available

class BookExtraction(BaseModel):
    book: Optional[BookItem] = None
    sources: BookSources = Field(default_factory=BookSources)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_book_info() -> str:
    return """
    Extract the single book identified in the answer along with criterion-specific reference URLs.

    Return a JSON object with two top-level keys:
      - book: object with fields
          • title: the book title exactly as written in the answer
          • author: the author name exactly as written in the answer
          • award: the specific award mentioned as won in 2024 (e.g., "National Book Award", "Booker Prize", or "Pulitzer Prize for Fiction")
          • publisher: publisher name
          • publication_year_us: the first US publication year
          • genre: genre classification as mentioned (e.g., "Literary Fiction")
          • hardcover_page_count: page count string for the standard hardcover edition (e.g., "352 pages")
          • formats: array of formats that the answer claims are available (e.g., ["hardcover", "ebook", "paperback"])
          • award_announcement_date: the announcement date string if explicitly mentioned (e.g., "November 20, 2024"); otherwise null
          • author_prev_works_mentioned: any textual mention that author has previous works; otherwise null

      - sources: object with arrays of URLs that the answer explicitly provides for verifying EACH criterion:
          • award_urls: URLs that verify the book won the specified major award in 2024
          • award_timeline_urls: URLs that verify the award announcement date is between October and December 2024
          • publication_year_urls: URLs that verify first US publication year is 2024
          • genre_urls: URLs that verify literary fiction classification
          • publisher_urls: URLs that verify the stated publisher for the book
          • page_count_urls: URLs that verify the hardcover page count
          • author_prev_urls: URLs that verify the author had published at least one previous book
          • format_urls: URLs that verify the book is available in both hardcover and ebook formats

    IMPORTANT:
    - Only include URLs that are explicitly present in the answer.
    - If a required field is missing in the answer, set it to null.
    - If no URLs are provided for a criterion, return an empty array for that criterion.
    - Accept URLs in plain or markdown form; extract the actual URL.
    """

# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())

def _has_both_formats(formats: List[str]) -> bool:
    """Check if formats list includes both 'hardcover' and an ebook synonym."""
    fmts = {f.lower().strip() for f in formats}
    has_hard = any("hardcover" in f for f in fmts)
    # Accept synonyms for ebook
    ebook_synonyms = ["ebook", "e-book", "digital", "kindle", "nook"]
    has_ebook = any(any(e in f for e in ebook_synonyms) for f in fmts)
    return has_hard and has_ebook

# --------------------------------------------------------------------------- #
# Verification builders per criterion                                         #
# --------------------------------------------------------------------------- #
async def add_award_verification(evaluator: Evaluator, parent, b: BookItem, src: BookSources):
    group = evaluator.add_sequential(
        id="Award_Won_2024",
        desc="The book won at least one major literary award in 2024 (National Book Award, Booker Prize, or Pulitzer Prize for Fiction)",
        parent=parent,
        critical=True,
    )
    # Existence / sources check
    evaluator.add_custom_node(
        result=_non_empty(b.award) and len(src.award_urls) > 0,
        id="award_sources_provided",
        desc="Award verification sources are provided in the answer",
        parent=group,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="award_won_2024_verify",
        desc="Verify the 2024 major award win via cited sources",
        parent=group,
        critical=True,
    )
    claim = f"The book '{b.title or ''}' by {b.author or ''} won the {b.award or ''} in 2024; the award must be one of: National Book Award, Booker Prize, or Pulitzer Prize for Fiction."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=src.award_urls,
        additional_instruction="Use the provided sources to confirm both (a) the specific award name and (b) the win year is 2024. Accept reasonable variants (e.g., 'National Book Award for Fiction') as long as the award is one of the three specified major awards."
    )

async def add_award_timeline_verification(evaluator: Evaluator, parent, b: BookItem, src: BookSources):
    group = evaluator.add_sequential(
        id="Award_Timeline",
        desc="The award was announced between October 2024 and December 2024",
        parent=parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(src.award_timeline_urls) > 0,
        id="award_timeline_sources_provided",
        desc="Award announcement timeline sources are provided in the answer",
        parent=group,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="award_timeline_verify",
        desc="Verify award announcement occurred Oct–Dec 2024 via sources",
        parent=group,
        critical=True,
    )
    claim = f"The {b.award or ''} for '{b.title or ''}' was announced between October 2024 and December 2024."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=src.award_timeline_urls,
        additional_instruction="Use the official award site or credible news sources. Match announcement date (not nomination) to be within Oct–Dec 2024. Minor time-zone differences are acceptable."
    )

async def add_publication_year_verification(evaluator: Evaluator, parent, b: BookItem, src: BookSources):
    group = evaluator.add_sequential(
        id="Publication_Year_2024",
        desc="The book was first published in the United States in 2024",
        parent=parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(b.publication_year_us) and len(src.publication_year_urls) > 0,
        id="publication_year_sources_provided",
        desc="US first publication year sources are provided in the answer",
        parent=group,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="publication_year_verify",
        desc="Verify first US publication year is 2024",
        parent=group,
        critical=True,
    )
    claim = f"The book '{b.title or ''}' was first published in the United States in 2024."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=src.publication_year_urls,
        additional_instruction="Verify the US edition first publication year (not international). If multiple dates exist, ensure the first US publication date is in 2024."
    )

async def add_genre_verification(evaluator: Evaluator, parent, b: BookItem, src: BookSources):
    group = evaluator.add_sequential(
        id="Literary_Fiction_Genre",
        desc="The book is classified as literary fiction",
        parent=parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(b.genre) and len(src.genre_urls) > 0,
        id="genre_sources_provided",
        desc="Genre classification sources are provided in the answer",
        parent=group,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="genre_verify",
        desc="Verify literary fiction classification",
        parent=group,
        critical=True,
    )
    claim = f"The book '{b.title or ''}' is classified as literary fiction."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=src.genre_urls,
        additional_instruction="Accept classification from the publisher, reputable review outlets, library catalogues, or bookstore listings if they explicitly classify the work as literary fiction."
    )

async def add_publisher_verification(evaluator: Evaluator, parent, b: BookItem, src: BookSources):
    group = evaluator.add_sequential(
        id="Major_Publisher",
        desc="The book is published by one of the Big Five publishers (Penguin Random House, HarperCollins, Simon & Schuster, Hachette Book Group, or Macmillan Publishers)",
        parent=parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(b.publisher) and len(src.publisher_urls) > 0,
        id="publisher_sources_provided",
        desc="Publisher verification sources are provided in the answer",
        parent=group,
        critical=True,
    )

    # Verify the book's publisher via sources
    leaf_verify_pub = evaluator.add_leaf(
        id="publisher_verify_via_sources",
        desc="Verify the stated publisher for the book via sources",
        parent=group,
        critical=True,
    )
    claim_pub = f"The book '{b.title or ''}' is published by {b.publisher or ''}."
    await evaluator.verify(
        claim=claim_pub,
        node=leaf_verify_pub,
        sources=src.publisher_urls,
        additional_instruction="Confirm the publisher listed on official publisher pages, bookstore listings, or authoritative bibliographic sources."
    )

    # Verify Big Five membership (logical check)
    leaf_big5 = evaluator.add_leaf(
        id="publisher_big_five_membership",
        desc="Publisher is one of the Big Five",
        parent=group,
        critical=True,
    )
    claim_big5 = f"The publisher '{b.publisher or ''}' is one of the Big Five publishers: {', '.join(BIG_FIVE_PUBLISHERS)}."
    await evaluator.verify(
        claim=claim_big5,
        node=leaf_big5,
        additional_instruction="This is a logical membership check: pass if the stated publisher string exactly matches one of the five names provided."
    )

async def add_page_count_verification(evaluator: Evaluator, parent, b: BookItem, src: BookSources):
    group = evaluator.add_sequential(
        id="Page_Count_Range",
        desc="The standard hardcover edition has between 200 and 500 pages",
        parent=parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_non_empty(b.hardcover_page_count) and len(src.page_count_urls) > 0,
        id="page_count_sources_provided",
        desc="Hardcover page count sources are provided in the answer",
        parent=group,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="page_count_range_verify",
        desc="Verify hardcover page count is between 200 and 500 pages",
        parent=group,
        critical=True,
    )
    claim = f"The standard hardcover edition of '{b.title or ''}' has {b.hardcover_page_count or ''}, which is between 200 and 500 pages."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=src.page_count_urls,
        additional_instruction="Use the hardcover edition page listing. Treat 200 and 500 as inclusive bounds. Accept reasonable variants like '352 pages'."
    )

async def add_author_prev_works_verification(evaluator: Evaluator, parent, b: BookItem, src: BookSources):
    group = evaluator.add_sequential(
        id="Author_Previous_Works",
        desc="The author has published at least one previous book before this work",
        parent=parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(src.author_prev_urls) > 0,
        id="author_prev_sources_provided",
        desc="Author previous works sources are provided in the answer",
        parent=group,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="author_prev_works_verify",
        desc="Verify the author has at least one previous book",
        parent=group,
        critical=True,
    )
    claim = f"The author {b.author or ''} had published at least one book before '{b.title or ''}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=src.author_prev_urls,
        additional_instruction="Use bibliographies, author pages, or reputable catalogs to confirm at least one earlier book existed before the award-winning work."
    )

async def add_format_availability_verification(evaluator: Evaluator, parent, b: BookItem, src: BookSources):
    group = evaluator.add_sequential(
        id="Format_Availability",
        desc="The book is available in both hardcover and ebook formats",
        parent=parent,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_both_formats(b.formats) and len(src.format_urls) > 0,
        id="format_sources_provided",
        desc="Format availability (hardcover + ebook) is stated in the answer and sources are provided",
        parent=group,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="format_availability_verify",
        desc="Verify the book is available in both hardcover and ebook formats",
        parent=group,
        critical=True,
    )
    claim = f"The book '{b.title or ''}' is available in both hardcover and ebook formats."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=src.format_urls,
        additional_instruction="Treat 'Kindle', 'digital', or similar terms as ebook equivalents. Confirm both hardcover and an ebook format are available via the cited sources."
    )

# --------------------------------------------------------------------------- #
# Main verification orchestration                                             #
# --------------------------------------------------------------------------- #
async def verify_book(evaluator: Evaluator, root_node, extraction: BookExtraction) -> None:
    # Prepare objects
    book = extraction.book or BookItem()
    srcs = extraction.sources or BookSources()

    # Top-level critical group mirroring rubric root
    top = evaluator.add_parallel(
        id="Book_Identification",
        desc="Identify a book that satisfies all specified award, publication, and format criteria",
        parent=root_node,
        critical=True,
    )

    # Core identification presence check (title + author present)
    evaluator.add_custom_node(
        result=_non_empty(book.title) and _non_empty(book.author),
        id="book_core_info_provided",
        desc="Book title and author are provided in the answer",
        parent=top,
        critical=True,
    )

    # Add all criterion verifications under the top-level critical node
    await add_award_verification(evaluator, top, book, srcs)
    await add_award_timeline_verification(evaluator, top, book, srcs)
    await add_publication_year_verification(evaluator, top, book, srcs)
    await add_genre_verification(evaluator, top, book, srcs)
    await add_publisher_verification(evaluator, top, book, srcs)
    await add_page_count_verification(evaluator, top, book, srcs)
    await add_author_prev_works_verification(evaluator, top, book, srcs)
    await add_format_availability_verification(evaluator, top, book, srcs)

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
    Evaluate an answer for the 2024 award-winning literary fiction book identification task.
    """
    # Initialize evaluator (root is non-critical container; we add critical node beneath)
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

    # Record ground truth lists for reference
    evaluator.add_ground_truth({
        "allowed_awards": ALLOWED_AWARDS,
        "big_five_publishers": BIG_FIVE_PUBLISHERS
    }, gt_type="constraints")

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_book_info(),
        template_class=BookExtraction,
        extraction_name="book_extraction",
    )

    # Build and run verification tree
    await verify_book(evaluator, root, extraction)

    # Return structured summary
    return evaluator.get_summary()