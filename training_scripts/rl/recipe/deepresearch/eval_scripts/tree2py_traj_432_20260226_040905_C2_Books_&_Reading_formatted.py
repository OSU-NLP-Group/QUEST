import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "celebrity_memoir_2024_2025"
TASK_DESCRIPTION = (
    "Identify a celebrity memoir that was published by a major U.S. publishing house between January 1, 2024, and "
    "December 31, 2025. The memoir must meet the following requirements: (1) Published by a major U.S. publishing "
    "house (such as Crown Publishing Group, Simon & Schuster, Penguin Random House, HarperCollins, Hachette Book Group, "
    "or other established major publishers); (2) Written by a celebrity or public figure with notable recognition in "
    "entertainment, media, sports, politics, or similar fields; (3) Has an audiobook version available that is at least "
    "8 hours in length; (4) The print edition has at least 300 pages; (5) The audiobook narrator information is publicly "
    "available. For your answer, provide the following information with reference URLs from official sources (publisher "
    "websites, Amazon, Audible, Goodreads, or library catalogs): book title and author, publisher name, publication date, "
    "page count, ISBN, audiobook length, and audiobook narrator(s)."
)

MAJOR_PUBLISHERS_EXAMPLES = [
    "Penguin Random House",
    "Crown Publishing Group",
    "Simon & Schuster",
    "HarperCollins",
    "Hachette Book Group",
    "Macmillan",
    "Vintage",
    "Knopf",
    "Scribner",
    "Random House",
    "Little, Brown and Company"
]

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MemoirExtraction(BaseModel):
    """
    Structured extraction of the memoir details from the agent's answer.
    All fields are extracted as strings (or list of strings) to maximize flexibility.
    URLs must be explicitly present in the answer text and categorized when possible.
    """
    title: Optional[str] = None
    author: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[str] = None  # e.g., "October 15, 2024"
    page_count: Optional[str] = None        # e.g., "320", "320 pages"
    isbn: Optional[str] = None
    audiobook_length: Optional[str] = None  # e.g., "9 hours 12 minutes"
    narrators: List[str] = Field(default_factory=list)

    # URLs from official sources (must be explicitly present in the answer)
    publisher_refs: List[str] = Field(default_factory=list)
    amazon_refs: List[str] = Field(default_factory=list)
    audible_refs: List[str] = Field(default_factory=list)
    goodreads_refs: List[str] = Field(default_factory=list)
    library_refs: List[str] = Field(default_factory=list)
    other_refs: List[str] = Field(default_factory=list)

    # All references (union of above, but extracted explicitly as provided)
    references: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_memoir() -> str:
    return """
    Extract the details of a single celebrity memoir from the provided answer. Return the following fields:
    - title: Exact book title as given in the answer.
    - author: Author name(s) as given.
    - publisher: Publisher or imprint name as given.
    - publication_date: The publication/release date string as stated.
    - page_count: The print edition page count (string, e.g., "320" or "320 pages").
    - isbn: The ISBN string as stated.
    - audiobook_length: Audiobook runtime string (e.g., "9 hours 12 minutes", "08:45:00").
    - narrators: List of narrator names (each as a string).
    - publisher_refs: List of official publisher website URLs provided in the answer for this book.
    - amazon_refs: List of official Amazon product URLs (book or audiobook) provided in the answer.
    - audible_refs: List of official Audible product URLs provided in the answer.
    - goodreads_refs: List of Goodreads URLs provided in the answer.
    - library_refs: List of library catalog URLs provided in the answer.
    - other_refs: Any other official source URLs provided (if any).
    - references: A list containing all URLs that are explicitly present in the answer.

    IMPORTANT:
    - Only include URLs explicitly present in the answer (including markdown links).
    - If a required field is missing, set it to null.
    - If a URL category has no entries, return an empty list for that category.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def concat_sources(*lists: List[str]) -> List[str]:
    """Concatenate and deduplicate URL lists, preserving order."""
    seen = set()
    result = []
    for lst in lists:
        for url in lst:
            if url and url not in seen:
                seen.add(url)
                result.append(url)
    return result


def str_or_placeholder(value: Optional[str], placeholder: str = "N/A") -> str:
    """Return a safe string value for claims."""
    return value.strip() if isinstance(value, str) and value.strip() else placeholder


def join_names(names: List[str]) -> str:
    """Join list of names into a readable string for verification claims."""
    cleaned = [n.strip() for n in names if isinstance(n, str) and n.strip()]
    return ", ".join(cleaned) if cleaned else "N/A"


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_basic_publication_criteria(
    evaluator: Evaluator,
    parent_node,
    ex: MemoirExtraction
) -> None:
    """
    Build and verify the 'basic_publication_criteria' parallel critical node:
    - major_publisher
    - publication_timeframe
    - celebrity_author
    """
    basic_node = evaluator.add_parallel(
        id="basic_publication_criteria",
        desc="Verify that the memoir meets fundamental publication requirements including publisher type, publication timeframe, and author status",
        parent=parent_node,
        critical=True
    )

    # 1) Major publisher check
    major_pub_node = evaluator.add_leaf(
        id="major_publisher",
        desc="The memoir is published by a major U.S. publishing house",
        parent=basic_node,
        critical=True
    )

    publisher_val = str_or_placeholder(ex.publisher)
    major_pub_sources = concat_sources(ex.publisher_refs, ex.amazon_refs, ex.goodreads_refs, ex.library_refs, ex.references)

    major_pub_claim = (
        f"The publisher of this book is '{publisher_val}', and it is a major U.S. publishing house "
        f"(e.g., among {', '.join(MAJOR_PUBLISHERS_EXAMPLES)} or an imprint thereof)."
    )
    await evaluator.verify(
        claim=major_pub_claim,
        node=major_pub_node,
        sources=major_pub_sources,
        additional_instruction=(
            "Confirm the publisher identity from official sources (publisher site, Amazon, Goodreads, library). "
            "Consider well-known imprints owned by major houses as 'major'."
        )
    )

    # 2) Publication timeframe check (must be between 2024-01-01 and 2025-12-31)
    pub_time_node = evaluator.add_leaf(
        id="publication_timeframe",
        desc="The memoir was released between January 1, 2024, and December 31, 2025",
        parent=basic_node,
        critical=True
    )

    pub_time_sources = concat_sources(ex.publisher_refs, ex.amazon_refs, ex.goodreads_refs, ex.library_refs, ex.references)
    pub_date_val = str_or_placeholder(ex.publication_date)
    pub_time_claim = (
        f"The memoir's publication/release date is '{pub_date_val}', and that date falls between January 1, 2024 "
        f"and December 31, 2025."
    )
    await evaluator.verify(
        claim=pub_time_claim,
        node=pub_time_node,
        sources=pub_time_sources,
        additional_instruction=(
            "Use the cited official source pages to verify the publication date and confirm that it lies within "
            "the required timeframe [2024-01-01, 2025-12-31]."
        )
    )

    # 3) Celebrity author check
    celeb_node = evaluator.add_leaf(
        id="celebrity_author",
        desc="The memoir is written by a celebrity or public figure with notable recognition",
        parent=basic_node,
        critical=True
    )

    celeb_sources = concat_sources(ex.publisher_refs, ex.amazon_refs, ex.goodreads_refs, ex.library_refs, ex.references)
    author_val = str_or_placeholder(ex.author)
    celeb_claim = (
        f"The author '{author_val}' is a celebrity or public figure with notable recognition in entertainment, media, "
        f"sports, politics, or similar fields."
    )
    await evaluator.verify(
        claim=celeb_claim,
        node=celeb_node,
        sources=celeb_sources,
        additional_instruction=(
            "Rely on official or authoritative product/catalouge pages provided to judge public recognition. "
            "If the pages clearly indicate wide public prominence (e.g., mentions awards, public office, "
            "major media presence), consider this criterion satisfied."
        )
    )


async def verify_format_specifications(
    evaluator: Evaluator,
    parent_node,
    ex: MemoirExtraction
) -> None:
    """
    Build and verify the 'format_specifications' parallel critical node:
    - audiobook_availability_and_length
    - print_page_count
    - verification_documentation (sub-parallel critical node with individual field checks)
    """
    format_node = evaluator.add_parallel(
        id="format_specifications",
        desc="Verify that the memoir meets audiobook and print format requirements with proper documentation",
        parent=parent_node,
        critical=True
    )

    # Audiobook availability and length >= 8 hours
    audio_len_node = evaluator.add_leaf(
        id="audiobook_availability_and_length",
        desc="The memoir has an audiobook version available and the audiobook is at least 8 hours in length",
        parent=format_node,
        critical=True
    )

    audio_sources = concat_sources(ex.audible_refs, ex.amazon_refs, ex.references)
    audio_len_val = str_or_placeholder(ex.audiobook_length)
    audio_len_claim = (
        f"An audiobook version exists and its runtime is '{audio_len_val}', which is at least 8 hours."
    )
    await evaluator.verify(
        claim=audio_len_claim,
        node=audio_len_node,
        sources=audio_sources,
        additional_instruction=(
            "Verify on Audible/Amazon audiobook pages (or equivalent official sources) that the audiobook exists and "
            "its length is ≥ 8 hours (480 minutes). Consider typical runtime formats like 'X hours Y minutes' or 'HH:MM:SS'."
        )
    )

    # Print page count >= 300
    page_count_node = evaluator.add_leaf(
        id="print_page_count",
        desc="The print edition of the memoir has at least 300 pages",
        parent=format_node,
        critical=True
    )

    page_sources = concat_sources(ex.publisher_refs, ex.amazon_refs, ex.goodreads_refs, ex.library_refs, ex.references)
    page_count_val = str_or_placeholder(ex.page_count)
    page_count_claim = (
        f"The print edition page count is '{page_count_val}', which is at least 300 pages."
    )
    await evaluator.verify(
        claim=page_count_claim,
        node=page_count_node,
        sources=page_sources,
        additional_instruction=(
            "Verify the print page count from official sources (publisher site, Amazon detail page, Goodreads, library catalog). "
            "Accept reasonable formatting variants (e.g., '320 pages')."
        )
    )

    # Documentation verification: verify each field value is supported by official sources
    docs_node = evaluator.add_parallel(
        id="verification_documentation",
        desc="All required details are verifiable via official sources with reference URLs provided",
        parent=format_node,
        critical=True
    )

    # Official sources presence gate
    official_sources_present = evaluator.add_custom_node(
        result=bool(ex.publisher_refs or ex.amazon_refs or ex.audible_refs or ex.goodreads_refs or ex.library_refs),
        id="official_sources_present",
        desc="At least one official source URL (publisher, Amazon, Audible, Goodreads, or library catalog) is provided",
        parent=docs_node,
        critical=True
    )

    # Publisher name verifiable
    doc_publisher_node = evaluator.add_leaf(
        id="document_publisher",
        desc="Publisher name is verifiable through official sources",
        parent=docs_node,
        critical=True
    )
    doc_publisher_claim = f"The publisher of this book is '{str_or_placeholder(ex.publisher)}'."
    await evaluator.verify(
        claim=doc_publisher_claim,
        node=doc_publisher_node,
        sources=concat_sources(ex.publisher_refs, ex.amazon_refs, ex.goodreads_refs, ex.library_refs, ex.references),
        additional_instruction="Confirm the publisher name shown on official source pages."
    )

    # Publication date verifiable
    doc_pub_date_node = evaluator.add_leaf(
        id="document_publication_date",
        desc="Publication date is verifiable through official sources",
        parent=docs_node,
        critical=True
    )
    doc_pub_date_claim = f"The publication/release date is '{str_or_placeholder(ex.publication_date)}'."
    await evaluator.verify(
        claim=doc_pub_date_claim,
        node=doc_pub_date_node,
        sources=concat_sources(ex.publisher_refs, ex.amazon_refs, ex.goodreads_refs, ex.library_refs, ex.references),
        additional_instruction="Confirm the publication date shown on the official source pages."
    )

    # Page count verifiable
    doc_page_count_node = evaluator.add_leaf(
        id="document_page_count",
        desc="Page count is verifiable through official sources",
        parent=docs_node,
        critical=True
    )
    doc_page_count_claim = f"The print edition page count is '{str_or_placeholder(ex.page_count)}'."
    await evaluator.verify(
        claim=doc_page_count_claim,
        node=doc_page_count_node,
        sources=concat_sources(ex.publisher_refs, ex.amazon_refs, ex.goodreads_refs, ex.library_refs, ex.references),
        additional_instruction="Confirm the page count shown on the official source pages."
    )

    # ISBN verifiable
    doc_isbn_node = evaluator.add_leaf(
        id="document_isbn",
        desc="ISBN is verifiable through official sources",
        parent=docs_node,
        critical=True
    )
    doc_isbn_claim = f"The ISBN is '{str_or_placeholder(ex.isbn)}'."
    await evaluator.verify(
        claim=doc_isbn_claim,
        node=doc_isbn_node,
        sources=concat_sources(ex.publisher_refs, ex.amazon_refs, ex.goodreads_refs, ex.library_refs, ex.references),
        additional_instruction="Confirm the ISBN shown on the official source pages."
    )

    # Audiobook length verifiable
    doc_audio_len_node = evaluator.add_leaf(
        id="document_audiobook_length",
        desc="Audiobook length is verifiable through official sources",
        parent=docs_node,
        critical=True
    )
    doc_audio_len_claim = f"The audiobook length is '{str_or_placeholder(ex.audiobook_length)}'."
    await evaluator.verify(
        claim=doc_audio_len_claim,
        node=doc_audio_len_node,
        sources=concat_sources(ex.audible_refs, ex.amazon_refs, ex.references),
        additional_instruction="Confirm the audiobook runtime on Audible/Amazon audiobook pages."
    )

    # Narrator info verifiable
    doc_narrators_node = evaluator.add_leaf(
        id="document_narrators",
        desc="Audiobook narrator information is publicly available and verifiable",
        parent=docs_node,
        critical=True
    )
    narrators_str = join_names(ex.narrators)
    doc_narrators_claim = f"The audiobook narrator(s) are: {narrators_str}."
    await evaluator.verify(
        claim=doc_narrators_claim,
        node=doc_narrators_node,
        sources=concat_sources(ex.audible_refs, ex.amazon_refs, ex.references),
        additional_instruction=(
            "Verify the narrator names listed on the Audible page or Amazon audiobook page. "
            "Allow minor naming variants or formatting differences."
        )
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
    Evaluate an answer for the celebrity memoir task (2024-2025).
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract memoir info from the agent's answer
    ex: MemoirExtraction = await evaluator.extract(
        prompt=prompt_extract_memoir(),
        template_class=MemoirExtraction,
        extraction_name="memoir_info"
    )

    # Add the root critical node as per rubric
    main_node = evaluator.add_sequential(
        id="memoir_identification_and_verification",
        desc="Identify and verify a celebrity memoir published in 2024-2025 that meets all specified criteria",
        parent=root,
        critical=True
    )

    # Subtree: basic publication criteria (critical)
    await verify_basic_publication_criteria(evaluator, main_node, ex)

    # Subtree: format specifications (critical)
    await verify_format_specifications(evaluator, main_node, ex)

    # Optional: record a small custom info block with counts of URLs extracted
    url_stats = {
        "publisher_refs_count": len(ex.publisher_refs),
        "amazon_refs_count": len(ex.amazon_refs),
        "audible_refs_count": len(ex.audible_refs),
        "goodreads_refs_count": len(ex.goodreads_refs),
        "library_refs_count": len(ex.library_refs),
        "other_refs_count": len(ex.other_refs),
        "total_references_count": len(ex.references),
    }
    evaluator.add_custom_info(url_stats, info_type="url_statistics")

    # Return evaluation summary
    return evaluator.get_summary()