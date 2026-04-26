import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pulitzer_2024_fiction"
TASK_DESCRIPTION = "Identify the book that won the 2024 Pulitzer Prize for Fiction. Provide the author's name, the publisher and publication date of the hardcover edition, and the ISBN-13 of the hardcover edition."


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BookExtraction(BaseModel):
    """Structured information extracted from the agent's answer."""
    book_title: Optional[str] = None
    author_name: Optional[str] = None
    hardcover_publisher: Optional[str] = None
    hardcover_publication_date: Optional[str] = None
    hardcover_isbn_13: Optional[str] = None
    award_sources: List[str] = Field(default_factory=list)
    book_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_book_info() -> str:
    return """
    Extract the following information from the answer exactly as stated. If something is missing, return null for that field.

    Required fields:
    - book_title: The title of the book identified as the official winner of the 2024 Pulitzer Prize for Fiction.
    - author_name: The author of that winning book, as stated in the answer.
    - hardcover_publisher: The publisher of the hardcover edition of the winning book.
    - hardcover_publication_date: The publication date of the hardcover edition (keep the format used in the answer; do not reformat).
    - hardcover_isbn_13: The ISBN-13 of the hardcover edition (include any hyphens or spaces exactly as presented in the answer).

    Also extract source URLs:
    - award_sources: All URLs specifically cited in the answer to support the identification of the official 2024 Pulitzer Prize for Fiction winner. Include the official Pulitzer website page if present. Include credible news or organization pages that explicitly state the winner. Extract actual URLs even if presented in markdown.
    - book_sources: All URLs that provide bibliographic details for the book’s hardcover edition (e.g., publisher’s book page, major retailer book pages). Do not include unrelated URLs.

    Return a single JSON object with exactly these fields:
    {
      "book_title": ...,
      "author_name": ...,
      "hardcover_publisher": ...,
      "hardcover_publication_date": ...,
      "hardcover_isbn_13": ...,
      "award_sources": [...],
      "book_sources": [...]
    }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _sources_for_award(extracted: BookExtraction) -> List[str]:
    """Prefer award_sources; if empty, fall back to book_sources."""
    primary = extracted.award_sources or []
    fallback = extracted.book_sources or []
    return _unique_urls(primary if primary else fallback)


def _sources_for_book_details(extracted: BookExtraction) -> List[str]:
    """Prefer book_sources; if empty, fall back to award_sources."""
    primary = extracted.book_sources or []
    fallback = extracted.award_sources or []
    return _unique_urls(primary if primary else fallback)


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: BookExtraction) -> None:
    """
    Build the verification tree according to the rubric and dispatch verifications.
    """
    # Top-level category node (critical, parallel)
    book_info_node = evaluator.add_parallel(
        id="Book_Information",
        desc="Provide accurate information about the official 2024 Pulitzer Prize for Fiction winning book and its hardcover edition details.",
        parent=evaluator.root,
        critical=True
    )

    # Leaf: Winning Book Identification (critical)
    win_leaf = evaluator.add_leaf(
        id="Winning_Book_Identification",
        desc="Correctly identify the book that is the official 2024 Pulitzer Prize for Fiction winner (i.e., the answer’s book matches the official winner).",
        parent=book_info_node,
        critical=True
    )
    win_claim = f"The book titled '{extracted.book_title}' is the official 2024 Pulitzer Prize for Fiction winner."
    await evaluator.verify(
        claim=win_claim,
        node=win_leaf,
        sources=_sources_for_award(extracted),
        additional_instruction=(
            "Rely on the official Pulitzer website if available. Otherwise, accept credible sources that explicitly state "
            "the book won the 2024 Pulitzer Prize for Fiction. Focus only on the 2024 Fiction category."
        ),
    )

    # Leaf: Author (critical)
    author_leaf = evaluator.add_leaf(
        id="Author",
        desc="Correctly identify the author of the official winning book.",
        parent=book_info_node,
        critical=True
    )
    author_claim = f"The author of the book '{extracted.book_title}' is '{extracted.author_name}'."
    await evaluator.verify(
        claim=author_claim,
        node=author_leaf,
        sources=_unique_urls((_sources_for_award(extracted) or []) + (_sources_for_book_details(extracted) or [])),
        additional_instruction=(
            "Allow minor naming variants (middle names/initials, casing). Verify the author associated with the identified winning title."
        ),
    )

    # Node: Hardcover Edition Details (critical, parallel)
    hc_node = evaluator.add_parallel(
        id="Hardcover_Edition_Details",
        desc="Provide correct hardcover-edition bibliographic details for the winning book.",
        parent=book_info_node,
        critical=True
    )

    # Leaf: Hardcover Publisher (critical)
    hc_pub_leaf = evaluator.add_leaf(
        id="Hardcover_Publisher",
        desc="Correctly identify the publisher of the hardcover edition.",
        parent=hc_node,
        critical=True
    )
    hc_pub_claim = f"The hardcover edition of '{extracted.book_title}' is published by '{extracted.hardcover_publisher}'."
    await evaluator.verify(
        claim=hc_pub_claim,
        node=hc_pub_leaf,
        sources=_sources_for_book_details(extracted),
        additional_instruction=(
            "Prefer publisher pages or reputable retailer listings. The claim should match the hardcover edition’s publisher specifically."
        ),
    )

    # Leaf: Hardcover Publication Date (critical)
    hc_date_leaf = evaluator.add_leaf(
        id="Hardcover_Publication_Date",
        desc="Correctly identify the publication date of the hardcover edition.",
        parent=hc_node,
        critical=True
    )
    hc_date_claim = f"The hardcover edition of '{extracted.book_title}' was published on '{extracted.hardcover_publication_date}'."
    await evaluator.verify(
        claim=hc_date_claim,
        node=hc_date_leaf,
        sources=_sources_for_book_details(extracted),
        additional_instruction=(
            "Accept reasonable date format variations (e.g., 'April 2, 2024' vs '2024-04-02'). Verify the date corresponds to the hardcover edition."
        ),
    )

    # Leaf: Hardcover ISBN-13 (critical)
    hc_isbn_leaf = evaluator.add_leaf(
        id="Hardcover_ISBN_13",
        desc="Provide the correct ISBN-13 for the hardcover edition.",
        parent=hc_node,
        critical=True
    )
    hc_isbn_claim = f"The ISBN-13 of the hardcover edition of '{extracted.book_title}' is '{extracted.hardcover_isbn_13}'."
    await evaluator.verify(
        claim=hc_isbn_claim,
        node=hc_isbn_leaf,
        sources=_sources_for_book_details(extracted),
        additional_instruction=(
            "Treat hyphens/spaces as formatting only. Consider the numeric equivalence of 13 digits after stripping hyphens."
        ),
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
    Evaluate an answer for the 2024 Pulitzer Prize for Fiction identification and hardcover details task.
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_book_info(),
        template_class=BookExtraction,
        extraction_name="book_extraction",
    )

    await build_verification_tree(evaluator, extracted)

    return evaluator.get_summary()