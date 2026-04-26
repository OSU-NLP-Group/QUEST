import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nba_fiction_2024_reimagining_jim"
TASK_DESCRIPTION = "Which book won the 2024 National Book Award for Fiction and is a reimagining of Mark Twain's 'Adventures of Huckleberry Finn' told from the perspective of the enslaved character Jim?"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BookCandidate(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BookExtraction(BaseModel):
    books: List[BookCandidate] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_book() -> str:
    return """
    Extract up to one book the answer identifies as satisfying BOTH of the following:
    1) It won the 2024 National Book Award for Fiction.
    2) It is a reimagining or retelling of Mark Twain's "Adventures of Huckleberry Finn" from Jim's perspective.

    For the selected book, extract:
    - title: the book title as stated in the answer
    - author: the author name if provided
    - sources: a list of all URLs explicitly cited in the answer that support either the award claim or the reimagining-from-Jim's-perspective claim. Include any official award pages, publisher pages, reputable news coverage, or reviews cited.

    Rules:
    - Do not invent information. Only extract what's present in the answer.
    - If multiple books are mentioned, return only the first one that the answer claims satisfies both conditions (otherwise return the first mentioned).
    - If any field is missing, set it to null or an empty list accordingly.
    """


# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #
def get_primary_book(extraction: BookExtraction) -> BookCandidate:
    if extraction.books:
        return extraction.books[0]
    return BookCandidate()


# --------------------------------------------------------------------------- #
# Main evaluation logic                                                       #
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
    Evaluate whether the answer correctly identifies the book that:
    - won the 2024 National Book Award for Fiction, and
    - is a reimagining of 'Adventures of Huckleberry Finn' told from Jim's perspective.
    """
    # Initialize evaluator
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

    # Extract candidate book from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_book(),
        template_class=BookExtraction,
        extraction_name="book_extraction"
    )

    # Use the first (primary) book if any
    book = get_primary_book(extraction)

    # Record simple summary info for debugging
    evaluator.add_custom_info(
        info={
            "extracted_title": book.title,
            "author": book.author,
            "num_sources": len(book.sources),
            "sources": book.sources
        },
        info_type="extraction_summary"
    )

    # Build verification tree according to the rubric
    book_node = evaluator.add_parallel(
        id="Book_Identification",
        desc="Correctly identifies the book that won the 2024 National Book Award for Fiction and is a reimagining of a Mark Twain classic novel",
        parent=root,
        critical=False
    )

    # Leaf 1: NBA Fiction Winner 2024
    nba_leaf = evaluator.add_leaf(
        id="NBA_Fiction_Winner_2024",
        desc="The identified book won the 2024 National Book Award for Fiction",
        parent=book_node,
        critical=True
    )

    nba_claim_title = book.title or ""
    nba_claim = f"The book '{nba_claim_title}' won the 2024 National Book Award for Fiction."
    nba_additional_instruction = (
        "Use the provided URLs as evidence. Accept reasonable phrasing variants like 'Fiction winner at the 2024 National Book Awards'. "
        "The evidence must unambiguously indicate the book won (not just nominated or longlisted) the 2024 National Book Award in the Fiction category. "
        "If the provided URLs are missing, invalid, or do not support the claim clearly, mark the claim as Incorrect."
    )

    # Leaf 2: Reimagining of Huck Finn from Jim's perspective
    huck_leaf = evaluator.add_leaf(
        id="Huckleberry_Finn_Reimagining",
        desc="The identified book is a reimagining of 'Adventures of Huckleberry Finn' narrated from Jim's perspective",
        parent=book_node,
        critical=True
    )

    huck_claim_title = book.title or ""
    huck_claim = (
        f"The book '{huck_claim_title}' is a reimagining of Mark Twain's 'Adventures of Huckleberry Finn' told from Jim's perspective."
    )
    huck_additional_instruction = (
        "Use the provided URLs as evidence. Accept equivalent phrasings such as 'retelling', 'reimagining', 'told from Jim's point of view', "
        "'first-person narrative from Jim', or 'the story centers on Jim's perspective'. "
        "The evidence should clearly attribute this narrative perspective to the book. "
        "If the provided URLs are missing, invalid, or do not support the claim clearly, mark the claim as Incorrect."
    )

    # Perform verifications (in parallel if desired)
    claims_and_sources = [
        (nba_claim, book.sources, nba_leaf, nba_additional_instruction),
        (huck_claim, book.sources, huck_leaf, huck_additional_instruction),
    ]
    await evaluator.batch_verify(claims_and_sources)

    return evaluator.get_summary()