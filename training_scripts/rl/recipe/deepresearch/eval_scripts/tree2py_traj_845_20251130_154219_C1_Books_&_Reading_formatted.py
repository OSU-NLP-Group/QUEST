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
TASK_ID = "book_elsewhere_2024_eval"
TASK_DESCRIPTION = "Who is the co-author of Keanu Reeves' 2024 novel 'The Book of Elsewhere', and which publisher released it?"

EXPECTED_TITLE = "The Book of Elsewhere"
EXPECTED_AUTHOR = "Keanu Reeves"
EXPECTED_YEAR = "2024"
EXPECTED_COAUTHOR = "China Miéville"
EXPECTED_PUBLISHER = "Del Rey"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BookAnswerExtraction(BaseModel):
    """Structured extraction of what the answer claims about the target book."""
    referenced_title: Optional[str] = None
    referenced_primary_author: Optional[str] = None
    referenced_year: Optional[str] = None
    coauthor: Optional[str] = None
    publisher: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_book_answer_info() -> str:
    return """
    From the answer, extract what the answer claims about the target book. Extract only what is explicitly stated.

    Return a JSON object with:
    - referenced_title: The book title the answer refers to (e.g., "The Book of Elsewhere"). If multiple titles are mentioned, pick the primary one relevant to the question.
    - referenced_primary_author: The primary author named by the answer (e.g., "Keanu Reeves"), if stated.
    - referenced_year: The publication year the answer associates with the referenced book (e.g., "2024"), if stated.
    - coauthor: The co-author named in the answer for the referenced book, if any.
    - publisher: The publisher/imprint named in the answer for the referenced book, if any (e.g., "Del Rey" or "Del Rey Books").
    - source_urls: A list of all URLs that the answer includes (if any). Extract valid URLs only; include markdown links' URL targets.

    Notes:
    - Do not infer or invent anything not present in the answer.
    - If a field is not clearly stated, set it to null (or empty list for source_urls).
    - Keep names exactly as written in the answer (case can vary).
    """


# --------------------------------------------------------------------------- #
# Verification subroutine                                                     #
# --------------------------------------------------------------------------- #
async def verify_book_information_accuracy(
    evaluator: Evaluator,
    parent_node,
    extracted: BookAnswerExtraction
) -> None:
    """
    Build the BookInformationAccuracy subtree and verify each critical leaf.
    """
    # Parent critical node: BookInformationAccuracy (parallel aggregation)
    accuracy_node = evaluator.add_parallel(
        id="BookInformationAccuracy",
        desc="Verifies the answer provides correct co-author and publisher information for the specified 2024 Keanu Reeves novel 'The Book of Elsewhere'.",
        parent=parent_node,
        critical=True
    )

    # Leaf 1: Target book is correctly referenced (critical)
    n_target = evaluator.add_leaf(
        id="CorrectTargetBook",
        desc="The answer clearly refers to the target work: 'The Book of Elsewhere' by Keanu Reeves, published in 2024 (i.e., does not confuse it with another title/year/author).",
        parent=accuracy_node,
        critical=True
    )

    target_claim = (
        "In the provided answer, the referenced work is the 2024 novel 'The Book of Elsewhere' by Keanu Reeves, "
        "and it is not confused with a different title, author, or year."
    )
    target_instruction = (
        "Judge based only on the answer content. Pass if the answer clearly mentions 'The Book of Elsewhere' and "
        "indicates Keanu Reeves as the author (explicitly or by clear context). Regarding the year: "
        "if the answer explicitly states a different year than 2024, fail; if the year is omitted but nothing contradicts 2024, pass. "
        "Fail if the answer focuses on a different book, different author, or contradictory year."
    )
    await evaluator.verify(
        claim=target_claim,
        node=n_target,
        additional_instruction=target_instruction
    )

    # Leaf 2: Co-author is correctly identified (critical)
    n_coauthor = evaluator.add_leaf(
        id="CoauthorCorrect",
        desc="The co-author is correctly identified as China Miéville.",
        parent=accuracy_node,
        critical=True
    )

    coauthor_claim = (
        "In the answer, the named co-author of 'The Book of Elsewhere' is China Miéville."
    )
    coauthor_instruction = (
        "Judge strictly based on the answer text. Accept minor spelling/casing variations (e.g., 'China Mieville' without accent) "
        "or inclusion of middle name(s). Fail if a different co-author is named or if no co-author is stated."
    )
    await evaluator.verify(
        claim=coauthor_claim,
        node=n_coauthor,
        additional_instruction=coauthor_instruction
    )

    # Leaf 3: Publisher is correctly identified (critical)
    n_publisher = evaluator.add_leaf(
        id="PublisherCorrect",
        desc="The publisher is correctly identified as Del Rey.",
        parent=accuracy_node,
        critical=True
    )

    publisher_claim = (
        "In the answer, the named publisher (imprint) of 'The Book of Elsewhere' is Del Rey (also acceptable: 'Del Rey Books')."
    )
    publisher_instruction = (
        "Judge based only on the answer content. Accept 'Del Rey' or 'Del Rey Books'. "
        "If the answer only states 'Random House' or 'Penguin Random House' without mentioning 'Del Rey', consider it incorrect. "
        "Fail if a different publisher is stated or if no publisher is stated."
    )
    await evaluator.verify(
        claim=publisher_claim,
        node=n_publisher,
        additional_instruction=publisher_instruction
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
    Evaluate an answer for the 'Book of Elsewhere' co-author and publisher question.
    """
    # 1) Initialize evaluator
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

    # 2) Extract structured info from the answer (for record-keeping and potential debugging)
    extracted = await evaluator.extract(
        prompt=prompt_extract_book_answer_info(),
        template_class=BookAnswerExtraction,
        extraction_name="book_answer_info"
    )

    # 3) Add ground truth information for transparency
    evaluator.add_ground_truth({
        "expected_title": EXPECTED_TITLE,
        "expected_primary_author": EXPECTED_AUTHOR,
        "expected_year": EXPECTED_YEAR,
        "expected_coauthor": EXPECTED_COAUTHOR,
        "expected_publisher": EXPECTED_PUBLISHER
    }, gt_type="ground_truth")

    # 4) Build verification subtree and run checks
    await verify_book_information_accuracy(evaluator, root, extracted)

    # 5) Return final structured summary
    return evaluator.get_summary()