import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "booker_prize_2024_winner_info"
TASK_DESCRIPTION = (
    "What fiction book won the 2024 Booker Prize? Provide the following information: "
    "the book's title, the author's full name, the exact page count, the publisher's name, "
    "the publication year, and a reference URL from a reliable source confirming these details."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BookerWinnerInfo(BaseModel):
    book_title: Optional[str] = None
    author_full_name: Optional[str] = None
    page_count: Optional[str] = None  # keep as string to allow formats like "304 pages"
    publisher_name: Optional[str] = None
    publication_year: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_booker_winner_info() -> str:
    return (
        "Extract the bibliographic details for the 2024 Booker Prize winning fiction book from the provided answer. "
        "Return a JSON object with the following fields:\n"
        "1) book_title: the exact title of the winning book (string)\n"
        "2) author_full_name: the author's full name exactly as provided (string)\n"
        "3) page_count: the exact page count mentioned in the answer (string; keep any units like 'pages' if present)\n"
        "4) publisher_name: the publisher's name (string)\n"
        "5) publication_year: the publication year mentioned (string)\n"
        "6) reference_urls: an array of all URLs explicitly cited in the answer that support these details; "
        "include URLs in any format (plain links, markdown links). If none are provided, return an empty array.\n"
        "If any field is not present in the answer, set it to null (or empty array for reference_urls). "
        "Do not infer or invent details not found in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s.strip() if isinstance(s, str) else ""


# --------------------------------------------------------------------------- #
# Main verification logic                                                     #
# --------------------------------------------------------------------------- #
async def _build_and_verify(
    evaluator: Evaluator,
    root: Any,
    info: BookerWinnerInfo,
) -> None:
    """
    Build the verification tree under a critical parallel node and run verifications.
    """
    # Critical parent node representing the rubric's main item
    main_node = evaluator.add_parallel(
        id="booker_prize_2024_main",
        desc="Provide complete information about the fiction book that won the 2024 Booker Prize, including a reliable reference URL that confirms the listed bibliographic details.",
        parent=root,
        critical=True
    )

    # Critical existence check for sources to enforce URL-grounded verification
    sources_exist_node = evaluator.add_custom_node(
        result=bool(info.reference_urls),
        id="sources_provided",
        desc="At least one reference URL is provided in the answer",
        parent=main_node,
        critical=True
    )

    # 1) Winner identification
    winner_leaf = evaluator.add_leaf(
        id="is_2024_booker_prize_winner",
        desc="The answer identifies a book that won the 2024 Booker Prize.",
        parent=main_node,
        critical=True
    )
    winner_claim = (
        f"The book '{_safe(info.book_title)}' by {_safe(info.author_full_name)} won the 2024 Booker Prize."
    )
    await evaluator.verify(
        claim=winner_claim,
        node=winner_leaf,
        sources=info.reference_urls,
        additional_instruction=(
            "Verify that the page explicitly states the book as the winner of the Booker Prize in 2024. "
            "Accept reasonable wording variants like 'Booker Prize 2024 winner', 'won the 2024 Booker', etc."
        ),
    )

    # 2) Fiction check
    fiction_leaf = evaluator.add_leaf(
        id="is_fiction",
        desc="The identified winning book is fiction.",
        parent=main_node,
        critical=True
    )
    fiction_claim = (
        f"The book '{_safe(info.book_title)}' is a work of fiction (a novel)."
    )
    await evaluator.verify(
        claim=fiction_claim,
        node=fiction_leaf,
        sources=info.reference_urls,
        additional_instruction=(
            "Confirm the work is fiction. Accept synonyms such as 'novel', 'fiction', 'literary fiction'. "
            "If it is a poetry collection, non-fiction, short-story anthology (not a novel), or otherwise not fiction, mark incorrect."
        ),
    )

    # 3) Title verification
    title_leaf = evaluator.add_leaf(
        id="book_title",
        desc="The correct title of the 2024 Booker Prize winning book is provided.",
        parent=main_node,
        critical=True
    )
    title_claim = (
        f"The title of the 2024 Booker Prize winning book is '{_safe(info.book_title)}'."
    )
    await evaluator.verify(
        claim=title_claim,
        node=title_leaf,
        sources=info.reference_urls,
        additional_instruction=(
            "Check that the page clearly shows the book's title and that it matches the provided title, "
            "allowing minor casing or punctuation differences."
        ),
    )

    # 4) Author full name verification
    author_leaf = evaluator.add_leaf(
        id="author_full_name",
        desc="The author's full name of the 2024 Booker Prize winning book is provided correctly.",
        parent=main_node,
        critical=True
    )
    author_claim = (
        f"The author of the 2024 Booker Prize winning book is '{_safe(info.author_full_name)}'."
    )
    await evaluator.verify(
        claim=author_claim,
        node=author_leaf,
        sources=info.reference_urls,
        additional_instruction=(
            "Confirm the author's full name as shown on the source page. "
            "Allow minor variants such as middle initials or accents."
        ),
    )

    # 5) Exact page count verification
    pages_leaf = evaluator.add_leaf(
        id="exact_page_count",
        desc="The exact page count of the book is provided accurately.",
        parent=main_node,
        critical=True
    )
    pages_claim = (
        f"The book '{_safe(info.book_title)}' has exactly {_safe(info.page_count)}."
    )
    await evaluator.verify(
        claim=pages_claim,
        node=pages_leaf,
        sources=info.reference_urls,
        additional_instruction=(
            "Verify the page count for the standard edition referenced by the source page (e.g., hardcover or primary trade edition). "
            "If the page count is missing or only approximate, mark incorrect."
        ),
    )

    # 6) Publisher name verification
    publisher_leaf = evaluator.add_leaf(
        id="publisher_name",
        desc="The publisher's name is provided correctly.",
        parent=main_node,
        critical=True
    )
    publisher_claim = (
        f"The publisher of '{_safe(info.book_title)}' is '{_safe(info.publisher_name)}'."
    )
    await evaluator.verify(
        claim=publisher_claim,
        node=publisher_leaf,
        sources=info.reference_urls,
        additional_instruction=(
            "Confirm the publisher name for the edition associated with the details. "
            "Accept imprints (e.g., 'Vintage', 'Knopf', 'Faber & Faber') where the imprint functions as the publisher as shown on the source page."
        ),
    )

    # 7) Publication year must be 2024 (enforce both provided and value)
    pub_year_leaf = evaluator.add_leaf(
        id="publication_year_2024",
        desc="The publication year is provided and is 2024.",
        parent=main_node,
        critical=True
    )
    pub_year_claim = (
        f"The book '{_safe(info.book_title)}' was published in {_safe(info.publication_year)}."
    )
    await evaluator.verify(
        claim=pub_year_claim,
        node=pub_year_leaf,
        sources=info.reference_urls,
        additional_instruction=(
            "First, check whether the source page gives a publication year. "
            "If the extracted year from the answer is not '2024', return Incorrect. "
            "Otherwise, verify that the page shows publication year 2024."
        ),
    )

    # 8) Reliable reference URL confirming all details
    reference_leaf = evaluator.add_leaf(
        id="reference_url_reliable_and_confirming_details",
        desc="A reference URL from a reliable source is provided that confirms the listed book details (title, author, page count, publisher, publication year).",
        parent=main_node,
        critical=True
    )
    reference_claim = (
        f"At least one of these sources is a reliable bibliographic source and explicitly confirms all of the following for the same edition of '{_safe(info.book_title)}': "
        f"title '{_safe(info.book_title)}', author '{_safe(info.author_full_name)}', page count '{_safe(info.page_count)}', "
        f"publisher '{_safe(info.publisher_name)}', and publication year '2024'."
    )
    await evaluator.verify(
        claim=reference_claim,
        node=reference_leaf,
        sources=info.reference_urls,
        additional_instruction=(
            "Judge reliability and completeness: Prefer official Booker Prize website, the publisher's official page, "
            "major reputable news outlets (e.g., BBC, The Guardian), or authoritative library/catalog records (e.g., WorldCat, British Library). "
            "Personal blogs or low-quality sites are not reliable. "
            "Pass only if at least one URL explicitly confirms all listed details in one place."
        ),
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
    Evaluate an answer for the Booker Prize 2024 winner information task.
    """
    # Initialize evaluator with a non-critical root (framework default),
    # then we add a critical node under it to represent the rubric's root.
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

    # Extract structured information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_booker_winner_info(),
        template_class=BookerWinnerInfo,
        extraction_name="booker_winner_info",
    )

    # Build verification tree and run checks
    await _build_and_verify(evaluator, root, extracted_info)

    # Return structured summary
    return evaluator.get_summary()