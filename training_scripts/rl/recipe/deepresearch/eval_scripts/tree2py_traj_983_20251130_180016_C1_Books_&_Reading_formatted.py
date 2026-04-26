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
TASK_ID = "nba_2025_fiction"
TASK_DESCRIPTION = (
    "Identify the book that won the 2025 National Book Award for Fiction. Provide the complete title of the book, "
    "the publisher of the hardcover edition, and the official release date (including month, day, and year)."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NBABookWinnerExtraction(BaseModel):
    """Structured information extracted from the agent's answer for the 2025 NBA Fiction winner."""
    book_title: Optional[str] = None
    hardcover_publisher: Optional[str] = None
    release_date: Optional[str] = None
    winner_sources: List[str] = Field(default_factory=list)
    publisher_sources: List[str] = Field(default_factory=list)
    release_date_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_winner_info() -> str:
    return """
    Extract the key information about the 2025 National Book Award for Fiction winner from the provided answer.

    Return a JSON object with the following fields:
    1. book_title: The complete title of the book that won the 2025 National Book Award for Fiction. Include any subtitle if present.
    2. hardcover_publisher: The publisher or imprint of the hardcover edition of the book (e.g., "Alfred A. Knopf", "Riverhead Books").
    3. release_date: The official publication/release date of the hardcover edition, including month, day, and year (e.g., "November 12, 2025"). If the answer only provides month and year, still extract exactly as written.
    4. winner_sources: An array of all URLs in the answer that specifically support the claim that this book won the 2025 National Book Award for Fiction. Prefer official or reputable sources (e.g., National Book Foundation announcement, major news coverage), but extract all URLs mentioned.
    5. publisher_sources: An array of URLs in the answer that specifically support the publisher/imprint of the hardcover edition (e.g., the publisher's official book page, ISBN metadata pages).
    6. release_date_sources: An array of URLs in the answer that specifically support the official release/publication date of the hardcover edition.

    SPECIAL RULES:
    - Extract only URLs explicitly present in the answer; do not invent or infer URLs.
    - Accept URLs in plain form or markdown links; extract the underlying URL string.
    - If a field is missing from the answer, set it to null (for strings) or an empty array (for URL lists).

    Be precise and faithful to the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(*source_lists: List[str]) -> List[str]:
    """Combine multiple lists of URLs while preserving order and removing duplicates."""
    seen = set()
    combined: List[str] = []
    for lst in source_lists:
        for url in lst:
            if url and url not in seen:
                seen.add(url)
                combined.append(url)
    return combined


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def verify_winner_information(
    evaluator: Evaluator,
    parent_node,
    extracted: NBABookWinnerExtraction,
) -> None:
    """
    Build and execute the verification tree for the 2025 National Book Award Fiction winner information.
    """
    # Critical parallel node as specified by rubric JSON
    info_node = evaluator.add_parallel(
        id="2025_NBA_Fiction_Winner_Information",
        desc="Information about the 2025 National Book Award Fiction winner",
        parent=parent_node,
        critical=True,
    )

    # ---------------------- Book Title Verification ---------------------- #
    title_node = evaluator.add_leaf(
        id="Book_Title",
        desc="The complete title of the book that won the 2025 National Book Award for Fiction",
        parent=info_node,
        critical=True,
    )

    title_str = extracted.book_title or ""
    # Prefer winner_sources; if empty, allow other in-answer sources that might also mention the win
    title_sources = extracted.winner_sources or combine_sources(
        extracted.publisher_sources, extracted.release_date_sources
    )

    title_claim = f"The book that won the 2025 National Book Award for Fiction is titled '{title_str}'."

    await evaluator.verify(
        claim=title_claim,
        node=title_node,
        sources=title_sources if title_sources else None,
        additional_instruction=(
            "Only mark as supported if the webpage explicitly states that this book is the WINNER of the "
            "2025 National Book Award in the Fiction category (e.g., 'Winner' or 'Fiction winner'). "
            "Allow minor punctuation or subtitle variations in the title."
        ),
    )

    # ---------------------- Publisher Verification ----------------------- #
    publisher_node = evaluator.add_leaf(
        id="Publisher",
        desc="The publisher of the hardcover edition",
        parent=info_node,
        critical=True,
    )

    pub_str = extracted.hardcover_publisher or ""
    # Prefer publisher_sources; otherwise allow other sources from the answer
    publisher_sources = extracted.publisher_sources or combine_sources(
        extracted.release_date_sources, extracted.winner_sources
    )

    if extracted.book_title:
        publisher_claim = f"The hardcover edition of '{extracted.book_title}' is published by '{pub_str}'."
    else:
        publisher_claim = f"The hardcover edition's publisher is '{pub_str}'."

    await evaluator.verify(
        claim=publisher_claim,
        node=publisher_node,
        sources=publisher_sources if publisher_sources else None,
        additional_instruction=(
            "Confirm that the named organization/imprint is the publisher of the hardcover edition. "
            "Prefer publisher/imprint official pages or authoritative bibliographic listings (e.g., ISBN metadata). "
            "Allow equivalent imprint naming (e.g., 'Alfred A. Knopf' vs 'Knopf', 'Riverhead Books' vs 'Penguin Random House')."
        ),
    )

    # ---------------------- Release Date Verification -------------------- #
    release_node = evaluator.add_leaf(
        id="Release_Date",
        desc="The official publication date including month, day, and year",
        parent=info_node,
        critical=True,
    )

    date_str = extracted.release_date or ""
    # Prefer release_date_sources; otherwise allow other supporting sources from the answer
    release_sources = extracted.release_date_sources or combine_sources(
        extracted.publisher_sources, extracted.winner_sources
    )

    if extracted.book_title:
        release_claim = f"The hardcover edition of '{extracted.book_title}' was officially released on {date_str}."
    else:
        release_claim = f"The hardcover edition was officially released on {date_str}."

    await evaluator.verify(
        claim=release_claim,
        node=release_node,
        sources=release_sources if release_sources else None,
        additional_instruction=(
            "Verify the official publication/release date for the hardcover edition. "
            "The stated date must include month, day, and year. "
            "Accept minor formatting differences (e.g., 'Jan 1, 2025' vs 'January 1, 2025'). "
            "If the page only lists month and year without the day, do NOT mark supported for a full M/D/Y claim."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an agent's answer for the 2025 NBA Fiction winner task and return a structured result dictionary.
    """
    # Initialize evaluator (root node is non-critical, parallel aggregation)
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
        prompt=prompt_extract_winner_info(),
        template_class=NBABookWinnerExtraction,
        extraction_name="winner_info_extraction",
    )

    # Build verification tree and perform checks
    await verify_winner_information(evaluator, root, extracted_info)

    # Return final summary with tree and scores
    return evaluator.get_summary()