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
TASK_ID = "literary_awards_2025_4_major"
TASK_DESCRIPTION = (
    "Identify four major English-language fiction literary awards that announced their 2025 winners between January and December 2025. "
    "For each award, provide: (1) The official name of the award; (2) The winning author's full name; "
    "(3) The complete title of the winning book; (4) The publisher of the winning book; "
    "(5) The ceremony date or official announcement date (Month Day, Year); "
    "(6) The ceremony venue or location (at minimum the city; include venue if available). "
    "All four awards must be recognized, established international literary prizes for fiction. "
    "The information provided must be verifiable through reliable sources."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AwardItem(BaseModel):
    award_name: Optional[str] = None
    author_name: Optional[str] = None
    book_title: Optional[str] = None
    publisher: Optional[str] = None
    date: Optional[str] = None  # e.g., "October 14, 2025"
    location: Optional[str] = None  # e.g., "London" or "New York City, Lincoln Center"
    source_urls: List[str] = Field(default_factory=list)


class AwardsExtraction(BaseModel):
    awards: List[AwardItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_awards() -> str:
    return """
    Extract up to 6 awards mentioned in the answer that relate to 2025 winners of major English-language fiction literary awards.
    For each award, return an object with:
    - award_name: The official name of the award (string)
    - author_name: The full name of the winning author (string)
    - book_title: The complete title of the winning book, including subtitle if provided (string)
    - publisher: The publisher or imprint of the winning book (string)
    - date: The ceremony or official announcement date exactly as stated in the answer (string; e.g., "October 14, 2025")
    - location: The ceremony or announcement location; at least the city, include venue if present (string)
    - source_urls: An array of all URLs in the answer that specifically support this award’s 2025 winner and/or the required details (author, title, publisher, date, location). 
      Extract actual URLs only. Include all that apply for this award.
    Important:
    - Do not invent information. Extract exactly what is present in the answer text.
    - If a field is missing, set it to null (or an empty list for source_urls).
    - If the answer lists more than 4 awards, still extract them all (up to 6). The evaluator will consider only the first 4.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s if s is not None else ""


def _ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(n, f"Award {n}")


# --------------------------------------------------------------------------- #
# Verification for one award                                                  #
# --------------------------------------------------------------------------- #
async def verify_one_award(
    evaluator: Evaluator,
    parent_node,
    award: AwardItem,
    index_zero_based: int,
) -> None:
    """
    Build the verification subtree for one award and run URL-grounded checks.
    The structure mirrors the provided rubric tree:
      - Award_n (parallel, non-critical)
        - Identification (parallel, critical)
          - Award Name (leaf, critical)
        - Winner Information (parallel, critical)
          - Author Name (leaf, critical)
          - Book Title (leaf, critical)
        - Publisher Information (parallel, critical)
          - Publisher Name (leaf, critical)
        - Ceremony Details (parallel, critical)
          - Date (leaf, critical)
          - Location (leaf, critical)
        - Reference URL (existence custom node, critical)
    """
    award_number = index_zero_based + 1
    award_title_for_desc = _ordinal(award_number)

    # Top-level award node (non-critical to allow partial credit across awards)
    award_node = evaluator.add_parallel(
        id=f"award_{award_number}",
        desc=f"{award_title_for_desc} major English-language fiction award with 2025 winner announced",
        parent=parent_node,
        critical=False
    )

    # Reference URL existence (Critical). Placed early so other leaves can auto-depend on it.
    has_urls = bool(award.source_urls)
    evaluator.add_custom_node(
        result=has_urls,
        id=f"award_{award_number}_reference_url",
        desc=f"Provide a reliable URL source confirming this award's 2025 winner and all required details",
        parent=award_node,
        critical=True
    )

    # Identification group (Critical)
    ident_node = evaluator.add_parallel(
        id=f"award_{award_number}_identification",
        desc="Award identification information",
        parent=award_node,
        critical=True
    )

    # Winner Information group (Critical)
    winner_node = evaluator.add_parallel(
        id=f"award_{award_number}_winner_info",
        desc="Information about the winning author and book",
        parent=award_node,
        critical=True
    )

    # Publisher Information group (Critical)
    publisher_node = evaluator.add_parallel(
        id=f"award_{award_number}_publisher_info",
        desc="Publishing details of the winning book",
        parent=award_node,
        critical=True
    )

    # Ceremony Details group (Critical)
    ceremony_node = evaluator.add_parallel(
        id=f"award_{award_number}_ceremony_details",
        desc="Details about the award ceremony or announcement",
        parent=award_node,
        critical=True
    )

    # Award Name leaf
    award_name_leaf = evaluator.add_leaf(
        id=f"award_{award_number}_name",
        desc="The official name of the award is provided",
        parent=ident_node,
        critical=True
    )
    claim_award_name = (
        f"The sources confirm that the 2025 winner announcement is for the award named '{_safe(award.award_name)}'."
    )
    await evaluator.verify(
        claim=claim_award_name,
        node=award_name_leaf,
        sources=award.source_urls,
        additional_instruction=(
            "Verify that at least one provided source clearly names the award. Allow minor naming variants or "
            "branding (e.g., 'The Booker Prize' vs 'Booker Prize'). The claim should be supported by the source."
        )
    )

    # Author Name leaf
    author_leaf = evaluator.add_leaf(
        id=f"award_{award_number}_author_name",
        desc="The full name of the winning author is provided",
        parent=winner_node,
        critical=True
    )
    claim_author = (
        f"The sources confirm that the 2025 winner of '{_safe(award.award_name)}' is author '{_safe(award.author_name)}'."
    )
    await evaluator.verify(
        claim=claim_author,
        node=author_leaf,
        sources=award.source_urls,
        additional_instruction=(
            "Confirm that the 2025 winner is the specified author. Allow minor variations in name formatting "
            "(e.g., middle initials, diacritics). If the award credits multiple authors, the provided author "
            "must be included among the winners."
        )
    )

    # Book Title leaf
    book_leaf = evaluator.add_leaf(
        id=f"award_{award_number}_book_title",
        desc="The complete title of the winning book is provided",
        parent=winner_node,
        critical=True
    )
    claim_book = (
        f"The sources confirm that the winning book title for the 2025 '{_safe(award.award_name)}' is "
        f"'{_safe(award.book_title)}'."
    )
    await evaluator.verify(
        claim=claim_book,
        node=book_leaf,
        sources=award.source_urls,
        additional_instruction=(
            "Verify the book title as presented on the source. Allow minor punctuation or styling differences. "
            "If the title includes a subtitle, consider it a match if the core title and subtitle align."
        )
    )

    # Publisher Name leaf
    publisher_leaf = evaluator.add_leaf(
        id=f"award_{award_number}_publisher_name",
        desc="The name of the publisher is provided",
        parent=publisher_node,
        critical=True
    )
    claim_publisher = (
        f"The sources confirm that the publisher of the winning book '{_safe(award.book_title)}' is "
        f"'{_safe(award.publisher)}'."
    )
    await evaluator.verify(
        claim=claim_publisher,
        node=publisher_leaf,
        sources=award.source_urls,
        additional_instruction=(
            "Check if the sources (award announcement, publisher page, or credible media) state the publisher. "
            "Accept imprint vs parent publisher equivalence where clear (e.g., 'Knopf' vs 'Alfred A. Knopf')."
        )
    )

    # Date leaf
    date_leaf = evaluator.add_leaf(
        id=f"award_{award_number}_date",
        desc="The ceremony date or announcement date is provided",
        parent=ceremony_node,
        critical=True
    )
    claim_date = (
        f"The sources confirm that the 2025 '{_safe(award.award_name)}' winner was officially announced on "
        f"'{_safe(award.date)}'."
    )
    await evaluator.verify(
        claim=claim_date,
        node=date_leaf,
        sources=award.source_urls,
        additional_instruction=(
            "Verify the specific date of the winner announcement or ceremony. The date must fall within the year 2025. "
            "Allow formatting variations (e.g., '5 January 2025' vs 'January 5, 2025') when determining correctness."
        )
    )

    # Location leaf
    location_leaf = evaluator.add_leaf(
        id=f"award_{award_number}_location",
        desc="The ceremony venue or location (at minimum, the city) is provided",
        parent=ceremony_node,
        critical=True
    )
    claim_location = (
        f"The sources confirm that the ceremony or official announcement location for the 2025 "
        f"'{_safe(award.award_name)}' was '{_safe(award.location)}'."
    )
    await evaluator.verify(
        claim=claim_location,
        node=location_leaf,
        sources=award.source_urls,
        additional_instruction=(
            "Verify at least the city name. If a specific venue is included in the provided location, check that as well. "
            "Accept 'City, Country' vs 'City' as a match if the city aligns."
        )
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
    Entry point for evaluating an answer for the 'Four Major Fiction Literary Awards (2025 winners)' task.
    """
    # Initialize evaluator (root is non-critical; allow partial credit aggregation across awards)
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

    # Extract awards information
    extracted = await evaluator.extract(
        prompt=prompt_extract_awards(),
        template_class=AwardsExtraction,
        extraction_name="awards_extraction",
    )

    # Keep only the first 4 awards; pad with empty awards if fewer than 4
    awards = list(extracted.awards[:4])
    while len(awards) < 4:
        awards.append(AwardItem())

    # Add a logical container node (critical=False to permit partial credit)
    awards_root = evaluator.add_parallel(
        id="four_major_fiction_literary_awards_2025",
        desc="Identify four major English-language fiction literary awards that announced their 2025 winners, providing complete information and verifiable sources for each",
        parent=root,
        critical=False
    )

    # Build verification subtrees for each of the four awards
    for idx in range(4):
        await verify_one_award(evaluator, awards_root, awards[idx], idx)

    # Return evaluation summary
    return evaluator.get_summary()